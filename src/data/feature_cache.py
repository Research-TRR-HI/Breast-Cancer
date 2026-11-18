"""Augmentation-Aware Sharded Feature Cache with Integrity & Resume Recovery.

Design rationale (Part 6 of AnchorShift audit):
    Because TRUST uses stochastic training augmentation (random rotation, scaling,
    translation, brightness, contrast) before the frozen Mammo-CLIP encoder, one
    fixed embedding per original DICOM does NOT reproduce online training across
    20 epochs.

    - Evaluation splits (`train_calib`, `val_calib`) are deterministic (unaugmented)
      and therefore cached once per image in FP32 (`eval_<split>_fp32.pt`).
    - Training occurrences (`train`) are cached per `(epoch, batch_idx, slot_idx)`
      in atomic FP32 shards (`train_epoch_EE_shard_SSS.pt`) alongside exact sample
      indices, image IDs, and augmentation parameters.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch

from src.data.preprocessing import AugmentationParams
from src.data.rsna_dataset import BatchDrawRecord


CACHE_SCHEMA_VERSION = "anchorshift_aug_cache_v1"


class CacheIntegrityError(RuntimeError):
    """Raised when cached feature shards or metadata fail verification."""


def compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _atomic_write_torch(path: Path, payload: Dict[str, Any]) -> str:
    """Atomically write a torch payload and return its SHA-256 checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(payload, str(tmp_path))
    with open(tmp_path, "rb") as f:
        f.flush()
    sha256 = compute_file_sha256(tmp_path)
    os.replace(tmp_path, path)
    return sha256


def _cleanup_stale_temp_files(cache_root: Path) -> List[str]:
    removed = []
    if not cache_root.exists():
        return removed
    for p in cache_root.rglob("*.tmp"):
        p.unlink(missing_ok=True)
        removed.append(str(p))
    return removed


def shard_filename(epoch: int, shard_idx: int) -> str:
    return f"train_epoch_{epoch:02d}_shard_{shard_idx:03d}.pt"


def eval_cache_filename(split_name: str, dtype_tag: str = "fp32") -> str:
    return f"eval_{split_name}_{dtype_tag}.pt"


def save_eval_split_cache(
    cache_root: Path,
    split_name: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    image_ids: Sequence[str],
    patient_ids: Sequence[str],
) -> Dict[str, Any]:
    """Atomically store deterministic unaugmented evaluation split features in FP32."""
    fname = eval_cache_filename(split_name, dtype_tag="fp32")
    fpath = cache_root / fname
    feat_fp32 = features.detach().cpu().to(torch.float32).contiguous()
    lbl_fp32 = labels.detach().cpu().to(torch.float32).contiguous()
    payload = {
        "split": split_name,
        "dtype": "float32",
        "features": feat_fp32,
        "labels": lbl_fp32,
        "image_ids": [str(x) for x in image_ids],
        "patient_ids": [str(x) for x in patient_ids],
    }
    sha = _atomic_write_torch(fpath, payload)
    return {
        "file": fname,
        "sha256": sha,
        "num_samples": int(feat_fp32.shape[0]),
        "embed_dim": int(feat_fp32.shape[1]),
        "dtype": "float32",
        "size_bytes": int(fpath.stat().st_size),
    }


def save_train_shard(
    cache_root: Path,
    epoch: int,
    shard_idx: int,
    batch_start: int,
    batch_end: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_indices: torch.Tensor,
    image_ids: List[List[str]],
    aug_params: List[List[Dict[str, float]]],
) -> Dict[str, Any]:
    """Atomically store one occurrence-indexed FP32 training feature shard."""
    fname = shard_filename(epoch, shard_idx)
    fpath = cache_root / fname
    feat_fp32 = features.detach().cpu().to(torch.float32).contiguous()
    lbl_fp32 = labels.detach().cpu().to(torch.float32).contiguous()
    idx_i64 = sample_indices.detach().cpu().to(torch.long).contiguous()

    payload = {
        "epoch": int(epoch),
        "shard_idx": int(shard_idx),
        "batch_start": int(batch_start),
        "batch_end": int(batch_end),
        "dtype": "float32",
        "features": feat_fp32,          # (N_batches, B, embed_dim)
        "labels": lbl_fp32,             # (N_batches, B)
        "sample_indices": idx_i64,      # (N_batches, B)
        "image_ids": image_ids,         # List[List[str]]
        "aug_params": aug_params,       # List[List[dict]]
    }
    sha = _atomic_write_torch(fpath, payload)
    return {
        "epoch": int(epoch),
        "shard_idx": int(shard_idx),
        "batch_start": int(batch_start),
        "batch_end": int(batch_end),
        "num_batches": int(feat_fp32.shape[0]),
        "batch_size": int(feat_fp32.shape[1]),
        "embed_dim": int(feat_fp32.shape[2]),
        "file": fname,
        "sha256": sha,
        "size_bytes": int(fpath.stat().st_size),
    }


def build_cache_manifest(
    model_checkpoint_sha256: str,
    preprocess_config_hash: str,
    split_manifest_sha256: str,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    n_pos: int,
    epoch_policy: str,
    embed_dim: int,
    eval_splits: Dict[str, Dict[str, Any]],
    train_shards: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dtype": "float32",
        "model_checkpoint_sha256": str(model_checkpoint_sha256),
        "preprocess_config_hash": str(preprocess_config_hash),
        "split_manifest_sha256": str(split_manifest_sha256),
        "seed": int(seed),
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "batch_size": int(batch_size),
        "n_pos": int(n_pos),
        "epoch_policy": str(epoch_policy),
        "embed_dim": int(embed_dim),
        "eval_splits": eval_splits,
        "train_shards": sorted(train_shards, key=lambda s: (s["epoch"], s["shard_idx"])),
    }


def write_cache_manifest(cache_root: Path, manifest: Dict[str, Any]) -> Path:
    mpath = cache_root / "cache_manifest.json"
    _atomic_write_json(mpath, manifest)
    return mpath


def load_cache_manifest(cache_root: Path) -> Dict[str, Any]:
    mpath = cache_root / "cache_manifest.json"
    if not mpath.exists():
        raise CacheIntegrityError(f"Missing cache manifest at {mpath}")
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_cache_integrity(
    cache_root: Path,
    expected_model_sha256: str,
    expected_preprocess_hash: str,
    expected_split_sha256: Optional[str] = None,
    expected_epochs: Optional[int] = None,
    expected_steps_per_epoch: Optional[int] = None,
    expected_schedules: Optional[Dict[int, List[BatchDrawRecord]]] = None,
    verify_shard_checksums: bool = True,
) -> Dict[str, Any]:
    """Validate cache metadata, shard SHA-256 checksums, completeness, and duplicate/missing draws."""
    manifest = load_cache_manifest(cache_root)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheIntegrityError(
            f"Schema mismatch: {manifest.get('schema_version')} != {CACHE_SCHEMA_VERSION}"
        )
    if manifest.get("model_checkpoint_sha256") != expected_model_sha256:
        raise CacheIntegrityError(
            f"Model checkpoint hash mismatch: {manifest.get('model_checkpoint_sha256')} != {expected_model_sha256}"
        )
    if manifest.get("preprocess_config_hash") != expected_preprocess_hash:
        raise CacheIntegrityError(
            f"Preprocessing hash mismatch: {manifest.get('preprocess_config_hash')} != {expected_preprocess_hash}"
        )
    if expected_split_sha256 is not None and manifest.get("split_manifest_sha256") != expected_split_sha256:
        raise CacheIntegrityError(
            f"Split manifest hash mismatch: {manifest.get('split_manifest_sha256')} != {expected_split_sha256}"
        )

    epochs = int(manifest["epochs"]) if expected_epochs is None else int(expected_epochs)
    steps_per_epoch = (
        int(manifest["steps_per_epoch"])
        if expected_steps_per_epoch is None
        else int(expected_steps_per_epoch)
    )

    # Verify eval split caches
    for split_name, meta in manifest.get("eval_splits", {}).items():
        fpath = cache_root / meta["file"]
        if not fpath.exists():
            raise CacheIntegrityError(f"Missing eval split cache file: {fpath}")
        if verify_shard_checksums:
            actual_sha = compute_file_sha256(fpath)
            if actual_sha != meta["sha256"]:
                raise CacheIntegrityError(
                    f"Eval cache checksum mismatch for {fpath}: {actual_sha} != {meta['sha256']}"
                )

    # Verify train shards for missing/duplicate (epoch, batch_idx) draws
    seen_batches: Dict[Tuple[int, int], str] = {}
    for shard_meta in manifest.get("train_shards", []):
        ep = int(shard_meta["epoch"])
        fpath = cache_root / shard_meta["file"]
        if not fpath.exists():
            raise CacheIntegrityError(f"Missing train shard file: {fpath}")
        if verify_shard_checksums:
            actual_sha = compute_file_sha256(fpath)
            if actual_sha != shard_meta["sha256"]:
                raise CacheIntegrityError(
                    f"Train shard checksum mismatch for {fpath}: {actual_sha} != {shard_meta['sha256']}"
                )

        shard_data = torch.load(str(fpath), map_location="cpu", weights_only=False)
        b_start = int(shard_data["batch_start"])
        b_end = int(shard_data["batch_end"])
        n_batches = int(shard_data["features"].shape[0])
        if (b_end - b_start) != n_batches:
            raise CacheIntegrityError(
                f"Shard {fpath} batch range [{b_start}, {b_end}) != tensor batch count {n_batches}"
            )

        for local_b, global_b in enumerate(range(b_start, b_end)):
            key = (ep, global_b)
            if key in seen_batches:
                raise CacheIntegrityError(
                    f"Duplicate draw detected for epoch={ep}, batch={global_b} in {fpath} and {seen_batches[key]}"
                )
            seen_batches[key] = shard_meta["file"]

            if expected_schedules is not None and ep in expected_schedules:
                rec = expected_schedules[ep][global_b]
                cached_idx = shard_data["sample_indices"][local_b].tolist()
                if cached_idx != rec.indices:
                    raise CacheIntegrityError(
                        f"Sample index mismatch at epoch={ep}, batch={global_b}"
                    )
                cached_aug = shard_data["aug_params"][local_b]
                expected_aug = [a.to_dict() for a in rec.aug_params]
                if cached_aug != expected_aug:
                    raise CacheIntegrityError(
                        f"Augmentation param mismatch at epoch={ep}, batch={global_b}"
                    )

    for ep in range(1, epochs + 1):
        for b in range(steps_per_epoch):
            if (ep, b) not in seen_batches:
                raise CacheIntegrityError(f"Missing cached draw for epoch={ep}, batch={b}")

    return manifest


def extract_or_resume_cache(
    cache_root: Path,
    schedules_by_epoch: Dict[int, List[BatchDrawRecord]],
    extract_batch_fn: Callable[[BatchDrawRecord], Tuple[torch.Tensor, torch.Tensor, List[str]]],
    eval_extract_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]],
    model_checkpoint_sha256: str,
    preprocess_config_hash: str,
    split_manifest_sha256: str,
    seed: int,
    batch_size: int,
    n_pos: int,
    epoch_policy: str,
    embed_dim: int = 2048,
    batches_per_shard: int = 10,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Extract sharded augmentation-aware cache with automatic checkpoint/resume recovery.

    Valid existing shards (matching manifest checksums and hashes) are preserved;
    any temporary or corrupted shards are discarded and re-extracted cleanly.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_temp_files(cache_root)

    epochs = len(schedules_by_epoch)
    steps_per_epoch = len(next(iter(schedules_by_epoch.values())))

    existing_manifest = None
    manifest_path = cache_root / "cache_manifest.json"
    if manifest_path.exists():
        try:
            cand = load_cache_manifest(cache_root)
            if (
                cand.get("schema_version") == CACHE_SCHEMA_VERSION
                and cand.get("model_checkpoint_sha256") == model_checkpoint_sha256
                and cand.get("preprocess_config_hash") == preprocess_config_hash
                and cand.get("split_manifest_sha256") == split_manifest_sha256
                and int(cand.get("seed", -1)) == int(seed)
                and int(cand.get("epochs", -1)) == int(epochs)
                and int(cand.get("steps_per_epoch", -1)) == int(steps_per_epoch)
            ):
                existing_manifest = cand
        except Exception:
            existing_manifest = None

    eval_splits: Dict[str, Dict[str, Any]] = {}
    if existing_manifest is not None:
        for sname, smeta in existing_manifest.get("eval_splits", {}).items():
            fpath = cache_root / smeta["file"]
            if fpath.exists() and compute_file_sha256(fpath) == smeta["sha256"]:
                eval_splits[sname] = smeta

    for split_name in ("train_calib", "val_calib"):
        if split_name not in eval_splits:
            if on_progress:
                on_progress(f"Extracting deterministic {split_name} features...")
            e_feats, e_labels, e_img_ids, e_pt_ids = eval_extract_fn(split_name)
            eval_splits[split_name] = save_eval_split_cache(
                cache_root, split_name, e_feats, e_labels, e_img_ids, e_pt_ids
            )

    valid_shards_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if existing_manifest is not None:
        for smeta in existing_manifest.get("train_shards", []):
            fpath = cache_root / smeta["file"]
            if fpath.exists() and compute_file_sha256(fpath) == smeta["sha256"]:
                valid_shards_by_key[(int(smeta["epoch"]), int(smeta["shard_idx"]))] = smeta

    train_shards: List[Dict[str, Any]] = []
    for epoch in sorted(schedules_by_epoch.keys()):
        epoch_recs = schedules_by_epoch[epoch]
        for shard_idx, b_start in enumerate(range(0, len(epoch_recs), batches_per_shard)):
            b_end = min(b_start + batches_per_shard, len(epoch_recs))
            key = (int(epoch), int(shard_idx))
            if key in valid_shards_by_key:
                train_shards.append(valid_shards_by_key[key])
                continue

            shard_feats = []
            shard_labels = []
            shard_indices = []
            shard_img_ids = []
            shard_augs = []
            for rec in epoch_recs[b_start:b_end]:
                b_feats, b_labels, b_img_ids = extract_batch_fn(rec)
                shard_feats.append(b_feats.detach().cpu().to(torch.float32))
                shard_labels.append(b_labels.detach().cpu().to(torch.float32))
                shard_indices.append(torch.tensor(rec.indices, dtype=torch.long))
                shard_img_ids.append(b_img_ids)
                shard_augs.append([a.to_dict() for a in rec.aug_params])

            meta = save_train_shard(
                cache_root=cache_root,
                epoch=epoch,
                shard_idx=shard_idx,
                batch_start=b_start,
                batch_end=b_end,
                features=torch.stack(shard_feats, dim=0),
                labels=torch.stack(shard_labels, dim=0),
                sample_indices=torch.stack(shard_indices, dim=0),
                image_ids=shard_img_ids,
                aug_params=shard_augs,
            )
            train_shards.append(meta)
            manifest = build_cache_manifest(
                model_checkpoint_sha256=model_checkpoint_sha256,
                preprocess_config_hash=preprocess_config_hash,
                split_manifest_sha256=split_manifest_sha256,
                seed=seed,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                batch_size=batch_size,
                n_pos=n_pos,
                epoch_policy=epoch_policy,
                embed_dim=embed_dim,
                eval_splits=eval_splits,
                train_shards=train_shards,
            )
            write_cache_manifest(cache_root, manifest)
            if on_progress:
                on_progress(f"Saved epoch {epoch} shard {shard_idx} ({b_end}/{len(epoch_recs)} batches)")

    final_manifest = build_cache_manifest(
        model_checkpoint_sha256=model_checkpoint_sha256,
        preprocess_config_hash=preprocess_config_hash,
        split_manifest_sha256=split_manifest_sha256,
        seed=seed,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        n_pos=n_pos,
        epoch_policy=epoch_policy,
        embed_dim=embed_dim,
        eval_splits=eval_splits,
        train_shards=train_shards,
    )
    write_cache_manifest(cache_root, final_manifest)
    return final_manifest


def load_epoch_batches_from_cache(
    cache_root: Path,
    manifest: Dict[str, Any],
    epoch: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load all cached training batches for a given epoch in deterministic batch order."""
    epoch_shards = sorted(
        [s for s in manifest["train_shards"] if int(s["epoch"]) == int(epoch)],
        key=lambda s: int(s["batch_start"]),
    )
    batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for smeta in epoch_shards:
        data = torch.load(str(cache_root / smeta["file"]), map_location="cpu", weights_only=False)
        feats = data["features"]
        labels = data["labels"]
        indices = data["sample_indices"]
        for i in range(feats.shape[0]):
            batches.append((feats[i], labels[i], indices[i]))
    return batches
