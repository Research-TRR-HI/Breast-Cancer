"""Augmentation-Aware Sharded Feature Cache with Strict Validation & Recovery (Critical Fix H)."""

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch

from src.data.rsna_dataset import BatchDrawRecord


CACHE_SCHEMA_VERSION = "anchorshift_aug_cache_v2"


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


def get_cache_manifest_sha256(cache_root: Path) -> str:
    """Return SHA-256 of cache_manifest.json for strict checkpoint linkage."""
    mpath = Path(cache_root) / "cache_manifest.json"
    if not mpath.exists():
        raise CacheIntegrityError(f"Missing cache manifest at {mpath}")
    return compute_file_sha256(mpath)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _atomic_write_torch(path: Path, payload: Dict[str, Any]) -> str:
    """Atomically write a torch payload with flush + fsync on writable descriptor before replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pt.tmp")
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
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
    if not torch.isfinite(feat_fp32).all():
        raise CacheIntegrityError(f"Non-finite features detected while saving {split_name} cache")
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
    if not torch.isfinite(feat_fp32).all():
        raise CacheIntegrityError(f"Non-finite features detected while saving {fname}")

    payload = {
        "epoch": int(epoch),
        "shard_idx": int(shard_idx),
        "batch_start": int(batch_start),
        "batch_end": int(batch_end),
        "dtype": "float32",
        "features": feat_fp32,
        "labels": lbl_fp32,
        "sample_indices": idx_i64,
        "image_ids": image_ids,
        "aug_params": aug_params,
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
    stage1_config_hash: str,
    split_manifest_sha256: str,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    n_pos: int,
    epoch_policy: str,
    embed_dim: int,
    encoder_precision: str,
    eval_splits: Dict[str, Dict[str, Any]],
    train_shards: List[Dict[str, Any]],
    environment_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cache_identity = compute_bytes_sha256(
        "|".join(
            [
                CACHE_SCHEMA_VERSION,
                str(model_checkpoint_sha256),
                str(preprocess_config_hash),
                str(stage1_config_hash),
                str(split_manifest_sha256),
                str(seed),
                str(epochs),
                str(steps_per_epoch),
                str(batch_size),
                str(n_pos),
                str(epoch_policy),
                str(embed_dim),
                str(encoder_precision),
            ]
        ).encode("utf-8")
    )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_identity": cache_identity,
        "dtype": "float32",
        "encoder_precision": str(encoder_precision),
        "model_checkpoint_sha256": str(model_checkpoint_sha256),
        "preprocess_config_hash": str(preprocess_config_hash),
        "stage1_config_hash": str(stage1_config_hash),
        "split_manifest_sha256": str(split_manifest_sha256),
        "seed": int(seed),
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "batch_size": int(batch_size),
        "n_pos": int(n_pos),
        "epoch_policy": str(epoch_policy),
        "embed_dim": int(embed_dim),
        "environment_metadata": environment_metadata or {},
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
    expected_stage1_hash: Optional[str] = None,
    expected_split_sha256: Optional[str] = None,
    expected_epochs: Optional[int] = None,
    expected_steps_per_epoch: Optional[int] = None,
    expected_batch_size: Optional[int] = None,
    expected_n_pos: Optional[int] = None,
    expected_epoch_policy: Optional[str] = None,
    expected_embed_dim: Optional[int] = 2048,
    expected_encoder_precision: Optional[str] = None,
    expected_seed: Optional[int] = None,
    expected_schedules: Optional[Dict[int, List[BatchDrawRecord]]] = None,
    expected_eval_meta: Optional[Dict[str, Tuple[List[str], List[str], List[int]]]] = None,
    verify_shard_checksums: bool = True,
) -> Dict[str, Any]:
    """Validate every cache metadata field, tensor shape/dtype/finiteness, checksum, and draw."""
    manifest = load_cache_manifest(cache_root)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheIntegrityError(
            f"Schema mismatch: {manifest.get('schema_version')} != {CACHE_SCHEMA_VERSION}"
        )
    if manifest.get("dtype") != "float32":
        raise CacheIntegrityError(f"Stored cache dtype mismatch: {manifest.get('dtype')} != float32")

    checks = [
        ("model_checkpoint_sha256", expected_model_sha256),
        ("preprocess_config_hash", expected_preprocess_hash),
        ("stage1_config_hash", expected_stage1_hash),
        ("split_manifest_sha256", expected_split_sha256),
        ("epochs", expected_epochs),
        ("steps_per_epoch", expected_steps_per_epoch),
        ("batch_size", expected_batch_size),
        ("n_pos", expected_n_pos),
        ("epoch_policy", expected_epoch_policy),
        ("embed_dim", expected_embed_dim),
        ("encoder_precision", expected_encoder_precision),
        ("seed", expected_seed),
    ]
    for key, exp_val in checks:
        if exp_val is not None and manifest.get(key) != exp_val:
            raise CacheIntegrityError(f"Cache {key} mismatch: {manifest.get(key)!r} != {exp_val!r}")

    epochs = int(manifest["epochs"])
    steps_per_epoch = int(manifest["steps_per_epoch"])
    batch_size = int(manifest["batch_size"])
    embed_dim = int(manifest["embed_dim"])

    # Verify eval splits
    eval_splits = manifest.get("eval_splits", {})
    for req_split in ("train_calib", "val_calib"):
        if req_split not in eval_splits:
            raise CacheIntegrityError(f"Missing required eval split {req_split!r} in cache manifest")
        meta = eval_splits[req_split]
        fpath = cache_root / meta["file"]
        if not fpath.exists():
            raise CacheIntegrityError(f"Missing eval split cache file: {fpath}")
        if verify_shard_checksums and compute_file_sha256(fpath) != meta["sha256"]:
            raise CacheIntegrityError(f"Eval cache checksum mismatch for {fpath}")

        edata = torch.load(str(fpath), map_location="cpu", weights_only=False)
        efeats = edata["features"]
        elbls = edata["labels"]
        if efeats.dtype != torch.float32 or efeats.dim() != 2 or efeats.shape[1] != embed_dim:
            raise CacheIntegrityError(f"Invalid eval feature tensor shape/dtype in {fpath}: {efeats.shape} {efeats.dtype}")
        if not torch.isfinite(efeats).all():
            raise CacheIntegrityError(f"Non-finite values in eval features {fpath}")
        if len(elbls) != efeats.shape[0] or len(edata["image_ids"]) != efeats.shape[0] or len(edata["patient_ids"]) != efeats.shape[0]:
            raise CacheIntegrityError(f"Metadata length mismatch in eval cache {fpath}")
        if expected_eval_meta is not None and req_split in expected_eval_meta:
            exp_img_ids, exp_pt_ids, exp_lbls = expected_eval_meta[req_split]
            if edata["image_ids"] != exp_img_ids or edata["patient_ids"] != exp_pt_ids:
                raise CacheIntegrityError(f"Eval ID order mismatch for {req_split}")
            if [int(v) for v in elbls.tolist()] != [int(v) for v in exp_lbls]:
                raise CacheIntegrityError(f"Eval label mismatch for {req_split}")

    # Verify train shards & draw completeness
    seen_batches: Dict[Tuple[int, int], str] = {}
    for shard_meta in manifest.get("train_shards", []):
        ep = int(shard_meta["epoch"])
        fpath = cache_root / shard_meta["file"]
        if not fpath.exists():
            raise CacheIntegrityError(f"Missing train shard file: {fpath}")
        if verify_shard_checksums and compute_file_sha256(fpath) != shard_meta["sha256"]:
            raise CacheIntegrityError(f"Train shard checksum mismatch for {fpath}")

        shard_data = torch.load(str(fpath), map_location="cpu", weights_only=False)
        b_start = int(shard_data["batch_start"])
        b_end = int(shard_data["batch_end"])
        feats = shard_data["features"]
        lbls = shard_data["labels"]
        idxs = shard_data["sample_indices"]
        img_ids = shard_data["image_ids"]
        augs = shard_data["aug_params"]

        expected_nb = b_end - b_start
        if feats.dtype != torch.float32 or tuple(feats.shape) != (expected_nb, batch_size, embed_dim):
            raise CacheIntegrityError(
                f"Invalid train shard shape/dtype in {fpath}: got {tuple(feats.shape)} {feats.dtype}, "
                f"expected {(expected_nb, batch_size, embed_dim)} float32"
            )
        if not torch.isfinite(feats).all():
            raise CacheIntegrityError(f"Non-finite features in train shard {fpath}")
        if tuple(lbls.shape) != (expected_nb, batch_size) or tuple(idxs.shape) != (expected_nb, batch_size):
            raise CacheIntegrityError(f"Label/index shape mismatch in train shard {fpath}")
        if not torch.all((lbls == 0.0) | (lbls == 1.0)):
            raise CacheIntegrityError(f"Non-binary labels in train shard {fpath}")

        for local_b, global_b in enumerate(range(b_start, b_end)):
            key = (ep, global_b)
            if key in seen_batches:
                raise CacheIntegrityError(
                    f"Duplicate draw detected for epoch={ep}, batch={global_b} in {fpath} and {seen_batches[key]}"
                )
            seen_batches[key] = shard_meta["file"]

            if expected_schedules is not None and ep in expected_schedules:
                rec = expected_schedules[ep][global_b]
                if idxs[local_b].tolist() != rec.indices:
                    raise CacheIntegrityError(f"Sample index mismatch at epoch={ep}, batch={global_b}")
                if augs[local_b] != [a.to_dict() for a in rec.aug_params]:
                    raise CacheIntegrityError(f"Augmentation param mismatch at epoch={ep}, batch={global_b}")
                if len(img_ids[local_b]) != batch_size:
                    raise CacheIntegrityError(f"Image ID length mismatch at epoch={ep}, batch={global_b}")

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
    stage1_config_hash: str,
    split_manifest_sha256: str,
    seed: int,
    batch_size: int,
    n_pos: int,
    epoch_policy: str,
    embed_dim: int = 2048,
    encoder_precision: str = "amp",
    batches_per_shard: int = 10,
    environment_metadata: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Extract sharded augmentation-aware cache with automatic checkpoint/resume recovery."""
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
                and cand.get("stage1_config_hash") == stage1_config_hash
                and cand.get("split_manifest_sha256") == split_manifest_sha256
                and cand.get("encoder_precision") == encoder_precision
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
    total_batches_all = sum(len(schedules_by_epoch[e]) for e in schedules_by_epoch)
    done_batches_all = 0
    newly_done_batches = 0
    t_extract_start = time.perf_counter()

    for epoch in sorted(schedules_by_epoch.keys()):
        epoch_recs = schedules_by_epoch[epoch]
        for shard_idx, b_start in enumerate(range(0, len(epoch_recs), batches_per_shard)):
            b_end = min(b_start + batches_per_shard, len(epoch_recs))
            key = (int(epoch), int(shard_idx))
            if key in valid_shards_by_key:
                train_shards.append(valid_shards_by_key[key])
                done_batches_all += (b_end - b_start)
                continue

            shard_feats, shard_labels, shard_indices, shard_img_ids, shard_augs = [], [], [], [], []
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
                stage1_config_hash=stage1_config_hash,
                split_manifest_sha256=split_manifest_sha256,
                seed=seed,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                batch_size=batch_size,
                n_pos=n_pos,
                epoch_policy=epoch_policy,
                embed_dim=embed_dim,
                encoder_precision=encoder_precision,
                eval_splits=eval_splits,
                train_shards=train_shards,
                environment_metadata=environment_metadata,
            )
            write_cache_manifest(cache_root, manifest)
            nb_done = b_end - b_start
            done_batches_all += nb_done
            newly_done_batches += nb_done
            if on_progress:
                elapsed_s = max(time.perf_counter() - t_extract_start, 1e-6)
                done_imgs = done_batches_all * batch_size
                total_imgs = total_batches_all * batch_size
                rate = (newly_done_batches * batch_size) / elapsed_s
                eta_hr = ((total_imgs - done_imgs) / max(rate, 1e-6)) / 3600.0 if rate > 0 else 0.0
                pct = (100.0 * done_batches_all) / max(total_batches_all, 1)
                filled = int(round(20 * done_batches_all / max(total_batches_all, 1)))
                bar = "█" * filled + "░" * (20 - filled)
                on_progress(
                    f"🔥 [{bar}] {done_imgs:,}/{total_imgs:,} ({pct:5.1f}%) | ⚡ {rate:4.1f} img/s | "
                    f"⏳ ETA {eta_hr:4.2f}h | Ep {epoch:02d}/{epochs} (batch {b_end}/{len(epoch_recs)})"
                )

    final_manifest = build_cache_manifest(
        model_checkpoint_sha256=model_checkpoint_sha256,
        preprocess_config_hash=preprocess_config_hash,
        stage1_config_hash=stage1_config_hash,
        split_manifest_sha256=split_manifest_sha256,
        seed=seed,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        n_pos=n_pos,
        epoch_policy=epoch_policy,
        embed_dim=embed_dim,
        encoder_precision=encoder_precision,
        eval_splits=eval_splits,
        train_shards=train_shards,
        environment_metadata=environment_metadata,
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
