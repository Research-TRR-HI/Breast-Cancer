"""Corrected TRUST & CE-Only Training Pipeline on Modal.

Reference:
    TRUST (Shafie et al., 2026, arXiv:2609.00300v1): Sections 3.2 - 3.10
    - Backbone: Mammo-CLIP EfficientNet-B5 (frozen, strict 852/852 parity, pinned SHA-256).
    - Head: LayerNorm(2048) -> Dropout(0.3) -> Linear(2048, 1) via isolated deterministic init.
    - Stage-1 Cache: Deterministic lossless uint16 cropped images (`/data/cache/stage1_rsna_uint16`)
      so raw DICOMs are decoded once instead of 1,000,000 times during training/extraction.
    - Minibatch: 80 (20 positives, 60 negatives), independent minibatch draws per TRUST Sec 3.9
      (ceil(37,444 / 60) = 625 batches/epoch -> 1,000,000 occurrences over 20 epochs).
    - Loss: L_CE + 1.0 * L_focal (gamma=2.0, no alpha) + 0.05 * L_dismiss (margin=0.10).
    - Dynamic tau_t: Initialized at 0.05; updated at beginning of each epoch >= 2 on image-level
      train_calib scores using 95% Clopper-Pearson upper bound <= 1% (0.0 if none satisfies).
    - Model Selection: Fixed 20 epochs, NO early stopping or val_calib checkpoint selection.
      Final 20th-epoch checkpoint evaluated once on val_calib for 95% and 98% source thresholds.

Usage (DO NOT EXECUTE AUTOMATICALLY - for user execution):
    modal run scripts/train_trust_modal.py --benchmark-extract-batches 5 --gpu L40S
    modal run scripts/train_trust_modal.py --extract-cache --gpu L40S
    modal run scripts/train_trust_modal.py --verify-cache
    modal run scripts/train_trust_modal.py --baseline ce --epochs 20 --gpu L40S --use-cached-features
    modal run scripts/train_trust_modal.py --baseline trust --epochs 20 --gpu L40S --use-cached-features
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import signal
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "anchorshift-stage-b-train"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
MOUNT_PATH = Path("/data")
COMMIT_INTERVAL_SEC = 60
SUPPORTED_GPUS = ("T4", "L4", "A10G", "L40S", "A100", "H100")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES)
    .add_local_python_source("src")
)


def _start_periodic_commit(vol: modal.Volume, stop_event: threading.Event) -> threading.Thread:
    def _loop() -> None:
        while not stop_event.wait(COMMIT_INTERVAL_SEC):
            try:
                vol.commit()
                print("[modal] Periodic volume.commit() completed.", flush=True)
            except Exception as exc:
                print(f"[modal] Periodic commit warning: {exc}", file=sys.stderr, flush=True)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def _register_signal_handlers(vol: modal.Volume, stop_event: threading.Event) -> None:
    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"\n[signal] Caught signal {signum}. Committing volume before exit...", flush=True)
        stop_event.set()
        try:
            vol.commit()
        except Exception:
            pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def _build_frozen_encoder(mammo_clip_tar: Path, device):
    from src.models.mammo_clip_loader import MAMMO_CLIP_SHA256, load_and_verify_mammo_clip
    from src.models.trust_model import MammoCLIPEncoder

    if not mammo_clip_tar.exists():
        raise FileNotFoundError(f"Mammo-CLIP checkpoint missing at {mammo_clip_tar}")
    encoder = MammoCLIPEncoder(checkpoint_path=None, freeze=False, model_name="efficientnet-b5")
    diag = load_and_verify_mammo_clip(encoder, mammo_clip_tar, strict=True, verify_sha256=True)
    assert (
        diag["is_fully_compatible"]
        and diag["matched_keys_count"] == 852
        and diag["total_target_keys"] == 852
        and diag["missing_keys_count"] == 0
        and diag["unexpected_keys_count"] == 0
    ), f"Expected strict 852/852 Mammo-CLIP key parity, got: {diag}"
    encoder.freeze_encoder()
    return encoder.to(device), MAMMO_CLIP_SHA256, diag


def _load_verified_stage1_loader(
    df,
    stage1_root: Path,
    split_sha256: str,
    stage1_hash: str,
    strict_segmentation: bool = True,
):
    from src.data.stage1_cache import Stage1CacheError, Stage1CroppedLoader, verify_stage1_cache

    if not (stage1_root / "stage1_manifest.json").exists():
        raise Stage1CacheError(
            f"Missing Stage-1 uint16 cache at {stage1_root}. "
            "Run `modal run scripts/cache_stage1_rsna_modal.py --build` first (or pass --require-stage1-cache False)."
        )
    s1_manifest = verify_stage1_cache(
        cache_root=stage1_root,
        expected_df=df,
        expected_stage1_hash=stage1_hash,
        expected_split_sha256=split_sha256,
        allow_fallbacks=not strict_segmentation,
        verify_pixel_hashes=False,
    )
    return Stage1CroppedLoader(stage1_root, s1_manifest, max_cached_shards=16), s1_manifest


def _prepare_stage2_batch(
    image_ids: List[str],
    rel_paths: List[str],
    lateralities: List[str],
    aug_params_list: Optional[List[Any]],
    rsna_raw: Path,
    stage1_loader=None,
    strict_segmentation: bool = True,
    num_workers: int = 4,
):
    import torch
    from src.data.preprocessing import prepare_tensor_from_cropped, preprocess_dicom

    augs = aug_params_list if aug_params_list is not None else [None] * len(image_ids)

    if stage1_loader is not None:
        # Read uint16 crops from shard LRU sequentially (shard LRU already caches shards in RAM),
        # then run Stage-2 resize/aug across worker threads or inline cleanly.
        crops = [stage1_loader.load_uint16(img_id) for img_id in image_ids]

        def _s2(item):
            crop, aug = item
            return prepare_tensor_from_cropped(
                crop,
                target_size=(1520, 912),
                augment_params=aug,
                standardize=True,
            )

        items = list(zip(crops, augs))
        if num_workers <= 1 or len(items) <= 1:
            return torch.stack([_s2(it) for it in items])
        with ThreadPoolExecutor(max_workers=min(num_workers, len(items))) as pool:
            return torch.stack(list(pool.map(_s2, items)))

    def _raw_one(item):
        rel, lat, aug = item
        eff_lat = lat if lat in ("L", "R") else None
        return preprocess_dicom(
            rsna_raw / rel,
            target_size=(1520, 912),
            invert_monochrome1=True,
            mirror_right=True,
            laterality_policy="strict",
            crop_breast=True,
            use_largest_cc=False,
            strict_segmentation=strict_segmentation,
            simulate_16bit_png=True,
            apply_luts=True,
            laterality=eff_lat,
            augment_params=aug,
            standardize=True,
        )

    items = list(zip(rel_paths, lateralities, augs))
    if num_workers <= 1 or len(items) <= 1:
        return torch.stack([_raw_one(it) for it in items])
    with ThreadPoolExecutor(max_workers=min(num_workers, len(items))) as pool:
        return torch.stack(list(pool.map(_raw_one, items)))


def _encode_batch_features(encoder, bx, device, micro_batch_size: int, use_amp: bool):
    import torch

    chunks = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.no_grad():
        for s in range(0, len(bx), micro_batch_size):
            mb = bx[s : s + micro_batch_size].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                chunks.append(encoder(mb).float().cpu())
    return torch.cat(chunks, dim=0)


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=1800,
)
def benchmark_extract_batches_fn(
    num_batches: int = 5,
    epochs: int = 20,
    batch_size: int = 80,
    n_pos: int = 20,
    steps_per_epoch: Optional[int] = None,
    epoch_policy: str = "paper_independent",
    micro_batch_size: int = 80,
    precision: str = "amp",
    require_stage1_cache: bool = True,
    strict_segmentation: bool = True,
    num_workers: int = 4,
    seed: int = 42,
) -> dict:
    """Measure end-to-end Stage-1 -> Stage-2 -> L40S feature extraction speed on a few batches."""
    import numpy as np
    import pandas as pd
    import torch

    from src.data.feature_cache import compute_file_sha256, save_train_shard
    from src.data.preprocessing import get_stage1_config_hash
    from src.data.rsna_dataset import BalancedBatchSampler, build_epoch_draw_schedule
    from src.utils.env_metadata import collect_runtime_environment_metadata

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = precision == "amp"
    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    stage1_root = MOUNT_PATH / "cache" / "stage1_rsna_uint16"

    df = pd.read_csv(splits_csv)
    split_sha256 = compute_file_sha256(splits_csv)
    s1_hash = get_stage1_config_hash()
    stage1_loader = None
    if require_stage1_cache:
        stage1_loader, _ = _load_verified_stage1_loader(
            df=df, stage1_root=stage1_root, split_sha256=split_sha256, stage1_hash=s1_hash, strict_segmentation=strict_segmentation
        )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    tc_df = df[df["split"] == "train_calib"].reset_index(drop=True)
    vc_df = df[df["split"] == "val_calib"].reset_index(drop=True)

    train_labels = train_df["cancer"].values.astype(np.int64)
    train_paths = train_df["rel_path"].tolist()
    train_lats = train_df["laterality"].astype(str).tolist()
    train_img_ids = train_df["image_id"].astype(str).tolist()

    sampler = BalancedBatchSampler(
        labels=train_labels,
        batch_size=batch_size,
        n_pos=n_pos,
        seed=seed,
        epoch_policy=epoch_policy,
        steps_per_epoch=steps_per_epoch,
    )
    aug_rng = np.random.default_rng(seed + 1000)
    ep1_schedule = build_epoch_draw_schedule(sampler, 1, aug_rng)
    encoder, _, _ = _build_frozen_encoder(mammo_clip_tar, device)

    run_n = min(max(int(num_batches), 2), len(ep1_schedule))
    timed_batches = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for b_idx in range(run_n):
            rec = ep1_schedule[b_idx]
            t0 = time.perf_counter()
            bx = _prepare_stage2_batch(
                image_ids=[train_img_ids[i] for i in rec.indices],
                rel_paths=[train_paths[i] for i in rec.indices],
                lateralities=[train_lats[i] for i in rec.indices],
                aug_params_list=rec.aug_params,
                rsna_raw=rsna_raw,
                stage1_loader=stage1_loader,
                strict_segmentation=strict_segmentation,
                num_workers=num_workers,
            )
            t1 = time.perf_counter()
            feats = _encode_batch_features(encoder, bx, device, micro_batch_size, use_amp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            lbls = torch.tensor([train_labels[i] for i in rec.indices], dtype=torch.float32)
            idx_t = torch.tensor(rec.indices, dtype=torch.long)
            meta = save_train_shard(
                cache_root=tmp_path,
                epoch=1,
                shard_idx=b_idx,
                batch_start=b_idx,
                batch_end=b_idx + 1,
                features=feats.unsqueeze(0),
                labels=lbls.unsqueeze(0),
                sample_indices=idx_t.unsqueeze(0),
                image_ids=[[train_img_ids[i] for i in rec.indices]],
                aug_params=[[a.to_dict() for a in rec.aug_params]],
            )
            t3 = time.perf_counter()
            row_t = {
                "batch_idx": b_idx,
                "stage2_prep_sec": round(t1 - t0, 3),
                "encode_sec": round(t2 - t1, 3),
                "shard_write_sec": round(t3 - t2, 3),
                "total_sec": round(t3 - t0, 3),
                "shard_bytes": int(meta["size_bytes"]),
            }
            if b_idx > 0:
                timed_batches.append(row_t)
            print(f"[bench-extract] batch {b_idx + 1}/{run_n}: {row_t}", flush=True)

    mean_sec = float(np.mean([x["total_sec"] for x in timed_batches]))
    mean_prep_sec = float(np.mean([x["stage2_prep_sec"] for x in timed_batches]))
    mean_enc_sec = float(np.mean([x["encode_sec"] for x in timed_batches]))
    mean_disk_bytes = float(np.mean([x["shard_bytes"] for x in timed_batches]))

    total_train_batches = epochs * len(sampler)
    total_eval_imgs = len(tc_df) + len(vc_df)
    total_eval_equiv_batches = total_eval_imgs / float(batch_size)
    projected_total_sec = mean_sec * (total_train_batches + total_eval_equiv_batches)
    projected_storage_gb = ((mean_disk_bytes * total_train_batches) + (total_eval_imgs * 2048 * 4)) / (1024**3)

    report = {
        "require_stage1_cache": bool(require_stage1_cache),
        "precision": precision,
        "micro_batch_size": micro_batch_size,
        "warmup_excluded_batches": len(timed_batches),
        "mean_sec_per_batch_80": round(mean_sec, 4),
        "mean_stage2_prep_sec_per_batch": round(mean_prep_sec, 4),
        "mean_encode_sec_per_batch": round(mean_enc_sec, 4),
        "measured_imgs_per_sec": round(batch_size / max(mean_sec, 1e-6), 2),
        "steps_per_epoch": int(len(sampler)),
        "total_train_batches_20ep": int(total_train_batches),
        "total_train_occurrences_20ep": int(total_train_batches * batch_size),
        "projected_full_extract_hours": round(projected_total_sec / 3600.0, 3),
        "projected_full_extract_usd_l40s_est": round((projected_total_sec / 3600.0) * 1.95, 2),
        "projected_cache_storage_gib": round(projected_storage_gb, 3),
        "environment_metadata": collect_runtime_environment_metadata(precision_policy=precision),
        "per_batch_timings": timed_batches,
    }
    out_path = MOUNT_PATH / "models" / "l40s_extract_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    RSNA_VOL.commit()
    return report


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=86400,
)
def extract_augmentation_cache(
    epochs: int = 20,
    batch_size: int = 80,
    n_pos: int = 20,
    steps_per_epoch: Optional[int] = None,
    epoch_policy: str = "paper_independent",
    micro_batch_size: int = 80,
    batches_per_shard: int = 20,
    precision: str = "amp",
    require_stage1_cache: bool = True,
    strict_segmentation: bool = True,
    num_workers: int = 4,
    seed: int = 42,
) -> dict:
    """Build or resume sharded augmentation-aware FP32 cache for full 20-epoch RSNA training."""
    import numpy as np
    import pandas as pd
    import torch

    from src.data.feature_cache import (
        compute_file_sha256,
        extract_or_resume_cache,
        get_cache_manifest_sha256,
        verify_cache_integrity,
    )
    from src.data.preprocessing import get_preprocessing_config_hash, get_stage1_config_hash
    from src.data.rsna_dataset import BalancedBatchSampler, build_epoch_draw_schedule
    from src.utils.env_metadata import collect_runtime_environment_metadata

    use_amp = precision == "amp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    stage1_root = MOUNT_PATH / "cache" / "stage1_rsna_uint16"
    cache_root = MOUNT_PATH / "cache" / f"rsna_aug_seed{seed}"

    stop_event = threading.Event()
    _register_signal_handlers(RSNA_VOL, stop_event)
    commit_thread = _start_periodic_commit(RSNA_VOL, stop_event)

    try:
        df = pd.read_csv(splits_csv)
        split_sha256 = compute_file_sha256(splits_csv)
        s1_hash = get_stage1_config_hash()
        preprocess_hash = get_preprocessing_config_hash()

        stage1_loader = None
        if require_stage1_cache:
            stage1_loader, _ = _load_verified_stage1_loader(
                df=df,
                stage1_root=stage1_root,
                split_sha256=split_sha256,
                stage1_hash=s1_hash,
                strict_segmentation=strict_segmentation,
            )

        train_df = df[df["split"] == "train"].reset_index(drop=True)
        tc_df = df[df["split"] == "train_calib"].reset_index(drop=True)
        vc_df = df[df["split"] == "val_calib"].reset_index(drop=True)

        encoder, ckpt_sha256, _ = _build_frozen_encoder(mammo_clip_tar, device)
        env_meta = collect_runtime_environment_metadata(precision_policy=precision)

        train_labels = train_df["cancer"].values.astype(np.int64)
        train_paths = train_df["rel_path"].tolist()
        train_lats = train_df["laterality"].astype(str).tolist()
        train_img_ids = train_df["image_id"].astype(str).tolist()

        tc_img_ids = tc_df["image_id"].astype(str).tolist()
        tc_pt_ids = tc_df["patient_id"].astype(str).tolist()
        tc_lbls = tc_df["cancer"].values.astype(np.int64).tolist()
        vc_img_ids = vc_df["image_id"].astype(str).tolist()
        vc_pt_ids = vc_df["patient_id"].astype(str).tolist()
        vc_lbls = vc_df["cancer"].values.astype(np.int64).tolist()

        sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=batch_size,
            n_pos=n_pos,
            seed=seed,
            epoch_policy=epoch_policy,
            steps_per_epoch=steps_per_epoch,
        )
        aug_rng = np.random.default_rng(seed + 1000)
        schedules = {ep: build_epoch_draw_schedule(sampler, ep, aug_rng) for ep in range(1, epochs + 1)}

        def _extract_batch(rec):
            bx = _prepare_stage2_batch(
                image_ids=[train_img_ids[i] for i in rec.indices],
                rel_paths=[train_paths[i] for i in rec.indices],
                lateralities=[train_lats[i] for i in rec.indices],
                aug_params_list=rec.aug_params,
                rsna_raw=rsna_raw,
                stage1_loader=stage1_loader,
                strict_segmentation=strict_segmentation,
                num_workers=num_workers,
            )
            feats = _encode_batch_features(encoder, bx, device, micro_batch_size, use_amp)
            return (
                feats,
                torch.tensor([train_labels[i] for i in rec.indices], dtype=torch.float32),
                [train_img_ids[i] for i in rec.indices],
            )

        def _eval_extract(split_name: str):
            sdf = tc_df if split_name == "train_calib" else vc_df
            img_ids = sdf["image_id"].astype(str).tolist()
            rels = sdf["rel_path"].tolist()
            lats = sdf["laterality"].astype(str).tolist()
            chunks = []
            for s in range(0, len(img_ids), micro_batch_size):
                bx = _prepare_stage2_batch(
                    image_ids=img_ids[s : s + micro_batch_size],
                    rel_paths=rels[s : s + micro_batch_size],
                    lateralities=lats[s : s + micro_batch_size],
                    aug_params_list=None,
                    rsna_raw=rsna_raw,
                    stage1_loader=stage1_loader,
                    strict_segmentation=strict_segmentation,
                    num_workers=num_workers,
                )
                chunks.append(_encode_batch_features(encoder, bx, device, micro_batch_size, use_amp))
            return (
                torch.cat(chunks, dim=0),
                torch.tensor(sdf["cancer"].values, dtype=torch.float32),
                img_ids,
                sdf["patient_id"].astype(str).tolist(),
            )

        manifest = extract_or_resume_cache(
            cache_root=cache_root,
            schedules_by_epoch=schedules,
            extract_batch_fn=_extract_batch,
            eval_extract_fn=_eval_extract,
            model_checkpoint_sha256=ckpt_sha256,
            preprocess_config_hash=preprocess_hash,
            stage1_config_hash=s1_hash,
            split_manifest_sha256=split_sha256,
            seed=seed,
            batch_size=batch_size,
            n_pos=n_pos,
            epoch_policy=epoch_policy,
            embed_dim=2048,
            encoder_precision=precision,
            batches_per_shard=batches_per_shard,
            environment_metadata=env_meta,
            on_progress=lambda m: print(f"[cache] {m}", flush=True),
        )
        verify_cache_integrity(
            cache_root=cache_root,
            expected_model_sha256=ckpt_sha256,
            expected_preprocess_hash=preprocess_hash,
            expected_stage1_hash=s1_hash,
            expected_split_sha256=split_sha256,
            expected_epochs=epochs,
            expected_steps_per_epoch=len(sampler),
            expected_batch_size=batch_size,
            expected_n_pos=n_pos,
            expected_epoch_policy=epoch_policy,
            expected_embed_dim=2048,
            expected_encoder_precision=precision,
            expected_seed=seed,
            expected_schedules=schedules,
            expected_eval_meta={
                "train_calib": (tc_img_ids, tc_pt_ids, tc_lbls),
                "val_calib": (vc_img_ids, vc_pt_ids, vc_lbls),
            },
            verify_shard_checksums=True,
        )
        cache_manifest_sha = get_cache_manifest_sha256(cache_root)
        RSNA_VOL.commit()
        return {
            "status": "VERIFIED",
            "cache_root": str(cache_root),
            "cache_manifest_sha256": cache_manifest_sha,
            "num_shards": len(manifest["train_shards"]),
            "steps_per_epoch": len(sampler),
            "total_occurrences": epochs * len(sampler) * batch_size,
        }
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)
        try:
            RSNA_VOL.commit()
        except Exception:
            pass


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    cpu=4.0,
    memory=16384,
    timeout=3600,
)
def verify_cache_only(
    epochs: int = 20,
    batch_size: int = 80,
    n_pos: int = 20,
    steps_per_epoch: Optional[int] = None,
    epoch_policy: str = "paper_independent",
    precision: str = "amp",
    seed: int = 42,
) -> dict:
    """Strictly verify the RSNA augmentation-aware feature cache on Modal CPU."""
    import numpy as np
    import pandas as pd

    from src.data.feature_cache import (
        compute_file_sha256,
        get_cache_manifest_sha256,
        verify_cache_integrity,
    )
    from src.data.preprocessing import get_preprocessing_config_hash, get_stage1_config_hash
    from src.data.rsna_dataset import BalancedBatchSampler, build_epoch_draw_schedule
    from src.models.mammo_clip_loader import MAMMO_CLIP_SHA256

    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    cache_root = MOUNT_PATH / "cache" / f"rsna_aug_seed{seed}"
    df = pd.read_csv(splits_csv)
    split_sha256 = compute_file_sha256(splits_csv)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    tc_df = df[df["split"] == "train_calib"].reset_index(drop=True)
    vc_df = df[df["split"] == "val_calib"].reset_index(drop=True)

    sampler = BalancedBatchSampler(
        labels=train_df["cancer"].values.astype(np.int64),
        batch_size=batch_size,
        n_pos=n_pos,
        seed=seed,
        epoch_policy=epoch_policy,
        steps_per_epoch=steps_per_epoch,
    )
    aug_rng = np.random.default_rng(seed + 1000)
    schedules = {ep: build_epoch_draw_schedule(sampler, ep, aug_rng) for ep in range(1, epochs + 1)}

    manifest = verify_cache_integrity(
        cache_root=cache_root,
        expected_model_sha256=MAMMO_CLIP_SHA256,
        expected_preprocess_hash=get_preprocessing_config_hash(),
        expected_stage1_hash=get_stage1_config_hash(),
        expected_split_sha256=split_sha256,
        expected_epochs=epochs,
        expected_steps_per_epoch=len(sampler),
        expected_batch_size=batch_size,
        expected_n_pos=n_pos,
        expected_epoch_policy=epoch_policy,
        expected_embed_dim=2048,
        expected_encoder_precision=precision,
        expected_seed=seed,
        expected_schedules=schedules,
        expected_eval_meta={
            "train_calib": (
                tc_df["image_id"].astype(str).tolist(),
                tc_df["patient_id"].astype(str).tolist(),
                tc_df["cancer"].values.astype(np.int64).tolist(),
            ),
            "val_calib": (
                vc_df["image_id"].astype(str).tolist(),
                vc_df["patient_id"].astype(str).tolist(),
                vc_df["cancer"].values.astype(np.int64).tolist(),
            ),
        },
        verify_shard_checksums=True,
    )
    return {
        "status": "VERIFIED",
        "cache_root": str(cache_root),
        "cache_manifest_sha256": get_cache_manifest_sha256(cache_root),
        "num_train_shards": len(manifest["train_shards"]),
        "epochs": epochs,
        "steps_per_epoch": len(sampler),
        "total_occurrences": epochs * len(sampler) * batch_size,
    }


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=86400,
)
def train_model(
    baseline: str = "trust",
    epochs: int = 20,
    batch_size: int = 80,
    n_pos: int = 20,
    micro_batch_size: int = 80,
    steps_per_epoch: Optional[int] = None,
    epoch_policy: str = "paper_independent",
    lr: float = 3e-5,
    weight_decay: float = 1e-4,
    margin: float = 0.10,
    lambda_focal: float = 1.0,
    lambda_dismiss: float = 0.05,
    initial_tau: float = 0.05,
    grad_clip: float = 5.0,
    precision: str = "amp",
    require_stage1_cache: bool = True,
    strict_segmentation: bool = True,
    seed: int = 42,
    num_workers: int = 4,
    use_cached_features: bool = True,
    resume: bool = False,
) -> dict:
    """Train CE-only, CE+Focal, Fixed-tau, or TRUST for 20 epochs using final-epoch selection."""
    import numpy as np
    import pandas as pd
    import torch

    from src.data.feature_cache import (
        compute_file_sha256,
        get_cache_manifest_sha256,
        load_epoch_batches_from_cache,
        verify_cache_integrity,
    )
    from src.data.preprocessing import get_preprocessing_config_hash, get_stage1_config_hash, sample_augmentation_params
    from src.data.rsna_dataset import BalancedBatchSampler
    from src.models.losses import TRUSTLoss
    from src.models.mammo_clip_loader import MAMMO_CLIP_SHA256
    from src.models.trust_model import aggregate_case_labels, aggregate_case_scores
    from src.training.calibration import (
        compute_dynamic_training_threshold_with_diagnostics,
        compute_target_recall_threshold,
        evaluate_dismissal_at_threshold,
    )
    from src.training.checkpointing import load_training_checkpoint, save_training_checkpoint
    from src.training.engine import build_trust_head, run_effective_batch_step
    from src.utils.env_metadata import collect_runtime_environment_metadata

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_amp = precision == "amp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_meta = collect_runtime_environment_metadata(precision_policy=precision)
    print(
        f"[train] Starting {baseline.upper()} on {device} ({env_meta['gpu_name']}) | "
        f"policy={epoch_policy} | cached={use_cached_features} | precision={precision}",
        flush=True,
    )

    stop_event = threading.Event()
    _register_signal_handlers(RSNA_VOL, stop_event)
    commit_thread = _start_periodic_commit(RSNA_VOL, stop_event)

    try:
        rsna_raw = MOUNT_PATH / "rsna" / "raw"
        stage1_root = MOUNT_PATH / "cache" / "stage1_rsna_uint16"
        ckpt_dir = MOUNT_PATH / "models" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
        mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
        cache_root = MOUNT_PATH / "cache" / f"rsna_aug_seed{seed}"

        df = pd.read_csv(splits_csv)
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        tc_df = df[df["split"] == "train_calib"].reset_index(drop=True)
        vc_df = df[df["split"] == "val_calib"].reset_index(drop=True)

        s1_hash = get_stage1_config_hash()
        preprocess_hash = get_preprocessing_config_hash()
        split_sha256 = compute_file_sha256(splits_csv)

        sampler = BalancedBatchSampler(
            labels=train_df["cancer"].values,
            batch_size=batch_size,
            n_pos=n_pos,
            seed=seed,
            epoch_policy=epoch_policy,
            steps_per_epoch=steps_per_epoch,
        )
        aug_rng = np.random.default_rng(seed + 1000)

        cache_manifest = None
        cache_manifest_sha = ""
        tc_cache = None
        vc_cache = None
        encoder = None
        stage1_loader = None

        if use_cached_features:
            cache_manifest = verify_cache_integrity(
                cache_root=cache_root,
                expected_model_sha256=MAMMO_CLIP_SHA256,
                expected_preprocess_hash=preprocess_hash,
                expected_stage1_hash=s1_hash,
                expected_split_sha256=split_sha256,
                expected_epochs=epochs,
                expected_steps_per_epoch=len(sampler),
                expected_batch_size=batch_size,
                expected_n_pos=n_pos,
                expected_epoch_policy=epoch_policy,
                expected_embed_dim=2048,
                expected_encoder_precision=precision,
                expected_seed=seed,
                expected_eval_meta={
                    "train_calib": (
                        tc_df["image_id"].astype(str).tolist(),
                        tc_df["patient_id"].astype(str).tolist(),
                        tc_df["cancer"].values.astype(np.int64).tolist(),
                    ),
                    "val_calib": (
                        vc_df["image_id"].astype(str).tolist(),
                        vc_df["patient_id"].astype(str).tolist(),
                        vc_df["cancer"].values.astype(np.int64).tolist(),
                    ),
                },
                verify_shard_checksums=True,
            )
            cache_manifest_sha = get_cache_manifest_sha256(cache_root)
            tc_cache = torch.load(
                str(cache_root / cache_manifest["eval_splits"]["train_calib"]["file"]),
                map_location="cpu",
                weights_only=False,
            )
            vc_cache = torch.load(
                str(cache_root / cache_manifest["eval_splits"]["val_calib"]["file"]),
                map_location="cpu",
                weights_only=False,
            )
            ckpt_sha256 = MAMMO_CLIP_SHA256
        else:
            if require_stage1_cache:
                stage1_loader, _ = _load_verified_stage1_loader(
                    df=df,
                    stage1_root=stage1_root,
                    split_sha256=split_sha256,
                    stage1_hash=s1_hash,
                    strict_segmentation=strict_segmentation,
                )
            encoder, ckpt_sha256, _ = _build_frozen_encoder(mammo_clip_tar, device)

        head = build_trust_head(embed_dim=2048, dropout_rate=0.3, seed=seed, device=device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and device.type == "cuda"))
        criterion = TRUSTLoss(
            baseline=baseline,
            margin=margin,
            focal_gamma=2.0,
            focal_alpha=None,
            lambda_focal=lambda_focal,
            lambda_dismiss=lambda_dismiss,
        )

        training_config: Dict[str, Any] = {
            "baseline": str(baseline),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "n_pos": int(n_pos),
            "steps_per_epoch": int(len(sampler)),
            "epoch_policy": str(epoch_policy),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "margin": float(margin),
            "lambda_focal": float(lambda_focal),
            "lambda_dismiss": float(lambda_dismiss),
            "initial_tau": float(initial_tau),
            "grad_clip": float(grad_clip),
            "micro_batch_size": int(micro_batch_size),
            "precision_policy": str(precision),
            "seed": int(seed),
            "preprocess_config_hash": str(preprocess_hash),
            "stage1_config_hash": str(s1_hash),
            "model_checkpoint_sha256": str(ckpt_sha256),
            "split_manifest_sha256": str(split_sha256),
            "use_cached_features": bool(use_cached_features),
            "cache_manifest_sha256": str(cache_manifest_sha),
        }

        latest_ckpt = ckpt_dir / f"{baseline}_rsna_seed{seed}_latest.pt"
        final_ckpt = ckpt_dir / f"{baseline}_rsna_seed{seed}_final_epoch{epochs}.pt"

        start_epoch = 1
        tau_t = float(initial_tau)
        history: List[Dict[str, Any]] = []

        if resume:
            completed_epoch, tau_t, history, _ = load_training_checkpoint(
                checkpoint_path=latest_ckpt,
                head=head,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                sampler=sampler,
                aug_rng=aug_rng,
                expected_config=training_config,
            )
            start_epoch = completed_epoch + 1
            print(f"[resume] Resumed epoch {completed_epoch} -> continuing at {start_epoch}/{epochs} (tau_t={tau_t:.5f})", flush=True)

        train_img_ids = train_df["image_id"].astype(str).tolist()
        train_paths = train_df["rel_path"].tolist()
        train_lats = train_df["laterality"].astype(str).tolist()
        train_labels = train_df["cancer"].values

        tc_img_ids = tc_df["image_id"].astype(str).tolist()
        tc_paths = tc_df["rel_path"].tolist()
        tc_lats = tc_df["laterality"].astype(str).tolist()
        tc_labels = tc_df["cancer"].values.astype(np.int64)

        for epoch in range(start_epoch, epochs + 1):
            t_ep0 = time.time()
            tau_diag: Dict[str, Any] = {"tau_t": float(tau_t), "is_epoch1_or_non_trust": True}

            if baseline == "trust" and epoch >= 2:
                head.eval()
                if use_cached_features:
                    amp_enabled = bool(use_amp and device.type == "cuda")
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
                        tc_probs = torch.sigmoid(head(tc_cache["features"].to(device))).float().squeeze(1).cpu().numpy()
                else:
                    probs_chunks = []
                    amp_enabled = bool(use_amp and device.type == "cuda")
                    with torch.no_grad():
                        for s in range(0, len(tc_img_ids), micro_batch_size):
                            bx = _prepare_stage2_batch(
                                image_ids=tc_img_ids[s : s + micro_batch_size],
                                rel_paths=tc_paths[s : s + micro_batch_size],
                                lateralities=tc_lats[s : s + micro_batch_size],
                                aug_params_list=None,
                                rsna_raw=rsna_raw,
                                stage1_loader=stage1_loader,
                                strict_segmentation=strict_segmentation,
                                num_workers=num_workers,
                            ).to(device)
                            with torch.amp.autocast("cuda", enabled=amp_enabled):
                                probs_chunks.append(torch.sigmoid(head(encoder(bx).float())).float().squeeze(1).cpu())
                    tc_probs = torch.cat(probs_chunks, dim=0).numpy()

                tau_t, tau_diag = compute_dynamic_training_threshold_with_diagnostics(
                    image_scores=tc_probs,
                    image_labels=tc_labels,
                    max_upper_bound=0.01,
                    confidence=0.95,
                )

            head.train()
            ep_loss = 0.0
            if use_cached_features:
                batches = load_epoch_batches_from_cache(cache_root, cache_manifest, epoch)
                for b_idx, (b_feats, b_labels, _) in enumerate(batches):
                    step_loss, _, _, _ = run_effective_batch_step(
                        head=head,
                        optimizer=optimizer,
                        scaler=scaler,
                        criterion=criterion,
                        batch_targets=b_labels,
                        tau_t=tau_t,
                        micro_batch_size=micro_batch_size,
                        grad_clip=grad_clip,
                        use_amp=use_amp,
                        device=device,
                        seed=seed,
                        epoch=epoch,
                        batch_idx=b_idx,
                        batch_features=b_feats,
                        capture_features_and_grads=False,
                    )
                    ep_loss += step_loss
                n_batches = len(batches)
            else:
                batch_list = list(sampler)
                for b_idx, indices in enumerate(batch_list):
                    augs = [sample_augmentation_params(aug_rng) for _ in indices]
                    bx = _prepare_stage2_batch(
                        image_ids=[train_img_ids[i] for i in indices],
                        rel_paths=[train_paths[i] for i in indices],
                        lateralities=[train_lats[i] for i in indices],
                        aug_params_list=augs,
                        rsna_raw=rsna_raw,
                        stage1_loader=stage1_loader,
                        strict_segmentation=strict_segmentation,
                        num_workers=num_workers,
                    )
                    by = torch.tensor([train_labels[i] for i in indices], dtype=torch.float32)
                    step_loss, _, _, _ = run_effective_batch_step(
                        head=head,
                        optimizer=optimizer,
                        scaler=scaler,
                        criterion=criterion,
                        batch_targets=by,
                        tau_t=tau_t,
                        micro_batch_size=micro_batch_size,
                        grad_clip=grad_clip,
                        use_amp=use_amp,
                        device=device,
                        seed=seed,
                        epoch=epoch,
                        batch_idx=b_idx,
                        batch_inputs=bx,
                        encoder=encoder,
                        capture_features_and_grads=False,
                    )
                    ep_loss += step_loss
                n_batches = len(batch_list)

            ep_loss /= max(n_batches, 1)
            ep_sec = time.time() - t_ep0
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(ep_loss, 5),
                    "tau_t": round(tau_t, 5),
                    "tau_diagnostics": tau_diag,
                    "time_sec": round(ep_sec, 2),
                }
            )
            print(f"Epoch {epoch:02d}/{epochs:02d} | train_loss={ep_loss:.5f} | tau_t={tau_t:.5f} | {ep_sec:.1f}s", flush=True)

            save_training_checkpoint(
                checkpoint_path=latest_ckpt,
                head=head,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                tau_t=tau_t,
                history=history,
                training_config=training_config,
                sampler=sampler,
                aug_rng=aug_rng,
            )
            RSNA_VOL.commit()

        # Save final 20th-epoch checkpoint (no early stopping, no val_calib selection)
        save_training_checkpoint(
            checkpoint_path=final_ckpt,
            head=head,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epochs,
            tau_t=tau_t,
            history=history,
            training_config=training_config,
            sampler=sampler,
            aug_rng=aug_rng,
        )

        head.eval()
        amp_enabled = bool(use_amp and device.type == "cuda")
        if use_cached_features:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
                vc_probs = torch.sigmoid(head(vc_cache["features"].to(device))).float().squeeze(1).cpu()
            vc_pids = vc_cache["patient_ids"]
            vc_lbls = [int(v) for v in vc_cache["labels"].tolist()]
        else:
            vc_img_ids = vc_df["image_id"].astype(str).tolist()
            vc_paths = vc_df["rel_path"].tolist()
            vc_lats = vc_df["laterality"].astype(str).tolist()
            probs_chunks = []
            with torch.no_grad():
                for s in range(0, len(vc_img_ids), micro_batch_size):
                    bx = _prepare_stage2_batch(
                        image_ids=vc_img_ids[s : s + micro_batch_size],
                        rel_paths=vc_paths[s : s + micro_batch_size],
                        lateralities=vc_lats[s : s + micro_batch_size],
                        aug_params_list=None,
                        rsna_raw=rsna_raw,
                        stage1_loader=stage1_loader,
                        strict_segmentation=strict_segmentation,
                        num_workers=num_workers,
                    ).to(device)
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        probs_chunks.append(torch.sigmoid(head(encoder(bx).float())).float().squeeze(1).cpu())
            vc_probs = torch.cat(probs_chunks, dim=0)
            vc_pids = vc_df["patient_id"].astype(str).tolist()
            vc_lbls = vc_df["cancer"].astype(int).tolist()

        case_scores_map = aggregate_case_scores(vc_probs, vc_pids)
        case_labels_map = aggregate_case_labels(vc_lbls, vc_pids)
        pids_sorted = sorted(case_scores_map.keys())
        c_scores = np.array([case_scores_map[p] for p in pids_sorted], dtype=np.float64)
        c_labels = np.array([case_labels_map[p] for p in pids_sorted], dtype=np.int64)
        pos_c_scores = c_scores[c_labels == 1]

        tau_95 = compute_target_recall_threshold(pos_c_scores, target_recall=0.95)
        tau_98 = compute_target_recall_threshold(pos_c_scores, target_recall=0.98)
        eval_results = {
            "target_95": evaluate_dismissal_at_threshold(c_scores, c_labels, tau_95, confidence=0.99, target_recall=0.95),
            "target_98": evaluate_dismissal_at_threshold(c_scores, c_labels, tau_98, confidence=0.99, target_recall=0.98),
            "checkpoint_path": str(final_ckpt),
            "training_config": training_config,
            "history": history,
            "environment_metadata": env_meta,
        }

        res_path = MOUNT_PATH / "models" / f"rsna_{baseline}_eval_results.json"
        res_path.write_text(json.dumps(eval_results, indent=2))
        RSNA_VOL.commit()

        return {
            "status": "SUCCESS",
            "baseline": baseline,
            "epochs": epochs,
            "checkpoint_path": str(final_ckpt),
            "operating_thresholds": {
                "target_95": eval_results["target_95"],
                "target_98": eval_results["target_98"],
            },
        }
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)
        try:
            RSNA_VOL.commit()
        except Exception:
            pass


@app.local_entrypoint()
def main(
    baseline: str = "trust",
    epochs: int = 20,
    batch_size: int = 80,
    n_pos: int = 20,
    micro_batch_size: int = 80,
    steps_per_epoch: int = 0,
    epoch_policy: str = "paper_independent",
    lr: float = 3e-5,
    weight_decay: float = 1e-4,
    margin: float = 0.10,
    lambda_focal: float = 1.0,
    lambda_dismiss: float = 0.05,
    initial_tau: float = 0.05,
    grad_clip: float = 5.0,
    precision: str = "amp",
    require_stage1_cache: bool = True,
    strict_segmentation: bool = True,
    num_workers: int = 4,
    batches_per_shard: int = 20,
    gpu: str = "L40S",
    seed: int = 42,
    extract_cache: bool = False,
    verify_cache: bool = False,
    benchmark_extract_batches: int = 0,
    use_cached_features: bool = True,
    resume: bool = False,
) -> None:
    gpu_norm = gpu.upper()
    if gpu_norm not in SUPPORTED_GPUS:
        raise ValueError(f"Unsupported GPU {gpu!r}. Choose from {SUPPORTED_GPUS}.")
    if precision not in ("amp", "fp32"):
        raise ValueError(f"Unsupported precision={precision!r}; choose 'amp' or 'fp32'.")
    eff_steps = steps_per_epoch if steps_per_epoch > 0 else None

    if benchmark_extract_batches > 0:
        print(f"Benchmarking {benchmark_extract_batches} batches of Stage-1 -> Stage-2 -> {gpu_norm} extraction...")
        res = benchmark_extract_batches_fn.with_options(gpu=gpu_norm).remote(
            num_batches=benchmark_extract_batches,
            epochs=epochs,
            batch_size=batch_size,
            n_pos=n_pos,
            steps_per_epoch=eff_steps,
            epoch_policy=epoch_policy,
            micro_batch_size=micro_batch_size,
            precision=precision,
            require_stage1_cache=require_stage1_cache,
            strict_segmentation=strict_segmentation,
            num_workers=num_workers,
            seed=seed,
        )
        print(json.dumps(res, indent=2))
        return

    if verify_cache:
        print("Verifying RSNA augmentation-aware feature cache on Modal CPU...")
        res = verify_cache_only.remote(
            epochs=epochs,
            batch_size=batch_size,
            n_pos=n_pos,
            steps_per_epoch=eff_steps,
            epoch_policy=epoch_policy,
            precision=precision,
            seed=seed,
        )
        print(json.dumps(res, indent=2))
        return

    if extract_cache:
        print(f"Launching augmentation-aware cache extraction on Modal ({gpu_norm}, precision={precision})...")
        res = extract_augmentation_cache.with_options(gpu=gpu_norm).remote(
            epochs=epochs,
            batch_size=batch_size,
            n_pos=n_pos,
            steps_per_epoch=eff_steps,
            epoch_policy=epoch_policy,
            micro_batch_size=micro_batch_size,
            batches_per_shard=batches_per_shard,
            precision=precision,
            require_stage1_cache=require_stage1_cache,
            strict_segmentation=strict_segmentation,
            num_workers=num_workers,
            seed=seed,
        )
        print("Extraction finished:", json.dumps(res, indent=2))
        return

    print(
        f"Launching {baseline.upper()} training on Modal "
        f"(GPU={gpu_norm}, epochs={epochs}, cached={use_cached_features}, resume={resume}, precision={precision})..."
    )
    res = train_model.with_options(gpu=gpu_norm).remote(
        baseline=baseline,
        epochs=epochs,
        batch_size=batch_size,
        n_pos=n_pos,
        micro_batch_size=micro_batch_size,
        steps_per_epoch=eff_steps,
        epoch_policy=epoch_policy,
        lr=lr,
        weight_decay=weight_decay,
        margin=margin,
        lambda_focal=lambda_focal,
        lambda_dismiss=lambda_dismiss,
        initial_tau=initial_tau,
        grad_clip=grad_clip,
        precision=precision,
        require_stage1_cache=require_stage1_cache,
        strict_segmentation=strict_segmentation,
        seed=seed,
        num_workers=num_workers,
        use_cached_features=use_cached_features,
        resume=resume,
    )
    print("Training finished:", json.dumps(res, indent=2))
