"""Corrected TRUST & CE-Only Training Pipeline on Modal.

Reference:
    TRUST (Shafie et al., 2026, arXiv:2609.00300v1): Sections 3.2 - 3.10
    - Backbone: Mammo-CLIP EfficientNet-B5 (frozen, strict 852/852 parity, pinned SHA-256).
    - Head: LayerNorm(2048) -> Dropout(0.3) -> Linear(2048, 1).
    - Preprocessing: Modality/VOI LUT -> MONOCHROME1 invert -> right-breast mirror ->
      border-median breast segmentation & 10% crop -> 1520x912 linear resize -> [0,1] norm ->
      stochastic training augmentation -> 3ch Mammo-CLIP standardization.
    - Minibatch: 80 (20 positives, 60 negatives), independent minibatch draws per TRUST Sec 3.9.
    - Loss: L_CE + 1.0 * L_focal (gamma=2.0, no alpha) + 0.05 * L_dismiss (margin=0.10).
    - Dynamic tau_t: Initialized at 0.05; updated at beginning of each epoch >= 2 on image-level
      train_calib scores using 95% Clopper-Pearson upper bound <= 1% (0.0 if none satisfies).
    - Model Selection: Fixed 20 epochs, NO early stopping or val_calib checkpoint selection.
      Final 20th-epoch checkpoint evaluated once on val_calib for 95% and 98% source thresholds.
    - Supports online training (with optional Stage-1 deterministic cropped cache), full
      augmentation-aware cache extraction (--extract-cache), cached training (--use-cached-features),
      and exact complete-RNG resume (--resume).

Usage (DO NOT EXECUTE AUTOMATICALLY - for user execution):
    modal run scripts/train_trust_modal.py --extract-cache --gpu L40S
    modal run scripts/train_trust_modal.py --baseline ce --epochs 20 --gpu L40S --use-cached-features
    modal run scripts/train_trust_modal.py --baseline trust --epochs 20 --gpu L40S --use-cached-features
"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import random
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import modal

APP_NAME = "anchorshift-stage-b-train"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
MOUNT_PATH = Path("/data")
COMMIT_INTERVAL_SEC = 60
SUPPORTED_GPUS = ("T4", "L4", "A10G", "L40S", "A100", "H100")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "omegaconf>=2.3.0",
        "pandas>=2.2.0",
        "pydicom>=2.4.4",
        "opencv-python-headless>=4.10.0",
        "pylibjpeg>=2.0.0",
        "pylibjpeg-libjpeg>=2.1.0",
        "pylibjpeg-openjpeg>=2.1.0",
        "python-gdcm>=3.0.24",
        "scipy>=1.12.0",
        "numpy>=1.26.0",
    )
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


def _preprocess_batch_online(
    rsna_raw: Path,
    rel_paths: List[str],
    lateralities: List[str],
    aug_params_list: Optional[List[Any]],
    num_workers: int = 4,
):
    import torch
    from src.data.preprocessing import preprocess_dicom

    def _one(args):
        rel, lat, aug = args
        eff_lat = lat if lat in ("L", "R") else None
        return preprocess_dicom(
            rsna_raw / rel,
            target_size=(1520, 912),
            invert_monochrome1=True,
            mirror_right=True,
            crop_breast=True,
            apply_luts=True,
            laterality=eff_lat,
            augment_params=aug,
            standardize=True,
        )

    augs = aug_params_list if aug_params_list is not None else [None] * len(rel_paths)
    items = list(zip(rel_paths, lateralities, augs))
    if num_workers <= 1 or len(items) <= 1:
        return torch.stack([_one(it) for it in items])
    with ThreadPoolExecutor(max_workers=min(num_workers, len(items))) as pool:
        return torch.stack(list(pool.map(_one, items)))


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
        verify_cache_integrity,
    )
    from src.data.preprocessing import get_preprocessing_config_hash
    from src.data.rsna_dataset import BalancedBatchSampler, build_epoch_draw_schedule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    cache_root = MOUNT_PATH / "cache" / f"rsna_aug_seed{seed}"

    stop_event = threading.Event()
    _register_signal_handlers(RSNA_VOL, stop_event)
    commit_thread = _start_periodic_commit(RSNA_VOL, stop_event)

    try:
        df = pd.read_csv(splits_csv)
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        tc_df = df[df["split"] == "train_calib"].reset_index(drop=True)
        vc_df = df[df["split"] == "val_calib"].reset_index(drop=True)

        encoder, ckpt_sha256, _ = _build_frozen_encoder(mammo_clip_tar, device)
        preprocess_hash = get_preprocessing_config_hash()
        split_sha256 = compute_file_sha256(splits_csv)

        train_labels = train_df["cancer"].values.astype(np.int64)
        train_paths = train_df["rel_path"].tolist()
        train_lats = train_df["laterality"].astype(str).tolist() if "laterality" in train_df.columns else [""] * len(train_df)
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
        schedules = {ep: build_epoch_draw_schedule(sampler, ep, aug_rng) for ep in range(1, epochs + 1)}

        def _extract_batch(rec):
            rels = [train_paths[i] for i in rec.indices]
            lats = [train_lats[i] for i in rec.indices]
            bx = _preprocess_batch_online(rsna_raw, rels, lats, rec.aug_params, num_workers=num_workers)
            chunks = []
            with torch.no_grad():
                for s in range(0, len(bx), micro_batch_size):
                    mb = bx[s : s + micro_batch_size].to(device)
                    with torch.amp.autocast("cuda"):
                        chunks.append(encoder(mb).float().cpu())
            return (
                torch.cat(chunks, dim=0),
                torch.tensor([train_labels[i] for i in rec.indices], dtype=torch.float32),
                [train_img_ids[i] for i in rec.indices],
            )

        def _eval_extract(split_name: str):
            sdf = tc_df if split_name == "train_calib" else vc_df
            rels = sdf["rel_path"].tolist()
            lats = sdf["laterality"].astype(str).tolist() if "laterality" in sdf.columns else [""] * len(sdf)
            chunks = []
            with torch.no_grad():
                for s in range(0, len(rels), micro_batch_size):
                    bx = _preprocess_batch_online(
                        rsna_raw, rels[s : s + micro_batch_size], lats[s : s + micro_batch_size], None, num_workers=num_workers
                    ).to(device)
                    with torch.amp.autocast("cuda"):
                        chunks.append(encoder(bx).float().cpu())
            return (
                torch.cat(chunks, dim=0),
                torch.tensor(sdf["cancer"].values, dtype=torch.float32),
                sdf["image_id"].astype(str).tolist(),
                sdf["patient_id"].astype(str).tolist(),
            )

        manifest = extract_or_resume_cache(
            cache_root=cache_root,
            schedules_by_epoch=schedules,
            extract_batch_fn=_extract_batch,
            eval_extract_fn=_eval_extract,
            model_checkpoint_sha256=ckpt_sha256,
            preprocess_config_hash=preprocess_hash,
            split_manifest_sha256=split_sha256,
            seed=seed,
            batch_size=batch_size,
            n_pos=n_pos,
            epoch_policy=epoch_policy,
            embed_dim=2048,
            batches_per_shard=batches_per_shard,
            on_progress=lambda m: print(f"[cache] {m}", flush=True),
        )
        verify_cache_integrity(
            cache_root=cache_root,
            expected_model_sha256=ckpt_sha256,
            expected_preprocess_hash=preprocess_hash,
            expected_split_sha256=split_sha256,
            expected_epochs=epochs,
            expected_steps_per_epoch=len(sampler),
            expected_schedules=schedules,
            verify_shard_checksums=True,
        )
        RSNA_VOL.commit()
        return {"status": "VERIFIED", "cache_root": str(cache_root), "num_shards": len(manifest["train_shards"])}
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
    seed: int = 42,
    num_workers: int = 4,
    use_cached_features: bool = True,
    resume: bool = False,
) -> dict:
    """Train CE-only, CE+Focal, Fixed-tau, or TRUST for 20 epochs using final-epoch selection."""
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn

    from src.data.feature_cache import (
        compute_file_sha256,
        load_epoch_batches_from_cache,
        verify_cache_integrity,
    )
    from src.data.preprocessing import get_preprocessing_config_hash, sample_augmentation_params
    from src.data.rsna_dataset import BalancedBatchSampler
    from src.models.losses import TRUSTLoss
    from src.models.mammo_clip_loader import MAMMO_CLIP_SHA256
    from src.models.trust_model import aggregate_case_labels, aggregate_case_scores
    from src.training.calibration import (
        compute_dynamic_training_threshold,
        compute_target_recall_threshold,
        evaluate_dismissal_at_threshold,
    )
    from src.training.checkpointing import load_training_checkpoint, save_training_checkpoint

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[train] Starting {baseline.upper()} on {device} ({gpu_name}) | policy={epoch_policy} | cached={use_cached_features}", flush=True)

    stop_event = threading.Event()
    _register_signal_handlers(RSNA_VOL, stop_event)
    commit_thread = _start_periodic_commit(RSNA_VOL, stop_event)

    try:
        rsna_raw = MOUNT_PATH / "rsna" / "raw"
        ckpt_dir = MOUNT_PATH / "models" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
        mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
        cache_root = MOUNT_PATH / "cache" / f"rsna_aug_seed{seed}"

        df = pd.read_csv(splits_csv)
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        tc_df = df[df["split"] == "train_calib"].reset_index(drop=True)
        vc_df = df[df["split"] == "val_calib"].reset_index(drop=True)

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
        tc_cache = None
        vc_cache = None
        encoder = None
        if use_cached_features:
            cache_manifest = verify_cache_integrity(
                cache_root=cache_root,
                expected_model_sha256=MAMMO_CLIP_SHA256,
                expected_preprocess_hash=preprocess_hash,
                expected_split_sha256=split_sha256,
                expected_epochs=epochs,
                expected_steps_per_epoch=len(sampler),
                verify_shard_checksums=True,
            )
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
            encoder, ckpt_sha256, _ = _build_frozen_encoder(mammo_clip_tar, device)

        head = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Dropout(0.3),
            nn.Linear(2048, 1),
        ).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        criterion = TRUSTLoss(
            baseline=baseline,
            margin=margin,
            focal_gamma=2.0,
            focal_alpha=None,
            lambda_focal=lambda_focal,
            lambda_dismiss=lambda_dismiss,
        )

        training_config = {
            "baseline": baseline,
            "epochs": epochs,
            "batch_size": batch_size,
            "n_pos": n_pos,
            "steps_per_epoch": len(sampler),
            "epoch_policy": epoch_policy,
            "lr": lr,
            "weight_decay": weight_decay,
            "margin": margin,
            "lambda_focal": lambda_focal,
            "lambda_dismiss": lambda_dismiss,
            "initial_tau": initial_tau,
            "seed": seed,
            "preprocess_config_hash": preprocess_hash,
            "model_checkpoint_sha256": ckpt_sha256,
            "use_cached_features": use_cached_features,
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
            print(f"[resume] Resumed from epoch {completed_epoch} -> continuing at {start_epoch}/{epochs} (tau_t={tau_t:.5f})", flush=True)

        train_paths = train_df["rel_path"].tolist()
        train_lats = train_df["laterality"].astype(str).tolist() if "laterality" in train_df.columns else [""] * len(train_df)
        train_labels = train_df["cancer"].values
        tc_paths = tc_df["rel_path"].tolist()
        tc_lats = tc_df["laterality"].astype(str).tolist() if "laterality" in tc_df.columns else [""] * len(tc_df)
        tc_labels = tc_df["cancer"].values.astype(np.int64)

        for epoch in range(start_epoch, epochs + 1):
            t_ep0 = time.time()

            # TRUST Section 3.8: recompute tau_t at beginning of each epoch (epoch >= 2) using CP95 <= 1% on image-level train_calib
            if baseline == "trust" and epoch >= 2:
                head.eval()
                if use_cached_features:
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        tc_probs = torch.sigmoid(head(tc_cache["features"].to(device))).float().squeeze(1).cpu().numpy()
                else:
                    probs_chunks = []
                    with torch.no_grad():
                        for s in range(0, len(tc_paths), micro_batch_size):
                            bx = _preprocess_batch_online(
                                rsna_raw, tc_paths[s : s + micro_batch_size], tc_lats[s : s + micro_batch_size], None, num_workers=num_workers
                            ).to(device)
                            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                                probs_chunks.append(torch.sigmoid(head(encoder(bx))).float().squeeze(1).cpu())
                    tc_probs = torch.cat(probs_chunks, dim=0).numpy()
                tau_t = compute_dynamic_training_threshold(
                    image_scores=tc_probs,
                    image_labels=tc_labels,
                    max_upper_bound=0.01,
                    confidence=0.95,
                )

            head.train()
            ep_loss = 0.0
            if use_cached_features:
                batches = load_epoch_batches_from_cache(cache_root, cache_manifest, epoch)
                for b_feats, b_labels, _ in batches:
                    optimizer.zero_grad()
                    by = b_labels.unsqueeze(1).to(device)
                    step_loss = 0.0
                    for s in range(0, len(b_feats), micro_batch_size):
                        mf = b_feats[s : s + micro_batch_size].to(device)
                        my = by[s : s + micro_batch_size]
                        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                            logits = head(mf)
                            loss_mb, _ = criterion(logits.float(), my, tau_t=tau_t)
                            loss_scaled = loss_mb * (len(mf) / float(len(b_feats)))
                        scaler.scale(loss_scaled).backward()
                        step_loss += float(loss_scaled.detach().cpu().item())
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    ep_loss += step_loss
                n_batches = len(batches)
            else:
                batch_list = list(sampler)
                for indices in batch_list:
                    optimizer.zero_grad()
                    augs = [sample_augmentation_params(aug_rng) for _ in indices]
                    rels = [train_paths[i] for i in indices]
                    lats = [train_lats[i] for i in indices]
                    by = torch.tensor([train_labels[i] for i in indices], dtype=torch.float32).unsqueeze(1).to(device)
                    step_loss = 0.0
                    for s in range(0, len(indices), micro_batch_size):
                        bx = _preprocess_batch_online(
                            rsna_raw, rels[s : s + micro_batch_size], lats[s : s + micro_batch_size], augs[s : s + micro_batch_size], num_workers=num_workers
                        ).to(device)
                        my = by[s : s + micro_batch_size]
                        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                            with torch.no_grad():
                                feats = encoder(bx)
                            logits = head(feats)
                            loss_mb, _ = criterion(logits.float(), my, tau_t=tau_t)
                            loss_scaled = loss_mb * (len(bx) / float(len(indices)))
                        scaler.scale(loss_scaled).backward()
                        step_loss += float(loss_scaled.detach().cpu().item())
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    ep_loss += step_loss
                n_batches = len(batch_list)

            ep_loss /= max(n_batches, 1)
            ep_sec = time.time() - t_ep0
            history.append({"epoch": epoch, "train_loss": round(ep_loss, 5), "tau_t": round(tau_t, 5), "time_sec": round(ep_sec, 2)})
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

        # Single post-training evaluation on held-out val_calib to lock source operating thresholds
        head.eval()
        if use_cached_features:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                vc_probs = torch.sigmoid(head(vc_cache["features"].to(device))).float().squeeze(1).cpu()
            vc_pids = vc_cache["patient_ids"]
            vc_lbls = [int(v) for v in vc_cache["labels"].tolist()]
        else:
            vc_paths = vc_df["rel_path"].tolist()
            vc_lats = vc_df["laterality"].astype(str).tolist() if "laterality" in vc_df.columns else [""] * len(vc_df)
            probs_chunks = []
            with torch.no_grad():
                for s in range(0, len(vc_paths), micro_batch_size):
                    bx = _preprocess_batch_online(
                        rsna_raw, vc_paths[s : s + micro_batch_size], vc_lats[s : s + micro_batch_size], None, num_workers=num_workers
                    ).to(device)
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        probs_chunks.append(torch.sigmoid(head(encoder(bx))).float().squeeze(1).cpu())
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
            "gpu_name": gpu_name,
            "epoch_policy": epoch_policy,
            "use_cached_features": use_cached_features,
        }

        res_path = MOUNT_PATH / "models" / f"rsna_{baseline}_eval_results.json"
        res_path.write_text(json.dumps(eval_results, indent=2))
        RSNA_VOL.commit()

        return {
            "status": "SUCCESS",
            "baseline": baseline,
            "epochs": epochs,
            "checkpoint_path": str(final_ckpt),
            "operating_thresholds": eval_results,
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
    margin: float = 0.10,
    lambda_focal: float = 1.0,
    lambda_dismiss: float = 0.05,
    initial_tau: float = 0.05,
    num_workers: int = 4,
    gpu: str = "L40S",
    extract_cache: bool = False,
    use_cached_features: bool = True,
    resume: bool = False,
) -> None:
    gpu_norm = gpu.upper()
    if gpu_norm not in SUPPORTED_GPUS:
        raise ValueError(f"Unsupported GPU {gpu!r}. Choose from {SUPPORTED_GPUS}.")
    eff_steps = steps_per_epoch if steps_per_epoch > 0 else None

    if extract_cache:
        print(f"Launching augmentation-aware cache extraction on Modal ({gpu_norm})...")
        res = extract_augmentation_cache.with_options(gpu=gpu_norm).remote(
            epochs=epochs,
            batch_size=batch_size,
            n_pos=n_pos,
            steps_per_epoch=eff_steps,
            epoch_policy=epoch_policy,
            micro_batch_size=micro_batch_size,
            num_workers=num_workers,
        )
        print("Extraction finished:", res)
        return

    print(f"Launching {baseline.upper()} training on Modal (GPU={gpu_norm}, epochs={epochs}, cached={use_cached_features}, resume={resume})...")
    res = train_model.with_options(gpu=gpu_norm).remote(
        baseline=baseline,
        epochs=epochs,
        batch_size=batch_size,
        n_pos=n_pos,
        micro_batch_size=micro_batch_size,
        steps_per_epoch=eff_steps,
        epoch_policy=epoch_policy,
        lr=lr,
        margin=margin,
        lambda_focal=lambda_focal,
        lambda_dismiss=lambda_dismiss,
        initial_tau=initial_tau,
        num_workers=num_workers,
        use_cached_features=use_cached_features,
        resume=resume,
    )
    print("Training finished:", res)
