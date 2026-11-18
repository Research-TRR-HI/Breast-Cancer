"""Controlled Online-vs-Cached Equivalence Pilot (Part 6.4).

Compares:
    Approach A (Online):
        cropped image -> Stage 2 resize & norm -> stochastic augmentation(aug_params)
        -> 3ch + standardize -> frozen encoder -> classification head
    Approach B (Augmentation-Aware Cached):
        cropped image -> Stage 2 resize & norm -> IDENTICAL stochastic augmentation(aug_params)
        -> 3ch + standardize -> frozen encoder -> sharded FP32 feature cache
        -> classification head

Both pipelines share identical seeds, TrainingDrawSchedule, initial head weights,
Dropout RNG sequences, AdamW optimizer, TRUSTLoss, dynamic CP95 tau_t updates,
and post-training val_calib case-level evaluation.
"""

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.feature_cache import (
    compute_bytes_sha256,
    extract_or_resume_cache,
    load_epoch_batches_from_cache,
    verify_cache_integrity,
)
from src.data.preprocessing import (
    MAMMO_CLIP_MEAN,
    MAMMO_CLIP_STD,
    get_preprocessing_config_hash,
    prepare_tensor_from_cropped,
)
from src.data.rsna_dataset import (
    BalancedBatchSampler,
    BatchDrawRecord,
    build_epoch_draw_schedule,
)
from src.models.losses import TRUSTLoss
from src.models.trust_model import aggregate_case_labels, aggregate_case_scores
from src.training.calibration import (
    compute_dynamic_training_threshold,
    compute_target_recall_threshold,
    evaluate_dismissal_at_threshold,
)


def _make_trust_head(embed_dim: int, device: torch.device, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(0.3),
        nn.Linear(embed_dim, 1),
    ).to(device)


def _split_records_eval(
    cropped_list: List[np.ndarray],
    target_size: Tuple[int, int],
    encoder: nn.Module,
    head: Optional[nn.Module],
    device: torch.device,
    use_amp: bool,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run deterministic Stage-2 preprocessing and frozen encoder (and optional head) on eval split."""
    feats_chunks = []
    logits_chunks = []
    probs_chunks = []
    encoder.eval()
    if head is not None:
        head.eval()

    with torch.no_grad():
        for start in range(0, len(cropped_list), batch_size):
            end = min(start + batch_size, len(cropped_list))
            bx = torch.stack(
                [
                    prepare_tensor_from_cropped(
                        cropped_list[i],
                        target_size=target_size,
                        augment_params=None,
                        standardize=True,
                        mean=MAMMO_CLIP_MEAN,
                        std=MAMMO_CLIP_STD,
                    )
                    for i in range(start, end)
                ]
            ).to(device)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                feats = encoder(bx).float()
                feats_chunks.append(feats.cpu())
                if head is not None:
                    logits = head(feats)
                    probs = torch.sigmoid(logits)
                    logits_chunks.append(logits.cpu())
                    probs_chunks.append(probs.cpu())

    all_feats = torch.cat(feats_chunks, dim=0)
    if head is None:
        return all_feats, None, None
    return all_feats, torch.cat(logits_chunks, dim=0), torch.cat(probs_chunks, dim=0)


def run_online_vs_cached_parity_pilot(
    encoder: nn.Module,
    train_cropped: List[np.ndarray],
    train_calib_cropped: List[np.ndarray],
    val_calib_cropped: List[np.ndarray],
    train_df: pd.DataFrame,
    train_calib_df: pd.DataFrame,
    val_calib_df: pd.DataFrame,
    cache_root: Path,
    embed_dim: int = 2048,
    target_size: Tuple[int, int] = (1520, 912),
    epochs: int = 3,
    batch_size: int = 80,
    n_pos: int = 20,
    steps_per_epoch: Optional[int] = None,
    epoch_policy: str = "paper_independent",
    lr: float = 3e-5,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    margin: float = 0.10,
    lambda_focal: float = 1.0,
    lambda_dismiss: float = 0.05,
    initial_tau: float = 0.05,
    seed: int = 42,
    use_amp: bool = False,
    device: Optional[torch.device] = None,
    model_checkpoint_sha256: str = "pilot_encoder",
    micro_batch_size: int = 80,
    batches_per_shard: int = 10,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run controlled Approach A (Online) vs Approach B (Augmentation-Aware Cached) pilot."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    train_labels = train_df["cancer"].values.astype(np.int64)
    train_image_ids = train_df["image_id"].astype(str).tolist()
    tc_labels = train_calib_df["cancer"].values.astype(np.int64)
    vc_labels = val_calib_df["cancer"].values.astype(np.int64)
    vc_patient_ids = val_calib_df["patient_id"].astype(str).tolist()

    # Build identical TrainingDrawSchedule across all epochs
    sampler = BalancedBatchSampler(
        labels=train_labels,
        batch_size=batch_size,
        n_pos=n_pos,
        seed=seed,
        epoch_policy=epoch_policy,
        steps_per_epoch=steps_per_epoch,
    )
    aug_rng = np.random.default_rng(seed + 1000)
    schedules_by_epoch: Dict[int, List[BatchDrawRecord]] = {
        ep: build_epoch_draw_schedule(sampler, ep, aug_rng) for ep in range(1, epochs + 1)
    }

    preprocess_hash = get_preprocessing_config_hash(target_size=target_size)
    split_sha = compute_bytes_sha256(
        ",".join(train_image_ids + train_calib_df["image_id"].astype(str).tolist()).encode("utf-8")
    )

    # -------------------------------------------------------------------------
    # Build & verify augmentation-aware cache for Approach B
    # -------------------------------------------------------------------------
    def _extract_batch(rec: BatchDrawRecord) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        imgs = [
            prepare_tensor_from_cropped(
                train_cropped[idx],
                target_size=target_size,
                augment_params=aug,
                standardize=True,
                mean=MAMMO_CLIP_MEAN,
                std=MAMMO_CLIP_STD,
            )
            for idx, aug in zip(rec.indices, rec.aug_params)
        ]
        bx = torch.stack(imgs)
        feat_chunks = []
        with torch.no_grad():
            for s in range(0, len(bx), micro_batch_size):
                mb = bx[s : s + micro_batch_size].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    feat_chunks.append(encoder(mb).float().cpu())
        feats_fp32 = torch.cat(feat_chunks, dim=0)
        lbls = torch.tensor([train_labels[i] for i in rec.indices], dtype=torch.float32)
        img_ids = [train_image_ids[i] for i in rec.indices]
        return feats_fp32, lbls, img_ids

    def _eval_extract(split_name: str) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        if split_name == "train_calib":
            sdf, scrop = train_calib_df, train_calib_cropped
        elif split_name == "val_calib":
            sdf, scrop = val_calib_df, val_calib_cropped
        else:
            raise ValueError(f"Unsupported eval split: {split_name}")
        feats, _, _ = _split_records_eval(
            scrop, target_size, encoder, None, device, use_amp=use_amp, batch_size=micro_batch_size
        )
        return (
            feats,
            torch.tensor(sdf["cancer"].values, dtype=torch.float32),
            sdf["image_id"].astype(str).tolist(),
            sdf["patient_id"].astype(str).tolist(),
        )

    manifest = extract_or_resume_cache(
        cache_root=cache_root,
        schedules_by_epoch=schedules_by_epoch,
        extract_batch_fn=_extract_batch,
        eval_extract_fn=_eval_extract,
        model_checkpoint_sha256=model_checkpoint_sha256,
        preprocess_config_hash=preprocess_hash,
        split_manifest_sha256=split_sha,
        seed=seed,
        batch_size=batch_size,
        n_pos=n_pos,
        epoch_policy=epoch_policy,
        embed_dim=embed_dim,
        batches_per_shard=batches_per_shard,
        on_progress=on_progress,
    )
    verify_cache_integrity(
        cache_root=cache_root,
        expected_model_sha256=model_checkpoint_sha256,
        expected_preprocess_hash=preprocess_hash,
        expected_split_sha256=split_sha,
        expected_epochs=epochs,
        expected_steps_per_epoch=len(sampler),
        expected_schedules=schedules_by_epoch,
        verify_shard_checksums=True,
    )

    tc_cache = torch.load(
        str(cache_root / manifest["eval_splits"]["train_calib"]["file"]),
        map_location="cpu",
        weights_only=False,
    )
    vc_cache = torch.load(
        str(cache_root / manifest["eval_splits"]["val_calib"]["file"]),
        map_location="cpu",
        weights_only=False,
    )

    # -------------------------------------------------------------------------
    # Run Approach A (Online) and Approach B (Cached) with identical RNG & setup
    # -------------------------------------------------------------------------
    criterion = TRUSTLoss(
        baseline="trust",
        margin=margin,
        focal_gamma=2.0,
        focal_alpha=None,
        lambda_focal=lambda_focal,
        lambda_dismiss=lambda_dismiss,
    )

    def _run_pipeline(mode: str) -> Dict[str, Any]:
        head = _make_trust_head(embed_dim, device, seed=seed)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

        tau_t = float(initial_tau)
        tau_trajectory: List[float] = []
        step_losses: List[float] = []
        epoch_losses: List[float] = []
        captured_grads: Dict[str, Dict[str, torch.Tensor]] = {}
        first_batch_feats: Optional[torch.Tensor] = None

        global_step = 0
        total_steps = epochs * len(sampler)

        for epoch in range(1, epochs + 1):
            # TRUST Section 3.8: recalculate tau_t at beginning of each epoch (using epoch>=2 update; epoch 1 uses initial_tau=0.05)
            if epoch >= 2:
                head.eval()
                if mode == "online":
                    _, _, tc_probs = _split_records_eval(
                        train_calib_cropped, target_size, encoder, head, device, use_amp=use_amp, batch_size=micro_batch_size
                    )
                    tc_probs_np = tc_probs.squeeze(1).numpy()
                else:
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                        tc_probs_np = (
                            torch.sigmoid(head(tc_cache["features"].to(device)))
                            .float()
                            .squeeze(1)
                            .cpu()
                            .numpy()
                        )
                tau_t = compute_dynamic_training_threshold(
                    tc_probs_np, tc_labels, max_upper_bound=0.01, confidence=0.95
                )

            tau_trajectory.append(float(tau_t))
            head.train()
            ep_loss_sum = 0.0

            if mode == "online":
                iter_items = schedules_by_epoch[epoch]
            else:
                cached_batches = load_epoch_batches_from_cache(cache_root, manifest, epoch)
                iter_items = list(zip(schedules_by_epoch[epoch], cached_batches))

            for b_idx, item in enumerate(iter_items):
                global_step += 1
                # Lock per-step Dropout RNG identically across Online and Cached
                torch.manual_seed(seed * 100000 + global_step)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed * 100000 + global_step)

                optimizer.zero_grad()

                if mode == "online":
                    rec: BatchDrawRecord = item
                    bx = torch.stack(
                        [
                            prepare_tensor_from_cropped(
                                train_cropped[idx],
                                target_size=target_size,
                                augment_params=aug,
                                standardize=True,
                                mean=MAMMO_CLIP_MEAN,
                                std=MAMMO_CLIP_STD,
                            )
                            for idx, aug in zip(rec.indices, rec.aug_params)
                        ]
                    )
                    by = torch.tensor([train_labels[i] for i in rec.indices], dtype=torch.float32).unsqueeze(1).to(device)
                    accum_steps = max(math.ceil(len(bx) / micro_batch_size), 1)
                    step_loss_val = 0.0
                    feats_acc = []
                    for mb_start in range(0, len(bx), micro_batch_size):
                        mb_x = bx[mb_start : mb_start + micro_batch_size].to(device)
                        mb_y = by[mb_start : mb_start + micro_batch_size]
                        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                            with torch.no_grad():
                                f = encoder(mb_x)
                            if global_step == 1:
                                feats_acc.append(f.float().detach().cpu())
                            logits = head(f.float() if not use_amp else f)
                            loss_mb, _ = criterion(logits.float(), mb_y, tau_t=tau_t)
                            loss_scaled = loss_mb * (len(mb_x) / float(len(bx)))
                        scaler.scale(loss_scaled).backward()
                        step_loss_val += float(loss_scaled.detach().cpu().item())
                    if global_step == 1:
                        first_batch_feats = torch.cat(feats_acc, dim=0)
                else:
                    rec, (c_feats, c_labels, _c_idx) = item
                    if global_step == 1:
                        first_batch_feats = c_feats.clone()
                    by = c_labels.unsqueeze(1).to(device)
                    accum_steps = max(math.ceil(len(c_feats) / micro_batch_size), 1)
                    step_loss_val = 0.0
                    for mb_start in range(0, len(c_feats), micro_batch_size):
                        mb_f = c_feats[mb_start : mb_start + micro_batch_size].to(device)
                        mb_y = by[mb_start : mb_start + micro_batch_size]
                        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                            logits = head(mb_f.half() if (use_amp and device.type == "cuda") else mb_f)
                            loss_mb, _ = criterion(logits.float(), mb_y, tau_t=tau_t)
                            loss_scaled = loss_mb * (len(mb_f) / float(len(c_feats)))
                        scaler.scale(loss_scaled).backward()
                        step_loss_val += float(loss_scaled.detach().cpu().item())

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)

                if global_step in (1, total_steps):
                    tag = "step_1" if global_step == 1 else "final_step"
                    captured_grads[tag] = {
                        n: p.grad.detach().cpu().clone()
                        for n, p in head.named_parameters()
                        if p.grad is not None
                    }

                scaler.step(optimizer)
                scaler.update()

                step_losses.append(step_loss_val)
                ep_loss_sum += step_loss_val

            epoch_losses.append(ep_loss_sum / max(len(iter_items), 1))

        # Final val_calib evaluation (after final epoch)
        head.eval()
        if mode == "online":
            vc_feats_out, vc_logits, vc_probs = _split_records_eval(
                val_calib_cropped, target_size, encoder, head, device, use_amp=use_amp, batch_size=micro_batch_size
            )
        else:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                vf = vc_cache["features"].to(device)
                vc_logits = head(vf.half() if (use_amp and device.type == "cuda") else vf).float().cpu()
                vc_probs = torch.sigmoid(vc_logits)
                vc_feats_out = vc_cache["features"]

        case_scores_map = aggregate_case_scores(vc_probs.squeeze(1), vc_patient_ids)
        case_labels_map = aggregate_case_labels(vc_labels.tolist(), vc_patient_ids)
        sorted_pids = sorted(case_scores_map.keys())
        c_scores = np.array([case_scores_map[pid] for pid in sorted_pids], dtype=np.float64)
        c_labels = np.array([case_labels_map[pid] for pid in sorted_pids], dtype=np.int64)
        pos_c_scores = c_scores[c_labels == 1]

        tau_95 = compute_target_recall_threshold(pos_c_scores, target_recall=0.95)
        tau_98 = compute_target_recall_threshold(pos_c_scores, target_recall=0.98)
        eval_95 = evaluate_dismissal_at_threshold(c_scores, c_labels, tau_95, confidence=0.99, target_recall=0.95)
        eval_98 = evaluate_dismissal_at_threshold(c_scores, c_labels, tau_98, confidence=0.99, target_recall=0.98)

        return {
            "first_batch_feats": first_batch_feats,
            "val_feats": vc_feats_out,
            "step_losses": step_losses,
            "epoch_losses": epoch_losses,
            "captured_grads": captured_grads,
            "head_weights": {n: p.detach().cpu().clone() for n, p in head.named_parameters()},
            "val_logits": vc_logits.squeeze(1),
            "val_probs": vc_probs.squeeze(1),
            "case_scores": c_scores,
            "tau_trajectory": tau_trajectory,
            "tau_95": tau_95,
            "tau_98": tau_98,
            "eval_95": eval_95,
            "eval_98": eval_98,
        }

    res_a = _run_pipeline("online")
    res_b = _run_pipeline("cached")

    # Compute all 10 numerical parity comparisons
    feat_diff_train = float((res_a["first_batch_feats"] - res_b["first_batch_feats"]).abs().max().item())
    feat_diff_val = float((res_a["val_feats"] - res_b["val_feats"]).abs().max().item())
    max_feat_diff = max(feat_diff_train, feat_diff_val)

    max_step_loss_diff = float(np.max(np.abs(np.array(res_a["step_losses"]) - np.array(res_b["step_losses"]))))
    max_epoch_loss_diff = float(np.max(np.abs(np.array(res_a["epoch_losses"]) - np.array(res_b["epoch_losses"]))))

    grad_diffs = []
    for tag in ("step_1", "final_step"):
        for pname in res_a["captured_grads"][tag]:
            diff = float(
                (res_a["captured_grads"][tag][pname] - res_b["captured_grads"][tag][pname]).abs().max().item()
            )
            grad_diffs.append(diff)
    max_grad_diff = max(grad_diffs) if grad_diffs else 0.0

    max_weight_diff = max(
        float((res_a["head_weights"][n] - res_b["head_weights"][n]).abs().max().item())
        for n in res_a["head_weights"]
    )
    max_logit_diff = float((res_a["val_logits"] - res_b["val_logits"]).abs().max().item())
    max_prob_diff = float((res_a["val_probs"] - res_b["val_probs"]).abs().max().item())
    max_tau_diff = float(
        max(
            np.max(np.abs(np.array(res_a["tau_trajectory"]) - np.array(res_b["tau_trajectory"]))),
            abs(res_a["tau_95"] - res_b["tau_95"]),
            abs(res_a["tau_98"] - res_b["tau_98"]),
        )
    )

    agree_95 = float(np.mean((res_a["case_scores"] < res_a["tau_95"]) == (res_b["case_scores"] < res_b["tau_95"])))
    agree_98 = float(np.mean((res_a["case_scores"] < res_a["tau_98"]) == (res_b["case_scores"] < res_b["tau_98"])))
    near_mask_98 = np.abs(res_a["case_scores"] - res_a["tau_98"]) < 0.02
    near_agree_98 = (
        float(np.mean((res_a["case_scores"][near_mask_98] < res_a["tau_98"]) == (res_b["case_scores"][near_mask_98] < res_b["tau_98"])))
        if np.any(near_mask_98)
        else 1.0
    )

    tol = 1e-3 if use_amp else 1e-5
    passed = bool(
        max_feat_diff <= tol
        and max_step_loss_diff <= tol
        and max_epoch_loss_diff <= tol
        and max_grad_diff <= tol
        and max_weight_diff <= tol
        and max_logit_diff <= tol
        and max_prob_diff <= tol
        and max_tau_diff <= tol
        and agree_95 == 1.0
        and agree_98 == 1.0
        and near_agree_98 == 1.0
    )

    return {
        "status": "PASS" if passed else "FAIL",
        "precision": "amp" if use_amp else "fp32",
        "tolerance": tol,
        "epochs": epochs,
        "steps_per_epoch": len(sampler),
        "max_abs_feature_diff": max_feat_diff,
        "max_abs_step_loss_diff": max_step_loss_diff,
        "max_abs_epoch_loss_diff": max_epoch_loss_diff,
        "max_abs_grad_diff": max_grad_diff,
        "max_abs_weight_diff": max_weight_diff,
        "max_abs_logit_diff": max_logit_diff,
        "max_abs_prob_diff": max_prob_diff,
        "max_abs_tau_diff": max_tau_diff,
        "prediction_agreement_95": agree_95,
        "prediction_agreement_98": agree_98,
        "boundary_agreement_98": near_agree_98,
        "online_eval_95": res_a["eval_95"],
        "cached_eval_95": res_b["eval_95"],
        "online_eval_98": res_a["eval_98"],
        "cached_eval_98": res_b["eval_98"],
        "online_tau_trajectory": res_a["tau_trajectory"],
        "cached_tau_trajectory": res_b["tau_trajectory"],
    }
