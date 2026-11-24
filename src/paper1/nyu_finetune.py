"""Paper 1 Supplementary Exploratory — NYU Four-View (SplitBreastModel) RSNA Fine-Tuning.

Implements scientifically defensible supervised fine-tuning of `sample_image_model.p`
on RSNA `train` (with `train_calib` used strictly for checkpoint selection, and
`val_calib` / NLBS strictly held out).

Key scientific & numerical safeguards:
1. RSNA Screening Mammography has `cancer` (0/1) labels only — NO benign biopsy
   labels. `SplitBreastModel` outputs `log_softmax` of shape `(B, 4, 2)` for
   `[left_benign, right_benign, left_malignant, right_malignant]`. Loss is computed
   STRICTLY on indices `[2, 3]` (`left_malignant`, `right_malignant`), never
   treating `cancer == 0` as `benign == 0`.
2. Deterministic label-blind 4-view exam selection (`min(int(image_id))` per
   `(patient_id, view_slot)`) matches `rsna_val_calib_4view_selection.csv`.
3. `SplitBreastModel` runs in FP32 by default (matching Paper 1 Step 2 parity).
"""

from pathlib import Path
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.paper1.nyu_fourview_adapter import NYU_VIEWS

NYU_FINETUNE_STRATEGIES = ("head_only", "partial", "full")


def select_rsna_4view_split_exams(
    splits_df: pd.DataFrame,
    split_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically select 1 image per standard view per RSNA patient in `split_name`.

    Returns:
        image_df: 4 rows per patient sorted by `(patient_id, view_slot)`.
        exam_df: 1 row per patient with `patient_id`, `left_malignant`, `right_malignant`, `exam_cancer`.
    """
    sub = splits_df[splits_df["split"] == split_name].copy()
    if sub.empty:
        raise ValueError(f"No rows found for split={split_name!r}")
    sub["view_slot"] = sub["laterality"].astype(str).str.strip().str.upper() + "-" + sub["view"].astype(str).str.strip().str.upper()
    sub = sub[sub["view_slot"].isin(NYU_VIEWS)].copy()
    sub["_img_id_num"] = sub["image_id"].astype(int)
    sub = sub.sort_values(["patient_id", "view_slot", "_img_id_num"], kind="mergesort")
    picked = sub.drop_duplicates(subset=["patient_id", "view_slot"], keep="first").copy()

    counts = picked.groupby("patient_id")["view_slot"].nunique()
    complete_pids = counts[counts == len(NYU_VIEWS)].index
    picked = picked[picked["patient_id"].isin(complete_pids)].drop(columns=["_img_id_num"]).reset_index(drop=True)

    view_order = {v: i for i, v in enumerate(NYU_VIEWS)}
    picked["_v_ord"] = picked["view_slot"].map(view_order)
    picked = picked.sort_values(["patient_id", "_v_ord"], kind="mergesort").drop(columns=["_v_ord"]).reset_index(drop=True)

    # Derive per-breast malignant labels from all images of that patient & laterality
    all_sub = splits_df[splits_df["split"] == split_name]
    lat_max = (
        all_sub.groupby(["patient_id", "laterality"])["cancer"]
        .max()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "L" not in lat_max.columns:
        lat_max["L"] = 0
    if "R" not in lat_max.columns:
        lat_max["R"] = 0
    exam_df = pd.DataFrame(
        {
            "patient_id": lat_max["patient_id"].astype(str),
            "left_malignant": lat_max["L"].astype(int),
            "right_malignant": lat_max["R"].astype(int),
        }
    )
    exam_df["exam_cancer"] = np.maximum(exam_df["left_malignant"], exam_df["right_malignant"]).astype(int)
    valid_pids = set(picked["patient_id"].astype(str))
    exam_df = exam_df[exam_df["patient_id"].isin(valid_pids)].sort_values("patient_id").reset_index(drop=True)
    return picked, exam_df


def configure_nyu_trainable_params(
    model: nn.Module,
    strategy: str = "head_only",
    base_lr: float = 1e-4,
    backbone_lr_mult: float = 0.1,
) -> List[Dict[str, Any]]:
    """Freeze/unfreeze SplitBreastModel parameters by strategy and return AdamW param groups."""
    if strategy not in NYU_FINETUNE_STRATEGIES:
        raise ValueError(f"Unsupported NYU strategy={strategy!r}; expected {NYU_FINETUNE_STRATEGIES}")

    for p in model.parameters():
        p.requires_grad = False

    head_prefixes = ("fc1_cc.", "fc1_mlo.", "output_layer_cc.", "output_layer_mlo.")
    partial_prefixes = (
        "four_view_resnet.cc.layer_list.4.",
        "four_view_resnet.cc.final_bn.",
        "four_view_resnet.mlo.layer_list.4.",
        "four_view_resnet.mlo.final_bn.",
    )

    head_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if strategy == "head_only":
            if name.startswith(head_prefixes):
                p.requires_grad = True
                head_params.append(p)
        elif strategy == "partial":
            if name.startswith(head_prefixes):
                p.requires_grad = True
                head_params.append(p)
            elif name.startswith(partial_prefixes):
                p.requires_grad = True
                backbone_params.append(p)
        elif strategy == "full":
            p.requires_grad = True
            if name.startswith(head_prefixes):
                head_params.append(p)
            else:
                backbone_params.append(p)

    groups: List[Dict[str, Any]] = []
    if head_params:
        groups.append({"params": head_params, "lr": float(base_lr)})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(base_lr) * float(backbone_lr_mult)})
    if not groups:
        raise RuntimeError(f"No trainable parameters selected for NYU strategy={strategy!r}")
    return groups


def compute_nyu_rsna_loss(
    out_dict: Dict[str, torch.Tensor],
    left_malignant: torch.Tensor,
    right_malignant: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute NLL loss strictly on malignant indices [2, 3] of CC and MLO log-softmax outputs.

    Benign indices [0, 1] are NEVER supervised because RSNA lacks benign labels.
    """
    lm = left_malignant.view(-1).long()
    rm = right_malignant.view(-1).long()

    cc_logp = out_dict["CC"].float()
    mlo_logp = out_dict["MLO"].float()

    loss_cc = 0.5 * (F.nll_loss(cc_logp[:, 2, :], lm) + F.nll_loss(cc_logp[:, 3, :], rm))
    loss_mlo = 0.5 * (F.nll_loss(mlo_logp[:, 2, :], lm) + F.nll_loss(mlo_logp[:, 3, :], rm))
    total = 0.5 * (loss_cc + loss_mlo)
    return total, {
        "loss_total": float(total.detach().item()),
        "loss_cc": float(loss_cc.detach().item()),
        "loss_mlo": float(loss_mlo.detach().item()),
    }


def save_nyu_finetune_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    best_auroc: float,
    best_epoch: int,
    history: List[Dict[str, Any]],
    sampler: Any,
    aug_seed_base: int,
    strategy: str,
    seed: int,
) -> Path:
    """Atomically save NYU fine-tuning state including RNGs and sampler state."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".pt.tmp")
    payload = {
        "schema_version": "paper1_nyu_finetune_ckpt_v1",
        "strategy": str(strategy),
        "epoch": int(epoch),
        "best_auroc": float(best_auroc),
        "best_epoch": int(best_epoch),
        "seed": int(seed),
        "aug_seed_base": int(aug_seed_base),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": list(history),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler_rng_state": sampler.get_state(),
    }
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


def load_nyu_finetune_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: Any,
    device: torch.device,
) -> Tuple[int, float, int, List[Dict[str, Any]]]:
    """Restore NYU fine-tuning checkpoint and RNG states."""
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if ckpt.get("schema_version") != "paper1_nyu_finetune_ckpt_v1":
        raise RuntimeError(f"Unsupported NYU checkpoint schema: {ckpt.get('schema_version')}")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    random.setstate(ckpt["python_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    torch.set_rng_state(ckpt["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["torch_cuda_rng_states"]])
    sampler.set_state(ckpt["sampler_rng_state"])
    return (
        int(ckpt["epoch"]),
        float(ckpt["best_auroc"]),
        int(ckpt["best_epoch"]),
        list(ckpt.get("history", [])),
    )


def validate_nyu_finetune_cli_guard(
    stage: str,
    approve_full_train: bool,
    approve_eval: bool,
    max_spend_usd: float,
) -> None:
    """Block unapproved full GPU training or evaluation runs."""
    if stage not in ("pilot", "train", "eval"):
        raise ValueError(f"Unknown --stage={stage!r}")
    if float(max_spend_usd) <= 0.0:
        raise ValueError(f"--max-spend-usd must be > 0, got {max_spend_usd}")
    if stage == "train" and not approve_full_train:
        raise RuntimeError("BLOCKED: `--stage train` requires `--approve-full-train`.")
    if stage == "eval" and not approve_eval:
        raise RuntimeError("BLOCKED: `--stage eval` requires `--approve-eval`.")
