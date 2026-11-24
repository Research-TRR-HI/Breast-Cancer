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

from src.evaluation.stage_c_eval import compute_sha256_file
from src.paper1.nyu_fourview_adapter import NYU_REPO_DEFAULT, NYU_VIEWS, _ensure_nyu_importable

NYU_FINETUNE_STRATEGIES = ("head_only", "partial", "full")
NYU_FINETUNED_MODEL_ID = "nyu_fourview_rsna_finetuned"
NYU_FINETUNED_CKPT_SCHEMA = "paper1_nyu_finetune_ckpt_v1"


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
    valid_stages = ("pilot", "train", "eval", "extract_head_features", "train_cached_head")
    if stage not in valid_stages:
        raise ValueError(f"Unknown --stage={stage!r}; expected one of {valid_stages}")
    if float(max_spend_usd) <= 0.0:
        raise ValueError(f"--max-spend-usd must be > 0, got {max_spend_usd}")
    if stage in ("train", "extract_head_features", "train_cached_head") and not approve_full_train:
        raise RuntimeError(f"BLOCKED: `--stage {stage}` requires `--approve-full-train`.")
    if stage == "eval" and not approve_eval:
        raise RuntimeError("BLOCKED: `--stage eval` requires `--approve-eval`.")


def set_nyu_backbone_eval_mode(model: nn.Module, strategy: str) -> None:
    """Ensure frozen NYU backbone BatchNorm & GaussianNoise layers stay in `.eval()` during fine-tuning."""
    model.train()
    if strategy == "full":
        return
    if strategy == "head_only":
        if hasattr(model, "four_view_resnet"):
            model.four_view_resnet.eval()
        if hasattr(model, "all_views_gaussian_noise_layer"):
            model.all_views_gaussian_noise_layer.eval()
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            params = list(mod.parameters())
            if params and all(not p.requires_grad for p in params):
                mod.eval()


@torch.no_grad()
def extract_nyu_backbone_features(
    model: nn.Module,
    tensor_batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen `(h_cc_in, h_mlo_in)` 512-d vectors in `.eval()` mode without GaussianNoise/BN drift."""
    was_training = model.training
    model.eval()
    try:
        x_clean = model.all_views_gaussian_noise_layer(tensor_batch) if hasattr(model, "all_views_gaussian_noise_layer") else tensor_batch
        feats = model.four_view_resnet(x_clean)
        pooled = model.all_views_avg_pool(feats)
        h_cc = torch.cat([pooled["L-CC"], pooled["R-CC"]], dim=1)
        h_mlo = torch.cat([pooled["L-MLO"], pooled["R-MLO"]], dim=1)
        return h_cc, h_mlo
    finally:
        if was_training:
            model.train()


def forward_nyu_head_from_features(
    model: nn.Module,
    h_cc_in: torch.Tensor,
    h_mlo_in: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run trainable SplitBreastModel head (`fc1_{cc,mlo}` + `output_layer_{cc,mlo}`) from cached 512-d features."""
    z_cc = F.relu(model.fc1_cc(h_cc_in))
    z_mlo = F.relu(model.fc1_mlo(h_mlo_in))
    return {
        "CC": model.output_layer_cc(z_cc),
        "MLO": model.output_layer_mlo(z_mlo),
    }


@torch.no_grad()
def score_nyu_head_from_tta_features(
    model: nn.Module,
    h_cc_tta: torch.Tensor,
    h_mlo_tta: torch.Tensor,
) -> List[Dict[str, float]]:
    """Score `(B, T, 512)` cached TTA feature tensors identically to `score_nyu_fourview_exam_batch`."""
    b_exams, t_epochs, feat_dim = h_cc_tta.shape
    out = forward_nyu_head_from_features(
        model,
        h_cc_tta.reshape(b_exams * t_epochs, feat_dim),
        h_mlo_tta.reshape(b_exams * t_epochs, feat_dim),
    )
    cc_probs = np.exp(out["CC"].float().cpu().numpy())[:, :, 1].reshape(b_exams, t_epochs, 4)
    mlo_probs = np.exp(out["MLO"].float().cpu().numpy())[:, :, 1].reshape(b_exams, t_epochs, 4)
    mean_4 = np.mean(0.5 * (cc_probs + mlo_probs), axis=1)
    results: List[Dict[str, float]] = []
    for row in mean_4:
        lb, rb, lm, rm = [float(x) for x in row]
        results.append(
            {
                "left_benign": lb,
                "right_benign": rb,
                "left_malignant": lm,
                "right_malignant": rm,
                "exam_score": max(lm, rm),
            }
        )
    return results


def validate_nyu_eval_cli_args(
    *,
    split: str,
    batch_exams: int,
    num_shards: int,
    num_tta_epochs: int,
    checkpoint_path: str,
    approve_eval: bool,
    max_spend_usd: float,
) -> None:
    """Validate CLI args for NYU fine-tuned evaluation-only mode."""
    validate_nyu_finetune_cli_guard(
        stage="eval",
        approve_full_train=False,
        approve_eval=approve_eval,
        max_spend_usd=max_spend_usd,
    )
    if split not in ("rsna_val_calib", "nlbs_4view"):
        raise ValueError(f"Unsupported NYU eval --split={split!r}; expected 'rsna_val_calib' or 'nlbs_4view'")
    if int(batch_exams) < 1:
        raise ValueError(f"--batch-exams must be >= 1, got {batch_exams}")
    if int(num_shards) < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if int(num_tta_epochs) != 10:
        raise ValueError(f"Official NYU evaluation protocol requires --num-tta-epochs=10, got {num_tta_epochs}")
    if not str(checkpoint_path).strip():
        raise ValueError("--checkpoint-path must be non-empty for --stage eval")


def load_nyu_finetuned_model_for_eval(
    checkpoint_path: Path,
    *,
    expected_sha256: Optional[str] = None,
    device: Optional[torch.device] = None,
    nyu_repo_dir: Path = NYU_REPO_DEFAULT,
    model_override: Optional[nn.Module] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Load fine-tuned NYU SplitBreastModel checkpoint with SHA-256 verification and zero-grad eval guarantee."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing fine-tuned NYU checkpoint: {ckpt_path}")
    actual_sha = compute_sha256_file(ckpt_path).lower()
    if expected_sha256 and actual_sha != str(expected_sha256).strip().lower():
        raise RuntimeError(
            f"Fine-tuned NYU checkpoint SHA-256 mismatch: got {actual_sha}, expected {expected_sha256}"
        )

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if ckpt.get("schema_version") != NYU_FINETUNED_CKPT_SCHEMA:
        raise RuntimeError(f"Unsupported fine-tuned NYU checkpoint schema: {ckpt.get('schema_version')}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_override is not None:
        model = model_override
        mods: Dict[str, Any] = {}
    else:
        mods = _ensure_nyu_importable(nyu_repo_dir)
        model = mods["models"].SplitBreastModel(input_channels=1)

    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = {
        "model_id": NYU_FINETUNED_MODEL_ID,
        "checkpoint_file": ckpt_path.name,
        "checkpoint_sha256": actual_sha,
        "strategy": ckpt.get("strategy"),
        "epoch": int(ckpt["epoch"]),
        "best_epoch": int(ckpt["best_epoch"]),
        "best_train_calib_auroc": float(ckpt["best_auroc"]),
        "seed": int(ckpt.get("seed", 42)),
    }
    return model, mods, meta


def train_nyu_head_from_cached_features(
    *,
    model: nn.Module,
    train_h_cc: torch.Tensor,
    train_h_mlo: torch.Tensor,
    train_lm: np.ndarray,
    train_rm: np.ndarray,
    train_exam_cancer: np.ndarray,
    tc_h_cc: torch.Tensor,
    tc_h_mlo: torch.Tensor,
    tc_exam_cancer: np.ndarray,
    out_dir: Path,
    epochs: int = 8,
    batch_exams: int = 16,
    steps_per_epoch: int = 400,
    lr: float = 1e-4,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train `head_only` NYU SplitBreastModel layers from cached 512-d CC/MLO features."""
    from src.data.rsna_dataset import BalancedBatchSampler
    from src.evaluation.stage_c_eval import compute_auroc

    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    param_groups = configure_nyu_trainable_params(model, strategy="head_only", base_lr=lr)
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    sampler = BalancedBatchSampler(
        labels=train_exam_cancer,
        batch_size=batch_exams,
        n_pos=max(1, batch_exams // 4),
        seed=seed,
        epoch_policy="paper_independent",
        steps_per_epoch=steps_per_epoch,
    )
    draw_rng = np.random.default_rng(seed + 77)
    n_draws = int(train_h_cc.shape[1])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_pt = out_dir / f"nyu_rsna_head_only_cached_seed{seed}_best_train_calib.pt"
    latest_pt = out_dir / f"nyu_rsna_head_only_cached_seed{seed}_latest.pt"

    best_auroc, best_epoch, history = -1.0, 0, []
    for ep in range(1, epochs + 1):
        set_nyu_backbone_eval_mode(model, "head_only")
        ep_losses: List[float] = []
        for indices in sampler:
            d_idx = draw_rng.integers(0, n_draws, size=len(indices))
            idx_t = torch.tensor(indices, dtype=torch.long)
            draw_t = torch.tensor(d_idx, dtype=torch.long)
            bh_cc = train_h_cc[idx_t, draw_t].to(device)
            bh_mlo = train_h_mlo[idx_t, draw_t].to(device)
            lm_t = torch.tensor([train_lm[i] for i in indices], dtype=torch.long, device=device)
            rm_t = torch.tensor([train_rm[i] for i in indices], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            out = forward_nyu_head_from_features(model, bh_cc, bh_mlo)
            loss, ld = compute_nyu_rsna_loss(out, lm_t, rm_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            ep_losses.append(ld["loss_total"])
        sched.step()

        model.eval()
        tc_scores: List[float] = []
        for s0 in range(0, len(tc_h_cc), 64):
            res_list = score_nyu_head_from_tta_features(
                model,
                tc_h_cc[s0 : s0 + 64].to(device),
                tc_h_mlo[s0 : s0 + 64].to(device),
            )
            tc_scores.extend(r["exam_score"] for r in res_list)
        auroc = float(compute_auroc(tc_exam_cancer, np.asarray(tc_scores)))
        if auroc > best_auroc:
            best_auroc, best_epoch = auroc, ep
            save_nyu_finetune_checkpoint(
                best_pt, model, opt, sched,
                epoch=ep, best_auroc=best_auroc, best_epoch=best_epoch,
                history=history, sampler=sampler, aug_seed_base=seed, strategy="head_only", seed=seed,
            )
        history.append({"epoch": ep, "mean_loss": float(np.mean(ep_losses)), "train_calib_auroc": auroc})
        save_nyu_finetune_checkpoint(
            latest_pt, model, opt, sched,
            epoch=ep, best_auroc=best_auroc, best_epoch=best_epoch,
            history=history, sampler=sampler, aug_seed_base=seed, strategy="head_only", seed=seed,
        )

    return {
        "best_epoch": best_epoch,
        "best_train_calib_auroc": round(best_auroc, 6),
        "best_checkpoint_path": str(best_pt),
        "best_checkpoint_sha256": compute_sha256_file(best_pt),
        "history": history,
    }
