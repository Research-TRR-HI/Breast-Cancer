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
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
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


def parse_nyu_batch_sizes(raw: str) -> List[int]:
    """Parse comma-separated positive exam batch sizes."""
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"Expected positive comma-separated exam batch sizes, got {raw!r}")
    return vals


def run_progressive_nyu_batch_benchmark(
    model: nn.Module,
    exam_batch_loader: Callable[[int, int], Dict[str, np.ndarray]],
    device: torch.device,
    *,
    batch_sizes: Sequence[int] = (2, 4, 8, 16),
    mode: str = "extract",
    warmup_steps: int = 1,
    bench_steps: int = 3,
    draws_per_exam: int = 1,
    stop_on_throughput_drop: bool = True,
    drop_tolerance: float = 0.90,
) -> Dict[str, Any]:
    """Progressively benchmark NYU exam batch sizes (`2, 4, 8, 16`) for extraction/evaluation."""
    if mode not in ("extract", "eval"):
        raise ValueError(f"Unsupported NYU benchmark mode={mode!r}; expected 'extract' or 'eval'")
    sorted_bs = sorted({int(b) for b in batch_sizes if int(b) >= 1})
    model = model.to(device)
    model.eval()
    results: List[Dict[str, Any]] = []
    best_eps = -1.0

    with torch.no_grad():
        for bs_idx, bs in enumerate(sorted_bs):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            data_load_sec = 0.0
            h2d_sec = 0.0
            fwd_sec = 0.0
            completed = 0
            status = "ok"
            stopped_reason = ""
            try:
                total_steps = int(warmup_steps) + int(bench_steps)
                for step in range(total_steps):
                    t_d0 = time.perf_counter()
                    host_dict = exam_batch_loader(bs, step)
                    t_d1 = time.perf_counter()

                    host_t = {}
                    for v in NYU_VIEWS:
                        arr = torch.from_numpy(np.ascontiguousarray(host_dict[v], dtype=np.float32))
                        host_t[v] = arr.pin_memory() if device.type == "cuda" else arr
                    t_h0 = time.perf_counter()
                    tbatch = {v: host_t[v].to(device, non_blocking=True) for v in NYU_VIEWS}
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t_h1 = time.perf_counter()

                    t_f0 = time.perf_counter()
                    if mode == "extract":
                        _ = extract_nyu_backbone_features(model, tbatch)
                    else:
                        _ = model(tbatch)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t_f1 = time.perf_counter()

                    if step >= warmup_steps:
                        data_load_sec += t_d1 - t_d0
                        h2d_sec += t_h1 - t_h0
                        fwd_sec += t_f1 - t_f0
                        completed += 1
            except Exception as exc:
                msg = str(exc).lower()
                if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg:
                    status = "oom"
                    stopped_reason = str(exc)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

            total_sec = max(data_load_sec + h2d_sec + fwd_sec, 1e-9)
            n_exams = completed * bs
            exams_per_sec = float(n_exams / total_sec) if completed > 0 else 0.0
            imgs_per_sec = float((n_exams * 4 * int(draws_per_exam)) / total_sec) if completed > 0 else 0.0
            peak_alloc = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
            peak_res = float(torch.cuda.max_memory_reserved(device) / (1024**3)) if device.type == "cuda" else 0.0
            row = {
                "mode": mode,
                "batch_exams": bs,
                "draws_per_exam": int(draws_per_exam),
                "status": status,
                "stopped_reason": stopped_reason,
                "exams_per_sec": round(exams_per_sec, 4),
                "imgs_per_sec": round(imgs_per_sec, 3),
                "peak_allocated_vram_gb": round(peak_alloc, 4),
                "peak_reserved_vram_gb": round(peak_res, 4),
                "data_load_sec": round(data_load_sec, 4),
                "h2d_sec": round(h2d_sec, 4),
                "fwd_sec": round(fwd_sec, 4),
                "pin_memory": device.type == "cuda",
                "non_blocking": True,
            }
            if status == "ok":
                if exams_per_sec > best_eps:
                    best_eps = exams_per_sec
                elif stop_on_throughput_drop and bs_idx >= 1 and exams_per_sec < best_eps * float(drop_tolerance):
                    row["stopped_reason"] = f"throughput_drop ({exams_per_sec:.3f} < {best_eps * drop_tolerance:.3f})"
                    results.append(row)
                    break
            results.append(row)
            if status == "oom":
                break

    valid = [r for r in results if r["status"] == "ok"]
    best_row = max(valid, key=lambda r: r["exams_per_sec"]) if valid else None
    return {"mode": mode, "sweep_results": results, "recommended": best_row}


def verify_nyu_multidraw_feature_equivalence(
    model: nn.Module,
    exam_draws: Dict[str, np.ndarray],
    *,
    num_draws: int,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """Verify exact equivalence between multi-draw cached features and online SplitBreastModel forward.

    Mathematical basis:
    1. `AllViewsGaussianNoise(0.01)` only adds noise when `self.training == True`; both
       frozen-backbone feature extraction (`model.eval()`) and `head_only` training
       (`set_nyu_backbone_eval_mode(model, "head_only")`) hold it in `.eval()` (identity map).
    2. All `BatchNorm2d` layers reside inside `four_view_resnet.{cc,mlo}` and stay in
       `.eval()` mode, freezing `running_mean` and `running_var`.
    3. Random crops/flips (`augment_and_normalize_image`) happen upstream of the backbone,
       so evaluating `head(backbone(aug(x, seed_k)))` over `k=0..K-1` draws is identical
       to precomputing `h_k = backbone(aug(x, seed_k))` and calling `head(h_k)`.
    """
    device = next(model.parameters()).device
    set_nyu_backbone_eval_mode(model, "head_only")

    bn_means_before = {
        name: mod.running_mean.detach().clone()
        for name, mod in model.named_modules()
        if isinstance(mod, nn.BatchNorm2d)
    }

    t_in = {
        v: torch.from_numpy(np.ascontiguousarray(exam_draws[v], dtype=np.float32)).to(device)
        for v in NYU_VIEWS
    }
    n_total = int(t_in["L-CC"].shape[0])
    if n_total % int(num_draws) != 0:
        raise ValueError(f"Expected first dim ({n_total}) divisible by num_draws ({num_draws})")
    n_exams = n_total // int(num_draws)

    # Path A: Multi-draw cached features -> score_nyu_head_from_tta_features
    h_cc_multi, h_mlo_multi = extract_nyu_backbone_features(model, t_in)
    h_cc_shaped = h_cc_multi.reshape(n_exams, num_draws, -1)
    h_mlo_shaped = h_mlo_multi.reshape(n_exams, num_draws, -1)
    cached_multi_scores = score_nyu_head_from_tta_features(model, h_cc_shaped, h_mlo_shaped)

    # Path B: Single-draw feature extraction draw-by-draw -> averaged post-softmax
    h_cc_single = torch.empty_like(h_cc_shaped)
    h_mlo_single = torch.empty_like(h_mlo_shaped)
    with torch.no_grad():
        for ex_i in range(n_exams):
            for k in range(num_draws):
                idx = ex_i * num_draws + k
                one = {v: t_in[v][idx : idx + 1] for v in NYU_VIEWS}
                hc, hm = extract_nyu_backbone_features(model, one)
                h_cc_single[ex_i, k] = hc[0]
                h_mlo_single[ex_i, k] = hm[0]
    cached_single_scores = score_nyu_head_from_tta_features(model, h_cc_single, h_mlo_single)

    # Path C: Full end-to-end SplitBreastModel forward in .eval()
    model.eval()
    with torch.no_grad():
        out_full = model(t_in)
        probs_cc = torch.exp(out_full["CC"].float()).reshape(n_exams, num_draws, 4, 2)[:, :, :, 1]
        probs_mlo = torch.exp(out_full["MLO"].float()).reshape(n_exams, num_draws, 4, 2)[:, :, :, 1]
        full_mean_4 = (0.5 * (probs_cc + probs_mlo)).mean(dim=1).cpu().numpy()

    max_feat_diff = max(
        float(torch.max(torch.abs(h_cc_shaped - h_cc_single)).item()),
        float(torch.max(torch.abs(h_mlo_shaped - h_mlo_single)).item()),
    )
    max_score_diff_vs_full = max(
        abs(cached_multi_scores[i]["exam_score"] - max(float(full_mean_4[i, 2]), float(full_mean_4[i, 3])))
        for i in range(n_exams)
    )
    max_score_diff_vs_single = max(
        abs(cached_multi_scores[i]["exam_score"] - cached_single_scores[i]["exam_score"])
        for i in range(n_exams)
    )

    bn_unchanged = all(
        torch.equal(bn_means_before[name], mod.running_mean)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.BatchNorm2d)
    )
    gn_active_flags = [
        bool(mod.training)
        for mod in model.modules()
        if "gaussiannoise" in type(mod).__name__.lower()
    ]
    if max_feat_diff > atol or max_score_diff_vs_full > atol or max_score_diff_vs_single > atol or not bn_unchanged:
        raise AssertionError(
            f"NYU multi-draw equivalence check failed: max_feat_diff={max_feat_diff}, "
            f"max_score_diff_vs_full={max_score_diff_vs_full}, bn_unchanged={bn_unchanged}"
        )
    return {
        "n_exams": n_exams,
        "num_draws": int(num_draws),
        "max_feature_abs_diff": max_feat_diff,
        "max_score_abs_diff_vs_full_eval": max_score_diff_vs_full,
        "max_score_abs_diff_vs_single_draw": max_score_diff_vs_single,
        "batchnorm_running_stats_frozen": bn_unchanged,
        "gaussian_noise_modules_training_flags": gn_active_flags,
    }

