"""Paper 1 Supplementary Exploratory — GMIC (nyukat/GMIC) RSNA Fine-Tuning Module.

Implements scientifically defensible supervised fine-tuning of `sample_model_1.p`
on RSNA `train` (with `train_calib` used strictly for checkpoint selection, and
`val_calib` / NLBS strictly held out).

Key scientific & numerical safeguards:
1. RSNA Screening Mammography has `cancer` (0/1) labels only — NO benign biopsy
   labels. Therefore, loss is computed STRICTLY on the malignant output channel
   (`[:, 1]`), never treating `cancer == 0` as `benign == 0`.
2. `GMIC.forward()` applies `torch.sigmoid()` internally to `y_global`, `y_local`,
   and `y_fusion`. BCE loss is therefore computed in FP32 outside `torch.amp.autocast`
   with probability clamping `[1e-6, 1 - 1e-6]`.
3. Supports three fine-tuning regimes: `head_only`, `partial`, and `full`
   (differential backbone LR).
"""

from pathlib import Path
import math
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.stage_c_eval import compute_sha256_file
from src.paper1.gmic_adapter import GMIC_PERCENT_T_MAP, GMIC_REPO_DEFAULT, _ensure_gmic_importable

GMIC_FINETUNE_STRATEGIES = ("head_only", "partial", "full")
GMIC_FINETUNED_MODEL_ID = "gmic_rsna_finetuned"
GMIC_FINETUNED_CKPT_SCHEMA = "paper1_gmic_finetune_ckpt_v1"


def configure_gmic_trainable_params(
    model: nn.Module,
    strategy: str = "partial",
    base_lr: float = 1e-4,
    backbone_lr_mult: float = 0.1,
) -> List[Dict[str, Any]]:
    """Freeze/unfreeze GMIC parameters by strategy and return AdamW param groups."""
    if strategy not in GMIC_FINETUNE_STRATEGIES:
        raise ValueError(f"Unsupported GMIC strategy={strategy!r}; expected {GMIC_FINETUNE_STRATEGIES}")

    for p in model.parameters():
        p.requires_grad = False

    head_prefixes = (
        "left_postprocess_net.",
        "mil_attn_V.",
        "mil_attn_U.",
        "mil_attn_w.",
        "classifier_linear.",
        "fusion_dnn.",
    )
    partial_backbone_prefixes = (
        "ds_net.layer_list.4.",
        "ds_net.final_bn.",
        "dn_resnet.layer4.",
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
            elif name.startswith(partial_backbone_prefixes):
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
        raise RuntimeError(f"No trainable parameters selected for GMIC strategy={strategy!r}")
    return groups


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def compute_gmic_rsna_loss(
    model: nn.Module,
    y_fusion: torch.Tensor,
    targets: torch.Tensor,
    *,
    saliency_l1_weight: float = 1e-5,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute FP32 malignant-only GMIC multi-branch BCE + saliency L1 loss.

    Args:
        model: GMIC model after forward pass (exposes `y_global`, `y_local`, `saliency_map`).
        y_fusion: Output tensor of shape `(B, 2)` with probabilities `[benign, malignant]`.
        targets: Binary RSNA cancer labels of shape `(B,)` or `(B, 1)` in `{0, 1}`.
        saliency_l1_weight: L1 sparsity weight on malignant saliency map `[:, 1]`.
    """
    y_t = targets.view(-1).float()
    p_fus = y_fusion[:, 1].float().clamp(eps, 1.0 - eps)
    p_glb = model.y_global[:, 1].float().clamp(eps, 1.0 - eps)
    p_loc = model.y_local[:, 1].float().clamp(eps, 1.0 - eps)

    loss_fus = F.binary_cross_entropy(p_fus, y_t)
    loss_glb = F.binary_cross_entropy(p_glb, y_t)
    loss_loc = F.binary_cross_entropy(p_loc, y_t)
    loss_sal = model.saliency_map[:, 1].float().abs().mean() * float(saliency_l1_weight)

    total = loss_fus + loss_glb + loss_loc + loss_sal
    metrics = {
        "loss_total": float(total.detach().item()),
        "loss_fusion": float(loss_fus.detach().item()),
        "loss_global": float(loss_glb.detach().item()),
        "loss_local": float(loss_loc.detach().item()),
        "loss_saliency_l1": float(loss_sal.detach().item()),
    }
    return total, metrics


def gmic_amp_train_step(
    model: nn.Module,
    bx: torch.Tensor,
    by: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    saliency_l1_weight: float = 1e-5,
    max_grad_norm: float = 5.0,
    amp_enabled: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
) -> Tuple[Dict[str, float], float]:
    """Execute one GMIC training step with FP32 probability BCE and unscaled grad clipping."""
    optimizer.zero_grad(set_to_none=True)
    device_type = "cuda" if bx.device.type == "cuda" else "cpu"
    use_autocast = bool(amp_enabled and device_type == "cuda")
    resolved_dtype = amp_dtype if amp_dtype is not None else (torch.float16 if use_autocast else torch.float32)
    with torch.amp.autocast(device_type, dtype=resolved_dtype, enabled=use_autocast):
        y_fusion = model(bx)
    # Compute BCE outside autocast because GMIC outputs post-sigmoid probabilities
    loss, loss_dict = compute_gmic_rsna_loss(
        model=model,
        y_fusion=y_fusion,
        targets=by,
        saliency_l1_weight=saliency_l1_weight,
    )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm))
    scaler.step(optimizer)
    scaler.update()
    return loss_dict, grad_norm


def save_gmic_finetune_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    best_auroc: float,
    best_epoch: int,
    history: List[Dict[str, Any]],
    sampler: Any,
    aug_rng: np.random.Generator,
    strategy: str,
    seed: int,
) -> Path:
    """Atomically save GMIC fine-tuning state including RNGs and sampler state."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".pt.tmp")
    payload = {
        "schema_version": "paper1_gmic_finetune_ckpt_v1",
        "strategy": str(strategy),
        "epoch": int(epoch),
        "best_auroc": float(best_auroc),
        "best_epoch": int(best_epoch),
        "seed": int(seed),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": list(history),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler_rng_state": sampler.get_state(),
        "aug_rng_state": aug_rng.bit_generator.state,
    }
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


def load_gmic_finetune_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    sampler: Any,
    aug_rng: np.random.Generator,
    device: torch.device,
) -> Tuple[int, float, int, List[Dict[str, Any]]]:
    """Restore GMIC fine-tuning checkpoint and all RNG states."""
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if ckpt.get("schema_version") != "paper1_gmic_finetune_ckpt_v1":
        raise RuntimeError(f"Unsupported GMIC checkpoint schema: {ckpt.get('schema_version')}")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    random.setstate(ckpt["python_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    torch.set_rng_state(ckpt["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["torch_cuda_rng_states"]])
    sampler.set_state(ckpt["sampler_rng_state"])
    aug_rng.bit_generator.state = ckpt["aug_rng_state"]
    return (
        int(ckpt["epoch"]),
        float(ckpt["best_auroc"]),
        int(ckpt["best_epoch"]),
        list(ckpt.get("history", [])),
    )


def validate_gmic_finetune_cli_guard(
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


def set_gmic_backbone_eval_mode(model: nn.Module, strategy: str) -> None:
    """Put trainable modules in `.train()` while keeping frozen BatchNorm modules in `.eval()`."""
    model.train()
    if strategy == "full":
        return
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            params = list(mod.parameters())
            if params and all(not p.requires_grad for p in params):
                mod.eval()


def compute_gmic_schedule_stats(
    n_pos_imgs: int,
    n_neg_imgs: int,
    batch_size: int,
    steps_per_epoch: int,
    epochs: int,
) -> Dict[str, float]:
    """Compute exact expected sample coverage under BalancedBatchSampler(paper_independent)."""
    n_pos_batch = max(2, int(batch_size) // 4)
    n_neg_batch = max(1, int(batch_size) - n_pos_batch)
    pos_draws_ep = int(steps_per_epoch) * n_pos_batch
    neg_draws_ep = int(steps_per_epoch) * n_neg_batch
    total_pos_draws = pos_draws_ep * int(epochs)
    total_neg_draws = neg_draws_ep * int(epochs)

    p_pos_1ep = 1.0 - (max(0.0, 1.0 - n_pos_batch / max(n_pos_imgs, 1)) ** int(steps_per_epoch))
    p_neg_1ep = 1.0 - (max(0.0, 1.0 - n_neg_batch / max(n_neg_imgs, 1)) ** int(steps_per_epoch))
    p_pos_all = 1.0 - (max(0.0, 1.0 - n_pos_batch / max(n_pos_imgs, 1)) ** (int(steps_per_epoch) * int(epochs)))
    p_neg_all = 1.0 - (max(0.0, 1.0 - n_neg_batch / max(n_neg_imgs, 1)) ** (int(steps_per_epoch) * int(epochs)))
    return {
        "n_pos_per_batch": float(n_pos_batch),
        "n_neg_per_batch": float(n_neg_batch),
        "draws_per_epoch": float(int(steps_per_epoch) * int(batch_size)),
        "total_draws": float(int(epochs) * int(steps_per_epoch) * int(batch_size)),
        "avg_pos_exposure_per_epoch": pos_draws_ep / max(n_pos_imgs, 1),
        "avg_pos_exposure_total": total_pos_draws / max(n_pos_imgs, 1),
        "avg_neg_exposure_per_epoch": neg_draws_ep / max(n_neg_imgs, 1),
        "avg_neg_exposure_total": total_neg_draws / max(n_neg_imgs, 1),
        "expected_unique_pos_frac_1ep": float(p_pos_1ep),
        "expected_unique_neg_frac_1ep": float(p_neg_1ep),
        "expected_unique_pos_frac_total": float(p_pos_all),
        "expected_unique_neg_frac_total": float(p_neg_all),
    }


def validate_gmic_eval_cli_args(
    *,
    split: str,
    batch_size: int,
    num_shards: int,
    checkpoint_path: str,
    approve_eval: bool,
    max_spend_usd: float,
) -> None:
    """Validate CLI args for evaluation-only GMIC execution."""
    validate_gmic_finetune_cli_guard(
        stage="eval",
        approve_full_train=False,
        approve_eval=approve_eval,
        max_spend_usd=max_spend_usd,
    )
    if split not in ("rsna_val_calib", "nlbs"):
        raise ValueError(f"Unsupported GMIC eval --split={split!r}; expected 'rsna_val_calib' or 'nlbs'")
    if int(batch_size) < 1:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
    if int(num_shards) < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if not str(checkpoint_path).strip():
        raise ValueError("--checkpoint-path must be non-empty for --stage eval")


def load_gmic_finetuned_model_for_eval(
    checkpoint_path: Path,
    *,
    expected_sha256: Optional[str] = None,
    model_index: str = "1",
    device: Optional[torch.device] = None,
    gmic_repo_dir: Path = GMIC_REPO_DEFAULT,
    model_override: Optional[nn.Module] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Load fine-tuned GMIC checkpoint with independent SHA-256 verification and zero-grad eval guarantee."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing fine-tuned GMIC checkpoint: {ckpt_path}")
    actual_sha = compute_sha256_file(ckpt_path).lower()
    if expected_sha256 and actual_sha != str(expected_sha256).strip().lower():
        raise RuntimeError(
            f"Fine-tuned GMIC checkpoint SHA-256 mismatch: got {actual_sha}, expected {expected_sha256}"
        )

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if ckpt.get("schema_version") != GMIC_FINETUNED_CKPT_SCHEMA:
        raise RuntimeError(f"Unsupported fine-tuned GMIC checkpoint schema: {ckpt.get('schema_version')}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_override is not None:
        model = model_override
        mods: Dict[str, Any] = {}
    else:
        mods = _ensure_gmic_importable(gmic_repo_dir)
        params = {
            "device_type": "gpu" if device.type == "cuda" else "cpu",
            "gpu_number": device.index or 0,
            "max_crop_noise": (100, 100),
            "max_crop_size_noise": 100,
            "cam_size": (46, 30),
            "K": 6,
            "crop_shape": (256, 256),
            "post_processing_dim": 256,
            "num_classes": 2,
            "use_v1_global": False,
            "percent_t": GMIC_PERCENT_T_MAP[str(model_index)],
        }
        model = mods["gmic"].GMIC(params)

    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = {
        "model_id": GMIC_FINETUNED_MODEL_ID,
        "checkpoint_file": ckpt_path.name,
        "checkpoint_sha256": actual_sha,
        "strategy": ckpt.get("strategy"),
        "epoch": int(ckpt["epoch"]),
        "best_epoch": int(ckpt["best_epoch"]),
        "best_train_calib_auroc": float(ckpt["best_auroc"]),
        "seed": int(ckpt.get("seed", 42)),
    }
    return model, mods, meta


def parse_int_csv(raw: str) -> List[int]:
    """Parse a comma-separated list of positive integers."""
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"Expected positive comma-separated ints, got {raw!r}")
    return vals


def parse_str_csv(raw: str) -> List[str]:
    """Parse a comma-separated list of lowercase string tokens."""
    vals = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Expected non-empty comma-separated strings, got {raw!r}")
    return vals


def scale_steps_and_lr_for_batch_size(
    batch_size: int,
    *,
    base_batch_size: int = 8,
    base_steps_per_epoch: int = 600,
    base_lr: float = 1e-4,
    lr_scaling: str = "sqrt",
    base_warmup_steps: int = 100,
) -> Dict[str, Any]:
    """Preserve total training image exposure per epoch and scale LR/warmup fairly."""
    bs = int(batch_size)
    base_bs = int(base_batch_size)
    base_steps = int(base_steps_per_epoch)
    if bs < 1 or base_bs < 1 or base_steps < 1:
        raise ValueError("batch_size, base_batch_size, and base_steps_per_epoch must be >= 1")
    if lr_scaling not in ("sqrt", "linear", "none"):
        raise ValueError(f"Unsupported lr_scaling={lr_scaling!r}; expected 'sqrt', 'linear', or 'none'")

    target_draws = base_bs * base_steps
    steps_per_epoch = max(1, round(target_draws / bs))
    actual_draws = steps_per_epoch * bs
    ratio = float(bs) / float(base_bs)
    if lr_scaling == "sqrt":
        lr_mult = math.sqrt(ratio)
    elif lr_scaling == "linear":
        lr_mult = ratio
    else:
        lr_mult = 1.0
    scaled_lr = float(base_lr) * lr_mult
    warmup_steps = min(steps_per_epoch, max(0, round(float(base_warmup_steps) / ratio))) if base_warmup_steps > 0 else 0
    return {
        "batch_size": bs,
        "base_batch_size": base_bs,
        "base_steps_per_epoch": base_steps,
        "steps_per_epoch": int(steps_per_epoch),
        "target_draws_per_epoch": int(target_draws),
        "actual_draws_per_epoch": int(actual_draws),
        "lr_scaling": lr_scaling,
        "lr_mult": round(lr_mult, 6),
        "base_lr": float(base_lr),
        "scaled_lr": float(scaled_lr),
        "warmup_steps": int(warmup_steps),
    }


def configure_gmic_precision_and_scaler(precision: str, device: torch.device) -> Dict[str, Any]:
    """Return autocast dtype, enablement, and GradScaler for `bf16`, `fp16`, `tf32`, or `fp32`."""
    prec = str(precision).strip().lower()
    if prec not in ("bf16", "fp16", "tf32", "fp32"):
        raise ValueError(f"Unsupported GMIC precision={precision!r}; expected 'bf16', 'fp16', 'tf32', or 'fp32'")

    is_cuda = device.type == "cuda"
    scaler_device = "cuda" if is_cuda else "cpu"
    if prec == "bf16":
        if is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return {
            "precision": prec,
            "amp_enabled": is_cuda,
            "amp_dtype": torch.bfloat16,
            "scaler": torch.amp.GradScaler(scaler_device, enabled=False),
        }
    if prec == "fp16":
        if is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return {
            "precision": prec,
            "amp_enabled": is_cuda,
            "amp_dtype": torch.float16,
            "scaler": torch.amp.GradScaler(scaler_device, enabled=is_cuda),
        }
    if is_cuda:
        allow_tf32 = prec == "tf32"
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    return {
        "precision": prec,
        "amp_enabled": False,
        "amp_dtype": torch.float32,
        "scaler": torch.amp.GradScaler(scaler_device, enabled=False),
    }


def create_fused_adamw(
    param_groups: Sequence[Dict[str, Any]],
    *,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Tuple[torch.optim.Optimizer, bool]:
    """Create AdamW with `fused=True` on CUDA when supported, cleanly falling back otherwise."""
    if device is not None and device.type == "cuda":
        try:
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay, fused=True), True
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay), False


def apply_linear_warmup(
    optimizer: torch.optim.Optimizer,
    step_idx: int,
    warmup_steps: int,
    base_lrs: Sequence[float],
) -> float:
    """Apply linear LR warmup over `[0, warmup_steps)` and restore `base_lrs` thereafter."""
    if warmup_steps <= 0 or step_idx >= warmup_steps:
        scale = 1.0
    else:
        scale = float(step_idx + 1) / float(warmup_steps)
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = float(base_lr) * scale
    return float(scale)


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg or "cudaerrormemoryallocation" in msg


def run_progressive_gmic_benchmark_loop(
    model: nn.Module,
    batch_loader: Callable[[int, int], Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    *,
    batch_sizes: Sequence[int] = (8, 16, 32, 64, 128, 256),
    precisions: Sequence[str] = ("bf16", "fp16", "tf32"),
    strategy: str = "partial",
    base_lr: float = 1e-4,
    backbone_lr_mult: float = 0.1,
    saliency_l1_weight: float = 1e-5,
    warmup_steps: int = 2,
    bench_steps: int = 4,
    val_loader: Optional[Callable[[int, int], np.ndarray]] = None,
    val_steps: int = 2,
    stop_on_throughput_drop: bool = True,
    drop_tolerance: float = 0.90,
) -> Dict[str, Any]:
    """Progressively benchmark GMIC batch sizes & precision modes, stopping cleanly on OOM/instability/drop."""
    sorted_bs = sorted({int(b) for b in batch_sizes if int(b) >= 1})
    results: List[Dict[str, Any]] = []
    model = model.to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for prec in precisions:
        best_tps = -1.0
        for bs_idx, bs in enumerate(sorted_bs):
            model.load_state_dict(initial_state)
            prec_cfg = configure_gmic_precision_and_scaler(prec, device)
            param_groups = configure_gmic_trainable_params(
                model, strategy=strategy, base_lr=base_lr, backbone_lr_mult=backbone_lr_mult
            )
            opt, used_fused = create_fused_adamw(param_groups, weight_decay=1e-4, device=device)
            scaler = prec_cfg["scaler"]
            set_gmic_backbone_eval_mode(model, strategy)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            data_load_sec = 0.0
            h2d_sec = 0.0
            fwd_bwd_sec = 0.0
            losses: List[float] = []
            grad_norms: List[float] = []
            status = "ok"
            stopped_reason = ""
            completed_bench_steps = 0

            try:
                total_steps = int(warmup_steps) + int(bench_steps)
                for step in range(total_steps):
                    t_d0 = time.perf_counter()
                    x_np, y_np = batch_loader(bs, step)
                    t_d1 = time.perf_counter()

                    tx_host = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32))
                    ty_host = torch.from_numpy(np.ascontiguousarray(y_np, dtype=np.float32))
                    if device.type == "cuda":
                        tx_host = tx_host.pin_memory()
                        ty_host = ty_host.pin_memory()
                    t_h0 = time.perf_counter()
                    bx = tx_host.to(device, non_blocking=True)
                    by = ty_host.to(device, non_blocking=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t_h1 = time.perf_counter()

                    t_f0 = time.perf_counter()
                    ld, gn = gmic_amp_train_step(
                        model=model,
                        bx=bx,
                        by=by,
                        optimizer=opt,
                        scaler=scaler,
                        saliency_l1_weight=saliency_l1_weight,
                        amp_enabled=prec_cfg["amp_enabled"],
                        amp_dtype=prec_cfg["amp_dtype"],
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    t_f1 = time.perf_counter()

                    if not (math.isfinite(ld["loss_total"]) and math.isfinite(gn)):
                        status = "unstable_nonfinite"
                        stopped_reason = f"non-finite loss/grad at step={step}"
                        break

                    if step >= warmup_steps:
                        data_load_sec += t_d1 - t_d0
                        h2d_sec += t_h1 - t_h0
                        fwd_bwd_sec += t_f1 - t_f0
                        losses.append(float(ld["loss_total"]))
                        grad_norms.append(float(gn))
                        completed_bench_steps += 1
            except Exception as exc:
                if _is_oom_error(exc):
                    status = "oom"
                    stopped_reason = str(exc)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

            total_step_sec = max(data_load_sec + h2d_sec + fwd_bwd_sec, 1e-9)
            n_bench_imgs = completed_bench_steps * bs
            imgs_per_sec = float(n_bench_imgs / total_step_sec) if completed_bench_steps > 0 else 0.0
            steps_per_sec = float(completed_bench_steps / total_step_sec) if completed_bench_steps > 0 else 0.0
            peak_alloc = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
            peak_res = float(torch.cuda.max_memory_reserved(device) / (1024**3)) if device.type == "cuda" else 0.0

            val_imgs_per_sec = None
            if status == "ok" and val_loader is not None:
                model.eval()
                val_imgs = 0
                t_v0 = time.perf_counter()
                try:
                    with torch.no_grad():
                        for vstep in range(int(val_steps)):
                            vx_np = val_loader(bs, vstep)
                            vtx = torch.from_numpy(np.ascontiguousarray(vx_np, dtype=np.float32))
                            if device.type == "cuda":
                                vtx = vtx.pin_memory()
                            vbx = vtx.to(device, non_blocking=True)
                            with torch.amp.autocast(
                                "cuda" if device.type == "cuda" else "cpu",
                                dtype=prec_cfg["amp_dtype"],
                                enabled=prec_cfg["amp_enabled"],
                            ):
                                _ = model(vbx)
                            val_imgs += int(vx_np.shape[0])
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                    val_imgs_per_sec = round(val_imgs / max(time.perf_counter() - t_v0, 1e-9), 3)
                except Exception as exc:
                    if _is_oom_error(exc):
                        val_imgs_per_sec = 0.0
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise

            row = {
                "precision": prec_cfg["precision"],
                "batch_size": bs,
                "status": status,
                "stopped_reason": stopped_reason,
                "imgs_per_sec": round(imgs_per_sec, 3),
                "steps_per_sec": round(steps_per_sec, 4),
                "val_imgs_per_sec": val_imgs_per_sec,
                "peak_allocated_vram_gb": round(peak_alloc, 4),
                "peak_reserved_vram_gb": round(peak_res, 4),
                "data_load_sec": round(data_load_sec, 4),
                "h2d_sec": round(h2d_sec, 4),
                "fwd_bwd_sec": round(fwd_bwd_sec, 4),
                "mean_loss": round(float(np.mean(losses)), 6) if losses else None,
                "all_finite": status == "ok" and bool(losses),
                "used_fused_adamw": used_fused,
                "pin_memory": device.type == "cuda",
                "non_blocking": True,
            }
            if status == "ok":
                if imgs_per_sec > best_tps:
                    best_tps = imgs_per_sec
                elif stop_on_throughput_drop and bs_idx >= 1 and imgs_per_sec < best_tps * float(drop_tolerance):
                    row["stopped_reason"] = f"throughput_drop ({imgs_per_sec:.2f} < {best_tps * drop_tolerance:.2f})"
                    results.append(row)
                    break
            results.append(row)
            if status in ("oom", "unstable_nonfinite"):
                break

    model.load_state_dict(initial_state)
    valid_rows = [r for r in results if r["status"] == "ok" and r["all_finite"]]
    best_row = max(valid_rows, key=lambda r: r["imgs_per_sec"]) if valid_rows else None
    return {
        "sweep_results": results,
        "recommended": best_row,
    }

