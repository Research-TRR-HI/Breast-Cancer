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
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
) -> Tuple[Dict[str, float], float]:
    """Execute one GMIC training step with FP32 probability BCE and unscaled grad clipping."""
    optimizer.zero_grad(set_to_none=True)
    device_type = "cuda" if bx.device.type == "cuda" else "cpu"
    use_autocast = bool(amp_enabled and device_type == "cuda")
    with torch.amp.autocast(device_type, enabled=use_autocast):
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
