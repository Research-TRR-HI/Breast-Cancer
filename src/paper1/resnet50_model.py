"""Official Torchvision ResNet-50 (IMAGENET1K_V2) RSNA Fine-Tuning Model.

Note: ImageNet-1K pretrained ResNet-50 is NOT a pretrained mammography detector.
Requires end-to-end RSNA supervised training/fine-tuning on RSNA `train`, with
`train_calib` used solely for checkpoint selection.
"""

from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

from src.data.preprocessing import AugmentationParams, apply_trust_augmentation

RESNET50_IMAGENET1K_V2_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET50_DEFAULT_SIZE = (854, 512)  # Preserves 1520:912 (5:3) aspect ratio


def prepare_resnet50_tensor_from_uint16(
    img_u16: np.ndarray,
    target_size: tuple = RESNET50_DEFAULT_SIZE,
    augment_params: Optional[AugmentationParams] = None,
) -> torch.Tensor:
    """Convert Stage-1 1520x912 uint16 crop into aspect-preserving ImageNet-normalized tensor."""
    if img_u16 is None:
        raise ValueError("prepare_resnet50_tensor_from_uint16 received None image array.")
    arr = np.asarray(img_u16, dtype=np.float32)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

    min_v = float(t.min())
    max_v = float(t.max())
    if max_v > min_v:
        t = (t - min_v) / (max_v - min_v)
    else:
        t = torch.zeros_like(t)

    if augment_params is not None:
        t = apply_trust_augmentation(t.unsqueeze(0), augment_params).squeeze(0)

    t3 = t.unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (t3 - mean) / std


def build_resnet50_classifier(
    pretrained: bool = True,
    dropout: float = 0.3,
    local_weights_path: Optional[Path] = None,
) -> nn.Module:
    """Build ResNet-50 with official IMAGENET1K_V2 weights and a binary cancer head."""
    if local_weights_path is not None and Path(local_weights_path).exists():
        model = resnet50(weights=None)
        sd = torch.load(str(local_weights_path), map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=True)
    elif pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1),
    )
    return model


def amp_train_step(
    model: nn.Module,
    bx: torch.Tensor,
    by: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float = 5.0,
    amp_enabled: bool = True,
) -> Tuple[float, float]:
    """Execute one AMP training step with mandatory scaler.unscale_(optimizer) before clipping."""
    optimizer.zero_grad(set_to_none=True)
    device_type = "cuda" if bx.device.type == "cuda" else "cpu"
    use_autocast = bool(amp_enabled and device_type == "cuda")
    with torch.amp.autocast(device_type, enabled=use_autocast):
        loss = criterion(model(bx), by)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
    scaler.step(optimizer)
    scaler.update()
    return float(loss.item()), grad_norm


def save_resnet50_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_auroc: float,
    best_epoch: int,
    history: List[Dict[str, Any]],
    sampler: Any,
    aug_rng: np.random.Generator,
    seed: int,
) -> Path:
    """Atomically save ResNet-50 training state including all CPU/CUDA/NumPy/sampler/aug RNG states."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".pt.tmp")
    payload = {
        "schema_version": "paper1_resnet50_ckpt_v2",
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


def load_resnet50_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    sampler: Any,
    aug_rng: np.random.Generator,
    device: torch.device,
) -> Tuple[int, float, int, List[Dict[str, Any]]]:
    """Restore ResNet-50 model, optimizer, scheduler, scaler, and exact RNG/sampler states."""
    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    for req in ("model", "optimizer", "scheduler", "scaler", "sampler_rng_state", "aug_rng_state", "torch_cpu_rng_state"):
        if req not in ck or ck[req] is None:
            raise ValueError(f"ResNet-50 checkpoint missing required state key: {req}")
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    scaler.load_state_dict(ck["scaler"])
    random.setstate(ck["python_rng_state"])
    np.random.set_state(ck["numpy_rng_state"])
    torch.set_rng_state(ck["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and ck.get("torch_cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["torch_cuda_rng_states"]])
    sampler.set_state(ck["sampler_rng_state"])
    aug_rng.bit_generator.state = ck["aug_rng_state"]
    return int(ck["epoch"]), float(ck["best_auroc"]), int(ck.get("best_epoch", ck["epoch"])), list(ck["history"])
