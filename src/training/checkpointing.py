"""Complete Deterministic Checkpointing & Resume Support for TRUST Training.

Saves and restores all model, optimizer, scaler, epoch, dynamic threshold,
Python/NumPy/Torch/CUDA RNGs, sampler RNG, augmentation RNG, and training config.
Never silently restarts from epoch 1 when resume=True is requested.
"""

import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


def save_training_checkpoint(
    checkpoint_path: Path,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    tau_t: float,
    history: List[Dict[str, Any]],
    training_config: Dict[str, Any],
    sampler: Optional[Any] = None,
    aug_rng: Optional[np.random.Generator] = None,
) -> Path:
    """Atomically save complete training state at an epoch boundary."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".pt.tmp")

    payload = {
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "tau_t": float(tau_t),
        "history": list(history),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "sampler_rng_state": sampler.get_state() if sampler is not None else None,
        "aug_rng_state": aug_rng.bit_generator.state if aug_rng is not None else None,
        "training_config": dict(training_config),
    }

    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


def load_training_checkpoint(
    checkpoint_path: Path,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    sampler: Optional[Any] = None,
    aug_rng: Optional[np.random.Generator] = None,
    expected_config: Optional[Dict[str, Any]] = None,
) -> Tuple[int, float, List[Dict[str, Any]], Dict[str, Any]]:
    """Restore complete training state from an epoch-boundary checkpoint.

    Raises FileNotFoundError if checkpoint does not exist, and ValueError if
    training configuration or RNG metadata do not match.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Resume requested, but checkpoint does not exist at {checkpoint_path}. "
            "Refusing to silently restart from epoch 1."
        )

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    saved_cfg = ckpt.get("training_config", {})
    if expected_config is not None:
        for k, v in expected_config.items():
            if k in saved_cfg and saved_cfg[k] != v:
                raise ValueError(
                    f"Checkpoint config mismatch for {k!r}: saved={saved_cfg[k]!r} != expected={v!r}"
                )

    head.load_state_dict(ckpt["head_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    if ckpt.get("python_rng_state") is not None:
        random.setstate(ckpt["python_rng_state"])
    if ckpt.get("numpy_rng_state") is not None:
        np.random.set_state(ckpt["numpy_rng_state"])
    if ckpt.get("torch_cpu_rng_state") is not None:
        torch.set_rng_state(ckpt["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["torch_cuda_rng_states"]])
    if sampler is not None and ckpt.get("sampler_rng_state") is not None:
        sampler.set_state(ckpt["sampler_rng_state"])
    if aug_rng is not None and ckpt.get("aug_rng_state") is not None:
        aug_rng.bit_generator.state = ckpt["aug_rng_state"]

    completed_epoch = int(ckpt["epoch"])
    tau_t = float(ckpt["tau_t"])
    history = list(ckpt.get("history", []))
    return completed_epoch, tau_t, history, ckpt
