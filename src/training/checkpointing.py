"""Strict Versioned Checkpointing & Resume Validation (Critical Fix B).

Enforces schema version anchorshift_trust_ckpt_v2, mandatory training configuration
validation, mandatory RNG state presence (Python, NumPy, Torch CPU/CUDA, sampler,
augmentation), cache manifest SHA-256 linkage, and fsynced atomic writes.
"""

import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

CHECKPOINT_SCHEMA_VERSION = "anchorshift_trust_ckpt_v2"

MANDATORY_CONFIG_KEYS: Tuple[str, ...] = (
    "baseline",
    "epochs",
    "batch_size",
    "n_pos",
    "steps_per_epoch",
    "epoch_policy",
    "lr",
    "weight_decay",
    "margin",
    "lambda_focal",
    "lambda_dismiss",
    "initial_tau",
    "grad_clip",
    "micro_batch_size",
    "precision_policy",
    "seed",
    "preprocess_config_hash",
    "stage1_config_hash",
    "model_checkpoint_sha256",
    "split_manifest_sha256",
    "use_cached_features",
    "cache_manifest_sha256",
)

MANDATORY_PAYLOAD_KEYS: Tuple[str, ...] = (
    "schema_version",
    "head_state_dict",
    "optimizer_state_dict",
    "scaler_state_dict",
    "epoch",
    "total_epochs",
    "tau_t",
    "history",
    "python_rng_state",
    "numpy_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_states",
    "sampler_rng_state",
    "aug_rng_state",
    "training_config",
)


def _validate_training_config(cfg: Dict[str, Any]) -> None:
    missing = [k for k in MANDATORY_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"training_config missing mandatory keys: {missing}")
    if bool(cfg["use_cached_features"]) and not cfg["cache_manifest_sha256"]:
        raise ValueError("training_config requires non-empty cache_manifest_sha256 when use_cached_features=True")


def save_training_checkpoint(
    checkpoint_path: Path,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    tau_t: float,
    history: List[Dict[str, Any]],
    training_config: Dict[str, Any],
    sampler: Any,
    aug_rng: np.random.Generator,
) -> Path:
    """Atomically save complete versioned training state with fsync."""
    _validate_training_config(training_config)
    if sampler is None or aug_rng is None:
        raise ValueError("Both sampler and aug_rng are required to save a resumable checkpoint.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".pt.tmp")

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "total_epochs": int(training_config["epochs"]),
        "tau_t": float(tau_t),
        "history": list(history),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "sampler_rng_state": sampler.get_state(),
        "aug_rng_state": aug_rng.bit_generator.state,
        "training_config": dict(training_config),
    }

    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


def load_training_checkpoint(
    checkpoint_path: Path,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    sampler: Any,
    aug_rng: np.random.Generator,
    expected_config: Dict[str, Any],
) -> Tuple[int, float, List[Dict[str, Any]], Dict[str, Any]]:
    """Strictly validate schema, config, cache linkage, and RNG states before resuming."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Resume requested, but checkpoint does not exist at {checkpoint_path}. "
            "Refusing to silently restart from epoch 1."
        )

    _validate_training_config(expected_config)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(ckpt)}")
    if ckpt.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible checkpoint schema_version={ckpt.get('schema_version')!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION!r}."
        )

    missing_payload = [k for k in MANDATORY_PAYLOAD_KEYS if k not in ckpt]
    if missing_payload:
        raise ValueError(f"Checkpoint missing mandatory payload keys: {missing_payload}")

    saved_cfg = ckpt["training_config"]
    if not isinstance(saved_cfg, dict):
        raise ValueError("Checkpoint training_config must be a dict.")
    _validate_training_config(saved_cfg)

    for k in MANDATORY_CONFIG_KEYS:
        if saved_cfg[k] != expected_config[k]:
            raise ValueError(
                f"Checkpoint training_config mismatch for {k!r}: saved={saved_cfg[k]!r} != expected={expected_config[k]!r}"
            )

    for rng_key in ("python_rng_state", "numpy_rng_state", "torch_cpu_rng_state", "sampler_rng_state", "aug_rng_state"):
        if ckpt[rng_key] is None:
            raise ValueError(f"Checkpoint missing mandatory RNG state: {rng_key}")

    if device.type == "cuda" and ckpt.get("torch_cuda_rng_states") is None:
        raise ValueError("Checkpoint missing torch_cuda_rng_states required for CUDA resume.")

    if scaler is not None and ckpt.get("scaler_state_dict") is None and expected_config.get("precision_policy") == "amp":
        raise ValueError("Checkpoint missing scaler_state_dict required for AMP resume.")

    head.load_state_dict(ckpt["head_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    random.setstate(ckpt["python_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    torch.set_rng_state(ckpt["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_states") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["torch_cuda_rng_states"]])
    sampler.set_state(ckpt["sampler_rng_state"])
    aug_rng.bit_generator.state = ckpt["aug_rng_state"]

    completed_epoch = int(ckpt["epoch"])
    tau_t = float(ckpt["tau_t"])
    history = list(ckpt["history"])
    return completed_epoch, tau_t, history, ckpt
