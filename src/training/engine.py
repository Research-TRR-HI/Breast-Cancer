"""Shared Production Training Engine Used by Both Full Training and Parity Pilot.

Guarantees identical:
    - Classification head initialization regardless of encoder construction order.
    - Step-level Dropout RNG seeding.
    - Effective-batch loss calculation and micro-batch positive weighting.
    - Gradient clipping, AMP scaling, and AdamW optimizer updates.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.models.losses import TRUSTLoss


def build_trust_head(
    embed_dim: int = 2048,
    dropout_rate: float = 0.3,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Deterministically construct LayerNorm -> Dropout(0.3) -> Linear(2048, 1) head.

    Saves and restores outer PyTorch CPU/CUDA RNG state so prior encoder
    initialization can NEVER shift the head's initial weights.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, 1),
        )
        if device is not None:
            head = head.to(device)
        return head
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def seed_dropout_step_rng(seed: int, epoch: int, batch_idx: int, micro_idx: int = 0) -> None:
    """Set deterministic PyTorch RNG for Dropout immediately before head forward pass."""
    step_seed = (int(seed) * 1000003 + int(epoch) * 10007 + int(batch_idx) * 101 + int(micro_idx)) & 0x7FFFFFFF
    torch.manual_seed(step_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(step_seed)


def run_effective_batch_step(
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    criterion: TRUSTLoss,
    batch_targets: torch.Tensor,
    tau_t: Optional[float],
    micro_batch_size: int,
    grad_clip: float,
    use_amp: bool,
    device: torch.device,
    seed: int,
    epoch: int,
    batch_idx: int,
    batch_features: Optional[torch.Tensor] = None,
    batch_inputs: Optional[torch.Tensor] = None,
    encoder: Optional[nn.Module] = None,
    capture_features_and_grads: bool = False,
) -> Tuple[float, Dict[str, float], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """Execute one exact effective-batch training step using either online inputs or cached features."""
    if (batch_features is None) == (batch_inputs is None):
        raise ValueError("Provide exactly one of batch_features (cached) or batch_inputs (online).")
    if batch_inputs is not None and encoder is None:
        raise ValueError("encoder must be provided when batch_inputs is passed.")

    total_batch_size = int(batch_targets.shape[0])
    targets_col = batch_targets.view(total_batch_size, 1).to(device=device, dtype=torch.float32)
    total_pos_in_batch = int((targets_col == 1.0).sum().item())

    mb_size = min(max(int(micro_batch_size), 1), total_batch_size)
    amp_enabled = bool(use_amp and device.type == "cuda")

    optimizer.zero_grad()
    agg_comp = {"loss_bce": 0.0, "loss_focal": 0.0, "loss_dismiss": 0.0, "loss_total": 0.0}
    captured_feats_chunks = []

    if batch_inputs is not None and encoder is not None:
        encoder.eval()

    for mb_idx, start in enumerate(range(0, total_batch_size, mb_size)):
        end = min(start + mb_size, total_batch_size)
        my = targets_col[start:end]

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            if batch_features is not None:
                feats = batch_features[start:end].to(device=device, non_blocking=True)
            else:
                bx = batch_inputs[start:end].to(device=device, non_blocking=True)
                with torch.no_grad():
                    feats = encoder(bx)

            if capture_features_and_grads:
                captured_feats_chunks.append(feats.float().detach().cpu())

            # Seed Dropout RNG immediately prior to head forward so online encoder calls cannot shift RNG
            seed_dropout_step_rng(seed=seed, epoch=epoch, batch_idx=batch_idx, micro_idx=mb_idx)
            logits = head(feats)
            loss_mb, comp_mb = criterion.forward_microbatch(
                logits.float(),
                my,
                tau_t=tau_t,
                total_batch_size=total_batch_size,
                total_pos_in_batch=total_pos_in_batch,
            )

        if scaler is not None and amp_enabled:
            scaler.scale(loss_mb).backward()
        else:
            loss_mb.backward()

        for k in agg_comp:
            agg_comp[k] += float(comp_mb[k])

    if scaler is not None and amp_enabled:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(head.parameters(), float(grad_clip))

    captured_grads = None
    if capture_features_and_grads:
        captured_grads = {
            n: p.grad.detach().cpu().clone()
            for n, p in head.named_parameters()
            if p.grad is not None
        }

    if scaler is not None and amp_enabled:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    captured_feats = torch.cat(captured_feats_chunks, dim=0) if capture_features_and_grads else None
    return float(agg_comp["loss_total"]), agg_comp, captured_feats, captured_grads
