"""Loss Functions for CE Baseline, Ablations, and TRUST Training.

Reference:
    TRUST (Shafie et al., 2026, arXiv:2609.00300v1): Section 3.8 & 3.10 (Eq. 6-8, 11-14)
    - Binary Cross Entropy (BCE) Loss:
        L_CE = -(1/N) * sum [y_i log(p_i) + (1 - y_i) log(1 - p_i)]
    - Focal Loss (no alpha weighting in Eq. 12, gamma = 2.0, lambda_focal = 1.0):
        L_focal = -(1/N) * sum [y_i (1 - p_i)^gamma log(p_i) + (1 - y_i) p_i^gamma log(1 - p_i)]
    - Dynamic Dismissal Loss (margin m = 0.10, lambda_dismiss = 0.05, initial tau_t = 0.05):
        L_dismiss = (1 / N_C+) * sum_{i: y_i=1} max(0, tau_t + m - p_i)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Numerically stable binary Focal Loss with logits input matching TRUST Eq. (12).

    When ``alpha`` is ``None`` (default, matching TRUST Eq. 12), no alpha class
    weighting is applied:
        FL = -(1/N) * sum [ y*(1-p)^gamma*log(p) + (1-y)*p^gamma*log(1-p) ]
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if alpha is not None and not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1] or None, got {alpha}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma = float(gamma)
        self.alpha = float(alpha) if alpha is not None else None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary focal loss from logits and binary targets."""
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).float()

        bce = F.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction="none")
        probs = torch.sigmoid(logits_flat)
        p_t = targets_flat * probs + (1.0 - targets_flat) * (1.0 - probs)
        modulating_factor = (1.0 - p_t).clamp(min=0.0) ** self.gamma

        loss = modulating_factor * bce
        if self.alpha is not None:
            alpha_t = targets_flat * self.alpha + (1.0 - targets_flat) * (1.0 - self.alpha)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DynamicDismissalLoss(nn.Module):
    """TRUST Dynamic Dismissal Loss on cancer-positive images (Eq. 7 & 14).

    Penalizes cancer-positive images whose prediction score p_i falls below
    tau_t + margin:

        L_dismiss = (1 / N_C+) * sum_{i: y_i = 1} max(0, tau_t + m - p_i)

    Args:
        margin: Safety margin m above provisional threshold tau_t (paper default: 0.10).
    """

    def __init__(self, margin: float = 0.10) -> None:
        super().__init__()
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        self.margin = float(margin)

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        tau_t: float,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, int]:
        """Compute dismissal hinge penalty on positive samples."""
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        pos_mask = targets_flat == 1
        n_pos = int(pos_mask.sum().item())
        if n_pos == 0:
            return probs_flat.sum() * 0.0, 0

        pos_probs = probs_flat[pos_mask]
        penalty = F.relu(float(tau_t) + self.margin - pos_probs)
        if reduction == "sum":
            return penalty.sum(), n_pos
        if reduction == "mean":
            return penalty.mean(), n_pos
        raise ValueError(f"Unsupported reduction: {reduction}")


class TRUSTLoss(nn.Module):
    """Unified objective for CE-only, CE+Focal, Fixed-tau, and TRUST closed-loop training.

    Total loss (TRUST Eq. 6 & 13):
        L = L_CE + lambda_focal * L_focal + lambda_dismiss * L_dismiss

    Paper defaults (arXiv:2609.00300v1 Section 3.8 & 3.10):
        - gamma = 2.0, focal_alpha = None (unweighted Eq. 12)
        - lambda_focal = 1.0
        - lambda_dismiss = 0.05
        - margin = 0.10
        - initial tau_t = 0.05
    """

    SUPPORTED_BASELINES = ("ce", "ce_focal", "fixed_tau", "trust")

    def __init__(
        self,
        baseline: str = "trust",
        margin: float = 0.10,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[float] = None,
        lambda_focal: float = 1.0,
        lambda_dismiss: float = 0.05,
    ) -> None:
        super().__init__()
        self.baseline = baseline.lower()
        if self.baseline not in self.SUPPORTED_BASELINES:
            raise ValueError(
                f"Unsupported baseline={baseline!r}. Choose from {self.SUPPORTED_BASELINES}."
            )

        self.margin = float(margin)
        if self.baseline == "ce":
            self.lambda_focal = 0.0
            self.lambda_dismiss = 0.0
        elif self.baseline == "ce_focal":
            self.lambda_focal = float(lambda_focal)
            self.lambda_dismiss = 0.0
        else:
            self.lambda_focal = float(lambda_focal)
            self.lambda_dismiss = float(lambda_dismiss)

        self.bce_mean = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_sum = nn.BCEWithLogitsLoss(reduction="sum")
        self.focal_mean = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="mean")
        self.focal_sum = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="sum")
        self.dismissal = DynamicDismissalLoss(margin=self.margin)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        tau_t: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute total loss over a complete effective batch."""
        logits_col = logits.view(-1, 1)
        targets_col = targets.view(-1, 1).float()
        return self.forward_microbatch(
            logits_col,
            targets_col,
            tau_t=tau_t,
            total_batch_size=int(logits_col.shape[0]),
            total_pos_in_batch=int((targets_col == 1).sum().item()),
        )

    def forward_microbatch(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        tau_t: Optional[float],
        total_batch_size: int,
        total_pos_in_batch: int,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute micro-batch contribution whose sum across all micro-batches equals full-batch loss.

        BCE and Focal Loss are divided by ``total_batch_size`` across the full effective batch,
        whereas Dynamic Dismissal Loss is divided by ``max(total_pos_in_batch, 1)`` so uneven
        positive counts across micro-batches remain mathematically exact.
        """
        if total_batch_size <= 0:
            raise ValueError(f"total_batch_size must be positive, got {total_batch_size}")

        logits_col = logits.view(-1, 1)
        targets_col = targets.view(-1, 1).float()

        l_bce = self.bce_sum(logits_col, targets_col) / float(total_batch_size)
        effective_tau = 0.05 if (tau_t is None and self.baseline == "fixed_tau") else tau_t

        components = {
            "loss_bce": float(l_bce.detach().cpu().item()),
            "loss_focal": 0.0,
            "loss_dismiss": 0.0,
            "tau_t": float(effective_tau) if effective_tau is not None else 0.0,
        }
        total_loss = l_bce

        if self.lambda_focal > 0.0:
            l_focal = self.focal_sum(logits_col, targets_col) / float(total_batch_size)
            components["loss_focal"] = float(l_focal.detach().cpu().item())
            total_loss = total_loss + self.lambda_focal * l_focal

        if self.lambda_dismiss > 0.0 and effective_tau is not None:
            probs = torch.sigmoid(logits_col)
            l_dismiss_sum, _ = self.dismissal(probs, targets_col, tau_t=effective_tau, reduction="sum")
            l_dismiss = l_dismiss_sum / float(max(total_pos_in_batch, 1))
            components["loss_dismiss"] = float(l_dismiss.detach().cpu().item())
            total_loss = total_loss + self.lambda_dismiss * l_dismiss

        components["loss_total"] = float(total_loss.detach().cpu().item())
        return total_loss, components
