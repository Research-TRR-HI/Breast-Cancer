"""Loss Functions for CE Baseline and TRUST Training.

Reference:
    TRUST (2026): Section 3.2
    - Binary Cross Entropy (BCE) Loss
    - Focal Loss (focuses on hard negatives/positives and class imbalance)
    - Dynamic Dismissal Loss:
        L_dismiss = (1 / N_pos) * sum_{i in pos} max(0, tau_t + margin - p_i)
        where tau_t is the dynamic provisional threshold recalculated per epoch,
        margin is safety distance (default 0.05), and p_i is the model probability.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Numerically stable binary Focal Loss with logits input.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            logits: Predicted logits of shape (B, 1) or (B,).
            targets: Binary ground truth labels (0 or 1) of same shape.

        Returns:
            Scalar loss tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DynamicDismissalLoss(nn.Module):
    """TRUST Dynamic Dismissal Loss on positive images.

    Penalizes cancer-positive images whose prediction score p_i is below
    or dangerously close to the current provisional dismissal threshold tau_t:

        L_dismiss = (1 / max(N_pos, 1)) * sum_{i in pos} relu(tau_t + margin - p_i)

    Args:
        margin: Safety margin above threshold (default: 0.05).
    """

    def __init__(self, margin: float = 0.05) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        tau_t: float,
    ) -> Tuple[torch.Tensor, int]:
        """Forward pass.

        Args:
            probs: Predicted probabilities in [0, 1] of shape (B, 1) or (B,).
            targets: Binary ground truth labels of same shape.
            tau_t: Current epoch's provisional threshold.

        Returns:
            Tuple of (scalar dismissal loss, number of positive images in batch).
        """
        probs = probs.view(-1)
        targets = targets.view(-1)

        pos_mask = targets == 1
        n_pos = int(pos_mask.sum().item())

        if n_pos == 0:
            return probs.sum() * 0.0, 0

        pos_probs = probs[pos_mask]
        penalty = F.relu(tau_t + self.margin - pos_probs)
        loss = penalty.sum() / max(n_pos, 1)
        return loss, n_pos


class TRUSTLoss(nn.Module):
    """Unified loss module for CE-only baseline and TRUST training.

    Total loss:
        L_total = L_bce + lambda_focal * L_focal + lambda_dismiss * L_dismiss

    Setting lambda_focal = 0 and lambda_dismiss = 0 recovers the pure CE baseline.
    """

    def __init__(
        self,
        baseline: str = "trust",
        margin: float = 0.05,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        lambda_focal: float = 1.0,
        lambda_dismiss: float = 1.0,
    ) -> None:
        super().__init__()
        self.baseline = baseline.lower()
        self.margin = margin
        self.lambda_focal = lambda_focal if self.baseline == "trust" else 0.0
        self.lambda_dismiss = lambda_dismiss if self.baseline == "trust" else 0.0

        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dismissal = DynamicDismissalLoss(margin=margin)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        tau_t: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss and return loss breakdown.

        Args:
            logits: Output logits of shape (B, 1) or (B,).
            targets: Target binary labels of shape (B, 1) or (B,).
            tau_t: Provisional dismissal threshold for current epoch (TRUST only).

        Returns:
            Tuple of (total_loss, dict_of_components).
        """
        targets_f = targets.view(-1, 1).float()
        l_bce = self.bce(logits, targets_f)

        components = {
            "loss_bce": float(l_bce.detach().cpu().item()),
            "loss_focal": 0.0,
            "loss_dismiss": 0.0,
            "tau_t": float(tau_t) if tau_t is not None else 0.0,
        }
        total_loss = l_bce

        if self.baseline == "trust":
            l_focal = self.focal(logits, targets_f)
            components["loss_focal"] = float(l_focal.detach().cpu().item())
            total_loss = total_loss + self.lambda_focal * l_focal

            if tau_t is not None and self.lambda_dismiss > 0:
                probs = torch.sigmoid(logits)
                l_dismiss, _ = self.dismissal(probs, targets_f, tau_t=tau_t)
                components["loss_dismiss"] = float(l_dismiss.detach().cpu().item())
                total_loss = total_loss + self.lambda_dismiss * l_dismiss

        components["loss_total"] = float(total_loss.detach().cpu().item())
        return total_loss, components
