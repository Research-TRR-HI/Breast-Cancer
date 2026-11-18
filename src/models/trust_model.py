"""TRUST Model Architecture and Case-Level Aggregation.

Reference:
    TRUST (2026): Threshold-Recalibrated Uncertainty-Safe Training
    for Certified Dismissal in Breast Cancer Screening.
    arXiv:2609.00300

Components:
    - EfficientNet-B5 vision encoder (Mammo-CLIP pretrained).
    - Frozen encoder (requires_grad = False).
    - LayerNorm on the 2048-dimensional feature embedding.
    - Dropout(0.3).
    - Linear classification head: nn.Linear(2048, 1).
    - Sigmoid score per image.
    - Case-level maximum score aggregation: max_{i in exam} p_i.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .efficientnet_custom import EfficientNet


class MammoCLIPEncoder(nn.Module):
    """Mammo-CLIP EfficientNet-B5 visual feature extractor.

    Output embedding dimension: 2048.
    Uses batmanlab/Mammo-CLIP's custom EfficientNet architecture so all 852
    checkpoint tensors from `b5-model-best-epoch-7.tar` match strictly.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
        model_name: str = "efficientnet-b5",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = EfficientNet.from_name(
            model_name,
            num_classes=1,
        )
        self.embed_dim = 2048  # out_dim of efficientnet-b5

        if checkpoint_path is not None:
            self.load_mammo_clip_weights(checkpoint_path)

        if freeze:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze all parameters in the encoder backbone."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters (if fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()

    def load_mammo_clip_weights(self, checkpoint_path: str) -> Tuple[int, int]:
        """Load weights from a Mammo-CLIP b5 checkpoint (.tar).

        Returns:
            Tuple of (num_loaded_keys, total_target_keys).
        """
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

        # Extract keys starting with 'image_encoder.'
        encoder_weights = {
            k[len("image_encoder.") :]: v
            for k, v in state_dict.items()
            if k.startswith("image_encoder.")
        }
        if not encoder_weights:
            encoder_weights = {
                k: v
                for k, v in state_dict.items()
                if not any(
                    k.startswith(p)
                    for p in ["text_encoder", "logit_scale", "projection", "image_projection", "text_projection"]
                )
            }

        self.encoder.load_state_dict(encoder_weights, strict=True)
        total_keys = len(self.encoder.state_dict())
        return len(encoder_weights), total_keys

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 3, H, W).

        Returns:
            Features tensor of shape (B, 2048).
        """
        return self.encoder(x)


class TRUSTClassifier(nn.Module):
    """TRUST / CE-Only Classifier Head on top of Mammo-CLIP backbone.

    Head architecture:
        LayerNorm(2048) -> Dropout(0.3) -> Linear(2048, 1) -> Sigmoid
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        embed_dim: int = 2048,
        dropout_rate: float = 0.3,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.embed_dim = embed_dim
        self.freeze_backbone = freeze_encoder

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(embed_dim, 1)

    def train(self, mode: bool = True) -> "TRUSTClassifier":
        """Keep encoder in eval mode if frozen."""
        super().train(mode)
        if self.freeze_backbone and hasattr(self.encoder, "eval"):
            self.encoder.eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract encoder feature vector."""
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.encoder(x)
        else:
            feats = self.encoder(x)
        return feats

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images tensor (B, 3, H, W) OR pre-extracted features (B, embed_dim).

        Returns:
            Tuple of (logits, probs), both shape (B, 1).
        """
        if x.dim() == 4:
            feats = self.forward_features(x)
        else:
            feats = x

        normed = self.norm(feats)
        dropped = self.dropout(normed)
        logits = self.head(dropped)
        probs = torch.sigmoid(logits)
        return logits, probs


def aggregate_case_scores(
    image_scores: Union[torch.Tensor, List[float]],
    case_ids: List[str],
) -> Dict[str, float]:
    """Aggregate image-level cancer scores to case-level using max aggregation (TRUST Eq. 1).

    Args:
        image_scores: Sequence of float scores in [0, 1].
        case_ids: Corresponding case/exam identifier for each score (same length).

    Returns:
        Dict mapping case_id to maximum score.
    """
    if isinstance(image_scores, torch.Tensor):
        scores_list = image_scores.detach().cpu().view(-1).tolist()
    else:
        scores_list = [float(s) for s in image_scores]

    if len(scores_list) == 0:
        raise ValueError("image_scores and case_ids must not be empty")
    if len(scores_list) != len(case_ids):
        raise ValueError(
            f"Length mismatch between image_scores ({len(scores_list)}) and case_ids ({len(case_ids)})"
        )

    case_max: Dict[str, float] = {}
    for score, cid in zip(scores_list, case_ids):
        key = str(cid)
        val = float(score)
        if key not in case_max or val > case_max[key]:
            case_max[key] = val

    return case_max


def aggregate_case_labels(
    image_labels: Union[torch.Tensor, List[int]],
    case_ids: List[str],
) -> Dict[str, int]:
    """Aggregate image-level binary cancer labels to case-level via max aggregation (TRUST Eq. 2)."""
    if isinstance(image_labels, torch.Tensor):
        labels_list = [int(v) for v in image_labels.detach().cpu().view(-1).tolist()]
    else:
        labels_list = [int(v) for v in image_labels]

    if len(labels_list) == 0:
        raise ValueError("image_labels and case_ids must not be empty")
    if len(labels_list) != len(case_ids):
        raise ValueError(
            f"Length mismatch between image_labels ({len(labels_list)}) and case_ids ({len(case_ids)})"
        )

    case_max: Dict[str, int] = {}
    for label, cid in zip(labels_list, case_ids):
        key = str(cid)
        if key not in case_max or label > case_max[key]:
            case_max[key] = label

    return case_max
