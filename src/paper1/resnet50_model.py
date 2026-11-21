"""Official Torchvision ResNet-50 (IMAGENET1K_V2) RSNA Fine-Tuning Model.

Note: ImageNet-1K pretrained ResNet-50 is NOT a pretrained mammography detector.
Requires end-to-end RSNA supervised training/fine-tuning on RSNA `train`, with
`train_calib` used solely for checkpoint selection.
"""

from pathlib import Path
from typing import Optional
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
        t = apply_trust_augmentation(t, augment_params)

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
