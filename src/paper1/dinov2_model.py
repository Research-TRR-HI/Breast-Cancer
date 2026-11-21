"""DINOv2 (facebookresearch/dinov2) ViT-S/14 Backbone + RSNA Cancer Head.

Note: DINOv2 is self-supervised on LVD-142M natural images (Apache-2.0) and is NOT
a pretrained mammography cancer detector. All cancer-classification weights are trained
strictly on RSNA `train` labels, with `train_calib` used solely for model selection.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2_REPO_DEFAULT = Path(__file__).resolve().parent.parent.parent / "external" / "dinov2"
DINOV2_VITS14_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINOV2_VITS14_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOV2_INPUT_SIZE = 518


def prepare_dinov2_tensor_from_uint16(img_u16: np.ndarray, size: int = DINOV2_INPUT_SIZE) -> torch.Tensor:
    """Convert Stage-1 1520x912 uint16 cropped mammogram into padded 518x518 3-ch ImageNet-normalized tensor."""
    arr = np.asarray(img_u16, dtype=np.float32)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v > min_v:
        arr = (arr - min_v) / (max_v - min_v)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    h, w = arr.shape
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_bot = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left

    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    t = F.pad(t, (pad_left, pad_right, pad_top, pad_bot), mode="constant", value=0.0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False).squeeze(0)  # [1, S, S]
    t = t.repeat(3, 1, 1)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (t - mean) / std


def build_frozen_dinov2_vits14(
    checkpoint_path: Optional[Path] = None,
    dinov2_repo_dir: Path = DINOV2_REPO_DEFAULT,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load official dinov2_vits14 encoder and freeze all parameters."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model = torch.hub.load(str(dinov2_repo_dir), "dinov2_vits14", source="local", pretrained=False)
        sd = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=True)
    else:
        source = "local" if Path(dinov2_repo_dir).exists() else "github"
        repo_arg = str(dinov2_repo_dir) if source == "local" else "facebookresearch/dinov2"
        model = torch.hub.load(repo_arg, "dinov2_vits14", source=source, pretrained=True)

    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    model.eval()
    return model


class DINOv2CancerHead(nn.Module):
    """RSNA-supervised binary cancer head over frozen 384-d DINOv2 CLS embeddings."""

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 0, dropout: float = 0.3):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
