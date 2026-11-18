"""DICOM Preprocessing Pipeline for Mammography.

Handles:
- Loading DICOM files using pydicom.
- Inverting MONOCHROME1 images to MONOCHROME2 standard (white=dense tissue).
- Robust min-max intensity normalization to [0.0, 1.0].
- Resizing to TRUST target resolution (1520, 912) using bilinear interpolation.
- Channel expansion from 1 to 3 channels for EfficientNet-B5 backbone.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pydicom
import torch
import torch.nn.functional as F


MAMMO_CLIP_MEAN = 0.3089279
MAMMO_CLIP_STD = 0.25053555408335154


def preprocess_dicom(
    dicom_path: Union[str, Path],
    target_size: Tuple[int, int] = (1520, 912),
    invert_monochrome1: bool = True,
    standardize: bool = True,
    mean: float = MAMMO_CLIP_MEAN,
    std: float = MAMMO_CLIP_STD,
) -> torch.Tensor:
    """Read a DICOM mammogram, normalize, resize, and convert to 3-channel tensor.

    Args:
        dicom_path: Path to .dcm file.
        target_size: (height, width) tuple. Default: (1520, 912).
        invert_monochrome1: Whether to invert MONOCHROME1 to standard MONOCHROME2.
        standardize: Apply Mammo-CLIP pretrained mean/std standardization after [0, 1] scaling.
        mean: Dataset mean used by Mammo-CLIP B5 pretraining (0.3089279).
        std: Dataset std used by Mammo-CLIP B5 pretraining (0.25053555408335154).

    Returns:
        torch.FloatTensor of shape (3, target_size[0], target_size[1]).
    """
    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array.astype(np.float32)

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    if invert_monochrome1 and photometric == "MONOCHROME1":
        # Invert so higher intensity corresponds to higher tissue density/radiation attenuation
        arr = np.max(arr) - arr

    # Min-max normalization to [0, 1]
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr)

    # Convert to torch tensor (1, 1, H, W) for interpolation
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    # Bilinear resize to target_size (height, width)
    resized = F.interpolate(
        tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )  # shape (1, 1, H, W)

    # Squeeze batch, repeat to 3 channels (3, H, W)
    out_tensor = resized.squeeze(0).repeat(3, 1, 1).clamp(0.0, 1.0)
    if standardize:
        out_tensor = (out_tensor - mean) / std
    return out_tensor
