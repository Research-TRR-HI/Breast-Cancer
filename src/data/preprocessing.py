"""Full TRUST DICOM Preprocessing and Stochastic Augmentation Pipeline.

Reference:
    TRUST (Shafie et al., 2026, arXiv:2609.00300v1): Section 3.2 (Pages 5-6)
    Stage 1 (DICOM -> cropped 16-bit grayscale image):
        1. Apply DICOM Modality LUT and (when available) VOI LUT.
        2. Invert MONOCHROME1 so breast tissue is brighter than background.
        3. Min-max normalize to [0, 1] for initial segmentation and cropping.
        4. Horizontally mirror right-breast images for consistent left-oriented chest wall.
        5. Segment breast using border-median threshold, fill holes, morphological closing (9x9),
           zero background outside mask, and crop to bounding box + 10% margin (no stretching).
        6. Quantize to 16-bit grayscale representation (uint16 PNG stage in paper).

    Stage 2 (Data loader -> tensor (3, 1520, 912)):
        7. Resize cropped image to (1520, 912) via bilinear/linear interpolation.
        8. Recalculate min-max normalization to [0, 1].
        9. Training only: stochastic affine augmentation (rotation +-10 deg, scale 0.9-1.1,
           translation +-5%), brightness +-0.05, contrast 0.90-1.10, and clip to [0, 1].
       10. Replicate single channel across 3 channels and standardize using Mammo-CLIP
           mean (0.3089279) and std (0.25053555408335154).
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
import pydicom
from scipy import ndimage
import torch
import torch.nn.functional as F

try:
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError:  # pragma: no cover - pydicom < 3.0 fallback
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


MAMMO_CLIP_MEAN = 0.3089279
MAMMO_CLIP_STD = 0.25053555408335154
PREPROCESS_PIPELINE_VERSION = "trust_sec3_2_v1"

ROTATION_RANGE_DEG = (-10.0, 10.0)
SCALE_RANGE = (0.90, 1.10)
TRANSLATION_FRAC_RANGE = (-0.05, 0.05)
BRIGHTNESS_DELTA_RANGE = (-0.05, 0.05)
CONTRAST_SCALE_RANGE = (0.90, 1.10)


@dataclass(frozen=True)
class AugmentationParams:
    """Exact stochastic augmentation parameters from TRUST Section 3.2 (Page 6)."""

    angle_deg: float
    scale: float
    tx_frac: float
    ty_frac: float
    brightness_delta: float
    contrast_factor: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AugmentationParams":
        return cls(
            angle_deg=float(data["angle_deg"]),
            scale=float(data["scale"]),
            tx_frac=float(data["tx_frac"]),
            ty_frac=float(data["ty_frac"]),
            brightness_delta=float(data["brightness_delta"]),
            contrast_factor=float(data["contrast_factor"]),
        )


def sample_augmentation_params(rng: np.random.Generator) -> AugmentationParams:
    """Sample TRUST Section 3.2 stochastic augmentation parameters from a seeded Generator."""
    return AugmentationParams(
        angle_deg=float(rng.uniform(*ROTATION_RANGE_DEG)),
        scale=float(rng.uniform(*SCALE_RANGE)),
        tx_frac=float(rng.uniform(*TRANSLATION_FRAC_RANGE)),
        ty_frac=float(rng.uniform(*TRANSLATION_FRAC_RANGE)),
        brightness_delta=float(rng.uniform(*BRIGHTNESS_DELTA_RANGE)),
        contrast_factor=float(rng.uniform(*CONTRAST_SCALE_RANGE)),
    )


def get_preprocessing_config_hash(
    target_size: Tuple[int, int] = (1520, 912),
    crop_margin_frac: float = 0.10,
    simulate_16bit_png: bool = True,
) -> str:
    """Return deterministic SHA-256 identifying the full TRUST preprocessing configuration."""
    cfg = {
        "version": PREPROCESS_PIPELINE_VERSION,
        "target_size": list(target_size),
        "crop_margin_frac": float(crop_margin_frac),
        "closing_kernel": [9, 9],
        "simulate_16bit_png": bool(simulate_16bit_png),
        "mammo_clip_mean": MAMMO_CLIP_MEAN,
        "mammo_clip_std": MAMMO_CLIP_STD,
        "aug_rotation_deg": list(ROTATION_RANGE_DEG),
        "aug_scale": list(SCALE_RANGE),
        "aug_translation_frac": list(TRANSLATION_FRAC_RANGE),
        "aug_brightness_delta": list(BRIGHTNESS_DELTA_RANGE),
        "aug_contrast_scale": list(CONTRAST_SCALE_RANGE),
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()


def _minmax_normalize_np(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    return np.zeros_like(arr, dtype=np.float32)


def _extract_laterality_from_dataset(ds: pydicom.Dataset) -> str:
    for attr in ("ImageLaterality", "Laterality"):
        val = getattr(ds, attr, None)
        if val is not None:
            s = str(val).strip().upper()
            if s in ("L", "R"):
                return s
    return ""


def _should_mirror_right_breast(arr: np.ndarray, laterality: str, auto_detect_right: bool) -> bool:
    lat = (laterality or "").strip().upper()
    if lat == "R":
        return True
    if lat == "L":
        return False
    if not auto_detect_right or arr.shape[1] < 2:
        return False
    mid = arr.shape[1] // 2
    left_mean = float(np.mean(arr[:, :mid]))
    right_mean = float(np.mean(arr[:, mid:]))
    return right_mean > left_mean


def _segment_mask_and_crop_breast(
    arr: np.ndarray,
    crop_margin_frac: float = 0.10,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Segment breast region via border-median threshold, fill holes, 9x9 closing, and 10% bbox crop."""
    h, w = arr.shape
    if h < 4 or w < 4:
        return arr.copy(), (0, h, 0, w)

    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    border_median = float(np.median(border_pixels))
    thresh = border_median + 1e-5
    mask = arr > thresh

    if not np.any(mask) or np.all(mask):
        return arr.copy(), (0, h, 0, w)

    # Keep largest connected component first to suppress stray markers near corners
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest_label = int(np.argmax(counts))
        mask = labeled == largest_label

    # Fill internal holes and apply 9x9 morphological closing (TRUST Section 3.2)
    mask = ndimage.binary_fill_holes(mask)
    kernel = np.ones((9, 9), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    if not np.any(mask):
        return arr.copy(), (0, h, 0, w)

    masked = np.where(mask, arr, np.float32(0.0))
    ys, xs = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    x_min, x_max = int(xs.min()), int(xs.max()) + 1

    box_h = y_max - y_min
    box_w = x_max - x_min
    margin_y = int(round(crop_margin_frac * box_h))
    margin_x = int(round(crop_margin_frac * box_w))

    y0 = max(0, y_min - margin_y)
    y1 = min(h, y_max + margin_y)
    x0 = max(0, x_min - margin_x)
    x1 = min(w, x_max + margin_x)

    return masked[y0:y1, x0:x1].copy(), (y0, y1, x0, x1)


def decode_dicom_array(
    dicom_path: Union[str, Path],
    apply_luts: bool = True,
) -> Tuple[np.ndarray, str, str]:
    """Decode DICOM pixels, apply Modality LUT & VOI LUT safely, and return (array, photometric, laterality)."""
    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array

    if apply_luts:
        try:
            arr = apply_modality_lut(arr, ds)
        except Exception:
            pass
        has_voi = any(
            hasattr(ds, tag)
            for tag in ("VOILUTSequence", "WindowCenter", "WindowWidth")
        )
        if has_voi:
            try:
                arr = apply_voi_lut(arr, ds)
            except Exception:
                pass

    arr_f32 = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    laterality = _extract_laterality_from_dataset(ds)
    return arr_f32, photometric, laterality


def extract_cropped_mammogram(
    arr: np.ndarray,
    photometric: str = "",
    laterality: str = "",
    invert_monochrome1: bool = True,
    mirror_right: bool = True,
    auto_detect_right: bool = True,
    crop_breast: bool = True,
    crop_margin_frac: float = 0.10,
    simulate_16bit_png: bool = True,
) -> np.ndarray:
    """Execute TRUST Stage 1 preprocessing (invert MONOCHROME1, norm, right-breast flip, segment & crop).

    Returns float32 grayscale array (quantized via uint16 if simulate_16bit_png=True) in [0, 1].
    """
    work = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if invert_monochrome1 and (photometric or "").strip().upper() == "MONOCHROME1":
        work = np.max(work) - work

    work = _minmax_normalize_np(work)

    if mirror_right and _should_mirror_right_breast(work, laterality, auto_detect_right):
        work = np.fliplr(work).copy()

    if crop_breast:
        work, _ = _segment_mask_and_crop_breast(work, crop_margin_frac=crop_margin_frac)

    work = np.clip(work, 0.0, 1.0)
    if simulate_16bit_png:
        q16 = np.round(work * 65535.0).astype(np.uint16)
        work = (q16.astype(np.float32) / 65535.0)
    return work


def apply_trust_augmentation(
    img_2d: torch.Tensor,
    params: AugmentationParams,
) -> torch.Tensor:
    """Apply exact TRUST Section 3.2 stochastic affine + brightness/contrast augmentation.

    Args:
        img_2d: Single-channel tensor of shape (1, H, W) in [0, 1].
        params: Sampled AugmentationParams.

    Returns:
        Augmented tensor of shape (1, H, W) clipped to [0, 1].
    """
    if img_2d.dim() != 3 or img_2d.shape[0] != 1:
        raise ValueError(f"Expected shape (1, H, W), got {tuple(img_2d.shape)}")

    _, h, w = img_2d.shape
    theta_rad = math.radians(params.angle_deg)
    cos_a = math.cos(theta_rad)
    sin_a = math.sin(theta_rad)
    inv_s = 1.0 / max(params.scale, 1e-6)

    # PyTorch grid_sample uses normalized [-1, 1] output->input inverse mapping.
    # To shift the image content by (+tx_frac*w, +ty_frac*h), inverse shifts by -2*tx_frac, -2*ty_frac.
    aspect = float(w) / max(float(h), 1.0)
    # Build inverse affine in normalized coordinates accounting for non-square aspect ratio
    a11 = inv_s * cos_a
    a12 = inv_s * (-sin_a) / aspect
    a21 = inv_s * sin_a * aspect
    a22 = inv_s * cos_a
    tx_norm = 2.0 * params.tx_frac
    ty_norm = 2.0 * params.ty_frac
    # Inverse translation = -R^{-1} * t
    inv_tx = -(a11 * tx_norm + a12 * ty_norm)
    inv_ty = -(a21 * tx_norm + a22 * ty_norm)

    theta = torch.tensor(
        [[[a11, a12, inv_tx], [a21, a22, inv_ty]]],
        dtype=torch.float32,
        device=img_2d.device,
    )
    grid = F.affine_grid(theta, size=(1, 1, h, w), align_corners=False)
    warped = F.grid_sample(
        img_2d.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0)

    adjusted = warped * float(params.contrast_factor) + float(params.brightness_delta)
    return adjusted.clamp(0.0, 1.0)


def prepare_tensor_from_cropped(
    cropped: np.ndarray,
    target_size: Tuple[int, int] = (1520, 912),
    augment_params: Optional[AugmentationParams] = None,
    standardize: bool = True,
    mean: float = MAMMO_CLIP_MEAN,
    std: float = MAMMO_CLIP_STD,
) -> torch.Tensor:
    """Execute TRUST Stage 2 preprocessing on a Stage-1 cropped mammogram.

    Steps:
        1. Linear/bilinear resize to target_size (1520, 912).
        2. Recalculate min-max normalization to [0, 1].
        3. Apply optional stochastic augmentation and clip to [0, 1].
        4. Replicate to 3 channels and apply Mammo-CLIP mean/std standardization.
    """
    arr = np.asarray(cropped, dtype=np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    resized = F.interpolate(
        tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (1, H_t, W_t)

    min_v = float(resized.min().item())
    max_v = float(resized.max().item())
    if max_v > min_v:
        resized = (resized - min_v) / (max_v - min_v)
    else:
        resized = torch.zeros_like(resized)

    if augment_params is not None:
        resized = apply_trust_augmentation(resized, augment_params)

    out_tensor = resized.repeat(3, 1, 1).clamp(0.0, 1.0)
    if standardize:
        out_tensor = (out_tensor - float(mean)) / float(std)
    return out_tensor


def preprocess_pixel_array(
    arr: np.ndarray,
    photometric: str = "",
    laterality: str = "",
    target_size: Tuple[int, int] = (1520, 912),
    invert_monochrome1: bool = True,
    mirror_right: bool = True,
    auto_detect_right: bool = True,
    crop_breast: bool = True,
    crop_margin_frac: float = 0.10,
    simulate_16bit_png: bool = True,
    augment_params: Optional[AugmentationParams] = None,
    standardize: bool = True,
    mean: float = MAMMO_CLIP_MEAN,
    std: float = MAMMO_CLIP_STD,
) -> torch.Tensor:
    """Run full TRUST Stage 1 + Stage 2 preprocessing from a decoded float32 pixel array."""
    cropped = extract_cropped_mammogram(
        arr=arr,
        photometric=photometric,
        laterality=laterality,
        invert_monochrome1=invert_monochrome1,
        mirror_right=mirror_right,
        auto_detect_right=auto_detect_right,
        crop_breast=crop_breast,
        crop_margin_frac=crop_margin_frac,
        simulate_16bit_png=simulate_16bit_png,
    )
    return prepare_tensor_from_cropped(
        cropped=cropped,
        target_size=target_size,
        augment_params=augment_params,
        standardize=standardize,
        mean=mean,
        std=std,
    )


def preprocess_dicom(
    dicom_path: Union[str, Path],
    target_size: Tuple[int, int] = (1520, 912),
    invert_monochrome1: bool = True,
    mirror_right: bool = True,
    auto_detect_right: bool = True,
    crop_breast: bool = True,
    crop_margin_frac: float = 0.10,
    simulate_16bit_png: bool = True,
    apply_luts: bool = True,
    laterality: Optional[str] = None,
    augment_params: Optional[AugmentationParams] = None,
    standardize: bool = True,
    mean: float = MAMMO_CLIP_MEAN,
    std: float = MAMMO_CLIP_STD,
) -> torch.Tensor:
    """Read a DICOM mammogram and run the complete TRUST Section 3.2 preprocessing pipeline."""
    arr, photometric, dcm_laterality = decode_dicom_array(dicom_path, apply_luts=apply_luts)
    eff_laterality = (laterality or dcm_laterality or "").strip().upper()
    return preprocess_pixel_array(
        arr=arr,
        photometric=photometric,
        laterality=eff_laterality,
        target_size=target_size,
        invert_monochrome1=invert_monochrome1,
        mirror_right=mirror_right,
        auto_detect_right=auto_detect_right,
        crop_breast=crop_breast,
        crop_margin_frac=crop_margin_frac,
        simulate_16bit_png=simulate_16bit_png,
        augment_params=augment_params,
        standardize=standardize,
        mean=mean,
        std=std,
    )
