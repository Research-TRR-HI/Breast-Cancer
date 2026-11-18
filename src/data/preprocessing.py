"""Full TRUST DICOM Preprocessing and Stochastic Augmentation Pipeline.

Reference:
    TRUST (Shafie et al., 2026, arXiv:2609.00300v1): Section 3.2 (Pages 5-6)
    Stage 1 (DICOM -> lossless cropped 16-bit representation):
        1. Apply DICOM Modality LUT and (when available) VOI LUT with strict error checking.
        2. Invert MONOCHROME1 so breast tissue is brighter than background.
        3. Min-max normalize to [0, 1] for initial segmentation and cropping.
        4. Horizontally mirror right-breast images using explicit laterality metadata.
        5. Segment breast using border-median threshold, fill holes, 9x9 morphological closing,
           zero background outside mask, and crop to bounding box + 10% margin (no stretching).
        6. Quantize to lossless 16-bit grayscale (uint16 [0, 65535], matching 16-bit PNG stage).

    Stage 2 (Loader -> tensor (3, 1520, 912)):
        7. Linear/bilinear resize cropped image to (1520, 912).
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
PREPROCESS_PIPELINE_VERSION = "trust_sec3_2_v2"

ROTATION_RANGE_DEG = (-10.0, 10.0)
SCALE_RANGE = (0.90, 1.10)
TRANSLATION_FRAC_RANGE = (-0.05, 0.05)
BRIGHTNESS_DELTA_RANGE = (-0.05, 0.05)
CONTRAST_SCALE_RANGE = (0.90, 1.10)

SUPPORTED_LATERALITY_POLICIES = ("strict", "allow_unknown_no_mirror", "heuristic_fallback")


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


@dataclass(frozen=True)
class Stage1CropResult:
    """Complete Stage-1 output with lossless uint16 image and segmentation/laterality diagnostics."""

    cropped_uint16: np.ndarray
    cropped_float32: np.ndarray
    orig_shape: Tuple[int, int]
    crop_shape: Tuple[int, int]
    crop_bbox: Tuple[int, int, int, int]
    border_median: float
    segmentation_status: str
    used_fallback_uncropped: bool
    mirrored: bool
    laterality: str
    laterality_status: str
    photometric: str


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


def get_stage1_config_hash(
    apply_luts: bool = True,
    invert_monochrome1: bool = True,
    mirror_right: bool = True,
    laterality_policy: str = "strict",
    segmentation_threshold_eps: float = 1e-5,
    use_largest_cc: bool = False,
    crop_margin_frac: float = 0.10,
    simulate_16bit_png: bool = True,
) -> str:
    """Return deterministic SHA-256 covering every Stage-1 preprocessing setting."""
    cfg = {
        "pipeline_version": PREPROCESS_PIPELINE_VERSION,
        "apply_luts": bool(apply_luts),
        "invert_monochrome1": bool(invert_monochrome1),
        "mirror_right": bool(mirror_right),
        "laterality_policy": str(laterality_policy),
        "segmentation_threshold_eps": float(segmentation_threshold_eps),
        "use_largest_cc": bool(use_largest_cc),
        "fill_holes": True,
        "closing_kernel": [9, 9],
        "zero_outside_mask": True,
        "crop_margin_frac": float(crop_margin_frac),
        "simulate_16bit_png": bool(simulate_16bit_png),
        "quantization_bits": 16 if simulate_16bit_png else 32,
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()


def get_preprocessing_config_hash(
    target_size: Tuple[int, int] = (1520, 912),
    crop_margin_frac: float = 0.10,
    simulate_16bit_png: bool = True,
    apply_luts: bool = True,
    invert_monochrome1: bool = True,
    mirror_right: bool = True,
    laterality_policy: str = "strict",
    segmentation_threshold_eps: float = 1e-5,
    use_largest_cc: bool = False,
    mean: float = MAMMO_CLIP_MEAN,
    std: float = MAMMO_CLIP_STD,
) -> str:
    """Return deterministic SHA-256 identifying all Stage-1 and Stage-2 preprocessing settings."""
    cfg = {
        "stage1_hash": get_stage1_config_hash(
            apply_luts=apply_luts,
            invert_monochrome1=invert_monochrome1,
            mirror_right=mirror_right,
            laterality_policy=laterality_policy,
            segmentation_threshold_eps=segmentation_threshold_eps,
            use_largest_cc=use_largest_cc,
            crop_margin_frac=crop_margin_frac,
            simulate_16bit_png=simulate_16bit_png,
        ),
        "target_size": list(target_size),
        "resize_interpolation": "bilinear_align_corners_false",
        "post_resize_minmax_norm": True,
        "aug_rotation_deg": list(ROTATION_RANGE_DEG),
        "aug_scale": list(SCALE_RANGE),
        "aug_translation_frac": list(TRANSLATION_FRAC_RANGE),
        "aug_brightness_delta": list(BRIGHTNESS_DELTA_RANGE),
        "aug_contrast_scale": list(CONTRAST_SCALE_RANGE),
        "aug_post_clip": [0.0, 1.0],
        "mammo_clip_mean": float(mean),
        "mammo_clip_std": float(std),
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()


def _minmax_normalize_np(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
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


def _resolve_laterality(
    arr: np.ndarray,
    laterality: str,
    laterality_policy: str,
    context_label: str,
) -> Tuple[bool, str, str]:
    """Resolve mirroring decision under explicit laterality policy."""
    if laterality_policy not in SUPPORTED_LATERALITY_POLICIES:
        raise ValueError(
            f"Unsupported laterality_policy={laterality_policy!r}. "
            f"Choose from {SUPPORTED_LATERALITY_POLICIES}."
        )

    lat = (laterality or "").strip().upper()
    if lat == "R":
        return True, "R", "explicit_R"
    if lat == "L":
        return False, "L", "explicit_L"

    if laterality_policy == "strict":
        raise ValueError(
            f"Missing or unknown breast laterality ({laterality!r}) for {context_label}. "
            "Strict production policy forbids silently guessing laterality."
        )
    if laterality_policy == "allow_unknown_no_mirror":
        return False, "UNKNOWN", "unknown_unmirrored"

    # Explicit documented heuristic fallback (never active in strict mode)
    if arr.shape[1] >= 2:
        mid = arr.shape[1] // 2
        should_flip = float(np.mean(arr[:, mid:])) > float(np.mean(arr[:, :mid]))
        return should_flip, ("R" if should_flip else "L"), "heuristic_guessed"
    return False, "UNKNOWN", "unknown_unmirrored"


def _segment_mask_and_crop_breast(
    arr: np.ndarray,
    crop_margin_frac: float = 0.10,
    segmentation_threshold_eps: float = 1e-5,
    use_largest_cc: bool = False,
    strict_segmentation: bool = True,
    context_label: str = "<array>",
) -> Tuple[np.ndarray, Tuple[int, int, int, int], float, str, bool]:
    """Segment breast region via border-median threshold, hole filling, 9x9 closing, and 10% bbox crop."""
    h, w = arr.shape
    if h < 4 or w < 4:
        if strict_segmentation:
            raise ValueError(f"Degenerate small image shape {arr.shape} in {context_label}")
        return arr.copy(), (0, h, 0, w), 0.0, "degenerate_small", True

    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    border_median = float(np.median(border_pixels))
    mask = arr > (border_median + float(segmentation_threshold_eps))

    if not np.any(mask):
        if strict_segmentation:
            raise ValueError(
                f"Breast segmentation produced an empty mask (border_median={border_median:.5f}) in {context_label}"
            )
        return arr.copy(), (0, h, 0, w), border_median, "empty_mask", True

    if np.all(mask):
        if strict_segmentation:
            raise ValueError(
                f"Breast segmentation classified all pixels as foreground (border_median={border_median:.5f}) in {context_label}"
            )
        return arr.copy(), (0, h, 0, w), border_median, "all_foreground", True

    # Optional connected-component filtering (OFF by default to match TRUST Sec 3.2 verbatim)
    if use_largest_cc:
        labeled, num_features = ndimage.label(mask)
        if num_features > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            mask = labeled == int(np.argmax(counts))

    # TRUST Section 3.2: fill holes inside breast mask, then 9x9 morphological closing
    mask = ndimage.binary_fill_holes(mask)
    kernel = np.ones((9, 9), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    if not np.any(mask):
        if strict_segmentation:
            raise ValueError(f"Breast mask became empty after closing in {context_label}")
        return arr.copy(), (0, h, 0, w), border_median, "empty_mask", True

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

    return masked[y0:y1, x0:x1].copy(), (y0, y1, x0, x1), border_median, "success", False


def decode_dicom_array(
    dicom_path: Union[str, Path],
    apply_luts: bool = True,
) -> Tuple[np.ndarray, str, str]:
    """Decode DICOM pixels, apply Modality/VOI LUTs with strict error reporting, and return (arr, phot, lat)."""
    path_str = str(dicom_path)
    ds = pydicom.dcmread(path_str)
    try:
        arr = ds.pixel_array
    except Exception as exc:
        raise RuntimeError(f"Failed to decode pixel_array for {path_str}: {exc}") from exc

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"Expected non-empty 2D mammogram array in {path_str}, got shape {arr.shape}")

    if apply_luts:
        has_modality_lut = any(
            hasattr(ds, tag)
            for tag in ("ModalityLUTSequence", "RescaleSlope", "RescaleIntercept")
        )
        if has_modality_lut:
            try:
                arr = apply_modality_lut(arr, ds)
            except Exception as exc:
                raise RuntimeError(
                    f"Modality LUT transformation failed for {path_str}: {exc}"
                ) from exc

        has_voi_lut = any(
            hasattr(ds, tag)
            for tag in ("VOILUTSequence", "WindowCenter", "WindowWidth")
        )
        if has_voi_lut:
            try:
                arr = apply_voi_lut(arr, ds)
            except Exception as exc:
                raise RuntimeError(
                    f"VOI LUT transformation failed for {path_str}: {exc}"
                ) from exc

    arr_f32 = np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(arr_f32)):
        raise ValueError(f"Non-finite pixel intensities (NaN/Inf) after LUT transform in {path_str}")

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    laterality = _extract_laterality_from_dataset(ds)
    return arr_f32, photometric, laterality


def extract_cropped_mammogram_with_metadata(
    arr: np.ndarray,
    photometric: str = "",
    laterality: str = "",
    invert_monochrome1: bool = True,
    mirror_right: bool = True,
    laterality_policy: str = "strict",
    crop_breast: bool = True,
    crop_margin_frac: float = 0.10,
    segmentation_threshold_eps: float = 1e-5,
    use_largest_cc: bool = False,
    strict_segmentation: bool = True,
    simulate_16bit_png: bool = True,
    context_label: str = "<array>",
) -> Stage1CropResult:
    """Run TRUST Stage-1 preprocessing and return lossless uint16 image plus complete diagnostics."""
    work = np.asarray(arr, dtype=np.float32)
    if work.ndim != 2 or work.size == 0 or not np.all(np.isfinite(work)):
        raise ValueError(f"Invalid pixel array in Stage 1 for {context_label}: shape={work.shape}")

    orig_shape = (int(work.shape[0]), int(work.shape[1]))
    phot_norm = (photometric or "").strip().upper()
    if invert_monochrome1 and phot_norm == "MONOCHROME1":
        work = np.max(work) - work

    work = _minmax_normalize_np(work)

    if mirror_right:
        should_flip, resolved_lat, lat_status = _resolve_laterality(
            work, laterality=laterality, laterality_policy=laterality_policy, context_label=context_label
        )
        if should_flip:
            work = np.fliplr(work).copy()
        mirrored = should_flip
    else:
        resolved_lat = (laterality or "").strip().upper() or "UNKNOWN"
        lat_status = f"explicit_{resolved_lat}" if resolved_lat in ("L", "R") else "mirror_disabled"
        mirrored = False

    if crop_breast:
        cropped, bbox, border_med, seg_status, used_fallback = _segment_mask_and_crop_breast(
            work,
            crop_margin_frac=crop_margin_frac,
            segmentation_threshold_eps=segmentation_threshold_eps,
            use_largest_cc=use_largest_cc,
            strict_segmentation=strict_segmentation,
            context_label=context_label,
        )
    else:
        cropped = work.copy()
        bbox = (0, orig_shape[0], 0, orig_shape[1])
        border_med = 0.0
        seg_status = "crop_disabled"
        used_fallback = False

    cropped_clipped = np.clip(cropped, 0.0, 1.0)
    cropped_u16 = np.round(cropped_clipped * 65535.0).astype(np.uint16)
    cropped_f32 = (cropped_u16.astype(np.float32) / 65535.0) if simulate_16bit_png else cropped_clipped.astype(np.float32)

    return Stage1CropResult(
        cropped_uint16=cropped_u16,
        cropped_float32=cropped_f32,
        orig_shape=orig_shape,
        crop_shape=(int(cropped_u16.shape[0]), int(cropped_u16.shape[1])),
        crop_bbox=bbox,
        border_median=float(border_med),
        segmentation_status=seg_status,
        used_fallback_uncropped=bool(used_fallback),
        mirrored=bool(mirrored),
        laterality=resolved_lat,
        laterality_status=lat_status,
        photometric=phot_norm,
    )


def extract_cropped_mammogram(
    arr: np.ndarray,
    photometric: str = "",
    laterality: str = "",
    invert_monochrome1: bool = True,
    mirror_right: bool = True,
    laterality_policy: str = "strict",
    crop_breast: bool = True,
    crop_margin_frac: float = 0.10,
    segmentation_threshold_eps: float = 1e-5,
    use_largest_cc: bool = False,
    strict_segmentation: bool = True,
    simulate_16bit_png: bool = True,
    context_label: str = "<array>",
) -> np.ndarray:
    """Convenience wrapper returning the Stage-1 float32 array."""
    return extract_cropped_mammogram_with_metadata(
        arr=arr,
        photometric=photometric,
        laterality=laterality,
        invert_monochrome1=invert_monochrome1,
        mirror_right=mirror_right,
        laterality_policy=laterality_policy,
        crop_breast=crop_breast,
        crop_margin_frac=crop_margin_frac,
        segmentation_threshold_eps=segmentation_threshold_eps,
        use_largest_cc=use_largest_cc,
        strict_segmentation=strict_segmentation,
        simulate_16bit_png=simulate_16bit_png,
        context_label=context_label,
    ).cropped_float32


def apply_trust_augmentation(
    img_2d: torch.Tensor,
    params: AugmentationParams,
) -> torch.Tensor:
    """Apply exact TRUST Section 3.2 stochastic affine + brightness/contrast augmentation."""
    if img_2d.dim() != 3 or img_2d.shape[0] != 1:
        raise ValueError(f"Expected shape (1, H, W), got {tuple(img_2d.shape)}")

    _, h, w = img_2d.shape
    theta_rad = math.radians(params.angle_deg)
    cos_a = math.cos(theta_rad)
    sin_a = math.sin(theta_rad)
    inv_s = 1.0 / max(params.scale, 1e-6)

    aspect = float(w) / max(float(h), 1.0)
    a11 = inv_s * cos_a
    a12 = inv_s * (-sin_a) / aspect
    a21 = inv_s * sin_a * aspect
    a22 = inv_s * cos_a
    tx_norm = 2.0 * params.tx_frac
    ty_norm = 2.0 * params.ty_frac
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
    """Execute TRUST Stage-2 preprocessing on a Stage-1 cropped array (uint16 or float32)."""
    arr = np.asarray(cropped)
    if arr.dtype == np.uint16:
        arr_f32 = arr.astype(np.float32) / 65535.0
    else:
        arr_f32 = arr.astype(np.float32)

    tensor = torch.from_numpy(arr_f32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

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
    laterality_policy: str = "strict",
    crop_breast: bool = True,
    crop_margin_frac: float = 0.10,
    segmentation_threshold_eps: float = 1e-5,
    use_largest_cc: bool = False,
    strict_segmentation: bool = True,
    simulate_16bit_png: bool = True,
    augment_params: Optional[AugmentationParams] = None,
    standardize: bool = True,
    mean: float = MAMMO_CLIP_MEAN,
    std: float = MAMMO_CLIP_STD,
    context_label: str = "<array>",
) -> torch.Tensor:
    """Run full TRUST Stage 1 + Stage 2 preprocessing from a decoded float32 pixel array."""
    cropped = extract_cropped_mammogram(
        arr=arr,
        photometric=photometric,
        laterality=laterality,
        invert_monochrome1=invert_monochrome1,
        mirror_right=mirror_right,
        laterality_policy=laterality_policy,
        crop_breast=crop_breast,
        crop_margin_frac=crop_margin_frac,
        segmentation_threshold_eps=segmentation_threshold_eps,
        use_largest_cc=use_largest_cc,
        strict_segmentation=strict_segmentation,
        simulate_16bit_png=simulate_16bit_png,
        context_label=context_label,
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
    laterality_policy: str = "strict",
    crop_breast: bool = True,
    crop_margin_frac: float = 0.10,
    segmentation_threshold_eps: float = 1e-5,
    use_largest_cc: bool = False,
    strict_segmentation: bool = True,
    simulate_16bit_png: bool = True,
    apply_luts: bool = True,
    laterality: Optional[str] = None,
    augment_params: Optional[AugmentationParams] = None,
    standardize: bool = True,
    mean: float = MAMMO_CLIP_MEAN,
    std: float = MAMMO_CLIP_STD,
) -> torch.Tensor:
    """Read a DICOM mammogram and run the complete TRUST Section 3.2 preprocessing pipeline."""
    path_str = str(dicom_path)
    arr, photometric, dcm_laterality = decode_dicom_array(dicom_path, apply_luts=apply_luts)

    arg_lat = (laterality or "").strip().upper()
    if arg_lat in ("L", "R") and dcm_laterality in ("L", "R") and arg_lat != dcm_laterality:
        if laterality_policy == "strict":
            raise ValueError(
                f"Conflicting laterality between argument/manifest ({arg_lat!r}) "
                f"and DICOM header ({dcm_laterality!r}) for {path_str}"
            )
    eff_laterality = arg_lat if arg_lat in ("L", "R") else dcm_laterality

    return preprocess_pixel_array(
        arr=arr,
        photometric=photometric,
        laterality=eff_laterality,
        target_size=target_size,
        invert_monochrome1=invert_monochrome1,
        mirror_right=mirror_right,
        laterality_policy=laterality_policy,
        crop_breast=crop_breast,
        crop_margin_frac=crop_margin_frac,
        segmentation_threshold_eps=segmentation_threshold_eps,
        use_largest_cc=use_largest_cc,
        strict_segmentation=strict_segmentation,
        simulate_16bit_png=simulate_16bit_png,
        augment_params=augment_params,
        standardize=standardize,
        mean=mean,
        std=std,
        context_label=path_str,
    )
