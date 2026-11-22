"""Official GMIC (nyukat/GMIC) Adapter and Native 2944x1920 Preprocessing Pipeline.

Preserves exact architecture, weights (sample_model_1.p, MODEL_INDEX='1'),
cropping, optimal-center selection, and malignant_pred extraction from:
    https://github.com/nyukat/GMIC (AGPL-3.0)
"""

import importlib
import os
from pathlib import Path
import sys
import types
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
import torch

from src.data.preprocessing import decode_dicom_array

GMIC_REPO_DEFAULT = Path(__file__).resolve().parent.parent.parent / "external" / "GMIC"
GMIC_INPUT_SIZE = (2944, 1920)
GMIC_PERCENT_T_MAP = {"1": 0.02, "2": 0.03, "3": 0.03, "4": 0.05, "5": 0.1}


def _ensure_gmic_importable(gmic_repo_dir: Path = GMIC_REPO_DEFAULT) -> Any:
    """Temporarily import nyukat/GMIC modules without polluting sys.modules['src']."""
    if "h5py" not in sys.modules:
        try:
            importlib.import_module("h5py")
        except ImportError:
            sys.modules["h5py"] = types.ModuleType("h5py")

    gmic_str = str(Path(gmic_repo_dir).resolve())
    saved_src = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in list(saved_src.keys()):
        del sys.modules[k]
    sys.path.insert(0, gmic_str)
    try:
        g_crop = importlib.import_module("src.cropping.crop_mammogram")
        g_load = importlib.import_module("src.data_loading.loading")
        g_model = importlib.import_module("src.modeling.gmic")
        g_coc = importlib.import_module("src.optimal_centers.calc_optimal_centers")
        return {
            "crop_mammogram": g_crop,
            "loading": g_load,
            "gmic": g_model,
            "calc_optimal_centers": g_coc,
        }
    finally:
        if sys.path[0] == gmic_str:
            sys.path.pop(0)
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(saved_src)


def prepare_dicom_uint16_for_nyukat(
    dicom_path: Path,
    laterality: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decode DICOM with LUTs, invert MONOCHROME1, zero border background, and enforce chest-wall orientation.

    NYU/GMIC assume L views have chest wall on the left (facing right) and R views have chest wall
    on the right (facing left) prior to `flip_image(..., horizontal_flip='NO')`.
    """
    arr_f32, phot, lat_dcm = decode_dicom_array(dicom_path, apply_luts=True)
    eff_lat = lat_dcm if lat_dcm in ("L", "R") else str(laterality).strip().upper()
    if eff_lat not in ("L", "R"):
        raise ValueError(f"Unknown laterality {laterality!r} for {dicom_path}")

    if phot == "MONOCHROME1":
        arr_f32 = float(np.max(arr_f32)) - arr_f32

    border_pixels = np.concatenate([arr_f32[0, :], arr_f32[-1, :], arr_f32[:, 0], arr_f32[:, -1]])
    border_median = float(np.median(border_pixels))
    arr_f32 = np.maximum(arr_f32 - border_median, 0.0)
    max_val = float(np.max(arr_f32))
    if max_val > 0.0:
        arr_u16 = np.round((arr_f32 / max_val) * 65535.0).astype(np.uint16)
    else:
        arr_u16 = np.zeros_like(arr_f32, dtype=np.uint16)

    # Ensure standard NYU/GMIC orientation: L chest wall on left, R chest wall on right
    mid = arr_u16.shape[1] // 2
    left_mean = float(np.mean(arr_u16[:, :mid]))
    right_mean = float(np.mean(arr_u16[:, mid:]))
    oriented_flipped = False
    if eff_lat == "L" and right_mean > left_mean:
        arr_u16 = np.fliplr(arr_u16).copy()
        oriented_flipped = True
    elif eff_lat == "R" and left_mean > right_mean:
        arr_u16 = np.fliplr(arr_u16).copy()
        oriented_flipped = True

    return arr_u16, {
        "photometric": phot,
        "laterality": eff_lat,
        "border_median": border_median,
        "oriented_flipped": oriented_flipped,
    }


def standard_normalize_gmic_uint16(arr_u16: np.ndarray) -> np.ndarray:
    """Apply exact GMIC z-score standardization to a cached (2944, 1920) uint16 image."""
    img = np.asarray(arr_u16, dtype=np.float32).copy()
    img -= np.mean(img)
    img /= np.maximum(np.std(img), np.float32(1e-5))
    return img


def save_uint16_png_atomic(arr_u16: np.ndarray, target_path: Path, compression: int = 1) -> int:
    """Atomically write a lossless 16-bit PNG via tmp file + fsync + replace; returns file size."""
    arr = np.asarray(arr_u16)
    if arr.dtype != np.uint16 or arr.ndim != 2:
        raise ValueError(f"Expected 2D uint16 image, got shape={arr.shape}, dtype={arr.dtype}")
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {target_path}")
    raw_bytes = buf.tobytes()
    tmp_path = target_path.with_suffix(".png.tmp")
    with open(tmp_path, "wb") as f:
        f.write(raw_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)
    return len(raw_bytes)


def _canonical_nyukat_view_slot(view_slot: str) -> str:
    """Map supplemental RSNA view labels (e.g. L-ML, R-LM, L-AT) to canonical nyukat slots."""
    side, pos = str(view_slot).strip().upper().split("-", 1)
    if pos in ("CC", "MLO"):
        return f"{side}-{pos}"
    if pos in ("ML", "LM", "LMO", "SIO"):
        return f"{side}-MLO"
    return f"{side}-CC"


def preprocess_uint16_for_nyukat(
    arr_u16: np.ndarray,
    view_slot: str,
    target_size: Tuple[int, int],
    nyukat_mods: Dict[str, Any],
    augmentation: bool = False,
    rng: Optional[np.random.Generator] = None,
    max_crop_noise: Tuple[int, int] = (0, 0),
    max_crop_size_noise: int = 0,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run exact NYU/GMIC connected-component crop, flip, optimal center, window crop, and z-score."""
    g_crop = nyukat_mods["crop_mammogram"]
    g_load = nyukat_mods["loading"]
    g_coc = nyukat_mods["calc_optimal_centers"]

    proxy_view = _canonical_nyukat_view_slot(view_slot)
    side = proxy_view.split("-")[0]
    fallback_used = False
    try:
        cinfo = g_crop.crop_img_from_largest_connected(
            arr_u16,
            g_crop.image_orientation("NO", side),
            True,
            100,
            50,
            1.0 / 3.0,
        )
        top, bot, left, right = cinfo[0]
        cropped_u16 = arr_u16[top:bot, left:right]
        cropped_flip_u16 = g_load.flip_image(cropped_u16, proxy_view, "NO")
        if "MLO" in proxy_view:
            tlbr = g_coc.get_bottomrightmost_pixel_constraint(
                rightmost_x=cinfo[1][1],
                bottommost_y=cinfo[2][0],
            )
        else:
            tlbr = g_coc.get_rightmost_pixel_constraint(rightmost_x=cinfo[1][1])
        oc = g_coc.get_image_optimal_window_info(
            cropped_flip_u16,
            com=np.array(cropped_flip_u16.shape) // 2,
            window_dim=np.array(target_size),
            tl_br_constraint=tlbr,
        )
        best_center = (oc["best_center_y"], oc["best_center_x"])
        cropped_flip_f32 = cropped_flip_u16.astype(np.float32)

        # Call random_augmentation_best_center with exact target_size
        np_rng = np.random.RandomState(0) if rng is None else rng
        if "augmentations" in dir(g_load):
            aug_mod = g_load.augmentations
        else:
            raise RuntimeError("Missing augmentations module in nyukat loading")

        if augmentation:
            proc, _ = aug_mod.random_augmentation_best_center(
                image=cropped_flip_f32,
                input_size=target_size,
                random_number_generator=np_rng,
                max_crop_noise=max_crop_noise,
                max_crop_size_noise=max_crop_size_noise,
                auxiliary_image=None,
                best_center=best_center,
                view=proxy_view,
            )
        else:
            proc, _ = aug_mod.random_augmentation_best_center(
                image=cropped_flip_f32,
                input_size=target_size,
                random_number_generator=np_rng,
                max_crop_noise=(0, 0),
                max_crop_size_noise=0,
                auxiliary_image=None,
                best_center=best_center,
                view=proxy_view,
            )
        proc = proc.copy()
        if normalize:
            g_load.standard_normalize_single_image(proc)
        else:
            proc = np.round(np.clip(proc, 0.0, 65535.0)).astype(np.uint16)
    except Exception as exc:
        # Preserve 1:1 physical pixel scale (no cv2.resize); extract/pad target window at native COM
        fallback_used = True
        cropped_flip_u16 = g_load.flip_image(arr_u16, proxy_view, "NO")
        nz_y, nz_x = np.nonzero(cropped_flip_u16 > 0)
        if len(nz_y) > 0:
            com_pt = np.array([int(np.mean(nz_y)), int(np.mean(nz_x))], dtype=np.int64)
        else:
            com_pt = np.array(cropped_flip_u16.shape, dtype=np.int64) // 2
        try:
            oc = g_coc.get_image_optimal_window_info(
                cropped_flip_u16,
                com=com_pt,
                window_dim=np.array(target_size),
                tl_br_constraint=None,
            )
            best_center = (int(oc["best_center_y"]), int(oc["best_center_x"]))
        except Exception:
            best_center = (int(com_pt[0]), int(com_pt[1]))
        aug_mod = g_load.augmentations
        proc, _ = aug_mod.random_augmentation_best_center(
            image=cropped_flip_u16.astype(np.float32),
            input_size=target_size,
            random_number_generator=np.random.RandomState(0) if rng is None else rng,
            max_crop_noise=(0, 0),
            max_crop_size_noise=0,
            auxiliary_image=None,
            best_center=best_center,
            view=proxy_view,
        )
        proc = proc.copy()
        if normalize:
            g_load.standard_normalize_single_image(proc)
        else:
            proc = np.round(np.clip(proc, 0.0, 65535.0)).astype(np.uint16)
        return proc, {
            "crop_fallback": True,
            "crop_error": str(exc),
            "fallback_strategy": "native_scale_optimal_window_no_resize",
            "nyukat_proxy_view": proxy_view,
            "best_center": [int(best_center[0]), int(best_center[1])],
        }

    return proc, {
        "crop_fallback": fallback_used,
        "crop_error": "",
        "nyukat_proxy_view": proxy_view,
        "best_center": [int(best_center[0]), int(best_center[1])],
    }


def build_gmic_model(
    checkpoint_path: Path,
    model_index: str = "1",
    device: Optional[torch.device] = None,
    gmic_repo_dir: Path = GMIC_REPO_DEFAULT,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Construct official GMIC model and verify strict checkpoint compatibility."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mods = _ensure_gmic_importable(gmic_repo_dir)
    params = {
        "device_type": "gpu" if device.type == "cuda" else "cpu",
        "gpu_number": device.index or 0,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "post_processing_dim": 256,
        "num_classes": 2,
        "use_v1_global": False,
        "percent_t": GMIC_PERCENT_T_MAP[str(model_index)],
    }
    model = mods["gmic"].GMIC(params)
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected != ["shared_rep_filter.weight"]:
        raise RuntimeError(
            f"Unexpected GMIC checkpoint keys: missing={missing}, unexpected={unexpected}"
        )
    model = model.to(device)
    model.eval()
    return model, mods


def score_gmic_preprocessed_tensor(
    model: torch.nn.Module,
    img_2d: np.ndarray,
    device: torch.device,
) -> Tuple[float, float]:
    """Run GMIC inference on a single (2944, 1920) normalized float32 image and return (benign, malignant)."""
    x = torch.from_numpy(np.expand_dims(np.expand_dims(img_2d, 0), 0).copy()).to(device)
    with torch.no_grad():
        out = model(x).float().cpu().numpy()[0]
    return float(out[0]), float(out[1])
