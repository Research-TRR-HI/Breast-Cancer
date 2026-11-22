"""Official NYU Four-View CNN (nyukat/breast_cancer_classifier) Adapter.

Preserves exact image-only architecture (SplitBreastModel), checkpoint
(sample_image_model.p), native CC (2677x1942) & MLO (2974x1748) inputs,
10-epoch TTA averaging (seed=0), and max(left_malignant, right_malignant)
examination scoring from:
    https://github.com/nyukat/breast_cancer_classifier (AGPL-3.0)
"""

import importlib
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

from src.paper1.gmic_adapter import prepare_dicom_uint16_for_nyukat

NYU_REPO_DEFAULT = Path(__file__).resolve().parent.parent.parent / "external" / "breast_cancer_classifier"
NYU_INPUT_SIZES = {
    "L-CC": (2677, 1942),
    "R-CC": (2677, 1942),
    "L-MLO": (2974, 1748),
    "R-MLO": (2974, 1748),
}
NYU_VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")


def _ensure_nyu_importable(nyu_repo_dir: Path = NYU_REPO_DEFAULT) -> Dict[str, Any]:
    """Temporarily import nyukat/breast_cancer_classifier modules without polluting sys.modules['src']."""
    if "h5py" not in sys.modules:
        try:
            importlib.import_module("h5py")
        except ImportError:
            sys.modules["h5py"] = types.ModuleType("h5py")

    nyu_str = str(Path(nyu_repo_dir).resolve())
    saved_src = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in list(saved_src.keys()):
        del sys.modules[k]
    sys.path.insert(0, nyu_str)
    try:
        n_const = importlib.import_module("src.constants")
        n_crop = importlib.import_module("src.cropping.crop_mammogram")
        n_load = importlib.import_module("src.data_loading.loading")
        n_models = importlib.import_module("src.modeling.models")
        n_rm = importlib.import_module("src.modeling.run_model")
        n_coc = importlib.import_module("src.optimal_centers.calc_optimal_centers")
        return {
            "crop_mammogram": n_crop,
            "loading": n_load,
            "models": n_models,
            "run_model": n_rm,
            "calc_optimal_centers": n_coc,
            "VIEWS": n_const.VIEWS,
            "VIEWANGLES": n_const.VIEWANGLES,
            "LABELS": n_const.LABELS,
            "MODELMODES": n_const.MODELMODES,
        }
    finally:
        if sys.path[0] == nyu_str:
            sys.path.pop(0)
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(saved_src)


def build_nyu_fourview_model(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
    nyu_repo_dir: Path = NYU_REPO_DEFAULT,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load official NYU image-only SplitBreastModel(1) strictly from sample_image_model.p."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mods = _ensure_nyu_importable(nyu_repo_dir)
    model = mods["models"].SplitBreastModel(input_channels=1)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"NYU SplitBreastModel key mismatch: missing={missing}, unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    return model, mods


def extract_nyu_cropped_view(
    dicom_path: Path,
    view_slot: str,
    nyu_mods: Dict[str, Any],
) -> Dict[str, Any]:
    """Decode DICOM, apply NYU connected-component crop + optimal center, and return flipped float32 crop + best_center."""
    laterality = view_slot.split("-")[0]
    arr_u16, dcm_meta = prepare_dicom_uint16_for_nyukat(dicom_path, laterality=laterality)
    n_crop = nyu_mods["crop_mammogram"]
    n_load = nyu_mods["loading"]
    n_coc = nyu_mods["calc_optimal_centers"]
    target_size = NYU_INPUT_SIZES[view_slot]

    fallback_used = False
    try:
        cinfo = n_crop.crop_img_from_largest_connected(
            arr_u16,
            n_crop.image_orientation("NO", laterality),
            True,
            100,
            50,
            1.0 / 3.0,
        )
        top, bot, left, right = cinfo[0]
        cropped_u16 = arr_u16[top:bot, left:right]
        cropped_flip_u16 = n_load.flip_image(cropped_u16, view_slot, "NO")
        if "MLO" in view_slot:
            tlbr = n_coc.get_bottomrightmost_pixel_constraint(
                rightmost_x=cinfo[1][1],
                bottommost_y=cinfo[2][0],
            )
        else:
            tlbr = n_coc.get_rightmost_pixel_constraint(rightmost_x=cinfo[1][1])
        oc = n_coc.get_image_optimal_window_info(
            cropped_flip_u16,
            com=np.array(cropped_flip_u16.shape) // 2,
            window_dim=np.array(target_size),
            tl_br_constraint=tlbr,
        )
        best_center = (int(oc["best_center_y"]), int(oc["best_center_x"]))
        img_f32 = cropped_flip_u16.astype(np.float32)
    except Exception as exc:
        # Preserve 1:1 physical pixel scale (NEVER cv2.resize across raw -> model dimensions).
        # Compute native-scale foreground center of mass + NYU optimal-density window on uncropped view.
        fallback_used = True
        cropped_flip_u16 = n_load.flip_image(arr_u16, view_slot, "NO")
        nz_y, nz_x = np.nonzero(cropped_flip_u16 > 0)
        if len(nz_y) > 0:
            com_pt = np.array([int(np.mean(nz_y)), int(np.mean(nz_x))], dtype=np.int64)
        else:
            com_pt = np.array(cropped_flip_u16.shape, dtype=np.int64) // 2
        try:
            oc = n_coc.get_image_optimal_window_info(
                cropped_flip_u16,
                com=com_pt,
                window_dim=np.array(target_size),
                tl_br_constraint=None,
            )
            best_center = (int(oc["best_center_y"]), int(oc["best_center_x"]))
        except Exception:
            best_center = (int(com_pt[0]), int(com_pt[1]))
        img_f32 = cropped_flip_u16.astype(np.float32)
        dcm_meta["crop_error"] = str(exc)
        dcm_meta["fallback_strategy"] = "native_scale_optimal_window_no_resize"
        dcm_meta["raw_shape"] = [int(arr_u16.shape[0]), int(arr_u16.shape[1])]

    return {
        "image_f32": img_f32,
        "best_center": best_center,
        "crop_fallback": fallback_used,
        "meta": dcm_meta,
    }


def score_nyu_fourview_exam(
    model: torch.nn.Module,
    nyu_mods: Dict[str, Any],
    view_crops: Dict[str, Dict[str, Any]],
    device: torch.device,
    num_tta_epochs: int = 10,
    seed: int = 0,
) -> Dict[str, float]:
    """Score one 4-view exam using official 10-epoch TTA averaging and pandas-2.3-safe CC/MLO mean."""
    n_load = nyu_mods["loading"]
    rng = np.random.RandomState(seed)
    use_aug = num_tta_epochs > 1

    # Batch all TTA draws into a single GPU forward pass when possible
    batch_views: Dict[str, List[np.ndarray]] = {v: [] for v in NYU_VIEWS}
    for _ in range(num_tta_epochs):
        for v in NYU_VIEWS:
            cropped, _ = n_load.augment_and_normalize_image(
                image=view_crops[v]["image_f32"],
                auxiliary_image=None,
                view=v,
                best_center=view_crops[v]["best_center"],
                random_number_generator=rng,
                augmentation=use_aug,
                max_crop_noise=(100, 100),
                max_crop_size_noise=100,
            )
            batch_views[v].append(cropped[np.newaxis, ...])

    tensor_batch = {
        v: torch.from_numpy(np.stack(batch_views[v], axis=0).copy()).to(device)
        for v in NYU_VIEWS
    }
    with torch.no_grad():
        out = model(tensor_batch)
    cc_probs = np.exp(out["CC"].cpu().numpy())[:, :, 1]  # [N, 4] -> LB, RB, LM, RM
    mlo_probs = np.exp(out["MLO"].cpu().numpy())[:, :, 1]
    mean_4 = np.mean(0.5 * (cc_probs + mlo_probs), axis=0)

    lb, rb, lm, rm = [float(x) for x in mean_4]
    return {
        "left_benign": lb,
        "right_benign": rb,
        "left_malignant": lm,
        "right_malignant": rm,
        "exam_score": max(lm, rm),
    }
