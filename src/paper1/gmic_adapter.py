"""Official GMIC (nyukat/GMIC) Adapter and Native 2944x1920 Preprocessing Pipeline.

Preserves exact architecture, weights (sample_model_1.p, MODEL_INDEX='1'),
cropping, optimal-center selection, and malignant_pred extraction from:
    https://github.com/nyukat/GMIC (AGPL-3.0)
"""

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import decode_dicom_array
from src.evaluation.stage_c_eval import assert_no_forbidden_columns, compute_sha256_file

GMIC_REPO_DEFAULT = Path(__file__).resolve().parent.parent.parent / "external" / "GMIC"
GMIC_INPUT_SIZE = (2944, 1920)
GMIC_PERCENT_T_MAP = {"1": 0.02, "2": 0.03, "3": 0.03, "4": 0.05, "5": 0.1}
GMIC_MODEL_ID = "gmic_pretrained"
GMIC_CKPT_SHA256 = "3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991"
GMIC_PREPROCESS_VERSION = "paper1_gmic_uint16_v1"


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
    from src.data.preprocessing import decode_mammogram_array

    arr_f32, phot, lat_dcm = decode_mammogram_array(dicom_path, apply_luts=True)
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


def compute_item_ids_sha256(item_ids: Sequence[str]) -> str:
    """Compute deterministic SHA-256 digest of an ordered sequence of image IDs."""
    payload = json.dumps([str(i) for i in item_ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_gmic_shard_payload(
    payload: Dict[str, Any],
    *,
    expected_item_ids: Sequence[str],
    batch_size: int,
    split_name: str,
    shard_idx: int,
    num_shards: int,
    model_id: str = GMIC_MODEL_ID,
    checkpoint_sha256: str = GMIC_CKPT_SHA256,
    preprocess_version: str = GMIC_PREPROCESS_VERSION,
    preprocess_manifest_sha256: Optional[str] = None,
) -> Tuple[bool, str]:
    """Verify resumed shard provenance (batch size, checkpoint hash, preprocess version, exact item IDs, finite scores)."""
    if not isinstance(payload, dict):
        return False, "payload_not_dict"
    if payload.get("model_id") != model_id:
        return False, f"model_id_mismatch:{payload.get('model_id')}"
    if str(payload.get("checkpoint_sha256", "")).lower() != str(checkpoint_sha256).lower():
        return False, f"checkpoint_sha256_mismatch:{payload.get('checkpoint_sha256')}"
    if payload.get("preprocess_version") != preprocess_version:
        return False, f"preprocess_version_mismatch:{payload.get('preprocess_version')}"
    if preprocess_manifest_sha256 and payload.get("preprocess_manifest_sha256") != preprocess_manifest_sha256:
        return False, f"preprocess_manifest_sha256_mismatch:{payload.get('preprocess_manifest_sha256')}"
    if payload.get("split_name") != split_name:
        return False, f"split_name_mismatch:{payload.get('split_name')}"
    if payload.get("shard_idx") != int(shard_idx) or payload.get("num_shards") != int(num_shards):
        return False, f"shard_spec_mismatch:{payload.get('shard_idx')}/{payload.get('num_shards')}"
    if payload.get("batch_size") != int(batch_size):
        return False, f"batch_size_mismatch:{payload.get('batch_size')}!=expected_{int(batch_size)}"

    exp_ids = [str(i) for i in expected_item_ids]
    exp_hash = compute_item_ids_sha256(exp_ids)
    if payload.get("expected_item_ids_sha256") != exp_hash:
        return False, "expected_item_ids_sha256_mismatch"

    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(exp_ids):
        return False, f"records_len_mismatch:{len(records) if isinstance(records, list) else None}!={len(exp_ids)}"
    rec_ids = [str(r.get("item_id", "")) for r in records]
    if rec_ids != exp_ids:
        return False, "record_item_ids_order_or_set_mismatch"

    for idx, rec in enumerate(records):
        b = rec.get("benign_pred")
        m = rec.get("malignant_pred")
        if b is None or m is None or not (math.isfinite(float(b)) and math.isfinite(float(m))):
            return False, f"non_finite_prediction_at_idx_{idx}"
    return True, ""


def finalize_and_lock_gmic_predictions(
    split_name: str,
    shard_outputs: Sequence[Dict[str, Any]],
    expected_image_df: pd.DataFrame,
    *,
    item_col: str,
    case_col: str,
    case_out_col: str,
    batch_size: int,
    model_id: str = GMIC_MODEL_ID,
    checkpoint_file: str = "sample_model_1.p",
    checkpoint_sha256: str = GMIC_CKPT_SHA256,
    preprocess_version: str = GMIC_PREPROCESS_VERSION,
    out_dir: Path,
) -> Dict[str, Any]:
    """Validate 100% coverage/uniqueness across shards, aggregate max-view case scores, and SHA-256-lock outputs."""
    assert_no_forbidden_columns(expected_image_df.columns, context=f"gmic_lock_{split_name}_manifest")
    exp_df = expected_image_df.copy()
    exp_df[item_col] = exp_df[item_col].astype(str)
    exp_df[case_col] = exp_df[case_col].astype(str)
    exp_ids = set(exp_df[item_col].tolist())
    if len(exp_ids) != len(exp_df):
        raise ValueError(f"{split_name} expected manifest contains duplicate {item_col} rows")

    all_recs: List[Dict[str, Any]] = []
    all_fbs: List[Dict[str, Any]] = []
    shard_meta: List[Dict[str, Any]] = []
    preprocess_manifest_shas = set()
    for o in sorted(shard_outputs, key=lambda x: int(x["shard_idx"])):
        if o.get("model_id") != model_id:
            raise RuntimeError(f"Shard {o.get('shard_idx')} model_id mismatch: {o.get('model_id')} != {model_id}")
        if str(o.get("checkpoint_sha256", "")).lower() != checkpoint_sha256.lower():
            raise RuntimeError(f"Shard {o.get('shard_idx')} checkpoint_sha256 mismatch")
        if o.get("preprocess_version") != preprocess_version:
            raise RuntimeError(f"Shard {o.get('shard_idx')} preprocess_version mismatch: {o.get('preprocess_version')}")
        if o.get("batch_size") != int(batch_size):
            raise RuntimeError(f"Shard {o.get('shard_idx')} batch_size mismatch: {o.get('batch_size')} != {batch_size}")
        if o.get("preprocess_manifest_sha256"):
            preprocess_manifest_shas.add(str(o["preprocess_manifest_sha256"]))
        all_recs.extend(o["records"])
        all_fbs.extend(o.get("fallback_logs", []))
        shard_meta.append(
            {
                "shard_idx": int(o["shard_idx"]),
                "num_shards": int(o["num_shards"]),
                "n_images": len(o["records"]),
                "n_fallback_views": int(o.get("n_fallback_views", len(o.get("fallback_logs", [])))),
                "elapsed_sec": o.get("elapsed_sec"),
                "container_wall_sec": o.get("container_wall_sec"),
                "model_load_sec": o.get("model_load_sec"),
                "nvme_stage_sec": o.get("nvme_stage_sec"),
                "decode_norm_sec": o.get("decode_norm_sec"),
                "gpu_infer_sec": o.get("gpu_infer_sec"),
                "estimated_cost_usd": o.get("estimated_cost_usd"),
                "peak_vram_gb": o.get("peak_vram_gb"),
                "host_rss_gb": o.get("host_rss_gb"),
                "expected_item_ids_sha256": o.get("expected_item_ids_sha256"),
                "resumed_from_cache": bool(o.get("resumed_from_cache", False)),
            }
        )

    rec_df = pd.DataFrame(all_recs)
    assert_no_forbidden_columns(rec_df.columns, context=f"gmic_lock_{split_name}_records")
    if len(rec_df) != len(exp_df):
        raise RuntimeError(f"{split_name} record count mismatch: {len(rec_df)} != {len(exp_df)}")
    if rec_df["item_id"].duplicated().any():
        raise RuntimeError(f"{split_name} contains duplicate item_id predictions")
    if set(rec_df["item_id"].astype(str)) != exp_ids:
        raise RuntimeError(f"{split_name} item_id set does not match expected manifest")
    if not (np.all(np.isfinite(rec_df["benign_pred"].values)) and np.all(np.isfinite(rec_df["malignant_pred"].values))):
        raise RuntimeError(f"{split_name} contains non-finite predictions")

    rec_df = rec_df.sort_values(["case_id", "item_id"]).reset_index(drop=True)
    case_df = (
        rec_df.groupby("case_id", as_index=False)
        .agg(
            score=("malignant_pred", "max"),
            max_benign_pred=("benign_pred", "max"),
            n_views=("item_id", "count"),
            n_crop_fallbacks=("crop_fallback", "sum"),
        )
        .rename(columns={"case_id": case_out_col})
    )
    case_df["n_crop_fallbacks"] = case_df["n_crop_fallbacks"].astype(int)
    case_df["has_crop_fallback"] = case_df["n_crop_fallbacks"] > 0
    if len(case_df) != int(exp_df[case_col].nunique()):
        raise RuntimeError(f"{split_name} incomplete case coverage: {len(case_df)} != {exp_df[case_col].nunique()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    img_json_p = out_dir / f"gmic_{split_name}_image_scores.json"
    img_csv_p = out_dir / f"gmic_{split_name}_image_scores.csv"
    case_csv_p = out_dir / f"gmic_{split_name}_case_scores.csv"
    fb_json_p = out_dir / f"gmic_{split_name}_fallback_log.json"
    lock_json_p = out_dir / f"gmic_{split_name}_provenance_lock.json"

    img_payload = {
        "schema_version": "paper1_gmic_image_scores_v2",
        "model_id": model_id,
        "checkpoint_file": checkpoint_file,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocess_version": preprocess_version,
        "preprocess_manifest_sha256s": sorted(preprocess_manifest_shas),
        "split_name": split_name,
        "batch_size": int(batch_size),
        "n_images": int(len(rec_df)),
        "n_cases": int(len(case_df)),
        "records": rec_df.to_dict(orient="records"),
    }
    img_json_p.write_text(json.dumps(img_payload, indent=2), encoding="utf-8")
    rec_df.to_csv(img_csv_p, index=False)
    case_df.to_csv(case_csv_p, index=False)

    fb_df = pd.DataFrame(all_fbs)
    fb_payload = {
        "split_name": split_name,
        "n_images": int(len(rec_df)),
        "n_fallback_views": int(len(all_fbs)),
        "n_fallback_cases": int(fb_df["case_id"].nunique()) if len(all_fbs) else 0,
        "by_view_slot": fb_df["view_slot"].value_counts().to_dict() if len(all_fbs) else {},
        "by_photometric": fb_df["photometric"].value_counts().to_dict() if len(all_fbs) else {},
        "by_error": fb_df["crop_error"].value_counts().to_dict() if len(all_fbs) else {},
        "fallbacks": sorted(all_fbs, key=lambda x: (str(x["case_id"]), str(x["item_id"]))),
    }
    fb_json_p.write_text(json.dumps(fb_payload, indent=2), encoding="utf-8")

    mal_arr = rec_df["malignant_pred"].values.astype(np.float64)
    case_arr = case_df["score"].values.astype(np.float64)
    lock_manifest = {
        "schema_version": "paper1_gmic_step2_lock_v1",
        "model_id": model_id,
        "checkpoint_file": checkpoint_file,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocess_version": preprocess_version,
        "preprocess_manifest_sha256s": sorted(preprocess_manifest_shas),
        "split_name": split_name,
        "batch_size": int(batch_size),
        "label_quarantine_enforced": True,
        "n_unique_images": int(rec_df["item_id"].nunique()),
        "n_unique_cases": int(len(case_df)),
        "all_scores_finite": bool(np.all(np.isfinite(mal_arr)) and np.all(np.isfinite(case_arr))),
        "image_malignant_summary_label_free": {
            "min": round(float(np.min(mal_arr)), 6),
            "median": round(float(np.median(mal_arr)), 6),
            "max": round(float(np.max(mal_arr)), 6),
        },
        "case_malignant_summary_label_free": {
            "min": round(float(np.min(case_arr)), 6),
            "median": round(float(np.median(case_arr)), 6),
            "max": round(float(np.max(case_arr)), 6),
        },
        "n_fallback_views": int(len(all_fbs)),
        "n_cases_with_fallback": int(case_df["has_crop_fallback"].sum()),
        "total_estimated_cost_usd": round(sum(float(s["estimated_cost_usd"] or 0.0) for s in shard_meta), 5),
        "shards": shard_meta,
        "locked_files_sha256": {
            img_json_p.name: compute_sha256_file(img_json_p),
            img_csv_p.name: compute_sha256_file(img_csv_p),
            case_csv_p.name: compute_sha256_file(case_csv_p),
            fb_json_p.name: compute_sha256_file(fb_json_p),
        },
    }
    lock_json_p.write_text(json.dumps(lock_manifest, indent=2), encoding="utf-8")
    return lock_manifest
