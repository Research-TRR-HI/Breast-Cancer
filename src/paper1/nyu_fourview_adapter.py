"""Official NYU Four-View CNN (nyukat/breast_cancer_classifier) Adapter.

Preserves exact image-only architecture (SplitBreastModel), checkpoint
(sample_image_model.p), native CC (2677x1942) & MLO (2974x1748) inputs,
10-epoch TTA averaging (seed=0), and max(left_malignant, right_malignant)
examination scoring from:
    https://github.com/nyukat/breast_cancer_classifier (AGPL-3.0)
"""

import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch

from src.evaluation.stage_c_eval import assert_no_forbidden_columns, compute_sha256_file
from src.paper1.gmic_adapter import prepare_dicom_uint16_for_nyukat

NYU_REPO_DEFAULT = Path(__file__).resolve().parent.parent.parent / "external" / "breast_cancer_classifier"
NYU_INPUT_SIZES = {
    "L-CC": (2677, 1942),
    "R-CC": (2677, 1942),
    "L-MLO": (2974, 1748),
    "R-MLO": (2974, 1748),
}
NYU_VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")
NYU_CACHE_VERSION = "paper1_nyu_uint16_v1"


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
    include_uint16: bool = False,
) -> Dict[str, Any]:
    """Decode DICOM, apply NYU connected-component crop + optimal center, and return flipped float32 crop + best_center."""
    laterality = view_slot.split("-")[0]
    arr_u16, dcm_meta = prepare_dicom_uint16_for_nyukat(dicom_path, laterality=laterality)
    n_crop = nyu_mods["crop_mammogram"]
    n_load = nyu_mods["loading"]
    n_coc = nyu_mods["calc_optimal_centers"]
    target_size = NYU_INPUT_SIZES[view_slot]

    fallback_used = False
    crop_window = [0, int(arr_u16.shape[0]), 0, int(arr_u16.shape[1])]
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
        crop_window = [int(top), int(bot), int(left), int(right)]
        cropped_u16 = arr_u16[top:bot, left:right]
        cropped_flip_u16 = np.ascontiguousarray(n_load.flip_image(cropped_u16, view_slot, "NO"))
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
        cropped_flip_u16 = np.ascontiguousarray(n_load.flip_image(arr_u16, view_slot, "NO"))
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

    dcm_meta["raw_shape"] = dcm_meta.get("raw_shape") or [int(arr_u16.shape[0]), int(arr_u16.shape[1])]
    dcm_meta["crop_window"] = crop_window
    out: Dict[str, Any] = {
        "image_f32": img_f32,
        "best_center": best_center,
        "crop_fallback": fallback_used,
        "meta": dcm_meta,
    }
    if include_uint16:
        out["image_u16"] = cropped_flip_u16
    return out


def validate_and_load_nyu_4view_manifest(split_name: str, manifest_csv: Path) -> pd.DataFrame:
    """Load label-free 4-view manifest and reject any missing views, extra views, or duplicate IDs."""
    df = pd.read_csv(manifest_csv, dtype=str)
    assert_no_forbidden_columns(df.columns, context=f"nyu_4view_{split_name}")

    if split_name == "rsna_val_calib":
        case_col, item_col, path_col = "patient_id", "image_id", "rel_path"
    elif split_name == "nlbs_4view":
        case_col, item_col, path_col = "StudyInstanceUID", "image_uid", "dicom_rel_path"
    else:
        raise ValueError(f"Unsupported NYU split_name={split_name!r}")

    for req in (case_col, item_col, "view_slot", path_col):
        if req not in df.columns:
            raise ValueError(f"{manifest_csv} missing required column {req!r}")
        if df[req].isna().any() or (df[req].str.strip() == "").any():
            raise ValueError(f"{manifest_csv} has empty values in {req!r}")

    if df[item_col].duplicated().any():
        raise ValueError(f"{split_name}: duplicate {item_col} found in {manifest_csv}")
    if df.duplicated(subset=[case_col, "view_slot"]).any():
        raise ValueError(f"{split_name}: duplicate ({case_col}, view_slot) found in {manifest_csv}")
    bad_views = set(df["view_slot"]) - set(NYU_VIEWS)
    if bad_views:
        raise ValueError(f"{split_name}: invalid view_slot entries {sorted(bad_views)}")

    counts = df.groupby(case_col)["view_slot"].nunique()
    incomplete = counts[counts != len(NYU_VIEWS)]
    if len(incomplete) > 0:
        raise ValueError(
            f"{split_name}: {len(incomplete)} exams do not have all 4 standard views (e.g. {incomplete.index[:3].tolist()})"
        )

    view_order = {v: idx for idx, v in enumerate(NYU_VIEWS)}
    out = df.copy()
    out["_v_ord"] = out["view_slot"].map(view_order)
    out = out.sort_values([case_col, "_v_ord", item_col], kind="mergesort").drop(columns=["_v_ord"]).reset_index(drop=True)
    return out


def load_nyu_cached_view(
    png_path: Path,
    best_center: Tuple[int, int],
    expected_shape: Optional[Tuple[int, int]] = None,
    expected_u16_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Load cached lossless uint16 NYU cropped PNG into exact float32 + best_center input dict."""
    arr = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if arr is None or arr.dtype != np.uint16 or arr.ndim != 2:
        raise RuntimeError(f"Invalid NYU cached uint16 PNG: {png_path}")
    if expected_shape is not None and (int(arr.shape[0]), int(arr.shape[1])) != (int(expected_shape[0]), int(expected_shape[1])):
        raise RuntimeError(f"NYU cached PNG shape mismatch for {png_path}: {arr.shape} != {expected_shape}")
    if expected_u16_sha256:
        got_sha = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        if got_sha != expected_u16_sha256:
            raise RuntimeError(f"NYU cached PNG pixel SHA mismatch for {png_path}: {got_sha} != {expected_u16_sha256}")
    return {
        "image_f32": arr.astype(np.float32),
        "best_center": (int(best_center[0]), int(best_center[1])),
    }


def verify_and_load_nyu_resumed_records(
    prog_path: Path,
    png_dir: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Verify existing NYU shard JSONL records against both file_bytes AND disk PNG SHA-256 checksums."""
    done_map: Dict[str, Dict[str, Any]] = {}
    stats = {"total_lines": 0, "valid_resumed": 0, "rejected": 0}
    if not prog_path.exists():
        return done_map, stats

    seen_recs: Dict[str, Dict[str, Any]] = {}
    for line in prog_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stats["total_lines"] += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            stats["rejected"] += 1
            continue
        item_id = str(rec.get("item_id", ""))
        if item_id:
            seen_recs[item_id] = rec

    for item_id, rec in seen_recs.items():
        png_sha = str(rec.get("png_sha256", "")).lower()
        u16_sha = str(rec.get("u16_sha256", "")).lower()
        file_bytes = int(rec.get("file_bytes", 0) or 0)
        p_file = png_dir / f"{item_id}.png"
        if (
            len(png_sha) != 64
            or len(u16_sha) != 64
            or file_bytes <= 0
            or int(rec.get("height", 0) or 0) <= 0
            or int(rec.get("width", 0) or 0) <= 0
            or "best_center_y" not in rec
            or "best_center_x" not in rec
            or not p_file.exists()
            or p_file.stat().st_size != file_bytes
        ):
            stats["rejected"] += 1
            continue
        if compute_sha256_file(p_file).lower() != png_sha:
            stats["rejected"] += 1
            continue
        done_map[item_id] = rec
        stats["valid_resumed"] += 1

    return done_map, stats


def validate_nyu_cli_stage_and_guard(
    stage: str,
    split: str,
    num_shards: int,
    approve_full_build: bool,
) -> List[str]:
    """Validate CLI parameters and block unapproved full-dataset `--stage build` runs."""
    valid_stages = ("pilot", "validate_config", "build", "verify")
    if stage not in valid_stages:
        raise ValueError(f"Unknown --stage={stage!r}; expected one of {valid_stages}")
    if int(num_shards) < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if split == "all":
        splits = ["rsna_val_calib", "nlbs_4view"]
    elif split in ("rsna_val_calib", "nlbs_4view"):
        splits = [split]
    else:
        raise ValueError(f"Unsupported --split={split!r}; expected 'rsna_val_calib', 'nlbs_4view', or 'all'")

    if stage == "build" and not approve_full_build:
        raise RuntimeError(
            "BLOCKED: `--stage build` requires explicit `--approve-full-build` confirmation so no full-dataset "
            "preprocessing job starts unapproved. Re-run with `--stage build --approve-full-build` once ready."
        )
    return splits


def select_representative_pilot_exams(
    rsna_rows: List[Dict[str, Any]],
    nlbs_rows: List[Dict[str, Any]],
    gmic_fb_cases: List[str],
    n_exams_per_split: int = 40,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Select 80 complete 4-view exams (320 imgs) including patient 71, difficult fallbacks, and SIEMENS exams."""
    rsna_by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in rsna_rows:
        rsna_by_case.setdefault(r["case_id"], []).append(r)

    nlbs_by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in nlbs_rows:
        nlbs_by_case.setdefault(r["case_id"], []).append(r)

    rsna_chosen: List[str] = []
    if "71" in rsna_by_case:
        rsna_chosen.append("71")
    for cid in sorted(gmic_fb_cases, key=lambda x: (len(x), x)):
        if cid in rsna_by_case and cid not in rsna_chosen:
            rsna_chosen.append(cid)
        if len(rsna_chosen) >= 15:
            break

    rem_rsna = [c for c in sorted(rsna_by_case.keys(), key=int) if c not in set(rsna_chosen)]
    multi_rsna = [c for c in rem_rsna if any(r.get("n_candidates_in_slot", 1) > 1 for r in rsna_by_case[c])]
    for c in multi_rsna[:5]:
        if len(rsna_chosen) < n_exams_per_split:
            rsna_chosen.append(c)
    still_rem = [c for c in rem_rsna if c not in set(rsna_chosen)]
    step_r = max(1, len(still_rem) // max(1, n_exams_per_split - len(rsna_chosen)))
    for c in still_rem[::step_r]:
        if len(rsna_chosen) >= n_exams_per_split:
            break
        rsna_chosen.append(c)

    nlbs_chosen: List[str] = []
    siemens_cases = sorted(
        [c for c, rows in nlbs_by_case.items() if any(r.get("manufacturer") != "GE MEDICAL SYSTEMS" for r in rows)]
    )
    for c in siemens_cases:
        if len(nlbs_chosen) < n_exams_per_split:
            nlbs_chosen.append(c)
    rem_nlbs = [c for c in sorted(nlbs_by_case.keys()) if c not in set(nlbs_chosen)]
    multi_nlbs = [c for c in rem_nlbs if any(r.get("n_candidates_in_slot", 1) > 1 for r in nlbs_by_case[c])]
    step_m = max(1, len(multi_nlbs) // 12)
    for c in multi_nlbs[::step_m]:
        if len(nlbs_chosen) >= min(n_exams_per_split, len(siemens_cases) + 12):
            break
        nlbs_chosen.append(c)
    still_nlbs = [c for c in rem_nlbs if c not in set(nlbs_chosen)]
    step_n = max(1, len(still_nlbs) // max(1, n_exams_per_split - len(nlbs_chosen)))
    for c in still_nlbs[::step_n]:
        if len(nlbs_chosen) >= n_exams_per_split:
            break
        nlbs_chosen.append(c)

    sel_rsna_rows = [r for c in rsna_chosen for r in rsna_by_case[c]]
    sel_nlbs_rows = [r for c in nlbs_chosen for r in nlbs_by_case[c]]
    meta = {
        "n_rsna_exams": len(rsna_chosen),
        "n_rsna_images": len(sel_rsna_rows),
        "n_rsna_known_difficult_exams": sum(1 for c in rsna_chosen if c == "71" or c in set(gmic_fb_cases)),
        "includes_patient_71": "71" in rsna_chosen,
        "n_nlbs_exams": len(nlbs_chosen),
        "n_nlbs_images": len(sel_nlbs_rows),
        "n_nlbs_siemens_exams": len([c for c in siemens_cases if c in set(nlbs_chosen)]),
        "n_nlbs_multi_candidate_exams": sum(
            1 for c in nlbs_chosen if any(r.get("n_candidates_in_slot", 1) > 1 for r in nlbs_by_case[c])
        ),
    }
    return sel_rsna_rows, sel_nlbs_rows, meta


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
