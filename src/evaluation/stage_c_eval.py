"""Stage C Strict Zero-Target-Label NLBS Inference Locking & Evaluation.

Protocol guarantees:
    1. Target-Label Quarantine:
       `build_label_free_inference_manifest` creates `datasets/nlbs/inference_manifest_stage_c.csv`
       containing ONLY non-label technical metadata + physical DICOM paths, explicitly
       forbidding Cancer, exam_cancer, n_cancer_images, Hard to State Negative, and cohort_group.
    2. Separate Prediction Locking vs. Evaluation:
       `lock_stage_c_predictions` writes frozen `StudyInstanceUID` exam-level predictions,
       four frozen-threshold dismissal decisions, preprocessing failure/fallback counts,
       and cryptographic SHA-256 hashes BEFORE any NLBS outcome file is accessed.
       `verify_and_evaluate_locked_predictions` refuses to join `exam_manifest.csv` unless
       every SHA-256 digest and frozen threshold matches `locked_predictions_manifest.json`.
    3. Honest Repeat-Exam Denominators & Cluster-Aware CIs:
       Deduplicates 71 cross-folder duplicate `StudyInstanceUID`s -> N = 6,359 unique exams
       (preserving all distinct repeat exams) and computes patient-cluster bootstrap 95% CIs
       by resampling the N_clusters = 5,942 `split_cluster_id` connected components.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.trust_model import aggregate_case_scores
from src.training.calibration import clopper_pearson_upper_bound, evaluate_dismissal_at_threshold

FORBIDDEN_LABEL_COLUMNS: Tuple[str, ...] = (
    "Cancer",
    "cancer",
    "exam_cancer",
    "n_cancer_images",
    "Hard to State Negative",
    "hard_to_state_negative",
    "cohort_group",
)

LABEL_FREE_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "image_uid",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "exam_id",
    "case_id",
    "visit_idx",
    "split_cluster_id",
    "laterality",
    "view_position",
    "manufacturer",
    "manufacturer_model",
    "photometric_interpretation",
    "dicom_rel_path",
)

FROZEN_RSNA_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ce": {
        "target_95": 0.12445068359375,
        "target_98": 0.09619140625,
    },
    "trust": {
        "target_95": 0.1954345703125,
        "target_98": 0.1795654296875,
    },
}

FROZEN_STAGE_B_CHECKPOINT_SHA256: Dict[str, str] = {
    "ce": "adf1443455ee1b507485ee4b7f64932d5cf6f59bb6afcf11b9f415c634372d22",
    "trust": "63388d790ce9cf8a8287adc7724358ced392a69b332ce1a5e46972750faa51cb",
}

FROZEN_BACKBONE_SHA256 = "0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c"


def compute_sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file using streaming 1 MiB reads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_json(obj: Any) -> str:
    """Compute canonical SHA-256 digest of a JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def assert_no_forbidden_columns(columns: Iterable[str], context: str = "dataframe") -> None:
    """Abort immediately if any label-revealing column exists in a quarantined artifact."""
    col_set = set(columns)
    present = [c for c in FORBIDDEN_LABEL_COLUMNS if c in col_set]
    if present:
        raise ValueError(
            f"TARGET-LABEL QUARANTINE VIOLATION in {context}: forbidden columns present: {present}"
        )


def verify_frozen_thresholds_against_stage_b(
    rsna_ce_json: Path,
    rsna_trust_json: Path,
) -> Dict[str, Dict[str, float]]:
    """Verify hardcoded frozen RSNA thresholds match Stage-B committed artifacts exactly."""
    ce_data = json.loads(Path(rsna_ce_json).read_text(encoding="utf-8"))
    tr_data = json.loads(Path(rsna_trust_json).read_text(encoding="utf-8"))

    checks = [
        ("ce", "target_95", float(ce_data["target_95"]["threshold"])),
        ("ce", "target_98", float(ce_data["target_98"]["threshold"])),
        ("trust", "target_95", float(tr_data["target_95"]["threshold"])),
        ("trust", "target_98", float(tr_data["target_98"]["threshold"])),
    ]
    for model_key, op_key, disk_val in checks:
        expected = FROZEN_RSNA_THRESHOLDS[model_key][op_key]
        if abs(disk_val - expected) > 1e-15:
            raise ValueError(
                f"Frozen threshold mismatch for {model_key}.{op_key}: "
                f"disk={disk_val!r} != expected={expected!r}"
            )
    return {k: dict(v) for k, v in FROZEN_RSNA_THRESHOLDS.items()}


def build_label_free_inference_manifest(
    source_manifest_csv: Path,
    output_manifest_csv: Path,
) -> Dict[str, Any]:
    """Construct the strictly label-free Stage C inference manifest from datasets/nlbs/manifest.csv.

    Drops all label-revealing columns (`Cancer`, `Hard to State Negative`, `cohort_group`, etc.)
    and sorts rows by `(StudyInstanceUID, SeriesInstanceUID, rel_path)` rather than cohort folder.
    """
    src = pd.read_csv(source_manifest_csv)
    if len(src) == 0:
        raise ValueError(f"Empty source manifest: {source_manifest_csv}")

    required_src = (
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "exam_id",
        "case_id",
        "visit_idx",
        "split_cluster_id",
        "Image Laterality",
        "View Position",
        "Manufacturer",
        "ManufacturerModelName",
        "PhotometricInterpretation",
        "rel_path",
    )
    missing = [c for c in required_src if c not in src.columns]
    if missing:
        raise ValueError(f"Source NLBS manifest missing required columns: {missing}")

    if src["StudyInstanceUID"].isna().any() or src["rel_path"].isna().any():
        raise ValueError("Source NLBS manifest contains missing StudyInstanceUID or rel_path values.")

    lats = src["Image Laterality"].astype(str).str.strip().str.upper()
    if not lats.isin(["L", "R"]).all():
        bad = lats[~lats.isin(["L", "R"])].unique().tolist()
        raise ValueError(f"Invalid Image Laterality values in NLBS manifest: {bad}")

    out_df = pd.DataFrame(
        {
            "StudyInstanceUID": src["StudyInstanceUID"].astype(str),
            "SeriesInstanceUID": src["SeriesInstanceUID"].astype(str),
            "exam_id": src["exam_id"].astype(str),
            "case_id": src["case_id"].astype(str),
            "visit_idx": src["visit_idx"].astype(int),
            "split_cluster_id": src["split_cluster_id"].astype(str),
            "laterality": lats,
            "view_position": src["View Position"].fillna("").astype(str).str.strip().str.upper(),
            "manufacturer": src["Manufacturer"].fillna("").astype(str).str.strip(),
            "manufacturer_model": src["ManufacturerModelName"].fillna("").astype(str).str.strip(),
            "photometric_interpretation": src["PhotometricInterpretation"].fillna("").astype(str).str.strip().str.upper(),
            "dicom_rel_path": src["rel_path"].astype(str),
        }
    )

    # Sort by clinical UID, NOT cohort_group folder prefix, so batching is order-agnostic.
    out_df = out_df.sort_values(
        ["StudyInstanceUID", "SeriesInstanceUID", "dicom_rel_path"], kind="mergesort"
    ).reset_index(drop=True)
    out_df.insert(0, "image_uid", [f"nlbs_img_{i:05d}" for i in range(len(out_df))])

    if list(out_df.columns) != list(LABEL_FREE_MANIFEST_COLUMNS):
        raise RuntimeError(f"Unexpected output columns: {list(out_df.columns)}")
    assert_no_forbidden_columns(out_df.columns, context=str(output_manifest_csv))

    output_manifest_csv = Path(output_manifest_csv)
    output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_manifest_csv.with_suffix(".csv.tmp")
    out_df.to_csv(tmp_path, index=False)
    tmp_path.replace(output_manifest_csv)

    return {
        "inference_manifest_csv": str(output_manifest_csv),
        "inference_manifest_sha256": compute_sha256_file(output_manifest_csv),
        "n_images": int(len(out_df)),
        "n_unique_studies": int(out_df["StudyInstanceUID"].nunique()),
        "n_unique_exam_records": int(out_df["exam_id"].nunique()),
        "n_unique_clusters": int(out_df["split_cluster_id"].nunique()),
        "columns": list(out_df.columns),
    }


def load_label_free_inference_manifest(manifest_csv: Path) -> pd.DataFrame:
    """Load inference manifest and strictly enforce absence of label-revealing columns."""
    df = pd.read_csv(manifest_csv)
    assert_no_forbidden_columns(df.columns, context=str(manifest_csv))
    missing = [c for c in LABEL_FREE_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Inference manifest missing columns: {missing}")
    return df


def lock_stage_c_predictions(
    inference_manifest_csv: Path,
    image_results: List[Dict[str, Any]],
    output_dir: Path,
    checkpoint_metadata: Dict[str, Any],
    environment_metadata: Dict[str, Any],
    thresholds: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Aggregate finite NLBS image predictions to StudyInstanceUID level and lock outputs with SHA-256.

    Strictly refuses to touch any label-containing file. Never silently discards failed images/exams.
    """
    inf_df = load_label_free_inference_manifest(inference_manifest_csv)
    thr_cfg = thresholds if thresholds is not None else FROZEN_RSNA_THRESHOLDS

    res_df = pd.DataFrame(image_results)
    if len(res_df) == 0:
        raise ValueError("No image inference results supplied to lock_stage_c_predictions.")
    assert_no_forbidden_columns(res_df.columns, context="image_results")

    req_res_cols = {
        "image_uid",
        "decode_status",
        "segmentation_status",
        "used_fallback",
        "border_median",
        "laterality_status",
        "score_ce",
        "score_trust",
    }
    missing_res = req_res_cols - set(res_df.columns)
    if missing_res:
        raise ValueError(f"image_results missing required columns: {sorted(missing_res)}")

    if res_df["image_uid"].duplicated().any():
        raise ValueError("Duplicate image_uid entries found in image_results.")

    expected_uids = set(inf_df["image_uid"].astype(str))
    got_uids = set(res_df["image_uid"].astype(str))
    if expected_uids != got_uids:
        missing_n = len(expected_uids - got_uids)
        extra_n = len(got_uids - expected_uids)
        raise RuntimeError(
            f"Prediction completeness check failed: missing={missing_n}, extra={extra_n} image_uids. "
            "Refusing to lock incomplete or silently truncated predictions."
        )

    merged = inf_df.merge(res_df, on="image_uid", how="inner", validate="one_to_one")
    assert_no_forbidden_columns(merged.columns, context="merged_image_diagnostics")

    failed_mask = merged["decode_status"].astype(str) != "ok"
    n_failed_images = int(failed_mask.sum())
    ok_df = merged[~failed_mask].copy()

    if len(ok_df) == 0:
        raise RuntimeError("All NLBS images failed decoding/preprocessing.")

    for col in ("score_ce", "score_trust"):
        vals = ok_df[col].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            bad_uids = ok_df.loc[~np.isfinite(vals), "image_uid"].tolist()[:10]
            raise ValueError(f"Non-finite values detected in {col}: sample image_uids={bad_uids}")
        if np.any((vals < 0.0) | (vals > 1.0)):
            raise ValueError(f"Out-of-range probability scores detected in {col} outside [0, 1].")

    # Verify every StudyInstanceUID has at least one valid scored image (never drop studies silently!)
    all_studies = sorted(inf_df["StudyInstanceUID"].astype(str).unique().tolist())
    ok_studies = set(ok_df["StudyInstanceUID"].astype(str).unique().tolist())
    missing_studies = sorted(set(all_studies) - ok_studies)
    if missing_studies:
        raise RuntimeError(
            f"{len(missing_studies)}/{len(all_studies)} StudyInstanceUID exams lost all images due to "
            f"preprocessing failures: {missing_studies[:10]}. Refusing silent exam discard."
        )

    study_uids_ok = ok_df["StudyInstanceUID"].astype(str).tolist()
    ce_case_map = aggregate_case_scores(ok_df["score_ce"].tolist(), study_uids_ok)
    trust_case_map = aggregate_case_scores(ok_df["score_trust"].tolist(), study_uids_ok)

    # Non-label study structural summary
    study_meta = (
        merged.groupby("StudyInstanceUID", sort=True)
        .agg(
            split_cluster_id=("split_cluster_id", lambda s: sorted(set(s.astype(str)))[0]),
            n_cluster_ids=("split_cluster_id", "nunique"),
            n_alias_exams=("exam_id", "nunique"),
            n_manifest_images=("image_uid", "count"),
            n_succeeded_images=("decode_status", lambda s: int((s == "ok").sum())),
            n_failed_images=("decode_status", lambda s: int((s != "ok").sum())),
            n_fallback_uncropped=("used_fallback", lambda s: int(np.sum(np.asarray(s, dtype=bool)))),
        )
        .reset_index()
    )
    if (study_meta["n_cluster_ids"] != 1).any():
        raise ValueError("A StudyInstanceUID spans multiple split_cluster_ids; cluster integrity violated.")

    study_meta["score_ce"] = study_meta["StudyInstanceUID"].map(ce_case_map).astype(np.float64)
    study_meta["score_trust"] = study_meta["StudyInstanceUID"].map(trust_case_map).astype(np.float64)

    # Frozen-threshold dismissal decisions: 1 = DISMISSED (score < tau), 0 = RETAINED (score >= tau)
    study_meta["dismissed_ce_95"] = (study_meta["score_ce"] < float(thr_cfg["ce"]["target_95"])).astype(int)
    study_meta["dismissed_ce_98"] = (study_meta["score_ce"] < float(thr_cfg["ce"]["target_98"])).astype(int)
    study_meta["dismissed_trust_95"] = (study_meta["score_trust"] < float(thr_cfg["trust"]["target_95"])).astype(int)
    study_meta["dismissed_trust_98"] = (study_meta["score_trust"] < float(thr_cfg["trust"]["target_98"])).astype(int)

    pred_cols = [
        "StudyInstanceUID",
        "split_cluster_id",
        "n_alias_exams",
        "n_manifest_images",
        "n_succeeded_images",
        "n_failed_images",
        "n_fallback_uncropped",
        "score_ce",
        "score_trust",
        "dismissed_ce_95",
        "dismissed_ce_98",
        "dismissed_trust_95",
        "dismissed_trust_98",
    ]
    pred_df = study_meta[pred_cols].sort_values("StudyInstanceUID").reset_index(drop=True)
    assert_no_forbidden_columns(pred_df.columns, context="locked_nlbs_predictions")

    # Drop dicom_rel_path from diagnostics too? Keeping non-label technical diagnostics only
    diag_cols = [
        "image_uid",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "exam_id",
        "split_cluster_id",
        "laterality",
        "view_position",
        "manufacturer",
        "manufacturer_model",
        "photometric_interpretation",
        "decode_status",
        "segmentation_status",
        "used_fallback",
        "border_median",
        "laterality_status",
        "score_ce",
        "score_trust",
    ]
    diag_df = merged[diag_cols].copy()
    assert_no_forbidden_columns(diag_df.columns, context="locked_nlbs_image_diagnostics")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "locked_nlbs_predictions.csv"
    diag_path = output_dir / "locked_nlbs_image_diagnostics.csv.gz"
    manifest_path = output_dir / "locked_predictions_manifest.json"

    pred_tmp = pred_path.with_suffix(".csv.tmp")
    pred_df.to_csv(pred_tmp, index=False)
    pred_tmp.replace(pred_path)

    diag_tmp = output_dir / "locked_nlbs_image_diagnostics.csv.gz.tmp"
    diag_df.to_csv(diag_tmp, index=False, compression="gzip")
    diag_tmp.replace(diag_path)

    seg_counts = {str(k): int(v) for k, v in merged["segmentation_status"].value_counts(dropna=False).items()}
    decode_counts = {str(k): int(v) for k, v in merged["decode_status"].value_counts(dropna=False).items()}

    lock_manifest = {
        "schema_version": "anchorshift_stage_c_locked_predictions_v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_quarantine_verified": True,
        "forbidden_columns_checked": list(FORBIDDEN_LABEL_COLUMNS),
        "inference_manifest_path": str(inference_manifest_csv),
        "inference_manifest_sha256": compute_sha256_file(inference_manifest_csv),
        "locked_predictions_csv": str(pred_path),
        "locked_predictions_sha256": compute_sha256_file(pred_path),
        "locked_image_diagnostics_csv_gz": str(diag_path),
        "locked_image_diagnostics_sha256": compute_sha256_file(diag_path),
        "frozen_thresholds": thr_cfg,
        "frozen_thresholds_sha256": compute_sha256_json(thr_cfg),
        "checkpoint_metadata": checkpoint_metadata,
        "environment_metadata": environment_metadata,
        "counts": {
            "total_manifest_images": int(len(inf_df)),
            "succeeded_images": int(len(ok_df)),
            "failed_images": n_failed_images,
            "fallback_uncropped_images": int(merged["used_fallback"].astype(bool).sum()),
            "total_unique_study_exams": int(len(pred_df)),
            "unique_split_clusters": int(pred_df["split_cluster_id"].nunique()),
            "all_scores_finite": True,
            "decode_status_counts": decode_counts,
            "segmentation_status_counts": seg_counts,
        },
        "pre_label_unblinded_dismissal_summary": {
            "ce_95_dismissed_count": int(pred_df["dismissed_ce_95"].sum()),
            "ce_95_dismissal_rate": float(pred_df["dismissed_ce_95"].mean()),
            "ce_98_dismissed_count": int(pred_df["dismissed_ce_98"].sum()),
            "ce_98_dismissal_rate": float(pred_df["dismissed_ce_98"].mean()),
            "trust_95_dismissed_count": int(pred_df["dismissed_trust_95"].sum()),
            "trust_95_dismissal_rate": float(pred_df["dismissed_trust_95"].mean()),
            "trust_98_dismissed_count": int(pred_df["dismissed_trust_98"].sum()),
            "trust_98_dismissal_rate": float(pred_df["dismissed_trust_98"].mean()),
        },
    }
    tmp_json = manifest_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(lock_manifest, indent=2), encoding="utf-8")
    tmp_json.replace(manifest_path)
    return lock_manifest


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute exact Mann-Whitney / rank-based AUROC handling score ties cleanly."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    u = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u / float(n_pos * n_neg))


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute exact precision-recall step Average Precision (sklearn average_precision_score equiv)."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")

    # Sort descending by score; aggregate tied score groups deterministically
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    tp = np.cumsum(y_sorted == 1)
    total = np.arange(1, len(y_sorted) + 1)

    # Keep only the last index of each distinct score group to avoid step inflation on ties
    distinct_idx = np.r_[np.where(np.diff(s_sorted) != 0)[0], len(s_sorted) - 1]
    tp_g = tp[distinct_idx]
    total_g = total[distinct_idx]
    precision = tp_g / total_g
    recall = tp_g / float(n_pos)

    recall_prev = np.r_[0.0, recall[:-1]]
    ap = float(np.sum((recall - recall_prev) * precision))
    return ap


def _metric_snapshot(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Fast scalar metric dictionary used inside cluster bootstrap iterations."""
    dismissed = scores < threshold
    retained = ~dismissed
    n_tot = len(labels)
    n_pos = int(np.sum(labels == 1))
    n_dis = int(np.sum(dismissed))
    k_miss = int(np.sum((labels == 1) & dismissed))
    n_det = int(np.sum((labels == 1) & retained))

    sensitivity = float(n_det) / float(n_pos) if n_pos > 0 else float("nan")
    dismissal_rate = float(n_dis) / float(n_tot) if n_tot > 0 else float("nan")
    prev_dismissed = float(k_miss) / float(n_dis) if n_dis > 0 else 0.0
    missed_per_1000 = 1000.0 * float(k_miss) / float(n_tot) if n_tot > 0 else float("nan")
    return {
        "sensitivity": sensitivity,
        "dismissal_rate": dismissal_rate,
        "cancer_prev_dismissed": prev_dismissed,
        "missed_per_1000_screened": missed_per_1000,
    }


def _summarize_ci(arr: List[float]) -> Dict[str, Optional[float]]:
    vals = np.asarray(arr, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "ci95_low": float(np.percentile(vals, 2.5)),
        "ci95_high": float(np.percentile(vals, 97.5)),
    }


def compute_cluster_bootstrap_cis(
    df: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute patient-cluster-aware 95% CIs by resampling `split_cluster_id` units with replacement."""
    # Pre-index exam rows per cluster for fast O(N_clusters) numpy slicing
    labels_all = df["exam_cancer"].to_numpy(dtype=np.int64)
    ce_all = df["score_ce"].to_numpy(dtype=np.float64)
    tr_all = df["score_trust"].to_numpy(dtype=np.float64)

    clusters = sorted(df["split_cluster_id"].astype(str).unique().tolist())
    cluster_to_indices: List[np.ndarray] = []
    grouped = df.groupby("split_cluster_id", sort=False).indices
    for cid in clusters:
        cluster_to_indices.append(np.asarray(grouped[cid], dtype=np.int64))

    n_clusters = len(cluster_to_indices)
    rng = np.random.default_rng(int(seed))

    boot_store: Dict[str, List[float]] = {
        "ce_auroc": [],
        "ce_auprc": [],
        "trust_auroc": [],
        "trust_auprc": [],
        "delta_auroc_trust_minus_ce": [],
        "delta_auprc_trust_minus_ce": [],
    }
    for model_key in ("ce", "trust"):
        for op_key in ("target_95", "target_98"):
            prefix = f"{model_key}_{op_key}"
            for metric_name in ("sensitivity", "dismissal_rate", "cancer_prev_dismissed", "missed_per_1000_screened"):
                boot_store[f"{prefix}_{metric_name}"] = []
    for op_key in ("target_95", "target_98"):
        boot_store[f"delta_sensitivity_{op_key}_trust_minus_ce"] = []
        boot_store[f"delta_dismissal_{op_key}_trust_minus_ce"] = []

    for _ in range(int(n_bootstrap)):
        sampled_cids = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([cluster_to_indices[c] for c in sampled_cids])
        yb = labels_all[idx]
        ceb = ce_all[idx]
        trb = tr_all[idx]

        auroc_ce = compute_auroc(yb, ceb)
        auprc_ce = compute_auprc(yb, ceb)
        auroc_tr = compute_auroc(yb, trb)
        auprc_tr = compute_auprc(yb, trb)
        boot_store["ce_auroc"].append(auroc_ce)
        boot_store["ce_auprc"].append(auprc_ce)
        boot_store["trust_auroc"].append(auroc_tr)
        boot_store["trust_auprc"].append(auprc_tr)
        boot_store["delta_auroc_trust_minus_ce"].append(auroc_tr - auroc_ce)
        boot_store["delta_auprc_trust_minus_ce"].append(auprc_tr - auprc_ce)

        for op_key in ("target_95", "target_98"):
            snap_ce = _metric_snapshot(yb, ceb, float(thresholds["ce"][op_key]))
            snap_tr = _metric_snapshot(yb, trb, float(thresholds["trust"][op_key]))
            for metric_name, val in snap_ce.items():
                boot_store[f"ce_{op_key}_{metric_name}"].append(val)
            for metric_name, val in snap_tr.items():
                boot_store[f"trust_{op_key}_{metric_name}"].append(val)
            boot_store[f"delta_sensitivity_{op_key}_trust_minus_ce"].append(
                snap_tr["sensitivity"] - snap_ce["sensitivity"]
            )
            boot_store[f"delta_dismissal_{op_key}_trust_minus_ce"].append(
                snap_tr["dismissal_rate"] - snap_ce["dismissal_rate"]
            )

    return {
        "method": "cluster_bootstrap",
        "cluster_unit": "split_cluster_id",
        "n_clusters": n_clusters,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "metrics": {k: _summarize_ci(v) for k, v in boot_store.items()},
    }


def verify_and_evaluate_locked_predictions(
    locked_manifest_json: Path,
    exam_manifest_csv: Path,
    ce_checkpoint_path: Optional[Path] = None,
    trust_checkpoint_path: Optional[Path] = None,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
) -> Dict[str, Any]:
    """Cryptographically verify locked predictions BEFORE touching NLBS outcomes, then evaluate."""
    locked_manifest_path = Path(locked_manifest_json)
    lock_meta = json.loads(locked_manifest_path.read_text(encoding="utf-8"))

    # 1. Verify all SHA-256 checksums BEFORE loading exam_manifest.csv labels!
    pred_path = Path(lock_meta["locked_predictions_csv"])
    diag_path = Path(lock_meta["locked_image_diagnostics_csv_gz"])
    inf_path = Path(lock_meta["inference_manifest_path"])

    checks = [
        ("locked_predictions_sha256", pred_path, lock_meta["locked_predictions_sha256"]),
        ("locked_image_diagnostics_sha256", diag_path, lock_meta["locked_image_diagnostics_sha256"]),
        ("inference_manifest_sha256", inf_path, lock_meta["inference_manifest_sha256"]),
    ]
    for label, fpath, expected_hash in checks:
        actual = compute_sha256_file(fpath)
        if actual != expected_hash:
            raise RuntimeError(
                f"LOCK INTEGRITY FAILURE on {label} ({fpath}): expected={expected_hash} != actual={actual}. "
                "Aborting evaluation before joining NLBS outcomes."
            )

    thr_cfg = lock_meta["frozen_thresholds"]
    if compute_sha256_json(thr_cfg) != lock_meta["frozen_thresholds_sha256"]:
        raise RuntimeError("Frozen thresholds SHA-256 mismatch in locked_predictions_manifest.json.")
    for m_key in ("ce", "trust"):
        for op_key in ("target_95", "target_98"):
            if abs(float(thr_cfg[m_key][op_key]) - FROZEN_RSNA_THRESHOLDS[m_key][op_key]) > 1e-15:
                raise RuntimeError(
                    f"Locked threshold {m_key}.{op_key}={thr_cfg[m_key][op_key]} != frozen RSNA constant!"
                )

    # Optional on-disk checkpoint verification if provided locally
    if ce_checkpoint_path is not None:
        got_ce = compute_sha256_file(Path(ce_checkpoint_path))
        exp_ce = lock_meta["checkpoint_metadata"]["ce_checkpoint_sha256"]
        if got_ce != exp_ce:
            raise RuntimeError(f"CE checkpoint SHA-256 mismatch: {got_ce} != {exp_ce}")
    if trust_checkpoint_path is not None:
        got_tr = compute_sha256_file(Path(trust_checkpoint_path))
        exp_tr = lock_meta["checkpoint_metadata"]["trust_checkpoint_sha256"]
        if got_tr != exp_tr:
            raise RuntimeError(f"TRUST checkpoint SHA-256 mismatch: {got_tr} != {exp_tr}")

    # Assert the locked predictions CSV still has zero label columns before joining outcomes
    pred_df = pd.read_csv(pred_path)
    assert_no_forbidden_columns(pred_df.columns, context=str(pred_path))

    # 2. ONLY AFTER verification succeeds: load ground-truth exam_manifest.csv and deduplicate on StudyInstanceUID
    em_df = pd.read_csv(exam_manifest_csv)
    if em_df.groupby("StudyInstanceUID")["exam_cancer"].nunique().gt(1).any():
        raise ValueError("Outcome label disagreement found within duplicate StudyInstanceUID!")
    if em_df.groupby("StudyInstanceUID")["cohort_group"].nunique().gt(1).any():
        raise ValueError("cohort_group disagreement found within duplicate StudyInstanceUID!")
    if em_df.groupby("StudyInstanceUID")["split_cluster_id"].nunique().gt(1).any():
        raise ValueError("split_cluster_id disagreement found within duplicate StudyInstanceUID!")

    unique_em = (
        em_df.sort_values("exam_id")
        .drop_duplicates("StudyInstanceUID", keep="first")
        .loc[:, ["StudyInstanceUID", "case_id", "visit_idx", "cohort_group", "exam_cancer", "hard_to_state_negative", "split_cluster_id"]]
        .rename(columns={"split_cluster_id": "truth_split_cluster_id"})
        .reset_index(drop=True)
    )

    if len(pred_df) != len(unique_em) or set(pred_df["StudyInstanceUID"]) != set(unique_em["StudyInstanceUID"]):
        raise RuntimeError(
            f"StudyInstanceUID mismatch between locked predictions ({len(pred_df)}) and "
            f"deduplicated exam_manifest ({len(unique_em)})."
        )

    joined = pred_df.merge(unique_em, on="StudyInstanceUID", how="inner", validate="one_to_one")
    if not (joined["split_cluster_id"].astype(str) == joined["truth_split_cluster_id"].astype(str)).all():
        raise RuntimeError("split_cluster_id mismatch after joining locked predictions with exam_manifest.")

    # Verify locked dismissal flags match re-applying frozen thresholds to locked scores
    for m_key, sc_col in (("ce", "score_ce"), ("trust", "score_trust")):
        for op_key, suffix in (("target_95", "95"), ("target_98", "98")):
            expected_dis = (joined[sc_col].to_numpy(dtype=np.float64) < float(thr_cfg[m_key][op_key])).astype(int)
            col_dis = f"dismissed_{m_key}_{suffix}"
            if not np.array_equal(joined[col_dis].to_numpy(dtype=int), expected_dis):
                raise RuntimeError(f"Locked dismissal flag mismatch in {col_dis}!")

    labels = joined["exam_cancer"].to_numpy(dtype=np.int64)
    results_by_model: Dict[str, Any] = {}
    target_map = {"target_95": 0.95, "target_98": 0.98}

    for m_key, sc_col in (("ce", "score_ce"), ("trust", "score_trust")):
        scores = joined[sc_col].to_numpy(dtype=np.float64)
        model_out: Dict[str, Any] = {
            "auroc": compute_auroc(labels, scores),
            "auprc": compute_auprc(labels, scores),
            "score_summary": {
                "min": float(np.min(scores)),
                "p05": float(np.percentile(scores, 5)),
                "median": float(np.median(scores)),
                "p95": float(np.percentile(scores, 95)),
                "max": float(np.max(scores)),
                "mean_cancer_positive": float(np.mean(scores[labels == 1])),
                "mean_non_cancer": float(np.mean(scores[labels == 0])),
            },
        }
        for op_key, target_val in target_map.items():
            thr = float(thr_cfg[m_key][op_key])
            res_op = evaluate_dismissal_at_threshold(
                case_scores=scores,
                case_labels=labels,
                threshold=thr,
                confidence=0.99,
                target_recall=target_val,
            )
            # Post-lock breakdown by screening outcome cohort (normal vs False Positive)
            dismissed_mask = scores < thr
            cohort_breakdown = {}
            for cg in ("normal", "False Positive", "abnormal"):
                cmask = joined["cohort_group"].astype(str).to_numpy() == cg
                cn = int(cmask.sum())
                cdis = int((cmask & dismissed_mask).sum())
                cohort_breakdown[cg] = {
                    "total_exams": cn,
                    "n_dismissed": cdis,
                    "dismissal_rate": float(cdis / cn) if cn > 0 else 0.0,
                }
            res_op["cohort_group_breakdown"] = cohort_breakdown
            res_op["clopper_pearson_independence_caveat"] = (
                "Exact Clopper-Pearson binomial bound assumes independent exams; "
                f"NLBS contains {len(joined)} exams across {joined['split_cluster_id'].nunique()} patient clusters "
                "(see cluster_bootstrap CIs)."
            )
            model_out[op_key] = res_op
        results_by_model[m_key] = model_out

    # Cluster-level collapse check (5,942 clusters, worst/any exam in cluster) for repeat-exam independence check
    cluster_df = (
        joined.groupby("split_cluster_id", sort=True)
        .agg(
            exam_cancer=("exam_cancer", "max"),
            score_ce=("score_ce", "max"),
            score_trust=("score_trust", "max"),
        )
        .reset_index()
    )
    cluster_collapsed = {}
    c_labels = cluster_df["exam_cancer"].to_numpy(dtype=np.int64)
    for m_key, sc_col in (("ce", "score_ce"), ("trust", "score_trust")):
        c_scores = cluster_df[sc_col].to_numpy(dtype=np.float64)
        cluster_collapsed[m_key] = {
            "auroc": compute_auroc(c_labels, c_scores),
            "auprc": compute_auprc(c_labels, c_scores),
            "target_95": evaluate_dismissal_at_threshold(
                c_scores, c_labels, float(thr_cfg[m_key]["target_95"]), confidence=0.99, target_recall=0.95
            ),
            "target_98": evaluate_dismissal_at_threshold(
                c_scores, c_labels, float(thr_cfg[m_key]["target_98"]), confidence=0.99, target_recall=0.98
            ),
        }

    cluster_boot = compute_cluster_bootstrap_cis(
        df=joined,
        thresholds=thr_cfg,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )

    # Missed cancer case details (post-lock diagnostic inspection)
    missed_cases = {}
    for m_key, sc_col in (("ce", "score_ce"), ("trust", "score_trust")):
        missed_cases[m_key] = {}
        for op_key in ("target_95", "target_98"):
            thr = float(thr_cfg[m_key][op_key])
            miss_df = joined[(joined["exam_cancer"] == 1) & (joined[sc_col] < thr)]
            missed_cases[m_key][op_key] = [
                {
                    "StudyInstanceUID": str(r["StudyInstanceUID"]),
                    "case_id": str(r["case_id"]),
                    "split_cluster_id": str(r["split_cluster_id"]),
                    "score": float(r[sc_col]),
                    "threshold": thr,
                }
                for _, r in miss_df.sort_values(sc_col).iterrows()
            ]

    # Scanner-family image preprocessing diagnostics (post-lock)
    diag_df = pd.read_csv(diag_path)
    scanner_summary = {}
    for mfg, grp in diag_df.groupby("manufacturer", sort=True):
        scanner_summary[str(mfg)] = {
            "n_images": int(len(grp)),
            "n_failed": int((grp["decode_status"] != "ok").sum()),
            "n_fallback_uncropped": int(grp["used_fallback"].astype(bool).sum()),
            "ce_mean_score": float(grp.loc[grp["decode_status"] == "ok", "score_ce"].mean()),
            "trust_mean_score": float(grp.loc[grp["decode_status"] == "ok", "score_trust"].mean()),
        }

    return {
        "schema_version": "anchorshift_stage_c_nlbs_eval_v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_denominators": {
            "total_exam_manifest_rows": int(len(em_df)),
            "deduplicated_unique_study_exams": int(len(joined)),
            "deduplicated_shared_study_records_collapsed": int(len(em_df) - len(joined)),
            "unique_split_clusters": int(joined["split_cluster_id"].nunique()),
            "total_cancer_positive_exams": int(labels.sum()),
            "total_cancer_positive_clusters": int(c_labels.sum()),
            "total_false_positive_exams": int((joined["cohort_group"] == "False Positive").sum()),
            "total_normal_exams": int((joined["cohort_group"] == "normal").sum()),
            "nlbs_exam_prevalence": float(labels.mean()),
        },
        "lock_verification": {
            "verified_before_label_join": True,
            "locked_manifest_path": str(locked_manifest_path),
            "locked_predictions_sha256": lock_meta["locked_predictions_sha256"],
            "locked_image_diagnostics_sha256": lock_meta["locked_image_diagnostics_sha256"],
            "inference_manifest_sha256": lock_meta["inference_manifest_sha256"],
            "frozen_thresholds_sha256": lock_meta["frozen_thresholds_sha256"],
            "checkpoint_metadata": lock_meta["checkpoint_metadata"],
        },
        "preprocessing_and_integrity": lock_meta["counts"],
        "scanner_preprocessing_summary": scanner_summary,
        "exam_level_results": results_by_model,
        "cluster_bootstrap_95ci": cluster_boot,
        "cluster_collapsed_reference": cluster_collapsed,
        "missed_cancer_exams": missed_cases,
    }
