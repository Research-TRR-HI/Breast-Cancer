"""Paper 1 Supplementary Exploratory — Shared RSNA Calibration, Pre-Join Locking & NLBS Evaluation.

Enforces identical scientific & cryptographic guarantees for fine-tuned specialist
models (`gmic_rsna_finetuned`, `nyu_fourview_rsna_finetuned`) without modifying the
immutable primary 6-model benchmark (`src/paper1/six_model_eval.py`):
1. RSNA-only 98%/95% threshold calibration (`val_calib`, 2-fold OOF source eval)
   and cryptographic calibration lock BEFORE NLBS evaluation runs.
2. Label-free NLBS prediction locking (`score < tau -> dismissed`) and SHA-256
   verification BEFORE joining `datasets/nlbs/exam_manifest.csv`.
3. Cluster-bootstrap (`B=2000` over `split_cluster_id`) evaluation on Common 4-View
   (`N=6,184`, `5,868` clusters, `151` pos), plus optional Full (`N=6,359`) and
   Cluster-Collapsed (`N=5,868`) cohorts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.stage_c_eval import (
    assert_no_forbidden_columns,
    compute_auprc,
    compute_auroc,
    compute_sha256_file,
)
from src.models.trust_model import aggregate_case_labels
from src.paper1.six_model_eval import (
    _class_score_quantiles,
    calibrate_rsna_thresholds_only,
)
from src.training.calibration import evaluate_dismissal_at_threshold

SUPPLEMENTARY_MODEL_IDS = ("gmic_rsna_finetuned", "nyu_fourview_rsna_finetuned")


def verify_rsna_case_scores_lock(
    locked_pred_dir: Path,
    *,
    prefix: str,
    model_id: str,
    expected_checkpoint_sha256: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Verify SHA-256 provenance lock on RSNA val_calib case predictions."""
    lock_path = locked_pred_dir / f"{prefix}_rsna_val_calib_provenance_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing RSNA provenance lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("model_id") != model_id:
        raise RuntimeError(f"model_id mismatch: {lock.get('model_id')} != {model_id}")
    ckpt_sha = str(lock.get("checkpoint_sha256", "")).lower()
    if len(ckpt_sha) != 64:
        raise RuntimeError(f"Invalid checkpoint_sha256 in {lock_path}: {ckpt_sha!r}")
    if expected_checkpoint_sha256 and ckpt_sha != expected_checkpoint_sha256.lower():
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {ckpt_sha} != {expected_checkpoint_sha256.lower()}")

    if "locked_files_sha256" in lock:
        for fname, exp_sha in lock["locked_files_sha256"].items():
            fpath = locked_pred_dir / fname
            got_sha = compute_sha256_file(fpath)
            if got_sha != exp_sha:
                raise RuntimeError(f"SHA-256 tamper detected in {fpath}: {got_sha} != {exp_sha}")
    elif "artifacts" in lock:
        for art_key, art in lock["artifacts"].items():
            art_p = Path(art["path"])
            if not art_p.exists():
                art_p = locked_pred_dir / art_p.name
            got_sha = compute_sha256_file(art_p)
            if got_sha != art["sha256"]:
                raise RuntimeError(f"SHA-256 tamper detected in {art_key} ({art_p}): {got_sha} != {art['sha256']}")
    else:
        raise RuntimeError(f"Unrecognized lock schema in {lock_path}")

    case_csv = locked_pred_dir / f"{prefix}_rsna_val_calib_case_scores.csv"
    case_df = pd.read_csv(case_csv, dtype={"patient_id": str})
    assert_no_forbidden_columns(case_df.columns, context=f"verify_rsna_{model_id}")
    if len(case_df) != 2382 or case_df["patient_id"].duplicated().any():
        raise RuntimeError(f"Expected 2,382 unique RSNA val_calib patients, got {len(case_df)}")
    scores = case_df["score"].values.astype(np.float64)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise RuntimeError(f"Non-finite or out-of-range scores in {case_csv}")
    return case_df, lock, compute_sha256_file(lock_path)


def calibrate_finetuned_rsna_thresholds(
    *,
    model_id: str,
    prefix: str,
    locked_pred_dir: Path,
    rsna_splits_csv: Path,
    rsna_fold_csv: Path,
    out_dir: Path,
    expected_checkpoint_sha256: Optional[str] = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calibrate RSNA-only 98%/95% operating points and write a SHA-256 calibration lock."""
    if model_id not in SUPPLEMENTARY_MODEL_IDS:
        raise ValueError(f"Unsupported supplementary model_id={model_id!r}")

    case_df, lock, lock_sha = verify_rsna_case_scores_lock(
        locked_pred_dir=locked_pred_dir,
        prefix=prefix,
        model_id=model_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )

    splits_lbl = pd.read_csv(rsna_splits_csv, usecols=["patient_id", "split", "cancer"])
    vc_lbl = splits_lbl[splits_lbl["split"] == "val_calib"].copy()
    case_lbl_map = aggregate_case_labels(
        vc_lbl["cancer"].astype(int).tolist(),
        vc_lbl["patient_id"].astype(str).tolist(),
    )

    eval_df = case_df.copy()
    eval_df["cancer"] = eval_df["patient_id"].map(case_lbl_map).astype(int)
    eval_df["patient_id_int"] = eval_df["patient_id"].astype(int)
    rsna_for_calib = eval_df[["patient_id_int", "cancer", "score"]].rename(columns={"patient_id_int": "patient_id"})

    source_eval = calibrate_rsna_thresholds_only(
        rsna_for_calib,
        fold_csv=rsna_fold_csv,
        n_boot=n_boot,
        seed=seed,
    )
    full_taus = source_eval["full_val_calib_thresholds"]
    y_r = rsna_for_calib["cancer"].values.astype(np.int64)
    s_r = rsna_for_calib["score"].values.astype(np.float64)

    calib_report = {
        "schema_version": "paper1_specialist_rsna_calibration_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_quarantine_note": "Computed strictly using RSNA val_calib cancer labels ONLY prior to NLBS inference/join.",
        "model_provenance": {
            "model_id": model_id,
            "checkpoint_file": lock.get("checkpoint_file", ""),
            "checkpoint_sha256": lock["checkpoint_sha256"],
            "preprocess_version": lock.get("preprocess_version", ""),
            "rsna_step2_lock_sha256": lock_sha,
        },
        "rsna_source_eval": source_eval,
        "full_val_calib_resubstitution_NOT_INDEPENDENT": {
            "n_patients": int(len(eval_df)),
            "n_pos": int(y_r.sum()),
            "auroc": float(source_eval["auroc"]),
            "auprc": float(source_eval["auprc"]),
            "target_98": evaluate_dismissal_at_threshold(s_r, y_r, full_taus["target_98"], 0.99, 0.98),
            "target_95": evaluate_dismissal_at_threshold(s_r, y_r, full_taus["target_95"], 0.99, 0.95),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    calib_json_p = out_dir / f"{model_id}_rsna_calibration.json"
    calib_lock_p = out_dir / f"{model_id}_rsna_calibration_lock.json"
    calib_json_p.write_text(json.dumps(calib_report, indent=2), encoding="utf-8")

    calib_lock = {
        "schema_version": "paper1_specialist_rsna_calib_lock_v1",
        "generated_at_utc": calib_report["generated_at_utc"],
        "model_id": model_id,
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "rsna_step2_lock_sha256": lock_sha,
        "frozen_thresholds": full_taus,
        "calibration_json_path": str(calib_json_p),
        "calibration_json_sha256": compute_sha256_file(calib_json_p),
    }
    calib_lock_p.write_text(json.dumps(calib_lock, indent=2), encoding="utf-8")
    return calib_lock


def verify_rsna_calibration_lock(
    calib_lock_path: Path,
    *,
    expected_model_id: str,
    expected_checkpoint_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Verify RSNA calibration lock & underlying calibration JSON integrity."""
    if not calib_lock_path.exists():
        raise FileNotFoundError(
            f"BLOCKED: Missing RSNA calibration lock {calib_lock_path}. Calibrate RSNA first."
        )
    lock = json.loads(calib_lock_path.read_text(encoding="utf-8"))
    if lock.get("model_id") != expected_model_id:
        raise RuntimeError(f"Calibration lock model_id mismatch: {lock.get('model_id')} != {expected_model_id}")
    ckpt_sha = str(lock.get("checkpoint_sha256", "")).lower()
    if expected_checkpoint_sha256 and ckpt_sha != expected_checkpoint_sha256.lower():
        raise RuntimeError(f"Calibration checkpoint_sha256 mismatch: {ckpt_sha} != {expected_checkpoint_sha256.lower()}")
    calib_p = Path(lock["calibration_json_path"])
    if not calib_p.exists():
        calib_p = calib_lock_path.parent / calib_p.name
    got_sha = compute_sha256_file(calib_p)
    if got_sha != lock.get("calibration_json_sha256"):
        raise RuntimeError(f"Calibration JSON SHA-256 mismatch: {got_sha} != {lock.get('calibration_json_sha256')}")
    calib = json.loads(calib_p.read_text(encoding="utf-8"))
    return lock, calib


def lock_finetuned_nlbs_predictions(
    *,
    model_id: str,
    nlbs_case_scores_csv: Path,
    calib_lock_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """Apply frozen RSNA thresholds to label-free NLBS scores and SHA-256 lock prior to outcome join."""
    calib_lock, _ = verify_rsna_calibration_lock(calib_lock_path, expected_model_id=model_id)
    case_df = pd.read_csv(nlbs_case_scores_csv, dtype={"StudyInstanceUID": str})
    assert_no_forbidden_columns(case_df.columns, context=f"lock_nlbs_{model_id}")
    if case_df["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{model_id}: duplicate StudyInstanceUID in {nlbs_case_scores_csv}")
    case_df = case_df.sort_values("StudyInstanceUID").reset_index(drop=True)
    case_df["score"] = case_df["score"].astype(float)
    if not np.all(np.isfinite(case_df["score"].values)):
        raise ValueError(f"{model_id}: non-finite NLBS scores")

    taus = calib_lock["frozen_thresholds"]
    tau98 = float(taus["target_98"])
    tau95 = float(taus["target_95"])
    case_df["dismissed_98"] = case_df["score"].values < tau98
    case_df["dismissed_95"] = case_df["score"].values < tau95
    if "n_crop_fallbacks" not in case_df.columns:
        case_df["n_crop_fallbacks"] = 0
    case_df["has_crop_fallback"] = case_df["n_crop_fallbacks"].astype(int) > 0

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{model_id}_locked_nlbs_predictions.csv"
    manifest_path = out_dir / f"{model_id}_lock_manifest.json"
    cols = ["StudyInstanceUID", "score", "dismissed_98", "dismissed_95", "n_crop_fallbacks", "has_crop_fallback"]
    case_df[cols].to_csv(csv_path, index=False)

    manifest = {
        "schema_version": "paper1_specialist_locked_predictions_v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "stratum": "supplementary_exploratory_rsna_finetuned",
        "model_sha256": calib_lock["checkpoint_sha256"],
        "rsna_calibration_json_sha256": calib_lock["calibration_json_sha256"],
        "frozen_rsna_thresholds": {"target_98": tau98, "target_95": tau95},
        "n_studies": int(len(case_df)),
        "n_studies_with_crop_fallback": int(case_df["has_crop_fallback"].sum()),
        "predictions_csv": csv_path.name,
        "predictions_sha256": compute_sha256_file(csv_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_and_load_finetuned_locked_predictions(
    lock_dir: Path,
    model_id: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Verify SHA-256 lock and exact frozen-threshold boolean consistency before NLBS outcome join."""
    manifest_path = lock_dir / f"{model_id}_lock_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing NLBS lock manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    csv_path = lock_dir / manifest["predictions_csv"]
    got_sha = compute_sha256_file(csv_path)
    if got_sha != manifest["predictions_sha256"]:
        raise RuntimeError(f"Lock tamper detected for {model_id}: {got_sha} != {manifest['predictions_sha256']}")

    df = pd.read_csv(csv_path, dtype={"StudyInstanceUID": str})
    assert_no_forbidden_columns(df.columns, context=f"verify_locked_{model_id}")
    tau98 = float(manifest["frozen_rsna_thresholds"]["target_98"])
    tau95 = float(manifest["frozen_rsna_thresholds"]["target_95"])
    if not np.array_equal(df["dismissed_98"].values.astype(bool), df["score"].values < tau98):
        raise RuntimeError(f"dismissed_98 threshold inconsistency in {model_id}")
    if not np.array_equal(df["dismissed_95"].values.astype(bool), df["score"].values < tau95):
        raise RuntimeError(f"dismissed_95 threshold inconsistency in {model_id}")
    return df, manifest


def _evaluate_cohort_single_model(
    merged_df: pd.DataFrame,
    *,
    tau98: float,
    tau95: float,
    cluster_col: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    y = merged_df["exam_cancer"].values.astype(np.int64)
    s = merged_df["score"].values.astype(np.float64)
    d98 = s < tau98
    d95 = s < tau95
    clusters = merged_df[cluster_col].astype(str).values
    uniq_clusters = np.array(sorted(pd.unique(clusters)))
    cluster_rows = {c: np.where(clusters == c)[0] for c in uniq_clusters}
    n_clusters = len(uniq_clusters)

    rng = np.random.default_rng(seed)
    aurocs, auprcs, s98_b, d98_b, s95_b, d95_b = [], [], [], [], [], []
    for _ in range(n_boot):
        sampled_c = uniq_clusters[rng.integers(0, n_clusters, size=n_clusters)]
        idx = np.concatenate([cluster_rows[c] for c in sampled_c])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        sb = s[idx]
        pos_m = yb == 1
        aurocs.append(compute_auroc(yb, sb))
        auprcs.append(compute_auprc(yb, sb))
        s98_b.append(float(np.mean(~d98[idx][pos_m])))
        d98_b.append(float(np.mean(d98[idx])))
        s95_b.append(float(np.mean(~d95[idx][pos_m])))
        d95_b.append(float(np.mean(d95[idx])))

    def _ci(vals: list[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }

    return {
        "n_units": int(len(merged_df)),
        "n_clusters": int(n_clusters),
        "n_positive": int(y.sum()),
        "auroc": float(compute_auroc(y, s)),
        "auprc": float(compute_auprc(y, s)),
        "target_98": evaluate_dismissal_at_threshold(s, y, tau98, 0.99, 0.98),
        "target_95": evaluate_dismissal_at_threshold(s, y, tau95, 0.99, 0.95),
        "bootstrap_95ci": {
            "auroc": _ci(aurocs),
            "auprc": _ci(auprcs),
            "target_98_sens": _ci(s98_b),
            "target_98_dis": _ci(d98_b),
            "target_95_sens": _ci(s95_b),
            "target_95_dis": _ci(d95_b),
        },
        "score_quantiles": _class_score_quantiles(y, s),
    }


def evaluate_finetuned_nlbs_against_labels(
    *,
    model_id: str,
    calib_lock_path: Path,
    nlbs_lock_dir: Path,
    exam_manifest_csv: Path,
    common_4view_csv: Path,
    out_dir: Path,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify frozen locks, join NLBS outcomes, and evaluate common 4-view (+ full if available) cohorts."""
    calib_lock, calib = verify_rsna_calibration_lock(calib_lock_path, expected_model_id=model_id)
    locked_df, lock_manifest = verify_and_load_finetuned_locked_predictions(nlbs_lock_dir, model_id)
    tau98 = float(calib_lock["frozen_thresholds"]["target_98"])
    tau95 = float(calib_lock["frozen_thresholds"]["target_95"])
    if not math.isclose(lock_manifest["frozen_rsna_thresholds"]["target_98"], tau98, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Frozen NLBS tau_98 mismatch vs RSNA calibration lock")
    if not math.isclose(lock_manifest["frozen_rsna_thresholds"]["target_95"], tau95, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Frozen NLBS tau_95 mismatch vs RSNA calibration lock")

    ex_raw = pd.read_csv(exam_manifest_csv)
    ex_raw["StudyInstanceUID"] = ex_raw["StudyInstanceUID"].astype(str)
    dedup_exam = (
        ex_raw.drop_duplicates(subset=["StudyInstanceUID"], keep="first")[
            ["StudyInstanceUID", "split_cluster_id", "exam_cancer"]
        ]
        .sort_values("StudyInstanceUID")
        .reset_index(drop=True)
    )
    c4v_df = pd.read_csv(common_4view_csv, dtype={"StudyInstanceUID": str})
    assert_no_forbidden_columns(c4v_df.columns, context=f"eval_c4v_{model_id}")
    c4v_uids = set(c4v_df["StudyInstanceUID"].tolist())

    merged_common = (
        dedup_exam[dedup_exam["StudyInstanceUID"].isin(c4v_uids)]
        .merge(locked_df, on="StudyInstanceUID", how="inner")
        .sort_values("StudyInstanceUID")
        .reset_index(drop=True)
    )
    common_eval = _evaluate_cohort_single_model(
        merged_common, tau98=tau98, tau95=tau95, cluster_col="split_cluster_id", n_boot=n_boot, seed=seed
    )

    full_eval = None
    if len(locked_df) == len(dedup_exam):
        merged_full = dedup_exam.merge(locked_df, on="StudyInstanceUID", how="inner").sort_values("StudyInstanceUID").reset_index(drop=True)
        full_eval = _evaluate_cohort_single_model(
            merged_full, tau98=tau98, tau95=tau95, cluster_col="split_cluster_id", n_boot=n_boot, seed=seed
        )

    collapsed_df = (
        merged_common.groupby("split_cluster_id", as_index=False)
        .agg({"exam_cancer": "max", "score": "max"})
        .sort_values("split_cluster_id")
        .reset_index(drop=True)
    )
    collapsed_eval = _evaluate_cohort_single_model(
        collapsed_df, tau98=tau98, tau95=tau95, cluster_col="split_cluster_id", n_boot=n_boot, seed=seed
    )

    report = {
        "schema_version": "paper1_specialist_nlbs_eval_v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "checkpoint_sha256": calib_lock["checkpoint_sha256"],
        "rsna_calibration_json_sha256": calib_lock["calibration_json_sha256"],
        "nlbs_predictions_sha256": lock_manifest["predictions_sha256"],
        "frozen_rsna_thresholds": {"target_98": tau98, "target_95": tau95},
        "rsna_source_eval": calib["rsna_source_eval"],
        "nlbs_common_4view_primary": common_eval,
        "nlbs_full_cohort_secondary": full_eval,
        "nlbs_cluster_collapsed_common_4view": collapsed_eval,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{model_id}_nlbs_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
