"""Paper 1 — NYU Four-View RSNA-Only Threshold Calibration & Provenance Lock (CPU ONLY).

Run STRICTLY after RSNA Step 2 GPU inference (`--split rsna_val_calib`) and BEFORE NLBS Step 2 GPU inference.
Zero NLBS cancer labels or files are imported, opened, or touched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.stage_c_eval import assert_no_forbidden_columns, compute_sha256_file
from src.models.trust_model import aggregate_case_labels
from src.paper1.nyu_fourview_adapter import (
    NYU_CACHE_VERSION,
    NYU_CKPT_SHA256,
    NYU_MODEL_ID,
    NYU_STEP1_MANIFEST_SHA256,
    NYU_TTA_EPOCHS_OFFICIAL,
    NYU_TTA_SEED_OFFICIAL,
)
from src.paper1.six_model_eval import calibrate_rsna_thresholds_only
from src.training.calibration import compute_target_recall_threshold, evaluate_dismissal_at_threshold


def verify_and_calibrate_nyu_rsna_only(
    locked_pred_dir: Path,
    rsna_step1_manifest_csv: Path,
    rsna_splits_csv: Path,
    rsna_fold_csv: Path,
    out_dir: Path,
    *,
    expected_batch_exams: int = 4,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify locked RSNA NYU Step-2 outputs, calibrate source thresholds ONLY on RSNA, and write SHA-256 lock."""
    lock_path = locked_pred_dir / "nyu_rsna_val_calib_provenance_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing NYU RSNA Step 2 provenance lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    if lock.get("model_id") != NYU_MODEL_ID:
        raise RuntimeError(f"model_id mismatch: {lock.get('model_id')}")
    if str(lock.get("checkpoint_sha256", "")).lower() != NYU_CKPT_SHA256:
        raise RuntimeError("NYU checkpoint_sha256 mismatch")
    if lock.get("preprocess_version") != NYU_CACHE_VERSION:
        raise RuntimeError(f"preprocess_version mismatch: {lock.get('preprocess_version')}")
    if lock.get("preprocess_manifest_sha256s") != [NYU_STEP1_MANIFEST_SHA256["rsna_val_calib"]]:
        raise RuntimeError(f"RSNA Step 1 manifest SHA mismatch in lock: {lock.get('preprocess_manifest_sha256s')}")
    if lock.get("num_tta_epochs") != NYU_TTA_EPOCHS_OFFICIAL or lock.get("tta_seed") != NYU_TTA_SEED_OFFICIAL:
        raise RuntimeError("NYU TTA config mismatch in RSNA lock")
    if lock.get("batch_exams") != int(expected_batch_exams):
        raise RuntimeError(f"Expected batch_exams={expected_batch_exams}, got {lock.get('batch_exams')}")

    for art_key, art in lock["artifacts"].items():
        art_p = Path(art["path"])
        if not art_p.exists():
            art_p = locked_pred_dir / art_p.name
        got_sha = compute_sha256_file(art_p)
        if got_sha != art["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch in locked RSNA artifact {art_key}: {got_sha} != {art['sha256']}")

    case_p = locked_pred_dir / "nyu_rsna_val_calib_case_scores.csv"
    fb_p = locked_pred_dir / "nyu_rsna_val_calib_fallback_log.json"
    case_df = pd.read_csv(case_p, dtype={"patient_id": str})
    assert_no_forbidden_columns(case_df.columns, context="nyu_rsna_case_scores")
    fb_json = json.loads(fb_p.read_text(encoding="utf-8"))

    step1_sha = compute_sha256_file(rsna_step1_manifest_csv)
    if step1_sha != NYU_STEP1_MANIFEST_SHA256["rsna_val_calib"]:
        raise RuntimeError(f"RSNA Step 1 CSV SHA mismatch: {step1_sha} != {NYU_STEP1_MANIFEST_SHA256['rsna_val_calib']}")
    s1_df = pd.read_csv(rsna_step1_manifest_csv, dtype={"case_id": str, "item_id": str})
    exp_cases = set(s1_df["case_id"].tolist())
    if len(case_df) != len(exp_cases) or case_df["patient_id"].duplicated().any() or set(case_df["patient_id"]) != exp_cases:
        raise RuntimeError(f"RSNA patient coverage mismatch ({len(case_df)} vs {len(exp_cases)})")
    if not (case_df["n_views"] == 4).all():
        raise RuntimeError("Incomplete 4-view coverage in RSNA NYU predictions")

    for col in ("score", "left_benign", "right_benign", "left_malignant", "right_malignant"):
        vals = case_df[col].values.astype(np.float64)
        if not np.all(np.isfinite(vals)) or np.any((vals < 0.0) | (vals > 1.0)):
            raise RuntimeError(f"Non-finite or out-of-range probabilities in {col}")
    if not np.allclose(
        case_df["score"].values,
        np.maximum(case_df["left_malignant"].values, case_df["right_malignant"].values),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("Exam score != max(left_malignant, right_malignant)")

    # Phase 2: Read RSNA labels ONLY (zero NLBS files touched)
    splits_lbl = pd.read_csv(rsna_splits_csv, usecols=["patient_id", "split", "cancer"])
    vc_lbl = splits_lbl[splits_lbl["split"] == "val_calib"].copy()
    case_lbl_map = aggregate_case_labels(vc_lbl["cancer"].astype(int).tolist(), vc_lbl["patient_id"].astype(str).tolist())

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
    resub_98 = evaluate_dismissal_at_threshold(s_r, y_r, full_taus["target_98"], 0.99, 0.98)
    resub_95 = evaluate_dismissal_at_threshold(s_r, y_r, full_taus["target_95"], 0.99, 0.95)

    fb_cases = eval_df[eval_df["has_crop_fallback"].astype(bool)].copy()
    clean_cases = eval_df[~eval_df["has_crop_fallback"].astype(bool)].copy()
    pos_clean = clean_cases[clean_cases["cancer"] == 1]["score"].values
    tau98_clean = float(compute_target_recall_threshold(pos_clean, 0.98))
    tau95_clean = float(compute_target_recall_threshold(pos_clean, 0.95))

    calib_report = {
        "schema_version": "paper1_nyu_rsna_calibration_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_quarantine_note": "Computed strictly using RSNA val_calib cancer labels ONLY prior to NLBS GPU inference.",
        "model_provenance": {
            "model_id": NYU_MODEL_ID,
            "checkpoint_file": "sample_image_model.p",
            "checkpoint_sha256": NYU_CKPT_SHA256,
            "preprocess_version": NYU_CACHE_VERSION,
            "rsna_step1_manifest_sha256": step1_sha,
            "precision": "fp32",
            "batch_exams": int(expected_batch_exams),
            "num_tta_epochs": NYU_TTA_EPOCHS_OFFICIAL,
            "tta_seed": NYU_TTA_SEED_OFFICIAL,
            "rsna_step2_lock_sha256": compute_sha256_file(lock_path),
        },
        "rsna_source_eval": source_eval,
        "full_val_calib_resubstitution_NOT_INDEPENDENT": {
            "n_patients": int(len(eval_df)),
            "n_pos": int(y_r.sum()),
            "auroc": float(source_eval["auroc"]),
            "auprc": float(source_eval["auprc"]),
            "target_98": resub_98,
            "target_95": resub_95,
        },
        "rsna_fallback_calibration_sensitivity": {
            "n_fallback_views": int(fb_json["n_fallback_views"]),
            "n_fallback_exams": int(len(fb_cases)),
            "n_fallback_exams_cancer_positive": int(fb_cases["cancer"].sum()),
            "clean_exams_only_n": int(len(clean_cases)),
            "clean_exams_only_pos": int(clean_cases["cancer"].sum()),
            "clean_exams_only_taus": {"target_98": tau98_clean, "target_95": tau95_clean},
            "delta_tau98_clean_exams": float(tau98_clean - full_taus["target_98"]),
            "delta_tau95_clean_exams": float(tau95_clean - full_taus["target_95"]),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    calib_json_p = out_dir / "nyu_rsna_calibration.json"
    calib_md_p = out_dir / "nyu_rsna_calibration.md"
    lock_json_p = out_dir / "nyu_rsna_calibration_lock.json"
    calib_json_p.write_text(json.dumps(calib_report, indent=2), encoding="utf-8")

    md_lines = [
        "# Paper 1 — NYU Four-View RSNA Calibration & Source Evaluation Lock",
        "",
        f"- **Generated (UTC):** `{calib_report['generated_at_utc']}`",
        f"- **Checkpoint SHA-256:** `{NYU_CKPT_SHA256}` | **Step-1 RSNA Manifest SHA-256:** `{step1_sha}`",
        f"- **Cohort:** `{len(eval_df)}` RSNA `val_calib` patients (`{int(y_r.sum())}` cancer positive)",
        f"- **OOF AUROC:** `{source_eval['auroc']:.4f}` | **AUPRC:** `{source_eval['auprc']:.4f}`",
        f"- **Frozen RSNA Thresholds:** `tau_98 = {full_taus['target_98']:.6f}`, `tau_95 = {full_taus['target_95']:.6f}`",
        f"- **Fallback Calibration Sensitivity ({len(fb_cases)} fallback exams / {fb_json['n_fallback_views']} views, {int(fb_cases['cancer'].sum())} pos):** `delta_tau98 = {tau98_clean - full_taus['target_98']:.6f}`, `delta_tau95 = {tau95_clean - full_taus['target_95']:.6f}`",
    ]
    calib_md_p.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    calib_lock = {
        "schema_version": "paper1_nyu_rsna_calib_lock_v1",
        "generated_at_utc": calib_report["generated_at_utc"],
        "model_id": NYU_MODEL_ID,
        "checkpoint_sha256": NYU_CKPT_SHA256,
        "rsna_step1_manifest_sha256": step1_sha,
        "rsna_step2_lock_sha256": calib_report["model_provenance"]["rsna_step2_lock_sha256"],
        "frozen_thresholds": full_taus,
        "calibration_json_path": str(calib_json_p),
        "calibration_json_sha256": compute_sha256_file(calib_json_p),
        "calibration_md_path": str(calib_md_p),
        "calibration_md_sha256": compute_sha256_file(calib_md_p),
    }
    lock_json_p.write_text(json.dumps(calib_lock, indent=2), encoding="utf-8")
    return calib_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify locked NYU RSNA predictions and freeze RSNA thresholds.")
    parser.add_argument("--locked-pred-dir", type=Path, default=Path("reports/paper1/predictions_locked"))
    parser.add_argument("--step1-manifest", type=Path, default=Path("reports/paper1/nyu_step1_rsna_val_calib_manifest.csv"))
    parser.add_argument("--rsna-splits", type=Path, default=Path("datasets/rsna/splits_seed42.csv"))
    parser.add_argument("--rsna-folds", type=Path, default=Path("datasets/paper1/rsna_val_calib_2fold_seed42.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/paper1"))
    parser.add_argument("--expected-batch-exams", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    lock = verify_and_calibrate_nyu_rsna_only(
        locked_pred_dir=args.locked_pred_dir,
        rsna_step1_manifest_csv=args.step1_manifest,
        rsna_splits_csv=args.rsna_splits,
        rsna_fold_csv=args.rsna_folds,
        out_dir=args.out_dir,
        expected_batch_exams=args.expected_batch_exams,
        n_boot=args.n_boot,
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
