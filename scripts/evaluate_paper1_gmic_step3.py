"""Paper 1 — GMIC Step 3: Strict Two-Phase CPU Evaluation (RSNA Calibration -> Locked NLBS Eval).

STRICT RULES ENFORCED BY CODE:
    1. CPU ONLY: Zero GPU allocation or cloud jobs. Reuses existing SHA-256-locked Step 2 outputs.
    2. Phase 1 (Verification): Verifies both Step 2 provenance locks and all 37,970 image/case
       predictions (zero missing IDs, zero duplicates, all finite, exact max-view aggregation).
    3. Phase 2 (RSNA Calibration FIRST): Reads RSNA labels ONLY (`splits_seed42.csv` +
       `rsna_val_calib_2fold_seed42.csv`). Computes full `val_calib` thresholds (`tau_98`,
       `tau_95`), independent 2-fold OOF source-test metrics, and 80-fallback calibration sensitivity.
       Writes & SHA-256-locks `gmic_rsna_calibration.json` AND freezes NLBS predictions
       (`lock_nlbs_model_predictions`) BEFORE Phase 3 ever opens `exam_manifest.csv`.
    4. Phase 3 (Canadian NLBS Eval SECOND): Verifies SHA-256 lock manifests prior to label join,
       evaluates Full NLBS (`N=6,359`) + Primary Common 4-View (`N=6,184`, `151` pos, `5,868` clusters)
       + Cluster-Collapsed (`N=5,868` / `5,942`) + sensitivity thresholds, and compares against
       frozen Stage C `ce_b5` and `trust_b5` predictions with paired cluster-bootstrap CIs,
       McNemar tests, KS shift decomposition, and Jaccard/UpSet missed-cancer overlap.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.stage_c_eval import (
    FROZEN_RSNA_THRESHOLDS,
    assert_no_forbidden_columns,
    compute_auprc,
    compute_auroc,
    compute_sha256_file,
    load_label_free_inference_manifest,
)
from src.models.trust_model import aggregate_case_labels
from src.paper1.gmic_adapter import GMIC_CKPT_SHA256, GMIC_MODEL_ID, GMIC_PREPROCESS_VERSION
from src.paper1.six_model_eval import (
    _evaluate_single_cohort_df,
    calibrate_rsna_thresholds_only,
    compute_rsna_to_nlbs_shift_decomposition,
    lock_nlbs_model_predictions,
    verify_and_load_locked_model,
)
from src.training.calibration import compute_target_recall_threshold, evaluate_dismissal_at_threshold

STAGE_C_LOCKED_SHA256 = "526434d058510208ee825b3db36380d0faec7760c17dd2c867dc6288eeb10cc2"


def verify_locked_gmic_predictions(
    locked_pred_dir: Path,
    rsna_splits_csv: Path,
    nlbs_inf_manifest_csv: Path,
    stage_c_locked_csv: Path,
) -> Dict[str, Any]:
    """Verify SHA-256 locks, label quarantine, counts, finiteness, and max-view aggregation for Step 2 outputs."""
    rsna_man = pd.read_csv(
        rsna_splits_csv,
        usecols=["image_id", "patient_id", "split", "laterality", "view"],
    )
    assert_no_forbidden_columns(rsna_man.columns, context="phase1_verify_rsna_manifest")
    rsna_man = rsna_man[rsna_man["split"] == "val_calib"].copy()
    rsna_man["image_id"] = rsna_man["image_id"].astype(str)
    rsna_man["patient_id"] = rsna_man["patient_id"].astype(str)

    nlbs_man = load_label_free_inference_manifest(nlbs_inf_manifest_csv)
    nlbs_man["image_uid"] = nlbs_man["image_uid"].astype(str)
    nlbs_man["StudyInstanceUID"] = nlbs_man["StudyInstanceUID"].astype(str)

    split_specs = (
        ("rsna_val_calib", set(rsna_man["image_id"]), set(rsna_man["patient_id"]), "patient_id"),
        ("nlbs", set(nlbs_man["image_uid"]), set(nlbs_man["StudyInstanceUID"]), "StudyInstanceUID"),
    )

    verified: Dict[str, Any] = {}
    for split_name, exp_imgs, exp_cases, case_col in split_specs:
        lock_path = locked_pred_dir / f"gmic_{split_name}_provenance_lock.json"
        if not lock_path.exists():
            raise FileNotFoundError(f"Missing GMIC provenance lock: {lock_path}")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))

        if lock.get("model_id") != GMIC_MODEL_ID:
            raise RuntimeError(f"{split_name}: model_id mismatch {lock.get('model_id')}")
        if str(lock.get("checkpoint_sha256", "")).lower() != GMIC_CKPT_SHA256:
            raise RuntimeError(f"{split_name}: checkpoint_sha256 mismatch")
        if lock.get("preprocess_version") != GMIC_PREPROCESS_VERSION:
            raise RuntimeError(f"{split_name}: preprocess_version mismatch {lock.get('preprocess_version')}")
        if lock.get("batch_size") != 1:
            raise RuntimeError(f"{split_name}: expected batch_size=1, got {lock.get('batch_size')}")

        file_hashes: Dict[str, str] = {}
        for fname, exp_sha in lock["locked_files_sha256"].items():
            fpath = locked_pred_dir / fname
            got_sha = compute_sha256_file(fpath)
            if got_sha != exp_sha:
                raise RuntimeError(f"SHA-256 tamper/corruption in {fpath}: {got_sha} != {exp_sha}")
            file_hashes[fname] = got_sha

        img_df = pd.read_csv(
            locked_pred_dir / f"gmic_{split_name}_image_scores.csv",
            dtype={"item_id": str, "case_id": str},
        )
        case_df = pd.read_csv(
            locked_pred_dir / f"gmic_{split_name}_case_scores.csv",
            dtype={case_col: str},
        )
        fb_json = json.loads((locked_pred_dir / f"gmic_{split_name}_fallback_log.json").read_text(encoding="utf-8"))

        assert_no_forbidden_columns(img_df.columns, context=f"phase1_{split_name}_images")
        assert_no_forbidden_columns(case_df.columns, context=f"phase1_{split_name}_cases")

        if img_df["item_id"].duplicated().any():
            raise RuntimeError(f"{split_name}: duplicate item_id in image scores")
        if case_df[case_col].duplicated().any():
            raise RuntimeError(f"{split_name}: duplicate {case_col} in case scores")
        if set(img_df["item_id"]) != exp_imgs or len(img_df) != len(exp_imgs):
            raise RuntimeError(f"{split_name}: image ID set mismatch ({len(img_df)} vs {len(exp_imgs)})")
        if set(case_df[case_col]) != exp_cases or len(case_df) != len(exp_cases):
            raise RuntimeError(f"{split_name}: case ID set mismatch ({len(case_df)} vs {len(exp_cases)})")

        for arr_name, arr in (
            ("benign_pred", img_df["benign_pred"].values),
            ("malignant_pred", img_df["malignant_pred"].values),
            ("case_score", case_df["score"].values),
        ):
            if not np.all(np.isfinite(arr)) or np.any((arr < 0.0) | (arr > 1.0)):
                raise RuntimeError(f"{split_name}: invalid/non-finite probabilities in {arr_name}")

        reagg = img_df.groupby("case_id")["malignant_pred"].max().reindex(case_df[case_col]).values
        if not np.allclose(case_df["score"].values, reagg, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"{split_name}: case max-view aggregation mismatch")

        if int(img_df["crop_fallback"].sum()) != int(fb_json["n_fallback_views"]):
            raise RuntimeError(f"{split_name}: fallback count mismatch vs fallback_log.json")

        verified[split_name] = {
            "lock_file": str(lock_path),
            "lock_sha256": compute_sha256_file(lock_path),
            "locked_files_sha256": file_hashes,
            "n_images": int(len(img_df)),
            "n_cases": int(len(case_df)),
            "n_fallback_views": int(fb_json["n_fallback_views"]),
            "n_fallback_cases": int(fb_json["n_fallback_cases"]),
            "preprocess_manifest_sha256s": lock.get("preprocess_manifest_sha256s", []),
            "image_df": img_df,
            "case_df": case_df,
            "fallback_log": fb_json,
        }

    if verified["rsna_val_calib"]["n_images"] + verified["nlbs"]["n_images"] != 37970:
        raise RuntimeError("Expected 37,970 total GMIC verified images")

    stage_c_sha = compute_sha256_file(stage_c_locked_csv)
    if stage_c_sha != STAGE_C_LOCKED_SHA256:
        raise RuntimeError(f"Immutable Stage C lock changed: {stage_c_sha} != {STAGE_C_LOCKED_SHA256}")

    return {
        "verified": verified,
        "stage_c_locked_sha256": stage_c_sha,
    }


def run_rsna_calibration_phase(
    rsna_image_df: pd.DataFrame,
    rsna_case_df: pd.DataFrame,
    rsna_splits_csv: Path,
    rsna_fold_csv: Path,
    nlbs_case_df: pd.DataFrame,
    verification_meta: Dict[str, Any],
    calib_out_dir: Path,
    lock_out_dir: Path,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 2: Calibrate RSNA thresholds ONLY (never accesses NLBS labels), save report, and SHA-lock NLBS."""
    calib_out_dir.mkdir(parents=True, exist_ok=True)
    splits_lbl = pd.read_csv(rsna_splits_csv, usecols=["patient_id", "split", "cancer"])
    vc_lbl = splits_lbl[splits_lbl["split"] == "val_calib"].copy()
    case_lbl_map = aggregate_case_labels(vc_lbl["cancer"].astype(int).tolist(), vc_lbl["patient_id"].astype(str).tolist())

    rsna_cases = rsna_case_df.copy()
    rsna_cases["patient_id_str"] = rsna_cases["patient_id"].astype(str)
    rsna_cases["cancer"] = rsna_cases["patient_id_str"].map(case_lbl_map).astype(int)
    rsna_cases["patient_id_int"] = rsna_cases["patient_id"].astype(int)
    rsna_eval_df = rsna_cases[["patient_id_int", "cancer", "score", "n_views", "n_crop_fallbacks", "has_crop_fallback"]].rename(
        columns={"patient_id_int": "patient_id"}
    )

    rsna_source_eval = calibrate_rsna_thresholds_only(
        rsna_eval_df[["patient_id", "cancer", "score"]],
        fold_csv=rsna_fold_csv,
        n_boot=n_boot,
        seed=seed,
    )

    full_taus = rsna_source_eval["full_val_calib_thresholds"]
    y_r = rsna_eval_df["cancer"].values.astype(np.int64)
    s_r = rsna_eval_df["score"].values.astype(np.float64)
    resub_98 = evaluate_dismissal_at_threshold(s_r, y_r, full_taus["target_98"], 0.99, 0.98)
    resub_95 = evaluate_dismissal_at_threshold(s_r, y_r, full_taus["target_95"], 0.99, 0.95)

    # 80 RSNA preprocessing fallbacks: quantify impact on RSNA AUROC & calibrated thresholds
    fb_imgs = rsna_image_df[rsna_image_df["crop_fallback"].astype(bool)].copy()
    clean_imgs = rsna_image_df[~rsna_image_df["crop_fallback"].astype(bool)].copy()
    fb_cases = rsna_eval_df[rsna_eval_df["has_crop_fallback"].astype(bool)].copy()
    clean_cases = rsna_eval_df[~rsna_eval_df["has_crop_fallback"].astype(bool)].copy()

    # Sensitivity (a): exclude fallback views before case max-aggregation (retaining patients with clean views)
    clean_view_agg = (
        clean_imgs.groupby("case_id", as_index=False)["malignant_pred"]
        .max()
        .rename(columns={"case_id": "patient_id_str", "malignant_pred": "score"})
    )
    clean_view_agg["cancer"] = clean_view_agg["patient_id_str"].map(case_lbl_map).astype(int)
    pos_cv = clean_view_agg[clean_view_agg["cancer"] == 1]["score"].values
    tau98_cv = float(compute_target_recall_threshold(pos_cv, 0.98))
    tau95_cv = float(compute_target_recall_threshold(pos_cv, 0.95))

    # Sensitivity (b): exclude all 36 fallback patients entirely
    pos_cc = clean_cases[clean_cases["cancer"] == 1]["score"].values
    tau98_cc = float(compute_target_recall_threshold(pos_cc, 0.98))
    tau95_cc = float(compute_target_recall_threshold(pos_cc, 0.95))

    fb_sensitivity = {
        "n_fallback_views": int(len(fb_imgs)),
        "n_fallback_patients": int(len(fb_cases)),
        "n_fallback_patients_cancer_positive": int(fb_cases["cancer"].sum()),
        "n_clean_view_agg_patients": int(len(clean_view_agg)),
        "n_clean_view_agg_pos": int(clean_view_agg["cancer"].sum()),
        "clean_views_only_auroc": float(compute_auroc(clean_view_agg["cancer"].values, clean_view_agg["score"].values)),
        "clean_views_only_taus": {"target_98": tau98_cv, "target_95": tau95_cv},
        "clean_cases_only_n": int(len(clean_cases)),
        "clean_cases_only_pos": int(clean_cases["cancer"].sum()),
        "clean_cases_only_auroc": float(compute_auroc(clean_cases["cancer"].values, clean_cases["score"].values)),
        "clean_cases_only_taus": {"target_98": tau98_cc, "target_95": tau95_cc},
        "delta_tau98_clean_views": float(tau98_cv - full_taus["target_98"]),
        "delta_tau95_clean_views": float(tau95_cv - full_taus["target_95"]),
        "delta_tau98_clean_cases": float(tau98_cc - full_taus["target_98"]),
        "delta_tau95_clean_cases": float(tau95_cc - full_taus["target_95"]),
    }

    calib_report = {
        "schema_version": "paper1_gmic_rsna_calibration_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_quarantine_note": "Computed strictly using RSNA val_calib cancer labels ONLY prior to unblinding NLBS outcomes.",
        "model_provenance": {
            "model_id": GMIC_MODEL_ID,
            "checkpoint_file": "sample_model_1.p",
            "checkpoint_sha256": GMIC_CKPT_SHA256,
            "preprocess_version": GMIC_PREPROCESS_VERSION,
            "batch_size": 1,
            "aggregation_rule": "max(malignant_pred across all views per patient_id / StudyInstanceUID)",
            "dismissal_rule": "score < tau -> dismissed; score >= tau -> retained",
            "tie_handling_rule": "np.sort(pos_scores)[C - ceil(target_recall * C - 1e-12)]",
            "rsna_step2_lock_sha256": verification_meta["rsna_val_calib"]["lock_sha256"],
            "nlbs_step2_lock_sha256": verification_meta["nlbs"]["lock_sha256"],
        },
        "rsna_source_eval": rsna_source_eval,
        "full_val_calib_resubstitution_NOT_INDEPENDENT": {
            "note": "Resubstitution metrics on the same 2,382 patients (97 pos) used to freeze NLBS thresholds; NOT independent source testing.",
            "n_patients": int(len(rsna_eval_df)),
            "n_pos": int(y_r.sum()),
            "auroc": float(rsna_source_eval["auroc"]),
            "auprc": float(rsna_source_eval["auprc"]),
            "target_98": resub_98,
            "target_95": resub_95,
        },
        "rsna_80_fallback_calibration_sensitivity": fb_sensitivity,
    }

    calib_json_p = calib_out_dir / "gmic_rsna_calibration.json"
    calib_md_p = calib_out_dir / "gmic_rsna_calibration.md"
    calib_json_p.write_text(json.dumps(calib_report, indent=2), encoding="utf-8")
    calib_md_p.write_text(_format_rsna_calib_markdown(calib_report), encoding="utf-8")

    # Freeze and SHA-256 lock label-free NLBS predictions BEFORE Phase 3 accesses NLBS cancer labels
    nlbs_for_lock = nlbs_case_df[["StudyInstanceUID", "score", "n_crop_fallbacks"]].copy()
    nlbs_lock_manifest = lock_nlbs_model_predictions(
        model_id=GMIC_MODEL_ID,
        exam_scores_df=nlbs_for_lock,
        thresholds_full=full_taus,
        model_sha256=GMIC_CKPT_SHA256,
        out_dir=lock_out_dir,
    )

    return {
        "calib_json": calib_json_p,
        "calib_md": calib_md_p,
        "calib_json_sha256": compute_sha256_file(calib_json_p),
        "calib_report": calib_report,
        "rsna_eval_df": rsna_eval_df,
        "nlbs_lock_manifest": nlbs_lock_manifest,
        "nlbs_lock_manifest_path": lock_out_dir / f"{GMIC_MODEL_ID}_lock_manifest.json",
    }


def _enrich_neg_dismissal(
    cohort_eval: Dict[str, Any],
    merged_df: pd.DataFrame,
    model_ids: Sequence[str],
    thresholds_map: Dict[str, Dict[str, float]],
) -> None:
    """Add specificity (non-cancer exam dismissal rate) cleanly to each model's target_98 / target_95 block."""
    neg_mask = merged_df["exam_cancer"].values.astype(int) == 0
    n_neg = int(np.sum(neg_mask))
    for m in model_ids:
        scores = merged_df[f"{m}_score"].values.astype(np.float64)
        for op in ("target_98", "target_95"):
            tau = float(thresholds_map[m][op])
            n_neg_dis = int(np.sum((scores < tau) & neg_mask))
            cohort_eval["models"][m][op]["n_neg_total"] = n_neg
            cohort_eval["models"][m][op]["n_neg_dismissed"] = n_neg_dis
            cohort_eval["models"][m][op]["non_cancer_dismissal_rate"] = (
                float(n_neg_dis / n_neg) if n_neg > 0 else 0.0
            )


def run_nlbs_evaluation_phase(
    calib_json_path: Path,
    expected_calib_sha256: str,
    lock_dir: Path,
    rsna_calib_bundle: Dict[str, Any],
    nlbs_image_df: pd.DataFrame,
    nlbs_inf_manifest_csv: Path,
    stage_c_locked_csv: Path,
    exam_manifest_csv: Path,
    common_4view_csv: Path,
    eligibility_csv: Path,
    out_dir: Path,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 3: Verify SHA-256 locks first, then join NLBS outcomes and compare CE vs TRUST vs GMIC."""
    got_calib_sha = compute_sha256_file(calib_json_path)
    if got_calib_sha != expected_calib_sha256:
        raise RuntimeError(f"Calibration tamper check failed: {got_calib_sha} != {expected_calib_sha256}")
    calib_report = json.loads(calib_json_path.read_text(encoding="utf-8"))

    gmic_locked_df, gmic_lock_manifest = verify_and_load_locked_model(lock_dir, GMIC_MODEL_ID)
    gmic_taus = calib_report["rsna_source_eval"]["full_val_calib_thresholds"]
    if not math.isclose(
        gmic_lock_manifest["frozen_rsna_thresholds"]["target_98"],
        gmic_taus["target_98"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Frozen NLBS threshold mismatch vs verified RSNA calibration")

    # Verify immutable Stage C locked CSV before loading CE/TRUST scores
    stage_c_sha = compute_sha256_file(stage_c_locked_csv)
    if stage_c_sha != STAGE_C_LOCKED_SHA256:
        raise RuntimeError(f"Immutable Stage C CSV hash mismatch: {stage_c_sha}")
    sc_df = pd.read_csv(stage_c_locked_csv, dtype={"StudyInstanceUID": str})

    # Verify label-free 4-view and full manifest completeness BEFORE outcome join
    c4v_df = pd.read_csv(common_4view_csv, dtype={"StudyInstanceUID": str, "image_uid": str})
    assert_no_forbidden_columns(c4v_df.columns, context="phase3_common_4view")
    elig_df = pd.read_csv(eligibility_csv, dtype={"StudyInstanceUID": str})
    assert_no_forbidden_columns(elig_df.columns, context="phase3_eligibility")
    c4v_uids = set(c4v_df["StudyInstanceUID"].unique().tolist())

    # Deduplicated NLBS exam outcomes (joined ONLY for retrospective evaluation here)
    ex_raw = pd.read_csv(exam_manifest_csv)
    ex_raw["StudyInstanceUID"] = ex_raw["StudyInstanceUID"].astype(str)
    dedup_exam = (
        ex_raw.drop_duplicates(subset=["StudyInstanceUID"], keep="first")[
            ["StudyInstanceUID", "split_cluster_id", "exam_cancer", "cohort_group"]
        ]
        .sort_values("StudyInstanceUID")
        .reset_index(drop=True)
    )
    if len(dedup_exam) != 6359 or int(dedup_exam["exam_cancer"].sum()) != 151:
        raise RuntimeError(f"Unexpected deduplicated NLBS exam counts: {len(dedup_exam)}, {dedup_exam['exam_cancer'].sum()}")

    merged_full = dedup_exam.merge(
        gmic_locked_df.rename(
            columns={"score": "gmic_pretrained_score", "has_crop_fallback": "gmic_pretrained_has_crop_fallback"}
        )[["StudyInstanceUID", "gmic_pretrained_score", "gmic_pretrained_has_crop_fallback"]],
        on="StudyInstanceUID",
        how="inner",
    ).merge(
        sc_df.rename(
            columns={
                "score_ce": "ce_b5_score",
                "score_trust": "trust_b5_score",
                "n_fallback_uncropped": "ce_trust_fb_cnt",
            }
        )[["StudyInstanceUID", "ce_b5_score", "trust_b5_score", "ce_trust_fb_cnt"]],
        on="StudyInstanceUID",
        how="inner",
    )
    merged_full["ce_b5_has_crop_fallback"] = merged_full["ce_trust_fb_cnt"].astype(int) > 0
    merged_full["trust_b5_has_crop_fallback"] = merged_full["ce_trust_fb_cnt"].astype(int) > 0

    if len(merged_full) != 6359:
        raise RuntimeError(f"Full NLBS cohort join mismatch: {len(merged_full)} != 6359")

    merged_common = merged_full[merged_full["StudyInstanceUID"].isin(c4v_uids)].sort_values("StudyInstanceUID").reset_index(drop=True)
    if len(merged_common) != 6184 or int(merged_common["exam_cancer"].sum()) != 151 or merged_common["split_cluster_id"].nunique() != 5868:
        raise RuntimeError("Primary common 4-view NLBS cohort verification failed")

    compare_models = ("ce_b5", "trust_b5", "gmic_pretrained")
    thresholds_map = {
        "ce_b5": FROZEN_RSNA_THRESHOLDS["ce"],
        "trust_b5": FROZEN_RSNA_THRESHOLDS["trust"],
        "gmic_pretrained": gmic_taus,
    }

    common_eval = _evaluate_single_cohort_df(
        merged_common, model_ids=compare_models, thresholds_map=thresholds_map,
        cluster_col="split_cluster_id", n_boot=n_boot, seed=seed,
    )
    _enrich_neg_dismissal(common_eval, merged_common, compare_models, thresholds_map)

    full_eval = _evaluate_single_cohort_df(
        merged_full.sort_values("StudyInstanceUID").reset_index(drop=True),
        model_ids=compare_models, thresholds_map=thresholds_map,
        cluster_col="split_cluster_id", n_boot=n_boot, seed=seed,
    )
    _enrich_neg_dismissal(full_eval, merged_full, compare_models, thresholds_map)

    # Cluster-collapsed sensitivity (both 4-view 5,868 clusters and full 5,942 clusters)
    def _collapse(df_in: pd.DataFrame) -> Dict[str, Any]:
        agg_spec: Dict[str, Any] = {"exam_cancer": "max"}
        for m in compare_models:
            agg_spec[f"{m}_score"] = "max"
            agg_spec[f"{m}_has_crop_fallback"] = "max"
        cdf = df_in.groupby("split_cluster_id", as_index=False).agg(agg_spec).sort_values("split_cluster_id").reset_index(drop=True)
        res = _evaluate_single_cohort_df(
            cdf, model_ids=compare_models, thresholds_map=thresholds_map,
            cluster_col="split_cluster_id", n_boot=n_boot, seed=seed,
        )
        _enrich_neg_dismissal(res, cdf, compare_models, thresholds_map)
        return res

    collapsed_common = _collapse(merged_common)
    collapsed_full = _collapse(merged_full)

    # Shift decomposition (RSNA vs NLBS common 4-view and full)
    rsna_eval_df = rsna_calib_bundle["rsna_eval_df"]
    shift_common = compute_rsna_to_nlbs_shift_decomposition(
        rsna_eval_df, calib_report["rsna_source_eval"], merged_common, common_eval["models"]["gmic_pretrained"], "gmic_pretrained"
    )
    shift_full = compute_rsna_to_nlbs_shift_decomposition(
        rsna_eval_df, calib_report["rsna_source_eval"], merged_full, full_eval["models"]["gmic_pretrained"], "gmic_pretrained"
    )

    # Additional Sensitivity Audits (WITHOUT altering primary protocol)
    fb_sens = calib_report["rsna_80_fallback_calibration_sensitivity"]
    fold_a_taus = calib_report["rsna_source_eval"]["fold_a_sensitivity_thresholds"]
    sens_checks: Dict[str, Any] = {}
    for s_key, s_taus in (
        ("clean_rsna_views_taus_frozen_to_nlbs", fb_sens["clean_views_only_taus"]),
        ("clean_rsna_cases_taus_frozen_to_nlbs", fb_sens["clean_cases_only_taus"]),
        ("fold_a_49pos_taus_frozen_to_nlbs", fold_a_taus),
    ):
        sc_vals = merged_common["gmic_pretrained_score"].values
        y_vals = merged_common["exam_cancer"].values
        sens_checks[s_key] = {
            "target_98": evaluate_dismissal_at_threshold(sc_vals, y_vals, s_taus["target_98"], 0.99, 0.98),
            "target_95": evaluate_dismissal_at_threshold(sc_vals, y_vals, s_taus["target_95"], 0.99, 0.95),
        }

    # Check if GMIC max over canonical 4 selected views differs from max over all NLBS views on N=6,184
    c4v_uids_img = set(c4v_df["image_uid"].astype(str))
    gmic_4v_only = (
        nlbs_image_df[nlbs_image_df["item_id"].astype(str).isin(c4v_uids_img)]
        .groupby("case_id", as_index=False)["malignant_pred"]
        .max()
        .rename(columns={"case_id": "StudyInstanceUID", "malignant_pred": "score_4v_only"})
    )
    chk_4v = merged_common[["StudyInstanceUID", "exam_cancer", "gmic_pretrained_score"]].merge(gmic_4v_only, on="StudyInstanceUID")
    n_diff_4v = int(np.sum(~np.isclose(chk_4v["gmic_pretrained_score"].values, chk_4v["score_4v_only"].values, atol=1e-12)))
    sens_checks["nlbs_selected_4view_max_vs_allview_max_on_6184"] = {
        "n_studies_score_changed": n_diff_4v,
        "auroc_4v_only": float(compute_auroc(chk_4v["exam_cancer"].values, chk_4v["score_4v_only"].values)),
        "target_98_4v_only": evaluate_dismissal_at_threshold(
            chk_4v["score_4v_only"].values, chk_4v["exam_cancer"].values, gmic_taus["target_98"], 0.99, 0.98
        ),
        "target_95_4v_only": evaluate_dismissal_at_threshold(
            chk_4v["score_4v_only"].values, chk_4v["exam_cancer"].values, gmic_taus["target_95"], 0.99, 0.95
        ),
    }

    # Verify cohort subgroup breakdown (cancer vs false-positive vs normal)
    cohort_group_counts = merged_full["cohort_group"].value_counts().to_dict()
    common_group_counts = merged_common["cohort_group"].value_counts().to_dict()

    summary = {
        "schema_version": "paper1_gmic_step3_eval_v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_guarantees": {
            "cpu_only": True,
            "nlbs_labels_used_in_calibration": False,
            "calibration_verified_sha256": got_calib_sha,
            "gmic_locked_nlbs_csv_sha256": gmic_lock_manifest["predictions_sha256"],
            "stage_c_locked_csv_sha256": stage_c_sha,
        },
        "provenance": calib_report["model_provenance"],
        "cohort_integrity": {
            "full_nlbs_cohort_group_counts": cohort_group_counts,
            "common_4view_cohort_group_counts": common_group_counts,
            "excluded_175_ineligible_cancers": int(151 - merged_common["exam_cancer"].sum()),
        },
        "thresholds_used_frozen_from_rsna_val_calib": thresholds_map,
        "rsna_calibration_and_2fold_source_eval": calib_report["rsna_source_eval"],
        "rsna_resubstitution_NOT_INDEPENDENT": calib_report["full_val_calib_resubstitution_NOT_INDEPENDENT"],
        "rsna_80_fallback_calibration_sensitivity": fb_sens,
        "nlbs_common_4view_primary": common_eval,
        "nlbs_full_cohort_secondary": full_eval,
        "nlbs_cluster_collapsed_common_4view": collapsed_common,
        "nlbs_cluster_collapsed_full": collapsed_full,
        "shift_decomposition_gmic": {"common_4view": shift_common, "full_cohort": shift_full},
        "sensitivity_checks": sens_checks,
        "published_trust_reference_disclosure": {
            "source": "Shafie et al. (2026), arXiv:2609.00300v1 Section 4.3 (20% NLBS labeled calibration -> 80% test)",
            "published_trust_nlbs_98_dismissal": 0.1295,
            "published_trust_nlbs_98_cp99_ub": 0.0106,
            "published_trust_nlbs_95_dismissal": 0.1987,
            "published_trust_nlbs_95_cp99_ub": 0.0137,
            "published_ce_nlbs_98_met_target": False,
            "our_benchmark_difference": "OUR CE-only, TRUST, and GMIC runs enforce 0% NLBS label usage before threshold freeze (Protocol 1); results are NEVER adjusted with NLBS labels.",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_json = out_dir / "gmic_step3_nlbs_evaluation.json"
    comp_csv = out_dir / "ce_trust_gmic_comparison_table.csv"
    eval_md = out_dir / "gmic_step3_nlbs_evaluation.md"

    eval_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _build_comparison_csv(summary).to_csv(comp_csv, index=False)
    eval_md.write_text(_format_full_markdown(summary), encoding="utf-8")
    return summary


def _build_comparison_csv(summary: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cohorts = (
        ("common_4view_N6184", summary["nlbs_common_4view_primary"]),
        ("full_nlbs_N6359", summary["nlbs_full_cohort_secondary"]),
        ("cluster_collapsed_4view_N5868", summary["nlbs_cluster_collapsed_common_4view"]),
    )
    for cohort_lbl, cd in cohorts:
        for m in ("ce_b5", "trust_b5", "gmic_pretrained"):
            md = cd["models"][m]
            b = md["bootstrap_95ci"]
            for op in ("target_98", "target_95"):
                t = md[op]
                rows.append(
                    {
                        "cohort": cohort_lbl,
                        "n_units": cd["n_units"],
                        "n_clusters": cd["n_clusters"],
                        "n_pos": cd["n_positive"],
                        "stratum": md["stratum"],
                        "model_id": m,
                        "operating_target": op,
                        "frozen_rsna_tau": t["threshold"],
                        "auroc": md["auroc"],
                        "auroc_ci95_low": b["auroc"]["ci95_low"],
                        "auroc_ci95_high": b["auroc"]["ci95_high"],
                        "auprc": md["auprc"],
                        "sensitivity": t["triage_recall"],
                        "sens_ci95_low": b[f"{op}_sens"]["ci95_low"],
                        "sens_ci95_high": b[f"{op}_sens"]["ci95_high"],
                        "missed_cancers": t["missed_cancers"],
                        "dismissal_rate_all": t["dismissal_rate"],
                        "dis_ci95_low": b[f"{op}_dis"]["ci95_low"],
                        "dis_ci95_high": b[f"{op}_dis"]["ci95_high"],
                        "non_cancer_dismissal_rate": t["non_cancer_dismissal_rate"],
                        "cancer_prev_dismissed": t["cancer_prev_dismissed"],
                        "cp99_ub_dismissed": t["clopper_pearson_99_ub"],
                        "meets_target": t["meets_target_recall"],
                    }
                )
    return pd.DataFrame(rows)


def _format_rsna_calib_markdown(calib: Dict[str, Any]) -> str:
    ev = calib["rsna_source_eval"]
    oof = ev["oof_pooled"]
    b = ev["bootstrap_95ci"]
    resub = calib["full_val_calib_resubstitution_NOT_INDEPENDENT"]
    taus = ev["full_val_calib_thresholds"]
    fb = calib["rsna_80_fallback_calibration_sensitivity"]
    lines = [
        "# Paper 1 — GMIC Step 3 Phase 2: RSNA Source Threshold Calibration (CPU ONLY)",
        "",
        f"**Model:** `{calib['model_provenance']['model_id']}` (`{calib['model_provenance']['checkpoint_file']}`, SHA-256 `{calib['model_provenance']['checkpoint_sha256']}`)  ",
        f"**Quarantine:** RSNA labels ONLY (`val_calib`: `{ev['n_patients']}` patients / `{ev['n_pos']}` cancers). Zero NLBS outcomes read.",
        "",
        "## 1. Frozen NLBS Operating Thresholds (`full val_calib`, NOT Independent Evaluation)",
        f"* **Primary `target_98` (`tau_98`):** `{taus['target_98']:.8f}` -> resubstitution RSNA sens `{resub['target_98']['triage_recall']*100:.2f}%` (`{resub['target_98']['missed_cancers']}/97` missed), dismissal `{resub['target_98']['dismissal_rate']*100:.2f}%`",
        f"* **Secondary `target_95` (`tau_95`):** `{taus['target_95']:.8f}` -> resubstitution RSNA sens `{resub['target_95']['triage_recall']*100:.2f}%` (`{resub['target_95']['missed_cancers']}/97` missed), dismissal `{resub['target_95']['dismissal_rate']*100:.2f}%`",
        "",
        "## 2. Independent 2-Fold Patient-Disjoint RSNA Source Test",
        f"* **RSNA `val_calib` Discrimination:** AUROC = **`{ev['auroc']:.4f}`** (95% CI `[{b['auroc']['ci95_low']:.4f}, {b['auroc']['ci95_high']:.4f}]`), AUPRC = `{ev['auprc']:.4f}` (`[{b['auprc']['ci95_low']:.4f}, {b['auprc']['ci95_high']:.4f}]`)",
        f"* **2-Fold OOF Pooled `@98%`:** Sensitivity = **`{oof['target_98_sens']*100:.2f}%`** (`{oof['target_98_missed']}/97` missed, CI `[{b['s98']['ci95_low']*100:.2f}%, {b['s98']['ci95_high']*100:.2f}%]`), Dismissal = **`{oof['target_98_dis']*100:.2f}%`** (`[{b['d98']['ci95_low']*100:.2f}%, {b['d98']['ci95_high']*100:.2f}%]`)",
        f"* **2-Fold OOF Pooled `@95%`:** Sensitivity = **`{oof['target_95_sens']*100:.2f}%`** (`{oof['target_95_missed']}/97` missed, CI `[{b['s95']['ci95_low']*100:.2f}%, {b['s95']['ci95_high']*100:.2f}%]`), Dismissal = **`{oof['target_95_dis']*100:.2f}%`** (`[{b['d95']['ci95_low']*100:.2f}%, {b['d95']['ci95_high']*100:.2f}%]`)",
        "",
        "## 3. RSNA `80`-Fallback Calibration Sensitivity Check",
        f"* **Fallback Cases:** `{fb['n_fallback_views']}` views across `{fb['n_fallback_patients']}` patients (`{fb['n_fallback_patients_cancer_positive']}` cancer-positive).",
        f"* **Clean Views Only (`N={fb['n_clean_view_agg_patients']}`, `{fb['n_clean_view_agg_pos']}` pos):** AUROC = `{fb['clean_views_only_auroc']:.4f}`, `tau_98 = {fb['clean_views_only_taus']['target_98']:.8f}` (delta `{fb['delta_tau98_clean_views']:+.8f}`), `tau_95 = {fb['clean_views_only_taus']['target_95']:.8f}` (delta `{fb['delta_tau95_clean_views']:+.8f}`).",
        f"* **Clean Patients Only (`N={fb['clean_cases_only_n']}`, `{fb['clean_cases_only_pos']}` pos):** AUROC = `{fb['clean_cases_only_auroc']:.4f}`, `tau_98 = {fb['clean_cases_only_taus']['target_98']:.8f}` (delta `{fb['delta_tau98_clean_cases']:+.8f}`), `tau_95 = {fb['clean_cases_only_taus']['target_95']:.8f}` (delta `{fb['delta_tau95_clean_cases']:+.8f}`).",
    ]
    return "\n".join(lines) + "\n"


def _format_full_markdown(summary: Dict[str, Any]) -> str:
    c4 = summary["nlbs_common_4view_primary"]
    cf = summary["nlbs_full_cohort_secondary"]
    taus = summary["thresholds_used_frozen_from_rsna_val_calib"]
    rsna = summary["rsna_calibration_and_2fold_source_eval"]
    fb = summary["rsna_80_fallback_calibration_sensitivity"]
    ov = c4["missed_cancer_overlap"]["target_98"]
    sh = summary["shift_decomposition_gmic"]["common_4view"]

    def _row(cd: Dict[str, Any], m: str, op: str, lbl: str) -> str:
        md = cd["models"][m]
        b = md["bootstrap_95ci"]
        t = md[op]
        met = "YES" if t["meets_target_recall"] else "NO"
        return (
            f"| {lbl} | {md['auroc']:.4f} [{b['auroc']['ci95_low']:.4f}, {b['auroc']['ci95_high']:.4f}] | "
            f"{t['threshold']:.6f} | **{t['triage_recall']*100:.2f}%** [{b[f'{op}_sens']['ci95_low']*100:.2f}%, {b[f'{op}_sens']['ci95_high']*100:.2f}%] | "
            f"{t['missed_cancers']}/151 | {t['dismissal_rate']*100:.2f}% ({t['non_cancer_dismissal_rate']*100:.2f}% neg) | "
            f"{t['clopper_pearson_99_ub']*100:.2f}% | **{met}** |"
        )

    lines = [
        "# Paper 1 — GMIC Step 3 External Validation Report (RSNA -> Canadian NLBS)",
        "",
        "**Execution Mode:** CPU Only (`0` GPU / `USD 0.00` cloud spend) | **GMIC Checkpoint:** `sample_model_1.p` (`sha256: " + summary["provenance"]["checkpoint_sha256"][:16] + "...`)  ",
        "**Strict Quarantine:** RSNA thresholds frozen & SHA-256 locked (`" + summary["strict_guarantees"]["calibration_verified_sha256"][:16] + "...`) BEFORE joining NLBS outcomes.",
        "",
        "## 1. RSNA Calibration vs. Independent 2-Fold OOF Source Performance (GMIC)",
        f"* **Frozen NLBS Operating Points (`val_calib`, 97 pos — resubstitution, NOT independent):** `tau_98 = {taus['gmic_pretrained']['target_98']:.6f}` (`98.97%` sens / `{summary['rsna_resubstitution_NOT_INDEPENDENT']['target_98']['dismissal_rate']*100:.2f}%` dis), `tau_95 = {taus['gmic_pretrained']['target_95']:.6f}` (`95.88%` sens / `{summary['rsna_resubstitution_NOT_INDEPENDENT']['target_95']['dismissal_rate']*100:.2f}%` dis).",
        f"* **Independent 2-Fold Patient-Disjoint RSNA OOF (`2,382` pts / `97` pos):** AUROC **`{rsna['auroc']:.4f}`** (`[{rsna['bootstrap_95ci']['auroc']['ci95_low']:.4f}, {rsna['bootstrap_95ci']['auroc']['ci95_high']:.4f}]`), `@98%` OOF sens **`{rsna['oof_pooled']['target_98_sens']*100:.2f}%`** (`{rsna['oof_pooled']['target_98_missed']}/97` missed, dismissal `{rsna['oof_pooled']['target_98_dis']*100:.2f}%`), `@95%` OOF sens **`{rsna['oof_pooled']['target_95_sens']*100:.2f}%`** (`{rsna['oof_pooled']['target_95_missed']}/97` missed, dismissal `{rsna['oof_pooled']['target_95_dis']*100:.2f}%`).",
        f"* **RSNA `80`-Fallback Calibration Audit (`{fb['n_fallback_patients']}` pts, `{fb['n_fallback_patients_cancer_positive']}` pos):** Excluding fallback views changes `tau_98` by `{fb['delta_tau98_clean_views']:+.6f}` and excluding fallback cases changes `tau_98` by `{fb['delta_tau98_clean_cases']:+.6f}` (sensitivity checks below).",
        "",
        f"## 2. Primary Common 4-View NLBS Comparison: OUR CE vs. TRUST vs. GMIC (`N = {c4['n_units']}` studies, `{c4['n_clusters']}` clusters, `{c4['n_positive']}` cancers)",
        "",
        "> **Disclosure:** `ce_b5` and `trust_b5` below are OUR Stage C zero-NLBS-label runs (`sha256: 526434d0...`), NOT the published TRUST (2026) 20%-target-calibrated external result (`12.95%` @98%, `19.87%` @95%; CE failed 98%). `gmic_pretrained` is official Stratum B pretrained weights (`nyukat/GMIC` `sample_model_1.p`, trained on NYU 2010–2017; `0` NLBS labels touched).",
        "",
        "### At Frozen Source 98% Sensitivity Threshold (`target_98` — Primary)",
        "| Model (Stratum) | NLBS AUROC [95% CI] | Frozen `tau_98` | Sensitivity [95% Cluster CI] | Missed | Dismissal (Neg Dis) | CP99 UB | Met 98%? |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | :---: |",
        _row(c4, "ce_b5", "target_98", "**CE-only** (Stratum A · OUR)"),
        _row(c4, "trust_b5", "target_98", "**TRUST** (Stratum A · OUR)"),
        _row(c4, "gmic_pretrained", "target_98", "**GMIC** (Stratum B · Pretrained)"),
        "",
        "### At Frozen Source 95% Sensitivity Threshold (`target_95` — Secondary)",
        "| Model (Stratum) | NLBS AUROC [95% CI] | Frozen `tau_95` | Sensitivity [95% Cluster CI] | Missed | Dismissal (Neg Dis) | CP99 UB | Met 95%? |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | :---: |",
        _row(c4, "ce_b5", "target_95", "**CE-only** (Stratum A · OUR)"),
        _row(c4, "trust_b5", "target_95", "**TRUST** (Stratum A · OUR)"),
        _row(c4, "gmic_pretrained", "target_95", "**GMIC** (Stratum B · Pretrained)"),
        "",
        f"## 3. Secondary Full NLBS Cohort (`N = {cf['n_units']}` studies, `{cf['n_clusters']}` clusters, `{cf['n_positive']}` cancers)",
        "| Model | Target | Frozen `tau` | Sensitivity [95% CI] | Missed | Dismissal (Neg Dis) | AUROC [95% CI] | Met Target? |",
        "| :--- | :---: | ---: | :--- | ---: | ---: | :--- | :---: |",
    ]
    for m in ("ce_b5", "trust_b5", "gmic_pretrained"):
        md = cf["models"][m]
        b = md["bootstrap_95ci"]
        auc_s = f"{md['auroc']:.4f} [{b['auroc']['ci95_low']:.4f}, {b['auroc']['ci95_high']:.4f}]"
        for op, tlbl in (("target_98", "98%"), ("target_95", "95%")):
            t = md[op]
            lines.append(
                f"| `{m}` | {tlbl} | {t['threshold']:.6f} | {t['triage_recall']*100:.2f}% [{b[f'{op}_sens']['ci95_low']*100:.2f}%, {b[f'{op}_sens']['ci95_high']*100:.2f}%] | "
                f"{t['missed_cancers']}/151 | {t['dismissal_rate']*100:.2f}% ({t['non_cancer_dismissal_rate']*100:.2f}%) | {auc_s} | **{'YES' if t['meets_target_recall'] else 'NO'}** |"
            )

    lines.extend(["", "## 4. Paired Differences (`B = 2,000` cluster bootstrap) & Missed-Cancer Overlap (`N = 6,184`)"])
    for key in ("ce_b5_minus_gmic_pretrained", "trust_b5_minus_gmic_pretrained", "ce_b5_minus_trust_b5"):
        pc = c4["paired_contrasts"][key]
        mc = pc["mcnemar_cancer_retained_98"]
        lines.append(
            f"* **`{key}`:** ΔAUROC = `{pc['delta_auroc']['mean']:+.4f}` (`[{pc['delta_auroc']['ci95_low']:+.4f}, {pc['delta_auroc']['ci95_high']:+.4f}]`), "
            f"ΔSens@98 = `{pc['delta_sens98']['mean']*100:+.2f} pp` (`[{pc['delta_sens98']['ci95_low']*100:+.2f}, {pc['delta_sens98']['ci95_high']*100:+.2f}]`), "
            f"ΔDis@98 = `{pc['delta_dis98']['mean']*100:+.2f} pp` (`[{pc['delta_dis98']['ci95_low']*100:+.2f}, {pc['delta_dis98']['ci95_high']*100:+.2f}]`), "
            f"McNemar cancers@98 `p = {mc['p_value']:.4f}` (`only_a={mc['only_a']}, only_b={mc['only_b']}`)"
        )
    lines.extend([
        f"* **GMIC RSNA -> NLBS Shift Decomposition (`N=6,184`):** ΔAUROC = `{sh['delta_auroc_nlbs_minus_rsna']:+.4f}`, ΔSens@98 vs RSNA OOF = `{sh['delta_sens98_nlbs_minus_rsna']*100:+.2f} pp`, "
        f"neg median shift = `{sh['neg_median_shift']:+.6f}` (`KS p={sh['ks_neg_pvalue']:.3e}`), pos median shift = `{sh['pos_median_shift']:+.6f}` (`KS p={sh['ks_pos_pvalue']:.3e}`).",
        f"* **Missed Cancers at `@98%` (`151` pos):** CE=`{ov['missed_counts']['ce_b5']}`, TRUST=`{ov['missed_counts']['trust_b5']}`, GMIC=`{ov['missed_counts']['gmic_pretrained']}` | UpSet patterns: `{json.dumps(ov['upset_patterns'])}`.",
        f"* **Jaccard Overlap (`@98%` missed cancers):** CE–TRUST `{ov['pairwise_jaccard']['ce_b5']['trust_b5']:.3f}`, CE–GMIC `{ov['pairwise_jaccard']['ce_b5']['gmic_pretrained']:.3f}`, TRUST–GMIC `{ov['pairwise_jaccard']['trust_b5']['gmic_pretrained']:.3f}`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="GMIC Step 3 CPU-only evaluation (RSNA calibration -> locked NLBS eval).")
    parser.add_argument("--locked-pred-dir", type=Path, default=Path("reports/paper1/predictions_locked"))
    parser.add_argument("--rsna-splits", type=Path, default=Path("datasets/rsna/splits_seed42.csv"))
    parser.add_argument("--rsna-fold-csv", type=Path, default=Path("datasets/paper1/rsna_val_calib_2fold_seed42.csv"))
    parser.add_argument("--nlbs-inf-manifest", type=Path, default=Path("datasets/nlbs/inference_manifest_stage_c.csv"))
    parser.add_argument("--stage-c-locked-csv", type=Path, default=Path("reports/stage_c/locked_nlbs_predictions.csv"))
    parser.add_argument("--exam-manifest", type=Path, default=Path("datasets/nlbs/exam_manifest.csv"))
    parser.add_argument("--common-4view-csv", type=Path, default=Path("datasets/paper1/nlbs_4view_common_cohort_label_free.csv"))
    parser.add_argument("--eligibility-csv", type=Path, default=Path("datasets/paper1/nlbs_4view_study_eligibility.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/paper1/gmic_step3_eval"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Phase 1: Verify Step 2 locked outputs strictly label-free
    p1 = verify_locked_gmic_predictions(
        locked_pred_dir=args.locked_pred_dir,
        rsna_splits_csv=args.rsna_splits,
        nlbs_inf_manifest_csv=args.nlbs_inf_manifest,
        stage_c_locked_csv=args.stage_c_locked_csv,
    )
    ver = p1["verified"]

    # Phase 2: RSNA calibration FIRST (zero NLBS cancer outcomes accessed)
    p2 = run_rsna_calibration_phase(
        rsna_image_df=ver["rsna_val_calib"]["image_df"],
        rsna_case_df=ver["rsna_val_calib"]["case_df"],
        rsna_splits_csv=args.rsna_splits,
        rsna_fold_csv=args.rsna_fold_csv,
        nlbs_case_df=ver["nlbs"]["case_df"],
        verification_meta=ver,
        calib_out_dir=args.out_dir,
        lock_out_dir=args.out_dir / "locked_nlbs",
        n_boot=args.n_boot,
        seed=args.seed,
    )

    # Phase 3: Canadian NLBS evaluation SECOND (verifies SHA-256 locks before unblinding outcomes)
    summary = run_nlbs_evaluation_phase(
        calib_json_path=p2["calib_json"],
        expected_calib_sha256=p2["calib_json_sha256"],
        lock_dir=args.out_dir / "locked_nlbs",
        rsna_calib_bundle=p2,
        nlbs_image_df=ver["nlbs"]["image_df"],
        nlbs_inf_manifest_csv=args.nlbs_inf_manifest,
        stage_c_locked_csv=args.stage_c_locked_csv,
        exam_manifest_csv=args.exam_manifest,
        common_4view_csv=args.common_4view_csv,
        eligibility_csv=args.eligibility_csv,
        out_dir=args.out_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "rsna_auroc": summary["rsna_calibration_and_2fold_source_eval"]["auroc"],
                "gmic_tau98": summary["thresholds_used_frozen_from_rsna_val_calib"]["gmic_pretrained"]["target_98"],
                "gmic_tau95": summary["thresholds_used_frozen_from_rsna_val_calib"]["gmic_pretrained"]["target_95"],
                "nlbs_6184_gmic_auroc": summary["nlbs_common_4view_primary"]["models"]["gmic_pretrained"]["auroc"],
                "nlbs_6184_gmic_sens98": summary["nlbs_common_4view_primary"]["models"]["gmic_pretrained"]["target_98"]["triage_recall"],
                "nlbs_6184_gmic_dis98": summary["nlbs_common_4view_primary"]["models"]["gmic_pretrained"]["target_98"]["dismissal_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
