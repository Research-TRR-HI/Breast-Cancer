"""CPU Unit Tests for GMIC Step 3 Evaluation: Calibration Locking, Quarantine & Cohort Matching."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_paper1_gmic_step3 import (
    run_nlbs_evaluation_phase,
    run_rsna_calibration_phase,
    verify_locked_gmic_predictions,
)
from src.training.calibration import compute_target_recall_threshold


class TestGMICStep3Evaluation(unittest.TestCase):
    def test_1_threshold_tie_handling_and_quarantined_calibration_lock(self):
        # 100 positives -> ceil(0.98 * 100 - 1e-12) = 98 retained, max missed = 2 -> index 2 sorted
        pos = np.array([0.01, 0.05, 0.05, 0.12] + [0.5] * 96)
        tau98 = compute_target_recall_threshold(pos, 0.98)
        self.assertAlmostEqual(tau98, 0.05, places=12)
        # Strict dismissal (score < tau): ties at 0.05 are retained -> sensitivity 99/100 >= 0.98
        self.assertGreaterEqual(float(np.mean(pos >= tau98)), 0.98)

    def test_2_phase1_verification_passes_locked_37970_outputs(self):
        locked_dir = ROOT / "reports/paper1/predictions_locked"
        if not (locked_dir / "gmic_nlbs_provenance_lock.json").exists():
            self.skipTest("Locked Step 2 GMIC predictions not present")
        p1 = verify_locked_gmic_predictions(
            locked_pred_dir=locked_dir,
            rsna_splits_csv=ROOT / "datasets/rsna/splits_seed42.csv",
            nlbs_inf_manifest_csv=ROOT / "datasets/nlbs/inference_manifest_stage_c.csv",
            stage_c_locked_csv=ROOT / "reports/stage_c/locked_nlbs_predictions.csv",
        )
        ver = p1["verified"]
        self.assertEqual(ver["rsna_val_calib"]["n_images"], 10982)
        self.assertEqual(ver["rsna_val_calib"]["n_cases"], 2382)
        self.assertEqual(ver["rsna_val_calib"]["n_fallback_views"], 80)
        self.assertEqual(ver["nlbs"]["n_images"], 26988)
        self.assertEqual(ver["nlbs"]["n_cases"], 6359)
        self.assertEqual(ver["nlbs"]["n_fallback_views"], 0)

    def test_3_synthetic_two_phase_tamper_rejection_and_cohort_matching(self):
        rsna_splits = ROOT / "datasets/rsna/splits_seed42.csv"
        rsna_folds = ROOT / "datasets/paper1/rsna_val_calib_2fold_seed42.csv"
        nlbs_inf = ROOT / "datasets/nlbs/inference_manifest_stage_c.csv"
        exam_man = ROOT / "datasets/nlbs/exam_manifest.csv"
        c4v_csv = ROOT / "datasets/paper1/nlbs_4view_common_cohort_label_free.csv"
        elig_csv = ROOT / "datasets/paper1/nlbs_4view_study_eligibility.csv"
        sc_csv = ROOT / "reports/stage_c/locked_nlbs_predictions.csv"

        rsna_vc = pd.read_csv(rsna_splits, usecols=["image_id", "patient_id", "split", "laterality", "view"])
        rsna_vc = rsna_vc[rsna_vc["split"] == "val_calib"].copy()
        rsna_vc["item_id"] = rsna_vc["image_id"].astype(str)
        rsna_vc["case_id"] = rsna_vc["patient_id"].astype(str)
        rng = np.random.default_rng(7)
        rsna_img_df = pd.DataFrame(
            {
                "item_id": rsna_vc["item_id"],
                "case_id": rsna_vc["case_id"],
                "view_slot": rsna_vc["laterality"] + "-" + rsna_vc["view"],
                "benign_pred": rng.uniform(0.05, 0.3, len(rsna_vc)),
                "malignant_pred": rng.uniform(0.01, 0.4, len(rsna_vc)),
                "crop_fallback": False,
            }
        )
        # Mark first 5 RSNA images as fallback to verify calibration sensitivity code
        rsna_img_df.loc[:4, "crop_fallback"] = True
        rsna_case_df = (
            rsna_img_df.groupby("case_id", as_index=False)
            .agg(
                score=("malignant_pred", "max"),
                max_benign_pred=("benign_pred", "max"),
                n_views=("item_id", "count"),
                n_crop_fallbacks=("crop_fallback", "sum"),
            )
            .rename(columns={"case_id": "patient_id"})
        )
        rsna_case_df["has_crop_fallback"] = rsna_case_df["n_crop_fallbacks"] > 0

        n_df = pd.read_csv(nlbs_inf, usecols=["image_uid", "StudyInstanceUID", "laterality", "view_position"])
        n_img_df = pd.DataFrame(
            {
                "item_id": n_df["image_uid"].astype(str),
                "case_id": n_df["StudyInstanceUID"].astype(str),
                "view_slot": n_df["laterality"] + "-" + n_df["view_position"],
                "benign_pred": rng.uniform(0.05, 0.3, len(n_df)),
                "malignant_pred": rng.uniform(0.01, 0.4, len(n_df)),
                "crop_fallback": False,
            }
        )
        n_case_df = (
            n_img_df.groupby("case_id", as_index=False)
            .agg(score=("malignant_pred", "max"), n_crop_fallbacks=("crop_fallback", "sum"))
            .rename(columns={"case_id": "StudyInstanceUID"})
        )
        n_case_df["has_crop_fallback"] = False

        ver_meta = {
            "rsna_val_calib": {"lock_sha256": "syn_rsna_lock"},
            "nlbs": {"lock_sha256": "syn_nlbs_lock"},
        }
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            lock_dir = out_dir / "locked_nlbs"
            p2 = run_rsna_calibration_phase(
                rsna_image_df=rsna_img_df,
                rsna_case_df=rsna_case_df,
                rsna_splits_csv=rsna_splits,
                rsna_fold_csv=rsna_folds,
                nlbs_case_df=n_case_df,
                verification_meta=ver_meta,
                calib_out_dir=out_dir,
                lock_out_dir=lock_dir,
                n_boot=25,
                seed=42,
            )
            # Thresholds frozen & locked BEFORE exam_manifest.csv is ever read
            self.assertTrue(p2["calib_json"].exists())
            self.assertTrue(p2["nlbs_lock_manifest_path"].exists())

            # Tampering calibration JSON must abort Phase 3 before outcome join
            orig = p2["calib_json"].read_text(encoding="utf-8")
            p2["calib_json"].write_text(orig + " ", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                run_nlbs_evaluation_phase(
                    calib_json_path=p2["calib_json"],
                    expected_calib_sha256=p2["calib_json_sha256"],
                    lock_dir=lock_dir,
                    rsna_calib_bundle=p2,
                    nlbs_image_df=n_img_df,
                    nlbs_inf_manifest_csv=nlbs_inf,
                    stage_c_locked_csv=sc_csv,
                    exam_manifest_csv=exam_man,
                    common_4view_csv=c4v_csv,
                    eligibility_csv=elig_csv,
                    out_dir=out_dir,
                    n_boot=25,
                    seed=42,
                )
            p2["calib_json"].write_text(orig, encoding="utf-8")

            summary = run_nlbs_evaluation_phase(
                calib_json_path=p2["calib_json"],
                expected_calib_sha256=p2["calib_json_sha256"],
                lock_dir=lock_dir,
                rsna_calib_bundle=p2,
                nlbs_image_df=n_img_df,
                nlbs_inf_manifest_csv=nlbs_inf,
                stage_c_locked_csv=sc_csv,
                exam_manifest_csv=exam_man,
                common_4view_csv=c4v_csv,
                eligibility_csv=elig_csv,
                out_dir=out_dir,
                n_boot=25,
                seed=42,
            )
            self.assertEqual(summary["nlbs_common_4view_primary"]["n_units"], 6184)
            self.assertEqual(summary["nlbs_common_4view_primary"]["n_clusters"], 5868)
            self.assertEqual(summary["nlbs_common_4view_primary"]["n_positive"], 151)
            self.assertEqual(summary["nlbs_full_cohort_secondary"]["n_units"], 6359)
            self.assertEqual(summary["nlbs_full_cohort_secondary"]["n_clusters"], 5942)
            self.assertIn("ce_b5_minus_gmic_pretrained", summary["nlbs_common_4view_primary"]["paired_contrasts"])
            self.assertIn("trust_b5_minus_gmic_pretrained", summary["nlbs_common_4view_primary"]["paired_contrasts"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
