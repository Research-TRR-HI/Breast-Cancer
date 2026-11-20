"""Unit tests for Stage C zero-target-label quarantine, prediction locking, and cluster evaluation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.stage_c_eval import (
    FROZEN_RSNA_THRESHOLDS,
    assert_no_forbidden_columns,
    build_label_free_inference_manifest,
    compute_auprc,
    compute_auroc,
    lock_stage_c_predictions,
    verify_and_evaluate_locked_predictions,
    verify_frozen_thresholds_against_stage_b,
)


class TestStageCQuarantineAndEval(unittest.TestCase):
    def test_1_frozen_thresholds_match_committed_stage_b_jsons(self) -> None:
        ce_json = Path("reports/stage_b/rsna_ce_eval_results.json")
        tr_json = Path("reports/stage_b/rsna_trust_eval_results.json")
        if ce_json.exists() and tr_json.exists():
            thr = verify_frozen_thresholds_against_stage_b(ce_json, tr_json)
            self.assertAlmostEqual(thr["ce"]["target_95"], 0.12445068359375, places=14)
            self.assertAlmostEqual(thr["ce"]["target_98"], 0.09619140625, places=14)
            self.assertAlmostEqual(thr["trust"]["target_95"], 0.1954345703125, places=14)
            self.assertAlmostEqual(thr["trust"]["target_98"], 0.1795654296875, places=14)

    def test_2_forbidden_columns_blocked_and_manifest_stripping(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_forbidden_columns(["image_uid", "StudyInstanceUID", "exam_cancer"])
        with self.assertRaises(ValueError):
            assert_no_forbidden_columns(["image_uid", "cohort_group"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            src_csv = tmp_dir / "manifest.csv"
            out_csv = tmp_dir / "inference_manifest_stage_c.csv"
            pd.DataFrame([
                {
                    "cohort_group": "abnormal",
                    "case_id": "p_1",
                    "exam_id": "p_1_v1",
                    "visit_idx": 1,
                    "StudyInstanceUID": "1.2.3",
                    "SeriesInstanceUID": "1.2.3.1",
                    "Image Laterality": "L",
                    "View Position": "CC",
                    "Cancer": 1,
                    "Hard to State Negative": 0,
                    "Manufacturer": "GE MEDICAL SYSTEMS",
                    "ManufacturerModelName": "Senographe",
                    "PhotometricInterpretation": "MONOCHROME2",
                    "rel_path": "abnormal/p_1/left/CC/im1.dcm",
                    "split_cluster_id": "c_1",
                },
                {
                    "cohort_group": "False Positive",
                    "case_id": "p_2",
                    "exam_id": "p_2_v1",
                    "visit_idx": 1,
                    "StudyInstanceUID": "1.2.4",
                    "SeriesInstanceUID": "1.2.4.1",
                    "Image Laterality": "R",
                    "View Position": "MLO",
                    "Cancer": 0,
                    "Hard to State Negative": 1,
                    "Manufacturer": "SIEMENS",
                    "ManufacturerModelName": "Novation",
                    "PhotometricInterpretation": "MONOCHROME2",
                    "rel_path": "False Positive/p_2/right/MLO/im2.dcm",
                    "split_cluster_id": "c_2",
                },
            ]).to_csv(src_csv, index=False)

            meta = build_label_free_inference_manifest(src_csv, out_csv)
            self.assertEqual(meta["n_images"], 2)
            out_df = pd.read_csv(out_csv)
            assert_no_forbidden_columns(out_df.columns)

    def test_3_lock_tamper_detection_and_cluster_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            src_m = tmp_dir / "manifest.csv"
            em_csv = tmp_dir / "exam_manifest.csv"
            inf_csv = tmp_dir / "inference_manifest_stage_c.csv"
            out_dir = tmp_dir / "stage_c"

            # 4 images across 3 unique StudyInstanceUIDs (1 duplicate StudyUID + 1 repeat cluster)
            m_rows = [
                # Study S1 duplicated across alias p_1 and p_1b (testing study dedup without losing views!)
                ("normal", "p_1", "p_1_v1", 1, "S1", "Ser1", "L", "CC", 0, 0, "GE", "M1", "MONOCHROME2", "normal/p_1/1.dcm", "cl_1"),
                ("normal", "p_1b", "p_1b_v1", 1, "S1", "Ser1", "R", "MLO", 0, 0, "GE", "M1", "MONOCHROME2", "normal/p_1b/2.dcm", "cl_1"),
                # Repeat exam S2 in same cluster cl_2, S3 (cancer=1) in cl_2
                ("False Positive", "p_2", "p_2_v1", 1, "S2", "Ser2", "L", "CC", 0, 1, "GE", "M1", "MONOCHROME2", "False Positive/p_2/3.dcm", "cl_2"),
                ("abnormal", "p_2", "p_2_v2", 2, "S3", "Ser3", "R", "CC", 1, 0, "SIEMENS", "M2", "MONOCHROME2", "abnormal/p_2/4.dcm", "cl_2"),
            ]
            cols = [
                "cohort_group", "case_id", "exam_id", "visit_idx", "StudyInstanceUID", "SeriesInstanceUID",
                "Image Laterality", "View Position", "Cancer", "Hard to State Negative",
                "Manufacturer", "ManufacturerModelName", "PhotometricInterpretation", "rel_path", "split_cluster_id"
            ]
            pd.DataFrame(m_rows, columns=cols).to_csv(src_m, index=False)

            em_rows = [
                ("p_1_v1", "p_1", 1, "normal", "S1", 1, 0, 0, "cl_1"),
                ("p_1b_v1", "p_1b", 1, "normal", "S1", 1, 0, 0, "cl_1"),
                ("p_2_v1", "p_2", 1, "False Positive", "S2", 0, 0, 1, "cl_2"),
                ("p_2_v2", "p_2", 2, "abnormal", "S3", 1, 1, 0, "cl_2"),
            ]
            pd.DataFrame(
                em_rows,
                columns=["exam_id", "case_id", "visit_idx", "cohort_group", "StudyInstanceUID", "n_images", "exam_cancer", "hard_to_state_negative", "split_cluster_id"],
            ).to_csv(em_csv, index=False)

            build_label_free_inference_manifest(src_m, inf_csv)
            inf_df = pd.read_csv(inf_csv)

            # Build image predictions mapping Study S1->max(0.05,0.08)=0.08, S2->0.15, S3->0.25
            sc_map = {"S1": (0.08, 0.05), "S2": (0.15, 0.14), "S3": (0.25, 0.30)}
            img_results = []
            for _, r in inf_df.iterrows():
                ce_s, tr_s = sc_map[r["StudyInstanceUID"]]
                img_results.append({
                    "image_uid": r["image_uid"],
                    "decode_status": "ok",
                    "segmentation_status": "success",
                    "used_fallback": False,
                    "border_median": 0.0,
                    "laterality_status": "explicit_L",
                    "score_ce": ce_s,
                    "score_trust": tr_s,
                })

            lock_stage_c_predictions(
                inference_manifest_csv=inf_csv,
                image_results=img_results,
                output_dir=out_dir,
                checkpoint_metadata={"ce_checkpoint_sha256": "abc", "trust_checkpoint_sha256": "def"},
                environment_metadata={"gpu_name": "CPU"},
                thresholds=FROZEN_RSNA_THRESHOLDS,
            )

            ev = verify_and_evaluate_locked_predictions(
                locked_manifest_json=out_dir / "locked_predictions_manifest.json",
                exam_manifest_csv=em_csv,
                n_bootstrap=50,
                bootstrap_seed=42,
            )
            self.assertEqual(ev["cohort_denominators"]["deduplicated_unique_study_exams"], 3)
            self.assertEqual(ev["cohort_denominators"]["unique_split_clusters"], 2)
            self.assertEqual(ev["exam_level_results"]["trust"]["target_98"]["detected_cancers"], 1)
            self.assertEqual(ev["exam_level_results"]["trust"]["target_98"]["n_dismissed"], 2)
            self.assertAlmostEqual(compute_auroc(np.array([0, 0, 1]), np.array([0.05, 0.14, 0.30])), 1.0)
            self.assertAlmostEqual(compute_auprc(np.array([0, 0, 1]), np.array([0.05, 0.14, 0.30])), 1.0)

            # Verify tamper triggers abort BEFORE joining labels
            pred_csv = out_dir / "locked_nlbs_predictions.csv"
            pred_csv.write_text(pred_csv.read_text() + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_and_evaluate_locked_predictions(
                    locked_manifest_json=out_dir / "locked_predictions_manifest.json",
                    exam_manifest_csv=em_csv,
                    n_bootstrap=10,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
