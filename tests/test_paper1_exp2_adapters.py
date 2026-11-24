"""Offline CPU Unit Tests for Paper 1 Experiment 2 Adapters, Quarantine & Guards."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_paper1_exp2_modal import build_unit_work_items, validate_exp2_cli_and_guard
from src.evaluation.stage_c_eval import compute_sha256_file
from src.paper1.exp2_adapters import (
    EXP2_END2END_INPUT_SIZE,
    EXP2_PETRINI_INPUT_SIZE,
    aggregate_exp2_exam_scores,
    apply_threshold_policy,
    assert_model_eligible,
    build_petrini_breast_pairs,
    build_petrini_twoview_tensor,
    compute_unit_ids_sha256,
    finalize_and_lock_exp2_predictions,
    load_exp2_nlbs_manifest,
    preprocess_end2end_from_stage1_uint16,
    preprocess_glam_from_gmic_uint16,
    preprocess_mammoscreen_view_from_stage1_uint16,
    validate_exp2_shard_payload,
    verify_checkpoint_sha256,
)


class TestPaper1Exp2Adapters(unittest.TestCase):
    def test_1_ineligible_model_blocked_and_eligible_checked(self):
        with self.assertRaises(RuntimeError):
            assert_model_eligible("frcnn_cad")
        spec = assert_model_eligible("glam_pretrained")
        self.assertEqual(spec.threshold_category, "C")
        spec_r1 = assert_model_eligible("mammoscreen_r1")
        self.assertEqual(spec_r1.threshold_category, "A")

    def test_2_real_nlbs_manifests_quarantine_and_petrini_pairing(self):
        df_4v = load_exp2_nlbs_manifest(
            ROOT / "datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
            model_id="petrini_twoview_efnb3",
        )
        self.assertEqual(len(df_4v), 6184 * 4)
        pairs = build_petrini_breast_pairs(df_4v)
        self.assertEqual(len(pairs), 6184 * 2)  # 2 breasts (L, R) per 4-view study
        self.assertEqual(pairs["StudyInstanceUID"].nunique(), 6184)

        items, studies = build_unit_work_items("glam_pretrained", "nlbs_full", root=ROOT)
        self.assertEqual(len(items), 26988)
        self.assertEqual(len(studies), 6359)

    def test_3_manifest_rejects_forbidden_labels_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as td:
            bad_label_csv = Path(td) / "bad_labels.csv"
            pd.DataFrame(
                [
                    {
                        "image_uid": "u1",
                        "StudyInstanceUID": "s1",
                        "laterality": "L",
                        "view_position": "CC",
                        "dicom_rel_path": "a.dcm",
                        "exam_cancer": "1",
                    }
                ]
            ).to_csv(bad_label_csv, index=False)
            with self.assertRaises(ValueError):
                load_exp2_nlbs_manifest(bad_label_csv, model_id="glam_pretrained")

            dup_csv = Path(td) / "dup_uid.csv"
            pd.DataFrame(
                [
                    {"image_uid": "u1", "StudyInstanceUID": "s1", "laterality": "L", "view_position": "CC", "dicom_rel_path": "a.dcm"},
                    {"image_uid": "u1", "StudyInstanceUID": "s1", "laterality": "L", "view_position": "MLO", "dicom_rel_path": "b.dcm"},
                ]
            ).to_csv(dup_csv, index=False)
            with self.assertRaises(ValueError):
                load_exp2_nlbs_manifest(dup_csv, model_id="glam_pretrained")

    def test_4_checkpoint_sha256_verification(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "dummy.ckpt"
            ckpt.write_bytes(b"glam-weights-payload")
            expected = hashlib.sha256(b"glam-weights-payload").hexdigest()
            self.assertEqual(verify_checkpoint_sha256(ckpt, expected), expected)
            with self.assertRaises(RuntimeError):
                verify_checkpoint_sha256(ckpt, "0" * 64)
            with self.assertRaises(RuntimeError):
                verify_checkpoint_sha256(ckpt, "UNKNOWN", allow_unknown=False)

    def test_5_preprocessing_shapes_norms_and_petrini_channel_order(self):
        gmic_u16 = np.full((2944, 1920), 1000, dtype=np.uint16)
        gmic_u16[0, 0] = 50000
        glam_norm = preprocess_glam_from_gmic_uint16(gmic_u16)
        self.assertEqual(glam_norm.shape, (2944, 1920))
        self.assertEqual(glam_norm.dtype, np.float32)
        self.assertAlmostEqual(float(np.mean(glam_norm)), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(glam_norm)), 1.0, places=4)

        cc_u16 = np.full((1520, 912), 13369, dtype=np.uint16)
        mlo_u16 = np.full((1520, 912), 26738, dtype=np.uint16)
        t_cc_mlo = build_petrini_twoview_tensor(cc_u16, mlo_u16, dataset="CBIS-DDSM")
        t_mlo_cc = build_petrini_twoview_tensor(mlo_u16, cc_u16, dataset="CBIS-DDSM")
        self.assertEqual(tuple(t_cc_mlo.shape), (6, *EXP2_PETRINI_INPUT_SIZE))
        # Strict [CC_RGB, MLO_RGB] channel order: CC channels ~ 0.0, MLO channels > 0.0
        self.assertAlmostEqual(float(t_cc_mlo[0:3].abs().max()), 0.0, places=6)
        self.assertGreater(float(t_cc_mlo[3:6].mean()), 0.15)
        self.assertGreater(float((t_cc_mlo - t_mlo_cc).abs().max()), 0.15)

        e2e = preprocess_end2end_from_stage1_uint16(cc_u16)
        self.assertEqual(e2e.shape, (3, *EXP2_END2END_INPUT_SIZE))
        self.assertEqual(e2e.dtype, np.float32)

        ms = preprocess_mammoscreen_view_from_stage1_uint16(mlo_u16)
        self.assertEqual(ms.shape, (1520, 912))
        self.assertEqual(ms.dtype, np.uint8)

    def test_6_aggregation_and_threshold_category_enforcement(self):
        # Image-level max aggregation (GLAM / End2End)
        img_df = pd.DataFrame(
            [
                {"StudyInstanceUID": "s1", "image_uid": "i1", "score": 0.10},
                {"StudyInstanceUID": "s1", "image_uid": "i2", "score": 0.42},
                {"StudyInstanceUID": "s2", "image_uid": "i3", "score": 0.005},
            ]
        )
        exam_img = aggregate_exp2_exam_scores(img_df, "glam_pretrained")
        self.assertAlmostEqual(float(exam_img.set_index("StudyInstanceUID").loc["s1", "score"]), 0.42)

        # Breast-level max(L, R) aggregation (Petrini / MammoScreen)
        br_df = pd.DataFrame(
            [
                {"StudyInstanceUID": "s1", "laterality": "L", "score": 0.02},
                {"StudyInstanceUID": "s1", "laterality": "R", "score": 0.004},
                {"StudyInstanceUID": "s2", "laterality": "L", "score": 0.003},
                {"StudyInstanceUID": "s2", "laterality": "R", "score": 0.006},
            ]
        )
        exam_br = aggregate_exp2_exam_scores(br_df, "mammoscreen_r1")
        self.assertAlmostEqual(float(exam_br.set_index("StudyInstanceUID").loc["s1", "score"]), 0.02)

        # Category A uses published thresholds (0.0072 / 0.0127)
        t_df, t_meta = apply_threshold_policy(exam_br, "mammoscreen_r1")
        self.assertEqual(t_meta["threshold_category"], "A")
        self.assertFalse(bool(t_df.set_index("StudyInstanceUID").loc["s1", "dismissed_98"]))
        self.assertTrue(bool(t_df.set_index("StudyInstanceUID").loc["s2", "dismissed_98"]))

        # Category C refuses RSNA/NLBS thresholds and emits threshold-independent metadata only
        c_df, c_meta = apply_threshold_policy(exam_img, "glam_pretrained")
        self.assertEqual(c_meta["threshold_category"], "C")
        self.assertIsNone(c_meta["target_98"])
        self.assertNotIn("dismissed_98", c_df.columns)
        with self.assertRaises(ValueError):
            apply_threshold_policy(
                exam_img,
                "glam_pretrained",
                external_thresholds={"calibration_domain": "RSNA_val_calib", "target_98": 0.1, "target_95": 0.2},
            )

        # Duplicate image_uid / (StudyInstanceUID, laterality) rejected in aggregation
        dup_img = pd.concat([img_df, img_df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            aggregate_exp2_exam_scores(dup_img, "glam_pretrained")

        # Category B enforcement (mock Category B spec via registry patch)
        from unittest.mock import patch
        from src.paper1 import exp2_adapters as ea

        cat_b_spec = ea.Exp2ModelSpec(
            model_id="mock_cat_b",
            display_name="Mock Cat B",
            eligibility="eligible_new_inference",
            threshold_category="B",
            official_repo="https://example.org",
            repo_revision="main",
            checkpoint_name="mock.pt",
            checkpoint_sha256="a" * 64,
            license="MIT",
            training_dataset="CBIS-DDSM",
            input_level="image",
            nlbs_cache_source="stage1_1520x912",
            aggregation_rule="max",
        )
        with patch.dict(ea.EXP2_MODEL_REGISTRY, {"mock_cat_b": cat_b_spec}):
            with self.assertRaises(ValueError):
                apply_threshold_policy(
                    exam_img,
                    "mock_cat_b",
                    external_thresholds={"calibration_domain": "RSNA", "target_98": 0.1, "target_95": 0.2},
                )
            with self.assertRaises(ValueError):
                apply_threshold_policy(
                    exam_img,
                    "mock_cat_b",
                    external_thresholds={"calibration_domain": "NLBS", "target_98": 0.1, "target_95": 0.2},
                )
            b_df, b_meta = apply_threshold_policy(
                exam_img,
                "mock_cat_b",
                external_thresholds={"calibration_domain": "CBIS_DDSM_TEST", "target_98": 0.15, "target_95": 0.30},
            )
            self.assertEqual(b_meta["threshold_category"], "B")
            self.assertTrue(bool(b_df.set_index("StudyInstanceUID").loc["s2", "dismissed_98"]))

    def test_7_shard_resume_validation_and_prediction_locking(self):
        unit_ids = ["s1_L", "s1_R", "s2_L", "s2_R"]
        sha = "a" * 64
        payload = {
            "schema_version": "paper1_exp2_shard_v1",
            "model_id": "petrini_twoview_efnb3",
            "checkpoint_sha256": sha,
            "split_name": "nlbs_4view",
            "shard_idx": 0,
            "num_shards": 1,
            "expected_unit_ids_sha256": compute_unit_ids_sha256(unit_ids),
            "records": [
                {"unit_id": "s1_L", "StudyInstanceUID": "s1", "laterality": "L", "score": 0.05},
                {"unit_id": "s1_R", "StudyInstanceUID": "s1", "laterality": "R", "score": 0.15},
                {"unit_id": "s2_L", "StudyInstanceUID": "s2", "laterality": "L", "score": 0.01},
                {"unit_id": "s2_R", "StudyInstanceUID": "s2", "laterality": "R", "score": 0.02},
            ],
        }
        ok, reason = validate_exp2_shard_payload(
            payload,
            model_id="petrini_twoview_efnb3",
            expected_unit_ids=unit_ids,
            split_name="nlbs_4view",
            shard_idx=0,
            num_shards=1,
            checkpoint_sha256=sha,
        )
        self.assertTrue(ok, reason)

        bad_payload = json.loads(json.dumps(payload))
        bad_payload["records"][0]["score"] = float("nan")
        ok_bad, _ = validate_exp2_shard_payload(
            bad_payload,
            model_id="petrini_twoview_efnb3",
            expected_unit_ids=unit_ids,
            split_name="nlbs_4view",
            shard_idx=0,
            num_shards=1,
            checkpoint_sha256=sha,
        )
        self.assertFalse(ok_bad)

        with tempfile.TemporaryDirectory() as td:
            lock = finalize_and_lock_exp2_predictions(
                model_id="petrini_twoview_efnb3",
                split_name="nlbs_4view",
                shard_outputs=[payload],
                expected_studies=["s1", "s2"],
                checkpoint_sha256=sha,
                out_dir=Path(td),
            )
            case_csv = Path(td) / "petrini_twoview_efnb3_nlbs_4view_case_scores.csv"
            self.assertEqual(compute_sha256_file(case_csv), lock["locked_files_sha256"][case_csv.name])

    def test_8_cli_approval_and_ineligibility_guards(self):
        # Plan is allowed without approval flags
        validate_exp2_cli_and_guard(
            stage="plan",
            model_id="glam_pretrained",
            split="nlbs_4view",
            num_shards=4,
            batch_size=8,
            pilot_limit=200,
            approve_pilot=False,
            approve_full_infer=False,
            max_total_spend_usd=5.0,
        )
        # Pilot & infer blocked without explicit approval
        with self.assertRaises(RuntimeError):
            validate_exp2_cli_and_guard(
                stage="pilot",
                model_id="glam_pretrained",
                split="nlbs_4view",
                num_shards=4,
                batch_size=8,
                pilot_limit=200,
                approve_pilot=False,
                approve_full_infer=False,
                max_total_spend_usd=5.0,
            )
        with self.assertRaises(RuntimeError):
            validate_exp2_cli_and_guard(
                stage="infer",
                model_id="glam_pretrained",
                split="nlbs_4view",
                num_shards=4,
                batch_size=8,
                pilot_limit=200,
                approve_pilot=True,
                approve_full_infer=False,
                max_total_spend_usd=5.0,
            )
        # Disqualified & locked-reuse models blocked
        with self.assertRaises(RuntimeError):
            validate_exp2_cli_and_guard(
                stage="plan",
                model_id="frcnn_cad",
                split="nlbs_4view",
                num_shards=4,
                batch_size=8,
                pilot_limit=200,
                approve_pilot=False,
                approve_full_infer=False,
                max_total_spend_usd=5.0,
            )
        with self.assertRaises(RuntimeError):
            validate_exp2_cli_and_guard(
                stage="plan",
                model_id="gmic_pretrained",
                split="nlbs_4view",
                num_shards=4,
                batch_size=8,
                pilot_limit=200,
                approve_pilot=False,
                approve_full_infer=False,
                max_total_spend_usd=5.0,
            )


if __name__ == "__main__":
    unittest.main()
