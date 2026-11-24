"""CPU Unit Tests for NYU Four-View Step 2: Batched 10-TTA Parity, Provenance/FP32 Locks, Cost Guards & RSNA Calibration."""

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

from scripts.calibrate_paper1_nyu_rsna import verify_and_calibrate_nyu_rsna_only
from src.evaluation.stage_c_eval import compute_sha256_file
from src.paper1.nyu_fourview_adapter import (
    NYU_CKPT_SHA256,
    NYU_INPUT_SIZES,
    NYU_STEP1_MANIFEST_SHA256,
    NYU_VIEWS,
    build_nyu_fourview_model,
    check_nyu_aggregate_spend_guard,
    compute_nyu_case_ids_sha256,
    compute_text_lf_sha256,
    finalize_and_lock_nyu_predictions,
    prepare_nyu_exam_tta_tensors,
    score_nyu_fourview_exam,
    score_nyu_fourview_exam_batch,
    validate_nyu_shard_payload,
    validate_nyu_step2_cli_and_guard,
    verify_nyu_step1_against_frozen_source,
)


class TestNYUStep2Inference(unittest.TestCase):
    def test_1_batched_vs_single_exam_10tta_exact_parity(self):
        nyu_dir = ROOT / "external/breast_cancer_classifier"
        ckpt = nyu_dir / "models/sample_image_model.p"
        if not (ckpt.exists() and (nyu_dir / "sample_data/images/0_L_CC.png").exists()):
            self.skipTest("NYU checkpoint or sample_data not present")

        model, mods = build_nyu_fourview_model(ckpt, device=torch.device("cpu"))
        n_crop, n_load, n_coc = mods["crop_mammogram"], mods["loading"], mods["calc_optimal_centers"]

        exams_vc = []
        for idx in (0, 1):
            vc = {}
            for v in NYU_VIEWS:
                img = n_load.read_image_png(str(nyu_dir / f"sample_data/images/{idx}_{v.replace('-', '_')}.png"))
                cinfo = n_crop.crop_img_from_largest_connected(
                    img, n_crop.image_orientation("NO", v.split("-")[0]), True, 100, 50, 1.0 / 3.0
                )
                t, b, l, r = cinfo[0]
                cu = np.ascontiguousarray(n_load.flip_image(img[t:b, l:r], v, "NO"))
                tlbr = (
                    n_coc.get_bottomrightmost_pixel_constraint(cinfo[1][1], cinfo[2][0])
                    if "MLO" in v
                    else n_coc.get_rightmost_pixel_constraint(cinfo[1][1])
                )
                oc = n_coc.get_image_optimal_window_info(
                    cu, com=np.array(cu.shape) // 2, window_dim=np.array(NYU_INPUT_SIZES[v]), tl_br_constraint=tlbr
                )
                vc[v] = {"image_f32": cu.astype(np.float32), "best_center": (int(oc["best_center_y"]), int(oc["best_center_x"]))}
            exams_vc.append(vc)

        s0 = score_nyu_fourview_exam(model, mods, exams_vc[0], torch.device("cpu"), num_tta_epochs=10, seed=0)
        s1 = score_nyu_fourview_exam(model, mods, exams_vc[1], torch.device("cpu"), num_tta_epochs=10, seed=0)

        tta0 = prepare_nyu_exam_tta_tensors(exams_vc[0], mods, num_tta_epochs=10, seed=0)
        tta1 = prepare_nyu_exam_tta_tensors(exams_vc[1], mods, num_tta_epochs=10, seed=0)
        b_res = score_nyu_fourview_exam_batch(model, [tta0, tta1], torch.device("cpu"))

        for orig, batched in zip((s0, s1), b_res):
            for k in ("left_benign", "right_benign", "left_malignant", "right_malignant", "exam_score"):
                self.assertAlmostEqual(orig[k], batched[k], places=5)

    def test_2_shard_validation_fp32_and_provenance_lock(self):
        cases = ["c_10", "c_20"]
        recs = [
            {
                "case_id": c,
                "n_views": 4,
                "n_crop_fallbacks": 1 if c == "c_10" else 0,
                "has_crop_fallback": c == "c_10",
                "left_benign": 0.1,
                "right_benign": 0.2,
                "left_malignant": 0.05 if c == "c_10" else 0.30,
                "right_malignant": 0.40 if c == "c_10" else 0.12,
                "exam_score": 0.40 if c == "c_10" else 0.30,
            }
            for c in cases
        ]
        payload = {
            "schema_version": "paper1_nyu_shard_v2",
            "model_id": "nyu_fourview_pretrained",
            "checkpoint_sha256": NYU_CKPT_SHA256,
            "preprocess_version": "paper1_nyu_uint16_v1",
            "preprocess_manifest_sha256": "abc123",
            "precision": "fp32",
            "split_name": "rsna_val_calib",
            "shard_idx": 0,
            "num_shards": 1,
            "batch_exams": 4,
            "num_tta_epochs": 10,
            "tta_seed": 0,
            "expected_case_ids_sha256": compute_nyu_case_ids_sha256(cases),
            "fallback_logs": [{"case_id": "c_10", "item_id": "i1", "view_slot": "L-CC"}],
            "records": recs,
        }
        ok, reason = validate_nyu_shard_payload(
            payload,
            expected_case_ids=cases,
            batch_exams=4,
            split_name="rsna_val_calib",
            shard_idx=0,
            num_shards=1,
            preprocess_manifest_sha256="abc123",
        )
        self.assertTrue(ok, reason)

        for field, bad_val in (("num_tta_epochs", 5), ("precision", "fp16"), ("preprocess_manifest_sha256", "wrong")):
            bad = json.loads(json.dumps(payload))
            bad[field] = bad_val
            ok_bad, _ = validate_nyu_shard_payload(
                bad,
                expected_case_ids=cases,
                batch_exams=4,
                split_name="rsna_val_calib",
                shard_idx=0,
                num_shards=1,
                preprocess_manifest_sha256="abc123",
            )
            self.assertFalse(ok_bad, f"Expected rejection for {field}={bad_val}")

        step1_df = pd.DataFrame(
            [{"case_id": c, "item_id": f"{c}_{v}", "view_slot": v} for c in cases for v in NYU_VIEWS]
        )
        with tempfile.TemporaryDirectory() as td:
            lock = finalize_and_lock_nyu_predictions(
                split_name="rsna_val_calib",
                shard_outputs=[payload],
                expected_step1_df=step1_df,
                case_out_col="patient_id",
                batch_exams=4,
                expected_manifest_sha256="abc123",
                out_dir=Path(td),
            )
            self.assertEqual(lock["n_unique_cases"], 2)
            self.assertEqual(lock["n_unique_images"], 8)
            csv_p = Path(lock["artifacts"]["case_scores_csv"]["path"])
            self.assertEqual(hashlib.sha256(csv_p.read_bytes()).hexdigest(), lock["artifacts"]["case_scores_csv"]["sha256"])

    def test_3_cli_guard_step1_hash_enforcement_and_aggregate_budget(self):
        with self.assertRaises(RuntimeError):
            validate_nyu_step2_cli_and_guard("infer", "rsna_val_calib", 4, 4, 10, approve_full_infer=False)
        with self.assertRaises(ValueError):
            validate_nyu_step2_cli_and_guard("infer", "rsna_val_calib", 4, 4, 10, approve_full_infer=True, max_total_spend_usd=0.0)

        rsna_man_p = ROOT / "reports/paper1/nyu_step1_rsna_val_calib_manifest.csv"
        src_p = ROOT / "datasets/paper1/rsna_val_calib_4view_selection.csv"
        rsna_df = pd.read_csv(rsna_man_p, dtype={"item_id": str, "case_id": str})
        src_df = pd.read_csv(src_p, dtype=str)
        verify_nyu_step1_against_frozen_source(
            "rsna_val_calib",
            rsna_df,
            src_df,
            actual_manifest_sha256=compute_text_lf_sha256(rsna_man_p),
            expected_manifest_sha256=NYU_STEP1_MANIFEST_SHA256["rsna_val_calib"],
        )
        with self.assertRaises(RuntimeError):
            verify_nyu_step1_against_frozen_source(
                "rsna_val_calib",
                rsna_df,
                src_df,
                actual_manifest_sha256="deadbeef" * 8,
            )

        spent = check_nyu_aggregate_spend_guard(
            [{"estimated_cost_usd": 0.60, "resumed_from_cache": False}, {"estimated_cost_usd": 0.99, "resumed_from_cache": True}],
            max_total_spend_usd=1.50,
            next_wave_projected_usd=0.50,
        )
        self.assertEqual(spent, 0.60)
        with self.assertRaises(RuntimeError):
            check_nyu_aggregate_spend_guard(
                [{"estimated_cost_usd": 0.80, "resumed_from_cache": False}, {"estimated_cost_usd": 0.80, "resumed_from_cache": False}],
                max_total_spend_usd=1.50,
            )

    def test_4_rsna_only_calibration_and_lock(self):
        rsna_man_p = ROOT / "reports/paper1/nyu_step1_rsna_val_calib_manifest.csv"
        s1_df = pd.read_csv(rsna_man_p, dtype={"item_id": str, "case_id": str})
        cases = sorted(s1_df["case_id"].unique().tolist())

        # Deterministic synthetic locked RSNA predictions correlated with RSNA cancer labels
        splits_lbl = pd.read_csv(ROOT / "datasets/rsna/splits_seed42.csv", usecols=["patient_id", "split", "cancer"])
        vc = splits_lbl[splits_lbl["split"] == "val_calib"].copy()
        vc["patient_id"] = vc["patient_id"].astype(str)
        lbl_map = vc.groupby("patient_id")["cancer"].max().to_dict()
        fb_cases = set(s1_df.loc[s1_df["crop_fallback"].astype(bool), "case_id"].tolist())

        rng = np.random.RandomState(7)
        recs = []
        fbs = []
        for idx, cid in enumerate(cases):
            pos = int(lbl_map[cid])
            lm = float(np.clip(0.75 if pos else 0.05 + rng.uniform(0, 0.15), 0.001, 0.99))
            rm = float(np.clip(0.20 + rng.uniform(0, 0.10), 0.001, 0.99))
            recs.append(
                {
                    "case_id": cid,
                    "n_views": 4,
                    "n_crop_fallbacks": 1 if cid in fb_cases else 0,
                    "has_crop_fallback": cid in fb_cases,
                    "left_benign": 0.10,
                    "right_benign": 0.12,
                    "left_malignant": lm,
                    "right_malignant": rm,
                    "exam_score": max(lm, rm),
                }
            )
            if cid in fb_cases:
                fbs.append({"case_id": cid, "item_id": f"fb_{idx}", "view_slot": "L-CC"})

        shard_payload = {
            "schema_version": "paper1_nyu_shard_v2",
            "model_id": "nyu_fourview_pretrained",
            "checkpoint_file": "sample_image_model.p",
            "checkpoint_sha256": NYU_CKPT_SHA256,
            "preprocess_version": "paper1_nyu_uint16_v1",
            "preprocess_manifest_sha256": NYU_STEP1_MANIFEST_SHA256["rsna_val_calib"],
            "precision": "fp32",
            "split_name": "rsna_val_calib",
            "shard_idx": 0,
            "num_shards": 1,
            "batch_exams": 4,
            "num_tta_epochs": 10,
            "tta_seed": 0,
            "expected_case_ids_sha256": compute_nyu_case_ids_sha256(cases),
            "fallback_logs": fbs,
            "records": recs,
        }
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            pred_dir = td_p / "predictions_locked"
            finalize_and_lock_nyu_predictions(
                split_name="rsna_val_calib",
                shard_outputs=[shard_payload],
                expected_step1_df=s1_df,
                case_out_col="patient_id",
                batch_exams=4,
                expected_manifest_sha256=NYU_STEP1_MANIFEST_SHA256["rsna_val_calib"],
                out_dir=pred_dir,
            )
            calib_lock = verify_and_calibrate_nyu_rsna_only(
                locked_pred_dir=pred_dir,
                rsna_step1_manifest_csv=rsna_man_p,
                rsna_splits_csv=ROOT / "datasets/rsna/splits_seed42.csv",
                rsna_fold_csv=ROOT / "datasets/paper1/rsna_val_calib_2fold_seed42.csv",
                out_dir=td_p / "calib_out",
                expected_batch_exams=4,
                n_boot=20,
            )
            self.assertEqual(calib_lock["rsna_step1_manifest_sha256"], NYU_STEP1_MANIFEST_SHA256["rsna_val_calib"])
            self.assertTrue(Path(calib_lock["calibration_json_path"]).exists())
            self.assertIn("target_98", calib_lock["frozen_thresholds"])
            self.assertIn("target_95", calib_lock["frozen_thresholds"])

    def test_5_crlf_vs_lf_csv_manifest_sha256_equivalence(self):
        for split_name, csv_rel in (
            ("rsna_val_calib", "reports/paper1/nyu_step1_rsna_val_calib_manifest.csv"),
            ("nlbs_4view", "reports/paper1/nyu_step1_nlbs_4view_manifest.csv"),
        ):
            src_bytes = (ROOT / csv_rel).read_bytes()
            lf_bytes = src_bytes.replace(b"\r\n", b"\n")
            crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
            with tempfile.TemporaryDirectory() as td:
                lf_p = Path(td) / "lf.csv"
                crlf_p = Path(td) / "crlf.csv"
                lf_p.write_bytes(lf_bytes)
                crlf_p.write_bytes(crlf_bytes)
                self.assertEqual(compute_text_lf_sha256(lf_p), NYU_STEP1_MANIFEST_SHA256[split_name])
                self.assertEqual(compute_text_lf_sha256(crlf_p), NYU_STEP1_MANIFEST_SHA256[split_name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
