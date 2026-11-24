"""CPU Unit Tests for NYU Four-View Step 2: Batched 10-TTA Parity, Shard Validation, Locking & CLI Guard."""

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

from src.paper1.nyu_fourview_adapter import (
    NYU_CKPT_SHA256,
    NYU_INPUT_SIZES,
    NYU_VIEWS,
    build_nyu_fourview_model,
    compute_nyu_case_ids_sha256,
    finalize_and_lock_nyu_predictions,
    prepare_nyu_exam_tta_tensors,
    score_nyu_fourview_exam,
    score_nyu_fourview_exam_batch,
    validate_nyu_shard_payload,
    validate_nyu_step2_cli_and_guard,
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

        # Single-exam 10-TTA predictions (original path)
        s0 = score_nyu_fourview_exam(model, mods, exams_vc[0], torch.device("cpu"), num_tta_epochs=10, seed=0)
        s1 = score_nyu_fourview_exam(model, mods, exams_vc[1], torch.device("cpu"), num_tta_epochs=10, seed=0)

        # Batched 2-exam 10-TTA prediction via precomputed per-exam RandomState(0) stacks
        tta0 = prepare_nyu_exam_tta_tensors(exams_vc[0], mods, num_tta_epochs=10, seed=0)
        tta1 = prepare_nyu_exam_tta_tensors(exams_vc[1], mods, num_tta_epochs=10, seed=0)
        b_res = score_nyu_fourview_exam_batch(model, [tta0, tta1], torch.device("cpu"))

        for orig, batched in zip((s0, s1), b_res):
            for k in ("left_benign", "right_benign", "left_malignant", "right_malignant", "exam_score"):
                self.assertAlmostEqual(orig[k], batched[k], places=5)

    def test_2_shard_validation_and_provenance_lock(self):
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

        bad = json.loads(json.dumps(payload))
        bad["num_tta_epochs"] = 5
        ok_bad, _ = validate_nyu_shard_payload(
            bad, expected_case_ids=cases, batch_exams=4, split_name="rsna_val_calib", shard_idx=0, num_shards=1
        )
        self.assertFalse(ok_bad)

        bad_max = json.loads(json.dumps(payload))
        bad_max["records"][0]["exam_score"] = 0.99
        ok_max, _ = validate_nyu_shard_payload(
            bad_max, expected_case_ids=cases, batch_exams=4, split_name="rsna_val_calib", shard_idx=0, num_shards=1
        )
        self.assertFalse(ok_max)

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
                out_dir=Path(td),
            )
            self.assertEqual(lock["n_unique_cases"], 2)
            self.assertEqual(lock["n_unique_images"], 8)
            csv_p = Path(lock["artifacts"]["case_scores_csv"]["path"])
            self.assertEqual(hashlib.sha256(csv_p.read_bytes()).hexdigest(), lock["artifacts"]["case_scores_csv"]["sha256"])

    def test_3_cli_guard_blocks_unapproved_infer_and_wrong_tta(self):
        with self.assertRaises(RuntimeError):
            validate_nyu_step2_cli_and_guard("infer", "rsna_val_calib", 4, 4, 10, approve_full_infer=False)
        with self.assertRaises(ValueError):
            validate_nyu_step2_cli_and_guard("pilot", "rsna_val_calib", 4, 4, 5, approve_full_infer=False)
        validate_nyu_step2_cli_and_guard("pilot", "rsna_val_calib", 4, 4, 10, approve_full_infer=False)
        validate_nyu_step2_cli_and_guard("infer", "nlbs_4view", 4, 4, 10, approve_full_infer=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
