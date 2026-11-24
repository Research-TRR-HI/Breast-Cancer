"""CPU Unit Tests for NYU Four-View Step 1: 4-View Manifest Validation, Lossless Cache & 10-TTA Parity."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from preprocess_paper1_nyu_cpu import _load_split_rows
from src.paper1.gmic_adapter import save_uint16_png_atomic
from src.paper1.nyu_fourview_adapter import (
    NYU_INPUT_SIZES,
    NYU_VIEWS,
    load_nyu_cached_view,
    score_nyu_fourview_exam,
    select_representative_pilot_exams,
    validate_and_load_nyu_4view_manifest,
    validate_nyu_cli_stage_and_guard,
    verify_and_load_nyu_resumed_records,
)


class TestNYUStep1CPUPreprocessing(unittest.TestCase):
    def test_1_4view_manifest_validation_rejects_missing_and_duplicates(self):
        rsna_df = validate_and_load_nyu_4view_manifest(
            "rsna_val_calib", ROOT / "datasets/paper1/rsna_val_calib_4view_selection.csv"
        )
        nlbs_df = validate_and_load_nyu_4view_manifest(
            "nlbs_4view", ROOT / "datasets/paper1/nlbs_4view_common_cohort_label_free.csv"
        )
        self.assertEqual(len(rsna_df), 9528)
        self.assertEqual(len(nlbs_df), 24736)

        with tempfile.TemporaryDirectory() as td:
            bad_missing = Path(td) / "bad_missing.csv"
            rsna_df.iloc[1:8].to_csv(bad_missing, index=False)  # drops L-CC of first patient
            with self.assertRaises(ValueError):
                validate_and_load_nyu_4view_manifest("rsna_val_calib", bad_missing)

            bad_dup = Path(td) / "bad_dup.csv"
            pd.concat([rsna_df.iloc[:4], rsna_df.iloc[:1]], ignore_index=True).to_csv(bad_dup, index=False)
            with self.assertRaises(ValueError):
                validate_and_load_nyu_4view_manifest("rsna_val_calib", bad_dup)

    def test_2_pilot_selection_includes_patient71_and_siemens(self):
        rsna_df = validate_and_load_nyu_4view_manifest(
            "rsna_val_calib", ROOT / "datasets/paper1/rsna_val_calib_4view_selection.csv"
        )
        nlbs_df = validate_and_load_nyu_4view_manifest(
            "nlbs_4view", ROOT / "datasets/paper1/nlbs_4view_common_cohort_label_free.csv"
        )
        r_rows = [
            {
                "split_name": "rsna_val_calib",
                "case_id": str(r.patient_id),
                "item_id": str(r.image_id),
                "view_slot": str(r.view_slot),
                "n_candidates_in_slot": int(r.n_candidates_in_slot),
            }
            for r in rsna_df.itertuples(index=False)
        ]
        n_rows = [
            {
                "split_name": "nlbs_4view",
                "case_id": str(r.StudyInstanceUID),
                "item_id": str(r.image_uid),
                "view_slot": str(r.view_slot),
                "manufacturer": str(r.manufacturer),
                "n_candidates_in_slot": int(r.n_candidates_in_slot),
            }
            for r in nlbs_df.itertuples(index=False)
        ]
        sr, sn, meta = select_representative_pilot_exams(r_rows, n_rows, ["71", "1625"], n_exams_per_split=40)
        self.assertEqual(len(sr), 160)
        self.assertEqual(len(sn), 160)
        self.assertTrue(meta["includes_patient_71"])
        self.assertEqual(meta["n_nlbs_siemens_exams"], 3)

    def test_3_lossless_uint16_cache_10tta_exact_parity(self):
        nyu_dir = ROOT / "external/breast_cancer_classifier"
        if not (nyu_dir / "sample_data/images/0_L_CC.png").exists():
            self.skipTest("NYU sample_data images not present")
        import hashlib
        import torch
        from src.paper1.nyu_fourview_adapter import build_nyu_fourview_model

        ckpt = nyu_dir / "models/sample_image_model.p"
        model, mods = build_nyu_fourview_model(ckpt, device=torch.device("cpu"))
        n_crop = mods["crop_mammogram"]
        n_load = mods["loading"]
        n_coc = mods["calc_optimal_centers"]

        online_views = {}
        u16_views = {}
        for v in NYU_VIEWS:
            img = n_load.read_image_png(str(nyu_dir / f"sample_data/images/0_{v.replace('-', '_')}.png"))
            cinfo = n_crop.crop_img_from_largest_connected(img, n_crop.image_orientation("NO", v.split("-")[0]), True, 100, 50, 1.0 / 3.0)
            t, b, l, r = cinfo[0]
            cu = np.ascontiguousarray(n_load.flip_image(img[t:b, l:r], v, "NO"))
            tlbr = (
                n_coc.get_bottomrightmost_pixel_constraint(cinfo[1][1], cinfo[2][0])
                if "MLO" in v
                else n_coc.get_rightmost_pixel_constraint(cinfo[1][1])
            )
            oc = n_coc.get_image_optimal_window_info(cu, com=np.array(cu.shape) // 2, window_dim=np.array(NYU_INPUT_SIZES[v]), tl_br_constraint=tlbr)
            bc = (int(oc["best_center_y"]), int(oc["best_center_x"]))
            online_views[v] = {"image_f32": cu.astype(np.float32), "best_center": bc}
            u16_views[v] = (cu, bc)

        with tempfile.TemporaryDirectory() as td:
            cached_views = {}
            for v, (cu, bc) in u16_views.items():
                p = Path(td) / f"{v}.png"
                save_uint16_png_atomic(cu, p, compression=1)
                sha = hashlib.sha256(cu.tobytes()).hexdigest()
                cached_views[v] = load_nyu_cached_view(p, bc, expected_shape=cu.shape, expected_u16_sha256=sha)
                self.assertEqual(float(np.max(np.abs(cached_views[v]["image_f32"] - online_views[v]["image_f32"]))), 0.0)

            res_on = score_nyu_fourview_exam(model, mods, online_views, torch.device("cpu"), num_tta_epochs=10, seed=0)
            res_ca = score_nyu_fourview_exam(model, mods, cached_views, torch.device("cpu"), num_tta_epochs=10, seed=0)
            self.assertEqual(res_on["left_malignant"], res_ca["left_malignant"])
            self.assertEqual(res_on["right_malignant"], res_ca["right_malignant"])
            self.assertEqual(res_on["exam_score"], res_ca["exam_score"])

    def test_4_sha256_resume_rejects_same_size_corruption_and_missing_hash(self):
        import hashlib
        import json
        from src.evaluation.stage_c_eval import compute_sha256_file

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            png_dir = td_p / "png"
            png_dir.mkdir()
            prog = td_p / "shard.jsonl"
            lines = []
            for idx, item_id in enumerate(("a", "b", "c")):
                arr = np.full((8, 8), idx + 1, dtype=np.uint16)
                p = png_dir / f"{item_id}.png"
                nb = save_uint16_png_atomic(arr, p, compression=1)
                lines.append(
                    {
                        "item_id": item_id,
                        "file_bytes": int(nb),
                        "png_sha256": compute_sha256_file(p) if item_id != "c" else "",
                        "u16_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                        "height": 8,
                        "width": 8,
                        "best_center_y": 4,
                        "best_center_x": 4,
                    }
                )
            prog.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

            # Corrupt 1 byte in `b.png` keeping EXACT file_bytes length
            bp = png_dir / "b.png"
            raw = bytearray(bp.read_bytes())
            raw[len(raw) // 2] ^= 0xFF
            bp.write_bytes(bytes(raw))

            done, stats = verify_and_load_nyu_resumed_records(prog, png_dir)
            self.assertEqual(list(done.keys()), ["a"])
            self.assertEqual(stats["valid_resumed"], 1)
            self.assertEqual(stats["rejected"], 2)

    def test_5_guarded_build_cli_blocks_without_approval(self):
        with self.assertRaises(RuntimeError):
            validate_nyu_cli_stage_and_guard("build", "all", 12, approve_full_build=False)
        with self.assertRaises(RuntimeError):
            validate_nyu_cli_stage_and_guard("build", "train_all", 12, approve_full_build=False)
        self.assertEqual(
            validate_nyu_cli_stage_and_guard("build", "all", 12, approve_full_build=True),
            ["rsna_val_calib", "nlbs_4view"],
        )
        self.assertEqual(
            validate_nyu_cli_stage_and_guard("build", "train_all", 12, approve_full_build=True),
            ["rsna_train", "rsna_train_calib"],
        )
        self.assertEqual(
            validate_nyu_cli_stage_and_guard("validate_config", "rsna_val_calib", 8, approve_full_build=False),
            ["rsna_val_calib"],
        )

    def test_6_rsna_train_and_train_calib_row_loading_label_free_and_4view(self):
        tr_rows = _load_split_rows("rsna_train", local_root=ROOT)
        tc_rows = _load_split_rows("rsna_train_calib", local_root=ROOT)
        self.assertEqual(len(tr_rows), 8339 * 4)
        self.assertEqual(len(tc_rows), 1192 * 4)
        self.assertNotIn("cancer", tr_rows[0])
        self.assertNotIn("cancer", tc_rows[0])
        self.assertEqual(len({r["case_id"] for r in tr_rows}), 8339)
        self.assertEqual(len({r["case_id"] for r in tc_rows}), 1192)
        vc_rows = _load_split_rows("rsna_val_calib", local_root=ROOT)
        self.assertEqual(len(vc_rows), 2382 * 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
