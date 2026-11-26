"""CPU Unit Tests for Paper 1 Exp 3 IMDB (r1/r2, FFDM/SM, 5 confidence tiers) & MIAS (PGM, Truth, Gates)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_paper1_exp3_modal import (
    enforce_dataset_model_eligibility,
    resolve_inference_batch_size,
)
from src.data.preprocessing import (
    decode_mammogram_array,
    decode_pgm_array,
    extract_cropped_mammogram_with_metadata,
)
from src.evaluation.stage_c_eval import compute_sha256_file
from src.paper1.external_datasets import (
    DATASET_REGISTRY,
    IMDB_OUTCOME_CONFIDENCE_CATEGORIES,
    assert_no_forbidden_columns,
    build_imdb_external_manifests,
    build_mias_external_manifests,
    filter_imdb_manifests_by_modality,
    normalize_imdb_outcome_confidence,
    parse_mias_truth_file,
    select_common_4view_cohort,
)
from src.paper1.universal_eval import (
    MODEL_HANDOFF_SPECS,
    assert_no_cross_setting_pooling,
    evaluate_external_dataset,
    lock_external_predictions,
    validate_agent1_handoff_bundle,
    verify_and_load_locked_external_predictions,
)


def _write_synthetic_pgm(path: Path, width: int = 128, height: int = 128, side: str = "L") -> None:
    """Create a synthetic 8-bit binary P5 PGM mammogram with dark border and bright breast region."""
    img = np.zeros((height, width), dtype=np.uint8)
    if side == "L":
        img[20:108, 10:70] = 200
    else:
        img[20:108, 58:118] = 200
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write synthetic PGM at {path}")


class TestPaper1ImdbAndMias(unittest.TestCase):
    def test_01_imdb_r1_r2_separate_dataset_ids_and_ffdm_sm_separation(self):
        self.assertIn("imdb_r1", DATASET_REGISTRY)
        self.assertIn("imdb_r2", DATASET_REGISTRY)
        self.assertFalse(DATASET_REGISTRY["imdb_r1"]["population_screening_dismissal_allowed"])
        self.assertFalse(DATASET_REGISTRY["imdb_r2"]["population_screening_dismissal_allowed"])

        rows = []
        for rel, pt, lbl, src in (
            ("r1.0", "pt_001", 1, "histopathology"),
            ("r2.0", "pt_001", 0, "birads12_negative"),
        ):
            for lat, view in (("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")):
                rows.append(
                    {
                        "release": rel,
                        "patient_id": pt,
                        "exam_id": "ex_01",
                        "image_id": f"{lat}_{view}_ffdm",
                        "laterality": lat,
                        "view": view,
                        "image_path": f"{rel}/{pt}/{lat}_{view}.dcm",
                        "modality_subtype": "FFDM",
                        "outcome_label": lbl,
                        "label_source": src,
                    }
                )
            rows.append(
                {
                    "release": rel,
                    "patient_id": pt,
                    "exam_id": "ex_01",
                    "image_id": "L_CC_sm",
                    "laterality": "L",
                    "view": "CC",
                    "image_path": f"{rel}/{pt}/L_CC_sm.dcm",
                    "modality_subtype": "SM",
                    "outcome_label": lbl,
                    "label_source": src,
                }
            )
        raw_df = pd.DataFrame(rows)

        inf_r1, out_r1 = build_imdb_external_manifests(raw_df, dataset_id="imdb_r1")
        inf_r2, out_r2 = build_imdb_external_manifests(raw_df, dataset_id="imdb_r2")
        self.assertEqual(inf_r1["dataset_id"].unique().tolist(), ["imdb_r1"])
        self.assertEqual(inf_r2["dataset_id"].unique().tolist(), ["imdb_r2"])
        self.assertEqual(out_r1["exam_id"].tolist(), ["imdb_r1_ex_01"])
        self.assertEqual(out_r2["exam_id"].tolist(), ["imdb_r2_ex_01"])
        self.assertEqual(len(inf_r1), 5)
        self.assertEqual(len(inf_r2), 5)

        # FFDM vs SM separation
        inf_ffdm, out_ffdm = filter_imdb_manifests_by_modality(inf_r1, out_r1, "FFDM")
        inf_sm, out_sm = filter_imdb_manifests_by_modality(inf_r1, out_r1, "SM")
        self.assertEqual(len(inf_ffdm), 4)
        self.assertEqual(len(inf_sm), 1)
        self.assertEqual(len(out_ffdm), 1)
        self.assertEqual(len(out_sm), 1)

        # 4-view completeness picks FFDM over SM and preserves 4 views
        c4v, elig = select_common_4view_cohort(inf_r1)
        self.assertEqual(len(c4v), 4)
        self.assertTrue(bool(elig.iloc[0]["is_4view_complete"]))
        self.assertFalse(any("sm" in iid.lower() for iid in c4v["image_id"]))

    def test_02_imdb_five_outcome_confidence_categories(self):
        self.assertEqual(len(IMDB_OUTCOME_CONFIDENCE_CATEGORIES), 5)
        cases = [
            ("histopathology", 1, None, "histopathology_malignant"),
            ("biopsy_benign", 0, None, "histopathology_benign"),
            ("followup_imaging_or_phone", 0, None, "followup_negative"),
            ("birads12_negative", 0, None, "birads12_negative"),
            ("clinical_imaging_confirmed", 1, None, "clinical_or_imaging_malignant"),
        ]
        seen = []
        for src, lbl, raw_cat, expected in cases:
            got = normalize_imdb_outcome_confidence(src, lbl, raw_cat)
            self.assertEqual(got, expected)
            seen.append(got)
        self.assertEqual(tuple(seen), IMDB_OUTCOME_CONFIDENCE_CATEGORIES)

        with self.assertRaises(ValueError):
            normalize_imdb_outcome_confidence("histopathology", 0, "histopathology_malignant")
        with self.assertRaises(ValueError):
            normalize_imdb_outcome_confidence("unknown_source", 0, None)

    def test_03_mias_truth_parsing_and_label_quarantine(self):
        truth_text = """
        # mini-MIAS Info.txt sample
        mdb001 G CIRC B 535 425 197
        mdb002 G NORM
        mdb003 D NORM
        mdb004 D SPIC M 492 435 42
        mdb005 F CIRC B 477 133 30
        mdb005 F CALC M 500 168 26
        mdb006 F NORM
        """
        parsed = parse_mias_truth_file(truth_text)
        self.assertEqual(len(parsed), 6)
        by_id = parsed.set_index("image_id")
        self.assertEqual(int(by_id.loc["mdb001", "is_malignant"]), 0)
        self.assertEqual(int(by_id.loc["mdb002", "is_malignant"]), 0)
        self.assertEqual(int(by_id.loc["mdb004", "is_malignant"]), 1)
        # mdb005 has both B and M abnormalities -> aggregated to malignant (1)
        self.assertEqual(int(by_id.loc["mdb005", "is_malignant"]), 1)

        img_paths = {f"mdb{i:03d}": f"mdb{i:03d}.pgm" for i in range(1, 7)}
        inf_m, out_m = build_mias_external_manifests(parsed, img_paths)
        assert_no_forbidden_columns(inf_m.columns, context="mias_inf")
        for tok in ("malign", "benign", "spic", "calc"):
            self.assertFalse(inf_m["image_path"].str.lower().str.contains(tok).any())
            self.assertFalse(inf_m["acquisition_meta"].str.lower().str.contains(tok).any())

        out_by_pair = out_m.set_index("exam_id")
        self.assertEqual(int(out_by_pair.loc["mias_pair_001", "outcome_label"]), 0)
        self.assertEqual(int(out_by_pair.loc["mias_pair_002", "outcome_label"]), 1)
        self.assertEqual(int(out_by_pair.loc["mias_pair_003", "outcome_label"]), 1)

    def test_04_mias_pgm_decoding_and_stage1_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pgm_path = Path(tmp) / "mdb001.pgm"
            _write_synthetic_pgm(pgm_path, width=128, height=128, side="L")
            arr, phot, lat = decode_pgm_array(pgm_path)
            self.assertEqual(arr.shape, (128, 128))
            self.assertEqual(arr.dtype, np.float32)
            self.assertEqual(phot, "MONOCHROME2")
            self.assertEqual(lat, "")

            arr2, phot2, _ = decode_mammogram_array(pgm_path)
            self.assertTrue(np.array_equal(arr, arr2))
            self.assertEqual(phot2, "MONOCHROME2")

            res = extract_cropped_mammogram_with_metadata(arr2, photometric=phot2, laterality="L")
            self.assertEqual(res.cropped_uint16.dtype, np.uint16)
            self.assertFalse(res.used_fallback_uncropped)
            self.assertFalse(res.mirrored)

    def test_05_mias_model_eligibility_gates_and_4view_block(self):
        truth_text = "\n".join(
            [
                "mdb001 G NORM",
                "mdb002 G NORM",
                "mdb003 D SPIC M 492 435 42",
                "mdb004 D NORM",
            ]
        )
        img_paths = {f"mdb{i:03d}": f"mdb{i:03d}.pgm" for i in range(1, 5)}
        inf_m, _ = build_mias_external_manifests(truth_text, img_paths)

        # Single-view models are allowed on MIAS
        elig_ce = enforce_dataset_model_eligibility("mias", "ce_b5", inf_m, cohort_mode="full")
        self.assertEqual(len(elig_ce), 4)
        self.assertEqual(resolve_inference_batch_size("mias", "gmic_rsna_finetuned"), 16)

        # NYU 4-view and common_4view mode MUST be strictly blocked on MIAS (MLO-only)
        with self.assertRaises(ValueError):
            enforce_dataset_model_eligibility("mias", "nyu_fourview_rsna_finetuned", inf_m, cohort_mode="full")
        with self.assertRaises(ValueError):
            enforce_dataset_model_eligibility("mias", "ce_b5", inf_m, cohort_mode="common_4view")

    def test_06_checkpoint_threshold_lock_and_cluster_bootstrap_eval(self):
        handoff_path = ROOT / "reports/paper1/expanded/exp1/agent1_handoff.json"
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        for mid in ("gmic_rsna_finetuned", "nyu_fourview_rsna_finetuned"):
            res = validate_agent1_handoff_bundle(handoff[mid])
            self.assertTrue(res["valid"])

        # Synthetic MIAS + IMDB_R1 locked prediction + cluster bootstrap eval
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            lock_dir = tmp_dir / "locked_preds"
            truth_lines = []
            img_paths = {}
            for pair_i in range(1, 7):
                m_odd = f"mdb{2*pair_i - 1:03d}"
                m_even = f"mdb{2*pair_i:03d}"
                img_paths[m_odd] = f"{m_odd}.pgm"
                img_paths[m_even] = f"{m_even}.pgm"
                if pair_i <= 2:
                    truth_lines.append(f"{m_odd} D SPIC M 400 400 30")
                else:
                    truth_lines.append(f"{m_odd} G NORM")
                truth_lines.append(f"{m_even} G NORM")

            inf_m, out_m = build_mias_external_manifests("\n".join(truth_lines), img_paths)
            inf_csv = tmp_dir / "mias_inf.csv"
            out_csv = tmp_dir / "mias_out.csv"
            inf_m.to_csv(inf_csv, index=False)
            out_m.to_csv(out_csv, index=False)

            exam_scores = pd.DataFrame(
                {
                    "exam_id": out_m["exam_id"],
                    "cluster_id": out_m["cluster_id"],
                    "score": [0.85, 0.78, 0.05, 0.08, 0.11, 0.02],
                    "n_crop_fallbacks": [0] * 6,
                }
            )
            thr = MODEL_HANDOFF_SPECS["ce_b5"]["frozen_rsna_thresholds"]
            lock_external_predictions(
                dataset_id="mias",
                model_id="ce_b5",
                exam_scores_df=exam_scores,
                expected_exam_ids=out_m["exam_id"].tolist(),
                thresholds=thr,
                model_sha256=MODEL_HANDOFF_SPECS["ce_b5"]["expected_sha256"]["checkpoint"],
                inference_manifest_sha256=compute_sha256_file(inf_csv),
                out_dir=lock_dir,
            )
            loaded_df, _ = verify_and_load_locked_external_predictions(lock_dir, "mias", "ce_b5")
            self.assertEqual(len(loaded_df), 6)

            ev = evaluate_external_dataset(
                dataset_id="mias",
                lock_dir=lock_dir,
                outcome_manifest_csv=out_csv,
                model_ids=["ce_b5"],
                cohort_mode="full",
                n_boot=50,
                seed=42,
            )
            self.assertEqual(ev["dataset_id"], "mias")
            self.assertAlmostEqual(ev["full_cohort_view_agnostic"]["models"]["ce_b5"]["auroc"], 1.0)
            ci = ev["full_cohort_view_agnostic"]["models"]["ce_b5"]["bootstrap_95ci"]
            self.assertIn("auroc", ci)

            # Never pool screening (mias) with mixed (imdb_r1)
            with self.assertRaises(ValueError):
                assert_no_cross_setting_pooling(["mias", "imdb_r1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
