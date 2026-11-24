"""CPU Unit Tests for Paper 1 Exp 3 Universal External Dataset Pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_paper1_exp3_modal import (
    enforce_tigerwood697_profile,
    resolve_model_handoff,
    validate_exp3_cli_guards,
)
from src.paper1.external_datasets import (
    INFERENCE_MANIFEST_COLUMNS,
    assert_no_forbidden_columns,
    build_cmmd_external_manifests,
    build_imdb_external_manifests,
    build_lumina_external_manifests,
    build_nlbs_external_manifests,
    build_sensitivity_cohorts,
    select_common_4view_cohort,
    validate_inference_manifest,
    write_manifest_pair,
)
from src.paper1.universal_eval import (
    RSNA_SUPERVISED_MODELS,
    aggregate_image_records_to_exams,
    assert_no_cross_setting_pooling,
    evaluate_external_dataset,
    generate_multi_dataset_report,
    get_frozen_thresholds,
    lock_external_predictions,
    reuse_completed_nlbs_predictions,
    save_inference_shard_atomic,
    validate_agent1_handoff_bundle,
    validate_and_load_inference_shard,
    verify_and_load_locked_external_predictions,
)


def _make_4view_rows(dataset_id: str, exam_id: str, cluster_id: str, start_idx: int = 0) -> list[dict]:
    rows = []
    for k, (lat, view) in enumerate((("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO"))):
        rows.append(
            {
                "dataset_id": dataset_id,
                "cluster_id": cluster_id,
                "exam_id": exam_id,
                "image_id": f"{exam_id}_img_{start_idx + k}",
                "laterality": lat,
                "view": view,
                "image_path": f"{exam_id}/{lat}_{view}.dcm",
                "acquisition_meta": json.dumps({"manufacturer": "GE MEDICAL SYSTEMS", "modality_subtype": "FFDM"}),
                "preprocess_version": "trust_sec3_2_v2",
            }
        )
    return rows


class TestPaper1Exp3Universal(unittest.TestCase):
    def test_01_manifest_schema_and_forbidden_column_quarantine(self):
        rows = _make_4view_rows("cmmd", "cmmd_p1", "cmmd_p1")
        df = pd.DataFrame(rows)
        self.assertEqual(tuple(df.columns), INFERENCE_MANIFEST_COLUMNS)
        cleaned = validate_inference_manifest(df, context="unit_test")
        self.assertEqual(len(cleaned), 4)

        for bad_col in ("Cancer", "birads", "BIRADS", "pathology", "classification", "malignant"):
            bad_df = df.copy()
            bad_df[bad_col] = "1"
            with self.assertRaises(ValueError):
                validate_inference_manifest(bad_df, context=f"bad_{bad_col}")

        # Forbidden key hidden inside acquisition_meta JSON must also abort
        leak_df = df.copy()
        leak_df.loc[0, "acquisition_meta"] = json.dumps({"manufacturer": "GE", "birads": "5"})
        with self.assertRaises(ValueError):
            validate_inference_manifest(leak_df, context="meta_leak")

    def test_02_dataset_adapters_namespacing_and_dedup(self):
        # IMDB: colliding pt_001 across r1 and r2 + duplicate SOPInstanceUID + invalid outcome rejection
        imdb_raw = pd.DataFrame(
            [
                {"release": "r1", "patient_id": "pt_001", "exam_id": "ex1", "image_id": "i1", "laterality": "L", "view": "CC", "image_path": "r1/1.dcm", "sop_instance_uid": "uid_1", "outcome_label": 1, "label_source": "histopathology", "modality_subtype": "FFDM"},
                {"release": "r2", "patient_id": "pt_001", "exam_id": "ex1", "image_id": "i1", "laterality": "L", "view": "CC", "image_path": "r2/1.dcm", "sop_instance_uid": "uid_2", "outcome_label": 0, "label_source": "birads12_negative", "modality_subtype": "FFDM"},
                {"release": "r2", "patient_id": "pt_001", "exam_id": "ex1", "image_id": "i2", "laterality": "L", "view": "CC", "image_path": "r2/dup.dcm", "sop_instance_uid": "uid_2", "outcome_label": 0, "label_source": "birads12_negative", "modality_subtype": "FFDM"},
            ]
        )
        inf_i, out_i = build_imdb_external_manifests(imdb_raw)
        self.assertEqual(len(inf_i), 2)
        self.assertEqual(sorted(out_i["exam_id"].tolist()), ["imdb_r1_ex1", "imdb_r2_ex1"])
        self.assertEqual(sorted(out_i["cluster_id"].tolist()), ["imdb_r1_pt_001", "imdb_r2_pt_001"])

        bad_imdb = imdb_raw.iloc[:1].copy()
        bad_imdb["outcome_label"] = np.nan
        with self.assertRaises(ValueError):
            build_imdb_external_manifests(bad_imdb)

        # CMMD: bilateral patient with L=Malignant and R=Benign -> exam_label=1, breast labels preserved
        cmmd_clin = pd.DataFrame(
            [
                {"ID1": "D1-0021", "LeftRight": "L", "classification": "Malignant", "abnormality": "mass"},
                {"ID1": "D1-0021", "LeftRight": "R", "classification": "Benign", "abnormality": "calcification"},
            ]
        )
        cmmd_dcm = pd.DataFrame(
            [
                {"patient_id": "D1-0021", "image_id": "c1", "laterality": "L", "view": "CC", "image_path": "L_CC.dcm"},
                {"patient_id": "D1-0021", "image_id": "c2", "laterality": "L", "view": "MLO", "image_path": "L_MLO.dcm"},
                {"patient_id": "D1-0021", "image_id": "c3", "laterality": "R", "view": "CC", "image_path": "R_CC.dcm"},
                {"patient_id": "D1-0021", "image_id": "c4", "laterality": "R", "view": "MLO", "image_path": "R_MLO.dcm"},
            ]
        )
        inf_c, out_c = build_cmmd_external_manifests(cmmd_clin, cmmd_dcm)
        self.assertEqual(len(inf_c), 4)
        self.assertEqual(int(out_c.iloc[0]["outcome_label"]), 1)
        flags = json.loads(out_c.iloc[0]["eligibility_flags"])
        self.assertEqual(flags["breast_labels"], {"L": 1, "R": 0})
        assert_no_forbidden_columns(inf_c.columns, context="cmmd_inf")

        # LUMINA: colliding case_id='1' across Benign and Malign folders
        lum_cases = pd.DataFrame(
            [
                {"class_folder": "Benign", "case_id": "1", "image_id": "1_L_CC", "laterality": "L", "view": "CC", "image_path": "Benign/1/L_CC.dcm", "pathology": "BENIGN", "manufacturer": "Metaltronica"},
                {"class_folder": "Malign", "case_id": "1", "image_id": "1_R_MLO", "laterality": "R", "view": "MLO", "image_path": "Malign/1/R_MLO.dcm", "pathology": "MALIGN", "manufacturer": "SIEMENS"},
            ]
        )
        inf_l, out_l = build_lumina_external_manifests(lum_cases)
        self.assertEqual(sorted(out_l["exam_id"].tolist()), ["lumina_benign_1", "lumina_malign_1"])
        self.assertEqual(out_l.set_index("exam_id")["outcome_label"].to_dict(), {"lumina_benign_1": 0, "lumina_malign_1": 1})

    def test_03_common_4view_selection_never_fabricates_views(self):
        rows = _make_4view_rows("imdb", "ex_complete", "c1")
        # Add an SM duplicate for L-CC so FFDM must win
        rows.append(
            {
                "dataset_id": "imdb",
                "cluster_id": "c1",
                "exam_id": "ex_complete",
                "image_id": "ex_complete_sm",
                "laterality": "L",
                "view": "CC",
                "image_path": "ex_complete/0_SM_L_CC.dcm",
                "acquisition_meta": json.dumps({"manufacturer": "HOLOGIC", "modality_subtype": "SM"}),
                "preprocess_version": "trust_sec3_2_v2",
            }
        )
        # Incomplete 2-view exam MUST be excluded, never fabricated into 4 views
        rows.extend(_make_4view_rows("imdb", "ex_incomplete", "c2")[:2])
        inf_df = validate_inference_manifest(pd.DataFrame(rows))
        c4v, elig = select_common_4view_cohort(inf_df)
        self.assertEqual(sorted(c4v["exam_id"].unique().tolist()), ["ex_complete"])
        self.assertEqual(len(c4v), 4)
        self.assertNotIn("ex_complete_sm", c4v["image_id"].tolist())
        inc_row = elig[elig["exam_id"] == "ex_incomplete"].iloc[0]
        self.assertFalse(bool(inc_row["is_4view_complete"]))
        self.assertEqual(inc_row["missing_views"], "R-CC,R-MLO")

    def test_04_missing_duplicate_id_and_score_range_rejection(self):
        inf_df = pd.DataFrame(_make_4view_rows("cmmd", "e1", "c1") + _make_4view_rows("cmmd", "e2", "c2"))
        good_recs = pd.DataFrame([{"image_id": iid, "score": 0.25} for iid in inf_df["image_id"]])
        exam_df, breast_df = aggregate_image_records_to_exams(inf_df, good_recs)
        self.assertEqual(len(exam_df), 2)
        self.assertEqual(len(breast_df), 4)

        # Missing image_id
        with self.assertRaises(ValueError):
            aggregate_image_records_to_exams(inf_df, good_recs.iloc[:-1])

        # Duplicate image_id
        dup_recs = pd.concat([good_recs, good_recs.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            aggregate_image_records_to_exams(inf_df, dup_recs)

        # Non-finite score
        bad_recs = good_recs.copy()
        bad_recs.loc[0, "score"] = np.nan
        with self.assertRaises(ValueError):
            aggregate_image_records_to_exams(inf_df, bad_recs)

        # Out-of-range score
        oob_recs = good_recs.copy()
        oob_recs.loc[0, "score"] = 1.05
        with self.assertRaises(ValueError):
            aggregate_image_records_to_exams(inf_df, oob_recs)

    def test_05_frozen_thresholds_and_agent1_handoff_contract(self):
        for m in ("ce_b5", "trust_b5", "dinov2_rsna", "resnet50_rsna"):
            thr = get_frozen_thresholds(m)
            self.assertGreater(thr["target_95"], thr["target_98"])

        with self.assertRaises(RuntimeError):
            get_frozen_thresholds("gmic_rsna_finetuned")
        with self.assertRaises(RuntimeError):
            resolve_model_handoff("nyu_fourview_rsna_finetuned", handoff_json=None)

        with tempfile.TemporaryDirectory() as tmp:
            hpath = Path(tmp) / "handoff.json"
            h_payload = {
                "gmic_rsna_finetuned": {
                    "model_id": "gmic_rsna_finetuned",
                    "checkpoint_path": "models/paper1/checkpoints/gmic_rsna_finetuned_best_train_calib.pt",
                    "checkpoint_sha256": "a" * 64,
                    "schema_version": "paper1_gmic_finetune_ckpt_v1",
                    "rsna_val_calib_thresholds": {"target_98": 0.031, "target_95": 0.052},
                }
            }
            hpath.write_text(json.dumps(h_payload), encoding="utf-8")
            val = validate_agent1_handoff_bundle(h_payload["gmic_rsna_finetuned"])
            self.assertTrue(val["valid"])
            spec = resolve_model_handoff("gmic_rsna_finetuned", handoff_json=hpath)
            self.assertAlmostEqual(spec["frozen_rsna_thresholds"]["target_98"], 0.031)

    def test_06_resumable_shards_and_sha256_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            shard_p = tdir / "shard_00_of_01.json"
            recs = [{"item_id": "img1", "score": 0.12}, {"item_id": "img2", "score": 0.45}]
            save_inference_shard_atomic(
                shard_p,
                dataset_id="cmmd",
                model_id="resnet50_rsna",
                shard_idx=0,
                num_shards=1,
                checkpoint_sha256="b" * 64,
                preprocess_version="trust_sec3_2_v2",
                expected_item_ids=["img1", "img2"],
                records=recs,
            )
            ok, loaded, reason = validate_and_load_inference_shard(
                shard_p,
                dataset_id="cmmd",
                model_id="resnet50_rsna",
                shard_idx=0,
                num_shards=1,
                checkpoint_sha256="b" * 64,
                preprocess_version="trust_sec3_2_v2",
                expected_item_ids=["img1", "img2"],
            )
            self.assertTrue(ok, reason)
            self.assertEqual(len(loaded["records"]), 2)

            # Wrong expected_item_ids rejected cleanly
            ok2, _, _ = validate_and_load_inference_shard(
                shard_p,
                dataset_id="cmmd",
                model_id="resnet50_rsna",
                shard_idx=0,
                num_shards=1,
                checkpoint_sha256="b" * 64,
                preprocess_version="trust_sec3_2_v2",
                expected_item_ids=["img1", "img3"],
            )
            self.assertFalse(ok2)

            # Lock & tamper rejection
            exam_scores = pd.DataFrame(
                [
                    {"exam_id": "e1", "cluster_id": "c1", "score": 0.0005},
                    {"exam_id": "e2", "cluster_id": "c2", "score": 0.5},
                ]
            )
            lock_dir = tdir / "locks"
            man = lock_external_predictions(
                dataset_id="cmmd",
                model_id="resnet50_rsna",
                exam_scores_df=exam_scores,
                expected_exam_ids=["e1", "e2"],
                thresholds=get_frozen_thresholds("resnet50_rsna"),
                model_sha256="b" * 64,
                inference_manifest_sha256="c" * 64,
                out_dir=lock_dir,
            )
            df_v, _ = verify_and_load_locked_external_predictions(lock_dir, "cmmd", "resnet50_rsna")
            self.assertEqual(len(df_v), 2)

            # Tamper prediction CSV
            csv_p = lock_dir / "cmmd" / man["predictions_csv"]
            csv_p.write_text(csv_p.read_text(encoding="utf-8").replace("0.5", "0.9"), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_and_load_locked_external_predictions(lock_dir, "cmmd", "resnet50_rsna")

    def test_07_synthetic_6model_multi_dataset_eval_and_no_pooling(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            lock_dir = tdir / "locks"
            per_ds_results = {}

            for ds_id, role in (("nlbs", "screening"), ("imdb", "mixed_diagnostic_opportunistic"), ("cmmd", "diagnostic_biopsy_enriched"), ("lumina", "diagnostic_biopsy_enriched")):
                n_exams = 30
                inf_rows = []
                out_rows = []
                for i in range(n_exams):
                    eid = f"{ds_id}_ex_{i:03d}"
                    cid = f"{ds_id}_cl_{i % 24:03d}"
                    y = 1 if i < 6 else 0
                    n_views = 4 if (ds_id != "lumina" or i >= 6) else 2  # LUMINA malign unilateral
                    inf_rows.extend(_make_4view_rows(ds_id, eid, cid)[:n_views])
                    out_rows.append(
                        {
                            "dataset_id": ds_id,
                            "cluster_id": cid,
                            "exam_id": eid,
                            "outcome_label": y,
                            "label_source": "synthetic_truth",
                            "cohort_role": role,
                            "eligibility_flags": json.dumps({"pathology_confirmed": True}),
                        }
                    )
                inf_df = pd.DataFrame(inf_rows)
                out_df = pd.DataFrame(out_rows)
                pair_meta = write_manifest_pair(inf_df, out_df, tdir / "manifests", ds_id)
                sens = build_sensitivity_cohorts(inf_df, out_df)

                for m in RSNA_SUPERVISED_MODELS:
                    thr = {"target_98": 0.15, "target_95": 0.25} if "gmic" in m or "nyu" in m else get_frozen_thresholds(m)
                    is_4v = m == "nyu_fourview_rsna_finetuned"
                    target_eids = sens["exam_common_4view_ids"] if is_4v else sens["exam_full_ids"]
                    if ds_id == "lumina" and is_4v:
                        # Simulate balanced-malign check: lumina 4view has 0 malignant exams here
                        continue
                    ex_sub = out_df[out_df["exam_id"].isin(set(target_eids))].copy()
                    sc = np.where(ex_sub["outcome_label"].values == 1, rng.uniform(0.3, 0.9, len(ex_sub)), rng.uniform(0.01, 0.4, len(ex_sub)))
                    lock_external_predictions(
                        dataset_id=ds_id,
                        model_id=m,
                        exam_scores_df=pd.DataFrame({"exam_id": ex_sub["exam_id"], "cluster_id": ex_sub["cluster_id"], "score": sc}),
                        expected_exam_ids=target_eids,
                        thresholds=thr,
                        model_sha256="d" * 64,
                        inference_manifest_sha256=pair_meta["inference_manifest_sha256"],
                        out_dir=lock_dir,
                    )

                eval_models = [m for m in RSNA_SUPERVISED_MODELS if not (ds_id == "lumina" and m == "nyu_fourview_rsna_finetuned")]
                c4v_ids = sens["exam_common_4view_ids"] if ds_id != "lumina" else None
                per_ds_results[ds_id] = evaluate_external_dataset(
                    dataset_id=ds_id,
                    lock_dir=lock_dir,
                    outcome_manifest_csv=Path(pair_meta["outcome_manifest_csv"]),
                    model_ids=eval_models,
                    common_4view_exam_ids=c4v_ids,
                    n_boot=25,
                    seed=42,
                )

            summary = generate_multi_dataset_report(per_ds_results, tdir / "eval_out")
            self.assertFalse(summary["pooled_analysis_performed"])
            self.assertTrue(per_ds_results["nlbs"]["population_screening_dismissal_allowed"])
            self.assertFalse(per_ds_results["cmmd"]["population_screening_dismissal_allowed"])
            self.assertIn("dismissal_clinical_caveat", per_ds_results["cmmd"]["full_cohort_view_agnostic"]["models"]["ce_b5"]["target_98"])
            with self.assertRaises(ValueError):
                assert_no_cross_setting_pooling(["nlbs", "cmmd"])

    def test_08_nlbs_existing_predictions_reuse_and_cli_guards(self):
        six_lock_dir = ROOT / "reports/paper1/six_model_final/locked_nlbs"
        nlbs_inf = ROOT / "datasets/nlbs/inference_manifest_stage_c.csv"
        nlbs_em = ROOT / "datasets/nlbs/exam_manifest.csv"
        if six_lock_dir.exists() and nlbs_inf.exists() and nlbs_em.exists():
            with tempfile.TemporaryDirectory() as tmp:
                tdir = Path(tmp)
                reused = reuse_completed_nlbs_predictions(six_lock_dir, nlbs_em, tdir / "locks")
                self.assertEqual(set(reused.keys()), {"ce_b5", "trust_b5", "dinov2_rsna", "resnet50_rsna"})
                for m, man in reused.items():
                    self.assertEqual(man["n_exams"], 6359)
                inf_df, out_df = build_nlbs_external_manifests(nlbs_inf, nlbs_em)
                self.assertEqual(len(inf_df), 26988)
                self.assertEqual(len(out_df), 6359)
                self.assertEqual(int(out_df["outcome_label"].sum()), 151)

        # Modal CLI profile & approval guards
        old_prof = os.environ.get("MODAL_PROFILE")
        try:
            os.environ["MODAL_PROFILE"] = "hasinishrak2015"
            with self.assertRaises(RuntimeError):
                enforce_tigerwood697_profile(dry_run=False)
            os.environ["MODAL_PROFILE"] = "tigerwood697"
            self.assertEqual(enforce_tigerwood697_profile(dry_run=False), "tigerwood697")
        finally:
            if old_prof is None:
                os.environ.pop("MODAL_PROFILE", None)
            else:
                os.environ["MODAL_PROFILE"] = old_prof

        with self.assertRaises(RuntimeError):
            validate_exp3_cli_guards(
                stage="download",
                dataset_id="cmmd",
                model_id="resnet50_rsna",
                approve_download=False,
                approve_preprocess=False,
                approve_full_infer=False,
                max_spend_usd=5.0,
                pilot_limit=200,
            )
        with self.assertRaises(RuntimeError):
            validate_exp3_cli_guards(
                stage="infer",
                dataset_id="cmmd",
                model_id="resnet50_rsna",
                approve_download=False,
                approve_preprocess=False,
                approve_full_infer=False,
                max_spend_usd=5.0,
                pilot_limit=200,
            )


if __name__ == "__main__":
    unittest.main()
