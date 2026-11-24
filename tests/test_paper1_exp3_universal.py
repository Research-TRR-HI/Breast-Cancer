"""CPU Unit Tests for Paper 1 Exp 3 Universal External Dataset Pipeline."""

from __future__ import annotations

import io
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
    build_exp3_dry_run_plan,
    enforce_dataset_model_eligibility,
    enforce_tigerwood697_profile,
    orchestrate_external_inference,
    parse_pilot_batch_sizes,
    resolve_exp3_model_selection,
    resolve_inference_batch_size,
    resolve_model_handoff,
    run_cli as modal_run_cli,
    select_optimal_pilot_batch_size,
    validate_exp3_cli_guards,
)
from src.evaluation.stage_c_eval import compute_sha256_file
from src.paper1.external_datasets import (
    INFERENCE_MANIFEST_COLUMNS,
    assert_lumina_neutral_inference_manifest,
    assert_no_forbidden_columns,
    build_cmmd_external_manifests,
    build_imdb_external_manifests,
    build_lumina_external_manifests,
    build_lumina_pseudonym_ingest_map,
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


def _make_synth_manifests(dataset_id: str, n_exams: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    inf_rows = []
    out_rows = []
    cohort_role = "diagnostic_biopsy_enriched" if dataset_id in ("cmmd", "lumina") else "screening"
    pop_ok = dataset_id == "nlbs"
    for idx in range(n_exams):
        exam_id = f"{dataset_id}_e{idx}"
        cluster_id = f"{dataset_id}_c{idx}"
        inf_rows.extend(_make_4view_rows(dataset_id, exam_id, cluster_id, start_idx=idx * 4))
        y = 1 if idx < max(1, n_exams // 2) else 0
        out_rows.append(
            {
                "dataset_id": dataset_id,
                "cluster_id": cluster_id,
                "exam_id": exam_id,
                "outcome_label": y,
                "label_source": "histopathology",
                "cohort_role": cohort_role,
                "population_screening_dismissal_allowed": pop_ok,
                "laterality_scope": "exam",
                "modality_subtype": "FFDM",
                "eligibility_flags": "{}",
            }
        )
    return pd.DataFrame(inf_rows), pd.DataFrame(out_rows)


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

        # LUMINA: colliding case_id='1' across Benign and Malign folders -> neutral pseudonyms only
        lum_cases = pd.DataFrame(
            [
                {"class_folder": "Benign", "case_id": "1", "image_id": "1_L_CC", "laterality": "L", "view": "CC", "image_path": "Benign/1/L_CC.dcm", "pathology": "BENIGN", "manufacturer": "Metaltronica"},
                {"class_folder": "Malign", "case_id": "1", "image_id": "1_R_MLO", "laterality": "R", "view": "MLO", "image_path": "Malign/1/R_MLO.dcm", "pathology": "MALIGN", "manufacturer": "SIEMENS"},
            ]
        )
        inf_l, out_l = build_lumina_external_manifests(lum_cases)
        self.assertEqual(sorted(out_l["exam_id"].tolist()), ["lumina_case_0001", "lumina_case_0002"])
        self.assertEqual(sorted(inf_l["image_id"].tolist()), ["lumina_img_00001", "lumina_img_00002"])
        assert_lumina_neutral_inference_manifest(inf_l, context="test_02_lumina")
        # Verify outcomes mapped accurately to neutral IDs via eligibility_flags
        label_by_source = {
            json.loads(r["eligibility_flags"])["source_class_folder"]: int(r["outcome_label"])
            for _, r in out_l.iterrows()
        }
        self.assertEqual(label_by_source, {"benign": 0, "malign": 1})

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

    def test_09_lumina_neutral_pseudonym_quarantine(self):
        rows = []
        for idx in range(1, 7):
            cls_dir = "Benign" if idx <= 3 else "Malign"
            path_lbl = "BENIGN" if idx <= 3 else "MALIGN"
            rows.append(
                {
                    "class_folder": cls_dir,
                    "case_id": str(idx),
                    "image_id": f"{idx}_L_CC",
                    "laterality": "L",
                    "view": "CC",
                    "image_path": f"{cls_dir}/{idx}/L_CC.dcm",
                    "pathology": path_lbl,
                    "manufacturer": "IMS GIOTTO S.p.A.",
                }
            )
        lum_df = pd.DataFrame(rows)
        inf_l, out_l = build_lumina_external_manifests(lum_df)
        ingest_map = build_lumina_pseudonym_ingest_map(lum_df)
        self.assertEqual(len(ingest_map), 6)

        # Assert ZERO 'benign' or 'malign' substrings appear anywhere in LUMINA inference manifest
        for col in inf_l.columns:
            joined = " ".join(inf_l[col].astype(str).tolist()).lower()
            self.assertNotIn("benign", joined, f"Leak in LUMINA inference column {col}")
            self.assertNotIn("malign", joined, f"Leak in LUMINA inference column {col}")

        # Verify deterministic salted permutation interleaved classes (not strictly 0,0,0,1,1,1)
        ordered_labels = out_l.sort_values("exam_id")["outcome_label"].tolist()
        self.assertEqual(sorted(ordered_labels), [0, 0, 0, 1, 1, 1])

        # Verify validator rejects ANY leaked benign/malign substring in path, id, or acquisition_meta
        for leak_col, leak_val in (
            ("image_path", "Benign/1/L_CC.dcm"),
            ("exam_id", "lumina_benign_1"),
            ("image_id", "lumina_malign_1_img"),
            ("acquisition_meta", json.dumps({"manufacturer": "GE", "note": "malign_case"})),
        ):
            bad = inf_l.copy()
            bad.loc[0, leak_col] = leak_val
            if leak_col == "exam_id":
                bad.loc[0, "cluster_id"] = leak_val
            with self.assertRaises(ValueError):
                validate_inference_manifest(bad, context=f"leak_{leak_col}")

    def test_10_pilot_batch_size_benchmark_and_cpu_scaling_guards(self):
        self.assertEqual(parse_pilot_batch_sizes(None, "ce_b5"), [16, 32, 64])
        self.assertEqual(parse_pilot_batch_sizes("64, 16, 32, 16", "resnet50_rsna"), [16, 32, 64])
        with self.assertRaises(ValueError):
            parse_pilot_batch_sizes("0,16", "resnet50_rsna")

        sel = select_optimal_pilot_batch_size(
            [
                {"batch_size": 16, "img_per_sec": 120.0, "peak_vram_gb": 1.2, "max_abs_diff": 0.0},
                {"batch_size": 64, "img_per_sec": 265.0, "peak_vram_gb": 3.4, "max_abs_diff": 2e-4},
                {"batch_size": 128, "img_per_sec": 310.0, "peak_vram_gb": 6.8, "max_abs_diff": 5e-2},
            ],
            max_parity_diff=1e-3,
        )
        self.assertEqual(sel["recommended_batch_size"], 64)

        for valid_cpu in (2, 4, 8, 16):
            validate_exp3_cli_guards(
                stage="pilot",
                dataset_id="cmmd",
                model_id="resnet50_rsna",
                approve_download=False,
                approve_preprocess=False,
                approve_full_infer=False,
                max_spend_usd=1.0,
                pilot_limit=200,
                cpu_workers=valid_cpu,
                pilot_batch_sizes="16,32,64",
            )
        for bad_cpu in (1, 3, 32):
            with self.assertRaises(ValueError):
                validate_exp3_cli_guards(
                    stage="preprocess",
                    dataset_id="cmmd",
                    model_id="resnet50_rsna",
                    approve_download=False,
                    approve_preprocess=True,
                    approve_full_infer=False,
                    max_spend_usd=1.0,
                    pilot_limit=200,
                    cpu_workers=bad_cpu,
                )

    def test_11_exp3_model_selection_and_verified_batch_sizes(self):
        self.assertEqual(
            resolve_exp3_model_selection("all_ready"),
            ["ce_b5", "trust_b5", "dinov2_rsna", "resnet50_rsna"],
        )
        self.assertEqual(
            resolve_exp3_model_selection("agent1_handoff"),
            ["gmic_rsna_finetuned", "nyu_fourview_rsna_finetuned"],
        )
        self.assertEqual(len(resolve_exp3_model_selection("all")), 6)
        self.assertEqual(
            resolve_exp3_model_selection("ce_b5, resnet50_rsna, ce_b5"),
            ["ce_b5", "resnet50_rsna"],
        )
        with self.assertRaises(ValueError):
            resolve_exp3_model_selection("unknown_model")

        self.assertEqual(resolve_inference_batch_size("cmmd", "ce_b5"), 32)
        self.assertEqual(resolve_inference_batch_size("cmmd", "dinov2_rsna"), 32)
        self.assertEqual(resolve_inference_batch_size("lumina", "dinov2_rsna"), 128)
        self.assertEqual(resolve_inference_batch_size("lumina", "resnet50_rsna"), 32)
        self.assertEqual(resolve_inference_batch_size("lumina", "dinov2_rsna", batch_size=64), 64)

    def test_12_lumina_nyu_hard_block_and_cmmd_4view_restriction(self):
        inf_l, _ = _make_synth_manifests("lumina", n_exams=6)
        # Single-view full mode succeeds
        elig_sv = enforce_dataset_model_eligibility("lumina", "ce_b5", inf_l, cohort_mode="full")
        self.assertEqual(len(elig_sv), len(inf_l))
        # NYU 4-view and common_4view mode are strictly hard-blocked on LUMINA
        with self.assertRaises(ValueError):
            enforce_dataset_model_eligibility("lumina", "nyu_fourview_rsna_finetuned", inf_l)
        with self.assertRaises(ValueError):
            enforce_dataset_model_eligibility("lumina", "ce_b5", inf_l, cohort_mode="common_4view")

        # Synthetic CMMD mix: 4 complete 4-view exams + 2 unilateral 2-view exams
        inf_c, out_c = _make_synth_manifests("cmmd", n_exams=6)
        drop_mask = (inf_c["exam_id"].isin(["cmmd_e4", "cmmd_e5"])) & (inf_c["laterality"] == "R")
        inf_c_mix = inf_c[~drop_mask].reset_index(drop=True)
        elig_nyu = enforce_dataset_model_eligibility("cmmd", "nyu_fourview_rsna_finetuned", inf_c_mix)
        self.assertEqual(elig_nyu["exam_id"].nunique(), 4)
        self.assertEqual(len(elig_nyu), 16)

        # evaluate_external_dataset also rejects common_4view on lumina
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, out_l = _make_synth_manifests("lumina", n_exams=6)
            out_l_p = root / "lumina_out.csv"
            out_l.to_csv(out_l_p, index=False)
            ex_scores = pd.DataFrame(
                {"exam_id": out_l["exam_id"], "cluster_id": out_l["cluster_id"], "score": 0.2}
            )
            lock_external_predictions(
                dataset_id="lumina",
                model_id="ce_b5",
                exam_scores_df=ex_scores,
                expected_exam_ids=out_l["exam_id"].tolist(),
                thresholds={"target_98": 0.096, "target_95": 0.124},
                model_sha256="c" * 64,
                inference_manifest_sha256="d" * 64,
                out_dir=root / "locked",
            )
            with self.assertRaises(ValueError):
                evaluate_external_dataset(
                    dataset_id="lumina",
                    lock_dir=root / "locked",
                    outcome_manifest_csv=out_l_p,
                    model_ids=["ce_b5"],
                    n_boot=10,
                    cohort_mode="common_4view",
                )

    def test_13_orchestrate_external_inference_4shard_resume_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inf_c, _ = _make_synth_manifests("cmmd", n_exams=8)
            manifest_p = root / "cmmd_inf.csv"
            inf_c.to_csv(manifest_p, index=False)
            lock_dir = root / "locked"
            expected_sha = resolve_model_handoff("ce_b5")["expected_sha256"]["checkpoint"]

            calls = []

            def fake_shard_runner(**kwargs):
                calls.append(kwargs["shard_idx"])
                df = pd.read_csv(io.StringIO(kwargs["manifest_csv_content"]))
                s_idx = kwargs["shard_idx"]
                n_s = kwargs["num_shards"]
                sub = df.iloc[s_idx::n_s]
                recs = [
                    {"item_id": str(iid), "score": 0.15 + 0.01 * i, "crop_fallback": False}
                    for i, iid in enumerate(sub["image_id"].astype(str))
                ]
                return {
                    "dataset_id": kwargs["dataset_id"],
                    "model_id": kwargs["model_id"],
                    "shard_idx": s_idx,
                    "num_shards": n_s,
                    "checkpoint_sha256": expected_sha,
                    "preprocess_version": str(df["preprocess_version"].iloc[0]),
                    "n_records": len(recs),
                    "elapsed_sec": 0.0 if s_idx in (0, 1) else 0.5,
                    "resumed_from_cache": bool(s_idx in (0, 1)),
                    "records": recs,
                }

            res = orchestrate_external_inference(
                dataset_id="cmmd",
                model_arg="ce_b5",
                manifest_csv=manifest_p,
                out_lock_dir=lock_dir,
                num_shards=4,
                shard_runner=fake_shard_runner,
                skip_profile_check=True,
            )
            self.assertEqual(calls, [0, 1, 2, 3])
            meta = res["models"]["ce_b5"]
            self.assertEqual(meta["resumed_shards"], 2)
            self.assertEqual(meta["batch_size"], 32)
            locked_exams = pd.read_csv(meta["exam_scores_csv"])
            locked_imgs = pd.read_csv(meta["image_scores_csv"])
            self.assertEqual(len(locked_exams), 8)
            self.assertEqual(len(locked_imgs), 32)
            self.assertEqual(compute_sha256_file(Path(meta["exam_scores_csv"])), meta["predictions_sha256"])

            # Guard 1: missing item_id in a shard fails immediately
            def broken_missing_runner(**kwargs):
                out = fake_shard_runner(**kwargs)
                if kwargs["shard_idx"] == 2:
                    out["records"] = out["records"][:-1]
                return out

            with self.assertRaises(ValueError):
                orchestrate_external_inference(
                    dataset_id="cmmd",
                    model_arg="ce_b5",
                    manifest_csv=manifest_p,
                    out_lock_dir=lock_dir,
                    num_shards=4,
                    shard_runner=broken_missing_runner,
                    skip_profile_check=True,
                )

            # Guard 2: checkpoint SHA-256 mismatch fails immediately
            def broken_sha_runner(**kwargs):
                out = fake_shard_runner(**kwargs)
                out["checkpoint_sha256"] = "0" * 64
                return out

            with self.assertRaises(ValueError):
                orchestrate_external_inference(
                    dataset_id="cmmd",
                    model_arg="ce_b5",
                    manifest_csv=manifest_p,
                    out_lock_dir=lock_dir,
                    num_shards=4,
                    shard_runner=broken_sha_runner,
                    skip_profile_check=True,
                )

            # Guard 3: duplicate item_id inside a shard fails immediately
            def broken_dup_runner(**kwargs):
                out = fake_shard_runner(**kwargs)
                if kwargs["shard_idx"] == 0 and out["records"]:
                    out["records"].append(dict(out["records"][0]))
                return out

            with self.assertRaises(ValueError):
                orchestrate_external_inference(
                    dataset_id="cmmd",
                    model_arg="ce_b5",
                    manifest_csv=manifest_p,
                    out_lock_dir=lock_dir,
                    num_shards=4,
                    shard_runner=broken_dup_runner,
                    skip_profile_check=True,
                )

    def test_14_agent1_handoff_and_nlbs_reuse_and_dry_run_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inf_c, out_c = _make_synth_manifests("cmmd", n_exams=6)
            drop_mask = (inf_c["exam_id"] == "cmmd_e5") & (inf_c["laterality"] == "R")
            inf_c = inf_c[~drop_mask].reset_index(drop=True)
            cmmd_inf_p = root / "cmmd_inf.csv"
            cmmd_out_p = root / "cmmd_out.csv"
            inf_c.to_csv(cmmd_inf_p, index=False)
            out_c.to_csv(cmmd_out_p, index=False)

            handoff_p = root / "agent1_handoff.json"
            handoff_payload = {
                "gmic_rsna_finetuned": {
                    "model_id": "gmic_rsna_finetuned",
                    "schema_version": "paper1_gmic_finetune_ckpt_v1",
                    "checkpoint_path": "/rsna_vol/models/paper1/checkpoints/gmic_rsna_finetuned.pt",
                    "checkpoint_sha256": "a" * 64,
                    "rsna_val_calib_thresholds": {"target_98": 0.111, "target_95": 0.222},
                },
                "nyu_fourview_rsna_finetuned": {
                    "model_id": "nyu_fourview_rsna_finetuned",
                    "schema_version": "paper1_nyu_finetune_ckpt_v1",
                    "checkpoint_path": "/rsna_vol/models/paper1/checkpoints/nyu_rsna_finetuned.pt",
                    "checkpoint_sha256": "b" * 64,
                    "rsna_val_calib_thresholds": {"target_98": 0.333, "target_95": 0.444},
                },
            }
            handoff_p.write_text(json.dumps(handoff_payload), encoding="utf-8")

            plan = build_exp3_dry_run_plan(
                dataset_id="cmmd",
                model_arg="agent1_handoff",
                manifest_csv=cmmd_inf_p,
                handoff_json=handoff_p,
                num_shards=4,
                dry_run_profile=True,
            )
            self.assertEqual(plan["models"]["gmic_rsna_finetuned"]["eligible_exams"], 6)
            self.assertEqual(plan["models"]["nyu_fourview_rsna_finetuned"]["eligible_exams"], 5)
            self.assertEqual(plan["models"]["nyu_fourview_rsna_finetuned"]["thresholds"]["target_98"], 0.333)

            def handoff_runner(**kwargs):
                df = pd.read_csv(io.StringIO(kwargs["manifest_csv_content"]))
                s_idx = kwargs["shard_idx"]
                n_s = kwargs["num_shards"]
                req_4v = bool(kwargs["model_spec"].get("requires_4view"))
                if req_4v:
                    ids = sorted(df["exam_id"].astype(str).unique().tolist())[s_idx::n_s]
                else:
                    ids = df["image_id"].astype(str).tolist()[s_idx::n_s]
                return {
                    "dataset_id": "cmmd",
                    "model_id": kwargs["model_id"],
                    "shard_idx": s_idx,
                    "num_shards": n_s,
                    "checkpoint_sha256": kwargs["model_spec"]["expected_sha256"]["checkpoint"],
                    "preprocess_version": str(df["preprocess_version"].iloc[0]),
                    "elapsed_sec": 0.25,
                    "resumed_from_cache": False,
                    "records": [{"item_id": x, "score": 0.4, "crop_fallback": False} for x in ids],
                }

            inf_out = orchestrate_external_inference(
                dataset_id="cmmd",
                model_arg="agent1_handoff",
                manifest_csv=cmmd_inf_p,
                handoff_json=handoff_p,
                out_lock_dir=root / "locked",
                num_shards=4,
                shard_runner=handoff_runner,
                skip_profile_check=True,
            )
            self.assertEqual(inf_out["models"]["gmic_rsna_finetuned"]["n_exams"], 6)
            self.assertEqual(inf_out["models"]["nyu_fourview_rsna_finetuned"]["n_exams"], 5)

            # Verify run_cli dry_run succeeds at $0 and infer blocks without approve_full_infer
            cli_fn = getattr(modal_run_cli, "raw_f", modal_run_cli)
            cli_fn(stage="dry_run", dataset="cmmd", model="all_ready", manifest_csv=str(cmmd_inf_p))
            with self.assertRaises(RuntimeError):
                cli_fn(
                    stage="infer",
                    dataset="cmmd",
                    model="all_ready",
                    manifest_csv=str(cmmd_inf_p),
                    approve_full_infer=False,
                )

            # Verify NLBS $0 reuse if completed assets exist + isolated multi-dataset eval
            six_lock = ROOT / "reports/paper1/six_model_final/locked_nlbs"
            nlbs_em = ROOT / "datasets/nlbs/exam_manifest.csv"
            if six_lock.exists() and nlbs_em.exists():
                nlbs_res = orchestrate_external_inference(
                    dataset_id="nlbs",
                    model_arg="all_ready",
                    out_lock_dir=root / "locked",
                    nlbs_reuse_root=six_lock,
                    nlbs_exam_manifest_csv=nlbs_em,
                    skip_profile_check=True,
                )
                self.assertEqual(nlbs_res["mode"], "zero_cost_reuse")
                self.assertEqual(sorted(nlbs_res["models"].keys()), ["ce_b5", "dinov2_rsna", "resnet50_rsna", "trust_b5"])

            # Evaluate CMMD separately -> report forbids pooling
            ev_c = evaluate_external_dataset(
                dataset_id="cmmd",
                lock_dir=root / "locked",
                outcome_manifest_csv=cmmd_out_p,
                model_ids=["gmic_rsna_finetuned", "nyu_fourview_rsna_finetuned"],
                common_4view_exam_ids=[f"cmmd_e{i}" for i in range(5)],
                n_boot=20,
            )
            self.assertFalse(ev_c["population_screening_dismissal_allowed"])
            self.assertEqual(ev_c["common_4view_cohort"]["n_exams"], 5)
            rep = generate_multi_dataset_report({"cmmd": ev_c}, out_dir=root / "eval_out")
            self.assertFalse(rep["pooled_analysis_performed"])


if __name__ == "__main__":
    unittest.main()
