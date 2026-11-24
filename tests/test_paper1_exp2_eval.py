"""Offline CPU Unit Tests for Paper 1 Exp 2 Cryptographic Verification & Category C Evaluation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_paper1_exp2 import evaluate_exp2, verify_and_load_exp2_model_scores
from src.evaluation.stage_c_eval import compute_sha256_file
from src.paper1.exp2_adapters import EXP2_MODEL_REGISTRY


def _write_synthetic_exp2_fixture(root: Path) -> dict[str, Path]:
    reuse_dir = root / "reuse_locks"
    exp2_dir = root / "exp2_locks"
    reuse_dir.mkdir(parents=True, exist_ok=True)
    exp2_dir.mkdir(parents=True, exist_ok=True)

    studies = [f"s{i:02d}" for i in range(20)]
    exam_rows = []
    c4v_rows = []
    rng = np.random.default_rng(7)
    labels = [1 if i < 4 else 0 for i in range(20)]
    for i, sid in enumerate(studies):
        exam_rows.append(
            {
                "StudyInstanceUID": sid,
                "split_cluster_id": f"cl_{i % 15:02d}",
                "exam_cancer": labels[i],
            }
        )
        if i < 16:
            c4v_rows.append({"StudyInstanceUID": sid, "split_cluster_id": f"cl_{i % 15:02d}"})

    exam_csv = root / "exam_manifest.csv"
    c4v_csv = root / "common_4view.csv"
    pd.DataFrame(exam_rows).to_csv(exam_csv, index=False)
    pd.DataFrame(c4v_rows).to_csv(c4v_csv, index=False)

    def _scores_for(n: int, boost: float) -> pd.DataFrame:
        vals = np.clip(rng.uniform(0.01, 0.25, size=n) + np.asarray([boost if labels[i] else 0.0 for i in range(n)]), 0.0, 0.99)
        return pd.DataFrame({"StudyInstanceUID": studies[:n], "score": vals})

    # Reuse locks: gmic (full 20) & nyu (common 16)
    gmic_csv = reuse_dir / "gmic_nlbs_case_scores.csv"
    _scores_for(20, 0.5).to_csv(gmic_csv, index=False)
    (reuse_dir / "gmic_nlbs_provenance_lock.json").write_text(
        json.dumps(
            {
                "model_id": "gmic_pretrained",
                "checkpoint_sha256": EXP2_MODEL_REGISTRY["gmic_pretrained"].checkpoint_sha256,
                "label_quarantine_enforced": True,
                "locked_files_sha256": {"gmic_nlbs_case_scores.csv": compute_sha256_file(gmic_csv)},
            }
        ),
        encoding="utf-8",
    )

    nyu_csv = reuse_dir / "nyu_nlbs_4view_case_scores.csv"
    _scores_for(16, 0.4).to_csv(nyu_csv, index=False)
    (reuse_dir / "nyu_nlbs_4view_provenance_lock.json").write_text(
        json.dumps(
            {
                "model_id": "nyu_fourview_pretrained",
                "checkpoint_sha256": EXP2_MODEL_REGISTRY["nyu_fourview_pretrained"].checkpoint_sha256,
                "label_quarantine_enforced": True,
                "artifacts": {"case_scores_csv": {"sha256": compute_sha256_file(nyu_csv)}},
            }
        ),
        encoding="utf-8",
    )

    # New Phase 2 locks: glam (full 20) & end2end (full 20)
    for mid, boost in (("glam_pretrained", 0.45), ("end2end_cbis_ddsm", 0.25)):
        cname = f"{mid}_nlbs_full_case_scores.csv"
        cpath = exp2_dir / cname
        _scores_for(20, boost).to_csv(cpath, index=False)
        (exp2_dir / f"{mid}_nlbs_full_provenance_lock.json").write_text(
            json.dumps(
                {
                    "model_id": mid,
                    "checkpoint_sha256": EXP2_MODEL_REGISTRY[mid].checkpoint_sha256,
                    "label_quarantine_enforced": True,
                    "threshold_policy": {
                        "threshold_category": "C",
                        "calibration_domain": "none_threshold_independent_auroc_auprc_only",
                    },
                    "locked_files_sha256": {cname: compute_sha256_file(cpath)},
                }
            ),
            encoding="utf-8",
        )

    return {
        "reuse_dir": reuse_dir,
        "exp2_dir": exp2_dir,
        "exam_csv": exam_csv,
        "c4v_csv": c4v_csv,
        "out_dir": root / "out",
    }


class TestPaper1Exp2Eval(unittest.TestCase):
    def test_1_end_to_end_category_c_eval_and_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            fx = _write_synthetic_exp2_fixture(Path(td))
            rep = evaluate_exp2(
                reuse_lock_dir=fx["reuse_dir"],
                exp2_lock_dir=fx["exp2_dir"],
                exam_manifest_csv=fx["exam_csv"],
                common_4view_csv=fx["c4v_csv"],
                out_dir=fx["out_dir"],
                n_boot=100,
                seed=42,
            )
            self.assertEqual(rep["protocol_policy"]["primary_threshold_category"], "C")
            self.assertFalse(rep["protocol_policy"]["rsna_calibration_applied"])
            self.assertFalse(rep["protocol_policy"]["arbitrary_05_cutoff_applied"])
            self.assertEqual(rep["nlbs_common_4view_primary"]["n_units"], 16)
            self.assertEqual(len(rep["nlbs_common_4view_primary"]["models"]), 4)
            self.assertEqual(rep["nlbs_full_cohort_secondary"]["n_units"], 20)
            self.assertEqual(
                list(rep["nlbs_full_cohort_secondary"]["models"].keys()),
                ["gmic_pretrained", "glam_pretrained", "end2end_cbis_ddsm"],
            )
            self.assertIn("petrini_twoview_efnb3", rep["excluded_models"])
            self.assertEqual(rep["excluded_models"]["petrini_twoview_efnb3"]["eligibility"], "blocked_upstream")
            self.assertTrue((fx["out_dir"] / "exp2_summary.json").exists())
            self.assertTrue((fx["out_dir"] / "exp2_summary.md").exists())

    def test_2_rejects_tamper_forbidden_columns_and_rsna_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            fx = _write_synthetic_exp2_fixture(Path(td))

            # Tamper CSV after lock
            glam_csv = fx["exp2_dir"] / "glam_pretrained_nlbs_full_case_scores.csv"
            glam_csv.write_text(glam_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_and_load_exp2_model_scores(
                    model_id="glam_pretrained",
                    reuse_lock_dir=fx["reuse_dir"],
                    exp2_lock_dir=fx["exp2_dir"],
                )

            # Reject RSNA calibration in threshold_policy
            e2e_lock = fx["exp2_dir"] / "end2end_cbis_ddsm_nlbs_full_provenance_lock.json"
            payload = json.loads(e2e_lock.read_text(encoding="utf-8"))
            payload["threshold_policy"]["calibration_domain"] = "RSNA_val_calib"
            e2e_lock.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_and_load_exp2_model_scores(
                    model_id="end2end_cbis_ddsm",
                    reuse_lock_dir=fx["reuse_dir"],
                    exp2_lock_dir=fx["exp2_dir"],
                )


if __name__ == "__main__":
    unittest.main()
