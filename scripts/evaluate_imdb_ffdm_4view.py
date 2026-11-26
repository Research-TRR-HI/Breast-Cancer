"""CPU-only Evaluation of All 6 Locked RSNA-Supervised Models on IMDB 4-View FFDM Cohort.

Verifies all 6 SHA-256 pre-join prediction locks BEFORE joining `imdb_ffdm_4view_outcome_manifest.csv`,
then runs:
  - Analysis A (Strict Pathology-Confirmed): imdb_r1 (N=61), imdb_r2 (N=223), combined (N=284)
  - Analysis B (Original-Label Exploratory, 2 ambiguous excluded): imdb_r1 (N=304), imdb_r2 (N=1359), combined (N=1663)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.stage_c_eval import compute_auprc, compute_auroc
from src.paper1.universal_eval import (
    RSNA_SUPERVISED_MODELS,
    _bootstrap_cluster_metrics,
    verify_and_load_locked_external_predictions,
)
from src.training.calibration import evaluate_dismissal_at_threshold

LOCK_ROOT = Path("reports/paper1/expanded/exp3/locked_preds/imdb_ffdm_4view")
OUTCOME_CSV = Path("datasets/exp3_meta/imdb_ffdm_4view_outcome_manifest.csv")
OUT_DIR = Path("reports/paper1/expanded/exp3/imdb_mias")


def _eval_subset(
    merged: pd.DataFrame,
    thresholds: Dict[str, float],
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    y = merged["outcome_label"].to_numpy(dtype=np.int64)
    s = merged["score"].to_numpy(dtype=np.float64)
    cids = merged["cluster_id"].astype(str).to_numpy()
    n_pos = int(y.sum())
    n_total = int(len(y))
    n_neg = n_total - n_pos

    has_both_classes = (n_pos > 0) and (n_neg > 0)
    auroc = float(compute_auroc(y, s)) if has_both_classes else None
    auprc = float(compute_auprc(y, s)) if has_both_classes else None
    boot = (
        _bootstrap_cluster_metrics(
            y, s, cids,
            tau98=float(thresholds["target_98"]),
            tau95=float(thresholds["target_95"]),
            n_boot=n_boot,
            seed=seed,
        )
        if has_both_classes and n_neg >= 2
        else {}
    )

    ops = {}
    for op_key in ("target_98", "target_95"):
        tau = float(thresholds[op_key])
        res = evaluate_dismissal_at_threshold(s, y, tau)
        tp = int(res["detected_cancers"])
        fn = int(res["missed_cancers"])
        ret = int(res["n_retained"])
        dis = int(res["n_dismissed"])
        fp = ret - tp
        tn = dis - fn
        ops[op_key] = {
            "threshold": tau,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "missed_cancers": fn,
            "sensitivity": float(tp / n_pos) if n_pos > 0 else 0.0,
            "specificity": float(tn / n_neg) if n_neg > 0 else None,
            "fpr": float(fp / n_neg) if n_neg > 0 else None,
            "fnr": float(fn / n_pos) if n_pos > 0 else 0.0,
            "below_thr_rate": float(dis / n_total) if n_total > 0 else 0.0,
            "false_omission_rate": float(fn / dis) if dis > 0 else 0.0,
            "non_cancer_retained": fp,
        }

    return {
        "n_exams": n_total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "auroc": auroc,
        "auprc": auprc,
        "bootstrap_95ci": boot,
        "target_98": ops["target_98"],
        "target_95": ops["target_95"],
    }


def main() -> None:
    # 1. Verify all 6 SHA-256 locks BEFORE touching outcome labels
    locked_data: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]] = {}
    lock_metadata: Dict[str, Any] = {}
    for model_id in RSNA_SUPERVISED_MODELS:
        model_lock_dir = LOCK_ROOT / model_id
        df_pred, manifest = verify_and_load_locked_external_predictions(
            model_lock_dir, dataset_id="imdb", model_id=model_id
        )
        if int(len(df_pred)) != 1665:
            raise RuntimeError(f"Expected 1665 exams for {model_id}, got {len(df_pred)}")
        locked_data[model_id] = (df_pred, manifest)
        lock_metadata[model_id] = {
            "model_sha256": manifest["model_sha256"],
            "predictions_sha256": manifest["predictions_sha256"],
            "frozen_rsna_thresholds": manifest["frozen_rsna_thresholds"],
            "frozen_rsna_thresholds_sha256": manifest["frozen_rsna_thresholds_sha256"],
            "n_exams": manifest["n_exams"],
            "n_exams_with_crop_fallback": manifest["n_exams_with_crop_fallback"],
        }

    # 2. Load protected outcome manifest AFTER all 6 locks verified
    out_df = pd.read_csv(OUTCOME_CSV)
    out_df["release"] = out_df["exam_id"].astype(str).apply(lambda x: "imdb_r1" if x.startswith("imdb_r1_") else "imdb_r2")

    subsets: Dict[str, pd.DataFrame] = {
        "analysis_a_r1_path": out_df[(out_df["release"] == "imdb_r1") & (out_df["label_source"].isin(["pathology_confirmed_malignant", "pathology_confirmed_benign"]))],
        "analysis_a_r2_path": out_df[(out_df["release"] == "imdb_r2") & (out_df["label_source"].isin(["pathology_confirmed_malignant", "pathology_confirmed_benign"]))],
        "analysis_a_combined_path": out_df[out_df["label_source"].isin(["pathology_confirmed_malignant", "pathology_confirmed_benign"])],
        "analysis_b_r1_orig": out_df[(out_df["release"] == "imdb_r1") & (out_df["label_source"] != "unknown_or_ambiguous")],
        "analysis_b_r2_orig": out_df[(out_df["release"] == "imdb_r2") & (out_df["label_source"] != "unknown_or_ambiguous")],
        "analysis_b_combined_orig": out_df[out_df["label_source"] != "unknown_or_ambiguous"],
    }

    results: Dict[str, Any] = {
        "schema_version": "paper1_exp3_imdb_ffdm_4view_eval_v1",
        "cohort_note": "Common 4-view Hologic FFDM cohort (1665 exams: 304 r1 + 1361 r2). Never pooled with NLBS/CMMD/LUMINA/MIAS.",
        "locks": lock_metadata,
        "analyses": {},
    }
    csv_rows: List[Dict[str, Any]] = []

    for sub_name, sub_out in subsets.items():
        sub_res: Dict[str, Any] = {
            "n_exams": int(len(sub_out)),
            "n_positive": int(sub_out["outcome_label"].sum()),
            "n_negative": int(len(sub_out) - int(sub_out["outcome_label"].sum())),
            "models": {},
        }
        for model_id in RSNA_SUPERVISED_MODELS:
            df_pred, manifest = locked_data[model_id]
            merged = sub_out.merge(df_pred[["exam_id", "score"]], on="exam_id", how="inner")
            if len(merged) != len(sub_out):
                raise RuntimeError(f"Merge count mismatch for {sub_name}/{model_id}: {len(merged)} vs {len(sub_out)}")
            m_res = _eval_subset(merged, manifest["frozen_rsna_thresholds"], n_boot=2000, seed=42)
            sub_res["models"][model_id] = m_res
            ci = m_res.get("bootstrap_95ci", {})
            for op_key in ("target_98", "target_95"):
                op = m_res[op_key]
                s_key = "s98" if op_key == "target_98" else "s95"
                d_key = "d98" if op_key == "target_98" else "d95"
                csv_rows.append({
                    "analysis": sub_name,
                    "operating_point": op_key,
                    "model_id": model_id,
                    "n_exams": m_res["n_exams"],
                    "n_positive": m_res["n_positive"],
                    "n_negative": m_res["n_negative"],
                    "auroc": m_res["auroc"],
                    "auroc_ci95_low": ci.get("auroc", {}).get("ci95_low"),
                    "auroc_ci95_high": ci.get("auroc", {}).get("ci95_high"),
                    "auprc": m_res["auprc"],
                    "auprc_ci95_low": ci.get("auprc", {}).get("ci95_low"),
                    "auprc_ci95_high": ci.get("auprc", {}).get("ci95_high"),
                    "threshold": op["threshold"],
                    "tp": op["tp"],
                    "tn": op["tn"],
                    "fp": op["fp"],
                    "fn": op["fn"],
                    "sensitivity": op["sensitivity"],
                    "sens_ci95_low": ci.get(s_key, {}).get("ci95_low"),
                    "sens_ci95_high": ci.get(s_key, {}).get("ci95_high"),
                    "specificity": op["specificity"],
                    "fpr": op["fpr"],
                    "fnr": op["fnr"],
                    "below_thr_rate": op["below_thr_rate"],
                    "below_thr_ci95_low": ci.get(d_key, {}).get("ci95_low"),
                    "below_thr_ci95_high": ci.get(d_key, {}).get("ci95_high"),
                    "false_omission_rate": op["false_omission_rate"],
                    "non_cancer_retained": op["non_cancer_retained"],
                })
        results["analyses"][sub_name] = sub_res

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "imdb_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(OUT_DIR / "imdb_results_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    for sub_name, sub_res in results["analyses"].items():
        print(f"\n=== {sub_name} (N={sub_res['n_exams']}, pos={sub_res['n_positive']}, neg={sub_res['n_negative']}) ===")
        for m_id, mr in sub_res["models"].items():
            t98 = mr["target_98"]
            au = f"{mr['auroc']:.4f}" if mr["auroc"] is not None else "N/A"
            sp = f"{t98['specificity']*100:.2f}%" if t98["specificity"] is not None else "N/A"
            print(f"  {m_id:28s} | AUROC={au} | @98%: Sens={t98['sensitivity']*100:6.2f}% (missed={t98['fn']:3d}) | Spec={sp} | BelowThr={t98['below_thr_rate']*100:5.2f}%")


if __name__ == "__main__":
    main()
