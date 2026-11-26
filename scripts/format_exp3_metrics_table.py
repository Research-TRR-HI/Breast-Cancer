"""Format complete Paper 1 Exp 3 evaluation metrics from exp3_summary.json into Markdown + CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def extract_rows(summary_json: Path) -> List[Dict[str, Any]]:
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for ds_id, ds_block in data.get("datasets", {}).items():
        for cohort_key in ("common_4view_cohort", "full_cohort_view_agnostic"):
            c_block = ds_block.get(cohort_key)
            if not c_block:
                continue
            n_total = int(c_block["n_exams"])
            n_pos = int(c_block["n_positive"])
            n_neg = n_total - n_pos
            for model_id, m_block in c_block.get("models", {}).items():
                ci = m_block.get("bootstrap_95ci", {})
                auroc_ci = ci.get("auroc", {})
                auprc_ci = ci.get("auprc", {})
                for op_key in ("target_98", "target_95"):
                    t = m_block[op_key]
                    tp = int(t["detected_cancers"])
                    fn = int(t["missed_cancers"])
                    ret = int(t["n_retained"])
                    dis = int(t["n_dismissed"])
                    fp = ret - tp
                    tn = dis - fn
                    sens = tp / n_pos if n_pos > 0 else 0.0
                    spec = tn / n_neg if n_neg > 0 else 0.0
                    fpr = fp / n_neg if n_neg > 0 else 0.0
                    fnr = fn / n_pos if n_pos > 0 else 0.0
                    dis_rate = dis / n_total if n_total > 0 else 0.0
                    for_rate = fn / dis if dis > 0 else 0.0
                    s_key = "s98" if op_key == "target_98" else "s95"
                    d_key = "d98" if op_key == "target_98" else "d95"
                    s_ci = ci.get(s_key, {})
                    d_ci = ci.get(d_key, {})
                    rows.append({
                        "dataset_id": ds_id,
                        "cohort": cohort_key,
                        "operating_point": op_key,
                        "model_id": model_id,
                        "n_exams": n_total,
                        "n_clusters": int(c_block["n_clusters"]),
                        "n_positive": n_pos,
                        "n_negative": n_neg,
                        "auroc": float(m_block["auroc"]),
                        "auroc_ci95_low": float(auroc_ci.get("ci95_low", 0.0)),
                        "auroc_ci95_high": float(auroc_ci.get("ci95_high", 0.0)),
                        "auprc": float(m_block["auprc"]),
                        "auprc_ci95_low": float(auprc_ci.get("ci95_low", 0.0)),
                        "auprc_ci95_high": float(auprc_ci.get("ci95_high", 0.0)),
                        "threshold": float(t["threshold"]),
                        "tp": tp,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "missed_cancers": fn,
                        "sensitivity": sens,
                        "sens_ci95_low": float(s_ci.get("ci95_low", 0.0)),
                        "sens_ci95_high": float(s_ci.get("ci95_high", 0.0)),
                        "specificity": spec,
                        "fpr": fpr,
                        "fnr": fnr,
                        "below_thr_rate": dis_rate,
                        "below_thr_ci95_low": float(d_ci.get("ci95_low", 0.0)),
                        "below_thr_ci95_high": float(d_ci.get("ci95_high", 0.0)),
                        "false_omission_rate": for_rate,
                        "non_cancer_retained": fp,
                        "screening_dismissal_allowed": bool(ds_block.get("population_screening_dismissal_allowed", False)),
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    rows = extract_rows(args.summary_json)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    for r in rows:
        print(
            f"{r['dataset_id']} | {r['cohort']} | {r['operating_point']} | {r['model_id']} | "
            f"AUROC={r['auroc']:.4f} [{r['auroc_ci95_low']:.4f},{r['auroc_ci95_high']:.4f}] | "
            f"AUPRC={r['auprc']:.4f} [{r['auprc_ci95_low']:.4f},{r['auprc_ci95_high']:.4f}] | "
            f"TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']} | "
            f"Sens={r['sensitivity']*100:.2f}% [{r['sens_ci95_low']*100:.2f}%,{r['sens_ci95_high']*100:.2f}%] | "
            f"Spec={r['specificity']*100:.2f}% FPR={r['fpr']*100:.2f}% FNR={r['fnr']*100:.2f}% | "
            f"BelowThr={r['below_thr_rate']*100:.2f}% FOR={r['false_omission_rate']*100:.2f}% | NonCancerRet={r['non_cancer_retained']}"
        )


if __name__ == "__main__":
    main()
