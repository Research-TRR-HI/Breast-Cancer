"""Generate reports/paper1/expanded/exp3/imdb_mias/imdb_results.md directly from imdb_results.json & CSV."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

RES_JSON = Path("reports/paper1/expanded/exp3/imdb_mias/imdb_results.json")
RES_CSV = Path("reports/paper1/expanded/exp3/imdb_mias/imdb_results_metrics.csv")
OUT_MD = Path("reports/paper1/expanded/exp3/imdb_mias/imdb_results.md")


def _fmt_ci(val, lo, hi, pct=False):
    if pd.isna(val):
        return "N/A"
    if pct:
        s = f"{val*100:.2f}%"
        if pd.notna(lo) and pd.notna(hi):
            s += f" ({lo*100:.2f}–{hi*100:.2f}%)"
        return s
    s = f"{val:.4f}"
    if pd.notna(lo) and pd.notna(hi):
        s += f" ({lo:.4f}–{hi:.4f})"
    return s


def _table_for(df: pd.DataFrame, analysis_key: str, op_key: str) -> str:
    sub = df[(df["analysis"] == analysis_key) & (df["operating_point"] == op_key)]
    lines = [
        "| Model | AUROC (95% CI) | AUPRC (95% CI) | TP | TN | FP | FN (Missed) | Sens (95% CI) | Spec | FPR | FNR | Below-Thr (95% CI) | FOR | Non-Cancer Retained |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---:|",
    ]
    for _, r in sub.iterrows():
        au = _fmt_ci(r["auroc"], r["auroc_ci95_low"], r["auroc_ci95_high"], pct=False)
        ap = _fmt_ci(r["auprc"], r["auprc_ci95_low"], r["auprc_ci95_high"], pct=False)
        se = _fmt_ci(r["sensitivity"], r["sens_ci95_low"], r["sens_ci95_high"], pct=True)
        bt = _fmt_ci(r["below_thr_rate"], r["below_thr_ci95_low"], r["below_thr_ci95_high"], pct=True)
        sp = f"{r['specificity']*100:.2f}%" if pd.notna(r["specificity"]) else "N/A"
        fpr = f"{r['fpr']*100:.2f}%" if pd.notna(r["fpr"]) else "N/A"
        fnr = f"{r['fnr']*100:.2f}%"
        for_r = f"{r['false_omission_rate']*100:.2f}%"
        lines.append(
            f"| `{r['model_id']}` | {au} | {ap} | {int(r['tp'])} | {int(r['tn'])} | {int(r['fp'])} | **{int(r['fn'])}** | {se} | {sp} | {fpr} | {fnr} | {bt} | {for_r} | {int(r['non_cancer_retained'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    data = json.loads(RES_JSON.read_text(encoding="utf-8"))
    df = pd.read_csv(RES_CSV)

    sections = [
        ("analysis_a_r1_path", "Analysis A (Strict Pathology-Confirmed) — Release 1 (`imdb_r1`, `N=61`, `60` pos / `1` benign)", "Note: Only `1` pathology-confirmed benign case in `r1`; specificity/AUROC is statistically uninformative."),
        ("analysis_a_r2_path", "Analysis A (Strict Pathology-Confirmed) — Release 2 (`imdb_r2`, `N=223`, `192` pos / `31` benign)", "Primary path-confirmed diagnostic discrimination benchmark (`31` benign gives wide specificity CIs)."),
        ("analysis_a_combined_path", "Analysis A (Strict Pathology-Confirmed) — Combined (`imdb`, `N=284`, `252` pos / `32` benign)", "Secondary combined view (cross-release patient independence unverified)."),
        ("analysis_b_r1_orig", "Analysis B (Original-Label Exploratory) — Release 1 (`imdb_r1`, `N=304`, `60` pos / `244` neg)", "243/244 negatives lack longitudinal follow-up; apparent specificity/below-threshold only."),
        ("analysis_b_r2_orig", "Analysis B (Original-Label Exploratory) — Release 2 (`imdb_r2`, `N=1,359`, `192` pos / `1,167` neg)", "Excludes 2 ambiguous cases; 1,136/1,167 negatives lack longitudinal follow-up."),
        ("analysis_b_combined_orig", "Analysis B (Original-Label Exploratory) — Combined (`imdb`, `N=1,663`, `252` pos / `1,411` neg)", "Secondary combined exploratory view (1,379/1,411 negatives lack follow-up; cross-release independence unverified)."),
    ]

    out_lines = [
        "# IMDB (India) — Six-Model External Evaluation Results (Common 4-View FFDM Cohort)",
        "",
        "**Cohort:** `datasets/exp3_meta/imdb_ffdm_4view_inference_manifest.csv` (`1,665` exams / `6,660` 12-bit Hologic FFDM images; `0` crop fallbacks across all 6 models)  ",
        "**Clinical Role:** `mixed_diagnostic_opportunistic` (`population_screening_dismissal_allowed = False`; `61.9%` of `r1` are prior-cancer follow-ups, `3.2%` screening)  ",
        "**Thresholds:** Frozen RSNA `val_calib` `98%` and `95%` empirical sensitivity thresholds (zero Indian labels used for training, selection, or calibration).  ",
        "",
        "> **Critical Scientific Caveat:** IMDB is NOT a population screening cohort, and `1,379 / 1,411` (`97.7%`) non-cancer exams lack longitudinal follow-up. Apparent specificity and below-threshold rates in Analysis B reflect frozen-threshold score-distribution shift, **NOT** validated screening dismissal utility.",
        "",
    ]

    for key, title, note in sections:
        out_lines.append(f"## {title}")
        out_lines.append(f"*{note}*")
        out_lines.append("")
        out_lines.append("### Primary Threshold (`RSNA 98%` Source Sensitivity)")
        out_lines.append(_table_for(df, key, "target_98"))
        out_lines.append("")
        out_lines.append("### Secondary Threshold (`RSNA 95%` Source Sensitivity)")
        out_lines.append(_table_for(df, key, "target_95"))
        out_lines.append("")

    OUT_MD.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {OUT_MD} ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()
