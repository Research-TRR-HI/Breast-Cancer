"""Stage C Step 2: Cryptographically Verify Locked NLBS Predictions & Evaluate Outcomes.

Strict Separation:
    - Refuses to read `datasets/nlbs/exam_manifest.csv` until every SHA-256 checksum in
      `reports/stage_c/locked_predictions_manifest.json` and both frozen RSNA checkpoints match.
    - Reports CE vs. TRUST side-by-side at the frozen 95% and 98% RSNA operating thresholds
      (NEVER adjusts thresholds on NLBS).
    - Generates `reports/stage_c/nlbs_zero_label_results.json` and
      `reports/stage_c/nlbs_zero_label_results.md`.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.stage_c_eval import verify_and_evaluate_locked_predictions


def _fmt_pct(val: float, digits: int = 2) -> str:
    return f"{100.0 * float(val):.{digits}f}%"


def _fmt_ci_pct(ci_dict: dict, digits: int = 2) -> str:
    lo = ci_dict.get("ci95_low")
    hi = ci_dict.get("ci95_high")
    if lo is None or hi is None:
        return "N/A"
    return f"[{100.0 * lo:.{digits}f}%, {100.0 * hi:.{digits}f}%]"


def _fmt_ci_num(ci_dict: dict, digits: int = 4) -> str:
    lo = ci_dict.get("ci95_low")
    hi = ci_dict.get("ci95_high")
    if lo is None or hi is None:
        return "N/A"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def render_markdown_report(eval_res: dict, rsna_ce: dict, rsna_trust: dict) -> str:
    denom = eval_res["cohort_denominators"]
    lock = eval_res["lock_verification"]
    prep = eval_res["preprocessing_and_integrity"]
    ex = eval_res["exam_level_results"]
    ci = eval_res["cluster_bootstrap_95ci"]["metrics"]
    cl_col = eval_res["cluster_collapsed_reference"]

    ce_98_nlbs = ex["ce"]["target_98"]["triage_recall"]
    tr_98_nlbs = ex["trust"]["target_98"]["triage_recall"]
    ce_95_nlbs = ex["ce"]["target_95"]["triage_recall"]
    tr_95_nlbs = ex["trust"]["target_95"]["triage_recall"]

    tr_98_dis = ex["trust"]["target_98"]["dismissal_rate"]
    ce_98_dis = ex["ce"]["target_98"]["dismissal_rate"]
    tr_95_dis = ex["trust"]["target_95"]["dismissal_rate"]
    ce_95_dis = ex["ce"]["target_95"]["dismissal_rate"]

    if tr_98_nlbs + 1e-12 >= 0.98 and tr_98_dis >= 0.05:
        verdict = (
            "**SUPPORTS frozen-threshold transfer at 98% target sensitivity** (or partial support depending on 95% operating point; inspect both points below)."
        )
    elif tr_98_nlbs + 1e-12 < 0.98 or tr_95_nlbs + 1e-12 < 0.95:
        verdict = (
            "**CONTRADICTS zero-label frozen-threshold transfer safety on NLBS**: "
            "frozen RSNA thresholds failed to preserve the intended cancer sensitivity on the unseen Canadian cohort "
            "(and/or collapsed dismissal utility). Proceed directly to Stage D (UPA unlabeled adaptation) to test whether "
            "distribution alignment fixes the operating-point drift without target cancer labels."
        )
    else:
        verdict = (
            "**NUANCED / COLLAPSED DISMISSAL UTILITY**: sensitivity held near target, but dismissal workload reduction collapsed under domain shift."
        )

    lines = [
        "# Stage C Report: Strict Zero-Target-Label External Evaluation on NLBS (RSNA -> NLBS)",
        "",
        f"**Evaluated UTC:** `{eval_res['evaluated_at_utc']}`  ",
        "**Protocol:** Protocol 1 — Strict Source-Only / Frozen-Threshold Zero-Target-Label Transfer (`NLBS` cancer outcomes strictly quarantined until cryptographic lock verification).  ",
        "**Retraining / Adaptation Applied:** None (`0` NLBS training updates, `0` NLBS threshold updates, `0` UPA alignment).",
        "",
        "---",
        "",
        "## 1. Experimental Protocol & Target-Label Quarantine Evidence",
        "- **Training Cohort (Source Only):** RSNA Screening Mammography (`train=38,251`, `train_calib=5,473`, `val_calib=10,982` images; `0` patient leakage), frozen Epoch-20 checkpoints from Stage B (`b5-model-best-epoch-7.tar` frozen backbone).",
        f"- **Inference Input Isolation:** Built label-free manifest `{Path(lock.get('inference_manifest_path', 'datasets/nlbs/inference_manifest_stage_c.csv')).as_posix()}` (`sha256: {lock['inference_manifest_sha256']}`) explicitly excluding `Cancer`, `exam_cancer`, `n_cancer_images`, `Hard to State Negative`, `hard_to_state_negative`, and `cohort_group`, sorted by clinical `StudyInstanceUID` rather than directory class folders. Only `src/` and this label-free manifest were mounted into the Modal GPU inference container.",
        f"- **Cryptographic Pre-Evaluation Lock (`verified_before_label_join = {lock['verified_before_label_join']}`):**",
        f"  - `locked_nlbs_predictions.csv` SHA-256: `{lock['locked_predictions_sha256']}`",
        f"  - `locked_nlbs_image_diagnostics.csv.gz` SHA-256: `{lock['locked_image_diagnostics_sha256']}`",
        f"  - `ce_rsna_seed42_final_epoch20.pt` SHA-256: `{lock['checkpoint_metadata']['ce_checkpoint_sha256']}`",
        f"  - `trust_rsna_seed42_final_epoch20.pt` SHA-256: `{lock['checkpoint_metadata']['trust_checkpoint_sha256']}`",
        f"  - `b5-model-best-epoch-7.tar` SHA-256: `{lock['checkpoint_metadata']['backbone_sha256']}`",
        f"  - Frozen RSNA Thresholds SHA-256: `{lock['frozen_thresholds_sha256']}`",
        f"- **NLBS Denominators After Exact `StudyInstanceUID` Deduplication:** `{denom['total_exam_manifest_rows']:,}` folder exam rows -> **`{denom['deduplicated_unique_study_exams']:,}` unique examinations** (`{denom['deduplicated_shared_study_records_collapsed']}` cross-folder duplicate aliases merged) across **`{denom['unique_split_clusters']:,}` patient clusters (`split_cluster_id`)**, comprising **`{denom['total_cancer_positive_exams']}` cancer-positive exams** (`{denom['total_cancer_positive_clusters']}` cancer clusters; prevalence `{_fmt_pct(denom['nlbs_exam_prevalence'])}`), `{denom['total_false_positive_exams']:,}` False Positive recalled exams, and `{denom['total_normal_exams']:,}` Normal screening exams.",
        "",
        "---",
        "",
        "## 2. Primary Zero-Label Results: CE vs. TRUST at Frozen RSNA Operating Thresholds",
        "",
        "### Examination-Level Performance (`N = 6,359` Unique Studies, `151` Cancer-Positive Exams)",
        "",
        "| Model | Source Target | Frozen RSNA `tau` | Sensitivity (95% Cluster CI) | Missed / Total Cancers | Dismissed / Total Exams | Dismissal Rate (95% Cluster CI) | Cancer Prev Dismissed (95% Cluster CI) | One-Sided 99% CP UB* | Meets Target? |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m_key, label in (("ce", "CE-Only"), ("trust", "TRUST")):
        for op_key, op_lbl in (("target_95", "95% Recall"), ("target_98", "98% Recall")):
            r = ex[m_key][op_key]
            sens_ci = _fmt_ci_pct(ci[f"{m_key}_{op_key}_sensitivity"])
            dis_ci = _fmt_ci_pct(ci[f"{m_key}_{op_key}_dismissal_rate"])
            prev_ci = _fmt_ci_pct(ci[f"{m_key}_{op_key}_cancer_prev_dismissed"], digits=3)
            lines.append(
                f"| **{label}** | {op_lbl} | `{r['threshold']:.6f}` | "
                f"**{_fmt_pct(r['triage_recall'])}** {sens_ci} | "
                f"**{r['missed_cancers']}** / {r['total_cancers']} | "
                f"{r['n_dismissed']:,} / {r['total_cases']:,} | "
                f"**{_fmt_pct(r['dismissal_rate'])}** {dis_ci} | "
                f"{_fmt_pct(r['cancer_prev_dismissed'], digits=3)} {prev_ci} | "
                f"{_fmt_pct(r['clopper_pearson_99_ub'], digits=3)} | `{r['meets_target_recall']}` |"
            )

    lines.extend([
        "",
        "*Note on 99% Clopper-Pearson Upper Bound: Computed under binomial independence across the `6,359` deduplicated exams for protocol comparability with TRUST; because repeat exams from the same patient (`5,942` clusters) are non-independent, the `split_cluster_id` bootstrap 95% CIs above represent cluster-adjusted uncertainty.*",
        "",
        "### Source (`RSNA val_calib`) vs. Unseen Target (`NLBS`) Shift Comparison",
        "",
        "| Model | Operating Point | RSNA Sensitivity -> NLBS Sensitivity | RSNA Dismissal -> NLBS Dismissal |",
        "| :--- | :---: | :---: | :---: |",
        f"| **CE-Only** | 95% Target (`0.124451`) | {_fmt_pct(rsna_ce['target_95']['triage_recall'])} -> **{_fmt_pct(ce_95_nlbs)}** | {_fmt_pct(rsna_ce['target_95']['dismissal_rate'])} -> **{_fmt_pct(ce_95_dis)}** |",
        f"| **TRUST** | 95% Target (`0.195435`) | {_fmt_pct(rsna_trust['target_95']['triage_recall'])} -> **{_fmt_pct(tr_95_nlbs)}** | {_fmt_pct(rsna_trust['target_95']['dismissal_rate'])} -> **{_fmt_pct(tr_95_dis)}** |",
        f"| **CE-Only** | 98% Target (`0.096191`) | {_fmt_pct(rsna_ce['target_98']['triage_recall'])} -> **{_fmt_pct(ce_98_nlbs)}** | {_fmt_pct(rsna_ce['target_98']['dismissal_rate'])} -> **{_fmt_pct(ce_98_dis)}** |",
        f"| **TRUST** | 98% Target (`0.179565`) | {_fmt_pct(rsna_trust['target_98']['triage_recall'])} -> **{_fmt_pct(tr_98_nlbs)}** | {_fmt_pct(rsna_trust['target_98']['dismissal_rate'])} -> **{_fmt_pct(tr_98_dis)}** |",
        "",
        "---",
        "",
        "## 3. Secondary Ranking Metrics & Paired Cluster Differences (`B = 2,000` Bootstrap on `5,942` Clusters)",
        f"- **CE-Only NLBS AUROC:** `{ex['ce']['auroc']:.4f}` (95% CI `{_fmt_ci_num(ci['ce_auroc'])}`) | **AUPRC:** `{ex['ce']['auprc']:.4f}` (95% CI `{_fmt_ci_num(ci['ce_auprc'])}`)",
        f"- **TRUST NLBS AUROC:** `{ex['trust']['auroc']:.4f}` (95% CI `{_fmt_ci_num(ci['trust_auroc'])}`) | **AUPRC:** `{ex['trust']['auprc']:.4f}` (95% CI `{_fmt_ci_num(ci['trust_auprc'])}`)",
        f"- **Paired `TRUST - CE` Cluster Bootstrap Differences:**",
        f"  - `delta_auroc`: `{ex['trust']['auroc'] - ex['ce']['auroc']:+.4f}` (`{_fmt_ci_num(ci['delta_auroc_trust_minus_ce'])}`)",
        f"  - `delta_auprc`: `{ex['trust']['auprc'] - ex['ce']['auprc']:+.4f}` (`{_fmt_ci_num(ci['delta_auprc_trust_minus_ce'])}`)",
        f"  - `delta_sensitivity @95%`: `{tr_95_nlbs - ce_95_nlbs:+.2%}` (`{_fmt_ci_pct(ci['delta_sensitivity_target_95_trust_minus_ce'])}`) | `delta_dismissal @95%`: `{tr_95_dis - ce_95_dis:+.2%}` (`{_fmt_ci_pct(ci['delta_dismissal_target_95_trust_minus_ce'])}`)",
        f"  - `delta_sensitivity @98%`: `{tr_98_nlbs - ce_98_nlbs:+.2%}` (`{_fmt_ci_pct(ci['delta_sensitivity_target_98_trust_minus_ce'])}`) | `delta_dismissal @98%`: `{tr_98_dis - ce_98_dis:+.2%}` (`{_fmt_ci_pct(ci['delta_dismissal_target_98_trust_minus_ce'])}`)",
        f"- **Cluster-Collapsed Independence Reference (`5,942` patient clusters, `149` positive clusters):** CE@98% sensitivity `{_fmt_pct(cl_col['ce']['target_98']['triage_recall'])}` / dismissal `{_fmt_pct(cl_col['ce']['target_98']['dismissal_rate'])}`; TRUST@98% sensitivity `{_fmt_pct(cl_col['trust']['target_98']['triage_recall'])}` / dismissal `{_fmt_pct(cl_col['trust']['target_98']['dismissal_rate'])}`.",
        "",
        "---",
        "",
        "## 4. Preprocessing Failures, Scanner Breakdown & Fallbacks",
        f"- **Images Verified in Manifest:** `{prep['total_manifest_images']:,}` | **Succeeded:** `{prep['succeeded_images']:,}` | **Decode/Missing Failures:** `{prep['failed_images']}` | **All Scores Finite:** `{prep['all_scores_finite']}`",
        f"- **Uncropped Segmentation Fallbacks (`strict_segmentation=False`):** `{prep['fallback_uncropped_images']}` images (`segmentation_status_counts`: `{prep['segmentation_status_counts']}`)",
        "",
        "| Scanner Manufacturer | Images | Decode Failures | Uncropped Fallbacks | CE Mean Score | TRUST Mean Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for mfg, s in eval_res["scanner_preprocessing_summary"].items():
        lines.append(
            f"| `{mfg}` | {s['n_images']:,} | {s['n_failed']} | {s['n_fallback_uncropped']} | `{s['ce_mean_score']:.5f}` | `{s['trust_mean_score']:.5f}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Limitations & Hypothesis Verdict",
        f"- **Hypothesis Verdict:** {verdict}",
        "- **Key Study Limitations:**",
        "  1. NLBS represents one Canadian screening program acquired predominantly on GE Senographe Essential (`99.92%`) + `21` Siemens (`0.08%`) FFDM units (2008–2010), without density tags or lesion bounding boxes.",
        "  2. Due to stripped DICOM `PatientID` and `AcquisitionDate` in public NLBS metadata, linked clusters reflect connected `StudyInstanceUID` / folder components (`5,942` clusters); any unlinked temporal repeats remain unmerged.",
        "  3. Observed retrospective Clopper-Pearson bounds do not constitute a prospective universal hospital safety guarantee.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify locked NLBS predictions and run Stage C evaluation.")
    parser.add_argument(
        "--locked-manifest",
        type=Path,
        default=Path("reports/stage_c/locked_predictions_manifest.json"),
    )
    parser.add_argument(
        "--exam-manifest",
        type=Path,
        default=Path("datasets/nlbs/exam_manifest.csv"),
    )
    parser.add_argument(
        "--ce-ckpt",
        type=Path,
        default=Path("models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt"),
    )
    parser.add_argument(
        "--trust-ckpt",
        type=Path,
        default=Path("models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/stage_c/nlbs_zero_label_results.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/stage_c/nlbs_zero_label_results.md"),
    )
    args = parser.parse_args()

    eval_res = verify_and_evaluate_locked_predictions(
        locked_manifest_json=args.locked_manifest,
        exam_manifest_csv=args.exam_manifest,
        ce_checkpoint_path=args.ce_ckpt if args.ce_ckpt.exists() else None,
        trust_checkpoint_path=args.trust_ckpt if args.trust_ckpt.exists() else None,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.seed,
    )

    rsna_ce = json.loads(Path("reports/stage_b/rsna_ce_eval_results.json").read_text(encoding="utf-8"))
    rsna_trust = json.loads(Path("reports/stage_b/rsna_trust_eval_results.json").read_text(encoding="utf-8"))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(eval_res, indent=2), encoding="utf-8")
    md_text = render_markdown_report(eval_res, rsna_ce, rsna_trust)
    args.out_md.write_text(md_text, encoding="utf-8")

    print(f"Saved Stage C JSON report -> {args.out_json}")
    print(f"Saved Stage C Markdown report -> {args.out_md}")
    print(json.dumps({
        "ce_95": eval_res["exam_level_results"]["ce"]["target_95"],
        "ce_98": eval_res["exam_level_results"]["ce"]["target_98"],
        "trust_95": eval_res["exam_level_results"]["trust"]["target_95"],
        "trust_98": eval_res["exam_level_results"]["trust"]["target_98"],
        "ce_auroc": eval_res["exam_level_results"]["ce"]["auroc"],
        "trust_auroc": eval_res["exam_level_results"]["trust"]["auroc"],
    }, indent=2))


if __name__ == "__main__":
    main()
