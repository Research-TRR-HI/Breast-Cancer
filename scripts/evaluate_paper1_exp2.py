"""Paper 1 Experiment 2 — CPU-Only Cryptographic Verification & Category C Evaluation CLI.

Strictly enforces Experiment 2 guardrails:
  1. Verifies SHA-256 provenance locks & checkpoint SHA-256 hashes BEFORE reading outcome labels:
     - Reused $0.00 locks (`reports/paper1/predictions_locked/`):
       * `gmic_pretrained` (`gmic_nlbs_provenance_lock.json` / `gmic_nlbs_case_scores.csv`)
       * `nyu_fourview_pretrained` (`nyu_nlbs_4view_provenance_lock.json` / `nyu_nlbs_4view_case_scores.csv`)
     - New Phase 2 locks (`reports/paper1/expanded/exp2/predictions_locked/`):
       * `glam_pretrained` (`glam_pretrained_nlbs_full_provenance_lock.json`)
       * `end2end_cbis_ddsm` (`end2end_cbis_ddsm_nlbs_full_provenance_lock.json`)
  2. Rejects any forbidden outcome columns in locked predictions and rejects any RSNA/NLBS threshold
     calibration or arbitrary 0.5 cutoffs (strictly Category C: threshold-independent AUROC & AUPRC).
  3. Evaluates:
     - Common 4-View NLBS Cohort (N=6,184 studies; all 4 finalized models)
     - Full NLBS Cohort (N=6,359 studies; 3 view-agnostic models: gmic, glam, end2end)
     - Cluster-Collapsed Common 4-View Sensitivity (N=5,868 clusters)
  4. Computes 2,000-replicate paired cluster-bootstrap 95% CIs (over `split_cluster_id`) and
     paired delta-AUROC / delta-AUPRC contrasts, writing `exp2_summary.{json,md}`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.stage_c_eval import (
    assert_no_forbidden_columns,
    compute_auprc,
    compute_auroc,
    compute_sha256_file,
)
from src.paper1.exp2_adapters import (
    EXP2_ELIGIBLE_MODELS,
    EXP2_MODEL_REGISTRY,
    assert_model_eligible,
)
from src.paper1.six_model_eval import _class_score_quantiles, _diff_ci

REUSE_LOCK_FILES: Dict[str, Tuple[str, str]] = {
    "gmic_pretrained": (
        "gmic_nlbs_provenance_lock.json",
        "gmic_nlbs_case_scores.csv",
    ),
    "nyu_fourview_pretrained": (
        "nyu_nlbs_4view_provenance_lock.json",
        "nyu_nlbs_4view_case_scores.csv",
    ),
}


def verify_and_load_exp2_model_scores(
    *,
    model_id: str,
    reuse_lock_dir: Path,
    exp2_lock_dir: Path,
    split_name: str = "nlbs_full",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Verify SHA-256 provenance lock, checkpoint SHA-256, and strict label-free Category C schema."""
    spec = assert_model_eligible(model_id)

    if model_id in REUSE_LOCK_FILES:
        lock_name, csv_name = REUSE_LOCK_FILES[model_id]
        lock_path = reuse_lock_dir / lock_name
        csv_path = reuse_lock_dir / csv_name
        if not lock_path.exists():
            raise FileNotFoundError(f"Missing reused lock manifest for {model_id}: {lock_path}")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("model_id") != model_id:
            raise RuntimeError(f"Reuse lock model_id mismatch: {lock.get('model_id')} != {model_id}")
        if not lock.get("label_quarantine_enforced", False):
            raise RuntimeError(f"{model_id}: label_quarantine_enforced must be true")
        got_ckpt_sha = str(lock.get("checkpoint_sha256", "")).lower()
        if got_ckpt_sha != spec.checkpoint_sha256.lower():
            raise RuntimeError(
                f"{model_id}: checkpoint_sha256 mismatch: {got_ckpt_sha} != {spec.checkpoint_sha256.lower()}"
            )
        if "locked_files_sha256" in lock:
            exp_csv_sha = str(lock["locked_files_sha256"][csv_name]).lower()
        elif "artifacts" in lock and "case_scores_csv" in lock["artifacts"]:
            exp_csv_sha = str(lock["artifacts"]["case_scores_csv"]["sha256"]).lower()
        else:
            raise RuntimeError(f"{model_id}: unrecognized lock checksum schema in {lock_path}")
        got_csv_sha = compute_sha256_file(csv_path).lower()
        if got_csv_sha != exp_csv_sha:
            raise RuntimeError(f"{model_id}: tamper detected in {csv_path}: {got_csv_sha} != {exp_csv_sha}")
    else:
        lock_path = exp2_lock_dir / f"{model_id}_{split_name}_provenance_lock.json"
        csv_name = f"{model_id}_{split_name}_case_scores.csv"
        csv_path = exp2_lock_dir / csv_name
        if not lock_path.exists():
            raise FileNotFoundError(f"Missing Phase 2 lock manifest for {model_id}: {lock_path}")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("model_id") != model_id:
            raise RuntimeError(f"Exp 2 lock model_id mismatch: {lock.get('model_id')} != {model_id}")
        if not lock.get("label_quarantine_enforced", False):
            raise RuntimeError(f"{model_id}: label_quarantine_enforced must be true")
        got_ckpt_sha = str(lock.get("checkpoint_sha256", "")).lower()
        if got_ckpt_sha != spec.checkpoint_sha256.lower():
            raise RuntimeError(
                f"{model_id}: checkpoint_sha256 mismatch: {got_ckpt_sha} != {spec.checkpoint_sha256.lower()}"
            )
        tp = lock.get("threshold_policy") or {}
        if tp.get("threshold_category") != "C":
            raise RuntimeError(f"{model_id}: expected Category C lock policy, got {tp.get('threshold_category')!r}")
        cal_dom = str(tp.get("calibration_domain", "")).upper()
        if "RSNA" in cal_dom or "NLBS" in cal_dom:
            raise RuntimeError(f"{model_id}: forbidden calibration_domain={cal_dom!r}")
        exp_csv_sha = str(lock["locked_files_sha256"][csv_name]).lower()
        got_csv_sha = compute_sha256_file(csv_path).lower()
        if got_csv_sha != exp_csv_sha:
            raise RuntimeError(f"{model_id}: tamper detected in {csv_path}: {got_csv_sha} != {exp_csv_sha}")

    df = pd.read_csv(csv_path, dtype={"StudyInstanceUID": str})
    assert_no_forbidden_columns(df.columns, context=f"exp2_eval_locked_{model_id}")
    if "dismissed_98" in df.columns or "dismissed_95" in df.columns or "dismissed" in df.columns:
        raise RuntimeError(f"{model_id}: Category C lock file {csv_path} must not contain dismissal flags")
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{model_id}: duplicate StudyInstanceUID rows in {csv_path}")
    df["score"] = df["score"].astype(float)
    if not np.all(np.isfinite(df["score"].values)):
        raise ValueError(f"{model_id}: non-finite scores in {csv_path}")

    meta = {
        "model_id": model_id,
        "display_name": spec.display_name,
        "eligibility": spec.eligibility,
        "threshold_category": "C",
        "license": spec.license,
        "training_dataset": spec.training_dataset,
        "checkpoint_name": spec.checkpoint_name,
        "checkpoint_sha256": got_ckpt_sha,
        "lock_json_path": lock_path.as_posix(),
        "lock_json_sha256": compute_sha256_file(lock_path).lower(),
        "case_scores_csv_path": csv_path.as_posix(),
        "case_scores_csv_sha256": got_csv_sha,
        "n_locked_studies": int(len(df)),
    }
    return df[["StudyInstanceUID", "score"]].sort_values("StudyInstanceUID").reset_index(drop=True), meta


def _ci_summary(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
    }


def evaluate_category_c_cohort(
    merged_df: pd.DataFrame,
    *,
    model_ids: Sequence[str],
    cluster_col: str = "split_cluster_id",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate Category C AUROC/AUPRC with paired cluster-bootstrap 95% CIs and pairwise contrasts."""
    y = merged_df["exam_cancer"].values.astype(np.int64)
    clusters = merged_df[cluster_col].astype(str).values
    uniq_clusters = np.array(sorted(pd.unique(clusters)))
    cluster_rows = {c: np.where(clusters == c)[0] for c in uniq_clusters}
    n_clusters = len(uniq_clusters)

    rng = np.random.default_rng(seed)
    boot_indices: List[np.ndarray] = []
    for _ in range(n_boot):
        sampled = uniq_clusters[rng.integers(0, n_clusters, size=n_clusters)]
        idx = np.concatenate([cluster_rows[c] for c in sampled])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        boot_indices.append(idx)

    per_model: Dict[str, Any] = {}
    boot_traces: Dict[str, Dict[str, np.ndarray]] = {}
    for m in model_ids:
        s = merged_df[f"{m}_score"].values.astype(np.float64)
        auroc_pt = float(compute_auroc(y, s))
        auprc_pt = float(compute_auprc(y, s))
        aurocs = np.asarray([compute_auroc(y[idx], s[idx]) for idx in boot_indices], dtype=np.float64)
        auprcs = np.asarray([compute_auprc(y[idx], s[idx]) for idx in boot_indices], dtype=np.float64)
        boot_traces[m] = {"auroc": aurocs, "auprc": auprcs}
        per_model[m] = {
            "threshold_category": "C",
            "auroc": auroc_pt,
            "auprc": auprc_pt,
            "bootstrap_95ci": {
                "auroc": _ci_summary(aurocs),
                "auprc": _ci_summary(auprcs),
            },
            "score_quantiles": _class_score_quantiles(y, s),
        }

    paired_contrasts: Dict[str, Any] = {}
    for i, m_a in enumerate(model_ids):
        for m_b in model_ids[i + 1 :]:
            key = f"{m_a}_minus_{m_b}"
            paired_contrasts[key] = {
                "delta_auroc_point": float(per_model[m_a]["auroc"] - per_model[m_b]["auroc"]),
                "delta_auroc": _diff_ci(boot_traces[m_a]["auroc"] - boot_traces[m_b]["auroc"]),
                "delta_auprc_point": float(per_model[m_a]["auprc"] - per_model[m_b]["auprc"]),
                "delta_auprc": _diff_ci(boot_traces[m_a]["auprc"] - boot_traces[m_b]["auprc"]),
            }

    return {
        "n_units": int(len(merged_df)),
        "n_clusters": int(n_clusters),
        "n_positive": int(y.sum()),
        "prevalence": float(np.mean(y)),
        "n_valid_bootstrap_reps": int(len(boot_indices)),
        "models": per_model,
        "paired_contrasts": paired_contrasts,
    }


def evaluate_exp2(
    *,
    reuse_lock_dir: Path,
    exp2_lock_dir: Path,
    exam_manifest_csv: Path,
    common_4view_csv: Path,
    out_dir: Path,
    split_name: str = "nlbs_full",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify all 4 finalized Exp 2 locks, join NLBS outcomes, evaluate Category C metrics, and write summary."""
    locked_scores: Dict[str, pd.DataFrame] = {}
    provenance: Dict[str, Dict[str, Any]] = {}
    for mid in EXP2_ELIGIBLE_MODELS:
        df_s, meta = verify_and_load_exp2_model_scores(
            model_id=mid,
            reuse_lock_dir=reuse_lock_dir,
            exp2_lock_dir=exp2_lock_dir,
            split_name=split_name,
        )
        locked_scores[mid] = df_s
        provenance[mid] = meta

    exam_raw = pd.read_csv(exam_manifest_csv)
    exam_raw["StudyInstanceUID"] = exam_raw["StudyInstanceUID"].astype(str)
    dedup_exam = (
        exam_raw.drop_duplicates(subset=["StudyInstanceUID"], keep="first")[
            ["StudyInstanceUID", "split_cluster_id", "exam_cancer"]
        ]
        .sort_values("StudyInstanceUID")
        .reset_index(drop=True)
    )

    c4v_df = pd.read_csv(common_4view_csv, dtype={"StudyInstanceUID": str})
    assert_no_forbidden_columns(c4v_df.columns, context=f"exp2_eval_{common_4view_csv.name}")
    c4v_studies = set(c4v_df["StudyInstanceUID"].unique().tolist())

    merged_common = dedup_exam[dedup_exam["StudyInstanceUID"].isin(c4v_studies)].copy()
    for mid in EXP2_ELIGIBLE_MODELS:
        merged_common = merged_common.merge(
            locked_scores[mid].rename(columns={"score": f"{mid}_score"}),
            on="StudyInstanceUID",
            how="inner",
        )
    merged_common = merged_common.sort_values("StudyInstanceUID").reset_index(drop=True)
    if len(merged_common) != len(c4v_studies):
        raise RuntimeError(
            f"Common 4-view alignment mismatch: {len(merged_common)} merged vs {len(c4v_studies)} expected"
        )

    common_eval = evaluate_category_c_cohort(
        merged_common,
        model_ids=EXP2_ELIGIBLE_MODELS,
        cluster_col="split_cluster_id",
        n_boot=n_boot,
        seed=seed,
    )

    full_models = [m for m in EXP2_ELIGIBLE_MODELS if m != "nyu_fourview_pretrained"]
    merged_full = dedup_exam.copy()
    for mid in full_models:
        merged_full = merged_full.merge(
            locked_scores[mid].rename(columns={"score": f"{mid}_score"}),
            on="StudyInstanceUID",
            how="inner",
        )
    merged_full = merged_full.sort_values("StudyInstanceUID").reset_index(drop=True)
    if len(merged_full) != len(dedup_exam):
        raise RuntimeError(
            f"Full NLBS alignment mismatch: {len(merged_full)} merged vs {len(dedup_exam)} expected"
        )

    full_eval = evaluate_category_c_cohort(
        merged_full,
        model_ids=full_models,
        cluster_col="split_cluster_id",
        n_boot=n_boot,
        seed=seed,
    )

    agg_spec: Dict[str, Any] = {"exam_cancer": "max"}
    for mid in EXP2_ELIGIBLE_MODELS:
        agg_spec[f"{mid}_score"] = "max"
    collapsed_df = (
        merged_common.groupby("split_cluster_id", as_index=False)
        .agg(agg_spec)
        .sort_values("split_cluster_id")
        .reset_index(drop=True)
    )
    collapsed_eval = evaluate_category_c_cohort(
        collapsed_df,
        model_ids=EXP2_ELIGIBLE_MODELS,
        cluster_col="split_cluster_id",
        n_boot=n_boot,
        seed=seed,
    )

    excluded = {
        mid: {
            "eligibility": spec.eligibility,
            "reason": spec.exclusion_reason,
            "checkpoint_name": spec.checkpoint_name,
        }
        for mid, spec in EXP2_MODEL_REGISTRY.items()
        if spec.eligibility in ("blocked_upstream", "disqualified", "rejected")
    }

    report: Dict[str, Any] = {
        "schema_version": "paper1_exp2_summary_v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_policy": {
            "primary_threshold_category": "C",
            "primary_metrics": ["AUROC", "AUPRC"],
            "rsna_calibration_applied": False,
            "arbitrary_05_cutoff_applied": False,
            "n_boot": int(n_boot),
            "seed": int(seed),
            "nyu_fourview_caveat": "Single-checkpoint sample_image_model.p vs 100-model ensemble; requires 4 views (evaluated on Common 4-View only).",
        },
        "provenance_locks": provenance,
        "excluded_models": excluded,
        "nlbs_common_4view_primary": common_eval,
        "nlbs_full_cohort_secondary": full_eval,
        "nlbs_cluster_collapsed_common_4view": collapsed_eval,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exp2_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "exp2_summary.md").write_text(render_exp2_markdown(report), encoding="utf-8")
    return report


def render_exp2_markdown(report: Dict[str, Any]) -> str:
    c4v = report["nlbs_common_4view_primary"]
    full = report["nlbs_full_cohort_secondary"]
    col = report["nlbs_cluster_collapsed_common_4view"]
    prov = report["provenance_locks"]

    lines = [
        "# Paper 1 Experiment 2 — Original Pretrained Mammography Models (NLBS Evaluation)",
        "",
        f"**Evaluated UTC:** `{report['evaluated_at_utc']}`  ",
        "**Protocol Policy:** Strict **Category C** (threshold-independent AUROC & AUPRC only; zero RSNA threshold calibration; zero arbitrary `0.5` cutoff; zero target-label tuning).  ",
        f"**Cluster Bootstrap:** `{report['protocol_policy']['n_boot']}` paired cluster-bootstrap replicates (`seed={report['protocol_policy']['seed']}`, resampled by `split_cluster_id`).",
        "",
        "---",
        "",
        f"## 1. Common 4-View NLBS Cohort (All 4 Finalized Models — N = {c4v['n_units']:,}, {c4v['n_clusters']:,} clusters, {c4v['n_positive']} cancers)",
        "",
        "| Model ID | Source & License | AUROC (95% CI) | AUPRC (95% CI) | Pos Median [p05, p95] | Neg Median [p05, p95] |",
        "|---|---|---|---|---|---|",
    ]
    for mid, m in c4v["models"].items():
        b_auc = m["bootstrap_95ci"]["auroc"]
        b_prc = m["bootstrap_95ci"]["auprc"]
        qp = m["score_quantiles"]["positive"]
        qn = m["score_quantiles"]["negative"]
        lines.append(
            f"| `{mid}` | {prov[mid]['license']} | "
            f"**{m['auroc']:.4f}** [{b_auc['ci95_low']:.4f}, {b_auc['ci95_high']:.4f}] | "
            f"**{m['auprc']:.4f}** [{b_prc['ci95_low']:.4f}, {b_prc['ci95_high']:.4f}] | "
            f"`{qp['p50']:.4f}` [{qp['p05']:.4f}, {qp['p95']:.4f}] | "
            f"`{qn['p50']:.4f}` [{qn['p05']:.4f}, {qn['p95']:.4f}] |"
        )

    lines.extend(
        [
            "",
            "### Paired Contrasts (Common 4-View NLBS)",
            "",
            "| Contrast | Delta AUROC (95% CI) | Delta AUPRC (95% CI) |",
            "|---|---|---|",
        ]
    )
    for cname, cval in c4v["paired_contrasts"].items():
        da = cval["delta_auroc"]
        dp = cval["delta_auprc"]
        lines.append(
            f"| `{cname}` | {cval['delta_auroc_point']:+.4f} [{da['ci95_low']:+.4f}, {da['ci95_high']:+.4f}] | "
            f"{cval['delta_auprc_point']:+.4f} [{dp['ci95_low']:+.4f}, {dp['ci95_high']:+.4f}] |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 2. Full NLBS Cohort (3 View-Agnostic Models — N = {full['n_units']:,}, {full['n_clusters']:,} clusters, {full['n_positive']} cancers)",
            "",
            "*Note: `nyu_fourview_pretrained` is excluded from Full NLBS because 175 non-4-view exams lack all 4 standard views (no view fabrication).* ",
            "",
            "| Model ID | AUROC (95% CI) | AUPRC (95% CI) |",
            "|---|---|---|",
        ]
    )
    for mid, m in full["models"].items():
        b_auc = m["bootstrap_95ci"]["auroc"]
        b_prc = m["bootstrap_95ci"]["auprc"]
        lines.append(
            f"| `{mid}` | **{m['auroc']:.4f}** [{b_auc['ci95_low']:.4f}, {b_auc['ci95_high']:.4f}] | "
            f"**{m['auprc']:.4f}** [{b_prc['ci95_low']:.4f}, {b_prc['ci95_high']:.4f}] |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 3. Cluster-Collapsed Common 4-View Sensitivity (N = {col['n_units']:,} clusters, {col['n_positive']} positive clusters)",
            "",
            "| Model ID | AUROC (95% CI) | AUPRC (95% CI) |",
            "|---|---|---|",
        ]
    )
    for mid, m in col["models"].items():
        b_auc = m["bootstrap_95ci"]["auroc"]
        b_prc = m["bootstrap_95ci"]["auprc"]
        lines.append(
            f"| `{mid}` | {m['auroc']:.4f} [{b_auc['ci95_low']:.4f}, {b_auc['ci95_high']:.4f}] | "
            f"{m['auprc']:.4f} [{b_prc['ci95_low']:.4f}, {b_prc['ci95_high']:.4f}] |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Cryptographic SHA-256 Locks & Excluded Candidates",
            "",
            "| Model ID | Checkpoint SHA-256 | Case Scores CSV SHA-256 | Lock JSON SHA-256 |",
            "|---|---|---|---|",
        ]
    )
    for mid, p in prov.items():
        lines.append(
            f"| `{mid}` | `{p['checkpoint_sha256']}` | `{p['case_scores_csv_sha256']}` | `{p['lock_json_sha256']}` |"
        )
    lines.append("")
    for mid, ex in report["excluded_models"].items():
        lines.append(f"- **`{mid}` (`{ex['eligibility']}`):** {ex['reason']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only Category C evaluation for Paper 1 Exp 2.")
    parser.add_argument("--reuse-lock-dir", type=Path, default=Path("reports/paper1/predictions_locked"))
    parser.add_argument(
        "--exp2-lock-dir",
        type=Path,
        default=Path("reports/paper1/expanded/exp2/predictions_locked"),
    )
    parser.add_argument("--exam-manifest", type=Path, default=Path("datasets/nlbs/exam_manifest.csv"))
    parser.add_argument(
        "--common-4view-csv",
        type=Path,
        default=Path("datasets/paper1/nlbs_4view_common_cohort_label_free.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("reports/paper1/expanded/exp2"))
    parser.add_argument("--split-name", default="nlbs_full")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = evaluate_exp2(
        reuse_lock_dir=args.reuse_lock_dir,
        exp2_lock_dir=args.exp2_lock_dir,
        exam_manifest_csv=args.exam_manifest,
        common_4view_csv=args.common_4view_csv,
        out_dir=args.out_dir,
        split_name=args.split_name,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "common_4view": {
                    m: {"auroc": v["auroc"], "auprc": v["auprc"]}
                    for m, v in report["nlbs_common_4view_primary"]["models"].items()
                },
                "full_cohort": {
                    m: {"auroc": v["auroc"], "auprc": v["auprc"]}
                    for m, v in report["nlbs_full_cohort_secondary"]["models"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
