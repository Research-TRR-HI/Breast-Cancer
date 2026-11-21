"""Paper 1 — Six-Model External Validation, Paired Statistical Contrasts & Failure Decomposition.

Implements the prespecified analysis plan in `studies/paper1/analysis_plan.md`:
    - Strict separation between Stratum A (RSNA-supervised: ce_b5, trust_b5, dinov2_rsna, resnet50_rsna)
      and Stratum B (Pretrained-direct: gmic_pretrained, nyu_fourview_pretrained).
    - Independent 2-fold patient-disjoint cross-fitting on RSNA `val_calib` vs. full-`val_calib`
      threshold freezing for NLBS.
    - Target-label quarantine and cryptographic pre-join prediction locking.
    - Primary Common 4-View NLBS evaluation (N=6,184, 5,868 split_cluster_id clusters, 151 pos),
      Secondary Full-Cohort (N=6,359, 5 view-agnostic models), and Cluster-Collapsed sensitivity.
    - Paired cluster bootstrap (B=2,000), exact McNemar comparisons, score-shift quantiles + KS drift,
      crop-fallback sensitivity auditing, and missed-cancer Jaccard/UpSet overlap analysis.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.stage_c_eval import (
    assert_no_forbidden_columns,
    compute_auprc,
    compute_auroc,
    compute_sha256_file,
)
from src.training.calibration import (
    clopper_pearson_upper_bound,
    compute_target_recall_threshold,
    evaluate_dismissal_at_threshold,
)

STRATUM_A_MODELS: Tuple[str, ...] = ("ce_b5", "trust_b5", "dinov2_rsna", "resnet50_rsna")
STRATUM_B_MODELS: Tuple[str, ...] = ("gmic_pretrained", "nyu_fourview_pretrained")
ALL_SIX_MODELS: Tuple[str, ...] = STRATUM_A_MODELS + STRATUM_B_MODELS

MODEL_STRATUM: Dict[str, str] = {
    **{m: "A_rsna_supervised" for m in STRATUM_A_MODELS},
    **{m: "B_pretrained_direct" for m in STRATUM_B_MODELS},
}


def calibrate_rsna_thresholds_only(
    rsna_cases_df: pd.DataFrame,
    fold_csv: Path,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calibrate RSNA source thresholds (2-fold OOF independent eval + full val_calib freeze)."""
    fold_df = pd.read_csv(fold_csv)
    pt_fold = fold_df.groupby("patient_id")["fold"].first().to_dict()

    df = rsna_cases_df[["patient_id", "cancer", "score"]].copy()
    df["patient_id"] = df["patient_id"].astype(int)
    df["fold"] = df["patient_id"].map(pt_fold)
    if df["fold"].isna().any():
        raise ValueError("Some RSNA val_calib patients are missing from the 2-fold assignment manifest.")

    oof_parts: List[pd.DataFrame] = []
    fold_eval: Dict[str, Any] = {}
    for cal_fold, test_fold in (("fold_a", "fold_b"), ("fold_b", "fold_a")):
        cal = df[df["fold"] == cal_fold]
        tst = df[df["fold"] == test_fold].copy()
        pos_cal = cal[cal["cancer"] == 1]["score"].values
        t98 = float(compute_target_recall_threshold(pos_cal, 0.98))
        t95 = float(compute_target_recall_threshold(pos_cal, 0.95))
        tst["dismissed_98"] = tst["score"].values < t98
        tst["dismissed_95"] = tst["score"].values < t95
        oof_parts.append(tst)
        fold_eval[f"{cal_fold}_to_{test_fold}"] = {
            "tau_98": t98,
            "tau_95": t95,
            "target_98": evaluate_dismissal_at_threshold(tst["score"].values, tst["cancer"].values, t98, 0.99, 0.98),
            "target_95": evaluate_dismissal_at_threshold(tst["score"].values, tst["cancer"].values, t95, 0.99, 0.95),
        }

    oof = pd.concat(oof_parts, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
    y = oof["cancer"].values.astype(np.int64)
    s = oof["score"].values.astype(np.float64)
    d98 = oof["dismissed_98"].values.astype(bool)
    d95 = oof["dismissed_95"].values.astype(bool)

    rng = np.random.default_rng(seed)
    boot_vals: Dict[str, List[float]] = {k: [] for k in ("auroc", "auprc", "s98", "d98", "s95", "d95")}
    n = len(oof)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        sb = s[idx]
        pos_m = yb == 1
        boot_vals["auroc"].append(compute_auroc(yb, sb))
        boot_vals["auprc"].append(compute_auprc(yb, sb))
        boot_vals["s98"].append(float(np.mean(~d98[idx][pos_m])))
        boot_vals["d98"].append(float(np.mean(d98[idx])))
        boot_vals["s95"].append(float(np.mean(~d95[idx][pos_m])))
        boot_vals["d95"].append(float(np.mean(d95[idx])))

    def _ci(v: List[float]) -> Dict[str, float]:
        arr = np.asarray(v, dtype=np.float64)
        return {"mean": float(np.mean(arr)), "ci95_low": float(np.percentile(arr, 2.5)), "ci95_high": float(np.percentile(arr, 97.5))}

    pos_full = df[df["cancer"] == 1]["score"].values
    tau98_full = float(compute_target_recall_threshold(pos_full, 0.98))
    tau95_full = float(compute_target_recall_threshold(pos_full, 0.95))

    return {
        "n_patients": int(n),
        "n_pos": int(y.sum()),
        "auroc": float(compute_auroc(y, s)),
        "auprc": float(compute_auprc(y, s)),
        "oof_pooled": {
            "target_98_sens": float(np.mean(~d98[y == 1])),
            "target_98_missed": int(np.sum(d98[y == 1])),
            "target_98_dis": float(np.mean(d98)),
            "target_95_sens": float(np.mean(~d95[y == 1])),
            "target_95_missed": int(np.sum(d95[y == 1])),
            "target_95_dis": float(np.mean(d95)),
        },
        "bootstrap_95ci": {k: _ci(v) for k, v in boot_vals.items()},
        "fold_details": fold_eval,
        "full_val_calib_thresholds": {"target_98": tau98_full, "target_95": tau95_full},
        "fold_a_sensitivity_thresholds": {
            "target_98": fold_eval["fold_a_to_fold_b"]["tau_98"],
            "target_95": fold_eval["fold_a_to_fold_b"]["tau_95"],
        },
        "score_quantiles": _class_score_quantiles(y, s),
    }


def lock_nlbs_model_predictions(
    model_id: str,
    exam_scores_df: pd.DataFrame,
    thresholds_full: Dict[str, float],
    model_sha256: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Cryptographically freeze label-free NLBS predictions and RSNA thresholds prior to label join."""
    assert_no_forbidden_columns(exam_scores_df.columns, context=f"lock_nlbs_{model_id}")
    req_cols = {"StudyInstanceUID", "score"}
    missing = req_cols - set(exam_scores_df.columns)
    if missing:
        raise ValueError(f"{model_id} exam_scores_df missing required columns: {sorted(missing)}")

    df = exam_scores_df.copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{model_id} contains duplicate StudyInstanceUID rows.")
    df = df.sort_values("StudyInstanceUID").reset_index(drop=True)

    tau98 = float(thresholds_full["target_98"])
    tau95 = float(thresholds_full["target_95"])
    df["score"] = df["score"].astype(float)
    if not np.all(np.isfinite(df["score"].values)):
        raise ValueError(f"{model_id} contains non-finite NLBS exam scores.")

    df["dismissed_98"] = df["score"].values < tau98
    df["dismissed_95"] = df["score"].values < tau95
    if "n_crop_fallbacks" not in df.columns:
        df["n_crop_fallbacks"] = 0
    df["has_crop_fallback"] = df["n_crop_fallbacks"].astype(int) > 0

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{model_id}_locked_nlbs_predictions.csv"
    manifest_path = out_dir / f"{model_id}_lock_manifest.json"
    df[["StudyInstanceUID", "score", "dismissed_98", "dismissed_95", "n_crop_fallbacks", "has_crop_fallback"]].to_csv(
        csv_path, index=False
    )

    manifest = {
        "schema_version": "paper1_locked_predictions_v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "stratum": MODEL_STRATUM[model_id],
        "model_sha256": str(model_sha256),
        "frozen_rsna_thresholds": {"target_98": tau98, "target_95": tau95},
        "n_studies": int(len(df)),
        "n_studies_with_crop_fallback": int(df["has_crop_fallback"].sum()),
        "predictions_csv": str(csv_path.name),
        "predictions_sha256": compute_sha256_file(csv_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_and_load_locked_model(lock_dir: Path, model_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Verify SHA-256 checksums and strict threshold inequalities before any label join."""
    manifest_path = lock_dir / f"{model_id}_lock_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    csv_path = lock_dir / manifest["predictions_csv"]
    got_sha = compute_sha256_file(csv_path)
    if got_sha != manifest["predictions_sha256"]:
        raise RuntimeError(f"Lock tamper detected for {model_id}: {got_sha} != {manifest['predictions_sha256']}")

    df = pd.read_csv(csv_path)
    assert_no_forbidden_columns(df.columns, context=f"locked_{model_id}")
    tau98 = float(manifest["frozen_rsna_thresholds"]["target_98"])
    tau95 = float(manifest["frozen_rsna_thresholds"]["target_95"])
    if not np.array_equal(df["dismissed_98"].values.astype(bool), df["score"].values < tau98):
        raise RuntimeError(f"dismissed_98 threshold inconsistency in {model_id}")
    if not np.array_equal(df["dismissed_95"].values.astype(bool), df["score"].values < tau95):
        raise RuntimeError(f"dismissed_95 threshold inconsistency in {model_id}")
    return df, manifest


def _class_score_quantiles(y: np.ndarray, s: np.ndarray) -> Dict[str, Dict[str, float]]:
    qs = (5, 25, 50, 75, 95)
    out: Dict[str, Dict[str, float]] = {}
    for cls_name, mask in (("positive", y == 1), ("negative", y == 0)):
        arr = s[mask]
        if len(arr) == 0:
            out[cls_name] = {}
        else:
            out[cls_name] = {f"p{q:02d}": float(np.percentile(arr, q)) for q in qs}
            out[cls_name]["mean"] = float(np.mean(arr))
    return out


def _mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Two-sided exact McNemar binomial test on paired binary outcomes."""
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    n10 = int(np.sum(a_b & ~b_b))
    n01 = int(np.sum(~a_b & b_b))
    n_disc = n10 + n01
    p_val = 1.0 if n_disc == 0 else float(stats.binomtest(min(n10, n01), n_disc, 0.5, alternative="two-sided").pvalue)
    return {"only_a": n10, "only_b": n01, "p_value": p_val}


def _evaluate_single_cohort_df(
    merged_df: pd.DataFrame,
    model_ids: Sequence[str],
    thresholds_map: Dict[str, Dict[str, float]],
    cluster_col: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate aligned cohort across specified models with identical cluster-bootstrap resamples."""
    y = merged_df["exam_cancer"].values.astype(np.int64)
    clusters = merged_df[cluster_col].astype(str).values
    uniq_clusters = np.array(sorted(pd.unique(clusters)))
    cluster_rows = {c: np.where(clusters == c)[0] for c in uniq_clusters}
    n_clusters = len(uniq_clusters)

    rng = np.random.default_rng(seed)
    boot_indices_list: List[np.ndarray] = []
    for _ in range(n_boot):
        sampled_c = uniq_clusters[rng.integers(0, n_clusters, size=n_clusters)]
        row_idx = np.concatenate([cluster_rows[c] for c in sampled_c])
        yb = y[row_idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        boot_indices_list.append(row_idx)

    per_model: Dict[str, Any] = {}
    boot_traces: Dict[str, Dict[str, np.ndarray]] = {}

    for m in model_ids:
        s = merged_df[f"{m}_score"].values.astype(np.float64)
        tau98 = float(thresholds_map[m]["target_98"])
        tau95 = float(thresholds_map[m]["target_95"])
        d98 = s < tau98
        d95 = s < tau95
        fb_col = f"{m}_has_crop_fallback"
        has_fb = merged_df[fb_col].values.astype(bool) if fb_col in merged_df.columns else np.zeros(len(s), dtype=bool)

        ev98 = evaluate_dismissal_at_threshold(s, y, tau98, 0.99, 0.98)
        ev95 = evaluate_dismissal_at_threshold(s, y, tau95, 0.99, 0.95)
        auroc_pt = float(compute_auroc(y, s))
        auprc_pt = float(compute_auprc(y, s))

        aurocs, auprcs, s98_b, d98_b, s95_b, d95_b = [], [], [], [], [], []
        for idx in boot_indices_list:
            yb = y[idx]
            sb = s[idx]
            pos_m = yb == 1
            aurocs.append(compute_auroc(yb, sb))
            auprcs.append(compute_auprc(yb, sb))
            s98_b.append(float(np.mean(~d98[idx][pos_m])))
            d98_b.append(float(np.mean(d98[idx])))
            s95_b.append(float(np.mean(~d95[idx][pos_m])))
            d95_b.append(float(np.mean(d95[idx])))

        def _ci(arr_l: List[float]) -> Dict[str, float]:
            a = np.asarray(arr_l, dtype=np.float64)
            return {"mean": float(np.mean(a)), "ci95_low": float(np.percentile(a, 2.5)), "ci95_high": float(np.percentile(a, 97.5))}

        boot_traces[m] = {
            "auroc": np.asarray(aurocs, dtype=np.float64),
            "sens98": np.asarray(s98_b, dtype=np.float64),
            "dis98": np.asarray(d98_b, dtype=np.float64),
            "sens95": np.asarray(s95_b, dtype=np.float64),
            "dis95": np.asarray(d95_b, dtype=np.float64),
        }

        fb_excluded = None
        if np.any(has_fb):
            clean_m = ~has_fb
            fb_excluded = {
                "n_fallback_studies": int(np.sum(has_fb)),
                "n_fallback_pos": int(np.sum(y[has_fb] == 1)),
                "clean_auroc": float(compute_auroc(y[clean_m], s[clean_m])),
                "clean_target_98": evaluate_dismissal_at_threshold(s[clean_m], y[clean_m], tau98, 0.99, 0.98),
            }

        per_model[m] = {
            "stratum": MODEL_STRATUM[m],
            "auroc": auroc_pt,
            "auprc": auprc_pt,
            "target_98": ev98,
            "target_95": ev95,
            "bootstrap_95ci": {
                "auroc": _ci(aurocs),
                "auprc": _ci(auprcs),
                "target_98_sens": _ci(s98_b),
                "target_98_dis": _ci(d98_b),
                "target_95_sens": _ci(s95_b),
                "target_95_dis": _ci(d95_b),
            },
            "score_quantiles": _class_score_quantiles(y, s),
            "fallback_audit": {
                "n_studies_with_fallback": int(np.sum(has_fb)),
                "pct_studies_with_fallback": float(np.mean(has_fb) * 100.0),
                "excluded_sensitivity_check": fb_excluded,
            },
        }

    # Paired contrasts (vs ce_b5 anchor & all pairs within/across strata)
    paired_contrasts: Dict[str, Any] = {}
    for i, m_a in enumerate(model_ids):
        for m_b in model_ids[i + 1 :]:
            key = f"{m_a}_minus_{m_b}"
            sa_d98 = merged_df[f"{m_a}_score"].values < thresholds_map[m_a]["target_98"]
            sb_d98 = merged_df[f"{m_b}_score"].values < thresholds_map[m_b]["target_98"]
            pos_m = y == 1
            paired_contrasts[key] = {
                "delta_auroc": _diff_ci(boot_traces[m_a]["auroc"] - boot_traces[m_b]["auroc"]),
                "delta_sens98": _diff_ci(boot_traces[m_a]["sens98"] - boot_traces[m_b]["sens98"]),
                "delta_dis98": _diff_ci(boot_traces[m_a]["dis98"] - boot_traces[m_b]["dis98"]),
                "delta_sens95": _diff_ci(boot_traces[m_a]["sens95"] - boot_traces[m_b]["sens95"]),
                "delta_dis95": _diff_ci(boot_traces[m_a]["dis95"] - boot_traces[m_b]["dis95"]),
                "mcnemar_cancer_retained_98": _mcnemar_exact(~sa_d98[pos_m], ~sb_d98[pos_m]),
                "mcnemar_dismissed_98": _mcnemar_exact(sa_d98, sb_d98),
            }

    overlap = _missed_cancer_overlap(merged_df, model_ids, thresholds_map)
    return {
        "n_units": int(len(merged_df)),
        "n_clusters": int(n_clusters),
        "n_positive": int(y.sum()),
        "models": per_model,
        "paired_contrasts": paired_contrasts,
        "missed_cancer_overlap": overlap,
    }


def _diff_ci(diff: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(diff)),
        "ci95_low": float(np.percentile(diff, 2.5)),
        "ci95_high": float(np.percentile(diff, 97.5)),
    }


def _missed_cancer_overlap(
    merged_df: pd.DataFrame,
    model_ids: Sequence[str],
    thresholds_map: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Compute pairwise Jaccard overlap and UpSet pattern counts among dismissed cancers."""
    pos_df = merged_df[merged_df["exam_cancer"] == 1].reset_index(drop=True)
    out: Dict[str, Any] = {"n_pos": int(len(pos_df))}
    for op_key in ("target_98", "target_95"):
        missed_sets: Dict[str, np.ndarray] = {}
        jaccard: Dict[str, Dict[str, float]] = {}
        for m in model_ids:
            missed_sets[m] = pos_df[f"{m}_score"].values < float(thresholds_map[m][op_key])
        for m_a in model_ids:
            jaccard[m_a] = {}
            for m_b in model_ids:
                inter = int(np.sum(missed_sets[m_a] & missed_sets[m_b]))
                union = int(np.sum(missed_sets[m_a] | missed_sets[m_b]))
                jaccard[m_a][m_b] = 1.0 if union == 0 else float(inter / union)

        upset: Dict[str, int] = {}
        for idx in range(len(pos_df)):
            pattern = "|".join([m for m in model_ids if missed_sets[m][idx]]) or "NONE_MISSED"
            upset[pattern] = upset.get(pattern, 0) + 1

        out[op_key] = {
            "missed_counts": {m: int(np.sum(missed_sets[m])) for m in model_ids},
            "pairwise_jaccard": jaccard,
            "upset_patterns": dict(sorted(upset.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
    return out


def compute_rsna_to_nlbs_shift_decomposition(
    rsna_cases_df: pd.DataFrame,
    rsna_eval: Dict[str, Any],
    nlbs_merged_df: pd.DataFrame,
    nlbs_model_metrics: Dict[str, Any],
    model_id: str,
) -> Dict[str, Any]:
    """Compare RSNA vs. NLBS discrimination and score calibration distributions."""
    r_y = rsna_cases_df["cancer"].values.astype(int)
    r_s = rsna_cases_df["score"].values.astype(float)
    n_y = nlbs_merged_df["exam_cancer"].values.astype(int)
    n_s = nlbs_merged_df[f"{model_id}_score"].values.astype(float)

    ks_neg = stats.ks_2samp(r_s[r_y == 0], n_s[n_y == 0])
    ks_pos = stats.ks_2samp(r_s[r_y == 1], n_s[n_y == 1])

    rsna_auroc = float(rsna_eval["auroc"])
    nlbs_auroc = float(nlbs_model_metrics["auroc"])
    rsna_s98 = float(rsna_eval["oof_pooled"]["target_98_sens"])
    nlbs_s98 = float(nlbs_model_metrics["target_98"]["triage_recall"])

    return {
        "stratum": MODEL_STRATUM[model_id],
        "delta_auroc_nlbs_minus_rsna": nlbs_auroc - rsna_auroc,
        "delta_sens98_nlbs_minus_rsna": nlbs_s98 - rsna_s98,
        "neg_median_shift": float(np.median(n_s[n_y == 0]) - np.median(r_s[r_y == 0])),
        "pos_median_shift": float(np.median(n_s[n_y == 1]) - np.median(r_s[r_y == 1])),
        "ks_neg_stat": float(ks_neg.statistic),
        "ks_neg_pvalue": float(ks_neg.pvalue),
        "ks_pos_stat": float(ks_pos.statistic),
        "ks_pos_pvalue": float(ks_pos.pvalue),
    }


def evaluate_six_models(
    locked_dir: Path,
    rsna_case_dfs: Dict[str, pd.DataFrame],
    fold_csv: Path,
    exam_manifest_csv: Path,
    common_4view_csv: Path,
    out_dir: Path,
    model_ids: Sequence[str] = ALL_SIX_MODELS,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run complete six-model verification, RSNA OOF eval, NLBS multi-cohort eval, and statistical decomposition."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rsna_summary: Dict[str, Any] = {}
    thresholds_map: Dict[str, Dict[str, float]] = {}
    locked_dfs: Dict[str, pd.DataFrame] = {}

    for m in model_ids:
        rsna_summary[m] = calibrate_rsna_thresholds_only(rsna_case_dfs[m], fold_csv=fold_csv, n_boot=n_boot, seed=seed)
        thresholds_map[m] = rsna_summary[m]["full_val_calib_thresholds"]
        df_lock, manifest = verify_and_load_locked_model(locked_dir, m)
        # Ensure lock manifest thresholds match RSNA-calibrated thresholds cleanly
        if not math.isclose(manifest["frozen_rsna_thresholds"]["target_98"], thresholds_map[m]["target_98"], rel_tol=0.0, abs_tol=1e-7):
            raise RuntimeError(f"Frozen threshold mismatch vs RSNA calibration for {m} @98%")
        locked_dfs[m] = df_lock

    exam_df = pd.read_csv(exam_manifest_csv)
    exam_df["StudyInstanceUID"] = exam_df["StudyInstanceUID"].astype(str)
    dedup_exam = exam_df.drop_duplicates(subset=["StudyInstanceUID"], keep="first")[
        ["StudyInstanceUID", "split_cluster_id", "exam_cancer"]
    ].reset_index(drop=True)

    c4v_df = pd.read_csv(common_4view_csv)
    assert_no_forbidden_columns(c4v_df.columns, context=str(common_4view_csv))
    c4v_studies = set(c4v_df["StudyInstanceUID"].astype(str).unique().tolist())

    merged_common = dedup_exam[dedup_exam["StudyInstanceUID"].isin(c4v_studies)].copy()
    for m in model_ids:
        sub = locked_dfs[m].rename(
            columns={"score": f"{m}_score", "has_crop_fallback": f"{m}_has_crop_fallback"}
        )[["StudyInstanceUID", f"{m}_score", f"{m}_has_crop_fallback"]]
        merged_common = merged_common.merge(sub, on="StudyInstanceUID", how="inner")

    common_eval = _evaluate_single_cohort_df(
        merged_common.sort_values("StudyInstanceUID").reset_index(drop=True),
        model_ids=model_ids,
        thresholds_map=thresholds_map,
        cluster_col="split_cluster_id",
        n_boot=n_boot,
        seed=seed,
    )

    # Secondary Full-Cohort Evaluation across view-agnostic models (excluding nyu_fourview_pretrained)
    full_models = [m for m in model_ids if m != "nyu_fourview_pretrained"]
    full_eval = None
    if full_models:
        merged_full = dedup_exam.copy()
        for m in full_models:
            sub = locked_dfs[m].rename(
                columns={"score": f"{m}_score", "has_crop_fallback": f"{m}_has_crop_fallback"}
            )[["StudyInstanceUID", f"{m}_score", f"{m}_has_crop_fallback"]]
            merged_full = merged_full.merge(sub, on="StudyInstanceUID", how="inner")
        if len(merged_full) > 0:
            full_eval = _evaluate_single_cohort_df(
                merged_full.sort_values("StudyInstanceUID").reset_index(drop=True),
                model_ids=full_models,
                thresholds_map=thresholds_map,
                cluster_col="split_cluster_id",
                n_boot=n_boot,
                seed=seed,
            )

    # Secondary Cluster-Collapsed Common Cohort (max over split_cluster_id visits)
    agg_spec: Dict[str, Any] = {"exam_cancer": "max"}
    for m in model_ids:
        agg_spec[f"{m}_score"] = "max"
        agg_spec[f"{m}_has_crop_fallback"] = "max"
    collapsed_df = merged_common.groupby("split_cluster_id", as_index=False).agg(agg_spec)
    collapsed_eval = _evaluate_single_cohort_df(
        collapsed_df.sort_values("split_cluster_id").reset_index(drop=True),
        model_ids=model_ids,
        thresholds_map=thresholds_map,
        cluster_col="split_cluster_id",
        n_boot=n_boot,
        seed=seed,
    )

    shift_decomp = {
        m: compute_rsna_to_nlbs_shift_decomposition(
            rsna_case_dfs[m], rsna_summary[m], merged_common, common_eval["models"][m], m
        )
        for m in model_ids
    }

    report = {
        "study_id": "paper1_external_validation_6model",
        "stratum_a_rsna_supervised": [m for m in model_ids if MODEL_STRATUM[m] == "A_rsna_supervised"],
        "stratum_b_pretrained_direct": [m for m in model_ids if MODEL_STRATUM[m] == "B_pretrained_direct"],
        "rsna_source_eval": rsna_summary,
        "nlbs_common_4view_primary": common_eval,
        "nlbs_full_cohort_secondary": full_eval,
        "nlbs_cluster_collapsed_secondary": collapsed_eval,
        "shift_vs_discrimination": shift_decomp,
    }
    (out_dir / "six_model_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "six_model_summary.md").write_text(_format_markdown_report(report), encoding="utf-8")
    return report


def _format_markdown_report(report: Dict[str, Any]) -> str:
    c4 = report["nlbs_common_4view_primary"]
    lines = [
        "# Paper 1 — Six-Model External Validation Summary",
        "",
        f"Primary paired 4-view NLBS cohort: `N = {c4['n_units']}` studies, `{c4['n_clusters']}` clusters, `{c4['n_positive']}` cancers.",
        "",
        "## Primary Common 4-View NLBS Results (Stratum A vs. Stratum B)",
        "",
        "| Stratum | Model | AUROC | @98% Sens | @98% Dismissal | @95% Sens | @95% Dismissal | Fallback Studies |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for stratum_key, label in (("stratum_a_rsna_supervised", "A (RSNA-Supervised)"), ("stratum_b_pretrained_direct", "B (Pretrained-Direct)")):
        for m in report[stratum_key]:
            md = c4["models"][m]
            t98 = md["target_98"]
            t95 = md["target_95"]
            fb = md["fallback_audit"]["n_studies_with_fallback"]
            lines.append(
                f"| {label} | `{m}` | {md['auroc']:.4f} | {t98['triage_recall']*100:.2f}% ({t98['missed_cancers']} missed) | "
                f"{t98['dismissal_rate']*100:.2f}% | {t95['triage_recall']*100:.2f}% ({t95['missed_cancers']} missed) | "
                f"{t95['dismissal_rate']*100:.2f}% | {fb} |"
            )
    return "\n".join(lines) + "\n"
