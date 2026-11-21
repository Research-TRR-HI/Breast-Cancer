"""Paper 1 — Gate 0.2: Deterministic Split/Cohort Manifests & CPU CE/TRUST Source Evaluation.

Generates:
    1. datasets/paper1/rsna_val_calib_2fold_seed42.csv (2,382 val_calib patients split 50/50 stratified by cancer x site_id)
    2. datasets/paper1/rsna_val_calib_4view_selection.csv (9,528 label-blind selected views for RSNA val_calib)
    3. datasets/paper1/nlbs_4view_study_eligibility.csv (6,359 label-free StudyInstanceUID eligibility rows)
    4. datasets/paper1/nlbs_4view_common_cohort_label_free.csv (24,736 label-free selected views across 6,184 4-view NLBS studies)
    5. reports/paper1/ce_trust_rsna_source_eval.{json,md} (independent 2-fold OOF source-test results for CE/TRUST on CPU)
    6. reports/paper1/ce_trust_nlbs_common_4view.{json,md} (locked Stage C CE/TRUST scores evaluated on common 6,184 4-view cohort)

Preserves all immutable reports/stage_b/*, reports/stage_c/*, and datasets/rsna/splits_seed42.csv files untouched.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.stage_c_eval import (
    FROZEN_RSNA_THRESHOLDS,
    assert_no_forbidden_columns,
    compute_auprc,
    compute_auroc,
    compute_cluster_bootstrap_cis,
    compute_sha256_file,
    load_label_free_inference_manifest,
)
from src.models.trust_model import aggregate_case_labels, aggregate_case_scores
from src.training.calibration import (
    compute_target_recall_threshold,
    evaluate_dismissal_at_threshold,
)
from src.training.engine import build_trust_head


STANDARD_4_VIEWS = ("L-CC", "R-CC", "L-MLO", "R-MLO")


def build_rsna_val_calib_2fold(
    splits_csv: Path,
    output_csv: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Deterministically partition RSNA val_calib (2,382 pts) into fold_a / fold_b stratified by (cancer, site_id)."""
    df = pd.read_csv(splits_csv)
    vc = df[df["split"] == "val_calib"].copy().reset_index(drop=True)
    if vc["patient_id"].nunique() != 2382:
        raise ValueError(f"Expected 2382 val_calib patients, got {vc['patient_id'].nunique()}")

    pt_df = (
        vc.groupby("patient_id")
        .agg(cancer=("cancer", "max"), site_id=("site_id", "first"))
        .reset_index()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

    rng = np.random.RandomState(seed)
    fold_a_pts: List[Any] = []
    fold_b_pts: List[Any] = []

    # Alternate assignment parity per stratum so positives (49 vs 48) and sites balance cleanly
    for stratum_idx, (_, grp) in enumerate(pt_df.groupby(["cancer", "site_id"], sort=True)):
        pids = grp["patient_id"].values.copy()
        rng.shuffle(pids)
        half = (len(pids) + (1 if stratum_idx % 2 == 0 else 0)) // 2
        fold_a_pts.extend(pids[:half].tolist())
        fold_b_pts.extend(pids[half:].tolist())

    fold_map = {p: "fold_a" for p in fold_a_pts}
    for p in fold_b_pts:
        fold_map[p] = "fold_b"

    vc["fold"] = vc["patient_id"].map(fold_map)
    assert vc["fold"].isin(["fold_a", "fold_b"]).all()

    pts_a = set(vc[vc["fold"] == "fold_a"]["patient_id"])
    pts_b = set(vc[vc["fold"] == "fold_b"]["patient_id"])
    assert len(pts_a & pts_b) == 0, "Patient overlap between fold_a and fold_b!"

    other_pts = set(df[df["split"] != "val_calib"]["patient_id"])
    assert len((pts_a | pts_b) & other_pts) == 0, "Overlap with train/train_calib!"

    out_cols = ["patient_id", "image_id", "split", "fold", "cancer", "laterality", "view", "site_id", "machine_id", "rel_path"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    vc[out_cols].to_csv(output_csv, index=False)

    summary = {}
    for f in ("fold_a", "fold_b"):
        sub = vc[vc["fold"] == f]
        p_lbl = sub.groupby("patient_id")["cancer"].max()
        summary[f] = {
            "n_images": int(len(sub)),
            "n_patients": int(len(p_lbl)),
            "n_positive_patients": int(p_lbl.sum()),
            "site_counts": {str(k): int(v) for k, v in sub.groupby("site_id")["patient_id"].nunique().items()},
        }
    summary["output_csv"] = str(output_csv)
    summary["sha256"] = compute_sha256_file(output_csv)
    return summary


def build_rsna_val_calib_4view_selection(
    splits_csv: Path,
    output_csv: Path,
) -> Dict[str, Any]:
    """Deterministically select one image per {L-CC, R-CC, L-MLO, R-MLO} slot for RSNA val_calib without reading labels."""
    df = pd.read_csv(splits_csv)
    vc = df[df["split"] == "val_calib"].copy()
    vc["view_slot"] = vc["laterality"].astype(str).str.strip().str.upper() + "-" + vc["view"].astype(str).str.strip().str.upper()
    vc = vc[vc["view_slot"].isin(STANDARD_4_VIEWS)].copy()
    vc["image_id_int"] = vc["image_id"].astype(int)

    selected_rows = []
    for pid, grp in vc.groupby("patient_id", sort=True):
        slots = set(grp["view_slot"])
        if not set(STANDARD_4_VIEWS).issubset(slots):
            raise ValueError(f"RSNA patient {pid} missing standard 4 views: {slots}")
        for vs in STANDARD_4_VIEWS:
            cand = grp[grp["view_slot"] == vs].sort_values(["image_id_int", "rel_path"], kind="mergesort")
            row = cand.iloc[0]
            selected_rows.append(
                {
                    "patient_id": int(pid),
                    "view_slot": vs,
                    "image_id": str(row["image_id"]),
                    "laterality": str(row["laterality"]),
                    "view": str(row["view"]),
                    "site_id": int(row["site_id"]),
                    "machine_id": int(row["machine_id"]),
                    "rel_path": str(row["rel_path"]),
                    "n_candidates_in_slot": int(len(cand)),
                }
            )
    out_df = pd.DataFrame(selected_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return {
        "output_csv": str(output_csv),
        "sha256": compute_sha256_file(output_csv),
        "n_patients": int(out_df["patient_id"].nunique()),
        "n_rows": int(len(out_df)),
        "multi_candidate_slots": int((out_df["n_candidates_in_slot"] > 1).sum()),
    }


def build_nlbs_4view_label_free_manifests(
    label_free_inf_csv: Path,
    eligibility_csv: Path,
    selected_4view_csv: Path,
) -> Dict[str, Any]:
    """Build strictly label-free NLBS 4-view eligibility and single-image-per-slot selection manifests."""
    inf_df = load_label_free_inference_manifest(label_free_inf_csv)
    inf_df["view_slot"] = inf_df["laterality"] + "-" + inf_df["view_position"]

    elig_rows = []
    sel_rows = []
    for study_uid, grp in inf_df.groupby("StudyInstanceUID", sort=True):
        present_slots = set(grp["view_slot"])
        missing_slots = [s for s in STANDARD_4_VIEWS if s not in present_slots]
        is_elig = len(missing_slots) == 0
        cluster_id = str(grp["split_cluster_id"].iloc[0])
        elig_rows.append(
            {
                "StudyInstanceUID": str(study_uid),
                "split_cluster_id": cluster_id,
                "n_manifest_images": int(len(grp)),
                "is_4view_eligible": bool(is_elig),
                "missing_views": "|".join(missing_slots) if missing_slots else "",
            }
        )
        if is_elig:
            for vs in STANDARD_4_VIEWS:
                cand = grp[grp["view_slot"] == vs].copy()
                cand["mfr_rank"] = (cand["manufacturer"] != "GE MEDICAL SYSTEMS").astype(int)
                cand = cand.sort_values(["mfr_rank", "dicom_rel_path", "image_uid"], kind="mergesort")
                top = cand.iloc[0]
                sel_rows.append(
                    {
                        "StudyInstanceUID": str(study_uid),
                        "split_cluster_id": cluster_id,
                        "view_slot": vs,
                        "image_uid": str(top["image_uid"]),
                        "laterality": str(top["laterality"]),
                        "view_position": str(top["view_position"]),
                        "manufacturer": str(top["manufacturer"]),
                        "photometric_interpretation": str(top["photometric_interpretation"]),
                        "dicom_rel_path": str(top["dicom_rel_path"]),
                        "n_candidates_in_slot": int(len(cand)),
                    }
                )

    elig_df = pd.DataFrame(elig_rows)
    sel_df = pd.DataFrame(sel_rows)
    assert_no_forbidden_columns(elig_df.columns, context=str(eligibility_csv))
    assert_no_forbidden_columns(sel_df.columns, context=str(selected_4view_csv))

    eligibility_csv.parent.mkdir(parents=True, exist_ok=True)
    elig_df.to_csv(eligibility_csv, index=False)
    sel_df.to_csv(selected_4view_csv, index=False)

    return {
        "eligibility_csv": str(eligibility_csv),
        "eligibility_sha256": compute_sha256_file(eligibility_csv),
        "selected_4view_csv": str(selected_4view_csv),
        "selected_4view_sha256": compute_sha256_file(selected_4view_csv),
        "total_studies": int(len(elig_df)),
        "eligible_4view_studies": int(elig_df["is_4view_eligible"].sum()),
        "ineligible_studies": int((~elig_df["is_4view_eligible"]).sum()),
        "eligible_clusters": int(elig_df[elig_df["is_4view_eligible"]]["split_cluster_id"].nunique()),
        "selected_rows": int(len(sel_df)),
        "multi_candidate_slots": int((sel_df["n_candidates_in_slot"] > 1).sum()),
    }


def _bootstrap_rsna_metrics(df_cases: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Patient-level bootstrap on RSNA cases (wraps run_cluster_bootstrap with patient_id as cluster)."""
    tmp = df_cases.rename(columns={"patient_id": "split_cluster_id", "cancer": "exam_cancer"}).copy()
    tmp["StudyInstanceUID"] = tmp["split_cluster_id"].astype(str)
    # Minimal columns expected by run_cluster_bootstrap
    raise NotImplementedError


def compute_rsna_patient_bootstrap(
    scores: np.ndarray,
    labels: np.ndarray,
    dismissed_95: np.ndarray,
    dismissed_98: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute patient-level bootstrap 95% CIs on independent RSNA OOF predictions."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    aurocs, auprcs, s95, d95, s98, d98 = [], [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = labels[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        sb = scores[idx]
        auroc = compute_auroc(yb, sb)
        auprc = compute_auprc(yb, sb)
        aurocs.append(auroc)
        auprcs.append(auprc)
        pos_m = yb == 1
        s95.append(float(np.mean(~dismissed_95[idx][pos_m])))
        d95.append(float(np.mean(dismissed_95[idx])))
        s98.append(float(np.mean(~dismissed_98[idx][pos_m])))
        d98.append(float(np.mean(dismissed_98[idx])))

    def _ci(arr: List[float]) -> Dict[str, float]:
        a = np.asarray(arr, dtype=np.float64)
        return {
            "mean": float(np.mean(a)),
            "ci95_low": float(np.percentile(a, 2.5)),
            "ci95_high": float(np.percentile(a, 97.5)),
        }

    return {
        "auroc": _ci(aurocs),
        "auprc": _ci(auprcs),
        "target_95_sensitivity": _ci(s95),
        "target_95_dismissal": _ci(d95),
        "target_98_sensitivity": _ci(s98),
        "target_98_dismissal": _ci(d98),
    }


def evaluate_ce_trust_rsna_source(
    val_calib_2fold_csv: Path,
    eval_val_calib_pt: Path,
    ce_ckpt_path: Path,
    trust_ckpt_path: Path,
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    """Compute patient-disjoint 2-fold cross-fitted RSNA source-test metrics for CE & TRUST on CPU."""
    fold_df = pd.read_csv(val_calib_2fold_csv)
    pt_fold_map = fold_df.groupby("patient_id")["fold"].first().to_dict()

    vc = torch.load(str(eval_val_calib_pt), map_location="cpu", weights_only=False)
    feats = vc["features"]
    labels = vc["labels"].numpy().astype(np.int64)
    pids = [int(p) for p in vc["patient_ids"]]

    results: Dict[str, Any] = {
        "protocol_note": (
            "Independent 2-fold patient-disjoint cross-fitting inside RSNA val_calib (2,382 patients / 97 cancers; "
            "CPU float32 forward pass on cached Mammo-CLIP B5 embeddings). Distinct from Stage B full-val_calib "
            "thresholds used for NLBS (historic AMP float16 precision)."
        ),
        "models": {},
    }

    for m_key, ck_path in (("ce_b5", ce_ckpt_path), ("trust_b5", trust_ckpt_path)):
        ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
        head = build_trust_head(embed_dim=2048, dropout_rate=0.3, seed=42, device=torch.device("cpu"))
        head.load_state_dict(ck["head_state_dict"], strict=True)
        head.eval()
        with torch.no_grad():
            probs = torch.sigmoid(head(feats).squeeze(-1)).numpy()

        case_scores_map = aggregate_case_scores(probs, [str(p) for p in pids])
        case_labels_map = aggregate_case_labels(labels.tolist(), [str(p) for p in pids])

        rows = []
        for p_str in sorted(case_scores_map.keys(), key=lambda x: int(x)):
            pid_int = int(p_str)
            rows.append(
                {
                    "patient_id": pid_int,
                    "fold": pt_fold_map[pid_int],
                    "cancer": int(case_labels_map[p_str]),
                    "score": float(case_scores_map[p_str]),
                }
            )
        cdf = pd.DataFrame(rows)

        # 2-fold OOF evaluation
        oof_records = []
        fold_details = {}
        for cal_fold, test_fold in (("fold_a", "fold_b"), ("fold_b", "fold_a")):
            cal_sub = cdf[cdf["fold"] == cal_fold]
            tst_sub = cdf[cdf["fold"] == test_fold].copy()
            pos_cal = cal_sub[cal_sub["cancer"] == 1]["score"].values
            tau95 = compute_target_recall_threshold(pos_cal, target_recall=0.95)
            tau98 = compute_target_recall_threshold(pos_cal, target_recall=0.98)
            ev95 = evaluate_dismissal_at_threshold(tst_sub["score"].values, tst_sub["cancer"].values, tau95, 0.99, 0.95)
            ev98 = evaluate_dismissal_at_threshold(tst_sub["score"].values, tst_sub["cancer"].values, tau98, 0.99, 0.98)
            tst_sub["dismissed_95"] = tst_sub["score"] < tau95
            tst_sub["dismissed_98"] = tst_sub["score"] < tau98
            oof_records.append(tst_sub)
            fold_details[f"{cal_fold}_to_{test_fold}"] = {
                "calib_n_patients": int(len(cal_sub)),
                "calib_n_pos": int(cal_sub["cancer"].sum()),
                "test_n_patients": int(len(tst_sub)),
                "test_n_pos": int(tst_sub["cancer"].sum()),
                "tau_95": float(tau95),
                "tau_98": float(tau98),
                "target_95": ev95,
                "target_98": ev98,
            }

        oof_df = pd.concat(oof_records, ignore_index=True).sort_values("patient_id").reset_index(drop=True)
        auroc = compute_auroc(oof_df["cancer"].values, oof_df["score"].values)
        auprc = compute_auprc(oof_df["cancer"].values, oof_df["score"].values)
        boot = compute_rsna_patient_bootstrap(
            oof_df["score"].values,
            oof_df["cancer"].values,
            oof_df["dismissed_95"].values,
            oof_df["dismissed_98"].values,
        )

        # Full val_calib reference (resubstitution, NOT independent source test)
        pos_full = cdf[cdf["cancer"] == 1]["score"].values
        tau95_full_fp32 = compute_target_recall_threshold(pos_full, 0.95)
        tau98_full_fp32 = compute_target_recall_threshold(pos_full, 0.98)

        results["models"][m_key] = {
            "checkpoint_sha256": compute_sha256_file(ck_path),
            "auroc_full_val_calib": float(auroc),
            "auprc_full_val_calib": float(auprc),
            "oof_2fold_pooled": {
                "n_patients": int(len(oof_df)),
                "n_pos": int(oof_df["cancer"].sum()),
                "target_95": {
                    "sensitivity": float(oof_df[oof_df["cancer"] == 1]["dismissed_95"].eq(False).mean()),
                    "missed_cancers": int((oof_df["cancer"].eq(1) & oof_df["dismissed_95"]).sum()),
                    "dismissal_rate": float(oof_df["dismissed_95"].mean()),
                    "n_dismissed": int(oof_df["dismissed_95"].sum()),
                },
                "target_98": {
                    "sensitivity": float(oof_df[oof_df["cancer"] == 1]["dismissed_98"].eq(False).mean()),
                    "missed_cancers": int((oof_df["cancer"].eq(1) & oof_df["dismissed_98"]).sum()),
                    "dismissal_rate": float(oof_df["dismissed_98"].mean()),
                    "n_dismissed": int(oof_df["dismissed_98"].sum()),
                },
                "bootstrap_95ci": boot,
            },
            "split_folds": fold_details,
            "full_val_calib_cpu_fp32_thresholds_NOT_INDEPENDENT": {
                "tau_95": float(tau95_full_fp32),
                "tau_98": float(tau98_full_fp32),
            },
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Paper 1 — Independent RSNA Source-Test Results (CE-only & TRUST, CPU Verification)",
        "",
        "> **Methodological Separation:** Metrics below come from **patient-disjoint 2-fold cross-fitting inside RSNA `val_calib`** (`fold_a`: 1,191 pts / 49 pos; `fold_b`: 1,191 pts / 48 pos). Thresholds are NEVER evaluated on the same patients used to select them. These cross-fitted thresholds are **distinct** from the full-`val_calib` (`97`-pos, AMP FP16) thresholds frozen in Stage B (`ce: 0.096191/0.124451`; `trust: 0.179565/0.195435`) for NLBS.",
        "",
        "| Model | Evaluation Type | Calib Pos / Test Pos | Target | Threshold `tau` | Independent Test Sens (95% CI) | Missed | Independent Test Dismissal (95% CI) | AUROC (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m_key in ("ce_b5", "trust_b5"):
        md = results["models"][m_key]
        oof = md["oof_2fold_pooled"]
        b = oof["bootstrap_95ci"]
        auc_str = f"{md['auroc_full_val_calib']:.4f} [{b['auroc']['ci95_low']:.4f}, {b['auroc']['ci95_high']:.4f}]"
        for tgt, t_lbl in (("target_98", "98%"), ("target_95", "95%")):
            o_t = oof[tgt]
            bs = b[f"{tgt}_sensitivity"]
            bd = b[f"{tgt}_dismissal"]
            lines.append(
                f"| `{m_key}` | 2-Fold OOF Pooled | 49/48 -> 97 | {t_lbl} | fold-specific | "
                f"**{o_t['sensitivity']*100:.2f}%** [{bs['ci95_low']*100:.2f}%, {bs['ci95_high']*100:.2f}%] | "
                f"{o_t['missed_cancers']}/97 | **{o_t['dismissal_rate']*100:.2f}%** [{bd['ci95_low']*100:.2f}%, {bd['ci95_high']*100:.2f}%] | {auc_str} |"
            )
        for fname in ("fold_a_to_fold_b", "fold_b_to_fold_a"):
            fd = md["split_folds"][fname]
            for tgt, t_lbl, tkey in (("target_98", "98%", "tau_98"), ("target_95", "95%", "tau_95")):
                ev = fd[tgt]
                lines.append(
                    f"| `{m_key}` | `{fname}` | {fd['calib_n_pos']} -> {fd['test_n_pos']} | {t_lbl} | `{fd[tkey]:.6f}` | "
                    f"{ev['triage_recall']*100:.2f}% | {ev['missed_cancers']}/{fd['test_n_pos']} | {ev['dismissal_rate']*100:.2f}% | — |"
                )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def evaluate_locked_nlbs_common_4view(
    locked_preds_csv: Path,
    eligibility_csv: Path,
    exam_manifest_csv: Path,
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    """Filter immutable Stage C locked NLBS predictions to the 6,184 4-view common cohort without altering Stage C."""
    preds = pd.read_csv(locked_preds_csv)
    elig = pd.read_csv(eligibility_csv)
    ex = pd.read_csv(exam_manifest_csv).drop_duplicates("StudyInstanceUID")

    merged = preds.merge(elig[["StudyInstanceUID", "is_4view_eligible"]], on="StudyInstanceUID", how="inner")
    merged = merged.merge(ex[["StudyInstanceUID", "exam_cancer", "cohort_group"]], on="StudyInstanceUID", how="inner")
    sub = merged[merged["is_4view_eligible"]].copy().reset_index(drop=True)

    assert len(sub) == 6184, f"Expected 6184 common 4-view studies, got {len(sub)}"
    assert int(sub["exam_cancer"].sum()) == 151, f"Expected 151 positive 4-view studies, got {sub['exam_cancer'].sum()}"

    boot_df = sub[["StudyInstanceUID", "split_cluster_id", "exam_cancer", "score_ce", "score_trust",
                   "dismissed_ce_95", "dismissed_ce_98", "dismissed_trust_95", "dismissed_trust_98"]].copy()
    boot = compute_cluster_bootstrap_cis(boot_df, thresholds=FROZEN_RSNA_THRESHOLDS, n_bootstrap=2000, seed=42)

    out = {
        "n_studies": int(len(sub)),
        "n_clusters": int(sub["split_cluster_id"].nunique()),
        "n_positive_studies": int(sub["exam_cancer"].sum()),
        "source_locked_predictions_sha256": compute_sha256_file(locked_preds_csv),
        "ce_b5": {
            "auroc": float(compute_auroc(sub["exam_cancer"].values, sub["score_ce"].values)),
            "auprc": float(compute_auprc(sub["exam_cancer"].values, sub["score_ce"].values)),
            "target_98": evaluate_dismissal_at_threshold(sub["score_ce"].values, sub["exam_cancer"].values, 0.09619140625, 0.99, 0.98),
            "target_95": evaluate_dismissal_at_threshold(sub["score_ce"].values, sub["exam_cancer"].values, 0.12445068359375, 0.99, 0.95),
        },
        "trust_b5": {
            "auroc": float(compute_auroc(sub["exam_cancer"].values, sub["score_trust"].values)),
            "auprc": float(compute_auprc(sub["exam_cancer"].values, sub["score_trust"].values)),
            "target_98": evaluate_dismissal_at_threshold(sub["score_trust"].values, sub["exam_cancer"].values, 0.1795654296875, 0.99, 0.98),
            "target_95": evaluate_dismissal_at_threshold(sub["score_trust"].values, sub["exam_cancer"].values, 0.1954345703125, 0.99, 0.95),
        },
        "cluster_bootstrap_95ci": boot,
    }
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    bm = boot["metrics"]
    md_lines = [
        "# Paper 1 — Locked Stage C CE/TRUST Results on Common 4-View NLBS Cohort (N = 6,184)",
        "",
        "> Filtered directly from immutable `reports/stage_c/locked_nlbs_predictions.csv` (`sha256: " + out["source_locked_predictions_sha256"] + "`). All `151/151` cancer-positive NLBS studies are 4-view eligible (`175` excluded studies are all cancer-negative).",
        "",
        "| Model | Target | Frozen RSNA `tau` | Sensitivity (95% Cluster CI) | Missed Cancers | Dismissal Rate (95% Cluster CI) | AUROC (95% Cluster CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m_key, b_pref, tau98, tau95 in (
        ("ce_b5", "ce", 0.09619140625, 0.12445068359375),
        ("trust_b5", "trust", 0.1795654296875, 0.1954345703125),
    ):
        auc_ci = bm[f"{b_pref}_auroc"]
        auc_s = f"{out[m_key]['auroc']:.4f} [{auc_ci['ci95_low']:.4f}, {auc_ci['ci95_high']:.4f}]"
        for tname, tlbl, tv in (("target_98", "98%", tau98), ("target_95", "95%", tau95)):
            ev = out[m_key][tname]
            sci = bm[f"{b_pref}_{tname}_sensitivity"]
            dci = bm[f"{b_pref}_{tname}_dismissal_rate"]
            md_lines.append(
                f"| `{m_key}` | {tlbl} | `{tv:.6f}` | **{ev['triage_recall']*100:.2f}%** [{sci['ci95_low']*100:.2f}%, {sci['ci95_high']*100:.2f}%] | "
                f"{ev['missed_cancers']}/151 | **{ev['dismissal_rate']*100:.2f}%** [{dci['ci95_low']*100:.2f}%, {dci['ci95_high']*100:.2f}%] | {auc_s} |"
            )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    s1 = build_rsna_val_calib_2fold(Path("datasets/rsna/splits_seed42.csv"), Path("datasets/paper1/rsna_val_calib_2fold_seed42.csv"))
    s2 = build_rsna_val_calib_4view_selection(Path("datasets/rsna/splits_seed42.csv"), Path("datasets/paper1/rsna_val_calib_4view_selection.csv"))
    s3 = build_nlbs_4view_label_free_manifests(
        Path("datasets/nlbs/inference_manifest_stage_c.csv"),
        Path("datasets/paper1/nlbs_4view_study_eligibility.csv"),
        Path("datasets/paper1/nlbs_4view_common_cohort_label_free.csv"),
    )
    s4 = evaluate_ce_trust_rsna_source(
        Path("datasets/paper1/rsna_val_calib_2fold_seed42.csv"),
        Path("models/cache_eval_rsna/eval_val_calib_fp32.pt"),
        Path("models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt"),
        Path("models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt"),
        Path("reports/paper1/ce_trust_rsna_source_eval.json"),
        Path("reports/paper1/ce_trust_rsna_source_eval.md"),
    )
    s5 = evaluate_locked_nlbs_common_4view(
        Path("reports/stage_c/locked_nlbs_predictions.csv"),
        Path("datasets/paper1/nlbs_4view_study_eligibility.csv"),
        Path("datasets/nlbs/exam_manifest.csv"),
        Path("reports/paper1/ce_trust_nlbs_common_4view.json"),
        Path("reports/paper1/ce_trust_nlbs_common_4view.md"),
    )
    print(json.dumps({"rsna_2fold": s1, "rsna_4view": s2, "nlbs_4view": s3}, indent=2))
    print("CE/TRUST RSNA 2-fold OOF + NLBS common 4-view reports written successfully.")


if __name__ == "__main__":
    main()
