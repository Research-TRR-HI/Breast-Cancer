"""Paper 1 — Six-Model Evaluation, Pre-Join Locking & Paired Analysis CLI.

Consumes raw image/exam score CSVs/JSONs from the 6 Paper 1 models, computes source-only
RSNA thresholds, cryptographically locks label-free NLBS predictions BEFORE joining outcomes,
and runs paired cluster-bootstrap, McNemar, KS shift, and Jaccard/UpSet missed-cancer analyses.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.trust_model import aggregate_case_labels, aggregate_case_scores
from src.paper1.six_model_eval import (
    ALL_SIX_MODELS,
    calibrate_rsna_thresholds_only,
    evaluate_six_models,
    lock_nlbs_model_predictions,
)


def _aggregate_rsna_image_scores(df_scores: pd.DataFrame, splits_csv: Path) -> pd.DataFrame:
    splits = pd.read_csv(splits_csv)
    vc = splits[splits["split"] == "val_calib"][["image_id", "patient_id", "cancer"]].copy()
    vc["image_id"] = vc["image_id"].astype(str)
    s_df = df_scores.copy()
    s_df["image_id"] = s_df["image_id"].astype(str)
    m = vc.merge(s_df[["image_id", "prob_cancer"]], on="image_id", how="inner")
    cs = aggregate_case_scores(m["prob_cancer"].values, m["patient_id"].astype(str).tolist())
    cl = aggregate_case_labels(m["cancer"].astype(int).tolist(), m["patient_id"].astype(str).tolist())
    return pd.DataFrame(
        [{"patient_id": int(p), "cancer": int(cl[p]), "score": float(cs[p])} for p in sorted(cs.keys(), key=int)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper 1 six-model locking and paired statistical evaluation.")
    parser.add_argument("--config-json", type=Path, required=True, help="JSON mapping model_id -> rsna/nlbs score file paths and sha256.")
    parser.add_argument("--fold-csv", type=Path, default=Path("datasets/paper1/rsna_val_calib_2fold_seed42.csv"))
    parser.add_argument("--splits-csv", type=Path, default=Path("datasets/rsna/splits_seed42.csv"))
    parser.add_argument("--exam-manifest", type=Path, default=Path("datasets/nlbs/exam_manifest.csv"))
    parser.add_argument("--common-4view", type=Path, default=Path("datasets/paper1/nlbs_4view_common_cohort_label_free.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/paper1/six_model_final"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
    lock_dir = args.out_dir / "locked_nlbs"
    rsna_cases: Dict[str, pd.DataFrame] = {}

    for m in ALL_SIX_MODELS:
        if m not in cfg:
            continue
        entry = cfg[m]
        r_df = pd.read_csv(entry["rsna_case_csv"]) if "rsna_case_csv" in entry else _aggregate_rsna_image_scores(pd.read_csv(entry["rsna_image_csv"]), args.splits_csv)
        rsna_cases[m] = r_df
        r_eval = calibrate_rsna_thresholds_only(r_df, fold_csv=args.fold_csv, n_boot=args.n_boot, seed=args.seed)
        n_df = pd.read_csv(entry["nlbs_exam_csv"])
        lock_nlbs_model_predictions(
            model_id=m,
            exam_scores_df=n_df,
            thresholds_full=r_eval["full_val_calib_thresholds"],
            model_sha256=str(entry.get("model_sha256", "unknown")),
            out_dir=lock_dir,
        )

    res = evaluate_six_models(
        locked_dir=lock_dir,
        rsna_case_dfs=rsna_cases,
        fold_csv=args.fold_csv,
        exam_manifest_csv=args.exam_manifest,
        common_4view_csv=args.common_4view,
        out_dir=args.out_dir,
        model_ids=[m for m in ALL_SIX_MODELS if m in rsna_cases],
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(json.dumps({"common_4view_models": list(res["nlbs_common_4view_primary"]["models"].keys())}, indent=2))


if __name__ == "__main__":
    main()
