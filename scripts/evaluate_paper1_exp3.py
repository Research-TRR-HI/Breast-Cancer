"""Paper 1 Exp 3 — CPU Evaluation CLI Across Multiple Unseen External Datasets.

Workflow:
    1. Optionally verifies & reuses completed NLBS predictions (`ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`)
       from `reports/paper1/six_model_final/locked_nlbs/` at $0.00 cost.
    2. Verifies SHA-256 locked predictions per `(dataset_id, model_id)` BEFORE joining any outcome file.
    3. Evaluates each dataset independently (`nlbs`, `imdb`, `cmmd`, `lumina`) with cluster-bootstrap CIs
       and strict clinical-role guardrails (NEVER pools screening NLBS with diagnostic CMMD/LUMINA).
    4. Writes `exp3_summary.json` and `exp3_summary.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paper1.external_datasets import select_common_4view_cohort, validate_inference_manifest
from src.paper1.universal_eval import (
    RSNA_SUPERVISED_MODELS,
    evaluate_external_dataset,
    generate_multi_dataset_report,
    reuse_completed_nlbs_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SHA-256 locked Exp 3 multi-dataset predictions.")
    parser.add_argument("--config-json", type=Path, required=True, help="JSON mapping dataset_id -> {inference_manifest, outcome_manifest, models}.")
    parser.add_argument("--lock-dir", type=Path, default=Path("reports/paper1/expanded/exp3/locked_preds"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/paper1/expanded/exp3/eval"))
    parser.add_argument("--reuse-nlbs-from", type=Path, default=None, help="Optional six_model_final/locked_nlbs dir to verify & import NLBS predictions.")
    parser.add_argument("--nlbs-exam-manifest", type=Path, default=Path("datasets/nlbs/exam_manifest.csv"))
    parser.add_argument("--cohort-mode", default="all", choices=["all", "full", "common_4view"])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.reuse_nlbs_from is not None:
        reuse_completed_nlbs_predictions(
            six_model_lock_dir=args.reuse_nlbs_from,
            nlbs_exam_manifest_csv=args.nlbs_exam_manifest,
            out_lock_dir=args.lock_dir,
        )

    cfg = json.loads(args.config_json.read_text(encoding="utf-8"))
    per_ds: Dict[str, Dict[str, Any]] = {}
    for ds_id, ds_cfg in cfg.items():
        inf_df = validate_inference_manifest(pd.read_csv(ds_cfg["inference_manifest"]), context=f"cli_{ds_id}_inf")
        c4v_df, _ = select_common_4view_cohort(inf_df)
        c4v_ids = sorted(c4v_df["exam_id"].unique().tolist())
        models = ds_cfg.get("models", list(RSNA_SUPERVISED_MODELS))
        ds_cohort_mode = ds_cfg.get("cohort_mode", args.cohort_mode)
        per_ds[ds_id] = evaluate_external_dataset(
            dataset_id=ds_id,
            lock_dir=args.lock_dir,
            outcome_manifest_csv=Path(ds_cfg["outcome_manifest"]),
            model_ids=models,
            common_4view_exam_ids=c4v_ids,
            cohort_mode=ds_cohort_mode,
            n_boot=args.n_boot,
            seed=args.seed,
        )

    summary = generate_multi_dataset_report(per_ds, args.out_dir)
    print(json.dumps({"evaluated_datasets": list(summary["datasets"].keys()), "pooled": summary["pooled_analysis_performed"]}, indent=2))


if __name__ == "__main__":
    main()
