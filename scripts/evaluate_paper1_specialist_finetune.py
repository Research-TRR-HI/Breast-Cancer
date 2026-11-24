"""Paper 1 Supplementary Exploratory — CPU-Only Calibration, Pre-Join Locking & NLBS Evaluation.

Strictly CPU-only ($0.00 cloud compute):
1. `--phase calibrate_rsna`: verifies locked RSNA `val_calib` predictions, computes
   2-fold OOF source eval + full `val_calib` frozen `tau_98` / `tau_95`, and writes
   `reports/paper1/expanded/exp1/<model_id>_rsna_calibration{_lock}.json`.
2. `--phase lock_nlbs`: verifies RSNA calibration lock + label-free NLBS case scores,
   freezes dismissal flags (`score < tau`), and SHA-256 locks NLBS predictions BEFORE
   any outcome label file is opened.
3. `--phase eval_nlbs`: verifies SHA-256 locks, joins `datasets/nlbs/exam_manifest.csv`,
   and writes `reports/paper1/expanded/exp1/<model_id>_nlbs_evaluation.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paper1.specialist_eval import (
    SUPPLEMENTARY_MODEL_IDS,
    calibrate_finetuned_rsna_thresholds,
    evaluate_finetuned_nlbs_against_labels,
    lock_finetuned_nlbs_predictions,
)

MODEL_PREFIX = {
    "gmic_rsna_finetuned": "gmic",
    "nyu_fourview_rsna_finetuned": "nyu",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only RSNA calibration, NLBS lock & eval for fine-tuned specialists.")
    parser.add_argument("--model", required=True, choices=SUPPLEMENTARY_MODEL_IDS)
    parser.add_argument("--phase", required=True, choices=("calibrate_rsna", "lock_nlbs", "eval_nlbs"))
    parser.add_argument(
        "--locked-pred-dir",
        type=Path,
        default=Path("reports/paper1/expanded/exp1/predictions_locked"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/paper1/expanded/exp1"),
    )
    parser.add_argument("--expected-checkpoint-sha256", type=str, default="")
    parser.add_argument("--rsna-splits", type=Path, default=Path("datasets/rsna/splits_seed42.csv"))
    parser.add_argument("--rsna-folds", type=Path, default=Path("datasets/paper1/rsna_val_calib_2fold_seed42.csv"))
    parser.add_argument("--exam-manifest", type=Path, default=Path("datasets/nlbs/exam_manifest.csv"))
    parser.add_argument(
        "--common-4view-csv",
        type=Path,
        default=Path("datasets/paper1/nlbs_4view_common_cohort_label_free.csv"),
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prefix = MODEL_PREFIX[args.model]
    exp_sha = args.expected_checkpoint_sha256.strip() or None
    calib_lock_path = args.out_dir / f"{args.model}_rsna_calibration_lock.json"

    if args.phase == "calibrate_rsna":
        res = calibrate_finetuned_rsna_thresholds(
            model_id=args.model,
            prefix=prefix,
            locked_pred_dir=args.locked_pred_dir,
            rsna_splits_csv=args.rsna_splits,
            rsna_fold_csv=args.rsna_folds,
            out_dir=args.out_dir,
            expected_checkpoint_sha256=exp_sha,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        print(json.dumps(res, indent=2))
        return

    if args.phase == "lock_nlbs":
        nlbs_split = "nlbs" if args.model == "gmic_rsna_finetuned" else "nlbs_4view"
        nlbs_case_csv = args.locked_pred_dir / f"{prefix}_{nlbs_split}_case_scores.csv"
        res = lock_finetuned_nlbs_predictions(
            model_id=args.model,
            nlbs_case_scores_csv=nlbs_case_csv,
            calib_lock_path=calib_lock_path,
            out_dir=args.locked_pred_dir,
        )
        print(json.dumps(res, indent=2))
        return

    if args.phase == "eval_nlbs":
        res = evaluate_finetuned_nlbs_against_labels(
            model_id=args.model,
            calib_lock_path=calib_lock_path,
            nlbs_lock_dir=args.locked_pred_dir,
            exam_manifest_csv=args.exam_manifest,
            common_4view_csv=args.common_4view_csv,
            out_dir=args.out_dir,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        print(json.dumps(res, indent=2))
        return


if __name__ == "__main__":
    main()
