"""Build the strictly label-free NLBS Stage C inference manifest and verify frozen thresholds."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.stage_c_eval import (
    build_label_free_inference_manifest,
    verify_frozen_thresholds_against_stage_b,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label-free Stage C NLBS inference manifest.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("datasets/nlbs/manifest.csv"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("datasets/nlbs/inference_manifest_stage_c.csv"),
    )
    parser.add_argument(
        "--ce-eval-json",
        type=Path,
        default=Path("reports/stage_b/rsna_ce_eval_results.json"),
    )
    parser.add_argument(
        "--trust-eval-json",
        type=Path,
        default=Path("reports/stage_b/rsna_trust_eval_results.json"),
    )
    args = parser.parse_args()

    thresholds = verify_frozen_thresholds_against_stage_b(args.ce_eval_json, args.trust_eval_json)
    summary = build_label_free_inference_manifest(args.source_manifest, args.output_manifest)
    summary["verified_frozen_thresholds"] = thresholds
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
