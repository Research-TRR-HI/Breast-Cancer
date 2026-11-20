"""Sequential Orchestrator for AnchorShift Stage B (Gates 1-4 + Main Extraction & Training).

Runs the exact 7 mandatory Stage B commands in order, halting immediately if any step
fails or if Step 4's projected extraction cost exceeds --max-extract-usd.

IMPORTANT:
    - Do NOT run automatically without user intent; spends Modal credits.
    - Preview commands safely first with:
        python scripts/run_stage_b_pipeline.py --dry-run
"""

import argparse
from dataclasses import dataclass
import json
import os
import re
import subprocess
import sys
import time
from typing import List, Optional


@dataclass(frozen=True)
class PipelineStep:
    step_num: int
    name: str
    phase: str
    est_time: str
    cmd: List[str]


def build_pipeline_steps(
    gpu: str = "L40S",
    pilot_precision: str = "amp",
    steps_per_epoch: int = 625,
) -> List[PipelineStep]:
    return [
        PipelineStep(
            step_num=1,
            name="Real RSNA DICOM Preprocessing Audit",
            phase="Phase 1 - Gate 1",
            est_time="~1 min (CPU)",
            cmd=[
                "modal", "run", "scripts/cache_stage1_rsna_modal.py",
                "--audit-real", "--samples-per-view-lat", "4",
            ],
        ),
        PipelineStep(
            step_num=2,
            name="Build & Auto-Verify Full Stage-1 uint16 Cache (54,706 imgs)",
            phase="Phase 1 - Gate 2",
            est_time="~3.5-6.5 hrs (CPU)",
            cmd=[
                "modal", "run", "scripts/cache_stage1_rsna_modal.py",
                "--build", "--strict-segmentation",
            ],
        ),
        PipelineStep(
            step_num=3,
            name=f"1,200-Image L40S Online-vs-Cached Parity Pilot ({pilot_precision})",
            phase="Phase 1 - Gate 3",
            est_time="~17-20 min (amp)" if pilot_precision == "amp" else "~45-60 min (both)",
            cmd=[
                "modal", "run", "scripts/pilot_parity_modal.py",
                "--epochs", "20",
                "--steps-per-epoch", "12",
                "--micro-batch-size", "20",
                "--precision", pilot_precision,
                "--no-strict-segmentation",
                "--gpu", gpu,
            ],
        ),
        PipelineStep(
            step_num=4,
            name="5-Batch Extraction Throughput & Cost Benchmark",
            phase="Phase 1 - Gate 4",
            est_time="~1-2 min (L40S)",
            cmd=[
                "modal", "run", "scripts/train_trust_modal.py",
                "--benchmark-extract-batches", "5",
                "--epochs", "20",
                "--steps-per-epoch", str(steps_per_epoch),
                "--micro-batch-size", "20",
                "--precision", "amp",
                "--no-strict-segmentation",
                "--gpu", gpu,
            ],
        ),
        PipelineStep(
            step_num=5,
            name="Extract & Auto-Verify Full 1,000,000-Occurrence FP32 Cache",
            phase="Phase 2 - Cache Extract",
            est_time="~7.5+ hrs (L40S)",
            cmd=[
                "modal", "run", "scripts/train_trust_modal.py",
                "--extract-cache",
                "--epochs", "20",
                "--steps-per-epoch", str(steps_per_epoch),
                "--micro-batch-size", "20",
                "--precision", "amp",
                "--no-strict-segmentation",
                "--gpu", gpu,
            ],
        ),
        PipelineStep(
            step_num=6,
            name="MAIN TRAINING 1: CE-Only Baseline (20 epochs)",
            phase="Phase 2 - Main Training",
            est_time="< 2 min (L40S)",
            cmd=[
                "modal", "run", "scripts/train_trust_modal.py",
                "--baseline", "ce",
                "--epochs", "20",
                "--steps-per-epoch", str(steps_per_epoch),
                "--micro-batch-size", "80",
                "--precision", "amp",
                "--gpu", gpu,
                "--use-cached-features",
            ],
        ),
        PipelineStep(
            step_num=7,
            name="MAIN TRAINING 2: TRUST Closed-Loop Model (20 epochs)",
            phase="Phase 2 - Main Training",
            est_time="< 2 min (L40S)",
            cmd=[
                "modal", "run", "scripts/train_trust_modal.py",
                "--baseline", "trust",
                "--epochs", "20",
                "--steps-per-epoch", str(steps_per_epoch),
                "--micro-batch-size", "80",
                "--precision", "amp",
                "--gpu", gpu,
                "--use-cached-features",
            ],
        ),
    ]


def _extract_last_json_block(text: str) -> Optional[dict]:
    """Best-effort parse of the trailing JSON object printed by Modal entrypoints."""
    matches = list(re.finditer(r"\{[\s\S]*\}", text))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _check_gate(step: PipelineStep, output: str, max_extract_usd: float) -> None:
    payload = _extract_last_json_block(output)
    if payload is None:
        return

    if step.step_num == 1:
        fb = payload.get("fallback_count", 0)
        if fb != 0:
            raise RuntimeError(f"Gate 1 FAILED: audit-real reported fallback_count={fb} != 0.")
    elif step.step_num == 3:
        modes = payload.get("modes", {})
        for mname, mres in modes.items():
            if mres.get("status") != "PASS":
                raise RuntimeError(f"Gate 3 FAILED: parity pilot mode={mname} status={mres.get('status')}.")
    elif step.step_num == 4:
        est_usd = payload.get("projected_full_extract_usd_l40s_est")
        est_hrs = payload.get("projected_full_extract_hours")
        print(
            f"\n[GATE 4] Projected full extraction: {est_hrs} hrs | ~${est_usd} USD "
            f"(budget cap: ${max_extract_usd:.2f})",
            flush=True,
        )
        if est_usd is not None and float(est_usd) > float(max_extract_usd):
            raise RuntimeError(
                f"Gate 4 STOPPED pipeline: projected extraction cost ${est_usd} > cap ${max_extract_usd:.2f}."
            )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Sequential Stage B Modal pipeline runner.")
    parser.add_argument("--start-step", type=int, default=1, choices=range(1, 8))
    parser.add_argument("--end-step", type=int, default=7, choices=range(1, 8))
    parser.add_argument("--gpu", type=str, default="L40S")
    parser.add_argument("--pilot-precision", type=str, default="amp", choices=["amp", "both"])
    parser.add_argument("--steps-per-epoch", type=int, default=625)
    parser.add_argument("--max-extract-usd", type=float, default=26.0)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    if args.start_step > args.end_step:
        raise ValueError(f"--start-step ({args.start_step}) cannot exceed --end-step ({args.end_step}).")

    steps = [
        s for s in build_pipeline_steps(
            gpu=args.gpu,
            pilot_precision=args.pilot_precision,
            steps_per_epoch=args.steps_per_epoch,
        )
        if args.start_step <= s.step_num <= args.end_step
    ]

    print("=" * 78)
    print(f"AnchorShift Stage B Sequential Pipeline (Steps {args.start_step}..{args.end_step} | dry_run={args.dry_run})")
    print("=" * 78)

    for step in steps:
        cmd_str = " ".join(step.cmd)
        print(f"\n[Step {step.step_num}/7 | {step.phase}] {step.name} ({step.est_time})")
        print(f"  CMD: {cmd_str}", flush=True)
        if args.dry_run:
            continue

        t0 = time.perf_counter()
        child_env = os.environ.copy()
        child_env["PYTHONUTF8"] = "1"
        child_env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            step.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            bufsize=1,
        )
        lines: List[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        rc = proc.wait()
        elapsed_min = (time.perf_counter() - t0) / 60.0

        if rc != 0:
            print(f"\n[ERROR] Step {step.step_num} exited with code {rc} after {elapsed_min:.2f} min.", file=sys.stderr)
            print(f"Fix issue and resume with: python scripts/run_stage_b_pipeline.py --start-step {step.step_num}", file=sys.stderr)
            return rc

        _check_gate(step, "".join(lines), max_extract_usd=args.max_extract_usd)
        print(f"[OK] Step {step.step_num}/7 finished in {elapsed_min:.2f} min.", flush=True)

    print("\nPipeline finished cleanly!" if not args.dry_run else "\nDry-run complete (0 commands executed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
