# Research Log — 2026-11-24: [Paper 1] NYU Four-View Step 2 L40S GPU Pilot, Batch Parity & Cost Analysis

- **Date:** 2026-11-24 (`12:24 +06:00` / `06:24 UTC`)
- **Author / Track:** Paper 1 — Multi-Model External Validation Benchmark (`nyu_fourview_pretrained`)
- **Modal Workspace / App:** `hasinishrak2015` / [`ap-1WtcQA2ZWaNFBzKQy5CFAU`](https://modal.com/apps/hasinishrak2015/main/ap-1WtcQA2ZWaNFBzKQy5CFAU) (`1x L40S`, `6 vCPU`, `24 GiB`)
- **Quarantine & Guardrails:** Label-free manifests only (`0` cancer labels accessed), FP32 `eval()`, `10` TTA (`seed=0`), no training/fine-tuning, no cache deletions, pilot capped at `$0.18` (`< $0.20`).

---

## 1. Implementation & CPU Unit Verification
- Updated [`src/paper1/nyu_fourview_adapter.py`](../src/paper1/nyu_fourview_adapter.py) (`prepare_nyu_exam_tta_tensors`, `score_nyu_fourview_exam_batch`, `validate_nyu_shard_payload`, `finalize_and_lock_nyu_predictions`, `validate_nyu_step2_cli_and_guard`), [`scripts/run_paper1_nyu_modal.py`](../scripts/run_paper1_nyu_modal.py) (parallel NVMe staging, `u16_sha256` load check, bounded CPU prefetch, batched L40S inference, `$0.18` pilot cost guard + `--approve-full-infer` CLI gate), and added [`tests/test_paper1_nyu_step2.py`](../tests/test_paper1_nyu_step2.py).
- Verified `python -m py_compile`, `tests/test_paper1_nyu_step1_cpu.py` (`5/5` PASS, `52.342s`), and `tests/test_paper1_nyu_step2.py` (`3/3` PASS, `95.621s` / `65.588s`) prior to L40S launch.

## 2. Live L40S Pilot Measurements (`400` Cached PNGs / `100` Exams)
- **Sample:** `50` RSNA (`all 34 fallback exams + 16 standard`) + `50` NLBS (`200 + 200 = 400` images, `10` TTA, `seed=0`, `batch_exams=4`).
- **Throughput & Memory:**
  - `nvme_stage_sec=1.903s`, `decode_tta_sec=10.686s`, `gpu_infer_sec=41.639s`, `e2e_pilot_sec=54.331s`, `total_container_sec=114.385s`.
  - **`batch_exams=4`: `1.84 exams/sec` (`7.36 imgs/sec` E2E; `2.40 exams/sec` GPU-only; `9.309 GiB` VRAM; `23.616 GiB` host RSS)** vs original `bs=1` (`0.80 exams/sec`) and `bs=8` probe (`1.58 exams/sec`, `18.578 GiB` VRAM).
- **Parity (`max_abs_diff`):**
  - Batched `bs=4` vs unbatched `bs=1` `score_nyu_fourview_exam` (`12` exams incl. `8` RSNA fallback exams): **`4.306e-06`** exam score (`6.199e-06` all-4 classes).
  - Batched `bs=4` cached PNG vs online raw DICOM (`6` exams incl. `12943`, `14995`, `1625` fallbacks): **`4.306e-06`**.
  - `bs=8` vs `bs=4` (`16` exams): `4.2e-06`.

## 3. Billing & Full-Run Feasibility vs `$1.50` Buffer
- **Pilot Settled Cost:** **`USD 0.0695`** (`L40S $0.05092 + CPU $0.01361 + Mem $0.00501`, credits `-22.41 -> -22.48` = **`USD 7.52` remaining** until `2026-12-01` UTC).
- **Full `8,566`-Exam Projection (`batch_exams=4`, `8` shards):** **`1.3425` hrs / `USD 3.2566`** (`~USD 3.35–3.47` at recommended `8 vCPU / 32 GiB RAM` to eliminate the `23.616 GiB` near-`24 GiB` RSS pressure).
- **Storage + Buffer Impact (`USD 0.485/day` excess * `~6.73` Nov days = `~USD 3.26`):**
  - Running both splits (`~USD 3.35`) + holding `157.5 GiB` NYU cache through Nov 30 (`~USD 3.26`) leaves `~USD 0.91` (`< $1.50` buffer).
  - **Safe Options:** (A) Lock both splits then reclaim `157.506 GiB` cache (`~USD 3.62` buffer `>= $1.50`), OR (B) Zero cache deletion: run `rsna_val_calib` now (`~USD 0.96` -> `USD 3.30` buffer) and `nlbs_4view` (`~USD 2.40`) after Dec 1 reset.

## 4. Artifacts Generated
- [`reports/paper1/nyu_step2_gpu_pilot.json`](../reports/paper1/nyu_step2_gpu_pilot.json)
- [`reports/paper1/nyu_step2_gpu_pilot_report.md`](../reports/paper1/nyu_step2_gpu_pilot_report.md)
