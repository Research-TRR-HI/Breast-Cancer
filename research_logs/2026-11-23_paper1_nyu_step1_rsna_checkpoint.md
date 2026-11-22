# Research Log — 2026-11-23: [Paper 1] NYU Four-View Step 1 — Full RSNA CPU Cache Verified & NLBS Cost/Storage Stop-Gate Pause

- **Date:** 2026-11-23 (Modal UTC jobs `2026-11-22T20:26Z–21:24Z`)
- **Author / Track:** Paper 1 — Multi-Model External Validation Benchmark (`nyu_fourview_pretrained`)
- **Modal Workspace:** `hasinishrak2015` (CPU ONLY; `0` GPUs allocated)
- **Dataset Scope Completed:** `rsna_val_calib` (`9,528 / 9,528` images, `2,382 / 2,382` four-view exams)
- **Dataset Scope Held at Stop Gate:** `nlbs_4view` (`24,736` images — **NOT started**; stopped before NLBS launch per cost/storage monitoring guardrail)
- **Label Quarantine:** Zero Canadian (`cancer`) outcome labels accessed (`nlbs_4view_common_cohort_label_free.csv` strictly label-free)

---

## 1. Objectives & Pre-Run Checks
- Execute CPU-only (`2 vCPU / 8 GiB`, `12` shards/split, `compression=1`, `0` GPUs) NYU Step 1 lossless `uint16` cache build starting with `rsna_val_calib` as an intermediate billing, RSS, throughput, and storage checkpoint before firing `nlbs_4view` (`24,736` images).
- Preserve all locked GMIC (`paper1_gmic_uint16_v1`, `159.371 GiB`), CE, TRUST, and raw DICOM/Stage-1 caches untouched.

## 2. Implementation Update & Unit Verification
- Updated [`scripts/preprocess_paper1_nyu_cpu.py`](../scripts/preprocess_paper1_nyu_cpu.py):
  - Added per-split local receipt export (`nyu_step1_{split}_fallback_log.json`, `nyu_step1_{split}_verify_summary.json` / `build_summary.json`), `by_view_slot` & `by_photometric` summary counts, and `pixel_spot_n=100` decoded `u16_sha256` verification inside `verify_nyu_cpu_cache`.
  - After initial build `ap-1dMjmBD9jw2X6b150L9X4u` completed all `12` RSNA shard conversions cleanly (`792/792` and `796/796`) and timed out inside single-container serial `verify_nyu_cpu_cache` (`1800s` hitting sequential Modal Volume FUSE latency across `9,528` PNGs), added parallel **`verify_nyu_cpu_shard`** (`cpu=1.0, memory=2048, timeout=1800`, 12 concurrent workers verifying 100% PNG SHA-256s) + fast JSONL/completeness/`pixel_spot_n=100` consolidation in `verify_nyu_cpu_cache`.
- Verified `python -m py_compile scripts/preprocess_paper1_nyu_cpu.py src/paper1/nyu_fourview_adapter.py` and **`tests/test_paper1_nyu_step1_cpu.py` (`5/5` PASS)** prior to live `verify` execution.

## 3. Executed Runs & Empirical Verification (`rsna_val_calib`)
1. **RSNA 12-Shard CPU Build (`ap-1dMjmBD9jw2X6b150L9X4u`, `hasinishrak2015`):**
   - All `12` workers completed `796/796` (`s00–s05`) and `792/792` (`s06–s11`) = `9,528` lossless PNGs written and committed to `rsna-mammography-vol:/cache/paper1_nyu_uint16_v1/rsna_val_calib/` before serial verify hit `FunctionTimeoutError: 1800s`.
   - Settled billing (`ap-1dMjmBD…`): `$0.1984` Memory + `$0.3375` CPU = **`$0.5359 USD`** (`~$0.063` from the aborted verify container -> **`~$0.473 USD`** net compute shards).
2. **RSNA 12-Shard Parallel SHA-256 Verification (`ap-yaZkL5TEdpCYQTcEqIWQfl`, `hasinishrak2015`):**
   - Verified all `12` shards (`79.11s–437.56s`, `0` rejected PNGs, `$0.0574` worker compute; settled app cost `$0.0184` Mem + `$0.0543` CPU = **`$0.0727 USD`**).
   - `verify_nyu_cpu_cache` verified exact `9,528` image coverage across `2,382` four-view exams (`L-CC: 2382`, `L-MLO: 2382`, `R-CC: 2382`, `R-MLO: 2382`), `MONOCHROME1: 1808`, `MONOCHROME2: 7720`, `101` decoded `u16_sha256` spot checks, `66` cropping fallbacks across `34` exams (`native_scale_optimal_window_no_resize`, zero resizing), and `manifest_sha256: bde6ae3fa12cff0c0d121de2b1344ddc775f1020b3645670b4d62ca00db8f866`.
   - Pulled and inspected `reports/paper1/nyu_step1_rsna_val_calib_manifest.csv` (`2,532,220` bytes): `9,528` rows, `2,382` exams, **`69.779 GiB`** (`mean 7.499 MiB`, `median 6.779 MiB`), `66` fallbacks.

## 4. Cost & Storage Gate Analysis (Why Execution Stopped Before `nlbs_4view`)
- **RSNA Compute Overrun (`2.2x–2.5x`):** `$0.546` net (`$0.6086` gross) vs `$0.2464` projected -> projects `~USD 1.42–1.55` for `nlbs_4view` (`2.596x`) and `~USD 2.03–2.16` total vs `$0.87` (`<= $1.13`) estimate.
- **RSNA Storage Overrun (`3.58x`):** `69.779 GiB` actual vs `19.48 GiB` pre-build estimate (`+50.30 GiB`). Combined NYU storage updates from `137.00 GiB` to **`~187.30–250 GiB`** (`~USD 28–37.5/month` / `~USD 0.94–1.25/day`).
- **Remaining Starter Credit Runway:** `modal billing summary` after RSNA reports `Metered Cost: 46.09`, `Volumes: 26.36`, `Free Storage: -25.34` (exhausted), `Credits: -20.66` -> **`USD 9.34` remaining** until Dec 1. Adding `nlbs_4view` compute + 8 days of `~187–250 GiB` storage risks consuming the remaining `USD 9.34` credit balance before Dec 1.
- **NLBS Verify Timeout Guard Needed Before Resuming:** `verify_nyu_cpu_shard` peaked at `437.56s` for `792` RSNA PNGs (`~0.55s/file`); `2,061` NLBS PNGs per shard would project `~1,135s+`, so `verify_nyu_cpu_shard` should be bumped from `timeout=1800` to `timeout=3600` (or threaded) prior to NLBS launch.

## 5. Artifacts Generated
- [`reports/paper1/nyu_step1_rsna_checkpoint_report.md`](../reports/paper1/nyu_step1_rsna_checkpoint_report.md)
- [`reports/paper1/nyu_step1_rsna_val_calib_verify_summary.json`](../reports/paper1/nyu_step1_rsna_val_calib_verify_summary.json)
- [`reports/paper1/nyu_step1_rsna_val_calib_fallback_log.json`](../reports/paper1/nyu_step1_rsna_val_calib_fallback_log.json)
- [`reports/paper1/nyu_step1_rsna_val_calib_manifest.csv`](../reports/paper1/nyu_step1_rsna_val_calib_manifest.csv)
