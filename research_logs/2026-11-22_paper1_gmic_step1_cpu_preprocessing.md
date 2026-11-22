# Research Log: Paper 1 — GMIC Step 1 CPU-Only Preprocessing & Verification

**Date**: 2026-11-22  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: DICOM Preprocessing & Reusable Volume Caching  
**Milestone Scope**: `src/paper1/gmic_adapter.py`, `scripts/preprocess_paper1_gmic_cpu.py`, `reports/paper1/gmic_step1_preprocessing_report.md`  
**Status**: VERIFIED COMPLETE (STEP 2 READY)  

---

## 1. Research Objective & Context
Execute GMIC Step 1 strictly on Modal CPU containers (`hasinishrak2015`, zero GPU) to generate reusable `(2944, 1920)` `uint16` PNG caches across `10,982` RSNA `val_calib` and `26,988` NLBS mammograms before any GPU prediction or threshold evaluation.

---

## 2. Forensic Diagnostics & Bug Caught During Execution
1. **Bit-Exact Lossless PNG Cache Formula:** Verified `external/GMIC/src/data_loading/augmentations.py` slices/zero-pads `[0..65535]` integers cleanly before `standard_normalize_single_image`. Caching unnormalized `(2944, 1920)` `uint16` PNGs + applying `standard_normalize_gmic_uint16` reproduces online `preprocess_uint16_for_nyukat` with `max_abs_diff <= 1e-6` locally (`test_10`) and **`0.0` empirically across all 20 checked Modal images**.
2. **Supplemental RSNA View Position Crash:** Initial build hit `KeyError: ('Unknown view', 'L-ML')` because RSNA `val_calib` includes 5 non-standard views (`L-ML`, `L-AT`, `R-AT`, `L-LM`, `R-LM`) outside `VIEWS.LIST`. Added `_canonical_nyukat_view_slot` (`ML/LM/LMO/SIO -> {side}-MLO`, `AT/XCCL/CV -> {side}-CC`) ensuring proper right-breast mirroring and padding geometry.

---

## 3. Benchmarks, Execution Spend & Cache Verification
- **200-Image CPU Benchmark (`ap-MuRdTTfrd9slnLiv2Ai6PQ`):** `135.549s` (`1.475 img/s`), cost **`$0.00951`**, projected full `$1.805` (< `$3.00` gate), `0` GPU, `max_parity_abs_diff = 0.0`.
- **16-Shard Resumable CPU Build (`ap-i9HV9RJvsFrAoyBBZ7mlk5`):**
  - `rsna_val_calib`: `10,982 / 10,982` (`10,282` new + `700` skipped on resume), `80` fallbacks (`0.7285%`), `62.058 GB`, cost **`$0.8280`**.
  - `nlbs`: `26,988 / 26,988`, `0` fallbacks (`0.0%`), `97.291 GB`, cost **`$0.7564`**.
  - **Build Spend:** **`$1.5844 USD`** (`~$1.66 USD` total incl. bench + partial run + verify).
- **Full Integrity & Parity Audit (`ap-iYfqLVbUg2s26E9DosJ9WG`):** All `37,970 / 37,970` PNG headers verified `(2944, 1920)` 16-bit grayscale, `20 / 20` sampled DICOM parity checks **`max_parity_abs_diff = 0.0`**.
- **Step 2 Wiring:** Updated `scripts/run_paper1_gmic_modal.py` to consume `/cache/paper1_gmic_uint16_v1/` directly (not yet executed).
