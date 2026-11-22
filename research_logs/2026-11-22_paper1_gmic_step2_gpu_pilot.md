# Research Log: Paper 1 — GMIC Step 2 L40S GPU Pilot & Fallback/View Audit

**Date**: 2026-11-22  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Benchmark Evaluation & Model Verification  
**Milestone Scope**: `scripts/run_paper1_gmic_modal.py`, `reports/paper1/gmic_step2_gpu_pilot.json`, `reports/paper1/gmic_step2_gpu_pilot_report.md`  
**Status**: VERIFIED COMPLETE (SAFE TO PROCEED WITHIN $2 GPU BUDGET)  

---

## 1. Research Objective & Context
Validate GMIC (`sample_model_1.p`) Step 2 GPU inference end-to-end on `hasinishrak2015` using a 400-image L40S pilot (`200` RSNA + `200` NLBS from immutable Step 1 cache `/cache/paper1_gmic_uint16_v1/`) without running full inference or accessing outcomes, and audit all `80` RSNA fallbacks and `5` supplemental RSNA views.

---

## 2. Pilot Evidence & Parity Verification (`ap-YYXH9MfyVOBIXfb5IPctzQ`)
- **NVMe-Staged L40S Pipeline:** Added `_stage_pngs_nvme` (`ThreadPoolExecutor(16)`) + parallel `ThreadPoolExecutor(8)` PNG decode/z-score + batched (`batch_size=8`) FP32 GMIC forward pass.
- **L40S `sample_data` Exam 0 Parity:** `0_L-CC` (`0.13576/0.00812` vs `0.1356/0.0081`), `0_R-CC` (`0.89243/0.32523` vs `0.8929/0.3259`), `0_L-MLO` (`0.23641/0.03340` vs `0.2368/0.0335`), `0_R-MLO` (`0.95082/0.18122` vs `0.9509/0.1812`).
- **Cached vs. Raw Online DICOM Parity (`10` images):** Batch-1 cached PNG vs raw DICOM online preprocessing identical (`<= 1e-6`); `batch=8` cuDNN accumulation max diff `0.001931` (`602126376 R-AT`).
- **`80` RSNA Fallbacks Audit (`36` patients, `0` NLBS):** `51` `MONOCHROME1` + `29` `MONOCHROME2` (`29` L-MLO, `26` R-MLO, `13` R-CC, `12` L-CC); `80/80` valid nonzero `uint16` PNGs (`nonzero_frac` `0.0032..0.8765`), zero Step 1 error.
- **`5` Supplemental Views Audit:** `1760744211` (`L-ML->L-MLO`), `602126376` (`R-AT->R-CC`), `917413880` (`L-AT->L-CC`), `771225790` (`L-LM->L-MLO`), `1473114297` (`R-LM->R-MLO`); `0` fallbacks, all valid (`nonzero_frac` `0.5384..0.9036`).

---

## 3. Speed, Memory & Cost Receipts
- **Throughput (`400` images, `batch=8`):** `infer_e2e_sec = 12.30s` (`32.52 img/s` e2e; `gpu_only = 197.47 img/s`), `peak_vram_gb = 1.511 GB`, `host_rss_gb = 6.167 GB`.
- **Cost:** Measured pilot infer-only `$0.00807` (`$0.01737` full container); **projected full `37,970`-image L40S run = `$0.771 USD` (`within_2usd_budget = true`)**.
