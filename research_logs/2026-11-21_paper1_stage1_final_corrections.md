# Research Log: Paper 1 — Stage 1 Final Critical Corrections & Six-Model Evaluation Verification

**Date**: 2026-11-21  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Mixed-Precision Training Hygiene, Cloud IO Optimization & Preprocessing Audit  
**Milestone Scope**: `src/paper1/*`, `scripts/run_paper1_*`, `scripts/evaluate_paper1_six_models.py`, `reports/paper1/stage1_final_corrections.md`  
**Status**: VERIFIED COMPLETE (STAGE 1 FINAL READINESS = GO)  

---

## 1. Research Objective & Context
Resolve three pre-launch engineering/scientific bugs identified after `de14b44` (`ResNet-50` AMP gradient clipping order + RNG resume, `DINOv2` cold FUSE IO timeout risk, and `NYU` 2/32 crop fallbacks) plus build and test the complete six-model paired statistical analysis pipeline prior to running full Paper 1 Modal experiments.

---

## 2. Forensic Findings & Corrections
1. **ResNet-50 AMP Unscale + RNG/Sampler Resume (`src/paper1/resnet50_model.py`, `scripts/run_paper1_resnet50_modal.py`):**
   - Implemented `amp_train_step` executing `scaler.scale(loss).backward() -> scaler.unscale_(optimizer) -> clip_grad_norm_(..., 5.0) -> scaler.step(optimizer) -> scaler.update()`.
   - Implemented `save_resnet50_checkpoint` and `load_resnet50_checkpoint` saving `python`, `numpy`, `torch` CPU/CUDA RNGs, `BalancedBatchSampler.get_state()`, and `aug_rng.bit_generator.state`.
   - Verified on CPU (`test_7` bit-exact 2+2 resume parity) and live L40S (`ap-vMdXP2h5eFu2wqzAOGyAYs`: initial overflow `[Infinity]` properly skipped & rescaled `-> 16384.0`, subsequent steps `[4.3176, 5.3117, 3.9838]`, losses `[0.6908 -> 0.6189]`).
2. **DINOv2 NVMe Staging & Resumable Chunks (`scripts/run_paper1_dinov2_modal.py`):**
   - Staged PNGs onto `/tmp` NVMe (`ThreadPoolExecutor(16)`, `ephemeral_disk=524288`, `timeout=7200`) and chunked extraction into atomic `4,096`-image `.pt` shards with shape/finiteness/ID-order checks and `modal volume get` instructions.
   - Live L40S `512`-image benchmark (`ap-acCim6bfGllCdJjTKtPDv6`) improved throughput from `5.11 img/s` to **`74.2 img/s` (14.5x speedup; ~18.4 min / ~$0.72 USD total across all 81,694 images)**.
3. **NYU / GMIC Crop-Fallback Root Cause & Non-Warping Fix (`ap-D6Q1uTvVprqm2TIIc9tmlg`):**
   - Both `2/32` smoke fallbacks occurred on **RSNA `patient_id = 71`** (`R-CC`, `R-MLO`; `4740 x 3540`, `MONOCHROME1`). Small thin breast tissue (`5.87%`–`7.19%` nonzero after border subtraction) eroded to zero under upstream `binary_erosion(..., iterations=100)`, throwing `ValueError: attempt to get argmax of an empty sequence`.
   - Disabling erosion (`erode_dialate=False`) triggered `>300s` timeouts in `nyukat`'s `for i in range(num_labels)` Python loop (`ap-eX8AZcUDyd1WBvoxbwyulN`).
   - Replaced `cv2.resize` fallbacks in `nyu_fourview_adapter.py` and `gmic_adapter.py` with `1:1` native-scale foreground COM optimal-window placement (`native_scale_optimal_window_no_resize`), logged exact errors per view, and added automatic **`fallback_audit.excluded_sensitivity_check`** in `src/paper1/six_model_eval.py`.
4. **Six-Model Evaluation & Paired Statistics (`src/paper1/six_model_eval.py`, `scripts/evaluate_paper1_six_models.py`):**
   - End-to-end verified on synthetic 6-model cohorts (`test_8`): Stratum A vs B tables, pre-join SHA-256 locking & tamper abort, Common 4-View (`6,184`) / Full (`6,359`) / Cluster-Collapsed (`5,868`), `B=2,000` paired cluster bootstrap CIs, exact McNemar, KS shift decomposition, and Jaccard/UpSet missed-cancer overlap.

---

## 3. Quality Assurance Verification
- **Local CPU Tests (`32/32 OK` post-final-edit):** `test_stage_b_smoke.py` (`20/20`), `test_stage_c_quarantine_and_eval.py` (`3/3`), `test_paper1_models_and_gate02.py` (`9/9`).
- **Live Modal L40S Runs (`hasinishrak2015`):** `ap-acCim6bfGllCdJjTKtPDv6` (DINOv2 NVMe benchmark `512` imgs) & `ap-vMdXP2h5eFu2wqzAOGyAYs` (4-model smoke/bench). Combined full run cost recalculated to **`~$13.15 USD`** (`GMIC $0.77 + NYU $8.06 + DINOv2 $0.72 + ResNet-50 $3.60`).
