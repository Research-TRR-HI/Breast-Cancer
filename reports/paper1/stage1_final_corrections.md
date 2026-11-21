# Paper 1 — Stage 1 Final Critical Corrections & Execution Readiness Audit

**Date:** 2026-11-21  
**Modal Execution Profile:** `hasinishrak2015` (holds both `rsna-mammography-vol` and `nlbs-mammography-vol`)  
**Decision:** **GO** — All four pre-launch corrections and the six-model statistical pipeline verified cleanly on CPU and NVIDIA L40S.

---

## 1. Summary of Pre-Launch Corrections

### Issue 1 — ResNet-50 AMP Gradient Unscaling & Bit-Exact Checkpoint Resume
* **Defect Fixed (`scripts/run_paper1_resnet50_modal.py`, `src/paper1/resnet50_model.py`):**
  * Previously called `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)` before `scaler.unscale_(opt)`, clipping FP16-scaled gradients (`~65536x`) and corrupting updates.
  * Extracted canonical `amp_train_step` calling **`scaler.unscale_(optimizer)` strictly before `clip_grad_norm_(..., 5.0)`**, logging per-epoch finite unscaled gradient norms and final `scaler.get_scale()`.
  * Added `save_resnet50_checkpoint` and `load_resnet50_checkpoint` with atomic `.pt.tmp -> .pt` fsynced writes preserving `model`, `optimizer`, `scheduler`, `scaler`, `python_rng_state`, `numpy_rng_state`, `torch_cpu_rng_state`, `torch_cuda_rng_states`, `sampler.get_state()`, and `aug_rng.bit_generator.state`.
* **Verification:**
  * CPU test `test_7_resnet50_amp_unscale_clip_and_resume_parity` asserts call order `["scale", "unscale", "clip", "step", "update"]` and bit-exact weight parity between a 4-epoch continuous run and an interrupted/resumed run (`ep=2 -> 3..4`).
  * Live L40S smoke run (`ap-vMdXP2h5eFu2wqzAOGyAYs`) observed initial overflow detection on batch 1 (`grad_norm = Infinity`), clean dynamic backoff (`65536 -> 32768 -> 16384.0`), and finite unscaled norms `[4.3176, 5.3117, 3.9838]` with monotonic loss decay `[0.6908, 0.6841, 0.6469, 0.6189]`.

### Issue 2 — DINOv2 NVMe Staging, Resumable Chunking & Timeout Margin
* **Defect Fixed (`scripts/run_paper1_dinov2_modal.py`):**
  * Replaced slow FUSE Volume PNG reads (`5.11 img/s`, projected `4.44 hrs > 3,600s` timeout) with local NVMe `/tmp` pre-staging via `ThreadPoolExecutor(max_workers=16)` (`ephemeral_disk=524288`, `timeout=7200`).
  * Added resumable `4,096`-image sharded chunks (`chunk_XXX.pt` with atomic fsync replace + `RSNA_VOL.commit()`), strict ID-order, shape `[N, 384]`, finiteness verification, and direct `modal volume get` retrieval paths.
* **Measured L40S Benchmark (`ap-acCim6bfGllCdJjTKtPDv6`, `512` Stage-1 RSNA PNGs):**
  * NVMe stage: `2.895s` (`512` files) | CPU load/preprocess: `2.118s` | L40S extract: `1.11s` | Peak VRAM: `2.396 GB`
  * End-to-end throughput: **`74.2 img/s`** (**14.5x speedup** vs. `5.11 img/s`)
  * Full projection (`81,694` images = `54,706` RSNA + `26,988` NLBS): **`~18.4 min` total** (`~12.3 min` RSNA + `~6.1 min` NLBS, vs. `7,200s` timeout = `9.7x` safety margin), estimated cost **`~$0.72 USD`**.

### Issue 3 — NYU Four-View & GMIC Crop-Fallback Root Cause & Non-Warping Fix
* **Root Cause Diagnosed (`ap-D6Q1uTvVprqm2TIIc9tmlg` on Modal CPU):**
  * Inspected all `32` images (`4` RSNA + `4` NLBS 4-view exams). Both fallbacks came exclusively from **RSNA `patient_id = 71`** (`R-CC` and `R-MLO`; `4740 x 3540`, `MONOCHROME1`).
  * After `MONOCHROME1` inversion and border subtraction (`border_median = 484.97` / `607.09`), foreground tissue occupied only `7.19%` (`R-CC`) and `5.87%` (`R-MLO`) of the large plate; upstream `nyukat/breast_cancer_classifier` `crop_img_from_largest_connected(..., iterations=100)` eroded the thin breast silhouette to all-zeros, raising `ValueError: attempt to get argmax of an empty sequence` at `crop_mammogram.py:59`.
  * Disabling erosion naively (`erode_dialate=False`) triggered `>300s` timeouts (`ap-eX8AZcUDyd1WBvoxbwyulN`) inside `nyukat`'s `for i in range(num_labels)` Python loop over thousands of background pixels.
* **Scientific Fix (`src/paper1/nyu_fourview_adapter.py`, `src/paper1/gmic_adapter.py`, `scripts/run_paper1_{nyu,gmic}_modal.py`, `studies/paper1/analysis_plan.md`):**
  * Eliminated spatial-warping `cv2.resize(...)` from all fallback branches. On empty-mask `Exception`, both adapters preserve **`1:1` native physical pixel resolution** and call `get_image_optimal_window_info` centered at foreground center of mass (`native_scale_optimal_window_no_resize`), matching NYU's physical-scale `augment_and_normalize_image` pad/crop semantics.
  * Live verification (`ap-vMdXP2h5eFu2wqzAOGyAYs`): Non-fallback exams `#1–#3` stayed identical to 5 decimal places (`0.42713`, `0.10142`, `0.06492`), while `patient_id = 71` moved from warped `0.05123` to physically consistent **`0.09617`**.
  * Every fallback view is now logged (`crop_error`, `photometric`, `raw_shape`, `view_slot`, `case_id`), saved to `{model}_{split}_fallback_log.json`, and audited before external evaluation in `src/paper1/six_model_eval.py` via **`fallback_audit.excluded_sensitivity_check`** (primary paired cohort + fallback-excluded sensitivity).

### Issue 4 — Complete Six-Model Evaluation & Paired Statistical Pipeline
* Implemented [six_model_eval.py](file:///d:/Breast%20Cancer%20Research/src/paper1/six_model_eval.py) and CLI [evaluate_paper1_six_models.py](file:///d:/Breast%20Cancer%20Research/scripts/evaluate_paper1_six_models.py):
  * Preserves strict separation between **Stratum A (`ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`)** and **Stratum B (`gmic_pretrained`, `nyu_fourview_pretrained`)**.
  * Enforces pre-join SHA-256 locking (`lock_nlbs_model_predictions` + `verify_and_load_locked_model`), RSNA-only `@98%` and `@95%` threshold calibration, 2-fold OOF independent source-test metrics, `B = 2,000` `split_cluster_id` bootstrap CIs, exact two-sided McNemar tests, KS score-drift decomposition (`RSNA -> NLBS`), Primary Common 4-View (`6,184`), Secondary Full (`6,359`), and Cluster-Collapsed (`5,868`) evaluations, fallback-excluded sensitivity checks, and pairwise Jaccard + UpSet missed-cancer overlap.

---

## 2. Updated Verification Summary & Recalculated Full-Run Costs

* **Local CPU Test Suites (`32/32 OK`, executed after all final code edits):**
  * `tests/test_stage_b_smoke.py`: `20/20 OK` (`5.153s`)
  * `tests/test_stage_c_quarantine_and_eval.py`: `3/3 OK` (`0.170s`)
  * `tests/test_paper1_models_and_gate02.py`: `9/9 OK` (`48.782s`, covering Gate 0.2 quarantine, `GMIC` & `NYU` official `sample_data` parity, DINOv2 CPU training/locking, ResNet-50 AMP unscale+clip & bit-exact resume, full 6-model paired/tamper eval, and native-scale crop fallback).
* **Preserve Intact Verified State:** `datasets/rsna/splits_seed42.csv`, `reports/stage_b/*`, `reports/stage_c/*`, `datasets/nlbs/exam_manifest.csv` (never exposed pre-lock), and `AnchorShift_Research_Context.md` remain completely unmodified.

| Step & Model | Active Profile | Peak VRAM | Updated L40S + Staging Cost Projection |
| :--- | :--- | :--- | :--- |
| 1. `gmic_pretrained` (`37,970` DICOMs) | `hasinishrak2015` | `0.292 GB` | **`$0.77`** (`0.172` L40S hrs + `5.98` CPU hrs) |
| 2. `nyu_fourview_pretrained` (`8,566` exams) | `hasinishrak2015` | `2.410 GB` | **`$8.06`** (`3.259` L40S hrs + `6.08` CPU hrs) |
| 3. `dinov2_rsna` (`81,694` Stage-1 PNGs, NVMe) | `hasinishrak2015` | `2.396 GB` | **`~$0.72`** (`74.2 img/s` -> `~18.4 min` total; CPU head `$0`) |
| 4. `resnet50_rsna` (`10` ep + infer, NVMe) | `hasinishrak2015` | `26.244 GB` | **`~$3.60`** (`$3.25` 10-ep train + `~$0.35` infer) |
| **Total All 4 New Models** | `hasinishrak2015` | — | **`~$13.15 USD`** *(well inside single `$30` profile cap)* |

---

## 3. Exact Execution Sequence (`GMIC -> NYU -> DINOv2 -> ResNet-50`) & Final Decision

**Decision: GO**

Execute strictly in the verified sequence below once authorized:

```powershell
$env:MODAL_PROFILE="hasinishrak2015"

# Step 1: GMIC Pretrained Model 1 (~$0.77)
modal run scripts/run_paper1_gmic_modal.py --split rsna_val_calib --num-shards 8
modal run scripts/run_paper1_gmic_modal.py --split nlbs --num-shards 8

# Step 2: NYU Four-View Image-Only CNN (~$8.06)
modal run scripts/run_paper1_nyu_modal.py --split rsna_val_calib --num-shards 8
modal run scripts/run_paper1_nyu_modal.py --split nlbs_4view --num-shards 12

# Step 3: DINOv2 ViT-S/14 NVMe Embedding Extraction (~$0.72) + CPU Head Training ($0)
modal run scripts/run_paper1_dinov2_modal.py --stage extract
modal volume get rsna-mammography-vol /cache/paper1_dinov2_vits14/rsna_embeddings_fp32.pt reports/paper1/dinov2_cache/
modal volume get rsna-mammography-vol /cache/paper1_dinov2_vits14/nlbs_embeddings_fp32.pt reports/paper1/dinov2_cache/
python scripts/train_paper1_dinov2_head.py --rsna-emb reports/paper1/dinov2_cache/rsna_embeddings_fp32.pt --nlbs-emb reports/paper1/dinov2_cache/nlbs_embeddings_fp32.pt --out-dir reports/paper1/dinov2_head

# Step 4: ResNet-50 (IMAGENET1K_V2) 10-Epoch RSNA Training & Frozen Inference (~$3.60)
modal run scripts/run_paper1_resnet50_modal.py --stage train --epochs 10 --seed 42
modal run scripts/run_paper1_resnet50_modal.py --stage infer --seed 42

# Step 5: Unified Pre-Join Prediction Locking & Paired 6-Model Statistical Evaluation (CPU, $0)
python scripts/evaluate_paper1_six_models.py --config-json <six_model_paths.json> --out-dir reports/paper1/six_model_final
```
