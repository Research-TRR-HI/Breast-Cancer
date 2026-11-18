# AnchorShift Final Pre-Training Readiness Audit

**Audit Date:** 2026-11-18  
**Repository:** `d:\Breast Cancer Research` (`https://github.com/Research-TRR-HI/Breast-Cancer`)  
**Target Reference:** Shafie et al. (2026), *TRUST: Threshold-Recalibrated Uncertainty-Safe Training for Certified Dismissal in Breast Cancer Screening*, `arXiv:2609.00300v1`  
**Audit Verdict:** **CONDITIONAL GO** — All 9 engineering and scientific blockers (Critical Fixes A–I) are fixed and verified across **20/20 local CPU unit/integration tests (`0` failures, `0` skipped)**. Cloud execution remains **explicitly gated** behind the Stage-1 visual audit, Stage-1 full build/verification, Tesla T4 smoke test, 1,200-image L40S Online-vs-Cached Mammo-CLIP Parity Pilot (`fp32` + `amp`), and 5-batch end-to-end extraction throughput benchmark before launching full 1,000,000-occurrence extraction or 20-epoch RSNA training.

---

## 1. Executive Verdict (`CONDITIONAL GO`)

The AnchorShift RSNA training reproduction codebase (`src/` and `scripts/`) now implements the `arXiv:2609.00300v1` objective, preprocessing, independent balanced sampling, dynamic Clopper–Pearson threshold calibration, strict final-epoch selection, and strict NLBS quarantine with full deterministic parity between Online and Cached execution on CPU.

- **Local CPU Verification:** Complete (**20/20 passed**, `python tests/test_stage_b_smoke.py` & `python -m unittest discover -s tests -p "test_*.py" -v`).
- **Why CONDITIONAL GO (not full GO):** Cloud GPU behavior — specifically CUDA RNG save/resume determinism, real RSNA `L/R` x `CC/MLO` x `MONOCHROME1/2` segmentation across all 54,706 DICOMs, real Mammo-CLIP B5 (`852/852`) FP32/AMP parity across 20 epochs, and real L40S Stage-1 -> Stage-2 -> Encoder extraction throughput — cannot be claimed as verified from local synthetic CPU tests alone and **must pass the sequential manual Modal gates below**.

---

## 2. Critical Fixes A–I Summary Table

| Fix | Root Cause Identified | Exact Code Correction Implemented | Verification Status |
|---|---|---|---|
| **A. Deterministic Head Init Parity** | Constructing `MammoCLIPEncoder` before head init in Online mode shifted outer PyTorch RNG compared to Cached mode. | Created [`build_trust_head`](../src/training/engine.py#L18-L47) saving/restoring outer CPU & CUDA RNG states around head construction. | **PASSED (Local CPU):** `test_a4_head_initialization_invariant_to_prior_encoder_init` (`0.0` diff, outer RNG unchanged). |
| **B. Strict Versioned Checkpointing (`v2`)** | Old checkpoints lacked schema enforcement, full hyperparameter linkage, cache manifest SHA-256 binding, and strict RNG/scaler completeness checks. | Rewrote [`src/training/checkpointing.py`](../src/training/checkpointing.py) to enforce `CHECKPOINT_SCHEMA_VERSION = "anchorshift_trust_ckpt_v2"`, 22 `MANDATORY_CONFIG_KEYS`, 15 `MANDATORY_PAYLOAD_KEYS`, `os.fsync`, and strict rejection of missing/mismatched fields or RNG states. | **PASSED (Local CPU):** `test_g2` (exact resume parity) & `test_g3` (rejects legacy schema, missing keys/RNG, cache hash mismatch). **CUDA resume parity pending Modal GPU check.** |
| **C. Single Shared Production Engine** | Parity pilot and Modal training loop maintained separate forward/backward/clip/step logic. | Extracted [`src/training/engine.py`](../src/training/engine.py) (`run_effective_batch_step`, `seed_dropout_step_rng`); wired into BOTH [`src/training/parity_pilot.py`](../src/training/parity_pilot.py) and [`scripts/train_trust_modal.py`](../scripts/train_trust_modal.py). | **PASSED (Local CPU):** `test_g1` & `test_g2`; production Modal run pending. |
| **D. 1,200-Image Parity Pilot & Nonzero `tau_t`** | Legacy `100`-img `train_calib` subset (`96` neg) could mathematically never satisfy zero-miss 95% Clopper–Pearson `<= 1%` (minimum `N = 299`), forcing `tau_t = 0.0` throughout the pilot. | Added [`compute_dynamic_training_threshold_with_diagnostics`](../src/training/calibration.py#L70-L154) and [`sample_pilot_1200`](../src/training/parity_pilot.py#L51-L88) (`500` train `[21 pos, 479 neg]`, `500` train_calib `[20 pos, 480 neg >= 299]`, `200` val_calib `[8 pos, 192 neg]`, 0 patient leakage); updated [`scripts/pilot_parity_modal.py`](../scripts/pilot_parity_modal.py). | **PASSED (Local CPU):** `test_c4_pilot_1200_split_and_nonzero_tau_calibration`. Real L40S pilot pending. |
| **E. Exact Effective-Batch Micro-Batch Weighting** | Standard `mean()` per micro-batch misweighted Dismissal Loss whenever positive counts varied across micro-batches of an effective 80-image batch. | Implemented [`DynamicDismissalLoss(..., reduction=...)`](../src/models/losses.py#L44-L82) and [`TRUSTLoss.forward_microbatch`](../src/models/losses.py#L140-L185) scaling BCE/Focal by `total_batch_size` and Dismissal by `max(total_pos_in_batch, 1)`; defaulted `micro_batch_size=80` on L40S. | **PASSED (Local CPU):** `test_b2_microbatch_positive_weighting_exact_equivalence` (`0/0/15/5` skewed pos split matches full-batch `<= 1e-6`). |
| **F. CPU Stage-1 Deterministic `uint16` Crop Cache** | Decoding raw compressed DICOMs across 1,000,000 20-epoch occurrences would waste `~120.7 hrs` (`~$285`) on L40S. | Created [`src/data/stage1_cache.py`](../src/data/stage1_cache.py) (`anchorshift_stage1_uint16_v1`, atomic `fsync` shards, `Stage1CroppedLoader`) and [`scripts/cache_stage1_rsna_modal.py`](../scripts/cache_stage1_rsna_modal.py); wired `Stage1CroppedLoader` + `--benchmark-extract-batches` into [`scripts/train_trust_modal.py`](../scripts/train_trust_modal.py). | **PASSED (Local CPU):** `test_f2_stage1_uint16_cache_build_verify_resume_and_stage2_parity` (bitwise equal to `preprocess_dicom`, shard corruption & recovery tested). Cloud build pending. |
| **G. Preprocessing Safety, Strict LUT/Laterality & Verbatim TRUST Crop** | Prior code could silently swallow LUT errors, fall back to heuristic left/right guessing, fail silently on empty segmentation masks, and applied connected-component filtering not stated in TRUST Sec 3.2. | Rewrote [`src/data/preprocessing.py`](../src/data/preprocessing.py) (`PREPROCESS_PIPELINE_VERSION = "trust_sec3_2_v2"`): strict LUT error propagation, `laterality_policy="strict"` default, explicit `segmentation_status` + `strict_segmentation=True`, verbatim TRUST `use_largest_cc=False` default, full Stage-1/Stage-2 hashes; added `--audit-real` mode; passed explicit CSV `laterality` in Modal scripts. | **PASSED (Local CPU):** `test_d1` & `test_d2`; real RSNA visual `--audit-real` pending Modal run. |
| **H. Feature Cache `v2` Validation & Proper `os.fsync`** | Prior `_atomic_write_torch` called `os.fsync` outside open file scope/on closed or missing fd check cleanly; missing `encoder_precision`/`stage1_config_hash`/finiteness/eval ID checks. | Updated [`src/data/feature_cache.py`](../src/data/feature_cache.py) (`CACHE_SCHEMA_VERSION = "anchorshift_aug_cache_v2"`): `os.fsync(f.fileno())` inside write block, `get_cache_manifest_sha256`, `stage1_config_hash`, `encoder_precision`, strict dtype/shape/`torch.isfinite`/binary-label/eval ID order checks; added `--verify-cache`. | **PASSED (Local CPU):** `test_f1` & `test_f3`. |
| **I. Pinned Environment & Runtime Metadata** | Modal containers used unpinned `>=` package bounds on Python `3.11` while local used Python `3.12`. | Created [`src/utils/env_metadata.py`](../src/utils/env_metadata.py) (`PINNED_MODAL_PYTHON_VERSION = "3.12"`, 13 exact `==` packages), updated [`requirements-lock.txt`](../requirements-lock.txt) and all Stage B Modal scripts (`cache_stage1_rsna_modal.py`, `pilot_parity_modal.py`, `smoke_test_stage_b_modal.py`, `benchmark_l40s_comparison_modal.py`, `train_trust_modal.py`), and logged runtime metadata in manifests & eval JSONs. | **PASSED (Local CPU):** lockfile/code equality checked in `test_g3`; `py_compile` passed on all 5 Modal scripts. |

---

## 3. TRUST (`arXiv:2609.00300v1`) Traceability & Documented Research Choices

| Specification Area | Original Paper (`arXiv:2609.00300v1`) | AnchorShift Implementation | Classification |
|---|---|---|---|
| **Backbone & Head** | Frozen Mammo-CLIP EfficientNet-B5 (`eval()`, 2048-D) -> `LayerNorm(2048) -> Dropout(0.3) -> Linear(2048, 1) -> Sigmoid` | Exact match; `shawn24/Mammo-CLIP` pinned commit `2f356926a00fc3f0d9fdad1193c1464fd9adf564`, SHA-256 `0baccb0c...6fd02c`, `852/852` keys `strict=True`, isolated `build_trust_head`. | **EXACT MATCH** |
| **Objective (`TRUSTLoss`)** | `L_CE + 1.0 * L_focal(gamma=2.0, no alpha) + 0.05 * L_dismiss(m=0.10, tau_1=0.05)` | Exact match across both `forward()` and `forward_microbatch()`. | **EXACT MATCH** |
| **Dynamic Threshold `tau_t`** | Recomputed at start of epochs `>= 2` on image-level `train_calib` predictions: max `tau` with one-sided 95% Clopper–Pearson upper bound `<= 1%` (`0.0` fallback). | Exact match (`compute_dynamic_training_threshold_with_diagnostics`), with Eq. 5 exact boundary handling (`n_d=0 -> 1.0`, `k_d=n_d -> 1.0`). | **EXACT MATCH** |
| **Stage-1 & Stage-2 Preprocessing** | LUTs -> `MONOCHROME1` invert -> right-breast mirror -> border-median mask, fill holes, `9x9` closing, zero outside mask, 10% crop -> 16-bit -> `1520x912` linear resize -> `[0,1]` norm -> affine (`+-10 deg`, `0.9-1.1` scale, `+-5%` shift) + brightness `+-0.05` + contrast `0.90-1.10` + `[0,1]` clip -> 3ch Mammo-CLIP standardize. | Exact match (`use_largest_cc=False` default per paper, `laterality_policy="strict"`). | **EXACT MATCH** |
| **Sampling (`steps_per_epoch`)** | 20 epochs; independent minibatches of `20 pos + 60 neg = 80` (`paper_independent`); AdamW `lr=3e-5, wd=1e-4`, clip `5.0`, final 20th-epoch checkpoint. | Exact algorithm match. **Explicit Choice:** Paper omits integer batch count per epoch; we set `steps_per_epoch = ceil(37,444 / 60) = 625` batches/epoch (`1,000,000` occurrences across 20 epochs) matching exact `train` negative count (`37,444`) in `datasets/rsna/splits_seed42.csv`. | **EXACT + DOCUMENTED STEP COUNT** |
| **Split Protocol & NLBS Quarantine** | Paper uses 5-fold RSNA CV internally and 90/10 RSNA + 20% labeled NLBS externally. | AnchorShift intentionally uses single patient-level 70/10/20 RSNA split (Emory USA + BreastScreen Victoria Australia; `train=38,251`, `train_calib=5,473`, `val_calib=10,982`; 0 patient overlap) so Protocol A thresholds lock with **zero NLBS labels**; NLBS remains strictly quarantined. | **INTENTIONAL ANCHORSHIFT PROTOCOL** |

---

## 4. Complete Local CPU Test Matrix (`20/20 PASSED`)

Executed after final code edits on Python `3.12.10` (`torch==2.5.1+cu121`):
- `python tests/test_stage_b_smoke.py` -> **Ran 20 tests in 4.613s — OK (0 failures, 0 errors, 0 skipped)**
- `python -m unittest discover -s tests -p "test_*.py" -v` -> **Ran 20 tests in 4.496s — OK (0 failures, 0 errors, 0 skipped)**
- `python -m py_compile` on `scripts/cache_stage1_rsna_modal.py`, `scripts/pilot_parity_modal.py`, `scripts/train_trust_modal.py`, `scripts/smoke_test_stage_b_modal.py`, `scripts/benchmark_l40s_comparison_modal.py` -> **Exit code 0**

| Category | Test IDs Covered | What Passed Locally |
|---|---|---|
| **A. Model & Init Parity** | `test_a1` – `test_a4` | `852/852` key loading on synthetic tarball, frozen encoder gradient isolation, max case aggregation + validation, RNG-isolated `build_trust_head` identical with/without prior encoder construction (`0.0` diff). |
| **B. Losses & Micro-Batching** | `test_b1` – `test_b2` | Hand-verified BCE, unweighted Focal (`gamma=2.0`), Dismissal (`m=0.10`), TRUST (`lambda_dismiss=0.05`), zero-positive batches, and exact micro-batch equivalence under skewed `0/0/15/5` positive ordering. |
| **C. Calibration & 1,200-Img Pilot Split** | `test_c1` – `test_c4` | Clopper–Pearson Eq. 5 edges, CP95 `tau_t` + fallback, tie-safe target recall, and real `splits_seed42.csv` `sample_pilot_1200` verification (`500 / 500 / 200`, `0` leakage, `480 >= 299` achieving nonzero `tau_t`). |
| **D. DICOM Preprocessing & Safety** | `test_d1` – `test_d2` | `MONOCHROME1/2`, LUTs, right-breast mirroring, `use_largest_cc=False` crop, strict missing-laterality rejection, malformed LUT error propagation, empty-mask strict failure vs explicit diagnostic fallback status, and config hashes. |
| **E. Splits & Sampler** | `test_e1` – `test_e2` | `splits_seed42.csv` (`54,706` imgs, `11,913` pts, `0` overlap), `paper_independent` & `full_neg_pass`, exact sampler state restore. |
| **F. Stage-1 & Feature Cache `v2`** | `test_f1` – `test_f3` | Stage-1 `uint16` shard build/verify/resume + bitwise Stage-2 parity vs `preprocess_dicom`; Feature Cache `v2` integrity, checksums, duplicate/missing draw recovery, NaN rejection, precision/stage1/seed/eval metadata rejection. |
| **G. Parity & Checkpoint `v2` Resume** | `test_g1` – `test_g3` | 3-epoch CPU Online-vs-Cached pilot (`<= 1e-5` across all 10 metrics), 2-epoch exact save/resume parity (`0.0` divergence), rejection of legacy schema, missing config keys, missing RNG states, cache hash mismatch, and lockfile consistency. |

---

## 5. Honest Verification Boundary: Verified Locally vs. Pending Cloud Execution

| Check | Status | Notes |
|---|---|---|
| Mathematical losses, calibration bounds, split leakage, cache/checkpoint validation | **VERIFIED LOCALLY (CPU)** | Deterministic logic covered by `20/20` unit/integration tests. |
| Real RSNA DICOM LUT/segmentation across `L/R x CC/MLO x MONOCHROME1/2` | **PENDING MODAL (`--audit-real` + `--build`)** | Synthetic DICOM fixtures passed locally; real dataset corner cases must be audited on the actual `rsna-mammography-vol` DICOMs before full training. |
| CUDA RNG save/resume determinism & AMP numerical parity (`<= 1e-3`) | **PENDING MODAL (`pilot_parity_modal.py`)** | Local tests ran on CPU FP32; CUDA `GradScaler`, `torch.cuda` RNG restore, and real Mammo-CLIP B5 AMP parity require Modal L40S execution. |
| End-to-end Stage-1 -> Stage-2 -> L40S throughput & exact cloud cost | **PENDING MODAL (`--benchmark-extract-batches 5`)** | Only empirical raw-DICOM decode + B5 encoding (`ap-KidtVEoMzokd7fsFzpMeFo`) exists; Stage-1 `uint16` shard I/O + Stage-2 affine extraction rate has NOT been measured on Modal yet. |

---

## 6. Recalculated Storage, Throughput & Modal Cost Analysis

Exact `datasets/rsna/splits_seed42.csv` counts verified on disk:
- `train`: `38,251` images (`8,339` patients, `807` pos, **`37,444` neg**) -> `ceil(37,444 / 60) = 625` batches/epoch of 80 -> **`1,000,000` occurrences over 20 epochs**.
- `train_calib`: `5,473` images (`1,192` patients, `114` pos, `5,359` neg).
- `val_calib`: `10,982` images (`2,382` patients, `237` pos, `10,745` neg).

### Storage Budget
1. **Stage-1 Lossless `uint16` RSNA Cache (`54,706` cropped images):** Compressed/sharded `.pt` tensors stored once (`~15–25 GB` estimated depending on average cropped bounding-box area; verified during `--build --pilot-only` / `--build`). Note: delete temporary pilot Stage-1/feature caches once full caches are verified if disk quota needs cleanup, or leave on volume if ample space.
2. **Stage-2 Augmentation-Aware FP32 Feature Cache (`rsna_aug_seed42`):**
   - Training tensor bytes: `1,000,000 * 2048 * 4 = 8,192,000,000 bytes` = **`8.192 GB` (`7.629 GiB`)**.
   - Deterministic eval (`train_calib + val_calib = 16,455` imgs): `16,455 * 2048 * 4 = 134,799,360 bytes` = **`134.80 MB` (`128.55 MiB`)**.
   - **Total FP32 Feature Cache:** **`8.327 GB` (`7.755 GiB`)** raw feature tensors (~`8.35 GB` including labels, sample indices, and augmentation metadata).

### Throughput & Cost Projection (Measured vs. Estimated)
- **Measured Historical Baseline (`ap-KidtVEoMzokd7fsFzpMeFo`, L40S):** Raw DICOM decode `434.40 s / 1k` (`91.5%`), Stage-2 preproc `13.88 s / 1k`, Mammo-CLIP B5 encode `26.49 s / 1k` (`2.12 s / 80 imgs`). Assumed L40S container rate = `$2.3608/hr` (`$1.95` GPU + `6 * $0.04716` CPU + `16 * $0.00799` RAM).
- **Without Stage-1 Cache (DO NOT RUN):** Decoding `1,000,000` raw DICOMs on L40S would take `~120.7 hrs` (**`~$285`**, ~`9.5x` our `$30` credit).
- **With CPU Stage-1 Cache + One-Time L40S Feature Extraction (ESTIMATED):**
  1. CPU Stage-1 pre-crop (`54,706` imgs once on `cpu=8.0, memory=16384` at `~$0.505/hr`): estimated `~3–6 hrs` (`~$1.5–$3.1`).
  2. L40S Feature Extraction (`1,000,000` Stage-2 resize/aug + Mammo-CLIP B5 passes): lower bound at pure encode `26.49 s / 1k` = `~7.36 hrs` (`~$17.37`); allowing Stage-2 affine + shard write overhead, measure exact throughput first via **`--benchmark-extract-batches 5`** before committing to full extraction!
  3. Cached Head Training (`ce` 20 epochs + `trust` 20 epochs): `< 2 min` each (`~$0.08` total), reusing the single verified `8.33 GB` cache identically across both baselines and any ablations.

---

## 7. Pinned Environment & Runtime Metadata Contract

Both [`requirements-lock.txt`](../requirements-lock.txt) and [`src/utils/env_metadata.py`](../src/utils/env_metadata.py) pin Python `3.12` and all 13 direct packages (`torch==2.5.1`, `torchvision==0.20.1`, `numpy==2.5.3`, `pandas==2.3.3`, `scipy==1.17.1`, `pydicom==3.0.2`, `opencv-python-headless==4.11.0.86`, `pillow==12.3.0`, `omegaconf==2.3.0`, `pylibjpeg==2.0.0`, `pylibjpeg-libjpeg==2.1.0`, `pylibjpeg-openjpeg==2.1.0`, `python-gdcm==3.0.24`). All 5 Stage B Modal scripts import `PINNED_MODAL_PYTHON_VERSION` and `PINNED_MODAL_PIP_PACKAGES`, and `collect_runtime_environment_metadata` logs actual container Python, PyTorch, CUDA, cuDNN, GPU model, and precision policy inside every benchmark, cache manifest, Stage-1 build report, and final evaluation JSON.

---

## 8. Mandatory Pre-Training Cloud Verification Gates & Stop/Go Criteria

| Gate | Command Target | Explicit Pass Criteria (STOP if failed) |
|---|---|---|
| **Gate 1: Real DICOM Audit** | `cache_stage1_rsna_modal.py --audit-real` | `fallback_count == 0`, `success_count == 16/16` across `L/R x CC/MLO`, `MONOCHROME1` tissue bright & right breasts mirrored left-aligned upon preview check. |
| **Gate 2: T4 Smoke Test** | `smoke_test_stage_b_modal.py` | `checkpoint_loaded_strict == True`, `matched_keys == 852/852`, `missing == 0`, `unexpected == 0`, `encoder_frozen_verified == True`. |
| **Gate 3: Full Stage-1 Cache** | `cache_stage1_rsna_modal.py --build` & `--verify` | `54,706/54,706` images verified with SHA-256 pixel checks (`missing_laterality_count == 0`, `fallback_count == 0` under `--strict-segmentation`; if isolated RSNA frames fail segmentation, inspect IDs before deciding `--no-strict-segmentation`). |
| **Gate 4: L40S Parity Pilot** | `pilot_parity_modal.py` (`1,200` imgs, `20` ep) | `status == "PASS"` in BOTH `fp32` (`<= 1e-5`) and `amp` (`<= 1e-3`), `prediction_agreement_95 == 1.0`, `prediction_agreement_98 == 1.0`, `saw_nonzero_epoch2_plus_tau == True` (confirming dynamic `tau_t` activation on `N_calib=500`). |
| **Gate 5: 5-Batch Extract Timing** | `train_trust_modal.py --benchmark-extract-batches 5` | Projected full cost + already-spent credits fits safely within `$30` credit budget before launching full `--extract-cache`. |

---

## 9. Exact Manual Execution Commands (In Order — Matching `@app.local_entrypoint()` Flags)

> [!CAUTION]
> **DO NOT** run Step 7 (`--extract-cache`) or Steps 8–9 (`train_trust_modal.py` training) until Steps 1–6 pass all Stop/Go gates above!

```powershell
# 1. Local CPU 20-test verification (free, already verified locally)
python tests/test_stage_b_smoke.py
python -m unittest discover -s tests -p "test_*.py" -v

# 2. Gate 1: Real RSNA Stage-1 preprocessing & segmentation visual audit (CPU-only, ~1 min)
modal run scripts/cache_stage1_rsna_modal.py --audit-real --samples-per-view-lat 4

# 3. Gate 2: Quick strict GPU smoke test on Tesla T4 (852/852 keys + updated strict LUT/crop, ~$0.01)
modal run scripts/smoke_test_stage_b_modal.py

# 4. Gate 3: Build & strictly verify full 54,706-image Stage-1 uint16 cache on Modal CPU (~$1.5-$3.1 est)
modal run --detach scripts/cache_stage1_rsna_modal.py --build --strict-segmentation
modal run scripts/cache_stage1_rsna_modal.py --verify --strict-segmentation

# 5. Gate 4: MANDATORY 1,200-image L40S Online-vs-Cached Mammo-CLIP Parity Pilot (FP32 + AMP, 20 epochs)
modal run scripts/pilot_parity_modal.py --epochs 20 --steps-per-epoch 12 --micro-batch-size 80 --precision both --strict-segmentation --gpu L40S

# 6. Gate 5: Measure Stage-1 -> Stage-2 -> L40S extraction speed on 5 batches before full 1M extraction
modal run scripts/train_trust_modal.py --benchmark-extract-batches 5 --epochs 20 --steps-per-epoch 625 --micro-batch-size 80 --precision amp --gpu L40S

# 7. ONLY AFTER GATES 1-5 PASS: Extract & verify full 20-epoch augmentation-aware FP32 cache (1,000,000 occurrences + 16,455 eval)
modal run --detach scripts/train_trust_modal.py --extract-cache --epochs 20 --steps-per-epoch 625 --micro-batch-size 80 --precision amp --gpu L40S
modal run scripts/train_trust_modal.py --verify-cache --epochs 20 --steps-per-epoch 625 --precision amp

# 8. Train CE-only baseline (20 epochs, final-epoch checkpoint) on verified cached features
modal run scripts/train_trust_modal.py --baseline ce --epochs 20 --steps-per-epoch 625 --micro-batch-size 80 --precision amp --gpu L40S --use-cached-features

# 9. Train TRUST closed-loop model (20 epochs, CP95 dynamic tau_t, final-epoch checkpoint) on verified cached features
modal run scripts/train_trust_modal.py --baseline trust --epochs 20 --steps-per-epoch 625 --micro-batch-size 80 --precision amp --gpu L40S --use-cached-features

# 10. Optional Recovery Command (only if Step 8 or 9 is interrupted; enforces v2 schema + 22 config fields + cache SHA + RNG states)
modal run scripts/train_trust_modal.py --baseline trust --epochs 20 --steps-per-epoch 625 --micro-batch-size 80 --precision amp --gpu L40S --use-cached-features --resume
```

---

## 10. Final Pre-Training Checklist

- [x] Research direction preserved: RSNA (`Emory USA + BreastScreen Victoria Australia`) -> NLBS transfer; NLBS strictly quarantined (`0` images/labels touched in Stage B).
- [x] Exact TRUST (`arXiv:2609.00300v1`) loss equations, unweighted focal `gamma=2.0`, `m=0.10`, `lambda_dismiss=0.05`, `tau_1=0.05`, CP95 `<= 1%` image-level `tau_t`, `20:60` independent batches, `steps_per_epoch=625` (`37,444` train negatives), final 20th-epoch checkpointing.
- [x] All Critical Fixes A–I implemented across `src/` and `scripts/` without duplicated training logic.
- [x] All `20/20` local CPU unit & integration tests passing (`0` failures, `0` skipped) + `py_compile` clean across all Stage B Modal scripts.
- [ ] User executes Gate 1 (`--audit-real`) through Gate 5 (`--benchmark-extract-batches 5`) manually on Modal and verifies PASS criteria before launching Steps 7–9.
