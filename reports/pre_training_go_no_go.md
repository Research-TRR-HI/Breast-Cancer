# AnchorShift Pre-Training Go / No-Go Audit

**Audit Date:** 2026-11-18  
**Target Experiment:** Stage B4 — Source Training (CE-Only vs. TRUST, 20 Epochs) on RSNA Screening Mammography  
**Decision:** **CONDITIONAL GO** (Local code fixes & all 13 CPU unit/parity tests PASSED; cloud Modal online-vs-cached Mammo-CLIP parity pilot remains pending user execution before authorizing full feature extraction & 20-epoch training)

---

## 1. Corrections Completed

| File | Summary of Changes |
|---|---|
| [`src/models/losses.py`](../src/models/losses.py) | Fixed `FocalLoss` to match TRUST Eq. 12 (`gamma=2.0`, default `alpha=None`, removing unintended `0.25` class weighting); fixed `DynamicDismissalLoss` (`margin=0.10`); fixed `TRUSTLoss` defaults (`lambda_focal=1.0`, `lambda_dismiss=0.05`, `margin=0.10`) and added `ce`, `ce_focal`, `fixed_tau`, `trust` modes. |
| [`src/training/calibration.py`](../src/training/calibration.py) | Fixed `clopper_pearson_upper_bound` (`n_d=0 -> 1.0`, `k_d=n_d -> 1.0`, `k_d=0` exact formula, `k>n` & invalid confidence `ValueError`); replaced percentile-based training `tau_t` with `compute_dynamic_training_threshold` (image-level `train_calib`, CP95 `<= 1%`, `0.0` fallback); fixed finite-sample & tie-safe `compute_target_recall_threshold` and `N/A` reporting when target recall not met. |
| [`src/models/trust_model.py`](../src/models/trust_model.py) | Added strict length & non-empty validation in `aggregate_case_scores`; added `aggregate_case_labels`. |
| [`src/models/mammo_clip_loader.py`](../src/models/mammo_clip_loader.py) & [`scripts/download_mammoclip_modal.py`](../scripts/download_mammoclip_modal.py) | Pinned HuggingFace commit `2f356926a00fc3f0d9fdad1193c1464fd9adf564` and exact SHA-256 `0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c` (`1,654,145,453` bytes); added `verify_sha256` & `compute_file_sha256`. |
| [`src/data/preprocessing.py`](../src/data/preprocessing.py) | Implemented full TRUST Sec. 3.2 pipeline: `apply_modality_lut`, `apply_voi_lut`, `MONOCHROME1` inversion, right-breast horizontal mirroring, border-median breast segmentation, `binary_fill_holes`, `9x9` morphological closing, mask background zeroing, `10%` bbox cropping (no frame stretching), 16-bit quantization, `1520x912` bilinear resize, `[0,1]` re-norm, seeded stochastic augmentation (`+-10 deg`, `0.9-1.1` scale, `+-5%` shift, `+-0.05` brightness, `0.9-1.1` contrast, `[0,1]` clip), 3ch replication, Mammo-CLIP standardization, and `get_preprocessing_config_hash`. |
| [`src/data/rsna_dataset.py`](../src/data/rsna_dataset.py) | Unified canonical `BalancedBatchSampler` defaulting to `paper_independent` (independent draws of 20 pos + 60 neg per minibatch, configurable `steps_per_epoch`), added full sampler RNG `get_state`/`set_state`, and added `build_epoch_draw_schedule`. |
| [`src/data/feature_cache.py`](../src/data/feature_cache.py) | Implemented augmentation-aware sharded FP32 feature cache (`(epoch, batch_idx, slot_idx)`), atomic `.tmp -> fsync -> replace` shard writes, SHA-256 manifest verification, missing/duplicate/corrupted draw detection, hash mismatch rejection, and interrupted extraction recovery. |
| [`src/training/checkpointing.py`](../src/training/checkpointing.py) | Implemented atomic epoch-boundary checkpointing & resume saving/restoring model, optimizer, scaler, epoch, `tau_t`, history, Python/NumPy/PyTorch CPU/CUDA RNGs, sampler RNG, augmentation RNG, and training config; raises `FileNotFoundError` when `--resume` checkpoint is missing. |
| [`src/training/parity_pilot.py`](../src/training/parity_pilot.py) & [`scripts/pilot_parity_modal.py`](../scripts/pilot_parity_modal.py) | Built controlled 1,000-image online-vs-cached equivalence pilot measuring all 10 parity metrics (features, step/epoch losses, grads, head weights, logits, probs, prediction agreement, `tau_t` & operating thresholds, final dismissal/recall/CP99). |
| [`scripts/train_trust_modal.py`](../scripts/train_trust_modal.py) | Updated to corrected TRUST hyperparameters, dynamic CP95 `tau_t`, final 20th-epoch checkpointing (no `val_calib` double-use), complete RNG `--resume`, `--extract-cache`, and `--use-cached-features`. |
| [`scripts/smoke_test_stage_b_modal.py`](../scripts/smoke_test_stage_b_modal.py) & [`scripts/benchmark_l40s_comparison_modal.py`](../scripts/benchmark_l40s_comparison_modal.py) | Updated for `opencv-python-headless`, 3-tuple `decode_dicom_array`, and corrected TRUST loss parameters. |
| [`requirements-lock.txt`](../requirements-lock.txt) | Created pinned Python/package lockfile. |
| [`tests/test_stage_b_smoke.py`](../tests/test_stage_b_smoke.py) | Rebuilt comprehensive 13-test suite covering Categories A–G (runnable via `unittest` and direct script). |

---

## 2. TRUST Fidelity Table & Explicit Deviations

| Requirement (`arXiv:2609.00300v1`) | Fidelity Status | Remaining Intentional / Documented Deviations |
|---|---|---|
| **Loss (`L_CE + 1.0*L_focal(gamma=2, no alpha) + 0.05*L_dismiss(m=0.10)`)** | **EXACT MATCH** | None. |
| **Dynamic `tau_t` (init `0.05`, CP95 `<= 1%` on image-level `train_calib`, `0.0` fallback)** | **EXACT MATCH** | None. |
| **Clopper–Pearson Eq. 5 (`n_d=0 -> 1.0`, `k_d=n_d -> 1.0`, 95% train / 99% eval)** | **EXACT MATCH** | None. |
| **Preprocessing & Augmentation (LUTs, `MONOCHROME1`, R-mirror, `9x9` close + 10% crop, 16-bit, `1520x912`, `+-10°`/`0.9-1.1`/`+-5%`/`+-0.05`/`0.9-1.1`)** | **EXACT MATCH** | Minor implementation detail: when DICOM `ImageLaterality`/`Laterality` tags are absent, right-breast mirroring falls back to half-image mean intensity heuristic (`auto_detect_right=True`). Affine augmentation uses pure PyTorch `affine_grid`/`grid_sample` for CPU/GPU determinism. |
| **Sampling (`20 pos + 60 neg` independent minibatches, 20 epochs)** | **EXACT MATCH + DOCUMENTED STEP COUNT** | Paper does not specify exact integer minibatches per epoch; we default to `steps_per_epoch = ceil(N_neg / 60) = 620` batches/epoch on our RSNA `train` split (configurable via CLI). |
| **Model Selection (20 epochs, final-epoch checkpoint, no `val_calib` double-use)** | **EXACT MATCH** | None. |
| **Dataset Split Protocol** | **INTENTIONAL ANCHORSHIFT DEVIATION** | Paper uses 5-fold 70/10/20 CV on RSNA for in-domain, and 90/10 RSNA + 20% **labeled NLBS** for external threshold search. AnchorShift uses a single patient-level **70/10/20 RSNA split** (`train` 70%, `train_calib` 10%, `val_calib` 20%) so operating thresholds are frozen using **RSNA labels only** (strict zero-NLBS-label Protocol A). |
| **NLBS Quarantine** | **STRICTLY ENFORCED** | Zero NLBS images or cancer labels accessed anywhere in Stage B. |

---

## 3. Test Results

### Local CPU Tests Executed & Passed (2026-11-18)
- Command 1: `python tests/test_stage_b_smoke.py` -> **13/13 PASSED (`OK`, 4.210s, 0 failed, 0 skipped)**
- Command 2: `python -m unittest discover -s tests -p "test_*.py" -v` -> **13/13 PASSED (`OK`, 3.871s, 0 failed, 0 skipped)**

| Test Category | Tests Executed Locally | Result | Cloud / Modal Status |
|---|---|---|---|
| **A. Model (`test_a1`–`test_a3`)** | 852-key `efficientnet_custom` match on synthetic tarball, frozen encoder, single-sigmoid, grad isolation, case aggregation length guard | **PASS (Local)** | Real `b5-model-best-epoch-7.tar` 852/852 & SHA-256 verification on Modal **PENDING re-run with updated preprocessing** |
| **B. Losses (`test_b1`)** | Hand-calculated exact checks for BCE, un-weighted Focal, Dismissal (`m=0.10`), TRUST (`lambda_d=0.05`), CE-only, `N_pos=0` | **PASS (Local)** | Included in Modal smoke test |
| **C. Calibration (`test_c1`–`test_c3`)** | CP edge cases (`n=0`, `k=n`, `k=0`, `k>n`), CP95 dynamic `tau_t` + `0.0` fallback, discrete ties, recall threshold & `N/A` reporting | **PASS (Local)** | N/A (CPU math only) |
| **D. Preprocessing (`test_d1`)** | Synthetic DICOM fixtures: `MONOCHROME1/2`, Modality/VOI LUT, R-breast mirroring, breast mask/holes/`9x9`/crop, `1520x912`, standardization, deterministic eval, seeded augmentation | **PASS (Local)** | Real RSNA DICOM visual/stats check via Modal smoke test **PENDING** |
| **E. Dataset & Sampler (`test_e1`–`test_e2`)** | `splits_seed42.csv` 0-leakage (11,913 pts, 54,706 imgs), `paper_independent` & `full_neg_pass` 20:60 batches, RNG state restore | **PASS (Local)** | N/A (CSV & sampler verified locally) |
| **F. Augmentation-Aware Caching (`test_f1`)** | Sharded cache integrity, SHA-256 mismatch, missing/corrupt shard, duplicate draw, hash mismatch, interrupted recovery | **PASS (Local)** | Real L40S cache extraction **PENDING pilot** |
| **G. Parity & Checkpoint Resume (`test_g1`–`test_g2`)** | 3-epoch CPU Online-vs-Cached pilot (`<= 1e-5` across all 10 metrics) + 2-epoch exact save/resume parity (`0.0` divergence) + `--resume` missing error | **PASS (Local)** | **Real Mammo-CLIP L40S Online-vs-Cached Parity Pilot (`scripts/pilot_parity_modal.py`) PENDING** |

---

## 4. Online-vs-Cached Comparison

- **Local CPU Parity Pilot (`test_g1_online_vs_cached_parity_pilot_cpu`):** **PASSED** across 3 epochs (`batch_size=80`, `20 pos + 60 neg`, deterministic frozen CNN encoder, FP32) with all 10 metrics (`max_abs_feature_diff`, `step_loss`, `epoch_loss`, `grad`, `weight`, `logit`, `prob`, `tau_trajectory`, `100%` prediction agreement, identical `val_calib` dismissal & recall metrics) within `<= 1e-5`.
- **Cloud L40S Mammo-CLIP Parity Pilot (`scripts/pilot_parity_modal.py`):** **NOT YET EXECUTED** (by project rule prohibiting autonomous Modal runs). Must be executed and pass (both `fp32` `<= 1e-5` and `amp` `<= 1e-3`) before authorizing full RSNA extraction or training.

---

## 5. Recalculated Resource, Storage & Modal Cost Estimates

> [!IMPORTANT]
> The previously reported `$17.06` / `429.6 MB` estimate from `ap-KidtVEoMzokd7fsFzpMeFo` measured **1 fixed feature per image (no augmentation)** and is **NOT valid** for stochastic-augmentation 20-epoch TRUST training. It is retained strictly as an empirical measurement of single-pass DICOM decode (`434.40 s / 1k imgs`) and Mammo-CLIP L40S inference throughput (`26.49 s / 1k imgs` = `2.12 s / 80 imgs`).

### Exact Corrected Workload Scale (RSNA `splits_seed42.csv`)
- `train`: `38,251` images (`37,160` negative / `1,091` positive -> `ceil(37,160 / 60) = 620` batches/epoch of 80).
- `train_calib`: `5,473` images (deterministic, unaugmented).
- `val_calib`: `10,982` images (deterministic, unaugmented).
- **20-Epoch Augmented Training Occurrences:** `620 batches/epoch * 80 imgs * 20 epochs = 992,000` occurrences.
- **Modal L40S Combined Rate (Assumed):** GPU `$1.95/hr` + `6 vCPU * $0.04716/hr` + `16 GiB RAM * $0.00799/GiB-hr` = **`$2.3608/hr`**.

### Option 1 — Full Augmentation-Aware 20-Epoch Feature Cache (`--extract-cache` + `--use-cached-features`)
*Note: `scripts/train_trust_modal.py --extract-cache` decodes DICOMs per occurrence unless Stage-1 cropped 16-bit PNGs are cached first. Using measured `ap-KidtVEoMzokd7fsFzpMeFo` timings:*
- **Storage Required:**
  - `992,000` train occurrences × `2048` × `4 bytes` = **`8.126 GB` (`7.57 GiB`)** FP32 (across sharded `.pt` files) + `16,455` deterministic eval features = **`134.8 MB`** -> **`~8.26 GB` total**.
- **Extraction Time & Cost (ESTIMATE — unmeasured on Modal with Stage-1 crop):**
  - **If decoding raw DICOMs on the fly per occurrence (NOT recommended on L40S):** `992,000` DICOM decodes at measured `434.40 s / 1k` = `~120 hrs` (`~$283`), which would blow the `$30` credit.
  - **If Stage-1 cropped 16-bit images (54,706 imgs, `~1-3 GB`) are pre-cached once on cheap CPU workers (`~7.22 hrs / 6-vCPU` or 10 parallel CPU workers in `~45 min` ≈ `~$2.05`), and then 992,000 Stage-2 augment + L40S encoder passes run at measured `26.49 s / 1k` (`2.12 s / 80`):**
    - L40S Feature Extraction (`992,000` passes): **`~7.30 hrs`** -> **`~$17.23`** (GPU `$14.24`, CPU `$2.07`, RAM `$0.93`).
  - **Cached CE (20 epochs) + Cached TRUST (20 epochs):** Measured head-only speed (`< 1 min` each) -> **`~2 min` combined** -> **`~$0.08`**.
  - **Total Option 1 (with CPU Stage-1 pre-crop):** **`~$19.36`** (within `$30` budget, leaving `~$10.64` for NLBS + UPA).

### Option 2 — Stage-1 Deterministic CPU Cache + Online Stage-2 Augmentation & Live Frozen Encoder (`--use-cached-features=False` on Stage-1 PNGs)
- **Storage Required:** 54,706 cropped 16-bit PNGs (`~1.5–3 GB`), zero `8.1 GB` feature cache needed.
- **Cost (ESTIMATE):** CPU Stage-1 (`~$2.05`) + Live L40S CE (`~7.30 hrs ≈ $17.23`) + Live L40S TRUST (`~7.30 hrs ≈ $17.23`) = **`~$36.51`** (EXCEEDS `$30` credit because the L40S encoder runs twice — once for CE, once for TRUST!).

**Cost Conclusion:** Pre-caching Stage-1 cropped 16-bit images on CPU workers + running `--extract-cache` once (Option 1) reuses the 992,000 augmented embeddings across both CE and TRUST (and any hyperparameter ablations), cutting total RSNA cost in half (**`~$19.36`** vs **`~$36.51`**) and staying inside our **`$30`** Modal budget.

---

## 6. Exact Manual Execution Commands (In Order)

```powershell
# 1. Local unit & parity test suite (CPU only, free)
python tests/test_stage_b_smoke.py
python -m unittest discover -s tests -p "test_*.py" -v

# 2. Strict Modal T4 smoke test verifying updated LUT/crop/standardize preprocessing + 852/852 keys (~1 min, ~$0.01)
modal run scripts/smoke_test_stage_b_modal.py

# 3. MANDATORY GATE: 1,000-image L40S Online-vs-Cached Mammo-CLIP Parity Pilot (FP32 + AMP, 20 epochs)
modal run scripts/pilot_parity_modal.py --epochs 20 --steps-per-epoch 12 --precision both --gpu L40S

# 4. ONLY AFTER STEP 3 PASSES: Full RSNA augmentation-aware feature extraction
# (Recommended: pre-cache Stage-1 cropped 16-bit images on CPU first before running --extract-cache on 992k occurrences!)
modal run --detach scripts/train_trust_modal.py --extract-cache --epochs 20 --steps-per-epoch 620 --gpu L40S

# 5. CE-only 20-epoch training on verified cached features
modal run scripts/train_trust_modal.py --baseline ce --epochs 20 --steps-per-epoch 620 --gpu L40S --use-cached-features

# 6. TRUST closed-loop 20-epoch training on verified cached features
modal run scripts/train_trust_modal.py --baseline trust --epochs 20 --steps-per-epoch 620 --gpu L40S --use-cached-features
```

---

## 7. Final Decision: **CONDITIONAL GO**

- **Why not NO-GO?** Every identified scientific and engineering blocker in Parts 1–9 (TRUST loss weights/margin/alpha, CP95 dynamic `tau_t`, CP99 edge cases, full DICOM LUT/segmentation/crop/mirror/augmentation preprocessing, independent 20:60 minibatch sampling, `val_calib` double-use removal, pinned Mammo-CLIP SHA-256, sharded augmentation-aware caching, and full RNG resume) is corrected and passes **13/13 local CPU tests**.
- **Why not full GO yet?** Two items remain before spending `$17+` on full extraction:
  1. The cloud L40S Mammo-CLIP **online-vs-cached parity pilot (`scripts/pilot_parity_modal.py`)** has not yet been executed manually.
  2. Before running `--extract-cache` on 992,000 occurrences, **Stage-1 cropped 16-bit image caching on CPU workers** should be added so `--extract-cache` does not re-decode raw DICOMs 992,000 times (`~$283` vs `~$19.36`).
