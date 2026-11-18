# Research Log: Pre-Training Correction, TRUST Paper Audit & Augmentation-Aware Caching

**Date**: 2026-11-18  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Scientific Fidelity Audit, Preprocessing, Loss/Calibration Repair & Cache Architecture  
**Milestone Scope**: Full audit of repository against original TRUST (`arXiv:2609.00300v1`) PDF, mathematical & preprocessing corrections, augmentation-aware sharded feature caching, deterministic RNG resume, 13-test local validation, and Go/No-Go audit  
**Status**: CODE & LOCAL TESTS COMPLETE (`13/13 PASS`) / CONDITIONAL GO PENDING CLOUD PARITY PILOT

---

## 1. Research Objective & Context
- Retrieve and verify the authoritative original TRUST manuscript (`arXiv:2609.00300v1`, Shafie et al., 2026) directly against every line of our Stage B reproduction codebase.
- Correct all mathematical, statistical, preprocessing, sampling, checkpointing, and feature-caching discrepancies prior to executing RSNA training on Modal L40S.
- Preserve strict zero-NLBS-label quarantine and avoid any cloud GPU execution until the controlled online-vs-cached parity pilot passes.

---

## 2. Authoritative Verification Sources & Pinned Artifacts
- **TRUST Reference Manuscript**: `arXiv:2609.00300v1 [cs.CV]` (31 Aug 2026, 25 pages; local inspection copy cached in session scratch). Confirmed that `A Territory Resonance... (TRUST).pdf` in workspace root is an unrelated WDBC paper by Mohandas et al. (2026).
- **Mammo-CLIP Pinned Revision**: `shawn24/Mammo-CLIP` commit `2f356926a00fc3f0d9fdad1193c1464fd9adf564`, file `Pre-trained-checkpoints/b5-model-best-epoch-7.tar` (`1,654,145,453` bytes, SHA-256 `0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c`).
- **Environment Lockfile**: `requirements-lock.txt` (`Python 3.12.10` / `torch 2.5.1`, `numpy 2.5.3`, `pandas 2.3.3`, `scipy 1.17.1`, `pydicom 3.0.2`, `opencv-python-headless 4.11.0.86`).

---

## 3. Forensic Audit Findings & Mathematical Corrections

### 3.1 TRUST Loss & Dynamic Thresholding (`src/models/losses.py`, `src/training/calibration.py`)
1. **Loss Hyperparameters (Eq. 6, 7, 12–14, Pages 8–11)**:
   - Corrected `DynamicDismissalLoss`: `margin = 0.10` (was `0.05`).
   - Corrected `TRUSTLoss`: `lambda_dismiss = 0.05` (was `1.0` — 20x too high), `initial_tau = 0.05` (was `0.50`).
   - Corrected `FocalLoss`: removed unintended `alpha = 0.25` class weighting (`alpha = None` default per Eq. 12), keeping `gamma = 2.0`, `lambda_focal = 1.0`.
2. **Dynamic `tau_t` Algorithm (Section 3.8, Page 9)**:
   - Replaced erroneous `np.percentile(..., 2.0)` with exact CP95 bound search (`compute_dynamic_training_threshold`): evaluates image-level `train_calib` predictions at epoch start (`epoch >= 2`), selecting the largest threshold where the one-sided **95% Clopper–Pearson upper bound** on dismissed positive-image prevalence is `<= 1%` (`0.01`), falling back to `0.0` if unsatisfied.
3. **Clopper–Pearson Edge Cases (Eq. 5, Page 8)**:
   - Fixed `n_dismissed == 0 -> 1.0` (was `0.0`!), `k_missed == n_dismissed -> 1.0`, exact `k_missed == 0` formula `1 - (1 - conf)**(1/n)`, and `ValueError` guards for `k > n`, negative counts, booleans, and invalid confidence levels.

### 3.2 Complete Section 3.2 Preprocessing & Stochastic Augmentation (`src/data/preprocessing.py`)
- Implemented Modality LUT & VOI LUT application, `MONOCHROME1` inversion, right-breast horizontal mirroring, border-median breast segmentation, hole filling (`binary_fill_holes`), `9x9` morphological closing, largest-component filtering, mask background zeroing, `10%` bounding-box cropping without frame stretching, 16-bit quantization simulation, `1520x912` linear resize, `[0,1]` re-normalization, and seeded stochastic augmentation (`+-10 deg`, `0.90-1.10` scale, `+-5%` shift, `+-0.05` brightness, `0.90-1.10` contrast, `[0,1]` clip).

### 3.3 Sampling, Model Selection & Cache Integrity (`src/data/rsna_dataset.py`, `src/data/feature_cache.py`, `src/training/checkpointing.py`)
- Unified `BalancedBatchSampler` defaulting to `paper_independent` (independent 20 pos + 60 neg draws per minibatch, configurable `steps_per_epoch` defaulting to `ceil(N_neg/60)=620`).
- Eliminated `val_calib` statistical double-use (`_best.pt` removed; fixed 20-epoch training, final-epoch checkpoint evaluated once on `val_calib`).
- Built occurrence-indexed `(epoch, batch_idx, slot_idx)` sharded FP32 feature cache (`src/data/feature_cache.py`) with atomic writes, SHA-256 checksums, hash guards, duplicate/missing draw checks, and recovery.
- Built complete RNG save/restore (`src/training/checkpointing.py`) including Python, NumPy, Torch CPU/CUDA, sampler, and augmentation RNGs, refusing silent epoch-1 restarts.

---

## 4. Local Quality Assurance Results
Executed on 2026-11-18 with zero errors, failures, or skips:
- `python tests/test_stage_b_smoke.py`: **13/13 PASSED (`4.210s`, `OK`)**
- `python -m unittest discover -s tests -p "test_*.py" -v`: **13/13 PASSED (`3.871s`, `OK`)**
- Verified locally: 852-key compatibility, frozen grad isolation, hand-computed losses (`< 1e-6`), CP95/CP99 edge cases, synthetic DICOM LUT/crop/mirror/aug fixtures, zero-leakage splits, sharded cache corruption/recovery, 3-epoch CPU online-vs-cached parity (`<= 1e-5` across all 10 metrics), and exact 2-epoch checkpoint save/resume parity (`0.0` weight diff).

---

## 5. Pending Cloud Gates (Before Full Training)
1. Run `modal run scripts/smoke_test_stage_b_modal.py` (verifies updated LUT/crop preprocessing + 852/852 on T4).
2. Run `modal run scripts/pilot_parity_modal.py --epochs 20 --steps-per-epoch 12 --precision both --gpu L40S` (1,000-image online-vs-cached parity gate on L40S).
3. Add CPU Stage-1 pre-crop cache before full `992,000`-occurrence extraction so total Modal cost stays near `~$19.36` within our `$30` credit budget.
