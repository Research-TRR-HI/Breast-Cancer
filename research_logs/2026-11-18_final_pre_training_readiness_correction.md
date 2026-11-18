# Research Log: Final Pre-Training Correction, Stage-1 uint16 Caching & Readiness Audit

**Date**: 2026-11-18  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Model Architecture & Loss | DICOM Preprocessing & Ingestion | Dynamic Calibration & Dismissal  
**Milestone Scope**: Final Stage B pre-training engineering & scientific remediation (`Critical Fixes A–I`) ahead of Stage-1 build, 1,200-image L40S Parity Pilot, and 20-epoch RSNA training  
**Status**: CONDITIONAL GO (20/20 LOCAL CPU TESTS PASSED; CLOUD GATES PENDING MANUAL EXECUTION)

---

## 1. Research Objective & Context
- Eliminate every remaining scientific, statistical, and systems-reproducibility gap identified in the final independent pre-training audit before spending cloud credits on full RSNA feature extraction and 20-epoch TRUST/CE training.
- Preserve AnchorShift's strict core design: RSNA (`Emory Healthcare, USA` + `BreastScreen Victoria, Australia`) source-only training & threshold calibration -> frozen transfer to `NLBS` (Canada, zero NLBS labels used during Stage B) vs. UPA (`2023`) vs. TRUST's 20% labeled NLBS reference.

---

## 2. Datasets, Volumes & Manifest References
- **Modal Volume / Storage Path**: `rsna-mammography-vol:/rsna/raw/`, `/cache/stage1_rsna_uint16/`, `/cache/rsna_aug_seed42/`, `/models/mammo_clip/b5-model-best-epoch-7.tar` (NLBS strictly quarantined).
- **Manifest File & Path**: [`datasets/rsna/splits_seed42.csv`](../datasets/rsna/splits_seed42.csv) (`54,706` images, `11,913` unique patients, 100% laterality present: `'R': 27,439`, `'L': 27,267`).
- **Exact RSNA Split Counts (Verified Locally from CSV)**:
  - `train`: `38,251` images across `8,339` unique patients (`807` positive, `37,444` negative; `'R': 19,214`, `'L': 19,037`).
  - `train_calib`: `5,473` images across `1,192` unique patients (`114` positive, `5,359` negative; `'L': 2,745`, `'R': 2,728`).
  - `val_calib`: `10,982` images across `2,382` unique patients (`237` positive, `10,745` negative; `'R': 5,497`, `'L': 5,485`).
- **Workload Correction**: `ceil(37,444 / 60) = 625` batches/epoch (`1,000,000` occurrences across `20` epochs = `8.192 GB` FP32 training embeddings + `134.80 MB` eval embeddings = `8.327 GB` raw tensors).

---

## 3. Forensic Evidence & Diagnostic Findings
1. **Head Initialization Order Drift (Fix A)**: Constructing `MammoCLIPEncoder` before initializing the head in Online mode consumed PyTorch RNG state relative to Cached mode. Resolved with `build_trust_head` saving/restoring CPU & CUDA RNG states.
2. **Legacy 100-Image Parity Pilot Statistical Blind Spot (Fix D)**: Minimum zero-miss dismissed count satisfying one-sided 95% Clopper–Pearson `<= 1%` is `299` (`1 - 0.05^(1/299) = 0.009969 <= 0.01`, whereas `N=298` gives `0.010002 > 0.01`). Old `N_calib = 100` forced `tau_t = 0.0` at every epoch. Redesigned pilot (`sample_pilot_1200`) gives `train_calib = 500` (`20 pos, 480 neg >= 299`) while preserving `0` patient overlap and ~4% positivity.
3. **Micro-Batch Dismissal Loss Bias (Fix E)**: Naive per-micro-batch averaging misweighted `DynamicDismissalLoss` whenever positive counts differed across micro-batches of an effective 80-image batch. Fixed via `TRUSTLoss.forward_microbatch` scaling BCE/Focal by `total_batch_size` and Dismissal by `max(total_pos_in_batch, 1)`.
4. **DICOM I/O Bottleneck (Fix F)**: Raw DICOM decoding (`434.40 s / 1k` on `ap-KidtVEoMzokd7fsFzpMeFo`) across `1,000,000` training occurrences would cost `~120.7 hrs` (`~$285`) on L40S. Adding CPU Stage-1 `uint16` crop caching (`src/data/stage1_cache.py`) decodes each raw DICOM at most once.
5. **Preprocessing Safety & Paper Fidelity (Fix G)**: Fixed `use_largest_cc=False` verbatim default (matching TRUST Sec 3.2), removed silent LUT error swallowing, disallowed silent heuristic left/right guessing (`laterality_policy="strict"`), enforced explicit segmentation status diagnostics (`strict_segmentation=True`), and added real DICOM visual audit `--audit-real`.
6. **Checkpointing, Caching & Environment Pinning (Fixes B, C, H, I)**: Unified Online, Cached, and Parity training onto `src/training/engine.py`; upgraded checkpoints to `anchorshift_trust_ckpt_v2` (22 mandatory config keys, `cache_manifest_sha256`, full RNG/scaler checks); upgraded feature cache to `anchorshift_aug_cache_v2` (`os.fsync(f.fileno())`, precision/stage1/dtype/finiteness/eval ID checks); pinned Python `3.12` + 13 `==` dependencies across `requirements-lock.txt` and all 5 Stage B Modal scripts.

---

## 4. Mathematical Formulations & Methodological Decisions
- **Exact Micro-Batch Gradient Identity**:
  - `L_micro_bce_focal = sum_{i in mb} [l_ce(z_i, y_i) + lambda_focal * l_focal(z_i, y_i)] / total_batch_size`
  - `L_micro_dismiss = lambda_dismiss * sum_{i in mb: y_i=1} max(0, tau_t + m - sigmoid(z_i)) / max(total_pos_in_batch, 1)`
- **Dynamic Threshold Minimum Feasible Calibration Sample Size**:
  - Zero-miss 95% Clopper–Pearson threshold `<= 0.01` requires `1 - 0.05^(1 / N_d) <= 0.01 => N_d >= ceil(ln(0.05) / ln(0.99)) = 299`.

---

## 5. Implementation Details & Exact Changes
- **New Modules**: [`src/training/engine.py`](../src/training/engine.py), [`src/data/stage1_cache.py`](../src/data/stage1_cache.py), [`src/utils/__init__.py`](../src/utils/__init__.py), [`src/utils/env_metadata.py`](../src/utils/env_metadata.py), [`scripts/cache_stage1_rsna_modal.py`](../scripts/cache_stage1_rsna_modal.py), [`reports/final_pre_training_readiness_audit.md`](../reports/final_pre_training_readiness_audit.md).
- **Updated Modules & Tests**: [`src/models/losses.py`](../src/models/losses.py), [`src/data/preprocessing.py`](../src/data/preprocessing.py), [`src/data/feature_cache.py`](../src/data/feature_cache.py), [`src/training/checkpointing.py`](../src/training/checkpointing.py), [`src/training/calibration.py`](../src/training/calibration.py), [`src/training/parity_pilot.py`](../src/training/parity_pilot.py), [`scripts/pilot_parity_modal.py`](../scripts/pilot_parity_modal.py), [`scripts/train_trust_modal.py`](../scripts/train_trust_modal.py), [`scripts/smoke_test_stage_b_modal.py`](../scripts/smoke_test_stage_b_modal.py), [`scripts/benchmark_l40s_comparison_modal.py`](../scripts/benchmark_l40s_comparison_modal.py), [`requirements-lock.txt`](../requirements-lock.txt), [`tests/test_stage_b_smoke.py`](../tests/test_stage_b_smoke.py).

---

## 6. Quality Assurance, Unit Tests & Empirical Smoke Tests
- `python tests/test_stage_b_smoke.py`: **20/20 passed (`OK`, `0` failed, `0` skipped)**.
- `python -m unittest discover -s tests -p "test_*.py" -v`: **20/20 passed (`OK`, `0` failed, `0` skipped)**.
- `python -m py_compile` across all modified Modal scripts: **Exit code 0**.
- Validation checklist:
  - [x] DICOM LUT error propagation, `MONOCHROME1` polarity, right mirroring & `use_largest_cc=False` crop verified locally
  - [x] Zero patient leakage verified on full `splits_seed42.csv` AND `sample_pilot_1200` (`500 / 500 / 200`)
  - [x] Head init invariance (`0.0` diff) & micro-batch positive weighting (`<= 1e-6`) verified
  - [x] Stage-1 `uint16` cache build/verify/resume & bitwise Stage-2 parity verified locally
  - [x] `v2` feature cache & `v2` checkpoint validation verified locally
  - [x] Unlabeled target isolation maintained (zero NLBS data/label exposure)

---

## 7. Remaining Limitations & Open Issues
- CUDA RNG save/resume parity, AMP L40S real Mammo-CLIP parity across `1,200` images, real RSNA visual `--audit-real` DICOM verification, full `54,706`-image Stage-1 build throughput, and 5-batch Stage-1 -> Stage-2 -> L40S timing remain **pending manual user Modal runs** (no cloud jobs executed by assistant).

---

## 8. Reproducibility & Execution Commands
*(See [`reports/final_pre_training_readiness_audit.md`](../reports/final_pre_training_readiness_audit.md) Section 9 for the full ordered 10-step execution runbook matching exact CLI flags.)*

---

## 9. Git & Commit Tracking
- **Git Commit SHA**: Uncommitted local working tree changes (awaiting explicit user instruction to commit per project rule)
- **Remote Synchronization**: Not pushed (awaiting user approval)
