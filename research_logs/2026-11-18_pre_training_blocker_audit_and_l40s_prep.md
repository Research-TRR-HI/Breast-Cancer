# Research Log: Pre-Training Blocker Audit, Sampling Fix & L40S Preparation

**Date**: 2026-11-18  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Training Pipeline Hardening & Pre-Flight Verification  
**Milestone Scope**: Pre-training audit of `Research-TRR-HI/Breast-Cancer` across 6 blockers (strict 852/852 smoke test, epoch sampling policy, L40S GPU selection, TRUST parameter/deviation audit, per-epoch val monitoring + resume, and L40S throughput benchmark)  
**Status**: CODE FIXES COMPLETE / PENDING USER EXECUTION (NOT READY FOR FULL TRAINING UNTIL CHECKS PASS)

---

## 1. Research Objective & Context
- Audit the `Research-TRR-HI/Breast-Cancer` repository prior to launching Stage B4 full 20-epoch training.
- Resolve six concrete engineering and methodological blockers without running any GPU jobs automatically.
- Prepare exact manual execution commands for local unit tests, strict smoke verification, and L40S throughput/cost benchmarking before committing to full 20-epoch runs.

---

## 2. Datasets, Volumes & Manifest References
- **Modal Volume**: `rsna-mammography-vol` mounted at `/data`
- **Splits Manifest**: `datasets/rsna/splits_seed42.csv` (`train`: 38,251 imgs, `train_calib`: 5,473 imgs, `val_calib`: 10,982 imgs; 0 patient leakage)
- **Foundation Checkpoint**: `/data/models/mammo_clip/b5-model-best-epoch-7.tar` (1.65 GB, canonical `efficientnet_custom` B5, 852 keys)

---

## 3. Forensic Evidence & Blocker Audit

### 3.1 Blocker 1 — Strict 852/852 Smoke Test Enforcement
- `scripts/smoke_test_stage_b_modal.py` already verified `strict=True` on Tesla T4 (`ap-YZU56SuJcHFBV9LR17YcnY`: `852/852` matched, `missing=0`, `unexpected=0`, `0.85 GB` VRAM).
- Hardened the assertion to explicitly require `matched_keys_count == 852`, `total_target_keys == 852`, `missing_keys_count == 0`, and `unexpected_keys_count == 0`.

### 3.2 Blocker 2 — Undersampled Negative Epoch Bug (`len(pos_idx) // 20`)
- **Finding**: `scripts/train_trust_modal.py` previously set `batches_per_epoch = len(pos_idx) // 20` (~40 batches/epoch = 2,400 negatives/epoch out of ~37,450 negatives, skipping ~93.6% of negative screening images per epoch!).
- **Fix**: Added `epoch_policy="full_neg_pass"` (default) to both `BalancedBatchSampler` (`src/data/rsna_dataset.py`) and `scripts/train_trust_modal.py`. Each epoch now runs `ceil(N_neg / 60)` batches so 100% of negative images are visited once per epoch, while positives cycle with reshuffled wraparound to preserve the exact 20:60 pos:neg minibatch ratio. Added `ThreadPoolExecutor(max_workers=8)` for parallel DICOM decoding per micro-batch.

### 3.3 Blocker 3 — Broken Modal GPU Selection (`gpu="L4"` hardcoded)
- **Finding**: `scripts/train_trust_modal.py` accepted `--gpu` in `main()`, printed it, and then called `train_model.remote(...)` whose `@app.function` decorator hardcoded `gpu="L4"`, silently ignoring `--gpu L40S`.
- **Fix**: Updated `main()` to validate `gpu` against `SUPPORTED_GPUS` and invoke `train_model.with_options(gpu=gpu_norm).remote(...)` and `benchmark_throughput.with_options(gpu=gpu_norm).remote(...)`. Actual `torch.cuda.get_device_name(0)` is logged and saved in checkpoints and benchmark JSON.

### 3.4 Blocker 4 — TRUST Parameter Verification & Documented Deviations
- **Verified Code Defaults**:
  - `DynamicDismissalLoss(margin=0.05)`
  - `FocalLoss(alpha=0.25, gamma=2.0)`
  - `TRUSTLoss(lambda_focal=1.0, lambda_dismiss=1.0)`: $L_{\text{total}} = L_{\text{bce}} + 1.0 \cdot L_{\text{focal}} + 1.0 \cdot L_{\text{dismiss}}$
  - Dynamic $\tau_t$ initialized at `0.5` and updated at the end of each epoch on `train_calib` positive scores at the $(1 - 0.98) \times 100 = 2\text{nd}$ percentile.
- **Documented Deviations / Verification Limits**:
  1. `arXiv:2609.00300` PDF is not stored locally (the workspace PDF by Mohandas et al. 2026 is a different paper sharing the acronym); scalar defaults rely on the prior Section 3 audit and are exposed as CLI flags (`--margin`, `--lambda-focal`, `--lambda-dismiss`, `--tau-target-recall`).
  2. $\tau_t$ is computed at **image level** on `train_calib` positives (matching image-level `DynamicDismissalLoss` inputs $p_i$), whereas `val_calib` evaluation aggregates by **case-level max** (`patient_id`).
  3. `full_neg_pass` replaces `pos_pass` to ensure 100% negative coverage per epoch.

### 3.5 Blocker 5 — Per-Epoch `val_calib` Monitoring, Checkpointing & Resume
- Added batched `_evaluate_split()` running on full `val_calib` (10,982 images, case-level max pooling by `patient_id`) at the end of every epoch, logging `val_loss`, `val_D@98`, `CP99`, and `val_D@95`.
- Saves `{baseline}_rsna_seed{seed}_latest.pt` every epoch (head, optimizer, scaler, `tau_t`, `history`, `best_val_dismiss_98`, numpy/torch RNG states) and `{baseline}_rsna_seed{seed}_best.pt` on improved `val_D@98`.
- Added `--resume` flag to seamlessly continue from `_latest.pt`.

### 3.6 Blocker 6 — L40S Throughput Benchmark & Cost Projection
- Added `benchmark_throughput` (`--benchmark --gpu L40S --benchmark-batches 10`) to `scripts/train_trust_modal.py`.
- Measures real `sec_per_batch_80`, `train_imgs_per_sec`, `eval_imgs_per_sec`, and `peak_vram_gb`, then projects 20-epoch runtime and Modal cost (`projected_hours * hourly_rate`) and saves `/data/models/benchmark_throughput.json`.

---

## 4. Implementation Deliverables
- `scripts/smoke_test_stage_b_modal.py`: Explicit `852/852` parity assertion.
- `src/data/rsna_dataset.py`: `BalancedBatchSampler` with `epoch_policy="full_neg_pass"` default.
- `scripts/train_trust_modal.py`: Dynamic `.with_options(gpu=...)`, `full_neg_pass` epoch loop, parallel DICOM decode, per-epoch `val_calib` eval, `_latest.pt`/`_best.pt` + `--resume`, and `--benchmark` mode.
- `tests/test_stage_b_smoke.py`: Added `test_balanced_batch_sampler_full_neg_pass`.
- `reports/trust_reproduction_plan.md`: Updated with `full_neg_pass`, `L40S`, and documented deviations.

---

## 5. Next Required Manual Checks (Before Full Training)
Full training is **NOT YET READY** until the user executes and verifies:
1. Local unit tests: `python -m pytest tests/test_stage_b_smoke.py -v`
2. Strict T4 smoke test (if re-verifying): `modal run scripts/smoke_test_stage_b_modal.py`
3. L40S throughput & cost benchmark: `modal run scripts/train_trust_modal.py --benchmark --gpu L40S --benchmark-batches 10`
