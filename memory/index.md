# Workspace Index

## Directory Structure & Detailed File Context

### Root (`d:\Breast Cancer Research`)
- `.gitignore`: Ignores Python cache, `.env`/secret keys, raw medical image/archive files (`*.dcm`, `*.zip`, `*.png`, etc.), model checkpoints, and local `git commit.md` while keeping `datasets/` manifest CSVs tracked.
- `.agents/rules/git-commit-rule.md`: Project Git commit rule enforcing one file per commit and strictly alternating author/committer identities between Tahsin Raihan Robbani and Hasin Ishrak (excluding `git commit.md`).
- `AnchorShift_Research_Context.md`: 531-line master research brief covering clinical triage task, strict source-only vs. UPA vs. target-label calibration protocols, TRUST (arXiv:2609.00300) background, dataset caveats, Stage A–D design, and guardrails.
- `requirements-lock.txt`: Pinned Python 3.12 / package versions (`torch==2.5.1`, `numpy==2.5.3`, `pandas==2.3.3`, `scipy==1.17.1`, `pydicom==3.0.2`, `opencv-python-headless==4.11.0.86`, `pillow==12.3.0`, etc.).

### `datasets/`
- `rsna/manifest.csv`: Deterministic 54,706-row image manifest for RSNA Screening Mammography (`11,913` patients).
- `rsna/splits_seed42.csv`: Zero-leakage patient-level RSNA splits (`38,251` train, `5,473` train_calib, `10,982` val_calib).
- `nlbs/manifest.csv`: Deterministic 26,988-row image manifest for NLBS enriched with scanner, photometric, and cluster metadata.
- `nlbs/exam_manifest.csv`: 6,430-row examination-level NLBS manifest with `5,942` Union-Find `split_cluster_id` components.

### `src/`
- `models/efficientnet_custom.py` & `models/efficient_net_custom_utils.py`: Canonical Mammo-CLIP EfficientNet-B5 implementation matching all 852 tensors of `b5-model-best-epoch-7.tar`.
- `models/trust_model.py`: `MammoCLIPEncoder`, `TRUSTClassifier`, `aggregate_case_scores` (with strict equal-length validation), and `aggregate_case_labels`.
- `models/losses.py`: Exact TRUST Eq. 6–8, 12–14 losses: `FocalLoss` (`gamma=2.0`, unweighted `alpha=None`), `DynamicDismissalLoss` (`margin=0.10`), and `TRUSTLoss` (`lambda_focal=1.0`, `lambda_dismiss=0.05`, supporting `ce`, `ce_focal`, `fixed_tau`, `trust`).
- `models/mammo_clip_loader.py`: Pinned `shawn24/Mammo-CLIP` revision `2f356926a00fc3f0d9fdad1193c1464fd9adf564`, expected SHA-256 `0baccb0c...6fd02c`, strict `852/852` loader & checksum verifier.
- `data/preprocessing.py`: Full TRUST Sec. 3.2 pipeline: DICOM decode + Modality/VOI LUTs, `MONOCHROME1` inversion, right-breast horizontal mirroring, border-median breast segmentation + hole filling + `9x9` morphological closing + `10%` bbox crop + 16-bit quantization, `1520x912` bilinear resize, `[0,1]` re-norm, seeded stochastic training augmentation, 3ch Mammo-CLIP standardization, and config hashing.
- `data/splits.py`: Stratified patient-level split generator for RSNA.
- `data/rsna_dataset.py`: `RSNADataset`, unified `BalancedBatchSampler` (defaulting to `paper_independent` 20 pos + 60 neg with `get_state`/`set_state`), and `build_epoch_draw_schedule`.
- `data/feature_cache.py`: Sharded augmentation-aware FP32 occurrence cache (`(epoch, batch_idx, slot_idx)`), atomic shard writes, SHA-256 checksums, hash guards, duplicate/missing draw checks, and interrupted extraction recovery.
- `training/calibration.py`: `clopper_pearson_upper_bound` (exact Eq. 5 with `n=0 -> 1.0`, `k=n -> 1.0`, input validation), `compute_dynamic_training_threshold` (CP95 `<= 1%` on image-level `train_calib`, `0.0` fallback), `compute_target_recall_threshold` (tie-safe discrete order statistic), and `evaluate_dismissal_at_threshold` (CP99 & `N/A` reporting).
- `training/checkpointing.py`: Complete atomic save/restore of head, optimizer, scaler, epoch, `tau_t`, history, Python/NumPy/Torch CPU/CUDA RNGs, sampler RNG, augmentation RNG, and training config; raises on missing `--resume` checkpoint.
- `training/parity_pilot.py`: Controlled Online-vs-Cached equivalence pilot verifying all 10 numerical & clinical agreement metrics.

### `reports/`
- `trust_paper_implementation_traceability.md`: Complete traceability matrix mapping `arXiv:2609.00300v1` equations, parameters, preprocessing, sampling, and thresholds to codebase implementation.
- `pre_training_go_no_go.md`: Pre-training Go/No-Go audit (`CONDITIONAL GO`), corrected resource/cost calculations, test results, and exact manual cloud commands.
- `trust_reproduction_plan.md`: Updated v2.0 Stage B reproduction & validation plan aligned with `arXiv:2609.00300v1`.
- `dataset_audit.md`, `audit_results.json`, `stage_a3_nlbs_results.json`, `discovery_summary.json`: Stage A & A3 forensic audit reports and JSON outputs.

### `scripts/`
- `download_mammoclip_modal.py`: CPU-only Modal downloader pinned to HuggingFace commit `2f356926a00fc3f0d9fdad1193c1464fd9adf564` with SHA-256 verification.
- `smoke_test_stage_b_modal.py`: Tesla T4 strict `852/852` smoke test using updated LUT/crop preprocessing and corrected TRUST parameters.
- `pilot_parity_modal.py`: 1,000-image L40S Online-vs-Cached Mammo-CLIP equivalence pilot (`fp32` + `amp`).
- `train_trust_modal.py`: Corrected Modal training & `--extract-cache` pipeline (final 20th-epoch selection, CP95 `tau_t`, full RNG resume, dynamic `.with_options(gpu=...)`).
- `benchmark_l40s_comparison_modal.py`: Historical 1,000-image L40S deterministic inference/encoding vs. live throughput benchmark (`ap-KidtVEoMzokd7fsFzpMeFo`).
- `stage_a3_nlbs_extract_modal.py`, `audit_datasets_modal.py`, `download_nlbs_modal.py`, `download_rsna_modal.py`: Stage A/A3 CPU-only Modal scripts.

### `research_logs/`
- `README.md`, `TEMPLATE.md`, `2026-11-17_*.md`, `2026-11-18_stage_b_gpu_smoke_test_and_mammoclip_verification.md`, `2026-11-18_pre_training_blocker_audit_and_l40s_prep.md`, `2026-11-18_l40s_sweep_and_caching_benchmark.md`, and `2026-11-18_pre_training_correction_and_trust_audit.md`.

### `tests/`
- `test_stage_b_smoke.py`: 13-test local CPU suite covering Categories A–G (Model, Losses, Calibration, Preprocessing, Dataset/Sampler, Caching, Online-vs-Cached Parity & Checkpoint Resume).
