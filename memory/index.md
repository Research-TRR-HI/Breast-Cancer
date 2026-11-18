# Workspace Index

## Directory Structure & Detailed File Context

### Root (`d:\Breast Cancer Research`)
- `.gitignore`: Ignores Python cache, `.env`/secret keys, raw medical image/archive files (`*.dcm`, `*.zip`, `*.png`, etc.), model checkpoints, and local `git commit.md` while keeping `datasets/` manifest CSVs tracked.
- `.agents/rules/git-commit-rule.md`: Project Git commit rule enforcing one file per commit and per-task GitHub author/committer identity consistency between Tahsin Raihan Robbani and Hasin Ishrak (excluding `git commit.md`).
- `AnchorShift_Research_Context.md`: 531-line master research brief covering clinical triage task, strict source-only vs. UPA vs. target-label calibration protocols, TRUST (arXiv:2609.00300) background, dataset caveats (RSNA Emory USA + BreastScreen Victoria Australia; NLBS Canada), Stage A–D design, and guardrails.
- `requirements-lock.txt`: Exact `==` pinned Python 3.12 direct dependencies matching `src/utils/env_metadata.py` (`torch==2.5.1`, `torchvision==0.20.1`, `numpy==2.5.3`, `pandas==2.3.3`, `scipy==1.17.1`, `pydicom==3.0.2`, `opencv-python-headless==4.11.0.86`, `pillow==12.3.0`, `omegaconf==2.3.0`, `pylibjpeg==2.0.0`, `pylibjpeg-libjpeg==2.1.0`, `pylibjpeg-openjpeg==2.1.0`, `python-gdcm==3.0.24`).

### `datasets/`
- `rsna/manifest.csv`: Deterministic 54,706-row image manifest for RSNA Screening Mammography (`11,913` patients).
- `rsna/splits_seed42.csv`: Zero-leakage patient-level RSNA splits (`train`: `38,251` imgs / `807` pos / `37,444` neg; `train_calib`: `5,473` imgs / `114` pos / `5,359` neg; `val_calib`: `10,982` imgs / `237` pos / `10,745` neg; 100% laterality).
- `nlbs/manifest.csv`: Deterministic 26,988-row image manifest for NLBS enriched with scanner, photometric, and cluster metadata.
- `nlbs/exam_manifest.csv`: 6,430-row examination-level NLBS manifest with `5,942` Union-Find `split_cluster_id` components.

### `src/`
- `models/efficientnet_custom.py` & `models/efficient_net_custom_utils.py`: Canonical Mammo-CLIP EfficientNet-B5 implementation matching all 852 tensors of `b5-model-best-epoch-7.tar`.
- `models/trust_model.py`: `MammoCLIPEncoder`, `TRUSTClassifier`, `aggregate_case_scores` (strict validation), and `aggregate_case_labels`.
- `models/losses.py`: Exact TRUST Eq. 6–8, 12–14 losses (`FocalLoss` `gamma=2.0`, `alpha=None`; `DynamicDismissalLoss` `margin=0.10`; `TRUSTLoss` `lambda_focal=1.0`, `lambda_dismiss=0.05`) plus exact effective-batch `forward_microbatch` scaling BCE/Focal by `total_batch_size` and Dismissal by `max(total_pos_in_batch, 1)`.
- `models/mammo_clip_loader.py`: Pinned `shawn24/Mammo-CLIP` revision `2f356926a00fc3f0d9fdad1193c1464fd9adf564`, SHA-256 `0baccb0c...6fd02c`, strict `852/852` loader & checksum verifier.
- `data/preprocessing.py`: Full TRUST Sec. 3.2 pipeline (`PREPROCESS_PIPELINE_VERSION = "trust_sec3_2_v2"`): strict Modality/VOI LUT propagation, `MONOCHROME1` inversion, `laterality_policy="strict"` right-breast mirroring, `use_largest_cc=False` border-median segmentation + hole fill + `9x9` closing + `10%` crop + explicit segmentation diagnostics (`strict_segmentation=True`) + `uint16` quantization, `1520x912` bilinear resize, `[0,1]` norm, seeded augmentation, 3ch Mammo-CLIP standardization, Stage-1 & Stage-2 config hashing.
- `data/stage1_cache.py`: CPU Stage-1 deterministic lossless `uint16` cropped image cache (`anchorshift_stage1_uint16_v1`), atomic `os.fsync` shards, pixel SHA-256 verification, and `Stage1CroppedLoader` LRU cache.
- `data/splits.py`: Stratified patient-level split generator for RSNA.
- `data/rsna_dataset.py`: `RSNADataset`, `BalancedBatchSampler` (`paper_independent` `20:60` defaulting to `ceil(37,444/60)=625` batches/ep, `get_state`/`set_state`), and `build_epoch_draw_schedule`.
- `data/feature_cache.py`: `v2` augmentation-aware sharded FP32 occurrence cache (`anchorshift_aug_cache_v2`), fixed `os.fsync(f.fileno())`, `get_cache_manifest_sha256`, `stage1_config_hash`, `encoder_precision`, strict dtype/shape/`torch.isfinite`/binary-label/eval-ID validation, and crash recovery.
- `training/engine.py`: Shared production training engine used by both `train_trust_modal.py` and `parity_pilot.py`, including outer-RNG-isolated `build_trust_head`, `seed_dropout_step_rng`, and `run_effective_batch_step`.
- `training/calibration.py`: `clopper_pearson_upper_bound` (exact Eq. 5), `compute_dynamic_training_threshold_with_diagnostics` (CP95 `<= 1%` image-level `train_calib`, `min_zero_miss_dismissed_required=299`, `0.0` fallback), `compute_target_recall_threshold`, and `evaluate_dismissal_at_threshold` (CP99).
- `training/checkpointing.py`: Strict `v2` atomic checkpointing (`anchorshift_trust_ckpt_v2`) requiring all 22 `MANDATORY_CONFIG_KEYS`, 15 `MANDATORY_PAYLOAD_KEYS`, `cache_manifest_sha256`, full Python/NumPy/Torch CPU/CUDA/sampler/augmentation RNG states, and `os.fsync`.
- `training/parity_pilot.py`: `sample_pilot_1200` (`500 train / 500 train_calib [480 neg >= 299] / 200 val_calib`, 0 patient overlap) and controlled Online-vs-Cached pilot via shared `engine.py`.
- `utils/env_metadata.py`: Pinned Python `3.12` + 13 `==` package constants and `collect_runtime_environment_metadata`.

### `reports/`
- `final_pre_training_readiness_audit.md`: Definitive 10-section final pre-training readiness audit (`CONDITIONAL GO`), Critical Fixes A–I table, `20/20` local test matrix, `625`-batch/`1M`-occurrence (`8.327 GB`) cache analysis, stop/go gates, and exact 10-command manual runbook.
- `pre_training_go_no_go.md`, `trust_paper_implementation_traceability.md`, `trust_reproduction_plan.md`: Updated v2.1 traceability & reproduction specs aligned with `arXiv:2609.00300v1`.
- `dataset_audit.md`, `audit_results.json`, `stage_a3_nlbs_results.json`, `discovery_summary.json`: Stage A & A3 forensic audit reports.

### `scripts/`
- `cache_stage1_rsna_modal.py`: Pinned CPU Modal script for `--audit-real` visual/segmentation inspection and `--build`/`--verify` of lossless Stage-1 `uint16` crops.
- `pilot_parity_modal.py`: Pinned 1,200-image L40S Online-vs-Cached Mammo-CLIP parity pilot (`fp32` + `amp`).
- `train_trust_modal.py`: Pinned Modal training & extraction pipeline integrating `Stage1CroppedLoader`, `--benchmark-extract-batches N`, `--extract-cache`, `--verify-cache`, `micro_batch_size=80`, shared `engine.py`, and `v2` checkpointing.
- `smoke_test_stage_b_modal.py`, `benchmark_l40s_comparison_modal.py`: Pinned Stage B Tesla T4 smoke test (explicit CSV laterality) & historical L40S benchmark scripts.
- `download_mammoclip_modal.py`, `stage_a3_nlbs_extract_modal.py`, `audit_datasets_modal.py`, `download_nlbs_modal.py`, `download_rsna_modal.py`: Completed Stage A/B2 scripts.

### `research_logs/`
- `README.md`, `TEMPLATE.md`, and dated research logs including `2026-11-18_final_pre_training_readiness_correction.md`.

### `tests/`
- `test_stage_b_smoke.py`: 20-test local CPU suite across Categories A–H (`20/20` passing via direct execution and `unittest discover`).
