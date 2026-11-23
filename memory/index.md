# Workspace Index

## Two-Paper Program Structure & Context Documents
This repository (`https://github.com/Research-TRR-HI/Breast-Cancer`) houses **two separate but related breast-cancer screening AI studies** built atop shared RSNA/NLBS data, Modal infrastructure, and baseline CE/TRUST artifacts:

- [`AnchorShift_Research_Context.md`](../AnchorShift_Research_Context.md): **Paper 2 (AnchorShift)** — 531-line master research brief (dated 17 Nov 2026) covering zero-target-label transfer vs. UPA (2023) vs. TRUST (2026) target-calibrated reference, Stage A–D methodology, and guardrails. **Keep intact**; note that its Nov 17 "no experiments run yet" statements are superseded by `memory/state.md` and `reports/stage_b/` + `reports/stage_c/`.
- [`Paper1_External_Validation_Research_Context.md`](../Paper1_External_Validation_Research_Context.md): **Paper 1 (Multi-Model External Validation Benchmark)** — 928-line research design brief (dated 21 Nov 2026) defining the 6-model source-frozen `95%/98%` dismissal comparison (`ce_b5`, `trust_b5`, `gmic_pretrained`, `nyu_fourview_pretrained`, `dinov2_rsna`, `resnet50_rsna`), mandatory cohort/source-test blockers, novelty/literature audit (*Meta-Repository* 2026 & *Sensors* June 2026), adversarial reviewer checklist, and hard **\$30** Modal L40S budget cap.

---

## Directory Structure & Detailed File Context

### Root (`d:\Breast Cancer Research`)
- `.gitignore`: Ignores Python cache, `.env`/secret keys, `memory/personal_info.md`, raw medical images/archives (`*.dcm`, `*.zip`, `*.png`, etc.), `*.pt`/`*.ckpt`/`*.p`/`*.h5`/`*.hdf5` model weights, `external/` cloned repos, `logs/`, duplicate `48MB` `stage1_manifest.json` files, and local `git commit.md`, while keeping `datasets/` & small report manifests tracked.
- `.agents/rules/git-commit-rule.md`: Strict Git commit rule (`1 file = 1 commit`, never commit `git commit.md`, alternate consistent author/committer identity per task between `Tahsin Raihan Robbani` and `Hasin Ishrak`).
- `requirements-lock.txt`: Exact `==` pinned Python `3.12` direct dependencies matching `src/utils/env_metadata.py` (`torch==2.5.1`, `torchvision==0.20.1`, `numpy==2.5.3`, `pandas==2.3.3`, `scipy==1.17.1`, `pydicom==3.0.2`, `opencv-python-headless==4.11.0.86`, `pillow==12.3.0`, `omegaconf==2.3.0`, `pylibjpeg==2.0.0`, `pylibjpeg-libjpeg==2.5.0`, `pylibjpeg-openjpeg==2.6.0`, `python-gdcm==3.0.24.1`).

### `memory/` *(Shared Persistent Agent Memory)*
- `memory/index.md`: Workspace map registering both papers, context briefs, shared modules/datasets, immutable baseline results, and planned paper directories.
- `memory/purpose.md`: Two-paper research mission, scientific boundary rules (anti-salami-slicing), budget/quarantine rules, and mandatory 8-step future-agent startup protocol.
- `memory/state.md`: Live unified progress tracker split into Shared Infrastructure (Stages A/B/C complete), Paper 1 track (blockers, 6-model status, \$30 budget cap), and Paper 2 track (Stage D UPA pending).
- `memory/history.md`: Chronological session & milestone log (always prepend newest entries at top; preserves all historical decisions and findings).
- `memory/personality.md`: Preserved developer communication voice & persona guidelines (`personal_info.md` remains gitignored locally).

### `datasets/` *(Shared Manifests — Raw DICOMs on Modal Volumes Only)*
- `rsna/manifest.csv`: Deterministic `54,706`-row image manifest for RSNA Screening Mammography (`11,913` patients; `486` cancer-positive).
- `rsna/splits_seed42.csv`: Original `70/10/20` patient-level RSNA splits used by historical Stage B CE/TRUST runs (`train`: `38,251` imgs / `8,339` pts; `train_calib`: `5,473` imgs / `1,192` pts; `val_calib`: `10,982` imgs / `2,382` pts / `97` pos; note Paper 1 Blocker 2 addresses separating `source_op_calib` from `source_heldout_test`).
- `nlbs/manifest.csv`: Deterministic `26,988`-row NLBS image manifest enriched with scanner, photometric (`100% MONOCHROME2`, `12-bit`), `StudyInstanceUID`, and `split_cluster_id`.
- `nlbs/exam_manifest.csv`: `6,430`-row `(case_id, StudyInstanceUID)` manifest resolving to `6,359` unique `StudyInstanceUID` studies (`151` positive exams) and `5,942` connected `split_cluster_id` components (`149` positive clusters; see Paper 1 Blocker 1 for reconciliation with published `5,997 / 5,935 / 149` counts).
- `nlbs/inference_manifest_stage_c.csv`: Strictly label-free Stage C inference manifest (`26,988` images, sorted by `StudyInstanceUID`, explicitly excluding `Cancer`, `exam_cancer`, `n_cancer_images`, `Hard to State Negative`, `cohort_group`, `sha256: c3d6a3b3...`).

- `datasets/paper1/`: Paper 1 Gate 0.2 frozen manifests (`rsna_val_calib_2fold_seed42.csv`, `rsna_val_calib_4view_selection.csv`, `nlbs_4view_study_eligibility.csv`, `nlbs_4view_common_cohort_label_free.csv`).

### `src/` *(Shared Verified Pipeline Modules)*
- `paper1/`: Paper 1 model adapters, preprocessors & evaluation (`gmic_adapter.py`, `nyu_fourview_adapter.py`, `dinov2_model.py`, `resnet50_model.py`, `six_model_eval.py`).
- `models/efficientnet_custom.py` & `models/efficient_net_custom_utils.py`: Canonical Mammo-CLIP EfficientNet-B5 backbone matching `852/852` tensors of `b5-model-best-epoch-7.tar`.
- `models/trust_model.py`: `MammoCLIPEncoder`, `TRUSTClassifier`, `aggregate_case_scores` (max aggregation + strict checks), and `aggregate_case_labels`.
- `models/losses.py`: Exact TRUST losses (`FocalLoss` `gamma=2.0`, `alpha=None`; `DynamicDismissalLoss` `m=0.10`, `lambda_dismiss=0.05`; `TRUSTLoss.forward_microbatch`).
- `models/mammo_clip_loader.py`: Pinned `shawn24/Mammo-CLIP` commit `2f356926a00fc3f0d9fdad1193c1464fd9adf564` & SHA-256 `0baccb0c...6fd02c`.
- `data/preprocessing.py`: TRUST Sec. 3.2 pipeline (`PREPROCESS_PIPELINE_VERSION = "trust_sec3_2_v2"`): LUTs, `MONOCHROME1` inversion, strict right-breast mirroring, `use_largest_cc=False` border-median crop + `uint16` quantization, `1520x912` bilinear resize (with identity skip fast-path), standardization, and config hashes.
- `data/stage1_cache.py`: Stage-1 lossless `uint16` crop cache supporting both `.pt` shards and multi-worker `png16_individual`.
- `data/splits.py` & `data/rsna_dataset.py`: RSNA stratified patient-level splitting, `RSNADataset`, `BalancedBatchSampler` (`paper_independent` `20:60`, `625` batches/ep), `build_epoch_draw_schedule`.
- `data/feature_cache.py`: `v2` sharded augmentation-aware FP32 occurrence cache (`anchorshift_aug_cache_v2`, `640` shards for `1,000,000` occurrences, SHA-256 verification, crash recovery, Unicode progress logging).
- `training/engine.py`, `training/calibration.py`, `training/checkpointing.py`, `training/parity_pilot.py`: Shared training loop, RNG-isolated `build_trust_head`, Clopper-Pearson CP95/CP99 bounds & exact order-statistic threshold selection, strict `v2` checkpointing, and `1,200`-img parity pilot.
- `evaluation/stage_c_eval.py`: Target-label quarantine builder/enforcer, cryptographic prediction locking (`anchorshift_stage_c_locked_predictions_v1`), hash-verified outcome join, AUROC/AUPRC, cluster-collapsed check, and `B=2,000` `split_cluster_id` bootstrap CIs.
- `utils/env_metadata.py`: Pinned Python `3.12` / 13-package metadata constants & runtime collector.

### `scripts/`
- **Paper 1 Gate 0.2 & Four-Model Pipelines:** `prepare_paper1_gate02.py`, `paper1_modal_smoke_and_bench.py`, `preprocess_paper1_gmic_cpu.py`, `run_paper1_gmic_modal.py`, `evaluate_paper1_gmic_step3.py`, `preprocess_paper1_nyu_cpu.py`, `run_paper1_nyu_modal.py`, `run_paper1_dinov2_modal.py`, `run_paper1_resnet50_modal.py`, `train_paper1_dinov2_head.py`, `evaluate_paper1_six_models.py`.
- **Shared / Stage A–C Executed:** `download_{rsna,nlbs,mammoclip}_modal.py`, `audit_datasets_modal.py`, `stage_a3_nlbs_extract_modal.py`, `smoke_test_stage_b_modal.py`, `benchmark_l40s_comparison_modal.py`, `cache_stage1_rsna_modal.py`, `pilot_parity_modal.py`, `train_trust_modal.py`, `run_stage_b_pipeline.py`, `build_nlbs_inference_manifest.py`, `eval_nlbs_stage_c_modal.py` (2-phase CPU Stage-1 -> NVMe L40S inference), `evaluate_locked_nlbs_stage_c.py`.

### `reports/` & `research_logs/` *(Shared Evidence & Immutable Historical Results)*
- **Immutable Baseline Stage B & C Outputs:**
  - `reports/stage_b/rsna_{ce,trust}_eval_results.json`: Historical 20-epoch RSNA `val_calib` calibration/eval results (`hasinishrak74001`, Stage-2 `640`-shard cache SHA-256 `6c34a8c0...`).
  - `reports/stage_c/locked_nlbs_predictions.csv`, `locked_nlbs_image_diagnostics.csv.gz`, `locked_predictions_manifest.json`, `nlbs_zero_label_results.{json,md}`: Immutable 20 Nov 2026 frozen-threshold NLBS zero-label test results (`6,359` studies, `26,988/26,988` decoded cleanly).
  - `reports/modal_archive/{hasinishrak2015,hasinishrak74001}/`: Archived benchmark JSONs, `16`-DICOM real audit PNGs, manifests, and canonical `splits_seed42.csv`. *(Trained `.pt` checkpoints, `eval_*_fp32.pt` feature tensors, and `logs/modal/` logs are archived locally in gitignored `models/` & `logs/`.)*
- **Design & Audit Docs:** `reports/dataset_audit.md`, `reports/trust_paper_implementation_traceability.md`, `reports/trust_reproduction_plan.md`, `reports/pre_training_go_no_go.md`, `reports/final_pre_training_readiness_audit.md`.
- **Research Logs (`research_logs/`):** Indexed chronological logs (`README.md`, `TEMPLATE.md`, 12 dated logs covering Stages A, A3, B1–B4, C, and Paper 1 Stages 0 & 1 — preserve immutable, tag future logs by `[Paper 1]` / `[Paper 2]` / `[Shared]`).
- **Paper 1 Protocol & Reports (`studies/paper1/`, `reports/paper1/`):**
  - `studies/paper1/`: Frozen Paper 1 study docs (`README.md`, `protocol_v1.md`, `model_manifest.yaml`, `eval_config.yaml`, `analysis_plan.md`).
  - `reports/paper1/stage0_readiness_audit.md`: Paper 1 Stage 0 CPU-only readiness audit (**CONDITIONAL GO**) covering NLBS `5,997 / 5,935 / 6,430 / 6,359 / 5,942` & `6,184` 4-view reconciliation, zero-retrain `val_calib` 2-fold source-test redesign, CE/TRUST `$0` reuse verification, pretrained overlap/license matrix, 4-model implementation specs, and `$17.50–$28.00` Modal budget strategy.
  - `reports/paper1/ce_trust_{rsna_source_eval,nlbs_common_4view}.{json,md}`: Gate 0.2 CPU CE/TRUST 2-fold OOF RSNA source metrics & common 4-view NLBS (`N=6,184`) evaluations.
  - `reports/paper1/l40s_smoke_and_benchmark.json`: Live `hasinishrak2015` NVIDIA L40S smoke & throughput benchmark across `gmic_pretrained`, `nyu_fourview_pretrained`, `dinov2_rsna`, and `resnet50_rsna`.
  - `reports/paper1/dinov2_nvme_benchmark.json`: NVMe-staged `512`-image L40S DINOv2 throughput benchmark (`74.2 img/s` end-to-end, `ap-acCim6bfGllCdJjTKtPDv6`).
  - `reports/paper1/stage1_model_readiness.md`: Paper 1 Stage 1 Readiness Audit (**GO**) consolidating Gate 0.2 manifests, model provenance/SHA-256 hashes, `sample_data` parity, L40S benchmarks, and full execution CLI commands.
  - `reports/paper1/stage1_final_corrections.md`: Paper 1 Stage 1 Final Critical Corrections (**GO**) documenting ResNet-50 `scaler.unscale_` + RNG resume, DINOv2 NVMe speedup, NYU `patient_id=71` non-warping fallback root cause, 6-model paired evaluation suite, and recalculated `~$13.15` total run plan.
  - `reports/paper1/gmic_cpu_{bench_200,build_summary,verify_summary}.json` & `reports/paper1/gmic_step1_preprocessing_report.md`: GMIC Step 1 CPU-only preprocessing receipts (`10,982` RSNA + `26,988` NLBS verified `(2944x1920, uint16)` PNGs, `$1.5844` build spend, `0.0` parity diff, Step 2 ready).
  - `reports/paper1/gmic_step2_gpu_pilot.json` & `reports/paper1/gmic_step2_gpu_pilot_report.md`: GMIC Step 2 L40S `400`-image pilot receipts (`32.52 img/s` e2e, `1.511 GB` VRAM, `$0.771` full-run projection <= `$2` budget, `80` RSNA fallbacks + `5` supplemental views audited clean).
  - `reports/paper1/gmic_step2_full_inference_report.md` & `reports/paper1/predictions_locked/gmic_{rsna_val_calib,nlbs}_*`: GMIC Step 2 full `batch_size=1` locked predictions (`10,982` RSNA / `2,382` pts + `26,988` NLBS / `6,359` studies = `37,970` images, SHA-256 provenance locks `fe65b68f...` / `4c84005c...`, **`$1.8101 USD`** settled L40S spend).
  - `reports/paper1/gmic_step3_eval/`: Immutable CPU-only GMIC Step 3 RSNA calibration (`gmic_rsna_calibration.{json,md}`, `5bae9b54...`), pre-join SHA-locked NLBS CSV (`locked_nlbs/gmic_pretrained_*`), full vs common 4-view (`N=6,184`) comparison vs OUR CE & TRUST (`gmic_step3_nlbs_evaluation.{json,md}`, `ce_trust_gmic_comparison_table.csv`).
  - `reports/paper1/nyu_step1_cpu_pilot.{json,_report.md}` (`nyu_step1_cpu_pilot.json` & `nyu_step1_cpu_pilot_report.md`): NYU Step 1 CPU-only `320`-image pilot receipts (`0.0` 10-TTA parity, `26` fallbacks logged, initial resource comparison).
  - `reports/paper1/nyu_step1_config_validation.{json,_report.md}` (`nyu_step1_config_validation.json` & `nyu_step1_config_validation_report.md`): `2vCPU/8GiB` `160`-image validation (`5.117 GB` RSS, `1.80 img/s`), live SHA-256 tamper-resume pass, guarded `--stage build` check, exact `968.214 GiB` volume walk vs `1 TiB` free tier, and `~$0.869` compute + `137.00 GiB` storage projections.
  - `reports/paper1/nyu_step1_rsna_checkpoint_report.md`, `nyu_step1_rsna_val_calib_verify_summary.json`, `nyu_step1_rsna_val_calib_fallback_log.json`, `nyu_step1_rsna_val_calib_manifest.csv`: NYU Step 1 full RSNA `val_calib` verified CPU cache receipts (`9,528` imgs / `2,382` 4-view exams, `69.779 GiB`, `66` fallbacks, `$0.6086` gross, `0` GPU; `nlbs_4view` paused before launch at cost/storage stop gate).
  - `reports/paper1/nyu_step1_storage_and_budget_audit.md`: Complete 18-volume `hasinishrak2015` inventory (`1,271.340 GiB`), verified `USD 0.0030/GiB-day` overage rate above `1,024 GiB`, `2026-12-01` cycle reset / `USD 8.66` credit runway, Tier 1 (`3.022 GiB`) + Tier 2 (`232.075 GiB`) cleanup candidates, and safest low-cost NLBS build plan.
  - `reports/paper1/tier2a_llm_volume_provenance.json` & `reports/paper1/nyu_step1_cleanup_execution_report.md`: Cryptographic HuggingFace provenance & complete file inventory for the `9` deleted Tier 2A LLM volumes (`170.341 GiB`) + `7` Tier 1 pilot dirs (`3.022 GiB`), post-cleanup `9`-volume verification (`1,097.976 GiB`), and revised NLBS budget before launch approval.
- **Planned Paper-Specific Folders *(NOT YET CREATED — add when corresponding tracks execute)*:**
  - `studies/anchorshift/`, `reports/anchorshift/` (Paper 2 Stage D UPA, Protocol 3 oracle reference, any candidate adaptation experiments).

### `tests/`
- `tests/test_stage_b_smoke.py`: 20-test local CPU suite across Categories A–H (`20/20` PASS).
- `tests/test_stage_c_quarantine_and_eval.py`: 3-test suite verifying forbidden-column quarantine, tamper-abort SHA-256 verification, and cluster CI evaluation (`3/3` PASS).
- `tests/test_paper1_models_and_gate02.py`: 11-test suite verifying Gate 0.2 splits/quarantine, `GMIC` & `NYU Four-View` official `sample_data` parity, `DINOv2` frozen forward + CPU head training/locking smoke test, `ResNet-50` AMP unscale+clip & bit-exact resume, 6-model paired statistical/tamper evaluation, native-scale crop fallback, lossless uint16 PNG parity, and GMIC shard resume provenance validation/locking.
- `tests/test_paper1_gmic_step3.py`: 3-test suite verifying Step 3 RSNA tie-handling thresholds, cryptographic verification of all 37,970 Step 2 locked predictions, tamper rejection prior to NLBS label join, and synthetic 3-model cohort matching.
- `tests/test_paper1_nyu_step1_cpu.py`: 5-test suite verifying strict NYU 4-view manifest validation (rejecting missing/duplicate views), pilot selection (`patient_id=71` & SIEMENS), lossless uint16 cache `10`-TTA bit-exact parity, same-size SHA-256 resume corruption rejection, and `--stage build` approval guard.
