# Workspace Index

## Directory Structure & Detailed File Context

### Root (`d:\Breast Cancer Research`)
- `.gitignore`: Ignores Python cache, `.env`/secret keys, raw medical image/archive files (`*.dcm`, `*.zip`, `*.png`, etc.), and model checkpoints while keeping `datasets/` manifest CSVs tracked.
- `AnchorShift_Research_Context.md`: 531-line master research brief for the project and AI sessions. Covers the clinical task (case-level screening mammography selective dismissal), the 3 target-label protocols (strict source-only frozen threshold, unsupervised target adaptation, target-label calibration), TRUST (2026, arXiv:2609.00300) model/loss/dataset/external-evaluation details, prior work (UPA 2023, Garrucho 2022, MC-SDS 2025, MammoDG 2026, TENT, SAR, Conformal Risk Control), dataset audit/caveats (RSNA, NLBS, IMDB, CMMD, VinDr-Mammo, EMBED, CBIS-DDSM), Stage A–D experimental protocol, go/no-go decision table, and strict project guardrails.

### `datasets/`
- `rsna/manifest.csv`: Deterministic 54,706-row image manifest for RSNA Screening Mammography (`patient_id`, `image_id`, `laterality`, `view`, `cancer`, `biopsy`, `invasive`, `BIRADS`, `implant`, `density`, `machine_id`, `site_id`, `age`, `difficult_negative_case`, `rel_path`).
- `rsna/splits_seed42.csv`: Deterministic patient-level RSNA split manifest (38,251 train, 5,473 train_calib, 10,982 val_calib) with 0 patient leakage across 11,913 unique patients.
- `nlbs/manifest.csv`: Deterministic 26,988-row image manifest for NLBS enriched with `exam_id`, `visit_idx`, `split_cluster_id`, `StudyInstanceUID`, `SeriesInstanceUID`, `Manufacturer`, `ManufacturerModelName`, `PhotometricInterpretation`, `BitsStored`, `Rows`, `Columns`, `TransferSyntaxUID`, and `rel_path`.
- `nlbs/exam_manifest.csv`: Deterministic 6,430-row examination-level manifest for NLBS grouped by `(case_id, StudyInstanceUID)` with `split_cluster_id` (`5,942` connected clusters) to prevent cross-split leakage across the 71 shared `StudyInstanceUID`s.

### `src/`
- `models/efficientnet_custom.py` & `models/efficient_net_custom_utils.py`: Canonical Mammo-CLIP EfficientNet-B5 implementation (`batmanlab/Mammo-CLIP` / `lukemelas`) matching all 852 checkpoint tensors of `b5-model-best-epoch-7.tar`.
- `models/trust_model.py`: Modular PyTorch implementations of `MammoCLIPEncoder` (frozen Mammo-CLIP `efficientnet-b5` with `strict=True` weight loading), `TRUSTClassifier` (LayerNorm + Dropout 0.3 + Linear(2048, 1) + Sigmoid), and `aggregate_case_scores` (max pooling per exam).
- `models/losses.py`: `FocalLoss`, `DynamicDismissalLoss` (penalizing positive scores near/below dynamic tau_t), and unified `TRUSTLoss` supporting both CE-only baseline and TRUST modes.
- `models/mammo_clip_loader.py`: Utilities to extract `image_encoder.` state dict from `b5-model-best-epoch-7.tar` and verify `strict=True` compatibility (852/852 keys) with `efficientnet_custom.EfficientNet`.
- `data/preprocessing.py`: `preprocess_dicom` pipeline handling DICOM decoding, MONOCHROME1 polarity inversion, min-max normalization, Mammo-CLIP mean/std standardization (`0.3089279 / 0.25053555`), 1520x912 bilinear interpolation, and 3-channel expansion.
- `data/splits.py`: Deterministic patient-level stratified split generator for RSNA.
- `data/rsna_dataset.py`: `RSNADataset` and `BalancedBatchSampler` (20 positive, 60 negative per batch of 80).
- `training/calibration.py`: `compute_target_recall_threshold` (quantile-based tau_t), `clopper_pearson_upper_bound`, and `evaluate_dismissal_at_threshold`.

### `reports/`
- `trust_reproduction_plan.md`: Master Stage B reproduction report detailing TRUST Methods 3.1–3.11, Mammo-CLIP licensing & architecture, loss formulations, RSNA split design, NLBS 151 vs 149 exam analysis, GPU recommendations, and copy-pasteable execution commands.
- `dataset_audit.md`: Stage A & A3 complete dataset audit report covering RSNA & NLBS counts, FRDR SHA-256 verification, DICOM header/pixel validation, scanner/repeat-exam findings, and TRUST readiness.
- `audit_results.json`: Machine-readable Stage A audit output from `scripts/audit_datasets_modal.py --mode audit`.
- `stage_a3_nlbs_results.json`: Machine-readable Stage A3 NLBS extraction, checksum verification (`8/8` matched), and exam-grouping summary from `scripts/stage_a3_nlbs_extract_modal.py`.
- `discovery_summary.json`: Initial Stage A volume tree and CSV schema preview from `scripts/audit_datasets_modal.py --mode discover`.

### `scripts/`
- `download_mammoclip_modal.py`: CPU-only (`cpu=1.0, memory=2048`, `ephemeral_disk=32768`) Modal script to download `b5-model-best-epoch-7.tar` (1.65 GB) from Hugging Face into Modal Volume `rsna-mammography-vol` at `/models/mammo_clip/`.
- `smoke_test_stage_b_modal.py`: Cost-optimized GPU smoke test on T4 (`gpu="T4"`, ~$0.59/hr, ~1 min runtime) using 10 real RSNA DICOMs to test preprocessing, forward/backward passes, loss calculation, and VRAM consumption.
- `train_trust_modal.py`: Full Modal training pipeline on L4 (`gpu="L4"`, 20 epochs) supporting `--baseline trust` and `--baseline ce` with gradient accumulation, dynamic `tau_t` recalculation, and operating-threshold evaluation.
- `stage_a3_nlbs_extract_modal.py`: CPU-only Modal script to verify FRDR SHA-256 checksums, extract NLBS `.7z` archives, and build `exam_manifest.csv`.
- `audit_datasets_modal.py`: CPU-only Modal script for Stage A discovery and full dataset audit.
- `download_nlbs_modal.py`: Modal downloader for NLBS from FRDR.
- `download_rsna_modal.py`: Modal downloader for RSNA from Kaggle.

### `research_logs/`
- `README.md`: Master directory overview, role and relationship to `memory/`, mandatory research logging rules, and chronological index table of all research logs.
- `TEMPLATE.md`: Standardized 9-section template for AnchorShift research logs (Objective, Datasets/Volumes, Forensic Evidence, Mathematical Formulations, Implementation Details, QA/Tests, Limitations, Reproducibility Commands, Git Tracking).
- `2026-11-18_stage_b_gpu_smoke_test_and_mammoclip_verification.md`: Deep-dive research log documenting the live Tesla T4 GPU smoke test (Run 1 `weights_only` unpickler diagnosis and Run 2 certified pass: `checkpoint_loaded: True`, 0.78 GB / 14.58 GB peak VRAM, 8.75s DICOM ingestion, frozen encoder verified).
- `2026-11-17_trust_reproduction_and_rsna_splits.md`: Deep-dive research log documenting Stage B TRUST reproduction (arXiv:2609.00300), Mammo-CLIP B5 loader, zero-leakage patient-level RSNA split (8,339 train, 1,192 train-calib, 2,382 val-calib), BalancedBatchSampler, and 5/5 unit test verification.
- `2026-11-17_nlbs_exam_manifest_and_leakage_clustering.md`: Deep-dive research log documenting Stage A3 NLBS extraction (26,988 DICOMs), 8/8 FRDR SHA-256 verification, clinical resolution of 149 positive patients vs 151 positive exams (`p_4334` & `p_4461`), discovery of 71 shared `StudyInstanceUID`s, and Union-Find graph clustering (`5,942` `split_cluster_id`s).
- `2026-11-17_rsna_nlbs_dataset_audit.md`: Deep-dive research log documenting Stage A forensic audit of RSNA (54,706 DICOMs, 4.08% patient positive) and NLBS (26,988 DICOMs across 3 archives), discovery of 40/200 `MONOCHROME1` inverted polarities in RSNA, and discovery of Siemens scanner in NLBS.

### `tests/`
- `test_stage_b_smoke.py`: Standalone local unit test suite (no cloud or GPU needed) verifying model forward pass, frozen encoder isolation, BCE/Focal/Dismissal losses, dynamic threshold calculation, Clopper-Pearson bound, case max aggregation, and patient split integrity (5/5 PASS).

### `memory/`
- `personality.md`: Assistant persona and communication rules.
- `personal_info.md`: User profile (Hasin Ishrak), skills, projects, and background.
- `purpose.md`: AnchorShift mission, protocol discipline, and guardrails.
- `state.md`: Active Stage A–D checklist, blockers, and session progress.
- `history.md`: Chronological conversation log (most recent first).
- `index.md`: Detailed map of all files and directories in the workspace.
