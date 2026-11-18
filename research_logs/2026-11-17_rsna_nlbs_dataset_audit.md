# Research Log: Stage A Complete Dataset Audit — RSNA & NLBS Mammography Volumes

**Date**: 2026-11-17  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Dataset Forensic Audit  
**Milestone Scope**: RSNA Screening Mammography (`rsna-mammography-vol`) and NLBS (`nlbs-mammography-vol`) volumes on Modal  
**Status**: VERIFIED COMPLETE  

---

## 1. Research Objective & Context
- Execute AnchorShift Stage A: an exhaustive, forensic dataset audit across both the source training cohort (RSNA Screening Mammography) and target evaluation cohort (NLBS / NL-Breast-Screening) stored in Modal cloud volumes.
- Verify download completion, physical file integrity, metadata cross-referencing, DICOM pixel decodability, photometric polarities, scanner models, view completeness, and patient/visit grouping before any Stage B model training or checkpoint downloads.
- Strictly enforce research integrity: do not delete, modify, or overwrite raw data, do not spend GPU resources for audits, and strictly quarantine NLBS cancer labels from model training and threshold calibration.

---

## 2. Datasets, Volumes & Manifest References
- **RSNA Volume**: `rsna-mammography-vol:/rsna/` (Total 54,714 files, 293.11 GB)
  - Raw Images: `raw/train_images/<patient_id>/<image_id>.dcm` (54,706 files)
  - Test Stub: `raw/test_images/10008/*.dcm` (4 files)
  - Metadata: `raw/train.csv` (54,706 rows)
  - Manifest Generated: `datasets/rsna/manifest.csv` (54,706 rows)
- **NLBS Volume**: `nlbs-mammography-vol:/nlbs/raw/` (9 files, 84.562 GB)
  - Archives: `normal.7z` (61.381 GB), `abnormal.7z` (2.161 GB), `False Positive.7z` (21.017 GB)
  - Metadata: `NLBSD_Metadata.csv` (26,988 rows)
  - Manifest Generated: `datasets/nlbs/manifest.csv` (26,988 rows)

---

## 3. Forensic Evidence & Diagnostic Findings

### 3.1 RSNA Screening Mammography Audit
1. **File Count & Metadata Parity**:
   - `54,706` DICOM images present on disk vs `54,706` metadata rows in `train.csv`. Exactly 0 missing files, 0 extra files, 0 duplicate `image_id` rows.
2. **Cohort & Prevalence Breakdown**:
   - Unique patients: `11,913`
   - Cancer-positive patients: `486` (`4.08%` prevalence)
   - Cancer-negative patients: `11,427` (`95.92%`)
   - Unique breasts (`patient_id`, `laterality`): `23,826` (exactly 2 breasts per patient)
   - Cancer-positive breasts: `492` (6 patients exhibit bilateral cancer)
   - Cancer-negative breasts: `23,334`
   - Image cancer labels: `1,158` positive, `53,548` negative. `7,705` flagged as `difficult_negative_case == True`.
3. **Sites & Hardware Distribution**:
   - `site_id = 1`: 5,818 patients / 29,519 images
   - `site_id = 2`: 6,095 patients / 25,187 images
   - `machine_id`: 10 distinct scanner units (Machine `49`: 23,529; `48`: 8,699; `29`: 8,267; `21`: 8,221; `93`: 1,915; `216`: 1,908; `210`: 1,070; `170`: 923; `190`: 145; `197`: 29).
4. **View Completeness**:
   - `27,903` MLO, `26,765` CC, plus 38 non-standard views (19 AT, 10 LM, 8 ML, 1 LMO).
   - 100% of breasts (`23,826 / 23,826`) have at least one CC and one MLO view.
5. **DICOM Decoding & Polarity Anomaly (Sample n = 200)**:
   - Decode success: `200 / 200` (100%).
   - Transfer syntax: 140 JPEG Lossless (`1.2.840.10008.1.2.4.70`), 60 JPEG 2000 Lossless (`1.2.840.10008.1.2.4.90`).
   - **CRITICAL ANOMALY DETECTED**: `40 / 200` (`20%`) of sampled DICOMs have `PhotometricInterpretation == "MONOCHROME1"` (where minimum pixel value is intended to be white / radiopaque), while `160 / 200` are `MONOCHROME2` (minimum is black / background).
   - *Action required*: Preprocessing pipeline must dynamically inspect `PhotometricInterpretation` and invert `MONOCHROME1` to ensure dense fibroglandular tissue consistently has high intensity values.

### 3.2 NLBS Screening Mammography Audit
1. **Archive Integrity & Metadata Cross-Check**:
   - `7z t` integrity test passed 100% across `normal.7z` (19,942 DICOMs), `abnormal.7z` (652 DICOMs), and `False Positive.7z` (6,394 DICOMs).
   - Exactly `26,988` total DICOMs across the 3 archives, matching `26,988` rows in `NLBSD_Metadata.csv` (0 missing, 0 extra).
2. **Cohort & Prevalence Breakdown**:
   - Unique folder cases (`case_id`): `5,997` (0 `case_id` overlap across cohort folders)
     - `normal`: 4,332 cases / 19,942 images
     - `False Positive`: 1,516 cases / 6,394 images (`Hard to State Negative == 1` across all images)
     - `abnormal`: 149 cases / 652 images
   - Cancer-positive cases: `149` (`2.48%` prevalence)
   - Cancer-negative cases: `5,848` (`97.52%`)
   - Cancer images: `337` positive images, `26,651` negative images.
   - Breast-level label structure: all 337 positive images reside in `-c/` folders (e.g. `abnormal/p_4333/left-c/CC/...`), while the 315 contralateral images in `abnormal` have `Cancer == 0`. Every abnormal case has at least 2 cancer-positive images.
3. **Hardware & Metadata Discrepancies vs Published Literature**:
   - **Discrepancy 1 (Scanner Hardware)**: Literature and `AnchorShift_Research_Context.md` report NLBS as purely *"GE Senographe Essential"*. DICOM header inspection revealed `56/60` GE Senographe Essential, but **`4/60` SIEMENS `Mammomat Novation DR`** (shape `[4084, 3328]`). NLBS is predominantly GE, but multi-vendor.
   - **Discrepancy 2 (Acquisition Dates)**: `NLBS README.txt` records dates as `2010-2011 approximately`, whereas prior context stated `2008-2010`.
   - **Discrepancy 3 (Repeat Exams & Multi-Visit Grouping)**:
     - `README_version_history_2025-11-25.txt` states: *"Update made to rename directories to prevent link between similarly named exams (repeats) as we cannot reliably establish that link at this stage."*
     - DICOM `PatientID` is stripped (`""` in sampled headers), and `Exam Date` in metadata is 100% null (`0 / 26,988`). Renamed cross-folder repeats cannot be recovered from metadata.
     - However, within-folder multi-visit cases still exist: `1,208` `case_id`s have `> 4` images, and `320` `normal` `case_id`s have multiple `Age` values (e.g., `p_0004` has 8 images across Ages 53 and 54; **`p_3280` contains 180 images across Ages 62–73**).

---

## 4. Mathematical Formulations & Methodological Decisions
1. **MONOCHROME1 Polarity Inversion**:
   $$\tilde{x} = \begin{cases} \max(x) - x & \text{if } \text{PhotometricInterpretation} = \text{"MONOCHROME1"} \\ x & \text{if } \text{PhotometricInterpretation} = \text{"MONOCHROME2"} \end{cases}$$
2. **Intensity Rescaling**:
   $$\hat{x} = \frac{\tilde{x} - \min(\tilde{x})}{\max(\tilde{x}) - \min(\tilde{x}) + \epsilon}$$
3. **Partition & Evaluation Unit Separation**:
   - Model training and threshold calibration must be performed exclusively on RSNA screening cases.
   - Patient grouping must be strictly preserved to prevent multi-view / bilateral leakage.
   - For NLBS, multi-visit cases (e.g. `p_3280`) must not be pooled into a single screening case, but split by examination UID during evaluation.

---

## 5. Implementation Details & Exact Changes
- Script: `scripts/audit_datasets_modal.py`
  - Deployed on CPU-only container (`cpu=1.0, memory=2048`, zero GPUs).
  - Inspects both `rsna-mammography-vol` and `nlbs-mammography-vol` concurrently.
  - Implemented 7z archive test harness via `py7zr` and system subprocess.
  - Generates JSON audit dump `reports/audit_results.json` and markdown report `reports/dataset_audit.md`.
  - Emits `datasets/rsna/manifest.csv` and `datasets/nlbs/manifest.csv`.

---

## 6. Quality Assurance, Unit Tests & Empirical Smoke Tests
- [x] RSNA DICOM count matches metadata (54,706 / 54,706, 100%).
- [x] RSNA pixel decoding passed (200 / 200, 100%).
- [x] `MONOCHROME1` polarity anomaly identified (40 / 200).
- [x] NLBS 7z archive integrity verified (3 / 3 archives, 100%).
- [x] NLBS archived DICOM count matches metadata (26,988 / 26,988, 100%).
- [x] NLBS pixel decoding passed (60 / 60, 100%).
- [x] Zero GPU resources consumed during audit.
- [x] Zero raw data files modified or overwritten.

---

## 7. Remaining Limitations & Open Issues
1. NLBS DICOMs remain compressed inside `.7z` archives; must be extracted to a separate volume directory (`/nlbs/extracted/`) to enable fast random I/O during evaluation.
2. DICOM `StudyInstanceUID`s must be extracted for all 26,988 NLBS images to resolve within-folder multi-visit cases and establish a precise examination-level manifest.
3. Unlinkable repeats from the 2025-11-25 directory rename cannot be recovered without original hospital master keys.

---

## 8. Reproducibility & Execution Commands
```bash
# Execute Stage A dataset audit on Modal (CPU only)
modal run scripts/audit_datasets_modal.py
```
- Runtime: ~4.5 minutes.
- Compute allocation: 1 vCPU, 2GB RAM.
- Cloud cost: ~$0.015 USD.

---

## 9. Git & Commit Tracking
- **Git Commit SHA**: `b45d290` (Initial audit scripts & reports)
- **Committed Files**:
  - `scripts/audit_datasets_modal.py`
  - `reports/dataset_audit.md`
  - `reports/audit_results.json`
  - `datasets/rsna/manifest.csv`
  - `datasets/nlbs/manifest.csv`
- **Remote Synchronization**: Pushed to `origin/main`.
