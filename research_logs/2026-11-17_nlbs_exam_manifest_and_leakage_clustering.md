# Research Log: Stage A3 NLBS Extraction, FRDR SHA-256 Checksum Verification & Exam-Level Leakage Clustering

**Date**: 2026-11-17  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Dataset Forensic Audit & Leakage Prevention  
**Milestone Scope**: NLBS volume extraction (`/nlbs/extracted/`), FRDR checksum certification, DICOM header harvesting, and graph-connected cluster manifest generation  
**Status**: VERIFIED COMPLETE  

---

## 1. Research Objective & Context
- Execute AnchorShift Stage A3: fully extract all three compressed NLBS `.7z` archives in `nlbs-mammography-vol` to enable high-throughput random I/O during target-domain evaluation, while strictly preserving the original raw archives.
- Resolve the previous Stage A parser defect regarding FRDR official checksum verification.
- Harvest DICOM metadata across all 26,988 images (specifically `StudyInstanceUID`, `SeriesInstanceUID`, `Manufacturer`, `ManufacturerModelName`, `PhotometricInterpretation`, `BitsStored`).
- Resolve the mathematical anomaly between 149 cancer-positive patient folders and 151 cancer-positive examination records.
- Investigate multi-visit patient folders (e.g. `p_3280` containing 180 images) and cross-folder shared `StudyInstanceUID`s to establish a zero-leakage examination grouping methodology.

---

## 2. Datasets, Volumes & Manifest References
- **Target Modal Volume**: `nlbs-mammography-vol`
  - Raw Archives Preserved: `/nlbs/raw/` (`normal.7z`, `abnormal.7z`, `False Positive.7z`, 84.562 GB total)
  - Extracted Directory: `/nlbs/extracted/` (26,988 `.dcm` files)
- **Manifests Generated & Committed**:
  - `datasets/nlbs/manifest.csv`: 26,988 image records enriched with DICOM UIDs, hardware metadata, and cluster identifiers.
  - `datasets/nlbs/exam_manifest.csv`: 6,430 `(case_id, StudyInstanceUID)` examination records.
- **Machine-Readable Audit Output**: `reports/stage_a3_nlbs_results.json`

---

## 3. Forensic Evidence & Diagnostic Findings

### 3.1 Official FRDR SHA-256 Checksum Verification
The Stage A audit script experienced regex mismatch on the Canadian Federated Research Data Repository (`frdr-dfdr-checksums.txt`) format. In Stage A3, a dedicated parser verified all 8 published files against their official SHA-256 digests:
- `normal.7z` (`65,907,091,947` bytes): `227a9eb57e36620f61bd86674d43e3b61fdcf3db87e30a2728cc5e2ea4dfffd7` -> **MATCH**
- `abnormal.7z` (`2,319,999,125` bytes): `e59c4957403631bc8d9b232847bce90b096009c0c3493ef9110989bcc4c5fcf2` -> **MATCH**
- `False Positive.7z` (`22,567,101,176` bytes): `bb50508ad0e2a237a06af2149cfcf1b6630db524456c45d18cdc16db94a337d2` -> **MATCH**
- `NLBSD_Metadata.csv` (`3,933,003` bytes): `4e1a2bca27051a48240330646fba8ba020ab4623ed0f1e27e8c025175fe37af3` -> **MATCH**
- `README.txt`, `README_version_history_2025-11-25.txt`, `CITATION.txt`, `LICENSE.txt` -> **ALL 100% MATCH**
- **Verdict**: Absolute bit-for-bit physical archive integrity confirmed (`all_checksums_match = True`).

### 3.2 Full-Cohort Extraction & Hardware Inventory (All 26,988 Images)
- **Extraction Result**: Exactly `26,988` `.dcm` files extracted into `/nlbs/extracted/` with 0 missing, 0 extra, and 0 DICOM header read errors.
- **Hardware Distribution**:
  - `GE MEDICAL SYSTEMS`: `26,967` images (`99.92%`)
    - `Senographe Essential VERSION ADS_53.10.10`: `26,863`
    - `ADS_53.40`: `96`
    - `ADS_53.30`: `4`
    - Empty model string: `4`
  - `SIEMENS`: `21` images (`0.08%`)
    - `Mammomat Novation DR`: `21`
- **Photometric & Pixel Storage**:
  - `PhotometricInterpretation`: `100% MONOCHROME2` (`26,988 / 26,988`).
  - `BitsStored`: `100% 12-bit` (`26,988 / 26,988`).
  - `TransferSyntaxUID`: `100% Explicit VR Little Endian (1.2.840.10008.1.2.1)` (`26,988 / 26,988`).
  - `PatientID`: Stripped/empty in `26,984 / 26,988` images (`99.99%`).

### 3.3 Resolution of 149 Positive Cases vs 151 Positive Exam Records
Forensic inspection of the two "extra" positive exams revealed that patients **`p_4334`** and **`p_4461`** each contain **two distinct 4-view mammography examinations** acquired under separate `StudyInstanceUID`s:
- `p_4334`:
  - Exam 1 (`StudyInstanceUID: ...7213`): 4 views (R-CC, R-MLO, L-CC, L-MLO), Cancer = 1 on Right.
  - Exam 2 (`StudyInstanceUID: ...7214`): 4 views (R-CC, R-MLO, L-CC, L-MLO), Cancer = 1 on Right.
- `p_4461`:
  - Exam 1 (`StudyInstanceUID: ...7311`): 4 views (R-CC, R-MLO, L-CC, L-MLO), Cancer = 1 on Left.
  - Exam 2 (`StudyInstanceUID: ...7312`): 4 views (R-CC, R-MLO, L-CC, L-MLO), Cancer = 1 on Left.
- Both patients had biopsy-confirmed malignancy present during both acquisitions. Clinically and mathematically, these represent 4 valid screening examinations from 2 patients. Treating them as separate examinations is 100% sound.

### 3.4 Discovery of Cross-Folder Duplicate Studies (`71` Shared UIDs)
1. Across the `6,430` `(case_id, StudyInstanceUID)` tuples, there are only **`6,359` unique `StudyInstanceUID`s**.
2. **`71` `StudyInstanceUID`s appear in multiple `case_id` folders**:
   - `43` shared between folder `p_3280` (which holds 180 images) and 27 other separate `normal` folders.
   - `28` shared across pairs of `False Positive` folders (e.g. identical examination placed in two separate folders).
3. If an experiment splits by `case_id`, an image from `p_3280` could land in an adaptation set while its identical twin lands in an evaluation set, causing fatal data leakage!

---

## 4. Mathematical Formulations & Methodological Decisions

### 4.1 Connected-Component Graph Clustering (`split_cluster_id`)
To prevent cross-folder leakage, we construct an bipartite graph $G = (V_{\text{case}} \cup V_{\text{study}}, E)$ where an edge $(c, s) \in E$ exists if examination study $s$ appears in patient folder $c$.

Using the disjoint-set / Union-Find algorithm with path compression:
$$\text{Find}(u) = \text{Find}(v) \iff u, v \text{ belong to the same connected component}$$
This merges all co-linked `case_id`s and `StudyInstanceUID`s into a unified cluster.

**Empirical Result**:
- Total connected clusters: **`5,942` `split_cluster_id`s**
- `5,913` clusters contain exactly 1 `case_id`.
- `29` clusters link multiple `case_id`s.
- `nlbs_cluster_4240` merges `p_3280` with all 27 associated `normal` folders into a single atomic partition.

### 4.2 Standardized Evaluation Denominators
1. **Case-Level Max-Score Unit**: Unique `StudyInstanceUID` (deduplicated across the 71 folder duplicates, giving exactly $N = 6,359$ true screening examinations: $4,692$ normal, $1,516$ false positive, $151$ abnormal).
2. **Cluster-Bootstrap Confidence Intervals**: Resample at the level of `split_cluster_id` ($N = 5,942$) to correctly model intra-patient and intra-study correlation.

---

## 5. Implementation Details & Exact Changes
- Script: `scripts/stage_a3_nlbs_extract_modal.py`
  - Deployed on CPU-only container (`cpu=1.0, memory=2048, ephemeral_disk=524288`).
  - Implemented streaming 7z archive extraction to `/nlbs_vol/nlbs/extracted/`.
  - Implemented robust DICOM header extraction with fallback for missing tags.
  - Implemented Union-Find graph clustering to assign deterministic `split_cluster_id`.
  - Generated:
    - `datasets/nlbs/exam_manifest.csv` (6,430 rows)
    - `datasets/nlbs/manifest.csv` (26,988 rows)
    - `reports/stage_a3_nlbs_results.json`

---

## 6. Quality Assurance, Unit Tests & Empirical Smoke Tests
- [x] All 8 FRDR SHA-256 checksums verified matching official registry (8 / 8, 100%).
- [x] All 26,988 DICOM files extracted to disk without error (26,988 / 26,988, 100%).
- [x] Zero DICOM header parsing errors across full cohort (0 / 26,988).
- [x] 100% of images verified as `MONOCHROME2` and 12-bit.
- [x] 149 cases vs 151 positive exam records resolved with clinical justification.
- [x] 71 shared `StudyInstanceUID`s isolated and clustered via Union-Find.
- [x] 5,942 `split_cluster_id` components verified disjoint.
- [x] Raw `.7z` archives preserved completely untouched.

---

## 7. Remaining Limitations & Open Issues
1. Unlinkable repeats from the 2025-11-25 directory rename remain unlinked due to stripped `PatientID` and missing `Exam Date`. This is an inherent property of the published dataset and must be stated in the limitations section of any resulting publication.
2. Contralateral images in `abnormal` folders are labeled `Cancer == 0`. Care must be taken during evaluation that ground truth is evaluated at the examination/patient level (Cancer = 1 if any breast is malignant).

---

## 8. Reproducibility & Execution Commands
```bash
# Execute Stage A3 NLBS extraction, checksum verification, and manifest building
modal run scripts/stage_a3_nlbs_extract_modal.py
```
- Runtime: ~22 minutes (full extraction of 84.5 GB compressed archives).
- Compute allocation: 1 vCPU, 2GB RAM, 512GB ephemeral disk.
- Cloud cost: ~$0.08 USD.

---

## 9. Git & Commit Tracking
- **Git Commit SHA**: `b0f4127` (Stage A3 manifests, scripts, and reports)
- **Committed Files**:
  - `scripts/stage_a3_nlbs_extract_modal.py`
  - `datasets/nlbs/exam_manifest.csv`
  - `datasets/nlbs/manifest.csv`
  - `reports/stage_a3_nlbs_results.json`
- **Remote Synchronization**: Pushed to `origin/main`.
