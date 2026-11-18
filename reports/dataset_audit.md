# Stage A — Complete Dataset Audit Report (RSNA & NLBS)

**Date:** 2026-11-17  
**Modal Workspace:** `hasinishrak2015`  
**Audit Script:** `scripts/audit_datasets_modal.py` (CPU-only: `cpu=1.0, memory=2048`, zero GPUs)  
**Machine-Readable Output:** `reports/audit_results.json`  
**Deterministic Manifests:**
- `datasets/rsna/manifest.csv` (54,706 rows)
- `datasets/nlbs/manifest.csv` (26,988 rows)

---

## 1. Executive Summary & Readiness Verdict

| Dataset | Modal Volume | Storage State | Integrity & Cross-Check | TRUST Reproduction Readiness |
|---|---|---|---|---|
| **RSNA Screening Mammography** | `rsna-mammography-vol` | Extracted (`raw/train_images`, `raw/test_images`, CSVs), 293.11 GB | **100% match:** 54,706/54,706 train DICOMs present, 0 missing, 0 extra, 200/200 stratified pixel decodes OK | **READY** (must invert `MONOCHROME1` -> `MONOCHROME2` during preprocessing) |
| **NLBS (NL-Breast-Screen)** | `nlbs-mammography-vol` | **Compressed** in 3 `.7z` archives (`normal.7z`, `abnormal.7z`, `False Positive.7z`), 84.562 GB | **100% match:** `7z t` passed all 3 archives; 26,988/26,988 DICOMs match `NLBSD_Metadata.csv`, 60/60 sample pixel decodes OK | **READY AFTER EXTRACTION** (must extract `.7z` archives to a separate directory and handle `case_id` vs `StudyInstanceUID` exam grouping) |

**Safety & Protocol Compliance:** Raw datasets were never modified, overwritten, or deleted. No models were trained. NLBS cancer labels were inspected strictly for data auditing and remain isolated from training, adaptation, or threshold selection.

---

## 2. RSNA Screening Mammography (`rsna-mammography-vol`)

### 2.1 Folder Structure & Counts
- **Root:** `/rsna_vol/rsna/` (total 54,714 files, 293.11 GB; includes `train.csv.zip` + `raw/`)
- **Metadata files:** `raw/train.csv` (54,706 rows), `raw/test.csv` (4 stub rows), `raw/sample_submission.csv`
- **Image files on disk:**
  - `raw/train_images/<patient_id>/<image_id>.dcm`: **54,706** `.dcm` files
  - `raw/test_images/10008/*.dcm`: **4** `.dcm` files
- **Path cross-check:** `0` missing on disk, `0` extra on disk.
- **Duplicates:** `0` duplicate `image_id` rows, `0` duplicate `(patient_id, image_id)` rows. (`7,018` rows share `(patient_id, laterality, view)` due to repeat/extra views within an exam.)

### 2.2 Cohort & Label Breakdown
- **Unique patients (exams):** `11,913`
  - Cancer-positive patients: **`486`** (`4.08%`)
  - Cancer-negative patients: **`11,427`**
- **Unique breasts `(patient_id, laterality)`:** `23,826` (exact 2 breasts per patient)
  - Cancer-positive breasts: **`492`** (6 bilateral cancer patients)
  - Cancer-negative breasts: **`23,334`**
  - Inconsistent labels within `(patient_id, laterality)`: **`0`**
- **Images:**
  - Cancer-positive images: **`1,158`**
  - Cancer-negative images: **`53,548`**
  - `difficult_negative_case == True`: **`7,705`** images
- **Sites & Machines:**
  - `site_id = 1`: `5,818` patients / `29,519` images
  - `site_id = 2`: `6,095` patients / `25,187` images
  - `machine_id`: 10 distinct IDs (`49`: 23,529; `48`: 8,699; `29`: 8,267; `21`: 8,221; `93`: 1,915; `216`: 1,908; `210`: 1,070; `170`: 923; `190`: 145; `197`: 29)
- **Missingness in optional clinical columns:** `BIRADS` is null in `28,420` rows; `density` is null in `25,236` rows; `age` is null in `37` rows.

### 2.3 View Completeness
- **View counts:** `MLO` (`27,903`), `CC` (`26,765`), plus 38 non-standard views (`AT`: 19, `LM`: 10, `ML`: 8, `LMO`: 1).
- **Laterality counts:** `R` (`27,439`), `L` (`27,267`).
- **Per-breast completeness:** All **`23,826` / `23,826` (`100%`)** breasts have at least one `CC` AND one `MLO` view (`0` missing CC, `0` missing MLO).

### 2.4 DICOM Decoding & Metadata Validation (Stratified Sample: `n = 200`)
- **Pixel decode success:** **`200 / 200` (`100%`)**, `0` failures.
- **Transfer Syntaxes:**
  - `1.2.840.10008.1.2.4.70` (JPEG Lossless): `140`
  - `1.2.840.10008.1.2.4.90` (JPEG 2000 Lossless): `60`
- **Photometric Interpretation:**
  - `MONOCHROME2`: `160`
  - **`MONOCHROME1`: `40`** (inverted polarity — preprocessing MUST check `PhotometricInterpretation` and invert `MONOCHROME1`!)
- **Bits Stored:** `12` (`160`), `10` (`20`), `16` (`20`).

---

## 3. NLBS / NL-Breast-Screening (`nlbs-mammography-vol`)

### 3.1 Archive Integrity & File Cross-Check
- **Root:** `/nlbs_vol/nlbs/raw/` (9 files, 84.562 GB)
- **Archives (`7z t` integrity test):**
  - `normal.7z` (`61.381 GB`): **`19,942` `.dcm` files** — `integrity_ok = True`
  - `abnormal.7z` (`2.161 GB`): **`652` `.dcm` files** — `integrity_ok = True`
  - `False Positive.7z` (`21.017 GB`): **`6,394` `.dcm` files** — `integrity_ok = True`
- **Metadata cross-check (`NLBSD_Metadata.csv`):**
  - Total rows: **`26,988`** vs. total archived `.dcm` files: **`26,988`**
  - Missing in archives: **`0`**, Extra in archives: **`0`**, Duplicate paths: **`0`**
  - *(Note: `NLBSD_Metadata.csv` stores Windows backslashes `\` in `File Path`; normalized to `/` in `datasets/nlbs/manifest.csv`.)*

### 3.2 Cohort & Label Breakdown
- **Total cases (`case_id`):** **`5,997`** (zero `case_id` overlap across the 3 cohort folders)
  - `normal`: `4,332` cases / `19,942` images
  - `False Positive`: `1,516` cases / `6,394` images (`Hard to State Negative == 1` on all `6,394` images)
  - `abnormal`: `149` cases / `652` images
- **Cancer labels:**
  - Cancer-positive cases: **`149`** (`2.48%`)
  - Cancer-negative cases: **`5,848`**
  - Cancer-positive images: **`337`**; Cancer-negative images: **`26,651`**
  - **Breast-level label structure:** All `337` `Cancer == 1` images reside in `-c/` folders (e.g., `abnormal/p_4333/left-c/CC/...`), while the `315` contralateral images in `abnormal` have `Cancer == 0`. Every abnormal case has `>= 2` `Cancer == 1` images (`2`: 120, `3`: 21, `4`: 7, `6`: 1).

### 3.3 View Completeness
- **View counts:** `CC` (`13,800`), `MLO` (`13,188`).
- **Laterality counts:** `R` (`13,549`), `L` (`13,439`).
- **Per-breast completeness (grouped by `cohort_group, case_id, Image Laterality`):**
  - Both CC & MLO: `11,825`
  - Missing CC: `7`
  - Missing MLO: `159`

### 3.4 DICOM Decoding & Key Discrepancies vs. Published Descriptions (Sample: `n = 60`, 20/archive)
- **Pixel decode success:** **`60 / 60` (`100%`)**, `0` failures.
- **Transfer Syntax:** `1.2.840.10008.1.2.1` (Explicit VR Little Endian): `60/60`.
- **Photometric / Bits:** `MONOCHROME2`: `60/60`, `BitsStored = 12`: `60/60`.
- **DISCREPANCY 1 — Scanner Hardware:** `AnchorShift_Research_Context.md` (and TRUST summary) describes NLBS as *"GE Senographe Essential systems"*. Our DICOM header sample found:
  - `GE MEDICAL SYSTEMS` (`Senographe Essential VERSION ADS_53.10.10` / `ADS_53.40`): `56/60`
  - **`SIEMENS` (`Mammomat Novation DR`, shape `[4084, 3328]`): `4/60`**
  - NLBS is **predominantly GE**, not 100% GE.
- **DISCREPANCY 2 — Collection Dates:** `NLBS README.txt` states collection was `2010-2011 approximately`, whereas `AnchorShift_Research_Context.md` says `2008-2010`.
- **DISCREPANCY 3 — Repeat Exams & Patient Grouping (CRITICAL):**
  - `README_version_history_2025-11-25.txt` states: *"Update made to rename directories to prevent link between similarly named exams (repeats) as we cannot reliably establish that link at this stage."*
  - In the DICOM headers sampled, `PatientID` is stripped (`""`), and in `NLBSD_Metadata.csv`, `Exam Date` is `100%` null (`0 / 26,988`). Thus, any repeats split into separate `p_XXXX` folders in the 2025-11-25 rebuild cannot be re-linked from metadata.
  - **HOWEVER, within-folder multi-exam cases still exist:** `1,208` `case_id`s have `> 4` images, and **`320` `normal` `case_id`s span multiple `Age` values** (e.g., `p_0004` has 8 images at Ages 53 & 54; **`p_3280` has `180` images spanning Ages 62–73!**).
  - **Protocol implication:**
    1. **Splitting/clustering (adapt vs. eval):** Group by `case_id` so multi-visit folders like `p_3280` never leak across partitions.
    2. **Case-level max-score inference:** Do NOT blindly pool all `180` images of `p_3280` into a single screening "case" — split exams within a `case_id` by DICOM `StudyInstanceUID` (once extracted) or `(case_id, Age)` when computing per-exam dismissal rates, and use patient/cluster bootstrap by `case_id` for confidence intervals.

---

## 4. Next Steps (Before Stage B Training)
1. Extract NLBS `.7z` archives into `/nlbs_vol/nlbs/extracted/` (leaving `/nlbs_vol/nlbs/raw/*.7z` untouched) and extract DICOM `StudyInstanceUID` + `ManufacturerModelName` into an enriched NLBS exam-level manifest.
2. Begin **Stage B**: check TRUST code/checkpoint availability, obtain Mammo-CLIP pretrained EfficientNet-B5 weights, and reproduce CE-only + TRUST on RSNA with patient-level splits.

---

## 5. Stage A3 — NLBS Extraction, Full Checksum Verification & Exam-Level Grouping

**Script:** `scripts/stage_a3_nlbs_extract_modal.py` (CPU-only: `cpu=1.0, memory=2048`, `ephemeral_disk=524288`, zero GPUs)  
**Machine-Readable Output:** `reports/stage_a3_nlbs_results.json`  
**Updated Manifests:**
- `datasets/nlbs/manifest.csv` (26,988 images, enriched with `exam_id`, `visit_idx`, `split_cluster_id`, `StudyInstanceUID`, `SeriesInstanceUID`, `Manufacturer`, `ManufacturerModelName`, `PhotometricInterpretation`, `BitsStored`, `Rows`, `Columns`, `TransferSyntaxUID`)
- `datasets/nlbs/exam_manifest.csv` (6,430 `(case_id, StudyInstanceUID)` exam records with `split_cluster_id`)

### 5.1 Official FRDR Checksum & Extraction Verification
- **FRDR SHA-256 Checksums:** **`8 / 8` matched (`all_checksums_match = True`)** across `normal.7z`, `abnormal.7z`, `False Positive.7z`, `NLBSD_Metadata.csv`, `README.txt`, `README_version_history_2025-11-25.txt`, `CITATION.txt`, and `LICENSE.txt`.
- **Extraction:** All 3 `.7z` archives extracted to `/nlbs_vol/nlbs/extracted/` while preserving `/nlbs_vol/nlbs/raw/*.7z` untouched.
- **File & Header Verification:** **`26,988 / 26,988`** `.dcm` files verified (`0` missing, `0` extra, `0` header read errors).

### 5.2 Full-Cohort DICOM Hardware & Pixel Format (All 26,988 Images)
- **Manufacturers:** `GE MEDICAL SYSTEMS` (`26,967`, `99.92%`), `SIEMENS` (`21`, `0.08%`).
- **Models:** `Senographe Essential VERSION ADS_53.10.10` (`26,863`), `ADS_53.40` (`96`), `ADS_53.30` (`4`), `Mammomat Novation DR` (`21`), empty (`4`).
- **Photometric / Bits / Transfer Syntax:** `100%` `MONOCHROME2` (`26,988`), `100%` `BitsStored = 12` (`26,988`), `100%` Explicit VR Little Endian `1.2.840.10008.1.2.1` (`26,988`).
- **DICOM `PatientID`:** Empty in `26,984 / 26,988` images (non-empty in only `4` images).

### 5.3 Examination Grouping & Leakage Findings (CRITICAL)
1. **Within-folder multi-visit cases separated by `StudyInstanceUID`:**
   - `5,997` `case_id`s expand into **`6,430`** `(case_id, StudyInstanceUID)` examination rows (`4,763` normal, `1,516` False Positive, `151` abnormal; `0` exams have multiple `Age` values).
   - `385` `case_id`s have `> 1` `StudyInstanceUID` (`378` have 2 visits, `6` have 3 visits, and **`p_3280` has `44` distinct `StudyInstanceUID`s**).
2. **Cross-folder duplicate `StudyInstanceUID`s discovered (`71` UIDs):**
   - `6,359` unique `StudyInstanceUID`s vs. `6,430` `(case_id, StudyInstanceUID)` rows -> **`71` `StudyInstanceUID`s appear in MORE THAN ONE `case_id` folder** (`43` shared between `p_3280` and other `normal` folders; `28` shared across `False Positive` folder pairs).
   - **Connected-component splitting (`split_cluster_id`):** Grouping by connected components over `(case_id, StudyInstanceUID)` yields **`5,942` independent `split_cluster_id`s** (`29` clusters span `> 1` `case_id`, including `nlbs_cluster_4240` linking `p_3280` and 27 other `normal` `case_id`s).
3. **Unlinkable renamed repeats (`2025-11-25` rebuild):**
   - Per `README_version_history_2025-11-25.txt`, directories for some repeat exams were renamed to prevent linking because that link could not be reliably established; with `PatientID` stripped (`99.99%` empty) and `Exam Date` 100% null, those unlinked repeats cannot be re-linked from metadata.
4. **Evaluation Readiness Verdict:**
   - **NLBS examination grouping IS reliable enough for evaluation** when using:
     1. **`StudyInstanceUID`** (deduplicated across the 71 cross-folder duplicates, `6,359` unique exams) as the unit of case-level max-score inference (never pooling all 180 images of `p_3280` as one exam).
     2. **`split_cluster_id`** as the atomic unit for adaptation/evaluation splitting and cluster-bootstrap CIs (guarantees zero `case_id` or `StudyInstanceUID` leakage across partitions).
     3. Explicit acknowledgment in limitations that any separate-folder repeats unlinked by the 2025-11-25 rebuild remain unlinked.

