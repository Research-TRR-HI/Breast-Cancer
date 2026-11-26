# IMDB (India) Label, Identity & Outcome Reliability Audit

**Dataset:** Indian Mammography DataBase (IMDB) — Release 1.0 (`MAMOS_1000000004`) & Release 2.0 (`MAMOS_1000000050`)  
**Source Portal:** Indian Biological Images Archive (IBIA), DBT India (`https://ibdc.dbt.gov.in/ibia/studybrowse/`)  
**Official Code Repo:** `https://github.com/krithikarangarajan-coe/imdb_ibia`  
**Audit Date:** `2026-11-26`  
**Cohort Role:** `mixed_diagnostic_opportunistic` (`population_screening_dismissal_allowed = False`)  

---

## 1. Cross-Release Patient Identity Audit (`imdb_r1` vs `imdb_r2`)

- **Release 1.0 (`MAMOS_1000000004`):** `583` unique `De-identified Patient ID` values formatted as zero-padded 4-digit strings (`0000`–`0582`), `3,577` images.
- **Release 2.0 (`MAMOS_1000000050`):** `2,636` unique `De-identified Patient ID` values formatted as `pt_001`–`pt_2636` (`9,097` images).
- **Collision Finding (`CRITICAL`):** Direct string overlap is `0`, **but numeric-suffix overlap is `582`** (`0001`–`0582` vs `pt_001`–`pt_582`) because `deidentification/deidentify.py` assigns sequential `pt_XXX` IDs starting from `1` independently per release batch.
- **Resolution:** All manifests namespace `cluster_id` and `exam_id` by release (`imdb_r1_0000` vs `imdb_r2_pt_001`), preventing false cross-release patient merges. Total unique release-namespaced exams/clusters = **`3,219`** (`583 + 2,636`), **`12,674`** images.

---

## 2. Modality Separation: FFDM (`2D-Mammo`) vs Synthetic Mammography (`C-View`)

Both releases mix **Full-Field Digital Mammography (`2D-Mammo` -> `FFDM`)** and **Synthetic 2D Mammography (`C-View` / `C-view` -> `SM`)** reconstructed from Digital Breast Tomosynthesis (DBT) on Hologic Selenia Dimensions systems:

| Cohort | Modality Subtype | Images | Exams / Patients | Positives (`Cancer`) | Complete 4-View Exams | 4-View Positives |
|---|---|---:|---:|---:|---:|---:|
| `imdb_r1` | `FFDM` (`2D-Mammo`) | `1,869` | `583` | `73` | `304` | `60` |
| `imdb_r1` | `SM` (`C-View`) | `1,708` | `549` | `71` | `285` | `56` |
| `imdb_r1` | **Combined** | **`3,577`** | **`583`** | **`73`** | **`304`** | **`60`** |
| `imdb_r2` | `FFDM` (`2D-Mammo`) | `5,996` | `1,645` | `233` | `1,361` | `193` |
| `imdb_r2` | `SM` (`C-view`) | `3,101` | `1,051` | `91` | `493` | `85` |
| `imdb_r2` | **Combined** | **`9,097`** | **`2,636`** | **`316`** | **`1,849`** | **`278`** |
| **Total (`imdb`)** | **FFDM + SM** | **`12,674`** | **`3,219`** | **`389`** | **`2,153`** | **`338`** |

- **Key Observation:** In `imdb_r1`, almost every patient (`549/583`) has *both* FFDM and SM views (averaging `~6.1` images/patient). In `imdb_r2`, `FFDM` and `SM` are largely disjoint (`1,645 + 1,051 = 2,696` vs `2,636` total patients, meaning only `60` patients have both).
- **Quarantine & Cohort Rule:** `select_common_4view_cohort` strictly prefers `FFDM` over `SM` when selecting one image per `{L-CC, L-MLO, R-CC, R-MLO}` slot, and sensitivity analyses MUST report `FFDM`-only and `SM`-only separately to avoid confounding synthetic C-View domain shift with FFDM generalization.

---

## 3. Clinical Setting & Ground Truth Reliability Audit

### 3.1 Clinical Spectrum (`r1` Clinical Indications)
IMDB is a **tertiary oncology hospital (AIIMS New Delhi)** diagnostic and cancer-survivor follow-up cohort, NOT a population screening cohort:
- `Prior history of breast cancer on FU`: `2,213 / 3,577` (`61.9%`)
- `Diagnostic mammogram for breast symptoms (lump/pain/discharge)`: `607 / 3,577` (`17.0%`)
- `Women with other primary cancer`: `539 / 3,577` (`15.1%`)
- `Screening mammogram`: `114 / 3,577` (`3.2%`)
- `Breast cancer interim assessment on chemotherapy`: `104 / 3,577` (`2.9%`)

### 3.2 Five-Category Outcome Reliability Breakdown

Because `deidentify.py` neutralizes all DICOM dates to `20260101` and IBIA does not provide longitudinal 12–24 month post-exam follow-up timestamps for `Not cancer` cases, `Not cancer` BI-RADS 1/2/3 exams cannot be classified as `followup_confirmed_nonmalignant`:

| Outcome Reliability Category | `imdb_r1` Images | `imdb_r1` Exams | `imdb_r2` Images | `imdb_r2` Exams | Combined Images | Combined Exams |
|---|---:|---:|---:|---:|---:|---:|
| `pathology_confirmed_malignant` | `296` | `72` | `866` | `315` | `1,162` | `387` |
| `pathology_confirmed_benign` | `4` | `1` | `374` | `49` | `378` | `50` |
| `followup_confirmed_nonmalignant` | `0` | `0` | `0` | `0` | `0` | `0` |
| `noncancer_without_adequate_followup` | `3,273` | `509` | `7,825` | `2,267` | `11,098` | `2,776` |
| `unknown_or_ambiguous` | `4` | `1` | `32` | `5` | `36` | `6` |
| **Total** | **`3,577`** | **`583`** | **`9,097`** | **`2,636`** | **`12,674`** | **`3,219`** |

### 3.3 Data Hygiene & Discordant Label Findings
1. **Case inconsistency in `r2`:** `Ground truth label` uses both `'cancer'` (`868`) and `'Cancer'` (`8`); `Mammography type` uses `'C-View'` in `r1` vs `'C-view'` in `r2`; `Breast density` in `r2` has `'non_dense'` (`1`) alongside `'not_dense'` (`5,600`). All manifest builders normalize case.
2. **Unilateral / Mixed-Label Patients:** `61` patients in `r1` and `168` patients in `r2` have mixed image-level labels (`Cancer` on affected breast, `Not cancer` on contralateral breast). Exam-level `outcome_label` is `max` across images (`1` if any image is `Cancer`), matching Paper 1 exam-level evaluation.
3. **Discordant `BI-RADS` vs `Ground truth label` (`unknown_or_ambiguous` = `6` exams):**
   - `r1` patient `0406` (`4` images): `BI-RADS 1`, `Clinical Indication = Prior history of breast cancer on FU`, `ICD-11 = 2C6Z`, labeled `Cancer`.
   - `r2` `pt_2205` (`2` images): `BI-RADS 2 (Post op changes)`, labeled `cancer` without histopathology text.
   - `r2` `BI-RADS 0` (`30` images total: `22` `Not cancer`, `8` `cancer` where `pt_297` and `pt_1352` have `Histopath: Invasive/Metastatic carcinoma` in report text).

---

## 4. Label-Blind Inference & Protected Outcome Manifests

All manifests were generated deterministically via `scripts/build_imdb_manifests.py` and validated with `src.paper1.external_datasets.write_manifest_pair` (zero forbidden label columns, clean `acquisition_meta` JSON):

| Dataset ID | Inference Manifest (Label-Blind) | Outcome Manifest (Protected) | Exams | Positives |
|---|---|---|---:|---:|
| `imdb_r1` | `datasets/exp3_meta/imdb_r1_inference_manifest.csv` | `datasets/exp3_meta/imdb_r1_outcome_manifest.csv` | `583` | `73` |
| `imdb_r2` | `datasets/exp3_meta/imdb_r2_inference_manifest.csv` | `datasets/exp3_meta/imdb_r2_outcome_manifest.csv` | `2,636` | `316` |
| `imdb` | `datasets/exp3_meta/imdb_inference_manifest.csv` | `datasets/exp3_meta/imdb_outcome_manifest.csv` | `3,219` | `389` |
