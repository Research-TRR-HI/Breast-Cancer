# IMDB (India) Download, Integrity & DICOM Audit

**Dataset:** Indian Mammography DataBase (IMDB) — Release 1.0 (`MAMOS_1000000004`) & Release 2.0 (`MAMOS_1000000050`)  
**Modal Profile:** `tigerwood697` (`rsna-mammography-vol`)  
**Audit Date:** `2026-11-26`  
**Status:** **COMPLETE — `12,674 / 12,674` DICOMs downloaded & SHA-256 verified (`0` failures, `0` corrupt)**  

---

## 1. Programmatic Access & Concurrency Benchmark

Both `MAMOS_1000000004` (`r1`) and `MAMOS_1000000050` (`r2`) on IBIA (`https://ibdc.dbt.gov.in/ibia/study_details_browse/...`) are **open-access HTTP** with direct `.dcm` URLs and embedded HTML metadata tables (no login, no CAPTCHA, no anti-bot gate).

A `cpu=1.0, memory=2048` concurrency benchmark on `24` `r1` DICOMs (`424.30 MB`) established `16` streams as the throughput peak (server throttles at `32`):

| Streams | Files | OK / Fail | Elapsed (s) | Throughput (MB/s) | Files/s |
|---:|---:|---|---:|---:|---:|
| `8` | `24` | `24 / 0` | `30.41` | `13.95` | `0.79` |
| **`16`** | **`24`** | **`24 / 0`** | **`14.09`** | **`30.11`** | **`1.70`** |
| `32` | `24` | `24 / 0` | `28.32` | `14.98` | `0.85` |

---

## 2. Full Download & Integrity Verification (`rsna-mammography-vol`)

- **Download (`cpu=1.0, memory=2048`, `16` streams, 60s periodic `volume.commit()` + SIGINT/SIGTERM resume):**
  - **IMDB r1.0 (`/external_datasets/imdb/raw/r1/`):** `3,577 / 3,577` DICOMs (`0` failed), **`66,828,916,454` bytes (`62.2393 GiB`)**, `1,000.3s` (`63.71 MB/s`). Survived 1 Modal preemption with clean resume (`1,042` skipped, `2,535` downloaded).
  - **IMDB r2.0 (`/external_datasets/imdb/raw/r2/`):** `9,097 / 9,097` DICOMs (`0` failed), **`48,960,842,922` bytes (`45.5983 GiB`)**, `978.45s` (`47.72 MB/s`).
  - **Combined Total:** **`12,674 / 12,674` DICOMs**, **`115,789,759,376` bytes (`107.8376 GiB`)**, `0` failures.
- **Post-Download SHA-256 & DICOM Header Audit (`cpu=16.0, memory=8192` parallel `.map(["r1", "r2"])`):**
  - `r1`: `3,577` verified, **`0` corrupt**, `76.66s` -> `/rsna_vol/cache/imdb/r1_dicom_audit.csv`.
  - `r2`: `9,097` verified, **`0` corrupt**, `363.94s` -> `/rsna_vol/cache/imdb/r2_dicom_audit.csv`.

---

## 3. DICOM Acquisition & Cross-Release Verification

| Metric | `imdb_r1` (`MAMOS_1000000004`) | `imdb_r2` (`MAMOS_1000000050`) | Combined |
|---|---:|---:|---:|
| DICOM Files (`Unique SOPInstanceUID`) | `3,577` (`3,577`) | `9,097` (`9,097`) | `12,674` (`12,674`) |
| Unique `StudyInstanceUID` | `585` | `2,651` | `3,236` (`0` overlap) |
| Manufacturer / Model | `HOLOGIC, Inc.` / `Selenia Dimensions` (`100%`) | `HOLOGIC, Inc.` / `Selenia Dimensions` (`100%`) | `100%` Hologic |
| Photometric Interpretation | `MONOCHROME2` (`100%`) | `MONOCHROME2` (`100%`) | `100%` `MONOCHROME2` |
| `12-bit` (`FFDM`: `4096x3328` / `3328x2560`) | `1,869` (`1,786` / `83`) | `5,996` (`5,749` / `247`) | `7,865` |
| `10-bit` (`SM` C-View: `2457x1996` / `2457x1890`) | `1,708` (`1,634` / `74`) | `3,101` (`3,082` / `19`) | `4,809` |

- **Exact 1-to-1 Modality Match:** Every `12-bit` image matches IBIA `2D-Mammo` (`FFDM`), and every `10-bit` image matches IBIA `C-View`/`C-view` (`SM`).

---

## 4. Modal Cost & Storage Summary (`tigerwood697`)

- **Pre-Task Ephemeral Apps:** `$9.94` -> **Post-Task Ephemeral Apps:** `$10.20` (**`+$0.26` total compute spend** across benchmark, full download, and 16-CPU SHA-256/DICOM audit).
- **Billed Cost:** **`$0.00`** (`0` running apps via `modal app list`).
- **Recommendation:** **GO** (Conditional on FFDM vs SM stratification & non-screening diagnostic framing — see `imdb_label_audit.md` and `imdb_model_eligibility.md`).
