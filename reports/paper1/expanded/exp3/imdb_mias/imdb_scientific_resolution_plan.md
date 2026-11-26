# IMDB (India) — Independent Scientific Resolution & Six-Model Inference Plan

**Date:** `2026-11-26`  
**Status:** **AUDIT COMPLETE — AWAITING `GO` FOR WAVE 1 (READY-4) -> WAVE 2 (GMIC + NYU)**  
**Modal Profile:** `tigerwood697` (`0` GPU inference launched on IMDB; `+$2.44` spent so far, **`$3.06` remaining** of `$5.50` expansion cap)

---

## 1. Exact Reconciliation of the `1,665` Four-View FFDM Cohort

Filtering `datasets/exp3_meta/imdb_inference_manifest.csv` (`12,674` images / `3,219` exams) strictly to `modality_subtype == "FFDM"` (`7,865` images / `2,228` exams) and requiring all 4 standard views (`L-CC`, `L-MLO`, `R-CC`, `R-MLO`) with zero view fabrication yields **`1,665` examinations (`6,660` images)**:

| Outcome-Reliability Category | `imdb_r1` (`MAMOS_1000000004`) | `imdb_r2` (`MAMOS_1000000050`) | Combined (`imdb`) | `outcome_label` |
|---|---:|---:|---:|---:|
| `pathology_confirmed_malignant` | `60` | `192` | **`252`** | `1` |
| `pathology_confirmed_benign` | `1` | `31` | **`32`** | `0` |
| `followup_confirmed_nonmalignant` | `0` | `0` | **`0`** | `0` |
| `noncancer_without_adequate_followup` | `243` | `1,136` | **`1,379`** | `0` |
| `unknown_or_ambiguous` (`BI-RADS 0` / discordant) | `0` | `2` (`1` pos, `1` neg) | **`2`** (`1` pos, `1` neg) | `1` / `0` |
| **Total 4-View FFDM Examinations** | **`304`** | **`1,361`** | **`1,665`** | **`253` original pos / `252` path-pos** |

- **Exact Arithmetic Check:** `252 + 32 + 1,379 + 2 = 1,665` (`304 + 1,361 = 1,665`).
- **Locked Manifests:** `datasets/exp3_meta/imdb_ffdm_4view_inference_manifest.csv` (`6,660` rows, label-blind) and `datasets/exp3_meta/imdb_ffdm_4view_outcome_manifest.csv` (`1,665` rows, protected).

---

## 2. Cross-Release Numeric ID Collisions (`582`) & Identity Limitations

- **Observed Collision:** `imdb_r1` uses zero-padded sequential IDs `0000`–`0582` (`583` IDs), while `imdb_r2` uses `pt_001`–`pt_2636` (`2,636` IDs). Direct string overlap is `0`, but **numeric-suffix overlap is `582`** (`0001`–`0582` vs `pt_001`–`pt_582`) because `deidentification/deidentify.py` re-initializes sequential IDs per release batch.
- **Why Release-Namespaced IDs Do Not Prove Unique Individuals:** Because all DICOM dates are neutralized to `20260101` and no patient DOB, MRN, or cryptographic patient hash is provided, **neither** identical numeric suffixes prove the same person **nor** do disjoint `StudyInstanceUID`s (`585` in `r1` vs `2,651` in `r2`, `0` overlap) prove distinct individuals (a returning patient in `r2` receives both a new `pt_XXX` and a new `StudyInstanceUID`).
- **Methodological Rule:** Release-namespacing (`imdb_r1_...` vs `imdb_r2_...`) prevents erroneous merging of unrelated patients sharing a sequential index, **but does not prove `1,665` (or `3,219`) independent individuals**. Therefore, **`imdb_r1` and `imdb_r2` must be analyzed separately as primary per-release cohorts**, and any combined `imdb` summary must explicitly disclose potential unlinked cross-release repeats.

---

## 3. Release-Wise & Combined Analysis Plan

1. **Release-Wise Primary Evaluation:**
   - **`imdb_r1` (`N = 304` 4-view FFDM, `60` pos):** Valid within-release patient-cluster bootstrap CIs (`304` clusters).
   - **`imdb_r2` (`N = 1,361` 4-view FFDM, `193` original pos / `192` path-pos):** Valid within-release patient-cluster bootstrap CIs (`1,361` clusters).
2. **Combined Secondary Evaluation (`imdb`, `N = 1,665`):**
   - Reported with explicit disclosure that cross-release patient independence is unverified.
   - **Never pooled** with `NLBS`, `CMMD`, `LUMINA`, or `MIAS`.
   - **Zero Indian label leakage:** All six models use frozen RSNA `val_calib` `98%` and `95%` thresholds; Indian labels are joined only after SHA-256 prediction locking.

---

## 4. Strict Pathology-Confirmed vs Secondary Original-Label Analyses

| Analysis Track | Cohort Definition | `imdb_r1` | `imdb_r2` | Combined (`imdb`) | Purpose |
|---|---|---|---|---|---|
| **Analysis A: Strict Pathology-Confirmed** | `pathology_confirmed_malignant` + `pathology_confirmed_benign` ONLY | `61` (`60` pos / `1` benign) | `223` (`192` pos / `31` benign) | `284` (`252` pos / `32` benign) | Evaluates cancer detection & biopsy-confirmed discrimination without unverified negatives. |
| **Analysis B: Secondary Original-Label Exploratory** | All clean original labels (`2` ambiguous excluded -> `1,663`; sensitivity w/ all `1,665`) | `304` (`60` pos / `244` neg) | `1,359` (`192` pos / `1,167` neg) | `1,663` (`252` pos / `1,411` neg) | Tests frozen-threshold calibration drift across the full tertiary-oncology FFDM cohort. |

---

## 5. Defensible Metrics Under Each Analysis

- **Under Analysis A (Pathology-Confirmed, `N = 284`):**
  - **Defensible:** Cancer sensitivity, missed cancer count, false-negative rate (`252` pos); `AUROC` & `AUPRC` on `imdb_r2` (`223` exams) and combined (`284` exams).
  - **Severely Underpowered / Caveated:** Specificity and FPR — **`imdb_r1` has only `1` pathology-confirmed benign case** (making `r1` Analysis A specificity/AUROC statistically uninformative), and `imdb_r2` has only `31` benign cases (wide `~±15%` CIs).
  - **Not Defensible:** Clinical screening **dismissal rate** or **false omission rate (FOR)** (biopsy-enriched `88.7%` malignant prevalence).
- **Under Analysis B (Original-Label Exploratory, `N = 1,663` / `1,665`):**
  - **Defensible:** Exploratory sensitivity, apparent specificity, apparent FPR, and apparent below-threshold rate at frozen RSNA `tau_98`/`tau_95` thresholds (quantifying score-distribution shift on Indian Hologic FFDM).
  - **Not Defensible:** True population-screening **dismissal utility**, **true specificity**, **NPV**, or **FOR**, because `1,379 / 1,411` (`97.7%`) non-cancer cases lack `12–24` month follow-up confirmation and `61.9%` of `r1` are prior-cancer follow-ups (`3.2%` screening).

---

## 6. Scientific Value & Exact Six-Model Cost Recalculation (`N = 1,665` / `6,660` Images)

### Scientific Value Assessment: **PROCEED (Matched 6-Model Hologic FFDM External Benchmark)**
- Running all six models on the exact same `1,665` 4-view FFDM cohort (`6,660` images) adds a **4th country (India)** and **Hologic Selenia Dimensions FFDM** with `252` pathology-confirmed cancers, enabling matched 6-model comparison on `imdb_r2` (`N=1,359` / `223` path-confirmed) and `imdb_r1` (`N=304`).
- **Explicit Boundary:** NLBS remains our **only** independent population-screening cohort; IMDB does not substitute for a second screening cohort.

### Exact Cost Estimate (`tigerwood697`, L40S @ `$2.3618/hr`, `$3.06` remaining of `$5.50` cap)
1. **Wave 1 — 4 View-Compatible Models (`6,660` cached `1520x912` PNGs, `num_shards=4`):**
   - `ce_b5` (`bs=32`, measured `23.94 img/s`): `278.2s` -> `$0.1825` (`~$0.20` w/ startup)
   - `trust_b5` (`bs=32`, `~24 img/s`): `277.5s` -> `$0.1821` (`~$0.20` w/ startup)
   - `dinov2_rsna` (`bs=32`, `~160 img/s`): `41.6s` -> `$0.0273` (`~$0.05` w/ startup)
   - `resnet50_rsna` (`bs=32`, `~140 img/s`): `47.6s` -> `$0.0312` (`~$0.05` w/ startup)
   - **Wave 1 Subtotal:** **`~$0.42` GPU (`~$0.50` w/ startup)** -> leaves `~$2.56` of `$5.50` cap.
2. **Wave 2 — GMIC + NYU Native-Resolution (`6,660` raw 12-bit Hologic FFDM DICOMs):**
   - `gmic_rsna_finetuned` (`bs=16`, native `2944x1920`, `~12–15 img/s`): `~444–555s` -> `$0.29–$0.36` (`~$0.42` w/ startup)
   - `nyu_fourview_rsna_finetuned` (`bs=4` exams, `1,665` exams, 10-TTA, `~1.3–1.8 exams/s`): `~925–1,280s` -> `$0.61–$0.84` (`~$0.90` w/ startup)
   - **Wave 2 Subtotal:** **`~$0.90–$1.20` GPU (`~$1.32` w/ startup)**.
3. **Total Projected 6-Model IMDB Inference:** **`~$1.32–$1.62` GPU (`~$1.82` w/ startup)** -> Cumulative expansion spend **`~$4.26 <= $5.50` cap** (`~$1.24` reserve under `$5.50`).
