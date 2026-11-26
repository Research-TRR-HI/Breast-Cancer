# Paper 1 — Final External Dataset Expansion Summary (IMDB India + MIAS UK)

**Date:** `2026-11-26`  
**Repository:** `https://github.com/Research-TRR-HI/Breast-Cancer`  
**Modal Account:** `tigerwood697` (`0` running apps, `$0.00` billed, `$6.54` free Starter credits remaining)

---

## 1. Mandatory Final Dataset Table

| Dataset | Downloaded | Verified files | Unique patients | Examinations | Cancer-positive | Non-cancer | Four-view eligible | Outcome quality |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **IMDB r1** (`MAMOS_1000000004`) | Yes (`62.24 GiB`) | `3,577 / 3,577` DICOMs (`0` corrupt) | `583` (`0000–0582`, release-namespaced) | `583` (`585` StudyUIDs) | `73` (`72` path-confirmed, `1` ambiguous) | `510` (`1` path-benign, `509` no follow-up) | `304` FFDM (`60` pos, `1` path-benign, `243` no follow-up) | Mixed tertiary oncology (`61.9%` prior-cancer FU, `3.2%` screening); dates neutralized (`0` longitudinal FU) |
| **IMDB r2** (`MAMOS_1000000050`) | Yes (`45.60 GiB`) | `9,097 / 9,097` DICOMs (`0` corrupt) | `2,636` (`pt_001–pt_2636`, `582` numeric collisions w/ `r1`) | `2,636` (`2,651` StudyUIDs) | `316` (`315` path-confirmed, `1` ambiguous) | `2,320` (`49` path-benign, `2,267` no follow-up, `4` ambiguous) | `1,361` FFDM (`192` path-pos, `31` path-benign, `1,136` no follow-up, `2` ambiguous) | Mixed tertiary oncology; `192` path-confirmed pos + `31` path-confirmed benign (`0` longitudinal FU) |
| **MIAS** (`CAM.105113`) | Yes (`1.51 GiB`) | `322 / 322` PGMs (`0` corrupt) | `161` pairs (no separate patient ID) | `161` bilateral MLO pairs | `52` malignant pairs (`32.3%`) | `109` (`56` benign-only, `53` normal) | **`0`** (`0` CC views) | 1994 UK NHSBSP digitized screen-film; no per-lesion pathology/BI-RADS provenance |

---

## 2. Mandatory Model Eligibility Table

| Model | IMDB FFDM (`1,665` 4v) | IMDB SM (`4,809` imgs) | MIAS (`161` pairs) | Reason if ineligible |
|---|---|---|---|---|
| `ce_b5` | **ELIGIBLE (Evaluated)** | Eligible (Deferred) | **CONDITIONAL (Evaluated)** | MIAS is 8-bit digitized film rotated 90° upright (exploratory domain-shift only) |
| `trust_b5` | **ELIGIBLE (Evaluated)** | Eligible (Deferred) | **CONDITIONAL (Evaluated)** | Same as `ce_b5` |
| `dinov2_rsna` | **ELIGIBLE (Evaluated)** | Eligible (Deferred) | **CONDITIONAL (Evaluated)** | Same as `ce_b5` |
| `resnet50_rsna` | **ELIGIBLE (Evaluated)** | Eligible (Deferred) | **CONDITIONAL (Evaluated)** | Same as `ce_b5` |
| `gmic_rsna_finetuned` | **ELIGIBLE (Evaluated)** | Eligible (Deferred) | **CONDITIONAL (Evaluated)** | Native `2944x1920` crop from 8-bit film (`5` fallbacks on MIAS, `0` on IMDB) |
| `nyu_fourview_rsna_finetuned` | **ELIGIBLE (Evaluated)** | Eligible (Deferred) | **INELIGIBLE (HARD-BLOCKED)** | MIAS has **0 CC views** (`L-CC`/`R-CC` absent; zero view fabrication) |

---

## 3. Cross-Dataset Scientific Synthesis (NLBS vs CMMD vs LUMINA vs IMDB vs MIAS)

1. **NLBS (Canada, `N=6,184` 4-view, `151` pos) Remains Our ONLY Population-Screening Cohort:**
   - Only NLBS supports valid population-screening dismissal claims (`gmic_rsna_finetuned` `0.8624` AUROC, `96.03%` @98% sens, `33.15%` dismissal; `nyu_fourview_rsna_finetuned` `0.7846` AUROC, `96.69%` @98% sens, `14.68%` dismissal).
2. **CMMD (China, `N=826` 4-view) & LUMINA (Turkey, `N=468`):**
   - Biopsy-enriched diagnostic cohorts (`gmic_rsna_finetuned` `0.7059` / `0.7426` AUROC).
3. **IMDB (India, `N=1,665` 4-view Hologic FFDM — `r1`: `304`, `r2`: `1,361`):**
   - **All 6 RSNA-trained models drop to near-chance discrimination (`AUROC 0.5035 – 0.5515` combined; `0.5040 – 0.5632` on `r2` path-confirmed `N=223`)** due to severe clinical-spectrum shift (`61.9%` prior-cancer follow-up / post-treatment breasts, `3.2%` screening).
   - **Severe upward score shift at frozen RSNA `tau_98`:** `nyu_fourview_rsna_finetuned` flags `99.22%` of all exams (`99.60%` sens, `1` missed, `0.85%` apparent spec), `dinov2_rsna` (`99.21%` sens, `0.35%` spec), `resnet50_rsna` (`98.41%` sens, `1.28%` spec), `gmic_rsna_finetuned` (`96.83%` sens, `4.25%` spec), while `ce_b5` (`93.25%` sens, `17` missed, `11.91%` spec) and `trust_b5` (`91.27%` sens, `22` missed, `14.67%` spec) still miss cancers despite flagging `>85%` of exams.
4. **MIAS (UK, `N=161` digitized screen-film pairs):**
   - **Catastrophic SFM vs FFDM domain collapse (`AUROC 0.4486 – 0.5567`):** `dinov2_rsna` (`7.69%` sens, `48/52` missed) and `gmic_rsna_finetuned` (`15.38%` sens, `44/52` missed) invert completely below frozen RSNA thresholds (`>90%` below-threshold, `FOR 30.14%–32.21%`).

---

## 4. Honest Scientific Recommendation & Remaining Stage 7 Tasks

- **IMDB Recommendation:** **MODIFY / CONDITIONAL PROCEED** — Include strictly as a **non-screening tertiary-oncology / post-treatment FFDM failure-mode stress test** (reporting `r1` and `r2` separately, Analysis A `N=223`/`284` vs Analysis B `N=1,663`, zero dismissal claims).
- **MIAS Recommendation:** **MODIFY / CONDITIONAL PROCEED** — Include strictly as an **exploratory legacy-film negative boundary condition** (5 view-agnostic models only, zero screening claims).
- **Combined Paper 1 Contribution:** **STRENGTHENS PAPER 1 (NOT automatically Q1-ready)** — Broadens external evidence across 5 countries (`CA`, `CN`, `TR`, `IN`, `UK`) and 3 acquisition regimes (FFDM screening, FFDM diagnostic/post-treatment, digitized screen-film), proving that source-frozen RSNA models fail for **two distinct, opposite reasons**: upward score shift / post-treatment confounding on Indian tertiary oncology FFDM vs downward score collapse on UK digitized film.
- **Remaining Stage 7 Tasks:**
  1. Finalize Paper 1 main tables/figures integrating NLBS (primary screening) + CMMD/LUMINA/IMDB (diagnostic/oncology FFDM) + MIAS (exploratory SFM).
  2. Write Paper 1 manuscript methods, limitations (non-screening external cohorts, unlinked `r1`/`r2` IDs, `0` longitudinal FU in IMDB), and adversarial-literature positioning (*Meta-Repository* 2026, *Sensors* 2026).
  3. Decide Paper 2 (`AnchorShift` Stage D UPA) launch timing.
