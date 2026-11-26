# IMDB (India) 6-Model Eligibility & Cohort Design

**Dataset:** IMDB (`imdb_r1` + `imdb_r2`, `N = 12,674` images / `3,219` exams / `389` positives)  
**Audit Date:** `2026-11-26`  
**Protocol:** Paper 1 Exp 3 Isolated External Validation (`src/paper1/universal_eval.py`)  

---

## 1. 6-Model Compatibility Matrix

| Model ID | View Mode | Requires 4-View | Native Input Spec | Eligible on `imdb_r1` | Eligible on `imdb_r2` | Eligible on `imdb` (Combined) |
|---|---|---|---|---:|---:|---:|
| `ce_b5` | `single_image` | No | `1520x912 uint16`, right-mirrored | **Yes** (`3,577` imgs / `583` exams) | **Yes** (`9,097` imgs / `2,636` exams) | **Yes** (`12,674` imgs / `3,219` exams) |
| `trust_b5` | `single_image` | No | `1520x912 uint16`, right-mirrored | **Yes** (`3,577` imgs / `583` exams) | **Yes** (`9,097` imgs / `2,636` exams) | **Yes** (`12,674` imgs / `3,219` exams) |
| `dinov2_rsna` | `single_image` | No | `1520x912 uint16` -> `518x518` | **Yes** (`3,577` imgs / `583` exams) | **Yes** (`9,097` imgs / `2,636` exams) | **Yes** (`12,674` imgs / `3,219` exams) |
| `resnet50_rsna` | `single_image` | No | `1520x912 uint16` -> `854x512` | **Yes** (`3,577` imgs / `583` exams) | **Yes** (`9,097` imgs / `2,636` exams) | **Yes** (`12,674` imgs / `3,219` exams) |
| `gmic_rsna_finetuned` | `single_image` | No | Native `2944x1920 uint16` | **Yes** (`3,577` imgs / `583` exams) | **Yes** (`9,097` imgs / `2,636` exams) | **Yes** (`12,674` imgs / `3,219` exams) |
| `nyu_fourview_rsna_finetuned` | `four_view_exam` | **Yes** | Native `CC 2677x1942 / MLO 2974x1748 uint16` | **Yes (4-view subset)** (`304` exams, `60` pos) | **Yes (4-view subset)** (`1,849` exams, `278` pos) | **Yes (4-view subset)** (`2,153` exams, `338` pos) |

---

## 2. 4-View Completeness & Modality Sub-Cohorts (Zero View Fabrication)

Per `select_common_4view_cohort(..., prefer_modality_subtypes=("FFDM",))`, exams missing any of `{L-CC, L-MLO, R-CC, R-MLO}` are **strictly excluded** from `nyu_fourview_rsna_finetuned` and common-cohort comparisons (never duplicated or synthesized):

| Cohort | Full Exams (`Pos`) | 4-View Combined (`Pos`) | 4-View `FFDM` Only (`Pos`) | 4-View `SM` (`C-View`) Only (`Pos`) |
|---|---:|---:|---:|---:|
| `imdb_r1` | `583` (`73`) | `304` (`60`) | `304` (`60`) | `285` (`56`) |
| `imdb_r2` | `2,636` (`316`) | `1,849` (`278`) | `1,361` (`193`) | `493` (`85`) |
| **Combined (`imdb`)** | **`3,219` (`389`)** | **`2,153` (`338`)** | **`1,665` (`253`)** | **`778` (`141`)** |

---

## 3. Scientific Guardrails for Phase 2 Execution

1. **No Population Screening Dismissal Claims:** Because `61.9%` of `r1` is prior-cancer follow-up and only `3.2%` is screening (`population_screening_dismissal_allowed = False`), IMDB must be reported strictly as a **mixed diagnostic / tertiary oncology external validation** (`AUROC`, `AUPRC`, sensitivity/specificity at frozen RSNA `val_calib` thresholds), NEVER as population screening dismissal.
2. **FFDM vs SM Stratification:** Single-view models run on all `12,674` images, and evaluation MUST stratify:
   - Primary 6-model common 4-view comparison (`N = 2,153` exams, `338` pos; `FFDM`-preferred).
   - `FFDM`-only sensitivity (`1,665` 4-view / `2,228` all exams) vs `SM` (`C-View`)-only sensitivity (`778` 4-view / `1,600` all exams).
   - Release-stratified (`imdb_r1` vs `imdb_r2`) sensitivity to verify no batch/acquisition drift.
3. **Outcome Reliability Sensitivity:** Exclude the `6` `unknown_or_ambiguous` exams in a clean sensitivity check (`N = 3,213` exams, `387` pos).
