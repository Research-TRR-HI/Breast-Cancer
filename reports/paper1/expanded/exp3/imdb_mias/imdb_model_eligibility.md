# IMDB (India) 6-Model Eligibility & Cohort Design

**Dataset:** IMDB (`imdb_r1` + `imdb_r2`, `N = 12,674` images / `3,219` exams / `389` positives)  
**Audit Date:** `2026-11-26`  
**Protocol:** Paper 1 Exp 3 Isolated External Validation (`src/paper1/universal_eval.py`)  

---

## 1. Exact Release-Wise & Combined Counts (Verified from Manifests)

| Metric | `imdb_r1` (`MAMOS_1000000004`) | `imdb_r2` (`MAMOS_1000000050`) | Combined (`imdb`, release-namespaced) |
|---|---:|---:|---:|
| Unique patients (`cluster_id`) | `583` | `2,636` | `3,219` |
| Examinations (`exam_id`) | `583` (`585` StudyUIDs) | `2,636` (`2,651` StudyUIDs) | `3,219` (`3,236` StudyUIDs) |
| Total images | `3,577` | `9,097` | `12,674` |
| `FFDM` (`2D-Mammo`, 12-bit) | `1,869` imgs (`583` exams) | `5,996` imgs (`1,645` exams) | `7,865` imgs (`2,228` exams) |
| `SM` (`C-View`, 10-bit) | `1,708` imgs (`549` exams) | `3,101` imgs (`1,051` exams) | `4,809` imgs (`1,600` exams) |
| **Complete 4-view `FFDM` exams** | **`304`** (`1,216` imgs) | **`1,361`** (`5,444` imgs) | **`1,665`** (`6,660` imgs) |
| Complete 4-view `SM` exams | `285` (`1,140` imgs) | `493` (`1,972` imgs) | `778` (`3,112` imgs) |
| Cancer-pos (`pathology_confirmed_malignant`) | `72` all / `60` 4v-FFDM | `315` all / `192` 4v-FFDM | `387` all / `252` 4v-FFDM |
| Cancer-neg (`pathology_confirmed_benign`) | `1` all / `1` 4v-FFDM | `49` all / `31` 4v-FFDM | `50` all / `32` 4v-FFDM |
| Cancer-neg (`followup_confirmed_nonmalignant`) | `0` | `0` | `0` |
| `noncancer_without_adequate_followup` | `509` all / `243` 4v-FFDM | `2,267` all / `1,136` 4v-FFDM | `2,776` all / `1,379` 4v-FFDM |
| `unknown_or_ambiguous` (`BI-RADS 0`/discordant) | `1` pos (`0` 4v-FFDM) | `5` (`4` neg, `1` pos; `2` 4v-FFDM) | `6` (`4` neg, `2` pos; `2` 4v-FFDM) |

- **Cross-Release Identity:** `0` string overlap (`0000–0582` vs `pt_001–pt_2636`), `582` numeric-suffix collisions (`0001–0582` vs `pt_001–pt_582`), `0` `StudyInstanceUID` overlap. All manifests namespace by release (`imdb_r1_...` / `imdb_r2_...`).

---

## 2. 6-Model Compatibility Matrix on Common 4-View FFDM Cohort (`N = 1,665` exams / `6,660` images)

| Model ID | View Mode | Native Input & Preprocessing | Frozen RSNA Thresholds (`98%` / `95%`) | Eligible on 4v-FFDM (`1,665` exams) |
|---|---|---|---|---|
| `ce_b5` | `single_image` -> exam max | `1520x912 uint16`, right-mirrored, Mammo-CLIP norm | `0.096193` / `0.124415` | **YES** (`304` `r1` + `1,361` `r2`) |
| `trust_b5` | `single_image` -> exam max | `1520x912 uint16`, right-mirrored, Mammo-CLIP norm | `0.179548` / `0.195463` | **YES** (`304` `r1` + `1,361` `r2`) |
| `dinov2_rsna` | `single_image` -> exam max | `1520x912 uint16` -> pad square `518x518` ImageNet norm | `0.183436` / `0.206516` | **YES** (`304` `r1` + `1,361` `r2`) |
| `resnet50_rsna` | `single_image` -> exam max | `1520x912 uint16` -> `854x512` ImageNet norm | `0.000993` / `0.006931` | **YES** (`304` `r1` + `1,361` `r2`) |
| `gmic_rsna_finetuned` | `single_image` -> exam max | Native `2944x1920 uint16` crop from `4096x3328`/`3328x2560` FFDM | `0.130208` / `0.158398` | **YES** (`304` `r1` + `1,361` `r2`) |
| `nyu_fourview_rsna_finetuned` | `four_view_exam` | Native `CC 2677x1942 / MLO 2974x1748 uint16`, 10-TTA | `0.047732` / `0.069086` | **YES** (`304` `r1` + `1,361` `r2`) |

---

## 3. Scientific Guardrails & Locked Cohort Manifests

1. **Matched 4-View FFDM Manifests:** `datasets/exp3_meta/imdb_ffdm_4view_inference_manifest.csv` (`6,660` images, label-blind) and `datasets/exp3_meta/imdb_ffdm_4view_outcome_manifest.csv` (`1,665` exams, `253` pos: `252` pathology-confirmed + `1` ambiguous).
2. **No Population Screening Dismissal Claims:** `61.9%` of `r1` is prior-cancer follow-up and only `3.2%` is screening (`population_screening_dismissal_allowed = False`).
3. **Zero Label Leakage:** Indian cancer labels are strictly quarantined until after SHA-256 prediction locking; thresholds remain frozen from RSNA `val_calib`.
