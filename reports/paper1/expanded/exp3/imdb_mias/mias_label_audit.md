# MIAS v1.21 Truth-Data & Reference-Standard Audit

**Source:** `MIASDBv1.21/00README.pdf` (12 pages, Version 1.21, 25 August 2015; Truth-Data: C R M Boggis & Ian Hutt; Reference: Suckling et al. 1994)  
**Cross-Check Result:** **322 / 322** PGM images (`mdb001`–`mdb322`) match `00README.pdf` truth entries (`330` total truth rows).

---

## 1. Cohort Structure & Pair-to-Exam Mapping

- **Images:** `322` single-view mediolateral oblique (**MLO**) digitized screen-film mammograms (`161` left `l` with odd IDs `mdb001`..`mdb321`, `161` right `r` with even IDs `mdb002`..`mdb322`).
- **Exams / Pairs:** `161` bilateral pairs (`mias_pair_001`..`mias_pair_161`), where each odd/even pair represents the left and right MLO mammograms of a single individual (`00README.pdf` Page 2, Note 1).
- **Views Available:** **MLO only** (`L-MLO`, `R-MLO`). **Zero craniocaudal (CC) views** exist in MIAS.

---

## 2. Image-Level & Pair-Level Label Distribution

### Image-Level (`n = 322`)
- **Normal (`NORM`):** `207` (`64.3%`)
- **Benign-only (`B`):** `63` (`19.6%`)
- **Malignant-only (`M`):** `51` (`15.8%`)
- **Mixed Benign + Malignant (`B` and `M`):** `1` (`mdb144rx`: `MISC B` + `MISC M`)
- **Total Malignant-Containing Images:** `52` (`16.1%`) | **Total Abnormal Images:** `115` (`35.7%`)

### Background Tissue (`n = 322`)
- **Dense-glandular (`D`):** `112`
- **Fatty (`F`):** `106`
- **Fatty-glandular (`G`):** `104`

### Row-Level Abnormality × Severity Breakdown (`330` rows = `123` lesion rows + `207` `NORM`)
| Abnormality Code | Description | Benign (`B`) | Malignant (`M`) | Total Lesion Rows |
| :--- | :--- | :---: | :---: | :---: |
| `CALC` | Calcification | `15` | `15` | `30` |
| `CIRC` | Well-defined / circumscribed masses | `21` | `4` | `25` |
| `SPIC` | Spiculated masses | `11` | `8` | `19` |
| `ARCH` | Architectural distortion | `9` | `10` | `19` |
| `MISC` | Other, ill-defined masses | `7` | `8` | `15` |
| `ASYM` | Asymmetry | `6` | `9` | `15` |
| **Total Abnormal Rows** | | **`69`** | **`54`** | **`123`** |

- **Multi-Lesion Images (`7`):** `mdb005ll` (2 `CIRC B`), `mdb132rx` (2 `CIRC B`), `mdb144rx` (`MISC B` + `MISC M`), `mdb223ls` (2 `CALC B`), `mdb226rm` (3 `CALC B`), `mdb239ll` (2 `CALC M`), `mdb249lm` (2 `CALC M`).
- **Abnormal Rows Without `(x, y, r)` Coordinates (`6`):** `mdb059ls` (`CIRC B`), `mdb212rm` (`CALC B`), `mdb214rs` (`CALC B`), plus `3` widely distributed malignant calcifications (`mdb216rl`, `mdb233lm`, `mdb245ls`, marked `*** see note 2`).

### Pair-Level / Exam-Level (`n = 161`)
- **Malignant Pair (≥1 `M`):** `52` (`32.3%`) — all `52` malignant-containing images occur in distinct pairs.
- **Benign-Only Pair (≥1 `B`, no `M`):** `56` (`34.8%`)
- **Normal Pair (both `NORM`):** `53` (`32.9%`)
- **Bilateral Abnormal Pairs (`7`):** `1` bilateral benign (`mdb001lm`/`mdb002rl`), and `6` pairs with malignant in one breast and benign in the contralateral breast (`mdb091/092`, `mdb125/126`, `mdb141/142`, `mdb211/212`, `mdb213/214`, `mdb239/240`).

---

## 3. Reference-Standard Limitations & Scientific Guardrails

1. **Historical 1994 UK NHSBSP Screen-Film Reference Standard:** Labels (`NORM`, `B`, `M`) were curated by C. R. M. Boggis and I. Hutt (1994) from UK screening films. `00README.pdf` does **not** provide per-lesion pathology/histology provenance (e.g., biopsy vs negative follow-up duration for `NORM`/`B`), nor modern BI-RADS density/assessment scores.
2. **Image-Level Annotation Mapped to Pair-Level Exams:** Truth rows are annotated per image/lesion (`mdbXXX`). Because each odd/even pair is one individual's L/R MLO exam, exam-level evaluation (`mias_outcome_manifest.csv`) defines `outcome_label = 1` iff any lesion in the pair has severity `M` (`52/161`), while preserving per-breast/per-image flags in `eligibility_flags`.
3. **Diagnostic/Enriched Prevalence (`32.3%` Malignant Exams):** MIAS is heavily enriched for abnormalities (`108/161` abnormal pairs, `52/161` malignant) and is **strictly ineligible** for population-screening dismissal claims (`population_screening_dismissal_allowed = False`).
4. **Manifest Quarantine:**
   - Label-blind inference manifest: `datasets/exp3_meta/mias_inference_manifest.csv` (`322` rows, validated by `validate_inference_manifest`).
   - Protected outcome manifest: `datasets/exp3_meta/mias_outcome_manifest.csv` (`161` rows, validated by `validate_outcome_manifest`).
