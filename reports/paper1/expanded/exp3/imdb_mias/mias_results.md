# MIAS v1.21 (UK) — Exploratory Film-Domain Evaluation Results

**Dataset:** MIAS v1.21 (`161` bilateral MLO exam pairs / `322` digitized screen-film PGMs, `52` malignant exams, `109` non-malignant exams; `0` CC views)  
**Clinical Role:** `legacy_film_exploratory` (`population_screening_dismissal_allowed = False`)  
**Decision:** **CONDITIONAL GO (Exploratory Film-Domain Stress Test ONLY — NO-GO for Population Screening Dismissal or 4-View Models)**  
**Modal Execution (`tigerwood697`):** `0` fallbacks in Stage-1 `1520x912` cache (`322/322`); `5` crop fallbacks in `gmic_rsna_finetuned`; `$0.1889` estimated GPU spend across 5 eligible models.

---

## 1. Domain-Transfer Preprocessing & Eligibility Disclosure

1. **90° Upright Rotation + Bit-Depth Scaling:** Raw MIAS PGMs are 8-bit (`0–255`) landscape (`width > height`, chest wall at top edge). `decode_pgm_array()` rotated all 322 images 90° upright (`L` CCW -> chest wall left; `R` CW -> chest wall right before horizontal mirroring) and rescaled to `uint16` (`1520×912` Stage-1 / `2944×1920` GMIC).
2. **`nyu_fourview_rsna_finetuned`:** **NO-GO (HARD-BLOCKED)** — MIAS has `0` CC views; views were never fabricated.
3. **Reference-Standard Caveat:** 1994 UK NHSBSP screen-film curation (`32.3%` malignant enrichment) without per-lesion pathology/BI-RADS provenance.

---

## 2. Exploratory Evaluation at Frozen RSNA `val_calib` Thresholds (`N=161`, `161` clusters, `52` pos)

> **Warning:** `Below-Thr Rate` reflects score-distribution collapse on digitized film, **NOT** clinical screening dismissal utility.

### Primary Threshold (`RSNA 98%` Source Sensitivity)

| Model | AUROC (95% CI) | AUPRC (95% CI) | TP | TN | FP | FN (Missed) | Sens (95% CI) | Spec | FPR | FNR | Below-Thr Rate | FOR | Non-Cancer Retained |
|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| `ce_b5` | `0.5258` (`0.4360–0.6191`) | `0.3426` (`0.2613–0.4654`) | 49 | 13 | 96 | **3** | `94.23%` (`86.79–100.00%`) | `11.93%` | `88.07%` | `5.77%` | `9.94%` | `18.75%` | 96 |
| `trust_b5` | `0.5319` (`0.4435–0.6247`) | `0.3419` (`0.2619–0.4646`) | 45 | 26 | 83 | **7** | `86.54%` (`76.19–95.31%`) | `23.85%` | `76.15%` | `13.46%` | `20.50%` | `21.21%` | 83 |
| `dinov2_rsna` | `0.5567` (`0.4505–0.6529`) | `0.3981` (`0.3005–0.5340`) | 4 | 101 | 8 | **48** | `7.69%` (`1.69–15.56%`) | `92.66%` | `7.34%` | `92.31%` | `92.55%` | `32.21%` | 8 |
| `resnet50_rsna` | `0.4486` (`0.3530–0.5492`) | `0.3064` (`0.2252–0.4144`) | 34 | 33 | 76 | **18** | `65.38%` (`52.50–78.18%`) | `30.28%` | `69.72%` | `34.62%` | `31.68%` | `35.29%` | 76 |
| `gmic_rsna_finetuned` | `0.5453` (`0.4502–0.6397`) | `0.3965` (`0.2941–0.5244`) | 8 | 102 | 7 | **44** | `15.38%` (`6.00–26.23%`) | `93.58%` | `6.42%` | `84.62%` | `90.68%` | `30.14%` | 7 |

### Secondary Threshold (`RSNA 95%` Source Sensitivity)

| Model | TP | TN | FP | FN (Missed) | Sens (95% CI) | Spec | FPR | FNR | Below-Thr Rate | FOR | Non-Cancer Retained |
|---|---:|---:|---:|---:|---|---|---|---|---|---|---:|
| `ce_b5` | 45 | 26 | 83 | **7** | `86.54%` (`76.19–95.31%`) | `23.85%` | `76.15%` | `13.46%` | `20.50%` | `21.21%` | 83 |
| `trust_b5` | 45 | 28 | 81 | **7** | `86.54%` (`76.19–95.31%`) | `25.69%` | `74.31%` | `13.46%` | `21.74%` | `20.00%` | 81 |
| `dinov2_rsna` | 3 | 107 | 2 | **49** | `5.77%` (`0.00–12.96%`) | `98.17%` | `1.83%` | `94.23%` | `96.89%` | `31.41%` | 2 |
| `resnet50_rsna` | 15 | 68 | 41 | **37** | `28.85%` (`16.00–41.67%`) | `62.39%` | `37.61%` | `71.15%` | `65.22%` | `35.24%` | 41 |
| `gmic_rsna_finetuned` | 7 | 105 | 4 | **45** | `13.46%` (`4.84–24.00%`) | `96.33%` | `3.67%` | `86.54%` | `93.17%` | `30.00%` | 4 |

---

## 3. Key Scientific Takeaways

1. **Catastrophic SFM vs FFDM Domain Collapse:** Every RSNA-trained FFDM model collapses to near-chance discrimination on digitized screen-film (`AUROC 0.4486 – 0.5567`, all 95% CIs straddling or near `0.50`).
2. **Severe Inverted Threshold Failure:** `dinov2_rsna` (`7.69%` sens, `48/52` missed) and `gmic_rsna_finetuned` (`15.38%` sens, `44/52` missed) score digitized film dramatically lower than RSNA FFDM, pushing `>90%` of cases below the frozen RSNA `tau_98` cutoff (`FOR 30.14%–32.21%`).
3. **Paper 1 Role:** Strengthens Paper 1's domain-shift narrative as an explicit **legacy-film negative boundary condition**, proving RSNA-calibrated FFDM models cannot generalize zero-shot to digitized film.
