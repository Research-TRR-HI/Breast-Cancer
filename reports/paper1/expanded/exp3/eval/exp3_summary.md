# Paper 1 Exp 3 — Multi-Dataset External Validation Summary

> **Clinical Pooling Policy:** Per-dataset reporting ONLY. Screening (`nlbs`), mixed (`imdb`), and biopsy-enriched diagnostic (`cmmd`, `lumina`) cohorts are NEVER pooled.

## Dataset: `NLBS` (Canada — `screening`)
- Screening Dismissal Claim Allowed: `True`

### Common 4-View Cohort (`N=6184`, clusters=`5868`, pos=`151`)

| Model | AUROC | AUPRC | @98% Sens | @98% Below-Thr Rate | @95% Sens | @95% Below-Thr Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ce_b5` | 0.7089 | 0.0548 | 90.07% (15 missed) | 30.48% | 80.13% (30 missed) | 49.66% |
| `trust_b5` | 0.7040 | 0.0539 | 84.77% (23 missed) | 39.57% | 80.79% (29 missed) | 49.29% |
| `dinov2_rsna` | 0.6299 | 0.0425 | 98.01% (3 missed) | 9.67% | 94.04% (9 missed) | 17.43% |
| `resnet50_rsna` | 0.6452 | 0.1039 | 97.35% (4 missed) | 5.35% | 84.77% (23 missed) | 19.78% |
| `gmic_rsna_finetuned` | 0.8624 | 0.3345 | 96.03% (6 missed) | 33.15% | 92.05% (12 missed) | 40.96% |
| `nyu_fourview_rsna_finetuned` | 0.7846 | 0.1599 | 96.69% (5 missed) | 14.68% | 91.39% (13 missed) | 35.22% |

### Full Cohort (View-Agnostic) (`N=6359`, clusters=`5942`, pos=`151`)

| Model | AUROC | AUPRC | @98% Sens | @98% Below-Thr Rate | @95% Sens | @95% Below-Thr Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ce_b5` | 0.7101 | 0.0539 | 90.07% (15 missed) | 30.62% | 80.13% (30 missed) | 49.90% |
| `trust_b5` | 0.7052 | 0.0529 | 84.77% (23 missed) | 39.77% | 80.79% (29 missed) | 49.57% |
| `dinov2_rsna` | 0.6301 | 0.0416 | 98.01% (3 missed) | 9.66% | 94.04% (9 missed) | 17.39% |
| `resnet50_rsna` | 0.6452 | 0.1027 | 97.35% (4 missed) | 5.38% | 84.77% (23 missed) | 19.77% |
| `gmic_rsna_finetuned` | 0.8637 | 0.3338 | 96.03% (6 missed) | 33.65% | 92.05% (12 missed) | 41.42% |

## Dataset: `CMMD` (China — `diagnostic_biopsy_enriched`)
- Screening Dismissal Claim Allowed: `False`

### Common 4-View Cohort (`N=826`, clusters=`826`, pos=`765`)

| Model | AUROC | AUPRC | @98% Sens | @98% Below-Thr Rate | @95% Sens | @95% Below-Thr Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ce_b5` | 0.6414 | 0.9583 | 48.76% (392 missed) | 52.78% | 35.29% (495 missed) | 66.22% |
| `trust_b5` | 0.6474 | 0.9588 | 45.36% (418 missed) | 56.42% | 38.43% (471 missed) | 63.20% |
| `dinov2_rsna` | 0.6567 | 0.9558 | 70.98% (222 missed) | 30.75% | 59.08% (313 missed) | 42.37% |
| `resnet50_rsna` | 0.4741 | 0.9303 | 99.35% (5 missed) | 0.61% | 95.29% (36 missed) | 4.60% |
| `gmic_rsna_finetuned` | 0.7059 | 0.9699 | 96.86% (24 missed) | 3.51% | 93.07% (53 missed) | 7.14% |
| `nyu_fourview_rsna_finetuned` | 0.7137 | 0.9674 | 92.94% (54 missed) | 7.51% | 81.57% (141 missed) | 19.73% |

### Full Cohort (View-Agnostic) (`N=1775`, clusters=`1775`, pos=`1310`)

| Model | AUROC | AUPRC | @98% Sens | @98% Below-Thr Rate | @95% Sens | @95% Below-Thr Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ce_b5` | 0.6631 | 0.8484 | 44.58% (726 missed) | 61.63% | 32.60% (883 missed) | 72.90% |
| `trust_b5` | 0.6654 | 0.8494 | 41.53% (766 missed) | 64.56% | 35.11% (850 missed) | 70.76% |
| `dinov2_rsna` | 0.6794 | 0.8460 | 64.66% (463 missed) | 42.42% | 53.36% (611 missed) | 53.24% |
| `resnet50_rsna` | 0.5776 | 0.8002 | 98.93% (14 missed) | 1.01% | 93.13% (90 missed) | 7.72% |
| `gmic_rsna_finetuned` | 0.7671 | 0.9075 | 94.81% (68 missed) | 8.96% | 91.15% (116 missed) | 13.52% |

## Dataset: `LUMINA` (Turkey — `diagnostic_biopsy_enriched`)
- Screening Dismissal Claim Allowed: `False`

### Full Cohort (View-Agnostic) (`N=468`, clusters=`468`, pos=`218`)

| Model | AUROC | AUPRC | @98% Sens | @98% Below-Thr Rate | @95% Sens | @95% Below-Thr Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ce_b5` | 0.6693 | 0.6654 | 95.87% (9 missed) | 2.35% | 93.12% (15 missed) | 5.13% |
| `trust_b5` | 0.6708 | 0.6688 | 94.04% (13 missed) | 3.42% | 93.12% (15 missed) | 4.49% |
| `dinov2_rsna` | 0.5730 | 0.5879 | 92.20% (17 missed) | 6.62% | 86.70% (29 missed) | 10.90% |
| `resnet50_rsna` | 0.5753 | 0.5713 | 89.91% (22 missed) | 10.68% | 77.52% (49 missed) | 25.85% |
| `gmic_rsna_finetuned` | 0.7426 | 0.7800 | 93.58% (14 missed) | 5.34% | 91.74% (18 missed) | 8.12% |
