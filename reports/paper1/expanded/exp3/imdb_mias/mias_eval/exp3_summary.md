# Paper 1 Exp 3 — Multi-Dataset External Validation Summary

> **Clinical Pooling Policy:** Per-dataset reporting ONLY. Screening (`nlbs`), mixed (`imdb`), and biopsy-enriched diagnostic (`cmmd`, `lumina`) cohorts are NEVER pooled.

## Dataset: `MIAS` (UK — `legacy_film_exploratory`)
- Screening Dismissal Claim Allowed: `False`

### Full Cohort (View-Agnostic) (`N=161`, clusters=`161`, pos=`52`)

| Model | AUROC | AUPRC | @98% Sens | @98% Below-Thr Rate | @95% Sens | @95% Below-Thr Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ce_b5` | 0.5258 | 0.3426 | 94.23% (3 missed) | 9.94% | 86.54% (7 missed) | 20.50% |
| `trust_b5` | 0.5319 | 0.3419 | 86.54% (7 missed) | 20.50% | 86.54% (7 missed) | 21.74% |
| `dinov2_rsna` | 0.5567 | 0.3981 | 7.69% (48 missed) | 92.55% | 5.77% (49 missed) | 96.89% |
| `resnet50_rsna` | 0.4486 | 0.3064 | 65.38% (18 missed) | 31.68% | 28.85% (37 missed) | 65.22% |
| `gmic_rsna_finetuned` | 0.5453 | 0.3965 | 15.38% (44 missed) | 90.68% | 13.46% (45 missed) | 93.17% |
