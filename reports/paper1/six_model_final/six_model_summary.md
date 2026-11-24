# Paper 1 — Six-Model External Validation Summary

Primary paired 4-view NLBS cohort: `N = 6184` studies, `5868` clusters, `151` cancers.

## Primary Common 4-View NLBS Results (Stratum A vs. Stratum B)

| Stratum | Model | AUROC | @98% Sens | @98% Dismissal | @95% Sens | @95% Dismissal | Fallback Studies |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A (RSNA-Supervised) | `ce_b5` | 0.7089 | 90.07% (15 missed) | 30.48% | 80.13% (30 missed) | 49.66% | 0 |
| A (RSNA-Supervised) | `trust_b5` | 0.7040 | 84.77% (23 missed) | 39.57% | 80.79% (29 missed) | 49.29% | 0 |
| A (RSNA-Supervised) | `dinov2_rsna` | 0.6299 | 98.01% (3 missed) | 9.67% | 94.04% (9 missed) | 17.43% | 0 |
| A (RSNA-Supervised) | `resnet50_rsna` | 0.6452 | 97.35% (4 missed) | 5.35% | 84.77% (23 missed) | 19.78% | 0 |
| B (Pretrained-Direct) | `gmic_pretrained` | 0.8541 | 97.35% (4 missed) | 24.27% | 93.38% (10 missed) | 38.47% | 0 |
| B (Pretrained-Direct) | `nyu_fourview_pretrained` | 0.7925 | 100.00% (0 missed) | 1.94% | 99.34% (1 missed) | 15.60% | 0 |
