# Paper 1 — Independent RSNA Source-Test Results (CE-only & TRUST, CPU Verification)

> **Methodological Separation:** Metrics below come from **patient-disjoint 2-fold cross-fitting inside RSNA `val_calib`** (`fold_a`: 1,191 pts / 49 pos; `fold_b`: 1,191 pts / 48 pos). Thresholds are NEVER evaluated on the same patients used to select them. These cross-fitted thresholds are **distinct** from the full-`val_calib` (`97`-pos, AMP FP16) thresholds frozen in Stage B (`ce: 0.096191/0.124451`; `trust: 0.179565/0.195435`) for NLBS.

| Model | Evaluation Type | Calib Pos / Test Pos | Target | Threshold `tau` | Independent Test Sens (95% CI) | Missed | Independent Test Dismissal (95% CI) | AUROC (95% CI) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ce_b5` | 2-Fold OOF Pooled | 49/48 -> 97 | 98% | fold-specific | **98.97%** [96.46%, 100.00%] | 1/97 | **11.67%** [10.41%, 12.97%] | 0.7428 [0.6930, 0.7882] |
| `ce_b5` | 2-Fold OOF Pooled | 49/48 -> 97 | 95% | fold-specific | **89.69%** [83.33%, 95.19%] | 10/97 | **29.09%** [27.33%, 30.86%] | 0.7428 [0.6930, 0.7882] |
| `ce_b5` | `fold_a_to_fold_b` | 48 -> 49 | 98% | `0.075850` | 100.00% | 0/49 | 8.31% | — |
| `ce_b5` | `fold_a_to_fold_b` | 48 -> 49 | 95% | `0.174851` | 81.63% | 9/49 | 36.86% | — |
| `ce_b5` | `fold_b_to_fold_a` | 49 -> 48 | 98% | `0.096193` | 97.92% | 1/48 | 15.03% | — |
| `ce_b5` | `fold_b_to_fold_a` | 49 -> 48 | 95% | `0.119386` | 97.92% | 1/48 | 21.33% | — |
| `trust_b5` | 2-Fold OOF Pooled | 49/48 -> 97 | 98% | fold-specific | **98.97%** [96.46%, 100.00%] | 1/97 | **12.80%** [11.50%, 14.15%] | 0.7442 [0.6939, 0.7896] |
| `trust_b5` | 2-Fold OOF Pooled | 49/48 -> 97 | 95% | fold-specific | **89.69%** [83.33%, 95.19%] | 10/97 | **29.43%** [27.66%, 31.23%] | 0.7442 [0.6939, 0.7896] |
| `trust_b5` | `fold_a_to_fold_b` | 48 -> 49 | 98% | `0.136995` | 100.00% | 0/49 | 7.47% | — |
| `trust_b5` | `fold_a_to_fold_b` | 48 -> 49 | 95% | `0.257731` | 81.63% | 9/49 | 37.70% | — |
| `trust_b5` | `fold_b_to_fold_a` | 49 -> 48 | 98% | `0.179548` | 97.92% | 1/48 | 18.14% | — |
| `trust_b5` | `fold_b_to_fold_a` | 49 -> 48 | 95% | `0.194888` | 97.92% | 1/48 | 21.16% | — |
