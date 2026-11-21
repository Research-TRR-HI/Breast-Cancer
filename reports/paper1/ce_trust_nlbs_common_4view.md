# Paper 1 — Locked Stage C CE/TRUST Results on Common 4-View NLBS Cohort (N = 6,184)

> Filtered directly from immutable `reports/stage_c/locked_nlbs_predictions.csv` (`sha256: 526434d058510208ee825b3db36380d0faec7760c17dd2c867dc6288eeb10cc2`). All `151/151` cancer-positive NLBS studies are 4-view eligible (`175` excluded studies are all cancer-negative).

| Model | Target | Frozen RSNA `tau` | Sensitivity (95% Cluster CI) | Missed Cancers | Dismissal Rate (95% Cluster CI) | AUROC (95% Cluster CI) |
|---|---:|---:|---:|---:|---:|---:|
| `ce_b5` | 98% | `0.096191` | **90.07%** [85.00%, 94.33%] | 15/151 | **30.37%** [29.15%, 31.54%] | 0.7089 [0.6668, 0.7463] |
| `ce_b5` | 95% | `0.124451` | **80.13%** [73.44%, 86.30%] | 30/151 | **49.66%** [48.37%, 50.90%] | 0.7089 [0.6668, 0.7463] |
| `trust_b5` | 98% | `0.179565` | **84.77%** [78.72%, 90.23%] | 23/151 | **39.57%** [38.31%, 40.83%] | 0.7040 [0.6617, 0.7423] |
| `trust_b5` | 95% | `0.195435` | **80.79%** [74.03%, 86.63%] | 29/151 | **49.16%** [47.85%, 50.43%] | 0.7040 [0.6617, 0.7423] |
