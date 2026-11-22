# Paper 1 — GMIC Step 3 External Validation Report (RSNA -> Canadian NLBS)

**Execution Mode:** CPU Only (`0` GPU / `USD 0.00` cloud spend) | **GMIC Checkpoint:** `sample_model_1.p` (`sha256: 3b792e6de22fdccd...`)  
**Strict Quarantine:** RSNA thresholds frozen & SHA-256 locked (`5bae9b5467cae9c7...`) BEFORE joining NLBS outcomes.

## 1. RSNA Calibration vs. Independent 2-Fold OOF Source Performance (GMIC)
* **Frozen NLBS Operating Points (`val_calib`, 97 pos — resubstitution, NOT independent):** `tau_98 = 0.028874` (`98.97%` sens / `17.34%` dis), `tau_95 = 0.046641` (`95.88%` sens / `30.77%` dis).
* **Independent 2-Fold Patient-Disjoint RSNA OOF (`2,382` pts / `97` pos):** AUROC **`0.8301`** (`[0.7885, 0.8710]`), `@98%` OOF sens **`98.97%`** (`1/97` missed, dismissal `13.39%`), `@95%` OOF sens **`91.75%`** (`8/97` missed, dismissal `38.54%`).
* **RSNA `80`-Fallback Calibration Audit (`36` pts, `0` pos):** Excluding fallback views changes `tau_98` by `+0.000000` and excluding fallback cases changes `tau_98` by `+0.000000` (sensitivity checks below).

## 2. Primary Common 4-View NLBS Comparison: OUR CE vs. TRUST vs. GMIC (`N = 6184` studies, `5868` clusters, `151` cancers)

> **Disclosure:** `ce_b5` and `trust_b5` below are OUR Stage C zero-NLBS-label runs (`sha256: 526434d0...`), NOT the published TRUST (2026) 20%-target-calibrated external result (`12.95%` @98%, `19.87%` @95%; CE failed 98%). `gmic_pretrained` is official Stratum B pretrained weights (`nyukat/GMIC` `sample_model_1.p`, trained on NYU 2010–2017; `0` NLBS labels touched).

### At Frozen Source 98% Sensitivity Threshold (`target_98` — Primary)
| Model (Stratum) | NLBS AUROC [95% CI] | Frozen `tau_98` | Sensitivity [95% Cluster CI] | Missed | Dismissal (Neg Dis) | CP99 UB | Met 98%? |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | :---: |
| **CE-only** (Stratum A · OUR) | 0.7089 [0.6668, 0.7463] | 0.096191 | **90.07%** [85.00%, 94.33%] | 15/151 | 30.37% (30.88% neg) | 1.42% | **NO** |
| **TRUST** (Stratum A · OUR) | 0.7040 [0.6617, 0.7423] | 0.179565 | **84.77%** [78.72%, 90.23%] | 23/151 | 39.57% (40.18% neg) | 1.50% | **NO** |
| **GMIC** (Stratum B · Pretrained) | 0.8541 [0.8187, 0.8874] | 0.028874 | **97.35%** [94.44%, 99.39%] | 4/151 | 24.27% (24.81% neg) | 0.77% | **NO** |

### At Frozen Source 95% Sensitivity Threshold (`target_95` — Secondary)
| Model (Stratum) | NLBS AUROC [95% CI] | Frozen `tau_95` | Sensitivity [95% Cluster CI] | Missed | Dismissal (Neg Dis) | CP99 UB | Met 95%? |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | :---: |
| **CE-only** (Stratum A · OUR) | 0.7089 [0.6668, 0.7463] | 0.124451 | **80.13%** [73.44%, 86.30%] | 30/151 | 49.66% (50.41% neg) | 1.47% | **NO** |
| **TRUST** (Stratum A · OUR) | 0.7040 [0.6617, 0.7423] | 0.195435 | **80.79%** [74.03%, 86.63%] | 29/151 | 49.16% (49.91% neg) | 1.45% | **NO** |
| **GMIC** (Stratum B · Pretrained) | 0.8541 [0.8187, 0.8874] | 0.046641 | **93.38%** [89.17%, 97.16%] | 10/151 | 38.47% (39.27% neg) | 0.84% | **NO** |

## 3. Secondary Full NLBS Cohort (`N = 6359` studies, `5942` clusters, `151` cancers)
| Model | Target | Frozen `tau` | Sensitivity [95% CI] | Missed | Dismissal (Neg Dis) | AUROC [95% CI] | Met Target? |
| :--- | :---: | ---: | :--- | ---: | ---: | :--- | :---: |
| `ce_b5` | 98% | 0.096191 | 90.07% [84.96%, 94.38%] | 15/151 | 30.49% (30.99%) | 0.7101 [0.6694, 0.7469] | **NO** |
| `ce_b5` | 95% | 0.124451 | 80.13% [73.75%, 86.23%] | 30/151 | 49.90% (50.63%) | 0.7101 [0.6694, 0.7469] | **NO** |
| `trust_b5` | 98% | 0.179565 | 84.77% [78.72%, 90.20%] | 23/151 | 39.77% (40.37%) | 0.7052 [0.6644, 0.7429] | **NO** |
| `trust_b5` | 95% | 0.195435 | 80.79% [74.12%, 86.91%] | 29/151 | 49.43% (50.16%) | 0.7052 [0.6644, 0.7429] | **NO** |
| `gmic_pretrained` | 98% | 0.028874 | 97.35% [94.67%, 99.39%] | 4/151 | 24.69% (25.23%) | 0.8554 [0.8205, 0.8883] | **NO** |
| `gmic_pretrained` | 95% | 0.046641 | 93.38% [89.35%, 97.10%] | 10/151 | 38.95% (39.74%) | 0.8554 [0.8205, 0.8883] | **NO** |

## 4. Paired Differences (`B = 2,000` cluster bootstrap) & Missed-Cancer Overlap (`N = 6,184`)
* **`ce_b5_minus_gmic_pretrained`:** ΔAUROC = `-0.1452` (`[-0.1866, -0.1014]`), ΔSens@98 = `-7.25 pp` (`[-12.27, -2.61]`), ΔDis@98 = `+6.09 pp` (`[+4.76, +7.41]`), McNemar cancers@98 `p = 0.0074` (`only_a=2, only_b=13`)
* **`trust_b5_minus_gmic_pretrained`:** ΔAUROC = `-0.1500` (`[-0.1916, -0.1065]`), ΔSens@98 = `-12.53 pp` (`[-18.44, -7.09]`), ΔDis@98 = `+15.30 pp` (`[+13.90, +16.65]`), McNemar cancers@98 `p = 0.0000` (`only_a=1, only_b=20`)
* **`ce_b5_minus_trust_b5`:** ΔAUROC = `+0.0048` (`[+0.0005, +0.0090]`), ΔSens@98 = `+5.28 pp` (`[+1.90, +9.15]`), ΔDis@98 = `-9.21 pp` (`[-9.91, -8.51]`), McNemar cancers@98 `p = 0.0078` (`only_a=8, only_b=0`)
* **GMIC RSNA -> NLBS Shift Decomposition (`N=6,184`):** ΔAUROC = `+0.0240`, ΔSens@98 vs RSNA OOF = `-1.62 pp`, neg median shift = `-0.017515` (`KS p=1.097e-12`), pos median shift = `+0.143957` (`KS p=1.693e-02`).
* **Missed Cancers at `@98%` (`151` pos):** CE=`15`, TRUST=`23`, GMIC=`4` | UpSet patterns: `{"NONE_MISSED": 127, "ce_b5|trust_b5": 13, "trust_b5": 7, "ce_b5|trust_b5|gmic_pretrained": 2, "gmic_pretrained": 1, "trust_b5|gmic_pretrained": 1}`.
* **Jaccard Overlap (`@98%` missed cancers):** CE–TRUST `0.652`, CE–GMIC `0.118`, TRUST–GMIC `0.125`.
