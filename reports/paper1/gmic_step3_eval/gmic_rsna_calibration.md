# Paper 1 — GMIC Step 3 Phase 2: RSNA Source Threshold Calibration (CPU ONLY)

**Model:** `gmic_pretrained` (`sample_model_1.p`, SHA-256 `3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991`)  
**Quarantine:** RSNA labels ONLY (`val_calib`: `2382` patients / `97` cancers). Zero NLBS outcomes read.

## 1. Frozen NLBS Operating Thresholds (`full val_calib`, NOT Independent Evaluation)
* **Primary `target_98` (`tau_98`):** `0.02887365` -> resubstitution RSNA sens `98.97%` (`1/97` missed), dismissal `17.34%`
* **Secondary `target_95` (`tau_95`):** `0.04664113` -> resubstitution RSNA sens `95.88%` (`4/97` missed), dismissal `30.77%`

## 2. Independent 2-Fold Patient-Disjoint RSNA Source Test
* **RSNA `val_calib` Discrimination:** AUROC = **`0.8301`** (95% CI `[0.7885, 0.8710]`), AUPRC = `0.2509` (`[0.1719, 0.3346]`)
* **2-Fold OOF Pooled `@98%`:** Sensitivity = **`98.97%`** (`1/97` missed, CI `[96.52%, 100.00%]`), Dismissal = **`13.39%`** (`[12.01%, 14.78%]`)
* **2-Fold OOF Pooled `@95%`:** Sensitivity = **`91.75%`** (`8/97` missed, CI `[86.05%, 96.84%]`), Dismissal = **`38.54%`** (`[36.65%, 40.56%]`)

## 3. RSNA `80`-Fallback Calibration Sensitivity Check
* **Fallback Cases:** `80` views across `36` patients (`0` cancer-positive).
* **Clean Views Only (`N=2378`, `97` pos):** AUROC = `0.8306`, `tau_98 = 0.02887365` (delta `+0.00000000`), `tau_95 = 0.04664113` (delta `+0.00000000`).
* **Clean Patients Only (`N=2346`, `97` pos):** AUROC = `0.8377`, `tau_98 = 0.02887365` (delta `+0.00000000`), `tau_95 = 0.04664113` (delta `+0.00000000`).
