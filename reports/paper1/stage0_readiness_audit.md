# Paper 1 — Stage 0: Evaluation, Cohort, Leakage, and Compute Readiness Audit

**Date:** 2026-11-21  
**Paper Track:** Paper 1 — *Multi-Model External Validation of Breast Cancer Screening AI* (`Paper1_External_Validation_Research_Context.md`)  
**Stage:** Stage 0 — Evaluation & Dataset Audit (CPU-Only; Zero GPU Runs Executed)  
**Verified Baseline Repo Head:** `ba3205b` (`main`)  
**Decision:** **CONDITIONAL GO** — Evaluation design, NLBS denominator reconciliation, zero-retrain CE/TRUST reuse, and a budget-viable execution plan are verified; **paid Modal GPU runs remain gated** behind 3 mandatory pre-flight checks (Modal account credit balance verification, CPU protocol/split freeze in `studies/paper1/`, and per-model parity + 200-image throughput pilots).

---

## Executive Summary

| Audit Area | Key Verified Finding | Required Correction / Guardrail | Status |
|---|---|---|---|
| **1. NLBS Cohort Reconciliation** | `5,997` `case_id` folders expand to `6,430` `(case_id, StudyInstanceUID)` rows (`385` multi-visit folders), deduplicate to **`6,359` unique `StudyInstanceUID`s** (`71` cross-folder shared UIDs), and collapse into **`5,942` `split_cluster_id` components** (`29` multi-case clusters). Positives: **`149` clusters / `151` studies** (`p_4334` & `p_4461` have 2 positive visits each). **`6,184 / 6,359` (`97.25%`)** studies—including **`151 / 151` (`100%`)** cancers—have all 4 standard views. | Primary scoring unit = deduplicated `StudyInstanceUID` (`6,184` common 4-view cohort; `6,359` full cohort for single-view models). Primary statistical CI/splitting unit = `split_cluster_id` (`5,942` / `5,868`), plus mandatory cluster-collapsed (`149`-pos) sensitivity analysis. | **RESOLVED** |
| **2. RSNA Split & Source Test Fix** | `splits_seed42.csv` (`11,913` pts, `486` pos) has `0` patient overlap, `100%` 4-view completeness, and balanced sites (`Site 1: 5,818`, `Site 2: 6,095`), **but** Stage B double-used `val_calib` (`2,382` pts, `97` pos) for both `tau` selection and source reporting. | Partition `val_calib` into patient-disjoint `source_op_calib` (`1,191` pts / `49` pos) and `source_heldout_test` (`1,191` pts / `48` pos) + 2-fold out-of-fold RSNA cross-fitting (`97` pos total) for unbiased source reporting; train `dinov2_rsna` and `resnet50_rsna` strictly on `train` (`8,339` pts) + `train_calib` (`1,192` pts, hyperparameter/early-stopping only). | **RESOLVED (DESIGN)** |
| **3. CE & TRUST Reuse** | Local `models/cache_eval_rsna/eval_val_calib_fp32.pt` (`10,982 × 2048`), `eval_train_calib_fp32.pt`, Stage-B Epoch-20 checkpoints, and `reports/stage_c/locked_nlbs_predictions.csv` (`6,359` studies) are intact. CPU FP32 vs GPU AMP causes `<= 3.55e-5` score rounding (`1` borderline case shift on `ce@95` RSNA). | **100% reusable without GPU retraining ($0 cost).** Preserve `reports/stage_b/*` and `reports/stage_c/*` immutable; compute Paper 1 RSNA sub-split metrics directly on cached embeddings on CPU (documenting FP32 vs AMP parity) and filter `locked_nlbs_predictions.csv` for the `6,184` common 4-view cohort. | **VERIFIED REUSABLE** |
| **4. Leakage & Provenance** | Zero RSNA split leakage; NLBS cluster leakage controlled via `split_cluster_id` (residual unlinked repeats from 2025-11-25 anonymization disclosed). Mammo-CLIP (`UPMC`), GMIC (`NYU 2010–2017`), NYU Four-View (`NYU 2010–2017`), DINOv2 (`LVD-142M`), and ResNet-50 (`ImageNet-1K`) have **zero known RSNA/NLBS training overlap**. | Disclose Paper 1 as a **post hoc multi-model expansion** (CE/TRUST NLBS results already seen); enforce label-free NLBS manifest (`inference_manifest_stage_c.csv`), SHA-256 pre-join prediction locks, and `AGPL-3.0` (`GMIC`, `NYU`) / `CC BY-NC-SA 4.0` (`Mammo-CLIP`) license boundaries. | **AUDITED** |
| **5. Four-Model Plan & Caches** | Existing `1520×912` Stage-1 `uint16` PNG caches (`54,706` RSNA + `26,988` NLBS) can be **reused directly** for `dinov2_rsna` and `resnet50_rsna`. **Pretrained-direct GMIC and NYU need inference ONLY on RSNA `val_calib` + NLBS**, cutting inference volume by **53%** (`37,970` GMIC imgs; `8,566` NYU exams). | Never feed `1520×912` PNGs to GMIC (`2944×1920`) or NYU (`2677×1942` CC / `2974×1748` MLO). Use CPU-only containers for native DICOM cropping and L40S NVMe-staged inference. Execution order: `DINOv2` -> `GMIC` -> `NYU Four-View` -> `ResNet-50` (cost-gated). | **READY TO IMPLEMENT** |

---

## 1. NLBS Cohort & Denominator Reconciliation (Task 1)

### 1.1 Exact Arithmetic Reconciling Published Counts vs. Repository Manifests

Audited directly across `datasets/nlbs/manifest.csv` (`26,988` rows), `datasets/nlbs/exam_manifest.csv` (`6,430` rows), `datasets/nlbs/inference_manifest_stage_c.csv` (`26,988` rows), and `reports/stage_a3_nlbs_results.json`:

| Level / Entity | Count | Exact Reconciliation Formula / Explanation |
|---|---:|---|
| **Raw DICOM images** | **26,988** | `19,942` normal + `6,394` False Positive + `652` abnormal; `100%` `MONOCHROME2`, `12-bit`, `1.2.840.10008.1.2.1` (`26,967` GE Senographe Essential, `21` Siemens Mammomat Novation DR); `8/8` FRDR SHA-256 verified. |
| **Published descriptor `case_id` folders** | **5,997** | `4,332` normal + `1,516` False Positive + `149` abnormal folders (`case_id` `p_0001`..`p_5997`). |
| **Published descriptor "unique cases"** | **5,935** | Kendall et al. (*Scientific Data* 2025) report `5,997` exams and `5,935` unique cases (`62` repeat exams). Per `README_version_history_2025-11-25.txt`, directories were renamed to sever unverified repeat links (`PatientID` stripped in `26,984/26,988` DICOMs; `Exam Date` `100%` null). |
| **Folder-study pairs `(case_id, StudyInstanceUID)`** | **6,430** | `385` of the `5,997` `case_id` folders contain `>1` distinct `StudyInstanceUID`: `5,612` have 1 study, `378` have 2 (`+378`), `6` have 3 (`+12`), and `1` (`p_3280`) has `44` (`+43`). Exact sum: `5,997 + 433 = 6,430` (`4,763` normal, `1,516` False Positive, `151` abnormal). |
| **Deduplicated `StudyInstanceUID` exams** | **6,359** | **`71`** `StudyInstanceUID`s appear in exactly `2` `case_id` folders (`43` in `normal`, `28` in `False Positive`, `0` in `abnormal`, `0` crossing cohorts), yielding `142` duplicate rows -> `71` unique studies. Exact sum: `6,430 - 71 = 6,359` (`4,720` normal, `1,488` False Positive, `151` abnormal). |
| **Connected `split_cluster_id` components** | **5,942** | Union-Find over `(case_id, StudyInstanceUID)` merges `84` `case_id`s into `29` multi-case clusters (`28` False Positive pairs + `1` normal cluster `nlbs_cluster_4240` linking `p_3280` and `27` other normal folders across `45` UIDs). Remaining `5,913` clusters have 1 `case_id`. Exact sum: `5,997 - (84 - 29) = 5,942` (`5,571` 1-study, `367` 2-study, `3` 3-study, `1` 45-study clusters). |
| **Cancer-positive cases / clusters vs. exams** | **149 / 151** | All `149` `abnormal` `case_id`s are distinct clusters (`0` cross-folder shared UIDs). Exactly `2` folders—**`p_4334`** (`nlbs_cluster_4306`, `p_4334_v1`/`v2`, Age 70) and **`p_4461`** (`nlbs_cluster_4433`, `p_4461_v1`/`v2`, Age 74)—each contain `2` distinct cancer-positive `StudyInstanceUID`s (`4` images & `2` cancer-positive images per visit), yielding `149 + 2 = 151` positive exams (`2.37%` exam prevalence). |

### 1.2 Four-View Eligibility Audit (Required for NYU Four-View & Common Cohort)

Auditing `{L-CC, L-MLO, R-CC, R-MLO}` completeness across deduplicated `StudyInstanceUID`s:

- **4-View Eligible Studies:** **`6,184 / 6,359` (`97.25%`)**, including **`151 / 151` (`100.00%`)** of all cancer-positive exams!
- **Ineligible Studies (`175` total, `0` cancers):** `163` normal + `12` False Positive (`n_images` distribution: `2` imgs: 2, `3` imgs: 3, `4` imgs with duplicate/missing view slot: 145, `5` imgs: 17, `6` imgs: 8).
- **Multi-Image Slots Among Eligible 4-View Studies:** **`935 / 6,184` (`15.12%`)** have `>4` images (repeat/extra views within a standard view slot), requiring a deterministic label-blind view-selection rule for NYU Four-View CNN.
- **Cluster-Level 4-View Eligibility:** **`5,868 / 5,942` (`98.75%`)** clusters have `>=1` 4-view study, including **`149 / 149` (`100.00%`)** positive clusters (`74` ineligible clusters: `62` normal, `12` False Positive).

### 1.3 Confirmed Evaluation Units & Reporting Mandate

1. **Primary Inference & Scoring Unit:** Deduplicated **`StudyInstanceUID` examination**.
   - **Common 6-Model Primary Cohort:** **`N = 6,184`** 4-view eligible examinations (`151` cancer-positive, `6,033` negative), enabling exact paired comparisons across all 6 models (including `nyu_fourview_pretrained`).
   - **Full-NLBS 5-Model Cohort (Sensitivity Analysis):** **`N = 6,359`** examinations (`151` positive, `6,208` negative) for the 5 models that score arbitrary view counts (`ce_b5`, `trust_b5`, `gmic_pretrained`, `dinov2_rsna`, `resnet50_rsna`), proving whether excluding `175` non-4-view negative exams shifts dismissal rates.
2. **Primary Statistical Resampling & Splitting Unit:** **`split_cluster_id`** (`5,868` clusters in 4-view cohort; `5,942` in full cohort; `149` positive clusters). All 95% CIs (`B >= 2,000` cluster bootstrap replicates) and paired model-difference tests MUST resample by `split_cluster_id`.
3. **Mandatory Cluster-Collapsed Sensitivity Analysis:** Collapse scores to `split_cluster_id` (`max` across visits, or deterministically first study by `StudyInstanceUID` — predeclared in `studies/paper1/protocol_v1.md`) giving **`5,868` / `5,942` independent cluster units (`149` positives)**.
4. **Explicit Linkage Limitation:** `split_cluster_id` is a **patient-like linkage proxy**, not a verified ground-truth patient ID, because NLBS stripped `PatientID`/`Exam Date` and renamed unverified repeat folders on 2025-11-25.

---

## 2. RSNA Split Audit & Scientifically Valid Calibration/Test Redesign (Task 2)

### 2.1 Audit of `datasets/rsna/splits_seed42.csv`

| Split | Images | Patients | Cancer+ Pts (Prev %) | Site 1 Pts (Pos) | Site 2 Pts (Pos) | 4-View Eligible Pts | Stage B Role |
|---|---:|---:|---:|---:|---:|---:|---|
| `train` | 38,251 | 8,339 (70.0%) | 340 (4.08%) | 4,040 (174) | 4,299 (166) | 8,339 / 8,339 (100%) | Frozen CE/TRUST head fitting |
| `train_calib` | 5,473 | 1,192 (10.0%) | 49 (4.11%) | 595 (30) | 597 (19) | 1,192 / 1,192 (100%) | TRUST dynamic `tau_t` per epoch |
| `val_calib` | 10,982 | 2,382 (20.0%) | 97 (4.07%) | 1,183 (48) | 1,199 (49) | 2,382 / 2,382 (100%) | **FLAW:** Both threshold selection & RSNA reporting |
| **Total** | **54,706** | **11,913** | **486 (4.08%)** | **5,818 (252)** | **6,095 (234)** | **11,913 / 11,913 (100%)** | `3,680` pts have `>4` views |

**Identified Flaw:** In Stage B, `val_calib` (`2,382` patients, `97` positives) was used **both** to compute `tau_95`/`tau_98` AND to report RSNA sensitivity/dismissal (`98.97%` / `14.48%` CE; `98.97%` / `17.17%` TRUST). Those numbers are **calibration-set resubstitution estimates**, not independent held-out source test performance.

### 2.2 Evaluation of Redesign Options & Statistical Precision

| Option | Patient Partition (`train` / `train_calib` / `source_op_calib` / `source_heldout_test`) | Positives per Source Eval Split | CE/TRUST Retrain Cost | Statistical & Budget Verdict |
|---|---|---|---:|---|
| **Option B (Fresh 4-Way Split)** | `50 / 10 / 20 / 20` (`5,956 / 1,191 / 2,383 / 2,383`) or `60 / 10 / 15 / 15` | `97` / `97` (or `73` / `73`) | **~$12–$19** (must rebuild `1M`-occ Mammo-CLIP aug cache) | **REJECTED for $30 cap:** Consumes 40–63% of total budget just to re-run CE/TRUST, starving GMIC/NYU/ResNet-50. |
| **Option A (Zero-Retrain `val_calib` Partition — RECOMMENDED)** | Keep `train` (`8,339`) & `train_calib` (`1,192`) fixed. Partition `val_calib` (`2,382` pts, `97` pos) 50/50 stratified by `(cancer, site_id)` into **`source_op_calib` (`1,191` pts, `49` pos)** and **`source_heldout_test` (`1,191` pts, `48` pos)**, plus **2-fold OOF cross-fit** across both halves (`97` pos total). | **`49` / `48`** (single split); **`97`** (2-fold OOF) | **$0.00** (reuses `eval_val_calib_fp32.pt` locally on CPU) | **ADOPTED:** Zero leakage from training into `val_calib` for all 6 models; zero extra GPU spend for CE/TRUST. |

### 2.3 Exact Precision Caveat & Threshold Protocol Under Option A

1. **Statistical Precision of `49` vs `97` Positives:**
   - At `n_pos = 49` (50% of `val_calib`), `1` missed cancer = `2.04 pp`, so `ceil(0.98 * 49) = 49` (`0` misses allowed -> `tau_98` equals the single lowest positive score in `source_op_calib`, yielding high variance; 95% Clopper-Pearson CI `[92.75%, 100.00%]`).
   - At `n_pos = 97` (full `val_calib`), `1` missed cancer = `1.03 pp`, so `ceil(0.98 * 97) = 96` (`1` miss allowed -> `96/97 = 98.97%`, 95% CP CI `[94.39%, 99.97%]`).
2. **Protocol Separation (No Threshold Cherry-Picking):**
   - **For RSNA Independent Source Evaluation (Fixing Blocker 2):** Report **2-fold patient-disjoint cross-fitted source performance** across `val_calib` (`Fold A [49 pos] -> Fold B [48 pos]` and `Fold B -> Fold A`, pooled over all `2,382` patients / `97` positives), plus the single fixed `Fold A -> Fold B` split. Neither half ever sees its own labels when choosing `tau`.
   - **For NLBS External Evaluation (Continuity with Immutable Stage C):** Predeclare **full `val_calib` (`2,382` pts, `97` pos)** as the NLBS operating-threshold calibration set for all 6 models (preserving Stage C's exact frozen thresholds `ce: 0.096191/0.124451`, `trust: 0.179565/0.195435`), **and** report `Fold A (49 pos)` sensitivity check on NLBS strictly as a documented secondary sensitivity table to quantify threshold sample-size instability.
   - **Training Rule for New RSNA Models (`dinov2_rsna`, `resnet50_rsna`):** Train strictly on `train` (`8,339` pts / `340` pos), using `train_calib` (`1,192` pts / `49` pos) ONLY for hyperparameter tuning, LR scheduling, or early stopping. **`val_calib` (`2,382` pts) MUST remain 100% unseen during training and model selection.**
   - **Mandatory Transparency:** Because CE/TRUST `val_calib` scores were previously inspected in Stage B, explicitly classify Paper 1's source calibration/test separation as a **retrospective patient-disjoint evaluation redesign**, not a prospectively untouched split.

---

## 3. CE & TRUST Artifact Reuse Verification (Task 3)

### 3.1 Inventory & Verification Results

All historical Stage B & Stage C artifacts were verified directly on disk:

- **RSNA Checkpoints (`models/checkpoints/stage_b_rsna/`):**
  - `ce_rsna_seed42_final_epoch20.pt` (SHA-256 `adf1443455ee1b507485ee4b7f64932d5cf6f59bb6afcf11b9f415c634372d22`)
  - `trust_rsna_seed42_final_epoch20.pt` (SHA-256 `63388d790ce9cf8a8287adc7724358ced392a69b332ce1a5e46972750faa51cb`)
- **RSNA Cached Embeddings (`models/cache_eval_rsna/`):**
  - `eval_train_calib_fp32.pt`: `torch.float32` tensor `[5473, 2048]` (`SHA-256: b7722965...`)
  - `eval_val_calib_fp32.pt`: `torch.float32` tensor `[10982, 2048]` (`SHA-256: bb983e4d...`)
- **NLBS Locked Predictions (`reports/stage_c/`):**
  - `locked_nlbs_predictions.csv`: `6,359` unique `StudyInstanceUID`s with `score_ce`, `score_trust`, and frozen dismissal flags (`SHA-256: 526434d0...`, verified pre-label-join).
  - `locked_nlbs_image_diagnostics.csv.gz`: `26,988` per-image records (`SHA-256: 1817a8ab...`).

### 3.2 CPU Reproduction Check & FP16 AMP Precision Finding

Running the Stage-B `head_state_dict` on `eval_val_calib_fp32.pt` locally on CPU (`float32`) vs. Stage B's L40S `torch.amp.autocast("cuda")` (`float16` linear/layernorm cast inside `scripts/train_trust_modal.py` L1137) revealed a tiny precision difference:

| Model / Target | Stage B L40S AMP Threshold | Local CPU FP32 Threshold | Abs Diff | Stage B Dismissal (`n=2,382`) | CPU FP32 Dismissal |
|---|---:|---:|---:|---:|---:|
| `ce @95%` | `0.12445068359375` | `0.124415136873722` | `3.55e-5` | `536` (`22.50%`) | `537` (`22.54%`) |
| `ce @98%` | `0.09619140625` | `0.096193462610245` | `2.06e-6` | `345` (`14.48%`) | `346` (`14.53%`) |
| `trust @95%` | `0.1954345703125` | `0.195463106036186` | `2.85e-5` | `493` (`20.70%`) | `493` (`20.70%`) |
| `trust @98%` | `0.1795654296875` | `0.179548069834709` | `1.74e-5` | `409` (`17.17%`) | `409` (`17.17%`) |

**Verdict:** **YES — reuse CE & TRUST 100% without retraining ($0 GPU cost).**
- Keep `reports/stage_b/*` and `reports/stage_c/*` **strictly untouched**.
- On NLBS, filter `reports/stage_c/locked_nlbs_predictions.csv` directly to the `6,184` 4-view cohort (exact L40S AMP scores preserved).
- On RSNA `val_calib`, evaluate the new 2-fold OOF / 50-50 split directly on `eval_val_calib_fp32.pt` on CPU (and optionally emulate FP16 head weights/activations or report exact FP32 CPU scores alongside historic AMP calibration numbers with the `<= 3.55e-5` precision note).

---

## 4. Data Leakage, Pretrained Provenance & License Audit (Task 4)

### 4.1 Dataset-Level Overlap Audit

- **RSNA (`11,913` patients):** `0` patient overlap across `train`, `train_calib`, and `val_calib`. `7,018` extra views share `(patient_id, laterality, view)` within the same patient split.
- **NLBS (`6,359` studies / `5,942` clusters):** Grouping by `split_cluster_id` eliminates all `71` cross-folder `StudyInstanceUID` aliases and `385` multi-visit folders (`p_3280` has `44` UIDs). **Residual unknown overlap:** unlinked repeats renamed in the 2025-11-25 NLBS rebuild (`PatientID` stripped, `Exam Date` null) cannot be detected from metadata and must be stated as a limitation.
- **Post-Hoc Selection Transparency:** Because Stage C CE/TRUST NLBS results (`90.07%` / `84.77%`) were inspected prior to Paper 1's 4-model expansion, Paper 1 **cannot** be framed as a prospectively blinded study; all 4 new models must use **strictly label-free NLBS inference** (`datasets/nlbs/inference_manifest_stage_c.csv`), SHA-256 prediction locking before outcome join, and zero NLBS-guided architecture/hyperparameter/preprocessing iteration.

### 4.2 Pretrained-Model Provenance, Overlap & License Matrix

| Model ID | Pretraining Source & Scale | Training Dates / Geography | RSNA Overlap | NLBS Overlap | Code / Weight License & Distribution Risk |
|---|---|---|---|---|---|
| **`ce_b5` / `trust_b5`** (`Mammo-CLIP B5`) | UPMC internal screening mammogram–report pairs (`13,829` patients / `25,355` images, BI-RADS 0–2; `upmc_clip` fold 0 `b5-model-best-epoch-7.tar`) | Pittsburgh, PA, USA (pre-MICCAI 2024; RSNA & VinDr used only for downstream eval in Ghosh et al.) | **None known** (Emory/Victoria vs UPMC) | **None known** (Newfoundland 2008–2011 vs UPMC) | **`CC BY-NC-SA 4.0`** (non-commercial research only; share-alike on derivatives) |
| **`gmic_pretrained`** (`nyukat/GMIC`) | NYU Breast Cancer Screening Dataset (`229,426` exams, `>1M` images, `141,472` patients) | NYU Langone Health, NY, USA (`2010–2017`) | **Presumed none** (different institutions; unverified patient cross-travel) | **Presumed none** (pre-dates 2025 NLBS release; different country) | **`AGPL-3.0`** (strong copyleft; verify publication/service-hosting & checkpoint terms) |
| **`nyu_fourview_pretrained`** (`nyukat/breast_cancer_classifier`) | NYU Breast Cancer Screening Dataset (`229,426` exams, `141,472` patients) | NYU Langone Health, NY, USA (`2010–2017`) | **Presumed none** | **Presumed none** | **`AGPL-3.0`** (same as GMIC; do not redistribute weights in repo) |
| **`dinov2_rsna`** (`dinov2_vits14`) | LVD-142M curated natural images (self-supervised) | Meta AI (`2023`) | **None** (natural images) | **None** (natural images) | **`Apache-2.0`** (permissive code & weights) |
| **`resnet50_rsna`** (`torchvision` `IMAGENET1K_V2`) | ImageNet-1K natural images (supervised) | Torchvision / PyTorch | **None** (natural images) | **None** (natural images) | **`BSD-3-Clause`** (permissive) |

*Note:* Because NYU and UPMC datasets are private clinical archives without shared patient hashes, cross-institutional patient migration is strictly marked **"Presumed none based on geography/dates; mathematically unverified"** rather than proven zero.

---

## 5. Exact Implementation Plan for the Four New Models (Task 5)

All 4 new models must preserve identical evaluation units (`StudyInstanceUID`, `split_cluster_id`), identical `max` exam-score aggregation, identical RSNA `val_calib` operating-point selection (`< tau` dismissed), and pre-join SHA-256 prediction locks.

### 5.1 `gmic_pretrained` (Stratum B — Pretrained-Direct)
- **Repository & Checkpoint Policy:** `nyukat/GMIC` (`AGPL-3.0`). Predeclare **1 primary single GMIC-ResNet-18 checkpoint** (`model_index=0` or authors' designated single checkpoint) prior to any NLBS scoring. Top-5 ensemble (`5x` compute) is strictly optional/secondary if surplus budget remains.
- **Native Preprocessing (Do NOT reuse `1520×912` PNGs):**
  - Decode DICOM with Modality/VOI LUTs and `MONOCHROME1 -> MONOCHROME2` inversion.
  - Apply `nyukat/GMIC`'s native cropping/orientation pipeline (`crop_mammogram.py` + `get_optimal_centers.py` logic) -> **16-bit PNGs at `2944 × 1920` (`H × W`)**, per-image mean/std normalization matching the authors' dataloader.
- **Scoring & Aggregation:** Extract `malignant_pred` (never `benign_pred`) per image; `exam_score = max(malignant_pred across valid images in StudyInstanceUID)`.
- **Scope Optimization:** Infer **ONLY** on RSNA `val_calib` (`10,982` images) + NLBS (`26,988` images) = **`37,970` images** (zero RSNA `train`/`train_calib` inference needed!).
- **Mandatory Gates:** (1) Exact output parity against `nyukat/GMIC/sample_data/` reference predictions; (2) 200-image L40S throughput/cost projection.

### 5.2 `nyu_fourview_pretrained` (Stratum B — Pretrained-Direct)
- **Repository & Checkpoint Policy:** `nyukat/breast_cancer_classifier` (`AGPL-3.0`), specifically the **image-only four-view model** (`IMAGE_MODEL_PATH`; exclude `image-and-heatmaps` so no auxiliary patch-heatmap network is introduced).
- **Native Preprocessing (Do NOT reuse `1520×912` PNGs):**
  - Decode DICOM (LUTs + `MONOCHROME1` inversion) and run NYU's native cropping/orientation -> **16-bit PNGs at `2677 × 1942` (CC views) and `2974 × 1748` (MLO views)** with authors' exact per-view standardization.
- **Deterministic Label-Blind View-Selection Policy (for `935` NLBS & `3,680` RSNA multi-view cases):**
  - Eligible cohort: studies with all 4 views `{L-CC, L-MLO, R-CC, R-MLO}` (`2,382 / 2,382` RSNA `val_calib`; `6,184 / 6,359` NLBS).
  - If `>1` image exists in a `(StudyInstanceUID, Laterality, ViewPosition)` slot: (a) on NLBS, prefer `GE MEDICAL SYSTEMS` if a slot mixes manufacturers (rare), then sort deterministically by `rel_path` (which encodes instance order without any label info—note `inference_manifest_stage_c.csv` strips `cohort_group`); (b) on RSNA, select minimum integer `image_id`. Unit-test zero correlation with cancer labels.
- **Scoring & Aggregation:** Model outputs `(left_malignant, right_malignant)` -> `exam_score = max(left_malignant, right_malignant)`.
- **Scope Optimization:** Infer **ONLY** on RSNA `val_calib` (`2,382` exams = `9,528` views) + NLBS 4-view (`6,184` exams = `24,736` views) = **`8,566` exams (`34,264` images)** (saving 53% vs full RSNA+NLBS).
- **Mandatory Gates:** (1) Exact parity on `nyukat/breast_cancer_classifier/sample_data/`; (2) 200-exam L40S throughput/cost projection.

### 5.3 `dinov2_rsna` (Stratum A — RSNA-Supervised Head on Frozen Encoder)
- **Backbone & Labeling:** Frozen `facebookresearch/dinov2` **`dinov2_vits14`** (`21M` params, `384`-d CLS token; `Apache-2.0`). Must be labeled **`DINOv2 + RSNA head`** in every table/figure.
- **Preprocessing (Reuses Existing `1520×912` Stage-1 Caches!):**
  - Load lossless cropped `1520×912` `uint16` PNGs -> min-max `[0, 1]` -> aspect-ratio-preserving pad-to-square + bilinear resize to **`518 × 518`** (`37 × 14` patch grid) -> 3-channel replicate -> ImageNet mean `(0.485, 0.456, 0.406)` / std `(0.229, 0.224, 0.225)`.
- **One-Pass Feature Cache & Head Training:**
  - Extract **1 deterministic `384`-d FP32 CLS embedding per unique image** once (`54,706` RSNA = `~84 MB`; `26,988` NLBS = `~41 MB`).
  - Train linear / lightweight MLP head (`LayerNorm(384) -> Dropout(0.3) -> Linear(384, 1)`) strictly on RSNA `train` (`8,339` pts) with `BalancedBatchSampler` (`20:60`) or `pos_weight`, AdamW, cosine LR, using `train_calib` (`1,192` pts) ONLY for epoch/hyperparameter selection (`val_calib` strictly untouched).
- **Aggregation:** `exam_score = max(image_scores in exam)`.

### 5.4 `resnet50_rsna` (Stratum A — RSNA End-to-End Fine-Tuning)
- **Backbone & Initialization:** `torchvision.models.resnet50(weights="IMAGENET1K_V2")` (`25.6M` params, `BSD-3-Clause`) with binary cancer head.
- **Preprocessing (Reuses Existing `1520×912` Stage-1 Caches!):**
  - Load lossless cropped `1520×912` `uint16` PNGs -> aspect-ratio-preserving resize to **`768 × 512`** (or `1024 × 614` if 1-epoch timing fits budget; benchmark both in pilot!) -> min-max `[0, 1]` -> TRUST Sec. 3.2 training augmentations (`±10°` rot, `0.9–1.1` scale, `±5%` shift, brightness/contrast) -> 3-channel replicate -> ImageNet mean/std.
- **Training Recipe:** End-to-end AMP fine-tuning on RSNA `train` (`8,339` pts) with `BalancedBatchSampler` (`20:60`), `AdamW` (`lr=1e-4`, `weight_decay=1e-4`), cosine schedule, gradient clipping, and `train_calib` early stopping / LR selection (`val_calib` strictly untouched).
- **Quality & Budget Gate:** Must pass a 1-epoch L40S timed benchmark fitting remaining Modal cap AND reach nontrivial `train_calib` source discrimination (AUROC `>= 0.70`) before full training. If budget cannot support adequate convergence, mark **`NOT COMPLETED`** honestly.

---

## 6. Reusable Caches & Cost-Optimized Execution Strategy ($30 Modal Cap) (Task 6)

### 6.1 Verified Reusable Assets ($0 Rebuild Cost)

| Asset | Location | Contents & Size | Reusable By |
|---|---|---|---|
| **CE & TRUST Checkpoints + RSNA Eval Cache** | Local `models/` | Epoch-20 `.pt` checkpoints + `eval_{train,val}_calib_fp32.pt` (`135 MB`) | `ce_b5`, `trust_b5` (CPU source sub-split eval) |
| **Locked NLBS Stage C Scores** | Local `reports/stage_c/` | `locked_nlbs_predictions.csv` (`6,359` studies) | `ce_b5`, `trust_b5` (4-view cohort filter) |
| **RSNA Stage-1 `1520×912` `uint16` PNGs** | Modal `rsna-mammography-vol` (`hasinishrak2015` & `74001`) | `54,706` lossless cropped PNGs (`65.9 GiB`) | `dinov2_rsna`, `resnet50_rsna` |
| **NLBS Stage-1 `1520×912` `uint16` PNGs** | Modal `rsna-mammography-vol` (`hasinishrak2015`) | `26,988` lossless cropped PNGs | `dinov2_rsna`, `resnet50_rsna` |
| **Raw DICOMs (RSNA + NLBS)** | Modal `rsna-mammography-vol` & `nlbs-mammography-vol` (`hasinishrak2015`) | `54,706` RSNA + `26,988` NLBS extracted `.dcm`s | `gmic_pretrained`, `nyu_fourview_pretrained` (CPU native crop staging) |

### 6.2 Cheapest Scientifically Valid Execution Sequence ($30 Envelope)

Pricing anchor (`Modal`, verified Nov 2026): `NVIDIA L40S` = `$1.9512/hr` (`~$2.11–$2.36/hr` with `2–6` CPUs & `8–16 GiB` RAM); CPU-only `2 cores / 4 GiB` = `~$0.126/hr/worker`. Always run DICOM decoding/cropping on **CPU workers** and stage PNGs to **NVMe `/tmp`** before L40S inference (proven `107×` I/O speedup in Stage B/C).

| Order | Work Package | Hardware & Scope | Projected Cost Range | Gate Before Full Run |
|---:|---|---|---:|---|
| **0** | **Scaffold, Split Freeze & Modal Credit Check** | Local CPU + `modal` billing check | **$0.00** | Confirm `>= $30` live credit on target profile (`hasinishrak2015` preferred—holds both RSNA & NLBS raw + Stage-1 PNGs); freeze `studies/paper1/` splits & 4-view selector. |
| **1** | **`dinov2_rsna` (ViT-S/14)** | Reuses `1520×912` PNGs (`81,694` imgs) -> 1-pass L40S extraction + CPU/T4 head train | **$1.50 – $3.00** | 500-image throughput pilot + verify `train_calib` AUROC on linear/MLP head (`$0` CPU train). |
| **2** | **`gmic_pretrained`** | CPU native `2944×1920` crop (`37,970` imgs: RSNA `val_calib` + NLBS only) + L40S single-ckpt inference | **$4.00 – $6.50** | `sample_data` numerical parity + 200-image L40S throughput/cost forecast `<= $7`. |
| **3** | **`nyu_fourview_pretrained`** | CPU native `2677×1942` / `2974×1748` crop (`8,566` exams: RSNA `val_calib` + NLBS 4-view) + L40S image-only inference | **$4.00 – $6.50** | `sample_data` numerical parity + 200-exam L40S throughput/cost forecast `<= $7`. |
| **4** | **`resnet50_rsna` (Cost-Gated)** | Reuses RSNA `1520×912` PNGs (`38,251` train) + NLBS `1520×912` (`26,988`) on L40S (`768×512` / `1024×614`) | **$8.00 – $12.00** | 1-epoch L40S timed benchmark; **ABORT/MARK NOT COMPLETED** if `1.3 × projected_total > remaining_budget`. |
| **5** | **6-Model Statistics, Bootstrap & Figures** | Local CPU (`B = 2,000` cluster bootstrap, UpSet overlap) | **$0.00** | Verify all NLBS prediction SHA-256 locks prior to outcome join. |
| **Total** | **Full 4-Model Addition** | `~$17.50 – $28.00` (+ `$2–$12` safety margin) | **<= $30.00 Cap** | **Hard Stop:** Never exceed `$30` additional spend. |

---

## 7. Stage 0 GO / NO-GO Decision

### Verdict: **CONDITIONAL GO**

Stage 0's scientific and engineering audit **PASSES**:
1. NLBS denominators (`5,997 / 5,935 / 6,430 / 6,359 / 5,942` and `149 / 151` positives) are reconciled to exact integer counts, with `6,184` common 4-view eligible studies (`151/151` cancers) verified.
2. RSNA `val_calib` double-use is resolved via a zero-retrain patient-disjoint 2-fold (`source_op_calib` / `source_heldout_test`) evaluation design paired with `val_calib`-only threshold continuity for NLBS.
3. CE and TRUST checkpoints, cached RSNA embeddings, and locked NLBS predictions are 100% reusable at `$0` GPU cost without touching Stage B/C files.
4. Pretrained provenance/leakage (`UPMC`, `NYU 2010–2017`, `LVD-142M`, `ImageNet-1K`) and license constraints (`CC BY-NC-SA 4.0`, `AGPL-3.0`, `Apache-2.0`) are mapped.
5. Reusing existing `1520×912` Stage-1 PNGs for DINOv2/ResNet-50 and restricting GMIC/NYU inference to RSNA `val_calib` + NLBS brings projected total cost (`~$17.50–$28.00`) inside the `$30` Modal cap.

### Mandatory Gates Before Launching Any Paid GPU Job

1. **Gate 0.1 (Account & Balance Check):** Verify live Modal USD balance on `hasinishrak2015` (where both `rsna-mammography-vol` and `nlbs-mammography-vol` reside) is sufficient before starting Phase 1.
2. **Gate 0.2 (CPU Protocol Freeze):** Create `studies/paper1/` (`README.md`, `protocol_v1.md`, `model_manifest.yaml`, `eval_config.yaml`, `analysis_plan.md`), deterministic RSNA `val_calib` 2-fold split CSV, deterministic NLBS 4-view eligibility + tie-break manifest, and CPU baseline CE/TRUST RSNA 2-fold metrics—**zero GPU spend**.
3. **Gate 0.3 (Per-Model Parity & Throughput Pilot):** Each new model must pass its small `sample_data`/smoke parity check and 200-unit L40S throughput/cost projection before full execution.
