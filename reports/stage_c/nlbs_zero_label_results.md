# Stage C Report: Strict Zero-Target-Label External Evaluation on NLBS (RSNA -> NLBS)

**Evaluated UTC:** `2026-11-20T13:25:24.521768+00:00`  
**Protocol:** Protocol 1 — Strict Source-Only / Frozen-Threshold Zero-Target-Label Transfer (`NLBS` cancer outcomes strictly quarantined until cryptographic lock verification).  
**Retraining / Adaptation Applied:** None (`0` NLBS training updates, `0` NLBS threshold updates, `0` UPA alignment).

---

## 1. Experimental Protocol & Target-Label Quarantine Evidence
- **Training Cohort (Source Only):** RSNA Screening Mammography (`train=38,251`, `train_calib=5,473`, `val_calib=10,982` images; `0` patient leakage), frozen Epoch-20 checkpoints from Stage B (`b5-model-best-epoch-7.tar` frozen backbone).
- **Inference Input Isolation:** Built label-free manifest `datasets/nlbs/inference_manifest_stage_c.csv` (`sha256: c3d6a3b390003139c75564c1ffb22a6af8f1a9c8cd57dc4c1d96b8ce3269e95f`) explicitly excluding `Cancer`, `exam_cancer`, `n_cancer_images`, `Hard to State Negative`, `hard_to_state_negative`, and `cohort_group`, sorted by clinical `StudyInstanceUID` rather than directory class folders. Only `src/` and this label-free manifest were mounted into the Modal GPU inference container.
- **Cryptographic Pre-Evaluation Lock (`verified_before_label_join = True`):**
  - `locked_nlbs_predictions.csv` SHA-256: `526434d058510208ee825b3db36380d0faec7760c17dd2c867dc6288eeb10cc2`
  - `locked_nlbs_image_diagnostics.csv.gz` SHA-256: `1817a8ab8e5ae31cbe357a2713ee859edd6e4d998127efc72babea0a361558b9`
  - `ce_rsna_seed42_final_epoch20.pt` SHA-256: `adf1443455ee1b507485ee4b7f64932d5cf6f59bb6afcf11b9f415c634372d22`
  - `trust_rsna_seed42_final_epoch20.pt` SHA-256: `63388d790ce9cf8a8287adc7724358ced392a69b332ce1a5e46972750faa51cb`
  - `b5-model-best-epoch-7.tar` SHA-256: `0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c`
  - Frozen RSNA Thresholds SHA-256: `734e9abb7cd1346e4c3e7be265794f1700af383f8f0eb6c2ca46e0d9bc791723`
- **NLBS Denominators After Exact `StudyInstanceUID` Deduplication:** `6,430` folder exam rows -> **`6,359` unique examinations** (`71` cross-folder duplicate aliases merged) across **`5,942` patient clusters (`split_cluster_id`)**, comprising **`151` cancer-positive exams** (`149` cancer clusters; prevalence `2.37%`), `1,488` False Positive recalled exams, and `4,720` Normal screening exams.

---

## 2. Primary Zero-Label Results: CE vs. TRUST at Frozen RSNA Operating Thresholds

### Examination-Level Performance (`N = 6,359` Unique Studies, `151` Cancer-Positive Exams)

| Model | Source Target | Frozen RSNA `tau` | Sensitivity (95% Cluster CI) | Missed / Total Cancers | Dismissed / Total Exams | Dismissal Rate (95% Cluster CI) | Cancer Prev Dismissed (95% Cluster CI) | One-Sided 99% CP UB* | Meets Target? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CE-Only** | 95% Recall | `0.124451` | **80.13%** [73.75%, 86.23%] | **30** / 151 | 3,173 / 6,359 | **49.90%** [48.57%, 51.13%] | 0.945% [0.644%, 1.292%] | 1.427% | `False` |
| **CE-Only** | 98% Recall | `0.096191` | **90.07%** [84.96%, 94.38%] | **15** / 151 | 1,939 / 6,359 | **30.49%** [29.35%, 31.71%] | 0.774% [0.418%, 1.181%] | 1.375% | `False` |
| **TRUST** | 95% Recall | `0.195435` | **80.79%** [74.12%, 86.91%] | **29** / 151 | 3,143 / 6,359 | **49.43%** [48.09%, 50.67%] | 0.923% [0.608%, 1.261%] | 1.403% | `False` |
| **TRUST** | 98% Recall | `0.179565` | **84.77%** [78.72%, 90.20%] | **23** / 151 | 2,529 / 6,359 | **39.77%** [38.48%, 41.09%] | 0.909% [0.582%, 1.295%] | 1.453% | `False` |

*Note on 99% Clopper-Pearson Upper Bound: Computed under binomial independence across the `6,359` deduplicated exams for protocol comparability with TRUST; because repeat exams from the same patient (`5,942` clusters) are non-independent, the `split_cluster_id` bootstrap 95% CIs above represent cluster-adjusted uncertainty.*

### Source (`RSNA val_calib`) vs. Unseen Target (`NLBS`) Shift Comparison

| Model | Operating Point | RSNA Sensitivity -> NLBS Sensitivity | RSNA Dismissal -> NLBS Dismissal |
| :--- | :---: | :---: | :---: |
| **CE-Only** | 95% Target (`0.124451`) | 95.88% -> **80.13%** | 22.50% -> **49.90%** |
| **TRUST** | 95% Target (`0.195435`) | 95.88% -> **80.79%** | 20.70% -> **49.43%** |
| **CE-Only** | 98% Target (`0.096191`) | 98.97% -> **90.07%** | 14.48% -> **30.49%** |
| **TRUST** | 98% Target (`0.179565`) | 98.97% -> **84.77%** | 17.17% -> **39.77%** |

---

## 3. Secondary Ranking Metrics & Paired Cluster Differences (`B = 2,000` Bootstrap on `5,942` Clusters)
- **CE-Only NLBS AUROC:** `0.7101` (95% CI `[0.6694, 0.7469]`) | **AUPRC:** `0.0539` (95% CI `[0.0414, 0.0760]`)
- **TRUST NLBS AUROC:** `0.7052` (95% CI `[0.6644, 0.7429]`) | **AUPRC:** `0.0529` (95% CI `[0.0406, 0.0743]`)
- **Paired `TRUST - CE` Cluster Bootstrap Differences:**
  - `delta_auroc`: `-0.0049` (`[-0.0093, -0.0006]`)
  - `delta_auprc`: `-0.0010` (`[-0.0029, 0.0007]`)
  - `delta_sensitivity @95%`: `+0.66%` (`[-1.39%, 2.94%]`) | `delta_dismissal @95%`: `-0.47%` (`[-0.90%, -0.05%]`)
  - `delta_sensitivity @98%`: `-5.30%` (`[-9.15%, -1.92%]`) | `delta_dismissal @98%`: `+9.28%` (`[8.55%, 10.00%]`)
- **Cluster-Collapsed Independence Reference (`5,942` patient clusters, `149` positive clusters):** CE@98% sensitivity `89.93%` / dismissal `29.43%`; TRUST@98% sensitivity `84.56%` / dismissal `38.54%`.

---

## 4. Preprocessing Failures, Scanner Breakdown & Fallbacks
- **Images Verified in Manifest:** `26,988` | **Succeeded:** `26,988` | **Decode/Missing Failures:** `0` | **All Scores Finite:** `True`
- **Uncropped Segmentation Fallbacks (`strict_segmentation=False`):** `0` images (`segmentation_status_counts`: `{'success': 26988}`)

| Scanner Manufacturer | Images | Decode Failures | Uncropped Fallbacks | CE Mean Score | TRUST Mean Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `GE MEDICAL SYSTEMS` | 26,967 | 0 | 0 | `0.10466` | `0.16704` |
| `SIEMENS` | 21 | 0 | 0 | `0.12415` | `0.19054` |

---

## 5. Limitations & Hypothesis Verdict
- **Hypothesis Verdict:** **CONTRADICTS zero-label frozen-threshold transfer safety on NLBS**: frozen RSNA thresholds failed to preserve the intended cancer sensitivity on the unseen Canadian cohort (and/or collapsed dismissal utility). Proceed directly to Stage D (UPA unlabeled adaptation) to test whether distribution alignment fixes the operating-point drift without target cancer labels.
- **Key Study Limitations:**
  1. NLBS represents one Canadian screening program acquired predominantly on GE Senographe Essential (`99.92%`) + `21` Siemens (`0.08%`) FFDM units (2008–2010), without density tags or lesion bounding boxes.
  2. Due to stripped DICOM `PatientID` and `AcquisitionDate` in public NLBS metadata, linked clusters reflect connected `StudyInstanceUID` / folder components (`5,942` clusters); any unlinked temporal repeats remain unmerged.
  3. Observed retrospective Clopper-Pearson bounds do not constitute a prospective universal hospital safety guarantee.
