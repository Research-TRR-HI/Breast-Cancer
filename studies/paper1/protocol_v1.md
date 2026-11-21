# Paper 1 — Experimental Protocol v1.0 (Gate 0.2 Frozen Specification)

**Effective Date:** 2026-11-21  
**Status:** FROZEN PRIOR TO MULTI-MODEL NLBS INFERENCE  
**Disclosure of Retrospective Expansion:** Stage C NLBS results for `ce_b5` (`90.07%` @98%) and `trust_b5` (`84.77%` @98%) were inspected on 2026-11-20 prior to selecting the four added models. Paper 1 is therefore explicitly registered as a **post hoc multi-model expansion** with strictly label-free external inference and frozen pre-join prediction checksums, not a prospectively blinded model-selection study.

---

## 1. Dataset & Evaluation Units

### 1.1 Source Dataset — RSNA Screening Mammography (`54,706` images, `11,913` patients, `486` cancers)
- **Training & Internal Selection Partition (`splits_seed42.csv`):**
  - `train`: `8,339` patients (`38,251` images, `340` cancer-positive patients) — used ONLY to fit `ce_b5`, `trust_b5`, `dinov2_rsna` head, and `resnet50_rsna`.
  - `train_calib`: `1,192` patients (`5,473` images, `49` cancer-positive patients) — used ONLY for `trust_b5` dynamic `tau_t` and `dinov2_rsna` / `resnet50_rsna` hyperparameter/early-stopping selection.
  - **`val_calib` (`2,382` patients, `10,982` images, `97` cancer-positive patients):** Strictly **100% unseen** during training and checkpoint selection for ALL six models.
- **Separation of Source Calibration vs. Independent Source Testing (`rsna_val_calib_2fold_seed42.csv`):**
  - `val_calib` is partitioned 50/50 at `patient_id` level, stratified by `(cancer, site_id)` (`seed=42`), into `fold_a` (`1,191` pts / `49` pos) and `fold_b` (`1,191` pts / `48` pos).
  - **Independent RSNA Source Test (Primary RSNA Eval):** 2-fold patient-disjoint cross-fitting (`fold_a -> fold_b` and `fold_b -> fold_a`, pooled out-of-fold over all `2,382` patients / `97` positives), plus individual `fold_a -> fold_b` (`48` test pos) and `fold_b -> fold_a` (`49` test pos) estimates. Thresholds are NEVER evaluated on the fold used to select them.
  - **NLBS Operating Threshold Calibration:** Calibrated on full RSNA `val_calib` (`2,382` pts, `97` pos) to match immutable Stage C thresholds (`ce_b5: 0.09619140625 / 0.12445068359375`; `trust_b5: 0.1795654296875 / 0.1954345703125`) and avoid single-fold `n_pos=49` min-score collapse (`0` misses allowed at 98%). A secondary NLBS sensitivity check using `fold_a` (`49` pos) thresholds is also reported.

### 1.2 Target Dataset — NLBS (`26,988` images, `6,359` unique studies, `5,942` clusters)
- **Primary Inference & Scoring Unit:** Deduplicated **`StudyInstanceUID` examination** (`6,359` full NLBS; `151` positive exams).
- **Primary Paired Common Cohort:** **`N = 6,184`** 4-view eligible examinations (`151` cancer-positive, `6,033` negative; `5,868` `split_cluster_id`s, `149` positive clusters) containing all 4 standard views `{L-CC, L-MLO, R-CC, R-MLO}`, enabling exact paired comparison across all 6 models including `nyu_fourview_pretrained`.
- **Secondary Full-Cohort Sensitivity:** `N = 6,359` examinations (`5,942` clusters) for the 5 view-agnostic models (`ce_b5`, `trust_b5`, `gmic_pretrained`, `dinov2_rsna`, `resnet50_rsna`).
- **Secondary Cluster-Collapsed Sensitivity:** Aggregate scores per `split_cluster_id` (`max` over visits -> `5,868` 4-view / `5,942` full independent proxy clusters, `149` positive clusters).

---

## 2. Label-Blind Multi-View Selection Rule (NYU Four-View)
Applied identically without ever reading cancer labels or cohort folders (`datasets/paper1/nlbs_4view_common_cohort_label_free.csv` & `datasets/paper1/rsna_val_calib_4view_selection.csv`):
1. Require >= 1 image in each of `{L-CC, L-MLO, R-CC, R-MLO}`.
2. If >1 image exists in a `(StudyInstanceUID, laterality, view_position)` slot:
   - **NLBS:** Sort candidates by `(0 if manufacturer == 'GE MEDICAL SYSTEMS' else 1, dicom_rel_path, image_uid)` and pick index `0`.
   - **RSNA:** Sort candidates by `(int(image_id),)` ascending and pick index `0`.

---

## 3. Target-Label Quarantine & Pre-Join Cryptographic Lock
1. NLBS inference scripts mount ONLY label-free manifests (`datasets/nlbs/inference_manifest_stage_c.csv` or `datasets/paper1/nlbs_4view_common_cohort_label_free.csv`) checked with `assert_no_forbidden_columns`.
2. Model checkpoints, preprocessing hashes, and RSNA-selected thresholds (`tau_98`, `tau_95`) MUST be frozen and hashed prior to NLBS scoring.
3. Raw per-exam NLBS scores and frozen dismissal flags MUST be written and SHA-256 locked BEFORE joining `datasets/nlbs/exam_manifest.csv` outcomes.
4. No NLBS outcomes, `cohort_group` folders, or score distributions may be used to tune any model, checkpoint, preprocessing, or threshold in Paper 1.
