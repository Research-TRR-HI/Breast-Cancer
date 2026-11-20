# Research Log: Stage C Strict Zero-Target-Label Evaluation on NLBS (RSNA -> NLBS)

**Date**: 2026-11-20  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Benchmark Evaluation / Calibration Drift Under Site Shift  
**Milestone Scope**: Target-label-quarantined frozen inference across all `26,988` NLBS DICOMs (`6,359` deduplicated `StudyInstanceUID` examinations across `5,942` connected `split_cluster_id` patient clusters), cryptographic prediction locking, and cluster-bootstrap evaluation (`reports/stage_c/nlbs_zero_label_results.{json,md}`)  
**Status**: VERIFIED COMPLETE  

---

## 1. Research Objective & Context
- Execute AnchorShift **Protocol 1 (Strict Source-Only / Frozen-Threshold Zero-Target-Label Transfer)**: test whether the Epoch-20 **CE** (`ce_rsna_seed42_final_epoch20.pt`) and **TRUST** (`trust_rsna_seed42_final_epoch20.pt`) models trained and thresholded strictly on RSNA maintain clinical `95%` and `98%` cancer sensitivity when deployed without target labels or adaptation on the unseen Canadian NLBS screening cohort.
- Zero NLBS cancer labels, `cohort_group` folders, `Hard to State Negative`, or UPA adaptations were allowed during inference or decision-making.

---

## 2. Target-Label Quarantine & Cryptographic Lock Evidence
- **Label-Free Inference Manifest:** `datasets/nlbs/inference_manifest_stage_c.csv` (`sha256: c3d6a3b390003139c75564c1ffb22a6af8f1a9c8cd57dc4c1d96b8ce3269e95f`, `26,988` images) explicitly strips all label/cohort fields (`Cancer`, `cancer`, `exam_cancer`, `n_cancer_images`, `Hard to State Negative`, `hard_to_state_negative`, `cohort_group`) and sorts rows by `StudyInstanceUID` rather than directory class prefixes.
- **Container Isolation:** Modal app `anchorshift-stage-c-nlbs-eval` (`ap-rxM08NSdg43qLOjpUZ4jkR`, profile `hasinishrak2015`) mounted ONLY `src/` and `inference_manifest_stage_c.csv`; `manifest.csv` and `exam_manifest.csv` never entered cloud containers.
- **Locked Artifact Digests (Verified Prior to Label Join at `2026-11-20T13:22:23Z`):**
  - `reports/stage_c/locked_nlbs_predictions.csv` SHA-256: `526434d058510208ee825b3db36380d0faec7760c17dd2c867dc6288eeb10cc2`
  - `reports/stage_c/locked_nlbs_image_diagnostics.csv.gz` SHA-256: `1817a8ab8e5ae31cbe357a2713ee859edd6e4d998127efc72babea0a361558b9`
  - `ce_rsna_seed42_final_epoch20.pt` SHA-256: `adf1443455ee1b507485ee4b7f64932d5cf6f59bb6afcf11b9f415c634372d22`
  - `trust_rsna_seed42_final_epoch20.pt` SHA-256: `63388d790ce9cf8a8287adc7724358ced392a69b332ce1a5e46972750faa51cb`
  - `b5-model-best-epoch-7.tar` SHA-256: `0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c`
  - Frozen RSNA Thresholds (`ce: 0.12445068359375 / 0.09619140625`, `trust: 0.1954345703125 / 0.1795654296875`) SHA-256: `734e9abb7cd1346e4c3e7be265794f1700af383f8f0eb6c2ca46e0d9bc791723`

---

## 3. Deduplication & Preprocessing Integrity
- **Asymmetric Shared-Study Resolution:** Among the `71` cross-folder duplicate `StudyInstanceUID` aliases (`6,430` folder exam rows -> `6,359` unique studies), `11` pairs had asymmetric image counts (`e.g. p_4576_v1: 12` vs `p_5677_v1: 4`). Aggregating `max` across all valid images sharing each `StudyInstanceUID` preserved 100% of views while collapsing duplicate studies cleanly (`6,359` studies across `5,942` `split_cluster_id`s; `151` cancer-positive exams / `149` cancer clusters).
- **Preprocessing Results (`NVIDIA L40S`, AMP, `32.05 img/s`):** `26,988 / 26,988` decoded cleanly (`0` missing/corrupt files, `0` uncropped fallbacks, `100%` finite scores) across both `GE MEDICAL SYSTEMS` (`26,967`) and `SIEMENS` (`21`).

---

## 4. Empirical Zero-Label Results (`N = 6,359` Exams, `151` Cancers, `B = 2,000` Cluster Bootstrap)

| Model | Target Recall | Frozen RSNA `tau` | NLBS Sensitivity (95% Cluster CI) | Missed Cancers | NLBS Dismissal (95% Cluster CI) | Dismissed Prevalence (95% Cluster CI) | 99% CP UB* | Meets Target? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CE-Only** | `95%` | `0.124451` | **80.13%** [73.75%, 86.23%] | **30 / 151** | **49.90%** [48.57%, 51.13%] | 0.945% [0.644%, 1.292%] | 1.427% | `False` |
| **CE-Only** | `98%` | `0.096191` | **90.07%** [84.96%, 94.38%] | **15 / 151** | **30.49%** [29.35%, 31.71%] | 0.774% [0.418%, 1.181%] | 1.375% | `False` |
| **TRUST** | `95%` | `0.195435` | **80.79%** [74.12%, 86.91%] | **29 / 151** | **49.43%** [48.09%, 50.67%] | 0.923% [0.608%, 1.261%] | 1.403% | `False` |
| **TRUST** | `98%` | `0.179565` | **84.77%** [78.72%, 90.20%] | **23 / 151** | **39.77%** [38.48%, 41.09%] | 0.909% [0.582%, 1.295%] | 1.453% | `False` |

- **Secondary Metrics:** CE AUROC `0.7101` (`[0.6694, 0.7469]`) / AUPRC `0.0539` (`[0.0414, 0.0760]`); TRUST AUROC `0.7052` (`[0.6644, 0.7429]`) / AUPRC `0.0529` (`[0.0406, 0.0743]`).
- **Paired `TRUST - CE` @ 98% Target:** `delta_sensitivity = -5.30%` (`95% CI [-9.15%, -1.92%]`, TRUST misses `+8` more cancers: `23` vs `15`) while over-dismissing `+9.28%` (`[8.55%, 10.00%]`).
- **Cluster-Collapsed Single-Visit Reference (`5,942` clusters, `149` positive):** CE@98% sensitivity `89.93%` (`134/149`), TRUST@98% sensitivity `84.56%` (`126/149`).

---

## 5. Scientific Interpretation & Next Stage
- **Hypothesis Verdict:** **CONTRADICTS zero-label frozen-threshold transfer safety on NLBS.** Under RSNA -> NLBS site shift, median exam scores shifted downward right onto the `95%` RSNA cutoffs (`CE median=0.12451 vs 0.12445`, `TRUST median=0.19617 vs 0.19543`), causing severe over-dismissal (`30.49%–49.90%` vs `14.48%–22.50%` on RSNA) and losing `10–15%` absolute sensitivity. Notably, TRUST's dismissal penalty pushed RSNA scores higher (elevating source `tau_98` from `0.0962` to `0.1796`), which amplified over-dismissal and missed cancers when applied uncalibrated on NLBS (`39.77%` dismissed / `23` missed vs CE `30.49%` dismissed / `15` missed).
- **Next Stage (Stage D):** Implement and evaluate **UPA (Roschewitz et al., Nature Comms 2023)** using unlabeled NLBS adaptation images (cluster-separated from NLBS evaluation) alongside TRUST's 20%-target-calibrated reference (`Protocol 3`) to determine whether score-CDF alignment restores target sensitivity or whether representation shift (`AUROC ~0.71`) limits safe dismissal.
