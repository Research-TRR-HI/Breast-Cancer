# Research Log: [Paper 1 — Stage 0] Evaluation, Cohort Reconciliation, Leakage, and Budget Readiness Audit

**Date**: 2026-11-21  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: `[Paper 1]` Dataset Forensic Audit | Patient Splitting & Leakage Audit | Compute & Budget Planning  
**Milestone Scope**: Stage 0 pre-execution readiness audit across `datasets/nlbs/*`, `datasets/rsna/*`, `models/{cache_eval_rsna,checkpoints}/*`, `reports/stage_{b,c}/*`, and `Paper1_External_Validation_Research_Context.md` (**CPU-only; zero GPU jobs run**)  
**Status**: CONDITIONAL GO (Report: `reports/paper1/stage0_readiness_audit.md`)

---

## 1. Research Objective & Context
- Execute **Paper 1 — Stage 0: Evaluation and Dataset Audit** prior to spending any of the **$30 USD** additional Modal L40S compute budget on the four added models (`gmic_pretrained`, `nyu_fourview_pretrained`, `dinov2_rsna`, `resnet50_rsna`).
- Reconcile NLBS published counts (`5,997 / 5,935 / 149`) with Stage A3/C counts (`6,430 / 6,359 / 5,942 / 151`), resolve RSNA `val_calib` calibration-vs-test double-use without blowing budget on CE/TRUST retraining, verify reuse of local CE/TRUST artifacts, audit patient/pretrained data overlap and licenses, specify exact 4-model pipelines, and design the cheapest valid Modal execution sequence.
- Preserve all Paper 2 (`AnchorShift_Research_Context.md`) plans and immutable Stage B (`reports/stage_b/`) and Stage C (`reports/stage_c/`) baselines untouched.

---

## 2. Datasets, Volumes & Manifest References
- **RSNA Manifest & Splits:**
  - `datasets/rsna/manifest.csv`: `54,706` images across `11,913` unique patients (`486` positive, `4.08%`); `100.00%` (`11,913 / 11,913`) have all 4 standard views `{L-CC, L-MLO, R-CC, R-MLO}`; `3,680` patients have `>4` images.
  - `datasets/rsna/splits_seed42.csv`: `train` (`8,339` pts, `340` pos, Site 1: `4,040` / Site 2: `4,299`), `train_calib` (`1,192` pts, `49` pos, Site 1: `595` / Site 2: `597`), `val_calib` (`2,382` pts, `97` pos, Site 1: `1,183` / Site 2: `1,199`). Zero patient overlap.
- **NLBS Manifests:**
  - `datasets/nlbs/manifest.csv`: `26,988` images (`26,967` GE Senographe Essential, `21` Siemens Mammomat Novation DR; `100%` `MONOCHROME2`, `12-bit`, `1.2.840.10008.1.2.1`).
  - `datasets/nlbs/exam_manifest.csv`: `6,430` `(case_id, StudyInstanceUID)` rows -> `6,359` unique `StudyInstanceUID`s -> `5,942` connected `split_cluster_id` components (`149` positive clusters, `151` positive exams).
  - `datasets/nlbs/inference_manifest_stage_c.csv` (`sha256: c3d6a3b3...`): strictly label-free inference manifest (`26,988` rows).
- **Local Reusable Artifacts Verified:**
  - `models/cache_eval_rsna/eval_train_calib_fp32.pt` (`[5473, 2048]`, SHA-256 `b7722965...`) & `eval_val_calib_fp32.pt` (`[10982, 2048]`, SHA-256 `bb983e4d...`).
  - `models/checkpoints/stage_b_rsna/{ce,trust}_rsna_seed42_final_epoch20.pt` (SHA-256 `adf14434...` / `63388d79...`).
  - `reports/stage_c/locked_nlbs_predictions.csv` (`6,359` deduplicated exam scores, SHA-256 `526434d0...`).

---

## 3. Forensic Evidence & Diagnostic Findings

1. **Exact NLBS Reconciliation:**
   - `5,997` `case_id` folders (`4,332` normal + `1,516` False Positive + `149` abnormal) expand to `6,430` `(case_id, StudyInstanceUID)` pairs because `385` folders hold multiple studies (`378 × 2`, `6 × 3`, `p_3280 × 44`, adding `378 + 12 + 43 = 433`).
   - `71` `StudyInstanceUID`s appear in `2` folders each (`43` normal, `28` False Positive, `0` abnormal), reducing `6,430 - 71 = 6,359` unique `StudyInstanceUID`s.
   - Union-Find collapses `84` `case_id`s across `29` multi-case clusters (`28` False Positive pairs + `1` normal cluster `nlbs_cluster_4240` linking `p_3280` with `27` other normal folders across `45` UIDs), giving `5,997 - (84 - 29) = 5,942` `split_cluster_id`s.
   - Positive `case_id` folders `p_4334` and `p_4461` each have `2` distinct positive `StudyInstanceUID`s (`4` images / `2` cancer images per visit), reconciling `149` positive clusters to `151` positive exams.
   - Published `5,935` "unique cases" reflects `62` repeat exams noted in Kendall et al. (*Scientific Data* 2025), partially unlinked in the 2025-11-25 directory-renaming rebuild (`PatientID` stripped in `26,984/26,988` images; `Exam Date` `100%` null).
   - **4-View Eligibility (NYU Common Cohort):** `6,184 / 6,359` (`97.25%`) NLBS studies have `{L-CC, L-MLO, R-CC, R-MLO}`, including **`151 / 151` (`100%`)** of all cancer-positive exams (`175` ineligible studies are all cancer-negative: `163` normal, `12` False Positive). `935 / 6,184` eligible 4-view studies contain `>4` images. At cluster level, `5,868 / 5,942` (`149/149` positive) clusters have `>=1` 4-view study.

2. **Local CPU FP32 vs. Stage-B L40S AMP Precision Finding:**
   - Re-evaluating `ce_rsna_seed42_final_epoch20.pt` and `trust_rsna_seed42_final_epoch20.pt` on `eval_val_calib_fp32.pt` in CPU `float32` matches Stage B's L40S `torch.amp.autocast("cuda")` (`float16` head matmul) within `2.06e-6` to `3.55e-5` on `tau` (`ce@95` `0.124415` vs `0.124451`, shifting `1` borderline negative dismissal: `537` vs `536`; `trust` `493` & `409` identical). Stage B/C JSONs remain canonical and untouched.

3. **Pretrained-Model Overlap & Licensing Audit:**
   - **Mammo-CLIP B5 (`b5-model-best-epoch-7.tar`):** Pretrained on UPMC (`13,829` patients / `25,355` BI-RADS 0–2 mammograms; `upmc_clip` fold 0). RSNA/VinDr were downstream evals only. License: `CC BY-NC-SA 4.0`.
   - **GMIC & NYU Four-View (`nyukat/*`):** Pretrained on private NYU Breast Cancer Screening Dataset (`229,426` exams, `2010–2017`). Presumed zero overlap with RSNA (Emory/Victoria) and NLBS (Canada, released 2025), though unverified cross-institutional patient migration is disclosed. License: `AGPL-3.0`.
   - **DINOv2 (`dinov2_vits14`):** Self-supervised on LVD-142M natural images (`Apache-2.0`); zero medical overlap.
   - **ResNet-50 (`IMAGENET1K_V2`):** Supervised on ImageNet-1K natural images (`BSD-3-Clause`); zero medical overlap.

---

## 4. Methodological Decisions & Cost-Optimization Plan

1. **RSNA Source Calibration vs. Independent Source Test (Zero-Retrain Design):**
   - Retain `train` (`8,339` pts) and `train_calib` (`1,192` pts) for training new RSNA-supervised models (`dinov2_rsna`, `resnet50_rsna`), keeping `val_calib` (`2,382` pts, `97` pos) strictly untouched during training/model selection for all 6 models.
   - Partition `val_calib` 50/50 stratified by `(cancer, site_id)` into `source_op_calib` (`1,191` pts / `49` pos) and `source_heldout_test` (`1,191` pts / `48` pos), and compute **2-fold out-of-fold RSNA source performance** (`Fold A -> Fold B` and `Fold B -> Fold A`, `97` total positives) on CPU at `$0` GPU cost for CE & TRUST.
   - For NLBS external thresholds, keep full `val_calib` (`97` pos) as primary calibration set (matching immutable Stage C CE/TRUST cutoffs and avoiding `n_pos=49` min-score instability where `0` misses are allowed at `98%`), with `Fold A (49 pos)` reported as a secondary sensitivity check.
2. **53% Compute Reduction for Pretrained-Direct Models:**
   - Because `gmic_pretrained` and `nyu_fourview_pretrained` require zero RSNA training, run native-resolution preprocessing & inference **ONLY on RSNA `val_calib` + NLBS** (`37,970` images for GMIC; `8,566` 4-view exams / `34,264` images for NYU), rather than all `81,694` images.
3. **Cache Reuse & Execution Order Within $30 Cap:**
   - Reuse existing `1520×912` Stage-1 `uint16` PNGs on Modal (`54,706` RSNA + `26,988` NLBS) for `dinov2_rsna` (`~$1.50–$3.00`) and `resnet50_rsna` (`~$8.00–$12.00`, 1-epoch cost-gated).
   - Stage native `2944×1920` (GMIC, `~$4.00–$6.50`) and `2677×1942`/`2974×1748` (NYU, `~$4.00–$6.50`) on CPU containers before L40S NVMe-staged inference. Projected total: `~$17.50–$28.00` (inside `$30.00` cap).

---

## 5. Quality Assurance & Validation Checklist
- [x] NLBS `5,997 / 5,935 / 6,430 / 6,359 / 5,942` and `149 / 151` counts verified directly against `datasets/nlbs/*.csv`
- [x] NLBS and RSNA 4-view eligibility (`6,184 / 6,359` NLBS, `11,913 / 11,913` RSNA) computed programmatically
- [x] RSNA `splits_seed42.csv` site/patient/positive distributions verified (`0` overlap)
- [x] Local CE/TRUST checkpoints & `eval_val_calib_fp32.pt` inspected and CPU parity verified
- [x] Zero GPU jobs or paid Modal runs executed
- [x] Immutable Stage B/C reports and Paper 2 files preserved untouched

---

## 6. Remaining Blockers / Pre-Flight Gates
- **Gate 0.1:** Verify live USD balance on active Modal account (`hasinishrak2015` vs `hasinishrak74001`) prior to Phase 1.
- **Gate 0.2:** Build `studies/paper1/` scaffold and freeze deterministic RSNA `val_calib` 2-fold CSV + NLBS 4-view tie-break manifest on CPU.
- **Gate 0.3:** Pass `sample_data` parity & 200-unit L40S cost projection for GMIC/NYU, and 1-epoch L40S cost + `train_calib` AUROC gate for ResNet-50.

---

## 7. Git & Commit Tracking
- **Commit Rule Applied:** `1 file = 1 commit` under alternating identity (`Tahsin Raihan Robbani <tahsinraihanrobbani@gmail.com>`).
- **Files Created/Updated:** `reports/paper1/stage0_readiness_audit.md`, `research_logs/2026-11-21_paper1_stage0_evaluation_and_dataset_audit.md`, `research_logs/README.md`, `memory/state.md`, `memory/history.md`, `memory/index.md`.
