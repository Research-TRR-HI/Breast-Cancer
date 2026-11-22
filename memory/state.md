# state.md — Unified Two-Paper Workspace State

**Last Updated:** 2026-11-22  
**Verified Repo Head (Prior):** `24a7013` (`main`)  
**Current Decision Across Both Papers:** Paper 1 **GMIC Step 2 Full L40S Inference = COMPLETE & SHA-256 LOCKED** (`batch_size=1`, `37,970` unique images = `10,982` RSNA / `2,382` pts + `26,988` NLBS / `6,359` studies, settled Modal spend **`$1.8101 USD <= $2.00`**, zero labels/thresholds accessed; next: NYU Four-View pipeline); Paper 2 AnchorShift Stage D pending (`MODIFY / INVESTIGATE`)

---

## 1. Shared Completed Foundation (Stages A -> B -> C — Preserved Immutable)
- [x] Smoke-test RSNA Kaggle -> Modal Volume (`rsna-mammography-vol`) download via `scripts/download_rsna_modal.py`.
- [x] Stage A1 (Download): Downloaded RSNA (`rsna-mammography-vol:/rsna/raw`) and NLBS (`nlbs-mammography-vol:/nlbs/raw`) on Modal (`hasinishrak2015`).
- [x] Stage A2 (Complete Dataset Audit): Audited both volumes via `scripts/audit_datasets_modal.py` (`cpu=1.0, memory=2048`), generated `datasets/rsna/manifest.csv` (54,706 rows) & `datasets/nlbs/manifest.csv` (26,988 rows), and documented findings in `reports/dataset_audit.md`.
- [x] Stage A3 (NLBS Extraction & Exam Manifest): Verified 8/8 FRDR SHA-256 checksums, extracted all 3 `.7z` archives to `/nlbs_vol/nlbs/extracted/` (raw untouched), verified 26,988/26,988 DICOMs, built `datasets/nlbs/exam_manifest.csv` (`6,430` `(case_id, StudyInstanceUID)` exams, `6,359` unique `StudyInstanceUID`s, `5,942` connected `split_cluster_id`s).
- [x] Stage B1 (Code & Implementation): Implemented modular TRUST & CE-only architecture (`src/models/`), preprocessing (`src/data/`), patient-level RSNA splits (`datasets/rsna/splits_seed42.csv`), Mammo-CLIP loader (`852/852` pinned keys), calibration utilities, local unit test suite, and Modal scripts.
- [x] Research Logs Architecture: Initialized `research_logs/` (`README.md`, `TEMPLATE.md`, and dated research logs for Stages A, A3, B1–B4, and C).
- [x] Stage B2 (Mammo-CLIP Download): Checkpoint `b5-model-best-epoch-7.tar` (1.65 GB, SHA-256 `0baccb0c...6fd02c`) verified on `rsna-mammography-vol:/models/mammo_clip/`.
- [x] Stage B3–B3.7 (GPU Smoke Tests, Paper Traceability & Fixes A–I): Verified strict T4 smoke test (`852/852`), L40S sweep (`optimal=80`), aligned code verbatim to `arXiv:2609.00300v1` (`reports/trust_paper_implementation_traceability.md`), implemented Fixes A–I, passed **20/20** CPU tests (`reports/final_pre_training_readiness_audit.md`).
- [x] Stage B3.8 (Cloud Gates 1–4 + `hasinishrak74001` Migration): PASSED `--audit-real` (`16/16` DICOMs, `fallback_count=0`), built Stage-1 `1520x912` lossless `uint16` PNGs (`65.9 GiB`, `54,706` imgs), PASSED 1,200-img L40S AMP parity pilot (`max_prob_diff=2.35e-04`), PASSED 5-batch extraction benchmark (`35.13 img/s`), migrated Stage-1 volume to `hasinishrak74001`.
- [x] Stage B4 (RSNA 1M Cache & 20-Epoch CE / TRUST Training on `hasinishrak74001`): Extracted & verified `1,000,000`-occurrence aug-aware feature cache (`640` shards, SHA-256 `6c34a8c0...`). Completed 20 epochs of `ce` (`rsna_ce_eval_results.json`: RSNA `val_calib` `95%=22.50%`, `98%=14.48%` dismissal) and `trust` (`rsna_trust_eval_results.json`: `95%=20.70%`, `98%=17.17%` dismissal, final `tau_t=0.25610`). Checkpoints archived locally in `models/checkpoints/stage_b_rsna/`.
- [x] Stage C (Strict Zero-Target-Label NLBS Baseline on `hasinishrak2015`, `ap-rxM08NSdg43qLOjpUZ4jkR`): Decoded & cropped all `26,988/26,988` NLBS images into label-free `1520x912` `uint16` PNG cache (`0` failures, `0` fallbacks), hash-locked `6,359` `StudyInstanceUID` exam scores (`sha256: 526434d0...`) prior to outcome join, and computed `B=2,000` cluster-bootstrap CIs (`5,942` clusters, `151` cancer exams):
  - **CE-only (`98%` RSNA thr `0.096191`):** NLBS sensitivity **`90.07%`** (`136/151`, `15` missed), dismissal **`30.49%`**, AUROC `0.7101` -> **FAILS 98% target**.
  - **TRUST (`98%` RSNA thr `0.179565`):** NLBS sensitivity **`84.77%`** (`128/151`, `23` missed), dismissal **`39.77%`**, AUROC `0.7052` -> **FAILS 98% target** (`-5.30%` vs CE, 95% CI `[-9.15%, -1.92%]`).
  - **95% thresholds:** CE `80.13%` sens (`30` missed, `49.90%` dis); TRUST `80.79%` sens (`29` missed, `49.43%` dis) -> **BOTH FAIL**.
  - Immutable reports saved in `reports/stage_c/nlbs_zero_label_results.{json,md}` and `research_logs/2026-11-20_stage_c_nlbs_zero_label_evaluation.md`.

---

## 2. PAPER 1 — Multi-Model External Validation Benchmark (`Paper1_External_Validation_Research_Context.md`)
- **Scientific Role:** Evaluation & failure-mechanism benchmark across **6 prespecified models** (`ce_b5`, `trust_b5`, `gmic_pretrained`, `nyu_fourview_pretrained`, `dinov2_rsna`, `resnet50_rsna`) under source-frozen RSNA `98%`/`95%` thresholds on NLBS (NO new algorithm; `MODIFY / INVESTIGATE`).
- **Budget & Compute Guardrail:** Max **$30 USD** total *additional* Modal spend on `NVIDIA L40S`. Mandatory account/credit verification + native preprocessing parity + 200–500-img cost projection gate before every full run.
- **Model Status Matrix:**
  - [x] `ce_b5` & `trust_b5`: Reused cleanly via 2-fold OOF RSNA source test (`0.7428` / `0.7442` AUROC; `@98%` `98.97%` sens) & common 4-view NLBS (`N=6,184`, `151` pos: `ce_b5` `0.7089`, `trust_b5` `0.7040` AUROC) in `reports/paper1/ce_trust_{rsna_source_eval,nlbs_common_4view}.{json,md}`.
  - [x] `gmic_pretrained` (`nyukat/GMIC` `594029efe6184bcca3a151fa1dd46c5fd620cb25`, `sample_model_1.p` `3b792e6d...`): **STEP 1 & STEP 2 COMPLETE & SHA-256 LOCKED** (`batch_size=1`, `10,982` RSNA / `2,382` pts [`80` fb] + `26,988` NLBS / `6,359` studies [`0` fb] in `reports/paper1/predictions_locked/`, settled L40S spend **`$1.8101 USD`**; threshold calibration deferred cleanly).
  - [x] `nyu_fourview_pretrained` (`nyukat/breast_cancer_classifier` `6176ebea6950b0a953339593d2da417ab846dae0`, `sample_image_model.p` `9a1d97f9...`): Parity reproduced (`0.0091/0.0179`), `pandas 2.3` bug fixed in adapter, Modal pipeline & L40S benchmark passed (`2/32` view fallbacks, `2.410 GB` VRAM, `$9.69` projected full run).
  - [x] `dinov2_rsna` (`facebookresearch/dinov2` `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, `dinov2_vits14_pretrain.pth` `b938bf1b...`): Frozen encoder + Stage-1 extraction + `train_paper1_dinov2_head.py` verified; L40S benchmark passed (`1.606 GB` VRAM, `519.88 img/s` GPU compute; `$10.99` cold IO / `~$0.60` NVMe-staged).
  - [x] `resnet50_rsna` (`torchvision` `ResNet50_Weights.IMAGENET1K_V2`, `resnet50-11ad3fa6.pth` `11ad3fa6...`): Fine-tuning & frozen-inference pipeline verified; L40S benchmark passed (`26.277 GB` VRAM, `115.1 img/s` AMP train, `$3.31` 10-epoch training projection).
- **Mandatory Gates & Remaining Work (`reports/paper1/stage1_model_readiness.md` — GO):**
  - [x] **Stage 0 Audit — Reconciliation, Splits, Provenance & Budget:** Complete (`reports/paper1/stage0_readiness_audit.md`).
  - [x] **Gate 0.2 — Study Protocol, Frozen Splits & CPU Baseline Source Eval:** Created `studies/paper1/*`, `datasets/paper1/*` (`rsna_val_calib_2fold_seed42.csv`, `rsna_val_calib_4view_selection.csv`, `nlbs_4view_study_eligibility.csv`, `nlbs_4view_common_cohort_label_free.csv`), and `reports/paper1/ce_trust_{rsna_source_eval,nlbs_common_4view}.{json,md}`.
  - [x] **Gate 0.3 — Four-Model Official Verification & L40S Smoke Benchmarks:** All 4 checkpoints verified & uploaded to `hasinishrak2015` `rsna-mammography-vol:/models/paper1/`; live L40S benchmarks (`ap-acCim6bfGllCdJjTKtPDv6`, `ap-vMdXP2h5eFu2wqzAOGyAYs`) & `32/32` CPU tests PASSED.
  - [x] **Stage 1.5 — Final Critical Corrections (`reports/paper1/stage1_final_corrections.md` — GO):** Fixed ResNet-50 `scaler.unscale_` + full RNG/sampler resume (`test_7`); added DINOv2 NVMe staging + resumable 4k chunks (`74.2 img/s`, `~$0.72` full); diagnosed NYU `2/32` fallbacks (`patient_id=71 MONOCHROME1`), removed `cv2.resize` distortion, added fallback sensitivity audit (`test_9`); built & tested complete 6-model paired eval & statistical decomposition suite (`src/paper1/six_model_eval.py`, `scripts/evaluate_paper1_six_models.py`, `test_8`).
  - [x] **GMIC Step 1 — CPU-Only Preprocessing & Volume Cache (`hasinishrak2015`, `ap-i9HV9RJvsFrAoyBBZ7mlk5`):** Processed & verified `10,982` RSNA `val_calib` (`80` fallbacks, `62.06 GB`, `$0.8280`) + `26,988` NLBS (`0` fallbacks, `97.29 GB`, `$0.7564`) = **`37,970` `(2944, 1920)` `uint16` PNGs** at `0.0` parity diff for **`$1.5844` build spend** (`0` GPU, `reports/paper1/gmic_step1_preprocessing_report.md`).
  - [x] **GMIC Step 2 GPU Pilot (`hasinishrak2015`, `ap-YYXH9MfyVOBIXfb5IPctzQ`):** Verified NVMe-staged L40S `batch=8` inference across `200` RSNA + `200` NLBS cached PNGs (`32.52 img/s` e2e, `1.511 GB` VRAM, `sample_data` & raw parity verified), audited all `80` RSNA fallbacks (`36` pts, `51` M1/`29` M2, `0` corrupt) & `5` supplemental views, and measured **`$0.00807` pilot** -> **`$0.771` full projection** (within `$2` budget, `reports/paper1/gmic_step2_gpu_pilot_report.md`).
  - [x] **GMIC Step 2 Full L40S Inference (`hasinishrak2015`, `ap-YBKnb6U` / `ap-Gxxgc1V` / `ap-xlcdupT`):** Completed `batch_size=1` 4-shard inference per split reusing Step-1 cache with strict provenance/resume validation (`--batch-size 1`, `sample_model_1.p`, `paper1_gmic_uint16_v1`, item IDs) & async CPU prefetch; verified & SHA-256 locked all `37,970` images + `2,382`/`6,359`/`6,184` coverage for **`$1.8101 USD`** settled spend (`reports/paper1/gmic_step2_full_inference_report.md`; zero labels/thresholds accessed).
  - [ ] **Stage 2 — Execute Remaining Paper 1 Runs:** Run `NYU Four-View -> DINOv2 -> ResNet-50 -> RSNA threshold calib -> evaluate_paper1_six_models.py` on `hasinishrak2015`.
  - [ ] **Blocker 3 (Manuscript Gate) — Adversarial Literature Matrix:** Formal *MEDINFO 2025/2026 Meta-Repository* & *Sensors (June 2026)* comparison before manuscript drafting.

---

## 3. PAPER 2 — AnchorShift (`AnchorShift_Research_Context.md`)
- **Scientific Role:** Test whether unlabeled-target adaptation (`UPA`, Roschewitz et al. 2023) fixes frozen-threshold sensitivity failure on NLBS, compare against TRUST's `20%` target-labeled calibration (`Protocol 3` upper reference), and develop a novel AnchorShift method **ONLY if** a genuine unresolved failure remains (`MODIFY / INVESTIGATE`).
- **Pending Tasks:**
  - [ ] **Stage D1 — UPA (2023) Unlabeled Adaptation Baseline (`Protocol 2`):** Split NLBS cleanly by `split_cluster_id` into unlabeled adaptation vs held-out evaluation; fit UPA CDF score alignment against RSNA calibration scores (no NLBS cancer outcomes used) for CE and TRUST.
  - [ ] **Stage D2 — Target-Label 20/80 Calibration Reference (`Protocol 3`):** Compute oracle target-calibrated thresholds on matched NLBS split using our exact Stage-B checkpoints (isolating zero-label threshold drift vs backbone `AUROC ~0.71`).
  - [ ] **Stage E — Method Decision Gate:** Decide `PROCEED` (new AnchorShift algorithm justified) / `MODIFY` / `MERGE into unified paper with Paper 1` / `ABANDON` based on observed UPA + Paper 1 evidence.

---

## 4. Infrastructure, Budget & Active Modal Volume Notes
- **Modal Volumes:**
  - `hasinishrak2015`: `nlbs-mammography-vol` (`26,988` raw `.dcm`s in `/nlbs/extracted/`), `rsna-mammography-vol` (`54,706` RSNA Stage-1 `1520x912` PNGs + `b5-model-best-epoch-7.tar` + Stage-B Epoch-20 checkpoints + newly built **`26,988` NLBS `1520x912` `uint16` PNGs** at `/cache/stage1_nlbs_uint16/png_1520x912/`).
  - `hasinishrak74001`: `rsna-mammography-vol` (`65.9 GiB` RSNA `1520x912` PNGs + `640`-shard `1M`-occurrence cache + Stage-B Epoch-20 checkpoints/results).
- **Mandatory Before Any Next Paid Run:** Check real remaining balance/credits on active Modal profile (`hasinishrak2015` vs `hasinishrak74001`), verify `.pt`/volume locations, and enforce the **$30 additional Modal cap**.

---

## 5. Preserved Historical Notes (`Convo ac9601be-6e61-4ccd-aa35-83938eac42d6`, 2026-11-17)
- Verified Mammo-CLIP checkpoint `shawn24/Mammo-CLIP/Pre-trained-checkpoints/b5-model-best-epoch-7.tar` (CC BY-NC-SA 4.0, 1.65 GB).
- Deterministic RSNA `splits_seed42.csv`: `8,339` train (4.08% cancer), `1,192` train_calib (4.11%), `2,382` val_calib (4.07%) with `0` patient leakage.
- Resolved NLBS `149` positive cases vs `151` exams (`p_4334` and `p_4461` repeat visits) and `71` shared `StudyInstanceUID`s via `split_cluster_id` (`5,942` clusters).
- Full reproduction details recorded in `reports/trust_reproduction_plan.md`.
