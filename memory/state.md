# state.md — Unified Two-Paper Workspace State

**Last Updated:** 2026-11-21  
**Verified Repo Head (Prior):** `ba3205b` (`main`)  
**Current Decision Across Both Papers:** `MODIFY / INVESTIGATE` (Paper 1 Stage 0 Audit: **CONDITIONAL GO** pending pre-flight gates 0.1–0.3; Novelty and Q1 readiness NOT confirmed for either paper yet)

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
  - [x] `ce_b5` & `trust_b5`: Historic Stage B/C baseline complete (immutable; note RSNA `val_calib` double-use caveat).
  - [ ] `gmic_pretrained` (Shen et al. 2021, `AGPL-3.0`): Pretrained lesion-aware ResNet-18 (`2944x1920` native cropped 16-bit PNGs) — NOT STARTED.
  - [ ] `nyu_fourview_pretrained` (Wu et al. 2019/2020, `AGPL-3.0`): Pretrained image-only 4-view CNN (`2677x1942` CC / `2974x1748` MLO) — NOT STARTED.
  - [ ] `dinov2_rsna` (`Apache-2.0` ViT-S/14 frozen probe + RSNA head) — NOT STARTED.
  - [ ] `resnet50_rsna` (ImageNet ResNet-50 RSNA fine-tune, cost-gated feasibility) — NOT STARTED.
- **Mandatory Pending Blockers & Stage 0 Status (`reports/paper1/stage0_readiness_audit.md` — CONDITIONAL GO, Zero Paid GPU Runs Yet):**
  - [x] **Stage 0 Audit — Reconcile NLBS Cohort Unit:** Reconciled `5,997` folders / `5,935` descriptor unique / `6,430` pairs / `6,359` unique `StudyInstanceUID` / `5,942` `split_cluster_id` / `149` pos clusters vs `151` pos exams (`p_4334`, `p_4461`); verified **`6,184 / 6,359` (`97.25%`)** 4-view studies (**`151 / 151` cancers**) and `935` multi-view ties (`reports/paper1/stage0_readiness_audit.md`).
  - [x] **Stage 0 Audit — RSNA Calibration vs. Independent Source Test Design:** Resolved `val_calib` double-use via zero-retrain patient-disjoint 50/50 split (`source_op_calib` `1,191` pts / `49` pos vs `source_heldout_test` `1,191` pts / `48` pos) + 2-fold OOF (`97` pos), reusing `eval_val_calib_fp32.pt` & `locked_nlbs_predictions.csv` at `$0` GPU cost (`<= 3.55e-5` CPU FP32 vs L40S AMP parity).
  - [x] **Stage 0 Audit — Provenance, Overlap, Caches & Budget Plan:** Audited `Mammo-CLIP` (`UPMC`, `CC BY-NC-SA 4.0`), `GMIC` & `NYU` (`NYU 2010–2017`, `AGPL-3.0`), `DINOv2` (`LVD-142M`, `Apache-2.0`), `ResNet-50` (`ImageNet-1K`, `BSD-3-Clause`). Cut GMIC/NYU inference scope by `53%` (RSNA `val_calib` + NLBS only) and planned `1520x912` Stage-1 PNG reuse for DINOv2/ResNet-50 (`~$17.50–$28.00` projected within `$30` cap).
  - [ ] **Gate 0.1 — Modal Credit Balance Check:** Verify exact USD credit balance on `hasinishrak2015` (and `hasinishrak74001`) before any GPU launch.
  - [ ] **Gate 0.2 — CPU Scaffold & Frozen Split CSVs:** Create `studies/paper1/` (`README.md`, `protocol_v1.md`, `model_manifest.yaml`, `eval_config.yaml`, `analysis_plan.md`), freeze deterministic RSNA `val_calib` 2-fold CSV + NLBS 4-view tie-break manifest, and compute CPU CE/TRUST 2-fold source metrics.
  - [ ] **Gate 0.3 — Per-Model Parity & Throughput Pilots:** Pass `sample_data` parity + 200-unit L40S cost projection for GMIC/NYU and 1-epoch L40S cost + `train_calib` AUROC gate for ResNet-50 prior to full runs.
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
