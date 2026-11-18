# state.md — Current Workspace State

## Active Goals & Todo
- [x] Smoke-test RSNA Kaggle -> Modal Volume (`rsna-mammography-vol`) download via `scripts/download_rsna_modal.py`.
- [x] Stage A1 (Download): Downloaded RSNA (`rsna-mammography-vol:/rsna/raw`) and NLBS (`nlbs-mammography-vol:/nlbs/raw`) on Modal (`hasinishrak2015`).
- [x] Stage A2 (Complete Dataset Audit): Audited both volumes via `scripts/audit_datasets_modal.py` (`cpu=1.0, memory=2048`), generated `datasets/rsna/manifest.csv` (54,706 rows) & `datasets/nlbs/manifest.csv` (26,988 rows), and documented findings in `reports/dataset_audit.md`.
- [x] Stage A3 (NLBS Extraction & Exam Manifest): Verified 8/8 FRDR SHA-256 checksums, extracted all 3 `.7z` archives to `/nlbs_vol/nlbs/extracted/` (raw untouched), verified 26,988/26,988 DICOMs, built `datasets/nlbs/exam_manifest.csv` (`6,430` `(case_id, StudyInstanceUID)` exams, `6,359` unique `StudyInstanceUID`s, `5,942` connected `split_cluster_id`s).
- [x] Stage B1 (Code & Implementation): Implemented modular TRUST & CE-only architecture (`src/models/`), preprocessing (`src/data/`), patient-level RSNA splits (`datasets/rsna/splits_seed42.csv`), Mammo-CLIP loader, calibration utilities, local unit test suite (100% PASS), and Modal GPU smoke/training scripts.
- [x] Research Logs Architecture: Initialized `research_logs/` mirroring `D:\Research BanglaBBQ\research_logs` (`README.md`, `TEMPLATE.md`, and 3 retroactive deep-dive research logs for Stage A, Stage A3, and Stage B).
- [x] Stage B2 (User Execution - Mammo-CLIP Download): Checkpoint `b5-model-best-epoch-7.tar` (1.65 GB) downloaded and verified in `rsna-mammography-vol:/models/mammo_clip/`.
- [x] Stage B3 (User Execution - GPU Smoke Test `strict=True`): PASSED on Tesla T4 (`ap-YZU56SuJcHFBV9LR17YcnY`): `852/852` matched keys (`missing=0, unexpected=0`), `checkpoint_loaded_strict: True`, `encoder_frozen_verified: True`, 0.85 GB peak VRAM, `Total=0.7483` (`BCE=0.6601`, `Focal=0.0406`, `Dismissal=0.0476`).
- [x] Stage B3.5 (User Execution - Pre-Training Blocker Verification & L40S Benchmarks): Strict smoke test PASSED on T4 (`ap-49qo1JiWn1u7NGCioPcRkT`, `852/852`). L40S sweep PASSED (`ap-mY4VfFocqIRVEnGIeV2CtD`: `highest=128`, `fastest=96`, `optimal=80` at `24.90 GB` / `37.55 img/s`). Historical 1-feature-per-image L40S inference/encoding benchmark completed (`ap-KidtVEoMzokd7fsFzpMeFo`).
- [x] Stage B3.6 (Complete Pre-Training Correction & Local Validation): Audited directly against `arXiv:2609.00300v1` (`reports/trust_paper_implementation_traceability.md`); fixed TRUST losses (`m=0.10`, `lambda_dismiss=0.05`, `tau_1=0.05`, unweighted focal `gamma=2.0`), CP95 dynamic `tau_t` + CP99 edge cases, full Sec. 3.2 DICOM LUT/crop/mirror/augmentation preprocessing, `paper_independent` 20:60 sampler, final-epoch checkpoint selection (no `val_calib` double-use), pinned Mammo-CLIP SHA-256 + `requirements-lock.txt`, sharded augmentation-aware caching, complete RNG save/resume, and **13/13 local CPU tests PASSED** (`python tests/test_stage_b_smoke.py` & `unittest discover`). Status: **CONDITIONAL GO** (`reports/pre_training_go_no_go.md`).
- [ ] Stage B3.7 (User Execution - Cloud Verification Gates): Re-run `modal run scripts/smoke_test_stage_b_modal.py` (with updated preprocessing) and execute the mandatory 1,000-image L40S online-vs-cached Mammo-CLIP parity pilot (`modal run scripts/pilot_parity_modal.py --epochs 20 --steps-per-epoch 12 --precision both --gpu L40S`) before full extraction. (Also recommended: add CPU Stage-1 pre-crop cache to avoid 992k raw DICOM decodes during full extraction.)
- [ ] Stage B4 (User Execution - Full Training — NOT STARTED): Extract verified augmentation-aware RSNA cache (`--extract-cache`), train CE-only and TRUST (20 epochs each, final-epoch checkpoint) on RSNA, and calibrate frozen 95% and 98% recall operating thresholds on `val_calib`.
- [ ] Stage C (Strict Zero-Label Test): Freeze weights, preprocessing, aggregation, and source threshold; run inference on NLBS and hash-lock predictions before revealing labels.
- [ ] Stage D (Adaptation Comparison): Run UPA (2023) using unlabeled NLBS adaptation subset (patient-separated from evaluation) and compare against frozen-threshold results + TRUST target-calibrated reference.
- [ ] Decide PROCEED / MODIFY / ABANDON based on the go/no-go table.

## Last Session (Convo ac9601be-6e61-4ccd-aa35-83938eac42d6)
- Stage B Preparation Complete:
  - Investigated TRUST (2026, arXiv:2609.00300) Methods 3.1–3.11; confirmed no official TRUST code/weights exist; verified foundation checkpoint is `shawn24/Mammo-CLIP/Pre-trained-checkpoints/b5-model-best-epoch-7.tar` (CC BY-NC-SA 4.0, 1.65 GB).
  - Built modular PyTorch codebase (`src/models/`, `src/data/`, `src/training/`).
  - Generated deterministic RSNA patient-level splits (`datasets/rsna/splits_seed42.csv`): 8,339 train (4.08% cancer), 1,192 train_calib (4.11%), 2,382 val_calib (4.07%) with 0 patient leakage.
  - Investigated NLBS 149 positive cases vs 151 exams (`p_4334` and `p_4461` each have 2 distinct positive exams); confirmed `split_cluster_id` (`5,942` clusters) solves the 71 shared `StudyInstanceUID`s.
  - Created local unit test suite `tests/test_stage_b_smoke.py` (5/5 tests PASS).
  - Created Modal GPU smoke test (`scripts/smoke_test_stage_b_modal.py`) and training script (`scripts/train_trust_modal.py`).
  - Documented full reproduction details in `reports/trust_reproduction_plan.md`.

## Current Blockers & Notes
- **CONDITIONAL GO (`reports/pre_training_go_no_go.md`):** All code corrections against `arXiv:2609.00300v1` and 13/13 local CPU tests (`tests/test_stage_b_smoke.py`) PASS. Cloud T4 smoke test re-run and L40S 1,000-image online-vs-cached Mammo-CLIP parity pilot (`scripts/pilot_parity_modal.py`) remain pending manual execution before full training. Historical `$17.06` / `429.6 MB` projection was for unaugmented 1-feature-per-image caching and must not be reused for stochastic-augmentation 20-epoch training.
