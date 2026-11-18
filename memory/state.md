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
- [x] Stage B3.5 (User Execution - Pre-Training Blocker Verification & L40S Benchmarks): Strict smoke test PASSED on T4 (`ap-49qo1JiWn1u7NGCioPcRkT`, `852/852`). L40S sweep PASSED (`ap-mY4VfFocqIRVEnGIeV2CtD`: `highest=128`, `fastest=96`, `optimal=80` at `24.90 GB` / `37.55 img/s`). Fair Approach A vs. B benchmark PASSED (`ap-KidtVEoMzokd7fsFzpMeFo`): `0.00e+00` numerical diff, Approach A `181.84 hrs / $429.29` (infeasible) vs. Approach B `7.23 hrs / $17.06` (`25.16x` speedup, `429.6 MB` fp32 cache). Still need `python tests/test_stage_b_smoke.py` + optional CPU-decoupled extraction optimization before Stage B4.
- [ ] Stage B4 (User Execution - Full Training): Train CE-only baseline and TRUST model on RSNA (20 epochs on L40S GPU, gated on B3.5 benchmark), evaluate on `val_calib`, and freeze source operating thresholds for 95% and 98% recall targets.
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
- Caught silent `missing: 736, unexpected: 852` mismatch in Run 2 (`timm` vs Mammo-CLIP's `efficientnet_custom`). Replaced backbone with canonical `efficientnet_custom` (`852/852` keys, `strict=True`), aligned Mammo-CLIP mean/std standardization, and mounted `src/` directly on Modal scripts. Awaiting user re-run of T4 smoke test (`modal run scripts/smoke_test_stage_b_modal.py`).
