# Research Log: Paper 1 — Gate 0.2 Manifests & Four-Model Readiness Benchmarks

**Date**: 2026-11-21  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Benchmarking, Model Provenance & External Validation Protocol  
**Milestone Scope**: `studies/paper1/`, `datasets/paper1/`, `src/paper1/`, `scripts/paper1_*`, `reports/paper1/`  
**Status**: VERIFIED COMPLETE (STAGE 1 READINESS = GO)  

---

## 1. Research Objective & Context
- Prepare **Paper 1: Multi-Model External Validation of Breast Cancer Screening AI** (`GMIC`, `NYU Four-View CNN`, `DINOv2 ViT-S/14 + RSNA Head`, and `ResNet-50 IMAGENET1K_V2 + RSNA Fine-Tune`) alongside frozen Stage B/C baselines (`ce_b5` and `trust_b5`).
- Enforce complete scientific and file-level separation from Paper 2 (`AnchorShift_Research_Context.md`), historical Stage B/C reports (`reports/stage_b/*`, `reports/stage_c/*`), and canonical RSNA splits (`datasets/rsna/splits_seed42.csv`).

---

## 2. Datasets, Volumes & Manifest References
- **Modal Profile (`hasinishrak2015`)**: `rsna-mammography-vol` (`/rsna/raw`, `/cache/stage1_rsna_uint16`, `/cache/stage1_nlbs_uint16`, `/models/paper1/`) and `nlbs-mammography-vol` (`/nlbs/extracted`).
- **Frozen Gate 0.2 Manifests (`datasets/paper1/`)**:
  - `rsna_val_calib_2fold_seed42.csv` (`ea0c291b6ee48cfae94cbd7725c09b58f0771939bd4efc41acb7e109b0215767`): `2,382` patients (`1,191` / `1,191`), `10,982` images (`97` pos, Site 1/2 stratified).
  - `rsna_val_calib_4view_selection.csv` (`90c420fc5f28d8e58b2f7a45f1d80fa64082df7658a875912306171d834ec679`): `9,528` deterministic label-blind rows (`1,333` multi-candidate slots).
  - `nlbs_4view_study_eligibility.csv` (`34c12224ffecf60f688c7b6d18d16afe85e001c2239230d0c1415d97f7b88bb0`): `6,359` total studies -> `6,184` eligible 4-view (`5,868` `split_cluster_id` clusters) / `175` ineligible.
  - `nlbs_4view_common_cohort_label_free.csv` (`c5d8d8b1301f0dffb7a6a429f780571c5dc81dd844e98ab60ea7f069a7815aed`): `24,736` label-free rows (`1,342` multi-candidate slots).
- **Uploaded Checkpoints (`rsna-mammography-vol:/models/paper1/`)**:
  - `sample_model_1.p` (`3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991`)
  - `sample_image_model.p` (`9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014`)
  - `dinov2_vits14_pretrain.pth` (`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`)
  - `resnet50-11ad3fa6.pth` (`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`)

---

## 3. Forensic Evidence & Diagnostic Findings
1. **RSNA Source Evaluation Separation:** 2-fold patient-disjoint cross-fitting inside `val_calib` on CPU shows `ce_b5` AUROC `0.7428` (`@98%` sens `98.97%` `1/97` missed / `11.67%` dismissal) and `trust_b5` AUROC `0.7442` (`@98%` sens `98.97%` `1/97` missed / `12.80%` dismissal).
2. **Common 4-View NLBS Re-Aggregation:** Filtering immutable `reports/stage_c/locked_nlbs_predictions.csv` to the `6,184` 4-view eligible studies preserves all `151/151` cancers (`ce_b5` AUROC `0.7089`, `@98%` sens `90.07%` `15/151` missed / `30.37%` dismissal; `trust_b5` AUROC `0.7040`, `@98%` sens `84.77%` `23/151` missed / `39.57%` dismissal).
3. **Upstream Compatibility Fixes:**
   - Patched `torch.load(..., weights_only=False)` for `nyukat` checkpoints; verified `GMIC` `sample_model_1.p` single unused key `shared_rep_filter.weight`.
   - Bypassed `pandas 2.3.3` `groupby("label").mean()` crash in `nyukat/breast_cancer_classifier` via explicit `0.5 * (cc_probs + mlo_probs)`.
   - Handled missing `h5py` locally without touching the pinned Stage B/C environment; added `"imageio==2.37.0"` and `"tqdm==4.67.1"` inside Paper 1 Modal container specs.

---

## 4. Implementation Details & Exact Changes
- **Protocol & Configs:** `studies/paper1/{README.md, protocol_v1.md, model_manifest.yaml, eval_config.yaml, analysis_plan.md}`.
- **Adapters (`src/paper1/`):** `gmic_adapter.py`, `nyu_fourview_adapter.py`, `dinov2_model.py`, `resnet50_model.py`.
- **Gate 0.2 & Pipeline Scripts (`scripts/`):** `prepare_paper1_gate02.py`, `paper1_modal_smoke_and_bench.py`, `run_paper1_gmic_modal.py`, `run_paper1_nyu_modal.py`, `run_paper1_dinov2_modal.py`, `run_paper1_resnet50_modal.py`, `train_paper1_dinov2_head.py`.
- **Unit Tests:** `tests/test_paper1_models_and_gate02.py`.

---

## 5. Quality Assurance, Unit Tests & Empirical Smoke Tests
- **Local CPU Test Suites (all executed after final edit):**
  - `python tests/test_stage_b_smoke.py`: `20/20 OK` (`4.299s`).
  - `python tests/test_stage_c_quarantine_and_eval.py`: `3/3 OK` (`0.114s`).
  - `python tests/test_paper1_models_and_gate02.py`: `6/6 OK` (`30.431s`) — includes official `sample_data` parity (`GMIC` `0.1356/0.0081`, `NYU` `0.0091/0.0179`), frozen DINOv2 + CPU head training/locking end-to-end smoke, and ResNet-50 backward pass.
- **Modal L40S Benchmark (`ap-VqaH4fOfCP46dJKO9V9Gei` on `hasinishrak2015`, `reports/paper1/l40s_smoke_and_benchmark.json`):**
  - `gmic_pretrained`: `20` images, `0` fallbacks, `0.292 GB` VRAM, GPU `70.25 img/s`, projected full cost `$1.12`.
  - `nyu_fourview_pretrained`: `8` exams (`10x` TTA), `2` crop fallbacks, `2.410 GB` VRAM, GPU `0.617 exams/s`, projected full cost `$9.69`.
  - `dinov2_rsna`: `200` PNGs, `1.606 GB` VRAM, GPU `519.88 img/s`, cold IO projected `$10.99` (`~$0.60` with NVMe pre-stage).
  - `resnet50_rsna`: `256` images (`4 x 64`), `26.277 GB` VRAM, GPU `115.1 img/s`, decreasing losses `[0.6937, 0.6936, 0.6499, 0.6231]`, projected 10-epoch NVMe training `$3.31`.
- **Checklist:**
  - [x] Official repository commits and weight SHA-256 hashes verified
  - [x] Official `sample_data` predictions reproduced (`GMIC` & `NYU Four-View`)
  - [x] Zero patient leakage & strict target-label quarantine verified
  - [x] Live L40S smoke tests passed with `0` crashes / `all_*_finite = true`
  - [x] Paper 2 & historical Stage B/C outputs completely preserved

---

## 6. Remaining Limitations & Open Issues
- `nyu_fourview_pretrained` showed `2/32` view crop fallbacks (`6.25%`) on the 8-exam smoke batch; monitor total fallback rate during full inference.
- `run_paper1_dinov2_modal.py` currently reads Stage-1 PNGs directly from Modal Volume FUSE (`$10.99` cold projection); optional `/tmp` NVMe pre-copy cuts runtime cost by ~90%.
- Full-scale RSNA/NLBS training and inference runs remain intentionally **paused** pending execution approval.

---

## 7. Git & Commit Tracking
- Identity switched to **Account B — Hasin Ishrak (`hasinishrak2015@gmail.com`)** alternating from prior Stage 0 commits (`Tahsin Raihan Robbani`), committed **1 file per commit**.
