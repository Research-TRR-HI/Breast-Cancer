# Paper 1 — Stage 1 Model Readiness & Modal Benchmark Audit

**Date:** 2026-11-21  
**Study:** Paper 1 — Multi-Model External Validation of Breast Cancer Screening AI (`studies/paper1/`)  
**Modal Execution Profile:** `hasinishrak2015` (holds both `rsna-mammography-vol` and `nlbs-mammography-vol`; L40S app `ap-VqaH4fOfCP46dJKO9V9Gei`)  
**Decision:** **GO** (Conditional only on funding/credit capacity on `hasinishrak2015` and optional NVMe/CPU staging optimization before launching full runs)

---

## 1. Executive Summary

Evaluation Gate 0.2 and all four new model implementations (`GMIC`, `NYU Four-View CNN`, `DINOv2 ViT-S/14`, and `ResNet-50 IMAGENET1K_V2`) are verified end-to-end:

1. **Gate 0.2 Frozen & Audited:** Patient-disjoint 2-fold cross-fitting splits (`datasets/paper1/rsna_val_calib_2fold_seed42.csv`), label-blind 4-view RSNA/NLBS manifests (`6,184` common 4-view NLBS studies / `5,868` `split_cluster_id` clusters / `151` cancer-positive studies), and independent CE/TRUST RSNA source-test + common-cohort NLBS metrics are locked without modifying `reports/stage_b/*`, `reports/stage_c/*`, or `datasets/rsna/splits_seed42.csv`.
2. **Official Sources & Bitwise Hashes Verified:** Every repository commit, upstream weight SHA-256, license, strict parameter load, native preprocessing path, and official `sample_data` parity (`GMIC` and `NYU Four-View`) was reproduced verbatim locally and uploaded to `rsna-mammography-vol:/models/paper1/` on Modal (`$0` GPU upload).
3. **Real L40S Cloud Smoke & Throughput Benchmarks Passed (`ap-VqaH4fOfCP46dJKO9V9Gei`):** All four pipelines executed with `0` NaN/Inf values, exact verified checkpoints, and real NVIDIA L40S timings (`$2.3608/hr` L40S rate).
4. **Local Verification Battery:** `29/29` CPU unit tests passed cleanly across `tests/test_stage_b_smoke.py` (`20/20`), `tests/test_stage_c_quarantine_and_eval.py` (`3/3`), and `tests/test_paper1_models_and_gate02.py` (`6/6`).

---

## 2. Gate 0.2 Completion & Existing Baseline Reuse (`ce_b5`, `trust_b5`)

### 2.1 Frozen Manifest Checksums

| Manifest | Scope & Denominator | SHA-256 |
| :--- | :--- | :--- |
| `datasets/paper1/rsna_val_calib_2fold_seed42.csv` | `2,382` patients (`1,191` / `1,191`), `10,982` images (`97` pos; Site 1/2 `592/599` & `591/600`) | `ea0c291b6ee48cfae94cbd7725c09b58f0771939bd4efc41acb7e109b0215767` |
| `datasets/paper1/rsna_val_calib_4view_selection.csv` | `2,382` patients x `4` views = `9,528` rows (`1,333` multi-candidate slots resolved deterministically) | `90c420fc5f28d8e58b2f7a45f1d80fa64082df7658a875912306171d834ec679` |
| `datasets/paper1/nlbs_4view_study_eligibility.csv` | `6,359` unique NLBS studies (`6,184` eligible / `175` ineligible; `5,868` eligible clusters) | `34c12224ffecf60f688c7b6d18d16afe85e001c2239230d0c1415d97f7b88bb0` |
| `datasets/paper1/nlbs_4view_common_cohort_label_free.csv` | `24,736` label-free rows (`1,342` multi-candidate slots resolved deterministically) | `c5d8d8b1301f0dffb7a6a429f780571c5dc81dd844e98ab60ea7f069a7815aed` |

### 2.2 Independent RSNA Source-Test vs. Historical Full-`val_calib` Thresholds

*Strict separation notice:* Historical NLBS dismissal evaluations used full-`val_calib` Stage B AMP-float16 operating thresholds (`ce_b5`: `0.09619140625` @98% / `0.12445068359375` @95%; `trust_b5`: `0.1795654296875` @98% / `0.1954345703125` @95%). Independent RSNA source-test operating points use patient-disjoint 2-fold cross-fitting (`fold_a <-> fold_b`) inside `val_calib` (`reports/paper1/ce_trust_rsna_source_eval.{json,md}`):

| Model | Cohort | AUROC (95% CI) | AUPRC (95% CI) | @98% Sens / Missed | @98% Dismissal | @95% Sens / Missed | @95% Dismissal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ce_b5` | RSNA 2-fold OOF (`N=2,382`, `97` pos) | `0.7428` (`[0.6930, 0.7882]`) | `0.1064` (`[0.0785, 0.1504]`) | `98.97%` (`1/97`) | `11.67%` (`278`) | `89.69%` (`10/97`) | `29.09%` (`693`) |
| `trust_b5` | RSNA 2-fold OOF (`N=2,382`, `97` pos) | `0.7442` (`[0.6939, 0.7896]`) | `0.1072` (`[0.0794, 0.1500]`) | `98.97%` (`1/97`) | `12.80%` (`305`) | `89.69%` (`10/97`) | `29.43%` (`701`) |
| `ce_b5` | NLBS Common 4-View (`N=6,184`, `151` pos) | `0.7089` (`[0.6668, 0.7463]`) | `0.0548` (`[0.0418, 0.0764]`) | `90.07%` (`15/151`) | `30.37%` (`1,878`) | `80.13%` (`30/151`) | `49.66%` (`3,071`) |
| `trust_b5` | NLBS Common 4-View (`N=6,184`, `151` pos) | `0.7040` (`[0.6617, 0.7423]`) | `0.0539` (`[0.0409, 0.0747]`) | `84.77%` (`23/151`) | `39.57%` (`2,447`) | `80.79%` (`29/151`) | `49.16%` (`3,040`) |

Both Stage B/C baseline models exhibit severe external calibration collapse when common-cohort 4-view evaluation (`6,184` / `6,359` studies, retaining all `151/151` cancers) is isolated.

---

## 3. Provenance, Weight Verification & Official `sample_data` Parity

All external upstream repositories were cloned into git-ignored `external/` and weights were verified before uploading to `rsna-mammography-vol:/models/paper1/`:

| Model | Official Repository & Commit | Checkpoint & Exact SHA-256 | License | Preprocessing & Aggregation | Key-Load & Parity Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`gmic_pretrained`** | `https://github.com/nyukat/GMIC` (`594029efe6184bcca3a151fa1dd46c5fd620cb25`) | `sample_model_1.p` (`3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991`) | `AGPL-3.0` | Native `nyukat` ROI crop + optimal centers `2944 x 1920`, per-image z-score -> `malignant_pred`, exam `max()` | `weights_only=False`; `strict=False` with exact check `missing == []` & `unexpected == ["shared_rep_filter.weight"]`. `sample_data/0_L-CC` reproduces README `0.1356` benign / `0.0081` malignant. |
| **`nyu_fourview_pretrained`** | `https://github.com/nyukat/breast_cancer_classifier` (`6176ebea6950b0a953339593d2da417ab846dae0`) | `sample_image_model.p` (`9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014`) | `AGPL-3.0` | Native `nyukat` crop + optimal centers (`CC`: `2677 x 1942`, `MLO`: `2974 x 1748`), 10-epoch TTA (`seed=0`), exam `max(left_malignant, right_malignant)` | `strict=True` (0 missing/unexpected). Fixed `pandas 2.3` `groupby('label').mean()` bug via direct `0.5 * (cc_probs + mlo_probs)`. `sample_data/0_*` reproduces README `0.0091` left / `0.0179` right. |
| **`dinov2_rsna`** | `https://github.com/facebookresearch/dinov2` (`7764ea0f912e53c92e82eb78a2a1631e92725fc8`) | `dinov2_vits14_pretrain.pth` (`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`) | `Apache-2.0` | Reuse `1520 x 912` Stage-1 `uint16` PNGs -> pad-to-square `518 x 518` ImageNet-norm RGB -> frozen `384-d` CLS -> RSNA-trained head (`DINOv2CancerHead`) | `strict=True` (`22,056,576` params, `requires_grad=False`); outputs verified `[1, 384]`; CPU trainer (`scripts/train_paper1_dinov2_head.py`) unit-tested end-to-end. |
| **`resnet50_rsna`** | `https://github.com/pytorch/vision` (`ResNet50_Weights.IMAGENET1K_V2`) | `resnet50-11ad3fa6.pth` (`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`) | `BSD-3-Clause` | Reuse `1520 x 912` Stage-1 `uint16` PNGs -> bilinear `854 x 512` (5:3 aspect) + TRUST augs + ImageNet norm -> RSNA-trained head | `strict=True`; replaces `fc` with `Dropout(0.3) -> Linear(2048, 1)`; AMP forward/backward verified with non-null gradients. |

*Secondary GMIC ensemble weights (`sample_model_2..5.p`) were also hashed for optional sensitivity checks (`model_manifest.yaml` keeps `sample_model_1.p` as primary). Neither NYU model requires RSNA cancer-classifier training; both `dinov2_rsna` and `resnet50_rsna` train strictly on RSNA `train` and select epochs/configs strictly on `train_calib`.*

---

## 4. Measured L40S Smoke Tests & Cost Projections (`ap-VqaH4fOfCP46dJKO9V9Gei`)

All measurements below were collected inside a live `NVIDIA L40S` Modal container on profile `hasinishrak2015` (`$2.3608/hr`, pinned Python `3.12`, PyTorch `2.5.1`, `reports/paper1/l40s_smoke_and_benchmark.json`):

| Pipeline | Smoke Scope | Measured Timing & Throughput | Peak VRAM | Fallbacks / Finite Check | Projected Full Volume Work | Projected Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gmic_pretrained`** | `20` DICOMs (`RSNA` + `NLBS`) | CPU crop `24.079s` (`0.83 img/s`, 1-thread); GPU `0.285s` (`70.25 img/s`) | `0.292 GB` | `0` crop fallbacks; `all_scores_finite=true` | `37,970` images (`10,982` RSNA `val_calib` + `26,988` NLBS): `0.15` L40S hrs + `12.7` CPU hrs | **`$1.12`** *(w/ CPU staging)* |
| **`nyu_fourview_pretrained`** | `8` exams (`32` DICOMs, `10x` TTA) | CPU crop `32.703s`; GPU `12.968s` (`0.617 exams/s`) | `2.410 GB` | `2` crop fallbacks (`6.25%` views); `all_scores_finite=true` | `8,566` 4-view exams (`2,382` RSNA + `6,184` NLBS): `3.857` L40S hrs + `9.73` CPU hrs | **`$9.69`** *(w/ CPU staging)* |
| **`dinov2_rsna`** | `200` Stage-1 PNGs (`1520x912`) | Cold Volume IO `38.764s`; GPU extract `0.385s` (`519.88 img/s`) | `1.606 GB` | `0` failures; `all_probs_finite=true` | `81,694` images (`54,706` RSNA + `26,988` NLBS): `4.442` L40S hrs cold (`0.044` hrs GPU-only) | **`$10.99`** *(cold IO; `~$0.60` if NVMe-staged)* |
| **`resnet50_rsna`** | `256` images (`4 x 64`, `854x512` AMP) | Cold IO `52.428s`; GPU step `2.224s` (`115.1 img/s`), losses `[0.6937..0.6231]` | `26.277 GB` | `0` NaNs; `all_losses_finite=true` | `10` epochs (`600` steps/ep, NVMe-staged + `train_calib` eval): `0.979` L40S hrs | **`$3.31`** *(+ `~$0.35` infer)* |

**Key Engineering Insights from Benchmarks:**
1. **GMIC / NYU CPU-Bound Bottleneck:** Both NYU models are CPU-decode + `scipy` crop bound if executed inline on an L40S container. Decoupling sharded CPU crop-staging onto cheap CPU containers keeps L40S GPU utilization near peak (`$1.12` for GMIC, `$9.69` for NYU).
2. **NYU Crop Fallbacks:** Observed `2` fallback views out of `32` (`6.25%`) where `nyukat` largest-connected-component mask returned blank/undersized; `extract_nyu_cropped_view` fell back cleanly to the uncropped full mammogram without crashing or producing NaNs (`crop_fallback` counter will be audited across full inference).
3. **DINOv2 IO Optimization Opportunity:** Pure L40S feature extraction is `519.88 img/s` (`~2.6 min` total GPU compute for all `81,694` images). Current `scripts/run_paper1_dinov2_modal.py` streams PNGs over network Volume FUSE (`$10.99`); adding `/tmp` NVMe pre-copy (`ThreadPoolExecutor(16)`, already used in `run_paper1_resnet50_modal.py`) will cut DINOv2 cost to `~$0.60-$1.00`.
4. **Total Combined Baseline Cost Across All 4 New Models:** **`~$25.11`** worst-case unoptimized (`$1.12 + $9.69 + $10.99 + $3.31`), or **`~$14.70`** once DINOv2 adds NVMe staging — comfortably under a `$30` budget per profile and easily executable across your multi-profile Modal accounts (`hasinishrak2015` holds both volumes).

---

## 5. Copy-Pasteable Full Execution Commands *(DO NOT RUN UNTIL READY FOR STAGE 2)*

```powershell
# 1. GMIC (Pretrained Model 1) — Sharded RSNA val_calib & NLBS Inference (~$1.12)
$env:MODAL_PROFILE="hasinishrak2015"
modal run scripts/run_paper1_gmic_modal.py --split rsna_val_calib
modal run scripts/run_paper1_gmic_modal.py --split nlbs

# 2. NYU Four-View CNN (Image-Only) — Sharded RSNA val_calib & NLBS 4-View Inference (~$9.69)
$env:MODAL_PROFILE="hasinishrak2015"
modal run scripts/run_paper1_nyu_modal.py --split rsna_val_calib
modal run scripts/run_paper1_nyu_modal.py --split nlbs_4view

# 3. DINOv2 ViT-S/14 — Feature Extraction (Modal) + Head Training/Locking (Local CPU)
$env:MODAL_PROFILE="hasinishrak2015"
modal run scripts/run_paper1_dinov2_modal.py --stage extract
# After downloading rsna_embeddings_fp32.pt and nlbs_embeddings_fp32.pt locally:
python scripts/train_paper1_dinov2_head.py --rsna-emb <rsna_pt> --nlbs-emb <nlbs_pt> --out-dir reports/paper1/dinov2_head

# 4. ResNet-50 (IMAGENET1K_V2) — 10-Epoch RSNA Training & Frozen Inference (~$3.66)
$env:MODAL_PROFILE="hasinishrak2015"
modal run scripts/run_paper1_resnet50_modal.py --stage train --epochs 10 --seed 42
modal run scripts/run_paper1_resnet50_modal.py --stage infer --seed 42
```

---

## 6. GO / NO-GO Decision

**Decision: GO**

All Gate 0.2 cohorts, frozen patient-disjoint CV manifests, target-label quarantine checks, official upstream checkpoints, adapter parity tests (`29/29` CPU tests passing), and real NVIDIA L40S smoke benchmarks (`0` crashes, finite outputs across all 4 models) are verified. Paper 2 (`AnchorShift_Research_Context.md`) and Stage B/C CE & TRUST baselines remain untouched.
