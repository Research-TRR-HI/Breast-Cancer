# Paper 1 Experiment 2 — Phase 1 Readiness & Audit Report (Zero-Cost Wave 1)

**Date:** 2026-11-24  
**Branch:** `paper1-exp2-phase1`  
**Modal Target Profile:** `tigerwood693` (explicit `MODAL_PROFILE=tigerwood693`; `modal profile activate` never called)  
**Wave 1 Cloud Spend:** `$0.00 USD` (read-only Modal inspection, free HTTP/GitHub/HF metadata audit, local CPU tests only)  
**Overall Verdict:** **CONDITIONAL GO (5/6 Original Candidates Verified + Vetted Replacement `ianpan/mammoscreen` Ready; `frcnn_cad` Formally Disqualified)**

---

## 1. Executive Summary & 24-Item Readiness Checklist

| # | Audit Item | Status | Verified Finding |
|---:|---|---|---|
| 1 | **Phase 1 Scope & Spend Gate** | **PASS** | Zero paid GPU pilots, zero volume transfers, and zero large downloads executed in Wave 1. |
| 2 | **Git & Artifact Isolation** | **PASS** | Working solely on `paper1-exp2-phase1`; `main`, `memory/`, `studies/paper1/protocol_v1.md`, and locked Stage B/C/6-model artifacts untouched. |
| 3 | **`tigerwood693` Modal Billing & Apps** | **PASS** | `0` running/deployed apps; Nov 2026 metered `$12.51` (`$4.23` credits drawn, `$8.24` free storage offset), `$0.00` billed, **`$25.77` credits remaining** (resets `2026-12-01`). |
| 4 | **`tigerwood693` Storage & DINOv2 Preservation** | **PASS** | `9` volumes totaling **`224.8539 GiB`** (`410,266` files; `$0.00/day`, `799.15 GiB` below `1,024 GiB` free tier). `rsna-mammography-vol` (`102.2373 GiB`, `81,806` files) intact with `/cache/stage1_nlbs_uint16` (`35.94 GiB`) & `/cache/paper1_dinov2_vits14` (`0.24 GiB`). |
| 5 | **`hasinishrak2015` Read-Only Source Check** | **PASS** | `nlbs-mammography-vol:/nlbs/extracted` (`26,988` files, `324.4480 GiB`) and `rsna-mammography-vol:/cache/paper1_gmic_uint16_v1/nlbs` (`27,007` files, `97.3070 GiB`) verified intact (`$5.97` credits remaining). |
| 6 | **Model 1 (`gmic_pretrained`)** | **ELIGIBLE (LOCKED REUSE)** | `nyukat/GMIC` (`594029ef...`), `sample_model_1.p` (`3b792e6d...`). Reuse `reports/paper1/predictions_locked/gmic_nlbs_case_scores.csv` (`94521810...`, `6,359` studies, `0` fallbacks) at `$0.00`. |
| 7 | **Model 2 (`nyu_fourview_pretrained`)** | **ELIGIBLE (LOCKED REUSE)** | `nyukat/breast_cancer_classifier` (`6176ebea...`), `sample_image_model.p` (`9a1d97f9...`). Reuse `nyu_nlbs_4view_case_scores.csv` (`03e19975...`, `6,184` studies, `0` fallbacks) at `$0.00`; single-ckpt vs 5-copy ensemble & 2019 deprecation noted. |
| 8 | **Model 3 (`glam_pretrained`)** | **ELIGIBLE (NEW INFER)** | `nyukat/GLAM` (`17a00198...`), `model_joint.ckpt` (`101.24 MB`, live 200 OK). Identical preprocessing to GMIC (`2944x1920` uint16 PNGs). **SHA-256 `UNKNOWN` until Wave 2 download.** License conflict (`LICENSE` NYU non-commercial vs README AGPL-3.0) flagged. |
| 9 | **Model 4 (`end2end_cbis_ddsm`)** | **ELIGIBLE (CONDITIONAL)** | `lishen/end2end-all-conv` (`58401e22...`), `ddsm_resnet50_s10_[512-512-1024]x2.h5` (whole-mammogram classifier, NOT patch-only) verified inside live GDrive folder (`resourcekey=0-t4vtopuv27D9NnMC97w6hg`). **SHA-256 `UNKNOWN` until Wave 2.** Requires Keras 2.0.5 -> PyTorch conversion + parity test. |
| 10 | **Model 5 (`frcnn_cad`)** | **INELIGIBLE (DISQUALIFIED)** | `riblidezso/frcnn_cad` (`6c2deb57...`). Official Caffe weight URL `http://dkrib.web.elte.hu/cad_faster_rcnn/vgg16_frcnn_cad.caffemodel` is **DEAD** (`error.elte.hu/error/452.html`), no official mirror exists, and CUDA 8.0/cuDNN 6 Caffe fails on L40S (`sm_89`). |
| 11 | **Model 5 Replacement (`R1` / `R2`)** | **PROPOSED** | **`R1`: `ianpan/mammoscreen`** (Apache-2.0, `244.3 MB`, LFS SHA-256 `8ace0998...`, Category A published RSNA thresholds `0.0072`/`0.0127`, 100% unseen NLBS). **`R2`: `dpetrini/vindr-convnext-multiple-view`** (MIT, `561.3 MB`, VinDr-Mammo BI-RADS >=3). |
| 12 | **Model 6 (`petrini_twoview_efnb3`)** | **ELIGIBLE (NEW INFER)** | `dpetrini/multiple-view` (`2331b545...`, MIT) + HF `eafc5d16...`, `516.95 MB`, LFS SHA-256 **`82add409...`**, CBIS-DDSM trained, strict `[CC_RGB, MLO_RGB]` order & chest-wall-on-left verified. |
| 13 | **Meta-Repository Audit** | **PASS** | `nyukat/mammography_metarepository` (`4f9b8a08...`, `BSD-2-Clause`, MEDINFO 2025/2026) audited; reusable canonical views & `model_joint` config extracted; legacy CUDA 8.0, random multi-view selection, and breast-level averaging rejected. |
| 14 | **Literature Overlap Boundary** | **PASS** | Cannot claim "first multi-model external benchmark" (MEDINFO 2025/2026 evaluated 5/6 overlap models on AUROC/AUPRC). Novelty restricted to **NLBS screening cohort + high-recall triage thresholds + missed-cancer overlap**. |
| 15 | **Threshold Audit (NO RSNA Calib)** | **PASS** | Strict **Category A/B/C** classification: `mammoscreen_r1` = **A** (`0.0072`/`0.0127`); `gmic`, `nyu`, `glam`, `end2end`, `petrini` = **C** (threshold-independent AUROC/AUPRC only unless non-RSNA **B** calibration approved). |
| 16 | **NLBS Cache Reuse Plan** | **PASS** | Zero raw DICOM (`324.45 GiB`) transfer needed! `glam` reuses `paper1_gmic_uint16_v1/nlbs` (`97.31 GiB` transfer -> `tigerwood693`); `end2end`, `petrini`, `mammoscreen_r1` reuse existing `tigerwood693:/cache/stage1_nlbs_uint16` (`35.94 GiB`). |
| 17 | **Target-Label Quarantine** | **PASS** | Enforced via `assert_no_forbidden_columns` over `nlbs_4view_common_cohort_label_free.csv` (`6,184` studies / `24,736` imgs / `12,368` breast pairs) and `inference_manifest_stage_c.csv` (`6,359` studies / `26,988` imgs / `12,374` breast pairs). |
| 18 | **Adapter Implementation** | **PASS** | Built [`src/paper1/exp2_adapters.py`](../../../../src/paper1/exp2_adapters.py) with registry, SHA-256 verification, preprocessing, aggregation, Category A/B/C enforcement, shard resume, and pre-join locks. |
| 19 | **Modal Runner & Guards** | **PASS** | Built [`scripts/run_paper1_exp2_modal.py`](../../../../scripts/run_paper1_exp2_modal.py) targeting `tigerwood693` with `--approve-pilot`, `--approve-full-infer`, `--max-total-spend-usd`, and `frcnn_cad` hard block. |
| 20 | **Offline CPU Unit Tests** | **PASS** | [`tests/test_paper1_exp2_adapters.py`](../../../../tests/test_paper1_exp2_adapters.py): **8/8 tests PASS** covering quarantine, SHA-256, Petrini channel order, aggregation, Category A/B/C, resume, and CLI guards. |
| 21 | **Wave 2 Commands & Parity** | **READY** | Exact commands prepared for weight download, SHA-256 pinning, Keras->PyTorch parity, and `97.31 GiB` cloud-to-cloud GMIC PNG transfer (`~USD 0.25`). |
| 22 | **Wave 3 Pilot Commands** | **READY** | Exact `200`-unit L40S pilot commands & official `sample_data` parity checks prepared (`~USD 0.20–0.40` total). |
| 23 | **Phase 2 Full NLBS Commands** | **READY** | Resumable L40S shard commands & immutable SHA-256 prediction lock path prepared (`~USD 1.70–3.50` total). |
| 24 | **All-Inclusive Budget Projection** | **PASS** | Total Wave 2 + Wave 3 + Phase 2 projected spend = **`~$2.15 – $4.20 USD`** on `tigerwood693` (`$25.77` credits available) and peak storage `~323 GiB << 1,024 GiB` (`$0.00/day` storage). |

---

## 2. Read-Only Modal Account & Volume Audit

### 2.1 Target Profile: `tigerwood693`
- **Billing Summary (`modal billing summary --for "this month" --json`):**
  - Metered cost: `$12.5146` (`volumes: $8.2395`, `ephemeral_apps: $3.8234`, `deployed_apps: $0.4037`, `network_egress: $0.0450`)
  - Offsets: `free_storage: -$8.2395`, `network_egress_allowance: -$0.0450`, `credits: -$4.2300` -> **`$0.00` billed**, **`$25.77 USD` credits remaining** (monthly Starter `$30.00` reset on `2026-12-01`).
- **Active/Deployed Apps (`modal app list`):** `0` apps.
- **Volume Inventory (`9` volumes, `410,266` files, `224.8539 GiB` total -> `$0.00/day` storage):**
  - `rsna-mammography-vol`: `81,806` files, `102.2373 GiB`
    - `/cache/stage1_nlbs_uint16`: `27,020` files, `35.9386 GiB` *(reused by Exp 2)*
    - `/cache/stage1_rsna_uint16`: `54,709` files, `65.9501 GiB` *(preserved)*
    - `/cache/paper1_dinov2_vits14`: `23` files, `0.2369 GiB` *(preserved)*
    - `/models/paper1`: `1` file (`dinov2_vits14_pretrain.pth`), `0.0822 GiB` *(preserved)*
    - `/rsna`: `1` file (`splits_seed42.csv`), `0.0046 GiB` *(preserved)*
    - `/_transfer_manifests`: `52` files, `0.0249 GiB` *(preserved)*
  - Other existing non-Paper-1 volumes preserved untouched: `moo-data` (`51.8016 GiB`), `beef-behavior-data` (`51.5941 GiB`), `cvb-data` (`14.2921 GiB`), `viewpoint-real-data` (`1.8247 GiB`), `behavior-perception-cache` (`1.4592 GiB`), `behavior-checkpoints` (`1.3943 GiB`), `viewpoint-checkpoints` (`0.2501 GiB`), `medmnist-custom-results` (`0.0004 GiB`).

### 2.2 Source Profile: `hasinishrak2015` (Read-Only)
- `nlbs-mammography-vol:/nlbs/extracted`: `26,988` files, `324.4480 GiB` (raw NLBS DICOMs; **NOT needed for transfer**).
- `rsna-mammography-vol:/cache/paper1_gmic_uint16_v1/nlbs`: `27,007` files, `97.3070 GiB` (`26,988` `(2944, 1920)` uint16 PNGs + manifests; **reusable directly by `glam_pretrained`**).
- Remaining credits: `$5.97` (`-$24.03` of `$30.00`). Note: transferring `97.31 GiB` from `hasinishrak2015` using `scripts/transfer_modal_volumes_cloud.py` incurs `$0.00` compute/egress on `hasinishrak2015` (verified in `2026-11-24_modal_cloud_to_cloud_dataset_transfer.md`).

---

## 3. Six-Model (+ Replacement) Eligibility & Provenance Matrix

| Model ID | Repo / Revision | Checkpoint & SHA-256 | Original Training Data | Input Spec & Ordering | Output & Exam Aggregation | License & Caveats | Eligibility |
|---|---|---|---|---|---|---|---|
| `gmic_pretrained` | `nyukat/GMIC` (`594029efe6184bcca3a151fa1dd46c5fd620cb25`) | `sample_model_1.p` (`60.74 MB`, `3b792e6d...`) | NYU (`229,426` exams, 2010–2017) | `2944x1920` uint16 PNG, per-img z-score | `malignant_pred` -> `max` across exam | `AGPL-3.0`; single ckpt `1` vs 5-ckpt ensemble | **ELIGIBLE (LOCKED REUSE)** |
| `nyu_fourview_pretrained` | `nyukat/breast_cancer_classifier` (`6176ebea6950b0a953339593d2da417ab846dae0`) | `sample_image_model.p` (`24.71 MB`, `9a1d97f9...`) | NYU (`229,426` exams, 2010–2017) | `2677x1942` CC / `2974x1748` MLO, 10-TTA `seed=0` | `max(left_mal, right_mal)` | `AGPL-3.0`; single ckpt vs 5-copy ensemble; 2019 clinical deprecation | **ELIGIBLE (LOCKED REUSE)** |
| `glam_pretrained` | `nyukat/GLAM` (`17a0019860441e2ea8d7b7c7e0aaeada735e871f`) | `model_joint.ckpt` (`101.24 MB`, blob `d8b6e133...`, SHA-256 `UNKNOWN`) | NYU (2010–2017) | Reuses GMIC `2944x1920` uint16 PNG (`0` crop diff) | `malignant_pred` -> `max` across exam | **NYU Non-Commercial `LICENSE`** vs AGPL-3.0 README; PyTorch 1.1.0 assert bypassed cleanly | **ELIGIBLE (NEW INFER)** |
| `end2end_cbis_ddsm` | `lishen/end2end-all-conv` (`58401e220a2dd5120b373eef40e9737d7ce8d56c`) | `ddsm_resnet50_s10_[512-512-1024]x2.h5` (GDrive `resourcekey=0-t4vtopuv27D9NnMC97w6hg`, SHA-256 `UNKNOWN`) | CBIS-DDSM whole-image (film, `S10` split) | `1152x896` RGB-rep, `*255/65535 - 52.18` (Stage-1 reuse) | 2-class softmax `pred[1]` -> `max` across exam | `CC BY-NC-SA 4.0`; requires Wave 2 Keras 2.0.5 -> PyTorch parity gate | **ELIGIBLE (CONDITIONAL)** |
| `frcnn_cad` | `riblidezso/frcnn_cad` (`6c2deb57e37832c281588da11aed8808ba6ee1e9`) | `vgg16_frcnn_cad.caffemodel` (**DEAD** `error.elte.hu/error/452.html`, SHA-256 `UNKNOWN`) | DDSM + private Hungarian (INbreast/DREAM eval) | Faster R-CNN RoI `cls_ind=2` (`MALIGNANT`) | `max` NMS RoI score | **No LICENSE file**; dead weights + CUDA 8.0/cuDNN 6 Caffe incompatible w/ L40S | **INELIGIBLE (DISQUALIFIED)** |
| `petrini_twoview_efnb3` | `dpetrini/multiple-view` (`2331b545...`) / HF (`eafc5d16...`) | `2024-10-01-...Best_2-Views-CBIS-EFB3-CVUBP.pt` (`516.95 MB`, LFS `82add409...`) | CBIS-DDSM 2-view (`N=2,694`, original split) | `1152x896` `(x-13369)/65535`, strict `[CC_RGB, MLO_RGB]`, chest-wall-left | 2-class softmax `pred[1]` per breast -> `max(L, R)` | `MIT`; bypass redundant single-view `.pt` load in `multi_view_clf_test.py` | **ELIGIBLE (NEW INFER)** |
| `mammoscreen_r1` *(Replacement R1)* | `huggingface.co/ianpan/mammoscreen` (`main`) | `model.safetensors` (`244.31 MB`, LFS `8ace0998...`) | CBIS-DDSM pretrain + RSNA (`80/10/10` Split 1; NLBS unseen) | 3-net ensemble (`2048x1024`, `1920x1280`, `1536x1536`) uint8 `[-1,1]` | View-avg per breast -> `max(L, R)` | `Apache-2.0`; **Category A** published RSNA thresholds (`0.0072` / `0.0127`) | **RECOMMENDED REPLACEMENT** |
| `petrini_vindr_r2` *(Replacement R2)* | `huggingface.co/dpetrini/vindr-convnext-multiple-view` (`main`) | `2024-08-07-...ConvNext_VinDr_Collab.pt` (`561.29 MB`, LFS `4619a4d3...`) | VinDr-Mammo (BI-RADS 1-2 vs 3-5; zero RSNA/NLBS overlap) | `1152x896` per-img z-score, `[CC_RGB, MLO_RGB]` | 2-class softmax `pred[1]` -> `max(L, R)` | `MIT`; endpoint is BI-RADS >=3 ("abnormal"), not biopsy cancer | **BACKUP REPLACEMENT** |

---

## 4. Original-Threshold Audit (Category A / B / C — Zero RSNA Calibration)

| Model ID | Threshold Category | Published Threshold Evidence | Permitted Exp 2 Evaluation | Forbidden Practices Blocked |
|---|---|---|---|---|
| `mammoscreen_r1` *(R1)* | **Category A** | Model card explicitly publishes RSNA-derived thresholds: `tau_98 = 0.0072` (`98.1%` sens, `65.4%` spec) and `tau_95 = 0.0127` (`94.3%` sens, `78.7%` spec). | Frozen decision sensitivity & dismissal on `100%` unseen NLBS + AUROC/AUPRC. | No RSNA recalibration, no NLBS threshold tuning. |
| `gmic_pretrained` | **Category C** *(opt. B)* | Shen et al. (2021) matched radiologist sensitivity dynamically on NYU reader study; `README` notes uncalibrated scores; no fixed numerical cutoff published. | Primary: threshold-independent AUROC/AUPRC. Optional B only on non-RSNA public source. | RSNA calibration & arbitrary `0.5` cutoffs blocked. |
| `nyu_fourview_pretrained` | **Category C** *(opt. B)* | Wu et al. (2019/2020) matched radiologist operating point on NYU internal study; no fixed numerical cutoff published. | Primary: threshold-independent AUROC/AUPRC. | RSNA calibration & arbitrary `0.5` cutoffs blocked. |
| `glam_pretrained` | **Category C** | Liu et al. (2021) & `README` explicitly state *"output scores are not calibrated"*; only `percent_k=0.03` pooling hyperparameter reported. | Threshold-independent AUROC/AUPRC only. | RSNA calibration & arbitrary `0.5` cutoffs blocked. |
| `end2end_cbis_ddsm` | **Category C** *(opt. B)* | Shen et al. (2019) reported `86.1%` sens / `80.1%` spec on CBIS-DDSM without publishing numerical threshold `tau`. | Primary: AUROC/AUPRC; optional B on CBIS-DDSM test set (`dpetrini/cbis_ddsm_rev`). | RSNA calibration & arbitrary `0.5` cutoffs blocked. |
| `petrini_twoview_efnb3` | **Category C** *(opt. B)* | Petrini & Kim (2025/2026) reported AUROC `0.8658` without publishing a numerical decision cutoff `tau`. | Primary: AUROC/AUPRC; optional B on CBIS-DDSM test set (`dpetrini/cbis_ddsm_rev`). | RSNA calibration & arbitrary `0.5` cutoffs blocked. |

---

## 5. Meta-Repository (`nyukat/mammography_metarepository`) Overlap & Reuse Audit

- **Repository & Paper:** Commit `4f9b8a08d7446b6a872f547e9794efdcbeb56271` (`BSD-2-Clause`), Chłędowski / Stadnick et al. (*MEDINFO 2025 / Stud Health Technol Inform 2025*, 5 models across 7 international datasets).
- **Reusable Findings:** Confirmed `model_joint` as default GLAM configuration, confirmed exact `2944x1920` GMIC preprocessing equivalence for `nyu_glam`, confirmed `end2end` scaling `255/65535 = 0.003891` and CBIS-DDSM `MEAN_PIXEL_INTENSITY = 52.18`, and confirmed `frcnn_cad` `cls_ind=2` (`MALIGNANT`) mapping.
- **Rejected Anti-Patterns:** (1) `end2end`/`frcnn_cad` CUDA 8.0 Dockerfiles fail on Ada Lovelace L40S (`sm_89`); (2) `generate_predictions.py` uses `RandomState(0).randint` when a view slot has >1 image, violating deterministic slot selection; (3) `evaluation/score.py` averages scores within each breast (`left_score /= left_images`), diluting unilateral lesions compared to Paper 1's `max` aggregation; (4) `models/end2end/predict/preprocess.py` computes target-dataset `MEAN_PIXEL_INTENSITY`, which would leak target intensity statistics if used unexamined.
- **Exact Novelty Overlap:** MEDINFO 2025/2026 already benchmarked `GMIC`, `NYU-4view`, `GLAM`, `End2End`, and `frcnn_cad` using AUROC/AUPRC. Paper 1 Exp 2 MUST NOT claim "first multi-model external benchmark"; its distinct contributions are (a) **NLBS (2025 FRDR Canadian screening cohort)**, (b) **direct comparison against Stratum A RSNA-supervised models + 2025/2026 `petrini` & `mammoscreen`**, (c) **Category A/B/C original-threshold audit vs source-frozen high-sensitivity dismissal**, and (d) **missed-cancer overlap**.

---

## 6. Local Verification & Next-Phase Commands (NOT EXECUTED IN WAVE 1)

### 6.1 Wave 1 Local Verification (Completed at `$0.00`)
```powershell
python -m unittest discover -s tests -p "test_paper1_exp2_adapters.py" -v
python scripts/run_paper1_exp2_modal.py --stage plan --model-id glam_pretrained --split nlbs_full
python scripts/run_paper1_exp2_modal.py --stage plan --model-id petrini_twoview_efnb3 --split nlbs_4view
python scripts/run_paper1_exp2_modal.py --stage plan --model-id mammoscreen_r1 --split nlbs_4view
```

### 6.2 Wave 2 Execution Plan (Awaiting Approval — `~USD 0.30` total)
1. **Cloud-to-Cloud Transfer of GMIC `2944x1920` NLBS PNGs (`97.31 GiB`, `hasinishrak2015` -> `tigerwood693`):**
   - Extend `src/utils/modal_transfer.py` with a `glam_nlbs` bundle targeting `/cache/paper1_gmic_uint16_v1/nlbs` (`27,007` files, `97.31 GiB`) and run `scripts/transfer_modal_volumes_cloud.py --bundle glam_nlbs --dst-profile tigerwood693` (`~USD 0.25`, `0` source delta).
2. **Download & Pin Checkpoints onto `tigerwood693:/rsna-mammography-vol/models/paper1_exp2/` (`~USD 0.05` CPU):**
   - `glam_pretrained`: `git clone https://github.com/nyukat/GLAM` (`17a00198...`), download `models/model_joint.ckpt` (`101.24 MB`), record exact SHA-256, verify `sample_data` 4-decimal parity (`0_L-CC`..`3_R-MLO` against `GLAM/README.md`).
   - `petrini_twoview_efnb3`: `git clone https://github.com/dpetrini/multiple-view` (`2331b545...`), download HF `2024-10-01-14h36m_50ep_best_model_AUC_08643_Best_2-Views-CBIS-EFB3-CVUBP.pt` (`516.95 MB`), verify SHA-256 `82add409...`, verify `samples/Calc-Test_P_00127_RIGHT_{CC,MLO}.png -> 0.1214` README parity!
   - `mammoscreen_r1` *(if approved)*: download HF `ianpan/mammoscreen` (`model.safetensors`, `244.31 MB`, verify SHA-256 `8ace0998...`).
   - `end2end_cbis_ddsm`: `gdown --folder "https://drive.google.com/drive/folders/0B1PVLadG_dCKV2pZem5MTjc1cHc?resourcekey=0-t4vtopuv27D9NnMC97w6hg"`, pin `ddsm_resnet50_s10_[512-512-1024]x2.h5` SHA-256, convert Keras `.h5` -> PyTorch `.pt` in isolated Keras 2.0.5/TF 1.4.0 CPU container and verify `max_abs_diff < 1e-5` before GPU run!

### 6.3 Wave 3 L40S Pilot Commands (`200` units each, `~USD 0.25–0.40` total — DO NOT RUN WITHOUT APPROVAL)
```powershell
$env:MODAL_PROFILE="tigerwood693"
modal run scripts/run_paper1_exp2_modal.py --stage pilot --model-id glam_pretrained --split nlbs_full --pilot-limit 200 --batch-size 1 --checkpoint-rel-path models/paper1_exp2/model_joint.ckpt --approve-pilot
modal run scripts/run_paper1_exp2_modal.py --stage pilot --model-id petrini_twoview_efnb3 --split nlbs_4view --pilot-limit 200 --batch-size 8 --checkpoint-rel-path models/paper1_exp2/2024-10-01-14h36m_50ep_best_model_AUC_08643_Best_2-Views-CBIS-EFB3-CVUBP.pt --approve-pilot
modal run scripts/run_paper1_exp2_modal.py --stage pilot --model-id mammoscreen_r1 --split nlbs_4view --pilot-limit 200 --batch-size 4 --checkpoint-rel-path models/paper1_exp2/mammoscreen --approve-pilot
```

### 6.4 Phase 2 Full NLBS Inference Commands (`~USD 1.60–3.50` total — DO NOT RUN WITHOUT PILOT APPROVAL)
```powershell
$env:MODAL_PROFILE="tigerwood693"
modal run scripts/run_paper1_exp2_modal.py --stage infer --model-id glam_pretrained --split nlbs_full --num-shards 4 --batch-size 1 --checkpoint-rel-path models/paper1_exp2/model_joint.ckpt --max-total-spend-usd 2.00 --approve-full-infer
modal run scripts/run_paper1_exp2_modal.py --stage infer --model-id petrini_twoview_efnb3 --split nlbs_4view --num-shards 4 --batch-size 8 --checkpoint-rel-path models/paper1_exp2/2024-10-01-14h36m_50ep_best_model_AUC_08643_Best_2-Views-CBIS-EFB3-CVUBP.pt --max-total-spend-usd 1.50 --approve-full-infer
modal run scripts/run_paper1_exp2_modal.py --stage infer --model-id mammoscreen_r1 --split nlbs_4view --num-shards 4 --batch-size 4 --checkpoint-rel-path models/paper1_exp2/mammoscreen --max-total-spend-usd 2.00 --approve-full-infer
```

---

## 7. All-Inclusive Cost & Storage Projections (`tigerwood693`)

| Phase / Work Package | Units | Projected Compute + Transfer | Peak Storage Delta | Cumulative Storage (`< 1,024 GiB` Free) |
|---|---:|---:|---:|---:|
| **Wave 1 (This Audit)** | — | **`$0.00`** | `+0.00 GiB` | `224.85 GiB` (`$0.00/day`) |
| **Wave 2: GMIC NLBS PNG Transfer + Weights + Keras Conversion** | `97.31 GiB` + 4 ckpts | `$0.25 – $0.35` | `+98.7 GiB` | `323.55 GiB` (`$0.00/day`) |
| **Wave 3: L40S Pilots (GLAM, Petrini, End2End, MammoScreen)** | `200` units x 4 | `$0.20 – $0.40` | `+0.02 GiB` | `323.57 GiB` (`$0.00/day`) |
| **Phase 2: `glam_pretrained` Full NLBS (`26,988` imgs)** | `6,359` studies | `$0.50 – $0.90` | `+0.05 GiB` | `323.62 GiB` (`$0.00/day`) |
| **Phase 2: `petrini_twoview_efnb3` NLBS (`12,368` pairs)** | `6,184` studies | `$0.30 – $0.60` | `+0.02 GiB` | `323.64 GiB` (`$0.00/day`) |
| **Phase 2: `end2end_cbis_ddsm` NLBS (`26,988` imgs)** | `6,359` studies | `$0.35 – $0.75` | `+0.02 GiB` | `323.66 GiB` (`$0.00/day`) |
| **Phase 2: `mammoscreen_r1` NLBS (`12,368` pairs, 3-net ensemble)** | `6,184` studies | `$0.55 – $1.20` | `+0.02 GiB` | `323.68 GiB` (`$0.00/day`) |
| **TOTAL EXP 2 PROJECTION** | **6 Models** | **`$2.15 – $4.20 USD`** | **`+98.8 GiB`** | **`~323.7 GiB` (`$0.00/day`)** |

*Headroom:* `$25.77` credits remaining on `tigerwood693` easily covers `$2.15–$4.20` (`>6x` safety margin), and `~323.7 GiB` remains `700 GiB` below Modal Starter's `1,024 GiB` continuous free storage quota.
