# Paper 1 — Experiment 2 (Original Pretrained Mammography Models) Phase 1 Readiness & Wave 2–3 Approval Request

**Branch:** `paper1-exp2-phase1` (merged `main` at `409ec2b`)  
**Phase / Gate:** Phase 1 Continuation — Step A / Wave 2–3 Prep (`$0.00` cloud spend executed; zero Modal GPU/transfer containers launched)  
**Target Modal Profile:** `tigerwood693` (explicit `MODAL_PROFILE=tigerwood693`, never `modal profile activate`)

---

## 1. Executive Summary & Locked Model Cohort (5 Eligible, 1 Disqualified, 3 Rejected)

Experiment 2 evaluates off-the-shelf, publicly released pretrained mammography models on the quarantined NLBS cohort without RSNA-derived threshold calibration and without altering existing DINOv2 assets on `tigerwood693`.

### Locked Eligible Models (5 Models)

| # | Model ID | Source Repo / Weights | NLBS Execution Mode | NLBS Input Cache / Resolution | Primary Metric Policy |
|---|---|---|---|---|---|
| 1 | `gmic_pretrained` | `nyukat/GMIC` (`sample_model_1.p`) | **Reuse locked predictions** (`reports/paper1/predictions_locked/gmic_nlbs_case_scores.csv`) | N/A (locked `$0.00`) | **Category C** (AUROC & AUPRC primary) |
| 2 | `nyu_fourview_pretrained` | `nyukat/breast_cancer_classifier` (`sample_image_model.p`) | **Reuse locked predictions** (`reports/paper1/predictions_locked/nyu_nlbs_4view_case_scores.csv`; single-checkpoint vs ensemble caveat documented) | N/A (locked `$0.00`) | **Category C** (AUROC & AUPRC primary) |
| 3 | `glam_pretrained` | `nyukat/GLAM` (`17a00198...`, `model_joint.ckpt`, `101.24 MB`) | **New Modal L40S inference** | `glam_nlbs` bundle (`paper1_gmic_uint16_v1/nlbs`, `27,007` files, `97.31 GiB`, `2944x1920` uint16 PNG) | **Category C** (AUROC & AUPRC primary) |
| 4 | `end2end_cbis_ddsm` | `lishen/end2end-all-conv` (`58401e22...`, `ddsm_resnet50_s10_[512-512-1024]x2.h5`, CBIS-DDSM whole-mammogram classifier; Keras 2.0.5 -> PyTorch parity) | **New Modal L40S inference** | **Option B1 (Recommended):** `cache/paper1_exp2_uncropped_1152x896/nlbs` (`~30–38 GiB`, uncropped `1152x896` uint16 PNG) | **Category C** (AUROC & AUPRC primary) |
| 5 | `petrini_twoview_efnb3` | `dpetrini/multiple-view` (`2331b545...`) & HF `dpetrini/cbis-ddsm-efficientnetb3-multiple-view` (`eafc5d16...`, MIT, `516.95 MB`, SHA-256 `82add40900054835f23737d9f3a01922275f1e4dcfd3156a0f7059cb58147071`) | **New Modal L40S inference** | **Option B1 (Recommended):** `cache/paper1_exp2_uncropped_1152x896/nlbs` (`~30–38 GiB`, strict `[CC_RGB, MLO_RGB]`, chest-wall-left) | **Category C** (AUROC & AUPRC primary) |

### Formally Disqualified & Rejected Candidates (Hard-Blocked in `src/paper1/exp2_adapters.py`)

| Candidate ID | Status | Formal Justification |
|---|---|---|
| `frcnn_cad` | **DISQUALIFIED** | Upstream ELTE checkpoint URL (`error.elte.hu/error/452.html`) is dead; legacy Caffe / CUDA 8.0 runtime is incompatible with L40S (`sm_89`); outputs generic lesion detections rather than calibrated exam malignancy. |
| `mammoscreen_r1` | **REJECTED** | Trained on RSNA Mammography data — violates cross-cohort independence for Experiment 2 NLBS evaluation. |
| `petrini_vindr_r2` | **REJECTED** | VinDr-Mammo label target (BI-RADS >= 3) is screening assessment rather than biopsy-confirmed cancer ground truth. |
| `mammo_clip_zeroshot` | **REJECTED** | Contrastive vision-language model (VLM), not a supervised mammography cancer classifier aligned with Exp 2 scope. |

---

## 2. Threshold & Metric Policy Audit (Zero RSNA Calibration)

1. **Primary Metrics Locked (Category C):** Threshold-independent **AUROC** and **AUPRC** are locked as the primary Experiment 2 evaluation metrics across all 5 eligible models.
2. **Category A / B Guardrail:** No arbitrary `0.5` cutoff and zero RSNA-derived threshold calibration are permitted. Operating-point sensitivity/specificity (Category A/B) may only be reported if an independent published/source-domain threshold is explicitly provided and validated via `apply_threshold_policy` in `src/paper1/exp2_adapters.py`.
3. **Label Quarantine:** Inference manifests remain strictly label-free (`datasets/paper1/nlbs_4view_common_cohort_label_free.csv`, `datasets/nlbs/inference_manifest_stage_c.csv`).

---

## 3. Read-Only Modal Account & Volume Inventory Audit

Verified read-only prior to any Wave 2 spend:

- **`tigerwood693` (Execution Profile):**
  - Active running apps: `0`
  - Remaining credits: **`$25.77`** (resets `2026-12-01`); billed: `$0.00`
  - Existing storage: **`224.8539 GiB`** across `9` volumes (`410,266` files, `$0.00/day` storage, **`799.15 GiB`** headroom under `1,024 GiB` limit)
  - `rsna-mammography-vol` (`102.2373 GiB`, `81,806` files): `/cache/stage1_nlbs_uint16` (`27,020` files, `35.9386 GiB`), `/cache/stage1_rsna_uint16` (`54,709` files, `65.9501 GiB`), `/cache/paper1_dinov2_vits14` (`23` files, `0.2369 GiB` — preserved untouched), `/models/paper1` (`1` file, `0.0822 GiB`)
- **`hasinishrak2015` (Read-Only Source Profile):**
  - Remaining credits: `$5.97`
  - `nlbs-mammography-vol:/nlbs/extracted`: `26,988` files, `324.4480 GiB`
  - `rsna-mammography-vol:/cache/paper1_gmic_uint16_v1/nlbs`: `27,007` files, `97.3070 GiB`

---

## 4. NLBS Preparation & Cache Option Reconciliation

### A. `glam_pretrained` Cache (`glam_nlbs` Bundle)
Added `"glam_nlbs"` to `src/utils/modal_transfer.py` (`get_bundle_specs("glam_nlbs")`), transferring `hasinishrak2015:rsna-mammography-vol/cache/paper1_gmic_uint16_v1/nlbs` (`27,007` files, `97.31 GiB`, `2944x1920` uint16 PNG) to `tigerwood693:rsna-mammography-vol/cache/paper1_gmic_uint16_v1/nlbs` (`~$0.25 USD`, `$0.00` source delta).

### B. `end2end_cbis_ddsm` & `petrini_twoview_efnb3` Cache Reconciliation (Option A vs Option B1 vs Option B2)

| Option | Strategy | Storage on `tigerwood693` | Estimated Cost | Preprocessing Parity | Recommendation |
|---|---|---|---|---|---|
| **Option A** | Transfer full raw `nlbs-mammography-vol:/nlbs/extracted` DICOMs (`324.45 GiB`) + decode | `+324.45 GiB` | `$1.15–$1.75` | Exact native | Rejected (unnecessary egress/storage) |
| **Option B1** | Stream read-only DICOMs from `hasinishrak2015` -> CPU decode/invert/right-mirror -> save uncropped `(1152, 896)` `uint16` PNGs to `cache/paper1_exp2_uncropped_1152x896/nlbs` (`--stage build_uncropped_cache`) | **`~30–38 GiB`** | **`$0.38–$0.55`** (`$0.00` source delta) | Matches CBIS-DDSM uncropped `1152x896` full-field aspect & pixel mean | **RECOMMENDED** |
| **Option B2** | Reuse existing `/cache/stage1_nlbs_uint16` (`35.94 GiB`, `1520x912` cropped) | `+0 GiB` | `$0.00` | Cropped breast bounding box shifts global pixel mean vs uncropped CBIS-DDSM training | Fallback only (documented ablation risk) |

**Peak `tigerwood693` Storage Projection (Option B1 + `glam_nlbs`):**
`224.85 GiB` (existing) + `97.31 GiB` (`glam_nlbs`) + `~35 GiB` (Option B1 `1152x896` cache) + `~1.5 GiB` (weights/outputs) = **`~358.7 GiB << 1,024 GiB`** (`$0.00/day` storage).

---

## 5. Adapter, Runner & Local Test Verification

- **`src/paper1/exp2_adapters.py`:** Locks `EXP2_ELIGIBLE_MODELS` (5 models), hard-blocks the 4 disqualified/rejected models in `assert_model_eligible`, implements `EXP2_NLBS_CACHE_OPTIONS`, `prepare_uncropped_1152x896_from_array`, `default_pilot_batch_sizes`, preprocessing/aggregation adapters, `apply_threshold_policy`, shard schema validation, and immutable prediction locking.
- **`scripts/run_paper1_exp2_modal.py`:** Implements `--stage plan`, `--stage build_uncropped_cache --approve-wave2-download`, progressive batch-size `--stage pilot` (`run_exp2_pilot_benchmark_l40s` benchmarking `vram_allocated_gb`, `vram_reserved_gb`, `imgs_per_sec`, `exams_per_sec`, `data_load_sec`, `gpu_infer_sec`, OOM handling, parity checks, saving `reports/paper1/expanded/exp2/pilots/{model_id}_pilot_bench.json`), and `--stage full`.
- **`src/utils/modal_transfer.py` & `tests/test_transfer_modal_volumes_cloud.py`:** Adds and unit-tests `"glam_nlbs"` bundle specification.
- **`tests/test_paper1_exp2_adapters.py`:** CPU unit test suite covering cohort lock, 4 hard-blocked candidates, Option B1 uncropped `1152x896` preprocessing, progressive pilot batch sizes, `--stage build_uncropped_cache` approval guard, Category B/C enforcement, shard resume, and locking.

---

## 6. Section 9 — Consolidated Wave 2 + Wave 3 Approval Request (And Phase 2 Projection)

> **STOP GATE:** Zero paid Modal actions have been run. Approval is requested before executing Wave 2 or Wave 3.

### Cost Breakdown (`tigerwood693`, `$25.77` credits available)

| Stage | Scope | Exact Estimated Cost |
|---|---|---|
| **Wave 2 (Data & Weights)** | `glam_nlbs` transfer (`~$0.25`) + Option B1 uncropped `1152x896` cache build (`~$0.38–$0.55`) + 3 weight downloads & Keras->PyTorch parity verification (`~$0.05–$0.10`) | **`~$0.68–$0.90 USD`** |
| **Wave 3 (L40S Pilots)** | Progressive batch-size `--stage pilot` (`200–500` units) across `glam_pretrained`, `end2end_cbis_ddsm`, `petrini_twoview_efnb3` | **`~$0.20–$0.35 USD`** |
| **Wave 2 + Wave 3 Subtotal** | **Immediate Approval Request** | **`~$0.88–$1.25 USD`** |
| **Phase 2 (Full NLBS)** | Full NLBS L40S inference + immutable prediction locks (`glam_pretrained`, `end2end_cbis_ddsm`, `petrini_twoview_efnb3`; `gmic` & `nyu` `$0.00` reuse) | **`~$1.15–$2.25 USD`** |
| **Combined Exp 2 Total** | **Wave 2 + Wave 3 + Phase 2** | **`~$2.03–$3.50 USD`** |

### Exact Pre-Staged Wave 2 Commands (After Approval)

```powershell
# 1. Free dry-run plan + Paid Modal transfer for GLAM 2944x1920 NLBS PNG cache (97.31 GiB) to tigerwood693
python scripts/transfer_modal_volumes_cloud.py --dst-profile tigerwood693 --bundle glam_nlbs --plan-only
$env:MODAL_PROFILE="tigerwood693"; modal --profile tigerwood693 run scripts/transfer_modal_volumes_cloud.py --dst-profile tigerwood693 --bundle glam_nlbs

# 2. Build Option B1 uncropped 1152x896 uint16 NLBS PNG cache (~30-38 GiB) on tigerwood693
$env:MODAL_PROFILE="tigerwood693"; python scripts/run_paper1_exp2_modal.py --stage build_uncropped_cache --model-id end2end_cbis_ddsm --cache-mode uncropped_1152x896 --approve-wave2-download
```

### Exact Pre-Staged Wave 3 Progressive Pilot Commands (After Wave 2)

```powershell
# GLAM L40S progressive pilot (batch sizes: 1,2,4)
$env:MODAL_PROFILE="tigerwood693"; python scripts/run_paper1_exp2_modal.py --stage pilot --model-id glam_pretrained --pilot-batch-sizes 1,2,4 --approve-pilot

# End2End CBIS-DDSM L40S progressive pilot (batch sizes: 4,8,16,32)
$env:MODAL_PROFILE="tigerwood693"; python scripts/run_paper1_exp2_modal.py --stage pilot --model-id end2end_cbis_ddsm --cache-mode uncropped_1152x896 --pilot-batch-sizes 4,8,16,32 --approve-pilot

# Petrini Two-View EfficientNet-B3 L40S progressive pilot (batch sizes: 2,4,8,16)
$env:MODAL_PROFILE="tigerwood693"; python scripts/run_paper1_exp2_modal.py --stage pilot --model-id petrini_twoview_efnb3 --cache-mode uncropped_1152x896 --pilot-batch-sizes 2,4,8,16 --approve-pilot
```

### Exact Phase 2 Commands (After Wave 3 Pilot Sign-Off)

```powershell
$env:MODAL_PROFILE="tigerwood693"; python scripts/run_paper1_exp2_modal.py --stage infer --model-id glam_pretrained --split nlbs_full --approve-full-infer
$env:MODAL_PROFILE="tigerwood693"; python scripts/run_paper1_exp2_modal.py --stage infer --model-id end2end_cbis_ddsm --cache-mode uncropped_1152x896 --split nlbs_full --approve-full-infer
$env:MODAL_PROFILE="tigerwood693"; python scripts/run_paper1_exp2_modal.py --stage infer --model-id petrini_twoview_efnb3 --cache-mode uncropped_1152x896 --split nlbs_full --approve-full-infer
```
