# Paper 1 — Experiment 2 (Original Pretrained Mammography Models) Final Phase 2 Readiness Audit

**Branch:** `paper1-exp2-phase1` (merged `main` at `465d3d4`)  
**Gate:** Final Phase 2 Readiness Audit — Ready for Phase 2 Approval (`$0.00` additional spend during audit)  
**Target Modal Profile:** `tigerwood693` (explicit `MODAL_PROFILE=tigerwood693`, never `modal profile activate`)

---

## 1. Finalized Eligible Cohort (4 Active Models, 1 Blocked Upstream, 1 Disqualified, 3 Rejected)

Experiment 2 evaluates **4 active eligible pretrained mammography models** on NLBS without RSNA supervision or RSNA threshold calibration. `petrini_twoview_efnb3` is formally marked `blocked_upstream` (never silently dropped from provenance) due to verified upstream file corruption on both HuggingFace and Google Drive; `frcnn_cad` remains `disqualified`; no invalid 5th/6th model is forced.

### Active Phase 2 Eligible Models (4 Models)

| # | Model ID | Source Repo & Commit | Checkpoint & SHA-256 | NLBS Execution Mode & Cache | Output Semantics & Aggregation | Primary Metric Policy |
|---|---|---|---|---|---|---|
| 1 | `gmic_pretrained` | `nyukat/GMIC` (`594029ef...`) | `sample_model_1.p` (`3b792e6d...`) | **Reuse locked ($0.00):** `reports/paper1/predictions_locked/gmic_nlbs_case_scores.csv` (`6,359` exams, `26,988` imgs, SHA-256 `94521810b9a78e7047a39d71bc54618ea2c7e95600a6f4fba717b79fdb3a6b36`) | Malignant class probability (`[:, 1]`), exam `max` | **Category C** (AUROC & AUPRC primary) |
| 2 | `nyu_fourview_pretrained` | `nyukat/breast_cancer_classifier` (`6176ebea...`) | `sample_image_model.p` (`9a1d97f9...`; single-checkpoint vs 100-model ensemble caveat) | **Reuse locked ($0.00):** `reports/paper1/predictions_locked/nyu_nlbs_4view_case_scores.csv` (`6,184` common 4-view exams, `24,736` imgs, `10-TTA`, SHA-256 `03e19975edb7a02d297d5b15b672169dd449a86c850488f3f543c5ceddede317`) | Breast malignant probabilities, exam `max(L, R)` | **Category C** (AUROC & AUPRC primary) |
| 3 | `glam_pretrained` | `nyukat/GLAM` (`17a00198...`) | `model_joint.ckpt` (`96.55 MiB`, SHA-256 `a647ba9176b3e11d2dec7252054504f33856bcabb2bbe763432d3d20e7c65aec`) | **New L40S inference:** `cache/paper1_gmic_uint16_v1/nlbs` (`27,007` files, `97.3070 GiB`, `2944x1920` uint16 PNG) | Softmax malignant prob (`model(x)[:, 1]`), exam `max` | **Category C** (AUROC & AUPRC primary) |
| 4 | `end2end_cbis_ddsm` | `lishen/end2end-all-conv` (`58401e22...`) | `end2end_cbis_ddsm_ts.pt` (`155.92 MiB`, SHA-256 `1c93c4c3ff7f997cc235140340181995d0219162cecd0bf495381b777a2389be`; Keras->TorchScript `max_abs_diff=4.51e-10`) | **New L40S inference:** `cache/paper1_exp2_uncropped_1152x896/nlbs` (`26,988` files, `16.0075 GiB`, uncropped `1152x896` uint16 PNG) | Keras `Dense(2, softmax)[:, 1]` direct output (no double-softmax), exam `max` | **Category C** (AUROC & AUPRC primary) |

### Blocked Upstream, Disqualified & Rejected Candidates (Hard-Blocked in `src/paper1/exp2_adapters.py`)

| Candidate ID | Status | Forensic / Protocol Justification |
|---|---|---|
| `petrini_twoview_efnb3` | **`blocked_upstream`** | Verified forensic corruption on both HuggingFace and Google Drive (`2024-10-01` & `2024-07-01` `.pt`): truncated at `88.0 MiB` (`nz_end = 92,274,688`) with `~405 MiB` zero-padding and no ZIP EOCD (`PK\x05\x06`); `torch.load` raises `BadZipFile`. |
| `frcnn_cad` | **`disqualified`** | Official ELTE weight URL (`error.elte.hu/error/452.html`) is dead; CUDA 8.0 / Caffe runtime is incompatible with L40S (`sm_89`); lesion detection outputs are not natively calibrated screening malignancy scores. |
| `mammoscreen_r1` | **`rejected`** | Fine-tuned on RSNA Screening Mammography (`80/10/10` Split 1), violating Exp 2's zero-RSNA-supervision rule. |
| `petrini_vindr_r2` | **`rejected`** | Trained on VinDr-Mammo BI-RADS `>= 3` screening assessments rather than confirmed cancer outcomes. |
| `mammo_clip_zeroshot` | **`rejected`** | Contrastive vision-language model rather than a supervised breast cancer classifier. |

---

## 2. License & Research-Use Restrictions Audit (4 Active Models)

Verified directly against upstream `LICENSE` and `README.md` files on GitHub:

1. **`nyukat/GMIC` (`gmic_pretrained`):** Licensed under **GNU AGPLv3** (`LICENSE` & `README.md`). Research/evaluation allowed; any modified/network-deployed service triggering AGPLv3 section 13 requires corresponding source disclosure under AGPLv3.
2. **`nyukat/breast_cancer_classifier` (`nyu_fourview_pretrained`):** Licensed under **GNU AGPLv3** (`LICENSE` & `README.md`). Explicit clinical disclaimer in `README.md` (`"not used clinically at NYU Langone Health"` — 2019 legacy model, non-clinical research use only).
3. **`nyukat/GLAM` (`glam_pretrained`):** **Internal Non-Commercial Research-Only (`LICENSE` file vs `README.md` AGPLv3 conflict).** Upstream `LICENSE` grants permission *solely for internal non-commercial research and evaluation purposes* (`Copyright 2021 New York University. All Rights Reserved`), explicitly prohibiting redistribution, sublicensing, modification, assignment, or commercial use (contact NYU Office of Industrial Liaison for commercial/distribution rights). Respect the stricter `LICENSE` terms: keep weights/outputs strictly internal and never redistribute checkpoints.
4. **`lishen/end2end-all-conv` (`end2end_cbis_ddsm`):** Licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (`CC BY-NC-SA 4.0`)** with explicit **Non-Commercial Terms of Use** in `README.md` (free for non-commercial research; commercial licensing via Mount Sinai Innovation Partners, MSIP; any adapted derivatives shared publicly must carry ShareAlike `CC BY-NC-SA 4.0`).

---

## 3. Threshold Restriction Audit (Strict Category C)

- **Primary Metrics:** Threshold-independent **AUROC** and **AUPRC** only (`threshold_category="C"`, `calibration_domain="none_threshold_independent_auroc_auprc_only"`).
- **Zero RSNA / Zero NLBS Calibration:** `apply_threshold_policy` in `src/paper1/exp2_adapters.py` explicitly raises `ValueError` if any `"RSNA"` or `"NLBS"` calibration domain is supplied.
- **No Arbitrary `0.5` Cutoffs & No False Dismissal Claims:** Because none of the 4 original publications specify compatible screening triage thresholds directly transportable to NLBS without external/target calibration, no arbitrary `0.5` cutoff and no uncalibrated sensitivity/specificity rule-out dismissal claims are made.
- **Label Quarantine:** `glam_pretrained` and `end2end_cbis_ddsm` read only `datasets/nlbs/inference_manifest_stage_c.csv` (`assert_no_forbidden_columns` enforced).

---

## 4. Live Read-Only Modal Audit on `tigerwood693` (`$0.00` Audit Spend)

Executed read-only against `MODAL_PROFILE=tigerwood693` after `main` merge (`465d3d4`):

- **Running Apps (`modal app list`):** `0`
- **Active Profile:** `tigerwood693`
- **Billing Summary (`modal billing summary`):**
  - Metered Cost: **`$14.79`** (`Volumes: $8.24`, `Ephemeral Apps: $5.97`, `Deployed Apps: $0.40`, `Network Egress: $0.17`)
  - Credits: **`-$6.38`** | Network Egress Allowance: **`-$0.17`** | Free Storage: **`-$8.24`** | **Billed Cost: `$0.00`**
- **Exact Storage Across All 9 Volumes (`464,431` files, `364,849,917,225` bytes = `339.7930 GiB` / `1,024 GiB` limit, `684.21 GiB` free, `$0.00/day`):**
  - `rsna-mammography-vol`: **`135,971` files, `233,191,388,679` bytes (`217.1764 GiB`)**
    - `/cache/stage1_nlbs_uint16`: `27,020` files, `38,588,761,450` bytes (`35.9386 GiB`)
    - `/cache/stage1_rsna_uint16`: `54,709` files, `70,813,338,933` bytes (`65.9501 GiB`)
    - `/cache/paper1_dinov2_vits14`: `23` files, `254,406,084` bytes (`0.2369 GiB`) — **INTACT**
    - `/cache/paper1_gmic_uint16_v1`: `27,007` files, `104,482,633,855` bytes (`97.3070 GiB`) — **Wave 2 `glam_nlbs` verified (`all_verified: true` in execution JSON)**
    - `/cache/paper1_exp2_uncropped_1152x896`: `26,988` files, `17,187,881,356` bytes (`16.0075 GiB`) — **Wave 2 Option B1 uncropped cache verified**
    - `/models`: `5` files, `1,173,103,648` bytes (`1.0925 GiB`) (`end2end_cbis_ddsm_ts.pt`, `ddsm_resnet50_s10_[512-512-1024]x2.h5`, `glam_model_joint.ckpt`, `petrini_twoview_efnb3.pt`, `dinov2_vits14_pretrain.pth`)
    - `/external_repos`: `150` files, `650,747,932` bytes (`0.6061 GiB`)
    - `/_transfer_manifests`: `68` files, `35,565,969` bytes (`0.0331 GiB`)
    - `/rsna`: `1` file, `4,949,452` bytes (`0.0046 GiB`)
  - `medmnist-custom-results`: `3` files, `401,479` bytes (`0.0004 GiB`)
  - `behavior-perception-cache`: `90,539` files, `1,566,805,946` bytes (`1.4592 GiB`)
  - `viewpoint-checkpoints`: `4` files, `268,560,534` bytes (`0.2501 GiB`)
  - `viewpoint-real-data`: `893` files, `1,959,235,393` bytes (`1.8247 GiB`)
  - `behavior-checkpoints`: `6,293` files, `1,497,166,591` bytes (`1.3943 GiB`)
  - `beef-behavior-data`: `4,351` files, `55,398,766,251` bytes (`51.5941 GiB`)
  - `cvb-data`: `226,363` files, `15,346,043,642` bytes (`14.2921 GiB`)
  - `moo-data`: `14` files, `55,621,548,710` bytes (`51.8016 GiB`)

---

## 5. Wave 3 Pilot Benchmarks & Exact Phase 2 Execution Plan (`tigerwood693`)

Verified from `reports/paper1/expanded/exp2/pilots/{glam_pretrained,end2end_cbis_ddsm}_pilot_bench.json`:

| Model ID | Recommended Batch Size | Throughput | Peak VRAM (Alloc / Res) | Pilot Diff vs Smallest BS | Projected Full NLBS (`26,988` imgs / `6,359` studies) |
|---|---|---|---|---|---|
| `glam_pretrained` | **`4`** | **`11.674 imgs/sec`** (`2.918 exams/sec`) | `1.1056 / 1.9121 GiB` | `0.0003297` | **`35.32 min` / `$1.3901` (`$1.39`)** |
| `end2end_cbis_ddsm` | **`32`** | **`29.385 imgs/sec`** (`7.346 exams/sec`) | `6.3919 / 7.1777 GiB` | `0.000184` | **`14.03 min` / `$0.5523` (`$0.55`)** |
| **Combined Phase 2** | — | — | — | — | **`49.35 min` / `~$1.94 USD`** (`~$1.94–$2.15` incl. NVMe staging) |

### Exact Executable Phase 2 CLI Commands (After Approval)

`scripts/run_paper1_exp2_modal.py` uses resumable shard caches (`rsna-mammography-vol:/cache/paper1_exp2_shards/<model_id>/nlbs_full/shard_XX_of_04.json` verified via `expected_unit_ids_sha256` and `checkpoint_sha256`) and writes immutable pre-join locks to `reports/paper1/expanded/exp2/predictions_locked/{model_id}_nlbs_full_{unit_scores.csv, case_scores.csv, provenance_lock.json}`:

```powershell
# 1. GLAM full NLBS inference (26,988 imgs / 6,359 studies, bs=4, 4 shards)
$env:MODAL_PROFILE="tigerwood693"; python scripts/run_paper1_exp2_modal.py --stage infer --model-id glam_pretrained --split nlbs_full --batch-size 4 --num-shards 4 --approve-full-infer

# 2. End2End CBIS-DDSM full NLBS inference (26,988 imgs / 6,359 studies, bs=32, 4 shards)
$env:MODAL_PROFILE="tigerwood693"; python scripts/run_paper1_exp2_modal.py --stage infer --model-id end2end_cbis_ddsm --split nlbs_full --cache-mode uncropped_1152x896 --batch-size 32 --num-shards 4 --approve-full-infer
```

*(Note: A dedicated CPU-only `scripts/evaluate_paper1_exp2.py` script for post-lock Category C AUROC/AUPRC & cluster-bootstrap reporting has not yet been authored and should be added once `glam_pretrained` and `end2end_cbis_ddsm` Phase 2 locks complete; `--split nlbs_full` covers both full `N=6,359` and common 4-view `N=6,184` evaluation via subsetting.)*
