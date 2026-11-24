# Paper 1 Exp 3 — Phase 1 Step A Readiness, Security Fix & Wave 2–3 Plan

**Branch:** `paper1-exp3-phase1` (synced with `main` commit `409ec2b`)  
**Modal Target Profile:** `tigerwood697` (explicit `MODAL_PROFILE=tigerwood697`; `modal profile activate` NEVER called)  
**Status:** **STEP A / WAVE 1 COMPLETE — CONDITIONAL GO FOR WAVE 2 + WAVE 3 (`$0.00` spent in Phase 1)**

---

## 1. Read-Only Modal Audit (`tigerwood697`)

- **Billing (`modal billing summary` / `report`):** Metered `$8.54` (`Ephemeral Apps: $4.52`, `Volumes: $3.99`, `Network Egress: $0.04`); Credits `-$4.52`; Free Storage `-$3.99`; Egress Allowance `-$0.04` -> **Billed Cost: `$0.00`**. Remaining starter credits: **`~$25.48`**. Running apps: **`0`**.
- **Storage (`~165.26 GiB` across 7 volumes `< 1,024 GiB` free tier -> `$0.00/day`):**
  - Preserved ResNet-50 assets on `rsna-mammography-vol`: `models/paper1/resnet50-11ad3fa6.pth` (`102,540,417` B), `models/paper1/checkpoints/resnet50_rsna_seed42_{best_train_calib,latest}.pt`, `cache/stage1_rsna_uint16/png_1520x912` (`54,706` PNGs), `cache/stage1_nlbs_uint16/png_1520x912` (`26,988` PNGs), `cache/paper1_resnet50_preds/` (`4` CSVs).
- **Exact Checkpoint Sync Commands Needed on `tigerwood697` (`ce_b5`, `trust_b5`, `dinov2`):**
  ```powershell
  # 1. DINOv2 ViT-S/14 backbone (from hasinishrak2015) + RSNA head (from local)
  $env:MODAL_PROFILE='hasinishrak2015'; modal volume get rsna-mammography-vol models/paper1/dinov2_vits14_pretrain.pth models/paper1/dinov2_vits14_pretrain.pth
  $env:MODAL_PROFILE='tigerwood697'; modal volume put rsna-mammography-vol models/paper1/dinov2_vits14_pretrain.pth /models/paper1/dinov2_vits14_pretrain.pth
  $env:MODAL_PROFILE='tigerwood697'; modal volume put rsna-mammography-vol reports/paper1/dinov2_head/dinov2_vits14_head_seed42.pt /models/paper1/checkpoints/dinov2_vits14_head_seed42.pt

  # 2. Mammo-CLIP B5 + CE-only & TRUST Epoch-20 checkpoints (from hasinishrak74001 -> tigerwood697)
  $env:MODAL_PROFILE='hasinishrak74001'; modal volume get rsna-mammography-vol models/mammo_clip/b5-model-best-epoch-7.tar models/mammo_clip/b5-model-best-epoch-7.tar
  $env:MODAL_PROFILE='hasinishrak74001'; modal volume get rsna-mammography-vol models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt
  $env:MODAL_PROFILE='hasinishrak74001'; modal volume get rsna-mammography-vol models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt
  $env:MODAL_PROFILE='tigerwood697'; modal volume put rsna-mammography-vol models/mammo_clip/b5-model-best-epoch-7.tar /models/mammo_clip/b5-model-best-epoch-7.tar
  $env:MODAL_PROFILE='tigerwood697'; modal volume put rsna-mammography-vol models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt /models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt
  $env:MODAL_PROFILE='tigerwood697'; modal volume put rsna-mammography-vol models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt /models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt
  ```
- **Agent 1 Pending Handoff (`gmic_rsna_finetuned` & `nyu_fourview_rsna_finetuned`):** Validated via `validate_agent1_handoff_bundle()` (checkpoint path, 64-hex SHA-256, `paper1_{gmic,nyu}_finetune_ckpt_v1` schema, and RSNA `val_calib` `tau_98`/`tau_95`).

---

## 2. CRITICAL SECURITY FIX — LUMINA Neutral Pseudonymous IDs

- **Issue Fixed in `src/paper1/external_datasets.py`:** Raw LUMINA archives organize images under `Benign/` and `Malign/` folders (`ID=1..218` colliding across both). Exposing `Benign/` / `Malign/` or `lumina_benign_` / `lumina_malign_` in `INFERENCE_MANIFEST` would leak ground-truth class labels during inference.
- **Implemented Mitigation:**
  1. `_assign_lumina_pseudonyms()` hashes each `(class_folder, case_id)` with a salted SHA-256 (`paper1_exp3_lumina_v1`) to deterministically interleave benign & malignant cases and assign **neutral pseudonymous IDs** (`lumina_case_0001`..`lumina_case_0468`, `lumina_img_00001`..) and **neutral paths** (`lumina_neutral/lumina_case_XXXX/lumina_img_YYYYY.dcm`).
  2. `ingest_lumina_neutral_cpu()` unzips `LUMINA_RAW_{Benign,Malign}.zip` directly onto neutral paths on disk at Wave 2 (`cpu=1.0, memory=2048`) BEFORE `INFERENCE_MANIFEST` is mounted.
  3. `assert_lumina_neutral_inference_manifest()` runs automatically inside `validate_inference_manifest()` for `dataset_id=="lumina"`, rejecting any occurrence of `benign` or `malign` substrings across ALL inference columns (`cluster_id`, `exam_id`, `image_id`, `image_path`, `acquisition_meta`).
  4. Verified by `test_09_lumina_neutral_pseudonym_quarantine` (`10/10` unit tests PASS).

---

## 3. Dataset Eligibility, IMDB Manual Download & Cross-Release Overlap Audit Plan

*(Strictly excludes OMAMA-DB, EMBED, OPTIMAM, and VinDr-Mammo.)*

| Dataset | Country | Accessibility & License | Scale & Role | Critical Guards |
| :--- | :--- | :--- | :--- | :--- |
| **NLBS** | Canada | Locked locally (`$0.00` reuse) / FRDR | `6,359` full (`6,184` 4-view, `151` pos). **Screening** | Primary screening dismissal benchmark (`ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`). |
| **IMDB** | India | IBIA Web UI (no login, no `curl`/`wget` API) / MIDAS CC BY 4.0 | `3,219` pts / `12,674` DICOMs (`388` pos). **Mixed diagnostic + opportunistic** | Prefix `r1_`/`r2_`; dedup `SOPInstanceUID` & pixel hash; separate FFDM vs SM; never map BI-RADS blindly. |
| **CMMD** | China | TCIA NBIA REST API (`10.7937/tcia.eqde-4b16`), CC BY 4.0 | `1,775` pts / `5,202` DICOMs (`21.292 GiB`), `826` 4-view. **Biopsy-enriched diagnostic** | **NO population-screening dismissal claims.** Track `97` bilateral cases at exam + breast levels. |
| **LUMINA** | Turkey | OSF Storage API (`osf.io/b63jc`), CC BY-NC 4.0 | `468` pts / `1,824` imgs (`8.476 GiB` `LUMINA_RAW`), 6+ vendors. **Multi-vendor diagnostic stress** | **Neutral pseudonymous IDs mandatory.** View-agnostic models only (malign discards contralateral breast -> no NYU 4-view). |

### Shortest Exact Manual/Browser Download Instructions for IMDB
1. Visit `https://ibdc.dbt.gov.in/ibia/studybrowse/` (no registration/login needed) and enter `mammo` in the search box — or open the two study pages directly:
   - **IMDB r1.0:** `https://ibdc.dbt.gov.in/ibia/study_details_browse/MAMOS_1000000004/` (`583` pts, `3,577` imgs)
   - **IMDB r2.0:** `https://ibdc.dbt.gov.in/ibia/study_details_browse/MAMOS_1000000050/` (`2,636` pts, `9,097` imgs)
2. Click the **Data** download icon on each study card to download the DICOM archives and accompanying metadata CSVs locally.
3. Upload to `tigerwood697` volume:
   ```powershell
   $env:MODAL_PROFILE='tigerwood697'
   modal volume put rsna-mammography-vol <LOCAL_IMDB_R1_DIR> /external_datasets/imdb/raw/r1/
   modal volume put rsna-mammography-vol <LOCAL_IMDB_R2_DIR> /external_datasets/imdb/raw/r2/
   ```
   *(Optional scripted mirror: request CC BY 4.0 access at `https://www.midas.iisc.ac.in/fe/datasets/breast/breast-mammography`.)*

### IMDB Cross-Release Patient-Overlap Audit Plan (`r1.0` vs `r2.0`)
1. Prefix sequential IDs (`imdb_r1_<pt_XXX>`, `imdb_r2_<pt_XXX>`) because `deidentify.py` starts `pt_001` independently in both releases.
2. Check DICOM `SOPInstanceUID` and `StudyInstanceUID` intersection across `r1.0` (2020) and `r2.0` (Aug–Dec 2025).
3. Compute raw pixel-array SHA-256 (`ds.pixel_array.tobytes()`) + acquisition fingerprint `(PatientAge, ManufacturerModelName, laterality, view)` across releases to catch any re-anonymized shared images or repeat patients.
4. Link any matched repeat patients under a single `cluster_id` for cluster-bootstrap CIs; separate `FFDM` vs `SM` (`ImageType`/`SeriesDescription`) and preserve composite `label_source` (`histopathology` vs `followup` vs `birads12_negative`).

---

## 4. Upgraded `run_paper1_exp3_modal.py` Capabilities (Zero Cloud Spend Yet)

- **Progressive Batch-Size L40S Pilot (`--stage pilot --pilot-batch-sizes ...`):** Benchmarks `16,32,64` (`ce_b5`, `trust_b5`) and `32,64,128` (`dinov2_rsna`, `resnet50_rsna`) on `200–500` images in `CMMD`/`LUMINA`, measuring `img/s`, `peak_vram_gb`, score parity (`max_abs_diff <= 1e-3`), and projected full-dataset USD cost.
- **CPU Worker Scaling (`--cpu-workers {2,4,8,16}`):** Dynamically scales `preprocess_external_dataset_cpu` via `.with_options(cpu=..., memory=max(4096, cpu*2048))` + intra-container `ThreadPoolExecutor`, while keeping download/unzip strictly at `cpu=1.0, memory=2048`.
- **Unit Tests:** `python tests/test_paper1_exp3_universal.py -v` -> **10/10 PASS** (`2.211s`); `tests/test_stage_c_quarantine_and_eval.py` -> **3/3 PASS**.

---

## 5. Section 9 — Wave 2 + Wave 3 Approval Request (`MODAL_PROFILE=tigerwood697`)

Approve **Wave 2 + Wave 3 (`~$1.25 USD` total on `tigerwood697`, `$0.00/day` storage)** before launching full Phase 2:

| Step | Scope | Est. Cost (USD) | Added Storage (GiB) | Command |
| :--- | :--- | :---: | :---: | :--- |
| **Wave 2a: Checkpoint Sync** | Sync `ce_b5`, `trust_b5`, `dinov2` weights -> `tigerwood697` | `~$0.02` | `+1.85` | See Section 1 commands above |
| **Wave 2b: LUMINA Ingest & Stage-1** | OSF `LUMINA_RAW` download (`cpu=1, 2G`) + neutral unzip + `--cpu-workers 4` Stage-1 PNGs | `~$0.08` | `+19.5` | `modal run scripts/run_paper1_exp3_modal.py --stage download --dataset lumina --approve-download` -> `preprocess --cpu-workers 4` |
| **Wave 2c: CMMD NBIA & Stage-1** | TCIA NBIA download (`21.29 GiB`, `cpu=1, 2G`) + `--cpu-workers 8` Stage-1 PNGs | `~$0.25` | `+48.8` | `modal run scripts/run_paper1_exp3_modal.py --stage download --dataset cmmd --approve-download` -> `preprocess --cpu-workers 8` |
| **Wave 2d: IMDB Stage-1** | *(After user uploads IBIA `r1/r2`)* `--cpu-workers 8` cross-release audit & Stage-1 PNGs | `~$0.65` | `+195.5` | `modal run scripts/run_paper1_exp3_modal.py --stage preprocess --dataset imdb --cpu-workers 8 --approve-preprocess` |
| **Wave 3: L40S Progressive Pilots** | `200`-img progressive batch-size L40S pilots (`ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`) | `~$0.25` | `~0.1` | `modal run scripts/run_paper1_exp3_modal.py --stage pilot --dataset cmmd --model resnet50_rsna --pilot-limit 200 --pilot-batch-sizes 32,64,128` |
| **Wave 2 + Wave 3 Total** | **Ready for Immediate Approval** | **`~$1.25`** | **`+265.7` (`430.9 GiB` total `< 1 TiB`)** | **Zero storage overage (`$0.00/day`)** |

- **Phase 2 Full Inference Envelope (Gated on Wave 3 Pilot Receipts + Agent 1 Handoff):** `~$1.70–$4.30` L40S (`~$2.95–$5.55` all-inclusive across Wave 2 + Wave 3 + Phase 2, well within `tigerwood697`'s `~$25.48` free credits).
