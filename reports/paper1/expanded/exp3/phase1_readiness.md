# Paper 1 Exp 3 — Phase 1 Wave 1 Readiness & Eligibility Audit

**Branch:** `paper1-exp3-phase1`  
**Modal Target Profile:** `tigerwood697` (explicit `MODAL_PROFILE=tigerwood697`; `modal profile activate` NEVER called)  
**Wave 1 Status:** **COMPLETE — CONDITIONAL GO FOR WAVE 2** (Zero paid GPU/preprocess runs executed in Wave 1)

---

## 1. Read-Only Modal Audit (`tigerwood697`)

- **Billing & Credits (`modal billing summary` / `report`):**
  - Metered total: `$8.54` (`Ephemeral Apps: $4.52`, `Volumes: $3.99`, `Network Egress: $0.04`).
  - Credits applied: `-$4.52`; Free Storage applied: `-$3.99`; Network Egress Allowance: `-$0.04` -> **Billed Cost: `$0.00`**.
  - Estimated monthly free credits remaining: **`~$25.48`** (out of `$30.00` starter cycle).
  - Running apps: **`0`**.
- **Storage & Existing ResNet-50 Assets (`rsna-mammography-vol`, `~165.26 GiB` total across 7 volumes `< 1 TiB` free tier -> `$0.00/day`):**
  - `models/paper1/resnet50-11ad3fa6.pth`: Verified (`102,540,417` bytes).
  - `models/paper1/checkpoints/resnet50_rsna_seed42_best_train_calib.pt` & `latest.pt`: Verified present.
  - `cache/stage1_rsna_uint16/png_1520x912`: Verified **`54,706`** lossless uint16 PNGs.
  - `cache/stage1_nlbs_uint16/png_1520x912`: Verified **`26,988`** lossless uint16 PNGs.
  - `cache/paper1_resnet50_preds/`: Verified all 4 CSVs (`nlbs_image_scores.csv`, `nlbs_case_scores.csv`, `rsna_val_calib_image_scores.csv`, `rsna_val_calib_case_scores.csv`).
- **Missing Checkpoints Needed on `tigerwood697` Before Phase 2:**
  - **Completed Models (Sync in Wave 2 from `hasinishrak74001` / `dryousufmozumder` / local):**
    - `dinov2_vits14_pretrain.pth` + `reports/paper1/dinov2_head/dinov2_vits14_head_seed42.pt` (frozen thresholds `tau_98 = 0.183435708284378`, `tau_95 = 0.2065157741308212`).
    - Mammo-CLIP `b5-model-best-epoch-7.tar` + `ce_rsna_seed42_final_epoch20.pt` (`tau_98 = 0.0961934626102447`, `tau_95 = 0.124415136873722`) + `trust_rsna_seed42_final_epoch20.pt` (`tau_98 = 0.1795480698347091`, `tau_95 = 0.1954631060361862`).
  - **Pending Agent 1 Handoff (`gmic_rsna_finetuned` & `nyu_fourview_rsna_finetuned`):**
    - Require: (1) best `train_calib` checkpoint matching schema `paper1_gmic_finetune_ckpt_v1` / `paper1_nyu_finetune_ckpt_v1`, (2) checkpoint SHA-256, (3) frozen `rsna_val_calib` `@98%` and `@95%` thresholds calibrated strictly on RSNA `val_calib` (validated via `validate_agent1_handoff_bundle`), and (4) native GMIC `(2944, 1920)` / NYU `(2677, 1942)` + `(2974, 1748)` preprocessing caches.

---

## 2. Dataset Eligibility, Accessibility, License & Caveat Matrix

*(Explicitly excludes OMAMA-DB, EMBED, OPTIMAM, and VinDr-Mammo.)*

| Dataset | Country | Accessibility & Route | License | Scale & Format | 4-View Subset | Ground Truth & Clinical Role | Critical Guardrails |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NLBS** | Canada | Already locked locally (`reports/paper1/six_model_final/locked_nlbs/`) | FRDR Academic | `26,988` DICOMs (`6,359` studies, `5,942` clusters) | `6,184` studies (`151` pos, `5,868` clusters) | Screening registry + biopsy (`2.4%` pos). **Organized screening** | Reuse `$0.00` for `ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`. |
| **IMDB** | India | **IBIA** web download (`MAMOS_1000000004` r1.0, `MAMOS_1000000050` r2.0) — no login, **no curl/wget API**; MIDAS on request | IBIA Open Access (~CC BY, DBT BIOTECH-PRIDE); MIDAS CC BY 4.0 | `3,219` pts / `12,674` DICOMs (`r1`: `583`/`3,577`; `r2`: `2,636`/`9,097`), Hologic Selenia 2 gens (~`180 GiB` raw est.) | Partial (check per-exam `{L-CC, L-MLO, R-CC, R-MLO}` after ingest) | Composite: BI-RADS 4/5 histopathology, BI-RADS 3 6-mo follow-up, BI-RADS 1/2 negative + follow-up (`388` cancer / `2,821` non-cancer / `10` unclassified). **Mixed diagnostic + opportunistic screening** | Prefix `r1_`/`r2_` (`pt_XXX` collision); dedup `SOPInstanceUID`; separate **FFDM vs SM**; never map BI-RADS blindly; no screening dismissal claims. |
| **CMMD** | China | **TCIA NBIA REST API** (`DOI 10.7937/tcia.eqde-4b16`), zero credentials | **CC BY 4.0** | `1,775` pts / `5,202` DICOMs (`21.292 GiB`), GE Senographe DS | `826 / 1,775` studies (`3,304` imgs); `949` 2-view unilateral | `100%` biopsy-confirmed (`1,316` malignant / `556` benign rows; `97` bilateral). **Biopsy-enriched diagnostic** | **NO population-screening dismissal claims.** Track both exam-max and breast-level (`97` bilateral). |
| **LUMINA** | Turkey | **OSF Storage API** (`https://osf.io/b63jc/`, direct ZIPs, zero login) & Kaggle | **CC BY-NC 4.0** | `468` pts (`250` benign, `218` malign) / `1,824` imgs (`8.476 GiB` `LUMINA_RAW` DICOMs, `2364x2964`–`4800x6000`, 6+ vendors) | **Not balanced** (malign discards contralateral breast: only 2/436 bilateral rows) | `100%` pathology-confirmed (`960` benign, `864` malignant imgs). **Multi-vendor / energy diagnostic stress test** | Namespace `lumina_benign_<ID>`/`lumina_malign_<ID>`; ingest `LUMINA_RAW` DICOMs primary; view-agnostic models only (`no NYU 4-view`). |

---

## 3. Universal Pipeline Architecture & Verification

- **`src/paper1/external_datasets.py`**:
  - Standardized non-outcome `INFERENCE_MANIFEST_COLUMNS` checked with `assert_no_forbidden_columns` (blocking `Cancer`, `BIRADS`, `pathology`, `malignant`, `classification`, etc. both at column level and inside `acquisition_meta` JSON).
  - Protected `OUTCOME_MANIFEST_COLUMNS` separated completely from inference inputs.
  - Dataset-specific builders for `nlbs`, `imdb`, `cmmd`, and `lumina` plus deterministic `select_common_4view_cohort` (never fabricating views) and `build_sensitivity_cohorts`.
- **`src/paper1/universal_eval.py`**:
  - Registry (`RSNA_SUPERVISED_MODELS`) for all 6 RSNA-supervised models + exact frozen thresholds + `validate_agent1_handoff_bundle`.
  - Resumable shard validation (`save_inference_shard_atomic`, `validate_and_load_inference_shard`), SHA-256 pre-join prediction locking (`lock_external_predictions`), tamper-abort verifier (`verify_and_load_locked_external_predictions`), `reuse_completed_nlbs_predictions`, cluster-bootstrap CIs, and `assert_no_cross_setting_pooling`.
- **`scripts/run_paper1_exp3_modal.py` & `scripts/evaluate_paper1_exp3.py`**:
  - Explicit `tigerwood697` guard, `cpu=1.0, memory=2048` downloads with 60s commits, L40S pilot (`200-500`), and per-dataset CPU evaluation CLI.
- **CPU Unit Tests (`tests/test_paper1_exp3_universal.py`):** **8/8 PASSED** (`2.048s`).

---

## 4. 24-Item Phase 1 Readiness Checklist

1. Branch isolation (`paper1-exp3-phase1`) — **DONE**
2. Read-only Modal `tigerwood697` billing & active apps audit — **DONE**
3. Modal `tigerwood697` storage & `< 1 TiB` free-tier verification (`~165.26 GiB`) — **DONE**
4. ResNet-50 weights, checkpoints & Stage-1 caches verified on `tigerwood697` — **DONE**
5. Missing RSNA-supervised checkpoints & frozen thresholds cataloged — **DONE**
6. Agent 1 fine-tuned GMIC & NYU handoff contract defined — **DONE**
7. NLBS `$0.00` reuse (`6,184` 4-view / `6,359` full) verified — **DONE**
8. IMDB IBIA/MIDAS accessibility & licensing verified — **DONE**
9. IMDB r1.0/r2.0 counts & `pt_XXX` / `SOPInstanceUID` deduplication designed — **DONE**
10. IMDB label provenance, diagnostic/screening mix & FFDM/SM separation designed — **DONE**
11. CMMD TCIA NBIA accessibility & CC BY 4.0 license verified — **DONE**
12. CMMD `1,775` pts / `5,202` DICOMs, `826` 4-view subset & biopsy caveat verified — **DONE**
13. LUMINA OSF accessibility & CC BY-NC 4.0 license verified — **DONE**
14. LUMINA multi-vendor DICOMs, ID collision & unilateral 4-view limitation verified — **DONE**
15. OMAMA-DB, EMBED, OPTIMAM, VinDr-Mammo exclusion enforced — **DONE**
16. Standardized label-free inference & protected outcome manifests built — **DONE**
17. Deterministic no-view-fabrication 4-view & sensitivity cohort selectors built — **DONE**
18. Universal 6-model registry & Agent 1 handoff validator built — **DONE**
19. Resumable shards & SHA-256 pre-join prediction locking built — **DONE**
20. Per-dataset cluster-bootstrap CIs & anti-pooling guardrail built — **DONE**
21. Local CPU unit tests (`tests/test_paper1_exp3_universal.py`) passing (`8/8`) — **DONE**
22. Wave 2 download/cache/sync commands specified — **DONE**
23. Wave 3 L40S pilot commands (`200-500` items) specified — **DONE**
24. Phase 2 commands & all-inclusive cost/storage budget projected — **DONE**

---

## 5. Exact Execution Commands (Waves 2, 3 & Phase 2 — Run Only After Approval)

### Wave 2 — Sync Checkpoints, Download & CPU Preprocess (`MODAL_PROFILE=tigerwood697`)
```powershell
$env:MODAL_PROFILE="tigerwood697"

# 1. Sync remaining completed RSNA-supervised weights (Mammo-CLIP + CE/TRUST + DINOv2) to tigerwood697
modal volume put rsna-mammography-vol models/checkpoints/stage_b_rsna/ /models/checkpoints/stage_b_rsna/
modal volume put rsna-mammography-vol reports/paper1/dinov2_head/dinov2_vits14_head_seed42.pt /models/paper1/checkpoints/dinov2_vits14_head_seed42.pt

# 2. CPU-only downloads (cpu=1.0, memory=2048, 60s commits)
modal run scripts/run_paper1_exp3_modal.py --stage download --dataset lumina --approve-download
modal run scripts/run_paper1_exp3_modal.py --stage download --dataset cmmd --approve-download
# For IMDB: download MAMOS_1000000004 & MAMOS_1000000050 via IBIA browser UI, upload to /external_datasets/imdb/raw/

# 3. CPU-only Stage-1 1520x912 uint16 PNG caching (after building per-dataset manifests)
modal run scripts/run_paper1_exp3_modal.py --stage preprocess --dataset lumina --manifest-csv datasets/paper1/exp3/lumina_inference_manifest.csv --approve-preprocess
modal run scripts/run_paper1_exp3_modal.py --stage preprocess --dataset cmmd   --manifest-csv datasets/paper1/exp3/cmmd_inference_manifest.csv   --approve-preprocess
modal run scripts/run_paper1_exp3_modal.py --stage preprocess --dataset imdb   --manifest-csv datasets/paper1/exp3/imdb_inference_manifest.csv   --approve-preprocess
```

### Wave 3 — L40S `200-500` Image/Exam Pilots (`MODAL_PROFILE=tigerwood697`)
```powershell
$env:MODAL_PROFILE="tigerwood697"
modal run scripts/run_paper1_exp3_modal.py --stage pilot --dataset lumina --model resnet50_rsna --manifest-csv datasets/paper1/exp3/lumina_inference_manifest.csv --pilot-limit 200 --max-spend-usd 1.0
modal run scripts/run_paper1_exp3_modal.py --stage pilot --dataset cmmd   --model resnet50_rsna --manifest-csv datasets/paper1/exp3/cmmd_inference_manifest.csv   --pilot-limit 200 --max-spend-usd 1.0
modal run scripts/run_paper1_exp3_modal.py --stage pilot --dataset imdb   --model resnet50_rsna --manifest-csv datasets/paper1/exp3/imdb_inference_manifest.csv   --pilot-limit 200 --max-spend-usd 1.0
```

### Phase 2 — Full L40S Inference & CPU Evaluation
```powershell
$env:MODAL_PROFILE="tigerwood697"
# View-agnostic models across LUMINA, CMMD, IMDB (+ pending Agent 1 GMIC/NYU with --handoff-json)
modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset lumina --model ce_b5 --manifest-csv datasets/paper1/exp3/lumina_inference_manifest.csv --approve-full-infer --max-spend-usd 2.0
modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset cmmd   --model all    --manifest-csv datasets/paper1/exp3/cmmd_inference_manifest.csv   --handoff-json reports/paper1/expanded/exp3/agent1_handoff.json --approve-full-infer --max-spend-usd 4.0
modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset imdb   --model all    --manifest-csv datasets/paper1/exp3/imdb_inference_manifest.csv   --handoff-json reports/paper1/expanded/exp3/agent1_handoff.json --approve-full-infer --max-spend-usd 6.0

# CPU-only SHA-256 locked multi-dataset evaluation (reusing NLBS at $0.00)
python scripts/evaluate_paper1_exp3.py --config-json reports/paper1/expanded/exp3/eval_config.json --reuse-nlbs-from reports/paper1/six_model_final/locked_nlbs
```

---

## 6. All-Inclusive Storage & Compute Cost Projections (`tigerwood697`)

- **Storage Projection:**
  - Current `tigerwood697`: `165.26 GiB`
  - Added checkpoints (`~1.85 GiB`) + LUMINA (`~19.5 GiB`) + CMMD (`~48.8 GiB`) + IMDB (`~195.5 GiB` est.) = **`~265.65 GiB`**
  - Projected Peak: **`~430.91 GiB`** (`< 1,024 GiB` free tier -> **`$0.00/day`** storage overage!).
- **Compute Cost Projection:**
  - Wave 2 (CPU downloads + Stage-1 caches + sync): **`~$1.00`**
  - Wave 3 (L40S 200-img pilots across datasets/models): **`~$0.25`**
  - Phase 2 (Full L40S inference across LUMINA `1,824`, CMMD `5,202`, IMDB `12,674` + pending NLBS `gmic/nyu` fine-tuned): **`~$1.70 - $4.30`**
  - **Total All-Inclusive Projection:** **`~$2.95 - $5.55 USD`** (well within `tigerwood697`'s **`~$25.48`** remaining free monthly credits).
