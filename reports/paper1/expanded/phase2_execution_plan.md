# Paper 1 Expanded — Phase 2 Execution Plan & Budget Reconciliation

**Generated:** `2026-11-25` (Audit Base `main` after Exp 2 + Exp 3 integration)  
**Gate Status:** **AWAITING `GO PHASE 2`** (Zero full Phase 2 training or inference launched)  
**Project-Wide Additional Modal Cap:** **`$30.00`** (`~$3.59–$3.64` spent in Wave 2+3 -> **`~$26.36–$26.41` remaining project headroom**)

---

## 1. Account Credits, Modal Balances & Recalculated All-Inclusive Budget

### A. Live Account Balances (Read-Only Verified `2026-11-25`, `0` Running Apps)

| Modal Profile | Assigned Scope | Metered Credits Used (`billing summary`) | Remaining Starter Credits (`$30.00` cap) | Live Storage (`GiB` / `1,024 GiB`) | Free-Tier Headroom (`GiB`) | Daily Storage Fee |
|---|---|---:|---:|---:|---:|---:|
| `dryousufmozumder` | **Exp 1** (GMIC & NYU RSNA fine-tune + NLBS eval) | `$4.44` (`+$0.56` Wave 2+3) | **`~$25.56`** | `963.74 GiB` (`6` vols) | `60.26 GiB` | `$0.00/day` |
| `tigerwood693` | **Exp 2** (GLAM & End2End NLBS inference) | `$6.38` (`+$2.15` Wave 2+3) | **`~$23.62`** | `339.79 GiB` (`9` vols) | `684.21 GiB` | `$0.00/day` |
| `tigerwood697` | **Exp 3** (CMMD, LUMINA, conditional IMDB, NLBS reuse) | `$5.45` (`+$0.93` Wave 2+3) | **`~$24.55`** | `~243.49 GiB` (`7` vols) | `~780.51 GiB` | `$0.00/day` |
| **Combined** | **All 3 Accounts** | **`+$3.64` (`~$3.59` reported) Wave 2+3** | **`~$73.73` account credits** | — | — | **`$0.00/day`** |

### B. Recalculated Phase 2 Cost Breakdown (From Wave 3 Measured Pilots)

| Experiment / Stage | Modal Profile | Basis | Est. CPU Runtime | Est. GPU Runtime (`L40S`) | Est. Cost (USD) |
|---|---|---|---:|---:|---:|
| **Wave 2+3 Actual Spend** | All 3 | Measured billing delta (`$0.56 + $2.15 + $0.93`) | — | — | **`~$3.59–$3.64`** |
| **Exp 1 — GMIC Train (`bf16, bs=64`)** | `dryousufmozumder` | Pilot `49.001 img/s` (`0.531h` w/ val) | — | `~32 min` | **`$1.25`** |
| **Exp 1 — GMIC Eval (`val_calib` + `nlbs`) + `nlbs` transfer** | `dryousufmozumder` | Measured eval + `97.31 GiB` 96-stream transfer | `~7–12 min` (`transfer`) | `~48–58 min` | **`~$2.10`** (`$0.70 + $0.23 + $1.20`) |
| **Exp 1 — NYU `train_all` CPU PNGs (`38,124` imgs)** | `dryousufmozumder` | Scaled `4.001x` from `9,528` RSNA (`$0.5488`) | `~25–35 min` (`24` shards) | — | **`~$2.25`** (`$2.20–$2.44`) |
| **Exp 1 — NYU Feature Extract (`bs=16`) + CPU Head Train** | `dryousufmozumder` | Pilot `6.8278 exam/s` (`110.52 min`) + `< 1 min` CPU | `< 1 min` | `~111 min` | **`$4.76`** |
| **Exp 1 — NYU 10-TTA Eval (`rsna_val_calib` + `nlbs_4view`, `bs=8`)** | `dryousufmozumder` | Pilot `1.2611 exam/s` (`2,382 + 6,184` exams) | — | `~113 min` | **`$4.88`** (`$1.36 + $3.52`) |
| **Exp 1 Subtotal** | `dryousufmozumder` | — | **`~33–48 min`** | **`~5.1 hrs`** | **`~$15.24`** *(fits `$25.56`)* |
| **Exp 2 — `glam_pretrained` NLBS (`bs=4`)** | `tigerwood693` | Pilot `11.674 img/s` (`26,988` imgs) | — | `~35.3 min` | **`$1.39`** |
| **Exp 2 — `end2end_cbis_ddsm` NLBS (`bs=32`)** | `tigerwood693` | Pilot `29.385 img/s` (`26,988` imgs) | — | `~14.0 min` | **`$0.55`** |
| **Exp 2 — GMIC + NYU NLBS Reuse** | Local CPU | Pre-locked SHA-256 reuse | `< 1 min` | `0 min` | **`$0.00`** |
| **Exp 2 Subtotal** | `tigerwood693` | `~$1.94` pure GPU (`~$1.94–$2.15` w/ NVMe stage) | **`< 1 min`** | **`~49.4 min`** | **`~$1.94–$2.15`** *(fits `$23.62`)* |
| **Exp 3 — CMMD + LUMINA (4 Ready Models) + NLBS Reuse** | `tigerwood697` | Pilots (`30.1–163.5 img/s` CMMD; `23.6–62.7` LUMINA) | `< 1 min` (`NLBS`) | `~9–11 min` (`~3m` 4-shard) | **`~$0.42`** (`$0.3677` GPU) |
| **Exp 3 — Agent 1 Handoff (`gmic` CMMD+LUMINA, `nyu` CMMD `826`)** | `tigerwood697` | Scaled from Exp 1/2 throughputs | — | `~40–65 min` | **`~$1.50–$2.50`** |
| **Exp 3 Subtotal (Without IMDB)** | `tigerwood697` | — | **`< 1 min`** | **`~50–76 min`** | **`~$1.92–$2.92`** *(fits `$24.55`)* |
| **Phase 2 Baseline Total (Exp 1 + Exp 2 + Exp 3 Ready)** | All 3 | Excludes conditional IMDB | `~35–50 min` | `~6.8–7.2 hrs` | **`~$19.10–$20.31`** |
| **Conditional: Exp 3 — IMDB (`12,674` DICOMs, after manual IBIA upload)** | `tigerwood697` | CPU decode + 6-model L40S inference | `~15–25 min` | `~25–45 min` | **`~$1.60–$2.80`** |
| **Grand Total (Wave 2+3 Actual + Phase 2 w/o IMDB)** | All 3 | Against **`$30.00` project cap** | — | — | **`~$22.74–$23.95`** *(Reserve: **`~$6.05–$7.26`**)* |
| **Grand Total (Wave 2+3 Actual + Phase 2 + Conditional IMDB)** | All 3 | Against **`$30.00` project cap** | — | — | **`~$24.34–$26.75`** *(Reserve: **`~$3.25–$5.66`**)* |

---

## 2. Required Execution Order & Checkpoint Handoff Dependencies

1. **Track 2 (`tigerwood693`, Exp 2) — INDEPENDENT, READY IMMEDIATELY:** Run `glam_pretrained` and `end2end_cbis_ddsm` full NLBS inference (`~$1.94–$2.15`). Reuse `gmic_pretrained` and `nyu_fourview_pretrained` (`$0.00`). Author & run CPU-only `scripts/evaluate_paper1_exp2.py` Category C AUROC/AUPRC evaluation.
2. **Track 3A (`tigerwood697`, Exp 3 Ready-4) — INDEPENDENT, READY IMMEDIATELY:** Run `ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna` on `cmmd` and `lumina` (`~$0.42`), plus `$0.00` `nlbs` reuse verification.
3. **Track 1 (`dryousufmozumder`, Exp 1) — SEQUENTIAL WITH TWO USER-APPROVED STORAGE GATES:**
   - **Stage 1A:** GMIC train (`bf16, bs=64`) -> GMIC `rsna_val_calib` eval -> CPU `calibrate_rsna` -> **Record `<GMIC_BEST_SHA256>`**.
   - **USER-APPROVAL CLEANUP GATE #1:** Delete regenerable GMIC `rsna_train` + `rsna_train_calib` PNGs (`-245.23 GiB`) so `rsna-mammography-vol` stays `< 1,024 GiB`.
   - **Stage 1B:** Transfer `dryousuf_gmic_eval_nlbs` (`+97.31 GiB`) -> GMIC `nlbs` eval -> CPU `lock_nlbs` -> CPU `eval_nlbs`.
   - **USER-APPROVAL CLEANUP GATE #2:** Delete GMIC `rsna_val_calib` + `nlbs` PNGs (`-159.37 GiB`) before building NYU `train_all`.
   - **Stage 1C:** NYU `train_all` CPU PNG build (`+279.20 GiB`) -> NYU `extract_head_features` (`--batch-exams 16 --max-spend-usd 6.5`) -> NYU `train_cached_head` -> **Record `<NYU_BEST_SHA256>`** -> NYU `rsna_val_calib` eval (`--checkpoint-path ..._cached_...`) -> CPU `calibrate_rsna` -> NYU `nlbs_4view` eval (`--num-shards 4`) -> CPU `lock_nlbs` + `eval_nlbs`.
4. **Track 3B (`tigerwood697`, Exp 3 Agent 1 Handoff) — DEPENDS ON STAGE 1A + STAGE 1C:**
   - Sync `gmic_rsna_partial_seed42_best_train_calib.pt` (stored at `models/paper1/checkpoints/gmic_rsna_finetuned_best_train_calib.pt` or matched via `checkpoint_path_override`) and `nyu_rsna_head_only_cached_seed42_best_train_calib.pt` (stored at `models/paper1/checkpoints/nyu_fourview_rsna_finetuned_best_train_calib.pt` or matched via `checkpoint_path_override`) + `sample_model_1.p` / `sample_image_model.p` to `tigerwood697:rsna-mammography-vol`.
   - Build `<AGENT1_HANDOFF_JSON>` containing `checkpoint_sha256`, `schema_version`, and frozen `rsna_val_calib_thresholds` (`target_98`, `target_95`).
   - Run Exp 3 `agent1_handoff` on `cmmd` (`nyu` auto-restricted to `826` 4-view exams) and `gmic_rsna_finetuned` on `lumina` (`nyu` hard-blocked).
   - Run isolated `scripts/evaluate_paper1_exp3.py --config-json <CONFIG_JSON>` (never pooling NLBS with CMMD/LUMINA/IMDB).
5. **Track 3C (Conditional IMDB) — BLOCKED BY EXTERNAL ACCESS:** Only after manual IBIA portal download (`MAMOS_1000000004`, `MAMOS_1000000050`) or MIDAS access + `modal volume put` to `tigerwood697` + cross-release dedup & FFDM/SM manifest separation.

---

## 3. Exact Executable Phase 2 Commands

### Experiment 1 (`MODAL_PROFILE=dryousufmozumder`, run from main repo root with `external/` present)

```powershell
$env:MODAL_PROFILE="dryousufmozumder"

# 1A-1. GMIC Fine-Tuning (L40S, bf16, bs=64, auto steps_per_epoch=75, scaled_lr=2.8284e-4, warmup=12) [~$1.25, ~32m]
modal run scripts/run_paper1_gmic_finetune_modal.py --stage train --strategy partial --precision bf16 --batch-size 64 --eval-batch-size 64 --steps-per-epoch 0 --lr 1e-4 --lr-scaling sqrt --warmup-steps 100 --epochs 8 --early-stopping-patience 3 --max-spend-usd 3.5 --approve-full-train

# 1A-2. GMIC RSNA val_calib Eval-Only + Calibration Lock [~$0.70, ~18m GPU + <1m CPU]
modal run scripts/run_paper1_gmic_finetune_modal.py --stage eval --split rsna_val_calib --batch-size 1 --num-shards 4 --checkpoint-path /rsna_vol/models/paper1/checkpoints/gmic_rsna_partial_seed42_best_train_calib.pt --checkpoint-sha256 <GMIC_BEST_SHA256> --max-spend-usd 1.5 --approve-eval
python scripts/evaluate_paper1_specialist_finetune.py --model gmic_rsna_finetuned --phase calibrate_rsna --expected-checkpoint-sha256 <GMIC_BEST_SHA256>

# === PAUSE: USER-APPROVED CLEANUP GATE #1 (-245.23 GiB GMIC rsna_train & rsna_train_calib PNGs) ===

# 1B-1. Transfer GMIC NLBS Cache (97.31 GiB) [~$0.23, ~7-12m]
$env:MODAL_SRC_PROFILES="hasinishrak2015"
modal --profile dryousufmozumder run scripts/transfer_modal_volumes_cloud.py --dst-profile dryousufmozumder --bundle dryousuf_gmic_eval_nlbs --num-containers 16 --streams-per-container 6 --max-budget-usd 0.80

# 1B-2. GMIC NLBS Eval-Only + Pre-Join Lock + Outcome Eval [~$1.20, ~30-40m GPU + <1m CPU]
modal run scripts/run_paper1_gmic_finetune_modal.py --stage eval --split nlbs --batch-size 1 --num-shards 4 --checkpoint-path /rsna_vol/models/paper1/checkpoints/gmic_rsna_partial_seed42_best_train_calib.pt --checkpoint-sha256 <GMIC_BEST_SHA256> --max-spend-usd 2.5 --approve-eval
python scripts/evaluate_paper1_specialist_finetune.py --model gmic_rsna_finetuned --phase lock_nlbs
python scripts/evaluate_paper1_specialist_finetune.py --model gmic_rsna_finetuned --phase eval_nlbs

# === PAUSE: USER-APPROVED CLEANUP GATE #2 (-159.37 GiB GMIC rsna_val_calib & nlbs PNGs) ===

# 1C-1. NYU Full Native Train/Train_Calib CPU PNG Build (38,124 imgs) [~$2.25, ~25-35m CPU]
modal run scripts/preprocess_paper1_nyu_cpu.py --stage build --split train_all --num-shards 24 --compression 1 --approve-full-build

# 1C-2. NYU Feature Extraction (176.86 MiB cache; --max-spend-usd MUST BE >= 6.0) [~$4.76, ~111m GPU]
modal run scripts/run_paper1_nyu_finetune_modal.py --stage extract_head_features --batch-exams 16 --max-spend-usd 6.5 --approve-full-train

# 1C-3. NYU Cached-Head Training (CPU) [~$0.001, <1m] -> record <NYU_BEST_SHA256>
modal run scripts/run_paper1_nyu_finetune_modal.py --stage train_cached_head --epochs 8 --batch-exams 16 --steps-per-epoch 400 --lr 1e-4 --max-spend-usd 0.5 --approve-full-train

# 1C-4. NYU RSNA val_calib 10-TTA Eval (--checkpoint-path MUST include _cached_) + Calibration Lock [~$1.36, ~31m GPU]
modal run scripts/run_paper1_nyu_finetune_modal.py --stage eval --split rsna_val_calib --batch-exams 8 --num-shards 4 --checkpoint-path /rsna_vol/models/paper1/checkpoints/nyu_rsna_head_only_cached_seed42_best_train_calib.pt --checkpoint-sha256 <NYU_BEST_SHA256> --max-spend-usd 2.5 --approve-eval
python scripts/evaluate_paper1_specialist_finetune.py --model nyu_fourview_rsna_finetuned --phase calibrate_rsna --expected-checkpoint-sha256 <NYU_BEST_SHA256>

# 1C-5. NYU NLBS 4-View 10-TTA Eval (--num-shards 4 required for $2/shard cap) + Lock + Eval [~$3.52, ~82m GPU]
modal run scripts/run_paper1_nyu_finetune_modal.py --stage eval --split nlbs_4view --batch-exams 8 --num-shards 4 --checkpoint-path /rsna_vol/models/paper1/checkpoints/nyu_rsna_head_only_cached_seed42_best_train_calib.pt --checkpoint-sha256 <NYU_BEST_SHA256> --max-spend-usd 4.5 --approve-eval
python scripts/evaluate_paper1_specialist_finetune.py --model nyu_fourview_rsna_finetuned --phase lock_nlbs
python scripts/evaluate_paper1_specialist_finetune.py --model nyu_fourview_rsna_finetuned --phase eval_nlbs
```

### Experiment 2 (`MODAL_PROFILE=tigerwood693`)

```powershell
$env:MODAL_PROFILE="tigerwood693"

# 2-1. GLAM Full NLBS Inference (bs=4, 4 shards) [~$1.39, ~35.3m GPU]
python scripts/run_paper1_exp2_modal.py --stage infer --model-id glam_pretrained --split nlbs_full --batch-size 4 --num-shards 4 --approve-full-infer

# 2-2. End2End CBIS-DDSM Full NLBS Inference (uncropped 1152x896, bs=32, 4 shards) [~$0.55, ~14.0m GPU]
python scripts/run_paper1_exp2_modal.py --stage infer --model-id end2end_cbis_ddsm --split nlbs_full --cache-mode uncropped_1152x896 --batch-size 32 --num-shards 4 --approve-full-infer

# 2-3. GMIC + NYU NLBS Reuse: pre-locked at reports/paper1/predictions_locked/ ($0.00)
```

### Experiment 3 (`MODAL_PROFILE=tigerwood697`)

```powershell
$env:MODAL_PROFILE="tigerwood697"

# 3-1. CMMD Ready-4 Full Inference (5,202 imgs / 1,775 exams, bs=32 across all 4) [~$0.27 GPU]
modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset cmmd --model all_ready --manifest-csv datasets/exp3_meta/cmmd_inference_manifest.csv --num-shards 4 --max-spend-usd 1.5 --approve-full-infer

# 3-2. LUMINA Ready-4 Full Inference (1,285 imgs / 468 exams, bs=32/32/128/32) [~$0.10 GPU]
modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset lumina --model all_ready --manifest-csv datasets/exp3_meta/lumina_inference_manifest.csv --num-shards 4 --max-spend-usd 1.0 --approve-full-infer

# 3-3. NLBS Ready-4 Verification & Lock Import ($0.00 CPU)
python -c "from scripts.run_paper1_exp3_modal import orchestrate_external_inference; print(orchestrate_external_inference(dataset_id='nlbs', model_arg='all_ready', skip_profile_check=True))"

# 3-4. Agent 1 Handoff (After Exp 1 Completes & Checkpoints Synced to tigerwood697)
modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset cmmd --model agent1_handoff --manifest-csv datasets/exp3_meta/cmmd_inference_manifest.csv --handoff-json <AGENT1_HANDOFF_JSON> --num-shards 4 --max-spend-usd 2.5 --approve-full-infer
modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset lumina --model gmic_rsna_finetuned --manifest-csv datasets/exp3_meta/lumina_inference_manifest.csv --handoff-json <AGENT1_HANDOFF_JSON> --num-shards 4 --max-spend-usd 1.5 --approve-full-infer

# 3-5. Isolated Multi-Dataset Evaluation (CPU, $0.00 — never pools NLBS with CMMD/LUMINA/IMDB)
python scripts/evaluate_paper1_exp3.py --config-json <EVAL_CONFIG_JSON> --out-dir reports/paper1/expanded/exp3/eval --n-boot 2000
```

---

## 4. Storage Lifecycle (`dryousufmozumder` `< 1,024 GiB` Guardrail)

| Step | Action | Delta (`GiB`) | Total Used (`GiB`) | Margin Under `1,024 GiB` |
|---|---|---:|---:|---:|
| 1 | Current Post-Wave 3 (Live Measured) | — | `963.74` | `60.26` |
| 2 | GMIC Train + `rsna_val_calib` Eval & Lock | `+0.12` | `~963.86` | `~60.14` |
| 3 | **CLEANUP GATE #1 (User-Approved):** Delete GMIC `rsna_train` + `rsna_train_calib` PNGs | `-245.23` | `~718.63` | `~305.37` |
| 4 | Transfer GMIC `nlbs` Eval Cache + GMIC `nlbs` Eval & Lock | `+97.31` | `~815.94` | `~208.06` |
| 5 | **CLEANUP GATE #2 (User-Approved):** Delete GMIC `rsna_val_calib` + `nlbs` PNGs | `-159.37` | `~656.57` | `~367.43` |
| 6 | NYU `train_all` Build + Feature Extract (`176.86 MiB`) + Head Train + `val_calib`/`nlbs_4view` Eval | `+279.44` | **`~936.01`** *(or `~930.10` w/ pilot reclaim)* | **`~87.99`** |

*Never delete `rsna/raw` (`293.11 GiB`), `stage1_*` (`101.89 GiB`), original weights, locked predictions, or selected checkpoints.* (`tigerwood693` is `339.79 GiB` and `tigerwood697` is `~243.49 GiB` — both have `> 680 GiB` headroom.)

---

## 5. Stop and Resume Procedures

- **Exp 1:** GMIC saves `_latest.pt` and `_best_train_calib.pt` per epoch (`run_paper1_gmic_finetune_modal.py`); NYU feature extraction persists per-split `.pt` caches once completed, and `--stage eval` across GMIC/NYU writes atomic per-shard caches (`shard_XX_of_04.json`) and supports `--shard-indices` retry. Always verify `modal app list` has `0` stale apps before resuming.
- **Exp 2:** `scripts/run_paper1_exp2_modal.py` saves resumable per-shard caches at `rsna-mammography-vol:/cache/paper1_exp2_shards/<model_id>/nlbs_full/shard_XX_of_04.json` keyed by `expected_unit_ids_sha256` and `checkpoint_sha256`; re-running `--stage infer` skips verified shards automatically.
- **Exp 3:** `scripts/run_paper1_exp3_modal.py` saves atomic per-shard JSONs at `rsna-mammography-vol:/cache/paper1_exp3_shards_<dataset>/<model_id>/shard_XX_of_04.json` (`validate_and_load_inference_shard` checks `checkpoint_sha256`, `preprocess_version`, and exact `expected_item_ids`). Re-running `--stage infer` resumes incomplete shards and rejects duplicates/missing IDs before `lock_external_predictions`.

---

## 6. Scientific Restrictions & Unresolved Blockers

### Scientific Restrictions
- **Exp 1 & Exp 3:** Strictly RSNA `train` (gradient updates) -> `train_calib` (epoch/checkpoint selection) -> `val_calib` (frozen `98%` primary / `95%` secondary sensitivity operating thresholds). Zero target cancer labels for training, tuning, debugging, or calibration. Predictions are SHA-256 locked BEFORE outcome joining. Never pool screening (`nlbs`) with biopsy-enriched diagnostic cohorts (`cmmd`, `lumina`) or mixed cohorts (`imdb`).
- **Exp 3 Dataset-Specific:** `cmmd` (`826` 4-view exams for common-4view comparison; `1,775` single-view) is a biopsy-enriched diagnostic stress test (never population-screening workload dismissal). `lumina` (`468` exams, neutral pseudonyms) discards contralateral views on malignant cases — **NYU Four-View is hard-blocked on `lumina`**, views are never fabricated, and selection-bias caveats apply.
- **Exp 2:** Original mammography weights only; **strictly Category C** (`AUROC`/`AUPRC` primary). Zero RSNA calibration, zero arbitrary `0.5` cutoff, and no uncalibrated dismissal claims. Enforce stricter `nyukat/GLAM` internal non-commercial research-only `LICENSE` and `CC BY-NC-SA 4.0` on `end2end_cbis_ddsm`.

### Unresolved Blockers
1. **`petrini_twoview_efnb3` (Exp 2):** `blocked_upstream` — verified upstream truncation (`88.0 MiB` + zero-padding, no ZIP EOCD) on HuggingFace & Google Drive. Excluded; no repair spend.
2. **`frcnn_cad` (Exp 2):** `disqualified` — dead ELTE URL, CUDA 8.0/Caffe incompatible. Excluded.
3. **IMDB Dataset (Exp 3):** Blocked by external manual download — requires interactive IBIA web download (`MAMOS_1000000004`, `MAMOS_1000000050`) or authorized MIDAS CC BY 4.0 access + `modal volume put` to `tigerwood697`.
4. **Exp 1 Sequential Storage Gates:** Cleanup Gates #1 (`-245.23 GiB`) and #2 (`-159.37 GiB`) require explicit user approval during Phase 2.
5. **Exp 2 Post-Lock CPU Eval Script:** `scripts/evaluate_paper1_exp2.py` (Category C AUROC/AUPRC + cluster bootstrap) should be authored once Exp 2 NLBS predictions are locked.
