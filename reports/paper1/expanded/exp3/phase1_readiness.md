# Paper 1 Exp 3 — Phase 2 Readiness Audit & Orchestration Release

**Branch:** `paper1-exp3-phase1` (synced with `main` commit `465d3d4`)  
**Modal Target Profile:** `tigerwood697` (explicit `MODAL_PROFILE=tigerwood697`; `modal profile activate` NEVER called)  
**Status:** **PHASE 2 READY FOR 4 READY MODELS ON CMMD + LUMINA + NLBS REUSE (PENDING EXECUTION APPROVAL)**

---

## 1. Live Read-Only Modal Audit (`tigerwood697`)

- **Apps Running:** `0` (`modal app list` clean).
- **Billing (`modal billing summary`):** Metered `$9.48` (`Ephemeral Apps: $5.45`, `Volumes: $3.99`, `Network Egress: $0.04`), Credits `-$5.45`, Free Storage `-$3.99`, Egress Allowance `-$0.04` -> **Billed Cost: `$0.00`**, remaining monthly credits **`~$24.55`**.
- **Storage (`~243.49 GiB` total across 7 volumes `< 1,024 GiB` free tier -> `$0.00/day`):**
  - `rsna-mammography-vol`: **`180.22 GiB` (`193,506,112,761` bytes)** across 7 volumes total `~243.49 GiB`.
- **Verified Model Assets & Caches on `rsna-mammography-vol`:**
  - `ce_b5` / `trust_b5`: `models/mammo_clip/b5-model-best-epoch-7.tar` (`1,654,145,453` B) + `ce_rsna_seed42_final_epoch20.pt` (`99,040` B) + `trust_rsna_seed42_final_epoch20.pt` (`100,128` B) — SHA-256 verified (`reports/transfers/tigerwood697_exp3_backbones_full_20261124T151605Z_execution.json`: `all_verified: true`).
  - `dinov2_rsna`: `models/paper1/dinov2_vits14_pretrain.pth` (`88,283,115` B) + `dinov2_vits14_head_seed42.pt` (`7,072` B).
  - `resnet50_rsna`: `models/paper1/resnet50-11ad3fa6.pth` (`102,540,417` B) + `resnet50_rsna_seed42_best_train_calib.pt` (`282,593,192` B) + `latest.pt` (`282,594,536` B).
  - Wave 2 Stage-1 PNG caches (`1520x912 uint16`): **CMMD `5,202` PNGs** (`9.9146 GiB`, `0` skipped, `0` fallbacks) and **LUMINA `1,285` PNGs** (`6.4424 GiB`, `0` skipped, `0` fallbacks), plus preserved RSNA (`54,706`) & NLBS (`26,988`) caches.
  - **Pending Agent 1 Handoff:** Base NYU/GMIC weights (`sample_image_model.p`, `sample_model_1.p`) are present; waiting on Agent 1 fine-tuned RSNA checkpoints, SHA-256s, and frozen `val_calib` thresholds (`--handoff-json`).

---

## 2. Critical Phase 2 Blocker Fixed (`--stage infer` Orchestration)

Upgraded `scripts/run_paper1_exp3_modal.py`, `src/paper1/universal_eval.py`, and `scripts/evaluate_paper1_exp3.py`:
- **Full `run_cli` + `orchestrate_external_inference`:** Supports `--stage dry_run|preprocess|pilot|infer` across `--model all_ready|agent1_handoff|all|<id>`, comma-separated model IDs, `--handoff-json`, `--cohort-mode full|common_4view`, `--out-lock-dir`, `--num-shards`, and `--max-spend-usd`.
- **Hard Scientific Eligibility Enforcement:**
  - `nyu_fourview_rsna_finetuned` and `--cohort-mode common_4view` are **hard-blocked on `lumina`** (malignant cases discard contralateral views: `4view_pos=1/174`, never fabricate views). Single-view models run across all `1,285` LUMINA images (`468` exams, `218` pos) with explicit selection-bias caveats.
  - `nyu_fourview_rsna_finetuned` on `cmmd` automatically restricts to the **826 complete 4-view examinations** (`3,304` images, `765` pos) via `select_common_4view_cohort`, while single-view models evaluate all `1,775` exams (`5,202` images, `1,310` pos) strictly as a biopsy-enriched diagnostic stress test (`population_screening_dismissal_allowed=False`).
- **Deterministic Sharding, Integrity & Pre-Join Locking:** Validates checkpoint SHA-256 + exact expected item IDs per shard (rejecting missing or duplicate IDs), supports cache resume, aggregates image scores -> exam max/mean scores, locks SHA-256 pre-join predictions under `reports/paper1/expanded/exp3/locked_preds/<dataset>/<model>/`, and reuses NLBS at **`$0.00`** (`6,184` common 4-view / `6,359` full).

---

## 3. Verified Wave 3 Pilot Winners & Projected Phase 2 Cost (`tigerwood697`, L40S)

| Dataset | Model | Winner BS | Throughput (`img/s`) | Peak VRAM (GB) | Proj. Time (s) | Proj. GPU Cost (USD) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CMMD** (`5,202`) | `ce_b5` | `32` | `30.10` | `10.032` | `172.8` | `$0.1134` | `bs=64` OOM/`canUse32BitIndexMath` |
| **CMMD** (`5,202`) | `trust_b5` | `32` | `30.56` | `10.031` | `170.2` | `$0.1117` | `bs=64` OOM/`canUse32BitIndexMath` |
| **CMMD** (`5,202`) | `dinov2_rsna` | `32` | `163.45` | `0.625` | `31.8` | `$0.0209` | Parity clean (`max_diff=0.0005`) |
| **CMMD** (`5,202`) | `resnet50_rsna` | `32` | `139.76` | `1.611` | `37.2` | `$0.0244` | `bs=64/128` failed `1e-3` parity (`0.00195`) |
| **LUMINA** (`1,285`) | `ce_b5` | `32` | `23.58` | `10.031` | `54.5` | `$0.0357` | Neutral pseudonyms verified |
| **LUMINA** (`1,285`) | `trust_b5` | `32` | `24.35` | `10.031` | `52.8` | `$0.0346` | Neutral pseudonyms verified |
| **LUMINA** (`1,285`) | `dinov2_rsna` | `128` | `61.97` | `2.230` | `20.7` | `$0.0136` | Zero parity drift |
| **LUMINA** (`1,285`) | `resnet50_rsna` | `32` | `62.73` | `1.611` | `20.5` | `$0.0134` | Winner parity-safe |

- **Total Projected Phase 2 (4 Ready Models on CMMD + LUMINA + NLBS `$0` reuse):** **`$0.3677` pure GPU (~`$0.42` with container startup/commit overhead, ~9–11 min serial or ~3 min w/ 4 shards)**.
- **Agent 1 Handoff Increment (`gmic` + `nyu` on CMMD, `gmic` only on LUMINA):** **`~$1.50–$2.50`**.
- **Conditional IMDB (once manually downloaded & uploaded):** **`~$1.60–$2.80`** (`12,674` DICOMs, `r1/r2` overlap + FFDM/SM separation).

---

## 4. Exact IMDB Manual Download Instructions & Executable Phase 2 Commands

### IMDB Manual Download (No Login Required)
1. Open `https://ibdc.dbt.gov.in/ibia/studybrowse/` and search `mammo` — or visit **IMDB r1.0** (`https://ibdc.dbt.gov.in/ibia/study_details_browse/MAMOS_1000000004/`, `583` pts) and **IMDB r2.0** (`https://ibdc.dbt.gov.in/ibia/study_details_browse/MAMOS_1000000050/`, `2,636` pts).
2. Click **Data** on both cards to download DICOMs + metadata CSVs locally (or request CC BY 4.0 scripted mirror via `https://www.midas.iisc.ac.in/fe/datasets/breast/breast-mammography`).
3. Upload to `tigerwood697`:
   ```powershell
   $env:MODAL_PROFILE='tigerwood697'
   modal volume put rsna-mammography-vol <LOCAL_IMDB_R1_DIR> /external_datasets/imdb/raw/r1/
   modal volume put rsna-mammography-vol <LOCAL_IMDB_R2_DIR> /external_datasets/imdb/raw/r2/
   ```

### Executable Phase 2 Commands (`MODAL_PROFILE=tigerwood697`)
```powershell
# 1. CMMD (all 4 ready models, 4 shards, verified batch size 32)
$env:MODAL_PROFILE='tigerwood697'; modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset cmmd --model all_ready --manifest-csv datasets/exp3_meta/cmmd_inference_manifest.csv --num-shards 4 --max-spend-usd 1.5 --approve-full-infer

# 2. LUMINA (all 4 ready models, 4 shards, verified batch sizes 32/32/128/32)
$env:MODAL_PROFILE='tigerwood697'; modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset lumina --model all_ready --manifest-csv datasets/exp3_meta/lumina_inference_manifest.csv --num-shards 4 --max-spend-usd 1.0 --approve-full-infer

# 3. NLBS ($0.00 lock verification from completed six_model_final predictions)
python -c "from scripts.run_paper1_exp3_modal import orchestrate_external_inference; print(orchestrate_external_inference(dataset_id='nlbs', model_arg='all_ready', skip_profile_check=True))"

# 4. Agent 1 Handoff (CMMD GMIC+NYU 4-view 826 exams; LUMINA GMIC only — NYU hard-blocked)
$env:MODAL_PROFILE='tigerwood697'; modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset cmmd --model agent1_handoff --manifest-csv datasets/exp3_meta/cmmd_inference_manifest.csv --handoff-json <AGENT1_HANDOFF_JSON> --num-shards 4 --max-spend-usd 2.5 --approve-full-infer
$env:MODAL_PROFILE='tigerwood697'; modal run scripts/run_paper1_exp3_modal.py --stage infer --dataset lumina --model gmic_rsna_finetuned --manifest-csv datasets/exp3_meta/lumina_inference_manifest.csv --handoff-json <AGENT1_HANDOFF_JSON> --num-shards 4 --max-spend-usd 1.5 --approve-full-infer

# 5. Isolated Multi-Dataset Evaluation (never pooled)
python scripts/evaluate_paper1_exp3.py --eval-specs-json <EVAL_SPECS_JSON> --out-dir reports/paper1/expanded/exp3/eval --n-boot 2000
```
