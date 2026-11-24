# Paper 1 Experiment 1 — Phase 1 Readiness, Audit & Execution Plan

- **Date:** 2026-11-24
- **Branch:** `paper1-exp1-phase1`
- **Phase / Wave:** Phase 1 Wave 1 + Step A Speed/Benchmarking Prep (Zero-Cost Engineering, Audit & Read-Only Modal Checks)
- **Decision:** **CONDITIONAL GO** *(zero paid GPU, CPU, or transfer jobs launched; zero volume deletions executed)*

---

## 1. Isolation, Scientific Protocol & Compliance
- All work occurred strictly on `paper1-exp1-phase1`; `main`, `memory/`, `studies/paper1/protocol_v1.md`, and immutable Stage B / Stage C / primary 6-model outputs (`reports/paper1/six_model_final/`) are untouched.
- Registered under [`studies/paper1/specialist_finetuning_protocol_v1.md`](../../../studies/paper1/specialist_finetuning_protocol_v1.md) as **supplementary exploratory** RSNA-supervised sensitivity models (`gmic_rsna_finetuned`, `nyu_fourview_rsna_finetuned`) with **malignant-only** supervision (never fabricating benign biopsy labels) and strict target-label quarantine.

## 2. Modal Account Billing & Credit Status (`dryousufmozumder`) — MEASURED
- Verified via `modal billing summary --json`, `modal billing report --for "this month" --show-resources --json`, and `modal app list`:
  - **Running Apps:** `0`
  - **Metered Cost:** `$10.6399` (`Volumes: $6.5741`, `Ephemeral Apps: $3.8773`, `Network Egress: $0.1858`)
  - **Adjustments:** `Free Storage: -$6.5741`, `Network Egress Allowance: -$0.1858`, `Credits: -$3.8800`
  - **Billed Cost:** **`$0.00`**
  - **Estimated Remaining Starter Credits (`$30.00` monthly cap, reset `2026-12-01 00:00 UTC`):** **`~$26.12`**

## 3. Exact Volume Storage Inventory (`dryousufmozumder`) — MEASURED
Measured via read-only Modal SDK `Volume.listdir('/', recursive=True)` (`$0.00`):

| Volume | Files | Size (GiB) | Bytes | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `rsna-mammography-vol` | `214,534` | `799.5792` | `858,541,630,379` | Active Paper 1 + RSNA raw + caches |
| `nlbs-mammography-vol` | `0` | `0.0000` | `0` | Empty |
| `beef-behavior-data` | `455,890` | `56.7679` | `60,954,056,814` | Prior project |
| `sideview-data` | `160,524` | `23.4630` | `25,193,181,352` | Prior project |
| `cvb-data` | `226,344` | `14.2900` | `15,343,738,998` | Prior project |
| `reid-checkpoints` | `64` | `1.6563` | `1,778,393,533` | Prior project |
| **Total (`6` Volumes)** | **`1,057,356`** | **`895.7563`** | **`961,811,001,076`** | **`128.2437 GiB` below `1,024 GiB` free tier (`$0.00/day`)** |

<details>
<summary>Exact prefix breakdown inside rsna-mammography-vol (799.5792 GiB)</summary>

- `rsna/raw`: `54,713` files, `293.1100 GiB`
- `cache/paper1_gmic_uint16_v1/rsna_train`: `38,267` files, `214.5180 GiB`
- `cache/paper1_gmic_uint16_v1/rsna_train_calib`: `5,489` files, `30.7042 GiB`
- `cache/paper1_nyu_uint16_v1/nlbs_4view`: `24,751` files, `87.7196 GiB`
- `cache/paper1_nyu_uint16_v1/rsna_val_calib`: `9,543` files, `69.7860 GiB`
- `cache/stage1_rsna_uint16/*`: `54,709` files, `65.9501 GiB`
- `cache/stage1_nlbs_uint16/*`: `27,020` files, `35.9386 GiB`
- `models/mammo_clip/b5-model-best-epoch-7.tar`: `1` file, `1.5405 GiB`
- `models/paper1/*` (`dinov2`, `resnet50`, `sample_image_model.p`, `sample_model_1.p`): `4` files, `0.2573 GiB`
- `_transfer_manifests`: `32` files, `0.0499 GiB`
- `rsna/splits_seed42.csv`: `1` file, `0.0046 GiB`
- `models/checkpoints/*`: `4` files, `0.0004 GiB`

</details>

## 4. Source Asset Verification (`dryousufmozumder` & `hasinishrak2015`) — MEASURED
- **`dryousufmozumder` verified present:** `rsna/raw` (`293.11 GiB`), `stage1_rsna_uint16` (`65.95 GiB`), `stage1_nlbs_uint16` (`35.94 GiB`), `paper1_nyu_uint16_v1/{rsna_val_calib,nlbs_4view}` (`69.79 + 87.72 GiB`, both with `manifest.csv` & `verify_summary.json`), `paper1_gmic_uint16_v1/{rsna_train,rsna_train_calib}` (`214.52 + 30.70 GiB` PNGs + shards), `models/paper1/*`, `models/mammo_clip/*`, `models/checkpoints/*`.
- **`hasinishrak2015` GMIC eval cache verified present:** `cache/paper1_gmic_uint16_v1/rsna_val_calib` (`11,001` files / `62.0636 GiB`) + `nlbs` (`27,007` files / `97.3070 GiB`) = **`38,008` files / `159.3706 GiB`**.

## 5. Peak Storage Risk (> 1,024 GiB) & Safe Sequential Plan
- **Risk:** Adding `159.37 GiB` GMIC eval cache directly -> `1,055.13 GiB` (`+31.13 GiB` overage). Adding NYU `rsna_train` + `rsna_train_calib` (`38,124` PNGs ≈ `279.2 GiB` extrapolated at `7.324 MiB/img`) simultaneously -> `~1,334 GiB` (`+310 GiB` overage = `~$0.93/day`).
- **Safe Zero-Overage Sequence (zero unapproved deletions; cleanup ONLY after explicit approval post-checkpoint/lock):**
  1. Verify existing GMIC `train_all` PNGs (`0` GiB delta -> `895.76 GiB`).
  2. Run GMIC Wave 3 pilot + Phase 2 training -> lock best checkpoint (`895.82 GiB`). *(Or transfer `rsna_val_calib` `+62.06 GiB -> 957.82 GiB` first if desired.)*
  3. **User-approved cleanup #1:** Delete GMIC `rsna_train` + `rsna_train_calib` (`-245.22 GiB` -> **`650.60 GiB`**).
  4. Transfer GMIC eval cache `rsna_val_calib` + `nlbs` (`+159.37 GiB` -> **`809.97 GiB < 1,024 GiB`**) and run GMIC `--stage eval` + calibration + NLBS lock.
  5. **User-approved cleanup #2:** Delete GMIC eval cache (`-159.37 GiB` -> **`650.62 GiB`**).
  6. Build NYU `rsna_train` + `rsna_train_calib` PNGs (`+279.2 GiB` -> **`929.82 GiB < 1,024 GiB`**).
  7. Extract NYU `head_only` cached features (`+0.18 GiB`), train `head_only` on CPU (`< 5s`), and run NYU `--stage eval` reusing existing `157.51 GiB` NYU eval cache. *(With user approval after feature extraction, the `279.2 GiB` NYU train PNGs can also be reclaimed immediately.)*

### Sequential Cleanup Regeneration Cost & Size Verification (Zero Unapproved Deletions)
| Cache Tier (Cleanup Only After Explicit Approval) | Files | Size (GiB) | Rebuild Cost from Raw (`CPU`) | Cloud Re-Transfer Cost (`hasinishrak2015`) | Safe Gate Before Deleting |
| :--- | :---: | :---: | :---: | :---: | :--- |
| GMIC `rsna_train` + `rsna_train_calib` | `43,756` | `245.2222` | `~$3.30–$3.52` (RSNA rate) / `~$1.82` (mixed rate) | N/A | Only after best GMIC checkpoint SHA-256 locked |
| GMIC eval cache (`rsna_val_calib` + `nlbs`) | `38,008` | `159.3706` | `$1.5844` (Step 1 measured) | `$0.25–$0.38` (`$0.92` @ 50 MB/s) | Only after GMIC RSNA calib + NLBS predictions locked |
| NYU eval cache (`rsna_val_calib` + `nlbs_4view`) | `34,294` | `157.5056` | `$1.6203` net / `$1.6834` gross (Step 1 measured) | `~$0.43` (if kept on source) | Keep through NYU `--stage eval` (never delete early) |
| NYU `train_all` (`rsna_train` + `rsna_train_calib`, to build) | `~38,125` | `~279.2` (extrap.) | `~$2.20–$2.44` (extrap. `4.001x` RSNA) | N/A | Only after `176.9 MiB` `head_only` feature cache verified |

## 6. Worktree `external/` Dependency Check
- `external/GMIC` and `external/breast_cancer_classifier` are gitignored and absent in git worktree checkouts. Before launching `modal run`, run from the main repository root (or link `external/`) so `.add_local_dir('external/...')` resolves.

## 7. Critical GMIC Cache Finding (`manifest.csv` Missing)
- `dryousufmozumder:/cache/paper1_gmic_uint16_v1/{rsna_train,rsna_train_calib}` has all `43,724` PNGs + `32` shard JSONLs, **but `manifest.csv`, `fallback_log.json`, and `verify_summary.json` were not generated** (`--stage verify --split train_all` was skipped). Wave 2 runs `--stage verify --split train_all` first (`~$0.03`).

## 8. GMIC Schedule Audit (`steps_per_epoch=600, bs=8, epochs=10` vs Recommended)
- `RSNA train`: `38,251` imgs (`807` pos, `37,444` neg), `8,339` pts (`340` pos).
- **Legacy (`bs=8, n_pos=2, n_neg=6, steps=600, ep=10`, `48,000` draws):**
  - Positive exposure: `1,200` draws/ep (`1.49x`, `77.4%` unique/ep), `12,000` total (`14.87x` avg, `100%` unique).
  - Negative coverage: `3,600` draws/ep (**`9.17%` unique/ep**), `36,000` total (**only `61.77%` unique negatives seen**, `~14,316` unseen!).
- **Recommended Source-Only Schedule (`bs=16, n_pos=4, n_neg=12, steps=400, ep=8`, `51,200` draws):**
  - `1,600` pos + `4,800` neg draws/ep (`12.03%` unique neg/ep, `64.14%` unique neg in 8 ep), paired with `eval_batch_size=16`, frozen BN `.eval()` mode, and `early_stopping_patience=3` (`min_epochs=4`) to limit small-positive-set overfitting (`807` imgs).

## 9. GMIC Overhead & Throughput Audit
- Agent 4's `$6.18` projection (`5.1 img/s`, `1.078 GB` VRAM, `bs=8`) omitted:
  1. `train_calib` validation (`5,473` imgs/ep × 10 = `54,730` imgs; previously single-threaded PNG reads -> `0.47–1.40h` / `$1.10–$3.30`).
  2. Cold NVMe staging of `245.22 GiB` (`~10–17 min` / `$0.40–$0.67`).
- **Fixed in `src/paper1/gmic_finetune.py` & `scripts/run_paper1_gmic_finetune_modal.py`:**
  - Cross-batch `deque` prefetch + `ThreadPoolExecutor(8)` for both train & `train_calib`, `set_gmic_backbone_eval_mode`, and `early_stopping_patience=3` (`min_epochs=4`).
  - Progressive Wave 3 pilot (`--bench-batch-sizes "8,16,32,64,128,256" --bench-precisions "bf16,fp16,tf32"`) with clean OOM/instability/throughput-drop early stop, recording `imgs_per_sec`, `steps_per_sec`, `val_imgs_per_sec`, `peak_allocated_vram_gb`, `peak_reserved_vram_gb`, `data_load_sec`, `h2d_sec`, `fwd_bwd_sec`, `pin_memory=True`, `non_blocking=True`, and CUDA `AdamW(..., fused=True)`.
  - Fairness scaling (`scale_steps_and_lr_for_batch_size` + `apply_linear_warmup`) preserves `8 * 600 = 4,800` image draws/epoch across batch sizes (`steps_per_epoch = round(4800 / batch_size)`) with `sqrt`/`linear` LR scaling and linear warmup.

## 10. GMIC Recommended Schedule Cost Projection — EXTRAPOLATED
- At `bs=16` with async prefetch (extrapolated from `5.1 img/s` -> `6.5–10.0 img/s` train + `32.5 img/s` val + `0.27h` NVMe stage): **`~2.0–2.8h`** (**`$4.72–$6.61`**, gated on Wave 3 `bs=16` pilot).

## 11. Critical Blocker Fixed: GMIC `--stage eval`
- **Bug Fixed:** `run_gmic_finetune_l40s()` previously had no `stage=='eval'` branch and fell through into training!
- **Fix (`src/paper1/gmic_finetune.py`, `src/paper1/gmic_adapter.py`, `scripts/run_paper1_gmic_finetune_modal.py`):** added `eval_gmic_finetuned_shard` + `load_gmic_finetuned_model_for_eval` with independent checkpoint SHA-256 verification (pretrained `GMIC_CKPT_SHA256` untouched), `requires_grad=False`, zero optimizer, resumable shards (`ckpt_sha[:12]`), RSNA-first calibration lock gate, and SHA-256 locked predictions.

## 12. Authorized GMIC Eval Cache Transfer Plan (`$0.00` Plan-Only Verified)
- Added bundles `dryousuf_gmic_eval_cache`, `dryousuf_gmic_eval_rsna_val_calib`, and `dryousuf_gmic_eval_nlbs` in `src/utils/modal_transfer.py`:
  - `rsna_val_calib`: `11,001` files, `62.0636 GiB` (`$0.3576` @ 50 MB/s | `$0.1490` @ 120 MB/s | `$0.0715` @ 250 MB/s)
  - `nlbs`: `27,007` files, `97.3070 GiB` (`$0.5607` @ 50 MB/s | `$0.2336` @ 120 MB/s | `$0.1121` @ 250 MB/s)
  - **Combined (`38,008` files, `159.3706 GiB`):** **`$0.3826` @ 120 MB/s** (`$0.1836–$0.9182`).

## 13. NYU Four-View Eligibility & Critical Preprocessing Fixes
- **Verified 4-View Eligibility (`100%` of RSNA patients):** `train = 8,339/8,339` pts (`33,356` imgs, `340` pos exams), `train_calib = 1,192/1,192` pts (`4,768` imgs, `49` pos exams), `val_calib = 2,382/2,382` pts (`9,528` imgs, `97` pos exams; exact 9,528-row match vs `rsna_val_calib_4view_selection.csv`).
- **Fixes Applied:**
  - Extended `scripts/preprocess_paper1_nyu_cpu.py` + `validate_nyu_cli_stage_and_guard` for `rsna_train`, `rsna_train_calib`, and `train_all` (label-free, native `2677x1942` CC / `2974x1748` MLO, no Stage-1 upscaling).
  - Fixed fatal `png_native` vs `png` directory mismatch in `scripts/run_paper1_nyu_finetune_modal.py`.

## 14. NYU `head_only` Frozen-Backbone Feature Caching & BatchNorm Audit
- **BN & GaussianNoise Finding:** `SplitBreastModel` applies `AllViewsGaussianNoise(0.01)` and updates `FourViewResNet` `BatchNorm2d` running stats whenever `.training==True`. Added `set_nyu_backbone_eval_mode` keeping `four_view_resnet` + `all_views_gaussian_noise_layer` in `.eval()`.
- **Feature Cache Design (`src/paper1/nyu_finetune.py`, `--stage extract_head_features` + `--stage train_cached_head`):**
  - Pooled `(h_cc, h_mlo)` vector = `1,024` float32 scalars (`4 KB` / exam draw).
  - `4` augmented train draws (`8,339 × 4 × 4 KB = 130.3 MiB`) + `10`-TTA `train_calib` (`1,192 × 10 × 4 KB = 46.6 MiB`) = **`~176.9 MiB` total** (vs `~279 GiB` PNG cache!).
  - Exact parity (`< 1e-6` diff vs `SplitBreastModel.forward` with `.eval()` backbone, verified by `verify_nyu_multidraw_feature_equivalence` with frozen `BatchNorm2d` running stats and `.eval()` `AllViewsGaussianNoise(0.01)`) and **`>100x` training speedup** (`< 5s` on CPU for 8 epochs, `$0.00` local / `~$0.0001` Modal CPU).
  - Added progressive NYU exam-batch-size pilot (`--bench-batch-exams "2,4,8,16"`) in `scripts/run_paper1_nyu_finetune_modal.py` benchmarking both `extract` (`1`-draw) and `eval` (`10`-TTA) throughput, VRAM (`allocated` & `reserved`), and `data_load_sec`/`h2d_sec`/`fwd_sec` with `pin_memory=True`.

## 15. Critical Blocker Fixed: NYU `--stage eval`
- **Bug Fixed:** `run_nyu_finetune_l40s()` previously ignored `stage=='eval'` and fell through into full training!
- **Fix:** Added `eval_nyu_finetuned_shard` + `load_nyu_finetuned_model_for_eval` preserving `10`-TTA (`seed=0`), strict FP32 non-autocast `eval()`, `requires_grad=False`, zero optimizer, `max(left_malignant, right_malignant)`, `NYU_STEP1_MANIFEST_SHA256` checks, resumable shards, RSNA-first calibration lock gate, and SHA-256 locked predictions.

## 16. Unified Supplementary Calibration, Locking & Evaluation CLI
- Added [`src/paper1/specialist_eval.py`](../../../../src/paper1/specialist_eval.py) and [`scripts/evaluate_paper1_specialist_finetune.py`](../../../../scripts/evaluate_paper1_specialist_finetune.py) (`--phase calibrate_rsna`, `lock_nlbs`, `eval_nlbs`) enforcing RSNA-only `98%/95%` threshold freeze and pre-join SHA-256 locks before NLBS outcome evaluation (`B=2,000` cluster bootstrap).

## 17. Local Unit Test Verification (After Final Edit)
- `python -m py_compile src/paper1/gmic_finetune.py src/paper1/nyu_finetune.py scripts/run_paper1_gmic_finetune_modal.py scripts/run_paper1_nyu_finetune_modal.py tests/test_paper1_specialist_finetuning.py`: **PASS**
- `python -m unittest discover -s tests -p "test_paper1_*.py" -v`: **`47 / 47` runnable PASS** (`55` total discovered, `8` skipped solely because gitignored `external/` repos & local weights are absent in worktree; includes **`14 / 14` PASS** in `tests/test_paper1_specialist_finetuning.py` and `5 / 5` runnable + `1` skipped in `tests/test_paper1_nyu_step1_cpu.py`).
- `python -m unittest discover -s tests -p "test_transfer_*.py" -v`: **`8 / 8` PASS**.

## 18. Section 9 — Consolidated Wave 2 + Wave 3 Approval Table (`dryousufmozumder`)
*Current remaining Starter credits: `~$26.12 / $30.00`. Current storage: `895.7563 GiB / 1,024 GiB` (`$0.00/day`).*

| Workload | Account | Expected CPU / GPU Runtime | Expected Cost | Max Authorized Spend | Storage Impact | Timeouts & Cancellation |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Wave 2A:** GMIC `train_all` `--stage verify --num-shards 16` | `dryousufmozumder` | `~6–8 min` total CPU-shard time (`~1 min` wall w/ 16 shards @ 2 vCPU) | `~$0.03` | `$0.20` | `+0.02 GiB` (`895.78 GiB`, `< 1 TiB`) | `1,800s` per shard; `Ctrl+C` or `modal app stop` |
| **Wave 2B:** GMIC Eval Cache Transfer (`dryousuf_gmic_eval_rsna_val_calib`, RSNA-first `< 1 TiB`) | `dryousufmozumder` *(src: `hasinishrak2015`)* | `~5–9 min` wall (`16` containers × `6` streams @ 120–181 MB/s) | `$0.10–$0.15` (`$0.36` @ 50 MB/s) | `$0.60` (`--max-budget-usd 0.60`) | `+62.06 GiB` (`957.84 GiB`, `< 1 TiB`) | `1,800s` + `--max-budget-usd 0.60` hard stop; `Ctrl+C` / `modal app stop` |
| **Wave 2C *(Deferred until post-GMIC-eval reclaim OR overage opt-in)*:** NYU `train_all` CPU PNGs (`38,124` imgs) | `dryousufmozumder` | `~8–10 CPU-hrs` (`~25–35 min` wall w/ 24 shards @ 4 vCPU/8 GiB) | `~$2.20–$2.44` | `$3.20` (`--approve-full-build`) | `+279.2 GiB` (`929.8 GiB` after GMIC cleanup; `~1,237 GiB` if uncleaned) | `3,600s`/shard + `--max-spend-usd`; `Ctrl+C` / `modal app stop` |
| **Wave 3A:** GMIC L40S Progressive Pilot (`--bench-batch-sizes "8,16,32,64,128,256" --bench-precisions "bf16,fp16,tf32"`) | `dryousufmozumder` | `~3–7 min` L40S wall (OOM/drop early stops) | `~$0.12–$0.28` | `$0.75` (`--max-spend-usd 0.75`) | `+0.001 GiB` (`gmic_pilot_benchmark_report.json`) | `14,400s` container + `--max-spend-usd` + OOM/drop early exit |
| **Wave 3B:** NYU L40S Progressive Pilot (`--bench-batch-exams "2,4,8,16"` extract & 10-TTA + equivalence check) | `dryousufmozumder` | `~2–4 min` L40S wall | `~$0.08–$0.16` | `$0.60` (`--max-spend-usd 0.60`) | `+0.001 GiB` (`nyu_pilot_benchmark_report.json`) | `14,400s` container + `--max-spend-usd` + OOM/drop early exit |
| **Wave 2A+2B + Wave 3 Total (Immediate Approval Gate)** | `dryousufmozumder` | **`~15–28 min` wall** | **`~$0.33–$0.62`** | **`$2.15` cap** | **`957.84 GiB` (`66.16 GiB` < 1 TiB, `$0.00/day`)** | All jobs bounded by `--max-spend-usd` / `--max-budget-usd` |
| *Phase 2 Projection (Gated on Wave 3 Pilots): GMIC + NYU Train/Extract/Eval* | `dryousufmozumder` | *`~2.5–4.0h` L40S* | *`~$11–$15` + Wave 2C `$2.25`* | *Gated post-Wave 3* | *Peak `<= 957.84 GiB` sequential* | `--max-spend-usd` per command + `--approve-full-train` / `--approve-eval` |

## 19. Measured vs. Extrapolated Cost Ledger (`dryousufmozumder`)

| Stage | Basis | Measured / Extrapolated | Projected Cost (USD) |
| :--- | :--- | :--- | :---: |
| Wave 1 + Step A Prep | Actual | **MEASURED** | **`$0.00`** |
| Wave 2A: GMIC `train_all` `--stage verify` | Scaled from `rsna_val_calib` verify | **EXTRAPOLATED** | `~$0.03` |
| Wave 2B: GMIC Eval Transfer (`rsna_val_calib`, `62.06 GiB`) | `--plan-only` @ 120–181 MB/s | **EXTRAPOLATED** | `$0.10–$0.15` (`+ $0.23` for NLBS later) |
| Wave 2C: NYU `train_all` CPU PNGs (`38,124` imgs) | Scaled `4.001x` from `9,528`-img RSNA (`$0.5488`) | **EXTRAPOLATED** | `~$2.20–$2.44` |
| Wave 3: GMIC (`bs=8..256`) + NYU (`be=2..16`) L40S Pilots | Progressive sweep w/ OOM/drop stop | **EXTRAPOLATED** | `~$0.20–$0.44` |
| Phase 2A: GMIC Fine-Tune (optimal bs/prec) + Eval (`bs=1`) | Gated on Wave 3A pilot (`$4.72–$6.61` train @ bs16, potentially faster @ larger bs/bf16) + `$1.81` eval | **EXTRAPOLATED** | `~$4.50–$8.46` |
| Phase 2B: NYU Feature Extract + Head Train + Eval (`10-TTA`) | Step-2 NYU (`$0.92 + $2.30` @ `hasinishrak2015`) | **EXTRAPOLATED** | `~$4.35` |
| **Total Wave 2 + Wave 3 + Phase 2** | — | **EXTRAPOLATED** | **`~$11.50–$16.00`** *(fits `$26.12` credits)* |

## 19. Wave 2 Commands (Download / Verify / CPU Preprocessing — DO NOT RUN UNAPPROVED)
```powershell
# Ensure running from workspace root with `external/` present
$env:MODAL_PROFILE="dryousufmozumder"

# 1. Verify existing 43,724 GMIC train/train_calib PNGs and write manifest.csv (~$0.03)
modal run scripts/preprocess_paper1_gmic_cpu.py --stage verify --split train_all --num-shards 16

# 2. Optional RSNA-first GMIC eval cache transfer (62.06 GiB -> 957.82 GiB < 1 TiB; ~$0.15)
$env:MODAL_SRC_PROFILES="hasinishrak2015"
modal --profile dryousufmozumder run scripts/transfer_modal_volumes_cloud.py --dst-profile dryousufmozumder --bundle dryousuf_gmic_eval_rsna_val_calib --num-containers 16 --streams-per-container 6 --max-budget-usd 0.60

# 3. NYU train + train_calib native CPU PNG build (run AFTER reclaiming GMIC caches OR if overage approved; ~$2.25)
modal run scripts/preprocess_paper1_nyu_cpu.py --stage build --split train_all --num-shards 24 --compression 1 --approve-full-build
```

## 20. Wave 3 Commands (Mandatory Progressive L40S Pilots Before Full Training)
```powershell
$env:MODAL_PROFILE="dryousufmozumder"
modal run scripts/run_paper1_gmic_finetune_modal.py --stage pilot --strategy partial --bench-batch-sizes "8,16,32,64,128,256" --bench-precisions "bf16,fp16,tf32" --lr-scaling sqrt --warmup-steps 100 --epochs 8 --max-spend-usd 0.75
modal run scripts/run_paper1_nyu_finetune_modal.py --stage pilot --strategy head_only --bench-batch-exams "2,4,8,16" --epochs 8 --steps-per-epoch 400 --max-spend-usd 0.60
```

## 21. Phase 2 Commands (Full Training, RSNA Calibration & NLBS Evaluation)
```powershell
$env:MODAL_PROFILE="dryousufmozumder"

# --- Track A: GMIC Fine-Tuning & Evaluation ---
modal run scripts/run_paper1_gmic_finetune_modal.py --stage train --strategy partial --batch-size 16 --eval-batch-size 16 --epochs 8 --steps-per-epoch 400 --early-stopping-patience 3 --max-spend-usd 9.0 --approve-full-train
# (After user-approved cleanup of GMIC train cache, transfer dryousuf_gmic_eval_cache or nlbs)
modal run scripts/run_paper1_gmic_finetune_modal.py --stage eval --split rsna_val_calib --batch-size 1 --num-shards 4 --checkpoint-path /rsna_vol/models/paper1/checkpoints/gmic_rsna_partial_seed42_best_train_calib.pt --checkpoint-sha256 <GMIC_BEST_SHA256> --max-spend-usd 1.5 --approve-eval
python scripts/evaluate_paper1_specialist_finetune.py --model gmic_rsna_finetuned --phase calibrate_rsna --expected-checkpoint-sha256 <GMIC_BEST_SHA256>
modal run scripts/run_paper1_gmic_finetune_modal.py --stage eval --split nlbs --batch-size 1 --num-shards 4 --checkpoint-path /rsna_vol/models/paper1/checkpoints/gmic_rsna_partial_seed42_best_train_calib.pt --checkpoint-sha256 <GMIC_BEST_SHA256> --max-spend-usd 2.0 --approve-eval
python scripts/evaluate_paper1_specialist_finetune.py --model gmic_rsna_finetuned --phase lock_nlbs
python scripts/evaluate_paper1_specialist_finetune.py --model gmic_rsna_finetuned --phase eval_nlbs

# --- Track B: NYU Four-View Fast Feature-Cached Head Fine-Tuning & 10-TTA Evaluation ---
modal run scripts/run_paper1_nyu_finetune_modal.py --stage extract_head_features --batch-exams 4 --max-spend-usd 3.0 --approve-full-train
modal run scripts/run_paper1_nyu_finetune_modal.py --stage train_cached_head --epochs 8 --batch-exams 16 --steps-per-epoch 400 --max-spend-usd 0.5 --approve-full-train
modal run scripts/run_paper1_nyu_finetune_modal.py --stage eval --split rsna_val_calib --batch-exams 4 --num-shards 4 --checkpoint-path /rsna_vol/models/paper1/checkpoints/nyu_rsna_head_only_cached_seed42_best_train_calib.pt --checkpoint-sha256 <NYU_BEST_SHA256> --max-spend-usd 2.0 --approve-eval
python scripts/evaluate_paper1_specialist_finetune.py --model nyu_fourview_rsna_finetuned --phase calibrate_rsna --expected-checkpoint-sha256 <NYU_BEST_SHA256>
modal run scripts/run_paper1_nyu_finetune_modal.py --stage eval --split nlbs_4view --batch-exams 4 --num-shards 4 --checkpoint-path /rsna_vol/models/paper1/checkpoints/nyu_rsna_head_only_cached_seed42_best_train_calib.pt --checkpoint-sha256 <NYU_BEST_SHA256> --max-spend-usd 3.5 --approve-eval
python scripts/evaluate_paper1_specialist_finetune.py --model nyu_fourview_rsna_finetuned --phase lock_nlbs
python scripts/evaluate_paper1_specialist_finetune.py --model nyu_fourview_rsna_finetuned --phase eval_nlbs
```

## 22. Checkpoint Handoff Specification for Agent 3
- **GMIC Fine-Tuned Checkpoint:** `rsna-mammography-vol:/models/paper1/checkpoints/gmic_rsna_partial_seed42_best_train_calib.pt` (`schema_version = paper1_gmic_finetune_ckpt_v1`, `model_id = gmic_rsna_finetuned`).
- **NYU Fine-Tuned Checkpoint:** `rsna-mammography-vol:/models/paper1/checkpoints/nyu_rsna_head_only_cached_seed42_best_train_calib.pt` (`schema_version = paper1_nyu_finetune_ckpt_v1`, `model_id = nyu_fourview_rsna_finetuned`).
- **Locked Calibration & Prediction Directory (`reports/paper1/expanded/exp1/`):**
  - `<model_id>_rsna_calibration.json` & `<model_id>_rsna_calibration_lock.json`
  - `predictions_locked/<model_id>_locked_nlbs_predictions.csv` & `predictions_locked/<model_id>_lock_manifest.json`
  - `<model_id>_nlbs_evaluation.json`

## 23. Risks & Mitigations
| Risk | Mitigation |
| :--- | :--- |
| Exceeding `1,024 GiB` free storage tier (`$0.003/GiB-day`) | Sequential `< 1 TiB` staging/reclaim sequence (`<= 957.82 GiB` peak). |
| Gitignored `external/` missing in worktree | Launch `modal run` from main repo root or symlink `external/`. |
| GMIC overfitting on repeated `807` positives | Differential backbone LR (`0.1x`), `train_calib` selection, early stopping (`patience=3`), frozen BN `.eval()`. |
| NYU BN/GaussianNoise drift during `head_only` | `set_nyu_backbone_eval_mode` + bit-exact frozen feature caching. |

## 24. Final Gate Recommendation: **CONDITIONAL GO**
- Ready to execute Wave 2 (`--stage verify --split train_all`, `dryousuf_gmic_eval_rsna_val_calib` transfer) and Wave 3 L40S pilots (`~$0.30` total) upon explicit approval, provided:
  1. `modal run` executes from a directory containing `external/GMIC` & `external/breast_cancer_classifier`.
  2. Sequential storage cleanup gates (post-GMIC-train & post-GMIC-eval) are explicitly approved to keep storage `< 1,024 GiB` (`$0.00/day`).
