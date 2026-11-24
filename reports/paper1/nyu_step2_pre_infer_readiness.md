# Paper 1 — NYU Step 2 Production Corrections, Pilot Reconciliation & Final Readiness

- **Date:** 2026-11-24 (`12:37 +06:00` / `06:37 UTC`)
- **Status:** **READY FOR STAGE-GATED STEP 2 EXECUTION** (`RSNA Inference -> RSNA-Only Calibration Lock -> NLBS Inference`; zero GPU runs launched, zero caches deleted, zero NLBS cancer labels accessed)

---

## 1. Pilot Run Reconciliation & Latest Modal Credits (`hasinishrak2015`)

- **Run `ap-1WtcQA2ZWaNFBzKQy5CFAU` (Real Executed Step 2 Pilot):** Verified in `modal billing report --for today --show-resources`: `L40S: USD 0.068675` + `CPU: USD 0.016253` + `Memory: USD 0.006762` = **`USD 0.09169` settled** (`126.78s` L40S, `1.84 exams/sec`, `400` cached PNGs / `100` exams).
- **Run `ap-f195vW40fG2W6K2G7n0Q78`:** **Non-existent / `$0.0000` billed** (confirmed absent from `modal billing report --for today --show-resources`; `ephemeral_apps` delta before/after the pilot was `14.377985 - 14.286294 = USD 0.09169` exact).
- **Combined Actual Pilot Spend:** **`USD 0.0917`**.
- **Latest Starter Credits (`modal billing summary --for "this month"`):** `USD 30.00 - USD 22.50 =` **`USD 7.50` remaining** (`billed_cost: USD 0.00`, resets **`2026-12-01` `00:00 UTC`**; `~6.72` days left).

---

## 2. Production Hardening & Guardrails Applied

1. **Worker Spec Upgraded (`scripts/run_paper1_nyu_modal.py`):** `infer_nyu_shard` now allocates **`gpu="L40S", cpu=8.0, memory=32768` (`32 GiB RAM`), `ephemeral_disk=524288`, `timeout=7200`**, default **`batch_exams=4`**. Hourly base rate `L40S_PROD_HOURLY_USD = 2.5844/hr` (`$1.95 L40S + $0.3784 8-vCPU + $0.2560 32-GiB RAM`, conservative `~USD 2.68/hr` w/ multi-threaded CPU burst).
2. **Aggregate Spending Protection Across Shards:** Pre-launch split estimate check (`1.60 exams/sec` conservative + `35s`/shard) plus wave-based execution (`--wave-size 2`) calling `check_nyu_aggregate_spend_guard` against `--max-total-spend-usd` (`USD 1.50` default for RSNA; `USD 3.50` default for NLBS) between waves.
3. **Strict Provenance & Scientific Protocol Enforcement (`src/paper1/nyu_fourview_adapter.py`):**
   - Frozen Step-1 `manifest.csv` SHA-256 hashes hardcoded and checked both on Volume and locally (`rsna_val_calib`: `bde6ae3fa12cff0c0d121de2b1344ddc775f1020b3645670b4d62ca00db8f866`; `nlbs_4view`: `e941df848f2baeac58bbe35b11e4b5ec4e30074c28b893b59f18258008beec65`).
   - Verified 4-view identity match (`verify_nyu_step1_against_frozen_source`), `NYU_CKPT_SHA256`, `num_tta_epochs == 10`, `tta_seed == 0`, strict FP32 non-autocast `eval()` assertion (`precision == "fp32"`), and `finalize_and_lock_nyu_predictions`.
4. **RSNA-First Calibration & Lock Gate (`scripts/calibrate_paper1_nyu_rsna.py`):**
   - `--stage infer --split nlbs_4view` enforces presence and valid SHA-256 of `reports/paper1/nyu_rsna_calibration_lock.json` before NLBS GPU containers launch.
5. **Local CPU Tests (`9 / 9` PASS):** `tests/test_paper1_nyu_step1_cpu.py` (`5/5`) + `tests/test_paper1_nyu_step2.py` (`4/4`, testing FP32 enforcement, frozen Step-1 SHA rejection, wave aggregate budget cutoff, and end-to-end RSNA-only calibration locking).

---

## 3. Conservative Cost Projections & Exact Execution Commands

| Split | Exams | Pilot Rate (`1.84 ex/s`, `USD 2.5844/hr`) | Conservative (`1.60 ex/s`, `USD 2.68/hr` w/ burst) | Fits Nov w/ `USD 1.50` Buffer? (`USD 7.50` left, `~USD 3.26` storage) |
| :--- | :---: | :---: | :---: | :--- |
| **`rsna_val_calib`** (`4` shards) | `2,382` | `0.3985h` (`USD 1.03`) | `0.4524h` (**`USD 1.21`**) | **YES** (`~USD 3.03` buffer left) |
| **`nlbs_4view`** (`4` shards) | `6,184` | `0.9725h` (`USD 2.51`) | `1.1114h` (**`USD 2.98`**) | **After `2026-12-01` reset** *(or immediate ONLY if reclaiming `157.5 GiB` NYU cache after locking)* |
| **Combined Total** (`8` shards) | `8,566` | `1.3710h` (`USD 3.54`) | `1.5638h` (**`USD 4.19`**) | Run **Step 1 (RSNA)** now, **Step 3 (NLBS)** Dec 1 (or reclaim cache post-lock) |

### Exact Staged Commands (Run Only After Explicit Approval)
1. **RSNA Full Step 2 GPU Inference (`~USD 1.03–1.21`):**
   ```powershell
   $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_nyu_modal.py --stage infer --split rsna_val_calib --num-shards 4 --batch-exams 4 --wave-size 2 --max-total-spend-usd 1.50 --approve-full-infer
   ```
2. **RSNA-Only Threshold Calibration & Lock (`USD 0.00`, CPU Only):**
   ```powershell
   python scripts/calibrate_paper1_nyu_rsna.py --expected-batch-exams 4
   ```
3. **NLBS Full Step 2 GPU Inference (`~USD 2.51–2.98`, Blocked Until #2 Passes):**
   ```powershell
   $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_nyu_modal.py --stage infer --split nlbs_4view --num-shards 4 --batch-exams 4 --wave-size 2 --max-total-spend-usd 3.50 --approve-full-infer
   ```
