# Paper 1 — NYU Four-View Step 2 L40S GPU Pilot & Benchmark Report

- **Date:** 2026-11-24 (`12:24 +06:00` / `06:24 UTC`)
- **Modal Profile & Run:** `hasinishrak2015` / [`ap-1WtcQA2ZWaNFBzKQy5CFAU`](https://modal.com/apps/hasinishrak2015/main/ap-1WtcQA2ZWaNFBzKQy5CFAU) (`1x NVIDIA L40S 48 GB`, `6 vCPU`, `24 GiB RAM`, **NO training, NO label access, NO cache deletions**)
- **Result:** **PASS** — `400 / 400` cached images (`100` 4-view exams incl. ALL `34` RSNA fallback exams) scored at **`1.84 exams/sec`** (`7.36 imgs/sec` E2E; `2.3x` faster than `batch_exams=1`), **`4.306e-06`** max absolute parity difference vs both unbatched `score_nyu_fourview_exam` and online raw DICOM preprocessing, and **`USD 0.0695` settled Modal spend** (`<< USD 0.20` limit).

---

## 1. Optimization Architecture & Preservation Checks

- **Modified Pipeline:** [`scripts/run_paper1_nyu_modal.py`](../../scripts/run_paper1_nyu_modal.py) + [`src/paper1/nyu_fourview_adapter.py`](../../src/paper1/nyu_fourview_adapter.py)
  - Consumes the verified Step-1 lossless `uint16` PNG cache (`/rsna_vol/cache/paper1_nyu_uint16_v1/{rsna_val_calib,nlbs_4view}/`) with zero repeated DICOM decoding or CC cropping.
  - Stages shard PNGs to local container NVMe (`/tmp`) in parallel (`ThreadPoolExecutor(max_workers=16)`), verifies shape + pixel `u16_sha256` on decode, prefetches per-exam `10`-TTA tensor stacks (`ThreadPoolExecutor(max_workers=6)`, bounded queue) with **independent per-exam `np.random.RandomState(seed=0)`** so TTA random crops are **100% invariant to `batch_exams`**, executes FP32 `eval()` forward passes batched across `batch_exams`, validates strict shard provenance (`validate_nyu_shard_payload` + `finalize_and_lock_nyu_predictions`), and enforces hard cost guards (`max_pilot_cost_usd` / `--approve-full-infer` / `max_shard_cost_usd`).
- **Official Parity & Unit Verification (`8 / 8` Local CPU Tests PASS):**
  - `tests/test_paper1_nyu_step1_cpu.py`: **`5 / 5` PASS** (`52.342s`)
  - `tests/test_paper1_nyu_step2.py`: **`3 / 3` PASS** (`65.588s` initial / `95.621s` run — batched vs `bs=1` `10`-TTA parity on `sample_data`, shard provenance & tamper rejection, CLI approval & `num_tta_epochs=10` guard).

---

## 2. Live L40S Pilot Benchmark & Parity Results (`ap-1WtcQA2ZWaNFBzKQy5CFAU`)

### A. Pilot Scope & Output Validity
- **Checkpoint:** `sample_image_model.p` (`sha256: 9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014`, strict `SplitBreastModel(1)` FP32).
- **Sample Composition (`100` 4-view exams / `400` cached PNGs):**
  - `50` RSNA `val_calib` exams (`200` imgs — includes **ALL `34` RSNA cropping fallback exams** + `16` diverse `M1`/`M2` standard exams).
  - `50` NLBS `4-view` exams (`200` imgs, sampled uniformly across `6,184` studies; label-free).
- **Numerical Sanity:** `all_scores_finite: true` (`min: 0.010808`, `median: 0.060193`, `max: 0.707167`); `sample0_10tta_l40s` `exam_score = 0.01794210`.

### B. Exact Parity Measurements (`10` TTA, `seed=0`, FP32)
1. **Batched (`batch_exams=4`) Cached vs Original Unbatched (`batch_exams=1`) `score_nyu_fourview_exam` (`12` exams = `8` RSNA fallback exams incl. 1–3 fallbacks each + `4` NLBS):**
   - `max_abs_diff_exam_score`: **`4.306e-06`**
   - `max_abs_diff_all4_classes`: **`6.199e-06`**
2. **Batched (`batch_exams=4`) Cached PNG vs Original Online Raw DICOM `extract_nyu_cropped_view` + `score_nyu_fourview_exam` (`6` exams = RSNA `12943`, `14995`, `1625` w/ fallbacks + `3` NLBS):**
   - `max_abs_diff_vs_raw_online`: **`4.306e-06`** (`1.34e-07` on `12943`, `1.11e-06` on `14995`, `4.306e-06` on `1625`, `<= 1.431e-06` on NLBS).
3. **`batch_exams=8` vs `batch_exams=4` (`16` exams):** `max_exam_score_diff_vs_bs = 4.2e-06`.

### C. Throughput & Hardware Comparison Across Batch Sizes
| Config on `1x L40S` (`48 GB`) | Throughput (`exams/sec`) | `imgs/sec` | Peak VRAM (GiB) | Host RSS (GiB) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Original `batch_exams=1` *(NVMe cached, serial)* | `0.80` | `3.20` | — | — | `15.023s` for `12` exams |
| **Optimized `batch_exams=4` (`6 vCPU / 24 GiB`, Prefetch)** | **`1.84` E2E** (`2.40` GPU) | **`7.36` E2E** | **`9.309`** | `23.616` | **Fastest! (`nvme=1.903s`, `tta_stall=10.686s`, `gpu=41.639s`, `e2e=54.331s`)** |
| Mini-probe `batch_exams=8` (`16` exams) | `1.58` E2E | `6.32` E2E | `18.578` | `23.616` | Slower than `bs=4` (`2x` VRAM, bigger CPU TTA stalls) |

> **Critical Host RAM Note:** Each exam's pre-augmented `10`-TTA FP32 input tensor stack (`[10, 1, H, W]` × 4 views) uses `~0.78 GiB` host RAM, pushing `host_rss_gb` to **`23.616 GiB`** during the `24 GiB` container pilot. Full shards MUST use **`memory=32768` (`32 GiB`)** (and `cpu=8.0` to eliminate the `10.69s` CPU TTA stall) for wide OOM safety margin.

---

## 3. Actual Pilot Billing, Full Run Projection & `$1.50` Safety Buffer Analysis

- **Actual Pilot Spend (`ap-1WtcQA2ZWaNFBzKQy5CFAU` via `modal billing report`):**
  - `L40S`: `USD 0.050917` | `CPU`: `USD 0.013605` | `Memory`: `USD 0.005013` -> **`USD 0.0695` settled** (`credits` moved `-22.41 -> -22.48`, i.e., `USD 0.07`, well below `USD 0.20` limit; script E2E estimate `USD 0.03661`, total container `USD 0.07708`).
- **Remaining Starter Credits (`hasinishrak2015`):**
  - `USD 30.00 - USD 22.48 =` **`USD 7.52` remaining** until **`2026-12-01` `00:00 UTC`** (`~6.73` days left as of `2026-11-24T06:24Z`).
- **Projected Full Inference Cost (`8,566` Exams = `2,382` RSNA + `6,184` NLBS, `10` TTA):**
  - Measured `batch_exams=4` baseline (`1.84 exams/sec`, `8` shards w/ `22.376s` load overhead/shard): **`1.3425` L40S hours** = **`USD 3.2566`** (`~USD 0.93` RSNA `2,382` exams + `~USD 2.33` NLBS `6,184` exams).
  - Recommended `cpu=8.0, memory=32768` config (`USD 2.5842/hr`): conservative **`USD 3.35–3.47`** (less if `8 vCPU` trims the `10.7s` TTA wait).
- **Accounting for Daily Storage (`USD 0.485 / day` for `1,185.696 GiB`) & `$1.50` Safety Buffer:**
  1. **If keeping `cache/paper1_nyu_uint16_v1` (`157.506 GiB`) untouched through Nov 30 (`6.73 days * USD 0.485 = ~USD 3.26` storage):**
     - `USD 7.52 (credits) - USD 3.26 (storage) - USD 1.50 (buffer) =` **`USD 2.76` available compute in Nov** (`< USD 3.26–3.47` full 8,566-exam inference; running BOTH splits in Nov + keeping the cache would leave only `~USD 0.80–1.00` buffer!).
  2. **Recommended Safe Schedule (Preserves `>= USD 1.50` Buffer Cleanly):**
     - **Option 1 (Immediate Full Run + Reclaim NYU Cache Once Predictions Lock):** If approved to delete `cache/paper1_nyu_uint16_v1` (`157.506 GiB`) right after RSNA + NLBS Step-2 predictions are SHA-256 locked, workspace storage drops to `1,028.19 GiB` (`USD 0.013/day`), total remaining Nov storage drops to `~USD 0.56`, leaving **`~USD 3.50–3.70` credit buffer (`>> USD 1.50`)**!
     - **Option 2 (Zero Cache Deletion — Split Across Billing Reset):** Run **`rsna_val_calib` (`2,382` exams, `~USD 0.96`)** now (`USD 7.52 - USD 3.26 - USD 0.96 =` **`USD 3.30` buffer `>> USD 1.50`**), then run **`nlbs_4view` (`6,184` exams, `~USD 2.40`)** on/after **`2026-12-01` `00:00 UTC`** (`USD 30.00` fresh monthly credits)!

---

## 4. Recommended GPU Configuration for Step 2 Full Run

- **GPU:** `1x NVIDIA L40S` (`48 GB`) per shard (optimal FP32 throughput/USD; `L4`/`A10G` have ~3x lower FP32 TFLOPS and would cost more overall)
- **Host / Batching:** **`cpu=8.0`**, **`memory=32768` (`32 GiB RAM`)**, **`batch_exams=4`**, **`num_tta_epochs=10`**, **`seed=0`**, FP32 `eval()` mode, **`ephemeral_disk=524288`** (`512 GiB` NVMe), `max_workers=6` prefetch (`prefetch_limit=6`) + `gc.collect()`
- **Sharding:** `num_shards=4` per split (`~596` RSNA exams/shard = `~5.4 min/shard`; `~1,546` NLBS exams/shard = `~14.0 min/shard` under `timeout=7200s`), strictly resumable with `validate_nyu_shard_payload` + `finalize_and_lock_nyu_predictions`.
