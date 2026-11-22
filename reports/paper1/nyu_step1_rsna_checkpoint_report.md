# Paper 1 — NYU Four-View Step 1 CPU Preprocessing Report (RSNA Complete; Paused Before NLBS at Cost/Storage Stop Gate)

- **Date:** 2026-11-23 (UTC runs `2026-11-22T20:26Z–21:24Z`)
- **Modal Workspace:** `hasinishrak2015` (Starter tier; `0` GPUs allocated)
- **Status:** `rsna_val_calib` **100% COMPLETE & VERIFIED** (`9,528` images / `2,382` exams); `nlbs_4view` (`24,736` images) **PAUSED BEFORE LAUNCH** per monitoring stop-gate rule (`RSNA storage 3.58x estimate`, `RSNA compute 2.2x–2.5x estimate`, `USD 9.34` Starter credits remaining until Dec 1)

---

## 1. Executive Summary & Stop-Gate Trigger

1. **RSNA `val_calib` (`9,528` images / `2,382` four-view exams):** Processed across `12` resumable CPU shards (`2 vCPU / 8 GiB`, `compression=1`, `0` GPUs) and verified 100% end-to-end (`0` rejected PNGs, `101` decoded pixel `u16_sha256` spot checks passed, complete `L-CC` / `L-MLO` / `R-CC` / `R-MLO` coverage).
2. **Parallel Verification Bottleneck Caught & Fixed:** In the initial build run (`ap-1dMjmBD9jw2X6b150L9X4u`), all `12` RSNA shards finished converting and committing all `9,528` PNGs in ~16 min, then the single-container `verify_nyu_cpu_cache` timed out at `1800s` reading `9,528` PNGs serially over Modal Volume FUSE. Added `verify_nyu_cpu_shard` (`12` parallel workers checking 100% PNG SHA-256s concurrently) + fast manifest consolidation (`verify_nyu_cpu_cache` with `preverified_shards`), verified with `5/5` passing unit tests (`tests/test_paper1_nyu_step1_cpu.py`) and completed cleanly in `ap-yaZkL5TEdpCYQTcEqIWQfl`.
3. **Why Execution Paused Before `nlbs_4view` (`24,736` images):**
   - **Storage Growth (`3.58x` higher on RSNA):** RSNA consumed **`69.779 GiB`** (`mean 7.499 MiB/img`, `median 6.779 MiB/img`) vs the **`19.48 GiB`** pre-build estimate (`+50.30 GiB` overrun on RSNA alone).
   - **Compute Cost (`~2.2x–2.5x` estimate):** RSNA spent **`USD 0.6086` gross** (`USD 0.5359` build excl./incl. `~$0.063` aborted `1800s` verify hang -> `~USD 0.473` net shard build + `USD 0.0727` parallel verify = `~USD 0.546` net) vs `USD 0.2464` estimated. Scaling `2.596x` to NLBS (`24,736` imgs) projects **`~USD 1.42–1.55`** NLBS compute -> **`~USD 2.03–2.16`** total compute vs `USD 0.87` (`<= USD 1.13`) projected.
   - **Remaining Credit Runway (`USD 9.34` until Dec 1):** `modal billing summary` shows `Free Storage: -25.34` already 100% exhausted by pre-existing workspace volumes (`968+ GiB` Paper 1 + 16 legacy volumes) and `Credits: -20.66` (`USD 9.34` remaining). Running `nlbs_4view` pushes new cache storage to **`~187.30 GiB`** (at `117.52 GiB` NLBS pilot rate) or up to **`~250 GiB`** (if NLBS tracks closer to RSNA), costing **`~USD 0.94–1.25/day`** (`~USD 7.50–10.00` over 8 days of November) + `~USD 1.50` compute -> right at or exceeding the remaining `USD 9.34` balance before Dec 1 reset.

---

## 2. RSNA `val_calib` Cache Integrity & Verification Receipts

- **Cache Volume Path:** `rsna-mammography-vol:/cache/paper1_nyu_uint16_v1/rsna_val_calib/`
- **Modal Runs:**
  - Build (`12` shards, `2 vCPU / 8 GiB`, `compression=1`): [`ap-1dMjmBD9jw2X6b150L9X4u`](https://modal.com/apps/hasinishrak2015/main/ap-1dMjmBD9jw2X6b150L9X4u) (`Memory: $0.1984`, `CPU: $0.3375`, total **`$0.5359`**; all 12 shards `792/792` & `796/796` succeeded cleanly before single-thread `verify` timed out at `1800s`).
  - Verify (`12` parallel verify shards `1 vCPU / 2 GiB` + `verify_nyu_cpu_cache` `2 vCPU / 4 GiB`): [`ap-yaZkL5TEdpCYQTcEqIWQfl`](https://modal.com/apps/hasinishrak2015/main/ap-yaZkL5TEdpCYQTcEqIWQfl) (`Memory: $0.0184`, `CPU: $0.0543`, total **`$0.0727`**; shard wall times `79.11s–437.56s`, `0` rejected files).
- **Verified Counts & Metadata (`nyu_step1_rsna_val_calib_verify_summary.json`):**
  - Total images: **`9,528`** across **`2,382`** complete 4-view exams (`L-CC: 2382`, `L-MLO: 2382`, `R-CC: 2382`, `R-MLO: 2382`).
  - Photometric interpretation: `MONOCHROME1: 1808`, `MONOCHROME2: 7720`.
  - Checksums: `100%` PNG SHA-256 verified across all `12` shards (`rejected: 0`) + **`101`** decoded `uint16` `u16_sha256` pixel checks passed.
  - Cropping fallbacks: **`66`** views across **`34`** exams, all handled via `native_scale_optimal_window_no_resize` (zero `cv2.resize` distortion, native dimensions preserved).
  - Manifest SHA-256: `bde6ae3fa12cff0c0d121de2b1344ddc775f1020b3645670b4d62ca00db8f866`.
  - Local files saved:
    - [`reports/paper1/nyu_step1_rsna_val_calib_verify_summary.json`](nyu_step1_rsna_val_calib_verify_summary.json)
    - [`reports/paper1/nyu_step1_rsna_val_calib_fallback_log.json`](nyu_step1_rsna_val_calib_fallback_log.json)
    - [`reports/paper1/nyu_step1_rsna_val_calib_manifest.csv`](nyu_step1_rsna_val_calib_manifest.csv) (`2,532,220` bytes; `69.779 GiB` total cached PNG bytes)

---

## 3. Billing Comparison: Verified Estimate vs Actual Trajectory

| Metric | Pre-Build Estimate | Actual / Updated Projection | Delta / Ratio |
| :--- | :---: | :---: | :---: |
| RSNA Compute (`9,528` imgs) | `USD 0.2464` | **`USD 0.546` net** (`USD 0.6086` gross incl. `1800s` verify timeout) | `2.2x–2.5x` |
| NLBS Compute (`24,736` imgs) | `USD 0.6186` | `~USD 1.42–1.55` projected (`not run`) | `~2.3x–2.5x` |
| Total Step 1 Compute (`34,264` imgs) | `USD 0.87` (`<= 1.13`) | `~USD 2.03–2.16` projected | `~2.3x` |
| RSNA Storage (`9,528` imgs) | `19.48 GiB` | **`69.779 GiB`** (`7.499 MiB/img` mean) | **`3.58x`** (`+50.30 GiB`) |
| Combined NYU Cache Storage | `137.00 GiB` | `~187.30–250 GiB` projected | `+37%` to `+82%` |
| Modal Starter Credits Used (Total) | `USD 19.99` before build | `USD 20.66` after RSNA (**`USD 9.34` left**) | `USD 0.67` delta (`incl. billing drift`) |

> **Modal Billing Snapshot (`hasinishrak2015`, after RSNA verify):** `Metered Cost: 46.09` (`Volumes: 26.36`, `Ephemeral Apps: 13.27`, `Deployed Apps: 6.37`, `Network Egress: 0.08`), `Credits: -20.66`, `Network Egress Allowance: -0.08`, `Free Storage: -25.34`, `Billed Cost: $0.00`.

---

## 4. Technical Readiness & Options Before Launching `nlbs_4view`

1. **Before running 12-shard `nlbs_4view` (`2,061` images/shard):**
   - `rsna_val_calib` verification (`792–796` images/shard) took up to `437.56s` per `verify_nyu_cpu_shard` worker (`~0.55s/file`). Scaling `2.60x` to `2,061` images/shard on NLBS puts `verify_nyu_cpu_shard` around `~1,135s` (risking `1800s` FUSE jitter timeout unless bumped to `timeout=3600` or threaded inside `verify_nyu_cpu_shard`).
2. **Decision Options for User:**
   - **Option A (Proceed with `nlbs_4view` `compression=1`, `12` shards):** Bump `verify_nyu_cpu_shard` `timeout=3600` first, then run NLBS (`~USD 1.42–1.55` compute, `~187–250 GiB` total NYU storage -> monitor remaining `$9.34` Nov credits or run after Dec 1 reset / accept small overage if card attached).
   - **Option B (`compression=3` on `nlbs_4view`):** Still 100% lossless `uint16` PNG; trades ~`+$0.50` CPU time for ~15–25% smaller PNG storage (`~20–45 GiB` saved).
   - **Option C (Free non-GMIC temporary pilot storage first, with explicit permission):** Keep GMIC/Stage-1/raw caches untouched; removing legacy `paper1_nyu_uint16_v1_bench_*` / `validate_2c8g` dirs or unused legacy workspace volumes recovers storage headroom cleanly if approved.
