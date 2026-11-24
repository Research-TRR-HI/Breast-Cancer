# Paper 1 — Complete NYU Four-View Step 1 CPU Preprocessing Report

- **Completion Date:** 2026-11-24 (`07:58 +06:00` / `01:58 UTC`)
- **Modal Profile:** `hasinishrak2015` (CPU ONLY; **`0` GPUs allocated** across entire Step 1)
- **Overall Status:** **100% COMPLETE & VERIFIED** — **`34,264 / 34,264` images** across **`8,566 / 8,566` complete four-view examinations** (`9,528` RSNA `val_calib` + `24,736` NLBS `4-view`). **READY FOR NYU STEP 2 GPU INFERENCE.**

---

## 1. Complete Step 1 Dataset & Integrity Summary

| Split | Images | 4-View Exams | L-CC / R-CC / L-MLO / R-MLO | Corrupt / Rejected | Pixel Spot Checks (`u16_sha256`) | Cropping Fallbacks (`1:1` No Resize) | Photometric Breakdown | PNG Storage (GiB) | Total Dir (GiB, incl. JSONLs) | Manifest SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| `rsna_val_calib` | **`9,528`** | **`2,382`** | `2382 / 2382 / 2382 / 2382` | **`0`** | `101 / 101` | `66` (`34` exams) | `M1: 1808`, `M2: 7720` | `69.779` (`7.499 MiB` avg) | `69.786` (`9,543` files) | `bde6ae3fa12cff0c0d121de2b1344ddc775f1020b3645670b4d62ca00db8f866` |
| `nlbs_4view` | **`24,736`** | **`6,184`** | `6184 / 6184 / 6184 / 6184` | **`0`** | `101 / 101` | **`0`** (`0` exams) | `M2: 24736` | `87.698` (`3.630 MiB` avg) | `87.720` (`24,751` files) | `e941df848f2baeac58bbe35b11e4b5ec4e30074c28b893b59f18258008beec65` |
| **Combined Total** | **`34,264`** | **`8,566`** | `8566` per view | **`0`** | **`202 / 202`** | `66` (`34` exams) | `M1: 1808`, `M2: 32456` | **`157.477`** | **`157.506`** (`34,294` files) | — |

- **Strict Rules Enforced:**
  - Zero Canadian (`cancer`) outcome labels accessed (`nlbs_4view_common_cohort_label_free.csv` strictly label-free).
  - Zero GPU allocations (`cpu=2.0, memory=8192` build shards; `cpu=1.0, memory=2048` verify shards; `cpu=2.0, memory=4096` manifest consolidator).
  - Preserved all completed CE, TRUST, GMIC (`paper1_gmic_uint16_v1` `159.371 GiB`), Stage-1 (`65.950 + 35.939 GiB`), checkpoints, and raw DICOM datasets untouched.

---

## 2. NLBS `24,736`-Image Execution Details (`ap-Fik9WBtJXeOQ0CI2hHHrxD`)

- **Run:** [`ap-Fik9WBtJXeOQ0CI2hHHrxD`](https://modal.com/apps/hasinishrak2015/main/ap-Fik9WBtJXeOQ0CI2hHHrxD) (`--stage build --split nlbs_4view --num-shards 12 --compression 1 --approve-full-build`, `timeout=3600` on both compute & verify shards).
- **Build Shard Performance:**
  - Wall time across `12` shards (`2,060`–`2,064` imgs/shard): `969.56s` (`16.2 min`) to `1,949.45s` (`32.5 min`).
  - Peak host RSS across all `12` shards: **`4.320 GiB`** (`3.680 GiB` headroom below `8 GiB` RAM limit).
  - Live SHA-256 resume exercised seamlessly on shards `0`, `3`, and `5` (`skipped_resumed: 1200, 1100, 1200`, `rejected_corrupt_resumed: 0`, zero duplicate or lost images).
- **Parallel SHA-256 Verify Shards (`12` workers, `timeout=3600`):**
  - Wall time: `443.36s` (`7.4 min`) to **`1,473.98s` (`24.6 min`)** (`s02`, `s07` benefitted directly from the `3600s` timeout upgrade).
  - Rejected files: **`0 / 24,736`** (`100%` PNG SHA-256 integrity verified + `101` decoded `uint16` `u16_sha256` checks passed).
- **Local Artifacts Saved:**
  - [`reports/paper1/nyu_step1_nlbs_4view_manifest.csv`](nyu_step1_nlbs_4view_manifest.csv) (`7,681,660` bytes; SHA-256 verified identical to Volume manifest)
  - [`reports/paper1/nyu_step1_nlbs_4view_build_summary.json`](nyu_step1_nlbs_4view_build_summary.json)
  - [`reports/paper1/nyu_step1_nlbs_4view_fallback_log.json`](nyu_step1_nlbs_4view_fallback_log.json) (`[]` — `0` NLBS fallbacks)
  - Paired RSNA deliverables: `nyu_step1_rsna_val_calib_manifest.csv`, `nyu_step1_rsna_val_calib_verify_summary.json`, `nyu_step1_rsna_val_calib_fallback_log.json`, `nyu_step1_rsna_checkpoint_report.md`.

---

## 3. Actual Costs, Storage Usage & Remaining Credits (`hasinishrak2015`)

- **NLBS Settled Modal App Spend (`ap-Fik9WBtJXeOQ0CI2hHHrxD`):**
  - `CPU`: `USD 0.66860` | `Memory`: `USD 0.40288` -> **`USD 1.0715` settled compute** (`25%` cheaper than the `USD 1.42` estimate; `USD 0.9066` shard-loop estimate vs `USD 1.0715` incl. container & consolidator overhead).
- **Combined Step 1 Production Compute (`RSNA + NLBS`):**
  - `RSNA`: `USD 0.5488` net (`USD 0.6119` gross incl. aborted `1800s` verify in `ap-1dMjmBD…` + `ap-yaZkL5T…`)
  - `NLBS`: **`USD 1.0715`** (`ap-Fik9WBt…`)
  - **Combined RSNA + NLBS Compute:** **`USD 1.6203` net (`USD 1.6834` gross)**.
- **Modal Credit Balance (`modal billing summary --for "this month"`):**
  - `metered_cost: 50.93` (`volumes: 30.1734`, `ephemeral_apps: 14.3082`, `deployed_apps: 6.3687`, `network_egress: 0.0803`)
  - `free_storage: -28.4169` | `credits: -22.4300` -> **`USD 7.57` Starter credits remaining** (`billed_cost: USD 0.00`), resetting **`2026-12-01` `00:00 UTC`**.
- **Workspace Storage Usage (`9` Volumes on `hasinishrak2015`):**
  - Post-cleanup baseline (`1,097.976 GiB`) + verified `nlbs_4view` (`87.720 GiB`) = **`1,185.696 GiB`** total (`161.696 GiB` over the `1,024 GiB` continuous free tier).
  - Daily overage cost (`@USD 0.0030 / GiB-day`): **`USD 0.485 / day`** (`~USD 14.55 / month`). Remaining `~6.96` November days ≈ `~USD 3.38`, leaving **`~USD 4.19`** headroom at month end (`2.8x` above `$1.50` safety buffer).

---

## 4. Readiness for NYU Step 2 GPU Inference

- **Ready:** `rsna-mammography-vol:/cache/paper1_nyu_uint16_v1/` holds complete lossless `uint16` `cropped_flip` PNGs + exact `(best_center_y, best_center_x)` coordinates for all **`2,382`** RSNA and **`6,184`** NLBS 4-view examinations (`34,264` images total) with `5/5` parity/resume/CLI unit tests passing (`tests/test_paper1_nyu_step1_cpu.py`).
- **Next Step (Awaiting Separate Approval):** Configure & pilot NYU Step 2 L40S inference (`10`-epoch random-crop test-time augmentation matching `nyukat/breast_cancer_classifier` `sample_image_model.p`), then reclaim `cache/paper1_nyu_uint16_v1` (`157.506 GiB`) once predictions are SHA-256 locked cleanly.
