# Research Log — 2026-11-24: [Paper 1] NYU Four-View Step 1 COMPLETE (`24,736` NLBS Verified; `34,264 / 34,264` Total Images)

- **Date:** 2026-11-24 (`06:57–08:00 +06:00` / `00:57–02:00 UTC`)
- **Author / Track:** Paper 1 — Multi-Model External Validation Benchmark (`nyu_fourview_pretrained`)
- **Modal Workspace:** `hasinishrak2015` (CPU ONLY; `0` GPUs allocated)
- **Scope Completed:** Remaining **`24,736`** NLBS four-view images (`6,184` exams) built, verified, and consolidated alongside **`9,528`** RSNA `val_calib` (`2,382` exams) -> **`34,264 / 34,264` images / `8,566` exams 100% COMPLETE**.
- **Label Quarantine:** Strictly label-free (`datasets/paper1/nlbs_4view_common_cohort_label_free.csv`); zero Canadian cancer labels accessed.

---

## 1. Pre-Run Gate Verification
- Synced `origin/main` (`87d4fd8`), confirmed clean `9`-volume workspace (`173.363 GiB` cleanup verified), verified RSNA `verify_summary.json` intact (`9,528` imgs, `2,382` exams, `69.779 GiB`, SHA-256 `bde6ae3f...`), and verified `USD 8.66` Starter credits remaining (`~USD 1.87–3.27 > USD 1.50` buffer).

## 2. NLBS Full 12-Shard Build & Verification (`ap-Fik9WBtJXeOQ0CI2hHHrxD`)
- Executed `modal run scripts/preprocess_paper1_nyu_cpu.py --stage build --split nlbs_4view --num-shards 12 --compression 1 --approve-full-build` (`2 vCPU / 8 GiB` build shards, `1 vCPU / 2 GiB` verify shards, `timeout=3600`).
- **Compute Stability & Resume Verification:** Shards finished in `969.56s–1,949.45s` (`peak RSS 4.320 GiB <= 8 GiB`); shards `0, 3, 5` verified seamless live SHA-256 resume (`skipped_resumed: 1200, 1100, 1200`, `rejected: 0`). Parallel verification completed in `443.36s–1,473.98s` (`0` rejected across all `24,736` PNGs).
- **Verified Counts & Integrity:**
  - `24,736` unique images, `0` duplicates, `6,184` complete 4-view exams (`L-CC / R-CC / L-MLO / R-MLO = 6184` each), `MONOCHROME2: 24736`, **`0` fallbacks**, `101 / 101` decoded `u16_sha256` pixel checks PASS.
  - Downloaded & verified `reports/paper1/nyu_step1_nlbs_4view_manifest.csv` (`7,681,660` bytes, SHA-256 exact match `e941df848f2baeac58bbe35b11e4b5ec4e30074c28b893b59f18258008beec65`).

## 3. Settled Billing & Storage Totals
- **NLBS Compute (`ap-Fik9WBt…`):** `USD 0.66860` CPU + `USD 0.40288` Mem = **`USD 1.0715`** (`credits: -22.43` -> **`USD 7.57` remaining** until `2026-12-01` `00:00 UTC`).
- **Combined Step 1 Production Compute (`RSNA + NLBS`):** `USD 1.6203` net (`USD 1.6834` gross).
- **NYU Cache Size (`rsna-mammography-vol:/cache/paper1_nyu_uint16_v1/`):** `157.506 GiB` (`69.786 GiB` RSNA + `87.720 GiB` NLBS; `34,294` files). Total workspace across `9` volumes = **`1,185.696 GiB`** (`USD 0.485/day` excess above `1,024 GiB`, leaving `~USD 4.19` month-end credit buffer).

## 4. Deliverables & Step 2 Readiness
- [`reports/paper1/nyu_step1_preprocessing_report.md`](../reports/paper1/nyu_step1_preprocessing_report.md)
- [`reports/paper1/nyu_step1_nlbs_4view_manifest.csv`](../reports/paper1/nyu_step1_nlbs_4view_manifest.csv)
- [`reports/paper1/nyu_step1_nlbs_4view_build_summary.json`](../reports/paper1/nyu_step1_nlbs_4view_build_summary.json)
- [`reports/paper1/nyu_step1_nlbs_4view_fallback_log.json`](../reports/paper1/nyu_step1_nlbs_4view_fallback_log.json)
- **Status:** NYU Four-View Step 1 is **100% COMPLETE** and ready for Step 2 GPU inference (not started; stopped cleanly).
