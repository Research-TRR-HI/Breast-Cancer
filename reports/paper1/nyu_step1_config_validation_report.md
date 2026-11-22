# Paper 1 — NYU Step 1 Pre-Build Validation, Storage Audit & Guardrails Report

**Date:** 2026-11-22  
**Modal Profile:** `hasinishrak2015` (App `ap-p0FrhTarpdGuiJaUUYKJ6Q`, CPU-only `2 vCPU / 8 GiB RAM`, zero GPU allocated)  
**Verdict:** **SAFE TO PROCEED FOR FULL PREPROCESSING ONCE APPROVED** (`~$0.87–$1.13` CPU compute incl. overhead; `2.883 GiB` RAM headroom; SHA-256 resume verified; `--stage build` CLI guard verified).

---

## 1. Modal Free Storage Allowance & Exact Existing Volume Usage (No Deletions Performed)
* **Verified Modal Starter Tier Allowances (`modal.com/pricing` + `modal billing summary`):**
  * Compute credits: **`USD 30.00 / month`** (`-$19.99` consumed → **`~USD 10.01`** remaining; `Billed Cost = $0.00`).
  * Volume storage allowance: **`1 TiB (1,024 GiB)` per month free**, billed at **`USD 0.09 / GiB / month`** (`~$0.003 / GiB / day`) beyond `1,024 GiB`.
* **Measured Current Paper 1 Volume Footprint (Read-Only Walk — Zero Files Deleted):**
  * `rsna-mammography-vol`: **`559.204 GiB`** (`rsna`: `293.115 GiB`, `cache/paper1_gmic_uint16_v1`: `159.371 GiB`, `cache/stage1_rsna_uint16`: `65.950 GiB`, `cache/stage1_nlbs_uint16`: `35.939 GiB`, `models`: `1.798 GiB`, pilots/benches: `3.031 GiB`).
  * `nlbs-mammography-vol`: **`409.010 GiB`**.
  * **Combined Paper 1 Volumes Alone:** **`968.214 GiB`** (`55.79 GiB` below 1 TiB by themselves), **PLUS** `16` other pre-existing thesis/BanglaLLM Volumes on `hasinishrak2015` pushing total workspace storage past `1,024 GiB` (`Volumes = $26.36`, `Free Storage = -$25.34`, `$1.02` overage covered by compute credits).
* **Projected NYU Step 1 Cache Storage (`34,264` images, weighted per split):**
  * `rsna_val_calib` (`9,528` imgs × `2.094 MB/img` measured): **`19.48 GiB`**
  * `nlbs_4view` (`24,736` imgs × `4.865 MB/img` measured): **`117.52 GiB`**
  * **Total Projected NYU Cache:** **`137.00 GiB`**
  * **Actual Incremental Storage Cost Beyond 1 TiB (`USD 0.09/GiB/mo`):** Because `hasinishrak2015` is already at/above the 1 TiB free cap across its 18 volumes, storing `137.00 GiB` adds **`~USD 0.41 / day`** (**`~USD 3.29`** for the remaining `8` days of November, or **`USD 12.33 / month`** if kept for a full 30-day cycle — NOT an automatic fixed `$12` charge if Step 2 runs promptly).

---

## 2. Actual Full-Run CPU Config Validation (`2 vCPU / 8 GiB RAM`, `160` Images)
* **Container Spec Tested (`ap-p0FrhTarpdGuiJaUUYKJ6Q`):** `cpu=2.0`, `memory=8192 MiB`, `2` workers, `compression=1`.
* **Representative Sample (`160` images = `40` exams):** `20` RSNA exams (`patient_id=71` + `14` difficult GMIC-fallback cases; `26` fallbacks logged cleanly) + `20` NLBS exams (all `3` `SIEMENS` exams + `14` multi-candidate ties).
* **Measured Performance & Safety:**
  * Wall time: **`88.874s`** (`1.800 img/s` end-to-end)
  * Peak RSS (`host_rss_gb`): **`5.117 GB`** → **`2.883 GiB` safe margin below `8.0 GiB` RAM** (zero OOM risk).
  * Measured run cost: **`USD 0.00392`** (`$0.1586 / hr`).

---

## 3. SHA-256 Resume Verification & Guarded `--stage build` CLI Tests
* **SHA-256 Cache-Resume Upgrade (`verify_and_load_nyu_resumed_records`):**
  * `_process_one_nyu_view` now stores both **`png_sha256`** (compressed PNG file hash) and **`u16_sha256`** (raw uint16 pixel array hash).
  * Resume checks require `file_bytes > 0`, `st_size == file_bytes`, AND exact **`compute_sha256_file(png_path) == png_sha256`**.
  * **Live Cloud Tamper Test (`PASSED`):** Corrupted 1 byte inside `198449006.png` preserving exact byte count + unlinked `2004100590.png` → resumed `158` valid PNGs, rejected `2` tampered/missing files, recomputed those `2`, and verified `160/160` valid (`rejected = 0`).
  * **Local Unit Tests (`tests/test_paper1_nyu_step1_cpu.py`, `5/5 PASS`):** Added `test_4_sha256_resume_rejects_same_size_corruption_and_missing_hash` and `test_5_guarded_build_cli_blocks_without_approval`.
* **Live `--stage build` CLI Guard Test (`PASSED`):**
  * Running `modal run scripts/preprocess_paper1_nyu_cpu.py --stage build` WITHOUT `--approve-full-build` aborted locally (`exit=1`, `RuntimeError: BLOCKED: --stage build requires explicit --approve-full-build confirmation...`) before launching any worker container.

---

## 4. Recalculated Full Build Compute + Overhead Projections (`34,264` Images)
Assuming `num_shards=12` per split (`24` total build shards) + `25s` container startup/commit overhead per shard + `2` `verify_nyu_cpu_cache` runs (`cpu=2.0, memory=4096`):
* **At measured `2vCPU/8GiB` validation rate (`1.800 img/s`):**
  * `RSNA (9,528 imgs)`: **`USD 0.2464`** (`~15 min` wall across `12` shards)
  * `NLBS (24,736 imgs)`: **`USD 0.6186`** (`~38 min` wall across `12` shards)
  * `Verify` (`2` splits): `~USD 0.0042`
  * **Total Compute (incl. shard overhead & verification):** **`~USD 0.869 USD`** *(conservative slower-throughput bound using `1.377 img/s`: `~USD 1.127`)*.
* **Credit Headroom Check:** Easily fits within `hasinishrak2015`'s **`~USD 10.01`** remaining Starter compute credits (`<12%` of remaining balance).

**Stopped cleanly after validation (`--stage build` not executed).**
