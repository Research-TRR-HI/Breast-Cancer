# Research Log — 2026-11-23: [Paper 1] Complete `hasinishrak2015` 18-Volume Storage Audit, Revised Modal Rate & NLBS Low-Cost Plan

- **Date:** 2026-11-23 (`20:29 +06:00`)
- **Author / Track:** Paper 1 — Multi-Model External Validation Benchmark (`nyu_fourview_pretrained`)
- **Modal Workspace:** `hasinishrak2015` (pure read-only SDK + billing inspection; `$0.00` compute, `0` deletions, `0` GPUs)

---

## 1. Purpose & Constraints
- Audit all `18` Modal Volumes in `hasinishrak2015`, isolate ONLY obsolete/reproducible pilot caches & unrelated temporary/re-downloadable files without touching raw datasets, verified RSNA/GMIC/Stage-1 caches, checkpoints, or locked predictions.
- Determine actual Modal Volume storage rate, remaining credit runway, and exact billing-cycle reset date; update `verify_nyu_cpu_shard` timeout (`1800 -> 3600s`) and verify local tests (`5/5`).

## 2. Empirical Modal Billing & Reset Date Verification (`$0.00` RPCs)
- `modal billing summary --help` confirms calendar-month cycles (`2026-11-01` -> `2026-11-30`), resetting **`2026-12-01` (`00:00 UTC`)**.
- Current Nov 1–23 summary (`hasinishrak2015`): `metered_cost: 49.84` (`volumes: 30.1734`, `ephemeral_apps: 13.2135`, `deployed_apps: 6.3687`, `network_egress: 0.0803`), `free_storage: -28.4169`, `credits: -21.3400` -> **`USD 8.66` remaining** (`billed_cost: $0.00`).
- Comparing 24h billing delta (`Nov 22 -> Nov 23`) against actual volume total (`1,271.340 GiB` across `18` volumes):
  - `Free Storage` (`1,024 GiB` continuous daily allowance) increased by `+USD 3.0769/day` = **`USD 0.003005 / GiB-day` (`~USD 0.090 / GiB-month`)**.
  - Excess `247.34 GiB` billed `+USD 0.7365/day` = **`USD 0.002978 / GiB-day` (`~USD 0.089 / GiB-month`)**.
  - Corrects earlier conservative assumptions (`$25.34` cap / `$0.15/GiB-mo`): only storage **above `1,024 GiB`** consumes credits at **`~USD 0.0030 / GiB-day`**.

## 3. Full 18-Volume Breakdown (`1,271.340 GiB`)
- **Breast Cancer Core (KEEP — `1,034.977 GiB`):** `nlbs-mammography-vol` (`409.010 GiB`: `nlbs/extracted` `324.448`, `nlbs/raw` `84.562`); `rsna-mammography-vol` (`625.967 GiB` core: `rsna/raw` `293.110`, `paper1_gmic_uint16_v1` `159.371`, `paper1_nyu_uint16_v1` `69.786`, `stage1_rsna_uint16` `65.950`, `stage1_nlbs_uint16` `35.939`, checkpoints/receipts `1.811`).
- **Tier 1 Safe Breast-Cancer Pilot/Temp Cleanup (`3.022 GiB` on `rsna-mammography-vol`):** `paper1_nyu_uint16_v1_bench_40ex_c4_z1` (`1.258`), `paper1_gmic_uint16_v1_bench200` (`0.897`), `paper1_nyu_uint16_v1_validate_2c8g` (`0.544`), `paper1_nyu_uint16_v1_bench_5ex_c2_z1` (`0.156`), `parity_pilot_1200` (`0.154`), `bench_1000` (`0.012`), `paper1_dinov2_vits14_bench` (`0.001`).
- **Tier 2A Re-downloadable HF LLM Weight Volumes (`170.341 GiB` across `9` BanglaLLM volumes):** `mistral-nemo-12b-volume` (`45.654`), `llama-3.1-8b-volume` (`29.926`), `ministral-8b-volume` (`29.906`), `gemma-2-9b-volume` (`17.243`), `qwen-7b-volume` (`14.196`), `banglallm-qwen-3b-volume` (`12.670`), `ministral-3b-volume` (`8.734`), `titulm-3b-volume` (`6.253`), `qwen-3b-volume` (`5.759`). Keeps `banglabbq-eval-volume` (`1.266 GiB`) eval outputs intact.
- **Tier 2B Completed Cattle Thesis Volumes (`61.733 GiB` across `6` volumes):** `sideview-data` (`23.463`), `cvb-data` (`14.565`), `mtl-data` (`11.472`), `beef-behavior-data` (`6.373`), `sciencedb-data` (`4.026`), `mtl-checkpoints` (`1.834`).

## 4. Timeout Fix, Unit Tests & Recommended Plan
- Updated `verify_nyu_cpu_shard` timeout (`1800 -> 3600s`) in [`scripts/preprocess_paper1_nyu_cpu.py`](../scripts/preprocess_paper1_nyu_cpu.py) and passed `py_compile` + **`tests/test_paper1_nyu_step1_cpu.py` (`5/5` PASS, `54.211s`)**.
- **Safest Low-Cost Plan:** Reclaim **Tier 1 + Tier 2A (`173.363 GiB`)** first (drops usage to `1,097.98 GiB`, saves `~$0.52/day` without touching any Breast Cancer raw/cache/checkpoint data) -> run `nlbs_4view` (`~USD 1.42` compute; post-NLBS excess `USD 0.57–0.77/day`; total remaining Nov cost `USD 5.56–6.94` fits inside **`USD 8.66`** credits before `2026-12-01` reset) -> delete `cache/paper1_nyu_uint16_v1` once Step 2 GPU inference locks predictions.

## 5. Artifacts Generated
- [`reports/paper1/nyu_step1_storage_and_budget_audit.md`](../reports/paper1/nyu_step1_storage_and_budget_audit.md)
