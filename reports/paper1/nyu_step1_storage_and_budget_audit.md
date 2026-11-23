# Modal Volume Audit, Cleanup Candidates & Revised NLBS Budget (`hasinishrak2015`)

- **Audit Date:** 2026-11-23 (`20:29 +06:00`)
- **Modal Profile:** `hasinishrak2015` (Starter tier; **$0.00 compute**, pure read-only SDK metadata scan + CLI billing queries)
- **Actions Enforced:** Zero deletions executed, zero NLBS preprocessing launched, zero GPUs allocated; `verify_nyu_cpu_shard` timeout updated to `3600s` (`5/5` unit tests PASS).

---

## 1. Billing Status, Real Storage Rate & Reset Date

- **Billing Cycle & Reset Date:** Modal's billing summary aligns strictly to calendar month boundaries (`--for "last month"` = Oct 2026; `--for "this month"` = Nov 2026). Next Starter credit reset (`$30.00/month`) occurs on **`2026-12-01` (`00:00 UTC`)**.
- **Latest `modal billing summary --for "this month"` (`hasinishrak2015`, Nov 1–23):**
  - `metered_cost`: `USD 49.84` (`volumes: 30.1734`, `ephemeral_apps: 13.2135`, `deployed_apps: 6.3687`, `network_egress: 0.0803`)
  - `free_storage`: `-USD 28.4169`
  - `network_egress_allowance`: `-USD 0.0803`
  - `credits`: `-USD 21.3400` -> **`USD 8.66` Starter credits remaining** (`USD 30.00 - USD 21.34`)
  - `billed_cost`: `USD 0.00`
- **Actual Observed Storage Billing Rate (Corrects Earlier `$0.15/GiB-mo` Estimate):**
  - Comparing Nov 22 (`Volumes: 26.36`, `Free Storage: -25.34`) -> Nov 23 (`Volumes: 30.1734`, `Free Storage: -28.4169`, `1,271.340 GiB` total across 18 volumes):
    - `24h` `Free Storage` delta = `+USD 3.0769/day` covering `1,024 GiB` -> **`USD 0.003005 / GiB-day`** (`~USD 0.090 / GiB-month`).
    - `24h` net billed excess (`247.34 GiB` over `1,024 GiB`) = `+USD 0.7365/day` -> **`USD 0.002978 / GiB-day`** (`~USD 0.089 / GiB-month`).
  - **Key Correction:** Modal Starter's `1 TiB` (`1,024 GiB`) free storage quota is **NOT exhausted at `$25.34`** — it continuously offsets up to **`1,024 GiB/day`** (`~USD 3.08/day` in a 30-day month), and ONLY storage **above `1,024 GiB`** consumes `$30` Starter credits at **`USD 0.0030 / GiB-day` (`USD 0.090 / GiB-month`)**!

---

## 2. Complete Inventory of All 18 Modal Volumes (`1,271.340 GiB` Total)

### A. Active Breast Cancer Project Volumes (`1,037.999 GiB`, `81.65%`)

| Volume | Path | Size (GiB) | Files | Purpose & Keep / Cleanup Classification |
| :--- | :--- | :---: | :---: | :--- |
| `nlbs-mammography-vol` | `/nlbs/extracted` | `324.448` | `26,988` | Raw extracted NLBS DICOMs -> **KEEP (required for NLBS NYU Step 1)** |
| `nlbs-mammography-vol` | `/nlbs/raw` | `84.562` | `9` | Original FRDR `.7z` raw archives -> **KEEP (raw dataset untouched)** |
| `rsna-mammography-vol` | `/rsna/raw` | `293.110` | `54,713` | Raw RSNA DICOMs -> **KEEP (raw dataset untouched)** |
| `rsna-mammography-vol` | `/rsna/*` | `0.005` | `2` | `splits_seed42.csv`, `train.csv.zip` -> **KEEP** |
| `rsna-mammography-vol` | `/cache/paper1_gmic_uint16_v1` | `159.371` | `38,008` | Verified GMIC `2944x1920` uint16 PNG cache -> **KEEP (locked GMIC)** |
| `rsna-mammography-vol` | `/cache/paper1_nyu_uint16_v1` | `69.786` | `9,543` | Verified NYU RSNA `val_calib` PNG cache (`9,528` PNGs = `69.779 GiB` + 12 JSONL/manifest files) -> **KEEP** |
| `rsna-mammography-vol` | `/cache/stage1_rsna_uint16` | `65.950` | `54,710` | Stage-1 RSNA `1520x912` PNG cache -> **KEEP (needed for DINOv2 & ResNet-50)** |
| `rsna-mammography-vol` | `/cache/stage1_nlbs_uint16` | `35.939` | `27,020` | Stage-1 NLBS `1520x912` PNG cache -> **KEEP (needed for DINOv2, ResNet-50, Paper 2)** |
| `rsna-mammography-vol` | `/models/mammo_clip` | `1.541` | `1` | Pinned `b5-model-best-epoch-7.tar` -> **KEEP** |
| `rsna-mammography-vol` | `/models/paper1` | `0.257` | `4` | Pinned Paper 1 pretrained weights (GMIC, NYU, DINOv2, ResNet-50) -> **KEEP** |
| `rsna-mammography-vol` | `/cache/paper1_gmic_shards` | `0.007` | `8` | Step-2 GMIC prediction shard JSONL receipts -> **KEEP** |
| `rsna-mammography-vol` | `/reports`, `/models/*` | `0.002` | `24` | Checkpoints & Modal bench JSON receipts -> **KEEP** |
| `rsna-mammography-vol` | **`/cache/paper1_nyu_uint16_v1_bench_40ex_c4_z1`** | **`1.258`** | `321` | **Tier 1 Cleanup:** Obsolete NYU `4vCPU/4GiB` CPU pilot cache -> **NOT NEEDED (reproducible / superseded)** |
| `rsna-mammography-vol` | **`/cache/paper1_gmic_uint16_v1_bench200`** | **`0.897`** | `200` | **Tier 1 Cleanup:** Obsolete GMIC 200-img benchmark cache -> **NOT NEEDED (superseded)** |
| `rsna-mammography-vol` | **`/cache/paper1_nyu_uint16_v1_validate_2c8g`** | **`0.544`** | `161` | **Tier 1 Cleanup:** Obsolete NYU 160-img config/tamper validation cache -> **NOT NEEDED (superseded)** |
| `rsna-mammography-vol` | **`/cache/paper1_nyu_uint16_v1_bench_5ex_c2_z1`** | **`0.156`** | `81` | **Tier 1 Cleanup:** Obsolete NYU `2vCPU/3GiB` 5-exam probe cache -> **NOT NEEDED (superseded)** |
| `rsna-mammography-vol` | **`/cache/parity_pilot_1200`** | **`0.154`** | `43` | **Tier 1 Cleanup:** Obsolete Stage-B 1,200-img CE/TRUST parity pilot shards -> **NOT NEEDED (Stage B done)** |
| `rsna-mammography-vol` | **`/cache/bench_1000`** | **`0.012`** | `2` | **Tier 1 Cleanup:** Obsolete Stage-B 1,000-img Approach A/B bench file -> **NOT NEEDED (Stage B done)** |
| `rsna-mammography-vol` | **`/cache/paper1_dinov2_vits14_bench`** | **`0.001`** | `2` | **Tier 1 Cleanup:** Obsolete DINOv2 512-img NVMe benchmark output -> **NOT NEEDED (reproducible)** |

- **Breast Cancer Core Total (Excluding Tier 1):** `1,034.977 GiB` (`rsna`: `625.967 GiB` + `nlbs`: `409.010 GiB`).
- **Tier 1 Safe Breast-Cancer Pilot/Temp Cleanup Total:** **`3.022 GiB`** (`$0.009/day`).

---

### B. Unrelated Older Project Volumes on `hasinishrak2015` (`233.341 GiB`, `18.35%`)

> These 16 volumes account for **94.3% of the current `247.34 GiB` overage above Modal's `1,024 GiB` free cap** (`~USD 0.70/day`).

| Project | Volume Name | Size (GiB) | Files | Contents & Cleanup Assessment |
| :--- | :--- | :---: | :---: | :--- |
| BanglaBBQ / LLM | `mistral-nemo-12b-volume` | `45.654` | `57` | `/mistral-nemo-12b-it` HF weights -> **Tier 2A (re-downloadable HF weights)** |
| BanglaBBQ / LLM | `llama-3.1-8b-volume` | `29.926` | `54` | `/llama-3.1-8b-it` HF weights -> **Tier 2A (re-downloadable HF weights)** |
| BanglaBBQ / LLM | `ministral-8b-volume` | `29.906` | `51` | `/ministral-8b-it` HF weights -> **Tier 2A (re-downloadable HF weights)** |
| BanglaBBQ / LLM | `gemma-2-9b-volume` | `17.243` | `45` | `/gemma-2-9b-it` HF weights -> **Tier 2A (re-downloadable HF weights)** |
| BanglaBBQ / LLM | `qwen-7b-volume` | `14.196` | `45` | `/qwen-7b-it` HF weights -> **Tier 2A (re-downloadable HF weights)** |
| BanglaBBQ / LLM | `banglallm-qwen-3b-volume` | `12.670` | `51` | `/banglallm-qwen-3b-it` weights -> **Tier 2A (HF/archived model weights)** |
| BanglaBBQ / LLM | `ministral-3b-volume` | `8.734` | `42` | `/ministral-3b-it` HF weights -> **Tier 2A (re-downloadable HF weights)** |
| BanglaBBQ / LLM | `titulm-3b-volume` | `6.253` | `39` | `/titulm-3b-it` weights -> **Tier 2A (HF/archived model weights)** |
| BanglaBBQ / LLM | `qwen-3b-volume` | `5.759` | `39` | `/qwen-3b-it` HF weights -> **Tier 2A (re-downloadable HF weights)** |
| BanglaBBQ / LLM | `banglabbq-eval-volume` | `1.266` | `225` | `predictions/` & `reports/` -> **KEEP (small eval artifacts)** |
| Cattle Thesis *(A+ Done)* | `sideview-data` | `23.463` | `160,524` | `/sideviewcows2026` raw/processed data -> **Tier 2B (delete ONLY if backed up elsewhere)** |
| Cattle Thesis *(A+ Done)* | `cvb-data` | `14.565` | `226,345` | `/cvb` dataset (`226k` files) -> **Tier 2B (delete ONLY if backed up elsewhere)** |
| Cattle Thesis *(A+ Done)* | `mtl-data` | `11.472` | `87,788` | `/bcs`, `/behavior`, `/reid` -> **Tier 2B (delete ONLY if backed up elsewhere)** |
| Cattle Thesis *(A+ Done)* | `beef-behavior-data` | `6.373` | `4,337` | `/beef_behavior` -> **Tier 2B (delete ONLY if backed up elsewhere)** |
| Cattle Thesis *(A+ Done)* | `sciencedb-data` | `4.026` | `107,132` | `/dataset` -> **Tier 2B (delete ONLY if backed up elsewhere)** |
| Cattle Thesis *(A+ Done)* | `mtl-checkpoints` | `1.834` | `33` | Trained weights + `0.896 GiB` `*_smoke` checkpoints -> **Tier 2B (`*_smoke` obsolete; keep final weights if unarchived)** |

- **Tier 2A (`9` HF/LLM model-weight volumes):** **`170.341 GiB`** (`$0.511/day`).
- **Tier 2B (`6` completed Cattle Thesis volumes):** **`61.733 GiB`** (`$0.185/day`).

---

## 3. Corrected NLBS Compute & Storage Projections (`24,736` Images)

- **Compute Baseline (`rsna_val_calib`, `9,528` imgs actual billing JSON):**
  - `ap-1dMjmBD…` net 12-shard build (`excl. ~$0.0631 aborted 1800s verify`): `~USD 0.4728`
  - `ap-yaZkL5T…` 12-shard parallel verify: `USD 0.0760` (`$0.0568 CPU + $0.0192 Mem`)
  - Total RSNA net compute = **`USD 0.5488`** -> Scaled `2.5961x` (`24,736 / 9,528`), **NLBS (`24,736` imgs) projected compute = `~USD 1.42` (`USD 1.35–1.55`)**.
- **Storage Projections (`@$0.0030 / GiB-day` = `$0.090 / GiB-month` over `1,024 GiB` free quota; `7.2` days left until `2026-12-01` reset vs `USD 8.66` credits remaining):**

| Cleanup Scenario | Pre-NLBS Total (GiB) | Reclaimed (GiB) | Post-NLBS Total (`117.5–181.1 GiB` added) | Post-NLBS Daily Rate (`> 1,024 GiB`) | Monthly Equiv (`30d`) | Remaining Nov (`7.2d` Storage + `USD 1.42` Compute) | Fits `USD 8.66` Credits Before Dec 1? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. No Cleanup** | `1,271.34` | `0.00` | `1,388.86–1,452.48` | **`USD 1.09–1.29/day`** | `USD 32.8–38.6` | `USD 9.31–10.68` | **TIGHT / RISKS EXCEEDING** |
| **1. Tier 1 Only** *(7 Paper1/StageB pilots)* | `1,268.32` | `3.02` | `1,385.84–1,449.46` | `USD 1.09–1.28/day` | `USD 32.6–38.3` | `USD 9.24–10.61` | **TIGHT / RISKS EXCEEDING** |
| **2. Tier 1 + Tier 2A** *(+ 9 HF LLM vols)* | `1,097.98` | **`173.36`** | `1,215.50–1,279.12` | **`USD 0.57–0.77/day`** | `USD 17.2–23.0` | **`USD 5.56–6.94`** | **YES (`~USD 1.72–3.10` buffer)** |
| **3. Tier 1 + Tier 2A + Tier 2B** *(+ 6 Thesis vols)* | `1,036.24` | **`235.10`** | `1,153.76–1,217.38` | **`USD 0.39–0.58/day`** | `USD 11.7–17.4` | **`USD 4.23–5.60`** | **YES (`~USD 3.06–4.43` buffer)** |

---

## 4. `verify_nyu_cpu_shard` Timeout Update & Safest Low-Cost Execution Plan

1. **Completed Now:** Updated `verify_nyu_cpu_shard` in [`scripts/preprocess_paper1_nyu_cpu.py`](../../scripts/preprocess_paper1_nyu_cpu.py) from `timeout=1800` -> **`timeout=3600`** (`cpu=1.0, memory=2048`), giving `>3x` safety margin over projected `~1,135s` FUSE SHA-256 verification per NLBS shard (`2,061` files/shard). Re-verified `python -m py_compile` and **`tests/test_paper1_nyu_step1_cpu.py` (`5/5` PASS, `54.211s`)**.
2. **Safest Low-Cost Plan to Finish All `24,736` NLBS Images:**
   - **Step 1 (Cleanup First):** Delete Tier 1 obsolete Paper 1/Stage B pilot caches (`3.022 GiB` in `rsna-mammography-vol`) AND delete the `9` re-downloadable HuggingFace LLM weight volumes (`170.341 GiB`, Tier 2A — zero impact on `banglabbq-eval-volume` predictions/reports or any Breast Cancer data). This drops total storage to `1,097.98 GiB`, keeps `raw` / `RSNA` / `GMIC` / `Stage-1` / `checkpoints` 100% untouched, and keeps all remaining Nov cost + NLBS build inside the **`USD 8.66`** balance before **`2026-12-01`** (`USD 30.00` credit reset).
   - **Step 2 (Launch `nlbs_4view` CPU Build):** Keep `2 vCPU / 8 GiB`, `compression=1`, `0` GPUs, `timeout=3600` (`--num-shards 12` per protocol, or `--num-shards 16` if extra wall-time margin below `60m` is desired for `~$0.01–0.02`), costing **`~USD 1.42`** compute.
   - **Step 3 (Reclaim NYU Cache After Step 2 Inference):** Once NYU Step 2 L40S inference locks predictions cleanly (`~187–251 GiB` temporary cache served its purpose), delete `cache/paper1_nyu_uint16_v1` to return total workspace storage below **`1,024 GiB`** (`USD 0.00/day` storage overhead).
