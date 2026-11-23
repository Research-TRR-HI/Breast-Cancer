# Paper 1 — Modal Tier 1 & Tier 2A Storage Cleanup Execution Report (`hasinishrak2015`)

- **Date:** 2026-11-23 (`21:20–21:30 +06:00`)
- **Modal Profile:** `hasinishrak2015` (Starter tier, `$0.00` compute used, `0` GPUs)
- **Result:** Reclaimed **`173.363 GiB`** (`1,271.340 GiB -> 1,097.976 GiB`) across `7` obsolete pilot caches (`3.022 GiB`) and `9` byte-exact re-downloadable HuggingFace LLM volumes (`170.341 GiB`). All raw datasets, verified RSNA/GMIC/Stage-1 caches, checkpoints, `banglabbq-eval-volume`, and all `6` cattle thesis volumes verified **100% untouched**. **NLBS `24,736`-image preprocessing NOT launched (stopped for separate approval).**

---

## 1. Pre-Deletion Cryptographic Verification of Tier 2A Volumes (`9 / 9` SAFE)

Every Tier 2A volume was audited read-only against `.cache/huggingface/trees/<revision>.json` and live `https://huggingface.co/api/models/...` BEFORE any deletion. Full per-file inventories (`paths` and `bytes`) are archived in [`reports/paper1/tier2a_llm_volume_provenance.json`](tier2a_llm_volume_provenance.json).

- **Non-tree custom files found:** `0` (`extra_not_in_hf_tree: []` on all 9)
- **Byte size mismatches vs HF tree:** `0` (`size_mismatches: []` on all 9)
- **Live HF Hub API status:** `HTTP 200`, `private: False` on all 9

| Volume Deleted | Size (GiB) | Files | Exact HuggingFace `repo_id` | Commit SHA (`revision`) | Access Requirement to Re-Download |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `mistral-nemo-12b-volume` | `45.6545` | `57` (`18` repo) | `mistralai/Mistral-Nemo-Instruct-2407` | `04d8a90549d23fc6bd7f642064003592df51e9b3` | Public (`apache-2.0`, no token required) |
| `llama-3.1-8b-volume` | `29.9258` | `54` (`17` repo) | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | **Gated manual** (`HF_TOKEN` + Meta Llama 3.1 license) |
| `ministral-8b-volume` | `29.9065` | `51` (`16` repo) | `mistralai/Ministral-8B-Instruct-2410` | `2f494a194c5b980dfb9772cb92d26cbb671fce5a` | Public HF API (`MRL-0.1` in card) |
| `gemma-2-9b-volume` | `17.2430` | `45` (`14` repo) | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | **Gated manual** (`HF_TOKEN` + Google Gemma license) |
| `qwen-7b-volume` | `14.1960` | `45` (`14` repo) | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | Public (`apache-2.0`, no token required) |
| `banglallm-qwen-3b-volume` | `12.6702` | `51` (`16` repo) | `BanglaLLM/Bangla-s1k-qwen-2.5-3B-Instruct` | `c82816421594d2cbd8ff2246941c593382cab4b6` | Public (no token required) |
| `ministral-3b-volume` | `8.7340` | `42` (`13` repo) | `mistralai/Ministral-3-3B-Instruct-2512` | `b35d4dfe56c142746f54dbd64f579faab2744308` | Public (`apache-2.0`, no token required) |
| `titulm-3b-volume` | `6.2535` | `39` (`12` repo) | `hishab/titulm-llama-3.2-3b-v2.0-Instruct-v1.0` | `6a9600483daf9a3af7494e7fa91f2e72e2799b96` | Public (no token required; `trainer_log.jsonl` is upstream) |
| `qwen-3b-volume` | `5.7588` | `39` (`12` repo) | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Public (`qwen-research`, no token required) |

---

## 2. Tier 1 Obsolete Pilot Paths Deleted on `rsna-mammography-vol` (`7 / 7`, `3.021 GiB`, `810` Files)

1. `cache/paper1_nyu_uint16_v1_bench_40ex_c4_z1` (`321` files, `1.2577 GiB`)
2. `cache/paper1_gmic_uint16_v1_bench200` (`200` files, `0.8969 GiB`)
3. `cache/paper1_nyu_uint16_v1_validate_2c8g` (`161` files, `0.5437 GiB`)
4. `cache/paper1_nyu_uint16_v1_bench_5ex_c2_z1` (`81` files, `0.1560 GiB`)
5. `cache/parity_pilot_1200` (`43` files, `0.1539 GiB`)
6. `cache/bench_1000` (`2` files, `0.0115 GiB`)
7. `cache/paper1_dinov2_vits14_bench` (`2` files, `0.0015 GiB`)

---

## 3. Post-Cleanup Verified Storage & Preserved Volumes (`9` Volumes Remaining)

Read-only SDK re-scan across all `9` remaining volumes confirmed **`1,097.976 GiB`** total (`73.976 GiB` above Modal's `1,024 GiB` continuous free tier -> **`USD 0.22/day`**, down from `USD 0.74/day`):

- `nlbs-mammography-vol`: **`409.010 GiB`** (`26,997` files — `nlbs/extracted` `324.448`, `nlbs/raw` `84.562`) — **UNTOUCHED**
- `rsna-mammography-vol`: **`625.967 GiB`** (`184,033` files — `rsna/raw` `293.110`, `paper1_gmic_uint16_v1` `159.371`, `paper1_nyu_uint16_v1` `69.786`, `stage1_rsna_uint16` `65.950`, `stage1_nlbs_uint16` `35.939`, `models/mammo_clip` `1.541`, `models/paper1` `0.257`, `paper1_gmic_shards` `0.007`, `rsna/*` `0.005`, `reports` `0.002`) — **UNTOUCHED**
- `banglabbq-eval-volume`: **`1.266 GiB`** (`225` files — predictions & eval reports) — **UNTOUCHED**
- Cattle Thesis (`6` volumes = **`61.733 GiB`**): `sideview-data` (`23.463`), `cvb-data` (`14.565`), `mtl-data` (`11.472`), `beef-behavior-data` (`6.373`), `sciencedb-data` (`4.026`), `mtl-checkpoints` (`1.834`) — **ALL UNTOUCHED**

---

## 4. Revised NLBS Budget Before Separate Approval (`24,736` Images)

- **Current Starter Credits (`hasinishrak2015`):** `USD 21.34` used of `USD 30.00` -> **`USD 8.66` remaining** (`~7.1` days left until **`2026-12-01` `00:00 UTC`** reset; `billed_cost: USD 0.00`).
- **Expected NLBS CPU Build Compute (`2 vCPU / 8 GiB`, `compression=1`, `0` GPUs, `timeout=3600`):** **`~USD 1.42`** (`USD 1.35–1.55`).
- **Expected Post-NLBS Storage (`+117.52` to `+181.14 GiB`):** Total **`1,215.50–1,279.12 GiB`** (`191.50–255.12 GiB` over `1,024 GiB` -> **`USD 0.57–0.77/day`**).
- **Remaining Nov Total (`~7.1` days storage + `~USD 1.42` compute):** **`~USD 5.47–6.89` <= `USD 8.66` remaining credits** (leaves **`~USD 1.77–3.19`** headroom before Dec 1 reset).
- **Status:** Stopped cleanly; awaiting separate user approval to launch the `24,736`-image NLBS CPU build.
