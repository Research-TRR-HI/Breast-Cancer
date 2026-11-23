# Research Log — 2026-11-23: [Paper 1] Verified Tier 1 + Tier 2A Modal Storage Cleanup (`173.363 GiB` Reclaimed)

- **Date:** 2026-11-23 (`21:20–21:30 +06:00`)
- **Author / Track:** Paper 1 — Multi-Model External Validation Benchmark (`nyu_fourview_pretrained`)
- **Modal Workspace:** `hasinishrak2015` (CPU/GPU = `0`, `$0.00` SDK cleanup + verification)

---

## 1. Cryptographic Pre-Deletion Safety Verification
- Audited all `9` Tier 2A LLM volumes (`170.341 GiB`, `403` files) against embedded `.cache/huggingface/trees/<sha>.json` and live HuggingFace Hub API (`https://huggingface.co/api/models/...`):
  - Verified **`extra_not_in_hf_tree: []`**, **`size_mismatches: []`**, `HTTP 200`, `private: False` for all `9` volumes, confirming zero unique fine-tuned weights, predictions, logs, or custom checkpoints.
  - Saved full provenance (repo IDs, commit SHAs, licenses/gating, complete file byte lists) in [`reports/paper1/tier2a_llm_volume_provenance.json`](../reports/paper1/tier2a_llm_volume_provenance.json).
- Audited all `7` Tier 1 directories (`3.022 GiB`, `810` files) on `rsna-mammography-vol`: verified every entry belonged strictly to superseded Paper 1/Stage B benchmarks or pilot caches.

## 2. Executed Deletions & Post-Cleanup Re-Scan
- Deleted the `7` Tier 1 paths (`cache/paper1_nyu_uint16_v1_bench_40ex_c4_z1`, `cache/paper1_gmic_uint16_v1_bench200`, `cache/paper1_nyu_uint16_v1_validate_2c8g`, `cache/paper1_nyu_uint16_v1_bench_5ex_c2_z1`, `cache/parity_pilot_1200`, `cache/bench_1000`, `cache/paper1_dinov2_vits14_bench`) and `9` Tier 2A volumes (`mistral-nemo-12b-volume`, `llama-3.1-8b-volume`, `ministral-8b-volume`, `gemma-2-9b-volume`, `qwen-7b-volume`, `banglallm-qwen-3b-volume`, `ministral-3b-volume`, `titulm-3b-volume`, `qwen-3b-volume`).
- Verified all `9` remaining volumes read-only: total storage dropped from **`1,271.340 GiB -> 1,097.976 GiB`** (`173.363 GiB` reclaimed; `73.976 GiB` over `1,024 GiB` free cap = `USD 0.22/day`).
- Confirmed untouched: `nlbs-mammography-vol` (`409.010 GiB`), core `rsna-mammography-vol` (`625.967 GiB`), `banglabbq-eval-volume` (`1.266 GiB`), and all `6` cattle thesis volumes (`61.733 GiB`).

## 3. Revised Budget & Stop Status
- Remaining credits: **`USD 8.66`** (`2026-12-01` UTC reset). Post-NLBS storage (`1,215.50–1,279.12 GiB` -> `USD 0.57–0.77/day`) + `~USD 1.42` build compute = **`~USD 5.47–6.89`** over remaining November days (`~USD 1.77–3.19` buffer). Stopped cleanly pending separate NLBS approval.

## 4. Artifacts Generated
- [`reports/paper1/tier2a_llm_volume_provenance.json`](../reports/paper1/tier2a_llm_volume_provenance.json)
- [`reports/paper1/nyu_step1_cleanup_execution_report.md`](../reports/paper1/nyu_step1_cleanup_execution_report.md)
