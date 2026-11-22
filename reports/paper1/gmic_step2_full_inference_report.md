# Paper 1 — GMIC Step 2 Full L40S GPU Inference & Provenance Lock Report

**Date:** 2026-11-22  
**Modal Profile:** `hasinishrak2015` (Apps `ap-YBKnb6ULOTtuKNkEc51Q7K`, `ap-Gxxgc1VMiZ8cIB1yt7xBvr`, `ap-xlcdupTXpYMDZgHMizsRWx`, NVIDIA L40S)  
**Preprocessed Input Cache Consumed:** `rsna-mammography-vol:/cache/paper1_gmic_uint16_v1/` (zero preprocessing repeated)  
**Status:** **COMPLETE — LOCKED & VERIFIED (`37,970` unique images, `$1.8101 USD` settled Modal cost <= `$2.00` budget)**  
**Strict Quarantine Gate:** **Zero NLBS cancer labels accessed, zero thresholds selected, zero clinical evaluation run, and no other models started.**

---

## 1. CLI, Provenance & Shard-Resume Validation Upgrades
* **Explicit `--batch-size` CLI Option & Default `batch_size=1`:** Added `--batch-size` (default `1`) and `--shard-indices` to [`scripts/run_paper1_gmic_modal.py`](../../scripts/run_paper1_gmic_modal.py) with `num_shards=4` per split (`rsna_val_calib`, `nlbs`), preserving verified single-image FP32 behavior without cuDNN batch-reduction drift.
* **Strict Incompatible-Cache Rejection (`validate_gmic_shard_payload`):** Implemented in [`src/paper1/gmic_adapter.py`](../../src/paper1/gmic_adapter.py) and unit-tested (`test_11_gmic_shard_resume_validation_and_lock`). Every shard JSON (`paper1_gmic_shard_v2`) records and validates on resume:
  * `batch_size` (exact match, e.g., `1`)
  * `checkpoint_sha256` (`3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991` for `sample_model_1.p`)
  * `preprocess_version` (`paper1_gmic_uint16_v1`) & Step 1 `preprocess_manifest_sha256`
  * Ordered `expected_item_ids` + `expected_item_ids_sha256`
  * Finite-value checks on all `benign_pred` / `malignant_pred` scores.
* **Bounded CPU Prefetch Pipeline:** After observing synchronous `pool.map` of length `1` bottleneck RSNA decoding at `~10.6 img/s`, added a bounded `deque` (`prefetch=24`) over 8 decoder threads overlapped with exact `batch_size=1` L40S forward passes — preserving identical `[1, 1, 2944, 1920]` single-image inference while boosting NLBS end-to-end throughput >4x (`~45.5 img/s`).

---

## 2. Per-Shard Execution & Actual Settled Modal Costs

### 2.1 RSNA `val_calib` (`10,982` images across `4` shards, App `ap-YBKnb6ULOTtuKNkEc51Q7K`)
| Shard | Images | Fallbacks | Decode (s) | GPU Infer (s) | Container Wall (s) | Peak VRAM | Est. Cost (USD) | Resumed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0 / 4` | `2,746` | `15` | `259.114` | `36.837` | `315.674` | `0.292 GB` | `$0.20710` | `false` |
| `1 / 4` | `2,746` | `20` | `263.389` | `37.941` | `321.886` | `0.292 GB` | `$0.21118` | `false` |
| `2 / 4` | `2,745` | `24` | `256.909` | `34.479` | `312.316` | `0.292 GB` | `$0.20490` | `false` |
| `3 / 4` | `2,745` | `21` | `257.971` | `36.231` | `315.141` | `0.292 GB` | `$0.20675` | `false` |
| **Total** | **`10,982`** | **`80`** | `1037.383` | `145.488` | `1265.017` | **`0.292 GB`** | `$0.82993` | `0 / 4` |

### 2.2 NLBS Full Label-Free Cohort (`26,988` images across `4` shards, Apps `ap-Gxxgc1VMiZ8cIB1yt7xBvr` & `ap-xlcdupTXpYMDZgHMizsRWx`)
| Shard | Images | Fallbacks | Decode (s) | GPU Infer (s) | Container Wall (s) | Peak VRAM | Est. Cost (USD) | Resumed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0 / 4` | `6,747` | `0` | `13.808` | `82.535` | `148.381` | `0.292 GB` | `$0.09735` | `false` -> `true` (`ap-xlcdup...`) |
| `1 / 4` | `6,747` | `0` | `13.771` | `82.208` | `140.089` | `0.292 GB` | `$0.09191` | `false` |
| `2 / 4` | `6,747` | `0` | `15.993` | `100.151` | `180.532` | `0.292 GB` | `$0.11844` | `false` |
| `3 / 4` | `6,747` | `0` | `16.015` | `98.715` | `179.987` | `0.292 GB` | `$0.11808` | `false` |
| **Total** | **`26,988`** | **`0`** | `59.587` | `363.609` | `648.989` | **`0.292 GB`** | `$0.42578` | `1 / 4` (`shard 0`) |

### 2.3 Official Settled Modal Billing (`hasinishrak2015` · `modal billing report --show-resources`)
| Modal App ID | Scope | L40S (USD) | CPU (USD) | Memory (USD) | Settled Total (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `ap-YBKnb6ULOTtuKNkEc51Q7K` | RSNA `val_calib` (`4` shards, `10,982` imgs) | `$0.89299` | `$0.12996` | `$0.09379` | **`$1.11674`** |
| `ap-Gxxgc1VMiZ8cIB1yt7xBvr` | NLBS probe shard `0 / 4` (`6,747` imgs) | `$0.10518` | `$0.01704` | `$0.01105` | **`$0.13327`** |
| `ap-xlcdupTXpYMDZgHMizsRWx` | NLBS shards `1–3` + `shard 0` resume verify | `$0.44475` | `$0.06859` | `$0.04671` | **`$0.56005`** |
| **GMIC Step 2 Grand Total** | **All `37,970` images (`8` shards, `bs=1`)** | **`$1.44292`** | **`$0.21559`** | **`$0.15155`** | **`$1.81005 USD`** |

*Workspace summary post-run:* `Metered Cost = $45.72`, `Credits = -$20.29`, `Free Storage = -$25.34`, `Network Egress Allowance = -$0.08`, **`Billed Cost = $0.00`** (`~$9.71 USD` remaining in Starter workspace credits).

---

## 3. Verification Summary (`37,970` Unique Images, Coverage & SHA-256 Locks)

* **Total Unique Image Predictions:** **`37,970 / 37,970`** (`10,982` RSNA `val_calib` + `26,988` NLBS; zero duplicates; `100%` set match against label-free manifests; `all_scores_finite = true`; all scores in `[0, 1]`).
* **Complete Examination & 4-View Coverage (Label-Free):**
  * **RSNA `val_calib`:** `2,382 / 2,382` patients (`100%`), covering all `9,528` images / `2,382` patients in `datasets/paper1/rsna_val_calib_4view_selection.csv`. Label-free image `malignant_pred` summary: `min=0.000326`, `median=0.032904`, `max=0.992370`; case max summary: `min=0.004634`, `median=0.084093`, `max=0.992370`. Fallbacks: **`80` views across `36` cases**.
  * **NLBS Full & 4-View Common Cohort:** `6,359 / 6,359` unique `StudyInstanceUID` exams (`100%`), covering all `6,184 / 6,184` studies (`24,736` selected views) in `datasets/paper1/nlbs_4view_common_cohort_label_free.csv`. Label-free image `malignant_pred` summary: `min=0.001237`, `median=0.021416`, `max=0.999662`; exam max summary: `min=0.002809`, `median=0.063755`, `max=0.999662`. Fallbacks: **`0` views / `0` cases**.
* **Locked Prediction Artifacts (`reports/paper1/predictions_locked/`):**
  * **RSNA Lock Manifest:** `gmic_rsna_val_calib_provenance_lock.json` (`sha256: fe65b68f87794637a0ab13a6241d6a0a948858410974e48e1e1e009c8b63e7ad`, Step 1 manifest `sha256: f950713d...`):
    * `gmic_rsna_val_calib_image_scores.json`: `a5f80bd496edc4338005c2c692f5ed43ee91c4132adf5be8a676ca28eed4dcfa`
    * `gmic_rsna_val_calib_image_scores.csv`: `e8601950bdceed0e83cf67bbdd9c2695af74d353b6954716e35a829c098670b6`
    * `gmic_rsna_val_calib_case_scores.csv`: `b6a9e44a119238d1ebf2d7c7e31126e8897bb0610d01e73bb4d81f076fe96354`
    * `gmic_rsna_val_calib_fallback_log.json`: `591742a5203994efd3e1e1b41f91095e110d4d0548200e0233903e3332bfd190`
  * **NLBS Lock Manifest:** `gmic_nlbs_provenance_lock.json` (`sha256: 4c84005c751fe321e13262e2d2eea4d73d969e527e6cf458183cbe44d02cbc44`, Step 1 manifest `sha256: 32e2b806...`):
    * `gmic_nlbs_image_scores.json`: `4f8cf3cf044fc9d7e7d467e859a5751b42cba7bed752802e56b5e62985b39c25`
    * `gmic_nlbs_image_scores.csv`: `2b55bd19e25d657ed6ea3e3625543d77bac5d5341cb722ac6f224523346fe3b2`
    * `gmic_nlbs_case_scores.csv`: `94521810b9a78e7047a39d71bc54618ea2c7e95600a6f4fba717b79fdb3a6b36`
    * `gmic_nlbs_fallback_log.json`: `3a04b63b99b01952178c5b0269703555653d24f72485c89aa382e0732fa644be`

**Next Step (Stopped Here Per Instructions):** GMIC Step 2 is complete and locked. Next approved stage is **NYU Four-View (`nyu_fourview_pretrained`)** or GMIC Step 3 source threshold calibration once authorized.
