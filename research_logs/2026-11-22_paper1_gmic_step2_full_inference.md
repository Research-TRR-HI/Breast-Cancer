# Research Log — 2026-11-22: `[Paper 1]` GMIC Step 2 Full L40S GPU Inference & Provenance Lock

- **Scope & Strict Gate:** Full label-free `batch_size=1` inference across Step-1 cached `(2944, 1920)` `uint16` PNGs (`rsna_val_calib`: `10,982` images / `2,382` patients; `nlbs`: `26,988` images / `6,359` studies) under a strict `$2.00 USD` Modal L40S budget on `hasinishrak2015`. Zero NLBS cancer labels accessed, zero thresholds calibrated, zero clinical evaluations executed, and zero other models launched.
- **Implementation Updates (`src/paper1/gmic_adapter.py`, `scripts/run_paper1_gmic_modal.py`, `tests/test_paper1_models_and_gate02.py`):**
  1. Added `--batch-size` (default `1`) and `--shard-indices` CLI options to `run_paper1_gmic_modal.py` with `num_shards=4` per split and `assert_no_forbidden_columns` + `usecols` excluding `cancer`.
  2. Added `validate_gmic_shard_payload` ensuring resumed shard JSONs (`paper1_gmic_shard_v2`) verify exact `batch_size`, `checkpoint_sha256` (`3b792e6d...`), `preprocess_version` (`paper1_gmic_uint16_v1`), Step 1 `preprocess_manifest_sha256`, ordered `expected_item_ids_sha256`, and finite scores before reuse (`test_11_gmic_shard_resume_validation_and_lock` PASS).
  3. After RSNA unpipelined decode took `~259s/shard` on `batch_size=1`, added bounded async CPU prefetch (`deque`, `prefetch=24`) over `8` threads while feeding exact single-image `(1, 1, 2944, 1920)` tensors to `model(bx)`, accelerating NLBS `6,747`-image shards to `140–180s` (`~45 img/s`).
  4. Added `finalize_and_lock_gmic_predictions` verifying full uniqueness, coverage (`2,382` RSNA patients, `6,359` NLBS studies, `6,184` common 4-view studies), max-view aggregation, fallback tallies, and SHA-256 locking in `reports/paper1/predictions_locked/`.
- **Execution & Modal Settled Cost (`$1.81005 USD <= $2.00` Budget):**
  - `ap-YBKnb6ULOTtuKNkEc51Q7K` (`rsna_val_calib`, `4` shards, `10,982` imgs, `80` fallbacks / `36` cases): `$1.11674 USD` (`L40S $0.89299`, `CPU $0.12996`, `Memory $0.09379`).
  - `ap-Gxxgc1VMiZ8cIB1yt7xBvr` (NLBS shard `0/4` budget gate probe, `6,747` imgs, `148.38s`): `$0.13327 USD` (`L40S $0.10518`, `CPU $0.01704`, `Memory $0.01105`). Confirmed `RSNA + 4*NLBS < $2.00`.
  - `ap-xlcdupTXpYMDZgHMizsRWx` (NLBS shards `1–3` + validated cache resume on shard `0`, `26,988` total imgs, `0` fallbacks): `$0.56005 USD` (`L40S $0.44475`, `CPU $0.06859`, `Memory $0.04671`).
  - **Settled Grand Total:** **`$1.81005 USD`** (`$45.72` workspace metered / `-$20.29` credits used / `$0.00` billed).
- **Verified Locked Outputs (`reports/paper1/gmic_step2_full_inference_report.md`):**
  - `gmic_rsna_val_calib_provenance_lock.json` (`sha256: fe65b68f...`) & `gmic_nlbs_provenance_lock.json` (`sha256: 4c84005c...`) — all `8/8` file SHA-256 digests and `37,970` image / `8,741` exam scores verified finite and label-free.
