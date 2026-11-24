# Research Log — 2026-11-24: [Paper 1] NYU Step 2 Production Hardening (`8vCPU/32GiB`), Pilot Reconciliation & RSNA-First Gate

- **Date:** 2026-11-24 (`12:37 +06:00` / `06:37 UTC`)
- **Author / Track:** Paper 1 — Multi-Model External Validation Benchmark (`nyu_fourview_pretrained`)
- **Execution Mode:** Code corrections, live billing reconciliation & CPU unit testing (`USD 0.00` GPU spend, `0` cache deletions, `0` NLBS label reads)

---

## 1. Pilot Reconciliation & Credits
- Reconciled `ap-1WtcQA2ZWaNFBzKQy5CFAU` (`USD 0.09169` settled: `0.06868 L40S + 0.01625 CPU + 0.00676 Mem`) vs non-existent/zero-cost `ap-f195vW40fG2W6K2G7n0Q78` (`USD 0.0000`, absent from Modal billing report). Combined actual pilot cost = **`USD 0.0917`**, leaving **`USD 7.50`** Starter credits (`-22.50 / 30.00`, resetting `2026-12-01` `00:00 UTC`).

## 2. Production Hardening Implemented & Tested
- Upgraded `infer_nyu_shard` in [`scripts/run_paper1_nyu_modal.py`](../scripts/run_paper1_nyu_modal.py) to `cpu=8.0, memory=32768` (`32 GiB RAM`), `batch_exams=4`, `L40S_PROD_HOURLY_USD=2.5844`, wave-based aggregate spend guard (`check_nyu_aggregate_spend_guard`, `--wave-size 2`, `--max-total-spend-usd`), strict FP32 non-autocast `eval()` check, and frozen Step-1 SHA-256 enforcement (`NYU_STEP1_MANIFEST_SHA256`, `verify_nyu_step1_against_frozen_source`).
- Created RSNA-only source calibration & lock script [`scripts/calibrate_paper1_nyu_rsna.py`](../scripts/calibrate_paper1_nyu_rsna.py) producing `nyu_rsna_calibration.{json,md}` + `nyu_rsna_calibration_lock.json`, required before `--split nlbs_4view` can run.
- Passed `py_compile` + `tests/test_paper1_nyu_step1_cpu.py` (`5/5`) + expanded `tests/test_paper1_nyu_step2.py` (`4/4` PASS, `89.261s`).

## 3. Conservative Budget Projections (`1.60 exams/sec`, `USD 2.68/hr` w/ Burst CPU)
- `rsna_val_calib` (`2,382` exams, `4` shards): `0.4524h` = **`USD 1.21`** (`USD 1.03` @ `1.84 exams/s`) — safely inside `USD 7.50` w/ `USD 3.03` buffer (`>> USD 1.50`).
- `nlbs_4view` (`6,184` exams, `4` shards): `1.1114h` = **`USD 2.98`** (`USD 2.51` @ `1.84 exams/s`). Combined conservative = **`USD 4.19`** -> run RSNA + calibration lock first, then schedule NLBS after Dec 1 reset (or reclaim `157.5 GiB` cache post-lock) to protect `USD 1.50` buffer.

## 4. Artifact Generated
- [`reports/paper1/nyu_step2_pre_infer_readiness.md`](../reports/paper1/nyu_step2_pre_infer_readiness.md)
