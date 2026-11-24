# [Paper 1 Expanded] Final Phase 2 Readiness Audit (Three-Experiment Benchmark)

**Date:** `2026-11-25`  
**Track:** Paper 1 Expanded (`studies/paper1/expanded_three_experiment_protocol_v1.md`)  
**Status:** **PHASE 2 AUDIT COMPLETE — AWAITING `GO PHASE 2`** (`$0.00` additional spend during audit)

---

## 1. Live Modal Balances & Storage Audit (`$0` Read-Only)
- **`dryousufmozumder` (Exp 1):** `0` running apps, `$4.44` credits used (`~$25.56` left), **`963.7435 GiB`** across 6 volumes (`60.2565 GiB` under `1,024 GiB`, `$0.00/day`).
- **`tigerwood693` (Exp 2):** `0` running apps, `$6.38` credits used (`~$23.62` left), **`339.7930 GiB`** across 9 volumes (`684.21 GiB` free, `$0.00/day`), `cache/paper1_dinov2_vits14` intact.
- **`tigerwood697` (Exp 3):** `0` running apps, `$5.45` credits used (`~$24.55` left), **`~243.49 GiB`** across 7 volumes (`180.22 GiB` in `rsna-mammography-vol`, `$0.00/day`).

## 2. Technical & Scientific Finalization
- **Exp 1 (`dryousufmozumder`):** Audited GMIC (`bf16, bs=64`, `49.001 img/s`, `steps_per_epoch=75`, `scaled_lr=2.8284e-4`, `warmup=12`, `$1.25`) & NYU (`extract bs=16` `$4.76`, `10-TTA bs=8` `$4.88`, multi-draw cache diff `3.57e-4`). Caught 3 critical CLI requirements (`--max-spend-usd 6.5` for extraction, `--checkpoint-path ..._cached_...` for eval, `--num-shards 4` for `nlbs_4view` eval) and locked sequential `< 1,024 GiB` Cleanup Gates #1 (`-245.23 GiB`) and #2 (`-159.37 GiB`).
- **Exp 2 (`tigerwood693`):** Locked **4 eligible models** (`gmic_pretrained` & `nyu_fourview_pretrained` `$0` reuse; `glam_pretrained` `bs=4` `$1.39`; `end2end_cbis_ddsm` `bs=32` `$0.55`). Marked `petrini_twoview_efnb3` as `blocked_upstream` and kept `frcnn_cad` `disqualified`. Documented licenses (`AGPLv3`, `NYU Internal Non-Commercial` on GLAM, `CC BY-NC-SA 4.0` on End2End) & strict Category C AUROC/AUPRC policy.
- **Exp 3 (`tigerwood697`):** Implemented missing `run_cli` `--stage infer` sharded orchestration in `scripts/run_paper1_exp3_modal.py`, enforced NYU hard-block on `lumina` and `826`-exam 4-view restriction on `cmmd`, verified `$0.00` NLBS reuse, and documented manual IBIA/MIDAS download steps for `IMDB`.

## 3. Verification & Budget
- **Full Suite:** `Ran 92 tests in 175.075s — OK (skipped=2)` (`90/90` runnable PASS).
- **Specialist Suite:** `Ran 14 tests in 3.835s — OK` (`14/14` PASS).
- **Recalculated Budget vs `$30.00` Cap:** `~$3.59–$3.64` (Wave 2+3) + `~$19.10–$20.31` (Phase 2 w/o IMDB) = **`~$22.74–$23.95`** (Reserve `~$6.05–$7.26`; `~$24.34–$26.75` w/ IMDB). See `reports/paper1/expanded/phase2_execution_plan.md`.
