# state.md — Current Workspace State

## Active Goals & Todo
- [x] **Paper 1 Primary Six-Model Benchmark (`reports/paper1/six_model_final/six_model_summary.{md,json}`):** Completed & pushed (`bad3689`).
- [x] **Paper 1 Expanded Three-Experiment Protocol — Phase 1 Wave 1 (`reports/paper1/expanded/phase1_master_readiness.{md,json}`):** Completed & pushed (`6d8f8e1`).
- [x] **Paper 1 Expanded — Wave 2 + Wave 3 Execution (`~USD 3.59–3.64 <= USD 5.90` cap):** Completed & pushed (`465d3d4`).
- [x] **Final Phase 2 Readiness Audit (`2026-11-25`, `$0.00` spend, `92` tests OK `skipped=2`, `14/14` specialist OK):**
  - **Exp 1 (`dryousufmozumder`, `~$25.56` left, `963.74 GiB`):** GMIC (`bf16 bs=64`, `$1.25`) & NYU (`extract bs=16` `$4.76`, `10-TTA bs=8` `$4.88`) audited; 3 critical NYU CLI flags verified (`--max-spend-usd 6.5`, `_cached_` ckpt path, `--num-shards 4`); `< 1,024 GiB` Cleanup Gates #1 (`-245.23 GiB`) & #2 (`-159.37 GiB`) locked; `~$15.24` Phase 2 total.
  - **Exp 2 (`tigerwood693`, `~$23.62` left, `339.79 GiB`):** Integrated into `main` (`834a453..6ff3115`); finalized **4 models** (`gmic` & `nyu` `$0` reuse, `glam` `bs=4` `$1.39`, `end2end` `bs=32` `$0.55`); `petrini_twoview_efnb3` hard-blocked `blocked_upstream`, `frcnn_cad` `disqualified`; licenses & Category C verified (`~$1.94–$2.15` total).
  - **Exp 3 (`tigerwood697`, `~$24.55` left, `~243.49 GiB`):** Integrated into `main` (`246474f..15b89c6`); fixed missing `run_cli` `--stage infer` sharded orchestration, NYU-on-LUMINA hard block, CMMD `826`-exam 4-view restriction, pre-join SHA-256 locking, and dry-runs (`~$0.42` Ready-4 + `~$1.50–$2.50` Agent 1 handoff).
  - **Phase 2 Execution Plan & `$30` Budget:** Written to `reports/paper1/expanded/phase2_execution_plan.md` (`~$22.74–$23.95` total w/o IMDB, Reserve `~$6.05–$7.26`; `~$24.34–$26.75` w/ IMDB).
- [ ] **FINAL APPROVAL GATE: Awaiting user `GO PHASE 2`** before launching full training or full inference.
- [ ] **External Access Blocker (`IMDB`):** User-assisted manual IBIA download (`MAMOS_1000000004`, `MAMOS_1000000050`) or MIDAS access + `modal volume put` to `tigerwood697`.

## Last Session (Convo 4ea0ac0d-67d7-40c5-b5c8-ebd4cc866015 — `2026-11-25`)
- Executed 3-subagent Final Phase 2 Readiness Audit across `dryousufmozumder`, `tigerwood693`, and `tigerwood697` (`$0.00` spend, `0` running apps).
- Integrated Exp 2 (`4`-model lock) and Exp 3 (`--stage infer` orchestration + 4 new tests) into `main`.
- Verified full test suite (`Ran 92 tests in 175.075s — OK (skipped=2)`), specialist suite (`14/14 OK` in `3.835s`), and Exp 3 `dry_run` (`cmmd` & `lumina`).

## Current Blockers & Notes
- **IMDB (Exp 3):** IBIA portal requires interactive web download (`MAMOS_1000000004`, `MAMOS_1000000050`).
- **Petrini (`blocked_upstream`) & Faster R-CNN (`disqualified`):** Permanently excluded from Exp 2 (no repair spend).
- **LUMINA NYU 4-View:** Hard-blocked in `scripts/run_paper1_exp3_modal.py` (malignant cases lack contralateral views).
- **DO NOT** run full Phase 2 training/inference until explicit user `GO PHASE 2`.
