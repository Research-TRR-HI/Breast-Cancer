# AnchorShift Research Logs

Welcome to **AnchorShift Research Logs** directory (`research_logs/`). This directory houses detailed, evidence-backed, reproducible records of every substantial milestone, forensic audit, dataset leakage analysis, architecture design, loss formulation, calibration verification, and benchmark evaluation in the AnchorShift breast cancer screening AI project.

---

## 1. Role & Relationship to `memory/`

To maintain clarity and separation of concerns across human researchers, AI coding agents (Antigravity), and external collaborators:

- **`memory/` (High-Level State & Dashboard)**:
  - Houses the authoritative, concise project dashboard (`state.md`), chronological overview (`history.md`), official mission (`purpose.md`), directory map (`index.md`), and developer persona guidelines (`personality.md`).
  - Read at the start of every session for rapid orientation and tracking active goals.
- **`research_logs/` (Detailed Technical Evidence & Audit Reports)**:
  - Houses deep-dive forensic audits, mathematical formulations, loss derivations, DICOM integrity verifications, patient split leak proofs, unit test traces, and copy-pasteable execution commands.
  - Every major audit, dataset split, model architecture milestone, or cloud experiment must produce a dedicated, numbered research log.

---

## 2. Mandatory Research Logging Rules

1. **Standardized Naming Convention**:
   - Every research log file must follow the pattern:
     `research_logs/YYYY-MM-DD_<topic_in_snake_case>.md`
     *(Example: `research_logs/2026-11-17_trust_reproduction_and_rsna_splits.md`)*
2. **Template Adherence**:
   - All logs must strictly follow the standard structure defined in [`research_logs/TEMPLATE.md`](TEMPLATE.md).
3. **Empirical Evidence & Zero Fabrication**:
   - Never invent numbers, patient counts, DICOM tags, test results, or commit SHAs.
   - Record exact paths, Modal volumes, split seeds, and cryptographic hashes.
4. **Unlabeled Target Quarantine Rule**:
   - NLBS cancer labels are strictly quarantined for evaluation only. Zero exposure during training, hyperparameter tuning, checkpoint selection, or threshold calibration.
5. **Index Maintenance**:
   - Whenever a new research log is created, add an entry to the master index table below (ordered newest first).
6. **Research Quality & Limitation Disclosure**:
   - Explicitly document all scanner heterogeneity, inverted polarities, unlinked cross-folder repeats, and compute constraints. Never claim a pipeline is publication-ready without empirical verification.

---

## 3. Chronological Index of Research Logs

| Date                 | Log File                                                                                                                          | Milestone / Topic                                                              | Summary of Actions & Findings                                                                                                                                                                                                                                                                                                          | Status                                           |
| :------------------- | :-------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------- |
| **2026-11-20** | [2026-11-20_stage_c_nlbs_zero_label_evaluation.md](2026-11-20_stage_c_nlbs_zero_label_evaluation.md)                             | Stage C Execution — Strict Zero-Label NLBS Evaluation (`RSNA -> NLBS`)        | Quarantined NLBS outcomes via label-free manifest (`c3d6a3b3...`), scored `26,988/26,988` images (`0` failures/`0` fallbacks), hash-locked `6,359` studies (`526434d0...`), and evaluated `5,942`-cluster CIs: both CE (`80.1%/90.1%` sens, `49.9%/30.5%` dis) & TRUST (`80.8%/84.8%` sens, `49.4%/39.8%` dis; AUROC `0.710/0.705`) **FAILED** frozen target recall (`TRUST@98` `-5.30%` vs `CE@98`), motivating Stage D UPA. | VERIFIED COMPLETE (STAGE D NEXT)                 |
| **2026-11-20** | [2026-11-20_stage_b_rsna_cache_and_training.md](2026-11-20_stage_b_rsna_cache_and_training.md)                                   | Stage B Execution — 1M Aug Cache & 20-Epoch RSNA Training (CE vs. TRUST)       | Cleared Cloud Gates 1–4, migrated Stage-1 `1520x912` PNGs (`65.9 GiB`) + canonical splits to `hasinishrak74001`, extracted & verified `1,000,000` aug-aware occurrences (`640` shards, SHA-256 `6c34a8c0...`), and completed 20-ep `ce` (`95%: 22.50%`, `98%: 14.48%`) and `trust` (`95%: 20.70%`, `98%: 17.17%`, `tau_t=0.25610`) training. | VERIFIED COMPLETE (STAGE C READY)                |
| **2026-11-18** | [2026-11-18_final_pre_training_readiness_correction.md](2026-11-18_final_pre_training_readiness_correction.md)                 | Final Pre-Training Correction (Critical Fixes A–I) & Stage-1 `uint16` Cache   | Implemented shared training engine (`src/training/engine.py`), RNG-isolated head init, strict `v2` checkpointing & feature cache validation, exact micro-batch positive weighting, CPU Stage-1 `uint16` crop cache (`src/data/stage1_cache.py`), strict LUT/laterality/segmentation diagnostics with `use_largest_cc=False`, `1,200`-img parity pilot enabling nonzero `tau_t`, pinned Python 3.12 + exact package lock across Modal scripts, `625` batches/ep (`1M` occurrences / `8.327 GB`), and `20/20` passing CPU tests (`reports/final_pre_training_readiness_audit.md`). | CONDITIONAL GO (PENDING CLOUD GATES 1–5)         |
| **2026-11-18** | [2026-11-18_pre_training_correction_and_trust_audit.md](2026-11-18_pre_training_correction_and_trust_audit.md)                     | Pre-Training Correction, TRUST Paper Audit & Augmentation-Aware Caching        | Audited repository directly against `arXiv:2609.00300v1`: fixed TRUST loss params (`m=0.10`, `lambda_dismiss=0.05`, `tau_1=0.05`, no focal alpha), CP95 dynamic `tau_t`, CP99 edge cases, full Sec. 3.2 LUT/crop/mirror/aug preprocessing, independent 20:60 sampling, final-epoch selection (no `val_calib` double-use), pinned Mammo-CLIP SHA-256, sharded aug-aware cache, full RNG resume, and 13/13 passing local CPU tests (`reports/pre_training_go_no_go.md`). | CONDITIONAL GO (PENDING CLOUD PARITY PILOT)      |
| **2026-11-18** | [2026-11-18_l40s_sweep_and_caching_benchmark.md](2026-11-18_l40s_sweep_and_caching_benchmark.md)                                   | L40S Micro-Batch Sweep & Live vs. Feature Caching Benchmark                    | Swept L40S batch sizes to OOM (`ap-mY4VfFocqIRVEnGIeV2CtD`: highest=128, fastest=96, optimal TRUST=80 at 24.90 GB / 37.55 img/s). Ran 1-feature-per-image inference/encoding benchmark (`ap-KidtVEoMzokd7fsFzpMeFo`: `0.00e+00` deterministic prob diff, `434.40s` decode vs `26.49s` encode per 1k imgs). **Clarification:** Does NOT establish stochastic-augmentation 20-epoch training equivalence or final augmented cache cost. | HISTORICAL BENCHMARK (INFERENCE/ENCODING RATE)   |
| **2026-11-18** | [2026-11-18_pre_training_blocker_audit_and_l40s_prep.md](2026-11-18_pre_training_blocker_audit_and_l40s_prep.md)                 | Pre-Training Blocker Audit, Sampling Fix & L40S Prep                           | Audited `Research-TRR-HI/Breast-Cancer` across 6 pre-training blockers: enforced strict `852/852` parity, fixed undersampled negative epochs (`full_neg_pass`), enabled dynamic `.with_options(gpu=...)` for `L40S`, documented TRUST deviations, added per-epoch `val_calib` + `_latest`/`_best` + `--resume`, and added L40S throughput benchmark. | CODE FIXES COMPLETE / PENDING USER CHECKS        |
| **2026-11-18** | [2026-11-18_stage_b_gpu_smoke_test_and_mammoclip_verification.md](2026-11-18_stage_b_gpu_smoke_test_and_mammoclip_verification.md) | Stage B Cloud GPU Smoke Test & Mammo-CLIP Weights Verification                 | End-to-end GPU smoke test on Tesla T4; resolved PyTorch 2.6 `weights_only=True` unpickler collision for `omegaconf`; verified 1.65 GB Mammo-CLIP B5 weights loaded (`checkpoint_loaded: True`); 10 RSNA DICOMs decoded in 8.75s (0.88s/img); verified peak VRAM = 0.78 GB / 14.58 GB (5.3%); confirmed frozen encoder isolation. | VERIFIED COMPLETE (CERTIFIED PASS)               |
| **2026-11-17** | [2026-11-17_trust_reproduction_and_rsna_splits.md](2026-11-17_trust_reproduction_and_rsna_splits.md)                               | Stage B TRUST Reproduction, Mammo-CLIP B5 Loading & Zero-Leakage RSNA Splits   | Complete PyTorch reproduction of TRUST (arXiv:2609.00300) focal/dismissal loss, frozen Mammo-CLIP EfficientNet-B5 loader, deterministic patient-level RSNA split (8,339 train, 1,192 train-calib, 2,382 val-calib; 0 patient leakage), BalancedBatchSampler (20 pos / 60 neg), and 5/5 unit tests passed locally.                      | CODE & SPLITS COMPLETE / AWAITING USER MODAL RUN |
| **2026-11-17** | [2026-11-17_nlbs_exam_manifest_and_leakage_clustering.md](2026-11-17_nlbs_exam_manifest_and_leakage_clustering.md)                 | Stage A3 NLBS Extraction, FRDR SHA-256 Certification & Exam Leakage Clustering | Verified 8/8 FRDR SHA-256 hashes bit-for-bit; extracted 26,988 DICOMs; resolved 149 positive patients vs 151 positive exams (`p_4334` & `p_4461` repeat visits); isolated 71 shared `StudyInstanceUID`s; derived 5,942 atomic `split_cluster_id` components via Union-Find.                                                    | VERIFIED COMPLETE / MANIFESTS COMMITTED          |
| **2026-11-17** | [2026-11-17_rsna_nlbs_dataset_audit.md](2026-11-17_rsna_nlbs_dataset_audit.md)                                                     | Stage A Complete Dataset Audit (RSNA & NLBS Volumes)                           | Complete forensic audit on CPU Modal; verified 54,706 RSNA DICOMs (4.08% patient positive), discovered 40/200 `MONOCHROME1` inverted polarities; verified 26,988 NLBS DICOMs across 3 archives, discovered Siemens Mammomat Novation DR scanner in NLBS.                                                                             | AUDIT COMPLETE / READINESS CERTIFIED             |
