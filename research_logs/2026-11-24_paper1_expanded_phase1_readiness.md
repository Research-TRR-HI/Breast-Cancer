# [Paper 1] Expanded Three-Experiment Phase 1 Readiness & Audits

**Date:** 2026-11-24  
**Track:** Paper 1 (`studies/paper1/expanded_three_experiment_protocol_v1.md`)  
**Status:** PHASE 1 WAVE 1 COMPLETE (`$0.00` Cloud Spend) — CONDITIONAL GO FOR WAVE 2 + WAVE 3 APPROVAL

---

## 1. Summary of Decisions & Findings

1. **GitHub & Protocol Hygiene:** Verified `origin/main` at `bad3689` (matching prompt baseline); kept completed 6-model benchmark (`reports/paper1/six_model_final/six_model_summary.{md,json}`) and `studies/paper1/protocol_v1.md` completely immutable; created `studies/paper1/expanded_three_experiment_protocol_v1.md` with explicit post hoc expansion disclosure and an adversarial literature matrix (`MEDINFO 2025/2026`, *Sensors* June 2026, *J. Digital Imaging* `arXiv:2206.12407`, `Garrucho 2022`, `MC-SDS 2025`, `MammoDG 2026`, `TRUST 2026`, `UPA 2023`, `Bernstein 2026`, `Dataset-Origin Signatures 2026`).
2. **Experiment 1 (`paper1-exp1-phase1` `fa997b7`, `dryousufmozumder` `~$26.12` remaining, `895.76 GiB` used):**
   - Verified `293.11 GiB` RSNA raw DICOMs, `245.22 GiB` GMIC `train`/`train_calib` PNGs (needs `--stage verify --split train_all`, `~$0.03`), and `157.51 GiB` NYU eval PNGs (`hasinishrak2015` holds `159.37 GiB` GMIC eval cache).
   - Fixed critical `--stage eval` fall-through bugs in `run_paper1_{gmic,nyu}_finetune_modal.py`; added independent SHA-256 checkpoint loading, resumable eval shards, RSNA-first calibration gates, `set_{gmic,nyu}_backbone_eval_mode` BN/noise freezes, `preprocess_paper1_nyu_cpu.py` `rsna_train`/`train_calib` (`100%` 4-view eligible), and `176.9 MiB` NYU `head_only` frozen feature caching (`27/27` runnable `test_paper1_*.py` + `8/8` `test_transfer_*.py` PASS).
   - Defined sequential `< 1,024 GiB` cleanup plan (peak `<= 957.82 GiB`, `$0.00/day` overage) with projected Wave 2–Phase 2 spend `~$13.50–$17.50`.
3. **Experiment 2 (`paper1-exp2-phase1` `1b8ead3`, `tigerwood693` `$25.77` remaining, `224.85 GiB` used):**
   - Audited `gmic_pretrained` & `nyu_fourview_pretrained` (`$0.00` locked NLBS reuse), `glam_pretrained` (`2944x1920` GMIC PNG reuse, uncalibrated Category C, NYU Non-Commercial vs AGPL-3.0 conflict), `end2end_cbis_ddsm` (whole-image GDrive `.h5`, Keras 2.0.5 -> PyTorch parity gate), `petrini_twoview_efnb3` (MIT, LFS SHA-256 `82add409...`, `[CC_RGB, MLO_RGB]` chest-wall-left), and `nyukat/mammography_metarepository` (`4f9b8a08`).
   - **Disqualified `frcnn_cad`** (dead ELTE weight URL `HTTP 452`, CUDA 8.0/Caffe incompatible with L40S, generic lesion detector semantics); proposed `ianpan/mammoscreen` (Apache-2.0, Category A `0.0072`/`0.0127`, note RSNA Split 1 pretraining) or `dpetrini/vindr-convnext` (MIT, BI-RADS >=3) vs keeping 5 strict non-RSNA models.
   - Built `src/paper1/exp2_adapters.py` + `scripts/run_paper1_exp2_modal.py` (`8/8` `test_paper1_exp2_adapters.py` PASS); projected Wave 2–Phase 2 spend `~$2.15–$4.20`, peak `~323.7 GiB`.
4. **Experiment 3 (`paper1-exp3-phase1` `a0a15d2`, `tigerwood697` `~$25.48` remaining, `~165.26 GiB` used):**
   - Audited `NLBS` (`$0.00` reuse for 4 models), `IMDB` (`3,219` pts / `12,674` DICOMs, IBIA interactive UI only, `r1/r2` `pt_XXX` collision + FFDM vs SM + mixed diagnostic/opportunistic), `CMMD` (TCIA NBIA `21.29 GiB`, CC BY 4.0, `826` 4-view + `949` 2-view, `100%` biopsy-enriched diagnostic), and `LUMINA` (OSF `8.48 GiB`, CC BY-NC 4.0, colliding IDs `1..218`, unilateral malignant -> view-agnostic only).
   - Built `src/paper1/external_datasets.py`, `src/paper1/universal_eval.py`, `scripts/run_paper1_exp3_modal.py`, and `scripts/evaluate_paper1_exp3.py` (`8/8` `test_paper1_exp3_universal.py` + `3/3` Stage-C PASS); projected Wave 2–Phase 2 spend `~$2.95–$5.55`, peak `~430.91 GiB < 1 TiB`.
5. **Master Reports:** Consolidated into `reports/paper1/expanded/phase1_master_readiness.{md,json}`.
