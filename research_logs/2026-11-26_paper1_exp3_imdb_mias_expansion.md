# [Paper 1 Expanded] IMDB India & MIAS UK External Dataset Expansion

**Date:** `2026-11-26`  
**Track:** Paper 1 Exp 3 (`studies/paper1/expanded_three_experiment_protocol_v1.md`)  
**Modal Profile:** `tigerwood697` (`rsna-mammography-vol`, Volume v1, `0` running apps, `$0.00` billed, `$6.54` free Starter credits remaining)  
**Status:** **COMPLETE — MIAS (5 models, `N=161`) & IMDB 4-View FFDM (6 models, `N=1,665`) LOCKED & EVALUATED**

---

## 1. Summary of Actions & Key Findings

1. **MIAS UK (`legacy_film_exploratory`, `N=161` bilateral MLO pairs / `322` PGMs):**
   - Downloaded `miasdbv1.21.zip` (`1.506 GiB`, `0` corrupt) from Cambridge Apollo (`CAM.105113`).
   - Detected 90° landscape orientation (`chest wall at top`) and `0` CC views (`nyu_fourview_rsna_finetuned` hard-blocked).
   - Rotated 90° upright in `decode_pgm_array()`, built `322/322` Stage-1 PNGs (`0` fallbacks), and ran 5 eligible models (`$0.1889` GPU).
   - **Result:** Catastrophic SFM vs FFDM domain collapse (`AUROC 0.4486–0.5567`; `dinov2_rsna` `7.69%` sens / `48` missed, `gmic_rsna_finetuned` `15.38%` sens / `44` missed).
2. **IMDB India (`mixed_diagnostic_opportunistic`, `12,674` DICOMs / `107.84 GiB`):**
   - Programmatically downloaded `r1` (`3,577` DICOMs, `62.24 GiB`) and `r2` (`9,097` DICOMs, `45.60 GiB`) from IBIA (`0` corrupt, `100%` Hologic Selenia Dimensions).
   - Patched missing `(0002,0010) TransferSyntaxUID` in `decode_dicom_array()`, built `12,674/12,674` Stage-1 `1520x912` PNGs (`0` fallbacks).
   - Audited `582` cross-release numeric ID collisions (`0` `StudyInstanceUID` overlap), separated `FFDM` (`7,865`) vs `SM` (`4,809`), and isolated the matched 4-view FFDM cohort (`1,665` exams / `6,660` imgs: `304` `r1` + `1,361` `r2`; `252` path-pos, `32` path-benign, `1,379` no follow-up, `2` ambiguous).
   - Executed all 6 RSNA-supervised models concurrently (`--num-shards 4` + 16-thread FUSE prefetch) and SHA-256 locked all `1,665` exams (`0` fallbacks across all 6 models).
   - **Result:** All 6 models drop to near-chance discrimination (`AUROC 0.5035–0.5515` combined; `0.5040–0.5632` `r2` path-confirmed `N=223`) with severe upward score shift (`nyu` `99.60%` sens / `0.85%` apparent spec; `dinov2` `99.21%` / `0.35%`; `ce_b5` `93.25%` / `11.91%`; `trust_b5` `91.27%` / `14.67%`).
3. **Reports & Artifacts:** `reports/paper1/expanded/exp3/imdb_mias/{imdb_results.md,imdb_results.json,imdb_results_metrics.csv,imdb_execution_summary.md,mias_results.md,mias_results_metrics.csv,final_external_expansion_summary.md}`.
