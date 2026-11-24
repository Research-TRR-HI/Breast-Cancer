# state.md — Current Workspace State

## Active Goals & Todo
- [x] **Paper 1 Primary Six-Model Benchmark (`reports/paper1/six_model_final/six_model_summary.{md,json}`):** Completed & pushed (`bad3689`).
- [x] **Paper 1 Expanded Three-Experiment Protocol — Phase 1 Wave 1 (`reports/paper1/expanded/phase1_master_readiness.{md,json}`):** Completed & pushed (`6d8f8e1`).
- [x] **Paper 1 Expanded — Wave 2 + Wave 3 Execution (`USD 3.59 <= USD 5.90` cap):**
  - **Exp 1 (`dryousufmozumder`, `~USD 0.58`):** `GMIC train_all` (`43,724` imgs) verified, `rsna_val_calib` (`62.06 GiB`) transferred & verified, `rsna_train_pilot` (`200` exams, `5.913 GiB`) NYU cache built & verified; GMIC L40S pilot (`bf16`, `bs=64`, `49.001 imgs/sec`, `8.22 GB` VRAM, proj `0.531h` / `USD 1.25`) & NYU L40S pilot (`extract bs=16` `6.83 exams/sec`, `10-TTA eval bs=8` `1.26 exams/sec`) completed.
  - **Exp 2 (`tigerwood693`, `~USD 2.05`):** `glam_nlbs` (`97.307 GiB`) transferred & verified, Option B1 `1152x896` cache (`27,020` imgs, `0` failures) built, `GLAM` (`a647ba91...`), `End2End` Keras->TorchScript (`1c93c4c3...`, `max_abs_diff=4.51e-10`), and `Petrini` downloaded; `GLAM` (`bs=4`, `11.674 imgs/sec`, `USD 0.037`, proj `USD 1.39`) & `End2End` (`bs=32`, `29.385 imgs/sec`, `USD 0.028`, proj `USD 0.55`) L40S pilots completed. `petrini_twoview_efnb3` blocked upstream (both HF & Google Drive `.pt` truncated at `88.0 MiB` with `405 MiB` zero-padding and no ZIP EOCD `PK\x05\x06`).
  - **Exp 3 (`tigerwood697`, `~USD 0.96`):** `exp3_backbones` (`1.80 GiB`), `LUMINA` (`1,285` imgs / `468` exams, neutral pseudonyms, `0` fallbacks), and `CMMD` (`5,202` imgs / `1,775` exams, `0` fallbacks) downloaded & preprocessed; `LUMINA` & `CMMD` L40S pilots completed (`ce_b5` `bs=32`, `trust_b5` `bs=32`, `dinov2_rsna` `bs=128`/`32`, `resnet50_rsna` `bs=32`). `IMDB` still pending manual IBIA download.
- [ ] **Phase 2 Approval Gate:** Awaiting user approval for full Phase 2 fine-tuning (Exp 1) and full external inference (Exp 2 & Exp 3).

## Last Session (Convo 4ea0ac0d-67d7-40c5-b5c8-ebd4cc866015)
- Executed Wave 2 + Wave 3 concurrently across `dryousufmozumder`, `tigerwood693`, and `tigerwood697` (`~USD 3.59` total billed vs `USD 5.90` cap).
- Fixed NYU L40S TF32 feature tolerance (`atol=1e-3`, `score_diff=0.0`), GMIC `Tuple` import, GLAM PyTorch-2 `get_max_window` floor division + 7 legacy checkpoint keys, End2End double-softmax bug, Exp 3 `Stage1CropResult` attribute access + `canUse32BitIndexMath` guard, and CMMD `ViewCodeSequence` fallback.
- Verified `petrini_twoview_efnb3` upstream corruption forensically (`nz_end = 92,274,688`, zero ZIP EOCD on both HF and Google Drive).

## Current Blockers & Notes
- **IMDB (Exp 3):** IBIA portal requires interactive web login/download.
- **Petrini Two-View EfficientNet-B3 (Exp 2):** Upstream `.pt` files (`2024-10-01` & `2024-07-01`) lack ZIP EOCD (`PK\x05\x06`) and are zero-padded after `88.0 MiB`; 4 valid Exp 2 models (`gmic_pretrained`, `nyu_fourview_pretrained`, `glam_pretrained`, `end2end_cbis_ddsm`) are ready.
- **DO NOT** run full Phase 2 training/inference until explicit user approval.
