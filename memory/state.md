# state.md — Current Workspace State

## Active Goals & Todo
- [x] **Paper 1 Primary Six-Model Benchmark (`reports/paper1/six_model_final/six_model_summary.{md,json}`):** Completed & pushed (`bad3689`).
- [x] **Paper 1 Expanded Three-Experiment Protocol — Phase 1 + Wave 2 + Wave 3 + Readiness Audit:** Completed & pushed (`9537fee`).
- [x] **Paper 1 Expanded — PHASE 2 COMPLETE ACROSS ALL 3 EXPERIMENTS (`$22.51 <= $30.00` cap, `$0.00` billed):**
  - **Exp 1 (`dryousufmozumder`, `$11.61` Phase 2 settled, `~$13.95` left, `934.51 GiB < 1 TiB`):**
    - `gmic_rsna_finetuned` (`11efb7c6...`): RSNA `val_calib` AUROC `0.8570`; NLBS 4-view (`N=6,184`) AUROC **`0.8624`**, `@98%` recall **`96.03%`** (`6` missed), **`33.15%` dismissal**.
    - `nyu_fourview_rsna_finetuned` (`b42a8b7d...`): RSNA `val_calib` AUROC `0.8013`; NLBS 4-view (`N=6,184`) AUROC **`0.7846`**, `@98%` recall **`96.69%`** (`5` missed), **`14.68%` dismissal**.
    - Both authorized GMIC cache cleanups executed (`-245.23 GiB` + `-159.37 GiB`).
  - **Exp 2 (`tigerwood693`, `$2.42` Phase 2 settled, `~$21.20` left, `reports/paper1/expanded/exp2/exp2_summary.{md,json}`):**
    - Category C NLBS 4-view (`N=6,184`, `151` pos): `gmic_pretrained` **`0.8541`**, `glam_pretrained` **`0.8006`**, `nyu_fourview_pretrained` **`0.7925`**, `end2end_cbis_ddsm` **`0.5292`**. (`petrini_twoview_efnb3` `blocked_upstream`, `frcnn_cad` `disqualified`.)
  - **Exp 3 (`tigerwood697`, `$4.84` Phase 2 settled, `~$19.71` left, `reports/paper1/expanded/exp3/eval/exp3_summary.{md,json}`):**
    - 6-model NLBS (`$0` reuse) + CMMD (`N=1,775` / `826` 4-view) + 5-model LUMINA (`N=468`, NYU hard-blocked): CMMD 4-view `nyu_fourview_rsna_finetuned` `0.7137`, `gmic_rsna_finetuned` `0.7059` (CMMD full `0.7671`), `dinov2_rsna` `0.6567`, `trust_b5` `0.6474`, `ce_b5` `0.6414`, `resnet50_rsna` `0.4741`; LUMINA `gmic_rsna_finetuned` `0.7426`, `trust_b5` `0.6708`, `ce_b5` `0.6693`.
- [x] **MIAS UK (`legacy_film_exploratory`) Complete (`$0.19` GPU, `161` pairs / `52` pos):** Downloaded (`1.506 GiB`), rotated 90° upright (`322/322`, `0` fallbacks), evaluated 5 view-agnostic models (`nyu_fourview` hard-blocked): all collapse on digitized film (`AUROC 0.4486–0.5567`; `dinov2` `7.69%`, `gmic` `15.38%` @98%; `reports/paper1/expanded/exp3/imdb_mias/mias_results.md`).
- [x] **IMDB India (`mixed_diagnostic_opportunistic`) 6-Model 4-View FFDM Complete (`N=1,665`, `0` fallbacks):** All 6 RSNA-trained models locked & evaluated across Analysis A (`N=284`: `61` `r1` + `223` `r2`) and Analysis B (`N=1,663`: `304` `r1` + `1,359` `r2`). Near-chance discrimination (`AUROC 0.5035–0.5515`) & upward score shift (`nyu` `99.60%` sens / `0.85%` spec; `dinov2` `99.21%` / `0.35%`; `ce_b5` `93.25%` / `11.91%`; `trust_b5` `91.27%` / `14.67%`). `tigerwood697` Ephemeral Apps `$23.46` (`$0.00` billed, `$6.54` free credits left, `0` running apps). See `reports/paper1/expanded/exp3/imdb_mias/final_external_expansion_summary.md`.
- [x] **Paper 1 Complete Manuscript Dossier (`Paper1_Complete_Manuscript_Dossier.md`):** Synthesized all Stage A–C, Primary 6-Model, Exp 1–3, IMDB, and MIAS results into one paper-writing master reference.
- [ ] **Stage 7 Manuscript Synthesis & Paper 2 (`AnchorShift`):** Ready for manuscript table/figure synthesis and Paper 2 Stage D UPA decision.

## Last Session (Convo 0049effa-3768-43a3-b704-b6446823173b — `2026-11-26`)
- Completed full external dataset expansion on `tigerwood697` (`0` running apps, `$0.00` billed): MIAS 5-model (`N=161`, `AUROC 0.4486–0.5567`) and IMDB 6-model 4-view FFDM (`N=1,665`, `AUROC 0.5035–0.5515`), with all predictions SHA-256 locked, Analysis A & B evaluated (`n_boot=2000`), and consolidated reports pushed to GitHub.

## Current Blockers & Notes
- **IMDB (Exp 3):** Pending user manual download from IBIA (`MAMOS_1000000004`, `MAMOS_1000000050`).
- **Petrini (`blocked_upstream`) & Faster R-CNN (`disqualified`):** Permanently excluded from Exp 2.
