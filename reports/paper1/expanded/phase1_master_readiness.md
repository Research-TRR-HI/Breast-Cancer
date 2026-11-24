# Paper 1 Expanded Three-Experiment Benchmark — Phase 1 Master Readiness Report

**Date:** 2026-11-24  
**Repository:** `https://github.com/Research-TRR-HI/Breast-Cancer` (`main` base: `bad3689`)  
**Scope:** Phase 1 Wave 1 Planning, Audits, Engineering & Local Verification (**$0.00 Phase 1 cloud spend; zero full runs launched**)  
**Protocol:** [expanded_three_experiment_protocol_v1.md](../../../studies/paper1/expanded_three_experiment_protocol_v1.md)

---

## 1. Experiment Readiness

| Experiment | Branch | Existing Work Reused | New Preparation Completed | Remaining Blockers | Readiness |
|---|---|---|---|---|---|
| **Exp 1: Common RSNA Supervision** (`dryousufmozumder`) | `paper1-exp1-phase1` (`fa997b7`) | `ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna` completed; `293.11 GiB` `rsna/raw`; `43,724` GMIC train/tc PNGs (`245.22 GiB`); `34,264` NYU eval PNGs (`157.51 GiB`); `Stage-1` caches | Fixed critical `--stage eval` fall-through bugs; added independent fine-tuned SHA-256 loading & resumable shards; audited GMIC schedule (`bs=16, steps=400, ep=8`); extended `preprocess_paper1_nyu_cpu.py` for `rsna_train`/`train_calib`; added BN/GaussianNoise `.eval()` + `176.9 MiB` NYU `head_only` feature caching; `35/35` runnable tests PASS (`27/27` paper1 + `8/8` transfer) | GMIC `train_all` missing `manifest.csv` (`--stage verify`, `~$0.03`); sequential cleanup gates needed post-GMIC-train & post-GMIC-eval to keep `< 1 TiB` (`$0/day`); `external/` needed at `modal run` CWD | **CONDITIONAL GO** |
| **Exp 2: Original Pretrained Mammography Models** (`tigerwood693`) | `paper1-exp2-phase1` | Locked `gmic_pretrained` (`6,359`) & `nyu_fourview_pretrained` (`6,184`) NLBS predictions (`$0.00`); `35.94 GiB` NLBS Stage-1 PNGs; `97.31 GiB` GMIC NLBS PNGs on `hasinishrak2015`; DINOv2 assets preserved | Audited all 6 candidates + `nyukat/mammography_metarepository` (`4f9b8a08`); classified Category A/B/C original thresholds (zero RSNA calib); built `exp2_adapters.py` + `run_paper1_exp2_modal.py`; `8/8` unit tests PASS (`0.610s`) | **`frcnn_cad` DISQUALIFIED** (dead ELTE weight URL + CUDA 8.0/Caffe incompatible with L40S); decision needed: keep 5 non-RSNA models vs replace with `ianpan/mammoscreen` (Category A `0.0072/0.0127`, note RSNA-trained) or `dpetrini/vindr-convnext` (BI-RADS >=3); Wave 2 downloads + Keras->PyTorch parity for `end2end_cbis_ddsm` | **CONDITIONAL GO** |
| **Exp 3: Multiple Unseen Populations** (`tigerwood697`) | `paper1-exp3-phase1` (`a0a15d2`) | Locked NLBS predictions (`6,184`/`6,359`) for `ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna` (`$0.00`); `resnet50` ckpt + `Stage-1` caches (`102 GiB`) preserved | Deep metadata/license/label audits for `IMDB`, `CMMD`, `LUMINA`; built universal label-free manifest pipeline (`external_datasets.py`, `universal_eval.py`, `run_paper1_exp3_modal.py`, `evaluate_paper1_exp3.py`) + Agent 1 handoff validator; `8/8` new + `3/3` Stage-C tests PASS | `IMDB` IBIA requires interactive browser download (no public `curl`/`wget` API) + `r1/r2` ID dedup + FFDM/SM split; `LUMINA` discards contralateral malignant breast (`no NYU 4-view`); `CMMD` only `826/1,775` 4-view (`biopsy-enriched`); sync 3 completed checkpoints to `tigerwood697` & await Agent 1 checkpoints for GMIC/NYU fine-tuned | **CONDITIONAL GO** |

---

## 2. Modal Account Status (Read-Only Verified `2026-11-24`)

| Account | Starting / Current Balance | Actual Phase 1 Spend | Remaining Balance | Storage Usage (`< 1,024 GiB` Free Tier) | Running Apps |
|---|---|---|---|---|---|
| **`dryousufmozumder`** (Exp 1) | `~$26.12` (`$3.88` used of `$30` Starter) | **`$0.00`** | **`~$26.12`** | `895.76 GiB` (`128.24 GiB` headroom, `$0.00/day`) | `0` |
| **`tigerwood693`** (Exp 2) | `~$25.77` (`$4.23` used of `$30` Starter) | **`$0.00`** | **`~$25.77`** | `224.85 GiB` (`799.15 GiB` headroom, `$0.00/day`) | `0` |
| **`tigerwood697`** (Exp 3) | `~$25.48` (`$4.52` used of `$30` Starter) | **`$0.00`** | **`~$25.48`** | `~165.26 GiB` (`~858.74 GiB` headroom, `$0.00/day`) | `0` |

*(Note: Combined Wave 2 + Wave 3 + Phase 2 cost across all 3 accounts is projected at `~$18.60–$27.25 USD`, which fits within both the individual `~$25` account balances and the `$30` additional project cap — subject to explicit per-wave approval.)*

---

## 3. Resource Performance (Measured Baselines vs Planned Allocations)

| Agent | CPU Cores / Container | Parallel Containers | GPU | Measured / Reference Throughput |
|---|---|---|---|---|
| **Agent 1** (`dryousufmozumder`) | `2–4 vCPU / 8 GiB` (CPU PNG), `1 vCPU / 2 GiB` (transfer) | `16–24` (CPU build), `16 × 6 = 96` streams (transfer) | `1× L40S` (`6–8 vCPU / 16–32 GiB`, NVMe `/tmp`) | **Measured:** `181.02 MiB/s` transfer; `5.1 img/s` GMIC pilot (`bs=8`, `1.08 GB` VRAM); `1.84 exams/s` NYU eval (`bs=4`, `9.31 GB` VRAM). **Planned:** `bs=16` GMIC + `176.9 MiB` NYU head feature cache |
| **Agent 2** (`tigerwood693`) | `1 vCPU / 2 GiB` (transfer), `2–4 vCPU` (downloads) | `16 × 6 = 96` streams (transfer) | `1× L40S` (`8 vCPU / 24 GiB`, NVMe `/tmp`) | **Measured:** `130.7 img/s` DINOv2 (`2.40 GB` VRAM). **Planned:** `200`-unit L40S pilots for `glam`, `end2end`, `petrini`, and replacement |
| **Agent 3** (`tigerwood697`) | `1 vCPU / 2 GiB` (downloads/sync), `2–4 vCPU` (Stage-1 PNG) | `8–16` shards | `1× L40S` (`6 vCPU / 16 GiB`, NVMe `/tmp`) | **Measured:** `~270 img/s` ResNet-50 NVMe inference (`98.7s` for `26,988` NLBS imgs). **Planned:** `200`-img L40S pilots on `LUMINA`, `CMMD`, `IMDB` |

---

## 4. Phase 2 Execution Projection (Measured vs Extrapolated Separated)

| Experiment | Wave 2 (Download / Cache) | Wave 3 (L40S Pilots) | Phase 2 CPU Runtime | Phase 2 GPU Runtime | Total Est. Wall Runtime | All-Inclusive Projected Cost |
|---|---|---|---|---|---|---|
| **Exp 1** (`dryousufmozumder`) | `~$2.40–$2.62` (GMIC verify `$0.03` + RSNA `val_calib` transfer `$0.15` + NYU `train_all` PNGs `$2.20–$2.44`) | `~$0.12` (GMIC `bs=16` + NYU `bs=4` probes) | `~1.5–2.0h` (parallel sharded) + `<5s` head train | `~3.8–5.0h` L40S (`GMIC` train+eval + `NYU` feat extract + 10-TTA eval) | `~5.5–7.0h` | **`~$13.50–$17.50`** *(EXTRAPOLATED from pilot & Step 2)* |
| **Exp 2** (`tigerwood693`) | `~$0.25–$0.35` (`97.31 GiB` GMIC NLBS transfer + 4 ckpts + Keras parity) | `~$0.20–$0.40` (`200`-unit x 3–4 models) | `~0.25h` (transfer/weights/eval) | `~0.7–1.4h` L40S (`GLAM`, `End2End`, `Petrini`, optional `R1`) | `~1.0–1.8h` | **`~$2.15–$4.20`** *(EXTRAPOLATED)* |
| **Exp 3** (`tigerwood697`) | `~$1.00` (sync 3 ckpts + `LUMINA` `$0.08` + `CMMD` `$0.25` + `IMDB` `$0.65`) | `~$0.25` (`200`-img pilots across datasets) | `~0.5–1.0h` (download + Stage-1 PNGs + CPU eval) | `~0.6–1.5h` L40S (`19,700` new imgs + pending `gmic/nyu` NLBS) | `~1.2–2.5h` | **`~$2.95–$5.55`** *(EXTRAPOLATED)* |
| **COMBINED TOTAL** | **`~$3.65–$3.97`** | **`~$0.57–$0.77`** | **`~2.3–3.3h`** | **`~5.1–7.9h` L40S** | **`~5.5–7.0h` parallel** | **`~$18.60–$27.25`** |

---

## 5. Scientific Eligibility Summary

| Model or Dataset | Eligible? | Limitations & Caveats | Required Decision |
|---|---|---|---|
| **Exp 1: `gmic_rsna_finetuned` & `nyu_fourview_rsna_finetuned`** | **YES (Exploratory)** | Post-observation supplementary; malignant-only loss (no benign RSNA labels); requires sequential cleanup gates to stay `< 1 TiB` | Approve Wave 2 verify/transfer + sequential cleanup gates |
| **Exp 2: `gmic_pretrained` & `nyu_fourview_pretrained`** | **YES (Reused, Cat C)** | Locked NLBS reuse (`$0.00`); single ckpts vs 5-copy ensembles; no published numerical cutoff `tau` | None (ready) |
| **Exp 2: `glam_pretrained`** | **YES (Cat C)** | Exact GMIC preprocessing parity; uncalibrated output scores; NYU Non-Commercial vs AGPL-3.0 license conflict; SHA-256 pinned in Wave 2 | Approve Wave 2 download + `97.31 GiB` transfer |
| **Exp 2: `end2end_cbis_ddsm`** | **CONDITIONAL (Cat C / opt. B)** | CBIS-DDSM digitized-film domain; Google Drive download + Keras 2.0.5 -> PyTorch conversion parity (`<1e-5`) required in Wave 2 | Approve Wave 2 download + Keras conversion |
| **Exp 2: `petrini_twoview_efnb3`** | **YES (Cat C / opt. B)** | MIT license, LFS SHA-256 `82add409...` verified; CBIS-DDSM trained; strict `[CC_RGB, MLO_RGB]` chest-wall-left (`12,368` pairs) | Approve Wave 2 download |
| **Exp 2: `frcnn_cad`** | **NO (DISQUALIFIED)** | Dead ELTE weight URL (HTTP 452, no mirror) + CUDA 8.0/Caffe incompatible with L40S + lesion detector semantics | **CHOOSE:** Proceed with **5** strict non-RSNA models **OR** approve **`ianpan/mammoscreen`** (Cat A `0.0072/0.0127`, *note: RSNA-trained*) **OR** **`dpetrini/vindr-convnext`** (BI-RADS >=3) |
| **Exp 3: `NLBS` (Canada)** | **YES (Screening)** | Reused `$0.00` for 4 models; primary population-screening dismissal benchmark | None (ready) |
| **Exp 3: `CMMD` (China)** | **YES (Diagnostic Stress)** | Public TCIA NBIA (`21.29 GiB`, CC BY 4.0); `826/1,775` 4-view (`949` 2-view); **100% biopsy-enriched (`0` screening dismissal claims)** | Approve Wave 2 download |
| **Exp 3: `LUMINA` (Turkey)** | **YES (View-Agnostic Stress)** | Public OSF (`8.48 GiB` RAW DICOMs, CC BY-NC 4.0); IDs `1..218` collide; malign discards contralateral breast (**NO NYU 4-view**) | Approve Wave 2 download |
| **Exp 3: `IMDB` (India)** | **CONDITIONAL (Mixed Cohort)** | IBIA open access has **no `curl`/`wget` API** (requires manual web download or MIDAS request); `r1/r2` ID dedup + FFDM vs SM split; `~12.1%` cancer (no screening dismissal claims) | User to download `MAMOS_1000000004` & `MAMOS_1000000050` via IBIA UI (or request MIDAS) |

---

## 6. Parallel Dependency Plan (Phase 2)
- **Can Start Immediately Upon Wave 2 Approval (Zero Cross-Agent Blocking):**
  1. **Agent 1 (`dryousufmozumder`):** Wave 2 GMIC verify + `rsna_val_calib` transfer (`~$0.18`) -> Wave 3 GMIC `bs=16` pilot (`~$0.04`) -> Phase 2 GMIC fine-tuning & NLBS evaluation -> NYU `train_all` + feature caching & evaluation.
  2. **Agent 2 (`tigerwood693`):** Completely independent — Wave 2 `97.31 GiB` transfer + 3–4 checkpoint downloads/parity (`~$0.30`) -> Wave 3 `200`-unit L40S pilots (`~$0.30`) -> Phase 2 NLBS inference.
  3. **Agent 3 (`tigerwood697`):** Independent for **4 completed RSNA-supervised models** (`ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`) — Wave 2 checkpoint sync + `LUMINA` & `CMMD` downloads/preprocessing (`~$0.35`, plus `IMDB` once uploaded) -> Wave 3 pilots -> Phase 2 external inference on `LUMINA`, `CMMD`, `IMDB`.
- **Agent 1 -> Agent 3 Handoff Dependency:**
  - Once Agent 1 finishes GMIC & NYU RSNA fine-tuning and RSNA `val_calib` calibration (`exp1_handoff_manifest.json` + `.pt` checkpoints + SHA-256s + frozen `tau_98`/`tau_95`), Agent 3 runs ONLY the two new fine-tuned models on `CMMD` (`826` 4-view / `5,202` imgs) and `IMDB` (and `LUMINA` for `gmic_rsna_finetuned`).

---

## 7. Single Consolidated Approval Request (Wave 2 + Wave 3 First)

Before launching full Phase 2 runs, approve **Merge + Wave 2 + Wave 3 Pilots** (`~USD 4.22–$4.74` total across all 3 accounts):
1. **Merge Branches into `main` & Run Unskipped Integration Tests** (`$0.00`, from `D:\Breast Cancer Research` where `external/` repos reside).
2. **Exp 2 Shortlist Decision:** Confirm **5 strict non-RSNA models** vs adding `ianpan/mammoscreen` (with RSNA-pretraining disclosure) vs `dpetrini/vindr-convnext` (BI-RADS >=3 caveat).
3. **Authorize Wave 2 + Wave 3 (`~USD 4.22–$4.74` combined):**
   - **Exp 1 (`dryousufmozumder`, `~$2.52–$2.74`):** GMIC `train_all` `--stage verify` (`~$0.03`) + `rsna_val_calib` transfer (`~$0.15`) + Wave 3 GMIC `bs=16` pilot (`~$0.04`), *or* hold NYU `train_all` PNGs (`~$2.25`) until after GMIC training + approved GMIC train cache cleanup (`-245.22 GiB`) so storage never exceeds `1,024 GiB` (`$0.00/day`).
   - **Exp 2 (`tigerwood693`, `~$0.45–$0.75`):** `glam_nlbs` `97.31 GiB` transfer + `GLAM`/`Petrini`/`End2End` checkpoint pinning & Keras->PyTorch parity + `200`-unit L40S pilots.
   - **Exp 3 (`tigerwood697`, `~$1.25`):** Sync `ce_b5`/`trust_b5`/`dinov2` checkpoints + `LUMINA` & `CMMD` CPU downloads & Stage-1 preprocessing + `200`-img L40S pilots (`IMDB` queued for manual IBIA browser download).
4. **Expanded Budget Cap Acknowledgment:** Acknowledge that combined Wave 2 + Wave 3 + Phase 2 across all 3 accounts (`~$18.60–$27.25`) fits within the `USD 30` additional cap if kept lean (`~$18.60–$27.25 <= $30`), with explicit Phase 2 go/no-go after Wave 3 pilot receipts.
