# Paper 1 Expanded — Master Phase 2 Readiness & Budget Reconciliation Report

**Generated:** `2026-11-25` (Audit Base `main` after Exp 2 + Exp 3 integration)  
**Project Cap:** `$30.00` additional Modal usage (`~$3.59–$3.64` spent in Wave 2+3 -> **`~$26.36–$26.41` project headroom**)  
**Approval Gate:** **AWAITING `GO PHASE 2`** — zero full Phase 2 training or inference launched. See [phase2_execution_plan.md](phase2_execution_plan.md) for all exact commands, storage gates, and stop/resume procedures.

---

## 1. Integrated Subagent Branches & Verified Test Suites

- **Exp 1 (`paper1-exp1-phase1`, `dryousufmozumder`):** GMIC (`bf16, bs=64`, `49.001 img/s`, `$1.25` proj) + NYU (`extract bs=16`, `6.8278 exam/s`, `176.86 MiB`, `$4.76` proj; `10-TTA bs=8`, `1.2611 exam/s`) audited; 3 critical NYU CLI flags verified; sequential `< 1,024 GiB` storage lifecycle table locked.
- **Exp 2 (`paper1-exp2-phase1`, `tigerwood693`):** Integrated into `main` (`834a453..6ff3115`). Finalized **4 eligible models** (`gmic_pretrained` & `nyu_fourview_pretrained` `$0` reuse; `glam_pretrained` `bs=4` `$1.39`; `end2end_cbis_ddsm` `bs=32` `$0.55`). `petrini_twoview_efnb3` hard-blocked as `blocked_upstream`; `frcnn_cad` `disqualified`. Licenses & strict Category C policy verified.
- **Exp 3 (`paper1-exp3-phase1`, `tigerwood697`):** Integrated into `main` (`246474f..15b89c6`). Fixed missing `run_cli` `--stage infer` orchestration with deterministic resumable shards, duplicate/missing exam guards, pre-join SHA-256 locking, `nyu` hard-block on `lumina`, and `826`-exam 4-view restriction on `cmmd`.
- **Tests Re-Run on `main` After Final Code Edits:**
  - Full suite (`python -m unittest discover -s tests -p "test_*.py" -v`): **`Ran 92 tests in 175.075s — OK (skipped=2)`** (`90/90` runnable PASS; 2 skipped only for uncached local DINOv2/ResNet-50 weights verified on Modal).
  - Specialist suite (`python -m unittest discover -s tests -p "test_paper1_specialist_finetuning.py" -v`): **`Ran 14 tests in 3.835s — OK`** (`14/14` PASS).
  - Exp 3 CLI `--stage dry_run`: verified exit 0 on `cmmd` (`5,202` imgs / `1,775` exams / `826` 4-view) and `lumina` (`1,285` imgs / `468` exams).

---

## 2. Live Modal Account Balances, Storage & Recalculated Phase 2 Budget

| Modal Profile | Experiment | Running Apps | Credits Used (`billing summary`) | Remaining Credits (`$30` cap) | Live Storage (`GiB` / `1,024`) | Daily Storage Fee | Recalculated Phase 2 Est. |
|---|---|---:|---:|---:|---:|---:|---:|
| `dryousufmozumder` | **Exp 1** | `0` | `$4.44` | **`~$25.56`** | `963.74 GiB` (`60.26` free) | `$0.00/day` | **`~$15.24`** |
| `tigerwood693` | **Exp 2** | `0` | `$6.38` | **`~$23.62`** | `339.79 GiB` (`684.21` free) | `$0.00/day` | **`~$1.94–$2.15`** |
| `tigerwood697` | **Exp 3** | `0` | `$5.45` | **`~$24.55`** | `~243.49 GiB` (`~780.51` free) | `$0.00/day` | **`~$1.92–$2.92`** *(+`$1.60–$2.80` w/ IMDB)* |
| **Combined** | **All 3** | **`0`** | **`+$3.64` Wave 2+3** | **`~$73.73` credits** | All `< 1 TiB` | **`$0.00/day`** | **`~$19.10–$20.31`** *(w/o IMDB)* |

- **Project-Wide `$30.00` Additional Cap Reconciliation:** `~$3.59–$3.64` (Wave 2+3 actual) + `~$19.10–$20.31` (Phase 2 w/o IMDB) = **`~$22.74–$23.95`** (Reserve **`~$6.05–$7.26`**); with conditional IMDB: **`~$24.34–$26.75`** (Reserve **`~$3.25–$5.66`**).

---

## 3. Consolidated Readiness Categories

- **Ready to Execute Immediately (on `GO PHASE 2`):**
  - Exp 2: `glam_pretrained` (`bs=4`), `end2end_cbis_ddsm` (`bs=32`), plus `$0` GMIC/NYU reuse on `tigerwood693`.
  - Exp 3 (Ready-4): `ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna` on `cmmd` & `lumina`, plus `$0` NLBS reuse on `tigerwood697`.
  - Exp 1 (Track A Steps 1–3): GMIC `bf16 bs=64` train -> `rsna_val_calib` eval -> `calibrate_rsna` on `dryousufmozumder`.
- **Ready After Storage Preparation / Dependency:**
  - Exp 1 (Track A Steps 4–5 & Track B Steps 6–10): gated on User-Approved Cleanup Gate #1 (`-245.23 GiB` GMIC train PNGs) and Gate #2 (`-159.37 GiB` GMIC eval PNGs) to stay `< 1,024 GiB`.
  - Exp 3 (Agent 1 Handoff): `gmic_rsna_finetuned` (CMMD + LUMINA) & `nyu_fourview_rsna_finetuned` (CMMD `826` 4-view) gated on Agent 1 completion & `--handoff-json`.
- **Blocked by External Access:**
  - Exp 3 `IMDB`: awaiting manual IBIA portal download (`MAMOS_1000000004`, `MAMOS_1000000050`) or authorized MIDAS CC BY 4.0 access.
- **Blocked by Scientific Eligibility / Upstream Corruption:**
  - `petrini_twoview_efnb3` (`blocked_upstream`), `frcnn_cad` (`disqualified`), `nyu_fourview_rsna_finetuned` on `lumina` (hard-blocked: missing contralateral malignant views).
