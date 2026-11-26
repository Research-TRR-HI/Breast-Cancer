# Paper 1 — Complete Master Research & Manuscript Dossier

**Last Updated:** `2026-11-26`  
**Repository:** `https://github.com/Research-TRR-HI/Breast-Cancer` (`main`)  
**Related Study (Strictly Separate):** Paper 2 — *AnchorShift* (`AnchorShift_Research_Context.md`, Stage D unlabeled-target adaptation / UPA — NOT included in Paper 1 claims).  
**Purpose:** Single self-contained, verified, end-to-end technical and empirical reference containing **every** protocol decision, dataset audit, model specification, cryptographic provenance hash, statistical result, 95% confidence interval, failure-mechanism analysis, limitation, and manuscript blueprint from Project Stage A through the final IMDB + MIAS external expansion.

---

## 1. Manuscript Framing, Title Candidates & Scientific Boundaries

### 1.1 Candidate Manuscript Titles
- **Primary Multi-Model Multi-Country Title:** *When High-Sensitivity Screening Mammography AI Fails to Transport: A Ten-Model, Five-Country External Benchmark of Source-Frozen Triage Decisions*
- **Focused Screening-Triage Title:** *Discrimination Versus Threshold Stability in Cross-Hospital Breast Cancer Screening Triage: Benchmarking RSNA-Supervised and Pretrained Mammography Models Across Six Cohorts*

### 1.2 Core Research Questions & Hypotheses
1. **Primary Screening Triage Question (NLBS):** When materially different mammography AI architectures (Mammo-CLIP CE/TRUST, frozen DINOv2 + head, fine-tuned ResNet-50, and pretrained/RSNA-fine-tuned specialist architectures GMIC & NYU Four-View) are calibrated to empirical `98%` (primary) and `95%` (secondary) sensitivity targets using RSNA labels alone, do their frozen decision cutoffs (`score < tau` dismissed) preserve cancer sensitivity and dismissal utility on an unseen Canadian population-screening cohort (NLBS)?
2. **Experiment 1 (Common RSNA Supervision Hypothesis — `H1_exp1`):** Does re-training/fine-tuning specialist mammography architectures (`gmic_rsna_finetuned`, `nyu_fourview_rsna_finetuned`) on identical RSNA partitions (`train` / `train_calib` / `val_calib`) recover or alter source-frozen threshold transportability compared with NYU-pretrained checkpoints?
3. **Experiment 2 (Original Pretrained Discrimination Hypothesis — `H2_exp2`):** Without any RSNA supervision or calibration (Category C threshold-independent evaluation), how heterogeneous is external discrimination (`AUROC`, `AUPRC`) across original mammography models (`gmic_pretrained`, `glam_pretrained`, `nyu_fourview_pretrained`, `end2end_cbis_ddsm`) on NLBS?
4. **Experiment 3 (Multi-Country & Acquisition-Domain Stress Test — `H3_exp3`):** Across four additional unseen external cohorts spanning biopsy-enriched FFDM (`CMMD` China, `LUMINA` Turkey), mixed tertiary-oncology FFDM (`IMDB` India), and legacy digitized screen-film (`MIAS` UK), do source-frozen RSNA models fail through **score-distribution shift**, **discrimination collapse**, or both?

### 1.3 Mandatory Retrospective Disclosures & Non-Claims
- **Post-Hoc Protocol Extensions:**
  - Stage C (`ce_b5` `90.07%` and `trust_b5` `84.77%` @98% sensitivity on NLBS) was observed on `2026-11-20` before adding the 4 models in the primary 6-model benchmark (`2026-11-21`).
  - Primary 6-model NLBS results were observed on `2026-11-24` before launching the three-experiment expansion (`Exp 1–3`, `2026-11-24`–`2026-11-26`).
  - Disclose openly as **post-hoc protocol extensions** protected by strict label-blind inference manifests and pre-join SHA-256 prediction locks.
- **Anti-Salami-Slicing Boundary:** Paper 1 contains **no new adaptation algorithm** (all adaptation methods — UPA, AnchorShift — belong strictly to Paper 2).
- **Independent Implementation Disclosure:** `ce_b5` and `trust_b5` are our **independent PyTorch reproductions** of Mammo-CLIP (`b5-model-best-epoch-7.tar`) + TRUST loss (`arXiv:2609.00300`), evaluated **strictly zero-target-label** (`Protocol 1`), NOT the original authors' `Protocol 3` (which used 20% labeled NLBS cases for threshold recalibration).
- **Screening vs Non-Screening Boundary:** **NLBS is our ONLY population-screening cohort.** CMMD, LUMINA, IMDB, and MIAS **must NEVER** be pooled with NLBS or used to claim clinical screening workload dismissal.

---

## 2. Adversarial Literature Positioning Matrix

| Prior Study | Venue / Year / ID | Scope & Models | What They Measured | Overlap Threat | Paper 1 Residual Distinction |
|---|---|---|---|---|---|
| **Meta-Repository of Screening Mammography Classifiers** (Chledowski et al.) | MEDINFO 2025 / *SHTI* 2026 (`10.3233/SHTI251263`, `nyukat/mammography_metarepository`) | 5 models across 7 international datasets; Dockerized evaluation | Discrimination (`AUROC`/`AUPRC`) + uncertainty | **HIGH** | Isolates **source-frozen `98%/95%` empirical triage thresholds**, paired RSNA-supervised vs direct-pretrained regimes, pre-join SHA-256 locks, and case-level missed-cancer overlap. |
| **Cross-Sensor & Cross-Population Generalization** (Gabbualoy et al.) | *Sensors* June 2026 (`10.3390/s26123911`) | ResNet-50, DINOv2-B/14, Rad-DINO, Mammo-CLIP B5, Mammo-FM; CBIS-DDSM -> CMMD, DMID, MIAS | AUC drops (`0.165–0.320`) & sensitivity @ `95%` specificity | **HIGH** | Modern FFDM source (**RSNA**) -> population screening (**NLBS**) + CMMD/LUMINA/IMDB/MIAS at **high-sensitivity rule-out thresholds (`98%/95%` recall)** rather than 95% specificity. |
| **Robustness of Deep Networks for Mammography** | *J. Digital Imaging* 2023 (`10.1007/s10278-023-00943-5`) | Cross-dataset CNN replication across public archives | Replication AUC degradation & dataset/split pitfalls | **MODERATE** | Separates **ranking degradation** (`AUROC` / `AUPRC`) from **score-distribution shift** (`KS` / median shift) at clinical rule-out thresholds. |
| **TRUST** | `arXiv:2609.00300` (2026) | Mammo-CLIP B5 + dynamic dismissal loss; RSNA -> NLBS (`12.95%` @98%) | Uses **20% labeled NLBS** cases to recalibrate threshold (`Protocol 3`) | **MODERATE** | Tests **zero target labels** (`Protocol 1`) across 10 configurations and proves source-frozen cutoffs fail under domain/threshold drift. |
| **UPA** (Roschewitz et al.) | *Nat. Commun.* 2023 (`10.1038/s41467-023-42396-y`) | Unlabeled CDF score alignment | Unlabeled target adaptation (`Protocol 2`) | **MODERATE** | Reserved for Paper 2 (`AnchorShift`); Paper 1 establishes the unadapted baseline failure mechanisms. |
| **Bernstein et al.** | *PLOS Digit. Health* 2026 (`10.1371/journal.pdig.0001231`) | Risk-benefit statistical framework for AI screening rule-out | Single-model safety trade-offs | **LOW-MODERATE** | Empirical multi-architecture, multi-country stress test of source-frozen rule-out. |
| **Dataset-Origin Signatures** | `arXiv:2607.15416` (2026) | Cross-dataset origin detectability & shortcuts | Shortcut bias when mixing CMMD/CBIS-DDSM with screening | **SUPPORTING** | Justifies strict non-pooling between NLBS screening and CMMD/LUMINA/IMDB/MIAS. |

---

## 3. Complete Dataset Inventory, Audits & Denominator Reconciliation

### 3.1 Master Six-Dataset Summary Table

| Dataset | Country | Clinical Role | Raw Format & Hardware | Verified Files / Raw Size | Patients / Clusters | Examinations | Cancer+ | Non-Cancer | 4-View Eligible | Screening Dismissal Claims? |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| **RSNA** *(Source)* | USA / Australia | Organized screening (multi-site) | DICOM (`MONOCHROME2` + `20% MONOCHROME1`; 10 machines, 2 sites) | `54,706` DICOMs (`293.11 GB`) | `11,913` | `11,913` (`val_calib`: `2,382`) | `486` (`val_calib`: `97`) | `11,427` | `11,913` (`100%`) | Source training & calibration |
| **NLBS** *(Primary External)* | Canada | Routine provincial screening (`2010–2011` / `2008–2010`) | DICOM (`100%` 12-bit `MONOCHROME2`; `99.92%` GE Senographe Essential, `21` imgs Siemens) | `26,988` DICOMs (`84.562 GB` `.7z`, `8/8` FRDR SHA-256) | `5,942` clusters (`5,997` folders) | `6,359` unique StudyUIDs | `151` (`149` clusters) | `6,208` | **`6,184`** (`151` pos, `5,868` clusters) | **YES (ONLY valid cohort)** |
| **CMMD** *(Exp 3)* | China | Biopsy-enriched diagnostic | DICOM (GE Senographe DS) | `5,202` DICOMs (`21.292 GiB`) | `1,775` | `1,775` | `1,310` (`73.8%`) | `465` (benign) | **`826`** (`765` pos, `61` neg) | **NO** |
| **LUMINA** *(Exp 3)* | Turkey | Multi-vendor biopsy-enriched diagnostic | DICOM/PNG (`12–14` bit, `6+` vendors; neutral pseudonyms) | `1,285` matched (`1,824` raw) | `468` | `468` | `218` (`46.6%`) | `250` | **`0` usable** (unilateral malign) | **NO** |
| **IMDB r1** *(Exp 3)* | India | Tertiary oncology (`61.9%` prior-cancer FU, `17%` symp., `3.2%` screen) | DICOM (`100%` Hologic Selenia Dimensions, `MONOCHROME2`, 12-bit FFDM + 10-bit SM) | `3,577` DICOMs (`62.24 GiB`, `0` corrupt) | `583` (`0000–0582`, release-ns) | `583` (`585` StudyUIDs) | `73` (`72` path, `1` ambig) | `510` (`1` benign, `509` no FU) | **`304` FFDM** (`60` path-pos, `1` benign, `243` no FU) | **NO** |
| **IMDB r2** *(Exp 3)* | India | Mixed tertiary oncology / diagnostic | DICOM (`100%` Hologic Selenia Dimensions, `MONOCHROME2`, 12-bit FFDM + 10-bit SM) | `9,097` DICOMs (`45.60 GiB`, `0` corrupt) | `2,636` (`pt_001–pt_2636`, `582` num. coll.) | `2,636` (`2,651` StudyUIDs) | `316` (`315` path, `1` ambig) | `2,320` (`49` benign, `2,267` no FU, `4` ambig) | **`1,361` FFDM** (`192` path-pos, `31` benign, `1,136` no FU, `2` ambig) | **NO** |
| **MIAS v1.21** *(Exp 3)* | UK | Legacy digitized screen-film (1994 NHSBSP) | PGM (`P5` 8-bit `uint8`, landscape `4320x1600..5200x4000`, chest wall at top) | `322` PGMs (`1.506 GiB` zip, `0` corrupt) | `161` pairs | `161` bilateral MLO pairs | `52` (`32.3%`) | `109` (`56` benign, `53` norm) | **`0`** (`0` CC views) | **NO** |

### 3.2 Critical Dataset Audits & Reconciliations

#### A. RSNA Split & Zero-Retrain `val_calib` Redesign (`datasets/rsna/splits_seed42.csv` + `rsna_val_calib_2fold_seed42.csv`)
- **`train`:** `8,339` patients (`38,251` imgs, `340` pos, `4.08%`) — supervised fitting for `ce_b5`, `trust_b5`, `dinov2_rsna` head, `resnet50_rsna`, `gmic_rsna_finetuned`, `nyu_fourview_rsna_finetuned`.
- **`train_calib`:** `1,192` patients (`5,473` imgs, `49` pos, `4.11%`) — early stopping / checkpoint selection (`best_train_calib.pt`) & TRUST dynamic `tau_t`.
- **`val_calib`:** `2,382` patients (`10,982` imgs, `97` pos, `4.07%`) — **100% unseen during training/checkpoint selection** across ALL RSNA-supervised models. Partitioned 50/50 stratified by `(cancer, site_id)` (`seed=42`) into `fold_a` (`1,191` pts / `49` pos) and `fold_b` (`1,191` pts / `48` pos) for **2-fold out-of-fold (OOF)** independent source testing (`97` pos total), while using **full `val_calib` (`97` pos)** as the frozen external operating-threshold calibration set (`tau_98`, `tau_95`) to avoid single-fold `n_pos=49` min-score collapse (`0` misses allowed at `98%`).

#### B. NLBS Denominator Reconciliation (`5,997 -> 5,935 -> 6,430 -> 6,359 -> 5,942 -> 6,184`)
- `5,997` `case_id` folders (`4,332` normal, `1,516` False Positive, `149` abnormal) expand to `6,430` `(case_id, StudyInstanceUID)` rows because `385` folders contain multiple visits (notably `p_3280` with `44` `StudyInstanceUID`s / `180` images across Ages `62–73`).
- `71` `StudyInstanceUID`s appear in exactly `2` folders (`43` normal, `28` False Positive), deduplicating to **`6,359` unique `StudyInstanceUID` examinations** (`151` positive exams because `p_4334` and `p_4461` each have `2` positive visits).
- Union-Find over `(case_id, StudyInstanceUID)` yields **`5,942` independent `split_cluster_id` components** (`149` positive clusters).
- Requiring `{L-CC, L-MLO, R-CC, R-MLO}` leaves **`6,184 / 6,359` (`97.25%`)** 4-view eligible studies (**`151 / 151` cancers**, `5,868` clusters, `149` pos clusters; `935` studies have `>4` images resolved deterministically without reading labels).
- *Residual Limitation:* Per NLBS `README_version_history_2025-11-25.txt`, directories of unverified repeats were renamed (`PatientID` empty in `26,984/26,988` DICOMs, `Exam Date` 100% null), so unlinked cross-folder repeats remain undetectable.

#### C. IMDB India Audit (`imdb_r1` + `imdb_r2` = `12,674` DICOMs, `107.8376 GiB`)
- **Cross-Release Identity (`582` Collisions):** `r1` uses `0000`–`0582` (`583` IDs); `r2` uses `pt_001`–`pt_2636` (`2,636` IDs). Direct string overlap is `0`, numeric-suffix overlap is **`582`**, `StudyInstanceUID` overlap is `0` (`585` vs `2,651`), and all DICOM dates are neutralized to `20260101`. Release-namespacing prevents erroneous merging, **but cannot prove cross-release individual independence** -> `imdb_r1` and `imdb_r2` must be analyzed separately first.
- **FFDM vs Synthetic Mammography (`SM` / `C-View`):** `12-bit` (`4096x3328` / `3328x2560`) = `FFDM` (`1,869` `r1` + `5,996` `r2` = `7,865` imgs); `10-bit` (`2457x1996` / `2457x1890`) = `SM` (`1,708` `r1` + `3,101` `r2` = `4,809` imgs).
- **4-View FFDM Cohort (`datasets/exp3_meta/imdb_ffdm_4view_{inference,outcome}_manifest.csv`):** **`1,665` exams (`6,660` images)**: `304` `r1` + `1,361` `r2`.
- **Exact Outcome-Confidence Partition of the `1,665` 4-View FFDM Cohort:**
  - `pathology_confirmed_malignant`: **`252`** (`60` `r1` + `192` `r2`)
  - `pathology_confirmed_benign`: **`32`** (`1` `r1` + `31` `r2`)
  - `followup_confirmed_nonmalignant`: **`0`** (`0` `r1` + `0` `r2`)
  - `noncancer_without_adequate_followup`: **`1,379`** (`243` `r1` + `1,136` `r2`)
  - `unknown_or_ambiguous` (`BI-RADS 0` / discordant): **`2`** (`0` `r1` + `2` `r2`: `1` pos, `1` neg)
  - *Check:* `252 + 32 + 0 + 1,379 + 2 = 1,665`.
  - **Analysis A (Strict Pathology-Confirmed):** `N = 284` (`61` `r1` [`60` pos / `1` benign], `223` `r2` [`192` pos / `31` benign]).
  - **Analysis B (Original-Label Exploratory, `2` ambiguous excluded):** `N = 1,663` (`304` `r1` [`60` pos / `244` neg], `1,359` `r2` [`192` pos / `1,167` neg]).

#### D. MIAS UK Audit (`CAM.105113`, `322` PGMs / `161` pairs)
- Landscape `8-bit` PGM (`4320x1600`: 48, `4320x2048`: 101, `4320x2600`: 149, `5200x4000`: 24) with chest wall at top edge; rotated 90° upright in `decode_pgm_array()` (`L` CCW, `R` CW) and rescaled to `uint16`.
- `330` truth rows (`7` multi-lesion images including `mdb144rx` with both `B` and `M`); `161` pairs = `52` malignant (`32.3%`), `56` benign-only (`34.8%`), `53` normal (`32.9%`), **`0` CC views**.

---

## 4. Model Configurations, Hyperparameters & Cryptographic Provenance

### 4.1 Complete Ten-Model Provenance & Frozen Threshold Table

| Model ID | Regime / Exp | Backbone / Repo & License | Training Data & Hyperparameters | Input Spec & Aggregation | Checkpoint SHA-256 | Frozen RSNA `tau_98` / `tau_95` |
|---|---|---|---|---|---|---|
| `ce_b5` | Primary + Exp 3 (RSNA-Sup) | Mammo-CLIP EffNet-B5 (`shawn24/Mammo-CLIP` `2f356926`, `CC BY-NC-SA 4.0`) | `UPMC` pretrain + RSNA `train` (`20` ep, `bs=80` [`20:60`], `lr=3e-5`, BCE, final ep20) | `1520x912 uint16`, right-mirrored, Mammo-CLIP norm -> exam `max` | `adf1443455ee1b507485ee4b7f64932d5cf6f59bb6afcf11b9f415c634372d22` | `0.096193` / `0.124415` |
| `trust_b5` | Primary + Exp 3 (RSNA-Sup) | Mammo-CLIP EffNet-B5 (`CC BY-NC-SA 4.0`) | `UPMC` pretrain + RSNA `train` (`20` ep, `bs=80` [`20:60`], `lr=3e-5`, Focal `gamma=2` + Dynamic Dismissal `m=0.10, lambda=0.05`, `tau_t=0.25610`) | `1520x912 uint16`, right-mirrored, Mammo-CLIP norm -> exam `max` | `63388d790ce9cf8a8287adc7724358ced392a69b332ce1a5e46972750faa51cb` | `0.179548` / `0.195463` |
| `dinov2_rsna` | Primary + Exp 3 (RSNA-Sup) | DINOv2 ViT-S/14 (`dinov2_vits14`, `Apache-2.0`) | `LVD-142M` frozen + `LayerNorm->Dropout(0.3)->Linear(384,1)` on RSNA `train` (`AdamW`, `wd=1e-4`, BCE, best ep `2` on `train_calib` `0.7469`) | `1520x912` -> pad square -> `518x518` ImageNet norm -> exam `max` | `b141c2631ca8e54e5d233239823aafaf18fa3f5d8c62c755339888f41f67d5f5` | `0.183436` / `0.206516` |
| `resnet50_rsna` | Primary + Exp 3 (RSNA-Sup) | `torchvision` `ResNet50_Weights.IMAGENET1K_V2` (`BSD-3-Clause`) | ImageNet-1K -> full fine-tune on RSNA `train` (`15` ep, `600` steps/ep, `bs=64`, `AdamW`, `wd=1e-4`, BCE, best ep `2` on `train_calib` `0.7127`) | `1520x912` -> `854x512` ImageNet norm -> exam `max` | `ed620c61e38ce47ceac99d73fb1b02c2dae4946552e4469f5395f19b2a0c9d32` | `0.000993` / `0.006931` |
| `gmic_pretrained` | Primary + Exp 2 (NYU-Pretrained) | `nyukat/GMIC` `sample_model_1.p` (`AGPL-3.0`) | NYU `229,426` exams (`2010–2017`), `0` RSNA training | Native `2944x1920 uint16`, per-img z-score -> `malignant_pred` `max` | `3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991` | `0.028874` / `0.046641` *(Primary only; Cat C in Exp 2)* |
| `nyu_fourview_pretrained` | Primary + Exp 2 (NYU-Pretrained) | `nyukat/breast_cancer_classifier` `sample_image_model.p` (`AGPL-3.0`) | NYU `229,426` exams (`2010–2017`), `0` RSNA training | Native 4-view (`CC 2677x1942`, `MLO 2974x1748`), 10-TTA `seed=0` -> `max(L, R)` | `9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014` | `0.021428` / `0.034991` *(Primary only; Cat C in Exp 2)* |
| `gmic_rsna_finetuned` | Exp 1 + Exp 3 (RSNA-Fine-Tuned) | `nyukat/GMIC` partial fine-tune (`AGPL-3.0`) | RSNA `train` (`10.63M/14.11M` params, `bf16`, `bs=64`, `lr=1e-4/1e-5`, malignant-only BCE+L1 saliency, best ep `1` `train_calib` `0.8733`) | Native `2944x1920 uint16`, per-img z-score -> `malignant_pred` `max` | `11efb7c65f773fc7f4f7a7bf8b0620c276c63536a3e7e7b2bf84eed6024146d5` | `0.130208` / `0.158398` |
| `nyu_fourview_rsna_finetuned` | Exp 1 + Exp 3 (RSNA-Fine-Tuned) | NYU Four-View `head_only` (`AGPL-3.0`) | RSNA `train` (`533,520` head params, FP32, `NLLLoss` malignant `[2,3]`, best ep `5` `train_calib` `0.8056`) | Native 4-view (`CC 2677x1942`, `MLO 2974x1748`), 10-TTA -> `max(L, R)` | `b42a8b7d623210cd88369b402981b4d1050ea72052851c656c0bbd380532b00e` | `0.047732` / `0.069086` |
| `glam_pretrained` | Exp 2 (NYU-Pretrained) | NYU GLAM `model_joint.ckpt` (NYU Non-Commercial / AGPL conflict) | NYU `2010–2017`, `0` RSNA training | Option B1 `1152x896 uint16` (native GLAM crop), `bs=4` | `a647ba9176b3e11d2dec7252054504f33856bcabb2bbe763432d3d20e7c65aec` | *Category C only (no RSNA calib)* |
| `end2end_cbis_ddsm` | Exp 2 (Film-Pretrained) | Shen End2End All-Conv ResNet-50 (`CC BY-NC-SA 4.0`) | CBIS-DDSM digitized film (`S10` whole-image), Keras `.h5` -> TorchScript (`4.51e-10` diff) | `1152x896` -> model spec, `bs=32` | `1c93c4c3ff7f997cc235140340181995d0219162cecd0bf495381b777a2389be` | *Category C only (no RSNA calib)* |

*Exp 2 Excluded Candidates:* `petrini_twoview_efnb3` (`blocked_upstream`: HF/GDrive `.pt` truncated at `88.0 MiB` + `405 MiB` zeros, no ZIP EOCD), `frcnn_cad` (`disqualified`: dead ELTE URL, CUDA 8.0/Caffe), `mammoscreen_r1` (`rejected`: RSNA Split 1), `petrini_vindr_r2` (`rejected`: BI-RADS `>=3` labels), `mammo_clip_zeroshot` (`rejected`: not a supervised cancer classifier).

---

## 5. Complete Empirical Results Across All Datasets & Experiments

### 5.1 RSNA Source Evaluation (`val_calib`, `N = 2,382` patients, `97` cancers — 2-Fold Patient-Disjoint OOF)

| Model | Source Family | RSNA AUROC (95% CI) | RSNA AUPRC (95% CI) | OOF `@98%` Sens (Missed) | OOF `@98%` Dismissal | OOF `@95%` Sens (Missed) | OOF `@95%` Dismissal |
|---|---|---|---|---|---|---|---|
| `ce_b5` | RSNA-Sup | `0.7428` (`0.6930–0.7882`) | `0.1064` (`0.0785–0.1504`) | `98.97%` (`1`) | `11.67%` | `89.69%` (`10`) | `29.09%` |
| `trust_b5` | RSNA-Sup | `0.7442` (`0.6939–0.7896`) | `0.1072` (`0.0794–0.1500`) | `98.97%` (`1`) | `12.80%` | `89.69%` (`10`) | `29.43%` |
| `dinov2_rsna` | RSNA-Sup | `0.6302` (`0.5791–0.6802`) | `0.0684` (`0.0482–0.1110`) | `96.91%` (`3`) | `7.77%` | `95.88%` (`4`) | `14.11%` |
| `resnet50_rsna` | RSNA-Sup | `0.6580` (`0.6046–0.7098`) | `0.0869` (`0.0609–0.1346`) | `97.94%` (`2`) | `7.51%` | `95.88%` (`4`) | `14.02%` |
| `gmic_pretrained` | NYU-Pretrained | `0.8301` (`0.7885–0.8710`) | `0.2509` (`0.1719–0.3346`) | `98.97%` (`1`) | `13.39%` | `91.75%` (`8`) | `38.54%` |
| `nyu_fourview_pretrained` | NYU-Pretrained | `0.7721` (`0.7211–0.8200`) | `0.1924` (`0.1296–0.2813`) | `96.91%` (`3`) | `6.76%` | `96.91%` (`3`) | `14.57%` |
| `gmic_rsna_finetuned` *(Exp 1)* | RSNA-FT | **`0.8570`** (`0.8177–0.8936`) | **`0.3132`** (`0.2193–0.4060`) | `96.91%` (`3`) | **`21.70%`** | `95.88%` (`4`) | **`39.50%`** |
| `nyu_fourview_rsna_finetuned` *(Exp 1)* | RSNA-FT | **`0.8013`** (`0.7549–0.8450`) | **`0.2362`** (`0.1605–0.3322`) | `98.97%` (`1`) | **`9.32%`** | `92.78%` (`7`) | **`32.70%`** |

### 5.2 Primary Six-Model Benchmark + Exp 1 Fine-Tuned Specialists on NLBS (`N = 6,184` Common 4-View, `5,868` Clusters, `151` Cancers)

| Model (Cohort / Regime) | NLBS AUROC (95% CI) | NLBS AUPRC (95% CI) | `@98%` Sens (95% CI) | Missed `@98%` | `@98%` Dismissal | CP99 UB `@98%` | `@95%` Sens (Missed) | `@95%` Dismissal |
|---|---|---|---|---:|---|---|---|---:|
| `ce_b5` *(Primary Stratum A)* | `0.7089` (`0.6668–0.7463`) | `0.0548` (`0.0418–0.0764`) | `90.07%` (`85.00–94.33%`) | **15** | `30.48%` | `1.41%` | `80.13%` (`30`) | `49.66%` |
| `trust_b5` *(Primary Stratum A)* | `0.7040` (`0.6617–0.7423`) | `0.0539` (`0.0409–0.0747`) | `84.77%` (`78.72–90.23%`) | **23** | `39.57%` | `1.50%` | `80.79%` (`29`) | `49.29%` |
| `dinov2_rsna` *(Primary Stratum A)* | `0.6299` (`0.5861–0.6710`) | `0.0425` (`0.0320–0.0628`) | **`98.01%`** (`95.45–100.00%`) | **3** | `9.67%` | `1.67%` | `94.04%` (`9`) | `17.43%` |
| `resnet50_rsna` *(Primary Stratum A)* | `0.6452` (`0.5935–0.6941`) | `0.1039` (`0.0656–0.1597`) | `97.35%` (`94.48–99.38%`) | **4** | `5.35%` | `3.47%` | `84.77%` (`23`) | `19.78%` |
| `gmic_pretrained` *(Primary Stratum B)* | `0.8541` (`0.8187–0.8874`) | `0.3437` (`0.2660–0.4240`) | `97.35%` (`94.44–99.39%`) | **4** | `24.27%` | `0.77%` | `93.38%` (`10`) | `38.47%` |
| `nyu_fourview_pretrained` *(Primary Stratum B)* | `0.7925` (`0.7551–0.8280`) | `0.1490` (`0.1066–0.2118`) | **`100.00%`** (`100.00–100.00%`) | **0** | `1.94%` | `3.76%` | `99.34%` (`1`) | `15.60%` |
| `gmic_rsna_finetuned` *(Exp 1)* | **`0.8624`** (`0.8273–0.8954`) | **`0.3345`** (`0.2563–0.4126`) | `96.03%` (`92.67–98.76%`) | **6** | **`33.15%`** | **`0.71%`** | `92.05%` (`12`) | `40.96%` |
| `nyu_fourview_rsna_finetuned` *(Exp 1)* | `0.7846` (`0.7414–0.8251`) | `0.1599` (`0.1143–0.2227`) | `96.69%` (`93.46–99.32%`) | **5** | `14.68%` | `1.44%` | `91.39%` (`13`) | `35.22%` |

- **Full-Cohort Sensitivity (`N=6,359`, `5,942` clusters, `151` pos):** `ce_b5` `0.7101 / 90.07% / 30.62%`; `trust_b5` `0.7052 / 84.77% / 39.77%`; `dinov2_rsna` `0.6301 / 98.01% / 9.66%`; `resnet50_rsna` `0.6452 / 97.35% / 5.38%`; `gmic_pretrained` `0.8554 / 97.35% / 24.69%`; `gmic_rsna_finetuned` `0.8637 / 96.03% / 33.65%`.
- **Cluster-Collapsed Sensitivity (`N=5,868` clusters, `149` pos):** `ce_b5` `0.7045 / 89.93%`; `trust_b5` `0.6997 / 84.56%`; `dinov2_rsna` `0.6252 / 97.99%`; `resnet50_rsna` `0.6363 / 97.32%`; `gmic_pretrained` `0.8505 / 97.32%`; `nyu_fourview_pretrained` `0.7890 / 100.00%`; `gmic_rsna_finetuned` `0.8578 / 95.97%`; `nyu_fourview_rsna_finetuned` `0.7816 / 96.64%`.
- **Key Paired Contrasts & McNemar Tests on NLBS (`N=6,184`):**
  - `ce_b5 - trust_b5`: `delta_AUROC = +0.0048` `[+0.0005, +0.0090]`, `delta_sens@98 = +5.28 pp` `[+1.90, +9.15]` (McNemar `p = 0.0078`, `8` vs `0` cancers retained), `delta_dis@98 = -9.10 pp`.
  - `ce_b5 - gmic_pretrained`: `delta_AUROC = -0.1452` `[-0.1866, -0.1014]`, `delta_sens@98 = -7.25 pp` (McNemar `p = 0.0074`).
  - `trust_b5 - gmic_pretrained`: `delta_AUROC = -0.1500` `[-0.1916, -0.1065]`, `delta_sens@98 = -12.53 pp` (McNemar `p = 2.10e-5`).
  - `gmic_pretrained - nyu_fourview_pretrained`: `delta_AUROC = +0.0613` `[+0.0264, +0.0971]`, `delta_sens@98 = -2.63 pp` (McNemar `p = 0.125`), `delta_dis@98 = +22.32 pp`.
- **Missed-Cancer Overlap (`151` Positives, `@98%` Primary Benchmark):**
  - `121 / 151` (`80.1%`) detected by all 6 models; **Jaccard (`ce_b5`, `trust_b5`) = `0.652`** (`15` shared out of `23` TRUST misses; `0.903` at `@95%`), whereas **all misses by `dinov2_rsna` (`3`) had `0.000` Jaccard overlap** with every other model, and `gmic_pretrained` (`4`) had `0.000` overlap with `dinov2_rsna`/`resnet50_rsna`!
- **Shift vs Discrimination Diagnostics (`RSNA -> NLBS`):**
  - `ce_b5`: `delta_AUROC = -0.0339`, `pos_median_shift = -0.2365` (`KS = 0.5538, p = 3.14e-17`), `neg_median_shift = -0.0920` (`KS = 0.3568, p = 3.41e-188`).
  - `trust_b5`: `delta_AUROC = -0.0403`, `pos_median_shift = -0.1991` (`KS = 0.5649, p = 6.00e-18`), `neg_median_shift = -0.0970` (`KS = 0.3799, p = 6.29e-214`).
  - `gmic_pretrained`: `delta_AUROC = +0.0240`, `pos_median_shift = +0.1440` (`KS = 0.1970, p = 0.0169`), `neg_median_shift = -0.0175`.
  - `nyu_fourview_pretrained`: `delta_AUROC = +0.0203`, `pos_median_shift = -0.0040` (`KS = 0.0662, p = 0.937`), `neg_median_shift = -0.0018` (`KS = 0.0229, p = 0.344`) — **near-zero score shift**!

### 5.3 Experiment 2 — Original Pretrained Mammography Models (Category C on NLBS, `N = 6,184` & `N = 6,359`)

| Model ID | 4-View AUROC (95% CI) | 4-View AUPRC (95% CI) | Full NLBS AUROC (`N=6,359`) | Cluster-Collapsed AUROC (`N=5,868`) |
|---|---|---|---|---|
| `gmic_pretrained` | **`0.8541`** (`0.8187–0.8874`) | **`0.3437`** (`0.2660–0.4240`) | `0.8554` (`0.8205–0.8883`) | `0.8505` (`0.8141–0.8844`) |
| `glam_pretrained` | **`0.8006`** (`0.7613–0.8371`) | `0.1163` (`0.0891–0.1608`) | `0.8019` (`0.7635–0.8383`) | `0.7953` (`0.7556–0.8325`) |
| `nyu_fourview_pretrained` | `0.7925` (`0.7551–0.8280`) | `0.1490` (`0.1066–0.2118`) | *Excluded (4-view)* | `0.7890` (`0.7524–0.8242`) |
| `end2end_cbis_ddsm` | **`0.5292`** (`0.4796–0.5783`) | `0.0446` (`0.0295–0.0777`) | `0.5286` (`0.4782–0.5777`) | `0.5312` (`0.4826–0.5792`) |

- **Paired Contrasts:** `gmic - glam = +0.0535` `[+0.0257, +0.0801]`; `gmic - nyu = +0.0617` `[+0.0264, +0.0971]`; `nyu - glam = -0.0082` `[-0.0386, +0.0229]`; `gmic - end2end = +0.3249` `[+0.2653, +0.3823]`.

### 5.4 Experiment 3 — Diagnostic Stress Tests: CMMD (China) & LUMINA (Turkey)
*(Biopsy-enriched diagnostic cohorts — `population_screening_dismissal_allowed = False`.)*

| Dataset & Cohort | Model | AUROC (95% CI) | AUPRC (95% CI) | `@98%` Sens (Missed) | `@98%` Below-Thr | `@95%` Sens (Missed) | `@95%` Below-Thr |
|---|---|---|---|---|---|---|---|
| **CMMD 4-View** (`N=826`, `765` pos) | `ce_b5` | `0.6414` (`0.5768–0.7065`) | `0.9583` (`0.9430–0.9719`) | `48.76%` (`392`) | `52.78%` | `35.29%` (`495`) | `66.22%` |
| **CMMD 4-View** | `trust_b5` | `0.6474` (`0.5832–0.7121`) | `0.9588` (`0.9437–0.9724`) | `45.36%` (`418`) | `56.42%` | `38.43%` (`471`) | `63.20%` |
| **CMMD 4-View** | `dinov2_rsna` | `0.6567` (`0.5810–0.7285`) | `0.9558` (`0.9376–0.9711`) | `70.98%` (`222`) | `30.75%` | `59.08%` (`313`) | `42.37%` |
| **CMMD 4-View** | `resnet50_rsna` | `0.4741` (`0.4071–0.5392`) | `0.9303` (`0.9074–0.9511`) | `99.35%` (`5`) | `0.61%` | `95.29%` (`36`) | `4.60%` |
| **CMMD 4-View** | `gmic_rsna_finetuned` | `0.7059` (`0.6520–0.7574`) | `0.9699` (`0.9580–0.9796`) | `96.86%` (`24`) | `3.51%` | `93.07%` (`53`) | `7.14%` |
| **CMMD 4-View** | `nyu_fourview_rsna_finetuned` | **`0.7137`** (`0.6559–0.7699`) | `0.9674` (`0.9505–0.9801`) | `92.94%` (`54`) | `7.51%` | `81.57%` (`141`) | `19.73%` |
| **CMMD Full** (`N=1,775`, `1,310` pos) | `gmic_rsna_finetuned` | **`0.7671`** (`0.7450–0.7894`) | `0.9075` (`0.8929–0.9211`) | `94.81%` (`68`) | `8.96%` | `91.15%` (`116`) | `13.52%` |
| **CMMD Full** | `ce_b5` / `trust_b5` | `0.6631` / `0.6654` | `0.8484` / `0.8494` | `44.58%` (`726`) / `41.53%` (`766`) | `61.63%` / `64.56%` | `32.60%` (`883`) / `35.11%` (`850`) | `72.90%` / `70.76%` |
| **LUMINA Full** (`N=468`, `218` pos) | `ce_b5` | `0.6693` (`0.6186–0.7206`) | `0.6654` (`0.6017–0.7345`) | `95.87%` (`9`) | `2.35%` | `93.12%` (`15`) | `5.13%` |
| **LUMINA Full** | `trust_b5` | `0.6708` (`0.6201–0.7230`) | `0.6688` (`0.6052–0.7366`) | `94.04%` (`13`) | `3.42%` | `93.12%` (`15`) | `4.49%` |
| **LUMINA Full** | `dinov2_rsna` | `0.5730` (`0.5216–0.6241`) | `0.5879` (`0.5184–0.6606`) | `92.20%` (`17`) | `6.62%` | `86.70%` (`29`) | `10.90%` |
| **LUMINA Full** | `resnet50_rsna` | `0.5753` (`0.5209–0.6274`) | `0.5713` (`0.5035–0.6410`) | `89.91%` (`22`) | `10.68%` | `77.52%` (`49`) | `25.85%` |
| **LUMINA Full** | `gmic_rsna_finetuned` | **`0.7426`** (`0.6943–0.7879`) | **`0.7800`** (`0.7283–0.8279`) | `93.58%` (`14`) | `5.34%` | `91.74%` (`18`) | `8.12%` |

*(Full CMMD `N=1,775` also had `dinov2_rsna` `0.6794` [`64.66%` / `463` miss] and `resnet50_rsna` `0.5776` [`98.93%` / `14` miss]; LUMINA excluded `nyu_fourview_rsna_finetuned`.)*

### 5.5 Experiment 3 Expansion — IMDB India (Matched 4-View FFDM, `N = 1,665` / `6,660` imgs, `0` Fallbacks)

#### Analysis A: Strict Pathology-Confirmed (`imdb_r2` `N=223` [`192` pos / `31` benign] & Combined `N=284` [`252` pos / `32` benign])
*(Note: `imdb_r1` Analysis A has `60` pos / `1` benign [`ce`/`trust`/`dinov2`/`nyu` `100.00%` sens, `resnet`/`gmic` `98.33%`]; `1` benign makes specificity/AUROC uninformative.)*

| Subset | Model | AUROC (95% CI) | AUPRC (95% CI) | `@98%` TP/TN/FP/FN | `@98%` Sens (95% CI) | `@98%` Spec | `@95%` Sens (Missed) |
|---|---|---|---|---|---|---|---|
| **`imdb_r2` (`N=223`)** | `ce_b5` | `0.5061` (`0.3935–0.6210`) | `0.8649` (`0.8019–0.9216`) | `175/3/28/17` | `91.15%` (`86.77–94.85%`) | `9.68%` | `83.85%` (`31`) |
| **`imdb_r2` (`N=223`)** | `trust_b5` | `0.5040` (`0.3917–0.6169`) | `0.8634` (`0.8001–0.9210`) | `170/3/28/22` | `88.54%` (`83.68–92.94%`) | `9.68%` | `85.42%` (`28`) |
| **`imdb_r2` (`N=223`)** | `dinov2_rsna` | `0.5204` (`0.4076–0.6290`) | `0.8822` (`0.8291–0.9316`) | `190/0/31/2` | `98.96%` (`97.31–100.00%`) | `0.00%` | `98.96%` (`2`) |
| **`imdb_r2` (`N=223`)** | `resnet50_rsna` | **`0.5632`** (`0.4509–0.6653`) | `0.8817` (`0.8229–0.9366`) | `189/0/31/3` | `98.44%` (`96.37–100.00%`) | `0.00%` | `89.58%` (`20`) |
| **`imdb_r2` (`N=223`)** | `gmic_rsna_finetuned` | `0.5109` (`0.4081–0.6093`) | `0.8824` (`0.8292–0.9297`) | `185/0/31/7` | `96.35%` (`93.47–98.51%`) | `0.00%` | `91.67%` (`16`) |
| **`imdb_r2` (`N=223`)** | `nyu_fourview_rsna_finetuned` | `0.5576` (`0.4500–0.6680`) | `0.8909` (`0.8360–0.9374`) | `191/0/31/1` | **`99.48%`** (`98.40–100.00%`) | `0.00%` | `94.27%` (`11`) |
| **Combined (`N=284`)** | `ce_b5` / `trust_b5` | `0.5510` / `0.5515` | `0.9038` / `0.9034` | `235/3/29/17` / `230/3/29/22` | `93.25%` / `91.27%` | `9.38%` / `9.38%` | `87.70%` (`31`) / `88.89%` (`28`) |
| **Combined (`N=284`)** | `dinov2` / `resnet50` | `0.5213` / `0.5442` | `0.9066` / `0.9017` | `250/0/32/2` / `248/0/32/4` | `99.21%` / `98.41%` | `0.00%` / `0.00%` | `99.21%` (`2`) / `90.08%` (`25`) |
| **Combined (`N=284`)** | `gmic_ft` / `nyu_ft` | `0.5035` / `0.5501` | `0.9033` / `0.9094` | `244/0/32/8` / `251/0/32/1` | `96.83%` / `99.60%` | `0.00%` / `0.00%` | `92.86%` (`18`) / `94.84%` (`13`) |

#### Analysis B: Original-Label Exploratory (`imdb_r1` `N=304`, `imdb_r2` `N=1,359`, Combined `N=1,663` — `97.7%` of negatives lack follow-up)

| Subset | Model | AUROC (95% CI) | AUPRC (95% CI) | `@98%` TP/TN/FP/FN | `@98%` Sens (95% CI) | Apparent Spec / FPR | Apparent Below-Thr | `@95%` Sens (Missed) |
|---|---|---|---|---|---|---|---|---|
| **`imdb_r1` (`N=304`)** | `ce_b5` / `trust_b5` | `0.5212` / `0.5082` | `0.2398` / `0.2323` | `60/0/244/0` | `100.00%` (`0`) | `0.00%` / `100.00%` | `0.00%` | `100.00%` (`0`) |
| **`imdb_r1` (`N=304`)** | `dinov2` / `resnet50` | `0.5055` / `0.4971` | `0.2627` / `0.2449` | `60/0/244/0` / `59/4/240/1` | `100.00%` / `98.33%` | `0.00%` / `1.64%` | `0.00%` / `1.64%` | `100.00%` / `91.67%` (`5`) |
| **`imdb_r1` (`N=304`)** | `gmic_ft` / `nyu_ft` | `0.5169` / `0.5476` | `0.2511` / `0.2410` | `59/3/241/1` / `60/2/242/0` | `98.33%` / `100.00%` | `1.23%` / `0.82%` | `1.32%` / `0.66%` | `96.67%` (`2`) / `96.67%` (`2`) |
| **`imdb_r2` (`N=1,359`)** | `ce_b5` | `0.5379` (`0.4955–0.5783`) | `0.1591` (`0.1328–0.2004`) | `175/168/999/17` | `91.15%` (`86.98–94.90%`) | `14.40%` / `85.60%` | `13.61%` | `83.85%` (`31`) |
| **`imdb_r2` (`N=1,359`)** | `trust_b5` | `0.5365` (`0.4955–0.5780`) | `0.1589` (`0.1329–0.1997`) | `170/207/960/22` | `88.54%` (`84.00–92.79%`) | `17.74%` / `82.26%` | `16.85%` | `85.42%` (`28`) |
| **`imdb_r2` (`N=1,359`)** | `dinov2_rsna` | `0.5103` (`0.4693–0.5529`) | `0.1529` (`0.1252–0.1887`) | `190/5/1162/2` | `98.96%` (`97.33–100.00%`) | `0.43%` / `99.57%` | `0.52%` | `98.96%` (`2`) |
| **`imdb_r2` (`N=1,359`)** | `resnet50_rsna` | `0.5299` (`0.4864–0.5760`) | `0.1630` (`0.1344–0.2067`) | `189/14/1153/3` | `98.44%` (`96.43–100.00%`) | `1.20%` / `98.80%` | `1.25%` | `89.58%` (`20`) |
| **`imdb_r2` (`N=1,359`)** | `gmic_rsna_finetuned` | `0.5170` (`0.4709–0.5622`) | `0.1539` (`0.1299–0.1897`) | `185/57/1110/7` | `96.35%` (`93.41–98.91%`) | `4.88%` / `95.12%` | `4.71%` | `91.67%` (`16`) |
| **`imdb_r2` (`N=1,359`)** | `nyu_fourview_rsna_finetuned` | `0.5115` (`0.4677–0.5571`) | `0.1573` (`0.1312–0.1990`) | `191/10/1157/1` | `99.48%` (`98.30–100.00%`) | `0.86%` / `99.14%` | `0.81%` | `94.27%` (`11`) |
| **Combined (`N=1,663`)** | `ce_b5` / `trust_b5` | `0.5444` / `0.5421` | `0.1779` / `0.1773` | `235/168/1243/17` / `230/207/1204/22` | `93.25%` / `91.27%` | `11.91%` / `14.67%` | `11.12%` / `13.77%` | `87.70%` (`31`) / `88.89%` (`28`) |
| **Combined (`N=1,663`)** | `dinov2` / `resnet50` | `0.5102` / `0.5182` | `0.1660` / `0.1760` | `250/5/1406/2` / `248/18/1393/4` | `99.21%` / `98.41%` | `0.35%` / `1.28%` | `0.42%` / `1.32%` | `99.21%` (`2`) / `90.08%` (`25`) |
| **Combined (`N=1,663`)** | `gmic_ft` / `nyu_ft` | `0.5180` / `0.5154` | `0.1694` / `0.1661` | `244/60/1351/8` / `251/12/1399/1` | `96.83%` / `99.60%` | `4.25%` / `0.85%` | `4.09%` / `0.78%` | `92.86%` (`18`) / `94.84%` (`13`) |

### 5.6 Experiment 3 Expansion — MIAS UK (`N = 161` Legacy Screen-Film Pairs, `52` pos, `nyu_fourview` Hard-Blocked)

| Model | AUROC (95% CI) | AUPRC (95% CI) | `@98%` TP/TN/FP/FN | `@98%` Sens (95% CI) | `@98%` Spec / FPR | `@98%` Below-Thr / FOR | `@95%` Sens (Missed) |
|---|---|---|---|---|---|---|---|
| `ce_b5` | `0.5258` (`0.4360–0.6191`) | `0.3426` (`0.2613–0.4654`) | `49/13/96/3` | `94.23%` (`86.79–100.00%`) | `11.93%` / `88.07%` | `9.94%` / `18.75%` | `86.54%` (`7`) |
| `trust_b5` | `0.5319` (`0.4435–0.6247`) | `0.3419` (`0.2619–0.4646`) | `45/26/83/7` | `86.54%` (`76.19–95.31%`) | `23.85%` / `76.15%` | `20.50%` / `21.21%` | `86.54%` (`7`) |
| `dinov2_rsna` | `0.5567` (`0.4505–0.6529`) | `0.3981` (`0.3005–0.5340`) | `4/101/8/48` | **`7.69%`** (`1.69–15.56%`) | `92.66%` / `7.34%` | `92.55%` / `32.21%` | `5.77%` (`49`) |
| `resnet50_rsna` | `0.4486` (`0.3530–0.5492`) | `0.3064` (`0.2252–0.4144`) | `34/33/76/18` | `65.38%` (`52.50–78.18%`) | `30.28%` / `69.72%` | `31.68%` / `35.29%` | `28.85%` (`37`) |
| `gmic_rsna_finetuned` | `0.5453` (`0.4502–0.6397`) | `0.3965` (`0.2941–0.5244`) | `8/102/7/44` | **`15.38%`** (`6.00–26.23%`) | `93.58%` / `6.42%` | `90.68%` / `30.14%` | `13.46%` (`45`) |

---

## 6. Unified Cross-Experiment Scientific Synthesis (Core Narrative for Discussion)

1. **Finding 1 — High External AUROC Does NOT Guarantee Source-Frozen Sensitivity Preservation (`NLBS`):**
   - `gmic_rsna_finetuned` (`0.8624`) and `gmic_pretrained` (`0.8541`) rank strongly on NLBS, yet **both miss the `98%` frozen target** (`96.03%` / `6` missed and `97.35%` / `4` missed). Meanwhile, `dinov2_rsna` hits `98.01%` (`3` missed) despite weak `0.6299` AUROC only by retaining `90.33%` of exams.
2. **Finding 2 — Common RSNA Supervision Shifts Calibration Without Resolving Heterogeneity (`Exp 1`):**
   - Fine-tuning GMIC on RSNA improves RSNA `val_calib` (`0.8301 -> 0.8570`) and NLBS AUROC (`0.8541 -> 0.8624`) and raises `@98%` dismissal (`24.27% -> 33.15%`), **but costs 2 extra missed cancers (`97.35% -> 96.03%`)**. Fine-tuning NYU Four-View head on RSNA lifts RSNA (`0.7721 -> 0.8013`) and NLBS dismissal (`1.94% -> 14.68%`), **but drops NLBS `@98%` sensitivity from `100.00%` (`0` missed) to `96.69%` (`5` missed)**.
3. **Finding 3 — Pretrained Specialist Generalization Varies Widely (`Exp 2`):**
   - On NLBS without RSNA calibration, NYU-trained global-local/multi-view models (`gmic_pretrained` `0.8541`, `glam_pretrained` `0.8006`, `nyu_fourview_pretrained` `0.7925`) vastly outperform CBIS-DDSM digitized-film CNN (`end2end_cbis_ddsm` `0.5292`, `delta_AUROC = +0.3249`).
4. **Finding 4 — Three Distinct External Failure Mechanisms Across Geography & Acquisition Regimes (`Exp 3`):**
   - **Downward Score Shift (Canadian Screening `NLBS` + Chinese Diagnostic `CMMD`):** `ce_b5` and `trust_b5` output systematically lower scores externally (`NLBS` `pos_median_shift = -0.2365` / `-0.1991`; `CMMD` `48.76%` / `45.36%` sens @98%, missing `392`/`418` of `765` cancers!), creating dangerous false dismissal despite moderate AUROC (`0.7089` NLBS, `0.6414` CMMD).
   - **Clinical-Spectrum Collapse + Upward Score Shift (Indian Tertiary Oncology `IMDB`):** On `1,665` 4-view Hologic FFDM exams dominated by prior-cancer follow-ups (`61.9%` in `r1`), **all 6 models drop to near-chance discrimination (`AUROC 0.5035–0.5515`)** while scoring post-treatment breasts aggressively high (`nyu` `99.60%` sens / `0.85%` apparent spec; `dinov2` `99.21%` / `0.35%`; `ce_b5` `93.25%` / `11.91%`).
   - **Digitized Film Domain Collapse + Inverted Threshold Failure (UK `MIAS`):** All 5 eligible FFDM-trained models collapse (`AUROC 0.4486–0.5567`), with `dinov2_rsna` (`7.69%` sens, `48/52` missed) and `gmic_rsna_finetuned` (`15.38%` sens, `44/52` missed) falling almost entirely below RSNA `tau_98`.
5. **Finding 5 — Architecture Diversity in Missed Cancers:**
   - On NLBS (`151` pos), `ce_b5` and `trust_b5` share most errors (`Jaccard = 0.652` @98%, `0.903` @95%), whereas `dinov2_rsna` (`3` misses) and `gmic_pretrained` (`4` misses) fail on disjoint cases (`Jaccard = 0.000` between them).

---

## 7. Complete Modal Infrastructure, Billing & Verification Ledger

- **Pre-Expansion (Primary 6-Model + Exp 1–3 Phase 2 without IMDB/MIAS):** `$22.51 <= $30.00` planning cap (`$0.00` billed across `dryousufmozumder` `$11.61` Phase 2, `tigerwood693` `$2.42` Phase 2, `tigerwood697` `$4.84` Phase 2, plus `$3.64` Wave 2+3).
- **IMDB + MIAS Expansion (`tigerwood697`, L40S / CPU):**
  - Baseline `tigerwood697` Ephemeral Apps before expansion: `$9.94` -> Final Ephemeral Apps: **`$23.46`** (`+$13.52` expansion spend across 12,674-DICOM download + CPU cache + MIAS 5-model + initial 1-shard FUSE bottleneck + user-authorized 4-shard 6-model IMDB completion), **`$6.54` free Starter credits remaining**, `$5.45` Volumes covered by Free Storage, **`$0.00` billed**, `0` running apps.
- **Automated Verification (`23/23` Exp 3/Quarantine + `92` Full Suite):** All prediction locks verified SHA-256 untouched before outcome join (`tests/test_paper1_exp3_universal.py` `14/14`, `tests/test_paper1_imdb_mias.py` `6/6`, `tests/test_stage_c_quarantine_and_eval.py` `3/3`).

---

## 8. Manuscript Writing Blueprint (Tables, Figures & Limitations Checklist)

### 8.1 Recommended Paper Tables & Figures
- **Table 1:** Dataset & Clinical Spectrum Summary across RSNA, NLBS, CMMD, LUMINA, IMDB (`r1`/`r2`), and MIAS.
- **Table 2:** Ten-Model Specifications, Pretraining Domains, Licenses, Input Resolutions, and Frozen RSNA Thresholds.
- **Table 3:** Primary Screening Benchmark — RSNA 2-Fold OOF vs NLBS (`N=6,184`) across all 8 RSNA-calibrated / direct models (`AUROC`, `AUPRC`, `@98%`/`@95%` Sens, Dismissal, Missed Cancers, `CP99 UB`).
- **Table 4:** Exp 2 Pretrained-Only Category C NLBS Benchmark (`gmic_pretrained`, `glam_pretrained`, `nyu_fourview_pretrained`, `end2end_cbis_ddsm`) + Paired Contrasts.
- **Table 5:** Multi-Country External Stress Tests — CMMD (`N=826`/`1,775`), LUMINA (`N=468`), IMDB (`r1` `N=304`, `r2` `N=1,359`, Analysis A `N=223`/`284`), and MIAS (`N=161`).
- **Figure 1:** Study Design & Strict Label-Quarantine / Pre-Join SHA-256 Lock Pipeline Diagram.
- **Figure 2:** NLBS `@98%` Sensitivity vs Dismissal Tradeoff & Score-Distribution Shift (`RSNA -> NLBS` KS plots for `ce_b5`/`trust_b5` vs `nyu_fourview_pretrained` vs `gmic`).
- **Figure 3:** UpSet / Pairwise Jaccard Missed-Cancer Overlap Plot on NLBS (`151` cancers).
- **Figure 4:** Multi-Country Failure-Mechanism Comparison (Downward Shift `NLBS`/`CMMD` vs Spectrum/Upward Shift `IMDB` vs Film Collapse `MIAS`).

### 8.2 Mandatory Limitations Paragraph (Reviewer-Proof)
1. **Single Population-Screening Cohort:** NLBS is the sole external population-screening cohort; CMMD, LUMINA, IMDB, and MIAS are diagnostic, post-treatment, or legacy-film cohorts and cannot validate population dismissal.
2. **Retrospective Post-Hoc Expansion:** Stage C and primary 6-model NLBS results were inspected before expanding to Exp 1–3; mitigated via label-free inference manifests and pre-join SHA-256 locking.
3. **Patient-Linkage & Follow-Up Constraints:** NLBS anonymized unverified repeats in its 2025-11-25 rebuild; IMDB neutralized DICOM dates (`20260101`), restarted `pt_XXX` IDs across `r1`/`r2` (`582` collisions), and lacked longitudinal follow-up for `1,379 / 1,411` (`97.7%`) non-cancer exams.
4. **Independent Reproduction & Single-Checkpoint Caveats:** `ce_b5` and `trust_b5` are independent reproductions without `Protocol 3` target recalibration; `nyu_fourview_pretrained` uses `sample_image_model.p` rather than the 100-model clinical ensemble; MIAS required 90° upright rotation and 8-to-16 bit scaling.
5. **Key Artifact Index:** Primary 6-Model (`reports/paper1/six_model_final/six_model_summary.{md,json}`), Exp 1 (`reports/paper1/expanded/exp1/`), Exp 2 (`reports/paper1/expanded/exp2/exp2_summary.{md,json}`), Exp 3 (`reports/paper1/expanded/exp3/eval/exp3_summary.{md,json}`), IMDB+MIAS (`reports/paper1/expanded/exp3/imdb_mias/final_external_expansion_summary.md`).
