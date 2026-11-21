# purpose.md — Two-Paper Breast Cancer Screening AI Research Program

**Repository:** `https://github.com/Research-TRR-HI/Breast-Cancer`  
**Program Structure:** Two distinct but related screening-mammography research papers sharing one codebase, RSNA/NLBS datasets, Modal infrastructure, and base CE/TRUST artifacts, while keeping scientific claims, protocols, and manuscripts strictly separated.

---

## Program Overview (60-Second Orientation)

We investigate **high-sensitivity case-level screening mammography triage (selective dismissal)**—NOT autonomous cancer diagnosis and NOT a universal prospective safety guarantee. An examination score `< threshold` is proposed for dismissal from manual review; `>= threshold` is retained for radiologist review. Across hospitals, shifts in scanner vendor, acquisition, patient population, prevalence, and score calibration can degrade fixed source-calibrated thresholds even when ranking (AUROC) looks moderate.

---

## Paper 1 — Multi-Model External Validation Benchmark
- **Context Document:** [`Paper1_External_Validation_Research_Context.md`](../Paper1_External_Validation_Research_Context.md)
- **Research Goal:** Evaluate whether six materially different screening-AI model configurations calibrated to empirical high-sensitivity targets (`98%` primary, `95%` secondary) using RSNA source labels preserve cancer sensitivity and dismissal utility on unseen Canadian screening exams (NLBS) with **zero Canadian cancer labels** used for model training, selection, or threshold choice, and diagnose failure mechanisms (ranking deterioration vs. score/threshold shift vs. cross-model missed-cancer overlap).
- **Fixed Six-Model Cohort:**
  1. `ce_b5` — Existing CE-only Mammo-CLIP EfficientNet-B5 head (Stage B completed).
  2. `trust_b5` — Existing independent TRUST Mammo-CLIP B5 head (Stage B completed).
  3. `gmic_pretrained` — Pretrained GMIC lesion-aware global/local ResNet-18 (Shen et al., 2021; `AGPL-3.0`).
  4. `nyu_fourview_pretrained` — Pretrained NYU image-only four-view CNN (Wu et al., 2019/2020; `AGPL-3.0`).
  5. `dinov2_rsna` — Frozen DINOv2 ViT backbone (`Apache-2.0`) + **newly RSNA-trained** cancer head (NOT a pretrained mammography detector).
  6. `resnet50_rsna` — ImageNet-initialized ResNet-50 **fine-tuned on RSNA** for binary mammography cancer classification.
- **Datasets:** RSNA (source training + operating calibration + corrected independent source test), NLBS (primary external test), optional IMDB India / additional screening cohort only if label verification & budget permit.
- **Nature of Contribution:** Pure **evaluation, comparison, and failure-mechanism benchmarking** — **NO new algorithm**. Competes closely with MEDINFO 2025/2026 Meta-Repository and *Sensors* (June 2026); novelty & Q1 readiness are **NOT confirmed** (`MODIFY / INVESTIGATE`).
- **Compute Constraint:** Maximum **~$30 USD** additional Modal spend (`NVIDIA L40S`); mandatory throughput/cost benchmarks before full runs and reuse of existing volumes/caches where compatible.

---

## Paper 2 — AnchorShift (Zero-Label Transfer & Method Investigation)
- **Context Document:** [`AnchorShift_Research_Context.md`](../AnchorShift_Research_Context.md)
- **One-Line Mission:** Test whether high-sensitivity case-level mammography dismissal calibrated on RSNA transfers to NLBS with zero target cancer labels (`Protocol 1`), test whether unlabeled-target adaptation via UPA (Roschewitz et al., *Nat Comms* 2023) fixes observed threshold drift without hiding missed cancers (`Protocol 2`), and develop a novel AnchorShift method **ONLY if** a genuine unresolved research gap persists beyond UPA and domain-generalization baselines.
- **Status:** Stage B (`ce`/`trust` RSNA training) and Stage C (`Protocol 1` zero-label NLBS evaluation) complete; **UPA adaptation (`Stage D`) and target-label 20/80 reference (`Protocol 3`) remain pending**. **No novel AnchorShift algorithm has been established**; novelty & Q1 readiness are **NOT confirmed** (`MODIFY / INVESTIGATE`).

---

## How Paper 1 & Paper 2 Relate (Anti-Salami-Slicing Discipline)
- **Shared Infrastructure:** Single GitHub repo, RSNA (`54,706` imgs / `11,913` pts) and NLBS (`26,988` imgs) datasets, Modal AI volumes/scripts, baseline Stage B/C CE & TRUST checkpoints/results (preserved immutable), DICOM preprocessing, evaluation utilities, and `research_logs/`/`memory/`.
- **Strict Scientific Separation:**
  - **Paper 1 asks:** *How do six diverse existing AI models behave under source-frozen `95%/98%` dismissal on an unseen hospital, and why do failures occur?*
  - **Paper 2 asks:** *Can an unlabeled-target adaptation or robust transfer method reliably restore safe high-sensitivity dismissal better than UPA without using target cancer outcomes?*
- Never duplicate identical baseline tables as separate novelty across both manuscripts. If downstream results show both tracks lack independently sufficient contributions, **merge into one stronger unified paper** with full disclosure.

---

## Mandatory Future Agent Protocol (Every Session Startup)
1. **Read Shared Memory & Git History First:** Read `memory/index.md`, `memory/purpose.md`, `memory/state.md`, `memory/history.md`, `memory/personality.md`, and recent `git log` commits before editing or running compute.
2. **Classify the Active Track:** Explicitly identify whether the requested task belongs to **Paper 1**, **Paper 2**, or **Shared Infrastructure**.
3. **Load Track Context:** Read [`Paper1_External_Validation_Research_Context.md`](../Paper1_External_Validation_Research_Context.md) for Paper 1 tasks and [`AnchorShift_Research_Context.md`](../AnchorShift_Research_Context.md) for Paper 2 tasks (remembering that older status notes in root docs are superseded by `memory/state.md` and `reports/`/`research_logs/`).
4. **Verify Actual Artifacts:** Verify real progress, schemas, and numbers directly from disk reports, scripts, and test outputs — **never treat planned work as completed**.
5. **Prevent Cross-Paper Contamination:** Never mix Paper 1 multi-model benchmarking findings with Paper 2 algorithm-development claims, and never alter immutable Stage B/C baseline files in place.
6. **Enforce Resource & Quarantine Rules:** Maintain strict NLBS target-label quarantine for source-only protocols, honor the **$30 additional Modal GPU spend cap** (`L40S`, cost-gated pilots before full runs), never commit raw DICOMs/credentials/checkpoints, follow `.agents/rules/git-commit-rule.md` (`1 file = 1 commit`, alternating accounts per task), and update `memory/` + `research_logs/` after meaningful progress.

---

## Preserved Baseline Guardrails
- Keep patient/cluster grouping airtight (`patient_id` on RSNA; `split_cluster_id` / reconciled unit on NLBS across repeat exams, left/right breasts, and CC/MLO views).
- Distinguish clearly between: (1) **Protocol 1** strict source-only frozen-threshold inference, (2) **Protocol 2** unlabeled-target adaptation (UPA/TTA), and (3) **Protocol 3** target-label threshold calibration (TRUST 20/80 oracle reference — never mislabeled as zero-label).
- Do not misrepresent CMMD (biopsy-enriched), VinDr-Mammo (BI-RADS only), EMBED (restricted DUA), or IMDB (opportunistic/diagnostic caveats) as equivalents to unselected population screening cohorts.
