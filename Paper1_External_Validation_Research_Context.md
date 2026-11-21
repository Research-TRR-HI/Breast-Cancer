# Paper 1 — Multi-Model External Validation of Breast Cancer Screening AI

**Working title:** *Do Mammography AI Models Preserve High-Sensitivity Screening Decisions Across Hospitals? A Six-Model, Source-Frozen Threshold Benchmark*  
**Document purpose:** Standalone, reusable research context for ChatGPT, Gemini Agent IDE, other research agents, and the two-person research team.  
**Last researched / prepared:** 21 November 2026 (Asia/Dhaka).  
**Repository:** https://github.com/Research-TRR-HI/Breast-Cancer  
**GitHub verification point:** `main` at `ee94c63bdea7b871768d3fb1d8d1508d7436b6fa` (latest observed commit on 21 November 2026; committed 20 November).  
**Related study:** Paper 2, **AnchorShift**, documented in `AnchorShift_Research_Context.md`.  
**Decision:** **MODIFY / INVESTIGATE** — a worthwhile study question but novelty and Q1-readiness are **not confirmed**.  
**Computing constraint:** Maximum **USD $30 of additional Modal usage**, single NVIDIA L40S where practical; do not assume this amount guarantees all full experiments.  
**Status of this document:** Research design, **not a completed six-model experimental report**. The newly selected models have not been run by the team as of the verified GitHub state.

---

## 0. Read this first: the project in 60 seconds

We are studying **screening mammography triage**: an AI assigns a cancer-suspicion score to a screening examination, and a threshold determines which examinations would remain for radiologist review versus be proposed for dismissal. This is **not autonomous diagnosis** or proof that dismissed patients are cancer-free.

**Exact candidate research question:**

> When six materially different breast-cancer screening AI systems are calibrated to the same *empirical* high-sensitivity target using RSNA labels only, how often do their fixed decision rules preserve cancer sensitivity and dismissal utility on an external Canadian screening population (NLBS), and what explains differences among the models?

**Six fixed models in the current plan:**

1. **CE-only** — our already-trained Mammo-CLIP EfficientNet-B5 model with binary cross-entropy head.
2. **Independent TRUST implementation** — our already-trained B5 model with TRUST's specialized training objective.
3. **GMIC** — the released pretrained global/local, lesion-aware mammography classifier.
4. **NYU Four-View CNN** — the released pretrained, image-only four-view mammography classifier.
5. **DINOv2 (2023)** — pretrained general-purpose vision transformer **plus a new RSNA-trained cancer head** (not a pretrained mammography cancer predictor).
6. **ResNet-50** — ImageNet-pretrained standard CNN **fine-tuned on RSNA** for the cancer task.

**Datasets:** RSNA is the source for training/operating-point calibration. NLBS is the external Canadian population-screening test, with **no Canadian cancer labels used for training, model choice, or threshold selection**. IMDB or another independent dataset is an *optional, not-yet-validated* additional assessment.

**Current evidence:** The team trained CE and TRUST for 20 epochs and evaluated both on NLBS using source-selected thresholds. At their respective RSNA 98%-target thresholds, CE achieved **90.07% NLBS sensitivity** (15/151 cancer examinations dismissed) and TRUST achieved **84.77%** (23/151 dismissed). These are **our independent implementations**, not the original authors' TRUST checkpoint. The RSNA performance estimates in that experiment were obtained on the same `val_calib` subset used for threshold selection, so they are not independent source-test estimates.

**Why a paper might exist:** Prior work already shows mammography domain shift, external model benchmarking, and operating-point drift. The *candidate*, not yet proven, distinction is a **controlled examination-level comparison of source-frozen 95%/98% sensitivity decisions**, with missed-cancer overlap and explanations separating **loss of ranking quality** from **shifted scores/threshold behavior**. We must be willing to find that some models transfer successfully.

**First action:** Fix the evaluation design and NLBS examination counts **before** spending the $30 on additional models. Then benchmark original model preprocessing and throughput, run GMIC/NYU pretrained inference, extract DINOv2 features, and train ResNet-50 only if the measured projected cost fits the cap.

**Two papers, one GitHub repository:** Paper 1 is the **evaluation and explanation** study. Paper 2 (AnchorShift) is a **possible new label-free transfer method**. Shared datasets, code, and logs are acceptable; the hypotheses, main contributions, protocols, and manuscript claims must remain clearly distinct. Neither paper is guaranteed to be publishable independently.

---

## 1. Evidence tags and how future agents should use this file

| Tag | Meaning |
|---|---|
| **[VERIFIED — REPO]** | Found in the named GitHub repository files/commits as of the verification point above; this verifies a recorded result, not independent re-execution. |
| **[VERIFIED — SOURCE]** | Confirmed from the identified academic paper, official model repository, dataset description, or official pricing page checked on 21 November 2026. |
| **[REPORTED]** | A team result in project logs; not independently regenerated by the author of this context. |
| **[PROPOSED]** | Experimental design or action that has **not** been executed. |
| **[UNVERIFIED]** | A license detail, runtime, cohort construction, overlap, or hypothesis that still needs inspection. |

**Every new research session must recheck** `memory/index.md`, `memory/purpose.md`, `memory/state.md`, `memory/history.md`, `memory/personality.md`, the latest commits, `AnchorShift_Research_Context.md`, and this file. The existing root AnchorShift context was last dated 17 November and includes statements that no experiments had yet run; those statements have been superseded by `memory/state.md` and the 20 November Stage B/C reports. **Do not copy outdated status from the older root document.**

Do not invent research citations, model accuracy, GPU costs, checkpoints, or journal quartiles. If a source is inaccessible, mark it [UNVERIFIED].

---

## 2. Team, workflow, and strict limits

- **Team:** Hasin Ishrak and one research collaborator.
- **Objective:** Genuinely informative, reproducible medical-AI research with a possible Q1 journal submission. Q1 is a goal, not a guarantee or numerical acceptance probability.
- **Gemini Agent IDE:** Primary implementer, debugger, code writer, and experiment runner.
- **ChatGPT:** Research design, literature challenge, methodology, code, statistical and interpretation review.
- **GitHub:** The **single source of project memory** and versioned collaboration. Gemini and ChatGPT do **not** synchronize automatically; changes need an actual commit and report.
- **Modal:** GPU training/inference, checkpoints, original DICOMs, processed images, and large cached features. Inspect the actual active profile, account credits and billing before each run.
- **Local computer:** GTX 1050 Ti (4 GB), i5-9400F, 16 GB RAM. Use local CPU for scripts/unit tests, not full high-resolution training.
- **Hard spending cap:** $30 **new total spending**, not $30 per model. The available balance might be less; confirm it on the active Modal account.
- **No pushing** raw mammograms, labels identifying individuals, credentials, external model weights with restrictive licenses, or huge cached features to GitHub.

---

## 3. The clinical problem and key terminology

A single screening **examination** usually has several image views. The study's main unit is the **examination/patient cluster**, not an individual image or crop.

| Concept | Plain-English definition | Why it matters |
|---|---|---|
| Cancer-suspicion score | A numeric model output; may **not** be a calibrated cancer probability. | Different models have different score scales. |
| Operating threshold `tau` | Exams with score below `tau` are proposed for dismissal; the others are retained. | A threshold calibrated on RSNA may not transport to NLBS. |
| High-sensitivity triage | Seek to retain approximately 95% or 98% of cancer-positive exams, while dismissing some other exams. | Missing cancers matters more than ordinary accuracy. |
| Cancer sensitivity / recall | Cancer-positive exams retained / all cancer-positive exams. | Primary safety-relevant outcome. |
| Dismissal rate | All exams proposed for dismissal / all exams. | Approximate radiologist workload reduction, **not** a real clinical workflow measurement. |
| Missed cancer | Cancer-positive exam falling below the threshold and proposed for dismissal. | Report the numerator and denominator. |
| Cancer prevalence among dismissed | Missed cancers / dismissed exams. | Must not be confused with sensitivity. |
| AUROC | How well the score orders positives above negatives across thresholds. | Can be stable even when a *fixed* threshold behaves differently. |
| AUPRC | Precision/recall ranking summary, strongly affected by cancer prevalence. | Use with cohort prevalence and uncertainty. |
| Score shift | Prediction distributions change across datasets. | Can move an operating point without a new model. |
| Discrimination failure | Cancer cases become harder to rank above noncancer cases. | A monotonic threshold rescaling cannot repair lost ranking. |
| Target labels | Outcome-confirmed cancer/noncancer labels from NLBS or another external site. | Prohibited for model development/threshold selection in primary protocol. |
| Patient-level independence | Repeats, views and exams from the same person stay grouped in one split. | Prevents leakage and misleading confidence intervals. |

**Operational rule:** For the primary source-only protocol, adapt neither the model nor the decision threshold using NLBS images or cancer labels. NLBS images are used solely for prediction. NLBS outcomes are used **only for retrospective evaluation** after predictions are locked.

---

## 4. Our exact scientific aim, hypotheses, and falsification rules

### 4.1 Primary aim

**[PROPOSED]** Quantify **transportability of source-selected high-sensitivity examination-level dismissal thresholds** across six prespecified, scientifically different screening-AI configurations from RSNA to NLBS.

**Primary question:** At each model's **RSNA-only 98% empirical calibration target**, what are the cancer sensitivity, proportion of exams dismissed, and cancer-positive exams dismissed on NLBS?

**Secondary questions:**

1. Do model families differ in the magnitude of **source-to-target operating-point change**, beyond bootstrap uncertainty?
2. Does apparent failure mainly reflect **ranking deterioration**, **score-distribution shift**, or both? Such explanations are descriptive, not automatically causal.
3. Do the six models incorrectly dismiss the **same cancer-positive examinations**, or different ones?
4. Do specialized mammography models transfer more consistently than general-purpose visual representations **after accounting for different pretraining and supervision histories**?
5. Does a high AUROC predict reliable sensitivity/dismissal at a source-fixed threshold? (Do **not** claim this question itself is novel.)

### 4.2 Explicit testable hypotheses — provisional until protocol freeze

- **H1:** Source-frozen 98% operating points do **not** preserve equivalent cancer sensitivity across the six model configurations on NLBS; model-dependent differences are nontrivial. No direction of effect is assumed.
- **H2:** Source-to-NLBS sensitivity changes cannot be completely explained by external AUROC changes; empirical score-scale/threshold instability provides additional useful information.
- **H3:** Cross-model overlap of wrongly dismissed cancer examinations is measurable and may reveal shared or model-specific failure patterns.

**Null/counterexamples:** Some or all models may preserve sensitivity, deterioration may be statistically indistinguishable, differences may disappear on the matched four-view cohort, or the strongest models may recover acceptable operating points. The study must publish these outcomes, not manufacture a failure claim.

### 4.3 What we must not claim

- “We are the first to prove medical AI fails on unseen hospitals.” **False as a novelty claim.**
- “Six-model comparison itself is new.” Already contradicted by prior multi-model mammography benchmarks.
- “DINOv2 is a pretrained breast-cancer detector.” It is **not**; it needs an RSNA cancer-prediction head.
- “The original TRUST model failed on NLBS.” We tested **our independent implementation**, without the original paper's target-label recalibration.
- “The models are clinically safe/unsafe everywhere.” One retrospective external dataset cannot justify universal deployment claims.
- “98% target means certified 98% in every future hospital.” An empirical calibration target is not a population-wide guarantee.
- “Target data were never seen when choosing the new project.” The team already inspected NLBS results before selecting the additional models.

---

## 5. One GitHub repository, two independent paper tracks

**Decision by the team:** Keep **both papers** in https://github.com/Research-TRR-HI/Breast-Cancer.

### Paper 1 — this document

**Type:** Independent external-validation, reproducibility and failure-mechanism benchmark.  
**Contribution sought:** Evidence on **which existing model configurations transport high-sensitivity decisions**, under what observed conditions, and which failure mechanisms appear.  
**Outputs:** Six-model benchmark, source/target threshold behavior, comparisons with uncertainty, case-level missed-cancer overlap, detailed reproducibility and validation reports.  
**New algorithm required?** **No.**

### Paper 2 — AnchorShift

**Type:** Potential new **unlabeled-target adaptation** or robust decision method.  
**Contribution sought:** A reproducible, genuinely differentiated method that improves performance relative to existing methods, especially UPA, without target cancer outcomes for adaptation.  
**Outputs:** New method (only if justified), baseline comparisons, ablations, independent validation, demonstrated limits.  
**Current novelty status:** **Not confirmed.**

### Anti-salami-slicing rule

Paper 1 must stand as a **valuable evaluation study even if no AnchorShift algorithm is developed**. Paper 2 must make a methodological contribution beyond re-describing Paper 1's failure tables. If later experiments show the papers have indistinguishable contributions, **merge into one stronger paper** and disclose all reused data, methods, figures and cohort selections.

### Recommended minimal repo layout — preserve current working paths

```text
Breast-Cancer/
├── AnchorShift_Research_Context.md                # Existing context: Paper 2; do not overwrite
├── Paper1_External_Validation_Research_Context.md # THIS document; add at repo root
├── memory/
│   ├── index.md
│   ├── purpose.md
│   ├── state.md
│   ├── history.md
│   └── personality.md
├── datasets/                                       # Shared metadata/manifests, no raw images
│   ├── rsna/
│   └── nlbs/
├── src/                                            # Shared tested components
├── scripts/                                        # Existing training / inference scripts
├── tests/                                          # Shared + study-specific validation tests
├── studies/                                        # ADD, without moving working code
│   ├── paper1/
│   │   ├── README.md
│   │   ├── protocol_v1.md
│   │   ├── model_manifest.yaml
│   │   ├── eval_config.yaml
│   │   └── analysis_plan.md
│   └── anchorshift/
│       └── README.md
├── reports/
│   ├── stage_b/                                    # Preserve historic CE/TRUST output
│   ├── stage_c/                                    # Preserve locked NLBS CE/TRUST results
│   ├── paper1/                                     # ADD six-model analysis and cost audits
│   └── anchorshift/                                # ADD when Paper 2 experiments occur
└── research_logs/                                  # Date-stamped logs with paper-track tags
```

**Do not reorganize existing working code merely for neatness.** Add paper-specific files gradually. Keep the original Stage B/C reports immutable; write revised analyses under `reports/paper1/` with explicit version references. Record shared input hashes. The five core `memory/` files should continue to act as a **workspace-level index**; add links to this context and Paper 1 state when actually committed, preserving old history.

**Repo actions NOT yet performed by this file:** No GitHub branch, folder or commit has been created; this document is ready to add to the same repository.

---

## 6. Verified repository state, 20–21 November 2026

**[VERIFIED — REPO]** Latest observed `main` commit: `ee94c63bdea7b871768d3fb1d8d1508d7436b6fa` (20 November 2026). Memory files are present: `index.md`, `purpose.md`, `state.md`, `history.md`, `personality.md`.

### 6.1 RSNA source data and assets

- `datasets/rsna/manifest.csv`: **54,706 image rows**, **11,913 unique patients/examinations**.
- Cancer-positive RSNA patients/exams: **486**; cancer-negative: **11,427**.
- Prior split `datasets/rsna/splits_seed42.csv`: training **38,251 images**, training-internal calibration **5,473 images**, `val_calib` **10,982 images**.
- Prior split at examination level: training **8,339**, `train_calib` **1,192**, `val_calib` **2,382**. `val_calib` contains **97 cancer-positive examinations**.
- Patient-disjoint partitions; RSNA has two `site_id` sources and 10 `machine_id` values in the existing audit.
- Raw RSNA DICOMs stored in Modal `rsna-mammography-vol`; approximately **65.9 GiB** processed Stage-1 RSNA image cache already exists in the archived team workflow.
- Mammo-CLIP EfficientNet-B5 pretrained checkpoint is downloaded and has been verified against **852/852** expected keys and pinned hash.
- Augmentation-aware Mammo-CLIP feature cache covers approximately **1,000,000 image occurrences** for the existing head-training protocol. **This feature cache cannot be reused as DINOv2 or ResNet feature tensors.**

### 6.2 NLBS external data and assets

- `datasets/nlbs/manifest.csv`: **26,988 DICOMs**.
- `datasets/nlbs/exam_manifest.csv`: **6,430** `(case_id, StudyInstanceUID)` rows; **6,359** unique `StudyInstanceUID` examination records in the team's Stage C evaluation, with **5,942** connected `split_cluster_id` components.
- Stage C: **151 cancer-positive examination records** corresponding to **149 cancer-positive patient-like clusters**; **1,488** false-positive screening recalls and **4,720** normal examinations.
- Prediction and threshold files were hash-locked **before** an outcome-label join on 20 November 2026. Source-only inference completed for **26,988/26,988 images**, zero reported decode failures.
- Original dataset descriptor instead describes **5,997 examinations**, **5,935 unique examinations/patients as reported by authors**, and **149 cancer-positive cases**; this needs rigorous **denominator reconciliation**, not a casual assumption that the two representations are interchangeable.
- The public data strip some patient/date identifiers; 5,942 connected components are **patient-like grouping proxies**, not guaranteed ground-truth patient IDs.

### 6.3 Existing CE/TRUST training and evaluation — REPORTED historical results

**[REPORTED]** Both models trained for **20 epochs** on the RSNA workflow. Both use the same frozen Mammo-CLIP B5 encoder and train small classification heads. Training-objective differences do **not** establish architecture diversity.

| Model | RSNA threshold target | RSNA `val_calib` sensitivity* | RSNA `val_calib` dismissal* | NLBS sensitivity | NLBS dismissal | NLBS cancers dismissed |
|---|---:|---:|---:|---:|---:|---:|
| CE-only | 98% | 98.97% | 14.48% | **90.07%** | 30.49% | **15 / 151** |
| TRUST independent | 98% | 98.97% | 17.17% | **84.77%** | 39.77% | **23 / 151** |
| CE-only | 95% | 95.88% | 22.50% | **80.13%** | 49.90% | **30 / 151** |
| TRUST independent | 95% | 95.88% | 20.70% | **80.79%** | 49.43% | **29 / 151** |

\* **NOT independent held-out source-test results**: the same `val_calib` labels were used to choose and describe each operating point.

**Frozen historical thresholds:** CE `tau_98 = 0.09619140625`, `tau_95 = 0.12445068359375`; TRUST `tau_98 = 0.1795654296875`, `tau_95 = 0.1954345703125`.

**[REPORTED]** NLBS AUROC: CE **0.7101** (cluster-bootstrap 95% CI 0.6694–0.7469), TRUST **0.7052** (0.6644–0.7429). NLBS paired TRUST-minus-CE sensitivity difference at source 98% target: **−5.30 percentage points** (cluster-bootstrap 95% CI −9.15 to −1.92), though replication uncertainty and identical backbone limit interpretation.

**Actual files:**

- `reports/stage_b/rsna_ce_eval_results.json`
- `reports/stage_b/rsna_trust_eval_results.json`
- `reports/stage_c/nlbs_zero_label_results.md`
- `reports/stage_c/nlbs_zero_label_results.json`
- `reports/stage_c/locked_nlbs_predictions.csv` (hash documented)
- `reports/trust_paper_implementation_traceability.md`
- `reports/dataset_audit.md`
- `research_logs/2026-11-20_stage_c_nlbs_zero_label_evaluation.md`

**Historical interpretation:** Our particular frozen-source decisions **did not preserve 95%/98% sensitivity on NLBS**. This does **not** show that the original authors' exact TRUST checkpoint would fail, that every mammography model fails, or that TRUST's target-label-recalibrated 20% NLBS reference is a zero-label test.

---

## 7. Direct competing literature and novelty audit

**Research window:** Selected primary sources reviewed through **21 November 2026**. This is a **targeted literature audit**, not a formal systematic-review search with completed PRISMA screening. Expand it before claiming novelty or submitting to a journal.

| Work | Verified contribution / protocol | Why it challenges Paper 1 | What, if anything, remains different |
|---|---|---|---|
| **Meta-Repository of Screening Mammography Classifiers** (2026 online proceedings; MEDINFO 2025 volume) | **5 models on 7 international mammography datasets**, standardized Docker-style evaluation, AUC/AUPRC with uncertainty. | Multi-model international benchmarking is **already published**. | A *specific source-only, 98%/95% exam-dismissal operating-point* evaluation with documented target-label exclusion and case-error overlap may be different. |
| **Cross-Sensor and Cross-Population Generalization ...** (*Sensors*, June 2026) | **ResNet-50, DINOv2-B/14, Rad-DINO, Mammo-CLIP B5, Mammo-FM** compared using CBIS-DDSM source; tested CMMD/DMID/MIAS, including cross-population performance and calibration analysis. | Even this exact architectural family pairing is **not new**; cross-dataset changes are known. | The prior study primarily uses different, often diagnostic/film datasets and **sensitivity at 95% specificity**, not our source-frozen **95%/98% sensitivity** dismissal rule in an actual screening cohort. |
| **TRUST** (2026 preprint, arXiv:2609.00300) | High-sensitivity dismissal with dynamic training loss and 20% labeled NLBS cases for external threshold selection. | Already addresses dismissal and threshold recalibration. | Our primary question forbids Canadian outcome labels for threshold selection; **our** TRUST is independently implemented and has deviations. |
| **UPA** (Roschewitz et al., *Nature Communications*, 2023) | Adjusts prediction distributions with **unlabeled target images** to correct acquisition-shift performance drift. | Label-free correction and operating-point drift are established. | Paper 1 focuses on *unadapted source-frozen model-family comparison*; UPA belongs mainly to Paper 2 or a separately labeled ancillary condition. |
| **Garrucho et al.** (*Artificial Intelligence in Medicine*, 2022) | Large-scale cross-domain mammography **mass detection** study. | Shift and robustness are established. | Lesion/mass detection differs from whole-exam cancer dismissal at frozen high-recall thresholds. |
| **MammoDG** (IJCV, 2026) and **MC-SDS** (MICCAI, 2025) | Mammography-specific domain-generalization approaches. | Domain-generalization research is not new. | Paper 1 does not need to introduce a new DG method, but should not claim other methods are absent. |
| **GMIC** (Shen et al., *Medical Image Analysis*, 2021) | Released high-resolution lesion-aware mammography model and checkpoints. | Strong specialist baseline exists; ignoring it would weaken the comparison. | Include as directly pretrained external model. |
| **NYU four-view** (Wu et al., IEEE TMI, online 2019 / issue 2020) | Released four-view mammogram cancer classifier with breast-level output. | Multi-view modeling is established. | Include as direct-pretrained model and state its original age/limitations. |
| **DINOv2** (Oquab et al., 2023) | General-purpose self-supervised visual representations with public backbones. | Foundation-feature transfer is already established. | Test only a new RSNA cancer head; do not call DINOv2 an independently pretrained cancer detector. |
| **Bernstein et al.** (PLOS Digital Health, 2026) | Statistical framework for screening automation risks/benefits. | Missed-cancer and automation trade-offs already have a formal literature. | Apply appropriate clinically relevant estimates and uncertainty. |

**Closest threats to novelty:** (1) the **2026 Meta-Repository** and (2) the **June 2026 Sensors five-backbone comparison**. A manuscript whose main finding is merely “six models show lower AUROC across datasets” would likely add little. The specific **source-fixed high-sensitivity triage** test and its mechanistic analysis are the only *candidate* contribution, not a proven new research gap.

### 7.1 Novelty claim audit

| Proposed claim | Verdict today |
|---|---|
| Multiple breast screening AI models behave differently across datasets. | **ESTABLISHED.** |
| DINOv2 vs ResNet-50 vs Mammo-CLIP external benchmark is new. | **ESTABLISHED/ALREADY STUDIED** in 2026 Sensors paper. |
| Source-selected thresholds may lose sensitivity externally. | **ESTABLISHED IN GENERAL**; precise six-model clinical screening protocol needs targeted prior-work search. |
| AUROC can hide fixed-threshold sensitivity deterioration. | **ESTABLISHED PRINCIPLE**, not a standalone new insight. |
| Which specific six models retain high-recall *whole-exam dismissal* at frozen RSNA thresholds on NLBS? | **POTENTIALLY UNDEREXPLORED**, contingent on search and valid experiments. |
| Shared vs unique cancer exams dismissed by several model families under the same rule. | **POTENTIALLY INFORMATIVE**, not necessarily unprecedented. |
| A new algorithm was developed. | **FALSE** for Paper 1. |
| The study is already Q1-ready. | **FALSE**. |

### 7.2 Mandatory additional literature search before submission

Search PubMed, Google Scholar, Semantic Scholar, Crossref, IEEE Xplore, MICCAI/MIDL, arXiv, Radiology, *Medical Image Analysis*, and *Nature Communications* using combinations of:

`mammography AND (external validation OR domain shift OR transferability) AND (triage OR rule-out OR dismissal)`;
`fixed threshold`, `source-only calibration`, `cancer recall`, `95% sensitivity`, `98% sensitivity`, `missed cancers`, `workload reduction`, `multi-model benchmark`, `cross-site comparison`, `DINOv2 mammography`, `GMIC external validation`, `NYU multi-view screening`.

Record the actual date, databases, query strings, filters, screened candidate count and full-text reasons for inclusion/exclusion. Make a directly competing paper matrix with **threshold selection domain, original task, outcome label definition, screening vs diagnostic status, model families, case aggregation, and patient-independent evaluation**. If a paper already answers our exact question with stronger evidence, **stop or merge**; do not invent novelty.

---

## 8. The fixed six-model study cohort

**[PROPOSED]** These are the six models the team specifically selected. No additions should be made after viewing NLBS performance simply because a new candidate appears more favorable.

| ID | Model and paper | Pretraining and architecture | Public code | Weights today | RSNA training needed? | Clinical output / comparability | Status |
|---|---|---|---|---|---|---|---|
| `ce_b5` | **CE-only**, project 2026 | Frozen Mammo-CLIP EfficientNet-B5 + BCE head | Our repo | Our 20-epoch checkpoint | **Already complete** | Per-image malignancy score → max exam | **EXISTS** |
| `trust_b5` | **TRUST independent**, TRUST 2026 | Same frozen B5 + focal/dismissal loss head | Our repo | Our independent 20-epoch checkpoint | **Already complete** | Per-image malignancy score → max exam | **EXISTS** |
| `gmic_pretrained` | **GMIC**, Shen et al. 2021 | Lesion-aware global/local high-res ResNet-18 | Authors' GitHub | **Five released cancer-prediction checkpoints** | **No** (direct pretrained) | Use `malignant_pred`, not `benign_pred`; max over images | **INTEGRATION NEEDED** |
| `nyu_fourview_pretrained` | **NYU four-view CNN**, Wu et al. 2019/2020 | Four-view ResNet-style network | Authors' GitHub | **Released pretrained cancer classifiers** | **No** (direct pretrained) | Max of left/right breast malignancy score | **INTEGRATION NEEDED** |
| `dinov2_rsna` | **DINOv2**, Oquab et al. 2023 | General natural-image self-supervised ViT encoder | FAIR GitHub | **Encoder weights**, **not mammography cancer head** | **Yes — train RSNA cancer head** | Per-image learned malignancy score → max exam | **PROPOSED** |
| `resnet50_rsna` | **ResNet-50**, He et al. 2016 | ImageNet-pretrained conventional CNN | torchvision/PyTorch | **ImageNet weights**, **not mammography cancer head** | **Yes — fine-tune on RSNA** | Per-image malignancy score → max exam | **PROPOSED** |

### 8.1 GMIC — exact integration constraints

- Official code: https://github.com/nyukat/GMIC ; paper DOI: https://doi.org/10.1016/j.media.2020.101908.
- Repo states **5 GMIC-ResNet-18 pretrained checkpoints**; expects high-resolution **cropped 16-bit PNGs (2944 × 1920)**; `predictions.csv` includes `benign_pred` and `malignant_pred`.
- **Primary option:** Predeclare a single checkpoint by its original release identifier / predetermined selection rule **without looking at NLBS outcomes**. An author-recommended top-5 ensemble is an **optional, separately reported, up-to-5× inference-cost** analysis, not a free substitute.
- Reproduce authors' sample-exam outputs on a pinned environment before running our data. Do not simply feed our old **1520 × 912** Mammo-CLIP tensors to GMIC and pretend it is the released pipeline.
- Implement a deterministic image adapter and metadata audit; log cases whose preprocessing fails. Verify published code and weights license independently for publication and distribution; GitHub repo uses **AGPL-3.0**.
- If output score is for a different clinical endpoint or prevalence composition, report this difference instead of representing the model as RSNA-trained.

### 8.2 NYU four-view CNN — exact integration constraints

- Official code: https://github.com/nyukat/breast_cancer_classifier ; DOI https://doi.org/10.1109/TMI.2019.2945514.
- Select authors' **image-only four-view model**, not image+heatmap, so no separate patch/heatmap model is silently added.
- Four required views: **L-CC, R-CC, L-MLO, R-MLO**. README describes CC resolution **2677 × 1942** and MLO **2974 × 1748**, with 16-bit PNGs, separate image standardization and orientation conventions.
- Output includes `left_malignant` and `right_malignant`; set examination score to `max(left_malignant, right_malignant)`.
- Implement a **view-selection policy** for exams with more than one image per view. It must be prespecified, never use cancer labels, and document missing/ineligible exams.
- Repo disclaimer says the 2019 model is **not clinically used at NYU** and is behind newer models. Treat it as a published reproducible historical comparator, not today's state-of-the-art.
- Verify sample outputs and code compatibility. GitHub repo uses **AGPL-3.0**.

### 8.3 DINOv2 — exact integration constraints

- Official code: https://github.com/facebookresearch/dinov2 ; paper https://arxiv.org/abs/2304.07193 ; standard DINOv2 code/weights are **Apache-2.0** per official README (not to be confused with newer XRay-DINO/Cell-DINO licenses).
- DINOv2 is **not a cancer detection network**. Use its visual encoder with an **RSNA-supervised binary head**, and label it `DINOv2 + RSNA head` in every figure/table.
- **Budget-first primary proposal:** frozen **ViT-S/14** with a lightweight supervised head and precomputed embeddings. **ViT-B/14** is a stronger optional sensitivity analysis if measured costs permit; distinguish it from the DINOv2-B/14 used in the June 2026 Sensors paper.
- Choose one documented image size, aspect-ratio-preserving resize, padding policy, 3-channel replication, normalization and feature pooling recipe *before* NLBS inference. Test whether resolution destroys subtle mammography signals; low-resolution performance is not proof DINOv2 cannot work on mammograms.
- Train head using RSNA labels from patient-disjoint training cases. No NLBS cancer labels and no NLBS unlabeled adaptation in Paper 1 primary protocol.
- Prefer precomputing **one embedding per unique RSNA image** when the encoder is frozen; do not repeat mammogram encoding for each head-training epoch.
- If the frozen probe performs poorly in-domain, **report that fact and the training protocol**. Do not secretly fine-tune for an unknown number of experiments until NLBS looks better.

### 8.4 ResNet-50 — exact integration constraints

- Architecture paper: He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016); official implementations through PyTorch/torchvision.
- ImageNet pretrained weights are **not** breast-cancer weights. Fine-tune a binary mammography classification model on the RSNA training set, with adequate positive sampling, class-imbalance handling and source-only model selection.
- Declare input resolution, learning-rate schedule, augmentation, optimizer, epochs, effective batch size, resolution limitations and early stopping **before NLBS analysis**.
- For budget, start with a **512–768 pixel** feasibility pilot, with aspect-ratio preservation and potential higher-resolution follow-up. Low resolution may damage small-lesion detection; a cheap but poorly trained ResNet is **not a strong conventional baseline**.
- Precompute source evaluation and NLBS scores once; do not re-encode entire datasets for each threshold.
- If full fine-tuning cannot be done properly within the budget, mark the model **unfinished**, do not pretend a 1–2 epoch pilot is comparable to the pretrained specialist models.

### 8.5 CE and TRUST — existing model caveats

- Our models are **not two independent architecture families**; their shared frozen B5 encoder makes them a training-objective pair.
- TRUST paper's source/data split and external 20%-label threshold-selection protocol differ from our independent training and RSNA-only threshold evaluation. Describe reproduction deviations with `reports/trust_paper_implementation_traceability.md`.
- Preserve original 20-epoch checkpoints and locked Stage C results. If revising source calibration splits or retraining on new partitions, record new experiment IDs/checkpoint hashes and do not overwrite previous predictions.

---

## 9. Two model-comparison strata — avoid false equivalence

| Stratum | Models | What the comparison can establish | What it cannot isolate |
|---|---|---|---|
| **A: RSNA-supervised** | CE, TRUST, DINOv2+RSNA head, ResNet-50+RSNA fine-tune | Performance under a common source supervised task and RSNA-only threshold selection. | Differences also reflect initial pretraining (Mammo-CLIP, DINOv2, ImageNet), training capacity and preprocessing. |
| **B: Direct authors' pretrained cancer predictors** | GMIC, NYU four-view | Zero-new-training performance on the same RSNA/NLBS cases with RSNA-only threshold selection. | These models had different original training institutions, label definitions, architectures and sample sizes. |

It is legitimate to show **all six** in one descriptive results table. It is **not legitimate** to attribute every difference solely to architecture or claim “identical training conditions.” Explicitly show a **training-regime** column in every main result.

**Model provenance audit before inference:** Record original model training institutions, public training datasets, checkpoint source/revision/hash, license and any possible overlap with RSNA or NLBS. An unverified overlap must be described as **unknown**, not absent. Investigate Mammo-CLIP pretraining provenance as well; the newly selected GMIC/NYU checkpoints must not have knowingly trained on any NLBS examination.

---

## 10. Datasets, access, and label rules

### 10.1 RSNA — source

- **[VERIFIED — REPO]** 54,706 images, 11,913 patient/examination groups, 486 cancer-positive.
- Two originating screening settings (Emory Healthcare, USA, and BreastScreen Victoria, Australia); inspect `site_id`, manufacturer IDs, age, breast density missingness and original image metadata before subgroup work.
- Cancer labels are breast-level outcomes in the public competition data. **For a positive breast, views may share a breast label even if the tumor is not visible in each image.** This is weak per-image supervision, not pixel-level verified cancer location.
- Publication/usage: official RSNA dataset and Kaggle competition rules allow **noncommercial research**, with **no redistribution** of DICOMs. Recheck current terms before any data/model publication.
- Sources: https://doi.org/10.1148/ryai.250375 ; https://www.kaggle.com/competitions/rsna-breast-cancer-detection/rules .

### 10.2 NLBS — external screening dataset

- **[VERIFIED — SOURCE]** Original paper: 5,997 exams, 26,988 full-field digital mammograms, 5,935 unique cases/exams as reported in description, with 149 cancer-positive cases.
- The Canadian population screening program used predominantly GE Senographe Essential machines acquired around 2008–2010. Images are open via FRDR under **CC BY 4.0** (attribution required).
- Cancer labels come from the screening-program follow-up/diagnostic workup; negative labels involve diagnostic clearance or lack of interval cancer as defined by authors. **Not all negatives have biopsy proof; do not describe them that way.**
- **[VERIFIED — REPO]** Our manifest-based 6,359 unique StudyInstanceUID/151 positive exam view is **not yet reconciled** with the published study's case unit; 5,942 connected clusters approximate patient grouping.
- The examined NLBS cohort has roughly **2.4% positive examinations** in our current unit. Do **not** claim its prevalence and case mix represent an unselected real-world screening service; original dataset selection and false-positive enrichment must be analyzed.
- Source: https://doi.org/10.1038/s41597-025-05866-0 .

### 10.3 Additional cohorts — NOT part of the mandatory $30 experiment

| Dataset | Potential role | Essential caveat |
|---|---|---|
| **IMDB, India (2026)** | Additional population/acquisition stress test if label definitions support it. | Includes diagnostic/opportunistic-screening or screening-naive contexts; mixed label verification. Not equivalent to an ordinary Canadian population-screening cohort. |
| **EMBED** | Potential larger screening data and metadata. | Controlled access and model/embedding release restrictions; possible overlap with Emory/RSNA. |
| **CMMD** | Diagnostic-domain stress test. | Biopsy/tumor-enriched; **not a valid population-screening dismissal-rate test**. |
| **VinDr-Mammo** | Radiological-score/detection and scanner analyses. | BI-RADS labels are **not confirmed cancer outcomes**; credentialed access. |
| **CBIS-DDSM** | Legacy film/cancer localization stress test. | Digitized film and lesion-enriched sampling are not modern population FFDM screening. |

A **second independent true screening cohort with verified outcomes** is the most useful long-term improvement to clinical validity. Do not claim that IMDB automatically solves this requirement.

---

## 11. Mandatory blocker: reconcile NLBS denominators

**Do not launch a new full-model comparison before this audit is completed.**

**Observed mismatch:**

- Published: **5,997** exams; **5,935** unique, **149** cancer cases.
- Our Stage A3: **6,430** `(case_id, StudyInstanceUID)` pairs.
- Our Stage C: **6,359** unique study identifiers, **5,942** connected clusters, **151** cancer-positive exam records and **149** positive clusters.

**[PROPOSED] Exact audit:**

1. Re-read the original paper's unit definitions and check metadata release counts, folder grouping, plus 26,988 image-to-row mapping.
2. Build crosswalk with `folder case_id`, physical DICOM path, `StudyInstanceUID`, view, `split_cluster_id`, exam positive flag and any filename aliases.
3. Count duplicates at each level; confirm whether some source `case_id`s contain multiple studies and whether `StudyInstanceUID` collisions across case folders create connected components.
4. Verify why 149 cancer clusters become 151 cancer-positive study records; audit the two positive multi-exam entities noted in `memory/state.md` without releasing identifiers.
5. Make a **primary patient-like cluster-collapsed analysis** and an **examination-level sensitivity analysis**. Never silently switch denominators across models or figures.
6. Audit whether all four views of each intended examination were captured; compare one-study versus all-image aggregation; exclude only under written, label-blind criteria.
7. Record inclusion/exclusion flowchart counts that sum exactly from files → images → folders → studies → clusters → final eligible common cohort.
8. Document residual linkage uncertainty caused by stripped public identifiers. Do not call `split_cluster_id` a proven patient ID.

**Completion criterion:** A reproducible script and `reports/paper1/nlbs_cohort_reconciliation.md` explaining every number and the **primary analysis unit**.

---

## 12. Mandatory blocker: correct source calibration versus source testing

**Current flaw:** `val_calib` was used both for selecting `tau` and reporting source sensitivity/dismissal. This is **calibration-set** performance, not an unbiased estimate on independent source patients.

### 12.1 Preferred corrected prospective-style protocol for new analyses

**[PROPOSED]** Predeclare mutually patient-disjoint RSNA partitions:

1. `model_train`: train RSNA-supervised heads/fine-tuned networks.
2. `trust_internal_train_calib`: if re-training TRUST, used only for its training-time dynamic threshold. This is **not** final operating threshold calibration and not a source test.
3. `source_op_calib`: choose the final case-level 95% and 98% operating thresholds for **all six models**.
4. `source_heldout_test`: independent source estimate of sensitivity/dismissal/AUROC, **not used** in training or operating threshold selection.

A single patient-disjoint split trades training size against high-recall threshold precision. There are only **486 positive RSNA patients**; splitting into many groups gives few cancers to estimate 98% sensitivity. **Do not pick percentages without checking positive counts and threshold uncertainty.** A 60/10/15/15 training/internal/calibration/test design is merely an initial planning example, not an approved immutable split. Consider repeated cross-fitting only if compute allows; no hindsight split optimization based on NLBS outcomes.

### 12.2 If the $30 cap prevents retraining existing models

A **lower-cost exploratory bridge** may reuse frozen CE/TRUST checkpoints and disjoint subsets of the previously untouched-by-training RSNA `val_calib` cases for **new calibration vs separate source score testing**, using existing per-case score files if available. However:

- Those RSNA scores were already examined historically, so this is a **retrospective redesign**, not a genuinely untouched prospective experiment.
- With only **97 cancer-positive cases** in `val_calib`, splitting it gives very imprecise source-calibration and source-test sensitivity.
- The newly trained ResNet/DINO models would need the **same declared patient groups** and no forbidden use of the source test.
- Clearly identify any nonmatched training cohorts for the existing models.
- This bridge is **exploratory**, not a full replacement for a rigorously trained/tested, matched study.

**Stop condition:** If no defensible patient-disjoint source threshold/test design can be implemented, do not write a Q1-ready clinical deployment claim.

---

## 13. Full experimental protocol and information barriers

### Protocol P0 — Historical CE/TRUST report (COMPLETE)

Frozen 20-epoch CE/TRUST model heads, source thresholds selected on RSNA `val_calib`, original 20 November NLBS prediction files hash-locked before label join. **Preserve immutable** as historical feasibility evidence.

### Protocol P1 — Main six-model, source-frozen benchmark (PLANNED)

1. Lock cohort definition (NLBS) and RSNA patient partitions.
2. Lock six architectures, direct-pretrained checkpoint IDs, DINOv2 variant, ResNet training recipe, views and score aggregations.
3. Train ONLY RSNA-supervised models using source labels; freeze authors' released GMIC/NYU predictors.
4. Generate and store per-image/per-breast scores with valid input metadata and stable exam/cluster IDs.
5. On `source_op_calib` **alone**, select each model's largest feasible threshold achieving *empirical* 98% cancer sensitivity (primary), and 95% (secondary). Define ties and strict `< tau` behavior before scoring.
6. Freeze model weights, config, code revision, hash, chosen thresholds and patient mapping **before applying to NLBS**.
7. Run NLBS inference. No NLBS outcomes, folder outcome categories, previous NLBS failure-case identifiers, or unlabeled NLBS score distributions may guide individual model choice or preprocessing modifications.
8. Lock raw NLBS scores and full exam-level predictions with SHA-256 **before label join**. Since old NLBS results were already seen, describe this as a **post hoc expansion with label-free final inference**, not a pristine prospectively blinded model-selection process.
9. Join external labels only in a separated evaluation stage and compute metrics/confidence intervals.
10. Report both **all-model common eligible examination cohort** and model-specific coverage/failure counts; give a sensitivity analysis for excluded cases.

### Protocol P2 — Target-outcome calibration reference (OPTIONAL / separate)

After primary decisions and model rankings are locked, one may *retrospectively* calculate how thresholds **would** behave if NLBS labels were permitted. This is an **oracle/target-labeled reference**, not a zero-label result and not evidence that such calibration is available at a new hospital. Do not use it to tune or replace P1 outcomes.

### Protocol P3 — Unlabeled NLBS adaptation (MAINLY PAPER 2)

UPA and any AnchorShift method may use **unlabeled external images** in a predeclared adaptation subset, with patient-like cluster separation from evaluation. This violates strict **zero-target-data adaptation**, although it remains zero-target-**label** adaptation. Keep the comparison separate in Paper 2; if mentioned in Paper 1, clearly identify it as an ancillary **different-information** condition.

### Information barrier implementation

- The inference image manifest must not contain cancer fields or a class-folder signal (same principle as `datasets/nlbs/inference_manifest_stage_c.csv`).
- Model selection/checkpoint/threshold scripts must refuse all `nlbs` label paths and require a `source_only=true` assertion.
- Analysis scripts must not run before predictions and thresholds are checksum-locked.
- If any mistake forces a rerun after seeing NLBS outcomes, log the exact modification; do not claim renewed blindness.

---

## 14. Harmonization: fair comparisons without breaking models

### 14.1 Use each model's *validated native preprocessing*

- DICOM decode/VOI/photometric inversion must preserve clinical intensity and laterality.
- **GMIC:** Cropped 16-bit high-resolution images as official repository expects.
- **NYU:** Four-view canonical crops, model-specific resolutions, orientation, standardization and image-only checkpoint.
- **DINOv2:** Predeclared channel conversion, padding, image size and pretrained normalization.
- **ResNet-50:** Predeclared resolution, RGB replication, normalization and train/test consistency.
- **CE/TRUST:** Existing pinned Mammo-CLIP B5/TUST preprocessing, without silently changing historical checkpoint behavior.

Using all models with one arbitrary 512 × 512 resized image may improve code uniformity but **invalidate original pretrained model inference**. Harmonize **patient cohorts, labels, and decision rules**, not necessarily pixels that each original model was never trained to accept.

### 14.2 Examination-level score aggregation

| Family | Native level | Prespecified examination score |
|---|---|---|
| CE/TRUST/GMIC/DINOv2/ResNet | Image score | `exam_score = max(valid image cancer scores across this exam)` |
| NYU Four-View | Left/right breast malignant scores | `exam_score = max(left_malignant, right_malignant)` |

Do not average away a possible unilateral cancer without documented reason. If there are extra/missing views, choose the deterministic inclusion rule without reading labels. If only four-view cases are valid for NYU, create **primary common-four-view cohort**, plus a larger model-specific cohort when justified.

### 14.3 Model failure gates

- Check official sample output parity first; log tolerance and exact model versions.
- Minimum inference test: 10–20 anonymized images/exams across sites, scanner/vendor, view, and photometric polarity.
- Reject NaN/non-finite scores, view misorientation, label leakage, duplicate StudyInstanceUID ambiguities, and missing reproducibility hashes.
- Keep a complete exception file; **do not** remove failed cases after reading their outcomes.
- Train DINO/ResNet enough to demonstrate nontrivial source discrimination; do not portray a deliberately undertrained conventional model as a fair baseline.
- Pretrained cancer predictors are acceptable even if source AUROC is lower; **report**, do not quietly exclude.

---

## 15. Statistical analysis plan — prespecify before new NLBS result inspection

### 15.1 Primary endpoint

On the same matched cohort, at the threshold chosen using **98% empirical source-calibration sensitivity**, report for each model:

- NLBS examination or patient-cluster **sensitivity** with 95% confidence interval.
- NLBS **dismissal fraction** with 95% confidence interval.
- **Absolute missed-cancer count** and denominator.
- **Cancer prevalence among dismissed exams**, with uncertainty and explicit dependence assumptions.

**Secondary operating point:** RSNA-only **95% empirical** sensitivity threshold.

### 15.2 Equations and clear clinical meaning

Let `Y_i = 1` denote a cancer-positive examination and `S_i` the model's examination score. An examination is **dismissed** when `S_i < tau`; it is **retained** when `S_i >= tau`.

```text
Sensitivity(tau) = sum_i [Y_i = 1 and S_i >= tau] / sum_i [Y_i = 1]
Dismissal(tau)   = sum_i [S_i < tau] / N_exams
Missed cancers  = sum_i [Y_i = 1 and S_i < tau]
Dismissed cancer prevalence = Missed cancers / sum_i [S_i < tau]
```

Choose `tau` via source calibration only; with tied scores, apply a deterministic rank/tie rule specified in protocol. Never divide by zero without an explicit NA convention.

### 15.3 Uncertainty, dependency, and paired tests

- **Primary cluster-aware confidence intervals:** Patient-like `split_cluster_id` bootstrap resampling (e.g., ≥2,000 replicates, seeded) for sensitivity, dismissal, and **paired model differences**; explain linkage uncertainty.
- **Exact binomial Clopper–Pearson interval:** Show as an auxiliary descriptive reference when assumptions are reasonable; exam-level iid CP bounds may be **overconfident** if exams repeat within patients.
- **Paired comparisons:** Resample the **same patient clusters** across models to estimate sensitivity and dismissal differences; record discordant cancer decisions. Use paired tests (including McNemar when valid) cautiously given few positive clusters.
- **AUROC/AUPRC:** Report with uncertainty on source held-out and NLBS; AUPRC must be interpreted with prevalence. DeLong can be considered for paired AUROC, but clustering and multiplicity must be handled explicitly.
- **Multiple comparisons:** Predefine a small set of primary contrasts (e.g., each added model vs CE and vs TRUST on external sensitivity; plus model-family exploratory comparisons), adjust or clearly label exploratory results. Do not fish for the smallest p-value.
- **Missing/excluded cases:** Describe exact N for every denominator and subgroup; failures may be informative and should not disappear silently.
- **Precision caution:** With only ~149 NLBS cancer-positive clusters, 1 missed cancer changes sensitivity by approximately **0.67 percentage points**; a nominal 98% target cannot be assumed precise or prospectively guaranteed.

### 15.4 Source-target differences and mechanism decomposition

1. **Ranking change:** Source held-out vs external AUROC/AUPRC, with uncertainty; compare changes on matched definitions.
2. **Threshold transport:** Source-calibration-selected threshold, source-test observed sensitivity, external observed sensitivity and dismissal. Report all three, not only calibration vs NLBS.
3. **Class-conditional scores:** Cancer and noncancer score distributions on RSNA and NLBS, using NLBS labels **only in post hoc analysis**; compare quantiles/overlap, not only mean/median.
4. **ROC / sensitivity-versus-dismissal curves:** Show whether poor fixed-threshold performance is accompanied by a loss of **best attainable** case ranking, without recalibrating the deployed threshold.
5. **Cross-model cancer-error overlap:** Jaccard overlap and paired 2×2 tables of which cancer-positive exams each model would dismiss; avoid overinterpreting small numbers.
6. **Preprocessing shift:** Check scanner, view, intensity, bit depth, laterality and segmentation differences where reliably available.
7. **Attribution caution:** Stable AUROC plus shifted threshold behavior suggests score/operating-point shift, **not proof** of a causal mechanism. Differences in prevalence, labels, cohort selection and preprocessing can co-occur.

---

## 16. Pretrained-model and source-leakage investigation

Before any external-validation claim:

1. Record source institutions/datasets and training periods for each pretrained checkpoint (GMIC, NYU, Mammo-CLIP, DINOv2 ImageNet/curated pretraining, ResNet ImageNet).
2. Search original papers, released checkpoint cards, and code for **RSNA, EMBED/Emory, NLBS, OPTIMAM, NYU** or merged mammography collections.
3. Differentiate supervised cancer training, natural-image self-supervised pretraining and mammography-specific pretraining.
4. Where verifiable, exclude duplicated images/patients from source test; where not possible, explicitly label overlap **unknown** and present site-based sensitivity analysis.
5. Do not relabel all pretrained models as “trained on RSNA.” GMIC/NYU were originally developed using other source cohorts.
6. Preserve release SHA/commit, exact weight hashes, license versions and methods of downloading weights; never commit weights if upstream terms disallow it.

**Important precedent:** A 2026 Sensors benchmark already compared DINOv2-B/14, ResNet-50 and Mammo-CLIP among other backbones. That study is a **novelty competitor**, not permission to assume its checkpoints are appropriate for our different screening-triage task.

---

## 17. Budget: one Modal L40S, maximum $30 total additional spend

**[VERIFIED — SOURCE, 21 Nov 2026]** Official Modal pricing page listed L40S at **$0.000542 per GPU-second = $1.9512 GPU-hour**. CPU was **$0.0000131/core-second** and memory **$0.00000222/GiB-second** (terms/rates may change). At a *hypothetical allocation* of **2 CPU cores and 8 GiB RAM**, estimated resource cost is **about $2.11 per GPU-hour**, excluding other usage. At that rate, **$30 represents about 14.2 GPU-hours before reserve, extra preprocessing, volumes and retries**. Source: https://modal.com/pricing .

**[VERIFIED — REPO]** Prior Mammo-CLIP extraction benchmark: **2.2775 sec / batch of 80 images**, approximately **35 images/second**, yielding an **~8.0-hour projection for 1,000,000 repeated image occurrences**. That is a **different backbone and workload**; it is **not** a measured GMIC, NYU, ResNet or DINOv2 rate.

### 17.1 Spend envelope (planning CAP, not runtime promise)

| Work item | Maximum suggested spending | Gate before full run |
|---|---:|---|
| Preparation, CPU audit and misc. | **$3** | Dataset IDs and source split audited; reuse existing volumes. |
| GMIC full RSNA+NLBS inference | **$4** | Original sample-output parity, 200–500-image throughput projection fits cap. |
| NYU four-view full RSNA+NLBS inference | **$4** | Four-view adapter verified; test shows expected RAM/VRAM/throughput. |
| DINOv2 frozen embedding + RSNA head | **$6** | Correct ViT variant and nontrivial source pilot; embeddings written once. |
| ResNet-50 RSNA training + NLBS inference | **$9** | Projected *scientifically adequate training* and inference fit cap. |
| Safety margin for unexpected expenses | **$4** | Stop all new runs if projected total exceeds $30. |
| **TOTAL CAP** | **$30** | **Do not exceed.** |

This table describes **spending allocation only**. Full GMIC/NYU high-resolution inference or a properly trained ResNet-50 **may not finish within those cells**. The model is **not complete** merely because the planned allocation has been spent. If measured costs exceed available credit, **reduce wasted preprocessing, use cheaper CPU preprocessing and cache reuse, seek more compute, or report partial feasibility**; do not compromise clinical evaluation by silently dropping difficult cases or drastically degrading resolution.

### 17.2 Mandatory cost controls

- Confirm the **active Modal profile and remaining real credits**. Existing project logs refer to `hasinishrak2015` and `hasinishrak74001`; verify which **actual account** has `rsna-mammography-vol`, `nlbs-mammography-vol`, pretrained weights, and the $30 balance. Do not assume credits or volumes transfer across workspaces.
- Keep already-audited DICOMs/caches in volumes; prefer CPU decoding and cache staging over repeatedly decoding 81,694 raw images on GPU.
- For each model: run a **small reproducibility/smoke test**, then a **200–500-image (or comparable examination) timed benchmark** including data loading and preprocessing. For ResNet, also benchmark forward/backward steps and evaluate a realistic epoch plan.
- Estimate: `GPU-hours = number_of_units / measured_units_per_sec / 3600`, plus initialization, image processing, transfers, training epochs and reasonable overhead. Price **actual CPU/memory/GPU** together.
- Build per-job wall-clock and spending caps / an automated abort condition; checkpoint and flush predictions frequently enough to restart without redoing all work. Cancel stuck containers promptly.
- Do not use all budget on runs that are primarily environment/debugging work; perform code checks locally first.
- Do not promise **all four** added models fit into $30 before per-model benchmarks. No reliable runtime estimate is currently verified for those four workloads on **our** data.

### 17.3 Recommended order under tight budget

1. **Free/cheap first:** reconcile cohort, static data audits, protocol, source score file checks, integration and sample tests.
2. **GMIC** native pretrained inference pilot, then complete only if cost forecast fits.
3. **NYU four-view** native pretrained inference pilot, then complete only if cost forecast fits.
4. **DINOv2 frozen ViT-S/14** once-per-image feature extraction + train head on cached features.
5. **ResNet-50** only after full-train cost projection; if cost inadequate, do not label a weak pilot as a complete sixth model.
6. Statistics/figures can run largely on **CPU** using stored per-exam scores. Save the remainder for validation and reruns.

---

## 18. Exact implementation roadmap for Gemini Agent IDE

### Stage 0 — Create Paper 1 scaffold (ESSENTIAL, CPU)

- Read all `memory/` files, both root context files, latest commits, manifests, reports and current scripts.
- Preserve the functioning AnchorShift/Stage B-C pipeline; do not rewrite it just to implement Paper 1.
- Add `studies/paper1/{README.md,protocol_v1.md,model_manifest.yaml,eval_config.yaml,analysis_plan.md}` and `reports/paper1/` **only after version-control review**.
- Record six model IDs, candidate checkpoint URLs, licensing, original training datasets, fixed aggregation/operating thresholds, model-input needs and researcher-decided analysis tiers.
- **Deliverable:** Versioned protocol and code plan, no NLBS-driven model selection.

### Stage 1 — Dataset and split correction (ESSENTIAL, CPU)

- Reconcile 5,997/149 vs 6,359/151; produce patient-cluster and exam-level mapping table.
- Document RSNA patient/train/op-calib/source-test proposal and its positive counts. Identify where existing CE/TRUST scores are valid without retraining.
- Create **label-blind** four-view eligibility and extra-view selection rules for NYU; common-cohort mask across models.
- **Deliverable:** `reports/paper1/cohort_and_split_audit.md` plus deterministic split/coverage hashes.
- **Stop if:** Primary study unit remains unexplained or patient grouping is clearly invalid.

### Stage 2 — Pretrained model compatibility (ESSENTIAL, cheap pilots)

- Inspect official GMIC and NYU repositories, sample outputs, exact checkpoint URLs, license and software environment. Run sample inference and compare with authors' provided predictions.
- Reuse RSNA/NLBS raw or cropped DICOM paths but generate model-appropriate 16-bit images; do not assume the preexisting 1520×912 cache is sufficient.
- Perform leakage/provenance audit and 200–500-image cost pilots.
- **Deliverable:** `reports/paper1/gmic_nyu_parity_and_cost.md`; full inference GO/NO-GO.

### Stage 3 — GMIC and NYU source/target predictions (CONDITIONAL on budget)

- Inference all source calibration+test cases, and all eligible target cases, using native preprocessing.
- Save per-image or per-breast scores plus per-case aggregation, timing, failures, preprocessing hashes and model IDs.
- Select each model's final operating thresholds only on the source calibration set.
- **Deliverable:** Frozen RSNA and NLBS prediction artifacts with integrity hashes and coverage table.

### Stage 4 — DINOv2 RSNA head (CONDITIONAL on budget)

- Pin standard DINOv2 variant and preprocessing; extract a unique feature per image.
- Train reproducible logistic/linear or small head on RSNA **training** patients only. Separate model hyperparameter tuning from source operating calibration and source test.
- Evaluate source performance and lock checkpoint/threshold before NLBS test.
- **Deliverable:** DINOv2 source-trained head and fully labeled method provenance, not “pretrained DINO cancer model.”

### Stage 5 — ResNet-50 RSNA fine-tune (CONDITIONAL on budget)

- Implement adequate ImageNet-initialized end-to-end fine-tuning with positive sampling and loss suited to imbalance; evaluate a full source training curve.
- Use the same outcome/task definitions and clinical unit, but not necessarily the exact native pixels of the published GMIC/NYU models.
- Measure cost from a real training-steps benchmark before committing the final run; stop if the available money is insufficient for scientifically credible training.
- **Deliverable:** Adequately trained ResNet checkpoint, source test and locked external predictions, or an honest **NOT COMPLETED** entry.

### Stage 6 — Six-model analysis (ESSENTIAL after completed predictions, CPU)

- Merge keyed study/patient-cluster records after verifying no duplicate/missing labels and score freezes.
- Evaluate sensitivity, dismissal, cancers dismissed, missed-cancer overlap, AUROC/AUPRC, paired uncertainty, coverage and source-target score/threshold behavior.
- Primary analysis = common eligible cohort. Alternative unit = cluster-collapsed or exam analysis as declared.
- Interpret successful and failed transfers equally; no ranking based solely on favorable NLBS p-values.
- **Deliverable:** `reports/paper1/six_model_external_validation_results.md`, machine-readable JSON/CSV and figures.

### Stage 7 — Independent confirmation and manuscript gate (ESSENTIAL for strong Q1 aspiration)

- Review whether a second independent outcome-verified screening population is feasible and genuinely untouched for hypothesis/method selection.
- Perform an adversarial novelty audit against Meta-Repository, Sensors 2026 and any closer studies discovered since.
- Check CLAIM/STARD-AI reporting and risk of overlap with Paper 2.
- **Deliverable:** PROCEED / MODIFY / MERGE / ABANDON decision and an evidence-based journal shortlist with current *verified* Q1 categories, not guessed quartiles.

---

## 19. Project files and reproducibility artifacts to retain

**Do not change historic files in place.** For all six model runs write:

- `model_id`, `paper_track`, `run_id`, training-regime stratum, model architecture/version, public repository/revision, checkpoint URL/hash, software dependency snapshot, input preprocessing hash, image size, RGB/bit-depth policy, seed(s).
- Source split CSV checksum, patient-cluster policy, case aggregation rule, exact threshold algorithm and `source_op_calib` positive/negative counts.
- Model training history (if applicable), training selection criterion, learning rate, batch, resolution, augmentation, epochs, sample weighting and hardware.
- Inference execution time and measured throughput, actual Modal cost where billable, failure counts, GPU/CPU/memory allocation.
- Source calibration predictions, source held-out predictions, NLBS **pre-label-join** scores/predictions with SHA-256.
- Final evaluation script version, statistics seed/bootstrap draws, analysis cohort and dataset count reconciliation.
- Researcher notes documenting any deviations after NLBS results became known and why they occurred.

Suggested new files (proposals):

```text
reports/paper1/
  prior_work_novelty_matrix.md
  cohort_and_split_audit.md
  model_release_license_audit.md
  budget_and_benchmarks.csv
  gmic_nyu_parity_and_cost.md
  model_manifest_locked.json
  source_thresholds_locked.json
  nlbs_predictions_locked/     # only small prediction CSVs if repository policy permits
  six_model_external_validation_results.md
  figure_index.md
  reviewer_readiness_audit.md
research_logs/YYYY-MM-DD_paper1_<task>.md
```

Large raw images, licensed weights, patient-identifying information and unrestricted data exports remain outside GitHub, on authorized Modal volumes/local secure locations. **Never put access credentials in logs.**

---

## 20. Required figures and tables for the actual paper

| Exhibit | Essential scientific function |
|---|---|
| **Figure 1: cohort and information-flow diagram** | Explain RSNA source training/calibration/test, blinded external scoring, then label-only evaluation. Include historic vs corrected protocol. |
| **Figure 2: six-model frozen operating points** | Plot NLBS sensitivity (with CIs) against dismissal, at RSNA 98% and 95% thresholds. Compare along clinically meaningful trade-offs. |
| **Figure 3: source vs target discrimination and threshold shift** | Show AUROC change alongside frozen threshold sensitivity change, for each model. |
| **Figure 4: score distributions by outcome** | Separate cancer/noncancer score quantiles on source and target; show threshold location. |
| **Figure 5: false-dismissal case overlap** | Which cancer exams were dismissed by which models; use a readable overlap/UpSet diagram rather than uninterpretable tiny Venns. |
| **Table 1: cohorts and labels** | Patients/clusters, examinations, images, confirmed cancers, missing views, vendor composition, eligibility and exclusion reasons. |
| **Table 2: six-model provenance** | Code, weights, original training data, training regime, preprocessing, model task, release permissions. |
| **Table 3: source and external operating metrics** | Sensitivity, dismissed fraction, cancer misses, CIs, AUROC/AUPRC and exact denominators. |
| **Table 4: paired cancer-decision differences** | Paired confidence intervals and false-dismissal discordance without overclaiming significance. |
| **Supplement: budgets and runtime** | Actual measured compute/cost, failure logs, code hashes and negative/reproduction findings. |

**Never interpret dismissal rate as prospectively achieved radiologist workload savings**; workflow, staffing, double-reading and prospective reader behavior are not tested here.

---

## 21. Clinical/ethical interpretation, licensing and deployment limits

- Research is retrospective; independent ethics/data-access determinations may be required depending on source terms and institution, even when public de-identified data are used.
- No AI is actually dismissing patient examinations in this study.
- NLBS and RSNA label definitions and verification histories differ; these differences can create outcome/selection shift not solved by model architecture alone.
- NLBS 2008–2010 images predominantly come from one GE vendor; avoid worldwide or modern multi-vendor deployment generalizations.
- A 98% empirical calibration sensitivity target must not be described as a prospective safety certificate. Confidence bounds are sample- and assumption-dependent.
- Use the relevant current publication checklists: **CLAIM 2024** for medical-imaging AI and **STARD-AI 2025** for diagnostic accuracy reporting when applicable; consider **TRIPOD+AI** for prediction-model development/validation aspects. Do not claim every guideline is mandatory for every paper section.
- Model software licenses (e.g., GMIC/NYU **AGPL-3.0**, DINOv2 standard **Apache-2.0**) are not equivalent to permission to distribute every pretrained checkpoint or dataset. Record exact checkpoint license terms.
- Kaggle/RSNA noncommercial and no redistribution terms must be followed. NLBS is CC BY 4.0 with attribution. Never re-identify individuals or redistribute medical images from restricted sources.

---

## 22. Adversarial reviewer questions — prepare actual answers

### Reviewer A — “This novelty is already published.”

**Challenge:** The 2026 Meta-Repository compares five models on seven datasets; the 2026 Sensors benchmark already includes DINOv2, ResNet-50 and Mammo-CLIP with external results.

**Correct response:** Acknowledge both prominently, show a detailed protocol-level comparison. Our potentially different primary clinical outcome is **source-frozen high-recall whole-exam dismissal** with cancer-miss counts and error overlap on a population-screening program. If this exact focus also has strong direct precedent, recommend **MERGE** or repurpose the work; do not relabel old ideas as novel.

### Reviewer B — “Your source performance is optimistically estimated.”

**Challenge:** Prior `val_calib` was simultaneously used for operating-threshold selection and reported RSNA sensitivity.

**Correction:** New patient-disjoint source operating calibration and independent source test, or clearly mark the budget-constrained retrospective redesign as exploratory. Do not present calibration metrics as independent test results.

### Reviewer C — “These are not matched models.”

**Challenge:** GMIC/NYU are directly pretrained on another institution, while DINOv2/ResNet require RSNA task supervision, and CE/TRUST share a mammography-pretrained backbone.

**Correction:** Separate **pretrained-direct** and **RSNA-supervised** strata; common outcome/case/threshold protocol; record origins and possible overlap; avoid attributing all differences to architecture.

### Reviewer D — “Canadian outcomes have already influenced the study.”

**Challenge:** The first NLBS CE/TRUST results have been seen; new models were selected in response to them.

**Correction:** Explicitly date/document this as a **post hoc extension**. Keep new NLBS labels out of training/threshold and predeclare all additional analyses. Obtain a distinct independent screening cohort before strong confirmatory claims.

### Reviewer E — “Your sample is too small to establish screening safety.”

**Challenge:** ~149 cancer-positive clusters and 2.4% case mix are not enough for highly precise high-recall performance or universal safety.

**Correction:** Exact counts, cluster bootstrap CIs, no universal guarantees, transparent confidence bounds and external confirmatory cohort where possible.

### Reviewer F — “You undertrained the conventional comparator.”

**Challenge:** A $30 budget may force low resolution or insufficient ResNet training; DINOv2 is natural-image pretraining, not medical mammography training.

**Correction:** Run source-quality and throughput gates. If the conventional model is weak, improve appropriately or state **not completed**. Do not claim a specialist win against an intentionally handicapped baseline.

---

## 23. Go/no-go criteria for Paper 1 as a standalone submission

| Decision | Conditions |
|---|---|
| **PROCEED** | Meaningful protocol-level distinction from closest prior papers, resolved cohort units and source testing, clinically useful six-model (or justified subset) result, sound uncertainty, independently interpretable data, and preferably external confirmation. |
| **MODIFY** | Important finding possible but key model, calibration, legal/compute, or validation limitations remain. **Current verdict.** |
| **MERGE with AnchorShift** | Benchmark is useful primarily as the motivation/control experiment for Paper 2, but no independent scientific question/insight remains after literature comparison. |
| **ABANDON Paper 1** | Exact study already done more rigorously elsewhere or resource/outcome limitations prevent an honest, scientifically informative evaluation. |

**More models do not create novelty.** A rigorous **six-model** result showing different clinical threshold behavior with explanation can be interesting; a naive table of six AUROCs is unlikely to be enough.

---

## 24. Manuscript skeleton (draft structure, not paper conclusions)

1. **Abstract:** Specific high-recall transport question, retrospective methods, all sample sizes, frozen source threshold protocol, uncertainty and actual (not predicted) outcomes.
2. **Introduction:** Why source-operating sensitivity matters; existing mammography external-validation literature; what Meta-Repository, Sensors, UPA and TRUST already establish.
3. **Methods:** Dataset construction and eligibility, patient-level linkage and labels, all six models and original train regimes, native preprocessing, source train/calibration/test, target-label quarantine, threshold rules, metrics, paired bootstrap and registered exploratory status.
4. **Results:** Flowchart; per-model source and external performance; sensitivity/dismissal and cancer miss count; ROC/threshold decomposition; cross-model false-dismissal overlaps; sensitivity analyses.
5. **Discussion:** Evidence for successful or failed transfer; explanations and alternative explanations (cohort mix/preprocessing/vendor); comparison to prior work; clinical limits; implications for the *separate* AnchorShift method study.
6. **Limitations:** Single external program, few cancer-positive clusters, post hoc model expansion, cohort count ambiguity (if still not corrected), pretrained overlap uncertainty, matched-cohort selection, restricted compute and potential undertraining.
7. **Reproducibility:** Code commit, training/setup details, weights/terms, statistics, predeclared model list, data access statements.

Do not write a conclusion before the four new models have generated reliable predictions and the statistics have been reviewed.

---

## 25. Target journals and realistic publication posture

**Current assessment: weak as a completed two-model paper; plausible only with major improvements as a six-model evaluation study.** There is **no verified current JCR/SCImago Q1 category audit** in this document. Do not label an outlet Q1 on memory alone.

Potential **scope checks**, not acceptance promises: *Radiology: Artificial Intelligence*, *IEEE Journal of Biomedical and Health Informatics*, *Artificial Intelligence in Medicine*, *Computers in Biology and Medicine*, *PLOS Digital Health*, and *Scientific Reports*. The last journal's quartile must not be assumed in every metric/category; select journals based on a verified category/year and comparator **evaluation-only** articles.

**A strong Q1 submission likely needs:** independent source testing, denominator reconciliation, robust model implementations, source-frozen target evaluation, paired uncertainty and specific mechanistic findings. A **second independent screening outcome cohort** would materially improve the clinical evidence; the current $30 budget may cover only a feasibility-stage portion.

No fabricated publication probabilities, timelines or claim that “six models = Q1.”

---

## 26. What information this document does **not** yet verify

- That GMIC or NYU full native preprocessing plus all **81,694** RSNA/NLBS images can be completed within assigned GPU budget.
- That ResNet-50 can be fully trained with sufficient mammography resolution and multiple seeds for $9.
- That DINOv2's chosen natural-image ViT-S/14 representation will learn the RSNA breast-cancer task at adequate source performance.
- Whether every external checkpoint's training population has **zero overlap** with RSNA or NLBS.
- That the NLBS dataset has truly independent patient identifiers for all apparent connected clusters.
- That our case count mapping will reproduce the source authors' intended 5,997/5,935 unit exactly.
- That additional dataset IMDB contains outcome/selection definitions adequate for **primary population-screening dismissal** evaluation.
- That a distinct, previously unseen outcome-verified screening cohort can be obtained and used within licensing/compute limits.
- That the exact source-frozen six-model triage research question has no earlier stronger study.
- Which listed journal is Q1 in a particular subject/year and whether the work is accepted.

Keep these explicitly open until a verifiable source, log, or completed experiment resolves them.

---

## 27. Source register — academic papers, model code, data and reporting guidance

**Citation policy:** Link to the original paper/official repository where possible. All links below were verified as accessible or derived from the repository's research context as of **21 November 2026**. A link is not proof that every scientific implication in our proposed protocol is established. Refresh bibliographic metadata and versions before manuscript submission.

### Most directly relevant external validation and triage research

1. **Chłędowski et al. (2026 online / MEDINFO 2025 proceedings)**, *Meta-Repository of Screening Mammography Classifiers*. DOI: https://doi.org/10.3233/SHTI251263 . Code: https://github.com/nyukat/mammography_metarepository . **Five models / seven datasets: critical novelty competitor.**
2. **Cross-Sensor and Cross-Population Generalization of Deep Learning Models for Digital Mammography: A Controlled Four-Country Benchmark of Five Backbone Architectures with Statistical Significance Testing** (*Sensors*, 2026). DOI: https://doi.org/10.3390/s26123911 . **Already compares ResNet-50, DINOv2-B/14, Rad-DINO, Mammo-CLIP, Mammo-FM.**
3. **TRUST (2026)**, *TRUST: Threshold-Recalibrated Uncertainty-Safe Training for Certified Dismissal in Breast Cancer Screening*. https://arxiv.org/abs/2609.00300 . Original external threshold uses 20% labeled NLBS subset; author's name metadata should be checked against the precise cited version.
4. **Roschewitz et al. (2023)**, *Automatic correction of performance drift under acquisition shift in medical image classification*. *Nature Communications*. https://doi.org/10.1038/s41467-023-42396-y . Code: https://github.com/biomedia-mira/upa .
5. **Garrucho et al. (2022)**, *Domain generalization in deep learning based mass detection in mammography: A large-scale multi-center study*. *Artificial Intelligence in Medicine*. https://doi.org/10.1016/j.artmed.2022.102386 .
6. **MC-SDS (MICCAI 2025)**, *Domain Generalization for Mammogram Classification by Suppressing Domain-Specific Features*. https://doi.org/10.1007/978-3-032-04981-0_17 .
7. **MammoDG (2026)**, *MammoDG: Generalisable Deep Learning Advances Cross-Domain Multi-Center Breast Cancer Screening*. https://doi.org/10.1007/s11263-026-02979-w .
8. **Bernstein et al. (2026)**, *A novel statistical framework for quantifying risks and benefits of AI automation in screening mammography*. https://doi.org/10.1371/journal.pdig.0001231 .

### Exact six-model scientific references

9. **Shen et al. (2021)**, *An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization*. *Medical Image Analysis* 68:101908. https://doi.org/10.1016/j.media.2020.101908 . GMIC code: https://github.com/nyukat/GMIC .
10. **Wu et al. (online 2019, issue 2020)**, *Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening*. *IEEE Transactions on Medical Imaging*. https://doi.org/10.1109/TMI.2019.2945514 . Code: https://github.com/nyukat/breast_cancer_classifier .
11. **Oquab et al. (2023)**, *DINOv2: Learning Robust Visual Features without Supervision*. https://arxiv.org/abs/2304.07193 . Official code/weights: https://github.com/facebookresearch/dinov2 .
12. **He et al. (2016)**, *Deep Residual Learning for Image Recognition*. CVPR. https://www.microsoft.com/en-us/research/publication/deep-residual-learning-for-image-recognition/ . ResNet-50 weights/code: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html .
13. **Mammo-CLIP** (foundation encoder already used in our repo). Official project / weights as pinned in `src/models/mammo_clip_loader.py`: https://github.com/Shawn24/Mammo-CLIP . Check model-card license and original training provenance before describing independence.

### Dataset descriptions and conditions

14. **Trivedi et al. (2026)**, *Open-Source Dataset for the RSNA Screening Mammography Cancer Detection Challenge*. *Radiology: Artificial Intelligence*. https://doi.org/10.1148/ryai.250375 . Dataset access/terms: https://www.kaggle.com/competitions/rsna-breast-cancer-detection/rules .
15. **Kendall et al. (2025)**, *Full Field Digital Mammography Dataset from a Population Screening Program*. *Scientific Data*. https://doi.org/10.1038/s41597-025-05866-0 . Public NLBS data CC BY 4.0.
16. **Indian Mammography DataBase (2026)**, source description. https://doi.org/10.59275/j.melba.2026-bd61 . **Optional only**; verify population and outcome definition before inclusion.
17. **VinDr-Mammo**: https://doi.org/10.1038/s41597-023-02100-7 . Different cancer-outcome definition; cannot substitute for pathology-confirmed screening cohort.
18. **EMBED**: https://doi.org/10.1148/ryai.220047 . Access/terms: https://docs.hitilab.com/docs/datasets/embed/open-data-faq .

### Reporting/statistical research guidance

19. **Tejani et al. (2024)**, *Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update*. https://doi.org/10.1148/ryai.240300 .
20. **Sounderajah et al. (2025)**, *The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence*. https://www.nature.com/articles/s41591-025-03953-8 .
21. **TRIPOD+AI** — check the relevant 2024 reporting statement before using the prediction-model development and external-validation checklist. Do not confuse it with the separate TRIPOD-LLM document.
22. **Official Modal pricing** (price may change): https://modal.com/pricing .

---

## 28. Mandatory independent review / change log procedure

Before training any new model, Gemini must create a dated log that states:

- **Literature gate:** Any discovered paper that preempts or narrows the exact high-recall external-validation claim.
- **Code gate:** Existing scripts inspected; any modifications and their test coverage.
- **Data gate:** Dataset version, label definition and valid common-cohort counts.
- **Model gate:** The six fixed model IDs, release provenance and native preprocessing proof.
- **Budget gate:** Active profile, actual spend/credits, measured pilot throughput and per-model projected cost including CPU/memory.
- **Protocol gate:** RSNA calibration/test separation, NLBS outcome isolation, source threshold freeze.
- **Execution gate:** Command, version/hash, log URI or identifier, failure handling.
- **Results gate:** All numeric metrics from actual generated outputs, with no invented success probabilities.

After each completed stage, append a **new dated entry** to Paper 1's log and update `memory/state.md`/`memory/history.md` by preserving existing history. The project memory is **not** updated just because one assistant drafted a plan; GitHub requires an actual commit.

---

## 29. The three immediate next actions

1. **Gemini: audit and freeze the Paper 1 protocol.** Read both contexts and current scripts, reconcile NLBS 5,997/149 vs 6,359/151, and propose a patient-disjoint RSNA train/source-op-calib/source-test structure. Produce written exclusions, model strata, score rules and leakage checks. **No GPU run yet.**
2. **Gemini: integrate/pilot GMIC and NYU.** Obtain released cancer checkpoints under their terms, reproduce official sample outputs, prepare label-blind model-specific image adapters, and benchmark real Modal throughput. **Do not run the full external dataset until the cost fits the remaining credit.**
3. **Gemini: budget-gate DINOv2 and ResNet-50.** Use frozen DINOv2 features and a source-only classifier head; plan enough ResNet fine-tuning to be scientifically fair. Only run full jobs that fit $30 **total**. Save exact logs, source scores, checkpoints and predicted costs before requesting interpretation.

**What to tell ChatGPT after execution:** Provide commit SHA, model manifest, cohort reconciliation report, source calibration/test split counts, GMIC/NYU sample-parity logs, per-model projected/actual Modal cost, and any completed score CSVs. ChatGPT should critically inspect method, statistical validity and new scientific claims rather than assume success.

---

## 30. One-paragraph reusable project-memory summary

**Paper 1** is a separate retrospective evaluation study inside the **same `Research-TRR-HI/Breast-Cancer` GitHub repository** as **Paper 2, AnchorShift**. Paper 1 compares six prespecified screening-mammography AI configurations—our CE-only and independent TRUST Mammo-CLIP B5 heads, released pretrained GMIC, released pretrained NYU four-view CNN, a frozen DINOv2 encoder with RSNA-trained cancer head, and an ImageNet-initialized ResNet-50 trained on RSNA. The core question is whether **RSNA-selected empirical 98% (primary) and 95% (secondary) sensitivity thresholds transfer without Canadian outcome-label calibration to NLBS**, measured by cancer sensitivity, dismissal, missed cancers and uncertainty, and explained using ranking/score shifts and cross-model error overlap. The team already saw CE/TRUST NLBS results (90.07% and 84.77% sensitivity at the 98% source target, 15/151 and 23/151 missed exams); those historical results were prediction-locked before label join but the new extension is **post hoc**. Source calibration/test double-use and the NLBS 5,997/149 versus 6,359/151 unit discrepancy must be corrected. Four new models have **not** been tested. Budget is **$30 total** on Modal L40S, requiring cost-gated pilots and reuse of existing images. Multi-model generalization is already researched (2026 Meta-Repository and Sensors study), so Paper 1 novelty is **unconfirmed**; do not claim universal safety, guaranteed Q1 publication, or that DINOv2 itself is a breast-cancer classifier. AnchorShift Paper 2 is reserved for any genuinely distinct label-free adaptation method, benchmarked against UPA.

---

## 31. Final decision as of 21 November 2026

**MODIFY / INVESTIGATE.** The fixed six-model study has a clear, clinically meaningful research question and a realistic staged engineering plan, but the narrow research novelty remains unproven and the current dataset/statistical design has important weaknesses. We should **not** spend the $30 blindly. Correct denominators, fix source testing, independently reproduce GMIC/NYU native inference, benchmark actual cost, then complete DINOv2/ResNet only if scientific quality and credit permit. If the broader literature or results remove the claimed independent contribution, **MERGE Paper 1 into AnchorShift** rather than publishing an artificial second paper.
