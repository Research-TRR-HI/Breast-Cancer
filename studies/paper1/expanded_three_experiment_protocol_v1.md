# Paper 1 — Expanded Three-Experiment External Validation Protocol (v1.0)

**Effective Date:** 2026-11-24  
**Status:** PHASE 1 DESIGN & AUDIT (POST HOC PROTOCOL EXTENSION — ZERO PHASE 2 FULL RUNS LAUNCHED)  
**Primary Benchmark Preservation:** The original six-model Paper 1 benchmark ([protocol_v1.md](protocol_v1.md)) and its locked results ([six_model_summary.md](../../reports/paper1/six_model_final/six_model_summary.md)) remain completely unmodified and immutable.  
**Retrospective Disclosure:** The original six-model NLBS evaluation (`N = 6,184` common 4-view exams, `151` cancers) was completed and observed on `2026-11-24` (`ce_b5`: `0.7089` AUROC / `90.07%` @98%; `trust_b5`: `0.7040` / `84.77%`; `dinov2_rsna`: `0.6299` / `98.01%`; `resnet50_rsna`: `0.6452` / `97.35%`; `gmic_pretrained`: `0.8541` / `97.35%`; `nyu_fourview_pretrained`: `0.7925` / `100.00%` [`1.94%` dis]). This three-experiment expansion is therefore explicitly registered as a **post hoc protocol extension** designed after observing those outcomes, NOT a prospectively blinded pre-observation design.

---

## 1. Scientific Motivation & Three Hypotheses

The primary six-model benchmark revealed two major confounding dimensions:
1. **Supervision-Domain Confounding:** Strong Stratum B models (`gmic_pretrained`, `nyu_fourview_pretrained`) were trained on `229,426` NYU clinical exams rather than RSNA, whereas Stratum A models (`ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`) were supervised on RSNA (`8,339` train pts) and calibrated on RSNA `val_calib`.
2. **Single External Population:** NLBS represents one Canadian GE Senographe Essential screening program (`2008–2010`), preventing conclusions about multi-population or multi-vendor recurrence.

Three complementary experiments test three explicit falsifiable hypotheses (preserving strong and weak transfer findings equally):
- **Experiment 1 (Common RSNA Supervision, `dryousufmozumder`):** Compare all six architectures under identical RSNA training (`train` `8,339` pts), checkpoint selection (`train_calib` `1,192` pts), and frozen threshold calibration (`val_calib` `2,382` pts) on NLBS. *Hypothesis H1:* Common RSNA supervision still alters or degrades source-frozen operating-point transportability relative to original specialist checkpoints (or alternatively recovers utility—tested symmetrically).
- **Experiment 2 (Original Pretrained Mammography Models, `tigerwood693`):** Evaluate up to six eligible mammography-trained models (`GMIC`, `NYU Four-View`, `GLAM`, `End2End CNN`, `Faster R-CNN CAD` [subject to malignancy-endpoint eligibility], `Petrini Two-View EfficientNet-B3`) on NLBS **without RSNA training or RSNA calibration**, using Category A (compatible original threshold), Category B (approved non-RSNA source calibration), or Category C (threshold-independent AUROC/AUPRC only). *Hypothesis H2:* Generalization and operating-point stability remain heterogeneous even without RSNA supervision/calibration.
- **Experiment 3 (Multiple Unseen Populations, `tigerwood697`):** Evaluate all six RSNA-supervised configurations from Experiment 1 with frozen RSNA thresholds across unseen populations: **NLBS** (Canada, screening), **IMDB** (India, screening-naive/opportunistic-diagnostic audit required), **CMMD** (China, biopsy-enriched diagnostic stress test), and **LUMINA** (Turkey, diagnostic/multi-vendor stress test). *Hypothesis H3:* Threshold-transfer and discrimination behaviors recur or diverge systematically across unseen geographic and clinical domains.

---

## 2. Shared Scientific Guardrails & Label Quarantine

1. **Strict Target-Label Isolation:** No NLBS, IMDB, CMMD, or LUMINA cancer outcomes may ever be used for training, fine-tuning, hyperparameter tuning, checkpoint selection, threshold calibration, preprocessing optimization, or performance-driven case exclusion.
2. **Pre-Join Cryptographic Locking:** All external model predictions must be generated from label-free inference manifests (`assert_no_forbidden_columns`), aggregated to the prespecified unit (exam / breast / image per dataset eligibility), and SHA-256 locked BEFORE outcome joining.
3. **Clinical Composition Separation (Experiment 3):**
   - **NLBS** is the primary routine population-screening evaluation.
   - **IMDB** must be audited for diagnostic vs opportunistic-screening mix and histopathology vs follow-up label reliability; BI-RADS must never substitute for confirmed cancer outcomes.
   - **CMMD** and **LUMINA** are biopsy-enriched / diagnostic-domain stress tests: report discrimination (AUROC/AUPRC) and sensitivity/specificity cleanly, but **NEVER** interpret dismissal rates, PPV/NPV, or workload reduction as representative of routine population screening, and **NEVER** pool clinically heterogeneous cohorts into one headline metric.
4. **No Fabricated Views or Benign Labels:** Never duplicate views to satisfy multi-view models (evaluate 4-view models on verified common 4-view subsets without fabrication), and never treat `cancer == 0` in RSNA as confirmed benign biopsy status.
5. **Operating Points:**
   - Experiments 1 & 3 use RSNA `val_calib` frozen `98%` (primary) and `95%` (secondary) empirical recall thresholds (`score < tau` dismissed).
   - Experiment 2 strictly forbids RSNA calibration; each model is gated into Category A / B / C before inference.
6. **Budget & Scope Boundary:** Existing project cap is `USD 30` additional spend; combined Phase 2 execution across the three accounts (`~USD 20` planning envelope each) requires explicit consolidated budget approval. Paper 2 (`AnchorShift`) remains strictly separate.

---

## 3. Adversarial Literature Matrix (Novelty & Overlap Audit)

| Prior Study | Venue / Year / DOI | Scope & Models | Threshold & Outcome Protocol | Overlap Threat to Paper 1 | Candidate Residual Distinction (Unconfirmed Until Verified) |
|---|---|---|---|---|---|
| **Meta-Repository of Screening Mammography Classifiers** (Chłędowski et al.) | MEDINFO 2025 / *SHTI* 2026 (`10.3233/SHTI251263`), `nyukat/mammography_metarepository` | **5 models across 7 international datasets**; standardized Docker evaluation, AUC/AUPRC + uncertainty. | Discrimination & calibration focus; does not isolate source-frozen 98%/95% screening dismissal across RSNA-supervised vs direct-pretrained strata. | **HIGH:** Directly occupies multi-model international mammography benchmarking. | Paired comparison of **common-RSNA vs direct-pretrained** regimes under **source-frozen 98%/95% triage dismissal** with pre-join locking and missed-cancer overlap. |
| **Cross-Sensor & Cross-Population Generalization** (Gabbualoy et al.) | *Sensors* June 2026 (`10.3390/s26123911`) | **ResNet-50, DINOv2-B/14, Rad-DINO, Mammo-CLIP B5, Mammo-FM** trained on CBIS-DDSM -> CMMD, DMID, MIAS. | AUC drop (`0.165–0.320`) and **sensitivity at 95% specificity**; legacy film source (CBIS-DDSM). | **HIGH:** Exact backbone family overlap (ResNet-50, DINOv2, Mammo-CLIP) + cross-country shift. | Modern FFDM source (**RSNA**) -> population screening (**NLBS**) + IMDB/CMMD/LUMINA at **high-sensitivity dismissal (`98%/95%` recall)** rather than 95% specificity. |
| **Robustness of Deep Networks for Mammography** (Replication Across Public Datasets) | *J. Digital Imaging* 2023 / `arXiv:2206.12407` (`10.1007/s10278-023-00943-5`) | Cross-dataset replication of mammography CNNs across public archives; highlights steep external AUC drops & split/label pitfalls. | Focuses on replication AUC drops and dataset pitfalls rather than fixed high-recall triage thresholds. | **MODERATE:** Establishes cross-public-archive AUC degradation. | Isolates **ranking degradation vs score-distribution shift** at clinical rule-out thresholds (`tau_98`, `tau_95`). |
| **Garrucho et al.** | *Artif. Intell. Med.* 2022 (`10.1016/j.artmed.2022.102386`) | Multi-center mass detection across 5 unseen domains (vendor, density, lesion shifts). | Mass detection localization/FROC, not whole-exam screening dismissal. | **LOW-MODERATE:** Establishes multi-vendor shift. | Whole-exam cancer-dismissal endpoint and cross-model error overlap. |
| **MC-SDS** & **MammoDG** | MICCAI 2025 (`10.1007/978-3-032-04981-0_17`) & IJCV 2026 (`10.1007/s11263-026-02979-w`) | Mammography domain-generalization architectures (frequency perturbation / contrastive cross-view). | Method development for AUC generalization. | **LOW-MODERATE:** Shows DG methods exist. | Paper 1 makes **no new DG algorithm claim**; benchmarks off-the-shelf & standard-supervised regimes. |
| **TRUST** | `arXiv:2609.00300` (2026) | Mammo-CLIP B5 + focal/dismissal loss; RSNA -> NLBS triage dismissal (`12.95%` @98%). | Uses **20% labeled NLBS** cases to recalibrate external threshold (`Protocol 3`). | **MODERATE:** Same RSNA -> NLBS high-recall dismissal setting. | Tests **zero target labels** (`Protocol 1`) across 6 architectures and isolates score-shift failure of source-frozen cutoffs. |
| **UPA** (Roschewitz et al.) | *Nat. Commun.* 2023 (`10.1038/s41467-023-42396-y`) | Unlabeled CDF score alignment under acquisition shift (`biomedia-mira/upa`). | Unlabeled target adaptation (`Protocol 2`). | **MODERATE:** Shows score drift is known and sometimes correctable. | Reserved for Paper 2 (`AnchorShift`) comparator; Paper 1 tests **unadapted** frozen deployment. |
| **Bernstein et al.** | *PLOS Digit. Health* 2026 (`10.1371/journal.pdig.0001231`) | Statistical risk-benefit framework for AI screening automation / rule-out. | Evaluates fixed-model trade-offs, not cross-hospital source-frozen model comparison. | **LOW-MODERATE:** Defines rule-out safety framing. | Empirical multi-model, multi-country stress test of source-frozen rule-out. |
| **Dataset-Origin Signatures** | `arXiv:2607.15416` (2026) | Cross-dataset origin detectability & shortcut risks when mixing CMMD/CBIS-DDSM with screening. | Shortcut & dataset-signature analysis. | **SUPPORTING:** Explains why CMMD/diagnostic cohorts behave differently from NLBS. | Reinforces strict separation of screening (`NLBS`) vs diagnostic (`CMMD`, `LUMINA`) metrics. |

---

## 4. Parallel Execution & Handoff Contract (Phase 1 -> Phase 2)

1. **Agent 1 (`paper1-exp1-phase1`, `dryousufmozumder`):** Completes GMIC & NYU RSNA fine-tuning pipelines + Phase 2 training, then exports frozen checkpoints, SHA-256 hashes, preprocessing versions, and frozen RSNA `tau_98`/`tau_95` thresholds to **Agent 3** (`exp1_handoff_manifest.json`).
2. **Agent 2 (`paper1-exp2-phase1`, `tigerwood693`):** Runs independently on NLBS using eligible original mammography checkpoints and Category A/B/C threshold rules (reusing locked `gmic_pretrained` and `nyu_fourview_pretrained` NLBS scores where compatible).
3. **Agent 3 (`paper1-exp3-phase1`, `tigerwood697`):** Independently executes dataset audits/preparations (`IMDB`, `CMMD`, `LUMINA`) and external evaluation for the 4 completed RSNA-supervised models (`ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`) immediately, then plugs in Agent 1's `gmic_rsna_finetuned` and `nyu_fourview_rsna_finetuned` checkpoints upon handoff.
