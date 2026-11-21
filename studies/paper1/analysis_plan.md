# Paper 1 — Statistical Analysis Plan (Prespecified)

## 1. Primary & Secondary Endpoints
- **Primary Endpoint:** On the common 4-view NLBS cohort (`N = 6,184` studies, `151` cancer-positive exams, `5,868` `split_cluster_id`s), at each model's **RSNA-only 98% empirical sensitivity threshold (`tau_98`)**:
  - NLBS cancer sensitivity (`retained cancers / 151`) with 95% cluster-bootstrap CI (`B = 2,000`).
  - NLBS dismissal rate (`dismissed studies / 6,184`) with 95% cluster-bootstrap CI.
  - Absolute missed cancers (`k / 151`) and cancer prevalence among dismissed exams (`k / n_dismissed`, plus one-sided 99% Clopper–Pearson upper bound).
- **Secondary Operating Point:** RSNA-only 95% empirical sensitivity threshold (`tau_95`).
- **Secondary Cohorts:**
  1. Full NLBS cohort (`N = 6,359` studies, `151` pos) for the 5 view-agnostic models (`ce_b5`, `trust_b5`, `gmic_pretrained`, `dinov2_rsna`, `resnet50_rsna`).
  2. Cluster-collapsed NLBS cohort (`N = 5,868` 4-view / `5,942` full clusters, `149` positive clusters, `max` aggregation).
  3. RSNA independent source-test performance (2-fold patient-disjoint cross-fitting within `val_calib`, `N = 2,382` patients / `97` pos, plus single-split `fold_a -> fold_b`).

## 2. Mechanism Decomposition & Cross-Model Overlap
1. **Discrimination vs. Operating-Point Shift:** Compare RSNA 2-fold OOF AUROC/AUPRC vs NLBS AUROC/AUPRC alongside sensitivity drop (`RSNA OOF sens -> NLBS sens`) and class-conditional score quantiles (`p05, p25, median, p75, p95` for cancer vs non-cancer).
2. **Paired Cluster Contrasts:** Resample identical `split_cluster_id` units across models (`B = 2,000`) to estimate paired `delta_sensitivity`, `delta_dismissal`, and `delta_auroc` with 95% CIs.
3. **Missed-Cancer Overlap:** Compute pairwise Jaccard overlap and an UpSet matrix across the `151` cancer-positive examinations dismissed at `tau_98` and `tau_95`.

## 3. Stratum & Falsification Guardrails
- Always separate **Stratum A (RSNA-supervised: `ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`)** and **Stratum B (Pretrained-direct: `gmic_pretrained`, `nyu_fourview_pretrained`)** in main results tables.
- Do not pool across strata to claim architectural superiority without acknowledging differing training data, labels, and supervision histories.
- Publish null/successful-transfer results equally if any model preserves 95%/98% sensitivity on NLBS.
