# purpose.md — AnchorShift (Breast Cancer Screening AI Research)

## The One Line Version
Test whether high-sensitivity, case-level mammography dismissal trained and thresholded on RSNA transfers to NLBS with zero target cancer labels, and whether unlabeled-target adaptation (UPA) fixes any failure or leaves an unsolved shift gap.

## Core Mission & Protocol Discipline
- **Task:** Screening mammography triage / selective dismissal (case-level max image score vs. frozen threshold), NOT autonomous cancer diagnosis.
- **Key Distinctions:**
  1. Strict source-only / frozen-threshold transfer (protocol 1 — first experiment).
  2. Unsupervised target adaptation (protocol 2 — UPA on unlabeled NLBS images).
  3. Target-label calibration (protocol 3 — TRUST's 20% NLBS threshold-search reference only).
- **Guardrails:**
  - No premature novelty claims or new AnchorShift architecture before reproducing baselines.
  - Strict patient-level grouping (no view/laterality/repeat-exam leakage; handle NLBS's 62 repeat exams).
  - Do not misrepresent CMMD (biopsy-enriched), VinDr-Mammo (BI-RADS), or EMBED (restricted DUA) as replacements for population screening cohorts.
