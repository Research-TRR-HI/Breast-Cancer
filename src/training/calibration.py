"""Threshold Calibration and Certified Dismissal Metrics.

Reference:
    TRUST (2026): Section 3.2, 3.11, 4.3
    - Dynamic training-time threshold tau_t computed on internal training-calibration set.
    - Post-training operating threshold selection at target recall (95% and 98%).
    - Clopper-Pearson one-sided 99% upper confidence bound on cancer prevalence among dismissed cases.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


def compute_target_recall_threshold(
    positive_scores: np.ndarray,
    target_recall: float = 0.98,
) -> float:
    """Compute the decision threshold tau that achieves target recall on positive scores.

    To achieve recall >= target_recall, threshold tau must be at or below
    the (1 - target_recall) empirical quantile of positive scores:
        Recall = P(p >= tau | Y = 1) >= target_recall

    Args:
        positive_scores: 1D array of predicted probabilities for cancer-positive samples.
        target_recall: Desired recall (e.g. 0.95 or 0.98).

    Returns:
        Float threshold tau.
    """
    if len(positive_scores) == 0:
        return 0.0

    scores_sorted = np.sort(positive_scores)
    # Target percentile: (1 - target_recall)
    percentile = (1.0 - target_recall) * 100.0
    tau = float(np.percentile(scores_sorted, percentile))
    return tau


def clopper_pearson_upper_bound(
    k_missed: int,
    n_dismissed: int,
    confidence: float = 0.99,
) -> float:
    """Calculate the one-sided Clopper-Pearson exact upper confidence bound on missed-cancer rate.

    Args:
        k_missed: Number of cancer-positive cases among dismissed cases.
        n_dismissed: Total number of dismissed cases.
        confidence: Desired one-sided confidence level (default: 0.99).

    Returns:
        Upper confidence bound in [0.0, 1.0].
    """
    if n_dismissed <= 0:
        return 0.0
    if k_missed <= 0:
        # Rule of three / exact beta formula for 0 successes
        return float(1.0 - (1.0 - confidence) ** (1.0 / n_dismissed))

    # Upper bound from Beta distribution percent-point function
    ub = stats.beta.ppf(confidence, k_missed + 1, n_dismissed - k_missed)
    return float(ub)


def evaluate_dismissal_at_threshold(
    case_scores: np.ndarray,
    case_labels: np.ndarray,
    threshold: float,
    confidence: float = 0.99,
) -> Dict[str, float]:
    """Evaluate screening dismissal metrics at a fixed operating threshold.

    Decision rule:
        If case_score < threshold: DISMISSED (safe to skip radiologist review)
        If case_score >= threshold: RETAINED (sent for radiologist review)

    Args:
        case_scores: 1D array of case-level maximum scores.
        case_labels: 1D array of binary case-level cancer labels (1 = cancer, 0 = benign/normal).
        threshold: Operating cutoff tau.
        confidence: Confidence level for Clopper-Pearson bound (default: 0.99).

    Returns:
        Dictionary of clinical screening triage metrics.
    """
    total_cases = len(case_scores)
    total_cancers = int(np.sum(case_labels == 1))

    dismissed_mask = case_scores < threshold
    retained_mask = ~dismissed_mask

    n_dismissed = int(np.sum(dismissed_mask))
    n_retained = int(np.sum(retained_mask))

    dismissal_rate = float(n_dismissed) / max(total_cases, 1)

    # Missed cancers = cancer cases that fall below dismissal threshold
    missed_cancers = int(np.sum((case_labels == 1) & dismissed_mask))
    detected_cancers = int(np.sum((case_labels == 1) & retained_mask))

    triage_recall = float(detected_cancers) / max(total_cancers, 1) if total_cancers > 0 else 1.0

    # Cancer prevalence among dismissed cases (false omission rate)
    cancer_prev_dismissed = float(missed_cancers) / max(n_dismissed, 1) if n_dismissed > 0 else 0.0
    cp_upper_bound = clopper_pearson_upper_bound(missed_cancers, n_dismissed, confidence=confidence)

    return {
        "threshold": float(threshold),
        "total_cases": total_cases,
        "total_cancers": total_cancers,
        "n_dismissed": n_dismissed,
        "n_retained": n_retained,
        "dismissal_rate": dismissal_rate,
        "detected_cancers": detected_cancers,
        "missed_cancers": missed_cancers,
        "triage_recall": triage_recall,
        "cancer_prev_dismissed": cancer_prev_dismissed,
        "clopper_pearson_99_ub": cp_upper_bound,
    }
