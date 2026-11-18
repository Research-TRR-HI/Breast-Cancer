"""Threshold Calibration, Dynamic Training Threshold, and Certified Dismissal Metrics.

Reference:
    TRUST (Shafie et al., 2026, arXiv:2609.00300v1): Sections 3.5, 3.7, 3.8, 3.12
    - Training-time dynamic threshold tau_t (Section 3.8, Page 9):
      Updated at the beginning of each epoch using image-level predictions on the
      training-calibration subset. Selects the largest threshold for which the one-sided
      95% Clopper-Pearson upper bound on positive-image rate among dismissed calibration
      images is <= 1% (0.01). Defaults to 0.0 if no threshold satisfies the bound.
    - Post-training operating threshold selection (Section 3.7, Page 8):
      Selects the largest threshold whose empirical case-level recall meets or exceeds
      the target recall (98% or 95%) under strict dismissal (score < tau -> dismissed).
    - Clopper-Pearson one-sided upper confidence bound (Eq. 5, Page 8):
      1.0 when n_d == 0 or k_d == n_d; Beta^{-1}_{confidence}(k_d + 1, n_d - k_d) otherwise.
      Training uses 95% confidence (delta=0.05); final case-level evaluation uses 99% (delta=0.01).
"""

import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy import stats


def clopper_pearson_upper_bound(
    k_missed: int,
    n_dismissed: int,
    confidence: float = 0.99,
) -> float:
    """Calculate the one-sided Clopper-Pearson exact upper confidence bound (TRUST Eq. 5).

    Args:
        k_missed: Number of positive observations among dismissed observations.
        n_dismissed: Total number of dismissed observations.
        confidence: One-sided confidence level in (0, 1) (e.g. 0.95 training, 0.99 eval).

    Returns:
        Exact one-sided upper confidence bound in [0.0, 1.0].
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    if isinstance(k_missed, bool) or isinstance(n_dismissed, bool):
        raise ValueError("k_missed and n_dismissed must be integer counts, not bool")
    k_val = int(k_missed)
    n_val = int(n_dismissed)
    if k_val != k_missed or n_val != n_dismissed:
        raise ValueError(f"k_missed ({k_missed}) and n_dismissed ({n_dismissed}) must be integers")
    if n_val < 0 or k_val < 0:
        raise ValueError(f"Counts must be non-negative, got k_missed={k_val}, n_dismissed={n_val}")
    if k_val > n_val:
        raise ValueError(f"k_missed ({k_val}) cannot exceed n_dismissed ({n_val})")

    # TRUST Eq. (5): return 1.0 when n_d == 0 or k_d == n_d
    if n_val == 0 or k_val == n_val:
        return 1.0

    if k_val == 0:
        # Exact Beta^{-1}_{confidence}(1, n_d) = 1 - (1 - confidence)^(1 / n_d)
        ub = 1.0 - (1.0 - float(confidence)) ** (1.0 / float(n_val))
    else:
        ub = float(stats.beta.ppf(float(confidence), k_val + 1, n_val - k_val))

    if not np.isfinite(ub):
        raise ValueError(
            f"Non-finite Clopper-Pearson bound computed for k={k_val}, n={n_val}, confidence={confidence}"
        )
    return float(min(max(ub, 0.0), 1.0))


def compute_dynamic_training_threshold_with_diagnostics(
    image_scores: np.ndarray,
    image_labels: np.ndarray,
    max_upper_bound: float = 0.01,
    confidence: float = 0.95,
) -> Tuple[float, Dict[str, Any]]:
    """Compute training-time provisional threshold tau_t plus explicit Clopper-Pearson diagnostics.

    Matches TRUST Section 3.8 (Page 9):
        - Evaluates candidate thresholds under dismissal rule ``score < tau``.
        - Computes one-sided ``confidence`` (95%) Clopper-Pearson upper bound on
          positive-image prevalence among dismissed calibration images.
        - Returns ``(tau_t, diagnostics)`` where ``tau_t`` is the largest candidate threshold
          whose upper bound <= ``max_upper_bound`` (0.01), or ``0.0`` if none satisfies the bound.
    """
    scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(image_labels, dtype=np.int64).reshape(-1)
    n_calib = int(len(scores))
    if n_calib == 0 or n_calib != len(labels):
        raise ValueError(
            f"image_scores ({len(scores)}) and image_labels ({len(labels)}) must be non-empty and equal length"
        )
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("image_labels must contain only binary values {0, 1}")
    if not (0.0 < max_upper_bound < 1.0):
        raise ValueError(f"max_upper_bound must be in (0, 1), got {max_upper_bound}")

    # Exact minimum zero-miss dismissed count so 1 - (1 - confidence)^(1 / N) <= max_upper_bound
    min_zero_miss_n = int(math.ceil(math.log(1.0 - float(confidence)) / math.log(1.0 - float(max_upper_bound))))
    n_neg = int(np.sum(labels == 0))
    n_pos = int(np.sum(labels == 1))
    zero_miss_feasible = bool(n_calib >= min_zero_miss_n and n_neg >= min_zero_miss_n)

    unique_scores = np.unique(scores)
    candidates = np.concatenate([unique_scores, [float(unique_scores[-1]) + 1e-7]])

    for tau in candidates[::-1]:
        dismissed_mask = scores < tau
        n_d = int(np.sum(dismissed_mask))
        if n_d == 0:
            continue
        k_d = int(np.sum((labels == 1) & dismissed_mask))
        ub = clopper_pearson_upper_bound(k_missed=k_d, n_dismissed=n_d, confidence=confidence)
        if ub <= max_upper_bound:
            tau_val = float(tau)
            diagnostics = {
                "tau_t": tau_val,
                "n_calib": n_calib,
                "n_pos_calib": n_pos,
                "n_neg_calib": n_neg,
                "min_zero_miss_dismissed_required": min_zero_miss_n,
                "zero_miss_feasible": zero_miss_feasible,
                "n_dismissed": n_d,
                "k_missed": k_d,
                "cp_upper_bound": float(ub),
                "is_nonzero_threshold": bool(tau_val > 0.0),
                "used_zero_fallback": bool(tau_val == 0.0),
            }
            return tau_val, diagnostics

    diagnostics = {
        "tau_t": 0.0,
        "n_calib": n_calib,
        "n_pos_calib": n_pos,
        "n_neg_calib": n_neg,
        "min_zero_miss_dismissed_required": min_zero_miss_n,
        "zero_miss_feasible": zero_miss_feasible,
        "n_dismissed": 0,
        "k_missed": 0,
        "cp_upper_bound": 1.0,
        "is_nonzero_threshold": False,
        "used_zero_fallback": True,
    }
    return 0.0, diagnostics


def compute_dynamic_training_threshold(
    image_scores: np.ndarray,
    image_labels: np.ndarray,
    max_upper_bound: float = 0.01,
    confidence: float = 0.95,
) -> float:
    """Compute training-time provisional threshold tau_t on image-level train_calib predictions."""
    tau, _ = compute_dynamic_training_threshold_with_diagnostics(
        image_scores=image_scores,
        image_labels=image_labels,
        max_upper_bound=max_upper_bound,
        confidence=confidence,
    )
    return float(tau)


def compute_target_recall_threshold(
    positive_scores: np.ndarray,
    target_recall: float = 0.98,
) -> float:
    """Select the largest operating threshold achieving empirical recall >= target_recall.

    Matches TRUST Section 3.7 (Page 8):
        Under strict dismissal (``score < tau`` is dismissed; ``score >= tau`` is retained),
        the largest threshold retaining at least ``ceil(target_recall * C)`` positive cases
        (accounting for ties and finite-sample discrete recall) is the ``k_max``-th smallest
        positive score where ``k_max = C - ceil(target_recall * C)``.
    """
    if not (0.0 < target_recall <= 1.0):
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")
    pos = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    n_pos = len(pos)
    if n_pos == 0:
        return 0.0

    scores_sorted = np.sort(pos)
    min_retained = int(np.ceil(float(target_recall) * float(n_pos) - 1e-12))
    min_retained = min(max(min_retained, 1), n_pos)
    k_max_missed = n_pos - min_retained
    tau = float(scores_sorted[k_max_missed])

    achieved = float(np.mean(pos >= tau))
    if achieved + 1e-12 < target_recall:
        raise RuntimeError(
            f"Selected threshold {tau} achieved recall {achieved:.6f} < target {target_recall:.6f}"
        )
    return tau


def evaluate_dismissal_at_threshold(
    case_scores: np.ndarray,
    case_labels: np.ndarray,
    threshold: float,
    confidence: float = 0.99,
    target_recall: Optional[float] = None,
) -> Dict[str, Union[float, int, bool, Optional[float]]]:
    """Evaluate case-level screening dismissal metrics at a fixed operating threshold.

    Decision rule:
        case_score < threshold  -> DISMISSED
        case_score >= threshold -> RETAINED
    """
    scores = np.asarray(case_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(case_labels, dtype=np.int64).reshape(-1)
    if len(scores) == 0 or len(scores) != len(labels):
        raise ValueError(
            f"case_scores ({len(scores)}) and case_labels ({len(labels)}) must be non-empty and equal length"
        )
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("case_labels must contain only binary values {0, 1}")

    total_cases = len(scores)
    total_cancers = int(np.sum(labels == 1))

    dismissed_mask = scores < float(threshold)
    retained_mask = ~dismissed_mask

    n_dismissed = int(np.sum(dismissed_mask))
    n_retained = int(np.sum(retained_mask))
    dismissal_rate = float(n_dismissed) / float(total_cases)

    missed_cancers = int(np.sum((labels == 1) & dismissed_mask))
    detected_cancers = int(np.sum((labels == 1) & retained_mask))
    triage_recall = float(detected_cancers) / float(total_cancers) if total_cancers > 0 else 1.0

    cancer_prev_dismissed = float(missed_cancers) / float(n_dismissed) if n_dismissed > 0 else 0.0
    cp_upper_bound = clopper_pearson_upper_bound(missed_cancers, n_dismissed, confidence=confidence)

    meets_target = (
        bool(triage_recall + 1e-12 >= float(target_recall))
        if target_recall is not None
        else True
    )

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
        "meets_target_recall": meets_target,
        "reported_dismissal_rate": dismissal_rate if meets_target else None,
    }
