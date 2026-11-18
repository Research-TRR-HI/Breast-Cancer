"""Stage B Unit and Smoke Tests.

Runs locally (CPU-only, no cloud resources) to verify:
1. TRUSTClassifier forward pass with input resolution (B, 3, 1520, 912).
2. Frozen backbone parameter verification (encoder requires_grad=False, head=True).
3. Loss calculations: BCE baseline, Focal loss, Dynamic Dismissal loss, Combined TRUST loss.
4. Dynamic threshold selection (compute_target_recall_threshold).
5. Clopper-Pearson 99% one-sided upper confidence bound.
6. Case-level max-score aggregation.
7. Patient-level split integrity (zero patient/image leakage).
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.trust_model import TRUSTClassifier, aggregate_case_scores
from src.models.losses import FocalLoss, DynamicDismissalLoss, TRUSTLoss
from src.training.calibration import (
    compute_target_recall_threshold,
    clopper_pearson_upper_bound,
    evaluate_dismissal_at_threshold,
)


def test_model_forward_and_frozen_encoder() -> None:
    """Verify that encoder is frozen and forward pass produces (B, 1) logits and probs."""
    # Use dummy linear feature extractor for fast local unit test
    dummy_encoder = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 2048),
    )
    for p in dummy_encoder.parameters():
        p.requires_grad = False

    model = TRUSTClassifier(encoder=dummy_encoder, embed_dim=2048, dropout_rate=0.3, freeze_encoder=True)

    # Check requires_grad
    for name, p in model.encoder.named_parameters():
        assert not p.requires_grad, f"Encoder parameter {name} should be frozen!"
    assert model.norm.weight.requires_grad, "LayerNorm weights should be trainable!"
    assert model.head.weight.requires_grad, "Linear head weights should be trainable!"

    # Forward pass with TRUST input resolution (batch of 2)
    x = torch.randn(2, 3, 1520, 912)
    logits, probs = model(x)

    assert logits.shape == (2, 1), f"Expected logits shape (2, 1), got {logits.shape}"
    assert probs.shape == (2, 1), f"Expected probs shape (2, 1), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probs must be in [0, 1]"

    # Backward pass
    loss = probs.sum()
    loss.backward()

    for name, p in model.encoder.named_parameters():
        assert p.grad is None, f"Encoder parameter {name} should have no gradient!"
    assert model.head.weight.grad is not None, "Linear head should have gradients!"


def test_losses() -> None:
    """Verify BCE, Focal, Dismissal, and combined TRUST loss computations."""
    logits = torch.tensor([[-2.0], [2.0], [-1.0], [3.0]], requires_grad=True)
    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    probs = torch.sigmoid(logits)

    # 1. Focal Loss
    fl = FocalLoss(gamma=2.0, alpha=0.25)
    loss_fl = fl(logits, targets)
    assert loss_fl.item() > 0.0
    assert torch.isfinite(loss_fl)

    # 2. Dynamic Dismissal Loss
    dl = DynamicDismissalLoss(margin=0.05)
    # Threshold at 0.8: positive with prob sigmoid(2.0) ~ 0.8808 -> 0.8 + 0.05 - 0.8808 < 0 -> 0 penalty
    # If tau_t is 0.9: 0.9 + 0.05 - 0.8808 = 0.0692 penalty
    loss_dl, n_pos = dl(probs, targets, tau_t=0.90)
    assert n_pos == 2
    assert loss_dl.item() > 0.0

    # 3. Combined TRUSTLoss
    trust_loss_fn = TRUSTLoss(baseline="trust", margin=0.05, lambda_focal=1.0, lambda_dismiss=1.0)
    total_loss, comp = trust_loss_fn(logits, targets, tau_t=0.90)

    assert total_loss.item() > 0.0
    assert "loss_bce" in comp
    assert "loss_focal" in comp
    assert "loss_dismiss" in comp
    assert abs(comp["loss_total"] - (comp["loss_bce"] + comp["loss_focal"] + comp["loss_dismiss"])) < 1e-4

    # 4. Pure CE Baseline mode
    ce_loss_fn = TRUSTLoss(baseline="ce")
    ce_total, ce_comp = ce_loss_fn(logits, targets)
    assert ce_comp["loss_focal"] == 0.0
    assert ce_comp["loss_dismiss"] == 0.0
    assert abs(ce_total.item() - ce_comp["loss_bce"]) < 1e-5


def test_dynamic_threshold_and_clopper_pearson() -> None:
    """Verify quantile-based threshold selection and Clopper-Pearson bound."""
    pos_scores = np.array([0.1, 0.2, 0.5, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 0.99])
    # Target recall 0.90 -> cutoff at 10th percentile
    tau_90 = compute_target_recall_threshold(pos_scores, target_recall=0.90)
    # 9 of 10 scores should be >= tau_90
    recall = np.mean(pos_scores >= tau_90)
    assert recall >= 0.90, f"Expected recall >= 0.90, got {recall}"

    # Clopper-Pearson 99% one-sided upper bound
    # 0 missed out of 100 dismissed -> upper bound ~ 0.045
    ub_0 = clopper_pearson_upper_bound(k_missed=0, n_dismissed=100, confidence=0.99)
    assert 0.04 < ub_0 < 0.05, f"Expected ~0.045, got {ub_0}"

    # 2 missed out of 100 dismissed
    ub_2 = clopper_pearson_upper_bound(k_missed=2, n_dismissed=100, confidence=0.99)
    assert ub_2 > ub_0, "Upper bound must increase with missed cancers"


def test_case_aggregation_and_dismissal_eval() -> None:
    """Verify maximum score aggregation per examination and clinical triage metrics."""
    scores = torch.tensor([0.2, 0.4, 0.9, 0.1, 0.3])
    case_ids = ["exam_1", "exam_1", "exam_2", "exam_3", "exam_3"]

    case_scores = aggregate_case_scores(scores, case_ids)
    assert abs(case_scores["exam_1"] - 0.4) < 1e-5
    assert abs(case_scores["exam_2"] - 0.9) < 1e-5
    assert abs(case_scores["exam_3"] - 0.3) < 1e-5

    c_scores = np.array([case_scores["exam_1"], case_scores["exam_2"], case_scores["exam_3"]])
    c_labels = np.array([0, 1, 0])  # exam_2 is cancer

    # Threshold at 0.5: exam_1 (0.4) and exam_3 (0.3) dismissed; exam_2 (0.9) retained
    metrics = evaluate_dismissal_at_threshold(c_scores, c_labels, threshold=0.5)
    assert metrics["n_dismissed"] == 2
    assert metrics["n_retained"] == 1
    assert metrics["detected_cancers"] == 1
    assert metrics["missed_cancers"] == 0
    assert metrics["triage_recall"] == 1.0
    assert metrics["dismissal_rate"] == 2.0 / 3.0


def test_rsna_splits_leakage_and_proportions() -> None:
    """Verify generated RSNA splits have ZERO patient leakage and correct proportions."""
    splits_path = Path("datasets/rsna/splits_seed42.csv")
    if not splits_path.exists():
        pytest.skip("datasets/rsna/splits_seed42.csv does not exist yet.")

    df = pd.read_csv(splits_path)
    train_pts = set(df[df["split"] == "train"]["patient_id"])
    calib_pts = set(df[df["split"] == "train_calib"]["patient_id"])
    test_pts = set(df[df["split"] == "val_calib"]["patient_id"])

    assert len(train_pts & calib_pts) == 0, "Patient leakage between train and train_calib!"
    assert len(train_pts & test_pts) == 0, "Patient leakage between train and val_calib!"
    assert len(calib_pts & test_pts) == 0, "Patient leakage between train_calib and val_calib!"

    total_pts = df["patient_id"].nunique()
    assert total_pts == 11913, f"Expected 11,913 unique patients, got {total_pts}"

    train_pct = len(train_pts) / total_pts
    calib_pct = len(calib_pts) / total_pts
    test_pct = len(test_pts) / total_pts

    assert 0.69 < train_pct < 0.71, f"Train frac {train_pct} deviated from 0.70"
    assert 0.09 < calib_pct < 0.11, f"Calib frac {calib_pct} deviated from 0.10"
    assert 0.19 < test_pct < 0.21, f"Test frac {test_pct} deviated from 0.20"


if __name__ == "__main__":
    test_model_forward_and_frozen_encoder()
    print("PASS: test_model_forward_and_frozen_encoder")
    test_losses()
    print("PASS: test_losses")
    test_dynamic_threshold_and_clopper_pearson()
    print("PASS: test_dynamic_threshold_and_clopper_pearson")
    test_case_aggregation_and_dismissal_eval()
    print("PASS: test_case_aggregation_and_dismissal_eval")
    test_rsna_splits_leakage_and_proportions()
    print("PASS: test_rsna_splits_leakage_and_proportions")
    print("\nALL UNIT TESTS PASSED LOCALLY!")
