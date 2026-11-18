"""Comprehensive Local CPU Test Suite for AnchorShift TRUST Pipeline (Categories A-G).

Runnable locally with zero cloud or GPU resources:
    python tests/test_stage_b_smoke.py
    python -m unittest discover -s tests -p "test_*.py" -v
"""

import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage
import torch
import torch.nn as nn

from src.data.feature_cache import (
    CacheIntegrityError,
    extract_or_resume_cache,
    load_cache_manifest,
    verify_cache_integrity,
    write_cache_manifest,
)
from src.data.preprocessing import (
    MAMMO_CLIP_MEAN,
    MAMMO_CLIP_STD,
    apply_trust_augmentation,
    decode_dicom_array,
    extract_cropped_mammogram,
    get_preprocessing_config_hash,
    prepare_tensor_from_cropped,
    preprocess_dicom,
    sample_augmentation_params,
)
from src.data.rsna_dataset import BalancedBatchSampler, build_epoch_draw_schedule
from src.models.losses import DynamicDismissalLoss, FocalLoss, TRUSTLoss
from src.models.mammo_clip_loader import load_and_verify_mammo_clip
from src.models.trust_model import (
    MammoCLIPEncoder,
    TRUSTClassifier,
    aggregate_case_labels,
    aggregate_case_scores,
)
from src.training.calibration import (
    clopper_pearson_upper_bound,
    compute_dynamic_training_threshold,
    compute_target_recall_threshold,
    evaluate_dismissal_at_threshold,
)
from src.training.checkpointing import load_training_checkpoint, save_training_checkpoint
from src.training.parity_pilot import run_online_vs_cached_parity_pilot


def _create_synthetic_dicom(
    path: Path,
    pixel_array: np.ndarray,
    photometric: str = "MONOCHROME2",
    laterality: str = "L",
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
    window_center: float = None,
    window_width: float = None,
) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = pixel_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.ImageLaterality = laterality
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept
    if window_center is not None and window_width is not None:
        ds.WindowCenter = window_center
        ds.WindowWidth = window_width
    ds.PixelData = pixel_array.astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


class TestAnchorShiftPipeline(unittest.TestCase):
    """Comprehensive unit and integration tests for TRUST pipeline corrections."""

    # =========================================================================
    # A. Model Tests
    # =========================================================================
    def test_a1_mammo_clip_852_strict_key_parity_and_freeze(self) -> None:
        encoder = MammoCLIPEncoder(checkpoint_path=None, freeze=True, model_name="efficientnet-b5")
        self.assertEqual(len(encoder.encoder.state_dict()), 852)

        # Save synthetic 852-key Mammo-CLIP tarball with 'image_encoder.' prefix and load strictly
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "synth_b5.tar"
            raw_sd = {f"image_encoder.{k}": v.clone() for k, v in encoder.encoder.state_dict().items()}
            torch.save({"model": raw_sd}, str(ckpt_path))

            fresh = MammoCLIPEncoder(checkpoint_path=None, freeze=False, model_name="efficientnet-b5")
            diag = load_and_verify_mammo_clip(fresh, ckpt_path, strict=True, verify_sha256=False)
            self.assertTrue(diag["is_fully_compatible"])
            self.assertEqual(diag["matched_keys_count"], 852)
            self.assertEqual(diag["total_target_keys"], 852)
            self.assertEqual(diag["missing_keys_count"], 0)
            self.assertEqual(diag["unexpected_keys_count"], 0)

    def test_a2_model_forward_frozen_isolation_and_single_sigmoid(self) -> None:
        dummy_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, 2048),
        )
        for p in dummy_encoder.parameters():
            p.requires_grad = False

        model = TRUSTClassifier(encoder=dummy_encoder, embed_dim=2048, dropout_rate=0.3, freeze_encoder=True)
        model.train()
        self.assertFalse(model.encoder.training)

        for name, p in model.encoder.named_parameters():
            self.assertFalse(p.requires_grad, f"Encoder parameter {name} leaked requires_grad")
        self.assertTrue(model.norm.weight.requires_grad)
        self.assertTrue(model.head.weight.requires_grad)

        x = torch.randn(2, 3, 128, 96)
        logits, probs = model(x)
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(probs.shape, (2, 1))
        self.assertTrue(torch.allclose(probs, torch.sigmoid(logits), atol=1e-6))

        loss = logits.sum()
        loss.backward()
        for name, p in model.encoder.named_parameters():
            self.assertIsNone(p.grad, f"Encoder parameter {name} leaked gradient")
        self.assertIsNotNone(model.head.weight.grad)

    def test_a3_case_aggregation_and_length_validation(self) -> None:
        scores = torch.tensor([0.2, 0.4, 0.9, 0.1, 0.3])
        case_ids = ["exam_1", "exam_1", "exam_2", "exam_3", "exam_3"]
        agg = aggregate_case_scores(scores, case_ids)
        self.assertAlmostEqual(agg["exam_1"], 0.4, places=5)
        self.assertAlmostEqual(agg["exam_2"], 0.9, places=5)
        self.assertAlmostEqual(agg["exam_3"], 0.3, places=5)

        lbl_agg = aggregate_case_labels([0, 0, 1, 0, 0], case_ids)
        self.assertEqual(lbl_agg, {"exam_1": 0, "exam_2": 1, "exam_3": 0})

        with self.assertRaises(ValueError):
            aggregate_case_scores([0.1, 0.2], ["exam_1"])
        with self.assertRaises(ValueError):
            aggregate_case_scores([], [])

    # =========================================================================
    # B. Loss Tests (Hand-Calculated Mathematical Verification)
    # =========================================================================
    def test_b1_losses_match_paper_equations_exactly(self) -> None:
        logits = torch.tensor([[-2.0], [0.5], [1.2], [-0.8]], dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
        probs = torch.sigmoid(logits)

        # Manual BCE
        p_np = probs.detach().numpy().reshape(-1)
        y_np = targets.numpy().reshape(-1)
        manual_bce = float(-np.mean(y_np * np.log(p_np) + (1.0 - y_np) * np.log(1.0 - p_np)))

        # Manual Focal (TRUST Eq. 12: gamma=2.0, NO alpha weighting)
        gamma = 2.0
        manual_focal = float(
            -np.mean(
                y_np * ((1.0 - p_np) ** gamma) * np.log(p_np)
                + (1.0 - y_np) * (p_np ** gamma) * np.log(1.0 - p_np)
            )
        )

        # Manual Dismissal (TRUST Eq. 7: tau_t=0.60, margin=0.10 -> boundary 0.70)
        tau_t = 0.60
        margin = 0.10
        pos_probs = p_np[y_np == 1.0]
        manual_dismiss = float(np.mean(np.maximum(0.0, tau_t + margin - pos_probs)))
        manual_total = manual_bce + 1.0 * manual_focal + 0.05 * manual_dismiss

        fl = FocalLoss(gamma=2.0, alpha=None)
        self.assertAlmostEqual(float(fl(logits, targets).item()), manual_focal, places=6)

        dl = DynamicDismissalLoss(margin=0.10)
        dl_val, n_pos = dl(probs, targets, tau_t=tau_t)
        self.assertEqual(n_pos, 2)
        self.assertAlmostEqual(float(dl_val.item()), manual_dismiss, places=6)

        trust_loss_fn = TRUSTLoss(
            baseline="trust", margin=0.10, focal_gamma=2.0, focal_alpha=None, lambda_focal=1.0, lambda_dismiss=0.05
        )
        tot, comp = trust_loss_fn(logits, targets, tau_t=tau_t)
        self.assertAlmostEqual(comp["loss_bce"], manual_bce, places=6)
        self.assertAlmostEqual(comp["loss_focal"], manual_focal, places=6)
        self.assertAlmostEqual(comp["loss_dismiss"], manual_dismiss, places=6)
        self.assertAlmostEqual(float(tot.item()), manual_total, places=6)

        # Pure CE mode
        ce_loss_fn = TRUSTLoss(baseline="ce")
        ce_tot, ce_comp = ce_loss_fn(logits, targets, tau_t=tau_t)
        self.assertEqual(ce_comp["loss_focal"], 0.0)
        self.assertEqual(ce_comp["loss_dismiss"], 0.0)
        self.assertAlmostEqual(float(ce_tot.item()), manual_bce, places=6)

        # Zero positive samples edge case in DynamicDismissalLoss
        dl_zero, n_zero = dl(probs[[0, 3]], targets[[0, 3]], tau_t=0.5)
        self.assertEqual(n_zero, 0)
        self.assertEqual(float(dl_zero.item()), 0.0)

    # =========================================================================
    # C. Calibration & Clopper-Pearson Tests
    # =========================================================================
    def test_c1_clopper_pearson_edge_cases_and_validation(self) -> None:
        # n_d == 0 -> 1.0 (NEVER 0.0)
        self.assertEqual(clopper_pearson_upper_bound(0, 0, confidence=0.99), 1.0)
        self.assertEqual(clopper_pearson_upper_bound(0, 0, confidence=0.95), 1.0)
        # k_d == n_d -> 1.0
        self.assertEqual(clopper_pearson_upper_bound(5, 5, confidence=0.99), 1.0)
        # k_d == 0 -> 1 - (1 - conf)^(1/n)
        expected_zero = 1.0 - (1.0 - 0.95) ** (1.0 / 300.0)
        self.assertAlmostEqual(clopper_pearson_upper_bound(0, 300, confidence=0.95), expected_zero, places=10)
        self.assertTrue(expected_zero <= 0.01)

        # Invalid inputs raise ValueError
        with self.assertRaises(ValueError):
            clopper_pearson_upper_bound(5, 4, confidence=0.99)
        with self.assertRaises(ValueError):
            clopper_pearson_upper_bound(-1, 10, confidence=0.99)
        with self.assertRaises(ValueError):
            clopper_pearson_upper_bound(0, 10, confidence=1.0)
        with self.assertRaises(ValueError):
            clopper_pearson_upper_bound(0, 10, confidence=0.0)

    def test_c2_dynamic_training_threshold_cp95_and_fallback(self) -> None:
        # 300 negative images at 0.01..0.30, 5 positives at 0.50..0.90
        neg_scores = np.linspace(0.01, 0.30, 300)
        pos_scores = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
        scores = np.concatenate([neg_scores, pos_scores])
        labels = np.array([0] * 300 + [1] * 5)

        tau = compute_dynamic_training_threshold(scores, labels, max_upper_bound=0.01, confidence=0.95)
        # Dismissing all 300 negatives (k=0, n=300) gives CP95 = 1 - 0.05^(1/300) = 0.009936 <= 0.01 -> tau == 0.50
        self.assertAlmostEqual(tau, 0.50, places=6)

        # When no threshold can satisfy CP95 <= 0.01 (e.g., positive at minimum score or N too small), returns 0.0
        small_scores = np.array([0.05, 0.10, 0.20, 0.80])
        small_labels = np.array([0, 0, 0, 1])
        self.assertEqual(
            compute_dynamic_training_threshold(small_scores, small_labels, max_upper_bound=0.01, confidence=0.95),
            0.0,
        )

    def test_c3_target_recall_threshold_ties_and_eval_reporting(self) -> None:
        # Ties at 0.20 with C=10 positives; 95% recall requires >= 10 -> tau must be 0.10 or 0.20 depending on ties
        pos_scores = np.array([0.10, 0.20, 0.20, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
        tau_98 = compute_target_recall_threshold(pos_scores, target_recall=0.98)
        self.assertAlmostEqual(tau_98, 0.10, places=6)
        self.assertGreaterEqual(np.mean(pos_scores >= tau_98), 0.98)

        # Target 0.80 -> min_retained=8 -> k_max=2 -> 3rd smallest = 0.20 -> ties at 0.20 retain 9/10 >= 0.80
        tau_80 = compute_target_recall_threshold(pos_scores, target_recall=0.80)
        self.assertAlmostEqual(tau_80, 0.20, places=6)
        self.assertGreaterEqual(np.mean(pos_scores >= tau_80), 0.80)

        # Evaluation reporting: N/A when recall target not met
        case_scores = np.array([0.05, 0.15, 0.85, 0.95])
        case_labels = np.array([0, 1, 0, 1])
        res_fail = evaluate_dismissal_at_threshold(case_scores, case_labels, threshold=0.50, target_recall=0.98)
        self.assertFalse(res_fail["meets_target_recall"])
        self.assertIsNone(res_fail["reported_dismissal_rate"])

    # =========================================================================
    # D. Preprocessing & Augmentation Tests (Synthetic DICOM Fixtures)
    # =========================================================================
    def test_d1_dicom_luts_monochrome_flip_crop_and_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Create synthetic 120x100 mammogram with breast on the LEFT side (cols 0..45 bright, dark background)
            arr_left = np.full((120, 100), 100, dtype=np.uint16)
            arr_left[20:100, 0:45] = 2000  # bright breast tissue on left
            arr_left[50:60, 15:25] = 100   # internal hole to test binary_fill_holes

            # Right-breast image is horizontally mirrored version
            arr_right = np.fliplr(arr_left).copy()

            # MONOCHROME1 version of left image (inverted raw intensities)
            arr_mono1 = (4000 - arr_left).astype(np.uint16)

            p_left = _create_synthetic_dicom(
                tmp_dir / "left.dcm", arr_left, photometric="MONOCHROME2", laterality="L",
                rescale_slope=2.0, rescale_intercept=-100.0, window_center=1900, window_width=3800
            )
            p_right = _create_synthetic_dicom(
                tmp_dir / "right.dcm", arr_right, photometric="MONOCHROME2", laterality="R",
                rescale_slope=2.0, rescale_intercept=-100.0, window_center=1900, window_width=3800
            )
            p_mono1 = _create_synthetic_dicom(
                tmp_dir / "mono1.dcm", arr_mono1, photometric="MONOCHROME1", laterality="L"
            )

            # 1. Right-breast horizontal mirroring produces identical cropped array as left-breast
            a_l, ph_l, lat_l = decode_dicom_array(p_left)
            a_r, ph_r, lat_r = decode_dicom_array(p_right)
            c_l = extract_cropped_mammogram(a_l, ph_l, lat_l)
            c_r = extract_cropped_mammogram(a_r, ph_r, lat_r)
            self.assertEqual(c_l.shape, c_r.shape)
            self.assertLess(float(np.max(np.abs(c_l - c_r))), 1e-5)

            # Verify cropping reduced dimensions (did not stretch full 120x100 without cropping)
            self.assertLess(c_l.shape[0], 120)
            self.assertLess(c_l.shape[1], 100)

            # 2. MONOCHROME1 inversion yields bright breast on dark background
            a_m1, ph_m1, lat_m1 = decode_dicom_array(p_mono1)
            c_m1 = extract_cropped_mammogram(a_m1, ph_m1, lat_m1)
            self.assertGreater(float(c_m1[c_m1.shape[0] // 2, 10]), 0.9)
            self.assertEqual(float(c_m1[0, -1]), 0.0)

            # 3. Deterministic evaluation tensor shape and Mammo-CLIP standardization
            t1 = preprocess_dicom(p_left, target_size=(1520, 912), augment_params=None)
            t2 = preprocess_dicom(p_left, target_size=(1520, 912), augment_params=None)
            self.assertEqual(tuple(t1.shape), (3, 1520, 912))
            self.assertTrue(torch.equal(t1, t2))
            # Check all 3 channels identical
            self.assertTrue(torch.equal(t1[0], t1[1]))
            self.assertTrue(torch.equal(t1[0], t1[2]))
            # Verify standardized bounds correspond to [0, 1] input
            unnorm = t1 * MAMMO_CLIP_STD + MAMMO_CLIP_MEAN
            self.assertGreaterEqual(float(unnorm.min().item()), -1e-5)
            self.assertLessEqual(float(unnorm.max().item()), 1.0 + 1e-5)

            # 4. Reproducible seeded augmentation within paper bounds
            rng1 = np.random.default_rng(123)
            rng2 = np.random.default_rng(123)
            aug1 = sample_augmentation_params(rng1)
            aug2 = sample_augmentation_params(rng2)
            self.assertEqual(aug1, aug2)
            self.assertTrue(-10.0 <= aug1.angle_deg <= 10.0)
            self.assertTrue(0.90 <= aug1.scale <= 1.10)
            self.assertTrue(-0.05 <= aug1.tx_frac <= 0.05)
            self.assertTrue(-0.05 <= aug1.brightness_delta <= 0.05)
            self.assertTrue(0.90 <= aug1.contrast_factor <= 1.10)

            ta1 = preprocess_dicom(p_left, target_size=(1520, 912), augment_params=aug1)
            ta2 = preprocess_dicom(p_left, target_size=(1520, 912), augment_params=aug2)
            self.assertTrue(torch.equal(ta1, ta2))
            self.assertFalse(torch.equal(t1, ta1))

    # =========================================================================
    # E. Dataset & Sampler Tests
    # =========================================================================
    def test_e1_rsna_splits_zero_leakage_and_proportions(self) -> None:
        splits_path = Path("datasets/rsna/splits_seed42.csv")
        self.assertTrue(splits_path.exists(), "datasets/rsna/splits_seed42.csv must exist")

        df = pd.read_csv(splits_path)
        train_pts = set(df[df["split"] == "train"]["patient_id"])
        calib_pts = set(df[df["split"] == "train_calib"]["patient_id"])
        test_pts = set(df[df["split"] == "val_calib"]["patient_id"])

        self.assertEqual(len(train_pts & calib_pts), 0)
        self.assertEqual(len(train_pts & test_pts), 0)
        self.assertEqual(len(calib_pts & test_pts), 0)
        self.assertEqual(df["patient_id"].nunique(), 11913)
        self.assertEqual(len(df), 54706)

    def test_e2_balanced_batch_sampler_policies_and_state(self) -> None:
        labels = np.array([1] * 40 + [0] * 180)

        # Paper independent sampling: 20 pos + 60 neg per batch
        s_ind = BalancedBatchSampler(labels, batch_size=80, n_pos=20, seed=42, epoch_policy="paper_independent")
        state = s_ind.get_state()
        batches1 = list(s_ind)
        self.assertEqual(len(batches1), 3)
        for b in batches1:
            self.assertEqual(len(b), 80)
            self.assertEqual(int((labels[b] == 1).sum()), 20)
            self.assertEqual(int((labels[b] == 0).sum()), 60)

        # State restoration yields identical sequence
        s_ind.set_state(state)
        batches2 = list(s_ind)
        self.assertEqual(batches1, batches2)

        # Full negative pass visits 100% of negatives
        s_full = BalancedBatchSampler(labels, batch_size=80, n_pos=20, seed=42, epoch_policy="full_neg_pass")
        seen_neg = set()
        for b in s_full:
            seen_neg.update(idx for idx in b if labels[idx] == 0)
        self.assertEqual(len(seen_neg), 180)

    # =========================================================================
    # F. Augmentation-Aware Caching Integrity & Recovery Tests
    # =========================================================================
    def test_f1_cache_integrity_checksum_corruption_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            labels = np.array([1] * 25 + [0] * 100)
            sampler = BalancedBatchSampler(labels, batch_size=80, n_pos=20, seed=42, steps_per_epoch=2)
            aug_rng = np.random.default_rng(99)
            schedules = {ep: build_epoch_draw_schedule(sampler, ep, aug_rng) for ep in (1, 2)}

            def _extract_batch(rec):
                feats = torch.tensor(rec.indices, dtype=torch.float32).unsqueeze(1).repeat(1, 16)
                lbls = torch.tensor([labels[i] for i in rec.indices], dtype=torch.float32)
                img_ids = [f"img_{i}" for i in rec.indices]
                return feats, lbls, img_ids

            def _eval_extract(split_name: str):
                return (
                    torch.ones(10, 16),
                    torch.tensor([1, 0] * 5, dtype=torch.float32),
                    [f"{split_name}_{i}" for i in range(10)],
                    [f"pt_{i}" for i in range(10)],
                )

            manifest = extract_or_resume_cache(
                cache_root=cache_root,
                schedules_by_epoch=schedules,
                extract_batch_fn=_extract_batch,
                eval_extract_fn=_eval_extract,
                model_checkpoint_sha256="model_sha_abc",
                preprocess_config_hash="prep_hash_xyz",
                split_manifest_sha256="split_sha_123",
                seed=42,
                batch_size=80,
                n_pos=20,
                epoch_policy="paper_independent",
                embed_dim=16,
                batches_per_shard=1,
            )

            # Valid verification passes
            verify_cache_integrity(
                cache_root, "model_sha_abc", "prep_hash_xyz", "split_sha_123",
                expected_epochs=2, expected_steps_per_epoch=2, expected_schedules=schedules
            )

            # Wrong checkpoint hash rejected
            with self.assertRaises(CacheIntegrityError):
                verify_cache_integrity(cache_root, "wrong_sha", "prep_hash_xyz", "split_sha_123")

            # Wrong preprocessing hash rejected
            with self.assertRaises(CacheIntegrityError):
                verify_cache_integrity(cache_root, "model_sha_abc", "wrong_prep", "split_sha_123")

            # Duplicate draw detection
            dup_manifest = dict(manifest)
            dup_manifest["train_shards"] = list(manifest["train_shards"]) + [manifest["train_shards"][0]]
            write_cache_manifest(cache_root, dup_manifest)
            with self.assertRaises(CacheIntegrityError):
                verify_cache_integrity(cache_root, "model_sha_abc", "prep_hash_xyz", "split_sha_123")
            write_cache_manifest(cache_root, manifest)

            # Missing/corrupted shard detected, and extract_or_resume_cache recovers cleanly
            shard0 = cache_root / manifest["train_shards"][0]["file"]
            shard0.unlink()
            (cache_root / "stale.pt.tmp").write_bytes(b"corrupt")
            with self.assertRaises(CacheIntegrityError):
                verify_cache_integrity(cache_root, "model_sha_abc", "prep_hash_xyz", "split_sha_123")

            recovered = extract_or_resume_cache(
                cache_root=cache_root,
                schedules_by_epoch=schedules,
                extract_batch_fn=_extract_batch,
                eval_extract_fn=_eval_extract,
                model_checkpoint_sha256="model_sha_abc",
                preprocess_config_hash="prep_hash_xyz",
                split_manifest_sha256="split_sha_123",
                seed=42,
                batch_size=80,
                n_pos=20,
                epoch_policy="paper_independent",
                embed_dim=16,
                batches_per_shard=1,
            )
            self.assertFalse((cache_root / "stale.pt.tmp").exists())
            verify_cache_integrity(
                cache_root, "model_sha_abc", "prep_hash_xyz", "split_sha_123",
                expected_epochs=2, expected_steps_per_epoch=2, expected_schedules=schedules
            )
            self.assertEqual(len(recovered["train_shards"]), 4)

    # =========================================================================
    # G. Online-vs-Cached Parity Pilot & Checkpoint Save/Resume Parity Tests
    # =========================================================================
    def test_g1_online_vs_cached_parity_pilot_cpu(self) -> None:
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        embed_dim = 32
        target_size = (64, 48)

        # Deterministic frozen encoder
        encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, embed_dim),
        )
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

        def _make_split(n_pos: int, n_neg: int, split_name: str):
            n_tot = n_pos + n_neg
            crops = [rng.uniform(0.0, 1.0, size=(48, 36)).astype(np.float32) for _ in range(n_tot)]
            df = pd.DataFrame({
                "patient_id": [f"{split_name}_pt_{i // 2}" for i in range(n_tot)],
                "image_id": [f"{split_name}_img_{i}" for i in range(n_tot)],
                "cancer": [1] * n_pos + [0] * n_neg,
            })
            return crops, df

        tr_c, tr_df = _make_split(24, 96, "train")
        tc_c, tc_df = _make_split(10, 90, "train_calib")
        vc_c, vc_df = _make_split(12, 88, "val_calib")

        with tempfile.TemporaryDirectory() as tmp:
            res = run_online_vs_cached_parity_pilot(
                encoder=encoder,
                train_cropped=tr_c,
                train_calib_cropped=tc_c,
                val_calib_cropped=vc_c,
                train_df=tr_df,
                train_calib_df=tc_df,
                val_calib_df=vc_df,
                cache_root=Path(tmp) / "pilot_cache",
                embed_dim=embed_dim,
                target_size=target_size,
                epochs=3,
                batch_size=80,
                n_pos=20,
                steps_per_epoch=2,
                epoch_policy="paper_independent",
                seed=42,
                use_amp=False,
                device=torch.device("cpu"),
                micro_batch_size=40,
                batches_per_shard=2,
            )

        self.assertEqual(res["status"], "PASS", f"Parity pilot failed: {res}")
        self.assertLessEqual(res["max_abs_feature_diff"], 1e-5)
        self.assertLessEqual(res["max_abs_step_loss_diff"], 1e-5)
        self.assertLessEqual(res["max_abs_epoch_loss_diff"], 1e-5)
        self.assertLessEqual(res["max_abs_grad_diff"], 1e-5)
        self.assertLessEqual(res["max_abs_weight_diff"], 1e-5)
        self.assertLessEqual(res["max_abs_logit_diff"], 1e-5)
        self.assertLessEqual(res["max_abs_prob_diff"], 1e-5)
        self.assertLessEqual(res["max_abs_tau_diff"], 1e-6)
        self.assertEqual(res["prediction_agreement_95"], 1.0)
        self.assertEqual(res["prediction_agreement_98"], 1.0)
        self.assertEqual(res["online_eval_98"], res["cached_eval_98"])

    def test_g2_checkpoint_save_and_exact_resume_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "resume.pt"
            labels = np.array([1] * 25 + [0] * 100)
            cfg = {"baseline": "trust", "seed": 42, "epochs": 2}

            def _make_state():
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                head = nn.Sequential(nn.LayerNorm(16), nn.Dropout(0.3), nn.Linear(16, 1))
                opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
                sampler = BalancedBatchSampler(labels, batch_size=80, n_pos=20, seed=42, steps_per_epoch=2)
                aug_rng = np.random.default_rng(777)
                return head, opt, sampler, aug_rng

            def _run_one_epoch(head, opt, sampler, aug_rng, crit, tau_t):
                losses = []
                for idxs in sampler:
                    _ = sample_augmentation_params(aug_rng)
                    bx = torch.randn(len(idxs), 16)
                    by = torch.tensor(labels[idxs], dtype=torch.float32).unsqueeze(1)
                    opt.zero_grad()
                    loss, _ = crit(head(bx), by, tau_t=tau_t)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))
                return losses

            crit = TRUSTLoss(baseline="trust")

            # Missing checkpoint must raise FileNotFoundError (never silently restart at epoch 1)
            h0, o0, s0, a0 = _make_state()
            with self.assertRaises(FileNotFoundError):
                load_training_checkpoint(ckpt_path, h0, o0, None, torch.device("cpu"), sampler=s0, aug_rng=a0)

            # Uninterrupted 2-epoch run
            h_full, o_full, s_full, a_full = _make_state()
            _run_one_epoch(h_full, o_full, s_full, a_full, crit, 0.05)
            ep2_losses_full = _run_one_epoch(h_full, o_full, s_full, a_full, crit, 0.12)

            # Split run: epoch 1 -> save -> load -> epoch 2
            h_part, o_part, s_part, a_part = _make_state()
            ep1_losses = _run_one_epoch(h_part, o_part, s_part, a_part, crit, 0.05)
            save_training_checkpoint(
                ckpt_path, h_part, o_part, None, epoch=1, tau_t=0.12,
                history=[{"epoch": 1, "loss": ep1_losses[-1]}],
                training_config=cfg, sampler=s_part, aug_rng=a_part,
            )

            h_res, o_res, s_res, a_res = _make_state()
            # Corrupt weights deliberately before loading to prove restore overrides them
            with torch.no_grad():
                h_res[2].weight.zero_()
            comp_ep, tau_res, _, _ = load_training_checkpoint(
                ckpt_path, h_res, o_res, None, torch.device("cpu"), sampler=s_res, aug_rng=a_res, expected_config=cfg
            )
            self.assertEqual(comp_ep, 1)
            self.assertEqual(tau_res, 0.12)
            ep2_losses_res = _run_one_epoch(h_res, o_res, s_res, a_res, crit, tau_res)

            self.assertEqual(ep2_losses_full, ep2_losses_res)
            for (n1, p1), (n2, p2) in zip(h_full.named_parameters(), h_res.named_parameters()):
                self.assertTrue(torch.equal(p1, p2), f"Resumed parameter {n1} diverged")

            # Mismatched expected config raises ValueError
            with self.assertRaises(ValueError):
                load_training_checkpoint(
                    ckpt_path, h_res, o_res, None, torch.device("cpu"), expected_config={"baseline": "ce"}
                )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAnchorShiftPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("\nALL UNIT TESTS PASSED LOCALLY!")
