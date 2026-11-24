"""CPU Unit Tests for Paper 1 Supplementary Specialist Fine-Tuning (GMIC & NYU Four-View)."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.rsna_dataset import BalancedBatchSampler
from src.evaluation.stage_c_eval import compute_sha256_file
from src.paper1.gmic_adapter import (
    GMIC_CKPT_SHA256,
    compute_item_ids_sha256,
    finalize_and_lock_gmic_predictions,
    validate_gmic_shard_payload,
)
from src.paper1.gmic_finetune import (
    GMIC_FINETUNED_MODEL_ID,
    apply_linear_warmup,
    compute_gmic_rsna_loss,
    compute_gmic_schedule_stats,
    configure_gmic_precision_and_scaler,
    configure_gmic_trainable_params,
    count_trainable_parameters,
    create_fused_adamw,
    gmic_amp_train_step,
    load_gmic_finetune_checkpoint,
    load_gmic_finetuned_model_for_eval,
    run_progressive_gmic_benchmark_loop,
    save_gmic_finetune_checkpoint,
    scale_steps_and_lr_for_batch_size,
    set_gmic_backbone_eval_mode,
    validate_gmic_eval_cli_args,
    validate_gmic_finetune_cli_guard,
)
from src.paper1.nyu_finetune import (
    NYU_FINETUNED_MODEL_ID,
    compute_nyu_rsna_loss,
    configure_nyu_trainable_params,
    extract_nyu_backbone_features,
    forward_nyu_head_from_features,
    load_nyu_finetune_checkpoint,
    load_nyu_finetuned_model_for_eval,
    run_progressive_nyu_batch_benchmark,
    save_nyu_finetune_checkpoint,
    score_nyu_head_from_tta_features,
    select_rsna_4view_split_exams,
    set_nyu_backbone_eval_mode,
    train_nyu_head_from_cached_features,
    validate_nyu_eval_cli_args,
    validate_nyu_finetune_cli_guard,
    verify_nyu_multidraw_feature_equivalence,
)
from src.paper1.nyu_fourview_adapter import (
    NYU_CKPT_SHA256,
    NYU_VIEWS,
    compute_nyu_case_ids_sha256,
    finalize_and_lock_nyu_predictions,
    validate_nyu_shard_payload,
)
from src.paper1.specialist_eval import (
    calibrate_finetuned_rsna_thresholds,
    evaluate_finetuned_nlbs_against_labels,
    lock_finetuned_nlbs_predictions,
    verify_and_load_finetuned_locked_predictions,
)
from src.utils.modal_transfer import get_bundle_specs


class _DummyGMIC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ds_net = nn.Module()
        self.ds_net.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.ds_net.layer_list = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(5)])
        self.ds_net.final_bn = nn.BatchNorm2d(4)
        self.left_postprocess_net = nn.Conv2d(4, 2, 1)
        self.dn_resnet = nn.Module()
        self.dn_resnet.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.dn_resnet.layer4 = nn.Conv2d(4, 4, 1)
        self.mil_attn_V = nn.Linear(4, 2, bias=False)
        self.mil_attn_U = nn.Linear(4, 2, bias=False)
        self.mil_attn_w = nn.Linear(2, 1, bias=False)
        self.classifier_linear = nn.Linear(4, 2, bias=False)
        self.fusion_dnn = nn.Linear(8, 2)
        self.saliency_map = None
        self.y_global = None
        self.y_local = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ds_net.conv1(x)
        for blk in self.ds_net.layer_list:
            h = F.relu(blk(h))
        h = self.ds_net.final_bn(h)
        self.saliency_map = torch.sigmoid(self.left_postprocess_net(h))
        self.y_global = self.saliency_map.mean(dim=(2, 3))
        hc = F.relu(self.dn_resnet.layer4(self.dn_resnet.conv1(x))).mean(dim=(2, 3))
        attn_p = torch.sigmoid(self.mil_attn_U(hc)) * torch.tanh(self.mil_attn_V(hc))
        _ = self.mil_attn_w(attn_p)
        self.y_local = torch.sigmoid(self.classifier_linear(hc))
        g_vec = h.mean(dim=(2, 3))
        return torch.sigmoid(self.fusion_dnn(torch.cat([g_vec, hc], dim=1)))


class _DummyOutputLayer(nn.Module):
    def __init__(self, in_features: int, output_shape: tuple[int, int] = (4, 2)) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.fc_layer = nn.Linear(in_features, output_shape[0] * output_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.fc_layer(x).view(x.shape[0], *self.output_shape), dim=-1)


class _DummyAllViewsGaussianNoise(nn.Module):
    def __init__(self, gaussian_noise_std: float = 0.01) -> None:
        super().__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x_dict: dict) -> dict:
        if self.training and self.gaussian_noise_std > 0:
            return {k: v + torch.randn_like(v) * self.gaussian_noise_std for k, v in x_dict.items()}
        return x_dict


class _DummyNYU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.four_view_resnet = nn.Module()
        self.four_view_resnet.cc = nn.Module()
        self.four_view_resnet.cc.first_conv = nn.Conv2d(1, 4, 3, padding=1)
        self.four_view_resnet.cc.layer_list = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(5)])
        self.four_view_resnet.cc.final_bn = nn.BatchNorm2d(4)
        self.four_view_resnet.mlo = nn.Module()
        self.four_view_resnet.mlo.first_conv = nn.Conv2d(1, 4, 3, padding=1)
        self.four_view_resnet.mlo.layer_list = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(5)])
        self.four_view_resnet.mlo.final_bn = nn.BatchNorm2d(4)
        self.four_view_resnet.forward = self._forward_fv
        self.all_views_gaussian_noise_layer = _DummyAllViewsGaussianNoise(0.01)
        self.all_views_avg_pool = self._pool_views
        self.fc1_cc = nn.Linear(8, 4)
        self.fc1_mlo = nn.Linear(8, 4)
        self.output_layer_cc = _DummyOutputLayer(4, (4, 2))
        self.output_layer_mlo = _DummyOutputLayer(4, (4, 2))

    def _pool_views(self, x_dict: dict) -> dict:
        return {k: v.mean(dim=(2, 3)) for k, v in x_dict.items()}

    def _forward_fv(self, x_dict: dict) -> dict:
        return {
            "L-CC": self.four_view_resnet.cc.final_bn(self.four_view_resnet.cc.layer_list[-1](self.four_view_resnet.cc.first_conv(x_dict["L-CC"]))),
            "R-CC": self.four_view_resnet.cc.final_bn(self.four_view_resnet.cc.layer_list[-1](self.four_view_resnet.cc.first_conv(x_dict["R-CC"]))),
            "L-MLO": self.four_view_resnet.mlo.final_bn(self.four_view_resnet.mlo.layer_list[-1](self.four_view_resnet.mlo.first_conv(x_dict["L-MLO"]))),
            "R-MLO": self.four_view_resnet.mlo.final_bn(self.four_view_resnet.mlo.layer_list[-1](self.four_view_resnet.mlo.first_conv(x_dict["R-MLO"]))),
        }

    def forward(self, x_dict: dict) -> dict:
        x_clean = self.all_views_gaussian_noise_layer(x_dict)
        fv = self._forward_fv(x_clean)
        h = self._pool_views(fv)
        h_cc = torch.cat([h["L-CC"], h["R-CC"]], dim=1)
        h_mlo = torch.cat([h["L-MLO"], h["R-MLO"]], dim=1)
        return {
            "CC": self.output_layer_cc(F.relu(self.fc1_cc(h_cc))),
            "MLO": self.output_layer_mlo(F.relu(self.fc1_mlo(h_mlo))),
        }


class TestSpecialistFinetuning(unittest.TestCase):
    def test_gmic_malignant_only_loss_ignores_benign_channel(self) -> None:
        m = _DummyGMIC()
        x = torch.randn(4, 1, 16, 16)
        y_fus = m(x)
        # Modify benign channel arbitrarily; malignant-only loss must be identical
        y_fus_mod = y_fus.clone()
        y_fus_mod[:, 0] = 0.99
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
        l1, _ = compute_gmic_rsna_loss(m, y_fus, targets)
        l2, _ = compute_gmic_rsna_loss(m, y_fus_mod, targets)
        self.assertAlmostEqual(float(l1.item()), float(l2.item()), places=6)

    def test_gmic_strategies_and_checkpoint_resume(self) -> None:
        m = _DummyGMIC()
        c_head = count_trainable_parameters(m) if False else None
        _ = configure_gmic_trainable_params(m, strategy="head_only", base_lr=1e-4)
        c_head = count_trainable_parameters(m)["trainable"]
        _ = configure_gmic_trainable_params(m, strategy="partial", base_lr=1e-4)
        c_part = count_trainable_parameters(m)["trainable"]
        groups_full = configure_gmic_trainable_params(m, strategy="full", base_lr=1e-4)
        c_full = count_trainable_parameters(m)["trainable"]
        self.assertLess(c_head, c_part)
        self.assertLess(c_part, c_full)

        opt = torch.optim.AdamW(groups_full)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        sampler = BalancedBatchSampler(np.array([0, 0, 1, 0, 1, 0]), batch_size=2, n_pos=1, seed=42, steps_per_epoch=2)
        aug_rng = np.random.default_rng(142)

        ld, gn = gmic_amp_train_step(
            m, torch.randn(2, 1, 16, 16), torch.tensor([0.0, 1.0]), opt, scaler, amp_enabled=False
        )
        self.assertTrue(np.isfinite(ld["loss_total"]) and np.isfinite(gn))

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "gmic.pt"
            save_gmic_finetune_checkpoint(
                p, m, opt, sched, scaler,
                epoch=1, best_auroc=0.81, best_epoch=1,
                history=[{"epoch": 1}], sampler=sampler, aug_rng=aug_rng, strategy="full", seed=42,
            )
            ep, ba, be, hist = load_gmic_finetune_checkpoint(
                p, m, opt, sched, scaler, sampler, aug_rng, torch.device("cpu")
            )
            self.assertEqual((ep, be, len(hist)), (1, 1, 1))
            self.assertAlmostEqual(ba, 0.81, places=6)

    def test_nyu_malignant_only_loss_ignores_benign_indices(self) -> None:
        m = _DummyNYU()
        x = {v: torch.randn(3, 1, 8, 8) for v in ("L-CC", "R-CC", "L-MLO", "R-MLO")}
        out = m(x)
        out_mod = {"CC": out["CC"].clone(), "MLO": out["MLO"].clone()}
        out_mod["CC"][:, 0:2, :] = -10.0
        out_mod["MLO"][:, 0:2, :] = -10.0
        lm = torch.tensor([0, 1, 0])
        rm = torch.tensor([1, 0, 0])
        l1, _ = compute_nyu_rsna_loss(out, lm, rm)
        l2, _ = compute_nyu_rsna_loss(out_mod, lm, rm)
        self.assertAlmostEqual(float(l1.item()), float(l2.item()), places=6)

    def test_nyu_4view_selection_and_checkpoint_resume(self) -> None:
        splits = pd.read_csv("datasets/rsna/splits_seed42.csv")
        img_df, ex_df = select_rsna_4view_split_exams(splits, "train_calib")
        self.assertEqual(len(ex_df), 1192)
        self.assertEqual(len(img_df), 1192 * 4)
        self.assertEqual(int(ex_df["exam_cancer"].sum()), 49)

        m = _DummyNYU()
        groups = configure_nyu_trainable_params(m, strategy="head_only", base_lr=1e-4)
        opt = torch.optim.AdamW(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
        sampler = BalancedBatchSampler(np.array([0, 1, 0, 1]), batch_size=2, n_pos=1, seed=42, steps_per_epoch=1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nyu.pt"
            save_nyu_finetune_checkpoint(
                p, m, opt, sched,
                epoch=2, best_auroc=0.79, best_epoch=2,
                history=[{"epoch": 1}, {"epoch": 2}], sampler=sampler, aug_seed_base=42, strategy="head_only", seed=42,
            )
            ep, ba, be, hist = load_nyu_finetune_checkpoint(p, m, opt, sched, sampler, torch.device("cpu"))
            self.assertEqual((ep, be, len(hist)), (2, 2, 2))
            self.assertAlmostEqual(ba, 0.79, places=6)

    def test_cli_approval_guards(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_gmic_finetune_cli_guard("train", approve_full_train=False, approve_eval=False, max_spend_usd=5.0)
        with self.assertRaises(RuntimeError):
            validate_gmic_finetune_cli_guard("eval", approve_full_train=False, approve_eval=False, max_spend_usd=5.0)
        with self.assertRaises(RuntimeError):
            validate_nyu_finetune_cli_guard("train", approve_full_train=False, approve_eval=False, max_spend_usd=5.0)
        with self.assertRaises(RuntimeError):
            validate_nyu_finetune_cli_guard("eval", approve_full_train=False, approve_eval=False, max_spend_usd=5.0)
        validate_gmic_finetune_cli_guard("pilot", approve_full_train=False, approve_eval=False, max_spend_usd=5.0)
        validate_nyu_finetune_cli_guard("pilot", approve_full_train=False, approve_eval=False, max_spend_usd=5.0)
        validate_gmic_eval_cli_args(
            split="rsna_val_calib", batch_size=1, num_shards=4,
            checkpoint_path="/tmp/ckpt.pt", approve_eval=True, max_spend_usd=2.0,
        )
        validate_nyu_eval_cli_args(
            split="rsna_val_calib", batch_exams=4, num_shards=4, num_tta_epochs=10,
            checkpoint_path="/tmp/ckpt.pt", approve_eval=True, max_spend_usd=2.0,
        )

    def test_rsna_train_and_val_calib_4view_selection_parity(self) -> None:
        splits = pd.read_csv("datasets/rsna/splits_seed42.csv")
        tr_img, tr_ex = select_rsna_4view_split_exams(splits, "train")
        self.assertEqual(len(tr_ex), 8339)
        self.assertEqual(len(tr_img), 8339 * 4)
        self.assertEqual(int(tr_ex["exam_cancer"].sum()), 340)

        vc_img, _ = select_rsna_4view_split_exams(splits, "val_calib")
        frozen = pd.read_csv("datasets/paper1/rsna_val_calib_4view_selection.csv")
        a = vc_img[["patient_id", "image_id", "view_slot"]].astype(str).sort_values(["patient_id", "view_slot"]).reset_index(drop=True)
        b = frozen[["patient_id", "image_id", "view_slot"]].astype(str).sort_values(["patient_id", "view_slot"]).reset_index(drop=True)
        self.assertTrue(a.equals(b))

    def test_eval_checkpoint_sha256_zero_grad_and_no_weight_mutation(self) -> None:
        # Pretrained hashes must remain intact
        self.assertEqual(GMIC_CKPT_SHA256, "3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991")
        self.assertEqual(NYU_CKPT_SHA256, "9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014")

        with tempfile.TemporaryDirectory() as tmp:
            # GMIC fine-tuned eval loader
            mg = _DummyGMIC()
            groups_g = configure_gmic_trainable_params(mg, strategy="partial", base_lr=1e-4)
            opt_g = torch.optim.AdamW(groups_g)
            sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=2)
            scaler_g = torch.amp.GradScaler("cuda", enabled=False)
            smp_g = BalancedBatchSampler(np.array([0, 1, 0, 1]), batch_size=2, n_pos=1, seed=42, steps_per_epoch=1)
            p_g = Path(tmp) / "gmic_ft.pt"
            save_gmic_finetune_checkpoint(
                p_g, mg, opt_g, sched_g, scaler_g,
                epoch=2, best_auroc=0.85, best_epoch=2, history=[], sampler=smp_g,
                aug_rng=np.random.default_rng(1), strategy="partial", seed=42,
            )
            sha_g = compute_sha256_file(p_g)
            mg_eval = _DummyGMIC()
            loaded_g, _, meta_g = load_gmic_finetuned_model_for_eval(
                p_g, expected_sha256=sha_g, device=torch.device("cpu"), model_override=mg_eval
            )
            self.assertFalse(loaded_g.training)
            self.assertTrue(all(not p.requires_grad for p in loaded_g.parameters()))
            self.assertEqual(meta_g["checkpoint_sha256"], sha_g)
            w_before = [p.detach().clone() for p in loaded_g.parameters()]
            with torch.no_grad():
                _ = loaded_g(torch.randn(2, 1, 16, 16))
            for b, a in zip(w_before, loaded_g.parameters()):
                self.assertTrue(torch.equal(b, a))
            with self.assertRaises(RuntimeError):
                load_gmic_finetuned_model_for_eval(
                    p_g, expected_sha256="0" * 64, device=torch.device("cpu"), model_override=_DummyGMIC()
                )

            # NYU fine-tuned eval loader
            mn = _DummyNYU()
            groups_n = configure_nyu_trainable_params(mn, strategy="head_only", base_lr=1e-4)
            opt_n = torch.optim.AdamW(groups_n)
            sched_n = torch.optim.lr_scheduler.CosineAnnealingLR(opt_n, T_max=2)
            smp_n = BalancedBatchSampler(np.array([0, 1, 0, 1]), batch_size=2, n_pos=1, seed=42, steps_per_epoch=1)
            p_n = Path(tmp) / "nyu_ft.pt"
            save_nyu_finetune_checkpoint(
                p_n, mn, opt_n, sched_n,
                epoch=3, best_auroc=0.80, best_epoch=3, history=[], sampler=smp_n,
                aug_seed_base=42, strategy="head_only", seed=42,
            )
            sha_n = compute_sha256_file(p_n)
            mn_eval = _DummyNYU()
            loaded_n, _, meta_n = load_nyu_finetuned_model_for_eval(
                p_n, expected_sha256=sha_n, device=torch.device("cpu"), model_override=mn_eval
            )
            self.assertFalse(loaded_n.training)
            self.assertTrue(all(not p.requires_grad for p in loaded_n.parameters()))
            self.assertEqual(meta_n["checkpoint_sha256"], sha_n)

    def test_nyu_frozen_backbone_bn_and_feature_cache_parity(self) -> None:
        torch.manual_seed(0)
        mn = _DummyNYU()
        configure_nyu_trainable_params(mn, strategy="head_only", base_lr=1e-4)
        set_nyu_backbone_eval_mode(mn, "head_only")
        self.assertFalse(mn.four_view_resnet.cc.final_bn.training)
        self.assertFalse(mn.four_view_resnet.mlo.final_bn.training)
        rm_before = mn.four_view_resnet.cc.final_bn.running_mean.clone()

        x = {v: torch.randn(3, 1, 8, 8) for v in NYU_VIEWS}
        h_cc, h_mlo = extract_nyu_backbone_features(mn, x)
        out_cached = forward_nyu_head_from_features(mn, h_cc, h_mlo)
        mn.eval()
        with torch.no_grad():
            out_full = mn(x)
        self.assertLess(float(torch.max(torch.abs(out_cached["CC"] - out_full["CC"]))), 1e-6)
        self.assertLess(float(torch.max(torch.abs(out_cached["MLO"] - out_full["MLO"]))), 1e-6)
        self.assertTrue(torch.equal(rm_before, mn.four_view_resnet.cc.final_bn.running_mean))

        # Test cached head trainer produces a valid eval-loadable checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            res = train_nyu_head_from_cached_features(
                model=_DummyNYU(),
                train_h_cc=torch.randn(8, 2, 8),
                train_h_mlo=torch.randn(8, 2, 8),
                train_lm=np.array([0, 1, 0, 1, 0, 0, 1, 0]),
                train_rm=np.array([0, 0, 1, 0, 0, 0, 0, 1]),
                train_exam_cancer=np.array([0, 1, 1, 1, 0, 0, 1, 1]),
                tc_h_cc=torch.randn(4, 3, 8),
                tc_h_mlo=torch.randn(4, 3, 8),
                tc_exam_cancer=np.array([0, 1, 0, 1]),
                out_dir=Path(tmp),
                epochs=2,
                batch_exams=4,
                steps_per_epoch=2,
            )
            _, _, meta = load_nyu_finetuned_model_for_eval(
                Path(res["best_checkpoint_path"]),
                expected_sha256=res["best_checkpoint_sha256"],
                device=torch.device("cpu"),
                model_override=_DummyNYU(),
            )
            self.assertEqual(meta["model_id"], NYU_FINETUNED_MODEL_ID)

        # Also test TTA score helper shape/finiteness
        tta_scores = score_nyu_head_from_tta_features(mn, torch.randn(2, 5, 8), torch.randn(2, 5, 8))
        self.assertEqual(len(tta_scores), 2)
        for row in tta_scores:
            self.assertAlmostEqual(row["exam_score"], max(row["left_malignant"], row["right_malignant"]), places=10)

    def test_resumable_shards_and_end_to_end_rsna_calib_lock_and_tamper_rejection(self) -> None:
        ckpt_sha = "a" * 64
        # 1. GMIC fine-tuned shard provenance validation
        items = ["i1", "i2"]
        g_payload = {
            "model_id": GMIC_FINETUNED_MODEL_ID,
            "checkpoint_sha256": ckpt_sha,
            "preprocess_version": "paper1_gmic_uint16_v1",
            "preprocess_manifest_sha256": "m1",
            "split_name": "rsna_val_calib",
            "shard_idx": 0,
            "num_shards": 1,
            "batch_size": 1,
            "expected_item_ids_sha256": compute_item_ids_sha256(items),
            "records": [
                {"item_id": "i1", "case_id": "p1", "view_slot": "L-CC", "benign_pred": 0.2, "malignant_pred": 0.1, "crop_fallback": False},
                {"item_id": "i2", "case_id": "p2", "view_slot": "R-CC", "benign_pred": 0.3, "malignant_pred": 0.8, "crop_fallback": False},
            ],
        }
        ok_g, r_g = validate_gmic_shard_payload(
            g_payload, expected_item_ids=items, batch_size=1, split_name="rsna_val_calib",
            shard_idx=0, num_shards=1, model_id=GMIC_FINETUNED_MODEL_ID,
            checkpoint_sha256=ckpt_sha, preprocess_manifest_sha256="m1",
        )
        self.assertTrue(ok_g, r_g)

        # 2. NYU fine-tuned shard provenance validation
        cases = ["c1", "c2"]
        n_payload = {
            "schema_version": "paper1_nyu_shard_v2",
            "model_id": NYU_FINETUNED_MODEL_ID,
            "checkpoint_sha256": ckpt_sha,
            "preprocess_version": "paper1_nyu_uint16_v1",
            "preprocess_manifest_sha256": "m2",
            "precision": "fp32",
            "split_name": "nlbs_4view",
            "shard_idx": 0,
            "num_shards": 1,
            "batch_exams": 4,
            "num_tta_epochs": 10,
            "tta_seed": 0,
            "expected_case_ids_sha256": compute_nyu_case_ids_sha256(cases),
            "records": [
                {"case_id": c, "n_views": 4, "left_benign": 0.1, "right_benign": 0.1, "left_malignant": 0.2, "right_malignant": 0.4, "exam_score": 0.4}
                for c in cases
            ],
        }
        ok_n, r_n = validate_nyu_shard_payload(
            n_payload, expected_case_ids=cases, batch_exams=4, split_name="nlbs_4view",
            shard_idx=0, num_shards=1, model_id=NYU_FINETUNED_MODEL_ID,
            checkpoint_sha256=ckpt_sha, preprocess_manifest_sha256="m2",
        )
        self.assertTrue(ok_n, r_n)

        # 3. End-to-end RSNA-only calibration -> NLBS pre-join lock -> tamper check -> NLBS eval
        splits = pd.read_csv("datasets/rsna/splits_seed42.csv")
        vc = splits[splits["split"] == "val_calib"].copy()
        vc_cases = vc.groupby("patient_id")["cancer"].max().reset_index()
        vc_cases["patient_id"] = vc_cases["patient_id"].astype(str)
        c4v = pd.read_csv("datasets/paper1/nlbs_4view_common_cohort_label_free.csv", dtype=str)
        nlbs_uids = sorted(c4v["StudyInstanceUID"].unique().tolist())

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            pred_dir = td / "preds"
            out_dir = td / "eval_out"
            pred_dir.mkdir()
            # Create synthetic locked RSNA predictions
            rng = np.random.default_rng(7)
            rsna_scores = np.where(vc_cases["cancer"].values == 1, 0.75 + rng.uniform(0, 0.2, len(vc_cases)), 0.05 + rng.uniform(0, 0.2, len(vc_cases)))
            rsna_case_df = pd.DataFrame({"patient_id": vc_cases["patient_id"], "score": rsna_scores, "n_views": 4, "n_crop_fallbacks": 0, "has_crop_fallback": False})
            rsna_csv = pred_dir / "nyu_rsna_val_calib_case_scores.csv"
            rsna_case_df.to_csv(rsna_csv, index=False)
            (pred_dir / "nyu_rsna_val_calib_provenance_lock.json").write_text(
                json.dumps({
                    "model_id": NYU_FINETUNED_MODEL_ID,
                    "checkpoint_file": "nyu_ft.pt",
                    "checkpoint_sha256": ckpt_sha,
                    "preprocess_version": "paper1_nyu_uint16_v1",
                    "artifacts": {"case_scores_csv": {"path": str(rsna_csv), "sha256": compute_sha256_file(rsna_csv)}},
                }),
                encoding="utf-8",
            )
            calib_lock = calibrate_finetuned_rsna_thresholds(
                model_id=NYU_FINETUNED_MODEL_ID,
                prefix="nyu",
                locked_pred_dir=pred_dir,
                rsna_splits_csv=Path("datasets/rsna/splits_seed42.csv"),
                rsna_fold_csv=Path("datasets/paper1/rsna_val_calib_2fold_seed42.csv"),
                out_dir=out_dir,
                expected_checkpoint_sha256=ckpt_sha,
                n_boot=20,
            )
            nlbs_raw_csv = pred_dir / "nyu_nlbs_4view_case_scores.csv"
            pd.DataFrame({"StudyInstanceUID": nlbs_uids, "score": rng.uniform(0.01, 0.9, len(nlbs_uids)), "n_crop_fallbacks": 0}).to_csv(nlbs_raw_csv, index=False)
            calib_lock_p = out_dir / f"{NYU_FINETUNED_MODEL_ID}_rsna_calibration_lock.json"
            lock_finetuned_nlbs_predictions(
                model_id=NYU_FINETUNED_MODEL_ID,
                nlbs_case_scores_csv=nlbs_raw_csv,
                calib_lock_path=calib_lock_p,
                out_dir=pred_dir,
            )
            ev = evaluate_finetuned_nlbs_against_labels(
                model_id=NYU_FINETUNED_MODEL_ID,
                calib_lock_path=calib_lock_p,
                nlbs_lock_dir=pred_dir,
                exam_manifest_csv=Path("datasets/nlbs/exam_manifest.csv"),
                common_4view_csv=Path("datasets/paper1/nlbs_4view_common_cohort_label_free.csv"),
                out_dir=out_dir,
                n_boot=20,
            )
            self.assertEqual(ev["nlbs_common_4view_primary"]["n_units"], 6184)

            # Tamper locked CSV and verify rejection before join
            locked_csv = pred_dir / f"{NYU_FINETUNED_MODEL_ID}_locked_nlbs_predictions.csv"
            tampered_df = pd.read_csv(locked_csv)
            tampered_df.loc[0, "dismissed_98"] = not bool(tampered_df.loc[0, "dismissed_98"])
            tampered_df.to_csv(locked_csv, index=False)
            with self.assertRaises(RuntimeError):
                verify_and_load_finetuned_locked_predictions(pred_dir, NYU_FINETUNED_MODEL_ID)

    def test_gmic_schedule_stats_and_eval_transfer_bundles(self) -> None:
        st = compute_gmic_schedule_stats(807, 37444, batch_size=8, steps_per_epoch=600, epochs=10)
        self.assertAlmostEqual(st["expected_unique_neg_frac_1ep"], 0.09169, places=3)
        self.assertAlmostEqual(st["expected_unique_neg_frac_total"], 0.6177, places=3)
        specs = get_bundle_specs("dryousuf_gmic_eval_cache")
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].src_profile, "hasinishrak2015")
        self.assertEqual(specs[0].src_prefix, "cache/paper1_gmic_uint16_v1/rsna_val_calib")
        self.assertEqual(specs[1].src_prefix, "cache/paper1_gmic_uint16_v1/nlbs")

    def test_gmic_fair_scaling_precision_and_warmup(self) -> None:
        p32 = scale_steps_and_lr_for_batch_size(32, base_batch_size=8, base_steps_per_epoch=600, base_lr=1e-4, lr_scaling="sqrt", base_warmup_steps=100)
        self.assertEqual(p32["steps_per_epoch"], 150)
        self.assertEqual(p32["actual_draws_per_epoch"], 4800)
        self.assertAlmostEqual(p32["scaled_lr"], 2e-4, places=9)
        self.assertEqual(p32["warmup_steps"], 25)

        p16_lin = scale_steps_and_lr_for_batch_size(16, base_batch_size=8, base_steps_per_epoch=600, base_lr=1e-4, lr_scaling="linear", base_warmup_steps=100)
        self.assertEqual(p16_lin["steps_per_epoch"], 300)
        self.assertAlmostEqual(p16_lin["scaled_lr"], 2e-4, places=9)

        for prec, expected_dt in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("tf32", torch.float32), ("fp32", torch.float32)):
            cfg = configure_gmic_precision_and_scaler(prec, torch.device("cpu"))
            self.assertEqual(cfg["amp_dtype"], expected_dt)
            self.assertFalse(cfg["amp_enabled"])

        m = _DummyGMIC()
        groups = configure_gmic_trainable_params(m, strategy="partial", base_lr=2e-4)
        opt, used_fused = create_fused_adamw(groups, device=torch.device("cpu"))
        self.assertFalse(used_fused)
        base_lrs = [pg["lr"] for pg in opt.param_groups]
        apply_linear_warmup(opt, 0, 4, base_lrs)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.25 * base_lrs[0], places=9)
        apply_linear_warmup(opt, 4, 4, base_lrs)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], base_lrs[0], places=9)

    def test_gmic_progressive_benchmark_and_oom_stop(self) -> None:
        m = _DummyGMIC()
        rng = np.random.default_rng(12)
        attempted: list[tuple[str, int]] = []

        def _loader(bs: int, step: int) -> tuple[np.ndarray, np.ndarray]:
            attempted.append(("train", bs))
            if bs >= 8:
                raise RuntimeError("CUDA out of memory. Simulated OOM at bs=8")
            x = rng.normal(size=(bs, 1, 16, 16)).astype(np.float32)
            y = np.asarray([i % 2 for i in range(bs)], dtype=np.float32)
            return x, y

        def _vloader(bs: int, step: int) -> np.ndarray:
            return rng.normal(size=(bs, 1, 16, 16)).astype(np.float32)

        res = run_progressive_gmic_benchmark_loop(
            m,
            _loader,
            torch.device("cpu"),
            batch_sizes=(2, 4, 8, 16),
            precisions=("bf16", "fp16"),
            warmup_steps=1,
            bench_steps=2,
            val_loader=_vloader,
            val_steps=1,
            stop_on_throughput_drop=False,
        )
        rows = res["sweep_results"]
        self.assertTrue(any(r["precision"] == "bf16" and r["batch_size"] == 4 and r["status"] == "ok" for r in rows))
        self.assertTrue(any(r["precision"] == "bf16" and r["batch_size"] == 8 and r["status"] == "oom" for r in rows))
        # Must stop before attempting bs=16 after OOM at bs=8
        self.assertFalse(any(r["batch_size"] == 16 for r in rows))
        self.assertIsNotNone(res["recommended"])
        self.assertIn("val_imgs_per_sec", res["recommended"])

    def test_nyu_progressive_exam_benchmark_and_oom_stop(self) -> None:
        m = _DummyNYU()
        rng = np.random.default_rng(7)

        def _exam_loader(be: int, step: int) -> dict[str, np.ndarray]:
            if be >= 4:
                raise RuntimeError("CUDA out of memory. Simulated NYU OOM at be=4")
            return {v: rng.normal(size=(be, 1, 8, 8)).astype(np.float32) for v in NYU_VIEWS}

        res = run_progressive_nyu_batch_benchmark(
            m,
            _exam_loader,
            torch.device("cpu"),
            batch_sizes=(1, 2, 4, 8),
            mode="extract",
            warmup_steps=1,
            bench_steps=2,
            draws_per_exam=1,
            stop_on_throughput_drop=False,
        )
        rows = res["sweep_results"]
        self.assertEqual([r["batch_exams"] for r in rows], [1, 2, 4])
        self.assertEqual(rows[-1]["status"], "oom")
        self.assertIsNotNone(res["recommended"])

    def test_nyu_multidraw_feature_cache_equivalence(self) -> None:
        torch.manual_seed(3)
        mn = _DummyNYU()
        rng = np.random.default_rng(99)
        exam_draws = {v: rng.normal(size=(2 * 4, 1, 8, 8)).astype(np.float32) for v in NYU_VIEWS}
        eq = verify_nyu_multidraw_feature_equivalence(mn, exam_draws, num_draws=4, atol=1e-6)
        self.assertEqual(eq["n_exams"], 2)
        self.assertEqual(eq["num_draws"], 4)
        self.assertLess(eq["max_feature_abs_diff"], 1e-6)
        self.assertLess(eq["max_score_abs_diff_vs_full_eval"], 1e-6)
        self.assertLess(eq["max_score_abs_diff_vs_single_draw"], 1e-6)
        self.assertTrue(eq["batchnorm_running_stats_frozen"])
        self.assertEqual(eq["gaussian_noise_modules_training_flags"], [False])


if __name__ == "__main__":
    unittest.main()
