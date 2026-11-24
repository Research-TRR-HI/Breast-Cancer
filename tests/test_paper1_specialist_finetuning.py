"""CPU Unit Tests for Paper 1 Supplementary Specialist Fine-Tuning (GMIC & NYU Four-View)."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.rsna_dataset import BalancedBatchSampler
from src.paper1.gmic_finetune import (
    compute_gmic_rsna_loss,
    configure_gmic_trainable_params,
    count_trainable_parameters,
    gmic_amp_train_step,
    load_gmic_finetune_checkpoint,
    save_gmic_finetune_checkpoint,
    validate_gmic_finetune_cli_guard,
)
from src.paper1.nyu_finetune import (
    compute_nyu_rsna_loss,
    configure_nyu_trainable_params,
    load_nyu_finetune_checkpoint,
    save_nyu_finetune_checkpoint,
    select_rsna_4view_split_exams,
    validate_nyu_finetune_cli_guard,
)


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
        self.fc1_cc = nn.Linear(8, 4)
        self.fc1_mlo = nn.Linear(8, 4)
        self.output_layer_cc = nn.Linear(4, 8)
        self.output_layer_mlo = nn.Linear(4, 8)

    def forward(self, x_dict: dict) -> dict:
        def _branch(v1, v2, net, fc, out_l):
            h1 = net.final_bn(net.layer_list[-1](net.first_conv(v1))).mean(dim=(2, 3))
            h2 = net.final_bn(net.layer_list[-1](net.first_conv(v2))).mean(dim=(2, 3))
            z = F.relu(fc(torch.cat([h1, h2], dim=1)))
            return F.log_softmax(out_l(z).view(-1, 4, 2), dim=-1)

        return {
            "CC": _branch(x_dict["L-CC"], x_dict["R-CC"], self.four_view_resnet.cc, self.fc1_cc, self.output_layer_cc),
            "MLO": _branch(x_dict["L-MLO"], x_dict["R-MLO"], self.four_view_resnet.mlo, self.fc1_mlo, self.output_layer_mlo),
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
            validate_nyu_finetune_cli_guard("train", approve_full_train=False, approve_eval=False, max_spend_usd=5.0)
        validate_gmic_finetune_cli_guard("pilot", approve_full_train=False, approve_eval=False, max_spend_usd=5.0)
        validate_nyu_finetune_cli_guard("pilot", approve_full_train=False, approve_eval=False, max_spend_usd=5.0)


if __name__ == "__main__":
    unittest.main()
