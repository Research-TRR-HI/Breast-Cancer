"""Paper 1 CPU Unit Tests: Gate 0.2 Splits/Quarantine & Four-Model Parity/Shapes."""

import os
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.stage_c_eval import assert_no_forbidden_columns
from src.paper1.dinov2_model import (
    DINOv2CancerHead,
    build_frozen_dinov2_vits14,
    prepare_dinov2_tensor_from_uint16,
)
from src.paper1.gmic_adapter import (
    GMIC_INPUT_SIZE,
    build_gmic_model,
    preprocess_uint16_for_nyukat,
    score_gmic_preprocessed_tensor,
)
from src.paper1.nyu_fourview_adapter import (
    NYU_VIEWS,
    build_nyu_fourview_model,
    score_nyu_fourview_exam,
)
from src.paper1.resnet50_model import (
    build_resnet50_classifier,
    prepare_resnet50_tensor_from_uint16,
)


class TestPaper1Preparation(unittest.TestCase):
    def test_1_gate02_splits_and_label_quarantine(self):
        base = pd.read_csv("datasets/rsna/splits_seed42.csv")
        folds = pd.read_csv("datasets/paper1/rsna_val_calib_2fold_seed42.csv")
        self.assertEqual(len(folds), 10982)
        self.assertEqual(folds["patient_id"].nunique(), 2382)

        pts_a = set(folds[folds["fold"] == "fold_a"]["patient_id"])
        pts_b = set(folds[folds["fold"] == "fold_b"]["patient_id"])
        pts_train = set(base[base["split"].isin(["train", "train_calib"])]["patient_id"])
        self.assertEqual(len(pts_a & pts_b), 0)
        self.assertEqual(len((pts_a | pts_b) & pts_train), 0)
        self.assertEqual(len(pts_a), 1191)
        self.assertEqual(len(pts_b), 1191)

        rsna_4v = pd.read_csv("datasets/paper1/rsna_val_calib_4view_selection.csv")
        self.assertEqual(len(rsna_4v), 2382 * 4)
        assert_no_forbidden_columns(rsna_4v.columns, "rsna_val_calib_4view_selection")

        nlbs_elig = pd.read_csv("datasets/paper1/nlbs_4view_study_eligibility.csv")
        nlbs_4v = pd.read_csv("datasets/paper1/nlbs_4view_common_cohort_label_free.csv")
        assert_no_forbidden_columns(nlbs_elig.columns, "nlbs_4view_study_eligibility")
        assert_no_forbidden_columns(nlbs_4v.columns, "nlbs_4view_common_cohort_label_free")
        self.assertEqual(len(nlbs_elig), 6359)
        self.assertEqual(int(nlbs_elig["is_4view_eligible"].sum()), 6184)
        self.assertEqual(len(nlbs_4v), 6184 * 4)

    def test_2_gmic_load_and_sample0_lcc_parity(self):
        ckpt = Path("external/GMIC/models/sample_model_1.p")
        if not ckpt.exists():
            self.skipTest("external/GMIC checkpoint not present")
        model, mods = build_gmic_model(ckpt, model_index="1", device=torch.device("cpu"))
        img = mods["loading"].read_image_png("external/GMIC/sample_data/images/0_L-CC.png")
        proc, meta = preprocess_uint16_for_nyukat(img, "L-CC", GMIC_INPUT_SIZE, mods)
        self.assertFalse(meta["crop_fallback"])
        self.assertEqual(proc.shape, GMIC_INPUT_SIZE)
        benign, malignant = score_gmic_preprocessed_tensor(model, proc, torch.device("cpu"))
        self.assertAlmostEqual(benign, 0.1356, places=3)
        self.assertAlmostEqual(malignant, 0.0081, places=3)

    def test_3_nyu_fourview_load_and_sample0_parity(self):
        ckpt = Path("external/breast_cancer_classifier/models/sample_image_model.p")
        if not ckpt.exists():
            self.skipTest("external/breast_cancer_classifier checkpoint not present")
        model, mods = build_nyu_fourview_model(ckpt, device=torch.device("cpu"))
        n_crop = mods["crop_mammogram"]
        n_load = mods["loading"]
        n_coc = mods["calc_optimal_centers"]
        from src.paper1.nyu_fourview_adapter import NYU_INPUT_SIZES

        view_crops = {}
        for v in NYU_VIEWS:
            img = n_load.read_image_png(f"external/breast_cancer_classifier/sample_data/images/0_{v.replace('-', '_')}.png")
            side = v.split("-")[0]
            cinfo = n_crop.crop_img_from_largest_connected(img, n_crop.image_orientation("NO", side), True, 100, 50, 1.0 / 3.0)
            t, b, l, r = cinfo[0]
            cu = n_load.flip_image(img[t:b, l:r], v, "NO")
            tlbr = (
                n_coc.get_bottomrightmost_pixel_constraint(cinfo[1][1], cinfo[2][0])
                if "MLO" in v
                else n_coc.get_rightmost_pixel_constraint(cinfo[1][1])
            )
            oc = n_coc.get_image_optimal_window_info(cu, com=np.array(cu.shape) // 2, window_dim=np.array(NYU_INPUT_SIZES[v]), tl_br_constraint=tlbr)
            view_crops[v] = {"image_f32": cu.astype(np.float32), "best_center": (int(oc["best_center_y"]), int(oc["best_center_x"]))}

        res = score_nyu_fourview_exam(model, mods, view_crops, torch.device("cpu"), num_tta_epochs=10, seed=0)
        self.assertAlmostEqual(res["left_malignant"], 0.0091, places=3)
        self.assertAlmostEqual(res["right_malignant"], 0.0179, places=3)
        self.assertAlmostEqual(res["exam_score"], max(res["left_malignant"], res["right_malignant"]), places=7)

    def test_4_dinov2_frozen_and_head_forward(self):
        dino_ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"
        if not dino_ckpt.exists():
            self.skipTest("DINOv2 local checkpoint not cached")
        enc = build_frozen_dinov2_vits14(checkpoint_path=dino_ckpt, device=torch.device("cpu"))
        self.assertTrue(all(not p.requires_grad for p in enc.parameters()))
        dummy_u16 = np.random.randint(0, 65535, size=(1520, 912), dtype=np.uint16)
        t = prepare_dinov2_tensor_from_uint16(dummy_u16)
        self.assertEqual(tuple(t.shape), (3, 518, 518))
        with torch.no_grad():
            emb = enc(t.unsqueeze(0))
        self.assertEqual(tuple(emb.shape), (1, 384))
        head = DINOv2CancerHead(embed_dim=384)
        logits = head(emb)
        self.assertEqual(tuple(logits.shape), (1, 1))

    def test_5_resnet50_forward_and_grad(self):
        res_ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth"
        if not res_ckpt.exists():
            self.skipTest("ResNet-50 local checkpoint not cached")
        model = build_resnet50_classifier(pretrained=False, local_weights_path=res_ckpt)
        dummy_u16 = np.random.randint(0, 65535, size=(1520, 912), dtype=np.uint16)
        t = prepare_resnet50_tensor_from_uint16(dummy_u16)
        self.assertEqual(tuple(t.shape), (3, 854, 512))
        out = model(t.unsqueeze(0))
        self.assertEqual(tuple(out.shape), (1, 1))
        out.sum().backward()
        self.assertIsNotNone(model.fc[1].weight.grad)

    def test_6_dinov2_head_cpu_training_and_locking(self):
        import tempfile
        import pandas as pd
        from scripts.train_paper1_dinov2_head import train_and_lock_dinov2_head
        from src.paper1.dinov2_model import DINOV2_VITS14_SHA256

        splits_df = pd.read_csv(ROOT / "datasets/rsna/splits_seed42.csv")
        fold_df = pd.read_csv(ROOT / "datasets/paper1/rsna_val_calib_2fold_seed42.csv")
        nlbs_df = pd.read_csv(ROOT / "datasets/nlbs/inference_manifest_stage_c.csv").head(40)

        # Sample small balanced subset per split (ensuring fold_a & fold_b both have pos & neg)
        tr_sub = pd.concat([
            splits_df[(splits_df["split"] == "train") & (splits_df["cancer"] == 1)].head(8),
            splits_df[(splits_df["split"] == "train") & (splits_df["cancer"] == 0)].head(16),
        ])
        tc_sub = pd.concat([
            splits_df[(splits_df["split"] == "train_calib") & (splits_df["cancer"] == 1)].head(4),
            splits_df[(splits_df["split"] == "train_calib") & (splits_df["cancer"] == 0)].head(8),
        ])
        fa_pids = set(fold_df[fold_df["fold"] == "fold_a"]["patient_id"].astype(str))
        fb_pids = set(fold_df[fold_df["fold"] == "fold_b"]["patient_id"].astype(str))
        vc_all = splits_df[splits_df["split"] == "val_calib"]
        vc_a = vc_all[vc_all["patient_id"].astype(str).isin(fa_pids)]
        vc_b = vc_all[vc_all["patient_id"].astype(str).isin(fb_pids)]
        vc_sub = pd.concat([
            vc_a[vc_a["cancer"] == 1].head(4),
            vc_a[vc_a["cancer"] == 0].head(6),
            vc_b[vc_b["cancer"] == 1].head(4),
            vc_b[vc_b["cancer"] == 0].head(6),
        ])
        sub = pd.concat([tr_sub, tc_sub, vc_sub], ignore_index=True)

        rng = np.random.default_rng(7)
        r_embs = torch.from_numpy(rng.normal(0, 0.5, size=(len(sub), 384)).astype(np.float32))
        r_embs[sub["cancer"].values == 1, :16] += 4.0
        n_embs = torch.from_numpy(rng.normal(0, 0.5, size=(len(nlbs_df), 384)).astype(np.float32))

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            r_pt = td_p / "rsna_emb.pt"
            n_pt = td_p / "nlbs_emb.pt"
            n_csv = td_p / "nlbs_manifest.csv"
            torch.save({"image_ids": sub["image_id"].astype(str).tolist(), "embeddings": r_embs, "backbone_sha256": DINOV2_VITS14_SHA256}, r_pt)
            torch.save({"image_ids": nlbs_df["image_uid"].astype(str).tolist(), "embeddings": n_embs, "backbone_sha256": DINOV2_VITS14_SHA256}, n_pt)
            nlbs_df.to_csv(n_csv, index=False)

            res = train_and_lock_dinov2_head(
                rsna_emb_pt=r_pt,
                nlbs_emb_pt=n_pt,
                splits_csv=ROOT / "datasets/rsna/splits_seed42.csv",
                val_calib_2fold_csv=ROOT / "datasets/paper1/rsna_val_calib_2fold_seed42.csv",
                nlbs_manifest_csv=n_csv,
                out_dir=td_p / "out",
                epochs=12,
                batch_size=8,
                lr=5e-3,
                n_boot=20,
            )
            self.assertGreater(res["rsna_val_calib_auroc"], 0.75)
            self.assertTrue((td_p / "out" / "locked_nlbs_predictions.csv").exists())
            self.assertEqual(res["locked_nlbs_studies"], int(nlbs_df["StudyInstanceUID"].nunique()))

    def test_7_resnet50_amp_unscale_clip_and_resume_parity(self):
        import random
        import tempfile
        import torch.nn as nn
        from torchvision.models import resnet18
        from src.data.rsna_dataset import BalancedBatchSampler
        from src.paper1.resnet50_model import (
            amp_train_step,
            load_resnet50_checkpoint,
            save_resnet50_checkpoint,
        )

        # Verify scaler.unscale_(optimizer) is called strictly BEFORE clip_grad_norm_
        call_order = []

        class _MockScaler:
            def scale(self, loss):
                call_order.append("scale")
                return loss

            def unscale_(self, opt):
                call_order.append("unscale")

            def step(self, opt):
                call_order.append("step")
                opt.step()

            def update(self):
                call_order.append("update")

        m_mock = resnet18(weights=None)
        m_mock.fc = nn.Linear(m_mock.fc.in_features, 1)
        opt_m = torch.optim.AdamW(m_mock.parameters(), lr=1e-3)
        orig_clip = torch.nn.utils.clip_grad_norm_

        def _spy_clip(*args, **kwargs):
            call_order.append("clip")
            return orig_clip(*args, **kwargs)

        torch.nn.utils.clip_grad_norm_ = _spy_clip
        try:
            loss_v, gn = amp_train_step(
                m_mock,
                torch.randn(4, 3, 64, 64),
                torch.ones(4, 1),
                nn.BCEWithLogitsLoss(),
                opt_m,
                _MockScaler(),
                max_grad_norm=5.0,
                amp_enabled=False,
            )
            self.assertEqual(call_order, ["scale", "unscale", "clip", "step", "update"])
            self.assertTrue(np.isfinite(loss_v) and np.isfinite(gn))
        finally:
            torch.nn.utils.clip_grad_norm_ = orig_clip

        # Verify bit-exact 2-epoch pause & resume vs uninterrupted 4-epoch run (incl. RNG & sampler)
        labels = np.array([1] * 8 + [0] * 16)

        def _run_sim(resume_at_ep2: bool):
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            mod = resnet18(weights=None)
            mod.fc = nn.Linear(mod.fc.in_features, 1)
            opt = torch.optim.AdamW(mod.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            crit = nn.BCEWithLogitsLoss()
            sampler = BalancedBatchSampler(labels, batch_size=8, n_pos=2, seed=42, steps_per_epoch=2)
            aug_rng = np.random.default_rng(142)

            with tempfile.TemporaryDirectory() as td:
                ckpt_p = Path(td) / "ckpt.pt"
                for ep in range(1, 5):
                    for idx in sampler:
                        _ = aug_rng.uniform(-1, 1)
                        bx = torch.randn(len(idx), 3, 32, 32)
                        by = torch.from_numpy(labels[idx].astype(np.float32)).unsqueeze(1)
                        amp_train_step(mod, bx, by, crit, opt, scaler, 5.0, amp_enabled=False)
                    sched.step()
                    if resume_at_ep2 and ep == 2:
                        save_resnet50_checkpoint(ckpt_p, mod, opt, sched, scaler, 2, 0.5, 2, [], sampler, aug_rng, 42)
                        # Fresh objects + perturbed RNG before restoring
                        torch.manual_seed(999)
                        np.random.seed(999)
                        mod = resnet18(weights=None)
                        mod.fc = nn.Linear(mod.fc.in_features, 1)
                        opt = torch.optim.AdamW(mod.parameters(), lr=1e-3)
                        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
                        scaler = torch.amp.GradScaler("cuda", enabled=False)
                        sampler = BalancedBatchSampler(labels, batch_size=8, n_pos=2, seed=42, steps_per_epoch=2)
                        aug_rng = np.random.default_rng(999)
                        load_resnet50_checkpoint(ckpt_p, mod, opt, sched, scaler, sampler, aug_rng, torch.device("cpu"))
                return {k: v.clone() for k, v in mod.state_dict().items()}

        sd_cont = _run_sim(resume_at_ep2=False)
        sd_res = _run_sim(resume_at_ep2=True)
        for k in sd_cont:
            self.assertTrue(torch.equal(sd_cont[k], sd_res[k]), f"Resume mismatch on parameter {k}")

    def test_8_six_model_eval_locking_and_paired_contrasts(self):
        import tempfile
        from src.paper1.six_model_eval import (
            ALL_SIX_MODELS,
            calibrate_rsna_thresholds_only,
            evaluate_six_models,
            lock_nlbs_model_predictions,
            verify_and_load_locked_model,
        )

        fold_df = pd.read_csv(ROOT / "datasets/paper1/rsna_val_calib_2fold_seed42.csv")
        rsna_pts = fold_df.groupby("patient_id", as_index=False)["cancer"].max()
        exam_df = pd.read_csv(ROOT / "datasets/nlbs/exam_manifest.csv").drop_duplicates("StudyInstanceUID").reset_index(drop=True)
        c4v_df = pd.read_csv(ROOT / "datasets/paper1/nlbs_4view_common_cohort_label_free.csv")
        c4v_uids = set(c4v_df["StudyInstanceUID"].astype(str).unique())

        rng = np.random.default_rng(123)
        rsna_cases = {}
        nlbs_preds = {}
        for idx, m in enumerate(ALL_SIX_MODELS):
            r_sc = np.clip(rng.uniform(0.05, 0.35, size=len(rsna_pts)) + 0.35 * rsna_pts["cancer"].values + 0.02 * idx, 0.0, 1.0)
            rsna_cases[m] = pd.DataFrame({"patient_id": rsna_pts["patient_id"], "cancer": rsna_pts["cancer"], "score": r_sc})

            sub_ex = exam_df[exam_df["StudyInstanceUID"].isin(c4v_uids)] if m == "nyu_fourview_pretrained" else exam_df
            # Negative score shift on NLBS positives -> tests calibration collapse & fallback sensitivity
            n_sc = np.clip(rng.uniform(0.05, 0.35, size=len(sub_ex)) + 0.25 * sub_ex["exam_cancer"].values, 0.0, 1.0)
            n_fb = np.zeros(len(sub_ex), dtype=int)
            if m == "nyu_fourview_pretrained":
                n_fb[:20] = 1
            nlbs_preds[m] = pd.DataFrame({"StudyInstanceUID": sub_ex["StudyInstanceUID"].astype(str), "score": n_sc, "n_crop_fallbacks": n_fb})

        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            lock_dir = td_p / "locked"
            out_dir = td_p / "out"
            for m in ALL_SIX_MODELS:
                r_eval = calibrate_rsna_thresholds_only(rsna_cases[m], ROOT / "datasets/paper1/rsna_val_calib_2fold_seed42.csv", n_boot=30, seed=42)
                lock_nlbs_model_predictions(m, nlbs_preds[m], r_eval["full_val_calib_thresholds"], f"sha_{m}", lock_dir)

            res = evaluate_six_models(
                locked_dir=lock_dir,
                rsna_case_dfs=rsna_cases,
                fold_csv=ROOT / "datasets/paper1/rsna_val_calib_2fold_seed42.csv",
                exam_manifest_csv=ROOT / "datasets/nlbs/exam_manifest.csv",
                common_4view_csv=ROOT / "datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
                out_dir=out_dir,
                n_boot=30,
                seed=42,
            )
            self.assertEqual(res["nlbs_common_4view_primary"]["n_units"], 6184)
            self.assertEqual(res["nlbs_common_4view_primary"]["n_clusters"], 5868)
            self.assertEqual(res["nlbs_full_cohort_secondary"]["n_units"], 6359)
            self.assertEqual(res["nlbs_cluster_collapsed_secondary"]["n_units"], 5868)
            self.assertIn("ce_b5_minus_trust_b5", res["nlbs_common_4view_primary"]["paired_contrasts"])
            self.assertEqual(
                res["nlbs_common_4view_primary"]["models"]["nyu_fourview_pretrained"]["fallback_audit"]["n_studies_with_fallback"],
                20,
            )
            self.assertTrue((out_dir / "six_model_summary.json").exists())
            self.assertTrue((out_dir / "six_model_summary.md").exists())

            # Tamper check must raise RuntimeError
            tamper_csv = lock_dir / "ce_b5_locked_nlbs_predictions.csv"
            tamper_csv.write_text(tamper_csv.read_text() + " ", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_and_load_locked_model(lock_dir, "ce_b5")

    def test_9_nyu_gmic_fallback_preserves_native_scale(self):
        ckpt = Path("external/GMIC/models/sample_model_1.p")
        if not ckpt.exists():
            self.skipTest("external/GMIC not present")
        _, mods = build_gmic_model(ckpt, model_index="1", device=torch.device("cpu"))
        # Construct thin 3500x2600 mammogram strip (<200 px wide) that fails 100-iter erosion -> triggers fallback
        thin = np.zeros((3500, 2600), dtype=np.uint16)
        thin[400:3000, :80] = 30000
        proc, meta = preprocess_uint16_for_nyukat(thin, "L-ML", GMIC_INPUT_SIZE, mods)
        self.assertTrue(meta["crop_fallback"])
        self.assertEqual(meta["nyukat_proxy_view"], "L-MLO")
        self.assertEqual(meta["fallback_strategy"], "native_scale_optimal_window_no_resize")
        self.assertEqual(proc.shape, GMIC_INPUT_SIZE)

    def test_10_gmic_uint16_png_cache_bit_exact_parity(self):
        import tempfile
        import cv2
        from src.paper1.gmic_adapter import save_uint16_png_atomic, standard_normalize_gmic_uint16

        ckpt = Path("external/GMIC/models/sample_model_1.p")
        if not ckpt.exists():
            self.skipTest("external/GMIC checkpoint not present")
        model, mods = build_gmic_model(ckpt, model_index="1", device=torch.device("cpu"))
        img = mods["loading"].read_image_png("external/GMIC/sample_data/images/0_L-CC.png")

        online_f32, _ = preprocess_uint16_for_nyukat(img, "L-CC", GMIC_INPUT_SIZE, mods, normalize=True)
        cached_u16, _ = preprocess_uint16_for_nyukat(img, "L-CC", GMIC_INPUT_SIZE, mods, normalize=False)
        self.assertEqual(cached_u16.dtype, np.uint16)
        self.assertEqual(cached_u16.shape, GMIC_INPUT_SIZE)

        with tempfile.TemporaryDirectory() as td:
            png_p = Path(td) / "0_L-CC.png"
            n_bytes = save_uint16_png_atomic(cached_u16, png_p, compression=1)
            self.assertGreater(n_bytes, 0)
            loaded_u16 = cv2.imread(str(png_p), cv2.IMREAD_UNCHANGED)
            self.assertEqual(loaded_u16.dtype, np.uint16)
            self.assertEqual(loaded_u16.shape, GMIC_INPUT_SIZE)
            reconstructed_f32 = standard_normalize_gmic_uint16(loaded_u16)
            max_diff = float(np.max(np.abs(online_f32 - reconstructed_f32)))
            self.assertLessEqual(max_diff, 1e-6)
            b_on, m_on = score_gmic_preprocessed_tensor(model, online_f32, torch.device("cpu"))
            b_ca, m_ca = score_gmic_preprocessed_tensor(model, reconstructed_f32, torch.device("cpu"))
            self.assertAlmostEqual(b_on, b_ca, places=6)
            self.assertAlmostEqual(m_on, m_ca, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
