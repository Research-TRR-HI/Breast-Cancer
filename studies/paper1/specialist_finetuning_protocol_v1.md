# Paper 1 — Supplementary Exploratory Specialist Fine-Tuning Protocol v1.0

**Effective Date:** 2026-11-24  
**Branch:** `paper1-specialist-finetuning`  
**Status:** DRAFT / EXPLORATORY (ZERO GPU TRAINING LAUNCHED)  
**Primary Benchmark Disclosure:** The frozen 6-model primary Paper 1 benchmark (`studies/paper1/protocol_v1.md`) remains unmodified. External NLBS results for `gmic_pretrained` (`0.8541` AUROC, `97.35%` @98% sensitivity, `24.27%` dismissal) were already observed on `2026-11-22`. Accordingly, `gmic_rsna_finetuned` and `nyu_fourview_rsna_finetuned` are **strictly registered as supplementary exploratory sensitivity experiments** (post-observation), NOT part of the primary 6-model benchmark.

---

## 1. Scientific Rationale & Hypothesis
In the primary Paper 1 benchmark, `gmic_pretrained` and `nyu_fourview_pretrained` were trained on NYU clinical mammography data (2010–2017), whereas `ce_b5`, `trust_b5`, `dinov2_rsna`, and `resnet50_rsna` were trained on RSNA. Comparing RSNA-fine-tuned variants (`gmic_rsna_finetuned` [Priority 1], `nyu_fourview_rsna_finetuned` [Priority 2]) against their NYU-pretrained counterparts isolates **source-domain adaptation vs. architecture/resolution effects** on RSNA-to-NLBS operating-point transportability.

---

## 2. Dataset Partitioning & Strict Label Hygiene
1. **RSNA `train` (`38,251` images / `8,339` patients / `340` cancer-positive):** Used exclusively for supervised gradient updates (`BalancedBatchSampler`, `25%` positive micro-batches).
2. **RSNA `train_calib` (`5,473` images / `1,192` patients / `49` cancer-positive):** Used exclusively for per-epoch AUROC checkpoint selection (`best_train_calib.pt`).
3. **RSNA `val_calib` (`10,982` images / `2,382` patients / `97` cancer-positive):** **100% unseen** during training and checkpoint selection; used solely for frozen `95%`/`98%` threshold calibration (`tau_98`, `tau_95`) and 2-fold OOF source evaluation.
4. **NLBS (`N=6,184` common 4-view / `N=6,359` full):** Strictly label-free inference using frozen RSNA checkpoints and frozen RSNA thresholds; scores SHA-256 locked before outcome join.
5. **No Benign Label Fabrication:** RSNA `splits_seed42.csv` provides `cancer` (`0/1`) only and lacks benign biopsy/finding labels. Both fine-tuning losses (`compute_gmic_rsna_loss`, `compute_nyu_rsna_loss`) supervise **exclusively the malignant outputs** and NEVER treat `cancer == 0` as `benign == 0`.

---

## 3. Native-Resolution Requirement (No Stage-1 Upscaling)
- Stage-1 `1520x912` PNGs MUST NOT be upscaled to simulate native inputs.
- **GMIC (`gmic_rsna_finetuned`):** Requires native `(2944, 1920)` `uint16` PNGs (`paper1_gmic_uint16_v1`) built from raw RSNA `train` + `train_calib` DICOMs.
- **NYU Four-View (`nyu_fourview_rsna_finetuned`):** Requires native `(2677, 1942)` CC and `(2974, 1748)` MLO `uint16` PNGs (`paper1_nyu_uint16_v1`) built from raw RSNA `train` + `train_calib` DICOMs.

---

## 4. Training Configurations & Budget-Gated Execution
1. **GMIC (`gmic_rsna_finetuned`, Priority 1):**
   - `strategy="partial"` (default): trains `ds_net.layer_list.4`, `ds_net.final_bn`, `left_postprocess_net`, `dn_resnet.layer4`, `mil_attn_*`, `classifier_linear`, `fusion_dnn` with differential LR (`1e-4` head, `1e-5` backbone).
   - Loss: FP32 malignant-only `BCE(y_fusion) + BCE(y_global) + BCE(y_local) + 1e-5 * ||saliency_map_mal||_1`.
2. **NYU Four-View (`nyu_fourview_rsna_finetuned`, Priority 2):**
   - `strategy="head_only"` (default for budget/overfitting control): trains `fc1_cc`, `fc1_mlo`, `output_layer_cc`, `output_layer_mlo` (`533,520` params) in FP32 with `NLLLoss` on malignant indices `[2, 3]` averaged across CC and MLO branches.
3. **Mandatory Pilot Gate:** Both scripts block `--stage train` without `--approve-full-train` and require a completed `--stage pilot` throughput/VRAM/cost projection first.
