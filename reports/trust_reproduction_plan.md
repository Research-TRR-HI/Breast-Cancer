# AnchorShift Stage B — TRUST Reproduction & Validation Plan

**Document Version:** 1.0  
**Date:** 2026-11-17  
**Status:** Smoke Test Verified (T4 GPU, `checkpoint_loaded=True`) / Ready for Full L4 Training  
**Workspace:** `hasinishrak2015` (Modal) / `d:\Breast Cancer Research`  
**GitHub Repository:** `https://github.com/Hasinish/Breast-Cancer-Research`

---

## 1. Executive Summary

This document details the reproduction of **TRUST (2026)** (*Threshold-Recalibrated Uncertainty-Safe Training for Certified Dismissal in Breast Cancer Screening*, arXiv:2609.00300) and its comparison with the standard Cross-Entropy (CE-only) baseline on the **RSNA Screening Mammography** dataset, establishing locked operating thresholds before zero-label transfer to **NLBS**.

All code has been prepared in modular PyTorch modules under `src/` and Modal scripts under `scripts/`. Unit tests verifying frozen gradient isolation, forward/backward passes, loss calculations, dynamic thresholding, and zero patient leakage passed locally with 100% compliance.

---

## 2. TRUST Reproduction Audit (Methods 3.1–3.11)

### 2.1 Code and Checkpoint Availability
1. **Official TRUST Code:** No public repository or official code release is linked in arXiv:2609.00300 or its ancillary files. The algorithm is reproduced faithfully from the mathematical formulations in Section 3.
2. **Official TRUST Model Weights:** No trained TRUST checkpoints on RSNA are publicly released. We train the model from scratch starting from the official foundation checkpoint.
3. **Foundation Backbone Checkpoint:** TRUST initializes its EfficientNet-B5 encoder from **Mammo-CLIP** (*A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography*, MICCAI 2024, Batman Lab):
   - Hugging Face repository: `shawn24/Mammo-CLIP`
   - File: `Pre-trained-checkpoints/b5-model-best-epoch-7.tar` (1,654,145,453 bytes, ~1.65 GB)
   - License: **CC BY-NC-SA 4.0** (strictly non-commercial research and educational use)
   - Pretrained nature: Genuinely contrastively pretrained on UPMC screening mammogram-report pairs, NOT fine-tuned downstream on RSNA.

### 2.2 Exact Architecture
- **Backbone:** Canonical Mammo-CLIP `efficientnet_custom` EfficientNet-B5 (`852/852` keys, `strict=True`, feature dimension $d = 2048$).
- **Encoder State:** Strictly frozen (`requires_grad = False`, evaluated in `eval()` mode throughout training).
- **Classification Head:**
  $$\text{Embedding } (B, 2048) \longrightarrow \text{LayerNorm}(2048) \longrightarrow \text{Dropout}(0.3) \longrightarrow \text{Linear}(2048, 1) \longrightarrow \text{Sigmoid} \longrightarrow p_i \in [0, 1]$$
- **Case-Level Aggregation:** Maximum image score across all views belonging to that examination:
  $$s_{\text{case}} = \max_{i \in \text{case}} p_i$$

### 2.3 Loss Formulations
1. **Binary Cross-Entropy Loss ($L_{\text{bce}}$):**
   $$L_{\text{bce}} = -\frac{1}{B} \sum_{i=1}^B \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$
2. **Focal Loss ($L_{\text{focal}}$):**
   $$L_{\text{focal}} = -\frac{1}{B} \sum_{i=1}^B \alpha_t (1 - p_{t, i})^\gamma \log(p_{t, i})$$
   with $\gamma = 2.0$, $\alpha = 0.25$.
3. **Dynamic Dismissal Loss ($L_{\text{dismiss}}$):**
   $$L_{\text{dismiss}} = \frac{1}{\max(N_{\text{pos}}, 1)} \sum_{i \in \text{pos}} \max(0, \tau_t + \text{margin} - p_i)$$
   where:
   - $\tau_t$ is the provisional threshold dynamically recalculated at the end of each epoch on the internal training-calibration set.
   - $\text{margin} = 0.05$ enforces a safety margin pushing cancer-positive scores above the dismissal boundary.
4. **Total Objective:**
   $$L_{\text{total}} = L_{\text{bce}} + \lambda_{\text{focal}} L_{\text{focal}} + \lambda_{\text{dismiss}} L_{\text{dismiss}}$$
   - **CE Baseline:** $\lambda_{\text{focal}} = 0, \lambda_{\text{dismiss}} = 0$.
   - **TRUST:** $\lambda_{\text{focal}} = 1.0, \lambda_{\text{dismiss}} = 1.0$.

### 2.4 Training Hyperparameters
- **Epochs:** 20
- **Epoch Sampling Policy (`full_neg_pass`):** Every epoch iterates through 100% of negative `train` images once (`ceil(N_neg / 60)` batches/epoch) while cycling and reshuffling positive `train` images with wraparound to preserve the exact 20:60 pos:neg ratio per 80-image minibatch (replacing the legacy `len(pos_idx) // 20` policy which only covered ~6.4% of negatives per epoch).
- **Batch Size:** 80 images (strictly balanced: 20 cancer-positive, 60 cancer-negative). On GPU hardware, implemented using micro-batching (8 images per micro-batch: 2 pos, 6 neg) and gradient accumulation over 10 steps ($8 \times 10 = 80$).
- **Optimizer:** AdamW, learning rate $\eta = 3 \times 10^{-5}$, weight decay $10^{-4}$.
- **Input Resolution:** $1520 \times 912$ (Height 1520, Width 912), 3 channels.
- **Precision:** Mixed precision (`torch.amp.autocast("cuda")`).
- **Gradient Clipping:** Max norm 5.0.
- **Dynamic GPU Selection & Resume:** Modal GPU selection via `.with_options(gpu=gpu)` supporting `L40S`; per-epoch `val_calib` monitoring, `_latest.pt` / `_best.pt` checkpointing, and `--resume`.

### 2.5 Parameter Verification & Documented Deviations vs. TRUST (arXiv:2609.00300)
1. **Reference Source Constraint:** `arXiv:2609.00300` PDF is not stored locally (the workspace PDF by Mohandas et al. 2026 is a separate tabular-WDBC paper sharing the acronym); hyperparameters (`margin=0.05`, `focal_alpha=0.25`, `focal_gamma=2.0`, `lambda_focal=1.0`, `lambda_dismiss=1.0`, `tau_target_recall=0.98`) are reproduced from the prior Section 3 audit and exposed via CLI flags.
2. **Dynamic `tau_t` Granularity:** `tau_t` is updated per epoch from **image-level** `train_calib` positive scores (matching image-level `DynamicDismissalLoss` inputs $p_i$), whereas `val_calib` operating thresholds use **case-level max** aggregation across `patient_id`.
3. **Epoch Coverage:** `full_neg_pass` ensures 100% negative coverage per epoch while keeping 20:60 pos:neg minibatches.
4. **Input Standardization:** Applies Mammo-CLIP ImageNet mean/std standardization (`standardize=True`) after `MONOCHROME1` inversion and min-max scaling.

---

## 3. RSNA Dataset Pipeline & Zero-Leakage Split Protocol

### 3.1 Patient-Level Stratified Splits (`datasets/rsna/splits_seed42.csv`)
Splits are generated strictly by `patient_id` to prevent any image, view, or laterality leakage:
- **`train` (Model Fitting, 70%):** 8,339 unique patients (340 cancer-positive, 4.08% prevalence), 38,251 images.
- **`train_calib` (Dynamic $\tau_t$ Updates, 10%):** 1,192 unique patients (49 cancer-positive, 4.11% prevalence), 5,473 images.
- **`val_calib` (Operating Threshold Calibration, 20%):** 2,382 unique patients (97 cancer-positive, 4.07% prevalence), 10,982 images.
- **Leakage Verification:** Pairwise intersection of patient sets is confirmed to be exactly **0**.

### 3.2 Preprocessing Integrity
- **Photometric Inversion:** Images with `PhotometricInterpretation == "MONOCHROME1"` (40/200 observed in Stage A audit) are inverted ($x \leftarrow \max(x) - x$) so high intensity universally represents radiopaque dense tissue.
- **Normalization:** Robust min-max intensity scaling to $[0, 1]$.
- **Interpolation:** Bilinear interpolation to $(1520, 912)$ preserving the full breast field.

---

## 4. NLBS Examination Grouping Audit & Denominator Review

### 4.1 Findings from Stage A3 Audit
1. **Resolution of 149 Cases vs. 151 Exams:**
   - Forensic examination revealed that two abnormal patient folders—`p_4334` and `p_4461`—each contain **two distinct 4-view acquisitions** recorded under separate DICOM `StudyInstanceUID`s.
   - For both patients, biopsy-confirmed cancer is present in both visits. Treating them as separate examinations is clinically correct.
2. **The 71 Shared `StudyInstanceUID`s & Cross-Folder Leakage:**
   - `6,359` unique `StudyInstanceUID`s exist across `6,430` folder records.
   - `p_3280` contains 180 images across 44 `StudyInstanceUID`s, 43 of which are duplicate copies of examinations that also reside in individual `p_XXXX` folders.
   - Another 28 `StudyInstanceUID`s are duplicated between pairs of `False Positive` folders.
3. **Connected-Component Grouping (`split_cluster_id`):**
   - Union-Find clustering across `(case_id, StudyInstanceUID)` produced **`5,942` independent clusters**.
   - `split_cluster_id` clusters all related folders (such as `p_3280` and its 27 co-linked folders) into a single atomic block, guaranteeing **zero leakage** during any downstream evaluation or adaptation splits.
4. **Denominators for Evaluation:**
   - **Case-Level Max-Score Unit:** Deduplicated `StudyInstanceUID` ($N = 6,359$).
   - **Clustering / Confidence Interval Unit:** `split_cluster_id` ($N = 5,942$) for cluster bootstrapping.

---

## 5. Hardware Recommendations & Cost Estimates

| Stage | Recommended Modal GPU | VRAM Required | Rationale | Cost Estimate |
|---|---|---|---|---|
| **Mammo-CLIP Download** | **CPU-only** (`cpu=1.0, memory=2048`) | 0 GB | Network I/O only. | ~$0.005 |
| **Stage B Smoke Test** | **T4** (`gpu="T4"`) | ~4.5 GB peak | 10 samples, verify forward/backward & VRAM. | ~$0.01 (1 min) |
| **Full 20-Epoch Training** | **L4** (`gpu="L4"`) | ~12–16 GB peak | 24GB VRAM, Ada Lovelace FP8/BF16 tensor cores; ~3.5x faster than T4 at ~$0.80/hr. | ~$2.40–$3.20 (3–4 hrs) |

*(Note: Never allocate A100/H100 for linear-probe training with a frozen backbone; the classification head has only 2,049 trainable parameters!)*

---

## 6. Implementation Deliverables Summary

The following modular code is committed and tracked:
1. `src/models/trust_model.py`: `MammoCLIPEncoder`, `TRUSTClassifier`, and `aggregate_case_scores`.
2. `src/models/losses.py`: `FocalLoss`, `DynamicDismissalLoss`, and unified `TRUSTLoss`.
3. `src/models/mammo_clip_loader.py`: `extract_mammo_clip_encoder_state` and `load_and_verify_mammo_clip`.
4. `src/data/preprocessing.py`: `preprocess_dicom` with `MONOCHROME1` inversion.
5. `src/data/splits.py`: `generate_patient_splits` for RSNA.
6. `src/data/rsna_dataset.py`: `RSNADataset` and `BalancedBatchSampler`.
7. `src/training/calibration.py`: `compute_target_recall_threshold`, `clopper_pearson_upper_bound`, and `evaluate_dismissal_at_threshold`.
8. `datasets/rsna/splits_seed42.csv`: Generated deterministic RSNA splits.
9. `scripts/download_mammoclip_modal.py`: Checkpoint downloader.
10. `scripts/smoke_test_stage_b_modal.py`: GPU smoke test.
11. `scripts/train_trust_modal.py`: Full training entrypoint.
12. `tests/test_stage_b_smoke.py`: Local unit test suite (100% passing).
