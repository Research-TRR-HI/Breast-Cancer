# AnchorShift Stage B — TRUST Reproduction & Validation Plan

**Document Version:** 2.1 (Final Pre-Training Correction — Critical Fixes A–I)  
**Date:** 2026-11-18  
**Status:** **CONDITIONAL GO** — Critical Fixes A–I implemented & **20/20** local CPU tests PASSED (`reports/final_pre_training_readiness_audit.md`); cloud Stage-1 audit/build, T4 smoke test, 1,200-image L40S Mammo-CLIP parity pilot, and 5-batch extraction benchmark pending user execution before Stage B4 training.  
**Workspace:** `hasinishrak2015` (Modal) / `d:\Breast Cancer Research`  
**GitHub Repository:** `https://github.com/Research-TRR-HI/Breast-Cancer`

---

## 1. Executive Summary

This document details the reproduction of **TRUST (2026)** (*Threshold-Recalibrated Uncertainty-Safe Training for Certified Dismissal in Breast Cancer Screening*, arXiv:2609.00300) and its comparison with the standard Cross-Entropy (CE-only) baseline on the **RSNA Screening Mammography** dataset, establishing locked operating thresholds before zero-label transfer to **NLBS**.

All code has been audited directly against `arXiv:2609.00300v1` (see [`reports/trust_paper_implementation_traceability.md`](trust_paper_implementation_traceability.md) and [`reports/final_pre_training_readiness_audit.md`](final_pre_training_readiness_audit.md)). The 20-test local CPU verification suite (covering model loading, RNG-isolated head init, hand-calculated losses + exact micro-batch positive weighting, CP95/CP99 calibration, strict DICOM LUT/laterality/segmentation preprocessing, sampler parity, Stage-1 `uint16` & `v2` feature cache integrity, 3-epoch CPU parity pilot, and deterministic `v2` RNG checkpoint resume) passed 20/20 locally.

---

## 2. TRUST Reproduction Audit (Methods 3.1–3.11)

### 2.1 Code and Checkpoint Availability
1. **Official TRUST Code:** No public repository or official code release is linked in arXiv:2609.00300 or its ancillary files. The algorithm is reproduced faithfully from the mathematical formulations in Section 3.
2. **Official TRUST Model Weights:** No trained TRUST checkpoints on RSNA are publicly released. We train the model from scratch starting from the official foundation checkpoint.
3. **Foundation Backbone Checkpoint:** TRUST initializes its EfficientNet-B5 encoder from **Mammo-CLIP** (*A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography*, MICCAI 2024, Batman Lab):
   - Hugging Face repository: `shawn24/Mammo-CLIP` (pinned revision `2f356926a00fc3f0d9fdad1193c1464fd9adf564`)
   - File: `Pre-trained-checkpoints/b5-model-best-epoch-7.tar` (`1,654,145,453` bytes, SHA-256 `0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c`)
   - License: **CC BY-NC-SA 4.0** (strictly non-commercial research and educational use)
   - Pretrained nature: Genuinely contrastively pretrained on UPMC screening mammogram-report pairs, NOT fine-tuned downstream on RSNA.

### 2.2 Exact Architecture
- **Backbone:** Canonical Mammo-CLIP `efficientnet_custom` EfficientNet-B5 (`852/852` keys, `strict=True`, feature dimension $d = 2048$).
- **Encoder State:** Strictly frozen (`requires_grad = False`, evaluated in `eval()` mode throughout training).
- **Classification Head:**
  $$\text{Embedding } (B, 2048) \longrightarrow \text{LayerNorm}(2048) \longrightarrow \text{Dropout}(0.3) \longrightarrow \text{Linear}(2048, 1) \longrightarrow \text{Sigmoid} \longrightarrow p_i \in [0, 1]$$
- **Case-Level Aggregation:** Maximum image score across all views belonging to that examination:
  $$s_{\text{case}} = \max_{i \in \text{case}} p_i$$

### 2.3 Loss Formulations (Verified Against `arXiv:2609.00300v1` Eq. 6–8, 12–14)
1. **Binary Cross-Entropy Loss ($L_{\text{bce}}$):**
   $$L_{\text{bce}} = -\frac{1}{B} \sum_{i=1}^B \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$
2. **Focal Loss ($L_{\text{focal}}$, Eq. 12 — no alpha weighting):**
   $$L_{\text{focal}} = -\frac{1}{B} \sum_{i=1}^B \left[ y_i (1 - p_i)^\gamma \log(p_i) + (1 - y_i) p_i^\gamma \log(1 - p_i) \right]$$
   with $\gamma = 2.0$ and no $\alpha$ class weighting (`alpha=None`).
3. **Dynamic Dismissal Loss ($L_{\text{dismiss}}$, Eq. 7 & 14):**
   $$L_{\text{dismiss}} = \frac{1}{\max(N_{\text{pos}}, 1)} \sum_{i \in \text{pos}} \max(0, \tau_t + m - p_i)$$
   where:
   - $\tau_t$ is initialized at **`0.05`** for Epoch 1 and recalculated at the **beginning of each epoch ($\ge 2$)** on **image-level** `train_calib` predictions as the **largest threshold where the one-sided 95% Clopper–Pearson upper bound on dismissed positive-image prevalence is $\le 1\%$** (`0.0` if none satisfies).
   - $m = \mathbf{0.10}$ enforces a safety margin pushing cancer-positive scores above the dismissal boundary.
4. **Total Objective (Eq. 6 & 13):**
   $$L_{\text{total}} = L_{\text{bce}} + \lambda_{\text{focal}} L_{\text{focal}} + \lambda_{\text{dismiss}} L_{\text{dismiss}}$$
   - **CE Baseline:** $\lambda_{\text{focal}} = 0, \lambda_{\text{dismiss}} = 0$.
   - **TRUST:** $\lambda_{\text{focal}} = 1.0, \lambda_{\text{dismiss}} = \mathbf{0.05}$.

### 2.4 Training Hyperparameters
- **Epochs:** 20 (fixed 20-epoch schedule; **no early stopping and no `val_calib` best-checkpoint selection** — final 20th-epoch checkpoint used).
- **Epoch Sampling Policy (`paper_independent`):** Per TRUST Section 3.9, each minibatch of 80 independently draws 20 cancer-positive and 60 cancer-negative images (allowing minority reuse across batches). Because the paper does not state an explicit integer batch count per epoch, `steps_per_epoch` defaults to `ceil(37,444 / 60) = 625` batches/epoch (`1,000,000` 20-epoch occurrences; `full_neg_pass` remains available only as a documented alternative).
- **Batch Size:** 80 images (20 pos, 60 neg); fits in a single pass (`micro_batch_size = 80`) on NVIDIA L40S (`24.90 GB / 44.44 GB`).
- **Optimizer:** AdamW, learning rate $\eta = 3 \times 10^{-5}$, weight decay $10^{-4}$.
- **Input Resolution:** $1520 \times 912$ (Height 1520, Width 912), 3 channels.
- **Precision:** Mixed precision (`torch.amp.autocast("cuda")`).
- **Gradient Clipping:** Max norm 5.0.
- **Augmentation-Aware Feature Caching & Resume:** Because stochastic augmentation alters the frozen encoder input across occurrences, `src/data/feature_cache.py` stores occurrence-indexed `(epoch, batch_idx, slot_idx)` FP32 embeddings in atomic SHA-256-verified shards alongside deterministic `train_calib` / `val_calib` caches. Checkpointing (`src/training/checkpointing.py`) saves/restores full Python, NumPy, Torch CPU/CUDA, sampler, and augmentation RNG states.

### 2.5 Verified Deviations vs. TRUST (`arXiv:2609.00300v1`)
1. **Source-Only RSNA Split Protocol (Intentional AnchorShift Design):** Paper uses 5-fold CV (70/10/20) for in-domain evaluation and 90/10 RSNA + 20% **labeled NLBS** threshold calibration for external RSNA→NLBS evaluation. AnchorShift intentionally uses a single patient-level **70/10/20 RSNA split** so all source operating thresholds (95% & 98% recall) are calibrated on RSNA alone with zero NLBS label exposure.
2. **Epoch Step Count:** `steps_per_epoch = 625` (`ceil(37,444 / 60)` matching `datasets/rsna/splits_seed42.csv` -> `1,000,000` occurrences across 20 epochs) is explicitly specified since Section 3.9 does not publish an explicit minibatch count per epoch.
3. **One-Feature-Per-Image Benchmark Note:** Historical L40S benchmark (`ap-KidtVEoMzokd7fsFzpMeFo`, `$17.06` / `429.6 MB`) measured unaugmented 1-feature-per-image throughput and does NOT establish stochastic-augmentation 20-epoch training equivalence. See [`reports/final_pre_training_readiness_audit.md`](final_pre_training_readiness_audit.md) Section 6 for recalculated storage (`8.327 GB` FP32 `v2` feature cache) and Stage-1 CPU pre-crop projections.

---

## 3. RSNA Dataset Pipeline & Zero-Leakage Split Protocol

### 3.1 Patient-Level Stratified Splits (`datasets/rsna/splits_seed42.csv`)
Splits are generated strictly by `patient_id` to prevent any image, view, or laterality leakage:
- **`train` (Model Fitting, 70%):** 8,339 unique patients (340 cancer-positive, 4.08% prevalence), 38,251 images.
- **`train_calib` (Dynamic $\tau_t$ Updates, 10%):** 1,192 unique patients (49 cancer-positive, 4.11% prevalence), 5,473 images.
- **`val_calib` (Operating Threshold Calibration, 20%):** 2,382 unique patients (97 cancer-positive, 4.07% prevalence), 10,982 images.
- **Leakage Verification:** Pairwise intersection of patient sets is confirmed to be exactly **0**.

### 3.2 Complete Section 3.2 Preprocessing & Stochastic Augmentation
- **Stage 1 (DICOM -> cropped 16-bit representation):** Applies DICOM Modality LUT & VOI LUT (`pydicom.pixels`, strict error propagation), inverts `MONOCHROME1` so dense tissue is bright, min-max normalizes to `[0, 1]`, horizontally mirrors right-breast images under `laterality_policy="strict"`, segments the breast using border-median thresholding + hole filling + `9x9` morphological closing (`use_largest_cc=False` verbatim TRUST default, `strict_segmentation=True`), zeroes background outside the breast mask, crops to bounding box + `10%` margin (never stretching uncropped frames), and quantizes to lossless `uint16` shards (`src/data/stage1_cache.py`).
- **Stage 2 (Loader -> `(3, 1520, 912)` tensor):** Bilinear resize to `(1520, 912)`, `[0, 1]` re-normalization, training-only stochastic augmentation (rotation `±10°`, isotropic scale `0.90–1.10`, translation `±5%`, brightness `±0.05`, contrast `0.90–1.10`, `[0, 1]` clip), 3-channel replication, and Mammo-CLIP standardization (`mean=0.3089279`, `std=0.25053555408335154`). Evaluation (`train_calib`, `val_calib`) is strictly deterministic with zero augmentation.

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

## 5. Hardware Recommendations & Recalculated Cost Estimates

*(See `reports/final_pre_training_readiness_audit.md` Section 6 for complete recalculated storage and cost projections.)*

| Stage | Recommended Modal Compute | VRAM / Storage | Rationale | Estimated Cost |
|---|---|---|---|---|
| **Mammo-CLIP Download** | CPU-only (`cpu=1.0, memory=2048`) | 0 GB / 1.65 GB | Pinned SHA-256 download only. | ~$0.005 (DONE) |
| **Stage-1 Audit & Build** | CPU-only (`cpu=8.0, memory=16384`) | 0 GB / uint16 shards | Pre-crop all `54,706` RSNA DICOMs once so DICOMs are never decoded 1M times. | ~$1.5–$3.1 (estimate) |
| **Stage B Smoke Test** | T4 (`gpu="T4"`) | ~0.85 GB peak | 10 DICOMs, strict `852/852` & strict LUT/crop pipeline. | ~$0.01 (1 min) |
| **L40S Parity Pilot** | L40S (`gpu="L40S"`) | ~24.9 GB peak | `1,200` imgs (`500/500/200`), 20-ep online vs. cached equivalence gate (`scripts/pilot_parity_modal.py`). | Pending run |
| **Aug Cache Extract (`1M`)** | L40S (`mb=80`, Stage-1 input) | 24.9 GB / `8.327 GB` FP32 | Extract `1,000,000` augmented embeddings + `16,455` eval features once on L40S (benchmark 5 batches first). | Measure via Gate 5 |
| **Cached CE + TRUST (20 ep each)** | L40S (`gpu="L40S"`) | <0.5 GB | Trains `LayerNorm+Dropout+Linear` heads on verified FP32 cache (~2 min total). | ~$0.08 (estimate) |

*(Note: The prior `$17.06` / `429.6 MB` single-feature-per-image projection from `ap-KidtVEoMzokd7fsFzpMeFo` characterizes unaugmented inference/encoding throughput only and does NOT apply to stochastic-augmentation 20-epoch TRUST training.)*

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
