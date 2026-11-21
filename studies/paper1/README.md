# Paper 1 — Multi-Model External Validation of Breast Cancer Screening AI

**Study Track:** Paper 1 (Evaluation, Calibration Transport & Failure-Mechanism Benchmark)  
**Repository:** `https://github.com/Research-TRR-HI/Breast-Cancer`  
**Context Brief:** [Paper1_External_Validation_Research_Context.md](../../Paper1_External_Validation_Research_Context.md)  
**Stage 0 Audit:** [stage0_readiness_audit.md](../../reports/paper1/stage0_readiness_audit.md)

---

## Core Research Question
> When six materially different breast-cancer screening AI systems are calibrated to the same empirical high-sensitivity target (`98%` primary, `95%` secondary) using RSNA labels only, how well do their frozen decision rules preserve cancer sensitivity and dismissal utility on an unseen Canadian screening population (NLBS), and do failures reflect ranking deterioration, score-distribution shift, or model-specific missed-cancer patterns?

## Six Prespecified Models (Two Evaluation Strata)

| Stratum | Model ID | Model Family & Pretraining | Status |
|---|---|---|---|
| **A: RSNA-Supervised** | `ce_b5` | Mammo-CLIP EfficientNet-B5 (`UPMC`, frozen) + BCE head | Complete (Stage B/C reused) |
| **A: RSNA-Supervised** | `trust_b5` | Mammo-CLIP EfficientNet-B5 (`UPMC`, frozen) + TRUST focal/dismissal head | Complete (Stage B/C reused) |
| **A: RSNA-Supervised** | `dinov2_rsna` | DINOv2 ViT-S/14 (`LVD-142M`, frozen) + RSNA cancer head | Integration & Training Pending |
| **A: RSNA-Supervised** | `resnet50_rsna` | ImageNet ResNet-50 (`IMAGENET1K_V2`) end-to-end RSNA fine-tune | Integration & Training Pending |
| **B: Pretrained-Direct** | `gmic_pretrained` | GMIC ResNet-18 (`NYU 2010–2017`, `sample_model_1.p`, `2944×1920`) | Integration & Inference Pending |
| **B: Pretrained-Direct** | `nyu_fourview_pretrained` | NYU Four-View CNN (`NYU 2010–2017`, `sample_image_model.p`, image-only) | Integration & Inference Pending |

## Relationship to Paper 2 (AnchorShift)
Paper 1 is strictly an **unadapted, source-frozen benchmark and failure-mechanism study** (NO new adaptation algorithm). Paper 2 ([AnchorShift_Research_Context.md](../../AnchorShift_Research_Context.md)) investigates unlabeled-target adaptation (`UPA`, candidate `AnchorShift` methods) and `Protocol 3` oracle target-label calibration. All original Stage B (`reports/stage_b/`) and Stage C (`reports/stage_c/`) artifacts remain immutable.
