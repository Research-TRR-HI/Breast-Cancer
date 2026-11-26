# MIAS v1.21 — 6-Model Compatibility & Eligibility Audit

**Cohort:** MIAS v1.21 (`322` images across `161` bilateral MLO pairs; `52` malignant exams, `109` non-malignant exams)  
**Modality:** Digitized Screen-Film Mammography (SFM), Joyce-Loebl scanning microdensitometer at 50 µm pitch, 8-bit `uint8` PGM (`P5`), landscape-oriented (`4320×1600/2048/2600`, `5200×4000`).

---

## 1. Critical Domain & Preprocessing Gaps vs RSNA-Trained Models

1. **Digitized Film (SFM) vs FFDM Domain Gap:** All 6 Paper 1 models (`ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`, `gmic_rsna_finetuned`, `nyu_fourview_rsna_finetuned`) were trained/calibrated on modern digital mammography (RSNA FFDM). Digitized film exhibits film grain, scanner border artifacts, non-linear H&D optical density response, and lower contrast resolution than FFDM.
2. **Bit-Depth & Dynamic Range:** MIAS is `8-bit` (`0–255`), whereas the RSNA/CMMD/LUMINA/NLBS Stage-1 pipeline expects `uint16` (`0–65535`). Rescaling (`uint8 * 257` or min-max float normalization) preserves relative ordering but cannot recover 12–16 bit subtle microcalcification contrast.
3. **Orientation & Aspect Ratio:** Raw MIAS PGMs are **landscape (`width > height`)** with the chest wall at the top edge. Before any Stage-1 (`1520×912`) or GMIC (`2944×1920`) breast-crop/resize, MIAS images **must** be rotated 90° to upright portrait orientation and standardized for laterality/chest-wall side; feeding raw unrotated PGMs would break spatial priors.
4. **Missing CC Views:** MIAS contains **only MLO views** (`L-MLO`, `R-MLO`; `0` CC views).

---

## 2. Per-Model Eligibility Matrix (GO / CONDITIONAL GO / NO-GO)

| Model ID | View Requirement | Input Spec | Eligibility Decision | Conditions / Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **`ce_b5`** | Single-image (`L-MLO`/`R-MLO` -> exam max/mean) | `1520×912` `uint16` crop, right-mirrored, Mammo-CLIP norm | **CONDITIONAL GO** *(Exploratory Film-Domain Only)* | Requires 90° upright rotation + breast-crop + `uint8→uint16` scaling; quarantine from primary FFDM external validation. |
| **`trust_b5`** | Single-image | `1520×912` `uint16` crop, right-mirrored, Mammo-CLIP norm | **CONDITIONAL GO** *(Exploratory Film-Domain Only)* | Same as `ce_b5`; valid only as an exploratory cross-era film-domain stress test. |
| **`dinov2_rsna`** | Single-image | `1520×912` `uint16` -> pad square -> `518×518` 3-ch ImageNet norm | **CONDITIONAL GO** *(Exploratory Film-Domain Only)* | Min-max per-image norm handles 8-bit range cleanly after 90° upright rotation + breast-crop. |
| **`resnet50_rsna`** | Single-image | `1520×912` `uint16` -> `854×512` 3-ch ImageNet norm | **CONDITIONAL GO** *(Exploratory Film-Domain Only)* | Aspect-preserving `5:3` resize after 90° upright rotation + breast-crop; exploratory film-domain only. |
| **`gmic_rsna_finetuned`** | Single-image | `2944×1920` `uint16` crop, per-image z-score | **CONDITIONAL GO** *(Exploratory Film-Domain Only)* | Pending Agent 1 RSNA fine-tuned checkpoint handoff; requires 90° upright rotation + native `2944×1920` crop. |
| **`nyu_fourview_rsna_finetuned`** | **4-view exam** (`L-CC`, `L-MLO`, `R-CC`, `R-MLO`) | Native NYU 4-view (`2677×1942` CC, `2974×1748` MLO) | **NO-GO (HARD-BLOCKED)** | **0 CC views** in MIAS. Never fabricate missing CC views; `nyu_fourview_rsna_finetuned` and `common_4view` mode must be hard-blocked on MIAS. |

---

## 3. Summary Recommendation

- **Primary FFDM External Evaluation:** Exclude MIAS from primary FFDM external benchmarking (`NLBS`, `CMMD`, `LUMINA`, `IMDB`).
- **Exploratory Film-Domain Stress Test:** **CONDITIONAL GO** for the **5 single-view models** (`ce_b5`, `trust_b5`, `dinov2_rsna`, `resnet50_rsna`, and `gmic_rsna_finetuned` once handed off) **only** after implementing a deterministic 90°-rotation + breast-crop Stage-1/GMIC preprocessor; **NO-GO (HARD-BLOCKED)** for `nyu_fourview_rsna_finetuned`.
- **No GPU Inference Launched:** Stopped after CPU download, verification, and audit as instructed.
