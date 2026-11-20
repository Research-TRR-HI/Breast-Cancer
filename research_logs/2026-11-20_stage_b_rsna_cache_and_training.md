# Research Log: Stage B Execution — 1M Augmentation Cache & RSNA 20-Epoch Training (CE vs. TRUST)

**Date**: 2026-11-20  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Dynamic Calibration & Dismissal / Benchmark Evaluation  
**Milestone Scope**: Cloud Gates 1–4 (`hasinishrak2015`), Cloud Transfer (`hasinishrak74001`), Step 5 (`1,000,000`-occurrence extraction & SHA-256 verification), Step 6 (`ce` 20-epoch training), Step 7 (`trust` 20-epoch training)  
**Status**: VERIFIED COMPLETE  

---

## 1. Research Objective & Context
- Execute the audited Stage B RSNA source-only training protocol (`TRUST`, arXiv:2609.00300v1 Sec. 3.2–3.10; frozen Mammo-CLIP EfficientNet-B5 backbone) across `20` epochs (`625` steps/epoch × `80` images/batch = `1,000,000` stochastic training occurrences) for both:
  - **Step 6 (`ce`):** Cross-entropy baseline head (`LayerNorm(2048) -> Dropout(0.3) -> Linear(2048, 1)`).
  - **Step 7 (`trust`):** Full TRUST head with Focal (`gamma=2.0`, `lambda_focal=1.0`) + Dismissal penalty (`margin=0.10`, `lambda_dismiss=0.05`) + dynamic image-level 95% Clopper-Pearson upper-bound (`<= 1%`) training threshold `tau_t`.
- Freeze final-epoch (Epoch 20) checkpoints and `val_calib` source operating thresholds (`95%` and `98%` target recall) strictly on RSNA (`train=38,251`, `train_calib=5,473`, `val_calib=10,982`), keeping NLBS completely quarantined for Stage C zero-label transfer evaluation.

---

## 2. Datasets, Volumes & Manifest References
- **Active Execution Modal Profile & App:** `hasinishrak74001` / `anchorshift-stage-b-train` (`ap-3b5QkJkuOYFN0FDCYvTYQw`)
- **Volume:** `rsna-mammography-vol` (`/data`)
- **Stage-1 Preprocessed Cache:** `/data/cache/stage1_rsna_uint16/png_1520x912/` (`54,706` lossless 16-bit PNGs, `65.9 GiB`, `stage1_config_hash: d4f4ec48b700f91dfc90e5036c64899c9033d6784998ab22f4ea688f5df3402b`)
- **Canonical Split Manifest:** `/data/rsna/splits_seed42.csv` (`split_manifest_sha256: 902c2add501a36eba0b320adc68fb4e758f34509956fa19bbc98ef23d8868a5b`, `54,706` rows across `11,913` patients: `train=37,444 neg / 807 pos`, `train_calib=5,359 neg / 114 pos`, `val_calib=10,745 neg / 237 pos`)
- **Foundation Backbone Checkpoint:** `/data/models/mammo_clip/b5-model-best-epoch-7.tar` (`model_checkpoint_sha256: 0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c`)
- **Stage-2 Augmentation Cache:** `/data/cache/rsna_aug_seed42` (`640` shards, `cache_manifest_sha256: 6c34a8c0add35e7514b45f0e001fc1189f46ebe3b482bfff302db13e16d8fdbd`, `preprocess_config_hash: 454474d007676616fa43e367c6ca0cad722bd527f6deb294414b5a0b11b7c01f`)
- **Result Artifacts Stored Locally:**
  - `reports/stage_b/rsna_ce_eval_results.json` (`/data/models/checkpoints/ce_rsna_seed42_final_epoch20.pt`)
  - `reports/stage_b/rsna_trust_eval_results.json` (`/data/models/checkpoints/trust_rsna_seed42_final_epoch20.pt`)

---

## 3. Forensic & Execution Findings (Cloud Gates 1–5 & Steps 6–7)
1. **Cloud Gates 1–4 (`hasinishrak2015`) Passed Prior to Full Extraction:**
   - **Gate 1 (`--audit-real`, `ap-GXKosoQHFpQosDzDNfuvP8`):** `16/16` real RSNA DICOMs verified across `L/R x CC/MLO` (`MONOCHROME1: 2`, `MONOCHROME2: 14`, `fallback_count: 0`).
   - **Stage-1 Cache Build & Resize (`54,706` images):** Repacked `219` shards (`1.05 TB`) -> 16-bit PNGs (`394.68 GiB`) -> pre-resized `1520x912` PNGs (`65.9 GiB`, `2,263.27 s`, `54,705` clean crops + `1` `empty_mask` edge DICOM `822/1942326353.dcm`).
   - **Gate 3 (1,200-img L40S AMP Parity Pilot, `ap-DUiCE1N2knIHA84aij0mWG`):** `status=PASS`, `max_prob_diff=2.35e-04 <= 0.001` (`43.00 min`), resolving CUDA `canUse32BitIndexMath` at `batch=80` via exact `micro_batch_size=20`.
   - **Gate 4 (5-Batch Extraction Benchmark, `ap-TTf1PLmeaAxpfnm5tlf5DZ`):** With `ephemeral_disk=524288` MiB NVMe `/tmp` staging + GPU Stage-2 normalization/augmentation, Stage-2 prep dropped from `34.04s` to `0.317s` (`107x` speedup), yielding `2.2775s/batch` (`35.13 img/s`, projected `8.038h`).
2. **Cloud Migration & Split CSV Canonicalization (`hasinishrak74001`):**
   - Initial Step 5 spawn (`fc-01M7TQ9YVXW308RJE2SP80AZ2P`) flagged a `Stage1CacheError` SHA-256 mismatch because local 9-column `datasets/rsna/splits_seed42.csv` (`e6f8272c...`) differed from `hasinishrak2015`'s canonical 16-column `splits_seed42.csv` (`902c2add...`) baked into `stage1_manifest.json`.
   - Verified `100%` equality on all `54,706` `(patient_id, image_id, split, cancer)` mappings (`SPLIT_MATCH: True`, `CANCER_MATCH: True`) and synced `902c2add...` onto `hasinishrak74001:/rsna/splits_seed42.csv`.
3. **Step 5 Full Cache Extraction & Verification (`fc-01M7TQHVX5BKVY29SPJDSJGACA`):**
   - Completed `1,000,000 / 1,000,000` occurrences in a single uninterrupted run at **`35.0 img/s`** (`640` shards).
   - Independent CPU `verify_cache_only` confirmed `status: VERIFIED` across all `640` shard SHA-256 checksums, `train_calib`/`val_calib` eval tensors, and schedule metadata.

---

## 4. Training Progression & Final RSNA `val_calib` Operating Thresholds

### Runtime Environment (`NVIDIA L40S`, AMP)
- Python `3.12.10` | PyTorch `2.5.1+cu124` | CUDA `12.4` | cuDNN `90100` | `precision=amp` | `seed=42` | `lr=3e-5`, `weight_decay=1e-4`, `grad_clip=5.0`.

### Training Convergence (20 Epochs, Final-Epoch Selection)
| Baseline | Call ID | Epoch 1 Loss | Epoch 20 Loss | Dynamic `tau_t` (Ep 2 -> Ep 20) | Ep 20 `train_calib` Dismissal / Miss (`n=5,473`, `pos=114`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CE-Only (`ce`)** | `fc-01M7VTBZZXY4949AC979WFVZ3F` | `0.54880` | `0.44916` | N/A (fixed `0.05000`) | N/A |
| **TRUST (`trust`)** | `fc-01M7VTC0JD27AKR088MB32AM9R` | `0.72754` | `0.59672` | `0.26050 -> 0.25610` | `3,380 / 5,473` dismissed, `k=24` missed (`CP95 UB = 0.009972 <= 0.01`) |

### Case-Level RSNA `val_calib` Evaluation (`2,382` Cases, `97` Cancers)
| Baseline | Target Recall | Frozen Threshold (`tau`) | Dismissed / Total | **Dismissal Rate** | Detected / Cancers (`Recall`) | Missed Cancers | Dismissed Prev | CP99 Upper Bound | Meets Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CE-Only** | `95%` | `0.124451` | `536 / 2,382` | **`22.50%`** | `93 / 97` (`95.88%`) | `4` | `0.746%` | `2.150%` | `True` |
| **TRUST** | `95%` | `0.195435` | `493 / 2,382` | **`20.70%`** | `93 / 97` (`95.88%`) | `4` | `0.811%` | `2.336%` | `True` |
| **CE-Only** | `98%` | `0.096191` | `345 / 2,382` | **`14.48%`** | `96 / 97` (`98.97%`) | `1` | `0.290%` | `1.909%` | `True` |
| **TRUST** | **`98%`** | `0.179565` | **`409 / 2,382`** | **`17.17%`** | `96 / 97` (`98.97%`) | `1` | **`0.244%`** | **`1.612%`** | `True` |

- **Key Source Observation:** At the stringent **98% clinical recall target**, TRUST dismisses **`+2.69%` more normal exams** (`+64` cases: `409` vs `345`) with lower dismissed prevalence (`0.244%` vs `0.290%`) and tighter `99%` Clopper-Pearson upper bound (`1.612%` vs `1.909%`). At `95%` recall on source `val_calib`, CE dismisses `+1.81%` more (`536` vs `493`). Both frozen Epoch-20 checkpoints advance directly to Stage C zero-label transfer evaluation on NLBS.

---

## 5. Next Stage (Stage C — Quarantined NLBS Evaluation)
- Implement Stage-1 preprocessing + Mammo-CLIP B5 inference for NLBS (`26,988` images / `6,430` exams), evaluate frozen `ce_rsna_seed42_final_epoch20.pt` and `trust_rsna_seed42_final_epoch20.pt` at frozen RSNA `tau_95`/`tau_98`, and hash-lock predictions prior to revealing NLBS cancer labels.
