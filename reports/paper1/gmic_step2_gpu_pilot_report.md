# Paper 1 — GMIC Step 2 L40S GPU Pilot & Fallback/View Audit Report

**Date:** 2026-11-22  
**Modal Profile:** `hasinishrak2015` (App `ap-YYXH9MfyVOBIXfb5IPctzQ`, NVIDIA L40S)  
**Source Cache Consumed:** `rsna-mammography-vol:/cache/paper1_gmic_uint16_v1/` (immutable; zero preprocessing repeated)  
**Decision:** **GO FOR FULL STEP 2** — Projected full `37,970`-image L40S cost is **`$0.771 USD`**, well within the **`$2.00 USD`** GPU budget.

---

## 1. Pilot Execution Summary (`200` RSNA + `200` NLBS Cached Images)
* **Checkpoint Integrity:** `sample_model_1.p` verified bit-for-bit (`sha256: 3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991`, `missing == []`, `unexpected == ["shared_rep_filter.weight"]`).
* **Pilot Scope (Label-Free):** `400` cached `(2944, 1920)` `uint16` PNGs (`200` RSNA `val_calib` including all `5` supplemental views + `5` fallback samples + `190` standard views; `200` NLBS label-free views), staged onto local NVMe `/tmp` and scored at `batch_size = 8` (FP32).
* **Score Sanity:** `all_scores_finite = true` (`min = 0.001441`, `median = 0.026833`, `max = 0.931492`). Zero cancer sensitivity, thresholds, or NLBS outcome labels were accessed.

---

## 2. Prediction Parity Verification (`sample_data` + Cached PNG vs. Raw DICOM)

### 2.1 Official `sample_data` Exam 0 Parity on NVIDIA L40S
| View | L40S `benign_pred` | README `benign` | Abs Err | L40S `malignant_pred` | README `malignant` | Abs Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0_L-CC` | `0.13576` | `0.1356` | `0.00016` | `0.00812` | `0.0081` | `0.00002` |
| `0_R-CC` | `0.89243` | `0.8929` | `0.00047` | `0.32523` | `0.3259` | `0.00067` |
| `0_L-MLO` | `0.23641` | `0.2368` | `0.00039` | `0.03340` | `0.0335` | `0.00010` |
| `0_R-MLO` | `0.95082` | `0.9509` | `0.00008` | `0.18122` | `0.1812` | `0.00002` |

### 2.2 Step 1 Cached PNG vs. Fresh Raw-DICOM Online Inference (`10` Real Samples)
* **Single-image parity (`batch=1` cached PNG vs `batch=1` fresh raw DICOM):** Exact to `<= 1e-6` across all `10` images (`5` RSNA incl. `L-ML`, `R-AT`, `R-MLO` fallback + `5` NLBS):
  * `822390278` (`R-CC`): `0.258738` == `0.258738`
  * `1442180348` (`L-MLO`): `0.102983` == `0.102983`
  * `1760744211` (`L-ML` supplemental): `0.130642` == `0.130642`
  * `602126376` (`R-AT` supplemental): `0.170115` == `0.170115`
  * `1249745058` (`R-MLO` fallback): `0.240995` == `0.240995`
  * NLBS `nlbs_img_00000..00004`: `0.023267`, `0.042032`, `0.019928`, `0.019928`, `0.027043` (identical online vs cached!)
* **Batch=8 vs. Batch=1 cuDNN FP32 Accumulation Delta:** `max |m_b8 - m_b1| = 0.001931` (`602126376 R-AT`), typical `1e-5`–`9e-5` elsewhere. Step 2 full inference can keep `batch_size=8` (`$0.771`) or use `batch_size=1` (`~16s / 400 imgs`, `~$0.98 USD`) if zero batch-reduction variance is preferred.

---

## 3. Audit of `80` RSNA Fallbacks & `5` Supplemental Views (No Step 1 Cache Rework Needed)

### 3.1 All `80` RSNA Fallback Images (`0.7285%` of `10,982`; `0` on `26,988` NLBS)
* **Unique RSNA Cases:** `36` patients (`by_view_slot`: `29` `L-MLO`, `26` `R-MLO`, `13` `R-CC`, `12` `L-CC`).
* **Photometric Split:** `51` (`63.75%`) `MONOCHROME1` + `29` (`36.25%`) `MONOCHROME2`.
* **Uniform Error Cause:** `80 / 80` triggered upstream `nyukat/GMIC` `ValueError: attempt to get argmax of an empty sequence` inside `crop_img_from_largest_connected(..., iterations=100)` on small/thin breast silhouettes.
* **Overlap with Supplemental Views:** `0 / 80`.
* **Cached PNG Pixel Verification:** `all_fallback_pngs_valid_nonzero = true` (`80/80` valid `(2944, 1920)` `uint16` PNGs, nonzero fraction `min = 0.0032`, `median = 0.0799`, `max = 0.8765`), and fallback `1249745058` (`R-MLO`) matched raw online DICOM inference cleanly (`0.240923` vs `0.240995`). **Verdict: Cache is clean; preserve Step 1 cache untouched.**

### 3.2 All `5` Supplemental RSNA Views Mapped via `_canonical_nyukat_view_slot`
| `image_id` | `patient_id` | Raw View | Mapped `nyukat` Proxy | Photometric | Oriented Flip | Crop Fallback | Nonzero Frac / Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1760744211` | `1511` | `L-ML` | `L-MLO` | `MONOCHROME2` | `false` | `false` | `0.9036` / `10983.95` |
| `602126376` | `3386` | `R-AT` | `R-CC` | `MONOCHROME2` | `true` | `false` | `0.5891` / `10857.94` |
| `917413880` | `3386` | `L-AT` | `L-CC` | `MONOCHROME2` | `true` | `false` | `0.5384` / `10629.20` |
| `771225790` | `40235` | `L-LM` | `L-MLO` | `MONOCHROME2` | `false` | `false` | `0.5579` / `10349.01` |
| `1473114297` | `40235` | `R-LM` | `R-MLO` | `MONOCHROME2` | `false` | `false` | `0.5896` / `10678.88` |

---

## 4. Measured Speed, Memory & Cost vs. `$2.00` Budget
* **Hardware & Memory:** `NVIDIA L40S` (`$2.3608/hr`), `peak_vram_gb = 1.511 GB`, `host_rss_gb = 6.167 GB`.
* **Measured Pilot Timing (`400` images, `batch_size=8`):**
  * `model_load_sec = 2.060s`, `nvme_stage_sec = 2.252s`, `decode_norm_sec = 8.022s`, `gpu_infer_sec = 2.026s` (`197.47 img/s` GPU-only) -> **`infer_e2e_sec = 12.300s` (`32.52 img/s` end-to-end)**.
* **Measured Cost (`ap-YYXH9MfyVOBIXfb5IPctzQ`):**
  * Pilot inference-only cost: **`$0.00807 USD`** (full container incl. `sample_data`, 10-img DICOM online parity & 85-image audit = **`$0.01737 USD`**).
* **Projected Full `37,970`-Image Cost (`10,982` RSNA + `26,988` NLBS, `4` shards):** **`0.3266` L40S hrs = `$0.771 USD`** -> **SAFE TO PROCEED WITHIN `$2.00` GPU BUDGET** (`within_2usd_budget: true`).
