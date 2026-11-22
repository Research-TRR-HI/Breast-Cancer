# Paper 1 — NYU Four-View Step 1 CPU-Only Preprocessing Pilot & Feasibility Report

**Date:** 2026-11-22  
**Modal Profile:** `hasinishrak2015` (App `ap-8zAp1GZzkNZp4k7tPRSWrT`, strictly CPU-only / zero GPU allocated)  
**Target Cohorts:** Frozen 4-view cohorts (`9,528` RSNA `val_calib` images across `2,382` patients + `24,736` NLBS common 4-view images across `6,184` studies = **`34,264` images across `8,566` complete exams**)  
**Decision:** **READY FOR FULL CPU PREPROCESSING (`~$0.99 USD` compute)** — bit-exact **`0.0` parity** verified across full 10-epoch TTA draws, with **two explicit operational notes before approving full build** (`6 GiB` container memory sizing + `~135 GB` storage overage above Modal's `$25.34/mo` free volume cap).  
**Strict Stop Compliance:** Zero GPUs allocated, zero full preprocessing started, zero NYU inference or labels touched, zero modifications to CE/TRUST/GMIC locked files.

---

## 1. Architecture Separation & Cache Design (`paper1_nyu_uint16_v1`)
* **Why Caching Exact Cropped `uint16` ROIs + `best_center` Is Mandatory for NYU 10-TTA Parity:**
  Upstream `nyukat/breast_cancer_classifier` `augment_and_normalize_image(..., num_tta_epochs=10, seed=0)` applies random spatial shift `max_crop_noise=(100, 100)` and scale crop noise `max_crop_size_noise=100` around `best_center` **before** resizing back to `CC (2677x1942)` / `MLO (2974x1748)`. Saving pre-shrunk `(2677, 1942)` PNGs would lose border margin context for TTA draws.
* **Implemented Solution ([`nyu_fourview_adapter.py`](../../src/paper1/nyu_fourview_adapter.py) & [`preprocess_paper1_nyu_cpu.py`](../../scripts/preprocess_paper1_nyu_cpu.py)):**
  * Extracted DICOM LUT decode + `MONOCHROME1` inversion + native NYU connected-component crop (`or` `1:1` native-scale COM optimal-window fallback) + lateral flip as lossless 16-bit PNGs (`uint16`) alongside exact `(best_center_y, best_center_x)`, `u16_sha256`, `crop_window`, and fallback diagnostics.
  * Added `validate_and_load_nyu_4view_manifest` enforcing strict zero-label quarantine (`assert_no_forbidden_columns`), rejecting missing views, extra views, duplicate image IDs, or duplicate `(exam, view_slot)` entries (`tests/test_paper1_nyu_step1_cpu.py`: `3/3` PASS).

---

## 2. Small Representative CPU Pilot Results (`320` Images = `80` Complete Exams)
* **Sample Selection (`160` RSNA + `160` NLBS, label-free):**
  * **RSNA (`40` exams):** Included `patient_id = 71` (known `MONOCHROME1` fallback), `14` additional known difficult GMIC-fallback patients, `5` multi-candidate tie exams, and spread across both `site_id`s.
  * **NLBS (`40` exams):** Included all `3` `SIEMENS` exams (`12` views), `15` multi-candidate tie exams, and spread across `GE MEDICAL SYSTEMS`.
* **Data Integrity & Fallback Audit:**
  * `320 / 320` PNGs verified valid non-zero `uint16` (`all_uint16_valid_nonzero = true`, zero errors).
  * **Recorded Fallbacks:** `26` views (`12` unique RSNA cases incl. `patient_id=71`; `0` on NLBS `160` views), all hitting upstream `crop_mammogram.py` `ValueError: attempt to get argmax of an empty sequence` and cleanly handled via `native_scale_optimal_window_no_resize`.
* **Exact Parity Check (`8` Complete 4-View Exams = `32` Views × `10` TTA Draws = `320` Tensors):**
  * Verified `patient_id=71` + `3` RSNA + `2` `SIEMENS` NLBS + `2` `GE` NLBS exams against fresh raw-DICOM online `extract_nyu_cropped_view` + 10-epoch TTA (`seed=0`):
  * **`best_center_mismatches = 0`**
  * **`max_image_abs_diff = 0.0`**
  * **`max_10tta_tensor_abs_diff = 0.0`** *(exact bit-for-bit parity verified).*

---

## 3. CPU Configuration Efficiency, Storage & Cost Projections (`34,264` Images)

### 3.1 Measured Pilot Resource Comparison (`ap-8zAp1GZzkNZp4k7tPRSWrT`, `USD 0.0473/core/hr`, `USD 0.0080/GiB/hr`)
| Config | Pilot Images | Wall (s) | Throughput (`img/s`) | Peak `host_rss` | Measured Pilot USD | Projected Full `34,264` Compute USD | Full Wall Hours (1 Container / `8` Shards) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4vCPU / 4GiB` | `320` | `164.925` | `1.940` | **`6.734 GB`** *(risk!)* | `$0.01013` | `$1.0852` (`$0.3018` RSNA + `$0.7834` NLBS) | `4.91h` / `~37 min` |
| `2vCPU / 3GiB` | `80` | `58.093` | `1.377` | **`5.160 GB`** *(risk!)* | `$0.00191` | `$0.8198` (`$0.2280` RSNA + `$0.5918` NLBS) | `6.91h` / `~52 min` |
| **`2vCPU / 6GiB` *(Recommended)*** | — | — | `~1.377` *(2 threads)* | safely `< 6.0 GiB` | — | **`~$0.9857`** (`$0.274` RSNA + `$0.712` NLBS) | **`~6.91h` / `~52 min`** |

* **Key Engineering Finding:** SciPy morphological erosion/labeling partially serializes under Python threading (`4` threads gave only `1.41x` speedup over `2` threads at `2x` CPU cost), so **`2 vCPU` is `~24.5%` cheaper per image than `4 vCPU`**. However, `4` threads peaked at `6.734 GB` RSS and `2` threads peaked at `5.160 GB` RSS — so `process_nyu_cpu_shard` is configured with **`cpu=2.0, memory=6144` (`6 GiB`)** (`~$0.99 USD` total compute) to eliminate OOM risk over multi-hour runs.

### 3.2 Cache Storage, Compression & Workspace Credit Headroom
* **Workspace Credits (`hasinishrak2015`):** `Metered Cost = $45.42`, `Credits = -$19.99` (`~$10.01` remaining of `$30/mo` Starter credits), `Free Storage = -$25.34`, **`Billed Cost = $0.00`**. Notice `Volumes` meter is already at **`$26.36`** (`~$1.02` past `$25.34/mo` free quota) due to existing RSNA/Stage-1/GMIC caches.
* **Compression Comparison (`20`-image subcheck):** `compression=3` shrunk files by only **`1.24%`** (`ratio = 0.9876`, `1.267 -> 1.251 MB`) while adding `~0.17s/img` encode time, so **`compression=1` stays optimal**.
* **Estimated Storage Size (`34,264` images at pilot `4.024 MB/img` average):** **`~134.65 GB`** (`~USD 12.12 / month` at `$0.09/GiB/month` if kept alongside the `159.35 GB` GMIC cache, which is already completed & locked in `reports/paper1/predictions_locked/`). *Recommendation:* If keeping storage inside Modal free quota matters, either archive/verify before cleaning GMIC's finished PNG cache (`Step 1`+`2`+`3` already SHA-locked!) OR execute NYU Step 2 shortly after Step 1 so NYU's `135 GB` isn't retained for a full month.

### 3.3 Minor Non-Blocking Diagnostic Note
* **Unrelated Warning in Decode:** `pydicom` emits `UserWarning: Applying a VOI LUT on a float input array may give incorrect results` during `decode_dicom_array`. Since both online and cached NYU call the identical decoder and achieved **`0.0` float32 diff**, this does not alter NYU parity (`step 2` comparison remains exact), though keeping raw uint16 pre-LUT vs LUT consistency untouched preserves parity with GMIC/Stage-1.

**Readiness Verdict:** **READY FOR FULL CPU PREPROCESSING** once user approves `~USD 0.99` CPU compute + `~135 GB` temporary volume storage.
