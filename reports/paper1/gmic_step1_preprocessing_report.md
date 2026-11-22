# Paper 1 — GMIC Step 1 CPU-Only Preprocessing Report

**Date:** 2026-11-22  
**Modal Profile:** `hasinishrak2015` (Zero GPU allocated)  
**Script:** `scripts/preprocess_paper1_gmic_cpu.py` (`4 vCPU / 8 GiB RAM` containers, 16 resumable shards)  
**Status:** **COMPLETE — STEP 2 (GPU INFERENCE) IS READY**

---

## 1. Strict Quarantine & Hardware Compliance Checklist
* **Zero GPU Allocation:** Strictly CPU-only Modal functions (`gpu=None`, asserted `torch.cuda.is_available() == False` inside containers).
* **Zero Predictions / Evaluation / NLBS Labels:** GMIC checkpoint `sample_model_1.p` was never loaded on Modal, no predictions or thresholds were computed, and `datasets/nlbs/exam_manifest.csv` was never mounted (`datasets/nlbs/inference_manifest_stage_c.csv` checked via `assert_no_forbidden_columns`; `splits_seed42.csv` read strictly for `image_id, patient_id, split, laterality, view, rel_path` on `split == 'val_calib'`).
* **Official GMIC Method + View Mapping Catch:** Reused exact `nyukat/GMIC` connected-component cropping + optimal center + z-score preparation saved losslessly as `(2944, 1920)` 16-bit grayscale PNGs pre-standardization (`standard_normalize_gmic_uint16`). Resolved 5 supplemental RSNA `val_calib` views (`L-ML`, `L-AT`, `R-AT`, `L-LM`, `R-LM`) cleanly via `_canonical_nyukat_view_slot`.

---

## 2. 200-Image Cost & Parity Benchmark Gate (`ap-MuRdTTfrd9slnLiv2Ai6PQ`)
* **Scope:** `100` RSNA `val_calib` + `100` NLBS DICOMs on a single `4 vCPU / 8 GiB RAM` CPU container.
* **Wall Time & Throughput:** `135.549s` (`1.475 img/s`).
* **Benchmark Cost:** **`$0.00951 USD`** (`cost_gate_passed_under_3usd: true`).
* **Projected Full 37,970-Image Cost:** `$1.805 USD` (comfortably below `$3.00` ceiling).
* **Integrity & Parity:** `200/200` `(2944, 1920)` `uint16` PNGs, `2` native-scale fallbacks, **`max_parity_abs_diff = 0.0`**.

---

## 3. Full Step 1 Execution, Actual CPU Spend & Verification (`ap-i9HV9RJvsFrAoyBBZ7mlk5` / `ap-iYfqLVbUg2s26E9DosJ9WG`)

| Split | Expected Images | Verified PNGs `(2944x1920, uint16)` | Resumed Skips | Fallbacks | Volume Storage | Max Parity Abs Diff | Build CPU+RAM Spend (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`rsna_val_calib`** | `10,982` | **`10,982`** (`100%`) | `700` *(resumed cleanly)* | `80` (`0.7285%`) | `62.058 GB` | **`0.0`** (`n=10`) | **`$0.8280`** |
| **`nlbs`** | `26,988` | **`26,988`** (`100%`) | `0` | **`0`** (`0.0%`) | `97.291 GB` | **`0.0`** (`n=10`) | **`$0.7564`** |
| **Total Build** | **`37,970`** | **`37,970`** (`100%`) | **`700`** | **`80`** (`0.2107%`) | **`159.349 GB`** | **`0.0`** (`n=20`) | **`$1.5844`** |

*Note on auxiliary Step 1 CPU runs:* Adding the 200-image benchmark (`$0.00951`), brief pre-`L-ML`-fix partial run (`~$0.05` whose `700` PNGs were reused identically via resumability), and post-build `verify_gmic_cpu_cache` (`~$0.02`), total Step 1 spend stayed near **`~$1.66 USD`** — well below the `$3.00` limit.

**Reusable Modal Storage Paths (`rsna-mammography-vol`):**
* `/cache/paper1_gmic_uint16_v1/rsna_val_calib/png_2944x1920/<image_id>.png` (`manifest.csv`, `fallback_log.json`, `verify_summary.json`)
* `/cache/paper1_gmic_uint16_v1/nlbs/png_2944x1920/<image_uid>.png` (`manifest.csv`, `fallback_log.json`, `verify_summary.json`)

**Step 2 Readiness:** **READY** (`scripts/run_paper1_gmic_modal.py` updated to load `/cache/paper1_gmic_uint16_v1/` directly with zero DICOM decode/cropping overhead, preserving Step 1 fallback audit metadata).
