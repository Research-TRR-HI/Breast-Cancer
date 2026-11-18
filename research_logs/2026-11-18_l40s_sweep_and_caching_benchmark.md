# Research Log: L40S Micro-Batch OOM Sweep & Live vs. Feature-Caching Benchmark

**Date**: 2026-11-18  
**Researcher / Agent**: Hasin Ishrak / Antigravity AI  
**Research Track**: Compute Optimization, Numerical Parity & Cost Feasibility  
**Milestone Scope**: Empirical NVIDIA L40S (44.44 GB VRAM) micro-batch ceiling sweep (`ap-mY4VfFocqIRVEnGIeV2CtD`) and fair 1,000-image Approach A (Live TRUST) vs. Approach B (Frozen 2048-D Feature Caching) speed & Modal cost comparison (`ap-KidtVEoMzokd7fsFzpMeFo`)  
**Status**: VERIFIED COMPLETE (EMPIRICALLY BENCHMARKED ON L40S)

---

## 1. Research Objective & Context
- Determine the highest stable and optimal micro-batch size on **NVIDIA L40S (44.44 GB VRAM)** while preserving TRUST's exact effective batch size of `80` (20 cancer-positive, 60 cancer-negative) at `1520 x 912`.
- Execute a controlled, fair benchmark comparing:
  - **Approach A (Live TRUST):** Repeated DICOM decoding, preprocessing, and frozen Mammo-CLIP EfficientNet-B5 forward passes during 20-epoch CE and TRUST training.
  - **Approach B (Feature Caching):** Extracting all 54,706 RSNA 2048-D embeddings once from the frozen encoder in `.eval()` mode, saving to disk, and training CE and TRUST heads on cached features.
- Verify numerical equivalence between live and cached predictions and project total runtime and Modal USD costs (GPU + CPU + RAM) against our `$30` credit budget.

---

## 2. Datasets, Volumes & Modal Runs
- **Source Volume**: `rsna-mammography-vol` mounted at `/data`
- **Splits Manifest**: `datasets/rsna/splits_seed42.csv` (54,706 images: `train`=38,251, `train_calib`=5,473, `val_calib`=10,982; zero patient leakage)
- **Representative Benchmark Subset**: 1,000 stratified RSNA images (`train`=700 [29 pos, 671 neg], `train_calib`=100 [4 pos, 96 neg], `val_calib`=200 [8 pos, 192 neg]; 4.10% prevalence)
- **Foundation Checkpoint**: `/data/models/mammo_clip/b5-model-best-epoch-7.tar` (strict `852/852` parity, `missing=0`, `unexpected=0`)
- **NLBS Exposure**: Zero NLBS images or labels accessed.
- **Certified Modal Run IDs**:
  - Smoke Test (Strict 852/852 on T4): `ap-49qo1JiWn1u7NGCioPcRkT`
  - L40S Micro-Batch OOM Sweep: `ap-mY4VfFocqIRVEnGIeV2CtD`
  - L40S Approach A vs. B Benchmark: `ap-KidtVEoMzokd7fsFzpMeFo`

---

## 3. Empirical Results

### 3.1 L40S Micro-Batch OOM Sweep (`ap-mY4VfFocqIRVEnGIeV2CtD`)
80 real RSNA DICOMs (`20 pos + 60 neg`) were pre-decoded once into CPU RAM and swept upward from `80` until `CUDA OOM`:

| Batch Size | Status | Peak VRAM (GB) | VRAM % | GPU Util | Step Time (s) | Throughput (img/s) | Notes |
|---|---|---|---|---|---|---|---|
| **80** | **OK** | **24.90** | **56.0%** | **100.0%** | **2.131** | **37.55** | **Optimal TRUST config (`accum_steps=1`, 19.5 GB headroom)** |
| **96** | OK | 29.85 | 67.2% | 100.0% | 2.525 | **38.02** | Fastest raw GPU throughput |
| **112** | OK | 34.81 | 78.3% | 100.0% | 2.989 | 37.48 | High memory |
| **128** | OK | 39.77 | 89.5% | 100.0% | 3.505 | 36.52 | Highest stable before OOM |
| **144** | `CUDA_OOM` | — | — | — | — | — | Tried to allocate 13.39 GiB; stopped sweep |

### 3.2 Component Timing on 1,000 RSNA Images (`ap-KidtVEoMzokd7fsFzpMeFo`)
Hardware: `NVIDIA L40S (44.44 GB)`, `6.0 vCPU`, `16.0 GiB RAM`:
- **DICOM Decoding (`decode_dicom_array`, 4 workers):** `434.40 s` (**91.5%** of compute time)
- **Preprocessing (`preprocess_pixel_array`, MONOCHROME1 invert + 1520x912 + Mammo-CLIP norm):** `13.88 s` (**2.9%**)
- **Frozen Mammo-CLIP B5 GPU Encoding (`batch=80`, `amp=True`):** `26.49 s` (**5.6%**)
- **Disk Write (`fp32` `.pt` + `RSNA_VOL.commit()`):** `1.05 s`
- **Cache Storage (`fp32`):** `7.85 MB` per 1,000 images -> **`429.6 MB` projected for all 54,706 images**

### 3.3 Live vs. Cached Numerical Equivalence Check
Evaluated on 80 images (`20 pos + 60 neg`) with identical fixed `TRUSTClassifier` head in `.eval()` mode:
- **`fp32` max abs probability difference (`|p_live - p_cached|`)**: **`0.00e+00`** (`PASS`, exact parity)

### 3.4 Full-Dataset (54,706 Images) Projection: Approach A vs. Approach B (CE + TRUST, 20 Epochs Each)
Assumed Modal rates: L40S GPU `$1.95/hr`, CPU `$0.04716/core-hr` (`6 cores`), RAM `$0.00799/GiB-hr` (`16 GiB`) -> `$2.3608/hr` combined.

| Metric | Approach A: Live TRUST (`1520x912`, `mb=80`) | Approach B: Feature Cache (`2048-D fp32`) |
|---|---|---|
| **One-Time Feature Extraction (54,706 imgs)** | N/A (repeated every epoch) | **7.22 hrs (432.92 min)** / **$17.0338** |
| **CE Head Training (20 epochs)** | 90.83 hrs / $214.43 | **0.27 min (16 sec)** / **$0.0105** |
| **TRUST Head Training (20 epochs)** | 91.01 hrs / $214.86 | **0.50 min (30 sec)** / **$0.0198** |
| **Total Wall-Clock Time** | **181.84 hrs** | **7.23 hrs (433.69 min)** |
| **Training Speed** | **3.78 img/s** (eval: 3.52 img/s) | **CE: 64,937 img/s** / **TRUST: 34,346 img/s** |
| **Peak GPU VRAM** | **26.14 GB / 44.44 GB** | ~25 GB (extract) / < 0.5 GB (head) |
| **Cache Disk Storage** | 0 MB | **429.6 MB (`fp32`)** |
| **Total Modal Cost (GPU + CPU + RAM)** | **$429.29** (GPU $354.59 + CPU $51.45 + RAM $23.25) | **$17.06** (GPU $14.09 + CPU $2.05 + RAM $0.92) |
| **Speedup & Savings** | 1x | **25.16x faster / Saves $412.23** |

---

## 4. Methodological Verification & Bottleneck Insight
1. **Exact Mathematical Equivalence**: Because TRUST keeps Mammo-CLIP frozen in `.eval()` mode with zero random image augmentation, caching `fp32` 2048-D features preserves exact predictions (`0.00e+00` diff) and identical `TRUSTLoss` / `AdamW` / `full_neg_pass` training dynamics.
2. **Infeasibility of Approach A**: At `$429.29` and `181.84 hrs`, live training would exhaust the `$30` credit budget before completing Epoch 2.
3. **Key Optimization Opportunity Before Extraction**: During Approach B extraction on L40S, **91.5% of time (`434.40 s`) was spent in CPU `pydicom` JPEG2000 decompression** while the `$1.95/hr` L40S GPU spent only `26.49 s` (`5.6%`) actually encoding. Decoupling DICOM decoding onto cheap CPU workers or sharded `T4`/`L4` workers can reduce the `$17.03` extraction cost substantially further before running Stage B4.

---

## 5. Artifacts & Code References
- `src/data/preprocessing.py`: Split `decode_dicom_array()` and `preprocess_pixel_array()`.
- `scripts/benchmark_l40s_comparison_modal.py`: Reproducible 1,000-image benchmark script.
- Modal Volume artifact: `/data/models/l40s_approach_a_vs_b_benchmark.json` and `/data/cache/bench_1000/rsna_bench1000_features_fp32.pt`.
