"""Fair L40S Training Speed & Cost Comparison: Live TRUST (Approach A) vs. Feature Caching (Approach B).

Compares on the exact same 1,000 representative RSNA images (and projects to all 54,706 RSNA images):
  - Approach A: Live DICOM decode -> preprocessing -> frozen Mammo-CLIP B5 encoder -> head (batch=80, micro_batch=80, 1520x912).
  - Approach B: One-time frozen Mammo-CLIP B5 2048-D feature extraction + disk cache -> CE & TRUST head training.

Scientific guardrails:
  - Frozen Mammo-CLIP EfficientNet-B5 in eval() mode (strict 852/852 key parity).
  - Identical preprocessing (MONOCHROME1 inversion, min-max [0,1], 1520x912 bilinear, Mammo-CLIP mean/std).
  - Identical TRUST objective (BCE + Focal(alpha=0.25, gamma=2.0) + DynamicDismissal(margin=0.05)), AdamW(lr=3e-5, wd=1e-4).
  - Zero NLBS data or labels touched.

Usage (DO NOT EXECUTE AUTOMATICALLY - for user execution):
    modal run scripts/benchmark_l40s_comparison_modal.py
"""

import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import modal

APP_NAME = "anchorshift-l40s-comparison"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
MOUNT_PATH = Path("/data")

CPU_CORES = 4.0
MEMORY_MB = 16384
MEMORY_GIB = MEMORY_MB / 1024.0

# Default Modal list rates (USD/hr); override via CLI flags if account pricing differs.
DEFAULT_L40S_GPU_RATE = 1.95
DEFAULT_CPU_CORE_RATE = 0.04716
DEFAULT_RAM_GIB_RATE = 0.00799

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "omegaconf>=2.3.0",
        "pandas>=2.2.0",
        "pydicom>=2.4.4",
        "pylibjpeg>=2.0.0",
        "pylibjpeg-libjpeg>=2.1.0",
        "pylibjpeg-openjpeg>=2.1.0",
        "python-gdcm>=3.0.24",
        "scipy>=1.12.0",
        "numpy>=1.26.0",
    )
    .add_local_python_source("src")
)


def _read_gpu_util() -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        ).strip()
        return float(out.splitlines()[0].strip())
    except Exception:
        return None


def _sample_representative_1000(df, seed: int = 42):
    """Select 1,000 representative RSNA images preserving split proportions and ~4.1% cancer prevalence."""
    import pandas as pd

    targets = {
        "train": (29, 671),       # 700
        "train_calib": (4, 96),   # 100
        "val_calib": (8, 192),    # 200
    }
    parts = []
    for split_name, (n_pos, n_neg) in targets.items():
        sub = df[df["split"] == split_name]
        pos = sub[sub["cancer"] == 1].sample(n=n_pos, random_state=seed)
        neg = sub[sub["cancer"] == 0].sample(n=n_neg, random_state=seed)
        parts.extend([pos, neg])
    return pd.concat(parts, ignore_index=True)


def _decode_and_preprocess_timed(
    rsna_raw: Path,
    rel_paths: List[str],
    num_workers: int = 4,
):
    """Decode DICOMs and preprocess separately so both phases are timed cleanly."""
    import torch
    from src.data.preprocessing import decode_dicom_array, preprocess_pixel_array

    full_paths = [rsna_raw / r for r in rel_paths]

    t0 = time.perf_counter()
    if num_workers > 1 and len(full_paths) > 1:
        with ThreadPoolExecutor(max_workers=min(num_workers, len(full_paths))) as pool:
            decoded = list(pool.map(decode_dicom_array, full_paths))
    else:
        decoded = [decode_dicom_array(p) for p in full_paths]
    t1 = time.perf_counter()

    def _prep(item):
        arr, phot = item
        return preprocess_pixel_array(
            arr,
            photometric=phot,
            target_size=(1520, 912),
            invert_monochrome1=True,
            standardize=True,
        )

    if num_workers > 1 and len(decoded) > 1:
        with ThreadPoolExecutor(max_workers=min(num_workers, len(decoded))) as pool:
            tensors = list(pool.map(_prep, decoded))
    else:
        tensors = [_prep(item) for item in decoded]
    batch_tensor = torch.stack(tensors)
    t2 = time.perf_counter()

    photometrics = [phot for _, phot in decoded]
    return batch_tensor, (t1 - t0), (t2 - t1), photometrics


def _make_head(embed_dim: int, device, seed: int = 42):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(0.3),
        nn.Linear(embed_dim, 1),
    ).to(device)


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    gpu="L40S",
    cpu=CPU_CORES,
    memory=MEMORY_MB,
    timeout=3600,
)
def run_l40s_comparison(
    sample_size: int = 1000,
    batch_size: int = 80,
    epochs: int = 20,
    num_workers: int = 4,
    seed: int = 42,
    gpu_rate: float = DEFAULT_L40S_GPU_RATE,
    cpu_rate: float = DEFAULT_CPU_CORE_RATE,
    ram_rate: float = DEFAULT_RAM_GIB_RATE,
) -> dict:
    """Execute fair Approach A vs Approach B benchmark on 1,000 RSNA images and project full 54,706-image cost."""
    import numpy as np
    import pandas as pd
    import torch

    from src.data.rsna_dataset import BalancedBatchSampler
    from src.models.trust_model import MammoCLIPEncoder
    from src.models.mammo_clip_loader import load_and_verify_mammo_clip
    from src.models.losses import TRUSTLoss
    from src.training.calibration import compute_target_recall_threshold

    assert sample_size == 1000, "Representative benchmark subset is fixed at 1,000 images."
    assert batch_size == 80, "TRUST effective & micro batch size is fixed at 80."

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    cache_dir = MOUNT_PATH / "cache" / "bench_1000"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] GPU={gpu_name} ({total_vram_gb:.2f} GB) | CPU={CPU_CORES} | RAM={MEMORY_GIB:.1f} GiB", flush=True)
    if not splits_csv.exists():
        raise FileNotFoundError(f"Expected RSNA splits manifest at {splits_csv}")
    df_full = pd.read_csv(splits_csv)

    # Full-dataset counts for Stage 8 projection
    full_train = df_full[df_full["split"] == "train"]
    full_train_calib = df_full[df_full["split"] == "train_calib"]
    full_val_calib = df_full[df_full["split"] == "val_calib"]
    full_counts = {
        "total_images": int(len(df_full)),
        "train_images": int(len(full_train)),
        "train_pos": int((full_train["cancer"] == 1).sum()),
        "train_neg": int((full_train["cancer"] == 0).sum()),
        "train_calib_images": int(len(full_train_calib)),
        "train_calib_pos": int((full_train_calib["cancer"] == 1).sum()),
        "val_calib_images": int(len(full_val_calib)),
        "batches_per_epoch_full_neg_pass": int(math.ceil((full_train["cancer"] == 0).sum() / 60)),
    }

    bench_df = _sample_representative_1000(df_full, seed=seed)
    bench_train = bench_df[bench_df["split"] == "train"].reset_index(drop=True)
    bench_train_calib = bench_df[bench_df["split"] == "train_calib"].reset_index(drop=True)
    bench_val_calib = bench_df[bench_df["split"] == "val_calib"].reset_index(drop=True)
    print(
        f"[data] 1,000-img subset: train={len(bench_train)} ({int((bench_train['cancer']==1).sum())} pos), "
        f"train_calib={len(bench_train_calib)}, val_calib={len(bench_val_calib)}",
        flush=True,
    )

    # Load frozen Mammo-CLIP EfficientNet-B5 (strict 852/852)
    encoder = MammoCLIPEncoder(checkpoint_path=None, freeze=False, model_name="efficientnet-b5")
    diag = load_and_verify_mammo_clip(encoder, mammo_clip_tar, strict=True)
    assert diag["matched_keys_count"] == 852 and diag["missing_keys_count"] == 0 and diag["unexpected_keys_count"] == 0
    encoder.freeze_encoder()
    encoder = encoder.to(device)
    encoder.eval()
    print("[model] Mammo-CLIP B5 frozen & verified (852/852).", flush=True)

    # =========================================================================
    # PHASE 1: Approach B One-Time Feature Extraction (1,000 images) + Component Timing
    # =========================================================================
    print("\n[phase 1/4] Extracting 2048-D features for 1,000 images (timing decode / preprocess / encode)...", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Warmup 4 images so CUDA kernels & cuDNN autotune are settled before timing
    warmup_x, _, _, _ = _decode_and_preprocess_timed(rsna_raw, bench_df["rel_path"].iloc[:4].tolist(), num_workers=num_workers)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        _ = encoder(warmup_x.to(device))
    torch.cuda.synchronize()
    del warmup_x
    torch.cuda.reset_peak_memory_stats(device)

    all_feats = []
    all_photometrics = []
    total_decode_s = 0.0
    total_preproc_s = 0.0
    total_encode_s = 0.0
    extract_utils = []

    rel_paths = bench_df["rel_path"].tolist()
    for start in range(0, len(rel_paths), batch_size):
        chunk = rel_paths[start : start + batch_size]
        bx_cpu, dt_dec, dt_prep, phots = _decode_and_preprocess_timed(rsna_raw, chunk, num_workers=num_workers)
        all_photometrics.extend(phots)

        torch.cuda.synchronize()
        t_enc0 = time.perf_counter()
        bx = bx_cpu.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            feats = encoder(bx)
        torch.cuda.synchronize()
        t_enc1 = time.perf_counter()

        u = _read_gpu_util()
        if u is not None:
            extract_utils.append(u)

        all_feats.append(feats.float().cpu())
        total_decode_s += dt_dec
        total_preproc_s += dt_prep
        total_encode_s += (t_enc1 - t_enc0)

    extract_peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    feats_fp32 = torch.cat(all_feats, dim=0)  # (1000, 2048)

    # Save cached features (fp32 canonical + fp16 size check)
    cache_fp32_path = cache_dir / "rsna_bench1000_features_fp32.pt"
    cache_fp16_path = cache_dir / "rsna_bench1000_features_fp16.pt"
    payload = {
        "features": feats_fp32,
        "labels": torch.tensor(bench_df["cancer"].values, dtype=torch.float32),
        "split": bench_df["split"].tolist(),
        "patient_id": bench_df["patient_id"].astype(str).tolist(),
        "image_id": bench_df["image_id"].astype(str).tolist(),
    }

    t_w0 = time.perf_counter()
    torch.save(payload, str(cache_fp32_path))
    disk_save_only_s = time.perf_counter() - t_w0
    t_c0 = time.perf_counter()
    RSNA_VOL.commit()
    vol_commit_s = time.perf_counter() - t_c0
    disk_write_fp32_s = disk_save_only_s + vol_commit_s

    payload_fp16 = dict(payload)
    payload_fp16["features"] = feats_fp32.half()
    torch.save(payload_fp16, str(cache_fp16_path))
    RSNA_VOL.commit()

    fp32_mb = cache_fp32_path.stat().st_size / (1024**2)
    fp16_mb = cache_fp16_path.stat().st_size / (1024**2)

    t_r0 = time.perf_counter()
    reloaded_payload = torch.load(str(cache_fp32_path), map_location="cpu", weights_only=False)
    disk_read_fp32_s = time.perf_counter() - t_r0

    total_extract_compute_s = total_decode_s + total_preproc_s + total_encode_s
    print(
        f"[extract] 1,000 imgs | decode={total_decode_s:.2f}s | preproc={total_preproc_s:.2f}s | "
        f"encode={total_encode_s:.2f}s | disk_write={disk_write_fp32_s:.2f}s | size_fp32={fp32_mb:.2f} MB",
        flush=True,
    )

    # =========================================================================
    # PHASE 2: Numerical Equivalence Check (Live Encoder vs Cached Features)
    # =========================================================================
    print("\n[phase 2/4] Verifying numerical equivalence (live encoder vs cached fp32/fp16)...", flush=True)
    eq_pos = bench_df[bench_df["cancer"] == 1].index.tolist()[:20]
    eq_neg = bench_df[bench_df["cancer"] == 0].index.tolist()[:60]
    eq_idx = eq_pos + eq_neg
    eq_paths = [rel_paths[i] for i in eq_idx]

    eq_head = _make_head(encoder.embed_dim, device, seed=seed)
    eq_head.eval()

    eq_bx_cpu, _, _, eq_phots = _decode_and_preprocess_timed(rsna_raw, eq_paths, num_workers=num_workers)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        live_feats = encoder(eq_bx_cpu.to(device)).float()
        live_logits = eq_head(live_feats)
        live_probs = torch.sigmoid(live_logits)

    cached_feats_fp32 = reloaded_payload["features"][eq_idx].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        cached_logits_fp32 = eq_head(cached_feats_fp32)
        cached_probs_fp32 = torch.sigmoid(cached_logits_fp32)

    cached_feats_fp16 = reloaded_payload["features"][eq_idx].half().float().to(device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        cached_logits_fp16 = eq_head(cached_feats_fp16)
        cached_probs_fp16 = torch.sigmoid(cached_logits_fp16)

    eq_metrics = {
        "sample_count": len(eq_idx),
        "monochrome1_in_sample": sum(1 for p in eq_phots if p == "MONOCHROME1"),
        "monochrome2_in_sample": sum(1 for p in eq_phots if p == "MONOCHROME2"),
        "fp32_max_abs_feature_diff": float((live_feats - cached_feats_fp32).abs().max().item()),
        "fp32_max_abs_logit_diff": float((live_logits - cached_logits_fp32).abs().max().item()),
        "fp32_max_abs_prob_diff": float((live_probs - cached_probs_fp32).abs().max().item()),
        "fp16_max_abs_feature_diff": float((live_feats - cached_feats_fp16).abs().max().item()),
        "fp16_max_abs_logit_diff": float((live_logits - cached_logits_fp16).abs().max().item()),
        "fp16_max_abs_prob_diff": float((live_probs - cached_probs_fp16).abs().max().item()),
        "fp32_allclose_1e5": bool(torch.allclose(live_probs, cached_probs_fp32, atol=1e-5)),
        "fp32_allclose_1e3": bool(torch.allclose(live_probs, cached_probs_fp32, atol=1e-3)),
    }
    assert eq_metrics["fp32_allclose_1e3"], f"Numerical parity failed (>1e-3): {eq_metrics}"
    print(f"[equiv] PASS | fp32 max_abs_prob_diff={eq_metrics['fp32_max_abs_prob_diff']:.2e}", flush=True)

    # =========================================================================
    # PHASE 3: Approach A Benchmark (Live Training, batch=80, micro_batch=80)
    # =========================================================================
    print("\n[phase 3/4] Benchmarking Approach A (live DICOM -> encoder -> head, batch=80)...", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    sampler_a = BalancedBatchSampler(
        bench_train["cancer"].values, batch_size=80, n_pos=20, seed=seed, epoch_policy="full_neg_pass"
    )
    batches_a = list(sampler_a)  # ceil(671 / 60) = 12 batches of 80
    head_a = _make_head(encoder.embed_dim, device, seed=seed)
    opt_a = torch.optim.AdamW(head_a.parameters(), lr=3e-5, weight_decay=1e-4)
    scaler_a = torch.amp.GradScaler("cuda")
    crit_a = TRUSTLoss(baseline="trust", margin=0.05, lambda_focal=1.0, lambda_dismiss=1.0)

    head_a.train()
    a_step_times = []
    a_utils = []
    for b_idx, batch_indices in enumerate(batches_a):
        sub = bench_train.iloc[batch_indices]
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        bx_cpu, _, _, _ = _decode_and_preprocess_timed(rsna_raw, sub["rel_path"].tolist(), num_workers=num_workers)
        bx = bx_cpu.to(device, non_blocking=True)
        by = torch.tensor(sub["cancer"].values, dtype=torch.float32).unsqueeze(1).to(device, non_blocking=True)

        opt_a.zero_grad()
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                feats = encoder(bx)
            logits = head_a(feats)
            loss, _ = crit_a(logits, by, tau_t=0.5)
        scaler_a.scale(loss).backward()
        scaler_a.unscale_(opt_a)
        torch.nn.utils.clip_grad_norm_(head_a.parameters(), 5.0)
        scaler_a.step(opt_a)
        scaler_a.update()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if b_idx > 0:  # skip batch 0 as warmup
            a_step_times.append(dt)
            u = _read_gpu_util()
            if u is not None:
                a_utils.append(u)

    # Time live eval on bench_val_calib (200 images)
    head_a.eval()
    val_paths_a = bench_val_calib["rel_path"].tolist()
    torch.cuda.synchronize()
    t_va0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, len(val_paths_a), batch_size):
            vx_cpu, _, _, _ = _decode_and_preprocess_timed(
                rsna_raw, val_paths_a[s : s + batch_size], num_workers=num_workers
            )
            with torch.amp.autocast("cuda"):
                _ = torch.sigmoid(head_a(encoder(vx_cpu.to(device))))
    torch.cuda.synchronize()
    a_eval_s = time.perf_counter() - t_va0

    a_peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    a_sec_per_batch = float(np.mean(a_step_times))
    a_imgs_per_sec = batch_size / max(a_sec_per_batch, 1e-6)
    a_eval_imgs_per_sec = len(val_paths_a) / max(a_eval_s, 1e-6)
    print(
        f"[approach A] {a_sec_per_batch:.3f}s/batch80 ({a_imgs_per_sec:.2f} img/s) | "
        f"eval={a_eval_imgs_per_sec:.2f} img/s | VRAM={a_peak_vram_gb:.2f} GB",
        flush=True,
    )

    # =========================================================================
    # PHASE 4: Approach B Benchmark (Cached CE + Cached TRUST, 20 epochs each)
    # =========================================================================
    print("\n[phase 4/4] Benchmarking Approach B (cached 2048-D CE & TRUST heads, 20 epochs each)...", flush=True)
    cached_all_feats = reloaded_payload["features"]
    cached_all_labels = reloaded_payload["labels"]
    splits_arr = np.array(reloaded_payload["split"])

    tr_mask = splits_arr == "train"
    tc_mask = splits_arr == "train_calib"
    vc_mask = splits_arr == "val_calib"

    tr_feats, tr_labels = cached_all_feats[tr_mask].to(device), cached_all_labels[tr_mask].to(device)
    tc_feats, tc_labels = cached_all_feats[tc_mask].to(device), cached_all_labels[tc_mask].to(device)
    vc_feats = cached_all_feats[vc_mask].to(device)

    tc_pos_feats = tc_feats[tc_labels == 1]

    def _run_cached_20_epochs(mode: str) -> dict:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        head_b = _make_head(encoder.embed_dim, device, seed=seed)
        opt_b = torch.optim.AdamW(head_b.parameters(), lr=3e-5, weight_decay=1e-4)
        crit_b = TRUSTLoss(baseline=mode, margin=0.05, lambda_focal=1.0, lambda_dismiss=1.0)
        sampler_b = BalancedBatchSampler(
            tr_labels.cpu().numpy(), batch_size=80, n_pos=20, seed=seed, epoch_policy="full_neg_pass"
        )
        tau_t = 0.5
        batch_times = []
        eval_times = []
        utils = []

        # Warmup 1 step
        w_idx = next(iter(sampler_b))
        opt_b.zero_grad()
        wl, _ = crit_b(head_b(tr_feats[w_idx]), tr_labels[w_idx].unsqueeze(1), tau_t=tau_t)
        wl.backward()
        opt_b.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        t_run0 = time.perf_counter()
        for _ep in range(epochs):
            head_b.train()
            for b_indices in sampler_b:
                idx_t = torch.tensor(b_indices, dtype=torch.long, device=device)
                torch.cuda.synchronize()
                tb0 = time.perf_counter()
                opt_b.zero_grad()
                logits = head_b(tr_feats[idx_t])
                loss, _ = crit_b(logits, tr_labels[idx_t].unsqueeze(1), tau_t=tau_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_b.parameters(), 5.0)
                opt_b.step()
                torch.cuda.synchronize()
                batch_times.append(time.perf_counter() - tb0)

            u = _read_gpu_util()
            if u is not None:
                utils.append(u)

            head_b.eval()
            torch.cuda.synchronize()
            te0 = time.perf_counter()
            with torch.no_grad():
                if mode == "trust" and len(tc_pos_feats) > 0:
                    pos_probs = torch.sigmoid(head_b(tc_pos_feats)).squeeze(1).cpu().numpy()
                    tau_t = compute_target_recall_threshold(pos_probs, target_recall=0.98)
                _ = torch.sigmoid(head_b(vc_feats))
            torch.cuda.synchronize()
            eval_times.append(time.perf_counter() - te0)

        total_20ep_s = time.perf_counter() - t_run0
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        sec_per_b = float(np.mean(batch_times))
        return {
            "mode": mode,
            "bench_20ep_total_sec": round(total_20ep_s, 4),
            "sec_per_batch_80": sec_per_b,
            "imgs_per_sec": batch_size / max(sec_per_b, 1e-9),
            "eval_sec_per_epoch": float(np.mean(eval_times)),
            "eval_imgs_per_sec": (len(vc_feats) + (len(tc_pos_feats) if mode == "trust" else 0)) / max(float(np.mean(eval_times)), 1e-9),
            "peak_vram_gb": round(peak_gb, 4),
            "gpu_util_pct": round(float(np.mean(utils)), 1) if utils else None,
        }

    ce_stats = _run_cached_20_epochs("ce")
    trust_stats = _run_cached_20_epochs("trust")
    print(
        f"[approach B] CE={ce_stats['imgs_per_sec']:.0f} img/s ({ce_stats['bench_20ep_total_sec']}s for 20ep on 1k) | "
        f"TRUST={trust_stats['imgs_per_sec']:.0f} img/s ({trust_stats['bench_20ep_total_sec']}s for 20ep on 1k)",
        flush=True,
    )

    # =========================================================================
    # STAGE 8-10: Full-Dataset (54,706 Images) Runtime & Modal Cost Projections
    # =========================================================================
    total_hourly_rate = gpu_rate + CPU_CORES * cpu_rate + MEMORY_GIB * ram_rate
    bpe_full = full_counts["batches_per_epoch_full_neg_pass"]
    n_full = full_counts["total_images"]
    scale_full = n_full / float(sample_size)

    # Approach A full projections (CE + TRUST = 2 * 20 epochs)
    a_train_sec_per_ep = bpe_full * a_sec_per_batch
    a_eval_imgs_per_ep_trust = full_counts["train_calib_pos"] + full_counts["val_calib_images"]
    a_eval_imgs_per_ep_ce = full_counts["val_calib_images"]
    a_ce_20ep_sec = epochs * (a_train_sec_per_ep + a_eval_imgs_per_ep_ce / a_eval_imgs_per_sec)
    a_trust_20ep_sec = epochs * (a_train_sec_per_ep + a_eval_imgs_per_ep_trust / a_eval_imgs_per_sec)
    a_total_sec = a_ce_20ep_sec + a_trust_20ep_sec

    # Approach B full projections (1x extraction of 54,706 + CE 20ep + TRUST 20ep)
    b_extract_compute_full_sec = total_extract_compute_s * scale_full
    b_disk_write_full_sec = disk_save_only_s * scale_full + vol_commit_s
    b_extract_total_full_sec = b_extract_compute_full_sec + b_disk_write_full_sec

    b_ce_20ep_full_sec = disk_read_fp32_s * scale_full + epochs * (
        bpe_full * ce_stats["sec_per_batch_80"] + a_eval_imgs_per_ep_ce / ce_stats["eval_imgs_per_sec"]
    )
    b_trust_20ep_full_sec = disk_read_fp32_s * scale_full + epochs * (
        bpe_full * trust_stats["sec_per_batch_80"] + a_eval_imgs_per_ep_trust / trust_stats["eval_imgs_per_sec"]
    )
    b_total_sec = b_extract_total_full_sec + b_ce_20ep_full_sec + b_trust_20ep_full_sec

    def _cost_breakdown(seconds: float) -> dict:
        hrs = seconds / 3600.0
        g = hrs * gpu_rate
        c = hrs * CPU_CORES * cpu_rate
        r = hrs * MEMORY_GIB * ram_rate
        return {
            "hours": round(hrs, 4),
            "minutes": round(seconds / 60.0, 2),
            "gpu_usd": round(g, 4),
            "cpu_usd": round(c, 4),
            "ram_usd": round(r, 4),
            "total_usd": round(g + c + r, 4),
        }

    report = {
        "hardware": {
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "cpu_cores": CPU_CORES,
            "ram_gib": MEMORY_GIB,
            "hourly_rates_usd": {
                "gpu": gpu_rate,
                "cpu_per_core": cpu_rate,
                "ram_per_gib": ram_rate,
                "combined_total": round(total_hourly_rate, 4),
            },
        },
        "full_dataset_counts": full_counts,
        "component_timing_1000_images": {
            "dicom_decode_sec": round(total_decode_s, 3),
            "preprocessing_sec": round(total_preproc_s, 3),
            "encoder_extract_sec": round(total_encode_s, 3),
            "disk_write_fp32_sec": round(disk_write_fp32_s, 3),
            "disk_read_fp32_sec": round(disk_read_fp32_s, 3),
            "per_image_ms": {
                "decode": round((total_decode_s / sample_size) * 1000.0, 3),
                "preprocess": round((total_preproc_s / sample_size) * 1000.0, 3),
                "encode": round((total_encode_s / sample_size) * 1000.0, 3),
            },
            "photometrics_1000": {
                "MONOCHROME1": sum(1 for p in all_photometrics if p == "MONOCHROME1"),
                "MONOCHROME2": sum(1 for p in all_photometrics if p == "MONOCHROME2"),
            },
        },
        "storage_requirements": {
            "bench_1000_fp32_mb": round(fp32_mb, 2),
            "bench_1000_fp16_mb": round(fp16_mb, 2),
            "projected_54706_fp32_mb": round(fp32_mb * scale_full, 2),
            "projected_54706_fp16_mb": round(fp16_mb * scale_full, 2),
        },
        "numerical_equivalence": eq_metrics,
        "approach_a_measured_1000": {
            "sec_per_batch_80": round(a_sec_per_batch, 3),
            "train_imgs_per_sec": round(a_imgs_per_sec, 2),
            "eval_imgs_per_sec": round(a_eval_imgs_per_sec, 2),
            "peak_vram_gb": round(a_peak_vram_gb, 2),
            "gpu_util_pct": round(float(np.mean(a_utils)), 1) if a_utils else None,
        },
        "approach_b_measured_1000": {
            "extract_peak_vram_gb": round(extract_peak_vram_gb, 2),
            "extract_gpu_util_pct": round(float(np.mean(extract_utils)), 1) if extract_utils else None,
            "ce_head": {k: round(v, 4) if isinstance(v, float) else v for k, v in ce_stats.items()},
            "trust_head": {k: round(v, 4) if isinstance(v, float) else v for k, v in trust_stats.items()},
        },
        "full_dataset_projection_ce_plus_trust_20ep_each": {
            "approach_a": {
                "ce_20ep": _cost_breakdown(a_ce_20ep_sec),
                "trust_20ep": _cost_breakdown(a_trust_20ep_sec),
                "total": _cost_breakdown(a_total_sec),
            },
            "approach_b": {
                "one_time_extraction_54706": _cost_breakdown(b_extract_total_full_sec),
                "cached_ce_20ep": _cost_breakdown(b_ce_20ep_full_sec),
                "cached_trust_20ep": _cost_breakdown(b_trust_20ep_full_sec),
                "total": _cost_breakdown(b_total_sec),
            },
            "speedup_factor_total": round(a_total_sec / max(b_total_sec, 1e-6), 2),
            "savings_usd": round(_cost_breakdown(a_total_sec)["total_usd"] - _cost_breakdown(b_total_sec)["total_usd"], 2),
        },
    }

    out_path = MOUNT_PATH / "models" / "l40s_approach_a_vs_b_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    RSNA_VOL.commit()

    proj = report["full_dataset_projection_ce_plus_trust_20ep_each"]
    print("\n" + "=" * 78, flush=True)
    print("L40S APPROACH A (LIVE) vs APPROACH B (FEATURE CACHE) - 54,706 IMGS / 20 EP (CE+TRUST)", flush=True)
    print("=" * 78, flush=True)
    print(f"Approach A Total: {proj['approach_a']['total']['hours']:.2f} hrs | ${proj['approach_a']['total']['total_usd']:.2f} | VRAM {report['approach_a_measured_1000']['peak_vram_gb']} GB | {report['approach_a_measured_1000']['train_imgs_per_sec']} img/s", flush=True)
    print(f"Approach B Extract (1x): {proj['approach_b']['one_time_extraction_54706']['minutes']:.2f} min | ${proj['approach_b']['one_time_extraction_54706']['total_usd']:.4f} | Storage {report['storage_requirements']['projected_54706_fp32_mb']:.1f} MB (fp32)", flush=True)
    print(f"Approach B CE+TRUST (2x20ep): {(proj['approach_b']['cached_ce_20ep']['minutes'] + proj['approach_b']['cached_trust_20ep']['minutes']):.2f} min | CE {ce_stats['imgs_per_sec']:.0f} img/s | TRUST {trust_stats['imgs_per_sec']:.0f} img/s", flush=True)
    print(f"Approach B Total: {proj['approach_b']['total']['hours']:.4f} hrs ({proj['approach_b']['total']['minutes']:.2f} min) | ${proj['approach_b']['total']['total_usd']:.4f}", flush=True)
    print(f"Speedup: {proj['speedup_factor_total']}x | Savings: ${proj['savings_usd']:.2f} (out of $30 credit)", flush=True)
    print("=" * 78, flush=True)
    return report


@app.local_entrypoint()
def main(
    gpu_rate: float = DEFAULT_L40S_GPU_RATE,
    cpu_rate: float = DEFAULT_CPU_CORE_RATE,
    ram_rate: float = DEFAULT_RAM_GIB_RATE,
) -> None:
    print("Launching fair L40S Approach A vs Approach B benchmark (1,000 RSNA images)...")
    report = run_l40s_comparison.remote(
        gpu_rate=gpu_rate,
        cpu_rate=cpu_rate,
        ram_rate=ram_rate,
    )
    print("\nJSON Report Summary Saved to /data/models/l40s_approach_a_vs_b_benchmark.json")
    print(json.dumps(report["full_dataset_projection_ce_plus_trust_20ep_each"], indent=2))
