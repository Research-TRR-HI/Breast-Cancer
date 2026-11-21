"""Paper 1 — Small Cost-Controlled Modal L40S Smoke Tests & Throughput Benchmarks.

STRICTLY FOR SMALL VALIDATION & TIMING BENCHMARKS ONLY (NOT full training/inference).
Measures actual CPU/GPU preprocessing + inference/training throughput on real RSNA/NLBS
DICOMs and existing Stage-1 1520x912 PNG caches on NVIDIA L40S, returning exact
cost projections per model.

Usage:
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/paper1_modal_smoke_and_bench.py
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-l40s-smoke-and-bench"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
NLBS_VOL = modal.Volume.from_name("nlbs-mammography-vol")

RSNA_MOUNT = Path("/rsna_vol")
NLBS_MOUNT = Path("/nlbs_vol")

# L40S + 6 CPU + 16 GiB RAM combined hourly price on Modal (verified 2026-11)
L40S_COMBINED_HOURLY_USD = 2.3608

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES, "imageio==2.37.0", "tqdm==4.67.1")
    .add_local_python_source("src")
    .add_local_dir("external/GMIC/src", remote_path="/root/external/GMIC/src")
    .add_local_dir(
        "external/breast_cancer_classifier/src",
        remote_path="/root/external/breast_cancer_classifier/src",
    )
    .add_local_dir("external/dinov2/dinov2", remote_path="/root/external/dinov2/dinov2")
    .add_local_file("external/dinov2/hubconf.py", remote_path="/root/external/dinov2/hubconf.py")
    .add_local_file("datasets/rsna/splits_seed42.csv", remote_path="/root/datasets/rsna/splits_seed42.csv")
    .add_local_file(
        "datasets/paper1/rsna_val_calib_4view_selection.csv",
        remote_path="/root/datasets/paper1/rsna_val_calib_4view_selection.csv",
    )
    .add_local_file(
        "datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
        remote_path="/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
    )
    .add_local_file(
        "datasets/nlbs/inference_manifest_stage_c.csv",
        remote_path="/root/datasets/nlbs/inference_manifest_stage_c.csv",
    )
)


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=1800,
)
def run_all_benchmarks_l40s(
    n_gmic_imgs: int = 20,
    n_nyu_exams: int = 8,
    n_dino_imgs: int = 200,
    n_resnet_batches: int = 4,
    resnet_batch_size: int = 64,
) -> Dict[str, Any]:
    """Run small, representative L40S smoke & timing benchmarks for GMIC, NYU Four-View, DINOv2, and ResNet-50."""
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn

    from src.paper1.dinov2_model import (
        DINOv2CancerHead,
        build_frozen_dinov2_vits14,
        prepare_dinov2_tensor_from_uint16,
    )
    from src.paper1.gmic_adapter import (
        GMIC_INPUT_SIZE,
        build_gmic_model,
        prepare_dicom_uint16_for_nyukat,
        preprocess_uint16_for_nyukat,
        score_gmic_preprocessed_tensor,
    )
    from src.paper1.nyu_fourview_adapter import (
        NYU_VIEWS,
        build_nyu_fourview_model,
        extract_nyu_cropped_view,
        score_nyu_fourview_exam,
    )
    from src.paper1.resnet50_model import (
        build_resnet50_classifier,
        prepare_resnet50_tensor_from_uint16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    results: Dict[str, Any] = {"gpu_name": gpu_name, "hourly_rate_usd": L40S_COMBINED_HOURLY_USD}

    rsna_df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
    nlbs_inf = pd.read_csv("/root/datasets/nlbs/inference_manifest_stage_c.csv")
    nlbs_4v = pd.read_csv("/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv")
    rsna_4v = pd.read_csv("/root/datasets/paper1/rsna_val_calib_4view_selection.csv")

    models_dir = RSNA_MOUNT / "models" / "paper1"
    rsna_s1_dir = RSNA_MOUNT / "cache" / "stage1_rsna_uint16" / "png_1520x912"
    nlbs_s1_dir = RSNA_MOUNT / "cache" / "stage1_nlbs_uint16" / "png_1520x912"

    # ------------------------------------------------------------------
    # 1. GMIC Benchmark (10 RSNA val_calib + 10 NLBS raw DICOMs -> 2944x1920)
    # ------------------------------------------------------------------
    gmic_ckpt = models_dir / "sample_model_1.p"
    gmic_model, gmic_mods = build_gmic_model(
        gmic_ckpt, model_index="1", device=device, gmic_repo_dir=Path("/root/external/GMIC")
    )
    vc_sample = rsna_df[rsna_df["split"] == "val_calib"].head(n_gmic_imgs // 2)
    nl_sample = nlbs_inf.head(n_gmic_imgs - len(vc_sample))
    gmic_items = [
        (RSNA_MOUNT / "rsna" / "raw" / r.rel_path, f"{r.laterality}-{r.view}")
        for r in vc_sample.itertuples(index=False)
    ] + [
        (NLBS_MOUNT / "nlbs" / "extracted" / r.dicom_rel_path, f"{r.laterality}-{r.view_position}")
        for r in nl_sample.itertuples(index=False)
    ]

    t_g0 = time.perf_counter()
    gmic_prepped = []
    gmic_fallbacks = 0
    for dpath, vs in gmic_items:
        arr_u16, _ = prepare_dicom_uint16_for_nyukat(dpath, vs.split("-")[0])
        proc, meta = preprocess_uint16_for_nyukat(arr_u16, vs, GMIC_INPUT_SIZE, gmic_mods)
        if meta["crop_fallback"]:
            gmic_fallbacks += 1
        gmic_prepped.append(proc)
    t_g_prep = time.perf_counter() - t_g0

    # Warmup 1 image
    _ = score_gmic_preprocessed_tensor(gmic_model, gmic_prepped[0], device)
    torch.cuda.synchronize()
    t_g1 = time.perf_counter()
    gmic_scores = [score_gmic_preprocessed_tensor(gmic_model, p, device)[1] for p in gmic_prepped]
    torch.cuda.synchronize()
    t_g_gpu = time.perf_counter() - t_g1
    gmic_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    torch.cuda.reset_peak_memory_stats()

    del gmic_model
    torch.cuda.empty_cache()

    gmic_gpu_rate = len(gmic_prepped) / max(t_g_gpu, 1e-6)
    gmic_full_imgs = 10982 + 26988  # RSNA val_calib + NLBS
    gmic_proj_gpu_hrs = gmic_full_imgs / gmic_gpu_rate / 3600.0
    # CPU crop prep on cheap 16-worker CPU containers (~$0.126/hr each)
    gmic_prep_imgs_per_sec_1core = len(gmic_prepped) / max(t_g_prep, 1e-6)
    gmic_proj_cpu_hrs = gmic_full_imgs / gmic_prep_imgs_per_sec_1core / 3600.0
    results["gmic_pretrained"] = {
        "n_tested_images": len(gmic_prepped),
        "crop_fallbacks": gmic_fallbacks,
        "all_scores_finite": bool(np.all(np.isfinite(gmic_scores))),
        "sample_scores_head": [round(s, 5) for s in gmic_scores[:4]],
        "prep_sec": round(t_g_prep, 3),
        "gpu_infer_sec": round(t_g_gpu, 3),
        "gpu_imgs_per_sec": round(gmic_gpu_rate, 2),
        "prep_imgs_per_sec_single_thread": round(gmic_prep_imgs_per_sec_1core, 2),
        "peak_vram_gb": round(gmic_vram_gb, 3),
        "full_target_images": gmic_full_imgs,
        "projected_l40s_gpu_hours": round(gmic_proj_gpu_hrs, 3),
        "projected_cpu_core_hours": round(gmic_proj_cpu_hrs, 2),
        "projected_total_usd_with_cpu_staging": round(
            gmic_proj_gpu_hrs * L40S_COMBINED_HOURLY_USD + gmic_proj_cpu_hrs * 0.06, 2
        ),
    }

    # ------------------------------------------------------------------
    # 2. NYU Four-View Benchmark (4 RSNA val_calib + 4 NLBS 4-view exams)
    # ------------------------------------------------------------------
    nyu_ckpt = models_dir / "sample_image_model.p"
    nyu_model, nyu_mods = build_nyu_fourview_model(
        nyu_ckpt, device=device, nyu_repo_dir=Path("/root/external/breast_cancer_classifier")
    )
    rsna_study_ids = rsna_4v["patient_id"].unique()[: max(1, n_nyu_exams // 2)]
    nlbs_study_ids = nlbs_4v["StudyInstanceUID"].unique()[: max(1, n_nyu_exams - len(rsna_study_ids))]

    exam_batches: List[Dict[str, Dict[str, Any]]] = []
    nyu_fallbacks = 0
    t_n0 = time.perf_counter()
    for pid in rsna_study_ids:
        sub = rsna_4v[rsna_4v["patient_id"] == pid]
        vc_map = {}
        for r in sub.itertuples(index=False):
            out_c = extract_nyu_cropped_view(RSNA_MOUNT / "rsna" / "raw" / r.rel_path, r.view_slot, nyu_mods)
            if out_c["crop_fallback"]:
                nyu_fallbacks += 1
            vc_map[r.view_slot] = out_c
        exam_batches.append(vc_map)
    for suid in nlbs_study_ids:
        sub = nlbs_4v[nlbs_4v["StudyInstanceUID"] == suid]
        vc_map = {}
        for r in sub.itertuples(index=False):
            out_c = extract_nyu_cropped_view(NLBS_MOUNT / "nlbs" / "extracted" / r.dicom_rel_path, r.view_slot, nyu_mods)
            if out_c["crop_fallback"]:
                nyu_fallbacks += 1
            vc_map[r.view_slot] = out_c
        exam_batches.append(vc_map)
    t_n_prep = time.perf_counter() - t_n0

    _ = score_nyu_fourview_exam(nyu_model, nyu_mods, exam_batches[0], device, num_tta_epochs=10, seed=0)
    torch.cuda.synchronize()
    t_n1 = time.perf_counter()
    nyu_scores = [
        score_nyu_fourview_exam(nyu_model, nyu_mods, eb, device, num_tta_epochs=10, seed=0)["exam_score"]
        for eb in exam_batches
    ]
    torch.cuda.synchronize()
    t_n_gpu = time.perf_counter() - t_n1
    nyu_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    torch.cuda.reset_peak_memory_stats()

    del nyu_model
    torch.cuda.empty_cache()

    nyu_gpu_rate = len(exam_batches) / max(t_n_gpu, 1e-6)
    nyu_full_exams = 2382 + 6184
    nyu_proj_gpu_hrs = nyu_full_exams / nyu_gpu_rate / 3600.0
    nyu_prep_exams_per_sec = len(exam_batches) / max(t_n_prep, 1e-6)
    nyu_proj_cpu_hrs = nyu_full_exams / nyu_prep_exams_per_sec / 3600.0
    results["nyu_fourview_pretrained"] = {
        "n_tested_exams": len(exam_batches),
        "tta_epochs": 10,
        "crop_fallbacks": nyu_fallbacks,
        "all_scores_finite": bool(np.all(np.isfinite(nyu_scores))),
        "sample_exam_scores": [round(s, 5) for s in nyu_scores[:4]],
        "prep_sec": round(t_n_prep, 3),
        "gpu_infer_sec": round(t_n_gpu, 3),
        "gpu_exams_per_sec": round(nyu_gpu_rate, 3),
        "peak_vram_gb": round(nyu_vram_gb, 3),
        "full_target_exams": nyu_full_exams,
        "projected_l40s_gpu_hours": round(nyu_proj_gpu_hrs, 3),
        "projected_cpu_core_hours": round(nyu_proj_cpu_hrs, 2),
        "projected_total_usd_with_cpu_staging": round(
            nyu_proj_gpu_hrs * L40S_COMBINED_HOURLY_USD + nyu_proj_cpu_hrs * 0.06, 2
        ),
    }

    # ------------------------------------------------------------------
    # 3. DINOv2 ViT-S/14 Benchmark (200 existing Stage-1 1520x912 PNGs)
    # ------------------------------------------------------------------
    dino_ckpt = models_dir / "dinov2_vits14_pretrain.pth"
    dino_enc = build_frozen_dinov2_vits14(
        checkpoint_path=dino_ckpt, dinov2_repo_dir=Path("/root/external/dinov2"), device=device
    )
    dino_head = DINOv2CancerHead(embed_dim=384).to(device)
    rsna_ids_bench = rsna_df["image_id"].astype(str).head(n_dino_imgs // 2).tolist()
    nlbs_ids_bench = nlbs_inf["image_uid"].astype(str).head(n_dino_imgs - len(rsna_ids_bench)).tolist()
    dino_png_paths = [rsna_s1_dir / f"{i}.png" for i in rsna_ids_bench] + [
        nlbs_s1_dir / f"{i}.png" for i in nlbs_ids_bench
    ]

    def _load_dino_t(p: Path) -> torch.Tensor:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(p)
        return prepare_dinov2_tensor_from_uint16(arr)

    t_d0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        dino_tensors = list(pool.map(_load_dino_t, dino_png_paths))
    dino_stack = torch.stack(dino_tensors, dim=0)
    t_d_io = time.perf_counter() - t_d0

    # Warmup + timed AMP extraction
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        _ = dino_enc(dino_stack[:16].to(device))
    torch.cuda.synchronize()
    t_d1 = time.perf_counter()
    embs = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        for s in range(0, len(dino_stack), 64):
            bx = dino_stack[s : s + 64].to(device, non_blocking=True)
            embs.append(dino_enc(bx))
    emb_all = torch.cat(embs, dim=0).float()
    probs_dino = torch.sigmoid(dino_head(emb_all)).squeeze(1).detach().cpu().numpy()
    torch.cuda.synchronize()
    t_d_gpu = time.perf_counter() - t_d1
    dino_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    torch.cuda.reset_peak_memory_stats()

    del dino_enc, dino_head
    torch.cuda.empty_cache()

    dino_total_rate = len(dino_png_paths) / max(t_d_io + t_d_gpu, 1e-6)
    dino_full_imgs = 54706 + 26988
    dino_proj_hrs = dino_full_imgs / dino_total_rate / 3600.0
    results["dinov2_rsna"] = {
        "n_tested_images": len(dino_png_paths),
        "embed_dim": int(emb_all.shape[1]),
        "all_probs_finite": bool(np.all(np.isfinite(probs_dino))),
        "io_prep_sec": round(t_d_io, 3),
        "gpu_extract_sec": round(t_d_gpu, 3),
        "end_to_end_imgs_per_sec": round(dino_total_rate, 2),
        "gpu_only_imgs_per_sec": round(len(dino_png_paths) / max(t_d_gpu, 1e-6), 2),
        "peak_vram_gb": round(dino_vram_gb, 3),
        "full_target_images": dino_full_imgs,
        "projected_l40s_hours": round(dino_proj_hrs, 3),
        "projected_usd_excl_nvme_stage": round(dino_proj_hrs * L40S_COMBINED_HOURLY_USD + 0.50, 2),
    }

    # ------------------------------------------------------------------
    # 4. ResNet-50 AMP Training Benchmark (4 batches x 64 cached Stage-1 PNGs)
    # ------------------------------------------------------------------
    res_ckpt = models_dir / "resnet50-11ad3fa6.pth"
    res_model = build_resnet50_classifier(pretrained=False, local_weights_path=res_ckpt).to(device)
    res_opt = torch.optim.AdamW(res_model.parameters(), lr=1e-4, weight_decay=1e-4)
    res_scaler = torch.amp.GradScaler("cuda", enabled=True)
    res_crit = nn.BCEWithLogitsLoss()

    res_ids = rsna_df[rsna_df["split"] == "train"]["image_id"].astype(str).head(n_resnet_batches * resnet_batch_size).tolist()
    res_paths = [rsna_s1_dir / f"{i}.png" for i in res_ids]

    def _load_res_t(p: Path) -> torch.Tensor:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(p)
        return prepare_resnet50_tensor_from_uint16(arr)

    t_r0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as pool:
        res_tensors = list(pool.map(_load_res_t, res_paths))
    res_stack = torch.stack(res_tensors, dim=0)
    t_r_io = time.perf_counter() - t_r0

    # Warmup 1 step
    res_model.train()
    bx0 = res_stack[:resnet_batch_size].to(device)
    by0 = torch.zeros((len(bx0), 1), device=device)
    with torch.amp.autocast("cuda", enabled=True):
        l0 = res_crit(res_model(bx0), by0)
    res_scaler.scale(l0).backward()
    res_scaler.step(res_opt)
    res_scaler.update()
    res_opt.zero_grad()
    torch.cuda.synchronize()

    t_r1 = time.perf_counter()
    losses = []
    for b in range(n_resnet_batches):
        bx = res_stack[b * resnet_batch_size : (b + 1) * resnet_batch_size].to(device, non_blocking=True)
        by = torch.zeros((len(bx), 1), device=device)
        by[:16] = 1.0
        res_opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            logits = res_model(bx)
            loss = res_crit(logits, by)
        res_scaler.scale(loss).backward()
        res_scaler.step(res_opt)
        res_scaler.update()
        losses.append(float(loss.item()))
    torch.cuda.synchronize()
    t_r_train = time.perf_counter() - t_r1
    res_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)

    n_res_imgs = n_resnet_batches * resnet_batch_size
    gpu_train_rate = n_res_imgs / max(t_r_train, 1e-6)
    e2e_rate = n_res_imgs / max(t_r_io + t_r_train, 1e-6)
    # Typical full run: 10 epochs x 600 steps x 64 imgs = 384,000 train imgs + 43,443 eval/NLBS infer
    full_train_imgs = 10 * 600 * resnet_batch_size
    full_infer_imgs = 5473 + 10982 + 26988
    # Note: with local NVMe PNG staging (like Stage B/C), I/O overlaps with GPU compute
    proj_hrs_nvme = (full_train_imgs / gpu_train_rate + full_infer_imgs / (gpu_train_rate * 2.0)) / 3600.0
    results["resnet50_rsna"] = {
        "input_resolution": [854, 512],
        "batch_size": resnet_batch_size,
        "n_batches": n_resnet_batches,
        "n_tested_images": n_res_imgs,
        "all_losses_finite": bool(np.all(np.isfinite(losses))),
        "sample_losses": [round(x, 4) for x in losses],
        "io_prep_sec": round(t_r_io, 3),
        "gpu_train_sec": round(t_r_train, 3),
        "gpu_train_imgs_per_sec": round(gpu_train_rate, 2),
        "cold_io_plus_gpu_imgs_per_sec": round(e2e_rate, 2),
        "peak_vram_gb": round(res_vram_gb, 3),
        "projected_10ep_hours_nvme": round(proj_hrs_nvme, 3),
        "projected_10ep_usd_nvme": round(proj_hrs_nvme * L40S_COMBINED_HOURLY_USD + 1.00, 2),
    }

    return results


@app.local_entrypoint()
def main() -> None:
    res = run_all_benchmarks_l40s.remote()
    out_path = Path("reports/paper1/l40s_smoke_and_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
