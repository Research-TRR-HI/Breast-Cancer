"""Paper 1 — NVMe-Staged, Resumable DINOv2 ViT-S/14 Feature Extraction & Retrieval Pipeline.

Reuses existing Stage-1 1520x912 uint16 PNG caches on RSNA (54,706) and NLBS (26,988).
Stages PNGs onto ephemeral NVMe (/tmp), extracts deterministic FP32 CLS embeddings in
resumable 4,096-image chunks with integrity verification, commits atomically, and merges into:
    /rsna_vol/cache/paper1_dinov2_vits14/rsna_embeddings_fp32.pt
    /rsna_vol/cache/paper1_dinov2_vits14/nlbs_embeddings_fp32.pt

Usage (DO NOT RUN FULL EXTRACTION UNTIL APPROVED):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_dinov2_modal.py --stage benchmark
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_dinov2_modal.py --stage extract
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-dinov2-rsna"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")
CHUNK_SIZE = 4096

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES)
    .add_local_python_source("src")
    .add_local_dir("external/dinov2/dinov2", remote_path="/root/external/dinov2/dinov2")
    .add_local_file("external/dinov2/hubconf.py", remote_path="/root/external/dinov2/hubconf.py")
    .add_local_file("datasets/rsna/splits_seed42.csv", remote_path="/root/datasets/rsna/splits_seed42.csv")
    .add_local_file(
        "datasets/nlbs/inference_manifest_stage_c.csv",
        remote_path="/root/datasets/nlbs/inference_manifest_stage_c.csv",
    )
)


def _atomic_torch_save(payload: Dict[str, Any], target_path: Path) -> None:
    import torch

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".pt.tmp")
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, target_path)


def _stage_pngs_to_nvme(ids: List[str], vol_png_dir: Path, nvme_dir: Path, max_workers: int = 16) -> Dict[str, Any]:
    nvme_dir.mkdir(parents=True, exist_ok=True)
    missing = [img_id for img_id in ids if not (nvme_dir / f"{img_id}.png").exists()]
    t0 = time.perf_counter()
    if missing:

        def _copy_one(img_id: str) -> None:
            src_p = vol_png_dir / f"{img_id}.png"
            dst_p = nvme_dir / f"{img_id}.png"
            if not src_p.exists() or src_p.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty Stage-1 PNG on Volume: {src_p}")
            shutil.copy2(src_p, dst_p)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_copy_one, missing))
    elapsed = time.perf_counter() - t0
    return {"total_ids": len(ids), "newly_staged": len(missing), "stage_sec": round(elapsed, 3)}


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=8.0,
    memory=24576,
    ephemeral_disk=524288,
    timeout=7200,
)
def extract_dinov2_embeddings(
    dataset_name: str,
    limit_n: int = 0,
    batch_size: int = 128,
    chunk_size: int = CHUNK_SIZE,
    resume: bool = True,
) -> Dict[str, Any]:
    """Extract deterministic 384-d FP32 CLS embeddings via NVMe staging & resumable chunking."""
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file, load_label_free_inference_manifest
    from src.paper1.dinov2_model import (
        DINOV2_VITS14_SHA256,
        build_frozen_dinov2_vits14,
        prepare_dinov2_tensor_from_uint16,
    )

    is_benchmark = limit_n > 0
    out_dir = RSNA_MOUNT / "cache" / ("paper1_dinov2_vits14_bench" if is_benchmark else "paper1_dinov2_vits14")
    chunk_dir = out_dir / "chunks" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / f"{dataset_name}_embeddings_fp32.pt"

    ckpt_path = RSNA_MOUNT / "models" / "paper1" / "dinov2_vits14_pretrain.pth"
    if compute_sha256_file(ckpt_path).lower() != DINOV2_VITS14_SHA256:
        raise RuntimeError("DINOv2 checkpoint SHA-256 mismatch.")

    if dataset_name == "rsna":
        df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
        ids = df["image_id"].astype(str).tolist()
        vol_png_dir = RSNA_MOUNT / "cache" / "stage1_rsna_uint16" / "png_1520x912"
    elif dataset_name == "nlbs":
        df = load_label_free_inference_manifest(Path("/root/datasets/nlbs/inference_manifest_stage_c.csv"))
        ids = df["image_uid"].astype(str).tolist()
        vol_png_dir = RSNA_MOUNT / "cache" / "stage1_nlbs_uint16" / "png_1520x912"
    else:
        raise ValueError(dataset_name)

    if is_benchmark:
        ids = ids[: int(limit_n)]

    if not is_benchmark and resume and out_pt.exists():
        pack = torch.load(str(out_pt), map_location="cpu", weights_only=False)
        if (
            pack.get("backbone_sha256") == DINOV2_VITS14_SHA256
            and pack.get("image_ids") == ids
            and tuple(pack["embeddings"].shape) == (len(ids), 384)
            and bool(torch.isfinite(pack["embeddings"]).all())
        ):
            return {"dataset": dataset_name, "cached": True, "path": str(out_pt), "n_images": len(ids), "sha256": compute_sha256_file(out_pt)}

    nvme_dir = Path(f"/tmp/dinov2_{dataset_name}_png")
    stage_stats = _stage_pngs_to_nvme(ids, vol_png_dir, nvme_dir, max_workers=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = build_frozen_dinov2_vits14(
        checkpoint_path=ckpt_path, dinov2_repo_dir=Path("/root/external/dinov2"), device=device
    )

    def _load_from_nvme(img_id: str) -> torch.Tensor:
        arr = cv2.imread(str(nvme_dir / f"{img_id}.png"), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"Failed to decode staged PNG: {nvme_dir / f'{img_id}.png'}")
        return prepare_dinov2_tensor_from_uint16(arr)

    t_proc0 = time.perf_counter()
    chunk_paths: List[Path] = []
    n_chunks = (len(ids) + chunk_size - 1) // chunk_size
    gpu_time_sec = 0.0
    io_prep_sec = 0.0

    with ThreadPoolExecutor(max_workers=8) as pool:
        for c_idx in range(n_chunks):
            c_start = c_idx * chunk_size
            c_end = min(len(ids), c_start + chunk_size)
            c_ids = ids[c_start:c_end]
            c_path = chunk_dir / f"chunk_{c_idx:03d}.pt"
            chunk_paths.append(c_path)

            if not is_benchmark and resume and c_path.exists():
                ck = torch.load(str(c_path), map_location="cpu", weights_only=False)
                if (
                    ck.get("backbone_sha256") == DINOV2_VITS14_SHA256
                    and ck.get("image_ids") == c_ids
                    and tuple(ck["embeddings"].shape) == (len(c_ids), 384)
                    and bool(torch.isfinite(ck["embeddings"]).all())
                ):
                    continue

            c_embs = []
            for s in range(0, len(c_ids), batch_size):
                b_ids = c_ids[s : s + batch_size]
                t_io0 = time.perf_counter()
                bx = torch.stack(list(pool.map(_load_from_nvme, b_ids)), dim=0).to(device, non_blocking=True)
                io_prep_sec += time.perf_counter() - t_io0

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_g0 = time.perf_counter()
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                    emb = enc(bx).float().cpu()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                gpu_time_sec += time.perf_counter() - t_g0

                if not bool(torch.isfinite(emb).all()):
                    raise RuntimeError(f"Non-finite DINOv2 embedding detected in chunk {c_idx}!")
                c_embs.append(emb)

            c_tensor = torch.cat(c_embs, dim=0)
            _atomic_torch_save({"image_ids": c_ids, "embeddings": c_tensor, "backbone_sha256": DINOV2_VITS14_SHA256}, c_path)
            RSNA_VOL.commit()
            print(f"[DINOv2 {dataset_name}] chunk {c_idx + 1}/{n_chunks} ({c_end}/{len(ids)})", flush=True)

    embs_all = []
    ids_all = []
    for c_path in chunk_paths:
        ck = torch.load(str(c_path), map_location="cpu", weights_only=False)
        ids_all.extend(ck["image_ids"])
        embs_all.append(ck["embeddings"].float())
    embs = torch.cat(embs_all, dim=0)
    if ids_all != ids or tuple(embs.shape) != (len(ids), 384) or not bool(torch.isfinite(embs).all()):
        raise RuntimeError(f"Integrity verification failed after merging {dataset_name} DINOv2 embeddings.")

    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
    total_extract_sec = time.perf_counter() - t_proc0
    total_end_to_end_sec = float(stage_stats["stage_sec"]) + total_extract_sec

    _atomic_torch_save({"image_ids": ids, "embeddings": embs, "backbone_sha256": DINOV2_VITS14_SHA256}, out_pt)
    RSNA_VOL.commit()

    return {
        "dataset": dataset_name,
        "is_benchmark": is_benchmark,
        "n_images": len(ids),
        "shape": list(embs.shape),
        "nvme_stage": stage_stats,
        "io_prep_sec": round(io_prep_sec, 3),
        "gpu_infer_sec": round(gpu_time_sec, 3),
        "extract_after_stage_sec": round(total_extract_sec, 3),
        "end_to_end_sec": round(total_end_to_end_sec, 3),
        "end_to_end_imgs_per_sec": round(len(ids) / max(total_end_to_end_sec, 1e-6), 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "out_pt": str(out_pt),
        "sha256": compute_sha256_file(out_pt),
        "download_cmd": f"modal volume get --profile tigerwood693 rsna-mammography-vol {str(out_pt).replace('/rsna_vol', '')} reports/paper1/dinov2_cache/",
    }


@app.local_entrypoint()
def main(stage: str = "extract", benchmark_n: int = 512) -> None:
    profile = os.environ.get("MODAL_PROFILE", "tigerwood693")
    if stage == "benchmark":
        res = extract_dinov2_embeddings.remote("rsna", limit_n=benchmark_n, resume=False)
        res["download_cmd"] = f"modal volume get --profile {profile} rsna-mammography-vol {str(res['out_pt']).replace('/rsna_vol', '')} reports/paper1/dinov2_cache/"
        out_p = Path("reports/paper1/dinov2_nvme_benchmark.json")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
    elif stage == "extract":
        for ds in ("rsna", "nlbs"):
            res = extract_dinov2_embeddings.remote(ds, limit_n=0, resume=True)
            res["download_cmd"] = f"modal volume get --profile {profile} rsna-mammography-vol {str(res['out_pt']).replace('/rsna_vol', '')} reports/paper1/dinov2_cache/"
            print(json.dumps(res, indent=2))
    else:
        raise ValueError("Use --stage benchmark or --stage extract (head training runs locally via scripts/train_paper1_dinov2_head.py).")
