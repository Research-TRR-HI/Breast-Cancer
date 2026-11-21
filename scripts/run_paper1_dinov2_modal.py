"""Paper 1 — Resumable DINOv2 ViT-S/14 Feature Extraction & RSNA Head Training Pipeline.

Reuses existing Stage-1 1520x912 uint16 PNG caches on RSNA (54,706) and NLBS (26,988).
Trains classification head strictly on RSNA `train` (8,339 pts), selects best epoch/head
strictly on `train_calib` (1,192 pts), and freezes before `val_calib` & NLBS evaluation.

Usage (DO NOT RUN FULL PIPELINE UNTIL APPROVED):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_dinov2_modal.py --stage extract
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_dinov2_modal.py --stage train_and_eval
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

APP_NAME = "paper1-dinov2-rsna"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")

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


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=3600,
)
def extract_dinov2_embeddings(dataset_name: str) -> Dict[str, Any]:
    """Extract and save deterministic 384-d FP32 CLS embeddings once per unique image."""
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

    out_dir = RSNA_MOUNT / "cache" / "paper1_dinov2_vits14"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / f"{dataset_name}_embeddings_fp32.pt"
    if out_pt.exists():
        return {"dataset": dataset_name, "cached": True, "path": str(out_pt), "sha256": compute_sha256_file(out_pt)}

    ckpt_path = RSNA_MOUNT / "models" / "paper1" / "dinov2_vits14_pretrain.pth"
    if compute_sha256_file(ckpt_path).lower() != DINOV2_VITS14_SHA256:
        raise RuntimeError("DINOv2 checkpoint SHA-256 mismatch.")

    if dataset_name == "rsna":
        df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
        ids = df["image_id"].astype(str).tolist()
        png_dir = RSNA_MOUNT / "cache" / "stage1_rsna_uint16" / "png_1520x912"
    elif dataset_name == "nlbs":
        df = load_label_free_inference_manifest(Path("/root/datasets/nlbs/inference_manifest_stage_c.csv"))
        ids = df["image_uid"].astype(str).tolist()
        png_dir = RSNA_MOUNT / "cache" / "stage1_nlbs_uint16" / "png_1520x912"
    else:
        raise ValueError(dataset_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = build_frozen_dinov2_vits14(
        checkpoint_path=ckpt_path, dinov2_repo_dir=Path("/root/external/dinov2"), device=device
    )

    def _load(img_id: str) -> torch.Tensor:
        arr = cv2.imread(str(png_dir / f"{img_id}.png"), cv2.IMREAD_UNCHANGED)
        return prepare_dinov2_tensor_from_uint16(arr)

    t0 = time.perf_counter()
    embs_list = []
    batch_sz = 128
    with ThreadPoolExecutor(max_workers=8) as pool:
        for s in range(0, len(ids), batch_sz):
            b_ids = ids[s : s + batch_sz]
            bx = torch.stack(list(pool.map(_load, b_ids)), dim=0).to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                e = enc(bx).float().cpu()
            embs_list.append(e)
            if (s // batch_sz + 1) % 50 == 0:
                print(f"[DINOv2 {dataset_name}] {min(s + batch_sz, len(ids))}/{len(ids)}", flush=True)

    embs = torch.cat(embs_list, dim=0)
    torch.save({"image_ids": ids, "embeddings": embs, "backbone_sha256": DINOV2_VITS14_SHA256}, out_pt)
    RSNA_VOL.commit()
    return {
        "dataset": dataset_name,
        "n_images": len(ids),
        "shape": list(embs.shape),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "sha256": compute_sha256_file(out_pt),
    }


@app.local_entrypoint()
def main(stage: str = "extract") -> None:
    if stage == "extract":
        for ds in ("rsna", "nlbs"):
            print(json.dumps(extract_dinov2_embeddings.remote(ds), indent=2))
    else:
        raise ValueError("Train/eval runs cleanly on CPU once embeddings are extracted locally.")
