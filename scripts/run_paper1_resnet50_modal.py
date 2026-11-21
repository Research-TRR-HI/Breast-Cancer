"""Paper 1 — Resumable ResNet-50 (IMAGENET1K_V2) RSNA Fine-Tuning & NLBS Inference Pipeline.

Reuses existing Stage-1 1520x912 uint16 PNG caches on RSNA & NLBS.
Trains strictly on RSNA `train`, selects best epoch strictly on `train_calib` AUROC,
freezes checkpoint before `val_calib` threshold selection and label-free NLBS scoring.

Usage (DO NOT RUN FULL TRAINING UNTIL APPROVED):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_resnet50_modal.py --epochs 10
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-resnet50-rsna"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES)
    .add_local_python_source("src")
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
    ephemeral_disk=524288,
    timeout=14400,
)
def train_resnet50_rsna(epochs: int = 10, batch_size: int = 64, lr: float = 1e-4, seed: int = 42, resume: bool = True) -> Dict[str, Any]:
    """Resumable AMP fine-tuning of ImageNet-1K V2 ResNet-50 on RSNA train (854x512)."""
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from src.data.preprocessing import sample_augmentation_params
    from src.data.rsna_dataset import BalancedBatchSampler
    from src.evaluation.stage_c_eval import compute_auroc, compute_sha256_file
    from src.models.trust_model import aggregate_case_labels, aggregate_case_scores
    from src.paper1.resnet50_model import (
        RESNET50_IMAGENET1K_V2_SHA256,
        build_resnet50_classifier,
        prepare_resnet50_tensor_from_uint16,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_in = RSNA_MOUNT / "models" / "paper1" / "resnet50-11ad3fa6.pth"
    if compute_sha256_file(ckpt_in).lower() != RESNET50_IMAGENET1K_V2_SHA256:
        raise RuntimeError("ResNet-50 checkpoint SHA-256 mismatch.")

    out_dir = RSNA_MOUNT / "models" / "paper1" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_pt = out_dir / f"resnet50_rsna_seed{seed}_latest.pt"
    best_pt = out_dir / f"resnet50_rsna_seed{seed}_best_train_calib.pt"

    df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    tc_df = df[df["split"] == "train_calib"].reset_index(drop=True)

    # Stage RSNA Stage-1 PNGs onto local NVMe for fast multi-epoch loading
    nvme_dir = Path("/tmp/rsna_s1_png")
    nvme_dir.mkdir(parents=True, exist_ok=True)
    vol_s1 = RSNA_MOUNT / "cache" / "stage1_rsna_uint16" / "png_1520x912"
    all_ids = pd.concat([train_df["image_id"], tc_df["image_id"]]).astype(str).tolist()
    to_copy = [i for i in all_ids if not (nvme_dir / f"{i}.png").exists()]
    if to_copy:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda i: shutil.copy2(vol_s1 / f"{i}.png", nvme_dir / f"{i}.png"), to_copy))

    model = build_resnet50_classifier(pretrained=False, local_weights_path=ckpt_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    crit = nn.BCEWithLogitsLoss()

    sampler = BalancedBatchSampler(
        labels=train_df["cancer"].values,
        batch_size=batch_size,
        n_pos=max(16, batch_size // 4),
        seed=seed,
        epoch_policy="paper_independent",
        steps_per_epoch=600,
    )
    aug_rng = np.random.default_rng(seed + 100)

    start_ep, best_auroc, history = 1, -1.0, []
    if resume and latest_pt.exists():
        ck = torch.load(latest_pt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_ep = int(ck["epoch"]) + 1
        best_auroc = float(ck["best_auroc"])
        history = ck["history"]

    train_ids = train_df["image_id"].astype(str).tolist()
    train_lbls = train_df["cancer"].values.astype(np.float32)
    tc_ids = tc_df["image_id"].astype(str).tolist()
    tc_pids = tc_df["patient_id"].astype(str).tolist()
    tc_lbls = tc_df["cancer"].values.astype(int).tolist()

    for ep in range(start_ep, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        ep_loss = 0.0
        with ThreadPoolExecutor(max_workers=8) as pool:
            for b_idx, indices in enumerate(sampler):
                augs = [sample_augmentation_params(aug_rng) for _ in indices]
                items = [(train_ids[i], a) for i, a in zip(indices, augs)]
                bx = torch.stack(
                    list(pool.map(lambda it: prepare_resnet50_tensor_from_uint16(cv2.imread(str(nvme_dir / f"{it[0]}.png"), cv2.IMREAD_UNCHANGED), augment_params=it[1]), items)),
                    dim=0,
                ).to(device, non_blocking=True)
                by = torch.tensor([train_lbls[i] for i in indices], dtype=torch.float32, device=device).unsqueeze(1)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=True):
                    loss = crit(model(bx), by)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                ep_loss += float(loss.item())
        sched.step()

        # Select strictly on train_calib case AUROC (never touch val_calib!)
        model.eval()
        tc_probs = []
        with torch.no_grad(), ThreadPoolExecutor(max_workers=8) as pool:
            for s in range(0, len(tc_ids), batch_size):
                b_ids = tc_ids[s : s + batch_size]
                bx = torch.stack(
                    list(pool.map(lambda i: prepare_resnet50_tensor_from_uint16(cv2.imread(str(nvme_dir / f"{i}.png"), cv2.IMREAD_UNCHANGED)), b_ids)),
                    dim=0,
                ).to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=True):
                    tc_probs.extend(torch.sigmoid(model(bx)).squeeze(1).float().cpu().tolist())
        cs = aggregate_case_scores(tc_probs, tc_pids)
        cl = aggregate_case_labels(tc_lbls, tc_pids)
        pids_s = sorted(cs.keys())
        tc_auroc = compute_auroc(np.array([cl[p] for p in pids_s]), np.array([cs[p] for p in pids_s]))
        history.append({"epoch": ep, "loss": round(ep_loss / len(sampler), 5), "train_calib_auroc": round(tc_auroc, 5), "sec": round(time.perf_counter() - t0, 1)})

        payload = {"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "best_auroc": max(best_auroc, tc_auroc), "history": history}
        torch.save(payload, latest_pt)
        if tc_auroc > best_auroc:
            best_auroc = tc_auroc
            torch.save(payload, best_pt)
        RSNA_VOL.commit()

    return {"best_train_calib_auroc": best_auroc, "history": history, "best_ckpt": str(best_pt)}


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=3600,
)
def infer_resnet50_split(split_name: str, ckpt_name: str = "resnet50_rsna_seed42_best_train_calib.pt", batch_size: int = 128) -> Dict[str, Any]:
    """Run frozen ResNet-50 RSNA checkpoint inference on RSNA val_calib or label-free NLBS Stage-1 PNGs."""
    import cv2
    import pandas as pd
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file, load_label_free_inference_manifest
    from src.paper1.resnet50_model import build_resnet50_classifier, prepare_resnet50_tensor_from_uint16

    ckpt_path = RSNA_MOUNT / "models" / "paper1" / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Frozen ResNet-50 checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet50_classifier(pretrained=False, local_weights_path=None).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    if split_name == "rsna_val_calib":
        df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
        df = df[df["split"] == "val_calib"].sort_values(["patient_id", "image_id"]).reset_index(drop=True)
        id_col, case_col = "image_id", "patient_id"
        png_dir = RSNA_MOUNT / "cache" / "stage1_rsna_uint16" / "png_1520x912"
    elif split_name == "nlbs":
        df = load_label_free_inference_manifest(Path("/root/datasets/nlbs/inference_manifest_stage_c.csv"))
        id_col, case_col = "image_uid", "StudyInstanceUID"
        png_dir = RSNA_MOUNT / "cache" / "stage1_nlbs_uint16" / "png_1520x912"
    else:
        raise ValueError(f"Unsupported split_name={split_name}")

    ids = df[id_col].astype(str).tolist()
    cases = df[case_col].astype(str).tolist()
    t0 = time.perf_counter()
    probs = []
    with torch.no_grad(), ThreadPoolExecutor(max_workers=8) as pool:
        for s in range(0, len(ids), batch_size):
            b_ids = ids[s : s + batch_size]
            bx = torch.stack(
                list(pool.map(lambda i: prepare_resnet50_tensor_from_uint16(cv2.imread(str(png_dir / f"{i}.png"), cv2.IMREAD_UNCHANGED)), b_ids)),
                dim=0,
            ).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                probs.extend(torch.sigmoid(model(bx)).squeeze(1).float().cpu().tolist())

    out_dir = RSNA_MOUNT / "cache" / "paper1_resnet50_preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{split_name}_image_scores.csv"
    pd.DataFrame({id_col: ids, case_col: cases, "prob_cancer": probs}).to_csv(out_csv, index=False)
    RSNA_VOL.commit()
    return {
        "split": split_name,
        "n_images": len(ids),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "ckpt_sha256": compute_sha256_file(ckpt_path),
        "out_csv": str(out_csv),
        "out_sha256": compute_sha256_file(out_csv),
    }


@app.local_entrypoint()
def main(stage: str = "train", epochs: int = 10, seed: int = 42) -> None:
    if stage == "train":
        print(json.dumps(train_resnet50_rsna.remote(epochs=epochs, seed=seed), indent=2))
    elif stage == "infer":
        for sp in ("rsna_val_calib", "nlbs"):
            print(json.dumps(infer_resnet50_split.remote(sp, ckpt_name=f"resnet50_rsna_seed{seed}_best_train_calib.pt"), indent=2))
    else:
        raise ValueError(f"Unknown stage={stage}; use train or infer")
