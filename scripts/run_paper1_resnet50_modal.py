"""Paper 1 — Resumable ResNet-50 (IMAGENET1K_V2) RSNA Fine-Tuning & NLBS Inference Pipeline.

Reuses existing Stage-1 1520x912 uint16 PNG caches on RSNA & NLBS.
Trains strictly on RSNA `train`, selects best epoch strictly on `train_calib` AUROC,
freezes checkpoint before `val_calib` threshold selection and label-free NLBS scoring.

Usage (DO NOT RUN FULL TRAINING UNTIL APPROVED):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_resnet50_modal.py --epochs 10
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
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


def _stage_pngs_nvme(item_ids: List[str], src_dir: Path, nvme_dir: Path, max_workers: int = 16) -> float:
    """Copy Stage-1 PNGs onto local container NVMe /tmp if not already staged."""
    nvme_dir.mkdir(parents=True, exist_ok=True)
    missing = [i for i in dict.fromkeys(item_ids) if not (nvme_dir / f"{i}.png").exists()]
    t0 = time.perf_counter()
    if missing:

        def _cp(img_id: str) -> None:
            s = src_dir / f"{img_id}.png"
            d = nvme_dir / f"{img_id}.png"
            if not s.exists() or s.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty Stage-1 PNG: {s}")
            shutil.copy2(s, d)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_cp, missing))
    return time.perf_counter() - t0


def _load_batch_cpu(
    pool: ThreadPoolExecutor,
    nvme_dir: Path,
    items: List[Tuple[str, Any]],
    pin_memory: bool,
) -> Any:
    """Decode and preprocess one batch of Stage-1 PNGs on CPU threads."""
    import cv2
    import torch
    from src.paper1.resnet50_model import prepare_resnet50_tensor_from_uint16

    def _one(it: Tuple[str, Any]) -> torch.Tensor:
        img = cv2.imread(str(nvme_dir / f"{it[0]}.png"), cv2.IMREAD_UNCHANGED)
        return prepare_resnet50_tensor_from_uint16(img, augment_params=it[1])

    bx = torch.stack(list(pool.map(_one, items)), dim=0)
    return bx.pin_memory() if pin_memory else bx


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    ephemeral_disk=524288,
    timeout=14400,
)
def train_resnet50_rsna(epochs: int = 15, batch_size: int = 64, lr: float = 1e-4, seed: int = 42, resume: bool = True) -> Dict[str, Any]:
    """Resumable AMP fine-tuning of ImageNet-1K V2 ResNet-50 on RSNA train (854x512)."""
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
        amp_train_step,
        build_resnet50_classifier,
        load_resnet50_checkpoint,
        save_resnet50_checkpoint,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"

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
    vol_s1 = RSNA_MOUNT / "cache" / "stage1_rsna_uint16" / "png_1520x912"
    all_ids = pd.concat([train_df["image_id"], tc_df["image_id"]]).astype(str).tolist()
    _stage_pngs_nvme(all_ids, vol_s1, nvme_dir, max_workers=16)

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

    start_ep, best_auroc, best_epoch, history = 1, -1.0, 0, []
    if resume and latest_pt.exists():
        last_ep, best_auroc, best_epoch, history = load_resnet50_checkpoint(
            latest_pt, model, opt, sched, scaler, sampler, aug_rng, device
        )
        start_ep = last_ep + 1

    train_ids = train_df["image_id"].astype(str).tolist()
    train_lbls = train_df["cancer"].values.astype(np.float32)
    tc_ids = tc_df["image_id"].astype(str).tolist()
    tc_pids = tc_df["patient_id"].astype(str).tolist()
    tc_lbls = tc_df["cancer"].values.astype(int).tolist()

    for ep in range(start_ep, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        ep_loss = 0.0
        grad_norms = []
        # Deterministic main-thread draw schedule so prefetching never alters sampler/aug_rng state
        epoch_draws: List[Tuple[List[Tuple[str, Any]], List[int]]] = []
        for indices in sampler:
            augs = [sample_augmentation_params(aug_rng) for _ in indices]
            epoch_draws.append(([(train_ids[i], a) for i, a in zip(indices, augs)], indices))

        with ThreadPoolExecutor(max_workers=1) as prefetch_exec, ThreadPoolExecutor(max_workers=8) as img_pool:
            queue = deque()
            for k in range(min(2, len(epoch_draws))):
                items_k, idx_k = epoch_draws[k]
                queue.append((prefetch_exec.submit(_load_batch_cpu, img_pool, nvme_dir, items_k, pin_mem), idx_k))
            next_submit = min(2, len(epoch_draws))

            while queue:
                fut, indices = queue.popleft()
                if next_submit < len(epoch_draws):
                    items_n, idx_n = epoch_draws[next_submit]
                    queue.append((prefetch_exec.submit(_load_batch_cpu, img_pool, nvme_dir, items_n, pin_mem), idx_n))
                    next_submit += 1

                bx = fut.result().to(device, non_blocking=True)
                by = torch.tensor([train_lbls[i] for i in indices], dtype=torch.float32, device=device).unsqueeze(1)
                step_loss, gnorm = amp_train_step(
                    model=model,
                    bx=bx,
                    by=by,
                    criterion=crit,
                    optimizer=opt,
                    scaler=scaler,
                    max_grad_norm=5.0,
                    amp_enabled=True,
                )
                ep_loss += step_loss
                if np.isfinite(gnorm):
                    grad_norms.append(gnorm)
        sched.step()

        # Select strictly on train_calib case AUROC (never touch val_calib!)
        model.eval()
        tc_probs = []
        with torch.no_grad(), ThreadPoolExecutor(max_workers=8) as img_pool:
            for s in range(0, len(tc_ids), batch_size):
                b_ids = tc_ids[s : s + batch_size]
                bx = _load_batch_cpu(img_pool, nvme_dir, [(i, None) for i in b_ids], pin_mem).to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=True):
                    tc_probs.extend(torch.sigmoid(model(bx)).squeeze(1).float().cpu().tolist())
        cs = aggregate_case_scores(tc_probs, tc_pids)
        cl = aggregate_case_labels(tc_lbls, tc_pids)
        pids_s = sorted(cs.keys())
        tc_auroc = compute_auroc(np.array([cl[p] for p in pids_s]), np.array([cs[p] for p in pids_s]))
        if tc_auroc > best_auroc:
            best_auroc = tc_auroc
            best_epoch = ep
        history.append(
            {
                "epoch": ep,
                "loss": round(ep_loss / len(sampler), 5),
                "train_calib_auroc": round(tc_auroc, 5),
                "mean_unscaled_grad_norm": round(float(np.mean(grad_norms)), 4) if grad_norms else None,
                "finite_grad_steps": f"{len(grad_norms)}/{len(sampler)}",
                "scaler_scale": float(scaler.get_scale()),
                "sec": round(time.perf_counter() - t0, 1),
            }
        )

        save_resnet50_checkpoint(
            latest_pt, model, opt, sched, scaler, ep, best_auroc, best_epoch, history, sampler, aug_rng, seed
        )
        if best_epoch == ep:
            save_resnet50_checkpoint(
                best_pt, model, opt, sched, scaler, ep, best_auroc, best_epoch, history, sampler, aug_rng, seed
            )
        RSNA_VOL.commit()

    return {"best_epoch": best_epoch, "best_train_calib_auroc": best_auroc, "history": history, "best_ckpt": str(best_pt)}


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    ephemeral_disk=524288,
    timeout=3600,
)
def infer_resnet50_split(split_name: str, ckpt_name: str = "resnet50_rsna_seed42_best_train_calib.pt", batch_size: int = 128) -> Dict[str, Any]:
    """Run frozen ResNet-50 RSNA checkpoint inference on RSNA val_calib or label-free NLBS Stage-1 PNGs."""
    import pandas as pd
    import torch
    from src.evaluation.stage_c_eval import assert_no_forbidden_columns, compute_sha256_file, load_label_free_inference_manifest
    from src.models.trust_model import aggregate_case_scores
    from src.paper1.resnet50_model import build_resnet50_classifier

    ckpt_path = RSNA_MOUNT / "models" / "paper1" / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Frozen ResNet-50 checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"
    model = build_resnet50_classifier(pretrained=False, local_weights_path=None).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    if split_name == "rsna_val_calib":
        df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
        df = df[df["split"] == "val_calib"].sort_values(["patient_id", "image_id"]).reset_index(drop=True)
        id_col, case_col = "image_id", "patient_id"
        vol_png_dir = RSNA_MOUNT / "cache" / "stage1_rsna_uint16" / "png_1520x912"
    elif split_name == "nlbs":
        df = load_label_free_inference_manifest(Path("/root/datasets/nlbs/inference_manifest_stage_c.csv"))
        assert_no_forbidden_columns(df.columns, context="resnet50_nlbs_infer")
        id_col, case_col = "image_uid", "StudyInstanceUID"
        vol_png_dir = RSNA_MOUNT / "cache" / "stage1_nlbs_uint16" / "png_1520x912"
    else:
        raise ValueError(f"Unsupported split_name={split_name}")

    ids = df[id_col].astype(str).tolist()
    cases = df[case_col].astype(str).tolist()
    nvme_dir = Path(f"/tmp/resnet50_infer_{split_name}")
    stage_sec = _stage_pngs_nvme(ids, vol_png_dir, nvme_dir, max_workers=16)

    t0 = time.perf_counter()
    probs: List[float] = []
    batches = [[(i, None) for i in ids[s : s + batch_size]] for s in range(0, len(ids), batch_size)]
    with torch.no_grad(), ThreadPoolExecutor(max_workers=1) as prefetch_exec, ThreadPoolExecutor(max_workers=8) as img_pool:
        queue = deque()
        for k in range(min(2, len(batches))):
            queue.append(prefetch_exec.submit(_load_batch_cpu, img_pool, nvme_dir, batches[k], pin_mem))
        next_submit = min(2, len(batches))

        while queue:
            bx = queue.popleft().result().to(device, non_blocking=True)
            if next_submit < len(batches):
                queue.append(prefetch_exec.submit(_load_batch_cpu, img_pool, nvme_dir, batches[next_submit], pin_mem))
                next_submit += 1
            with torch.amp.autocast("cuda", enabled=True):
                probs.extend(torch.sigmoid(model(bx)).squeeze(1).float().cpu().tolist())

    out_dir = RSNA_MOUNT / "cache" / "paper1_resnet50_preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{split_name}_image_scores.csv"
    pd.DataFrame({id_col: ids, case_col: cases, "prob_cancer": probs}).to_csv(out_csv, index=False)

    case_scores = aggregate_case_scores(probs, cases)
    case_csv = out_dir / f"{split_name}_case_scores.csv"
    case_df = pd.DataFrame([{case_col: k, "score": float(case_scores[k])} for k in sorted(case_scores.keys())])
    if split_name == "nlbs":
        assert_no_forbidden_columns(case_df.columns, context="resnet50_nlbs_case_scores")
    case_df.to_csv(case_csv, index=False)

    RSNA_VOL.commit()
    return {
        "split": split_name,
        "n_images": len(ids),
        "n_cases": len(case_df),
        "nvme_stage_sec": round(stage_sec, 2),
        "infer_sec": round(time.perf_counter() - t0, 2),
        "ckpt_sha256": compute_sha256_file(ckpt_path),
        "out_csv": str(out_csv),
        "out_sha256": compute_sha256_file(out_csv),
        "case_csv": str(case_csv),
        "case_sha256": compute_sha256_file(case_csv),
    }


@app.local_entrypoint()
def main(
    stage: str = "train",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-4,
    seed: int = 42,
    resume: bool = True,
) -> None:
    if stage == "train":
        print(json.dumps(train_resnet50_rsna.remote(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, resume=resume), indent=2))
    elif stage == "infer":
        for sp in ("rsna_val_calib", "nlbs"):
            print(json.dumps(infer_resnet50_split.remote(sp, ckpt_name=f"resnet50_rsna_seed{seed}_best_train_calib.pt"), indent=2))
    else:
        raise ValueError(f"Unknown stage={stage}; use train or infer")
