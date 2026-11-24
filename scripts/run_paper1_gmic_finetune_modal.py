"""Paper 1 Supplementary Exploratory — GMIC RSNA Fine-Tuning Modal Entrypoint.

Trains `gmic_rsna_finetuned` initialized from `sample_model_1.p` strictly on RSNA
`train` (38,251 imgs / 8,339 pts), selects best epoch on `train_calib` (5,473 imgs /
1,192 pts) AUROC, and keeps `val_calib` + NLBS strictly held out.

Requires native-resolution `(2944, 1920)` GMIC PNGs at:
    /rsna_vol/cache/paper1_gmic_uint16_v1/{rsna_train,rsna_train_calib}/png_2944x1920/
(or raw RSNA DICOMs at /rsna_vol/rsna/raw/ for pilot benchmarking).

Usage (DO NOT RUN FULL TRAINING WITHOUT EXPLICIT APPROVAL):
    $env:MODAL_PROFILE="dryousufmozumder"; modal run scripts/run_paper1_gmic_finetune_modal.py --stage pilot
    $env:MODAL_PROFILE="dryousufmozumder"; modal run scripts/run_paper1_gmic_finetune_modal.py --stage train --strategy partial --epochs 10 --approve-full-train
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paper1.gmic_finetune import validate_gmic_finetune_cli_guard
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-gmic-rsna-finetune"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")
GMIC_CKPT_SHA256 = "3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991"
L40S_HOURLY_USD = 2.3618  # 1x L40S ($1.95) + 6 vCPU ($0.2838) + 16 GiB RAM ($0.1280)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES, "imageio==2.37.0", "tqdm==4.67.1")
    .add_local_python_source("src")
    .add_local_dir("external/GMIC/src", remote_path="/root/external/GMIC/src")
    .add_local_file("datasets/rsna/splits_seed42.csv", remote_path="/root/datasets/rsna/splits_seed42.csv")
)


def _load_gmic_png_tensor(png_path: Path, augment: bool, rng: Any) -> Any:
    import cv2
    import numpy as np
    from src.paper1.gmic_adapter import standard_normalize_gmic_uint16

    arr_u16 = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if arr_u16 is None or arr_u16.dtype != np.uint16:
        raise RuntimeError(f"Invalid GMIC uint16 PNG: {png_path}")
    img = standard_normalize_gmic_uint16(arr_u16)
    if augment and rng is not None and float(rng.random()) < 0.5:
        img = np.fliplr(img).copy()
    return img[np.newaxis, :, :]


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    ephemeral_disk=524288,
    timeout=14400,
)
def run_gmic_finetune_l40s(
    stage: str = "pilot",
    strategy: str = "partial",
    epochs: int = 10,
    batch_size: int = 8,
    steps_per_epoch: int = 600,
    lr: float = 1e-4,
    backbone_lr_mult: float = 0.1,
    saliency_l1_weight: float = 1e-5,
    seed: int = 42,
    resume: bool = True,
    max_spend_usd: float = 10.0,
    pilot_steps: int = 16,
) -> Dict[str, Any]:
    """Run GMIC L40S pilot or full RSNA fine-tuning using native (2944, 1920) uint16 PNGs."""
    import numpy as np
    import pandas as pd
    import torch
    from src.data.rsna_dataset import BalancedBatchSampler
    from src.evaluation.stage_c_eval import compute_auroc, compute_sha256_file
    from src.models.trust_model import aggregate_case_labels, aggregate_case_scores
    from src.paper1.gmic_adapter import GMIC_INPUT_SIZE, build_gmic_model
    from src.paper1.gmic_finetune import (
        configure_gmic_trainable_params,
        count_trainable_parameters,
        gmic_amp_train_step,
        load_gmic_finetune_checkpoint,
        save_gmic_finetune_checkpoint,
    )

    RSNA_VOL.reload()
    t_wall0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L40S GPU required.")

    ckpt_in = RSNA_MOUNT / "models" / "paper1" / "sample_model_1.p"
    if compute_sha256_file(ckpt_in).lower() != GMIC_CKPT_SHA256:
        raise RuntimeError("GMIC pretrained checkpoint SHA-256 mismatch.")

    df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    tc_df = df[df["split"] == "train_calib"].reset_index(drop=True)

    train_png_dir = RSNA_MOUNT / "cache" / "paper1_gmic_uint16_v1" / "rsna_train" / "png_2944x1920"
    tc_png_dir = RSNA_MOUNT / "cache" / "paper1_gmic_uint16_v1" / "rsna_train_calib" / "png_2944x1920"

    if not train_png_dir.exists() or not tc_png_dir.exists():
        raise FileNotFoundError(
            f"Missing native (2944, 1920) GMIC train/train_calib PNG cache: {train_png_dir} / {tc_png_dir}. "
            "Transfer raw RSNA train+train_calib DICOMs and build native GMIC PNG cache first."
        )

    model, _ = build_gmic_model(
        ckpt_in, model_index="1", device=device, gmic_repo_dir=Path("/root/external/GMIC")
    )
    param_groups = configure_gmic_trainable_params(
        model, strategy=strategy, base_lr=lr, backbone_lr_mult=backbone_lr_mult
    )
    param_counts = count_trainable_parameters(model)
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if stage == "pilot":
        n_pos = max(2, batch_size // 4)
        sampler = BalancedBatchSampler(
            labels=train_df["cancer"].values,
            batch_size=batch_size,
            n_pos=n_pos,
            seed=seed,
            epoch_policy="paper_independent",
            steps_per_epoch=pilot_steps,
        )
        aug_rng = np.random.default_rng(seed + 100)
        train_ids = train_df["image_id"].astype(str).tolist()
        train_lbls = train_df["cancer"].values.astype(np.float32)

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        model.train()
        losses: List[float] = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for indices in sampler:
                imgs = list(
                    pool.map(
                        lambda i: _load_gmic_png_tensor(train_png_dir / f"{train_ids[i]}.png", True, aug_rng),
                        indices,
                    )
                )
                bx = torch.from_numpy(np.stack(imgs, axis=0)).to(device, non_blocking=True)
                by = torch.tensor([train_lbls[i] for i in indices], dtype=torch.float32, device=device)
                ld, _ = gmic_amp_train_step(
                    model=model,
                    bx=bx,
                    by=by,
                    optimizer=opt,
                    scaler=scaler,
                    saliency_l1_weight=saliency_l1_weight,
                    amp_enabled=True,
                )
                losses.append(ld["loss_total"])
        train_sec = time.perf_counter() - t0
        n_imgs = pilot_steps * batch_size
        imgs_per_sec = n_imgs / max(train_sec, 1e-6)
        peak_vram_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))
        est_full_hrs = (epochs * steps_per_epoch * batch_size) / max(imgs_per_sec, 1e-6) / 3600.0
        return {
            "stage": "pilot",
            "strategy": strategy,
            "input_size": list(GMIC_INPUT_SIZE),
            "batch_size": batch_size,
            "pilot_steps": pilot_steps,
            "n_pilot_images": n_imgs,
            "param_counts": param_counts,
            "mean_loss": float(np.mean(losses)),
            "train_sec": round(train_sec, 3),
            "imgs_per_sec": round(imgs_per_sec, 2),
            "peak_vram_gb": round(peak_vram_gb, 3),
            "projected_10ep_hours": round(est_full_hrs, 3),
            "projected_10ep_usd": round(est_full_hrs * L40S_HOURLY_USD, 2),
        }

    # Full training mode
    nvme_train = Path("/tmp/gmic_train_png")
    nvme_tc = Path("/tmp/gmic_tc_png")
    nvme_train.mkdir(parents=True, exist_ok=True)
    nvme_tc.mkdir(parents=True, exist_ok=True)
    train_ids = train_df["image_id"].astype(str).tolist()
    tc_ids = tc_df["image_id"].astype(str).tolist()

    def _stage_dir(ids: List[str], src: Path, dst: Path) -> None:
        todo = [i for i in ids if not (dst / f"{i}.png").exists()]
        if todo:
            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(lambda i: shutil.copy2(src / f"{i}.png", dst / f"{i}.png"), todo))

    _stage_dir(train_ids, train_png_dir, nvme_train)
    _stage_dir(tc_ids, tc_png_dir, nvme_tc)

    out_dir = RSNA_MOUNT / "models" / "paper1" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_pt = out_dir / f"gmic_rsna_{strategy}_seed{seed}_latest.pt"
    best_pt = out_dir / f"gmic_rsna_{strategy}_seed{seed}_best_train_calib.pt"

    sampler = BalancedBatchSampler(
        labels=train_df["cancer"].values,
        batch_size=batch_size,
        n_pos=max(2, batch_size // 4),
        seed=seed,
        epoch_policy="paper_independent",
        steps_per_epoch=steps_per_epoch,
    )
    aug_rng = np.random.default_rng(seed + 100)
    start_ep, best_auroc, best_epoch, history = 1, -1.0, 0, []
    if resume and latest_pt.exists():
        last_ep, best_auroc, best_epoch, history = load_gmic_finetune_checkpoint(
            latest_pt, model, opt, sched, scaler, sampler, aug_rng, device
        )
        start_ep = last_ep + 1

    train_lbls = train_df["cancer"].values.astype(np.float32)
    tc_pids = tc_df["patient_id"].astype(str).tolist()
    tc_lbls = tc_df["cancer"].values.astype(int).tolist()

    for ep in range(start_ep, epochs + 1):
        elapsed_hr = (time.perf_counter() - t_wall0) / 3600.0
        if elapsed_hr * L40S_HOURLY_USD >= max_spend_usd:
            raise RuntimeError(f"COST GUARD STOP at epoch {ep}: spend >= ${max_spend_usd:.2f}")

        model.train()
        ep_losses: List[float] = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for indices in sampler:
                imgs = list(
                    pool.map(
                        lambda i: _load_gmic_png_tensor(nvme_train / f"{train_ids[i]}.png", True, aug_rng),
                        indices,
                    )
                )
                bx = torch.from_numpy(np.stack(imgs, axis=0)).to(device, non_blocking=True)
                by = torch.tensor([train_lbls[i] for i in indices], dtype=torch.float32, device=device)
                ld, _ = gmic_amp_train_step(
                    model=model,
                    bx=bx,
                    by=by,
                    optimizer=opt,
                    scaler=scaler,
                    saliency_l1_weight=saliency_l1_weight,
                    amp_enabled=True,
                )
                ep_losses.append(ld["loss_total"])
        sched.step()

        model.eval()
        tc_scores: List[float] = []
        with torch.no_grad():
            for i0 in range(0, len(tc_ids), batch_size):
                chunk = tc_ids[i0 : i0 + batch_size]
                imgs = [_load_gmic_png_tensor(nvme_tc / f"{cid}.png", False, None) for cid in chunk]
                bx = torch.from_numpy(np.stack(imgs, axis=0)).to(device, non_blocking=True)
                probs = model(bx)[:, 1].float().cpu().numpy().tolist()
                tc_scores.extend(float(p) for p in probs)

        _, c_scores = aggregate_case_scores(tc_pids, tc_scores)
        _, c_lbls = aggregate_case_labels(tc_pids, tc_lbls)
        auroc = float(compute_auroc(np.asarray(c_lbls), np.asarray(c_scores)))
        if auroc > best_auroc:
            best_auroc, best_epoch = auroc, ep
            save_gmic_finetune_checkpoint(
                best_pt, model, opt, sched, scaler,
                epoch=ep, best_auroc=best_auroc, best_epoch=best_epoch,
                history=history, sampler=sampler, aug_rng=aug_rng, strategy=strategy, seed=seed,
            )
        history.append({"epoch": ep, "mean_loss": float(np.mean(ep_losses)), "train_calib_auroc": auroc})
        save_gmic_finetune_checkpoint(
            latest_pt, model, opt, sched, scaler,
            epoch=ep, best_auroc=best_auroc, best_epoch=best_epoch,
            history=history, sampler=sampler, aug_rng=aug_rng, strategy=strategy, seed=seed,
        )
        RSNA_VOL.commit()

    return {
        "stage": "train",
        "strategy": strategy,
        "best_epoch": best_epoch,
        "best_train_calib_auroc": round(best_auroc, 6),
        "param_counts": param_counts,
        "history": history,
    }


@app.local_entrypoint()
def main(
    stage: str = "pilot",
    strategy: str = "partial",
    epochs: int = 10,
    batch_size: int = 8,
    steps_per_epoch: int = 600,
    lr: float = 1e-4,
    max_spend_usd: float = 10.0,
    approve_full_train: bool = False,
    approve_eval: bool = False,
) -> None:
    validate_gmic_finetune_cli_guard(
        stage=stage,
        approve_full_train=approve_full_train,
        approve_eval=approve_eval,
        max_spend_usd=max_spend_usd,
    )
    res = run_gmic_finetune_l40s.remote(
        stage=stage,
        strategy=strategy,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        max_spend_usd=max_spend_usd,
    )
    print(json.dumps(res, indent=2))
