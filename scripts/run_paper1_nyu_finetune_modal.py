"""Paper 1 Supplementary Exploratory — NYU Four-View RSNA Fine-Tuning Modal Entrypoint.

Trains `nyu_fourview_rsna_finetuned` initialized from `sample_image_model.p`
strictly on RSNA `train` 4-view exams (8,339 pts), selects best epoch on
`train_calib` (1,192 pts) AUROC, and keeps `val_calib` + NLBS strictly held out.

Requires native-resolution NYU 4-view PNGs (`2677x1942` CC, `2974x1748` MLO) at:
    /rsna_vol/cache/paper1_nyu_uint16_v1/{rsna_train,rsna_train_calib}/png_native/

Usage (DO NOT RUN FULL TRAINING WITHOUT EXPLICIT APPROVAL):
    $env:MODAL_PROFILE="dryousufmozumder"; modal run scripts/run_paper1_nyu_finetune_modal.py --stage pilot
    $env:MODAL_PROFILE="dryousufmozumder"; modal run scripts/run_paper1_nyu_finetune_modal.py --stage train --strategy head_only --epochs 8 --approve-full-train
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
from src.paper1.nyu_finetune import validate_nyu_finetune_cli_guard
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-nyu-rsna-finetune"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")
NYU_CKPT_SHA256 = "9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014"
L40S_PROD_HOURLY_USD = 2.5844  # 1x L40S ($1.95) + 8 vCPU ($0.3784) + 32 GiB RAM ($0.2560)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES, "imageio==2.37.0", "tqdm==4.67.1")
    .add_local_python_source("src")
    .add_local_dir(
        "external/breast_cancer_classifier/src",
        remote_path="/root/external/breast_cancer_classifier/src",
    )
    .add_local_file("datasets/rsna/splits_seed42.csv", remote_path="/root/datasets/rsna/splits_seed42.csv")
)


def _prepare_single_draw_exam(
    exam_rows: List[Dict[str, Any]],
    png_dir: Path,
    nyu_mods: Dict[str, Any],
    augment: bool,
    seed: int,
) -> Dict[str, Any]:
    import numpy as np
    from src.paper1.nyu_fourview_adapter import NYU_INPUT_SIZES, load_nyu_cached_view

    n_load = nyu_mods["loading"]
    rng = np.random.RandomState(seed)
    out: Dict[str, Any] = {}
    for row in exam_rows:
        vs = str(row["view_slot"])
        vc = load_nyu_cached_view(
            png_path=png_dir / f"{row['image_id']}.png",
            best_center=(int(row["best_center_y"]), int(row["best_center_x"])),
        )
        cropped, _ = n_load.augment_and_normalize_image(
            image=vc["image_f32"],
            auxiliary_image=None,
            view=vs,
            best_center=vc["best_center"],
            random_number_generator=rng,
            augmentation=augment,
            max_crop_noise=(100, 100),
            max_crop_size_noise=100,
        )
        # Ensure exact native shape [1, 1, H, W]
        h, w = NYU_INPUT_SIZES[vs]
        out[vs] = np.ascontiguousarray(cropped[:h, :w][np.newaxis, np.newaxis, ...], dtype=np.float32)
    return out


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    ephemeral_disk=524288,
    timeout=14400,
)
def run_nyu_finetune_l40s(
    stage: str = "pilot",
    strategy: str = "head_only",
    epochs: int = 8,
    batch_exams: int = 4,
    steps_per_epoch: int = 400,
    lr: float = 1e-4,
    backbone_lr_mult: float = 0.1,
    seed: int = 42,
    resume: bool = True,
    max_spend_usd: float = 12.0,
    pilot_steps: int = 8,
) -> Dict[str, Any]:
    """Run NYU Four-View L40S pilot or RSNA fine-tuning on native cached PNGs."""
    import numpy as np
    import pandas as pd
    import torch
    from src.data.rsna_dataset import BalancedBatchSampler
    from src.evaluation.stage_c_eval import compute_auroc, compute_sha256_file
    from src.paper1.nyu_finetune import (
        compute_nyu_rsna_loss,
        configure_nyu_trainable_params,
        load_nyu_finetune_checkpoint,
        save_nyu_finetune_checkpoint,
        select_rsna_4view_split_exams,
    )
    from src.paper1.nyu_fourview_adapter import (
        NYU_VIEWS,
        build_nyu_fourview_model,
        score_nyu_fourview_exam_batch,
    )

    RSNA_VOL.reload()
    t_wall0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L40S GPU required.")

    ckpt_in = RSNA_MOUNT / "models" / "paper1" / "sample_image_model.p"
    if compute_sha256_file(ckpt_in).lower() != NYU_CKPT_SHA256:
        raise RuntimeError("NYU pretrained checkpoint SHA-256 mismatch.")

    train_cache = RSNA_MOUNT / "cache" / "paper1_nyu_uint16_v1" / "rsna_train"
    tc_cache = RSNA_MOUNT / "cache" / "paper1_nyu_uint16_v1" / "rsna_train_calib"
    if not (train_cache / "manifest.csv").exists() or not (tc_cache / "manifest.csv").exists():
        raise FileNotFoundError(
            f"Missing native NYU train/train_calib cache: {train_cache} / {tc_cache}. "
            "Transfer raw RSNA train+train_calib DICOMs and run NYU Step-1 CPU caching first."
        )

    splits_df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
    _, train_exams = select_rsna_4view_split_exams(splits_df, "train")
    _, tc_exams = select_rsna_4view_split_exams(splits_df, "train_calib")
    train_s1 = pd.read_csv(train_cache / "manifest.csv", dtype=str)
    tc_s1 = pd.read_csv(tc_cache / "manifest.csv", dtype=str)

    train_by_pid = {str(k): grp.to_dict(orient="records") for k, grp in train_s1.groupby("case_id")}
    tc_by_pid = {str(k): grp.to_dict(orient="records") for k, grp in tc_s1.groupby("case_id")}

    model, nyu_mods = build_nyu_fourview_model(
        ckpt_in, device=device, nyu_repo_dir=Path("/root/external/breast_cancer_classifier")
    )
    param_groups = configure_nyu_trainable_params(
        model, strategy=strategy, base_lr=lr, backbone_lr_mult=backbone_lr_mult
    )
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    if stage == "pilot":
        sampler = BalancedBatchSampler(
            labels=train_exams["exam_cancer"].values,
            batch_size=batch_exams,
            n_pos=max(1, batch_exams // 4),
            seed=seed,
            epoch_policy="paper_independent",
            steps_per_epoch=pilot_steps,
        )
        pids = train_exams["patient_id"].astype(str).tolist()
        lm_arr = train_exams["left_malignant"].values.astype(int)
        rm_arr = train_exams["right_malignant"].values.astype(int)

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        model.train()
        losses: List[float] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for step_idx, indices in enumerate(sampler):
                ex_list = list(
                    pool.map(
                        lambda it: _prepare_single_draw_exam(
                            train_by_pid[pids[it[1]]],
                            train_cache / "png_native",
                            nyu_mods,
                            True,
                            seed + step_idx * 100 + it[0],
                        ),
                        enumerate(indices),
                    )
                )
                tensor_batch = {
                    v: torch.from_numpy(np.concatenate([ex[v] for ex in ex_list], axis=0)).to(device)
                    for v in NYU_VIEWS
                }
                lm_t = torch.tensor([lm_arr[i] for i in indices], dtype=torch.long, device=device)
                rm_t = torch.tensor([rm_arr[i] for i in indices], dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                out = model(tensor_batch)
                loss, ld = compute_nyu_rsna_loss(out, lm_t, rm_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                opt.step()
                losses.append(ld["loss_total"])
        train_sec = time.perf_counter() - t0
        n_ex = pilot_steps * batch_exams
        ex_per_sec = n_ex / max(train_sec, 1e-6)
        peak_vram_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))
        est_full_hrs = (epochs * steps_per_epoch * batch_exams) / max(ex_per_sec, 1e-6) / 3600.0
        return {
            "stage": "pilot",
            "strategy": strategy,
            "batch_exams": batch_exams,
            "pilot_steps": pilot_steps,
            "n_pilot_exams": n_ex,
            "param_counts": {"total": n_total, "trainable": n_train},
            "mean_loss": float(np.mean(losses)),
            "train_sec": round(train_sec, 3),
            "exams_per_sec": round(ex_per_sec, 3),
            "peak_vram_gb": round(peak_vram_gb, 3),
            "projected_8ep_hours": round(est_full_hrs, 3),
            "projected_8ep_usd": round(est_full_hrs * L40S_PROD_HOURLY_USD, 2),
        }

    # Full training loop (requires explicit --approve-full-train)
    out_dir = RSNA_MOUNT / "models" / "paper1" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_pt = out_dir / f"nyu_rsna_{strategy}_seed{seed}_latest.pt"
    best_pt = out_dir / f"nyu_rsna_{strategy}_seed{seed}_best_train_calib.pt"

    sampler = BalancedBatchSampler(
        labels=train_exams["exam_cancer"].values,
        batch_size=batch_exams,
        n_pos=max(1, batch_exams // 4),
        seed=seed,
        epoch_policy="paper_independent",
        steps_per_epoch=steps_per_epoch,
    )
    start_ep, best_auroc, best_epoch, history = 1, -1.0, 0, []
    if resume and latest_pt.exists():
        last_ep, best_auroc, best_epoch, history = load_nyu_finetune_checkpoint(
            latest_pt, model, opt, sched, sampler, device
        )
        start_ep = last_ep + 1

    pids = train_exams["patient_id"].astype(str).tolist()
    lm_arr = train_exams["left_malignant"].values.astype(int)
    rm_arr = train_exams["right_malignant"].values.astype(int)

    tc_pids = tc_exams["patient_id"].astype(str).tolist()
    tc_lbls = tc_exams["exam_cancer"].values.astype(int)

    for ep in range(start_ep, epochs + 1):
        elapsed_hr = (time.perf_counter() - t_wall0) / 3600.0
        if elapsed_hr * L40S_PROD_HOURLY_USD >= max_spend_usd:
            raise RuntimeError(f"COST GUARD STOP at epoch {ep}: spend >= ${max_spend_usd:.2f}")

        model.train()
        ep_losses: List[float] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for step_idx, indices in enumerate(sampler):
                ex_list = list(
                    pool.map(
                        lambda it: _prepare_single_draw_exam(
                            train_by_pid[pids[it[1]]],
                            train_cache / "png_native",
                            nyu_mods,
                            True,
                            seed + ep * 100000 + step_idx * 100 + it[0],
                        ),
                        enumerate(indices),
                    )
                )
                tensor_batch = {
                    v: torch.from_numpy(np.concatenate([ex[v] for ex in ex_list], axis=0)).to(device)
                    for v in NYU_VIEWS
                }
                lm_t = torch.tensor([lm_arr[i] for i in indices], dtype=torch.long, device=device)
                rm_t = torch.tensor([rm_arr[i] for i in indices], dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                out = model(tensor_batch)
                loss, ld = compute_nyu_rsna_loss(out, lm_t, rm_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                opt.step()
                ep_losses.append(ld["loss_total"])
        sched.step()

        model.eval()
        tc_scores: List[float] = []
        for i0 in range(0, len(tc_pids), batch_exams):
            chunk_pids = tc_pids[i0 : i0 + batch_exams]
            ex_list = [
                _prepare_single_draw_exam(tc_by_pid[p], tc_cache / "png_native", nyu_mods, False, 0)
                for p in chunk_pids
            ]
            # Remove extra batch dim [1,1,H,W] -> [1,1,H,W] works with score_nyu_fourview_exam_batch (T=1)
            res_list = score_nyu_fourview_exam_batch(model=model, tta_exam_tensors=ex_list, device=device)
            tc_scores.extend(float(r["exam_score"]) for r in res_list)

        auroc = float(compute_auroc(tc_lbls, np.asarray(tc_scores)))
        if auroc > best_auroc:
            best_auroc, best_epoch = auroc, ep
            save_nyu_finetune_checkpoint(
                best_pt, model, opt, sched,
                epoch=ep, best_auroc=best_auroc, best_epoch=best_epoch,
                history=history, sampler=sampler, aug_seed_base=seed, strategy=strategy, seed=seed,
            )
        history.append({"epoch": ep, "mean_loss": float(np.mean(ep_losses)), "train_calib_auroc": auroc})
        save_nyu_finetune_checkpoint(
            latest_pt, model, opt, sched,
            epoch=ep, best_auroc=best_auroc, best_epoch=best_epoch,
            history=history, sampler=sampler, aug_seed_base=seed, strategy=strategy, seed=seed,
        )
        RSNA_VOL.commit()

    return {
        "stage": "train",
        "strategy": strategy,
        "best_epoch": best_epoch,
        "best_train_calib_auroc": round(best_auroc, 6),
        "param_counts": {"total": n_total, "trainable": n_train},
        "history": history,
    }


@app.local_entrypoint()
def main(
    stage: str = "pilot",
    strategy: str = "head_only",
    epochs: int = 8,
    batch_exams: int = 4,
    steps_per_epoch: int = 400,
    lr: float = 1e-4,
    max_spend_usd: float = 12.0,
    approve_full_train: bool = False,
    approve_eval: bool = False,
) -> None:
    validate_nyu_finetune_cli_guard(
        stage=stage,
        approve_full_train=approve_full_train,
        approve_eval=approve_eval,
        max_spend_usd=max_spend_usd,
    )
    res = run_nyu_finetune_l40s.remote(
        stage=stage,
        strategy=strategy,
        epochs=epochs,
        batch_exams=batch_exams,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        max_spend_usd=max_spend_usd,
    )
    print(json.dumps(res, indent=2))
