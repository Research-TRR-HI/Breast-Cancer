"""Paper 1 Supplementary Exploratory — GMIC RSNA Fine-Tuning & Evaluation Modal Entrypoint.

Trains `gmic_rsna_finetuned` initialized from `sample_model_1.p` strictly on RSNA
`train` (38,251 imgs / 8,339 pts), selects best epoch on `train_calib` (5,473 imgs /
1,192 pts) AUROC, and keeps `val_calib` + NLBS strictly held out.
Also provides a strict zero-optimizer evaluation-only path (`--stage eval`) with
independent checkpoint SHA-256 verification, resumable shards, RSNA-first calibration
gating, and SHA-256 locked predictions.

Requires native-resolution `(2944, 1920)` GMIC PNGs at:
    /rsna_vol/cache/paper1_gmic_uint16_v1/{split_name}/png_2944x1920/
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paper1.gmic_finetune import (
    GMIC_FINETUNED_MODEL_ID,
    compute_gmic_schedule_stats,
    validate_gmic_eval_cli_args,
    validate_gmic_finetune_cli_guard,
)
from src.paper1.specialist_eval import verify_rsna_calibration_lock
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-gmic-rsna-finetune"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")
GMIC_CKPT_SHA256 = "3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991"
PREPROCESS_VERSION = "paper1_gmic_uint16_v1"
L40S_HOURLY_USD = 2.3618  # 1x L40S ($1.95) + 6 vCPU ($0.2838) + 16 GiB RAM ($0.1280)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES, "imageio==2.37.0", "tqdm==4.67.1")
    .add_local_python_source("src")
    .add_local_dir("external/GMIC/src", remote_path="/root/external/GMIC/src")
    .add_local_file("datasets/rsna/splits_seed42.csv", remote_path="/root/datasets/rsna/splits_seed42.csv")
    .add_local_file(
        "datasets/nlbs/inference_manifest_stage_c.csv",
        remote_path="/root/datasets/nlbs/inference_manifest_stage_c.csv",
    )
)


def _load_gmic_png_tensor(png_path: Path, augment: bool, rng: Any) -> Any:
    import cv2
    import numpy as np
    from src.paper1.gmic_adapter import GMIC_INPUT_SIZE, standard_normalize_gmic_uint16

    arr_u16 = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if arr_u16 is None or arr_u16.dtype != np.uint16 or arr_u16.shape != GMIC_INPUT_SIZE:
        raise RuntimeError(f"Invalid GMIC uint16 PNG: {png_path}")
    img = standard_normalize_gmic_uint16(arr_u16)
    if augment and rng is not None and float(rng.random()) < 0.5:
        img = np.fliplr(img).copy()
    return img[np.newaxis, :, :]


def _stage_pngs_nvme(ids: List[str], src: Path, dst: Path, max_workers: int = 16) -> float:
    dst.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (dst / f"{i}.png").exists()]
    t0 = time.perf_counter()
    if todo:

        def _cp(img_id: str) -> None:
            s = src / f"{img_id}.png"
            d = dst / f"{img_id}.png"
            if not s.exists() or s.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty GMIC PNG: {s}")
            shutil.copy2(s, d)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_cp, todo))
    return time.perf_counter() - t0


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
    epochs: int = 8,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    steps_per_epoch: int = 0,
    lr: float = 1e-4,
    backbone_lr_mult: float = 0.1,
    saliency_l1_weight: float = 1e-5,
    early_stopping_patience: int = 3,
    min_epochs: int = 4,
    seed: int = 42,
    resume: bool = True,
    max_spend_usd: float = 10.0,
    pilot_steps: int = 6,
    precision: str = "bf16",
    bench_batch_sizes: str = "8,16,32,64,128,256",
    bench_precisions: str = "bf16,fp16,tf32",
    lr_scaling: str = "sqrt",
    warmup_steps: int = 100,
) -> Dict[str, Any]:
    """Run GMIC L40S progressive pilot or RSNA fine-tuning using native (2944, 1920) uint16 PNGs."""
    import numpy as np
    import pandas as pd
    import torch
    from src.data.rsna_dataset import BalancedBatchSampler
    from src.evaluation.stage_c_eval import compute_auroc, compute_sha256_file
    from src.models.trust_model import aggregate_case_labels, aggregate_case_scores
    from src.paper1.gmic_adapter import GMIC_INPUT_SIZE, build_gmic_model
    from src.paper1.gmic_finetune import (
        apply_linear_warmup,
        compute_gmic_schedule_stats,
        configure_gmic_precision_and_scaler,
        configure_gmic_trainable_params,
        count_trainable_parameters,
        create_fused_adamw,
        gmic_amp_train_step,
        load_gmic_finetune_checkpoint,
        parse_int_csv,
        parse_str_csv,
        run_progressive_gmic_benchmark_loop,
        save_gmic_finetune_checkpoint,
        scale_steps_and_lr_for_batch_size,
        set_gmic_backbone_eval_mode,
    )

    if stage not in ("pilot", "train"):
        raise ValueError(f"run_gmic_finetune_l40s supports stage='pilot' or 'train' only, got {stage!r}")

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

    train_png_dir = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / "rsna_train" / "png_2944x1920"
    tc_png_dir = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / "rsna_train_calib" / "png_2944x1920"
    if not train_png_dir.exists() or not tc_png_dir.exists():
        raise FileNotFoundError(
            f"Missing native (2944, 1920) GMIC train/train_calib PNG cache: {train_png_dir} / {tc_png_dir}."
        )

    model, _ = build_gmic_model(
        ckpt_in, model_index="1", device=device, gmic_repo_dir=Path("/root/external/GMIC")
    )
    param_counts = count_trainable_parameters(model)

    fair_plan = scale_steps_and_lr_for_batch_size(
        batch_size,
        base_batch_size=8,
        base_steps_per_epoch=600,
        base_lr=lr,
        lr_scaling=lr_scaling,
        base_warmup_steps=warmup_steps,
    )
    eff_steps_per_epoch = int(steps_per_epoch) if int(steps_per_epoch) > 0 else fair_plan["steps_per_epoch"]
    eff_lr = fair_plan["scaled_lr"]
    eff_warmup_steps = fair_plan["warmup_steps"]

    if stage == "pilot":
        bench_bs_list = parse_int_csv(bench_batch_sizes)
        bench_prec_list = parse_str_csv(bench_precisions)
        train_ids = train_df["image_id"].astype(str).tolist()
        train_lbls = train_df["cancer"].values.astype(np.float32)
        tc_ids = tc_df["image_id"].astype(str).tolist()

        pool_tr_n = min(len(train_ids), max(512, max(bench_bs_list) * 2))
        pool_tc_n = min(len(tc_ids), max(128, max(bench_bs_list)))
        pilot_train_ids = train_ids[:pool_tr_n]
        pilot_tc_ids = tc_ids[:pool_tc_n]
        nvme_pilot_tr = Path("/tmp/gmic_pilot_tr")
        nvme_pilot_tc = Path("/tmp/gmic_pilot_tc")
        shutil.rmtree(nvme_pilot_tr, ignore_errors=True)
        shutil.rmtree(nvme_pilot_tc, ignore_errors=True)
        stage_tr_sec = _stage_pngs_nvme(pilot_train_ids, train_png_dir, nvme_pilot_tr)
        stage_tc_sec = _stage_pngs_nvme(pilot_tc_ids, tc_png_dir, nvme_pilot_tc)
        staged_bytes = sum((nvme_pilot_tr / f"{i}.png").stat().st_size for i in pilot_train_ids) + sum(
            (nvme_pilot_tc / f"{i}.png").stat().st_size for i in pilot_tc_ids
        )
        stage_mb_s = (staged_bytes / (1024.0**2)) / max(stage_tr_sec + stage_tc_sec, 1e-6)
        aug_rng = np.random.default_rng(seed + 100)

        with ThreadPoolExecutor(max_workers=8) as pool:

            def _bench_train_loader(bs: int, step_idx: int) -> Tuple[np.ndarray, np.ndarray]:
                start = (step_idx * bs) % pool_tr_n
                idxs = [(start + k) % pool_tr_n for k in range(bs)]
                flips = [bool(aug_rng.random() < 0.5) for _ in idxs]
                imgs = list(
                    pool.map(
                        lambda it: _load_gmic_png_tensor(nvme_pilot_tr / f"{pilot_train_ids[it[0]]}.png", it[1], None),
                        zip(idxs, flips),
                    )
                )
                return np.stack(imgs, axis=0), np.asarray([train_lbls[i] for i in idxs], dtype=np.float32)

            def _bench_val_loader(bs: int, step_idx: int) -> np.ndarray:
                start = (step_idx * bs) % pool_tc_n
                idxs = [(start + k) % pool_tc_n for k in range(bs)]
                imgs = list(pool.map(lambda idx: _load_gmic_png_tensor(nvme_pilot_tc / f"{pilot_tc_ids[idx]}.png", False, None), idxs))
                return np.stack(imgs, axis=0)

            bench = run_progressive_gmic_benchmark_loop(
                model=model,
                batch_loader=_bench_train_loader,
                device=device,
                batch_sizes=bench_bs_list,
                precisions=bench_prec_list,
                strategy=strategy,
                base_lr=lr,
                backbone_lr_mult=backbone_lr_mult,
                saliency_l1_weight=saliency_l1_weight,
                warmup_steps=2,
                bench_steps=max(2, min(int(pilot_steps), 4)),
                val_loader=_bench_val_loader,
                val_steps=2,
            )

        rec = bench.get("recommended")
        rec_proj: Dict[str, Any] = {}
        if rec is not None:
            rec_plan = scale_steps_and_lr_for_batch_size(
                rec["batch_size"],
                base_batch_size=8,
                base_steps_per_epoch=600,
                base_lr=lr,
                lr_scaling=lr_scaling,
                base_warmup_steps=warmup_steps,
            )
            proj_train_hrs = (epochs * rec_plan["steps_per_epoch"] * rec["batch_size"]) / max(rec["imgs_per_sec"], 1e-6) / 3600.0
            val_tps = rec.get("val_imgs_per_sec") or rec["imgs_per_sec"]
            proj_val_hrs = (epochs * len(tc_ids)) / max(val_tps, 1e-6) / 3600.0
            proj_stage_hrs = (245.222 * 1024.0 / max(stage_mb_s, 50.0)) / 3600.0
            proj_total_hrs = proj_train_hrs + proj_val_hrs + proj_stage_hrs + (60.0 / 3600.0)
            rec_proj = {
                "recommended_fair_schedule": rec_plan,
                "projected_train_only_hours": round(proj_train_hrs, 3),
                "projected_val_hours": round(proj_val_hrs, 3),
                "projected_nvme_stage_hours": round(proj_stage_hrs, 3),
                "projected_all_inclusive_hours": round(proj_total_hrs, 3),
                "projected_all_inclusive_usd": round(proj_total_hrs * L40S_HOURLY_USD, 2),
                "schedule_coverage_stats": compute_gmic_schedule_stats(
                    n_pos_imgs=int((train_df["cancer"] == 1).sum()),
                    n_neg_imgs=int((train_df["cancer"] == 0).sum()),
                    batch_size=rec["batch_size"],
                    steps_per_epoch=rec_plan["steps_per_epoch"],
                    epochs=epochs,
                ),
            }

        pilot_report = {
            "stage": "pilot",
            "strategy": strategy,
            "input_size": list(GMIC_INPUT_SIZE),
            "epochs": epochs,
            "nvme_stage_mb_per_sec": round(stage_mb_s, 2),
            "param_counts": param_counts,
            "sweep_results": bench["sweep_results"],
            "recommended": rec,
            **rec_proj,
        }
        ckpt_dir = RSNA_MOUNT / "models" / "paper1" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "gmic_pilot_benchmark_report.json").write_text(json.dumps(pilot_report, indent=2), encoding="utf-8")
        RSNA_VOL.commit()
        return pilot_report

    # Full training mode
    prec_cfg = configure_gmic_precision_and_scaler(precision, device)
    param_groups = configure_gmic_trainable_params(
        model, strategy=strategy, base_lr=eff_lr, backbone_lr_mult=backbone_lr_mult
    )
    opt, used_fused_adamw = create_fused_adamw(param_groups, weight_decay=1e-4, device=device)
    target_lrs = [float(pg["lr"]) for pg in opt.param_groups]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = prec_cfg["scaler"]
    steps_per_epoch = eff_steps_per_epoch

    nvme_train = Path("/tmp/gmic_train_png")
    nvme_tc = Path("/tmp/gmic_tc_png")
    train_ids = train_df["image_id"].astype(str).tolist()
    tc_ids = tc_df["image_id"].astype(str).tolist()
    _stage_pngs_nvme(train_ids, train_png_dir, nvme_train)
    _stage_pngs_nvme(tc_ids, tc_png_dir, nvme_tc)

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
    no_improve_epochs = 0
    global_step = (start_ep - 1) * steps_per_epoch

    for ep in range(start_ep, epochs + 1):
        elapsed_hr = (time.perf_counter() - t_wall0) / 3600.0
        if elapsed_hr * L40S_HOURLY_USD >= max_spend_usd:
            raise RuntimeError(f"COST GUARD STOP at epoch {ep}: spend >= ${max_spend_usd:.2f}")

        set_gmic_backbone_eval_mode(model, strategy)
        ep_losses: List[float] = []
        batch_list = list(sampler)
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending = deque()
            next_b = 0
            while next_b < len(batch_list) and len(pending) < 3:
                idxs = batch_list[next_b]
                flips = [bool(aug_rng.random() < 0.5) for _ in idxs]
                futs = [pool.submit(_load_gmic_png_tensor, nvme_train / f"{train_ids[i]}.png", fl, None) for i, fl in zip(idxs, flips)]
                pending.append((idxs, futs))
                next_b += 1

            while pending:
                idxs, futs = pending.popleft()
                imgs = [f.result() for f in futs]
                while next_b < len(batch_list) and len(pending) < 3:
                    n_idxs = batch_list[next_b]
                    n_flips = [bool(aug_rng.random() < 0.5) for _ in n_idxs]
                    n_futs = [pool.submit(_load_gmic_png_tensor, nvme_train / f"{train_ids[i]}.png", fl, None) for i, fl in zip(n_idxs, n_flips)]
                    pending.append((n_idxs, n_futs))
                    next_b += 1

                if global_step < eff_warmup_steps:
                    apply_linear_warmup(opt, global_step, eff_warmup_steps, target_lrs)

                bx = torch.from_numpy(np.stack(imgs, axis=0)).pin_memory().to(device, non_blocking=True)
                by = torch.tensor([train_lbls[i] for i in idxs], dtype=torch.float32).pin_memory().to(device, non_blocking=True)
                ld, _ = gmic_amp_train_step(
                    model=model,
                    bx=bx,
                    by=by,
                    optimizer=opt,
                    scaler=scaler,
                    saliency_l1_weight=saliency_l1_weight,
                    amp_enabled=prec_cfg["amp_enabled"],
                    amp_dtype=prec_cfg["amp_dtype"],
                )
                global_step += 1
                ep_losses.append(ld["loss_total"])
        if global_step >= eff_warmup_steps:
            sched.step()

        model.eval()
        tc_scores: List[float] = []
        with torch.no_grad(), ThreadPoolExecutor(max_workers=8) as pool:
            tc_chunks = [tc_ids[i0 : i0 + eval_batch_size] for i0 in range(0, len(tc_ids), eval_batch_size)]
            val_pending = deque()
            next_vc = 0
            while next_vc < len(tc_chunks) and len(val_pending) < 3:
                val_pending.append([pool.submit(_load_gmic_png_tensor, nvme_tc / f"{cid}.png", False, None) for cid in tc_chunks[next_vc]])
                next_vc += 1
            while val_pending:
                vfuts = val_pending.popleft()
                imgs = [f.result() for f in vfuts]
                if next_vc < len(tc_chunks):
                    val_pending.append([pool.submit(_load_gmic_png_tensor, nvme_tc / f"{cid}.png", False, None) for cid in tc_chunks[next_vc]])
                    next_vc += 1
                bx = torch.from_numpy(np.stack(imgs, axis=0)).pin_memory().to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=prec_cfg["amp_dtype"], enabled=prec_cfg["amp_enabled"]):
                    probs = model(bx)[:, 1].float().cpu().numpy().tolist()
                tc_scores.extend(float(p) for p in probs)

        _, c_scores = aggregate_case_scores(tc_pids, tc_scores)
        _, c_lbls = aggregate_case_labels(tc_pids, tc_lbls)
        auroc = float(compute_auroc(np.asarray(c_lbls), np.asarray(c_scores)))
        if auroc > best_auroc:
            best_auroc, best_epoch = auroc, ep
            no_improve_epochs = 0
            save_gmic_finetune_checkpoint(
                best_pt, model, opt, sched, scaler,
                epoch=ep, best_auroc=best_auroc, best_epoch=best_epoch,
                history=history, sampler=sampler, aug_rng=aug_rng, strategy=strategy, seed=seed,
            )
        else:
            no_improve_epochs += 1

        history.append({"epoch": ep, "mean_loss": float(np.mean(ep_losses)), "train_calib_auroc": auroc})
        save_gmic_finetune_checkpoint(
            latest_pt, model, opt, sched, scaler,
            epoch=ep, best_auroc=best_auroc, best_epoch=best_epoch,
            history=history, sampler=sampler, aug_rng=aug_rng, strategy=strategy, seed=seed,
        )
        RSNA_VOL.commit()
        if early_stopping_patience > 0 and ep >= min_epochs and no_improve_epochs >= early_stopping_patience:
            print(f"[GMIC EarlyStop] Stopping at epoch {ep}; best_epoch={best_epoch} (AUROC={best_auroc:.5f})", flush=True)
            break

    best_sha = compute_sha256_file(best_pt) if best_pt.exists() else ""
    return {
        "stage": "train",
        "strategy": strategy,
        "best_epoch": best_epoch,
        "best_train_calib_auroc": round(best_auroc, 6),
        "best_checkpoint_path": str(best_pt),
        "best_checkpoint_sha256": best_sha,
        "param_counts": param_counts,
        "history": history,
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    ephemeral_disk=524288,
    timeout=7200,
)
def eval_gmic_finetuned_shard(
    split_name: str,
    shard_idx: int,
    num_shards: int,
    checkpoint_path: str,
    expected_checkpoint_sha256: str = "",
    batch_size: int = 1,
    max_shard_cost_usd: float = 2.0,
) -> Dict[str, Any]:
    """Evaluation-only resumable shard worker for `gmic_rsna_finetuned` (zero optimizer, SHA-256 verified)."""
    import cv2
    import numpy as np
    import pandas as pd
    import resource
    import torch
    from src.evaluation.stage_c_eval import (
        assert_no_forbidden_columns,
        compute_sha256_file,
        load_label_free_inference_manifest,
    )
    from src.paper1.gmic_adapter import (
        GMIC_INPUT_SIZE,
        compute_item_ids_sha256,
        standard_normalize_gmic_uint16,
        validate_gmic_shard_payload,
    )
    from src.paper1.gmic_finetune import load_gmic_finetuned_model_for_eval

    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    batch_size = int(batch_size)
    RSNA_VOL.reload()
    t_container0 = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L40S GPU expected for GMIC fine-tuned eval!")

    ckpt_p = Path(checkpoint_path)
    model, _, ckpt_meta = load_gmic_finetuned_model_for_eval(
        ckpt_p,
        expected_sha256=expected_checkpoint_sha256.strip() or None,
        model_index="1",
        device=device,
        gmic_repo_dir=Path("/root/external/GMIC"),
    )
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Eval-only invariant violated: model must be eval() with requires_grad=False.")
    ckpt_sha = ckpt_meta["checkpoint_sha256"]

    cache_root = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / split_name
    vol_png_dir = cache_root / "png_2944x1920"
    manifest_csv = cache_root / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Step 1 GMIC manifest missing: {manifest_csv}")
    step1_manifest_sha = compute_sha256_file(manifest_csv)
    step1_df = pd.read_csv(manifest_csv, dtype={"item_id": str, "case_id": str})
    step1_map = step1_df.set_index("item_id")[
        ["crop_fallback", "crop_error", "fallback_strategy", "photometric"]
    ].to_dict("index")

    if split_name == "rsna_val_calib":
        df = pd.read_csv(
            "/root/datasets/rsna/splits_seed42.csv",
            usecols=["image_id", "patient_id", "split", "laterality", "view"],
        )
        assert_no_forbidden_columns(df.columns, context="eval_gmic_finetuned_rsna")
        df = df[df["split"] == "val_calib"].sort_values(["patient_id", "image_id"]).reset_index(drop=True)
        rows = [
            {"item_id": str(r.image_id), "case_id": str(r.patient_id), "view_slot": f"{r.laterality}-{r.view}"}
            for r in df.iloc[shard_idx::num_shards].itertuples(index=False)
        ]
    elif split_name == "nlbs":
        df = load_label_free_inference_manifest(Path("/root/datasets/nlbs/inference_manifest_stage_c.csv"))
        rows = [
            {"item_id": str(r.image_uid), "case_id": str(r.StudyInstanceUID), "view_slot": f"{r.laterality}-{r.view_position}"}
            for r in df.iloc[shard_idx::num_shards].itertuples(index=False)
        ]
    else:
        raise ValueError(f"Unsupported split_name={split_name!r}")

    expected_item_ids = [r["item_id"] for r in rows]
    expected_item_ids_sha = compute_item_ids_sha256(expected_item_ids)

    out_dir = RSNA_MOUNT / "cache" / "paper1_gmic_finetune_shards" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_file = out_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}_bs{batch_size}_{ckpt_sha[:12]}.json"
    if shard_file.exists():
        try:
            cached = json.loads(shard_file.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
        ok, _ = validate_gmic_shard_payload(
            cached,
            expected_item_ids=expected_item_ids,
            batch_size=batch_size,
            split_name=split_name,
            shard_idx=shard_idx,
            num_shards=num_shards,
            model_id=GMIC_FINETUNED_MODEL_ID,
            checkpoint_sha256=ckpt_sha,
            preprocess_version=PREPROCESS_VERSION,
            preprocess_manifest_sha256=step1_manifest_sha,
        )
        if ok:
            cached["resumed_from_cache"] = True
            return cached

    nvme_dir = Path(f"/tmp/gmic_ft_eval_{split_name}_s{shard_idx:02d}")
    nvme_stage_sec = _stage_pngs_nvme(expected_item_ids, vol_png_dir, nvme_dir, max_workers=16)

    def _load_norm(row: Dict[str, str]) -> np.ndarray:
        arr = cv2.imread(str(nvme_dir / f"{row['item_id']}.png"), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.shape != GMIC_INPUT_SIZE or arr.dtype != np.uint16:
            raise RuntimeError(f"Invalid Step 1 cached PNG: {row['item_id']}")
        return standard_normalize_gmic_uint16(arr)

    torch.cuda.reset_peak_memory_stats()
    t_loop0 = time.perf_counter()
    decode_norm_sec = 0.0
    gpu_infer_sec = 0.0
    records = []
    fallback_logs = []
    prefetch_limit = max(24, batch_size * 4)

    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = deque()
        next_submit = 0
        while next_submit < len(rows) and len(pending) < prefetch_limit:
            pending.append((rows[next_submit], pool.submit(_load_norm, rows[next_submit])))
            next_submit += 1

        for s in range(0, len(rows), batch_size):
            spent = ((time.perf_counter() - t_container0) / 3600.0) * L40S_HOURLY_USD
            if spent >= max_shard_cost_usd:
                raise RuntimeError(f"COST GUARD STOP (GMIC eval {split_name} s{shard_idx}): ${spent:.4f} >= ${max_shard_cost_usd:.2f}")

            b_rows = rows[s : s + batch_size]
            t_d0 = time.perf_counter()
            norms = [pending.popleft()[1].result() for _ in b_rows]
            while next_submit < len(rows) and len(pending) < prefetch_limit:
                pending.append((rows[next_submit], pool.submit(_load_norm, rows[next_submit])))
                next_submit += 1
            if batch_size == 1:
                bx = torch.from_numpy(norms[0][np.newaxis, np.newaxis, :, :]).to(device, non_blocking=True)
            else:
                bx = torch.from_numpy(np.stack(norms, axis=0)[:, np.newaxis, :, :]).to(device, non_blocking=True)
            decode_norm_sec += time.perf_counter() - t_d0

            torch.cuda.synchronize()
            t_g0 = time.perf_counter()
            with torch.no_grad():
                outs = model(bx).float().cpu().numpy()
            torch.cuda.synchronize()
            gpu_infer_sec += time.perf_counter() - t_g0

            if not np.all(np.isfinite(outs)):
                raise RuntimeError(f"Non-finite GMIC eval output in {split_name} shard {shard_idx}")

            for row, pred in zip(b_rows, outs):
                fb_meta = step1_map[row["item_id"]]
                is_fb = bool(fb_meta["crop_fallback"])
                if is_fb:
                    fallback_logs.append(
                        {
                            "item_id": row["item_id"],
                            "case_id": row["case_id"],
                            "view_slot": row["view_slot"],
                            "photometric": str(fb_meta.get("photometric", "")),
                            "fallback_strategy": str(fb_meta.get("fallback_strategy", "")),
                            "crop_error": str(fb_meta.get("crop_error", "")),
                        }
                    )
                records.append(
                    {
                        "item_id": row["item_id"],
                        "case_id": row["case_id"],
                        "view_slot": row["view_slot"],
                        "benign_pred": float(pred[0]),
                        "malignant_pred": float(pred[1]),
                        "crop_fallback": is_fb,
                    }
                )

    elapsed_sec = time.perf_counter() - t_loop0
    container_wall_sec = time.perf_counter() - t_container0
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    host_rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))
    estimated_cost_usd = (container_wall_sec / 3600.0) * L40S_HOURLY_USD

    payload = {
        "schema_version": "paper1_gmic_shard_v2",
        "model_id": GMIC_FINETUNED_MODEL_ID,
        "checkpoint_file": ckpt_p.name,
        "checkpoint_sha256": ckpt_sha,
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_manifest_sha256": step1_manifest_sha,
        "split_name": split_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "batch_size": batch_size,
        "expected_n_images": len(expected_item_ids),
        "expected_item_ids_sha256": expected_item_ids_sha,
        "nvme_stage_sec": round(nvme_stage_sec, 3),
        "decode_norm_sec": round(decode_norm_sec, 3),
        "gpu_infer_sec": round(gpu_infer_sec, 3),
        "elapsed_sec": round(elapsed_sec, 3),
        "container_wall_sec": round(container_wall_sec, 3),
        "l40s_hourly_usd": L40S_HOURLY_USD,
        "estimated_cost_usd": round(estimated_cost_usd, 5),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "host_rss_gb": round(host_rss_gb, 3),
        "n_fallback_views": len(fallback_logs),
        "fallback_logs": fallback_logs,
        "records": records,
    }
    ok, reason = validate_gmic_shard_payload(
        payload,
        expected_item_ids=expected_item_ids,
        batch_size=batch_size,
        split_name=split_name,
        shard_idx=shard_idx,
        num_shards=num_shards,
        model_id=GMIC_FINETUNED_MODEL_ID,
        checkpoint_sha256=ckpt_sha,
        preprocess_version=PREPROCESS_VERSION,
        preprocess_manifest_sha256=step1_manifest_sha,
    )
    if not ok:
        raise RuntimeError(f"Generated GMIC eval shard failed self-validation: {reason}")

    tmp = shard_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(shard_file)
    RSNA_VOL.commit()
    payload["resumed_from_cache"] = False
    return payload


@app.local_entrypoint()
def main(
    stage: str = "pilot",
    strategy: str = "partial",
    epochs: int = 8,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    steps_per_epoch: int = 0,
    lr: float = 1e-4,
    early_stopping_patience: int = 3,
    precision: str = "bf16",
    bench_batch_sizes: str = "8,16,32,64,128,256",
    bench_precisions: str = "bf16,fp16,tf32",
    lr_scaling: str = "sqrt",
    warmup_steps: int = 100,
    split: str = "rsna_val_calib",
    num_shards: int = 4,
    shard_indices: str = "",
    checkpoint_path: str = "/rsna_vol/models/paper1/checkpoints/gmic_rsna_partial_seed42_best_train_calib.pt",
    checkpoint_sha256: str = "",
    max_spend_usd: float = 10.0,
    approve_full_train: bool = False,
    approve_eval: bool = False,
) -> None:
    import pandas as pd
    from src.evaluation.stage_c_eval import assert_no_forbidden_columns, load_label_free_inference_manifest
    from src.paper1.gmic_adapter import finalize_and_lock_gmic_predictions

    validate_gmic_finetune_cli_guard(
        stage=stage,
        approve_full_train=approve_full_train,
        approve_eval=approve_eval,
        max_spend_usd=max_spend_usd,
    )

    if stage == "eval":
        validate_gmic_eval_cli_args(
            split=split,
            batch_size=batch_size,
            num_shards=num_shards,
            checkpoint_path=checkpoint_path,
            approve_eval=approve_eval,
            max_spend_usd=max_spend_usd,
        )
        if split == "nlbs":
            calib_lock_p = Path(f"reports/paper1/expanded/exp1/{GMIC_FINETUNED_MODEL_ID}_rsna_calibration_lock.json")
            verify_rsna_calibration_lock(
                calib_lock_p,
                expected_model_id=GMIC_FINETUNED_MODEL_ID,
                expected_checkpoint_sha256=checkpoint_sha256.strip() or None,
            )

        if split == "rsna_val_calib":
            exp_df = pd.read_csv(
                "datasets/rsna/splits_seed42.csv",
                usecols=["image_id", "patient_id", "split", "laterality", "view"],
            )
            assert_no_forbidden_columns(exp_df.columns, context="local_eval_gmic_ft_rsna")
            exp_df = exp_df[exp_df["split"] == "val_calib"].reset_index(drop=True)
            item_col, case_col, case_out_col = "image_id", "patient_id", "patient_id"
        else:
            exp_df = load_label_free_inference_manifest(Path("datasets/nlbs/inference_manifest_stage_c.csv"))
            item_col, case_col, case_out_col = "image_uid", "StudyInstanceUID", "StudyInstanceUID"

        shards = (
            [int(x.strip()) for x in shard_indices.split(",") if x.strip()]
            if shard_indices.strip()
            else list(range(num_shards))
        )
        per_shard_cap = min(2.0, float(max_spend_usd))
        outs = list(
            eval_gmic_finetuned_shard.starmap(
                [(split, s, num_shards, checkpoint_path, checkpoint_sha256, batch_size, per_shard_cap) for s in shards]
            )
        )
        if len(shards) < num_shards or set(shards) != set(range(num_shards)):
            print(f"[Partial GMIC eval {shards}/{num_shards}] Skipping prediction lock until all shards finish.")
            return

        ckpt_sha = str(outs[0]["checkpoint_sha256"])
        out_dir = Path("reports/paper1/expanded/exp1/predictions_locked")
        lock_manifest = finalize_and_lock_gmic_predictions(
            split_name=split,
            shard_outputs=outs,
            expected_image_df=exp_df,
            item_col=item_col,
            case_col=case_col,
            case_out_col=case_out_col,
            batch_size=batch_size,
            model_id=GMIC_FINETUNED_MODEL_ID,
            checkpoint_file=Path(checkpoint_path).name,
            checkpoint_sha256=ckpt_sha,
            preprocess_version=PREPROCESS_VERSION,
            out_dir=out_dir,
        )
        print(json.dumps(lock_manifest, indent=2))
        return

    res = run_gmic_finetune_l40s.remote(
        stage=stage,
        strategy=strategy,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        early_stopping_patience=early_stopping_patience,
        max_spend_usd=max_spend_usd,
        precision=precision,
        bench_batch_sizes=bench_batch_sizes,
        bench_precisions=bench_precisions,
        lr_scaling=lr_scaling,
        warmup_steps=warmup_steps,
    )
    print(json.dumps(res, indent=2))
