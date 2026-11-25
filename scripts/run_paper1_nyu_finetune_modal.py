"""Paper 1 Supplementary Exploratory — NYU Four-View RSNA Fine-Tuning & Evaluation Modal Entrypoint.

Trains `nyu_fourview_rsna_finetuned` initialized from `sample_image_model.p`
strictly on RSNA `train` 4-view exams (8,339 pts), selects best epoch on
`train_calib` (1,192 pts) AUROC, and keeps `val_calib` + NLBS strictly held out.

Provides:
1. `--stage pilot` / `--stage train`: online fine-tuning from `/rsna_vol/cache/paper1_nyu_uint16_v1/{rsna_train,rsna_train_calib}/png/`
2. `--stage extract_head_features` + `--stage train_cached_head`: fast frozen-backbone feature caching for `head_only`
3. `--stage eval`: evaluation-only 10-TTA (`seed=0`) FP32 inference with SHA-256 checkpoint verification,
   zero optimizer updates, resumable shards, RSNA-first calibration lock gate, and SHA-256 locked predictions.
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
from src.paper1.nyu_finetune import (
    NYU_FINETUNED_MODEL_ID,
    validate_nyu_eval_cli_args,
    validate_nyu_finetune_cli_guard,
)
from src.paper1.specialist_eval import verify_rsna_calibration_lock
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-nyu-rsna-finetune"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")
NYU_CKPT_SHA256 = "9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014"
PREPROCESS_VERSION = "paper1_nyu_uint16_v1"
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
    .add_local_file(
        "datasets/paper1/rsna_val_calib_4view_selection.csv",
        remote_path="/root/datasets/paper1/rsna_val_calib_4view_selection.csv",
    )
    .add_local_file(
        "datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
        remote_path="/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
    )
)


def _stage_pngs_nvme(item_ids: List[str], src_dir: Path, nvme_dir: Path, max_workers: int = 16) -> float:
    nvme_dir.mkdir(parents=True, exist_ok=True)
    missing = [i for i in item_ids if not (nvme_dir / f"{i}.png").exists()]
    t0 = time.perf_counter()
    if missing:

        def _cp(img_id: str) -> None:
            s = src_dir / f"{img_id}.png"
            d = nvme_dir / f"{img_id}.png"
            if not s.exists() or s.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty NYU PNG: {s}")
            shutil.copy2(s, d)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_cp, missing))
    return time.perf_counter() - t0


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
            png_path=png_dir / f"{row['item_id']}.png",
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
    bench_batch_exams: str = "2,4,8,16",
) -> Dict[str, Any]:
    """Run NYU Four-View L40S progressive pilot or RSNA fine-tuning on native cached PNGs."""
    import numpy as np
    import pandas as pd
    import torch
    from src.data.rsna_dataset import BalancedBatchSampler
    from src.evaluation.stage_c_eval import compute_auroc, compute_sha256_file
    from src.paper1.nyu_finetune import (
        compute_nyu_rsna_loss,
        configure_nyu_trainable_params,
        load_nyu_finetune_checkpoint,
        parse_nyu_batch_sizes,
        run_progressive_nyu_batch_benchmark,
        save_nyu_finetune_checkpoint,
        select_rsna_4view_split_exams,
        set_nyu_backbone_eval_mode,
        verify_nyu_multidraw_feature_equivalence,
    )
    from src.paper1.nyu_fourview_adapter import (
        NYU_VIEWS,
        build_nyu_fourview_model,
        score_nyu_fourview_exam_batch,
    )

    if stage not in ("pilot", "train"):
        raise ValueError(f"run_nyu_finetune_l40s supports stage='pilot' or 'train' only, got {stage!r}")

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

    train_cache = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / "rsna_train"
    if stage == "pilot" and not (train_cache / "manifest.csv").exists():
        pilot_cache = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / "rsna_train_pilot"
        if (pilot_cache / "manifest.csv").exists():
            train_cache = pilot_cache
    tc_cache = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / "rsna_train_calib"
    if not (train_cache / "manifest.csv").exists() or (stage != "pilot" and not (tc_cache / "manifest.csv").exists()):
        raise FileNotFoundError(
            f"Missing native NYU train/train_calib cache: {train_cache} / {tc_cache}. "
            "Run `scripts/preprocess_paper1_nyu_cpu.py --stage build --split train_all --approve-full-build` first."
        )

    splits_df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
    _, train_exams = select_rsna_4view_split_exams(splits_df, "train")
    _, tc_exams = select_rsna_4view_split_exams(splits_df, "train_calib")
    train_s1 = pd.read_csv(train_cache / "manifest.csv", dtype=str)
    tc_s1 = pd.read_csv(tc_cache / "manifest.csv", dtype=str) if (tc_cache / "manifest.csv").exists() else pd.DataFrame(columns=["case_id"])

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
        from src.paper1.nyu_fourview_adapter import load_nyu_cached_view, prepare_nyu_exam_tta_tensors

        bench_bs_list = parse_nyu_batch_sizes(bench_batch_exams)
        pids = [p for p in train_exams["patient_id"].astype(str).tolist() if p in train_by_pid]
        pool_n = min(len(pids), max(48, max(bench_bs_list) * 3))
        pilot_pids = pids[:pool_n]
        pilot_item_ids = [str(r["item_id"]) for p in pilot_pids for r in train_by_pid[p]]
        nvme_pilot = Path("/tmp/nyu_pilot_png")
        shutil.rmtree(nvme_pilot, ignore_errors=True)
        stage_sec = _stage_pngs_nvme(pilot_item_ids, train_cache / "png", nvme_pilot, max_workers=16)

        with ThreadPoolExecutor(max_workers=6) as pool:

            def _extract_loader(be: int, step_idx: int) -> Dict[str, np.ndarray]:
                start = (step_idx * be) % pool_n
                batch_pids = [pilot_pids[(start + k) % pool_n] for k in range(be)]
                ex_list = list(
                    pool.map(
                        lambda it: _prepare_single_draw_exam(
                            train_by_pid[it[1]], nvme_pilot, nyu_mods, True, seed + step_idx * 100 + it[0]
                        ),
                        enumerate(batch_pids),
                    )
                )
                return {v: np.concatenate([ex[v] for ex in ex_list], axis=0) for v in NYU_VIEWS}

            def _eval10tta_loader(be: int, step_idx: int) -> Dict[str, np.ndarray]:
                start = (step_idx * be) % pool_n
                batch_pids = [pilot_pids[(start + k) % pool_n] for k in range(be)]

                def _prep10(p: str) -> Dict[str, np.ndarray]:
                    vc = {
                        str(r["view_slot"]): load_nyu_cached_view(
                            nvme_pilot / f"{r['item_id']}.png",
                            (int(r["best_center_y"]), int(r["best_center_x"])),
                        )
                        for r in train_by_pid[p]
                    }
                    return prepare_nyu_exam_tta_tensors(vc, nyu_mods, num_tta_epochs=10, seed=0)

                ex_list = list(pool.map(_prep10, batch_pids))
                return {v: np.concatenate([ex[v] for ex in ex_list], axis=0) for v in NYU_VIEWS}

            extract_bench = run_progressive_nyu_batch_benchmark(
                model=model,
                exam_batch_loader=_extract_loader,
                device=device,
                batch_sizes=bench_bs_list,
                mode="extract",
                warmup_steps=1,
                bench_steps=max(2, min(int(pilot_steps), 3)),
                draws_per_exam=1,
            )
            eval_bench = run_progressive_nyu_batch_benchmark(
                model=model,
                exam_batch_loader=_eval10tta_loader,
                device=device,
                batch_sizes=bench_bs_list,
                mode="eval",
                warmup_steps=1,
                bench_steps=2,
                draws_per_exam=10,
            )

        # Verify mathematical equivalence of multi-draw augmented feature caching on L40S
        eq_draws = []
        for idx in range(min(2, pool_n)):
            vc = {
                str(r["view_slot"]): load_nyu_cached_view(
                    nvme_pilot / f"{r['item_id']}.png",
                    (int(r["best_center_y"]), int(r["best_center_x"])),
                )
                for r in train_by_pid[pilot_pids[idx]]
            }
            eq_draws.append(prepare_nyu_exam_tta_tensors(vc, nyu_mods, num_tta_epochs=4, seed=seed + idx))
        eq_batch = {v: np.concatenate([d[v] for d in eq_draws], axis=0) for v in NYU_VIEWS}
        eq_report = verify_nyu_multidraw_feature_equivalence(model, eq_batch, num_draws=4, atol=1e-3)

        rec_ex = extract_bench.get("recommended")
        rec_ev = eval_bench.get("recommended")
        est_extract_min = (
            ((len(train_exams) * 4 + len(tc_exams) * 10) / max(rec_ex["exams_per_sec"], 1e-6)) / 60.0
            if rec_ex
            else None
        )
        pilot_report = {
            "stage": "pilot",
            "strategy": strategy,
            "param_counts": {"total": n_total, "trainable": n_train},
            "nvme_stage_pilot_sec": round(stage_sec, 3),
            "extract_benchmark": extract_bench,
            "eval_10tta_benchmark": eval_bench,
            "multidraw_equivalence": eq_report,
            "projected_extract_minutes": round(est_extract_min, 2) if est_extract_min is not None else None,
            "projected_extract_usd": round((est_extract_min / 60.0) * L40S_PROD_HOURLY_USD, 2)
            if est_extract_min is not None
            else None,
            "recommended_eval_batch_exams": rec_ev["batch_exams"] if rec_ev else None,
        }
        ckpt_dir = RSNA_MOUNT / "models" / "paper1" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "nyu_pilot_benchmark_report.json").write_text(json.dumps(pilot_report, indent=2), encoding="utf-8")
        RSNA_VOL.commit()
        return pilot_report

    # Full training loop (stage PNGs to NVMe first)
    nvme_train = Path("/tmp/nyu_train_png")
    nvme_tc = Path("/tmp/nyu_tc_png")
    _stage_pngs_nvme(train_s1["item_id"].astype(str).tolist(), train_cache / "png", nvme_train)
    _stage_pngs_nvme(tc_s1["item_id"].astype(str).tolist(), tc_cache / "png", nvme_tc)

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

        set_nyu_backbone_eval_mode(model, strategy)
        ep_losses: List[float] = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for step_idx, indices in enumerate(sampler):
                ex_list = list(
                    pool.map(
                        lambda it: _prepare_single_draw_exam(
                            train_by_pid[pids[it[1]]],
                            nvme_train,
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
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i0 in range(0, len(tc_pids), batch_exams):
                chunk_pids = tc_pids[i0 : i0 + batch_exams]
                ex_list = list(
                    pool.map(
                        lambda p: _prepare_single_draw_exam(tc_by_pid[p], nvme_tc, nyu_mods, False, 0),
                        chunk_pids,
                    )
                )
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

    best_sha = compute_sha256_file(best_pt) if best_pt.exists() else ""
    return {
        "stage": "train",
        "strategy": strategy,
        "best_epoch": best_epoch,
        "best_train_calib_auroc": round(best_auroc, 6),
        "best_checkpoint_path": str(best_pt),
        "best_checkpoint_sha256": best_sha,
        "param_counts": {"total": n_total, "trainable": n_train},
        "history": history,
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    ephemeral_disk=524288,
    timeout=7200,
)
def extract_nyu_head_features_modal(
    n_train_draws: int = 4,
    num_tta_calib: int = 10,
    batch_exams: int = 4,
    seed: int = 42,
    max_spend_usd: float = 5.0,
) -> Dict[str, Any]:
    """Extract frozen-backbone 1024-d NYU feature caches (`~134-372 MiB`) for ultra-fast head_only training."""
    import numpy as np
    import pandas as pd
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file
    from src.paper1.nyu_finetune import extract_nyu_backbone_features, select_rsna_4view_split_exams
    from src.paper1.nyu_fourview_adapter import (
        NYU_VIEWS,
        build_nyu_fourview_model,
        load_nyu_cached_view,
        prepare_nyu_exam_tta_tensors,
    )

    RSNA_VOL.reload()
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_in = RSNA_MOUNT / "models" / "paper1" / "sample_image_model.p"
    if compute_sha256_file(ckpt_in).lower() != NYU_CKPT_SHA256:
        raise RuntimeError("NYU pretrained checkpoint SHA-256 mismatch.")

    model, nyu_mods = build_nyu_fourview_model(
        ckpt_in, device=device, nyu_repo_dir=Path("/root/external/breast_cancer_classifier")
    )
    model.eval()
    splits_df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
    out_root = RSNA_MOUNT / "cache" / "paper1_nyu_head_features_v1"
    out_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}

    for split_name, n_draws, use_aug in (("rsna_train", n_train_draws, True), ("rsna_train_calib", num_tta_calib, True)):
        pt_path = out_root / f"{split_name}_features.pt"
        rsna_split = split_name.replace("rsna_", "", 1)
        _, exam_df = select_rsna_4view_split_exams(splits_df, rsna_split)
        pids = exam_df["patient_id"].astype(str).tolist()
        if pt_path.exists():
            try:
                existing = torch.load(pt_path, map_location="cpu", weights_only=False)
                if (
                    existing.get("backbone_sha256") == NYU_CKPT_SHA256
                    and list(existing.get("patient_ids", [])) == pids
                    and tuple(existing["h_cc"].shape) == (len(pids), n_draws, 512)
                    and tuple(existing["h_mlo"].shape) == (len(pids), n_draws, 512)
                    and bool(torch.isfinite(existing["h_cc"]).all())
                    and bool(torch.isfinite(existing["h_mlo"]).all())
                ):
                    summary[split_name] = {
                        "path": str(pt_path),
                        "shape": list(existing["h_cc"].shape),
                        "mb": round(pt_path.stat().st_size / (1024**2), 2),
                        "resumed_from_cache": True,
                    }
                    print(f"[NYU-Feat] Reusing verified {pt_path} ({summary[split_name]['shape']})", flush=True)
                    continue
            except Exception as exc:
                print(f"[NYU-Feat] Ignoring incomplete/corrupt {pt_path}: {exc}", flush=True)

        s1_df = pd.read_csv(RSNA_MOUNT / "cache" / PREPROCESS_VERSION / split_name / "manifest.csv", dtype=str)
        by_pid = {str(k): grp.to_dict(orient="records") for k, grp in s1_df.groupby("case_id")}
        nvme_dir = Path(f"/tmp/nyu_feat_{split_name}")
        _stage_pngs_nvme(s1_df["item_id"].astype(str).tolist(), RSNA_MOUNT / "cache" / PREPROCESS_VERSION / split_name / "png", nvme_dir)

        h_cc_all = torch.empty((len(pids), n_draws, 512), dtype=torch.float32)
        h_mlo_all = torch.empty((len(pids), n_draws, 512), dtype=torch.float32)
        eff_batch_exams = min(int(batch_exams), max(1, 64 // int(n_draws)))

        def _prep_exam(idx: int) -> Dict[str, np.ndarray]:
            rows = by_pid[pids[idx]]
            vc = {
                str(r["view_slot"]): load_nyu_cached_view(
                    nvme_dir / f"{r['item_id']}.png",
                    (int(r["best_center_y"]), int(r["best_center_x"])),
                )
                for r in rows
            }
            ex_seed = 0 if split_name == "rsna_train_calib" else (seed + idx)
            return prepare_nyu_exam_tta_tensors(vc, nyu_mods, num_tta_epochs=n_draws if use_aug else 1, seed=ex_seed)

        with ThreadPoolExecutor(max_workers=6) as pool:
            for s0 in range(0, len(pids), eff_batch_exams):
                spent = ((time.perf_counter() - t0) / 3600.0) * L40S_PROD_HOURLY_USD
                if spent >= max_spend_usd:
                    raise RuntimeError(f"COST GUARD STOP during NYU feature extraction: ${spent:.2f}")
                e_idx = list(range(s0, min(s0 + eff_batch_exams, len(pids))))
                prepped = list(pool.map(_prep_exam, e_idx))
                tbatch = {
                    v: (
                        torch.from_numpy(np.concatenate([p[v] for p in prepped], axis=0))
                        .pin_memory()
                        .to(device, non_blocking=True)
                        if device.type == "cuda"
                        else torch.from_numpy(np.concatenate([p[v] for p in prepped], axis=0)).to(device)
                    )
                    for v in NYU_VIEWS
                }
                h_cc, h_mlo = extract_nyu_backbone_features(model, tbatch)
                h_cc_all[s0 : s0 + len(e_idx)] = h_cc.cpu().reshape(len(e_idx), n_draws, 512)
                h_mlo_all[s0 : s0 + len(e_idx)] = h_mlo.cpu().reshape(len(e_idx), n_draws, 512)

        tmp_pt = pt_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "split_name": split_name,
                "patient_ids": pids,
                "left_malignant": exam_df["left_malignant"].values.astype(int).tolist(),
                "right_malignant": exam_df["right_malignant"].values.astype(int).tolist(),
                "exam_cancer": exam_df["exam_cancer"].values.astype(int).tolist(),
                "h_cc": h_cc_all,
                "h_mlo": h_mlo_all,
                "backbone_sha256": NYU_CKPT_SHA256,
            },
            tmp_pt,
        )
        tmp_pt.replace(pt_path)
        RSNA_VOL.commit()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        summary[split_name] = {
            "path": str(pt_path),
            "shape": list(h_cc_all.shape),
            "mb": round(pt_path.stat().st_size / (1024**2), 2),
            "eff_batch_exams": eff_batch_exams,
            "resumed_from_cache": False,
        }
    return summary


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def train_nyu_cached_head_cpu(
    epochs: int = 8,
    batch_exams: int = 16,
    steps_per_epoch: int = 400,
    lr: float = 1e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train NYU `head_only` classifier on CPU using cached frozen-backbone feature tensors."""
    import numpy as np
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file
    from src.paper1.nyu_finetune import train_nyu_head_from_cached_features
    from src.paper1.nyu_fourview_adapter import build_nyu_fourview_model

    RSNA_VOL.reload()
    ckpt_in = RSNA_MOUNT / "models" / "paper1" / "sample_image_model.p"
    if compute_sha256_file(ckpt_in).lower() != NYU_CKPT_SHA256:
        raise RuntimeError("NYU pretrained checkpoint SHA-256 mismatch.")

    feat_root = RSNA_MOUNT / "cache" / "paper1_nyu_head_features_v1"
    tr = torch.load(feat_root / "rsna_train_features.pt", map_location="cpu", weights_only=False)
    tc = torch.load(feat_root / "rsna_train_calib_features.pt", map_location="cpu", weights_only=False)
    model, _ = build_nyu_fourview_model(
        ckpt_in, device=torch.device("cpu"), nyu_repo_dir=Path("/root/external/breast_cancer_classifier")
    )
    res = train_nyu_head_from_cached_features(
        model=model,
        train_h_cc=tr["h_cc"],
        train_h_mlo=tr["h_mlo"],
        train_lm=np.asarray(tr["left_malignant"], dtype=int),
        train_rm=np.asarray(tr["right_malignant"], dtype=int),
        train_exam_cancer=np.asarray(tr["exam_cancer"], dtype=int),
        tc_h_cc=tc["h_cc"],
        tc_h_mlo=tc["h_mlo"],
        tc_exam_cancer=np.asarray(tc["exam_cancer"], dtype=int),
        out_dir=RSNA_MOUNT / "models" / "paper1" / "checkpoints",
        epochs=epochs,
        batch_exams=batch_exams,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        seed=seed,
        device=torch.device("cpu"),
    )
    RSNA_VOL.commit()
    return res


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    ephemeral_disk=524288,
    timeout=7200,
)
def eval_nyu_finetuned_shard(
    split_name: str,
    shard_idx: int,
    num_shards: int,
    checkpoint_path: str,
    expected_checkpoint_sha256: str = "",
    batch_exams: int = 4,
    num_tta_epochs: int = 10,
    max_shard_cost_usd: float = 2.0,
) -> Dict[str, Any]:
    """Evaluation-only resumable shard worker for `nyu_fourview_rsna_finetuned` (10-TTA FP32, zero optimizer)."""
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    import resource
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file
    from src.paper1.nyu_finetune import load_nyu_finetuned_model_for_eval
    from src.paper1.nyu_fourview_adapter import (
        NYU_STEP1_MANIFEST_SHA256,
        NYU_TTA_SEED_OFFICIAL,
        NYU_VIEWS,
        compute_nyu_case_ids_sha256,
        compute_text_lf_sha256,
        load_nyu_cached_view,
        prepare_nyu_exam_tta_tensors,
        score_nyu_fourview_exam_batch,
        validate_and_load_nyu_4view_manifest,
        validate_nyu_shard_payload,
        verify_nyu_step1_against_frozen_source,
    )

    if int(num_tta_epochs) != 10:
        raise ValueError(f"Official protocol requires num_tta_epochs=10, got {num_tta_epochs}")
    batch_exams = int(batch_exams)
    RSNA_VOL.reload()
    t_container0 = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L40S GPU expected for NYU fine-tuned eval!")

    ckpt_p = Path(checkpoint_path)
    model, nyu_mods, ckpt_meta = load_nyu_finetuned_model_for_eval(
        ckpt_p,
        expected_sha256=expected_checkpoint_sha256.strip() or None,
        device=device,
        nyu_repo_dir=Path("/root/external/breast_cancer_classifier"),
    )
    if next(model.parameters()).dtype != torch.float32 or model.training or torch.is_autocast_enabled():
        raise RuntimeError("Strict FP32 eval() non-autocast invariant violated for NYU eval!")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Eval-only invariant violated: requires_grad must be False.")
    ckpt_sha = ckpt_meta["checkpoint_sha256"]

    cache_root = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / split_name
    vol_png_dir = cache_root / "png"
    man_csv = cache_root / "manifest.csv"
    if not man_csv.exists():
        raise FileNotFoundError(f"Step 1 NYU manifest missing: {man_csv}")
    step1_manifest_sha = compute_text_lf_sha256(man_csv)
    expected_frozen_sha = NYU_STEP1_MANIFEST_SHA256[split_name]
    step1_df = pd.read_csv(man_csv, dtype={"item_id": str, "case_id": str})

    raw_man_path = (
        Path("/root/datasets/paper1/rsna_val_calib_4view_selection.csv")
        if split_name == "rsna_val_calib"
        else Path("/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv")
    )
    src_df = validate_and_load_nyu_4view_manifest(split_name, raw_man_path)
    verify_nyu_step1_against_frozen_source(
        split_name, step1_df, src_df, actual_manifest_sha256=step1_manifest_sha, expected_manifest_sha256=expected_frozen_sha
    )

    all_exams: List[Tuple[str, List[Dict[str, Any]]]] = []
    for cid, grp in step1_df.groupby("case_id", sort=True):
        recs = grp.to_dict(orient="records")
        if len(recs) != 4:
            raise RuntimeError(f"Incomplete exam {cid}: expected 4 views")
        all_exams.append((str(cid), recs))

    shard_exams = all_exams[shard_idx::num_shards]
    expected_case_ids = [cid for cid, _ in shard_exams]
    expected_case_ids_sha = compute_nyu_case_ids_sha256(expected_case_ids)

    out_dir = RSNA_MOUNT / "cache" / "paper1_nyu_finetune_shards" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_file = out_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}_be{batch_exams}_{ckpt_sha[:12]}.json"
    if shard_file.exists():
        try:
            cached = json.loads(shard_file.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
        ok, _ = validate_nyu_shard_payload(
            cached,
            expected_case_ids=expected_case_ids,
            batch_exams=batch_exams,
            split_name=split_name,
            shard_idx=shard_idx,
            num_shards=num_shards,
            num_tta_epochs=num_tta_epochs,
            tta_seed=NYU_TTA_SEED_OFFICIAL,
            model_id=NYU_FINETUNED_MODEL_ID,
            checkpoint_sha256=ckpt_sha,
            preprocess_version=PREPROCESS_VERSION,
            preprocess_manifest_sha256=expected_frozen_sha,
        )
        if ok:
            cached["resumed_from_cache"] = True
            return cached

    nvme_dir = Path(f"/tmp/nyu_ft_eval_{split_name}_s{shard_idx:02d}")
    shard_item_ids = [str(r["item_id"]) for _, vr in shard_exams for r in vr]
    nvme_stage_sec = _stage_pngs_nvme(shard_item_ids, vol_png_dir, nvme_dir, max_workers=16)

    def _prep_one_exam(exam_item: Tuple[str, List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        cid, view_rows = exam_item
        view_crops = {
            str(r["view_slot"]): load_nyu_cached_view(
                nvme_dir / f"{r['item_id']}.png",
                best_center=(int(r["best_center_y"]), int(r["best_center_x"])),
                expected_shape=(int(r["height"]), int(r["width"])),
                expected_u16_sha256=str(r["u16_sha256"]),
            )
            for r in view_rows
        }
        tta_dict = prepare_nyu_exam_tta_tensors(
            view_crops=view_crops, nyu_mods=nyu_mods, num_tta_epochs=num_tta_epochs, seed=NYU_TTA_SEED_OFFICIAL
        )
        return cid, view_rows, tta_dict

    torch.cuda.reset_peak_memory_stats()
    t_loop0 = time.perf_counter()
    decode_tta_sec = 0.0
    gpu_infer_sec = 0.0
    records = []
    fallback_logs = []
    prefetch_limit = max(batch_exams + 2, 6)

    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = deque()
        next_idx = 0
        while next_idx < len(shard_exams) and len(pending) < prefetch_limit:
            pending.append(pool.submit(_prep_one_exam, shard_exams[next_idx]))
            next_idx += 1

        for start in range(0, len(shard_exams), batch_exams):
            spent = ((time.perf_counter() - t_container0) / 3600.0) * L40S_PROD_HOURLY_USD
            if spent >= max_shard_cost_usd:
                raise RuntimeError(f"COST GUARD STOP (NYU eval {split_name} s{shard_idx}): ${spent:.4f} >= ${max_shard_cost_usd:.2f}")

            t_d0 = time.perf_counter()
            b_count = min(batch_exams, len(shard_exams) - start)
            prepped = [pending.popleft().result() for _ in range(b_count)]
            while next_idx < len(shard_exams) and len(pending) < prefetch_limit:
                pending.append(pool.submit(_prep_one_exam, shard_exams[next_idx]))
                next_idx += 1
            decode_tta_sec += time.perf_counter() - t_d0

            torch.cuda.synchronize()
            t_g0 = time.perf_counter()
            preds_list = score_nyu_fourview_exam_batch(
                model=model, tta_exam_tensors=[p[2] for p in prepped], device=device
            )
            torch.cuda.synchronize()
            gpu_infer_sec += time.perf_counter() - t_g0

            for (cid, view_rows, _), preds in zip(prepped, preds_list):
                n_fb = 0
                for vr in view_rows:
                    if bool(vr["crop_fallback"]):
                        n_fb += 1
                        fallback_logs.append(
                            {
                                "split_name": str(vr["split_name"]),
                                "case_id": cid,
                                "item_id": str(vr["item_id"]),
                                "view_slot": str(vr["view_slot"]),
                                "photometric": str(vr.get("photometric", "")),
                                "fallback_strategy": str(vr.get("fallback_strategy", "")),
                                "crop_error": str(vr.get("crop_error", "")),
                            }
                        )
                records.append(
                    {
                        "case_id": cid,
                        "n_views": len(view_rows),
                        "n_crop_fallbacks": n_fb,
                        "has_crop_fallback": n_fb > 0,
                        **preds,
                    }
                )

    elapsed_sec = time.perf_counter() - t_loop0
    container_wall_sec = time.perf_counter() - t_container0
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    host_rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))
    estimated_cost_usd = (container_wall_sec / 3600.0) * L40S_PROD_HOURLY_USD

    payload = {
        "schema_version": "paper1_nyu_shard_v2",
        "model_id": NYU_FINETUNED_MODEL_ID,
        "checkpoint_file": ckpt_p.name,
        "checkpoint_sha256": ckpt_sha,
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_manifest_sha256": expected_frozen_sha,
        "precision": "fp32",
        "split_name": split_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "batch_exams": batch_exams,
        "num_tta_epochs": num_tta_epochs,
        "tta_seed": NYU_TTA_SEED_OFFICIAL,
        "expected_n_exams": len(expected_case_ids),
        "expected_n_images": len(expected_case_ids) * 4,
        "expected_case_ids_sha256": expected_case_ids_sha,
        "nvme_stage_sec": round(nvme_stage_sec, 3),
        "decode_tta_sec": round(decode_tta_sec, 3),
        "gpu_infer_sec": round(gpu_infer_sec, 3),
        "elapsed_sec": round(elapsed_sec, 3),
        "container_wall_sec": round(container_wall_sec, 3),
        "l40s_hourly_usd": L40S_PROD_HOURLY_USD,
        "estimated_cost_usd": round(estimated_cost_usd, 5),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "host_rss_gb": round(host_rss_gb, 3),
        "n_fallback_views": len(fallback_logs),
        "fallback_logs": fallback_logs,
        "records": records,
    }
    ok, reason = validate_nyu_shard_payload(
        payload,
        expected_case_ids=expected_case_ids,
        batch_exams=batch_exams,
        split_name=split_name,
        shard_idx=shard_idx,
        num_shards=num_shards,
        num_tta_epochs=num_tta_epochs,
        tta_seed=NYU_TTA_SEED_OFFICIAL,
        model_id=NYU_FINETUNED_MODEL_ID,
        checkpoint_sha256=ckpt_sha,
        preprocess_version=PREPROCESS_VERSION,
        preprocess_manifest_sha256=expected_frozen_sha,
    )
    if not ok:
        raise RuntimeError(f"Generated NYU eval shard failed self-validation: {reason}")

    tmp = shard_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(shard_file)
    RSNA_VOL.commit()
    payload["resumed_from_cache"] = False
    return payload


@app.local_entrypoint()
def main(
    stage: str = "pilot",
    strategy: str = "head_only",
    epochs: int = 8,
    batch_exams: int = 4,
    bench_batch_exams: str = "2,4,8,16",
    steps_per_epoch: int = 400,
    lr: float = 1e-4,
    split: str = "rsna_val_calib",
    num_shards: int = 4,
    num_tta_epochs: int = 10,
    shard_indices: str = "",
    checkpoint_path: str = "/rsna_vol/models/paper1/checkpoints/nyu_rsna_head_only_seed42_best_train_calib.pt",
    checkpoint_sha256: str = "",
    max_spend_usd: float = 12.0,
    approve_full_train: bool = False,
    approve_eval: bool = False,
) -> None:
    import pandas as pd
    from src.paper1.nyu_fourview_adapter import (
        NYU_STEP1_MANIFEST_SHA256,
        compute_text_lf_sha256,
        finalize_and_lock_nyu_predictions,
        validate_and_load_nyu_4view_manifest,
        verify_nyu_step1_against_frozen_source,
    )

    validate_nyu_finetune_cli_guard(
        stage=stage,
        approve_full_train=approve_full_train,
        approve_eval=approve_eval,
        max_spend_usd=max_spend_usd,
    )

    if stage == "extract_head_features":
        res = extract_nyu_head_features_modal.remote(
            n_train_draws=4, num_tta_calib=10, batch_exams=batch_exams, max_spend_usd=max_spend_usd
        )
        print(json.dumps(res, indent=2))
        return

    if stage == "train_cached_head":
        res = train_nyu_cached_head_cpu.remote(
            epochs=epochs, batch_exams=max(batch_exams, 16), steps_per_epoch=steps_per_epoch, lr=lr
        )
        print(json.dumps(res, indent=2))
        return

    if stage == "eval":
        validate_nyu_eval_cli_args(
            split=split,
            batch_exams=batch_exams,
            num_shards=num_shards,
            num_tta_epochs=num_tta_epochs,
            checkpoint_path=checkpoint_path,
            approve_eval=approve_eval,
            max_spend_usd=max_spend_usd,
        )
        if split == "nlbs_4view":
            calib_lock_p = Path(f"reports/paper1/expanded/exp1/{NYU_FINETUNED_MODEL_ID}_rsna_calibration_lock.json")
            verify_rsna_calibration_lock(
                calib_lock_p,
                expected_model_id=NYU_FINETUNED_MODEL_ID,
                expected_checkpoint_sha256=checkpoint_sha256.strip() or None,
            )

        step1_local_csv = Path(f"reports/paper1/nyu_step1_{split}_manifest.csv")
        if not step1_local_csv.exists():
            raise FileNotFoundError(f"Expected Step 1 local manifest CSV: {step1_local_csv}")
        local_sha = compute_text_lf_sha256(step1_local_csv)
        expected_sha = NYU_STEP1_MANIFEST_SHA256[split]
        step1_df = pd.read_csv(step1_local_csv, dtype={"item_id": str, "case_id": str})
        src_csv = Path(
            "datasets/paper1/rsna_val_calib_4view_selection.csv"
            if split == "rsna_val_calib"
            else "datasets/paper1/nlbs_4view_common_cohort_label_free.csv"
        )
        verify_nyu_step1_against_frozen_source(
            split,
            step1_df,
            validate_and_load_nyu_4view_manifest(split, src_csv),
            actual_manifest_sha256=local_sha,
            expected_manifest_sha256=expected_sha,
        )

        shards = (
            [int(x.strip()) for x in shard_indices.split(",") if x.strip()]
            if shard_indices.strip()
            else list(range(num_shards))
        )
        per_shard_cap = min(2.0, float(max_spend_usd))
        outs = list(
            eval_nyu_finetuned_shard.starmap(
                [(split, s, num_shards, checkpoint_path, checkpoint_sha256, batch_exams, num_tta_epochs, per_shard_cap) for s in shards]
            )
        )
        if len(shards) < num_shards or set(shards) != set(range(num_shards)):
            print(f"[Partial NYU eval {shards}/{num_shards}] Skipping prediction lock until all shards finish.")
            return

        ckpt_sha = str(outs[0]["checkpoint_sha256"])
        case_out_col = "patient_id" if split == "rsna_val_calib" else "StudyInstanceUID"
        lock_manifest = finalize_and_lock_nyu_predictions(
            split_name=split,
            shard_outputs=outs,
            expected_step1_df=step1_df,
            case_out_col=case_out_col,
            batch_exams=batch_exams,
            model_id=NYU_FINETUNED_MODEL_ID,
            checkpoint_file=Path(checkpoint_path).name,
            checkpoint_sha256=ckpt_sha,
            preprocess_version=PREPROCESS_VERSION,
            expected_manifest_sha256=expected_sha,
            out_dir=Path("reports/paper1/expanded/exp1/predictions_locked"),
        )
        print(json.dumps(lock_manifest, indent=2))
        return

    res = run_nyu_finetune_l40s.remote(
        stage=stage,
        strategy=strategy,
        epochs=epochs,
        batch_exams=batch_exams,
        steps_per_epoch=steps_per_epoch,
        lr=lr,
        max_spend_usd=max_spend_usd,
        bench_batch_exams=bench_batch_exams,
    )
    print(json.dumps(res, indent=2))
