"""Full TRUST & CE-Only Training & L40S Benchmark Pipeline on Modal.

Reference:
    TRUST (Hajishafiezahramini et al., 2026, arXiv:2609.00300): Section 3.1 - 3.11
    - Backbone: Mammo-CLIP EfficientNet-B5 (frozen, strict 852/852 parity).
    - Head: LayerNorm(2048) -> Dropout(0.3) -> Linear(2048, 1).
    - Resolution: 1520 x 912 (MONOCHROME1 inverted, Mammo-CLIP mean/std standardized).
    - Minibatch: 80 (20 positives, 60 negatives) via gradient accumulation.
    - Epoch Policy: "full_neg_pass" (default) visits 100% of negative images per epoch
      while cycling positives with reshuffled wraparound to preserve the 20:60 ratio.
    - Optimizer: AdamW, lr=3e-5, weight_decay=1e-4, grad_clip=5.0.
    - Loss: BCE + Focal Loss (gamma=2.0, alpha=0.25) + Dynamic Dismissal Loss (margin=0.05).
    - Dynamic tau_t updated on train_calib positive images at end of each epoch.
    - Per-epoch val_calib monitoring, _latest.pt / _best.pt checkpointing, and --resume.

Usage (DO NOT EXECUTE AUTOMATICALLY - for user execution):
    modal run scripts/train_trust_modal.py --benchmark --gpu L40S --benchmark-batches 10
    modal run scripts/train_trust_modal.py --baseline ce --epochs 20 --gpu L40S
    modal run scripts/train_trust_modal.py --baseline trust --epochs 20 --gpu L40S
    modal run scripts/train_trust_modal.py --baseline trust --epochs 20 --gpu L40S --resume
"""

import json
import math
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import modal

APP_NAME = "anchorshift-stage-b-train"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
MOUNT_PATH = Path("/data")
COMMIT_INTERVAL_SEC = 60
SUPPORTED_GPUS = ("T4", "L4", "A10G", "L40S", "A100", "H100")
DEFAULT_HOURLY_RATES = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100": 2.50,
    "H100": 3.95,
}

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


def _start_periodic_commit(vol: modal.Volume, stop_event: threading.Event) -> threading.Thread:
    def _loop() -> None:
        while not stop_event.wait(COMMIT_INTERVAL_SEC):
            try:
                vol.commit()
                print("[modal] Periodic volume.commit() completed.", flush=True)
            except Exception as exc:
                print(f"[modal] Periodic commit warning: {exc}", file=sys.stderr, flush=True)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def _register_signal_handlers(vol: modal.Volume, stop_event: threading.Event) -> None:
    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"\n[signal] Caught signal {signum}. Committing volume before exit...", flush=True)
        stop_event.set()
        try:
            vol.commit()
        except Exception:
            pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def _load_rsna_splits(rsna_raw: Path, splits_csv: Path, seed: int):
    """Load deterministic patient-level RSNA splits, generating if absent on volume."""
    import numpy as np
    import pandas as pd

    if splits_csv.exists():
        return pd.read_csv(splits_csv)

    df_full = pd.read_csv(rsna_raw / "train.csv")
    patient_cancer = df_full.groupby("patient_id")["cancer"].max().reset_index()
    rng = np.random.RandomState(seed)
    pos_pts = patient_cancer[patient_cancer["cancer"] == 1]["patient_id"].to_numpy().copy()
    neg_pts = patient_cancer[patient_cancer["cancer"] == 0]["patient_id"].to_numpy().copy()
    rng.shuffle(pos_pts)
    rng.shuffle(neg_pts)

    def _split(arr, f1: float, f2: float):
        n1 = int(round(len(arr) * f1))
        n2 = int(round(len(arr) * f2))
        return arr[:n1], arr[n1 : n1 + n2], arr[n1 + n2 :]

    p_tr, p_tc, p_vc = _split(pos_pts, 0.70, 0.10)
    n_tr, n_tc, n_vc = _split(neg_pts, 0.70, 0.10)
    p_map = {}
    for p in np.concatenate([p_tr, n_tr]):
        p_map[p] = "train"
    for p in np.concatenate([p_tc, n_tc]):
        p_map[p] = "train_calib"
    for p in np.concatenate([p_vc, n_vc]):
        p_map[p] = "val_calib"
    df_full["split"] = df_full["patient_id"].map(p_map)
    df_full["rel_path"] = (
        "train_images/"
        + df_full["patient_id"].astype(str)
        + "/"
        + df_full["image_id"].astype(str)
        + ".dcm"
    )
    df_full.to_csv(splits_csv, index=False)
    RSNA_VOL.commit()
    return df_full


def _build_model_and_criterion(
    mammo_clip_tar: Path,
    device,
    baseline: str,
    lr: float,
    margin: float,
    lambda_focal: float,
    lambda_dismiss: float,
):
    """Build strictly verified Mammo-CLIP encoder, linear head, optimizer, scaler, and loss."""
    import torch
    import torch.nn as nn
    from src.models.trust_model import MammoCLIPEncoder
    from src.models.mammo_clip_loader import load_and_verify_mammo_clip
    from src.models.losses import TRUSTLoss

    if not mammo_clip_tar.exists():
        raise FileNotFoundError(f"Mammo-CLIP checkpoint missing at {mammo_clip_tar}")

    encoder = MammoCLIPEncoder(checkpoint_path=None, freeze=False, model_name="efficientnet-b5")
    diag = load_and_verify_mammo_clip(encoder, mammo_clip_tar, strict=True)
    assert (
        diag["is_fully_compatible"]
        and diag["matched_keys_count"] == 852
        and diag["total_target_keys"] == 852
        and diag["missing_keys_count"] == 0
        and diag["unexpected_keys_count"] == 0
    ), f"Expected strict 852/852 Mammo-CLIP key parity, got: {diag}"
    encoder.freeze_encoder()

    head = nn.Sequential(
        nn.LayerNorm(encoder.embed_dim),
        nn.Dropout(0.3),
        nn.Linear(encoder.embed_dim, 1),
    ).to(device)
    encoder = encoder.to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    criterion = TRUSTLoss(
        baseline=baseline,
        margin=margin,
        lambda_focal=lambda_focal,
        lambda_dismiss=lambda_dismiss,
    )
    return encoder, head, optimizer, scaler, criterion, diag


def _decode_paths_parallel(rsna_raw: Path, rel_paths: List[str], num_workers: int = 8):
    """Decode and preprocess a list of DICOM relative paths in parallel."""
    import torch
    from src.data.preprocessing import preprocess_dicom

    def _one(rel: str):
        return preprocess_dicom(
            rsna_raw / rel,
            target_size=(1520, 912),
            invert_monochrome1=True,
            standardize=True,
        )

    if num_workers <= 1 or len(rel_paths) <= 1:
        return torch.stack([_one(r) for r in rel_paths])

    with ThreadPoolExecutor(max_workers=min(num_workers, len(rel_paths))) as pool:
        tensors = list(pool.map(_one, rel_paths))
    return torch.stack(tensors)


def _compute_batches_per_epoch(n_pos: int, n_neg: int, batch_size: int, epoch_policy: str) -> int:
    """Compute batches per epoch under the documented sampling policy."""
    n_pos_per_batch = max(batch_size // 4, 1)
    n_neg_per_batch = max(batch_size - n_pos_per_batch, 1)
    if epoch_policy == "full_neg_pass":
        return max(math.ceil(n_neg / n_neg_per_batch), 1)
    if epoch_policy == "pos_pass":
        return max(n_pos // n_pos_per_batch, 1)
    raise ValueError(f"Unsupported epoch_policy={epoch_policy!r}. Use 'full_neg_pass' or 'pos_pass'.")


def _evaluate_split(
    eval_df,
    rsna_raw: Path,
    encoder,
    head,
    criterion,
    tau_t: float,
    device,
    eval_batch_size: int = 16,
    num_workers: int = 8,
) -> Tuple[dict, dict, float]:
    """Run batched inference on a split, returning case-level metrics, loss, and thresholds."""
    import numpy as np
    import torch
    from scipy import stats

    head.eval()
    case_scores: Dict[str, float] = {}
    case_labels: Dict[str, int] = {}
    total_loss = 0.0
    n_batches = 0

    rel_paths = eval_df["rel_path"].tolist()
    patient_ids = eval_df["patient_id"].astype(str).tolist()
    labels = eval_df["cancer"].astype(int).tolist()

    with torch.no_grad():
        for start in range(0, len(rel_paths), eval_batch_size):
            end = min(start + eval_batch_size, len(rel_paths))
            bx = _decode_paths_parallel(rsna_raw, rel_paths[start:end], num_workers=num_workers).to(device)
            by = torch.tensor(labels[start:end], dtype=torch.float32).unsqueeze(1).to(device)

            with torch.amp.autocast("cuda"):
                logits = head(encoder(bx))
                loss, _ = criterion(logits, by, tau_t=tau_t)
                probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()

            total_loss += float(loss.item())
            n_batches += 1

            for pid, lbl, p in zip(patient_ids[start:end], labels[start:end], probs):
                pf = float(p)
                if pid not in case_scores or pf > case_scores[pid]:
                    case_scores[pid] = pf
                if pid not in case_labels or lbl > case_labels[pid]:
                    case_labels[pid] = lbl

    c_scores = np.array([case_scores[k] for k in sorted(case_scores.keys())])
    c_labels = np.array([case_labels[k] for k in sorted(case_labels.keys())])
    pos_scores = c_scores[c_labels == 1]

    tau_95 = float(np.percentile(pos_scores, 5.0)) if len(pos_scores) > 0 else 0.5
    tau_98 = float(np.percentile(pos_scores, 2.0)) if len(pos_scores) > 0 else 0.5

    def _metrics(tau: float) -> dict:
        d_mask = c_scores < tau
        n_d = int(d_mask.sum())
        missed = int(((c_labels == 1) & d_mask).sum())
        tot_cancer = int((c_labels == 1).sum())
        ret_cancer = tot_cancer - missed
        recall = ret_cancer / max(tot_cancer, 1)
        d_rate = n_d / max(len(c_scores), 1)
        ub = float(stats.beta.ppf(0.99, missed + 1, n_d - missed)) if n_d > missed else 1.0
        return {
            "tau": round(tau, 5),
            "dismissal_rate_pct": round(d_rate * 100, 2),
            "recall_pct": round(recall * 100, 2),
            "missed_cancers": missed,
            "total_cancers": tot_cancer,
            "n_dismissed": n_d,
            "cp_99_ub_pct": round(ub * 100, 2),
        }

    eval_results = {
        "target_95": _metrics(tau_95),
        "target_98": _metrics(tau_98),
        "val_loss": round(total_loss / max(n_batches, 1), 4),
    }
    return eval_results, case_scores, eval_results["val_loss"]


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=86400,
)
def train_model(
    baseline: str = "trust",
    epochs: int = 20,
    batch_size: int = 80,
    micro_batch_size: int = 8,
    lr: float = 3e-5,
    margin: float = 0.05,
    lambda_focal: float = 1.0,
    lambda_dismiss: float = 1.0,
    tau_target_recall: float = 0.98,
    epoch_policy: str = "full_neg_pass",
    grad_clip: float = 5.0,
    seed: int = 42,
    num_workers: int = 8,
    resume: bool = False,
) -> dict:
    """Execute training for TRUST or CE baseline on RSNA with per-epoch val_calib & resume."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[train] Starting {baseline.upper()} on {device} ({gpu_name}) | policy={epoch_policy}", flush=True)

    stop_event = threading.Event()
    _register_signal_handlers(RSNA_VOL, stop_event)
    commit_thread = _start_periodic_commit(RSNA_VOL, stop_event)

    try:
        rsna_raw = MOUNT_PATH / "rsna" / "raw"
        ckpt_dir = MOUNT_PATH / "models" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
        splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"

        df = _load_rsna_splits(rsna_raw, splits_csv, seed)
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        train_calib_df = df[df["split"] == "train_calib"].reset_index(drop=True)
        val_calib_df = df[df["split"] == "val_calib"].reset_index(drop=True)
        print(
            f"[data] train: {len(train_df)} | train_calib: {len(train_calib_df)} | val_calib: {len(val_calib_df)}",
            flush=True,
        )

        encoder, head, optimizer, scaler, criterion, diag = _build_model_and_criterion(
            mammo_clip_tar, device, baseline, lr, margin, lambda_focal, lambda_dismiss
        )
        print(
            f"[model] Strict Load Verified: {diag['matched_keys_count']}/{diag['total_target_keys']} "
            f"(missing={diag['missing_keys_count']}, unexpected={diag['unexpected_keys_count']})",
            flush=True,
        )

        pos_idx = train_df[train_df["cancer"] == 1].index.values.copy()
        neg_idx = train_df[train_df["cancer"] == 0].index.values.copy()
        rng = np.random.RandomState(seed)

        accum_steps = max(batch_size // micro_batch_size, 1)
        n_pos_per_micro = max(micro_batch_size // 4, 1)
        n_neg_per_micro = micro_batch_size - n_pos_per_micro
        batches_per_epoch = _compute_batches_per_epoch(len(pos_idx), len(neg_idx), batch_size, epoch_policy)

        latest_ckpt = ckpt_dir / f"{baseline}_rsna_seed{seed}_latest.pt"
        best_ckpt = ckpt_dir / f"{baseline}_rsna_seed{seed}_best.pt"
        final_ckpt = ckpt_dir / f"{baseline}_rsna_seed{seed}.pt"

        start_epoch = 1
        tau_t = 0.5
        best_val_dismiss_98 = -1.0
        history = []

        if resume and latest_ckpt.exists():
            ckpt = torch.load(str(latest_ckpt), map_location=device, weights_only=False)
            head.load_state_dict(ckpt["head_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            tau_t = float(ckpt.get("tau_t", 0.5))
            best_val_dismiss_98 = float(ckpt.get("best_val_dismiss_98", -1.0))
            history = list(ckpt.get("history", []))
            start_epoch = int(ckpt["epoch"]) + 1
            if "np_rng_state" in ckpt:
                rng.set_state(ckpt["np_rng_state"])
            if "torch_rng_state" in ckpt:
                torch.set_rng_state(ckpt["torch_rng_state"].cpu())
            print(f"[resume] Resumed from {latest_ckpt} at epoch {start_epoch}/{epochs} (tau_t={tau_t:.4f})", flush=True)

        print(
            f"[train] Epochs {start_epoch}..{epochs} | {batches_per_epoch} batches/epoch | accum={accum_steps} | workers={num_workers}",
            flush=True,
        )

        pos_calib_df = train_calib_df[train_calib_df["cancer"] == 1].reset_index(drop=True)

        for epoch in range(start_epoch, epochs + 1):
            head.train()
            epoch_loss = 0.0
            t_epoch_start = time.time()

            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            pos_p = 0
            neg_p = 0

            for _b in range(batches_per_epoch):
                optimizer.zero_grad()
                batch_loss = 0.0

                for _m in range(accum_steps):
                    mb_indices = []
                    for _ in range(n_pos_per_micro):
                        if pos_p >= len(pos_idx):
                            rng.shuffle(pos_idx)
                            pos_p = 0
                        mb_indices.append(int(pos_idx[pos_p]))
                        pos_p += 1
                    for _ in range(n_neg_per_micro):
                        if neg_p >= len(neg_idx):
                            rng.shuffle(neg_idx)
                            neg_p = 0
                        mb_indices.append(int(neg_idx[neg_p]))
                        neg_p += 1

                    sub = train_df.iloc[mb_indices]
                    bx = _decode_paths_parallel(rsna_raw, sub["rel_path"].tolist(), num_workers=num_workers).to(device)
                    by = torch.tensor(sub["cancer"].values, dtype=torch.float32).unsqueeze(1).to(device)

                    with torch.amp.autocast("cuda"):
                        with torch.no_grad():
                            feats = encoder(bx)
                        logits = head(feats)
                        total_loss, _ = criterion(logits, by, tau_t=tau_t)
                        loss = total_loss / accum_steps

                    scaler.scale(loss).backward()
                    batch_loss += float(loss.item()) * accum_steps

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += batch_loss

            epoch_loss /= max(batches_per_epoch, 1)

            # Update dynamic tau_t on train_calib positive images (TRUST only)
            head.eval()
            if baseline == "trust" and len(pos_calib_df) > 0:
                calib_pos_scores = []
                calib_paths = pos_calib_df["rel_path"].tolist()
                with torch.no_grad():
                    for start in range(0, len(calib_paths), 16):
                        bx = _decode_paths_parallel(
                            rsna_raw, calib_paths[start : start + 16], num_workers=num_workers
                        ).to(device)
                        with torch.amp.autocast("cuda"):
                            probs = torch.sigmoid(head(encoder(bx))).squeeze(1).detach().cpu().numpy()
                        calib_pos_scores.extend(probs.tolist())
                percentile = (1.0 - tau_target_recall) * 100.0
                tau_t = float(np.percentile(calib_pos_scores, percentile))

            # Per-epoch validation monitoring on val_calib
            val_results, _, val_loss = _evaluate_split(
                val_calib_df, rsna_raw, encoder, head, criterion, tau_t, device,
                eval_batch_size=16, num_workers=num_workers,
            )
            epoch_time = time.time() - t_epoch_start
            d98 = val_results["target_98"]["dismissal_rate_pct"]
            d95 = val_results["target_95"]["dismissal_rate_pct"]
            ub98 = val_results["target_98"]["cp_99_ub_pct"]

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | "
                f"tau_t={tau_t:.4f} | val_D@98={d98:.2f}% (CP99={ub98:.2f}%) | val_D@95={d95:.2f}% | {epoch_time:.1f}s",
                flush=True,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(epoch_loss, 4),
                    "val_loss": val_loss,
                    "tau_t": round(tau_t, 5),
                    "val_target_98": val_results["target_98"],
                    "val_target_95": val_results["target_95"],
                    "time_sec": round(epoch_time, 1),
                }
            )

            ckpt_payload = {
                "baseline": baseline,
                "epoch": epoch,
                "epochs": epochs,
                "epoch_policy": epoch_policy,
                "gpu_name": gpu_name,
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "tau_t": tau_t,
                "best_val_dismiss_98": best_val_dismiss_98,
                "history": history,
                "np_rng_state": rng.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "seed": seed,
            }
            if d98 > best_val_dismiss_98:
                best_val_dismiss_98 = d98
                ckpt_payload["best_val_dismiss_98"] = best_val_dismiss_98
                torch.save(ckpt_payload, str(best_ckpt))

            torch.save(ckpt_payload, str(latest_ckpt))
            RSNA_VOL.commit()

        # Save final checkpoint and operating thresholds
        torch.save(
            {
                "baseline": baseline,
                "epoch": epochs,
                "epoch_policy": epoch_policy,
                "gpu_name": gpu_name,
                "head_state_dict": head.state_dict(),
                "history": history,
                "tau_final": tau_t,
                "seed": seed,
            },
            str(final_ckpt),
        )
        final_eval, _, _ = _evaluate_split(
            val_calib_df, rsna_raw, encoder, head, criterion, tau_t, device,
            eval_batch_size=16, num_workers=num_workers,
        )
        final_eval["checkpoint_path"] = str(final_ckpt)
        final_eval["best_checkpoint_path"] = str(best_ckpt)
        final_eval["gpu_name"] = gpu_name
        final_eval["epoch_policy"] = epoch_policy

        res_path = MOUNT_PATH / "models" / f"rsna_{baseline}_eval_results.json"
        res_path.write_text(json.dumps(final_eval, indent=2))
        RSNA_VOL.commit()

        return {
            "status": "SUCCESS",
            "gpu_name": gpu_name,
            "baseline": baseline,
            "epochs": epochs,
            "epoch_policy": epoch_policy,
            "batches_per_epoch": batches_per_epoch,
            "final_loss": history[-1]["train_loss"] if history else None,
            "tau_final": round(tau_t, 4),
            "checkpoint_path": str(final_ckpt),
            "best_checkpoint_path": str(best_ckpt),
            "operating_thresholds": final_eval,
        }
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)
        try:
            RSNA_VOL.commit()
        except Exception:
            pass


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    gpu="L40S",
    cpu=4.0,
    memory=16384,
    timeout=3600,
)
def benchmark_throughput(
    benchmark_batches: int = 3,
    eval_samples: int = 40,
    batch_size: int = 80,
    micro_batch_size: int = 8,
    epoch_policy: str = "full_neg_pass",
    num_workers: int = 4,
    hourly_rate: float = 1.95,
    seed: int = 42,
) -> dict:
    """Measure real training + eval throughput on Modal GPU and project 20-epoch runtime & cost."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"

    print(f"[bench] Starting L40S benchmark on {gpu_name} ({total_vram_gb:.2f} GB VRAM)...", flush=True)
    df = _load_rsna_splits(rsna_raw, splits_csv, seed)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    train_calib_df = df[df["split"] == "train_calib"].reset_index(drop=True)
    val_calib_df = df[df["split"] == "val_calib"].reset_index(drop=True)
    print(f"[bench] Splits loaded (train={len(train_df)}, val_calib={len(val_calib_df)}).", flush=True)

    encoder, head, optimizer, scaler, criterion, _ = _build_model_and_criterion(
        mammo_clip_tar, device, "trust", 3e-5, 0.05, 1.0, 1.0
    )
    print("[bench] Mammo-CLIP B5 (852/852) loaded. Running warmup + timed batches...", flush=True)

    pos_idx = train_df[train_df["cancer"] == 1].index.values.copy()
    neg_idx = train_df[train_df["cancer"] == 0].index.values.copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    accum_steps = max(batch_size // micro_batch_size, 1)
    n_pos_per_micro = max(micro_batch_size // 4, 1)
    n_neg_per_micro = micro_batch_size - n_pos_per_micro
    batches_per_epoch = _compute_batches_per_epoch(len(pos_idx), len(neg_idx), batch_size, epoch_policy)

    # Warmup 1 batch + time benchmark_batches
    torch.cuda.reset_peak_memory_stats(device)
    head.train()
    pos_p, neg_p = 0, 0
    batch_times = []

    for b in range(benchmark_batches + 1):
        t0 = time.time()
        optimizer.zero_grad()
        for _ in range(accum_steps):
            mb_indices = []
            for _ in range(n_pos_per_micro):
                mb_indices.append(int(pos_idx[pos_p % len(pos_idx)]))
                pos_p += 1
            for _ in range(n_neg_per_micro):
                mb_indices.append(int(neg_idx[neg_p % len(neg_idx)]))
                neg_p += 1
            sub = train_df.iloc[mb_indices]
            bx = _decode_paths_parallel(rsna_raw, sub["rel_path"].tolist(), num_workers=num_workers).to(device)
            by = torch.tensor(sub["cancer"].values, dtype=torch.float32).unsqueeze(1).to(device)
            with torch.amp.autocast("cuda"):
                with torch.no_grad():
                    feats = encoder(bx)
                logits = head(feats)
                loss, _ = criterion(logits, by, tau_t=0.5)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        dt = time.time() - t0
        if b == 0:
            print(f"[bench] Warmup batch (80 imgs): {dt:.2f}s", flush=True)
        else:
            batch_times.append(dt)
            print(f"[bench] Timed batch {b}/{benchmark_batches} (80 imgs): {dt:.2f}s", flush=True)

    # Benchmark eval throughput
    eval_sub = val_calib_df.head(eval_samples)
    print(f"[bench] Timing eval on {len(eval_sub)} val_calib images...", flush=True)
    t_eval0 = time.time()
    _evaluate_split(
        eval_sub, rsna_raw, encoder, head, criterion, 0.5, device, eval_batch_size=16, num_workers=num_workers
    )
    torch.cuda.synchronize()
    eval_dt = time.time() - t_eval0

    sec_per_batch = float(np.mean(batch_times))
    train_imgs_per_sec = batch_size / max(sec_per_batch, 1e-6)
    eval_imgs_per_sec = len(eval_sub) / max(eval_dt, 1e-6)
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    n_calib_pos = int((train_calib_df["cancer"] == 1).sum())
    n_val_imgs = len(val_calib_df)
    proj_train_sec_per_epoch = batches_per_epoch * sec_per_batch
    proj_eval_sec_per_epoch = (n_calib_pos + n_val_imgs) / max(eval_imgs_per_sec, 1e-6)
    proj_epoch_sec = proj_train_sec_per_epoch + proj_eval_sec_per_epoch
    proj_20_epochs_hr = (20 * proj_epoch_sec) / 3600.0
    proj_20_epochs_cost = proj_20_epochs_hr * hourly_rate

    report = {
        "gpu_name": gpu_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "epoch_policy": epoch_policy,
        "batches_per_epoch": batches_per_epoch,
        "benchmark_batches": benchmark_batches,
        "sec_per_batch_80": round(sec_per_batch, 3),
        "train_imgs_per_sec": round(train_imgs_per_sec, 2),
        "eval_imgs_per_sec": round(eval_imgs_per_sec, 2),
        "projected_epoch_min": round(proj_epoch_sec / 60.0, 2),
        "projected_20_epochs_hr": round(proj_20_epochs_hr, 2),
        "assumed_hourly_rate_usd": hourly_rate,
        "projected_20_epochs_cost_usd_per_run": round(proj_20_epochs_cost, 2),
        "projected_ce_plus_trust_cost_usd": round(proj_20_epochs_cost * 2, 2),
    }

    out_path = MOUNT_PATH / "models" / "benchmark_throughput.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    RSNA_VOL.commit()

    print("\n" + "=" * 60, flush=True)
    print("L40S THROUGHPUT BENCHMARK & 20-EPOCH PROJECTION:", flush=True)
    for k, v in report.items():
        print(f"  {k}: {v}", flush=True)
    print("=" * 60, flush=True)
    return report


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    gpu="L40S",
    cpu=4.0,
    memory=16384,
    timeout=1800,
)
def sweep_micro_batch(
    candidates: Tuple[int, ...] = (80, 96, 112, 128, 144, 160, 192),
    fallback_candidates: Tuple[int, ...] = (60, 40, 20, 16, 8),
    timed_steps: int = 1,
    num_workers: int = 4,
    seed: int = 42,
) -> dict:
    """Find the HIGHEST stable micro-batch size on L40S, starting at 80 and stepping up until CUDA OOM.

    If 80 OOMs, automatically falls back downward through (60, 40, 20, 16, 8).
    Pre-decodes 80 real RSNA DICOMs (20 pos, 60 neg) once into CPU RAM to minimize GPU credits.
    """
    import subprocess
    import numpy as np
    import pandas as pd
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    print(f"[sweep] Starting L40S micro-batch sweep on {gpu_name} ({total_vram_gb:.2f} GB VRAM)...", flush=True)
    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"

    df = _load_rsna_splits(rsna_raw, splits_csv, seed)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    pos_sub = train_df[train_df["cancer"] == 1].head(20)
    neg_sub = train_df[train_df["cancer"] == 0].head(60)
    sample_df = pd.concat([pos_sub, neg_sub], ignore_index=True)

    print("[sweep] Decoding 80 RSNA DICOMs (20 pos + 60 neg) once into CPU RAM...", flush=True)
    cpu_x = _decode_paths_parallel(rsna_raw, sample_df["rel_path"].tolist(), num_workers=num_workers)
    cpu_y = torch.tensor(sample_df["cancer"].values, dtype=torch.float32).unsqueeze(1)
    # Interleave 1:3 pos:neg order so every slice of size mb has exact 25% positives
    pos_tensors = cpu_x[:20]
    neg_tensors = cpu_x[20:]
    pos_labels = cpu_y[:20]
    neg_labels = cpu_y[20:]

    encoder, head, optimizer, scaler, criterion, _ = _build_model_and_criterion(
        mammo_clip_tar, device, "trust", 3e-5, 0.05, 1.0, 1.0
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

    def _test_mb(mb: int) -> Optional[dict]:
        print(f"\n[sweep] Testing batch_size={mb}...", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Build a single batch of size mb (preserving 1:3 pos:neg ratio) from cached 80 tensors
        n_pos_mb = max(mb // 4, 1)
        n_neg_mb = mb - n_pos_mb
        p_idx = torch.arange(n_pos_mb) % len(pos_tensors)
        n_idx = torch.arange(n_neg_mb) % len(neg_tensors)
        bx_cpu = torch.cat([pos_tensors[p_idx], neg_tensors[n_idx]], dim=0)
        by_cpu = torch.cat([pos_labels[p_idx], neg_labels[n_idx]], dim=0)

        try:
            head.train()
            step_times = []
            util_samples = []

            for step_idx in range(timed_steps + 1):  # 1 warmup + timed_steps
                torch.cuda.synchronize()
                t0 = time.time()
                optimizer.zero_grad()

                bx = bx_cpu.to(device, non_blocking=True)
                by = by_cpu.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    with torch.no_grad():
                        feats = encoder(bx)
                    logits = head(feats)
                    loss, _ = criterion(logits, by, tau_t=0.5)
                scaler.scale(loss).backward()

                if step_idx > 0:
                    u = _read_gpu_util()
                    if u is not None:
                        util_samples.append(u)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                dt = time.time() - t0
                if step_idx > 0:
                    step_times.append(dt)

            peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            mean_step_s = float(np.mean(step_times))
            imgs_per_s = mb / max(mean_step_s, 1e-6)
            mean_util = float(np.mean(util_samples)) if util_samples else None

            entry = {
                "batch_size": mb,
                "status": "OK",
                "peak_vram_gb": round(peak_vram_gb, 2),
                "vram_pct": round((peak_vram_gb / max(total_vram_gb, 1e-6)) * 100, 1),
                "gpu_util_pct": round(mean_util, 1) if mean_util is not None else None,
                "step_time_sec": round(mean_step_s, 3),
                "imgs_per_sec": round(imgs_per_s, 2),
            }
            print(
                f"[sweep] OK | batch={mb} | peak_vram={entry['peak_vram_gb']} GB ({entry['vram_pct']}%) | "
                f"gpu_util={entry['gpu_util_pct']}% | step={entry['step_time_sec']}s | {entry['imgs_per_sec']} img/s",
                flush=True,
            )
            return entry

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(f"[sweep] CUDA OOM at batch_size={mb}.", flush=True)
                return {
                    "batch_size": mb,
                    "status": "CUDA_OOM",
                    "error": str(exc).splitlines()[0],
                }
            raise

    results = []
    highest_stable_mb = None
    fastest_cfg = None

    for mb in candidates:
        entry = _test_mb(mb)
        results.append(entry)
        if entry["status"] == "CUDA_OOM":
            break
        highest_stable_mb = mb
        if fastest_cfg is None or entry["imgs_per_sec"] > fastest_cfg["imgs_per_sec"]:
            fastest_cfg = entry

    if highest_stable_mb is None:
        print("[sweep] 80 OOM'd — stepping downward to find highest stable size...", flush=True)
        for mb in fallback_candidates:
            entry = _test_mb(mb)
            results.append(entry)
            if entry["status"] == "OK":
                highest_stable_mb = mb
                fastest_cfg = entry
                break

    summary = {
        "gpu_name": gpu_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "highest_stable_micro_batch": highest_stable_mb,
        "fastest_stable_config": fastest_cfg,
        "results": results,
    }

    out_path = MOUNT_PATH / "models" / "l40s_micro_batch_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    RSNA_VOL.commit()

    print("\n" + "=" * 65, flush=True)
    print("L40S MICRO-BATCH SWEEP SUMMARY:", flush=True)
    print(f"  Highest stable micro_batch_size: {highest_stable_mb}", flush=True)
    print(f"  Fastest stable config: {fastest_cfg}", flush=True)
    for r in results:
        print(f"  {r}", flush=True)
    print("=" * 65, flush=True)
    return summary


@app.local_entrypoint()
def main(
    baseline: str = "trust",
    epochs: int = 20,
    batch_size: int = 80,
    micro_batch_size: int = 8,
    lr: float = 3e-5,
    margin: float = 0.05,
    lambda_focal: float = 1.0,
    lambda_dismiss: float = 1.0,
    tau_target_recall: float = 0.98,
    epoch_policy: str = "full_neg_pass",
    num_workers: int = 4,
    gpu: str = "L40S",
    resume: bool = False,
    benchmark: bool = False,
    benchmark_batches: int = 3,
    sweep_mb: bool = False,
    hourly_rate: float = -1.0,
) -> None:
    gpu_norm = gpu.upper()
    if gpu_norm not in SUPPORTED_GPUS:
        raise ValueError(f"Unsupported GPU {gpu!r}. Choose from {SUPPORTED_GPUS}.")

    rate = DEFAULT_HOURLY_RATES[gpu_norm] if hourly_rate < 0 else hourly_rate

    if sweep_mb:
        print(f"Launching micro-batch sweep on Modal (GPU={gpu_norm}, up to 80)...")
        res = sweep_micro_batch.with_options(gpu=gpu_norm).remote(num_workers=num_workers)
        print("Micro-batch sweep finished:", res)
        return

    if benchmark:
        print(f"Launching throughput benchmark on Modal (GPU={gpu_norm}, batches={benchmark_batches})...")
        res = benchmark_throughput.with_options(gpu=gpu_norm).remote(
            benchmark_batches=benchmark_batches,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            epoch_policy=epoch_policy,
            num_workers=num_workers,
            hourly_rate=rate,
        )
        print("Benchmark finished:", res)
        return

    print(
        f"Launching {baseline.upper()} training on Modal (GPU={gpu_norm}, epochs={epochs}, "
        f"policy={epoch_policy}, resume={resume})..."
    )
    res = train_model.with_options(gpu=gpu_norm).remote(
        baseline=baseline,
        epochs=epochs,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        lr=lr,
        margin=margin,
        lambda_focal=lambda_focal,
        lambda_dismiss=lambda_dismiss,
        tau_target_recall=tau_target_recall,
        epoch_policy=epoch_policy,
        num_workers=num_workers,
        resume=resume,
    )
    print("Training finished with result:", res)
