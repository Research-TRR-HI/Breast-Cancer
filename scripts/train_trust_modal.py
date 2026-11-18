"""Full TRUST & CE-Only Training Pipeline on Modal.

Reference:
    TRUST (2026): Section 3.1 - 3.11
    - Backbone: Mammo-CLIP EfficientNet-B5 (frozen).
    - Head: LayerNorm(2048) -> Dropout(0.3) -> Linear(2048, 1).
    - Resolution: 1520 x 912.
    - Minibatch: 80 (20 positives, 60 negatives) via gradient accumulation.
    - Optimizer: AdamW, lr=3e-5, weight_decay=1e-4.
    - Loss: BCE + Focal Loss (gamma=2.0, alpha=0.25) + Dynamic Dismissal Loss.
    - Dynamic tau_t updated on train_calib set at end of each epoch.
    - Post-training threshold selection on val_calib for 95% and 98% recall targets.
    - All predictions and checkpoints saved to rsna-mammography-vol.

Usage (DO NOT EXECUTE AUTOMATICALLY - for user execution):
    modal run scripts/train_trust_modal.py --baseline trust --epochs 20 --gpu L4
    modal run scripts/train_trust_modal.py --baseline ce --epochs 20 --gpu L4
"""

import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple
import modal

APP_NAME = "anchorshift-stage-b-train"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
MOUNT_PATH = Path("/data")
COMMIT_INTERVAL_SEC = 60

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
    grad_clip: float = 5.0,
    seed: int = 42,
) -> dict:
    """Execute training for TRUST or CE baseline on RSNA."""
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy import stats

    from src.data.preprocessing import preprocess_dicom
    from src.models.trust_model import MammoCLIPEncoder
    from src.models.mammo_clip_loader import load_and_verify_mammo_clip
    from src.models.losses import TRUSTLoss

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Starting {baseline.upper()} training on {device} ({torch.cuda.get_device_name(0)})", flush=True)

    stop_event = threading.Event()
    _register_signal_handlers(RSNA_VOL, stop_event)
    commit_thread = _start_periodic_commit(RSNA_VOL, stop_event)

    try:
        rsna_raw = MOUNT_PATH / "rsna" / "raw"
        ckpt_dir = MOUNT_PATH / "models" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"

        # Load RSNA splits
        splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
        if not splits_csv.exists():
            # If not yet copied to volume, load train.csv and generate deterministic splits
            df_full = pd.read_csv(rsna_raw / "train.csv")
            patient_cancer = df_full.groupby("patient_id")["cancer"].max().reset_index()
            rng = np.random.RandomState(seed)
            pos_pts = patient_cancer[patient_cancer["cancer"] == 1]["patient_id"].values
            neg_pts = patient_cancer[patient_cancer["cancer"] == 0]["patient_id"].values
            rng.shuffle(pos_pts)
            rng.shuffle(neg_pts)

            def _split(arr, f1, f2):
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
            df_full["rel_path"] = "train_images/" + df_full["patient_id"].astype(str) + "/" + df_full["image_id"].astype(str) + ".dcm"
            df_full.to_csv(splits_csv, index=False)
            RSNA_VOL.commit()
            df = df_full
        else:
            df = pd.read_csv(splits_csv)

        train_df = df[df["split"] == "train"].reset_index(drop=True)
        train_calib_df = df[df["split"] == "train_calib"].reset_index(drop=True)
        val_calib_df = df[df["split"] == "val_calib"].reset_index(drop=True)

        print(f"[data] train: {len(train_df)} imgs | train_calib: {len(train_calib_df)} imgs | val_calib: {len(val_calib_df)} imgs", flush=True)

        # Build Mammo-CLIP EfficientNet-B5 model (strict=True verification)
        if not mammo_clip_tar.exists():
            raise FileNotFoundError(f"Mammo-CLIP checkpoint missing at {mammo_clip_tar}")
        print(f"[model] Loading Mammo-CLIP pretrained weights from {mammo_clip_tar}", flush=True)
        encoder = MammoCLIPEncoder(checkpoint_path=None, freeze=False, model_name="efficientnet-b5")
        diag = load_and_verify_mammo_clip(encoder, mammo_clip_tar, strict=True)
        encoder.freeze_encoder()
        print(
            f"[model] Strict Load Verified: {diag['matched_keys_count']}/{diag['total_target_keys']} keys "
            f"(missing={diag['missing_keys_count']}, unexpected={diag['unexpected_keys_count']})",
            flush=True,
        )

        embed_dim = encoder.embed_dim
        head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 1),
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

        def _read_tensor(rel_path: str) -> torch.Tensor:
            return preprocess_dicom(rsna_raw / rel_path, target_size=(1520, 912), invert_monochrome1=True, standardize=True)

        # Positive & negative index pools for balanced sampling (20 pos, 60 neg per 80)
        pos_idx = train_df[train_df["cancer"] == 1].index.values
        neg_idx = train_df[train_df["cancer"] == 0].index.values
        rng = np.random.RandomState(seed)

        accum_steps = max(batch_size // micro_batch_size, 1)
        n_pos_per_micro = max(micro_batch_size // 4, 1)
        n_neg_per_micro = micro_batch_size - n_pos_per_micro
        batches_per_epoch = len(pos_idx) // 20  # ~ len(pos_idx) // 20 batches of 80

        tau_t = 0.5  # initial provisional threshold
        history = []

        print(f"[train] Training for {epochs} epochs ({batches_per_epoch} batches/epoch, accum={accum_steps})...", flush=True)

        for epoch in range(1, epochs + 1):
            head.train()
            optimizer.zero_grad()
            epoch_loss = 0.0
            t_epoch_start = time.time()

            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            pos_p = 0
            neg_p = 0

            for b in range(batches_per_epoch):
                optimizer.zero_grad()
                batch_loss = 0.0

                for m in range(accum_steps):
                    # Sample micro-batch
                    mb_indices = []
                    for _ in range(n_pos_per_micro):
                        mb_indices.append(pos_idx[pos_p % len(pos_idx)])
                        pos_p += 1
                    for _ in range(n_neg_per_micro):
                        mb_indices.append(neg_idx[neg_p % len(neg_idx)])
                        neg_p += 1

                    sub = train_df.iloc[mb_indices]
                    x_tensors = [_read_tensor(r["rel_path"]) for _, r in sub.iterrows()]
                    bx = torch.stack(x_tensors).to(device)
                    by = torch.tensor(sub["cancer"].values, dtype=torch.float32).unsqueeze(1).to(device)

                    with torch.amp.autocast("cuda"):
                        with torch.no_grad():
                            feats = encoder(bx)
                        logits = head(feats)
                        total_loss, _ = criterion(logits, by, tau_t=tau_t)
                        loss = total_loss / accum_steps

                    scaler.scale(loss).backward()
                    batch_loss += loss.item() * accum_steps

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += batch_loss

            epoch_loss /= max(batches_per_epoch, 1)
            epoch_time = time.time() - t_epoch_start

            # Evaluate on train_calib to update tau_t
            head.eval()
            calib_pos_scores = []
            with torch.no_grad():
                # Subsample 100 positive and 100 negative calib images for fast evaluation
                pos_calib = train_calib_df[train_calib_df["cancer"] == 1]
                for _, r in pos_calib.iterrows():
                    t = _read_tensor(r["rel_path"]).unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda"):
                        p = torch.sigmoid(head(encoder(t))).item()
                    calib_pos_scores.append(p)

            if len(calib_pos_scores) > 0 and baseline == "trust":
                tau_t = float(np.percentile(calib_pos_scores, 2.0))  # 98% target recall threshold
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | tau_t: {tau_t:.4f} | Time: {epoch_time:.1f}s", flush=True)
            history.append({"epoch": epoch, "loss": epoch_loss, "tau_t": tau_t, "time_sec": epoch_time})

        # Save model checkpoint
        save_path = ckpt_dir / f"{baseline}_rsna_seed{seed}.pt"
        torch.save(
            {
                "baseline": baseline,
                "epoch": epochs,
                "head_state_dict": head.state_dict(),
                "history": history,
                "tau_final": tau_t,
                "seed": seed,
            },
            str(save_path),
        )
        RSNA_VOL.commit()
        print(f"[save] Model saved to {save_path}", flush=True)

        # Calibrate Operating Thresholds on held-out val_calib
        print("[eval] Evaluating held-out val_calib for 95% and 98% operating points...", flush=True)
        head.eval()
        val_case_scores = {}
        val_case_labels = {}
        with torch.no_grad():
            for _, r in val_calib_df.iterrows():
                cid = str(r["patient_id"])
                label = int(r["cancer"])
                t = _read_tensor(r["rel_path"]).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda"):
                    p = torch.sigmoid(head(encoder(t))).item()
                if cid not in val_case_scores or p > val_case_scores[cid]:
                    val_case_scores[cid] = p
                if cid not in val_case_labels or label > val_case_labels[cid]:
                    val_case_labels[cid] = label

        c_scores = np.array([val_case_scores[k] for k in sorted(val_case_scores.keys())])
        c_labels = np.array([val_case_labels[k] for k in sorted(val_case_labels.keys())])
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
            d_rate = n_d / len(c_scores)
            prev_d = missed / max(n_d, 1)
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
            "checkpoint_path": str(save_path),
        }
        print(f"[eval] Target 95% Recall: {eval_results['target_95']}", flush=True)
        print(f"[eval] Target 98% Recall: {eval_results['target_98']}", flush=True)

        res_path = MOUNT_PATH / "models" / f"rsna_{baseline}_eval_results.json"
        res_path.write_text(json.dumps(eval_results, indent=2))
        RSNA_VOL.commit()

        return {
            "status": "SUCCESS",
            "baseline": baseline,
            "epochs": epochs,
            "final_loss": round(history[-1]["loss"], 4),
            "tau_final": round(tau_t, 4),
            "checkpoint_path": str(save_path),
            "operating_thresholds": eval_results,
        }
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)
        try:
            RSNA_VOL.commit()
        except Exception:
            pass


@app.local_entrypoint()
def main(
    baseline: str = "trust",
    epochs: int = 20,
    batch_size: int = 80,
    micro_batch_size: int = 8,
    lr: float = 3e-5,
    margin: float = 0.05,
    gpu: str = "L4",
) -> None:
    print(f"Launching {baseline.upper()} training on Modal (GPU: {gpu}, epochs={epochs})...")
    res = train_model.remote(
        baseline=baseline,
        epochs=epochs,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        lr=lr,
        margin=margin,
    )
    print("Training finished with result:", res)
