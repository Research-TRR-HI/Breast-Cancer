"""Stage B Tiny Modal GPU Smoke Test on RSNA.

Strict cost optimization:
- Uses lowest-cost GPU tier: gpu="T4" (~$0.59/hr).
- Minimal execution scope: loads 10 real RSNA DICOM images from rsna-mammography-vol.
- Reuses canonical modules from src/ (no duplicated logic).
- Verifies strict=True (852/852 keys, 0 missing, 0 unexpected) Mammo-CLIP B5 loading.
- Tests DICOM decoding, MONOCHROME1 inversion, Mammo-CLIP mean/std standardization, 1520x912 resize.
- Tests forward + backward pass through frozen MammoCLIPEncoder + TRUSTClassifier head.
- Tests TRUSTLoss (BCE + Focal + Dynamic Dismissal) and peak VRAM.

Usage (DO NOT EXECUTE AUTOMATICALLY - for user execution):
    modal run scripts/smoke_test_stage_b_modal.py
"""

from pathlib import Path
import sys
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "anchorshift-stage-b-smoke"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
MOUNT_PATH = Path("/data")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES)
    .add_local_python_source("src")
)


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    gpu="T4",
    cpu=2.0,
    memory=8192,
    timeout=600,
)
def run_gpu_smoke_test() -> dict:
    """Execute end-to-end GPU smoke test on a sample of 10 RSNA images."""
    import time
    import pandas as pd
    import pydicom
    import torch

    from src.data.preprocessing import preprocess_dicom
    from src.models.trust_model import MammoCLIPEncoder, TRUSTClassifier
    from src.models.mammo_clip_loader import load_and_verify_mammo_clip
    from src.models.losses import TRUSTLoss
    from src.utils.env_metadata import collect_runtime_environment_metadata

    env_meta = collect_runtime_environment_metadata(precision_policy="amp")

    print("[1/5] Checking GPU environment...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    print(f"  Device: {device} ({gpu_name}, {total_vram_gb:.2f} GB VRAM)", flush=True)

    print("[2/5] Checking RSNA raw DICOM access and Mammo-CLIP checkpoint...", flush=True)
    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    manifest_csv = rsna_raw / "train.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing RSNA train.csv at {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    pos_samples = df[df["cancer"] == 1].head(5)
    neg_samples = df[df["cancer"] == 0].head(5)
    sample_df = pd.concat([pos_samples, neg_samples]).reset_index(drop=True)

    ckpt_path = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Mammo-CLIP checkpoint missing at {ckpt_path}")
    print(f"  Mammo-CLIP checkpoint verified on volume: {ckpt_path}", flush=True)

    print("[3/5] Testing DICOM decoding, MONOCHROME1 inversion, standardization, and 1520x912 resize...", flush=True)
    t0 = time.time()
    tensors = []
    labels = []
    photometrics = []

    for _, row in sample_df.iterrows():
        p = rsna_raw / "train_images" / str(row["patient_id"]) / f"{row['image_id']}.dcm"
        ds_hdr = pydicom.dcmread(str(p), stop_before_pixels=True)
        photometrics.append(str(getattr(ds_hdr, "PhotometricInterpretation", "")).strip().upper())
        row_lat = str(row["laterality"]).strip().upper() if "laterality" in row and pd.notna(row["laterality"]) else None

        tensors.append(
            preprocess_dicom(
                p,
                target_size=(1520, 912),
                invert_monochrome1=True,
                laterality=row_lat,
                laterality_policy="strict",
                strict_segmentation=True,
                standardize=True,
            )
        )
        labels.append(int(row["cancer"]))

    batch_x = torch.stack(tensors).to(device)  # shape (10, 3, 1520, 912)
    batch_y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)
    load_time = time.time() - t0
    print(f"  Processed {len(sample_df)} DICOMs in {load_time:.2f}s ({load_time / len(sample_df):.2f}s/img)", flush=True)
    print(f"  Tensor stats: min={batch_x.min().item():.3f}, max={batch_x.max().item():.3f}, mean={batch_x.mean().item():.3f}", flush=True)
    print(f"  Photometrics observed: {dict(pd.Series(photometrics).value_counts())}", flush=True)

    print("[4/5] Building Mammo-CLIP TRUST model (strict=True weight verification)...", flush=True)
    encoder = MammoCLIPEncoder(checkpoint_path=None, freeze=False, model_name="efficientnet-b5")
    diag = load_and_verify_mammo_clip(encoder, ckpt_path, strict=True)
    encoder.freeze_encoder()
    print(
        f"  Strict Load Diagnostics: matched={diag['matched_keys_count']}/{diag['total_target_keys']}, "
        f"missing={diag['missing_keys_count']}, unexpected={diag['unexpected_keys_count']}",
        flush=True,
    )
    assert diag["is_fully_compatible"], f"Incompatible Mammo-CLIP checkpoint load: {diag}"
    assert (
        diag["matched_keys_count"] == 852
        and diag["total_target_keys"] == 852
        and diag["missing_keys_count"] == 0
        and diag["unexpected_keys_count"] == 0
    ), f"Expected strict 852/852 Mammo-CLIP key parity, got: {diag}"

    model = TRUSTClassifier(encoder=encoder, embed_dim=2048, dropout_rate=0.3, freeze_encoder=True).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-5,
        weight_decay=1e-4,
    )
    criterion = TRUSTLoss(baseline="trust", margin=0.10, lambda_focal=1.0, lambda_dismiss=0.05)

    # Forward + backward pass on micro-batch of 2
    torch.cuda.reset_peak_memory_stats(device)
    t_fwd_start = time.time()
    model.train()  # keeps frozen encoder in eval() automatically
    with torch.amp.autocast("cuda"):
        logits, _ = model(batch_x[:2])
        total_loss, loss_comp = criterion(logits, batch_y[:2], tau_t=0.05)

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
    optimizer.step()

    fwd_bwd_time = time.time() - t_fwd_start
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    # Strict gradient isolation verification
    for name, p in model.encoder.named_parameters():
        assert p.grad is None, f"Frozen encoder parameter leaked gradient: {name}"
    assert model.head.weight.grad is not None, "Linear head received no gradient!"

    print(f"  Forward + Backward step (batch=2, amp=True): {fwd_bwd_time:.3f}s", flush=True)
    print(f"  Peak VRAM: {peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB", flush=True)
    print(
        f"  Losses: BCE={loss_comp['loss_bce']:.4f}, Focal={loss_comp['loss_focal']:.4f}, "
        f"Dismissal={loss_comp['loss_dismiss']:.4f}, Total={loss_comp['loss_total']:.4f}",
        flush=True,
    )

    print("[5/5] Smoke test completed with 100% strict weight parity!", flush=True)
    return {
        "status": "PASS",
        "gpu_name": gpu_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "dicom_sample_size": len(sample_df),
        "load_time_sec": round(load_time, 2),
        "fwd_bwd_time_sec": round(fwd_bwd_time, 3),
        "matched_keys": f"{diag['matched_keys_count']}/{diag['total_target_keys']}",
        "missing_keys": diag["missing_keys_count"],
        "unexpected_keys": diag["unexpected_keys_count"],
        "bce_loss": round(loss_comp["loss_bce"], 4),
        "focal_loss": round(loss_comp["loss_focal"], 4),
        "dismissal_loss": round(loss_comp["loss_dismiss"], 4),
        "total_loss": round(loss_comp["loss_total"], 4),
        "checkpoint_exists": True,
        "checkpoint_loaded_strict": bool(diag["is_fully_compatible"]),
        "encoder_frozen_verified": True,
        "environment_metadata": env_meta,
    }


@app.local_entrypoint()
def main() -> None:
    res = run_gpu_smoke_test.remote()
    print("\n" + "=" * 50)
    print("STAGE B GPU SMOKE TEST RESULTS (STRICT):")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("=" * 50)
