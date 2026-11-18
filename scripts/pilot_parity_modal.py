"""Controlled 1,000-Image Online-vs-Cached Equivalence Pilot on Modal L40S.

Executes Approach A (Online) vs Approach B (Augmentation-Aware Cached) on the
1,000-image stratified RSNA pilot (700 train, 100 train_calib, 200 val_calib)
with the real frozen Mammo-CLIP EfficientNet-B5 encoder (852/852 strict parity).

Validates both:
    1. Deterministic FP32 reference parity (tolerance <= 1e-5).
    2. Production AMP mixed-precision parity (tolerance <= 1e-3).

Usage (DO NOT EXECUTE AUTOMATICALLY - for user execution):
    modal run scripts/pilot_parity_modal.py --epochs 20
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import modal

APP_NAME = "anchorshift-parity-pilot"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
MOUNT_PATH = Path("/data")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "omegaconf>=2.3.0",
        "pandas>=2.2.0",
        "pydicom>=2.4.4",
        "opencv-python-headless>=4.10.0",
        "pylibjpeg>=2.0.0",
        "pylibjpeg-libjpeg>=2.1.0",
        "pylibjpeg-openjpeg>=2.1.0",
        "python-gdcm>=3.0.24",
        "scipy>=1.12.0",
        "numpy>=1.26.0",
    )
    .add_local_python_source("src")
)


def _sample_pilot_1000(df, seed: int = 42):
    import pandas as pd

    targets = {"train": (29, 671), "train_calib": (4, 96), "val_calib": (8, 192)}
    parts = []
    for split_name, (n_pos, n_neg) in targets.items():
        sub = df[df["split"] == split_name]
        parts.append(sub[sub["cancer"] == 1].sample(n=n_pos, random_state=seed))
        parts.append(sub[sub["cancer"] == 0].sample(n=n_neg, random_state=seed))
    return pd.concat(parts, ignore_index=True)


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=14400,
)
def run_pilot(
    epochs: int = 20,
    steps_per_epoch: int = 12,
    micro_batch_size: int = 80,
    precision: str = "both",
    num_workers: int = 4,
    seed: int = 42,
) -> dict:
    """Run controlled online-vs-cached equivalence pilot on Modal L40S."""
    import pandas as pd
    import torch

    from src.data.preprocessing import decode_dicom_array, extract_cropped_mammogram
    from src.models.mammo_clip_loader import MAMMO_CLIP_SHA256, load_and_verify_mammo_clip
    from src.models.trust_model import MammoCLIPEncoder
    from src.training.parity_pilot import run_online_vs_cached_parity_pilot

    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    pilot_cache_root = MOUNT_PATH / "cache" / "parity_pilot_1000"

    df_full = pd.read_csv(splits_csv)
    pilot_df = _sample_pilot_1000(df_full, seed=seed)

    train_df = pilot_df[pilot_df["split"] == "train"].reset_index(drop=True)
    tc_df = pilot_df[pilot_df["split"] == "train_calib"].reset_index(drop=True)
    vc_df = pilot_df[pilot_df["split"] == "val_calib"].reset_index(drop=True)

    print(
        f"[pilot] Stage-1 decoding & breast-cropping 1,000 DICOMs once "
        f"(train={len(train_df)}, train_calib={len(tc_df)}, val_calib={len(vc_df)})...",
        flush=True,
    )

    def _stage1_one(row_tuple) -> object:
        rel_path, lat = row_tuple
        arr, phot, dcm_lat = decode_dicom_array(rsna_raw / rel_path, apply_luts=True)
        eff_lat = lat if lat in ("L", "R") else dcm_lat
        return extract_cropped_mammogram(arr, photometric=phot, laterality=eff_lat)

    def _stage1_split(sdf):
        lat_col = sdf["laterality"].astype(str).tolist() if "laterality" in sdf.columns else [""] * len(sdf)
        items = list(zip(sdf["rel_path"].tolist(), lat_col))
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            return list(pool.map(_stage1_one, items))

    train_cropped = _stage1_split(train_df)
    tc_cropped = _stage1_split(tc_df)
    vc_cropped = _stage1_split(vc_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MammoCLIPEncoder(checkpoint_path=None, freeze=False, model_name="efficientnet-b5")
    diag = load_and_verify_mammo_clip(encoder, mammo_clip_tar, strict=True, verify_sha256=True)
    assert diag["matched_keys_count"] == 852 and diag["missing_keys_count"] == 0 and diag["unexpected_keys_count"] == 0
    encoder.freeze_encoder()

    modes = ["fp32", "amp"] if precision == "both" else [precision]
    results = {}
    for pmode in modes:
        use_amp = pmode == "amp"
        print(f"\n[pilot] Running {pmode.upper()} online-vs-cached parity ({epochs} epochs, {steps_per_epoch} steps/ep)...", flush=True)
        res = run_online_vs_cached_parity_pilot(
            encoder=encoder,
            train_cropped=train_cropped,
            train_calib_cropped=tc_cropped,
            val_calib_cropped=vc_cropped,
            train_df=train_df,
            train_calib_df=tc_df,
            val_calib_df=vc_df,
            cache_root=pilot_cache_root / pmode,
            embed_dim=2048,
            target_size=(1520, 912),
            epochs=epochs,
            batch_size=80,
            n_pos=20,
            steps_per_epoch=steps_per_epoch,
            epoch_policy="paper_independent",
            seed=seed,
            use_amp=use_amp,
            device=device,
            model_checkpoint_sha256=MAMMO_CLIP_SHA256,
            micro_batch_size=micro_batch_size,
            on_progress=lambda msg: print(f"  [{pmode}] {msg}", flush=True),
        )
        results[pmode] = res
        print(f"[pilot] {pmode.upper()} status={res['status']} max_prob_diff={res['max_abs_prob_diff']:.2e}", flush=True)

    out_path = MOUNT_PATH / "models" / "l40s_parity_pilot_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    RSNA_VOL.commit()
    return results


@app.local_entrypoint()
def main(
    epochs: int = 20,
    steps_per_epoch: int = 12,
    micro_batch_size: int = 80,
    precision: str = "both",
    gpu: str = "L40S",
) -> None:
    print(f"Launching 1,000-image Online vs Cached Parity Pilot on Modal ({gpu}, epochs={epochs}, precision={precision})...")
    res = run_pilot.with_options(gpu=gpu.upper()).remote(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        micro_batch_size=micro_batch_size,
        precision=precision,
    )
    print(json.dumps(res, indent=2))
