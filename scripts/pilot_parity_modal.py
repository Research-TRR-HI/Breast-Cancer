"""Controlled 1,200-Image Online-vs-Cached Equivalence Pilot on Modal L40S.

Executes Approach A (Online) vs Approach B (Augmentation-Aware Cached) on the
1,200-image stratified RSNA pilot:
    - train:       500 (21 positive, 479 negative)
    - train_calib: 500 (20 positive, 480 negative -> >= 299 so CP95 <= 1% can activate tau_t > 0)
    - val_calib:   200 ( 8 positive, 192 negative)

Uses the real frozen Mammo-CLIP EfficientNet-B5 encoder (852/852 strict parity),
pinned Python 3.12 + exact dependency versions, Stage-1 uint16 crops, and the
shared production training engine.

Usage (DO NOT EXECUTE AUTOMATICALLY - for user execution):
    modal run scripts/pilot_parity_modal.py --epochs 20
"""

import json
from pathlib import Path
import sys
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "anchorshift-parity-pilot"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
MOUNT_PATH = Path("/data")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES)
    .add_local_python_source("src")
)


def _load_pilot_stage1_crops(
    pilot_df,
    train_df,
    tc_df,
    vc_df,
    raw_root: Path,
    splits_sha256: str,
    full_stage1_root: Path,
    pilot_stage1_root: Path,
    strict_segmentation: bool = True,
):
    from src.data.preprocessing import get_stage1_config_hash
    from src.data.stage1_cache import (
        STAGE1_CACHE_SCHEMA_VERSION,
        Stage1CroppedLoader,
        build_or_resume_stage1_cache,
        verify_stage1_cache,
    )

    s1_hash = get_stage1_config_hash()
    manifest_path = full_stage1_root / "stage1_manifest.json"
    use_full = False
    if manifest_path.exists():
        try:
            full_m = json.loads(manifest_path.read_text(encoding="utf-8"))
            full_ids = {str(e["image_id"]) for e in full_m.get("images", [])}
            need_ids = {str(x) for x in pilot_df["image_id"].tolist()}
            if (
                full_m.get("schema_version") == STAGE1_CACHE_SCHEMA_VERSION
                and full_m.get("stage1_config_hash") == s1_hash
                and full_m.get("split_manifest_sha256") == splits_sha256
                and need_ids.issubset(full_ids)
            ):
                use_full = True
        except Exception:
            use_full = False

    if use_full:
        print(f"[pilot] Reusing verified full Stage-1 uint16 cache at {full_stage1_root}", flush=True)
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        loader = Stage1CroppedLoader(full_stage1_root, m, max_cached_shards=16)
    else:
        print(f"[pilot] Building/resuming dedicated 1,200-image Stage-1 uint16 cache at {pilot_stage1_root}...", flush=True)
        m = build_or_resume_stage1_cache(
            cache_root=pilot_stage1_root,
            manifest_df=pilot_df,
            raw_root=raw_root,
            split_manifest_sha256=splits_sha256,
            images_per_shard=200,
            laterality_policy="strict",
            strict_segmentation=strict_segmentation,
            use_largest_cc=False,
            crop_margin_frac=0.10,
            on_progress=lambda msg: print(f"  [stage1] {msg}", flush=True),
        )
        verify_stage1_cache(
            cache_root=pilot_stage1_root,
            expected_df=pilot_df,
            expected_stage1_hash=s1_hash,
            allow_fallbacks=not strict_segmentation,
            verify_pixel_hashes=True,
        )
        RSNA_VOL.commit()
        loader = Stage1CroppedLoader(pilot_stage1_root, m, max_cached_shards=16)

    from concurrent.futures import ThreadPoolExecutor

    if getattr(loader, "id_to_png", None):
        with ThreadPoolExecutor(max_workers=8) as pool:
            train_crops = list(pool.map(loader.load_uint16, [str(x) for x in train_df["image_id"].tolist()]))
            tc_crops = list(pool.map(loader.load_uint16, [str(x) for x in tc_df["image_id"].tolist()]))
            vc_crops = list(pool.map(loader.load_uint16, [str(x) for x in vc_df["image_id"].tolist()]))
    else:
        train_crops = [loader.load_uint16(str(x)) for x in train_df["image_id"].tolist()]
        tc_crops = [loader.load_uint16(str(x)) for x in tc_df["image_id"].tolist()]
        vc_crops = [loader.load_uint16(str(x)) for x in vc_df["image_id"].tolist()]
    return train_crops, tc_crops, vc_crops, m.get("summary", {}), s1_hash


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
    strict_segmentation: bool = True,
    seed: int = 42,
) -> dict:
    """Run controlled 1,200-image online-vs-cached equivalence pilot on Modal L40S."""
    import pandas as pd
    import torch

    from src.data.feature_cache import compute_file_sha256
    from src.models.mammo_clip_loader import MAMMO_CLIP_SHA256, load_and_verify_mammo_clip
    from src.models.trust_model import MammoCLIPEncoder
    from src.training.parity_pilot import run_online_vs_cached_parity_pilot, sample_pilot_1200
    from src.utils.env_metadata import collect_runtime_environment_metadata

    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    mammo_clip_tar = MOUNT_PATH / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    pilot_cache_root = MOUNT_PATH / "cache" / "parity_pilot_1200"
    full_stage1_root = MOUNT_PATH / "cache" / "stage1_rsna_uint16"
    pilot_stage1_root = MOUNT_PATH / "cache" / "stage1_pilot1200_uint16"

    df_full = pd.read_csv(splits_csv)
    split_sha = compute_file_sha256(splits_csv)
    train_df, tc_df, vc_df = sample_pilot_1200(df_full, seed=seed)
    pilot_df = pd.concat([train_df, tc_df, vc_df], ignore_index=True)

    train_cropped, tc_cropped, vc_cropped, stage1_summary, s1_hash = _load_pilot_stage1_crops(
        pilot_df=pilot_df,
        train_df=train_df,
        tc_df=tc_df,
        vc_df=vc_df,
        raw_root=rsna_raw,
        splits_sha256=split_sha,
        full_stage1_root=full_stage1_root,
        pilot_stage1_root=pilot_stage1_root,
        strict_segmentation=strict_segmentation,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MammoCLIPEncoder(checkpoint_path=None, freeze=False, model_name="efficientnet-b5")
    diag = load_and_verify_mammo_clip(encoder, mammo_clip_tar, strict=True, verify_sha256=True)
    assert diag["matched_keys_count"] == 852 and diag["missing_keys_count"] == 0 and diag["unexpected_keys_count"] == 0
    encoder.freeze_encoder()

    modes = ["fp32", "amp"] if precision == "both" else [precision]
    results: dict = {
        "pilot_subset": {
            "total_images": int(len(pilot_df)),
            "train": int(len(train_df)),
            "train_calib": int(len(tc_df)),
            "val_calib": int(len(vc_df)),
            "stage1_summary": stage1_summary,
        },
        "environment_metadata": collect_runtime_environment_metadata(precision_policy=precision),
        "modes": {},
    }

    for pmode in modes:
        use_amp = pmode == "amp"
        env_meta = collect_runtime_environment_metadata(precision_policy=pmode)
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
            stage1_config_hash=s1_hash,
            micro_batch_size=micro_batch_size,
            environment_metadata=env_meta,
            on_progress=lambda msg: print(f"  [{pmode}] {msg}", flush=True),
        )
        results["modes"][pmode] = res
        print(
            f"[pilot] {pmode.upper()} status={res['status']} "
            f"max_prob_diff={res['max_abs_prob_diff']:.2e} "
            f"saw_nonzero_tau={res['saw_nonzero_epoch2_plus_tau']}",
            flush=True,
        )

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
    strict_segmentation: bool = True,
    gpu: str = "L40S",
) -> None:
    print(f"Launching 1,200-image Online vs Cached Parity Pilot on Modal ({gpu}, epochs={epochs}, precision={precision})...")
    res = run_pilot.with_options(gpu=gpu.upper()).remote(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        micro_batch_size=micro_batch_size,
        precision=precision,
        strict_segmentation=strict_segmentation,
    )
    print(json.dumps(res, indent=2))
