"""CPU-Only Modal Pipeline for Stage-1 Cropped 16-Bit RSNA Cache & Real-Data Audit.

Strictly CPU-only (NO GPU attached).
Implements:
  1. --audit-real: Inspects real RSNA DICOMs covering Left/Right CC/MLO across
     scanners/polarities, verifies Stage-1 LUT/mirroring/cropping, and saves
     side-by-side diagnostic PNGs to /data/reports/stage1_real_audit/.
  2. --build: Extracts lossless uint16 cropped images once per DICOM into
     /data/cache/stage1_rsna_uint16 (or /data/cache/stage1_pilot1200_uint16 with --pilot-only).
  3. --verify: Verifies complete Stage-1 cache integrity, checksums, and coverage.

Usage (DO NOT EXECUTE AUTOMATICALLY - for user execution):
    modal run scripts/cache_stage1_rsna_modal.py --audit-real
    modal run scripts/cache_stage1_rsna_modal.py --build --pilot-only
    modal run scripts/cache_stage1_rsna_modal.py --build
    modal run scripts/cache_stage1_rsna_modal.py --verify
"""

import json
from pathlib import Path
import signal
import sys
import threading
import time
import modal

from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "anchorshift-stage1-cpu-cache"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
MOUNT_PATH = Path("/data")
COMMIT_INTERVAL_SEC = 60

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES)
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

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def _register_signal_handlers(vol: modal.Volume, stop_event: threading.Event) -> None:
    def _handle(signum: int, _frame: object) -> None:
        print(f"\n[signal] Caught {signum}. Committing volume...", flush=True)
        stop_event.set()
        try:
            vol.commit()
        except Exception:
            pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    cpu=8.0,
    memory=16384,
    timeout=3600,
)
def audit_real_rsna_preprocessing(samples_per_view_lat: int = 4, seed: int = 42) -> dict:
    """Audit Stage-1 preprocessing on real RSNA Left/Right CC/MLO mammograms and save diagnostic PNGs."""
    import numpy as np
    import pandas as pd
    from PIL import Image

    from src.data.preprocessing import (
        decode_dicom_array,
        extract_cropped_mammogram_with_metadata,
        get_stage1_config_hash,
    )
    from src.utils.env_metadata import collect_runtime_environment_metadata

    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    out_dir = MOUNT_PATH / "reports" / "stage1_real_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(splits_csv)
    selected = []
    for lat in ("L", "R"):
        for view in ("CC", "MLO"):
            sub = df[(df["laterality"] == lat) & (df["view"] == view)]
            if len(sub) > 0:
                selected.append(sub.sample(n=min(samples_per_view_lat, len(sub)), random_state=seed))
    audit_df = pd.concat(selected, ignore_index=True)

    records = []
    for _, row in audit_df.iterrows():
        dcm_path = rsna_raw / str(row["rel_path"])
        arr, phot, dcm_lat = decode_dicom_array(dcm_path, apply_luts=True)
        res = extract_cropped_mammogram_with_metadata(
            arr=arr,
            photometric=phot,
            laterality=str(row["laterality"]),
            strict_segmentation=False,
            context_label=str(dcm_path),
        )

        # Save side-by-side 8-bit preview (raw min-max vs Stage-1 cropped) for human visual inspection
        raw_disp = arr if phot != "MONOCHROME1" else (np.max(arr) - arr)
        r_min, r_max = float(raw_disp.min()), float(raw_disp.max())
        raw_u8 = np.round(((raw_disp - r_min) / max(r_max - r_min, 1e-6)) * 255.0).astype(np.uint8)
        crop_u8 = np.round(res.cropped_float32 * 255.0).astype(np.uint8)

        raw_img = Image.fromarray(raw_u8).resize((384, 512))
        crop_img = Image.fromarray(crop_u8).resize((384, 512))
        canvas = Image.new("L", (784, 512), color=128)
        canvas.paste(raw_img, (0, 0))
        canvas.paste(crop_img, (400, 0))
        png_name = f"{row['laterality']}_{row['view']}_{row['patient_id']}_{row['image_id']}.png"
        canvas.save(out_dir / png_name)

        records.append(
            {
                "patient_id": str(row["patient_id"]),
                "image_id": str(row["image_id"]),
                "laterality": str(row["laterality"]),
                "dcm_laterality": dcm_lat,
                "view": str(row["view"]),
                "site_id": int(row["site_id"]) if "site_id" in row else None,
                "machine_id": int(row["machine_id"]) if "machine_id" in row else None,
                "photometric": phot,
                "orig_shape": list(res.orig_shape),
                "crop_shape": list(res.crop_shape),
                "crop_bbox": list(res.crop_bbox),
                "border_median": round(res.border_median, 6),
                "segmentation_status": res.segmentation_status,
                "used_fallback_uncropped": res.used_fallback_uncropped,
                "mirrored": res.mirrored,
                "preview_png": png_name,
            }
        )

    report = {
        "stage1_config_hash": get_stage1_config_hash(),
        "environment": collect_runtime_environment_metadata(precision_policy="cpu_stage1_uint16"),
        "total_audited": len(records),
        "success_count": sum(1 for r in records if r["segmentation_status"] == "success"),
        "fallback_count": sum(1 for r in records if r["used_fallback_uncropped"]),
        "photometrics": {k: sum(1 for r in records if r["photometric"] == k) for k in set(r["photometric"] for r in records)},
        "records": records,
    }
    (out_dir / "stage1_real_audit_report.json").write_text(json.dumps(report, indent=2))
    RSNA_VOL.commit()
    print(f"[audit] Audited {len(records)} real RSNA DICOMs -> {out_dir}", flush=True)
    return report


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    cpu=8.0,
    memory=16384,
    timeout=43200,
)
def build_stage1_cache_job(
    pilot_only: bool = False,
    images_per_shard: int = 250,
    strict_segmentation: bool = False,
    verify_only: bool = False,
    seed: int = 42,
) -> dict:
    """Build/resume or verify lossless CPU Stage-1 uint16 cropped RSNA cache."""
    import pandas as pd

    from src.data.preprocessing import get_stage1_config_hash
    from src.data.stage1_cache import build_or_resume_stage1_cache, verify_stage1_cache
    from src.training.parity_pilot import sample_pilot_1200
    from src.utils.env_metadata import collect_runtime_environment_metadata

    rsna_raw = MOUNT_PATH / "rsna" / "raw"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    df_full = pd.read_csv(splits_csv)
    split_sha = _file_sha256(splits_csv)
    stage1_hash = get_stage1_config_hash()

    if pilot_only:
        tr_p, tc_p, vc_p = sample_pilot_1200(df_full, seed=seed)
        target_df = pd.concat([tr_p, tc_p, vc_p], ignore_index=True)
        cache_root = MOUNT_PATH / "cache" / "stage1_pilot1200_uint16"
    else:
        target_df = df_full
        cache_root = MOUNT_PATH / "cache" / "stage1_rsna_uint16"

    if verify_only:
        manifest = verify_stage1_cache(
            cache_root=cache_root,
            expected_df=target_df,
            expected_stage1_hash=stage1_hash,
            expected_split_sha256=split_sha,
            allow_fallbacks=not strict_segmentation,
            verify_pixel_hashes=True,
        )
        return {"status": "VERIFIED", "cache_root": str(cache_root), "summary": manifest["summary"]}

    stop_event = threading.Event()
    _register_signal_handlers(RSNA_VOL, stop_event)
    commit_thread = _start_periodic_commit(RSNA_VOL, stop_event)

    t0 = time.perf_counter()
    try:
        manifest = build_or_resume_stage1_cache(
            cache_root=cache_root,
            manifest_df=target_df,
            raw_root=rsna_raw,
            split_manifest_sha256=split_sha,
            images_per_shard=images_per_shard,
            laterality_policy="strict",
            strict_segmentation=strict_segmentation,
            use_largest_cc=False,
            crop_margin_frac=0.10,
            on_progress=lambda m: print(f"[stage1] {m}", flush=True),
        )
        RSNA_VOL.commit()
        verify_stage1_cache(
            cache_root=cache_root,
            expected_df=target_df,
            expected_stage1_hash=stage1_hash,
            expected_split_sha256=split_sha,
            allow_fallbacks=not strict_segmentation,
            verify_pixel_hashes=True,
        )
        elapsed_s = time.perf_counter() - t0
        total_bytes = sum(s["size_bytes"] for s in manifest["shards"])
        imgs_per_sec = len(target_df) / max(elapsed_s, 1e-6)
        proj_54706_min = (54706.0 / max(imgs_per_sec, 1e-6)) / 60.0
        env = collect_runtime_environment_metadata(precision_policy="cpu_stage1_uint16")

        result = {
            "status": "COMPLETE_AND_VERIFIED",
            "cache_root": str(cache_root),
            "pilot_only": pilot_only,
            "elapsed_sec": round(elapsed_s, 2),
            "imgs_per_sec": round(imgs_per_sec, 3),
            "projected_full_54706_min": round(proj_54706_min, 2),
            "total_storage_mb": round(total_bytes / (1024**2), 2),
            "summary": manifest["summary"],
            "stage1_config_hash": stage1_hash,
            "environment": env,
        }
        (cache_root / "stage1_build_report.json").write_text(json.dumps(result, indent=2))
        RSNA_VOL.commit()
        return result
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)
        try:
            RSNA_VOL.commit()
        except Exception:
            pass


@app.local_entrypoint()
def main(
    audit_real: bool = False,
    build: bool = False,
    verify: bool = False,
    pilot_only: bool = False,
    images_per_shard: int = 250,
    strict_segmentation: bool = False,
    samples_per_view_lat: int = 4,
    seed: int = 42,
) -> None:
    if audit_real:
        res = audit_real_rsna_preprocessing.remote(samples_per_view_lat=samples_per_view_lat, seed=seed)
        print(json.dumps(res, indent=2))
        return
    if build or verify:
        res = build_stage1_cache_job.remote(
            pilot_only=pilot_only,
            images_per_shard=images_per_shard,
            strict_segmentation=strict_segmentation,
            verify_only=verify and not build,
            seed=seed,
        )
        print(json.dumps(res, indent=2))
        return
    raise ValueError("Specify one mode: --audit-real, --build, or --verify")
