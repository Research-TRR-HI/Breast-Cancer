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
    cpu=1.0,
    memory=2048,
    timeout=3600,
    max_containers=16,
)
def process_stage1_shard_worker(task: dict) -> dict:
    """Process one 250-image Stage-1 shard in an isolated 1-vCPU container."""
    import numpy as np
    from pathlib import Path
    from src.data.preprocessing import decode_dicom_array, extract_cropped_mammogram_with_metadata
    from src.data.stage1_cache import (
        _atomic_write_torch,
        compute_uint16_image_sha256,
        stage1_shard_filename,
    )

    shard_idx = int(task["shard_idx"])
    rows = task["rows"]
    cache_root = Path(task["cache_root"])
    raw_root = Path(task["raw_root"])
    expected_stage1_hash = str(task["expected_stage1_hash"])
    laterality_policy = str(task["laterality_policy"])
    strict_segmentation = bool(task["strict_segmentation"])
    crop_margin_frac = float(task["crop_margin_frac"])

    fname = stage1_shard_filename(shard_idx)
    shard_arrays = {}
    new_entries = []

    for row in rows:
        img_id = str(row["image_id"])
        dcm_path = raw_root / str(row["rel_path"])
        arr, phot, dcm_lat = decode_dicom_array(dcm_path, apply_luts=True)
        row_lat = str(row.get("laterality", "")).strip().upper()
        if row_lat in ("L", "R") and dcm_lat in ("L", "R") and row_lat != dcm_lat and laterality_policy == "strict":
            raise ValueError(
                f"Conflicting laterality between manifest ({row_lat!r}) and DICOM ({dcm_lat!r}) for {dcm_path}"
            )
        eff_lat = row_lat if row_lat in ("L", "R") else dcm_lat
        res = extract_cropped_mammogram_with_metadata(
            arr=arr,
            photometric=phot,
            laterality=eff_lat,
            invert_monochrome1=True,
            mirror_right=True,
            laterality_policy=laterality_policy,
            crop_breast=True,
            crop_margin_frac=crop_margin_frac,
            use_largest_cc=False,
            strict_segmentation=strict_segmentation,
            simulate_16bit_png=True,
            context_label=str(dcm_path),
        )
        img_sha = compute_uint16_image_sha256(res.cropped_uint16)
        shard_arrays[img_id] = np.ascontiguousarray(res.cropped_uint16)
        new_entries.append(
            {
                "image_id": img_id,
                "patient_id": str(row["patient_id"]),
                "split": str(row.get("split", "unknown")),
                "rel_path": str(row["rel_path"]),
                "laterality": res.laterality,
                "laterality_status": res.laterality_status,
                "photometric": res.photometric,
                "stage1_config_hash": expected_stage1_hash,
                "orig_shape": list(res.orig_shape),
                "crop_shape": list(res.crop_shape),
                "crop_bbox": list(res.crop_bbox),
                "border_median": round(res.border_median, 6),
                "segmentation_status": res.segmentation_status,
                "used_fallback_uncropped": bool(res.used_fallback_uncropped),
                "mirrored": bool(res.mirrored),
                "dtype": "uint16",
                "shard_file": fname,
                "image_sha256": img_sha,
            }
        )

    shard_path = cache_root / fname
    shard_sha = _atomic_write_torch(shard_path, {"images": shard_arrays})
    shard_meta = {
        "shard_idx": shard_idx,
        "file": fname,
        "sha256": shard_sha,
        "num_images": len(new_entries),
        "size_bytes": int(shard_path.stat().st_size),
    }
    RSNA_VOL.commit()
    return {"shard_meta": shard_meta, "entries": new_entries}


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    cpu=1.0,
    memory=4096,
    timeout=43200,
)
def build_stage1_cache_job(
    pilot_only: bool = False,
    images_per_shard: int = 250,
    strict_segmentation: bool = False,
    verify_only: bool = False,
    seed: int = 42,
) -> dict:
    """Build/resume or verify lossless CPU Stage-1 uint16 cropped RSNA cache using 16 parallel workers."""
    import pandas as pd

    from src.data.preprocessing import get_stage1_config_hash
    from src.data.stage1_cache import (
        STAGE1_CACHE_SCHEMA_VERSION,
        _atomic_write_json,
        _cleanup_stale_temps,
        _compute_file_sha256,
        _summarize_entries,
        stage1_shard_filename,
        verify_stage1_cache,
    )
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

    cache_root.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_temps(cache_root)
    manifest_path = cache_root / "stage1_manifest.json"
    df_reset = target_df.reset_index(drop=True)
    total_images = len(df_reset)
    total_shards = max((total_images + images_per_shard - 1) // images_per_shard, 1)

    valid_shards_by_idx = {}
    valid_entries_by_shard = {}
    if manifest_path.exists():
        try:
            old_m = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                old_m.get("schema_version") == STAGE1_CACHE_SCHEMA_VERSION
                and old_m.get("stage1_config_hash") == stage1_hash
                and old_m.get("split_manifest_sha256") == split_sha
                and int(old_m.get("images_per_shard", -1)) == int(images_per_shard)
            ):
                for sm in old_m.get("shards", []):
                    sp = cache_root / sm["file"]
                    if sp.exists() and _compute_file_sha256(sp) == sm["sha256"]:
                        valid_shards_by_idx[int(sm["shard_idx"])] = sm
                for e in old_m.get("images", []):
                    valid_entries_by_shard.setdefault(e["shard_file"], []).append(e)
        except Exception:
            valid_shards_by_idx = {}
            valid_entries_by_shard = {}

    shards_meta = []
    image_entries = []
    pending_tasks = []

    for shard_idx, start in enumerate(range(0, total_images, images_per_shard)):
        end = min(start + images_per_shard, total_images)
        shard_df = df_reset.iloc[start:end]
        fname = stage1_shard_filename(shard_idx)
        if shard_idx in valid_shards_by_idx and fname in valid_entries_by_shard:
            s_entries = valid_entries_by_shard[fname]
            if [e["image_id"] for e in s_entries] == [str(x) for x in shard_df["image_id"].tolist()]:
                shards_meta.append(valid_shards_by_idx[shard_idx])
                image_entries.extend(s_entries)
                continue

        rows_payload = [
            {
                "image_id": str(r["image_id"]),
                "patient_id": str(r["patient_id"]),
                "split": str(r["split"]) if "split" in r else "unknown",
                "rel_path": str(r["rel_path"]),
                "laterality": str(r["laterality"]) if ("laterality" in r and pd.notna(r["laterality"])) else "",
            }
            for _, r in shard_df.iterrows()
        ]
        pending_tasks.append(
            {
                "shard_idx": shard_idx,
                "rows": rows_payload,
                "cache_root": str(cache_root),
                "raw_root": str(rsna_raw),
                "expected_stage1_hash": stage1_hash,
                "laterality_policy": "strict",
                "strict_segmentation": strict_segmentation,
                "crop_margin_frac": 0.10,
            }
        )

    resumed_count = len(image_entries)
    print(
        f"[stage1] Resumed {len(shards_meta)}/{total_shards} shards ({resumed_count} imgs); "
        f"dispatching {len(pending_tasks)} shards across 16 parallel CPU workers...",
        flush=True,
    )

    t0 = time.perf_counter()
    newly_done = 0
    for worker_res in process_stage1_shard_worker.map(pending_tasks):
        s_meta = worker_res["shard_meta"]
        s_entries = worker_res["entries"]
        shards_meta.append(s_meta)
        image_entries.extend(s_entries)
        newly_done += len(s_entries)

        done_total = len(image_entries)
        elapsed_s = time.perf_counter() - t0
        rate = newly_done / max(elapsed_s, 1e-6)
        eta_hr = ((total_images - done_total) / max(rate, 1e-6)) / 3600.0 if rate > 0 else 0.0
        pct = (100.0 * done_total) / max(total_images, 1)
        filled = int(round(20 * done_total / max(total_images, 1)))
        bar = "#" * filled + "." * (20 - filled)
        print(
            f"[stage1] Shard {len(shards_meta)}/{total_shards} | [{bar}] "
            f"{done_total}/{total_images} ({pct:5.1f}%) | {rate:4.2f} img/s | ETA {eta_hr:4.2f}h",
            flush=True,
        )

        partial_manifest = {
            "schema_version": STAGE1_CACHE_SCHEMA_VERSION,
            "stage1_config_hash": stage1_hash,
            "split_manifest_sha256": str(split_sha),
            "images_per_shard": int(images_per_shard),
            "laterality_policy": "strict",
            "use_largest_cc": False,
            "crop_margin_frac": 0.10,
            "summary": _summarize_entries(image_entries),
            "shards": sorted(shards_meta, key=lambda x: int(x["shard_idx"])),
            "images": image_entries,
        }
        _atomic_write_json(manifest_path, partial_manifest)

    RSNA_VOL.reload()
    RSNA_VOL.commit()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    imgs_per_sec = total_images / max(elapsed_s, 1e-6)
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


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    cpu=1.0,
    memory=8192,
    timeout=1800,
    max_containers=16,
)
def repack_stage1_shard_to_png_worker(task: dict) -> dict:
    """Losslessly convert one 4.9 GB .pt shard into 250 tiny 16-bit PNGs with pixel SHA-256 parity check."""
    import cv2
    import numpy as np
    import torch
    from pathlib import Path
    from src.data.stage1_cache import compute_uint16_image_sha256

    cache_root = Path(task["cache_root"])
    png_dir = cache_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    shard_file = str(task["shard_file"])
    entries = task["entries"]

    shard_data = torch.load(str(cache_root / shard_file), map_location="cpu", weights_only=False)["images"]
    updated_entries = []
    total_png_bytes = 0

    for entry in entries:
        img_id = str(entry["image_id"])
        rel_png = f"png/{img_id}.png"
        out_path = cache_root / rel_png
        arr = shard_data[img_id]
        if not isinstance(arr, np.ndarray) or arr.dtype != np.uint16:
            raise ValueError(f"Expected uint16 ndarray for {img_id} in {shard_file}")

        ok = cv2.imwrite(str(out_path), arr)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {out_path}")

        read_back = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
        if read_back is None or compute_uint16_image_sha256(read_back) != entry["image_sha256"]:
            raise RuntimeError(f"Lossless 16-bit PNG roundtrip hash mismatch for {img_id}!")

        total_png_bytes += int(out_path.stat().st_size)
        new_e = dict(entry)
        new_e["png_rel_path"] = rel_png
        updated_entries.append(new_e)

    RSNA_VOL.commit()
    return {
        "shard_idx": int(task["shard_idx"]),
        "shard_file": shard_file,
        "entries": updated_entries,
        "png_bytes": total_png_bytes,
    }


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    cpu=1.0,
    memory=4096,
    timeout=14400,
)
def repack_stage1_to_png_job(delete_old_shards: bool = True) -> dict:
    """Repack all 219 4.9 GB Stage-1 .pt shards into lossless individual 16-bit PNGs and reclaim 1.05 TB."""
    import pandas as pd
    from src.data.preprocessing import get_stage1_config_hash
    from src.data.stage1_cache import (
        STAGE1_CACHE_SCHEMA_VERSION,
        _atomic_write_json,
        verify_stage1_cache,
    )

    cache_root = MOUNT_PATH / "cache" / "stage1_rsna_uint16"
    manifest_path = cache_root / "stage1_manifest.json"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    df_full = pd.read_csv(splits_csv)
    split_sha = _file_sha256(splits_csv)
    stage1_hash = get_stage1_config_hash()

    old_m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if old_m.get("storage_format") == "png16_individual":
        return {"status": "ALREADY_PNG16", "total_images": len(old_m.get("images", []))}

    by_shard = {}
    for e in old_m["images"]:
        by_shard.setdefault(e["shard_file"], []).append(e)

    tasks = [
        {
            "shard_idx": int(sm["shard_idx"]),
            "shard_file": sm["file"],
            "cache_root": str(cache_root),
            "entries": by_shard[sm["file"]],
        }
        for sm in sorted(old_m["shards"], key=lambda x: int(x["shard_idx"]))
    ]
    total_shards = len(tasks)
    total_images = len(old_m["images"])
    print(f"[repack] Converting {total_shards} shards ({total_images} imgs) -> lossless 16-bit PNGs (16 workers)...", flush=True)

    t0 = time.perf_counter()
    new_images = []
    total_png_bytes = 0
    for idx, res in enumerate(repack_stage1_shard_to_png_worker.map(tasks), start=1):
        new_images.extend(res["entries"])
        total_png_bytes += int(res["png_bytes"])
        elapsed = time.perf_counter() - t0
        rate = len(new_images) / max(elapsed, 1e-6)
        eta_min = ((total_images - len(new_images)) / max(rate, 1e-6)) / 60.0
        gb_so_far = total_png_bytes / (1024**3)
        pct = (100.0 * len(new_images)) / max(total_images, 1)
        filled = int(round(20 * len(new_images) / max(total_images, 1)))
        bar = "#" * filled + "." * (20 - filled)
        print(
            f"[repack] Shard {idx}/{total_shards} | [{bar}] {len(new_images)}/{total_images} "
            f"({pct:5.1f}%) | {gb_so_far:5.1f} GiB | {rate:5.1f} img/s | ETA {eta_min:4.1f}m",
            flush=True,
        )

    new_manifest = dict(old_m)
    new_manifest["schema_version"] = STAGE1_CACHE_SCHEMA_VERSION
    new_manifest["storage_format"] = "png16_individual"
    new_manifest["shards"] = []
    new_manifest["images"] = new_images
    _atomic_write_json(manifest_path, new_manifest)
    RSNA_VOL.reload()
    RSNA_VOL.commit()

    verify_stage1_cache(
        cache_root=cache_root,
        expected_df=df_full,
        expected_stage1_hash=stage1_hash,
        expected_split_sha256=split_sha,
        allow_fallbacks=True,
        verify_pixel_hashes=False,
    )

    deleted_shards = 0
    if delete_old_shards:
        for p in cache_root.glob("stage1_shard_*.pt"):
            p.unlink(missing_ok=True)
            deleted_shards += 1
        RSNA_VOL.commit()

    report = {
        "status": "REPACK_COMPLETE_AND_VERIFIED",
        "storage_format": "png16_individual",
        "total_images": len(new_images),
        "total_png_gib": round(total_png_bytes / (1024**3), 2),
        "deleted_old_pt_shards": deleted_shards,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    (cache_root / "stage1_repack_report.json").write_text(json.dumps(report, indent=2))
    RSNA_VOL.commit()
    return report


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    cpu=1.0,
    memory=4096,
    timeout=1800,
    max_containers=16,
)
def resize_stage1_png_worker(task: dict) -> dict:
    """Resize 250 4K Stage-1 uint16 PNGs -> exact (1520, 912) uint16 PNGs with [0, 1] min-max norm."""
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from src.data.stage1_cache import compute_uint16_image_sha256

    cache_root = Path(task["cache_root"])
    out_dir = cache_root / "png_1520x912"
    out_dir.mkdir(parents=True, exist_ok=True)

    updated_entries = []
    total_bytes = 0
    for entry in task["entries"]:
        img_id = str(entry["image_id"])
        in_path = cache_root / str(entry["png_rel_path"])
        rel_out = f"png_1520x912/{img_id}.png"
        out_path = cache_root / rel_out

        arr = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.dtype != np.uint16:
            raise ValueError(f"Invalid uint16 PNG at {in_path}")

        t = torch.from_numpy(arr.astype(np.float32) / 65535.0).unsqueeze(0).unsqueeze(0)
        if tuple(arr.shape) != (1520, 912):
            t = F.interpolate(t, size=(1520, 912), mode="bilinear", align_corners=False)
        t = t.squeeze(0).squeeze(0)
        mn = float(t.min().item())
        mx = float(t.max().item())
        if mx > mn:
            t = (t - mn) / (mx - mn)
        else:
            t = torch.zeros_like(t)
        out_u16 = np.round(t.clamp(0.0, 1.0).numpy() * 65535.0).astype(np.uint16)

        if not cv2.imwrite(str(out_path), out_u16):
            raise RuntimeError(f"cv2.imwrite failed for {out_path}")
        new_sha = compute_uint16_image_sha256(out_u16)
        total_bytes += int(out_path.stat().st_size)

        ne = dict(entry)
        ne["png_rel_path"] = rel_out
        ne["crop_shape"] = [1520, 912]
        ne["image_sha256"] = new_sha
        updated_entries.append(ne)

    RSNA_VOL.commit()
    return {"entries": updated_entries, "png_bytes": total_bytes}


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    cpu=1.0,
    memory=4096,
    timeout=14400,
)
def resize_stage1_to_1520_912_job(delete_old_4k: bool = True) -> dict:
    """Pre-resize all 54,706 Stage-1 PNGs to (1520, 912) across 16 workers and reclaim ~320 GiB."""
    import shutil
    import pandas as pd
    from src.data.preprocessing import get_stage1_config_hash
    from src.data.stage1_cache import STAGE1_CACHE_SCHEMA_VERSION, _atomic_write_json, verify_stage1_cache

    cache_root = MOUNT_PATH / "cache" / "stage1_rsna_uint16"
    manifest_path = cache_root / "stage1_manifest.json"
    splits_csv = MOUNT_PATH / "rsna" / "splits_seed42.csv"
    df_full = pd.read_csv(splits_csv)
    split_sha = _file_sha256(splits_csv)
    stage1_hash = get_stage1_config_hash()

    old_m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if old_m.get("pre_resized_shape") == [1520, 912]:
        return {"status": "ALREADY_1520x912", "total_images": len(old_m.get("images", []))}

    imgs = old_m["images"]
    chunk_size = 250
    tasks = [
        {"cache_root": str(cache_root), "entries": imgs[i : i + chunk_size]}
        for i in range(0, len(imgs), chunk_size)
    ]
    total_chunks = len(tasks)
    total_images = len(imgs)
    print(f"[resize-1520] Resizing {total_images} Stage-1 PNGs across 16 workers...", flush=True)

    t0 = time.perf_counter()
    new_images = []
    total_bytes = 0
    for idx, res in enumerate(resize_stage1_png_worker.map(tasks), start=1):
        new_images.extend(res["entries"])
        total_bytes += int(res["png_bytes"])
        elapsed = time.perf_counter() - t0
        rate = len(new_images) / max(elapsed, 1e-6)
        eta_min = ((total_images - len(new_images)) / max(rate, 1e-6)) / 60.0
        gb_so_far = total_bytes / (1024**3)
        pct = (100.0 * len(new_images)) / max(total_images, 1)
        filled = int(round(20 * len(new_images) / max(total_images, 1)))
        bar = "#" * filled + "." * (20 - filled)
        print(
            f"[resize-1520] Chunk {idx}/{total_chunks} | [{bar}] {len(new_images)}/{total_images} "
            f"({pct:5.1f}%) | {gb_so_far:5.1f} GiB | {rate:5.1f} img/s | ETA {eta_min:4.1f}m",
            flush=True,
        )

    new_m = dict(old_m)
    new_m["schema_version"] = STAGE1_CACHE_SCHEMA_VERSION
    new_m["storage_format"] = "png16_individual"
    new_m["pre_resized_shape"] = [1520, 912]
    new_m["images"] = new_images
    _atomic_write_json(manifest_path, new_m)
    RSNA_VOL.reload()
    RSNA_VOL.commit()

    verify_stage1_cache(
        cache_root=cache_root,
        expected_df=df_full,
        expected_stage1_hash=stage1_hash,
        expected_split_sha256=split_sha,
        allow_fallbacks=True,
        verify_pixel_hashes=False,
    )

    deleted_4k = False
    if delete_old_4k and (cache_root / "png").exists():
        shutil.rmtree(cache_root / "png", ignore_errors=True)
        deleted_4k = True
        RSNA_VOL.commit()

    report = {
        "status": "RESIZE_1520x912_COMPLETE",
        "total_images": len(new_images),
        "total_png_gib": round(total_bytes / (1024**3), 2),
        "deleted_old_4k_png_dir": deleted_4k,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    (cache_root / "stage1_resize_1520_report.json").write_text(json.dumps(report, indent=2))
    RSNA_VOL.commit()
    return report


@app.local_entrypoint()
def main(
    audit_real: bool = False,
    build: bool = False,
    verify: bool = False,
    repack_png: bool = False,
    delete_old_shards: bool = True,
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
    if repack_png:
        res = repack_stage1_to_png_job.remote(delete_old_shards=delete_old_shards)
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
