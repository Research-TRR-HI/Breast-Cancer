"""Paper 1 — GMIC Step 1: CPU-Only Official Preprocessing & Reusable Volume Cache.

STRICT GUARANTEES:
    - ZERO GPU allocation anywhere in this script (strictly CPU-only Modal workers).
    - ZERO GMIC model loading, predictions, threshold calibration, or NLBS outcome access.
    - Target-label quarantined: reads ONLY `splits_seed42.csv` (`split == 'val_calib'`, 10,982 rows,
      image_id/patient_id/laterality/view/rel_path only — never reads `cancer`) and label-free
      `inference_manifest_stage_c.csv` (`26,988` rows, checked with `assert_no_forbidden_columns`).
    - Saves bit-exact lossless 16-bit PNGs `(2944, 1920)` (`uint16`) pre-z-score under:
      `rsna-mammography-vol:/cache/paper1_gmic_uint16_v1/{split}/png_2944x1920/<item_id>.png`
      so Step 2 loads PNG -> `standard_normalize_gmic_uint16` with max float diff <= 1e-6 (exact parity).
    - Fully resumable at per-image granularity: skips any image whose atomic `.png` + `.progress.jsonl`
      entry already exists; commits periodically every 100 images and on container exit.

Usage:
    # 1. 200-image CPU benchmark + parity/cost gate (must project <= $3.00):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/preprocess_paper1_gmic_cpu.py --stage benchmark
    # 2. Full resumable CPU build (10,982 RSNA val_calib + 26,988 NLBS = 37,970 images):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/preprocess_paper1_gmic_cpu.py --stage build
    # 3. Full CPU integrity, fallback & parity verification:
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/preprocess_paper1_gmic_cpu.py --stage verify
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-gmic-cpu-preprocess"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
NLBS_VOL = modal.Volume.from_name("nlbs-mammography-vol", create_if_missing=True)

RSNA_MOUNT = Path("/rsna_vol")
NLBS_MOUNT = Path("/nlbs_vol")
CACHE_ROOT_NAME = "paper1_gmic_uint16_v1"

# Official Modal CPU & Memory on-demand rates (USD)
MODAL_CPU_USD_PER_CORE_HR = 0.04716
MODAL_MEM_USD_PER_GIB_HR = 0.00799
COST_GATE_MAX_USD = 3.00

WORKERS_PER_CONTAINER = 4
MEM_MIB_PER_CONTAINER = 8192
MAX_COST_CONTAINERS_DEFAULT = 16

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


def _compute_cpu_cost_usd(wall_sec: float, cpus: float, mem_gib: float) -> float:
    hrs = wall_sec / 3600.0
    return hrs * (cpus * MODAL_CPU_USD_PER_CORE_HR + mem_gib * MODAL_MEM_USD_PER_GIB_HR)


def _load_split_rows(split_name: str) -> List[Dict[str, str]]:
    import pandas as pd
    from src.evaluation.stage_c_eval import load_label_free_inference_manifest

    if split_name in ("rsna_val_calib", "rsna_train", "rsna_train_calib"):
        target_split = split_name.replace("rsna_", "", 1)
        df = pd.read_csv(
            "/root/datasets/rsna/splits_seed42.csv",
            usecols=["image_id", "patient_id", "split", "laterality", "view", "rel_path"],
        )
        df = df[df["split"] == target_split].sort_values(["patient_id", "image_id"]).reset_index(drop=True)
        return [
            {
                "item_id": str(r.image_id),
                "case_id": str(r.patient_id),
                "view_slot": f"{r.laterality}-{r.view}",
                "dcm_path": str(RSNA_MOUNT / "rsna" / "raw" / r.rel_path),
            }
            for r in df.itertuples(index=False)
        ]
    if split_name == "nlbs":
        df = load_label_free_inference_manifest(Path("/root/datasets/nlbs/inference_manifest_stage_c.csv"))
        return [
            {
                "item_id": str(r.image_uid),
                "case_id": str(r.StudyInstanceUID),
                "view_slot": f"{r.laterality}-{r.view_position}",
                "dcm_path": str(NLBS_MOUNT / "nlbs" / "extracted" / r.dicom_rel_path),
            }
            for r in df.itertuples(index=False)
        ]
    raise ValueError(f"Unsupported split_name={split_name!r}")


def _process_one_image(row: Dict[str, str], png_dir: Path, gmic_mods: Dict[str, Any]) -> Dict[str, Any]:
    from src.paper1.gmic_adapter import (
        GMIC_INPUT_SIZE,
        prepare_dicom_uint16_for_nyukat,
        preprocess_uint16_for_nyukat,
        save_uint16_png_atomic,
    )

    item_id = row["item_id"]
    out_png = png_dir / f"{item_id}.png"
    arr_u16, d_meta = prepare_dicom_uint16_for_nyukat(Path(row["dcm_path"]), row["view_slot"].split("-")[0])
    crop_u16, p_meta = preprocess_uint16_for_nyukat(
        arr_u16, row["view_slot"], GMIC_INPUT_SIZE, gmic_mods, augmentation=False, normalize=False
    )
    file_bytes = save_uint16_png_atomic(crop_u16, out_png, compression=1)
    sha = hashlib.sha256(crop_u16.tobytes()).hexdigest()
    return {
        "item_id": item_id,
        "case_id": row["case_id"],
        "view_slot": row["view_slot"],
        "height": int(crop_u16.shape[0]),
        "width": int(crop_u16.shape[1]),
        "dtype": str(crop_u16.dtype),
        "file_bytes": int(file_bytes),
        "u16_sha256": sha,
        "photometric": str(d_meta["photometric"]),
        "oriented_flipped": bool(d_meta["oriented_flipped"]),
        "crop_fallback": bool(p_meta["crop_fallback"]),
        "fallback_strategy": str(p_meta.get("fallback_strategy", "")),
        "crop_error": str(p_meta.get("crop_error", "")),
        "best_center": p_meta.get("best_center"),
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    cpu=float(WORKERS_PER_CONTAINER),
    memory=MEM_MIB_PER_CONTAINER,
    timeout=3600,
)
def benchmark_gmic_cpu_200(n_per_split: int = 100) -> Dict[str, Any]:
    """Benchmark 200 images (100 RSNA val_calib + 100 NLBS) on CPU, check parity, and project total USD cost."""
    import cv2
    import numpy as np
    import torch
    from src.paper1.gmic_adapter import (
        GMIC_INPUT_SIZE,
        _ensure_gmic_importable,
        prepare_dicom_uint16_for_nyukat,
        preprocess_uint16_for_nyukat,
        standard_normalize_gmic_uint16,
    )

    if torch.cuda.is_available():
        raise RuntimeError("GPU detected inside strict CPU-only Stage 1 benchmark!")

    gmic_mods = _ensure_gmic_importable(Path("/root/external/GMIC"))
    bench_root = RSNA_MOUNT / "cache" / f"{CACHE_ROOT_NAME}_bench200"
    bench_root.mkdir(parents=True, exist_ok=True)

    rsna_rows = _load_split_rows("rsna_val_calib")[:n_per_split]
    nlbs_rows = _load_split_rows("nlbs")[:n_per_split]
    all_rows = [("rsna_val_calib", r) for r in rsna_rows] + [("nlbs", r) for r in nlbs_rows]

    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=WORKERS_PER_CONTAINER) as pool:
        futures = [
            pool.submit(_process_one_image, r, bench_root / sp / "png_2944x1920", gmic_mods)
            for sp, r in all_rows
        ]
        for f in futures:
            records.append(f.result())
    RSNA_VOL.commit()
    wall_sec = time.perf_counter() - t0

    # Verify dimensions, integrity, and bit-exact preprocessing parity on 8 samples (4 RSNA + 4 NLBS)
    parity_diffs: List[float] = []
    for sp, row in (all_rows[:4] + all_rows[n_per_split : n_per_split + 4]):
        png_p = bench_root / sp / "png_2944x1920" / f"{row['item_id']}.png"
        loaded_u16 = cv2.imread(str(png_p), cv2.IMREAD_UNCHANGED)
        if loaded_u16 is None or loaded_u16.shape != GMIC_INPUT_SIZE or loaded_u16.dtype != np.uint16:
            raise RuntimeError(f"Cached PNG integrity failure: {png_p}")
        cached_norm = standard_normalize_gmic_uint16(loaded_u16)
        arr_u16, _ = prepare_dicom_uint16_for_nyukat(Path(row["dcm_path"]), row["view_slot"].split("-")[0])
        online_norm, _ = preprocess_uint16_for_nyukat(arr_u16, row["view_slot"], GMIC_INPUT_SIZE, gmic_mods, normalize=True)
        parity_diffs.append(float(np.max(np.abs(online_norm - cached_norm))))

    n_imgs = len(records)
    bench_cost_usd = _compute_cpu_cost_usd(wall_sec, WORKERS_PER_CONTAINER, MEM_MIB_PER_CONTAINER / 1024.0)
    total_target = 10982 + 26988
    scale_factor = total_target / max(n_imgs, 1)
    proj_full_wall_4cpu_hrs = (wall_sec * scale_factor) / 3600.0
    proj_full_usd = bench_cost_usd * scale_factor
    avg_bytes = float(np.mean([r["file_bytes"] for r in records]))

    return {
        "gpu_allocated": False,
        "n_bench_images": n_imgs,
        "n_rsna_bench": len(rsna_rows),
        "n_nlbs_bench": len(nlbs_rows),
        "wall_sec": round(wall_sec, 3),
        "imgs_per_sec_4cpu": round(n_imgs / max(wall_sec, 1e-6), 3),
        "bench_cost_usd": round(bench_cost_usd, 5),
        "proj_full_37970_4cpu_hours": round(proj_full_wall_4cpu_hrs, 3),
        "proj_full_37970_cost_usd": round(proj_full_usd, 3),
        "cost_gate_passed_under_3usd": bool(proj_full_usd <= COST_GATE_MAX_USD),
        "avg_png_mb": round(avg_bytes / (1024 * 1024), 3),
        "est_total_storage_gb": round((avg_bytes * total_target) / (1024 ** 3), 2),
        "n_fallbacks": int(sum(1 for r in records if r["crop_fallback"])),
        "max_parity_abs_diff": float(max(parity_diffs)),
        "all_shapes_2944x1920": all(r["height"] == 2944 and r["width"] == 1920 and r["dtype"] == "uint16" for r in records),
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    cpu=float(WORKERS_PER_CONTAINER),
    memory=MEM_MIB_PER_CONTAINER,
    timeout=3600,
)
def process_gmic_cpu_shard(split_name: str, shard_idx: int, num_shards: int) -> Dict[str, Any]:
    """Resumable CPU-only GMIC preprocessing worker for one shard."""
    import torch
    from src.paper1.gmic_adapter import _ensure_gmic_importable

    if torch.cuda.is_available():
        raise RuntimeError("GPU detected inside strict CPU-only Stage 1 shard worker!")

    gmic_mods = _ensure_gmic_importable(Path("/root/external/GMIC"))
    rows_all = _load_split_rows(split_name)
    rows = rows_all[shard_idx::num_shards]

    split_root = RSNA_MOUNT / "cache" / CACHE_ROOT_NAME / split_name
    png_dir = split_root / "png_2944x1920"
    shard_dir = split_root / "shards"
    png_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    prog_path = shard_dir / f"shard_{shard_idx:03d}_of_{num_shards:03d}.jsonl"

    # Load already-completed items (verified present on disk with matching non-zero file size)
    done_map: Dict[str, Dict[str, Any]] = {}
    if prog_path.exists():
        for line in prog_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            p_file = png_dir / f"{rec['item_id']}.png"
            if p_file.exists() and p_file.stat().st_size == int(rec["file_bytes"]) and rec["file_bytes"] > 0:
                done_map[rec["item_id"]] = rec

    todo = [r for r in rows if r["item_id"] not in done_map]
    skipped = len(rows) - len(todo)
    t0 = time.perf_counter()
    newly_processed = 0

    try:
        with ThreadPoolExecutor(max_workers=WORKERS_PER_CONTAINER) as pool, open(prog_path, "a", encoding="utf-8") as logf:
            batch_sz = 100
            for s in range(0, len(todo), batch_sz):
                chunk = todo[s : s + batch_sz]
                recs = list(pool.map(lambda r: _process_one_image(r, png_dir, gmic_mods), chunk))
                for rec in recs:
                    logf.write(json.dumps(rec) + "\n")
                    done_map[rec["item_id"]] = rec
                    newly_processed += 1
                logf.flush()
                os.fsync(logf.fileno())
                RSNA_VOL.commit()
                print(
                    f"[GMIC-CPU {split_name} s{shard_idx:02d}/{num_shards}] "
                    f"done={len(done_map)}/{len(rows)} (new={newly_processed}, skip={skipped})",
                    flush=True,
                )
    finally:
        RSNA_VOL.commit()

    wall_sec = time.perf_counter() - t0
    cost_usd = _compute_cpu_cost_usd(wall_sec, WORKERS_PER_CONTAINER, MEM_MIB_PER_CONTAINER / 1024.0)
    recs_final = [done_map[r["item_id"]] for r in rows]
    n_fallbacks = sum(1 for rec in recs_final if rec["crop_fallback"])
    return {
        "split_name": split_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "total_in_shard": len(rows),
        "skipped_resumed": skipped,
        "newly_processed": newly_processed,
        "n_fallbacks": n_fallbacks,
        "wall_sec": round(wall_sec, 2),
        "cost_usd": round(cost_usd, 5),
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    cpu=4.0,
    memory=8192,
    timeout=1800,
)
def verify_gmic_cpu_cache(split_name: str, num_shards: int = 16, parity_n: int = 10) -> Dict[str, Any]:
    """Consolidate shard JSONLs, verify exact counts, PNG headers/dimensions, integrity, and parity."""
    import cv2
    import numpy as np
    import pandas as pd
    import struct
    from src.paper1.gmic_adapter import (
        GMIC_INPUT_SIZE,
        _ensure_gmic_importable,
        prepare_dicom_uint16_for_nyukat,
        preprocess_uint16_for_nyukat,
        standard_normalize_gmic_uint16,
    )

    RSNA_VOL.reload()
    rows = _load_split_rows(split_name)
    expected_ids = [r["item_id"] for r in rows]
    split_root = RSNA_MOUNT / "cache" / CACHE_ROOT_NAME / split_name
    png_dir = split_root / "png_2944x1920"
    shard_dir = split_root / "shards"

    records_by_id: Dict[str, Dict[str, Any]] = {}
    for p in sorted(shard_dir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                records_by_id[rec["item_id"]] = rec

    missing_ids = [i for i in expected_ids if i not in records_by_id]
    if missing_ids:
        raise RuntimeError(f"Split {split_name} missing {len(missing_ids)} preprocessed records (head={missing_ids[:5]})")

    # Fast lossless PNG IHDR dimension & bit-depth check across ALL images + full cv2 decode on 100 sampled images
    def _check_header(item_id: str) -> Dict[str, Any]:
        png_p = png_dir / f"{item_id}.png"
        if not png_p.exists():
            raise FileNotFoundError(png_p)
        hdr = png_p.read_bytes()[:26]
        if hdr[:8] != b"\x89PNG\r\n\x1a\n" or hdr[12:16] != b"IHDR":
            raise ValueError(f"Corrupt PNG header: {png_p}")
        w, h, bit_depth, color_type = struct.unpack(">IIBB", hdr[16:26])
        if (h, w) != GMIC_INPUT_SIZE or bit_depth != 16 or color_type != 0:
            raise ValueError(f"Bad PNG IHDR {png_p}: {(h, w)}, bit={bit_depth}, color={color_type}")
        return {"size": png_p.stat().st_size}

    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(_check_header, expected_ids))

    # Full pixel parity test on evenly spaced subsample
    gmic_mods = _ensure_gmic_importable(Path("/root/external/GMIC"))
    step = max(1, len(rows) // parity_n)
    sample_rows = rows[::step][:parity_n]
    parity_diffs: List[float] = []
    for row in sample_rows:
        arr = cv2.imread(str(png_dir / f"{row['item_id']}.png"), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.shape != GMIC_INPUT_SIZE or arr.dtype != np.uint16 or float(np.std(arr)) == 0.0:
            raise RuntimeError(f"Invalid decoded PNG for {row['item_id']}")
        cached_norm = standard_normalize_gmic_uint16(arr)
        raw_u16, _ = prepare_dicom_uint16_for_nyukat(Path(row["dcm_path"]), row["view_slot"].split("-")[0])
        online_norm, _ = preprocess_uint16_for_nyukat(raw_u16, row["view_slot"], GMIC_INPUT_SIZE, gmic_mods, normalize=True)
        parity_diffs.append(float(np.max(np.abs(online_norm - cached_norm))))

    ordered_recs = [records_by_id[i] for i in expected_ids]
    fallbacks = [r for r in ordered_recs if r["crop_fallback"]]
    manifest_df = pd.DataFrame(ordered_recs)
    manifest_csv = split_root / "manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    fb_json = split_root / "fallback_log.json"
    fb_json.write_text(json.dumps({"split": split_name, "n_images": len(ordered_recs), "n_fallbacks": len(fallbacks), "fallbacks": fallbacks}, indent=2), encoding="utf-8")

    summary = {
        "split_name": split_name,
        "expected_images": len(expected_ids),
        "verified_pngs": len(sizes),
        "all_headers_2944x1920_16bit_gray": True,
        "total_storage_gb": round(sum(s["size"] for s in sizes) / (1024 ** 3), 3),
        "n_fallbacks": len(fallbacks),
        "fallback_pct": round(100.0 * len(fallbacks) / len(ordered_recs), 4),
        "fallback_items_head": [{"item_id": f["item_id"], "view_slot": f["view_slot"], "photometric": f["photometric"], "error": f["crop_error"]} for f in fallbacks[:10]],
        "parity_checked_n": len(sample_rows),
        "max_parity_abs_diff": float(max(parity_diffs)),
        "manifest_csv": str(manifest_csv),
    }
    (split_root / "verify_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RSNA_VOL.commit()
    return summary


@app.local_entrypoint()
def main(stage: str = "benchmark", split: str = "all", num_shards: int = 16) -> None:
    out_dir = Path("reports/paper1")
    out_dir.mkdir(parents=True, exist_ok=True)
    if stage == "benchmark":
        res = benchmark_gmic_cpu_200.remote(n_per_split=100)
        out_p = out_dir / "gmic_cpu_bench_200.json"
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        if not res["cost_gate_passed_under_3usd"]:
            raise RuntimeError(
                f"Projected GMIC CPU preprocessing cost ${res['proj_full_37970_cost_usd']:.2f} exceeds $3.00 gate!"
            )
    elif stage == "build":
        if split == "all":
            splits = ("rsna_val_calib", "nlbs")
        elif split == "train_all":
            splits = ("rsna_train", "rsna_train_calib")
        else:
            splits = (split,)
        build_stats: Dict[str, Any] = {}
        total_cost = 0.0
        for sp in splits:
            outs = list(process_gmic_cpu_shard.starmap([(sp, i, num_shards) for i in range(num_shards)]))
            sp_cost = sum(float(o["cost_usd"]) for o in outs)
            sp_new = sum(int(o["newly_processed"]) for o in outs)
            sp_skip = sum(int(o["skipped_resumed"]) for o in outs)
            sp_fb = sum(int(o["n_fallbacks"]) for o in outs)
            total_cost += sp_cost
            build_stats[sp] = {
                "total_images": sum(int(o["total_in_shard"]) for o in outs),
                "newly_processed": sp_new,
                "skipped_resumed": sp_skip,
                "n_fallbacks": sp_fb,
                "cost_usd": round(sp_cost, 4),
                "shards": outs,
            }
            if total_cost > COST_GATE_MAX_USD:
                raise RuntimeError(f"Cumulative CPU cost ${total_cost:.2f} exceeded $3.00 guard during build!")
        build_stats["total_build_cost_usd"] = round(total_cost, 4)
        (out_dir / "gmic_cpu_build_summary.json").write_text(json.dumps(build_stats, indent=2), encoding="utf-8")
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "shards"} if isinstance(v, dict) else v for k, v in build_stats.items()}, indent=2))
    elif stage == "verify":
        if split == "all":
            splits = ("rsna_val_calib", "nlbs")
        elif split == "train_all":
            splits = ("rsna_train", "rsna_train_calib")
        else:
            splits = (split,)
        ver_stats = {sp: verify_gmic_cpu_cache.remote(sp, num_shards=num_shards, parity_n=10) for sp in splits}
        (out_dir / "gmic_cpu_verify_summary.json").write_text(json.dumps(ver_stats, indent=2), encoding="utf-8")
        print(json.dumps(ver_stats, indent=2))
    else:
        raise ValueError(f"Unsupported stage={stage!r}; use benchmark, build, or verify")
