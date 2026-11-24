"""Paper 1 — NYU Four-View Step 1: CPU-Only Official Preprocessing & Reusable Volume Cache.

STRICT GUARANTEES:
    - ZERO GPU allocation anywhere in this script (strictly CPU-only Modal workers).
    - ZERO NYU model checkpoint loading, cancer prediction inference, or NLBS label access.
    - Uses ONLY frozen label-free 4-view manifests (`rsna_val_calib_4view_selection.csv` and
      `nlbs_4view_common_cohort_label_free.csv`), strictly rejecting missing/duplicate views.
    - Saves lossless uint16 cropped+oriented NYU views (`cropped_flip_u16`) + exact `best_center`
      so 10-epoch TTA in Step 2 produces 0.0 float32 difference vs. online raw DICOM preprocessing.
    - DO NOT launch `--stage build` without explicit user approval after pilot cost/parity check.

Usage:
    # Step 1 Small CPU Pilot ONLY (320 images = 80 4-view exams + parity/CPU-config checks):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/preprocess_paper1_nyu_cpu.py --stage pilot
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-nyu-cpu-preprocess"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
NLBS_VOL = modal.Volume.from_name("nlbs-mammography-vol", create_if_missing=True)

RSNA_MOUNT = Path("/rsna_vol")
NLBS_MOUNT = Path("/nlbs_vol")
CACHE_ROOT_NAME = "paper1_nyu_uint16_v1"

# Official Modal workspace rates verified via `modal billing rates` on 2026-11-22
MODAL_CPU_USD_PER_CORE_HR = 0.0473
MODAL_MEM_USD_PER_GIB_HR = 0.0080
MODAL_VOL_USD_PER_GIB_MONTH = 0.09

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
    .add_local_file(
        "datasets/rsna/splits_seed42.csv",
        remote_path="/root/datasets/rsna/splits_seed42.csv",
    )
    .add_local_file(
        "datasets/paper1/rsna_val_calib_4view_selection.csv",
        remote_path="/root/datasets/paper1/rsna_val_calib_4view_selection.csv",
    )
    .add_local_file(
        "datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
        remote_path="/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
    )
    .add_local_file(
        "reports/paper1/predictions_locked/gmic_rsna_val_calib_fallback_log.json",
        remote_path="/root/reports/paper1/predictions_locked/gmic_rsna_val_calib_fallback_log.json",
    )
)


def _compute_cpu_cost_usd(wall_sec: float, cpus: float, mem_gib: float) -> float:
    return (wall_sec / 3600.0) * (cpus * MODAL_CPU_USD_PER_CORE_HR + mem_gib * MODAL_MEM_USD_PER_GIB_HR)


def _load_split_rows(split_name: str, local_root: Optional[Path] = None) -> List[Dict[str, str]]:
    import pandas as pd
    from src.paper1.nyu_finetune import select_rsna_4view_split_exams
    from src.paper1.nyu_fourview_adapter import NYU_VIEWS, validate_and_load_nyu_4view_manifest

    base = Path("/root") if local_root is None else Path(local_root)
    if split_name in ("rsna_train", "rsna_train_calib", "rsna_train_pilot"):
        rsna_split = "train" if split_name == "rsna_train_pilot" else split_name.replace("rsna_", "", 1)
        splits_df = pd.read_csv(base / "datasets/rsna/splits_seed42.csv")
        picked_df, _ = select_rsna_4view_split_exams(splits_df, rsna_split)
        if split_name == "rsna_train_pilot":
            pilot_pids = sorted(picked_df["patient_id"].astype(str).unique())[:200]
            picked_df = picked_df[picked_df["patient_id"].astype(str).isin(pilot_pids)].copy()
        # Compute n_candidates_in_slot deterministically without leaking `cancer` into preprocessing rows
        sub = splits_df[splits_df["split"] == rsna_split].copy()
        sub["view_slot"] = (
            sub["laterality"].astype(str).str.strip().str.upper()
            + "-"
            + sub["view"].astype(str).str.strip().str.upper()
        )
        sub = sub[sub["view_slot"].isin(NYU_VIEWS)]
        cand_counts = sub.groupby(["patient_id", "view_slot"])["image_id"].count().to_dict()
        return [
            {
                "split_name": split_name,
                "case_id": str(r.patient_id),
                "item_id": str(r.image_id),
                "view_slot": str(r.view_slot),
                "dcm_path": str(RSNA_MOUNT / "rsna" / "raw" / r.rel_path),
                "rel_path": str(r.rel_path),
                "site_id": str(r.site_id),
                "machine_id": str(r.machine_id),
                "n_candidates_in_slot": int(cand_counts.get((r.patient_id, r.view_slot), 1)),
            }
            for r in picked_df.itertuples(index=False)
        ]
    if split_name == "rsna_val_calib":
        df = validate_and_load_nyu_4view_manifest(
            "rsna_val_calib", base / "datasets/paper1/rsna_val_calib_4view_selection.csv"
        )
        return [
            {
                "split_name": "rsna_val_calib",
                "case_id": str(r.patient_id),
                "item_id": str(r.image_id),
                "view_slot": str(r.view_slot),
                "dcm_path": str(RSNA_MOUNT / "rsna" / "raw" / r.rel_path),
                "rel_path": str(r.rel_path),
                "site_id": str(r.site_id),
                "machine_id": str(r.machine_id),
                "n_candidates_in_slot": int(r.n_candidates_in_slot),
            }
            for r in df.itertuples(index=False)
        ]
    if split_name == "nlbs_4view":
        df = validate_and_load_nyu_4view_manifest(
            "nlbs_4view", base / "datasets/paper1/nlbs_4view_common_cohort_label_free.csv"
        )
        return [
            {
                "split_name": "nlbs_4view",
                "case_id": str(r.StudyInstanceUID),
                "item_id": str(r.image_uid),
                "view_slot": str(r.view_slot),
                "dcm_path": str(NLBS_MOUNT / "nlbs" / "extracted" / r.dicom_rel_path),
                "rel_path": str(r.dicom_rel_path),
                "manufacturer": str(r.manufacturer),
                "photometric_manifest": str(r.photometric_interpretation),
                "n_candidates_in_slot": int(r.n_candidates_in_slot),
            }
            for r in df.itertuples(index=False)
        ]
    raise ValueError(f"Unsupported NYU split_name={split_name!r}")


def _process_one_nyu_view(
    row: Dict[str, str],
    png_dir: Path,
    nyu_mods: Dict[str, Any],
    compression: int = 1,
) -> Dict[str, Any]:
    from src.evaluation.stage_c_eval import compute_sha256_file
    from src.paper1.gmic_adapter import save_uint16_png_atomic
    from src.paper1.nyu_fourview_adapter import extract_nyu_cropped_view

    item_id = row["item_id"]
    out_png = png_dir / f"{item_id}.png"
    res = extract_nyu_cropped_view(Path(row["dcm_path"]), row["view_slot"], nyu_mods, include_uint16=True)
    u16 = res["image_u16"]
    meta = res["meta"]
    file_bytes = save_uint16_png_atomic(u16, out_png, compression=compression)
    png_sha = compute_sha256_file(out_png)
    u16_sha = hashlib.sha256(u16.tobytes()).hexdigest()
    return {
        "split_name": row["split_name"],
        "case_id": row["case_id"],
        "item_id": item_id,
        "view_slot": row["view_slot"],
        "height": int(u16.shape[0]),
        "width": int(u16.shape[1]),
        "dtype": str(u16.dtype),
        "best_center_y": int(res["best_center"][0]),
        "best_center_x": int(res["best_center"][1]),
        "crop_window": meta["crop_window"],
        "raw_shape": meta["raw_shape"],
        "file_bytes": int(file_bytes),
        "png_sha256": png_sha,
        "u16_sha256": u16_sha,
        "photometric": str(meta["photometric"]),
        "oriented_flipped": bool(meta["oriented_flipped"]),
        "crop_fallback": bool(res["crop_fallback"]),
        "fallback_strategy": str(meta.get("fallback_strategy", "")),
        "crop_error": str(meta.get("crop_error", "")),
    }


def _run_nyu_pilot_impl(
    n_exams_per_split: int = 40,
    workers: int = 4,
    compression: int = 1,
    verify_parity: bool = True,
) -> Dict[str, Any]:
    """Shared implementation for CPU worker efficiency comparison inside strict CPU-only containers."""
    import cv2
    import numpy as np
    import resource
    import torch
    from src.paper1.nyu_fourview_adapter import (
        NYU_INPUT_SIZES,
        NYU_VIEWS,
        _ensure_nyu_importable,
        extract_nyu_cropped_view,
        load_nyu_cached_view,
        select_representative_pilot_exams,
    )

    if torch.cuda.is_available():
        raise RuntimeError("GPU detected inside strict CPU-only NYU Step 1 pilot!")

    nyu_mods = _ensure_nyu_importable(Path("/root/external/breast_cancer_classifier"))
    n_load = nyu_mods["loading"]

    rsna_rows = _load_split_rows("rsna_val_calib")
    nlbs_rows = _load_split_rows("nlbs_4view")
    fb_doc = json.loads(
        Path("/root/reports/paper1/predictions_locked/gmic_rsna_val_calib_fallback_log.json").read_text(encoding="utf-8")
    )
    gmic_fb_cases = sorted({str(x["case_id"]) for x in fb_doc.get("fallbacks", [])})
    sel_rsna, sel_nlbs, sel_meta = select_representative_pilot_exams(
        rsna_rows, nlbs_rows, gmic_fb_cases, n_exams_per_split=n_exams_per_split
    )

    bench_tag = f"bench_{n_exams_per_split}ex_c{workers}_z{compression}"
    bench_root = RSNA_MOUNT / "cache" / f"{CACHE_ROOT_NAME}_{bench_tag}"
    for sp in ("rsna_val_calib", "nlbs_4view"):
        (bench_root / sp / "png").mkdir(parents=True, exist_ok=True)

    # Optional cold run (fresh files every pilot)
    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []
    all_rows = sel_rsna + sel_nlbs
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_process_one_nyu_view, r, bench_root / r["split_name"] / "png", nyu_mods, compression)
            for r in all_rows
        ]
        for f in futures:
            records.append(f.result())

    # Atomic shard/pilot JSONL + commit
    prog_path = bench_root / "pilot_manifest.jsonl"
    tmp_prog = prog_path.with_suffix(".jsonl.tmp")
    with open(tmp_prog, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_prog, prog_path)
    RSNA_VOL.commit()
    wall_sec = time.perf_counter() - t0
    host_rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))

    # Measure PNG compression trade-off (compression=3 vs compression=1) on 20 cached arrays (in memory, no write)
    alt_bytes: List[int] = []
    alt_t0 = time.perf_counter()
    for rec in records[:20]:
        png_p = bench_root / rec["split_name"] / "png" / f"{rec['item_id']}.png"
        arr = cv2.imread(str(png_p), cv2.IMREAD_UNCHANGED)
        ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError("cv2.imencode compression=3 check failed")
        alt_bytes.append(len(buf))
    comp3_encode_sec_20 = time.perf_counter() - alt_t0

    parity_summary: Dict[str, Any] = {}
    if verify_parity:
        # Check parity on 8 complete 4-view exams (32 views): RSNA patient 71 + 3 RSNA fallback/std + 2 SIEMENS + 2 GE NLBS
        rec_by_item = {r["item_id"]: r for r in records}
        rsna_by_case = {}
        nlbs_by_case = {}
        for r in sel_rsna:
            rsna_by_case.setdefault(r["case_id"], []).append(r)
        for r in sel_nlbs:
            nlbs_by_case.setdefault(r["case_id"], []).append(r)

        parity_cases: List[Tuple[str, List[Dict[str, str]]]] = []
        if "71" in rsna_by_case:
            parity_cases.append(("rsna_71_known_fallback", rsna_by_case["71"]))
        for c in sorted(rsna_by_case.keys(), key=int):
            if c != "71" and len([x for x in parity_cases if x[0].startswith("rsna")]) < 4:
                parity_cases.append((f"rsna_{c}", rsna_by_case[c]))
        for r_rows in list(nlbs_by_case.values()):
            is_sie = any(x.get("manufacturer") != "GE MEDICAL SYSTEMS" for x in r_rows)
            if is_sie and len([x for x in parity_cases if "siemens" in x[0]]) < 2:
                parity_cases.append((f"nlbs_siemens_{r_rows[0]['case_id'][-8:]}", r_rows))
        for r_rows in list(nlbs_by_case.values()):
            if len([x for x in parity_cases if x[0].startswith("nlbs")]) < 4 and all(
                r_rows[0]["case_id"] not in tag for tag, _ in parity_cases
            ):
                parity_cases.append((f"nlbs_ge_{r_rows[0]['case_id'][-8:]}", r_rows))

        img_max_diffs: List[float] = []
        tta_max_diffs: List[float] = []
        best_center_mismatches = 0
        for _, c_rows in parity_cases:
            online_views = {}
            cached_views = {}
            for r in c_rows:
                rec = rec_by_item[r["item_id"]]
                png_p = bench_root / rec["split_name"] / "png" / f"{rec['item_id']}.png"
                cached_vc = load_nyu_cached_view(
                    png_p,
                    best_center=(rec["best_center_y"], rec["best_center_x"]),
                    expected_shape=(rec["height"], rec["width"]),
                    expected_u16_sha256=rec["u16_sha256"],
                )
                online_vc = extract_nyu_cropped_view(Path(r["dcm_path"]), r["view_slot"], nyu_mods, include_uint16=False)
                if tuple(cached_vc["best_center"]) != tuple(online_vc["best_center"]):
                    best_center_mismatches += 1
                img_max_diffs.append(float(np.max(np.abs(cached_vc["image_f32"] - online_vc["image_f32"]))))
                cached_views[r["view_slot"]] = cached_vc
                online_views[r["view_slot"]] = online_vc

            # Verify full 10-epoch TTA tensor bit-exactness with identical RNG seed=0
            rng_on = np.random.RandomState(0)
            rng_ca = np.random.RandomState(0)
            for _ in range(10):
                for v in NYU_VIEWS:
                    t_on, _ = n_load.augment_and_normalize_image(
                        image=online_views[v]["image_f32"],
                        auxiliary_image=None,
                        view=v,
                        best_center=online_views[v]["best_center"],
                        random_number_generator=rng_on,
                        augmentation=True,
                        max_crop_noise=(100, 100),
                        max_crop_size_noise=100,
                    )
                    t_ca, _ = n_load.augment_and_normalize_image(
                        image=cached_views[v]["image_f32"],
                        auxiliary_image=None,
                        view=v,
                        best_center=cached_views[v]["best_center"],
                        random_number_generator=rng_ca,
                        augmentation=True,
                        max_crop_noise=(100, 100),
                        max_crop_size_noise=100,
                    )
                    if t_on.shape != NYU_INPUT_SIZES[v] or t_ca.shape != NYU_INPUT_SIZES[v]:
                        raise RuntimeError(f"TTA crop shape mismatch on {v}: {t_on.shape} vs {NYU_INPUT_SIZES[v]}")
                    tta_max_diffs.append(float(np.max(np.abs(t_on - t_ca))))

        parity_summary = {
            "n_parity_exams": len(parity_cases),
            "n_parity_views": sum(len(x[1]) for x in parity_cases),
            "n_tta_draws_per_exam": 10,
            "best_center_mismatches": best_center_mismatches,
            "max_image_abs_diff": float(max(img_max_diffs)) if img_max_diffs else None,
            "max_10tta_tensor_abs_diff": float(max(tta_max_diffs)) if tta_max_diffs else None,
        }

    # Verify view completeness & fallback counts across processed pilot records
    fbs = [r for r in records if r["crop_fallback"]]
    avg_bytes_z1 = float(np.mean([r["file_bytes"] for r in records]))
    avg_bytes_z1_20 = float(np.mean([r["file_bytes"] for r in records[:20]]))
    avg_bytes_z3_20 = float(np.mean(alt_bytes))

    return {
        "gpu_allocated": False,
        "cpu_cores": workers,
        "compression_level": compression,
        "selection_meta": sel_meta,
        "n_processed_images": len(records),
        "n_rsna_processed": sum(1 for r in records if r["split_name"] == "rsna_val_calib"),
        "n_nlbs_processed": sum(1 for r in records if r["split_name"] == "nlbs_4view"),
        "all_uint16_valid_nonzero": all(r["dtype"] == "uint16" and r["file_bytes"] > 0 and r["height"] > 0 and r["width"] > 0 for r in records),
        "n_crop_fallbacks": len(fbs),
        "fallback_cases": [
            {
                "split_name": r["split_name"],
                "case_id": r["case_id"],
                "item_id": r["item_id"],
                "view_slot": r["view_slot"],
                "photometric": r["photometric"],
                "raw_shape": r["raw_shape"],
                "crop_error": r["crop_error"],
            }
            for r in fbs
        ],
        "wall_sec": round(wall_sec, 3),
        "imgs_per_sec": round(len(records) / max(wall_sec, 1e-6), 3),
        "host_rss_gb": round(host_rss_gb, 3),
        "avg_png_mb": round(avg_bytes_z1 / (1024 * 1024), 3),
        "comp3_vs_comp1_sample20": {
            "avg_mb_z1": round(avg_bytes_z1_20 / (1024 * 1024), 3),
            "avg_mb_z3": round(avg_bytes_z3_20 / (1024 * 1024), 3),
            "size_ratio_z3_over_z1": round(avg_bytes_z3_20 / max(avg_bytes_z1_20, 1.0), 4),
            "extra_encode_sec_per_20": round(comp3_encode_sec_20, 3),
        },
        "parity": parity_summary,
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    cpu=4.0,
    memory=4096,
    timeout=3600,
)
def run_nyu_pilot_4c4g(n_exams_per_split: int = 40, compression: int = 1) -> Dict[str, Any]:
    """Primary 320-image (80-exam) CPU pilot on 4 vCPU / 4 GiB RAM."""
    res = _run_nyu_pilot_impl(
        n_exams_per_split=n_exams_per_split, workers=4, compression=compression, verify_parity=True
    )
    res["mem_gib"] = 4.0
    res["measured_cost_usd"] = round(_compute_cpu_cost_usd(res["wall_sec"], 4.0, 4.0), 5)
    return res


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    cpu=2.0,
    memory=3072,
    timeout=1800,
)
def run_nyu_efficiency_probe_2c3g(n_exams_per_split: int = 5, compression: int = 1) -> Dict[str, Any]:
    """Micro 40-image (10-exam) CPU efficiency check on 2 vCPU / 3 GiB RAM to compare cost per image."""
    res = _run_nyu_pilot_impl(
        n_exams_per_split=n_exams_per_split, workers=2, compression=compression, verify_parity=False
    )
    res["mem_gib"] = 3.0
    res["measured_cost_usd"] = round(_compute_cpu_cost_usd(res["wall_sec"], 2.0, 3.0), 5)
    return res


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    cpu=2.0,
    memory=8192,
    timeout=3600,
)
def validate_full_cpu_config_and_storage(n_exams_per_split: int = 20, compression: int = 1) -> Dict[str, Any]:
    """Benchmark actual full-run config (2 vCPU / 8 GiB RAM) on 160 imgs, test SHA-256 resume, and audit volume GiB."""
    import numpy as np
    import resource
    import shutil
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file
    from src.paper1.nyu_fourview_adapter import (
        _ensure_nyu_importable,
        select_representative_pilot_exams,
        verify_and_load_nyu_resumed_records,
    )

    if torch.cuda.is_available():
        raise RuntimeError("GPU detected inside strict CPU-only validation!")

    nyu_mods = _ensure_nyu_importable(Path("/root/external/breast_cancer_classifier"))
    rsna_rows = _load_split_rows("rsna_val_calib")
    nlbs_rows = _load_split_rows("nlbs_4view")
    fb_doc = json.loads(
        Path("/root/reports/paper1/predictions_locked/gmic_rsna_val_calib_fallback_log.json").read_text(encoding="utf-8")
    )
    gmic_fb_cases = sorted({str(x["case_id"]) for x in fb_doc.get("fallbacks", [])})
    sel_rsna, sel_nlbs, sel_meta = select_representative_pilot_exams(
        rsna_rows, nlbs_rows, gmic_fb_cases, n_exams_per_split=n_exams_per_split
    )

    val_root = RSNA_MOUNT / "cache" / f"{CACHE_ROOT_NAME}_validate_2c8g"
    if val_root.exists():
        shutil.rmtree(val_root)
    png_dir = val_root / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    prog_path = val_root / "shard_validate.jsonl"

    all_rows = sel_rsna + sel_nlbs
    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as pool, open(prog_path, "w", encoding="utf-8") as logf:
        recs = list(pool.map(lambda r: _process_one_nyu_view(r, png_dir, nyu_mods, compression), all_rows))
        for rec in recs:
            logf.write(json.dumps(rec) + "\n")
            records.append(rec)
        logf.flush()
        os.fsync(logf.fileno())
    RSNA_VOL.commit()
    wall_sec = time.perf_counter() - t0
    host_rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))

    # Live SHA-256 resume test: tamper 1 PNG preserving byte length + delete 1 PNG -> ensure BOTH rejected
    tampered_id = records[0]["item_id"]
    deleted_id = records[1]["item_id"]
    tp = png_dir / f"{tampered_id}.png"
    b = bytearray(tp.read_bytes())
    b[len(b) // 2] ^= 0x01
    tp.write_bytes(bytes(b))
    (png_dir / f"{deleted_id}.png").unlink()

    done_map, res_stats = verify_and_load_nyu_resumed_records(prog_path, png_dir)
    if res_stats["valid_resumed"] != len(records) - 2 or res_stats["rejected"] != 2:
        raise RuntimeError(f"Live SHA-256 resume check failed: {res_stats}")

    # Recompute the 2 rejected items and confirm 100% SHA-256 match
    todo_rows = [r for r in all_rows if r["item_id"] not in done_map]
    with ThreadPoolExecutor(max_workers=2) as pool, open(prog_path, "a", encoding="utf-8") as logf:
        for rec in pool.map(lambda r: _process_one_nyu_view(r, png_dir, nyu_mods, compression), todo_rows):
            logf.write(json.dumps(rec) + "\n")
        logf.flush()
        os.fsync(logf.fileno())
    RSNA_VOL.commit()
    done_after, stats_after = verify_and_load_nyu_resumed_records(prog_path, png_dir)
    if len(done_after) != len(all_rows) or stats_after["rejected"] != 0:
        raise RuntimeError(f"Post-repair SHA-256 resume verify failed: {stats_after}")
    # Also cross-check live hash vs stored png_sha256 for tampered item
    if compute_sha256_file(png_dir / f"{tampered_id}.png") != done_after[tampered_id]["png_sha256"]:
        raise RuntimeError("Repaired PNG hash mismatch")

    # Read-only exact storage footprint audit of /rsna_vol and /nlbs_vol (ZERO deletions!)
    def _dir_bytes(p: Path) -> int:
        tot = 0
        for dirpath, _, filenames in os.walk(p):
            dp = Path(dirpath)
            for fn in filenames:
                try:
                    tot += (dp / fn).lstat().st_size
                except OSError:
                    pass
        return tot

    rsna_vol_breakdown = {}
    for child in sorted(RSNA_MOUNT.iterdir()):
        if child.is_dir():
            if child.name == "cache":
                for csub in sorted(child.iterdir()):
                    if csub.is_dir():
                        rsna_vol_breakdown[f"cache/{csub.name}"] = round(_dir_bytes(csub) / (1024 ** 3), 3)
            else:
                rsna_vol_breakdown[child.name] = round(_dir_bytes(child) / (1024 ** 3), 3)
    nlbs_vol_gib = round(_dir_bytes(NLBS_MOUNT) / (1024 ** 3), 3)

    rsna_bytes = [r["file_bytes"] for r in records if r["split_name"] == "rsna_val_calib"]
    nlbs_bytes = [r["file_bytes"] for r in records if r["split_name"] == "nlbs_4view"]
    avg_rsna_mb = float(np.mean(rsna_bytes)) / (1024 * 1024)
    avg_nlbs_mb = float(np.mean(nlbs_bytes)) / (1024 * 1024)
    est_rsna_gib = (float(np.mean(rsna_bytes)) * 9528) / (1024 ** 3)
    est_nlbs_gib = (float(np.mean(nlbs_bytes)) * 24736) / (1024 ** 3)

    n_imgs = len(records)
    ips = n_imgs / max(wall_sec, 1e-6)
    cost_usd = _compute_cpu_cost_usd(wall_sec, 2.0, 8.0)

    return {
        "gpu_allocated": False,
        "config": "2vCPU/8GiB",
        "selection_meta": sel_meta,
        "n_processed_images": n_imgs,
        "n_fallbacks": int(sum(1 for r in records if r["crop_fallback"])),
        "wall_sec": round(wall_sec, 3),
        "imgs_per_sec": round(ips, 3),
        "host_rss_gb": round(host_rss_gb, 3),
        "rss_headroom_gb_under_8gib": round(8.0 - host_rss_gb, 3),
        "measured_cost_usd": round(cost_usd, 5),
        "sha256_resume_tamper_test": {
            "pass1_tampered_same_size": tampered_id,
            "pass1_deleted": deleted_id,
            "resumed_before_repair": res_stats,
            "resumed_after_repair": stats_after,
            "passed": True,
        },
        "storage_per_split": {
            "rsna_avg_mb": round(avg_rsna_mb, 3),
            "nlbs_avg_mb": round(avg_nlbs_mb, 3),
            "est_rsna_9528_gib": round(est_rsna_gib, 2),
            "est_nlbs_24736_gib": round(est_nlbs_gib, 2),
            "est_total_34264_gib": round(est_rsna_gib + est_nlbs_gib, 2),
        },
        "existing_volume_usage_gib_no_deletions": {
            "rsna_mammography_vol_breakdown_gib": rsna_vol_breakdown,
            "rsna_mammography_vol_total_gib": round(sum(rsna_vol_breakdown.values()), 3),
            "nlbs_mammography_vol_total_gib": nlbs_vol_gib,
            "combined_paper_volumes_gib": round(sum(rsna_vol_breakdown.values()) + nlbs_vol_gib, 3),
        },
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    cpu=2.0,
    memory=8192,
    timeout=3600,
)
def process_nyu_cpu_shard(
    split_name: str, shard_idx: int, num_shards: int, compression: int = 1
) -> Dict[str, Any]:
    """Resumable CPU-only NYU preprocessing worker (2 vCPU / 8 GiB RAM, exam-aligned sharding + SHA-256 resume)."""
    import resource
    import torch
    from src.paper1.nyu_fourview_adapter import (
        _ensure_nyu_importable,
        verify_and_load_nyu_resumed_records,
    )

    if torch.cuda.is_available():
        raise RuntimeError("GPU detected inside strict CPU-only NYU shard worker!")

    t_container0 = time.perf_counter()
    nyu_mods = _ensure_nyu_importable(Path("/root/external/breast_cancer_classifier"))
    rows_all = _load_split_rows(split_name)
    all_cases = sorted({r["case_id"] for r in rows_all})
    shard_case_set = set(all_cases[shard_idx::num_shards])
    rows = [r for r in rows_all if r["case_id"] in shard_case_set]

    split_root = RSNA_MOUNT / "cache" / CACHE_ROOT_NAME / split_name
    png_dir = split_root / "png"
    shard_dir = split_root / "shards"
    png_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    prog_path = shard_dir / f"shard_{shard_idx:03d}_of_{num_shards:03d}.jsonl"

    done_map, resume_stats = verify_and_load_nyu_resumed_records(prog_path, png_dir)
    todo = [r for r in rows if r["item_id"] not in done_map]
    skipped = len(rows) - len(todo)
    t0 = time.perf_counter()
    newly_processed = 0

    try:
        if todo:
            with ThreadPoolExecutor(max_workers=2) as pool, open(prog_path, "a", encoding="utf-8") as logf:
                batch_sz = 100
                for s in range(0, len(todo), batch_sz):
                    chunk = todo[s : s + batch_sz]
                    recs = list(pool.map(lambda r: _process_one_nyu_view(r, png_dir, nyu_mods, compression), chunk))
                    for rec in recs:
                        logf.write(json.dumps(rec) + "\n")
                        done_map[rec["item_id"]] = rec
                        newly_processed += 1
                    logf.flush()
                    os.fsync(logf.fileno())
                    RSNA_VOL.commit()
                    print(
                        f"[NYU-CPU {split_name} s{shard_idx:02d}/{num_shards}] "
                        f"done={len(done_map)}/{len(rows)} (new={newly_processed}, skip={skipped})",
                        flush=True,
                    )
    finally:
        RSNA_VOL.commit()

    wall_sec = time.perf_counter() - t0
    container_wall_sec = time.perf_counter() - t_container0
    host_rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))
    cost_usd = _compute_cpu_cost_usd(container_wall_sec, 2.0, 8.0)
    recs_final = [done_map[r["item_id"]] for r in rows]
    n_fallbacks = sum(1 for rec in recs_final if rec["crop_fallback"])
    return {
        "split_name": split_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "total_exams_in_shard": len(shard_case_set),
        "total_images_in_shard": len(rows),
        "skipped_resumed": skipped,
        "rejected_corrupt_resumed": resume_stats["rejected"],
        "newly_processed": newly_processed,
        "n_fallbacks": n_fallbacks,
        "loop_wall_sec": round(wall_sec, 2),
        "container_wall_sec": round(container_wall_sec, 2),
        "host_rss_gb": round(host_rss_gb, 3),
        "cost_usd": round(cost_usd, 5),
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    cpu=1.0,
    memory=2048,
    timeout=3600,
)
def verify_nyu_cpu_shard(split_name: str, shard_idx: int, num_shards: int = 12) -> Dict[str, Any]:
    """Verify 100% PNG SHA-256 checksums for one shard in parallel on Modal Volume."""
    import time
    import torch
    from src.paper1.nyu_fourview_adapter import verify_and_load_nyu_resumed_records

    if torch.cuda.is_available():
        raise RuntimeError("GPU detected inside CPU shard verification!")

    t0 = time.monotonic()
    RSNA_VOL.reload()
    split_root = RSNA_MOUNT / "cache" / CACHE_ROOT_NAME / split_name
    png_dir = split_root / "png"
    shard_path = split_root / "shards" / f"shard_{shard_idx:03d}_of_{num_shards:03d}.jsonl"
    sub_done, sub_stats = verify_and_load_nyu_resumed_records(shard_path, png_dir)
    if sub_stats["rejected"] > 0:
        raise RuntimeError(f"{split_name} shard {shard_idx}: {sub_stats['rejected']} PNGs failed SHA-256 check")
    wall_sec = time.monotonic() - t0
    return {
        "shard_idx": int(shard_idx),
        "verified_count": int(len(sub_done)),
        "rejected": int(sub_stats["rejected"]),
        "wall_sec": round(wall_sec, 2),
        "cost_usd": round(_compute_cpu_cost_usd(wall_sec, 1.0, 2.0), 5),
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def verify_nyu_cpu_cache(
    split_name: str,
    num_shards: int = 12,
    pixel_spot_n: int = 16,
    preverified_shards: Any = None,
) -> Dict[str, Any]:
    """Consolidate shard JSONLs, verify 100% PNG SHA-256 + 4-view exam completeness, and write manifest."""
    from concurrent.futures import ThreadPoolExecutor
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file
    from src.paper1.nyu_fourview_adapter import (
        NYU_VIEWS,
        verify_and_load_nyu_resumed_records,
    )

    if torch.cuda.is_available():
        raise RuntimeError("GPU detected inside CPU cache verification!")

    RSNA_VOL.reload()
    rows_all = _load_split_rows(split_name)
    split_root = RSNA_MOUNT / "cache" / CACHE_ROOT_NAME / split_name
    png_dir = split_root / "png"
    shard_dir = split_root / "shards"

    all_done: Dict[str, Dict[str, Any]] = {}
    if preverified_shards is not None:
        if len(preverified_shards) != num_shards:
            raise RuntimeError(f"{split_name}: expected {num_shards} shard verifications, got {len(preverified_shards)}")
        if any(int(sv.get("rejected", 1)) > 0 for sv in preverified_shards):
            raise RuntimeError(f"{split_name}: shard verification reported rejected PNGs")
        for s in range(num_shards):
            p = shard_dir / f"shard_{s:03d}_of_{num_shards:03d}.jsonl"
            if not p.exists():
                raise RuntimeError(f"{split_name}: missing shard JSONL {p}")
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                all_done[str(rec["item_id"])] = rec
    else:
        def _verify_one(s_idx: int):
            p = shard_dir / f"shard_{s_idx:03d}_of_{num_shards:03d}.jsonl"
            return verify_and_load_nyu_resumed_records(p, png_dir)

        with ThreadPoolExecutor(max_workers=min(12, max(1, num_shards))) as pool:
            for sub_done, sub_stats in pool.map(_verify_one, range(num_shards)):
                if sub_stats["rejected"] > 0:
                    raise RuntimeError(f"{split_name}: {sub_stats['rejected']} PNGs failed SHA-256 check")
                all_done.update(sub_done)

    exp_ids = [r["item_id"] for r in rows_all]
    if len(all_done) != len(exp_ids) or set(all_done.keys()) != set(exp_ids):
        raise RuntimeError(f"{split_name}: missing cached items ({len(all_done)} != {len(exp_ids)})")

    recs_ordered = [all_done[i] for i in exp_ids]
    df_man = pd.DataFrame(recs_ordered)
    if df_man["item_id"].duplicated().any():
        raise RuntimeError(f"{split_name}: duplicate item_id in consolidated manifest")
    v_counts = df_man.groupby("case_id")["view_slot"].nunique()
    if not (v_counts == len(NYU_VIEWS)).all():
        raise RuntimeError(f"{split_name}: incomplete 4-view exam coverage")

    # Pixel decode + u16_sha256 spot check on subset (all PNG files already SHA-256 verified on resume!)
    step = max(1, len(recs_ordered) // max(1, pixel_spot_n))
    for rec in recs_ordered[::step]:
        arr = cv2.imread(str(png_dir / f"{rec['item_id']}.png"), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.dtype != np.uint16 or arr.shape != (rec["height"], rec["width"]):
            raise RuntimeError(f"Decoded PNG mismatch for {rec['item_id']}")
        if hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest() != rec["u16_sha256"]:
            raise RuntimeError(f"Pixel u16_sha256 mismatch for {rec['item_id']}")

    man_csv = split_root / "manifest.csv"
    tmp_csv = split_root / "manifest.csv.tmp"
    df_man.to_csv(tmp_csv, index=False)
    tmp_csv.replace(man_csv)

    fb_df = df_man[df_man["crop_fallback"].astype(bool)]
    fallback_records = fb_df.to_dict(orient="records")
    (split_root / "fallback_log.json").write_text(json.dumps(fallback_records, indent=2), encoding="utf-8")
    summary = {
        "split_name": split_name,
        "n_images": int(len(df_man)),
        "n_exams": int(df_man["case_id"].nunique()),
        "n_fallback_views": int(len(fb_df)),
        "n_fallback_exams": int(fb_df["case_id"].nunique()) if len(fb_df) else 0,
        "pixel_spot_checked": int(len(recs_ordered[::step])),
        "total_storage_gib": round(float(df_man["file_bytes"].sum()) / (1024 ** 3), 3),
        "manifest_sha256": compute_sha256_file(man_csv),
    }
    if "view_slot" in df_man.columns:
        summary["by_view_slot"] = {str(k): int(v) for k, v in df_man["view_slot"].value_counts().sort_index().items()}
    if "photometric" in df_man.columns:
        summary["by_photometric"] = {str(k): int(v) for k, v in df_man["photometric"].value_counts().sort_index().items()}
    (split_root / "verify_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    RSNA_VOL.commit()
    return {**summary, "fallback_records": fallback_records}


@app.local_entrypoint()
def main(
    stage: str = "pilot",
    split: str = "all",
    num_shards: int = 12,
    n_exams_per_split: int = 40,
    compression: int = 1,
    approve_full_build: bool = False,
) -> None:
    from src.paper1.nyu_fourview_adapter import validate_nyu_cli_stage_and_guard

    target_splits = validate_nyu_cli_stage_and_guard(
        stage=stage, split=split, num_shards=num_shards, approve_full_build=approve_full_build
    )

    if stage == "validate_config":
        res = validate_full_cpu_config_and_storage.remote(n_exams_per_split=20, compression=compression)
        rsna_gib = res["storage_per_split"]["est_rsna_9528_gib"]
        nlbs_gib = res["storage_per_split"]["est_nlbs_24736_gib"]
        rate = res["imgs_per_sec"]
        # Account for container startup/commit overhead (~25s per container across num_shards containers/split)
        overhead_sec_per_shard = 25.0
        rsna_sec = (9528 / max(rate, 1e-6)) + (num_shards * overhead_sec_per_shard)
        nlbs_sec = (24736 / max(rate, 1e-6)) + (num_shards * overhead_sec_per_shard)
        total_sec = rsna_sec + nlbs_sec
        res["full_projection_2c8g_with_overhead"] = {
            "num_shards_per_split": num_shards,
            "overhead_sec_per_shard": overhead_sec_per_shard,
            "proj_rsna_9528_usd": round(_compute_cpu_cost_usd(rsna_sec, 2.0, 8.0), 4),
            "proj_nlbs_24736_usd": round(_compute_cpu_cost_usd(nlbs_sec, 2.0, 8.0), 4),
            "proj_total_34264_usd": round(_compute_cpu_cost_usd(total_sec, 2.0, 8.0), 4),
            "est_total_storage_gib": round(rsna_gib + nlbs_gib, 2),
        }
        out_p = Path("reports/paper1/nyu_step1_config_validation.json")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return

    if stage == "build":
        from pathlib import Path

        out_dir = Path("reports/paper1")
        out_dir.mkdir(parents=True, exist_ok=True)
        build_report: Dict[str, Any] = {}
        for sp in target_splits:
            outs = list(process_nyu_cpu_shard.starmap([(sp, s, num_shards, compression) for s in range(num_shards)]))
            (out_dir / f"nyu_step1_{sp}_shards_raw.json").write_text(json.dumps(outs, indent=2), encoding="utf-8")
            shard_vers = list(verify_nyu_cpu_shard.starmap([(sp, s, num_shards) for s in range(num_shards)]))
            ver = verify_nyu_cpu_cache.remote(
                sp, num_shards=num_shards, pixel_spot_n=100, preverified_shards=shard_vers
            )
            fallback_records = ver.pop("fallback_records", [])
            split_report = {
                "shards": outs,
                "verify_shards": shard_vers,
                "total_shard_compute_usd": round(sum(float(o["cost_usd"]) for o in outs), 4),
                "total_verify_shard_compute_usd": round(sum(float(v["cost_usd"]) for v in shard_vers), 4),
                "total_compute_usd": round(
                    sum(float(o["cost_usd"]) for o in outs) + sum(float(v["cost_usd"]) for v in shard_vers), 4
                ),
                "verify": ver,
            }
            (out_dir / f"nyu_step1_{sp}_build_summary.json").write_text(
                json.dumps(split_report, indent=2), encoding="utf-8"
            )
            (out_dir / f"nyu_step1_{sp}_fallback_log.json").write_text(
                json.dumps(fallback_records, indent=2), encoding="utf-8"
            )
            build_report[sp] = split_report
        print(json.dumps(build_report, indent=2))
        return

    if stage == "verify":
        from pathlib import Path

        out_dir = Path("reports/paper1")
        out_dir.mkdir(parents=True, exist_ok=True)
        ver_out: Dict[str, Any] = {}
        for sp in target_splits:
            shard_vers = list(verify_nyu_cpu_shard.starmap([(sp, s, num_shards) for s in range(num_shards)]))
            ver = verify_nyu_cpu_cache.remote(
                sp, num_shards=num_shards, pixel_spot_n=100, preverified_shards=shard_vers
            )
            fallback_records = ver.pop("fallback_records", [])
            split_ver = {
                "verify_shards": shard_vers,
                "total_verify_shard_compute_usd": round(sum(float(v["cost_usd"]) for v in shard_vers), 4),
                "verify": ver,
            }
            (out_dir / f"nyu_step1_{sp}_verify_summary.json").write_text(
                json.dumps(split_ver, indent=2), encoding="utf-8"
            )
            (out_dir / f"nyu_step1_{sp}_fallback_log.json").write_text(
                json.dumps(fallback_records, indent=2), encoding="utf-8"
            )
            ver_out[sp] = split_ver
        print(json.dumps(ver_out, indent=2))
        return

    main_res = run_nyu_pilot_4c4g.remote(n_exams_per_split=n_exams_per_split, compression=compression)
    probe_2c = run_nyu_efficiency_probe_2c3g.remote(n_exams_per_split=5, compression=compression)

    rsna_total = 9528
    nlbs_total = 24736
    grand_total = rsna_total + nlbs_total  # 34,264 images across 8,566 4-view exams

    def _project(cfg_res: Dict[str, Any], cpus: float, mem_gib: float) -> Dict[str, Any]:
        rate = cfg_res["imgs_per_sec"]
        full_sec = grand_total / max(rate, 1e-6)
        return {
            "config": f"{int(cpus)}vCPU/{mem_gib:.0f}GiB",
            "measured_imgs_per_sec": rate,
            "host_rss_gb": cfg_res["host_rss_gb"],
            "measured_cost_usd": cfg_res["measured_cost_usd"],
            "proj_full_wall_hours_single_container": round(full_sec / 3600.0, 3),
            "proj_rsna_9528_usd": round(_compute_cpu_cost_usd(rsna_total / max(rate, 1e-6), cpus, mem_gib), 4),
            "proj_nlbs_24736_usd": round(_compute_cpu_cost_usd(nlbs_total / max(rate, 1e-6), cpus, mem_gib), 4),
            "proj_total_34264_usd": round(_compute_cpu_cost_usd(full_sec, cpus, mem_gib), 4),
        }

    proj_4c4g = _project(main_res, 4.0, 4.0)
    proj_2c3g = _project(probe_2c, 2.0, 3.0)
    est_storage_gb_z1 = round((main_res["avg_png_mb"] * grand_total) / 1024.0, 2)
    z3_ratio = main_res["comp3_vs_comp1_sample20"]["size_ratio_z3_over_z1"]
    est_storage_gb_z3 = round(est_storage_gb_z1 * z3_ratio, 2)
    est_monthly_storage_usd_z1 = round(est_storage_gb_z1 * MODAL_VOL_USD_PER_GIB_MONTH, 2)
    est_monthly_storage_usd_z3 = round(est_storage_gb_z3 * MODAL_VOL_USD_PER_GIB_MONTH, 2)

    report = {
        "modal_profile": "hasinishrak2015",
        "cache_version": CACHE_ROOT_NAME,
        "pilot_main_4c4g": main_res,
        "efficiency_probe_2c3g": {
            "n_processed_images": probe_2c["n_processed_images"],
            "wall_sec": probe_2c["wall_sec"],
            "imgs_per_sec": probe_2c["imgs_per_sec"],
            "host_rss_gb": probe_2c["host_rss_gb"],
            "measured_cost_usd": probe_2c["measured_cost_usd"],
        },
        "projections_full_34264_images": {
            "4vCPU_4GiB": proj_4c4g,
            "2vCPU_3GiB": proj_2c3g,
            "storage_estimate_z1_gb": est_storage_gb_z1,
            "storage_estimate_z3_gb": est_storage_gb_z3,
            "additional_monthly_volume_usd_z1": est_monthly_storage_usd_z1,
            "additional_monthly_volume_usd_z3": est_monthly_storage_usd_z3,
        },
    }
    out_p = Path("reports/paper1/nyu_step1_cpu_pilot.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
