"""Paper 1 — Resumable GMIC (Pretrained Model 1) Step 2 L40S GPU Pilot & Production Inference.

Consumes the verified Step-1 lossless `(2944, 1920)` uint16 PNG cache at:
    /rsna_vol/cache/paper1_gmic_uint16_v1/{rsna_val_calib,nlbs}/png_2944x1920/
Stages cached PNGs to local NVMe `/tmp`, applies exact GMIC z-score standardization
(`standard_normalize_gmic_uint16`), and runs FP32 L40S inference with `sample_model_1.p`.

Usage:
    # Step 2 GPU Pilot (400 cached images + sample_data parity + 80 fallback & 5 view audit):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_gmic_modal.py --stage pilot
    # Full Step 2 Inference (DO NOT RUN UNTIL PILOT APPROVED):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_gmic_modal.py --stage infer --split rsna_val_calib
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_gmic_modal.py --stage infer --split nlbs
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-gmic-inference"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
NLBS_VOL = modal.Volume.from_name("nlbs-mammography-vol")

RSNA_MOUNT = Path("/rsna_vol")
NLBS_MOUNT = Path("/nlbs_vol")
GMIC_CKPT_SHA256 = "3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991"
L40S_COMBINED_HOURLY_USD = 2.3618  # Nvidia L40S ($1.95) + 6 vCPU ($0.2838) + 16 GiB RAM ($0.1280)
PREPROCESS_VERSION = "paper1_gmic_uint16_v1"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES, "imageio==2.37.0", "tqdm==4.67.1")
    .add_local_python_source("src")
    .add_local_dir("external/GMIC/src", remote_path="/root/external/GMIC/src")
    .add_local_dir("external/GMIC/sample_data/images", remote_path="/root/external/GMIC/sample_data/images")
    .add_local_file("datasets/rsna/splits_seed42.csv", remote_path="/root/datasets/rsna/splits_seed42.csv")
    .add_local_file(
        "datasets/nlbs/inference_manifest_stage_c.csv",
        remote_path="/root/datasets/nlbs/inference_manifest_stage_c.csv",
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
                raise FileNotFoundError(f"Missing/empty Step-1 PNG: {s}")
            shutil.copy2(s, d)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_cp, missing))
    return time.perf_counter() - t0


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=3600,
)
def run_gmic_step2_pilot(n_per_split: int = 200, batch_size: int = 8) -> Dict[str, Any]:
    """Run GMIC Step 2 L40S pilot: 200 RSNA + 200 NLBS cached PNGs, parity checks, and fallback audit."""
    import cv2
    import numpy as np
    import pandas as pd
    import resource
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file, load_label_free_inference_manifest
    from src.paper1.gmic_adapter import (
        GMIC_INPUT_SIZE,
        _canonical_nyukat_view_slot,
        build_gmic_model,
        prepare_dicom_uint16_for_nyukat,
        preprocess_uint16_for_nyukat,
        score_gmic_preprocessed_tensor,
        standard_normalize_gmic_uint16,
    )

    RSNA_VOL.reload()
    t_total0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L40S GPU expected for Step 2 pilot!")

    ckpt_path = RSNA_MOUNT / "models" / "paper1" / "sample_model_1.p"
    got_sha = compute_sha256_file(ckpt_path).lower()
    if got_sha != GMIC_CKPT_SHA256:
        raise RuntimeError(f"GMIC checkpoint SHA mismatch: {got_sha} != {GMIC_CKPT_SHA256}")

    t_load0 = time.perf_counter()
    model, gmic_mods = build_gmic_model(
        ckpt_path, model_index="1", device=device, gmic_repo_dir=Path("/root/external/GMIC")
    )
    model_load_sec = time.perf_counter() - t_load0

    # 1. Official sample_data L40S parity check (all 4 views of exam 0)
    expected_readme = {
        "0_L-CC": (0.1356, 0.0081),
        "0_R-CC": (0.8929, 0.3259),
        "0_L-MLO": (0.2368, 0.0335),
        "0_R-MLO": (0.9509, 0.1812),
    }
    sample_parity: Dict[str, Any] = {}
    for s_name, (exp_b, exp_m) in expected_readme.items():
        view_slot = s_name.split("_", 1)[1]
        img = gmic_mods["loading"].read_image_png(f"/root/external/GMIC/sample_data/images/{s_name}.png")
        proc_u16, _ = preprocess_uint16_for_nyukat(img, view_slot, GMIC_INPUT_SIZE, gmic_mods, normalize=False)
        proc_f32 = standard_normalize_gmic_uint16(proc_u16)
        pred_b, pred_m = score_gmic_preprocessed_tensor(model, proc_f32, device)
        sample_parity[s_name] = {
            "benign_pred": round(pred_b, 5),
            "malignant_pred": round(pred_m, 5),
            "readme_benign": exp_b,
            "readme_malignant": exp_m,
            "benign_abs_err": round(abs(pred_b - exp_b), 5),
            "malignant_abs_err": round(abs(pred_m - exp_m), 5),
        }

    # 2. Label-free audit of ALL 80 RSNA fallbacks and ALL 5 supplemental views from Step 1 cache
    rsna_cache_root = RSNA_MOUNT / "cache" / "paper1_gmic_uint16_v1" / "rsna_val_calib"
    nlbs_cache_root = RSNA_MOUNT / "cache" / "paper1_gmic_uint16_v1" / "nlbs"
    rsna_step1 = pd.read_csv(rsna_cache_root / "manifest.csv", dtype={"item_id": str, "case_id": str})
    nlbs_step1 = pd.read_csv(nlbs_cache_root / "manifest.csv", dtype={"item_id": str, "case_id": str})

    # Attach only raw rel_path (never reading `cancer`) so parity check can decode 5 RSNA DICOMs
    rsna_paths_df = pd.read_csv(
        "/root/datasets/rsna/splits_seed42.csv", usecols=["image_id", "split", "rel_path"]
    )
    rsna_paths_df = rsna_paths_df[rsna_paths_df["split"] == "val_calib"].copy()
    rsna_paths_df["image_id"] = rsna_paths_df["image_id"].astype(str)
    rsna_rel_map = rsna_paths_df.set_index("image_id")["rel_path"].to_dict()

    nlbs_manifest = load_label_free_inference_manifest(Path("/root/datasets/nlbs/inference_manifest_stage_c.csv"))
    nlbs_rel_map = nlbs_manifest.set_index(nlbs_manifest["image_uid"].astype(str))["dicom_rel_path"].to_dict()

    fb_rows = rsna_step1[rsna_step1["crop_fallback"].astype(bool)].copy()
    supp_rows = rsna_step1[~rsna_step1["view_slot"].isin(["L-CC", "R-CC", "L-MLO", "R-MLO"])].copy()

    # Check pixel integrity across ALL 80 fallback PNGs + ALL 5 supplemental-view PNGs via ThreadPool
    audit_ids = sorted(set(fb_rows["item_id"].tolist() + supp_rows["item_id"].tolist()))
    audit_nvme = Path("/tmp/gmic_pilot_audit_png")
    _stage_pngs_nvme(audit_ids, rsna_cache_root / "png_2944x1920", audit_nvme, max_workers=16)

    def _inspect_png(item_id: str) -> Dict[str, Any]:
        arr = cv2.imread(str(audit_nvme / f"{item_id}.png"), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.shape != GMIC_INPUT_SIZE or arr.dtype != np.uint16:
            return {"item_id": item_id, "valid": False}
        nz_frac = float(np.mean(arr > 0))
        return {
            "item_id": item_id,
            "valid": True,
            "nonzero_frac": round(nz_frac, 4),
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "max": int(np.max(arr)),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        pixel_stats_map = {r["item_id"]: r for r in pool.map(_inspect_png, audit_ids)}

    fallback_audit = {
        "n_fallback_images": int(len(fb_rows)),
        "n_unique_cases": int(fb_rows["case_id"].nunique()),
        "by_view_slot": fb_rows["view_slot"].value_counts().to_dict(),
        "by_photometric": fb_rows["photometric"].value_counts().to_dict(),
        "by_error": fb_rows["crop_error"].value_counts().to_dict(),
        "overlap_with_supplemental_views": int(len(set(fb_rows["item_id"]) & set(supp_rows["item_id"]))),
        "all_fallback_pngs_valid_nonzero": all(
            pixel_stats_map[i]["valid"] and pixel_stats_map[i]["std"] > 0.0 for i in fb_rows["item_id"]
        ),
        "fallback_nonzero_frac_min_med_max": [
            round(float(np.min([pixel_stats_map[i]["nonzero_frac"] for i in fb_rows["item_id"]])), 4),
            round(float(np.median([pixel_stats_map[i]["nonzero_frac"] for i in fb_rows["item_id"]])), 4),
            round(float(np.max([pixel_stats_map[i]["nonzero_frac"] for i in fb_rows["item_id"]])), 4),
        ],
    }

    supplemental_view_audit = [
        {
            "item_id": str(r.item_id),
            "case_id": str(r.case_id),
            "raw_view_slot": str(r.view_slot),
            "nyukat_proxy_view": _canonical_nyukat_view_slot(str(r.view_slot)),
            "photometric": str(r.photometric),
            "oriented_flipped": bool(r.oriented_flipped),
            "crop_fallback": bool(r.crop_fallback),
            "png_stats": pixel_stats_map[str(r.item_id)],
        }
        for r in supp_rows.itertuples(index=False)
    ]

    # 3. Pilot Batch Selection: 200 RSNA (incl. 5 supplemental + 5 fallback + 190 standard) & 200 NLBS
    supp_ids = supp_rows["item_id"].tolist()
    fb_sample_ids = [i for i in fb_rows["item_id"].tolist() if i not in set(supp_ids)][:5]
    std_ids = [
        i for i in rsna_step1["item_id"].tolist() if i not in set(supp_ids) and i not in set(fb_sample_ids)
    ][: max(0, n_per_split - len(supp_ids) - len(fb_sample_ids))]
    rsna_pilot_ids = std_ids + supp_ids + fb_sample_ids
    nlbs_pilot_ids = nlbs_step1["item_id"].tolist()[:n_per_split]

    pilot_nvme = Path("/tmp/gmic_pilot_400_png")
    if pilot_nvme.exists():
        shutil.rmtree(pilot_nvme)
    stage_rsna_sec = _stage_pngs_nvme(rsna_pilot_ids, rsna_cache_root / "png_2944x1920", pilot_nvme / "rsna", max_workers=16)
    stage_nlbs_sec = _stage_pngs_nvme(nlbs_pilot_ids, nlbs_cache_root / "png_2944x1920", pilot_nvme / "nlbs", max_workers=16)
    nvme_stage_sec = stage_rsna_sec + stage_nlbs_sec

    pilot_items = [("rsna", i, pilot_nvme / "rsna" / f"{i}.png") for i in rsna_pilot_ids] + [
        ("nlbs", i, pilot_nvme / "nlbs" / f"{i}.png") for i in nlbs_pilot_ids
    ]

    # Warmup GPU with 1 image outside timed loop
    w_arr = cv2.imread(str(pilot_items[0][2]), cv2.IMREAD_UNCHANGED)
    _ = score_gmic_preprocessed_tensor(model, standard_normalize_gmic_uint16(w_arr), device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def _load_norm_png(item):
        arr = cv2.imread(str(item[2]), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.shape != GMIC_INPUT_SIZE or arr.dtype != np.uint16:
            raise RuntimeError(f"Bad cached PNG: {item[2]}")
        return standard_normalize_gmic_uint16(arr)

    decode_sec = 0.0
    gpu_sec = 0.0
    pilot_preds: Dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for s in range(0, len(pilot_items), batch_size):
            batch = pilot_items[s : s + batch_size]
            t_d0 = time.perf_counter()
            norms = list(pool.map(_load_norm_png, batch))
            bx = torch.from_numpy(np.stack(norms, axis=0)[:, np.newaxis, :, :].copy()).to(device, non_blocking=True)
            decode_sec += time.perf_counter() - t_d0

            torch.cuda.synchronize()
            t_g0 = time.perf_counter()
            with torch.no_grad():
                outs = model(bx).float().cpu().numpy()
            torch.cuda.synchronize()
            gpu_sec += time.perf_counter() - t_g0

            for (ds_tag, item_id, _), row_out in zip(batch, outs):
                pilot_preds[f"{ds_tag}:{item_id}"] = (float(row_out[0]), float(row_out[1]))

    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    host_rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))

    # 4. Batch=8 vs Batch=1 equivalence + Cached PNG vs Raw DICOM Online Inference Parity (5 RSNA + 5 NLBS)
    rsna_check_ids = std_ids[:2] + supp_ids[:2] + fb_sample_ids[:1]
    nlbs_check_ids = nlbs_pilot_ids[:5]
    rsna_slot_map = rsna_step1.set_index("item_id")["view_slot"].to_dict()
    nlbs_slot_map = nlbs_step1.set_index("item_id")["view_slot"].to_dict()

    batch_vs_b1_diffs: List[float] = []
    cached_vs_raw_diffs: List[float] = []
    parity_rows = []
    for ds_tag, item_id, slot, dcm_p in (
        [("rsna", i, rsna_slot_map[i], RSNA_MOUNT / "rsna" / "raw" / rsna_rel_map[i]) for i in rsna_check_ids]
        + [("nlbs", i, nlbs_slot_map[i], NLBS_MOUNT / "nlbs" / "extracted" / nlbs_rel_map[i]) for i in nlbs_check_ids]
    ):
        cached_arr = cv2.imread(str(pilot_nvme / ds_tag / f"{item_id}.png"), cv2.IMREAD_UNCHANGED)
        _, m_b1_cached = score_gmic_preprocessed_tensor(model, standard_normalize_gmic_uint16(cached_arr), device)
        _, m_batch_cached = pilot_preds[f"{ds_tag}:{item_id}"]
        raw_u16, _ = prepare_dicom_uint16_for_nyukat(dcm_p, slot.split("-")[0])
        online_proc, _ = preprocess_uint16_for_nyukat(raw_u16, slot, GMIC_INPUT_SIZE, gmic_mods, normalize=True)
        _, m_online_raw = score_gmic_preprocessed_tensor(model, online_proc, device)
        b_diff = abs(m_batch_cached - m_b1_cached)
        o_diff = abs(m_batch_cached - m_online_raw)
        batch_vs_b1_diffs.append(b_diff)
        cached_vs_raw_diffs.append(o_diff)
        parity_rows.append(
            {
                "dataset": ds_tag,
                "item_id": item_id,
                "view_slot": slot,
                "malignant_cached_b8": round(m_batch_cached, 6),
                "malignant_cached_b1": round(m_b1_cached, 6),
                "malignant_online_raw": round(m_online_raw, 6),
                "abs_diff_vs_raw": round(o_diff, 7),
            }
        )

    n_pilot = len(pilot_items)
    infer_e2e_sec = nvme_stage_sec + decode_sec + gpu_sec
    pilot_imgs_per_sec = n_pilot / max(infer_e2e_sec, 1e-6)
    pilot_gpu_only_ips = n_pilot / max(gpu_sec, 1e-6)
    total_wall_sec = time.perf_counter() - t_total0
    # Marginal inference cost + full projected cost (400-img runtime + 1 model load overhead per shard across 4 shards)
    pilot_infer_cost_usd = (infer_e2e_sec / 3600.0) * L40S_COMBINED_HOURLY_USD
    pilot_total_container_cost_usd = (total_wall_sec / 3600.0) * L40S_COMBINED_HOURLY_USD

    full_target_imgs = 10982 + 26988
    proj_full_infer_sec = full_target_imgs / max(pilot_imgs_per_sec, 1e-6)
    proj_full_4shard_sec = proj_full_infer_sec + (4 * model_load_sec)
    proj_full_usd = (proj_full_4shard_sec / 3600.0) * L40S_COMBINED_HOURLY_USD

    mal_scores = np.array([v[1] for v in pilot_preds.values()], dtype=np.float64)
    return {
        "modal_profile": "hasinishrak2015",
        "gpu_name": torch.cuda.get_device_name(0),
        "checkpoint_sha256": got_sha,
        "checkpoint_strict_verified": True,
        "pilot_counts": {"rsna": len(rsna_pilot_ids), "nlbs": len(nlbs_pilot_ids), "total": n_pilot, "batch_size": batch_size},
        "all_scores_finite": bool(np.all(np.isfinite(mal_scores))),
        "score_summary_no_labels": {
            "min": round(float(np.min(mal_scores)), 6),
            "median": round(float(np.median(mal_scores)), 6),
            "max": round(float(np.max(mal_scores)), 6),
        },
        "sample_data_readme_parity_l40s": sample_parity,
        "cached_vs_raw_parity_10imgs": {
            "max_batch8_vs_b1_diff": float(max(batch_vs_b1_diffs)),
            "max_cached_vs_online_raw_diff": float(max(cached_vs_raw_diffs)),
            "samples": parity_rows,
        },
        "fallback_audit_all_80_rsna": fallback_audit,
        "supplemental_view_audit_all_5_rsna": supplemental_view_audit,
        "timing_sec": {
            "model_load_sec": round(model_load_sec, 3),
            "nvme_stage_sec": round(nvme_stage_sec, 3),
            "decode_norm_sec": round(decode_sec, 3),
            "gpu_infer_sec": round(gpu_sec, 3),
            "infer_e2e_sec": round(infer_e2e_sec, 3),
            "total_container_sec": round(total_wall_sec, 3),
        },
        "throughput": {
            "e2e_imgs_per_sec": round(pilot_imgs_per_sec, 2),
            "gpu_only_imgs_per_sec": round(pilot_gpu_only_ips, 2),
        },
        "memory": {
            "peak_vram_gb": round(peak_vram_gb, 3),
            "host_rss_gb": round(host_rss_gb, 3),
        },
        "cost": {
            "l40s_hourly_usd": L40S_COMBINED_HOURLY_USD,
            "measured_pilot_infer_only_usd": round(pilot_infer_cost_usd, 5),
            "measured_pilot_total_container_usd": round(pilot_total_container_cost_usd, 5),
            "projected_full_37970_hours": round(proj_full_4shard_sec / 3600.0, 4),
            "projected_full_37970_usd": round(proj_full_usd, 3),
            "within_2usd_budget": bool(proj_full_usd <= 2.00),
        },
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    ephemeral_disk=524288,
    timeout=7200,
)
def infer_gmic_shard(split_name: str, shard_idx: int, num_shards: int, batch_size: int = 1) -> Dict[str, Any]:
    """Run resumable GMIC sample_model_1.p inference using NVMe-staged Step-1 (2944x1920, uint16) PNGs."""
    import cv2
    import numpy as np
    import pandas as pd
    import resource
    import torch
    from src.evaluation.stage_c_eval import (
        assert_no_forbidden_columns,
        compute_sha256_file,
        load_label_free_inference_manifest,
    )
    from src.paper1.gmic_adapter import (
        GMIC_INPUT_SIZE,
        build_gmic_model,
        compute_item_ids_sha256,
        standard_normalize_gmic_uint16,
        validate_gmic_shard_payload,
    )

    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    batch_size = int(batch_size)
    RSNA_VOL.reload()
    t_container0 = time.perf_counter()

    ckpt_path = RSNA_MOUNT / "models" / "paper1" / "sample_model_1.p"
    got_sha = compute_sha256_file(ckpt_path).lower()
    if got_sha != GMIC_CKPT_SHA256:
        raise RuntimeError(f"GMIC checkpoint SHA mismatch: {got_sha} != {GMIC_CKPT_SHA256}")

    cache_root = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / split_name
    vol_png_dir = cache_root / "png_2944x1920"
    manifest_csv = cache_root / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Step 1 GMIC manifest missing: {manifest_csv}")
    step1_manifest_sha = compute_sha256_file(manifest_csv)
    step1_df = pd.read_csv(manifest_csv, dtype={"item_id": str, "case_id": str})
    step1_map = step1_df.set_index("item_id")[["crop_fallback", "crop_error", "fallback_strategy", "photometric"]].to_dict("index")

    if split_name == "rsna_val_calib":
        df = pd.read_csv(
            "/root/datasets/rsna/splits_seed42.csv",
            usecols=["image_id", "patient_id", "split", "laterality", "view"],
        )
        assert_no_forbidden_columns(df.columns, context="infer_gmic_shard_rsna_val_calib")
        df = df[df["split"] == "val_calib"].sort_values(["patient_id", "image_id"]).reset_index(drop=True)
        rows = [
            {"item_id": str(r.image_id), "case_id": str(r.patient_id), "view_slot": f"{r.laterality}-{r.view}"}
            for r in df.iloc[shard_idx::num_shards].itertuples(index=False)
        ]
    elif split_name == "nlbs":
        df = load_label_free_inference_manifest(Path("/root/datasets/nlbs/inference_manifest_stage_c.csv"))
        rows = [
            {"item_id": str(r.image_uid), "case_id": str(r.StudyInstanceUID), "view_slot": f"{r.laterality}-{r.view_position}"}
            for r in df.iloc[shard_idx::num_shards].itertuples(index=False)
        ]
    else:
        raise ValueError(f"Unsupported split_name={split_name}")

    expected_item_ids = [r["item_id"] for r in rows]
    expected_item_ids_sha = compute_item_ids_sha256(expected_item_ids)

    out_dir = RSNA_MOUNT / "cache" / "paper1_gmic_shards" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_file = out_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}_bs{batch_size}.json"
    # Validate requested file AND reject any incompatible payload even if a legacy file path existed
    if shard_file.exists():
        try:
            cached = json.loads(shard_file.read_text(encoding="utf-8"))
        except Exception as exc:
            cached = {}
            print(f"[GMIC {split_name} shard {shard_idx}] Corrupt cached shard ignored ({exc}); recomputing.", flush=True)
        ok, reason = validate_gmic_shard_payload(
            cached,
            expected_item_ids=expected_item_ids,
            batch_size=batch_size,
            split_name=split_name,
            shard_idx=shard_idx,
            num_shards=num_shards,
            checkpoint_sha256=GMIC_CKPT_SHA256,
            preprocess_version=PREPROCESS_VERSION,
            preprocess_manifest_sha256=step1_manifest_sha,
        )
        if ok:
            cached["resumed_from_cache"] = True
            print(f"[GMIC {split_name} shard {shard_idx}] Resumed validated cache ({len(expected_item_ids)} imgs, bs={batch_size}).", flush=True)
            return cached
        print(f"[GMIC {split_name} shard {shard_idx}] Incompatible cached shard ({reason}); recomputing cleanly.", flush=True)

    nvme_dir = Path(f"/tmp/gmic_{split_name}_s{shard_idx:02d}")
    nvme_stage_sec = _stage_pngs_nvme(expected_item_ids, vol_png_dir, nvme_dir, max_workers=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L40S GPU expected for GMIC shard inference!")
    t_load0 = time.perf_counter()
    model, _ = build_gmic_model(
        ckpt_path, model_index="1", device=device, gmic_repo_dir=Path("/root/external/GMIC")
    )
    model_load_sec = time.perf_counter() - t_load0

    def _load_norm(row: Dict[str, str]) -> np.ndarray:
        arr = cv2.imread(str(nvme_dir / f"{row['item_id']}.png"), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.shape != GMIC_INPUT_SIZE or arr.dtype != np.uint16:
            raise RuntimeError(f"Invalid Step 1 cached PNG: {row['item_id']}")
        return standard_normalize_gmic_uint16(arr)

    from collections import deque

    torch.cuda.reset_peak_memory_stats()
    t_loop0 = time.perf_counter()
    decode_norm_sec = 0.0
    gpu_infer_sec = 0.0
    records = []
    fallback_logs = []
    prefetch_limit = max(24, batch_size * 4)
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = deque()
        next_submit = 0
        while next_submit < len(rows) and len(pending) < prefetch_limit:
            pending.append((rows[next_submit], pool.submit(_load_norm, rows[next_submit])))
            next_submit += 1

        for s in range(0, len(rows), batch_size):
            b_rows = rows[s : s + batch_size]
            t_d0 = time.perf_counter()
            norms = [pending.popleft()[1].result() for _ in b_rows]
            while next_submit < len(rows) and len(pending) < prefetch_limit:
                pending.append((rows[next_submit], pool.submit(_load_norm, rows[next_submit])))
                next_submit += 1
            if batch_size == 1:
                bx = torch.from_numpy(norms[0][np.newaxis, np.newaxis, :, :]).to(device, non_blocking=True)
            else:
                bx = torch.from_numpy(np.stack(norms, axis=0)[:, np.newaxis, :, :]).to(device, non_blocking=True)
            decode_norm_sec += time.perf_counter() - t_d0

            torch.cuda.synchronize()
            t_g0 = time.perf_counter()
            with torch.no_grad():
                outs = model(bx).float().cpu().numpy()
            torch.cuda.synchronize()
            gpu_infer_sec += time.perf_counter() - t_g0

            if not np.all(np.isfinite(outs)):
                raise RuntimeError(f"Non-finite GMIC output in {split_name} shard {shard_idx} batch {s}")

            for row, pred in zip(b_rows, outs):
                fb_meta = step1_map[row["item_id"]]
                is_fb = bool(fb_meta["crop_fallback"])
                if is_fb:
                    fallback_logs.append(
                        {
                            "item_id": row["item_id"],
                            "case_id": row["case_id"],
                            "view_slot": row["view_slot"],
                            "photometric": str(fb_meta.get("photometric", "")),
                            "fallback_strategy": str(fb_meta.get("fallback_strategy", "")),
                            "crop_error": str(fb_meta.get("crop_error", "")),
                        }
                    )
                records.append(
                    {
                        "item_id": row["item_id"],
                        "case_id": row["case_id"],
                        "view_slot": row["view_slot"],
                        "benign_pred": float(pred[0]),
                        "malignant_pred": float(pred[1]),
                        "crop_fallback": is_fb,
                    }
                )
            done = min(s + batch_size, len(rows))
            if done % 500 == 0 or done == len(rows):
                print(f"[GMIC {split_name} shard {shard_idx}] {done}/{len(rows)} (bs={batch_size})", flush=True)

    elapsed_sec = time.perf_counter() - t_loop0
    container_wall_sec = time.perf_counter() - t_container0
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    host_rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))
    estimated_cost_usd = (container_wall_sec / 3600.0) * L40S_COMBINED_HOURLY_USD

    payload = {
        "schema_version": "paper1_gmic_shard_v2",
        "model_id": "gmic_pretrained",
        "checkpoint_file": "sample_model_1.p",
        "checkpoint_sha256": GMIC_CKPT_SHA256,
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_manifest_sha256": step1_manifest_sha,
        "split_name": split_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "batch_size": batch_size,
        "expected_n_images": len(expected_item_ids),
        "expected_item_ids_sha256": expected_item_ids_sha,
        "model_load_sec": round(model_load_sec, 3),
        "nvme_stage_sec": round(nvme_stage_sec, 3),
        "decode_norm_sec": round(decode_norm_sec, 3),
        "gpu_infer_sec": round(gpu_infer_sec, 3),
        "elapsed_sec": round(elapsed_sec, 3),
        "container_wall_sec": round(container_wall_sec, 3),
        "l40s_hourly_usd": L40S_COMBINED_HOURLY_USD,
        "estimated_cost_usd": round(estimated_cost_usd, 5),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "host_rss_gb": round(host_rss_gb, 3),
        "n_fallback_views": len(fallback_logs),
        "fallback_logs": fallback_logs,
        "records": records,
    }
    ok, reason = validate_gmic_shard_payload(
        payload,
        expected_item_ids=expected_item_ids,
        batch_size=batch_size,
        split_name=split_name,
        shard_idx=shard_idx,
        num_shards=num_shards,
        checkpoint_sha256=GMIC_CKPT_SHA256,
        preprocess_version=PREPROCESS_VERSION,
        preprocess_manifest_sha256=step1_manifest_sha,
    )
    if not ok:
        raise RuntimeError(f"Generated shard failed self-validation: {reason}")

    tmp = shard_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(shard_file)
    RSNA_VOL.commit()
    payload["resumed_from_cache"] = False
    return payload


@app.local_entrypoint()
def main(
    stage: str = "pilot",
    split: str = "rsna_val_calib",
    num_shards: int = 4,
    batch_size: int = 1,
    shard_indices: str = "",
) -> None:
    import pandas as pd
    from src.evaluation.stage_c_eval import assert_no_forbidden_columns, load_label_free_inference_manifest
    from src.paper1.gmic_adapter import finalize_and_lock_gmic_predictions

    if batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")

    if stage == "pilot":
        res = run_gmic_step2_pilot.remote(n_per_split=200, batch_size=batch_size)
        out_p = Path("reports/paper1/gmic_step2_gpu_pilot.json")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return

    if stage == "infer":
        if split == "rsna_val_calib":
            exp_df = pd.read_csv(
                "datasets/rsna/splits_seed42.csv",
                usecols=["image_id", "patient_id", "split", "laterality", "view"],
            )
            assert_no_forbidden_columns(exp_df.columns, context="local_infer_rsna_val_calib")
            exp_df = exp_df[exp_df["split"] == "val_calib"].reset_index(drop=True)
            item_col, case_col, case_out_col = "image_id", "patient_id", "patient_id"
        elif split == "nlbs":
            exp_df = load_label_free_inference_manifest(Path("datasets/nlbs/inference_manifest_stage_c.csv"))
            item_col, case_col, case_out_col = "image_uid", "StudyInstanceUID", "StudyInstanceUID"
        else:
            raise ValueError(f"Unsupported split={split!r}; use rsna_val_calib or nlbs")

        shards = (
            [int(x.strip()) for x in shard_indices.split(",") if x.strip()]
            if shard_indices.strip()
            else list(range(num_shards))
        )
        outs = list(infer_gmic_shard.starmap([(split, s, num_shards, batch_size) for s in shards]))
        for o in sorted(outs, key=lambda x: int(x["shard_idx"])):
            print(
                f"[Shard {o['shard_idx']}/{o['num_shards']}] n={len(o['records'])} "
                f"bs={o['batch_size']} wall={o['container_wall_sec']}s "
                f"decode={o['decode_norm_sec']}s gpu={o['gpu_infer_sec']}s "
                f"est_cost=${o['estimated_cost_usd']:.5f} resumed={o.get('resumed_from_cache', False)}"
            )

        if len(shards) < num_shards or set(shards) != set(range(num_shards)):
            print(f"[Partial shard run {shards}/{num_shards}] Skipping final split lock until all shards complete.")
            return

        out_dir = Path("reports/paper1/predictions_locked")
        lock_manifest = finalize_and_lock_gmic_predictions(
            split_name=split,
            shard_outputs=outs,
            expected_image_df=exp_df,
            item_col=item_col,
            case_col=case_col,
            case_out_col=case_out_col,
            batch_size=batch_size,
            checkpoint_sha256=GMIC_CKPT_SHA256,
            preprocess_version=PREPROCESS_VERSION,
            out_dir=out_dir,
        )
        print(json.dumps(lock_manifest, indent=2))
        return

    raise ValueError(f"Unknown stage={stage!r}; use pilot or infer")
