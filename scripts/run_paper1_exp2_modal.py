"""Paper 1 Experiment 2 — Modal Runner, Progressive Pilot Benchmarker & Cache Builder.

Targets Modal profile `tigerwood693` explicitly (NEVER calls `modal profile activate`).
Enforces Section 6 locked shortlist (`glam_pretrained`, `end2end_cbis_ddsm`, `petrini_twoview_efnb3`)
while hard-blocking disqualified/rejected candidates and locking Category C (AUROC/AUPRC).

Stages:
  - `plan` (CPU-local, $0.00): dry-run work-item counts & progressive pilot batch schedule.
  - `build_uncropped_cache` (CPU Modal, requires `--approve-wave2-download`): Option B1 stream-decode
    of read-only `hasinishrak2015:nlbs-mammography-vol:/nlbs/extracted` DICOMs into uncropped
    `(1152, 896)` uint16 PNGs on `tigerwood693:/rsna_vol/cache/paper1_exp2_uncropped_1152x896/nlbs`.
  - `pilot` (L40S Modal, requires `--approve-pilot`): progressive batch-size benchmark (`200-500` units)
    measuring VRAM allocated/reserved, `imgs/sec`, `exams/sec`, data load vs GPU time, parity, and cost.
  - `infer` (L40S Modal, requires `--approve-full-infer`): full resumable sharded NLBS inference + lock.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paper1.exp2_adapters import (
    EXP2_MODEL_REGISTRY,
    assert_model_eligible,
    build_petrini_breast_pairs,
    compute_unit_ids_sha256,
    default_pilot_batch_sizes,
    finalize_and_lock_exp2_predictions,
    load_exp2_nlbs_manifest,
    validate_exp2_shard_payload,
)
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION
from src.utils.modal_transfer import load_modal_profile_credentials

APP_NAME = "paper1-exp2-pretrained-inference"
TARGET_MODAL_PROFILE = "tigerwood693"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")
L40S_COMBINED_HOURLY_USD = 2.3618  # 1x L40S ($1.9512) + 6 vCPU ($0.2838) + 16 GiB RAM ($0.1280)
CPU_CACHE_WORKER_HOURLY_USD = 0.1266  # 2 vCPU ($0.0946) + 4 GiB RAM ($0.0320)


def _build_hasin2015_readonly_secret() -> modal.Secret:
    if not modal.is_local():
        return modal.Secret.from_dict({})
    if os.environ.get("EXP2_ATTACH_HASIN2015_SECRET", "") != "1":
        return modal.Secret.from_dict({})
    creds = load_modal_profile_credentials(["hasinishrak2015"])
    tid, tsec = creds["hasinishrak2015"]
    return modal.Secret.from_dict(
        {"SRC_TOKEN_ID_HASINISHRAK2015": tid, "SRC_TOKEN_SECRET_HASINISHRAK2015": tsec}
    )


SRC_2015_SECRET = _build_hasin2015_readonly_secret()
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        *PINNED_MODAL_PIP_PACKAGES,
        "h5py==3.13.0",
        "imageio==2.37.0",
        "tqdm==4.67.1",
    )
    .add_local_python_source("src")
    .add_local_file(
        "datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
        remote_path="/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
    )
    .add_local_file(
        "datasets/nlbs/inference_manifest_stage_c.csv",
        remote_path="/root/datasets/nlbs/inference_manifest_stage_c.csv",
    )
)


def parse_pilot_batch_sizes(raw: str, model_id: str) -> List[int]:
    """Parse and validate progressive batch-size list for `--stage pilot`."""
    if not raw.strip():
        return default_pilot_batch_sizes(model_id)
    sizes = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not sizes or any(s < 1 for s in sizes):
        raise ValueError(f"Invalid --pilot-batch-sizes={raw!r}; all sizes must be >= 1")
    return sizes


def validate_exp2_cli_and_guard(
    *,
    stage: str,
    model_id: str,
    split: str,
    num_shards: int,
    batch_size: int,
    pilot_limit: int,
    cache_mode: str,
    approve_wave2_download: bool,
    approve_pilot: bool,
    approve_full_infer: bool,
    max_total_spend_usd: float,
) -> None:
    """Validate CLI parameters and block unapproved Wave 2 cache builds, Wave 3 pilots, or Phase 2 runs."""
    valid_stages = ("plan", "build_uncropped_cache", "pilot", "infer")
    if stage not in valid_stages:
        raise ValueError(f"Unknown --stage={stage!r}; expected {valid_stages}")
    spec = assert_model_eligible(model_id)
    if spec.eligibility == "eligible_locked_reuse":
        raise RuntimeError(
            f"{model_id} already has locked NLBS predictions (reuse reports/paper1/predictions_locked/ at $0.00)."
        )
    if split not in ("nlbs_4view", "nlbs_full"):
        raise ValueError(f"Unsupported --split={split!r}; expected 'nlbs_4view' or 'nlbs_full'")
    if cache_mode not in ("auto", "gmic_2944x1920", "uncropped_1152x896", "stage1_1520x912"):
        raise ValueError(f"Unsupported --cache-mode={cache_mode!r}")
    if int(num_shards) < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if int(batch_size) < 1:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
    if not (200 <= int(pilot_limit) <= 500):
        raise ValueError(f"--pilot-limit must be in [200, 500], got {pilot_limit}")
    if float(max_total_spend_usd) <= 0.0:
        raise ValueError(f"--max-total-spend-usd must be > 0, got {max_total_spend_usd}")

    if stage == "build_uncropped_cache" and not approve_wave2_download:
        raise RuntimeError(
            "BLOCKED: `--stage build_uncropped_cache` requires explicit `--approve-wave2-download`."
        )
    if stage == "pilot" and not approve_pilot:
        raise RuntimeError(
            "BLOCKED: `--stage pilot` requires explicit `--approve-pilot` confirmation (Wave 3 L40S spend)."
        )
    if stage == "infer" and not approve_full_infer:
        raise RuntimeError(
            "BLOCKED: `--stage infer` requires explicit `--approve-full-infer` confirmation (Phase 2 full L40S spend)."
        )


def _resolve_cache_dir(model_id: str, cache_mode: str) -> Path:
    mode = EXP2_MODEL_REGISTRY[model_id].nlbs_cache_source if cache_mode == "auto" else cache_mode
    if mode == "gmic_2944x1920":
        return RSNA_MOUNT / "cache" / "paper1_gmic_uint16_v1" / "nlbs" / "png_2944x1920"
    if mode == "uncropped_1152x896":
        return RSNA_MOUNT / "cache" / "paper1_exp2_uncropped_1152x896" / "nlbs"
    if mode == "stage1_1520x912":
        return RSNA_MOUNT / "cache" / "stage1_nlbs_uint16"
    raise RuntimeError(f"Invalid cache_mode={mode!r} for model_id={model_id!r}")


def _stage_pngs_to_nvme(file_ids: List[str], src_dir: Path, nvme_dir: Path, max_workers: int = 16) -> float:
    nvme_dir.mkdir(parents=True, exist_ok=True)
    missing = [fid for fid in file_ids if not (nvme_dir / f"{fid}.png").exists()]
    t0 = time.perf_counter()
    if missing:

        def _cp(fid: str) -> None:
            src_p = src_dir / f"{fid}.png"
            dst_p = nvme_dir / f"{fid}.png"
            if not src_p.exists() or src_p.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty cached PNG: {src_p}")
            shutil.copy2(src_p, dst_p)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_cp, missing))
    return time.perf_counter() - t0


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=3600,
    secrets=[SRC_2015_SECRET],
    volumes={str(RSNA_MOUNT): RSNA_VOL},
)
def build_uncropped_1152x896_cache_shard(
    shard_idx: int,
    num_shards: int,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Option B1 CPU worker: stream raw NLBS DICOMs read-only from hasinishrak2015 -> uncropped 1152x896 PNGs."""
    import asyncio
    import cv2
    from src.data.preprocessing import decode_dicom_array
    from src.paper1.exp2_adapters import prepare_uncropped_1152x896_from_array
    from src.paper1.gmic_adapter import save_uint16_png_atomic

    tid = os.environ.get("SRC_TOKEN_ID_HASINISHRAK2015", "").strip()
    tsec = os.environ.get("SRC_TOKEN_SECRET_HASINISHRAK2015", "").strip()
    if not tid or not tsec:
        raise RuntimeError("Missing SRC_TOKEN_*_HASINISHRAK2015 for Option B1 streamed cache build.")

    out_dir = RSNA_MOUNT / "cache" / "paper1_exp2_uncropped_1152x896" / "nlbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dcm = Path("/tmp/exp2_dcm_buf")
    tmp_dcm.mkdir(parents=True, exist_ok=True)

    async def _run_shard() -> Dict[str, Any]:
        client = await modal.Client.from_credentials.aio(tid, tsec)
        src_vol = modal.Volume.from_name("nlbs-mammography-vol", client=client)
        written = 0
        skipped = 0
        total_bytes = 0
        t0 = time.perf_counter()
        last_commit = t0
        for row in rows:
            uid = str(row["image_uid"])
            dst_png = out_dir / f"{uid}.png"
            if dst_png.exists() and dst_png.stat().st_size > 0:
                skipped += 1
                total_bytes += dst_png.stat().st_size
                continue
            rel = str(row["dicom_rel_path"]).lstrip("/")
            src_path = rel if rel.startswith("nlbs/extracted/") else f"nlbs/extracted/{rel}"
            local_dcm = tmp_dcm / f"{uid}.dcm"
            with open(local_dcm, "wb") as f:
                async for chunk in src_vol.read_file.aio(src_path):
                    f.write(chunk)
            arr_f32, phot, lat_dcm = decode_dicom_array(local_dcm, apply_luts=True)
            local_dcm.unlink(missing_ok=True)
            u16 = prepare_uncropped_1152x896_from_array(
                arr_f32, photometric=phot, laterality=lat_dcm or str(row["laterality"])
            )
            sz = save_uint16_png_atomic(u16, dst_png, compression=1)
            written += 1
            total_bytes += sz
            if time.perf_counter() - last_commit >= 60.0:
                await RSNA_VOL.commit.aio()
                last_commit = time.perf_counter()
        await RSNA_VOL.commit.aio()
        elapsed = time.perf_counter() - t0
        return {
            "shard_idx": shard_idx,
            "num_shards": num_shards,
            "written": written,
            "skipped": skipped,
            "total_gib": round(total_bytes / (1024.0**3), 4),
            "elapsed_sec": round(elapsed, 3),
            "estimated_cost_usd": round((elapsed / 3600.0) * CPU_CACHE_WORKER_HOURLY_USD, 5),
        }

    return asyncio.run(_run_shard())


def _load_model_for_exp2(model_id: str, checkpoint_rel_path: str, device: torch.device) -> Any:
    import torch
    from src.paper1.exp2_adapters import build_glam_model, build_petrini_twoview_model, verify_checkpoint_sha256

    ckpt_path = RSNA_MOUNT / checkpoint_rel_path
    spec = assert_model_eligible(model_id)
    if model_id == "glam_pretrained":
        model, ckpt_sha = build_glam_model(
            ckpt_path,
            glam_repo_dir=RSNA_MOUNT / "external_repos" / "GLAM",
            device=device,
            expected_sha256=spec.checkpoint_sha256,
        )
        return model, ckpt_sha
    if model_id == "petrini_twoview_efnb3":
        model, ckpt_sha = build_petrini_twoview_model(
            ckpt_path,
            petrini_repo_dir=RSNA_MOUNT / "external_repos" / "multiple-view",
            device=device,
        )
        return model, ckpt_sha
    if model_id == "end2end_cbis_ddsm":
        ckpt_sha = verify_checkpoint_sha256(ckpt_path, spec.checkpoint_sha256)
        model = torch.jit.load(str(ckpt_path), map_location=device)
        model.eval()
        return model, ckpt_sha
    raise RuntimeError(f"Unsupported model_id={model_id!r}")


def _run_batch_forward(
    model_id: str,
    model: Any,
    chunk: List[Dict[str, Any]],
    nvme_dir: Path,
    device: Any,
) -> Tuple[List[float], float, float]:
    import cv2
    import numpy as np
    import torch
    from src.paper1.exp2_adapters import (
        build_petrini_twoview_tensor,
        preprocess_end2end_from_uint16,
        preprocess_glam_from_gmic_uint16,
        score_twoview_batch,
    )

    t_load0 = time.perf_counter()
    if model_id == "glam_pretrained":
        imgs = []
        for r in chunk:
            arr = cv2.imread(str(nvme_dir / f"{r['image_uids'][0]}.png"), cv2.IMREAD_UNCHANGED)
            imgs.append(preprocess_glam_from_gmic_uint16(arr)[np.newaxis, ...])
        batch_t = torch.from_numpy(np.stack(imgs, axis=0).copy())
    elif model_id == "end2end_cbis_ddsm":
        imgs = []
        for r in chunk:
            arr = cv2.imread(str(nvme_dir / f"{r['image_uids'][0]}.png"), cv2.IMREAD_UNCHANGED)
            imgs.append(preprocess_end2end_from_uint16(arr))
        batch_t = torch.from_numpy(np.stack(imgs, axis=0).copy())
    elif model_id == "petrini_twoview_efnb3":
        tensors = []
        for r in chunk:
            cc_arr = cv2.imread(str(nvme_dir / f"{r['image_uids'][0]}.png"), cv2.IMREAD_UNCHANGED)
            mlo_arr = cv2.imread(str(nvme_dir / f"{r['image_uids'][1]}.png"), cv2.IMREAD_UNCHANGED)
            tensors.append(build_petrini_twoview_tensor(cc_arr, mlo_arr, dataset="CBIS-DDSM"))
        batch_t = torch.stack(tensors, dim=0)
    else:
        raise RuntimeError(f"Unsupported model_id={model_id!r}")
    data_load_sec = time.perf_counter() - t_load0

    torch.cuda.synchronize(device)
    t_gpu0 = time.perf_counter()
    if model_id == "glam_pretrained":
        with torch.no_grad():
            probs = model(batch_t.to(device, non_blocking=True)).float()[:, 1].cpu().numpy()
    elif model_id == "end2end_cbis_ddsm":
        with torch.no_grad():
            out = model(batch_t.to(device, non_blocking=True)).float()
            probs = (torch.softmax(out, dim=-1)[:, 1] if out.shape[-1] == 2 else out.reshape(-1)).cpu().numpy()
    else:
        probs = score_twoview_batch(model, batch_t, device)
    torch.cuda.synchronize(device)
    gpu_sec = time.perf_counter() - t_gpu0
    return [float(p) for p in probs], data_load_sec, gpu_sec


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=3600,
)
def run_exp2_pilot_benchmark_l40s(
    *,
    model_id: str,
    split_name: str,
    pilot_units: List[Dict[str, Any]],
    batch_sizes: List[int],
    checkpoint_rel_path: str,
    cache_mode: str,
    full_total_units: int,
    full_total_studies: int,
) -> Dict[str, Any]:
    """Wave 3 progressive batch-size L40S pilot measuring VRAM, imgs/s, exams/s, data vs GPU time."""
    import numpy as np
    import torch

    RSNA_VOL.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L40S CUDA device required for Wave 3 pilot benchmark.")

    t_load0 = time.perf_counter()
    model, ckpt_sha = _load_model_for_exp2(model_id, checkpoint_rel_path, device)
    model_load_sec = time.perf_counter() - t_load0

    src_png_dir = _resolve_cache_dir(model_id, cache_mode)
    nvme_dir = Path("/tmp/exp2_pilot_nvme") / model_id / split_name
    needed_ids = sorted({str(uid) for r in pilot_units for uid in r["image_uids"]})
    nvme_stage_sec = _stage_pngs_to_nvme(needed_ids, src_png_dir, nvme_dir)

    imgs_per_unit = 2 if model_id == "petrini_twoview_efnb3" else 1
    units_per_exam = float(full_total_units) / max(1, int(full_total_studies))
    ref_scores: Optional[List[float]] = None
    sweep_results: List[Dict[str, Any]] = []

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        data_sec_sum = 0.0
        gpu_sec_sum = 0.0
        scores: List[float] = []
        oom_hit = False
        try:
            for i in range(0, len(pilot_units), bs):
                chunk = pilot_units[i : i + bs]
                probs, dl_s, g_s = _run_batch_forward(model_id, model, chunk, nvme_dir, device)
                data_sec_sum += dl_s
                gpu_sec_sum += g_s
                scores.extend(probs)
        except torch.cuda.OutOfMemoryError:
            oom_hit = True
            torch.cuda.empty_cache()

        if oom_hit:
            sweep_results.append({"batch_size": bs, "status": "OOM"})
            break

        total_infer_s = data_sec_sum + gpu_sec_sum
        n_imgs = len(pilot_units) * imgs_per_unit
        units_per_sec = len(pilot_units) / max(1e-6, total_infer_s)
        imgs_per_sec = n_imgs / max(1e-6, total_infer_s)
        exams_per_sec = units_per_sec / max(1e-6, units_per_exam)
        proj_full_s = float(full_total_units) / max(1e-6, units_per_sec)
        proj_full_usd = (proj_full_s / 3600.0) * L40S_COMBINED_HOURLY_USD
        if ref_scores is None:
            ref_scores = scores
        max_diff_vs_ref = float(np.max(np.abs(np.asarray(scores) - np.asarray(ref_scores))))

        sweep_results.append(
            {
                "batch_size": bs,
                "status": "OK",
                "n_units": len(pilot_units),
                "n_images": n_imgs,
                "data_load_sec": round(data_sec_sum, 3),
                "gpu_infer_sec": round(gpu_sec_sum, 3),
                "total_infer_sec": round(total_infer_s, 3),
                "imgs_per_sec": round(imgs_per_sec, 3),
                "exams_per_sec": round(exams_per_sec, 3),
                "vram_allocated_gb": round(float(torch.cuda.max_memory_allocated(device) / (1024**3)), 4),
                "vram_reserved_gb": round(float(torch.cuda.max_memory_reserved(device) / (1024**3)), 4),
                "max_abs_diff_vs_smallest_bs": round(max_diff_vs_ref, 7),
                "projected_full_nlbs_min": round(proj_full_s / 60.0, 2),
                "projected_full_nlbs_usd": round(proj_full_usd, 4),
            }
        )

    ok_runs = [r for r in sweep_results if r.get("status") == "OK" and float(r.get("vram_reserved_gb", 99.0)) < 42.0]
    best = max(ok_runs, key=lambda r: float(r["imgs_per_sec"])) if ok_runs else None
    total_wall = model_load_sec + nvme_stage_sec + sum(float(r.get("total_infer_sec", 0.0)) for r in ok_runs)
    return {
        "schema_version": "paper1_exp2_pilot_bench_v1",
        "model_id": model_id,
        "checkpoint_sha256": ckpt_sha,
        "split_name": split_name,
        "cache_mode": cache_mode,
        "model_load_sec": round(model_load_sec, 3),
        "nvme_stage_sec": round(nvme_stage_sec, 3),
        "pilot_wall_sec": round(total_wall, 3),
        "pilot_estimated_cost_usd": round((total_wall / 3600.0) * L40S_COMBINED_HOURLY_USD, 5),
        "recommended_batch_size": best["batch_size"] if best else None,
        "sweep_results": sweep_results,
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=3600,
)
def run_exp2_shard_l40s(
    *,
    model_id: str,
    split_name: str,
    shard_idx: int,
    num_shards: int,
    batch_size: int,
    unit_records: List[Dict[str, Any]],
    checkpoint_rel_path: str,
    cache_mode: str,
) -> Dict[str, Any]:
    """Execute resumable full L40S inference shard for an eligible Exp 2 model."""
    import torch

    RSNA_VOL.reload()
    t0 = time.perf_counter()
    spec = assert_model_eligible(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L40S CUDA device required for Modal GPU shard.")

    unit_ids = [str(r["unit_id"]) for r in unit_records]
    exp_ids_sha = compute_unit_ids_sha256(unit_ids)
    shard_dir = RSNA_MOUNT / "cache" / "paper1_exp2_shards" / model_id / split_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_json = shard_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}.json"

    if shard_json.exists():
        cached = json.loads(shard_json.read_text(encoding="utf-8"))
        ok, reason = validate_exp2_shard_payload(
            cached,
            model_id=model_id,
            expected_unit_ids=unit_ids,
            split_name=split_name,
            shard_idx=shard_idx,
            num_shards=num_shards,
            checkpoint_sha256=cached.get("checkpoint_sha256", spec.checkpoint_sha256),
        )
        if ok:
            cached["resumed_from_cache"] = True
            return cached
        raise RuntimeError(f"Corrupt/incompatible shard cache {shard_json}: {reason}")

    model, ckpt_sha = _load_model_for_exp2(model_id, checkpoint_rel_path, device)
    src_png_dir = _resolve_cache_dir(model_id, cache_mode)
    nvme_dir = Path("/tmp/exp2_nvme") / model_id / split_name
    needed_ids = sorted({str(uid) for r in unit_records for uid in r["image_uids"]})
    nvme_sec = _stage_pngs_to_nvme(needed_ids, src_png_dir, nvme_dir)

    out_records: List[Dict[str, Any]] = []
    data_sec_sum = 0.0
    gpu_sec_sum = 0.0
    for i in range(0, len(unit_records), batch_size):
        chunk = unit_records[i : i + batch_size]
        probs, dl_s, g_s = _run_batch_forward(model_id, model, chunk, nvme_dir, device)
        data_sec_sum += dl_s
        gpu_sec_sum += g_s
        for r, p in zip(chunk, probs):
            rec = {
                "unit_id": str(r["unit_id"]),
                "StudyInstanceUID": str(r["StudyInstanceUID"]),
                "score": float(p),
            }
            if model_id == "petrini_twoview_efnb3":
                rec["laterality"] = str(r["laterality"])
                rec["cc_image_uid"] = str(r["image_uids"][0])
                rec["mlo_image_uid"] = str(r["image_uids"][1])
            else:
                rec["image_uid"] = str(r["image_uids"][0])
            out_records.append(rec)

    elapsed_sec = time.perf_counter() - t0
    est_cost = round((elapsed_sec / 3600.0) * L40S_COMBINED_HOURLY_USD, 5)
    payload = {
        "schema_version": "paper1_exp2_shard_v1",
        "model_id": model_id,
        "checkpoint_sha256": ckpt_sha,
        "split_name": split_name,
        "shard_idx": int(shard_idx),
        "num_shards": int(num_shards),
        "batch_size": int(batch_size),
        "expected_unit_ids_sha256": exp_ids_sha,
        "nvme_stage_sec": round(nvme_sec, 3),
        "data_load_sec": round(data_sec_sum, 3),
        "gpu_infer_sec": round(gpu_sec_sum, 3),
        "elapsed_sec": round(elapsed_sec, 3),
        "estimated_cost_usd": est_cost,
        "vram_allocated_gb": round(float(torch.cuda.max_memory_allocated(device) / (1024**3)), 4),
        "vram_reserved_gb": round(float(torch.cuda.max_memory_reserved(device) / (1024**3)), 4),
        "resumed_from_cache": False,
        "records": out_records,
    }
    shard_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    RSNA_VOL.commit()
    return payload


def build_unit_work_items(model_id: str, split_name: str, root: Path = Path(".")) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build strictly label-free unit work items (images or breast CC+MLO pairs) on local CPU."""
    manifest_rel = (
        "datasets/paper1/nlbs_4view_common_cohort_label_free.csv"
        if split_name == "nlbs_4view"
        else "datasets/nlbs/inference_manifest_stage_c.csv"
    )
    df = load_exp2_nlbs_manifest(root / manifest_rel, model_id=model_id)
    spec = assert_model_eligible(model_id)
    expected_studies = sorted(df["StudyInstanceUID"].unique().tolist())

    if spec.input_level == "image":
        items = [
            {
                "unit_id": str(row.image_uid),
                "StudyInstanceUID": str(row.StudyInstanceUID),
                "image_uids": [str(row.image_uid)],
            }
            for row in df.sort_values(["StudyInstanceUID", "image_uid"]).itertuples(index=False)
        ]
        return items, expected_studies

    pairs = build_petrini_breast_pairs(df)
    expected_studies = sorted(pairs["StudyInstanceUID"].unique().tolist())
    items = [
        {
            "unit_id": str(row.pair_id),
            "StudyInstanceUID": str(row.StudyInstanceUID),
            "laterality": str(row.laterality),
            "image_uids": [str(row.cc_image_uid), str(row.mlo_image_uid)],
        }
        for row in pairs.itertuples(index=False)
    ]
    return items, expected_studies


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 1 Exp 2 Modal Runner (tigerwood693)")
    parser.add_argument(
        "--stage",
        default="plan",
        choices=["plan", "build_uncropped_cache", "pilot", "infer"],
    )
    parser.add_argument("--model-id", required=True, choices=sorted(EXP2_MODEL_REGISTRY.keys()))
    parser.add_argument("--split", default="nlbs_4view", choices=["nlbs_4view", "nlbs_full"])
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pilot-limit", type=int, default=200)
    parser.add_argument("--pilot-batch-sizes", default="")
    parser.add_argument(
        "--cache-mode",
        default="auto",
        choices=["auto", "gmic_2944x1920", "uncropped_1152x896", "stage1_1520x912"],
    )
    parser.add_argument("--checkpoint-rel-path", default="")
    parser.add_argument("--max-total-spend-usd", type=float, default=5.0)
    parser.add_argument("--approve-wave2-download", action="store_true")
    parser.add_argument("--approve-pilot", action="store_true")
    parser.add_argument("--approve-full-infer", action="store_true")
    args = parser.parse_args()

    validate_exp2_cli_and_guard(
        stage=args.stage,
        model_id=args.model_id,
        split=args.split,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
        pilot_limit=args.pilot_limit,
        cache_mode=args.cache_mode,
        approve_wave2_download=args.approve_wave2_download,
        approve_pilot=args.approve_pilot,
        approve_full_infer=args.approve_full_infer,
        max_total_spend_usd=args.max_total_spend_usd,
    )

    items, expected_studies = build_unit_work_items(args.model_id, args.split)
    pilot_bs_list = parse_pilot_batch_sizes(args.pilot_batch_sizes, args.model_id)
    if args.stage == "plan":
        print(
            json.dumps(
                {
                    "stage": "plan",
                    "target_modal_profile": TARGET_MODAL_PROFILE,
                    "model_id": args.model_id,
                    "split": args.split,
                    "cache_mode": (
                        EXP2_MODEL_REGISTRY[args.model_id].nlbs_cache_source
                        if args.cache_mode == "auto"
                        else args.cache_mode
                    ),
                    "n_units": len(items),
                    "n_studies": len(expected_studies),
                    "num_shards": args.num_shards,
                    "pilot_limit": args.pilot_limit,
                    "pilot_batch_sizes": pilot_bs_list,
                    "cloud_spend_usd": 0.0,
                },
                indent=2,
            )
        )
        return

    with app.run():
        if args.stage == "build_uncropped_cache":
            df = load_exp2_nlbs_manifest(Path("datasets/nlbs/inference_manifest_stage_c.csv"), model_id="end2end_cbis_ddsm")
            rows = df[["image_uid", "laterality", "dicom_rel_path"]].to_dict(orient="records")
            shards = [rows[i :: args.num_shards] for i in range(args.num_shards)]
            res = list(
                build_uncropped_1152x896_cache_shard.starmap(
                    [(idx, args.num_shards, shard) for idx, shard in enumerate(shards)]
                )
            )
            print(json.dumps(res, indent=2))
            return

        if args.stage == "pilot":
            pilot_items = items[: args.pilot_limit]
            res = run_exp2_pilot_benchmark_l40s.remote(
                model_id=args.model_id,
                split_name=args.split,
                pilot_units=pilot_items,
                batch_sizes=pilot_bs_list,
                checkpoint_rel_path=args.checkpoint_rel_path,
                cache_mode=args.cache_mode,
                full_total_units=len(items),
                full_total_studies=len(expected_studies),
            )
            pilot_out = Path("reports/paper1/expanded/exp2/pilots")
            pilot_out.mkdir(parents=True, exist_ok=True)
            (pilot_out / f"{args.model_id}_pilot_bench.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(json.dumps(res, indent=2))
            return

        shards = [items[i :: args.num_shards] for i in range(args.num_shards)]
        outputs: List[Dict[str, Any]] = []
        incurred = 0.0
        for s_idx, s_items in enumerate(shards):
            res = run_exp2_shard_l40s.remote(
                model_id=args.model_id,
                split_name=args.split,
                shard_idx=s_idx,
                num_shards=args.num_shards,
                batch_size=args.batch_size,
                unit_records=s_items,
                checkpoint_rel_path=args.checkpoint_rel_path,
                cache_mode=args.cache_mode,
            )
            if not res.get("resumed_from_cache", False):
                incurred += float(res.get("estimated_cost_usd") or 0.0)
            if incurred > args.max_total_spend_usd:
                raise RuntimeError(
                    f"SPEND CAP HIT: incurred ${incurred:.4f} > max ${args.max_total_spend_usd:.4f}"
                )
            outputs.append(res)

        out_dir = Path("reports/paper1/expanded/exp2/predictions_locked")
        lock = finalize_and_lock_exp2_predictions(
            model_id=args.model_id,
            split_name=args.split,
            shard_outputs=outputs,
            expected_studies=expected_studies,
            checkpoint_sha256=outputs[0]["checkpoint_sha256"],
            out_dir=out_dir,
        )
        print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
