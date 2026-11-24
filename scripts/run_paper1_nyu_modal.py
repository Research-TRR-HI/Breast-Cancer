"""Paper 1 — NYU Four-View (SplitBreastModel) Step 2 Optimized L40S GPU Pilot & Resumable Inference.

Reuses the verified Step-1 lossless uint16 PNG cache (`157.506 GiB` across 34,264 images / 8,566 exams):
    /rsna_vol/cache/paper1_nyu_uint16_v1/{rsna_val_calib,nlbs_4view}/

Features:
- Parallel NVMe staging (`/tmp`) via ThreadPoolExecutor (no repeated DICOM decoding or CC cropping)
- Parallel CPU prefetching with per-exam `RandomState(seed=0)` 10-TTA generation (batch-size invariant)
- Configurable L40S GPU exam batching (`batch_exams`) strictly in FP32 eval mode
- Strict resumable shard provenance locks (`validate_nyu_shard_payload` + `finalize_and_lock_nyu_predictions`)
- Practical cost stop guard (`max_pilot_cost_usd` on pilot; `--approve-full-infer` + `--max-spend-usd` on infer)

Usage:
    # Step 2 GPU Pilot ONLY (400 cached imgs / 100 exams incl. fallback cases, parity & batch benchmark):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_nyu_modal.py --stage pilot
    # Full Step 2 GPU Inference (BLOCKED without --approve-full-infer):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_nyu_modal.py --stage infer --split rsna_val_calib --approve-full-infer
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-nyu-fourview-inference"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
NLBS_VOL = modal.Volume.from_name("nlbs-mammography-vol")

RSNA_MOUNT = Path("/rsna_vol")
NLBS_MOUNT = Path("/nlbs_vol")
NYU_CKPT_SHA256 = "9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014"
PREPROCESS_VERSION = "paper1_nyu_uint16_v1"
# Nvidia L40S ($1.95/hr) + 6 vCPU ($0.2838/hr) + 24 GiB RAM ($0.1920/hr)
L40S_COMBINED_HOURLY_USD = 2.4258

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
    .add_local_dir(
        "external/breast_cancer_classifier/sample_data/images",
        remote_path="/root/external/breast_cancer_classifier/sample_data/images",
    )
    .add_local_file(
        "datasets/paper1/rsna_val_calib_4view_selection.csv",
        remote_path="/root/datasets/paper1/rsna_val_calib_4view_selection.csv",
    )
    .add_local_file(
        "datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
        remote_path="/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
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
                raise FileNotFoundError(f"Missing/empty Step-1 NYU PNG: {s}")
            shutil.copy2(s, d)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_cp, missing))
    return time.perf_counter() - t0


def _group_step1_exams(step1_df: Any) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Group validated Step-1 manifest DataFrame into ordered 4-view exams."""
    exams: List[Tuple[str, List[Dict[str, Any]]]] = []
    for cid, grp in step1_df.groupby("case_id", sort=True):
        recs = grp.to_dict(orient="records")
        if len(recs) != 4:
            raise RuntimeError(f"Incomplete exam {cid}: expected 4 views, got {len(recs)}")
        exams.append((str(cid), recs))
    return exams


def _run_prefetched_batched_inference(
    exams: List[Tuple[str, List[Dict[str, Any]]]],
    nvme_dir_fn: Any,
    model: Any,
    nyu_mods: Dict[str, Any],
    device: Any,
    batch_exams: int,
    num_tta_epochs: int,
    seed: int,
    max_workers: int,
    verify_u16_sha256: bool,
    t_budget_start: float | None = None,
    max_spend_usd: float | None = None,
    label: str = "NYU",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, float]:
    """Prefetch exam PNG decode + 10-TTA generation on CPU threads overlapped with L40S GPU batches."""
    import numpy as np
    import torch
    from src.paper1.nyu_fourview_adapter import (
        NYU_VIEWS,
        load_nyu_cached_view,
        prepare_nyu_exam_tta_tensors,
        score_nyu_fourview_exam_batch,
    )

    def _prep_one_exam(exam_item: Tuple[str, List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        cid, view_rows = exam_item
        vdir = nvme_dir_fn(view_rows[0])
        view_crops: Dict[str, Dict[str, Any]] = {}
        for r in view_rows:
            slot = str(r["view_slot"])
            view_crops[slot] = load_nyu_cached_view(
                vdir / f"{r['item_id']}.png",
                best_center=(int(r["best_center_y"]), int(r["best_center_x"])),
                expected_shape=(int(r["height"]), int(r["width"])),
                expected_u16_sha256=str(r["u16_sha256"]) if verify_u16_sha256 else None,
            )
        tta_dict = prepare_nyu_exam_tta_tensors(
            view_crops=view_crops, nyu_mods=nyu_mods, num_tta_epochs=num_tta_epochs, seed=seed
        )
        return cid, view_rows, tta_dict

    records: List[Dict[str, Any]] = []
    fallback_logs: List[Dict[str, Any]] = []
    decode_tta_sec = 0.0
    gpu_infer_sec = 0.0
    # Each exam's 10-TTA float32 tensor dict is ~0.78 GiB; keep queue tightly bounded to prevent host OOM
    prefetch_limit = max(batch_exams + 2, 6)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = deque()
        next_idx = 0
        while next_idx < len(exams) and len(pending) < prefetch_limit:
            pending.append(pool.submit(_prep_one_exam, exams[next_idx]))
            next_idx += 1

        for start in range(0, len(exams), batch_exams):
            if t_budget_start is not None and max_spend_usd is not None:
                spent = ((time.perf_counter() - t_budget_start) / 3600.0) * L40S_COMBINED_HOURLY_USD
                if spent >= max_spend_usd:
                    raise RuntimeError(
                        f"COST GUARD STOP ({label}): elapsed spend ${spent:.4f} reached limit ${max_spend_usd:.4f} "
                        f"at exam {start}/{len(exams)}"
                    )

            t_d0 = time.perf_counter()
            b_count = min(batch_exams, len(exams) - start)
            prepped = [pending.popleft().result() for _ in range(b_count)]
            while next_idx < len(exams) and len(pending) < prefetch_limit:
                pending.append(pool.submit(_prep_one_exam, exams[next_idx]))
                next_idx += 1
            decode_tta_sec += time.perf_counter() - t_d0

            torch.cuda.synchronize()
            t_g0 = time.perf_counter()
            preds_list = score_nyu_fourview_exam_batch(
                model=model, tta_exam_tensors=[p[2] for p in prepped], device=device
            )
            torch.cuda.synchronize()
            gpu_infer_sec += time.perf_counter() - t_g0

            for (cid, view_rows, _), preds in zip(prepped, preds_list):
                if not all(np.isfinite(float(preds[k])) for k in preds):
                    raise RuntimeError(f"Non-finite NYU prediction for {label} exam {cid}")
                n_fb = 0
                for vr in view_rows:
                    if bool(vr["crop_fallback"]):
                        n_fb += 1
                        fallback_logs.append(
                            {
                                "split_name": str(vr["split_name"]),
                                "case_id": cid,
                                "item_id": str(vr["item_id"]),
                                "view_slot": str(vr["view_slot"]),
                                "photometric": str(vr.get("photometric", "")),
                                "fallback_strategy": str(vr.get("fallback_strategy", "")),
                                "crop_error": str(vr.get("crop_error", "")),
                            }
                        )
                records.append(
                    {
                        "case_id": cid,
                        "n_views": len(view_rows),
                        "n_crop_fallbacks": n_fb,
                        "has_crop_fallback": n_fb > 0,
                        **preds,
                    }
                )
            done = start + b_count
            if done % 100 == 0 or done == len(exams):
                print(f"[{label}] {done}/{len(exams)} exams (bs={batch_exams}, fb_views={len(fallback_logs)})", flush=True)

    return records, fallback_logs, decode_tta_sec, gpu_infer_sec


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=24576,
    timeout=1800,
)
def run_nyu_step2_pilot(
    n_exams_per_split: int = 50,
    batch_exams: int = 4,
    max_pilot_cost_usd: float = 0.18,
    num_tta_epochs: int = 10,
) -> Dict[str, Any]:
    """L40S Step-2 pilot: 100 exams (400 imgs incl. all 34 RSNA fallback exams), batch & raw DICOM parity, cost cap."""
    import numpy as np
    import pandas as pd
    import resource
    import torch
    from src.evaluation.stage_c_eval import assert_no_forbidden_columns, compute_sha256_file
    from src.paper1.nyu_fourview_adapter import (
        NYU_INPUT_SIZES,
        NYU_VIEWS,
        build_nyu_fourview_model,
        extract_nyu_cropped_view,
        load_nyu_cached_view,
        score_nyu_fourview_exam,
        score_nyu_fourview_exam_batch,
        prepare_nyu_exam_tta_tensors,
    )

    RSNA_VOL.reload()
    t_total0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L40S GPU expected for NYU Step 2 pilot!")

    ckpt_path = RSNA_MOUNT / "models" / "paper1" / "sample_image_model.p"
    got_sha = compute_sha256_file(ckpt_path).lower()
    if got_sha != NYU_CKPT_SHA256:
        raise RuntimeError(f"NYU checkpoint SHA mismatch: {got_sha} != {NYU_CKPT_SHA256}")

    t_load0 = time.perf_counter()
    model, nyu_mods = build_nyu_fourview_model(
        ckpt_path, device=device, nyu_repo_dir=Path("/root/external/breast_cancer_classifier")
    )
    model_load_sec = time.perf_counter() - t_load0

    # 1. Official sample_data Exam 0 L40S deterministic sanity check (num_tta_epochs=1 vs 10)
    n_crop, n_load, n_coc = nyu_mods["crop_mammogram"], nyu_mods["loading"], nyu_mods["calc_optimal_centers"]
    sample0_views = {}
    for v in NYU_VIEWS:
        img = n_load.read_image_png(f"/root/external/breast_cancer_classifier/sample_data/images/0_{v.replace('-', '_')}.png")
        cinfo = n_crop.crop_img_from_largest_connected(
            img, n_crop.image_orientation("NO", v.split("-")[0]), True, 100, 50, 1.0 / 3.0
        )
        t, b, l, r = cinfo[0]
        cu = np.ascontiguousarray(n_load.flip_image(img[t:b, l:r], v, "NO"))
        tlbr = (
            n_coc.get_bottomrightmost_pixel_constraint(cinfo[1][1], cinfo[2][0])
            if "MLO" in v
            else n_coc.get_rightmost_pixel_constraint(cinfo[1][1])
        )
        oc = n_coc.get_image_optimal_window_info(
            cu, com=np.array(cu.shape) // 2, window_dim=np.array(NYU_INPUT_SIZES[v]), tl_br_constraint=tlbr
        )
        sample0_views[v] = {"image_f32": cu.astype(np.float32), "best_center": (int(oc["best_center_y"]), int(oc["best_center_x"]))}
    sample0_10tta = score_nyu_fourview_exam(model, nyu_mods, sample0_views, device, num_tta_epochs=num_tta_epochs, seed=0)

    # 2. Load label-free Step-1 manifests (and raw rel_paths strictly without cancer column)
    rsna_root = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / "rsna_val_calib"
    nlbs_root = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / "nlbs_4view"
    rsna_s1 = pd.read_csv(rsna_root / "manifest.csv", dtype={"item_id": str, "case_id": str})
    nlbs_s1 = pd.read_csv(nlbs_root / "manifest.csv", dtype={"item_id": str, "case_id": str})
    assert_no_forbidden_columns(rsna_s1.columns, "pilot_rsna_step1")
    assert_no_forbidden_columns(nlbs_s1.columns, "pilot_nlbs_step1")

    rsna_raw_df = pd.read_csv("/root/datasets/paper1/rsna_val_calib_4view_selection.csv", dtype=str)
    nlbs_raw_df = pd.read_csv("/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv", dtype=str)
    assert_no_forbidden_columns(rsna_raw_df.columns, "pilot_rsna_raw")
    assert_no_forbidden_columns(nlbs_raw_df.columns, "pilot_nlbs_raw")
    rsna_rel_map = rsna_raw_df.set_index("image_id")["rel_path"].to_dict()
    nlbs_rel_map = nlbs_raw_df.set_index("image_uid")["dicom_rel_path"].to_dict()

    # Select 50 RSNA exams = ALL 34 fallback exams + 16 standard (spread across MONOCHROME1/2), plus 50 NLBS exams
    rsna_exams_all = _group_step1_exams(rsna_s1)
    nlbs_exams_all = _group_step1_exams(nlbs_s1)
    fb_cases = sorted({str(c) for c in rsna_s1.loc[rsna_s1["crop_fallback"].astype(bool), "case_id"].tolist()})
    rsna_map = dict(rsna_exams_all)
    chosen_rsna_ids = list(fb_cases[:n_exams_per_split])
    rem_rsna = [cid for cid, _ in rsna_exams_all if cid not in set(chosen_rsna_ids)]
    step_r = max(1, len(rem_rsna) // max(1, n_exams_per_split - len(chosen_rsna_ids)))
    for cid in rem_rsna[::step_r]:
        if len(chosen_rsna_ids) >= n_exams_per_split:
            break
        chosen_rsna_ids.append(cid)
    rsna_pilot_exams = [(cid, rsna_map[cid]) for cid in chosen_rsna_ids]

    step_n = max(1, len(nlbs_exams_all) // max(1, n_exams_per_split))
    nlbs_pilot_exams = nlbs_exams_all[::step_n][:n_exams_per_split]
    pilot_exams = rsna_pilot_exams + nlbs_pilot_exams  # 100 exams = 400 images

    # 3. Parallel NVMe staging
    pilot_nvme = Path("/tmp/nyu_pilot_400")
    if pilot_nvme.exists():
        shutil.rmtree(pilot_nvme)
    rsna_ids = [str(r["item_id"]) for _, vr in rsna_pilot_exams for r in vr]
    nlbs_ids = [str(r["item_id"]) for _, vr in nlbs_pilot_exams for r in vr]
    stage_rsna_sec = _stage_pngs_nvme(rsna_ids, rsna_root / "png", pilot_nvme / "rsna_val_calib", max_workers=16)
    stage_nlbs_sec = _stage_pngs_nvme(nlbs_ids, nlbs_root / "png", pilot_nvme / "nlbs_4view", max_workers=16)
    nvme_stage_sec = stage_rsna_sec + stage_nlbs_sec

    # Warmup GPU outside timed loop
    torch.cuda.reset_peak_memory_stats()

    def _nvme_dir_fn(row: Dict[str, Any]) -> Path:
        return pilot_nvme / str(row["split_name"])

    # 4. Run optimized prefetched + batched L40S inference across all 100 exams (400 images)
    t_loop0 = time.perf_counter()
    pilot_recs, pilot_fbs, decode_tta_sec, gpu_infer_sec = _run_prefetched_batched_inference(
        exams=pilot_exams,
        nvme_dir_fn=_nvme_dir_fn,
        model=model,
        nyu_mods=nyu_mods,
        device=device,
        batch_exams=batch_exams,
        num_tta_epochs=num_tta_epochs,
        seed=0,
        max_workers=6,
        verify_u16_sha256=True,
        t_budget_start=t_total0,
        max_spend_usd=max_pilot_cost_usd,
        label=f"NYU-Pilot-bs{batch_exams}",
    )
    infer_loop_sec = time.perf_counter() - t_loop0
    peak_vram_gb_bs = float(torch.cuda.max_memory_allocated() / 1e9)
    pilot_pred_map = {f"{vr[0]['split_name']}:{cid}": rec for (cid, vr), rec in zip(pilot_exams, pilot_recs)}

    # Optional bs=8 mini-probe (16 exams) if batch_exams != 8 to measure higher batch VRAM/speed safely
    bs8_probe = None
    if batch_exams != 8:
        spent_so_far = ((time.perf_counter() - t_total0) / 3600.0) * L40S_COMBINED_HOURLY_USD
        if spent_so_far < max_pilot_cost_usd * 0.75:
            torch.cuda.reset_peak_memory_stats()
            t_b80 = time.perf_counter()
            recs_b8, _, dec_b8, gpu_b8 = _run_prefetched_batched_inference(
                exams=pilot_exams[:16],
                nvme_dir_fn=_nvme_dir_fn,
                model=model,
                nyu_mods=nyu_mods,
                device=device,
                batch_exams=8,
                num_tta_epochs=num_tta_epochs,
                seed=0,
                max_workers=6,
                verify_u16_sha256=False,
                t_budget_start=t_total0,
                max_spend_usd=max_pilot_cost_usd,
                label="NYU-Probe-bs8",
            )
            w8 = time.perf_counter() - t_b80
            bs8_probe = {
                "n_exams": 16,
                "wall_sec": round(w8, 3),
                "exams_per_sec": round(16.0 / max(w8, 1e-6), 2),
                "gpu_infer_sec": round(gpu_b8, 3),
                "peak_vram_gb": round(float(torch.cuda.max_memory_allocated() / 1e9), 3),
                "max_exam_score_diff_vs_bs": round(
                    max(
                        abs(r["exam_score"] - pilot_pred_map[f"{vr[0]['split_name']}:{cid}"]["exam_score"])
                        for (cid, vr), r in zip(pilot_exams[:16], recs_b8)
                    ),
                    8,
                ),
            }

    # 5. Strict parity checks:
    #    (a) Batched vs original single-exam score_nyu_fourview_exam across 12 exams (8 RSNA incl. fallbacks + 4 NLBS)
    parity_exams = rsna_pilot_exams[:8] + nlbs_pilot_exams[:4]
    batch_vs_bs1_rows = []
    t_bs1_0 = time.perf_counter()
    for cid, view_rows in parity_exams:
        sp = str(view_rows[0]["split_name"])
        vc_map = {
            str(r["view_slot"]): load_nyu_cached_view(
                pilot_nvme / sp / f"{r['item_id']}.png",
                (int(r["best_center_y"]), int(r["best_center_x"])),
                expected_shape=(int(r["height"]), int(r["width"])),
            )
            for r in view_rows
        }
        orig_res = score_nyu_fourview_exam(model, nyu_mods, vc_map, device, num_tta_epochs=num_tta_epochs, seed=0)
        batched_res = pilot_pred_map[f"{sp}:{cid}"]
        diff_exam = abs(float(orig_res["exam_score"]) - float(batched_res["exam_score"]))
        diff_max4 = max(
            abs(float(orig_res[k]) - float(batched_res[k]))
            for k in ("left_benign", "right_benign", "left_malignant", "right_malignant")
        )
        batch_vs_bs1_rows.append(
            {
                "split": sp,
                "case_id": cid,
                "n_fallbacks": int(batched_res["n_crop_fallbacks"]),
                "orig_bs1_exam_score": round(float(orig_res["exam_score"]), 8),
                "batched_exam_score": round(float(batched_res["exam_score"]), 8),
                "abs_diff_exam_score": round(diff_exam, 9),
                "abs_diff_max4": round(diff_max4, 9),
            }
        )
    bs1_12exams_sec = time.perf_counter() - t_bs1_0

    #    (b) Cached PNG vs Online Raw DICOM extract_nyu_cropped_view + score_nyu_fourview_exam parity (3 RSNA incl. fallback + 3 NLBS)
    online_vs_cached_rows = []
    raw_parity_exams = rsna_pilot_exams[:3] + nlbs_pilot_exams[:3]
    for cid, view_rows in raw_parity_exams:
        sp = str(view_rows[0]["split_name"])
        raw_vc = {}
        for r in view_rows:
            iid = str(r["item_id"])
            slot = str(r["view_slot"])
            dcm_p = (
                RSNA_MOUNT / "rsna" / "raw" / rsna_rel_map[iid]
                if sp == "rsna_val_calib"
                else NLBS_MOUNT / "nlbs" / "extracted" / nlbs_rel_map[iid]
            )
            raw_vc[slot] = extract_nyu_cropped_view(dcm_p, slot, nyu_mods)
        orig_raw_res = score_nyu_fourview_exam(model, nyu_mods, raw_vc, device, num_tta_epochs=num_tta_epochs, seed=0)
        batched_res = pilot_pred_map[f"{sp}:{cid}"]
        diff_raw = abs(float(orig_raw_res["exam_score"]) - float(batched_res["exam_score"]))
        online_vs_cached_rows.append(
            {
                "split": sp,
                "case_id": cid,
                "n_fallbacks": int(batched_res["n_crop_fallbacks"]),
                "online_raw_exam_score": round(float(orig_raw_res["exam_score"]), 8),
                "cached_batched_exam_score": round(float(batched_res["exam_score"]), 8),
                "abs_diff_vs_raw_online": round(diff_raw, 9),
            }
        )

    total_container_sec = time.perf_counter() - t_total0
    e2e_pilot_sec = nvme_stage_sec + infer_loop_sec
    pilot_n_exams = len(pilot_exams)
    pilot_n_imgs = pilot_n_exams * 4
    e2e_exams_per_sec = pilot_n_exams / max(e2e_pilot_sec, 1e-6)
    e2e_imgs_per_sec = pilot_n_imgs / max(e2e_pilot_sec, 1e-6)
    gpu_exams_per_sec = pilot_n_exams / max(gpu_infer_sec, 1e-6)
    host_rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))

    measured_infer_cost_usd = (e2e_pilot_sec / 3600.0) * L40S_COMBINED_HOURLY_USD
    measured_container_cost_usd = (total_container_sec / 3600.0) * L40S_COMBINED_HOURLY_USD

    # Project full 8,566 exams (2,382 RSNA + 6,184 NLBS) with 4 shards/split (= 8 container loads * model_load + startup)
    full_exams = 8566
    startup_overhead_per_shard_sec = 20.0 + model_load_sec
    proj_4shard_split_sec = (full_exams / max(e2e_exams_per_sec, 1e-6)) + (8 * startup_overhead_per_shard_sec)
    proj_full_usd = (proj_4shard_split_sec / 3600.0) * L40S_COMBINED_HOURLY_USD

    scores_arr = np.array([r["exam_score"] for r in pilot_recs], dtype=np.float64)
    return {
        "modal_profile": "hasinishrak2015",
        "gpu_name": torch.cuda.get_device_name(0),
        "checkpoint_sha256": got_sha,
        "preprocess_version": PREPROCESS_VERSION,
        "pilot_counts": {
            "n_rsna_exams": len(rsna_pilot_exams),
            "n_rsna_fallback_exams_tested": sum(1 for r in pilot_recs[: len(rsna_pilot_exams)] if r["has_crop_fallback"]),
            "n_nlbs_exams": len(nlbs_pilot_exams),
            "total_exams": pilot_n_exams,
            "total_images": pilot_n_imgs,
            "batch_exams": batch_exams,
            "num_tta_epochs": num_tta_epochs,
            "tta_seed": 0,
        },
        "all_scores_finite": bool(np.all(np.isfinite(scores_arr))),
        "score_summary_label_free": {
            "min": round(float(np.min(scores_arr)), 6),
            "median": round(float(np.median(scores_arr)), 6),
            "max": round(float(np.max(scores_arr)), 6),
        },
        "sample0_10tta_l40s": sample0_10tta,
        "parity_batched_vs_orig_bs1_12exams": {
            "max_abs_diff_exam_score": float(max(r["abs_diff_exam_score"] for r in batch_vs_bs1_rows)),
            "max_abs_diff_all4_classes": float(max(r["abs_diff_max4"] for r in batch_vs_bs1_rows)),
            "orig_bs1_12exams_sec": round(bs1_12exams_sec, 3),
            "orig_bs1_exams_per_sec": round(len(parity_exams) / max(bs1_12exams_sec, 1e-6), 2),
            "rows": batch_vs_bs1_rows,
        },
        "parity_cached_vs_online_raw_6exams": {
            "max_abs_diff_vs_raw_online": float(max(r["abs_diff_vs_raw_online"] for r in online_vs_cached_rows)),
            "rows": online_vs_cached_rows,
        },
        "bs8_mini_probe": bs8_probe,
        "timing_sec": {
            "model_load_sec": round(model_load_sec, 3),
            "nvme_stage_sec": round(nvme_stage_sec, 3),
            "decode_tta_sec": round(decode_tta_sec, 3),
            "gpu_infer_sec": round(gpu_infer_sec, 3),
            "infer_loop_sec": round(infer_loop_sec, 3),
            "e2e_pilot_sec": round(e2e_pilot_sec, 3),
            "total_container_sec": round(total_container_sec, 3),
        },
        "throughput": {
            "e2e_exams_per_sec": round(e2e_exams_per_sec, 2),
            "e2e_imgs_per_sec": round(e2e_imgs_per_sec, 2),
            "gpu_only_exams_per_sec": round(gpu_exams_per_sec, 2),
        },
        "memory": {
            "peak_vram_gb": round(peak_vram_gb_bs, 3),
            "host_rss_gb": round(host_rss_gb, 3),
        },
        "cost": {
            "l40s_hourly_usd": L40S_COMBINED_HOURLY_USD,
            "max_pilot_cost_usd": max_pilot_cost_usd,
            "measured_pilot_e2e_usd": round(measured_infer_cost_usd, 5),
            "measured_pilot_total_container_usd": round(measured_container_cost_usd, 5),
            "under_020_budget": bool(measured_container_cost_usd < 0.20),
            "projected_full_8566_exams_hours": round(proj_4shard_split_sec / 3600.0, 4),
            "projected_full_8566_exams_usd": round(proj_full_usd, 4),
        },
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=24576,
    ephemeral_disk=524288,
    timeout=7200,
)
def infer_nyu_shard(
    split_name: str,
    shard_idx: int,
    num_shards: int,
    batch_exams: int = 4,
    num_tta_epochs: int = 10,
    max_shard_cost_usd: float = 2.0,
) -> Dict[str, Any]:
    """Run resumable Step-2 NYU SplitBreastModel L40S inference from NVMe-staged cached uint16 PNGs."""
    import pandas as pd
    import resource
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file
    from src.paper1.nyu_fourview_adapter import (
        NYU_TTA_SEED_OFFICIAL,
        build_nyu_fourview_model,
        compute_nyu_case_ids_sha256,
        validate_and_load_nyu_4view_manifest,
        validate_nyu_shard_payload,
    )

    if int(num_tta_epochs) != 10:
        raise ValueError(f"Official protocol requires num_tta_epochs=10, got {num_tta_epochs}")
    batch_exams = int(batch_exams)
    RSNA_VOL.reload()
    t_container0 = time.perf_counter()

    ckpt_path = RSNA_MOUNT / "models" / "paper1" / "sample_image_model.p"
    got_sha = compute_sha256_file(ckpt_path).lower()
    if got_sha != NYU_CKPT_SHA256:
        raise RuntimeError(f"NYU checkpoint SHA mismatch: {got_sha} != {NYU_CKPT_SHA256}")

    cache_root = RSNA_MOUNT / "cache" / PREPROCESS_VERSION / split_name
    vol_png_dir = cache_root / "png"
    man_csv = cache_root / "manifest.csv"
    if not man_csv.exists():
        raise FileNotFoundError(f"Step 1 NYU manifest missing: {man_csv}")
    step1_manifest_sha = compute_sha256_file(man_csv)
    step1_df = pd.read_csv(man_csv, dtype={"item_id": str, "case_id": str})

    raw_man_path = (
        Path("/root/datasets/paper1/rsna_val_calib_4view_selection.csv")
        if split_name == "rsna_val_calib"
        else Path("/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv")
    )
    _ = validate_and_load_nyu_4view_manifest(split_name, raw_man_path)

    all_exams = _group_step1_exams(step1_df)
    shard_exams = all_exams[shard_idx::num_shards]
    expected_case_ids = [cid for cid, _ in shard_exams]
    expected_case_ids_sha = compute_nyu_case_ids_sha256(expected_case_ids)

    out_dir = RSNA_MOUNT / "cache" / "paper1_nyu_shards" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_file = out_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}_be{batch_exams}.json"
    if shard_file.exists():
        try:
            cached = json.loads(shard_file.read_text(encoding="utf-8"))
        except Exception as exc:
            cached = {}
            print(f"[NYU {split_name} shard {shard_idx}] Corrupt shard ignored ({exc}); recomputing.", flush=True)
        ok, reason = validate_nyu_shard_payload(
            cached,
            expected_case_ids=expected_case_ids,
            batch_exams=batch_exams,
            split_name=split_name,
            shard_idx=shard_idx,
            num_shards=num_shards,
            num_tta_epochs=num_tta_epochs,
            tta_seed=NYU_TTA_SEED_OFFICIAL,
            checkpoint_sha256=NYU_CKPT_SHA256,
            preprocess_version=PREPROCESS_VERSION,
            preprocess_manifest_sha256=step1_manifest_sha,
        )
        if ok:
            cached["resumed_from_cache"] = True
            print(f"[NYU {split_name} shard {shard_idx}] Resumed validated cache ({len(expected_case_ids)} exams).", flush=True)
            return cached
        print(f"[NYU {split_name} shard {shard_idx}] Incompatible cached shard ({reason}); recomputing.", flush=True)

    nvme_dir = Path(f"/tmp/nyu_{split_name}_s{shard_idx:02d}")
    shard_item_ids = [str(r["item_id"]) for _, vr in shard_exams for r in vr]
    nvme_stage_sec = _stage_pngs_nvme(shard_item_ids, vol_png_dir, nvme_dir, max_workers=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L40S GPU expected for NYU shard inference!")
    t_load0 = time.perf_counter()
    model, nyu_mods = build_nyu_fourview_model(
        ckpt_path, device=device, nyu_repo_dir=Path("/root/external/breast_cancer_classifier")
    )
    model_load_sec = time.perf_counter() - t_load0

    torch.cuda.reset_peak_memory_stats()
    t_loop0 = time.perf_counter()
    records, fallback_logs, decode_tta_sec, gpu_infer_sec = _run_prefetched_batched_inference(
        exams=shard_exams,
        nvme_dir_fn=lambda _r: nvme_dir,
        model=model,
        nyu_mods=nyu_mods,
        device=device,
        batch_exams=batch_exams,
        num_tta_epochs=num_tta_epochs,
        seed=NYU_TTA_SEED_OFFICIAL,
        max_workers=6,
        verify_u16_sha256=True,
        t_budget_start=t_container0,
        max_spend_usd=max_shard_cost_usd,
        label=f"NYU-{split_name}-s{shard_idx:02d}",
    )
    elapsed_sec = time.perf_counter() - t_loop0
    container_wall_sec = time.perf_counter() - t_container0
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    host_rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))
    estimated_cost_usd = (container_wall_sec / 3600.0) * L40S_COMBINED_HOURLY_USD

    payload = {
        "schema_version": "paper1_nyu_shard_v2",
        "model_id": "nyu_fourview_pretrained",
        "checkpoint_file": "sample_image_model.p",
        "checkpoint_sha256": NYU_CKPT_SHA256,
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_manifest_sha256": step1_manifest_sha,
        "split_name": split_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "batch_exams": batch_exams,
        "num_tta_epochs": num_tta_epochs,
        "tta_seed": NYU_TTA_SEED_OFFICIAL,
        "expected_n_exams": len(expected_case_ids),
        "expected_n_images": len(expected_case_ids) * 4,
        "expected_case_ids_sha256": expected_case_ids_sha,
        "model_load_sec": round(model_load_sec, 3),
        "nvme_stage_sec": round(nvme_stage_sec, 3),
        "decode_tta_sec": round(decode_tta_sec, 3),
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
    ok, reason = validate_nyu_shard_payload(
        payload,
        expected_case_ids=expected_case_ids,
        batch_exams=batch_exams,
        split_name=split_name,
        shard_idx=shard_idx,
        num_shards=num_shards,
        num_tta_epochs=num_tta_epochs,
        tta_seed=NYU_TTA_SEED_OFFICIAL,
        checkpoint_sha256=NYU_CKPT_SHA256,
        preprocess_version=PREPROCESS_VERSION,
        preprocess_manifest_sha256=step1_manifest_sha,
    )
    if not ok:
        raise RuntimeError(f"Generated NYU shard failed self-validation: {reason}")

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
    batch_exams: int = 4,
    num_tta_epochs: int = 10,
    n_exams_per_split: int = 50,
    max_pilot_cost_usd: float = 0.18,
    max_shard_cost_usd: float = 2.0,
    shard_indices: str = "",
    approve_full_infer: bool = False,
) -> None:
    import pandas as pd
    from src.paper1.nyu_fourview_adapter import (
        finalize_and_lock_nyu_predictions,
        validate_nyu_step2_cli_and_guard,
    )

    validate_nyu_step2_cli_and_guard(
        stage=stage,
        split=split,
        num_shards=num_shards,
        batch_exams=batch_exams,
        num_tta_epochs=num_tta_epochs,
        approve_full_infer=approve_full_infer,
    )

    if stage == "pilot":
        res = run_nyu_step2_pilot.remote(
            n_exams_per_split=n_exams_per_split,
            batch_exams=batch_exams,
            max_pilot_cost_usd=max_pilot_cost_usd,
            num_tta_epochs=num_tta_epochs,
        )
        out_p = Path("reports/paper1/nyu_step2_gpu_pilot.json")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return

    step1_local_csv = Path(f"reports/paper1/nyu_step1_{split}_manifest.csv")
    if not step1_local_csv.exists():
        raise FileNotFoundError(f"Expected Step 1 local manifest CSV: {step1_local_csv}")
    step1_df = pd.read_csv(step1_local_csv, dtype={"item_id": str, "case_id": str})
    case_out_col = "patient_id" if split == "rsna_val_calib" else "StudyInstanceUID"

    shards = (
        [int(x.strip()) for x in shard_indices.split(",") if x.strip()]
        if shard_indices.strip()
        else list(range(num_shards))
    )
    outs = list(
        infer_nyu_shard.starmap(
            [(split, s, num_shards, batch_exams, num_tta_epochs, max_shard_cost_usd) for s in shards]
        )
    )
    for o in sorted(outs, key=lambda x: int(x["shard_idx"])):
        print(
            f"[NYU Shard {o['shard_idx']}/{o['num_shards']}] exams={len(o['records'])} "
            f"be={o['batch_exams']} wall={o['container_wall_sec']}s "
            f"decode_tta={o['decode_tta_sec']}s gpu={o['gpu_infer_sec']}s "
            f"est_cost=${o['estimated_cost_usd']:.5f} resumed={o.get('resumed_from_cache', False)}"
        )

    if len(shards) < num_shards or set(shards) != set(range(num_shards)):
        print(f"[Partial shard run {shards}/{num_shards}] Skipping lock until all shards finish.")
        return

    lock_manifest = finalize_and_lock_nyu_predictions(
        split_name=split,
        shard_outputs=outs,
        expected_step1_df=step1_df,
        case_out_col=case_out_col,
        batch_exams=batch_exams,
        checkpoint_sha256=NYU_CKPT_SHA256,
        preprocess_version=PREPROCESS_VERSION,
        out_dir=Path("reports/paper1/predictions_locked"),
    )
    print(json.dumps(lock_manifest, indent=2))
