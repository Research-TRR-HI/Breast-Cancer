"""Paper 1 Experiment 2 — Modal Inference Runner & Spend/Approval Guarded Pipeline.

Targets Modal profile `tigerwood693` explicitly (NEVER calls `modal profile activate`).
Wave 1 (`--stage plan`) is 100% CPU-free/local dry-run planning.
Wave 2 transfers/downloads (`--approve-wave2-download`), Wave 3 L40S pilot (`--approve-pilot`),
and Phase 2 full NLBS inference (`--approve-full-infer`) are strictly gated behind explicit
CLI approval flags so no paid Modal compute or volume transfer runs unapproved.

Example Usage:
    # Wave 1 dry-run plan (local CPU, $0.00):
    python scripts/run_paper1_exp2_modal.py --stage plan --model-id glam_pretrained

    # Wave 3 L40S Pilot (200-500 units, requires explicit approval):
    $env:MODAL_PROFILE="tigerwood693"; modal run scripts/run_paper1_exp2_modal.py --stage pilot --model-id glam_pretrained --approve-pilot
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paper1.exp2_adapters import (
    EXP2_MODEL_REGISTRY,
    assert_model_eligible,
    build_petrini_breast_pairs,
    compute_unit_ids_sha256,
    finalize_and_lock_exp2_predictions,
    load_exp2_nlbs_manifest,
    validate_exp2_shard_payload,
)
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-exp2-pretrained-inference"
TARGET_MODAL_PROFILE = "tigerwood693"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")
L40S_COMBINED_HOURLY_USD = 2.3618  # 1x L40S ($1.9512) + 6 vCPU ($0.2838) + 16 GiB RAM ($0.1280)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        *PINNED_MODAL_PIP_PACKAGES,
        "timm==1.0.15",
        "transformers==4.49.0",
        "safetensors==0.5.3",
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


def validate_exp2_cli_and_guard(
    *,
    stage: str,
    model_id: str,
    split: str,
    num_shards: int,
    batch_size: int,
    pilot_limit: int,
    approve_pilot: bool,
    approve_full_infer: bool,
    max_total_spend_usd: float,
) -> None:
    """Validate CLI parameters and block unapproved Modal pilot or full inference runs."""
    valid_stages = ("plan", "pilot", "infer")
    if stage not in valid_stages:
        raise ValueError(f"Unknown --stage={stage!r}; expected {valid_stages}")
    spec = assert_model_eligible(model_id)
    if spec.eligibility == "eligible_locked_reuse":
        raise RuntimeError(
            f"{model_id} already has locked NLBS predictions (reuse reports/paper1/predictions_locked/ at $0.00)."
        )
    if split not in ("nlbs_4view", "nlbs_full"):
        raise ValueError(f"Unsupported --split={split!r}; expected 'nlbs_4view' or 'nlbs_full'")
    if int(num_shards) < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if int(batch_size) < 1:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
    if not (200 <= int(pilot_limit) <= 500):
        raise ValueError(f"--pilot-limit must be in [200, 500], got {pilot_limit}")
    if float(max_total_spend_usd) <= 0.0:
        raise ValueError(f"--max-total-spend-usd must be > 0, got {max_total_spend_usd}")

    if stage == "pilot" and not approve_pilot:
        raise RuntimeError(
            "BLOCKED: `--stage pilot` requires explicit `--approve-pilot` confirmation (Wave 3 L40S spend)."
        )
    if stage == "infer" and not approve_full_infer:
        raise RuntimeError(
            "BLOCKED: `--stage infer` requires explicit `--approve-full-infer` confirmation (Phase 2 full L40S spend)."
        )


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
    is_pilot: bool,
) -> Dict[str, Any]:
    """Execute resumable L40S inference shard for a single eligible Exp 2 model."""
    import cv2
    import numpy as np
    import torch
    from src.paper1.exp2_adapters import (
        build_glam_model,
        build_petrini_twoview_model,
        build_petrini_twoview_tensor,
        preprocess_glam_from_gmic_uint16,
        preprocess_mammoscreen_view_from_stage1_uint16,
        score_twoview_batch,
    )

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
    tag = "pilot" if is_pilot else f"shard_{shard_idx:02d}_of_{num_shards:02d}"
    shard_json = shard_dir / f"{tag}.json"

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

    ckpt_path = RSNA_MOUNT / checkpoint_rel_path
    nvme_dir = Path("/tmp/exp2_nvme") / model_id / split_name
    needed_img_ids: List[str] = []
    for r in unit_records:
        needed_img_ids.extend([str(x) for x in r["image_uids"]])
    needed_img_ids = sorted(set(needed_img_ids))

    if spec.nlbs_cache_source == "gmic_2944x1920":
        src_png_dir = RSNA_MOUNT / "cache" / "paper1_gmic_uint16_v1" / "nlbs" / "png_2944x1920"
    elif spec.nlbs_cache_source == "stage1_1520x912":
        src_png_dir = RSNA_MOUNT / "cache" / "stage1_nlbs_uint16"
    else:
        raise RuntimeError(f"Unsupported nlbs_cache_source={spec.nlbs_cache_source!r} for {model_id}")

    nvme_sec = _stage_pngs_to_nvme(needed_img_ids, src_png_dir, nvme_dir)
    out_records: List[Dict[str, Any]] = []

    if model_id == "glam_pretrained":
        model, ckpt_sha = build_glam_model(
            ckpt_path,
            glam_repo_dir=RSNA_MOUNT / "external_repos" / "GLAM",
            device=device,
            expected_sha256=spec.checkpoint_sha256,
        )
        for r in unit_records:
            img_uid = str(r["image_uids"][0])
            arr = cv2.imread(str(nvme_dir / f"{img_uid}.png"), cv2.IMREAD_UNCHANGED)
            norm = preprocess_glam_from_gmic_uint16(arr)
            x = torch.from_numpy(norm[np.newaxis, np.newaxis, ...].copy()).to(device)
            with torch.no_grad():
                pred = model(x).float().cpu().numpy()[0]
            out_records.append(
                {
                    "unit_id": str(r["unit_id"]),
                    "StudyInstanceUID": str(r["StudyInstanceUID"]),
                    "image_uid": img_uid,
                    "benign_pred": float(pred[0]),
                    "score": float(pred[1]),
                }
            )
    elif model_id in ("petrini_twoview_efnb3", "petrini_vindr_r2"):
        variant = "cbis_efnb3" if model_id == "petrini_twoview_efnb3" else "vindr_convnext"
        ds_name = "CBIS-DDSM" if model_id == "petrini_twoview_efnb3" else "VINDR-MAMMO"
        model, ckpt_sha = build_petrini_twoview_model(
            ckpt_path,
            petrini_repo_dir=RSNA_MOUNT / "external_repos" / "multiple-view",
            device=device,
            variant=variant,
        )
        for i in range(0, len(unit_records), batch_size):
            chunk = unit_records[i : i + batch_size]
            tensors = []
            for r in chunk:
                cc_arr = cv2.imread(str(nvme_dir / f"{r['image_uids'][0]}.png"), cv2.IMREAD_UNCHANGED)
                mlo_arr = cv2.imread(str(nvme_dir / f"{r['image_uids'][1]}.png"), cv2.IMREAD_UNCHANGED)
                tensors.append(build_petrini_twoview_tensor(cc_arr, mlo_arr, dataset=ds_name))
            probs = score_twoview_batch(model, torch.stack(tensors, dim=0), device)
            for r, p in zip(chunk, probs):
                out_records.append(
                    {
                        "unit_id": str(r["unit_id"]),
                        "StudyInstanceUID": str(r["StudyInstanceUID"]),
                        "laterality": str(r["laterality"]),
                        "cc_image_uid": str(r["image_uids"][0]),
                        "mlo_image_uid": str(r["image_uids"][1]),
                        "score": float(p),
                    }
                )
    elif model_id == "mammoscreen_r1":
        from transformers import AutoModel
        from src.paper1.exp2_adapters import verify_checkpoint_sha256

        ckpt_sha = verify_checkpoint_sha256(ckpt_path / "model.safetensors", spec.checkpoint_sha256)
        model = AutoModel.from_pretrained(str(ckpt_path), trust_remote_code=True, local_files_only=True)
        model = model.eval().to(device)
        for i in range(0, len(unit_records), batch_size):
            chunk = unit_records[i : i + batch_size]
            inputs = []
            for r in chunk:
                cc_u16 = cv2.imread(str(nvme_dir / f"{r['image_uids'][0]}.png"), cv2.IMREAD_UNCHANGED)
                mlo_u16 = cv2.imread(str(nvme_dir / f"{r['image_uids'][1]}.png"), cv2.IMREAD_UNCHANGED)
                inputs.append(
                    {
                        "cc": preprocess_mammoscreen_view_from_stage1_uint16(cc_u16),
                        "mlo": preprocess_mammoscreen_view_from_stage1_uint16(mlo_u16),
                    }
                )
            with torch.inference_mode():
                out = model(inputs, device=str(device))
            probs = out["cancer"].reshape(-1).float().cpu().numpy()
            for r, p in zip(chunk, probs):
                out_records.append(
                    {
                        "unit_id": str(r["unit_id"]),
                        "StudyInstanceUID": str(r["StudyInstanceUID"]),
                        "laterality": str(r["laterality"]),
                        "cc_image_uid": str(r["image_uids"][0]),
                        "mlo_image_uid": str(r["image_uids"][1]),
                        "score": float(p),
                    }
                )
    else:
        raise NotImplementedError(
            f"{model_id} requires Wave 2 Keras->PyTorch conversion + numerical parity test before Modal inference."
        )

    elapsed_sec = time.perf_counter() - t0
    est_cost = round((elapsed_sec / 3600.0) * L40S_COMBINED_HOURLY_USD, 5)
    payload = {
        "schema_version": "paper1_exp2_shard_v1",
        "model_id": model_id,
        "checkpoint_sha256": ckpt_sha,
        "split_name": split_name,
        "shard_idx": int(shard_idx),
        "num_shards": int(num_shards),
        "expected_unit_ids_sha256": exp_ids_sha,
        "nvme_stage_sec": round(nvme_sec, 3),
        "elapsed_sec": round(elapsed_sec, 3),
        "estimated_cost_usd": est_cost,
        "peak_vram_gb": round(float(torch.cuda.max_memory_allocated(device) / (1024**3)), 4),
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
    parser.add_argument("--stage", default="plan", choices=["plan", "pilot", "infer"])
    parser.add_argument("--model-id", required=True, choices=sorted(EXP2_MODEL_REGISTRY.keys()))
    parser.add_argument("--split", default="nlbs_4view", choices=["nlbs_4view", "nlbs_full"])
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pilot-limit", type=int, default=200)
    parser.add_argument("--checkpoint-rel-path", default="")
    parser.add_argument("--max-total-spend-usd", type=float, default=5.0)
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
        approve_pilot=args.approve_pilot,
        approve_full_infer=args.approve_full_infer,
        max_total_spend_usd=args.max_total_spend_usd,
    )

    items, expected_studies = build_unit_work_items(args.model_id, args.split)
    if args.stage == "plan":
        print(
            json.dumps(
                {
                    "stage": "plan",
                    "target_modal_profile": TARGET_MODAL_PROFILE,
                    "model_id": args.model_id,
                    "split": args.split,
                    "n_units": len(items),
                    "n_studies": len(expected_studies),
                    "num_shards": args.num_shards,
                    "pilot_limit": args.pilot_limit,
                    "cloud_spend_usd": 0.0,
                },
                indent=2,
            )
        )
        return

    with app.run():
        if args.stage == "pilot":
            pilot_items = items[: args.pilot_limit]
            res = run_exp2_shard_l40s.remote(
                model_id=args.model_id,
                split_name=args.split,
                shard_idx=0,
                num_shards=1,
                batch_size=args.batch_size,
                unit_records=pilot_items,
                checkpoint_rel_path=args.checkpoint_rel_path,
                is_pilot=True,
            )
            print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=2))
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
                is_pilot=False,
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
