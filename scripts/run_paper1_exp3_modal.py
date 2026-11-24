"""Paper 1 Exp 3 — Universal Modal Pipeline for External Datasets (IMDB, CMMD, LUMINA).

Targets `MODAL_PROFILE=tigerwood697` explicitly (NEVER calls `modal profile activate`).
Preserves all existing ResNet-50 assets on `rsna-mammography-vol`.

Wave Execution Gates:
    - Wave 1 (Zero-cost local/dry-run): `--dry-run` validates manifests, handoff specs, and cost/shard plan.
    - Wave 2 (CPU-only download/extract `cpu=1.0, memory=2048`): `--stage download --approve-download`
    - Wave 2 (CPU-only preprocessing cache): `--stage preprocess --approve-preprocess`
    - Wave 3 (L40S 200-500 img/exam pilot): `--stage pilot --dataset <ds> --model <m> --pilot-limit 200`
    - Phase 2 (Full L40S inference): `--stage infer --dataset <ds> --model <m> --approve-full-infer`

Usage:
    $env:MODAL_PROFILE="tigerwood697"; modal run scripts/run_paper1_exp3_modal.py --stage pilot --dataset cmmd --model resnet50_rsna --pilot-limit 200
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Sequence
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paper1.external_datasets import (
    DATASET_REGISTRY,
    select_common_4view_cohort,
    validate_inference_manifest,
)
from src.paper1.universal_eval import (
    MODEL_HANDOFF_SPECS,
    RSNA_SUPERVISED_MODELS,
    aggregate_image_records_to_exams,
    get_frozen_thresholds,
    lock_external_predictions,
    save_inference_shard_atomic,
    validate_agent1_handoff_bundle,
    validate_and_load_inference_shard,
)
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-exp3-external"
REQUIRED_MODAL_PROFILE = "tigerwood697"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")
L40S_HOURLY_USD = 2.3618  # 1x L40S ($1.95) + 6 vCPU ($0.2838) + 16 GiB RAM ($0.1280)
ALLOWED_CPU_WORKERS = (2, 4, 8, 16)
DEFAULT_PILOT_BATCH_SIZES: Dict[str, tuple[int, ...]] = {
    "ce_b5": (16, 32, 64),
    "trust_b5": (16, 32, 64),
    "dinov2_rsna": (32, 64, 128),
    "resnet50_rsna": (32, 64, 128),
    "gmic_rsna_finetuned": (1, 4, 8),
    "nyu_fourview_rsna_finetuned": (1, 2, 4),
}
CHECKPOINT_SYNC_COMMANDS: List[str] = [
    # 1. DINOv2 ViT-S/14 backbone (from hasinishrak2015/tigerwood693 -> tigerwood697) + RSNA head (local -> tigerwood697)
    "$env:MODAL_PROFILE='hasinishrak2015'; modal volume get rsna-mammography-vol models/paper1/dinov2_vits14_pretrain.pth models/paper1/dinov2_vits14_pretrain.pth",
    "$env:MODAL_PROFILE='tigerwood697'; modal volume put rsna-mammography-vol models/paper1/dinov2_vits14_pretrain.pth /models/paper1/dinov2_vits14_pretrain.pth",
    "$env:MODAL_PROFILE='tigerwood697'; modal volume put rsna-mammography-vol reports/paper1/dinov2_head/dinov2_vits14_head_seed42.pt /models/paper1/checkpoints/dinov2_vits14_head_seed42.pt",
    # 2. Mammo-CLIP B5 backbone + CE-only & TRUST Epoch-20 RSNA checkpoints (from hasinishrak74001 -> tigerwood697)
    "$env:MODAL_PROFILE='hasinishrak74001'; modal volume get rsna-mammography-vol models/mammo_clip/b5-model-best-epoch-7.tar models/mammo_clip/b5-model-best-epoch-7.tar",
    "$env:MODAL_PROFILE='hasinishrak74001'; modal volume get rsna-mammography-vol models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt",
    "$env:MODAL_PROFILE='hasinishrak74001'; modal volume get rsna-mammography-vol models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt",
    "$env:MODAL_PROFILE='tigerwood697'; modal volume put rsna-mammography-vol models/mammo_clip/b5-model-best-epoch-7.tar /models/mammo_clip/b5-model-best-epoch-7.tar",
    "$env:MODAL_PROFILE='tigerwood697'; modal volume put rsna-mammography-vol models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt /models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt",
    "$env:MODAL_PROFILE='tigerwood697'; modal volume put rsna-mammography-vol models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt /models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt",
]

app = modal.App(APP_NAME)

_REPO_ROOT = Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "unzip")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES, "openpyxl==3.1.5", "requests==2.32.3")
    .add_local_python_source("src")
)
for _rel_ext in ("external/dinov2/dinov2", "external/GMIC/src", "external/breast_cancer_classifier/src"):
    _p = _REPO_ROOT / _rel_ext
    if _p.exists():
        image = image.add_local_dir(str(_p), remote_path=f"/root/{_rel_ext}")
_hubconf = _REPO_ROOT / "external/dinov2/hubconf.py"
if _hubconf.exists():
    image = image.add_local_file(str(_hubconf), remote_path="/root/external/dinov2/hubconf.py")


def enforce_tigerwood697_profile(dry_run: bool = False) -> str:
    """Ensure remote Modal runs target `tigerwood697` explicitly without activating global profiles."""
    prof = os.environ.get("MODAL_PROFILE", "").strip()
    if not dry_run and prof != REQUIRED_MODAL_PROFILE:
        raise RuntimeError(
            f"BLOCKED: MODAL_PROFILE must be {REQUIRED_MODAL_PROFILE!r} (got {prof!r}). "
            "Set $env:MODAL_PROFILE='tigerwood697' explicitly; NEVER run `modal profile activate`."
        )
    return prof or "dry_run"


def parse_pilot_batch_sizes(raw: Optional[str], model_id: str) -> List[int]:
    """Parse progressive pilot batch-size sweep list (or return model-specific defaults)."""
    if not raw or not str(raw).strip():
        return list(DEFAULT_PILOT_BATCH_SIZES.get(model_id, (16, 32, 64)))
    out: List[int] = []
    for part in str(raw).split(","):
        val = int(part.strip())
        if val < 1:
            raise ValueError(f"Batch size must be >= 1, got {val}")
        if val not in out:
            out.append(val)
    if not out:
        raise ValueError("Empty pilot batch size list")
    return sorted(out)


def select_optimal_pilot_batch_size(
    bench_records: List[Dict[str, Any]],
    max_parity_diff: float = 1e-3,
) -> Dict[str, Any]:
    """Select fastest parity-verified batch size from progressive L40S pilot benchmark records."""
    if not bench_records:
        raise ValueError("No pilot benchmark records provided.")
    valid = [r for r in bench_records if float(r.get("max_abs_diff", 0.0)) <= float(max_parity_diff)]
    if not valid:
        raise RuntimeError(f"No pilot batch size satisfied max_abs_diff <= {max_parity_diff}")
    best = max(valid, key=lambda r: float(r["img_per_sec"]))
    return {
        "recommended_batch_size": int(best["batch_size"]),
        "img_per_sec": float(best["img_per_sec"]),
        "peak_vram_gb": float(best.get("peak_vram_gb", 0.0)),
        "max_abs_diff": float(best.get("max_abs_diff", 0.0)),
        "all_candidates": bench_records,
    }


def validate_exp3_cli_guards(
    *,
    stage: str,
    dataset_id: str,
    model_id: str,
    approve_download: bool,
    approve_preprocess: bool,
    approve_full_infer: bool,
    max_spend_usd: float,
    pilot_limit: int,
    cpu_workers: int = 4,
    pilot_batch_sizes: Optional[str] = None,
) -> None:
    """Enforce zero-cost approval gates and budget caps before any cloud execution."""
    if stage not in ("dry_run", "download", "preprocess", "pilot", "infer"):
        raise ValueError(f"Unsupported stage={stage!r}")
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset_id={dataset_id!r}; expected {sorted(DATASET_REGISTRY)}")
    if model_id not in RSNA_SUPERVISED_MODELS and model_id != "all":
        raise ValueError(f"Unknown model_id={model_id!r}; expected {RSNA_SUPERVISED_MODELS} or 'all'")
    if int(cpu_workers) not in ALLOWED_CPU_WORKERS:
        raise ValueError(f"--cpu-workers must be one of {ALLOWED_CPU_WORKERS}, got {cpu_workers}")
    if pilot_batch_sizes is not None:
        parse_pilot_batch_sizes(pilot_batch_sizes, model_id if model_id != "all" else "resnet50_rsna")
    if float(max_spend_usd) <= 0.0:
        raise ValueError(f"max_spend_usd must be > 0, got {max_spend_usd}")
    if stage == "pilot" and not (50 <= int(pilot_limit) <= 500):
        raise ValueError(f"pilot_limit must be in [50, 500], got {pilot_limit}")
    if stage == "download" and not approve_download:
        raise RuntimeError("BLOCKED: `--stage download` requires `--approve-download`.")
    if stage == "preprocess" and not approve_preprocess:
        raise RuntimeError("BLOCKED: `--stage preprocess` requires `--approve-preprocess`.")
    if stage == "infer" and not approve_full_infer:
        raise RuntimeError("BLOCKED: `--stage infer` requires `--approve-full-infer` (run `--stage pilot` first).")


def resolve_model_handoff(
    model_id: str,
    handoff_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve model weights & frozen RSNA thresholds, validating Agent 1 handoffs when pending."""
    spec = dict(MODEL_HANDOFF_SPECS[model_id])
    if spec.get("frozen_rsna_thresholds") is not None:
        return spec

    if handoff_json is None:
        raise RuntimeError(
            f"Model {model_id!r} requires Agent 1 RSNA fine-tuning handoff (`--handoff-json`) "
            "containing checkpoint path, SHA-256, schema_version, and calibrated RSNA val_calib thresholds."
        )
    data = json.loads(Path(handoff_json).read_text(encoding="utf-8"))
    entry = data[model_id] if model_id in data else data
    validated = validate_agent1_handoff_bundle(entry)
    spec["checkpoint_path_override"] = str(entry["checkpoint_path"])
    spec["expected_sha256"] = {**spec["expected_sha256"], "checkpoint": validated["checkpoint_sha256"]}
    spec["frozen_rsna_thresholds"] = {"target_98": validated["target_98"], "target_95": validated["target_95"]}
    return spec


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    cpu=1.0,
    memory=2048,
    timeout=14400,
)
def download_external_dataset_cpu(
    dataset_id: str,
    source_url: Optional[str] = None,
    max_items: int = 0,
) -> Dict[str, Any]:
    """Wave 2 CPU-only download (`cpu=1.0, memory=2048`) with 60s periodic volume commits & signal safety."""
    import requests

    if dataset_id not in ("cmmd", "lumina", "imdb"):
        raise ValueError(f"Unsupported download dataset_id={dataset_id!r}")

    raw_dir = RSNA_MOUNT / "external_datasets" / dataset_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = {"interrupted": False}

    def _on_sig(signum: int, _frame: Any) -> None:
        stop_flag["interrupted"] = True
        RSNA_VOL.commit()

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    last_commit = time.monotonic()

    def _maybe_commit(force: bool = False) -> None:
        nonlocal last_commit
        now = time.monotonic()
        if force or (now - last_commit) >= 60.0:
            RSNA_VOL.commit()
            last_commit = now

    downloaded = 0
    skipped = 0

    if dataset_id == "lumina":
        # OSF storage direct ZIP links for LUMINA_RAW (Benign.zip & Malign.zip)
        osf_api = "https://api.osf.io/v2/nodes/b63jc/files/osfstorage/"
        root_items = requests.get(osf_api, timeout=30).json()["data"]
        targets: Dict[str, str] = {}
        for item in root_items:
            name = item["attributes"]["name"]
            if name in ("Benign_Cases.xlsx", "Malign_Cases.xlsx", "LICENSE.txt"):
                targets[name] = item["links"]["download"]
            elif name == "LUMINA_RAW":
                rel = item["relationships"]["files"]["links"]["related"]["href"]
                for sub in requests.get(rel, timeout=30).json()["data"]:
                    targets[f"LUMINA_RAW_{sub['attributes']['name']}"] = sub["links"]["download"]

        for fname, url in targets.items():
            if stop_flag["interrupted"]:
                break
            dst = raw_dir / fname
            if dst.exists() and dst.stat().st_size > 0:
                skipped += 1
                continue
            tmp = dst.with_suffix(dst.suffix + ".part")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                        _maybe_commit(force=False)
                        if stop_flag["interrupted"]:
                            break
            if not stop_flag["interrupted"]:
                os.replace(tmp, dst)
                downloaded += 1
                _maybe_commit(force=True)
            if max_items > 0 and downloaded >= max_items:
                break

    elif dataset_id == "cmmd":
        # TCIA NBIA REST API series-level download (no credentials required)
        series_url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?Collection=CMMD&format=json"
        series_list = requests.get(series_url, timeout=60).json()
        clin_url = "https://www.cancerimagingarchive.net/wp-content/uploads/CMMD_clinicaldata_revision.xlsx"
        clin_dst = raw_dir / "CMMD_clinicaldata_revision.xlsx"
        if not clin_dst.exists():
            clin_dst.write_bytes(requests.get(clin_url, timeout=60).content)
            _maybe_commit(force=True)

        for s in series_list:
            if stop_flag["interrupted"]:
                break
            suid = str(s["SeriesInstanceUID"])
            pid = str(s["PatientID"])
            dst = raw_dir / "series_zips" / f"{pid}__{suid}.zip"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.stat().st_size > 0:
                skipped += 1
                continue
            img_url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={suid}"
            tmp = dst.with_suffix(".zip.part")
            with requests.get(img_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                        _maybe_commit(force=False)
            os.replace(tmp, dst)
            downloaded += 1
            _maybe_commit(force=False)
            if max_items > 0 and downloaded >= max_items:
                break

    elif dataset_id == "imdb":
        if not source_url:
            raise RuntimeError(
                "IMDB IBIA portal uses interactive web downloads (`https://ibdc.dbt.gov.in/ibia/studybrowse/`). "
                "Upload downloaded archive to volume or supply `--source-url` pointing to an accessible mirror/archive."
            )
        dst = raw_dir / Path(source_url).name
        if not dst.exists():
            tmp = dst.with_suffix(dst.suffix + ".part")
            with requests.get(source_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                        _maybe_commit(force=False)
            os.replace(tmp, dst)
            downloaded += 1

    _maybe_commit(force=True)
    return {"dataset_id": dataset_id, "downloaded": downloaded, "skipped": skipped, "interrupted": stop_flag["interrupted"]}


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    cpu=1.0,
    memory=2048,
    timeout=14400,
)
def ingest_lumina_neutral_cpu(ingest_map_csv_content: str) -> Dict[str, Any]:
    """Isolated Wave-2 CPU unzip (`cpu=1.0, memory=2048`) writing LUMINA files strictly to neutral paths.

    Extracts Benign.zip & Malign.zip to `extracted/lumina_neutral/<lumina_case_XXXX>/<lumina_img_YYYYY>.dcm`
    so zero `Benign/` or `Malign/` folder names exist anywhere on the inference mount.
    """
    import io
    import zipfile
    import pandas as pd

    map_df = pd.read_csv(io.StringIO(ingest_map_csv_content))
    raw_dir = RSNA_MOUNT / "external_datasets" / "lumina" / "raw"
    ext_root = RSNA_MOUNT / "external_datasets" / "lumina" / "extracted"
    ext_root.mkdir(parents=True, exist_ok=True)

    lookup = {str(r["source_image_path"]).lstrip("/"): str(r["neutral_image_path"]) for r in map_df.to_dict("records")}
    written = 0
    skipped = 0
    last_commit = time.monotonic()

    for cls_zip in ("LUMINA_RAW_Benign.zip", "LUMINA_RAW_Malign.zip"):
        zpath = raw_dir / cls_zip
        if not zpath.exists():
            raise FileNotFoundError(f"Missing raw LUMINA archive: {zpath}")
        with zipfile.ZipFile(zpath, "r") as zf:
            for member in zf.namelist():
                norm_m = member.lstrip("/")
                if norm_m not in lookup:
                    continue
                dst = ext_root / lookup[norm_m]
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and dst.stat().st_size > 0:
                    skipped += 1
                    continue
                tmp = dst.with_suffix(dst.suffix + ".tmp")
                tmp.write_bytes(zf.read(member))
                os.replace(tmp, dst)
                written += 1
                if (time.monotonic() - last_commit) >= 60.0:
                    RSNA_VOL.commit()
                    last_commit = time.monotonic()

    RSNA_VOL.commit()
    return {"dataset_id": "lumina", "neutral_files_written": written, "skipped": skipped}


def get_preprocess_fn_for_cpu_workers(cpu_workers: int = 4) -> Any:
    """Return Modal preprocess function scaled to `cpu_workers` in {2, 4, 8, 16}."""
    if int(cpu_workers) not in ALLOWED_CPU_WORKERS:
        raise ValueError(f"cpu_workers must be in {ALLOWED_CPU_WORKERS}, got {cpu_workers}")
    return preprocess_external_dataset_cpu.with_options(
        cpu=float(cpu_workers),
        memory=max(4096, int(cpu_workers) * 2048),
    )


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    cpu=4.0,
    memory=8192,
    timeout=14400,
)
def preprocess_external_dataset_cpu(
    dataset_id: str,
    manifest_csv_content: str,
    shard_idx: int = 0,
    num_shards: int = 1,
    worker_threads: int = 4,
) -> Dict[str, Any]:
    """Wave 2 CPU-only Stage-1 `1520x912` uint16 PNG cache builder with atomic resumable writes."""
    import io
    import cv2
    import numpy as np
    import pandas as pd
    from src.data.preprocessing import decode_dicom_array, extract_cropped_mammogram_with_metadata

    df = validate_inference_manifest(pd.read_csv(io.StringIO(manifest_csv_content)), context=f"preproc_{dataset_id}")
    shard_df = df.iloc[int(shard_idx) :: int(num_shards)].copy().reset_index(drop=True)

    cache_dir = RSNA_MOUNT / "cache" / f"stage1_{dataset_id}_uint16" / "png_1520x912"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_root = RSNA_MOUNT / "external_datasets" / dataset_id / "extracted"

    def _one_row(row: Dict[str, Any]) -> str:
        img_id = str(row["image_id"])
        out_png = cache_dir / f"{img_id}.png"
        if out_png.exists() and out_png.stat().st_size > 0:
            return "skipped"
        dcm_path = raw_root / str(row["image_path"])
        arr_f32, phot, lat_dcm = decode_dicom_array(dcm_path, apply_luts=True)
        eff_lat = lat_dcm if lat_dcm in ("L", "R") else str(row["laterality"])
        res = extract_cropped_mammogram_with_metadata(arr_f32, photometric=phot, laterality=eff_lat)
        u16 = np.round(np.clip(res["image_01"], 0.0, 1.0) * 65535.0).astype(np.uint16)
        ok, buf = cv2.imencode(".png", u16, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError(f"PNG encode failed for {img_id}")
        tmp = out_png.with_suffix(".png.tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, out_png)
        return "fallback" if res.get("used_fallback") else "ok"

    last_commit = time.monotonic()
    processed = 0
    skipped = 0
    fallbacks = 0
    rows = shard_df.to_dict("records")
    with ThreadPoolExecutor(max_workers=max(1, int(worker_threads))) as pool:
        for status in pool.map(_one_row, rows):
            if status == "skipped":
                skipped += 1
            else:
                processed += 1
                if status == "fallback":
                    fallbacks += 1
            if (time.monotonic() - last_commit) >= 60.0:
                RSNA_VOL.commit()
                last_commit = time.monotonic()

    RSNA_VOL.commit()
    return {
        "dataset_id": dataset_id,
        "shard_idx": int(shard_idx),
        "processed": processed,
        "skipped": skipped,
        "fallbacks": fallbacks,
    }


def _load_model_for_inference(model_id: str, spec: Dict[str, Any], device: Any) -> Any:
    """Instantiate one of the 6 RSNA-supervised architectures with strict SHA-256 verification."""
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file

    if model_id in ("ce_b5", "trust_b5"):
        from src.models.trust_model import MammoCLIPEncoder
        from src.training.engine import build_trust_head

        bb_path = RSNA_MOUNT / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
        ckpt_name = "ce_rsna_seed42_final_epoch20.pt" if model_id == "ce_b5" else "trust_rsna_seed42_final_epoch20.pt"
        ckpt_path = RSNA_MOUNT / "models" / "checkpoints" / "stage_b_rsna" / ckpt_name
        if compute_sha256_file(bb_path).lower() != spec["expected_sha256"]["backbone"]:
            raise RuntimeError(f"Mammo-CLIP backbone SHA-256 mismatch at {bb_path}")
        if compute_sha256_file(ckpt_path).lower() != spec["expected_sha256"]["checkpoint"]:
            raise RuntimeError(f"{model_id} checkpoint SHA-256 mismatch at {ckpt_path}")
        enc = MammoCLIPEncoder(checkpoint_path=str(bb_path), freeze=True).to(device).eval()
        head = build_trust_head(embed_dim=2048, dropout_rate=0.3, seed=42, device=device)
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        head.load_state_dict(payload["head_state_dict"], strict=True)
        head.eval()
        return {"encoder": enc, "head": head}

    if model_id == "dinov2_rsna":
        from src.paper1.dinov2_model import DINOv2CancerHead, build_frozen_dinov2_vits14

        bb_path = RSNA_MOUNT / "models" / "paper1" / "dinov2_vits14_pretrain.pth"
        head_path = RSNA_MOUNT / "models" / "paper1" / "checkpoints" / "dinov2_vits14_head_seed42.pt"
        if compute_sha256_file(bb_path).lower() != spec["expected_sha256"]["backbone"]:
            raise RuntimeError(f"DINOv2 backbone SHA-256 mismatch at {bb_path}")
        if compute_sha256_file(head_path).lower() != spec["expected_sha256"]["checkpoint"]:
            raise RuntimeError(f"DINOv2 head SHA-256 mismatch at {head_path}")
        enc = build_frozen_dinov2_vits14(bb_path, dinov2_repo_dir=Path("/root/external/dinov2"), device=device)
        hp = torch.load(head_path, map_location=device, weights_only=False)
        head = DINOv2CancerHead(embed_dim=int(hp.get("embed_dim", 384)), hidden_dim=int(hp.get("hidden_dim", 0)), dropout=float(hp.get("dropout", 0.3))).to(device)
        head.load_state_dict(hp["head_state_dict"], strict=True)
        head.eval()
        return {"encoder": enc, "head": head}

    if model_id == "resnet50_rsna":
        from src.paper1.resnet50_model import build_resnet50_classifier

        ckpt_path = RSNA_MOUNT / "models" / "paper1" / "checkpoints" / "resnet50_rsna_seed42_best_train_calib.pt"
        if compute_sha256_file(ckpt_path).lower() != spec["expected_sha256"]["checkpoint"]:
            raise RuntimeError(f"ResNet-50 checkpoint SHA-256 mismatch at {ckpt_path}")
        model = build_resnet50_classifier(pretrained=False, local_weights_path=None).to(device)
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        return {"model": model}

    if model_id == "gmic_rsna_finetuned":
        from src.paper1.gmic_adapter import build_gmic_model

        base_p = RSNA_MOUNT / "models" / "paper1" / "sample_model_1.p"
        ckpt_p = RSNA_MOUNT / spec["checkpoint_path_override"]
        if compute_sha256_file(ckpt_p).lower() != spec["expected_sha256"]["checkpoint"]:
            raise RuntimeError(f"GMIC fine-tuned checkpoint SHA-256 mismatch at {ckpt_p}")
        model, _ = build_gmic_model(base_p, model_index="1", device=device, gmic_repo_dir=Path("/root/external/GMIC"))
        payload = torch.load(ckpt_p, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        return {"model": model}

    if model_id == "nyu_fourview_rsna_finetuned":
        from src.paper1.nyu_fourview_adapter import build_nyu_fourview_model

        base_p = RSNA_MOUNT / "models" / "paper1" / "sample_image_model.p"
        ckpt_p = RSNA_MOUNT / spec["checkpoint_path_override"]
        if compute_sha256_file(ckpt_p).lower() != spec["expected_sha256"]["checkpoint"]:
            raise RuntimeError(f"NYU fine-tuned checkpoint SHA-256 mismatch at {ckpt_p}")
        model, mods = build_nyu_fourview_model(base_p, device=device, nyu_repo_dir=Path("/root/external/breast_cancer_classifier"))
        payload = torch.load(ckpt_p, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        return {"model": model, "mods": mods}

    raise ValueError(f"Unsupported model_id={model_id!r}")


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    ephemeral_disk=524288,
    timeout=7200,
)
def infer_external_dataset_shard_l40s(
    dataset_id: str,
    model_id: str,
    manifest_csv_content: str,
    model_spec: Dict[str, Any],
    shard_idx: int = 0,
    num_shards: int = 1,
    batch_size: int = 64,
    pilot_limit: int = 0,
    max_spend_usd: float = 10.0,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run resumable label-free L40S inference shard for one `(dataset_id, model_id)` pair."""
    import io
    import cv2
    import numpy as np
    import pandas as pd
    import torch

    RSNA_VOL.reload()
    t0 = time.perf_counter()
    df = validate_inference_manifest(pd.read_csv(io.StringIO(manifest_csv_content)), context=f"infer_{dataset_id}_{model_id}")
    if model_spec.get("requires_4view"):
        df, _ = select_common_4view_cohort(df)
    if pilot_limit > 0:
        keep_exams = sorted(df["exam_id"].unique().tolist())[: int(pilot_limit)] if model_spec.get("requires_4view") else None
        if keep_exams is not None:
            df = df[df["exam_id"].isin(set(keep_exams))].reset_index(drop=True)
        else:
            df = df.iloc[: int(pilot_limit)].reset_index(drop=True)

    if model_spec.get("requires_4view"):
        exams = sorted(df["exam_id"].unique().tolist())
        shard_exams = exams[int(shard_idx) :: int(num_shards)]
        shard_df = df[df["exam_id"].isin(set(shard_exams))].copy().reset_index(drop=True)
        expected_ids = shard_exams
    else:
        shard_df = df.iloc[int(shard_idx) :: int(num_shards)].copy().reset_index(drop=True)
        expected_ids = shard_df["image_id"].astype(str).tolist()

    ckpt_sha = str(model_spec["expected_sha256"].get("checkpoint", ""))
    preproc_ver = str(shard_df["preprocess_version"].iloc[0]) if not shard_df.empty else "none"
    shard_path = RSNA_MOUNT / "cache" / f"paper1_exp3_shards_{dataset_id}" / model_id / f"shard_{shard_idx:02d}_of_{num_shards:02d}.json"
    if resume and pilot_limit == 0:
        ok, payload, _ = validate_and_load_inference_shard(
            shard_path,
            dataset_id=dataset_id,
            model_id=model_id,
            shard_idx=shard_idx,
            num_shards=num_shards,
            checkpoint_sha256=ckpt_sha,
            preprocess_version=preproc_ver,
            expected_item_ids=expected_ids,
        )
        if ok and payload is not None:
            return {**payload, "resumed_from_cache": True}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = _load_model_for_inference(model_id, model_spec, device)
    png_dir = RSNA_MOUNT / "cache" / f"stage1_{dataset_id}_uint16" / "png_1520x912"

    records: List[Dict[str, Any]] = []
    if not model_spec.get("requires_4view"):
        from src.data.preprocessing import prepare_tensor_from_cropped
        from src.paper1.dinov2_model import prepare_dinov2_tensor_from_uint16
        from src.paper1.resnet50_model import prepare_resnet50_tensor_from_uint16

        img_ids = shard_df["image_id"].astype(str).tolist()
        for s in range(0, len(img_ids), batch_size):
            if ((time.perf_counter() - t0) / 3600.0) * L40S_HOURLY_USD > float(max_spend_usd):
                raise RuntimeError(f"Budget cap ${max_spend_usd:.2f} exceeded in shard {shard_idx}")
            b_ids = img_ids[s : s + batch_size]
            tensors = []
            for iid in b_ids:
                arr = cv2.imread(str(png_dir / f"{iid}.png"), cv2.IMREAD_UNCHANGED)
                if arr is None:
                    raise FileNotFoundError(f"Missing cached PNG: {png_dir / f'{iid}.png'}")
                if model_id in ("ce_b5", "trust_b5"):
                    tensors.append(prepare_tensor_from_cropped(arr))
                elif model_id == "dinov2_rsna":
                    tensors.append(prepare_dinov2_tensor_from_uint16(arr))
                elif model_id == "resnet50_rsna":
                    tensors.append(prepare_resnet50_tensor_from_uint16(arr))
                else:
                    from src.paper1.gmic_adapter import standard_normalize_gmic_uint16
                    norm = standard_normalize_gmic_uint16(arr)
                    tensors.append(torch.from_numpy(norm[np.newaxis, :, :].copy()))
            bx = torch.stack(tensors, dim=0).to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(model_id != "gmic_rsna_finetuned")):
                if model_id in ("ce_b5", "trust_b5", "dinov2_rsna"):
                    probs = torch.sigmoid(loaded["head"](loaded["encoder"](bx)).squeeze(1)).float().cpu().tolist()
                elif model_id == "resnet50_rsna":
                    probs = torch.sigmoid(loaded["model"](bx).squeeze(1)).float().cpu().tolist()
                else:
                    probs = loaded["model"](bx)[:, 1].float().cpu().tolist()
            for iid, p in zip(b_ids, probs):
                records.append({"item_id": iid, "score": float(p), "crop_fallback": False})
    else:
        from src.paper1.nyu_fourview_adapter import (
            NYU_TTA_EPOCHS_OFFICIAL,
            NYU_TTA_SEED_OFFICIAL,
            NYU_VIEWS,
            extract_nyu_cropped_view,
            prepare_nyu_exam_tta_tensors,
            score_nyu_fourview_exam_batch,
        )

        raw_root = RSNA_MOUNT / "external_datasets" / dataset_id / "extracted"
        nyu_mods = loaded["mods"]
        shard_df["view_slot"] = shard_df["laterality"] + "-" + shard_df["view"]
        exams = sorted(shard_df["exam_id"].unique().tolist())
        batch_exams = max(1, min(4, int(batch_size)))
        for s in range(0, len(exams), batch_exams):
            if ((time.perf_counter() - t0) / 3600.0) * L40S_HOURLY_USD > float(max_spend_usd):
                raise RuntimeError(f"Budget cap ${max_spend_usd:.2f} exceeded in NYU shard {shard_idx}")
            b_exams = exams[s : s + batch_exams]
            tta_tensors = []
            fb_counts = []
            for eid in b_exams:
                ex_rows = shard_df[shard_df["exam_id"] == eid]
                view_crops = {}
                n_fb = 0
                for vs in NYU_VIEWS:
                    match = ex_rows[ex_rows["view_slot"] == vs]
                    if len(match) != 1:
                        raise RuntimeError(f"NYU 4-view integrity error for {eid} slot {vs}: count={len(match)}")
                    dcm_p = raw_root / str(match.iloc[0]["image_path"])
                    vc = extract_nyu_cropped_view(dcm_p, view_slot=vs, nyu_mods=nyu_mods)
                    if vc.get("crop_fallback"):
                        n_fb += 1
                    view_crops[vs] = vc
                tta_tensors.append(
                    prepare_nyu_exam_tta_tensors(
                        view_crops,
                        nyu_mods=nyu_mods,
                        num_tta_epochs=NYU_TTA_EPOCHS_OFFICIAL,
                        seed=NYU_TTA_SEED_OFFICIAL,
                    )
                )
                fb_counts.append(n_fb)
            preds = score_nyu_fourview_exam_batch(loaded["model"], tta_tensors, device=device)
            for eid, pr, n_fb in zip(b_exams, preds, fb_counts):
                records.append({"item_id": eid, "score": float(pr["exam_score"]), "crop_fallback": bool(n_fb > 0)})

    save_inference_shard_atomic(
        shard_path,
        dataset_id=dataset_id,
        model_id=model_id,
        shard_idx=shard_idx,
        num_shards=num_shards,
        checkpoint_sha256=ckpt_sha,
        preprocess_version=preproc_ver,
        expected_item_ids=expected_ids,
        records=records,
    )
    RSNA_VOL.commit()
    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "shard_idx": int(shard_idx),
        "num_shards": int(num_shards),
        "n_records": len(records),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "records": records,
    }


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    ephemeral_disk=524288,
    timeout=3600,
)
def benchmark_pilot_batch_sizes_l40s(
    dataset_id: str,
    model_id: str,
    manifest_csv_content: str,
    model_spec: Dict[str, Any],
    batch_sizes: Sequence[int],
    pilot_limit: int = 200,
    max_spend_usd: float = 1.0,
) -> Dict[str, Any]:
    """Wave 3 progressive batch-size L40S pilot measuring img/s, peak VRAM, parity, and full-run cost."""
    import io
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from src.data.preprocessing import prepare_tensor_from_cropped
    from src.paper1.dinov2_model import prepare_dinov2_tensor_from_uint16
    from src.paper1.resnet50_model import prepare_resnet50_tensor_from_uint16

    RSNA_VOL.reload()
    df = validate_inference_manifest(pd.read_csv(io.StringIO(manifest_csv_content)), context=f"pilot_bs_{dataset_id}_{model_id}")
    total_dataset_imgs = int(len(df))
    sub_df = df.iloc[: int(pilot_limit)].copy().reset_index(drop=True)
    img_ids = sub_df["image_id"].astype(str).tolist()
    png_dir = RSNA_MOUNT / "cache" / f"stage1_{dataset_id}_uint16" / "png_1520x912"

    # Stage pilot PNGs onto local /tmp NVMe once so batch-size timing is clean
    nvme_dir = Path(f"/tmp/pilot_bs_{dataset_id}")
    nvme_dir.mkdir(parents=True, exist_ok=True)
    for iid in img_ids:
        src_p = png_dir / f"{iid}.png"
        dst_p = nvme_dir / f"{iid}.png"
        if not dst_p.exists():
            shutil.copy2(src_p, dst_p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = _load_model_for_inference(model_id, model_spec, device)

    def _prep_one(iid: str) -> torch.Tensor:
        arr = cv2.imread(str(nvme_dir / f"{iid}.png"), cv2.IMREAD_UNCHANGED)
        if model_id in ("ce_b5", "trust_b5"):
            return prepare_tensor_from_cropped(arr)
        if model_id == "dinov2_rsna":
            return prepare_dinov2_tensor_from_uint16(arr)
        return prepare_resnet50_tensor_from_uint16(arr)

    baseline_probs: Optional[np.ndarray] = None
    bench_records: List[Dict[str, Any]] = []
    t_wall0 = time.perf_counter()

    for bs in sorted(int(b) for b in batch_sizes):
        if ((time.perf_counter() - t_wall0) / 3600.0) * L40S_HOURLY_USD > float(max_spend_usd):
            raise RuntimeError(f"Pilot budget cap ${max_spend_usd:.2f} exceeded before batch_size={bs}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs: List[float] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for s in range(0, len(img_ids), bs):
                b_ids = img_ids[s : s + bs]
                bx = torch.stack(list(pool.map(_prep_one, b_ids)), dim=0).to(device, non_blocking=True)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                    if model_id in ("ce_b5", "trust_b5", "dinov2_rsna"):
                        p = torch.sigmoid(loaded["head"](loaded["encoder"](bx)).squeeze(1)).float().cpu().tolist()
                    else:
                        p = torch.sigmoid(loaded["model"](bx).squeeze(1)).float().cpu().tolist()
                probs.extend(p)
        if device.type == "cuda":
            torch.cuda.synchronize()
        sec = max(1e-6, time.perf_counter() - t0)
        arr_p = np.asarray(probs, dtype=np.float64)
        if baseline_probs is None:
            baseline_probs = arr_p
            max_diff = 0.0
        else:
            max_diff = float(np.max(np.abs(arr_p - baseline_probs)))
        peak_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
        img_s = len(img_ids) / sec
        proj_sec = total_dataset_imgs / max(img_s, 1e-6)
        proj_usd = (proj_sec / 3600.0) * L40S_HOURLY_USD
        bench_records.append(
            {
                "batch_size": bs,
                "n_images": len(img_ids),
                "elapsed_sec": round(sec, 3),
                "img_per_sec": round(img_s, 2),
                "peak_vram_gb": round(peak_gb, 3),
                "max_abs_diff": max_diff,
                "projected_full_dataset_sec": round(proj_sec, 1),
                "projected_full_dataset_usd": round(proj_usd, 4),
            }
        )

    rec = select_optimal_pilot_batch_size(bench_records)
    return {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "pilot_limit": int(pilot_limit),
        "total_dataset_images": total_dataset_imgs,
        **rec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 1 Exp 3 Universal Modal Inference Entrypoint.")
    parser.add_argument("--stage", required=True, choices=["dry_run", "download", "preprocess", "pilot", "infer"])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY.keys()))
    parser.add_argument("--model", default="resnet50_rsna")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--handoff-json", type=Path, default=None)
    parser.add_argument("--out-lock-dir", type=Path, default=Path("reports/paper1/expanded/exp3/locked_preds"))
    parser.add_argument("--pilot-limit", type=int, default=200)
    parser.add_argument("--pilot-batch-sizes", type=str, default=None, help="Comma-separated progressive batch sizes for --stage pilot (e.g. 16,32,64,128).")
    parser.add_argument("--cpu-workers", type=int, default=4, choices=list(ALLOWED_CPU_WORKERS), help="CPU vCPUs per Wave 2 preprocessing container (2, 4, 8, or 16).")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-spend-usd", type=float, default=5.0)
    parser.add_argument("--approve-download", action="store_true")
    parser.add_argument("--approve-preprocess", action="store_true")
    parser.add_argument("--approve-full-infer", action="store_true")
    args = parser.parse_args()

    validate_exp3_cli_guards(
        stage=args.stage,
        dataset_id=args.dataset,
        model_id=args.model,
        approve_download=args.approve_download,
        approve_preprocess=args.approve_preprocess,
        approve_full_infer=args.approve_full_infer,
        max_spend_usd=args.max_spend_usd,
        pilot_limit=args.pilot_limit,
        cpu_workers=args.cpu_workers,
        pilot_batch_sizes=args.pilot_batch_sizes,
    )
    prof = enforce_tigerwood697_profile(dry_run=(args.stage == "dry_run"))
    import pandas as pd

    inf_df = validate_inference_manifest(pd.read_csv(args.manifest_csv), context=str(args.manifest_csv))
    models = list(RSNA_SUPERVISED_MODELS) if args.model == "all" else [args.model]
    resolved_specs = {m: resolve_model_handoff(m, handoff_json=args.handoff_json) for m in models}

    if args.stage == "dry_run":
        c4v_df, _ = select_common_4view_cohort(inf_df)
        plan = {
            "profile": prof,
            "dataset": args.dataset,
            "n_images": int(len(inf_df)),
            "n_exams": int(inf_df["exam_id"].nunique()),
            "n_common_4view_exams": int(c4v_df["exam_id"].nunique()),
            "cpu_workers_per_container": int(args.cpu_workers),
            "checkpoint_sync_commands": CHECKPOINT_SYNC_COMMANDS,
            "models": {
                m: {
                    "thresholds": get_frozen_thresholds(m, {m: resolved_specs[m]["frozen_rsna_thresholds"]}),
                    "pilot_batch_sizes": parse_pilot_batch_sizes(args.pilot_batch_sizes, m),
                }
                for m in models
            },
        }
        print(json.dumps(plan, indent=2))
        return

    raise RuntimeError("Run cloud stages via `modal run scripts/run_paper1_exp3_modal.py ...` with explicit approval flags.")


if __name__ == "__main__":
    main()
