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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.stage_c_eval import compute_sha256_file, compute_sha256_json
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
    reuse_completed_nlbs_predictions,
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
READY_MODELS: Tuple[str, ...] = ("ce_b5", "trust_b5", "dinov2_rsna", "resnet50_rsna")
AGENT1_HANDOFF_MODELS: Tuple[str, ...] = ("gmic_rsna_finetuned", "nyu_fourview_rsna_finetuned")
DEFAULT_PILOT_BATCH_SIZES: Dict[str, tuple[int, ...]] = {
    "ce_b5": (16, 32, 64),
    "trust_b5": (16, 32, 64),
    "dinov2_rsna": (32, 64, 128),
    "resnet50_rsna": (32, 64, 128),
    "gmic_rsna_finetuned": (1, 4, 8),
    "nyu_fourview_rsna_finetuned": (1, 2, 4),
}
VERIFIED_WAVE3_BATCH_SIZES: Dict[str, Dict[str, int]] = {
    "cmmd": {
        "ce_b5": 32,
        "trust_b5": 32,
        "dinov2_rsna": 32,
        "resnet50_rsna": 32,
        "gmic_rsna_finetuned": 16,
        "nyu_fourview_rsna_finetuned": 4,
    },
    "lumina": {
        "ce_b5": 32,
        "trust_b5": 32,
        "dinov2_rsna": 128,
        "resnet50_rsna": 32,
        "gmic_rsna_finetuned": 16,
    },
    "imdb": {
        "ce_b5": 32,
        "trust_b5": 32,
        "dinov2_rsna": 32,
        "resnet50_rsna": 32,
        "gmic_rsna_finetuned": 16,
        "nyu_fourview_rsna_finetuned": 4,
    },
    "nlbs": {
        "ce_b5": 32,
        "trust_b5": 32,
        "dinov2_rsna": 128,
        "resnet50_rsna": 128,
        "gmic_rsna_finetuned": 16,
        "nyu_fourview_rsna_finetuned": 4,
    },
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
    .pip_install(*PINNED_MODAL_PIP_PACKAGES, "openpyxl==3.1.5", "requests==2.32.3", "imageio==2.37.0", "tqdm==4.67.1")
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


def resolve_exp3_model_selection(model_arg: str) -> List[str]:
    """Expand model selector (`all_ready`, `agent1_handoff`, `all`, or comma-separated IDs)."""
    key = str(model_arg or "").strip()
    if key == "all_ready":
        return list(READY_MODELS)
    if key == "agent1_handoff":
        return list(AGENT1_HANDOFF_MODELS)
    if key == "all":
        return list(RSNA_SUPERVISED_MODELS)
    selected: List[str] = []
    for part in key.split(","):
        mid = part.strip()
        if not mid:
            continue
        if mid not in RSNA_SUPERVISED_MODELS:
            raise ValueError(f"Unknown model_id={mid!r}; expected {RSNA_SUPERVISED_MODELS}")
        if mid not in selected:
            selected.append(mid)
    if not selected:
        raise ValueError(f"Empty model selection: {model_arg!r}")
    return selected


def resolve_inference_batch_size(dataset_id: str, model_id: str, batch_size: int = 0) -> int:
    """Return explicit positive batch_size override or verified Wave 3 L40S winner."""
    if int(batch_size) > 0:
        return int(batch_size)
    ds_map = VERIFIED_WAVE3_BATCH_SIZES.get(dataset_id, {})
    if model_id in ds_map:
        return int(ds_map[model_id])
    return int(DEFAULT_PILOT_BATCH_SIZES.get(model_id, (32,))[0])


def enforce_dataset_model_eligibility(
    dataset_id: str,
    model_id: str,
    inf_df: Any,
    cohort_mode: str = "full",
) -> Any:
    """Enforce dataset-specific scientific eligibility rules (NEVER fabricate views).

    - HARD-BLOCKS `nyu_fourview_rsna_finetuned` on `lumina` (malignant cases discard contralateral views).
    - Automatically restricts `nyu_fourview_rsna_finetuned` on `cmmd`/`imdb`/`nlbs` to verified complete
      4-view exams via `select_common_4view_cohort`.
    - Allows optional `cohort_mode='common_4view'` slicing for single-view models on eligible datasets.
    """
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset_id={dataset_id!r}")
    if cohort_mode not in ("full", "common_4view"):
        raise ValueError(f"Unsupported cohort_mode={cohort_mode!r}; expected 'full' or 'common_4view'")

    df = validate_inference_manifest(inf_df, context=f"elig_{dataset_id}_{model_id}")
    requires_4v = bool(MODEL_HANDOFF_SPECS[model_id].get("requires_4view", False))

    if dataset_id == "lumina" and (requires_4v or cohort_mode == "common_4view"):
        raise ValueError(
            "SCIENTIFIC ELIGIBILITY BLOCK: `nyu_fourview_rsna_finetuned` and `common_4view` mode are "
            "strictly blocked on 'lumina' because malignant cases discard contralateral views (4view_pos=1/174). "
            "Never fabricate views; evaluate single-view models on LUMINA instead."
        )

    if requires_4v or cohort_mode == "common_4view":
        c4v_df, _ = select_common_4view_cohort(df)
        if c4v_df.empty:
            raise ValueError(f"{dataset_id}/{model_id}: zero complete 4-view exams found in manifest.")
        return c4v_df
    return df


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
    cohort_mode: str = "full",
) -> None:
    """Enforce zero-cost approval gates and budget caps before any cloud execution."""
    if stage not in ("dry_run", "download", "preprocess", "pilot", "infer"):
        raise ValueError(f"Unsupported stage={stage!r}")
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset_id={dataset_id!r}; expected {sorted(DATASET_REGISTRY)}")
    models = resolve_exp3_model_selection(model_id)
    if cohort_mode not in ("full", "common_4view"):
        raise ValueError(f"Unsupported cohort_mode={cohort_mode!r}")
    if int(cpu_workers) not in ALLOWED_CPU_WORKERS:
        raise ValueError(f"--cpu-workers must be one of {ALLOWED_CPU_WORKERS}, got {cpu_workers}")
    if pilot_batch_sizes is not None:
        parse_pilot_batch_sizes(pilot_batch_sizes, models[0])
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
    if entry.get("canonical_handoff_relpath"):
        spec["canonical_handoff_relpath"] = str(entry["canonical_handoff_relpath"])
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


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    cpu=1.0,
    memory=2048,
    timeout=14400,
)
def prepare_exp3_manifests_and_ingest_cpu(dataset_id: str) -> Dict[str, Any]:
    """Wave 2 CPU-only (`cpu=1.0, memory=2048`) extraction + manifest builder for LUMINA & CMMD."""
    import io
    import zipfile
    import pandas as pd
    import pydicom
    from src.paper1.external_datasets import (
        build_cmmd_external_manifests,
        build_lumina_external_manifests,
        build_lumina_pseudonym_ingest_map,
        normalize_laterality,
        normalize_view_position,
    )

    if dataset_id not in ("lumina", "cmmd"):
        raise ValueError(f"Unsupported dataset_id={dataset_id!r} for prepare_exp3_manifests_and_ingest_cpu")

    ds_root = RSNA_MOUNT / "external_datasets" / dataset_id
    raw_dir = ds_root / "raw"
    ext_root = ds_root / "extracted"
    man_dir = ds_root / "manifests"
    ext_root.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)
    last_commit = time.monotonic()

    if dataset_id == "lumina":
        # Parse Benign_Cases.xlsx and Malign_Cases.xlsx and match against zip DICOMs
        clin_rows = []
        for cls_folder, xlsx_name in (("benign", "Benign_Cases.xlsx"), ("malign", "Malign_Cases.xlsx")):
            df = pd.read_excel(raw_dir / xlsx_name)
            df.columns = [str(c).strip() for c in df.columns]
            for col in ("ID", "PATHOLOGY", "MANUFACTOR"):
                if col in df.columns:
                    df[col] = df[col].ffill()
            df = df.dropna(subset=["VIEW", "RIGHT_OR_LEFT", "PATHOLOGY"]).copy()
            df["cls_folder"] = cls_folder
            clin_rows.append(df)
        clin_all = pd.concat(clin_rows, ignore_index=True)
        clin_all["ID_str"] = clin_all["ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        clin_all["view_norm"] = clin_all["VIEW"].map(normalize_view_position)
        clin_all["path_norm"] = clin_all["PATHOLOGY"].astype(str).str.strip().str.upper()
        clin_all = clin_all[clin_all["path_norm"].isin(["BENIGN", "MALIGN", "MALIGNANT"])].copy()

        # Build lookup by (cls_folder, case_id, lat, view) -> pathology, manufacturer
        clin_lookup: Dict[tuple, Dict[str, str]] = {}
        for _, r in clin_all.iterrows():
            rl = str(r["RIGHT_OR_LEFT"]).strip().upper()
            lats = ["L", "R"] if rl in ("BILATERAL", "B", "BOTH") else [normalize_laterality(rl)]
            meta_v = {
                "pathology": r["path_norm"],
                "manufacturer": str(r.get("MANUFACTOR", "")),
            }
            for lat_c in lats:
                clin_lookup[(r["cls_folder"], r["ID_str"], lat_c, r["view_norm"])] = meta_v

        # Scan zip members & read DICOM headers to assign exact case/lat/view without path leakage
        cases_records: List[Dict[str, Any]] = []
        zip_member_bytes_cache: Dict[str, tuple] = {}
        for cls_folder, zip_name in (("benign", "LUMINA_RAW_Benign.zip"), ("malign", "LUMINA_RAW_Malign.zip")):
            zpath = raw_dir / zip_name
            with zipfile.ZipFile(zpath, "r") as zf:
                for member in zf.namelist():
                    if member.endswith("/") or not member.lower().endswith(".dcm"):
                        continue
                    parts = [p for p in member.strip("/").split("/") if p]
                    # Try to find numeric case folder id from path
                    case_candidates = [p for p in parts[:-1] if p.isdigit()]
                    case_id = case_candidates[-1] if case_candidates else (parts[-2] if len(parts) >= 2 else Path(member).stem)
                    raw_bytes = zf.read(member)
                    ds = pydicom.dcmread(io.BytesIO(raw_bytes), stop_before_pixels=True, force=True)
                    lat_raw = str(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "") or "")
                    view_raw = str(getattr(ds, "ViewPosition", "") or "")
                    # Fallback to filename hints if DICOM tag missing
                    stem_up = Path(member).stem.upper()
                    if not lat_raw:
                        lat_raw = "L" if ("L" in stem_up or "LEFT" in stem_up) else ("R" if ("R" in stem_up or "RIGHT" in stem_up) else "")
                    if not view_raw:
                        view_raw = "MLO" if "MLO" in stem_up else ("CC" if "CC" in stem_up else "")
                    try:
                        lat_norm = normalize_laterality(lat_raw)
                        view_norm = normalize_view_position(view_raw)
                    except Exception:
                        continue
                    meta = clin_lookup.get((cls_folder, str(case_id).lstrip("0") or str(case_id), lat_norm, view_norm))
                    if meta is None:
                        meta = clin_lookup.get((cls_folder, str(case_id), lat_norm, view_norm))
                    if meta is None:
                        continue
                    sop_uid = str(getattr(ds, "SOPInstanceUID", "") or Path(member).stem)
                    phot = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
                    cases_records.append(
                        {
                            "class_folder": cls_folder,
                            "case_id": str(case_id),
                            "image_id": sop_uid,
                            "laterality": lat_norm,
                            "view": view_norm,
                            "image_path": f"{zip_name}::{member}",
                            "pathology": meta["pathology"],
                            "manufacturer": meta["manufacturer"],
                            "photometric_interpretation": phot,
                        }
                    )
                    zip_member_bytes_cache[f"{zip_name}::{member}"] = (zpath, member)

        cases_df = pd.DataFrame(cases_records)
        if cases_df.empty:
            raise RuntimeError("No LUMINA DICOM records matched Benign_Cases.xlsx / Malign_Cases.xlsx!")
        ingest_map = build_lumina_pseudonym_ingest_map(cases_df)
        inf_df, out_df = build_lumina_external_manifests(cases_df)

        # Write strictly neutral paths to extracted/lumina_neutral/...
        written = 0
        skipped = 0
        for cls_folder, zip_name in (("benign", "LUMINA_RAW_Benign.zip"), ("malign", "LUMINA_RAW_Malign.zip")):
            sub = ingest_map[ingest_map["source_image_path"].str.startswith(f"{zip_name}::")]
            if sub.empty:
                continue
            with zipfile.ZipFile(raw_dir / zip_name, "r") as zf:
                for row in sub.to_dict("records"):
                    member = str(row["source_image_path"]).split("::", 1)[1]
                    dst = ext_root / str(row["neutral_image_path"])
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists() and dst.stat().st_size > 0:
                        skipped += 1
                        continue
                    tmp = dst.with_suffix(".dcm.tmp")
                    tmp.write_bytes(zf.read(member))
                    os.replace(tmp, dst)
                    written += 1
                    if (time.monotonic() - last_commit) >= 60.0:
                        RSNA_VOL.commit()
                        last_commit = time.monotonic()

        inf_df.to_csv(man_dir / "inference_manifest.csv", index=False)
        out_df.to_csv(man_dir / "outcome_manifest.csv", index=False)
        RSNA_VOL.commit()
        return {
            "dataset_id": "lumina",
            "n_images": int(len(inf_df)),
            "n_exams": int(out_df["exam_id"].nunique()),
            "written": written,
            "skipped": skipped,
        }

    # CMMD: extract series_zips/*.zip and build manifests
    clin_df = pd.read_excel(raw_dir / "CMMD_clinicaldata_revision.xlsx")
    dcm_rows: List[Dict[str, Any]] = []
    zip_files = sorted((raw_dir / "series_zips").glob("*.zip"))
    extracted_n = 0
    for zpath in zip_files:
        with zipfile.ZipFile(zpath, "r") as zf:
            for member in zf.namelist():
                if member.endswith("/") or not member.lower().endswith(".dcm"):
                    continue
                rel_p = f"{zpath.stem}/{Path(member).name}"
                dst = ext_root / rel_p
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not (dst.exists() and dst.stat().st_size > 0):
                    tmp = dst.with_suffix(".dcm.tmp")
                    tmp.write_bytes(zf.read(member))
                    os.replace(tmp, dst)
                    extracted_n += 1
                ds = pydicom.dcmread(dst, stop_before_pixels=True, force=True)
                pid = str(getattr(ds, "PatientID", "")).strip()
                lat_raw = str(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "") or "")
                view_raw = str(getattr(ds, "ViewPosition", "") or "")
                if not view_raw and hasattr(ds, "ViewCodeSequence") and len(ds.ViewCodeSequence) > 0:
                    cm = str(getattr(ds.ViewCodeSequence[0], "CodeMeaning", "") or "").lower()
                    if "cranio" in cm:
                        view_raw = "CC"
                    elif "oblique" in cm:
                        view_raw = "MLO"
                sop = str(getattr(ds, "SOPInstanceUID", "") or dst.stem)
                try:
                    lat_norm = normalize_laterality(lat_raw)
                    view_norm = normalize_view_position(view_raw) or "UNK"
                except Exception:
                    continue
                dcm_rows.append(
                    {
                        "patient_id": pid,
                        "image_id": sop,
                        "laterality": lat_norm,
                        "view": view_norm,
                        "image_path": rel_p,
                        "manufacturer": str(getattr(ds, "Manufacturer", "GE MEDICAL SYSTEMS")),
                        "manufacturer_model": str(getattr(ds, "ManufacturerModelName", "Senographe DS")),
                        "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
                    }
                )
        if (time.monotonic() - last_commit) >= 60.0:
            RSNA_VOL.commit()
            last_commit = time.monotonic()

    dcm_df = pd.DataFrame(dcm_rows)
    if dcm_df.empty:
        raise RuntimeError("No valid CMMD DICOMs extracted from series_zips!")
    inf_df, out_df = build_cmmd_external_manifests(clin_df, dcm_df)
    inf_df.to_csv(man_dir / "inference_manifest.csv", index=False)
    out_df.to_csv(man_dir / "outcome_manifest.csv", index=False)
    RSNA_VOL.commit()
    return {
        "dataset_id": "cmmd",
        "n_images": int(len(inf_df)),
        "n_exams": int(out_df["exam_id"].nunique()),
        "extracted_new": extracted_n,
    }


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
        u16 = res.cropped_uint16
        ok, buf = cv2.imencode(".png", u16, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError(f"PNG encode failed for {img_id}")
        tmp = out_png.with_suffix(".png.tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, out_png)
        return "fallback" if res.used_fallback_uncropped else "ok"

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

    def _resolve_vol_ckpt_path(spec_obj: Dict[str, Any], default_name: str) -> Path:
        cands: List[Path] = []
        for raw in (spec_obj.get("canonical_handoff_relpath"), spec_obj.get("checkpoint_path_override")):
            if not raw:
                continue
            s = str(raw).strip()
            cands.append(Path(s) if s.startswith(str(RSNA_MOUNT)) else (RSNA_MOUNT / s.lstrip("/")))
        cands.append(RSNA_MOUNT / "models" / "paper1" / "checkpoints" / default_name)
        for cand in cands:
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Missing fine-tuned checkpoint across candidates: {[str(c) for c in cands]}")

    if model_id == "gmic_rsna_finetuned":
        from src.paper1.gmic_finetune import load_gmic_finetuned_model_for_eval

        ckpt_p = _resolve_vol_ckpt_path(spec, "gmic_rsna_finetuned_best_train_calib.pt")
        model, mods, _ = load_gmic_finetuned_model_for_eval(
            ckpt_p,
            expected_sha256=spec["expected_sha256"]["checkpoint"],
            model_index="1",
            device=device,
            gmic_repo_dir=Path("/root/external/GMIC"),
        )
        return {"model": model, "mods": mods}

    if model_id == "nyu_fourview_rsna_finetuned":
        from src.paper1.nyu_finetune import load_nyu_finetuned_model_for_eval

        ckpt_p = _resolve_vol_ckpt_path(spec, "nyu_fourview_rsna_finetuned_best_train_calib.pt")
        model, mods, _ = load_nyu_finetuned_model_for_eval(
            ckpt_p,
            expected_sha256=spec["expected_sha256"]["checkpoint"],
            device=device,
            nyu_repo_dir=Path("/root/external/breast_cancer_classifier"),
        )
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

        if model_id == "gmic_rsna_finetuned":
            from src.paper1.gmic_adapter import (
                GMIC_INPUT_SIZE,
                _ensure_gmic_importable,
                prepare_dicom_uint16_for_nyukat,
                preprocess_uint16_for_nyukat,
                standard_normalize_gmic_uint16,
            )

            gmic_png_dir = RSNA_MOUNT / "cache" / "paper1_gmic_uint16_v1" / dataset_id / "png_2944x1920"
            raw_root = RSNA_MOUNT / "external_datasets" / dataset_id / "extracted"
            nyukat_mods = loaded.get("mods") or _ensure_gmic_importable(Path("/root/external/GMIC"))
            rows_list = shard_df.to_dict("records")

            def _prep_gmic_row(row: Dict[str, Any]) -> Tuple[Any, bool]:
                iid = str(row["image_id"])
                gpng = gmic_png_dir / f"{iid}.png"
                if gpng.exists():
                    arr = cv2.imread(str(gpng), cv2.IMREAD_UNCHANGED)
                    if arr is None:
                        raise FileNotFoundError(f"Corrupt GMIC PNG: {gpng}")
                    return standard_normalize_gmic_uint16(arr), False
                dcm_p = raw_root / str(row["image_path"])
                arr_u16, _ = prepare_dicom_uint16_for_nyukat(dcm_p, laterality=str(row["laterality"]))
                vs = f"{str(row['laterality']).upper()}-{str(row['view']).upper()}"
                norm, meta = preprocess_uint16_for_nyukat(
                    arr_u16,
                    view_slot=vs,
                    target_size=GMIC_INPUT_SIZE,
                    nyukat_mods=nyukat_mods,
                    normalize=True,
                )
                return norm, bool(meta.get("crop_fallback", False))

            with ThreadPoolExecutor(max_workers=8) as pool:
                for s in range(0, len(rows_list), batch_size):
                    if ((time.perf_counter() - t0) / 3600.0) * L40S_HOURLY_USD > float(max_spend_usd):
                        raise RuntimeError(f"Budget cap ${max_spend_usd:.2f} exceeded in shard {shard_idx}")
                    b_rows = rows_list[s : s + batch_size]
                    prepped = list(pool.map(_prep_gmic_row, b_rows))
                    tensors = [torch.from_numpy(norm[np.newaxis, :, :].copy()) for norm, _ in prepped]
                    fb_flags = [fb for _, fb in prepped]
                    bx = torch.stack(tensors, dim=0).to(device, non_blocking=True)
                    with torch.no_grad():
                        probs = loaded["model"](bx)[:, 1].float().cpu().tolist()
                    for row, p, fb in zip(b_rows, probs, fb_flags):
                        records.append({"item_id": str(row["image_id"]), "score": float(p), "crop_fallback": fb})
        else:
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
                bx = torch.stack(tensors, dim=0).to(device, non_blocking=True)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                    if model_id in ("ce_b5", "trust_b5", "dinov2_rsna"):
                        probs = torch.sigmoid(loaded["head"](loaded["encoder"](bx)).squeeze(1)).float().cpu().tolist()
                    else:
                        probs = torch.sigmoid(loaded["model"](bx).squeeze(1)).float().cpu().tolist()
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

        def _prep_nyu_exam(eid: str) -> Tuple[Any, int]:
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
            tta_dict = prepare_nyu_exam_tta_tensors(
                view_crops,
                nyu_mods=nyu_mods,
                num_tta_epochs=NYU_TTA_EPOCHS_OFFICIAL,
                seed=NYU_TTA_SEED_OFFICIAL,
            )
            return tta_dict, n_fb

        with ThreadPoolExecutor(max_workers=4) as pool:
            for s in range(0, len(exams), batch_exams):
                if ((time.perf_counter() - t0) / 3600.0) * L40S_HOURLY_USD > float(max_spend_usd):
                    raise RuntimeError(f"Budget cap ${max_spend_usd:.2f} exceeded in NYU shard {shard_idx}")
                b_exams = exams[s : s + batch_exams]
                prepped = list(pool.map(_prep_nyu_exam, b_exams))
                tta_tensors = [t for t, _ in prepped]
                fb_counts = [nf for _, nf in prepped]
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
        "checkpoint_sha256": ckpt_sha,
        "preprocess_version": preproc_ver,
        "expected_item_ids_sha256": compute_sha256_json([str(x) for x in expected_ids]),
        "n_records": len(records),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "resumed_from_cache": False,
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
        try:
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
        except RuntimeError as exc:
            err_msg = str(exc)
            if "out of memory" in err_msg.lower() or "canUse32BitIndexMath" in err_msg:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                bench_records.append(
                    {
                        "batch_size": bs,
                        "n_images": len(img_ids),
                        "status": "stopped_limit",
                        "error": err_msg[:200],
                        "img_per_sec": 0.0,
                        "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0,
                        "max_abs_diff": 0.0,
                    }
                )
                break
            raise
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


def _resolve_manifest_path(dataset_id: str, manifest_csv: Optional[Path] = None) -> Path:
    if manifest_csv is not None and str(manifest_csv).strip():
        return Path(manifest_csv)
    default_path = Path("datasets/exp3_meta") / f"{dataset_id}_inference_manifest.csv"
    if not default_path.exists():
        raise FileNotFoundError(
            f"Default inference manifest not found for dataset={dataset_id!r}: {default_path}. "
            "Pass --manifest-csv explicitly."
        )
    return default_path


def _expected_shard_item_ids(
    eligible_df: Any,
    *,
    requires_4view: bool,
    shard_idx: int,
    num_shards: int,
) -> List[str]:
    if requires_4view:
        exams = sorted(eligible_df["exam_id"].astype(str).unique().tolist())
        return exams[int(shard_idx) :: int(num_shards)]
    return eligible_df["image_id"].astype(str).tolist()[int(shard_idx) :: int(num_shards)]


def build_exp3_dry_run_plan(
    *,
    dataset_id: str,
    model_arg: str,
    manifest_csv: Path,
    handoff_json: Optional[Path] = None,
    cpu_workers: int = 8,
    num_shards: int = 4,
    batch_size: int = 0,
    cohort_mode: str = "full",
    pilot_batch_sizes: Optional[str] = None,
    dry_run_profile: bool = True,
) -> Dict[str, Any]:
    """Build a zero-cost Wave 1 / Phase 2 execution plan without launching any Modal containers."""
    import pandas as pd

    prof = enforce_tigerwood697_profile(dry_run=dry_run_profile)
    models = resolve_exp3_model_selection(model_arg)
    inf_df = validate_inference_manifest(pd.read_csv(manifest_csv), context=str(manifest_csv))
    c4v_df, _ = select_common_4view_cohort(inf_df)

    model_plans: Dict[str, Any] = {}
    for m_id in models:
        elig_df = enforce_dataset_model_eligibility(
            dataset_id=dataset_id,
            model_id=m_id,
            inf_df=inf_df,
            cohort_mode=cohort_mode,
        )
        spec = resolve_model_handoff(m_id, handoff_json=handoff_json)
        eff_bs = resolve_inference_batch_size(dataset_id, m_id, batch_size=batch_size)
        eff_shards = min(
            int(num_shards),
            max(1, int(elig_df["exam_id"].nunique() if spec.get("requires_4view") else len(elig_df))),
        )
        model_plans[m_id] = {
            "requires_4view": bool(spec.get("requires_4view", False)),
            "cohort_mode_applied": "common_4view" if (spec.get("requires_4view") or cohort_mode == "common_4view") else "full",
            "eligible_images": int(len(elig_df)),
            "eligible_exams": int(elig_df["exam_id"].nunique()),
            "num_shards": eff_shards,
            "inference_batch_size": eff_bs,
            "thresholds": get_frozen_thresholds(m_id, {m_id: spec["frozen_rsna_thresholds"]}),
            "pilot_batch_sizes": parse_pilot_batch_sizes(pilot_batch_sizes, m_id),
        }

    return {
        "profile": prof,
        "dataset": dataset_id,
        "manifest_csv": str(manifest_csv),
        "n_images": int(len(inf_df)),
        "n_exams": int(inf_df["exam_id"].nunique()),
        "n_common_4view_exams": int(c4v_df["exam_id"].nunique()),
        "cpu_workers_per_container": int(cpu_workers),
        "checkpoint_sync_commands": CHECKPOINT_SYNC_COMMANDS,
        "models": model_plans,
    }


def _invoke_modal_infer_shard(**kwargs: Any) -> Dict[str, Any]:
    return infer_external_dataset_shard_l40s.remote(**kwargs)


def orchestrate_external_inference(
    *,
    dataset_id: str,
    model_arg: str,
    manifest_csv: Optional[Path] = None,
    handoff_json: Optional[Path] = None,
    out_lock_dir: Path = Path("reports/paper1/expanded/exp3/locked_preds"),
    cohort_mode: str = "full",
    num_shards: int = 4,
    batch_size: int = 0,
    max_spend_usd: float = 5.0,
    resume: bool = True,
    nlbs_reuse_root: Path = Path("reports/paper1/six_model_final/locked_nlbs"),
    nlbs_exp1_reuse_root: Path = Path("reports/paper1/expanded/exp1/predictions_locked"),
    nlbs_exam_manifest_csv: Path = Path("datasets/nlbs/exam_manifest.csv"),
    shard_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    starmap_runner: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    skip_profile_check: bool = False,
) -> Dict[str, Any]:
    """Execute or resume full Phase 2 sharded inference and pre-join SHA-256 locking.

    Supports:
    - `$0` reuse verification for `nlbs` via `reuse_completed_nlbs_predictions`.
    - Local/mockable shard execution for deterministic tests (`shard_runner` / `starmap_runner`).
    - Automatic dataset/model eligibility enforcement (hard-blocking NYU 4-view on `lumina`,
      restricting NYU 4-view on `cmmd`/`imdb` to verified complete 4-view exams).
    - Strict shard completeness, duplicate/missing item rejection, image->exam aggregation,
      and SHA-256 locking via `lock_external_predictions`.
    """
    import pandas as pd

    if not skip_profile_check:
        enforce_tigerwood697_profile(dry_run=False)

    models = resolve_exp3_model_selection(model_arg)
    out_lock_dir = Path(out_lock_dir)
    out_lock_dir.mkdir(parents=True, exist_ok=True)

    if dataset_id == "nlbs":
        ready_models = [m for m in models if m in ("ce_b5", "trust_b5", "dinov2_rsna", "resnet50_rsna")]
        exp1_models = [m for m in models if m in ("gmic_rsna_finetuned", "nyu_fourview_rsna_finetuned")]
        reused: Dict[str, Any] = {}
        if ready_models:
            reused.update(
                reuse_completed_nlbs_predictions(
                    six_model_lock_dir=nlbs_reuse_root,
                    nlbs_exam_manifest_csv=nlbs_exam_manifest_csv,
                    out_lock_dir=out_lock_dir,
                    model_ids=ready_models,
                )
            )
        if exp1_models:
            reused.update(
                reuse_completed_nlbs_predictions(
                    six_model_lock_dir=nlbs_exp1_reuse_root,
                    nlbs_exam_manifest_csv=nlbs_exam_manifest_csv,
                    out_lock_dir=out_lock_dir,
                    model_ids=exp1_models,
                )
            )
        return {
            "dataset_id": "nlbs",
            "mode": "zero_cost_reuse",
            "models": reused,
        }

    resolved_manifest = _resolve_manifest_path(dataset_id, manifest_csv)
    inf_df = validate_inference_manifest(pd.read_csv(resolved_manifest), context=str(resolved_manifest))

    model_summaries: Dict[str, Any] = {}
    total_spend_est = 0.0

    for m_id in models:
        elig_df = enforce_dataset_model_eligibility(
            dataset_id=dataset_id,
            model_id=m_id,
            inf_df=inf_df,
            cohort_mode=cohort_mode,
        )
        spec = resolve_model_handoff(m_id, handoff_json=handoff_json)
        requires_4v = bool(spec.get("requires_4view", False))
        expected_ckpt_sha = str(spec["expected_sha256"].get("checkpoint", ""))
        eff_bs = resolve_inference_batch_size(dataset_id, m_id, batch_size=batch_size)

        expected_all_ids = (
            sorted(elig_df["exam_id"].astype(str).unique().tolist())
            if requires_4v
            else elig_df["image_id"].astype(str).tolist()
        )
        eff_shards = max(1, min(int(num_shards), len(expected_all_ids)))
        elig_manifest_text = elig_df.to_csv(index=False)

        per_shard_kwargs: List[Dict[str, Any]] = []
        for s_idx in range(eff_shards):
            per_shard_kwargs.append(
                {
                    "dataset_id": dataset_id,
                    "model_id": m_id,
                    "manifest_csv_content": elig_manifest_text,
                    "shard_idx": s_idx,
                    "num_shards": eff_shards,
                    "batch_size": eff_bs,
                    "model_spec": spec,
                    "pilot_limit": 0,
                    "resume": bool(resume),
                    "max_spend_usd": float(max_spend_usd),
                }
            )

        if starmap_runner is not None:
            shard_outputs = list(starmap_runner(per_shard_kwargs))
        elif shard_runner is not None:
            shard_outputs = [shard_runner(**kw) for kw in per_shard_kwargs]
        else:
            ordered_args = [
                (
                    kw["dataset_id"],
                    kw["model_id"],
                    kw["manifest_csv_content"],
                    kw["model_spec"],
                    kw["shard_idx"],
                    kw["num_shards"],
                    kw["batch_size"],
                    kw["pilot_limit"],
                    kw["max_spend_usd"],
                    kw["resume"],
                )
                for kw in per_shard_kwargs
            ]
            shard_outputs = list(infer_external_dataset_shard_l40s.starmap(ordered_args))

        if len(shard_outputs) != eff_shards:
            raise RuntimeError(
                f"{dataset_id}/{m_id}: expected {eff_shards} shard results, received {len(shard_outputs)}"
            )

        all_records: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        resumed_shards = 0
        elapsed_total = 0.0

        for s_idx, out in enumerate(shard_outputs):
            if str(out.get("checkpoint_sha256", "")) != expected_ckpt_sha:
                raise ValueError(
                    f"{dataset_id}/{m_id} shard {s_idx}: checkpoint_sha256 mismatch "
                    f"{out.get('checkpoint_sha256')!r} != {expected_ckpt_sha!r}"
                )
            exp_ids = _expected_shard_item_ids(
                elig_df, requires_4view=requires_4v, shard_idx=s_idx, num_shards=eff_shards
            )
            recs = list(out.get("records", []))
            got_ids = [str(r["item_id"]) for r in recs]
            if len(got_ids) != len(set(got_ids)):
                raise ValueError(f"{dataset_id}/{m_id} shard {s_idx}: duplicate item_ids inside shard")
            if set(got_ids) != set(exp_ids):
                missing = sorted(set(exp_ids) - set(got_ids))
                extra = sorted(set(got_ids) - set(exp_ids))
                raise ValueError(
                    f"{dataset_id}/{m_id} shard {s_idx}: item_id mismatch "
                    f"(missing={missing[:5]}, extra={extra[:5]})"
                )
            overlap = seen_ids & set(got_ids)
            if overlap:
                raise ValueError(f"{dataset_id}/{m_id}: cross-shard duplicate item_ids: {sorted(overlap)[:5]}")
            seen_ids.update(got_ids)
            all_records.extend(recs)
            if out.get("resumed_from_cache"):
                resumed_shards += 1
            else:
                elapsed_total += float(out.get("elapsed_sec", 0.0))

        if set(seen_ids) != set(expected_all_ids):
            raise ValueError(
                f"{dataset_id}/{m_id}: final aggregated coverage incomplete "
                f"({len(seen_ids)}/{len(expected_all_ids)} items)"
            )

        model_est_usd = (elapsed_total / 3600.0) * L40S_HOURLY_USD
        total_spend_est += model_est_usd
        if total_spend_est > float(max_spend_usd):
            raise RuntimeError(
                f"Cumulative estimated spend ${total_spend_est:.4f} exceeded --max-spend-usd=${max_spend_usd:.2f}"
            )

        ds_lock_dir = out_lock_dir / dataset_id
        ds_lock_dir.mkdir(parents=True, exist_ok=True)
        image_scores_path: Optional[Path] = None
        breast_scores_path: Optional[Path] = None

        if requires_4v:
            exam_cluster = elig_df.groupby("exam_id", sort=True)["cluster_id"].first().to_dict()
            exam_scores_df = pd.DataFrame(
                [
                    {
                        "exam_id": str(r["item_id"]),
                        "cluster_id": str(exam_cluster[str(r["item_id"])]),
                        "score": float(r["score"]),
                        "n_crop_fallbacks": 1 if bool(r.get("crop_fallback", False)) else 0,
                    }
                    for r in all_records
                ]
            ).sort_values("exam_id", kind="mergesort").reset_index(drop=True)
        else:
            rec_df = pd.DataFrame(all_records).rename(columns={"item_id": "image_id"})
            image_scores_df = (
                elig_df[["image_id", "exam_id", "cluster_id", "laterality", "view"]]
                .merge(rec_df[["image_id", "score", "crop_fallback"]], on="image_id", how="inner")
                .sort_values("image_id", kind="mergesort")
                .reset_index(drop=True)
            )
            exam_scores_df, breast_scores_df = aggregate_image_records_to_exams(elig_df, rec_df)
            image_scores_path = ds_lock_dir / f"{m_id}_image_scores_{dataset_id}.csv"
            image_scores_df.to_csv(image_scores_path, index=False)
            breast_scores_path = ds_lock_dir / f"{m_id}_breast_scores_{dataset_id}.csv"
            breast_scores_df.to_csv(breast_scores_path, index=False)

        thresholds = get_frozen_thresholds(m_id, {m_id: spec["frozen_rsna_thresholds"]})
        elig_inf_sha = compute_sha256_json(elig_df.to_dict(orient="records"))
        expected_exam_ids = sorted(elig_df["exam_id"].astype(str).unique().tolist())
        lock_meta = lock_external_predictions(
            dataset_id=dataset_id,
            model_id=m_id,
            exam_scores_df=exam_scores_df,
            expected_exam_ids=expected_exam_ids,
            thresholds=thresholds,
            model_sha256=expected_ckpt_sha,
            inference_manifest_sha256=elig_inf_sha,
            out_dir=out_lock_dir,
        )
        lock_meta.update(
            {
                "cohort_mode": "common_4view" if (requires_4v or cohort_mode == "common_4view") else "full",
                "num_shards": eff_shards,
                "resumed_shards": resumed_shards,
                "batch_size": eff_bs,
                "estimated_gpu_seconds": round(elapsed_total, 2),
                "estimated_gpu_usd": round(model_est_usd, 4),
                "exam_scores_csv": str(ds_lock_dir / lock_meta["predictions_csv"]),
                "image_scores_csv": str(image_scores_path) if image_scores_path is not None else None,
                "breast_scores_csv": str(breast_scores_path) if breast_scores_path is not None else None,
            }
        )
        model_summaries[m_id] = lock_meta

    summary_out = {
        "dataset_id": dataset_id,
        "cohort_mode": cohort_mode,
        "estimated_total_usd": round(total_spend_est, 4),
        "models": model_summaries,
    }
    summary_path = out_lock_dir / dataset_id / "inference_lock_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    return summary_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 1 Exp 3 Universal Modal Inference Entrypoint.")
    parser.add_argument("--stage", required=True, choices=["dry_run", "download", "preprocess", "pilot", "infer"])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY.keys()))
    parser.add_argument("--model", default="resnet50_rsna")
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--handoff-json", type=Path, default=None)
    parser.add_argument("--out-lock-dir", type=Path, default=Path("reports/paper1/expanded/exp3/locked_preds"))
    parser.add_argument("--cohort-mode", default="full", choices=["full", "common_4view"])
    parser.add_argument("--pilot-limit", type=int, default=200)
    parser.add_argument("--pilot-batch-sizes", type=str, default=None, help="Comma-separated progressive batch sizes for --stage pilot (e.g. 16,32,64,128).")
    parser.add_argument("--cpu-workers", type=int, default=4, choices=list(ALLOWED_CPU_WORKERS), help="CPU vCPUs per Wave 2 preprocessing container (2, 4, 8, or 16).")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses verified Wave 3 L40S winner batch size per dataset/model.")
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
        cohort_mode=args.cohort_mode,
    )
    resolved_manifest = _resolve_manifest_path(args.dataset, args.manifest_csv)

    if args.stage == "dry_run":
        plan = build_exp3_dry_run_plan(
            dataset_id=args.dataset,
            model_arg=args.model,
            manifest_csv=resolved_manifest,
            handoff_json=args.handoff_json,
            cpu_workers=args.cpu_workers,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            cohort_mode=args.cohort_mode,
            pilot_batch_sizes=args.pilot_batch_sizes,
            dry_run_profile=True,
        )
        print(json.dumps(plan, indent=2))
        return

    raise RuntimeError("Run cloud stages via `modal run scripts/run_paper1_exp3_modal.py ...` with explicit approval flags.")


@app.local_entrypoint()
def run_cli(
    stage: str = "dry_run",
    dataset: str = "lumina",
    model: str = "resnet50_rsna",
    manifest_csv: str = "",
    handoff_json: str = "",
    out_lock_dir: str = "reports/paper1/expanded/exp3/locked_preds",
    cohort_mode: str = "full",
    cpu_workers: int = 8,
    num_shards: int = 4,
    batch_size: int = 0,
    pilot_limit: int = 200,
    pilot_batch_sizes: str = "",
    max_spend_usd: float = 5.0,
    no_resume: bool = False,
    approve_download: bool = False,
    approve_preprocess: bool = False,
    approve_full_infer: bool = False,
) -> None:
    """Modal CLI entrypoint supporting `dry_run`, `preprocess`, `pilot`, and `infer`."""
    import pandas as pd

    validate_exp3_cli_guards(
        stage=stage,
        dataset_id=dataset,
        model_id=model,
        approve_download=approve_download,
        approve_preprocess=approve_preprocess,
        approve_full_infer=approve_full_infer,
        max_spend_usd=max_spend_usd,
        pilot_limit=pilot_limit,
        cpu_workers=cpu_workers,
        pilot_batch_sizes=pilot_batch_sizes or None,
        cohort_mode=cohort_mode,
    )
    resolved_handoff = Path(handoff_json) if str(handoff_json).strip() else None

    if stage == "dry_run":
        if dataset == "nlbs":
            print(json.dumps({"dataset": "nlbs", "mode": "zero_cost_reuse", "models": resolve_exp3_model_selection(model)}, indent=2))
            return
        resolved_manifest = _resolve_manifest_path(dataset, Path(manifest_csv) if manifest_csv.strip() else None)
        plan = build_exp3_dry_run_plan(
            dataset_id=dataset,
            model_arg=model,
            manifest_csv=resolved_manifest,
            handoff_json=resolved_handoff,
            cpu_workers=cpu_workers,
            num_shards=num_shards,
            batch_size=batch_size,
            cohort_mode=cohort_mode,
            pilot_batch_sizes=pilot_batch_sizes or None,
            dry_run_profile=True,
        )
        print(json.dumps(plan, indent=2))
        return

    enforce_tigerwood697_profile(dry_run=False)

    if stage == "preprocess":
        resolved_manifest = _resolve_manifest_path(dataset, Path(manifest_csv) if manifest_csv.strip() else None)
        validate_inference_manifest(pd.read_csv(resolved_manifest), context=str(resolved_manifest))
        manifest_text = resolved_manifest.read_text(encoding="utf-8")
        fn = get_preprocess_fn_for_cpu_workers(cpu_workers)
        outs = list(fn.starmap([(dataset, manifest_text, s, num_shards, cpu_workers) for s in range(num_shards)]))
        out_p = Path(f"reports/paper1/expanded/exp3/{dataset}_preprocess_summary.json")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(outs, indent=2), encoding="utf-8")
        print(json.dumps(outs, indent=2))
        return

    if stage == "pilot":
        resolved_manifest = _resolve_manifest_path(dataset, Path(manifest_csv) if manifest_csv.strip() else None)
        inf_df = validate_inference_manifest(pd.read_csv(resolved_manifest), context=str(resolved_manifest))
        models = resolve_exp3_model_selection(model)
        results = {}
        for m_id in models:
            elig_df = enforce_dataset_model_eligibility(dataset, m_id, inf_df, cohort_mode=cohort_mode)
            bs_list = parse_pilot_batch_sizes(pilot_batch_sizes or None, m_id)
            spec = resolve_model_handoff(m_id, handoff_json=resolved_handoff)
            res = benchmark_pilot_batch_sizes_l40s.remote(
                dataset_id=dataset,
                model_id=m_id,
                manifest_csv_content=elig_df.to_csv(index=False),
                batch_sizes=bs_list,
                pilot_limit=pilot_limit,
                model_spec=spec,
            )
            results[m_id] = res
        out_p = Path(f"reports/paper1/expanded/exp3/{dataset}_pilot_bench.json")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        return

    if stage == "infer":
        out = orchestrate_external_inference(
            dataset_id=dataset,
            model_arg=model,
            manifest_csv=Path(manifest_csv) if manifest_csv.strip() else None,
            handoff_json=resolved_handoff,
            out_lock_dir=Path(out_lock_dir),
            cohort_mode=cohort_mode,
            num_shards=num_shards,
            batch_size=batch_size,
            max_spend_usd=max_spend_usd,
            resume=not bool(no_resume),
            skip_profile_check=True,
        )
        print(json.dumps(out, indent=2))
        return

    raise ValueError(f"Unsupported run_cli stage={stage!r}; use 'dry_run', 'preprocess', 'pilot', or 'infer'")


if __name__ == "__main__":
    main()
