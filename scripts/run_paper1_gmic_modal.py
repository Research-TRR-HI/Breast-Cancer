"""Paper 1 — Resumable GMIC (Pretrained Model 1) RSNA val_calib + NLBS Modal Pipeline.

Design:
    - Target-label quarantined: mounts ONLY label-free NLBS manifest + RSNA val_calib manifest.
    - Phase 1 (CPU workers, zero GPU): Decode raw DICOMs -> native GMIC 2944x1920 float32/int16
      normalized arrays saved as sharded .npz/.png? Lossless uint16 cropped PNG + best_center JSON
      or pre-normalized float16 tensors on volume so GPU Phase 2 runs at full L40S speed.
    - Phase 2 (L40S): Resumable sharded inference using sample_model_1.p (malignant_pred),
      saving per-shard JSONs + RSNA_VOL.commit() and outputting locked StudyInstanceUID/patient_id scores.

Usage (DO NOT RUN FULL INFERENCE UNTIL APPROVED):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_gmic_modal.py --split rsna_val_calib
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_gmic_modal.py --split nlbs
"""

import hashlib
import json
from pathlib import Path
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


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    gpu="L40S",
    cpu=6.0,
    memory=16384,
    timeout=7200,
)
def infer_gmic_shard(split_name: str, shard_idx: int, num_shards: int) -> Dict[str, Any]:
    """Run resumable GMIC sample_model_1.p inference on one shard of RSNA val_calib or NLBS."""
    import numpy as np
    import pandas as pd
    import torch
    from src.evaluation.stage_c_eval import compute_sha256_file, load_label_free_inference_manifest
    from src.paper1.gmic_adapter import (
        GMIC_INPUT_SIZE,
        build_gmic_model,
        prepare_dicom_uint16_for_nyukat,
        preprocess_uint16_for_nyukat,
        score_gmic_preprocessed_tensor,
    )

    out_dir = RSNA_MOUNT / "cache" / "paper1_gmic_shards" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_file = out_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}.json"
    if shard_file.exists():
        return json.loads(shard_file.read_text(encoding="utf-8"))

    ckpt_path = RSNA_MOUNT / "models" / "paper1" / "sample_model_1.p"
    got_sha = compute_sha256_file(ckpt_path).lower()
    if got_sha != GMIC_CKPT_SHA256:
        raise RuntimeError(f"GMIC checkpoint SHA mismatch: {got_sha} != {GMIC_CKPT_SHA256}")

    if split_name == "rsna_val_calib":
        df = pd.read_csv("/root/datasets/rsna/splits_seed42.csv")
        df = df[df["split"] == "val_calib"].sort_values(["patient_id", "image_id"]).reset_index(drop=True)
        rows = [
            {
                "item_id": str(r.image_id),
                "case_id": str(r.patient_id),
                "view_slot": f"{r.laterality}-{r.view}",
                "dcm_path": str(RSNA_MOUNT / "rsna" / "raw" / r.rel_path),
            }
            for r in df.iloc[shard_idx::num_shards].itertuples(index=False)
        ]
    elif split_name == "nlbs":
        df = load_label_free_inference_manifest(Path("/root/datasets/nlbs/inference_manifest_stage_c.csv"))
        rows = [
            {
                "item_id": str(r.image_uid),
                "case_id": str(r.StudyInstanceUID),
                "view_slot": f"{r.laterality}-{r.view_position}",
                "dcm_path": str(NLBS_MOUNT / "nlbs" / "extracted" / r.dicom_rel_path),
            }
            for r in df.iloc[shard_idx::num_shards].itertuples(index=False)
        ]
    else:
        raise ValueError(f"Unsupported split_name={split_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, gmic_mods = build_gmic_model(
        ckpt_path, model_index="1", device=device, gmic_repo_dir=Path("/root/external/GMIC")
    )

    t0 = time.perf_counter()
    records = []
    fallback_logs = []
    for idx, row in enumerate(rows, start=1):
        arr_u16, d_meta = prepare_dicom_uint16_for_nyukat(Path(row["dcm_path"]), row["view_slot"].split("-")[0])
        proc, p_meta = preprocess_uint16_for_nyukat(arr_u16, row["view_slot"], GMIC_INPUT_SIZE, gmic_mods)
        if p_meta["crop_fallback"]:
            fallback_logs.append(
                {
                    "item_id": row["item_id"],
                    "case_id": row["case_id"],
                    "view_slot": row["view_slot"],
                    "photometric": d_meta.get("photometric", ""),
                    "fallback_strategy": p_meta.get("fallback_strategy", ""),
                    "crop_error": p_meta.get("crop_error", ""),
                }
            )
        benign, malignant = score_gmic_preprocessed_tensor(model, proc, device)
        records.append(
            {
                "item_id": row["item_id"],
                "case_id": row["case_id"],
                "view_slot": row["view_slot"],
                "benign_pred": benign,
                "malignant_pred": malignant,
                "crop_fallback": bool(p_meta["crop_fallback"]),
            }
        )
        if idx % 200 == 0:
            print(f"[GMIC {split_name} shard {shard_idx}] {idx}/{len(rows)} (fallbacks={len(fallback_logs)})", flush=True)

    payload = {
        "model_id": "gmic_pretrained",
        "checkpoint_sha256": GMIC_CKPT_SHA256,
        "split_name": split_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "n_fallback_views": len(fallback_logs),
        "fallback_logs": fallback_logs,
        "records": records,
    }
    tmp = shard_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(shard_file)
    RSNA_VOL.commit()
    return payload


@app.local_entrypoint()
def main(split: str = "rsna_val_calib", num_shards: int = 8) -> None:
    shards = list(range(num_shards))
    outs = list(infer_gmic_shard.starmap([(split, s, num_shards) for s in shards]))
    all_recs = [r for o in outs for r in o["records"]]
    all_fbs = [f for o in outs for f in o.get("fallback_logs", [])]
    out_dir = Path("reports/paper1/predictions_locked")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"gmic_{split}_image_scores.json"
    out_file.write_text(json.dumps({"records": all_recs}, indent=2), encoding="utf-8")
    fb_file = out_dir / f"gmic_{split}_fallback_log.json"
    fb_file.write_text(json.dumps({"n_images": len(all_recs), "n_fallback_views": len(all_fbs), "fallbacks": all_fbs}, indent=2), encoding="utf-8")
    digest = hashlib.sha256(out_file.read_bytes()).hexdigest()
    print(f"Saved {len(all_recs)} GMIC records (fallback_views={len(all_fbs)}) to {out_file} (sha256={digest})")
