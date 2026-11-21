"""Paper 1 — Resumable NYU Four-View (Image-Only) RSNA val_calib + NLBS Modal Pipeline.

Usage (DO NOT RUN FULL INFERENCE UNTIL APPROVED):
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_nyu_modal.py --split rsna_val_calib
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/run_paper1_nyu_modal.py --split nlbs_4view
"""

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "paper1-nyu-fourview-inference"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
NLBS_VOL = modal.Volume.from_name("nlbs-mammography-vol")

RSNA_MOUNT = Path("/rsna_vol")
NLBS_MOUNT = Path("/nlbs_vol")
NYU_CKPT_SHA256 = "9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014"

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
        "datasets/paper1/rsna_val_calib_4view_selection.csv",
        remote_path="/root/datasets/paper1/rsna_val_calib_4view_selection.csv",
    )
    .add_local_file(
        "datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
        remote_path="/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
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
def infer_nyu_shard(split_name: str, shard_idx: int, num_shards: int, num_tta_epochs: int = 10) -> Dict[str, Any]:
    """Run resumable NYU image-only SplitBreastModel inference on one exam shard."""
    import pandas as pd
    import torch
    from src.evaluation.stage_c_eval import assert_no_forbidden_columns, compute_sha256_file
    from src.paper1.nyu_fourview_adapter import (
        build_nyu_fourview_model,
        extract_nyu_cropped_view,
        score_nyu_fourview_exam,
    )

    out_dir = RSNA_MOUNT / "cache" / "paper1_nyu_shards" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_file = out_dir / f"shard_{shard_idx:02d}_of_{num_shards:02d}.json"
    if shard_file.exists():
        return json.loads(shard_file.read_text(encoding="utf-8"))

    ckpt_path = RSNA_MOUNT / "models" / "paper1" / "sample_image_model.p"
    got_sha = compute_sha256_file(ckpt_path).lower()
    if got_sha != NYU_CKPT_SHA256:
        raise RuntimeError(f"NYU checkpoint SHA mismatch: {got_sha} != {NYU_CKPT_SHA256}")

    if split_name == "rsna_val_calib":
        df = pd.read_csv("/root/datasets/paper1/rsna_val_calib_4view_selection.csv")
        assert_no_forbidden_columns(df.columns, "rsna_val_calib_4view_selection")
        case_col, path_col, root_dir = "patient_id", "rel_path", RSNA_MOUNT / "rsna" / "raw"
    elif split_name == "nlbs_4view":
        df = pd.read_csv("/root/datasets/paper1/nlbs_4view_common_cohort_label_free.csv")
        assert_no_forbidden_columns(df.columns, "nlbs_4view_common_cohort_label_free")
        case_col, path_col, root_dir = "StudyInstanceUID", "dicom_rel_path", NLBS_MOUNT / "nlbs" / "extracted"
    else:
        raise ValueError(f"Unsupported split_name={split_name}")

    all_cases = sorted(df[case_col].astype(str).unique().tolist())
    shard_cases = all_cases[shard_idx::num_shards]
    df_str = df[df[case_col].astype(str).isin(set(shard_cases))].copy()
    df_str["_case"] = df_str[case_col].astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, nyu_mods = build_nyu_fourview_model(
        ckpt_path, device=device, nyu_repo_dir=Path("/root/external/breast_cancer_classifier")
    )

    t0 = time.perf_counter()
    records = []
    for idx, (cid, grp) in enumerate(df_str.groupby("_case", sort=True), start=1):
        view_crops = {}
        n_fallbacks = 0
        for r in grp.itertuples(index=False):
            vc = extract_nyu_cropped_view(root_dir / getattr(r, path_col), r.view_slot, nyu_mods)
            if vc["crop_fallback"]:
                n_fallbacks += 1
            view_crops[r.view_slot] = vc
        preds = score_nyu_fourview_exam(
            model, nyu_mods, view_crops, device=device, num_tta_epochs=num_tta_epochs, seed=0
        )
        records.append(
            {
                "case_id": cid,
                "n_crop_fallbacks": n_fallbacks,
                **preds,
            }
        )
        if idx % 100 == 0:
            print(f"[NYU {split_name} shard {shard_idx}] {idx}/{len(shard_cases)} exams", flush=True)

    payload = {
        "model_id": "nyu_fourview_pretrained",
        "checkpoint_sha256": NYU_CKPT_SHA256,
        "split_name": split_name,
        "shard_idx": shard_idx,
        "num_shards": num_shards,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "records": records,
    }
    tmp = shard_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(shard_file)
    RSNA_VOL.commit()
    return payload


@app.local_entrypoint()
def main(split: str = "rsna_val_calib", num_shards: int = 8, num_tta_epochs: int = 10) -> None:
    outs = list(infer_nyu_shard.starmap([(split, s, num_shards, num_tta_epochs) for s in range(num_shards)]))
    all_recs = [r for o in outs for r in o["records"]]
    out_dir = Path("reports/paper1/predictions_locked")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"nyu_{split}_exam_scores.json"
    out_file.write_text(json.dumps({"records": all_recs}, indent=2), encoding="utf-8")
    digest = hashlib.sha256(out_file.read_bytes()).hexdigest()
    print(f"Saved {len(all_recs)} NYU exam records to {out_file} (sha256={digest})")
