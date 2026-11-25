"""Paper 1 Exp 3 — Universal 6-Model External Inference Locking & Per-Dataset Evaluation.

Covers all 6 RSNA-supervised models:
    - Completed & locked in Stage B / Paper 1:
        1. `ce_b5`
        2. `trust_b5`
        3. `dinov2_rsna`
        4. `resnet50_rsna`
    - Pending Agent 1 RSNA fine-tuning & `val_calib` calibration lock:
        5. `gmic_rsna_finetuned`
        6. `nyu_fourview_rsna_finetuned`

Key invariants:
    1. Strictly label-free prediction locking (`lock_external_predictions`) with SHA-256 checksums
       prior to joining protected outcome manifests.
    2. Tamper-abort verification (`verify_and_load_locked_external_predictions`) checking file
       SHA-256, frozen threshold SHA-256, threshold inequalities (`score < tau`), and absence of
       forbidden label columns.
    3. Resumable inference shard validation & atomic writes.
    4. Cluster-bootstrap 95% CIs (`cluster_id`) and one-sided 99% Clopper-Pearson bounds.
    5. Clinical-setting guardrails: NEVER pools screening cohorts (`nlbs`) with diagnostic /
       biopsy-enriched cohorts (`cmmd`, `lumina`) or mixed cohorts (`imdb`).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.stage_c_eval import compute_auprc, compute_auroc, compute_sha256_file, compute_sha256_json
from src.models.trust_model import aggregate_case_scores
from src.paper1.external_datasets import (
    DATASET_REGISTRY,
    assert_no_forbidden_columns,
    validate_inference_manifest,
    validate_outcome_manifest,
)
from src.training.calibration import evaluate_dismissal_at_threshold

RSNA_SUPERVISED_MODELS: Tuple[str, ...] = (
    "ce_b5",
    "trust_b5",
    "dinov2_rsna",
    "resnet50_rsna",
    "gmic_rsna_finetuned",
    "nyu_fourview_rsna_finetuned",
)

MODEL_HANDOFF_SPECS: Dict[str, Dict[str, Any]] = {
    "ce_b5": {
        "model_id": "ce_b5",
        "architecture": "Mammo-CLIP EfficientNet-B5 + Linear Head (CE)",
        "view_mode": "single_image",
        "requires_4view": False,
        "input_spec": "1520x912 uint16 cropped, right-mirrored, Mammo-CLIP norm",
        "required_files": [
            "models/mammo_clip/b5-model-best-epoch-7.tar",
            "models/checkpoints/stage_b_rsna/ce_rsna_seed42_final_epoch20.pt",
        ],
        "expected_sha256": {
            "backbone": "0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c",
            "checkpoint": "adf1443455ee1b507485ee4b7f64932d5cf6f59bb6afcf11b9f415c634372d22",
        },
        "frozen_rsna_thresholds": {
            "target_98": 0.0961934626102447,
            "target_95": 0.124415136873722,
        },
        "checkpoint_status": "ready_in_repo_needs_tigerwood697_sync",
    },
    "trust_b5": {
        "model_id": "trust_b5",
        "architecture": "Mammo-CLIP EfficientNet-B5 + Linear Head (TRUST)",
        "view_mode": "single_image",
        "requires_4view": False,
        "input_spec": "1520x912 uint16 cropped, right-mirrored, Mammo-CLIP norm",
        "required_files": [
            "models/mammo_clip/b5-model-best-epoch-7.tar",
            "models/checkpoints/stage_b_rsna/trust_rsna_seed42_final_epoch20.pt",
        ],
        "expected_sha256": {
            "backbone": "0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c",
            "checkpoint": "63388d790ce9cf8a8287adc7724358ced392a69b332ce1a5e46972750faa51cb",
        },
        "frozen_rsna_thresholds": {
            "target_98": 0.1795480698347091,
            "target_95": 0.1954631060361862,
        },
        "checkpoint_status": "ready_in_repo_needs_tigerwood697_sync",
    },
    "dinov2_rsna": {
        "model_id": "dinov2_rsna",
        "architecture": "DINOv2 ViT-S/14 + RSNA Head",
        "view_mode": "single_image",
        "requires_4view": False,
        "input_spec": "1520x912 uint16 -> pad square -> 518x518 3ch ImageNet norm",
        "required_files": [
            "models/paper1/dinov2_vits14_pretrain.pth",
            "reports/paper1/dinov2_head/dinov2_vits14_head_seed42.pt",
        ],
        "expected_sha256": {
            "backbone": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
            "checkpoint": "b141c2631ca8e54e5d233239823aafaf18fa3f5d8c62c755339888f41f67d5f5",
        },
        "frozen_rsna_thresholds": {
            "target_98": 0.183435708284378,
            "target_95": 0.2065157741308212,
        },
        "checkpoint_status": "ready_needs_tigerwood697_sync",
    },
    "resnet50_rsna": {
        "model_id": "resnet50_rsna",
        "architecture": "ResNet-50 (IMAGENET1K_V2 + RSNA Fine-Tune)",
        "view_mode": "single_image",
        "requires_4view": False,
        "input_spec": "1520x912 uint16 -> 854x512 3ch ImageNet norm",
        "required_files": [
            "models/paper1/resnet50-11ad3fa6.pth",
            "models/paper1/checkpoints/resnet50_rsna_seed42_best_train_calib.pt",
        ],
        "expected_sha256": {
            "backbone": "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
            "checkpoint": "ed620c61e38ce47ceac99d73fb1b02c2dae4946552e4469f5395f19b2a0c9d32",
        },
        "frozen_rsna_thresholds": {
            "target_98": 0.0009927749633789,
            "target_95": 0.0069313049316406,
        },
        "checkpoint_status": "present_on_tigerwood697",
    },
    "gmic_rsna_finetuned": {
        "model_id": "gmic_rsna_finetuned",
        "architecture": "GMIC (ResNet-18 Global+Local+Fusion, RSNA Fine-Tuned)",
        "view_mode": "single_image",
        "requires_4view": False,
        "input_spec": "Native GMIC crop 2944x1920 uint16, per-image z-score",
        "required_files": [
            "models/paper1/sample_model_1.p",
            "models/paper1/checkpoints/gmic_rsna_finetuned_best_train_calib.pt",
        ],
        "expected_sha256": {
            "base_pretrained": "3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991",
            "checkpoint": "PENDING_AGENT1_HANDOFF",
        },
        "frozen_rsna_thresholds": None,
        "checkpoint_schema_version": "paper1_gmic_finetune_ckpt_v1",
        "checkpoint_status": "pending_agent1_handoff",
    },
    "nyu_fourview_rsna_finetuned": {
        "model_id": "nyu_fourview_rsna_finetuned",
        "architecture": "NYU Four-View SplitBreastModel (RSNA Fine-Tuned)",
        "view_mode": "four_view_exam",
        "requires_4view": True,
        "input_spec": "Native NYU 4-view CC 2677x1942 / MLO 2974x1748 uint16, 10-TTA",
        "required_files": [
            "models/paper1/sample_image_model.p",
            "models/paper1/checkpoints/nyu_fourview_rsna_finetuned_best_train_calib.pt",
        ],
        "expected_sha256": {
            "base_pretrained": "9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014",
            "checkpoint": "PENDING_AGENT1_HANDOFF",
        },
        "frozen_rsna_thresholds": None,
        "checkpoint_schema_version": "paper1_nyu_finetune_ckpt_v1",
        "checkpoint_status": "pending_agent1_handoff",
    },
}


def get_frozen_thresholds(
    model_id: str,
    override_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Return frozen RSNA val_calib 98%/95% thresholds or raise if pending handoff."""
    if override_thresholds and model_id in override_thresholds:
        thr = override_thresholds[model_id]
        return {"target_98": float(thr["target_98"]), "target_95": float(thr["target_95"])}
    spec = MODEL_HANDOFF_SPECS.get(model_id)
    if spec is None:
        raise ValueError(f"Unknown model_id={model_id!r}")
    thr = spec.get("frozen_rsna_thresholds")
    if not thr:
        raise RuntimeError(
            f"Model {model_id!r} does not have frozen RSNA val_calib thresholds yet "
            f"(status={spec.get('checkpoint_status')}). Supply calibrated thresholds via handoff config."
        )
    return {"target_98": float(thr["target_98"]), "target_95": float(thr["target_95"])}


def validate_agent1_handoff_bundle(handoff_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Agent 1 handoff contract for `gmic_rsna_finetuned` / `nyu_fourview_rsna_finetuned`."""
    req_top = {"model_id", "checkpoint_path", "checkpoint_sha256", "schema_version", "rsna_val_calib_thresholds"}
    missing = sorted(req_top - set(handoff_manifest.keys()))
    if missing:
        raise ValueError(f"Agent 1 handoff manifest missing fields: {missing}")

    mid = str(handoff_manifest["model_id"])
    if mid not in ("gmic_rsna_finetuned", "nyu_fourview_rsna_finetuned"):
        raise ValueError(f"Unexpected handoff model_id={mid!r}")

    expected_schema = MODEL_HANDOFF_SPECS[mid]["checkpoint_schema_version"]
    if handoff_manifest["schema_version"] != expected_schema:
        raise ValueError(f"Schema mismatch for {mid}: got {handoff_manifest['schema_version']!r}, expected {expected_schema!r}")

    sha = str(handoff_manifest["checkpoint_sha256"]).strip().lower()
    if len(sha) != 64 or sha == "pending_agent1_handoff":
        raise ValueError(f"Invalid checkpoint_sha256 in handoff manifest for {mid}: {sha!r}")

    thr = handoff_manifest["rsna_val_calib_thresholds"]
    t98 = float(thr["target_98"])
    t95 = float(thr["target_95"])
    if not (0.0 <= t98 <= 1.0 and 0.0 <= t95 <= 1.0):
        raise ValueError(f"Thresholds out of [0, 1] for {mid}: {thr}")

    return {"model_id": mid, "checkpoint_sha256": sha, "target_98": t98, "target_95": t95, "valid": True}


def save_inference_shard_atomic(
    shard_path: Path,
    *,
    dataset_id: str,
    model_id: str,
    shard_idx: int,
    num_shards: int,
    checkpoint_sha256: str,
    preprocess_version: str,
    expected_item_ids: Sequence[str],
    records: Sequence[Dict[str, Any]],
) -> Path:
    """Atomically write one resumable inference shard JSON with provenance & item-ID hash."""
    exp_ids = [str(i) for i in expected_item_ids]
    payload = {
        "schema_version": "paper1_exp3_shard_v1",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": str(dataset_id),
        "model_id": str(model_id),
        "shard_idx": int(shard_idx),
        "num_shards": int(num_shards),
        "checkpoint_sha256": str(checkpoint_sha256).lower(),
        "preprocess_version": str(preprocess_version),
        "expected_item_ids_sha256": compute_sha256_json(exp_ids),
        "records": list(records),
    }
    shard_path = Path(shard_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, shard_path)
    return shard_path


def validate_and_load_inference_shard(
    shard_path: Path,
    *,
    dataset_id: str,
    model_id: str,
    shard_idx: int,
    num_shards: int,
    checkpoint_sha256: str,
    preprocess_version: str,
    expected_item_ids: Sequence[str],
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Validate an existing shard before resuming; reject corrupted, mismatched, or non-finite shards."""
    if not Path(shard_path).exists():
        return False, None, "missing"
    try:
        payload = json.loads(Path(shard_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return False, None, f"corrupt_json:{exc}"

    if payload.get("schema_version") != "paper1_exp3_shard_v1":
        return False, None, "schema_mismatch"
    if payload.get("dataset_id") != dataset_id or payload.get("model_id") != model_id:
        return False, None, "dataset_or_model_mismatch"
    if payload.get("shard_idx") != int(shard_idx) or payload.get("num_shards") != int(num_shards):
        return False, None, "shard_spec_mismatch"
    if str(payload.get("checkpoint_sha256", "")).lower() != str(checkpoint_sha256).lower():
        return False, None, "checkpoint_sha256_mismatch"
    if payload.get("preprocess_version") != preprocess_version:
        return False, None, "preprocess_version_mismatch"

    exp_ids = [str(i) for i in expected_item_ids]
    if payload.get("expected_item_ids_sha256") != compute_sha256_json(exp_ids):
        return False, None, "expected_item_ids_hash_mismatch"

    recs = payload.get("records")
    if not isinstance(recs, list) or len(recs) != len(exp_ids):
        return False, None, "record_count_mismatch"
    if [str(r.get("item_id", "")) for r in recs] != exp_ids:
        return False, None, "record_item_ids_mismatch"

    for r in recs:
        assert_no_forbidden_columns(r.keys(), context=f"shard_{shard_idx}_record")
        sc = r.get("score")
        if sc is None or not math.isfinite(float(sc)) or not (0.0 <= float(sc) <= 1.0):
            return False, None, "invalid_or_nonfinite_score"
    return True, payload, "ok"


def aggregate_image_records_to_exams(
    inference_df: pd.DataFrame,
    image_records_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate 100% image coverage (no missing/duplicate IDs) and max-aggregate to exam & breast levels."""
    inf_df = validate_inference_manifest(inference_df, context="aggregate_inf_manifest")
    assert_no_forbidden_columns(image_records_df.columns, context="image_records_df")
    if not {"image_id", "score"}.issubset(image_records_df.columns):
        raise ValueError("image_records_df must contain 'image_id' and 'score'")

    rec = image_records_df.copy()
    rec["image_id"] = rec["image_id"].astype(str).str.strip()
    if rec["image_id"].duplicated().any():
        dups = rec.loc[rec["image_id"].duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"Duplicate image_id in predictions: {dups}")

    exp_ids = set(inf_df["image_id"])
    got_ids = set(rec["image_id"])
    if exp_ids != got_ids:
        raise ValueError(
            f"Image coverage mismatch: missing={len(exp_ids - got_ids)}, extra={len(got_ids - exp_ids)}"
        )

    scores = pd.to_numeric(rec["score"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Predictions contain non-finite or out-of-[0,1] scores.")
    rec["score"] = scores
    if "crop_fallback" not in rec.columns:
        rec["crop_fallback"] = False

    merged = inf_df.merge(rec[["image_id", "score", "crop_fallback"]], on="image_id", how="inner", validate="one_to_one")
    case_map = aggregate_case_scores(merged["score"].tolist(), merged["exam_id"].tolist())

    exam_df = (
        merged.groupby("exam_id", sort=True)
        .agg(
            dataset_id=("dataset_id", "first"),
            cluster_id=("cluster_id", "first"),
            n_crop_fallbacks=("crop_fallback", lambda s: int(np.sum(np.asarray(s, dtype=bool)))),
        )
        .reset_index()
    )
    exam_df["score"] = exam_df["exam_id"].map(case_map).astype(float)

    merged["breast_id"] = merged["exam_id"] + "__" + merged["laterality"]
    breast_df = (
        merged.groupby("breast_id", sort=True)
        .agg(
            dataset_id=("dataset_id", "first"),
            cluster_id=("cluster_id", "first"),
            exam_id=("exam_id", "first"),
            laterality=("laterality", "first"),
            score=("score", "max"),
            n_crop_fallbacks=("crop_fallback", lambda s: int(np.sum(np.asarray(s, dtype=bool)))),
        )
        .reset_index()
    )
    return exam_df, breast_df


def lock_external_predictions(
    *,
    dataset_id: str,
    model_id: str,
    exam_scores_df: pd.DataFrame,
    expected_exam_ids: Sequence[str],
    thresholds: Dict[str, float],
    model_sha256: str,
    inference_manifest_sha256: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Cryptographically freeze label-free exam predictions and RSNA thresholds prior to outcome join."""
    assert_no_forbidden_columns(exam_scores_df.columns, context=f"lock_{dataset_id}_{model_id}")
    req = {"exam_id", "cluster_id", "score"}
    missing = sorted(req - set(exam_scores_df.columns))
    if missing:
        raise ValueError(f"exam_scores_df missing required columns: {missing}")

    df = exam_scores_df.copy()
    df["exam_id"] = df["exam_id"].astype(str).str.strip()
    df["cluster_id"] = df["cluster_id"].astype(str).str.strip()
    if df["exam_id"].duplicated().any():
        raise ValueError(f"{dataset_id}/{model_id}: duplicate exam_id rows detected.")

    exp_set = {str(x).strip() for x in expected_exam_ids}
    got_set = set(df["exam_id"])
    if exp_set != got_set:
        raise ValueError(
            f"{dataset_id}/{model_id}: exam_id completeness check failed "
            f"(missing={len(exp_set - got_set)}, extra={len(got_set - exp_set)})."
        )

    df = df.sort_values("exam_id", kind="mergesort").reset_index(drop=True)
    scores = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"{dataset_id}/{model_id}: non-finite or out-of-[0,1] exam scores.")
    df["score"] = scores

    tau98 = float(thresholds["target_98"])
    tau95 = float(thresholds["target_95"])
    df["dismissed_98"] = df["score"].values < tau98
    df["dismissed_95"] = df["score"].values < tau95
    if "n_crop_fallbacks" not in df.columns:
        df["n_crop_fallbacks"] = 0
    df["n_crop_fallbacks"] = df["n_crop_fallbacks"].astype(int)
    df["has_crop_fallback"] = df["n_crop_fallbacks"] > 0

    out_dir = Path(out_dir) / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{model_id}_locked_{dataset_id}_predictions.csv"
    manifest_path = out_dir / f"{model_id}_lock_manifest.json"

    cols = ["exam_id", "cluster_id", "score", "dismissed_98", "dismissed_95", "n_crop_fallbacks", "has_crop_fallback"]
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    df[cols].to_csv(tmp_csv, index=False)
    tmp_csv.replace(csv_path)

    pred_sha = compute_sha256_file(csv_path)
    thr_obj = {"target_98": tau98, "target_95": tau95}
    manifest = {
        "schema_version": "paper1_exp3_locked_predictions_v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "model_id": model_id,
        "model_sha256": str(model_sha256).lower(),
        "inference_manifest_sha256": str(inference_manifest_sha256).lower(),
        "frozen_rsna_thresholds": thr_obj,
        "frozen_rsna_thresholds_sha256": compute_sha256_json(thr_obj),
        "n_exams": int(len(df)),
        "n_clusters": int(df["cluster_id"].nunique()),
        "n_exams_with_crop_fallback": int(df["has_crop_fallback"].sum()),
        "predictions_csv": csv_path.name,
        "predictions_sha256": pred_sha,
    }
    tmp_json = manifest_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_json.replace(manifest_path)
    return manifest


def verify_and_load_locked_external_predictions(
    lock_dir: Path,
    dataset_id: str,
    model_id: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Verify SHA-256 digests, label quarantine, and threshold consistency before any outcome join."""
    ds_dir = Path(lock_dir) / dataset_id
    manifest_path = ds_dir / f"{model_id}_lock_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing lock manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != dataset_id or manifest.get("model_id") != model_id:
        raise RuntimeError(f"Lock manifest identity mismatch in {manifest_path}")

    csv_path = ds_dir / manifest["predictions_csv"]
    got_sha = compute_sha256_file(csv_path)
    if got_sha != manifest["predictions_sha256"]:
        raise RuntimeError(f"TAMPER DETECTED on {dataset_id}/{model_id}: {got_sha} != {manifest['predictions_sha256']}")

    thr = manifest["frozen_rsna_thresholds"]
    if compute_sha256_json(thr) != manifest["frozen_rsna_thresholds_sha256"]:
        raise RuntimeError(f"Threshold tamper detected on {dataset_id}/{model_id}")

    df = pd.read_csv(csv_path)
    assert_no_forbidden_columns(df.columns, context=f"locked_{dataset_id}_{model_id}")
    if df["exam_id"].duplicated().any():
        raise RuntimeError(f"Duplicate exam_id in locked predictions for {dataset_id}/{model_id}")

    tau98 = float(thr["target_98"])
    tau95 = float(thr["target_95"])
    scores = df["score"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError(f"Non-finite scores in locked predictions for {dataset_id}/{model_id}")
    if not np.array_equal(df["dismissed_98"].to_numpy(dtype=bool), scores < tau98):
        raise RuntimeError(f"dismissed_98 threshold inconsistency in {dataset_id}/{model_id}")
    if not np.array_equal(df["dismissed_95"].to_numpy(dtype=bool), scores < tau95):
        raise RuntimeError(f"dismissed_95 threshold inconsistency in {dataset_id}/{model_id}")

    return df, manifest


def reuse_completed_nlbs_predictions(
    six_model_lock_dir: Path,
    nlbs_exam_manifest_csv: Path,
    out_lock_dir: Path,
    model_ids: Sequence[str] = ("ce_b5", "trust_b5", "dinov2_rsna", "resnet50_rsna"),
    nlbs_4view_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Verify existing Paper 1 NLBS locked predictions and register them into Exp 3 lock layout."""
    from src.paper1.six_model_eval import verify_and_load_locked_model

    em = pd.read_csv(nlbs_exam_manifest_csv, dtype=str)
    em_dedup = em.sort_values("exam_id").drop_duplicates("StudyInstanceUID", keep="first")
    study_cluster = dict(zip(em_dedup["StudyInstanceUID"].astype(str), em_dedup["split_cluster_id"].astype(str)))
    expected_full_ids = sorted(study_cluster.keys())
    valid_4v_set = {str(x).strip() for x in nlbs_4view_ids} if nlbs_4view_ids is not None else None

    reused: Dict[str, Any] = {}
    for m in model_ids:
        df_old, man_old = verify_and_load_locked_model(six_model_lock_dir, m)
        study_ids = sorted(df_old["StudyInstanceUID"].astype(str).tolist())
        is_4v = bool(MODEL_HANDOFF_SPECS.get(m, {}).get("requires_4view", False))
        if is_4v:
            if not set(study_ids).issubset(set(expected_full_ids)):
                raise RuntimeError(f"NLBS 4-view reuse contains unknown studies for {m}")
            if valid_4v_set is not None and set(study_ids) != valid_4v_set:
                raise RuntimeError(
                    f"NLBS 4-view study mismatch for {m}: got {len(study_ids)}, expected {len(valid_4v_set)}"
                )
            if "n_studies" in man_old and len(study_ids) != int(man_old["n_studies"]):
                raise RuntimeError(f"NLBS n_studies mismatch for {m}: {len(study_ids)} != {man_old['n_studies']}")
            expected_ids = study_ids
        else:
            expected_ids = expected_full_ids

        df_new = pd.DataFrame(
            {
                "exam_id": df_old["StudyInstanceUID"].astype(str),
                "cluster_id": df_old["StudyInstanceUID"].astype(str).map(study_cluster),
                "score": df_old["score"].astype(float),
                "n_crop_fallbacks": df_old["n_crop_fallbacks"].astype(int),
            }
        )
        man_new = lock_external_predictions(
            dataset_id="nlbs",
            model_id=m,
            exam_scores_df=df_new,
            expected_exam_ids=expected_ids,
            thresholds=man_old["frozen_rsna_thresholds"],
            model_sha256=man_old["model_sha256"],
            inference_manifest_sha256=compute_sha256_file(nlbs_exam_manifest_csv),
            out_dir=out_lock_dir,
        )
        reused[m] = man_new
    return reused


def _bootstrap_cluster_metrics(
    y: np.ndarray,
    s: np.ndarray,
    clusters: np.ndarray,
    tau98: float,
    tau95: float,
    n_boot: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    uniq_c = np.array(sorted(pd.unique(clusters)))
    c_rows = {c: np.where(clusters == c)[0] for c in uniq_c}
    n_c = len(uniq_c)
    rng = np.random.default_rng(seed)

    d98 = s < tau98
    d95 = s < tau95
    vals: Dict[str, List[float]] = {k: [] for k in ("auroc", "auprc", "s98", "d98", "s95", "d95")}
    for _ in range(n_boot):
        picked_c = uniq_c[rng.integers(0, n_c, size=n_c)]
        idx = np.concatenate([c_rows[c] for c in picked_c])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        sb = s[idx]
        pos_m = yb == 1
        vals["auroc"].append(compute_auroc(yb, sb))
        vals["auprc"].append(compute_auprc(yb, sb))
        vals["s98"].append(float(np.mean(~d98[idx][pos_m])))
        vals["d98"].append(float(np.mean(d98[idx])))
        vals["s95"].append(float(np.mean(~d95[idx][pos_m])))
        vals["d95"].append(float(np.mean(d95[idx])))

    out: Dict[str, Dict[str, float]] = {}
    for k, lst in vals.items():
        if not lst:
            out[k] = {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
        else:
            arr = np.asarray(lst, dtype=np.float64)
            out[k] = {
                "mean": float(np.mean(arr)),
                "ci95_low": float(np.percentile(arr, 2.5)),
                "ci95_high": float(np.percentile(arr, 97.5)),
            }
    return out


def evaluate_external_dataset(
    *,
    dataset_id: str,
    lock_dir: Path,
    outcome_manifest_csv: Path,
    model_ids: Sequence[str],
    common_4view_exam_ids: Optional[Sequence[str]] = None,
    cohort_mode: str = "all",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify locked predictions and evaluate one external dataset with clinical-role guardrails."""
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset_id={dataset_id!r}")
    if cohort_mode not in ("all", "full", "common_4view"):
        raise ValueError(f"Unsupported cohort_mode={cohort_mode!r}; expected 'all', 'full', or 'common_4view'")
    ds_info = DATASET_REGISTRY[dataset_id]
    if cohort_mode == "common_4view" and not ds_info.get("has_4view_paired_cohort", False):
        raise ValueError(
            f"SCIENTIFIC ELIGIBILITY BLOCK: cohort_mode='common_4view' is forbidden for {dataset_id!r} "
            f"({ds_info['notes']})."
        )

    locked_dfs: Dict[str, pd.DataFrame] = {}
    thresholds_map: Dict[str, Dict[str, float]] = {}
    for m in model_ids:
        df_l, man_l = verify_and_load_locked_external_predictions(lock_dir, dataset_id, m)
        locked_dfs[m] = df_l
        thresholds_map[m] = man_l["frozen_rsna_thresholds"]

    out_df = validate_outcome_manifest(pd.read_csv(outcome_manifest_csv), context=f"eval_{dataset_id}_outcome")
    if out_df["dataset_id"].iloc[0] != dataset_id:
        raise RuntimeError(f"Outcome manifest dataset_id mismatch: {out_df['dataset_id'].iloc[0]} != {dataset_id}")

    def _eval_subset(sub_exam_ids: Optional[Sequence[str]], subset_models: Sequence[str]) -> Dict[str, Any]:
        sub_out = out_df if sub_exam_ids is None else out_df[out_df["exam_id"].isin(set(sub_exam_ids))].copy()
        sub_out = sub_out.sort_values("exam_id", kind="mergesort").reset_index(drop=True)
        merged = sub_out[["exam_id", "cluster_id", "outcome_label", "label_source", "cohort_role"]].copy()
        for m in subset_models:
            m_df = locked_dfs[m].rename(
                columns={"score": f"{m}_score", "has_crop_fallback": f"{m}_has_crop_fallback", "cluster_id": f"{m}_cluster_id"}
            )
            merged = merged.merge(m_df[["exam_id", f"{m}_score", f"{m}_has_crop_fallback", f"{m}_cluster_id"]], on="exam_id", how="inner")
            if not (merged["cluster_id"].astype(str) == merged[f"{m}_cluster_id"].astype(str)).all():
                raise RuntimeError(f"cluster_id mismatch between outcome manifest and {dataset_id}/{m} locked predictions.")

        y = merged["outcome_label"].to_numpy(dtype=np.int64)
        clusters = merged["cluster_id"].astype(str).to_numpy()
        per_model: Dict[str, Any] = {}
        for m in subset_models:
            s = merged[f"{m}_score"].to_numpy(dtype=np.float64)
            tau98 = float(thresholds_map[m]["target_98"])
            tau95 = float(thresholds_map[m]["target_95"])
            ev98 = evaluate_dismissal_at_threshold(s, y, tau98, 0.99, 0.98)
            ev95 = evaluate_dismissal_at_threshold(s, y, tau95, 0.99, 0.95)
            # Enforce clinical-role guardrail on dismissal interpretation
            if not ds_info["population_screening_dismissal_allowed"]:
                ev98["dismissal_clinical_caveat"] = (
                    f"NOT_POPULATION_SCREENING ({ds_info['cohort_role']}): "
                    "Below-threshold rate reflects non-cancer/benign score calibration only; do NOT claim screening workload reduction."
                )
                ev95["dismissal_clinical_caveat"] = ev98["dismissal_clinical_caveat"]

            per_model[m] = {
                "auroc": float(compute_auroc(y, s)),
                "auprc": float(compute_auprc(y, s)),
                "target_98": ev98,
                "target_95": ev95,
                "bootstrap_95ci": _bootstrap_cluster_metrics(y, s, clusters, tau98, tau95, n_boot=n_boot, seed=seed),
                "n_fallback_exams": int(merged[f"{m}_has_crop_fallback"].sum()),
            }
        return {
            "n_exams": int(len(merged)),
            "n_clusters": int(merged["cluster_id"].nunique()),
            "n_positive": int(y.sum()),
            "prevalence": float(np.mean(y)) if len(y) > 0 else float("nan"),
            "models": per_model,
        }

    view_agnostic_models = [m for m in model_ids if not MODEL_HANDOFF_SPECS.get(m, {}).get("requires_4view", False)]
    full_eval = _eval_subset(None, view_agnostic_models) if (cohort_mode in ("all", "full") and view_agnostic_models) else None

    c4v_eval = None
    if cohort_mode in ("all", "common_4view") and ds_info.get("has_4view_paired_cohort", False):
        if common_4view_exam_ids is not None and len(common_4view_exam_ids) > 0:
            c4v_eval = _eval_subset(common_4view_exam_ids, model_ids)

    return {
        "dataset_id": dataset_id,
        "country": ds_info["country"],
        "cohort_role": ds_info["cohort_role"],
        "cohort_mode": cohort_mode,
        "population_screening_dismissal_allowed": ds_info["population_screening_dismissal_allowed"],
        "selection_bias_caveat": ds_info.get("notes", ""),
        "license": ds_info["license"],
        "full_cohort_view_agnostic": full_eval,
        "common_4view_cohort": c4v_eval,
    }


def assert_no_cross_setting_pooling(dataset_ids: Sequence[str]) -> None:
    """Guardrail prohibiting pooling raw exams across heterogeneous clinical settings."""
    roles = {DATASET_REGISTRY[d]["cohort_role"] for d in dataset_ids if d in DATASET_REGISTRY}
    if len(roles) > 1:
        raise ValueError(
            f"CLINICAL POOLING FORBIDDEN: cannot pool datasets with distinct cohort roles {sorted(roles)}. "
            "Report per-dataset results separately."
        )


def generate_multi_dataset_report(
    per_dataset_results: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    """Aggregate per-dataset results cleanly without ever pooling screening and diagnostic exams."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "paper1_exp3_multi_dataset_eval_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pooled_analysis_performed": False,
        "pooling_policy": "Strictly per-dataset reporting; never pool screening (NLBS) with diagnostic/mixed (IMDB/CMMD/LUMINA).",
        "datasets": per_dataset_results,
    }
    json_path = out_dir / "exp3_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Paper 1 Exp 3 — Multi-Dataset External Validation Summary",
        "",
        "> **Clinical Pooling Policy:** Per-dataset reporting ONLY. Screening (`nlbs`), mixed (`imdb`), and biopsy-enriched diagnostic (`cmmd`, `lumina`) cohorts are NEVER pooled.",
        "",
    ]
    for ds_id, ds_res in per_dataset_results.items():
        lines.append(f"## Dataset: `{ds_id.upper()}` ({ds_res['country']} — `{ds_res['cohort_role']}`)")
        lines.append(f"- Screening Dismissal Claim Allowed: `{ds_res['population_screening_dismissal_allowed']}`")
        for block_name, block in (("Common 4-View Cohort", ds_res.get("common_4view_cohort")), ("Full Cohort (View-Agnostic)", ds_res.get("full_cohort_view_agnostic"))):
            if not block:
                continue
            lines.append(f"\n### {block_name} (`N={block['n_exams']}`, clusters=`{block['n_clusters']}`, pos=`{block['n_positive']}`)")
            lines.append("")
            lines.append("| Model | AUROC | AUPRC | @98% Sens | @98% Below-Thr Rate | @95% Sens | @95% Below-Thr Rate |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for m, md in block["models"].items():
                t98 = md["target_98"]
                t95 = md["target_95"]
                lines.append(
                    f"| `{m}` | {md['auroc']:.4f} | {md['auprc']:.4f} | {t98['triage_recall']*100:.2f}% ({t98['missed_cancers']} missed) | "
                    f"{t98['dismissal_rate']*100:.2f}% | {t95['triage_recall']*100:.2f}% ({t95['missed_cancers']} missed) | {t95['dismissal_rate']*100:.2f}% |"
                )
        lines.append("")

    md_path = out_dir / "exp3_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return summary
