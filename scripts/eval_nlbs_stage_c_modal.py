"""Stage C Frozen GPU Inference & Prediction Locking on NLBS (Target-Label Quarantined).

Security / Quarantine Design:
    - The Modal container mounts ONLY `src/` and `datasets/nlbs/inference_manifest_stage_c.csv`.
    - NLBS label files (`manifest.csv`, `exam_manifest.csv`, metadata CSVs) are NEVER mounted
      into the cloud image or read during inference.
    - Both frozen Stage-B checkpoints (CE-only and TRUST final epoch 20) share the identical
      frozen Mammo-CLIP EfficientNet-B5 backbone and exact TRUST Sec. 3.2 preprocessing,
      so each NLBS DICOM is decoded/cropped/encoded once and scored by both heads in AMP.
    - Predictions, frozen-threshold dismissal flags, and SHA-256 checksums are locked to
      `reports/stage_c/` before any separate evaluation script touches `exam_manifest.csv`.

Usage:
    # 1. 20-image real NLBS pilot (10 GE + 10 Siemens DICOMs)
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/eval_nlbs_stage_c_modal.py --pilot 20

    # 2. Full 26,988-image NLBS frozen inference + prediction lock across 4 L40S workers
    $env:MODAL_PROFILE="hasinishrak2015"; modal run scripts/eval_nlbs_stage_c_modal.py --num-shards 4
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.env_metadata import PINNED_MODAL_PIP_PACKAGES, PINNED_MODAL_PYTHON_VERSION

APP_NAME = "anchorshift-stage-c-nlbs-eval"
NLBS_VOL = modal.Volume.from_name("nlbs-mammography-vol")
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")

NLBS_MOUNT = Path("/nlbs_vol")
RSNA_MOUNT = Path("/rsna_vol")
NLBS_EXTRACTED_DIR = NLBS_MOUNT / "nlbs" / "extracted"
CONTAINER_MANIFEST_STR = "/root/inference_manifest_stage_c.csv"
LOCAL_MANIFEST_PATH = Path("datasets/nlbs/inference_manifest_stage_c.csv")
SUPPORTED_GPUS = ("T4", "L4", "A10G", "L40S", "A100", "H100")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PINNED_MODAL_PYTHON_VERSION)
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(*PINNED_MODAL_PIP_PACKAGES)
    .add_local_python_source("src")
    .add_local_file(
        str(LOCAL_MANIFEST_PATH),
        remote_path=CONTAINER_MANIFEST_STR,
    )
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@app.function(
    image=image,
    volumes={
        str(NLBS_MOUNT): NLBS_VOL,
        str(RSNA_MOUNT): RSNA_VOL,
    },
    gpu="L40S",
    cpu=8.0,
    memory=24576,
    timeout=14400,
)
def run_nlbs_inference_shard(
    shard_idx: int = 0,
    num_shards: int = 1,
    target_image_uids: Optional[List[str]] = None,
    decode_batch_size: int = 80,
    micro_batch_size: int = 20,
    num_workers: int = 8,
    precision: str = "amp",
    save_shard_to_vol: bool = True,
) -> Dict[str, Any]:
    """Run deterministic Stage-1 + Stage-2 + frozen CE & TRUST inference on one NLBS shard."""
    import numpy as np
    import torch

    from src.data.preprocessing import (
        decode_dicom_array,
        extract_cropped_mammogram_with_metadata,
        get_preprocessing_config_hash,
        get_stage1_config_hash,
        prepare_tensor_from_cropped,
    )
    from src.evaluation.stage_c_eval import (
        FROZEN_BACKBONE_SHA256,
        FROZEN_STAGE_B_CHECKPOINT_SHA256,
        load_label_free_inference_manifest,
    )
    from src.models.mammo_clip_loader import load_and_verify_mammo_clip
    from src.models.trust_model import MammoCLIPEncoder
    from src.training.engine import build_trust_head
    from src.utils.env_metadata import collect_runtime_environment_metadata

    # 1. Verify strictly label-free inference manifest (aborts if any label column exists)
    inf_df = load_label_free_inference_manifest(Path(CONTAINER_MANIFEST_STR))
    if target_image_uids is not None:
        uid_set = set(target_image_uids)
        shard_df = inf_df[inf_df["image_uid"].isin(uid_set)].copy().reset_index(drop=True)
    else:
        shard_df = inf_df.iloc[int(shard_idx) :: int(num_shards)].copy().reset_index(drop=True)

    # Check if shard checkpoint already committed on RSNA_VOL
    shard_cache_path = RSNA_MOUNT / "cache" / "nlbs_stage_c_shards" / f"shard_{shard_idx:02d}_of_{num_shards:02d}.json"
    if save_shard_to_vol and target_image_uids is None and shard_cache_path.exists():
        cached_payload = json.loads(shard_cache_path.read_text(encoding="utf-8"))
        cached_uids = [r["image_uid"] for r in cached_payload.get("records", [])]
        if cached_uids == shard_df["image_uid"].tolist():
            print(f"[shard {shard_idx}/{num_shards}] Reusing verified cached shard output ({len(cached_uids)} images).", flush=True)
            return cached_payload

    # 2. Verify frozen Mammo-CLIP backbone + frozen Stage-B CE & TRUST checkpoints strictly by SHA-256
    mammo_clip_tar = RSNA_MOUNT / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    ce_ckpt_path = RSNA_MOUNT / "models" / "checkpoints" / "ce_rsna_seed42_final_epoch20.pt"
    trust_ckpt_path = RSNA_MOUNT / "models" / "checkpoints" / "trust_rsna_seed42_final_epoch20.pt"

    for p, expected_sha, label in [
        (mammo_clip_tar, FROZEN_BACKBONE_SHA256, "Mammo-CLIP B5"),
        (ce_ckpt_path, FROZEN_STAGE_B_CHECKPOINT_SHA256["ce"], "CE Epoch-20"),
        (trust_ckpt_path, FROZEN_STAGE_B_CHECKPOINT_SHA256["trust"], "TRUST Epoch-20"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing frozen artifact {label} at {p}")
        got_sha = _sha256_file(p)
        if got_sha != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch for {label} ({p}): got={got_sha} != expected={expected_sha}")

    expected_p_hash = get_preprocessing_config_hash()
    expected_s1_hash = get_stage1_config_hash()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = precision == "amp" and device.type == "cuda"

    encoder = MammoCLIPEncoder(checkpoint_path=None, freeze=False, model_name="efficientnet-b5")
    diag = load_and_verify_mammo_clip(encoder, mammo_clip_tar, strict=True, verify_sha256=True)
    if not (diag["is_fully_compatible"] and diag["matched_keys_count"] == 852 and diag["total_target_keys"] == 852):
        raise RuntimeError(f"Mammo-CLIP strict key check failed: {diag}")
    encoder.freeze_encoder()
    encoder = encoder.to(device)
    encoder.eval()

    heads = {}
    ckpt_infos = {}
    for b_key, cpath in (("ce", ce_ckpt_path), ("trust", trust_ckpt_path)):
        ck = torch.load(str(cpath), map_location="cpu", weights_only=False)
        cfg = ck["training_config"]
        if cfg["baseline"] != b_key:
            raise ValueError(f"Expected baseline={b_key}, got {cfg['baseline']}")
        if cfg["preprocess_config_hash"] != expected_p_hash:
            raise ValueError(f"Preprocess hash mismatch in {b_key}: {cfg['preprocess_config_hash']} != {expected_p_hash}")
        if cfg["stage1_config_hash"] != expected_s1_hash:
            raise ValueError(f"Stage-1 hash mismatch in {b_key}: {cfg['stage1_config_hash']} != {expected_s1_hash}")
        if cfg["model_checkpoint_sha256"] != FROZEN_BACKBONE_SHA256:
            raise ValueError(f"Backbone SHA mismatch in {b_key}")
        head = build_trust_head(embed_dim=2048, dropout_rate=0.3, seed=int(cfg["seed"]), device=device)
        head.load_state_dict(ck["head_state_dict"], strict=True)
        head.eval()
        heads[b_key] = head
        ckpt_infos[b_key] = {
            "epoch": int(ck["epoch"]),
            "tau_t": float(ck["tau_t"]),
            "sha256": FROZEN_STAGE_B_CHECKPOINT_SHA256[b_key],
            "cache_manifest_sha256": str(cfg["cache_manifest_sha256"]),
        }

    env_meta = collect_runtime_environment_metadata(precision_policy=precision)
    print(
        f"[shard {shard_idx}/{num_shards}] GPU={env_meta['gpu_name']} | images={len(shard_df)} | precision={precision}",
        flush=True,
    )

    def _preprocess_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
        uid = str(row_dict["image_uid"])
        rel_path = str(row_dict["dicom_rel_path"])
        manifest_lat = str(row_dict["laterality"]).strip().upper()
        manifest_phot = str(row_dict["photometric_interpretation"]).strip().upper()
        dcm_path = NLBS_EXTRACTED_DIR / rel_path
        try:
            if not dcm_path.exists():
                raise FileNotFoundError(f"Missing DICOM file at {dcm_path}")
            arr_f32, phot_dcm, lat_dcm = decode_dicom_array(dcm_path, apply_luts=True)
            eff_phot = phot_dcm or manifest_phot
            eff_lat = lat_dcm if lat_dcm in ("L", "R") else manifest_lat
            s1 = extract_cropped_mammogram_with_metadata(
                arr=arr_f32,
                photometric=eff_phot,
                laterality=eff_lat,
                invert_monochrome1=True,
                mirror_right=True,
                laterality_policy="strict",
                crop_breast=True,
                crop_margin_frac=0.10,
                segmentation_threshold_eps=1e-5,
                use_largest_cc=False,
                strict_segmentation=False,
                simulate_16bit_png=True,
                context_label=uid,
            )
            tensor = prepare_tensor_from_cropped(
                cropped=s1.cropped_uint16,
                target_size=(1520, 912),
                augment_params=None,
                standardize=True,
            )
            return {
                "image_uid": uid,
                "decode_status": "ok",
                "segmentation_status": str(s1.segmentation_status),
                "used_fallback": bool(s1.used_fallback_uncropped),
                "border_median": float(s1.border_median),
                "laterality_status": f"{s1.laterality_status}|dcm_lat={lat_dcm or 'EMPTY'}",
                "tensor": tensor,
            }
        except Exception as exc:
            return {
                "image_uid": uid,
                "decode_status": f"ERROR:{type(exc).__name__}:{exc}",
                "segmentation_status": "decode_failed",
                "used_fallback": False,
                "border_median": float("nan"),
                "laterality_status": "error",
                "tensor": None,
            }

    rows_list = shard_df.to_dict("records")
    records_out: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for b_start in range(0, len(rows_list), decode_batch_size):
            batch_rows = rows_list[b_start : b_start + decode_batch_size]
            prep_items = list(pool.map(_preprocess_row, batch_rows))

            valid_items = [it for it in prep_items if it["tensor"] is not None]
            scores_by_uid: Dict[str, tuple] = {}

            if valid_items:
                for mb_start in range(0, len(valid_items), micro_batch_size):
                    mb = valid_items[mb_start : mb_start + micro_batch_size]
                    bx = torch.stack([it["tensor"] for it in mb], dim=0).to(device, non_blocking=True)
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                        feats = encoder(bx)
                        if not torch.isfinite(feats).all():
                            raise ValueError("Non-finite Mammo-CLIP encoder features produced during NLBS inference.")
                        p_ce = torch.sigmoid(heads["ce"](feats)).float().squeeze(1).cpu().numpy()
                        p_tr = torch.sigmoid(heads["trust"](feats)).float().squeeze(1).cpu().numpy()
                    for idx_in_mb, it in enumerate(mb):
                        sc_ce = float(p_ce[idx_in_mb])
                        sc_tr = float(p_tr[idx_in_mb])
                        if not (np.isfinite(sc_ce) and np.isfinite(sc_tr)):
                            raise ValueError(f"Non-finite prediction for {it['image_uid']}: ce={sc_ce}, trust={sc_tr}")
                        scores_by_uid[it["image_uid"]] = (sc_ce, sc_tr)

            for it in prep_items:
                uid = it["image_uid"]
                sc_ce, sc_tr = scores_by_uid.get(uid, (float("nan"), float("nan")))
                records_out.append(
                    {
                        "image_uid": uid,
                        "decode_status": it["decode_status"],
                        "segmentation_status": it["segmentation_status"],
                        "used_fallback": bool(it["used_fallback"]),
                        "border_median": it["border_median"],
                        "laterality_status": it["laterality_status"],
                        "score_ce": sc_ce,
                        "score_trust": sc_tr,
                    }
                )

            done = b_start + len(batch_rows)
            if done % (decode_batch_size * 5) == 0 or done == len(rows_list):
                dt = max(time.perf_counter() - t0, 1e-6)
                rate = done / dt
                eta_min = (len(rows_list) - done) / max(rate, 1e-6) / 60.0
                print(
                    f"[shard {shard_idx}/{num_shards}] {done:,}/{len(rows_list):,} "
                    f"({100.0 * done / max(len(rows_list), 1):5.1f}%) | {rate:5.2f} img/s | ETA {eta_min:4.1f}m",
                    flush=True,
                )

    payload = {
        "shard_idx": int(shard_idx),
        "num_shards": int(num_shards),
        "n_images": int(len(records_out)),
        "elapsed_sec": float(time.perf_counter() - t0),
        "checkpoint_metadata": {
            "backbone_sha256": FROZEN_BACKBONE_SHA256,
            "preprocess_config_hash": expected_p_hash,
            "stage1_config_hash": expected_s1_hash,
            "precision": precision,
            "ce_checkpoint_sha256": ckpt_infos["ce"]["sha256"],
            "ce_epoch": ckpt_infos["ce"]["epoch"],
            "ce_cache_manifest_sha256": ckpt_infos["ce"]["cache_manifest_sha256"],
            "trust_checkpoint_sha256": ckpt_infos["trust"]["sha256"],
            "trust_epoch": ckpt_infos["trust"]["epoch"],
            "trust_tau_t_final": ckpt_infos["trust"]["tau_t"],
            "trust_cache_manifest_sha256": ckpt_infos["trust"]["cache_manifest_sha256"],
        },
        "environment_metadata": env_meta,
        "records": records_out,
    }
    if save_shard_to_vol and target_image_uids is None:
        shard_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = shard_cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(shard_cache_path)
        RSNA_VOL.commit()
    return payload


@app.function(
    image=image,
    volumes={
        str(NLBS_MOUNT): NLBS_VOL,
        str(RSNA_MOUNT): RSNA_VOL,
    },
    cpu=2.0,
    memory=4096,
    timeout=3600,
    max_containers=16,
)
def build_nlbs_stage1_1520_shard(
    shard_idx: int,
    num_shards: int = 32,
) -> Dict[str, Any]:
    """CPU worker: decode NLBS DICOMs, run TRUST Stage-1 crop + 1520x912 resize, save label-free uint16 PNGs."""
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    from src.data.preprocessing import decode_dicom_array, extract_cropped_mammogram_with_metadata
    from src.evaluation.stage_c_eval import load_label_free_inference_manifest

    inf_df = load_label_free_inference_manifest(Path(CONTAINER_MANIFEST_STR))
    shard_df = inf_df.iloc[int(shard_idx) :: int(num_shards)].copy().reset_index(drop=True)

    stage1_dir = RSNA_MOUNT / "cache" / "stage1_nlbs_uint16"
    png_dir = stage1_dir / "png_1520x912"
    diag_dir = stage1_dir / "shard_diags"
    png_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    diag_file = diag_dir / f"stage1_shard_{shard_idx:02d}_of_{num_shards:02d}.json"
    if diag_file.exists():
        cached = json.loads(diag_file.read_text(encoding="utf-8"))
        if len(cached.get("records", [])) == len(shard_df):
            all_ok = all((png_dir / f"{r['image_uid']}.png").exists() for r in cached["records"] if r["decode_status"] == "ok")
            if all_ok:
                print(f"[stage1 {shard_idx}/{num_shards}] Reusing verified Stage-1 shard ({len(shard_df)} imgs).", flush=True)
                return cached

    t0 = time.perf_counter()
    records: List[Dict[str, Any]] = []
    for i, row in enumerate(shard_df.itertuples(index=False), start=1):
        uid = str(row.image_uid)
        rel_path = str(row.dicom_rel_path)
        manifest_lat = str(row.laterality).strip().upper()
        manifest_phot = str(row.photometric_interpretation).strip().upper()
        dcm_path = NLBS_EXTRACTED_DIR / rel_path
        out_png = png_dir / f"{uid}.png"

        try:
            if not dcm_path.exists():
                raise FileNotFoundError(f"Missing DICOM file at {dcm_path}")
            arr_f32, phot_dcm, lat_dcm = decode_dicom_array(dcm_path, apply_luts=True)
            eff_phot = phot_dcm or manifest_phot
            eff_lat = lat_dcm if lat_dcm in ("L", "R") else manifest_lat
            s1 = extract_cropped_mammogram_with_metadata(
                arr=arr_f32,
                photometric=eff_phot,
                laterality=eff_lat,
                invert_monochrome1=True,
                mirror_right=True,
                laterality_policy="strict",
                crop_breast=True,
                crop_margin_frac=0.10,
                segmentation_threshold_eps=1e-5,
                use_largest_cc=False,
                strict_segmentation=False,
                simulate_16bit_png=True,
                context_label=uid,
            )

            # Match RSNA resize_stage1_png_worker exactly: bilinear (1520, 912) -> min-max [0,1] -> uint16
            t = torch.from_numpy(s1.cropped_uint16.astype(np.float32) / 65535.0).unsqueeze(0).unsqueeze(0)
            if tuple(s1.cropped_uint16.shape) != (1520, 912):
                t = F.interpolate(t, size=(1520, 912), mode="bilinear", align_corners=False)
            t = t.squeeze(0).squeeze(0)
            mn = float(t.min().item())
            mx = float(t.max().item())
            t = (t - mn) / (mx - mn) if mx > mn else torch.zeros_like(t)
            out_u16 = np.round(t.clamp(0.0, 1.0).numpy() * 65535.0).astype(np.uint16)

            tmp_png = png_dir / f"{uid}.tmp.png"
            if not cv2.imwrite(str(tmp_png), out_u16):
                raise RuntimeError(f"cv2.imwrite failed for {out_png}")
            tmp_png.replace(out_png)

            records.append(
                {
                    "image_uid": uid,
                    "decode_status": "ok",
                    "segmentation_status": str(s1.segmentation_status),
                    "used_fallback": bool(s1.used_fallback_uncropped),
                    "border_median": float(s1.border_median),
                    "laterality_status": f"{s1.laterality_status}|dcm_lat={lat_dcm or 'EMPTY'}",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "image_uid": uid,
                    "decode_status": f"ERROR:{type(exc).__name__}:{exc}",
                    "segmentation_status": "decode_failed",
                    "used_fallback": False,
                    "border_median": float("nan"),
                    "laterality_status": "error",
                }
            )

        if i % 250 == 0 or i == len(shard_df):
            RSNA_VOL.commit()
            print(f"[stage1 {shard_idx}/{num_shards}] {i}/{len(shard_df)} ({100.0 * i / len(shard_df):.1f}%)", flush=True)

    payload = {
        "shard_idx": int(shard_idx),
        "num_shards": int(num_shards),
        "elapsed_sec": float(time.perf_counter() - t0),
        "records": records,
    }
    tmp_diag = diag_file.with_suffix(".json.tmp")
    tmp_diag.write_text(json.dumps(payload), encoding="utf-8")
    tmp_diag.replace(diag_file)
    RSNA_VOL.commit()
    return payload


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
    gpu="L40S",
    cpu=8.0,
    memory=24576,
    ephemeral_disk=524288,
    timeout=14400,
)
def score_nlbs_from_stage1_l40s(
    stage1_records: List[Dict[str, Any]],
    micro_batch_size: int = 20,
    encode_batch_size: int = 80,
    num_workers: int = 8,
    precision: str = "amp",
) -> Dict[str, Any]:
    """GPU worker: stage 1520x912 NLBS uint16 PNGs to local NVMe, score CE & TRUST heads in AMP."""
    import shutil
    import cv2
    import numpy as np
    import torch

    from src.data.preprocessing import get_preprocessing_config_hash, get_stage1_config_hash, prepare_tensor_from_cropped
    from src.evaluation.stage_c_eval import (
        FROZEN_BACKBONE_SHA256,
        FROZEN_STAGE_B_CHECKPOINT_SHA256,
        load_label_free_inference_manifest,
    )
    from src.models.mammo_clip_loader import load_and_verify_mammo_clip
    from src.models.trust_model import MammoCLIPEncoder
    from src.training.engine import build_trust_head
    from src.utils.env_metadata import collect_runtime_environment_metadata

    inf_df = load_label_free_inference_manifest(Path(CONTAINER_MANIFEST_STR))
    s1_map = {str(r["image_uid"]): r for r in stage1_records}
    if set(s1_map.keys()) != set(inf_df["image_uid"].astype(str)):
        raise RuntimeError("Stage-1 diagnostics do not match inference_manifest_stage_c image_uids.")

    mammo_clip_tar = RSNA_MOUNT / "models" / "mammo_clip" / "b5-model-best-epoch-7.tar"
    ce_ckpt_path = RSNA_MOUNT / "models" / "checkpoints" / "ce_rsna_seed42_final_epoch20.pt"
    trust_ckpt_path = RSNA_MOUNT / "models" / "checkpoints" / "trust_rsna_seed42_final_epoch20.pt"

    for p, expected_sha, label in [
        (mammo_clip_tar, FROZEN_BACKBONE_SHA256, "Mammo-CLIP B5"),
        (ce_ckpt_path, FROZEN_STAGE_B_CHECKPOINT_SHA256["ce"], "CE Epoch-20"),
        (trust_ckpt_path, FROZEN_STAGE_B_CHECKPOINT_SHA256["trust"], "TRUST Epoch-20"),
    ]:
        got_sha = _sha256_file(p)
        if got_sha != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch for {label} ({p}): got={got_sha} != expected={expected_sha}")

    expected_p_hash = get_preprocessing_config_hash()
    expected_s1_hash = get_stage1_config_hash()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = precision == "amp" and device.type == "cuda"

    encoder = MammoCLIPEncoder(checkpoint_path=None, freeze=False, model_name="efficientnet-b5")
    diag = load_and_verify_mammo_clip(encoder, mammo_clip_tar, strict=True, verify_sha256=True)
    if not (diag["is_fully_compatible"] and diag["matched_keys_count"] == 852 and diag["total_target_keys"] == 852):
        raise RuntimeError(f"Mammo-CLIP strict key check failed: {diag}")
    encoder.freeze_encoder()
    encoder = encoder.to(device)
    encoder.eval()

    heads = {}
    ckpt_infos = {}
    for b_key, cpath in (("ce", ce_ckpt_path), ("trust", trust_ckpt_path)):
        ck = torch.load(str(cpath), map_location="cpu", weights_only=False)
        cfg = ck["training_config"]
        if cfg["preprocess_config_hash"] != expected_p_hash or cfg["stage1_config_hash"] != expected_s1_hash:
            raise RuntimeError(f"Hash mismatch inside {b_key} checkpoint.")
        head = build_trust_head(embed_dim=2048, dropout_rate=0.3, seed=int(cfg["seed"]), device=device)
        head.load_state_dict(ck["head_state_dict"], strict=True)
        head.eval()
        heads[b_key] = head
        ckpt_infos[b_key] = {
            "epoch": int(ck["epoch"]),
            "tau_t": float(ck["tau_t"]),
            "sha256": FROZEN_STAGE_B_CHECKPOINT_SHA256[b_key],
            "cache_manifest_sha256": str(cfg["cache_manifest_sha256"]),
        }

    # Stage label-free PNGs to local NVMe /tmp for 40+ img/s GPU throughput
    src_png_dir = RSNA_MOUNT / "cache" / "stage1_nlbs_uint16" / "png_1520x912"
    local_png_dir = Path("/tmp/stage1_nlbs_local")
    local_png_dir.mkdir(parents=True, exist_ok=True)
    ok_uids = [uid for uid, r in s1_map.items() if r["decode_status"] == "ok"]
    t_stage0 = time.perf_counter()
    print(f"[nvme] Staging {len(ok_uids):,} NLBS 1520x912 PNGs to local NVMe across 16 threads...", flush=True)

    def _copy_one(uid: str) -> None:
        dst = local_png_dir / f"{uid}.png"
        if not dst.exists():
            shutil.copy2(src_png_dir / f"{uid}.png", dst)

    with ThreadPoolExecutor(max_workers=16) as cp_pool:
        list(cp_pool.map(_copy_one, ok_uids))
    print(f"[nvme] Staged {len(ok_uids):,} PNGs in {time.perf_counter() - t_stage0:.1f}s.", flush=True)

    env_meta = collect_runtime_environment_metadata(precision_policy=precision)
    ordered_uids = inf_df["image_uid"].astype(str).tolist()
    records_out: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    def _load_tensor(uid: str):
        arr = cv2.imread(str(local_png_dir / f"{uid}.png"), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.dtype != np.uint16:
            raise ValueError(f"Missing or corrupt Stage-1 PNG for {uid}")
        return prepare_tensor_from_cropped(
            cropped=arr,
            target_size=(1520, 912),
            augment_params=None,
            standardize=True,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for b_start in range(0, len(ordered_uids), encode_batch_size):
            batch_uids = ordered_uids[b_start : b_start + encode_batch_size]
            valid_uids = [u for u in batch_uids if s1_map[u]["decode_status"] == "ok"]

            scores_by_uid: Dict[str, tuple] = {}
            if valid_uids:
                tensors = list(pool.map(_load_tensor, valid_uids))
                for mb_start in range(0, len(valid_uids), micro_batch_size):
                    mb_u = valid_uids[mb_start : mb_start + micro_batch_size]
                    bx = torch.stack(tensors[mb_start : mb_start + micro_batch_size], dim=0).to(device, non_blocking=True)
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                        feats = encoder(bx)
                        if not torch.isfinite(feats).all():
                            raise ValueError("Non-finite Mammo-CLIP features during NLBS inference.")
                        p_ce = torch.sigmoid(heads["ce"](feats)).float().squeeze(1).cpu().numpy()
                        p_tr = torch.sigmoid(heads["trust"](feats)).float().squeeze(1).cpu().numpy()
                    for idx_mb, u in enumerate(mb_u):
                        sc_ce, sc_tr = float(p_ce[idx_mb]), float(p_tr[idx_mb])
                        if not (np.isfinite(sc_ce) and np.isfinite(sc_tr)):
                            raise ValueError(f"Non-finite probability for {u}")
                        scores_by_uid[u] = (sc_ce, sc_tr)

            for u in batch_uids:
                r = s1_map[u]
                sc_ce, sc_tr = scores_by_uid.get(u, (float("nan"), float("nan")))
                records_out.append(
                    {
                        "image_uid": u,
                        "decode_status": r["decode_status"],
                        "segmentation_status": r["segmentation_status"],
                        "used_fallback": bool(r["used_fallback"]),
                        "border_median": r["border_median"],
                        "laterality_status": r["laterality_status"],
                        "score_ce": sc_ce,
                        "score_trust": sc_tr,
                    }
                )

            done = b_start + len(batch_uids)
            if done % (encode_batch_size * 20) == 0 or done == len(ordered_uids):
                dt = max(time.perf_counter() - t0, 1e-6)
                rate = done / dt
                eta_min = (len(ordered_uids) - done) / max(rate, 1e-6) / 60.0
                print(
                    f"[score_l40s] {done:,}/{len(ordered_uids):,} ({100.0 * done / len(ordered_uids):5.1f}%) | "
                    f"{rate:5.2f} img/s | ETA {eta_min:4.1f}m",
                    flush=True,
                )

    return {
        "n_images": len(records_out),
        "elapsed_sec": float(time.perf_counter() - t0),
        "checkpoint_metadata": {
            "backbone_sha256": FROZEN_BACKBONE_SHA256,
            "preprocess_config_hash": expected_p_hash,
            "stage1_config_hash": expected_s1_hash,
            "precision": precision,
            "ce_checkpoint_sha256": ckpt_infos["ce"]["sha256"],
            "ce_epoch": ckpt_infos["ce"]["epoch"],
            "ce_cache_manifest_sha256": ckpt_infos["ce"]["cache_manifest_sha256"],
            "trust_checkpoint_sha256": ckpt_infos["trust"]["sha256"],
            "trust_epoch": ckpt_infos["trust"]["epoch"],
            "trust_tau_t_final": ckpt_infos["trust"]["tau_t"],
            "trust_cache_manifest_sha256": ckpt_infos["trust"]["cache_manifest_sha256"],
        },
        "environment_metadata": env_meta,
        "records": records_out,
    }


@app.local_entrypoint()
def main(
    pilot: int = 0,
    num_shards: int = 32,
    direct_gpu: bool = False,
    gpu: str = "L40S",
    micro_batch_size: int = 20,
    decode_batch_size: int = 80,
    num_workers: int = 8,
    precision: str = "amp",
    output_dir: str = "reports/stage_c",
) -> None:
    import pandas as pd

    from src.evaluation.stage_c_eval import (
        load_label_free_inference_manifest,
        lock_stage_c_predictions,
        verify_frozen_thresholds_against_stage_b,
    )

    gpu_norm = gpu.upper()
    if gpu_norm not in SUPPORTED_GPUS:
        raise ValueError(f"Unsupported GPU {gpu!r}; choose from {SUPPORTED_GPUS}.")
    if not LOCAL_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Label-free inference manifest missing at {LOCAL_MANIFEST_PATH}. Run scripts/build_nlbs_inference_manifest.py first."
        )

    inf_df = load_label_free_inference_manifest(LOCAL_MANIFEST_PATH)
    thresholds = verify_frozen_thresholds_against_stage_b(
        Path("reports/stage_b/rsna_ce_eval_results.json"),
        Path("reports/stage_b/rsna_trust_eval_results.json"),
    )

    fn = run_nlbs_inference_shard.with_options(gpu=gpu_norm)

    if pilot > 0:
        half = max(pilot // 2, 1)
        siemens_uids = inf_df.loc[inf_df["manufacturer"] == "SIEMENS", "image_uid"].head(half).tolist()
        ge_uids = inf_df.loc[inf_df["manufacturer"] != "SIEMENS", "image_uid"].head(pilot - len(siemens_uids)).tolist()
        pilot_uids = siemens_uids + ge_uids
        print(f"Running Stage C real NLBS pilot ({len(pilot_uids)} images: {len(ge_uids)} GE + {len(siemens_uids)} Siemens) on {gpu_norm}...")
        res = fn.remote(
            shard_idx=0,
            num_shards=1,
            target_image_uids=pilot_uids,
            decode_batch_size=decode_batch_size,
            micro_batch_size=micro_batch_size,
            num_workers=num_workers,
            precision=precision,
            save_shard_to_vol=False,
        )
        recs = res["records"]
        fails = [r for r in recs if r["decode_status"] != "ok"]
        fallbacks = [r for r in recs if r["used_fallback"]]
        print(json.dumps({
            "pilot_images": len(recs),
            "elapsed_sec": round(res["elapsed_sec"], 2),
            "throughput_img_s": round(len(recs) / max(res["elapsed_sec"], 1e-6), 2),
            "failed": len(fails),
            "fallbacks": len(fallbacks),
            "sample_records": recs[:3],
        }, indent=2))
        return

    if direct_gpu:
        print(f"Launching direct GPU Stage C inference across {num_shards} x {gpu_norm} shards ({len(inf_df):,} NLBS images)...")
        shard_args = [
            (s, num_shards, None, decode_batch_size, micro_batch_size, num_workers, precision, True)
            for s in range(num_shards)
        ]
        shard_outputs = list(fn.starmap(shard_args))
        all_records: List[Dict[str, Any]] = []
        for sh in shard_outputs:
            all_records.extend(sh["records"])
        ckpt_meta = shard_outputs[0]["checkpoint_metadata"]
        env_meta = shard_outputs[0]["environment_metadata"]
    else:
        print(f"Phase 1/2: Extracting Stage-1 1520x912 uint16 PNGs across 16 CPU workers ({num_shards} shards, {len(inf_df):,} NLBS DICOMs)...")
        s1_args = [(s, num_shards) for s in range(num_shards)]
        s1_outputs = list(build_nlbs_stage1_1520_shard.starmap(s1_args))
        s1_records: List[Dict[str, Any]] = []
        for sh in s1_outputs:
            s1_records.extend(sh["records"])
        print(f"Phase 1/2 complete ({len(s1_records):,} Stage-1 records). Launching Phase 2/2 frozen CE+TRUST scoring on {gpu_norm}...")
        score_fn = score_nlbs_from_stage1_l40s.with_options(gpu=gpu_norm)
        gpu_res = score_fn.remote(
            stage1_records=s1_records,
            micro_batch_size=micro_batch_size,
            encode_batch_size=decode_batch_size,
            num_workers=num_workers,
            precision=precision,
        )
        all_records = gpu_res["records"]
        ckpt_meta = gpu_res["checkpoint_metadata"]
        env_meta = gpu_res["environment_metadata"]

    lock_summary = lock_stage_c_predictions(
        inference_manifest_csv=LOCAL_MANIFEST_PATH,
        image_results=all_records,
        output_dir=Path(output_dir),
        checkpoint_metadata=ckpt_meta,
        environment_metadata=env_meta,
        thresholds=thresholds,
    )
    print(json.dumps(lock_summary, indent=2))
