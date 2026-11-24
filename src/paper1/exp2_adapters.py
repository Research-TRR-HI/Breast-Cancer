"""Paper 1 Experiment 2 — Original Pretrained Mammography Model Adapters & Quarantine Pipeline.

Locked Section 6 Decisions (5 Eligible Models, Category C Primary AUROC/AUPRC):
  - Eligible Locked Reuse ($0.00):
      1. `gmic_pretrained` (nyukat/GMIC, sample_model_1.p)
      2. `nyu_fourview_pretrained` (nyukat/breast_cancer_classifier, sample_image_model.p)
  - Eligible New Inference (tigerwood693):
      3. `glam_pretrained` (nyukat/GLAM, model_joint.ckpt; reuses GMIC 2944x1920 NLBS PNGs)
      4. `end2end_cbis_ddsm` (lishen/end2end-all-conv, ddsm_resnet50_s10_[512-512-1024]x2.h5)
      5. `petrini_twoview_efnb3` (dpetrini/multiple-view, CBIS-DDSM EfficientNet-B3 2-view)
  - Formally Disqualified & Rejected Candidates (Hard-Blocked):
      - `frcnn_cad`: DISQUALIFIED (dead ELTE weight URL + CUDA 8.0/Caffe + unvalidated semantics)
      - `mammoscreen_r1`: REJECTED (trained on RSNA 80/10/10 — violates zero-RSNA-supervision design)
      - `petrini_vindr_r2`: REJECTED (VinDr-Mammo BI-RADS >=3 is not biopsy-confirmed cancer)
      - `mammo_clip_zeroshot`: REJECTED (contrastive VLM; not a supervised cancer classifier)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.stage_c_eval import assert_no_forbidden_columns, compute_sha256_file
from src.paper1.gmic_adapter import GMIC_INPUT_SIZE, standard_normalize_gmic_uint16

EXP2_ELIGIBLE_MODELS: Tuple[str, ...] = (
    "gmic_pretrained",
    "nyu_fourview_pretrained",
    "glam_pretrained",
    "end2end_cbis_ddsm",
    "petrini_twoview_efnb3",
)
EXP2_NEW_INFER_MODELS: Tuple[str, ...] = (
    "glam_pretrained",
    "end2end_cbis_ddsm",
    "petrini_twoview_efnb3",
)
EXP2_PETRINI_INPUT_SIZE: Tuple[int, int] = (1152, 896)  # (height, width)
EXP2_PETRINI_CBIS_MEAN: float = 13369.0
EXP2_END2END_INPUT_SIZE: Tuple[int, int] = (1152, 896)  # (height, width)
EXP2_END2END_RESCALE_FACTOR: float = 255.0 / 65535.0
EXP2_END2END_CBIS_MEAN: float = 52.18
EXP2_ALLOWED_NLBS_MANIFESTS: Tuple[str, ...] = (
    "datasets/paper1/nlbs_4view_common_cohort_label_free.csv",
    "datasets/nlbs/inference_manifest_stage_c.csv",
)

EXP2_NLBS_CACHE_OPTIONS: Dict[str, Dict[str, Any]] = {
    "option_a_raw_dicom_transfer": {
        "description": "Transfer raw 26,988 NLBS DICOMs from hasinishrak2015:/nlbs/extracted to tigerwood693",
        "transfer_gib": 324.4480,
        "dest_storage_gib": 324.45,
        "estimated_cost_usd_range": [1.15, 1.75],
        "recommended": False,
        "reason": "Expensive raw transfer (+324.45 GiB) plus on-the-fly DICOM decode overhead.",
    },
    "option_b1_streamed_uncropped_1152x896": {
        "description": (
            "Stream read-only DICOMs from hasinishrak2015 into tigerwood693 CPU workers -> decode LUT + "
            "MONOCHROME1 invert + right-mirror -> resize uncropped full image to (1152, 896) uint16 PNGs"
        ),
        "transfer_gib": 0.0,
        "dest_storage_gib_range": [30.0, 38.0],
        "estimated_cost_usd_range": [0.38, 0.55],
        "recommended": True,
        "reason": (
            "Preserves exact uncropped whole-mammogram aspect/background statistics required by CBIS-DDSM "
            "models (End2End & Petrini) at ~9x lower storage than Option A and $0.00 hasinishrak2015 delta."
        ),
    },
    "option_b2_reuse_stage1_1520x912_cropped": {
        "description": "Reuse already-present tigerwood693:/cache/stage1_nlbs_uint16 (1520x912 breast-cropped)",
        "transfer_gib": 0.0,
        "dest_storage_gib": 0.0,
        "estimated_cost_usd_range": [0.0, 0.0],
        "recommended": False,
        "reason": (
            "Zero extra build cost, but tight TRUST breast-crop alters background fraction and pixel mean "
            "vs CBIS-DDSM uncropped 1152x896 training distribution (kept as fallback/ablation only)."
        ),
    },
}


@dataclass(frozen=True)
class Exp2ModelSpec:
    model_id: str
    display_name: str
    eligibility: str  # "eligible_locked_reuse", "eligible_new_inference", "disqualified", "rejected"
    threshold_category: str  # "C", "B", "A", "N/A"
    official_repo: str
    repo_revision: str
    checkpoint_name: str
    checkpoint_sha256: str
    license: str
    training_dataset: str
    input_level: str  # "image", "breast_2view", "exam_4view"
    nlbs_cache_source: str  # "locked_reuse", "gmic_2944x1920", "uncropped_1152x896", "blocked"
    aggregation_rule: str
    published_thresholds: Optional[Dict[str, float]] = None
    exclusion_reason: str = ""


EXP2_MODEL_REGISTRY: Dict[str, Exp2ModelSpec] = {
    "gmic_pretrained": Exp2ModelSpec(
        model_id="gmic_pretrained",
        display_name="GMIC (Pretrained ResNet-18, Model 1)",
        eligibility="eligible_locked_reuse",
        threshold_category="C",
        official_repo="https://github.com/nyukat/GMIC",
        repo_revision="594029efe6184bcca3a151fa1dd46c5fd620cb25",
        checkpoint_name="sample_model_1.p",
        checkpoint_sha256="3b792e6de22fdccd34a43b4c1cc87a94f5f4fa8aed42722dce4e4badac347991",
        license="AGPL-3.0",
        training_dataset="NYU Breast Cancer Screening Dataset (229,426 exams, 2010-2017)",
        input_level="image",
        nlbs_cache_source="locked_reuse",
        aggregation_rule="max(malignant_pred across exam images)",
    ),
    "nyu_fourview_pretrained": Exp2ModelSpec(
        model_id="nyu_fourview_pretrained",
        display_name="NYU Four-View CNN (Image-Only)",
        eligibility="eligible_locked_reuse",
        threshold_category="C",
        official_repo="https://github.com/nyukat/breast_cancer_classifier",
        repo_revision="6176ebea6950b0a953339593d2da417ab846dae0",
        checkpoint_name="sample_image_model.p",
        checkpoint_sha256="9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014",
        license="AGPL-3.0",
        training_dataset="NYU Breast Cancer Screening Dataset (229,426 exams, 2010-2017)",
        input_level="exam_4view",
        nlbs_cache_source="locked_reuse",
        aggregation_rule="max(left_malignant, right_malignant)",
    ),
    "glam_pretrained": Exp2ModelSpec(
        model_id="glam_pretrained",
        display_name="NYU GLAM (Global-Local Activation Maps, model_joint)",
        eligibility="eligible_new_inference",
        threshold_category="C",
        official_repo="https://github.com/nyukat/GLAM",
        repo_revision="17a0019860441e2ea8d7b7c7e0aaeada735e871f",
        checkpoint_name="model_joint.ckpt",
        checkpoint_sha256="a647ba9176b3e11d2dec7252054504f33856bcabb2bbe763432d3d20e7c65aec",
        license="NYU Non-Commercial Research-Only (LICENSE) / AGPL-3.0 (README conflict)",
        training_dataset="NYU Breast Cancer Screening Dataset (2010-2017)",
        input_level="image",
        nlbs_cache_source="gmic_2944x1920",
        aggregation_rule="max(malignant_pred across exam images)",
    ),
    "end2end_cbis_ddsm": Exp2ModelSpec(
        model_id="end2end_cbis_ddsm",
        display_name="Shen End2End All-Conv (CBIS-DDSM ResNet-50)",
        eligibility="eligible_new_inference",
        threshold_category="C",
        official_repo="https://github.com/lishen/end2end-all-conv",
        repo_revision="58401e220a2dd5120b373eef40e9737d7ce8d56c",
        checkpoint_name="end2end_cbis_ddsm_ts.pt",
        checkpoint_sha256="1c93c4c3ff7f997cc235140340181995d0219162cecd0bf495381b777a2389be",
        license="CC BY-NC-SA 4.0",
        training_dataset="CBIS-DDSM (digitized film, S10 whole-image split)",
        input_level="image",
        nlbs_cache_source="uncropped_1152x896",
        aggregation_rule="max(malignant_pred across exam images)",
    ),
    "petrini_twoview_efnb3": Exp2ModelSpec(
        model_id="petrini_twoview_efnb3",
        display_name="Petrini Two-View EfficientNet-B3 (CBIS-DDSM)",
        eligibility="eligible_new_inference",
        threshold_category="C",
        official_repo="https://github.com/dpetrini/multiple-view",
        repo_revision="2331b545ebdbd5e8c1da58e13b5dd14ba80d33c2",
        checkpoint_name="2024-10-01-14h36m_50ep_best_model_AUC_08643_Best_2-Views-CBIS-EFB3-CVUBP.pt",
        checkpoint_sha256="82add40900054835f23737d9f3a01922275f1e4dcfd3156a0f7059cb58147071",
        license="MIT",
        training_dataset="CBIS-DDSM (two-view CC+MLO, original split, N=2,694)",
        input_level="breast_2view",
        nlbs_cache_source="uncropped_1152x896",
        aggregation_rule="max(left_breast_malignancy_prob, right_breast_malignancy_prob)",
    ),
    "frcnn_cad": Exp2ModelSpec(
        model_id="frcnn_cad",
        display_name="Ribli Faster R-CNN CAD (VGG-16 Caffe)",
        eligibility="disqualified",
        threshold_category="N/A",
        official_repo="https://github.com/riblidezso/frcnn_cad",
        repo_revision="6c2deb57e37832c281588da11aed8808ba6ee1e9",
        checkpoint_name="vgg16_frcnn_cad.caffemodel",
        checkpoint_sha256="UNKNOWN",
        license="UNSPECIFIED (No LICENSE file in repository)",
        training_dataset="DDSM + private Hungarian clinical mammograms",
        input_level="image",
        nlbs_cache_source="blocked",
        aggregation_rule="N/A (disqualified)",
        exclusion_reason=(
            "DISQUALIFIED: Official weight URL http://dkrib.web.elte.hu/cad_faster_rcnn/vgg16_frcnn_cad.caffemodel "
            "redirects to error.elte.hu/error/452.html (dead, no mirror), CUDA 8.0/Caffe is incompatible with "
            "NVIDIA L40S (sm_89), and lesion detection outputs are not natively calibrated screening cancer scores."
        ),
    ),
    "mammoscreen_r1": Exp2ModelSpec(
        model_id="mammoscreen_r1",
        display_name="MammoScreen (ianpan/mammoscreen 3x EfficientNetV2-S)",
        eligibility="rejected",
        threshold_category="N/A",
        official_repo="https://huggingface.co/ianpan/mammoscreen",
        repo_revision="main",
        checkpoint_name="model.safetensors",
        checkpoint_sha256="8ace0998e2d534b08eaa673fad37df16eaffef6dca3f3b5ac2196857a2949596",
        license="Apache-2.0",
        training_dataset="CBIS-DDSM pretraining + RSNA Screening Mammography (80/10/10 Split 1)",
        input_level="breast_2view",
        nlbs_cache_source="blocked",
        aggregation_rule="N/A (rejected)",
        exclusion_reason=(
            "REJECTED FOR EXP 2: Fine-tuned on RSNA Screening Mammography (80/10/10 Split 1), violating "
            "Experiment 2's strict zero-RSNA-supervision separation from Experiment 1."
        ),
    ),
    "petrini_vindr_r2": Exp2ModelSpec(
        model_id="petrini_vindr_r2",
        display_name="Petrini Two-View ConvNeXt-Base (VinDr-Mammo BI-RADS>=3)",
        eligibility="rejected",
        threshold_category="N/A",
        official_repo="https://huggingface.co/dpetrini/vindr-convnext-multiple-view",
        repo_revision="main",
        checkpoint_name="2024-08-07-18h35m_50ep_best_model_AUC_08655_ConvNext_VinDr_Collab.pt",
        checkpoint_sha256="4619a4d337af31b7dcb622410bc0858ef1b20b872d0aea15f0c4f6c5705f8e71",
        license="MIT",
        training_dataset="VinDr-Mammo (BI-RADS 1-2 Normal vs 3-5 Abnormal)",
        input_level="breast_2view",
        nlbs_cache_source="blocked",
        aggregation_rule="N/A (rejected)",
        exclusion_reason=(
            "REJECTED FOR EXP 2: Trained on radiological BI-RADS >=3 ('Abnormal') labels rather than "
            "pathology/follow-up confirmed breast cancer."
        ),
    ),
    "mammo_clip_zeroshot": Exp2ModelSpec(
        model_id="mammo_clip_zeroshot",
        display_name="Mammo-CLIP B5 Zero-Shot Prompt Alignment",
        eligibility="rejected",
        threshold_category="N/A",
        official_repo="https://github.com/shawn24/Mammo-CLIP",
        repo_revision="2f356926a00fc3f0d9fdad1193c1464fd9adf564",
        checkpoint_name="b5-model-best-epoch-7.tar",
        checkpoint_sha256="0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c",
        license="CC BY-NC-SA 4.0",
        training_dataset="UPMC mammogram-report pairs (contrastive vision-language pretraining)",
        input_level="image",
        nlbs_cache_source="blocked",
        aggregation_rule="N/A (rejected)",
        exclusion_reason=(
            "REJECTED FOR EXP 2: Contrastive vision-language foundation model, not a supervised breast "
            "cancer screening classifier (already benchmarked with RSNA heads as ce_b5/trust_b5 in Exp 1)."
        ),
    ),
}


def default_pilot_batch_sizes(model_id: str) -> List[int]:
    """Return progressive L40S pilot batch sizes per model architecture and input resolution."""
    assert_model_eligible(model_id)
    if model_id == "glam_pretrained":
        return [1, 2, 4]
    if model_id == "end2end_cbis_ddsm":
        return [4, 8, 16, 32]
    if model_id == "petrini_twoview_efnb3":
        return [2, 4, 8, 16]
    return [1]


def assert_model_eligible(model_id: str) -> Exp2ModelSpec:
    """Return model spec or raise RuntimeError if unknown, disqualified, or rejected for Exp 2."""
    if model_id not in EXP2_MODEL_REGISTRY:
        raise ValueError(f"Unknown Experiment 2 model_id={model_id!r}")
    spec = EXP2_MODEL_REGISTRY[model_id]
    if spec.eligibility in ("disqualified", "rejected"):
        raise RuntimeError(
            f"BLOCKED ({spec.eligibility.upper()}): model_id={model_id!r} ({spec.exclusion_reason})"
        )
    return spec


def verify_checkpoint_sha256(
    checkpoint_path: Path,
    expected_sha256: str,
    *,
    allow_unknown: bool = False,
) -> str:
    """Verify checkpoint exists, is non-empty, and matches pinned SHA-256."""
    p = Path(checkpoint_path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty checkpoint file: {p}")
    actual = compute_sha256_file(p).lower()
    exp = str(expected_sha256).strip().lower()
    if exp == "unknown":
        if not allow_unknown:
            raise RuntimeError(
                f"Checkpoint SHA-256 for {p.name} is still 'UNKNOWN'. Pin SHA-256 in Wave 2 before inference."
            )
        return actual
    if actual != exp:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch for {p}: {actual} != {exp}")
    return actual


def load_exp2_nlbs_manifest(manifest_csv: Path, model_id: str) -> pd.DataFrame:
    """Load strictly label-free NLBS manifest and validate schema, uniqueness, and slot integrity."""
    spec = assert_model_eligible(model_id)
    df = pd.read_csv(manifest_csv, dtype=str)
    assert_no_forbidden_columns(df.columns, context=f"exp2_manifest_{model_id}")

    req = ["image_uid", "StudyInstanceUID", "laterality", "view_position", "dicom_rel_path"]
    for col in req:
        if col not in df.columns:
            raise ValueError(f"{manifest_csv} missing required column {col!r}")
        if df[col].isna().any() or (df[col].str.strip() == "").any():
            raise ValueError(f"{manifest_csv} contains empty values in {col!r}")

    if df["image_uid"].duplicated().any():
        dups = df[df["image_uid"].duplicated()]["image_uid"].head(5).tolist()
        raise ValueError(f"{manifest_csv} contains duplicate image_uid rows: {dups}")

    out = df.copy()
    out["laterality"] = out["laterality"].str.strip().str.upper()
    out["view_position"] = out["view_position"].str.strip().str.upper()
    out["view_slot"] = out["laterality"] + "-" + out["view_position"]

    if spec.input_level in ("breast_2view", "exam_4view"):
        if "manufacturer" not in out.columns:
            out["manufacturer"] = ""
        out["_mfg_rank"] = np.where(out["manufacturer"].fillna("").str.strip() == "GE MEDICAL SYSTEMS", 0, 1)
        out = (
            out.sort_values(
                ["StudyInstanceUID", "view_slot", "_mfg_rank", "dicom_rel_path", "image_uid"],
                kind="mergesort",
            )
            .drop_duplicates(subset=["StudyInstanceUID", "view_slot"], keep="first")
            .drop(columns=["_mfg_rank"])
            .reset_index(drop=True)
        )
    return out


def build_petrini_breast_pairs(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Pair deterministic (CC, MLO) views per (StudyInstanceUID, laterality) breast."""
    assert_no_forbidden_columns(manifest_df.columns, context="petrini_breast_pairs")
    cc = manifest_df[manifest_df["view_position"] == "CC"][
        ["StudyInstanceUID", "laterality", "image_uid", "dicom_rel_path"]
    ].rename(columns={"image_uid": "cc_image_uid", "dicom_rel_path": "cc_rel_path"})
    mlo = manifest_df[manifest_df["view_position"] == "MLO"][
        ["StudyInstanceUID", "laterality", "image_uid", "dicom_rel_path"]
    ].rename(columns={"image_uid": "mlo_image_uid", "dicom_rel_path": "mlo_rel_path"})

    if cc.duplicated(subset=["StudyInstanceUID", "laterality"]).any():
        raise ValueError("Duplicate CC view per (StudyInstanceUID, laterality) in Petrini pair builder")
    if mlo.duplicated(subset=["StudyInstanceUID", "laterality"]).any():
        raise ValueError("Duplicate MLO view per (StudyInstanceUID, laterality) in Petrini pair builder")

    pairs = cc.merge(mlo, on=["StudyInstanceUID", "laterality"], how="inner")
    pairs["pair_id"] = pairs["StudyInstanceUID"] + "_" + pairs["laterality"]
    return pairs.sort_values(["StudyInstanceUID", "laterality"]).reset_index(drop=True)


def prepare_uncropped_1152x896_from_array(
    arr_f32: np.ndarray,
    *,
    photometric: str,
    laterality: str,
    target_hw: Tuple[int, int] = EXP2_PETRINI_INPUT_SIZE,
) -> np.ndarray:
    """Option B1 preprocessor: invert MONOCHROME1, mirror R->left chest wall, resize uncropped to 1152x896 uint16."""
    arr = np.asarray(arr_f32, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mammogram array, got shape={arr.shape}")
    if str(photometric).strip().upper() == "MONOCHROME1":
        arr = float(np.max(arr)) - arr
    arr = arr - float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v > 0.0:
        u16 = np.round((arr / max_v) * 65535.0).clip(0.0, 65535.0).astype(np.uint16)
    else:
        u16 = np.zeros_like(arr, dtype=np.uint16)

    # Enforce chest-wall-on-left orientation matching Petrini CBIS-DDSM & End2End conventions
    mid = u16.shape[1] // 2
    left_mean = float(np.mean(u16[:, :mid]))
    right_mean = float(np.mean(u16[:, mid:]))
    eff_lat = str(laterality).strip().upper()
    if eff_lat == "R" and right_mean > left_mean:
        u16 = np.fliplr(u16).copy()
    elif eff_lat == "L" and right_mean > left_mean:
        u16 = np.fliplr(u16).copy()

    h, w = int(target_hw[0]), int(target_hw[1])
    if u16.shape != (h, w):
        resized = cv2.resize(u16, dsize=(w, h), interpolation=cv2.INTER_AREA)
        u16 = np.clip(resized, 0, 65535).astype(np.uint16)
    return u16


def preprocess_glam_from_gmic_uint16(arr_u16: np.ndarray) -> np.ndarray:
    """Standardize cached GMIC (2944, 1920) uint16 PNG identically for nyukat/GLAM."""
    arr = np.asarray(arr_u16)
    if arr.shape != GMIC_INPUT_SIZE or arr.dtype != np.uint16:
        raise ValueError(f"Expected {GMIC_INPUT_SIZE} uint16 for GLAM, got shape={arr.shape}, dtype={arr.dtype}")
    return standard_normalize_gmic_uint16(arr)


def _resize_uint16_to_model_hw(arr_u16: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr_u16)
    if arr.ndim != 2 or arr.dtype != np.uint16:
        raise ValueError(f"Expected 2D uint16 PNG, got shape={arr.shape}, dtype={arr.dtype}")
    h, w = int(target_hw[0]), int(target_hw[1])
    if arr.shape == (h, w):
        return arr
    resized = cv2.resize(arr, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(resized, 0, 65535).astype(np.uint16)


def preprocess_petrini_view_from_uint16(
    arr_u16: np.ndarray,
    *,
    dataset: str = "CBIS-DDSM",
) -> np.ndarray:
    """Normalize (1152, 896) uint16 PNG per Petrini and return (3, H, W) float32."""
    resized_u16 = _resize_uint16_to_model_hw(arr_u16, EXP2_PETRINI_INPUT_SIZE)
    img = resized_u16.astype(np.float32)
    if dataset == "CBIS-DDSM":
        img -= np.float32(EXP2_PETRINI_CBIS_MEAN)
        img /= np.float32(65535.0)
    else:
        raise ValueError(f"Unsupported Petrini dataset normalization={dataset!r}")
    return np.repeat(img[np.newaxis, :, :], 3, axis=0).astype(np.float32)


def build_petrini_twoview_tensor(
    cc_u16: np.ndarray,
    mlo_u16: np.ndarray,
    *,
    dataset: str = "CBIS-DDSM",
) -> torch.Tensor:
    """Construct exact (6, 1152, 896) float32 tensor with strict [CC_RGB, MLO_RGB] channel order."""
    cc_t = torch.from_numpy(preprocess_petrini_view_from_uint16(cc_u16, dataset=dataset))
    mlo_t = torch.from_numpy(preprocess_petrini_view_from_uint16(mlo_u16, dataset=dataset))
    return torch.cat([cc_t, mlo_t], dim=0)


def preprocess_end2end_from_uint16(
    arr_u16: np.ndarray,
    *,
    mean_pixel_intensity: float = EXP2_END2END_CBIS_MEAN,
) -> np.ndarray:
    """Rescale (1152, 896) uint16 PNG to [0, 255], subtract mean, return (3, H, W) float32."""
    resized_u16 = _resize_uint16_to_model_hw(arr_u16, EXP2_END2END_INPUT_SIZE)
    img = resized_u16.astype(np.float32) * np.float32(EXP2_END2END_RESCALE_FACTOR)
    img -= np.float32(mean_pixel_intensity)
    return np.repeat(img[np.newaxis, :, :], 3, axis=0).astype(np.float32)


def _ensure_repo_module(repo_dir: Path, module_names: Sequence[str], *, pkg_prefix: Optional[str] = None) -> Dict[str, Any]:
    """Import external repo modules cleanly without polluting sys.modules['src']."""
    if "h5py" not in sys.modules:
        try:
            importlib.import_module("h5py")
        except ImportError:
            sys.modules["h5py"] = types.ModuleType("h5py")

    repo_str = str(Path(repo_dir).resolve())
    saved_src: Dict[str, Any] = {}
    if pkg_prefix:
        saved_src = {k: v for k, v in sys.modules.items() if k == pkg_prefix or k.startswith(f"{pkg_prefix}.")}
        for k in list(saved_src.keys()):
            del sys.modules[k]
    sys.path.insert(0, repo_str)
    try:
        return {name: importlib.import_module(name) for name in module_names}
    finally:
        if sys.path and sys.path[0] == repo_str:
            sys.path.pop(0)
        if pkg_prefix:
            for k in [k for k in sys.modules if k == pkg_prefix or k.startswith(f"{pkg_prefix}.")]:
                del sys.modules[k]
            sys.modules.update(saved_src)


def build_glam_model(
    checkpoint_path: Path,
    *,
    glam_repo_dir: Path,
    device: Optional[torch.device] = None,
    expected_sha256: str = "UNKNOWN",
    allow_unknown_sha: bool = False,
) -> Tuple[nn.Module, str]:
    """Construct nyukat/GLAM MILSingleImageModel (PyTorch 2.x safe) and verify checkpoint."""
    sha = verify_checkpoint_sha256(checkpoint_path, expected_sha256, allow_unknown=allow_unknown_sha)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mods = _ensure_repo_module(glam_repo_dir, ["src.modeling.glam", "src.modeling.common_functions"], pkg_prefix="src")
    cf = mods["src.modeling.common_functions"]
    if not getattr(cf.get_max_window, "_pt2_floor_patched", False):
        def _get_max_window_pt2_safe(input_image: torch.Tensor, window_shape: Tuple[int, int], pooling_logic: str = "avg") -> torch.Tensor:
            N, C, H, W = input_image.size()
            if pooling_logic == "avg":
                pool_map = torch.nn.functional.avg_pool2d(input_image, window_shape, stride=1)
            else:
                pool_map = torch.nn.functional.avg_pool2d(input_image, window_shape, stride=1)
            _, _, _, W_map = pool_map.size()
            _, max_linear_idx = torch.max(pool_map.view(N, C, -1), -1)
            max_idx_x = torch.div(max_linear_idx, W_map, rounding_mode="floor")
            max_idx_y = max_linear_idx - max_idx_x * W_map
            upper_left = torch.cat((max_idx_x.unsqueeze(2), max_idx_y.unsqueeze(2)), dim=2)
            return upper_left.float()

        _get_max_window_pt2_safe._pt2_floor_patched = True  # type: ignore[attr-defined]
        cf.get_max_window = _get_max_window_pt2_safe
    params = {
        "device_type": "gpu" if device.type == "cuda" else "cpu",
        "gpu_number": device.index or 0,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "cam_size": (184, 120),
        "top_k_per_class": 1,
        "crop_shape": (512, 512),
        "percent_k": 0.03,
        "detection_pooling_percent_k": 0.3,
    }
    model = mods["src.modeling.glam"].MILSingleImageModel(params)
    local_ckpt = Path("/tmp") / checkpoint_path.name
    if not local_ckpt.exists() or local_ckpt.stat().st_size == 0:
        raw_bytes = checkpoint_path.read_bytes()
        eocd = raw_bytes.rfind(b"PK\x05\x06")
        if eocd != -1 and eocd + 22 <= len(raw_bytes):
            comment_len = int.from_bytes(raw_bytes[eocd + 20 : eocd + 22], "little")
            end_pos = min(len(raw_bytes), eocd + 22 + comment_len)
            local_ckpt.write_bytes(raw_bytes[:end_pos])
        else:
            local_ckpt.write_bytes(raw_bytes)
    state_dict = torch.load(str(local_ckpt), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading GLAM checkpoint {checkpoint_path}: {missing}")
    allowed_legacy_unexpected = {
        "mil.mil_attn_V.weight",
        "mil.mil_attn_U.weight",
        "mil.mil_attn_w.weight",
        "mil.classifier_linear.weight",
        "fusion_dnn.weight",
        "fusion_dnn.bias",
        "shared_rep_filter.weight",
    }
    bad_unexpected = [k for k in unexpected if k not in allowed_legacy_unexpected]
    if bad_unexpected:
        raise RuntimeError(f"Unexpected keys when loading GLAM checkpoint {checkpoint_path}: {bad_unexpected}")
    model = model.to(device)
    model.eval()
    return model, sha


def build_petrini_twoview_model(
    checkpoint_path: Path,
    *,
    petrini_repo_dir: Path,
    device: Optional[torch.device] = None,
    expected_sha256: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    """Construct Petrini SideMIDBreastModel directly without redundant single-view .pt load."""
    exp_sha = expected_sha256 or EXP2_MODEL_REGISTRY["petrini_twoview_efnb3"].checkpoint_sha256
    sha = verify_checkpoint_sha256(checkpoint_path, exp_sha)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_str = str(Path(petrini_repo_dir).resolve())
    sys.path.insert(0, repo_str)
    try:
        tvn = importlib.import_module("two_views_net_2024")
        sfn = importlib.import_module("single_full_net_2024")
    finally:
        if sys.path and sys.path[0] == repo_str:
            sys.path.pop(0)

    orig_load = tvn.torch.load
    orig_init = tvn.TwoViewsMIDBreastClassifier.__init__
    tvn.torch.load = lambda *args, **kwargs: None  # type: ignore[assignment]
    try:
        def _patched_init(self_tw, _dev, _mf, _net, exp_type="PBC", dataset="CBIS-DDSM"):
            nn.Module.__init__(self_tw)
            self_tw.single_clf_full = sfn.SingleBreastClassifier(_net, exp_type, dataset)
            self_tw.single_clf_core = self_tw.single_clf_full.feature_extractor

        tvn.TwoViewsMIDBreastClassifier.__init__ = _patched_init  # type: ignore[method-assign]
        model = tvn.SideMIDBreastModel(
            device,
            "UNUSED_INTERMEDIATE.pt",
            "EfficientNet-b3",
            n_blocks=2,
            b_type="mbconv",
            avg_pool=True,
            strides=2,
            exp_type="PBC",
            dataset="CBIS-DDSM",
        )
    finally:
        tvn.torch.load = orig_load
        tvn.TwoViewsMIDBreastClassifier.__init__ = orig_init

    local_ckpt = Path("/tmp") / checkpoint_path.name
    if not local_ckpt.exists() or local_ckpt.stat().st_size == 0:
        raw_bytes = checkpoint_path.read_bytes()
        eocd = raw_bytes.rfind(b"PK\x05\x06")
        if eocd != -1 and eocd + 22 <= len(raw_bytes):
            comment_len = int.from_bytes(raw_bytes[eocd + 20 : eocd + 22], "little")
            end_pos = min(len(raw_bytes), eocd + 22 + comment_len)
            local_ckpt.write_bytes(raw_bytes[:end_pos])
        else:
            local_ckpt.write_bytes(raw_bytes)
    state_dict = torch.load(str(local_ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, sha


def score_twoview_batch(model: nn.Module, batch_6ch: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run FP32 evaluation on a [B, 6, H, W] tensor and return positive-class probabilities."""
    with torch.no_grad():
        logits = model(batch_6ch.to(device))
        probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return probs.astype(np.float64)


def aggregate_exp2_exam_scores(records_df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """Validate finite predictions and aggregate image/breast scores to deduplicated StudyInstanceUID."""
    spec = assert_model_eligible(model_id)
    assert_no_forbidden_columns(records_df.columns, context=f"exp2_agg_{model_id}")
    if "StudyInstanceUID" not in records_df.columns or "score" not in records_df.columns:
        raise ValueError("records_df must include 'StudyInstanceUID' and 'score'")

    df = records_df.copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["score"] = df["score"].astype(float)
    if not np.all(np.isfinite(df["score"].values)):
        raise ValueError(f"{model_id}: non-finite scores detected prior to exam aggregation")

    if spec.input_level == "image":
        if "image_uid" not in df.columns:
            raise ValueError(f"{model_id}: image-level records require 'image_uid'")
        if df["image_uid"].duplicated().any():
            raise ValueError(f"{model_id}: duplicate image_uid predictions detected")
        exam_df = (
            df.groupby("StudyInstanceUID", as_index=False)
            .agg(score=("score", "max"), n_units=("image_uid", "count"))
            .sort_values("StudyInstanceUID")
            .reset_index(drop=True)
        )
    elif spec.input_level == "breast_2view":
        if "laterality" not in df.columns:
            raise ValueError(f"{model_id}: breast-level records require 'laterality'")
        if df.duplicated(subset=["StudyInstanceUID", "laterality"]).any():
            raise ValueError(f"{model_id}: duplicate (StudyInstanceUID, laterality) predictions detected")
        exam_df = (
            df.groupby("StudyInstanceUID", as_index=False)
            .agg(score=("score", "max"), n_units=("laterality", "count"))
            .sort_values("StudyInstanceUID")
            .reset_index(drop=True)
        )
    else:
        if df["StudyInstanceUID"].duplicated().any():
            raise ValueError(f"{model_id}: duplicate StudyInstanceUID exam predictions detected")
        exam_df = df[["StudyInstanceUID", "score"]].copy()
        exam_df["n_units"] = 4
        exam_df = exam_df.sort_values("StudyInstanceUID").reset_index(drop=True)

    return exam_df


def apply_threshold_policy(
    exam_scores_df: pd.DataFrame,
    model_id: str,
    *,
    external_thresholds: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enforce Category C primary (AUROC/AUPRC) and block RSNA/NLBS calibration or arbitrary 0.5."""
    spec = assert_model_eligible(model_id)
    assert_no_forbidden_columns(exam_scores_df.columns, context=f"exp2_thresh_{model_id}")
    df = exam_scores_df.copy()
    scores = df["score"].astype(float).values

    if spec.threshold_category == "A":
        if not spec.published_thresholds:
            raise RuntimeError(f"Category A model {model_id} missing published_thresholds")
        if external_thresholds is not None:
            raise ValueError(f"Category A model {model_id} must use published thresholds, not external override")
        t98 = float(spec.published_thresholds["target_98"])
        t95 = float(spec.published_thresholds["target_95"])
        df["dismissed_98"] = scores < t98
        df["dismissed_95"] = scores < t95
        return df, {
            "threshold_category": "A",
            "calibration_domain": "published_author_threshold",
            "target_98": t98,
            "target_95": t95,
        }

    if spec.threshold_category == "B":
        if not external_thresholds:
            raise ValueError(f"Category B model {model_id} requires non-RSNA external_thresholds")
        cal_dom = str(external_thresholds.get("calibration_domain", "")).strip().upper()
        if not cal_dom or "RSNA" in cal_dom or "NLBS" in cal_dom:
            raise ValueError(
                f"FORBIDDEN calibration_domain={cal_dom!r}: Exp 2 forbids RSNA and NLBS threshold calibration."
            )
        t98 = float(external_thresholds["target_98"])
        t95 = float(external_thresholds["target_95"])
        df["dismissed_98"] = scores < t98
        df["dismissed_95"] = scores < t95
        return df, {
            "threshold_category": "B",
            "calibration_domain": cal_dom,
            "target_98": t98,
            "target_95": t95,
        }

    if external_thresholds is not None:
        cal_dom = str(external_thresholds.get("calibration_domain", "")).strip().upper()
        if "RSNA" in cal_dom or "NLBS" in cal_dom:
            raise ValueError(
                f"FORBIDDEN calibration_domain={cal_dom!r}: Exp 2 forbids RSNA and NLBS threshold calibration."
            )
        raise ValueError(
            f"Category C model {model_id} has no compatible published threshold; evaluate AUROC/AUPRC only."
        )
    return df, {
        "threshold_category": "C",
        "calibration_domain": "none_threshold_independent_auroc_auprc_only",
        "target_98": None,
        "target_95": None,
    }


def compute_unit_ids_sha256(unit_ids: Sequence[str]) -> str:
    payload = json.dumps([str(x) for x in unit_ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_exp2_shard_payload(
    payload: Dict[str, Any],
    *,
    model_id: str,
    expected_unit_ids: Sequence[str],
    split_name: str,
    shard_idx: int,
    num_shards: int,
    checkpoint_sha256: str,
) -> Tuple[bool, str]:
    """Verify resumable Exp 2 shard payload provenance, completeness, and finite scores."""
    if not isinstance(payload, dict):
        return False, "payload_not_dict"
    if payload.get("schema_version") != "paper1_exp2_shard_v1":
        return False, f"schema_mismatch:{payload.get('schema_version')}"
    if payload.get("model_id") != model_id:
        return False, f"model_id_mismatch:{payload.get('model_id')}"
    if str(payload.get("checkpoint_sha256", "")).lower() != str(checkpoint_sha256).lower():
        return False, "checkpoint_sha256_mismatch"
    if payload.get("split_name") != split_name:
        return False, "split_name_mismatch"
    if payload.get("shard_idx") != int(shard_idx) or payload.get("num_shards") != int(num_shards):
        return False, "shard_spec_mismatch"

    exp_ids = [str(x) for x in expected_unit_ids]
    if payload.get("expected_unit_ids_sha256") != compute_unit_ids_sha256(exp_ids):
        return False, "expected_unit_ids_sha256_mismatch"

    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(exp_ids):
        return False, "records_len_mismatch"
    rec_ids = [str(r.get("unit_id", "")) for r in records]
    if rec_ids != exp_ids:
        return False, "record_unit_ids_mismatch"
    for idx, rec in enumerate(records):
        val = rec.get("score")
        if val is None or not math.isfinite(float(val)):
            return False, f"non_finite_score_at_{idx}"
    return True, ""


def finalize_and_lock_exp2_predictions(
    *,
    model_id: str,
    split_name: str,
    shard_outputs: Sequence[Dict[str, Any]],
    expected_studies: Sequence[str],
    checkpoint_sha256: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Validate 100% shard coverage, aggregate exam scores, apply threshold policy, and SHA-256 lock."""
    spec = assert_model_eligible(model_id)
    all_recs: List[Dict[str, Any]] = []
    shard_meta: List[Dict[str, Any]] = []
    for o in sorted(shard_outputs, key=lambda x: int(x["shard_idx"])):
        if str(o.get("checkpoint_sha256", "")).lower() != str(checkpoint_sha256).lower():
            raise RuntimeError(f"Shard {o.get('shard_idx')} checkpoint_sha256 mismatch")
        all_recs.extend(o["records"])
        shard_meta.append(
            {
                "shard_idx": int(o["shard_idx"]),
                "num_shards": int(o["num_shards"]),
                "n_units": len(o["records"]),
                "estimated_cost_usd": o.get("estimated_cost_usd"),
                "resumed_from_cache": bool(o.get("resumed_from_cache", False)),
            }
        )

    rec_df = pd.DataFrame(all_recs)
    exam_df = aggregate_exp2_exam_scores(rec_df, model_id=model_id)
    exp_set = set(str(s) for s in expected_studies)
    if set(exam_df["StudyInstanceUID"]) != exp_set:
        raise RuntimeError(
            f"{model_id}/{split_name} study coverage mismatch: {len(exam_df)} vs {len(exp_set)}"
        )

    thresh_df, thresh_meta = apply_threshold_policy(exam_df, model_id=model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_csv = out_dir / f"{model_id}_{split_name}_unit_scores.csv"
    case_csv = out_dir / f"{model_id}_{split_name}_case_scores.csv"
    lock_json = out_dir / f"{model_id}_{split_name}_provenance_lock.json"

    rec_df.to_csv(unit_csv, index=False)
    thresh_df.to_csv(case_csv, index=False)

    scores_arr = thresh_df["score"].values.astype(np.float64)
    lock_manifest = {
        "schema_version": "paper1_exp2_lock_v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "checkpoint_name": spec.checkpoint_name,
        "checkpoint_sha256": str(checkpoint_sha256).lower(),
        "split_name": split_name,
        "label_quarantine_enforced": True,
        "threshold_policy": thresh_meta,
        "n_unique_studies": int(len(thresh_df)),
        "n_unit_records": int(len(rec_df)),
        "all_scores_finite": bool(np.all(np.isfinite(scores_arr))),
        "score_summary_label_free": {
            "min": round(float(np.min(scores_arr)), 6),
            "median": round(float(np.median(scores_arr)), 6),
            "max": round(float(np.max(scores_arr)), 6),
        },
        "shards": shard_meta,
        "locked_files_sha256": {
            unit_csv.name: compute_sha256_file(unit_csv),
            case_csv.name: compute_sha256_file(case_csv),
        },
    }
    lock_json.write_text(json.dumps(lock_manifest, indent=2), encoding="utf-8")
    return lock_manifest
