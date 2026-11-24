"""Paper 1 Exp 3 — Universal Non-Outcome External Dataset Manifests & Cohort Selection.

Provides:
    - Strict target-label quarantine for standardized non-outcome inference manifests.
    - Separate protected outcome manifest schema (`dataset_id`, `cluster_id`, `exam_id`, ...).
    - Dataset-specific adapters & metadata normalizers for NLBS, IMDB, CMMD, and LUMINA.
    - Deterministic common 4-view cohort selection (NEVER fabricating missing views) and
      image/breast/FFDM/pathology sensitivity cohort builders.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from src.evaluation.stage_c_eval import FORBIDDEN_LABEL_COLUMNS as STAGE_C_FORBIDDEN_COLUMNS, compute_sha256_file

STANDARD_VIEWS: Tuple[str, ...] = ("L-CC", "L-MLO", "R-CC", "R-MLO")

INFERENCE_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "dataset_id",
    "cluster_id",
    "exam_id",
    "image_id",
    "laterality",
    "view",
    "image_path",
    "acquisition_meta",
    "preprocess_version",
)

OUTCOME_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "dataset_id",
    "cluster_id",
    "exam_id",
    "outcome_label",
    "label_source",
    "cohort_role",
    "eligibility_flags",
)

EXTENDED_FORBIDDEN_LABEL_COLUMNS: Tuple[str, ...] = tuple(
    dict.fromkeys(
        list(STAGE_C_FORBIDDEN_COLUMNS)
        + [
            "birads",
            "bi_rads",
            "BIRADS",
            "BI-RADS",
            "pathology",
            "PATHOLOGY",
            "malignant",
            "benign",
            "classification",
            "subtype",
            "abnormality",
            "label",
            "outcome_label",
            "label_source",
            "ground_truth",
            "diagnosis",
            "target",
            "cancer_status",
            "followup_outcome",
        ]
    )
)

_FORBIDDEN_LOWER = {c.lower() for c in EXTENDED_FORBIDDEN_LABEL_COLUMNS}

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nlbs": {
        "dataset_id": "nlbs",
        "country": "Canada",
        "cohort_role": "screening",
        "population_screening_dismissal_allowed": True,
        "license": "FRDR Restricted / Academic Research Use (already downloaded & verified)",
        "access_route": "Local Modal reuse (nlbs-mammography-vol / stage1_nlbs_uint16)",
        "native_format": "DICOM (12-bit MONOCHROME2, GE + Siemens)",
        "has_4view_paired_cohort": True,
        "notes": "Organized provincial screening cohort; 6,359 unique StudyInstanceUIDs (6,184 4-view, 151 pos).",
    },
    "imdb": {
        "dataset_id": "imdb",
        "country": "India",
        "cohort_role": "mixed_diagnostic_opportunistic",
        "population_screening_dismissal_allowed": False,
        "license": "IBIA Open Access (DBT BIOTECH-PRIDE / comparable to CC BY); MIDAS CC BY 4.0",
        "access_route": "IBIA web download (MAMOS_1000000004 r1.0, MAMOS_1000000050 r2.0) or MIDAS on request",
        "native_format": "DICOM (Hologic Selenia 2 gens; mixed FFDM + Synthetic Mammography SM)",
        "has_4view_paired_cohort": True,
        "notes": "Screening-naive tertiary care mix (diagnostic + opportunistic screening); composite ground truth; must separate FFDM vs SM and r1/r2 IDs.",
    },
    "cmmd": {
        "dataset_id": "cmmd",
        "country": "China",
        "cohort_role": "diagnostic_biopsy_enriched",
        "population_screening_dismissal_allowed": False,
        "license": "CC BY 4.0",
        "access_route": "TCIA NBIA REST API / Data Retriever (DOI 10.7937/tcia.eqde-4b16)",
        "native_format": "DICOM (GE Senographe DS; 5,202 images / 1,775 patients; 21.292 GiB)",
        "has_4view_paired_cohort": True,
        "notes": "100% biopsy-confirmed benign/malignant diagnostic cohort (~74% malignant). 826/1775 bilateral 4-view; 949/1775 unilateral 2-view. No population dismissal claims.",
    },
    "lumina": {
        "dataset_id": "lumina",
        "country": "Turkey",
        "cohort_role": "diagnostic_biopsy_enriched",
        "population_screening_dismissal_allowed": False,
        "license": "CC BY-NC 4.0",
        "access_route": "OSF Storage API (https://osf.io/b63jc/ -> LUMINA_RAW/*.zip, 9.10 GB)",
        "native_format": "DICOM in LUMINA_RAW (12-14 bit, 6+ vendors) + harmonized PNGs in LUMINA_PNG",
        "has_4view_paired_cohort": False,
        "notes": "Multi-vendor diagnostic/energy stress test (468 patients, 1,824 imgs). Malign cases mostly unilateral; insufficient balanced 4-view malignant exams for NYU.",
    },
}


def assert_no_forbidden_columns(columns: Iterable[str], context: str = "dataframe") -> None:
    """Abort immediately if any label-revealing column exists (case-insensitive)."""
    present = [str(c) for c in columns if str(c).strip().lower() in _FORBIDDEN_LOWER]
    if present:
        raise ValueError(
            f"TARGET-LABEL QUARANTINE VIOLATION in {context}: forbidden columns present: {present}"
        )


def _assert_clean_acquisition_meta(series: pd.Series, context: str) -> None:
    """Ensure JSON-serialized acquisition_meta contains no outcome/BI-RADS/pathology keys."""
    for idx, val in enumerate(series):
        if pd.isna(val) or str(val).strip() == "":
            continue
        try:
            obj = json.loads(str(val))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in acquisition_meta row {idx} ({context}): {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"acquisition_meta row {idx} ({context}) must be a JSON object.")
        bad = [k for k in obj.keys() if str(k).strip().lower() in _FORBIDDEN_LOWER]
        if bad:
            raise ValueError(
                f"TARGET-LABEL QUARANTINE VIOLATION in {context}.acquisition_meta row {idx}: forbidden keys {bad}"
            )


def normalize_laterality(val: Any) -> str:
    """Normalize laterality strings to 'L' or 'R'."""
    s = str(val or "").strip().upper()
    if s in ("L", "LEFT"):
        return "L"
    if s in ("R", "RIGHT"):
        return "R"
    raise ValueError(f"Invalid laterality value: {val!r}")


def normalize_view_position(val: Any) -> str:
    """Normalize mammographic view strings (e.g. 'CC', 'MLO', 'L-CC' -> 'CC'/'MLO')."""
    s = str(val or "").strip().upper()
    if "-" in s and s.split("-", 1)[0] in ("L", "R"):
        s = s.split("-", 1)[1]
    if "_" in s and s.split("_", 1)[0] in ("L", "R"):
        s = s.split("_", 1)[1]
    return s


def validate_inference_manifest(df: pd.DataFrame, context: str = "inference_manifest") -> pd.DataFrame:
    """Validate standardized non-outcome inference manifest schema, uniqueness, and quarantine."""
    if df.empty:
        raise ValueError(f"{context}: inference manifest is empty.")
    assert_no_forbidden_columns(df.columns, context=context)

    missing = [c for c in INFERENCE_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}")

    out = df[list(INFERENCE_MANIFEST_COLUMNS)].copy()
    for col in ("dataset_id", "cluster_id", "exam_id", "image_id", "laterality", "view", "image_path", "preprocess_version"):
        if out[col].isna().any() or (out[col].astype(str).str.strip() == "").any():
            raise ValueError(f"{context}: empty or NaN values detected in {col!r}")
        out[col] = out[col].astype(str).str.strip()

    if out["dataset_id"].nunique() != 1:
        raise ValueError(f"{context}: inference manifest must contain exactly 1 dataset_id, got {out['dataset_id'].unique().tolist()}")

    out["laterality"] = out["laterality"].map(normalize_laterality)
    out["view"] = out["view"].map(normalize_view_position)

    if out["image_id"].duplicated().any():
        dups = out.loc[out["image_id"].duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"{context}: duplicate image_id detected: {dups}")

    # Every exam_id must belong to a single cluster_id
    cl_per_exam = out.groupby("exam_id")["cluster_id"].nunique()
    if (cl_per_exam > 1).any():
        bad = cl_per_exam[cl_per_exam > 1].index.tolist()[:5]
        raise ValueError(f"{context}: exam_id spans multiple cluster_ids: {bad}")

    _assert_clean_acquisition_meta(out["acquisition_meta"], context=context)
    return out.sort_values(["dataset_id", "exam_id", "laterality", "view", "image_id"], kind="mergesort").reset_index(drop=True)


def validate_outcome_manifest(df: pd.DataFrame, context: str = "outcome_manifest") -> pd.DataFrame:
    """Validate protected exam-level outcome manifest schema and binary labels."""
    if df.empty:
        raise ValueError(f"{context}: outcome manifest is empty.")
    missing = [c for c in OUTCOME_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}")

    out = df[list(OUTCOME_MANIFEST_COLUMNS)].copy()
    for col in ("dataset_id", "cluster_id", "exam_id", "label_source", "cohort_role"):
        if out[col].isna().any() or (out[col].astype(str).str.strip() == "").any():
            raise ValueError(f"{context}: empty or NaN values in {col!r}")
        out[col] = out[col].astype(str).str.strip()

    if out["dataset_id"].nunique() != 1:
        raise ValueError(f"{context}: outcome manifest must contain exactly 1 dataset_id")

    if out["exam_id"].duplicated().any():
        dups = out.loc[out["exam_id"].duplicated(), "exam_id"].tolist()[:5]
        raise ValueError(f"{context}: duplicate exam_id detected in outcome manifest: {dups}")

    labels = pd.to_numeric(out["outcome_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{context}: outcome_label must be strictly binary {{0, 1}}")
    out["outcome_label"] = labels.astype(int)
    out["eligibility_flags"] = out["eligibility_flags"].fillna("{}").astype(str)
    return out.sort_values(["dataset_id", "exam_id"], kind="mergesort").reset_index(drop=True)


def write_manifest_pair(
    inference_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    out_dir: Path,
    dataset_id: str,
) -> Dict[str, Any]:
    """Validate and atomically write separated non-outcome inference + protected outcome manifests."""
    inf_clean = validate_inference_manifest(inference_df, context=f"{dataset_id}_inference")
    out_clean = validate_outcome_manifest(outcome_df, context=f"{dataset_id}_outcome")

    inf_exams = set(inf_clean["exam_id"])
    out_exams = set(out_clean["exam_id"])
    if inf_exams != out_exams:
        raise ValueError(
            f"{dataset_id}: exam_id mismatch between inference ({len(inf_exams)}) and outcome ({len(out_exams)}) manifests."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inf_path = out_dir / f"{dataset_id}_inference_manifest.csv"
    out_path = out_dir / f"{dataset_id}_outcome_manifest.csv"

    inf_tmp = inf_path.with_suffix(".csv.tmp")
    inf_clean.to_csv(inf_tmp, index=False)
    inf_tmp.replace(inf_path)

    out_tmp = out_path.with_suffix(".csv.tmp")
    out_clean.to_csv(out_tmp, index=False)
    out_tmp.replace(out_path)

    return {
        "dataset_id": dataset_id,
        "n_images": int(len(inf_clean)),
        "n_exams": int(len(out_clean)),
        "n_clusters": int(out_clean["cluster_id"].nunique()),
        "n_positives": int(out_clean["outcome_label"].sum()),
        "inference_manifest_csv": str(inf_path),
        "inference_manifest_sha256": compute_sha256_file(inf_path),
        "outcome_manifest_csv": str(out_path),
        "outcome_manifest_sha256": compute_sha256_file(out_path),
    }


def select_common_4view_cohort(
    inference_df: pd.DataFrame,
    prefer_manufacturers: Sequence[str] = ("GE MEDICAL SYSTEMS",),
    prefer_modality_subtypes: Sequence[str] = ("FFDM",),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically select exact 4-view {L-CC, L-MLO, R-CC, R-MLO} exams WITHOUT fabricating views.

    Exams missing any of the 4 standard views are strictly excluded from the 4-view cohort
    and recorded in the returned eligibility summary. Never duplicates or synthesizes missing views.
    """
    df = validate_inference_manifest(inference_df, context="select_common_4view")
    df = df.copy()
    df["view_slot"] = df["laterality"] + "-" + df["view"]

    def _meta_field(meta_json: str, key: str, default: str = "") -> str:
        try:
            return str(json.loads(meta_json).get(key, default)).strip()
        except Exception:
            return default

    df["_mfg"] = df["acquisition_meta"].map(lambda s: _meta_field(s, "manufacturer"))
    df["_mod_sub"] = df["acquisition_meta"].map(lambda s: _meta_field(s, "modality_subtype", "FFDM").upper())
    pref_mfg_set = {m.upper() for m in prefer_manufacturers}
    pref_mod_set = {m.upper() for m in prefer_modality_subtypes}

    df["_pref_mod_rank"] = df["_mod_sub"].map(lambda m: 0 if m in pref_mod_set else 1)
    df["_pref_mfg_rank"] = df["_mfg"].str.upper().map(lambda m: 0 if m in pref_mfg_set else 1)

    std_df = df[df["view_slot"].isin(STANDARD_VIEWS)].copy()
    std_df = std_df.sort_values(
        ["exam_id", "view_slot", "_pref_mod_rank", "_pref_mfg_rank", "image_path", "image_id"],
        kind="mergesort",
    )
    picked = std_df.drop_duplicates(subset=["exam_id", "view_slot"], keep="first").copy()

    slots_per_exam = picked.groupby("exam_id")["view_slot"].nunique()
    eligible_exams = sorted(slots_per_exam[slots_per_exam == len(STANDARD_VIEWS)].index.tolist())
    eligible_set = set(eligible_exams)

    view_order = {v: i for i, v in enumerate(STANDARD_VIEWS)}
    picked_4v = picked[picked["exam_id"].isin(eligible_set)].copy()
    picked_4v["_v_ord"] = picked_4v["view_slot"].map(view_order)
    picked_4v = (
        picked_4v.sort_values(["exam_id", "_v_ord", "image_id"], kind="mergesort")
        .drop(columns=["_mfg", "_mod_sub", "_pref_mod_rank", "_pref_mfg_rank", "_v_ord"])
        .reset_index(drop=True)
    )

    all_exams = sorted(df["exam_id"].unique().tolist())
    elig_rows = []
    for eid in all_exams:
        sub_slots = set(std_df.loc[std_df["exam_id"] == eid, "view_slot"].tolist())
        missing_views = [v for v in STANDARD_VIEWS if v not in sub_slots]
        elig_rows.append(
            {
                "exam_id": eid,
                "is_4view_complete": len(missing_views) == 0,
                "n_standard_views_present": len(sub_slots),
                "missing_views": ",".join(missing_views),
            }
        )
    elig_df = pd.DataFrame(elig_rows)
    return picked_4v, elig_df


def build_nlbs_external_manifests(
    nlbs_inf_csv: Path,
    nlbs_exam_csv: Path,
    preprocess_version: str = "trust_sec3_2_v2",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert existing Stage C NLBS manifests into standardized Exp 3 inference & outcome tables."""
    src_inf = pd.read_csv(nlbs_inf_csv, dtype=str)
    assert_no_forbidden_columns(src_inf.columns, context=str(nlbs_inf_csv))

    acq = [
        json.dumps(
            {
                "manufacturer": str(r.get("manufacturer", "")),
                "manufacturer_model": str(r.get("manufacturer_model", "")),
                "photometric_interpretation": str(r.get("photometric_interpretation", "")),
                "modality_subtype": "FFDM",
            },
            sort_keys=True,
        )
        for r in src_inf.to_dict("records")
    ]
    inf_df = pd.DataFrame(
        {
            "dataset_id": "nlbs",
            "cluster_id": src_inf["split_cluster_id"].astype(str),
            "exam_id": src_inf["StudyInstanceUID"].astype(str),
            "image_id": src_inf["image_uid"].astype(str),
            "laterality": src_inf["laterality"].astype(str),
            "view": src_inf["view_position"].astype(str),
            "image_path": src_inf["dicom_rel_path"].astype(str),
            "acquisition_meta": acq,
            "preprocess_version": preprocess_version,
        }
    )

    em = pd.read_csv(nlbs_exam_csv, dtype=str)
    em_dedup = em.sort_values("exam_id").drop_duplicates("StudyInstanceUID", keep="first").reset_index(drop=True)
    elig_flags = [
        json.dumps({"cohort_group": str(r.get("cohort_group", "")), "visit_idx": str(r.get("visit_idx", ""))}, sort_keys=True)
        for r in em_dedup.to_dict("records")
    ]
    out_df = pd.DataFrame(
        {
            "dataset_id": "nlbs",
            "cluster_id": em_dedup["split_cluster_id"].astype(str),
            "exam_id": em_dedup["StudyInstanceUID"].astype(str),
            "outcome_label": em_dedup["exam_cancer"].astype(int),
            "label_source": "screening_registry_and_biopsy",
            "cohort_role": "screening",
            "eligibility_flags": elig_flags,
        }
    )
    return validate_inference_manifest(inf_df, "nlbs_inf"), validate_outcome_manifest(out_df, "nlbs_out")


def build_imdb_external_manifests(
    records_df: pd.DataFrame,
    preprocess_version: str = "trust_sec3_2_v2",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build standardized IMDB inference + outcome manifests with r1/r2 namespacing & UID dedup.

    Important clinical safeguards:
        - Prefixes patient IDs with release (`imdb_r1_<pt>` / `imdb_r2_<pt>`) so r1.0 and r2.0
          sequential `pt_XXX` IDs never collide.
        - Deduplicates by `sop_instance_uid` (if present) or `file_sha256` / `image_path`.
        - Never substitutes BI-RADS directly for cancer; preserves `label_source` provenance
          (`histopathology`, `followup_imaging_or_phone`, `birads12_negative`) and rejects
          missing/ambiguous binary outcomes.
    """
    req = {"release", "patient_id", "exam_id", "image_id", "laterality", "view", "image_path", "outcome_label", "label_source"}
    missing = sorted(req - set(records_df.columns))
    if missing:
        raise ValueError(f"IMDB records_df missing required columns: {missing}")

    df = records_df.copy()
    df["release"] = df["release"].astype(str).str.strip().str.lower()
    if not df["release"].isin(["r1", "r2", "r1.0", "r2.0"]).all():
        raise ValueError("IMDB release must be one of r1, r2, r1.0, r2.0")
    df["rel_tag"] = df["release"].map(lambda x: "r1" if x.startswith("r1") else "r2")

    # Namespace patient & exam IDs by release to avoid r1/r2 pt_XXX collisions
    df["cluster_id"] = "imdb_" + df["rel_tag"] + "_" + df["patient_id"].astype(str).str.strip()
    df["ns_exam_id"] = "imdb_" + df["rel_tag"] + "_" + df["exam_id"].astype(str).str.strip()
    df["ns_image_id"] = "imdb_" + df["rel_tag"] + "_" + df["image_id"].astype(str).str.strip()

    # Deduplicate overlapping DICOM instances across releases if UID or file hash provided
    if "sop_instance_uid" in df.columns and df["sop_instance_uid"].notna().any():
        df = df.sort_values(["rel_tag", "ns_exam_id", "ns_image_id"], kind="mergesort")
        df = df.drop_duplicates(subset=["sop_instance_uid"], keep="first").reset_index(drop=True)
    elif "file_sha256" in df.columns and df["file_sha256"].notna().any():
        df = df.sort_values(["rel_tag", "ns_exam_id", "ns_image_id"], kind="mergesort")
        df = df.drop_duplicates(subset=["file_sha256"], keep="first").reset_index(drop=True)

    # Reject ambiguous or missing binary cancer outcomes (never auto-impute BI-RADS 3/4/5)
    lbl = pd.to_numeric(df["outcome_label"], errors="coerce")
    if lbl.isna().any() or not lbl.isin([0, 1]).all():
        raise ValueError("IMDB records contain missing or non-binary outcome_label; filter unresolved BI-RADS 3 before manifest build.")

    acq = []
    for r in df.to_dict("records"):
        acq.append(
            json.dumps(
                {
                    "release": str(r["rel_tag"]),
                    "modality_subtype": str(r.get("modality_subtype", "FFDM")).upper(),
                    "manufacturer": str(r.get("manufacturer", "HOLOGIC")),
                    "manufacturer_model": str(r.get("manufacturer_model", "Selenia")),
                    "photometric_interpretation": str(r.get("photometric_interpretation", "MONOCHROME2")),
                },
                sort_keys=True,
            )
        )

    inf_df = pd.DataFrame(
        {
            "dataset_id": "imdb",
            "cluster_id": df["cluster_id"],
            "exam_id": df["ns_exam_id"],
            "image_id": df["ns_image_id"],
            "laterality": df["laterality"],
            "view": df["view"],
            "image_path": df["image_path"],
            "acquisition_meta": acq,
            "preprocess_version": preprocess_version,
        }
    )

    exam_grp = df.groupby("ns_exam_id", sort=True)
    if exam_grp["outcome_label"].nunique().gt(1).any():
        raise ValueError("IMDB exam_id has conflicting image-level outcome_label values.")

    out_rows = []
    for eid, grp in exam_grp:
        first = grp.iloc[0]
        mods = sorted(set(grp.get("modality_subtype", pd.Series(["FFDM"])).astype(str).str.upper()))
        out_rows.append(
            {
                "dataset_id": "imdb",
                "cluster_id": str(first["cluster_id"]),
                "exam_id": str(eid),
                "outcome_label": int(first["outcome_label"]),
                "label_source": str(first["label_source"]),
                "cohort_role": "mixed_diagnostic_opportunistic",
                "eligibility_flags": json.dumps(
                    {
                        "release": str(first["rel_tag"]),
                        "has_ffdm": "FFDM" in mods,
                        "has_sm": "SM" in mods,
                        "pathology_confirmed": str(first["label_source"]).lower() in ("histopathology", "biopsy", "pathology"),
                    },
                    sort_keys=True,
                ),
            }
        )
    out_df = pd.DataFrame(out_rows)
    return validate_inference_manifest(inf_df, "imdb_inf"), validate_outcome_manifest(out_df, "imdb_out")


def build_cmmd_external_manifests(
    clinical_df: pd.DataFrame,
    dicom_df: pd.DataFrame,
    preprocess_version: str = "trust_sec3_2_v2",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build CMMD manifests merging TCIA clinicaldata (`ID1`, `LeftRight`, `classification`) with DICOMs.

    Handles the 97 bilateral CMMD patients (`ID1` having both L & R clinical rows) cleanly:
    exam-level `outcome_label` is `1` if ANY diseased breast for that patient is Malignant,
    and breast-level outcomes are stored in `eligibility_flags` for breast-level sensitivity.
    """
    req_clin = {"ID1", "LeftRight", "classification"}
    if not req_clin.issubset(clinical_df.columns):
        raise ValueError(f"CMMD clinical_df missing columns: {sorted(req_clin - set(clinical_df.columns))}")
    req_dcm = {"patient_id", "image_id", "laterality", "view", "image_path"}
    if not req_dcm.issubset(dicom_df.columns):
        raise ValueError(f"CMMD dicom_df missing columns: {sorted(req_dcm - set(dicom_df.columns))}")

    clin = clinical_df.copy()
    clin["ID1"] = clin["ID1"].astype(str).str.strip()
    clin["LeftRight"] = clin["LeftRight"].map(normalize_laterality)
    clin["cls_norm"] = clin["classification"].astype(str).str.strip().str.lower()
    if not clin["cls_norm"].isin(["benign", "malignant"]).all():
        raise ValueError("CMMD clinical_df classification must be Benign or Malignant.")
    clin["is_malignant"] = (clin["cls_norm"] == "malignant").astype(int)

    dcm = dicom_df.copy()
    dcm["patient_id"] = dcm["patient_id"].astype(str).str.strip()
    known_pids = set(clin["ID1"])
    dcm = dcm[dcm["patient_id"].isin(known_pids)].copy()
    if dcm.empty:
        raise ValueError("No CMMD DICOM rows matched clinical_df ID1.")

    acq = [
        json.dumps(
            {
                "manufacturer": str(r.get("manufacturer", "GE MEDICAL SYSTEMS")),
                "manufacturer_model": str(r.get("manufacturer_model", "Senographe DS")),
                "photometric_interpretation": str(r.get("photometric_interpretation", "MONOCHROME2")),
                "modality_subtype": "FFDM",
            },
            sort_keys=True,
        )
        for r in dcm.to_dict("records")
    ]
    inf_df = pd.DataFrame(
        {
            "dataset_id": "cmmd",
            "cluster_id": "cmmd_" + dcm["patient_id"],
            "exam_id": "cmmd_" + dcm["patient_id"],
            "image_id": dcm["image_id"].astype(str),
            "laterality": dcm["laterality"],
            "view": dcm["view"],
            "image_path": dcm["image_path"].astype(str),
            "acquisition_meta": acq,
            "preprocess_version": preprocess_version,
        }
    )

    out_rows = []
    for pid, cgrp in clin.groupby("ID1", sort=True):
        breast_map = {str(r["LeftRight"]): int(r["is_malignant"]) for _, r in cgrp.iterrows()}
        abns = sorted(set(cgrp.get("abnormality", pd.Series(["unknown"])).astype(str).str.strip().str.lower()))
        out_rows.append(
            {
                "dataset_id": "cmmd",
                "cluster_id": f"cmmd_{pid}",
                "exam_id": f"cmmd_{pid}",
                "outcome_label": int(cgrp["is_malignant"].max()),
                "label_source": "biopsy_confirmed_pathology",
                "cohort_role": "diagnostic_biopsy_enriched",
                "eligibility_flags": json.dumps(
                    {
                        "is_bilateral_clinical": len(cgrp) > 1,
                        "breast_labels": breast_map,
                        "abnormalities": abns,
                        "pathology_confirmed": True,
                    },
                    sort_keys=True,
                ),
            }
        )
    out_df = pd.DataFrame(out_rows)
    return validate_inference_manifest(inf_df, "cmmd_inf"), validate_outcome_manifest(out_df, "cmmd_out")


def build_lumina_external_manifests(
    cases_df: pd.DataFrame,
    preprocess_version: str = "trust_sec3_2_v2",
    is_harmonized: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build LUMINA manifests with strict benign/malign ID namespacing (`lumina_benign_<ID>` / `lumina_malign_<ID>`)."""
    req = {"class_folder", "case_id", "image_id", "laterality", "view", "image_path", "pathology"}
    missing = sorted(req - set(cases_df.columns))
    if missing:
        raise ValueError(f"LUMINA cases_df missing required columns: {missing}")

    df = cases_df.copy()
    df["cls_tag"] = df["class_folder"].astype(str).str.strip().str.lower()
    if not df["cls_tag"].isin(["benign", "malign", "malignant"]).all():
        raise ValueError("LUMINA class_folder must be benign or malign/malignant")
    df["cls_tag"] = df["cls_tag"].map(lambda x: "benign" if x == "benign" else "malign")

    df["path_norm"] = df["pathology"].astype(str).str.strip().str.upper()
    if not df["path_norm"].isin(["BENIGN", "MALIGN", "MALIGNANT"]).all():
        raise ValueError("LUMINA pathology must be BENIGN or MALIGN/MALIGNANT (filter '-' rows cleanly).")
    df["outcome_label"] = df["path_norm"].map(lambda x: 0 if x == "BENIGN" else 1)

    df["ns_exam_id"] = "lumina_" + df["cls_tag"] + "_" + df["case_id"].astype(str).str.strip()
    df["ns_image_id"] = "lumina_" + df["cls_tag"] + "_" + df["image_id"].astype(str).str.strip()

    acq = [
        json.dumps(
            {
                "manufacturer": str(r.get("manufacturer", "")),
                "photometric_interpretation": str(r.get("photometric_interpretation", "MONOCHROME2")),
                "modality_subtype": "FFDM",
                "energy_harmonized": bool(is_harmonized),
            },
            sort_keys=True,
        )
        for r in df.to_dict("records")
    ]
    inf_df = pd.DataFrame(
        {
            "dataset_id": "lumina",
            "cluster_id": df["ns_exam_id"],
            "exam_id": df["ns_exam_id"],
            "image_id": df["ns_image_id"],
            "laterality": df["laterality"],
            "view": df["view"],
            "image_path": df["image_path"],
            "acquisition_meta": acq,
            "preprocess_version": preprocess_version,
        }
    )

    out_rows = []
    for eid, grp in df.groupby("ns_exam_id", sort=True):
        if grp["outcome_label"].nunique() != 1:
            raise ValueError(f"LUMINA conflicting pathology within exam {eid}")
        mfgs = sorted(set(grp.get("manufacturer", pd.Series([""])).astype(str).str.strip()))
        out_rows.append(
            {
                "dataset_id": "lumina",
                "cluster_id": str(eid),
                "exam_id": str(eid),
                "outcome_label": int(grp["outcome_label"].iloc[0]),
                "label_source": "pathology_confirmed",
                "cohort_role": "diagnostic_biopsy_enriched",
                "eligibility_flags": json.dumps(
                    {
                        "manufacturers": mfgs,
                        "energy_harmonized": bool(is_harmonized),
                        "pathology_confirmed": True,
                    },
                    sort_keys=True,
                ),
            }
        )
    out_df = pd.DataFrame(out_rows)
    return validate_inference_manifest(inf_df, "lumina_inf"), validate_outcome_manifest(out_df, "lumina_out")


def build_sensitivity_cohorts(
    inference_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Derive deterministic common-4view, breast-level, image-level, and FFDM/pathology sensitivity cohorts."""
    inf_df = validate_inference_manifest(inference_df, "sensitivity_inf")
    out_df = validate_outcome_manifest(outcome_df, "sensitivity_out")
    picked_4v, elig_4v = select_common_4view_cohort(inf_df)
    c4v_exams = sorted(picked_4v["exam_id"].unique().tolist())

    def _get_meta(s: str, k: str, default: str = "") -> str:
        try:
            return str(json.loads(s).get(k, default))
        except Exception:
            return default

    ffdm_exams = sorted(
        inf_df.loc[
            inf_df["acquisition_meta"].map(lambda s: _get_meta(s, "modality_subtype", "FFDM").upper()) == "FFDM",
            "exam_id",
        ]
        .unique()
        .tolist()
    )

    path_exams = sorted(
        out_df.loc[
            out_df["eligibility_flags"].map(lambda s: bool(json.loads(s).get("pathology_confirmed", False))),
            "exam_id",
        ].tolist()
    )

    breast_units = (
        inf_df[["dataset_id", "cluster_id", "exam_id", "laterality"]]
        .drop_duplicates()
        .sort_values(["exam_id", "laterality"])
        .reset_index(drop=True)
    )
    breast_units["breast_id"] = breast_units["exam_id"] + "__" + breast_units["laterality"]

    return {
        "exam_full_ids": sorted(out_df["exam_id"].tolist()),
        "exam_common_4view_ids": c4v_exams,
        "common_4view_inference_df": picked_4v,
        "common_4view_eligibility_df": elig_4v,
        "breast_units_df": breast_units,
        "image_level_ids": sorted(inf_df["image_id"].tolist()),
        "ffdm_exam_ids": ffdm_exams,
        "pathology_confirmed_exam_ids": path_exams,
    }
