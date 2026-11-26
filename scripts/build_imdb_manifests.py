"""Build deterministic label-blind inference manifests and protected outcome manifests for IMDB (r1, r2, combined).

Uses official IBIA metadata tables extracted from MAMOS_1000000004 (r1) and MAMOS_1000000050 (r2).
Validates all manifests against `src.paper1.external_datasets.write_manifest_pair` and
`select_common_4view_cohort` (never fabricating missing views).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paper1.external_datasets import (
    STANDARD_VIEWS,
    select_common_4view_cohort,
    write_manifest_pair,
)


def _normalize_modality_subtype(mtype: str) -> str:
    s = (mtype or "").strip().lower()
    if s == "2d-mammo":
        return "FFDM"
    if "c-view" in s:
        return "SM"
    return "UNKNOWN"


def _classify_image_reliability(row: Dict[str, str]) -> str:
    lbl = row["Ground truth label"].strip().lower()
    birads = row["BIRADS category"].strip()
    report = row.get("Ground truth report", "").lower()
    if lbl == "cancer":
        if birads in ("4", "5", "6") or re.search(
            r"\b(biops|histopath|hpe|fnac|carcinoma|idc|ilc|dcis|malignan|invasive)\b",
            report,
        ):
            return "pathology_confirmed_malignant"
        return "unknown_or_ambiguous"
    if lbl == "not cancer":
        if birads == "0":
            return "unknown_or_ambiguous"
        if birads == "4" or re.search(r"\b(biops|histopath|hpe|fnac)\b", report):
            return "pathology_confirmed_benign"
        return "noncancer_without_adequate_followup"
    return "unknown_or_ambiguous"


def _load_ibia_release(
    scratch_dir: Path,
    release: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with (scratch_dir / f"ibia_{release}_table_4.csv").open(encoding="utf-8") as f:
        t4 = list(csv.DictReader(f))
    with (scratch_dir / f"ibia_{release}_table_8.csv").open(encoding="utf-8") as f:
        t8 = list(csv.DictReader(f))
    with (scratch_dir / f"ibia_{release}_table_9.csv").open(encoding="utf-8") as f:
        t9 = list(csv.DictReader(f))

    t8_by_sample = {r["Sample ID"]: r for r in t8}
    inf_rows: List[Dict[str, Any]] = []
    by_exam: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for i, r4 in enumerate(t4):
        sid = r4["Sample ID"].strip()
        r8 = t8_by_sample[sid]
        dcm_url = t9[i]["Experiment ID"].strip()
        filename = Path(dcm_url).name
        pid = r4["De-identified Patient ID"].strip()
        exam_id = f"imdb_{release}_{pid}"
        cluster_id = exam_id
        mod_sub = _normalize_modality_subtype(r4["Mammography type"])

        acq_meta = {
            "release": release,
            "manufacturer": "HOLOGIC, Inc.",
            "manufacturer_model": "Selenia Dimensions",
            "modality_subtype": mod_sub,
            "ibia_mammography_type": r4["Mammography type"].strip(),
            "image_type": r8["Image type (Original / Derived / Unknown)"].strip(),
        }
        inf_rows.append(
            {
                "dataset_id": f"imdb_{release}",
                "cluster_id": cluster_id,
                "exam_id": exam_id,
                "image_id": f"imdb_{release}_{sid}",
                "laterality": r4["Laterality"].strip().upper(),
                "view": r4["Mammography view"].strip().upper(),
                "image_path": f"{release}/{filename}",
                "acquisition_meta": json.dumps(acq_meta, sort_keys=True),
                "preprocess_version": "trust_sec3_2_v2",
            }
        )
        by_exam[exam_id].append(r4)

    out_rows: List[Dict[str, Any]] = []
    for exam_id, erows in sorted(by_exam.items()):
        lbls = {r["Ground truth label"].strip().lower() for r in erows}
        outcome_label = 1 if "cancer" in lbls else 0
        img_cats = [_classify_image_reliability(r) for r in erows]
        if outcome_label == 1:
            if "pathology_confirmed_malignant" in img_cats:
                rel_cat = "pathology_confirmed_malignant"
            else:
                rel_cat = "unknown_or_ambiguous"
        else:
            if "pathology_confirmed_benign" in img_cats:
                rel_cat = "pathology_confirmed_benign"
            elif all(c == "unknown_or_ambiguous" for c in img_cats):
                rel_cat = "unknown_or_ambiguous"
            else:
                rel_cat = "noncancer_without_adequate_followup"

        std_set = set(STANDARD_VIEWS)
        all_views = {f"{r['Laterality'].strip().upper()}-{r['Mammography view'].strip().upper()}" for r in erows}
        ffdm_views = {
            f"{r['Laterality'].strip().upper()}-{r['Mammography view'].strip().upper()}"
            for r in erows
            if _normalize_modality_subtype(r["Mammography type"]) == "FFDM"
        }
        sm_views = {
            f"{r['Laterality'].strip().upper()}-{r['Mammography view'].strip().upper()}"
            for r in erows
            if _normalize_modality_subtype(r["Mammography type"]) == "SM"
        }
        birads_vals = sorted({r["BIRADS category"].strip() for r in erows if r["BIRADS category"].strip()})
        flags = {
            "release": release,
            "n_images": len(erows),
            "has_ffdm": len(ffdm_views) > 0,
            "has_sm": len(sm_views) > 0,
            "has_4view_all": std_set.issubset(all_views),
            "has_4view_ffdm": std_set.issubset(ffdm_views),
            "has_4view_sm": std_set.issubset(sm_views),
            "birads_categories": birads_vals,
            "outcome_reliability_category": rel_cat,
            "is_bilateral_mixed_label": len(lbls) > 1,
        }
        if "Clinical Indication" in erows[0]:
            flags["clinical_indication"] = erows[0]["Clinical Indication"].strip()

        out_rows.append(
            {
                "dataset_id": f"imdb_{release}",
                "cluster_id": exam_id,
                "exam_id": exam_id,
                "outcome_label": outcome_label,
                "label_source": rel_cat,
                "cohort_role": "mixed_diagnostic_opportunistic",
                "eligibility_flags": json.dumps(flags, sort_keys=True),
            }
        )

    return inf_rows, out_rows


def main() -> None:
    scratch_dir = Path(r"C:\Users\Hasin\.gemini\antigravity\brain\bf01d93d-8e75-4bf7-8f4f-15299fa4015f\scratch")
    meta_dir = Path("datasets/exp3_meta")
    report_dir = Path("reports/paper1/expanded/exp3/imdb_mias")
    report_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {}
    comb_inf: List[Dict[str, Any]] = []
    comb_out: List[Dict[str, Any]] = []

    for rel in ("r1", "r2"):
        inf_rows, out_rows = _load_ibia_release(scratch_dir, rel)
        inf_df = pd.DataFrame(inf_rows)
        out_df = pd.DataFrame(out_rows)
        meta = write_manifest_pair(inf_df, out_df, meta_dir, f"imdb_{rel}")

        picked_4v, elig_df = select_common_4view_cohort(inf_df, prefer_manufacturers=("HOLOGIC, INC.",), prefer_modality_subtypes=("FFDM",))
        n_4v_exams = int(elig_df["is_4view_complete"].sum())
        n_4v_pos = int(
            out_df[out_df["exam_id"].isin(set(picked_4v["exam_id"]))]["outcome_label"].sum()
        )
        # FFDM-only and SM-only counts
        ffdm_df = inf_df[inf_df["acquisition_meta"].str.contains('"FFDM"')].copy()
        sm_df = inf_df[inf_df["acquisition_meta"].str.contains('"SM"')].copy()
        ffdm_4v, ffdm_elig = select_common_4view_cohort(ffdm_df, prefer_manufacturers=("HOLOGIC, INC.",), prefer_modality_subtypes=("FFDM",))
        sm_4v, sm_elig = select_common_4view_cohort(sm_df, prefer_manufacturers=("HOLOGIC, INC.",), prefer_modality_subtypes=("SM",))

        summary[f"imdb_{rel}"] = {
            **meta,
            "n_4view_exams_all": n_4v_exams,
            "n_4view_pos_all": n_4v_pos,
            "ffdm_images": int(len(ffdm_df)),
            "ffdm_exams": int(ffdm_df["exam_id"].nunique()),
            "ffdm_4view_exams": int(ffdm_elig["is_4view_complete"].sum()),
            "ffdm_4view_pos": int(out_df[out_df["exam_id"].isin(set(ffdm_4v["exam_id"]))]["outcome_label"].sum()),
            "sm_images": int(len(sm_df)),
            "sm_exams": int(sm_df["exam_id"].nunique()),
            "sm_4view_exams": int(sm_elig["is_4view_complete"].sum()),
            "sm_4view_pos": int(out_df[out_df["exam_id"].isin(set(sm_4v["exam_id"]))]["outcome_label"].sum()),
            "outcome_reliability_exams": dict(Counter(out_df["label_source"])),
        }

        for r in inf_rows:
            r_comb = dict(r)
            r_comb["dataset_id"] = "imdb"
            comb_inf.append(r_comb)
        for r in out_rows:
            r_comb = dict(r)
            r_comb["dataset_id"] = "imdb"
            comb_out.append(r_comb)

    comb_inf_df = pd.DataFrame(comb_inf)
    comb_out_df = pd.DataFrame(comb_out)
    comb_meta = write_manifest_pair(comb_inf_df, comb_out_df, meta_dir, "imdb")
    comb_4v, comb_elig = select_common_4view_cohort(comb_inf_df, prefer_manufacturers=("HOLOGIC, INC.",), prefer_modality_subtypes=("FFDM",))
    summary["imdb_combined"] = {
        **comb_meta,
        "n_4view_exams_all": int(comb_elig["is_4view_complete"].sum()),
        "n_4view_pos_all": int(comb_out_df[comb_out_df["exam_id"].isin(set(comb_4v["exam_id"]))]["outcome_label"].sum()),
        "outcome_reliability_exams": dict(Counter(comb_out_df["label_source"])),
    }

    (report_dir / "imdb_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
