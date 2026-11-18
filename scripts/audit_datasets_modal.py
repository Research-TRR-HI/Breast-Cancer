"""Stage A: Complete CPU-only Dataset Audit for RSNA and NLBS on Modal Volumes.

Strictly adheres to cost & safety rules:
- cpu=1.0, memory=2048, NO GPU attached.
- Read-only inspection of raw datasets (never modifies/deletes raw files).
- Does NOT train any model or use NLBS labels for threshold/model selection.

Usage:
    modal run scripts/audit_datasets_modal.py --mode discover
    modal run scripts/audit_datasets_modal.py --mode audit
"""

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import modal

APP_NAME = "anchorshift-stage-a-audit"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
NLBS_VOL = modal.Volume.from_name("nlbs-mammography-vol")

RSNA_MOUNT = Path("/rsna_vol")
NLBS_MOUNT = Path("/nlbs_vol")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("p7zip-full")
    .pip_install(
        "pandas>=2.2.0",
        "pydicom>=2.4.4",
        "pylibjpeg>=2.0.0",
        "pylibjpeg-libjpeg>=2.1.0",
        "pylibjpeg-openjpeg>=2.1.0",
        "python-gdcm>=3.0.24",
        "openpyxl>=3.1.2",
    )
)


def _scan_tree(root: Path) -> dict:
    """Summarize directory tree, extensions, and top-level non-image metadata files."""
    if not root.exists():
        return {"exists": False}

    ext_counts: Counter[str] = Counter()
    total_files = 0
    total_bytes = 0
    metadata_files = []
    sample_paths = []
    top_dirs: Counter[str] = Counter()

    for p in root.rglob("*"):
        if p.is_file():
            total_files += 1
            sz = p.stat().st_size
            total_bytes += sz
            ext = p.suffix.lower() or "<no_ext>"
            ext_counts[ext] += 1
            rel = p.relative_to(root)
            if len(rel.parts) > 1:
                top_dirs[rel.parts[0]] += 1
            else:
                top_dirs["<root>"] += 1
            if ext in {".csv", ".tsv", ".xlsx", ".xls", ".json", ".txt", ".md", ".zip"}:
                metadata_files.append({"path": str(rel), "size_bytes": sz})
            if len(sample_paths) < 15:
                sample_paths.append(str(rel))

    return {
        "exists": True,
        "total_files": total_files,
        "total_gb": round(total_bytes / (1024**3), 3),
        "extensions": dict(ext_counts.most_common(20)),
        "top_dirs": dict(top_dirs.most_common(20)),
        "metadata_files": sorted(metadata_files, key=lambda x: x["path"])[:50],
        "sample_paths": sample_paths,
    }


def _preview_table(path: Path) -> dict:
    """Read header and first 3 rows of a CSV/TSV/XLSX file."""
    import pandas as pd

    try:
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path, nrows=5)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=5)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(path, nrows=5)
        else:
            return {"error": f"Unsupported extension {ext}"}
        return {
            "columns": list(df.columns),
            "sample_rows": df.head(3).to_dict(orient="records"),
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    cpu=1.0,
    memory=2048,
    timeout=3600,
)
def discover_volumes() -> dict:
    """Inspect folder structures and metadata schemas of both Modal volumes."""
    rsna_info = _scan_tree(RSNA_MOUNT / "rsna")
    nlbs_info = _scan_tree(NLBS_MOUNT / "nlbs")

    rsna_tables = {}
    for item in rsna_info.get("metadata_files", []):
        p = RSNA_MOUNT / "rsna" / item["path"]
        if p.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls"}:
            rsna_tables[item["path"]] = _preview_table(p)

    nlbs_tables = {}
    for item in nlbs_info.get("metadata_files", []):
        p = NLBS_MOUNT / "nlbs" / item["path"]
        if p.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls"}:
            nlbs_tables[item["path"]] = _preview_table(p)

    return {
        "rsna": rsna_info,
        "rsna_tables": rsna_tables,
        "nlbs": nlbs_info,
        "nlbs_tables": nlbs_tables,
    }


def _audit_dicom_file(path: Path, decode_pixels: bool = False) -> dict:
    """Read DICOM header and optionally decode pixel_array."""
    import pydicom

    ds = pydicom.dcmread(str(path), stop_before_pixels=not decode_pixels)
    ts_uid = str(getattr(ds.file_meta, "TransferSyntaxUID", "UNKNOWN"))
    info = {
        "TransferSyntaxUID": ts_uid,
        "Manufacturer": str(getattr(ds, "Manufacturer", "")),
        "ManufacturerModelName": str(getattr(ds, "ManufacturerModelName", "")),
        "PhotometricInterpretation": str(getattr(ds, "PhotometricInterpretation", "")),
        "Rows": int(getattr(ds, "Rows", 0) or 0),
        "Columns": int(getattr(ds, "Columns", 0) or 0),
        "BitsStored": int(getattr(ds, "BitsStored", 0) or 0),
        "ViewPosition": str(getattr(ds, "ViewPosition", "")),
        "ImageLaterality": str(getattr(ds, "ImageLaterality", "")),
        "PatientID": str(getattr(ds, "PatientID", "")),
        "StudyInstanceUID": str(getattr(ds, "StudyInstanceUID", "")),
    }
    if decode_pixels:
        arr = ds.pixel_array
        info["pixel_shape"] = list(arr.shape)
        info["pixel_dtype"] = str(arr.dtype)
        info["pixel_min"] = float(arr.min())
        info["pixel_max"] = float(arr.max())
    return info


def _audit_rsna() -> tuple[dict, str]:
    """Run full audit on RSNA dataset and return (summary_dict, manifest_csv_str)."""
    import pandas as pd

    rsna_raw = RSNA_MOUNT / "rsna" / "raw"
    train_csv = rsna_raw / "train.csv"
    test_csv = rsna_raw / "test.csv"

    df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv) if test_csv.exists() else pd.DataFrame()

    # Scan train_images on disk
    train_img_dir = rsna_raw / "train_images"
    disk_files = {}
    for p in train_img_dir.rglob("*.dcm"):
        rel = p.relative_to(rsna_raw).as_posix()
        disk_files[rel] = p

    test_img_dir = rsna_raw / "test_images"
    test_disk_count = sum(1 for _ in test_img_dir.rglob("*.dcm")) if test_img_dir.exists() else 0

    df["rel_path"] = (
        "train_images/"
        + df["patient_id"].astype(str)
        + "/"
        + df["image_id"].astype(str)
        + ".dcm"
    )

    expected_paths = set(df["rel_path"])
    actual_paths = set(disk_files.keys())
    missing_on_disk = sorted(expected_paths - actual_paths)
    extra_on_disk = sorted(actual_paths - expected_paths)

    # Duplicates and label consistency
    dup_image_ids = int(df["image_id"].duplicated().sum())
    dup_pat_img = int(df.duplicated(subset=["patient_id", "image_id"]).sum())
    dup_lat_view = int(df.duplicated(subset=["patient_id", "laterality", "view"]).sum())

    # Patient & breast-level stats
    patient_cancer = df.groupby("patient_id")["cancer"].agg(["min", "max", "nunique"])
    breast_cancer = df.groupby(["patient_id", "laterality"])["cancer"].agg(["min", "max", "nunique"])
    inconsistent_breast_labels = int((breast_cancer["nunique"] > 1).sum())

    # View completeness per (patient_id, laterality)
    breast_views = df.groupby(["patient_id", "laterality"])["view"].apply(set)
    has_cc = breast_views.apply(lambda s: "CC" in s)
    has_mlo = breast_views.apply(lambda s: "MLO" in s)
    missing_cc_breasts = int((~has_cc).sum())
    missing_mlo_breasts = int((~has_mlo).sum())
    both_cc_mlo_breasts = int((has_cc & has_mlo).sum())

    # Stratified DICOM pixel-decode sample (200 images across site_id & machine_id)
    sample_df = (
        df[df["rel_path"].isin(actual_paths)]
        .groupby(["site_id", "machine_id"], group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), 20), random_state=42))
        .head(200)
    )
    dicom_ok = 0
    dicom_fail = []
    ts_counts: Counter[str] = Counter()
    photo_counts: Counter[str] = Counter()
    bits_counts: Counter[int] = Counter()
    shape_counts: Counter[str] = Counter()

    for _, row in sample_df.iterrows():
        p = disk_files[row["rel_path"]]
        try:
            info = _audit_dicom_file(p, decode_pixels=True)
            dicom_ok += 1
            ts_counts[info["TransferSyntaxUID"]] += 1
            photo_counts[info["PhotometricInterpretation"]] += 1
            bits_counts[info["BitsStored"]] += 1
            shape_counts[str(info["pixel_shape"])] += 1
        except Exception as exc:
            dicom_fail.append({"rel_path": row["rel_path"], "error": str(exc)})

    summary = {
        "train_csv_rows": int(len(df)),
        "train_images_on_disk": int(len(actual_paths)),
        "test_csv_rows": int(len(test_df)),
        "test_images_on_disk": int(test_disk_count),
        "missing_on_disk_count": len(missing_on_disk),
        "missing_on_disk_sample": missing_on_disk[:10],
        "extra_on_disk_count": len(extra_on_disk),
        "extra_on_disk_sample": extra_on_disk[:10],
        "duplicate_image_id_rows": dup_image_ids,
        "duplicate_patient_image_rows": dup_pat_img,
        "duplicate_patient_laterality_view_rows": dup_lat_view,
        "unique_patients": int(df["patient_id"].nunique()),
        "cancer_positive_patients": int((patient_cancer["max"] == 1).sum()),
        "cancer_negative_patients": int((patient_cancer["max"] == 0).sum()),
        "unique_breasts": int(len(breast_cancer)),
        "cancer_positive_breasts": int((breast_cancer["max"] == 1).sum()),
        "cancer_negative_breasts": int((breast_cancer["max"] == 0).sum()),
        "cancer_positive_images": int((df["cancer"] == 1).sum()),
        "cancer_negative_images": int((df["cancer"] == 0).sum()),
        "inconsistent_breast_labels": inconsistent_breast_labels,
        "site_id_patients": df.groupby("site_id")["patient_id"].nunique().to_dict(),
        "site_id_images": df["site_id"].value_counts().to_dict(),
        "machine_id_images": df["machine_id"].value_counts().to_dict(),
        "view_counts": df["view"].value_counts(dropna=False).to_dict(),
        "laterality_counts": df["laterality"].value_counts(dropna=False).to_dict(),
        "breasts_with_both_cc_mlo": both_cc_mlo_breasts,
        "breasts_missing_cc": missing_cc_breasts,
        "breasts_missing_mlo": missing_mlo_breasts,
        "dicom_sample_tested": int(len(sample_df)),
        "dicom_sample_ok": dicom_ok,
        "dicom_sample_fail_count": len(dicom_fail),
        "dicom_sample_failures": dicom_fail[:10],
        "dicom_transfer_syntaxes": dict(ts_counts),
        "dicom_photometric": dict(photo_counts),
        "dicom_bits_stored": dict(bits_counts),
        "dicom_top_shapes": dict(shape_counts.most_common(10)),
    }

    manifest_cols = [
        "patient_id",
        "image_id",
        "laterality",
        "view",
        "cancer",
        "biopsy",
        "invasive",
        "BIRADS",
        "implant",
        "density",
        "machine_id",
        "site_id",
        "age",
        "difficult_negative_case",
        "rel_path",
    ]
    manifest_df = df[manifest_cols].sort_values(["patient_id", "laterality", "view", "image_id"])
    return summary, manifest_df.to_csv(index=False)


def _verify_nlbs_checksums(nlbs_raw: Path) -> dict:
    """Verify checksums in frdr-dfdr-checksums.txt if present."""
    chk_file = nlbs_raw / "frdr-dfdr-checksums.txt"
    if not chk_file.exists():
        return {"exists": False}

    results = []
    for line in chk_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected_hash, rel_name = parts[0].lower(), parts[1].strip().lstrip("./")
        target = nlbs_raw / rel_name
        if not target.exists():
            results.append({"file": rel_name, "status": "missing"})
            continue
        algo = "sha256" if len(expected_hash) == 64 else ("md5" if len(expected_hash) == 32 else None)
        if not algo:
            results.append({"file": rel_name, "status": f"unknown_hash_len_{len(expected_hash)}"})
            continue
        h = hashlib.new(algo)
        with open(target, "rb") as fh:
            while chunk := fh.read(8 * 1024 * 1024):
                h.update(chunk)
        actual = h.hexdigest().lower()
        results.append({
            "file": rel_name,
            "algo": algo,
            "match": actual == expected_hash,
            "expected": expected_hash[:16] + "...",
            "actual": actual[:16] + "...",
        })
    return {"exists": True, "files": results}


def _list_7z_entries(archive_path: Path) -> tuple[bool, list[str], str]:
    """Run `7z t` (integrity test) and `7z l -slt` (file listing) on a .7z archive."""
    test_res = subprocess.run(
        ["7z", "t", str(archive_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    list_res = subprocess.run(
        ["7z", "l", "-slt", str(archive_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    paths = []
    for line in list_res.stdout.splitlines():
        if line.startswith("Path = "):
            p = line[len("Path = "):].strip().replace("\\", "/")
            if p != str(archive_path).replace("\\", "/") and p.lower().endswith(".dcm"):
                paths.append(p)
    return test_res.returncode == 0, paths, (test_res.stderr or test_res.stdout[-400:])


def _audit_nlbs() -> tuple[dict, str]:
    """Run full audit on NLBS archives + NLBSD_Metadata.csv and return (summary, manifest_csv)."""
    import pandas as pd

    nlbs_raw = NLBS_MOUNT / "nlbs" / "raw"
    readme_text = (nlbs_raw / "README.txt").read_text(encoding="utf-8", errors="replace") if (nlbs_raw / "README.txt").exists() else ""
    ver_text = (nlbs_raw / "README_version_history_2025-11-25.txt").read_text(encoding="utf-8", errors="replace") if (nlbs_raw / "README_version_history_2025-11-25.txt").exists() else ""

    checksums = _verify_nlbs_checksums(nlbs_raw)

    archives = ["normal.7z", "abnormal.7z", "False Positive.7z"]
    archive_reports = {}
    archived_paths: set[str] = set()
    for name in archives:
        arch_path = nlbs_raw / name
        if not arch_path.exists():
            archive_reports[name] = {"exists": False}
            continue
        ok, entries, msg = _list_7z_entries(arch_path)
        archive_reports[name] = {
            "exists": True,
            "size_gb": round(arch_path.stat().st_size / (1024**3), 3),
            "integrity_ok": ok,
            "dcm_count": len(entries),
            "sample_entries": entries[:5],
            "note": msg.strip()[-200:] if not ok else "OK",
        }
        archived_paths.update(entries)

    df = pd.read_csv(nlbs_raw / "NLBSD_Metadata.csv")
    df["rel_path"] = df["File Path"].astype(str).str.replace("\\", "/", regex=False)
    parts = df["rel_path"].str.split("/", expand=True)
    df["cohort_group"] = parts[0]
    df["case_id"] = parts[1]
    df["laterality_dir"] = parts[2]
    df["view_dir"] = parts[3]
    df["filename"] = parts[4]

    meta_paths = set(df["rel_path"])
    missing_in_archives = sorted(meta_paths - archived_paths)
    extra_in_archives = sorted(archived_paths - meta_paths)

    dup_paths = int(df["rel_path"].duplicated().sum())

    # Case-level stats
    case_cancer = df.groupby(["cohort_group", "case_id"])["Cancer"].agg(["min", "max", "nunique"])
    inconsistent_case_labels = int((case_cancer["nunique"] > 1).sum())

    # Repeat examination / patient grouping inspection
    case_id_counts = df.groupby("case_id")["cohort_group"].nunique()
    cross_cohort_case_ids = case_id_counts[case_id_counts > 1].index.tolist()
    age_per_case = df.groupby("case_id")["Age"].nunique()
    multi_age_case_ids = age_per_case[age_per_case > 1].index.tolist()

    # Check view completeness per (cohort_group, case_id, Image Laterality)
    breast_views = df.groupby(["cohort_group", "case_id", "Image Laterality"])["View Position"].apply(set)
    has_cc = breast_views.apply(lambda s: "CC" in s)
    has_mlo = breast_views.apply(lambda s: "MLO" in s)

    # Extract & decode a stratified sample of 60 DICOMs (20 from each .7z archive)
    dicom_sample_ok = 0
    dicom_sample_fail = []
    makers: Counter[str] = Counter()
    models: Counter[str] = Counter()
    ts_counts: Counter[str] = Counter()
    photo_counts: Counter[str] = Counter()
    bits_counts: Counter[int] = Counter()
    shape_counts: Counter[str] = Counter()
    sample_patient_ids = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for arch_name, grp_name in [
            ("normal.7z", "normal"),
            ("abnormal.7z", "abnormal"),
            ("False Positive.7z", "False Positive"),
        ]:
            arch_path = nlbs_raw / arch_name
            grp_paths = df[df["cohort_group"] == grp_name]["rel_path"].head(20).tolist()
            if not arch_path.exists() or not grp_paths:
                continue
            subprocess.run(
                ["7z", "x", "-y", f"-o{tmp}", str(arch_path)] + grp_paths,
                capture_output=True,
                check=False,
            )
            for rel in grp_paths:
                p = tmp / rel
                try:
                    info = _audit_dicom_file(p, decode_pixels=True)
                    dicom_sample_ok += 1
                    makers[info["Manufacturer"]] += 1
                    models[info["ManufacturerModelName"]] += 1
                    ts_counts[info["TransferSyntaxUID"]] += 1
                    photo_counts[info["PhotometricInterpretation"]] += 1
                    bits_counts[info["BitsStored"]] += 1
                    shape_counts[str(info["pixel_shape"])] += 1
                    if len(sample_patient_ids) < 10:
                        sample_patient_ids.append({
                            "rel_path": rel,
                            "dicom_PatientID": info["PatientID"],
                            "StudyInstanceUID_prefix": info["StudyInstanceUID"][:24],
                        })
                except Exception as exc:
                    dicom_sample_fail.append({"rel_path": rel, "error": str(exc)})

    summary = {
        "readme_excerpt": readme_text[:2500],
        "version_history": ver_text[:1000],
        "checksums": checksums,
        "archives": archive_reports,
        "metadata_csv_rows": int(len(df)),
        "archived_dcm_total": int(len(archived_paths)),
        "missing_in_archives_count": len(missing_in_archives),
        "missing_in_archives_sample": missing_in_archives[:10],
        "extra_in_archives_count": len(extra_in_archives),
        "extra_in_archives_sample": extra_in_archives[:10],
        "duplicate_rel_path_rows": dup_paths,
        "cohort_group_image_counts": df["cohort_group"].value_counts().to_dict(),
        "cohort_group_case_counts": df.groupby("cohort_group")["case_id"].nunique().to_dict(),
        "unique_case_ids": int(df["case_id"].nunique()),
        "total_cases_group_plus_id": int(len(case_cancer)),
        "cancer_positive_cases": int((case_cancer["max"] == 1).sum()),
        "cancer_negative_cases": int((case_cancer["max"] == 0).sum()),
        "cancer_positive_images": int((df["Cancer"] == 1).sum()),
        "cancer_negative_images": int((df["Cancer"] == 0).sum()),
        "hard_to_state_negative_images": int((df["Hard to State Negative"] == 1).sum()),
        "inconsistent_case_labels": inconsistent_case_labels,
        "cross_cohort_case_ids_count": len(cross_cohort_case_ids),
        "cross_cohort_case_ids": cross_cohort_case_ids[:20],
        "multi_age_case_ids_count": len(multi_age_case_ids),
        "breasts_with_both_cc_mlo": int((has_cc & has_mlo).sum()),
        "breasts_missing_cc": int((~has_cc).sum()),
        "breasts_missing_mlo": int((~has_mlo).sum()),
        "view_counts": df["View Position"].value_counts(dropna=False).to_dict(),
        "laterality_counts": df["Image Laterality"].value_counts(dropna=False).to_dict(),
        "dicom_sample_ok": dicom_sample_ok,
        "dicom_sample_fail_count": len(dicom_sample_fail),
        "dicom_sample_failures": dicom_sample_fail[:10],
        "dicom_manufacturers": dict(makers),
        "dicom_models": dict(models),
        "dicom_transfer_syntaxes": dict(ts_counts),
        "dicom_photometric": dict(photo_counts),
        "dicom_bits_stored": dict(bits_counts),
        "dicom_top_shapes": dict(shape_counts.most_common(10)),
        "sample_dicom_patient_ids": sample_patient_ids,
    }

    manifest_cols = [
        "cohort_group",
        "case_id",
        "Image Laterality",
        "View Position",
        "Cancer",
        "Hard to State Negative",
        "Age",
        "Exam Date",
        "Anode Target Material",
        "Energy Settings",
        "rel_path",
    ]
    manifest_df = df[manifest_cols].sort_values(["cohort_group", "case_id", "Image Laterality", "View Position", "rel_path"])
    return summary, manifest_df.to_csv(index=False)


@app.function(
    image=image,
    volumes={str(RSNA_MOUNT): RSNA_VOL, str(NLBS_MOUNT): NLBS_VOL},
    cpu=1.0,
    memory=2048,
    timeout=3600,
)
def run_full_audit() -> dict:
    """Execute complete RSNA and NLBS audit and return summaries + CSV manifests."""
    rsna_summary, rsna_csv = _audit_rsna()
    nlbs_summary, nlbs_csv = _audit_nlbs()
    return {
        "rsna": rsna_summary,
        "nlbs": nlbs_summary,
        "rsna_manifest_csv": rsna_csv,
        "nlbs_manifest_csv": nlbs_csv,
    }


@app.local_entrypoint()
def main(mode: str = "discover") -> None:
    if mode == "discover":
        result = discover_volumes.remote()
        out_path = Path("reports/discovery_summary.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"Saved discovery summary to {out_path}")
    elif mode == "audit":
        result = run_full_audit.remote()
        rsna_csv = result.pop("rsna_manifest_csv")
        nlbs_csv = result.pop("nlbs_manifest_csv")

        Path("datasets/rsna").mkdir(parents=True, exist_ok=True)
        Path("datasets/nlbs").mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(parents=True, exist_ok=True)

        Path("datasets/rsna/manifest.csv").write_text(rsna_csv, encoding="utf-8")
        Path("datasets/nlbs/manifest.csv").write_text(nlbs_csv, encoding="utf-8")
        Path("reports/audit_results.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        print("Saved datasets/rsna/manifest.csv, datasets/nlbs/manifest.csv, reports/audit_results.json")
    else:
        raise ValueError(f"Unknown mode: {mode}")
