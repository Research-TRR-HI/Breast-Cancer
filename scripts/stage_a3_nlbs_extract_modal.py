"""Stage A3: Extract NLBS .7z archives, verify FRDR checksums, and build exam-level manifest.

Strictly adheres to cost & safety rules:
- cpu=1.0, memory=2048, ephemeral_disk=524288, NO GPU attached.
- Extracts to /nlbs_vol/nlbs/extracted/ while preserving /nlbs_vol/nlbs/raw/ untouched.
- Periodic volume.commit() every 60s + SIGINT/SIGTERM handlers.
- Does NOT train models or use NLBS labels for threshold/model selection.

Usage:
    modal run scripts/stage_a3_nlbs_extract_modal.py
"""

import hashlib
import json
import re
import signal
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path

import modal

APP_NAME = "anchorshift-stage-a3-nlbs"
NLBS_VOL = modal.Volume.from_name("nlbs-mammography-vol")
NLBS_MOUNT = Path("/nlbs_vol")
COMMIT_INTERVAL_SEC = 60

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("p7zip-full")
    .pip_install(
        "pandas>=2.2.0",
        "pydicom>=2.4.4",
    )
)


def _start_periodic_commit(vol: modal.Volume, stop_event: threading.Event) -> threading.Thread:
    def _loop() -> None:
        while not stop_event.wait(COMMIT_INTERVAL_SEC):
            try:
                vol.commit()
                print("[modal] Periodic volume.commit() completed.", flush=True)
            except Exception as exc:
                print(f"[modal] Periodic commit warning: {exc}", file=sys.stderr, flush=True)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def _register_signal_handlers(vol: modal.Volume, stop_event: threading.Event) -> None:
    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"\n[signal] Caught signal {signum}. Committing volume before exit...", flush=True)
        stop_event.set()
        try:
            vol.commit()
        except Exception:
            pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def _verify_frdr_checksums(nlbs_raw: Path) -> list[dict]:
    """Parse FRDR multi-line checksum file and verify SHA-256 of every listed file."""
    chk_path = nlbs_raw / "frdr-dfdr-checksums.txt"
    lines = [ln.strip() for ln in chk_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    results = []
    sha_re = re.compile(r"Checksum/Sommes de contr[oô]le:\s*([0-9a-fA-F]{64})")

    for i, line in enumerate(lines):
        m = sha_re.search(line)
        if not m or i == 0:
            continue
        expected = m.group(1).lower()
        filename = lines[i - 1].replace("Filename/Fichier:", "").strip()
        target = nlbs_raw / filename
        if not target.exists():
            results.append({"file": filename, "exists": False, "match": False})
            continue
        h = hashlib.sha256()
        with open(target, "rb") as fh:
            while chunk := fh.read(8 * 1024 * 1024):
                h.update(chunk)
        actual = h.hexdigest().lower()
        results.append({
            "file": filename,
            "exists": True,
            "size_bytes": target.stat().st_size,
            "match": actual == expected,
            "sha256": actual,
        })
    return results


@app.function(
    image=image,
    volumes={str(NLBS_MOUNT): NLBS_VOL},
    cpu=1.0,
    memory=2048,
    ephemeral_disk=524288,
    timeout=14400,
)
def run_stage_a3() -> dict:
    """Verify checksums, extract NLBS .7z archives, and extract DICOM exam metadata."""
    import pandas as pd
    import pydicom

    nlbs_raw = NLBS_MOUNT / "nlbs" / "raw"
    nlbs_ext = NLBS_MOUNT / "nlbs" / "extracted"
    nlbs_ext.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    _register_signal_handlers(NLBS_VOL, stop_event)
    commit_thread = _start_periodic_commit(NLBS_VOL, stop_event)

    try:
        print("[1/4] Verifying FRDR SHA-256 checksums...", flush=True)
        checksum_results = _verify_frdr_checksums(nlbs_raw)
        for r in checksum_results:
            print(f"  {r['file']}: match={r['match']}", flush=True)

        print("[2/4] Extracting .7z archives into /nlbs_vol/nlbs/extracted/ (raw untouched)...", flush=True)
        for arch in ["normal.7z", "abnormal.7z", "False Positive.7z"]:
            arch_path = nlbs_raw / arch
            print(f"  Extracting {arch}...", flush=True)
            res = subprocess.run(
                ["7z", "x", "-y", "-aos", f"-o{nlbs_ext}", str(arch_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                raise RuntimeError(f"7z extraction failed for {arch}: {res.stderr[-500:]}")
            NLBS_VOL.commit()
            print(f"  {arch} extracted and committed.", flush=True)

        print("[3/4] Verifying 26,988 extracted DICOMs and reading headers...", flush=True)
        df = pd.read_csv(nlbs_raw / "NLBSD_Metadata.csv")
        df["rel_path"] = df["File Path"].astype(str).str.replace("\\", "/", regex=False)
        parts = df["rel_path"].str.split("/", expand=True)
        df["cohort_group"] = parts[0]
        df["case_id"] = parts[1]
        df["laterality_dir"] = parts[2]
        df["view_dir"] = parts[3]
        df["filename"] = parts[4]

        disk_dcms = {
            p.relative_to(nlbs_ext).as_posix()
            for p in nlbs_ext.rglob("*.dcm")
        }
        expected = set(df["rel_path"])
        missing_extracted = sorted(expected - disk_dcms)
        extra_extracted = sorted(disk_dcms - expected)

        study_uids = []
        series_uids = []
        manufacturers = []
        models = []
        photometrics = []
        bits_stored = []
        rows_list = []
        cols_list = []
        ts_uids = []
        patient_ids = []
        header_errors = []

        for idx, rel in enumerate(df["rel_path"]):
            p = nlbs_ext / rel
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                study_uids.append(str(getattr(ds, "StudyInstanceUID", "")))
                series_uids.append(str(getattr(ds, "SeriesInstanceUID", "")))
                manufacturers.append(str(getattr(ds, "Manufacturer", "")))
                models.append(str(getattr(ds, "ManufacturerModelName", "")))
                photometrics.append(str(getattr(ds, "PhotometricInterpretation", "")))
                bits_stored.append(int(getattr(ds, "BitsStored", 0) or 0))
                rows_list.append(int(getattr(ds, "Rows", 0) or 0))
                cols_list.append(int(getattr(ds, "Columns", 0) or 0))
                ts_uids.append(str(getattr(ds.file_meta, "TransferSyntaxUID", "")))
                patient_ids.append(str(getattr(ds, "PatientID", "")))
            except Exception as exc:
                header_errors.append({"rel_path": rel, "error": str(exc)})
                study_uids.append("")
                series_uids.append("")
                manufacturers.append("")
                models.append("")
                photometrics.append("")
                bits_stored.append(0)
                rows_list.append(0)
                cols_list.append(0)
                ts_uids.append("")
                patient_ids.append("")
            if (idx + 1) % 5000 == 0:
                print(f"  Read {idx + 1}/{len(df)} DICOM headers...", flush=True)

        df["StudyInstanceUID"] = study_uids
        df["SeriesInstanceUID"] = series_uids
        df["Manufacturer"] = manufacturers
        df["ManufacturerModelName"] = models
        df["PhotometricInterpretation"] = photometrics
        df["BitsStored"] = bits_stored
        df["Rows"] = rows_list
        df["Columns"] = cols_list
        df["TransferSyntaxUID"] = ts_uids
        df["dicom_PatientID"] = patient_ids

        # Build deterministic exam_id from (case_id, StudyInstanceUID)
        df["exam_key"] = df["case_id"] + "__" + df["StudyInstanceUID"]
        study_to_case_nunique = df.groupby("StudyInstanceUID")["case_id"].nunique()
        shared_studies_across_cases = int((study_to_case_nunique > 1).sum())

        # Assign clean sequential exam_id per case_id
        exam_rows = []
        for (case_id, study_uid), g in df.groupby(["case_id", "StudyInstanceUID"], sort=True):
            views = set(g["View Position"].dropna().astype(str))
            lats = set(g["Image Laterality"].dropna().astype(str))
            exam_rows.append({
                "case_id": case_id,
                "StudyInstanceUID": study_uid,
                "cohort_group": g["cohort_group"].iloc[0],
                "n_images": int(len(g)),
                "exam_cancer": int(g["Cancer"].max()),
                "n_cancer_images": int(g["Cancer"].sum()),
                "hard_to_state_negative": int(g["Hard to State Negative"].max()),
                "ages": "|".join(str(int(a)) for a in sorted(g["Age"].dropna().unique())),
                "n_distinct_ages": int(g["Age"].nunique()),
                "manufacturers": "|".join(sorted(set(g["Manufacturer"]) - {""})),
                "models": "|".join(sorted(set(g["ManufacturerModelName"]) - {""})),
                "has_L": int("L" in lats),
                "has_R": int("R" in lats),
                "has_CC": int("CC" in views),
                "has_MLO": int("MLO" in views),
            })

        exam_df = pd.DataFrame(exam_rows).sort_values(["cohort_group", "case_id", "StudyInstanceUID"])
        exam_df["visit_idx"] = exam_df.groupby("case_id").cumcount() + 1
        exam_df["exam_id"] = exam_df["case_id"] + "_v" + exam_df["visit_idx"].astype(str)

        # Merge exam_id back to image manifest
        df = df.merge(exam_df[["case_id", "StudyInstanceUID", "exam_id", "visit_idx"]], on=["case_id", "StudyInstanceUID"], how="left")

        print("[4/4] Building summary & manifests...", flush=True)
        exams_per_case = exam_df.groupby("case_id").size()
        p3280_exams = exam_df[exam_df["case_id"] == "p_3280"][
            ["exam_id", "n_images", "ages", "models", "StudyInstanceUID"]
        ].to_dict(orient="records")

        summary = {
            "checksums": checksum_results,
            "all_checksums_match": all(r.get("match", False) for r in checksum_results),
            "extracted_dcm_count": len(disk_dcms),
            "expected_dcm_count": len(expected),
            "missing_extracted_count": len(missing_extracted),
            "extra_extracted_count": len(extra_extracted),
            "header_error_count": len(header_errors),
            "header_errors_sample": header_errors[:10],
            "unique_case_ids": int(df["case_id"].nunique()),
            "unique_study_uids": int(df["StudyInstanceUID"].nunique()),
            "total_exams": int(len(exam_df)),
            "shared_studies_across_cases": shared_studies_across_cases,
            "exams_per_case_distribution": exams_per_case.value_counts().sort_index().to_dict(),
            "multi_visit_case_ids": int((exams_per_case > 1).sum()),
            "exams_with_multiple_ages": int((exam_df["n_distinct_ages"] > 1).sum()),
            "exam_n_images_distribution": exam_df["n_images"].value_counts().sort_index().to_dict(),
            "exam_cohort_counts": exam_df["cohort_group"].value_counts().to_dict(),
            "exam_cancer_positive": int((exam_df["exam_cancer"] == 1).sum()),
            "exam_cancer_negative": int((exam_df["exam_cancer"] == 0).sum()),
            "full_manufacturers": Counter(manufacturers).most_common(),
            "full_models": Counter(models).most_common(),
            "full_photometrics": Counter(photometrics).most_common(),
            "full_bits_stored": Counter(bits_stored).most_common(),
            "full_transfer_syntaxes": Counter(ts_uids).most_common(),
            "nonempty_dicom_patient_ids": int(sum(1 for x in patient_ids if x.strip())),
            "p_3280_exam_count": len(p3280_exams),
            "p_3280_sample_exams": p3280_exams[:10],
        }

        img_cols = [
            "cohort_group",
            "case_id",
            "exam_id",
            "visit_idx",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Image Laterality",
            "View Position",
            "Cancer",
            "Hard to State Negative",
            "Age",
            "Manufacturer",
            "ManufacturerModelName",
            "PhotometricInterpretation",
            "BitsStored",
            "Rows",
            "Columns",
            "TransferSyntaxUID",
            "Anode Target Material",
            "Energy Settings",
            "rel_path",
        ]
        img_csv = df[img_cols].sort_values(["cohort_group", "case_id", "visit_idx", "Image Laterality", "View Position", "rel_path"]).to_csv(index=False)
        exam_csv = exam_df[
            [
                "exam_id",
                "case_id",
                "visit_idx",
                "cohort_group",
                "StudyInstanceUID",
                "n_images",
                "exam_cancer",
                "n_cancer_images",
                "hard_to_state_negative",
                "ages",
                "n_distinct_ages",
                "manufacturers",
                "models",
                "has_L",
                "has_R",
                "has_CC",
                "has_MLO",
            ]
        ].to_csv(index=False)

        NLBS_VOL.commit()
        return {
            "summary": summary,
            "nlbs_manifest_csv": img_csv,
            "nlbs_exam_manifest_csv": exam_csv,
        }
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)
        try:
            NLBS_VOL.commit()
        except Exception:
            pass


@app.function(
    image=image,
    volumes={str(NLBS_MOUNT): NLBS_VOL},
    cpu=1.0,
    memory=2048,
    timeout=1800,
)
def verify_checksums_only() -> dict:
    """Verify FRDR SHA-256 checksums for all raw NLBS files."""
    nlbs_raw = NLBS_MOUNT / "nlbs" / "raw"
    results = _verify_frdr_checksums(nlbs_raw)
    return {
        "checksums": results,
        "all_checksums_match": all(r.get("match", False) for r in results),
    }


@app.local_entrypoint()
def main(checksums_only: bool = False) -> None:
    if checksums_only:
        chk = verify_checksums_only.remote()
        res_path = Path("reports/stage_a3_nlbs_results.json")
        data = json.loads(res_path.read_text(encoding="utf-8"))
        data["checksums"] = chk["checksums"]
        data["all_checksums_match"] = chk["all_checksums_match"]
        res_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"Updated checksums in {res_path}: all_match={chk['all_checksums_match']}")
        return
    result = run_stage_a3.remote()
    img_csv = result.pop("nlbs_manifest_csv")
    exam_csv = result.pop("nlbs_exam_manifest_csv")

    Path("datasets/nlbs").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    Path("datasets/nlbs/manifest.csv").write_text(img_csv, encoding="utf-8")
    Path("datasets/nlbs/exam_manifest.csv").write_text(exam_csv, encoding="utf-8")
    Path("reports/stage_a3_nlbs_results.json").write_text(
        json.dumps(result["summary"], indent=2, default=str), encoding="utf-8"
    )
    print("Saved datasets/nlbs/manifest.csv, datasets/nlbs/exam_manifest.csv, reports/stage_a3_nlbs_results.json")
