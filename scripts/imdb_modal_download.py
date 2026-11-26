"""CPU-only Modal downloader, concurrency benchmarker, and DICOM auditor for IMDB (IBIA).

Targets `MODAL_PROFILE=tigerwood697` explicitly (NEVER calls `modal profile activate`).
Allocates `cpu=1.0, memory=2048` ONLY (NO GPUs).
Writes ONLY under `/external_datasets/imdb/raw/{r1,r2}/` and `/cache/imdb/` on `rsna-mammography-vol`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import modal

REQUIRED_MODAL_PROFILE = "tigerwood697"
APP_NAME = "paper1-exp3-imdb-cpu"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
RSNA_MOUNT = Path("/rsna_vol")
IBIA_BASE = "https://ibdc.dbt.gov.in"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pydicom>=2.4.0",
    "pylibjpeg>=2.0.0",
    "pylibjpeg-libjpeg>=2.1.0",
    "requests>=2.31.0",
)


def _verify_modal_profile() -> None:
    prof = os.environ.get("MODAL_PROFILE", "").strip()
    if prof != REQUIRED_MODAL_PROFILE:
        raise RuntimeError(
            f"MODAL_PROFILE must be {REQUIRED_MODAL_PROFILE!r}, got {prof!r}. "
            "NEVER call `modal profile activate`."
        )


def _download_one(
    session_or_none: Any,
    rel_url: str,
    dest_path: Path,
    timeout: int = 60,
    retries: int = 3,
) -> Dict[str, Any]:
    """Download a single DICOM file atomically if not already present and valid."""
    url = IBIA_BASE + rel_url
    filename = Path(rel_url).name
    if dest_path.exists() and dest_path.stat().st_size > 132:
        with dest_path.open("rb") as f:
            f.seek(128)
            magic = f.read(4)
        if magic == b"DICM":
            return {
                "filename": filename,
                "rel_url": rel_url,
                "status": "skipped_existing",
                "bytes": dest_path.stat().st_size,
                "elapsed_s": 0.0,
                "error": "",
            }

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".dcm.tmp")
    t0 = time.monotonic()
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            if session_or_none is not None:
                resp = session_or_none.get(url, timeout=timeout, stream=True)
                resp.raise_for_status()
                h = hashlib.sha256()
                total = 0
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            h.update(chunk)
                            total += len(chunk)
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read()
                h = hashlib.sha256(data)
                total = len(data)
                tmp_path.write_bytes(data)

            if total <= 132:
                raise ValueError(f"File too small ({total} bytes)")
            with tmp_path.open("rb") as f:
                f.seek(128)
                if f.read(4) != b"DICM":
                    raise ValueError("Missing DICM magic header")
            tmp_path.replace(dest_path)
            return {
                "filename": filename,
                "rel_url": rel_url,
                "status": "downloaded",
                "bytes": total,
                "sha256": h.hexdigest(),
                "elapsed_s": time.monotonic() - t0,
                "error": "",
            }
        except Exception as exc:
            last_err = f"attempt {attempt}: {exc}"
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            time.sleep(min(2.0 * attempt, 5.0))

    return {
        "filename": filename,
        "rel_url": rel_url,
        "status": "failed",
        "bytes": 0,
        "sha256": "",
        "elapsed_s": time.monotonic() - t0,
        "error": last_err,
    }


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=3600,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
)
def bench_streams_remote(urls: List[str], stream_counts: List[int]) -> List[Dict[str, Any]]:
    """Benchmark 8/16/32 concurrent HTTP download streams into /tmp (no volume clutter)."""
    import requests
    from requests.adapters import HTTPAdapter

    results: List[Dict[str, Any]] = []
    for n_streams in stream_counts:
        bench_dir = Path(f"/tmp/imdb_bench_{n_streams}")
        if bench_dir.exists():
            for p in bench_dir.glob("*"):
                p.unlink(missing_ok=True)
        bench_dir.mkdir(parents=True, exist_ok=True)

        sess = requests.Session()
        adapter = HTTPAdapter(pool_connections=n_streams, pool_maxsize=n_streams)
        sess.mount("https://", adapter)
        sess.headers.update({"User-Agent": "Mozilla/5.0"})

        t0 = time.monotonic()
        ok = 0
        fail = 0
        total_bytes = 0
        with ThreadPoolExecutor(max_workers=n_streams) as ex:
            futs = [
                ex.submit(_download_one, sess, u, bench_dir / Path(u).name)
                for u in urls
            ]
            for fut in as_completed(futs):
                res = fut.result()
                if res["status"] in ("downloaded", "skipped_existing"):
                    ok += 1
                    total_bytes += res["bytes"]
                else:
                    fail += 1
        dt = time.monotonic() - t0
        for p in bench_dir.glob("*"):
            p.unlink(missing_ok=True)
        results.append(
            {
                "streams": n_streams,
                "n_files": len(urls),
                "ok": ok,
                "fail": fail,
                "total_bytes": total_bytes,
                "total_mb": round(total_bytes / (1024 * 1024), 2),
                "elapsed_s": round(dt, 2),
                "throughput_mb_s": round((total_bytes / (1024 * 1024)) / max(dt, 1e-6), 2),
                "files_per_s": round(ok / max(dt, 1e-6), 2),
            }
        )
        print(f"[BENCH streams={n_streams}] {results[-1]}")
    return results


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=14400,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
)
def download_release_remote(
    release: str,
    urls: List[str],
    max_workers: int = 16,
) -> Dict[str, Any]:
    """Download one IMDB release (r1 or r2) into /external_datasets/imdb/raw/<release>/ with periodic commits."""
    import requests
    from requests.adapters import HTTPAdapter

    out_dir = RSNA_MOUNT / "external_datasets" / "imdb" / "raw" / release
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = RSNA_MOUNT / "cache" / "imdb"
    cache_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: Any) -> None:
        print(f"[{release}] Received signal {signum}, committing volume...")
        stop_event.set()
        RSNA_VOL.commit()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def _periodic_commit() -> None:
        while not stop_event.wait(60.0):
            print(f"[{release}] Periodic 60s volume.commit()...")
            RSNA_VOL.commit()

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    sess = requests.Session()
    adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    t0 = time.monotonic()
    records: List[Dict[str, Any]] = []
    downloaded = 0
    skipped = 0
    failed = 0
    total_bytes = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(_download_one, sess, u, out_dir / Path(u).name)
                for u in urls
            ]
            for idx, fut in enumerate(as_completed(futs), start=1):
                res = fut.result()
                records.append(res)
                if res["status"] == "downloaded":
                    downloaded += 1
                    total_bytes += res["bytes"]
                elif res["status"] == "skipped_existing":
                    skipped += 1
                    total_bytes += res["bytes"]
                else:
                    failed += 1
                if idx % 250 == 0 or idx == len(urls):
                    dt = time.monotonic() - t0
                    print(
                        f"[{release}] {idx}/{len(urls)} "
                        f"(dl={downloaded}, skip={skipped}, fail={failed}, "
                        f"{total_bytes/(1024**3):.2f} GiB, {dt:.1f}s)"
                    )
    finally:
        stop_event.set()
        commit_thread.join(timeout=5.0)
        RSNA_VOL.commit()

    dt = time.monotonic() - t0
    summary = {
        "release": release,
        "total_urls": len(urls),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024**3), 4),
        "elapsed_s": round(dt, 2),
        "throughput_mb_s": round((total_bytes / (1024 * 1024)) / max(dt, 1e-6), 2),
        "max_workers": max_workers,
    }
    (cache_dir / f"{release}_download_summary.json").write_text(json.dumps(summary, indent=2))
    with (cache_dir / f"{release}_download_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["filename", "rel_url", "status", "bytes", "sha256", "elapsed_s", "error"],
        )
        w.writeheader()
        for r in sorted(records, key=lambda x: x["filename"]):
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    RSNA_VOL.commit()
    return summary


@app.function(
    image=image,
    cpu=16.0,
    memory=8192,
    timeout=3600,
    volumes={str(RSNA_MOUNT): RSNA_VOL},
)
def audit_release_dicoms_remote(release: str) -> Dict[str, Any]:
    """Audit all downloaded DICOMs in /external_datasets/imdb/raw/<release>/ (SHA-256 + DICOM tags)."""
    import pydicom

    raw_dir = RSNA_MOUNT / "external_datasets" / "imdb" / "raw" / release
    cache_dir = RSNA_MOUNT / "cache" / "imdb"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dcm_files = sorted(raw_dir.glob("*.dcm"))
    rows: List[Dict[str, Any]] = []
    t0 = time.monotonic()
    total_bytes = 0
    corrupt = 0

    def _inspect(p: Path) -> Dict[str, Any]:
        h = hashlib.sha256()
        sz = 0
        with p.open("rb") as f:
            while True:
                chunk = f.read(1024 * 256)
                if not chunk:
                    break
                h.update(chunk)
                sz += len(chunk)
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            image_type = "\\".join(str(x) for x in getattr(ds, "ImageType", [])) if hasattr(ds, "ImageType") else ""
            return {
                "release": release,
                "filename": p.name,
                "bytes": sz,
                "sha256": h.hexdigest(),
                "PatientID": str(getattr(ds, "PatientID", "")),
                "PatientAge": str(getattr(ds, "PatientAge", "")),
                "PatientSex": str(getattr(ds, "PatientSex", "")),
                "Manufacturer": str(getattr(ds, "Manufacturer", "")),
                "ManufacturerModelName": str(getattr(ds, "ManufacturerModelName", "")),
                "Modality": str(getattr(ds, "Modality", "")),
                "ImageType": image_type,
                "PresentationIntentType": str(getattr(ds, "PresentationIntentType", "")),
                "PhotometricInterpretation": str(getattr(ds, "PhotometricInterpretation", "")),
                "Rows": str(getattr(ds, "Rows", "")),
                "Columns": str(getattr(ds, "Columns", "")),
                "BitsAllocated": str(getattr(ds, "BitsAllocated", "")),
                "BitsStored": str(getattr(ds, "BitsStored", "")),
                "ViewPosition": str(getattr(ds, "ViewPosition", "")),
                "ImageLaterality": str(getattr(ds, "ImageLaterality", "")),
                "Laterality": str(getattr(ds, "Laterality", "")),
                "SOPInstanceUID": str(getattr(ds, "SOPInstanceUID", "")),
                "StudyInstanceUID": str(getattr(ds, "StudyInstanceUID", "")),
                "SeriesInstanceUID": str(getattr(ds, "SeriesInstanceUID", "")),
                "error": "",
            }
        except Exception as exc:
            return {
                "release": release,
                "filename": p.name,
                "bytes": sz,
                "sha256": h.hexdigest(),
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=16) as ex:
        for idx, res in enumerate(ex.map(_inspect, dcm_files), start=1):
            rows.append(res)
            total_bytes += res["bytes"]
            if res.get("error"):
                corrupt += 1
            if idx % 1000 == 0 or idx == len(dcm_files):
                print(f"[AUDIT {release}] {idx}/{len(dcm_files)} files inspected ({time.monotonic()-t0:.1f}s)")

    out_csv = cache_dir / f"{release}_dicom_audit.csv"
    fieldnames = [
        "release", "filename", "bytes", "sha256", "PatientID", "PatientAge", "PatientSex",
        "Manufacturer", "ManufacturerModelName", "Modality", "ImageType",
        "PresentationIntentType", "PhotometricInterpretation", "Rows", "Columns",
        "BitsAllocated", "BitsStored", "ViewPosition", "ImageLaterality", "Laterality",
        "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "error",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    RSNA_VOL.commit()
    return {
        "release": release,
        "num_files": len(dcm_files),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024**3), 4),
        "corrupt": corrupt,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "audit_csv": str(out_csv),
    }


@app.local_entrypoint()
def main(
    stage: str = "bench",
    urls_json: str = "",
    release: str = "both",
    max_workers: int = 16,
    bench_limit: int = 24,
) -> None:
    _verify_modal_profile()
    if stage == "bench":
        payload = json.loads(Path(urls_json).read_text(encoding="utf-8"))
        sample_urls = payload["r1"][:bench_limit]
        res = bench_streams_remote.remote(sample_urls, [8, 16, 32])
        print("BENCH_RESULT:", json.dumps(res, indent=2))
    elif stage == "download":
        payload = json.loads(Path(urls_json).read_text(encoding="utf-8"))
        targets = ["r1", "r2"] if release == "both" else [release]
        for rel in targets:
            res = download_release_remote.remote(rel, payload[rel], max_workers=max_workers)
            print(f"DOWNLOAD_RESULT_{rel.upper()}:", json.dumps(res, indent=2))
    elif stage == "audit":
        targets = ["r1", "r2"] if release == "both" else [release]
        for res in audit_release_dicoms_remote.map(targets):
            rel = res["release"]
            print(f"AUDIT_RESULT_{rel.upper()}:", json.dumps(res, indent=2))
    else:
        raise ValueError(f"Unknown stage: {stage}")
