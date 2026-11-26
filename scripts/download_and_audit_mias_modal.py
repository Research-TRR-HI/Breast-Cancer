"""Download and audit the official MIAS v1.21 dataset from Cambridge Apollo on CPU.

Strictly follows Modal cost-optimization rules:
- cpu=1.0, memory=2048, NO GPU attached.
- Periodic volume.commit() every 60s + SIGINT/SIGTERM handler.
- Resumable HTTP streaming with .part -> atomic rename, MD5 verification, and SHA-256.
- Writes ONLY under /external_datasets/mias/raw/ and /cache/mias/ on rsna-mammography-vol.

Usage:
    $env:MODAL_PROFILE='tigerwood697'
    modal run scripts/download_and_audit_mias_modal.py --out-json reports/paper1/expanded/exp3/imdb_mias/mias_raw_audit.json
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import signal
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import modal

APP_NAME = "paper1-exp3-mias-cpu"
VOLUME_NAME = "rsna-mammography-vol"
MOUNT_PATH = Path("/vol")
RAW_DIR = MOUNT_PATH / "external_datasets" / "mias" / "raw"
CACHE_DIR = MOUNT_PATH / "cache" / "mias"
COMMIT_INTERVAL_SEC = 60

APOLLO_FILES = [
    {
        "name": "miasdbv1.21.zip",
        "url": "https://api.repository.cam.ac.uk/server/api/core/bitstreams/5960ab2b-5ea2-4db1-96ac-15b3605e7485/content",
        "expected_bytes": 1617327652,
        "expected_md5": "d790185baed18ab199efc85388b75dfc",
    },
    {
        "name": "license.txt",
        "url": "https://api.repository.cam.ac.uk/server/api/core/bitstreams/aeed3b6a-b646-43c1-877d-9e83f3cd664b/content",
        "expected_bytes": 3893,
        "expected_md5": "87eda9de84448d1f82354d60eee3eb5f",
    },
    {
        "name": "license_rdf",
        "url": "https://api.repository.cam.ac.uk/server/api/core/bitstreams/fd7af62c-f348-4963-8f92-842b08a4182d/content",
        "expected_bytes": 1370,
        "expected_md5": "e9fbe0c18c54a5a24f2661023416007e",
    },
]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests>=2.31.0",
    "numpy>=1.26.0",
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
        print(f"[signal] Caught signal {signum}. Committing volume...", flush=True)
        stop_event.set()
        try:
            vol.commit()
        except Exception as exc:
            print(f"[signal] Commit failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def _hash_file(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            md5.update(chunk)
            sha256.update(chunk)
            total += len(chunk)
    return md5.hexdigest(), sha256.hexdigest(), total


def _download_with_resume(
    url: str,
    dest: Path,
    expected_bytes: int,
    expected_md5: str,
) -> dict[str, Any]:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_name(dest.name + ".part")

    if dest.exists():
        md5_hex, sha256_hex, size_bytes = _hash_file(dest)
        if size_bytes == expected_bytes and md5_hex == expected_md5:
            print(f"[skip] {dest.name} already verified ({size_bytes} bytes).", flush=True)
            return {
                "name": dest.name,
                "path": str(dest),
                "size_bytes": size_bytes,
                "expected_bytes": expected_bytes,
                "md5": md5_hex,
                "expected_md5": expected_md5,
                "md5_match": True,
                "sha256": sha256_hex,
                "resumed_or_skipped": "skipped_verified",
                "elapsed_sec": 0.0,
                "throughput_mb_s": None,
            }
        print(f"[warn] {dest.name} failed verification; re-downloading.", flush=True)
        dest.unlink()

    existing_bytes = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    t0 = time.monotonic()
    downloaded_this_run = 0
    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as resp:
        if resp.status_code == 416:
            os.replace(part_path, dest)
        else:
            resp.raise_for_status()
            is_resume = resp.status_code == 206 and existing_bytes > 0
            mode = "ab" if is_resume else "wb"
            with part_path.open(mode) as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded_this_run += len(chunk)
            os.replace(part_path, dest)

    elapsed = max(time.monotonic() - t0, 1e-6)
    md5_hex, sha256_hex, size_bytes = _hash_file(dest)
    if size_bytes != expected_bytes:
        raise RuntimeError(
            f"Size mismatch for {dest.name}: got {size_bytes}, expected {expected_bytes}"
        )
    if md5_hex != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {dest.name}: got {md5_hex}, expected {expected_md5}"
        )

    mb_s = (downloaded_this_run / (1024 * 1024)) / elapsed
    print(
        f"[ok] {dest.name}: {size_bytes} bytes in {elapsed:.2f}s ({mb_s:.2f} MB/s), MD5={md5_hex}",
        flush=True,
    )
    return {
        "name": dest.name,
        "path": str(dest),
        "size_bytes": size_bytes,
        "expected_bytes": expected_bytes,
        "md5": md5_hex,
        "expected_md5": expected_md5,
        "md5_match": True,
        "sha256": sha256_hex,
        "resumed_or_skipped": "downloaded",
        "downloaded_bytes": downloaded_this_run,
        "elapsed_sec": round(elapsed, 3),
        "throughput_mb_s": round(mb_s, 3),
    }


def _audit_single_pgm(path_str: str) -> dict[str, Any]:
    import numpy as np

    path = Path(path_str)
    raw = path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()

    # Parse PGM header (P5 binary or P2 ASCII)
    idx = 0
    n = len(raw)
    tokens: list[bytes] = []
    comments: list[str] = []

    while len(tokens) < 4 and idx < n:
        # Skip whitespace
        while idx < n and raw[idx] in b" \t\r\n":
            idx += 1
        if idx >= n:
            break
        if raw[idx : idx + 1] == b"#":
            end = raw.find(b"\n", idx)
            if end == -1:
                end = n
            comments.append(raw[idx:end].decode("ascii", errors="replace"))
            idx = end + 1
            continue
        start = idx
        while idx < n and raw[idx] not in b" \t\r\n#":
            idx += 1
        tokens.append(raw[start:idx])

    if len(tokens) < 4:
        return {
            "filename": path.name,
            "rel_path": str(path),
            "size_bytes": len(raw),
            "file_sha256": file_sha256,
            "decode_ok": False,
            "error": f"Incomplete PGM header: {tokens}",
        }

    magic = tokens[0].decode("ascii", errors="replace")
    width = int(tokens[1])
    height = int(tokens[2])
    maxval = int(tokens[3])

    # Exactly one whitespace byte separates maxval from raster data in P5
    if idx < n and raw[idx] in b" \t\r\n":
        data_offset = idx + 1
    else:
        data_offset = idx

    payload = raw[data_offset:]
    expected_pixels = width * height

    if magic == "P5":
        dtype = np.uint8 if maxval < 256 else np.dtype(">u2")
        bytes_per_px = 1 if maxval < 256 else 2
        expected_len = expected_pixels * bytes_per_px
        actual_len = len(payload)
        if actual_len != expected_len:
            return {
                "filename": path.name,
                "size_bytes": len(raw),
                "file_sha256": file_sha256,
                "decode_ok": False,
                "magic": magic,
                "width": width,
                "height": height,
                "maxval": maxval,
                "error": f"Payload length {actual_len} != expected {expected_len}",
            }
        arr = np.frombuffer(payload, dtype=dtype).reshape((height, width))
    elif magic == "P2":
        vals = np.fromstring(payload.decode("ascii", errors="replace"), sep=" ", dtype=np.int32)
        if vals.size != expected_pixels:
            return {
                "filename": path.name,
                "size_bytes": len(raw),
                "file_sha256": file_sha256,
                "decode_ok": False,
                "magic": magic,
                "width": width,
                "height": height,
                "maxval": maxval,
                "error": f"P2 pixel count {vals.size} != {expected_pixels}",
            }
        arr = vals.reshape((height, width))
    else:
        return {
            "filename": path.name,
            "size_bytes": len(raw),
            "file_sha256": file_sha256,
            "decode_ok": False,
            "magic": magic,
            "error": f"Unsupported PGM magic: {magic}",
        }

    pixel_sha256 = hashlib.sha256(arr.tobytes()).hexdigest()
    left_half_mean = float(arr[:, : width // 2].mean())
    right_half_mean = float(arr[:, width // 2 :].mean())
    top_half_mean = float(arr[: height // 2, :].mean())
    bottom_half_mean = float(arr[height // 2 :, :].mean())

    return {
        "filename": path.name,
        "rel_path": str(path.relative_to(RAW_DIR / "extracted")),
        "size_bytes": len(raw),
        "file_sha256": file_sha256,
        "pixel_sha256": pixel_sha256,
        "decode_ok": True,
        "magic": magic,
        "width": width,
        "height": height,
        "maxval": maxval,
        "dtype": str(arr.dtype),
        "bit_depth": 8 if maxval < 256 else 16,
        "pixel_min": int(arr.min()),
        "pixel_max": int(arr.max()),
        "pixel_mean": round(float(arr.mean()), 4),
        "pixel_std": round(float(arr.std()), 4),
        "left_half_mean": round(left_half_mean, 3),
        "right_half_mean": round(right_half_mean, 3),
        "top_half_mean": round(top_half_mean, 3),
        "bottom_half_mean": round(bottom_half_mean, 3),
        "inferred_breast_side": "L_chest_left" if left_half_mean > right_half_mean else "R_chest_right",
        "comments": comments,
    }


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): volume},
    cpu=1.0,
    memory=2048,
    timeout=3600,
)
def download_extract_and_audit_mias() -> dict[str, Any]:
    """Download MIAS v1.21 to /external_datasets/mias/raw/, extract, and audit all files."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    extract_dir = RAW_DIR / "extracted"

    stop_event = threading.Event()
    _register_signal_handlers(volume, stop_event)
    commit_thread = _start_periodic_commit(volume, stop_event)

    try:
        download_records = []
        for spec in APOLLO_FILES:
            rec = _download_with_resume(
                url=spec["url"],
                dest=RAW_DIR / spec["name"],
                expected_bytes=spec["expected_bytes"],
                expected_md5=spec["expected_md5"],
            )
            download_records.append(rec)

        zip_path = RAW_DIR / "miasdbv1.21.zip"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            zip_entries = [
                {
                    "filename": m.filename,
                    "file_size": m.file_size,
                    "compress_size": m.compress_size,
                    "is_dir": m.is_dir(),
                }
                for m in members
            ]
            # Extract only if missing or count mismatch
            existing_files = [p for p in extract_dir.rglob("*") if p.is_file()]
            non_dir_members = [m for m in members if not m.is_dir()]
            if len(existing_files) != len(non_dir_members):
                print(f"[extract] Extracting {len(non_dir_members)} files to {extract_dir}...", flush=True)
                zf.extractall(extract_dir)
            else:
                print(f"[extract] All {len(existing_files)} extracted files already present.", flush=True)

        volume.commit()

        all_extracted = sorted([p for p in extract_dir.rglob("*") if p.is_file()])
        pgm_files = [p for p in all_extracted if p.suffix.lower() == ".pgm"]
        non_pgm_files = [p for p in all_extracted if p.suffix.lower() != ".pgm"]

        text_artifacts: dict[str, Any] = {}
        for p in non_pgm_files + [RAW_DIR / "license.txt", RAW_DIR / "license_rdf"]:
            rel = str(p.relative_to(RAW_DIR))
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                content = f"<unreadable: {exc}>"
            _, sha256_hex, size_b = _hash_file(p)
            text_artifacts[rel] = {
                "size_bytes": size_b,
                "sha256": sha256_hex,
                "content": content,
            }

        print(f"[audit] Auditing {len(pgm_files)} PGM files with multiprocessing...", flush=True)
        t_audit = time.monotonic()
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
            pgm_records = list(pool.map(_audit_single_pgm, [str(p) for p in pgm_files], chunksize=16))
        audit_sec = round(time.monotonic() - t_audit, 3)

        # Duplicate detection
        by_pixel_hash: dict[str, list[str]] = {}
        by_file_hash: dict[str, list[str]] = {}
        for rec in pgm_records:
            if rec.get("decode_ok"):
                by_pixel_hash.setdefault(rec["pixel_sha256"], []).append(rec["filename"])
                by_file_hash.setdefault(rec["file_sha256"], []).append(rec["filename"])

        dup_pixels = {k: v for k, v in by_pixel_hash.items() if len(v) > 1}
        dup_files = {k: v for k, v in by_file_hash.items() if len(v) > 1}

        result = {
            "apollo_item_url": "https://www.repository.cam.ac.uk/items/b6a97f0c-3b9b-40ad-8f18-3d121eef1459",
            "apollo_doi": "https://doi.org/10.17863/CAM.105113",
            "downloads": download_records,
            "zip_entry_count": len(zip_entries),
            "extracted_file_count": len(all_extracted),
            "pgm_count": len(pgm_files),
            "non_pgm_count": len(non_pgm_files),
            "audit_elapsed_sec": audit_sec,
            "duplicate_file_hashes": dup_files,
            "duplicate_pixel_hashes": dup_pixels,
            "text_artifacts": text_artifacts,
            "pgm_records": pgm_records,
        }

        cache_out = CACHE_DIR / "mias_raw_audit.json"
        cache_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        volume.commit()
        print(f"[done] Audit complete. Saved to {cache_out}", flush=True)
        return result
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)
        try:
            volume.commit()
        except Exception:
            pass


@app.local_entrypoint()
def main(out_json: str = "reports/paper1/expanded/exp3/imdb_mias/mias_raw_audit.json") -> None:
    result = download_extract_and_audit_mias.remote()
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[local] Wrote audit JSON to {out_path}")
    print(
        f"[summary] pgm_count={result['pgm_count']}, "
        f"non_pgm_count={result['non_pgm_count']}, "
        f"dup_files={len(result['duplicate_file_hashes'])}, "
        f"dup_pixels={len(result['duplicate_pixel_hashes'])}"
    )
