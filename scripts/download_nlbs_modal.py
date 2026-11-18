"""Download the NLBS (NL-Breast-Screening) dataset from FRDR to a Modal Volume.

Follows strict cost optimization:
- cpu=1.0, memory=2048, NO GPU attached.
- Periodic volume.commit() every 60s + SIGINT/SIGTERM handler so progress is
  preserved if interrupted.
- Resumable HTTP streaming with live tqdm progress bar and optional extraction.

Usage:
    modal run scripts/download_nlbs_modal.py
    # Or detached with unzip + cleanup:
    modal run --detach scripts/download_nlbs_modal.py --unzip --delete-zip
"""

import signal
import sys
import threading
import zipfile
from pathlib import Path

import modal

APP_NAME = "anchorshift-nlbs-download"
VOLUME_NAME = "nlbs-mammography-vol"
MOUNT_PATH = Path("/data")
DEFAULT_URL = (
    "https://www.frdr-dfdr.ca/repo/downloadzip/"
    "29f1ad65bb5847759310d2bd20ae33bb1da349bcd9d94426940789554d3ea774/"
    "4834/FRDR_dataset_1521_download_321_202611170213.zip"
)
COMMIT_INTERVAL_SEC = 60

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests>=2.31.0",
    "tqdm>=4.66.0",
)


def _start_periodic_commit(vol: modal.Volume, stop_event: threading.Event) -> threading.Thread:
    """Commit Modal volume periodically in a background thread."""

    def _loop() -> None:
        while not stop_event.wait(COMMIT_INTERVAL_SEC):
            try:
                vol.commit()
                print("\n[modal] Periodic volume.commit() completed.", flush=True)
            except Exception as exc:
                print(f"\n[modal] Periodic commit warning: {exc}", file=sys.stderr, flush=True)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def _register_signal_handlers(vol: modal.Volume, stop_event: threading.Event) -> None:
    """Ensure partial downloads are committed on SIGINT/SIGTERM."""

    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"\n[signal] Caught signal {signum}. Committing volume before exit...", flush=True)
        stop_event.set()
        try:
            vol.commit()
            print("[signal] Volume committed safely.", flush=True)
        except Exception as exc:
            print(f"[signal] Commit failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): volume},
    cpu=1.0,
    memory=2048,
    ephemeral_disk=524288,  # 512 GiB in MiB for zip + extraction
    timeout=86400,
)
def download_nlbs(
    url: str = DEFAULT_URL,
    unzip: bool = False,
    delete_zip: bool = False,
) -> None:
    """Download NLBS zip from FRDR into Modal Volume with resume support."""
    import requests
    from tqdm import tqdm

    target_dir = MOUNT_PATH / "nlbs"
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "nlbs.zip"
    part_path = target_dir / "nlbs.zip.part"

    stop_event = threading.Event()
    _register_signal_handlers(volume, stop_event)
    commit_thread = _start_periodic_commit(volume, stop_event)

    try:
        if not zip_path.exists():
            existing_bytes = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes > 0 else {}

            print(f"[download] Requesting NLBS archive from FRDR -> {part_path}", flush=True)
            with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as resp:
                if resp.status_code == 416:
                    # Range not satisfiable -> file already complete
                    part_path.rename(zip_path)
                else:
                    resp.raise_for_status()
                    is_resume = resp.status_code == 206 and existing_bytes > 0
                    mode = "ab" if is_resume else "wb"
                    initial = existing_bytes if is_resume else 0

                    content_len = resp.headers.get("Content-Length")
                    total = (initial + int(content_len)) if content_len else None

                    with (
                        open(part_path, mode) as fh,
                        tqdm(
                            total=total,
                            initial=initial,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc="Downloading NLBS",
                            mininterval=5.0,
                        ) as pbar,
                    ):
                        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                                pbar.update(len(chunk))

                    part_path.rename(zip_path)

            volume.commit()
            print(
                f"[download] NLBS archive saved ({zip_path.stat().st_size / (1024**3):.2f} GB) and committed.",
                flush=True,
            )
        else:
            print(f"[download] Found existing {zip_path}, skipping download.", flush=True)

        if unzip:
            extract_dir = target_dir / "raw"
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"[unzip] Extracting {zip_path} -> {extract_dir}", flush=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.infolist()
                for member in tqdm(members, desc="Extracting NLBS", unit="file", mininterval=5.0):
                    out_file = extract_dir / member.filename
                    if out_file.exists() and out_file.stat().st_size == member.file_size:
                        continue
                    zf.extract(member, extract_dir)

            volume.commit()
            print("[unzip] Extraction complete and committed.", flush=True)

            if delete_zip:
                zip_path.unlink()
                volume.commit()
                print(f"[cleanup] Removed {zip_path} and committed.", flush=True)
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)
        try:
            volume.commit()
        except Exception:
            pass


@app.local_entrypoint()
def main(
    url: str = DEFAULT_URL,
    unzip: bool = False,
    delete_zip: bool = False,
) -> None:
    download_nlbs.remote(url=url, unzip=unzip, delete_zip=delete_zip)
