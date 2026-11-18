"""Download the RSNA Screening Mammography dataset to a Modal Volume.

Follows strict cost optimization:
- cpu=1.0, memory=2048, NO GPU attached.
- Periodic volume.commit() every 60s + SIGINT/SIGTERM handler so progress is
  preserved if interrupted.
- Live tqdm progress bar for download and optional extraction.

Prerequisites:
1. Accept the competition rules on Kaggle:
   https://www.kaggle.com/competitions/rsna-breast-cancer-detection/rules
2. Create a Modal secret named "kaggle-secret" containing KAGGLE_USERNAME and KAGGLE_KEY:
   modal secret create kaggle-secret KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>
3. Run:
   modal run scripts/download_rsna_modal.py
   # Or with extraction after download:
   modal run scripts/download_rsna_modal.py --unzip
"""

import os
import signal
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import modal

APP_NAME = "anchorshift-rsna-download"
VOLUME_NAME = "rsna-mammography-vol"
MOUNT_PATH = Path("/data")
COMPETITION = "rsna-breast-cancer-detection"
COMMIT_INTERVAL_SEC = 60

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "kaggle",
    "six>=1.16.0",
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
    secrets=[modal.Secret.from_name("kaggle-secret")],
    cpu=1.0,
    memory=2048,
    ephemeral_disk=1048576,  # 1 TiB in MiB so 314+ GB zip + extraction fits cleanly
    timeout=86400,
)
def download_rsna(
    unzip: bool = False, delete_zip: bool = False, smoke_test: bool = False
) -> None:
    """Download RSNA Breast Cancer Detection dataset into Modal Volume."""
    import time
    from tqdm import tqdm

    has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    has_legacy = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not (has_token or has_legacy):
        raise RuntimeError(
            "Missing Kaggle auth in Modal secret 'kaggle-secret'. "
            "Set KAGGLE_API_TOKEN or both KAGGLE_USERNAME and KAGGLE_KEY."
        )

    target_dir = MOUNT_PATH / "rsna"
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / f"{COMPETITION}.zip"

    stop_event = threading.Event()
    _register_signal_handlers(volume, stop_event)
    commit_thread = _start_periodic_commit(volume, stop_event)

    def _monitor_progress(watch_dir: Path, done_event: threading.Event) -> None:
        start = time.time()
        while not done_event.wait(15):
            total_bytes = sum(
                p.stat().st_size for p in watch_dir.rglob("*") if p.is_file()
            )
            elapsed = int(time.time() - start)
            gb = total_bytes / (1024**3)
            mb = total_bytes / (1024**2)
            rate = mb / max(elapsed, 1)
            print(
                f"[progress] {gb:.2f} GB ({mb:.1f} MB) on disk | elapsed {elapsed}s | avg {rate:.1f} MB/s",
                flush=True,
            )

    try:
        cmd = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(target_dir),
        ]
        if smoke_test:
            cmd.extend(["-f", "train.csv"])
            print(f"[smoke-test] Downloading only train.csv from '{COMPETITION}' -> {target_dir}", flush=True)
        else:
            print(f"[download] Starting Kaggle download for '{COMPETITION}' -> {target_dir}", flush=True)

        monitor_done = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_progress, args=(target_dir, monitor_done), daemon=True
        )
        monitor_thread.start()

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            result = subprocess.run(cmd, env=env, check=False)
        finally:
            monitor_done.set()
            monitor_thread.join(timeout=5)

        if result.returncode != 0:
            raise RuntimeError(
                f"Kaggle CLI exited with code {result.returncode}. "
                "Verify you accepted competition rules on Kaggle and your secret is valid."
            )

        volume.commit()
        files = sorted(p for p in target_dir.rglob("*") if p.is_file())
        for p in files:
            print(f"[volume] {p} ({p.stat().st_size / (1024**2):.2f} MB)", flush=True)
        print("[download] Download complete and committed to Modal Volume.", flush=True)

        if smoke_test:
            print("[smoke-test] PASSED: Kaggle auth, competition access, and Modal Volume write verified.", flush=True)
            return

        if unzip:
            if not zip_path.exists():
                raise FileNotFoundError(f"Expected zip not found at {zip_path}")

            extract_dir = target_dir / "raw"
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"[unzip] Extracting {zip_path} -> {extract_dir}", flush=True)

            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.infolist()
                for member in tqdm(members, desc="Extracting RSNA", unit="file"):
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
def main(unzip: bool = False, delete_zip: bool = False, smoke_test: bool = False) -> None:
    download_rsna.remote(unzip=unzip, delete_zip=delete_zip, smoke_test=smoke_test)
