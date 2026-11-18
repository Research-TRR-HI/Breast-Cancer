"""Download official Mammo-CLIP EfficientNet-B5 Pretrained Checkpoint to Modal Volume.

Strictly adheres to cost & safety rules:
- cpu=1.0, memory=2048, NO GPU attached.
- Stores checkpoint in Modal Volume rsna-mammography-vol at /models/mammo_clip/.
- Verifies Content-Length and SHA-256 after download.
- Periodic volume.commit() every 60s.

Checkpoint details:
- HuggingFace repository: shawn24/Mammo-CLIP
- File: Pre-trained-checkpoints/b5-model-best-epoch-7.tar
- Expected size: 1,654,145,453 bytes (~1.65 GB)
- License: CC BY-NC-SA 4.0 (Non-commercial research only)

Usage:
    modal run scripts/download_mammoclip_modal.py
"""

import hashlib
import os
import signal
import sys
import threading
from pathlib import Path
import modal

APP_NAME = "anchorshift-mammoclip-download"
RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol")
MOUNT_PATH = Path("/data")
COMMIT_INTERVAL_SEC = 60

PINNED_REVISION = "2f356926a00fc3f0d9fdad1193c1464fd9adf564"
CHECKPOINT_URL = (
    f"https://huggingface.co/shawn24/Mammo-CLIP/resolve/{PINNED_REVISION}/"
    "Pre-trained-checkpoints/b5-model-best-epoch-7.tar"
)
EXPECTED_SIZE = 1654145453
EXPECTED_SHA256 = "0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests>=2.31.0",
    "tqdm>=4.66.0",
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


@app.function(
    image=image,
    volumes={str(MOUNT_PATH): RSNA_VOL},
    cpu=1.0,
    memory=2048,
    timeout=3600,
)
def download_mammoclip_checkpoint(force: bool = False) -> dict:
    """Download b5-model-best-epoch-7.tar into Modal Volume."""
    import requests
    from tqdm import tqdm

    target_dir = MOUNT_PATH / "models" / "mammo_clip"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "b5-model-best-epoch-7.tar"

    stop_event = threading.Event()
    _register_signal_handlers(RSNA_VOL, stop_event)
    commit_thread = _start_periodic_commit(RSNA_VOL, stop_event)

    try:
        if target_file.exists() and not force:
            sz = target_file.stat().st_size
            if sz == EXPECTED_SIZE:
                print(f"[download] Checkpoint already exists ({sz} bytes). Skipping.", flush=True)
                return {"status": "already_exists", "path": str(target_file), "size": sz}

        print(f"[download] Streaming {CHECKPOINT_URL} -> {target_file}", flush=True)
        temp_file = target_file.with_suffix(".tar.tmp")
        h = hashlib.sha256()

        with requests.get(CHECKPOINT_URL, stream=True, timeout=60, allow_redirects=True) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers.get("Content-Length", EXPECTED_SIZE))
            downloaded = 0

            with open(temp_file, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading Mammo-CLIP b5",
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        h.update(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))

        temp_file.rename(target_file)
        RSNA_VOL.commit()
        sha256_hash = h.hexdigest()
        print(f"[download] Download complete: {target_file} ({target_file.stat().st_size} bytes)", flush=True)
        print(f"[download] SHA-256: {sha256_hash}", flush=True)
        if sha256_hash != EXPECTED_SHA256:
            raise ValueError(f"Checksum mismatch: expected {EXPECTED_SHA256}, got {sha256_hash}")

        return {
            "status": "downloaded",
            "path": str(target_file),
            "size": target_file.stat().st_size,
            "sha256": sha256_hash,
            "sha256_matches_expected": sha256_hash == EXPECTED_SHA256,
            "pinned_revision": PINNED_REVISION,
            "size_matches_expected": target_file.stat().st_size == EXPECTED_SIZE,
        }
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)
        try:
            RSNA_VOL.commit()
        except Exception:
            pass


@app.local_entrypoint()
def main(force: bool = False) -> None:
    res = download_mammoclip_checkpoint.remote(force=force)
    print("Download result:", res)
