"""Parallel Cloud-to-Cloud Modal Volume Transfer & Fresh-Container Verification.

Transfers verified datasets, Stage-1 uint16 PNG caches, and pretrained weights
between Modal workspaces directly inside Modal cloud containers (`CPU ONLY`,
`1.0 vCPU / 2048 MiB`, ZERO GPUs, ZERO local PC bulk data routing).

Security & Isolation:
    - Never runs `modal profile activate` or switches global profiles.
    - Always requires explicit `--dst-profile` and binds destination resources
      to that profile via `MODAL_PROFILE=<dst_profile>` at CLI invocation.
    - Reads source tokens ONLY locally from `~/.modal.toml` into ephemeral
      `modal.Secret.from_dict(...)` at container launch; never logs or commits tokens.
    - Rejects every NLBS label-bearing artifact (`exam_manifest.csv`,
      `datasets/nlbs/manifest.csv`, `/nlbs/raw/*`).

Example (`$0.00` plan inspection):
    python -m scripts.transfer_modal_volumes_cloud --dst-profile tigerwood693 --bundle dinov2 --plan-only

Example (approved 1.5 GiB Pilot using 16 concurrent streams = 8 containers x 2 streams):
    $env:MODAL_PROFILE="tigerwood693"
    $env:MODAL_SRC_PROFILES="hasinishrak74001,hasinishrak2015"
    modal --profile tigerwood693 run scripts/transfer_modal_volumes_cloud.py --dst-profile tigerwood693 --bundle dinov2 --pilot-max-gib 1.5 --num-containers 8 --streams-per-container 2
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:  # Windows local dry-run / unit tests
    resource = None  # type: ignore[assignment]
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.modal_transfer import (
    DEFAULT_WORKER_CPU,
    DEFAULT_WORKER_MEMORY_MIB,
    EXPECTED_STAGE1_WEIGHT_SHA256,
    TransferEntry,
    assert_transfer_path_allowed,
    async_transfer_shard,
    atomic_write_json,
    entries_to_dicts,
    estimate_transfer_compute_cost_usd,
    estimate_worker_hourly_cost_usd,
    get_bundle_specs,
    load_modal_profile_credentials,
    map_source_file_to_dest,
    partition_entries_by_bytes,
    redact_secrets,
    select_pilot_entries,
    select_pilot_slice,
)


APP_NAME = "paper1-cloud-volume-transfer"
REPORTS_DIR = Path("reports/transfers")
DST_RSNA_MOUNT = Path("/dst_rsna")
DST_NLBS_MOUNT = Path("/dst_nlbs")

TRANSFER_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pillow==12.3.0")
    .add_local_python_source("src")
)

DST_RSNA_VOL = modal.Volume.from_name("rsna-mammography-vol", create_if_missing=True)


def _build_local_src_secret() -> modal.Secret:
    """Create ephemeral Secret holding ONLY requested read-only source credentials.

    On local CLI launch (`modal.is_local() == True`), reads `MODAL_SRC_PROFILES`
    (comma-separated, defaulting to `hasinishrak74001,hasinishrak2015`) from `~/.modal.toml`.
    Inside remote containers (`modal.is_local() == False`), returns an empty placeholder
    while Modal injects the uploaded secret env vars into `os.environ`.
    """
    if not modal.is_local():
        return modal.Secret.from_dict({})
    raw_profiles = os.environ.get("MODAL_SRC_PROFILES", "").strip()
    if not raw_profiles:
        return modal.Secret.from_dict({})
    profiles = [p.strip() for p in raw_profiles.split(",") if p.strip()]
    creds = load_modal_profile_credentials(profiles)
    env_payload: Dict[str, str] = {}
    for prof, (tid, tsec) in creds.items():
        safe_key = prof.upper()
        env_payload[f"SRC_TOKEN_ID_{safe_key}"] = tid
        env_payload[f"SRC_TOKEN_SECRET_{safe_key}"] = tsec
    return modal.Secret.from_dict(env_payload)


SRC_CREDS_SECRET = _build_local_src_secret()
app = modal.App(APP_NAME, image=TRANSFER_IMAGE)


def _mount_root_for_volume(volume_name: str) -> Path:
    if volume_name == "rsna-mammography-vol":
        return DST_RSNA_MOUNT
    if volume_name == "nlbs-mammography-vol":
        return DST_NLBS_MOUNT
    raise ValueError(f"Unsupported destination volume '{volume_name}'")


@app.function(
    cpu=DEFAULT_WORKER_CPU,
    memory=DEFAULT_WORKER_MEMORY_MIB,
    timeout=21600,
    secrets=[SRC_CREDS_SECRET],
    volumes={str(DST_RSNA_MOUNT): DST_RSNA_VOL},
)
def transfer_shard_rsna_worker(
    job_id: str,
    shard_id: int,
    entries: List[Dict[str, Any]],
    streams_per_container: int = 2,
    commit_interval_s: float = 60.0,
) -> Dict[str, Any]:
    """Stream a shard of files directly from source Modal Volume(s) to `/dst_rsna`."""
    return asyncio.run(
        async_transfer_shard(
            job_id=job_id,
            shard_id=shard_id,
            entries=entries,
            dst_vol=DST_RSNA_VOL,
            mount_root=DST_RSNA_MOUNT,
            streams_per_container=streams_per_container,
            commit_interval_s=commit_interval_s,
        )
    )


@app.function(
    cpu=DEFAULT_WORKER_CPU,
    memory=DEFAULT_WORKER_MEMORY_MIB,
    timeout=7200,
    volumes={str(DST_RSNA_MOUNT): DST_RSNA_VOL},
)
def verify_destination_rsna_shard(
    job_id: str,
    shard_id: int,
    png_decode_spot_check_n: int = 8,
) -> Dict[str, Any]:
    """Fresh-container check: verify sizes, SHA-256 digests, and PNG decode on `/dst_rsna`."""
    from PIL import Image

    manifest_path = DST_RSNA_MOUNT / "_transfer_manifests" / job_id / f"shard_{shard_id:04d}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing shard manifest in fresh container: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    png_spot_indices = set()
    for idx, item in enumerate(files):
        if str(item.get("dst_path", "")).lower().endswith(".png"):
            png_spot_indices.add(idx)
            if len(png_spot_indices) >= png_decode_spot_check_n:
                break

    def _verify_one(idx_item: Tuple[int, Dict[str, Any]]) -> Tuple[bool, Optional[str], Optional[str], bool]:
        idx, item = idx_item
        dst_path = str(item["dst_path"])
        expected_size = int(item["size_bytes"])
        expected_sha = item.get("sha256")
        fpath = DST_RSNA_MOUNT / dst_path
        if not fpath.exists():
            return False, f"MISSING:{dst_path}", None, False
        actual_size = fpath.stat().st_size
        if actual_size != expected_size:
            return False, f"SIZE:{dst_path}:{actual_size}!={expected_size}", None, False

        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            while True:
                buf = f.read(1024 * 1024)
                if not buf:
                    break
                h.update(buf)
        actual_sha = h.hexdigest()
        pinned_weight_sha = EXPECTED_STAGE1_WEIGHT_SHA256.get(dst_path)
        if pinned_weight_sha and actual_sha.lower() != pinned_weight_sha.lower():
            return False, f"WEIGHT_SHA256:{dst_path}:{actual_sha}!={pinned_weight_sha}", None, False
        if expected_sha and actual_sha != expected_sha:
            return False, f"SHA256:{dst_path}:{actual_sha}!={expected_sha}", None, False

        png_err: Optional[str] = None
        png_ok = False
        if idx in png_spot_indices:
            try:
                with Image.open(fpath) as im:
                    im.verify()
                with Image.open(fpath) as im2:
                    im2.load()
                png_ok = True
            except Exception as exc:
                png_err = f"{dst_path}:{exc}"
        return True, None, png_err, png_ok

    checked = 0
    mismatches: List[str] = []
    png_checked = 0
    png_failures: List[str] = []

    with ThreadPoolExecutor(max_workers=16) as pool:
        for ok, mismatch, png_err, png_ok in pool.map(_verify_one, enumerate(files)):
            if mismatch:
                mismatches.append(mismatch)
            elif ok:
                checked += 1
            if png_err:
                png_failures.append(png_err)
            elif png_ok:
                png_checked += 1

    return {
        "shard_id": shard_id,
        "total_listed": len(files),
        "verified_ok": checked,
        "png_spot_checked": png_checked,
        "mismatches": mismatches[:20],
        "png_failures": png_failures[:20],
        "passed": (len(mismatches) == 0 and len(png_failures) == 0 and checked == len(files)),
    }


def collect_source_inventory_free(bundle_name: str) -> List[TransferEntry]:
    """Discover exact files and byte sizes via FREE Modal SDK `Volume.listdir`.

    Uses explicit local credentials from `~/.modal.toml` per source profile so
    `MODAL_PROFILE` never leaks across accounts and zero containers start ($0.00).
    """
    specs = get_bundle_specs(bundle_name)
    by_profile: Dict[str, List[Any]] = {}
    for spec in specs:
        by_profile.setdefault(spec.src_profile, []).append(spec)

    creds = load_modal_profile_credentials(by_profile.keys())
    entries: List[TransferEntry] = []
    seen_dst: set[Tuple[str, str]] = set()

    for prof, prof_specs in by_profile.items():
        tid, tsec = creds[prof]
        client = modal.Client.from_credentials(tid, tsec)
        vol_cache: Dict[str, modal.Volume] = {}
        for spec in prof_specs:
            if spec.src_volume not in vol_cache:
                vol_cache[spec.src_volume] = modal.Volume.from_name(spec.src_volume, client=client)
            vol = vol_cache[spec.src_volume]
            listed = vol.listdir(f"/{spec.src_prefix}", recursive=True)
            for fe in listed:
                # Only regular files (`FileEntryType.FILE` == 1)
                if int(fe.type) != 1:
                    continue
                src_rel = assert_transfer_path_allowed(fe.path, context=f"plan:{prof}")
                dst_rel = map_source_file_to_dest(src_rel, spec)
                key = (spec.dst_volume, dst_rel)
                if key in seen_dst:
                    continue
                seen_dst.add(key)
                entries.append(
                    TransferEntry(
                        src_profile=prof,
                        src_volume=spec.src_volume,
                        src_path=src_rel,
                        dst_volume=spec.dst_volume,
                        dst_path=dst_rel,
                        size_bytes=int(fe.size),
                    )
                )
    entries.sort(key=lambda e: (e.dst_volume, e.dst_path))
    return entries


def _build_plan_summary(
    dst_profile: str,
    bundle_name: str,
    entries: Sequence[TransferEntry],
    num_containers: int,
    streams_per_container: int,
    pilot_max_gib: Optional[float],
) -> Dict[str, Any]:
    active = select_pilot_entries(entries, max_gib=pilot_max_gib) if pilot_max_gib else list(entries)
    total_bytes = sum(e.size_bytes for e in active)
    by_src: Dict[str, Dict[str, Any]] = {}
    for e in active:
        bucket = by_src.setdefault(e.src_profile, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += e.size_bytes
    for bucket in by_src.values():
        bucket["gib"] = round(bucket["bytes"] / (1024.0**3), 4)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dst_profile": dst_profile,
        "bundle": bundle_name,
        "mode": "pilot" if pilot_max_gib else "full",
        "pilot_max_gib": pilot_max_gib,
        "total_files": len(active),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024.0**3), 4),
        "num_containers": num_containers,
        "streams_per_container": streams_per_container,
        "effective_concurrent_workers": num_containers * streams_per_container,
        "per_container_usd_hr": round(estimate_worker_hourly_cost_usd(), 5),
        "by_source_profile": by_src,
        "cost_projections": {
            "conservative_50mbps": estimate_transfer_compute_cost_usd(total_bytes, 50.0, num_containers),
            "baseline_120mbps": estimate_transfer_compute_cost_usd(total_bytes, 120.0, num_containers),
            "fast_250mbps": estimate_transfer_compute_cost_usd(total_bytes, 250.0, num_containers),
        },
    }


@app.local_entrypoint()
def main(
    dst_profile: str,
    bundle: str,
    num_containers: int = 8,
    streams_per_container: int = 2,
    pilot_max_gib: float = 0.0,
    slice_index: int = 0,
    num_slices: int = 1,
    job_tag: str = "",
    max_budget_usd: float = 1.0,
    verify: bool = True,
    verify_only: bool = False,
    plan_only: bool = False,
) -> None:
    """Modal entrypoint executing strictly on `dst_profile`."""
    active_profile = os.environ.get("MODAL_PROFILE", "").strip() or str(
        getattr(modal.config, "_profile", "") or ""
    ).strip()
    if active_profile != dst_profile:
        raise RuntimeError(
            f"MODAL_PROFILE mismatch: expected '{dst_profile}', got '{active_profile}'. "
            "Always pass --profile <dst> and set MODAL_PROFILE=<dst> explicitly."
        )

    if verify_only:
        job_id = job_tag.strip()
        if not job_id:
            raise ValueError("--verify-only requires --job-tag")
        t_v0 = time.monotonic()
        verify_args = [(job_id, shard_idx, 8) for shard_idx in range(num_containers)]
        verify_results = list(verify_destination_rsna_shard.starmap(verify_args))
        verify_wall_s = max(0.001, time.monotonic() - t_v0)
        summary = {
            "job_id": job_id,
            "dst_profile": dst_profile,
            "bundle": bundle,
            "mode": "verify_only",
            "verify_wall_seconds": round(verify_wall_s, 3),
            "num_containers": num_containers,
            "verify_results": verify_results,
            "all_verified": all(v.get("passed", False) for v in verify_results),
        }
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(REPORTS_DIR / f"{job_id}_verify.json", summary)
        print(json.dumps(summary, indent=2))
        return

    effective_workers = num_containers * streams_per_container
    if effective_workers < 16 and not plan_only:
        raise ValueError(
            f"Effective concurrency ({effective_workers}) must be >= 16 workers "
            "(adjust --num-containers / --streams-per-container)."
        )

    entries = collect_source_inventory_free(bundle)
    pilot_cap = pilot_max_gib if pilot_max_gib > 0 else None
    active_entries = select_pilot_entries(entries, max_gib=pilot_cap) if pilot_cap else list(entries)
    if num_slices > 1:
        active_entries = select_pilot_slice(active_entries, slice_index=slice_index, num_slices=num_slices)

    plan = _build_plan_summary(
        dst_profile=dst_profile,
        bundle_name=bundle,
        entries=active_entries,
        num_containers=num_containers,
        streams_per_container=streams_per_container,
        pilot_max_gib=None,
    )
    plan["mode"] = "pilot" if pilot_cap else "full"
    plan["pilot_max_gib"] = pilot_cap
    plan["slice_index"] = slice_index
    plan["num_slices"] = num_slices

    job_id = (
        job_tag.strip()
        if job_tag.strip()
        else f"{dst_profile}_{bundle}_{'pilot' if pilot_cap else 'full'}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plan_out = REPORTS_DIR / f"{job_id}_plan.json"
    atomic_write_json(plan_out, plan)
    print(json.dumps(plan, indent=2))

    if plan_only:
        return

    conservative_cost = float(plan["cost_projections"]["conservative_50mbps"]["estimated_cost_usd"])
    if conservative_cost > max_budget_usd:
        raise RuntimeError(
            f"Conservative estimated cost ${conservative_cost:.4f} exceeds --max-budget-usd=${max_budget_usd:.4f}"
        )

    # Current Modal entrypoint supports rsna-mammography-vol workers cleanly first.
    rsna_selected = [e for e in active_entries if e.dst_volume == "rsna-mammography-vol"]
    shards = partition_entries_by_bytes(rsna_selected, num_shards=num_containers)

    t0 = time.monotonic()
    shard_args = [
        (job_id, shard_idx, entries_to_dicts(shard), streams_per_container, 60.0)
        for shard_idx, shard in enumerate(shards)
    ]
    transfer_results = list(transfer_shard_rsna_worker.starmap(shard_args))
    elapsed_wall_s = max(0.001, time.monotonic() - t0)

    verify_results: List[Dict[str, Any]] = []
    verify_wall_s = 0.0
    if verify:
        t_v0 = time.monotonic()
        verify_args = [(job_id, r["shard_id"], 8) for r in transfer_results]
        verify_results = list(verify_destination_rsna_shard.starmap(verify_args))
        verify_wall_s = max(0.001, time.monotonic() - t_v0)

    total_xfer_bytes = sum(int(r["stats"]["transferred_bytes"]) for r in transfer_results)
    total_skip_bytes = sum(int(r["stats"]["skipped_bytes"]) for r in transfer_results)
    xfer_container_s = sum(float(r["elapsed_s"]) for r in transfer_results)
    max_worker_s = max((float(r["elapsed_s"]) for r in transfer_results), default=0.0)
    hourly_rate = estimate_worker_hourly_cost_usd()
    est_container_bill_usd = ((xfer_container_s + verify_wall_s * len(shards)) / 3600.0) * hourly_rate
    summary = {
        "job_id": job_id,
        "dst_profile": dst_profile,
        "bundle": bundle,
        "slice_index": slice_index,
        "num_slices": num_slices,
        "transfer_wall_seconds": round(elapsed_wall_s, 3),
        "startup_and_scheduling_overhead_s": round(max(0.0, elapsed_wall_s - max_worker_s), 3),
        "verify_wall_seconds": round(verify_wall_s, 3),
        "aggregate_worker_container_s": round(xfer_container_s, 3),
        "worker_throughput_mb_per_s": round((total_xfer_bytes / (1024.0**2)) / max(0.001, max_worker_s), 3),
        "wall_throughput_mb_per_s": round((total_xfer_bytes / (1024.0**2)) / elapsed_wall_s, 3),
        "estimated_container_compute_usd": round(est_container_bill_usd, 6),
        "transferred_bytes": total_xfer_bytes,
        "skipped_bytes": total_skip_bytes,
        "num_containers": len(shards),
        "streams_per_container": streams_per_container,
        "effective_concurrent_workers": len(shards) * streams_per_container,
        "transfer_results": transfer_results,
        "verify_results": verify_results,
        "all_verified": all(v.get("passed", False) for v in verify_results) if verify else None,
    }
    atomic_write_json(REPORTS_DIR / f"{job_id}_execution.json", summary)
    print(json.dumps(summary, indent=2))


def _parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Free ($0.00) Modal transfer plan generator.")
    parser.add_argument("--dst-profile", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--num-containers", type=int, default=8)
    parser.add_argument("--streams-per-container", type=int, default=2)
    parser.add_argument("--pilot-max-gib", type=float, default=0.0)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli_args()
    if not args.plan_only:
        raise SystemExit(
            "Local __main__ only runs --plan-only ($0.00). For remote execution run `modal --profile <dst> run scripts/transfer_modal_volumes_cloud.py ...`"
        )
    all_entries = collect_source_inventory_free(args.bundle)
    plan_data = _build_plan_summary(
        dst_profile=args.dst_profile,
        bundle_name=args.bundle,
        entries=all_entries,
        num_containers=args.num_containers,
        streams_per_container=args.streams_per_container,
        pilot_max_gib=args.pilot_max_gib if args.pilot_max_gib > 0 else None,
    )
    print(json.dumps(plan_data, indent=2))
