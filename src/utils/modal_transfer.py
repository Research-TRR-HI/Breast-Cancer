"""Cloud-to-Cloud Modal Volume Transfer Planning, Quarantine, Sharding & Cost Utilities.

All helpers here are pure-Python and strictly read-only/offline safe so they can
be unit-tested locally ($0.00) and imported by `scripts/transfer_modal_volumes_cloud.py`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Verified Modal Starter billing rates (`modal billing rates --json`, 2026-11-24).
MODAL_CPU_HOUR_USD: float = 0.04730
MODAL_MEM_GIB_HOUR_USD: float = 0.00800
MODAL_VOLUME_GIB_MONTH_USD: float = 0.09000
MODAL_FREE_STORAGE_GIB: float = 1024.0  # 1 TiB continuous free storage tier

DEFAULT_WORKER_CPU: float = 1.0
DEFAULT_WORKER_MEMORY_MIB: int = 2048

# Strictly quarantined paths/filenames (NLBS outcome labels & local secrets).
QUARANTINED_EXACT_FILENAMES: Tuple[str, ...] = (
    "exam_manifest.csv",
    "nlbsd_metadata.csv",
    "normal.7z",
    "false positive.7z",
    "abnormal.7z",
    ".modal.toml",
    "git commit.md",
    "personal_info.md",
)

QUARANTINED_PATH_SUBSTRINGS: Tuple[str, ...] = (
    "/nlbs/raw",
    "datasets/nlbs/manifest.csv",
    "datasets/nlbs/exam_manifest.csv",
    "nlbsd_metadata",
)

EXPECTED_STAGE1_WEIGHT_SHA256: Dict[str, str] = {
    "models/paper1/dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    "models/paper1/resnet50-11ad3fa6.pth": "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
    "models/paper1/sample_image_model.p": "9a1d97f93493ea2007c00e6c1805dfa634cbc321ecdeba2c2adecd351e86b014",
    "models/mammo_clip/b5-model-best-epoch-7.tar": "0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c",
}


@dataclass(frozen=True)
class SourceSpec:
    """One prefix/file mapping from a specific source Modal profile & volume."""

    src_profile: str
    src_volume: str
    src_prefix: str
    dst_volume: str
    dst_prefix: str
    description: str = ""


@dataclass(frozen=True)
class TransferEntry:
    """Single deterministic file transfer item (credential-free)."""

    src_profile: str
    src_volume: str
    src_path: str
    dst_volume: str
    dst_path: str
    size_bytes: int


def normalize_vol_path(path: str) -> str:
    """Normalize volume-relative POSIX path without leading slash."""
    cleaned = str(path).replace("\\", "/").strip()
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    return cleaned.lstrip("/")


def assert_transfer_path_allowed(path: str, context: str = "transfer") -> str:
    """Enforce strict NLBS target-label & secret quarantine on every path."""
    rel = normalize_vol_path(path)
    lowered = f"/{rel.lower()}"
    base = Path(rel).name.lower()

    if base in QUARANTINED_EXACT_FILENAMES:
        raise ValueError(
            f"QUARANTINE VIOLATION in {context}: forbidden file '{base}' at '{rel}'"
        )
    for bad in QUARANTINED_PATH_SUBSTRINGS:
        if bad in lowered:
            raise ValueError(
                f"QUARANTINE VIOLATION in {context}: forbidden path '{bad}' matched '{rel}'"
            )
    if rel == "nlbs/manifest.csv":
        raise ValueError(
            f"QUARANTINE VIOLATION in {context}: label-bearing '{rel}' is blocked"
        )
    return rel


def get_bundle_specs(bundle_name: str) -> List[SourceSpec]:
    """Return explicit deduplicated Source->Dest volume specs for a workload bundle.

    Prefers `hasinishrak74001` for Stage-1 RSNA (`65.95 GiB`), `splits_seed42.csv`,
    and Mammo-CLIP/Stage-B artifacts to avoid extra load on `hasinishrak2015`.
    """
    if bundle_name == "nyu_nlbs_failover":
        specs = [
            SourceSpec(
                src_profile="hasinishrak2015",
                src_volume="rsna-mammography-vol",
                src_prefix="cache/paper1_nyu_uint16_v1/nlbs_4view",
                dst_volume="rsna-mammography-vol",
                dst_prefix="cache/paper1_nyu_uint16_v1/nlbs_4view",
                description="Verified 24,736 NLBS four-view PNG cache",
            ),
            SourceSpec(
                src_profile="hasinishrak2015",
                src_volume="rsna-mammography-vol",
                src_prefix="models/paper1/sample_image_model.p",
                dst_volume="rsna-mammography-vol",
                dst_prefix="models/paper1/sample_image_model.p",
                description="Official NYU image-only four-view pretrained checkpoint",
            ),
        ]
        for spec in specs:
            assert_transfer_path_allowed(spec.src_prefix, context=f"bundle:{bundle_name}:src")
            assert_transfer_path_allowed(spec.dst_prefix, context=f"bundle:{bundle_name}:dst")
        return specs

    if bundle_name == "dryousuf_rsna_raw":
        specs = [
            SourceSpec(
                src_profile="hasinishrak2015",
                src_volume="rsna-mammography-vol",
                src_prefix="rsna/raw",
                dst_volume="rsna-mammography-vol",
                dst_prefix="rsna/raw",
                description="RSNA raw DICOM archive for native GMIC/NYU fine-tuning",
            )
        ]
        for spec in specs:
            assert_transfer_path_allowed(spec.src_prefix, context=f"bundle:{bundle_name}:src")
            assert_transfer_path_allowed(spec.dst_prefix, context=f"bundle:{bundle_name}:dst")
        return specs

    if bundle_name in (
        "dryousuf_gmic_eval_rsna_val_calib",
        "dryousuf_gmic_eval_nlbs",
        "dryousuf_gmic_eval_cache",
    ):
        specs = []
        if bundle_name in ("dryousuf_gmic_eval_rsna_val_calib", "dryousuf_gmic_eval_cache"):
            specs.append(
                SourceSpec(
                    src_profile="hasinishrak2015",
                    src_volume="rsna-mammography-vol",
                    src_prefix="cache/paper1_gmic_uint16_v1/rsna_val_calib",
                    dst_volume="rsna-mammography-vol",
                    dst_prefix="cache/paper1_gmic_uint16_v1/rsna_val_calib",
                    description="Verified 10,982 RSNA val_calib GMIC (2944x1920) uint16 PNG eval cache (~62.06 GiB)",
                )
            )
        if bundle_name in ("dryousuf_gmic_eval_nlbs", "dryousuf_gmic_eval_cache"):
            specs.append(
                SourceSpec(
                    src_profile="hasinishrak2015",
                    src_volume="rsna-mammography-vol",
                    src_prefix="cache/paper1_gmic_uint16_v1/nlbs",
                    dst_volume="rsna-mammography-vol",
                    dst_prefix="cache/paper1_gmic_uint16_v1/nlbs",
                    description="Verified 26,988 label-free NLBS GMIC (2944x1920) uint16 PNG eval cache (~97.31 GiB)",
                )
            )
        for spec in specs:
            assert_transfer_path_allowed(spec.src_prefix, context=f"bundle:{bundle_name}:src")
            assert_transfer_path_allowed(spec.dst_prefix, context=f"bundle:{bundle_name}:dst")
        return specs

    specs: List[SourceSpec] = [
        SourceSpec(
            src_profile="hasinishrak74001",
            src_volume="rsna-mammography-vol",
            src_prefix="cache/stage1_rsna_uint16",
            dst_volume="rsna-mammography-vol",
            dst_prefix="cache/stage1_rsna_uint16",
            description="Verified 54,706 RSNA Stage-1 uint16 1520x912 PNGs + manifests",
        ),
        SourceSpec(
            src_profile="hasinishrak74001",
            src_volume="rsna-mammography-vol",
            src_prefix="rsna/splits_seed42.csv",
            dst_volume="rsna-mammography-vol",
            dst_prefix="rsna/splits_seed42.csv",
            description="Canonical RSNA seed42 split table",
        ),
        SourceSpec(
            src_profile="hasinishrak2015",
            src_volume="rsna-mammography-vol",
            src_prefix="cache/stage1_nlbs_uint16",
            dst_volume="rsna-mammography-vol",
            dst_prefix="cache/stage1_nlbs_uint16",
            description="Verified 26,988 label-free NLBS Stage-1 uint16 1520x912 PNGs + shard_diags",
        ),
    ]

    if bundle_name == "dinov2":
        specs.append(
            SourceSpec(
                src_profile="hasinishrak2015",
                src_volume="rsna-mammography-vol",
                src_prefix="models/paper1/dinov2_vits14_pretrain.pth",
                dst_volume="rsna-mammography-vol",
                dst_prefix="models/paper1/dinov2_vits14_pretrain.pth",
                description="Official DINOv2 ViT-S/14 backbone weights",
            )
        )
    elif bundle_name == "resnet50":
        specs.append(
            SourceSpec(
                src_profile="hasinishrak2015",
                src_volume="rsna-mammography-vol",
                src_prefix="models/paper1/resnet50-11ad3fa6.pth",
                dst_volume="rsna-mammography-vol",
                dst_prefix="models/paper1/resnet50-11ad3fa6.pth",
                description="Official torchvision ResNet-50 V2 weights",
            )
        )
    elif bundle_name in (
        "dryousuf_stage1_models",
        "dryousuf_stage1_plus_nyu",
        "dryousuf_full_mirror",
    ):
        specs.extend(
            [
                SourceSpec(
                    src_profile="hasinishrak2015",
                    src_volume="rsna-mammography-vol",
                    src_prefix="models/paper1",
                    dst_volume="rsna-mammography-vol",
                    dst_prefix="models/paper1",
                    description="All 4 Paper 1 backbone checkpoints",
                ),
                SourceSpec(
                    src_profile="hasinishrak74001",
                    src_volume="rsna-mammography-vol",
                    src_prefix="models/mammo_clip",
                    dst_volume="rsna-mammography-vol",
                    dst_prefix="models/mammo_clip",
                    description="Mammo-CLIP EfficientNet-B5 Ensem checkpoint",
                ),
                SourceSpec(
                    src_profile="hasinishrak74001",
                    src_volume="rsna-mammography-vol",
                    src_prefix="models/checkpoints",
                    dst_volume="rsna-mammography-vol",
                    dst_prefix="models/checkpoints",
                    description="Stage-B RSNA CE & TRUST checkpoints",
                ),
            ]
        )
        if bundle_name in ("dryousuf_stage1_plus_nyu", "dryousuf_full_mirror"):
            specs.append(
                SourceSpec(
                    src_profile="hasinishrak2015",
                    src_volume="rsna-mammography-vol",
                    src_prefix="cache/paper1_nyu_uint16_v1",
                    dst_volume="rsna-mammography-vol",
                    dst_prefix="cache/paper1_nyu_uint16_v1",
                    description="Verified 34,264 NYU four-view PNG cache (RSNA + NLBS)",
                )
            )
        if bundle_name == "dryousuf_full_mirror":
            specs.extend(
                [
                    SourceSpec(
                        src_profile="hasinishrak2015",
                        src_volume="rsna-mammography-vol",
                        src_prefix="rsna/raw",
                        dst_volume="rsna-mammography-vol",
                        dst_prefix="rsna/raw",
                        description="RSNA raw DICOM archive",
                    ),
                    SourceSpec(
                        src_profile="hasinishrak2015",
                        src_volume="nlbs-mammography-vol",
                        src_prefix="nlbs/extracted",
                        dst_volume="nlbs-mammography-vol",
                        dst_prefix="nlbs/extracted",
                        description="Label-free extracted NLBS DICOMs (excluding nlbs/raw)",
                    ),
                ]
            )
    else:
        raise ValueError(f"Unknown transfer bundle '{bundle_name}'")

    for spec in specs:
        assert_transfer_path_allowed(spec.src_prefix, context=f"bundle:{bundle_name}:src")
        assert_transfer_path_allowed(spec.dst_prefix, context=f"bundle:{bundle_name}:dst")
    return specs


def map_source_file_to_dest(entry_path: str, spec: SourceSpec) -> str:
    """Map a concrete file path under `spec.src_prefix` to its destination volume path."""
    rel = assert_transfer_path_allowed(entry_path, context=spec.src_profile)
    src_pfx = normalize_vol_path(spec.src_prefix)
    dst_pfx = normalize_vol_path(spec.dst_prefix)

    if rel == src_pfx:
        return assert_transfer_path_allowed(dst_pfx, context="dst")
    if rel.startswith(src_pfx + "/"):
        suffix = rel[len(src_pfx) + 1 :]
        mapped = f"{dst_pfx}/{suffix}" if dst_pfx else suffix
        return assert_transfer_path_allowed(mapped, context="dst")
    raise ValueError(f"File '{rel}' does not belong to prefix '{src_pfx}'")


def partition_entries_by_bytes(
    entries: Sequence[TransferEntry],
    num_shards: int,
) -> List[List[TransferEntry]]:
    """Deterministically partition entries into byte-balanced shards (LPT greedy)."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not entries:
        return []

    actual_shards = min(num_shards, len(entries))
    sorted_items = sorted(
        entries,
        key=lambda item: (-int(item.size_bytes), item.dst_volume, item.dst_path),
    )

    heap: List[Tuple[int, int]] = [(0, idx) for idx in range(actual_shards)]
    heapq.heapify(heap)
    buckets: List[List[TransferEntry]] = [[] for _ in range(actual_shards)]

    for item in sorted_items:
        total_bytes, idx = heapq.heappop(heap)
        buckets[idx].append(item)
        heapq.heappush(heap, (total_bytes + max(1, int(item.size_bytes)), idx))

    # Restore deterministic path order within each bucket for readable logs/resume.
    for bucket in buckets:
        bucket.sort(key=lambda item: (item.dst_volume, item.dst_path))
    return [b for b in buckets if b]


def select_pilot_entries(
    entries: Sequence[TransferEntry],
    max_gib: float = 1.5,
) -> List[TransferEntry]:
    """Select a deterministic, multi-source <= `max_gib` subset for Pilot benchmarking.

    Round-robins across `(src_profile, dir_bucket)` groups after preferring small
    metadata/weight files (< 150 MiB) so every Pilot validates both source accounts
    (`hasinishrak74001`, `hasinishrak2015`) and actual image streams within budget.
    """
    if max_gib <= 0:
        raise ValueError("max_gib must be > 0")
    budget_bytes = int(max_gib * (1024**3))

    groups: Dict[Tuple[str, str], List[TransferEntry]] = {}
    for entry in entries:
        assert_transfer_path_allowed(entry.src_path, context="pilot")
        assert_transfer_path_allowed(entry.dst_path, context="pilot")
        parent = "/".join(entry.dst_path.split("/")[:2])
        groups.setdefault((entry.src_profile, parent), []).append(entry)

    # Sort deterministically: put smaller files first in each group so Pilot packs many files.
    for key in groups:
        groups[key].sort(key=lambda e: (int(e.size_bytes), e.dst_path))

    ordered_keys = sorted(groups.keys())
    cursors: Dict[Tuple[str, str], int] = {k: 0 for k in ordered_keys}
    selected: List[TransferEntry] = []
    used_bytes = 0

    progressed = True
    while progressed:
        progressed = False
        for key in ordered_keys:
            idx = cursors[key]
            items = groups[key]
            while idx < len(items) and (
                int(items[idx].size_bytes) <= 0
                or used_bytes + int(items[idx].size_bytes) > budget_bytes
            ):
                # Allow 0-byte marker files if small, otherwise skip oversized items in pilot
                if int(items[idx].size_bytes) == 0 and used_bytes <= budget_bytes:
                    break
                idx += 1
            if idx < len(items) and used_bytes + int(items[idx].size_bytes) <= budget_bytes:
                chosen = items[idx]
                selected.append(chosen)
                used_bytes += int(chosen.size_bytes)
                cursors[key] = idx + 1
                progressed = True

    return selected


def select_pilot_slice(
    pilot_entries: Sequence[TransferEntry],
    slice_index: int = 0,
    num_slices: int = 1,
) -> List[TransferEntry]:
    """Deterministically split pilot entries into disjoint byte-balanced slices."""
    if num_slices < 1:
        raise ValueError("num_slices must be >= 1")
    if slice_index < 0 or slice_index >= num_slices:
        raise ValueError(f"slice_index ({slice_index}) must be in [0, {num_slices})")
    if num_slices == 1:
        return list(pilot_entries)
    slices = partition_entries_by_bytes(pilot_entries, num_shards=num_slices)
    return slices[slice_index] if slice_index < len(slices) else []


def project_full_transfer_with_overhead(
    total_bytes: int,
    measured_worker_throughput_mbps: float,
    num_containers: int,
    startup_overhead_s_per_container: float = 15.0,
    commit_overhead_s_per_container: float = 5.0,
    verify_s_per_container: float = 15.0,
    retry_overhead_factor: float = 1.05,
    cpu: float = DEFAULT_WORKER_CPU,
    memory_mib: int = DEFAULT_WORKER_MEMORY_MIB,
) -> Dict[str, float]:
    """Project full transfer time & USD including startup, retries, commits & verify."""
    if measured_worker_throughput_mbps <= 0 or num_containers <= 0:
        raise ValueError("throughput and num_containers must be > 0")
    total_mb = float(total_bytes) / (1024.0**2)
    pure_transfer_wall_s = total_mb / float(measured_worker_throughput_mbps)
    xfer_wall_with_retries_s = pure_transfer_wall_s * float(retry_overhead_factor)
    xfer_container_s = (
        xfer_wall_with_retries_s
        + float(startup_overhead_s_per_container)
        + float(commit_overhead_s_per_container)
    ) * float(num_containers)
    verify_container_s = (
        float(startup_overhead_s_per_container) + float(verify_s_per_container)
    ) * float(num_containers)
    total_container_hours = (xfer_container_s + verify_container_s) / 3600.0
    hourly_rate = estimate_worker_hourly_cost_usd(cpu=cpu, memory_mib=memory_mib)
    total_wall_s = (
        float(startup_overhead_s_per_container)
        + xfer_wall_with_retries_s
        + float(commit_overhead_s_per_container)
        + float(startup_overhead_s_per_container)
        + float(verify_s_per_container)
    )
    return {
        "total_gib": float(total_bytes) / (1024.0**3),
        "total_wall_seconds": total_wall_s,
        "total_wall_minutes": total_wall_s / 60.0,
        "xfer_container_hours": xfer_container_s / 3600.0,
        "verify_container_hours": verify_container_s / 3600.0,
        "total_container_hours": total_container_hours,
        "per_container_usd_hr": hourly_rate,
        "estimated_total_usd": total_container_hours * hourly_rate,
    }


def should_skip_existing_file(
    dst_file: Path,
    expected_size: int,
    verified_manifest_entry: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True if destination file already matches expected size and isn't corrupt/empty."""
    if not dst_file.exists() or not dst_file.is_file():
        return False
    actual_size = dst_file.stat().st_size
    if actual_size != int(expected_size):
        return False
    if verified_manifest_entry is not None:
        return (
            int(verified_manifest_entry.get("size_bytes", -1)) == int(expected_size)
            and bool(verified_manifest_entry.get("sha256"))
            and bool(verified_manifest_entry.get("verified", True))
        )
    return True


def estimate_worker_hourly_cost_usd(
    cpu: float = DEFAULT_WORKER_CPU,
    memory_mib: int = DEFAULT_WORKER_MEMORY_MIB,
) -> float:
    """Compute hourly USD cost for one CPU Modal container from official Modal rates."""
    return float(cpu) * MODAL_CPU_HOUR_USD + (float(memory_mib) / 1024.0) * MODAL_MEM_GIB_HOUR_USD


def estimate_transfer_compute_cost_usd(
    total_bytes: int,
    aggregate_mb_per_s: float,
    num_containers: int,
    cpu: float = DEFAULT_WORKER_CPU,
    memory_mib: int = DEFAULT_WORKER_MEMORY_MIB,
) -> Dict[str, float]:
    """Estimate wall-clock hours, container-hours, and USD compute cost."""
    if aggregate_mb_per_s <= 0 or num_containers <= 0:
        raise ValueError("aggregate_mb_per_s and num_containers must be > 0")
    total_mb = float(total_bytes) / (1024.0**2)
    wall_seconds = total_mb / float(aggregate_mb_per_s)
    wall_hours = wall_seconds / 3600.0
    container_hours = wall_hours * float(num_containers)
    hourly_per_container = estimate_worker_hourly_cost_usd(cpu=cpu, memory_mib=memory_mib)
    est_usd = container_hours * hourly_per_container
    return {
        "total_gib": float(total_bytes) / (1024.0**3),
        "wall_seconds": wall_seconds,
        "wall_minutes": wall_seconds / 60.0,
        "container_hours": container_hours,
        "per_container_usd_hr": hourly_per_container,
        "estimated_cost_usd": est_usd,
    }


def estimate_storage_cost_usd(total_used_gib: float) -> Dict[str, float]:
    """Compute monthly and daily Modal Volume storage fees vs the 1 TiB free tier."""
    overage_gib = max(0.0, float(total_used_gib) - MODAL_FREE_STORAGE_GIB)
    monthly_usd = overage_gib * MODAL_VOLUME_GIB_MONTH_USD
    daily_usd = monthly_usd / 30.0
    return {
        "total_used_gib": float(total_used_gib),
        "free_tier_gib": MODAL_FREE_STORAGE_GIB,
        "overage_gib": overage_gib,
        "headroom_gib": max(0.0, MODAL_FREE_STORAGE_GIB - float(total_used_gib)),
        "monthly_usd": monthly_usd,
        "daily_usd": daily_usd,
    }


def load_modal_profile_credentials(
    profile_names: Iterable[str],
    modal_toml_path: Optional[Path] = None,
) -> Dict[str, Tuple[str, str]]:
    """Read `(token_id, token_secret)` pairs locally from `~/.modal.toml`.

    Credentials are returned solely for injecting encrypted `modal.Secret` objects
    and must NEVER be printed, written to logs, or serialized into manifests.
    """
    toml_path = Path(modal_toml_path) if modal_toml_path else Path.home() / ".modal.toml"
    if not toml_path.exists():
        raise FileNotFoundError(f"Modal config not found at {toml_path}")

    with open(toml_path, "rb") as f:
        parsed = tomllib.load(f)

    creds: Dict[str, Tuple[str, str]] = {}
    for profile in sorted(set(profile_names)):
        section = parsed.get(profile)
        if not isinstance(section, dict):
            raise KeyError(f"Profile '{profile}' missing in {toml_path}")
        tid = str(section.get("token_id", "")).strip()
        tsec = str(section.get("token_secret", "")).strip()
        if not tid or not tsec:
            raise ValueError(f"Profile '{profile}' has empty token_id/token_secret")
        creds[profile] = (tid, tsec)
    return creds


def redact_secrets(text: str) -> str:
    """Scrub Modal tokens or secret env values from error messages/logs."""
    redacted = re.sub(r"\bak-[A-Za-z0-9_-]{8,}\b", "ak-[REDACTED]", str(text))
    redacted = re.sub(r"\bas-[A-Za-z0-9_-]{8,}\b", "as-[REDACTED]", redacted)
    return redacted


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write JSON via temp `.part` file + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def entries_to_dicts(entries: Sequence[TransferEntry]) -> List[Dict[str, Any]]:
    """Convert TransferEntry sequence to JSON-serializable dicts."""
    return [asdict(e) for e in entries]


async def commit_volume_with_retry(vol: Any, max_attempts: int = 6) -> Tuple[int, float]:
    """Commit Modal Volume with exponential backoff + jitter; returns `(attempts, seconds)`."""
    import asyncio
    import random
    import time

    t0 = time.monotonic()
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        try:
            await vol.commit.aio()
            return attempt, (time.monotonic() - t0)
        except Exception as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Volume commit failed after {max_attempts} attempts: {redact_secrets(str(exc))}"
                ) from exc
            await asyncio.sleep(delay + random.uniform(0.2, 1.2))
            delay = min(delay * 2.0, 30.0)
    return max_attempts, (time.monotonic() - t0)


async def async_transfer_shard(
    job_id: str,
    shard_id: int,
    entries: List[Dict[str, Any]],
    dst_vol: Any,
    mount_root: Path,
    streams_per_container: int,
    commit_interval_s: float,
    client_factory: Optional[Any] = None,
    volume_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Stream a shard of files directly from source Modal Volume(s) to destination mount."""
    import asyncio
    import hashlib
    import signal
    import time
    try:
        import resource as _resource
    except ImportError:
        _resource = None  # type: ignore[assignment]

    if client_factory is None or volume_factory is None:
        import modal as _modal

        if client_factory is None:
            client_factory = _modal.Client.from_credentials.aio
        if volume_factory is None:
            volume_factory = _modal.Volume.from_name

    start_ts = time.monotonic()
    stop_requested = False

    def _handle_sig(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    src_clients: Dict[str, Any] = {}
    src_volumes: Dict[Tuple[str, str], Any] = {}

    async def _get_src_vol(src_profile: str, src_volume: str) -> Any:
        key = (src_profile, src_volume)
        if key not in src_volumes:
            safe = src_profile.upper()
            tid = os.environ.get(f"SRC_TOKEN_ID_{safe}", "").strip()
            tsec = os.environ.get(f"SRC_TOKEN_SECRET_{safe}", "").strip()
            if not tid or not tsec:
                raise RuntimeError(
                    f"Missing SRC credentials in Modal Secret for profile '{src_profile}'"
                )
            if src_profile not in src_clients:
                src_clients[src_profile] = await client_factory(tid, tsec)
            src_volumes[key] = volume_factory(
                src_volume,
                client=src_clients[src_profile],
            )
        return src_volumes[key]

    manifest_path = mount_root / "_transfer_manifests" / job_id / f"shard_{shard_id:04d}.json"
    existing_verified: Dict[str, Dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in prev.get("files", []):
                if item.get("dst_path"):
                    existing_verified[item["dst_path"]] = item
        except Exception:
            existing_verified = {}

    stream_state = {
        "active": 0,
        "limit": max(1, int(streams_per_container)),
    }
    fallback_streams = 2 if stream_state["limit"] > 2 else stream_state["limit"]
    gate_cond = asyncio.Condition()
    lock = asyncio.Lock()
    stats: Dict[str, Any] = {
        "transferred_files": 0,
        "skipped_files": 0,
        "failed_files": 0,
        "transferred_bytes": 0,
        "skipped_bytes": 0,
        "file_retries": 0,
        "commit_calls": 0,
        "commit_retries": 0,
        "commit_duration_s": 0.0,
        "adaptive_fallback_triggered": False,
        "final_streams_per_container": stream_state["limit"],
    }
    recorded_files: List[Dict[str, Any]] = []
    errors: List[str] = []

    async def _transfer_one(raw_item: Dict[str, Any]) -> None:
        if stop_requested:
            return
        src_profile = str(raw_item["src_profile"])
        src_volume = str(raw_item["src_volume"])
        src_path = assert_transfer_path_allowed(str(raw_item["src_path"]), context="worker_src")
        dst_path = assert_transfer_path_allowed(str(raw_item["dst_path"]), context="worker_dst")
        expected_size = int(raw_item["size_bytes"])
        dst_file = mount_root / dst_path

        prev_entry = existing_verified.get(dst_path)
        if should_skip_existing_file(dst_file, expected_size, prev_entry):
            async with lock:
                stats["skipped_files"] += 1
                stats["skipped_bytes"] += expected_size
                recorded_files.append(
                    {
                        "dst_path": dst_path,
                        "src_profile": src_profile,
                        "src_volume": src_volume,
                        "size_bytes": expected_size,
                        "sha256": prev_entry["sha256"] if prev_entry else None,
                        "status": "skipped_existing",
                    }
                )
            return

        src_vol = await _get_src_vol(src_profile, src_volume)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        part_file = dst_file.with_name(f".{dst_file.name}.{shard_id}.part")

        async with gate_cond:
            while stream_state["active"] >= stream_state["limit"] and not stop_requested:
                await gate_cond.wait()
            if stop_requested:
                return
            stream_state["active"] += 1
        try:
            delay = 1.0
            last_err = ""
            for attempt in range(1, 7):
                if stop_requested:
                    return
                try:
                    hasher = hashlib.sha256()
                    written = 0
                    with open(part_file, "wb") as out_f:
                        async for chunk in src_vol.read_file.aio(src_path):
                            out_f.write(chunk)
                            hasher.update(chunk)
                            written += len(chunk)
                    if written != expected_size:
                        raise IOError(
                            f"Size mismatch for {dst_path}: expected {expected_size}, got {written}"
                        )
                    digest = hasher.hexdigest()
                    expected_digest = EXPECTED_STAGE1_WEIGHT_SHA256.get(dst_path)
                    if expected_digest and digest.lower() != expected_digest.lower():
                        raise ValueError(
                            f"Weight SHA-256 mismatch for {dst_path}: got {digest}, expected {expected_digest}"
                        )
                    os.replace(part_file, dst_file)
                    async with lock:
                        stats["transferred_files"] += 1
                        stats["transferred_bytes"] += written
                        recorded_files.append(
                            {
                                "dst_path": dst_path,
                                "src_profile": src_profile,
                                "src_volume": src_volume,
                                "size_bytes": written,
                                "sha256": digest,
                                "status": "transferred",
                            }
                        )
                    return
                except Exception as exc:
                    last_err = redact_secrets(str(exc))
                    if part_file.exists():
                        try:
                            part_file.unlink()
                        except OSError:
                            pass
                    if attempt < 6:
                        async with lock:
                            stats["file_retries"] += 1
                            if (
                                not stats["adaptive_fallback_triggered"]
                                and stream_state["limit"] > fallback_streams
                                and stats["file_retries"] >= 5
                            ):
                                stats["adaptive_fallback_triggered"] = True
                                stats["final_streams_per_container"] = fallback_streams
                                stream_state["limit"] = fallback_streams
                        await asyncio.sleep(delay)
                        delay = min(delay * 2.0, 16.0)

            async with lock:
                stats["failed_files"] += 1
                errors.append(f"{dst_path}: {last_err}")
        finally:
            async with gate_cond:
                stream_state["active"] -= 1
                gate_cond.notify_all()

    commit_stop = asyncio.Event()

    async def _periodic_commit_loop() -> None:
        while not commit_stop.is_set() and not stop_requested:
            try:
                await asyncio.wait_for(commit_stop.wait(), timeout=commit_interval_s)
            except asyncio.TimeoutError:
                attempts, c_s = await commit_volume_with_retry(dst_vol)
                async with lock:
                    stats["commit_calls"] += 1
                    stats["commit_retries"] += max(0, attempts - 1)
                    stats["commit_duration_s"] += c_s
                    if (
                        not stats["adaptive_fallback_triggered"]
                        and stream_state["limit"] > fallback_streams
                        and stats["commit_retries"] >= 2
                    ):
                        stats["adaptive_fallback_triggered"] = True
                        stats["final_streams_per_container"] = fallback_streams
                        stream_state["limit"] = fallback_streams

    commit_task = asyncio.create_task(_periodic_commit_loop())
    try:
        tasks = [asyncio.create_task(_transfer_one(item)) for item in entries]
        await asyncio.gather(*tasks)
    finally:
        commit_stop.set()
        await commit_task

    elapsed_s = max(0.001, time.monotonic() - start_ts)
    if _resource is not None:
        ru = _resource.getrusage(_resource.RUSAGE_SELF)
        stats["peak_rss_mib"] = round(float(ru.ru_maxrss) / 1024.0, 2)
        stats["cpu_seconds"] = round(float(ru.ru_utime + ru.ru_stime), 2)
    shard_report = {
        "job_id": job_id,
        "shard_id": shard_id,
        "elapsed_s": elapsed_s,
        "streams_per_container": streams_per_container,
        "interrupted": stop_requested,
        "stats": stats,
        "errors": errors,
        "files": recorded_files,
    }
    atomic_write_json(manifest_path, shard_report)
    final_attempts, final_c_s = await commit_volume_with_retry(dst_vol)
    stats["commit_calls"] += 1
    stats["commit_retries"] += max(0, final_attempts - 1)
    stats["commit_duration_s"] = round(stats["commit_duration_s"] + final_c_s, 3)

    return {
        "job_id": job_id,
        "shard_id": shard_id,
        "elapsed_s": elapsed_s,
        "interrupted": stop_requested,
        "stats": stats,
        "errors": errors[:20],
        "manifest_rel_path": str(manifest_path.relative_to(mount_root)),
    }
