"""CPU Stage-1 Deterministic Cropped 16-Bit Image Cache.

Stores lossless uint16 cropped mammograms produced by TRUST Section 3.2 Stage 1
(Modality/VOI LUT -> MONOCHROME1 invert -> explicit right-breast mirror ->
border-median breast mask -> hole fill -> 9x9 closing -> zero background ->
10% bbox crop -> uint16 quantization) without resizing to 1520x912 or applying
stochastic augmentation.

Raw DICOMs are decoded at most once and never modified.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import (
    Stage1CropResult,
    decode_dicom_array,
    extract_cropped_mammogram_with_metadata,
    get_stage1_config_hash,
)

STAGE1_CACHE_SCHEMA_VERSION = "anchorshift_stage1_uint16_v1"


class Stage1CacheError(RuntimeError):
    """Raised when Stage-1 cache metadata, shards, or completeness checks fail."""


def _compute_file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_uint16_image_sha256(arr: np.ndarray) -> str:
    """Compute SHA-256 of a uint16 image's shape and raw bytes."""
    if arr.dtype != np.uint16:
        raise ValueError(f"Expected uint16 array, got {arr.dtype}")
    h = hashlib.sha256()
    h.update(f"{arr.shape[0]}x{arr.shape[1]}:".encode("utf-8"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _atomic_write_torch(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".pt.tmp")
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    sha256 = _compute_file_sha256(tmp_path)
    os.replace(tmp_path, path)
    return sha256


def _cleanup_stale_temps(cache_root: Path) -> None:
    if not cache_root.exists():
        return
    for p in cache_root.rglob("*.tmp"):
        p.unlink(missing_ok=True)


def stage1_shard_filename(shard_idx: int) -> str:
    return f"stage1_shard_{shard_idx:04d}.pt"


def _summarize_entries(entries: List[Dict[str, Any]], invalid_count: int = 0) -> Dict[str, int]:
    return {
        "total_images": len(entries),
        "successful_segmentation_count": sum(1 for e in entries if e["segmentation_status"] == "success"),
        "empty_mask_count": sum(1 for e in entries if e["segmentation_status"] == "empty_mask"),
        "all_foreground_count": sum(1 for e in entries if e["segmentation_status"] == "all_foreground"),
        "degenerate_small_count": sum(1 for e in entries if e["segmentation_status"] == "degenerate_small"),
        "fallback_count": sum(1 for e in entries if e["used_fallback_uncropped"]),
        "invalid_image_count": int(invalid_count),
        "missing_laterality_count": sum(1 for e in entries if e["laterality"] not in ("L", "R")),
    }


def build_or_resume_stage1_cache(
    cache_root: Path,
    manifest_df: pd.DataFrame,
    raw_root: Path,
    split_manifest_sha256: str,
    images_per_shard: int = 250,
    laterality_policy: str = "strict",
    strict_segmentation: bool = True,
    use_largest_cc: bool = False,
    crop_margin_frac: float = 0.10,
    num_workers: int = 1,
    process_row_fn: Optional[Callable[[pd.Series], Stage1CropResult]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Process each raw DICOM once into sharded lossless uint16 crops with atomic resume."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_temps(cache_root)

    expected_stage1_hash = get_stage1_config_hash(
        laterality_policy=laterality_policy,
        use_largest_cc=use_largest_cc,
        crop_margin_frac=crop_margin_frac,
    )

    df = manifest_df.reset_index(drop=True)
    manifest_path = cache_root / "stage1_manifest.json"
    valid_shards_by_idx: Dict[int, Dict[str, Any]] = {}
    valid_entries_by_shard: Dict[str, List[Dict[str, Any]]] = {}

    if manifest_path.exists():
        try:
            old_m = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                old_m.get("schema_version") == STAGE1_CACHE_SCHEMA_VERSION
                and old_m.get("stage1_config_hash") == expected_stage1_hash
                and old_m.get("split_manifest_sha256") == split_manifest_sha256
                and int(old_m.get("images_per_shard", -1)) == int(images_per_shard)
            ):
                for s_meta in old_m.get("shards", []):
                    s_path = cache_root / s_meta["file"]
                    if s_path.exists() and _compute_file_sha256(s_path) == s_meta["sha256"]:
                        valid_shards_by_idx[int(s_meta["shard_idx"])] = s_meta
                for entry in old_m.get("images", []):
                    valid_entries_by_shard.setdefault(entry["shard_file"], []).append(entry)
        except Exception:
            valid_shards_by_idx = {}
            valid_entries_by_shard = {}

    def _default_process(row: pd.Series) -> Stage1CropResult:
        dcm_path = raw_root / str(row["rel_path"])
        arr, phot, dcm_lat = decode_dicom_array(dcm_path, apply_luts=True)
        row_lat = str(row["laterality"]).strip().upper() if "laterality" in row and pd.notna(row["laterality"]) else ""
        if row_lat in ("L", "R") and dcm_lat in ("L", "R") and row_lat != dcm_lat and laterality_policy == "strict":
            raise ValueError(
                f"Conflicting laterality between manifest ({row_lat!r}) and DICOM ({dcm_lat!r}) for {dcm_path}"
            )
        eff_lat = row_lat if row_lat in ("L", "R") else dcm_lat
        return extract_cropped_mammogram_with_metadata(
            arr=arr,
            photometric=phot,
            laterality=eff_lat,
            invert_monochrome1=True,
            mirror_right=True,
            laterality_policy=laterality_policy,
            crop_breast=True,
            crop_margin_frac=crop_margin_frac,
            use_largest_cc=use_largest_cc,
            strict_segmentation=strict_segmentation,
            simulate_16bit_png=True,
            context_label=str(dcm_path),
        )

    proc_fn = process_row_fn if process_row_fn is not None else _default_process

    shards_meta: List[Dict[str, Any]] = []
    image_entries: List[Dict[str, Any]] = []
    seen_ids = set()
    total_images = len(df)
    total_shards = max((total_images + images_per_shard - 1) // images_per_shard, 1)
    t_build_start = time.perf_counter()
    newly_processed = 0

    def _format_bar(done_n: int, elapsed_s: float) -> str:
        pct = (100.0 * done_n) / max(total_images, 1)
        filled = int(round(20 * done_n / max(total_images, 1)))
        bar = "#" * filled + "." * (20 - filled)
        rate = newly_processed / max(elapsed_s, 1e-6)
        eta_hr = ((total_images - done_n) / max(rate, 1e-6)) / 3600.0 if rate > 0 else 0.0
        return f"[{bar}] {done_n}/{total_images} ({pct:5.1f}%) | {rate:4.2f} img/s | ETA {eta_hr:4.2f}h"

    for shard_idx, start in enumerate(range(0, total_images, images_per_shard)):
        end = min(start + images_per_shard, total_images)
        shard_df = df.iloc[start:end]
        fname = stage1_shard_filename(shard_idx)

        if shard_idx in valid_shards_by_idx and fname in valid_entries_by_shard:
            s_entries = valid_entries_by_shard[fname]
            if [e["image_id"] for e in s_entries] == [str(x) for x in shard_df["image_id"].tolist()]:
                shards_meta.append(valid_shards_by_idx[shard_idx])
                image_entries.extend(s_entries)
                seen_ids.update(e["image_id"] for e in s_entries)
                if on_progress:
                    on_progress(f"Resumed shard {shard_idx + 1}/{total_shards} | {end}/{total_images} cached")
                continue

        rows_list = [row for _, row in shard_df.iterrows()]
        for row in rows_list:
            img_id = str(row["image_id"])
            if img_id in seen_ids:
                raise Stage1CacheError(f"Duplicate image_id in input manifest: {img_id}")
            seen_ids.add(img_id)

        def _process_one(row: pd.Series) -> Tuple[pd.Series, Stage1CropResult, str]:
            res = proc_fn(row)
            img_sha = compute_uint16_image_sha256(res.cropped_uint16)
            return row, res, img_sha

        shard_arrays: Dict[str, np.ndarray] = {}
        new_entries: List[Dict[str, Any]] = []
        eff_workers = max(int(num_workers), 1)
        if eff_workers > 1 and len(rows_list) > 1:
            with ThreadPoolExecutor(max_workers=min(eff_workers, len(rows_list))) as pool:
                processed_items = list(pool.map(_process_one, rows_list))
        else:
            processed_items = [_process_one(r) for r in rows_list]

        for row, res, img_sha in processed_items:
            img_id = str(row["image_id"])
            shard_arrays[img_id] = np.ascontiguousarray(res.cropped_uint16)
            new_entries.append(
                {
                    "image_id": img_id,
                    "patient_id": str(row["patient_id"]),
                    "split": str(row["split"]) if "split" in row else "unknown",
                    "rel_path": str(row["rel_path"]),
                    "laterality": res.laterality,
                    "laterality_status": res.laterality_status,
                    "photometric": res.photometric,
                    "stage1_config_hash": expected_stage1_hash,
                    "orig_shape": list(res.orig_shape),
                    "crop_shape": list(res.crop_shape),
                    "crop_bbox": list(res.crop_bbox),
                    "border_median": round(res.border_median, 6),
                    "segmentation_status": res.segmentation_status,
                    "used_fallback_uncropped": bool(res.used_fallback_uncropped),
                    "mirrored": bool(res.mirrored),
                    "dtype": "uint16",
                    "shard_file": fname,
                    "image_sha256": img_sha,
                }
            )
        newly_processed += len(processed_items)

        shard_path = cache_root / fname
        shard_sha = _atomic_write_torch(shard_path, {"images": shard_arrays})
        shards_meta.append(
            {
                "shard_idx": int(shard_idx),
                "file": fname,
                "sha256": shard_sha,
                "num_images": len(new_entries),
                "size_bytes": int(shard_path.stat().st_size),
            }
        )
        image_entries.extend(new_entries)

        partial_manifest = {
            "schema_version": STAGE1_CACHE_SCHEMA_VERSION,
            "stage1_config_hash": expected_stage1_hash,
            "split_manifest_sha256": str(split_manifest_sha256),
            "images_per_shard": int(images_per_shard),
            "laterality_policy": laterality_policy,
            "use_largest_cc": bool(use_largest_cc),
            "crop_margin_frac": float(crop_margin_frac),
            "summary": _summarize_entries(image_entries),
            "shards": shards_meta,
            "images": image_entries,
        }
        _atomic_write_json(manifest_path, partial_manifest)
        if on_progress:
            elapsed = time.perf_counter() - t_build_start
            on_progress(f"Shard {shard_idx + 1}/{total_shards} | {_format_bar(end, elapsed)}")

    final_manifest = {
        "schema_version": STAGE1_CACHE_SCHEMA_VERSION,
        "stage1_config_hash": expected_stage1_hash,
        "split_manifest_sha256": str(split_manifest_sha256),
        "images_per_shard": int(images_per_shard),
        "laterality_policy": laterality_policy,
        "use_largest_cc": bool(use_largest_cc),
        "crop_margin_frac": float(crop_margin_frac),
        "summary": _summarize_entries(image_entries),
        "shards": shards_meta,
        "images": image_entries,
    }
    _atomic_write_json(manifest_path, final_manifest)
    return final_manifest


def verify_stage1_cache(
    cache_root: Path,
    expected_df: pd.DataFrame,
    expected_stage1_hash: str,
    expected_split_sha256: Optional[str] = None,
    allow_fallbacks: bool = False,
    verify_pixel_hashes: bool = True,
) -> Dict[str, Any]:
    """Verify Stage-1 cache completeness, checksums, uint16 dtype, and zero missing/duplicates."""
    cache_root = Path(cache_root)
    manifest_path = cache_root / "stage1_manifest.json"
    if not manifest_path.exists():
        raise Stage1CacheError(f"Missing Stage-1 cache manifest at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != STAGE1_CACHE_SCHEMA_VERSION:
        raise Stage1CacheError(
            f"Stage-1 schema mismatch: {manifest.get('schema_version')} != {STAGE1_CACHE_SCHEMA_VERSION}"
        )
    if manifest.get("stage1_config_hash") != expected_stage1_hash:
        raise Stage1CacheError(
            f"Stage-1 config hash mismatch: {manifest.get('stage1_config_hash')} != {expected_stage1_hash}"
        )
    if expected_split_sha256 is not None and manifest.get("split_manifest_sha256") != expected_split_sha256:
        raise Stage1CacheError(
            f"Stage-1 split hash mismatch: {manifest.get('split_manifest_sha256')} != {expected_split_sha256}"
        )

    expected_ids = [str(x) for x in expected_df["image_id"].tolist()]
    if len(expected_ids) != len(set(expected_ids)):
        raise Stage1CacheError("Input expected_df contains duplicate image_ids")

    storage_format = str(manifest.get("storage_format", "pt_shards"))
    shard_map: Dict[str, Dict[str, Any]] = {}
    if storage_format != "png16_individual":
        for smeta in manifest.get("shards", []):
            spath = cache_root / smeta["file"]
            if not spath.exists():
                raise Stage1CacheError(f"Missing Stage-1 shard file: {spath}")
            actual_sha = _compute_file_sha256(spath)
            if actual_sha != smeta["sha256"]:
                raise Stage1CacheError(
                    f"Stage-1 shard checksum mismatch for {spath}: {actual_sha} != {smeta['sha256']}"
                )
            shard_map[smeta["file"]] = smeta

    seen_ids: set = set()
    entries_by_shard: Dict[str, List[Dict[str, Any]]] = {}
    png_entries: List[Dict[str, Any]] = []
    for entry in manifest.get("images", []):
        img_id = str(entry["image_id"])
        if img_id in seen_ids:
            raise Stage1CacheError(f"Duplicate image_id in Stage-1 manifest: {img_id}")
        seen_ids.add(img_id)
        if entry.get("stage1_config_hash") != expected_stage1_hash:
            raise Stage1CacheError(f"Entry {img_id} stage1_config_hash mismatch")
        if entry.get("dtype") != "uint16":
            raise Stage1CacheError(f"Entry {img_id} expected uint16, got {entry.get('dtype')}")
        if not allow_fallbacks and entry.get("used_fallback_uncropped"):
            raise Stage1CacheError(
                f"Image {img_id} used uncropped fallback ({entry.get('segmentation_status')}) with allow_fallbacks=False"
            )
        if storage_format == "png16_individual":
            rel_png = entry.get("png_rel_path")
            if not rel_png or not (cache_root / str(rel_png)).exists():
                raise Stage1CacheError(f"Missing Stage-1 PNG for {img_id}: {rel_png}")
            png_entries.append(entry)
        else:
            if entry.get("shard_file") not in shard_map:
                raise Stage1CacheError(f"Entry {img_id} references unknown shard {entry.get('shard_file')}")
            entries_by_shard.setdefault(entry["shard_file"], []).append(entry)

    missing_ids = set(expected_ids) - seen_ids
    extra_ids = seen_ids - set(expected_ids)
    if missing_ids:
        raise Stage1CacheError(f"Missing {len(missing_ids)} expected images in Stage-1 cache (sample: {sorted(missing_ids)[:5]})")
    if extra_ids:
        raise Stage1CacheError(f"Found {len(extra_ids)} unexpected extra images in Stage-1 cache")

    if verify_pixel_hashes:
        if storage_format == "png16_individual":
            for entry in png_entries:
                img_id = str(entry["image_id"])
                ppath = cache_root / str(entry["png_rel_path"])
                arr = cv2.imread(str(ppath), cv2.IMREAD_UNCHANGED)
                if arr is None or not isinstance(arr, np.ndarray) or arr.dtype != np.uint16:
                    raise Stage1CacheError(f"Image {img_id} at {ppath} is not a valid uint16 PNG")
                if compute_uint16_image_sha256(arr) != entry["image_sha256"]:
                    raise Stage1CacheError(f"Pixel SHA-256 mismatch for PNG {img_id} at {ppath}")
        else:
            for fname, entries in entries_by_shard.items():
                shard_data = torch.load(str(cache_root / fname), map_location="cpu", weights_only=False)["images"]
                for entry in entries:
                    img_id = str(entry["image_id"])
                    if img_id not in shard_data:
                        raise Stage1CacheError(f"Image {img_id} missing inside shard {fname}")
                    arr = shard_data[img_id]
                    if not isinstance(arr, np.ndarray) or arr.dtype != np.uint16:
                        raise Stage1CacheError(f"Image {img_id} inside {fname} is not uint16 ndarray")
                    if compute_uint16_image_sha256(arr) != entry["image_sha256"]:
                        raise Stage1CacheError(f"Pixel SHA-256 mismatch for image {img_id} in {fname}")

    summary = _summarize_entries(manifest.get("images", []))
    if summary["missing_laterality_count"] > 0:
        raise Stage1CacheError(f"Stage-1 cache has {summary['missing_laterality_count']} images with missing laterality")
    return manifest


class Stage1CroppedLoader:
    """Thread-safe loader supporting individual lossless 16-bit PNGs or legacy sharded uint16 tensors."""

    def __init__(self, cache_root: Path, manifest: Dict[str, Any], max_cached_shards: int = 8) -> None:
        self.cache_root = Path(cache_root)
        self.manifest = manifest
        self.max_cached_shards = max(int(max_cached_shards), 1)
        self.id_to_png: Dict[str, str] = {
            str(e["image_id"]): str(e["png_rel_path"])
            for e in manifest["images"]
            if e.get("png_rel_path")
        }
        self.id_to_shard: Dict[str, str] = {
            str(e["image_id"]): str(e["shard_file"])
            for e in manifest["images"]
            if e.get("shard_file")
        }
        self._shard_lru: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def load_uint16(self, image_id: str) -> np.ndarray:
        key = str(image_id)
        if key in self.id_to_png:
            ppath = self.cache_root / self.id_to_png[key]
            arr = cv2.imread(str(ppath), cv2.IMREAD_UNCHANGED)
            if arr is None or arr.dtype != np.uint16:
                raise Stage1CacheError(f"Failed to read uint16 PNG for {key} at {ppath}")
            return arr
        if key not in self.id_to_shard:
            raise KeyError(f"Image {key!r} not found in Stage-1 manifest")
        fname = self.id_to_shard[key]
        if fname in self._shard_lru:
            self._shard_lru.move_to_end(fname)
            return self._shard_lru[fname][key]

        shard_data = torch.load(str(self.cache_root / fname), map_location="cpu", weights_only=False)["images"]
        self._shard_lru[fname] = shard_data
        self._shard_lru.move_to_end(fname)
        while len(self._shard_lru) > self.max_cached_shards:
            self._shard_lru.popitem(last=False)
        return shard_data[key]
