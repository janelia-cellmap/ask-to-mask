"""Zarr reading utilities for CellMap data, adapted from sam3m."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import zarr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zarr metadata helpers
# ---------------------------------------------------------------------------


def get_scale_info(zarr_grp_path: str):
    """Read multiscale metadata from a zarr group.

    Returns:
        offsets: dict mapping scale path -> [z, y, x] translation (world nm)
        resolutions: dict mapping scale path -> [z, y, x] voxel size (nm)
        shapes: dict mapping scale path -> volume shape (voxels)
    """
    grp = zarr.open(zarr_grp_path, mode="r")
    attrs = grp.attrs
    offsets, resolutions, shapes = {}, {}, {}
    for scale in attrs["multiscales"][0]["datasets"]:
        path = scale["path"]
        resolutions[path] = scale["coordinateTransformations"][0]["scale"]
        offsets[path] = scale["coordinateTransformations"][1]["translation"]
        shapes[path] = grp[path].shape
    return offsets, resolutions, shapes


def get_raw_path(em_base: str) -> str | None:
    """Select fibsem-uint8 if available, else fibsem-uint16."""
    uint8_path = os.path.join(em_base, "fibsem-uint8")
    uint16_path = os.path.join(em_base, "fibsem-uint16")
    if os.path.isdir(uint8_path):
        return uint8_path
    if os.path.isdir(uint16_path):
        return uint16_path
    return None


def find_scale_for_resolution(
    zarr_grp_path: str,
    target_res: float,
    max_ratio: float = 2.0,
) -> tuple[str, list[float], list[float], tuple[int, ...]] | None:
    """Find the scale level closest to target_res (matching on Y axis).

    Returns (scale_path, resolution, offset, shape) or None.
    """
    offsets, resolutions, shapes = get_scale_info(zarr_grp_path)
    candidates = []
    for name, res in resolutions.items():
        y_res = res[1]
        ratio = max(y_res / target_res, target_res / y_res)
        if ratio <= max_ratio:
            candidates.append(
                (abs(y_res - target_res), name, res, offsets[name], shapes[name])
            )
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, best_path, best_res, best_off, best_shape = candidates[0]
    return best_path, best_res, best_off, best_shape


def find_finest_scale(
    zarr_grp_path: str,
) -> tuple[str, list[float], list[float], tuple[int, ...]] | None:
    """Find the finest (highest resolution / smallest voxel size) scale level.

    Returns (scale_path, resolution, offset, shape) or None.
    """
    try:
        offsets, resolutions, shapes = get_scale_info(zarr_grp_path)
    except Exception:
        return None
    if not resolutions:
        return None
    # Pick scale with smallest Y resolution
    best = min(resolutions.items(), key=lambda item: item[1][1])
    name = best[0]
    return name, resolutions[name], offsets[name], shapes[name]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@dataclass
class NormParams:
    min_val: float
    max_val: float
    inverted: bool


def load_norms(csv_path: str) -> dict[str, NormParams]:
    """Load per-dataset normalization parameters from CSV."""
    norms = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["dataset"].strip()
            norms[name] = NormParams(
                min_val=float(row["min"]),
                max_val=float(row["max"]),
                inverted=row["inverted"].strip() == "True",
            )
    return norms


def normalize_raw(raw: np.ndarray, norm: NormParams) -> np.ndarray:
    """Normalize raw EM to [0, 1] using per-dataset params."""
    raw = raw.astype(np.float32)
    denom = norm.max_val - norm.min_val
    if denom == 0:
        denom = 1.0
    raw = (raw - norm.min_val) / denom
    np.clip(raw, 0.0, 1.0, out=raw)
    return raw


# ---------------------------------------------------------------------------
# Crop metadata
# ---------------------------------------------------------------------------


@dataclass
class ClassInfo:
    """Metadata for one class label within a crop."""

    zarr_path: str  # path to the class zarr group
    scale_path: str  # e.g., "s1"
    resolution: list[float]  # [z, y, x] in nm
    offset_world: list[float]  # [z, y, x] in nm
    shape: tuple[int, ...]  # voxels at this scale


@dataclass
class CropInfo:
    """All pre-computed metadata for one annotated crop."""

    dataset_name: str
    crop_id: str
    # Raw EM
    raw_zarr_path: str
    raw_scale_path: str
    raw_resolution: list[float]
    raw_offset_world: list[float]
    raw_shape: tuple[int, ...]
    # Normalization
    norm_params: NormParams
    # Per-class label info (only for classes present in this crop)
    class_info: dict[str, ClassInfo] = field(default_factory=dict)
    # Which of the target classes are annotated in this crop
    annotated_classes: set = field(default_factory=set)
    # Crop bounding box in world coordinates [z, y, x]
    crop_origin_world: list[float] | None = None
    crop_extent_world: list[float] | None = None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_crops(
    data_root: str,
    target_classes: list[str],
    norms: dict[str, NormParams] | None = None,
    skip_datasets: list[str] | None = None,
    include_datasets: list[str] | None = None,
    min_crop_voxels: int = 16,
    cache_dir: str | None = None,
) -> list[CropInfo]:
    """Walk data_root and build CropInfo for every valid crop.

    Uses the finest available scale for each dataset's raw and label data.

    Args:
        data_root: Root directory (e.g., /nrs/cellmap/data).
        target_classes: Fine-grained class names to look for in zarr.
        norms: Per-dataset normalization parameters.
        skip_datasets: Datasets to skip.
        include_datasets: If set, only include these datasets.
        min_crop_voxels: Skip crops smaller than this in any dimension.
        cache_dir: Directory for caching discovery results.

    Returns:
        List of CropInfo with metadata for each valid crop.
    """
    if norms is None:
        norms = {}
    skip = set(skip_datasets or [])
    include = set(include_datasets) if include_datasets else None

    # Check cache
    if cache_dir is not None:
        cache_key_data = json.dumps(
            {
                "data_root": data_root,
                "target_classes": sorted(target_classes),
                "skip_datasets": sorted(skip),
                "include_datasets": sorted(include) if include else None,
                "min_crop_voxels": min_crop_voxels,
            },
            sort_keys=True,
        )
        cache_hash = hashlib.md5(cache_key_data.encode()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"crops_cache_{cache_hash}.pkl")

        if os.path.exists(cache_path):
            logger.info(f"Loading crop cache from {cache_path}")
            with open(cache_path, "rb") as f:
                crops = pickle.load(f)
            # Re-apply current norms
            for crop in crops:
                crop.norm_params = norms.get(
                    crop.dataset_name,
                    NormParams(min_val=0.0, max_val=255.0, inverted=False),
                )
            return crops

    if not os.path.isdir(data_root):
        logger.warning(f"data_root {data_root} does not exist")
        return []

    crops: list[CropInfo] = []
    datasets = sorted(os.listdir(data_root))

    for dataset_name in datasets:
        if include is not None and dataset_name not in include:
            continue
        if dataset_name in skip:
            continue

        dataset_dir = os.path.join(data_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue

        gt_base = os.path.join(
            dataset_dir, f"{dataset_name}.zarr", "recon-1", "labels", "groundtruth"
        )
        if not os.path.isdir(gt_base):
            continue

        em_base = os.path.join(
            dataset_dir, f"{dataset_name}.zarr", "recon-1", "em"
        )
        raw_path = get_raw_path(em_base)
        if raw_path is None:
            continue

        raw_scale_info = find_finest_scale(raw_path)
        if raw_scale_info is None:
            continue
        raw_scale_path, raw_res, raw_off, raw_shape = raw_scale_info

        norm = norms.get(
            dataset_name, NormParams(min_val=0.0, max_val=255.0, inverted=False)
        )

        for crop_name in sorted(os.listdir(gt_base)):
            if not crop_name.startswith("crop"):
                continue
            crop_dir = os.path.join(gt_base, crop_name)
            if not os.path.isdir(crop_dir):
                continue

            crop = CropInfo(
                dataset_name=dataset_name,
                crop_id=crop_name,
                raw_zarr_path=raw_path,
                raw_scale_path=raw_scale_path,
                raw_resolution=raw_res,
                raw_offset_world=raw_off,
                raw_shape=raw_shape,
                norm_params=norm,
            )

            crop_subdirs = set(os.listdir(crop_dir))
            ref_class_info = None

            for cls_name in target_classes:
                if cls_name not in crop_subdirs:
                    continue
                cls_path = os.path.join(crop_dir, cls_name)
                if not os.path.isdir(cls_path):
                    continue

                info = find_finest_scale(cls_path)
                if info is None:
                    continue

                scale_path, res, off, shape = info
                ci = ClassInfo(
                    zarr_path=cls_path,
                    scale_path=scale_path,
                    resolution=res,
                    offset_world=off,
                    shape=shape,
                )
                crop.class_info[cls_name] = ci
                crop.annotated_classes.add(cls_name)
                if ref_class_info is None:
                    ref_class_info = ci

            if ref_class_info is None:
                continue

            # Compute crop bounding box
            ref_off = np.array(ref_class_info.offset_world)
            ref_res = np.array(ref_class_info.resolution)
            ref_shape = np.array(ref_class_info.shape)
            crop.crop_origin_world = ref_off.tolist()
            crop.crop_extent_world = (ref_shape * ref_res).tolist()

            # Skip very small crops
            crop_voxels = np.array(crop.crop_extent_world) / np.array(raw_res)
            if np.any(crop_voxels < min_crop_voxels):
                continue

            crops.append(crop)

    logger.info(f"Discovered {len(crops)} crops")

    # Save cache
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(crops, f)
        logger.info(f"Saved crop cache to {cache_path}")

    return crops


def compute_auto_norms(
    crops: list[CropInfo],
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    num_slices: int = 5,
) -> dict[str, NormParams]:
    """Compute percentile-based normalization params per dataset.

    Samples up to num_slices middle Z-slices across crops per dataset,
    then uses percentile clipping to determine min/max.
    """
    import zarr as zarr_mod

    by_dataset: dict[str, list[CropInfo]] = {}
    for c in crops:
        by_dataset.setdefault(c.dataset_name, []).append(c)

    auto_norms: dict[str, NormParams] = {}
    for dataset_name in sorted(by_dataset):
        ds_crops = by_dataset[dataset_name]
        indices = np.linspace(
            0, len(ds_crops) - 1, min(num_slices, len(ds_crops)), dtype=int
        )
        all_vals = []
        for i in indices:
            crop = ds_crops[i]
            try:
                raw_arr = zarr_mod.open(
                    os.path.join(crop.raw_zarr_path, crop.raw_scale_path), mode="r"
                )
            except Exception:
                continue

            raw_off = np.array(crop.raw_offset_world)
            raw_res = np.array(crop.raw_resolution)
            raw_shape = np.array(crop.raw_shape)
            crop_origin = np.array(crop.crop_origin_world)
            crop_extent = np.array(crop.crop_extent_world)

            z_extent_vox = int(crop_extent[0] / raw_res[0])
            if z_extent_vox < 1:
                continue
            z_mid = z_extent_vox // 2
            z_world = crop_origin[0] + z_mid * raw_res[0]
            z_vox = int(round((z_world - raw_off[0]) / raw_res[0]))
            z_vox = max(0, min(z_vox, raw_shape[0] - 1))

            y_start = int(round((crop_origin[1] - raw_off[1]) / raw_res[1]))
            x_start = int(round((crop_origin[2] - raw_off[2]) / raw_res[2]))
            y_size = int(round(crop_extent[1] / raw_res[1]))
            x_size = int(round(crop_extent[2] / raw_res[2]))
            y_start = max(0, min(y_start, raw_shape[1] - 1))
            x_start = max(0, min(x_start, raw_shape[2] - 1))
            y_end = min(y_start + y_size, raw_shape[1])
            x_end = min(x_start + x_size, raw_shape[2])

            raw_2d = np.array(raw_arr[z_vox, y_start:y_end, x_start:x_end])
            if raw_2d.size > 500_000:
                step = max(1, int(np.sqrt(raw_2d.size / 500_000)))
                raw_2d = raw_2d[::step, ::step]
            all_vals.append(raw_2d.ravel())

        if not all_vals:
            auto_norms[dataset_name] = NormParams(0.0, 255.0, False)
            continue

        combined = np.concatenate(all_vals)
        p_low = float(np.percentile(combined, percentile_low))
        p_high = float(np.percentile(combined, percentile_high))

        auto_norms[dataset_name] = NormParams(
            min_val=round(p_low, 1),
            max_val=round(p_high, 1),
            inverted=False,
        )
        logger.info(f"  auto-norm {dataset_name}: [{p_low:.1f}, {p_high:.1f}]")

    return auto_norms
