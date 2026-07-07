"""Fixed-FOV mitochondria dataset backed by CellMap inference segmentations."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    zoom as ndimage_zoom,
)
from torch.utils.data import Dataset

from .dataset import TARGET_SIZE
from .zarr_utils import get_scale_info

logger = logging.getLogger(__name__)


FIXED_PROMPT = "CLASS=mitochondria; OUTPUT=red_on_black"


@dataclass(frozen=True)
class ScaleSpec:
    path: str
    scale: str
    resolution: tuple[float, float, float]
    offset: tuple[float, float, float]
    shape: tuple[int, int, int]

    @property
    def begin_nm(self) -> np.ndarray:
        return np.array(self.offset, dtype=float)

    @property
    def end_nm(self) -> np.ndarray:
        return self.begin_nm + np.array(self.resolution, dtype=float) * np.array(
            self.shape, dtype=float
        )


@dataclass(frozen=True)
class InferenceMitoPair:
    dataset: str
    raw_base: str
    label_base: str
    raw: ScaleSpec
    label: ScaleSpec
    overlap_begin_nm: tuple[float, float, float]
    overlap_end_nm: tuple[float, float, float]


def _as_tuple(values) -> tuple:
    return tuple(float(v) for v in values)


def _shape_tuple(values) -> tuple[int, int, int]:
    return tuple(int(v) for v in values)


def _format_template(template: str, dataset: str) -> str:
    return template.format(dataset=dataset).rstrip("/")


def _select_scale(
    base_path: str,
    target_resolution_nm: float | None,
    require_exact: bool = False,
    tolerance_nm: float = 1e-6,
) -> ScaleSpec:
    offsets, resolutions, shapes = get_scale_info(base_path)
    if target_resolution_nm is None:
        scale = min(resolutions, key=lambda s: np.prod(resolutions[s]))
    else:
        target = np.array([target_resolution_nm] * 3, dtype=float)
        scale = min(
            resolutions,
            key=lambda s: float(np.linalg.norm(np.array(resolutions[s], dtype=float) - target)),
        )
        if require_exact and not np.allclose(
            np.array(resolutions[scale], dtype=float), target, atol=tolerance_nm
        ):
            raise ValueError(
                f"{base_path} has no exact {target_resolution_nm} nm scale; "
                f"closest is {scale} with {resolutions[scale]}"
            )
    return ScaleSpec(
        path=str(Path(base_path) / scale),
        scale=scale,
        resolution=_as_tuple(resolutions[scale]),
        offset=_as_tuple(offsets[scale]),
        shape=_shape_tuple(shapes[scale]),
    )


def _pair_to_json(pair: InferenceMitoPair) -> dict:
    def spec_to_json(spec: ScaleSpec) -> dict:
        return {
            "path": spec.path,
            "scale": spec.scale,
            "resolution": list(spec.resolution),
            "offset": list(spec.offset),
            "shape": list(spec.shape),
        }

    return {
        "dataset": pair.dataset,
        "raw_base": pair.raw_base,
        "label_base": pair.label_base,
        "raw": spec_to_json(pair.raw),
        "label": spec_to_json(pair.label),
        "overlap_begin_nm": list(pair.overlap_begin_nm),
        "overlap_end_nm": list(pair.overlap_end_nm),
    }


def _pair_from_json(data: dict) -> InferenceMitoPair:
    def spec_from_json(spec: dict) -> ScaleSpec:
        return ScaleSpec(
            path=spec["path"],
            scale=spec["scale"],
            resolution=_as_tuple(spec["resolution"]),
            offset=_as_tuple(spec["offset"]),
            shape=_shape_tuple(spec["shape"]),
        )

    return InferenceMitoPair(
        dataset=data["dataset"],
        raw_base=data["raw_base"],
        label_base=data["label_base"],
        raw=spec_from_json(data["raw"]),
        label=spec_from_json(data["label"]),
        overlap_begin_nm=_as_tuple(data["overlap_begin_nm"]),
        overlap_end_nm=_as_tuple(data["overlap_end_nm"]),
    )


def build_inference_mito_index(
    data_root: str,
    em_path_template: str,
    segmentation_path_template: str,
    label_name: str = "mito",
    include_datasets: list[str] | None = None,
    skip_datasets: list[str] | None = None,
    raw_target_resolution_nm: float | None = None,
    label_target_resolution_nm: float | None = None,
    require_exact_resolution: bool = False,
    raw_require_exact_resolution: bool | None = None,
    label_require_exact_resolution: bool | None = None,
    fov_nm: float = 1600.0,
) -> list[InferenceMitoPair]:
    """Build path pairs from immediate dataset directories under data_root."""
    root = Path(data_root)
    include = set(include_datasets) if include_datasets else None
    skip = set(skip_datasets or [])
    pairs: list[InferenceMitoPair] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        dataset = entry.name
        if include is not None and dataset not in include:
            continue
        if dataset in skip:
            continue

        raw_base = _format_template(em_path_template, dataset)
        label_base = _format_template(segmentation_path_template, dataset)
        if not Path(label_base).name == label_name:
            label_base = str(Path(label_base) / label_name)

        if not Path(raw_base).is_dir() or not Path(label_base).is_dir():
            continue

        try:
            raw = _select_scale(
                raw_base,
                raw_target_resolution_nm,
                require_exact=(
                    require_exact_resolution
                    if raw_require_exact_resolution is None
                    else raw_require_exact_resolution
                ),
            )
            label = _select_scale(
                label_base,
                label_target_resolution_nm,
                require_exact=(
                    require_exact_resolution
                    if label_require_exact_resolution is None
                    else label_require_exact_resolution
                ),
            )
        except Exception as exc:
            logger.debug("Skipping %s: %s", dataset, exc)
            continue

        overlap_begin = np.maximum(raw.begin_nm, label.begin_nm)
        overlap_end = np.minimum(raw.end_nm, label.end_nm)
        raw_shape_nm = raw.end_nm - raw.begin_nm
        if raw_shape_nm[1] < fov_nm or raw_shape_nm[2] < fov_nm:
            continue
        overlap_shape = overlap_end - overlap_begin
        if overlap_shape[0] < max(raw.resolution[0], label.resolution[0]):
            continue

        pairs.append(
            InferenceMitoPair(
                dataset=dataset,
                raw_base=raw_base,
                label_base=label_base,
                raw=raw,
                label=label,
                overlap_begin_nm=tuple(float(v) for v in overlap_begin),
                overlap_end_nm=tuple(float(v) for v in overlap_end),
            )
        )

    return pairs


def load_or_build_inference_mito_index(
    index_path: str | None,
    rebuild: bool = False,
    **kwargs,
) -> list[InferenceMitoPair]:
    if index_path and Path(index_path).is_file() and not rebuild:
        with open(index_path) as f:
            return [_pair_from_json(item) for item in json.load(f)]

    pairs = build_inference_mito_index(**kwargs)
    if index_path:
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([_pair_to_json(pair) for pair in pairs], f, indent=2)
    return pairs


class InferenceMitoFixedFovDataset(Dataset):
    """Samples fixed physical FOV EM/mitochondria pairs from inference segmentations."""

    def __init__(
        self,
        data_root: str = "/nrs/cellmap/data",
        em_path_template: str = "/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8",
        segmentation_path_template: str = "/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/labels/inference/segmentations",
        label_name: str = "mito",
        index_path: str | None = None,
        rebuild_index: bool = False,
        include_datasets: list[str] | None = None,
        skip_datasets: list[str] | None = None,
        samples_per_epoch: int = 2000,
        split: str = "train",
        val_holdout_fraction: float = 0.1,
        val_holdout_axis: str = "x",
        val_holdout_position: str = "high",
        fov_nm: float = 1600.0,
        fov_tolerance_nm: float = 0.0,
        raw_target_resolution_nm: float | None = None,
        label_target_resolution_nm: float | None = None,
        require_exact_resolution: bool = False,
        raw_require_exact_resolution: bool | None = None,
        label_require_exact_resolution: bool | None = None,
        min_mask_fraction: float = 0.01,
        min_valid_fraction: float = 0.25,
        label_quality: str = "pseudo",
        label_weight: float = 0.2,
        conservative_pseudo_targets: bool = True,
        foreground_erosion_px: int = 2,
        boundary_band_px: int = 3,
        boundary_weight: float = 0.0,
        max_sample_attempts: int = 100,
        output_size: int = TARGET_SIZE,
        prompt: str = FIXED_PROMPT,
        seed: int = 42,
        augment: bool = True,
        auto_norms_percentile_low: float = 1.0,
        auto_norms_percentile_high: float = 99.0,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.split = split
        self.val_holdout_fraction = float(val_holdout_fraction)
        self.val_holdout_axis = val_holdout_axis
        self.val_holdout_position = val_holdout_position
        self.fov_nm = float(fov_nm)
        self.fov_tolerance_nm = float(fov_tolerance_nm)
        self.min_mask_fraction = min_mask_fraction
        self.min_valid_fraction = min_valid_fraction
        self.label_quality = label_quality
        self.label_weight = float(label_weight)
        self.conservative_pseudo_targets = conservative_pseudo_targets
        self.foreground_erosion_px = int(foreground_erosion_px)
        self.boundary_band_px = int(boundary_band_px)
        self.boundary_weight = float(boundary_weight)
        self.max_sample_attempts = max_sample_attempts
        self.output_size = output_size
        self.prompt = prompt
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.auto_norms_percentile_low = auto_norms_percentile_low
        self.auto_norms_percentile_high = auto_norms_percentile_high
        self._zarr_cache = {}

        self.pairs = load_or_build_inference_mito_index(
            index_path=index_path,
            rebuild=rebuild_index,
            data_root=data_root,
            em_path_template=em_path_template,
            segmentation_path_template=segmentation_path_template,
            label_name=label_name,
            include_datasets=include_datasets,
            skip_datasets=skip_datasets,
            raw_target_resolution_nm=raw_target_resolution_nm,
            label_target_resolution_nm=label_target_resolution_nm,
            require_exact_resolution=require_exact_resolution,
            raw_require_exact_resolution=raw_require_exact_resolution,
            label_require_exact_resolution=label_require_exact_resolution,
            fov_nm=self.fov_nm,
        )
        if not self.pairs:
            raise RuntimeError("No fixed-FOV mitochondria inference pairs found")
        logger.info(
            "InferenceMitoFixedFovDataset: %d datasets, split=%s, fov=%.1f nm, prompt=%r",
            len(self.pairs),
            self.split,
            self.fov_nm,
            self.prompt,
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(
        self, idx: int
    ) -> tuple[Image.Image, Image.Image, str, Image.Image, float]:
        cond, target, valid_mask, _metadata = self.sample_with_metadata()
        return cond, target, self.prompt, valid_mask, self.label_weight

    def sample_with_metadata(self) -> tuple[Image.Image, Image.Image, Image.Image, dict]:
        for _ in range(self.max_sample_attempts):
            pair = self.pairs[self.rng.integers(len(self.pairs))]
            sample = self._try_sample_pair(pair)
            if sample is not None:
                return sample
        raise RuntimeError(
            f"Could not sample a mitochondria crop with min_mask_fraction="
            f"{self.min_mask_fraction} after {self.max_sample_attempts} attempts"
        )

    def _try_sample_pair(
        self, pair: InferenceMitoPair
    ) -> tuple[Image.Image, Image.Image, Image.Image, dict] | None:
        raw_res = np.array(pair.raw.resolution, dtype=float)
        raw_begin = pair.raw.begin_nm
        raw_end = pair.raw.end_nm

        z_min = raw_begin[0]
        z_max = raw_end[0] - raw_res[0]
        y_min = raw_begin[1]
        y_max = raw_end[1] - self.fov_nm
        x_min = raw_begin[2]
        x_max = raw_end[2] - self.fov_nm
        y_min, y_max, x_min, x_max = self._apply_split_bounds(
            y_min, y_max, x_min, x_max
        )
        if z_max < z_min or y_max < y_min or x_max < x_min:
            return None

        z_world = float(self.rng.uniform(z_min, z_max))
        y_world = float(self.rng.uniform(y_min, y_max))
        x_world = float(self.rng.uniform(x_min, x_max))

        raw_arr = self._open_zarr(pair.raw.path)
        label_arr = self._open_zarr(pair.label.path)

        raw_2d = self._read_raw(raw_arr, pair.raw, z_world, y_world, x_world)
        label_result = self._read_label(label_arr, pair.label, z_world, y_world, x_world)
        if raw_2d is None or label_result is None or raw_2d.size == 0:
            return None
        label_2d, label_valid = label_result

        mask = label_2d > 0
        valid_fraction = float(label_valid.mean())
        if valid_fraction < self.min_valid_fraction:
            return None
        valid_pixels = label_valid > 0
        mask_fraction_valid = float(mask[valid_pixels].mean()) if valid_pixels.any() else 0.0
        if mask_fraction_valid < self.min_mask_fraction:
            return None

        raw_uint8 = self._normalize_raw(raw_2d)
        raw_rgb = np.stack([raw_uint8] * 3, axis=-1)
        valid_resized = self._resize_mask(label_valid > 0, self.output_size)
        mask_resized = self._resize_mask(mask, self.output_size)
        target_mask, confidence = self._build_target_and_confidence(
            mask_resized, valid_resized
        )
        loss_weight = confidence
        raw_resized = Image.fromarray(raw_rgb).resize(
            (self.output_size, self.output_size), Image.LANCZOS
        )

        target_rgb = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        target_rgb[target_mask] = (255, 0, 0)
        target = Image.fromarray(target_rgb)
        valid_mask = Image.fromarray(
            np.clip(loss_weight * 255.0, 0, 255).astype(np.uint8), mode="L"
        )

        if self.augment:
            raw_resized, target, valid_mask = self._augment(raw_resized, target, valid_mask)

        metadata = {
            "dataset": pair.dataset,
            "split": self.split,
            "val_holdout_fraction": self.val_holdout_fraction,
            "val_holdout_axis": self.val_holdout_axis,
            "val_holdout_position": self.val_holdout_position,
            "raw_path": pair.raw.path,
            "label_path": pair.label.path,
            "raw_resolution_nm": list(pair.raw.resolution),
            "label_resolution_nm": list(pair.label.resolution),
            "fov_nm_yx": [self.fov_nm, self.fov_nm],
            "world_origin_nm_zyx": [z_world, y_world, x_world],
            "raw_shape_yx": list(raw_2d.shape),
            "label_shape_yx": list(label_2d.shape),
            "mask_fraction": float(mask.mean()),
            "mask_fraction_valid": mask_fraction_valid,
            "valid_loss_fraction": valid_fraction,
            "label_quality": self.label_quality,
            "LABEL_QUALITY": self.label_quality,
            "label_weight": self.label_weight,
            "conservative_pseudo_targets": self.conservative_pseudo_targets,
            "foreground_erosion_px": self.foreground_erosion_px,
            "boundary_band_px": self.boundary_band_px,
            "boundary_weight": self.boundary_weight,
            "loss_mask": (
                "valid where crop overlaps segmentation; pseudo foreground is eroded, "
                "boundary is ignored/low-weight, far background is confident"
            ),
            "prompt": self.prompt,
        }
        return raw_resized, target, valid_mask, metadata

    def _apply_split_bounds(
        self,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
    ) -> tuple[float, float, float, float]:
        if self.val_holdout_fraction <= 0:
            return y_min, y_max, x_min, x_max
        if self.split not in {"train", "val", "validation"}:
            return y_min, y_max, x_min, x_max

        axis = self.val_holdout_axis.lower()
        position = self.val_holdout_position.lower()
        if axis not in {"x", "y"}:
            raise ValueError(f"Unknown val_holdout_axis={self.val_holdout_axis!r}")
        if position not in {"low", "high"}:
            raise ValueError(
                f"Unknown val_holdout_position={self.val_holdout_position!r}"
            )

        lo, hi = (x_min, x_max) if axis == "x" else (y_min, y_max)
        span = hi - lo
        holdout = span * self.val_holdout_fraction
        if holdout <= 0:
            return y_min, y_max, x_min, x_max

        if position == "high":
            val_lo, val_hi = hi - holdout, hi
            train_lo, train_hi = lo, val_lo
        else:
            val_lo, val_hi = lo, lo + holdout
            train_lo, train_hi = val_hi, hi

        use_val = self.split in {"val", "validation"}
        new_lo, new_hi = (val_lo, val_hi) if use_val else (train_lo, train_hi)
        if axis == "x":
            return y_min, y_max, new_lo, new_hi
        return new_lo, new_hi, x_min, x_max

    def _open_zarr(self, path: str):
        arr = self._zarr_cache.get(path)
        if arr is None:
            arr = zarr.open(path, mode="r")
            self._zarr_cache[path] = arr
        return arr

    def _read_raw(self, arr, spec: ScaleSpec, z: float, y: float, x: float):
        return self._read_window(arr, spec, z, y, x, order="raw")

    def _read_label(self, arr, spec: ScaleSpec, z: float, y: float, x: float):
        return self._read_window_with_valid(arr, spec, z, y, x)

    def _read_window(self, arr, spec: ScaleSpec, z: float, y: float, x: float, order: str):
        res = np.array(spec.resolution, dtype=float)
        off = np.array(spec.offset, dtype=float)
        shape = np.array(spec.shape, dtype=int)
        z_idx = int(round((z - off[0]) / res[0]))
        y_start = int(round((y - off[1]) / res[1]))
        x_start = int(round((x - off[2]) / res[2]))
        y_size = int(round(self.fov_nm / res[1]))
        x_size = int(round(self.fov_nm / res[2]))

        if z_idx < 0 or z_idx >= shape[0]:
            return None
        y_start = max(0, min(y_start, shape[1] - y_size))
        x_start = max(0, min(x_start, shape[2] - x_size))
        y_end = min(y_start + y_size, shape[1])
        x_end = min(x_start + x_size, shape[2])
        if y_end <= y_start or x_end <= x_start:
            return None
        return np.array(arr[z_idx, y_start:y_end, x_start:x_end])

    def _read_window_with_valid(self, arr, spec: ScaleSpec, z: float, y: float, x: float):
        res = np.array(spec.resolution, dtype=float)
        off = np.array(spec.offset, dtype=float)
        shape = np.array(spec.shape, dtype=int)
        z_idx = int(round((z - off[0]) / res[0]))
        y_start = int(round((y - off[1]) / res[1]))
        x_start = int(round((x - off[2]) / res[2]))
        y_size = int(round(self.fov_nm / res[1]))
        x_size = int(round(self.fov_nm / res[2]))

        out = np.zeros((y_size, x_size), dtype=arr.dtype)
        valid = np.zeros((y_size, x_size), dtype=np.uint8)
        if z_idx < 0 or z_idx >= shape[0]:
            return out, valid

        src_y0 = max(0, y_start)
        src_x0 = max(0, x_start)
        src_y1 = min(shape[1], y_start + y_size)
        src_x1 = min(shape[2], x_start + x_size)
        if src_y1 <= src_y0 or src_x1 <= src_x0:
            return out, valid

        dst_y0 = src_y0 - y_start
        dst_x0 = src_x0 - x_start
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = np.array(
            arr[z_idx, src_y0:src_y1, src_x0:src_x1]
        )
        valid[dst_y0:dst_y1, dst_x0:dst_x1] = 1
        return out, valid

    def _normalize_raw(self, raw: np.ndarray) -> np.ndarray:
        raw = raw.astype(np.float32)
        low = float(np.percentile(raw, self.auto_norms_percentile_low))
        high = float(np.percentile(raw, self.auto_norms_percentile_high))
        if high <= low:
            high = low + 1.0
        raw = (raw - low) / (high - low)
        raw = np.clip(raw, 0.0, 1.0)
        return (raw * 255).astype(np.uint8)

    def _resize_mask(self, mask: np.ndarray, output_size: int) -> np.ndarray:
        zoom_y = output_size / mask.shape[0]
        zoom_x = output_size / mask.shape[1]
        resized = ndimage_zoom(mask.astype(np.uint8), (zoom_y, zoom_x), order=0)
        return resized[:output_size, :output_size] > 0

    def _resize_float(self, weights: np.ndarray, output_size: int) -> np.ndarray:
        zoom_y = output_size / weights.shape[0]
        zoom_x = output_size / weights.shape[1]
        resized = ndimage_zoom(weights.astype(np.float32), (zoom_y, zoom_x), order=0)
        return resized[:output_size, :output_size].astype(np.float32)

    def _build_target_and_confidence(
        self, mask: np.ndarray, valid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.conservative_pseudo_targets or self.label_quality == "gt":
            return mask, valid.astype(np.float32)

        eroded = mask.copy()
        if self.foreground_erosion_px > 0:
            eroded = binary_erosion(mask, iterations=self.foreground_erosion_px)
        dilated = mask.copy()
        if self.boundary_band_px > 0:
            dilated = binary_dilation(mask, iterations=self.boundary_band_px)

        confident_background = (~dilated) & valid
        confident_foreground = eroded & valid
        boundary = valid & ~(confident_background | confident_foreground)

        confidence = np.zeros(mask.shape, dtype=np.float32)
        confidence[confident_background] = 1.0
        confidence[confident_foreground] = 1.0
        if self.boundary_weight > 0:
            confidence[boundary] = self.boundary_weight
        return confident_foreground, confidence

    def _augment(
        self, raw: Image.Image, target: Image.Image, valid_mask: Image.Image
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        k = int(self.rng.integers(4))
        if k:
            raw = raw.rotate(90 * k)
            target = target.rotate(90 * k)
            valid_mask = valid_mask.rotate(90 * k)
        if self.rng.random() < 0.5:
            raw = raw.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
            valid_mask = valid_mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.rng.random() < 0.5:
            raw = raw.transpose(Image.FLIP_TOP_BOTTOM)
            target = target.transpose(Image.FLIP_TOP_BOTTOM)
            valid_mask = valid_mask.transpose(Image.FLIP_TOP_BOTTOM)
        return raw, target, valid_mask


def export_preview_pairs(
    dataset: InferenceMitoFixedFovDataset,
    output_dir: str | Path,
    num_samples: int = 8,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = []
    for i in range(num_samples):
        raw, target, valid_mask, meta = dataset.sample_with_metadata()
        raw_path = output / f"sample_{i:03d}_raw.png"
        target_path = output / f"sample_{i:03d}_target.png"
        valid_path = output / f"sample_{i:03d}_valid_loss_mask.png"
        raw.save(raw_path)
        target.save(target_path)
        valid_mask.save(valid_path)
        meta = dict(meta)
        meta["raw_preview"] = raw_path.name
        meta["target_preview"] = target_path.name
        meta["valid_loss_mask_preview"] = valid_path.name
        metadata.append(meta)
    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return output


def dataset_from_training_config(config: dict) -> InferenceMitoFixedFovDataset:
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    return InferenceMitoFixedFovDataset(
        data_root=data_cfg.get("data_root", "/nrs/cellmap/data"),
        em_path_template=data_cfg.get(
            "em_path_template",
            "/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8",
        ),
        segmentation_path_template=data_cfg.get(
            "segmentation_path_template",
            "/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/labels/inference/segmentations",
        ),
        label_name=data_cfg.get("label_name", "mito"),
        index_path=data_cfg.get("index_path"),
        rebuild_index=data_cfg.get("rebuild_index", False),
        include_datasets=data_cfg.get("include_datasets"),
        skip_datasets=data_cfg.get("skip_datasets"),
        samples_per_epoch=data_cfg.get("samples_per_epoch", 2000),
        fov_nm=data_cfg.get("fov_nm", 1600.0),
        fov_tolerance_nm=data_cfg.get("fov_tolerance_nm", 0.0),
        raw_target_resolution_nm=data_cfg.get("raw_target_resolution_nm"),
        label_target_resolution_nm=data_cfg.get("label_target_resolution_nm"),
        require_exact_resolution=data_cfg.get("require_exact_resolution", False),
        min_mask_fraction=data_cfg.get("min_mask_fraction", 0.01),
        min_valid_fraction=data_cfg.get("min_valid_fraction", 0.25),
        label_quality=data_cfg.get("label_quality", "pseudo"),
        label_weight=data_cfg.get("label_weight", 0.2),
        conservative_pseudo_targets=data_cfg.get("conservative_pseudo_targets", True),
        foreground_erosion_px=data_cfg.get("foreground_erosion_px", 2),
        boundary_band_px=data_cfg.get("boundary_band_px", 3),
        boundary_weight=data_cfg.get("boundary_weight", 0.0),
        max_sample_attempts=data_cfg.get("max_sample_attempts", 100),
        prompt=data_cfg.get("prompt", FIXED_PROMPT),
        seed=train_cfg.get("seed", 42),
        augment=data_cfg.get("augment", True),
        auto_norms_percentile_low=data_cfg.get("auto_norms_percentile_low", 1.0),
        auto_norms_percentile_high=data_cfg.get("auto_norms_percentile_high", 99.0),
    )
