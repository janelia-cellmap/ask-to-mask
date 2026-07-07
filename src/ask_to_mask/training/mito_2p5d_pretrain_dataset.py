"""Label-free, multi-scale EM dataset for masked-image-modeling pretraining.

Unlike the supervised 2.5D datasets, this needs no segmentation/GT at all --
just raw EM. It samples across every scale in each dataset's OME-NGFF
multiscale pyramid (not one fixed physical FOV), so a single pretraining run
sees mitochondria-scale, nucleus-scale, and cell-scale structure without any
per-organelle FOV tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from .mito_2p5d_dataset import (
    apply_stack_augment,
    centered_z_offsets,
    normalize_stack_percentile,
    resize_stack,
)
from .zarr_utils import get_scale_info


@dataclass(frozen=True)
class RawEmScale:
    path: str
    resolution: tuple[float, float, float]
    offset: tuple[float, float, float]
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class RawEmSource:
    dataset: str
    scales: list[RawEmScale]


def discover_raw_em_sources(
    data_root: str,
    em_path_template: str,
    include_datasets: list[str] | None = None,
    skip_datasets: list[str] | None = None,
) -> list[RawEmSource]:
    """Find every dataset's raw EM zarr and its full multiscale pyramid.

    No label/segmentation discovery at all -- covers every dataset with raw EM
    regardless of whether it has any GT or pseudo-labels.
    """
    root = Path(data_root)
    include = set(include_datasets) if include_datasets else None
    skip = set(skip_datasets or [])
    sources: list[RawEmSource] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        dataset = entry.name
        if include is not None and dataset not in include:
            continue
        if dataset in skip:
            continue

        base = em_path_template.format(dataset=dataset).rstrip("/")
        if not Path(base).is_dir():
            continue
        try:
            offsets, resolutions, shapes = get_scale_info(base)
        except Exception:
            continue

        scales = [
            RawEmScale(
                path=str(Path(base) / scale),
                resolution=tuple(float(v) for v in resolutions[scale]),
                offset=tuple(float(v) for v in offsets[scale]),
                shape=tuple(int(v) for v in shapes[scale]),
            )
            for scale in resolutions
        ]
        if scales:
            sources.append(RawEmSource(dataset=dataset, scales=scales))

    return sources


def _filter_sources_by_resolution(
    sources: list[RawEmSource],
    min_resolution_nm: float | None,
    max_resolution_nm: float | None,
) -> list[RawEmSource]:
    """Keep only pyramid scales whose YX resolution falls in [min, max] nm/px.

    Filtering by physical resolution (rather than pyramid index) is what makes
    this comparable across datasets -- pyramid depth and base resolution both
    vary per dataset, so "index 5" means a different physical scale in each.
    """
    filtered: list[RawEmSource] = []
    for source in sources:
        kept = []
        for scale in source.scales:
            yx_res = (scale.resolution[1] + scale.resolution[2]) / 2.0
            if min_resolution_nm is not None and yx_res < min_resolution_nm:
                continue
            if max_resolution_nm is not None and yx_res > max_resolution_nm:
                continue
            kept.append(scale)
        if kept:
            filtered.append(RawEmSource(dataset=source.dataset, scales=kept))
    return filtered


class Mito2p5DSelfSupervisedDataset(Dataset):
    """Label-free, multi-scale EM crops for masked-image-modeling pretraining."""

    def __init__(
        self,
        data_root: str = "/nrs/cellmap/data",
        em_path_template: str = "/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8",
        include_datasets: list[str] | None = None,
        skip_datasets: list[str] | None = None,
        samples_per_epoch: int = 2000,
        output_size: int = 512,
        stack_depth: int = 9,
        z_offsets: list[int] | None = None,
        min_resolution_nm: float | None = None,
        max_resolution_nm: float | None = 512.0,
        split: str = "train",
        val_holdout_fraction: float = 0.1,
        val_holdout_axis: str = "x",
        val_holdout_position: str = "high",
        max_sample_attempts: int = 50,
        seed: int = 42,
        augment: bool = True,
        intensity_augment: bool = True,
        auto_norms_percentile_low: float = 1.0,
        auto_norms_percentile_high: float = 99.0,
    ):
        self.z_offsets = z_offsets if z_offsets is not None else centered_z_offsets(stack_depth)
        self.stack_depth = len(self.z_offsets)
        self.samples_per_epoch = int(samples_per_epoch)
        self.output_size = int(output_size)
        self.min_resolution_nm = min_resolution_nm
        self.max_resolution_nm = max_resolution_nm
        self.split = split
        self.val_holdout_fraction = float(val_holdout_fraction)
        self.val_holdout_axis = val_holdout_axis
        self.val_holdout_position = val_holdout_position
        self.max_sample_attempts = int(max_sample_attempts)
        self.rng = np.random.default_rng(seed)
        self.augment = bool(augment)
        self.intensity_augment = bool(intensity_augment)
        self.auto_norms_percentile_low = float(auto_norms_percentile_low)
        self.auto_norms_percentile_high = float(auto_norms_percentile_high)
        self._zarr_cache: dict[str, Any] = {}

        all_sources = discover_raw_em_sources(
            data_root=data_root,
            em_path_template=em_path_template,
            include_datasets=include_datasets,
            skip_datasets=skip_datasets,
        )
        self.sources = _filter_sources_by_resolution(
            all_sources, self.min_resolution_nm, self.max_resolution_nm
        )
        if not self.sources:
            raise RuntimeError(
                f"No raw EM datasets under {data_root} have a pyramid scale in "
                f"[{self.min_resolution_nm}, {self.max_resolution_nm}] nm/px"
            )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        for _ in range(self.max_sample_attempts):
            sample = self._try_sample()
            if sample is not None:
                return sample
        raise RuntimeError(
            f"Could not sample a self-supervised EM crop after "
            f"{self.max_sample_attempts} attempts"
        )

    def _pick_scale(self, source: RawEmSource) -> RawEmScale:
        idx = int(self.rng.integers(len(source.scales)))
        return source.scales[idx]

    def _apply_split_bounds(
        self, y_min: float, y_max: float, x_min: float, x_max: float
    ) -> tuple[float, float, float, float]:
        if self.val_holdout_fraction <= 0 or self.split not in {"train", "val", "validation"}:
            return y_min, y_max, x_min, x_max
        axis = self.val_holdout_axis.lower()
        position = self.val_holdout_position.lower()
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

    def _try_sample(self) -> dict[str, Any] | None:
        source = self.sources[self.rng.integers(len(self.sources))]
        scale = self._pick_scale(source)
        res = np.array(scale.resolution, dtype=float)
        off = np.array(scale.offset, dtype=float)
        shape = np.array(scale.shape, dtype=int)

        fov_y = self.output_size * res[1]
        fov_x = self.output_size * res[2]
        z_min, z_max = off[0], off[0] + (shape[0] - 1) * res[0]
        y_min, y_max = off[1], off[1] + shape[1] * res[1] - fov_y
        x_min, x_max = off[2], off[2] + shape[2] * res[2] - fov_x
        y_min, y_max, x_min, x_max = self._apply_split_bounds(y_min, y_max, x_min, x_max)
        if z_max < z_min or y_max < y_min or x_max < x_min:
            return None

        z_world = float(self.rng.uniform(z_min, z_max))
        y_world = float(self.rng.uniform(y_min, y_max))
        x_world = float(self.rng.uniform(x_min, x_max))

        arr = self._open_zarr(scale.path)
        stack = self._read_stack(arr, scale, z_world, y_world, x_world)
        if stack is None:
            return None

        stack = normalize_stack_percentile(
            stack,
            low_percentile=self.auto_norms_percentile_low,
            high_percentile=self.auto_norms_percentile_high,
        )
        stack = resize_stack(stack, self.output_size)

        if self.augment:
            dummy = np.zeros(stack.shape[-2:], dtype=np.float32)
            stack, _, _, _ = apply_stack_augment(
                self.rng, stack, dummy, dummy, intensity=self.intensity_augment
            )

        metadata = {
            "dataset": source.dataset,
            "split": self.split,
            "scale_path": scale.path,
            "resolution_nm": list(scale.resolution),
            "world_origin_nm_zyx": [z_world, y_world, x_world],
            "z_offsets": list(self.z_offsets),
        }
        return {"em": torch.from_numpy(stack.astype(np.float32)), "metadata": metadata}

    def _read_stack(
        self, arr, scale: RawEmScale, z: float, y: float, x: float
    ) -> np.ndarray | None:
        res = np.array(scale.resolution, dtype=float)
        off = np.array(scale.offset, dtype=float)
        shape = np.array(scale.shape, dtype=int)
        center_z = int(round((z - off[0]) / res[0]))
        y_start = int(round((y - off[1]) / res[1]))
        x_start = int(round((x - off[2]) / res[2]))
        y_size = self.output_size
        x_size = self.output_size
        y_start = max(0, min(y_start, shape[1] - y_size))
        x_start = max(0, min(x_start, shape[2] - x_size))
        y_end = min(y_start + y_size, shape[1])
        x_end = min(x_start + x_size, shape[2])
        if y_end <= y_start or x_end <= x_start:
            return None

        slices = []
        for offset in self.z_offsets:
            z_idx = int(np.clip(center_z + offset, 0, shape[0] - 1))
            raw_2d = np.array(arr[z_idx, y_start:y_end, x_start:x_end])
            if raw_2d.shape != (y_end - y_start, x_end - x_start):
                return None
            slices.append(raw_2d)
        return np.stack(slices, axis=0)


def collate_self_supervised(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "em": torch.stack([item["em"] for item in batch], dim=0),
        "metadata": [item.get("metadata", {}) for item in batch],
    }


def build_mito_2p5d_pretrain_datasets(config: dict) -> tuple[Dataset, Dataset]:
    """Build train/val self-supervised datasets from a pretraining config."""
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    train_samples = int(data_cfg.get("samples_per_epoch", 2000))
    val_samples = int(train_cfg.get("num_validation_samples", 64))

    common = dict(
        data_root=data_cfg.get("data_root", "/nrs/cellmap/data"),
        em_path_template=data_cfg.get(
            "em_path_template",
            "/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8",
        ),
        include_datasets=data_cfg.get("include_datasets"),
        skip_datasets=data_cfg.get("skip_datasets"),
        output_size=data_cfg.get("output_size", 512),
        stack_depth=data_cfg.get("stack_depth", 9),
        z_offsets=data_cfg.get("z_offsets"),
        min_resolution_nm=data_cfg.get("min_resolution_nm"),
        max_resolution_nm=data_cfg.get("max_resolution_nm", 512.0),
        val_holdout_fraction=data_cfg.get("val_holdout_fraction", 0.1),
        val_holdout_axis=data_cfg.get("val_holdout_axis", "x"),
        val_holdout_position=data_cfg.get("val_holdout_position", "high"),
        max_sample_attempts=data_cfg.get("max_sample_attempts", 50),
        auto_norms_percentile_low=data_cfg.get("auto_norms_percentile_low", 1.0),
        auto_norms_percentile_high=data_cfg.get("auto_norms_percentile_high", 99.0),
    )
    train_dataset = Mito2p5DSelfSupervisedDataset(
        **common,
        samples_per_epoch=train_samples,
        split="train",
        seed=seed,
        augment=data_cfg.get("augment", True),
        intensity_augment=data_cfg.get("intensity_augment", True),
    )
    val_dataset = Mito2p5DSelfSupervisedDataset(
        **common,
        samples_per_epoch=val_samples,
        split="val",
        seed=seed + 100_000,
        augment=False,
        intensity_augment=False,
    )
    return train_dataset, val_dataset
