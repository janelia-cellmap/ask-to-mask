"""2.5D supervised mitochondria datasets.

The classes here intentionally reuse the existing fixed-FOV CellMap samplers for
path discovery and window selection. Their public output is a plain tensor
sample suitable for supervised image-to-image segmentation:

    EM z-stack [D, H, W] -> center-slice mitochondria mask [1, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    zoom as ndimage_zoom,
)
from torch.utils.data import Dataset

from .dataset import FixedFovCellMapGtDataset, TARGET_SIZE
from .inference_mito_dataset import (
    InferenceMitoFixedFovDataset,
    InferenceMitoPair,
    ScaleSpec,
    _select_scale,
)


@dataclass(frozen=True)
class Mito2p5DTargetConfig:
    """Auxiliary target settings for supervised mitochondria segmentation."""

    boundary_radius_px: int = 2
    include_boundary: bool = True
    include_distance: bool = False
    distance_max_px: float = 32.0


def centered_z_offsets(stack_depth: int) -> list[int]:
    """Return symmetric integer z offsets such as [-4, ..., 0, ..., 4]."""
    if stack_depth < 1 or stack_depth % 2 == 0:
        raise ValueError("stack_depth must be a positive odd integer")
    radius = stack_depth // 2
    return list(range(-radius, radius + 1))


def mask_to_boundary(mask: np.ndarray, radius_px: int = 2) -> np.ndarray:
    """Build a binary boundary band from a binary mask."""
    mask = mask.astype(bool)
    radius_px = int(radius_px)
    if radius_px <= 0:
        return np.zeros(mask.shape, dtype=np.float32)
    dilated = binary_dilation(mask, iterations=radius_px)
    eroded = binary_erosion(mask, iterations=radius_px)
    return (dilated ^ eroded).astype(np.float32)


def mask_to_signed_distance(mask: np.ndarray, max_distance_px: float = 32.0) -> np.ndarray:
    """Build a clipped signed distance target in [-1, 1].

    Positive values are inside mitochondria; negative values are outside.
    """
    mask = mask.astype(bool)
    max_distance_px = max(float(max_distance_px), 1.0)
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    sdf = inside - outside
    sdf = np.clip(sdf / max_distance_px, -1.0, 1.0)
    return sdf.astype(np.float32)


def resize_stack(stack: np.ndarray, output_size: int) -> np.ndarray:
    """Resize [D, H, W] float stack to [D, output_size, output_size]."""
    if stack.shape[-2:] == (output_size, output_size):
        return stack.astype(np.float32, copy=False)
    zoom_y = output_size / stack.shape[-2]
    zoom_x = output_size / stack.shape[-1]
    resized = ndimage_zoom(stack.astype(np.float32), (1.0, zoom_y, zoom_x), order=1)
    return resized[:, :output_size, :output_size].astype(np.float32)


def resize_mask(mask: np.ndarray, output_size: int) -> np.ndarray:
    """Nearest-neighbor resize for binary or weight masks."""
    if mask.shape == (output_size, output_size):
        return mask
    zoom_y = output_size / mask.shape[0]
    zoom_x = output_size / mask.shape[1]
    resized = ndimage_zoom(mask.astype(np.float32), (zoom_y, zoom_x), order=0)
    return resized[:output_size, :output_size]


def normalize_stack_percentile(
    stack: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Normalize a raw EM stack to [0, 1] using one percentile window."""
    stack = stack.astype(np.float32)
    low = float(np.percentile(stack, low_percentile))
    high = float(np.percentile(stack, high_percentile))
    if high <= low:
        high = low + 1.0
    stack = (stack - low) / (high - low)
    return np.clip(stack, 0.0, 1.0).astype(np.float32)


def apply_stack_augment(
    rng: np.random.Generator,
    stack: np.ndarray,
    mask: np.ndarray,
    valid_mask: np.ndarray,
    intensity: bool = True,
    extra_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Apply spatial transforms to stack/masks and intensity jitter to EM only.

    `extra_masks` (e.g. an aligned comparison-baseline mask) get the same
    rotation/flip as `mask`/`valid_mask` but no intensity jitter.
    """
    extra_masks = dict(extra_masks) if extra_masks else {}
    k = int(rng.integers(4))
    if k:
        stack = np.rot90(stack, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
        valid_mask = np.rot90(valid_mask, k, axes=(0, 1)).copy()
        extra_masks = {name: np.rot90(arr, k, axes=(0, 1)).copy() for name, arr in extra_masks.items()}
    if rng.random() < 0.5:
        stack = np.flip(stack, axis=2).copy()
        mask = np.flip(mask, axis=1).copy()
        valid_mask = np.flip(valid_mask, axis=1).copy()
        extra_masks = {name: np.flip(arr, axis=1).copy() for name, arr in extra_masks.items()}
    if rng.random() < 0.5:
        stack = np.flip(stack, axis=1).copy()
        mask = np.flip(mask, axis=0).copy()
        valid_mask = np.flip(valid_mask, axis=0).copy()
        extra_masks = {name: np.flip(arr, axis=0).copy() for name, arr in extra_masks.items()}

    if intensity:
        brightness = float(rng.uniform(-0.08, 0.08))
        contrast = float(rng.uniform(0.85, 1.20))
        gamma = float(rng.uniform(0.85, 1.20))
        noise_sigma = float(rng.uniform(0.0, 0.025))
        mean = float(stack.mean())
        stack = (stack - mean) * contrast + mean + brightness
        stack = np.clip(stack, 0.0, 1.0)
        stack = stack**gamma
        if noise_sigma > 0:
            stack = stack + rng.normal(0.0, noise_sigma, size=stack.shape)
        stack = np.clip(stack, 0.0, 1.0).astype(np.float32)

    return stack, mask, valid_mask, extra_masks


def finalize_mito_2p5d_sample(
    stack: np.ndarray,
    mask: np.ndarray,
    valid_mask: np.ndarray,
    metadata: dict[str, Any],
    target_cfg: Mito2p5DTargetConfig,
    sample_weight: float = 1.0,
    comparison_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Convert numpy arrays to the common supervised sample dict."""
    mask = mask.astype(bool)
    valid_mask = valid_mask.astype(np.float32)
    sample: dict[str, Any] = {
        "em": torch.from_numpy(stack.astype(np.float32)),
        "target": torch.from_numpy(mask.astype(np.float32)[None, ...]),
        "valid_mask": torch.from_numpy(valid_mask[None, ...]),
        "metadata": metadata,
        "sample_weight": float(sample_weight),
    }
    if target_cfg.include_boundary:
        boundary = mask_to_boundary(mask, target_cfg.boundary_radius_px)
        sample["boundary_target"] = torch.from_numpy(boundary[None, ...])
    if target_cfg.include_distance:
        distance = mask_to_signed_distance(mask, target_cfg.distance_max_px)
        sample["distance_target"] = torch.from_numpy(distance[None, ...])
    if comparison_masks:
        sample["comparison_masks"] = {
            name: torch.from_numpy(arr.astype(np.float32)[None, ...])
            for name, arr in comparison_masks.items()
        }
    return sample


class Mito2p5DInferenceMitoDataset(InferenceMitoFixedFovDataset):
    """Fixed-FOV pseudo-label dataset that returns 2.5D supervised tensors."""

    def __init__(
        self,
        *args,
        stack_depth: int = 9,
        z_offsets: list[int] | None = None,
        target_config: Mito2p5DTargetConfig | None = None,
        intensity_augment: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.z_offsets = z_offsets if z_offsets is not None else centered_z_offsets(stack_depth)
        self.stack_depth = len(self.z_offsets)
        self.target_config = target_config or Mito2p5DTargetConfig()
        self.intensity_augment = bool(intensity_augment)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        for _ in range(self.max_sample_attempts):
            pair = self.pairs[self.rng.integers(len(self.pairs))]
            sample = self._try_sample_pair_2p5d(pair)
            if sample is not None:
                return sample
        raise RuntimeError(
            "Could not sample a 2.5D mitochondria crop after "
            f"{self.max_sample_attempts} attempts"
        )

    def _try_sample_pair_2p5d(self, pair: InferenceMitoPair) -> dict[str, Any] | None:
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

        stack = self._read_raw_stack(raw_arr, pair.raw, z_world, y_world, x_world)
        label_result = self._read_label(label_arr, pair.label, z_world, y_world, x_world)
        if stack is None or label_result is None:
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

        stack = normalize_stack_percentile(
            stack,
            low_percentile=self.auto_norms_percentile_low,
            high_percentile=self.auto_norms_percentile_high,
        )
        stack = resize_stack(stack, self.output_size)
        valid_resized = resize_mask(label_valid > 0, self.output_size) > 0
        mask_resized = resize_mask(mask, self.output_size) > 0
        target_mask, confidence = self._build_target_and_confidence(
            mask_resized, valid_resized
        )

        if self.augment:
            stack, target_mask, confidence, _ = apply_stack_augment(
                self.rng,
                stack,
                target_mask,
                confidence,
                intensity=self.intensity_augment,
            )

        metadata = {
            "dataset": pair.dataset,
            "crop_id": None,
            "split": self.split,
            "raw_path": pair.raw.path,
            "label_path": pair.label.path,
            "raw_resolution_nm": list(pair.raw.resolution),
            "label_resolution_nm": list(pair.label.resolution),
            "fov_nm_yx": [self.fov_nm, self.fov_nm],
            "nm_per_px_yx": [
                self.fov_nm / float(self.output_size),
                self.fov_nm / float(self.output_size),
            ],
            "world_origin_nm_zyx": [z_world, y_world, x_world],
            "z_offsets": list(self.z_offsets),
            "center_index": self.stack_depth // 2,
            "raw_shape_yx": list(stack.shape[-2:]),
            "label_shape_yx": list(label_2d.shape),
            "mask_fraction": float(mask.mean()),
            "mask_fraction_valid": mask_fraction_valid,
            "valid_loss_fraction": valid_fraction,
            "label_quality": self.label_quality,
            "label_weight": self.label_weight,
        }
        return finalize_mito_2p5d_sample(
            stack=stack,
            mask=target_mask,
            valid_mask=confidence,
            metadata=metadata,
            target_cfg=self.target_config,
            sample_weight=self.label_weight,
        )

    def _read_raw_stack(
        self,
        arr,
        spec: ScaleSpec,
        z: float,
        y: float,
        x: float,
    ) -> np.ndarray | None:
        res = np.array(spec.resolution, dtype=float)
        off = np.array(spec.offset, dtype=float)
        shape = np.array(spec.shape, dtype=int)
        center_z = int(round((z - off[0]) / res[0]))
        y_start = int(round((y - off[1]) / res[1]))
        x_start = int(round((x - off[2]) / res[2]))
        y_size = int(round(self.fov_nm / res[1]))
        x_size = int(round(self.fov_nm / res[2]))
        if y_size <= 0 or x_size <= 0:
            return None

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


class Mito2p5DFixedFovGtDataset(FixedFovCellMapGtDataset):
    """Fixed-FOV ground-truth dataset that returns 2.5D supervised tensors."""

    def __init__(
        self,
        *args,
        stack_depth: int = 9,
        z_offsets: list[int] | None = None,
        target_config: Mito2p5DTargetConfig | None = None,
        intensity_augment: bool = True,
        split: str = "train",
        comparison_segmentation_path_template: str | None = None,
        comparison_label_name: str = "mito",
        comparison_resolution_nm: float | None = None,
        comparison_name: str = "unet",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.z_offsets = z_offsets if z_offsets is not None else centered_z_offsets(stack_depth)
        self.stack_depth = len(self.z_offsets)
        self.target_config = target_config or Mito2p5DTargetConfig()
        self.intensity_augment = bool(intensity_augment)
        self.split = split
        self.comparison_segmentation_path_template = comparison_segmentation_path_template
        self.comparison_label_name = comparison_label_name
        self.comparison_resolution_nm = comparison_resolution_nm or self.target_resolution_nm
        self.comparison_name = comparison_name
        self._comparison_scale_cache: dict[str, ScaleSpec | None] = {}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        for _ in range(self.max_sample_attempts):
            sample = self._try_sample_2p5d()
            if sample is not None:
                return sample
        sample = self._try_sample_2p5d(skip_mask_filter=True)
        if sample is None:
            raise RuntimeError("Could not sample a valid 2.5D fixed-FOV GT crop")
        return sample

    def _try_sample_2p5d(self, skip_mask_filter: bool = False) -> dict[str, Any] | None:
        crop = self.crops[self.rng.integers(len(self.crops))]
        window = self._sample_window_near_foreground(crop)
        if window is None:
            window = self._sample_window(crop)
        if window is None:
            return None
        z_world, y_world, x_world = window

        stack = self._read_gt_raw_stack(crop, z_world, y_world, x_world)
        if stack is None:
            return None
        label_result = self._read_label_window(crop, z_world, y_world, x_world)
        if label_result is None:
            return None
        mask, valid_loss_mask = label_result
        valid = (valid_loss_mask.astype(np.float32) / 255.0).clip(0.0, 1.0)
        valid_pixels = max(1, int((valid > 0).sum()))
        mask_fraction = float(mask.sum()) / float(valid_pixels)
        if not skip_mask_filter and mask_fraction < self.min_mask_fraction:
            return None

        stack = resize_stack(stack, self.target_size)
        mask = resize_mask(mask > 0, self.target_size) > 0
        valid = resize_mask(valid, self.target_size).astype(np.float32)

        comparison_mask = None
        if self.comparison_segmentation_path_template:
            raw_comparison = self._read_comparison_mask(
                crop.dataset_name, z_world, y_world, x_world
            )
            if raw_comparison is not None:
                comparison_mask = resize_mask(raw_comparison, self.target_size).astype(
                    np.float32
                )

        if self.augment:
            extra = {self.comparison_name: comparison_mask} if comparison_mask is not None else {}
            stack, mask, valid, extra = apply_stack_augment(
                self.rng,
                stack,
                mask,
                valid,
                intensity=self.intensity_augment,
                extra_masks=extra,
            )
            comparison_mask = extra.get(self.comparison_name)

        metadata = {
            "dataset": crop.dataset_name,
            "crop_id": crop.crop_id,
            "split": self.split,
            "raw_path": str(Path(crop.raw_zarr_path) / crop.raw_scale_path),
            "label_path": ",".join(
                str(Path(crop.class_info[name].zarr_path) / crop.class_info[name].scale_path)
                for name in self.fine_classes
                if name in crop.class_info
            ),
            "raw_resolution_nm": list(crop.raw_resolution),
            "fov_nm_yx": [self.fov_nm, self.fov_nm],
            "nm_per_px_yx": [
                self.fov_nm / float(self.target_size),
                self.fov_nm / float(self.target_size),
            ],
            "world_origin_nm_zyx": [z_world, y_world, x_world],
            "z_offsets": list(self.z_offsets),
            "center_index": self.stack_depth // 2,
            "mask_fraction_valid": mask_fraction,
            "valid_loss_fraction": float(valid.mean()),
            "label_quality": "gt",
            "label_weight": self.label_weight,
        }
        return finalize_mito_2p5d_sample(
            stack=stack,
            mask=mask,
            valid_mask=valid,
            metadata=metadata,
            target_cfg=self.target_config,
            sample_weight=self.label_weight,
            comparison_masks={self.comparison_name: comparison_mask}
            if comparison_mask is not None
            else None,
        )

    def _comparison_scale(self, dataset_name: str) -> ScaleSpec | None:
        if dataset_name in self._comparison_scale_cache:
            return self._comparison_scale_cache[dataset_name]
        spec = None
        base = self.comparison_segmentation_path_template.format(dataset=dataset_name).rstrip("/")
        if Path(base).name != self.comparison_label_name:
            base = str(Path(base) / self.comparison_label_name)
        if Path(base).is_dir():
            try:
                spec = _select_scale(base, self.comparison_resolution_nm, require_exact=False)
            except Exception:
                spec = None
        self._comparison_scale_cache[dataset_name] = spec
        return spec

    def _read_comparison_mask(
        self, dataset_name: str, z_world: float, y_world: float, x_world: float
    ) -> np.ndarray | None:
        """Read the existing baseline segmentation aligned to this GT crop, if available."""
        spec = self._comparison_scale(dataset_name)
        if spec is None:
            return None
        arr = self._open_zarr(spec.path)
        res = np.array(spec.resolution, dtype=float)
        off = np.array(spec.offset, dtype=float)
        shape = np.array(spec.shape, dtype=int)
        z_idx = int(round((z_world - off[0]) / res[0]))
        y_start = int(round((y_world - off[1]) / res[1]))
        x_start = int(round((x_world - off[2]) / res[2]))
        y_size = int(round(self.fov_nm / res[1]))
        x_size = int(round(self.fov_nm / res[2]))
        if z_idx < 0 or z_idx >= shape[0] or y_size <= 0 or x_size <= 0:
            return None
        y_start = max(0, min(y_start, shape[1] - y_size))
        x_start = max(0, min(x_start, shape[2] - x_size))
        y_end = min(y_start + y_size, shape[1])
        x_end = min(x_start + x_size, shape[2])
        if y_end <= y_start or x_end <= x_start:
            return None
        seg_2d = np.array(arr[z_idx, y_start:y_end, x_start:x_end])
        if seg_2d.shape != (y_end - y_start, x_end - x_start):
            return None
        return (seg_2d > 0).astype(np.float32)

    def _read_gt_raw_stack(self, crop, z_world: float, y_world: float, x_world: float):
        raw_res = np.array(crop.raw_resolution, dtype=float)
        slices = []
        for offset in self.z_offsets:
            raw_2d = self._read_raw_window(
                crop,
                z_world + float(offset) * raw_res[0],
                y_world,
                x_world,
            )
            if raw_2d is None:
                return None
            slices.append(raw_2d.astype(np.float32))
        return np.stack(slices, axis=0)


class Mito2p5DMixedDataset(Dataset):
    """Sample from pseudo and GT 2.5D datasets with a fixed GT probability."""

    def __init__(
        self,
        pseudo_dataset: Dataset,
        gt_dataset: Dataset,
        samples_per_epoch: int,
        gt_sample_prob: float = 0.25,
        seed: int = 42,
    ):
        self.pseudo_dataset = pseudo_dataset
        self.gt_dataset = gt_dataset
        self.samples_per_epoch = int(samples_per_epoch)
        self.gt_sample_prob = float(gt_sample_prob)
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if torch.rand((), generator=self.rng).item() < self.gt_sample_prob:
            gt_idx = int(torch.randint(len(self.gt_dataset), (), generator=self.rng).item())
            return self.gt_dataset[gt_idx]
        pseudo_idx = int(torch.randint(len(self.pseudo_dataset), (), generator=self.rng).item())
        return self.pseudo_dataset[pseudo_idx]


def collate_mito_2p5d(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate supervised 2.5D mitochondria samples."""
    out: dict[str, Any] = {
        "em": torch.stack([item["em"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "valid_mask": torch.stack([item["valid_mask"] for item in batch], dim=0),
        "metadata": [item.get("metadata", {}) for item in batch],
        "sample_weight": torch.tensor(
            [float(item.get("sample_weight", 1.0)) for item in batch],
            dtype=torch.float32,
        ),
    }
    if "boundary_target" in batch[0]:
        out["boundary_target"] = torch.stack(
            [item["boundary_target"] for item in batch], dim=0
        )
    if "distance_target" in batch[0]:
        out["distance_target"] = torch.stack(
            [item["distance_target"] for item in batch], dim=0
        )
    common_names: set[str] | None = None
    for item in batch:
        names = set(item.get("comparison_masks", {}))
        common_names = names if common_names is None else (common_names & names)
    if common_names:
        out["comparison_masks"] = {
            name: torch.stack([item["comparison_masks"][name] for item in batch], dim=0)
            for name in sorted(common_names)
        }
    return out


def _target_config_from_config(config: dict) -> Mito2p5DTargetConfig:
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    output_channels = int(model_cfg.get("output_channels", 1))
    return Mito2p5DTargetConfig(
        boundary_radius_px=loss_cfg.get("boundary_radius_px", 2),
        include_boundary=output_channels >= 2 or loss_cfg.get("boundary_weight", 0.0) > 0,
        include_distance=output_channels >= 3 or loss_cfg.get("distance_weight", 0.0) > 0,
        distance_max_px=loss_cfg.get("distance_max_px", 32.0),
    )


def _common_pseudo_kwargs(config: dict, split: str, samples_per_epoch: int, seed: int):
    data_cfg = config.get("data", {})
    target_cfg = _target_config_from_config(config)
    return dict(
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
        samples_per_epoch=samples_per_epoch,
        split=split,
        val_holdout_fraction=data_cfg.get("val_holdout_fraction", 0.1),
        val_holdout_axis=data_cfg.get("val_holdout_axis", "x"),
        val_holdout_position=data_cfg.get("val_holdout_position", "high"),
        fov_nm=data_cfg.get("fov_nm", 8192.0),
        fov_tolerance_nm=data_cfg.get("fov_tolerance_nm", 0.0),
        raw_target_resolution_nm=data_cfg.get("raw_target_resolution_nm"),
        label_target_resolution_nm=data_cfg.get("label_target_resolution_nm"),
        require_exact_resolution=data_cfg.get("require_exact_resolution", False),
        raw_require_exact_resolution=data_cfg.get("raw_require_exact_resolution"),
        label_require_exact_resolution=data_cfg.get("label_require_exact_resolution"),
        min_mask_fraction=data_cfg.get("min_mask_fraction", 0.01),
        min_valid_fraction=data_cfg.get("min_valid_fraction", 0.25),
        label_quality=data_cfg.get("label_quality", "pseudo"),
        label_weight=data_cfg.get("label_weight", 0.2),
        conservative_pseudo_targets=data_cfg.get("conservative_pseudo_targets", True),
        foreground_erosion_px=data_cfg.get("foreground_erosion_px", 2),
        boundary_band_px=data_cfg.get("boundary_band_px", 3),
        boundary_weight=data_cfg.get("boundary_weight", 0.0),
        max_sample_attempts=data_cfg.get("max_sample_attempts", 200),
        output_size=data_cfg.get("output_size", TARGET_SIZE),
        seed=seed,
        augment=data_cfg.get("augment", split == "train"),
        auto_norms_percentile_low=data_cfg.get("auto_norms_percentile_low", 1.0),
        auto_norms_percentile_high=data_cfg.get("auto_norms_percentile_high", 99.0),
        stack_depth=data_cfg.get("stack_depth", 9),
        z_offsets=data_cfg.get("z_offsets"),
        target_config=target_cfg,
        intensity_augment=data_cfg.get("intensity_augment", split == "train"),
    )


def _comparison_mask_kwargs(config: dict) -> dict:
    """Build kwargs to attach an aligned baseline segmentation (e.g. the existing
    UNet inference pipeline) to GT samples, for direct model-vs-baseline comparison.
    """
    data_cfg = config.get("data", {})
    comparison_cfg = data_cfg.get("comparison_masks", {}) or {}
    enabled = comparison_cfg.get("enabled", True)
    template = (
        comparison_cfg.get("segmentation_path_template", data_cfg.get("segmentation_path_template"))
        if enabled
        else None
    )
    return dict(
        comparison_segmentation_path_template=template,
        comparison_label_name=comparison_cfg.get("label_name", data_cfg.get("label_name", "mito")),
        comparison_resolution_nm=comparison_cfg.get(
            "target_resolution_nm", data_cfg.get("raw_target_resolution_nm")
        ),
        comparison_name=comparison_cfg.get("name", "unet"),
    )


def build_mito_2p5d_datasets(
    config: dict,
    gt_sample_prob_override: float | None = None,
    seed_offset: int = 0,
) -> tuple[Dataset, list[tuple[str, Dataset]]]:
    """Build train and validation datasets from a supervised 2.5D config.

    `gt_sample_prob_override`/`seed_offset` exist for curriculum staging
    (`train_mito_2p5d.py`): each stage calls this with a distinct `seed_offset`
    so the underlying pseudo/GT datasets get genuinely different RNG streams,
    rather than reusing the same already-constructed dataset objects, whose
    `self.rng` is only ever advanced inside forked DataLoader worker
    processes -- reusing them across a second DataLoader would fork fresh
    workers from the still-pristine main-process RNG state and replay stage
    1's early sampling sequence instead of continuing to explore.
    """
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42)) + int(seed_offset)
    dataset_type = data_cfg.get("dataset_type", "mixed_mito_gt_pseudo")
    train_samples = int(data_cfg.get("samples_per_epoch", 256))
    # Validation set size is independent of `num_validation_images`, which only
    # controls how many crops appear in the TensorBoard visual panel. Default it
    # to cover `max_validation_batches` so validation isn't silently truncated.
    default_val_samples = int(train_cfg.get("max_validation_batches", 8)) * int(
        train_cfg.get("validation_batch_size", train_cfg.get("batch_size", 1))
    )
    val_samples = int(train_cfg.get("num_validation_samples", default_val_samples))

    validation: list[tuple[str, Dataset]] = []
    if dataset_type in {"inference_mito_fixed_fov", "mixed_mito_gt_pseudo"}:
        pseudo_train = Mito2p5DInferenceMitoDataset(
            **_common_pseudo_kwargs(config, "train", train_samples, seed)
        )
        pseudo_val_kwargs = _common_pseudo_kwargs(
            config, "val", val_samples, seed + 100_000
        )
        pseudo_val_kwargs["augment"] = False
        pseudo_val_kwargs["intensity_augment"] = False
        pseudo_val = Mito2p5DInferenceMitoDataset(
            **pseudo_val_kwargs,
        )
        train_dataset: Dataset = pseudo_train
        validation.append(("val/pseudo", pseudo_val))

        if dataset_type == "mixed_mito_gt_pseudo":
            gt_cfg = data_cfg.get("gt", {})
            gt_val_datasets = gt_cfg.get("validation_datasets", [])
            gt_skip = list(gt_cfg.get("skip_datasets", []))
            gt_skip.extend(gt_val_datasets)
            target_cfg = _target_config_from_config(config)
            gt_common = dict(
                data_root=gt_cfg.get("data_root", data_cfg.get("data_root", "/nrs/cellmap/data")),
                norms_csv=gt_cfg.get("norms_csv", data_cfg.get("norms_csv")),
                organelle_keys=gt_cfg.get("organelles", ["mito"]),
                min_mask_fraction=gt_cfg.get("min_mask_fraction", data_cfg.get("min_mask_fraction", 0.01)),
                cache_dir=gt_cfg.get("cache_dir", data_cfg.get("cache_dir")),
                auto_norms=gt_cfg.get("auto_norms", True),
                auto_norms_per_image=gt_cfg.get("auto_norms_per_image", False),
                auto_norms_percentile_low=gt_cfg.get(
                    "auto_norms_percentile_low",
                    data_cfg.get("auto_norms_percentile_low", 1.0),
                ),
                auto_norms_percentile_high=gt_cfg.get(
                    "auto_norms_percentile_high",
                    data_cfg.get("auto_norms_percentile_high", 99.0),
                ),
                fov_nm=gt_cfg.get("fov_nm", data_cfg.get("fov_nm", 8192.0)),
                target_resolution_nm=gt_cfg.get(
                    "target_resolution_nm",
                    data_cfg.get("raw_target_resolution_nm", 8.0),
                ),
                target_size=gt_cfg.get("target_size", data_cfg.get("output_size", TARGET_SIZE)),
                raw_require_exact_resolution=gt_cfg.get("raw_require_exact_resolution", True),
                label_weight=gt_cfg.get("label_weight", 1.0),
                max_sample_attempts=gt_cfg.get(
                    "max_sample_attempts", data_cfg.get("max_sample_attempts", 200)
                ),
                stack_depth=data_cfg.get("stack_depth", 9),
                z_offsets=data_cfg.get("z_offsets"),
                target_config=target_cfg,
                intensity_augment=gt_cfg.get(
                    "intensity_augment", data_cfg.get("intensity_augment", True)
                ),
            )
            gt_train = Mito2p5DFixedFovGtDataset(
                **gt_common,
                samples_per_epoch=gt_cfg.get("samples_per_epoch", train_samples),
                skip_datasets=gt_skip,
                include_datasets=gt_cfg.get("include_datasets"),
                seed=seed + 200_000,
                augment=gt_cfg.get("augment", data_cfg.get("augment", True)),
                split="train",
            )
            train_dataset = Mito2p5DMixedDataset(
                pseudo_dataset=pseudo_train,
                gt_dataset=gt_train,
                samples_per_epoch=train_samples,
                gt_sample_prob=(
                    gt_sample_prob_override
                    if gt_sample_prob_override is not None
                    else gt_cfg.get("sample_prob", 0.25)
                ),
                seed=seed,
            )
            if gt_val_datasets:
                gt_val_common = dict(gt_common)
                gt_val_common["intensity_augment"] = False
                gt_val_common.update(_comparison_mask_kwargs(config))
                gt_val = Mito2p5DFixedFovGtDataset(
                    **gt_val_common,
                    samples_per_epoch=val_samples,
                    skip_datasets=gt_cfg.get("skip_datasets", []),
                    include_datasets=gt_val_datasets,
                    seed=seed + 300_000,
                    augment=False,
                    split="val",
                )
                validation.append(("val/gt", gt_val))
        return train_dataset, validation

    if dataset_type == "fixed_fov_gt":
        gt_cfg = data_cfg.get("gt", data_cfg)
        gt_val_datasets = gt_cfg.get("validation_datasets", [])
        if not gt_val_datasets:
            raise ValueError(
                "dataset_type=fixed_fov_gt requires data.gt.validation_datasets "
                "(or data.validation_datasets) to hold out whole datasets for "
                "validation; without it train/val would sample the same crops."
            )
        gt_skip = list(gt_cfg.get("skip_datasets", []))
        gt_skip.extend(gt_val_datasets)
        target_cfg = _target_config_from_config(config)
        common = dict(
            data_root=gt_cfg.get("data_root", data_cfg.get("data_root", "/nrs/cellmap/data")),
            norms_csv=gt_cfg.get("norms_csv", data_cfg.get("norms_csv")),
            organelle_keys=gt_cfg.get("organelles", ["mito"]),
            min_mask_fraction=gt_cfg.get("min_mask_fraction", data_cfg.get("min_mask_fraction", 0.01)),
            cache_dir=gt_cfg.get("cache_dir", data_cfg.get("cache_dir")),
            auto_norms=gt_cfg.get("auto_norms", True),
            auto_norms_per_image=gt_cfg.get("auto_norms_per_image", False),
            auto_norms_percentile_low=gt_cfg.get(
                "auto_norms_percentile_low", data_cfg.get("auto_norms_percentile_low", 1.0)
            ),
            auto_norms_percentile_high=gt_cfg.get(
                "auto_norms_percentile_high", data_cfg.get("auto_norms_percentile_high", 99.0)
            ),
            fov_nm=gt_cfg.get("fov_nm", data_cfg.get("fov_nm", 8192.0)),
            target_resolution_nm=gt_cfg.get(
                "target_resolution_nm", data_cfg.get("raw_target_resolution_nm", 8.0)
            ),
            target_size=gt_cfg.get("target_size", data_cfg.get("output_size", TARGET_SIZE)),
            raw_require_exact_resolution=gt_cfg.get("raw_require_exact_resolution", True),
            label_weight=gt_cfg.get("label_weight", 1.0),
            max_sample_attempts=gt_cfg.get("max_sample_attempts", data_cfg.get("max_sample_attempts", 200)),
            stack_depth=data_cfg.get("stack_depth", 9),
            z_offsets=data_cfg.get("z_offsets"),
            target_config=target_cfg,
            intensity_augment=gt_cfg.get("intensity_augment", data_cfg.get("intensity_augment", True)),
        )
        train_dataset = Mito2p5DFixedFovGtDataset(
            **common,
            samples_per_epoch=train_samples,
            skip_datasets=gt_skip,
            include_datasets=gt_cfg.get("include_datasets"),
            seed=seed,
            augment=gt_cfg.get("augment", data_cfg.get("augment", True)),
            split="train",
        )
        val_common = dict(common)
        val_common["intensity_augment"] = False
        val_common.update(_comparison_mask_kwargs(config))
        val_dataset = Mito2p5DFixedFovGtDataset(
            **val_common,
            samples_per_epoch=val_samples,
            skip_datasets=gt_cfg.get("skip_datasets", []),
            include_datasets=gt_val_datasets,
            seed=seed + 100_000,
            augment=False,
            split="val",
        )
        return train_dataset, [("val/gt", val_dataset)]

    raise ValueError(f"Unknown 2.5D dataset_type={dataset_type!r}")
