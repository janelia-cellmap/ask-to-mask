"""Training dataset: produces (conditioning_image, target_image, prompt) triplets."""

from __future__ import annotations

import logging
import os

import numpy as np
import zarr
from PIL import Image
from scipy.ndimage import zoom as ndimage_zoom
from torch.utils.data import Dataset

from ..config import ORGANELLE_FINE_CLASSES, ORGANELLES, OrganelleClass, build_multi_organelle_prompt
from .zarr_utils import (
    CropInfo,
    compute_auto_norms,
    discover_crops,
    find_scale_for_resolution,
    load_norms,
    normalize_raw,
)

logger = logging.getLogger(__name__)

TARGET_SIZE = 1024

# Threshold in YX voxels: crops smaller than this are read in full and resized;
# larger volumes get random sub-crops.
SMALL_CROP_THRESHOLD = 1024


class CellMapFluxDataset(Dataset):
    """Zarr-backed dataset for Flux LoRA training.

    Each sample:
    1. Picks a crop and organelle class (class-balanced).
    2. Picks a random Z-slice from the crop.
    3. Reads 2D raw EM + label slices from zarr.
    4. Creates a target image with organelle pixels colored.
    5. Returns (cond_pil, target_pil, prompt_str).
    """

    def __init__(
        self,
        data_root: str = "/nrs/cellmap/data",
        norms_csv: str | None = None,
        organelle_keys: list[str] | None = None,
        samples_per_epoch: int = 2000,
        min_mask_fraction: float = 0.01,
        skip_datasets: list[str] | None = None,
        include_datasets: list[str] | None = None,
        cache_dir: str | None = None,
        seed: int = 42,
        augment: bool = True,
        target_mode: str = "overlay",
        include_resolution: bool = False,
        auto_norms: bool = False,
        auto_norms_per_image: bool = False,
        auto_norms_percentile_low: float = 1.0,
        auto_norms_percentile_high: float = 99.0,
        multi_organelle_prob: float = 0.0,
        negative_example_prob: float = 0.0,
        prompt_variation: bool = False,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.min_mask_fraction = min_mask_fraction
        self.target_mode = target_mode
        self.include_resolution = include_resolution
        self.auto_norms_per_image = auto_norms_per_image
        self.auto_norms_percentile_low = auto_norms_percentile_low
        self.auto_norms_percentile_high = auto_norms_percentile_high
        self.multi_organelle_prob = multi_organelle_prob
        self.negative_example_prob = negative_example_prob
        self.prompt_variation = prompt_variation
        self.rng = np.random.default_rng(seed)
        self.augment = augment

        # Resolve organelles
        if organelle_keys is None:
            organelle_keys = list(ORGANELLES.keys())
        self.organelle_keys = [
            k for k in organelle_keys if ORGANELLE_FINE_CLASSES.get(k)
        ]
        if not self.organelle_keys:
            raise ValueError(
                "No valid organelles with fine-class mappings. "
                f"Requested: {organelle_keys}"
            )

        # Collect all fine classes needed
        all_fine_classes = set()
        for key in self.organelle_keys:
            all_fine_classes.update(ORGANELLE_FINE_CLASSES[key])

        # Load norms
        norms = {}
        if norms_csv is not None:
            norms = load_norms(norms_csv)

        # Discover crops
        self.crops = discover_crops(
            data_root=data_root,
            target_classes=sorted(all_fine_classes),
            norms=norms,
            skip_datasets=skip_datasets,
            include_datasets=include_datasets,
            cache_dir=cache_dir,
        )
        if not self.crops:
            raise RuntimeError(f"No crops found in {data_root}")

        # Optionally compute auto norms from data percentiles
        if auto_norms:
            logger.info(
                f"Computing auto norms (p{auto_norms_percentile_low}"
                f"-p{auto_norms_percentile_high})..."
            )
            auto = compute_auto_norms(
                self.crops,
                percentile_low=auto_norms_percentile_low,
                percentile_high=auto_norms_percentile_high,
            )
            for crop in self.crops:
                if crop.dataset_name in auto:
                    crop.norm_params = auto[crop.dataset_name]

        # Build organelle -> list of crops that have at least one fine class
        self.organelle_crops: dict[str, list[CropInfo]] = {}
        for key in self.organelle_keys:
            fine_classes = set(ORGANELLE_FINE_CLASSES[key])
            matching = [
                c for c in self.crops if c.annotated_classes & fine_classes
            ]
            if matching:
                self.organelle_crops[key] = matching

        # Remove organelles with no crops
        self.organelle_keys = [
            k for k in self.organelle_keys if k in self.organelle_crops
        ]
        if not self.organelle_keys:
            raise RuntimeError("No organelles have matching annotated crops")

        # Build reverse index: crop -> list of organelle keys present
        self._crop_organelles: dict[str, list[str]] = {}
        for key, crops_list in self.organelle_crops.items():
            for c in crops_list:
                crop_id = f"{c.dataset_name}:{c.crop_id}"
                self._crop_organelles.setdefault(crop_id, []).append(key)

        # Class-balanced sampling state
        self._class_counts = {k: 0 for k in self.organelle_keys}

        logger.info(
            f"CellMapFluxDataset: {len(self.crops)} crops, "
            f"{len(self.organelle_keys)} organelles: {self.organelle_keys}"
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image, str]:
        """Return (conditioning_image, target_image, prompt)."""
        max_retries = 50
        for _ in range(max_retries):
            result = self._try_sample()
            if result is not None:
                return result

        # Fallback: return a random valid sample without mask fraction filter
        return self._try_sample(skip_mask_filter=True)

    def _pick_organelle(self) -> str:
        """Pick the least-seen organelle (class-balanced)."""
        min_count = min(self._class_counts.values())
        candidates = [
            k for k, v in self._class_counts.items() if v == min_count
        ]
        key = candidates[self.rng.integers(len(candidates))]
        self._class_counts[key] += 1
        return key

    def _get_crop_geometry(self, crop: CropInfo):
        """Compute crop geometry for slice reading.

        Returns (is_small, z_extent_vox, subcrop_origin_world) or None.
        """
        crop_extent = np.array(crop.crop_extent_world)
        crop_origin = np.array(crop.crop_origin_world)
        raw_res = np.array(crop.raw_resolution)
        crop_yx_voxels = crop_extent[1:] / raw_res[1:]

        is_small = np.all(crop_yx_voxels < SMALL_CROP_THRESHOLD)
        z_extent_vox = int(crop_extent[0] / raw_res[0])
        if z_extent_vox < 1:
            return None

        subcrop_origin_world = None
        if not is_small:
            world_extent_yx = TARGET_SIZE * raw_res[1:]
            y_range = crop_extent[1] - world_extent_yx[0]
            x_range = crop_extent[2] - world_extent_yx[1]
            if y_range > 0 and x_range > 0:
                y_offset = self.rng.uniform(0, y_range)
                x_offset = self.rng.uniform(0, x_range)
                subcrop_origin_world = np.array([
                    crop_origin[1] + y_offset,
                    crop_origin[2] + x_offset,
                ])
            else:
                subcrop_origin_world = np.array([
                    crop_origin[1], crop_origin[2]
                ])

        return is_small, z_extent_vox, subcrop_origin_world

    def _try_sample(
        self, skip_mask_filter: bool = False
    ) -> tuple[Image.Image, Image.Image, str] | None:
        """Try to produce one valid sample, or return None."""
        organelle_key = self._pick_organelle()
        organelle = ORGANELLES[organelle_key]
        fine_classes = ORGANELLE_FINE_CLASSES[organelle_key]
        crops = self.organelle_crops[organelle_key]
        crop = crops[self.rng.integers(len(crops))]

        geom = self._get_crop_geometry(crop)
        if geom is None:
            return None
        is_small, z_extent_vox, subcrop_origin_world = geom
        z_idx_in_crop = self.rng.integers(z_extent_vox)

        # Read raw 2D slice
        raw_slice = self._read_raw_slice(
            crop, z_idx_in_crop, is_small, subcrop_origin_world
        )
        if raw_slice is None:
            return None

        # Check if this should be a negative example (no coloring)
        is_negative = (
            self.negative_example_prob > 0
            and self.rng.random() < self.negative_example_prob
        )

        # Build images
        raw_uint8 = (raw_slice * 255).astype(np.uint8)
        raw_rgb = np.stack([raw_uint8] * 3, axis=-1)  # [H, W, 3]

        if is_negative:
            # Negative example: target = input, no coloring
            if self.target_mode == "segmentation":
                target_rgb = np.zeros_like(raw_rgb)
            else:
                target_rgb = raw_rgb.copy()
            organelles_used = [organelle]
        else:
            # Determine which organelles to color
            organelles_to_color = [(organelle_key, organelle, fine_classes)]

            # Maybe add more organelles from the same crop
            if self.multi_organelle_prob > 0 and self.rng.random() < self.multi_organelle_prob:
                crop_id = f"{crop.dataset_name}:{crop.crop_id}"
                available = [
                    k for k in self._crop_organelles.get(crop_id, [])
                    if k != organelle_key
                ]
                if available:
                    n_extra = min(self.rng.integers(1, 3), len(available))
                    extras = self.rng.choice(available, size=n_extra, replace=False)
                    for ek in extras:
                        organelles_to_color.append(
                            (ek, ORGANELLES[ek], ORGANELLE_FINE_CLASSES[ek])
                        )

            # Create target image
            if self.target_mode == "segmentation":
                target_rgb = np.zeros_like(raw_rgb)
            else:
                target_rgb = raw_rgb.copy()

            has_any_mask = False
            organelles_used = []
            for org_key, org, fc in organelles_to_color:
                mask = self._read_label_slice(
                    crop, fc, z_idx_in_crop, raw_slice.shape,
                    is_small, subcrop_origin_world,
                )
                if mask is not None:
                    mask_fraction = mask.sum() / mask.size
                    if skip_mask_filter or mask_fraction >= self.min_mask_fraction:
                        color = np.array(org.rgb, dtype=np.uint8)
                        target_rgb[mask > 0] = color
                        has_any_mask = True
                        organelles_used.append(org)

            if not has_any_mask:
                return None

        # Apply augmentation (same transform to both)
        if self.augment:
            raw_rgb, target_rgb = self._augment(raw_rgb, target_rgb)

        # Pad to square and resize to TARGET_SIZE
        cond_pil = self._to_square_pil(raw_rgb)
        target_pil = self._to_square_pil(target_rgb)

        # Use YX resolution (average of Y and X) as nm/px
        resolution_nm = None
        if self.include_resolution:
            raw_res = crop.raw_resolution
            resolution_nm = (raw_res[1] + raw_res[2]) / 2.0

        # Build prompt
        if len(organelles_used) == 1:
            if self.prompt_variation:
                prompt = organelles_used[0].build_prompt_varied(
                    rng=self.rng, resolution_nm=resolution_nm,
                )
            else:
                prompt = organelles_used[0].build_prompt(resolution_nm=resolution_nm)
        else:
            prompt = build_multi_organelle_prompt(
                organelles_used, resolution_nm=resolution_nm,
                rng=self.rng if self.prompt_variation else None,
            )
        return cond_pil, target_pil, prompt

    def _read_raw_slice(
        self,
        crop: CropInfo,
        z_idx_in_crop: int,
        is_small: bool,
        subcrop_origin_world: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Read a 2D YX raw slice from the crop.

        Args:
            crop: Crop metadata.
            z_idx_in_crop: Z-slice index relative to crop start.
            is_small: Whether to read the full crop (True) or sub-crop (False).
            subcrop_origin_world: [y, x] world coords for sub-crop origin.

        Returns normalized float32 [H, W] slice, or None on error.
        """
        try:
            raw_arr = zarr.open(
                os.path.join(crop.raw_zarr_path, crop.raw_scale_path), mode="r"
            )
        except Exception:
            return None

        raw_off = np.array(crop.raw_offset_world)
        raw_res = np.array(crop.raw_resolution)
        raw_shape = np.array(crop.raw_shape)
        crop_origin = np.array(crop.crop_origin_world)

        # Z index in raw volume
        z_world = crop_origin[0] + z_idx_in_crop * raw_res[0]
        z_vox = int(round((z_world - raw_off[0]) / raw_res[0]))
        z_vox = max(0, min(z_vox, raw_shape[0] - 1))

        if is_small:
            # Read full crop YX extent
            y_start = int(round((crop_origin[1] - raw_off[1]) / raw_res[1]))
            x_start = int(round((crop_origin[2] - raw_off[2]) / raw_res[2]))
            crop_extent = np.array(crop.crop_extent_world)
            y_size = int(round(crop_extent[1] / raw_res[1]))
            x_size = int(round(crop_extent[2] / raw_res[2]))

            y_start = max(0, min(y_start, raw_shape[1] - 1))
            x_start = max(0, min(x_start, raw_shape[2] - 1))
            y_end = min(y_start + y_size, raw_shape[1])
            x_end = min(x_start + x_size, raw_shape[2])
        else:
            # Read TARGET_SIZE pixels at native resolution
            y_size_vox = TARGET_SIZE
            x_size_vox = TARGET_SIZE

            y_start = int(
                round((subcrop_origin_world[0] - raw_off[1]) / raw_res[1])
            )
            x_start = int(
                round((subcrop_origin_world[1] - raw_off[2]) / raw_res[2])
            )
            y_start = max(0, min(y_start, raw_shape[1] - y_size_vox))
            x_start = max(0, min(x_start, raw_shape[2] - x_size_vox))
            y_end = min(y_start + y_size_vox, raw_shape[1])
            x_end = min(x_start + x_size_vox, raw_shape[2])

        raw_2d = np.array(raw_arr[z_vox, y_start:y_end, x_start:x_end])
        if self.auto_norms_per_image:
            from .zarr_utils import NormParams
            p_low = float(np.percentile(raw_2d, self.auto_norms_percentile_low))
            p_high = float(np.percentile(raw_2d, self.auto_norms_percentile_high))
            raw_2d = normalize_raw(raw_2d, NormParams(p_low, p_high, False))
        else:
            raw_2d = normalize_raw(raw_2d, crop.norm_params)
        return raw_2d

    def _read_label_slice(
        self,
        crop: CropInfo,
        fine_classes: list[str],
        z_idx_in_crop: int,
        raw_yx_shape: tuple[int, int],
        is_small: bool,
        subcrop_origin_world: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Read and union label slices for the given fine classes.

        Args:
            crop: Crop metadata.
            fine_classes: List of fine-class names to union.
            z_idx_in_crop: Z-slice index relative to crop start.
            raw_yx_shape: Target (H, W) shape to match.
            is_small: Whether reading full crop or sub-crop.
            subcrop_origin_world: [y, x] world coords for sub-crop origin.

        Returns binary mask matching raw_yx_shape, or None if empty.
        """
        mask = np.zeros(raw_yx_shape, dtype=np.uint8)

        crop_origin = np.array(crop.crop_origin_world)
        raw_res = np.array(crop.raw_resolution)

        for cls_name in fine_classes:
            if cls_name not in crop.class_info:
                continue

            ci = crop.class_info[cls_name]
            try:
                label_arr = zarr.open(
                    os.path.join(ci.zarr_path, ci.scale_path), mode="r"
                )
            except Exception:
                continue

            label_res = np.array(ci.resolution)
            label_off = np.array(ci.offset_world)
            label_shape = np.array(label_arr.shape)

            # Z index in label volume
            z_world = crop_origin[0] + z_idx_in_crop * raw_res[0]
            z_vox = int(round((z_world - label_off[0]) / label_res[0]))
            z_vox = max(0, min(z_vox, label_shape[0] - 1))

            if is_small:
                y_start = int(
                    round((crop_origin[1] - label_off[1]) / label_res[1])
                )
                x_start = int(
                    round((crop_origin[2] - label_off[2]) / label_res[2])
                )
                crop_extent = np.array(crop.crop_extent_world)
                y_size = int(round(crop_extent[1] / label_res[1]))
                x_size = int(round(crop_extent[2] / label_res[2]))

                y_start = max(0, min(y_start, label_shape[1] - 1))
                x_start = max(0, min(x_start, label_shape[2] - 1))
                y_end = min(y_start + y_size, label_shape[1])
                x_end = min(x_start + x_size, label_shape[2])
            else:
                # Use crop's raw resolution to determine world extent,
                # then convert to label voxels
                raw_res_y = crop.raw_resolution[1]
                raw_res_x = crop.raw_resolution[2]
                world_extent_y = TARGET_SIZE * raw_res_y
                world_extent_x = TARGET_SIZE * raw_res_x
                y_size_vox = int(round(world_extent_y / label_res[1]))
                x_size_vox = int(round(world_extent_x / label_res[2]))

                y_start = int(
                    round(
                        (subcrop_origin_world[0] - label_off[1]) / label_res[1]
                    )
                )
                x_start = int(
                    round(
                        (subcrop_origin_world[1] - label_off[2]) / label_res[2]
                    )
                )
                y_start = max(0, min(y_start, label_shape[1] - y_size_vox))
                x_start = max(0, min(x_start, label_shape[2] - x_size_vox))
                y_end = min(y_start + y_size_vox, label_shape[1])
                x_end = min(x_start + x_size_vox, label_shape[2])

            label_2d = np.array(label_arr[z_vox, y_start:y_end, x_start:x_end])
            binary = ((label_2d > 0) & (label_2d != 255)).astype(np.uint8)

            # Resize label to match raw if needed
            if binary.shape != raw_yx_shape:
                zoom_y = raw_yx_shape[0] / binary.shape[0]
                zoom_x = raw_yx_shape[1] / binary.shape[1]
                binary = ndimage_zoom(binary, (zoom_y, zoom_x), order=0)
                binary = binary[: raw_yx_shape[0], : raw_yx_shape[1]]

            mask = np.maximum(mask, binary)

        if mask.sum() == 0:
            return None
        return mask

    def _augment(
        self, raw: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random spatial and intensity augmentations."""
        # --- Spatial augmentations (applied to both identically) ---
        # Random horizontal flip
        if self.rng.random() > 0.5:
            raw = np.flip(raw, axis=1).copy()
            target = np.flip(target, axis=1).copy()

        # Random vertical flip
        if self.rng.random() > 0.5:
            raw = np.flip(raw, axis=0).copy()
            target = np.flip(target, axis=0).copy()

        # Random 90-degree rotation
        k = self.rng.integers(4)
        if k > 0:
            raw = np.rot90(raw, k, axes=(0, 1)).copy()
            target = np.rot90(target, k, axes=(0, 1)).copy()

        # --- Intensity augmentations (EM pixels only) ---
        # Sample random params once so raw and target get the same transform
        intensity_params = self._sample_intensity_params()
        raw = self._apply_intensity(raw, intensity_params)
        if self.target_mode == "overlay":
            # For overlay targets, apply same intensity transform to EM
            # background but preserve colored organelle pixels.
            gray_mask = (
                (target[:, :, 0] == target[:, :, 1])
                & (target[:, :, 1] == target[:, :, 2])
            )
            target_aug = self._apply_intensity(target, intensity_params)
            target[gray_mask] = target_aug[gray_mask]

        return raw, target

    def _sample_intensity_params(self) -> dict:
        """Sample random intensity augmentation parameters."""
        return {
            "brightness": float(self.rng.uniform(-20, 20)),
            "contrast": float(self.rng.uniform(0.8, 1.2)),
            "gamma": float(self.rng.uniform(0.8, 1.2)),
            "noise_sigma": float(self.rng.uniform(0, 5)),
        }

    def _apply_intensity(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Apply intensity augmentations with pre-sampled parameters."""
        img = img.astype(np.float32)
        img = img + params["brightness"]
        mean = img.mean()
        img = (img - mean) * params["contrast"] + mean
        img = np.clip(img, 0, 255)
        img = 255.0 * (img / 255.0) ** params["gamma"]
        if params["noise_sigma"] > 0:
            noise = self.rng.normal(0, params["noise_sigma"], img.shape)
            img = img + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    def _to_square_pil(self, img_rgb: np.ndarray) -> Image.Image:
        """Pad to square and resize to TARGET_SIZE x TARGET_SIZE."""
        pil = Image.fromarray(img_rgb)
        w, h = pil.size
        size = max(w, h)
        if w != size or h != size:
            padded = Image.new("RGB", (size, size), (0, 0, 0))
            left = (size - w) // 2
            top = (size - h) // 2
            padded.paste(pil, (left, top))
            pil = padded
        if pil.size != (TARGET_SIZE, TARGET_SIZE):
            pil = pil.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        return pil


class FixedFovCellMapGtDataset(Dataset):
    """Ground-truth mitochondria crops sampled at one fixed physical FOV."""

    def __init__(
        self,
        data_root: str = "/nrs/cellmap/data",
        norms_csv: str | None = None,
        organelle_keys: list[str] | None = None,
        samples_per_epoch: int = 2000,
        min_mask_fraction: float = 0.01,
        skip_datasets: list[str] | None = None,
        include_datasets: list[str] | None = None,
        cache_dir: str | None = None,
        seed: int = 42,
        augment: bool = True,
        auto_norms: bool = False,
        auto_norms_per_image: bool = False,
        auto_norms_percentile_low: float = 1.0,
        auto_norms_percentile_high: float = 99.0,
        fov_nm: float = 8192.0,
        target_resolution_nm: float = 8.0,
        target_size: int | None = None,
        raw_require_exact_resolution: bool = True,
        label_weight: float = 1.0,
        background_weight: float = 0.05,
        prompt: str = "CLASS=mitochondria; OUTPUT=red_on_black",
        max_sample_attempts: int = 100,
    ):
        self.samples_per_epoch = int(samples_per_epoch)
        self.min_mask_fraction = float(min_mask_fraction)
        self.rng = np.random.default_rng(seed)
        self.augment = bool(augment)
        self.target_mode = "segmentation"
        self.auto_norms_per_image = bool(auto_norms_per_image)
        self.auto_norms_percentile_low = float(auto_norms_percentile_low)
        self.auto_norms_percentile_high = float(auto_norms_percentile_high)
        self.fov_nm = float(fov_nm)
        self.target_resolution_nm = float(target_resolution_nm)
        self.target_size = int(target_size or round(self.fov_nm / self.target_resolution_nm))
        self.raw_require_exact_resolution = bool(raw_require_exact_resolution)
        self.label_weight = float(label_weight)
        self.background_weight = float(background_weight)
        self.prompt = prompt
        self.max_sample_attempts = int(max_sample_attempts)
        self._zarr_cache: dict[str, object] = {}

        expected_fov = self.target_size * self.target_resolution_nm
        if not np.isclose(expected_fov, self.fov_nm, atol=1e-3):
            raise ValueError(
                f"fov_nm={self.fov_nm} must equal target_size * target_resolution_nm "
                f"({self.target_size} * {self.target_resolution_nm} = {expected_fov})"
            )

        if organelle_keys is None:
            organelle_keys = ["mito"]
        organelle_keys = [k for k in organelle_keys if ORGANELLE_FINE_CLASSES.get(k)]
        if organelle_keys != ["mito"]:
            raise ValueError("Fixed-FOV GT training is currently mitochondria-only")
        self.fine_classes = ORGANELLE_FINE_CLASSES["mito"]
        self.organelle = ORGANELLES["mito"]

        norms = load_norms(norms_csv) if norms_csv is not None else {}
        crops = discover_crops(
            data_root=data_root,
            target_classes=sorted(self.fine_classes),
            norms=norms,
            skip_datasets=skip_datasets,
            include_datasets=include_datasets,
            cache_dir=cache_dir,
        )
        if not crops:
            raise RuntimeError(f"No GT crops found in {data_root}")

        self.crops = self._filter_crops(crops)
        if not self.crops:
            raise RuntimeError(
                "No GT crops remain after fixed-FOV filtering. "
                f"Need raw YX resolution {self.target_resolution_nm} nm and "
                f"YX extent >= {self.fov_nm} nm."
            )

        if auto_norms:
            logger.info(
                f"Computing fixed-FOV GT auto norms (p{auto_norms_percentile_low}"
                f"-p{auto_norms_percentile_high})..."
            )
            auto = compute_auto_norms(
                self.crops,
                percentile_low=auto_norms_percentile_low,
                percentile_high=auto_norms_percentile_high,
            )
            for crop in self.crops:
                if crop.dataset_name in auto:
                    crop.norm_params = auto[crop.dataset_name]

        logger.info(
            "FixedFovCellMapGtDataset: %s crops, target=%spx, resolution=%s nm/px, "
            "FOV=%s nm, datasets=%s",
            len(self.crops),
            self.target_size,
            self.target_resolution_nm,
            self.fov_nm,
            sorted({c.dataset_name for c in self.crops}),
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        for _ in range(self.max_sample_attempts):
            result = self._try_sample()
            if result is not None:
                return result
        result = self._try_sample(skip_mask_filter=True)
        if result is None:
            raise RuntimeError("Could not sample a valid fixed-FOV GT crop")
        return result

    def _filter_crops(self, crops: list[CropInfo]) -> list[CropInfo]:
        filtered = []
        for crop in crops:
            scale_info = find_scale_for_resolution(
                crop.raw_zarr_path,
                self.target_resolution_nm,
                max_ratio=1.01 if self.raw_require_exact_resolution else 2.0,
            )
            if scale_info is None:
                continue
            raw_scale_path, raw_res, raw_off, raw_shape = scale_info
            crop.raw_scale_path = raw_scale_path
            crop.raw_resolution = raw_res
            crop.raw_offset_world = raw_off
            crop.raw_shape = raw_shape

            raw_res = np.array(crop.raw_resolution, dtype=float)
            crop_extent = np.array(crop.crop_extent_world, dtype=float)
            if self.raw_require_exact_resolution and not np.allclose(
                raw_res[1:], self.target_resolution_nm, atol=1e-3
            ):
                continue
            if crop_extent[0] < raw_res[0]:
                continue
            if not any(cls_name in crop.class_info for cls_name in self.fine_classes):
                continue
            filtered.append(crop)
        return filtered

    def _open_zarr(self, path: str):
        arr = self._zarr_cache.get(path)
        if arr is None:
            arr = zarr.open(path, mode="r")
            self._zarr_cache[path] = arr
        return arr

    def _sample_window(self, crop: CropInfo) -> tuple[float, float, float] | None:
        raw_res = np.array(crop.raw_resolution, dtype=float)
        raw_off = np.array(crop.raw_offset_world, dtype=float)
        raw_shape = np.array(crop.raw_shape, dtype=float)
        crop_origin = np.array(crop.crop_origin_world, dtype=float)
        crop_extent = np.array(crop.crop_extent_world, dtype=float)

        z_max = crop_origin[0] + crop_extent[0] - raw_res[0]
        if z_max < crop_origin[0]:
            return None
        z_world = float(self.rng.uniform(crop_origin[0], z_max + 1e-6))

        y_min = raw_off[1]
        x_min = raw_off[2]
        y_max = raw_off[1] + raw_shape[1] * raw_res[1] - self.fov_nm
        x_max = raw_off[2] + raw_shape[2] * raw_res[2] - self.fov_nm
        if y_max < y_min or x_max < x_min:
            return None
        crop_y0 = np.clip(crop_origin[1], y_min, y_max)
        crop_x0 = np.clip(crop_origin[2], x_min, x_max)
        crop_y1 = np.clip(crop_origin[1] + max(0.0, crop_extent[1] - self.fov_nm), y_min, y_max)
        crop_x1 = np.clip(crop_origin[2] + max(0.0, crop_extent[2] - self.fov_nm), x_min, x_max)
        y_world = float(self.rng.uniform(min(crop_y0, crop_y1), max(crop_y0, crop_y1) + 1e-6))
        x_world = float(self.rng.uniform(min(crop_x0, crop_x1), max(crop_x0, crop_x1) + 1e-6))
        return z_world, y_world, x_world

    def _try_sample(self, skip_mask_filter: bool = False):
        crop = self.crops[self.rng.integers(len(self.crops))]
        window = self._sample_window_near_foreground(crop)
        if window is None:
            window = self._sample_window(crop)
        if window is None:
            return None
        z_world, y_world, x_world = window

        raw_slice = self._read_raw_window(crop, z_world, y_world, x_world)
        if raw_slice is None:
            return None
        label_result = self._read_label_window(crop, z_world, y_world, x_world)
        if label_result is None:
            return None
        mask, valid_loss_mask = label_result
        valid_pixels = max(1, int((valid_loss_mask > 0).sum()))
        mask_fraction = float(mask.sum()) / float(valid_pixels)
        if not skip_mask_filter and mask_fraction < self.min_mask_fraction:
            return None

        raw_uint8 = (raw_slice * 255).clip(0, 255).astype(np.uint8)
        raw_rgb = np.stack([raw_uint8] * 3, axis=-1)
        target_rgb = np.zeros_like(raw_rgb)
        target_rgb[mask > 0] = np.array(self.organelle.rgb, dtype=np.uint8)

        if self.augment:
            raw_rgb, target_rgb = CellMapFluxDataset._augment(self, raw_rgb, target_rgb)

        cond_pil = Image.fromarray(raw_rgb).resize(
            (self.target_size, self.target_size), Image.LANCZOS
        )
        target_pil = Image.fromarray(target_rgb).resize(
            (self.target_size, self.target_size), Image.NEAREST
        )
        loss_weight = (valid_loss_mask > 0).astype(np.float32) * self.background_weight
        loss_weight[mask > 0] = 1.0
        valid_mask = Image.fromarray(
            np.clip(loss_weight * 255.0, 0, 255).astype(np.uint8), mode="L"
        )
        return cond_pil, target_pil, self.prompt, valid_mask, self.label_weight

    def _sample_window_near_foreground(
        self,
        crop: CropInfo,
        max_tries: int = 25,
    ) -> tuple[float, float, float] | None:
        raw_res = np.array(crop.raw_resolution, dtype=float)
        raw_off = np.array(crop.raw_offset_world, dtype=float)
        raw_shape = np.array(crop.raw_shape, dtype=float)
        y_min = raw_off[1]
        x_min = raw_off[2]
        y_max = raw_off[1] + raw_shape[1] * raw_res[1] - self.fov_nm
        x_max = raw_off[2] + raw_shape[2] * raw_res[2] - self.fov_nm
        if y_max < y_min or x_max < x_min:
            return None

        class_names = [c for c in self.fine_classes if c in crop.class_info]
        if not class_names:
            return None

        for _ in range(max_tries):
            cls_name = class_names[self.rng.integers(len(class_names))]
            ci = crop.class_info[cls_name]
            try:
                label_arr = self._open_zarr(os.path.join(ci.zarr_path, ci.scale_path))
            except Exception:
                continue

            label_res = np.array(ci.resolution, dtype=float)
            label_off = np.array(ci.offset_world, dtype=float)
            label_shape = np.array(ci.shape, dtype=int)
            z_vox = int(self.rng.integers(max(1, label_shape[0])))
            label_2d = np.array(label_arr[z_vox])
            ys, xs = np.nonzero((label_2d > 0) & (label_2d != 255))
            if len(ys) == 0:
                continue

            pick = int(self.rng.integers(len(ys)))
            fg_y_world = label_off[1] + float(ys[pick]) * label_res[1]
            fg_x_world = label_off[2] + float(xs[pick]) * label_res[2]
            y_world = fg_y_world - float(self.rng.uniform(0, self.fov_nm))
            x_world = fg_x_world - float(self.rng.uniform(0, self.fov_nm))
            y_world = float(np.clip(y_world, y_min, y_max))
            x_world = float(np.clip(x_world, x_min, x_max))
            z_world = label_off[0] + float(z_vox) * label_res[0]
            return z_world, y_world, x_world

        return None

    def _read_raw_window(
        self,
        crop: CropInfo,
        z_world: float,
        y_world: float,
        x_world: float,
    ) -> np.ndarray | None:
        try:
            raw_arr = self._open_zarr(os.path.join(crop.raw_zarr_path, crop.raw_scale_path))
        except Exception:
            return None

        raw_res = np.array(crop.raw_resolution, dtype=float)
        raw_off = np.array(crop.raw_offset_world, dtype=float)
        raw_shape = np.array(crop.raw_shape, dtype=int)
        y_size = int(round(self.fov_nm / raw_res[1]))
        x_size = int(round(self.fov_nm / raw_res[2]))
        if y_size <= 0 or x_size <= 0:
            return None

        z_vox = int(round((z_world - raw_off[0]) / raw_res[0]))
        y_start = int(round((y_world - raw_off[1]) / raw_res[1]))
        x_start = int(round((x_world - raw_off[2]) / raw_res[2]))
        z_vox = max(0, min(z_vox, raw_shape[0] - 1))
        y_start = max(0, min(y_start, raw_shape[1] - y_size))
        x_start = max(0, min(x_start, raw_shape[2] - x_size))
        y_end = y_start + y_size
        x_end = x_start + x_size
        if y_end > raw_shape[1] or x_end > raw_shape[2]:
            return None

        raw_2d = np.array(raw_arr[z_vox, y_start:y_end, x_start:x_end])
        if raw_2d.shape != (y_size, x_size):
            return None
        if self.auto_norms_per_image:
            from .zarr_utils import NormParams

            p_low = float(np.percentile(raw_2d, self.auto_norms_percentile_low))
            p_high = float(np.percentile(raw_2d, self.auto_norms_percentile_high))
            return normalize_raw(raw_2d, NormParams(p_low, p_high, False))
        return normalize_raw(raw_2d, crop.norm_params)

    def _read_label_window(
        self,
        crop: CropInfo,
        z_world: float,
        y_world: float,
        x_world: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        mask = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        valid_loss_mask = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        request_y0 = float(y_world)
        request_x0 = float(x_world)
        request_y1 = request_y0 + self.fov_nm
        request_x1 = request_x0 + self.fov_nm

        for cls_name in self.fine_classes:
            ci = crop.class_info.get(cls_name)
            if ci is None:
                continue
            try:
                label_arr = self._open_zarr(os.path.join(ci.zarr_path, ci.scale_path))
            except Exception:
                continue

            label_res = np.array(ci.resolution, dtype=float)
            label_off = np.array(ci.offset_world, dtype=float)
            label_shape = np.array(ci.shape, dtype=int)

            label_z0 = label_off[0]
            label_z1 = label_off[0] + label_shape[0] * label_res[0]
            if z_world < label_z0 or z_world >= label_z1:
                continue

            z_vox = int(round((z_world - label_off[0]) / label_res[0]))
            z_vox = max(0, min(z_vox, label_shape[0] - 1))

            label_y0 = label_off[1]
            label_x0 = label_off[2]
            label_y1 = label_off[1] + label_shape[1] * label_res[1]
            label_x1 = label_off[2] + label_shape[2] * label_res[2]
            intersect_y0 = max(request_y0, label_y0)
            intersect_x0 = max(request_x0, label_x0)
            intersect_y1 = min(request_y1, label_y1)
            intersect_x1 = min(request_x1, label_x1)
            if intersect_y1 <= intersect_y0 or intersect_x1 <= intersect_x0:
                continue

            y_start = int(np.floor((intersect_y0 - label_off[1]) / label_res[1]))
            x_start = int(np.floor((intersect_x0 - label_off[2]) / label_res[2]))
            y_end = int(np.ceil((intersect_y1 - label_off[1]) / label_res[1]))
            x_end = int(np.ceil((intersect_x1 - label_off[2]) / label_res[2]))
            y_start = max(0, min(y_start, label_shape[1]))
            x_start = max(0, min(x_start, label_shape[2]))
            y_end = max(y_start, min(y_end, label_shape[1]))
            x_end = max(x_start, min(x_end, label_shape[2]))
            if y_end <= y_start or x_end <= x_start:
                continue

            out_y0 = int(round((intersect_y0 - request_y0) / self.target_resolution_nm))
            out_x0 = int(round((intersect_x0 - request_x0) / self.target_resolution_nm))
            out_y1 = int(round((intersect_y1 - request_y0) / self.target_resolution_nm))
            out_x1 = int(round((intersect_x1 - request_x0) / self.target_resolution_nm))
            out_y0 = max(0, min(out_y0, self.target_size))
            out_x0 = max(0, min(out_x0, self.target_size))
            out_y1 = max(out_y0, min(out_y1, self.target_size))
            out_x1 = max(out_x0, min(out_x1, self.target_size))
            out_h = out_y1 - out_y0
            out_w = out_x1 - out_x0
            if out_h <= 0 or out_w <= 0:
                continue

            label_2d = np.array(label_arr[z_vox, y_start:y_end, x_start:x_end])
            binary = ((label_2d > 0) & (label_2d != 255)).astype(np.uint8)
            if binary.shape != (out_h, out_w):
                zoom_y = out_h / binary.shape[0]
                zoom_x = out_w / binary.shape[1]
                binary = ndimage_zoom(binary, (zoom_y, zoom_x), order=0)
                binary = binary[:out_h, :out_w]
            mask[out_y0:out_y1, out_x0:out_x1] = np.maximum(
                mask[out_y0:out_y1, out_x0:out_x1],
                binary.astype(np.uint8),
            )
            valid_loss_mask[out_y0:out_y1, out_x0:out_x1] = 255

        if mask.sum() == 0 or valid_loss_mask.sum() == 0:
            return None
        return mask, valid_loss_mask
