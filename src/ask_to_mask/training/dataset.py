"""Training dataset: produces (conditioning_image, target_image, prompt) triplets."""

from __future__ import annotations

import logging
import os

import numpy as np
import zarr
from PIL import Image
from scipy.ndimage import zoom as ndimage_zoom
from torch.utils.data import Dataset

from ..config import ORGANELLE_FINE_CLASSES, ORGANELLES, OrganelleClass
from .zarr_utils import (
    CropInfo,
    compute_auto_norms,
    discover_crops,
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
    ):
        self.samples_per_epoch = samples_per_epoch
        self.min_mask_fraction = min_mask_fraction
        self.target_mode = target_mode
        self.include_resolution = include_resolution
        self.auto_norms_per_image = auto_norms_per_image
        self.auto_norms_percentile_low = auto_norms_percentile_low
        self.auto_norms_percentile_high = auto_norms_percentile_high
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

    def _try_sample(
        self, skip_mask_filter: bool = False
    ) -> tuple[Image.Image, Image.Image, str] | None:
        """Try to produce one valid sample, or return None."""
        organelle_key = self._pick_organelle()
        organelle = ORGANELLES[organelle_key]
        fine_classes = ORGANELLE_FINE_CLASSES[organelle_key]
        crops = self.organelle_crops[organelle_key]
        crop = crops[self.rng.integers(len(crops))]

        # Determine crop YX size in voxels at native resolution
        crop_extent = np.array(crop.crop_extent_world)
        crop_origin = np.array(crop.crop_origin_world)
        raw_res = np.array(crop.raw_resolution)
        crop_yx_voxels = crop_extent[1:] / raw_res[1:]

        is_small = np.all(crop_yx_voxels < SMALL_CROP_THRESHOLD)

        # Pick a random Z-slice within the crop
        z_extent_vox = int(crop_extent[0] / raw_res[0])
        if z_extent_vox < 1:
            return None
        z_idx_in_crop = self.rng.integers(z_extent_vox)

        # For large crops, compute a random sub-crop origin in world coords
        # so both raw and label readers use the same region.
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

        # Read raw 2D slice
        raw_slice = self._read_raw_slice(
            crop, z_idx_in_crop, is_small, subcrop_origin_world
        )
        if raw_slice is None:
            return None

        # Read and union label slices
        mask = self._read_label_slice(
            crop, fine_classes, z_idx_in_crop, raw_slice.shape,
            is_small, subcrop_origin_world,
        )
        if mask is None:
            return None

        # Check mask fraction
        if not skip_mask_filter:
            mask_fraction = mask.sum() / mask.size
            if mask_fraction < self.min_mask_fraction:
                return None

        # Build images
        # raw_slice is float32 [0,1], shape [H, W]
        raw_uint8 = (raw_slice * 255).astype(np.uint8)
        raw_rgb = np.stack([raw_uint8] * 3, axis=-1)  # [H, W, 3]

        # Create target: color organelle pixels
        if self.target_mode == "segmentation":
            target_rgb = np.zeros_like(raw_rgb)
        else:
            target_rgb = raw_rgb.copy()
        color = np.array(organelle.rgb, dtype=np.uint8)
        target_rgb[mask > 0] = color

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

        prompt = organelle.build_prompt(resolution_nm=resolution_nm)
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
        """Apply random spatial augmentations to both images identically."""
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

        return raw, target

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
