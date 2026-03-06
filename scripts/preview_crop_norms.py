#!/usr/bin/env python
"""Generate a single preview slice per crop to verify normalization/contrast.

Saves one PNG per crop with the dataset name, crop ID, norm params, and raw
intensity stats annotated on the image.

Usage:
    pixi run python scripts/preview_crop_norms.py \
        --norms-csv configs/norms.csv \
        --cache-dir configs \
        --output-dir runs/norm_previews
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ask_to_mask.config import ORGANELLE_FINE_CLASSES
from src.ask_to_mask.training.zarr_utils import (
    CropInfo,
    NormParams,
    discover_crops,
    load_norms,
    normalize_raw,
)

logger = logging.getLogger(__name__)


def read_mid_slice(crop: CropInfo) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the middle Z-slice of a crop. Returns (raw_unnorm, raw_norm) or None."""
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
    crop_extent = np.array(crop.crop_extent_world)

    # Middle Z-slice
    z_extent_vox = int(crop_extent[0] / raw_res[0])
    if z_extent_vox < 1:
        return None
    z_mid = z_extent_vox // 2
    z_world = crop_origin[0] + z_mid * raw_res[0]
    z_vox = int(round((z_world - raw_off[0]) / raw_res[0]))
    z_vox = max(0, min(z_vox, raw_shape[0] - 1))

    # Full YX crop extent
    y_start = int(round((crop_origin[1] - raw_off[1]) / raw_res[1]))
    x_start = int(round((crop_origin[2] - raw_off[2]) / raw_res[2]))
    y_size = int(round(crop_extent[1] / raw_res[1]))
    x_size = int(round(crop_extent[2] / raw_res[2]))
    y_start = max(0, min(y_start, raw_shape[1] - 1))
    x_start = max(0, min(x_start, raw_shape[2] - 1))
    y_end = min(y_start + y_size, raw_shape[1])
    x_end = min(x_start + x_size, raw_shape[2])

    raw_2d = np.array(raw_arr[z_vox, y_start:y_end, x_start:x_end])
    raw_norm = normalize_raw(raw_2d, crop.norm_params)
    return raw_2d, raw_norm


def compute_auto_norms(
    crops_by_dataset: dict[str, list[CropInfo]],
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    num_slices: int = 5,
) -> dict[str, NormParams]:
    """Compute normalization params per dataset by sampling slices.

    Samples up to num_slices middle Z-slices across crops per dataset,
    then uses percentile clipping to determine min/max. Inverted is set
    when the majority of raw values are high (bright background, dark
    structures — typical for FIB-SEM).
    """
    auto_norms = {}
    for dataset_name in sorted(crops_by_dataset):
        ds_crops = crops_by_dataset[dataset_name]
        # Sample a few crops
        indices = np.linspace(0, len(ds_crops) - 1, min(num_slices, len(ds_crops)), dtype=int)
        all_vals = []
        for i in indices:
            crop = ds_crops[i]
            # Read raw without normalization
            try:
                raw_arr = zarr.open(
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
            # Subsample large slices to keep memory reasonable
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
        logger.info(f"  {dataset_name}: [{p_low:.1f}, {p_high:.1f}]")

    return auto_norms


def make_preview(
    raw_unnorm: np.ndarray,
    raw_norm: np.ndarray,
    crop: CropInfo,
    max_display: int = 512,
) -> Image.Image:
    """Create a side-by-side preview: raw (auto-stretched) | normalized."""
    # Auto-stretch the unnormalized data for comparison
    rmin, rmax = float(raw_unnorm.min()), float(raw_unnorm.max())
    if rmax > rmin:
        auto = ((raw_unnorm.astype(np.float32) - rmin) / (rmax - rmin) * 255).astype(
            np.uint8
        )
    else:
        auto = np.zeros_like(raw_unnorm, dtype=np.uint8)

    norm_uint8 = (raw_norm * 255).astype(np.uint8)

    # Resize both to fit max_display while preserving aspect
    h, w = auto.shape
    scale = min(max_display / h, max_display / w, 1.0)
    new_h, new_w = int(h * scale), int(w * scale)

    auto_pil = Image.fromarray(auto).resize((new_w, new_h), Image.LANCZOS)
    norm_pil = Image.fromarray(norm_uint8).resize((new_w, new_h), Image.LANCZOS)

    # Compose side-by-side with text header
    header_h = 80
    canvas_w = new_w * 2 + 10
    canvas_h = new_h + header_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))

    # Paste images
    canvas.paste(Image.merge("RGB", [auto_pil] * 3), (0, header_h))
    canvas.paste(Image.merge("RGB", [norm_pil] * 3), (new_w + 10, header_h))

    # Add text
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    norm = crop.norm_params
    lines = [
        f"{crop.dataset_name} / {crop.crop_id}",
        f"norm: [{norm.min_val}, {norm.max_val}] inv={norm.inverted}",
        f"raw range: [{rmin:.1f}, {rmax:.1f}]  shape: {raw_unnorm.shape}",
        f"res: {crop.raw_resolution} nm/vox",
        f"Left: auto-stretched  |  Right: dataset norms",
    ]
    y = 2
    for line in lines:
        draw.text((4, y), line, fill=(255, 255, 255), font=font)
        y += 14

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Preview crop normalizations")
    parser.add_argument("--data-root", default="/nrs/cellmap/data")
    parser.add_argument("--norms-csv", default="configs/norms.csv")
    parser.add_argument("--cache-dir", default="configs")
    parser.add_argument("--output-dir", default="runs/norm_previews")
    parser.add_argument(
        "--max-crops-per-dataset",
        type=int,
        default=1,
        help="Max crops to preview per dataset (default: 1)",
    )
    parser.add_argument(
        "--auto-norms",
        action="store_true",
        help="Compute norms automatically from percentile stats and use those "
        "for the 'normalized' column. Also saves a new norms CSV.",
    )
    parser.add_argument(
        "--auto-norms-output",
        default=None,
        help="Path to write auto-computed norms CSV (default: <output_dir>/auto_norms.csv)",
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=1.0,
        help="Lower percentile for auto-norm (default: 1.0)",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
        help="Upper percentile for auto-norm (default: 99.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    norms = load_norms(args.norms_csv) if args.norms_csv else {}

    # Collect all fine classes
    all_fine = set()
    for classes in ORGANELLE_FINE_CLASSES.values():
        all_fine.update(classes)

    crops = discover_crops(
        data_root=args.data_root,
        target_classes=sorted(all_fine),
        norms=norms,
        cache_dir=args.cache_dir,
    )
    logger.info(f"Found {len(crops)} crops total")

    os.makedirs(args.output_dir, exist_ok=True)

    # Group by dataset
    by_dataset: dict[str, list[CropInfo]] = {}
    for c in crops:
        by_dataset.setdefault(c.dataset_name, []).append(c)

    # Optionally compute auto norms
    if args.auto_norms:
        logger.info("Computing auto norms from percentile stats...")
        auto_norms = compute_auto_norms(
            by_dataset,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
        )
        # Apply auto norms to crops
        for ds_crops in by_dataset.values():
            for crop in ds_crops:
                if crop.dataset_name in auto_norms:
                    crop.norm_params = auto_norms[crop.dataset_name]

        # Save auto norms CSV
        auto_csv = args.auto_norms_output or os.path.join(
            args.output_dir, "auto_norms.csv"
        )
        import csv

        with open(auto_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "min", "max", "inverted", "set_norm"])
            for name in sorted(auto_norms):
                n = auto_norms[name]
                writer.writerow([name, n.min_val, n.max_val, n.inverted, True])
        logger.info(f"Saved auto norms to {auto_csv}")

    # Generate previews
    total = 0
    for dataset_name in sorted(by_dataset):
        ds_crops = by_dataset[dataset_name]
        # Pick evenly spaced crops
        n = min(args.max_crops_per_dataset, len(ds_crops))
        indices = np.linspace(0, len(ds_crops) - 1, n, dtype=int)

        for i in indices:
            crop = ds_crops[i]
            result = read_mid_slice(crop)
            if result is None:
                logger.warning(f"  SKIP {crop.dataset_name}/{crop.crop_id}: read failed")
                continue

            raw_unnorm, raw_norm = result
            preview = make_preview(raw_unnorm, raw_norm, crop)

            fname = f"{crop.dataset_name}__{crop.crop_id}.png"
            preview.save(os.path.join(args.output_dir, fname))
            total += 1
            logger.info(f"  [{total}] {fname}")

    logger.info(f"Done. Saved {total} previews to {args.output_dir}")


if __name__ == "__main__":
    main()
