#!/usr/bin/env python
"""Generate a grid image or video of orthogonal EM slices from a zarr volume.

Grid mode (default): 3 rows (XY, XZ, YZ) × N columns of sampled slices.
Video mode (--video): scroll through all slices of one plane as an mp4.

Usage:
    # Grid
    python scripts/ortho_grid.py \
        --config configs/refine_ortho_example.yaml \
        --output ortho_grid.png

    # Video of XY plane (every slice, not subsampled)
    python scripts/ortho_grid.py \
        --config configs/refine_ortho_example.yaml \
        --video --plane xy --output xy_video.mp4

    # Video with specific fps
    python scripts/ortho_grid.py \
        --config configs/refine_ortho_example.yaml \
        --video --plane xy --fps 10 --output xy_video.mp4
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw


def _load_config(args):
    """Apply YAML config values to args."""
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if not args.zarr_path:
            args.zarr_path = cfg.get("zarr_path")
        if not args.dataset_path:
            args.dataset_path = cfg.get("dataset_path")
        if not args.roi:
            args.roi = cfg.get("roi")
        if args.z_step_nm is None:
            args.z_step_nm = cfg.get("z_step_nm")


def _make_grid(args):
    """Generate a grid image of orthogonal slices."""
    from ask_to_mask.zarr_io import load_zarr_ortho_slices, parse_roi

    roi = parse_roi(args.roi)
    ortho = load_zarr_ortho_slices(
        args.zarr_path, args.dataset_path,
        roi=roi, z_step_nm=args.z_step_nm,
    )

    planes = ["xy", "xz", "yz"]
    plane_labels = ["XY", "XZ", "YZ"]

    all_slices = []
    for plane in planes:
        slices, _, indices = ortho[plane]
        all_slices.append(slices)

    # Collect actual indices for labeling
    all_indices = []
    for plane in planes:
        _, _, indices = ortho[plane]
        all_indices.append(indices)

    n_cols = max(len(s) for s in all_slices)
    n_rows = len(planes)

    max_h = max(np.array(s).shape[0] for sl in all_slices for s in sl)
    max_w = max(np.array(s).shape[1] for sl in all_slices for s in sl)

    target_cell = 256
    scale = min(1.0, target_cell / max(max_h, max_w))
    cell_w = int(max_w * scale)
    cell_h = int(max_h * scale)

    label_w = 40
    header_h = 16  # top margin for slice numbers
    pad = 10       # gap between cells (visible separator)

    grid_w = label_w + n_cols * (cell_w + pad) + pad
    grid_h = header_h + n_rows * (cell_h + pad) + pad

    grid = Image.new("RGB", (grid_w, grid_h), (80, 80, 80))
    draw = ImageDraw.Draw(grid)

    for row, (plane, label, slices, indices) in enumerate(
        zip(planes, plane_labels, all_slices, all_indices)
    ):
        y0 = header_h + row * (cell_h + pad) + pad

        # Row label
        draw.text((4, y0 + cell_h // 2 - 6), label, fill="white")

        for col, sl in enumerate(slices):
            x0 = label_w + col * (cell_w + pad) + pad

            # Slice number header (only draw once, from first row)
            if row == 0:
                idx = indices[col] if col < len(indices) else col
                draw.text((x0 + cell_w // 2 - 8, 1), str(idx), fill="yellow")

            resized = sl.resize((cell_w, cell_h), Image.BILINEAR)
            grid.paste(resized, (x0, y0))

    grid.save(args.output)
    print(f"Saved {grid_w}x{grid_h} grid ({n_rows}x{n_cols}) to {args.output}")


def _make_video(args):
    """Generate an mp4 video scrolling through all slices of one plane."""
    from ask_to_mask.zarr_io import load_zarr_roi, parse_roi, _normalize_to_uint8

    roi = parse_roi(args.roi)
    data_3d, actual_roi, voxel_size = load_zarr_roi(
        args.zarr_path, args.dataset_path, roi
    )
    nz, ny, nx = data_3d.shape
    plane = args.plane

    if plane == "xy":
        n_frames = nz
    elif plane == "xz":
        n_frames = ny
    else:
        n_frames = nx

    print(f"Generating {plane.upper()} video: {n_frames} frames from {data_3d.shape} volume")

    frames = []
    for i in range(n_frames):
        if plane == "xy":
            sl = data_3d[i, :, :]
        elif plane == "xz":
            sl = data_3d[:, i, :]
        else:
            sl = data_3d[:, :, i]

        normed = _normalize_to_uint8(sl)
        # Stack to RGB
        rgb = np.stack([normed] * 3, axis=-1)
        frames.append(rgb)

    output = args.output
    if not output.endswith((".mp4", ".mov")):
        output = output.rsplit(".", 1)[0] + ".mp4"

    import imageio_ffmpeg
    from imageio import get_writer

    writer = get_writer(output, fps=args.fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved {n_frames}-frame video to {output} ({args.fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Orthogonal EM slice grid or video")
    parser.add_argument("--zarr-path", default=None, help="Path to zarr volume")
    parser.add_argument("--dataset-path", default=None, help="Sub-path within zarr")
    parser.add_argument("--roi", default=None, help="ROI: '[z0:z1, y0:y1, x0:x1]'")
    parser.add_argument("--z-step-nm", type=float, default=None, help="Subsample step in nm (grid mode)")
    parser.add_argument("--output", default="ortho_grid.png", help="Output path (.png for grid, .mp4 for video)")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--video", action="store_true", help="Generate video instead of grid")
    parser.add_argument("--plane", default="xy", choices=["xy", "xz", "yz"], help="Plane for video mode")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for video")
    args = parser.parse_args()

    _load_config(args)

    if not args.zarr_path or not args.roi:
        parser.error("--zarr-path and --roi are required (or use --config)")

    if args.video:
        _make_video(args)
    else:
        _make_grid(args)


if __name__ == "__main__":
    main()
