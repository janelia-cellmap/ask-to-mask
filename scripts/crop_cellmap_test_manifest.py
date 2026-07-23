#!/usr/bin/env python
"""Crop CellMap test-manifest EM regions into zarr plus PNG frames.

The challenge manifest contains one row per class label, so this script
deduplicates rows by crop geometry before reading raw EM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr
from PIL import Image

from ask_to_mask.training.zarr_utils import get_raw_path, get_scale_info


DEFAULT_MANIFEST_URL = (
    "https://raw.githubusercontent.com/janelia-cellmap/"
    "cellmap-segmentation-challenge/main/src/cellmap_segmentation_challenge/"
    "utils/test_crop_manifest.csv"
)


@dataclass
class ManifestCrop:
    crop_name: str
    dataset: str
    voxel_size: list[float]
    translation: list[float]
    shape: list[int]
    class_labels: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (
            self.crop_name,
            self.dataset,
            tuple(self.voxel_size),
            tuple(self.translation),
            tuple(self.shape),
        )


def parse_manifest_array(value: str, cast=float) -> list:
    """Parse manifest arrays like ``[1.74;2.0;2.0]``."""
    value = value.strip().strip("[]")
    if not value:
        return []
    return [cast(v.strip()) for v in value.split(";")]


def load_manifest(path_or_url: str) -> list[ManifestCrop]:
    """Load and deduplicate manifest rows in file order."""
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url) as response:
            text = response.read().decode("utf-8")
        rows = list(csv.DictReader(text.splitlines()))
    else:
        with open(path_or_url, newline="") as f:
            rows = list(csv.DictReader(f))

    crops: dict[tuple, ManifestCrop] = {}
    order: list[tuple] = []
    for row in rows:
        crop = ManifestCrop(
            crop_name=row["crop_name"].strip(),
            dataset=row["dataset"].strip(),
            voxel_size=parse_manifest_array(row["voxel_size"], float),
            translation=parse_manifest_array(row["translation"], float),
            shape=parse_manifest_array(row["shape"], lambda x: int(float(x))),
            class_labels=[],
        )
        if crop.key not in crops:
            crops[crop.key] = crop
            order.append(crop.key)
        label = row.get("class_label", "").strip()
        if label and label not in crops[crop.key].class_labels:
            crops[crop.key].class_labels.append(label)

    return [crops[key] for key in order]


def select_crops(
    crops: list[ManifestCrop],
    select: str,
    crop_names: list[str] | None,
) -> list[ManifestCrop]:
    if crop_names:
        wanted = set(crop_names)
        selected = [crop for crop in crops if crop.crop_name in wanted]
        missing = wanted - {crop.crop_name for crop in selected}
        if missing:
            raise ValueError(f"Crop name(s) not found in manifest: {sorted(missing)}")
        return selected
    if select == "all":
        return crops
    if select == "last":
        return [crops[-1]]
    if select == "max-id":
        return [max(crops, key=lambda crop: int(crop.crop_name))]
    raise ValueError(f"Unknown selection mode: {select}")


def choose_scale(
    raw_zarr_path: str,
    target_voxel_size: list[float],
    scale_path: str | None,
    allow_nearest: bool,
) -> tuple[str, list[float], list[float], tuple[int, ...]]:
    offsets, resolutions, shapes = get_scale_info(raw_zarr_path)
    if scale_path is not None:
        if scale_path not in resolutions:
            available = sorted(resolutions)
            raise ValueError(
                f"Scale path {scale_path!r} not found in {raw_zarr_path}. "
                f"Available scales: {available}"
            )
        return scale_path, resolutions[scale_path], offsets[scale_path], shapes[scale_path]

    target = np.array(target_voxel_size, dtype=float)
    exact = []
    for scale_path, res in resolutions.items():
        if np.allclose(np.array(res, dtype=float), target, rtol=0, atol=1e-3):
            exact.append((scale_path, res, offsets[scale_path], shapes[scale_path]))
    if exact:
        return sorted(exact, key=lambda item: item[0])[0]

    if not allow_nearest:
        available = {path: resolutions[path] for path in sorted(resolutions)}
        raise ValueError(
            f"No raw scale exactly matches manifest voxel size {target_voxel_size}. "
            f"Available scales: {available}"
        )

    candidates = []
    for scale_path, res in resolutions.items():
        res_arr = np.array(res, dtype=float)
        score = float(np.linalg.norm(np.log2(res_arr / target)))
        candidates.append((score, scale_path, res, offsets[scale_path], shapes[scale_path]))
    _, scale_path, res, offset, shape = sorted(candidates, key=lambda item: item[0])[0]
    return scale_path, res, offset, shape


def normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Convert a 2D EM slice to uint8 for PNG frames."""
    if slice_2d.dtype == np.uint8:
        return slice_2d
    lo, hi = np.percentile(slice_2d, (1, 99))
    if hi <= lo:
        hi = lo + 1
    scaled = np.clip(slice_2d.astype(np.float32), lo, hi)
    return ((scaled - lo) / (hi - lo) * 255).astype(np.uint8)


def make_chunks(shape: list[int], requested: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(max(1, min(int(size), int(chunk))) for size, chunk in zip(shape, requested))


def parse_size(value: str) -> tuple[int, int]:
    """Parse a movie size like 1280x720."""
    try:
        width, height = value.lower().split("x")
        return int(width), int(height)
    except Exception as exc:
        raise ValueError(f"Expected size WIDTHxHEIGHT, got {value!r}") from exc


def encode_movie(
    frames_dir: Path,
    movie_path: Path,
    fps: int,
    size: tuple[int, int],
    fit: str,
) -> None:
    """Encode PNG frames into an H.264 MP4."""
    import imageio_ffmpeg

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise ValueError(f"No PNG frames found in {frames_dir}")

    seq_dir = movie_path.parent / ".movie_frames_tmp"
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    seq_dir.mkdir()
    try:
        for i, frame_path in enumerate(frame_paths):
            (seq_dir / f"frame_{i:06d}.png").symlink_to(frame_path.resolve())

        width, height = size
        if fit == "pad":
            vf = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                "format=yuv420p"
            )
        elif fit == "crop":
            vf = (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},format=yuv420p"
            )
        elif fit == "stretch":
            vf = f"scale={width}:{height},format=yuv420p"
        else:
            raise ValueError(f"Unknown movie fit mode: {fit}")

        movie_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(seq_dir / "frame_%06d.png"),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(movie_path),
        ]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(seq_dir, ignore_errors=True)


def crop_em(
    crop: ManifestCrop,
    data_root: Path,
    output_root: Path,
    scale_path: str | None,
    allow_nearest_scale: bool,
    z_step_frames: int,
    max_frames: int | None,
    write_zarr: bool,
    make_movie: bool,
    movie_fps: int,
    movie_size: tuple[int, int],
    movie_fit: str,
    max_movie_seconds: float | None,
    movie_seconds: float | None,
    overwrite: bool,
    chunks: tuple[int, int, int],
) -> Path:
    dataset_dir = data_root / crop.dataset
    em_base = dataset_dir / f"{crop.dataset}.zarr" / "recon-1" / "em"
    raw_zarr_path = get_raw_path(str(em_base))
    if raw_zarr_path is None:
        raise FileNotFoundError(f"No fibsem-uint8 or fibsem-uint16 found under {em_base}")

    raw_scale_path, raw_res, raw_offset, raw_shape = choose_scale(
        raw_zarr_path, crop.voxel_size, scale_path, allow_nearest_scale
    )
    raw_arr = zarr.open(str(Path(raw_zarr_path) / raw_scale_path), mode="r")

    raw_res_arr = np.array(raw_res, dtype=float)
    raw_offset_arr = np.array(raw_offset, dtype=float)
    translation = np.array(crop.translation, dtype=float)
    manifest_res = np.array(crop.voxel_size, dtype=float)
    manifest_shape = np.array(crop.shape, dtype=int)
    world_extent = manifest_res * manifest_shape

    start = np.rint((translation - raw_offset_arr) / raw_res_arr).astype(int)
    out_shape = np.rint(world_extent / raw_res_arr).astype(int)
    stop = start + out_shape
    raw_shape_arr = np.array(raw_shape, dtype=int)
    if np.any(start < 0) or np.any(stop > raw_shape_arr):
        raise ValueError(
            f"Crop {crop.crop_name} is out of bounds for {raw_zarr_path}/{scale_path}: "
            f"start={start.tolist()} stop={stop.tolist()} raw_shape={raw_shape}"
        )

    crop_dir = output_root / f"crop_{crop.crop_name}__{crop.dataset}"
    if crop_dir.exists() and overwrite:
        shutil.rmtree(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    out_zarr = crop_dir / "em.zarr"
    z_out = None
    if write_zarr:
        if out_zarr.exists() and overwrite:
            shutil.rmtree(out_zarr)
        z_out = zarr.open(
            str(out_zarr / "s0"),
            mode="w",
            shape=tuple(int(v) for v in out_shape),
            chunks=make_chunks(out_shape.tolist(), chunks),
            dtype=raw_arr.dtype,
        )
        z_out.attrs["resolution"] = [float(v) for v in raw_res]
        z_out.attrs["offset"] = [float(v) for v in translation]

    frames_dir = crop_dir / "frames"
    if frames_dir.exists() and overwrite:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(exist_ok=True)

    z_indices = list(range(int(out_shape[0])))
    frame_indices = z_indices[:: max(1, z_step_frames)]
    if make_movie and movie_seconds is not None:
        target_movie_frames = max(1, int(round(movie_seconds * movie_fps)))
        if len(frame_indices) >= target_movie_frames:
            frame_indices = np.linspace(
                frame_indices[0],
                frame_indices[-1],
                target_movie_frames,
                dtype=int,
            ).tolist()
        else:
            print(
                f"  Warning: only {len(frame_indices)} frames available; "
                f"movie will be {len(frame_indices) / movie_fps:.2f}s"
            )
    elif make_movie and max_movie_seconds is not None:
        max_movie_frames = max(1, int(math.floor(max_movie_seconds * movie_fps)))
        if len(frame_indices) > max_movie_frames:
            stride = max(1, math.ceil(len(frame_indices) / max_movie_frames))
            frame_indices = frame_indices[::stride][:max_movie_frames]
    if max_frames is not None and len(frame_indices) > max_frames:
        stride = max(1, math.ceil(len(frame_indices) / max_frames))
        frame_indices = frame_indices[::stride][:max_frames]

    frame_index_set = set(frame_indices)
    y_slice = slice(int(start[1]), int(stop[1]))
    x_slice = slice(int(start[2]), int(stop[2]))
    for z_local in z_indices:
        z_raw = int(start[0] + z_local)
        sl = np.asarray(raw_arr[z_raw, y_slice, x_slice])
        if z_out is not None:
            z_out[z_local, :, :] = sl
        if z_local in frame_index_set:
            img = Image.fromarray(normalize_to_uint8(sl), mode="L")
            img.save(frames_dir / f"z{z_local:04d}.png")

    mid = int(out_shape[0] // 2)
    if z_out is not None:
        mid_slice = np.asarray(z_out[mid, :, :])
    else:
        mid_slice = np.asarray(raw_arr[int(start[0] + mid), y_slice, x_slice])
    Image.fromarray(normalize_to_uint8(mid_slice), mode="L").save(crop_dir / "middle.png")

    movie_path = crop_dir / f"movie_{movie_size[0]}x{movie_size[1]}_{movie_fps}fps.mp4"
    if make_movie:
        if movie_path.exists() and overwrite:
            movie_path.unlink()
        encode_movie(frames_dir, movie_path, movie_fps, movie_size, movie_fit)

    metadata = {
        "crop_name": crop.crop_name,
        "dataset": crop.dataset,
        "class_labels": crop.class_labels,
        "manifest": {
            "voxel_size": crop.voxel_size,
            "translation": crop.translation,
            "shape": crop.shape,
        },
        "raw": {
            "zarr_path": raw_zarr_path,
            "scale_path": raw_scale_path,
            "voxel_size": raw_res,
            "offset": raw_offset,
            "shape": list(raw_shape),
            "voxel_start": start.tolist(),
            "voxel_stop": stop.tolist(),
            "output_shape": out_shape.tolist(),
        },
        "outputs": {
            "zarr": str(out_zarr) if write_zarr else None,
            "frames": str(frames_dir),
            "middle_png": str(crop_dir / "middle.png"),
            "movie": str(movie_path) if make_movie else None,
        },
        "movie": {
            "fps": movie_fps if make_movie else None,
            "size": list(movie_size) if make_movie else None,
            "fit": movie_fit if make_movie else None,
            "target_seconds": movie_seconds if make_movie else None,
            "max_seconds": max_movie_seconds if make_movie else None,
            "frames": len(frame_indices) if make_movie else None,
            "duration_seconds": len(frame_indices) / movie_fps if make_movie else None,
        },
    }
    with open(crop_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return crop_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop raw EM from CellMap challenge test manifest locations."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_URL, help="CSV path or URL.")
    parser.add_argument("--data-root", type=Path, default=Path("/nrs/cellmap/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cellmap_test_crops"))
    parser.add_argument(
        "--select",
        choices=["last", "max-id", "all"],
        default="last",
        help="Crop selection when --crop-name is not supplied. 'last' means last unique crop in CSV order.",
    )
    parser.add_argument(
        "--crop-name",
        action="append",
        help="Specific crop_name to export. Can be passed multiple times.",
    )
    parser.add_argument(
        "--allow-nearest-scale",
        action="store_true",
        help="Use nearest raw scale if no exact voxel-size match exists.",
    )
    parser.add_argument(
        "--scale-path",
        default="manifest",
        help=(
            "Raw EM scale path to crop from. Default 'manifest' matches the "
            "manifest voxel_size; pass e.g. s0 to force a scale."
        ),
    )
    parser.add_argument("--z-step-frames", type=int, default=1, help="Save every Nth z-slice as PNG.")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap number of PNG frames.")
    parser.add_argument(
        "--write-zarr",
        action="store_true",
        help="Also write the full cropped EM volume to em.zarr. PNG frames are always written.",
    )
    parser.add_argument("--make-movie", action="store_true", help="Encode PNG frames to MP4.")
    parser.add_argument("--movie-fps", type=int, default=24, help="Movie frame rate.")
    parser.add_argument(
        "--movie-seconds",
        type=float,
        default=None,
        help="Target exact movie duration in seconds by evenly sampling frames.",
    )
    parser.add_argument(
        "--max-movie-seconds",
        type=float,
        default=9.5,
        help="Subsample frames so encoded movie is no longer than this many seconds (default: 9.5).",
    )
    parser.add_argument(
        "--movie-size",
        default="1280x720",
        help="Movie canvas size WIDTHxHEIGHT (default: 1280x720).",
    )
    parser.add_argument(
        "--movie-fit",
        choices=["pad", "crop", "stretch"],
        default="pad",
        help="How to fit EM frames into the movie canvas (default: pad).",
    )
    parser.add_argument(
        "--chunks",
        default="64,256,256",
        help="Output zarr chunk shape as z,y,x (default: 64,256,256).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output crop dirs.")
    args = parser.parse_args()

    chunks = tuple(int(v.strip()) for v in args.chunks.split(","))
    if len(chunks) != 3:
        raise SystemExit("--chunks must have three comma-separated ints: z,y,x")
    movie_size = parse_size(args.movie_size)

    crops = load_manifest(args.manifest)
    selected = select_crops(crops, args.select, args.crop_name)
    scale_path = None if args.scale_path == "manifest" else args.scale_path
    print(f"Loaded {len(crops)} unique manifest crops; exporting {len(selected)}")
    for crop in selected:
        print(
            f"Cropping {crop.crop_name} {crop.dataset}: "
            f"voxel_size={crop.voxel_size} translation={crop.translation} shape={crop.shape}"
        )
        out_dir = crop_em(
            crop=crop,
            data_root=args.data_root,
            output_root=args.output_dir,
            scale_path=scale_path,
            allow_nearest_scale=args.allow_nearest_scale,
            z_step_frames=args.z_step_frames,
            max_frames=args.max_frames,
            write_zarr=args.write_zarr,
            make_movie=args.make_movie,
            movie_fps=args.movie_fps,
            movie_size=movie_size,
            movie_fit=args.movie_fit,
            max_movie_seconds=args.max_movie_seconds,
            movie_seconds=args.movie_seconds,
            overwrite=args.overwrite,
            chunks=chunks,
        )
        print(f"  Wrote {out_dir}")


if __name__ == "__main__":
    main()
