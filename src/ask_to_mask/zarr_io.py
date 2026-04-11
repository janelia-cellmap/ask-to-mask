"""Zarr I/O utilities for reading EM slices from zarr volumes.

Supports ROI specification in world coordinates (nm) via ``funlib.geometry.Roi``
and reading via ``funlib.persistence``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from PIL import Image

from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to uint8 [0, 255].

    - uint8 data is returned as-is.
    - Other dtypes are percentile-clipped (1st-99th) and rescaled.
    """
    if data.dtype == np.uint8:
        return data
    p_lo, p_hi = np.percentile(data, (1, 99))
    if p_hi <= p_lo:
        p_hi = p_lo + 1
    clipped = np.clip(data.astype(np.float32), p_lo, p_hi)
    return ((clipped - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)


def _slice_to_rgb(data_2d: np.ndarray) -> Image.Image:
    """Convert a 2D uint8 array to an RGB PIL Image."""
    normed = _normalize_to_uint8(data_2d)
    return Image.fromarray(np.stack([normed] * 3, axis=-1), mode="RGB")


def parse_roi(roi_str: str) -> Roi:
    """Parse an ROI string in world coordinates (nm).

    Accepted formats::

        [500:1000,500:1000,1000:11000]   # begin:end per axis
        500:1000,500:1000,1000:11000      # without brackets

    Returns:
        ``funlib.geometry.Roi`` with offset and shape derived from begin:end.
    """
    roi_str = roi_str.strip().strip("[]")
    parts = [p.strip() for p in roi_str.split(",")]
    begins, ends = [], []
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Expected 'begin:end' per axis, got '{part}'")
        b, e = part.split(":")
        begins.append(int(b))
        ends.append(int(e))
    offset = Coordinate(begins)
    shape = Coordinate(ends) - offset
    return Roi(offset, shape)


def _read_multiscale_metadata(zarr_path: str, dataset_path: str | None = None):
    """Read voxel_size and offset from OME-NGFF multiscales attributes.

    Looks for multiscales metadata in the zarr group that is the parent of the
    array. If ``dataset_path`` is provided (e.g., ``"s0"``), the parent is
    ``zarr_path``. Otherwise, tries the parent directory of ``zarr_path``.

    Returns:
        (voxel_size, offset) as lists of floats, or (None, None) if not found.
    """
    import os

    if dataset_path:
        group_path = zarr_path
        array_name = dataset_path
    else:
        # zarr_path might be e.g. .../fibsem-uint8/s0 — parent has metadata
        group_path = os.path.dirname(zarr_path)
        array_name = os.path.basename(zarr_path)

    try:
        grp = zarr.open(group_path, mode="r")
        if "multiscales" not in grp.attrs:
            return None, None
        ms = grp.attrs["multiscales"][0]
        for ds_info in ms["datasets"]:
            if ds_info["path"] == array_name:
                transforms = ds_info["coordinateTransformations"]
                voxel_size = transforms[0]["scale"]
                offset = transforms[1]["translation"] if len(transforms) > 1 else [0.0] * len(voxel_size)
                return voxel_size, offset
    except Exception:
        pass
    return None, None


def open_volume(zarr_path: str, dataset_path: str | None = None):
    """Open a zarr dataset via funlib.persistence.

    Reads voxel_size and offset from OME-NGFF multiscale metadata in the
    parent group, falling back to funlib.persistence defaults.

    Returns:
        A ``funlib.persistence.Array`` with ``.roi``, ``.voxel_size``, and
        numpy-style indexing.
    """
    store_path = zarr_path
    if dataset_path:
        store_path = f"{zarr_path}/{dataset_path}"

    voxel_size, offset = _read_multiscale_metadata(zarr_path, dataset_path)

    kwargs = {"mode": "r"}
    if voxel_size is not None:
        kwargs["voxel_size"] = [int(v) for v in voxel_size]
    if offset is not None:
        kwargs["offset"] = [int(v) for v in offset]

    return open_ds(store_path, **kwargs)


def load_zarr_roi(
    zarr_path: str,
    dataset_path: str | None = None,
    roi: Roi | None = None,
) -> tuple[np.ndarray, Roi, Coordinate]:
    """Load a 3D region from a zarr volume.

    Args:
        zarr_path: Path to zarr volume or group.
        dataset_path: Optional sub-path within the zarr.
        roi: Region of interest in world coordinates (nm).
            If None, loads the entire volume.

    Returns:
        Tuple of (data_3d, actual_roi, voxel_size).
    """
    ds = open_volume(zarr_path, dataset_path)
    voxel_size = ds.voxel_size

    if roi is not None:
        # Snap ROI to voxel grid
        roi = roi.snap_to_grid(voxel_size, mode="grow")
        # Intersect with available data
        roi = roi.intersect(ds.roi)
        if roi.empty:
            raise ValueError(
                f"ROI {roi} does not intersect dataset ROI {ds.roi}"
            )
    else:
        roi = ds.roi

    data = ds[roi]
    print(f"  Loaded ROI {roi} (voxel_size={voxel_size}, shape={data.shape})")
    return np.asarray(data), roi, voxel_size


def load_zarr_zstack(
    zarr_path: str,
    dataset_path: str | None = None,
    roi: Roi | None = None,
    z_start: int = 0,
    z_count: int = 1,
    z_step: int = 1,
    z_step_nm: float | None = None,
) -> list[Image.Image]:
    """Load 2D slices from a zarr volume.

    If ``roi`` is provided (in world coordinates / nm), it defines the full 3D
    region to read — Z range comes from the ROI's first axis, and XY cropping
    from the remaining axes.  Use ``z_step_nm`` to subsample in Z within the
    ROI (e.g., ``z_step_nm=40`` reads every 40 nm).

    If ``roi`` is None, falls back to ``z_start``/``z_count``/``z_step``
    (in voxel indices) with full XY extent.

    Returns:
        List of RGB PIL Images, one per slice.
    """
    if roi is not None:
        data_3d, actual_roi, voxel_size = load_zarr_roi(
            zarr_path, dataset_path, roi
        )
        # Compute stride in voxels from z_step_nm
        if z_step_nm is not None:
            z_voxel = int(voxel_size[0])
            stride = max(1, round(z_step_nm / z_voxel))
        else:
            stride = 1
        slices = []
        for z in range(0, data_3d.shape[0], stride):
            slices.append(_slice_to_rgb(data_3d[z]))
        if stride > 1:
            print(f"  Z subsampling: every {z_step_nm} nm = stride {stride} voxels → {len(slices)} slices")
        return slices

    # Fallback: raw index-based loading
    ds = open_volume(zarr_path, dataset_path)
    arr = ds.data  # underlying zarr array
    if arr.ndim < 3:
        raise ValueError(
            f"Expected >= 3D array, got {arr.ndim}D with shape {arr.shape}"
        )
    z_end = z_start + z_count * z_step
    if z_start < 0 or z_end > arr.shape[0]:
        raise IndexError(
            f"Z range [{z_start}:{z_end}:{z_step}] out of bounds for "
            f"axis 0 with size {arr.shape[0]}"
        )
    slices = []
    for z in range(z_start, z_end, z_step):
        data = arr[z, :, :]
        slices.append(_slice_to_rgb(np.asarray(data)))
    return slices


def load_zarr_slice(
    zarr_path: str,
    dataset_path: str | None = None,
    z_index: int = 0,
) -> Image.Image:
    """Load a single 2D YX slice from a 3D zarr array."""
    images = load_zarr_zstack(
        zarr_path, dataset_path, z_start=z_index, z_count=1,
    )
    return images[0]


def get_zarr_info(
    zarr_path: str,
    dataset_path: str | None = None,
) -> dict:
    """Return metadata about a zarr dataset.

    Returns:
        Dictionary with ``shape``, ``dtype``, ``voxel_size``, ``roi``.
    """
    ds = open_volume(zarr_path, dataset_path)
    return {
        "shape": ds.data.shape,
        "dtype": str(ds.data.dtype),
        "voxel_size": tuple(ds.voxel_size),
        "roi": str(ds.roi),
    }


def load_zarr_ortho_slices(
    zarr_path: str,
    dataset_path: str | None = None,
    roi: Roi | None = None,
    z_step_nm: float | None = None,
) -> dict[str, tuple[list[Image.Image], np.ndarray]]:
    """Load slices from 3 orthogonal planes (XY, XZ, YZ) within an ROI.

    Each plane is sampled to have a comparable number of slices: the
    number of XY slices (along Z) is computed from the ROI, and the XZ
    and YZ planes are uniformly sampled to the same count.

    Args:
        zarr_path: Path to zarr volume or group.
        dataset_path: Optional sub-path within the zarr.
        roi: Region of interest in world coordinates (nm).
        z_step_nm: Optional Z subsampling step in nm.

    Returns:
        Dict with keys ``"xy"``, ``"xz"``, ``"yz"``.
        Each value is ``(slices, data_3d)`` where slices is a list of
        RGB PIL Images and data_3d is the loaded 3D ROI array.
    """
    data_3d, actual_roi, voxel_size = load_zarr_roi(
        zarr_path, dataset_path, roi
    )
    nz, ny, nx = data_3d.shape

    # XY slices (along Z axis) — standard
    if z_step_nm is not None:
        z_voxel = int(voxel_size[0])
        z_stride = max(1, round(z_step_nm / z_voxel))
    else:
        z_stride = 1
    xy_indices = list(range(0, nz, z_stride))
    xy_slices = [_slice_to_rgb(data_3d[z]) for z in xy_indices]

    # Sample XZ and YZ to get roughly the same number of slices as XY
    n_target = len(xy_slices)

    # XZ slices (along Y axis) — each slice is (Z, X)
    xz_indices = np.linspace(0, ny - 1, min(n_target, ny), dtype=int).tolist()
    xz_slices = [_slice_to_rgb(data_3d[:, y, :]) for y in xz_indices]

    # YZ slices (along X axis) — each slice is (Z, Y)
    yz_indices = np.linspace(0, nx - 1, min(n_target, nx), dtype=int).tolist()
    yz_slices = [_slice_to_rgb(data_3d[:, :, x]) for x in yz_indices]

    print(f"  Orthogonal slices: XY={len(xy_slices)}, XZ={len(xz_slices)}, YZ={len(yz_slices)}")
    print(f"  Slice shapes: XY={ny}x{nx}, XZ={nz}x{nx}, YZ={nz}x{ny}")

    return {
        "xy": (xy_slices, data_3d, xy_indices),
        "xz": (xz_slices, data_3d, xz_indices),
        "yz": (yz_slices, data_3d, yz_indices),
    }


def save_masks_to_zarr(
    masks: np.ndarray,
    output_path: str,
    dataset_name: str = "masks",
    chunks: tuple[int, ...] | None = None,
    voxel_size: tuple[float, ...] | None = None,
    offset: tuple[float, ...] | None = None,
) -> None:
    """Write a 3D mask stack (Z, Y, X) as an OME-NGFF multiscale zarr v2.

    Writes all metadata files manually to exactly match the CellMap/neuroglancer
    zarr v2 convention with nested chunk directories.

    Structure::

        output_path/                    .zgroup + .zattrs (empty)
          dataset_name/                 .zgroup + .zattrs (multiscales)
            s0/                         .zarray + .zattrs (resolution/offset)
              0/0/0, 0/0/1, ...         nested chunk dirs

    Args:
        masks: 3D numpy array of masks in ZYX order.
        output_path: Path for the output zarr store.
        dataset_name: Name of the multiscale group (e.g. "merged", "xy").
        chunks: Chunk shape; defaults to (128, 128, 128).
        voxel_size: Voxel size in nm per axis (Z, Y, X).
        offset: World offset in nm per axis (Z, Y, X).
    """
    import gzip as _gzip
    import json as _json
    import shutil
    from pathlib import Path as _Path

    if chunks is None:
        chunks = tuple(min(128, s) for s in masks.shape)

    root_path = _Path(output_path)
    group_path = root_path / dataset_name
    array_path = group_path / "s0"

    # Clean up old array
    if array_path.exists():
        shutil.rmtree(array_path)
    array_path.mkdir(parents=True, exist_ok=True)

    # Write .zgroup for root and dataset group
    for gp in (root_path, group_path):
        (gp / ".zgroup").write_text('{\n    "zarr_format": 2\n}')
    # Root .zattrs — empty (no multiscales at root)
    root_zattrs = root_path / ".zattrs"
    if not root_zattrs.exists():
        root_zattrs.write_text("{}")

    # Write s0/.zarray — matching reference format exactly
    zarray_meta = {
        "chunks": list(chunks),
        "compressor": {"id": "gzip", "level": 5},
        "dimension_separator": "/",
        "dtype": f"|{masks.dtype.str[1:]}",
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": list(masks.shape),
        "zarr_format": 2,
    }
    (array_path / ".zarray").write_text(_json.dumps(zarray_meta, indent=4))

    # Write s0/.zattrs with resolution and offset
    scale = [float(v) for v in voxel_size] if voxel_size else [1.0, 1.0, 1.0]
    translation = [float(v) for v in offset] if offset else [0.0, 0.0, 0.0]
    (array_path / ".zattrs").write_text(_json.dumps({
        "offset": translation,
        "resolution": scale,
    }, indent=4))

    # Write dataset_name/.zattrs with OME-NGFF multiscales
    (group_path / ".zattrs").write_text(_json.dumps({
        "multiscales": [{
            "axes": [
                {"name": "z", "type": "space", "unit": "nanometer"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"},
            ],
            "coordinateTransformations": [{"scale": [1.0, 1.0, 1.0], "type": "scale"}],
            "datasets": [{
                "coordinateTransformations": [
                    {"scale": scale, "type": "scale"},
                    {"translation": translation, "type": "translation"},
                ],
                "path": "s0",
            }],
            "name": "",
            "version": "0.4",
        }],
    }, indent=4))

    # Write chunks as gzip-compressed nested directories (0/0/0, 0/0/1, ...)
    # Edge chunks must be padded to the full chunk shape with fill_value (0).
    cz, cy, cx = chunks
    for zi in range(0, masks.shape[0], cz):
        for yi in range(0, masks.shape[1], cy):
            for xi in range(0, masks.shape[2], cx):
                chunk = masks[zi:zi + cz, yi:yi + cy, xi:xi + cx]
                # Pad edge chunks to full chunk size
                if chunk.shape != (cz, cy, cx):
                    padded = np.zeros((cz, cy, cx), dtype=masks.dtype)
                    padded[:chunk.shape[0], :chunk.shape[1], :chunk.shape[2]] = chunk
                    chunk = padded
                chunk_key = f"{zi // cz}/{yi // cy}/{xi // cx}"
                chunk_path = array_path / chunk_key
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                with open(chunk_path, "wb") as f:
                    f.write(_gzip.compress(chunk.tobytes(order="C"), compresslevel=5))

    print(f"  Saved zarr: {output_path}/{dataset_name}/s0 {masks.shape}")
    if voxel_size:
        print(f"  Metadata: scale={scale}, offset={translation}")
