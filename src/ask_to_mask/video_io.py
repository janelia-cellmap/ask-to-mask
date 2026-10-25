"""Encode sequences of PIL images into video files."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def frames_to_video(
    frames: list[Image.Image],
    output_path: str | Path,
    duration_seconds: float | None = None,
    fps: float | None = None,
    upsample_factor: float = 1.0,
) -> Path:
    """Write a sequence of PIL images as an H.264 mp4.

    Exactly one of ``duration_seconds`` or ``fps`` must be given. If
    ``duration_seconds`` is given, fps is derived as
    ``len(frames) / duration_seconds`` (minimum 1 fps) so the encoded clip
    runs for approximately that long regardless of how many frames/slices
    were loaded.

    Args:
        frames: Ordered list of RGB PIL images, all the same size.
        output_path: Destination ``.mp4`` path.
        duration_seconds: Target clip length in seconds.
        fps: Explicit frame rate (overrides ``duration_seconds`` if both given).
        upsample_factor: Lanczos-upsample each frame by this factor before
            encoding (e.g. 2.0 to pre-compensate for downstream downsampling —
            same rationale as ``_upsample_for_edit``/``_upsample_for_points``
            elsewhere in this codebase).

    Returns:
        The output path.
    """
    if not frames:
        raise ValueError("frames must be a non-empty list")
    if fps is None:
        if duration_seconds is None:
            raise ValueError("Must supply one of duration_seconds or fps")
        fps = max(1.0, len(frames) / duration_seconds)

    import imageio.v3 as iio
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if upsample_factor != 1.0:
        w, h = frames[0].size
        new_size = (round(w * upsample_factor), round(h * upsample_factor))
        frames = [f.convert("RGB").resize(new_size, Image.LANCZOS) for f in frames]
        print(f"  Upsampled frames {w}x{h} -> {new_size[0]}x{new_size[1]} ({upsample_factor}x)")

    arrays = [np.asarray(f.convert("RGB")) for f in frames]
    iio.imwrite(str(output_path), arrays, fps=fps, codec="libx264")
    print(f"  Saved video: {output_path} ({len(frames)} frames @ {fps:.2f} fps)")
    return output_path
