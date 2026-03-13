"""Detect colored markers painted by a generative model and extract their centroids.

Used by the SAM3 painted-marker strategy to convert generative model output
(dots painted on an EM image) into point prompts for SAM3.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def detect_colored_markers(
    em_image: Image.Image,
    marked_image: Image.Image,
    marker_color_rgb: tuple[int, int, int],
    min_area: int = 5,
    max_area: int = 2000,
    saturation_threshold: float = 30.0,
) -> list[dict]:
    """Detect colored markers by comparing the EM image with the marked image.

    Finds regions where the marked image differs from the EM image in the
    direction of the target color, then extracts centroids of those regions.

    Args:
        em_image: Original EM image.
        marked_image: Image with colored markers painted on it.
        marker_color_rgb: Expected marker color (R, G, B).
        min_area: Minimum marker area in pixels.
        max_area: Maximum marker area in pixels.
        saturation_threshold: Minimum color score to count as a marker.

    Returns:
        List of point dicts: [{"x": int, "y": int, "label": 1}, ...]
    """
    from scipy import ndimage

    em = np.array(em_image.convert("RGB"), dtype=np.float32)
    marked = np.array(marked_image.convert("RGB"), dtype=np.float32)

    # Compute color difference
    diff = marked - em

    # Score each pixel by how much it matches the target color direction
    r, g, b = marker_color_rgb
    on_channels = [i for i, v in enumerate([r, g, b]) if v > 128]
    off_channels = [i for i, v in enumerate([r, g, b]) if v <= 128]

    if on_channels and off_channels:
        # Score = min(on_channel diffs) - max(off_channel diffs)
        on_min = np.min(diff[:, :, on_channels], axis=2)
        off_max = np.max(np.abs(diff[:, :, off_channels]), axis=2)
        score = on_min - off_max
    elif on_channels:
        score = np.min(diff[:, :, on_channels], axis=2)
    else:
        # Fallback: use overall difference magnitude
        score = np.linalg.norm(diff, axis=2)

    # Threshold to get marker regions
    binary = score > saturation_threshold

    # Label connected components
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return []

    # Extract centroids, filtering by area
    points = []
    for region_id in range(1, num_features + 1):
        region_mask = labeled == region_id
        area = region_mask.sum()
        if area < min_area or area > max_area:
            continue
        # Compute centroid
        ys, xs = np.where(region_mask)
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        points.append({"x": cx, "y": cy, "label": 1})

    return points
