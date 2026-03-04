"""Extract binary and instance segmentation masks from colored Flux output."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.morphology import opening, closing, disk
from skimage.measure import label


def extract_mask(
    input_image: Image.Image,
    output_image: Image.Image,
    target_rgb: tuple[int, int, int],
    threshold: float = 200.0,
    cleanup: bool = True,
) -> np.ndarray:
    """Extract a binary mask by finding pixels with high saturation in the target color.

    Uses the max channel value in the target color direction: for a red target (255,0,0),
    finds pixels where red is high and dominates over green and blue. This cleanly
    separates saturated colored regions from faint color washes.

    Args:
        input_image: Original EM image (RGB).
        output_image: Flux-edited image with colored organelles (RGB).
        target_rgb: The color used to highlight the organelle, e.g. (255, 0, 0).
        threshold: Minimum value in the target channel(s) to count as colored (0-255).
        cleanup: Apply morphological opening/closing to remove noise.

    Returns:
        Binary mask as uint8 array (0 or 255), same spatial dims as input.
    """
    out = np.array(output_image).astype(np.float32)
    target = np.array(target_rgb, dtype=np.float32)

    # Identify which channels are "on" (>0) and "off" (==0) in the target color
    on_channels = np.where(target > 0)[0]
    off_channels = np.where(target == 0)[0]

    # Score: minimum of the "on" channels minus maximum of the "off" channels.
    # For red (255,0,0): score = R - max(G, B)
    # For yellow (255,255,0): score = min(R, G) - B
    # This gives high scores only for saturated target-colored pixels.
    on_min = np.min(out[:, :, on_channels], axis=-1)

    if len(off_channels) > 0:
        off_max = np.max(out[:, :, off_channels], axis=-1)
        score = on_min - off_max
    else:
        score = on_min

    mask = (score > threshold).astype(np.uint8)

    if cleanup:
        selem = disk(2)
        mask = opening(mask, selem).astype(np.uint8)
        mask = closing(mask, selem).astype(np.uint8)

    return mask * 255


def extract_instance_mask(
    input_image: Image.Image,
    output_image: Image.Image,
    saturation_threshold: float = 50.0,
    cleanup: bool = True,
    min_size: int = 50,
) -> np.ndarray:
    """Extract an instance segmentation mask from a multi-color Flux output.

    Detects any pixel that gained significant color saturation (moved away from
    grayscale), then uses connected components to assign each spatially separate
    colored region a unique integer label.

    Args:
        input_image: Original EM image (RGB).
        output_image: Flux-edited image with each instance a different color (RGB).
        saturation_threshold: Minimum saturation to count as "colored" (0-255 scale).
        cleanup: Apply morphological opening/closing to remove noise.
        min_size: Remove instances smaller than this many pixels.

    Returns:
        Instance label array as uint16 (0 = background, 1..N = instances).
    """
    out = np.array(output_image).astype(np.float32)

    # Detect colored pixels: high saturation means far from grayscale.
    # Saturation = max(R,G,B) - min(R,G,B)
    saturation = np.max(out, axis=-1) - np.min(out, axis=-1)

    colored = (saturation > saturation_threshold).astype(np.uint8)

    if cleanup:
        selem = disk(2)
        colored = opening(colored, selem).astype(np.uint8)
        colored = closing(colored, selem).astype(np.uint8)

    # Label connected components — each spatially separate colored region
    # becomes a unique instance
    labels = label(colored, connectivity=2)

    # Remove small instances
    if min_size > 0:
        for region_id in range(1, labels.max() + 1):
            if np.sum(labels == region_id) < min_size:
                labels[labels == region_id] = 0
        # Re-label to fill gaps in IDs
        labels = label(labels > 0, connectivity=2)

    return labels.astype(np.uint16)


def save_mask(mask: np.ndarray, path: str) -> None:
    """Save a mask as a PNG image. Handles both binary (uint8) and instance (uint16) masks."""
    if mask.dtype == np.uint16:
        Image.fromarray(mask).save(path)
    else:
        Image.fromarray(mask, mode="L").save(path)
