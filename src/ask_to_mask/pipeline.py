"""Main pipeline: load image → prompt → infer → extract mask → save."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .config import ORGANELLES, OrganelleClass
from .model import run_inference
from .postprocess import extract_instance_mask, extract_mask, save_mask

# Flux models work best at this resolution
TARGET_SIZE = 1024


def load_em_image(path: str | Path) -> Image.Image:
    """Load an EM image and convert to RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pad_to_square(img: Image.Image, fill: int = 0) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Pad image to square, preserving aspect ratio.

    Returns:
        (padded_image, (left, top, right, bottom)) — the crop box to undo padding.
    """
    w, h = img.size
    size = max(w, h)
    left = (size - w) // 2
    top = (size - h) // 2
    padded = Image.new("RGB", (size, size), (fill, fill, fill))
    padded.paste(img, (left, top))
    crop_box = (left, top, left + w, top + h)
    return padded, crop_box


def unpad(img: Image.Image, crop_box: tuple[int, int, int, int], target_square: int) -> Image.Image:
    """Crop the padded region out of an image that was resized from square.

    Args:
        img: The square output image (at target_square x target_square).
        crop_box: (left, top, right, bottom) in the original padded-square coordinate space.
        target_square: The size of the square the image was resized to for inference.
    """
    # crop_box is in original padded-square coords; scale to inference resolution
    orig_square = crop_box[2] - crop_box[0]  # just need to know the padded square size
    # Reconstruct the padded square size from the crop box
    padded_size = max(crop_box[2] + crop_box[0], crop_box[3] + crop_box[1])
    # More robustly: left + right_margin = padded_size - width, so padded_size = left + width + left = 2*left + width
    # Actually we stored crop_box as (left, top, left+w, top+h), padded_size = max(original w, original h)
    # We can recover it: padded_size = max(crop_box[2]-crop_box[0] + 2*crop_box[0], crop_box[3]-crop_box[1] + 2*crop_box[1])
    # Simpler: the image is already target_square x target_square, just scale crop_box
    w_orig = crop_box[2] - crop_box[0]
    h_orig = crop_box[3] - crop_box[1]
    padded_size = max(w_orig + 2 * crop_box[0], h_orig + 2 * crop_box[1])

    scale = target_square / padded_size
    scaled_box = (
        round(crop_box[0] * scale),
        round(crop_box[1] * scale),
        round(crop_box[2] * scale),
        round(crop_box[3] * scale),
    )
    return img.crop(scaled_box)


def overlay_on_raw(raw: Image.Image, colored: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Blend the colored Flux output on top of the raw EM image."""
    raw_rgb = raw.convert("RGB") if raw.mode != "RGB" else raw
    if colored.size != raw_rgb.size:
        colored = colored.resize(raw_rgb.size, Image.LANCZOS)
    return Image.blend(raw_rgb, colored, alpha)


def segment_single(
    pipe,
    image_path: str | Path | None,
    organelle: OrganelleClass,
    output_dir: str | Path,
    model_key: str = "",
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    strength: float = 0.75,
    seed: int | None = None,
    threshold: float = 200.0,
    save_colored: bool = False,
    custom_prompt: str | None = None,
    instance: bool = False,
    detailed_prompt: bool = False,
    resolution_nm: float | None = None,
    image: Image.Image | None = None,
    image_stem: str | None = None,
) -> Path:
    """Run the full pipeline for one image and one organelle class.

    Either ``image_path`` or ``image`` (with ``image_stem``) must be provided.

    Returns:
        Path to the saved mask file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image is not None:
        img = image
        image_stem = image_stem or "slice"
    else:
        image_path = Path(image_path)
        img = load_em_image(image_path)
        image_stem = image_path.stem
    original_size = img.size  # (W, H)

    # Pad to square to preserve aspect ratio, then resize for inference
    img_padded, crop_box = pad_to_square(img)
    img_resized = img_padded.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    # Build prompt
    if custom_prompt:
        prompt = custom_prompt
    elif instance:
        prompt = organelle.build_instance_prompt(detailed_prompt, resolution_nm=resolution_nm)
    else:
        prompt = organelle.build_prompt(detailed_prompt, resolution_nm=resolution_nm)

    print(f"  Prompt: {prompt}")

    # Run model
    result = run_inference(
        pipe,
        img_resized,
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        seed=seed,
    )

    # Crop result back to original aspect ratio (undo padding)
    result_cropped = unpad(result, crop_box, TARGET_SIZE)

    # Organize outputs into subdirectories: output_dir/image_stem/model_key/
    sub_dir = output_dir / image_stem / model_key
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Optionally save the colored intermediate and overlay at original size
    if save_colored:
        colored_out = result_cropped.resize(original_size, Image.LANCZOS)
        suffix = "instance_colored" if instance else "colored"
        colored_path = sub_dir / f"{organelle.key}_{suffix}.png"
        colored_out.save(colored_path)

        # Save overlay: colored output blended on top of raw EM image
        overlay = overlay_on_raw(img, colored_out)
        overlay_path = sub_dir / f"{organelle.key}_overlay.png"
        overlay.save(overlay_path)

    # Extract mask at cropped inference resolution, then resize to original
    # Use the cropped versions for both input and output so they match
    img_resized_cropped = unpad(img_resized, crop_box, TARGET_SIZE)

    if instance:
        mask = extract_instance_mask(
            img_resized_cropped, result_cropped, saturation_threshold=threshold
        )
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize(original_size, Image.NEAREST)
        mask_path = sub_dir / f"{organelle.key}_instances.png"
    else:
        mask = extract_mask(img_resized_cropped, result_cropped, organelle.rgb, threshold=threshold)
        mask_img = Image.fromarray(mask, mode="L")
        mask_img = mask_img.resize(original_size, Image.NEAREST)
        mask_path = sub_dir / f"{organelle.key}_mask.png"

    mask_img.save(mask_path)
    return mask_path


def segment(
    pipe,
    image_path: str | Path | None,
    organelle_keys: list[str],
    output_dir: str | Path,
    **kwargs,
) -> list[Path]:
    """Run segmentation for multiple organelle classes on one image.

    ``image_path`` can be ``None`` if ``image`` and ``image_stem`` are passed
    via ``**kwargs``.

    Returns:
        List of paths to saved mask files.
    """
    masks = []
    for key in organelle_keys:
        organelle = ORGANELLES[key]
        mask_path = segment_single(pipe, image_path, organelle, output_dir, **kwargs)
        masks.append(mask_path)
    return masks
