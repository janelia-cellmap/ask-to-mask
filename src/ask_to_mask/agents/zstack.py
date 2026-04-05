"""Z-stack orchestrator: coordinates multi-slice segmentation and refinement."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import OrganelleClass
from .evaluator import EvaluatorAgent
from .gen_backend import ImageGenBackend
from .llm_backend import LLMBackend
from .loop import LoopConfig, LoopResult, run_refinement_loop
from .sam3_backend import SAM3Backend
from .schemas import GenerationParams


@dataclass
class ZStackResult:
    """Output from z-stack refinement."""

    masks: np.ndarray  # (Z, H, W) uint8 or uint16
    per_slice_scores: list[float]
    per_slice_points: dict[int, list[dict]] | None  # raw Molmo points per slice
    total_slices: int


def run_zstack_refinement(
    gen_backend: ImageGenBackend,
    llm_backend: LLMBackend,
    slices: list[Image.Image],
    organelle: OrganelleClass,
    initial_params: GenerationParams,
    config: LoopConfig = LoopConfig(),
    output_dir: Path | None = None,
    instance: bool = False,
    mask_mode: str = "overlay",
    gen_model: str = "",
    resolution_nm: float | None = None,
    llm_model: str = "",
    use_video_predictor: bool = False,
    multi_slice_points: bool = False,
    point_sample: int | None = None,
    point_backend: LLMBackend | None = None,
    point_model: str = "",
    point_prompt: str | None = None,
    z_start: int = 0,
    validate_points: bool = False,
) -> ZStackResult:
    """Run refinement across a z-stack of slices.

    Three modes:
    - Mode A (default): Independent per-slice refinement via ``run_refinement_loop``
    - Mode B: SAM3 video predictor with optional per-slice Molmo points
    - Mode C: Per-slice Molmo points with SAM3 image predictor (no video)
    """
    is_sam3 = isinstance(gen_backend, SAM3Backend)
    per_slice_points: dict[int, list[dict]] | None = None

    # Detect per-slice points if requested (use dedicated point_backend if provided)
    if multi_slice_points and is_sam3:
        pb = point_backend or llm_backend
        pm = point_model or llm_model
        evaluator = EvaluatorAgent(pb, llm_model=pm, point_prompt=point_prompt)
        per_slice_points = evaluator.generate_points_per_slice(
            slices, organelle, sample_count=point_sample,
        )
        print(f"  Point detection found points on {sum(1 for v in per_slice_points.values() if v)} / {len(slices)} slices")

    # Mode B: SAM3 video predictor
    if use_video_predictor and is_sam3:
        return _run_video_predictor(
            gen_backend=gen_backend,
            slices=slices,
            organelle=organelle,
            initial_params=initial_params,
            per_slice_points=per_slice_points,
            output_dir=output_dir,
            instance=instance,
            z_start=z_start,
        )

    # Mode A / Mode C: Independent per-slice processing
    return _run_per_slice(
        gen_backend=gen_backend,
        llm_backend=llm_backend,
        slices=slices,
        organelle=organelle,
        initial_params=initial_params,
        config=config,
        output_dir=output_dir,
        instance=instance,
        mask_mode=mask_mode,
        gen_model=gen_model,
        resolution_nm=resolution_nm,
        llm_model=llm_model,
        per_slice_points=per_slice_points,
        z_start=z_start,
        point_backend=point_backend,
        point_model=point_model,
        validate_points=validate_points,
    )


def _run_per_slice(
    gen_backend: ImageGenBackend,
    llm_backend: LLMBackend,
    slices: list[Image.Image],
    organelle: OrganelleClass,
    initial_params: GenerationParams,
    config: LoopConfig,
    output_dir: Path | None,
    instance: bool,
    mask_mode: str,
    gen_model: str,
    resolution_nm: float | None,
    llm_model: str,
    per_slice_points: dict[int, list[dict]] | None,
    z_start: int,
    point_backend: LLMBackend | None = None,
    point_model: str = "",
    validate_points: bool = False,
) -> ZStackResult:
    """Mode A/C: Process each slice independently via run_refinement_loop."""
    all_masks = []
    all_scores = []

    for i, sl in enumerate(slices):
        z_idx = z_start + i
        slice_dir = output_dir / f"z{z_idx:04d}" if output_dir else None
        print(f"\n--- Slice z={z_idx} ({i+1}/{len(slices)}) ---")

        # If we have per-slice Molmo points, inject them into params
        params = initial_params
        if per_slice_points and i in per_slice_points and per_slice_points[i]:
            from dataclasses import replace
            extra = {**initial_params.extra, "points": per_slice_points[i]}
            params = replace(initial_params, extra=extra)

        result = run_refinement_loop(
            gen_backend=gen_backend,
            llm_backend=llm_backend,
            em_image=sl,
            organelle=organelle,
            initial_params=params,
            config=config,
            output_dir=slice_dir,
            instance=instance,
            mask_mode=mask_mode,
            gen_model=gen_model,
            resolution_nm=resolution_nm,
            llm_model=llm_model,
            point_backend=point_backend,
            point_model=point_model,
            validate_points=validate_points,
        )

        all_masks.append(result.best_result.mask)
        all_scores.append(result.best_evaluation.score)

    # Stack masks into 3D array
    masks_3d = np.stack(all_masks, axis=0)

    # Save summary
    if output_dir:
        summary = {
            "total_slices": len(slices),
            "z_start": z_start,
            "per_slice_scores": all_scores,
            "mean_score": float(np.mean(all_scores)),
        }
        summary_path = output_dir / "zstack_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    return ZStackResult(
        masks=masks_3d,
        per_slice_scores=all_scores,
        per_slice_points=per_slice_points,
        total_slices=len(slices),
    )


def _run_video_predictor(
    gen_backend: SAM3Backend,
    slices: list[Image.Image],
    organelle: OrganelleClass,
    initial_params: GenerationParams,
    per_slice_points: dict[int, list[dict]] | None,
    output_dir: Path | None,
    instance: bool,
    z_start: int,
) -> ZStackResult:
    """Mode B: Use SAM3 video predictor to propagate masks across slices."""
    # Build prompt_frames: which frames get explicit prompts
    prompt_frames: dict[int, dict] = {}

    sam3_strategy = initial_params.extra.get("sam3_strategy", "text")

    if per_slice_points:
        # Feed each slice's detected points as frame-specific prompts
        for slice_idx, points in per_slice_points.items():
            if points:
                fg_points = [p for p in points if p.get("label", 1) == 1]
                bg_points = [p for p in points if p.get("label", 1) == 0]
                if fg_points:
                    coords = [[p["x"], p["y"]] for p in fg_points + bg_points]
                    labels = [1] * len(fg_points) + [0] * len(bg_points)
                    prompt_frames[slice_idx] = {
                        "points": coords,
                        "point_labels": labels,
                    }
    elif sam3_strategy == "text":
        # Text prompt on the middle frame
        mid = len(slices) // 2
        prompt_frames[mid] = {"text": initial_params.prompt}
    else:
        # vlm-coordinate with single set of points from initial_params
        points = initial_params.extra.get("points", [])
        if points:
            mid = len(slices) // 2
            fg_points = [p for p in points if p.get("label", 1) == 1]
            bg_points = [p for p in points if p.get("label", 1) == 0]
            coords = [[p["x"], p["y"]] for p in fg_points + bg_points]
            labels = [1] * len(fg_points) + [0] * len(bg_points)
            prompt_frames[mid] = {"points": coords, "point_labels": labels}

    if not prompt_frames:
        # Fallback: text prompt on middle frame
        mid = len(slices) // 2
        prompt_frames[mid] = {"text": organelle.name}

    print(f"  Video predictor: {len(prompt_frames)} prompted frame(s), {len(slices)} total frames")

    # Run video predictor
    per_frame_masks = gen_backend.generate_zstack(
        slices=slices,
        params=initial_params,
        prompt_frames=prompt_frames,
        instance=instance,
    )

    # Stack into 3D
    masks_3d = np.stack(per_frame_masks, axis=0)

    # Save per-slice outputs: input slice, points overlay, mask, and composite
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, mask in enumerate(per_frame_masks):
            z_idx = z_start + i
            slice_dir = output_dir / f"z{z_idx:04d}"
            slice_dir.mkdir(parents=True, exist_ok=True)

            # Save the raw input slice
            slices[i].save(slice_dir / "input.png")

            # Save input slice with Molmo points overlaid
            if per_slice_points and i in per_slice_points and per_slice_points[i]:
                points_img = _draw_points_on_image(slices[i], per_slice_points[i])
                points_img.save(slice_dir / "points.png")
                print(f"  z{z_idx:04d}: saved input + {len(per_slice_points[i])} points")
            else:
                print(f"  z{z_idx:04d}: saved input (no points)")

            # Save the mask
            if mask.dtype == np.uint16 or instance:
                mask_img = Image.fromarray(mask)
            else:
                mask_img = Image.fromarray(mask, mode="L")
            mask_img.save(slice_dir / f"{organelle.key}_mask.png")

            # Save composite overlay (mask on input)
            composite = _overlay_mask_on_image(slices[i], mask, organelle.rgb)
            composite.save(slice_dir / "composite.png")

        # Save per-slice points if available
        if per_slice_points:
            points_path = output_dir / "per_slice_points.json"
            serializable = {str(k): v for k, v in per_slice_points.items()}
            with open(points_path, "w") as f:
                json.dump(serializable, f, indent=2)

        # Summary
        summary = {
            "total_slices": len(slices),
            "z_start": z_start,
            "mode": "video_predictor",
            "prompted_frames": list(prompt_frames.keys()),
        }
        with open(output_dir / "zstack_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # No per-slice scores for video predictor (it doesn't evaluate)
    per_slice_scores = [0.0] * len(slices)

    return ZStackResult(
        masks=masks_3d,
        per_slice_scores=per_slice_scores,
        per_slice_points=per_slice_points,
        total_slices=len(slices),
    )


def _draw_points_on_image(
    image: Image.Image,
    points: list[dict],
    radius: int = 5,
) -> Image.Image:
    """Draw colored point markers on a copy of the image.

    Foreground points (label=1) are drawn in green, background (label=0) in red.
    """
    from PIL import ImageDraw

    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for p in points:
        x, y = p["x"], p["y"]
        label = p.get("label", 1)
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(255, 255, 255),
        )
    return img


def _overlay_mask_on_image(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay a mask on the image with semi-transparent color."""
    base = np.array(image.convert("RGB")).astype(np.float32)
    fg = mask > 0 if mask.ndim == 2 else mask.any(axis=-1)
    overlay_color = np.array(color, dtype=np.float32)
    base[fg] = base[fg] * (1 - alpha) + overlay_color * alpha
    return Image.fromarray(base.astype(np.uint8))
