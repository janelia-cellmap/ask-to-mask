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
    pre_detected_points: dict[int, list[dict]] | None = None,
) -> ZStackResult:
    """Run refinement across a z-stack of slices.

    Three modes:
    - Mode A (default): Independent per-slice refinement via ``run_refinement_loop``
    - Mode B: SAM3 video predictor with optional per-slice Molmo points
    - Mode C: Per-slice Molmo points with SAM3 image predictor (no video)
    """
    is_sam3 = isinstance(gen_backend, SAM3Backend)
    per_slice_points: dict[int, list[dict]] | None = pre_detected_points

    # Detect per-slice points if requested and not already provided
    if multi_slice_points and is_sam3 and per_slice_points is None:
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


# ---------------------------------------------------------------
# Orthogonal plane processing (XY + XZ + YZ with majority vote)
# ---------------------------------------------------------------


def _parallel_molmo_detection(
    ortho_slices: dict[str, tuple[list[Image.Image], np.ndarray]],
    organelle: OrganelleClass,
    point_backend: LLMBackend,
    point_model: str,
    point_prompt: str | None,
    point_sample: int | None,
) -> dict[str, dict[int, list[dict]]]:
    """Run Molmo point detection for all 3 planes in parallel.

    Spawns 3 concurrent batch Molmo subprocesses (one per plane),
    each loading its own model instance. Requires ~3x Molmo VRAM (~60 GB).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _detect_plane(plane: str) -> tuple[str, dict[int, list[dict]]]:
        slices_list = ortho_slices[plane][0]
        if not slices_list:
            return plane, {}
        print(f"  [parallel] Starting Molmo detection for {plane.upper()} ({len(slices_list)} slices)")
        evaluator = EvaluatorAgent(
            point_backend, llm_model=point_model, point_prompt=point_prompt
        )
        points = evaluator.generate_points_per_slice(
            slices_list, organelle, sample_count=point_sample,
        )
        n_found = sum(1 for v in points.values() if v)
        print(f"  [parallel] {plane.upper()} done: points on {n_found}/{len(slices_list)} slices")
        return plane, points

    planes = [p for p in ("xy", "xz", "yz") if ortho_slices.get(p, ([], None))[0]]
    results: dict[str, dict[int, list[dict]]] = {}

    print(f"\n=== Parallel Molmo detection across {len(planes)} planes ===")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_detect_plane, p): p for p in planes}
        for future in as_completed(futures):
            plane, points = future.result()
            results[plane] = points

    return results


def run_ortho_zstack_refinement(
    gen_backend: ImageGenBackend,
    llm_backend: LLMBackend,
    ortho_slices: dict[str, tuple[list[Image.Image], np.ndarray]],
    volume_shape: tuple[int, int, int],
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
    voxel_size: tuple[float, ...] | None = None,
    offset: tuple[float, ...] | None = None,
    parallel_points: bool = False,
) -> ZStackResult:
    """Run segmentation on 3 orthogonal planes and merge via majority vote.

    Processes XY, XZ, and YZ slices independently through the standard
    z-stack pipeline, then reconstructs 3D masks for each plane and
    combines them: a voxel is foreground if at least 2 of 3 planes agree.

    Args:
        ortho_slices: Dict from ``load_zarr_ortho_slices`` with keys
            ``"xy"``, ``"xz"``, ``"yz"``, each mapping to
            ``(slices, data_3d)``.
        volume_shape: (Z, Y, X) shape of the 3D ROI in voxels.
        parallel_points: If True, run Molmo point detection for all 3
            planes in parallel (requires ~3x Molmo VRAM, ~60 GB).
    """
    nz, ny, nx = volume_shape
    plane_masks_3d: dict[str, np.ndarray] = {}

    # Optionally detect points for all planes in parallel before SAM3
    pre_detected_points: dict[str, dict[int, list[dict]]] = {}
    is_molmo = point_backend is not None and "molmo" in (point_model or "").lower()

    if parallel_points and multi_slice_points and is_molmo:
        pre_detected_points = _parallel_molmo_detection(
            ortho_slices=ortho_slices,
            organelle=organelle,
            point_backend=point_backend,
            point_model=point_model,
            point_prompt=point_prompt,
            point_sample=point_sample,
        )

    for plane in ("xy", "xz", "yz"):
        slices_list = ortho_slices[plane][0]
        if not slices_list:
            print(f"\n=== Skipping {plane.upper()} plane (no slices) ===")
            continue

        plane_dir = output_dir / plane if output_dir else None
        print(f"\n{'='*60}")
        print(f"=== Processing {plane.upper()} plane ({len(slices_list)} slices) ===")
        print(f"{'='*60}")

        # If points were pre-detected in parallel, inject them so
        # run_zstack_refinement skips its own Molmo detection
        plane_params = initial_params
        skip_multi_slice = False
        if plane in pre_detected_points:
            # Inject pre-detected points into per-slice params
            # run_zstack_refinement will see multi_slice_points=True
            # but the evaluator in _run_per_slice will find points
            # already in params.extra["points"] and skip detection
            skip_multi_slice = True

        result = run_zstack_refinement(
            gen_backend=gen_backend,
            llm_backend=llm_backend,
            slices=slices_list,
            organelle=organelle,
            initial_params=plane_params,
            config=config,
            output_dir=plane_dir,
            instance=instance,
            mask_mode=mask_mode,
            gen_model=gen_model,
            resolution_nm=resolution_nm,
            llm_model=llm_model,
            use_video_predictor=use_video_predictor,
            multi_slice_points=multi_slice_points and not skip_multi_slice,
            point_sample=point_sample,
            point_backend=point_backend,
            point_model=point_model,
            point_prompt=point_prompt,
            z_start=z_start if plane == "xy" else 0,
            validate_points=validate_points,
            pre_detected_points=pre_detected_points.get(plane),
        )

        # Reconstruct full 3D mask from per-plane 2D masks
        plane_mask = _reconstruct_3d_mask(
            result.masks, plane, volume_shape, ortho_slices
        )
        plane_masks_3d[plane] = plane_mask

        # Save per-plane zarr immediately so results are visible during the run
        if output_dir:
            from ..zarr_io import save_masks_to_zarr
            plane_zarr = output_dir / "masks.zarr"
            save_masks_to_zarr(
                plane_mask, str(plane_zarr), dataset_name=plane,
                voxel_size=voxel_size, offset=offset,
            )
            print(f"  Saved {plane.upper()} plane mask to {plane_zarr}/{plane}")

    # Majority vote: voxel is foreground if >= 2 planes agree
    combined = _majority_vote(plane_masks_3d, volume_shape)

    # Save combined mask PNGs and zarr
    if output_dir:
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        for z in range(combined.shape[0]):
            mask_img = Image.fromarray(
                (combined[z] > 0).astype(np.uint8) * 255, mode="L"
            )
            mask_img.save(merged_dir / f"z{z_start + z:04d}_mask.png")

        # Save merged result to same zarr as per-plane masks
        from ..zarr_io import save_masks_to_zarr
        save_masks_to_zarr(
            combined, str(output_dir / "masks.zarr"), dataset_name="merged",
            voxel_size=voxel_size, offset=offset,
        )

        summary = {
            "planes_processed": list(plane_masks_3d.keys()),
            "volume_shape": list(volume_shape),
            "merge_strategy": "majority_vote",
        }
        with open(output_dir / "ortho_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return ZStackResult(
        masks=combined,
        per_slice_scores=[0.0] * nz,
        per_slice_points=None,
        total_slices=nz,
    )


def _reconstruct_3d_mask(
    masks_2d: np.ndarray,
    plane: str,
    volume_shape: tuple[int, int, int],
    ortho_slices: dict,
) -> np.ndarray:
    """Map 2D per-slice masks back into a 3D volume.

    The masks_2d array has shape (n_slices, h, w) where n_slices is the
    number of slices sampled along the slicing axis. We place each 2D
    mask back at its original position in the volume.

    Args:
        masks_2d: (n_slices, h, w) mask stack from the pipeline.
        plane: One of ``"xy"``, ``"xz"``, ``"yz"``.
        volume_shape: (Z, Y, X) of the full ROI.
        ortho_slices: The ortho_slices dict with actual slice indices.
    """
    nz, ny, nx = volume_shape
    vol = np.zeros(volume_shape, dtype=np.uint8)

    # Use the actual indices from loading (3rd element of the tuple)
    indices = ortho_slices[plane][2]

    if plane == "xy":
        for i, z in enumerate(indices):
            mask = (masks_2d[i] > 0).astype(np.uint8)
            if mask.shape != (ny, nx):
                from PIL import Image as PILImage
                mask = np.array(
                    PILImage.fromarray(mask * 255).resize((nx, ny), PILImage.NEAREST)
                ) > 0
                mask = mask.astype(np.uint8)
            vol[z] = mask
        vol = _interpolate_between_slices(vol, indices, axis=0)

    elif plane == "xz":
        for i, y in enumerate(indices):
            mask = (masks_2d[i] > 0).astype(np.uint8)
            if mask.shape != (nz, nx):
                from PIL import Image as PILImage
                mask = np.array(
                    PILImage.fromarray(mask * 255).resize((nx, nz), PILImage.NEAREST)
                ) > 0
                mask = mask.astype(np.uint8)
            vol[:, y, :] = mask
        vol = _interpolate_between_slices(vol, indices, axis=1)

    elif plane == "yz":
        for i, x in enumerate(indices):
            mask = (masks_2d[i] > 0).astype(np.uint8)
            if mask.shape != (nz, ny):
                from PIL import Image as PILImage
                mask = np.array(
                    PILImage.fromarray(mask * 255).resize((ny, nz), PILImage.NEAREST)
                ) > 0
                mask = mask.astype(np.uint8)
            vol[:, :, x] = mask
        vol = _interpolate_between_slices(vol, indices, axis=2)

    return vol


def _interpolate_between_slices(
    vol: np.ndarray,
    sampled_indices: list[int],
    axis: int,
) -> np.ndarray:
    """Fill gaps between sampled slices via nearest-neighbor interpolation.

    For each unsampled position along ``axis``, copies the mask from the
    nearest sampled slice.
    """
    n = vol.shape[axis]
    sampled = np.array(sampled_indices)

    for i in range(n):
        if i in sampled_indices:
            continue
        # Find nearest sampled index
        nearest = sampled[np.argmin(np.abs(sampled - i))]
        if axis == 0:
            vol[i] = vol[nearest]
        elif axis == 1:
            vol[:, i, :] = vol[:, nearest, :]
        else:
            vol[:, :, i] = vol[:, :, nearest]

    return vol


def _majority_vote(
    plane_masks: dict[str, np.ndarray],
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Merge masks from multiple planes via majority vote.

    A voxel is foreground if at least 2 of the available planes mark it
    as foreground.
    """
    votes = np.zeros(volume_shape, dtype=np.uint8)
    for mask in plane_masks.values():
        votes += (mask > 0).astype(np.uint8)

    n_planes = len(plane_masks)
    threshold = max(1, (n_planes + 1) // 2)  # majority: 2 of 3
    return (votes >= threshold).astype(np.uint8) * 255
