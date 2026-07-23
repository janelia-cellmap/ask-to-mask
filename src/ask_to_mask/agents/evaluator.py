"""Combined evaluator agent: critiques mask quality AND refines the prompt in one call."""

from __future__ import annotations

import json
import re

from PIL import Image

from ..config import OrganelleClass
from .llm_backend import LLMBackend
from .schemas import (
    DetailedScores,
    EvaluationResult,
    GenerationParams,
    GenerationResult,
    PointRefinement,
)

SYSTEM_PROMPT = """\
We generate segmentation masks from electron microscopy (EM) images by prompting an \
image-editing vision model. The output should be a mask image — colored organelle regions \
on a black background. The original EM image does NOT need to be preserved. You evaluate \
the result and write improved prompts for the generator.

You receive two images: the FIRST image is the original EM, the SECOND image is the mask output.

Compare the two images carefully:
1. Count how many organelles are CORRECTLY colored in the mask vs how many are visible in the EM
2. Check if NON-organelle areas are colored (false positives / background bleed)
3. Check if the coloring is precise (tight boundaries) or sloppy (bleeds into surroundings)

You MUST provide detailed sub-scores (all 0.0 to 1.0):
- tp_rate: fraction of real organelles that are correctly colored (1.0 = all colored)
- fp_rate: fraction of colored pixels that are on NON-organelle areas (0.0 = no false positives, 1.0 = all colored area is wrong)
- fn_rate: fraction of real organelles that were MISSED / not colored (0.0 = none missed, 1.0 = all missed)
- boundary_quality: how precise/tight the coloring boundaries are (1.0 = pixel-perfect, 0.0 = very sloppy bleed)
- dice_score: estimated overall quality = 2*TP / (2*TP + FP + FN). This is your best estimate of segmentation overlap.

The "score" field should equal the dice_score.

When score < 0.85, suggest a refined_prompt (under 100 words). The prompt should ask the \
generator to create a segmentation mask — colored organelles on black background. Be \
descriptive about what the organelle looks like in the EM image to help the generator \
identify the correct structures. The generator responds to clear, concrete instructions, \
not adjectives about quality or precision.
Only add structural descriptions if the model is coloring the wrong thing or missing structures.

DO NOT suggest param_adjustments — the loop controls parameters automatically. Set param_adjustments to {}.

Respond with ONLY JSON:
{"score": 0.0-1.0, "detailed_scores": {"tp_rate": 0.8, "fp_rate": 0.1, "fn_rate": 0.2, "boundary_quality": 0.7, "dice_score": 0.6}, "issues": ["specific issue"], "refined_prompt": "new prompt", "param_adjustments": {}, "should_stop": false, "reasoning": "brief reason"}\
"""

INSTANCE_SYSTEM_PROMPT = """\
We generate instance segmentation masks from electron microscopy (EM) images by prompting \
an image-editing vision model. The output should be a mask image — each organelle instance \
colored a different unique color on a black background. The original EM image does NOT need \
to be preserved. You evaluate the result and write improved prompts for the generator.

You receive two images: the FIRST image is the original EM, the SECOND image is the mask output.

In instance segmentation, each separate instance of the organelle should be colored a DIFFERENT unique color.
Adjacent/touching instances must have different colors to distinguish them.

Compare the two images carefully:
1. Count how many individual instances are visible in the EM vs how many are uniquely colored in the mask
2. Check if adjacent instances have DIFFERENT colors (they must be distinguishable)
3. Check if NON-organelle areas are colored (false positives)
4. Check if the coloring boundaries are precise

You MUST provide detailed sub-scores (all 0.0 to 1.0):
- tp_rate: fraction of real instances that are correctly colored with a unique color (1.0 = all instances colored)
- fp_rate: fraction of colored pixels that are on NON-organelle areas (0.0 = no false positives)
- fn_rate: fraction of real instances that were MISSED / not colored (0.0 = none missed)
- boundary_quality: how precise the boundaries are AND how well adjacent instances are separated by different colors (1.0 = perfect separation)
- dice_score: estimated overall quality = 2*TP / (2*TP + FP + FN)

The "score" field should equal the dice_score.

When score < 0.85, suggest a refined_prompt (under 100 words). The prompt should ask the \
generator to create an instance segmentation mask — each instance a different unique color \
on black background. Be descriptive about what the organelle looks like in the EM image to \
help the generator identify the correct structures. The generator responds to clear, \
concrete instructions, not adjectives about quality or precision.

DO NOT suggest param_adjustments — the loop controls parameters automatically. Set param_adjustments to {}.

Respond with ONLY JSON:
{"score": 0.0-1.0, "detailed_scores": {"tp_rate": 0.8, "fp_rate": 0.1, "fn_rate": 0.2, "boundary_quality": 0.7, "dice_score": 0.6}, "issues": ["specific issue"], "refined_prompt": "new prompt", "param_adjustments": {}, "should_stop": false, "reasoning": "brief reason"}\
"""

INITIAL_PROMPT_SYSTEM = """\
We generate segmentation masks from electron microscopy (EM) images by prompting an \
image-editing vision model. The output should be a mask image — the original EM image \
does NOT need to be preserved. The goal is colored organelle regions on a black background.

You will see the EM image and must write an optimal prompt for the generator model.

Write a clear, concrete prompt (under 100 words) that tells the model to create a \
segmentation mask: color the target organelles the specified color and make everything \
else black. Be descriptive about what the organelle looks like in the EM image so the \
generator can identify the correct structures. The generator responds to clear \
instructions, not adjectives about quality or precision.

Respond with ONLY JSON:
{"prompt": "your prompt here", "reasoning": "why this prompt"}\
"""


_PIXEL_COORD_INSTR = (
    "Provide coordinates as (x, y) pixel coordinates in the image shown. The origin (0, 0) is "
    "the top-left corner. x increases to the right, y increases downward."
)
_NORMALIZED_COORD_INSTR = (
    "Provide coordinates as (x, y) normalized to a 0-1000 scale, NOT raw pixels: (0, 0) is the "
    "top-left corner and (1000, 1000) is the bottom-right corner, regardless of the image's "
    "actual pixel dimensions. x increases to the right, y increases downward."
)


VLM_INITIAL_POINTS_PROMPT = """\
We segment organelles in electron microscopy (EM) images using SAM3 (Segment Anything Model 3) \
with point prompts. You identify points inside target organelles in the EM \
image so we can feed those coordinates to SAM3.

You will see the EM image. Identify each visible instance of the target organelle and provide \
one or more points inside it. {coord_instr}

Provide foreground points (label=1) inside organelles. If there are obvious non-organelle \
regions that might confuse the model, you may also add background points (label=0).

Assign each foreground point an instance ID (integer). Points with the same instance ID will be \
used to segment the same object. Each distinct organelle should have a unique instance ID \
(starting from 0). Background points (label=0) do not need an instance ID.

Respond with ONLY JSON:
{"points": [{"x": 100, "y": 200, "label": 1, "instance": 0}, {"x": 300, "y": 400, "label": 1, "instance": 1}], "reasoning": "why these points"}\
"""

VLM_INITIAL_REGIONS_PROMPT = """\
We segment organelles in electron microscopy (EM) images using SAM3 (Segment Anything Model 3) \
with VLM-provided geometric coordinates. You identify target organelles in the EM image and \
provide coordinates we can use as segmentation prompts and downstream cleanup metadata.

You will see the EM image. Identify each visible instance of the target organelle. {coord_instr}

For each instance, provide the requested geometry using that same coordinate scale:
- points: one or more foreground points inside the organelle, used as SAM3 point prompts
- bbox: a tight bounding box around the organelle as x1, y1, x2, y2, usable as a SAM3 box prompt
- polygon: a polygon surrounding the organelle, with vertices ordered around the boundary; this is \
saved for downstream cleanup/evaluation rather than passed directly to SAM3

Each distinct organelle should have a unique instance ID starting from 0. If requested geometry \
is uncertain, still provide at least one point inside the organelle. You may also add background \
points (label=0) outside organelles if they help avoid confusing non-organelle structures.

Respond with ONLY JSON:
{"instances": [{"instance": 0, "points": [{"x": 100, "y": 200}], "bbox": {"x1": 80, "y1": 170, "x2": 150, "y2": 230}, "polygon": [{"x": 82, "y": 190}, {"x": 110, "y": 172}, {"x": 148, "y": 205}, {"x": 118, "y": 228}]}], "background_points": [{"x": 20, "y": 20, "label": 0}], "reasoning": "why these regions"}\
"""

SAM3_COORDINATE_SYSTEM_PROMPT = """\
We segment organelles in electron microscopy (EM) images using SAM3 (Segment Anything Model 3) \
with point prompts. You evaluate the segmentation result and suggest point coordinate adjustments.

You receive two images: the FIRST is the original EM, the SECOND is the SAM3 mask output \
(organelles colored on the EM background).

You also receive the current point coordinates used for this iteration.

Compare the two images carefully:
1. Count how many organelles are CORRECTLY segmented vs how many are visible in the EM
2. Check if NON-organelle areas are included (false positives)
3. Check if boundaries are precise

You MUST provide detailed sub-scores (all 0.0 to 1.0):
- tp_rate, fp_rate, fn_rate, boundary_quality, dice_score (same as standard evaluation)

Additionally, suggest point refinements:
- add_points: new foreground (label=1) or background (label=0) points to improve the mask. \
For foreground points, include an instance ID matching an existing instance or a new unique ID \
for a newly detected organelle. Background points do not need an instance ID.
- remove_indices: indices (0-based) of existing points that are causing problems

Respond with ONLY JSON:
{"score": 0.0-1.0, "detailed_scores": {"tp_rate": 0.8, "fp_rate": 0.1, "fn_rate": 0.2, "boundary_quality": 0.7, "dice_score": 0.6}, "issues": ["specific issue"], "refined_prompt": null, "param_adjustments": {}, "should_stop": false, "reasoning": "brief reason", "point_refinement": {"add_points": [{"x": 150, "y": 250, "label": 1, "instance": 2}], "remove_indices": [], "reasoning": "why these changes"}}\
"""


MOLMO_POINTS_PROMPT = "Point inside each {organelle} in this EM image."

POINT_VALIDATION_PROMPT = """\
You are validating whether a marked location in an electron microscopy (EM) image \
correctly identifies a specific organelle. A red circle with crosshair marks the \
location being evaluated.

Look at the marked location and determine if it is on or very close to the target \
organelle. Consider the surrounding context and morphology.

Respond with ONLY JSON:
{"valid": true, "reasoning": "brief reason"}
or
{"valid": false, "reasoning": "brief reason"}\
"""


def _upsample_for_points(image: Image.Image, floor: int = 768) -> tuple[Image.Image, float]:
    """Upsample a crop before sending it to a VLM for point/box/polygon detection.

    Small crops sent as-is can end up below a VLM's internal working resolution,
    which means the API silently upsamples with an interpolation method we don't
    control and effectively coarsens the coordinate grid the model reasons over.
    Upsampling ourselves (Lanczos) beforehand caps how much detail that internal
    resize can destroy. Returns the (possibly resized) image and the scale factor
    applied — divide any pixel coordinates the VLM returns for the resized image
    by this factor to map back to the original image's pixel space.
    """
    w, h = image.size
    scale = max(1.0, floor / min(w, h))
    if scale == 1.0:
        return image, 1.0
    new_size = (round(w * scale), round(h * scale))
    return image.resize(new_size, Image.LANCZOS), scale


class EvaluatorAgent:
    """Evaluates mask quality and refines prompts using a VLM backend."""

    def __init__(
        self,
        backend: LLMBackend,
        instance: bool = False,
        gen_model: str = "",
        resolution_nm: float | None = None,
        llm_model: str = "",
        point_prompt: str | None = None,
        point_backend: LLMBackend | None = None,
        point_model: str = "",
        point_shape_mode: str = "points",
    ):
        self.backend = backend
        self.instance = instance
        self.gen_model = gen_model
        self.resolution_nm = resolution_nm
        self.point_prompt = point_prompt
        # Dedicated point detection backend/model (falls back to eval backend)
        self.point_backend = point_backend or backend
        self.point_model = point_model or llm_model
        self.point_shape_mode = point_shape_mode.lower()
        self.is_molmo = "molmo" in self.point_model.lower()
        # Stores VLM prompts used for the most recent initial generation call
        self.last_init_vlm_prompts: dict | None = None

    def generate_initial_prompt(
        self,
        em_image: Image.Image,
        organelle: OrganelleClass,
        mask_mode: str = "overlay",
    ) -> str:
        """Ask the VLM to write an optimal first prompt given the EM image."""
        parts = [f"Target organelle: {organelle.name} (color: {organelle.color_name})."]
        if organelle.description:
            parts.append(f"In EM, {organelle.name} appear as: {organelle.description}")
        if self.resolution_nm is not None:
            parts.append(f"Image resolution: {self.resolution_nm:.0f} nm/px.")
        if self.gen_model:
            parts.append(f"Generator model: {self.gen_model}.")
        parts.append("The output is a segmentation mask — the original EM image does NOT need to be preserved.")
        if mask_mode == "direct":
            parts.append("Mask format: white organelle regions on black background.")
        elif mask_mode == "invert":
            parts.append("Mask format: background/edges white on black, organelle interiors black.")
        elif self.instance:
            parts.append("Mask format: each instance a different unique bright color on black background.")
        else:
            parts.append(f"Mask format: {organelle.name} in {organelle.color_name}, everything else black.")
        parts.append("\nLook at the EM image and write the best prompt. Respond with JSON only.")
        user_prompt = "\n".join(parts)

        raw = self.backend.chat_with_image(INITIAL_PROMPT_SYSTEM, user_prompt, em_image)
        self.last_init_vlm_prompts = {
            "system": INITIAL_PROMPT_SYSTEM,
            "user": user_prompt,
            "raw_response": raw[:2000],
        }
        return self._parse_initial_prompt(raw, organelle)

    def _parse_initial_prompt(self, raw: str, organelle: OrganelleClass) -> str:
        """Extract the prompt from the VLM's initial prompt response."""
        text = raw[:4000]
        json_str = self._extract_json_object(text)
        if json_str:
            try:
                parsed = json.loads(json_str)
                prompt = parsed.get("prompt", "")
                if prompt:
                    print(f"  VLM initial prompt reasoning: {parsed.get('reasoning', '')}")
                    return prompt
            except json.JSONDecodeError:
                pass
        # Try regex fallback
        match = re.search(r'"prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if match:
            return match.group(1)
        # Fallback to default
        print("  Warning: could not parse VLM initial prompt, using default")
        return organelle.build_prompt(detailed=False, resolution_nm=self.resolution_nm)

    def evaluate_and_refine(
        self,
        em_image: Image.Image,
        result: GenerationResult,
        organelle: OrganelleClass,
        history: list[tuple[GenerationParams, EvaluationResult]] | None = None,
    ) -> EvaluationResult:
        """Critique the mask AND produce a refined prompt in one VLM call."""
        user_prompt = self._build_user_prompt(result, organelle, history)
        system = INSTANCE_SYSTEM_PROMPT if self.instance else SYSTEM_PROMPT
        raw = self.backend.chat_with_images(
            system, user_prompt, [em_image, result.colored_image]
        )
        eval_result = self._parse_response(raw)
        eval_result.vlm_prompts = {"system": system, "user": user_prompt}
        return eval_result

    def generate_initial_points(
        self,
        em_image: Image.Image,
        organelle: OrganelleClass,
    ) -> list[dict]:
        """Ask the VLM to identify organelle locations as point/box/polygon coordinates."""
        if self.is_molmo:
            return self._generate_initial_points_molmo(em_image, organelle)

        send_image, scale = _upsample_for_points(em_image)
        use_normalized = self.point_backend.native_point_scale == "normalized_1000"
        w, h = send_image.size
        parts = [
            f"Target organelle: {organelle.name}.",
            f"Image dimensions: {w} x {h} pixels.",
        ]
        if organelle.description:
            parts.append(f"In EM, {organelle.name} appear as: {organelle.description}")
        if self.resolution_nm is not None:
            parts.append(f"Image resolution: {self.resolution_nm:.0f} nm/px.")
        if self.point_prompt:
            parts.append(f"\nTask: {self.point_prompt}")
        elif self.point_shape_mode == "points":
            parts.append("\nIdentify each visible instance and provide point coordinates inside it.")
        else:
            parts.append(
                "\nIdentify each visible instance and provide the requested geometry "
                f"for mode '{self.point_shape_mode}'."
            )
        parts.append("Respond with JSON only.")
        user_prompt = "\n".join(parts)

        system_prompt = self._initial_points_system_prompt(use_normalized)
        raw = self.point_backend.chat_with_image(system_prompt, user_prompt, send_image)
        self.last_init_vlm_prompts = {
            "system": system_prompt,
            "user": user_prompt,
            "raw_response": raw[:2000],
        }
        return self._parse_initial_points(raw, em_image, scale=scale, normalized=use_normalized)

    def _initial_points_system_prompt(self, use_normalized: bool = False) -> str:
        """Build the system prompt for initial point/box/polygon detection."""
        coord_instr = _NORMALIZED_COORD_INSTR if use_normalized else _PIXEL_COORD_INSTR
        mode = self.point_shape_mode
        if mode == "points":
            return VLM_INITIAL_POINTS_PROMPT.replace("{coord_instr}", coord_instr)
        requested = {
            "boxes": "For each instance, provide bbox and at least one point inside it. polygon may be omitted.",
            "polygons": "For each instance, provide polygon and at least one point inside it. bbox may be omitted.",
            "all": "For each instance, provide points, bbox, and polygon.",
        }.get(mode, "For each instance, provide points inside it.")
        base = VLM_INITIAL_REGIONS_PROMPT.replace("{coord_instr}", coord_instr)
        return f"{base}\n\nRequested geometry mode: {mode}.\n{requested}"

    def generate_points_per_slice(
        self,
        slices: list[Image.Image],
        organelle: OrganelleClass,
        sample_count: int | None = None,
    ) -> dict[int, list[dict]]:
        """Run Molmo on each slice independently to detect organelle points.

        Args:
            slices: List of RGB PIL Images (one per z-slice).
            organelle: Organelle class to detect.
            sample_count: If set, uniformly sample this many slices instead of all.

        Returns:
            Dict mapping slice index -> list of point dicts from Molmo.
            Slices where Molmo finds nothing will have an empty list.
        """
        import numpy as np

        n = len(slices)
        if sample_count and sample_count < n:
            # Uniformly sample slice indices
            indices = np.linspace(0, n - 1, sample_count, dtype=int).tolist()
        else:
            indices = list(range(n))

        # Use batch mode for Molmo native pointing (load model once)
        if self.is_molmo:
            per_slice_points = self._batch_molmo_points(
                [slices[i] for i in indices], indices, organelle
            )
        else:
            per_slice_points: dict[int, list[dict]] = {}
            for idx in indices:
                print(f"  Point detection: slice {idx+1}/{n}")
                try:
                    points = self.generate_initial_points(slices[idx], organelle)
                    per_slice_points[idx] = points
                    print(f"    Found {len(points)} points")
                except Exception as e:
                    print(f"    Failed: {e}")
                    per_slice_points[idx] = []

        # Fill in non-sampled slices with empty lists
        for i in range(n):
            if i not in per_slice_points:
                per_slice_points[i] = []

        return per_slice_points

    def _batch_molmo_points(
        self,
        images: list[Image.Image],
        indices: list[int],
        organelle: OrganelleClass,
    ) -> dict[int, list[dict]]:
        """Run Molmo on multiple images in a single subprocess (batch mode).

        Loads the model once and processes all images sequentially, avoiding
        the per-image model loading overhead.
        """
        import json
        import subprocess
        import tempfile
        from pathlib import Path

        prompt = self.point_prompt or MOLMO_POINTS_PROMPT.format(
            organelle=organelle.name
        )

        project_root = Path(__file__).resolve().parents[3]
        molmo_python = project_root / ".pixi" / "envs" / "molmo" / "bin" / "python"
        script = project_root / "scripts" / "molmo_points.py"

        if not molmo_python.exists():
            raise RuntimeError(
                f"Molmo pixi environment not found at {molmo_python}. "
                "Run: pixi install -e molmo && pixi run -e molmo install-torch-cu126"
            )

        model_name = getattr(self.point_backend, "model_name", "allenai/Molmo2-8B")

        # Save all images to temp files (upsampled if small — see _upsample_for_points)
        # and create a JSON manifest
        tmp_dir = tempfile.mkdtemp()
        tmp_paths = []
        scales = []
        for i, img in enumerate(images):
            send_img, scale = _upsample_for_points(img)
            scales.append(scale)
            tmp_path = Path(tmp_dir) / f"slice_{i}.png"
            send_img.save(tmp_path, format="PNG")
            tmp_paths.append(str(tmp_path))

        manifest_path = Path(tmp_dir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(tmp_paths, f)

        print(f"  Molmo batch: {len(images)} images in single subprocess")

        try:
            result = subprocess.run(
                [
                    str(molmo_python), str(script),
                    "--images", str(manifest_path),
                    "--prompt", prompt,
                    "--model", model_name,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Molmo batch subprocess failed:\n{result.stderr[-1000:]}")

            # Parse JSON array from last line of stdout
            batch_results = json.loads(result.stdout.strip().split("\n")[-1])

            per_slice_points: dict[int, list[dict]] = {}
            for i, (idx, batch_result) in enumerate(zip(indices, batch_results)):
                points = batch_result.get("points", [])
                raw = batch_result.get("raw", "")
                print(f"  Slice {idx+1}: {len(points)} points")
                if points:
                    points = self._rescale_points(points, scales[i], images[i].size)
                    per_slice_points[idx] = self._assign_instance_ids(points)
                else:
                    per_slice_points[idx] = []

            return per_slice_points

        except Exception as e:
            print(f"  Molmo batch failed: {e}, falling back to sequential")
            per_slice_points = {}
            for i, idx in enumerate(indices):
                try:
                    points = self._generate_initial_points_molmo(images[i], organelle)
                    per_slice_points[idx] = points
                except Exception as e2:
                    print(f"    Slice {idx} failed: {e2}")
                    per_slice_points[idx] = []
            return per_slice_points

        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _raw_to_px(value: float, dim: int, scale: float, normalized: bool) -> int:
        """Convert one coordinate value from the VLM's response frame to pixel space.

        `dim` is the target (original image) size along that axis. `normalized`
        coordinates are on a 0-1000 scale regardless of image size; otherwise
        `value` is a pixel coordinate in the (possibly upsampled) image the VLM
        actually saw, so divide by `scale` to map back to the original image.
        """
        px = (value / 1000.0 * dim) if normalized else (value / scale)
        return max(0, min(dim - 1, round(px)))

    @staticmethod
    def _rescale_points(
        points: list[dict], scale: float, orig_size: tuple[int, int]
    ) -> list[dict]:
        """Map pixel points detected on an upsampled image back to the original size."""
        if scale == 1.0:
            return points
        w, h = orig_size
        return [
            {**p, "x": max(0, min(w - 1, round(p["x"] / scale))),
             "y": max(0, min(h - 1, round(p["y"] / scale)))}
            for p in points
        ]

    def _to_pixel_space(
        self, parsed: dict, w: int, h: int, scale: float, normalized: bool
    ) -> None:
        """Convert coordinates in a parsed VLM JSON response to original pixel space, in place.

        No-op when the VLM was sent the image unscaled and responded in plain pixel
        coordinates (the common case for backends without `native_point_scale ==
        "normalized_1000"` and crops already at/above the upsample floor).
        """
        if not normalized and scale == 1.0:
            return

        def conv(d: object) -> None:
            if not isinstance(d, dict):
                return
            for key, dim in (
                ("x", w), ("y", h), ("x1", w), ("y1", h), ("x2", w), ("y2", h),
            ):
                if key in d:
                    d[key] = self._raw_to_px(float(d[key]), dim, scale, normalized)

        for pt in parsed.get("points", []) or []:
            conv(pt)
        for pt in parsed.get("background_points", []) or []:
            conv(pt)
        for inst in (
            parsed.get("instances") or parsed.get("objects") or parsed.get("regions") or []
        ):
            if not isinstance(inst, dict):
                continue
            conv(inst.get("bbox") or inst.get("box"))
            for pt in inst.get("points", []) or inst.get("interior_points", []) or []:
                conv(pt)
            for pt in inst.get("polygon", []) or []:
                conv(pt)

    def _parse_initial_points(
        self,
        raw: str,
        em_image: Image.Image,
        scale: float = 1.0,
        normalized: bool = False,
    ) -> list[dict]:
        """Extract point coordinates from the VLM's response.

        `scale`/`normalized` describe the coordinate frame the VLM actually
        responded in (see `_upsample_for_points` and `native_point_scale`) —
        coordinates are converted back to `em_image`'s original pixel space
        before any clamping/parsing below.
        """
        text = raw[:200000]
        w, h = em_image.size
        json_str = self._extract_json_object(text)
        if json_str:
            try:
                parsed = json.loads(json_str)
                self._to_pixel_space(parsed, w, h, scale, normalized)
                region_points = self._parse_region_instances(parsed, em_image)
                if region_points:
                    reasoning = parsed.get("reasoning", "")
                    if reasoning:
                        print(f"  VLM points reasoning: {reasoning}")
                    return region_points
                points = parsed.get("points", [])
                if points and isinstance(points, list):
                    reasoning = parsed.get("reasoning", "")
                    if reasoning:
                        print(f"  VLM points reasoning: {reasoning}")
                    # Validate and clamp coordinates
                    valid_points = []
                    for i, p in enumerate(points):
                        if isinstance(p, dict) and "x" in p and "y" in p:
                            pt = {
                                "x": max(0, min(w - 1, round(float(p["x"])))),
                                "y": max(0, min(h - 1, round(float(p["y"])))),
                                "label": int(p.get("label", 1)),
                            }
                            # Preserve instance ID; default to index if omitted
                            if pt["label"] == 1:
                                pt["instance"] = int(p.get("instance", i))
                            valid_points.append(pt)
                    return valid_points
            except json.JSONDecodeError:
                pass

        region_points = self._parse_region_instances_from_text(
            text, em_image, scale=scale, normalized=normalized
        )
        if region_points:
            return region_points

        # Fallback: try to extract points with regex (JSON dict format)
        point_matches = re.findall(
            r'\{\s*"x"\s*:\s*(\d+)\s*,\s*"y"\s*:\s*(\d+)(?:\s*,\s*"label"\s*:\s*(\d+))?\s*\}',
            text,
        )
        if point_matches:
            result = []
            for i, m in enumerate(point_matches):
                pt = {
                    "x": self._raw_to_px(float(m[0]), w, scale, normalized),
                    "y": self._raw_to_px(float(m[1]), h, scale, normalized),
                    "label": int(m[2]) if m[2] else 1,
                }
                if pt["label"] == 1:
                    pt["instance"] = i  # default: each point is its own instance
                result.append(pt)
            return result

        # Fallback: tuple format like (10, 65), (15, 60) — e.g. from Molmo text mode
        # Group by line: points on the same line (e.g. "Boat 1: (10, 65), (15, 60)")
        # belong to the same instance
        w, h = em_image.size
        tuple_points: list[dict] = []
        instance_id = 0
        for line in text.split("\n"):
            line_matches = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', line)
            if line_matches:
                for m in line_matches:
                    tuple_points.append({
                        "x": max(0, min(w - 1, int(m[0]))),
                        "y": max(0, min(h - 1, int(m[1]))),
                        "label": 1,
                        "instance": instance_id,
                    })
                instance_id += 1
        if tuple_points:
            print(f"  Parsed {len(tuple_points)} points from tuple format ({instance_id} instances)")
            return tuple_points

        print("  Warning: could not parse VLM points, using image center as fallback")
        return [{"x": w // 2, "y": h // 2, "label": 1, "instance": 0}]

    def _parse_region_instances(self, parsed: dict, em_image: Image.Image) -> list[dict]:
        """Parse richer instance geometry JSON into SAM3 prompt point dicts."""
        instances = parsed.get("instances") or parsed.get("objects") or parsed.get("regions")
        if not isinstance(instances, list):
            return []

        w, h = em_image.size
        result: list[dict] = []
        for i, obj in enumerate(instances):
            if not isinstance(obj, dict):
                continue
            inst_id = int(obj.get("instance", obj.get("id", i)))
            bbox = self._normalize_bbox(obj.get("bbox") or obj.get("box"), w, h)
            polygon = self._normalize_polygon(obj.get("polygon"), w, h)
            raw_points = obj.get("points") or obj.get("interior_points") or []
            fg_points = []
            for p in raw_points:
                pt = self._normalize_point(p, w, h)
                if pt is not None:
                    fg_points.append(pt)

            if not fg_points and polygon:
                fg_points = [self._polygon_centroid(polygon, w, h)]
            if not fg_points and bbox:
                x1, y1, x2, y2 = bbox
                fg_points = [{"x": int(round((x1 + x2) / 2)), "y": int(round((y1 + y2) / 2))}]

            for j, pt in enumerate(fg_points):
                out = {
                    "x": pt["x"],
                    "y": pt["y"],
                    "label": 1,
                    "instance": inst_id,
                }
                if bbox:
                    out["bbox"] = bbox
                if polygon:
                    out["polygon"] = polygon
                result.append(out)

        for p in parsed.get("background_points", []) or []:
            pt = self._normalize_point(p, w, h)
            if pt is not None:
                result.append({"x": pt["x"], "y": pt["y"], "label": 0})

        return result

    def _parse_region_instances_from_text(
        self,
        text: str,
        em_image: Image.Image,
        scale: float = 1.0,
        normalized: bool = False,
    ) -> list[dict]:
        """Recover instance points/boxes/polygons from malformed long JSON-like VLM output.

        Splits the text into per-instance chunks (bounded by successive
        "instance": N markers) and parses each chunk's points/bbox/polygon
        independently — truncation can drop any one of those fields (e.g. cut
        off mid-polygon) without the other fields for that same instance being
        invalid, so requiring all of them together (as a single combined regex
        would) drops instances unnecessarily.
        """
        orig_w, orig_h = em_image.size
        # Clamp in the frame the VLM actually responded in; rescaled to
        # orig_w/orig_h at the end, once all points/boxes/polygons are collected.
        w, h = (1000, 1000) if normalized else (round(orig_w * scale), round(orig_h * scale))
        result: list[dict] = []

        inst_markers = list(re.finditer(r'"instance"\s*:\s*(\d+)', text))
        for i, marker in enumerate(inst_markers):
            inst_id = int(marker.group(1))
            chunk_end = inst_markers[i + 1].start() if i + 1 < len(inst_markers) else len(text)
            chunk = text[marker.end():chunk_end]

            points_match = re.search(r'"points"\s*:\s*(\[[^\]]*\])', chunk, re.DOTALL)
            points = self._parse_jsonish_points(points_match.group(1), w, h) if points_match else []

            bbox_match = re.search(r'"bbox"\s*:\s*(\{[^}]*\}|\[[^\]]*\])', chunk, re.DOTALL)
            bbox = self._parse_jsonish_bbox(bbox_match.group(1), w, h) if bbox_match else None

            polygon_match = re.search(r'"polygon"\s*:\s*(\[[^\]]*\])', chunk, re.DOTALL)
            polygon = self._parse_jsonish_points(polygon_match.group(1), w, h) if polygon_match else []
            polygon = polygon if len(polygon) >= 3 else None

            if not points and bbox:
                x1, y1, x2, y2 = bbox
                points = [
                    {
                        "x": int(round((x1 + x2) / 2)),
                        "y": int(round((y1 + y2) / 2)),
                    }
                ]
            if not points and polygon:
                points = [self._polygon_centroid(polygon, w, h)]

            for pt in points:
                out = {
                    "x": pt["x"],
                    "y": pt["y"],
                    "label": 1,
                    "instance": inst_id,
                }
                if bbox:
                    out["bbox"] = bbox
                if polygon:
                    out["polygon"] = polygon
                result.append(out)

        bg_match = re.search(r'"background_points"\s*:\s*(\[[^\]]*\])', text, re.DOTALL)
        if bg_match:
            for pt in self._parse_jsonish_points(bg_match.group(1), w, h):
                result.append({"x": pt["x"], "y": pt["y"], "label": 0})

        if result:
            n_boxes = sum(1 for p in result if p.get("bbox"))
            n_polys = sum(1 for p in result if p.get("polygon"))
            print(
                f"  Parsed {len(result)} geometry prompts from malformed JSON "
                f"({n_boxes} with boxes, {n_polys} with polygons)"
            )
            if normalized or scale != 1.0:
                for item in result:
                    item["x"] = self._raw_to_px(item["x"], orig_w, scale, normalized)
                    item["y"] = self._raw_to_px(item["y"], orig_h, scale, normalized)
                    if item.get("bbox"):
                        x1, y1, x2, y2 = item["bbox"]
                        item["bbox"] = [
                            self._raw_to_px(x1, orig_w, scale, normalized),
                            self._raw_to_px(y1, orig_h, scale, normalized),
                            self._raw_to_px(x2, orig_w, scale, normalized),
                            self._raw_to_px(y2, orig_h, scale, normalized),
                        ]
                    if item.get("polygon"):
                        item["polygon"] = [
                            {
                                "x": self._raw_to_px(p["x"], orig_w, scale, normalized),
                                "y": self._raw_to_px(p["y"], orig_h, scale, normalized),
                            }
                            for p in item["polygon"]
                        ]
        return result

    def _parse_jsonish_points(self, text: str, w: int, h: int) -> list[dict]:
        points: list[dict] = []
        for x, y in re.findall(r'"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"y"\s*:\s*(-?\d+(?:\.\d+)?)', text):
            points.append(
                {
                    "x": max(0, min(w - 1, int(round(float(x))))),
                    "y": max(0, min(h - 1, int(round(float(y))))),
                }
            )
        return points

    def _parse_jsonish_bbox(self, text: str, w: int, h: int) -> list[int] | None:
        try:
            return self._normalize_bbox(json.loads(text), w, h)
        except json.JSONDecodeError:
            pairs = dict(re.findall(r'"(x1|y1|x2|y2|x|y|width|height)"\s*:\s*(-?\d+(?:\.\d+)?)', text))
            if pairs:
                return self._normalize_bbox({k: float(v) for k, v in pairs.items()}, w, h)
        return None

    @staticmethod
    def _normalize_point(point: object, w: int, h: int) -> dict | None:
        if isinstance(point, dict) and "x" in point and "y" in point:
            x, y = point["x"], point["y"]
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            return None
        return {
            "x": max(0, min(w - 1, int(round(float(x))))),
            "y": max(0, min(h - 1, int(round(float(y))))),
        }

    @staticmethod
    def _normalize_bbox(box: object, w: int, h: int) -> list[int] | None:
        if isinstance(box, dict):
            if all(k in box for k in ("x1", "y1", "x2", "y2")):
                vals = [box["x1"], box["y1"], box["x2"], box["y2"]]
            elif all(k in box for k in ("x", "y", "width", "height")):
                vals = [box["x"], box["y"], float(box["x"]) + float(box["width"]), float(box["y"]) + float(box["height"])]
            else:
                return None
        elif isinstance(box, (list, tuple)) and len(box) >= 4:
            vals = list(box[:4])
        else:
            return None

        x1, y1, x2, y2 = [float(v) for v in vals]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        return [
            max(0, min(w - 1, int(round(x1)))),
            max(0, min(h - 1, int(round(y1)))),
            max(0, min(w - 1, int(round(x2)))),
            max(0, min(h - 1, int(round(y2)))),
        ]

    def _normalize_polygon(self, polygon: object, w: int, h: int) -> list[dict] | None:
        if not isinstance(polygon, list):
            return None
        pts = [self._normalize_point(p, w, h) for p in polygon]
        pts = [p for p in pts if p is not None]
        return pts if len(pts) >= 3 else None

    @staticmethod
    def _polygon_centroid(polygon: list[dict], w: int, h: int) -> dict:
        x = sum(p["x"] for p in polygon) / len(polygon)
        y = sum(p["y"] for p in polygon) / len(polygon)
        return {
            "x": max(0, min(w - 1, int(round(x)))),
            "y": max(0, min(h - 1, int(round(y)))),
        }

    def _generate_initial_points_molmo(
        self, em_image: Image.Image, organelle: OrganelleClass
    ) -> list[dict]:
        """Use Molmo's native pointing capability to locate organelles.

        Runs Molmo2 in a subprocess using the 'molmo' pixi environment
        (pinned to transformers <5 for compatibility).
        """
        import json
        import subprocess
        import tempfile
        from pathlib import Path

        prompt = self.point_prompt or MOLMO_POINTS_PROMPT.format(
            organelle=organelle.name
        )

        self.last_init_vlm_prompts = {
            "system": "",
            "user": prompt,
        }

        # Save image to temp file for subprocess (upsampled if small — see _upsample_for_points)
        send_image, scale = _upsample_for_points(em_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            send_image.save(f, format="PNG")
            tmp_path = f.name

        try:
            # Find the molmo env python and script
            project_root = Path(__file__).resolve().parents[3]
            molmo_python = project_root / ".pixi" / "envs" / "molmo" / "bin" / "python"
            script = project_root / "scripts" / "molmo_points.py"

            if not molmo_python.exists():
                raise RuntimeError(
                    f"Molmo pixi environment not found at {molmo_python}. "
                    "Run: pixi install -e molmo && pixi run -e molmo install-torch-cu126"
                )

            model_name = getattr(self.backend, "model_name", "allenai/Molmo2-8B")

            result = subprocess.run(
                [
                    str(molmo_python), str(script),
                    "--image", tmp_path,
                    "--prompt", prompt,
                    "--model", model_name,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Molmo subprocess failed:\n{result.stderr[-1000:]}")

            # Parse JSON from last line of stdout (skip any warnings on stderr)
            output = json.loads(result.stdout.strip().split("\n")[-1])
            raw = output.get("raw", "")
            points = output.get("points", [])

            self.last_init_vlm_prompts["raw_response"] = raw[:2000]
            print(f"  Molmo raw response: {raw[:500]}")

            if points:
                print(f"  Molmo detected {len(points)} points")
                points = self._rescale_points(points, scale, em_image.size)
                return self._assign_instance_ids(points)

        except Exception as e:
            print(f"  Molmo subprocess error: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Fallback to image center
        w, h = em_image.size
        print("  Warning: could not get Molmo points, using image center as fallback")
        return [{"x": w // 2, "y": h // 2, "label": 1, "instance": 0}]

    def _parse_molmo_points(self, raw: str, em_image: Image.Image) -> list[dict]:
        """Parse Molmo2's point output to pixel coordinates.

        Molmo2 uses a coords-based format with coordinates scaled by 1000:
          <points ... coords="1 0 523 412"/>  (frame_id idx x*1000 y*1000)
        Also handles legacy Molmo1 format (0-100 normalized):
          <point x="56.2" y="32.7" alt="desc">
          <points x1="26.0" y1="67.5" x2="44.2" y2="40.5" ...>

        Each point is assigned its own unique instance ID.
        """
        w, h = em_image.size
        points: list[dict] = []

        # Molmo2 coords format: <points ... coords="1 0 523 412"/> or similar
        # Pattern: "idx x y" where x,y are 3-4 digit numbers (scaled by 1000)
        coord_regex = re.compile(r'coords="([^"]+)"')
        points_num_regex = re.compile(r"(\d+)\s+(\d{3,4})\s+(\d{3,4})")
        for coord_match in coord_regex.finditer(raw):
            coord_str = coord_match.group(1)
            for m in points_num_regex.finditer(coord_str):
                px_x = max(0, min(w - 1, int(float(m.group(2)) / 1000 * w)))
                px_y = max(0, min(h - 1, int(float(m.group(3)) / 1000 * h)))
                points.append({"x": px_x, "y": px_y, "label": 1})

        # Fallback: bare coordinate triplets (when coords=" prefix was stripped)
        if not points:
            bare_match = re.search(r'((?:\d+\s+\d{3,4}\s+\d{3,4}\s*)+)"?\s*>', raw)
            if bare_match:
                for m in points_num_regex.finditer(bare_match.group(1)):
                    px_x = max(0, min(w - 1, int(float(m.group(2)) / 1000 * w)))
                    px_y = max(0, min(h - 1, int(float(m.group(3)) / 1000 * h)))
                    points.append({"x": px_x, "y": px_y, "label": 1})

        if points:
            print(f"  Molmo2 detected {len(points)} points (coords format)")
            return self._assign_instance_ids(points)

        # Legacy Molmo1: multi-point format <points x1="26.0" y1="67.5" ...>
        points_tag = re.search(r"<points\s+([^>]+)>", raw)
        if points_tag:
            attrs = points_tag.group(1)
            xs = re.findall(r'x(\d+)\s*=\s*"([^"]+)"', attrs)
            ys = re.findall(r'y(\d+)\s*=\s*"([^"]+)"', attrs)
            y_map = {idx: val for idx, val in ys}
            for idx, x_val in xs:
                if idx in y_map:
                    px_x = max(0, min(w - 1, int(float(x_val) * w / 100)))
                    px_y = max(0, min(h - 1, int(float(y_map[idx]) * h / 100)))
                    points.append({"x": px_x, "y": px_y, "label": 1})

        # Legacy Molmo1: single-point format <point x="56.2" y="32.7" ...>
        for m in re.finditer(r'<point\s+x\s*=\s*"([^"]+)"\s+y\s*=\s*"([^"]+)"', raw):
            px_x = max(0, min(w - 1, int(float(m.group(1)) * w / 100)))
            px_y = max(0, min(h - 1, int(float(m.group(2)) * h / 100)))
            points.append({"x": px_x, "y": px_y, "label": 1})

        if points:
            print(f"  Molmo detected {len(points)} points (legacy format)")
            return self._assign_instance_ids(points)

        print("  Warning: could not parse Molmo points, using image center as fallback")
        print(f"  Raw response: {raw[:500]}")
        return [{"x": w // 2, "y": h // 2, "label": 1, "instance": 0}]

    @staticmethod
    def _assign_instance_ids(points: list[dict]) -> list[dict]:
        """Assign each point its own unique instance ID."""
        return [{**p, "instance": i} for i, p in enumerate(points)]

    def validate_points(
        self,
        em_image: Image.Image,
        points: list[dict],
        organelle: OrganelleClass,
    ) -> list[dict]:
        """Validate each point by marking it on the full image and asking the eval VLM.

        For each point, draws a red circle + crosshair on a copy of the EM image
        and asks the eval VLM whether the marked location is on the target organelle.
        Returns only the points that the VLM confirms as valid.
        """
        import json
        from PIL import ImageDraw

        if not points:
            return points

        validated = []
        marker_radius = max(8, min(em_image.size) // 40)

        for i, pt in enumerate(points):
            x, y = pt["x"], pt["y"]
            label = pt.get("label", 1)

            # Skip background points — only validate foreground
            if label == 0:
                validated.append(pt)
                continue

            # Draw marker on a copy of the EM image
            marked = em_image.copy().convert("RGB")
            draw = ImageDraw.Draw(marked)
            r = marker_radius
            # Red circle
            draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=3)
            # Crosshair
            draw.line([x - r, y, x + r, y], fill="red", width=2)
            draw.line([x, y - r, x, y + r], fill="red", width=2)

            # Upsample small crops so the marker/organelle boundary stays legible
            # to the VLM's own internal resize (no coordinates come back here,
            # so no scale bookkeeping is needed beyond the text description below).
            send_marked, vscale = _upsample_for_points(marked)
            send_x, send_y = round(x * vscale), round(y * vscale)

            # Build user prompt
            parts = [
                f"The red marker is at pixel ({send_x}, {send_y}) in this "
                f"{send_marked.size[0]}x{send_marked.size[1]} EM image.",
                f"Target organelle: {organelle.name}.",
            ]
            if organelle.description:
                parts.append(f"In EM, {organelle.name} appear as: {organelle.description}")
            if self.resolution_nm is not None:
                parts.append(f"Image resolution: {self.resolution_nm:.0f} nm/px.")
            parts.append("Is the red marker correctly placed on this organelle? Respond with JSON only.")
            user_prompt = "\n".join(parts)

            try:
                raw = self.backend.chat_with_image(
                    POINT_VALIDATION_PROMPT, user_prompt, send_marked
                )
                # Parse response
                json_str = raw.strip()
                # Handle markdown code blocks
                if "```" in json_str:
                    json_str = json_str.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    json_str = json_str.strip()
                result = json.loads(json_str)
                is_valid = result.get("valid", True)
                reasoning = result.get("reasoning", "")
            except Exception as e:
                # If parsing fails, keep the point (conservative)
                print(f"  Point [{i}] ({x}, {y}): validation parse error ({e}), keeping")
                validated.append(pt)
                continue

            status = "VALID" if is_valid else "REJECTED"
            print(f"  Point [{i}] ({x}, {y}): {status} — {reasoning}")

            if is_valid:
                validated.append(pt)

        accepted = sum(1 for p in validated if p.get("label", 1) == 1)
        total_fg = sum(1 for p in points if p.get("label", 1) == 1)
        print(f"  Point validation: {accepted}/{total_fg} foreground points accepted")
        return validated

    def evaluate_and_refine_with_points(
        self,
        em_image: Image.Image,
        result: GenerationResult,
        organelle: OrganelleClass,
        history: list[tuple[GenerationParams, EvaluationResult]] | None = None,
    ) -> EvaluationResult:
        """Evaluate SAM3 mask and suggest point coordinate refinements."""
        user_prompt = self._build_user_prompt_with_points(result, organelle, history)
        raw = self.backend.chat_with_images(
            SAM3_COORDINATE_SYSTEM_PROMPT, user_prompt, [em_image, result.colored_image]
        )
        eval_result = self._parse_response_with_points(raw)
        eval_result.vlm_prompts = {"system": SAM3_COORDINATE_SYSTEM_PROMPT, "user": user_prompt}
        return eval_result

    def _build_user_prompt_with_points(
        self,
        result: GenerationResult,
        organelle: OrganelleClass,
        history: list[tuple[GenerationParams, EvaluationResult]] | None,
    ) -> str:
        w, h = result.input_image.size
        parts = [
            f"The first image is the original EM ({w}x{h}). The second image is the SAM3 {organelle.name} segmentation mask.",
            f"Evaluate the {organelle.name} segmentation quality.",
        ]
        if organelle.description:
            parts.append(f"In EM, {organelle.name} appear as: {organelle.description}")
        if self.resolution_nm is not None:
            parts.append(f"Image resolution: {self.resolution_nm:.0f} nm/px.")

        # Show current points
        current_points = result.params_used.extra.get("points", [])
        if current_points:
            def _fmt_pt(i: int, p: dict) -> str:
                s = f"[{i}] ({p['x']}, {p['y']}) label={p.get('label', 1)}"
                if p.get("label", 1) == 1:
                    s += f" instance={p.get('instance', '?')}"
                return s
            pts_str = ", ".join(_fmt_pt(i, p) for i, p in enumerate(current_points))
            parts.append(f"\nCurrent points ({len(current_points)}): {pts_str}")

        parts.append(f"Iteration {result.iteration + 1}.")

        if history:
            parts.append("\nPrevious attempts:")
            for i, (params, eval_result) in enumerate(history):
                scores_str = ""
                if eval_result.detailed_scores:
                    ds = eval_result.detailed_scores
                    scores_str = f"dice={ds.dice_score:.2f}"
                else:
                    scores_str = f"score={eval_result.score:.2f}"
                n_pts = len(params.extra.get("points", []))
                parts.append(f"  #{i+1}: {scores_str}, {n_pts} points, issues={eval_result.issues}")

        parts.append("\nRespond with JSON only (include point_refinement).")
        return "\n".join(parts)

    def _parse_response_with_points(self, raw: str) -> EvaluationResult:
        """Parse evaluation response that includes point refinement data."""
        base_result = self._parse_response(raw)

        # Try to extract point_refinement from the raw response
        text = raw[:4000]
        json_str = self._extract_json_object(text)
        if json_str:
            try:
                parsed = json.loads(json_str)
                pr = parsed.get("point_refinement")
                if isinstance(pr, dict):
                    base_result.point_refinement = PointRefinement(
                        add_points=pr.get("add_points", []),
                        remove_indices=pr.get("remove_indices", []),
                        reasoning=pr.get("reasoning", ""),
                    )
            except json.JSONDecodeError:
                pass

        return base_result

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Extract the first top-level JSON object by tracking brace depth.

        Handles strings (skips braces inside quotes) and is resilient to
        truncated output — if the JSON never closes, returns None.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _build_user_prompt(
        self,
        result: GenerationResult,
        organelle: OrganelleClass,
        history: list[tuple[GenerationParams, EvaluationResult]] | None,
    ) -> str:
        parts = [
            f"The first image is the original EM. The second image is the {organelle.name} segmentation mask output.",
            f"Evaluate the {organelle.name} segmentation quality.",
        ]
        if organelle.description:
            parts.append(
                f"In EM, {organelle.name} appear as: {organelle.description}"
            )
        if self.resolution_nm is not None:
            parts.append(f"Image resolution: {self.resolution_nm:.0f} nm/px.")
        if self.gen_model:
            parts.append(f"Generator model: {self.gen_model}.")
        param_parts = [f"guidance_scale={result.params_used.guidance_scale}"]
        if result.params_used.strength is not None:
            param_parts.append(f"strength={result.params_used.strength}")
        param_parts.append(f"num_inference_steps={result.params_used.num_inference_steps}")
        parts.extend([
            f"Current prompt: \"{result.params_used.prompt}\"",
            f"Current params: {', '.join(param_parts)}",
            f"Iteration {result.iteration + 1}.",
        ])

        if history:
            parts.append("\nPrevious attempts:")
            for i, (params, eval_result) in enumerate(history):
                scores_str = ""
                if eval_result.detailed_scores:
                    ds = eval_result.detailed_scores
                    scores_str = (
                        f"tp={ds.tp_rate:.2f} fp={ds.fp_rate:.2f} "
                        f"fn={ds.fn_rate:.2f} boundary={ds.boundary_quality:.2f} "
                        f"dice={ds.dice_score:.2f}"
                    )
                else:
                    scores_str = f"score={eval_result.score:.2f}"
                hist_params = f"prompt=\"{params.prompt}\", guidance={params.guidance_scale}"
                if params.strength is not None:
                    hist_params += f", strength={params.strength}"
                parts.append(
                    f"  #{i+1}: {scores_str}, {hist_params}, "
                    f"issues={eval_result.issues}"
                )
            best = max(history, key=lambda h: h[1].score)
            best_params = f"prompt=\"{best[0].prompt}\", guidance={best[0].guidance_scale}"
            if best[0].strength is not None:
                best_params += f", strength={best[0].strength}"
            parts.append(
                f"\nBEST so far: #{history.index(best)+1} with score={best[1].score:.2f}, "
                f"{best_params}"
            )
            parts.append("Build on what worked best. Do NOT repeat prompts that scored poorly.")

        parts.append("\nRespond with JSON only.")
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> EvaluationResult:
        """Extract JSON from VLM response, handling markdown fences and malformed output."""
        # Truncate degenerate responses (model sometimes loops on repeated chars)
        text = raw[:4000]

        # Try to extract JSON from markdown code fences
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1)
        else:
            # Try to find raw JSON object by tracking brace depth
            json_str = self._extract_json_object(text)

        parsed = None
        if json_str is not None:
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Fallback: extract individual fields with regex from truncated JSON
        if parsed is None:
            parsed = self._extract_fields_regex(text)

        if parsed is None:
            return EvaluationResult(
                score=0.5,
                detailed_scores=None,
                issues=["Could not parse VLM response"],
                refined_prompt=None,
                param_adjustments={},
                should_stop=False,
                reasoning="Parse failure — using defaults",
                raw_response=raw[:2000],
            )

        # Parse detailed_scores
        detailed = None
        ds = parsed.get("detailed_scores", {})
        if isinstance(ds, dict) and "tp_rate" in ds:
            detailed = DetailedScores(
                tp_rate=float(ds.get("tp_rate", 0.5)),
                fp_rate=float(ds.get("fp_rate", 0.5)),
                fn_rate=float(ds.get("fn_rate", 0.5)),
                boundary_quality=float(ds.get("boundary_quality", 0.5)),
                dice_score=float(ds.get("dice_score", 0.5)),
            )

        # Filter out null values from param_adjustments
        param_adj = parsed.get("param_adjustments", {})
        if isinstance(param_adj, dict):
            param_adj = {k: v for k, v in param_adj.items() if v is not None}
        else:
            param_adj = {}

        score = float(parsed.get("score", 0.5))
        # Use dice_score as the canonical score if available
        if detailed:
            score = detailed.dice_score

        return EvaluationResult(
            score=score,
            detailed_scores=detailed,
            issues=parsed.get("issues", []),
            refined_prompt=parsed.get("refined_prompt"),
            param_adjustments=param_adj,
            should_stop=bool(parsed.get("should_stop", False)),
            reasoning=parsed.get("reasoning", ""),
            raw_response=raw[:2000],
        )

    @staticmethod
    def _extract_fields_regex(text: str) -> dict | None:
        """Extract individual JSON fields from truncated/malformed JSON.

        When the model starts writing valid JSON but gets cut off, we can
        still salvage completed fields.
        """
        result = {}

        # Extract score
        score_match = re.search(r'"score"\s*:\s*([\d.]+)', text)
        if score_match:
            result["score"] = float(score_match.group(1))

        # Extract detailed_scores sub-fields
        ds = {}
        for field in ("tp_rate", "fp_rate", "fn_rate", "boundary_quality", "dice_score"):
            m = re.search(rf'"{field}"\s*:\s*([\d.]+)', text)
            if m:
                ds[field] = float(m.group(1))
        if ds:
            result["detailed_scores"] = ds

        # Extract issues array (try to get completed items)
        issues_match = re.search(r'"issues"\s*:\s*\[([^\]]*)\]', text)
        if issues_match:
            # Extract quoted strings from the array
            result["issues"] = re.findall(r'"([^"]*)"', issues_match.group(1))

        # Extract refined_prompt (only if the string value completed)
        prompt_match = re.search(
            r'"refined_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', text
        )
        if prompt_match:
            result["refined_prompt"] = prompt_match.group(1)

        # Extract should_stop
        stop_match = re.search(r'"should_stop"\s*:\s*(true|false)', text)
        if stop_match:
            result["should_stop"] = stop_match.group(1) == "true"

        # Extract reasoning
        reason_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if reason_match:
            result["reasoning"] = reason_match.group(1)

        # Extract param_adjustments
        param_adj = {}
        for param in ("guidance_scale", "strength", "threshold", "num_inference_steps"):
            m = re.search(rf'"{param}"\s*:\s*([\d.]+)', text)
            if m:
                param_adj[param] = float(m.group(1))
        if param_adj:
            result["param_adjustments"] = param_adj

        return result if result else None
