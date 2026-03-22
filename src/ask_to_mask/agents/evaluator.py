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


VLM_INITIAL_POINTS_PROMPT = """\
We segment organelles in electron microscopy (EM) images using SAM3 (Segment Anything Model 3) \
with point prompts. You identify the approximate center locations of target organelles in the EM \
image so we can feed those coordinates to SAM3.

You will see the EM image. Identify each visible instance of the target organelle and provide \
its approximate center (x, y) in pixel coordinates. The origin (0, 0) is the top-left corner. \
x increases to the right, y increases downward.

Provide foreground points (label=1) at organelle centers. If there are obvious non-organelle \
regions that might confuse the model, you may also add background points (label=0).

Assign each foreground point an instance ID (integer). Points with the same instance ID will be \
used to segment the same object. Each distinct organelle should have a unique instance ID \
(starting from 0). For large or elongated organelles, you may place multiple points on the same \
instance. Background points (label=0) do not need an instance ID.

Respond with ONLY JSON:
{"points": [{"x": 100, "y": 200, "label": 1, "instance": 0}, {"x": 300, "y": 400, "label": 1, "instance": 1}], "reasoning": "why these points"}\
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


MOLMO_POINTS_PROMPT = "Point to the {organelle}"


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
    ):
        self.backend = backend
        self.instance = instance
        self.gen_model = gen_model
        self.resolution_nm = resolution_nm
        self.is_molmo = "molmo" in llm_model.lower()
        self.point_prompt = point_prompt
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
        """Ask the VLM to identify organelle locations as (x, y) point coordinates."""
        if self.is_molmo:
            return self._generate_initial_points_molmo(em_image, organelle)

        w, h = em_image.size
        parts = [
            f"Target organelle: {organelle.name}.",
            f"Image dimensions: {w} x {h} pixels.",
        ]
        if organelle.description:
            parts.append(f"In EM, {organelle.name} appear as: {organelle.description}")
        if self.resolution_nm is not None:
            parts.append(f"Image resolution: {self.resolution_nm:.0f} nm/px.")
        parts.append("\nIdentify each visible instance and provide center coordinates. Respond with JSON only.")
        user_prompt = "\n".join(parts)

        raw = self.backend.chat_with_image(VLM_INITIAL_POINTS_PROMPT, user_prompt, em_image)
        self.last_init_vlm_prompts = {
            "system": VLM_INITIAL_POINTS_PROMPT,
            "user": user_prompt,
            "raw_response": raw[:2000],
        }
        return self._parse_initial_points(raw, em_image)

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

        per_slice_points: dict[int, list[dict]] = {}
        for idx in indices:
            print(f"  Molmo point detection: slice {idx+1}/{n}")
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

    def _parse_initial_points(self, raw: str, em_image: Image.Image) -> list[dict]:
        """Extract point coordinates from the VLM's response."""
        text = raw[:4000]
        json_str = self._extract_json_object(text)
        if json_str:
            try:
                parsed = json.loads(json_str)
                points = parsed.get("points", [])
                if points and isinstance(points, list):
                    reasoning = parsed.get("reasoning", "")
                    if reasoning:
                        print(f"  VLM points reasoning: {reasoning}")
                    # Validate and clamp coordinates
                    w, h = em_image.size
                    valid_points = []
                    for i, p in enumerate(points):
                        if isinstance(p, dict) and "x" in p and "y" in p:
                            pt = {
                                "x": max(0, min(w - 1, int(p["x"]))),
                                "y": max(0, min(h - 1, int(p["y"]))),
                                "label": int(p.get("label", 1)),
                            }
                            # Preserve instance ID; default to index if omitted
                            if pt["label"] == 1:
                                pt["instance"] = int(p.get("instance", i))
                            valid_points.append(pt)
                    return valid_points
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract points with regex (JSON dict format)
        point_matches = re.findall(
            r'\{\s*"x"\s*:\s*(\d+)\s*,\s*"y"\s*:\s*(\d+)(?:\s*,\s*"label"\s*:\s*(\d+))?\s*\}',
            text,
        )
        if point_matches:
            w, h = em_image.size
            result = []
            for i, m in enumerate(point_matches):
                pt = {
                    "x": max(0, min(w - 1, int(m[0]))),
                    "y": max(0, min(h - 1, int(m[1]))),
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

        prompt = self.point_prompt or MOLMO_POINTS_PROMPT.format(organelle=organelle.name)

        self.last_init_vlm_prompts = {
            "system": "",
            "user": prompt,
        }

        # Save image to temp file for subprocess
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            em_image.save(f, format="PNG")
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
