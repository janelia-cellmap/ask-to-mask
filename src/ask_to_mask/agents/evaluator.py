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


class EvaluatorAgent:
    """Evaluates mask quality and refines prompts using a VLM backend."""

    def __init__(
        self,
        backend: LLMBackend,
        instance: bool = False,
        gen_model: str = "",
        resolution_nm: float | None = None,
    ):
        self.backend = backend
        self.instance = instance
        self.gen_model = gen_model
        self.resolution_nm = resolution_nm

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
        return self._parse_response(raw)

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
