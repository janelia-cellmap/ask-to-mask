"""Combined evaluator agent: critiques mask quality AND refines the prompt in one call."""

from __future__ import annotations

import json
import re

from PIL import Image

from ..config import OrganelleClass
from .llm_backend import LLMBackend, images_to_composite
from .schemas import EvaluationResult, GenerationParams, GenerationResult

SYSTEM_PROMPT = """\
You evaluate EM image colorization. You see two panels: LEFT=original EM, RIGHT=colored output.

Compare LEFT and RIGHT carefully. The colored regions in RIGHT must correspond to where the target organelle actually appears in LEFT. If something is colored that doesn't match the organelle's appearance in the EM, that's a false positive.

The goal: ONLY the target organelle should be brightly colored. The background and other structures must stay grayscale/unchanged.

Be VERY HARSH:
- 0.0-0.2: entire image is tinted/colored (WORST failure — means everything is colored, not just the organelle)
- 0.2-0.4: organelle not colored, or wrong structures colored, or massive background coloring
- 0.4-0.6: some organelle colored but significant problems (many missed, heavy background color, uneven)
- 0.6-0.75: decent — most organelles colored, moderate background bleed
- 0.75-0.85: good — organelles well colored with only minor issues
- 0.85-1.0: excellent (RARE) — all organelles brightly colored, background stays gray

CRITICAL: If the background/non-organelle areas are visibly colored, that is a MAJOR problem. Score below 0.4 if most of the image is tinted.

When score < 0.85, you MUST change BOTH:
- refined_prompt: Write a DIFFERENT prompt. If background is colored, try "ONLY color the {organelle}" or "Do not color the background". If organelles are missed, describe their shape/location.
- param_adjustments: Change at least one. guidance_scale (3.5-30, higher=follows prompt more). strength (0.5-1.0, lower=less change to image). num_inference_steps (20-50, higher=more detail).

If the whole image is tinted: LOWER strength significantly (try 0.3-0.55) and RAISE guidance_scale (try 15-25). If still tinted after lowering, keep lowering strength.

Keep prompts under 50 words. Include "Color all the {organelle} in {color}." End with "Keep everything else unchanged."

Respond with ONLY JSON:
{"score": 0.0-1.0, "issues": ["specific issue"], "refined_prompt": "new different prompt", "param_adjustments": {"guidance_scale": 15.0, "strength": 0.5, "num_inference_steps": 28}, "should_stop": false, "reasoning": "brief reason"}\
"""


class EvaluatorAgent:
    """Evaluates mask quality and refines prompts using a VLM backend."""

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def evaluate_and_refine(
        self,
        em_image: Image.Image,
        result: GenerationResult,
        organelle: OrganelleClass,
        history: list[tuple[GenerationParams, EvaluationResult]] | None = None,
    ) -> EvaluationResult:
        """Critique the mask AND produce a refined prompt in one VLM call."""
        composite = images_to_composite(em_image, result.colored_image)

        user_prompt = self._build_user_prompt(result, organelle, history)
        raw = self.backend.chat_with_image(SYSTEM_PROMPT, user_prompt, composite)
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
            f"Evaluate this {organelle.name} coloring.",
        ]
        if organelle.description:
            parts.append(
                f"In EM, {organelle.name} appear as: {organelle.description}"
            )
        parts.extend([
            f"Current prompt: \"{result.params_used.prompt}\"",
            f"Current params: guidance_scale={result.params_used.guidance_scale}, "
            f"strength={result.params_used.strength}, "
            f"num_inference_steps={result.params_used.num_inference_steps}",
            f"Iteration {result.iteration + 1}.",
        ])

        if history:
            parts.append("\nPrevious attempts:")
            for i, (params, eval_result) in enumerate(history):
                parts.append(
                    f"  #{i+1}: score={eval_result.score:.2f}, "
                    f"prompt=\"{params.prompt}\", "
                    f"strength={params.strength}, "
                    f"guidance={params.guidance_scale}, "
                    f"issues={eval_result.issues}"
                )
            parts.append("Do NOT repeat prompts or params that scored poorly. Try something different.")

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
                issues=["Could not parse VLM response"],
                refined_prompt=None,
                param_adjustments={},
                should_stop=False,
                reasoning="Parse failure — using defaults",
                raw_response=raw[:2000],
            )

        # Filter out null values from param_adjustments
        param_adj = parsed.get("param_adjustments", {})
        if isinstance(param_adj, dict):
            param_adj = {k: v for k, v in param_adj.items() if v is not None}
        else:
            param_adj = {}

        return EvaluationResult(
            score=float(parsed.get("score", 0.5)),
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

        # Extract guidance_scale and strength from param_adjustments
        param_adj = {}
        for param in ("guidance_scale", "strength", "threshold", "num_inference_steps"):
            m = re.search(rf'"{param}"\s*:\s*([\d.]+)', text)
            if m:
                param_adj[param] = float(m.group(1))
        if param_adj:
            result["param_adjustments"] = param_adj

        return result if result else None
