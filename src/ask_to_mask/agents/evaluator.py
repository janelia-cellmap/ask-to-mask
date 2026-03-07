"""Combined evaluator agent: critiques mask quality AND refines the prompt in one call."""

from __future__ import annotations

import json
import re

from PIL import Image

from ..config import OrganelleClass
from .llm_backend import LLMBackend, images_to_composite
from .schemas import (
    DetailedScores,
    EvaluationResult,
    GenerationParams,
    GenerationResult,
)

SYSTEM_PROMPT = """\
You evaluate EM image colorization. You see two panels: LEFT=original EM, RIGHT=colored output.

Compare LEFT and RIGHT carefully:
1. Count how many organelles are CORRECTLY colored in RIGHT vs how many are visible in LEFT
2. Check if NON-organelle areas are colored (false positives / background bleed)
3. Check if the coloring is precise (tight boundaries) or sloppy (bleeds into surroundings)

You MUST provide detailed sub-scores (all 0.0 to 1.0):
- tp_rate: fraction of real organelles that are correctly colored (1.0 = all colored)
- fp_rate: fraction of colored pixels that are on NON-organelle areas (0.0 = no false positives, 1.0 = all colored area is wrong)
- fn_rate: fraction of real organelles that were MISSED / not colored (0.0 = none missed, 1.0 = all missed)
- boundary_quality: how precise/tight the coloring boundaries are (1.0 = pixel-perfect, 0.0 = very sloppy bleed)
- dice_score: estimated overall quality = 2*TP / (2*TP + FP + FN). This is your best estimate of segmentation overlap.

The "score" field should equal the dice_score.

When score < 0.85, suggest a refined_prompt (up to 75 words, end with "Keep everything else unchanged.").

DO NOT suggest param_adjustments — the loop controls parameters automatically. Set param_adjustments to {}.

Respond with ONLY JSON:
{"score": 0.0-1.0, "detailed_scores": {"tp_rate": 0.8, "fp_rate": 0.1, "fn_rate": 0.2, "boundary_quality": 0.7, "dice_score": 0.6}, "issues": ["specific issue"], "refined_prompt": "new prompt", "param_adjustments": {}, "should_stop": false, "reasoning": "brief reason"}\
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
                parts.append(
                    f"  #{i+1}: {scores_str}, "
                    f"prompt=\"{params.prompt}\", "
                    f"strength={params.strength}, "
                    f"guidance={params.guidance_scale}, "
                    f"issues={eval_result.issues}"
                )
            best = max(history, key=lambda h: h[1].score)
            parts.append(
                f"\nBEST so far: #{history.index(best)+1} with score={best[1].score:.2f}, "
                f"prompt=\"{best[0].prompt}\", strength={best[0].strength}, "
                f"guidance={best[0].guidance_scale}"
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
