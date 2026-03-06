"""Dataclasses for structured data exchange between agent components."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class GenerationParams:
    """Parameters for a single mask generation run."""

    prompt: str
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    strength: float = 0.75
    seed: int | None = None
    threshold: float = 30.0
    extra: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Output from a single generation run."""

    input_image: Image.Image
    colored_image: Image.Image
    mask: np.ndarray
    mask_image: Image.Image
    params_used: GenerationParams
    iteration: int


@dataclass
class EvaluationResult:
    """Combined critique + refinement output from the evaluator agent."""

    score: float  # 0.0 to 1.0
    issues: list[str]
    refined_prompt: str | None
    param_adjustments: dict[str, float]
    should_stop: bool
    reasoning: str
    raw_response: str
