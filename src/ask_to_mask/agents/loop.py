"""Orchestrator: runs the generate-critique-refine loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..config import OrganelleClass
from .evaluator import EvaluatorAgent
from .gen_backend import ImageGenBackend
from .llm_backend import LLMBackend, images_to_composite
from .schemas import EvaluationResult, GenerationParams, GenerationResult


@dataclass
class LoopConfig:
    """Configuration for the refinement loop."""

    max_iterations: int = 5
    min_acceptable_score: float = 0.8
    save_intermediates: bool = True


@dataclass
class LoopResult:
    """Final output from the refinement loop."""

    best_result: GenerationResult
    best_evaluation: EvaluationResult
    all_results: list[GenerationResult]
    all_evaluations: list[EvaluationResult]
    total_iterations: int
    converged: bool  # True if stopped due to quality, False if hit max_iterations


def run_refinement_loop(
    gen_backend: ImageGenBackend,
    llm_backend: LLMBackend,
    em_image: Image.Image,
    organelle: OrganelleClass,
    initial_params: GenerationParams,
    config: LoopConfig = LoopConfig(),
    output_dir: Path | None = None,
) -> LoopResult:
    """Run the generate-evaluate-refine loop.

    Flow:
        1. Generate mask with current params (via gen_backend)
        2. Evaluator agent critiques result and suggests refined prompt/params
        3. If good enough or max iterations reached, stop
        4. Apply refinements and go to step 1
    """
    evaluator = EvaluatorAgent(backend=llm_backend)

    current_params = initial_params
    all_results: list[GenerationResult] = []
    all_evaluations: list[EvaluationResult] = []
    history: list[tuple[GenerationParams, EvaluationResult]] = []

    for iteration in range(config.max_iterations):
        print(f"\n=== Iteration {iteration + 1}/{config.max_iterations} ===")
        print(f"  Prompt: {current_params.prompt}")

        # Step 1: Generate
        result = gen_backend.generate(em_image, current_params, iteration)
        all_results.append(result)

        # Save intermediates
        if config.save_intermediates and output_dir:
            _save_iteration(output_dir, iteration, result, em_image)

        # Step 2: Evaluate and refine
        evaluation = evaluator.evaluate_and_refine(
            em_image, result, organelle, history
        )
        all_evaluations.append(evaluation)
        history.append((current_params, evaluation))

        print(f"  Score: {evaluation.score:.2f}")
        if evaluation.issues:
            for issue in evaluation.issues:
                print(f"    - {issue}")
        if evaluation.reasoning:
            print(f"  Reasoning: {evaluation.reasoning}")
        if evaluation.param_adjustments:
            print(f"  Param adjustments: {evaluation.param_adjustments}")

        # Save evaluation
        if config.save_intermediates and output_dir:
            _save_evaluation(output_dir, iteration, evaluation)

        # Step 3: Check termination (only score-based, ignore should_stop from VLM)
        if evaluation.score >= config.min_acceptable_score:
            print(f"  Accepted at iteration {iteration + 1}")
            return LoopResult(
                best_result=result,
                best_evaluation=evaluation,
                all_results=all_results,
                all_evaluations=all_evaluations,
                total_iterations=iteration + 1,
                converged=True,
            )

        # Early stop: score plateau (3 consecutive scores within 0.05 of each other)
        if len(all_evaluations) >= 3:
            recent = [e.score for e in all_evaluations[-3:]]
            if max(recent) - min(recent) < 0.05:
                print("  Score plateau detected — stopping early")
                break

        # Step 4: Refine params for next iteration
        if iteration < config.max_iterations - 1:
            current_params = _apply_refinement(current_params, evaluation, iteration)
            if evaluation.refined_prompt:
                print(f"  Refined prompt: {current_params.prompt}")

    # Return best result by score
    best_idx = max(range(len(all_evaluations)), key=lambda i: all_evaluations[i].score)
    return LoopResult(
        best_result=all_results[best_idx],
        best_evaluation=all_evaluations[best_idx],
        all_results=all_results,
        all_evaluations=all_evaluations,
        total_iterations=len(all_results),
        converged=False,
    )


def _apply_refinement(
    current: GenerationParams, evaluation: EvaluationResult, iteration: int
) -> GenerationParams:
    """Create new GenerationParams by applying the evaluator's suggestions."""
    prompt = evaluation.refined_prompt if evaluation.refined_prompt else current.prompt

    # Start with current values
    new_steps = current.num_inference_steps
    new_guidance = current.guidance_scale
    new_strength = current.strength
    new_threshold = current.threshold

    # Apply adjustments from evaluator
    adj = evaluation.param_adjustments
    if "num_inference_steps" in adj:
        new_steps = int(adj["num_inference_steps"])
    if "guidance_scale" in adj:
        new_guidance = float(adj["guidance_scale"])
    if "strength" in adj:
        new_strength = float(adj["strength"])
    if "threshold" in adj:
        new_threshold = float(adj["threshold"])

    # Vary seed across iterations
    new_seed = current.seed
    if new_seed is not None:
        new_seed = current.seed + iteration + 1

    return GenerationParams(
        prompt=prompt,
        num_inference_steps=new_steps,
        guidance_scale=new_guidance,
        strength=new_strength,
        seed=new_seed,
        threshold=new_threshold,
        extra=current.extra,
    )


def _save_iteration(
    output_dir: Path, iteration: int, result: GenerationResult, em_image: Image.Image
) -> None:
    """Save intermediate images for an iteration."""
    iter_dir = output_dir / f"iteration_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    result.colored_image.save(iter_dir / "colored.png")
    result.mask_image.save(iter_dir / "mask.png")
    composite = images_to_composite(em_image, result.colored_image, result.mask_image)
    composite.save(iter_dir / "composite.png")


def _save_evaluation(
    output_dir: Path, iteration: int, evaluation: EvaluationResult
) -> None:
    """Save evaluation results as JSON."""
    iter_dir = output_dir / f"iteration_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "score": evaluation.score,
        "issues": evaluation.issues,
        "reasoning": evaluation.reasoning,
        "refined_prompt": evaluation.refined_prompt,
        "param_adjustments": evaluation.param_adjustments,
    }
    with open(iter_dir / "evaluation.json", "w") as f:
        json.dump(data, f, indent=4)
