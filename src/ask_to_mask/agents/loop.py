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


# The tunable numeric parameters and their allowed ranges
TUNABLE_PARAMS = {
    "strength": (0.3, 1.0),
    "guidance_scale": (1.0, 30.0),
    "num_inference_steps": (15, 50),
}

# How much to change a parameter per step (fraction of current value)
STEP_FRACTION = 0.05


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
    converged: bool  # True if stopped due to quality
    plateau: bool = False  # True if stopped due to score plateau


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

    Strategy: change ONE parameter at a time by ±5%, evaluate, keep if better,
    revert if worse. Cycle through params systematically. The evaluator only
    refines the prompt — param tuning is done here.
    """
    evaluator = EvaluatorAgent(backend=llm_backend)

    current_params = initial_params
    all_results: list[GenerationResult] = []
    all_evaluations: list[EvaluationResult] = []
    history: list[tuple[GenerationParams, EvaluationResult]] = []

    # Track which param to tweak next and which direction
    param_names = list(TUNABLE_PARAMS.keys())
    # Each param gets tried: +5%, if worse try -5%, if worse revert and move on
    # "param_to_try" cycles through params; "direction" is +1 or -1
    param_index = 0
    direction = 1  # +1 = increase, -1 = decrease
    tried_increase = False  # Have we tried +5% for this param yet?

    best_score = -1.0
    best_idx = 0
    best_params = current_params

    # Track param effects for logging
    param_effects: list[dict] = []

    for iteration in range(config.max_iterations):
        print(f"\n=== Iteration {iteration + 1}/{config.max_iterations} ===")
        print(f"  Prompt: {current_params.prompt}")
        print(
            f"  Params: strength={current_params.strength:.3f}, "
            f"guidance_scale={current_params.guidance_scale:.2f}, "
            f"num_inference_steps={current_params.num_inference_steps}"
        )

        # Step 1: Generate
        result = gen_backend.generate(em_image, current_params, iteration)
        all_results.append(result)

        # Save intermediates
        if config.save_intermediates and output_dir:
            _save_iteration(output_dir, iteration, result, em_image)

        # Step 2: Evaluate
        evaluation = evaluator.evaluate_and_refine(
            em_image, result, organelle, history
        )
        all_evaluations.append(evaluation)
        history.append((current_params, evaluation))

        # Print detailed scores
        print(f"  Score (dice): {evaluation.score:.3f}")
        if evaluation.detailed_scores:
            ds = evaluation.detailed_scores
            print(
                f"  Details: tp_rate={ds.tp_rate:.2f}  fp_rate={ds.fp_rate:.2f}  "
                f"fn_rate={ds.fn_rate:.2f}  boundary={ds.boundary_quality:.2f}  "
                f"dice={ds.dice_score:.3f}"
            )
        if evaluation.issues:
            for issue in evaluation.issues:
                print(f"    - {issue}")
        if evaluation.reasoning:
            print(f"  Reasoning: {evaluation.reasoning}")

        # Save evaluation
        if config.save_intermediates and output_dir:
            _save_evaluation(output_dir, iteration, evaluation, current_params,
                             param_effects[-1] if param_effects else None)

        # Step 3: Check if good enough
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

        # Step 4: Track best and decide next param change
        prev_best_score = best_score
        if evaluation.score > best_score:
            best_score = evaluation.score
            best_idx = iteration
            best_params = current_params

        if iteration == 0:
            # First iteration is baseline — apply prompt refinement then start param sweep
            next_params = _copy_params(current_params)
            if evaluation.refined_prompt:
                next_params = _with_prompt(next_params, evaluation.refined_prompt)
                print(f"  Refined prompt: {next_params.prompt}")
            # Apply first param change
            param_name = param_names[param_index]
            direction = 1
            tried_increase = True
            next_params = _step_param(next_params, param_name, direction)
            effect = {
                "changed_param": param_name,
                "direction": "+5%",
                "old_value": _get_param(current_params, param_name),
                "new_value": _get_param(next_params, param_name),
            }
            param_effects.append(effect)
            print(f"  Next: {param_name} {'+5%' if direction > 0 else '-5%'} "
                  f"({effect['old_value']:.3f} -> {effect['new_value']:.3f})")
            current_params = next_params
            continue

        # We just tested a param change — did it help?
        prev_score = all_evaluations[-2].score if len(all_evaluations) >= 2 else 0
        score_delta = evaluation.score - prev_score
        param_name = param_names[param_index]

        if param_effects:
            param_effects[-1]["score_before"] = prev_score
            param_effects[-1]["score_after"] = evaluation.score
            param_effects[-1]["score_delta"] = score_delta

        if score_delta > 0:
            # Improvement! Keep the change and move to next param
            print(f"  {param_name} change helped (+{score_delta:.3f}), keeping it")
            param_index = (param_index + 1) % len(param_names)
            direction = 1
            tried_increase = False
        elif tried_increase:
            # +5% didn't help, try -5% (revert first, then go other way)
            print(f"  {param_name} +5% didn't help ({score_delta:+.3f}), trying -5%")
            # Revert to before this change
            current_params = _revert_param(current_params, param_name, param_effects[-1])
            direction = -1
            tried_increase = False  # Now we're trying decrease
        else:
            # -5% also didn't help, revert and move to next param
            print(f"  {param_name} -5% didn't help ({score_delta:+.3f}), reverting and moving on")
            current_params = _revert_param(current_params, param_name, param_effects[-1])
            param_index = (param_index + 1) % len(param_names)
            direction = 1
            tried_increase = False

        # Prepare next iteration
        next_params = _copy_params(current_params)

        # Apply prompt refinement from evaluator
        if evaluation.refined_prompt:
            next_params = _with_prompt(next_params, evaluation.refined_prompt)
            print(f"  Refined prompt: {next_params.prompt}")

        # If score dropped significantly from best, revert to best params
        if evaluation.score < best_score - 0.1:
            print(f"  Score dropped significantly (best was {best_score:.3f}) — reverting to best params")
            next_params = _copy_params(best_params)
            if evaluation.refined_prompt:
                next_params = _with_prompt(next_params, evaluation.refined_prompt)

        # Apply next param step
        param_name = param_names[param_index]
        if direction == 1:
            tried_increase = True
        next_params = _step_param(next_params, param_name, direction)
        effect = {
            "changed_param": param_name,
            "direction": "+5%" if direction > 0 else "-5%",
            "old_value": _get_param(current_params, param_name),
            "new_value": _get_param(next_params, param_name),
        }
        param_effects.append(effect)
        print(f"  Next: {param_name} {effect['direction']} "
              f"({effect['old_value']:.3f} -> {effect['new_value']:.3f})")

        # Vary seed across iterations
        if next_params.seed is not None:
            next_params = GenerationParams(
                prompt=next_params.prompt,
                num_inference_steps=next_params.num_inference_steps,
                guidance_scale=next_params.guidance_scale,
                strength=next_params.strength,
                seed=initial_params.seed + iteration + 1,
                threshold=next_params.threshold,
                extra=next_params.extra,
            )

        current_params = next_params

    # Return best result by score
    return LoopResult(
        best_result=all_results[best_idx],
        best_evaluation=all_evaluations[best_idx],
        all_results=all_results,
        all_evaluations=all_evaluations,
        total_iterations=len(all_results),
        converged=False,
        plateau=False,
    )


def _get_param(params: GenerationParams, name: str) -> float:
    """Get a parameter value by name."""
    return float(getattr(params, name))


def _step_param(params: GenerationParams, name: str, direction: int) -> GenerationParams:
    """Change one parameter by ±5%, clamped to allowed range."""
    current_val = _get_param(params, name)
    delta = current_val * STEP_FRACTION * direction
    new_val = current_val + delta

    # Clamp to allowed range
    lo, hi = TUNABLE_PARAMS[name]
    new_val = max(lo, min(hi, new_val))

    return _with_param(params, name, new_val)


def _with_param(params: GenerationParams, name: str, value: float) -> GenerationParams:
    """Return a copy of params with one parameter changed."""
    kwargs = {
        "prompt": params.prompt,
        "num_inference_steps": params.num_inference_steps,
        "guidance_scale": params.guidance_scale,
        "strength": params.strength,
        "seed": params.seed,
        "threshold": params.threshold,
        "extra": params.extra,
    }
    if name == "num_inference_steps":
        kwargs[name] = int(round(value))
    else:
        kwargs[name] = value
    return GenerationParams(**kwargs)


def _with_prompt(params: GenerationParams, prompt: str) -> GenerationParams:
    """Return a copy of params with a new prompt."""
    return GenerationParams(
        prompt=prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        strength=params.strength,
        seed=params.seed,
        threshold=params.threshold,
        extra=params.extra,
    )


def _copy_params(params: GenerationParams) -> GenerationParams:
    """Return a copy of params."""
    return GenerationParams(
        prompt=params.prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        strength=params.strength,
        seed=params.seed,
        threshold=params.threshold,
        extra=params.extra,
    )


def _revert_param(params: GenerationParams, name: str, effect: dict) -> GenerationParams:
    """Revert a parameter to its value before the last change."""
    return _with_param(params, name, effect["old_value"])


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
    output_dir: Path,
    iteration: int,
    evaluation: EvaluationResult,
    params: GenerationParams,
    param_change: dict | None = None,
) -> None:
    """Save evaluation results as JSON."""
    iter_dir = output_dir / f"iteration_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "score": evaluation.score,
        "detailed_scores": None,
        "params_used": {
            "prompt": params.prompt,
            "strength": params.strength,
            "guidance_scale": params.guidance_scale,
            "num_inference_steps": params.num_inference_steps,
            "threshold": params.threshold,
            "seed": params.seed,
        },
        "issues": evaluation.issues,
        "reasoning": evaluation.reasoning,
        "refined_prompt": evaluation.refined_prompt,
    }

    if evaluation.detailed_scores:
        ds = evaluation.detailed_scores
        data["detailed_scores"] = {
            "tp_rate": ds.tp_rate,
            "fp_rate": ds.fp_rate,
            "fn_rate": ds.fn_rate,
            "boundary_quality": ds.boundary_quality,
            "dice_score": ds.dice_score,
        }

    if param_change:
        data["param_change"] = param_change

    with open(iter_dir / "evaluation.json", "w") as f:
        json.dump(data, f, indent=4)
