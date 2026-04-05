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
from .schemas import EvaluationResult, GenerationParams, GenerationResult, PointRefinement


# The tunable numeric parameters and their allowed ranges, per backend
TUNABLE_PARAMS_BY_BACKEND = {
    "flux": {
        "strength": (0.3, 1.0),
        "guidance_scale": (1.0, 30.0),
        "num_inference_steps": (15, 50),
    },
    "glm": {
        "guidance_scale": (1.0, 10.0),
        "num_inference_steps": (20, 100),
    },
    "qwen": {
        "guidance_scale": (0.5, 10.0),
        "num_inference_steps": (20, 80),
    },
    "gemini": {},  # No tunable numeric params
    "sam3": {},  # Refinement via prompt/points, not numeric params
}

# How much to change a parameter per step (fraction of current value)
STEP_FRACTION = 0.05


def _backend_name(gen_backend: ImageGenBackend) -> str:
    """Infer the backend key from the class name."""
    from .gen_backend import (
        FluxBackend,
        GeminiImageBackend,
        GLMImageBackend,
        QwenImageEditBackend,
    )
    from .sam3_backend import SAM3Backend

    if isinstance(gen_backend, SAM3Backend):
        return "sam3"
    if isinstance(gen_backend, FluxBackend):
        return "flux"
    if isinstance(gen_backend, GLMImageBackend):
        return "glm"
    if isinstance(gen_backend, QwenImageEditBackend):
        return "qwen"
    if isinstance(gen_backend, GeminiImageBackend):
        return "gemini"
    return "flux"


@dataclass
class LoopConfig:
    """Configuration for the refinement loop."""

    max_iterations: int = 5
    min_acceptable_score: float = 0.95
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
    instance: bool = False,
    mask_mode: str = "overlay",
    gen_model: str = "",
    resolution_nm: float | None = None,
    llm_model: str = "",
    point_backend: LLMBackend | None = None,
    point_model: str = "",
    validate_points: bool = False,
) -> LoopResult:
    """Run the generate-evaluate-refine loop.

    Strategy: change ONE parameter at a time by ±5%, evaluate, keep if better,
    revert if worse. Cycle through params systematically. The evaluator only
    refines the prompt — param tuning is done here.
    """
    evaluator = EvaluatorAgent(
        backend=llm_backend,
        instance=instance,
        gen_model=gen_model,
        resolution_nm=resolution_nm,
        llm_model=llm_model,
        point_backend=point_backend,
        point_model=point_model,
    )

    # Check if this is a SAM3 point-based strategy
    sam3_strategy = initial_params.extra.get("sam3_strategy")
    use_points = sam3_strategy == "vlm-coordinate"

    use_sam3_text = sam3_strategy == "text"

    if use_points:
        # Generate initial point coordinates via VLM
        print("\n=== Generating initial points via VLM ===")
        points = evaluator.generate_initial_points(em_image, organelle)
        print(f"  Initial points: {len(points)} locations")
        for p in points:
            print(f"    ({p['x']}, {p['y']}) label={p.get('label', 1)}")
        # Optionally validate each point via the eval VLM
        if validate_points and points:
            print("\n=== Validating points via eval VLM ===")
            points = evaluator.validate_points(em_image, points, organelle)
            if not points:
                print("  Warning: all points were rejected, using image center as fallback")
                w, h = em_image.size
                points = [{"x": w // 2, "y": h // 2, "label": 1, "instance": 0}]
        initial_params = _with_extra(
            initial_params, {**initial_params.extra, "points": points}
        )
    elif use_sam3_text:
        # SAM3 text mode: build a list of short candidate prompts to try
        # across iterations (SAM3 expects simple terms, not sentences)
        sam3_prompts = _sam3_candidate_prompts(organelle)
        print(f"\n=== SAM3 text mode — candidates: {sam3_prompts} ===")
        initial_params = _with_prompt(initial_params, sam3_prompts[0])
    else:
        # Use VLM to generate the initial prompt
        print("\n=== Generating initial prompt via VLM ===")
        initial_prompt = evaluator.generate_initial_prompt(em_image, organelle, mask_mode)
        print(f"  Initial prompt: {initial_prompt}")
        initial_params = _with_prompt(initial_params, initial_prompt)

    current_params = initial_params
    all_results: list[GenerationResult] = []
    all_evaluations: list[EvaluationResult] = []
    history: list[tuple[GenerationParams, EvaluationResult]] = []

    # Determine tunable params for this backend
    backend_key = _backend_name(gen_backend)
    tunable_params = TUNABLE_PARAMS_BY_BACKEND.get(backend_key, {})

    # Track which param to tweak next and which direction
    param_names = list(tunable_params.keys())
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
        param_strs = []
        if current_params.strength is not None:
            param_strs.append(f"strength={current_params.strength:.3f}")
        param_strs.append(f"guidance_scale={current_params.guidance_scale:.2f}")
        param_strs.append(f"num_inference_steps={current_params.num_inference_steps}")
        print(f"  Params: {', '.join(param_strs)}")

        # Step 1: Generate
        try:
            result = gen_backend.generate(
                em_image,
                current_params,
                iteration,
                instance=instance,
                mask_mode=mask_mode,
            )
        except (ImportError, ModuleNotFoundError):
            raise
        except Exception as e:
            print(f"  Generation failed: {e}")
            # Save params so we know what was attempted
            if config.save_intermediates and output_dir:
                iter_dir = output_dir / f"iteration_{iteration:02d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                _save_evaluation(
                    output_dir,
                    iteration,
                    EvaluationResult(
                        score=0.0,
                        detailed_scores=None,
                        issues=[f"Generation failed: {e}"],
                        refined_prompt=None,
                        param_adjustments={},
                        should_stop=False,
                        reasoning="generation error",
                        raw_response="",
                    ),
                    current_params,
                    init_vlm_prompts=evaluator.last_init_vlm_prompts if iteration == 0 else None,
                )
            continue
        all_results.append(result)

        # Save intermediates
        if config.save_intermediates and output_dir:
            _save_iteration(output_dir, iteration, result, em_image)

        # Step 2: Evaluate (skip if single iteration — no refinement needed)
        if config.max_iterations == 1:
            # No evaluation needed — just save generation info and return
            dummy_eval = EvaluationResult(
                score=0.0,
                detailed_scores=None,
                issues=[],
                refined_prompt=None,
                param_adjustments={},
                should_stop=True,
                reasoning="skipped (single iteration)",
                raw_response="",
            )
            all_evaluations.append(dummy_eval)
            if config.save_intermediates and output_dir:
                _save_evaluation(
                    output_dir,
                    iteration,
                    dummy_eval,
                    current_params,
                    init_vlm_prompts=evaluator.last_init_vlm_prompts if iteration == 0 else None,
                )
            print("  Evaluation skipped (single iteration)")
            return LoopResult(
                best_result=result,
                best_evaluation=dummy_eval,
                all_results=all_results,
                all_evaluations=all_evaluations,
                total_iterations=1,
                converged=False,
            )

        if use_points:
            evaluation = evaluator.evaluate_and_refine_with_points(
                em_image, result, organelle, history
            )
        else:
            evaluation = evaluator.evaluate_and_refine(em_image, result, organelle, history)
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
            _save_evaluation(
                output_dir,
                iteration,
                evaluation,
                current_params,
                param_effects[-1] if param_effects else None,
                init_vlm_prompts=evaluator.last_init_vlm_prompts if iteration == 0 else None,
            )

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

        # Step 4b: Apply prompt refinement
        next_params = _copy_params(current_params)
        if use_sam3_text:
            # Cycle through candidate prompts
            next_prompt_idx = (iteration + 1) % len(sam3_prompts)
            next_params = _with_prompt(next_params, sam3_prompts[next_prompt_idx])
            print(f"  Next SAM3 prompt: {sam3_prompts[next_prompt_idx]}")
        elif evaluation.refined_prompt:
            next_params = _with_prompt(next_params, evaluation.refined_prompt)
            print(f"  Refined prompt: {next_params.prompt}")

        # Step 4c: Apply point refinement (SAM3 coordinate mode)
        if evaluation.point_refinement:
            next_params = _apply_point_refinement(next_params, evaluation.point_refinement)
            n_pts = len(next_params.extra.get("points", []))
            pr = evaluation.point_refinement
            print(
                f"  Points updated: +{len(pr.add_points)} "
                f"-{len(pr.remove_indices)} → {n_pts} total"
            )
            if pr.reasoning:
                print(f"  Point reasoning: {pr.reasoning}")

        if not param_names:
            # No tunable params (e.g. Gemini) — only prompt refinement
            current_params = next_params
            continue

        if iteration == 0:
            # First iteration is baseline — start param sweep
            param_name = param_names[param_index]
            direction = 1
            tried_increase = True
            next_params = _step_param(
                next_params, param_name, direction, tunable_params
            )
            effect = {
                "changed_param": param_name,
                "direction": "+5%",
                "old_value": _get_param(current_params, param_name),
                "new_value": _get_param(next_params, param_name),
            }
            param_effects.append(effect)
            print(
                f"  Next: {param_name} +5% "
                f"({effect['old_value']:.3f} -> {effect['new_value']:.3f})"
            )
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
            print(f"  {param_name} change helped (+{score_delta:.3f}), keeping it")
            param_index = (param_index + 1) % len(param_names)
            direction = 1
            tried_increase = False
        elif tried_increase:
            print(f"  {param_name} +5% didn't help ({score_delta:+.3f}), trying -5%")
            current_params = _revert_param(
                current_params, param_name, param_effects[-1]
            )
            next_params = _copy_params(current_params)
            if evaluation.refined_prompt:
                next_params = _with_prompt(next_params, evaluation.refined_prompt)
            direction = -1
            tried_increase = False
        else:
            print(
                f"  {param_name} -5% didn't help ({score_delta:+.3f}), reverting and moving on"
            )
            current_params = _revert_param(
                current_params, param_name, param_effects[-1]
            )
            next_params = _copy_params(current_params)
            if evaluation.refined_prompt:
                next_params = _with_prompt(next_params, evaluation.refined_prompt)
            param_index = (param_index + 1) % len(param_names)
            direction = 1
            tried_increase = False

        # If score dropped significantly from best, revert to best params
        if evaluation.score < best_score - 0.1:
            print(
                f"  Score dropped significantly (best was {best_score:.3f}) — reverting to best params"
            )
            next_params = _copy_params(best_params)
            if evaluation.refined_prompt:
                next_params = _with_prompt(next_params, evaluation.refined_prompt)

        # Apply next param step
        param_name = param_names[param_index]
        if direction == 1:
            tried_increase = True
        next_params = _step_param(next_params, param_name, direction, tunable_params)
        effect = {
            "changed_param": param_name,
            "direction": "+5%" if direction > 0 else "-5%",
            "old_value": _get_param(current_params, param_name),
            "new_value": _get_param(next_params, param_name),
        }
        param_effects.append(effect)
        print(
            f"  Next: {param_name} {effect['direction']} "
            f"({effect['old_value']:.3f} -> {effect['new_value']:.3f})"
        )

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


def _sam3_candidate_prompts(organelle: OrganelleClass) -> list[str]:
    """Build a list of short text prompts for SAM3 open-vocabulary segmentation."""
    # SAM3 expects concise terms, not editing instructions
    prompts = [organelle.name]
    # Add morphological synonyms per organelle
    _SYNONYMS: dict[str, list[str]] = {
        "mito": ["oval", "circle", "organelle", "mitochondrion", "ellipse"],
        "er": ["membrane", "tubule", "network", "reticulum"],
        "nucleus": ["nucleus", "oval", "circle", "cell nucleus"],
        "lipid_droplet": ["droplet", "sphere", "circle", "vesicle"],
        "plasma_membrane": ["membrane", "cell boundary", "border"],
        "nuclear_envelope": ["nuclear membrane", "membrane", "envelope"],
        "nuclear_pore": ["pore", "ring", "dot"],
        "nucleolus": ["nucleolus", "sphere", "dense body"],
        "heterochromatin": ["dark region", "dense chromatin", "chromatin"],
        "euchromatin": ["light region", "diffuse chromatin", "chromatin"],
        "cell": ["cell", "oval", "blob"],
    }
    prompts.extend(_SYNONYMS.get(organelle.key, []))
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _get_param(params: GenerationParams, name: str) -> float:
    """Get a parameter value by name."""
    return float(getattr(params, name))


def _step_param(
    params: GenerationParams,
    name: str,
    direction: int,
    tunable: dict[str, tuple[float, float]] | None = None,
) -> GenerationParams:
    """Change one parameter by ±5%, clamped to allowed range."""
    current_val = _get_param(params, name)
    delta = current_val * STEP_FRACTION * direction
    new_val = current_val + delta

    # Clamp to allowed range
    if tunable and name in tunable:
        lo, hi = tunable[name]
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


def _revert_param(
    params: GenerationParams, name: str, effect: dict
) -> GenerationParams:
    """Revert a parameter to its value before the last change."""
    return _with_param(params, name, effect["old_value"])


def _with_extra(params: GenerationParams, extra: dict) -> GenerationParams:
    """Return a copy of params with updated extra dict."""
    return GenerationParams(
        prompt=params.prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        strength=params.strength,
        seed=params.seed,
        threshold=params.threshold,
        extra=extra,
    )


def _apply_point_refinement(
    params: GenerationParams, refinement: PointRefinement
) -> GenerationParams:
    """Apply point additions/removals from evaluator feedback."""
    current_points = list(params.extra.get("points", []))

    # Remove points by index (in reverse order to preserve indices)
    for idx in sorted(refinement.remove_indices, reverse=True):
        if 0 <= idx < len(current_points):
            current_points.pop(idx)

    # Add new points, ensuring foreground points have instance IDs
    max_instance = max(
        (p.get("instance", -1) for p in current_points if p.get("label", 1) == 1),
        default=-1,
    )
    for p in refinement.add_points:
        if p.get("label", 1) == 1 and "instance" not in p:
            max_instance += 1
            p["instance"] = max_instance
        current_points.append(p)

    new_extra = {**params.extra, "points": current_points}
    return _with_extra(params, new_extra)


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

    # Save point visualization if points were used
    points = result.params_used.extra.get("points")
    if points:
        points_img = _draw_points_on_image(em_image, points)
        points_img.save(iter_dir / "points.png")


def _draw_points_on_image(
    em_image: Image.Image, points: list[dict]
) -> Image.Image:
    """Draw point prompts on the EM image, color-coded by instance ID."""
    from PIL import ImageDraw

    # Distinct colors for instances (cycle if more than 10)
    INSTANCE_COLORS = [
        (0, 255, 0), (0, 200, 255), (255, 255, 0), (255, 0, 255),
        (255, 165, 0), (0, 255, 200), (200, 100, 255), (100, 255, 100),
        (255, 200, 100), (100, 200, 255),
    ]
    BG_COLOR = (255, 0, 0)  # red for background points

    img = em_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    # Scale radius to image size — at least 6px, up to ~2% of the shorter side
    radius = max(6, min(img.width, img.height) // 50)

    for i, p in enumerate(points):
        x, y = p["x"], p["y"]
        label = p.get("label", 1)
        if label == 1:
            inst_id = p.get("instance", i)
            color = INSTANCE_COLORS[inst_id % len(INSTANCE_COLORS)]
        else:
            color = BG_COLOR
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(255, 255, 255),
            width=max(1, radius // 4),
        )
        # Label: "index:instance" for fg, just index for bg
        if label == 1:
            label_text = f"{i}:i{p.get('instance', i)}"
        else:
            label_text = f"{i}:bg"
        draw.text(
            (x + radius + 2, y - radius),
            label_text,
            fill=(255, 255, 255),
        )

    return img


def _save_evaluation(
    output_dir: Path,
    iteration: int,
    evaluation: EvaluationResult,
    params: GenerationParams,
    param_change: dict | None = None,
    init_vlm_prompts: dict | None = None,
) -> None:
    """Save generation + evaluation results as JSON."""
    iter_dir = output_dir / f"iteration_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    sam3_strategy = params.extra.get("sam3_strategy")

    # --- Generation section: what was used to produce the segmentation ---
    generation: dict = {}

    if sam3_strategy:
        generation["sam3_strategy"] = sam3_strategy

    if sam3_strategy in ("vlm-coordinate", "painted-marker"):
        if "points" in params.extra:
            generation["points"] = params.extra["points"]
    else:
        generation["prompt"] = params.prompt

    generation["guidance_scale"] = params.guidance_scale
    generation["num_inference_steps"] = params.num_inference_steps
    generation["threshold"] = params.threshold
    generation["seed"] = params.seed
    if params.strength is not None:
        generation["strength"] = params.strength

    if param_change:
        generation["param_change"] = param_change

    # VLM prompts used to generate initial points or initial prompt (iteration 0)
    if init_vlm_prompts:
        generation["init_vlm_prompts"] = init_vlm_prompts

    # --- Evaluation section: VLM critique of the result ---
    eval_section: dict = {
        "score": evaluation.score,
        "detailed_scores": None,
        "issues": evaluation.issues,
        "reasoning": evaluation.reasoning,
        "refined_prompt": evaluation.refined_prompt,
    }

    if evaluation.detailed_scores:
        ds = evaluation.detailed_scores
        eval_section["detailed_scores"] = {
            "tp_rate": ds.tp_rate,
            "fp_rate": ds.fp_rate,
            "fn_rate": ds.fn_rate,
            "boundary_quality": ds.boundary_quality,
            "dice_score": ds.dice_score,
        }

    if evaluation.vlm_prompts:
        eval_section["vlm_prompts"] = evaluation.vlm_prompts

    if evaluation.point_refinement:
        pr = evaluation.point_refinement
        eval_section["point_refinement"] = {
            "add_points": pr.add_points,
            "remove_indices": pr.remove_indices,
            "reasoning": pr.reasoning,
        }

    eval_section["raw_response"] = evaluation.raw_response

    data = {
        "generation": generation,
        "evaluation": eval_section,
    }

    with open(iter_dir / "evaluation.json", "w") as f:
        json.dump(data, f, indent=4)
