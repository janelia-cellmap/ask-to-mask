"""CLI entry point for ask-to-mask."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env file from project root if it exists."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

from .config import DEFAULT_MODEL, MODELS, ORGANELLES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ask-to-mask",
        description="Generate organelle segmentation masks from EM images using Flux.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- segment ---
    seg = sub.add_parser("segment", help="Segment organelles in EM images.")
    input_group = seg.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Path to a single EM image.")
    input_group.add_argument(
        "--input-dir", type=Path, help="Directory of EM images for batch processing."
    )

    seg.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save output masks.",
    )
    seg.add_argument(
        "--organelles",
        nargs="+",
        required=True,
        choices=list(ORGANELLES.keys()),
        help="Organelle classes to segment.",
    )
    seg.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Flux model to use (default: {DEFAULT_MODEL}).",
    )
    seg.add_argument("--num-steps", type=int, default=28, help="Inference steps.")
    seg.add_argument(
        "--guidance-scale", type=float, default=3.5, help="Guidance scale."
    )
    seg.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="Edit strength (0=no change, 1=full regeneration).",
    )
    seg.add_argument("--seed", type=int, default=None, help="Random seed.")
    seg.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Color difference threshold for mask extraction.",
    )
    seg.add_argument(
        "--save-colored",
        action="store_true",
        help="Save the colored intermediate image alongside the mask.",
    )
    seg.add_argument(
        "--custom-prompt", type=str, default=None, help="Override the default prompt."
    )
    seg.add_argument(
        "--instance",
        action="store_true",
        help="Instance segmentation: color each instance a different color, then label connected components.",
    )
    seg.add_argument(
        "--detailed-prompt",
        action="store_true",
        help="Include EM-specific organelle descriptions in the prompt.",
    )
    seg.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Image resolution in nm/px. Included in the prompt to help the model understand scale.",
    )
    seg.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    seg.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path to LoRA weights directory or HuggingFace repo ID.",
    )

    # --- train ---
    train_parser = sub.add_parser(
        "train", help="Fine-tune Flux with LoRA on CellMap data."
    )
    train_parser.add_argument(
        "--config", type=Path, required=True, help="Training config YAML."
    )
    train_parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint directory."
    )

    # --- refine ---
    ref = sub.add_parser(
        "refine",
        help="Iteratively refine mask quality using a VLM evaluator agent.",
    )
    ref.add_argument("--input", type=Path, required=True, help="Path to an EM image.")
    ref.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save results.",
    )
    ref.add_argument(
        "--organelle",
        required=True,
        choices=list(ORGANELLES.keys()),
        help="Organelle class to segment.",
    )
    ref.add_argument(
        "--gen-backend",
        default="flux",
        choices=["flux", "gemini", "glm", "qwen", "sam3"],
        help="Image generation backend (default: flux).",
    )
    ref.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use. For flux: {list(MODELS.keys())}. For gemini: any image-capable model name. For glm: HF repo ID (default: zai-org/GLM-Image). For qwen: HF repo ID (default: Qwen/Qwen-Image-Edit-2511).",
    )
    ref.add_argument("--lora", type=str, default=None, help="Path to LoRA weights.")
    ref.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    ref.add_argument(
        "--gcp-project",
        default=None,
        help="GCP project ID for Vertex AI.",
    )
    ref.add_argument(
        "--gcp-location",
        default="us-central1",
        help="GCP location for Vertex AI (default: us-central1).",
    )
    ref.add_argument(
        "--vertex-ai",
        action="store_true",
        help="Use Vertex AI (service account / ADC) instead of API key.",
    )
    ref.add_argument(
        "--llm-provider",
        default="google",
        choices=["ollama", "anthropic", "google", "openai", "huggingface"],
        help="LLM/VLM provider for evaluation (default: google).",
    )
    ref.add_argument("--llm-model", default=None, help="LLM model name.")
    ref.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server URL.",
    )
    ref.add_argument(
        "--max-iterations", type=int, default=5, help="Max refinement iterations."
    )
    ref.add_argument(
        "--min-score", type=float, default=0.95, help="Min acceptable score to stop."
    )
    ref.add_argument(
        "--save-intermediates",
        action="store_true",
        default=True,
        help="Save each iteration's outputs (default: True).",
    )
    ref.add_argument(
        "--num-steps",
        type=int,
        default=28,
        help="Inference steps (default: 28 for flux/gemini, 50 for glm, 40 for qwen).",
    )
    ref.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale (default: 3.5 for flux/gemini, 1.5 for glm, 1.0 for qwen).",
    )
    ref.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="Qwen only: true_cfg_scale (default: 4.0).",
    )
    ref.add_argument(
        "--strength", type=float, default=0.75, help="Edit strength (flux only)."
    )
    ref.add_argument("--threshold", type=float, default=30.0, help="Mask threshold.")
    ref.add_argument("--seed", type=int, default=None, help="Random seed.")
    ref.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Image resolution in nm/px. Included in the prompt to help the model understand scale.",
    )
    ref.add_argument(
        "--instance",
        action="store_true",
        help="Instance segmentation: color each instance a different color.",
    )
    ref.add_argument(
        "--mask-mode",
        default="overlay",
        choices=["overlay", "direct", "invert"],
        help="Mask extraction mode: 'overlay' colors organelles on the EM image; 'direct' asks for white-on-black mask; 'invert' segments background/edges then inverts to get instances (default: overlay).",
    )
    # SAM3-specific arguments
    ref.add_argument(
        "--sam3-strategy",
        default="text",
        choices=["text", "vlm-coordinate", "painted-marker"],
        help="SAM3 prompting strategy (default: text). Only used with --gen-backend sam3.",
    )
    ref.add_argument(
        "--sam3-model",
        default="facebook/sam3",
        help="SAM3 HuggingFace model ID (default: facebook/sam3).",
    )
    ref.add_argument(
        "--sam3-confidence",
        type=float,
        default=0.5,
        help="SAM3 mask confidence threshold (default: 0.5).",
    )
    ref.add_argument(
        "--marker-backend",
        default=None,
        choices=["flux", "gemini", "glm", "qwen"],
        help="Gen backend for painting markers (only for --sam3-strategy painted-marker).",
    )

    # --- list-organelles ---
    sub.add_parser("list-organelles", help="List available organelle classes.")

    return parser.parse_args(argv)


def cmd_list_organelles() -> None:
    print("Available organelle classes:\n")
    for key, org in ORGANELLES.items():
        print(f"  {key:12s}  {org.name} ({org.color_name}, RGB {org.rgb})")
        print(f'               Prompt: "{org.prompt}"')
        print()


def cmd_segment(args: argparse.Namespace) -> None:
    from .model import load_pipeline
    from .pipeline import segment

    print(f"Loading model: {args.model} ({MODELS[args.model]})")
    pipe = load_pipeline(args.model, device=args.device, lora_weights=args.lora)

    # Collect image paths
    if args.input:
        image_paths = [args.input]
    else:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        image_paths = sorted(
            p for p in args.input_dir.iterdir() if p.suffix.lower() in exts
        )
        if not image_paths:
            print(f"No images found in {args.input_dir}")
            sys.exit(1)

    print(f"Processing {len(image_paths)} image(s) for organelles: {args.organelles}")

    kwargs = dict(
        model_key=args.model,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        seed=args.seed,
        threshold=args.threshold,
        save_colored=args.save_colored,
        custom_prompt=args.custom_prompt,
        instance=args.instance,
        detailed_prompt=args.detailed_prompt,
        resolution_nm=args.resolution,
    )

    for path in image_paths:
        print(f"\n--- {path.name} ---")
        masks = segment(pipe, path, args.organelles, args.output_dir, **kwargs)
        for m in masks:
            print(f"  Saved: {m}")

    print("\nDone.")


def cmd_refine(args: argparse.Namespace) -> None:
    from .agents import (
        GenerationParams,
        LoopConfig,
        create_gen_backend,
        create_llm_backend,
        run_refinement_loop,
    )
    from .pipeline import load_em_image

    organelle = ORGANELLES[args.organelle]

    # Build LLM backend kwargs
    llm_kwargs = {}
    if args.llm_model:
        llm_kwargs["model"] = args.llm_model
    if args.llm_provider == "ollama":
        llm_kwargs["host"] = args.ollama_host
    if args.vertex_ai:
        llm_kwargs["vertex_ai"] = True
        llm_kwargs["gcp_project"] = args.gcp_project
        llm_kwargs["gcp_location"] = args.gcp_location

    llm_backend = create_llm_backend(args.llm_provider, **llm_kwargs)

    # Build gen backend
    if args.gen_backend == "sam3":
        sam3_kwargs = {
            "strategy": args.sam3_strategy,
            "model_name": args.sam3_model,
            "device": args.device,
            "organelle_rgb": organelle.rgb,
            "confidence_threshold": args.sam3_confidence,
        }
        if args.sam3_strategy == "vlm-coordinate":
            sam3_kwargs["llm_backend"] = llm_backend
        if args.sam3_strategy == "painted-marker":
            if not args.marker_backend:
                raise SystemExit(
                    "Error: --marker-backend is required with --sam3-strategy painted-marker"
                )
            sam3_kwargs["marker_gen_backend"] = create_gen_backend(
                args.marker_backend,
                model_key=args.model,
                lora_path=args.lora,
                device=args.device,
                organelle_rgb=organelle.rgb,
                gcp_project=args.gcp_project,
                gcp_location=args.gcp_location,
                vertex_ai=args.vertex_ai,
            )
        gen_backend = create_gen_backend("sam3", **sam3_kwargs)
    else:
        gen_backend = create_gen_backend(
            args.gen_backend,
            model_key=args.model,
            lora_path=args.lora,
            device=args.device,
            organelle_rgb=organelle.rgb,
            gcp_project=args.gcp_project,
            gcp_location=args.gcp_location,
            vertex_ai=args.vertex_ai,
        )

    em_image = load_em_image(args.input)
    instance = getattr(args, "instance", False)
    mask_mode = getattr(args, "mask_mode", "overlay")
    resolution_nm = getattr(args, "resolution", None)
    if mask_mode == "invert":
        initial_prompt = organelle.build_invert_prompt(
            detailed=False, resolution_nm=resolution_nm
        )
    elif instance:
        initial_prompt = organelle.build_instance_prompt(
            detailed=False, direct=(mask_mode == "direct"), resolution_nm=resolution_nm
        )
    elif mask_mode == "direct":
        initial_prompt = organelle.build_direct_prompt(
            detailed=False, resolution_nm=resolution_nm
        )
    else:
        initial_prompt = organelle.build_prompt(
            detailed=False, resolution_nm=resolution_nm
        )

    # Backend-specific defaults
    BACKEND_DEFAULTS = {
        "flux": {"num_inference_steps": 28, "guidance_scale": 3.5, "strength": 0.75},
        "glm": {"num_inference_steps": 50, "guidance_scale": 1.5},
        "gemini": {"num_inference_steps": 28, "guidance_scale": 3.5},
        "qwen": {"num_inference_steps": 40, "guidance_scale": 1.0},
        "sam3": {"num_inference_steps": 1, "guidance_scale": 1.0},
    }
    defaults = BACKEND_DEFAULTS.get(args.gen_backend, {})

    param_kwargs: dict = {
        "prompt": initial_prompt,
        "num_inference_steps": (
            args.num_steps
            if args.num_steps != 28
            else defaults.get("num_inference_steps", 28)
        ),
        "guidance_scale": (
            args.guidance_scale
            if args.guidance_scale != 3.5
            else defaults.get("guidance_scale", 3.5)
        ),
        "seed": args.seed,
        "threshold": args.threshold,
        "extra": {},
    }
    # Only include strength for backends that use it
    if "strength" in defaults:
        param_kwargs["strength"] = (
            args.strength if args.strength != 0.75 else defaults["strength"]
        )
    else:
        param_kwargs["strength"] = None

    if args.gen_backend == "qwen":
        param_kwargs["extra"]["true_cfg_scale"] = args.true_cfg_scale
        param_kwargs["extra"]["negative_prompt"] = " "

    if args.gen_backend == "sam3":
        param_kwargs["strength"] = None
        param_kwargs["extra"]["sam3_confidence_threshold"] = args.sam3_confidence
        param_kwargs["extra"]["sam3_strategy"] = args.sam3_strategy
        param_kwargs["extra"]["organelle_name"] = organelle.name
        param_kwargs["extra"]["color_name"] = organelle.color_name
        # For text mode, use the organelle name as the prompt
        if args.sam3_strategy == "text":
            param_kwargs["prompt"] = organelle.name

    initial_params = GenerationParams(**param_kwargs)

    config = LoopConfig(
        max_iterations=args.max_iterations,
        min_acceptable_score=args.min_score,
        save_intermediates=args.save_intermediates,
    )

    # Build output subdirectory: gen_model_eval_model/datetime
    from datetime import datetime

    if args.gen_backend == "flux":
        gen_model_name = args.model
    elif args.gen_backend == "sam3":
        gen_model_name = f"sam3-{args.sam3_strategy}"
    else:
        gen_model_name = getattr(gen_backend, "model", args.gen_backend)
    eval_model_name = getattr(llm_backend, "model", args.llm_provider)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.output_dir
        / f"{gen_model_name}_{eval_model_name}"
        / args.organelle
        / timestamp
    )

    print(f"Refining {organelle.name} segmentation on {args.input.name}")
    if args.gen_backend == "flux":
        print(f"  Gen backend: flux ({args.model})")
    else:
        print(f"  Gen backend: {args.gen_backend} ({gen_model_name})")
    print(f"  Evaluator: {args.llm_provider} ({eval_model_name})")
    print(f"  Max iterations: {args.max_iterations}, min score: {args.min_score}")
    print(f"  Output: {run_dir}")

    result = run_refinement_loop(
        gen_backend=gen_backend,
        llm_backend=llm_backend,
        em_image=em_image,
        organelle=organelle,
        initial_params=initial_params,
        config=config,
        output_dir=run_dir,
        instance=instance,
        mask_mode=mask_mode,
        gen_model=gen_model_name,
        resolution_nm=resolution_nm,
        llm_model=getattr(args, "llm_model", "") or "",
    )

    if result.converged:
        status = "converged"
    elif result.plateau:
        status = "score plateau"
    else:
        status = "max iterations reached"
    print(f"\n=== Done ({status}) ===")
    print(f"  Best score: {result.best_evaluation.score:.3f}")
    if result.best_evaluation.detailed_scores:
        ds = result.best_evaluation.detailed_scores
        print(
            f"  Details: tp_rate={ds.tp_rate:.2f}  fp_rate={ds.fp_rate:.2f}  "
            f"fn_rate={ds.fn_rate:.2f}  boundary={ds.boundary_quality:.2f}  "
            f"dice={ds.dice_score:.3f}"
        )
    print(f"  Total iterations: {result.total_iterations}")
    if args.save_intermediates:
        print(f"  Results saved to: {args.output_dir}")



def cmd_train(args: argparse.Namespace) -> None:
    from .training.train import train

    train(config_path=str(args.config), resume_from=args.resume)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "list-organelles":
        cmd_list_organelles()
    elif args.command == "segment":
        cmd_segment(args)
    elif args.command == "refine":
        cmd_refine(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
