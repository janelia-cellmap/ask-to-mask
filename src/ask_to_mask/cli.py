"""CLI entry point for ask-to-mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    seg.add_argument("--device", default="cuda", help="Torch device (default: cuda).")

    # --- list-organelles ---
    sub.add_parser("list-organelles", help="List available organelle classes.")

    return parser.parse_args(argv)


def cmd_list_organelles() -> None:
    print("Available organelle classes:\n")
    for key, org in ORGANELLES.items():
        print(f"  {key:12s}  {org.name} ({org.color_name}, RGB {org.rgb})")
        print(f"               Prompt: \"{org.prompt}\"")
        print()


def cmd_segment(args: argparse.Namespace) -> None:
    from .model import load_pipeline
    from .pipeline import segment

    print(f"Loading model: {args.model} ({MODELS[args.model]})")
    pipe = load_pipeline(args.model, device=args.device)

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
    )

    for path in image_paths:
        print(f"\n--- {path.name} ---")
        masks = segment(pipe, path, args.organelles, args.output_dir, **kwargs)
        for m in masks:
            print(f"  Saved: {m}")

    print("\nDone.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "list-organelles":
        cmd_list_organelles()
    elif args.command == "segment":
        cmd_segment(args)


if __name__ == "__main__":
    main()
