#!/usr/bin/env python
"""Standalone Molmo2 point detection script.

Runs in the 'molmo' pixi environment (transformers <5) to avoid
incompatibilities with transformers 5.x.

Usage:
    .pixi/envs/molmo/bin/python scripts/molmo_points.py \
        --image path/to/image.png \
        --prompt "Point to the mitochondria" \
        --model allenai/Molmo2-8B

Outputs JSON to stdout:
    {"points": [{"x": 123, "y": 456}, ...], "raw": "model output..."}
"""

import argparse
import json
import re
import sys

import torch
from PIL import Image


def load_model(model_name: str):
    """Load Molmo2 model and processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    )
    return processor, model


def generate(processor, model, image: Image.Image, prompt: str) -> str:
    """Run Molmo2 inference and return raw text output."""
    # Molmo2 processor requires <|image|> placeholders in text
    placeholder = getattr(processor, "image_placeholder_token", "<|image|>")
    full_prompt = f"{placeholder} {prompt}"
    inputs = processor(text=full_prompt, images=[image], return_tensors="pt")
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=2048)

    generated = output[0, inputs["input_ids"].size(1) :]
    # Don't skip special tokens — Molmo2's <points coords="..."> tags
    # contain the coordinates and get stripped by skip_special_tokens=True.
    return processor.tokenizer.decode(generated, skip_special_tokens=False)


def parse_points(raw: str, width: int, height: int) -> list[dict]:
    """Parse Molmo2's point output to pixel coordinates.

    Molmo2 uses coords format: <points ... coords="1 0 523 412"/>
    where x,y are scaled by 1000.

    Also handles legacy Molmo1 formats (0-100 normalized).
    """
    points: list[dict] = []

    # Molmo2 coords format: <points coords="1 0 523 412"/>
    # The <points coords=" prefix may be stripped as a special token,
    # leaving bare coordinates like: 1 1 677 717 2 680 752...">label</points>
    coord_regex = re.compile(r'coords="([^"]+)"')
    points_num_regex = re.compile(r"(\d+)\s+(\d{3,4})\s+(\d{3,4})")

    # Try full coords="..." match first
    for coord_match in coord_regex.finditer(raw):
        coord_str = coord_match.group(1)
        for m in points_num_regex.finditer(coord_str):
            px_x = max(0, min(width - 1, int(float(m.group(2)) / 1000 * width)))
            px_y = max(0, min(height - 1, int(float(m.group(3)) / 1000 * height)))
            points.append({"x": px_x, "y": px_y, "label": 1})

    # Fallback: bare coordinate triplets (when coords=" prefix was stripped)
    # Look for sequences of (index, x_scaled, y_scaled) before ">
    if not points:
        # Extract everything before the closing "> of the points tag
        bare_match = re.search(r'((?:\d+\s+\d{3,4}\s+\d{3,4}\s*)+)"?\s*>', raw)
        if bare_match:
            for m in points_num_regex.finditer(bare_match.group(1)):
                px_x = max(0, min(width - 1, int(float(m.group(2)) / 1000 * width)))
                px_y = max(0, min(height - 1, int(float(m.group(3)) / 1000 * height)))
                points.append({"x": px_x, "y": px_y, "label": 1})

    if points:
        return points

    # Legacy Molmo1: multi-point format <points x1="26.0" y1="67.5" ...>
    points_tag = re.search(r"<points\s+([^>]+)>", raw)
    if points_tag:
        attrs = points_tag.group(1)
        xs = re.findall(r'x(\d+)\s*=\s*"([^"]+)"', attrs)
        ys = re.findall(r'y(\d+)\s*=\s*"([^"]+)"', attrs)
        y_map = {idx: val for idx, val in ys}
        for idx, x_val in xs:
            if idx in y_map:
                px_x = max(0, min(width - 1, int(float(x_val) * width / 100)))
                px_y = max(0, min(height - 1, int(float(y_map[idx]) * height / 100)))
                points.append({"x": px_x, "y": px_y, "label": 1})

    # Legacy Molmo1: single-point format <point x="56.2" y="32.7" ...>
    for m in re.finditer(r'<point\s+x\s*=\s*"([^"]+)"\s+y\s*=\s*"([^"]+)"', raw):
        px_x = max(0, min(width - 1, int(float(m.group(1)) * width / 100)))
        px_y = max(0, min(height - 1, int(float(m.group(2)) * height / 100)))
        points.append({"x": px_x, "y": px_y, "label": 1})

    return points


def _load_dotenv():
    """Load .env from project root if it exists."""
    import os
    from pathlib import Path

    env_file = Path(__file__).resolve().parents[1] / ".env"
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


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Molmo2 point detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Prompt for point detection")
    parser.add_argument(
        "--model", default="allenai/Molmo2-8B", help="HuggingFace model name"
    )
    parser.add_argument(
        "--output", default=None, help="Save image with points drawn (PNG path)"
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    w, h = image.size

    # Suppress warnings to keep stdout clean for JSON
    import warnings
    warnings.filterwarnings("ignore")

    processor, model = load_model(args.model)
    raw = generate(processor, model, image, args.prompt)
    points = parse_points(raw, w, h)

    if args.output and points:
        from PIL import ImageDraw

        vis = image.copy()
        draw = ImageDraw.Draw(vis)
        r = max(3, min(w, h) // 100)
        for i, pt in enumerate(points):
            x, y = pt["x"], pt["y"]
            draw.ellipse([x - r, y - r, x + r, y + r], fill="red", outline="white")
            draw.text((x + r + 2, y - r), str(i), fill="red")
        vis.save(args.output)
        print(f"Saved visualization to {args.output}", file=sys.stderr)

    result = {"points": points, "raw": raw[:2000]}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
