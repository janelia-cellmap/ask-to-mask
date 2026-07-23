"""Generate images or videos with Google's GenAI SDK.

This is a lightweight script for testing CellMap-billed Vertex/Agent Platform
media generation outside the segmentation pipeline.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import time
from pathlib import Path


def _make_client(args: argparse.Namespace):
    from google import genai

    if args.api_key:
        return genai.Client(api_key=args.api_key)

    project = args.gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = args.gcp_location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
    if not project:
        raise SystemExit(
            "Vertex mode requires --gcp-project or GOOGLE_CLOUD_PROJECT."
        )
    return genai.Client(vertexai=True, project=project, location=location)


def _image_part(path: Path) -> dict:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image", "data": data, "mime_type": mime_type}


def cmd_image(args: argparse.Namespace) -> None:
    client = _make_client(args)
    prompt_input: list[dict] | str
    if args.input_image:
        prompt_input = [{"type": "text", "text": args.prompt}]
        prompt_input.extend(_image_part(Path(p)) for p in args.input_image)
    else:
        prompt_input = args.prompt

    if hasattr(client, "interactions"):
        image_bytes = _generate_image_interactions(client, args.model, prompt_input)
    else:
        image_bytes = _generate_image_content(client, args.model, args.prompt, args.input_image)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(image_bytes)
    print(output)


def _generate_image_interactions(client, model: str, prompt_input: list[dict] | str) -> bytes:
    interaction = client.interactions.create(model=model, input=prompt_input)
    image = getattr(interaction, "output_image", None)
    if image is None:
        text = getattr(interaction, "output_text", "")
        raise RuntimeError(
            "Google GenAI returned no output_image. "
            f"Text response: {text[:500] if text else '(empty)'}"
        )
    return base64.b64decode(image.data)


def _generate_image_content(
    client,
    model: str,
    prompt: str,
    input_images: list[str],
) -> bytes:
    from google import genai
    from PIL import Image

    contents = [Image.open(path).convert("RGB") for path in input_images]
    contents.append(prompt)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=genai.types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise RuntimeError(
        "Google GenAI returned no image. "
        f"Text response: {response.text[:500] if response.text else '(empty)'}"
    )


def cmd_video(args: argparse.Namespace) -> None:
    if not args.output_gcs_uri:
        raise SystemExit("Video generation requires --output-gcs-uri in Vertex mode.")

    from google.genai.types import GenerateVideosConfig

    client = _make_client(args)
    operation = client.models.generate_videos(
        model=args.model,
        prompt=args.prompt,
        config=GenerateVideosConfig(
            aspect_ratio=args.aspect_ratio,
            output_gcs_uri=args.output_gcs_uri,
        ),
    )

    while not operation.done:
        time.sleep(args.poll_seconds)
        operation = client.operations.get(operation)
        print(operation)

    if not operation.response:
        raise RuntimeError("Video generation finished without a response.")

    video = operation.result.generated_videos[0].video
    print(video.uri)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images or videos using Google GenAI."
    )
    parser.add_argument("--gcp-project", default=None, help="Google Cloud project ID.")
    parser.add_argument(
        "--gcp-location",
        default="global",
        help="Google Cloud location for Vertex/Agent Platform (default: global).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Use Gemini API key mode instead of Vertex/Agent Platform ADC.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    image = sub.add_parser("image", help="Generate or edit an image.")
    image.add_argument("--prompt", required=True)
    image.add_argument(
        "--input-image",
        action="append",
        default=[],
        help="Optional reference image. Repeat for multiple images.",
    )
    image.add_argument("--output", required=True, help="Output image path.")
    image.add_argument(
        "--model",
        default="gemini-3.1-flash-image",
        help="Image model (default: gemini-3.1-flash-image).",
    )
    image.set_defaults(func=cmd_image)

    video = sub.add_parser("video", help="Generate a video from text.")
    video.add_argument("--prompt", required=True)
    video.add_argument(
        "--output-gcs-uri",
        required=True,
        help="GCS prefix for generated video output, e.g. gs://bucket/prefix.",
    )
    video.add_argument(
        "--model",
        default="veo-3.1-fast-generate-001",
        help="Video model (default: veo-3.1-fast-generate-001).",
    )
    video.add_argument("--aspect-ratio", default="16:9")
    video.add_argument("--poll-seconds", type=int, default=15)
    video.set_defaults(func=cmd_video)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
