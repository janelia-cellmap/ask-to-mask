"""Flux model loading and inference."""

from __future__ import annotations

import torch
from PIL import Image

from .config import MODELS


def load_pipeline(
    model_key: str, device: str = "cuda", lora_weights: str | None = None
):
    """Load a Flux pipeline by model key, optionally with LoRA weights.

    Args:
        model_key: Key from config.MODELS (e.g., "kontext-dev", "flux2-dev").
        device: Torch device string.
        lora_weights: Path to LoRA weights directory or HuggingFace repo ID.

    Returns:
        A diffusers pipeline ready for inference.
    """
    model_id = MODELS[model_key]

    if "Kontext" in model_id:
        from diffusers import FluxKontextPipeline

        pipe = FluxKontextPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
    elif "FLUX.2" in model_id:
        from diffusers import Flux2Pipeline

        pipe = Flux2Pipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
    else:
        from diffusers import FluxImg2ImgPipeline

        pipe = FluxImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )

    if lora_weights is not None:
        pipe.load_lora_weights(lora_weights)

    pipe.to(device)
    return pipe


def run_inference(
    pipe,
    image: Image.Image,
    prompt: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    strength: float = 0.75,
    seed: int | None = None,
) -> Image.Image:
    """Run image-to-image inference.

    Args:
        pipe: Loaded diffusers pipeline.
        image: Input PIL image (RGB).
        prompt: Text prompt describing the edit.
        num_inference_steps: Number of denoising steps.
        guidance_scale: How strongly to follow the prompt.
        strength: How much to modify the input (only used by img2img pipelines, ignored by Kontext).
        seed: Random seed for reproducibility.

    Returns:
        Modified PIL image.
    """
    from diffusers import FluxKontextPipeline

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    kwargs = dict(
        image=image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    # Only FluxImg2ImgPipeline supports 'strength'. FluxKontextPipeline and
    # Flux2Pipeline use the image as context, not as an img2img starting point.
    if not isinstance(pipe, FluxKontextPipeline):
        try:
            from diffusers import Flux2Pipeline

            if isinstance(pipe, Flux2Pipeline):
                pass  # Flux2Pipeline doesn't use strength either
            else:
                kwargs["strength"] = strength
        except ImportError:
            kwargs["strength"] = strength

    result = pipe(**kwargs).images[0]

    return result
