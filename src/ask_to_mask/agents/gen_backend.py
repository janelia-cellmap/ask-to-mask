"""Pluggable image generation backends for mask generation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from .schemas import GenerationParams, GenerationResult


class ImageGenBackend(ABC):
    """Abstract base for image generation models.

    Each backend takes an EM image + generation params and returns a
    GenerationResult with colored image and extracted mask.
    """

    @abstractmethod
    def generate(
        self, image: Image.Image, params: GenerationParams, iteration: int = 0
    ) -> GenerationResult:
        """Generate a colored image from an EM input + prompt, then extract a mask."""
        ...


class FluxBackend(ImageGenBackend):
    """Wraps the existing Flux pipeline (Kontext, Flux2, standard img2img)."""

    def __init__(
        self,
        model_key: str = "kontext-dev",
        lora_path: str | None = None,
        device: str = "cuda",
        organelle_rgb: tuple[int, int, int] = (255, 0, 0),
    ):
        from ..model import load_pipeline

        self.pipe = load_pipeline(model_key, device=device, lora_weights=lora_path)
        self.organelle_rgb = organelle_rgb

    def generate(
        self, image: Image.Image, params: GenerationParams, iteration: int = 0
    ) -> GenerationResult:
        from ..model import run_inference
        from ..pipeline import TARGET_SIZE, pad_to_square, unpad
        from ..postprocess import extract_mask

        img_padded, crop_box = pad_to_square(image)
        img_resized = img_padded.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

        result = run_inference(
            self.pipe,
            img_resized,
            params.prompt,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            strength=params.strength,
            seed=params.seed,
        )

        result_cropped = unpad(result, crop_box, TARGET_SIZE)
        img_resized_cropped = unpad(img_resized, crop_box, TARGET_SIZE)

        mask = extract_mask(
            img_resized_cropped,
            result_cropped,
            self.organelle_rgb,
            threshold=params.threshold,
        )
        mask_image = Image.fromarray(mask, mode="L")

        return GenerationResult(
            input_image=img_resized_cropped,
            colored_image=result_cropped,
            mask=mask,
            mask_image=mask_image,
            params_used=params,
            iteration=iteration,
        )


def create_gen_backend(backend: str, **kwargs) -> ImageGenBackend:
    """Factory function for image generation backends.

    Args:
        backend: Backend name (currently "flux").
        **kwargs: Backend-specific arguments.
    """
    if backend == "flux":
        return FluxBackend(**kwargs)
    raise ValueError(f"Unknown image generation backend: {backend!r}")
