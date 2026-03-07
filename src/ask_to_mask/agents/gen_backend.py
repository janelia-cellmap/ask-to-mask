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


class GeminiImageBackend(ImageGenBackend):
    """Google Gemini/Imagen image editing via the GenAI API.

    For Gemini models: uses generate_content with image modality (API key).
    For Imagen models: uses edit_image via Vertex AI (requires GCP project).
    """

    def __init__(
        self,
        model: str = "gemini-3-pro-image-preview",
        api_key: str | None = None,
        gcp_project: str | None = None,
        gcp_location: str = "us-central1",
        organelle_rgb: tuple[int, int, int] = (255, 0, 0),
        **_kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.gcp_project = gcp_project
        self.gcp_location = gcp_location
        self.organelle_rgb = organelle_rgb

    def _make_client(self):
        import os

        from google import genai

        is_imagen = self.model.startswith("imagen")

        if is_imagen:
            # Imagen requires Vertex AI client
            project = self.gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise ValueError(
                    "Imagen models require Vertex AI. Set GOOGLE_CLOUD_PROJECT "
                    "env var or pass --gcp-project. You also need `gcloud auth "
                    "application-default login`."
                )
            return genai.Client(
                vertexai=True,
                project=project,
                location=self.gcp_location,
            )

        # Gemini models use API key
        if self.api_key:
            return genai.Client(api_key=self.api_key)
        return genai.Client()

    def generate(
        self, image: Image.Image, params: GenerationParams, iteration: int = 0
    ) -> GenerationResult:
        import time

        from google.genai.errors import ClientError

        from ..postprocess import extract_mask

        client = self._make_client()
        is_imagen = self.model.startswith("imagen")

        for attempt in range(3):
            try:
                if is_imagen:
                    generated_image = self._generate_imagen(
                        client, image, params.prompt
                    )
                else:
                    generated_image = self._generate_gemini(
                        client, image, params.prompt
                    )
                break
            except ClientError as e:
                if "429" in str(e) and attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"  Rate limited — waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise

        # Resize generated image to match input if needed
        if generated_image.size != image.size:
            generated_image = generated_image.resize(image.size, Image.LANCZOS)

        mask = extract_mask(
            image.convert("RGB"),
            generated_image,
            self.organelle_rgb,
            threshold=params.threshold,
        )
        mask_image = Image.fromarray(mask, mode="L")

        return GenerationResult(
            input_image=image,
            colored_image=generated_image,
            mask=mask,
            mask_image=mask_image,
            params_used=params,
            iteration=iteration,
        )


    def _generate_gemini(self, client, image: Image.Image, prompt: str) -> Image.Image:
        """Generate via Gemini's generate_content with image modality."""
        import io

        from google import genai

        response = client.models.generate_content(
            model=self.model,
            contents=[image, prompt],
            config=genai.types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return Image.open(
                    io.BytesIO(part.inline_data.data)
                ).convert("RGB")

        raise RuntimeError(
            f"Gemini returned no image. Response text: "
            f"{response.text[:500] if response.text else '(empty)'}"
        )

    def _generate_imagen(self, client, image: Image.Image, prompt: str) -> Image.Image:
        """Generate via Imagen's edit_image API."""
        import io

        from google import genai

        # Convert PIL image to bytes for the API
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        ref_image = genai.types.RawReferenceImage(
            referenceImage=genai.types.Image(
                imageBytes=image_bytes, mimeType="image/png"
            ),
            referenceId=0,
        )

        response = client.models.edit_image(
            model=self.model,
            prompt=prompt,
            reference_images=[ref_image],
            config=genai.types.EditImageConfig(
                number_of_images=1,
            ),
        )

        if not response.generated_images:
            raise RuntimeError("Imagen returned no images.")

        return response.generated_images[0].image.to_pil().convert("RGB")


def create_gen_backend(backend: str, **kwargs) -> ImageGenBackend:
    """Factory function for image generation backends.

    Args:
        backend: Backend name. "flux" or "gemini".
        **kwargs: Backend-specific arguments.
    """
    if backend == "flux":
        kwargs.pop("gcp_project", None)
        return FluxBackend(**kwargs)
    if backend == "gemini":
        # Map model_key to gemini model param, remove flux-specific kwargs
        model_key = kwargs.pop("model_key", None)
        kwargs.pop("lora_path", None)
        kwargs.pop("device", None)
        # Allow overriding the gemini model via --model
        if model_key and model_key not in ("kontext-dev", "flux2-dev"):
            kwargs["model"] = model_key
        return GeminiImageBackend(**kwargs)
    raise ValueError(f"Unknown image generation backend: {backend!r}")
