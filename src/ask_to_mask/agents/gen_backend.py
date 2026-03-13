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
        self,
        image: Image.Image,
        params: GenerationParams,
        iteration: int = 0,
        instance: bool = False,
        mask_mode: str = "overlay",
    ) -> GenerationResult:
        """Generate a colored image from an EM input + prompt, then extract a mask."""
        ...

    def _extract_mask(
        self,
        input_image: Image.Image,
        generated_image: Image.Image,
        organelle_rgb: tuple[int, int, int],
        threshold: float,
        instance: bool,
        mask_mode: str,
    ) -> tuple[np.ndarray, Image.Image]:
        """Extract mask from generated image using the appropriate mode."""
        from ..postprocess import (
            extract_direct_mask,
            extract_instance_mask,
            extract_invert_mask,
            extract_mask,
        )

        if mask_mode == "invert":
            mask = extract_invert_mask(generated_image, brightness_threshold=threshold)
            mask_image = Image.fromarray(mask)
        elif mask_mode == "direct":
            mask = extract_direct_mask(generated_image, brightness_threshold=threshold)
            mask_image = Image.fromarray(mask, mode="L")
        elif instance:
            mask = extract_instance_mask(
                input_image,
                generated_image,
                saturation_threshold=threshold,
            )
            mask_image = Image.fromarray(mask)
        else:
            mask = extract_mask(
                input_image,
                generated_image,
                organelle_rgb,
                threshold=threshold,
            )
            mask_image = Image.fromarray(mask, mode="L")
        return mask, mask_image


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
        self,
        image: Image.Image,
        params: GenerationParams,
        iteration: int = 0,
        instance: bool = False,
        mask_mode: str = "overlay",
    ) -> GenerationResult:
        from ..model import run_inference
        from ..pipeline import TARGET_SIZE, pad_to_square, unpad

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

        mask, mask_image = self._extract_mask(
            img_resized_cropped,
            result_cropped,
            self.organelle_rgb,
            params.threshold,
            instance,
            mask_mode,
        )

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
        vertex_ai: bool = False,
        organelle_rgb: tuple[int, int, int] = (255, 0, 0),
        **_kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.gcp_project = gcp_project
        self.gcp_location = gcp_location
        self.vertex_ai = vertex_ai
        self.organelle_rgb = organelle_rgb

    def _make_client(self):
        import os

        from google import genai

        is_imagen = self.model.startswith("imagen")

        if self.vertex_ai or is_imagen:
            # Vertex AI client — uses ADC (GOOGLE_APPLICATION_CREDENTIALS or gcloud login)
            project = self.gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise ValueError(
                    "Vertex AI requires a GCP project. Set GOOGLE_CLOUD_PROJECT "
                    "env var or pass --gcp-project."
                )
            return genai.Client(
                vertexai=True,
                project=project,
                location=self.gcp_location,
            )

        # API key client
        if self.api_key:
            return genai.Client(api_key=self.api_key)
        return genai.Client()

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        iteration: int = 0,
        instance: bool = False,
        mask_mode: str = "overlay",
    ) -> GenerationResult:
        import time

        from google.genai.errors import ClientError

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

        mask, mask_image = self._extract_mask(
            image.convert("RGB"),
            generated_image,
            self.organelle_rgb,
            params.threshold,
            instance,
            mask_mode,
        )

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
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")

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


class GLMImageBackend(ImageGenBackend):
    """Local GLM-Image model via diffusers (zai-org/GLM-Image).

    Requires diffusers and transformers installed from source.
    Needs ~40-80GB VRAM (CPU offload possible at ~23GB).
    """

    def __init__(
        self,
        model_key: str = "zai-org/GLM-Image",
        device: str = "cuda",
        organelle_rgb: tuple[int, int, int] = (255, 0, 0),
    ):
        import torch
        from diffusers.pipelines.glm_image import GlmImagePipeline

        self.organelle_rgb = organelle_rgb
        self.pipe = GlmImagePipeline.from_pretrained(
            model_key,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        iteration: int = 0,
        instance: bool = False,
        mask_mode: str = "overlay",
    ) -> GenerationResult:
        # GLM-Image requires dimensions divisible by 32
        target_size = 1024
        w, h = image.size
        aspect = w / h
        if aspect >= 1:
            gen_w = target_size
            gen_h = max(32, round(target_size / aspect / 32) * 32)
        else:
            gen_h = target_size
            gen_w = max(32, round(target_size * aspect / 32) * 32)

        input_resized = image.resize((gen_w, gen_h), Image.LANCZOS)

        result = (
            self.pipe(
                prompt=params.prompt,
                image=[input_resized],
                height=gen_h,
                width=gen_w,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
            )
            .images[0]
            .convert("RGB")
        )

        mask, mask_image = self._extract_mask(
            input_resized,
            result,
            self.organelle_rgb,
            params.threshold,
            instance,
            mask_mode,
        )

        return GenerationResult(
            input_image=input_resized,
            colored_image=result,
            mask=mask,
            mask_image=mask_image,
            params_used=params,
            iteration=iteration,
        )


class QwenImageEditBackend(ImageGenBackend):
    """Qwen image editing model via diffusers (QwenImageEditPlusPipeline)."""

    def __init__(
        self,
        model: str = "Qwen/Qwen-Image-Edit-2511",
        device: str = "cuda",
        organelle_rgb: tuple[int, int, int] = (255, 0, 0),
    ):
        import torch
        from diffusers import QwenImageEditPlusPipeline

        self.model = model
        self.organelle_rgb = organelle_rgb
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to(device)

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        iteration: int = 0,
        instance: bool = False,
        mask_mode: str = "overlay",
    ) -> GenerationResult:
        import torch

        generator = None
        if params.seed is not None:
            generator = torch.manual_seed(params.seed)

        true_cfg_scale = params.extra.get("true_cfg_scale", 4.0)
        negative_prompt = params.extra.get("negative_prompt", " ")

        output = self.pipe(
            image=[image],
            prompt=params.prompt,
            generator=generator,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            num_images_per_prompt=1,
        )

        generated_image = output.images[0].convert("RGB")
        if generated_image.size != image.size:
            generated_image = generated_image.resize(image.size, Image.LANCZOS)

        mask, mask_image = self._extract_mask(
            image.convert("RGB"),
            generated_image,
            self.organelle_rgb,
            params.threshold,
            instance,
            mask_mode,
        )

        return GenerationResult(
            input_image=image,
            colored_image=generated_image,
            mask=mask,
            mask_image=mask_image,
            params_used=params,
            iteration=iteration,
        )


def create_gen_backend(backend: str, **kwargs) -> ImageGenBackend:
    """Factory function for image generation backends.

    Args:
        backend: Backend name. "flux", "gemini", "glm", "qwen", or "sam3".
        **kwargs: Backend-specific arguments.
    """
    if backend == "flux":
        for key in ("gcp_project", "gcp_location", "vertex_ai"):
            kwargs.pop(key, None)
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
    if backend == "glm":
        for key in ("gcp_project", "gcp_location", "vertex_ai", "lora_path", "api_key"):
            kwargs.pop(key, None)
        # Default to GLM repo if model_key is a flux model name
        model_key = kwargs.get("model_key", "")
        if model_key in ("kontext-dev", "flux2-dev", ""):
            kwargs["model_key"] = "zai-org/GLM-Image"
        return GLMImageBackend(**kwargs)
    if backend == "qwen":
        for key in ("gcp_project", "gcp_location", "vertex_ai", "lora_path", "api_key"):
            kwargs.pop(key, None)
        model_key = kwargs.pop("model_key", "")
        model = "Qwen/Qwen-Image-Edit-2511"
        if model_key not in ("", "kontext-dev", "flux2-dev"):
            model = model_key
        kwargs["model"] = model
        return QwenImageEditBackend(**kwargs)
    if backend == "sam3":
        for key in ("gcp_project", "gcp_location", "vertex_ai", "lora_path", "api_key", "model_key"):
            kwargs.pop(key, None)
        from .sam3_backend import SAM3Backend

        return SAM3Backend(**kwargs)
    raise ValueError(f"Unknown image generation backend: {backend!r}")
