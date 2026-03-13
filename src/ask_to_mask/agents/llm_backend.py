"""Pluggable LLM/VLM backends for the evaluator agent."""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import time
from abc import ABC, abstractmethod

from PIL import Image, ImageDraw, ImageFont


class LLMBackend(ABC):
    """Abstract base for LLM/VLM providers.

    Each backend sends a system prompt + user message (with an image)
    and returns the model's text response.
    """

    @abstractmethod
    def chat_with_images(
        self, system_prompt: str, user_prompt: str, images: list[Image.Image]
    ) -> str:
        """Send a message with one or more images and return the text response."""
        ...

    def chat_with_image(
        self, system_prompt: str, user_prompt: str, image: Image.Image
    ) -> str:
        """Convenience wrapper for a single image."""
        return self.chat_with_images(system_prompt, user_prompt, [image])


class OllamaBackend(LLMBackend):
    """Local VLM via ollama. Auto-starts the server if not running."""

    def __init__(
        self,
        model: str = "gemma3:27b",
        host: str = "http://localhost:11434",
        temperature: float = 0.3,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        # Molmo outputs XML-like point tags, not JSON — don't force JSON format
        self.force_json = "molmo" not in model.lower()
        self._server_process = None
        self._ensure_server()

    def _ensure_server(self) -> None:
        """Start ollama server in background if it's not already running."""
        import ollama

        client = ollama.Client(host=self.host)
        try:
            client.list()
            return  # Server is already running
        except Exception:
            pass

        # Server not running — try to start it
        ollama_bin = shutil.which("ollama")
        if ollama_bin is None:
            raise RuntimeError(
                "ollama binary not found on PATH. "
                "Install from https://ollama.com/download"
            )

        print(f"Starting ollama server in background...")
        self._server_process = subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for server to be ready
        for _ in range(30):
            time.sleep(1)
            try:
                client.list()
                print("ollama server started.")
                return
            except Exception:
                continue

        raise RuntimeError(
            "ollama server failed to start within 30 seconds. "
            "Try running 'ollama serve' manually."
        )

    @staticmethod
    def _resize_for_vlm(img: Image.Image, max_dim: int = 1008) -> Image.Image:
        """Resize image so dimensions are divisible by 28 and within max_dim.

        qwen2.5vl's vision encoder uses 28px patches and crashes on
        incompatible dimensions (GGML_ASSERT failure in RoPE).
        """
        patch = 28
        w, h = img.size
        # Scale down if needed
        scale = min(max_dim / w, max_dim / h, 1.0)
        w, h = int(w * scale), int(h * scale)
        # Round to nearest multiple of patch size
        w = max(patch, (w // patch) * patch)
        h = max(patch, (h // patch) * patch)
        if (w, h) != img.size:
            img = img.resize((w, h), Image.LANCZOS)
        return img

    def chat_with_images(
        self, system_prompt: str, user_prompt: str, images: list[Image.Image]
    ) -> str:
        import ollama

        image_bytes_list = []
        for img in images:
            img = self._resize_for_vlm(img)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes_list.append(buf.getvalue())

        client = ollama.Client(host=self.host)
        chat_kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": image_bytes_list},
            ],
            "options": {"temperature": self.temperature, "num_predict": 1024},
        }
        if self.force_json:
            chat_kwargs["format"] = "json"
        response = client.chat(**chat_kwargs)
        return response["message"]["content"]


class AnthropicBackend(LLMBackend):
    """Anthropic API (Claude)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.3,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def chat_with_images(
        self, system_prompt: str, user_prompt: str, images: list[Image.Image]
    ) -> str:
        import anthropic

        content = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_b64 = base64.standard_b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            })
        content.append({"type": "text", "text": user_prompt})

        client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
        )
        return response.content[0].text


class GoogleBackend(LLMBackend):
    """Google Gemini API (API key or Vertex AI)."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        gcp_project: str | None = None,
        gcp_location: str = "us-central1",
        vertex_ai: bool = False,
        temperature: float = 0.3,
    ):
        self.model = model
        self.api_key = api_key
        self.gcp_project = gcp_project
        self.gcp_location = gcp_location
        self.vertex_ai = vertex_ai
        self.temperature = temperature

    def chat_with_images(
        self, system_prompt: str, user_prompt: str, images: list[Image.Image]
    ) -> str:
        import os

        from google import genai
        from google.genai.errors import ClientError

        if self.vertex_ai:
            project = self.gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            client = genai.Client(
                vertexai=True,
                project=project,
                location=self.gcp_location,
            )
        elif self.api_key:
            client = genai.Client(api_key=self.api_key)
        else:
            client = genai.Client()

        contents: list = list(images) + [f"{system_prompt}\n\n{user_prompt}"]
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(
                        temperature=self.temperature
                    ),
                )
                return response.text
            except ClientError as e:
                if "429" in str(e) and attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"  Rate limited — waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise


class OpenAIBackend(LLMBackend):
    """OpenAI API (GPT-4o, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.3,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def chat_with_images(
        self, system_prompt: str, user_prompt: str, images: list[Image.Image]
    ) -> str:
        from openai import OpenAI

        content = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_b64 = base64.standard_b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                },
            })
        content.append({"type": "text", "text": user_prompt})

        client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content


class HuggingFaceBackend(LLMBackend):
    """Local VLM via HuggingFace transformers. Loads model on first use."""

    def __init__(
        self,
        model: str = "allenai/Molmo2-8B",
        temperature: float = 0.3,
        device: str | None = None,
    ):
        self.model_name = model
        self.temperature = temperature
        self.device = device or "cuda"
        self._processor = None
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        print(f"Loading HuggingFace model {self.model_name}...")
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True, padding_side="left"
            )
        except TypeError as e:
            if "Unexpected keyword argument" in str(e):
                # Molmo2's processor passes optional attrs to super().__init__()
                # which some transformers versions reject. Patch and retry.
                from transformers.processing_utils import ProcessorMixin
                orig_init = ProcessorMixin.__init__
                def _patched_init(self_proc, *args, **kwargs):
                    known = set(type(self_proc).get_attributes())
                    known.add("chat_template")
                    extra = {k: kwargs.pop(k) for k in list(kwargs) if k not in known}
                    orig_init(self_proc, *args, **kwargs)
                    for k, v in extra.items():
                        setattr(self_proc, k, v)
                ProcessorMixin.__init__ = _patched_init
                try:
                    self._processor = AutoProcessor.from_pretrained(
                        self.model_name, trust_remote_code=True
                    )
                finally:
                    ProcessorMixin.__init__ = orig_init
            else:
                raise

        # Try model classes in order: ImageTextToText (Molmo2), CausalLM, AutoModel
        model_kwargs = dict(
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
        )
        import transformers
        errors = []
        for auto_cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
            try:
                auto_cls = getattr(transformers, auto_cls_name)
                self._model = auto_cls.from_pretrained(self.model_name, **model_kwargs)
                print(f"Model loaded via {auto_cls_name} on {self.device}.")
                return
            except Exception as e:
                errors.append(f"{auto_cls_name}: {type(e).__name__}: {e}")
                continue
        raise RuntimeError(
            f"Could not load {self.model_name} with any AutoModel class:\n"
            + "\n".join(errors)
        )

    def chat_with_images(
        self, system_prompt: str, user_prompt: str, images: list[Image.Image]
    ) -> str:
        import torch
        self._load_model()

        prompt = f"{system_prompt}\n\n{user_prompt}"

        # Build chat messages for apply_chat_template
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image", "image": img})
        messages = [{"role": "user", "content": content}]

        try:
            # Molmo2 / newer HF VLMs use apply_chat_template
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except Exception:
            # Fallback for older Molmo / other HF VLMs
            inputs = self._processor.process(images=images, text=prompt)
            inputs = {
                k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else v
                for k, v in inputs.items()
            }

        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=2048)

        generated = output[0, inputs["input_ids"].size(1):]
        text = self._processor.tokenizer.decode(generated, skip_special_tokens=True)
        return text


def create_llm_backend(provider: str, **kwargs) -> LLMBackend:
    """Factory function for LLM/VLM backends.

    Args:
        provider: One of "ollama", "anthropic", "google", "openai", "huggingface".
        **kwargs: Provider-specific arguments (model, api_key, host, etc.).
    """
    backends = {
        "ollama": OllamaBackend,
        "anthropic": AnthropicBackend,
        "google": GoogleBackend,
        "openai": OpenAIBackend,
        "huggingface": HuggingFaceBackend,
    }
    if provider not in backends:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Available: {list(backends.keys())}"
        )
    # Strip kwargs not accepted by non-Google backends
    if provider != "google":
        for key in ("gcp_project", "gcp_location", "vertex_ai"):
            kwargs.pop(key, None)
    # Strip kwargs not accepted by HuggingFace backend
    if provider == "huggingface":
        for key in ("host", "api_key"):
            kwargs.pop(key, None)
    return backends[provider](**kwargs)


def images_to_composite(
    *panel_args: Image.Image,
    labels: list[str] | None = None,
    target_height: int = 512,
) -> Image.Image:
    """Create a labeled side-by-side composite of images.

    Call with positional Image args. Default labels are provided for 2 or 3 panels.
    """
    label_height = 20
    images = [img.convert("RGB") for img in panel_args]
    if labels is None:
        if len(images) == 2:
            labels = ["Original EM", "Colored Output"]
        else:
            labels = ["Original EM", "Colored Output", "Extracted Mask"]

    # Resize all to same height
    resized = []
    for img in images:
        ratio = target_height / img.height
        new_w = round(img.width * ratio)
        resized.append(img.resize((new_w, target_height), Image.LANCZOS))

    total_w = sum(img.width for img in resized)
    total_h = target_height + label_height
    composite = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(composite)

    x = 0
    for img, lbl in zip(resized, labels):
        # Draw label
        text_x = x + img.width // 2
        draw.text((text_x, 2), lbl, fill=(0, 0, 0), anchor="mt")
        # Paste image below label
        composite.paste(img, (x, label_height))
        x += img.width

    return composite
