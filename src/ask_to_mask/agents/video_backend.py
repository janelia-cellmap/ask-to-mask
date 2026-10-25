"""Gemini Veo video-generation backend.

Veo is a *generative* video model, not a frame-accurate video editor — it has
no "take this real video and overlay a segmentation on it" mode (see
``google.genai.types.GenerateVideosConfig``: only ``image``/``last_frame``/
``reference_images`` conditioning and ``video`` *extension* exist, none of
which round-trips real per-frame content). This backend asks Veo to imagine
a stylized clip of "segmentation happening" from a text prompt plus one real
reference frame — the output is illustrative, not a scientifically accurate
segmentation of the input crop.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

# Veo only accepts a small fixed set of clip lengths (seconds).
VEO_DURATION_OPTIONS = (4, 6, 8)


def _nearest_duration(requested: float) -> int:
    return min(VEO_DURATION_OPTIONS, key=lambda d: abs(d - requested))


class VeoBackend:
    """Generates a stylized video via Google's Veo models (Vertex AI)."""

    def __init__(
        self,
        model: str = "veo-3.1-fast-generate-001",
        gcp_project: str | None = None,
        gcp_location: str = "global",
        output_gcs_uri: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.gcp_project = gcp_project
        self.gcp_location = gcp_location
        self.output_gcs_uri = output_gcs_uri
        self.api_key = api_key

    def _make_client(self):
        import os

        from google import genai

        if self.api_key:
            return genai.Client(api_key=self.api_key)

        project = self.gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError(
                "Veo (Vertex AI) requires a GCP project. Set GOOGLE_CLOUD_PROJECT "
                "env var or pass --gcp-project."
            )
        return genai.Client(vertexai=True, project=project, location=self.gcp_location)

    def generate(
        self,
        prompt: str,
        reference_image: Image.Image | None = None,
        duration_seconds: float = 5.0,
        aspect_ratio: str = "16:9",
        poll_seconds: float = 15.0,
    ) -> str:
        """Request a Veo video and block until it's ready.

        Args:
            prompt: Text description of the desired stylized clip.
            reference_image: Optional real frame used as first-frame
                conditioning (``image=``) — grounds style/subject, does not
                guarantee per-frame content fidelity.
            duration_seconds: Requested clip length; snapped to Veo's nearest
                supported value (currently 4, 6, or 8 seconds).
            aspect_ratio: e.g. "16:9" or "1:1".
            poll_seconds: Delay between operation-status polls.

        Returns:
            The ``gs://`` URI of the generated video.
        """
        import io

        from google.genai.types import GenerateVideosConfig, Image as GenaiImage

        if not self.output_gcs_uri:
            raise ValueError(
                "Veo video generation requires an output GCS URI "
                "(e.g. gs://my-bucket/veo-out/) — set --veo-output-gcs-uri."
            )

        actual_duration = _nearest_duration(duration_seconds)
        if actual_duration != round(duration_seconds):
            print(
                f"  Note: Veo only supports {VEO_DURATION_OPTIONS}s clips — "
                f"requested {duration_seconds}s, using {actual_duration}s."
            )

        client = self._make_client()

        image_arg = None
        if reference_image is not None:
            buf = io.BytesIO()
            reference_image.convert("RGB").save(buf, format="PNG")
            image_arg = GenaiImage(image_bytes=buf.getvalue(), mime_type="image/png")

        operation = client.models.generate_videos(
            model=self.model,
            prompt=prompt,
            image=image_arg,
            config=GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                duration_seconds=actual_duration,
                output_gcs_uri=self.output_gcs_uri,
            ),
        )

        while not operation.done:
            time.sleep(poll_seconds)
            operation = client.operations.get(operation)
            print(f"  Veo operation status: done={operation.done}")

        if not operation.response:
            raise RuntimeError("Veo video generation finished without a response.")

        return operation.result.generated_videos[0].video.uri

    def download(self, gcs_uri: str, output_path: str | Path) -> Path | None:
        """Best-effort download of a gs:// video URI to a local file.

        Returns the local path on success, or ``None`` if no GCS client is
        available (caller should fall back to instructing the user to
        `gsutil cp` the URI manually).
        """
        output_path = Path(output_path)
        try:
            from google.cloud import storage
        except ImportError:
            print(
                f"  google-cloud-storage not installed — download manually with:\n"
                f"    gsutil cp {gcs_uri} {output_path}"
            )
            return None

        bucket_name, _, blob_name = gcs_uri.removeprefix("gs://").partition("/")
        client = storage.Client(project=self.gcp_project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(output_path))
        print(f"  Downloaded Veo video: {output_path}")
        return output_path


def build_veo_prompt(
    organelle_name: str,
    color_name: str,
    resolution_nm: float | None = None,
    detail: str | None = None,
) -> str:
    """Default text prompt asking Veo for a stylized segmentation-style clip."""
    scale = f" The tissue is imaged at roughly {resolution_nm:.1f} nm/pixel." if resolution_nm else ""
    extra = f" {detail}" if detail else ""
    return (
        f"A short video in the style of an electron-microscopy (EM) scientific "
        f"visualization, scrolling through a stack of grayscale EM tissue slices. "
        f"As the video progresses, {organelle_name} organelles are progressively "
        f"highlighted/segmented in {color_name}, as if an automated segmentation "
        f"algorithm is running live over the volume — a solid {color_name} overlay "
        f"grows to cover each {organelle_name} instance while the surrounding tissue "
        f"stays in grayscale.{scale}{extra} Keep the visual style clinical and "
        f"consistent with real EM data, not photorealistic or cartoonish."
    )
