"""SAM3 segmentation backend with text, VLM-coordinate, and painted-marker strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .gen_backend import ImageGenBackend
from .schemas import GenerationParams, GenerationResult

if TYPE_CHECKING:
    from .llm_backend import LLMBackend


# Prompt template for asking a generative model to paint location markers
MARKER_PROMPT_TEMPLATE = (
    "Place a small bright {color_name} dot (about 5-10 pixels) at the center of "
    "each {organelle_name} in the image. Do not color the entire organelle — "
    "only place a single small dot marker at each one's center."
)


class SAM3Backend(ImageGenBackend):
    """Segment Anything Model 3 backend.

    Uses SAM3 for precise segmentation with three prompting strategies:
    - "text": SAM3's open-vocabulary text prompts
    - "vlm-coordinate": VLM provides point coordinates, fed to SAM3
    - "painted-marker": generative model paints dots, detected and fed to SAM3
    """

    def __init__(
        self,
        strategy: str = "text",
        model_name: str = "facebook/sam3",
        device: str = "cuda",
        organelle_rgb: tuple[int, int, int] = (255, 0, 0),
        confidence_threshold: float = 0.5,
        marker_gen_backend: ImageGenBackend | None = None,
        llm_backend: LLMBackend | None = None,
        **_kwargs,
    ):
        self.strategy = strategy
        self.device = device
        self.organelle_rgb = organelle_rgb
        self.confidence_threshold = confidence_threshold
        self.marker_gen_backend = marker_gen_backend
        self.llm_backend = llm_backend

        if strategy == "painted-marker" and marker_gen_backend is None:
            raise ValueError(
                "painted-marker strategy requires a marker_gen_backend"
            )
        if strategy == "vlm-coordinate" and llm_backend is None:
            raise ValueError(
                "vlm-coordinate strategy requires an llm_backend"
            )

        # Lazy-load SAM3 model
        self._model = None
        self._processor = None
        self._point_predictor = None
        self._model_name = model_name

    def _load_model(self):
        """Load SAM3 model on first use."""
        if self._model is not None:
            return

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        print(f"  Loading SAM3 model: {self._model_name}")
        need_points = self.strategy in ("vlm-coordinate", "painted-marker")
        kwargs = {"device": self.device, "enable_inst_interactivity": need_points}
        # If a custom checkpoint path is provided (not the default HF repo ID),
        # pass it as checkpoint_path and disable HF download
        if self._model_name != "facebook/sam3":
            kwargs["checkpoint_path"] = self._model_name
            kwargs["load_from_HF"] = False
        self._model = build_sam3_image_model(**kwargs)
        self._processor = Sam3Processor(
            self._model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )
        if need_points:
            self._point_predictor = self._model.inst_interactive_predictor
            # The predictor's internal model needs the SAM3 backbone assigned
            self._point_predictor.model.backbone = self._model.backbone

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        iteration: int = 0,
        instance: bool = False,
        mask_mode: str = "overlay",
    ) -> GenerationResult:
        self._load_model()

        if self.strategy == "text":
            mask = self._generate_text(image, params)
        elif self.strategy == "vlm-coordinate":
            mask = self._generate_vlm_coordinate(image, params)
        elif self.strategy == "painted-marker":
            mask = self._generate_painted_marker(image, params, iteration, instance, mask_mode)
        else:
            raise ValueError(f"Unknown SAM3 strategy: {self.strategy!r}")

        # vlm-coordinate already returns instance-labeled uint16 mask
        if mask.dtype == np.uint16:
            mask_image = self._labeled_mask_to_image(mask)
        elif instance:
            from scipy import ndimage
            labeled, num_features = ndimage.label(mask > 0)
            mask = labeled.astype(np.uint16)
            mask_image = self._labeled_mask_to_image(mask)
        else:
            mask_image = Image.fromarray(mask, mode="L")

        colored_image = self._mask_to_colored_image(image, mask)

        return GenerationResult(
            input_image=image,
            colored_image=colored_image,
            mask=mask,
            mask_image=mask_image,
            params_used=params,
            iteration=iteration,
        )

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        """Ensure image is RGB (SAM3 requires 3-channel input)."""
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def _generate_text(
        self,
        image: Image.Image,
        params: GenerationParams,
    ) -> np.ndarray:
        """Use SAM3's open-vocabulary text prompt."""
        state = self._processor.set_image(self._to_rgb(image))
        state = self._processor.set_text_prompt(state=state, prompt=params.prompt)

        masks, scores = self._extract_masks_scores(state)

        return self._select_and_union_masks(
            masks, scores, params.extra.get("sam3_confidence_threshold", self.confidence_threshold)
        )

    def _generate_vlm_coordinate(
        self,
        image: Image.Image,
        params: GenerationParams,
    ) -> np.ndarray:
        """Use point coordinates from VLM (stored in params.extra['points'])."""
        points = params.extra.get("points", [])
        if not points:
            raise ValueError(
                "vlm-coordinate strategy requires points in params.extra['points']. "
                "The refinement loop should populate these via evaluator.generate_initial_points()."
            )

        return self._predict_with_points(image, points)

    def _generate_painted_marker(
        self,
        image: Image.Image,
        params: GenerationParams,
        iteration: int,
        instance: bool,
        mask_mode: str,
    ) -> np.ndarray:
        """Use a generative model to paint markers, detect them, then feed to SAM3."""
        from .marker_detection import detect_colored_markers

        # Build a marker-painting prompt
        organelle_name = params.extra.get("organelle_name", "organelle")
        color_name = params.extra.get("color_name", "red")
        marker_prompt = params.extra.get(
            "marker_prompt",
            MARKER_PROMPT_TEMPLATE.format(
                color_name=color_name,
                organelle_name=organelle_name,
            ),
        )

        # Use the secondary gen backend to paint markers
        marker_params = GenerationParams(
            prompt=marker_prompt,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            strength=params.strength,
            seed=params.seed,
            threshold=params.threshold,
            extra=params.extra,
        )
        marker_result = self.marker_gen_backend.generate(
            image, marker_params, iteration, instance=False, mask_mode=mask_mode
        )

        # Detect marker centroids
        marker_rgb = self.organelle_rgb
        points = detect_colored_markers(
            image, marker_result.colored_image, marker_rgb
        )

        if not points:
            print("  Warning: no markers detected, falling back to text prompt")
            return self._generate_text(image, params)

        print(f"  Detected {len(points)} markers")

        return self._predict_with_points(image, points)

    def _predict_with_points(
        self,
        image: Image.Image,
        points: list[dict],
    ) -> np.ndarray:
        """Run SAM3 point-prompt segmentation using the inst_interactive_predictor.

        Args:
            image: Input image.
            points: List of dicts with 'x', 'y', and optional 'label' (1=fg, 0=bg).

        Returns:
            Binary mask as uint8 numpy array (H, W), 0 or 255.
        """
        from collections import defaultdict

        rgb = self._to_rgb(image)
        predictor = self._point_predictor
        predictor.set_image(np.array(rgb))

        # SAM's point predictor is single-instance: all points in one predict()
        # call segment ONE object. Group foreground points by instance ID so
        # multiple points can refine the same instance. Background points are
        # included in every call.
        fg_points = [p for p in points if p.get("label", 1) == 1]
        bg_points = [p for p in points if p.get("label", 1) == 0]
        bg_coords = np.array([[p["x"], p["y"]] for p in bg_points], dtype=np.float32) if bg_points else None
        bg_labels = np.zeros(len(bg_points), dtype=np.int32) if bg_points else None

        # Group foreground points by instance ID
        instances: dict[int, list[dict]] = defaultdict(list)
        for i, fp in enumerate(fg_points):
            inst_id = fp.get("instance", i)
            instances[inst_id].append(fp)

        h, w = np.array(rgb).shape[:2]
        # Use uint16 labeled mask: each instance gets a unique label (1, 2, 3, ...)
        labeled_mask = np.zeros((h, w), dtype=np.uint16)
        label_counter = 0

        for inst_id, inst_points in instances.items():
            coords = np.array([[p["x"], p["y"]] for p in inst_points], dtype=np.float32)
            labels = np.ones(len(inst_points), dtype=np.int32)
            if bg_coords is not None:
                coords = np.concatenate([coords, bg_coords], axis=0)
                labels = np.concatenate([labels, bg_labels], axis=0)

            masks, scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores))
            pts_str = ", ".join(f"({p['x']}, {p['y']})" for p in inst_points)
            print(f"  Instance {inst_id} [{pts_str}]: best score={scores[best_idx]:.3f}")
            label_counter += 1
            instance_mask = masks[best_idx] > 0
            # Only label pixels not already claimed by another instance
            labeled_mask[instance_mask & (labeled_mask == 0)] = label_counter

        print(f"  SAM3 point predictor: {len(instances)} instances, {len(fg_points)} fg points, {len(bg_points)} bg points")
        return labeled_mask

    @staticmethod
    def _extract_masks_scores(state: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract masks and scores from SAM3 state dict as numpy arrays.

        SAM3 returns torch tensors: masks [N, 1, H, W] bool, scores [N] float.
        Convert to numpy: masks [N, H, W] uint8, scores [N] float.
        """
        import torch

        masks_t = state["masks"]  # [N, 1, H, W] bool tensor
        scores_t = state["scores"]  # [N] float tensor

        if isinstance(masks_t, torch.Tensor):
            masks_np = masks_t.squeeze(1).cpu().numpy().astype(np.uint8)  # [N, H, W]
            scores_np = scores_t.cpu().numpy()
        else:
            masks_np = np.array(masks_t)
            scores_np = np.array(scores_t)

        print(f"  SAM3 returned {len(masks_np)} masks, scores: {scores_np}")
        return masks_np, scores_np

    def _select_and_union_masks(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Select masks above confidence threshold and union them into one binary mask."""
        if len(masks) == 0:
            # Return empty mask — let the evaluator handle this
            return np.zeros((1, 1), dtype=np.uint8)

        # Filter by confidence
        good_indices = [i for i, s in enumerate(scores) if s >= threshold]

        if not good_indices:
            # Fall back to the best single mask
            best_idx = int(np.argmax(scores))
            good_indices = [best_idx]
            print(f"  No masks above threshold {threshold:.2f}, using best (score={scores[best_idx]:.3f})")

        # Union selected masks
        union = np.zeros_like(masks[0], dtype=np.uint8)
        for idx in good_indices:
            union = np.maximum(union, (masks[idx] > 0).astype(np.uint8) * 255)

        return union

    # Distinct colors for per-instance visualization
    INSTANCE_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255),
    ]

    def _labeled_mask_to_image(self, mask: np.ndarray) -> Image.Image:
        """Convert a uint16 labeled mask to a visible RGB image with distinct colors."""
        h, w = mask.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        unique_labels = [l for l in np.unique(mask) if l > 0]
        for i, label in enumerate(unique_labels):
            color = self.INSTANCE_COLORS[i % len(self.INSTANCE_COLORS)]
            rgb[mask == label] = color
        return Image.fromarray(rgb)

    def _mask_to_colored_image(
        self,
        em_image: Image.Image,
        mask: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Synthesize a semi-transparent colored overlay on the EM image.

        For instance-labeled masks (uint16), each instance gets a distinct color.
        For binary masks, uses the organelle color.
        """
        colored = np.array(em_image.convert("RGB")).copy().astype(np.float32)

        # Resize mask to match image if needed
        if mask.shape[:2] != colored.shape[:2]:
            from PIL import Image as PILImage
            mask_img = PILImage.fromarray(mask)
            mask_img = mask_img.resize(
                (colored.shape[1], colored.shape[0]), PILImage.NEAREST
            )
            mask = np.array(mask_img)

        # Instance-labeled mask (uint16 with labels 1, 2, 3, ...)
        if mask.dtype in (np.uint16, np.int32, np.int64) or mask.max() > 255:
            unique_labels = [l for l in np.unique(mask) if l > 0]
            for i, label in enumerate(unique_labels):
                color = np.array(self.INSTANCE_COLORS[i % len(self.INSTANCE_COLORS)], dtype=np.float32)
                region = mask == label
                colored[region] = colored[region] * (1 - alpha) + color * alpha
        else:
            # Binary mask
            fg = mask > 0 if mask.ndim == 2 else mask.any(axis=-1)
            color = np.array(self.organelle_rgb, dtype=np.float32)
            colored[fg] = colored[fg] * (1 - alpha) + color * alpha

        return Image.fromarray(colored.astype(np.uint8))
