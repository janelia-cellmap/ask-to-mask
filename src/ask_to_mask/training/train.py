"""LoRA finetuning for Flux models on CellMap EM data."""

from __future__ import annotations

import logging
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default LoRA target modules for Flux Kontext / Flux1 transformer
DEFAULT_TARGET_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
    "proj_mlp",
    "proj_out",
]

# Default LoRA target modules for Flux2 transformer
DEFAULT_TARGET_MODULES_FLUX2 = [
    # Double stream attention
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    # Double stream feedforward
    "ff.linear_in",
    "ff.linear_out",
    "ff_context.linear_in",
    "ff_context.linear_out",
    # Single stream (fused QKV + MLP projection)
    "attn.to_qkv_mlp_proj",
    # Output
    "proj_out",
]


def load_config(config_path: str) -> dict:
    """Load YAML training config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def collate_fn(batch):
    """Collate training samples, optionally with a per-pixel valid-loss mask."""
    if len(batch[0]) == 5:
        cond_images, target_images, prompts, valid_masks, sample_weights = zip(*batch)
        return (
            list(cond_images),
            list(target_images),
            list(prompts),
            list(valid_masks),
            list(sample_weights),
        )
    if len(batch[0]) == 4:
        cond_images, target_images, prompts, valid_masks = zip(*batch)
        return list(cond_images), list(target_images), list(prompts), list(valid_masks), None
    cond_images, target_images, prompts = zip(*batch)
    return list(cond_images), list(target_images), list(prompts), None, None


class WeightedDatasetAdapter(Dataset):
    """Adapt a training dataset to return masks, sample weights, and fixed prompts."""

    def __init__(
        self,
        dataset: Dataset,
        label_weight: float,
        prompt: str | None = None,
    ):
        self.dataset = dataset
        self.label_weight = float(label_weight)
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        cond, target, prompt = sample[:3]
        prompt = self.prompt or prompt
        if len(sample) >= 4 and sample[3] is not None:
            valid_mask = sample[3]
        else:
            valid_mask = Image.new("L", target.size, 255)
        weight = sample[4] if len(sample) >= 5 and sample[4] is not None else self.label_weight
        return cond, target, prompt, valid_mask, float(weight)


class WeightedMixedDataset(Dataset):
    """Sample from pseudo and GT datasets with configurable mixture probability."""

    def __init__(
        self,
        pseudo_dataset: Dataset,
        gt_dataset: Dataset,
        samples_per_epoch: int,
        gt_sample_prob: float = 0.25,
        seed: int = 42,
    ):
        self.pseudo_dataset = pseudo_dataset
        self.gt_dataset = gt_dataset
        self.samples_per_epoch = int(samples_per_epoch)
        self.gt_sample_prob = float(gt_sample_prob)
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        if torch.rand((), generator=self.rng).item() < self.gt_sample_prob:
            gt_idx = int(torch.randint(len(self.gt_dataset), (), generator=self.rng).item())
            return self.gt_dataset[gt_idx]
        pseudo_idx = int(torch.randint(len(self.pseudo_dataset), (), generator=self.rng).item())
        return self.pseudo_dataset[pseudo_idx]


def pil_to_tensor(images: list[Image.Image], device, dtype) -> torch.Tensor:
    """Convert list of PIL images to normalized tensor [B, C, H, W] in [-1, 1]."""
    import numpy as np

    tensors = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # [0,1] -> [-1,1]
        t = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]
        tensors.append(t)
    return torch.stack(tensors).to(device=device, dtype=dtype)


def pil_masks_to_tensor(images: list[Image.Image], device, dtype) -> torch.Tensor:
    """Convert PIL masks to [B, 1, H, W] tensors in [0, 1]."""
    import numpy as np

    tensors = []
    for img in images:
        arr = np.array(img.convert("L")).astype(np.float32) / 255.0
        t = torch.from_numpy(arr)[None, ...]
        tensors.append(t)
    return torch.stack(tensors).to(device=device, dtype=dtype)


def pack_loss_mask(
    masks: torch.Tensor,
    packed_h: int,
    packed_w: int,
) -> torch.Tensor:
    """Downsample image-space masks to packed latent token weights [B, N, 1]."""
    masks = F.interpolate(masks, size=(packed_h, packed_w), mode="area")
    return masks.flatten(2).transpose(1, 2)


def encode_images(vae, images_tensor: torch.Tensor, is_flux2: bool = False) -> torch.Tensor:
    """Encode images to latents using VAE.

    For Flux1/Kontext: applies shift_factor + scaling_factor normalization.
    For Flux2: applies patchify + batch norm normalization.
    """
    with torch.no_grad():
        latents = vae.encode(images_tensor).latent_dist.sample()
    if is_flux2:
        from diffusers import Flux2Pipeline

        latents = Flux2Pipeline._patchify_latents(latents)
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = (latents - bn_mean) / bn_std
    else:
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents


def pack_latents(latents: torch.Tensor, is_flux2: bool = False) -> torch.Tensor:
    """Pack latents into sequence format.

    Flux1/Kontext: [B, C, H, W] -> [B, H//2 * W//2, C*4] (patchify + flatten).
    Flux2: [B, C*4, H/2, W/2] -> [B, H/2 * W/2, C*4] (already patchified).
    """
    if is_flux2:
        b, c, h, w = latents.shape
        return latents.reshape(b, c, h * w).permute(0, 2, 1)
    else:
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        return latents


def prepare_latent_image_ids(
    height: int, width: int, device, dtype, is_flux2: bool = False,
    batch_size: int = 1,
) -> torch.Tensor:
    """Create position IDs for packed latents.

    Flux1/Kontext: [H*W, 3] with (type_id, y, x).
    Flux2: [B, H*W, 4] with (T, H, W, L).
    """
    if is_flux2:
        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)
        ids = torch.cartesian_prod(t, h, w, l)
        ids = ids.unsqueeze(0).expand(batch_size, -1, -1)
        return ids.to(device=device, dtype=dtype)
    else:
        ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        ids[..., 1] = torch.arange(height, device=device)[:, None]
        ids[..., 2] = torch.arange(width, device=device)[None, :]
        ids = ids.reshape(height * width, 3)
        return ids


def prepare_flux2_cond_ids(
    height: int, width: int, device, dtype, batch_size: int = 1, scale: int = 10,
) -> torch.Tensor:
    """Create conditioning image IDs for Flux2 concatenate mode.

    Uses T = scale (offset from target T=0) to distinguish conditioning tokens.
    Returns [B, H*W, 4] with (T, H, W, L).
    """
    t = torch.tensor([scale])
    h = torch.arange(height)
    w = torch.arange(width)
    l = torch.arange(1)
    ids = torch.cartesian_prod(t, h, w, l)
    ids = ids.unsqueeze(0).expand(batch_size, -1, -1)
    return ids.to(device=device, dtype=dtype)


def encode_prompt(pipe, prompts: list[str], device, dtype, is_flux2: bool = False):
    """Encode text prompts using the pipeline's text encoders.

    Returns (prompt_embeds, pooled_prompt_embeds, text_ids).
    pooled_prompt_embeds is None for Flux2.
    """
    if is_flux2:
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        return prompt_embeds, None, text_ids
    else:
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        return prompt_embeds, pooled_prompt_embeds, text_ids


def pil_to_tb_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a [C, H, W] float tensor in [0, 1]."""
    import numpy as np

    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]


def tensor_grid_to_pil(grid: torch.Tensor) -> Image.Image:
    """Convert a [C, H, W] float image grid in [0, 1] to PIL."""
    import numpy as np

    grid = grid.detach().cpu().clamp(0, 1)
    arr = (grid.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _tracker_names(report_to) -> set[str]:
    if report_to is None:
        return set()
    if isinstance(report_to, str):
        return {report_to.lower()}
    return {str(item).lower() for item in report_to}


def flatten_tracker_config(config: dict, prefix: str = "") -> dict:
    """Flatten YAML config to scalar values accepted by TensorBoard hparams."""
    flat = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_tracker_config(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = "" if value is None else value
        elif isinstance(value, (list, tuple)):
            flat[name] = ",".join(str(item) for item in value)
        else:
            flat[name] = str(value)
    return flat


def log_image_grid(accelerator, report_to, tag: str, grid: torch.Tensor, step: int) -> None:
    """Log an image grid to the configured Accelerate tracker backend."""
    trackers = _tracker_names(report_to)
    if "tensorboard" in trackers or "all" in trackers:
        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
        if tracker is not None:
            tracker.add_image(tag, grid, step)
            tracker.flush()
    if "wandb" in trackers or "all" in trackers:
        try:
            import wandb
        except ImportError:
            logger.warning("W&B logging requested but wandb is not installed")
            return
        accelerator.log({tag: wandb.Image(tensor_grid_to_pil(grid))}, step=step)


def composite_overlay(cond_pil: Image.Image, pred_pil: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Overlay prediction on EM input for visual comparison.

    Colored (non-gray) pixels from pred are blended onto the EM image.
    """
    import numpy as np

    cond = np.array(cond_pil).astype(np.float32)
    pred = np.array(pred_pil).astype(np.float32)

    # Detect colored pixels: where channels differ significantly
    max_ch = pred.max(axis=-1)
    min_ch = pred.min(axis=-1)
    colored = (max_ch - min_ch) > 30  # saturation threshold

    out = cond.copy()
    out[colored] = (1 - alpha) * cond[colored] + alpha * pred[colored]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def _build_image_grid(
    cond_pils: list[Image.Image],
    target_pils: list[Image.Image],
    pred_pils: list[Image.Image] | None = None,
) -> torch.Tensor:
    """Build a single image grid with rows: EM input, GT overlay, prediction overlay.

    For overlay mode targets already contain the EM background.
    For segmentation mode targets are on black — we composite them onto the EM.
    Predictions are always composited onto the EM for easy visual comparison.

    Returns a [C, H, W] tensor suitable for tensorboard.
    """
    from torchvision.utils import make_grid

    n = len(cond_pils)
    rows: list[torch.Tensor] = []

    # Row 1: raw EM input
    for img in cond_pils:
        rows.append(pil_to_tb_tensor(img))

    # Row 2: ground truth overlaid on EM
    for cond, target in zip(cond_pils, target_pils):
        overlay = composite_overlay(cond, target, alpha=0.7)
        rows.append(pil_to_tb_tensor(overlay))

    # Row 3: prediction overlaid on EM (if provided)
    if pred_pils is not None:
        for cond, pred in zip(cond_pils, pred_pils):
            overlay = composite_overlay(cond, pred, alpha=0.7)
            rows.append(pil_to_tb_tensor(overlay))

    # nrow=n means each "row" of n images stacks vertically
    return make_grid(rows, nrow=n, padding=4)


def run_validation(
    pipe,
    transformer,
    dataset,
    accelerator,
    global_step: int,
    report_to,
    tag: str = "val/samples",
    target_mode: str = "overlay",
    num_images: int = 4,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
):
    """Run inference on dataset samples and log a single image grid.

    Grid rows: EM input | GT overlay | prediction overlay.
    """
    from torchvision.utils import make_grid

    logger.info(f"Running validation at step {global_step}...")

    unwrapped = accelerator.unwrap_model(transformer)
    unwrapped.eval()

    # Temporarily set unwrapped transformer for inference
    pipe.transformer = unwrapped
    pipe.to(accelerator.device, dtype=torch.bfloat16)
    # Ensure VAE is explicitly in bf16 (accelerate may have moved it)
    pipe.vae.to(dtype=torch.bfloat16)

    cond_pils = []
    gt_pils = []
    pred_pils = []

    generator = torch.Generator(device=accelerator.device).manual_seed(42)

    for i in range(num_images):
        sample = dataset[i]
        if sample is None:
            continue
        cond_pil, target_pil, prompt = sample[:3]

        # Run inference
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = pipe(
                image=cond_pil,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        cond_pils.append(cond_pil)
        gt_pils.append(target_pil)
        pred_pils.append(result)

    if cond_pils:
        grid = _build_image_grid(cond_pils, gt_pils, pred_pils)
        log_image_grid(accelerator, report_to, tag, grid, global_step)

    unwrapped.train()
    logger.info(f"Validation complete: logged {len(cond_pils)} samples")


def compute_flow_matching_loss(
    model_pred: torch.Tensor,
    noise: torch.Tensor,
    target_latents: torch.Tensor,
    sigmas: torch.Tensor,
    weighting_scheme: str = "sigma_sqrt",
    loss_mask: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute flow matching loss.

    In flow matching, the target velocity is: v = noise - clean_latents
    The model predicts this velocity.
    """
    # Flow matching target: velocity = noise - clean
    target = noise - target_latents

    squared_error = (model_pred - target).pow(2)
    if loss_mask is not None:
        token_loss = squared_error.mean(dim=-1, keepdim=True)
        denom = loss_mask.sum(dim=list(range(1, loss_mask.ndim))).clamp_min(1.0)
        loss = (token_loss * loss_mask).sum(dim=list(range(1, token_loss.ndim))) / denom
    else:
        # Per-sample loss
        loss = squared_error.mean(dim=list(range(1, model_pred.ndim)))

    # Weighting
    if weighting_scheme == "sigma_sqrt":
        weights = sigmas
    elif weighting_scheme == "none":
        weights = torch.ones_like(sigmas)
    else:
        weights = torch.ones_like(sigmas)

    loss = weights * loss
    if sample_weights is not None:
        loss = loss * sample_weights
    loss = loss.mean()
    return loss


def train(config_path: str, resume_from: str | None = None):
    """Main training function."""
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from peft import LoraConfig, get_peft_model_state_dict

    config = load_config(config_path)

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    log_cfg = config.get("logging", {})
    report_to = log_cfg.get("report_to", "tensorboard")

    # Create datetime-stamped run directory for all outputs
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(train_cfg.get("output_dir", "runs/flux-lora")) / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
        log_with=report_to,
        project_dir=str(output_dir),
    )

    if train_cfg.get("seed") is not None:
        set_seed(train_cfg["seed"])

    # Save config and set up file logging in the run directory
    shutil.copy2(config_path, output_dir / "train_config.yaml")

    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Load model components
    pretrained = model_cfg["pretrained"]
    logger.info(f"Loading model: {pretrained}")

    # Determine pipeline class
    from ..config import MODELS

    # Reverse lookup model key from pretrained name
    model_key = None
    for k, v in MODELS.items():
        if v == pretrained:
            model_key = k
            break

    if "Kontext" in pretrained:
        from diffusers import FluxKontextPipeline as PipelineClass
    elif "FLUX.2" in pretrained:
        from diffusers import Flux2Pipeline as PipelineClass
    else:
        from diffusers import FluxImg2ImgPipeline as PipelineClass

    pipe = PipelineClass.from_pretrained(pretrained, torch_dtype=torch.bfloat16)

    transformer = pipe.transformer
    vae = pipe.vae
    vae_scale_factor = pipe.vae_scale_factor

    # Freeze everything
    vae.requires_grad_(False)
    if pipe.text_encoder is not None:
        pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(False)

    # Add LoRA to transformer
    lora_cfg = model_cfg.get("lora", {})
    lora_rank = lora_cfg.get("rank", 16)
    lora_alpha = lora_cfg.get("alpha", 16)
    lora_dropout = lora_cfg.get("dropout", 0.0)
    default_modules = DEFAULT_TARGET_MODULES_FLUX2 if "FLUX.2" in pretrained else DEFAULT_TARGET_MODULES
    target_modules = lora_cfg.get("target_modules") or default_modules

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights=True,
        target_modules=target_modules,
    )
    transformer.add_adapter(lora_config)

    if train_cfg.get("gradient_checkpointing", True):
        transformer.enable_gradient_checkpointing()

    # Move frozen components to device
    device = accelerator.device
    dtype = torch.bfloat16
    if torch.cuda.is_available() and accelerator.is_main_process:
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "CUDA device: %s, capability=%s.%s, total_memory=%.1f GB",
            props.name,
            props.major,
            props.minor,
            props.total_memory / 1024**3,
        )
    vae.to(device, dtype=dtype)
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)

    # Create dataset
    dataset_type = data_cfg.get("dataset_type", "groundtruth_crops")
    validation_dataset = None
    validation_datasets = []
    if dataset_type in {"inference_mito_fixed_fov", "mixed_mito_gt_pseudo"}:
        from .inference_mito_dataset import InferenceMitoFixedFovDataset

        common_dataset_kwargs = dict(
            data_root=data_cfg.get("data_root", "/nrs/cellmap/data"),
            em_path_template=data_cfg.get(
                "em_path_template",
                "/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8",
            ),
            segmentation_path_template=data_cfg.get(
                "segmentation_path_template",
                "/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/labels/inference/segmentations",
            ),
            label_name=data_cfg.get("label_name", "mito"),
            index_path=data_cfg.get("index_path"),
            rebuild_index=data_cfg.get("rebuild_index", False),
            include_datasets=data_cfg.get("include_datasets"),
            skip_datasets=data_cfg.get("skip_datasets"),
            fov_nm=data_cfg.get("fov_nm", 1600.0),
            fov_tolerance_nm=data_cfg.get("fov_tolerance_nm", 0.0),
            raw_target_resolution_nm=data_cfg.get("raw_target_resolution_nm"),
            label_target_resolution_nm=data_cfg.get("label_target_resolution_nm"),
            require_exact_resolution=data_cfg.get("require_exact_resolution", False),
            raw_require_exact_resolution=data_cfg.get("raw_require_exact_resolution"),
            label_require_exact_resolution=data_cfg.get("label_require_exact_resolution"),
            min_mask_fraction=data_cfg.get("min_mask_fraction", 0.01),
            min_valid_fraction=data_cfg.get("min_valid_fraction", 0.25),
            label_quality=data_cfg.get("label_quality", "pseudo"),
            label_weight=data_cfg.get("label_weight", 0.2),
            conservative_pseudo_targets=data_cfg.get("conservative_pseudo_targets", True),
            foreground_erosion_px=data_cfg.get("foreground_erosion_px", 2),
            boundary_band_px=data_cfg.get("boundary_band_px", 3),
            boundary_weight=data_cfg.get("boundary_weight", 0.0),
            background_weight=data_cfg.get("background_weight", 0.05),
            max_sample_attempts=data_cfg.get("max_sample_attempts", 100),
            prompt=data_cfg.get("prompt", "CLASS=mitochondria; OUTPUT=red_on_black"),
            auto_norms_percentile_low=data_cfg.get("auto_norms_percentile_low", 1.0),
            auto_norms_percentile_high=data_cfg.get("auto_norms_percentile_high", 99.0),
            val_holdout_fraction=data_cfg.get("val_holdout_fraction", 0.1),
            val_holdout_axis=data_cfg.get("val_holdout_axis", "x"),
            val_holdout_position=data_cfg.get("val_holdout_position", "high"),
        )
        pseudo_dataset = InferenceMitoFixedFovDataset(
            **common_dataset_kwargs,
            samples_per_epoch=data_cfg.get("samples_per_epoch", 2000),
            split="train",
            seed=train_cfg.get("seed", 42),
            augment=data_cfg.get("augment", True),
        )
        pseudo_validation_dataset = InferenceMitoFixedFovDataset(
            **common_dataset_kwargs,
            samples_per_epoch=train_cfg.get("num_validation_images", 4),
            split="val",
            seed=train_cfg.get("seed", 42) + 100_000,
            augment=False,
        )
        dataset = pseudo_dataset
        validation_dataset = pseudo_validation_dataset
        validation_datasets = [("val/pseudo_samples", pseudo_validation_dataset)]

        if dataset_type == "mixed_mito_gt_pseudo":
            from .dataset import CellMapFluxDataset, FixedFovCellMapGtDataset

            gt_cfg = data_cfg.get("gt", {})
            gt_val_datasets = gt_cfg.get("validation_datasets", [])
            gt_skip = list(gt_cfg.get("skip_datasets", []))
            gt_skip.extend(gt_val_datasets)
            prompt = data_cfg.get("prompt", "CLASS=mitochondria; OUTPUT=red_on_black")
            use_fixed_fov_gt = gt_cfg.get("fixed_fov", False)

            if use_fixed_fov_gt:
                gt_common_kwargs = dict(
                    data_root=gt_cfg.get("data_root", data_cfg.get("data_root", "/nrs/cellmap/data")),
                    norms_csv=gt_cfg.get("norms_csv", data_cfg.get("norms_csv")),
                    organelle_keys=gt_cfg.get("organelles", ["mito"]),
                    min_mask_fraction=gt_cfg.get("min_mask_fraction", data_cfg.get("min_mask_fraction", 0.01)),
                    cache_dir=gt_cfg.get("cache_dir", data_cfg.get("cache_dir")),
                    augment=gt_cfg.get("augment", data_cfg.get("augment", True)),
                    auto_norms=gt_cfg.get("auto_norms", True),
                    auto_norms_per_image=gt_cfg.get("auto_norms_per_image", False),
                    auto_norms_percentile_low=gt_cfg.get("auto_norms_percentile_low", data_cfg.get("auto_norms_percentile_low", 1.0)),
                    auto_norms_percentile_high=gt_cfg.get("auto_norms_percentile_high", data_cfg.get("auto_norms_percentile_high", 99.0)),
                    fov_nm=gt_cfg.get("fov_nm", data_cfg.get("fov_nm", 8192.0)),
                    target_resolution_nm=gt_cfg.get(
                        "target_resolution_nm",
                        data_cfg.get("raw_target_resolution_nm", 8.0),
                    ),
                    target_size=gt_cfg.get("target_size", 1024),
                    raw_require_exact_resolution=gt_cfg.get("raw_require_exact_resolution", True),
                    label_weight=gt_cfg.get("label_weight", 1.0),
                    background_weight=gt_cfg.get("background_weight", data_cfg.get("background_weight", 0.05)),
                    prompt=prompt,
                    max_sample_attempts=gt_cfg.get("max_sample_attempts", data_cfg.get("max_sample_attempts", 100)),
                )
            else:
                gt_common_kwargs = {}

            if use_fixed_fov_gt:
                gt_train_base = FixedFovCellMapGtDataset(
                    **gt_common_kwargs,
                    samples_per_epoch=gt_cfg.get("samples_per_epoch", data_cfg.get("samples_per_epoch", 2000)),
                    skip_datasets=gt_skip,
                    include_datasets=gt_cfg.get("include_datasets"),
                    seed=train_cfg.get("seed", 42) + 200_000,
                )
            else:
                gt_train_base = CellMapFluxDataset(
                    data_root=gt_cfg.get("data_root", data_cfg.get("data_root", "/nrs/cellmap/data")),
                    norms_csv=gt_cfg.get("norms_csv", data_cfg.get("norms_csv")),
                    organelle_keys=gt_cfg.get("organelles", ["mito"]),
                    samples_per_epoch=gt_cfg.get("samples_per_epoch", data_cfg.get("samples_per_epoch", 2000)),
                    min_mask_fraction=gt_cfg.get("min_mask_fraction", data_cfg.get("min_mask_fraction", 0.01)),
                    skip_datasets=gt_skip,
                    include_datasets=gt_cfg.get("include_datasets"),
                    cache_dir=gt_cfg.get("cache_dir", data_cfg.get("cache_dir")),
                    seed=train_cfg.get("seed", 42) + 200_000,
                    target_mode=gt_cfg.get("target_mode", "segmentation"),
                    include_resolution=gt_cfg.get("include_resolution", False),
                    auto_norms=gt_cfg.get("auto_norms", True),
                    auto_norms_per_image=gt_cfg.get("auto_norms_per_image", False),
                    auto_norms_percentile_low=gt_cfg.get("auto_norms_percentile_low", data_cfg.get("auto_norms_percentile_low", 1.0)),
                    auto_norms_percentile_high=gt_cfg.get("auto_norms_percentile_high", data_cfg.get("auto_norms_percentile_high", 99.0)),
                    multi_organelle_prob=0.0,
                    negative_example_prob=0.0,
                    prompt_variation=False,
                )
            gt_train = WeightedDatasetAdapter(
                gt_train_base,
                label_weight=gt_cfg.get("label_weight", 1.0),
                prompt=prompt,
            )
            pseudo_train = WeightedDatasetAdapter(
                pseudo_dataset,
                label_weight=data_cfg.get("label_weight", 0.2),
                prompt=prompt,
            )
            dataset = WeightedMixedDataset(
                pseudo_dataset=pseudo_train,
                gt_dataset=gt_train,
                samples_per_epoch=data_cfg.get("samples_per_epoch", 2000),
                gt_sample_prob=gt_cfg.get("sample_prob", 0.25),
                seed=train_cfg.get("seed", 42),
            )

            if gt_val_datasets:
                if use_fixed_fov_gt:
                    gt_val_kwargs = dict(gt_common_kwargs)
                    gt_val_kwargs["augment"] = False
                    gt_val_base = FixedFovCellMapGtDataset(
                        **gt_val_kwargs,
                        samples_per_epoch=train_cfg.get("num_validation_images", 4),
                        skip_datasets=gt_cfg.get("skip_datasets", []),
                        include_datasets=gt_val_datasets,
                        seed=train_cfg.get("seed", 42) + 300_000,
                    )
                else:
                    gt_val_base = CellMapFluxDataset(
                        data_root=gt_cfg.get("data_root", data_cfg.get("data_root", "/nrs/cellmap/data")),
                        norms_csv=gt_cfg.get("norms_csv", data_cfg.get("norms_csv")),
                        organelle_keys=gt_cfg.get("organelles", ["mito"]),
                        samples_per_epoch=train_cfg.get("num_validation_images", 4),
                        min_mask_fraction=gt_cfg.get("min_mask_fraction", data_cfg.get("min_mask_fraction", 0.01)),
                        skip_datasets=gt_cfg.get("skip_datasets", []),
                        include_datasets=gt_val_datasets,
                        cache_dir=gt_cfg.get("cache_dir", data_cfg.get("cache_dir")),
                        seed=train_cfg.get("seed", 42) + 300_000,
                        target_mode=gt_cfg.get("target_mode", "segmentation"),
                        include_resolution=False,
                        auto_norms=gt_cfg.get("auto_norms", True),
                        auto_norms_per_image=gt_cfg.get("auto_norms_per_image", False),
                        auto_norms_percentile_low=gt_cfg.get("auto_norms_percentile_low", data_cfg.get("auto_norms_percentile_low", 1.0)),
                        auto_norms_percentile_high=gt_cfg.get("auto_norms_percentile_high", data_cfg.get("auto_norms_percentile_high", 99.0)),
                        multi_organelle_prob=0.0,
                        negative_example_prob=0.0,
                        prompt_variation=False,
                    )
                validation_dataset = WeightedDatasetAdapter(
                    gt_val_base,
                    label_weight=gt_cfg.get("label_weight", 1.0),
                    prompt=prompt,
                )
                validation_datasets = [
                    ("val/pseudo_samples", pseudo_validation_dataset),
                    ("val/gt_samples", validation_dataset),
                ]
    else:
        from .dataset import CellMapFluxDataset

        dataset = CellMapFluxDataset(
            data_root=data_cfg.get("data_root", "/nrs/cellmap/data"),
            norms_csv=data_cfg.get("norms_csv"),
            organelle_keys=data_cfg.get("organelles"),
            samples_per_epoch=data_cfg.get("samples_per_epoch", 2000),
            min_mask_fraction=data_cfg.get("min_mask_fraction", 0.01),
            skip_datasets=data_cfg.get("skip_datasets"),
            include_datasets=data_cfg.get("include_datasets"),
            cache_dir=data_cfg.get("cache_dir"),
            seed=train_cfg.get("seed", 42),
            target_mode=data_cfg.get("target_mode", "overlay"),
            include_resolution=data_cfg.get("include_resolution", False),
            auto_norms=data_cfg.get("auto_norms", False),
            auto_norms_per_image=data_cfg.get("auto_norms_per_image", False),
            auto_norms_percentile_low=data_cfg.get("auto_norms_percentile_low", 1.0),
            auto_norms_percentile_high=data_cfg.get("auto_norms_percentile_high", 99.0),
            multi_organelle_prob=data_cfg.get("multi_organelle_prob", 0.0),
            negative_example_prob=data_cfg.get("negative_example_prob", 0.0),
            prompt_variation=data_cfg.get("prompt_variation", False),
        )

    num_workers = train_cfg.get("num_workers", min(len(os.sched_getaffinity(0)), 8))
    dataloader_kwargs = {
        "batch_size": train_cfg.get("batch_size", 1),
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": train_cfg.get("pin_memory", torch.cuda.is_available()),
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = train_cfg.get(
            "persistent_workers", True
        )
        dataloader_kwargs["prefetch_factor"] = train_cfg.get("prefetch_factor", 2)
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # Optimizer
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable LoRA parameters: {num_params:,}")

    if train_cfg.get("use_8bit_adam", True):
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                trainable_params, lr=train_cfg.get("learning_rate", 1e-4)
            )
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params, lr=train_cfg.get("learning_rate", 1e-4)
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=train_cfg.get("learning_rate", 1e-4)
        )

    # LR scheduler
    lr_scheduler_type = train_cfg.get("lr_scheduler", "constant")
    warmup_steps = train_cfg.get("lr_warmup_steps", 200)
    max_train_steps = train_cfg.get("max_train_steps", 5000)

    if lr_scheduler_type == "constant":
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

        lr_scheduler = LambdaLR(optimizer, lr_lambda)
    elif lr_scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR

        warmup_scheduler = LambdaLR(
            optimizer, lambda step: step / max(warmup_steps, 1)
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max_train_steps - warmup_steps
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        from torch.optim.lr_scheduler import LambdaLR
        lr_scheduler = LambdaLR(optimizer, lambda step: 1.0)

    # Prepare with accelerator
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    # Initialize tracker
    if accelerator.is_main_process:
        init_kwargs = {}
        if "wandb" in _tracker_names(report_to) or "all" in _tracker_names(report_to):
            wandb_kwargs = {}
            if log_cfg.get("run_name") is not None:
                wandb_kwargs["name"] = log_cfg["run_name"]
            if log_cfg.get("entity") is not None:
                wandb_kwargs["entity"] = log_cfg["entity"]
            if log_cfg.get("tags") is not None:
                wandb_kwargs["tags"] = log_cfg["tags"]
            wandb_kwargs["dir"] = str(output_dir)
            init_kwargs["wandb"] = wandb_kwargs
        accelerator.init_trackers(
            log_cfg.get("project", "flux-lora-training"),
            config=flatten_tracker_config(config),
            init_kwargs=init_kwargs,
        )

    # Latent dimensions for 1024x1024 images
    # VAE scale factor is 8, so 1024/8 = 128 latent size
    latent_h = 1024 // vae_scale_factor
    latent_w = 1024 // vae_scale_factor
    num_channels_latents = vae.config.latent_channels  # 16
    # After packing, spatial dims are halved (Flux1/Kontext and Flux2 both)
    packed_h = latent_h // 2
    packed_w = latent_w // 2

    # Training loop
    global_step = 0
    weighting_scheme = train_cfg.get("weighting_scheme", "sigma_sqrt")
    checkpointing_steps = train_cfg.get("checkpointing_steps", 500)
    validation_steps = train_cfg.get("validation_steps", 500)
    train_image_log_steps = train_cfg.get("train_image_log_steps")
    if not isinstance(train_image_log_steps, (int, float)) or not train_image_log_steps:
        train_image_log_steps = None

    # Check model type for conditioning strategy
    is_kontext = "Kontext" in pretrained
    is_flux2 = "FLUX.2" in pretrained
    flux2_conditioning = train_cfg.get("flux2_conditioning", "noise_endpoint")
    flux2_noise_mix = train_cfg.get("flux2_noise_mix", 0.5)
    # Whether to concatenate conditioning (Kontext always, Flux2 optionally)
    use_concat_cond = is_kontext or (is_flux2 and flux2_conditioning == "concatenate")
    # Whether to use EM as noise endpoint (Flux2 noise_endpoint mode)
    use_noise_endpoint = is_flux2 and flux2_conditioning == "noise_endpoint"

    logger.info(f"Starting training for {max_train_steps} steps")
    logger.info(f"Model: {pretrained} (kontext={is_kontext}, flux2={is_flux2})")
    if is_flux2:
        logger.info(f"Flux2 conditioning: {flux2_conditioning}, noise_mix: {flux2_noise_mix}")
    logger.info(f"LoRA rank={lora_rank}, alpha={lora_alpha}")

    num_epochs = math.ceil(max_train_steps / len(dataloader))

    progress_bar = tqdm(
        total=max_train_steps,
        desc="Training",
        disable=not accelerator.is_main_process,
    )

    for epoch in range(num_epochs):
        transformer.train()

        for batch in dataloader:
            if global_step >= max_train_steps:
                break

            cond_images, target_images, prompts, valid_masks, sample_weights = batch

            with accelerator.accumulate(transformer):
                # Encode images to latents
                cond_tensor = pil_to_tensor(cond_images, device, dtype)
                target_tensor = pil_to_tensor(target_images, device, dtype)
                loss_mask = None
                if valid_masks is not None:
                    mask_tensor = pil_masks_to_tensor(valid_masks, device, dtype)
                    loss_mask = pack_loss_mask(mask_tensor, packed_h, packed_w)
                sample_weights_tensor = None
                if sample_weights is not None:
                    sample_weights_tensor = torch.tensor(
                        sample_weights, device=device, dtype=dtype
                    )

                with torch.no_grad():
                    target_latents = encode_images(vae, target_tensor, is_flux2=is_flux2)
                    if use_concat_cond or use_noise_endpoint:
                        cond_latents = encode_images(vae, cond_tensor, is_flux2=is_flux2)

                # Pack latents to sequence format
                target_packed = pack_latents(target_latents, is_flux2=is_flux2)
                batch_size = target_packed.shape[0]

                if use_concat_cond or use_noise_endpoint:
                    cond_packed = pack_latents(cond_latents, is_flux2=is_flux2)

                # Prepare IDs
                latent_ids = prepare_latent_image_ids(
                    packed_h, packed_w, device, dtype,
                    is_flux2=is_flux2, batch_size=batch_size,
                )

                if use_concat_cond:
                    if is_flux2:
                        image_ids = prepare_flux2_cond_ids(
                            packed_h, packed_w, device, dtype,
                            batch_size=batch_size,
                        )
                        all_ids = torch.cat([latent_ids, image_ids], dim=1)
                    else:
                        image_ids = prepare_latent_image_ids(
                            packed_h, packed_w, device, dtype,
                        )
                        image_ids[..., 0] = 1  # Mark as conditioning
                        all_ids = torch.cat([latent_ids, image_ids], dim=0)
                else:
                    all_ids = latent_ids

                # Encode text
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                        pipe, prompts, device, dtype, is_flux2=is_flux2,
                    )

                # Sample random timesteps (flow matching: sigma in [0, 1])
                # Using logit-normal distribution as in Flux training
                u = torch.randn(batch_size, device=device, dtype=dtype)
                sigmas = torch.sigmoid(u)  # logit-normal -> [0, 1]
                sigmas = sigmas.view(-1, 1, 1)

                # Create noisy latents
                noise = torch.randn_like(target_packed)
                if use_noise_endpoint:
                    # Flux2 noise_endpoint: blend EM latents with noise as the
                    # flow endpoint, matching img2img inference where the EM
                    # image is the starting point.
                    endpoint = (1 - flux2_noise_mix) * cond_packed + flux2_noise_mix * noise
                    noisy_latents = (1 - sigmas) * target_packed + sigmas * endpoint
                else:
                    # Standard: x_t = (1 - sigma) * clean + sigma * noise
                    noisy_latents = (1 - sigmas) * target_packed + sigmas * noise

                # Concatenate conditioning for Kontext / Flux2-concatenate mode
                if use_concat_cond:
                    hidden_states = torch.cat(
                        [noisy_latents, cond_packed], dim=1
                    )
                else:
                    hidden_states = noisy_latents

                # Prepare timestep (sigma * 1000 to match inference convention)
                timestep = (sigmas.squeeze() * 1000).to(dtype)
                if timestep.dim() == 0:
                    timestep = timestep.unsqueeze(0)

                # Guidance embedding
                guidance = None
                t_config = transformer.module.config if hasattr(transformer, 'module') else transformer.config
                has_guidance = getattr(t_config, 'guidance_embeds', False) or is_flux2
                if has_guidance:
                    guidance_scale = train_cfg.get("guidance_scale", 3.5)
                    guidance = torch.full(
                        [batch_size], guidance_scale, device=device, dtype=torch.float32
                    )

                # Forward pass
                fwd_kwargs = dict(
                    hidden_states=hidden_states,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=all_ids,
                    return_dict=False,
                )
                if not is_flux2:
                    fwd_kwargs["pooled_projections"] = pooled_prompt_embeds
                model_output = transformer(**fwd_kwargs)[0]

                # Only take predictions for the main latents (not conditioning)
                model_pred = model_output[:, : target_packed.shape[1]]

                # Compute loss (endpoint is the "noise" side of the flow)
                flow_endpoint = endpoint if use_noise_endpoint else noise
                loss = compute_flow_matching_loss(
                    model_pred,
                    flow_endpoint,
                    target_packed,
                    sigmas.squeeze(-1).squeeze(-1),
                    weighting_scheme=weighting_scheme,
                    loss_mask=loss_mask,
                    sample_weights=sample_weights_tensor,
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            # Logging
            if accelerator.is_main_process:
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                logger.info(
                    f"step={global_step}, loss={logs['loss']:.4f}, lr={logs['lr']:.2e}"
                )

                # Log training images periodically
                if train_image_log_steps is not None and global_step % train_image_log_steps == 0:
                    n = min(4, len(cond_images))
                    grid = _build_image_grid(cond_images[:n], target_images[:n])
                    log_image_grid(
                        accelerator, report_to, "train/samples", grid, global_step
                    )

            # Checkpointing
            if (
                global_step % checkpointing_steps == 0
                and accelerator.is_main_process
            ):
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                unwrapped = accelerator.unwrap_model(transformer)
                lora_state_dict = get_peft_model_state_dict(unwrapped)

                PipelineClass.save_lora_weights(
                    str(ckpt_dir),
                    transformer_lora_layers=lora_state_dict,
                )
                logger.info(f"Saved checkpoint to {ckpt_dir}")

            # Validation: log input, ground truth, and prediction images
            if (
                global_step % validation_steps == 0
                and accelerator.is_main_process
            ):
                num_val_images = train_cfg.get("num_validation_images", 4)
                val_sets = validation_datasets or [
                    ("val/samples", validation_dataset or dataset)
                ]
                for val_tag, val_dataset in val_sets:
                    run_validation(
                        pipe=pipe,
                        transformer=transformer,
                        dataset=val_dataset,
                        accelerator=accelerator,
                        global_step=global_step,
                        report_to=report_to,
                        tag=val_tag,
                        target_mode=data_cfg.get("target_mode", "overlay"),
                        num_images=num_val_images,
                        num_inference_steps=train_cfg.get("num_steps", 28),
                        guidance_scale=train_cfg.get("guidance_scale", 3.5),
                    )

    progress_bar.close()

    # Save final weights
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(transformer)
        lora_state_dict = get_peft_model_state_dict(unwrapped)

        PipelineClass.save_lora_weights(
            str(output_dir),
            transformer_lora_layers=lora_state_dict,
        )
        logger.info(f"Saved final LoRA weights to {output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(args.config, args.resume)
