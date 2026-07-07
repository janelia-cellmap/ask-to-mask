"""Train a supervised 2.5D EM -> mitochondria mask model."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from .mito_2p5d_dataset import (
    Mito2p5DMixedDataset,
    build_mito_2p5d_datasets,
    collate_mito_2p5d,
    mask_to_boundary,
)
from .mito_2p5d_losses import Mito2p5DLoss
from .mito_2p5d_metrics import (
    average_metric_dicts,
    evaluate_batch_predictions,
    evaluate_comparison_masks,
)
from .mito_2p5d_model import build_mito_2p5d_model
from .prefetch import CudaPrefetcher
from .worker_seeding import reseed_dataset_worker

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _tracker_names(report_to) -> set[str]:
    if report_to is None:
        return set()
    if isinstance(report_to, str):
        return {report_to.lower()}
    return {str(item).lower() for item in report_to}


class RunLogger:
    """Small TensorBoard/W&B wrapper for scalar and visual logging."""

    def __init__(self, output_dir: Path, config: dict):
        log_cfg = config.get("logging", {})
        self.output_dir = output_dir
        self.report_to = _tracker_names(log_cfg.get("report_to", "tensorboard"))
        self.tb = None
        self.wandb = None
        if "tensorboard" in self.report_to or "all" in self.report_to:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                logger.warning("TensorBoard logging requested but tensorboard is not installed")
            else:
                self.tb = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        if "wandb" in self.report_to or "all" in self.report_to:
            try:
                import wandb
            except ImportError:
                logger.warning("W&B logging requested but wandb is not installed")
            else:
                self.wandb = wandb
                wandb.init(
                    project=log_cfg.get("project", "mito-2p5d"),
                    name=log_cfg.get("run_name"),
                    entity=log_cfg.get("entity"),
                    tags=log_cfg.get("tags"),
                    dir=str(output_dir),
                    config=config,
                )

    def scalar_dict(self, values: dict[str, float], step: int) -> None:
        if self.tb is not None:
            for key, value in values.items():
                self.tb.add_scalar(key, float(value), step)
            self.tb.flush()
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def image(self, tag: str, image: Image.Image, step: int, caption: str | None = None) -> None:
        image_dir = self.output_dir / "visuals"
        image_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = tag.replace("/", "_")
        path = image_dir / f"{safe_tag}_{step:07d}.png"
        image.save(path)
        if self.tb is not None:
            arr = np.asarray(image).transpose(2, 0, 1)
            self.tb.add_image(tag, arr, step)
            if caption:
                self.tb.add_text(f"{tag}/metadata", caption, step)
            self.tb.flush()
        if self.wandb is not None:
            self.wandb.log(
                {tag: self.wandb.Image(image, caption=caption or "")},
                step=step,
            )

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            moved[key] = {
                sub_key: sub_value.to(device, non_blocking=True)
                if torch.is_tensor(sub_value)
                else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def _autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision in {"no", "none", None}:
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unknown mixed_precision={mixed_precision!r}")


def _build_lr_scheduler(optimizer, train_cfg: dict):
    max_steps = int(train_cfg.get("max_train_steps", 5000))
    warmup = int(train_cfg.get("lr_warmup_steps", 200))
    min_ratio = float(train_cfg.get("min_lr_ratio", 0.05))
    scheduler_type = train_cfg.get("lr_scheduler", "cosine")

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if scheduler_type == "constant":
            return 1.0
        progress = (step - warmup) / max(1, max_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _model_state_dict(model: torch.nn.Module) -> dict:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model.state_dict()


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    global_step: int,
    config: dict,
    name: str,
) -> Path:
    ckpt_dir = output_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _model_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "global_step": global_step,
            "config": config,
        },
        ckpt_dir / "state.pt",
    )
    return ckpt_dir


def load_checkpoint(
    checkpoint: str | Path,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> int:
    state = torch.load(Path(checkpoint) / "state.pt", map_location=map_location)
    target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    target_model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state.get("global_step", 0))


def _gray_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def _mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask > 0] = color
    return rgb


def _resize_tile(arr: np.ndarray, size: int, resample=Image.BILINEAR) -> Image.Image:
    return Image.fromarray(arr).resize((size, size), resample=resample)


def _overlay(center: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    base = _gray_rgb(center).astype(np.float32)
    out = base.copy()
    target_color = np.array([0, 190, 80], dtype=np.float32)
    pred_color = np.array([255, 40, 40], dtype=np.float32)
    out[target] = 0.55 * out[target] + 0.45 * target_color
    out[pred] = 0.45 * out[pred] + 0.55 * pred_color
    both = target & pred
    out[both] = 0.35 * base[both] + 0.65 * np.array([255, 220, 0], dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def build_visual_panel(
    batch: dict,
    logits: torch.Tensor,
    max_items: int = 4,
    threshold: float = 0.5,
    tile_size: int = 160,
    stack_tile_size: int = 72,
) -> tuple[Image.Image, str]:
    em = batch["em"].detach().float().cpu().numpy()
    target = batch["target"].detach().float().cpu().numpy()[:, 0] > 0.5
    valid = batch["valid_mask"].detach().float().cpu().numpy()[:, 0] > 0
    prob = torch.sigmoid(logits[:, 0:1]).detach().float().cpu().numpy()[:, 0]
    pred = prob >= threshold
    boundary_prob = None
    if logits.shape[1] > 1:
        boundary_prob = torch.sigmoid(logits[:, 1:2]).detach().float().cpu().numpy()[:, 0]
    boundary_target = batch.get("boundary_target")
    if boundary_target is not None:
        boundary_target = boundary_target.detach().float().cpu().numpy()[:, 0] > 0.5

    n = min(max_items, em.shape[0])
    depth = em.shape[1]
    row_h = tile_size + 44
    stack_w = depth * stack_tile_size
    image_w = stack_w + 6 * tile_size
    image_h = row_h * n + 28
    panel = Image.new("RGB", (image_w, image_h), "white")
    draw = ImageDraw.Draw(panel)
    labels = ["z-stack", "center", "target", "pred", "overlay", "boundary", "valid"]
    x_positions = [0, stack_w]
    for _ in range(5):
        x_positions.append(x_positions[-1] + tile_size)
    for label_text, x in zip(labels, x_positions):
        draw.text((x + 4, 4), label_text, fill=(0, 0, 0))

    captions = []
    for i in range(n):
        y0 = 28 + i * row_h
        center_idx = depth // 2
        center = em[i, center_idx]

        stack_strip = Image.new("RGB", (stack_w, stack_tile_size), "white")
        for z in range(depth):
            tile = _resize_tile(_gray_rgb(em[i, z]), stack_tile_size)
            stack_strip.paste(tile, (z * stack_tile_size, 0))
        panel.paste(stack_strip, (0, y0))

        panel.paste(_resize_tile(_gray_rgb(center), tile_size), (stack_w, y0))
        panel.paste(
            _resize_tile(_mask_rgb(target[i], (0, 210, 90)), tile_size, Image.NEAREST),
            (stack_w + tile_size, y0),
        )
        panel.paste(
            _resize_tile(_mask_rgb(pred[i], (255, 40, 40)), tile_size, Image.NEAREST),
            (stack_w + 2 * tile_size, y0),
        )
        panel.paste(
            _resize_tile(_overlay(center, pred[i], target[i]), tile_size),
            (stack_w + 3 * tile_size, y0),
        )
        if boundary_prob is not None:
            boundary_img = _gray_rgb(boundary_prob[i])
        elif boundary_target is not None:
            boundary_img = _mask_rgb(boundary_target[i], (255, 255, 255))
        else:
            boundary_img = _mask_rgb(mask_to_boundary(pred[i]), (255, 255, 255))
        panel.paste(
            _resize_tile(boundary_img, tile_size),
            (stack_w + 4 * tile_size, y0),
        )
        panel.paste(
            _resize_tile(_gray_rgb(valid[i].astype(np.float32)), tile_size, Image.NEAREST),
            (stack_w + 5 * tile_size, y0),
        )

        meta = batch.get("metadata", [{}])[i]
        caption = (
            f"dataset={meta.get('dataset')} crop={meta.get('crop_id')} "
            f"fov={meta.get('fov_nm_yx')} nm_per_px={meta.get('nm_per_px_yx')} "
            f"quality={meta.get('label_quality')} "
            f"mask_frac={meta.get('mask_fraction_valid', meta.get('mask_fraction'))}"
        )
        draw.text((4, y0 + stack_tile_size + 4), caption[:220], fill=(0, 0, 0))
        captions.append(caption)

    return panel, "\n".join(captions)


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    loss_fn: Mito2p5DLoss,
    dataloader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    logger_obj: RunLogger,
    step: int,
    tag: str,
    config: dict,
) -> dict[str, float]:
    model.eval()
    train_cfg = config.get("training", {})
    metric_cfg = config.get("metrics", {})
    max_batches = int(train_cfg.get("max_validation_batches", 8))
    threshold = float(metric_cfg.get("threshold", 0.5))
    boundary_radius = int(metric_cfg.get("boundary_radius_px", 2))
    boundary_tolerance = int(metric_cfg.get("boundary_tolerance_px", 2))
    object_match_iou = float(metric_cfg.get("object_match_iou", 0.1))

    loss_items: list[dict[str, float]] = []
    metric_items: list[dict[str, float]] = []
    comparison_items: dict[str, list[dict[str, float]]] = defaultdict(list)
    first_batch = None
    first_logits = None

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        batch = _move_batch_to_device(batch, device)
        with _autocast_context(device, mixed_precision):
            logits = model(batch["em"])
            _loss, components = loss_fn(logits, batch)
        loss_items.append({k: float(v.detach().cpu()) for k, v in components.items()})
        metric_items.append(
            evaluate_batch_predictions(
                logits,
                batch,
                threshold=threshold,
                boundary_radius_px=boundary_radius,
                boundary_tolerance_px=boundary_tolerance,
                object_match_iou=object_match_iou,
            )
        )
        if "comparison_masks" in batch:
            comparisons = evaluate_comparison_masks(
                batch["comparison_masks"],
                batch,
                threshold=threshold,
                boundary_radius_px=boundary_radius,
                boundary_tolerance_px=boundary_tolerance,
                object_match_iou=object_match_iou,
            )
            for name, values in comparisons.items():
                comparison_items[name].append(values)
        if first_batch is None:
            first_batch = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
            first_logits = logits.detach().cpu()

    losses = average_metric_dicts(loss_items)
    metrics = average_metric_dicts(metric_items)
    scalars = {f"{tag}/{k}": v for k, v in {**losses, **metrics}.items()}
    for name, items in comparison_items.items():
        for key, value in average_metric_dicts(items).items():
            scalars[f"{tag}/comparison/{name}/{key}"] = value
    logger_obj.scalar_dict(scalars, step)

    if first_batch is not None and first_logits is not None:
        panel, caption = build_visual_panel(
            first_batch,
            first_logits,
            max_items=int(train_cfg.get("num_validation_images", 4)),
            threshold=threshold,
        )
        logger_obj.image(f"{tag}/samples", panel, step, caption=caption)

    model.train()
    return metrics


def train(config_path: str | Path, resume_from: str | None = None) -> Path:
    config = load_config(config_path)
    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})
    seed = train_cfg.get("seed", 42)
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    if resume_from is not None:
        # Continue writing into the run directory the checkpoint came from,
        # rather than starting a fresh timestamped directory.
        output_dir = Path(resume_from).parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(train_cfg.get("output_dir", "runs/mito-2p5d")) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_from is not None:
        # Record the resume config separately instead of overwriting the
        # original run's train_config.yaml/resolved_config.json -- otherwise
        # a `--resume ... --config edited.yaml` silently rewrites history,
        # making it look like the whole run (including steps before the
        # resume) used the edited hyperparameters.
        resume_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(config_path, output_dir / f"train_config_resumed_{resume_tag}.yaml")
        with open(output_dir / f"resolved_config_resumed_{resume_tag}.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
    else:
        shutil.copy2(config_path, output_dir / "train_config.yaml")
        with open(output_dir / "resolved_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    mixed_precision = train_cfg.get("mixed_precision", "bf16")

    train_dataset, validation_datasets = build_mito_2p5d_datasets(config)
    num_workers = int(train_cfg.get("num_workers", min(len(os.sched_getaffinity(0)), 8)))
    loader_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 2)),
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": collate_mito_2p5d,
        "pin_memory": bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        # Without this, forked workers inherit an identical pre-fork copy of
        # each dataset's self.rng (seeded once in the main process) and would
        # all sample the same sequence -- see worker_seeding.py.
        loader_kwargs["worker_init_fn"] = reseed_dataset_worker
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    val_loaders = []
    for tag, dataset in validation_datasets:
        val_kwargs = dict(loader_kwargs)
        val_kwargs["shuffle"] = False
        val_kwargs["batch_size"] = int(train_cfg.get("validation_batch_size", loader_kwargs["batch_size"]))
        val_loaders.append((tag, DataLoader(dataset, **val_kwargs)))

    # Optional two-stage curriculum: train mostly on plentiful pseudo labels
    # first to learn general mitochondria shape/scale, then shift toward the
    # scarce GT labels to correct the model's output toward true boundaries.
    # Implemented as two DataLoaders (different `gt_sample_prob`) rather than
    # a live-mutated sampling probability, since persistent DataLoader workers
    # keep their own copy of the dataset and won't see in-place attribute
    # changes. Stage 2 rebuilds the underlying pseudo/GT datasets from scratch
    # with a distinct `seed_offset` instead of reusing stage 1's dataset
    # objects: each dataset's `self.rng` only ever advances inside forked
    # DataLoader worker processes, so reusing the same (still-pristine, in the
    # main process) objects would make stage 2's freshly-forked workers replay
    # stage 1's early sampling sequence instead of exploring further. The
    # rebuild is cheap since crop discovery / the inference-pair index are
    # both cached to disk (`cache_dir`/`index_path`).
    curriculum_cfg = train_cfg.get("curriculum", {}) or {}
    stage1_steps = int(curriculum_cfg.get("stage1_steps", 0))
    stage2_gt_sample_prob = curriculum_cfg.get("stage2_gt_sample_prob")
    stage2_loader = None
    if curriculum_cfg.get("enabled", False) and (stage1_steps <= 0 or stage2_gt_sample_prob is None):
        logger.warning(
            "training.curriculum.enabled is true but stage1_steps=%r / "
            "stage2_gt_sample_prob=%r is missing/invalid; curriculum is disabled",
            stage1_steps,
            stage2_gt_sample_prob,
        )
    if curriculum_cfg.get("enabled", False) and stage1_steps > 0 and stage2_gt_sample_prob is not None:
        if isinstance(train_dataset, Mito2p5DMixedDataset):
            stage1_gt_sample_prob = train_dataset.gt_sample_prob
            stage2_dataset, _ = build_mito_2p5d_datasets(
                config,
                gt_sample_prob_override=float(stage2_gt_sample_prob),
                seed_offset=500_000,
            )
            stage2_loader = DataLoader(stage2_dataset, **loader_kwargs)
            logger.info(
                "Curriculum enabled: stage 1 (gt_sample_prob=%.3f) for %d steps, "
                "then stage 2 (gt_sample_prob=%.3f) for the remainder",
                stage1_gt_sample_prob,
                stage1_steps,
                float(stage2_gt_sample_prob),
            )
        else:
            logger.warning(
                "training.curriculum is enabled but dataset_type is not "
                "mixed_mito_gt_pseudo; ignoring curriculum"
            )

    model = build_mito_2p5d_model(config).to(device)
    if train_cfg.get("compile", False):
        model = torch.compile(model)
    loss_fn = Mito2p5DLoss.from_config(config)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    logger.info("Trainable supervised model parameters: %s", f"{num_params:,}")
    if torch.cuda.is_available() and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "CUDA device: %s, capability=%s.%s, total_memory=%.1f GB",
            props.name,
            props.major,
            props.minor,
            props.total_memory / 1024**3,
        )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
    )
    scheduler = _build_lr_scheduler(optimizer, train_cfg)
    global_step = 0
    if resume_from is not None:
        global_step = load_checkpoint(
            resume_from,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        logger.info("Resumed supervised training from %s at step %d", resume_from, global_step)

    run_logger = RunLogger(output_dir, config)
    max_steps = int(train_cfg.get("max_train_steps", 5000))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    log_steps = int(log_cfg.get("log_steps", train_cfg.get("log_steps", 10)))
    train_image_log_steps = int(train_cfg.get("train_image_log_steps", 250))
    validation_steps = int(train_cfg.get("validation_steps", 500))
    checkpointing_steps = int(train_cfg.get("checkpointing_steps", 500))
    clip_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and mixed_precision == "fp16")
    )

    logger.info("Starting supervised 2.5D mito training for %d steps", max_steps)
    logger.info("Output directory: %s", output_dir)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=max_steps, initial=global_step, desc="mito-2.5d")
    best_dice: dict[str, float] = defaultdict(lambda: -1.0)
    micro_step = 0
    active_loader = train_loader

    while global_step < max_steps:
        if stage2_loader is not None and active_loader is train_loader and global_step >= stage1_steps:
            logger.info(
                "Switching to curriculum stage 2 (gt_sample_prob=%.3f) at step %d",
                float(stage2_gt_sample_prob),
                global_step,
            )
            active_loader = stage2_loader
            # Drop the reference (rather than `del`) so `active_loader is
            # train_loader` below still evaluates instead of raising
            # NameError. Stage-1's persistent workers would otherwise sit
            # idle (holding their own dataset/zarr-handle copies) for the
            # rest of training.
            train_loader = None
        for batch_idx, batch in enumerate(CudaPrefetcher(active_loader, device, _move_batch_to_device)):
            if global_step >= max_steps:
                break
            with _autocast_context(device, mixed_precision):
                logits = model(batch["em"])
                loss, components = loss_fn(logits, batch)
                loss_to_backprop = loss / grad_accum

            scaler.scale(loss_to_backprop).backward()
            micro_step += 1
            do_step = micro_step % grad_accum == 0
            if not do_step:
                continue

            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, float(clip_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)

            if global_step % log_steps == 0:
                train_metrics = {
                    key.replace("loss/", "train/loss/"): float(value.detach().cpu())
                    for key, value in components.items()
                }
                train_metrics["train/lr"] = scheduler.get_last_lr()[0]
                run_logger.scalar_dict(train_metrics, global_step)
                progress.set_postfix(
                    loss=f"{float(components['loss/total']):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            if train_image_log_steps > 0 and global_step % train_image_log_steps == 0:
                panel, caption = build_visual_panel(
                    {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()},
                    logits.detach().cpu(),
                    max_items=int(train_cfg.get("num_train_images", 4)),
                    threshold=float(config.get("metrics", {}).get("threshold", 0.5)),
                )
                run_logger.image("train/samples", panel, global_step, caption=caption)

            if checkpointing_steps > 0 and global_step % checkpointing_steps == 0:
                path = save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    config,
                    name=f"checkpoint-{global_step}",
                )
                logger.info("Saved checkpoint to %s", path)

            if validation_steps > 0 and global_step % validation_steps == 0:
                for val_tag, val_loader in val_loaders:
                    metrics = run_validation(
                        model,
                        loss_fn,
                        val_loader,
                        device,
                        mixed_precision,
                        run_logger,
                        global_step,
                        val_tag,
                        config,
                    )
                    dice = metrics.get("dice")
                    if dice is not None and dice > best_dice[val_tag]:
                        best_dice[val_tag] = dice
                        save_checkpoint(
                            output_dir,
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            config,
                            name=f"best-{val_tag.replace('/', '_')}",
                        )

            if (
                stage2_loader is not None
                and active_loader is train_loader
                and global_step >= stage1_steps
            ):
                # Re-enter the outer while loop so it picks up stage2_loader.
                break

    progress.close()
    save_checkpoint(
        output_dir,
        model,
        optimizer,
        scheduler,
        global_step,
        config,
        name="final",
    )
    run_logger.close()
    logger.info("Finished supervised 2.5D mito training at step %d", global_step)
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(args.config, resume_from=args.resume)
