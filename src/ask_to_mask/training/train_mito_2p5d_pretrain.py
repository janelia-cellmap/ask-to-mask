"""Self-supervised masked-image-modeling pretraining for the 2.5D ConvNeXt encoder.

No labels, no GT, no pseudo-labels -- just raw EM crops sampled across every
scale in each dataset's multiscale pyramid. The resulting encoder checkpoint
is meant to be loaded into `ConvNeXtMitoUNet` via `model.encoder_checkpoint`
before the supervised curriculum fine-tune in `train_mito_2p5d.py`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from .mito_2p5d_model import build_mito_2p5d_pretrain_model, masked_reconstruction_loss
from .mito_2p5d_pretrain_dataset import (
    build_mito_2p5d_pretrain_datasets,
    collate_self_supervised,
)
from .prefetch import CudaPrefetcher
from .train_mito_2p5d import (
    RunLogger,
    _autocast_context,
    _build_lr_scheduler,
    _gray_rgb,
    _move_batch_to_device,
    _resize_tile,
    load_checkpoint,
    load_config,
    save_checkpoint,
)
from .worker_seeding import reseed_dataset_worker

logger = logging.getLogger(__name__)


def build_visual_panel(
    batch: dict, pred: torch.Tensor, mask: torch.Tensor, max_items: int = 4, tile_size: int = 160
) -> Image.Image:
    em = batch["em"].detach().float().cpu().numpy()
    pred_np = pred.detach().float().cpu().numpy()
    mask_np = mask.detach().float().cpu().numpy()[:, 0]

    n = min(max_items, em.shape[0])
    depth = em.shape[1]
    center_idx = depth // 2
    row_h = tile_size + 32
    image_w = tile_size * 4
    image_h = row_h * n + 24
    panel = Image.new("RGB", (image_w, image_h), "white")
    draw = ImageDraw.Draw(panel)
    for i, label in enumerate(["original", "masked input", "reconstruction", "mask"]):
        draw.text((i * tile_size + 4, 4), label, fill=(0, 0, 0))

    for i in range(n):
        y0 = 24 + i * row_h
        original = em[i, center_idx]
        masked_input = original * (1.0 - mask_np[i])
        reconstruction = np.clip(pred_np[i, center_idx], 0.0, 1.0)

        panel.paste(_resize_tile(_gray_rgb(original), tile_size), (0, y0))
        panel.paste(_resize_tile(_gray_rgb(masked_input), tile_size), (tile_size, y0))
        panel.paste(_resize_tile(_gray_rgb(reconstruction), tile_size), (2 * tile_size, y0))
        panel.paste(
            _resize_tile(_gray_rgb(mask_np[i]), tile_size, Image.NEAREST), (3 * tile_size, y0)
        )

        meta = batch.get("metadata", [{}])[i]
        caption = f"dataset={meta.get('dataset')} res_nm={meta.get('resolution_nm')}"
        draw.text((4, y0 + tile_size + 2), caption[:100], fill=(0, 0, 0))

    return panel


def _encoder_module(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod.encoder if hasattr(model, "_orig_mod") else model.encoder


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    run_logger: RunLogger,
    step: int,
    max_batches: int,
) -> float:
    model.eval()
    losses = []
    first_batch, first_pred, first_mask = None, None, None
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        batch = _move_batch_to_device(batch, device)
        with _autocast_context(device, mixed_precision):
            pred, mask = model(batch["em"])
            loss = masked_reconstruction_loss(pred, batch["em"], mask)
        losses.append(float(loss.detach().cpu()))
        if first_batch is None:
            first_batch = {
                k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()
            }
            first_pred, first_mask = pred.detach().cpu(), mask.detach().cpu()

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    run_logger.scalar_dict({"val/loss": mean_loss}, step)
    if first_batch is not None:
        panel = build_visual_panel(first_batch, first_pred, first_mask)
        run_logger.image("val/samples", panel, step)
    model.train()
    return mean_loss


def train(config_path: str | Path, resume_from: str | None = None) -> Path:
    config = load_config(config_path)
    train_cfg = config.get("training", {})
    seed = train_cfg.get("seed", 42)
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    if resume_from is not None:
        output_dir = Path(resume_from).parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(train_cfg.get("output_dir", "runs/mito-2p5d-pretrain")) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_from is not None:
        # Record the resume config separately instead of overwriting the
        # original run's train_config.yaml/resolved_config.json.
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

    train_dataset, val_dataset = build_mito_2p5d_pretrain_datasets(config)
    num_workers = int(train_cfg.get("num_workers", min(len(os.sched_getaffinity(0)), 8)))
    loader_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 8)),
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": collate_self_supervised,
        "pin_memory": bool(train_cfg.get("pin_memory", torch.cuda.is_available())),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        loader_kwargs["worker_init_fn"] = reseed_dataset_worker
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_kwargs = dict(loader_kwargs)
    val_kwargs["shuffle"] = False
    val_kwargs["batch_size"] = int(
        train_cfg.get("validation_batch_size", loader_kwargs["batch_size"])
    )
    val_loader = DataLoader(val_dataset, **val_kwargs)

    model = build_mito_2p5d_pretrain_model(config).to(device)
    if train_cfg.get("compile", False):
        model = torch.compile(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    logger.info("Trainable pretraining model parameters: %s", f"{num_params:,}")
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
        lr=float(train_cfg.get("learning_rate", 1.5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
    )
    scheduler = _build_lr_scheduler(optimizer, train_cfg)
    global_step = 0
    if resume_from is not None:
        global_step = load_checkpoint(
            resume_from, model, optimizer=optimizer, scheduler=scheduler, map_location=device
        )
        logger.info("Resumed pretraining from %s at step %d", resume_from, global_step)

    run_logger = RunLogger(output_dir, config)
    max_steps = int(train_cfg.get("max_train_steps", 20000))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    log_steps = int(train_cfg.get("log_steps", 20))
    validation_steps = int(train_cfg.get("validation_steps", 1000))
    max_validation_batches = int(train_cfg.get("max_validation_batches", 16))
    checkpointing_steps = int(train_cfg.get("checkpointing_steps", 2000))
    clip_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and mixed_precision == "fp16")
    )

    logger.info("Starting self-supervised 2.5D pretraining for %d steps", max_steps)
    logger.info("Output directory: %s", output_dir)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=max_steps, initial=global_step, desc="mito-2.5d-pretrain")
    best_val_loss = float("inf")
    micro_step = 0

    while global_step < max_steps:
        for batch in CudaPrefetcher(train_loader, device, _move_batch_to_device):
            if global_step >= max_steps:
                break
            with _autocast_context(device, mixed_precision):
                pred, mask = model(batch["em"])
                loss = masked_reconstruction_loss(pred, batch["em"], mask)
                loss_to_backprop = loss / grad_accum

            scaler.scale(loss_to_backprop).backward()
            micro_step += 1
            if micro_step % grad_accum != 0:
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
                run_logger.scalar_dict(
                    {
                        "train/loss": float(loss.detach().cpu()),
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    global_step,
                )
                progress.set_postfix(
                    loss=f"{float(loss):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

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
                torch.save(_encoder_module(model).state_dict(), path / "encoder.pt")
                logger.info("Saved checkpoint to %s", path)

            if validation_steps > 0 and global_step % validation_steps == 0:
                val_loss = run_validation(
                    model,
                    val_loader,
                    device,
                    mixed_precision,
                    run_logger,
                    global_step,
                    max_validation_batches,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    path = save_checkpoint(
                        output_dir, model, optimizer, scheduler, global_step, config, name="best"
                    )
                    torch.save(_encoder_module(model).state_dict(), path / "encoder.pt")

    progress.close()
    final_path = save_checkpoint(
        output_dir, model, optimizer, scheduler, global_step, config, name="final"
    )
    torch.save(_encoder_module(model).state_dict(), final_path / "encoder.pt")
    run_logger.close()
    logger.info("Finished self-supervised 2.5D pretraining at step %d", global_step)
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(args.config, resume_from=args.resume)
