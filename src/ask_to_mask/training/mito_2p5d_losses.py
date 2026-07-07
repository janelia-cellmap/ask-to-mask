"""Segmentation losses for supervised 2.5D mitochondria training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def _expand_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if mask.shape == value.shape:
        return mask
    return mask.expand_as(value)


def masked_mean(
    value: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample weighted mean, then batch mean."""
    valid_mask = _expand_mask(valid_mask, value).to(dtype=value.dtype)
    reduce_dims = tuple(range(1, value.ndim))
    denom = valid_mask.sum(dim=reduce_dims).clamp_min(1.0)
    per_sample = (value * valid_mask).sum(dim=reduce_dims) / denom
    if sample_weight is not None:
        per_sample = per_sample * sample_weight.to(value.device, value.dtype)
    return per_sample.mean()


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    pos_weight: float | None = None,
) -> torch.Tensor:
    kwargs = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = torch.tensor(
            [float(pos_weight)], device=logits.device, dtype=logits.dtype
        )
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none", **kwargs)
    return masked_mean(loss, valid_mask, sample_weight)


def masked_focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - pt).pow(gamma) * bce
    return masked_mean(loss, valid_mask, sample_weight)


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    valid_mask = _expand_mask(valid_mask, prob).to(dtype=prob.dtype)
    reduce_dims = tuple(range(1, prob.ndim))
    intersection = (prob * target * valid_mask).sum(dim=reduce_dims)
    denom = ((prob + target) * valid_mask).sum(dim=reduce_dims)
    loss = 1.0 - (2.0 * intersection + eps) / (denom + eps)
    if sample_weight is not None:
        loss = loss * sample_weight.to(loss.device, loss.dtype)
    return loss.mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    valid_mask = _expand_mask(valid_mask, prob).to(dtype=prob.dtype)
    reduce_dims = tuple(range(1, prob.ndim))
    tp = (prob * target * valid_mask).sum(dim=reduce_dims)
    fp = (prob * (1.0 - target) * valid_mask).sum(dim=reduce_dims)
    fn = ((1.0 - prob) * target * valid_mask).sum(dim=reduce_dims)
    loss = 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    if sample_weight is not None:
        loss = loss * sample_weight.to(loss.device, loss.dtype)
    return loss.mean()


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    beta: float = 0.1,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return masked_mean(loss, valid_mask, sample_weight)


@dataclass
class MitoLossConfig:
    bce_weight: float = 1.0
    focal_weight: float = 0.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_weight: float = 1.0
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    boundary_weight: float = 0.2
    distance_weight: float = 0.0
    distance_beta: float = 0.1
    pos_weight: float | None = None


class Mito2p5DLoss(nn.Module):
    """Composite loss for mask, optional boundary, and optional SDF channels."""

    def __init__(self, config: MitoLossConfig):
        super().__init__()
        self.config = config

    @classmethod
    def from_config(cls, config: dict) -> "Mito2p5DLoss":
        loss_cfg = config.get("loss", {})
        return cls(
            MitoLossConfig(
                bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
                focal_weight=float(loss_cfg.get("focal_weight", 0.0)),
                focal_alpha=float(loss_cfg.get("focal_alpha", 0.25)),
                focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
                dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
                tversky_weight=float(loss_cfg.get("tversky_weight", 0.0)),
                tversky_alpha=float(loss_cfg.get("tversky_alpha", 0.3)),
                tversky_beta=float(loss_cfg.get("tversky_beta", 0.7)),
                boundary_weight=float(loss_cfg.get("boundary_weight", 0.2)),
                distance_weight=float(loss_cfg.get("distance_weight", 0.0)),
                distance_beta=float(loss_cfg.get("distance_beta", 0.1)),
                pos_weight=loss_cfg.get("pos_weight"),
            )
        )

    def forward(
        self,
        logits: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.config
        target = batch["target"].to(logits.device, logits.dtype)
        valid_mask = batch["valid_mask"].to(logits.device, logits.dtype)
        sample_weight = batch.get("sample_weight")
        if sample_weight is not None:
            sample_weight = sample_weight.to(logits.device, logits.dtype)

        mask_logits = logits[:, 0:1]
        components: dict[str, torch.Tensor] = {}
        total = mask_logits.new_tensor(0.0)

        if cfg.bce_weight > 0:
            value = masked_bce_with_logits(
                mask_logits,
                target,
                valid_mask,
                sample_weight=sample_weight,
                pos_weight=cfg.pos_weight,
            )
            components["loss/bce"] = value.detach()
            total = total + cfg.bce_weight * value

        if cfg.focal_weight > 0:
            value = masked_focal_loss_with_logits(
                mask_logits,
                target,
                valid_mask,
                sample_weight=sample_weight,
                alpha=cfg.focal_alpha,
                gamma=cfg.focal_gamma,
            )
            components["loss/focal"] = value.detach()
            total = total + cfg.focal_weight * value

        if cfg.dice_weight > 0:
            value = soft_dice_loss(
                mask_logits,
                target,
                valid_mask,
                sample_weight=sample_weight,
            )
            components["loss/dice"] = value.detach()
            total = total + cfg.dice_weight * value

        if cfg.tversky_weight > 0:
            value = tversky_loss(
                mask_logits,
                target,
                valid_mask,
                sample_weight=sample_weight,
                alpha=cfg.tversky_alpha,
                beta=cfg.tversky_beta,
            )
            components["loss/tversky"] = value.detach()
            total = total + cfg.tversky_weight * value

        if cfg.boundary_weight > 0 and "boundary_target" in batch:
            boundary_target = batch["boundary_target"].to(logits.device, logits.dtype)
            boundary_logits = logits[:, 1:2] if logits.shape[1] > 1 else mask_logits
            value = masked_bce_with_logits(
                boundary_logits,
                boundary_target,
                valid_mask,
                sample_weight=sample_weight,
            )
            components["loss/boundary"] = value.detach()
            total = total + cfg.boundary_weight * value

        if cfg.distance_weight > 0 and logits.shape[1] > 2 and "distance_target" in batch:
            distance_target = batch["distance_target"].to(logits.device, logits.dtype)
            distance_pred = torch.tanh(logits[:, 2:3])
            value = masked_smooth_l1(
                distance_pred,
                distance_target,
                valid_mask,
                sample_weight=sample_weight,
                beta=cfg.distance_beta,
            )
            components["loss/distance"] = value.detach()
            total = total + cfg.distance_weight * value

        components["loss/total"] = total.detach()
        return total, components
