"""Strong encoder-decoder models for 2.5D EM -> mitochondria masks."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


CONVNEXT_CHANNELS = {
    "convnext_tiny": [96, 192, 384, 768],
    "convnext_small": [96, 192, 384, 768],
    "convnext_base": [128, 256, 512, 1024],
}

CONVNEXT_WEIGHTS = {
    "convnext_tiny": "ConvNeXt_Tiny_Weights",
    "convnext_small": "ConvNeXt_Small_Weights",
    "convnext_base": "ConvNeXt_Base_Weights",
}


def _load_convnext_encoder(name: str, pretrained: bool):
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for ConvNeXtMitoUNet. "
            "Install torch/torchvision in the training environment."
        ) from exc

    if name not in CONVNEXT_CHANNELS:
        raise ValueError(
            f"Unknown ConvNeXt encoder {name!r}. "
            f"Choices: {sorted(CONVNEXT_CHANNELS)}"
        )

    fn = getattr(models, name)
    if not pretrained:
        return fn(weights=None)

    weights_enum = getattr(models, CONVNEXT_WEIGHTS[name], None)
    weights = weights_enum.DEFAULT if weights_enum is not None else None
    try:
        return fn(weights=weights)
    except TypeError:
        return fn(pretrained=True)


def _replace_convnext_stem(model: nn.Module, in_channels: int) -> None:
    """Replace ConvNeXt's RGB stem with a z-stack stem."""
    stem = model.features[0][0]
    if stem.in_channels == in_channels:
        return

    new_stem = nn.Conv2d(
        in_channels,
        stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        dilation=stem.dilation,
        groups=stem.groups,
        bias=stem.bias is not None,
        padding_mode=stem.padding_mode,
    )
    with torch.no_grad():
        if stem.weight.shape[1] == 3:
            mean_weight = stem.weight.mean(dim=1, keepdim=True)
            new_stem.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
        else:
            nn.init.kaiming_normal_(new_stem.weight, mode="fan_out", nonlinearity="relu")
        if stem.bias is not None and new_stem.bias is not None:
            new_stem.bias.copy_(stem.bias)
    model.features[0][0] = new_stem


class ConvNormAct(nn.Module):
    """Small decoder block used after top-down feature fusion."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = max(g for g in range(1, min(32, out_channels) + 1) if out_channels % g == 0)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _build_convnext_backbone(encoder: str, pretrained: bool, in_channels: int) -> nn.Module:
    """Build the ConvNeXt `.features` encoder shared by `ConvNeXtMitoUNet` and
    `ConvNeXtMaskedAutoencoder`. Both classes must construct this identically
    for `load_pretrained_encoder` weight transfer between them to work --
    routing both through this one function makes that structural rather than
    a convention enforced only by comments.
    """
    backbone = _load_convnext_encoder(encoder, pretrained=pretrained)
    _replace_convnext_stem(backbone, in_channels)
    return backbone.features


class ConvNeXtMitoUNet(nn.Module):
    """ConvNeXt encoder with FPN/UNet-style decoder for 2.5D EM stacks."""

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 1,
        encoder: str = "convnext_small",
        pretrained: bool = False,
        decoder_channels: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder_name = encoder
        self.encoder = _build_convnext_backbone(encoder, pretrained, in_channels)
        encoder_channels = CONVNEXT_CHANNELS[encoder]
        self.feature_indices = (1, 3, 5, 7)

        self.lateral = nn.ModuleList(
            nn.Conv2d(ch, decoder_channels, kernel_size=1)
            for ch in encoder_channels
        )
        self.smooth = nn.ModuleList(
            ConvNormAct(decoder_channels, decoder_channels)
            for _ in encoder_channels
        )
        self.refine = nn.Sequential(
            ConvNormAct(decoder_channels, decoder_channels),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = []
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in self.feature_indices:
                features.append(x)

        if len(features) != len(self.lateral):
            raise RuntimeError(
                f"Expected {len(self.lateral)} encoder features, got {len(features)}"
            )

        y = self.lateral[-1](features[-1])
        y = self.smooth[-1](y)
        for i in range(len(features) - 2, -1, -1):
            y = F.interpolate(
                y,
                size=features[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            y = y + self.lateral[i](features[i])
            y = self.smooth[i](y)

        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        return self.refine(y)


def build_mito_2p5d_model(config: dict) -> nn.Module:
    """Build the supervised 2.5D mitochondria segmentation model from config."""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    z_offsets = data_cfg.get("z_offsets")
    in_channels = len(z_offsets) if z_offsets is not None else int(data_cfg.get("stack_depth", 9))
    architecture = model_cfg.get("architecture", "convnext_unet")
    if architecture not in {"convnext_unet", "convnext_fpn_unet"}:
        raise ValueError(
            f"Unknown model architecture={architecture!r}; expected convnext_unet"
        )
    model = ConvNeXtMitoUNet(
        in_channels=in_channels,
        out_channels=int(model_cfg.get("output_channels", 1)),
        encoder=model_cfg.get("encoder", "convnext_small"),
        pretrained=bool(model_cfg.get("pretrained", False)),
        decoder_channels=int(model_cfg.get("decoder_channels", 256)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    encoder_checkpoint = model_cfg.get("encoder_checkpoint")
    if encoder_checkpoint:
        load_pretrained_encoder(model, encoder_checkpoint)
    return model


def load_pretrained_encoder(model: "ConvNeXtMitoUNet", checkpoint_path: str) -> None:
    """Load self-supervised-pretrained ConvNeXt encoder weights (see
    `ConvNeXtMaskedAutoencoder`/`train_mito_2p5d_pretrain.py`) into a fresh
    `ConvNeXtMitoUNet`. Requires the same `encoder`/`in_channels` so shapes match.
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "encoder" in state:
        state = state["encoder"]
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Encoder checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )


ENCODER_TOTAL_STRIDE = 32


def random_patch_mask(
    batch_size: int,
    height: int,
    width: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Return a [B, 1, H, W] binary mask (1 = masked) with whole `patch_size` blocks
    masked at random, independently per sample. `height`/`width` must be divisible
    by `patch_size`.
    """
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"height/width ({height}x{width}) must be divisible by patch_size={patch_size}"
        )
    grid_h, grid_w = height // patch_size, width // patch_size
    num_patches = grid_h * grid_w
    num_mask = max(1, int(round(num_patches * mask_ratio)))
    noise = torch.rand(batch_size, num_patches, device=device)
    threshold = torch.topk(noise, num_mask, dim=1).values[:, -1:]
    patch_mask = (noise >= threshold).float().view(batch_size, 1, grid_h, grid_w)
    return patch_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)


def masked_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean L1 error over masked pixels only (SimMIM-style)."""
    mask = mask.expand_as(pred)
    diff = (pred - target).abs() * mask
    return diff.sum() / mask.sum().clamp(min=1.0)


class ConvNeXtMaskedAutoencoder(nn.Module):
    """Self-supervised masked-image-modeling pretraining for the same ConvNeXt
    encoder `ConvNeXtMitoUNet` uses, so the encoder weights transfer directly.

    Follows SimMIM: random patches of the raw input are replaced with a learned
    mask token, the (dense, unmodified) encoder runs over the whole image, and a
    single lightweight linear-projection + pixel-shuffle head reconstructs the
    input at full resolution. Loss is computed only on masked pixels.
    """

    def __init__(
        self,
        in_channels: int = 9,
        encoder: str = "convnext_small",
        pretrained: bool = True,
        mask_patch_size: int = 32,
        mask_ratio: float = 0.6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_name = encoder
        self.encoder = _build_convnext_backbone(encoder, pretrained, in_channels)
        self.mask_patch_size = int(mask_patch_size)
        self.mask_ratio = float(mask_ratio)

        self.mask_token = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)

        deepest_channels = CONVNEXT_CHANNELS[encoder][-1]
        self.decoder = nn.Conv2d(
            deepest_channels, in_channels * ENCODER_TOTAL_STRIDE**2, kernel_size=1
        )
        self.pixel_shuffle = nn.PixelShuffle(ENCODER_TOTAL_STRIDE)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        mask = random_patch_mask(b, h, w, self.mask_patch_size, self.mask_ratio, x.device)
        x_masked = x * (1.0 - mask) + self.mask_token.expand(b, c, h, w) * mask

        feat = x_masked
        for layer in self.encoder:
            feat = layer(feat)

        pred = self.pixel_shuffle(self.decoder(feat))
        if pred.shape[-2:] != (h, w):
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        return pred, mask


def build_mito_2p5d_pretrain_model(config: dict) -> ConvNeXtMaskedAutoencoder:
    """Build the self-supervised masked-autoencoder pretraining model from config."""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    z_offsets = data_cfg.get("z_offsets")
    in_channels = len(z_offsets) if z_offsets is not None else int(data_cfg.get("stack_depth", 9))
    return ConvNeXtMaskedAutoencoder(
        in_channels=in_channels,
        encoder=model_cfg.get("encoder", "convnext_small"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        mask_patch_size=int(model_cfg.get("mask_patch_size", ENCODER_TOTAL_STRIDE)),
        mask_ratio=float(model_cfg.get("mask_ratio", 0.6)),
    )
