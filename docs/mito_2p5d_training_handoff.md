# 2.5D Mitochondria Segmentation Training Handoff

## Goal

Build and evaluate a domain-specific supervised image-to-image model for EM mitochondria mask proposals.

The core question is:

> Can a dedicated 2.5D EM -> mitochondria mask model, trained on ground truth (plus the existing UNet inference pipeline's own pseudo-labels for scale/generality), produce mitochondria proposals that beat that existing UNet pipeline's own segmentations?

This is a comparison against the **existing UNet segmentation pipeline**, not against the Flux/Qwen/ask-to-mask diffusion path (that's a separate, unrelated question this project does not address). The pseudo-labels used for training here are themselves UNet-derived (cleaned-up post-processed UNet output), not diffusion output — see "Pseudo-Label Curriculum" below for why that matters for how training is structured. The initial scope is mitochondria only, fixed physical FOV, binary center-slice mask output, no prompts, no instance colors, and no decorative RGB targets.

**2.5D is a deliberate shortcut, not a claimed-correct architecture.** These structures (and their GT annotations — confirmed the crops carry real z-extent, not single slices) are genuinely 3D. See `docs/mito_3d_architecture_plan.md` for a scoped migration plan to a true 3D model, and why it isn't being started yet (short version: validate this cheaper 2.5D pipeline shows real signal first).

## What Makes It 2.5D

The model receives a small aligned z-stack of EM slices but predicts only the center-slice mitochondria mask.

```text
input:  [z-4, z-3, ..., z, ..., z+3, z+4]  -> [9, H, W]
output: center-slice mitochondria mask z      -> [1, H, W]
```

The network is still a 2D encoder-decoder. It treats neighboring z slices as input channels, so it gets axial context without predicting a full 3D volume.

## Main Code

- Dataset wrappers: `src/ask_to_mask/training/mito_2p5d_dataset.py`
- Model: `src/ask_to_mask/training/mito_2p5d_model.py`
- Losses: `src/ask_to_mask/training/mito_2p5d_losses.py`
- Metrics: `src/ask_to_mask/training/mito_2p5d_metrics.py`
- Trainer: `src/ask_to_mask/training/train_mito_2p5d.py`
- Script entry point: `scripts/train_mito_2p5d.py`
- CLI entry point: `src/ask_to_mask/cli.py`
- Default (mixed pseudo+GT, curriculum) config: `configs/train_mito_2p5d_fixed_fov.yaml`
- GT-only control config: `configs/train_mito_2p5d_gt_only.yaml`
- Self-supervised pretraining: `src/ask_to_mask/training/mito_2p5d_pretrain_dataset.py`, `train_mito_2p5d_pretrain.py`, `scripts/train_mito_2p5d_pretrain.py`, `configs/pretrain_mito_2p5d_mae.yaml` (see "Self-Supervised Pretraining" below)
- Pixi task: `pixi.toml`

## Data Contract

The implementation assumes the data loader can provide aligned EM crops, target masks, valid/loss masks, and metadata. It reuses the repo's existing fixed-FOV CellMap samplers rather than changing data-path plumbing.

Current dataset classes:

- `Mito2p5DInferenceMitoDataset`: fixed-FOV pseudo/inference mitochondria labels.
- `Mito2p5DFixedFovGtDataset`: fixed-FOV ground-truth mitochondria labels.
- `Mito2p5DMixedDataset`: mixes pseudo and GT samples with configurable GT probability.

Each sample is a dict:

```python
{
    "em": Tensor[D, H, W],
    "target": Tensor[1, H, W],
    "valid_mask": Tensor[1, H, W],
    "boundary_target": Tensor[1, H, W],      # optional
    "distance_target": Tensor[1, H, W],      # optional
    "metadata": dict,
    "sample_weight": float,
}
```

Important metadata logged in visual panels includes dataset, crop ID, FOV, nm/px, world origin, label quality, and mask fraction.

## Model

Current architecture: `ConvNeXtMitoUNet`.

Summary:

- ConvNeXt encoder from `torchvision`.
- Input channels equal z-stack depth, default `9`.
- FPN/UNet-style top-down decoder.
- Default output channels:
  - channel 0: mitochondria mask logits
  - channel 1: boundary logits
  - channel 2: signed-distance/SDF logits

The default config uses:

```yaml
model:
  architecture: convnext_unet
  encoder: convnext_small
  pretrained: false
  decoder_channels: 256
  output_channels: 3
```

`pretrained: true` should work through torchvision ConvNeXt weights, but the initial config leaves it false to avoid dependency/download surprises on the cluster.

## Losses

Composite loss in `mito_2p5d_losses.py`:

- BCE with logits for pixelwise foreground/background.
- Optional focal loss.
- Dice loss for foreground imbalance.
- Tversky loss for sparse mitochondria; current config weights false negatives more strongly.
- Boundary BCE for sharper object boundaries.
- Optional signed-distance/SDF SmoothL1 loss as a light shape regularizer.
- Valid-mask weighting so unannotated or uncertain pixels are ignored.
- Sample weighting so pseudo labels can be downweighted relative to GT.

Current default:

```yaml
loss:
  bce_weight: 1.0
  dice_weight: 0.7
  tversky_weight: 0.3
  tversky_alpha: 0.3
  tversky_beta: 0.7
  boundary_weight: 0.2
  distance_weight: 0.05
```

Review note: the loss is reasonable for mitochondria proposals. For noisy pseudo labels, consider comparing against `distance_weight: 0.0` and boundary-only auxiliary supervision before trusting SDF gains.

## Metrics

Validation metrics in `mito_2p5d_metrics.py`:

- Dice
- IoU
- pixel precision/recall
- boundary precision/recall/F1
- object recall
- false positive objects
- missed mitochondria
- merge errors
- split errors

There is also a `comparison_masks` hook, now wired up end to end. By default, GT validation datasets (`val/gt` in the mixed pipeline, or the single GT validation set in the `fixed_fov_gt` pathway) attach the existing UNet inference pipeline's own segmentation for the exact same crop (read from `data.segmentation_path_template`, the same zarr the pseudo-labels come from). This gives `val/gt/comparison/unet/*` scalars in TensorBoard/W&B — the UNet's own dice/IoU/boundary/object metrics against real GT, computed on the identical crops the model is evaluated on. That number is the actual baseline this project needs to beat; without it there was previously no way to answer the core question quantitatively.

Configurable via `data.comparison_masks` (all optional, defaults shown):

```yaml
data:
  comparison_masks:
    enabled: true                 # set false to disable entirely
    name: unet                    # scalar/metadata key, e.g. val/gt/comparison/unet/dice
    segmentation_path_template: ...  # defaults to data.segmentation_path_template
    label_name: mito                 # defaults to data.label_name
    target_resolution_nm: 8.0        # defaults to data.raw_target_resolution_nm
```

If a given dataset has no matching UNet segmentation on disk, that sample simply omits `comparison_masks` (no error); `collate_mito_2p5d` only stacks names common to every item in a batch, so heterogeneous availability across a batch degrades gracefully rather than crashing.

## Training

Trainer: `src/ask_to_mask/training/train_mito_2p5d.py`

Features:

- AdamW optimizer.
- Cosine learning-rate schedule with warmup.
- bf16/fp16 autocast.
- Gradient accumulation.
- Checkpointing: periodic, best, and final.
- TensorBoard/W&B logging.
- Visual panels for:
  - input z-stack
  - center EM slice
  - target mask
  - predicted mask
  - overlay
  - boundary output/target
  - valid mask
  - metadata caption

Default config uses conservative memory settings:

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 8
  mixed_precision: bf16
  num_workers: 2
  persistent_workers: true
  prefetch_factor: 2
  pin_memory: true
```

Effective batch size is `batch_size * gradient_accumulation_steps = 8`, not 1 — easy to misread from the config alone.

If `training.num_workers` is omitted from the config (unlike the default config above, which sets it explicitly), the trainer falls back to `min(len(os.sched_getaffinity(0)), 8)`, not a hardcoded `2`.

Note: `data.boundary_weight` (pseudo-label confidence/erosion weighting) and `loss.boundary_weight` (boundary BCE loss weight) are unrelated settings that happen to share a key name — do not confuse them when editing configs.

## Pseudo-Label Curriculum

Because the pseudo-labels are the existing UNet pipeline's own (post-processed) output, training mostly on them risks the new model just learning to reproduce that pipeline's habits — including its systematic errors — rather than exceeding it. A flat pseudo/GT mix for the whole run has no mechanism to prevent this.

Instead, training uses a two-stage curriculum (`training.curriculum` in the config):

- **Stage 1 (pseudo-heavy, most of the run)**: uses `data.gt.sample_prob` (e.g. `0.35`) to learn general mitochondria shape/scale/appearance across many datasets from the plentiful — but imperfect — pseudo-labels.
- **Stage 2 (GT-dominant, the remainder)**: switches to `training.curriculum.stage2_gt_sample_prob` (e.g. `0.8`) to correct the model toward true boundaries/shapes using the scarce ground truth.

```yaml
training:
  curriculum:
    enabled: true
    stage1_steps: 3500
    stage2_gt_sample_prob: 0.8
```

Implementation note: this is two separate `DataLoader`s over the same underlying pseudo/GT dataset objects (different `gt_sample_prob`), switched at the `stage1_steps` boundary — not a live-mutated sampling probability. Persistent `DataLoader` workers keep their own copy of the dataset object, so an in-place attribute change on the main-process dataset would not be visible to workers already spawned for the first loader.

### GT-only control run

`configs/train_mito_2p5d_gt_only.yaml` trains exclusively on ground truth (`data.dataset_type: fixed_fov_gt`, no pseudo-labels, no curriculum) as a required baseline, not just a nice-to-have. Compare its `val/gt` metrics against the mixed/curriculum run's `val/gt` metrics (and both against `val/gt/comparison/unet`):

- If GT-only already matches or beats the mixed run, pseudo-label training isn't adding value for this question — the curriculum should be simplified or dropped.
- If GT-only underperforms, pseudo pretraining is earning its complexity, but only if the curriculum keeps the model from converging back to imitating the UNet baseline it needs to beat.

Note: `fixed_fov_gt` now requires `data.gt.validation_datasets` to be set — it raises rather than silently sampling train/val from the same crop pool (see "Fixed Issues" below).

## Self-Supervised Pretraining

Separate from the pseudo/GT curriculum above: an optional stage that pretrains the ConvNeXt encoder on **raw, unlabeled EM data** before the supervised curriculum ever starts, so the encoder has genuine EM-domain understanding rather than relying on ImageNet-natural-image priors (`pretrained: true`) or a from-scratch discriminative fit to a comparatively small labeled set.

Motivation: `pretrained: true` gives generic edge/texture priors from natural photos, but nothing EM-specific. Full from-scratch diffusion-model-style pretraining (à la Flux/Nano Banana) is a different order of compute entirely and isn't a good fit for a discriminative model anyway — the goal here is a much cheaper, EM-native version of the same idea, scoped to a single small encoder.

**Code**: `mito_2p5d_pretrain_dataset.py`, `mito_2p5d_model.py` (`ConvNeXtMaskedAutoencoder`), `train_mito_2p5d_pretrain.py`, `scripts/train_mito_2p5d_pretrain.py`, `configs/pretrain_mito_2p5d_mae.yaml`.

**Objective**: SimMIM-style masked-image-modeling. Random 32×32 patches (matching the encoder's total downsampling stride) of the input z-stack are replaced with a learned mask token; the same dense ConvNeXt encoder `ConvNeXtMitoUNet` uses runs over the masked input; a single lightweight `1×1 conv + PixelShuffle(32)` head reconstructs the full input; loss is masked L1 computed only on the masked pixels. No labels of any kind are needed.

**Data — multi-scale via the existing OME-NGFF pyramid, not a fixed FOV**: `Mito2p5DSelfSupervisedDataset` discovers every dataset's raw EM zarr (same `em_path_template` as the pseudo-label pipeline) and samples a random *pyramid scale level* per crop, not just the finest resolution. Reading the same `output_size` pixel window at different pyramid levels naturally covers everything from mitochondria-scale to whole-cell-scale physical FOV in a single training run, without any per-organelle FOV configuration. Crops where the chosen scale is too small to contain `output_size` pixels are skipped and resampled (checked via the same bounds logic used elsewhere, no crash).

Scales are filtered by **physical resolution in nm/px** (`data.min_resolution_nm`/`max_resolution_nm`), not pyramid index — pyramid depth and base resolution both vary per dataset (checked against real data: `aic_desmosome-1` has 11 levels from 8nm/px to 8192nm/px; `jrc_hela-3` has 5 levels from 4nm/px to 64nm/px; `jrc_mus-liver-zon-2` has 14 levels up to 65536nm/px), so "pyramid index 5" means a different physical scale in each and isn't a meaningful cutoff across datasets. Including every pyramid level turned out to be overkill: at the coarsest levels (thousands to tens of thousands of nm/px) each pixel already averages over a huge patch of tissue, so there's little biological structure left to learn from, and it wastes training budget on crops that don't resemble anything the downstream fine-tuning task will see. The default `max_resolution_nm: 512.0` still gives a 512px crop a ~262µm span at its coarsest included scale — several times a whole cell's diameter (cells are roughly 20-50µm) — so it comfortably covers mito-through-cell context without wandering into the blurry, low-information tissue-scale end of the pyramid. `min_resolution_nm: null` means no lower bound, so the finest available scale (typically 4-8nm/px) is always included.

**Using the result**: the trainer saves `encoder.pt` (just the encoder's `state_dict`) alongside each full checkpoint. Point `model.encoder_checkpoint` in `train_mito_2p5d_fixed_fov.yaml`/`train_mito_2p5d_gt_only.yaml` at that file (`load_pretrained_encoder` in `mito_2p5d_model.py` loads it with `strict=True`, so `encoder`/`in_channels` must match between the pretraining and fine-tuning configs) instead of `pretrained: true`.

**Launch**:

```bash
pixi run pretrain-mito-2p5d --config configs/pretrain_mito_2p5d_mae.yaml
```

**Not yet done**: no pretraining run has actually been executed (only smoke-tested with synthetic tensors and a live check that `discover_raw_em_sources` correctly enumerates real multiscale pyramids). Step throughput, how many steps are needed before the encoder is actually useful downstream (checked via linear-probe or a short fine-tune), and whether `mask_ratio: 0.6`/`mask_patch_size: 32` are good defaults for EM data specifically are all open questions for the first real run.

## Launch

Preferred CLI:

```bash
pixi run train-mito-2p5d --config configs/train_mito_2p5d_fixed_fov.yaml
```

Equivalent script:

```bash
pixi run python scripts/train_mito_2p5d.py --config configs/train_mito_2p5d_fixed_fov.yaml
```

Resume:

```bash
pixi run train-mito-2p5d \
  --config configs/train_mito_2p5d_fixed_fov.yaml \
  --resume runs/mito-2p5d-fixed-fov/<timestamp>/checkpoint-1000
```

## Throughput And Loading

Current loading is reasonable but not fully optimized.

Already present:

- PyTorch `DataLoader` workers.
- `persistent_workers`.
- `prefetch_factor`.
- `pin_memory`.
- zarr handle caching inside dataset instances.
- gradient accumulation.

Not yet implemented:

- explicit CUDA async prefetcher
- adaptive real batch-size tuning
- chunk-aware crop batching
- in-worker decoded crop cache
- profiling-driven worker/batch selection

For initial runs, first tune real `batch_size` upward until GPU memory is near target usage, then tune `num_workers` and `prefetch_factor`. If GPU utilization stays low, add a CUDA prefetch wrapper and profile zarr read latency.

## Fixed Issues (previously "Known Issues", found in code review)

1. **Validation set was silently truncated to 4 samples.** Fixed: validation dataset size now defaults to `training.max_validation_batches * training.validation_batch_size` (overridable via `training.num_validation_samples`), fully decoupled from `training.num_validation_images` (which now only controls the TensorBoard visual panel, as originally intended).
2. **`best_dice` was shared across heterogeneous validation loaders.** Fixed: `best_dice` is now tracked per validation tag (`defaultdict` keyed by `val/pseudo`, `val/gt`, etc.), with separate `best-val_pseudo`/`best-val_gt` checkpoints.
3. **`--resume` did not continue the original run directory.** Fixed: when `--resume <path>` is passed, `output_dir` is now `Path(resume_from).parent` (the run directory the checkpoint came from) instead of a fresh timestamped directory.
4. **`fixed_fov_gt` (GT-only) had no train/val split.** Both train and validation datasets sampled from the exact same crop pool with only a different RNG seed — real leakage risk for the GT-only control run. Fixed: `fixed_fov_gt` now requires `data.gt.validation_datasets` and holds those datasets out entirely for validation, matching how the mixed pipeline already splits `val/gt`.
5. **`comparison_masks` was a dead code path.** Fixed: see "Metrics" and "Pseudo-Label Curriculum" above — GT validation crops now carry the existing UNet pipeline's own segmentation for direct comparison.

Z-stack/center-slice alignment and `valid_mask` application in losses/metrics were checked and are correct — no off-by-one or masking bugs found there.

## Suggested Review Questions

1. Does the fixed-FOV sampler correctly align the 9-slice EM input with the center-slice target mask?
2. Should the first serious run disable SDF loss for noisy pseudo labels?
3. Is `convnext_small` the right starting point, or should the first production run use `convnext_base`? Is `pretrained: false` costing real accuracy given the small GT set — worth an ablation with `pretrained: true`.
4. Does the validation split avoid leakage for both pseudo and GT sources? (Fixed for `fixed_fov_gt`; the mixed pipeline's pseudo split still uses a spatial holdout within a dataset rather than a fully separate volume — confirm the holdout margin exceeds the FOV.)
5. Is `stage1_steps: 3500` / `stage2_gt_sample_prob: 0.8` the right curriculum split, or should stage 2 be longer/shorter, or the transition annealed instead of a hard switch?
6. Is IO or compute the bottleneck once this runs on the actual cluster GPU?

## Validation Already Run

The code path has been checked with:

```bash
python3 -m compileall \
  src/ask_to_mask/training/mito_2p5d_dataset.py \
  src/ask_to_mask/training/mito_2p5d_model.py \
  src/ask_to_mask/training/mito_2p5d_losses.py \
  src/ask_to_mask/training/mito_2p5d_metrics.py \
  src/ask_to_mask/training/mito_2p5d_pretrain_dataset.py \
  src/ask_to_mask/training/train_mito_2p5d.py \
  src/ask_to_mask/training/train_mito_2p5d_pretrain.py \
  scripts/train_mito_2p5d.py \
  scripts/train_mito_2p5d_pretrain.py \
  src/ask_to_mask/cli.py
```

Also checked:

```bash
pixi run python -m ask_to_mask.cli train-mito-2p5d --help
pixi run python scripts/train_mito_2p5d.py --help
```

And a synthetic forward/loss/metrics smoke test using ConvNeXt-tiny on random tensors.

The curriculum/comparison-masks/leakage fixes above were additionally checked with a synthetic-tensor smoke test covering `apply_stack_augment` (with and without `extra_masks`), `finalize_mito_2p5d_sample` (with and without `comparison_masks`), and `collate_mito_2p5d` (homogeneous and heterogeneous `comparison_masks` across a batch), plus a config-loading check confirming `_comparison_mask_kwargs` and `training.curriculum` resolve as expected for both `configs/train_mito_2p5d_fixed_fov.yaml` and `configs/train_mito_2p5d_gt_only.yaml`, and that both CLI entry points still show `--help` correctly.

The self-supervised pretraining stage was checked with: a synthetic-tensor smoke test covering `random_patch_mask` (shape/validity, and that it correctly rejects non-`patch_size`-divisible dimensions), `ConvNeXtMaskedAutoencoder.forward` (shape round-trip on ConvNeXt-tiny), `masked_reconstruction_loss`, and — the critical integration check — that an encoder `state_dict` produced by `ConvNeXtMaskedAutoencoder` loads via `load_pretrained_encoder` into a freshly constructed `ConvNeXtMitoUNet` with matching weights (`strict=True`, no shape mismatches). Also live-verified against real data at `/nrs/cellmap/data` (not synthetic): `discover_raw_em_sources` correctly enumerates multiscale pyramids across several real datasets; `_filter_sources_by_resolution` correctly narrows those pyramids to the configured nm/px range; and a full `Mito2p5DSelfSupervisedDataset` instance, restricted to 3 real datasets with `max_resolution_nm=512`, successfully read complete `[9, 256, 256]` crops end-to-end, genuinely drawing from different scales across samples (observed both 8nm/px and 64nm/px crops in a small batch). `pretrain-mito-2p5d --help` and `scripts/train_mito_2p5d_pretrain.py --help` both work.

No full real-data training run (discriminative or pretraining) has been executed yet.

All files listed under "Main Code" and this doc itself are currently **untracked** in git (only `src/ask_to_mask/cli.py` is a tracked/modified file, for the `train-mito-2p5d` subcommand). Commit this work before treating any of the above file paths as stable references.
