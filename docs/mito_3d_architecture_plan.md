# 3D Migration Plan for the Discriminative Mitochondria Model

## Status

Not started. This is a plan, not an in-progress implementation. The current
2.5D pipeline (`docs/mito_2p5d_training_handoff.md`) is deliberately being
kept as-is and run first; this document exists so the 3D option is scoped
and ready to pick up once there's a reason to invest in it (see "When to
actually do this" below).

## Why this is on the table at all

The 2.5D model reads a 9-slice EM z-stack but only ever supervises the
*center* slice, treating the other 8 slices as input context channels, not
targets. Two things make this look like a real gap rather than just a design
preference:

1. **The GT crops are genuinely 3D-annotated, not 2D slices.** Checked
   directly against the code: `discover_crops` in
   `src/ask_to_mask/training/zarr_utils.py` filters out any crop with fewer
   than 16 voxels in *any* dimension, including z (`min_crop_voxels=16`), and
   `CropInfo.crop_extent_world` carries a real z-extent per crop. Most GT
   crops therefore have real 3D-consistent annotation across many z-slices,
   and the 2.5D pipeline is discarding almost all of it as mere "context."
2. **The baseline being challenged may itself be 3D-native.** The existing
   UNet inference pipeline at `data.segmentation_path_template` is a
   production connectomics segmentation tool; if it's 3D-consistent (typical
   for this kind of pipeline), comparing a 2.5D model against it on
   merge/split-error metrics -- which are inherently about 3D object
   continuity -- structurally disadvantages the 2.5D model for reasons
   unrelated to training quality (curriculum, pretraining, loss weights).

Also relevant background from earlier discussion: the field-standard
approach for this exact problem (dense organelle segmentation in FIB-SEM
connectomics volumes) is a full 3D U-Net predicting affinities or local shape
descriptors (LSDs), trained on genuine 3D sub-volumes -- not a 2D-channel-stacked
approximation. The 2.5D approach here was a deliberate, practical shortcut to
stay compute-light and reuse existing 2D-oriented sampling code, not a claim
that it's the more "correct" architecture for this data.

## What has to change

### 1. Data loading -- genuine 3D sub-volumes, not z-stack-as-channels

Current: `Mito2p5DFixedFovGtDataset`/`Mito2p5DInferenceMitoDataset` read a
`[stack_depth, H, W]` window and treat z as channels; `finalize_mito_2p5d_sample`
supervises only the center `[1, H, W]` slice.

3D version needs:

- A dataset that reads a true `[D, H, W]` sub-volume of **raw EM** and the
  matching `[D, H, W]` **label volume** (not a single center-slice label).
  GT crops already have this label depth available (per the 16-voxel-minimum
  check above); the pseudo-label path would need the inference segmentation
  read as a 3D block too, which `segmentation_path_template` should already
  support since it's presumably a 3D-consistent volume.
- Patch size in 3D is a real memory constraint. FOVs like the current
  1024x1024 (2D) don't translate directly -- 3D training patches are
  typically much smaller per side (something like 64-256 voxels/side
  depending on GPU memory and 3D conv depth), so this is a genuinely
  different sampling regime, not a drop-in resize.
- `apply_stack_augment`'s rotation/flip logic generalizes to 3D (rotate/flip
  along 3D axes) but needs a real 3D-aware rewrite, not a channel-axis reuse.
- The comparison-mask mechanism (existing UNet baseline attached to GT
  validation crops) should carry over conceptually -- same idea, just reading
  a 3D block from the baseline segmentation instead of a 2D window.

### 2. Model architecture -- 3D encoder/decoder

Current: `ConvNeXtMitoUNet`, a 2D ConvNeXt encoder (`nn.Conv2d` throughout)
with a 2D FPN/UNet-style decoder.

Options, roughly in order of how much they lean on existing code:

- **3D ConvNeXt-style encoder** (swap `Conv2d`→`Conv3d`, `GroupNorm`/`LayerNorm`
  adapted to 5D tensors `[B, C, D, H, W]`) with a 3D FPN decoder. Closest to
  the current code shape, but there's no off-the-shelf pretrained 3D ConvNeXt
  to lean on the way `pretrained: true` currently uses ImageNet weights --
  natural-image pretraining doesn't transfer to a 3D conv stem at all, so
  this option effectively forces training (or self-supervised pretraining,
  see below) entirely from scratch on EM data anyway.
- **Field-standard 3D U-Net predicting affinities/LSDs** (the "MALA"-style
  architecture common in connectomics, e.g. via `gunpowder`/`daisy`). More of
  a rewrite relative to the current codebase, but it's the actual
  battle-tested approach for this specific data modality and task, and it's
  plausible the existing UNet baseline pipeline already uses something like
  this -- worth checking directly before choosing an architecture, since
  matching it makes the "does our model beat the baseline" comparison cleaner.
- Whichever is chosen, output should probably still include a boundary/affinity
  channel alongside the raw mask, mirroring the current multi-channel output
  design (`output_channels: 3` for mask/boundary/SDF) -- in 3D, affinities
  (predicted connectivity between adjacent voxels) are usually more useful
  than a raw SDF channel for merge/split-error reduction specifically.

### 3. Losses -- 3D-native versions

`mito_2p5d_losses.py`'s BCE/Dice/Tversky are elementwise and technically
dimension-agnostic (they'd work on 3D tensors without change), but:

- Boundary BCE and the SDF loss currently compute 2D boundaries/distance
  transforms (`mask_to_boundary`, `mask_to_signed_distance` in
  `mito_2p5d_dataset.py` use 2D `binary_dilation`/`binary_erosion`/
  `distance_transform_edt`). These need real 3D versions (`scipy.ndimage`
  supports 3D inputs for all of these already, so it's a signature/shape
  change, not a new algorithm).
- If moving to an affinity-based 3D U-Net, the loss function changes more
  fundamentally (affinity loss, not a same-shape mask reconstruction loss) --
  this is the bigger design decision to make before writing loss code.

### 4. Metrics -- 3D-native merge/split/object metrics

`mito_2p5d_metrics.py`'s `object_metrics` (merge errors, split errors, object
recall) currently runs 2D connected-component labeling (`scipy.ndimage.label`
with a `(3,3)` structure) on a single slice. This is the metric that most
directly measures what full 3D context should improve -- but as written it
can't see across z at all, so it's currently blind to exactly the failure
mode 3D is supposed to fix. A 3D version needs 3D connected-component
labeling (`scipy.ndimage.label` already supports a `(3,3,3)` structure) over
the full predicted/target sub-volume.

### 5. What transfers conceptually without a rewrite

These ideas from the 2.5D work aren't 2D-specific and should carry over
directly to a 3D version:

- **Two-stage curriculum** (pseudo-heavy → GT-dominant): the routing logic in
  `Mito2p5DMixedDataset` doesn't care what shape the underlying samples are.
- **Self-supervised masked-image-modeling pretraining**: the SimMIM-style
  approach in `ConvNeXtMaskedAutoencoder` generalizes to 3D (mask 3D patches
  instead of 2D ones, 3D encoder, `PixelShuffle`→a 3D equivalent or a small
  3D transposed-conv head). The multi-scale-via-pyramid-level sampling
  (`Mito2p5DSelfSupervisedDataset`) also carries over directly, since the
  OME-NGFF pyramid is inherently 3D already -- 2D was ask-to-mask's own
  simplification, not a data limitation.
- **Comparison-baseline mechanism** (`comparison_masks`, attaching the
  existing UNet pipeline's output to validation samples): same idea, 3D
  read instead of 2D.
- **GT-only control run** as a required baseline: same rationale applies in 3D.

## Suggested phased approach (if/when this is pursued)

1. **Confirm the baseline's actual architecture.** Check whether
   `segmentation_path_template`'s pipeline is itself a 3D model (and ideally
   what it predicts -- affinities, LSDs, direct semantic masks). This
   directly informs whether a 3D ConvNeXt-UNet or a field-standard
   affinity-based 3D U-Net is the more apples-to-apples choice.
2. **Build the 3D data-loading path first**, independent of model choice --
   read real 3D sub-volumes (raw + label) at a workable patch size, verify
   memory/throughput on a real GPU before committing to an architecture.
3. **Port self-supervised pretraining to 3D** using the same unlabeled-data
   advantage as the 2.5D version, since it needs no labels and can start
   immediately once 3D data loading works.
4. **Pick and implement the 3D architecture + losses + metrics**, informed by
   step 1.
5. **Re-run the same curriculum + GT-only-control comparison structure**
   already built for 2.5D, now in 3D, against the same UNet baseline.

## When to actually do this

Not now. Recommended sequencing: let the 2.5D pipeline (curriculum + optional
self-supervised pretraining + GT-only control, all already built) run first
and produce real `val/gt` vs. `val/gt/comparison/unet` numbers. That tells you
cheaply whether the overall approach (pseudo+GT training, EM-native
pretraining) has any signal at all, before paying for a much larger rewrite.
If 2.5D shows real promise but is capped by known-y architecture mismatch
(bad merge/split-error numbers specifically, which the 2D metrics can't even
fully see -- another reason to look at raw dice/IoU alongside object metrics
in the meantime), that's the trigger to start this plan for real.
