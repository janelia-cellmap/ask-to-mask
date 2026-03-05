# ask-to-mask

Generate organelle segmentation masks from EM images using Flux image editing models.

The idea: send an EM image to a Flux model with a prompt like *"Color all the mitochondria in bright red"*, then extract a binary segmentation mask from the color difference between the original and edited images.

## Setup

Requires [pixi](https://pixi.sh).

```bash
pixi install
pixi run install-torch-cu126
```

## Usage

### Segment organelles

```bash
# Single image, one organelle
pixi run segment segment --input image.png --output-dir ./masks/ --organelles mito --save-colored

# Multiple organelles
pixi run segment segment --input image.png --output-dir ./masks/ --organelles mito nucleus er

# With resolution info (nm/pixel) for better prompts
pixi run segment segment --input image.png --output-dir ./masks/ --organelles mito --resolution 4.0 --save-colored

# Batch directory
pixi run segment segment --input-dir ./em_images/ --output-dir ./masks/ --organelles mito
```

`--save-colored` saves the intermediate colored image alongside the mask for visual inspection.

When `--resolution` is provided (in nm/pixel), prompts include organelle descriptions and expected sizes in pixels, helping the model distinguish organelles by scale.

### Instance segmentation

Use `--instance` to color each organelle instance a different color, then extract separate labels via connected components:

```bash
pixi run segment segment --input image.png --output-dir ./masks/ --organelles mito --instance --save-colored
```

Output is a uint16 PNG where each pixel value is an instance ID (0 = background).

### List available organelle classes

```bash
pixi run segment list-organelles
```

### Use LoRA weights

Run inference with finetuned LoRA weights:

```bash
pixi run segment segment --input image.png --output-dir ./masks/ --organelles mito --lora checkpoints/flux-kontext-lora --save-colored
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `kontext-dev` | Flux model (`kontext-dev` or `flux2-dev`) |
| `--lora` | None | Path to LoRA weights directory |
| `--resolution` | None | Image resolution in nm/pixel for enhanced prompts |
| `--num-steps` | `28` | Number of inference steps |
| `--guidance-scale` | `3.5` | How strongly to follow the prompt |
| `--strength` | `0.75` | Edit strength (0=no change, 1=full regeneration) |
| `--threshold` | `200.0` | Color difference threshold for mask extraction |
| `--instance` | off | Instance segmentation mode (each instance a different color) |
| `--seed` | None | Random seed for reproducibility |
| `--custom-prompt` | None | Override the default prompt |
| `--device` | `cuda` | Torch device |

## How it works

1. Load EM image, convert to RGB, resize to 1024x1024
2. Construct a prompt from the organelle config (e.g., *"Color all the mitochondria in bright red. Keep everything else unchanged."*). When `--resolution` is provided, the prompt includes EM-specific descriptions and expected pixel sizes.
3. Run the Flux model (locally via HuggingFace diffusers) to produce a colored version of the image
4. Extract a binary mask by finding saturated pixels in the target color direction (semantic mode), or detect any colored pixels and label connected components (instance mode)
5. Resize mask back to original image dimensions and save

## Supported organelles

Each organelle is assigned a distinct high-contrast color for clean mask extraction:

| Key | Organelle | Color | Approx. size |
|-----|-----------|-------|--------------|
| `mito` | Mitochondria | Red | 500-2000 nm |
| `er` | Endoplasmic reticulum | Green | 50-100 nm (tubule diameter) |
| `nucleus` | Nucleus | Blue | 5-15 μm |
| `lipid_droplet` | Lipid droplets | Yellow | 100-5000 nm |
| `plasma_membrane` | Plasma membrane | Cyan | 7-8 nm (thickness) |
| `nuclear_envelope` | Nuclear envelope | Magenta | 30-50 nm (thickness) |
| `nuclear_pore` | Nuclear pores | Orange | 100-140 nm |
| `nucleolus` | Nucleolus | Purple | 1-5 μm |
| `heterochromatin` | Heterochromatin | Spring green | 100-5000 nm |
| `euchromatin` | Euchromatin | Rose | 100-10000 nm |

## LoRA finetuning

Finetune Flux on annotated CellMap EM data to improve organelle recognition.

### Setup

```bash
pixi run install-train-deps
```

### Train

```bash
pixi run train --config configs/train_lora.yaml
```

Edit `configs/train_lora.yaml` to configure:
- `model.pretrained`: which Flux model to finetune (`FLUX.1-Kontext-dev` or `FLUX.2-dev`)
- `model.lora.rank`: LoRA rank (default 16)
- `data.organelles`: which organelles to train on
- `data.data_root`: path to CellMap zarr data
- `training.max_train_steps`: number of training steps
- `training.output_dir`: where to save LoRA weights

Training data is read directly from CellMap zarr volumes. The dataset creates (EM slice, colored EM slice, prompt) triplets by coloring annotated organelle regions with the organelle's designated color.

Checkpoints are saved periodically and can be used for inference with `--lora`.

### Hardware

Requires A100 80GB or equivalent. Uses gradient checkpointing and 8-bit Adam to fit in memory.

## Project structure

```
src/ask_to_mask/
  cli.py           # CLI entry point (segment, list-organelles, train)
  config.py        # Organelle class definitions and model registry
  model.py         # Flux model loading and inference (with LoRA support)
  pipeline.py      # Orchestrates load → prompt → infer → postprocess
  postprocess.py   # Mask extraction (semantic + instance)
  training/
    dataset.py     # CellMapFluxDataset: zarr-backed training data
    zarr_utils.py  # Zarr reading utilities (adapted from sam3m)
    train.py       # LoRA training loop with accelerate + PEFT
configs/
  train_lora.yaml  # Training configuration
  norms.csv        # Per-dataset intensity normalization
```
