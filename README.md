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

## Agentic refinement

Iteratively improve mask quality using a VLM evaluator agent. Each iteration generates a mask, sends it (alongside the original EM image) to a vision-language model for critique, and uses the feedback to refine the prompt and parameters.

### Setup

```bash
# Install agent dependencies (ollama backend)
pip install -e '.[agents]'

# For other LLM providers:
pip install -e '.[agents-anthropic]'  # Claude
pip install -e '.[agents-google]'     # Gemini
pip install -e '.[agents-openai]'     # GPT-4o
```

The ollama backend requires a running ollama server with a vision model:

```bash
ollama serve
ollama pull llama3.2-vision
```

### Run

```bash
# Basic refinement loop with ollama (free, local)
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito

# Use a different LLM provider
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
    --llm-provider anthropic --llm-model claude-sonnet-4-20250514

# Customize loop parameters
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
    --max-iterations 3 --min-score 0.9 --guidance-scale 4.0

# Use Flux2 with LoRA weights
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
    --model flux2-dev --lora checkpoints/flux2-lora
```

Each iteration saves `colored.png`, `mask.png`, and `evaluation.json` to `iteration_00/`, `iteration_01/`, etc. The loop stops when the evaluator scores the mask above `--min-score` or `--max-iterations` is reached.

### Architecture

Both the image generation model and the evaluator VLM are pluggable:

- **Image gen backends** (`--gen-backend`): `flux` (default). Extensible to other models.
- **LLM backends** (`--llm-provider`): `ollama`, `anthropic`, `google`, `openai`.

## LoRA finetuning

Finetune Flux on annotated CellMap EM data to improve organelle recognition.

### Setup

```bash
pixi run install-train-deps
```

### Train

```bash
# Kontext model (image editing approach)
pixi run train --config configs/train_lora.yaml

# Flux2-dev model (img2img approach)
pixi run train --config configs/train_lora_flux2.yaml

# Flux2-dev with auto-computed intensity normalization
pixi run train --config configs/train_lora_flux2_r64_autonorm.yaml
```

Training data is read directly from CellMap zarr volumes. The dataset creates (EM slice, target image, prompt) triplets by coloring annotated organelle regions.

Checkpoints are saved periodically and can be used for inference with `--lora`.

### Training config options

Edit `configs/train_lora.yaml` (Kontext) or `configs/train_lora_flux2.yaml` (Flux2) to configure:

| Key | Default | Description |
|-----|---------|-------------|
| `model.pretrained` | varies | Flux model to finetune (`FLUX.1-Kontext-dev` or `FLUX.2-dev`) |
| `model.lora.rank` | `16` | LoRA rank |
| `data.organelles` | all | Which organelles to train on |
| `data.data_root` | `/nrs/cellmap/data` | Path to CellMap zarr data |
| `data.target_mode` | `overlay` | `overlay` = EM + colored organelles, `segmentation` = colored mask on black background |
| `data.auto_norms` | `false` | Compute per-dataset intensity normalization from data percentiles instead of using `norms_csv` |
| `data.auto_norms_percentile_low` | `1.0` | Lower percentile for auto-norm clipping |
| `data.auto_norms_percentile_high` | `99.0` | Upper percentile for auto-norm clipping |
| `data.include_resolution` | `true` | Include image resolution (nm/px) in training prompts |
| `data.multi_organelle_prob` | `0.0` | Probability of coloring 2-3 organelles per image (e.g. `0.3`) |
| `data.negative_example_prob` | `0.0` | Probability of serving a no-organelle example where target = input (e.g. `0.15`) |
| `data.prompt_variation` | `false` | Randomly vary prompt wording during training |
| `training.max_train_steps` | `5000` | Number of training steps |
| `training.output_dir` | varies | Where to save LoRA weights |
| `training.flux2_conditioning` | `noise_endpoint` | Flux2 only: `noise_endpoint` (blend EM latents with noise) or `concatenate` (Kontext-style sequence concat) |
| `training.flux2_noise_mix` | `0.5` | Flux2 noise_endpoint only: blend between EM latents (0) and pure noise (1) |

### Preview normalization

Inspect per-dataset intensity normalization before training:

```bash
# Preview with current norms.csv
pixi run python scripts/preview_crop_norms.py --output-dir runs/norm_previews

# Preview with auto-computed norms (also saves auto_norms.csv)
pixi run python scripts/preview_crop_norms.py --auto-norms --output-dir runs/norm_previews
```

Each preview shows a side-by-side comparison: auto-stretched (full range) vs. the applied normalization, annotated with dataset name, norm params, and raw intensity stats.

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
  agents/
    gen_backend.py # Pluggable image generation backends (Flux, etc.)
    llm_backend.py # Pluggable LLM/VLM backends (ollama, Anthropic, Google, OpenAI)
    evaluator.py   # Combined critic+refiner agent
    loop.py        # Generate-evaluate-refine orchestrator
    schemas.py     # Dataclasses for structured data exchange
  training/
    dataset.py     # CellMapFluxDataset: zarr-backed training data
    zarr_utils.py  # Zarr reading utilities (adapted from sam3m)
    train.py       # LoRA training loop with accelerate + PEFT
scripts/
  preview_crop_norms.py # Preview and auto-compute intensity normalization per dataset
configs/
  train_lora.yaml                  # Kontext training configuration
  train_lora_flux2.yaml            # Flux2-dev training configuration
  train_lora_flux2_r64_autonorm.yaml           # Flux2-dev with auto-norms, rank 64
  train_lora_flux2_r64_autonorm_augmented.yaml # Flux2-dev with auto-norms + all augmentations
  norms.csv                                    # Per-dataset intensity normalization (manual)
```
