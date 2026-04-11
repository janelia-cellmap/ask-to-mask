# ask-to-mask

Generate organelle segmentation masks from EM images using vision-language models and SAM3.

Supports multiple backends — Flux image editing, SAM3 with VLM-guided point prompts (including Molmo), Gemini, and Qwen — with an agentic refinement loop that iteratively improves masks using VLM feedback. Reads directly from zarr volumes and can process orthogonal planes (XY, XZ, YZ) with majority-vote merging for robust 3D segmentation.

## Setup

Requires [pixi](https://pixi.sh).

```bash
pixi install
pixi run install-torch-cu126

# For Molmo2 point detection (separate env with transformers <5)
pixi install -e molmo
pixi run -e molmo install-torch-cu126
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

### Segment from zarr volumes

Read EM slices directly from zarr files instead of requiring pre-exported PNGs:

```bash
# Single slice from a zarr volume
pixi run segment segment --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --output-dir ./masks/ --organelles mito --save-colored

# Z-stack: 5 consecutive slices
pixi run segment segment --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --z-count 5 --output-dir ./masks/ --organelles mito

# With sub-path navigation and stride
pixi run segment segment --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8 \
  --dataset-path s0 --z-start 100 --z-count 10 --z-step 2 --output-dir ./masks/ --organelles mito

# Save 3D mask stack as zarr
pixi run segment segment --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --z-count 5 --output-dir ./masks/ --organelles mito --save-zarr ./masks/stack.zarr
```

**ROI in world coordinates (nm)** — crop to a specific region instead of full slices:

```bash
# ROI format: [z_start:z_end, y_start:y_end, x_start:x_end] in nm
pixi run segment segment --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --roi "[4000:5000, 4000:8000, 4000:8000]" --output-dir ./masks/ --organelles mito
```

Uses `funlib.geometry.Roi` under the hood — ROI is snapped to the voxel grid and intersected with available data.

| Flag | Default | Description |
|------|---------|-------------|
| `--zarr-path` | None | Path to zarr volume or group |
| `--dataset-path` | None | Sub-path within zarr (e.g., `s0`) |
| `--z-start` | `0` | First Z slice index |
| `--z-count` | `1` | Number of Z slices to process |
| `--z-step` | `1` | Step between Z slices |
| `--roi` | None | ROI in world coords (nm), e.g. `[500:1000,500:1000,1000:11000]`. Overrides z-start/z-count/z-step |
| `--z-step-nm` | None | Z spacing in nm when using `--roi` (e.g. `40` reads every 40 nm). Default: every voxel |
| `--save-zarr` | None | Path to save 3D mask stack as zarr |

### YAML config files

All CLI flags can be set in a YAML config file. CLI flags override config values. Both `segment` and `refine` subcommands support `--config`.

```bash
pixi run segment refine --config configs/refine_zarr_example.yaml

# Override a config value from CLI
pixi run segment refine --config configs/refine_zarr_example.yaml --organelle er

# Also works with segment
pixi run segment segment --config configs/my_segment_config.yaml
```

See [configs/refine_zarr_example.yaml](configs/refine_zarr_example.yaml) for a full example.

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
# Install agent dependencies (Gemini evaluator backend, default)
pip install -e '.[agents-google]'

# Or install ollama backend instead
pip install -e '.[agents]'

# For other LLM providers:
pip install -e '.[agents-anthropic]'  # Claude
pip install -e '.[agents-openai]'     # GPT-4o
```

The ollama backend requires a running ollama server with a vision model:

```bash
ollama serve
ollama pull llama3.2-vision
```

### Run

```bash
# Basic refinement loop (default evaluator: Gemini)
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito

# Use ollama evaluator instead (free, local)
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --llm-provider ollama --llm-model gemma3:27b

# Use a different LLM provider
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
    --llm-provider anthropic --llm-model claude-sonnet-4-20250514

# Customize loop parameters
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
    --max-iterations 3 --min-score 0.9 --guidance-scale 4.0

# Use Flux2 with LoRA weights
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
    --model flux2-dev --lora checkpoints/flux2-lora

# Use Qwen image editing backend
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend qwen --model Qwen/Qwen-Image-Edit-2511

# Qwen with explicit defaults (and you can adjust as needed)
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend qwen --model Qwen/Qwen-Image-Edit-2511 \
  --num-steps 40 --guidance-scale 1.0 --true-cfg-scale 4.0
```

Each iteration saves `colored.png`, `mask.png`, and `evaluation.json` to `iteration_00/`, `iteration_01/`, etc. The loop stops when the evaluator scores the mask above `--min-score` or `--max-iterations` is reached.

### Architecture

Both the image generation model and the evaluator VLM are pluggable:

- **Image gen backends** (`--gen-backend`): `flux` (default), `gemini`, `glm`, `qwen`, `sam3`.
- **LLM backends** (`--llm-provider`): `ollama`, `anthropic`, `google`, `openai`, `huggingface`.

Qwen backend note: `QwenImageEditPlusPipeline` may require a newer `diffusers` build. If import fails, run:

```bash
pixi run pip install --upgrade "git+https://github.com/huggingface/diffusers"
```

### SAM3 backend

Use SAM3 (Segment Anything Model 3) for precise segmentation with VLM-guided prompting. Instead of a generative model painting colored overlays, SAM3 directly segments organelles using point or text prompts.

#### Setup

```bash
pip install -e '.[sam3]'
# SAM3 checkpoints require HuggingFace auth:
huggingface-cli login
```

#### Strategies

SAM3 supports three prompting strategies via `--sam3-strategy`:

- **text** (default): Uses SAM3's open-vocabulary text prompts (e.g. "mitochondria"). The loop cycles through synonym prompts per iteration.
- **vlm-coordinate**: A VLM examines the EM image and provides (x, y) center coordinates for each organelle instance. Each point is fed to SAM3 independently as its own instance — SAM3 runs one `predict()` call per point and produces a separate mask for each. The evaluator VLM then reviews the result and suggests adding/removing points across refinement iterations.
- **painted-marker**: A secondary generative model paints dot markers at organelle centers, which are detected via color difference and converted to SAM3 point prompts (same one-point-per-instance approach).

```bash
# Text mode — simplest, uses organelle name as SAM3 text prompt
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy text --llm-provider google

# VLM-coordinate mode — VLM provides point coordinates for SAM3
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy vlm-coordinate --llm-provider anthropic

# Painted-marker mode — gen model paints dots, detected and fed to SAM3
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy painted-marker --marker-backend gemini --llm-provider google
```

| Flag | Default | Description |
|------|---------|-------------|
| `--sam3-strategy` | `text` | Prompting strategy: `text`, `vlm-coordinate`, `painted-marker` |
| `--sam3-model` | `facebook/sam3` | SAM3 HuggingFace model ID |
| `--sam3-confidence` | `0.5` | Mask confidence threshold |
| `--marker-backend` | None | Gen backend for painting markers (required for `painted-marker`) |

The VLM-coordinate strategy also supports a HuggingFace backend (`--llm-provider huggingface`) for local VLM inference, including Molmo models which use native pointing output rather than JSON coordinates.

Output intermediates include `colored.png` (semi-transparent overlay of instance masks on the EM image), `mask.png` (color-coded instance labels), `points.png` (point prompts visualized on the EM), and `evaluation.json`.

### Z-stack refinement with zarr

Refine segmentation across multiple z-slices from a zarr volume. The SAM3 video predictor can propagate masks across slices for consistent 3D segmentation.

```bash
# Independent per-slice refinement (any backend)
pixi run segment refine --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --z-count 5 --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy text --llm-provider google

# SAM3 video predictor: propagate masks across slices
pixi run segment refine --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --z-count 10 --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy vlm-coordinate --use-video-predictor \
  --llm-provider google

# Per-slice Molmo point detection + video predictor propagation
pixi run segment refine --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --z-count 10 --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy vlm-coordinate --use-video-predictor \
  --multi-slice-points --llm-provider google

# Save 3D mask stack as zarr
pixi run segment refine --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --z-count 5 --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --use-video-predictor --save-zarr ./refined/masks.zarr
```

| Flag | Default | Description |
|------|---------|-------------|
| `--use-video-predictor` | off | Use SAM3 video predictor for cross-slice mask propagation |
| `--multi-slice-points` | off | Run Molmo independently on each slice to find points |
| `--point-sample` | all | Number of slices to sample for Molmo point detection |
| `--point-provider` | same as `--llm-provider` | VLM provider for point detection (e.g. `huggingface` for Molmo) |
| `--point-model` | None | VLM model for point detection (e.g. `allenai/Molmo2-8B`) |
| `--point-prompt` | `"Point to the {organelle}"` | Custom prompt for Molmo point detection |
| `--skip-refinement` | off | Skip iterative evaluation/refinement loop — just detect points and run SAM3 once |

When `--multi-slice-points` is combined with `--use-video-predictor`, Molmo runs on each slice independently to find organelle locations, then those per-slice points are fed as frame-specific prompts to the SAM3 video predictor. The video predictor propagates masks forward and backward, handling cross-slice consistency and filling in slices where Molmo found nothing.

Batch Molmo detection: when using `--multi-slice-points` with Molmo, all slices are processed in a single subprocess call (loading the model once) rather than spawning a new process per slice.

### Orthogonal plane segmentation

Process all 3 orthogonal planes (XY, XZ, YZ) from a zarr ROI and merge via majority vote for more robust 3D segmentation. A voxel is marked foreground if at least 2 of 3 planes agree.

```bash
# Ortho mode: process XY, XZ, YZ planes and merge
pixi run segment refine --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --roi "[4000:5000, 4000:8000, 4000:8000]" --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy vlm-coordinate --multi-slice-points \
  --point-provider huggingface --point-model allenai/Molmo2-8B \
  --skip-refinement --ortho

# Parallel point detection across planes (~60 GB VRAM needed)
pixi run segment refine --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --roi "[4000:5000, 4000:8000, 4000:8000]" --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy vlm-coordinate --multi-slice-points \
  --point-provider huggingface --point-model allenai/Molmo2-8B \
  --skip-refinement --ortho --parallel-points
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ortho` | off | Process XY, XZ, YZ planes and merge via majority vote |
| `--parallel-points` | off | Run Molmo detection for all 3 planes concurrently (~3x VRAM) |

Output structure with `--ortho`:
- `xy/`, `xz/`, `yz/` — per-plane slice results
- `masks.zarr/xy/s0`, `masks.zarr/xz/s0`, `masks.zarr/yz/s0` — per-plane 3D masks
- `masks.zarr/merged/s0` — majority-vote merged mask
- `merged/` — PNG slices of the merged mask
- `ortho_summary.json` — processing metadata

Zarr output uses OME-NGFF v0.4 multiscale format with voxel size and offset metadata, compatible with neuroglancer and other zarr viewers.

### Run metadata

Every `refine` run automatically saves reproducibility metadata to the output directory:
- `config.yaml` — copy of the YAML config file (if `--config` was used)
- `args.json` — all resolved CLI arguments (CLI flags merged with config)
- `command.txt` — the original command line


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
  zarr_io.py       # Zarr I/O: load slices, z-stacks, orthogonal planes, and save OME-NGFF zarr
  agents/
    gen_backend.py      # Pluggable image generation backends (Flux, Gemini, GLM, Qwen)
    sam3_backend.py     # SAM3 segmentation backend (text, VLM-coordinate, painted-marker, video predictor)
    marker_detection.py # Colored marker detection for SAM3 painted-marker strategy
    llm_backend.py      # Pluggable LLM/VLM backends (ollama, Anthropic, Google, OpenAI, HuggingFace)
    evaluator.py        # Combined critic+refiner agent (with SAM3 point refinement + per-slice Molmo)
    loop.py             # Generate-evaluate-refine orchestrator
    zstack.py           # Z-stack orchestrator: multi-slice refinement, orthogonal plane majority vote
    schemas.py          # Dataclasses for structured data exchange
  training/
    dataset.py     # CellMapFluxDataset: zarr-backed training data
    zarr_utils.py  # Zarr reading utilities (adapted from sam3m)
    train.py       # LoRA training loop with accelerate + PEFT
scripts/
  molmo_points.py       # Standalone Molmo2 inference script (runs in molmo pixi env)
  preview_crop_norms.py # Preview and auto-compute intensity normalization per dataset
configs/
  refine_ortho_example.yaml                    # Example config for orthogonal plane refinement
  refine_zarr_example.yaml                     # Example config for z-stack zarr refinement
  train_lora.yaml                  # Kontext training configuration
  train_lora_flux2.yaml            # Flux2-dev training configuration
  train_lora_flux2_r64_autonorm.yaml           # Flux2-dev with auto-norms, rank 64
  train_lora_flux2_r64_autonorm_augmented.yaml # Flux2-dev with auto-norms + all augmentations
  norms.csv                                    # Per-dataset intensity normalization (manual)
```
