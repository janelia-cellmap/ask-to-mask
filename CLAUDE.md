# ask-to-mask

## Project overview

Python tool that generates organelle segmentation masks from EM images using VLMs and SAM3. Supports multiple backends (Flux, SAM3, Gemini, Qwen), agentic refinement, zarr I/O, and orthogonal plane majority-vote merging. Uses pixi for environment management.

## Build and run

```bash
pixi install
pixi run install-torch-cu126
pixi run segment segment --input image.png --output-dir ./masks/ --organelles mito --save-colored
pixi run segment list-organelles

# For Molmo2 point detection (separate env with transformers <5)
pixi install -e molmo
pixi run -e molmo install-torch-cu126
```

## Architecture

- `src/ask_to_mask/config.py` — Organelle class definitions (name, color, RGB, prompt template) and model registry
- `src/ask_to_mask/model.py` — Loads Flux pipelines via HuggingFace diffusers (supports Kontext Dev and Flux 2 Dev)
- `src/ask_to_mask/postprocess.py` — Extracts binary masks from color difference between input/output images
- `src/ask_to_mask/pipeline.py` — Orchestrates the full flow: load image → resize → infer → extract mask → save
- `src/ask_to_mask/cli.py` — argparse CLI with `segment`, `refine`, `train`, and `list-organelles` subcommands
- `src/ask_to_mask/agents/gen_backend.py` — Pluggable image generation backends (Flux, Gemini, GLM, Qwen, SAM3)
- `src/ask_to_mask/zarr_io.py` — Zarr I/O: load slices/z-stacks/orthogonal planes from zarr volumes, save OME-NGFF zarr
- `src/ask_to_mask/agents/sam3_backend.py` — SAM3 segmentation backend with text, VLM-coordinate, painted-marker strategies, and video predictor for z-stacks
- `src/ask_to_mask/agents/marker_detection.py` — Colored marker detection for SAM3 painted-marker strategy
- `src/ask_to_mask/agents/evaluator.py` — VLM evaluator with point-based refinement for SAM3 and batch Molmo point detection (Molmo runs as subprocess in separate pixi env)
- `scripts/molmo_points.py` — Standalone Molmo2 inference script with single and batch modes (runs in `molmo` pixi env with transformers <5)
- `src/ask_to_mask/agents/loop.py` — Generate-evaluate-refine orchestrator (supports pre-detected points)
- `src/ask_to_mask/agents/zstack.py` — Z-stack orchestrator: multi-slice refinement, orthogonal plane majority vote

## Conventions

- Always keep README.md up to date when adding/removing code or features
- Use pixi for environment management, PyTorch installed via `pixi run install-torch-cu126`
- Organelle colors must be maximally saturated pure-channel colors for clean mask extraction
