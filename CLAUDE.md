# ask-to-mask

## Project overview

Python tool that uses Flux image editing models to generate organelle segmentation masks from EM images. Uses pixi for environment management.

## Build and run

```bash
pixi install
pixi run install-torch-cu126
pixi run segment segment --input image.png --output-dir ./masks/ --organelles mito --save-colored
pixi run segment list-organelles
```

## Architecture

- `src/ask_to_mask/config.py` — Organelle class definitions (name, color, RGB, prompt template) and model registry
- `src/ask_to_mask/model.py` — Loads Flux pipelines via HuggingFace diffusers (supports Kontext Dev and Flux 2 Dev)
- `src/ask_to_mask/postprocess.py` — Extracts binary masks from color difference between input/output images
- `src/ask_to_mask/pipeline.py` — Orchestrates the full flow: load image → resize → infer → extract mask → save
- `src/ask_to_mask/cli.py` — argparse CLI with `segment` and `list-organelles` subcommands

## Conventions

- Always keep README.md up to date when adding/removing code or features
- Use pixi for environment management, PyTorch installed via `pixi run install-torch-cu126`
- Organelle colors must be maximally saturated pure-channel colors for clean mask extraction
