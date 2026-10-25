# ask-to-mask

Generate organelle segmentation masks from EM images using vision-language models and SAM3.

Supports multiple backends — Flux image editing, SAM3 with VLM-guided point/box prompts (including Molmo and Antigravity), Gemini, GLM, and Qwen — with an agentic refinement loop that iteratively improves masks using VLM feedback. Reads directly from zarr volumes and can process orthogonal planes (XY, XZ, YZ) with majority-vote merging for robust 3D segmentation.

Also includes a zarr crop-to-video tool, LoRA finetuning for Flux on CellMap data, and a separate supervised 2.5D mitochondria model.

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

When `--resolution` is provided (in nm/pixel), the prompt states the image resolution so the model can reason about scale. Add `--detailed-prompt` to also include an EM-specific description of the organelle's appearance.

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

# Save 3D mask stack as zarr (one group per organelle: stack.zarr/mito/s0, ...)
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
| `--save-zarr` | None | Path to save the 3D mask stack as zarr, one `<organelle>/s0` group per organelle |

### YAML config files

All CLI flags can be set in a YAML config file. CLI flags override config values. The `segment`, `refine`, and `video` subcommands support `--config`.

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
| `--resolution` | None | Image resolution in nm/pixel, stated in the prompt |
| `--detailed-prompt` | off | Include an EM-specific organelle description in the prompt |
| `--num-steps` | `28` | Number of inference steps |
| `--guidance-scale` | `3.5` | How strongly to follow the prompt |
| `--strength` | `0.75` | Edit strength (0=no change, 1=full regeneration) |
| `--threshold` | `30.0` | Mask extraction threshold (0-255): minimum target-color dominance in semantic mode, minimum saturation in instance mode |
| `--instance` | off | Instance segmentation mode (each instance a different color) |
| `--seed` | None | Random seed for reproducibility |
| `--custom-prompt` | None | Override the default prompt |
| `--device` | `cuda` | Torch device |

## How it works

1. Load EM image, convert to RGB, pad to square, and resize to 1024x1024
2. Construct a prompt from the organelle config (e.g., *"This is an EM image of cell(s). Create a segmentation mask: color all the mitochondria in bright red and make everything else black."*). `--resolution` adds the nm/px scale and `--detailed-prompt` adds an EM appearance description.
3. Run the Flux model (locally via HuggingFace diffusers) to produce a colored version of the image
4. Extract a binary mask by thresholding how strongly each pixel is dominated by the target color (semantic mode), or detect any saturated pixels and label connected components (instance mode)
5. Crop off the padding, resize the mask back to the original image dimensions, and save

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
| `cell` | Whole cells | Red | 5-50 μm |

## Agentic refinement

Iteratively improve mask quality using a VLM evaluator agent. Each iteration generates a mask, sends it (alongside the original EM image) to a vision-language model for critique, and uses the feedback to refine the prompt. Numeric generation parameters are tuned separately, by a small one-at-a-time hill-climb on the evaluator score.

### Setup

```bash
# Gemini evaluator (default --llm-provider google), Gemini image backend, and Veo
pixi run pip install -e '.[agents-google]'

# Other LLM providers (each extra also installs the ollama client):
pixi run pip install -e '.[agents-anthropic]'  # Claude
pixi run pip install -e '.[agents-openai]'     # GPT-4o
```

The `antigravity` provider needs no Python extra — it shells out to the local `agy` CLI, which must be on your `PATH` and logged in.

The `google` provider uses `GOOGLE_API_KEY` by default; pass `--vertex-ai --gcp-project <id>` (or set `GOOGLE_CLOUD_PROJECT`) to use Vertex AI with application-default credentials instead.

The ollama backend starts `ollama serve` itself if no server is running. Pull a vision model first; the default is `gemma3:27b`:

```bash
ollama pull gemma3:27b
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
    --llm-provider anthropic --llm-model claude-sonnet-5-5

# Customize loop parameters
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
    --max-iterations 3 --min-score 0.9 --guidance-scale 4.0

# Use Flux2 with LoRA weights
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
    --model flux2-dev --lora checkpoints/flux2-lora

# Gemini image editing backend (gemini-3-pro-image via Vertex AI), Antigravity evaluator
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend gemini --vertex-ai \
  --llm-provider antigravity --llm-model "Gemini 3.5 Flash (High)"

# Use Qwen image editing backend
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend qwen --model Qwen/Qwen-Image-Edit-2511

# Qwen with explicit defaults (and you can adjust as needed)
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend qwen --model Qwen/Qwen-Image-Edit-2511 \
  --num-steps 40 --guidance-scale 1.0 --true-cfg-scale 4.0
```

Each iteration saves `colored.png`, `mask.png`, `composite.png`, `overlay.png` (the binarized mask alpha-blended onto the raw EM image, for visual QC — sparse false positives that are easy to miss against `mask.png`'s black background show up clearly here), and `evaluation.json` to `iteration_00/`, `iteration_01/`, etc. The loop stops when the evaluator score reaches `--min-score` or `--max-iterations` is reached; with `--max-iterations 1` (or `--skip-refinement`) the mask is generated once and not evaluated.

All `refine` outputs go into a timestamped run directory, `<output-dir>/<gen-model>[_<point-model>]_<eval-model>/<organelle>/<YYYYmmdd_HHMMSS>/`, not into `--output-dir` itself.

| Flag | Default | Description |
|------|---------|-------------|
| `--gen-backend` | `flux` | Mask generator: `flux`, `gemini`, `glm`, `qwen`, `sam3` |
| `--llm-provider` | `google` | Evaluator VLM: `ollama`, `anthropic`, `google`, `antigravity`, `openai`, `huggingface` |
| `--llm-model` | provider default | Evaluator model name |
| `--max-iterations` | `5` | Maximum refinement iterations |
| `--min-score` | `0.95` | Stop once the evaluator score reaches this |
| `--mask-mode` | `overlay` | Generative backends: `overlay` (color on EM), `direct` (white-on-black mask), `invert` (segment boundaries, then invert) |
| `--threshold` | `30.0` | Mask extraction threshold (0-255) |
| `--vertex-ai` | off | Use Vertex AI (ADC) instead of an API key for Google models |
| `--gcp-project` / `--gcp-location` | None / `us-central1` | Vertex AI project and location (Gemini image models, including `--marker-backend gemini`, switch to `global` automatically) |

### Architecture

Both the image generation model and the evaluator VLM are pluggable:

- **Image gen backends** (`--gen-backend`): `flux` (default), `gemini`, `glm`, `qwen`, `sam3`.
- **LLM backends** (`--llm-provider`): `ollama`, `anthropic`, `google`, `antigravity`, `openai`, `huggingface`.

GLM backend note: `GlmImagePipeline` is not in released `diffusers` (0.36.0) yet, so `--gen-backend glm` needs a source build (Qwen's `QwenImageEditPlusPipeline` works on 0.36.0):

```bash
pixi run pip install --upgrade "git+https://github.com/huggingface/diffusers"
```

### SAM3 backend

Use SAM3 (Segment Anything Model 3) for precise segmentation with VLM-guided prompting. Instead of a generative model painting colored overlays, SAM3 directly segments organelles using point or text prompts.

#### Setup

```bash
pixi run pip install -e '.[sam3]'
# SAM3 checkpoints require HuggingFace auth:
huggingface-cli login
```

#### Strategies

SAM3 supports three prompting strategies via `--sam3-strategy`:

- **text** (default): Uses SAM3's open-vocabulary text prompts (e.g. "mitochondria"). The loop cycles through synonym prompts per iteration.
- **vlm-coordinate**: A VLM examines the EM image and provides (x, y) points inside each organelle instance. It can also return bounding boxes and polygon outlines with `--point-shape-mode`. Points with the same `instance` ID are fed to SAM3 together for one object; the first box per instance is passed as a SAM3 box prompt; polygons are not sent to SAM3, only recorded in `evaluation.json` and drawn in `points.png`. The evaluator VLM then reviews the result and suggests adding/removing points across refinement iterations.
- **painted-marker**: A secondary generative model paints dot markers at organelle centers, which are detected via color difference and converted to SAM3 point prompts (one point per instance). The marker-painting prompt is fixed; the evaluator's refined prompt only affects the text-prompt fallback.

```bash
# Text mode — simplest, uses organelle name as SAM3 text prompt
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy text --llm-provider google

# VLM-coordinate mode — VLM provides point coordinates for SAM3
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy vlm-coordinate --llm-provider anthropic

# Antigravity for both point detection and evaluation via the local agy CLI
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy vlm-coordinate \
  --point-provider antigravity --point-model "Gemini 3.5 Flash (High)" \
  --llm-provider antigravity --llm-model "Gemini 3.5 Flash (High)" \
  --point-shape-mode all

# Painted-marker mode — gen model paints dots, detected and fed to SAM3
pixi run segment refine --input image.png --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy painted-marker --marker-backend gemini --llm-provider google
```

| Flag | Default | Description |
|------|---------|-------------|
| `--sam3-strategy` | `text` | Prompting strategy: `text`, `vlm-coordinate`, `painted-marker` |
| `--sam3-model` | `facebook/sam3` | `facebook/sam3` downloads from HuggingFace; any other value is a local checkpoint path |
| `--sam3-confidence` | `0.5` | Confidence threshold for text-prompt masks (point prompts keep SAM3's best-scoring mask) |
| `--marker-backend` | None | Gen backend for painting markers (required for `painted-marker`) |
| `--point-shape-mode` | `points` | VLM geometry request: `points`, `boxes`, `polygons`, or `all`; boxes can guide SAM3, polygons are recorded only (ignored for Molmo) |
| `--point-temperature` | `0.0` | Sampling temperature for the point-detection VLM call (not applied by `antigravity` or `anthropic`) |
| `--validate-points` | off | Ask the evaluator VLM to confirm each detected point lies on the target organelle |

The VLM-coordinate strategy also supports Antigravity (`--point-provider antigravity`) via the local `agy` CLI, plus a HuggingFace backend (`--llm-provider huggingface`) for local VLM inference, including Molmo models which use native pointing output rather than JSON coordinates. With `antigravity`, pass a model explicitly: the `agy` CLI's own default model can be unavailable in enterprise deployments (at Janelia, `Gemini 3.5 Flash (High)` works and gave the best point detection).

Crops smaller than 768px on their shortest side are upsampled (Lanczos) before being sent for point detection, so a VLM's own internal image resize doesn't coarsen the coordinate grid it reasons over; returned coordinates are mapped back to the original crop size automatically. Initial point detection also asks each backend for coordinates in the scheme it was grounded on: pixel coordinates by default, or 0-1000 normalized coordinates for Gemini-backed providers (`google` and `antigravity`, whose `agy` CLI is Gemini 3.x under the hood). Molmo gets a plain-language pointing prompt and its native point output is parsed directly. The evaluator's add/remove-point suggestions during refinement are in pixel coordinates of the original crop for every provider.

Each iteration writes the same files as the generative backends (`colored.png`, `mask.png`, `composite.png`, `overlay.png`, `evaluation.json`). For point-prompted masks (`vlm-coordinate`, `painted-marker`) or with `--instance`, `mask.png` is a uint16 instance label image and `colored.png` overlays each instance in its own color; for `text` it is a binary mask with a single-color overlay. `points.png` (point/box/polygon prompts drawn on the EM) is written for `vlm-coordinate` only.

### Google media generation scripts

For programmatic image or movie generation through Google's GenAI SDK, use the standalone script rather than the segmentation loop:

```bash
# Image generation or editing through Vertex/Agent Platform ADC
pixi run google-media --gcp-project "$GOOGLE_CLOUD_PROJECT" image \
  --prompt "Create a clean EM-style mitochondria illustration" \
  --output outputs/google_media/mito.png

# Video generation writes to GCS
pixi run google-media --gcp-project "$GOOGLE_CLOUD_PROJECT" video \
  --prompt "A short scientific animation of mitochondria segmentation in EM data" \
  --output-gcs-uri gs://your-bucket/ask-to-mask/veo/
```

### Crop-to-video

Crop a zarr volume (by world-coordinate center + voxel counts, or an explicit ROI) into a real mp4 built directly from the raw EM slices, and optionally ask Gemini Veo to generate a stylized companion clip.

**Important:** Veo is a *generative* video model, not a frame-accurate video editor — its API only supports text/image-conditioned generation (`image=`, `last_frame=`, up to 3 `reference_images=`) and extending a Veo-generated video, with no way to take a real input video and get back the same content with an accurate segmentation overlay. The Veo output is an illustrative, stylized clip loosely grounded on one reference frame — not a scientifically accurate segmentation of the cropped volume. For accurate per-slice segmentation, use `refine` (above) and assemble its saved PNGs into a video yourself if needed.

```bash
# Crop by center + voxel counts, write a 5s raw video (no Veo call)
pixi run segment video \
  --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --center "50000,50000,50000" --xy-voxels 512 --z-voxels 60 \
  --duration-seconds 5 --output-dir ./outputs/crop_video/

# Same crop, also generate a stylized Veo video (requires a GCS bucket)
pixi run segment video \
  --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --center "50000,50000,50000" --xy-voxels 512 --z-voxels 60 \
  --organelle mito --generate-veo \
  --veo-output-gcs-uri gs://your-bucket/ask-to-mask/veo/ \
  --output-dir ./outputs/crop_video/

# Or drive it all from a config file (see configs/crop_video_example.yaml)
pixi run segment video --config configs/crop_video_example.yaml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--center` | None | Crop center in world coords (nm), `"z,y,x"`. Alternative to `--roi`. |
| `--roi` | None | Explicit ROI in world coords (nm). Overrides `--center` and the voxel counts |
| `--xy-voxels` | `512` | Crop extent in Y and X, in voxels |
| `--y-voxels` / `--x-voxels` | None | Per-axis extents that override `--xy-voxels`, for non-square (e.g. 16:9) crops |
| `--z-voxels` | `60` | Crop extent in Z, in voxels |
| `--z-step-nm` | None | Subsample Z within the crop (default: every voxel) |
| `--duration-seconds` | `5.0` | Length of the raw-input mp4; fps is derived from slice count / duration |
| `--upsample` | `1.0` | Lanczos-upsample each frame by this factor before encoding |
| `--generate-veo` | off | Also request a stylized Veo video |
| `--veo-model` | `veo-3.1-fast-generate-001` | Veo model name |
| `--veo-prompt` | built from `--organelle` | Custom Veo prompt |
| `--veo-output-gcs-uri` | None | GCS URI prefix for Veo's output (required by Vertex AI when `--generate-veo`) |
| `--veo-duration-seconds` | same as `--duration-seconds` | Snapped to Veo's nearest supported length: 4, 6, or 8 seconds |
| `--veo-aspect-ratio` | `16:9` | Veo output aspect ratio |
| `--veo-reference-frame` | `middle` | Which crop slice conditions Veo's first frame: `first`, `middle`, `last`, `none` |
| `--gcp-project` / `--gcp-location` | `$GOOGLE_CLOUD_PROJECT` / `global` | Vertex AI project and location for Veo |

`--generate-veo` needs the `agents-google` extra (`google-genai`) and Vertex AI application-default credentials.

Output: `raw_input.mp4` (always), and with `--generate-veo`: `veo_prompt.txt`, `veo_video_uri.txt` (the `gs://` URI), and `veo_video.mp4` if `google-cloud-storage` is installed for the download (otherwise copy it manually with `gsutil cp`).

### Z-stack refinement with zarr

Refine segmentation across multiple z-slices from a zarr volume. The SAM3 video predictor can propagate masks across slices for consistent 3D segmentation.

```bash
# Independent per-slice refinement (any backend)
pixi run segment refine --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --z-count 5 --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy text --llm-provider google

# SAM3 video predictor: text-prompt the middle slice, propagate across slices
pixi run segment refine --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --z-count 10 --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --use-video-predictor

# Per-slice Molmo point detection + video predictor propagation
pixi run segment refine --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --z-count 10 --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --sam3-strategy vlm-coordinate --use-video-predictor \
  --multi-slice-points --point-provider huggingface --point-model allenai/Molmo2-8B

# Save 3D mask stack as zarr (written to ./refined/masks.zarr/masks/s0)
pixi run segment refine --zarr-path /path/to/volume.zarr/recon-1/em/fibsem-uint8/s0 \
  --z-start 100 --z-count 5 --output-dir ./refined/ --organelle mito \
  --gen-backend sam3 --use-video-predictor --save-zarr ./refined/masks.zarr
```

| Flag | Default | Description |
|------|---------|-------------|
| `--use-video-predictor` | off | Use SAM3 video predictor for cross-slice mask propagation (SAM3 only; no evaluation/refinement loop) |
| `--multi-slice-points` | off | Run point detection independently on each slice (SAM3 only) |
| `--point-sample` | all | Number of slices to sample for point detection |
| `--point-provider` | same as `--llm-provider` | VLM provider for point detection (e.g. `antigravity` or `huggingface` for Molmo) |
| `--point-model` | None | VLM model for point detection (e.g. `allenai/Molmo2-8B`) |
| `--point-prompt` | built in | Custom point-detection prompt, used verbatim (`{organelle}` is not substituted) |
| `--point-shape-mode` | `points` | Ask for `points`, `boxes`, `polygons`, or `all` |
| `--skip-refinement` | off | Skip iterative evaluation/refinement loop — just detect points and run SAM3 once |

The video predictor always seeds with a text prompt (the organelle name) on the middle slice and propagates forward and backward. Without `--multi-slice-points` no VLM point detection runs, so `--sam3-strategy` doesn't matter in this mode. With `--multi-slice-points`, the point provider then runs on each slice and those points are added as refinement clicks on that slice's first detected object, before propagating again. Video-predictor runs skip evaluation, so per-slice scores are reported as 0.

Batch Molmo detection: with `--multi-slice-points` and Molmo, all slices go through a single subprocess call (loading the model once). If the batch call fails or times out, it falls back to one subprocess per slice.

### Orthogonal plane segmentation

Process all 3 orthogonal planes (XY, XZ, YZ) from a zarr ROI and merge via majority vote for more robust 3D segmentation. A voxel is marked foreground if at least 2 of 3 planes agree. `--ortho` requires `--roi`; without one it falls back to the normal z-stack path. XZ and YZ are sampled at the XY slice count and gaps are filled from the nearest processed slice.

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
| `--parallel-points` | off | Run Molmo detection for all 3 planes concurrently (~3x VRAM; needs `--multi-slice-points` and a Molmo `--point-model`) |

Output structure with `--ortho`:
- `xy/`, `xz/`, `yz/` — per-plane slice results
- `masks.zarr/xy/s0`, `masks.zarr/xz/s0`, `masks.zarr/yz/s0` — per-plane 3D masks
- `masks.zarr/merged/s0` — majority-vote merged mask
- `merged/` — PNG slices of the merged mask
- `ortho_summary.json` — processing metadata

Zarr output uses OME-NGFF v0.4 multiscale format with voxel size and offset metadata, compatible with neuroglancer and other zarr viewers. In ortho mode `masks.zarr` is always written in the run directory and `--save-zarr` is ignored.

### Run metadata

Every `refine` run saves reproducibility metadata to its timestamped run directory (`video` saves the same files to `--output-dir`):
- `config.yaml` — copy of the YAML config file (if `--config` was used)
- `args.json` — all resolved CLI arguments (CLI flags merged with config)
- `command.txt` — the original command line (`sys.argv` joined with spaces, not shell-quoted)


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

Each run writes to a timestamped directory, `<training.output_dir>/<YYYYmmdd_HHMMSS>/`. Periodic checkpoints go to `checkpoint-<step>/` inside it, and the final LoRA weights to the run directory itself; either can be used for inference with `--lora`. `train --resume` is not implemented yet: it logs a warning and starts a fresh run.

To sanity-check a config's (EM, target) pairs before spending GPU time on training (currently only for `data.dataset_type: inference_mito_fixed_fov` configs):

```bash
pixi run segment preview-train --config configs/train_lora_flux2_mito_fixed_fov.yaml \
  --output-dir runs/train_preview --num-samples 8
```

On the Janelia cluster, `scripts/submit_lora_lsf.py` submits a training run to LSF with GPU-queue defaults.

### Training config options

Start from `configs/train_lora.yaml` (Kontext) or `configs/train_lora_flux2.yaml` (Flux2). Keys not set in those files fall back to the code defaults below; the `train_lora_flux2_r64_*` configs show the auto-norm and augmentation keys in use.

| Key | Default | Description |
|-----|---------|-------------|
| `model.pretrained` | required | Flux model to finetune (`black-forest-labs/FLUX.1-Kontext-dev` or `black-forest-labs/FLUX.2-dev`) |
| `model.lora.rank` / `model.lora.alpha` | `16` / `16` | LoRA rank and alpha |
| `data.dataset_type` | `groundtruth_crops` | Data source; `inference_mito_fixed_fov` and `mixed_mito_gt_pseudo` are the fixed-FOV mito datasets |
| `data.organelles` | all with fine-class labels | Which organelles to train on (everything except `nucleolus` and `cell`; ignored by the fixed-FOV mito datasets) |
| `data.data_root` | `/nrs/cellmap/data` | Path to CellMap zarr data |
| `data.samples_per_epoch` | `2000` | Random crops drawn per epoch |
| `data.target_mode` | `overlay` | `overlay` = EM + colored organelles, `segmentation` = colored mask on black background |
| `data.norms_csv` | None | Per-dataset intensity normalization table (e.g. `configs/norms.csv`) |
| `data.auto_norms` | `false` | Compute per-dataset intensity normalization from data percentiles instead of using `norms_csv` |
| `data.auto_norms_per_image` | `false` | Compute auto-norms per crop instead of per dataset |
| `data.auto_norms_percentile_low` | `1.0` | Lower percentile for auto-norm clipping |
| `data.auto_norms_percentile_high` | `99.0` | Upper percentile for auto-norm clipping |
| `data.include_resolution` | `false` | Include image resolution (nm/px) in training prompts (both example configs set `true`) |
| `data.multi_organelle_prob` | `0.0` | Probability of coloring 2-3 organelles per image (e.g. `0.3`) |
| `data.negative_example_prob` | `0.0` | Probability of serving a no-organelle example where target = input (e.g. `0.15`) |
| `data.prompt_variation` | `false` | Randomly vary prompt wording during training |
| `training.max_train_steps` | `5000` | Number of training steps |
| `training.checkpointing_steps` | `500` | Save a checkpoint every N steps |
| `training.output_dir` | `runs/flux-lora` | Parent directory for timestamped run directories |
| `training.gradient_checkpointing` | `true` | Trade compute for activation memory |
| `training.use_8bit_adam` | `true` | bitsandbytes `AdamW8bit` (falls back to `torch.optim.AdamW` with a warning if bitsandbytes is missing) |
| `training.flux2_conditioning` | `noise_endpoint` | Flux2 only: `noise_endpoint` (blend EM latents with noise) or `concatenate` (Kontext-style sequence concat) |
| `training.flux2_noise_mix` | `0.5` | Flux2 noise_endpoint only: blend between EM latents (0) and pure noise (1). Every Flux2 config in the repo sets `0.0` |

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

Needs an 80 GB-class GPU (A100 80GB, H100, or H200); `scripts/submit_lora_lsf.py` targets H100 (default) or H200 on the Janelia cluster. Gradient checkpointing and 8-bit Adam are on by default to fit in memory.

## 2.5D discriminative mitochondria model

A separate, non-generative track: a supervised ConvNeXt-UNet trained on a 9-slice EM z-stack to directly predict the center-slice mitochondria mask (no prompts, no image generation). Validation scores it against GT, with the existing UNet inference pipeline's segmentations scored on the same GT as a baseline. See `docs/mito_2p5d_training_handoff.md` for the full design writeup.

```bash
# Optional: self-supervised masked-image-modeling pretraining on raw, unlabeled
# EM data (no GT needed). To use the result, set model.encoder_checkpoint in the
# supervised config to <pretrain run>/best/encoder.pt (it is commented out by default).
pixi run pretrain-mito-2p5d --config configs/pretrain_mito_2p5d_mae.yaml

# Supervised training: mixed pseudo-label + GT curriculum
pixi run train-mito-2p5d --config configs/train_mito_2p5d_fixed_fov.yaml

# GT-only control run (no pseudo-labels, no curriculum)
pixi run train-mito-2p5d --config configs/train_mito_2p5d_gt_only.yaml
```

Resume either with `--resume <output_dir>/<timestamp>/checkpoint-<step>`.

## Project structure

```
src/ask_to_mask/
  cli.py           # CLI entry point (segment, refine, video, list-organelles, train, preview-train, train-mito-2p5d, pretrain-mito-2p5d)
  config.py        # Organelle class definitions and model registry
  model.py         # Flux model loading and inference (with LoRA support)
  pipeline.py      # Orchestrates load → prompt → infer → postprocess
  postprocess.py   # Mask extraction (semantic + instance)
  zarr_io.py       # Zarr I/O: load slices, z-stacks, orthogonal planes, center+voxel-count crops, and save OME-NGFF zarr
  video_io.py      # Encode PIL frame sequences into mp4
  agents/
    gen_backend.py      # Pluggable image generation backends (Flux, Gemini, GLM, Qwen; SAM3 via sam3_backend)
    sam3_backend.py     # SAM3 segmentation backend (text, VLM-coordinate, painted-marker, video predictor)
    marker_detection.py # Colored marker detection for SAM3 painted-marker strategy
    llm_backend.py      # Pluggable LLM/VLM backends (ollama, Anthropic, Google, Antigravity, OpenAI, HuggingFace)
    evaluator.py        # Combined critic+refiner agent (with SAM3 point refinement + per-slice Molmo)
    loop.py             # Generate-evaluate-refine orchestrator
    zstack.py           # Z-stack orchestrator: multi-slice refinement, orthogonal plane majority vote
    video_backend.py    # Gemini Veo backend for stylized (non-frame-accurate) video from a crop
    schemas.py          # Dataclasses for structured data exchange
  training/
    dataset.py                    # CellMapFluxDataset: zarr-backed training data
    zarr_utils.py                 # Zarr reading utilities (adapted from sam3m)
    train.py                      # LoRA training loop with accelerate + PEFT
    inference_mito_dataset.py     # Fixed-FOV pseudo-label dataset backed by UNet inference segmentations
    mito_2p5d_dataset.py          # 2.5D supervised datasets: pseudo/GT/mixed, comparison-mask baseline
    mito_2p5d_model.py            # ConvNeXt-UNet (discriminative) + ConvNeXtMaskedAutoencoder (self-supervised)
    mito_2p5d_losses.py           # Composite BCE/Dice/Tversky/boundary/SDF loss
    mito_2p5d_metrics.py          # Pixel/boundary/object metrics + baseline comparison
    mito_2p5d_pretrain_dataset.py # Label-free multi-scale EM dataset for masked-image-modeling
    train_mito_2p5d.py            # Supervised 2.5D trainer (pseudo/GT curriculum)
    train_mito_2p5d_pretrain.py   # Self-supervised masked-image-modeling pretrainer
    prefetch.py                   # CUDA-stream batch prefetcher
    worker_seeding.py             # DataLoader worker_init_fn for per-worker RNG diversity
scripts/
  molmo_points.py               # Standalone Molmo2 inference script (runs in molmo pixi env)
  google_media.py               # Standalone Google GenAI image/video generation (pixi run google-media)
  ortho_grid.py                 # Grid image or scroll-through video of orthogonal EM slices from a zarr
  crop_cellmap_test_manifest.py # Crop CellMap test-manifest EM regions to zarr + PNG frames (pixi run crop-cellmap-test)
  preview_crop_norms.py         # Preview and auto-compute intensity normalization per dataset
  build_mito_inference_index.py # Build the fixed-FOV mito inference dataset path index
  submit_lora_lsf.py            # Submit Flux LoRA training to Janelia LSF
  train_mito_2p5d.py            # Script entry point for supervised 2.5D training
  train_mito_2p5d_pretrain.py   # Script entry point for self-supervised pretraining
configs/
  refine_ortho_example.yaml                     # Example config for orthogonal plane refinement
  refine_zarr_example.yaml                      # Example config for z-stack zarr refinement
  crop_video_example.yaml                       # Example config for the video subcommand
  train_lora.yaml                               # Kontext training configuration
  train_lora_flux2.yaml                         # Flux2-dev training configuration
  train_lora_flux2_r64*.yaml                    # Flux2-dev rank-64 variants (cosine LR, auto-norms, per-image norms, augmentations)
  train_lora_flux2_mito_*.yaml                  # Flux2-dev mito-only runs (fixed FOV, 8 nm 1024 px, mixed GT, GT finetune)
  train_mito_2p5d_fixed_fov.yaml                # Supervised 2.5D: mixed pseudo+GT, curriculum
  train_mito_2p5d_gt_only.yaml                  # Supervised 2.5D: GT-only control run
  pretrain_mito_2p5d_mae.yaml                   # Self-supervised masked-image-modeling pretraining
  mito_inference_*_index.json                   # Path indexes for the fixed-FOV mito inference datasets
  norms.csv                                     # Per-dataset intensity normalization (manual)
docs/
  mito_2p5d_training_handoff.md # Design writeup for the 2.5D mito model
  mito_3d_architecture_plan.md  # Plan for a 3D mito model (implemented on branch feature/mito-mask-diffusion-plan)
```
