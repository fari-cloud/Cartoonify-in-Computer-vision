# Cartoonify in Computer Vision

Turn photos into cartoon-style images with OpenCV. This project rebuilds the
original notebook (`notebooks/farwa_irfan.ipynb`) as an installable Python
package with a CLI and tests.

## Live demo

Try it in the browser — upload a photo, pick a style:

**[https://fari-cloud.github.io/Cartoonify-in-Computer-vision/](https://fari-cloud.github.io/Cartoonify-in-Computer-vision/)**

The demo is a static page in [`docs/`](docs/) powered by
[OpenCV.js](https://docs.opencv.org/4.x/d5/d10/tutorial_js_root.html): the same
four styles as the Python package, running entirely in the visitor's browser —
no server, no API keys, nothing to install.

## Examples

Input photo (`examples/input.jpg`) rendered in the four available styles:

| Classic | Sketch |
| --- | --- |
| ![Classic cartoon](examples/coffee_classic.jpg) | ![Pencil sketch](examples/coffee_sketch.jpg) |
| **Comic** | **HDR** |
| ![Comic cartoon](examples/coffee_comic.jpg) | ![HDR detail-enhanced](examples/coffee_hdr.jpg) |

## Styles

| Style | Effect |
| --- | --- |
| `classic` | Edge mask over bilateral-filtered colors — the classic OpenCV cartoon effect |
| `sketch` | Pencil-sketch look using the gray-dodge technique |
| `comic` | Vivid comic-book colors with bold ink outlines |
| `hdr` | Detail-enhanced photo look — sharper detail and more vivid colors |

The classic pipeline:

1. **Grayscale + median blur** — smooth noise before edge detection
2. **Adaptive threshold** — detect the outlines of the scene
3. **Bilateral filter** — flatten colors while keeping edges sharp
4. **Combine** — mask the flattened colors with the edge mask

## Install

```bash
pip install -e .
```

Or install the runtime dependencies only:

```bash
pip install -r requirements.txt
```

## Usage

Command line:

```bash
cartoonify photo.jpg -o cartoon.png
```

Or run it as a module:

```bash
python -m cartoonify photo.jpg -o cartoon.png
```

Save intermediate pipeline stages for inspection:

```bash
cartoonify photo.jpg -o cartoon.png --stages-dir stages/
```

Try a different style:

```bash
cartoonify photo.jpg --style sketch
cartoonify photo.jpg --style comic
cartoonify photo.jpg --style hdr
```

Tune the effect:

```bash
cartoonify photo.jpg --block-size 11 --c 7 --diameter 11
```

From Python:

```python
from cartoonify import cartoonify, read_image, save_image

stages = cartoonify(read_image("photo.jpg"), style="comic")
save_image(stages.cartoon, "cartoon.png")
```

CLI options:

| Option | Default | Description |
| --- | --- | --- |
| `input` | — | path to the input image |
| `-o, --output` | `<input>_<style>.png` | output image path |
| `--style` | `classic` | `classic`, `sketch`, `comic` or `hdr` |
| `--stages-dir` | off | directory to save every pipeline stage |
| `--block-size` | 9 | edge detection neighborhood size (odd) |
| `--c` | 9 | constant subtracted from the neighborhood mean |
| `--diameter` | 9 | bilateral filter diameter |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Mistral chat completion example

A minimal chat completion with the official Mistral Python SDK, using
`mistral-small-latest`, lives in `main.py`.

### 1. Generate an API key

1. Sign in at [console.mistral.ai](https://console.mistral.ai) (La Plateforme / Mistral Studio).
2. Go to **API Keys** (or your workspace settings → API Keys).
3. Click **Create new key**, give it a name, and copy the generated key — it is shown only once.

### 2. Install the SDK

```bash
pip install -r requirements.txt
```

### 3. Set the key in your terminal

```bash
export MISTRAL_API_KEY="your-api-key-here"
```

(Windows PowerShell: `$env:MISTRAL_API_KEY="your-api-key-here"`)

### 4. Run it

```bash
python main.py
```

Reference: [Chat completions docs](https://docs.mistral.ai/inference/conversations/chat-completion)
