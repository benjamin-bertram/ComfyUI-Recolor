## Installation

### 1. Copy to ComfyUI

```bash
cd ComfyUI/custom_nodes/
git clone <repo-url> ComfyUI-Product-Recolor
# or just copy the folder
```

### 2. Install dependencies

```bash
cd ComfyUI-Product-Recolor
pip install -r requirements.txt
```

### 3. Required companion nodes (for segmentation)

Install ONE of these for product masking:

**Option A: SAM (Recommended)**
```bash
# ComfyUI-Impact-Pack (includes SAM integration)
cd ComfyUI/custom_nodes/
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack
```

**Option B: GroundingDINO + SAM**
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/IDEA-Research/GroundingDINO
```

**Option C: rembg (simple background removal)**
```bash
pip install rembg onnxruntime
```

### 4. Restart ComfyUI

---

## Nodes

### 🎨 Precise LAB Recolor
Single-zone recoloring. Connect a mask + target RGB → recolored image.

| Input | Type | Description |
|-------|------|-------------|
| `image` | IMAGE | Source product photo |
| `mask` | MASK | Region to recolor (from SAM) |
| `target_r/g/b` | INT | Target RGB values (0–255) |
| `luminance_blend` | FLOAT | 0.0 = keep original brightness, 1.0 = match target exactly. **Recommended: 0.5–0.7** |
| `saturation_boost` | FLOAT | 1.0 = no change, >1 = more vivid |
| `edge_feather` | INT | Blur mask edges (pixels) for smooth blending |

### 🎨 Multi-Zone LAB Recolor
Recolor multiple zones at once. Takes up to 8 masks + a JSON config.

### 🔍 Auto Color Zone Segmenter
K-Means clustering to auto-detect color zones within a product mask. Outputs up to 8 separate zone masks.

### ⚡ Batch Colorway Processor
Process ALL colorways in a single pass. Input JSON array of colorway definitions → batch of recolored images.

### 🎯 RGB Color Input
Parse "R, G, B" strings into individual values.

### 👁️ Color Swatch Preview
Generate color swatch images for visual comparison.

---

## Workflows

### Workflow A: Single-Color Product (e.g., D4T Knit Pant)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────┐
│ Load Image  │────→│  SAM Segment │────→│ 🎨 Precise LAB  │────→│  Save    │
│ (KT4116)    │     │  (pants)     │     │   Recolor       │     │  Image   │
└─────────────┘     └──────────────┘     │                 │     └──────────┘
                                          │ target: 66,17,34│
                                          │ lum_blend: 0.6  │
                                          └─────────────────┘
```

**Steps:**
1. Load source product image
2. SAM: Click on the pants → generates mask
3. Precise LAB Recolor: Set target RGB from colorway spec
4. Save / preview

### Workflow B: Multi-Zone Product (e.g., 3-Stripes T-Shirt)

```
┌────────────┐     ┌──────────────┐
│ Load Image │────→│ SAM Segment  │──→ mask_0 (main body)
│ (JD1906)   │     │ (3 clicks)   │──→ mask_1 (stripes)
└────────────┘     └──────────────┘──→ mask_2 (logo)
                          │
                          ▼
                   ┌──────────────────┐     ┌──────────┐
                   │ 🎨 Multi-Zone    │────→│  Save    │
                   │   LAB Recolor    │     │  Image   │
                   │                  │     └──────────┘
                   │ zone_config JSON │
                   └──────────────────┘
```

**zone_config for JX0732 (Turqoise):**
```json
[
    {"mask_index": 0, "r": 88, "g": 148, "b": 134, "label": "Main"},
    {"mask_index": 1, "r": 4, "g": 0, "b": 0, "label": "3 stripes"},
    {"mask_index": 2, "r": 4, "g": 0, "b": 0, "label": "Logo"}
]
```

### Workflow C: Complex Multi-Zone (e.g., TIRO25C Jacket, 7 zones)

```
┌────────────┐     ┌──────────────┐
│ Load Image │────→│ SAM Segment  │──→ 7 masks (one per zone)
│ (TIRO25C)  │     │ or Auto Zone │
└────────────┘     └──────────────┘
                          │
                          ▼
                   ┌──────────────────┐
                   │ ⚡ Batch Colorway │──→ 4 recolored images
                   │   Processor      │    (JW4388, IW0454,
                   │                  │     JC7027, JC7025)
                   │ colorways JSON   │
                   └──────────────────┘
```

### Workflow D: Auto-Detect Zones (No Manual Masking)

```
┌────────────┐     ┌──────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Load Image │────→│ SAM/rembg│────→│ 🔍 Auto Color   │────→│ 🎨 Multi-Zone    │
│            │     │ (product)│     │   Zone Segmenter │     │   LAB Recolor    │
└────────────┘     └──────────┘     │ num_zones: 3    │     └──────────────────┘
                                     └─────────────────┘
                                     Outputs: zone masks + zone RGB info
```

---

## Recommended SAM Workflow in ComfyUI

For the best segmentation:

1. **SAM Model:** `sam_vit_h_4b8939.pth` (highest quality) or `sam_vit_l_0b3195.pth` (faster)
2. **Prompt Type:** Point prompts (click on the garment)
3. **For multi-zone:** Multiple SAM passes with different point prompts
4. **Alternative:** Use GroundingDINO with text prompts like "pants", "stripes on sleeves", "logo"

### SAM + GroundingDINO Prompt Examples

| Product Zone | GroundingDINO Prompt |
|---|---|
| Pants (full) | `"pants"` or `"trousers"` |
| T-shirt body | `"t-shirt body"` |
| Sleeve stripes | `"stripes on sleeves"` |
| Adidas logo | `"adidas logo"` or `"brand logo"` |
| Jacket main body | `"jacket body"` |
| Side panels | `"side panel"` |

---

## Parameter Tuning Guide

### `luminance_blend` — The Most Important Parameter

| Value | Effect | Use When |
|-------|--------|----------|
| 0.0 | Keep original brightness entirely | Source and target have similar lightness |
| 0.3 | Subtle shift | Light → slightly darker target |
| **0.5–0.7** | **Recommended default** | Most colorway changes |
| 0.8–0.9 | Strong shift | Light → very dark (e.g., white → navy) |
| 1.0 | Match target brightness exactly | Maximum color accuracy, may lose some shadow detail |

### `edge_feather`

| Value | Effect |
|-------|--------|
| 0 | Hard edges (visible if mask isn't perfect) |
| 2–3 | Slight softening (recommended) |
| 5–10 | Smooth blending (for imperfect masks) |

### `saturation_boost`

| Value | Effect |
|-------|--------|
| 0.5 | Muted/desaturated |
| 1.0 | Natural (default) |
| 1.2–1.5 | More vivid (useful for faded source photos) |

---

## Example Colorway Configs

See `examples/colorway_configs.json` for complete configs matching:
- Article #1: D4T TEE (3 zones: main, 3bar, logo)
- Article #2: D4T KNIT PANT (1 zone: main)
- Article #3: M 3S SJ T (3 zones: main, stripes, logo)
- Article #4-B: W ESS 3S TS (3 zones: main, stripes, logo)
- Article #5: TIRO25C AW JKTW (7 zones: main, stripes, logo, sleeve, panels, piping)

---

## Standalone Testing

Test the recoloring without ComfyUI:

```bash
# Single color
python test_recolor.py --image product.png --target_rgb 66,17,34

# Multi-zone
python test_recolor.py --image tshirt.png --zones 3 --target_rgbs "88,148,134;4,0,0;4,0,0"

# Batch from config
python test_recolor.py --image pant.png --config examples/colorway_configs.json \
    --article article_selection_2_d4t_knit_pant

# Just analyze zones
python test_recolor.py --image jacket.png --zones 7
```

---

## API / Automation Approach

For high-volume processing (100+ colorways), consider:

1. **ComfyUI API mode** — Script the workflow via HTTP API
2. **Standalone Python** — Use `test_recolor.py` with batch configs
3. **Nano Banana Pro** — Deploy as serverless endpoint

### Python API Example

```python
from nodes.recolor_nodes import PreciseLABRecolor, BatchColorwayProcessor
import torch
from PIL import Image
import numpy as np

# Load image as tensor
img = np.array(Image.open("product.png").convert("RGB")) / 255.0
img_tensor = torch.from_numpy(img).float().unsqueeze(0)

# Load mask (from SAM or any source)
mask = np.array(Image.open("mask.png").convert("L")) / 255.0
mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

# Recolor
node = PreciseLABRecolor()
result = node.recolor(img_tensor, mask_tensor, 
                      target_r=66, target_g=17, target_b=34,
                      luminance_blend=0.6)

# Save
out = (result[0][0].numpy() * 255).astype(np.uint8)
Image.fromarray(out).save("recolored.png")
```

---

## File Structure

```
ComfyUI-Product-Recolor/
├── __init__.py                    # ComfyUI entry point
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── nodes/
│   ├── __init__.py
│   └── recolor_nodes.py           # All custom nodes
├── examples/
│   └── colorway_configs.json      # Adidas article color specs
└── test_recolor.py                # Standalone testing script
```

---

## Credits

Developed for the GEN AICG PP — Colorways Pipeline  
Wiethe Content GmbH — Head of AI
