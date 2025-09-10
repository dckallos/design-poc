## What the script does (at a glance)

* **Engines (styles)**:

  * `memphis_doodles` – bright strokes/squiggles à la the images you shared
  * `line_field` – maze‑ish white lines on colored ground
  * `blob_overlay` – organic posterized blobs (tone‑on‑tone)

* **Shapes you can use** (predefined): `squiggle`, `blob`, `circle`, `rect`, `rounded_rect`, `triangle`, `plus`, `cross`
  You can also pass a **catalog** of custom shapes as normalized polygons/polylines/circles/rects (details below).

* **Seamless tiling**: `--seamless` wraps edges so the image tiles perfectly.

* **Huge images (up to 30,000×30,000)**:

  * Uses **paletted PNG** by default (≤256 colors) → \~1 byte per pixel.

    * Memory estimate: `width × height` bytes → 30k×30k ≈ **900 MB**.
  * Optional **tiling** (`--tile`) to constrain peak memory while drawing.
  * Deterministic results with `--seed`.

---

## Install requirements

```bash
pip install pillow numpy
```

---

## Command‑line usage

```bash
# Example 1: Memphis/doodle look on black
python patternlab.py \
  --out pattern.png \
  --size 8000x8000 \
  --engine memphis_doodles \
  --palette "#0b0c10,#ffffff,#ff2e93,#ffb703,#2ec4f1" \
  --bg "#0b0c10" \
  --density 0.9 \
  --seamless \
  --tile 2048 \
  --seed 7

# Example 2: Line field (maze-ish), tone on blue
python patternlab.py \
  --out linefield.png \
  --size 12000x9000 \
  --engine line_field \
  --palette "#1d3557,#ffffff" \
  --bg "#1d3557" \
  --density 0.7 \
  --seamless \
  --tile 2048 \
  --seed 4

# Example 3: Blob overlay (cool blues)
python patternlab.py \
  --out blobs.png \
  --size 10000x7000 \
  --engine blob_overlay \
  --palette "#0c2340,#1b5e9a,#4aa3df,#89ccff,#ffffff" \
  --bg "#0c2340" \
  --density 0.6 \
  --tile 2048 \
  --seed 8
```

### Important flags & arguments

* `--palette` – comma‑separated hex colors; first color is the background unless `--bg` is set.
* `--engine` – one of `memphis_doodles`, `line_field`, `blob_overlay`.
* `--density` – `0..1`, how busy the pattern is.
* `--seamless` – make the output a perfect repeating tile.
* `--tile` – tile size for memory control (e.g., 1024–2048). Use `0` to disable.
* `--no-paletted` – switch off paletted mode (uses RGBA; needs more memory).
* `--seed` – reproducibility.
* `--shapes` – restrict the shape set, e.g. `--shapes squiggle,blob,plus`.
* `--stroke` – min,max stroke width in pixels (e.g. `--stroke 6,18`).
* `--fill-prob`, `--outline-prob` – tune fill/outline balances.

---

## Using a **custom shape catalog**

You can provide a JSON file via `--catalog` describing shapes in **normalized coordinates** (0..1) inside a unit box. Supported kinds:

* **Polygon**:

  ```json
  {"kind":"polygon", "points": [[0.1,0.2],[0.8,0.2],[0.5,0.9]]}
  ```
* **Polyline** (stroked line):

  ```json
  {"kind":"polyline", "points": [[0.1,0.5],[0.5,0.8],[0.9,0.2]], "width": 0.08}
  ```

  `width` is **relative** to the shape box (0..1).
* **Circle**:

  ```json
  {"kind":"circle", "cx":0.5, "cy":0.5, "r":0.45}
  ```
* **Rect** (optionally rounded):

  ```json
  {"kind":"rect", "x":0.1, "y":0.1, "w":0.8, "h":0.8, "radius":0.15}
  ```

**Example catalog file** (`shapes.json`):

```json
[
  {"kind":"polygon", "points":[[0.5,0.0],[1.0,1.0],[0.0,1.0]]},
  {"kind":"polyline", "points":[[0.1,0.5],[0.5,0.2],[0.9,0.8]], "width":0.1},
  {"kind":"circle", "cx":0.5, "cy":0.5, "r":0.45},
  {"kind":"rect", "x":0.1, "y":0.2, "w":0.8, "h":0.6, "radius":0.2}
]
```

Use it like this:

```bash
python patternlab.py \
  --out custom.png \
  --size 6000x6000 \
  --engine memphis_doodles \
  --palette "#111111,#ffffff,#ff006e,#8338ec,#3a86ff,#ffbe0b" \
  --bg "#111111" \
  --catalog shapes.json \
  --shapes squiggle,blob,circle,plus,catalog \
  --density 0.85 \
  --seamless \
  --seed 11
```

---

## Python API (if you prefer importing)

```python
from patternlab import generate

generate(
    out_path="pattern.png",
    width=4096, height=4096,
    palette=["#0b0c10","#ffffff","#ff2e93","#ffb703","#2ec4f1"],
    engine="memphis_doodles",
    bg="#0b0c10",
    density=0.9,
    seamless=True,
    tile=2048,        # memory guard
    paletted=True,    # best for big images
    seed=7,
    shape_kinds=("squiggle","blob","circle","plus","cross"),
    stroke_px=(6,18),
    fill_prob=0.55,
    outline_prob=0.9,
    catalog=[{"kind":"polyline","points":[[0.1,0.2],[0.8,0.2],[0.2,0.8]],"width":0.12}]
)
```

---

## Notes for **30,000×30,000** outputs

* Keep **paletted mode** enabled (default). Your palette must be ≤256 colors.
  Memory estimate: `width × height` bytes. For 30k² → \~900 MB.
* Prefer `--tile 1024` or `2048` to bound working memory during draws.
* Avoid heavy translucency/antialiasing when targeting paletted output (the module uses solid fills/lines by default).
* If you need more than 256 colors, disable paletted mode (`--no-paletted`) and ensure your machine has enough RAM (RGBA is \~4 bytes/pixel → 30k² ≈ 3.35 GB just for pixels).

---

## How this maps to your request

1. **Hex color lists** – accepted as `--palette` or the `palette` parameter (string hex codes).
2. **Shapes** – choose from **predefined** or pass a **catalog** JSON/list of custom shapes (normalized to 0..1) that the script scales/rotates/places.
3. **Other arguments** – engines, density, seamless, stroke widths, min/max scale, outline/fill probabilities, seed, tile size, paletted vs RGBA, background color.

---

## What’s inside the module

* Utilities: palette building, hex parsing, RNG from seed.
* Shape catalog + helpers (rotate/scale/translate normalized shapes).
* Drawing primitives (polygon, polyline/stroke, circle, rect, rounded rect).
* Engines combining shapes into full patterns.
* Seamless wrapping logic (draws across edges when needed).
* A CLI with helpful defaults and a JSON catalog loader.

---

If you’d like, tell me the exact palette(s) and the style you prefer (e.g., “like the first/second/third example”), and I can preconfigure a recipe and output a tile or a full‑size 30k image with those settings.
