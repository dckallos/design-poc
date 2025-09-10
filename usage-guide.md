Below is a **complete usage guide** for `patternlab.py`—the pattern generator I provided. It explains every argument, how they affect different pattern types, and how to design and integrate your own shapes (e.g., hearts, leaves, snowflakes).

**Download the script:**

* [patternlab.py](sandbox:/mnt/data/patternlab.py)

**Quick sample outputs (for reference):**

* [demo\_memphis.png](sandbox:/mnt/data/demo_memphis.png)
* [demo\_linefield.png](sandbox:/mnt/data/demo_linefield.png)
* [demo\_blobs.png](sandbox:/mnt/data/demo_blobs.png)

---

## 1) Installation

```bash
pip install pillow numpy
```

---

## 2) What this tool generates

* **Pattern engines (styles):**

  * `memphis_doodles` — bright, playful strokes, squiggles, plus/cross, etc.
  * `line_field` — maze‑ish white lines on colored background.
  * `blob_overlay` — organic, posterized blobs.

* **Inputs:**

  * **Hex color lists** (palettes)
  * **Shapes** (predefined or custom catalog)
  * **Arguments** that control density, stroke width, seamless tiling, random seed, etc.

* **Output:** PNG files, designed to scale **up to 30,000 × 30,000 px** (see memory notes).

---

## 3) Command‑line quick start

```bash
# Memphis / doodle style on black
python patternlab.py --out memphis.png --size 6000x6000 \
  --engine memphis_doodles \
  --palette "#0b0c10,#ffffff,#ff2e93,#ffb703,#2ec4f1" \
  --bg "#0b0c10" \
  --density 0.85 --stroke 6,18 --seed 3

# Line field (white lines on blue, seamless tile)
python patternlab.py --out linefield.png --size 8000x6000 \
  --engine line_field \
  --palette "#1d3557,#ffffff" --bg "#1d3557" \
  --density 0.30 --stroke 2,6 --seamless --seed 4

# Blob overlay (tone-on-tone)
python patternlab.py --out blobs.png --size 8000x5000 \
  --engine blob_overlay \
  --palette "#0c2340,#1b5e9a,#4aa3df,#89ccff,#ffffff" \
  --bg "#0c2340" \
  --density 0.6 --seed 8
```

---

## 4) Python API quick start

```python
from patternlab import generate

generate(
    out_path="pattern.png",
    width=4096, height=4096,
    palette=["#0b0c10","#ffffff","#ff2e93","#ffb703","#2ec4f1"],
    engine="memphis_doodles",
    bg="#0b0c10",
    density=0.9, seed=7,
    seamless=True,          # make it a perfect repeating tile
    paletted=True,          # memory efficient for huge images
    stroke_px=(6,18),
    shape_kinds=("squiggle","blob","circle","plus","cross"),
)
```

---

## 5) All arguments, what they do, and engine‑specific advice

### Required

* **`--out`** *(str)* – Output PNG path.
* **`--size`** *(WIDTHxHEIGHT)* – Image size, e.g., `8000x8000`.

### Colors & palette

* **`--palette`** *(CSV of hex)* – The color set for everything.
  Example: `"#111111,#ffffff,#f72585,#4cc9f0,#f49d37"`
* **`--bg`** *(hex)* – Background color. Defaults to the **first** color in `--palette`.

**Engine tips:**

* `memphis_doodles`: use 4–6 bold colors plus a dark or light background.
* `line_field`: two colors work great (dark bg + white lines).
* `blob_overlay`: use 4–8 related hues for tone‑on‑tone looks.

### Style/engine

* **`--engine`**: `memphis_doodles` | `line_field` | `blob_overlay`

### Structure & variation

* **`--density`** *(0..1)* – How busy/dense the pattern is.

  * `memphis_doodles`:

    * `0.6–0.95` usually looks good.
  * `line_field`:

    * **Important**: Too high + thick strokes can cover the entire background.
    * Recommended by size:

      * 1800×1200 → `0.25–0.40`
      * 6000×4000 → `0.25–0.40`
      * 12000×9000 → `0.20–0.35`
  * `blob_overlay`:

    * `0.4–0.8` for rich layering.

* **`--seed`** *(int)* – Reproducible randomness. Same inputs + same seed → same image.

### Shape set & strokes (primarily affects `memphis_doodles`)

* **`--shapes`** *(CSV)* – Limit shapes chosen by the engine. Options:

  * `squiggle, blob, circle, rect, rounded_rect, triangle, plus, cross`
  * You can also include `catalog` to favor your custom shapes (see §7).
  * Examples:
    `--shapes squiggle,blob,plus`
    `--shapes catalog` (catalog‑only look)

* **`--stroke`** *(min,max)* – Stroke width in **pixels** for outlines/lines.

  * `memphis_doodles` often looks good with `6,18` at \~4–8k sizes.
  * `line_field` must be tuned to avoid overdraw; start with `2,6` at \~1800×1200 and scale up slowly for larger images.
  * `blob_overlay` ignores stroke (it draws filled blobs).

* **`--fill-prob`** *(0..1)* – Probability a chosen shape is filled with a color.

* **`--outline-prob`** *(0..1)* – Probability a chosen shape has an outline.

### Tiling and output mode

* **`--seamless`** – Makes outputs that tile perfectly.

  * `memphis_doodles`: uses edge wrapping per shape.
  * `line_field`: lines wrap with toroidal coordinates.
  * `blob_overlay`: *currently draws without edge wrapping* (tiling will not be perfect).
* **`--no-paletted`** – Switch to RGBA mode (uses more memory, allows later post‑processing).
  Default is paletted *P* mode (≤256 colors) which is very memory efficient.
* **`--tile`** *(int)* – Reserved for advanced memory management. Current engines draw directly on the full canvas; the main memory control is paletted output. You can leave this at the default.

---

## 6) Engine‑specific recipes

### A. `memphis_doodles` (doodle/Memphis look)

* **Good starting point:**

  ```bash
  --engine memphis_doodles \
  --palette "#0b0c10,#ffffff,#ff2e93,#ffb703,#2ec4f1" --bg "#0b0c10" \
  --density 0.85 --stroke 6,18 --fill-prob 0.55 --outline-prob 0.9 \
  --shapes squiggle,blob,circle,plus,cross --seed 3
  ```
* **Notes:**

  * Rotation is applied to **polygons and polylines**, not to circle/rect kinds.

    * If you need rotated rectangles, define them as **polygon** shapes.
  * To emphasize your custom shapes, include `--shapes catalog` (and pass `--catalog`).

### B. `line_field` (maze‑ish white lines)

* **Key risk:** Overdraw (white lines cover everything).
* **Good starting points by size:**

  ```bash
  # 1800×1200
  --density 0.30 --stroke 2,6
  # 6000×4000
  --density 0.30 --stroke 4,12
  # 12000×9000
  --density 0.25 --stroke 8,22
  ```
* **Seamless**: Works well; keep density modest and strokes moderate.

### C. `blob_overlay` (organic posterized)

* **Good starting point:**

  ```bash
  --engine blob_overlay \
  --palette "#0c2340,#1b5e9a,#4aa3df,#89ccff,#ffffff" --bg "#0c2340" \
  --density 0.6 --seed 8
  ```
* **Note:** Currently doesn’t wrap shapes for seamless—use for stand‑alone prints/wallpapers or add a border bleed.

---

## 7) Creating **your own shapes** and integrating them

### 7.1 Shape coordinate system

* Shapes live in a **unit box** `[0..1] × [0..1]`.
* The engine scales and positions them inside random bounding boxes.
* **Kinds** you can define in JSON:

  * **Polygon** (filled/outlined):

    ```json
    {"kind":"polygon","points":[[x1,y1],[x2,y2],...]}
    ```
  * **Polyline** (stroked path, not closed):

    ```json
    {"kind":"polyline","points":[[x1,y1],[x2,y2],...],"width":0.08}
    ```

    `width` is **relative to the shape box** (0..1).
  * **Circle**:

    ```json
    {"kind":"circle","cx":0.5,"cy":0.5,"r":0.45}
    ```
  * **Rect** (optionally rounded):

    ```json
    {"kind":"rect","x":0.1,"y":0.1,"w":0.8,"h":0.8,"radius":0.15}
    ```

> Rotation is applied by the engine to **polygons** and **polylines**.
> Circles and rects are drawn axis‑aligned.

### 7.2 Practical design tips

* Keep each shape **self‑contained** (no separate parts); the engine picks one shape at a time. For composite looks (e.g., a snowflake with many branches), either:

  * design a **single polygon** that looks like the shape, or
  * design a **single polyline** that draws the arms by repeatedly returning to the center (see example below).
* Keep the number of points reasonable (e.g., ≤ 64) for rendering speed.
* For smooth curves (hearts/leaves), approximate with more points.
* Ensure all points stay within `[0,1]`. Slight overshoot can be clipped.

### 7.3 Example: **hearts, leaves, snowflakes** (ready to paste)

Save as `shapes.json`:

```json
[
  {
    "kind": "polygon",
    "points": [
      [0.500,0.950],[0.380,0.870],[0.290,0.780],[0.230,0.680],
      [0.200,0.580],[0.200,0.500],[0.220,0.430],[0.270,0.370],
      [0.340,0.340],[0.410,0.345],[0.460,0.370],[0.500,0.410],
      [0.540,0.370],[0.590,0.345],[0.660,0.340],[0.730,0.370],
      [0.780,0.430],[0.800,0.500],[0.800,0.580],[0.770,0.680],
      [0.710,0.780],[0.620,0.870]
    ]
  },
  {
    "kind": "polygon",
    "points": [
      [0.500,0.050],[0.560,0.090],[0.610,0.150],[0.650,0.220],
      [0.680,0.300],[0.700,0.380],[0.710,0.460],[0.705,0.540],
      [0.680,0.620],[0.640,0.700],[0.590,0.770],[0.530,0.830],
      [0.470,0.870],[0.410,0.890],[0.350,0.890],[0.300,0.870],
      [0.260,0.830],[0.220,0.770],[0.190,0.700],[0.170,0.620],
      [0.160,0.540],[0.170,0.460],[0.190,0.380],[0.220,0.300],
      [0.260,0.220],[0.310,0.150],[0.360,0.090]
    ]
  },
  {
    "kind": "polyline",
    "points": [
      [0.500,0.500],[0.500,0.120],[0.500,0.500],[0.880,0.500],
      [0.500,0.500],[0.500,0.880],[0.500,0.500],[0.120,0.500],
      [0.500,0.500],[0.780,0.220],[0.500,0.500],[0.220,0.220],
      [0.500,0.500],[0.220,0.780],[0.500,0.500],[0.780,0.780]
    ],
    "width": 0.10
  }
]
```

* First polygon ≈ **heart**.
* Second polygon ≈ **leaf** (simple teardrop/leaf outline).
* Polyline is a **snowflake-like asterisk** (6+ arms). The path goes **center → tip → center** repeatedly to draw multiple arms in one polyline.

**Use them:**

```bash
python patternlab.py --out custom_shapes.png --size 6000x6000 \
  --engine memphis_doodles \
  --palette "#111111,#ffffff,#ff006e,#8338ec,#3a86ff,#ffbe0b" --bg "#111111" \
  --catalog shapes.json \
  --shapes squiggle,blob,circle,plus,catalog \
  --density 0.85 --stroke 6,18 --seed 11
```

**Catalog selection behavior:**
If you pass `--catalog`, the engine will automatically sample from your catalog alongside built‑ins. If you want **catalog only**, use `--shapes catalog`.

---

## 8) Working at very large sizes (e.g., 30,000 × 30,000)

* **Prefer paletted mode** (default). Your palette must be ≤256 colors.
  Memory ≈ `width × height` bytes → 30,000² ≈ **900 MB**.
* **RGBA mode** (`--no-paletted`) needs \~4× the memory. Use only if you plan to post‑process with alpha or >256 colors.
* PNG saving is streaming, but you should still have enough RAM/swap headroom.
* The current engines draw directly to the full canvas; the **main** memory control is paletted mode. (The `--tile` option is reserved for future tile‑wise rendering.)

---

## 9) Troubleshooting & FAQs

* **“My `line_field` output is all white.”**
  That’s overdraw (too many lines × too thick strokes). Lower density and/or stroke widths. Good start at 1800×1200: `--density 0.30 --stroke 2,6`.
* **“I see seams on edges when tiling.”**
  Use `--seamless`. Works in `memphis_doodles` and `line_field`.
  `blob_overlay` currently doesn’t wrap shapes; it’s not meant for perfect tiling.
* **“Shapes don’t rotate.”**
  Only **polygons** and **polylines** are rotated by the engine. Circles/rects stay axis‑aligned—convert rotated rectangles to polygons.
* **“Colors look banded.”**
  Paletted mode has ≤256 colors. If you need smooth gradients or soft edges, export with `--no-paletted` (RGBA) and quantize later if needed.

---

## 10) Advanced: add your own engine (optional)

Engines are simple functions:

```python
# patternlab.py
def engine_my_style(rng, canvas, p):
    from patternlab import hex_to_rgb
    W, H = canvas.size
    bg = hex_to_rgb(p.bg)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0,0,W,H], fill=bg)
    # draw whatever you want using p.palette, p.density, etc.

ENGINES["my_style"] = engine_my_style
```

Now call with `--engine my_style`.

---

## 11) Good defaults & quick recipes

**Memphis (dark):**

```bash
--engine memphis_doodles \
--palette "#0b0c10,#ffffff,#ff2e93,#ffb703,#2ec4f1" --bg "#0b0c10" \
--density 0.85 --stroke 6,18 --seed 3
```

**Memphis (light):**

```bash
--engine memphis_doodles \
--palette "#f5f3ef,#101010,#f72585,#4cc9f0,#f49d37" --bg "#f5f3ef" \
--density 0.80 --stroke 6,18 --seed 9
```

**Line field (seamless):**

```bash
--engine line_field --seamless \
--palette "#1d3557,#ffffff" --bg "#1d3557" \
--density 0.30 --stroke 2,6 --seed 4
```

**Blobs (tone-on-tone):**

```bash
--engine blob_overlay \
--palette "#143642,#335865,#55828B,#87BBA2,#EAF9F9" --bg "#143642" \
--density 0.6 --seed 8
```

---

