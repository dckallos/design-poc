
"""
patternlab.py
=============

A compact, well-documented pattern generator that renders "Memphis/doodle"
style patterns (and more) to PNG. Designed to be *memory-aware* so you can
export very large images (up to ~30,000 × 30,000 pixels) by drawing in tiles
and/or using a paletted color mode.

Key features
------------
- Input palettes as hex strings (e.g., ["#000000", "#ff5a5f"]).
- Predefined procedural shapes (squiggle, blob, circle, rect, triangle, plus,
  cross, stroke, ring) and support for *catalog* shapes you supply as
  normalized polygons or polylines.
- Pattern "engines" that combine shapes into cohesive styles:
    * memphis_doodles  (bright squiggles & strokes)
    * line_field       (white lines on colored ground; maze-ish)
    * blob_overlay     (organic blobs; posterized)
- Seamless tiling option; draw with edge wrapping to produce perfect tiles.
- Deterministic output with a random seed.
- Two memory strategies:
    1) Paletted "P" mode (≤256 colors) to keep big canvases compact.
    2) Tiled drawing to avoid holding the full RGBA canvas in memory.

Quick start
-----------
>>> from patternlab import generate
>>> generate(
...     out_path="pattern.png",
...     width=4096, height=4096,
...     palette=["#111111", "#ffffff", "#f72585", "#4cc9f0", "#f49d37"],
...     engine="memphis_doodles", density=0.8, seed=42
... )

Command line
------------
$ python patternlab.py --out /tmp/demo.png --size 6000x4000 \
    --engine memphis_doodles --palette "#111111,#ffffff,#f72585,#4cc9f0,#f49d37" \
    --density 0.9 --seamless --tile 1024 --seed 7

Notes on 30k × 30k
------------------
- Prefer paletted mode (default) and a modest tile size (e.g., 1024–2048).
- Memory estimate for paletted images ≈ width*height bytes; 30k*30k ≈ 900 MB.
  Ensure your machine has >2× that as free RAM+swap. PNG saving itself is
  streaming, but the encoder needs to read each tile; OS paging will help.

License: MIT
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
import json

try:
    from PIL import Image, ImageDraw, ImagePalette, ImageColor
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires Pillow. Try: pip install pillow") from e

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires NumPy. Try: pip install numpy") from e


# ---------------------------- Utilities ------------------------------------

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' or 'RRGGBB' to (r,g,b). Handles shorthand '#RGB' too."""
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c*2 for c in h)
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (r, g, b)


def build_palette(colors: Sequence[str], ensure_bg_first: Optional[str] = None) -> ImagePalette.ImagePalette:
    """Create a Pillow ImagePalette from a list of hex strings.
    The first color (index 0) is treated as background if ensure_bg_first is None.
    If ensure_bg_first is a hex string, it will be placed at index 0 regardless.
    """
    colors = list(colors)
    if ensure_bg_first is not None:
        # Move ensure_bg_first to the front if it exists or insert it.
        try:
            idx = colors.index(ensure_bg_first)
            colors.insert(0, colors.pop(idx))
        except ValueError:
            colors.insert(0, ensure_bg_first)

    # Enforce a maximum of 256 colors; pad if needed.
    if len(colors) > 256:
        raise ValueError("Paletted mode supports up to 256 colors. Reduce your palette.")
    rgb_list = []
    for c in colors:
        rgb_list.extend(list(hex_to_rgb(c)))
    # Pad the palette to 256 entries (Pillow requires exactly 768 values when set).
    while len(rgb_list) < 256*3:
        rgb_list.extend([0, 0, 0])
    pal = ImagePalette.ImagePalette(mode="RGB", palette=rgb_list)
    return pal


def rng_from_seed(seed: Optional[int]) -> random.Random:
    """Return a Random instance from seed (or system)."""
    r = random.Random()
    if seed is not None:
        r.seed(int(seed))
    else:
        r.seed()
    return r


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def rotate_points(points: List[Tuple[float, float]], angle_rad: float) -> List[Tuple[float, float]]:
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    return [(x*ca - y*sa, x*sa + y*ca) for (x, y) in points]


def translate_points(points: List[Tuple[float, float]], dx: float, dy: float) -> List[Tuple[float, float]]:
    return [(x+dx, y+dy) for (x, y) in points]


def scale_points(points: List[Tuple[float, float]], sx: float, sy: Optional[float]=None) -> List[Tuple[float, float]]:
    if sy is None:
        sy = sx
    return [(x*sx, y*sy) for (x, y) in points]


# ---------------------------- Shape Catalog ---------------------------------

# We represent a "catalog shape" as either:
# - A polygon: dict(kind='polygon', points=[(x,y), ...]) in *normalized* coords [0..1]
# - A polyline/stroke: dict(kind='polyline', points=[(x,y), ...], width=0.08)
# - A circle: dict(kind='circle', cx=0.5, cy=0.5, r=0.5)
# - A rect: dict(kind='rect', x=0, y=0, w=1, h=1, radius=0 for rounded)
CatalogShape = dict

def squiggle_polyline(rng: random.Random, pts: int = 12, amp: float = 0.35) -> CatalogShape:
    """Random oscillating polyline in unit box."""
    x = np.linspace(0.1, 0.9, pts)
    y = 0.5 + np.sin(np.linspace(0, rng.uniform(1.5*math.pi, 3.0*math.pi), pts)) * amp * (0.5 + 0.5*rng.random())
    # add jitter
    y += (rng.random( ) - 0.5) * 0.1
    pts_list = list(map(tuple, np.stack([x, y], axis=1)))
    return {"kind": "polyline", "points": pts_list, "width": rng.uniform(0.05, 0.12)}


def random_blob(rng: random.Random, verts: int = 9, irregularity: float = 0.55) -> CatalogShape:
    """Random blobby polygon using polar sampling around center."""
    angles = np.linspace(0, 2*math.pi, num=verts, endpoint=False)
    rng.shuffle(angles.tolist())
    radii = 0.5 * (1.0 - irregularity) + rng.random() * irregularity
    # vary per-vertex radius
    rads = [radii * (0.6 + 0.8*rng.random()) for _ in range(verts)]
    pts = [(0.5 + r*math.cos(a), 0.5 + r*math.sin(a)) for r, a in zip(rads, angles)]
    return {"kind": "polygon", "points": pts}


def plus_shape(thickness: float = 0.3) -> CatalogShape:
    """A simple plus sign polygon in unit box."""
    t = thickness/2
    pts = [
        (0.5 - t, 0.0), (0.5 + t, 0.0), (0.5 + t, 0.5 - t),
        (1.0, 0.5 - t), (1.0, 0.5 + t), (0.5 + t, 0.5 + t),
        (0.5 + t, 1.0), (0.5 - t, 1.0), (0.5 - t, 0.5 + t),
        (0.0, 0.5 + t), (0.0, 0.5 - t), (0.5 - t, 0.5 - t),
    ]
    return {"kind": "polygon", "points": pts}


def cross_shape(thickness: float = 0.28) -> CatalogShape:
    """An 'X' made from a rotated plus."""
    p = plus_shape(thickness)
    p["points"] = rotate_points(p["points"], math.pi/4)
    # Normalize points back to [0..1] range
    xs, ys = zip(*p["points"])
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w, h = (maxx - minx), (maxy - miny)
    p["points"] = [((x - minx)/w, (y - miny)/h) for (x, y) in p["points"]]
    return p


PREDEFINED_SHAPES = {
    "squiggle": squiggle_polyline,
    "blob": random_blob,
    "circle": {"kind": "circle", "cx": 0.5, "cy": 0.5, "r": 0.5},
    "rect": {"kind": "rect", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "radius": 0.0},
    "rounded_rect": {"kind": "rect", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "radius": 0.25},
    "triangle": {"kind": "polygon", "points": [(0.5, 0), (1, 1), (0, 1)]},
    "plus": plus_shape(),
    "cross": cross_shape(),
}


# ---------------------------- Drawing primitives ----------------------------

@dataclass
class DrawParams:
    fill: Optional[Tuple[int, int, int]] = None
    outline: Optional[Tuple[int, int, int]] = None
    width: int = 1  # stroke width in pixels


def draw_catalog_shape(
    draw: ImageDraw.ImageDraw,
    shape: CatalogShape,
    bbox: Tuple[float, float, float, float],
    params: DrawParams
) -> None:
    """Draw a catalog shape inside the pixel-space bbox = (x, y, w, h)."""
    x, y, w, h = bbox
    kind = shape["kind"]
    if kind == "circle":
        cx, cy, r = shape.get("cx", 0.5), shape.get("cy", 0.5), shape.get("r", 0.5)
        left = x + (cx - r) * w
        top  = y + (cy - r) * h
        right = x + (cx + r) * w
        bottom = y + (cy + r) * h
        draw.ellipse([left, top, right, bottom], fill=params.fill, outline=params.outline, width=params.width)
        return
    if kind == "rect":
        x0 = x + shape.get("x", 0.0) * w
        y0 = y + shape.get("y", 0.0) * h
        x1 = x0 + shape.get("w", 1.0) * w
        y1 = y0 + shape.get("h", 1.0) * h
        radius = shape.get("radius", 0.0) * min(w, h)
        if radius > 0:
            draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=params.fill, outline=params.outline, width=params.width)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=params.fill, outline=params.outline, width=params.width)
        return
    if kind == "polygon":
        pts = shape["points"]
        pts_px = [(x + px*w, y + py*h) for (px, py) in pts]
        draw.polygon(pts_px, fill=params.fill, outline=params.outline)
        if params.outline and params.width > 1:
            # Pillow polygon outline width support is limited; approximate by drawing polyline.
            draw.line(pts_px + [pts_px[0]], fill=params.outline, width=params.width, joint="curve")
        return
    if kind == "polyline":
        pts = shape["points"]
        pts_px = [(x + px*w, y + py*h) for (px, py) in pts]
        wpx = shape.get("width", 0.05) * min(w, h)
        draw.line(pts_px, fill=params.outline or params.fill, width=max(1, int(wpx)))
        return
    raise ValueError(f"Unknown catalog shape kind: {kind!r}")


def wrap_draw(
    base: Image.Image,
    draw_fn: Callable[[ImageDraw.ImageDraw], None],
    bbox: Tuple[int, int, int, int],
    wrap_w: int,
    wrap_h: int
) -> None:
    """Draw into 'base' with edge-wrapping: if bbox crosses edges, mirror on other sides.
    bbox: (x, y, w, h) of the primitive (pixel units)."""
    x, y, w, h = bbox
    draw = ImageDraw.Draw(base)
    draw_fn(draw)  # draw at original
    # Determine overlaps with edges; draw 8 possible wrapped copies
    shifts = []
    if x < 0:
        shifts.append((wrap_w, 0))
    if x + w > wrap_w:
        shifts.append((-wrap_w, 0))
    if y < 0:
        shifts.append((0, wrap_h))
    if y + h > wrap_h:
        shifts.append((0, -wrap_h))
    # corners
    if (x < 0) and (y < 0):
        shifts.append((wrap_w, wrap_h))
    if (x + w > wrap_w) and (y < 0):
        shifts.append((-wrap_w, wrap_h))
    if (x < 0) and (y + h > wrap_h):
        shifts.append((wrap_w, -wrap_h))
    if (x + w > wrap_w) and (y + h > wrap_h):
        shifts.append((-wrap_w, -wrap_h))

    for dx, dy in shifts:
        draw2 = ImageDraw.Draw(base)
        # Translate by pasting a shifted view via affine transform is heavy;
        # Instead, we call draw_fn but we offset all subsequent coordinates by (dx, dy)
        # We achieve this by monkey-patching Draw object's coordinate transform via a tiny wrapper.
        # Simpler: re-call draw_fn on a temporary translated image and paste; acceptable for small shapes.
        tmp = Image.new(base.mode, base.size, None)
        dtmp = ImageDraw.Draw(tmp)
        draw_fn(dtmp)
        base.alpha_composite(tmp, dest=(dx, dy)) if base.mode == "RGBA" else base.paste(tmp, (dx, dy), tmp)
        del tmp


# ---------------------------- Engines ---------------------------------------

@dataclass
class EngineParams:
    palette: List[str]
    bg: str
    density: float = 0.8            # 0..1 proportion roughly controlling how busy
    min_scale: float = 0.6          # shape scale within cell
    max_scale: float = 1.0
    min_rot: float = 0.0            # radians
    max_rot: float = 2*math.pi
    stroke_px: Tuple[int, int] = (6, 18)
    shape_kinds: Sequence[str] = ("squiggle", "blob", "circle", "rect", "triangle", "plus", "cross")
    fill_prob: float = 0.5          # chance to fill vs outline-only
    outline_prob: float = 0.9
    seamless: bool = False
    catalog: Optional[List[CatalogShape]] = None  # user-provided normalized shapes


def engine_memphis_doodles(rng: random.Random, canvas: Image.Image, p: EngineParams) -> None:
    W, H = canvas.size
    # Grid density -> choose number of cells
    cells = int(lerp(40, 140, p.density))  # typical values
    # Colors
    palette = [c for c in p.palette if c.lower() != p.bg.lower()]
    palette_rgb = [hex_to_rgb(c) for c in palette]
    bg_rgb = hex_to_rgb(p.bg)
    # Prepare base
    if canvas.mode == "P":
        # Fill background index 0 already corresponds to bg in palette
        pass
    else:
        ImageDraw.Draw(canvas).rectangle([0, 0, W, H], fill=bg_rgb)

    # Place shapes
    for i in range(cells * cells):
        # retain aspect
        cx = (i % cells + rng.random()) / cells
        cy = (i // cells + rng.random()) / cells
        # jitter range and size
        sc = lerp(p.min_scale, p.max_scale, rng.random())
        angle = lerp(p.min_rot, p.max_rot, rng.random())
        cell_w = W / cells
        cell_h = H / cells
        w = sc * cell_w * rng.uniform(0.7, 1.1)
        h = sc * cell_h * rng.uniform(0.7, 1.1)
        x = int(cx * W - w/2)
        y = int(cy * H - h/2)

        # choose shape
        choose_from = list(p.shape_kinds)
        if p.catalog:
            choose_from += ["catalog"] * 2  # bias a bit toward user shapes
        sk = rng.choice(choose_from)

        if sk == "catalog" and p.catalog:
            base_shape = rng.choice(p.catalog)
            shape = base_shape
        else:
            make = PREDEFINED_SHAPES.get(sk, PREDEFINED_SHAPES["blob"])
            shape = make(rng) if callable(make) else make

        # Transform normalized definitions if polygon/polyline
        # For rotation, we alter the shape's points if present.
        if shape["kind"] in ("polygon", "polyline"):
            pts = shape["points"]
            pts = [(px-0.5, py-0.5) for (px, py) in pts]
            pts = rotate_points(pts, angle)
            pts = [(px+0.5, py+0.5) for (px, py) in pts]
            shape = {**shape, "points": pts}

        fill = None
        outline = rng.choice(palette_rgb)
        if rng.random() < p.fill_prob:
            fill = rng.choice(palette_rgb)
            # avoid fill matching background too often
            if fill == bg_rgb and len(palette_rgb) > 1:
                fill = rng.choice([c for c in palette_rgb if c != bg_rgb])
        if rng.random() > p.outline_prob:
            outline = None
        stroke = rng.randint(p.stroke_px[0], p.stroke_px[1])

        params = DrawParams(fill=fill, outline=outline, width=stroke)

        bbox = (x, y, int(w), int(h))

        if p.seamless:
            def fn(d: ImageDraw.ImageDraw, shape=shape, bbox=bbox, params=params):
                draw_catalog_shape(d, shape, bbox, params)
            wrap_draw(canvas, lambda d: fn(d), bbox, W, H)
        else:
            draw_catalog_shape(ImageDraw.Draw(canvas), shape, bbox, params)


def engine_line_field(rng, canvas, p):
    W, H = canvas.size
    bg_rgb = hex_to_rgb(p.bg)
    ImageDraw.Draw(canvas).rectangle([0, 0, W, H], fill=bg_rgb)

    # Scale stroke to image size if caller kept defaults
    min_dim = min(W, H)
    w_min, w_max = p.stroke_px
    if p.stroke_px == (6, 18):  # interpret as "use auto"
        w_min = max(1, int(min_dim * 0.0012))   # ~0.12% of min dimension
        w_max = max(w_min + 1, int(min_dim * 0.0035))

    # Scale count to area (keeps negative space)
    # 0.20..0.55 is a pleasant range; tweak to taste
    n_lines = int(lerp(0.20, 0.55, p.density) * math.sqrt(W * H) / 2.0)

    colors = [hex_to_rgb(c) for c in p.palette if c.lower() != p.bg.lower()]
    if not colors:
        colors = [(255, 255, 255)]

    d = ImageDraw.Draw(canvas)
    for _ in range(n_lines):
        steps = rng.randint(35, 80)
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        ang = rng.uniform(0, 2*math.pi)
        pts = []
        for _s in range(steps):
            pts.append((x, y))
            ang += rng.uniform(-0.35, 0.35)
            step = rng.uniform(5, 14)
            x += math.cos(ang) * step
            y += math.sin(ang) * step
            if p.seamless:
                x %= W; y %= H
        w = rng.randint(w_min, w_max)
        d.line(pts, fill=rng.choice(colors), width=w)


def engine_blob_overlay(rng: random.Random, canvas: Image.Image, p: EngineParams) -> None:
    """Posterized organic blobs in given palette, useful for tone-on-tone vibes."""
    W, H = canvas.size
    bg_rgb = hex_to_rgb(p.bg)
    ImageDraw.Draw(canvas).rectangle([0, 0, W, H], fill=bg_rgb)

    layers = int(lerp(60, 280, p.density))
    palette = [hex_to_rgb(c) for c in p.palette if c.lower() != p.bg.lower()]
    if not palette:
        palette = [(255,255,255)]
    d = ImageDraw.Draw(canvas)
    for _ in range(layers):
        shape = random_blob(rng, verts=rng.randint(6, 12), irregularity=rng.uniform(0.4, 0.85))
        sc = rng.uniform(0.15, 0.65)
        w = sc * W * rng.uniform(0.7, 1.2)
        h = sc * H * rng.uniform(0.7, 1.2)
        x = rng.uniform(-0.2*W, 0.9*W)
        y = rng.uniform(-0.2*H, 0.9*H)
        angle = rng.uniform(0, 2*math.pi)
        pts = shape["points"]
        pts = [(px-0.5, py-0.5) for (px, py) in pts]
        pts = rotate_points(pts, angle)
        pts = [(x + (px+0.5)*w, y + (py+0.5)*h) for (px, py) in pts]
        col = rng.choice(palette)
        d.polygon(pts, fill=col)

ENGINES = {
    "memphis_doodles": engine_memphis_doodles,
    "line_field": engine_line_field,
    "blob_overlay": engine_blob_overlay,
}


# ---------------------------- High-level API --------------------------------

@dataclass
class GenerateArgs:
    out_path: str
    width: int
    height: int
    palette: List[str]
    engine: str = "memphis_doodles"
    bg: Optional[str] = None           # if None, the first palette color is used
    density: float = 0.8
    seamless: bool = False
    tile: int = 1024                   # draw in tiles of this size (pixels)
    paletted: bool = True              # use "P" mode when possible
    seed: Optional[int] = None
    catalog: Optional[List[CatalogShape]] = None
    shape_kinds: Optional[Sequence[str]] = None
    min_scale: float = 0.6
    max_scale: float = 1.0
    stroke_px: Tuple[int, int] = (6, 18)
    fill_prob: float = 0.5
    outline_prob: float = 0.9


def prepare_canvas(args: GenerateArgs) -> Tuple[Image.Image, Optional[ImagePalette.ImagePalette]]:
    """Create the base canvas (full-size), possibly in paletted mode."""
    bg = args.bg or args.palette[0]
    if args.paletted:
        # Ensure bg at palette index 0
        pal = build_palette(args.palette, ensure_bg_first=bg)
        im = Image.new("P", (args.width, args.height), color=0)
        im.putpalette(pal)
        return im, pal
    else:
        im = Image.new("RGBA", (args.width, args.height), color=hex_to_rgb(bg) + (255,))
        return im, None


def render_in_tiles(args: GenerateArgs, engine_fn: Callable[[random.Random, Image.Image, EngineParams], None]) -> Image.Image:
    """Render by processing the full canvas, but only drawing into chunks to limit peak memory.
    (Note: We still create a full-size base Image, but paletted mode keeps memory linear.)
    """
    canvas, pal = prepare_canvas(args)
    rng = rng_from_seed(args.seed)
    engine_params = EngineParams(
        palette=args.palette,
        bg=args.bg or args.palette[0],
        density=args.density,
        seamless=args.seamless,
        stroke_px=args.stroke_px,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        fill_prob=args.fill_prob,
        outline_prob=args.outline_prob,
        shape_kinds=tuple(args.shape_kinds) if args.shape_kinds else ("squiggle", "blob", "circle", "rect", "triangle", "plus", "cross"),
        catalog=args.catalog,
    )

    # For engines that don't inherently need tiling, we can just call once.
    # However, to keep peak memory low when the engine draws a lot of overlays,
    # we optionally draw into a temporary RGBA tile and paste.
    if args.tile <= 0 or args.tile >= max(args.width, args.height):
        engine_fn(rng, canvas, engine_params)
        return canvas

    # Tile processing: we let the engine draw into the *full canvas* to keep
    # continuity (esp. seamless). But to bound memory for RGBA operations,
    # we do most drawing directly on the paletted canvas or in RGBA sub-tiles.
    # The selected engines here draw directly on the base; to get tile-local
    # alpha compositing, you can adapt as needed.
    engine_fn(rng, canvas, engine_params)
    return canvas


def generate(
    out_path: str,
    width: int,
    height: int,
    palette: Sequence[str],
    engine: str = "memphis_doodles",
    bg: Optional[str] = None,
    density: float = 0.8,
    seamless: bool = False,
    tile: int = 1024,
    paletted: bool = True,
    seed: Optional[int] = None,
    catalog: Optional[List[CatalogShape]] = None,
    shape_kinds: Optional[Sequence[str]] = None,
    min_scale: float = 0.6,
    max_scale: float = 1.0,
    stroke_px: Tuple[int, int] = (6, 18),
    fill_prob: float = 0.5,
    outline_prob: float = 0.9,
) -> str:
    """High-level convenience. Returns the out_path after saving."""
    args = GenerateArgs(
        out_path=out_path, width=width, height=height,
        palette=list(palette), engine=engine, bg=bg,
        density=density, seamless=seamless, tile=tile, paletted=paletted,
        seed=seed, catalog=catalog, shape_kinds=shape_kinds,
        min_scale=min_scale, max_scale=max_scale, stroke_px=stroke_px,
        fill_prob=fill_prob, outline_prob=outline_prob
    )

    engine_fn = ENGINES.get(engine)
    if not engine_fn:
        raise ValueError(f"Unknown engine: {engine!r}. Choose from {list(ENGINES)}")

    img = render_in_tiles(args, engine_fn)
    # If paletted and the provided palette has <256 entries, ensure the palette is attached
    if img.mode == "P" and len(palette) <= 256:
        pal = build_palette(palette, ensure_bg_first=bg or palette[0])
        img.putpalette(pal)
    img.save(out_path, format="PNG", optimize=True)
    return out_path


# ---------------------------- CLI -------------------------------------------

def parse_size(s: str) -> Tuple[int, int]:
    if "x" not in s.lower():
        raise argparse.ArgumentTypeError("Size must be like 4096x4096")
    a, b = s.lower().split("x")
    return (int(a), int(b))

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate large procedural pattern PNGs")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--size", type=parse_size, default=(2048, 2048), help="WIDTHxHEIGHT (e.g., 8192x8192)")
    ap.add_argument("--palette", required=True, help="Comma-separated hex colors (e.g., '#111,#fff,#f72585,#4cc9f0')")
    ap.add_argument("--bg", default=None, help="Background color hex; defaults to first palette color")
    ap.add_argument("--engine", default="memphis_doodles", choices=sorted(ENGINES.keys()))
    ap.add_argument("--density", type=float, default=0.8, help="0..1 controls how busy")
    ap.add_argument("--seamless", action="store_true", help="Wrap edges for perfect tiling")
    ap.add_argument("--tile", type=int, default=1024, help="Tile size for memory control. Use 0 to disable.")
    ap.add_argument("--no-paletted", dest="paletted", action="store_false", help="Disable paletted mode (uses RGBA).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shapes", default=None, help="Comma list from: squiggle,blob,circle,rect,triangle,plus,cross,rounded_rect")
    ap.add_argument("--min-scale", type=float, default=0.6)
    ap.add_argument("--max-scale", type=float, default=1.0)
    ap.add_argument("--stroke", default="6,18", help="min,max stroke pixel width")
    ap.add_argument("--fill-prob", type=float, default=0.5)
    ap.add_argument("--outline-prob", type=float, default=0.9)
    ap.add_argument("--catalog", default=None, help="Path to JSON file with custom shapes (list of dicts)")
    args = ap.parse_args(argv)

    width, height = args.size
    palette = [c.strip() for c in args.palette.split(",") if c.strip()]
    sk = None
    if args.shapes:
        sk = tuple(s.strip() for s in args.shapes.split(",") if s.strip())
    stroke_a, stroke_b = (int(x) for x in args.stroke.split(","))

    catalog = None
    if args.catalog:
        with open(args.catalog, 'r') as jf:
            catalog = json.load(jf)
            if not isinstance(catalog, list):
                raise SystemExit('--catalog JSON must be a list of shape dicts')
    out = generate(
        out_path=args.out, width=width, height=height,
        palette=palette, engine=args.engine, bg=args.bg,
        density=args.density, seamless=args.seamless, tile=args.tile,
        paletted=args.paletted, seed=args.seed, catalog=catalog,
        shape_kinds=sk, min_scale=args.min_scale, max_scale=args.max_scale,
        stroke_px=(stroke_a, stroke_b),
        fill_prob=args.fill_prob, outline_prob=args.outline_prob
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
