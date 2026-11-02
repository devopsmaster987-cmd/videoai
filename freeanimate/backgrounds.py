"""Scene-specific procedural environments.

Each environment is constructed to match the requested situation rather than
reusing one generic backdrop.  These are *procedurally drawn 2D environments*
(clearly labelled as such) — not AI-generated images.  When a local AI image
model is installed the studio can optionally use it instead; this module is the
always-available, offline fallback.

Environments: garden, kitchen, beach, forest, space, school, city, castle,
mountains, room, cafe, park, underwater, snow, desert, horror_mansion, generic.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from .image_utils import blend_images, composite_images


# ---------------------------------------------------------------------------
@dataclass
class EnvironmentSpec:
    env_type: str = "generic"
    time_of_day: str = "day"
    weather: str = ""
    lighting: str = "warm"
    elements: list[str] = field(default_factory=list)
    seed: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _seed_rng(seed: str) -> random.Random:
    return random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16))


def _hsl(h: float, s: float, l: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return int(r * 255), int(g * 255), int(b * 255)


def _shade(rgb: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(v * f))) for v in rgb)  # type: ignore[return-value]


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Base sky + palette
# ---------------------------------------------------------------------------
def _sky_colors(time_of_day: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return {
        "day": ((120, 180, 235), (220, 238, 250)),
        "morning": ((255, 190, 140), (255, 235, 200)),
        "evening": ((70, 70, 130), (245, 150, 110)),
        "night": ((8, 12, 35), (30, 40, 80)),
    }.get(time_of_day, ((120, 180, 235), (220, 238, 250)))


def _vertical_gradient(w: int, h: int, top: tuple[int, int, int],
                       bottom: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        draw.line([(0, y), (w, y)], fill=_blend(top, bottom, t))
    return img


def _sun_or_moon(draw: ImageDraw.ImageDraw, w: int, h: int, time_of_day: str, rng: random.Random) -> None:
    if time_of_day == "night":
        cx, cy, r = int(w * rng.uniform(0.65, 0.85)), int(h * 0.18), int(h * 0.05)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 238, 250))
        draw.ellipse([cx - r + r // 3, cy - r, cx + r + r // 3, cy + r], fill=_sky_colors("night")[0])
    else:
        cx, cy = int(w * rng.uniform(0.2, 0.8)), int(h * rng.uniform(0.12, 0.28))
        r = int(h * 0.075)
        glow_col = (255, 220, 150) if time_of_day != "evening" else (255, 150, 90)
        for k in range(5, 0, -1):
            rr = int(r * (1 + k * 0.35))
            layer = (255, 240, 200)
            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=_blend(glow_col, layer, k * 0.06))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=glow_col)


def _stars(draw: ImageDraw.ImageDraw, w: int, h: int, rng: random.Random, count: int = 180) -> None:
    for _ in range(count):
        x = rng.randint(0, w)
        y = rng.randint(0, int(h * 0.65))
        s = rng.choice((1, 1, 1, 2))
        c = rng.randint(170, 255)
        draw.ellipse([x, y, x + s, y + s], fill=(c, c, min(255, c + 10)))


def _clouds(draw: ImageDraw.ImageDraw, w: int, h: int, rng: random.Random, n: int = 4,
            dark: bool = False) -> None:
    base = (110, 120, 140) if dark else (250, 250, 255)
    for _ in range(n):
        cx = rng.randint(0, w)
        cy = rng.randint(int(h * 0.08), int(h * 0.38))
        cw = rng.randint(w // 10, w // 4)
        ch = cw // 3
        for k in range(5):
            ox = rng.randint(-cw // 2, cw // 2)
            oy = rng.randint(-ch // 2, ch // 2)
            r = ch // 2 + rng.randint(0, ch // 3)
            draw.ellipse([cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r],
                         fill=_shade(base, 0.92 + rng.random() * 0.12))


def _hills(draw: ImageDraw.ImageDraw, w: int, h: int, base_y: int, amp: int,
           color: tuple[int, int, int], rng: random.Random) -> None:
    pts = [(0, base_y + amp)]
    phase = rng.uniform(0, 6.28)
    freq = rng.uniform(1.5, 3.5)
    for x in range(0, w + 25, 25):
        y = base_y - int(amp * (0.5 + 0.5 * math.sin(x / (w / freq) + phase)))
        pts.append((x, y))
    pts += [(w, base_y + amp), (w, h), (0, h)]
    draw.polygon(pts, fill=color)


def _tree(draw: ImageDraw.ImageDraw, x: int, ground_y: int, height: int,
          foliage: tuple[int, int, int], trunk: tuple[int, int, int], rng: random.Random) -> None:
    tw = max(3, height // 12)
    draw.rectangle([x - tw // 2, ground_y - height // 2, x + tw // 2, ground_y], fill=trunk)
    r = height // 3
    for _ in range(4):
        cx = x + rng.randint(-r // 2, r // 2)
        cy = ground_y - height // 2 + rng.randint(-r // 2, r // 2)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_shade(foliage, 0.85 + rng.random() * 0.3))


def _flower(draw: ImageDraw.ImageDraw, x: int, y: int, size: int,
            color: tuple[int, int, int]) -> None:
    for k in range(5):
        a = k * (2 * math.pi / 5)
        px = x + int(size * math.cos(a))
        py = y + int(size * math.sin(a))
        draw.ellipse([px - size // 2, py - size // 2, px + size // 2, py + size // 2], fill=color)
    draw.ellipse([x - size // 3, y - size // 3, x + size // 3, y + size // 3], fill=(250, 210, 80))


def _grass(draw: ImageDraw.ImageDraw, w: int, h: int, ground_y: int,
           color: tuple[int, int, int], rng: random.Random) -> None:
    for _ in range(w // 8):
        x = rng.randint(0, w)
        gh = rng.randint(h // 90, h // 40)
        draw.line([(x, ground_y + rng.randint(0, h // 20)), (x + rng.randint(-4, 4), ground_y - gh)],
                  fill=_shade(color, 0.8 + rng.random() * 0.4), width=1)


def _wood_table(img: Image.Image, w: int, h: int, y_ratio: float = 0.72) -> None:
    draw = ImageDraw.Draw(img)
    ty = int(h * y_ratio)
    tw = int(w * 0.46)
    th = int(h * 0.05)
    x0 = (w - tw) // 2
    wood = (150, 100, 60)
    draw.rounded_rectangle([x0, ty, x0 + tw, ty + th], radius=max(4, h // 120), fill=wood)
    draw.rectangle([x0, ty + th - max(3, h // 120), x0 + tw, ty + th], fill=_shade(wood, 0.7))
    for lx in (x0 + int(tw * 0.12), x0 + int(tw * 0.88)):
        draw.rectangle([lx - max(3, w // 160), ty + th, lx + max(3, w // 160), ty + th + int(h * 0.16)],
                       fill=_shade(wood, 0.65))


def _fountain(img: Image.Image, w: int, h: int, rng: random.Random) -> None:
    draw = ImageDraw.Draw(img)
    cx = int(w * 0.30)
    base_y = int(h * 0.80)
    rw, rh = int(w * 0.09), int(h * 0.07)
    stone = (170, 175, 185)
    draw.ellipse([cx - rw, base_y - rh, cx + rw, base_y + rh // 2], fill=stone)
    draw.ellipse([cx - rw + 8, base_y - rh + 8, cx + rw - 8, base_y + rh // 2 - 6], fill=(120, 170, 210))
    draw.rectangle([cx - w // 120, base_y - rh * 2, cx + w // 120, base_y - rh // 2], fill=_shade(stone, 1.1))
    draw.ellipse([cx - w // 80, base_y - rh * 2 - h // 60, cx + w // 80, base_y - rh * 2 + h // 60],
                 fill=_shade(stone, 1.15))
    for _ in range(10):
        a = rng.uniform(math.pi * 0.15, math.pi * 0.85)
        ln = rng.randint(h // 30, h // 16)
        draw.line([(cx, base_y - rh * 2), (cx + int(math.cos(a) * ln), base_y - rh * 2 - int(math.sin(a) * ln))],
                  fill=(200, 225, 245), width=2)


# ---------------------------------------------------------------------------
# Environment renderers
# ---------------------------------------------------------------------------
def _render_garden(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    if spec.time_of_day == "night":
        _stars(draw, w, h, rng)
    _sun_or_moon(draw, w, h, spec.time_of_day, rng)
    _clouds(draw, w, h, rng, 3)
    ground = int(h * 0.66)
    _hills(draw, w, h, ground, int(h * 0.10), _hsl(0.33, 0.45, 0.42), rng)
    grass = _hsl(0.30, 0.55, 0.45)
    draw.rectangle([0, ground, w, h], fill=grass)
    _grass(draw, w, h, ground, grass, rng)
    foliage = _hsl(0.32, 0.55, 0.32)
    trunk = _hsl(0.08, 0.45, 0.28)
    for _ in range(max(4, w // 220)):
        x = rng.randint(0, w)
        _tree(draw, x, ground + rng.randint(-h // 30, h // 25), rng.randint(h // 5, h // 3), foliage, trunk, rng)
    # Hedges.
    hedge = _hsl(0.31, 0.5, 0.30)
    draw.rounded_rectangle([0, ground - h // 30, w, ground + h // 30], radius=h // 40, fill=hedge)
    for _ in range(w // 12):
        fx = rng.randint(0, w)
        fy = rng.randint(ground + h // 40, h - h // 30)
        _flower(draw, fx, fy, max(3, h // 120), rng.choice([(240, 90, 120), (250, 200, 80), (180, 120, 240), (250, 250, 250)]))
    if "fountain" in spec.elements or True:
        _fountain(img, w, h, rng)
    _wood_table(img, w, h, 0.74)


def _render_kitchen(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    wall = _hsl(0.10, 0.18, 0.80)
    draw.rectangle([0, 0, w, int(h * 0.72)], fill=wall)
    for x in range(0, w, max(20, w // 40)):
        draw.line([(x, 0), (x, int(h * 0.72))], fill=_shade(wall, 0.95))
    # Window with daylight.
    win = [int(w * 0.60), int(h * 0.14), int(w * 0.90), int(h * 0.48)]
    draw.rectangle(win, fill=(170, 210, 240), outline=(230, 230, 235), width=max(3, h // 160))
    draw.line([(win[0], (win[1] + win[3]) // 2), (win[2], (win[1] + win[3]) // 2)], fill=(230, 230, 235), width=max(2, h // 200))
    draw.line([((win[0] + win[2]) // 2, win[1]), ((win[0] + win[2]) // 2, win[3])], fill=(230, 230, 235), width=max(2, h // 200))
    # Counter.
    cy = int(h * 0.72)
    draw.rectangle([0, cy, w, h], fill=_hsl(0.08, 0.35, 0.42))
    draw.rectangle([0, cy, w, cy + h // 40], fill=(210, 210, 215))
    # Stove + pots.
    sx = int(w * 0.18)
    draw.rounded_rectangle([sx, cy - h // 6, sx + int(w * 0.24), cy], radius=h // 60, fill=(80, 80, 88))
    for k in range(2):
        px = sx + int(w * 0.06) + k * int(w * 0.10)
        draw.ellipse([px - w // 60, cy - h // 40, px + w // 60, cy + h // 60], fill=(50, 50, 55))
        draw.rounded_rectangle([px - w // 55, cy - h // 9, px + w // 55, cy - h // 40], radius=6, fill=(170, 175, 185))
        for s in range(3):
            yy = cy - h // 9 - s * h // 60
            draw.arc([px - w // 50, yy - h // 60, px + w // 50, yy], 200, 340, fill=(240, 240, 245), width=2)
    # Shelf with jars.
    sy = int(h * 0.30)
    draw.rectangle([int(w * 0.06), sy, int(w * 0.42), sy + h // 90], fill=_hsl(0.08, 0.3, 0.35))
    for k in range(4):
        jx = int(w * 0.08) + k * int(w * 0.085)
        jc = rng.choice([(200, 90, 80), (110, 170, 110), (220, 190, 90), (150, 120, 200)])
        draw.rounded_rectangle([jx, sy - h // 22, jx + w // 42, sy], radius=4, fill=jc)
    _wood_table(img, w, h, 0.76)


def _render_beach(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    if spec.time_of_day == "night":
        _stars(draw, w, h, rng)
    _sun_or_moon(draw, w, h, spec.time_of_day, rng)
    _clouds(draw, w, h, rng, 3)
    horizon = int(h * 0.54)
    draw.rectangle([0, horizon, w, h], fill=(60, 140, 190))
    for i in range(14):
        y = horizon + int((h - horizon) * (i / 14))
        col = _blend((70, 150, 200), (125, 195, 225), i / 14)
        draw.line([(0, y), (w, y)], fill=col, width=max(2, h // 120))
        if i % 3 == 0:
            for _ in range(8):
                x = rng.randint(0, w)
                draw.arc([x - w // 40, y - h // 90, x + w // 40, y + h // 90], 200, 340, fill=(235, 245, 250), width=2)
    # Sand.
    sand_y = int(h * 0.78)
    draw.polygon([(0, sand_y), (w, sand_y - h // 40), (w, h), (0, h)], fill=(235, 215, 165))
    for _ in range(w // 30):
        x, y = rng.randint(0, w), rng.randint(sand_y, h)
        draw.ellipse([x, y, x + 3, y + 3], fill=(210, 190, 150))
    # Palm trees.
    for side in (int(w * 0.10), int(w * 0.88)):
        trunk_top = int(h * 0.36)
        draw.line([(side, sand_y + h // 30), (side + w // 30, trunk_top)], fill=(140, 100, 60), width=max(5, w // 120))
        for k in range(6):
            a = math.pi + k * (math.pi / 5)
            ex = side + w // 30 + int(math.cos(a) * w // 12)
            ey = trunk_top + int(math.sin(a) * h // 12)
            draw.arc([min(ex, side + w // 30) - 8, min(ey, trunk_top) - 8, max(ex, side + w // 30) + 8, max(ey, trunk_top) + 8],
                     190, 350, fill=(50, 130, 70), width=max(4, w // 160))


def _render_forest(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    if spec.time_of_day == "night":
        _stars(draw, w, h, rng)
    _sun_or_moon(draw, w, h, spec.time_of_day, rng)
    ground = int(h * 0.78)
    draw.rectangle([0, ground, w, h], fill=_hsl(0.25, 0.35, 0.28))
    for layer, (count, height, col) in enumerate([
        (10, int(h * 0.75), _hsl(0.32, 0.30, 0.42)),
        (8, int(h * 0.85), _hsl(0.31, 0.42, 0.30)),
        (6, int(h * 0.95), _hsl(0.30, 0.50, 0.20)),
    ]):
        for _ in range(count):
            x = rng.randint(-w // 10, w + w // 10)
            trunk = _shade(col, 0.6)
            tw = max(4, int(w * 0.012 * (layer + 1) * 0.7))
            draw.rectangle([x - tw // 2, ground - height, x + tw // 2, ground], fill=trunk)
            rr = int(height * 0.45)
            for k in range(3):
                cy = ground - height + int(k * height * 0.18)
                cr = rr - int(k * rr * 0.22)
                draw.ellipse([x - cr, cy - cr, x + cr, cy + cr], fill=_shade(col, 0.85 + k * 0.08))
    # Light shafts.
    if spec.time_of_day != "night":
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for _ in range(3):
            x = rng.randint(0, w)
            od.polygon([(x, 0), (x + w // 12, 0), (x + w // 4, ground), (x + w // 6, ground)],
                       fill=(255, 245, 200, 26))
        composited = composite_images(overlay, img)
        img.paste(composited.convert(img.mode), (0, 0))
    # Ferns.
    for _ in range(w // 40):
        x, y = rng.randint(0, w), rng.randint(ground, h)
        for k in range(4):
            draw.arc([x - 20, y - 30, x + 20, y], 200, 340, fill=(40, 110, 60), width=2)


def _render_space(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    _stars(draw, w, h, rng, count=w * h // 1800)
    # Nebula via blurred blobs.
    neb = Image.new("RGB", (w, h), (0, 0, 0))
    nd = ImageDraw.Draw(neb)
    for _ in range(5):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rr = rng.randint(w // 10, w // 4)
        col = rng.choice([(90, 60, 150), (40, 80, 150), (150, 60, 110)])
        nd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col)
    neb = neb.filter(ImageFilter.GaussianBlur(radius=max(12, w // 40)))
    # ``img`` is RGBA while the nebula is RGB: blend through the normalizer so
    # Pillow never raises "images do not match".
    blended = blend_images(img, neb, 0.55, mode="RGB")
    img.paste(blended.convert(img.mode), (0, 0))
    draw = ImageDraw.Draw(img)
    # Planets.
    for _ in range(2):
        cx, cy = rng.randint(int(w * 0.15), int(w * 0.85)), rng.randint(int(h * 0.2), int(h * 0.7))
        r = rng.randint(w // 18, w // 9)
        col = rng.choice([(200, 120, 80), (90, 140, 200), (180, 160, 210)])
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=_shade(col, 1.3), width=2)
        if rng.random() > 0.4:
            draw.arc([cx - int(r * 1.7), cy - r // 2, cx + int(r * 1.7), cy + r // 2], 190, 350,
                     fill=(230, 220, 190), width=max(3, r // 10))
    _stars(draw, w, h, rng, count=80)


def _render_school(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, h], fill=_hsl(0.12, 0.20, 0.86))
    # Blackboard.
    bx0, by0, bx1, by1 = int(w * 0.10), int(h * 0.14), int(w * 0.62), int(h * 0.52)
    draw.rectangle([bx0 - 8, by0 - 8, bx1 + 8, by1 + 8], fill=_hsl(0.08, 0.35, 0.30))
    draw.rectangle([bx0, by0, bx1, by1], fill=(38, 70, 55))
    for i in range(4):
        y = by0 + int((by1 - by0) * (0.25 + i * 0.18))
        draw.line([(bx0 + 30, y), (bx0 + 30 + rng.randint(int(w * 0.1), int(w * 0.35)), y)],
                  fill=(220, 235, 225), width=2)
    # Clock.
    ccx, ccy, cr = int(w * 0.82), int(h * 0.22), int(h * 0.055)
    draw.ellipse([ccx - cr, ccy - cr, ccx + cr, ccy + cr], fill=(250, 250, 250), outline=(90, 90, 95), width=3)
    draw.line([(ccx, ccy), (ccx, ccy - cr // 2)], fill=(40, 40, 40), width=3)
    draw.line([(ccx, ccy), (ccx + cr // 3, ccy)], fill=(40, 40, 40), width=2)
    # Floor + desks.
    floor_y = int(h * 0.60)
    draw.rectangle([0, floor_y, w, h], fill=_hsl(0.08, 0.30, 0.52))
    for row in range(2):
        for col in range(3):
            dx = int(w * (0.10 + col * 0.28))
            dy = floor_y + int(h * (0.06 + row * 0.16))
            draw.rounded_rectangle([dx, dy, dx + int(w * 0.20), dy + int(h * 0.045)], radius=4, fill=(170, 120, 70))
            draw.rectangle([dx + 8, dy + int(h * 0.045), dx + 12, dy + int(h * 0.12)], fill=(110, 110, 115))
            draw.rectangle([dx + int(w * 0.20) - 12, dy + int(h * 0.045), dx + int(w * 0.20) - 8, dy + int(h * 0.12)], fill=(110, 110, 115))


def _render_city(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    if spec.time_of_day == "night":
        _stars(draw, w, h, rng, count=100)
    # Skyline silhouettes in layers.
    layers = [(int(h * 0.42), (40, 48, 70), int(w * 0.05)),
              (int(h * 0.55), (30, 36, 56), int(w * 0.08)),
              (int(h * 0.68), (22, 26, 42), int(w * 0.10))]
    for base_y, col, min_w in layers:
        x = -20
        while x < w:
            bw = rng.randint(min_w, min_w * 3)
            bh = rng.randint(int(h * 0.12), int(h * 0.45))
            draw.rectangle([x, base_y - bh + int(h * 0.3), x + bw, h], fill=col)
            # Windows.
            for wy in range(base_y - bh + int(h * 0.32), h, max(10, h // 40)):
                for wx in range(x + 6, x + bw - 6, max(10, bw // 5)):
                    lit = rng.random() > (0.35 if spec.time_of_day == "night" else 0.8)
                    if lit:
                        draw.rectangle([wx, wy, wx + 4, wy + 5], fill=(250, 220, 120))
            x += bw + rng.randint(4, 14)
    # Street.
    draw.rectangle([0, int(h * 0.82), w, h], fill=(55, 55, 62))
    for x in range(0, w, max(40, w // 14)):
        draw.rectangle([x, int(h * 0.90), x + max(20, w // 28), int(h * 0.925)], fill=(230, 220, 130))
    # Lamps.
    for x in (int(w * 0.12), int(w * 0.5), int(w * 0.88)):
        draw.line([(x, int(h * 0.82)), (x, int(h * 0.62))], fill=(70, 70, 78), width=max(3, w // 200))
        draw.ellipse([x - w // 70, int(h * 0.60), x + w // 70, int(h * 0.62)], fill=(255, 230, 150))


def _render_castle(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    if spec.time_of_day == "night":
        _stars(draw, w, h, rng)
    _sun_or_moon(draw, w, h, spec.time_of_day, rng)
    ground = int(h * 0.72)
    _hills(draw, w, h, ground, int(h * 0.10), _hsl(0.30, 0.35, 0.40), rng)
    draw.rectangle([0, ground, w, h], fill=_hsl(0.28, 0.35, 0.35))
    stone = (150, 148, 155)
    # Castle body + towers.
    bx0, bx1 = int(w * 0.25), int(w * 0.75)
    draw.rectangle([bx0, int(h * 0.34), bx1, ground], fill=stone)
    for tx in (bx0 - int(w * 0.08), bx1 - int(w * 0.02)):
        draw.rectangle([tx, int(h * 0.22), tx + int(w * 0.10), ground], fill=_shade(stone, 0.9))
        # Crenellations.
        for k in range(5):
            cx = tx + k * int(w * 0.025)
            draw.rectangle([cx, int(h * 0.19), cx + int(w * 0.015), int(h * 0.22)], fill=_shade(stone, 0.9))
        # Roof.
        draw.polygon([(tx - 8, int(h * 0.22)), (tx + int(w * 0.05), int(h * 0.12)),
                      (tx + int(w * 0.10) + 8, int(h * 0.22))], fill=(140, 60, 70))
    # Gate.
    gx = (bx0 + bx1) // 2
    draw.rounded_rectangle([gx - w // 30, ground - int(h * 0.22), gx + w // 30, ground],
                           radius=w // 60, fill=(70, 50, 40))
    draw.arc([gx - w // 30, ground - int(h * 0.24), gx + w // 30, ground - int(h * 0.16)], 180, 360,
             fill=(70, 50, 40), width=max(6, w // 80))
    # Banners.
    for bx in (bx0 + int(w * 0.08), bx1 - int(w * 0.16)):
        draw.polygon([(bx, int(h * 0.38)), (bx + w // 40, int(h * 0.38)), (bx + w // 40, int(h * 0.56)),
                      (bx + w // 80, int(h * 0.52)), (bx, int(h * 0.56))], fill=(160, 40, 50))


def _render_mountains(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    if spec.time_of_day == "night":
        _stars(draw, w, h, rng)
    _sun_or_moon(draw, w, h, spec.time_of_day, rng)
    _clouds(draw, w, h, rng, 4)
    # Mountain ranges.
    for layer, (base_y, col, amp) in enumerate([
        (int(h * 0.52), _hsl(0.60, 0.20, 0.55), int(h * 0.22)),
        (int(h * 0.62), _hsl(0.60, 0.25, 0.40), int(h * 0.28)),
        (int(h * 0.72), _hsl(0.58, 0.30, 0.28), int(h * 0.30)),
    ]):
        pts = [(0, h)]
        step = max(20, w // 14)
        for x in range(0, w + step, step):
            peak = rng.randint(int(h * 0.05), int(h * 0.18))
            pts.append((x, base_y - peak))
        pts += [(w, h)]
        draw.polygon(pts, fill=col)
        # Snow caps.
        if layer < 2:
            for i in range(1, len(pts) - 1, 1):
                x, y = pts[i]
                if y < base_y - int(h * 0.06):
                    draw.polygon([(x - step // 3, y + h // 30), (x, y), (x + step // 3, y + h // 30)],
                                 fill=(245, 248, 252))
    draw.rectangle([0, int(h * 0.80), w, h], fill=_hsl(0.30, 0.32, 0.32))


def _render_room(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    wall = _hsl(0.08, 0.22, 0.72)
    draw.rectangle([0, 0, w, h], fill=wall)
    draw.rectangle([0, int(h * 0.68), w, h], fill=_hsl(0.08, 0.30, 0.40))
    # Window.
    wx0, wy0, wx1, wy1 = int(w * 0.60), int(h * 0.16), int(w * 0.86), int(h * 0.50)
    draw.rectangle([wx0, wy0, wx1, wy1], fill=(160, 200, 240), outline=(240, 240, 245), width=6)
    draw.line([((wx0 + wx1) // 2, wy0), ((wx0 + wx1) // 2, wy1)], fill=(240, 240, 245), width=4)
    draw.line([(wx0, (wy0 + wy1) // 2), (wx1, (wy0 + wy1) // 2)], fill=(240, 240, 245), width=4)
    # Sofa/table.
    draw.rounded_rectangle([int(w * 0.10), int(h * 0.58), int(w * 0.45), int(h * 0.66)],
                           radius=h // 40, fill=(120, 90, 140))
    draw.rectangle([int(w * 0.10), int(h * 0.66), int(w * 0.45), int(h * 0.72)], fill=(100, 75, 120))
    _wood_table(img, w, h, 0.76)
    # Lamp.
    lx = int(w * 0.90)
    draw.line([(lx, int(h * 0.76)), (lx, int(h * 0.60))], fill=(60, 60, 65), width=4)
    draw.polygon([(lx - w // 30, int(h * 0.60)), (lx + w // 30, int(h * 0.60)), (lx + w // 40, int(h * 0.54)),
                  (lx - w // 40, int(h * 0.54))], fill=(250, 225, 150))


def _render_cafe(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, h], fill=_hsl(0.07, 0.30, 0.42))
    # Warm back wall + windows.
    draw.rectangle([0, 0, w, int(h * 0.62)], fill=_hsl(0.08, 0.28, 0.55))
    for k in range(3):
        wx0 = int(w * (0.08 + k * 0.30))
        draw.rectangle([wx0, int(h * 0.10), wx0 + int(w * 0.18), int(h * 0.44)],
                       fill=(230, 200, 150), outline=(90, 60, 40), width=4)
    # Hanging lights.
    for k in range(4):
        lx = int(w * (0.14 + k * 0.24))
        draw.line([(lx, 0), (lx, int(h * 0.12))], fill=(60, 50, 45), width=2)
        draw.ellipse([lx - w // 50, int(h * 0.12), lx + w // 50, int(h * 0.16)], fill=(255, 225, 150))
    # Counter + tables.
    draw.rectangle([0, int(h * 0.62), w, int(h * 0.70)], fill=(110, 70, 45))
    for k in range(2):
        tx = int(w * (0.25 + k * 0.42))
        draw.ellipse([tx - w // 12, int(h * 0.72), tx + w // 12, int(h * 0.80)], fill=(150, 100, 60))
        draw.rectangle([tx - 4, int(h * 0.80), tx + 4, h], fill=(90, 60, 40))
        draw.rectangle([tx - w // 50, int(h * 0.70), tx + w // 50, int(h * 0.73)], fill=(235, 235, 235))


def _render_park(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    _sun_or_moon(draw, w, h, spec.time_of_day, rng)
    _clouds(draw, w, h, rng, 3)
    ground = int(h * 0.66)
    _hills(draw, w, h, ground, int(h * 0.07), _hsl(0.33, 0.40, 0.50), rng)
    draw.rectangle([0, ground, w, h], fill=_hsl(0.30, 0.55, 0.46))
    _grass(draw, w, h, ground, _hsl(0.30, 0.55, 0.46), rng)
    # Path.
    draw.polygon([(int(w * 0.35), h), (int(w * 0.48), ground), (int(w * 0.58), ground), (int(w * 0.75), h)],
                 fill=(215, 200, 165))
    for _ in range(4):
        x = rng.randint(0, w)
        _tree(draw, x, ground + rng.randint(-10, 10), rng.randint(h // 5, h // 3),
              _hsl(0.32, 0.55, 0.32), _hsl(0.08, 0.45, 0.28), rng)
    # Bench.
    bx = int(w * 0.70)
    draw.rectangle([bx, int(h * 0.74), bx + int(w * 0.16), int(h * 0.76)], fill=(130, 90, 55))
    draw.rectangle([bx, int(h * 0.70), bx + int(w * 0.16), int(h * 0.72)], fill=(130, 90, 55))
    for lx in (bx + 6, bx + int(w * 0.15)):
        draw.rectangle([lx - 2, int(h * 0.76), lx + 2, int(h * 0.82)], fill=(80, 80, 85))


def _render_underwater(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    # Water gradient overlay.
    for y in range(h):
        t = y / max(1, h - 1)
        draw_col = _blend((40, 130, 170), (10, 50, 90), t)
        ImageDraw.Draw(img).line([(0, y), (w, y)], fill=draw_col)
    draw = ImageDraw.Draw(img)
    # Light rays.
    for _ in range(5):
        x = rng.randint(0, w)
        draw.polygon([(x, 0), (x + w // 16, 0), (x + w // 5, h), (x + w // 10, h)],
                     fill=_blend((150, 220, 240), (30, 90, 130), 0.5))
    # Sand + coral.
    sy = int(h * 0.82)
    draw.ellipse([-w // 10, sy, w + w // 10, h + h // 10], fill=(210, 190, 140))
    for _ in range(10):
        cx = rng.randint(0, w)
        cy = sy + rng.randint(-h // 20, h // 40)
        ch = rng.randint(h // 20, h // 8)
        col = rng.choice([(230, 110, 130), (240, 160, 80), (160, 120, 220)])
        draw.line([(cx, cy), (cx, cy - ch)], fill=col, width=max(3, w // 120))
        for k in range(3):
            draw.line([(cx, cy - ch * (0.4 + k * 0.18)), (cx + (w // 30 if k % 2 else -w // 30), cy - ch * (0.5 + k * 0.18))],
                      fill=col, width=max(2, w // 160))
    # Bubbles + fish.
    for _ in range(w // 20):
        bx, by = rng.randint(0, w), rng.randint(0, h)
        br = rng.randint(3, 8)
        draw.ellipse([bx - br, by - br, bx + br, by + br], outline=(200, 235, 245), width=2)
    for _ in range(5):
        fx, fy = rng.randint(0, w), rng.randint(int(h * 0.2), sy)
        fcol = rng.choice([(240, 200, 90), (240, 130, 110), (150, 210, 240)])
        draw.ellipse([fx - 14, fy - 8, fx + 14, fy + 8], fill=fcol)
        draw.polygon([(fx - 14, fy), (fx - 26, fy - 8), (fx - 26, fy + 8)], fill=fcol)


def _render_snow(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, h], fill=(205, 215, 230))
    _clouds(draw, w, h, rng, 5, dark=True)
    ground = int(h * 0.66)
    _hills(draw, w, h, ground, int(h * 0.10), (230, 236, 245), rng)
    draw.rectangle([0, ground, w, h], fill=(240, 245, 252))
    # Bare trees.
    for _ in range(5):
        x = rng.randint(0, w)
        th = rng.randint(h // 5, h // 3)
        draw.line([(x, ground), (x, ground - th)], fill=(90, 70, 55), width=max(3, w // 160))
        for k in range(4):
            yy = ground - th * (0.4 + k * 0.14)
            draw.line([(x, yy), (x + rng.randint(-w // 20, w // 20), yy - h // 30)], fill=(90, 70, 55), width=2)
    # Snowman.
    sx = int(w * 0.75)
    draw.ellipse([sx - w // 22, ground - h // 16, sx + w // 22, ground + h // 30], fill=(250, 252, 255))
    draw.ellipse([sx - w // 30, ground - h // 8, sx + w // 30, ground - h // 16], fill=(250, 252, 255))
    draw.ellipse([sx - w // 40, ground - h // 7, sx + w // 40, ground - h // 8], fill=(250, 252, 255))
    # Snowflakes.
    for _ in range(w // 6):
        x, y = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(1, 3)
        draw.ellipse([x, y, x + r, y + r], fill=(255, 255, 255))


def _render_desert(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    _sun_or_moon(draw, w, h, spec.time_of_day, rng)
    horizon = int(h * 0.60)
    draw.rectangle([0, horizon, w, int(h * 0.78)], fill=(225, 175, 110))
    # Dunes.
    for layer, col in enumerate([(220, 165, 100), (205, 150, 90), (190, 135, 80)]):
        pts = [(0, h)]
        step = max(30, w // 12)
        base = horizon + int(h * 0.05 * layer)
        for x in range(0, w + step, step):
            pts.append((x, base - int(h * 0.05 * math.sin(x / (w / 2) + layer))))
        pts += [(w, h)]
        draw.polygon(pts, fill=col)
    # Cactus.
    for _ in range(2):
        cx = rng.randint(int(w * 0.1), int(w * 0.9))
        cy = int(h * 0.80)
        col = (70, 120, 70)
        draw.rounded_rectangle([cx - w // 90, cy - h // 5, cx + w // 90, cy], radius=w // 90, fill=col)
        draw.rounded_rectangle([cx - w // 22, cy - h // 9, cx - w // 90, cy - h // 9 + h // 30],
                               radius=w // 120, fill=col)
        draw.rounded_rectangle([cx + w // 90, cy - h // 8, cx + w // 22, cy - h // 8 + h // 30],
                               radius=w // 120, fill=col)


def _render_horror(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    _stars(draw, w, h, rng, count=60)
    _sun_or_moon(draw, w, h, "night", rng)
    # Fog.
    fog = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fog)
    for _ in range(5):
        cx, cy = rng.randint(0, w), rng.randint(int(h * 0.4), h)
        rr = rng.randint(w // 5, w // 2)
        fd.ellipse([cx - rr, cy - rr // 3, cx + rr, cy + rr // 3], fill=(150, 155, 170, 40))
    fog = fog.filter(ImageFilter.GaussianBlur(radius=max(20, w // 30)))
    img.paste(composite_images(fog, img).convert(img.mode), (0, 0))
    draw = ImageDraw.Draw(img)
    ground = int(h * 0.80)
    draw.rectangle([0, ground, w, h], fill=(28, 30, 36))
    # Mansion silhouette.
    mx = int(w * 0.55)
    draw.polygon([(mx - w // 5, ground), (mx - w // 5, int(h * 0.38)), (mx, int(h * 0.24)),
                  (mx + w // 5, int(h * 0.38)), (mx + w // 5, ground)], fill=(18, 20, 26))
    for k in range(3):
        wx = mx - w // 8 + k * w // 10
        draw.rectangle([wx, int(h * 0.48), wx + w // 30, int(h * 0.56)], fill=(180, 150, 70))
    # Dead trees.
    for _ in range(4):
        x = rng.randint(0, int(w * 0.5))
        th = rng.randint(h // 6, h // 3)
        draw.line([(x, ground), (x + rng.randint(-10, 10), ground - th)], fill=(40, 40, 45), width=max(3, w // 150))
        for k in range(3):
            yy = ground - th * (0.45 + k * 0.16)
            draw.line([(x, yy), (x + rng.randint(-w // 16, w // 16), yy - h // 24)], fill=(40, 40, 45), width=2)


def _render_generic(img: Image.Image, spec: EnvironmentSpec, rng: random.Random) -> None:
    w, h = img.size
    draw = ImageDraw.Draw(img)
    if spec.time_of_day == "night":
        _stars(draw, w, h, rng)
    _sun_or_moon(draw, w, h, spec.time_of_day, rng)
    _clouds(draw, w, h, rng, 3)
    ground = int(h * 0.70)
    _hills(draw, w, h, ground, int(h * 0.12), _hsl(0.55, 0.35, 0.45), rng)
    _hills(draw, w, h, ground + int(h * 0.06), int(h * 0.08), _hsl(0.58, 0.40, 0.35), rng)
    draw.rectangle([0, ground + int(h * 0.06), w, h], fill=_hsl(0.30, 0.40, 0.40))


_RENDERERS = {
    "garden": _render_garden,
    "kitchen": _render_kitchen,
    "beach": _render_beach,
    "forest": _render_forest,
    "space": _render_space,
    "school": _render_school,
    "city": _render_city,
    "castle": _render_castle,
    "mountains": _render_mountains,
    "room": _render_room,
    "cafe": _render_cafe,
    "park": _render_park,
    "underwater": _render_underwater,
    "snow": _render_snow,
    "desert": _render_desert,
    "horror_mansion": _render_horror,
    "generic": _render_generic,
}


def render_environment(spec: EnvironmentSpec, out_path: Path, *, width: int = 1920,
                       height: int = 1080) -> Path:
    """Render an environment to a PNG, with a darkening pass for lighting mood."""
    renderer = _RENDERERS.get(spec.env_type, _render_generic)
    rng = _seed_rng(f"{spec.env_type}|{spec.time_of_day}|{spec.seed or ''}")
    top, bottom = _sky_colors(spec.time_of_day)
    img = _vertical_gradient(width, height, top, bottom)
    img = img.convert("RGBA")
    renderer(img, spec, rng)
    img = img.convert("RGB")

    # Lighting mood pass.
    if spec.lighting == "dramatic" or spec.time_of_day == "night":
        from PIL import ImageEnhance

        img = ImageEnhance.Brightness(img).enhance(0.82 if spec.time_of_day != "night" else 0.66)
        img = ImageEnhance.Contrast(img).enhance(1.12)
    elif spec.lighting == "warm":
        overlay = Image.new("RGB", img.size, (255, 190, 120))
        img = blend_images(img, overlay, 0.10)
    elif spec.lighting == "cool":
        overlay = Image.new("RGB", img.size, (120, 160, 220))
        img = blend_images(img, overlay, 0.10)

    # Subtle vignette for cinematic framing.
    vign = Image.new("L", (width, height), 0)
    vd = ImageDraw.Draw(vign)
    m = int(min(width, height) * 0.06)
    vd.rectangle([m // 2, m // 2, width - m // 2, height - m // 2], fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(radius=max(20, width // 30)))
    dark = Image.new("RGB", (width, height), (0, 0, 0))
    img = composite_images(img, dark,
                           mask=vign.point(lambda p: 120 + p * 135 // 255)).convert("RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def environment_for_plan(plan: Any, seed: str = "") -> EnvironmentSpec:
    """Build an EnvironmentSpec from a parsed PromptPlan (duck-typed)."""
    return EnvironmentSpec(
        env_type=getattr(plan, "environment_type", "generic") or "generic",
        time_of_day=getattr(plan, "time_of_day", "day") or "day",
        weather=getattr(plan, "weather", "") or "",
        lighting=getattr(plan, "lighting", "warm") or "warm",
        elements=list(getattr(plan, "background_elements", []) or []),
        seed=seed or getattr(plan, "prompt", "")[:80],
    )


def render_cached(spec: EnvironmentSpec, cache_dir: Path, *, width: int, height: int) -> Path:
    """Render to a deterministic cache path; reuse when already present."""
    key = hashlib.sha256(
        f"{spec.env_type}|{spec.time_of_day}|{spec.lighting}|{spec.weather}|{spec.seed}|{width}x{height}".encode()
    ).hexdigest()[:16]
    out = cache_dir / f"env_{spec.env_type}_{key}.png"
    if out.exists() and out.stat().st_size > 0:
        return out
    return render_environment(spec, out, width=width, height=height)
