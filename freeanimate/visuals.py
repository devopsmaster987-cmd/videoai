"""Visual asset generation.

Two complementary strategies, both fully offline and GPU-free:

* **Illustrations** — layered PIL artwork generated from the scene's visual
  description (gradients, silhouettes, horizon, sun/moon, particles, title
  text).  Rendered at 1.35x the export resolution so ``zoompan`` always has
  headroom for camera motion without upscaling blur.

* **Motion graphics** — procedural diagrams, timelines, comparison layouts,
  and quote cards drawn with PIL and animated with the same camera vocabulary.

The module also produces a *motion spec* per scene (camera move + duration +
fps) consumed by the renderer; camera moves are never identical back-to-back
thanks to the scene planner's variety rule.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .fonts import find_font_for_script
from .scene_planner import Scene

# ---------------------------------------------------------------------------
@dataclass
class MotionSpec:
    """How the renderer should animate one still image."""

    camera_move: str
    duration_seconds: float
    fps: int
    input_w: int
    input_h: int
    out_w: int
    out_h: int
    max_zoom: float = 1.18
    min_zoom: float = 1.02

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class AssetResult:
    scene_id: str
    image_path: Path | None
    motion: MotionSpec | None
    method: str                 # animated_illustration | procedural_motion_graphics
    ok: bool
    error: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Palette derivation — deterministic per scene so repeated renders match.
# ---------------------------------------------------------------------------
def _seed_rng(scene_id: str, extra: str = "") -> random.Random:
    h = hashlib.sha256(f"{scene_id}|{extra}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def _palette_for(scene_id: str, style: str) -> dict[str, tuple[int, int, int]]:
    rng = _seed_rng(scene_id, style)
    base_hue = rng.random()
    def hsl(h: float, s: float, l: float) -> tuple[int, int, int]:
        r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
        return (int(r * 255), int(g * 255), int(b * 255))
    if style == "motion_graphics":
        # Calm, professional tones for diagrams.
        return {
            "bg_top": hsl(base_hue, 0.45, 0.16),
            "bg_bottom": hsl(base_hue + 0.08, 0.5, 0.08),
            "accent": hsl(base_hue + 0.45, 0.75, 0.58),
            "accent2": hsl(base_hue + 0.6, 0.6, 0.68),
            "ink": (240, 243, 248),
            "muted": (170, 178, 190),
        }
    return {
        "bg_top": hsl(base_hue, 0.55, 0.38),
        "bg_bottom": hsl(base_hue + 0.1, 0.6, 0.14),
        "sun": hsl(base_hue + 0.5, 0.8, 0.7),
        "hill_far": hsl(base_hue + 0.35, 0.45, 0.22),
        "hill_near": hsl(base_hue + 0.4, 0.5, 0.12),
        "accent": hsl(base_hue + 0.55, 0.7, 0.62),
        "ink": (245, 246, 250),
        "muted": (200, 205, 215),
    }


# ---------------------------------------------------------------------------
# Font loading (never assumes presence — falls back gracefully)
# ---------------------------------------------------------------------------
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _load_font(path: Path | None, size: int) -> Any:
    key = (str(path), size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font = None
    if path is not None and path.is_file():
        try:
            font = ImageFont.truetype(str(path), size)
        except OSError:
            font = None
    if font is None:
        try:
            font = ImageFont.load_default(size=size)  # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _vertical_gradient(w: int, h: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    return img


def _hill(draw: ImageDraw.ImageDraw, w: int, base_y: int, amp: int, color: tuple[int, int, int], phase: float) -> None:
    pts = [(0, base_y)]
    for x in range(0, w + 20, 20):
        y = base_y - int(amp * (0.5 + 0.5 * math.sin(x / (w / 3.0) + phase)))
        pts.append((x, y))
    pts += [(w, base_y), (w, base_y + 400), (0, base_y + 400)]
    draw.polygon(pts, fill=color)


def _wrap_text(text: str, font: Any, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = current + " " + word
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_centered_lines(
    base: Image.Image,
    lines: list[str],
    font: Any,
    color: tuple[int, int, int],
    *,
    center_y_ratio: float = 0.42,
    shadow: bool = True,
    max_line_width_ratio: float = 0.82,
) -> Image.Image:
    if not lines:
        return base
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = base.size
    max_w = int(w * max_line_width_ratio)
    line_height = int(font.size * 1.35) if hasattr(font, "size") else 24
    total_h = line_height * len(lines)
    y = int(h * center_y_ratio) - total_h // 2
    for line in lines:
        lw = draw.textlength(line, font=font)
        x = (w - lw) // 2
        if shadow:
            draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0, 160))
        draw.text((x, y), line, font=font, fill=color + (255,) if len(color) == 3 else color)
        y += line_height
    return Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")


# ---------------------------------------------------------------------------
# Illustration generator (Mode B base artwork)
# ---------------------------------------------------------------------------
def generate_illustration(
    scene: Scene,
    out_path: Path,
    *,
    width: int,
    height: int,
    script_language: str = "en",
    bake_title: bool = True,
) -> Path:
    """Draw a layered landscape/atmosphere illustration for a scene."""
    pal = _palette_for(scene.scene_id, "illustration")
    rng = _seed_rng(scene.scene_id, "art")

    img = _vertical_gradient(width, height, pal["bg_top"], pal["bg_bottom"])
    draw = ImageDraw.Draw(img)

    # Celestial body
    sun_r = int(height * rng.uniform(0.05, 0.09))
    sun_x = int(width * rng.uniform(0.18, 0.82))
    sun_y = int(height * rng.uniform(0.16, 0.34))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    for r in range(sun_r * 3, sun_r, -4):
        alpha = int(40 * (1 - (r - sun_r) / (sun_r * 2)))
        gdraw.ellipse([sun_x - r, sun_y - r, sun_x + r, sun_y + r], fill=pal["sun"] + (max(0, alpha),))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.ellipse([sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r], fill=pal["sun"])

    # Star / particle field
    for _ in range(int(width * height / 26000)):
        x, y = rng.randint(0, width - 1), rng.randint(0, int(height * 0.6))
        s = rng.randint(1, 2)
        draw.ellipse([x, y, x + s, y + s], fill=(255, 255, 255, rng.randint(40, 140)))

    # Hills (far then near)
    _hill(draw, width, int(height * 0.72), int(height * 0.10), pal["hill_far"], rng.uniform(0, 6.28))
    _hill(draw, width, int(height * 0.84), int(height * 0.08), pal["hill_near"], rng.uniform(0, 6.28))

    # Accent silhouette shapes for variety (simple geometric "structures")
    for _ in range(rng.randint(2, 4)):
        bx = rng.randint(0, width - 1)
        bw = rng.randint(int(width * 0.02), int(width * 0.06))
        bh = rng.randint(int(height * 0.06), int(height * 0.16))
        by = int(height * 0.80)
        draw.rectangle([bx, by - bh, bx + bw, by + height], fill=pal["hill_near"])

    # Title text (on-screen language) — short scene title only.
    lang_result = find_font_for_script(_script_family_for(script_language))
    title = (scene.title or "").strip()
    if title and bake_title:
        font_size = int(height * 0.052)
        font = _load_font(lang_result.path if lang_result.ok else None, font_size)
        lines = _wrap_text(title, font, int(width * 0.82), draw)[:2]
        img = _draw_centered_lines(img, lines, font, pal["ink"], center_y_ratio=0.40)

    # Vignette for cinematic feel
    vign = Image.new("L", (width, height), 0)
    vdraw = ImageDraw.Draw(vign)
    margin = int(min(width, height) * 0.08)
    vdraw.rectangle([margin // 2, margin // 2, width - margin // 2, height - margin // 2], fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(radius=max(20, width // 30)))
    dark = Image.new("RGB", (width, height), (0, 0, 0))
    img = Image.composite(img, dark, vign.point(lambda p: 128 + p // 2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


def _script_family_for(lang_code: str) -> str:
    """Best-effort writing-system family for a language code."""
    from .languages import LanguageRegistry
    try:
        lang = LanguageRegistry.builtin().require(lang_code)
        return lang.script
    except KeyError:
        return "latin"


# ---------------------------------------------------------------------------
# Motion-graphics generator (Mode D)
# ---------------------------------------------------------------------------
def generate_motion_graphic(
    scene: Scene,
    out_path: Path,
    *,
    width: int,
    height: int,
    script_language: str = "en",
    bake_title: bool = True,
) -> Path:
    """Draw a procedural diagram/quote/timeline card for a scene."""
    pal = _palette_for(scene.scene_id, "motion_graphics")
    rng = _seed_rng(scene.scene_id, "mg")

    img = _vertical_gradient(width, height, pal["bg_top"], pal["bg_bottom"])
    draw = ImageDraw.Draw(img)

    lang_result = find_font_for_script(_script_family_for(script_language))
    font_path = lang_result.path if lang_result.ok else None
    title_font = _load_font(font_path, int(height * 0.05))
    body_font = _load_font(font_path, int(height * 0.032))

    title = (scene.title or "").strip()
    hint = (scene.visual_description or "").lower()

    # Layout family chosen deterministically from the hint + seed.
    if "timeline" in hint or "origin" in hint or "development" in hint or "milestone" in hint:
        layout = "timeline"
    elif "vs" in hint or "comparison" in hint or "myth" in hint or "two-column" in hint or "balance" in hint:
        layout = "compare"
    elif "chart" in hint or "descending" in hint or "progress" in hint or "graph" in hint:
        layout = "bars"
    elif "list" in hint or "numbered" in hint or "checklist" in hint or "step" in hint:
        layout = "list"
    else:
        layout = rng.choice(("timeline", "bars", "list", "quote"))

    pad = int(width * 0.08)
    header_y = int(height * 0.10)
    if title and bake_title:
        lines = _wrap_text(title, title_font, width - 2 * pad, draw)[:2]
        for i, line in enumerate(lines):
            lw = draw.textlength(line, font=title_font)
            draw.text(((width - lw) // 2, header_y + i * int(height * 0.07)), line,
                      font=title_font, fill=pal["ink"])
        rule_y = header_y + len(lines) * int(height * 0.07) + int(height * 0.02)
    else:
        rule_y = header_y + int(height * 0.04)

    accent = pal["accent"]
    accent2 = pal["accent2"]
    muted = pal["muted"]

    if layout == "timeline":
        axis_y = int(height * 0.55)
        draw.line([(pad, axis_y), (width - pad, axis_y)], fill=accent, width=max(2, height // 220))
        nodes = 4
        for i in range(nodes):
            x = pad + int((width - 2 * pad) * i / (nodes - 1))
            r = max(6, height // 60)
            draw.ellipse([x - r, axis_y - r, x + r, axis_y + r], fill=accent2)
            stem = int(height * (0.08 + 0.04 * ((i + 1) % 2)))
            draw.line([(x, axis_y - r), (x, axis_y - r - stem)], fill=muted, width=2)
            lbl_font = _load_font(font_path, int(height * 0.026))
            label = f"0{i + 1}"
            lw = draw.textlength(label, font=lbl_font)
            draw.text((x - lw / 2, axis_y - r - stem - int(height * 0.04)), label, font=lbl_font, fill=muted)
        # Direction arrow
        draw.polygon(
            [(width - pad, axis_y), (width - pad - height // 40, axis_y - height // 90),
             (width - pad - height // 40, axis_y + height // 90)],
            fill=accent,
        )
    elif layout == "compare":
        col_w = (width - 3 * pad) // 2
        for c in range(2):
            x0 = pad + c * (col_w + pad)
            y0 = rule_y + int(height * 0.04)
            y1 = int(height * 0.82)
            draw.rounded_rectangle([x0, y0, x0 + col_w, y1], radius=height // 60,
                                   outline=accent if c == 0 else accent2, width=max(2, height // 300))
            bar_h = int(height * 0.012)
            draw.rectangle([x0 + bar_h, y0 + bar_h * 2, x0 + col_w - bar_h, y0 + bar_h * 2 + bar_h],
                           fill=accent if c == 0 else accent2)
            for r_i in range(4):
                ry = y0 + int(height * 0.08) + r_i * int(height * 0.07)
                rw = int(col_w * (0.85 - 0.1 * ((r_i * 7) % 3)))
                draw.rectangle([x0 + bar_h * 2, ry, x0 + bar_h * 2 + rw, ry + max(3, height // 140)],
                               fill=muted)
    elif layout == "bars":
        chart_top = rule_y + int(height * 0.06)
        chart_bottom = int(height * 0.80)
        n_bars = 6
        bar_slot = (width - 2 * pad) / n_bars
        bar_w = int(bar_slot * 0.55)
        heights = [rng.uniform(0.3, 1.0) for _ in range(n_bars)]
        for i, hv in enumerate(heights):
            x0 = int(pad + i * bar_slot + (bar_slot - bar_w) / 2)
            bh = int((chart_bottom - chart_top) * hv)
            color = accent if i % 2 == 0 else accent2
            draw.rounded_rectangle([x0, chart_bottom - bh, x0 + bar_w, chart_bottom],
                                   radius=max(2, bar_w // 8), fill=color)
        draw.line([(pad, chart_bottom), (width - pad, chart_bottom)], fill=muted, width=2)
    elif layout == "list":
        y = rule_y + int(height * 0.05)
        for i in range(5):
            num_font = _load_font(font_path, int(height * 0.04))
            num = str(i + 1)
            draw.ellipse([pad, y, pad + int(height * 0.05), y + int(height * 0.05)],
                         outline=accent, width=2)
            nw = draw.textlength(num, font=num_font)
            nh = int(height * 0.05)
            draw.text((pad + (nh - nw) / 2, y), num, font=num_font, fill=accent)
            lx = pad + nh + int(width * 0.02)
            lw = int((width - pad - lx) * (0.9 - 0.08 * ((i * 5) % 3)))
            ly = y + nh // 2 - max(2, height // 160)
            draw.rectangle([lx, ly, lx + lw, ly + max(3, height // 140)], fill=muted)
            y += int(height * 0.09)
    else:  # quote
        q_font = _load_font(font_path, int(height * 0.12))
        draw.text((pad, int(height * 0.22)), "“", font=q_font, fill=accent)
        body = (scene.narration_text or title or "").strip()
        words = body.split()
        snippet = " ".join(words[:14]) + ("…" if len(words) > 14 else "")
        lines = _wrap_text(snippet, body_font, width - 2 * pad - int(width * 0.06), draw)[:4]
        y = int(height * 0.36)
        for line in lines:
            lw = draw.textlength(line, font=body_font)
            draw.text(((width - lw) / 2, y), line, font=body_font, fill=pal["ink"])
            y += int(height * 0.055)

    # Bottom progress ticks — subtle motion-graphic signature.
    tick_y = int(height * 0.90)
    for i in range(12):
        x = pad + i * (width - 2 * pad) / 12
        draw.rectangle([x, tick_y, x + (width - 2 * pad) / 24, tick_y + max(2, height // 240)],
                       fill=accent if i % 3 == 0 else muted)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def generate_scene_asset(
    scene: Scene,
    assets_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
    script_language: str = "en",
    bake_title: bool = True,
) -> AssetResult:
    """Generate the still image + motion spec for one scene."""
    motion = MotionSpec(
        camera_move=scene.camera_move,
        duration_seconds=scene.duration_seconds,
        fps=fps,
        input_w=width,
        input_h=height,
        out_w=width,
        out_h=height,
    )
    png = assets_dir / f"{scene.scene_id}.png"
    try:
        if scene.style == "motion_graphics":
            generate_motion_graphic(scene, png, width=width, height=height,
                                    script_language=script_language,
                                    bake_title=bake_title)
            method = "procedural_motion_graphics"
        else:
            generate_illustration(scene, png, width=width, height=height,
                                  script_language=script_language,
                                  bake_title=bake_title)
            method = "animated_illustration"
        scene.assets["image"] = str(png)
        scene.animation_method = method
        return AssetResult(scene_id=scene.scene_id, image_path=png, motion=motion,
                           method=method, ok=True)
    except Exception as exc:  # noqa: BLE001 — asset failures must not kill the run
        return AssetResult(scene_id=scene.scene_id, image_path=None, motion=motion,
                           method="none", ok=False, error=f"{type(exc).__name__}: {exc}")
