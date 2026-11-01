"""Thumbnail generation, scoring, recommendation, and selection (section 12).

Everything in this module is fully offline, deterministic, and GPU-free. It
generates three distinct YouTube thumbnail options from the project's existing
scene assets and metadata, scores each using documented heuristics, recommends
the strongest one that passes quality checks, and supports user selection and
regeneration without re-rendering the video.

Key public API:

    from freeanimate.thumbnails import (
        analyze_project, generate_thumbnails, validate_thumbnail,
        select_thumbnail, regenerate_thumbnail, write_recommendation_json,
        render_shaped_text_to_png,
    )
    from freeanimate.constants import THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    THUMBNAIL_COUNT,
    THUMBNAIL_HEIGHT,
    THUMBNAIL_JPEG_QUALITY,
    THUMBNAIL_WIDTH,
)

# Thumbnail text budget: short enough to stay readable at small sizes.
SUBTITLE_TEXT_MAX_WORDS = 5


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

def _script_family_for(lang_code: str) -> str:
    """Best-effort writing-system family for a language code."""
    from .languages import LanguageRegistry
    try:
        lang = LanguageRegistry.builtin().require(lang_code)
        return lang.script
    except KeyError:
        return "latin"

def _rtl(lang_code: str) -> bool:
    from .languages import LanguageRegistry
    try:
        return LanguageRegistry.builtin().require(lang_code).rtl
    except KeyError:
        return False

# ---------------------------------------------------------------------------
# Project analysis
# ---------------------------------------------------------------------------

def _extract_keywords(text: str) -> set[str]:
    """Extract significant keyword tokens from text (lowercased, stopwords removed)."""
    stop = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "for",
            "nor", "so", "yet", "as", "until", "while", "of", "at", "by", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "s", "t", "can", "will", "just", "don", "should", "now"}
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return {t for t in tokens if t not in stop}

def analyze_project(project: Project, langs: Any, script: Script | None = None,
                  scene_manifest: SceneManifest | None = None,
                  config: dict | None = None) -> dict:
    """Analyze a project to produce thumbnail analysis data.

    Returns a dict with keys: title, topic, keywords, content_kind, scenes,
    mood, language, rtl, thumbnail_language, font_family, font_needs_shaping.
    """
    from .script_engine import extract_topic

    title = project.manifest.get("name", "")
    prompt = project.manifest.get("prompt", "")
    topic = script.topic if script else extract_topic(prompt)

    # Gather keywords from title + topic + scene titles + narration snippets.
    all_text = f"{title} {topic}"
    if script:
        for sec in script.sections:
            all_text += f" {sec.title} {sec.narration}"
        keywords = _extract_keywords(all_text)
    else:
        keywords = _extract_keywords(topic)

    # Content kind from script (if available) or default.
    content_kind = script.kind if script else "documentary"

    # Mood mapping from content kind.
    mood_map = {
        "documentary": "dramatic", "science": "wonder", "explainer": "clear",
        "story": "mysterious", "tutorial": "practical", "listicle": "energetic",
        "opinion": "bold",
    }
    mood = mood_map.get(content_kind, "informative")

    # Scene list with asset paths.
    scenes = []
    if scene_manifest:
        for s in scene_manifest.scenes:
            img_path = s.assets.get("image", "")
            scenes.append({
                "scene_id": s.scene_id,
                "title": s.title,
                "narration_text": s.narration_text,
                "duration_seconds": s.duration_seconds,
                "image_path": img_path,
                "camera_move": s.camera_move,
                "style": s.style,
                "animation_method": s.animation_method,
            })

    # Thumbnail language.
    thumb_lang_code = langs.resolved("thumbnail") if langs else "en"
    rtl = _rtl(thumb_lang_code)
    font_family = _script_family_for(thumb_lang_code)
    needs_shaping = False
    try:
        from .fonts import script_needs_shaping
        needs_shaping = script_needs_shaping(font_family)
    except Exception:
        needs_shaping = False

    analysis = {
        "title": title,
        "topic": topic,
        "keywords": keywords,
        "content_kind": content_kind,
        "mood": mood,
        "scenes": scenes,
        "language": thumb_lang_code,
        "rtl": rtl,
        "thumbnail_language": thumb_lang_code,
        "font_family": font_family,
        "font_needs_shaping": needs_shaping,
    }
    if config:
        analysis["config"] = config
    return analysis

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_LUM_CACHE: dict[str, "Any"] = {}

def _luminance(img):
    """Float32 luminance array in [0, 1] for scoring/quality math."""
    import numpy as np
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) / 255.0


def _cover_crop(img, width, height):
    """Resize + centre-crop to exactly cover width x height (no distortion)."""
    w, h = img.size
    scale = max(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh))
    left = (nw - width) // 2
    top = (nh - height) // 2
    return img.crop((left, top, left + width, top + height))


def _vertical_gradient(size, top, bottom):
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        d.line([(0, y), (w, y)], fill=tuple(
            int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    return img


def _scrim(img, direction="bottom", strength=0.75, extent=0.55):
    """Darken a region of the image with a gradient scrim (for text legibility)."""
    from PIL import Image
    w, h = img.size
    base = img.convert("RGB")
    if direction == "bottom":
        grad = _vertical_gradient((1, int(h * extent)), (0, 0, 0), (0, 0, 0))
        mask = Image.new("L", (w, h), 0)
        px = mask.load()
        start = h - int(h * extent)
        for y in range(start, h):
            v = int(255 * strength * ((y - start) / max(1, h - 1 - start)))
            for x in range(w):
                px[x, y] = v
        black = Image.new("RGB", (w, h), (0, 0, 0))
        base = Image.composite(black, base, mask)
    return base


def _font_for(lang_code, size):
    from .fonts import find_font_for_script
    res = find_font_for_script(_script_family_for(lang_code))
    from PIL import ImageFont
    if res.ok and res.path and res.path.is_file():
        try:
            return ImageFont.truetype(str(res.path), size), res.path
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size), None
    except TypeError:
        return ImageFont.load_default(), None


def render_text_image(text, *, lang_code, size, color, stroke=4, max_width):
    """Render multi-line text into a transparent RGBA image (Pillow path).

    Returns (rgba_image, bbox) or (None, None) when empty.  Used for scripts
    Pillow can draw directly (latin, cyrillic, cjk).  For shaping scripts the
    caller must use the libass route instead (b.txt 12.5/4.4).
    """
    from PIL import Image, ImageDraw
    font, _ = _font_for(lang_code, size)
    probe = Image.new("RGB", (max_width, 10))
    d = ImageDraw.Draw(probe)
    lines = []
    for raw in text.split("\n"):
        words = raw.split()
        cur = ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if d.textlength(trial, font=font) <= max_width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
    if not lines:
        return None, None
    line_h = int(size * 1.3)
    pad = stroke * 2 + 4
    wps = [int(d.textlength(ln, font=font)) + 2 * stroke for ln in lines]
    w = max(wps) + pad * 2
    h = line_h * len(lines) + pad * 2
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y = pad
    for ln in lines:
        lw = d.textlength(ln, font=font)
        x = (w - lw) // 2
        d.text((x, y), ln, font=font, fill=color + (255,), stroke_width=stroke,
               stroke_fill=(0, 0, 0, 220))
        y += line_h
    return layer, layer.getbbox()


def _render_shaped_text_layer(text, *, lang_code, size, color, work_dir, max_width):
    """Render shaped (RTL/complex) text via FFmpeg + libass.

    Output: white glyphs on black PNG; converted to a coloured RGBA layer.
    Returns (rgba_image, ok).  ok=False means "could not verify correct
    rendering" — per b.txt the caller then prefers a text-free thumbnail.
    """
    from pathlib import Path as _P
    from .ffmpeg_utils import run_ffmpeg, FFmpegError, has_filter
    from .fonts import find_font_for_script, family_name_for
    import re as _re

    if not has_filter("subtitles"):
        return None, False
    work_dir = _P(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    res = find_font_for_script(_script_family_for(lang_code))
    family = family_name_for(res.font.filename) if res.ok else "Arial"
    # ASS with one centred dialogue line over a transparent-safe black screen.
    playres = 1280
    lines = text.replace("\n", "\\N")
    rtl_mark = "\u202b" if _rtl(lang_code) else ""
    ass_text = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\nPlayResX: %d\nPlayResY: 720\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: T,%s,%d,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,0,0,5,20,20,20,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:10.00,T,,0,0,0,,%s%s\n"
    ) % (playres, family, size, rtl_mark, lines)
    ass_path = work_dir / "thumb_text.ass"
    png_path = work_dir / "thumb_text_mask.png"
    ass_path.write_text(ass_text, encoding="utf-8")
    try:
        run_ffmpeg([
            "-y", "-f", "lavfi", "-i",
            "color=c=black:s=%dx720:d=0.2" % playres,
            "-vf", "subtitles=%s" % ass_path.name,
            "-frames:v", "1", str(png_path),
        ], timeout=60, cwd=str(work_dir))
    except (FFmpegError, OSError):
        return None, False
    finally:
        try:
            ass_path.unlink()
        except OSError:
            pass
    if not png_path.exists() or png_path.stat().st_size == 0:
        return None, False
    from PIL import Image
    import numpy as np
    with Image.open(png_path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    try:
        png_path.unlink()
    except OSError:
        pass
    frac = float((arr > 40).sum()) / arr.size
    if frac < 0.002:  # blank output => shaping/font failed; do not ship text
        return None, False
    mask = Image.fromarray(arr.astype("uint8"), "L")
    layer = Image.new("RGBA", mask.size, color + (255,))
    layer.putalpha(mask)
    bbox = layer.getbbox() if layer else None
    if bbox is None or (bbox[2] - bbox[0]) > max_width * 1.02:
        return None, False
    # Crop to the glyphs: the libass canvas is full-frame with the text
    # centred, so pasting it untruncated would drop the text in the middle of
    # the thumbnail (overlapping scene titles) instead of at the designed spot.
    return layer.crop(bbox), True

# ---------------------------------------------------------------------------
# Source image selection + text budget
# ---------------------------------------------------------------------------

def _scene_images(analysis) -> list:
    """Cover-cropped scene asset paths that actually exist, in scene order."""
    imgs = []
    for sc in analysis.get("scenes", []):
        p = sc.get("image_path") or ""
        if p:
            from pathlib import Path as _P
            fp = _P(p)
            if fp.is_file():
                imgs.append(fp)
    return imgs


def _keyword_hits(text_words, keywords) -> int:
    words = set(text_words)
    return len(words & set(keywords))


def _short_text(analysis, *, max_words: int, short: bool = False) -> str:
    """Concise thumbnail text derived from the video title (12.5).

    No invented claims: reuses only words actually present in the title/topic.
    Trailing prepositions/articles are dropped so a budget cut never leaves a
    dangling phrase like "...form in under".
    """
    import re
    title = (analysis.get("title") or analysis.get("topic") or "").strip()
    if not title:
        return ""
    # Split on connectors; keep the most informative first clause.
    title = re.split(r"[—:|-]", title)[0].strip() or title
    words = title.split()
    n = min(max_words, 3) if short else min(max_words, 6)
    cut = words[:n]
    # Strip dangling function words from the end of the truncated phrase.
    while len(cut) > 1 and cut[-1].lower() in _DANGLING_TAIL:
        cut.pop()
    return " ".join(cut)


# Words that read as incomplete when a phrase is cut right after them.
_DANGLING_TAIL = {"in", "on", "at", "to", "of", "for", "with", "and", "or",
                  "the", "a", "an", "under", "over", "into", "from", "by",
                  "is", "are", "was", "were", "be", "it", "its"}


def _kind_tag(analysis) -> str:
    kind = analysis.get("content_kind") or "video"
    return kind.upper()


# ---------------------------------------------------------------------------
# Concept builders (section 12.2) — three genuinely distinct designs
# ---------------------------------------------------------------------------

def _thumb1_high_appeal(analysis, work_dir, *, text: str, size, quality):
    """Thumbnail 1: strong focal subject, bold large text, high contrast."""
    from PIL import Image, ImageDraw, ImageEnhance
    w, h = size
    imgs = _scene_images(analysis)
    if not imgs:
        raise RuntimeError("no scene assets available")
    hero = Image.open(imgs[len(imgs) // 2]).convert("RGB")
    img = _cover_crop(hero, w, h)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Color(img).enhance(1.12)

    # Dark scrim on the lower half for text legibility.
    img = _scrim(img, strength=0.72, extent=0.58)

    accent = (255, 170, 0)
    tag = _kind_tag(analysis)
    d = ImageDraw.Draw(img)
    tag_size = max(22, int(h * 0.045))
    tag_font, _ = _font_for(analysis["thumbnail_language"], tag_size)
    tag_pad_x, tag_pad_y = int(w * 0.022), int(h * 0.018)
    tw = d.textlength(tag, font=tag_font)
    tx, ty = int(w * 0.055), int(h * 0.09)
    d.rounded_rectangle([tx, ty, tx + tw + 2 * tag_pad_x, ty + tag_size + 2 * tag_pad_y],
                        radius=8, fill=accent)
    d.text((tx + tag_pad_x, ty + tag_pad_y), tag, font=tag_font, fill=(20, 20, 24))

    if text:
        tsize = int(h * 0.128)
        layer, bbox = render_text_image(
            text, lang_code=analysis["thumbnail_language"], size=tsize,
            color=(255, 255, 255), stroke=max(3, tsize // 14), max_width=int(w * 0.86))
        if layer is not None:
            lw, lh = layer.size
            img.paste(layer, (int(w * 0.05), h - lh - int(h * 0.06)), layer)
        else:
            text = ""  # honest fallback: text-free

    # Accent edge bar for punch.
    d = ImageDraw.Draw(img)
    d.rectangle([0, h - max(6, int(h * 0.014)), w, h], fill=accent)
    return img, text


def _thumb2_cinematic(analysis, work_dir, *, text: str, size, quality):
    """Thumbnail 2: cinematic composition — soft background, framed inset."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    w, h = size
    imgs = _scene_images(analysis)
    if not imgs:
        raise RuntimeError("no scene assets available")
    bg = Image.open(imgs[0]).convert("RGB")
    bg = _cover_crop(bg, w, h)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=max(6, w // 90)))
    bg = ImageEnhance.Brightness(bg).enhance(0.55)
    bg = ImageEnhance.Contrast(bg).enhance(0.95)
    img = bg

    inset_src = Image.open(imgs[-1]).convert("RGB")
    iw = int(w * 0.60)
    ih = int(iw * 9 / 16)
    inset = _cover_crop(inset_src, iw, ih)
    inset = ImageEnhance.Contrast(inset).enhance(1.05)
    ix, iy = (w - iw) // 2, int(h * 0.10)
    shadow_off = max(6, w // 200)
    d = ImageDraw.Draw(img)
    d.rectangle([ix + shadow_off, iy + shadow_off, ix + iw + shadow_off, iy + ih + shadow_off],
                fill=(8, 8, 12))
    img.paste(inset, (ix, iy))
    d = ImageDraw.Draw(img)
    border = max(3, h // 240)
    d.rectangle([ix - border, iy - border, ix + iw + border, iy + ih + border],
                outline=(232, 225, 205), width=border)
    # Thin cinematic base line (not a black bar).
    d.rectangle([int(w * 0.18), h - max(4, int(h * 0.01)),
                 int(w * 0.82), h - max(2, int(h * 0.006))], fill=(232, 225, 205))

    if text:
        tsize = int(h * 0.075)
        layer, bbox = render_text_image(
            text, lang_code=analysis["thumbnail_language"], size=tsize,
            color=(245, 240, 228), stroke=max(2, tsize // 16), max_width=int(w * 0.8))
        if layer is not None:
            lw, lh = layer.size
            img.paste(layer, ((w - lw) // 2, h - lh - int(h * 0.07)), layer)
        else:
            text = ""
    return img, text


def _thumb3_creative(analysis, work_dir, *, text: str, size, quality):
    """Thumbnail 3: creative split-panel collage with geometric accents."""
    from PIL import Image, ImageDraw, ImageEnhance
    w, h = size
    imgs = _scene_images(analysis)
    if not imgs:
        raise RuntimeError("no scene assets available")
    left = _cover_crop(Image.open(imgs[0]).convert("RGB"), int(w * 0.58), h)
    right_src = imgs[1] if len(imgs) > 1 else imgs[0]
    right = _cover_crop(Image.open(right_src).convert("RGB"), w - int(w * 0.58), h)
    left = ImageEnhance.Color(left).enhance(1.18)
    right = ImageEnhance.Brightness(right).enhance(0.9)
    img = Image.new("RGB", (w, h))
    img.paste(left, (0, 0))
    img.paste(right, (int(w * 0.58), 0))

    d = ImageDraw.Draw(img)
    # Diagonal accent divider between panels.
    dx, accent = int(w * 0.58), (60, 200, 160)
    pts = [(dx - int(w * 0.02), 0), (dx + int(w * 0.02), 0),
           (dx - int(w * 0.02), h), (dx - int(w * 0.06), h)]
    d.polygon(pts, fill=accent)
    # Geometric accent circles in a corner.
    r0 = int(h * 0.10)
    cx, cy = int(w * 0.87), int(h * 0.18)
    d.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], outline=accent, width=max(6, h // 90))
    d.ellipse([cx - r0 // 2, cy - r0 // 2, cx + r0 // 2, cy + r0 // 2], fill=accent)

    if text:
        tsize = int(h * 0.105)
        layer, bbox = render_text_image(
            text, lang_code=analysis["thumbnail_language"], size=tsize,
            color=(255, 255, 255), stroke=max(3, tsize // 13), max_width=int(w * 0.5))
        if layer is not None:
            lw, lh = layer.size
            img.paste(layer, (int(w * 0.035), h - lh - int(h * 0.07)), layer)
        else:
            text = ""
    else:
        text = ""
    return img, text

# ---------------------------------------------------------------------------
# Scoring (section 12.6) — documented heuristic measurements, not random values
# ---------------------------------------------------------------------------
SCORING_CRITERIA = {
    "topic_relevance": 25,
    "visual_clarity": 20,
    "text_readability": 15,
    "contrast": 10,
    "composition": 10,
    "audience_relevance": 10,
    "distinctiveness": 5,
    "honesty_and_accuracy": 5,
}
SCORING_DISCLAIMER = ("Scores are heuristic design assessments measured from the image "
                      "pixels and project metadata. They are NOT predictions of YouTube "
                      "click-through rate, views, or subscriber growth.")


def _edge_density(img) -> float:
    import numpy as np
    from PIL import ImageFilter
    edges = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    return float(edges.mean())


def _grid_energy(img) -> list[float]:
    import numpy as np
    L = _luminance(img)
    grad_x = np.abs(np.diff(L, axis=1))
    grad_y = np.abs(np.diff(L, axis=0))
    energy = grad_x[:-1, :] + grad_y[:, :-1]
    h, w = energy.shape
    cells = []
    for r in range(3):
        for c in range(3):
            cell = energy[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3]
            cells.append(float(cell.mean()))
    total = sum(cells) or 1.0
    return [v / total for v in cells]


def _michelson(L) -> float:
    import numpy as np
    hi, lo = np.percentile(L, 95), np.percentile(L, 5)
    return float((hi - lo) / max(1e-6, hi + lo))


def score_topic_relevance(img, analysis, text) -> tuple[int, str]:
    """25 pts: 16 from thumbnail-text keyword overlap, 5 from project-scene usage,
    4 from hero/title keyword overlap. All inputs are real project data."""
    text_words = (text or "").lower().split()
    overlap = _keyword_hits(text_words, analysis["keywords"])
    words_n = max(1, len(set(text_words)))
    pts_text = 16 * min(1.0, overlap / min(3, max(1, words_n))) if text_words else 0
    pts_scenes = 5.0  # hero/faces come from the video's own rendered scenes
    title_words = _extract_keywords_for(analysis["title"])
    img_pool = " ".join(s.get("title", "") for s in analysis["scenes"])
    pts_match = 4.0 if _keyword_hits(list(title_words), analysis["keywords"]) else 2.0
    del img_pool
    pts = pts_text + pts_scenes + pts_match
    note = f"text overlap {overlap}/{words_n}"
    return int(round(min(SCORING_CRITERIA["topic_relevance"], pts))), note


def _extract_keywords_for(text: str) -> set[str]:
    return set(_extract_keywords(text))


def score_visual_clarity(img, text_bbox) -> tuple[int, str]:
    """20 pts: mid-range edge density is ideal; blank (<0.008) or cluttered
    (>0.30) images score low."""
    d = _edge_density(img)
    if d < 0.008:
        return 4, f"nearly blank (edge density {d:.4f})"
    ideal = 0.10
    pts = 20 * (1.0 - min(1.0, abs(d - ideal) / 0.25))
    return int(round(pts)), f"edge density {d:.3f}"


def score_text_readability(img, text, text_mask_info) -> tuple[int, str]:
    import numpy as np
    """15 pts: text-free thumbs get a neutral 9/15 (allowed by 12.5 when
    rendering cannot be verified).  Text thumbs are measured on local contrast
    between glyph pixels and the region behind them."""
    if not text:
        return 9, "text-free (neutral)"
    L = _luminance(img)
    bbox = text_mask_info.get("bbox") if text_mask_info else None
    if not bbox:
        return 6, "text bbox unavailable"
    x0, y0, x1, y1 = bbox
    pad = 12
    region = L[max(0, y0 - pad):min(L.shape[0], y1 + pad),
               max(0, x0 - pad):min(L.shape[1], x1 + pad)]
    if region.size < 50:
        return 6, "text region too small"
    # Glyph pixels = light pixels inside region; background = the darkest half.
    hi, lo = np.percentile(region, 90), np.percentile(region, 40)
    local_contrast = abs(float(hi) - float(lo))
    pts = 15 * min(1.0, local_contrast / 0.55)
    return int(round(pts)), f"local contrast {local_contrast:.2f}"


def score_contrast(img) -> tuple[int, str]:
    import numpy as np
    """10 pts from Michelson contrast (5-95 percentile luminance)."""
    m = _michelson(_luminance(img))
    pts = 10 * min(1.0, m / 0.75)
    return int(round(pts)), f"michelson {m:.2f}"


def score_composition(img, text_bbox) -> tuple[int, str]:
    import numpy as np
    """10 pts: 6 for focal-cell alignment with a thirds intersection,
    4 for balanced (not single-cell-dominated) energy spread."""
    cells = _grid_energy(img)
    focal = int(np.argmax(cells))
    row, col = divmod(focal, 3)
    thirds = {(0, 0), (0, 2), (2, 0), (2, 2), (1, 0), (1, 2), (0, 1), (2, 1)}
    pts_align = 6.0 if (row, col) in thirds else (3.0 if (row, col) == (1, 1) else 4.0)
    top_share = max(cells)
    pts_balance = 4.0 * max(0.0, 1.0 - (top_share - 1 / 9) / 0.55)
    pts = pts_align + pts_balance
    return int(round(pts)), f"focal cell r{row}c{col}"


def score_audience_relevance(analysis, concept) -> tuple[int, str]:
    """10 pts: documented mood/concept affinity heuristic for the content kind."""
    kind = analysis.get("content_kind", "documentary")
    affinity = {
        "documentary": {"cinematic": 10, "high_appeal": 8, "creative": 7},
        "science": {"creative": 9, "cinematic": 8, "high_appeal": 8},
        "explainer": {"high_appeal": 10, "creative": 8, "cinematic": 7},
        "story": {"cinematic": 9, "creative": 9, "high_appeal": 7},
        "tutorial": {"high_appeal": 9, "creative": 8, "cinematic": 7},
        "listicle": {"high_appeal": 10, "creative": 8, "cinematic": 6},
        "opinion": {"high_appeal": 9, "creative": 8, "cinematic": 7},
    }
    pts = affinity.get(kind, {}).get(concept, 7)
    return pts, f"{kind} -> {concept}"


def score_distinctiveness(img, others) -> tuple[int, str]:
    import numpy as np
    """5 pts: mean absolute grayscale difference vs the OTHER two options."""
    import numpy as np
    if not others:
        return 3, "first reference"
    a = np.asarray(img.convert("L").resize((64, 36)), dtype=np.float32) / 255.0
    diffs = [float(np.mean(np.abs(a - np.asarray(o.convert("L").resize((64, 36)),
                                                dtype=np.float32) / 255.0))) for o in others]
    mad = sum(diffs) / len(diffs)
    pts = 5 * min(1.0, mad / 0.22)
    return int(round(pts)), f"mean diff {mad:.3f}"


_HYPE_WORDS = {"guaranteed", "shocking", "miracle", "unbelievable", "insane",
               "cure", "#1", "best ever", "you won\u2019t believe"}

def score_honesty(analysis, text) -> tuple[int, str]:
    """5 pts: penalises numbers or hype words absent from title+script
    (no fabricated statistics or promises in thumbnail text)."""
    pts = 5.0
    if text:
        source_words = {w.lower() for w in analysis.get("title", "").split()}
        source_words |= {w.lower() for w in analysis.get("topic", "").split()}
        text_words = [w.lower().strip("!.,?") for w in text.split()]
        nums = [w for w in text_words if any(ch.isdigit() for ch in w)]
        if nums and not any(n in source_words for n in nums):
            pts -= 3
        for hw in _HYPE_WORDS:
            if hw in text_words and hw not in source_words:
                pts -= 2
    return int(round(max(0.0, pts))), "no invented claims" if pts >= 4 else "invented wording penalised"

# ---------------------------------------------------------------------------
# Quality checks (sections 12.4 + 17) — critical defects reject a thumbnail
# ---------------------------------------------------------------------------

def validate_thumbnail(path, *, expected_size=(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)) -> dict:
    """Validate one thumbnail file. Returns dict with:
    ok, defects (critical), stats, checks (per-check detail strings)."""
    import numpy as np
    from PIL import Image, ImageFile
    out = {"ok": False, "defects": [], "stats": {}, "checks": []}
    p = Path(path)
    if not p.exists():
        out["defects"].append("missing_file")
        out["checks"].append(("file_exists", False, str(p)))
        return out
    out["checks"].append(("file_exists", True, str(p)))
    size_bytes = p.stat().st_size
    if size_bytes < 3000:
        out["defects"].append("file_too_small")
        out["checks"].append(("file_not_trivial", False, f"{size_bytes} bytes"))
        return out
    out["checks"].append(("file_not_trivial", True, f"{size_bytes:,} bytes"))
    try:
        with Image.open(p) as im:
            im.load()
            fmt = im.format
            mode = im.mode
            dims = im.size
            rgb = im.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        out["defects"].append("unreadable_image")
        out["checks"].append(("image_decodes", False, str(exc)[:120]))
        return out
    out["checks"].append(("image_decodes", True, fmt))
    if fmt != "JPEG":
        out["defects"].append("not_jpeg")
    if dims != expected_size:
        out["defects"].append(f"wrong_dimensions_{dims[0]}x{dims[1]}")
    out["checks"].append(("dimensions", dims == expected_size, f"{dims[0]}x{dims[1]}"))
    if mode not in ("RGB", "CMYK", "L"):
        out["defects"].append(f"unexpected_mode_{mode}")
    out["checks"].append(("rgb_compatible", mode not in ("RGBA", "P", "LA"), mode))
    L = _luminance(rgb)
    std255 = float(L.std()) * 255.0
    out["stats"]["luminance_std"] = round(std255, 2)
    out["stats"]["michelson"] = round(_michelson(L), 3)
    out["stats"]["edge_density"] = round(_edge_density(rgb), 4)
    if std255 < 6:
        out["defects"].append("blank_image")
    out["checks"].append(("not_blank", std255 >= 6, f"luminance std {std255:.1f}"))
    # Black-bar police: fully-dark edge rows/cols spanning the whole edge.
    H, W = L.shape
    bar_rows_top = float((L[:3, :] < 8 / 255).mean())
    bar_rows_bottom = float((L[-3:, :] < 8 / 255).mean())
    bar_cols_left = float((L[:, :3] < 8 / 255).mean())
    bar_cols_right = float((L[:, -3:] < 8 / 255).mean())
    if min(bar_rows_top, bar_rows_bottom) >= 0.98 and min(bar_cols_left, bar_cols_right) >= 0.98:
        out["defects"].append("black_bars")
    out["checks"].append(("no_black_bars",
                          not (min(bar_rows_top, bar_rows_bottom) >= 0.98
                               and min(bar_cols_left, bar_cols_right) >= 0.98),
                          f"t{bar_rows_top:.2f} b{bar_rows_bottom:.2f} "
                          f"l{bar_cols_left:.2f} r{bar_cols_right:.2f}"))
    out["ok"] = not out["defects"]
    return out


# ---------------------------------------------------------------------------
# Data model + persistence
# ---------------------------------------------------------------------------

@dataclass
class ThumbnailOption:
    index: int
    file: str
    concept: str
    concept_label: str
    design_description: str
    method: str = "procedural_pillow_composition"
    text: str = ""
    text_status: str = "none"     # rendered | text-free | empty
    scores: dict = field(default_factory=dict)
    total: int = 0
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    defects: list = field(default_factory=list)
    passed_quality: bool = False
    check_details: list = field(default_factory=list)
    dimensions: list = field(default_factory=lambda: [THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = f"{CONCEPT_LABELS[self.concept]}"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ThumbnailOption":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


CONCEPT_BUILDERS = {
    "high_appeal": _thumb1_high_appeal,
    "cinematic": _thumb2_cinematic,
    "creative": _thumb3_creative,
}
CONCEPT_LABELS = {
    "high_appeal": "High Click Appeal",
    "cinematic": "Cinematic & Professional",
    "creative": "Creative Alternative",
}
CONCEPT_BLURBS = {
    "high_appeal": "Bold hero scene from the video, dark gradient scrim, large high-contrast title text, content-kind tag and accent edge bar.",
    "cinematic": "Blurred darkened scene backdrop with a sharp framed inset from the closing scene, thin light border and understated caption.",
    "creative": "Diagonal split-panel collage of two distinct scene frames, geometric accents and corner composition, opposite mood from the hero concept.",
}


def _strengths_weaknesses(scores: dict) -> tuple[list, list]:
    strengths, weaknesses = [], []
    phrases = {
        "topic_relevance": "clear topic relevance",
        "visual_clarity": "clear subject focus and structure",
        "text_readability": "readable thumbnail text",
        "contrast": "strong contrast",
        "composition": "balanced composition",
        "audience_relevance": "style suits the content kind",
        "distinctiveness": "visually distinct from the other options",
        "honesty_and_accuracy": "honest, accurate wording",
    }
    for crit, mx in SCORING_CRITERIA.items():
        got = scores.get(crit, 0)
        if got >= 0.8 * mx and phrases[crit] not in strengths:
            strengths.append(phrases[crit])
        elif got <= 0.5 * mx and phrases[crit] not in weaknesses:
            weaknesses.append("limited " + phrases[crit])
    return strengths, weaknesses


# ---------------------------------------------------------------------------
# Orchestrator — generate, score, validate, recommend, persist
# ---------------------------------------------------------------------------

def _pick_text_for_concept(analysis, concept, text_max_words: int) -> str:
    return _short_text(analysis, max_words=text_max_words,
                       short=(concept != "high_appeal"))


def _load_analysis(project, languages=None) -> dict:
    from .languages import LanguageSettings
    from .script_engine import Script
    if languages is None:
        languages = LanguageSettings.from_dict(project.manifest.get("languages", {}))
    script_data = project.read_json("script", "script.json")
    script = Script.from_dict(script_data) if script_data else None
    analysis = analyze_project(project, languages, script)
    scene_data = project.read_json("scenes", "scenes.json")
    if scene_data:
        analysis["scenes"] = [
            {**s, "image_path": (s.get("assets") or {}).get("image", "")}
            for s in scene_data.get("scenes", [])
            if (s.get("assets") or {}).get("image")
        ]
    if not _scene_images(analysis):
        raise RuntimeError("no rendered scene asset images available for thumbnails")
    return analysis


def generate_thumbnails(project, *, languages=None, cfg: dict | None = None,
                        out_dir=None) -> dict:
    """Generate 3 thumbnails for a finished project; returns the report dict.

    A failed concept is recorded and reported honestly; only a total lack of
    scene assets raises.
    """
    cfg = cfg or {}
    width = int(cfg.get("width", THUMBNAIL_WIDTH))
    height = int(cfg.get("height", THUMBNAIL_HEIGHT))
    quality = int(cfg.get("quality", THUMBNAIL_JPEG_QUALITY))
    text_max_words = int(cfg.get("text_max_words", SUBTITLE_TEXT_MAX_WORDS))
    text_override = cfg.get("text_override")

    analysis = _load_analysis(project, languages)
    out_dir = out_dir or project.stage_dir("thumbnails")
    todo = cfg.get("regenerate") or [1, 2, 3]
    warnings = []
    options = []
    built = {}

    for index in todo:
        concept = list(CONCEPT_BUILDERS)[index - 1]
        builder = CONCEPT_BUILDERS[concept]
        fname = f"thumbnail_{index}.jpg"
        text = text_override or _pick_text_for_concept(analysis, concept, text_max_words)
        try:
            img, text = _build_and_shape(builder, analysis, text,
                                         (width, height), out_dir, warnings)
            img.save(out_dir / fname, "JPEG", quality=quality)
            built[index] = img
            options.append(ThumbnailOption(
                index=index, file=fname, concept=concept,
                concept_label=CONCEPT_LABELS[concept],
                design_description=CONCEPT_BLURBS[concept],
                text=text, text_status=("rendered" if text else "text-free")))
        except Exception as exc:  # noqa: BLE001
            opt = ThumbnailOption(
                index=index, file=fname, concept=concept,
                concept_label=CONCEPT_LABELS[concept],
                design_description=CONCEPT_BLURBS[concept],
                method="failed")
            opt.defects.append(f"generation_failed: {exc}")
            options.append(opt)
            warnings.append(f"thumbnail {index} generation failed: {exc}")

    # Merge prior options for indexes not (re)built in this run, so
    # regenerating a single thumbnail never erases the other two (12.8).
    prior = read_recommendation_json(project)
    prior_selected = None
    if prior:
        prior_selected = prior.get("selected")
        fresh_idx = {o.index for o in options}
        for od in prior.get("options", []):
            i = od.get("index")
            if i and i not in fresh_idx:
                opt = ThumbnailOption.from_dict(od)
                options.append(opt)
                fp = out_dir / opt.file
                if fp.exists() and i not in built and opt.method != "failed":
                    try:
                        from PIL import Image as _Img
                        built[i] = _Img.open(fp).convert("RGB")
                    except Exception:  # noqa: BLE001
                        pass
        options.sort(key=lambda o: o.index)
        # Regeneration never touches youtube_thumbnail.jpg (12.8), so a prior
        # selection record stays valid; warn when its source option changed.
        if prior_selected and prior_selected.get("index") in set(
                o.index for o in options if o.index in built):
            warnings.append(
                f"thumbnail {prior_selected['index']} was regenerated after "
                f"selection; youtube_thumbnail.jpg still holds the previously "
                f"selected copy. Re-select to adopt the new version.")

    return _score_and_report(project, analysis, options, built, out_dir,
                             (width, height), warnings,
                             selected=prior_selected)


def _build_and_shape(builder, analysis, text, size, out_dir, warnings):
    """Build one concept; route text through libass for shaping scripts.

    Returns (image, final_text).  When correct shaped rendering cannot be
    verified the thumbnail is rebuilt text-free (b.txt 12.5).
    """
    import tempfile
    width, height = size
    lang = analysis["thumbnail_language"]
    text_status = "rendered" if text else "empty"
    with tempfile.TemporaryDirectory(prefix="fathumb_") as td:
        img, text = builder(analysis, td, text=text, size=size, quality=90)
    if text and analysis.get("font_needs_shaping") and _rtl(lang):
        layer, ok = _render_shaped_text_layer(
            text, lang_code=lang, size=int(height * 0.115),
            color=(255, 255, 255), work_dir=out_dir, max_width=int(width * 0.86))
        with tempfile.TemporaryDirectory(prefix="fathumb2_") as td2:
            img_plain, _ = builder(analysis, td2, text="", size=size, quality=90)
        if ok and layer is not None:
            from PIL import ImageDraw
            lw, lh = layer.size
            img_plain.paste(layer, (int(width * 0.05),
                                    height - lh - int(height * 0.06)), layer)
            d = ImageDraw.Draw(img_plain)
            d.rectangle([0, height - max(6, int(height * 0.014)),
                         width, height], fill=(255, 170, 0))
            img, text_status = img_plain, "rendered"
        else:
            warnings.append(
                "Correct RTL/complex text rendering could not be verified, so "
                "the thumbnail is text-free (b.txt 12.5).")
            img, text, text_status = img_plain, "", "text-free"
    return img, (text or "")


def _text_bbox_for(opt: ThumbnailOption, width: int, height: int):
    """Approximate text bbox matching where concepts paste text layers."""
    if not opt.text:
        return None
    concept = opt.concept
    tsize = int(height * (0.128 if concept == "high_appeal" else
                          0.105 if concept == "creative" else 0.075))
    stroke = max(3, tsize // 14)
    layer, bbox = render_text_image(
        opt.text, lang_code="en", size=tsize, color=(0, 0, 0), stroke=stroke,
        max_width=int(width * 0.86))
    if layer is None:
        return None
    lw, lh = layer.size
    if concept == "cinematic":
        x0, y0 = (width - lw) // 2, height - lh - int(height * 0.07)
    elif concept == "creative":
        x0, y0 = int(width * 0.035), height - lh - int(height * 0.07)
    else:
        x0, y0 = int(width * 0.05), height - lh - int(height * 0.06)
    return (x0, y0, x0 + lw, y0 + lh)


def _score_and_report(project, analysis, options, built, out_dir, size, warnings,
                      *, selected=None):
    """Score all options, run quality checks, pick recommendation, persist."""
    from PIL import Image
    width, height = size
    for opt in options:
        if opt.method == "failed" or opt.index not in built:
            opt.scores = {c: 0 for c in SCORING_CRITERIA}
            opt.total = 0
            continue
        img = built[opt.index]
        crit = {}
        crit["topic_relevance"], _ = score_topic_relevance(img, analysis, opt.text)
        crit["visual_clarity"], _ = score_visual_clarity(img, None)
        tmask = {"bbox": _text_bbox_for(opt, width, height)} if opt.text else None
        crit["text_readability"], _ = score_text_readability(img, opt.text, tmask)
        crit["contrast"], _ = score_contrast(img)
        crit["composition"], _ = score_composition(img, tmask and tmask["bbox"])
        crit["audience_relevance"], _ = score_audience_relevance(analysis, opt.concept)
        others = [built[j] for j in built if j != opt.index] or list(built.values())
        crit["distinctiveness"], _ = score_distinctiveness(img, others)
        crit["honesty_and_accuracy"], _ = score_honesty(analysis, opt.text)
        opt.scores = crit
        opt.total = sum(crit.values())
        opt.strengths, opt.weaknesses = _strengths_weaknesses(crit)
        vd = validate_thumbnail(out_dir / opt.file, expected_size=size)
        # Keep only the fresh defect list for this final validated file.
        opt.defects = vd["defects"]
        opt.check_details = [f"[{'x' if c[1] else ' '}] {c[0]}: {c[2]}"
                             for c in vd["checks"]]
        opt.dimensions = [width, height]
        opt.passed_quality = vd["ok"]
        if not vd["ok"]:
            warnings.append(f"thumbnail {opt.index} has critical defects: "
                            f"{vd['defects']}")

    passing = [o for o in options if o.passed_quality and o.method != "failed"]
    recommended, reasoning = None, ""
    if passing:
        best = max(passing, key=lambda o: o.total)
        recommended = best.index
        strong = ", ".join(best.strengths[:3]) or "the overall strongest design"
        reasoning = (f"Thumbnail {best.index} scored the highest ({best.total}/100) among "
                     f"the options that passed quality validation. Main strengths: "
                     f"{strong}. " + SCORING_DISCLAIMER)
    else:
        reasoning = ("No thumbnail passed quality validation, so no recommendation can "
                     "be made. See warnings; regenerate with: freeanimate thumbnails "
                     "regenerate <project>")
        warnings.append("no recommendation: every option failed quality validation")

    report = {
        "video_title": analysis["title"],
        "project_id": project.manifest.get("slug"),
        "generated_at": time.time(),
        "thumbnail_language": analysis["thumbnail_language"],
        "generation_method": "procedural_composition_from_scene_assets",
        "options": [o.to_dict() for o in options],
        "scoring_criteria": dict(SCORING_CRITERIA),
        "scoring_disclaimer": SCORING_DISCLAIMER,
        "recommendation": {"index": recommended, "reasoning": reasoning},
        "warnings": warnings,
    }
    if selected:
        report["selected"] = selected
    write_recommendation_json(project, report)
    return report


# ---------------------------------------------------------------------------
# Persistence, selection, regeneration (12.8 / 12.9 / 15)
# ---------------------------------------------------------------------------

RECOMMENDATION_NAME = "thumbnail_recommendations.json"
SELECTED_NAME = "youtube_thumbnail.jpg"


def write_recommendation_json(project, report: dict) -> Path:
    return project.write_json("thumbnails", RECOMMENDATION_NAME, report)


def read_recommendation_json(project) -> dict | None:
    return project.read_json("thumbnails", RECOMMENDATION_NAME)


def select_thumbnail(project, choice, *, overwrite_confirmed: bool = False) -> dict:
    """Copy the chosen thumbnail to youtube_thumbnail.jpg (12.8).

    choice: 1 | 2 | 3 | "accept" (use recommendation).  Never overwrites a
    previously selected thumbnail without confirmation.
    """
    report = read_recommendation_json(project)
    if not report:
        raise RuntimeError("no thumbnail report found; run 'freeanimate thumbnails "
                           "generate <project>' first")
    if choice == "accept":
        idx = report.get("recommendation", {}).get("index")
        if not idx:
            raise RuntimeError("the report contains no recommendation "
                               "(no option passed quality checks); pick 1, 2 or 3 explicitly")
    else:
        idx = int(choice)
    if not (1 <= idx <= THUMBNAIL_COUNT):
        raise ValueError(f"thumbnail index must be 1..{THUMBNAIL_COUNT}, got {choice}")
    src = project.stage_dir("thumbnails") / f"thumbnail_{idx}.jpg"
    if not src.exists():
        raise FileNotFoundError(f"thumbnail file missing: {src}")
    dst = project.stage_dir("thumbnails") / SELECTED_NAME
    if dst.exists() and not overwrite_confirmed:
        raise FileExistsError(
            f"{dst.name} already exists (previously selected thumbnail). "
            "Pass overwrite_confirmed=True / answer yes at the prompt to replace it.")
    if choice != "accept" and report.get("recommendation", {}).get("index") == idx:
        pass
    shutil.copyfile(src, dst)
    report = read_recommendation_json(project)
    report["selected"] = {"index": idx, "file": dst.name,
                          "timestamp": time.time()}
    write_recommendation_json(project, report)
    return {"index": idx, "path": dst}


def regenerate_thumbnail(project, index=None, *, languages=None,
                         text: str | None = None, lang_code: str | None = None) -> dict:
    """Regenerate one thumbnail (index 1..3) or all three (index=None).

    Never touches the rendered video; only thumbnail assets and the report
    are rewritten.
    """
    from .languages import LanguageSettings
    regen = [int(index)] if index is not None else None
    cfg = {"regenerate": regen} if regen else {}
    if text is not None:
        cfg["text_override"] = text
    langs = languages or LanguageSettings.from_dict(project.manifest.get("languages", {}))
    if lang_code:
        from .languages import LanguageRegistry
        resolved = LanguageRegistry.builtin().resolve(lang_code).code
        langs = LanguageSettings.from_dict(
            {**project.manifest.get("languages", {}), "thumbnail": resolved})
    return generate_thumbnails(project, languages=langs, cfg=cfg)


def open_thumbnails_folder(project) -> None:
    """Open the thumbnails folder in Explorer (12.8, Windows-first)."""
    folder = project.stage_dir("thumbnails")
    if sys.platform.startswith("win"):
        os.startfile(folder)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder)], check=False)
    else:
        subprocess.run(["xdg-open", str(folder)], check=False)
