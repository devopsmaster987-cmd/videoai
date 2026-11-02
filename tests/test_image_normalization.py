"""Regression tests for the "images do not match" failure class.

The original defect: ``freeanimate.backgrounds._render_space`` passed an RGBA
canvas and an RGB nebula to ``PIL.Image.blend``.  Pillow raises
``ValueError: images do not match`` whenever modes or sizes differ.  These tests
exercise every realistic input type (RGB, RGBA, grayscale, palette, transparent
PNG, JPEG) and the exact space-scene reproducer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from freeanimate.animation2d import prepare_background
from freeanimate.backgrounds import EnvironmentSpec, render_environment
from freeanimate.image_utils import (blend_images, composite_images, normalize_image,
                                     paste_alpha)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA"])
def test_normalize_modes_to_rgb_and_rgba(tmp_path: Path, mode: str):
    img = Image.new(mode, (37, 23), 128 if mode in ("L", "LA") else (30, 80, 140, 160))
    rgb = normalize_image(img, (64, 48), mode="RGB")
    rgba = normalize_image(img, (64, 48), mode="RGBA")
    assert rgb.mode == "RGB" and rgb.size == (64, 48)
    assert rgba.mode == "RGBA" and rgba.size == (64, 48)


def test_palette_png_with_transparency(tmp_path: Path):
    src = Image.new("RGBA", (20, 20), (255, 0, 0, 0))
    for x in range(10):
        src.putpixel((x, x), (0, 128, 255, 255))
    palette = src.convert("P")
    path = tmp_path / "palette.png"
    palette.save(path, transparency=0)
    out = normalize_image(path, (32, 32), mode="RGB")
    assert out.mode == "RGB" and out.size == (32, 32)


def test_jpeg_and_transparent_png_are_handled(tmp_path: Path):
    jpg = tmp_path / "photo.jpg"
    Image.new("RGB", (100, 60), (200, 120, 40)).save(jpg)
    png = tmp_path / "alpha.png"
    Image.new("RGBA", (100, 60), (200, 120, 40, 0)).save(png)
    for path in (jpg, png):
        assert prepare_background(path, 320, 180).size == (320, 180)


def test_blend_and_composite_never_match_error():
    rgba = Image.new("RGBA", (50, 30), (10, 20, 30, 120))
    rgb = Image.new("RGB", (50, 30), (200, 150, 100))
    small = Image.new("RGB", (10, 10), (0, 0, 0))
    # Different modes and different sizes are normalized instead of raising.
    assert blend_images(rgba, rgb, 0.5).size == (50, 30)
    assert blend_images(rgb, small, 0.5, size=(50, 30)).size == (50, 30)
    mask = Image.new("L", (50, 30), 90)
    assert composite_images(rgba, rgb, mask=mask).size == (50, 30)
    base = Image.new("RGB", (50, 30), (1, 2, 3))
    assert paste_alpha(base, small, (35, 20)).size == (50, 30)


@pytest.mark.parametrize("env", ["space", "horror_mansion", "forest", "garden", "cafe"])
def test_environment_renderers_do_not_raise_mismatch(tmp_path: Path, env: str):
    """The space scene is the exact original failure (RGBA + RGB blend)."""
    out = render_environment(EnvironmentSpec(env_type=env), tmp_path / f"{env}.png",
                             width=320, height=180)
    with Image.open(out) as im:
        assert im.mode == "RGB" and im.size == (320, 180)


def test_original_space_reproducer(tmp_path: Path, monkeypatch):
    """Directly reproduce the old call shape and prove it is fixed."""
    import freeanimate.backgrounds as backgrounds

    original = backgrounds.blend_images

    def checked(base, overlay, alpha, size=None, **kwargs):
        # The helper itself normalizes; this asserts both inputs reach it.
        assert base.mode in ("RGB", "RGBA")
        return original(base, overlay, alpha, size, **kwargs)

    monkeypatch.setattr(backgrounds, "blend_images", checked)
    render_environment(EnvironmentSpec(env_type="space", time_of_day="night"),
                       tmp_path / "space.png", width=320, height=180)
