"""Safe image normalization and compositing helpers.

Pillow's compositing functions (``Image.blend``, ``Image.composite``,
``Image.alpha_composite`` and three-argument ``paste``) raise
``ValueError: images do not match`` whenever the participating images differ in
size or color mode.  Real projects mix RGB, RGBA, grayscale, palette PNGs and
JPEGs, so every compositing site must normalize first.

This module is the single place that does that normalization.  It deliberately
preserves alpha where asked, flattens it onto a declared background color when
converting to RGB, and supports explicit ``stretch``, ``cover`` and ``contain``
fit modes so aspect ratio is never changed accidentally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence, Union

from PIL import Image, ImageOps

ImageLike = Union[str, Path, Image.Image]
Mode = Literal["RGB", "RGBA", "L", "LA"]
Fit = Literal["stretch", "cover", "contain"]

DEFAULT_BACKGROUND = (0, 0, 0)


def _open(source: ImageLike) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.copy()
    with Image.open(source) as im:
        # Honour EXIF orientation for phone/camera JPEGs.
        try:
            im = ImageOps.exif_transpose(im)
        except (AttributeError, OSError, ValueError):
            pass
        return im.copy()


def normalize_image(
    source: ImageLike,
    size: tuple[int, int] | None = None,
    *,
    mode: Mode = "RGB",
    background: Sequence[int] = DEFAULT_BACKGROUND,
    fit: Fit = "stretch",
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """Return a new image with a guaranteed mode and (optionally) size.

    ``stretch`` resizes to exactly ``size`` (intentional distortion only).
    ``cover`` fills the box preserving aspect ratio and centre-crops.
    ``contain`` letterboxes the image inside the box preserving aspect ratio.
    """
    img = _open(source)

    # Palette images: keep transparency when present, otherwise plain RGB.
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    elif img.mode == "PA":
        img = img.convert("RGBA")
    elif img.mode in ("1", "CMYK", "YCbCr", "LAB", "HSV"):
        img = img.convert("RGB")
    elif img.mode not in ("RGB", "RGBA", "L", "LA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")

    if mode == "RGB":
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, tuple(int(c) for c in background[:3]))
            bg.paste(img, (0, 0), img)
            img = bg
        elif img.mode == "LA":
            bg = Image.new("L", img.size, int(background[0]) if background else 0)
            bg.paste(img.convert("L"), (0, 0), img.getchannel("A"))
            img = bg.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
    elif mode == "RGBA":
        img = img.convert("RGBA")
    elif mode == "L":
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, tuple(int(c) for c in background[:3]) + (255,))
            img = Image.alpha_composite(bg, img).convert("L")
        else:
            img = img.convert("L")
    elif mode == "LA":
        img = img.convert("RGBA").getchannel("A") if img.mode == "RGBA" else img.convert("LA")

    if size is not None:
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"invalid image size: {size!r}")
        img = resize_fit(img, size, fit=fit, resample=resample)
    return img


def resize_fit(img: Image.Image, size: tuple[int, int], *, fit: Fit = "stretch",
               resample: int = Image.LANCZOS,
               background: Sequence[int] = DEFAULT_BACKGROUND) -> Image.Image:
    width, height = int(size[0]), int(size[1])
    if img.size == (width, height):
        return img
    if fit == "stretch":
        return img.resize((width, height), resample)
    scale = max(width / img.width, height / img.height) if fit == "cover" else \
        min(width / img.width, height / img.height)
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    resized = img.resize(new_size, resample)
    if fit == "cover":
        left = max(0, (new_size[0] - width) // 2)
        top = max(0, (new_size[1] - height) // 2)
        return resized.crop((left, top, left + width, top + height))
    # contain
    canvas_mode = "RGBA" if "A" in resized.getbands() else "RGB"
    color = (*(int(c) for c in background[:3]), 255) if canvas_mode == "RGBA" else \
        tuple(int(c) for c in background[:3])
    canvas = Image.new(canvas_mode, (width, height), color)
    canvas.paste(resized, ((width - new_size[0]) // 2, (height - new_size[1]) // 2),
                 resized if canvas_mode == "RGBA" else None)
    return canvas


def ensure_rgba(source: ImageLike, size: tuple[int, int] | None = None) -> Image.Image:
    return normalize_image(source, size, mode="RGBA")


def ensure_rgb(source: ImageLike, size: tuple[int, int] | None = None, *,
               background: Sequence[int] = DEFAULT_BACKGROUND) -> Image.Image:
    return normalize_image(source, size, mode="RGB", background=background)


def blend_images(base: ImageLike, overlay: ImageLike, alpha: float,
                 size: tuple[int, int] | None = None, *,
                 mode: Mode = "RGB") -> Image.Image:
    """``Image.blend`` after normalizing both participants to one mode/size."""
    a = normalize_image(base, size, mode=mode)
    b = normalize_image(overlay, a.size, mode=mode)
    return Image.blend(a, b, max(0.0, min(1.0, float(alpha))))


def composite_images(front: ImageLike, back: ImageLike, mask: ImageLike | None = None,
                     size: tuple[int, int] | None = None) -> Image.Image:
    """``Image.composite`` (mask) or alpha compositing after normalization."""
    b = normalize_image(back, size, mode="RGBA")
    f = normalize_image(front, b.size, mode="RGBA")
    if mask is not None:
        m = normalize_image(mask, b.size, mode="L")
        return Image.composite(f, b, m)
    return Image.alpha_composite(b, f)


def alpha_composite(base: ImageLike, overlay: ImageLike,
                    size: tuple[int, int] | None = None) -> Image.Image:
    return composite_images(overlay, base, size=size)


def paste_alpha(base: Image.Image, overlay: ImageLike, xy: tuple[int, int] = (0, 0),
                size: tuple[int, int] | None = None) -> Image.Image:
    """Paste an RGBA overlay with alpha onto a possibly-RGB base safely."""
    result = normalize_image(base, mode="RGBA" if "A" in base.getbands() else "RGB")
    if result.mode == "RGB":
        layer = normalize_image(overlay, size, mode="RGBA")
        result.paste(layer, xy, layer)
    else:
        layer = normalize_image(overlay, size, mode="RGBA")
        result.alpha_composite(layer, dest=(max(0, xy[0]), max(0, xy[1])))
    return result
