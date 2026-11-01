"""Pictures of the result, without OpenGL: what the lit lithophane will look like, and its
relief seen in raking light."""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .image import linear_to_srgb
from .optics import Optics

LED = np.array([1.0, 0.93, 0.82])  # a warm white LED behind the panel
BACKGROUND = (236, 234, 240)
INK = (40, 40, 48)


def _u8(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def backlit(thickness: np.ndarray, optics: Optics, light=LED) -> Image.Image:
    """The panel lit from behind, as the viewer sees it (thickness in viewer orientation)."""
    lum = optics.light(thickness)  # 1 at t_min
    return _u8(linear_to_srgb(lum[..., None] * np.asarray(light)[None, None, :]))


def relief(thickness: np.ndarray, pitch: float) -> Image.Image:
    """The relief side in raking light from the upper left (a hillshade)."""
    gy, gx = np.gradient(thickness, pitch)
    normal = np.dstack([-gx, gy, np.ones_like(thickness)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    sun = np.array([-0.55, 0.55, 0.63])
    sun /= np.linalg.norm(sun)
    shade = np.clip(normal @ sun, 0, 1)
    base = 0.25 + 0.75 * shade
    return _u8(np.dstack([base * 0.95, base * 0.95, base]))


def _fit(im: Image.Image, box: int) -> Image.Image:
    scale = box / max(im.size)
    return im.resize(
        (max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS
    )


def sheet(panels: list[tuple[str, Image.Image]], path: Path, box: int = 420) -> Path:
    """Pictures side by side with a caption under each."""
    font = ImageFont.load_default(size=16)
    pics = [(title, _fit(im, box)) for title, im in panels]
    pad, caption = 16, 28
    width = pad + sum(p.width + pad for _, p in pics)
    height = pad + max(p.height for _, p in pics) + caption + pad
    out = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(out)
    x = pad
    for title, pic in pics:
        out.paste(pic, (x, pad))
        draw.text((x, pad + pic.height + 6), title, fill=INK, font=font)
        x += pic.width + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path


def thumbnail(im: Image.Image, size: int = 256) -> bytes:
    """A square PNG for the 3MF (PrusaSlicer shows it in its file dialogs)."""
    pic = _fit(im, size)
    square = Image.new("RGB", (size, size), BACKGROUND)
    square.paste(pic, ((size - pic.width) // 2, (size - pic.height) // 2))
    buf = io.BytesIO()
    square.save(buf, format="PNG")
    return buf.getvalue()
