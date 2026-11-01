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


def _masked(rgb: np.ndarray, mask) -> np.ndarray:
    if mask is not None:
        rgb = np.where(mask[..., None], rgb, np.array(BACKGROUND) / 255)
    return rgb


def backlit(thickness: np.ndarray, optics: Optics, light=LED, mask=None) -> Image.Image:
    """The panel lit from behind, as the viewer sees it (thickness in viewer orientation).
    `mask`: where the panel is (shaped panels); the rest shows the background."""
    lum = optics.light(thickness)  # 1 at t_min
    rgb = linear_to_srgb(lum[..., None] * np.asarray(light)[None, None, :])
    return _u8(_masked(rgb, mask))


def relief(thickness: np.ndarray, pitch: float, mask=None) -> Image.Image:
    """The relief side in raking light from the upper left (a hillshade)."""
    gy, gx = np.gradient(thickness, pitch)
    normal = np.dstack([-gx, gy, np.ones_like(thickness)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    sun = np.array([-0.55, 0.55, 0.63])
    sun /= np.linalg.norm(sun)
    shade = np.clip(normal @ sun, 0, 1)
    base = 0.25 + 0.75 * shade
    return _u8(_masked(np.dstack([base * 0.95, base * 0.95, base]), mask))


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


def colour(lin: np.ndarray, mask=None) -> Image.Image:
    """A simulated colour print (linear RGB per grid point, as the viewer sees it)."""
    return _u8(_masked(linear_to_srgb(np.clip(lin, 0, 1)), mask))


def palette_strip(plan, layer: float, width: int = 260, row: int = 56) -> Image.Image:
    """The filaments of the print from the bed up: swatch, name, where it starts."""
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    fils = plan.real.filaments
    im = Image.new("RGB", (width, row * len(fils) + 8), BACKGROUND)
    draw = ImageDraw.Draw(im)
    for k, (f, z) in enumerate(zip(fils, plan.real.starts)):
        y = 4 + row * (len(fils) - 1 - k)  # the first filament at the bottom
        draw.rectangle([4, y + 4, 48, y + row - 6], fill=f.hex, outline=INK)
        draw.text((58, y + 6), f.name.replace("Prusament ", ""), fill=INK, font=font)
        start = "first filament" if k == 0 else f"M600 at Z {z + layer:.2f} mm"
        draw.text((58, y + 28), f"{start}, TD {f.td:g}", fill=INK, font=small)
    return im
