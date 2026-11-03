"""Load a photo, frame it to the print, and turn it into a map of wanted light.

The photo is in sRGB (display gamma). A lithophane shows linear light, so the tone work
(levels, local contrast) runs on the perceptual lightness L* and the result goes back to
linear luminance at the end.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 400_000_000  # big camera files are fine


def load(path: Path) -> np.ndarray:
    """RGB floats in [0, 1] (sRGB), turned upright from the EXIF orientation, alpha on white."""
    try:
        return _load(path)
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"cannot read {path} as a photo: {e}") from None


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if is_deep(im):
            return np.repeat(deep_grey(im)[..., None], 3, axis=2)
        rgb = np.asarray(flatten(im, (255, 255, 255)).convert("RGB"), dtype=np.float64) / 255
    return rgb


def is_deep(im: Image.Image) -> bool:
    """16-bit grey (I;16, or I from older Pillow) or float: see `deep_grey`."""
    return im.mode.startswith("I") or im.mode == "F"


def deep_grey(im: Image.Image, dtype=np.float64) -> np.ndarray:
    """A 16-bit grey or float image as values from 0 to 1, scaled by its mode (I;16 always
    by 65535); convert("RGB") or convert("L") would clip it at 255 instead of scaling it.
    Photos and focus masks use the same rule."""
    grey = np.asarray(im, dtype=dtype)
    top = grey.max(initial=0)
    if im.mode.startswith("I;16") or top > 255:
        grey /= 65535
    elif im.mode != "F" or top > 1:
        grey /= 255
    return np.clip(grey, 0, 1)


def flatten(im: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """An image with transparency painted over a plain background; others as they are."""
    if im.mode in ("RGBA", "LA", "P", "PA") or "transparency" in im.info:
        im = im.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", im.size, (*background, 255)), im)
    return im


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Linear luminance Y (Rec. 709 / sRGB primaries)."""
    lin = srgb_to_linear(rgb)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def y_to_lstar(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0, 1)
    return np.where(y > (6 / 29) ** 3, 116 * np.cbrt(y) - 16, y * (29 / 3) ** 3)


def lstar_to_y(lstar: np.ndarray) -> np.ndarray:
    f = (np.asarray(lstar) + 16) / 116
    return np.where(f > 6 / 29, f**3, (np.asarray(lstar)) * (3 / 29) ** 3)


def parse_crop(text: str) -> tuple[int, int, int, int]:
    """'X,Y,W,H' in photo pixels."""
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError(f"crop must be X,Y,W,H with W, H > 0 (got {text!r})")
    return parts[0], parts[1], parts[2], parts[3]


def _crop_box(
    size: tuple[int, int], box: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int]:
    """The user's crop (X, Y, W, H) inside a photo of `size` (width, height), as x0, y0, x1,
    y1 in photo pixels."""
    w, h = size
    if not box:
        return 0, 0, w, h
    x, y, bw, bh = box
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("the crop lies outside the photo")
    return x0, y0, x1, y1


def crop(rgb: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    """The user's crop (X, Y, W, H in photo pixels), or the whole photo."""
    if not box:
        return rgb
    x0, y0, x1, y1 = _crop_box((rgb.shape[1], rgb.shape[0]), box)
    return rgb[y0:y1, x0:x1]


def window(
    size: tuple[int, int], aspect: float, box: tuple[int, int, int, int] | None = None
) -> tuple[int, int, int, int]:
    """The part of a photo of `size` (width, height) that `frame` keeps, as x0, y0, x1, y1
    in photo pixels: first the user's crop, then the largest centred window with `aspect`
    (width / height). A focus map in photo space follows the photo with it."""
    x0, y0, x1, y1 = _crop_box(size, box)
    w, h = x1 - x0, y1 - y0
    if w / h > aspect:
        nw = max(1, round(h * aspect))
        x0 += (w - nw) // 2
        return x0, y0, x0 + nw, y1
    nh = max(1, round(w / aspect))
    y0 += (h - nh) // 2
    return x0, y0, x1, y0 + nh


def frame(
    rgb: np.ndarray, aspect: float, box: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    """Crop the photo to `aspect` (width / height): first the user's crop, then the
    largest centred window with that aspect."""
    x0, y0, x1, y1 = window((rgb.shape[1], rgb.shape[0]), aspect, box)
    return rgb[y0:y1, x0:x1]


def to_pil(values: np.ndarray) -> Image.Image:
    """Floats from 0 to 1 (grey or RGB) as an 8-bit image, rounded to the nearest level."""
    return Image.fromarray((np.clip(values, 0, 1) * 255 + 0.5).astype(np.uint8))


def resize(rgb: np.ndarray, rows: int, cols: int) -> np.ndarray:
    im = to_pil(rgb)
    im = im.resize((cols, rows), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255


@dataclass(frozen=True)
class Tone:
    """Tone settings: what we do to the lightness before we print it."""

    levels: tuple[float, float] = (0.5, 99.5)  # percentiles that become black and white
    gamma: float = 1.0  # > 1 darkens the mid-tones, < 1 brightens them
    clahe: float = 0.0  # local contrast (CLAHE clip limit); 0 = off, 0.01 = mild
    invert: bool = False  # negative picture (rarely wanted)


def light_map(rgb: np.ndarray, tone: Tone | None = None) -> np.ndarray:
    """RGB (sRGB) -> wanted linear light in [0, 1], after levels, local contrast and gamma."""
    tone = tone or Tone()
    lstar = y_to_lstar(luminance(rgb)) / 100
    lo, hi = np.percentile(lstar, tone.levels)
    if hi - lo > 1e-6:
        lstar = np.clip((lstar - lo) / (hi - lo), 0, 1)
    if tone.clahe > 0:
        from skimage import exposure

        lstar = exposure.equalize_adapthist(lstar, clip_limit=tone.clahe)
    if tone.invert:
        lstar = 1 - lstar  # a negative in lightness: mid-grey stays mid-grey
    light = lstar_to_y(lstar * 100)
    if tone.gamma != 1.0:
        light = light**tone.gamma
    return np.clip(light, 0, 1)


@dataclass
class Analysis:
    """What we can tell about a photo before printing it."""

    size: tuple[int, int]  # width, height in pixels
    mean_lstar: float
    p1_lstar: float
    p99_lstar: float
    shadows: float  # share of pixels with L* < 5
    highlights: float  # share of pixels with L* > 95
    colourfulness: float  # Hasler & Suesstrunk; above ~40 the photo is very colourful
    notes: list[str] = field(default_factory=list)

    @property
    def dynamic_range(self) -> float:
        return self.p99_lstar - self.p1_lstar


def analyse(rgb: np.ndarray, grid: tuple[int, int] | None = None, colour=False) -> Analysis:
    """Statistics and advice. `grid` (rows, cols) is the print grid, to check resolution.

    The advice is for a white lithophane (--mode mono), or with `colour` for a colour
    print: there the height of a black or white part depends on the colour plan, colours
    count, and --clahe does not exist."""
    lstar = y_to_lstar(luminance(rgb))
    p1, p99 = np.percentile(lstar, [1, 99])
    r, g, b = (rgb[..., k] * 255 for k in range(3))
    rg, yb = r - g, (r + g) / 2 - b
    colourful = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    a = Analysis(
        (rgb.shape[1], rgb.shape[0]),
        float(lstar.mean()),
        float(p1),
        float(p99),
        float((lstar < 5).mean()),
        float((lstar > 95).mean()),
        colourful,
    )
    if a.dynamic_range < 50:
        a.notes.append(
            f"low contrast (L* {p1:.0f}..{p99:.0f}): levels will stretch it"
            + ("" if colour else "; try --clahe 0.01 for more local contrast")
        )
    if a.shadows > 0.25:
        a.notes.append(
            f"{a.shadows:.0%} of the photo is near black: those parts show no detail"
            + ("" if colour else " and print at full thickness (more time and plastic)")
        )
    if a.highlights > 0.25:
        a.notes.append(
            f"{a.highlights:.0%} of the photo is near white: those parts show no detail"
            + ("" if colour else " and print at minimum thickness")
        )
    if a.mean_lstar < 35:
        a.notes.append("dark photo: try --gamma 0.8 to open the shadows")
    elif a.mean_lstar > 75:
        a.notes.append("bright photo: try --gamma 1.3 to bring back the light tones")
    if a.colourfulness > 40 and not colour:
        a.notes.append(
            f"very colourful photo ({a.colourfulness:.0f}): a lithophane shows only "
            "lightness, so colours of the same lightness will merge"
        )
    if grid and (rgb.shape[0] < grid[0] or rgb.shape[1] < grid[1]):
        a.notes.append(
            f"the photo ({rgb.shape[1]} x {rgb.shape[0]} px after cropping) has "
            f"fewer pixels than the print grid ({grid[1]} x {grid[0]}): it will "
            "look soft; print it smaller"
        )
    return a
