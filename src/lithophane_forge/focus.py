"""Colour mode: the focus map, which says where the subject of the photo is.

The map lives in photo space: it has the photo's proportions and a value per pixel from 0
(the rest of the picture) to 1 (the subject). It follows the photo through the user's crop
and the automatic crop to the panel, so it fits the photo whatever the panel's shape. It
comes from a mask image (white = the subject, black or transparent = the rest, stretched to
the photo), from a box in photo pixels (X,Y,W,H, like --crop), from `centre` (a wide soft
spot in the middle of the picture) or from `subject`: segmentation models find it
(vision.py, with the optional vision extra).

The colour plan samples the photo's pixels with the weight floor + (1 - floor) x map, so
the colours and the change heights are chosen for the subject first. The points where the
map is at least SUBJECT are the subject: the report gives their Delta E apart.

A subject that covers a small share of the picture weighs little even at full weight (with
the default floor, a subject on 1 % of the picture gets about 9 % of the samples): mark a
wide region (the whole person, not only the face), or lower --focus-floor.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .image import deep_grey, flatten, is_deep, parse_crop, to_pil

FLOOR = 0.1  # the weight of the rest of the picture, against 1 for the subject
MIN_FLOOR = 0.01  # every pixel keeps some weight: the sampling must not run out of pixels
SUBJECT = 0.5  # a point with at least this map value is part of the subject
MAP_SIDE = 1000  # focus.png: pixels on the long side (fewer for a smaller photo)
CENTRE_SIGMA = 0.25  # centre: the spot's sigma, as a share of the picture's width and height
CENTRE = ("centre", "center")
FIND = "subject"  # found by the models of vision.py
DEVICES = ("auto", "cpu", "gpu")  # subject: where the models run (vision.session)
BOX = re.compile(r"[-+\d\s]+(,[-+\d\s]+)+")  # numbers and commas: a box, not a file name
ASPECT = 0.02  # a mask whose proportions differ more than this from the photo's gets a note
MIN_FOUND = 0.002  # subject: parts smaller than this share of the photo go (vision.drop_small)


@dataclass(frozen=True)
class Spec:
    """A parsed --focus value."""

    kind: str  # "mask", "box", "centre" or "subject"
    path: Path | None = None
    box: tuple[int, int, int, int] | None = None


def parse(text: str) -> Spec:
    """'centre', 'subject', 'X,Y,W,H' (photo pixels), or the name of a mask image (a file
    called subject is ./subject). It does not look at the disk: `Job.check` calls it."""
    text = text.strip()
    if not text:
        raise ValueError("--focus needs a value: a mask image, X,Y,W,H, centre or subject")
    if text.lower() in CENTRE:
        return Spec("centre")
    if text.lower() == FIND:
        return Spec("subject")
    if BOX.fullmatch(text):
        try:
            return Spec("box", box=parse_crop(text))
        except ValueError:
            raise ValueError(
                f"a --focus box must be X,Y,W,H in photo pixels with W, H > 0 (got {text!r})"
            ) from None
    return Spec("mask", path=Path(text))


def photo_map(spec: Spec, size: tuple[int, int], window: tuple[int, int, int, int]) -> np.ndarray:
    """The focus map in photo space for a photo of `size` (width, height): rows x columns
    with the photo's proportions, MAP_SIDE on the long side, values 0 to 1 on 8 bits, so
    that focus.png gives the same map back. `window` (x0, y0, x1, y1, from `image.window`)
    is the part of the photo that the picture shows: `centre` is its middle. `subject`
    needs the photo itself: see `build`."""
    w, h = size
    mw, mh = map_size(size)
    if spec.kind == "subject":
        raise ValueError("--focus subject needs the photo: use focus.build")
    if spec.kind == "mask":
        values = _mask(spec.path, (mw, mh))
    elif spec.kind == "box":
        bx, by, bw, bh = spec.box
        if bx >= w or by >= h or bx + bw <= 0 or by + bh <= 0:
            raise ValueError(f"the --focus box lies outside the photo ({w} x {h} px)")
        # the share of each map pixel that the box covers: a box smaller than a map pixel
        # still marks it in part
        values = _cover(by, by + bh, mh, h)[:, None] * _cover(bx, bx + bw, mw, w)[None, :]
    else:
        x = (np.arange(mw) + 0.5) * w / mw  # pixel centres, in photo pixels
        y = (np.arange(mh) + 0.5) * h / mh
        x0, y0, x1, y1 = window
        gx = ((x - (x0 + x1) / 2) / (CENTRE_SIGMA * (x1 - x0))) ** 2
        gy = ((y - (y0 + y1) / 2) / (CENTRE_SIGMA * (y1 - y0))) ** 2
        values = np.exp(-0.5 * (gy[:, None] + gx[None, :]))
    return _quantise(values)


def build(spec: Spec, photo: np.ndarray, window: tuple[int, int, int, int], device="auto"):
    """The focus map for a loaded photo (RGB floats, upright), as `photo_map` gives it, and
    what found the subject (a `vision.Run`, with `subject`; else None). `subject` runs the
    models of vision.py on the whole photo, on `device` ("auto", "cpu" or "gpu")."""
    size = (photo.shape[1], photo.shape[0])
    if spec.kind != "subject":
        return photo_map(spec, size, window), None
    from . import vision

    values, run = vision.subject_map(photo, map_size(size), device)
    return _quantise(values), run


def map_size(size: tuple[int, int]) -> tuple[int, int]:
    """The focus map's width and height for a photo of `size`: MAP_SIDE on the long side,
    never more than the photo."""
    w, h = size
    scale = min(1.0, MAP_SIDE / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def _quantise(values: np.ndarray) -> np.ndarray:
    """8 bits, so that focus.png gives the same map back."""
    return np.round(np.clip(values, 0, 1) * 255) / 255


def _cover(lo: float, hi: float, n: int, total: float) -> np.ndarray:
    """For n pixels that split 0..total evenly, the share of each between lo and hi."""
    edges = np.arange(n + 1) * total / n
    return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None) * n / total


def _mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    """A mask image as values 0 to 1, stretched to `size`: white is the subject; transparent
    pixels count as black (a mask painted on an empty layer). 16-bit and float masks scale
    like photos (`image.deep_grey`). It shrinks in PIL first: a full-size float copy of a
    camera-sized mask would take hundreds of MB."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if is_deep(im):
                grey = Image.fromarray(deep_grey(im, np.float32) * 255)
            else:
                grey = flatten(im, (0, 0, 0)).convert("L")
            small = np.asarray(grey.resize(size, Image.BILINEAR), dtype=np.float64)
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"cannot read {path} as a focus mask: {e}") from None
    return np.clip(small / 255, 0, 1)


def mask_note(spec: Spec, size: tuple[int, int]) -> str | None:
    """A note when a mask does not have the photo's proportions (upright, as `image.load`
    turns it): painted over a crop, or over a copy the editor did not turn, it would mark
    the wrong part once stretched to the photo."""
    if spec.kind != "mask":
        return None
    with Image.open(spec.path) as im:
        mw, mh = im.size
        if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # EXIF: turned by 90 degrees
            mw, mh = mh, mw
    w, h = size
    if abs(math.log((mw / mh) / (w / h))) <= ASPECT:
        return None
    return (
        f"the focus mask {spec.path.name} is {mw} x {mh} px and the photo {w} x {h} px "
        "(upright): the mask is stretched to the photo, so the subject may move; paint the "
        "mask over the whole photo, the right way up"
    )


def on_grid(
    values: np.ndarray,
    size: tuple[int, int],
    window: tuple[int, int, int, int],
    rows: int,
    cols: int,
) -> np.ndarray:
    """The map at the picture's grid points (rows x cols, row 0 at the top): the part of the
    photo-space map that `window` keeps, resized like the photo."""
    w, h = size
    mh, mw = values.shape
    x0, y0, x1, y1 = window
    box = (x0 * mw / w, y0 * mh / h, x1 * mw / w, y1 * mh / h)
    im = Image.fromarray(values.astype(np.float32))
    return np.clip(np.asarray(im.resize((cols, rows), Image.BILINEAR, box=box), np.float64), 0, 1)


def check_subject(subject: np.ndarray, text: str) -> None:
    if subject.any():
        return
    kind = parse(text).kind
    if kind == "subject":
        raise ValueError(
            "--focus subject found no subject in the picture: no person or main object covers "
            f"at least {MIN_FOUND:.1%} of the photo, or the crop leaves it out; for a landscape "
            "use centre, X,Y,W,H or a mask image"
        )
    if kind == "box":
        why = (
            "it is smaller than a point of the print, or the crop leaves it out (with a "
            "shaped panel: or the frame hides it); make it larger"
        )
    else:
        why = "use white (at least mid-grey) where the subject is, black for the rest"
    raise ValueError(f"--focus {text} marks no part of the picture as the subject: {why}")


def save(values: np.ndarray, path: Path) -> Path:
    """focus.png: the map as 8-bit grey (white = the subject)."""
    to_pil(values).save(path)
    return path
