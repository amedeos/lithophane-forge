"""Colour mode: find the subject of a photo by itself (--focus subject).

Three small segmentation models (data/models.toml) run through ONNX Runtime, one after the
other, each on the whole photo. The subject is the mean of their maps, so a point needs two
of the three models (a mean of at least 0.5) to be part of it: each model alone fails
somewhere (a lost hand, a rock or a fence added), but rarely where another one fails. With
people in the photo the subject is the people; without people it is the main object; an
open landscape has none. Parts smaller than focus.MIN_FOUND of the photo are dropped: on the
owner's photos the false ones (reed stalks, specks) were at most 0.07 % of the photo and the
real ones at least 0.6 %.

ONNX Runtime comes with an optional extra: `uv sync --extra vision` (the CPU) or `uv sync
--extra vision-gpu` (also an NVIDIA card, CUDA 13). This module imports it only inside the
functions that run a model, so the rest of the program works without it. The models are
downloaded on first use into `folder()` and checked against their SHA-256.
"""

import contextlib
import hashlib
import importlib.util
import io
import os
import sys
import time
import tomllib
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

from .focus import MIN_FOUND, SUBJECT
from .image import to_pil

EXTRA = "uv sync --extra vision"
GPU_EXTRA = "uv sync --extra vision-gpu"
NEEDS = f"--focus subject needs ONNX Runtime: {EXTRA} (or {GPU_EXTRA} with an NVIDIA card)"
FOLDER_ENV = "LITHOPHANE_FORGE_MODELS"
SUPPORT = 0.1  # a part reaches out to where the mean map falls below this
CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Model:
    """A row of data/models.toml."""

    name: str
    title: str
    url: str
    sha256: str
    size: int  # bytes
    licence: str
    source: str
    resample: str  # "lanczos" or "bilinear"
    scale: str | float  # "max": divide by the largest value; a number: divide by it
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    input: tuple[int, int] | None = None  # fixed input (width, height)
    short_side: int | None = None  # dynamic input: the short side, multiples of 32
    long_side: int | None = None  # dynamic input: at most this long (a panorama)

    @property
    def file(self) -> str:
        return f"{self.name}.onnx"


@dataclass(frozen=True)
class Run:
    """What found the subject."""

    models: tuple[str, ...]
    device: str  # "CPU", "CUDA", or "CUDA and CPU" when one model fell back
    seconds: float


def available() -> bool:
    """Whether ONNX Runtime is installed (it does not import it)."""
    return importlib.util.find_spec("onnxruntime") is not None


@cache
def models() -> dict[str, Model]:
    text = resources.files("lithophane_forge").joinpath("data/models.toml").read_text("utf-8")
    out = {}
    for name, row in tomllib.loads(text)["models"].items():
        row = {k: tuple(v) if isinstance(v, list) else v for k, v in row.items()}
        out[name] = Model(name=name, **row)
    return out


def folder() -> Path:
    """Where the models live: $LITHOPHANE_FORGE_MODELS, or the user's cache."""
    if os.environ.get(FOLDER_ENV):
        return Path(os.environ[FOLDER_ENV]).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache_home) / "lithophane-forge" / "models"


def model_path(model: Model) -> Path:
    """The model's file, downloaded first if it is not in the folder. Only its size is
    checked here; `state` checks the hash (the `models` command)."""
    path = folder() / model.file
    if not (path.is_file() and path.stat().st_size == model.size):
        fetch(model, path)
    return path


def fetch(model: Model, path: Path) -> None:
    """Download the model to `path`, checking its SHA-256. It goes through a temporary file
    of its own in the same folder, so two runs at once do not write into one file, and a
    failed download leaves nothing behind."""
    import tempfile

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f"{model.file}.", suffix=".part", dir=path.parent)
    except OSError as e:
        raise _cannot_write(path, e) from None
    part = Path(name)
    print(
        f"downloading the model {model.name} ({model.size / 1e6:.0f} MB) to {path.parent}",
        file=sys.stderr,
    )
    digest = hashlib.sha256()
    done = False
    try:
        try:
            with open(fd, "wb") as f:
                _copy(model, path, f, digest)
        except OSError as e:  # writing: network errors come as ValueError
            raise _cannot_write(path, e) from None
        if digest.hexdigest() != model.sha256:
            raise ValueError(
                f"the file at {model.url} is not the model {model.name} (SHA-256 "
                f"{digest.hexdigest()}, expected {model.sha256}): the download broke, or the "
                "file changed upstream. Try again; if it happens again, the model table "
                "(data/models.toml) needs an update"
            )
        try:
            part.replace(path)
        except OSError as e:
            raise _cannot_write(path, e) from None
        done = True
    finally:
        if not done:
            part.unlink(missing_ok=True)


def _copy(model: Model, path: Path, f, digest) -> None:
    """The download itself: a network error is a ValueError, a write error an OSError."""
    import http.client
    import urllib.request

    def cannot(e):
        return ValueError(
            f"cannot download {model.url} ({e}): check the connection, or put {model.file} "
            f"in {path.parent}"
        )

    try:
        r = urllib.request.urlopen(model.url, timeout=60)
    except (OSError, http.client.HTTPException) as e:  # URLError and timeouts too
        raise cannot(e) from None
    with r:
        while True:
            try:
                chunk = r.read(CHUNK)
            except (OSError, http.client.HTTPException) as e:  # IncompleteRead too
                raise cannot(e) from None
            if not chunk:
                return
            digest.update(chunk)
            f.write(chunk)


def _cannot_write(path: Path, e: OSError) -> ValueError:
    return ValueError(
        f"cannot write the model to {path.parent} ({e}): free some space, or set "
        f"${FOLDER_ENV} to another folder"
    )


def state(model: Model) -> str:
    """'ok', 'missing' or 'damaged' (the file's SHA-256 is not the model's)."""
    path = folder() / model.file
    if not path.is_file():
        return "missing"
    with open(path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return "ok" if digest == model.sha256 else "damaged"


def runtime() -> tuple[str, list[str]]:
    """ONNX Runtime's version and the providers it offers."""
    ort = _onnxruntime()
    return ort.__version__, ort.get_available_providers()


def _onnxruntime():
    """ONNX Runtime, imported: a broken install is a ValueError, not a traceback."""
    try:
        import onnxruntime
    except (ImportError, OSError) as e:
        raise ValueError(
            f"ONNX Runtime does not load ({e}): install it again with {EXTRA} (or {GPU_EXTRA})"
        ) from None
    return onnxruntime


def _errors(ort) -> tuple[type, ...]:
    """ONNX Runtime's own exceptions (InvalidProtobuf, EPFail, RuntimeException, ...)."""
    found = vars(ort.capi.onnxruntime_pybind11_state).values()
    return tuple(c for c in found if isinstance(c, type) and issubclass(c, Exception))


def _wants_cuda(ort, device: str) -> bool:
    """Whether to ask for CUDA: `auto` when ONNX Runtime offers it, always with `gpu`
    (an error if it does not offer it), never with `cpu`."""
    cuda = device != "cpu" and CUDA in ort.get_available_providers()
    if device == "gpu" and not cuda:
        raise ValueError(
            f"--device gpu: this ONNX Runtime has no CUDA; install the GPU extra ({GPU_EXTRA})"
        )
    return cuda


def _device(sess) -> str:
    return "CUDA" if sess.get_providers()[0] == CUDA else "CPU"


def _check(sess, device: str, said: str) -> None:
    """`gpu` must really run on CUDA. ONNX Runtime falls back to the CPU by itself when
    CUDA fails to load or to run, and prints why (`said`)."""
    if device == "gpu" and _device(sess) != "CUDA":
        why = next((ln for ln in said.splitlines() if ln.startswith("EP Error")), "")
        raise ValueError(
            f"--device gpu: CUDA did not load ({why or 'no reason given'}); an NVIDIA driver "
            "too old for CUDA 13, or missing libraries; use --device auto or cpu"
        )


def session(path: Path, device: str):
    """An ONNX Runtime session for the model at `path`. `auto` asks for CUDA when ONNX
    Runtime offers it, else the CPU; `gpu` without CUDA is an error. ONNX Runtime's
    warnings and its fallback messages stay off the terminal."""
    ort = _onnxruntime()
    ort.set_default_logger_severity(3)  # errors only
    options = ort.SessionOptions()
    options.log_severity_level = 3
    cuda = _wants_cuda(ort, device)
    said = io.StringIO()
    with contextlib.redirect_stdout(said):  # ONNX Runtime prints with print()
        if cuda and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # the CUDA and cuDNN libraries of the nvidia-* wheels
        try:
            sess = ort.InferenceSession(
                str(path), options, providers=[CUDA, CPU] if cuda else [CPU]
            )
        except _errors(ort) as e:
            raise ValueError(
                f"cannot load the model {path.name} ({e}): delete it and run again to "
                "download it once more"
            ) from None
    if device == "gpu":
        sess.disable_fallback()  # a CUDA failure while running stops, not moves to the CPU
    _check(sess, device, said.getvalue())
    return sess


def infer(sess, x: np.ndarray, device: str) -> tuple[np.ndarray, str]:
    """Run the session on `x`: its first output, and the device it really ran on (ONNX
    Runtime may move to the CPU when CUDA fails while running)."""
    ort = _onnxruntime()
    said = io.StringIO()
    try:
        with contextlib.redirect_stdout(said):
            y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    except _errors(ort) as e:  # out of GPU memory, for example
        hint = "; try --device cpu" if _device(sess) == "CUDA" else ""
        raise ValueError(f"a model failed on the {_device(sess)} ({e}){hint}") from None
    _check(sess, device, said.getvalue())
    return y, _device(sess)


def prepare(im: Image.Image, model: Model) -> np.ndarray:
    """The model's input for an RGB image: resized (stretched to a fixed input, as rembg
    does), scaled, normalised per channel, as a 1 x 3 x H x W float32 array."""
    if model.input:
        size = model.input
    else:
        s = model.short_side / min(im.size)
        if model.long_side:  # a panorama would make a huge input
            s = min(s, model.long_side / max(im.size))
        size = tuple(max(32, round(v * s) // 32 * 32) for v in im.size)
    resample = Image.LANCZOS if model.resample == "lanczos" else Image.BILINEAR
    x = np.asarray(im.resize(size, resample), dtype=np.float32)
    top = max(float(x.max()), 1e-6) if model.scale == "max" else float(model.scale)
    x = (x / top - np.array(model.mean, np.float32)) / np.array(model.std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def to_map(y: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A model's output (1 x 1 x H x W, a probability) as a map of `size` (width, height),
    with values from 0 to 1."""
    y = np.clip(np.asarray(y, dtype=np.float32)[0, 0], 0, 1)
    small = np.asarray(Image.fromarray(y).resize(size, Image.BILINEAR), dtype=np.float64)
    return np.clip(small, 0, 1)


def drop_small(values: np.ndarray, share: float = MIN_FOUND) -> np.ndarray:
    """The map without its small parts. A part is a connected region where the map is at
    least SUPPORT; it stays when its core (at least focus.SUBJECT) covers `share` of the
    map, else all of it goes to 0, so no soft rim is left behind."""
    from scipy import ndimage

    labels, n = ndimage.label(values >= SUPPORT)
    core = np.bincount(labels[values >= SUBJECT], minlength=n + 1)
    keep = core >= share * values.size
    keep[0] = True  # below SUPPORT: as it is
    return np.where(keep[labels], values, 0.0)


def subject_map(
    photo: np.ndarray, size: tuple[int, int], device: str = "auto"
) -> tuple[np.ndarray, Run]:
    """The subject of a photo (RGB floats from 0 to 1, upright, as `image.load` gives it)
    as a map of `size` (width, height) with values from 0 to 1, and what found it. ONNX
    Runtime and the device are checked before any download; the models run one at a time,
    so only one is in memory."""
    _wants_cuda(_onnxruntime(), device)
    paths = {name: model_path(m) for name, m in models().items()}
    im = to_pil(photo)
    start = time.perf_counter()
    maps, used = [], set()
    for name, model in models().items():
        sess = session(paths[name], device)
        y, dev = infer(sess, prepare(im, model), device)
        del sess
        maps.append(to_map(y, size))
        used.add(dev)
    run = Run(
        tuple(models()), " and ".join(sorted(used, reverse=True)), time.perf_counter() - start
    )
    return drop_small(np.mean(maps, axis=0)), run
