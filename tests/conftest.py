import numpy as np
import pytest
from PIL import Image

from lithophane_forge import printers, slicer
from lithophane_forge.optics import DEFAULTS


def write_photo(path, size=(160, 120)):
    """A small test photo: a gradient, a white disc at the upper left, a black bar at the
    lower right. Nothing in it is symmetric, so mirror mistakes show."""
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3))
    img[..., 0], img[..., 1], img[..., 2] = x / w, y / h, 0.5
    img[np.hypot(x - w * 0.3, y - h * 0.3) < h * 0.2] = 1.0
    img[int(h * 0.75) : int(h * 0.9), int(w * 0.65) : int(w * 0.9)] = 0.02
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return path


@pytest.fixture
def photo(tmp_path):
    return write_photo(tmp_path / "photo.png")


@pytest.fixture(scope="session")
def setup():
    return printers.setup("mk4", "petg")


@pytest.fixture
def petg():
    return DEFAULTS["petg"]


@pytest.fixture
def no_slicer(monkeypatch):
    """Run `make` as on a machine without PrusaSlicer."""
    monkeypatch.setattr(slicer, "command", lambda: None)


@pytest.fixture
def fake_subject(monkeypatch):
    """--focus subject without ONNX Runtime or the models: they "find" a disc in the middle
    of the photo. Returns the list of the devices asked for."""
    from lithophane_forge import vision

    devices = []

    def subject_map(photo, size, device="auto"):
        devices.append(device)
        w, h = size
        y, x = np.mgrid[0:h, 0:w]
        disc = np.hypot(x + 0.5 - w / 2, y + 0.5 - h / 2) < min(w, h) / 3
        return disc.astype(float), vision.Run(tuple(vision.models()), "CPU", 0.25)

    monkeypatch.setattr(vision, "available", lambda: True)
    monkeypatch.setattr(vision, "subject_map", subject_map)
    return devices
