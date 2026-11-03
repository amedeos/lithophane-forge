import hashlib
import http.client
import io
import sys
import types
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import ClassVar
from urllib.error import URLError

import numpy as np
import pytest
from conftest import write_photo
from PIL import Image

from lithophane_forge import focus, image, vision

CUDA, CPU = vision.CUDA, vision.CPU


def test_models_table():
    models = vision.models()
    assert list(models) == ["u2net_human_seg", "modnet", "isnet"]
    for m in models.values():
        assert m.url.startswith("https://") and len(m.sha256) == 64 and m.size > 10**6
        assert (m.input is None) != (m.short_side is None)
        assert len(m.mean) == len(m.std) == 3 and m.resample in ("lanczos", "bilinear")
        assert m.licence.startswith("Apache-2.0") and m.file == f"{m.name}.onnx"


def test_prepare_fixed_input():
    """u2net: stretched to 320 x 320, divided by the largest value, ImageNet mean and std,
    channels in R, G, B order."""
    im = Image.new("RGB", (50, 30), (255, 0, 0))
    x = vision.prepare(im, vision.models()["u2net_human_seg"])
    assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32
    assert np.allclose(x[0, :, 0, 0], [(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])


def test_prepare_dynamic_input():
    """MODNet: the short side at 512, both sides a multiple of 32, (x / 255 - 0.5) / 0.5."""
    im = Image.new("RGB", (300, 200), (255, 0, 51))
    x = vision.prepare(im, vision.models()["modnet"])
    assert x.shape == (1, 3, 512, 768)
    assert np.allclose(x[0, :, 7, 9], [1, -1, -0.6], atol=1e-6)


def test_to_map_keeps_the_orientation():
    y = np.zeros((1, 1, 4, 8), np.float32)
    y[0, 0, :, :4] = 1.5  # left half, above 1: clipped
    y[0, 0, 0, :] = np.minimum(y[0, 0, 0, :], -0.2)  # top row below 0: clipped
    m = vision.to_map(y, (16, 6))
    assert m.shape == (6, 16) and m.min() >= 0 and m.max() <= 1
    assert m[3, :6].min() > 0.9 and m[3, 10:].max() < 0.1


def test_drop_small():
    """A part stays when its core covers MIN_FOUND of the map; a small one goes with its
    soft rim; a soft region with no core goes too."""
    m = np.zeros((100, 100))
    m[10:40, 10:40] = 0.33  # a big part's rim (one model of three)
    m[15:35, 15:35] = 1.0  # its core: 4 % of the map
    m[70:76, 70:76] = 0.33  # a speck's rim
    m[72:74, 72:74] = 0.9  # a speck: 0.04 %
    m[80:95, 5:20] = 0.33  # one model alone
    out = vision.drop_small(m)
    assert np.array_equal(out[:50, :50], m[:50, :50])
    assert not out[60:, :].any()


def test_folder(monkeypatch, tmp_path):
    monkeypatch.setenv(vision.FOLDER_ENV, str(tmp_path / "mine"))
    assert vision.folder() == tmp_path / "mine"
    monkeypatch.delenv(vision.FOLDER_ENV)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert vision.folder() == tmp_path / "cache" / "lithophane-forge" / "models"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert vision.folder() == tmp_path / ".cache" / "lithophane-forge" / "models"


def test_prepare_caps_a_panorama():
    """MODNet's long side stays at most 1024: a strip does not make a huge input."""
    x = vision.prepare(Image.new("RGB", (6000, 500)), vision.models()["modnet"])
    assert x.shape == (1, 3, 64, 1024)


def tiny_model(data: bytes):
    m = vision.models()["modnet"]
    return replace(m, name="tiny", sha256=hashlib.sha256(data).hexdigest(), size=len(data))


class Broken(io.BytesIO):
    """A response that breaks while it is read (a chunked transfer cut short)."""

    def read(self, n=-1):
        raise http.client.IncompleteRead(b"")


def serve(monkeypatch, data):
    """urlopen gives `data` (bytes), fails like a network without a connection (None), or
    gives a response that breaks while it is read ("broken")."""
    urls = []

    def urlopen(url, timeout):
        urls.append(url)
        if data is None:
            raise URLError("no connection")
        return Broken() if data == "broken" else io.BytesIO(data)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return urls


def test_download_checks_the_hash(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(vision.FOLDER_ENV, str(tmp_path))
    model = tiny_model(b"model bytes")
    urls = serve(monkeypatch, b"model bytes")
    path = vision.model_path(model)
    assert path == tmp_path / "tiny.onnx" and path.read_bytes() == b"model bytes"
    assert list(tmp_path.iterdir()) == [path]  # no temporary file left
    assert "downloading the model tiny (0 MB)" in capsys.readouterr().err
    assert vision.model_path(model) == path and len(urls) == 1  # in the folder: no download
    assert vision.state(model) == "ok"
    path.write_bytes(b"model bytez")  # the same size, other bytes
    assert vision.state(model) == "damaged"
    path.unlink()
    assert vision.state(model) == "missing"


@pytest.mark.parametrize(
    "data, message",
    [
        (b"other bytes", "is not the model tiny.*changed upstream"),
        (None, "cannot download"),
        ("broken", "cannot download"),
    ],
)
def test_failed_download_leaves_no_file(monkeypatch, tmp_path, data, message):
    monkeypatch.setenv(vision.FOLDER_ENV, str(tmp_path))
    serve(monkeypatch, data)
    with pytest.raises(ValueError, match=message):
        vision.model_path(tiny_model(b"model bytes"))
    assert list(tmp_path.iterdir()) == []


def test_folder_that_cannot_be_written(monkeypatch, tmp_path):
    (tmp_path / "file").write_text("x", encoding="utf-8")
    monkeypatch.setenv(vision.FOLDER_ENV, str(tmp_path / "file" / "models"))
    serve(monkeypatch, b"model bytes")
    with pytest.raises(ValueError, match=f"cannot write the model.*set \\${vision.FOLDER_ENV}"):
        vision.model_path(tiny_model(b"model bytes"))


class FakeSession:
    """An onnxruntime.InferenceSession that behaves like the real one with CUDA: when CUDA
    does not load, or fails while running (`fails`: "epfail" or "oom"), it prints the same
    kind of message and moves to the CPU, unless its fallback is off. It returns
    `output` (a function of the model's path)."""

    loads_cuda = True
    fails = None
    output = staticmethod(lambda path: np.ones((1, 1, 8, 8), np.float32))
    asked: ClassVar[list] = []  # the providers each session asked for

    def __init__(self, path, options, providers):
        if Path(path).name == "damaged.onnx":
            raise fake_errors.InvalidProtobuf("INVALID_PROTOBUF")
        FakeSession.asked.append(providers)
        self.path, self.providers, self.fallback = path, list(providers), True
        if CUDA in providers and not self.loads_cuda:
            print("*************** EP Error ***************")
            print(f"EP Error libcudnn.so.9: cannot open shared object file when using {providers}")
            self.providers = [CPU]

    def disable_fallback(self):
        self.fallback = False

    def get_providers(self):
        return self.providers

    def get_inputs(self):
        return [types.SimpleNamespace(name="input")]

    def run(self, outputs, feeds):
        assert feeds["input"].dtype == np.float32
        if self.providers[0] == CUDA and self.fails == "oom":
            raise fake_errors.RuntimeException("CUDA failure 2: out of memory")
        if self.providers[0] == CUDA and self.fails == "epfail":
            if not self.fallback:
                raise fake_errors.EPFail("CUDA kernel failed")
            print("EP Error: CUDA kernel failed using ['CUDAExecutionProvider']")
            self.providers = [CPU]
        return [self.output(self.path)]


fake_errors = types.SimpleNamespace(
    **{
        name: type(name, (Exception,), {})
        for name in ("InvalidProtobuf", "InvalidGraph", "Fail", "EPFail", "RuntimeException")
    }
)


@pytest.fixture
def fake_ort(monkeypatch):
    """A stand-in for onnxruntime: set `offered` to the providers it has."""
    ort = types.ModuleType("onnxruntime")
    ort.__version__ = "0.0"
    ort.offered = [CPU]
    ort.preloaded = []
    ort.get_available_providers = lambda: list(ort.offered)
    ort.set_default_logger_severity = lambda level: None
    ort.SessionOptions = types.SimpleNamespace
    ort.preload_dlls = lambda: ort.preloaded.append(True)
    ort.capi = types.SimpleNamespace(onnxruntime_pybind11_state=fake_errors)
    ort.InferenceSession = FakeSession
    FakeSession.asked = []
    monkeypatch.setattr(FakeSession, "loads_cuda", True)
    monkeypatch.setattr(FakeSession, "fails", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return ort


X = np.zeros((1, 3, 8, 8), np.float32)


@pytest.mark.parametrize(
    "offered, loads, fails, device, asked, used",
    [
        ([CPU], True, None, "auto", [CPU], "CPU"),
        ([CUDA, CPU], True, None, "auto", [CUDA, CPU], "CUDA"),
        ([CUDA, CPU], False, None, "auto", [CUDA, CPU], "CPU"),  # CUDA did not load
        ([CUDA, CPU], True, "epfail", "auto", [CUDA, CPU], "CPU"),  # it failed while running
        ([CUDA, CPU], True, None, "cpu", [CPU], "CPU"),
        ([CUDA, CPU], True, None, "gpu", [CUDA, CPU], "CUDA"),
    ],
)
def test_device(fake_ort, monkeypatch, capsys, offered, loads, fails, device, asked, used):
    """The device is the one that really ran, and ONNX Runtime's messages stay quiet."""
    fake_ort.offered = offered
    monkeypatch.setattr(FakeSession, "loads_cuda", loads)
    monkeypatch.setattr(FakeSession, "fails", fails)
    _, got = vision.infer(vision.session(Path("m.onnx"), device), X, device)
    assert got == used and FakeSession.asked == [asked]
    assert fake_ort.preloaded == ([True] if CUDA in asked else [])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "offered, loads, fails, message",
    [
        ([CPU], True, None, "has no CUDA; install the GPU extra"),
        ([CUDA, CPU], False, None, "CUDA did not load .EP Error libcudnn"),
        ([CUDA, CPU], True, "epfail", "a model failed on the CUDA .CUDA kernel failed.; try --"),
    ],
)
def test_gpu_without_cuda_is_an_error(fake_ort, monkeypatch, offered, loads, fails, message):
    """--device gpu never moves to the CPU: not when it opens, not while it runs."""
    fake_ort.offered = offered
    monkeypatch.setattr(FakeSession, "loads_cuda", loads)
    monkeypatch.setattr(FakeSession, "fails", fails)
    with pytest.raises(ValueError, match=message):
        vision.infer(vision.session(Path("m.onnx"), "gpu"), X, "gpu")


def test_out_of_gpu_memory_is_an_error(fake_ort, monkeypatch):
    fake_ort.offered = [CUDA, CPU]
    monkeypatch.setattr(FakeSession, "fails", "oom")
    with pytest.raises(ValueError, match="failed on the CUDA .*out of memory.*--device cpu"):
        vision.infer(vision.session(Path("m.onnx"), "auto"), X, "auto")


def test_damaged_model_is_an_error(fake_ort):
    with pytest.raises(ValueError, match="cannot load the model damaged.onnx.*delete it"):
        vision.session(Path("damaged.onnx"), "auto")


def test_gpu_is_checked_before_any_download(fake_ort, monkeypatch):
    def download(model):
        raise AssertionError("downloaded")

    monkeypatch.setattr(vision, "model_path", download)
    with pytest.raises(ValueError, match="has no CUDA"):
        vision.subject_map(np.zeros((30, 40, 3)), (40, 30), "gpu")


def test_onnxruntime_that_does_not_load(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import raises ImportError
    with pytest.raises(ValueError, match="ONNX Runtime does not load"):
        vision.runtime()


def test_subject_is_the_mean(fake_ort, monkeypatch):
    """Two models of three make a point part of the subject; one alone does not."""
    monkeypatch.setattr(vision, "model_path", lambda m: Path(f"{m.name}.onnx"))
    left = np.zeros((1, 1, 40, 40), np.float32)
    left[..., :20] = 1  # the left half
    top = np.zeros((1, 1, 40, 40), np.float32)
    top[..., :20, :] = 1  # the top half
    outputs = {"u2net_human_seg.onnx": left, "modnet.onnx": left, "isnet.onnx": top}
    monkeypatch.setattr(FakeSession, "output", staticmethod(lambda p: outputs[Path(p).name]))
    photo = np.random.default_rng(0).random((60, 80, 3))
    values, run = vision.subject_map(photo, (40, 30), "auto")
    assert values.shape == (30, 40) and run.device == "CPU"
    assert run.models == ("u2net_human_seg", "modnet", "isnet") and run.seconds >= 0
    assert np.allclose(values[:12, :18], 1)  # all three
    assert np.allclose(values[18:, :18], 2 / 3)  # two
    assert np.allclose(values[:12, 22:], 1 / 3)  # one: it touches the part, so it stays
    assert np.allclose(values[18:, 22:], 0)


@pytest.mark.vision
def test_real_models(tmp_path):
    """The real models on a small test photo: a finite map of the right size, the same
    twice. Skipped without ONNX Runtime or without the models in the folder."""
    if not vision.available():
        pytest.skip(f"no ONNX Runtime: {vision.EXTRA}")
    missing = [m.name for m in vision.models().values() if vision.state(m) != "ok"]
    if missing:
        pytest.skip(f"models not in {vision.folder()}: lithophane-forge models --download")
    rgb = image.load(write_photo(tmp_path / "photo.png", (320, 240)))
    values, run = focus.build(focus.parse("subject"), rgb, (0, 0, 320, 240), "cpu")
    again, _ = focus.build(focus.parse("subject"), rgb, (0, 0, 320, 240), "cpu")
    assert values.shape == (240, 320) and np.isfinite(values).all()
    assert 0 <= values.min() and values.max() <= 1 and np.array_equal(values, again)
    assert run.device == "CPU" and run.models == tuple(vision.models())
