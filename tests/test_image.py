import numpy as np
import pytest

from lithophane_forge import image


def test_frame_keeps_the_aspect():
    rgb = np.zeros((300, 400, 3))
    for aspect in (0.5, 1.0, 4 / 3, 2.0):
        h, w = image.frame(rgb, aspect).shape[:2]
        assert w / h == pytest.approx(aspect, rel=0.01)


def test_crop_comes_first():
    rgb = np.zeros((300, 400, 3))
    rgb[50:150, 100:200] = 1
    out = image.frame(rgb, 1.0, (100, 50, 100, 100))
    assert out.shape[:2] == (100, 100) and out.min() == 1


def test_crop_alone_keeps_its_shape():
    rgb = np.zeros((300, 400, 3))
    assert image.crop(rgb, (10, 20, 100, 100)).shape[:2] == (100, 100)
    assert image.crop(rgb, None) is rgb
    with pytest.raises(ValueError, match="outside"):
        image.crop(rgb, (500, 0, 10, 10))


@pytest.mark.parametrize("mode, suffix", [("I;16", ".png"), ("I", ".tif")])
def test_load_16_bit_grey(tmp_path, mode, suffix):
    """16-bit grey images are scaled, not clipped at 255."""
    from PIL import Image

    ramp = np.linspace(0, 65535, 256).astype(np.uint16)[None, :].repeat(32, 0)
    path = tmp_path / f"grey16{suffix}"
    im = Image.fromarray(ramp)
    (im if mode == "I;16" else im.convert("I")).save(path)
    with Image.open(path) as check:
        assert check.mode == mode
    rgb = image.load(path)
    assert rgb.shape == (32, 256, 3)
    assert rgb.mean() == pytest.approx(0.5, abs=0.01)
    assert rgb[0, 0, 0] == 0 and rgb[0, -1, 2] == pytest.approx(1)


def test_parse_crop():
    assert image.parse_crop("1,2,3,4") == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        image.parse_crop("1,2,0,4")


def test_light_map_stretches_to_full_range():
    rgb = np.dstack([np.linspace(0.3, 0.6, 200)[None, :].repeat(50, 0)] * 3)
    light = image.light_map(rgb)
    assert light.min() == pytest.approx(0, abs=0.02)
    assert light.max() == pytest.approx(1, abs=0.02)
    assert np.all(np.diff(light[0]) >= -1e-9)


def test_invert_and_gamma():
    rgb = np.dstack([np.linspace(0, 1, 100)[None, :].repeat(10, 0)] * 3)
    plain = image.light_map(rgb)
    inverted = image.light_map(rgb, image.Tone(invert=True))
    # a negative in lightness: L* 100 - L*, so mid-grey stays mid-grey
    lstar = image.y_to_lstar(plain)
    assert np.allclose(image.y_to_lstar(inverted), 100 - lstar, atol=1e-6)
    darker = image.light_map(rgb, image.Tone(gamma=1.5))
    assert darker[0, 50] < plain[0, 50]


def test_lstar_round_trip():
    y = np.linspace(0, 1, 50)
    assert np.allclose(image.lstar_to_y(image.y_to_lstar(y)), y, atol=1e-9)


def test_analysis_notes():
    dark = np.full((100, 100, 3), 0.01)
    a = image.analyse(dark, (200, 200))
    text = " ".join(a.notes)
    assert "near black" in text and "fewer pixels" in text
