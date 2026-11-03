from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lithophane_forge import focus, image, render


@pytest.mark.parametrize(
    "text, kind",
    [("centre", "centre"), ("Center", "centre"), ("10, 20,30,40", "box"), ("me.png", "mask")],
)
def test_parse(text, kind):
    spec = focus.parse(text)
    assert spec.kind == kind
    if kind == "box":
        assert spec.box == (10, 20, 30, 40)
    if kind == "mask":
        assert spec.path.name == "me.png"


@pytest.mark.parametrize("text", ["1,2,3", "1,2,0,4", "1,2,3,4,5"])
def test_a_bad_box_is_an_error(text):
    with pytest.raises(ValueError, match="X,Y,W,H"):
        focus.parse(text)


@pytest.mark.parametrize("text", ["2024", "12-34", "+5"])
def test_a_name_without_commas_is_a_mask(text):
    assert focus.parse(text) == focus.Spec("mask", path=Path(text))


def test_an_empty_focus_is_an_error():
    with pytest.raises(ValueError, match="needs a value"):
        focus.parse("  ")


def old_frame(rgb, aspect, box):
    """`image.frame` as it was before `image.window`: the crop by slicing, then the largest
    centred window with the aspect."""
    if box:
        x, y, w, h = box
        rgb = rgb[max(y, 0) : y + h, max(x, 0) : x + w]
    h, w = rgb.shape[:2]
    if w / h > aspect:
        nw = max(1, round(h * aspect))
        return rgb[:, (w - nw) // 2 : (w - nw) // 2 + nw]
    nh = max(1, round(w / aspect))
    return rgb[(h - nh) // 2 : (h - nh) // 2 + nh]


def test_window_keeps_what_frame_kept():
    """A map in photo space follows the photo with `window`: it must cut what `frame` has
    always cut."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        h, w = (int(v) for v in rng.integers(20, 300, 2))
        rgb = rng.random((h, w, 3))
        aspect = float(rng.uniform(0.3, 3))
        box = None
        if rng.random() < 0.6:
            x, y = int(rng.integers(-20, w // 2)), int(rng.integers(-20, h // 2))
            box = (x, y, int(rng.integers(30, w + 40)), int(rng.integers(30, h + 40)))
        x0, y0, x1, y1 = image.window((w, h), aspect, box)
        assert np.array_equal(rgb[y0:y1, x0:x1], old_frame(rgb, aspect, box))
        assert np.array_equal(image.frame(rgb, aspect, box), old_frame(rgb, aspect, box))


def test_a_crop_above_the_photo_is_an_error():
    """Slicing took y + h < 0 as an index from the bottom; now it is outside the photo."""
    with pytest.raises(ValueError, match="outside the photo"):
        image.window((100, 80), 1.0, (0, -50, 40, 20))


def test_a_box_follows_the_crop():
    size = (400, 300)
    win = image.window(size, 1.0, (50, 0, 300, 300))  # the picture shows x 50 to 350
    assert win == (50, 0, 350, 300)
    values = focus.photo_map(focus.parse("100,50,100,100"), size, win)
    assert values.shape == (300, 400) and set(np.unique(values)) == {0.0, 1.0}
    grid = focus.on_grid(values, size, win, 60, 60)  # 5 photo pixels per grid point
    assert grid[20, 20] == 1 and grid[5, 5] == 0 and grid[20, 45] == 0
    rows, cols = np.nonzero(grid >= focus.SUBJECT)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (10, 29, 10, 29)


def test_a_box_outside_the_photo():
    with pytest.raises(ValueError, match="outside the photo"):
        focus.photo_map(focus.parse("500,0,10,10"), (400, 300), (0, 0, 400, 300))


def test_the_map_is_small_and_keeps_the_photo_shape():
    values = focus.photo_map(focus.parse("centre"), (5712, 4284), (0, 0, 5712, 4284))
    assert values.shape == (750, 1000)
    assert focus.photo_map(focus.parse("centre"), (300, 200), (0, 0, 300, 200)).shape == (200, 300)


def test_centre_is_the_middle_of_the_picture():
    size, win = (400, 200), (100, 0, 300, 200)  # the picture is the middle of the photo
    values = focus.photo_map(focus.parse("centre"), size, win)
    assert values[100, 199:201].min() == values.max()  # the peak is in the middle
    assert np.array_equal(values[:, 100:200], values[:, 299:199:-1])  # and even around it
    assert values[100, 100] == pytest.approx(np.exp(-2), abs=0.01)  # sigma = a quarter
    assert values[0, 0] < 0.01


def test_a_mask_round_trips_through_focus_png(tmp_path):
    """focus.png gives back the same map: passing it to --focus repeats the plan."""
    size, win = (4000, 3000), (500, 0, 3500, 3000)
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[100:200, 150:300] = 255
    mask[50:60, 10:20] = 128
    Image.fromarray(mask).save(tmp_path / "mask.png")
    first = focus.photo_map(focus.parse(str(tmp_path / "mask.png")), size, win)
    focus.save(first, tmp_path / "focus.png")
    again = focus.photo_map(focus.parse(str(tmp_path / "focus.png")), size, win)
    assert np.array_equal(first, again)
    grid = focus.on_grid(first, size, win, 90, 90)
    assert np.array_equal(grid, focus.on_grid(again, size, win, 90, 90))
    assert grid[45, 45] == 1 and grid[10, 10] == 0


def test_a_mask_is_stretched_and_transparent_is_background(tmp_path):
    rgba = np.zeros((200, 200, 4), dtype=np.uint8)  # transparent everywhere ...
    rgba[:, :100] = (255, 255, 255, 255)  # ... but an opaque white left half
    Image.fromarray(rgba).save(tmp_path / "layer.png")
    values = focus.photo_map(focus.parse(str(tmp_path / "layer.png")), (400, 100), (0, 0, 400, 100))
    assert values.shape == (100, 400)
    assert (values[:, :190] == 1).all() and (values[:, 210:] == 0).all()


def test_an_unreadable_mask(tmp_path):
    (tmp_path / "x.png").write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read .* as a focus mask"):
        focus.photo_map(focus.parse(str(tmp_path / "x.png")), (40, 30), (0, 0, 40, 30))


def test_no_subject_is_an_error():
    focus.check_subject(np.array([False, True]), "centre")
    with pytest.raises(ValueError, match="marks no part of the picture"):
        focus.check_subject(np.zeros((4, 4), dtype=bool), "mask.png")


def test_the_preview_outlines_the_subject():
    im = Image.new("RGB", (200, 100), (128, 128, 128))
    subject = np.zeros((10, 20), dtype=bool)
    subject[2:8, 5:15] = True
    out = np.asarray(render.outlined(im, subject, box=200))
    pink = (out == render.FOCUS).all(axis=2)
    assert pink[20:80, 50:150].any()  # along the subject's edge ...
    assert not pink[40:60, 70:130].any()  # ... not inside it
    assert not pink[:10].any() and not pink[:, :40].any()  # nor far outside


def test_a_tiny_box_still_marks_its_place():
    """A box smaller than a pixel of the map marks the pixels it covers in part, so the
    error, if any, is about its size, not about mask colours."""
    size = (5712, 4284)
    values = focus.photo_map(focus.parse("100,100,4,4"), size, (0, 0, *size))
    assert 0 < values.max() < focus.SUBJECT
    with pytest.raises(ValueError, match="smaller than a point of the print"):
        focus.check_subject(values >= focus.SUBJECT, "100,100,4,4")


@pytest.mark.parametrize("level, subject", [(200, False), (40000, True), (65535, True), (1, False)])
def test_16_bit_masks_scale_like_photos(tmp_path, level, subject):
    """I;16 always counts in 65535ths, as for photos: a dim 16-bit mask is dim, and a 0/1
    label mask is black (as the same mask on 8 bits)."""
    grey = np.zeros((30, 40), dtype=np.uint16)
    grey[10:20, 10:30] = level
    Image.fromarray(grey).save(tmp_path / "deep.png")
    with Image.open(tmp_path / "deep.png") as im:
        assert im.mode.startswith("I")
    values = focus.photo_map(focus.parse(str(tmp_path / "deep.png")), (40, 30), (0, 0, 40, 30))
    assert (values[15, 20] >= focus.SUBJECT) == subject
    assert values[15, 20] == pytest.approx(round(level / 65535 * 255) / 255, abs=1 / 255)


def test_a_mask_with_other_proportions_gets_a_note(tmp_path):
    Image.new("L", (400, 300), 255).save(tmp_path / "wide.png")
    Image.new("L", (300, 400), 255).save(tmp_path / "tall.png")
    turned = Image.new("L", (400, 300), 255)  # stored wide, shown tall (EXIF orientation 6)
    exif = Image.Exif()
    exif[0x0112] = 6
    turned.save(tmp_path / "turned.jpg", exif=exif)
    size = (3000, 4000)  # an upright portrait photo
    note = focus.mask_note(focus.parse(str(tmp_path / "wide.png")), size)
    assert "400 x 300 px and the photo 3000 x 4000 px" in note
    assert focus.mask_note(focus.parse(str(tmp_path / "tall.png")), size) is None
    assert focus.mask_note(focus.parse(str(tmp_path / "turned.jpg")), size) is None
    assert focus.mask_note(focus.parse("centre"), size) is None
