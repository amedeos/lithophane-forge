import json
import zipfile

import numpy as np
import pytest
import trimesh
from PIL import Image
from test_palette import sunset

from lithophane_forge import forge, threemf
from lithophane_forge.forge import Job
from lithophane_forge.image import Tone


@pytest.fixture
def sunset_photo(tmp_path):
    path = tmp_path / "sunset.png"
    Image.fromarray((sunset(120, 160) * 255).astype(np.uint8)).save(path)
    return path


@pytest.mark.parametrize("shape", ["flat", "shaped"])
def test_colour_make(sunset_photo, setup, petg, tmp_path, no_slicer, shape):
    j = Job(
        photo=sunset_photo,
        setup=setup,
        optics=petg,
        out=tmp_path / "out",
        shape=shape,
        mode="color",
        colors=3,
        width=60,
        pitch=0.4,
    )
    res = forge.make(j)
    plan = json.loads(res.files["plan"].read_text(encoding="utf-8"))
    assert plan["mode"] == "color" and plan["pose"] == "lying"
    changes = plan["colour_changes"]
    assert len(changes) == len(plan["colour"]["print"]["filaments"]) - 1 >= 1
    with zipfile.ZipFile(res.files["3mf"]) as z:
        xml = z.read(threemf.M600).decode()
    for zv, colour in changes:
        assert f'print_z="{zv:.3f}"' in xml and colour in xml
    tm = trimesh.load(res.files["stl"])
    assert tm.bounds[1][2] <= j.color_range[1] + 1e-6  # nothing above the stack
    report = res.files["report"].read_text(encoding="utf-8")
    assert "M600" in report and "Ideal colours" in report
    lo, hi = j.color_range
    assert f"thickness {lo:g} to {hi:g} mm" in report  # the stack, not the mono optics
    assert "Tungsten" not in json.dumps(plan["colour"]["print"])  # abrasive: left out
    first = plan["colour"]["print"]["filaments"][0]["hex"]
    assert plan["printer"]["filament_colour"] == first == forge.first_colour(j, res)
    assert "left out as abrasive" in plan["colour"]["source"]
    assert plan["optics"] is None and plan["color_range_mm"] == [lo, hi]


def test_colour_mode_checks(setup, petg, tmp_path):
    with pytest.raises(ValueError, match="lying"):
        Job(photo=tmp_path / "x", setup=setup, optics=petg, mode="color", shape="curved").check()
    with pytest.raises(ValueError, match="--colors"):
        Job(photo=tmp_path / "x", setup=setup, optics=petg, mode="color", colors=12).check()


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"color_range": (0.61, 0.69)}, "holds 0 layer tops"),
        ({"color_range": (2.95, 3.0)}, "holds 1 layer tops"),
        ({"color_range": (0.6, 1.0), "colors": 6}, "6 colours need at least 6"),
        ({"tone": Tone(invert=True)}, "mono mode only"),
        ({"tone": Tone(clahe=0.01)}, "mono mode only"),
    ],
)
def test_colour_options_are_checked(setup, petg, tmp_path, kw, message):
    """Options colour mode cannot use are errors, not crashes or silent no-ops."""
    j = Job(photo=tmp_path / "x", setup=setup, optics=petg, mode="color", **kw)
    with pytest.raises(ValueError, match=message):
        j.check()


def test_colour_frame_is_the_whole_stack(setup, petg, tmp_path):
    j = Job(photo=tmp_path / "x", setup=setup, optics=petg, mode="color")
    assert j.frame_thickness == forge.COLOR_RANGE[1]
    assert not j.standing
