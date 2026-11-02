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


@pytest.mark.parametrize("light, view", [("back", "flat"), ("front", "relief")])
@pytest.mark.parametrize("shape", ["flat", "shaped"])
def test_colour_make(sunset_photo, setup, petg, tmp_path, no_slicer, shape, light, view):
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
        light=light,
        view=view,
    )
    res = forge.make(j)
    plan = json.loads(res.files["plan"].read_text(encoding="utf-8"))
    assert plan["mode"] == "color" and plan["pose"] == "lying"
    assert plan["light"] == light and plan["view"] == view
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
    assert ("needs no light behind it" in report) == (light == "front")
    lo, hi = j.color_range
    assert f"thickness {lo:g} to {hi:g} mm" in report  # the stack, not the mono optics
    assert "Tungsten" not in json.dumps(plan["colour"]["print"])  # abrasive: left out
    first = plan["colour"]["print"]["filaments"][0]["hex"]
    assert forge.first_colour(j, res) == first
    assert plan["printer"]["filament_colour"] is None  # no_slicer: the 3MF carries no colour
    assert "left out as abrasive" in plan["colour"]["source"]
    assert plan["optics"] is None and plan["color_range_mm"] == [lo, hi]


def test_short_list_note(sunset_photo, setup, petg, tmp_path, no_slicer):
    """With fewer filaments in the list than colours, the note blames the list."""
    own = tmp_path / "two.toml"
    own.write_text(
        '[[filament]]\nname = "white"\nhex = "#ECE8DC"\ntd = 3.4\n'
        '[[filament]]\nname = "red"\nhex = "#BB2E32"\ntd = 2.6\n',
        encoding="utf-8",
    )
    j = Job(
        photo=sunset_photo,
        setup=setup,
        optics=petg,
        out=tmp_path / "out",
        mode="color",
        colors=4,
        filaments=own,
        width=40,
        pitch=0.5,
    )
    notes = " ".join(forge.make(j).notes)
    assert "only 2 filaments for 4 colours" in notes and "search found no place" not in notes


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


def test_preview_shows_the_shaped_frame():
    """Outside the inner outline the frame volume covers the picture with the whole stack."""

    class Fit:
        colours = np.array([[1.0, 1.0, 1.0], [0.1, 0.05, 0.0]])

    sim = np.full((4, 5, 3), 0.5)
    inner = np.zeros((4, 5), bool)
    inner[1:3, 1:4] = True
    seen = forge.with_frame(sim, inner, Fit)
    assert np.allclose(seen[inner], 0.5) and np.allclose(seen[~inner], Fit.colours[-1])
    assert forge.with_frame(sim, None, Fit) is sim


def test_front_light_checks(setup, petg, tmp_path):
    kw = {"photo": tmp_path / "x", "setup": setup, "optics": petg, "light": "front"}
    with pytest.raises(ValueError, match="--view relief"):
        Job(mode="color", view="flat", **kw).check()  # the flat face lies on the bed
    with pytest.raises(ValueError, match="colour prints"):
        Job(mode="mono", **kw).check()
    with pytest.raises(ValueError, match="unknown light"):
        Job(mode="color", **{**kw, "light": "side"}).check()
    Job(mode="color", **kw).check()


def test_the_view_follows_the_light(setup, petg, tmp_path):
    kw = {"photo": tmp_path / "x", "setup": setup, "optics": petg}
    assert Job(mode="color", light="front", **kw).view == "relief"
    assert Job(mode="color", **kw).view == "flat"  # Job's light is "back", like its mode
    assert Job(mode="color", light="back", view="relief", **kw).view == "relief"
    j = Job(mode="color", light="front", **kw)
    assert j.part_name("lithophane") == "picture" and j.part_name("stand") == "stand"


def base_notes_for(photo, setup, petg, tmp_path, **kw):
    j = Job(
        photo=photo,
        setup=setup,
        optics=petg,
        out=tmp_path / "out",
        mode="color",
        colors=3,
        width=40,
        pitch=0.5,
        **kw,
    )
    return " ".join(forge.make(j).notes)


def test_a_clear_base_is_a_note(sunset_photo, setup, petg, tmp_path, no_slicer):
    """In room light the first filament must hide what is behind the panel; with only
    clear filaments in the list, the note says the wall shows through."""
    own = tmp_path / "clear.toml"
    own.write_text(
        '[[filament]]\nname = "clear"\nhex = "#E4E7E3"\ntd = 100\n'
        '[[filament]]\nname = "smoke"\nhex = "#444444"\ntd = 20\n',
        encoding="utf-8",
    )
    notes = base_notes_for(sunset_photo, setup, petg, tmp_path, light="front", filaments=own)
    assert "show through" in notes
    notes = base_notes_for(sunset_photo, setup, petg, tmp_path, light="back", filaments=own)
    assert "tints every point" not in notes  # clear is light, neutral and translucent


def test_a_thin_panel_cannot_start_with_white(sunset_photo, setup, petg, tmp_path, no_slicer):
    """At 0.3 mm only a TD up to 3 mm hides the back: Signal White (3.4) cannot be the
    first filament, and the note says which --t-min allows it."""
    notes = base_notes_for(
        sunset_photo, setup, petg, tmp_path, light="front", color_range=(0.3, 2.0)
    )
    assert "--t-min 0.4 lets Prusament PETG Signal White" in notes
    notes = base_notes_for(sunset_photo, setup, petg, tmp_path, light="front")
    assert "first one" not in notes and "show through" not in notes


def test_front_light_reads_the_right_way(setup, petg, tmp_path):
    """Lit from the front you look at the top face from above, as PrusaSlicer shows it:
    the upper left of the picture is at -x, +y, and nothing is mirrored."""
    j = Job(
        photo=tmp_path / "x",
        setup=setup,
        optics=petg,
        mode="color",
        light="front",
        view="relief",
        width=80,
        height=60,
        pitch=0.5,
    )
    lay = forge.layout(j, 4 / 3)
    t = np.full((lay.rows, lay.cols), 3.0)
    r, c = lay.rows // 2, lay.cols // 2
    t[r - 20 : r - 5, c - 30 : c - 10] = 1.0  # upper left of the picture
    m = forge.volumes(j, lay, t)[0][1]
    v = m.vertices
    thin = (v[:, 2] > 0.1) & (v[:, 2] < 1.2)
    assert v[thin, 1].mean() > lay.height / 2 and v[thin, 0].mean() < 0


def test_colour_frame_is_the_whole_stack(setup, petg, tmp_path):
    j = Job(photo=tmp_path / "x", setup=setup, optics=petg, mode="color")
    assert j.frame_thickness == forge.COLOR_RANGE[1]
    assert not j.standing
