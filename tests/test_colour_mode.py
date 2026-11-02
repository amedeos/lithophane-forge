import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest
import trimesh
from PIL import Image
from test_palette import sunset

from lithophane_forge import forge, image, palette, threemf
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
    assert plan["min_patch_mm2"] == forge.MIN_PATCH and "--min-patch" in report


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
    assert Job(mode="color", light="front", **kw).face == "relief"
    assert Job(mode="color", **kw).face == "flat"  # Job's light is "back", like its mode
    assert Job(mode="color", light="back", view="relief", **kw).face == "relief"
    front = replace(Job(mode="color", **kw), light="front")  # no view given: it follows
    assert front.face == "relief"
    front.check()
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
    clear filaments in the list (allowed, although nothing measured them), the note says
    the wall shows through."""
    own = tmp_path / "clear.toml"
    own.write_text(
        '[[filament]]\nname = "clear"\nhex = "#E4E7E3"\ntd = 100\n'
        '[[filament]]\nname = "smoke"\nhex = "#444444"\ntd = 20\n',
        encoding="utf-8",
    )
    notes = base_notes_for(
        sunset_photo, setup, petg, tmp_path, light="front", filaments=own, allow_unmeasured=True
    )
    assert "show through" in notes and "even at the full thickness" in notes
    notes = base_notes_for(sunset_photo, setup, petg, tmp_path, light="back", filaments=own)
    assert "tints every point" not in notes  # clear is light, neutral and translucent


def test_the_default_base_hides_the_back(sunset_photo, setup, petg, tmp_path, no_slicer):
    """In room light only opaque filaments hide the back at 0.6 mm (Signal White needs
    about 1.8 mm): the plan starts with one, and no note asks for another first filament."""
    notes = base_notes_for(sunset_photo, setup, petg, tmp_path, light="front")
    assert "show through" not in notes and "--t-min" not in notes


def test_a_translucent_base_names_the_t_min(sunset_photo, setup, petg, tmp_path, no_slicer):
    """With only white in the list, the note says from which thickness it hides the back."""
    own = tmp_path / "white.toml"
    own.write_text('[[filament]]\nname = "white"\nhex = "#E5E0E1"\ntd = 3.4\n', encoding="utf-8")
    notes = base_notes_for(sunset_photo, setup, petg, tmp_path, light="front", filaments=own)
    assert "show through" in notes and "raise --t-min to 1.8 mm" in notes


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


def small_parts(idx, cells):
    """Islands (points at least z high) and holes (points lower than z) smaller than
    `cells` points, over every layer z of a height map."""
    from scipy import ndimage

    count = 0
    for z in range(int(idx.min()) + 1, int(idx.max()) + 1):
        for part in (idx >= z, idx < z):
            labels, _ = ndimage.label(part)
            count += int((np.bincount(labels.ravel())[1:] < cells).sum())
    return count


def test_remove_specks():
    idx = np.zeros((40, 40), dtype=np.int64)
    idx[4:27, 4:27] = 2  # a thin ring at height 2 around a large patch at height 5
    idx[5:26, 5:26] = 5
    idx[15, 15] = 1  # a hole in the patch
    idx[34, 34] = 3  # an island outside it
    out = forge.remove_specks(idx, 9)
    assert out[15, 15] == 5 and out[34, 34] == 0
    keep = np.ones_like(idx, dtype=bool)
    keep[15, 15] = keep[34, 34] = False
    assert np.array_equal(out[keep], idx[keep])  # the ring and the patch stay
    assert small_parts(idx, 9) == 3 + 4  # the island in layers 1-3, the hole in 2-5
    assert small_parts(out, 9) == 0
    assert forge.remove_specks(idx, 1) is idx  # 0 or 1 point: off
    with pytest.raises(ValueError, match="no patch of 1601 points"):
        forge.remove_specks(idx, 1601)


def test_remove_specks_sees_the_frame():
    """A patch that joins the frame is no island; a hole that runs under the frame counts
    only its visible part."""
    idx = np.zeros((30, 30), dtype=np.int64)
    hidden = np.zeros_like(idx, dtype=bool)
    hidden[:3] = hidden[-3:] = hidden[:, :3] = hidden[:, -3:] = True  # a frame at height 5
    idx[3, 10:12] = 5  # a small high patch along the frame
    idx[15, 15] = 5  # and one in the middle
    out = forge.remove_specks(idx, 9, hidden, frame=5)
    assert (out[3, 10:12] == 5).all() and out[15, 15] == 0 and (out[hidden] == 5).all()
    assert (forge.remove_specks(idx[3:-3, 3:-3], 9)[0, 7:9] == 0).all()  # with no frame

    idx = np.full((30, 30), 5, dtype=np.int64)
    hidden = np.zeros_like(idx, dtype=bool)
    hidden[:, 20:] = True  # a shaped panel's frame covers the right side
    idx[10, 19:] = 0  # a low strip: one point shows, ten lie under the frame
    assert forge.remove_specks(idx, 9, hidden, frame=5)[10, 19] == 5
    assert forge.remove_specks(idx, 9)[10, 19] == 0  # counted with its hidden part


def test_remove_specks_in_one_pass():
    """The closing after the opening makes no new island: a second pass changes nothing."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 8, (60, 80))
    out = forge.remove_specks(idx, 9)
    assert small_parts(idx, 9) > 0 and small_parts(out, 9) == 0
    assert np.array_equal(forge.remove_specks(out, 9), out)


def noisy_job(tmp_path, setup, petg, **kw):
    rng = np.random.default_rng(1)
    rgb = np.clip(sunset(100, 80) + rng.normal(0, 0.08, (100, 80, 3)), 0, 1)
    path = tmp_path / "noisy.png"
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
    j = Job(photo=path, setup=setup, optics=petg, mode="color", colors=3, width=30, **kw)
    photo = image.load(j.photo)
    return j, photo, forge.layout(j, photo.shape[1] / photo.shape[0])


@pytest.mark.parametrize("light", ["front", "back"])
def test_colour_map_has_no_specks(setup, petg, tmp_path, light, monkeypatch):
    """A noisy photo gives specks; the heights that print keep none smaller than
    min_patch, and the Delta E in the plan is the Delta E of those heights."""
    j, photo, lay = noisy_job(tmp_path, setup, petg, light=light)
    seen, remove = [], forge.remove_specks

    def spy(idx, cells, hidden=None, frame=0):
        out = remove(idx, cells, hidden, frame)
        seen.append((np.where(hidden, frame, idx), out, cells))
        return out

    monkeypatch.setattr(forge, "remove_specks", spy)
    cmap = forge.colour_map(j, photo, lay)
    cells = round(forge.MIN_PATCH / lay.pitch**2)
    assert cmap.speck_points == cells > 1 and len(seen) == 2  # the print and the ideal
    for before, after, n in seen:
        assert n == cells and small_parts(before, cells) > 0 and small_parts(after, cells) == 0
    in_v, in_u, _ = forge.picture_cells(j, lay)
    idx = seen[0][1][np.ix_(in_v, in_u)]
    h = palette.heights(*j.color_range, setup.first_layer, setup.layer)
    assert np.allclose(cmap.t_view[np.ix_(in_v, in_u)], h[idx])
    target = palette.target(image.resize(cmap.photo, *idx.shape), j.tone.levels, j.tone.gamma)
    assert cmap.plan.real.mean_de == palette.score(cmap.plan.real, target, idx).mean_de


def test_a_last_filament_lost_to_smoothing_is_a_note(setup, petg, tmp_path, monkeypatch):
    j, photo, lay = noisy_job(tmp_path, setup, petg, light="front")
    monkeypatch.setattr(forge, "remove_specks", lambda idx, *a: np.zeros_like(idx))
    cmap = forge.colour_map(j, photo, lay)
    last = cmap.plan.real.filaments[-1].name
    assert any(
        f"reaches the last filament, {last}: it shows only on the frame" in n for n in cmap.notes
    )


def test_min_patch_larger_than_the_picture(setup, petg, tmp_path):
    j, photo, lay = noisy_job(tmp_path, setup, petg, min_patch=2000)
    with pytest.raises(ValueError, match="as large as the whole picture"):
        forge.colour_map(j, photo, lay)


def test_the_report_says_when_the_filter_is_off(sunset_photo, setup, petg, tmp_path, no_slicer):
    j = Job(
        photo=sunset_photo, setup=setup, optics=petg, out=tmp_path / "out", mode="color",
        colors=2, width=30, pitch=0.5, min_patch=0.2,
    )  # fmt: skip
    res = forge.make(j)
    assert res.speck_points == 1  # 0.2 mm² is less than one point 0.5 mm apart
    assert "not larger than one grid point" in res.files["report"].read_text(encoding="utf-8")


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_min_patch_is_checked(setup, petg, tmp_path, value):
    j = Job(photo=tmp_path / "x", setup=setup, optics=petg, mode="color", min_patch=value)
    with pytest.raises(ValueError, match="--min-patch"):
        j.check()
