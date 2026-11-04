import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest
import trimesh
from PIL import Image
from test_palette import disc_photo, sunset

from lithophane_forge import forge, image, palette, render, threemf
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


def small_parts(idx, cells, outside=None):
    """Islands (points at least z high) and holes (points lower than z) smaller than
    `cells` points, over every layer z of a height map; with `outside` (a bool per point),
    only those that touch no point of it."""
    from scipy import ndimage

    count = 0
    for z in range(int(idx.min()) + 1, int(idx.max()) + 1):
        for part in (idx >= z, idx < z):
            labels, n = ndimage.label(part)
            small = np.bincount(labels.ravel(), minlength=n + 1) < cells
            if outside is not None:
                small &= np.bincount(labels[outside], minlength=n + 1) == 0
            count += int(small[1:].sum())
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


def test_smooth_heights_keeps_fine_detail_on_the_subject():
    """With --focus the subject keeps the small spots and the corners that the 3 x 3 median
    removes, but no part narrower than the width: a nozzle line cannot draw it."""
    from scipy.ndimage import median_filter

    raw = np.zeros((30, 40), dtype=np.int64)
    subject = np.zeros_like(raw, dtype=bool)
    subject[:, :20] = True
    for c in (3, 23):  # the same parts on the subject and outside it
        raw[3:5, c : c + 2] = 4  # a spot of 2 x 2 points
        raw[10:22, c] = 3  # a line one point wide
        raw[10:22, c + 5 : c + 7] = 2  # a line two points wide
    assert np.array_equal(forge.smooth_heights(raw), median_filter(raw, size=3))
    out = forge.smooth_heights(raw, subject, 2)
    assert (out[3:5, 3:5] == 4).all() and (out[3:5, 23:25] == 0).all()  # the spot
    assert (out[10:22, 3] == 0).all() and (out[10:22, 23] == 0).all()  # one point wide: gone
    assert (out[10:22, 8:10] == 2).all()  # two points wide, with its corners
    assert (out[10, 28:30] == 0).all() and (out[11:21, 28:30] == 2).all()  # median: no corners
    assert np.array_equal(out[:, 20:], median_filter(raw, size=3)[:, 20:])
    fine = forge.smooth_heights(raw, subject, 1)  # a coarse grid: one point is wide enough
    assert np.array_equal(fine[subject], raw[subject])


def test_remove_specks_keeps_smaller_patches_on_the_subject():
    idx = np.zeros((30, 40), dtype=np.int64)
    subject = np.zeros_like(idx, dtype=bool)
    subject[:, :20] = True
    idx[5:7, 5:8] = 3  # 6 points on the subject
    idx[5:7, 25:28] = 3  # 6 points outside it
    idx[20, 10:12] = 2  # 2 points on the subject
    idx[15:17, 18:21] = 4  # 6 points across the edge: 4 on the subject, 2 outside it
    out = forge.remove_specks(idx, 9, subject=subject, subject_cells=4)
    assert (out[5:7, 5:8] == 3).all() and (out[5:7, 25:28] == 0).all()
    assert (out[20, 10:12] == 0).all()
    assert (out[15:17, 18:21] == 4).all()  # it touches the subject: it stays whole
    assert small_parts(out, 4) == 0 and small_parts(out, 9, subject) == 0
    assert small_parts(idx, 9, subject) > 0
    plain = forge.remove_specks(idx, 9)
    # the subject's limit is never larger than the rest's, and no subject is the old filter
    assert np.array_equal(forge.remove_specks(idx, 9, subject=subject, subject_cells=20), plain)
    nobody = np.zeros_like(subject)
    assert np.array_equal(forge.remove_specks(idx, 9, subject=nobody, subject_cells=4), plain)


def test_remove_specks_at_the_seam():
    """A part that the large limit removes outside the subject must not leave a hole there
    that is smaller than the large limit (the code review's case)."""
    idx = np.full((30, 40), 3, dtype=np.int64)
    subject = np.zeros_like(idx, dtype=bool)
    subject[:, :20] = True
    idx[:, :20] = 0  # the subject is low
    idx[10:15, 19] = 3  # an island of 5 points on the subject, at its edge
    idx[10:15, 20:24] = 0  # a low patch of 20 points just outside, next to the island
    out = forge.remove_specks(idx, 25, subject=subject, subject_cells=5)
    assert (out[10:15, 19] == 3).all() and (out[10:15, 20:24] == 3).all()
    assert small_parts(out, 5) == 0 and small_parts(out, 25, subject) == 0


@pytest.mark.parametrize("seed", range(8))
def test_remove_specks_on_a_random_subject(seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 8, (60, 80))
    subject = np.zeros_like(idx, dtype=bool)
    subject[10:40, 15:50] = True
    out = forge.remove_specks(idx, 16, subject=subject, subject_cells=5)
    assert small_parts(out, 5) == 0 and small_parts(out, 16, subject) == 0
    assert np.array_equal(forge.remove_specks(out, 16, subject=subject, subject_cells=5), out)
    # with no point of the subject marked it is scikit-image's filter
    nobody = np.zeros_like(subject)
    plain = forge.remove_specks(idx, 16)
    assert np.array_equal(forge.remove_specks(idx, 16, subject=nobody, subject_cells=5), plain)


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


def heights_of(cmap, j, lay):
    """The height index of every grid point, from t_view (the frame included)."""
    h = palette.heights(*j.color_range, j.setup.first_layer, j.setup.layer)
    return np.abs(cmap.t_view[..., None] - h).argmin(-1)


def patches(idx, where):
    """Connected areas of one height that touch `where`."""
    from scipy import ndimage

    count = 0
    for z in np.unique(idx[where]):
        labels, _ = ndimage.label(idx == z)
        count += len(np.unique(labels[where & (idx == z)]))
    return count


def test_focus_regions(setup, petg, tmp_path):
    """With --focus the background is smoothed before the heights are chosen (fewer patches)
    and the subject keeps smaller patches; the filaments and change heights do not change,
    and the Delta E is still measured against the photo."""
    j, photo, lay = noisy_job(tmp_path, setup, petg, light="front", focus="20,25,40,50")
    on = forge.colour_map(j, photo, lay)
    off = forge.colour_map(replace(j, smooth_background=0), photo, lay)
    assert on.subject_points == round(forge.MIN_PATCH_SUBJECT / lay.pitch**2) == 16
    assert on.subject_width == 2 and on.speck_points == 25
    assert [f.name for f in on.plan.real.filaments] == [f.name for f in off.plan.real.filaments]
    assert np.array_equal(on.plan.real.starts, off.plan.real.starts)
    full_on, full_off = heights_of(on, j, lay), heights_of(off, j, lay)
    in_v, in_u, _ = forge.picture_cells(j, lay)
    box = np.ix_(in_v, in_u)
    subject = np.zeros(full_on.shape, dtype=bool)
    subject[box] = on.subject
    rest = forge.visible_points(j, lay) & ~subject
    assert patches(full_on, rest) < 0.7 * patches(full_off, rest)
    changed = full_on != full_off  # blended by 1 - the map: the box keeps the photo
    assert changed[subject].mean() < 0.2 * changed[rest].mean()
    assert small_parts(full_on, 16) == 0 and small_parts(full_on, 25, subject) == 0
    idx = full_on[box]
    target = palette.target(image.resize(on.photo, *idx.shape), j.tone.levels, j.tone.gamma)
    assert on.plan.real.mean_de == palette.score(on.plan.real, target, idx).mean_de


def test_smooth_background():
    """The bilateral filter keeps a strong edge, evens out small texture, and leaves the
    subject (weight 1) as it is."""
    rng = np.random.default_rng(0)
    lab = np.zeros((20, 20, 3))
    lab[..., 0] = 50 + rng.normal(0, 1, (20, 20))  # texture of about 1 Delta E
    lab[:, 10:, 0] += 4 * palette.SMOOTH_DE  # an edge four times the colour sigma
    rest, subject = np.zeros((20, 20)), np.ones((20, 20))
    out = palette.smooth_background(lab, rest, 2)
    assert out[:, :8, 0].mean() < 51 and out[:, 12:, 0].mean() > 49 + 4 * palette.SMOOTH_DE
    assert out[:, :8, 0].std() < 0.5 * lab[:, :8, 0].std()
    assert np.array_equal(palette.smooth_background(lab, subject, 2), lab)
    assert palette.smooth_background(lab, rest, 0) is lab


def test_a_last_filament_lost_with_focus_names_the_smoothing(setup, petg, tmp_path, monkeypatch):
    j, photo, lay = noisy_job(tmp_path, setup, petg, light="front", focus="centre")
    monkeypatch.setattr(forge, "remove_specks", lambda idx, *a, **k: np.zeros_like(idx))
    notes = " ".join(forge.colour_map(j, photo, lay).notes)
    assert "--min-patch 0 keeps the specks; --smooth-background 0 does not smooth" in notes


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


@pytest.mark.parametrize("name", ["min_patch_subject", "smooth_background"])
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_focus_regions_are_checked(setup, petg, tmp_path, name, value):
    kw = {"mode": "color", "focus": "centre", name: value}
    with pytest.raises(ValueError, match="--" + name.replace("_", "-") + " must be"):
        Job(photo=tmp_path / "x", setup=setup, optics=petg, **kw).check()


def test_the_subject_patch(setup, petg, tmp_path):
    """--min-patch-subject: at most --min-patch when given; when not, the default or a
    smaller --min-patch. The bilateral filter's sigma has a ceiling (its time grows with
    the square)."""
    kw = {"photo": tmp_path / "x", "setup": setup, "optics": petg, "mode": "color"}
    kw |= {"focus": "centre"}
    assert Job(**kw).subject_patch == forge.MIN_PATCH_SUBJECT
    assert Job(**kw, min_patch=0.5).subject_patch == 0.5
    assert Job(**kw, min_patch_subject=1).subject_patch == 1
    with pytest.raises(ValueError, match="larger than --min-patch 1 mm²: .* raise --min-patch"):
        Job(**kw, min_patch_subject=2).check()
    Job(**kw, min_patch_subject=2, min_patch=2).check()
    with pytest.raises(ValueError, match="--smooth-background must be between 0"):
        Job(**kw, smooth_background=forge.MAX_SMOOTH + 0.1).check()


@pytest.mark.parametrize(
    "kw, keeps",
    [
        ({}, "islands and holes from 0.64 mm² (--min-patch-subject)"),
        ({"min_patch": 0.5}, "islands and holes from 0.5 mm² (--min-patch)"),
        ({"min_patch_subject": 0.5}, "islands and holes from 0.5 mm² (--min-patch-subject)"),
        ({"min_patch": 0.03}, "--min-patch 0.03 mm² is not larger than one grid point"),
        ({"min_patch_subject": 0}, "all its islands and holes (--min-patch-subject 0)"),
    ],
)
def test_the_report_names_the_subject_limit(setup, petg, tmp_path, kw, keeps):
    j = replace(noisy_job(tmp_path, setup, petg, focus="centre")[0], **kw)
    res = forge.Result(layout=forge.layout(j, 0.8), speck_points=25)
    res.subject_width = 2
    res.subject_points = round(j.subject_patch / res.layout.pitch**2)
    lines = forge.speck_lines(j, res)
    assert lines[0].startswith("- In the rest of the picture, ") and keeps in lines[1]
    off = replace(j, min_patch=0)
    res.speck_points = res.subject_points = 0
    assert forge.speck_lines(off, res)[0].startswith("- In the rest of the picture, small")


def focus_job(tmp_path, setup, petg, **kw):
    rgb, _ = disc_photo()
    path = tmp_path / "disc.png"
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
    kw = {"colors": 3, "width": 40, "pitch": 0.5, "light": "front", "out": tmp_path / "out"} | kw
    return Job(photo=path, setup=setup, optics=petg, mode="color", **kw)


def test_focus_make(setup, petg, tmp_path, no_slicer):
    """--focus writes focus.png, the subject's Delta E and the outline; focus.png passed
    back gives the same plan, and a run without --focus removes it."""
    j = focus_job(tmp_path, setup, petg, focus="70,15,30,30")  # a box around the disc
    res = forge.make(j)
    path = j.folder / "focus.png"
    assert res.files["focus"] == path and path.is_file()
    with Image.open(path) as im:
        assert im.size == (120, 90) and im.mode == "L"  # the photo's size: it is small
    plan = json.loads(res.files["plan"].read_text(encoding="utf-8"))
    assert plan["focus"]["spec"] == "70,15,30,30" and plan["focus"]["floor"] == 0.1
    assert 0.05 < plan["focus"]["subject_share"] < 0.12
    # a grid 0.5 mm apart: one point is wide enough, 0.64 mm² is 3 points (2.56)
    assert plan["focus"]["subject_min_patch_mm2"] == forge.MIN_PATCH_SUBJECT
    assert plan["focus"]["subject_min_patch_points"] == 3 and plan["min_patch_points"] == 4
    assert plan["focus"]["subject_min_width_mm"] == 0.5
    assert plan["focus"]["smooth_background_mm"] == forge.SMOOTH_BACKGROUND
    assert plan["colour"]["print"]["subject_delta_e"] is not None
    report = res.files["report"].read_text(encoding="utf-8")
    assert "On the subject (`--focus 70,15,30,30`" in report
    flat = " ".join(report.split())
    assert "In the rest of the picture, islands and holes smaller than 1 mm²" in flat
    assert "the heights keep details from 0.5 mm wide and islands and holes from 0.64 mm²" in flat
    assert "smoothed (0.4 mm, --smooth-background; 0 turns it off)" in flat
    assert (
        f"copy it next to the photo, edit the copy and pass it with `--focus`: every run rewrites `{j.folder}`"
        in " ".join(report.split())
    )
    with Image.open(res.files["preview"]) as im:
        assert (np.asarray(im.convert("RGB")) == render.FOCUS).all(axis=2).any()

    again = forge.make(replace(j, focus=str(path)))
    assert path.is_file()  # read first, then written again: the same map
    plan2 = json.loads(again.files["plan"].read_text(encoding="utf-8"))
    assert plan2["colour"] == plan["colour"] and plan2["colour_changes"] == plan["colour_changes"]

    plain = forge.make(replace(j, focus=None))
    assert not path.exists() and "focus" not in plain.files
    plan3 = json.loads(plain.files["plan"].read_text(encoding="utf-8"))
    assert plan3["focus"] is None and plan3["colour"]["print"]["subject_delta_e"] is None


def test_focus_on_a_black_mask_is_an_error(setup, petg, tmp_path, no_slicer):
    Image.new("L", (40, 30), 0).save(tmp_path / "black.png")
    j = focus_job(tmp_path, setup, petg, focus=str(tmp_path / "black.png"))
    with pytest.raises(ValueError, match="marks no part of the picture"):
        forge.make(j)


def test_focus_outside_the_picture_is_an_error(setup, petg, tmp_path, no_slicer):
    """A box in the photo that the crop leaves out marks nothing in the picture."""
    j = focus_job(tmp_path, setup, petg, focus="0,0,20,20", crop=(40, 0, 80, 90))
    with pytest.raises(ValueError, match="marks no part of the picture"):
        forge.make(j)


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"mode": "mono"}, "--focus is for colour prints"),
        ({"focus": "1,2,3"}, "X,Y,W,H"),
        ({"focus_floor": 0}, "--focus-floor must be between"),
        ({"focus_floor": 1.5}, "--focus-floor must be between"),
        ({"device": "tpu"}, "--device must be one of auto, cpu, gpu"),
        ({"device": "cpu"}, "--device goes with --focus subject"),
        ({"focus": None, "device": "gpu"}, "--device goes with --focus subject"),
    ],
)
def test_focus_is_checked(setup, petg, tmp_path, kw, message):
    kw = {"mode": "color", "focus": "centre"} | kw
    with pytest.raises(ValueError, match=message):
        Job(photo=tmp_path / "x", setup=setup, optics=petg, **kw).check()


def test_focus_subject_make(setup, petg, tmp_path, no_slicer, fake_subject):
    """--focus subject: the models' map goes to focus.png, plan.json and report.md say
    what found it; focus.png passed back gives the same plan without the models."""
    j = focus_job(tmp_path, setup, petg, focus="subject", device="cpu")
    res = forge.make(j)
    assert fake_subject == ["cpu"] and res.found.device == "CPU"
    plan = json.loads(res.files["plan"].read_text(encoding="utf-8"))
    models = ["u2net_human_seg", "modnet", "isnet"]
    assert plan["focus"]["found"] == {"models": models, "device": "CPU"}
    assert 0.2 < plan["focus"]["subject_share"] < 0.5
    report = " ".join(res.files["report"].read_text(encoding="utf-8").split())
    assert "found by 3 models (u2net_human_seg, modnet, isnet) on the CPU in 0.2 s" in report
    assert "On the subject (`--focus subject`" in report

    again = forge.make(replace(j, focus=str(res.files["focus"]), device="auto"))
    assert fake_subject == ["cpu"]  # the map was read, the models did not run
    plan2 = json.loads(again.files["plan"].read_text(encoding="utf-8"))
    assert plan2["colour"] == plan["colour"] and plan2["focus"]["found"] is None


def test_focus_subject_found_nothing(setup, petg, tmp_path, monkeypatch, fake_subject):
    from lithophane_forge import vision

    def nothing(photo, size, device="auto"):
        return np.zeros(size[::-1]), vision.Run(("u2net_human_seg",), "CPU", 0.1)

    monkeypatch.setattr(vision, "subject_map", nothing)
    with pytest.raises(ValueError, match="--focus subject found no subject"):
        forge.make(focus_job(tmp_path, setup, petg, focus="subject"))


def test_focus_subject_needs_the_extra(setup, petg, tmp_path, monkeypatch):
    """Job.check refuses --focus subject without ONNX Runtime, before any download."""
    from lithophane_forge import vision

    monkeypatch.setattr(vision, "available", lambda: False)
    with pytest.raises(ValueError, match="needs ONNX Runtime: uv sync --extra vision"):
        focus_job(tmp_path, setup, petg, focus="subject").check()


def test_clear_outputs_removes_the_focus_map(tmp_path):
    for name in ("p.stl", "focus.png", "plan.json"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    forge.clear_outputs(tmp_path, "p", keep=tmp_path / "plan.json")
    assert sorted(f.name for f in tmp_path.iterdir()) == ["plan.json"]


def test_a_shaped_frame_hides_no_subject(setup, petg, tmp_path):
    """Where the frame of a shaped panel hides the picture there is no subject: no weight
    for the fit, no outline in the preview."""
    j = focus_job(tmp_path, setup, petg, shape="shaped", mask="heart", focus="0,0,120,90")
    photo = image.load(j.photo)
    lay = forge.layout(j, photo.shape[1] / photo.shape[0])
    cmap = forge.colour_map(j, photo, lay)
    in_v, in_u, _ = forge.picture_cells(j, lay)
    seen = forge.visible_points(j, lay)[np.ix_(in_v, in_u)]
    assert cmap.subject.any() and not cmap.subject[~seen].any()
    assert cmap.subject_share == 1.0  # the box covers the whole photo: all you see


@pytest.mark.parametrize(
    "worse, colors, note, more",
    [(0.5, 2, False, False), (2.0, 2, True, False), (2.0, 1, True, True), (2.0, 8, True, False)],
)
def test_a_subject_far_from_the_photo_is_a_note(setup, petg, tmp_path, worse, colors, note, more):
    """The note asks for one more filament only when the plan used all it could have and
    --colors can grow."""
    from types import SimpleNamespace

    h = palette.heights(*forge.COLOR_RANGE, setup.first_layer, setup.layer)
    black = next(f for f in palette.load_filaments() if f.name.endswith("Jet Black"))
    fit = palette.Fit([black] * min(colors, 1), np.array([0.0]), h, np.zeros((len(h), 3)))
    if colors == 8:
        fit = replace(fit, filaments=[black] * 8, starts=np.arange(8.0))
    fit = replace(fit, mean_de=8.0, focus_de=8.0 + worse)
    plan = SimpleNamespace(real=fit, available=10)
    j = Job(photo=tmp_path / "x", setup=setup, optics=petg, mode="color", colors=colors)
    notes = [n for n in forge.colour_notes(j, plan) if "even with --focus" in n]
    assert bool(notes) == note
    assert any(f"--colors {colors + 1}" in n for n in notes) == more
    assert all("--focus-floor" in n for n in notes)
