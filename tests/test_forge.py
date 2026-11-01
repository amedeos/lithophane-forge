import json
from dataclasses import replace

import numpy as np
import pytest
import trimesh

from lithophane_forge import forge
from lithophane_forge.forge import Job


def job(photo, setup, petg, tmp_path, **kw):
    return Job(photo=photo, setup=setup, optics=petg, out=tmp_path / "out", **kw)


@pytest.mark.parametrize("shape", forge.SHAPES)
def test_make_without_slicer(photo, setup, petg, tmp_path, no_slicer, shape):
    j = job(photo, setup, petg, tmp_path, shape=shape, width=60, pitch=0.4)
    res = forge.make(j)
    for key in ("preview", "stl", "3mf", "plan", "report"):
        assert res.files[key].is_file(), key
    assert "PrusaSlicer not found" in " ".join(res.notes)
    tm = trimesh.load(res.files["stl"])
    assert tm.is_watertight
    lo, hi = tm.bounds
    assert lo[2] == pytest.approx(0, abs=1e-6)
    assert (lo[:2] + hi[:2]) / 2 == pytest.approx([setup.bed[0] / 2, setup.bed[1] / 2])
    plan = json.loads(res.files["plan"].read_text(encoding="utf-8"))
    assert plan["shape"] == shape and plan["printer"]["filament"] == "Prusament PETG @PGIS"
    width, height = plan["outer_mm"]
    assert width == pytest.approx(60)
    assert hi[2] - lo[2] == pytest.approx(height)
    # the photo is 4:3 inside a 3 mm frame
    assert (width - 6) / (height - 6) == pytest.approx(4 / 3, rel=0.01)


def test_both_sizes_crop_the_photo(photo, setup, petg, tmp_path, no_slicer):
    res = forge.make(job(photo, setup, petg, tmp_path, width=50, height=50, pitch=0.5))
    plan = json.loads(res.files["plan"].read_text(encoding="utf-8"))
    assert plan["outer_mm"] == [50, 50]


def test_grid_gets_coarser_for_big_panels():
    rows, cols, pitch = forge.grid(200, 150, 0.2, 100_000)
    assert pitch > 0.2 and 2 * rows * cols <= 100_000 * 1.05


def test_too_tall_for_the_printer(photo, setup, petg, tmp_path, no_slicer):
    with pytest.raises(ValueError, match="prints at most"):
        forge.make(job(photo, setup, petg, tmp_path, height=230, pitch=1.0))


def test_too_wide_for_the_bed(setup):
    from lithophane_forge import mesh

    panel = mesh.standing(np.full((5, 5), 2.0), 235, 50)
    with pytest.raises(ValueError, match="does not fit"):
        forge.place(panel, setup.bed)
    assert forge.place(mesh.standing(np.full((5, 5), 2.0), 225, 50), setup.bed).size[0] == 225


def thin_spot(m, shape, width):
    """Centre (x, z) of the thin part of a panel (from `forge.panel`), and the box centre."""
    v = m.vertices
    if shape == "flat":
        depth = v[:, 1]  # the flat face is at y = 0
    else:  # bent around the z axis; the flat face is on the radius
        radius = width / np.radians(120)
        depth = radius - np.hypot(v[:, 0], v[:, 1])
    thin = (depth > 0.1) & (depth < 1.2)
    return v[thin][:, [0, 2]].mean(0), (v.min(0) + v.max(0))[[0, 2]] / 2


@pytest.mark.parametrize("shape", forge.SHAPES)
def test_picture_reads_the_right_way(setup, petg, tmp_path, shape):
    """A bright (thin) square at the upper left of the picture must be at the upper left as
    seen by the viewer: from -y for view="flat" (+x is right), from +y for "relief"."""
    t = np.full((60, 80), 3.0)
    t[5:20, 5:25] = 1.0  # row 0 is the top of the picture
    j = Job(photo=tmp_path / "x.png", setup=setup, optics=petg, shape=shape, arc=120)
    (x, z), (cx, cz) = thin_spot(forge.panel(j, t, 80, 60), shape, 80)
    assert z > cz and x < cx
    (x, z), (cx, cz) = thin_spot(forge.panel(replace(j, view="relief"), t, 80, 60), shape, 80)
    assert z > cz and x > cx


def test_overhang_limit():
    t = np.full((50, 10), 1.0)
    t[10] = 4.0  # a thick line above thin plastic: a ledge
    out = forge.limit_overhangs(t, 45, 0.2)
    assert np.all(out >= t)
    rise = out[:-1] - out[1:]  # row 0 is the top: how much thicker the row above is
    assert rise.max() <= 0.2 + 1e-9
    assert out[10:].max() == pytest.approx(4.0) and out[:10].max() == 1.0  # nothing above
    assert np.array_equal(forge.limit_overhangs(t, 90, 0.2), t)


def test_object_settings_fill_the_thickness(setup, petg, tmp_path):
    s = forge.object_settings(Job(photo=tmp_path / "x", setup=setup, optics=petg))
    assert s["fill_density"] == "100%"
    assert int(s["perimeters"]) * 2 * 0.45 >= petg.t_max + 0.8
