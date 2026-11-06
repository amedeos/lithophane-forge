import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh

from lithophane_forge import forge, image, shapes
from lithophane_forge.forge import Job

SIZES = {
    "flat": {"width": 60},
    "curved": {"width": 60},
    "cylinder": {"diameter": 40, "height": 40},
    "shaped": {"width": 50},
}


def job(photo, setup, petg, tmp_path, **kw):
    return Job(photo=photo, setup=setup, optics=petg, out=tmp_path / "out", **kw)


def watertight(m):
    return trimesh.Trimesh(m.vertices, m.faces, process=False).is_watertight


@pytest.mark.parametrize("shape", forge.SHAPES)
def test_make_without_slicer(photo, setup, petg, tmp_path, no_slicer, shape):
    j = job(photo, setup, petg, tmp_path, shape=shape, pitch=0.4, **SIZES[shape])
    res = forge.make(j)
    for key in ("preview", "stl", "3mf", "plan", "report"):
        assert res.files[key].is_file(), key
    assert "PrusaSlicer not found" in " ".join(res.notes)
    tm = trimesh.load(res.files["stl"])
    lo, hi = tm.bounds
    assert lo[2] == pytest.approx(0, abs=1e-6)
    assert (lo[:2] + hi[:2]) / 2 == pytest.approx([setup.bed[0] / 2, setup.bed[1] / 2])
    plan = json.loads(res.files["plan"].read_text(encoding="utf-8"))
    assert plan["shape"] == shape and plan["printer"]["filament"] == "Prusament PETG @PGIS"
    assert plan["printer"]["filament_colour"] is None  # no PrusaSlicer: the 3MF has no colour
    assert plan["pose"] == ("standing" if shape in forge.STANDING else "lying")


@pytest.mark.parametrize("pose", forge.POSES)
def test_flat_size_follows_the_photo(photo, setup, petg, tmp_path, no_slicer, pose):
    res = forge.make(job(photo, setup, petg, tmp_path, width=60, pitch=0.4, pose=pose))
    width, height = res.layout.width, res.layout.height
    assert width == pytest.approx(60)
    up = 2 if pose == "standing" else 1  # standing, the height goes up along z
    assert res.size[up] == pytest.approx(height)
    assert res.size[3 - up] == pytest.approx(4.4)  # the frame, t_max + 0.8
    assert (width - 6) / (height - 6) == pytest.approx(4 / 3, rel=0.01)  # 4:3 in a 3 mm frame


def test_mono_lying_is_on_layer_tops(photo, setup, petg, tmp_path, no_slicer):
    """Lying, a white panel's thickness goes in whole layers, and no layer keeps an island
    or a hole smaller than --min-patch; standing it stays smooth."""
    from scipy import ndimage

    from lithophane_forge.palette import heights

    j = job(photo, setup, petg, tmp_path, width=60, pitch=0.4, min_patch=1.0)
    rgb = image.load(photo)
    lay = forge.layout(j, rgb.shape[1] / rgb.shape[0])
    raw, _ = forge.thickness_map(j, rgb, lay)
    t, found, points = forge.on_layers(j, lay, raw)
    tops = heights(petg.t_min, petg.t_max, setup.first_layer, setup.layer)
    assert np.array_equal(found, tops) and len(tops) == 29 and points == round(1.0 / 0.4**2)
    seen = forge.visible_points(j, lay)
    assert np.isin(np.round(t[seen], 6), np.round(tops, 6)).all()
    assert np.array_equal(t[~seen], raw[~seen])  # the frame keeps its thickness
    assert np.abs(petg.light(t[seen]) - petg.light(raw[seen])).mean() < 0.06
    # without the filter every point is the layer top closest to it in light (no median)
    plain, _, _ = forge.on_layers(replace(j, min_patch=0), lay, raw)
    near = tops[np.abs(petg.light(raw[seen])[:, None] - petg.light(tops)).argmin(1)]
    assert np.allclose(plain[seen], near)
    for z in tops[1:]:
        for part in (t >= z - 1e-6, t < z - 1e-6):
            labels, _ = ndimage.label(part)
            sizes = np.bincount(labels.ravel())[1:]
            assert sizes.min(initial=points) >= points
    res = forge.make(j)
    plan = json.loads(res.files["plan"].read_text(encoding="utf-8"))
    assert plan["pose"] == "lying" and plan["tones"] == 29 and plan["min_patch_mm2"] == 1.0
    assert plan["tone_range_mm"] == [0.8, 3.6]
    assert plan["max_overhang_deg"] is None and "brim_width" not in plan["object_settings"]
    text = res.files["report"].read_text(encoding="utf-8")
    assert "29 tones from 0.8 to 3.6 mm. In smooth" in text
    assert "`lithophane-forge calibrate --slice`" in text
    # the mesh itself: every height of the panel is a layer top (or the bed)
    zs = np.unique(np.round(trimesh.load(res.files["stl"]).vertices[:, 2], 4))
    allowed = np.round(np.concatenate([[0.0], tops, [j.frame_thickness]]), 4)
    assert np.isin(zs, allowed).all() and len(zs) > 20
    standing = forge.make(replace(j, pose="standing", min_patch=None, name="standing"))
    plan = json.loads(standing.files["plan"].read_text(encoding="utf-8"))
    assert plan["tones"] is None and plan["min_patch_mm2"] is None
    assert "--pose standing --slice" in standing.files["report"].read_text(encoding="utf-8")


def test_mono_lying_defaults(photo, setup, petg, tmp_path, no_slicer):
    """A white panel lying keeps details down to one nozzle line (0.2 mm²), the colour
    default stays 1 mm², and a frame or range off the layer grid is rounded and said."""
    j = Job(photo=photo, setup=setup, optics=petg)
    assert (j.smallest_patch, replace(j, mode="color").smallest_patch) == (0.2, 1.0)
    assert j.frame_thickness == 4.4
    assert replace(j, optics=replace(petg, t_max=3.65)).frame_thickness == 4.5  # 4.45 lying
    assert replace(j, frame_t=4.45, pose="standing").frame_thickness == 4.45
    off = replace(petg, t_min=0.85, t_max=3.65)
    res = forge.make(job(photo, setup, off, tmp_path, width=40, pitch=0.5))
    text = res.files["report"].read_text(encoding="utf-8")
    assert "28 tones from 0.9 to 3.6 mm." in text and "thickness 0.9 to 3.6 mm" in text
    assert "the brightest parts print at 0.9 mm and the darkest at 3.6 mm" in text


@pytest.mark.parametrize("shape", ["curved", "cylinder"])
def test_standing_report_names_the_standing_strip(photo, setup, petg, tmp_path, no_slicer, shape):
    j = job(photo, setup, petg, tmp_path, shape=shape, pitch=0.5, **SIZES[shape])
    text = forge.make(j).files["report"].read_text(encoding="utf-8")
    assert "`lithophane-forge calibrate --pose standing --slice`), which prints standing" in text


def test_tall_lying_panel_names_the_standing_pose(photo, setup, petg, tmp_path, no_slicer):
    """A panel taller than the bed is deep may still fit standing."""
    with pytest.raises(ValueError, match="or print it standing"):
        forge.make(job(photo, setup, petg, tmp_path, height=215, pitch=1.0))
    with pytest.raises(ValueError, match="make it smaller$"):  # colour cannot stand
        forge.make(job(photo, setup, petg, tmp_path, height=215, pitch=1.0, mode="color"))


def test_both_sizes_crop_the_photo(photo, setup, petg, tmp_path, no_slicer):
    res = forge.make(job(photo, setup, petg, tmp_path, width=50, height=50, pitch=0.5))
    plan = json.loads(res.files["plan"].read_text(encoding="utf-8"))
    assert plan["outer_mm"] == [50, 50]


@pytest.mark.parametrize("shape", forge.SHAPES)
def test_every_volume_is_closed(photo, setup, petg, tmp_path, shape):
    j = job(
        photo,
        setup,
        petg,
        tmp_path,
        shape=shape,
        pitch=0.5,
        hole=shape in ("flat", "shaped"),
        **SIZES[shape],
    )
    rgb = image.load(photo)
    lay = forge.layout(j, rgb.shape[1] / rgb.shape[0])
    t, _ = forge.thickness_map(j, rgb, lay)
    parts = forge.volumes(j, lay, t)
    assert parts[0][0] == "lithophane"
    for name, m in parts:
        assert watertight(m), name
        assert m.volume() > 0, name


def test_cylinder_size_and_front(setup, petg, tmp_path):
    """The middle of the picture faces -y; the seam is at the back (+y)."""
    j = Job(photo=tmp_path / "x", setup=setup, optics=petg, shape="cylinder", diameter=60)
    lay = forge.layout(j, 4 / 3)
    assert lay.width == pytest.approx(math.pi * 60) and lay.height == pytest.approx(75)
    t = np.full((lay.rows, lay.cols), 3.0)
    t[:, lay.cols // 2 - 5 : lay.cols // 2 + 5] = 1.0  # a thin stripe in the middle
    m = forge.volumes(j, lay, t)[0][1]
    radial = np.hypot(m.vertices[:, 0], m.vertices[:, 1])
    assert radial.max() == pytest.approx(30)
    thin = (radial > 30 - 1.2) & (radial < 30 - 0.1)
    assert m.vertices[thin, 1].mean() < -25


def test_cylinder_note_suggests_repeat(setup, petg, tmp_path):
    j = Job(photo=tmp_path / "x", setup=setup, optics=petg, shape="cylinder", diameter=80)
    notes = forge.cylinder_notes(j, forge.layout(j, 4 / 3), 4 / 3)
    assert "--repeat 2" in notes[0]
    j2 = replace(j, repeat=2)
    assert forge.cylinder_notes(j2, forge.layout(j2, 4 / 3), 4 / 3) == []


def test_shaped_heart_parts(photo, setup, petg, tmp_path):
    j = job(photo, setup, petg, tmp_path, shape="shaped", width=60, hole=True, pitch=0.5)
    lay = forge.layout(j, 4 / 3)
    assert lay.width == pytest.approx(60)
    t, _ = forge.thickness_map(j, image.load(photo), lay)
    parts = dict(forge.volumes(j, lay, t))
    assert set(parts) == {"lithophane", "frame", "hanging tab"}
    # the panel lies on the bed: thickness along z, never above the frame
    assert parts["lithophane"].size[2] <= j.frame_thickness + 1e-6
    assert parts["frame"].size[2] == pytest.approx(j.frame_thickness)
    # the tab sits above the heart (y is up in the picture)
    assert parts["hanging tab"].bounds[1][1] > parts["frame"].bounds[1][1]


def test_circle_stays_round():
    unit = shapes.unit("circle")
    assert shapes.size("circle", unit, 80, 50, 4 / 3) == (50, 50)
    assert shapes.size("oval", unit, 80, None, 2.0) == (80, 40)


def test_svg_and_png_masks(tmp_path):
    svg = tmp_path / "star.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<path d="M50 0 L61 35 L98 35 L68 57 L79 91 L50 70 L21 91 L32 57 L2 35 '
        'L39 35 Z"/></svg>',
        encoding="utf-8",
    )
    star = shapes.fit(shapes.unit(str(svg)), 80, 80)
    assert star.bounds == pytest.approx((-40, 0, 40, 80))
    # svg y points down: the star's single top point must be at the top
    top = [p for p in star.exterior.coords if p[1] > 79.9]
    assert len(top) >= 1 and abs(top[0][0]) < 1
    from PIL import Image

    img = np.full((100, 200), 255, np.uint8)
    img[20:80, 50:150] = 0
    Image.fromarray(img).save(tmp_path / "box.png")
    rect = shapes.unit(str(tmp_path / "box.png"))
    assert shapes.aspect(rect) == pytest.approx(100 / 60, rel=0.05)
    with pytest.raises(ValueError):
        shapes.unit("nope")


def test_crop_gives_the_panel_its_aspect(photo, setup, petg, tmp_path, no_slicer):
    """A square crop of a 4:3 photo makes a square picture, cut only once."""
    res = forge.make(job(photo, setup, petg, tmp_path, width=60, crop=(0, 0, 100, 100)))
    plan = json.loads(res.files["plan"].read_text(encoding="utf-8"))
    assert plan["outer_mm"] == pytest.approx([60, 60])


def test_arc_is_limited(setup, petg, tmp_path):
    j = Job(photo=tmp_path / "x.png", setup=setup, optics=petg, shape="curved", arc=360)
    with pytest.raises(ValueError, match="cylinder"):
        j.check()


@pytest.mark.parametrize("material, colour", [("petg", "#E5E0E1"), ("pla", forge.PLAIN_WHITE)])
def test_first_colour_of_white_prints(photo, petg, tmp_path, material, colour):
    """White prints start with Signal White in PETG, plain white in PLA (no catalog)."""
    from lithophane_forge import printers

    j = job(photo, printers.setup("mk4", material), petg, tmp_path)
    assert forge.first_colour(j, forge.Result()) == colour


def test_old_outputs_go_away(photo, setup, petg, tmp_path, no_slicer):
    """G-code from an earlier --slice run must not stay next to a new project."""
    import shutil

    folder = tmp_path / "out" / "photo"
    folder.mkdir(parents=True)
    old = ["photo.bgcode", "photo.gcode", "photo-stand.3mf", "photo-lamp-base.stl"]
    for name in old:
        (folder / name).write_text("old")
    (folder / "notes.txt").write_text("mine")
    inside = shutil.copy(photo, folder / "photo.png")  # the photo itself, kept in the folder
    forge.make(job(Path(inside), setup, petg, tmp_path, width=40, pitch=0.5))
    assert not any((folder / name).exists() for name in old)
    assert (folder / "notes.txt").exists() and Path(inside).exists()
    assert (folder / "photo.3mf").exists()


def test_small_shaped_panel_is_a_clear_error(setup, petg, tmp_path):
    j = Job(photo=tmp_path / "x", setup=setup, optics=petg, shape="shaped", width=10)
    with pytest.raises(ValueError, match="too small for a 3 mm frame"):
        forge.layout(j, 4 / 3)


@pytest.mark.parametrize("mode", ["mono", "color"])
def test_shaped_frame_hides_the_steps(setup, petg, tmp_path, mode):
    """A shaped panel's frame must reach the top of the picture, or its steps show."""
    j = Job(photo=tmp_path / "x", setup=setup, optics=petg, shape="shaped", mode=mode)
    j.check()
    with pytest.raises(ValueError, match="--frame-t"):
        replace(j, frame_t=1.5).check()


def test_volume_counts_every_part(photo, setup, petg, tmp_path, no_slicer):
    """Frame ring and hanging tab are plastic too."""
    j = job(photo, setup, petg, tmp_path, shape="shaped", width=60, hole=True, pitch=0.5)
    res = forge.make(j)
    lay = res.layout
    t, _ = forge.thickness_map(j, image.load(photo), lay)
    t, _, _ = forge.on_layers(j, lay, t)  # mono, lying: on layer tops
    parts = forge.volumes(j, lay, t)
    assert res.volume_cm3 == pytest.approx(sum(m.volume() for _, m in parts) / 1000)
    assert res.volume_cm3 > parts[0][1].volume() / 1000 * 1.1


def test_lying_report_follows_the_view(photo, setup, petg, tmp_path, no_slicer):
    res = forge.make(
        job(photo, setup, petg, tmp_path, shape="shaped", width=50, pitch=0.5, view="relief")
    )
    text = res.files["report"].read_text(encoding="utf-8")
    assert "relief side looks at you" in text
    assert "that face is the one you look at" not in text


def test_resolution_note_counts_one_copy(photo, setup, petg, tmp_path, no_slicer):
    """A cylinder with --repeat 2 needs only half the photo's columns around."""
    soft = "fewer pixels than the print grid"

    def notes(repeat):
        j = job(photo, setup, petg, tmp_path, shape="cylinder", diameter=40, pitch=0.5)
        return " ".join(forge.make(replace(j, repeat=repeat)).notes)

    assert soft in notes(1)  # about 250 columns around, the photo has 160
    assert soft not in notes(2)


def test_grid_gets_coarser_for_big_panels():
    rows, cols, pitch = forge.grid(200, 150, 0.2, 100_000)
    assert pitch > 0.2 and 2 * rows * cols <= 100_000 * 1.05


def test_too_tall_for_the_printer(photo, setup, petg, tmp_path, no_slicer):
    with pytest.raises(ValueError, match="prints at most"):
        forge.make(job(photo, setup, petg, tmp_path, height=230, pitch=1.0, pose="standing"))


def test_too_wide_for_the_bed(setup):
    from lithophane_forge import mesh

    panel = mesh.standing(np.full((5, 5), 2.0), 235, 50)
    with pytest.raises(ValueError, match="does not fit"):
        forge.place([("x", panel)], setup.bed, forge.BRIM)
    ok = mesh.standing(np.full((5, 5), 2.0), 225, 50)
    assert forge.place([("x", ok)], setup.bed, forge.BRIM)[0][1].size[0] == 225


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"shape": "curved", "hole": True}, "--hole"),
        ({"shape": "curved", "stand": True}, "--stand"),
        ({"shape": "flat", "lamp_base": True}, "--lamp-base"),
        ({"shape": "cylinder", "width": 50}, "--diameter"),
        ({"shape": "shaped", "frame": 0}, "frame"),
        ({"pose": "sideways"}, "unknown pose"),
        ({"shape": "cylinder", "pose": "lying"}, "cannot lie on the bed"),
        ({"mode": "color", "pose": "standing"}, "colour panels print lying"),
        ({"shape": "shaped", "pose": "standing"}, "need support"),
        ({"min_patch": -1}, "--min-patch must be"),
        ({"shape": "cylinder", "min_patch": 1}, "a cylinder panel prints standing"),
        ({"pose": "standing", "min_patch": 1}, "with --pose standing prints standing"),
        ({"mode": "color", "pose": "standing", "min_patch": 1}, "colour panels print lying"),
    ],
)
def test_job_checks(setup, petg, tmp_path, kw, message):
    with pytest.raises(ValueError, match=message):
        Job(photo=tmp_path / "x", setup=setup, optics=petg, **kw).check()


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


@pytest.mark.parametrize("shape", ["flat", "curved"])
def test_picture_reads_the_right_way(setup, petg, tmp_path, shape):
    """A bright (thin) square at the upper left of the picture must be at the upper left as
    seen by the viewer: from -y for view="flat" (+x is right), from +y for "relief"."""
    t = np.full((60, 80), 3.0)
    t[5:20, 5:25] = 1.0  # row 0 is the top of the picture
    j = Job(
        photo=tmp_path / "x.png", setup=setup, optics=petg, shape=shape, arc=120, pose="standing"
    )
    (x, z), (cx, cz) = thin_spot(forge.panel(j, t, 80, 60), shape, 80)
    assert z > cz and x < cx
    (x, z), (cx, cz) = thin_spot(forge.panel(replace(j, view="relief"), t, 80, 60), shape, 80)
    assert z > cz and x > cx


def test_lying_flat_reads_the_right_way(setup, petg, tmp_path):
    """A white flat panel lies on the bed by default: seen from below (view="flat"), the
    viewer's left is +x; seen from the top (view="relief") it is -x."""
    t = np.full((60, 80), 3.0)
    t[5:20, 5:25] = 1.0  # row 0 is the top of the picture: a bright upper left
    j = Job(photo=tmp_path / "x.png", setup=setup, optics=petg)
    assert not j.standing
    for view, left in (("flat", 1), ("relief", -1)):
        v = forge.panel(replace(j, view=view), t, 80, 60).vertices
        centre = (v.min(0) + v.max(0)) / 2
        thin = (v[:, 2] > 0.1) & (v[:, 2] < 1.2)
        assert v[thin, 1].mean() > centre[1]  # upper
        assert np.sign(v[thin, 0].mean() - centre[0]) == left


def test_shaped_reads_the_right_way(setup, petg, tmp_path):
    """Lying on the bed: seen from below (view="flat"), the viewer's left is +x."""
    j = Job(
        photo=tmp_path / "x",
        setup=setup,
        optics=petg,
        shape="shaped",
        mask="oval",
        width=80,
        height=60,
        pitch=0.5,
    )
    lay = forge.layout(j, 4 / 3)
    t = np.full((lay.rows, lay.cols), 3.0)
    r, c = lay.rows // 2, lay.cols // 2
    t[r - 20 : r - 5, c - 30 : c - 10] = 1.0  # upper left of the picture
    for view, left in (("flat", 1), ("relief", -1)):
        m = forge.volumes(replace(j, view=view), lay, t)[0][1]
        v = m.vertices
        thin = (v[:, 2] > 0.1) & (v[:, 2] < 1.2)
        assert v[thin, 1].mean() > lay.height / 2  # upper
        assert np.sign(v[thin, 0].mean()) == left


def test_overhang_limit():
    t = np.full((50, 10), 1.0)
    t[10] = 4.0  # a thick line above thin plastic: a ledge
    out = forge.limit_overhangs(t, 45, 0.2)
    assert np.all(out >= t)
    rise = out[:-1] - out[1:]  # row 0 is the top: how much thicker the row above is
    assert rise.max() <= 0.2 + 1e-9
    assert out[10:].max() == pytest.approx(4.0) and out[:10].max() == 1.0  # nothing above
    assert np.array_equal(forge.limit_overhangs(t, 90, 0.2), t)


def test_object_settings(setup, petg, tmp_path):
    j = Job(photo=tmp_path / "x", setup=setup, optics=petg, pose="standing")
    standing = forge.object_settings(j)
    assert standing["fill_density"] == "100%" and standing["brim_width"] == "8"
    assert int(standing["perimeters"]) * 2 * 0.45 >= petg.t_max + 0.8
    lying = forge.object_settings(
        Job(photo=tmp_path / "x", setup=setup, optics=petg, shape="shaped")
    )
    assert lying == {"fill_density": "100%", "fill_pattern": "rectilinear"}


def test_accessories_are_written(photo, setup, petg, tmp_path, no_slicer):
    res = forge.make(job(photo, setup, petg, tmp_path, width=60, stand=True, pitch=0.5))
    assert res.files["stand 3mf"].is_file() and res.files["stand stl"].is_file()
    assert trimesh.load(res.files["stand stl"]).is_watertight
    assert "Stand:" in res.files["report"].read_text(encoding="utf-8")
    res = forge.make(
        job(
            photo,
            setup,
            petg,
            tmp_path,
            shape="cylinder",
            diameter=60,
            height=50,
            lamp_base=True,
            pitch=0.6,
        )
    )
    assert res.accessories["lamp-base"]["recess_mm"][0] == pytest.approx(60.6)
