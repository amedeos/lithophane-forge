import math

import numpy as np
import pytest
import trimesh

from lithophane_forge import mesh


def tri(m):
    return trimesh.Trimesh(m.vertices, m.faces, process=False)


def sample(rows=30, cols=40, seed=1):
    return 0.8 + 2.8 * np.random.default_rng(seed).random((rows, cols))


@pytest.mark.parametrize(
    "make",
    [
        lambda t: mesh.standing(t, 40, 30),
        lambda t: mesh.lying(t, 40, 30),
        lambda t: mesh.arc(t, 40, 30, radius=25),
        lambda t: mesh.arc(t, 2 * math.pi * 20, 30, radius=20, wrap=True),
    ],
    ids=["standing", "lying", "arc", "cylinder"],
)
def test_closed_and_outward(make):
    m = make(sample())
    tm = tri(m)
    assert tm.is_watertight and tm.is_winding_consistent
    assert m.volume() > 0
    assert (tm.area_faces > 1e-9).all()


def test_triangle_count_is_about_two_per_point():
    t = sample(50, 60)
    assert len(mesh.standing(t, 60, 50).faces) < 2 * 50 * 60 * 1.1


def test_standing_size_and_volume():
    t = np.full((21, 41), 2.0)
    m = mesh.standing(t, 40, 20)
    assert m.size == pytest.approx([40, 2, 20])
    assert m.volume() == pytest.approx(40 * 20 * 2)


def test_arc_length_and_radius():
    t = np.full((11, 61), 1.0)
    r, width = 30.0, 40.0
    m = mesh.arc(t, width, 10, radius=r)
    radial = np.hypot(m.vertices[:, 0], m.vertices[:, 1])
    assert radial.max() == pytest.approx(r) and radial.min() == pytest.approx(r - 1)
    # chord of an arc of length `width` on radius r
    assert m.size[0] == pytest.approx(2 * r * math.sin(width / r / 2), rel=1e-6)


def test_arc_thicker_than_radius_fails():
    with pytest.raises(ValueError):
        mesh.arc(np.full((5, 5), 3.0), 10, 10, radius=2.0)


def test_write_stl(tmp_path):
    m = mesh.standing(sample(), 40, 30)
    path = tmp_path / "x.stl"
    mesh.write_stl(m, path)
    loaded = trimesh.load(path)
    assert loaded.is_watertight
    assert loaded.volume == pytest.approx(m.volume(), rel=1e-4)


def test_on_bed():
    m = mesh.standing(sample(), 40, 30).on_bed((125, 105))
    lo, hi = m.bounds
    assert lo[2] == pytest.approx(0)
    assert (lo[:2] + hi[:2]) / 2 == pytest.approx([125, 105])
