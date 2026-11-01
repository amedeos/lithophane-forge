from pathlib import Path

import numpy as np
import pytest

from lithophane_forge import palette as P

EXAMPLE = Path(__file__).parents[1] / "examples" / "filaments.toml"


def sunset(h=90, w=120):
    """White sky to orange, a dark hill: colours that lie on one path from light to dark."""
    y = np.linspace(0, 1, h)[:, None, None]
    img = np.broadcast_to(
        np.array([1.0, 0.95, 0.85]) - y * np.array([0.1, 0.55, 0.8]), (h, w, 3)
    ).copy()
    img[int(h * 0.7) :] = [0.1, 0.06, 0.04]
    return img


@pytest.fixture(scope="module")
def heights():
    return P.heights(0.6, 3.0, 0.2, 0.1)


def test_colour_conversions_round_trip():
    lin = np.random.default_rng(0).random((50, 3))
    assert np.allclose(P.lab_to_linear(P.lab(lin)), lin, atol=1e-6)
    assert P.linear_to_hex(P.hex_to_linear("#E5E0E1")) == "#E5E0E1"
    assert P.lab(np.ones(3))[0] == pytest.approx(100, abs=0.01)


def test_heights_are_layer_tops(heights):
    assert heights[0] == pytest.approx(0.6) and heights[-1] == pytest.approx(3.0)
    assert np.allclose(np.diff(heights), 0.1)


def test_grey_filament_lets_ten_percent_through_at_td():
    grey = P.hex_to_linear("#808080")
    att = P.attenuation(grey[None], np.array([2.0]))
    t = P.stack(att, np.array([0.0]), np.array([2.0]))[0]
    assert t == pytest.approx([0.1] * 3, rel=1e-6)


def test_a_coloured_filament_passes_its_own_colour():
    red = P.hex_to_linear("#C62127")
    t = P.stack(P.attenuation(red[None], np.array([2.0])), np.array([0.0]), np.array([1.0]))[0]
    assert t[0] > t[1] and t[0] > t[2]


def test_stack_gets_darker_with_height(heights):
    fils = P.load_filaments()[:4]
    att = P.attenuation(np.array([f.linear for f in fils]), np.array([f.td for f in fils]))
    curve = P.stack(att, np.array([0, 1.0, 1.6, 2.2]), heights)
    assert np.all(np.diff(P.luminance(curve)) <= 1e-12)


def test_plan_fits_a_sunset(heights):
    t = P.target(sunset())
    plan = P.plan(t, 4, heights, P.load_filaments(), "catalog")
    assert plan.ideal.mean_de < plan.real.mean_de  # free colours fit this photo better
    assert plan.real.mean_de < 10
    assert P.is_base(plan.real.filaments[0])  # a white or natural base
    starts = plan.real.starts
    assert starts[0] == 0 and np.all(np.diff(starts) > 0)
    assert all(np.isclose(heights, z).any() for z in starts[1:])  # on layer tops
    changes = plan.real.colour_changes(0.1)
    assert [z for z, _ in changes] == pytest.approx(list(starts[1:] + 0.1))
    assert [c for _, c in changes] == [f.hex for f in plan.real.filaments[1:]]


def test_plan_beats_naive_kmeans(heights):
    """k-means colours with evenly spread changes is the naive answer; the fit is better."""
    t = P.target(sunset())
    plan = P.plan(t, 4, heights, P.load_filaments(), "catalog")
    fitter = P.Fitter(t, heights)
    lin = np.clip(P.lab_to_linear(P.kmeans_lab(fitter.sample, 4)), 0.02, 0.98)
    naive = [P.Filament("k", P.linear_to_hex(c), float(td)) for c, td in zip(lin, P.ideal_td(lin))]
    even = np.concatenate([[0], P._snap(heights[0] + np.arange(1, 4) * 0.6, heights)])
    assert plan.ideal.mean_de < P.evaluate(naive, even, heights, t).mean_de


def test_ideal_colours_do_not_depend_on_the_list(heights):
    t = P.target(sunset())
    a = P.plan(t, 3, heights, P.load_filaments(), "catalog")
    b = P.plan(t, 3, heights, P.load_filaments(EXAMPLE), "shelf")
    assert [(f.hex, f.td) for f in a.ideal.filaments] == [(f.hex, f.td) for f in b.ideal.filaments]
    assert np.array_equal(a.ideal.starts, b.ideal.starts)
    assert {f.name for f in b.real.filaments} <= {f.name for f in P.load_filaments(EXAMPLE)}


@pytest.mark.parametrize("k", [5, 6, 8])
def test_every_change_is_inside_the_print(heights, k):
    """No filament starts at the top layer: its M600 would be above the print."""
    plan = P.plan(P.target(sunset()), k, heights, P.load_filaments(), "catalog")
    for fit in (plan.ideal, plan.real):
        assert len(fit.filaments) <= k and np.all(np.diff(fit.starts) > 0)
        assert all(z <= heights[-1] + 1e-9 for z, _ in fit.colour_changes(0.1))
    assert len(plan.matches) == len(plan.ideal.filaments)


def test_more_colours_do_not_collapse(heights):
    """Changing one filament at a time can end on white and black only; the k-means start
    keeps more colours from doing worse than that."""
    t = P.target(sunset())
    four = P.plan(t, 4, heights, P.load_filaments(), "catalog").real.mean_de
    six = P.plan(t, 6, heights, P.load_filaments(), "catalog").real.mean_de
    assert six < 6 and six < four + 2


def test_abrasive_filaments_are_left_out(tmp_path):
    """Tungsten 75 % needs a hardened steel nozzle: only --allow-abrasive suggests it."""
    tungsten = "Prusament PETG Tungsten 75%"
    assert next(f for f in P.load_filaments() if f.name == tungsten).abrasive
    kept, source = P.choices()
    assert tungsten not in {f.name for f in kept} and "left out as abrasive" in source
    every, source = P.choices(allow_abrasive=True)
    assert tungsten in {f.name for f in every} and source == P.CATALOG
    own = tmp_path / "own.toml"
    own.write_text('[[filament]]\nname = "w"\nhex = "#808080"\ntd = 1\nabrasive = true\n')
    with pytest.raises(ValueError, match="allow-abrasive"):
        P.choices(own)
    own.write_text('[[filament]]\nname = "w"\nhex = "#808080"\ntd = 1\nabrasive = "yes"\n')
    with pytest.raises(ValueError, match="true or false"):
        P.load_filaments(own)


def test_load_filaments(tmp_path):
    assert len(P.load_filaments()) >= 20
    assert any(f.name == "Prusament PETG Signal White" for f in P.load_filaments())
    bad = tmp_path / "bad.toml"
    bad.write_text('[[filament]]\nname = "x"\nhex = "#zzzzzz"\ntd = 1\n', encoding="utf-8")
    with pytest.raises(ValueError):
        P.load_filaments(bad)
    bad.write_text('[[filament]]\nname = "x"\nhex = "#ffffff"\ntd = 0\n', encoding="utf-8")
    with pytest.raises(ValueError):
        P.load_filaments(bad)
