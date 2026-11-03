import math
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


def lake(h=90, w=120):
    """Blue sky, a white cloud, warm houses, dark blue water: a hard photo for one path."""
    y = np.linspace(0, 1, h)[:, None, None]
    img = np.broadcast_to(
        np.array([0.25, 0.45, 0.85]) + y * np.array([0.35, 0.3, 0.1]), (h, w, 3)
    ).copy()
    img[int(h * 0.15) : int(h * 0.25), int(w * 0.1) : int(w * 0.4)] = [0.95, 0.95, 0.97]
    img[int(h * 0.55) : int(h * 0.7)] = [0.75, 0.45, 0.3]
    img[int(h * 0.62) : int(h * 0.7), ::7] = [0.45, 0.25, 0.15]
    img[int(h * 0.7) :] = [0.1, 0.18, 0.3]
    return np.clip(img, 0, 1)


def blobs(seed: int, h=90, w=120):
    """Random coloured boxes on a random ground: no path fits them well."""
    r = np.random.default_rng(seed)
    img = np.zeros((h, w, 3)) + r.random(3)
    for _ in range(6):
        y0, x0 = r.integers(0, h - 10), r.integers(0, w - 10)
        img[y0 : y0 + r.integers(10, 40), x0 : x0 + r.integers(10, 50)] = r.random(3)
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
    grey = P.Filament("grey", "#808080", 2.0)
    seen = P.curve([grey], np.array([0.0]), np.array([2.0]), "back")[0]
    assert seen * P._exposure(2.0) == pytest.approx([0.1] * 3, rel=1e-6)  # of the LED


def test_a_coloured_filament_passes_its_own_colour():
    red = P.Filament("red", "#C62127", 2.0)
    t = P.curve([red], np.array([0.0]), np.array([1.0]), "back")[0]
    assert t[0] > t[1] and t[0] > t[2]


def test_stack_gets_darker_with_height(heights):
    fils = P.load_filaments()[:4]
    curve = P.curve(fils, np.array([0, 1.0, 1.6, 2.2]), heights, "back")
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


def test_white_is_signal_white():
    assert P.white().name == "Prusament PETG Signal White"
    assert (P.white().hex, P.white().td) == ("#E5E0E1", 3.4)  # Prusa's table


def test_four_colours_use_four_filaments_when_they_help(heights):
    """Choosing filaments with the heights fixed stopped at 3 filaments on this photo; with
    the change height chosen together with the filament, the fourth one finds its place."""
    t = P.target(lake())
    choices, source = P.choices()
    fitter = P.Fitter(t, heights)
    three = P.plan(t, 3, heights, choices, source).real
    four = P.plan(t, 4, heights, choices, source).real
    assert len(four.filaments) == 4
    assert (
        fitter.cost_of(four.filaments, four.starts)
        < fitter.cost_of(three.filaments, three.starts) - 1
    )


def test_snapping_cannot_spoil_the_choice(heights):
    """A search that fitted heights between layer tops chose a plan here that got worse
    once snapped to layers (cost 28.0, worse than the 26.6 of fixed heights). The search
    now stays on layer tops, so the printed plan is the one it scored."""
    t = P.target(blobs(0))
    choices, source = P.choices()
    plan = P.plan(t, 4, heights, choices, source)
    assert P.Fitter(t, heights).cost_of(plan.real.filaments, plan.real.starts) < 26.6


def test_choose_stays_on_layer_tops(heights):
    t = P.target(lake())
    fitter = P.Fitter(t, heights)
    choices, _ = P.choices()
    wanted = choices[:5]
    picked, starts = fitter.choose([(wanted, np.array([0, 0.9, 1.4, 2.0, 2.6]))], choices)
    assert len(picked) == 5 and starts[0] == 0
    assert all(np.isclose(heights, z).any() for z in starts[1:])
    assert np.all(np.diff(starts) > 0)


def test_no_optimiser_warnings(heights):
    """Warnings from scipy reached the user's terminal."""
    import warnings

    from scipy.optimize import OptimizeWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        P.plan(P.target(lake()), 6, heights, *P.choices())


def test_batched_cost_is_the_cost(heights):
    """Fitter.costs (all candidates at once, lower heights reused) agrees with cost()."""
    fitter = P.Fitter(P.target(lake()), heights)
    fil = P.load_filaments()[:4]
    starts = np.array([0, 0.9, 1.5, 2.2])
    att = fitter.props(fil)
    plain = fitter.cost(att, starts)
    low, high = heights[heights <= 0.9 + 1e-9], heights[heights > 0.9 + 1e-9]
    below = fitter._nearest_d2(att[None], starts[None], low)[0]
    assert fitter.costs(att, starts[None], high, below)[0] == pytest.approx(plain, rel=1e-6)
    many = fitter.costs(np.repeat(att[None], 3, 0), np.repeat(starts[None], 3, 0))
    assert many == pytest.approx([plain] * 3, rel=1e-6)


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


@pytest.mark.parametrize(
    "value, result",
    [
        ('"ece8dc"', "#ECE8DC"),
        ('"#ece8dc"', "#ECE8DC"),
        ('" #ECE8DC "', "#ECE8DC"),
        ('"#ECE8DC80"', None),
        ('"#E5E0E1;#000000"', None),
        ('"#ECE"', None),
    ],
)
def test_filament_hex_is_checked_and_spelled_one_way(tmp_path, value, result):
    """The hex goes to the preview, the M600 list and PrusaSlicer: only #RRGGBB passes."""
    path = tmp_path / "f.toml"
    path.write_text(f'[[filament]]\nname = "x"\nhex = {value}\ntd = 1\n', encoding="utf-8")
    if result is None:
        with pytest.raises(ValueError, match="#RRGGBB"):
            P.load_filaments(path)
    else:
        assert P.load_filaments(path)[0].hex == result


@pytest.mark.parametrize("td", ["nan", "inf", "-1"])
def test_filament_td_must_be_a_positive_number(tmp_path, td):
    path = tmp_path / "f.toml"
    path.write_text(f'[[filament]]\nname = "x"\nhex = "#808080"\ntd = {td}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="td must be"):
        P.load_filaments(path)


@pytest.mark.parametrize(
    "extra, message",
    [
        ("k = [1, 2, 3]\ns = 0.5\n", None),
        ("k = [1, 2, 3]\n", "go together"),
        ("s = 0.5\n", "go together"),
        ("k = [1, 2]\ns = 0.5\n", "three numbers"),
        ("k = [1, 2, -3]\ns = 0.5\n", "positive"),
        ("k = [1, 2, 3]\ns = 0\n", "positive"),
        ('k = [1, 2, 3]\ns = "a"\n', "positive"),
    ],
)
def test_measured_k_and_s(tmp_path, extra, message):
    path = tmp_path / "f.toml"
    path.write_text(f'[[filament]]\nname = "x"\nhex = "#808080"\ntd = 1\n{extra}', encoding="utf-8")
    if message:
        with pytest.raises(ValueError, match=message):
            P.load_filaments(path)
    else:
        assert P.load_filaments(path)[0].optics == ((1.0, 2.0, 3.0), 0.5)


# light from the front: a picture you look at in room light (two-flux)

YELLOW, BLUE = "Prusament PETG Yellow Gold", "Prusament PETG Ultramarine Blue Transparent"


def catalog(name: str) -> P.Filament:
    return next(f for f in P.load_filaments() if f.name == name)


def front_lab(layers) -> np.ndarray:
    """CIELAB of one stack in room light: (filament, mm) pairs from the bed up."""
    starts = np.concatenate([[0.0], np.cumsum([d for _, d in layers])[:-1]])
    top = np.array([sum(d for _, d in layers)])
    return P.lab(P.curve([f for f, _ in layers], starts, top, "front"))[0]


def test_two_flux_layer_limits():
    """A layer of no thickness shows what lies under it; a thick layer shows its own colour
    (for an unmeasured filament, its hex), whatever lies under it."""
    f = P.white()
    a, b, s = np.split(P.two_flux(f.linear, [f.td])[0], [3, 6])
    under = np.array([0.2, 0.5, 0.9])
    assert np.allclose(P.reflect(a, b, 0 * b, under), under)
    for below in (under, 0 * under):
        assert np.allclose(P.reflect(a, b, b * s * 50, below), f.linear, rtol=1e-6)


def test_derived_scattering_lets_a_tenth_through_td():
    fils = P.load_filaments()
    lin, td = P._arrays(fils)
    a, b, s = np.split(P.two_flux(lin, td), [3, 6], axis=1)
    through = P.luminance(P.transmit(a, b, b * s * td[:, None]))
    assert through == pytest.approx(np.full(len(fils), P.TD_THROUGH), rel=1e-3)


def test_measured_optics_win():
    yellow = catalog(YELLOW)
    k, s = yellow.optics
    props = P.two_flux(*P._arrays([yellow]), P._optics([yellow]))[0]
    assert props[6] == s and np.allclose(props[:3], 1 + np.array(k) / s)


def test_white_over_black_builds_up_slowly():
    """Measured on the owner's print: one layer of Signal White over black is a mid grey,
    three layers are still about L* 72, and every layer makes it lighter."""
    black, white = catalog("Prusament PETG Jet Black"), P.white()
    lstar = [front_lab([(black, 0.6), (white, 0.1 * n)])[0] for n in range(1, 9)]
    assert np.all(np.diff(lstar) > 0)
    assert 40 < lstar[0] < 70
    assert lstar[2] < 90  # the paint model of the first print said 96


def test_the_failed_print_is_predicted():
    """The owner's first print in room light: thin Yellow Gold over white printed as a
    bright lemon, and thin Ultramarine over that as a dark green to navy. The paint model
    had planned the yellow as a dark khaki."""
    black, white = catalog("Prusament PETG Jet Black"), P.white()
    stack = [(black, 0.6), (white, 0.8)]
    lemon = front_lab([*stack, (catalog(YELLOW), 0.2)])
    assert lemon[0] > 75 and lemon[2] > 60
    dark = front_lab([*stack, (catalog(YELLOW), 0.8), (catalog(BLUE), 0.3)])
    assert dark[0] < 40


def test_front_model_gives_no_numpy_warnings(heights):
    """Thick layers of very opaque and very clear filaments must not overflow: a warning
    would reach the terminal."""
    import warnings

    names = ["Prusament PETG Clear", "Prusament PETG Prusa Galaxy Black", YELLOW]
    fils = [catalog(n) for n in names]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        light = P.light_model("front", heights)
        props = light.props(*P._arrays(fils), P._optics(fils))
        depth = P._depths(np.array([[0.0, 0.6, 1.0]]), np.linspace(0, 50, 30))
        seen = light.seen(depth, props[None])
        assert light.hides_from(fils[0]) is None and light.base_td[1] > 0
    assert np.isfinite(seen).all()


def test_front_light_shows_the_top_filament(heights):
    """Black, then white from 1.0 mm: thin points are black, and above 1.0 mm the white
    hides the black more with every layer, but slowly, because it is translucent."""
    black, white = P.Filament("black", "#292B2B", 0.3), P.white()
    seen = P.lab(P.curve([black, white], np.array([0, 1.0]), heights, "front"))[:, 0]
    black_seen = P.lab(black.linear / P.luminance(white.linear))[0]
    assert seen[heights <= 1.0 + 1e-9] == pytest.approx(black_seen, abs=0.1)
    assert 85 < seen[-1] < 99
    assert np.all(np.diff(seen) >= -1e-9)
    thick = P.curve([black, white], np.array([0, 1.0]), np.array([30.0]), "front")
    assert P.lab(thick)[0, 0] == pytest.approx(100, abs=0.5)  # the viewer's white
    # the same stack lit from behind is darkest where it is thickest
    back = P.lab(P.curve([black, white], np.array([0, 1.0]), heights, "back"))[:, 0]
    assert back[-1] < back[0] < 30


def test_front_plan_fits_a_sunset(heights):
    t = P.target(sunset())
    plan = P.plan(t, 4, heights, P.load_filaments(), "catalog", "front")
    assert plan.real.mean_de < 10
    # the ideal colours may be any colour, so they are at least as close as the print (a
    # fit with free heights stalled in room light and lost quality on layer tops)
    assert plan.ideal.mean_de <= plan.real.mean_de
    base = plan.real.filaments[0]
    assert P.light_model("front", heights).is_base(base)  # it hides the back
    starts = plan.real.starts
    assert starts[0] == 0 and np.all(np.diff(starts) > 0)
    assert all(np.isclose(heights, z).any() for z in starts[1:])  # on layer tops


def test_front_ideal_colours_do_not_depend_on_the_list(heights):
    t = P.target(lake())
    a = P.plan(t, 3, heights, P.load_filaments(), "catalog", "front")
    b = P.plan(t, 3, heights, P.load_filaments(EXAMPLE), "shelf", "front")
    assert [(f.hex, f.td) for f in a.ideal.filaments] == [(f.hex, f.td) for f in b.ideal.filaments]
    assert np.array_equal(a.ideal.starts, b.ideal.starts)


def test_front_base_must_hide_the_back(heights):
    """At the thinnest point (0.6 mm) the base must look the same over a black and over a
    white wall: black does, Signal White needs about 1.8 mm, clear never does."""
    light = P.light_model("front", heights)
    assert light.is_base(P.Filament("black", "#000000", 0.3))
    assert not light.is_base(P.white())
    assert 1.4 <= light.hides_from(P.white()) <= 2.2
    assert light.hides_from(P.Filament("clear", "#E4E7E3", 100)) is None
    # the ideal base may be as translucent as a white that still hides the back
    td = light.base_td[1]
    assert light.is_base(P.Filament("w", P.white().hex, td * 0.99))
    assert not light.is_base(P.Filament("w", P.white().hex, td * 1.01))
    with pytest.raises(ValueError, match="unknown light"):
        P.light_model("side", heights)


def test_front_batched_cost_is_the_cost(heights):
    fitter = P.Fitter(P.target(lake()), heights, light="front")
    props = fitter.props(P.load_filaments()[:4])
    starts = np.array([0, 0.9, 1.5, 2.2])
    plain = fitter.cost(props, starts)
    low, high = heights[heights <= 0.9 + 1e-9], heights[heights > 0.9 + 1e-9]
    below = fitter._nearest_d2(props[None], starts[None], low)[0]
    assert fitter.costs(props, starts[None], high, below)[0] == pytest.approx(plain, rel=1e-6)


def test_front_stack_is_layer_by_layer(heights):
    """`Frontlight.seen` (all plans and heights at once, written with e^-2x) equals the
    two-flux formula with coth, applied one layer at a time to one point."""
    rng = np.random.default_rng(1)
    light = P.light_model("front", heights)
    props = light.props(rng.uniform(0.02, 0.9, (4, 3)), rng.uniform(0.3, 8, 4))
    starts = np.array([0, 0.9, 1.5, 2.2])
    ends = [*starts[1:], math.inf]
    got = light.seen(P._depths(starts[None], heights), props[None])[0]
    for z, seen in zip(heights, got):
        r = [0.0, 0.0, 0.0]
        for k in range(4):
            d = min(max(z - starts[k], 0), ends[k] - starts[k])
            for c in range(3):
                a, s = props[k, c], props[k, 6]
                b = math.sqrt(a * a - 1)
                if d > 0:
                    coth = 1 / math.tanh(b * s * d)
                    r[c] = (1 - r[c] * (a - b * coth)) / (a - r[c] + b * coth)
        assert seen == pytest.approx(np.array(r) / light.white, rel=1e-6, abs=1e-12)


def test_unpredictable_filaments_are_left_out_in_room_light(tmp_path):
    """Hex and TD cannot predict a translucent coloured or nearly clear filament in room
    light: without measured k and s it is left out, unless allow_unmeasured. Lit from
    behind it stays."""
    orange, natural = "Prusament PETG Prusa Orange", "Prusament PETG V0 Natural"
    kept, source = P.choices(light="front")
    names = {f.name for f in kept}
    assert {"Prusament PETG Signal White", "Prusament PETG Jet Black", YELLOW, BLUE} <= names
    assert orange not in names and "translucent filaments without measured k and s" in source
    assert natural not in names  # nearly clear: its hex says little about its scattering
    assert orange in {f.name for f in P.choices(light="front", allow_unmeasured=True)[0]}
    assert orange in {f.name for f in P.choices(light="back")[0]}
    own = tmp_path / "own.toml"
    own.write_text('[[filament]]\nname = "orange"\nhex = "#F07318"\ntd = 6.2\n')
    with pytest.raises(ValueError, match="allow-unmeasured"):
        P.choices(own, light="front")
    assert len(P.choices(own)[0]) == 1


@pytest.mark.parametrize("light", ["front", "back"])
def test_few_photo_colours_give_no_warnings(heights, light):
    """A photo with fewer colours than k-means asks for made scipy warn on the terminal."""
    import warnings

    img = np.zeros((40, 60, 3)) + [0.8, 0.7, 0.6]
    img[20:] = [0.1, 0.2, 0.4]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = P.plan(P.target(img), 6, heights, *P.choices(), light=light)
    assert len(plan.ideal.filaments) >= 1


def test_ideal_colours_come_from_the_photo(heights):
    """The candidates of the ideal search: the photo's colours and greys, nothing else."""
    t = P.target(sunset())
    found = P.photo_colours(P.Fitter(t, heights).sample)
    assert len(found) == (P.PHOTO_COLOURS + len(P.GREYS)) * len(P.TD_STEPS)
    assert all(P.TD_RANGE[0] <= f.td <= P.TD_RANGE[1] for f in found)


def test_a_fitter_without_weights_samples_as_before(heights):
    """No --focus, no change: the sample is the one NumPy draws without p (with p, even an
    even one, it draws another), so every plan without a focus stays the same."""
    t = P.target(sunset())
    flat = t.reshape(-1, 3)
    old = flat[np.random.default_rng(0).choice(len(flat), 3000, replace=False)]
    assert np.array_equal(P.Fitter(t, heights).sample, old)
    columns = np.zeros((90, 120, 3))
    columns[..., 0] = np.arange(120)  # tells which column a sampled pixel comes from
    weights = np.where(np.arange(120) < 12, 1.0, 0.1)  # a tenth of the pixels weighs 10 x
    fitter = P.Fitter(columns, heights, weights=np.broadcast_to(weights, (90, 120)))
    assert 0.48 < (fitter.sample[:, 0] < 12).mean() < 0.58  # 1080 / (1080 + 972), not 0.1


def disc_photo(colour=(0.1, 0.35, 0.8)):
    """A sunset with a small blue disc: a colour far from the sunset's path."""
    rgb = sunset()
    y, x = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]]
    disc = np.hypot(x - 85, y - 30) < 12  # 4 % of the picture
    rgb[disc] = colour
    return rgb, disc


def test_focus_serves_the_subject(heights):
    """With the disc as the focus, the plan's colours go to the disc."""
    rgb, disc = disc_photo()
    t = P.target(rgb)
    choices, source = P.choices(None, False, "front", False)
    plain = P.plan(t, 3, heights, choices, source, "front")
    assert plain.real.focus_de is None
    before = P.evaluate(plain.real.filaments, plain.real.starts, heights, t, "front", disc)
    focused = P.plan(t, 3, heights, choices, source, "front", disc.astype(float))
    assert focused.real.focus_de < 0.5 * before.focus_de
    assert focused.ideal.focus_de < focused.real.focus_de + 5
    ideal_before = P.evaluate(plain.ideal.filaments, plain.ideal.starts, heights, t, "front", disc)
    assert focused.ideal.focus_de < 0.5 * ideal_before.focus_de


def test_the_subject_delta_e_is_the_mean_over_it(heights):
    fit = P.evaluate([P.white()], np.array([0.0]), heights, P.target(sunset()), "front")
    rows = np.zeros((90, 120), dtype=bool)
    rows[:30] = True
    one = P.score(fit, P.target(sunset()), np.zeros((90, 120), dtype=int), rows)
    de = P.delta_e2000(P.target(sunset())[:30].reshape(-1, 3), P.lab(fit.colours[[0]]))
    assert one.focus_de == pytest.approx(de.mean())
    assert P.score(fit, P.target(sunset()), np.zeros((90, 120), dtype=int)).focus_de is None
