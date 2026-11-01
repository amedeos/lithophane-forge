import numpy as np
import pytest

from lithophane_forge.optics import DEFAULTS, Optics, optics


@pytest.mark.parametrize("material", sorted(DEFAULTS))
def test_thickness_is_monotonic_and_in_range(material):
    o = DEFAULTS[material]
    light = np.linspace(0, 1, 101)
    t = o.thickness(light)
    assert t[0] == pytest.approx(o.t_max)
    assert t[-1] == pytest.approx(o.t_min)
    assert np.all(np.diff(t) < 0)


def test_light_round_trip(petg):
    light = np.linspace(0, 1, 11)
    back = (petg.light(petg.thickness(light)) - petg.floor) / (1 - petg.floor)
    assert np.allclose(back, light)


def test_td_means_ten_percent():
    o = Optics(t_min=1.0, t_max=4.0, td=2.0)
    assert o.light(1.0 + 2.0) == pytest.approx(0.1)


def test_overrides_and_checks():
    assert optics("petg", t_max=4.0).t_max == 4.0
    with pytest.raises(ValueError):
        optics("petg", t_min=3.0, t_max=2.0)
