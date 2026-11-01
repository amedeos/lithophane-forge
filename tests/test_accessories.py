import pytest
import trimesh

from lithophane_forge import accessories, shapes


def closed(m):
    tm = trimesh.Trimesh(m.vertices, m.faces, process=False)
    return tm.is_watertight and tm.is_winding_consistent and m.volume() > 0


def test_stand_slot_and_size():
    m, info = accessories.stand(100, 80, 4.4, 3)
    assert closed(m)
    assert info["slot_mm"] == [pytest.approx(4.6), 3]
    assert m.bounds[0][2] == pytest.approx(0)
    assert m.size[0] == pytest.approx(60) and m.size[2] == pytest.approx(7)


def test_lamp_base():
    m, info = accessories.lamp_base(80, 4.4, 3)
    assert closed(m)
    assert info["recess_mm"][0] == pytest.approx(80 + 2 * accessories.CLR_LAMP)
    assert info["led_pocket_mm"][0] == pytest.approx(41)
    with pytest.raises(ValueError, match="does not fit"):
        accessories.lamp_base(45, 4.4, 3)


def test_frame_and_tab_follow_the_outline():
    heart = shapes.fit(shapes.unit("heart"), 80, 72)
    ring = accessories.frame_ring(heart, 3, 4.4)
    assert closed(ring)
    assert ring.size[:2] == pytest.approx([80, 72], abs=0.05)
    tab = accessories.tab_outline(heart)
    hole = next(iter(tab.interiors))
    from shapely.geometry import Polygon

    assert not Polygon(hole).intersects(heart)  # the hole is clear of the panel
    assert tab.intersects(heart)  # and the tab holds on to it
    assert closed(accessories.tab(heart, 4.4))
