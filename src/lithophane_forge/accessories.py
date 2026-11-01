"""Parametric parts made with build123d: frames for shaped panels, hanging tabs, a stand
for flat panels and a lamp base for cylinders.

The lithophane itself is a numpy mesh; these parts are simple solids. A part that belongs
to the panel (frame, tab) becomes another volume of the same 3MF object, overlapping the
panel: PrusaSlicer joins them when it slices. Stand and lamp base print on their own.

build123d gotchas (from the owner's 3d-linux project): always give `dir=(0, 0, 1)` when
you extrude a face (a clockwise polygon extrudes downwards otherwise), and keep corners on
a 1 micron grid.
"""

import math

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box

from .mesh import Mesh

GRID = 1e-3  # mm: every corner on this grid
CLR_SLOT = 0.10  # stand slot <-> panel, per side: fitted well in PETG on an MK4 (3d-linux)
CLR_LAMP = 0.30  # lamp base recess <-> cylinder, per side (a big circle prints a bit small)
HOLE_D = 5.0  # hanging hole
TAB_WALL = 2.5  # plastic around the hanging hole


def to_mesh(part, tolerance: float = 0.02) -> Mesh:
    """Tessellate a build123d part into one closed mesh (shared corners merged)."""
    verts, tris = part.tessellate(tolerance, 0.1)
    v = np.array([[p.X, p.Y, p.Z] for p in verts])
    t = np.array(tris, dtype=np.int64)
    key = np.round(v / 1e-5).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    t = inverse.reshape(-1)[t]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    m = Mesh(v[first], t)
    return m if m.volume() > 0 else m.flipped()


def _wire(coords):
    from build123d import Wire

    pts = []
    for x, y in list(coords)[:-1]:
        if not pts or math.dist((x, y), pts[-1]) > GRID / 2:
            pts.append((x, y))
    while len(pts) > 1 and math.dist(pts[0], pts[-1]) <= GRID / 2:
        pts.pop()
    if len(pts) < 3 or Polygon(pts).area < 1e-4:
        return None
    return Wire.make_polygon([(x, y, 0) for x, y in pts], close=True)


def extrude_xy(geom, thickness: float):
    """A shapely footprint extruded from z = 0 to `thickness` (a build123d part)."""
    from build123d import Face, Part, extrude

    solids = []
    geom = shapely.set_precision(geom, GRID)
    for poly in getattr(geom, "geoms", [geom]):
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        outer = _wire(poly.exterior.coords)
        if outer is None:
            continue
        holes = [w for w in (_wire(r.coords) for r in poly.interiors) if w is not None]
        solids.append(extrude(Face(outer, holes), thickness, dir=(0, 0, 1)))
    if not solids:
        raise ValueError("nothing to extrude")
    return solids[0] if len(solids) == 1 else Part(solids)


def remap(m: Mesh, axes: tuple[int, int, int]) -> Mesh:
    """New x, y, z = old axes[0], axes[1], axes[2] (the faces still point outwards)."""
    out = Mesh(m.vertices[:, list(axes)], m.faces)
    return out if out.volume() > 0 else out.flipped()


def frame_ring(outline: Polygon, width: float, thickness: float) -> Mesh:
    """A frame `width` mm wide along the inside of the outline, lying (z up)."""
    ring = outline.difference(outline.buffer(-width, join_style="round"))
    return to_mesh(extrude_xy(ring, thickness))


def tab_outline(outline: Polygon, hole_d: float = HOLE_D, wall: float = TAB_WALL) -> Polygon:
    """A hanging tab above the outline, centred on x = 0: a rectangle with a round top
    (no overhang when it prints standing) and a hole that stays clear of the outline."""
    r = hole_d / 2 + wall
    _, y0, _, y1 = outline.bounds
    probe = outline.intersection(box(-0.01, y0, 0.01, y1 + 1))
    top = probe.bounds[3] if not probe.is_empty else y1
    cy = top + hole_d / 2 + 1.2
    while Point(0, cy).buffer(hole_d / 2 + 0.6).intersects(outline):
        cy += 0.5
    body = Point(0, cy).buffer(r, quad_segs=32).union(box(-r, top - 2.0, r, cy))
    return body.difference(Point(0, cy).buffer(hole_d / 2, quad_segs=32))


def tab(outline: Polygon, thickness: float) -> Mesh:
    """The hanging tab, lying (z up), to join a shaped or flat panel."""
    return to_mesh(extrude_xy(tab_outline(outline), thickness))


def stand(
    panel_width: float, panel_height: float, frame_t: float, frame_w: float
) -> tuple[Mesh, dict]:
    """A stand for a flat panel: a sloped block with a slot along x, on the bed.

    The slot is as deep as the bottom frame (at least 3 mm), so it hides no picture."""
    slot_w = frame_t + 2 * CLR_SLOT
    slot_d = min(max(frame_w, 3.0), 8.0)
    length = min(max(0.6 * panel_width, 40.0), panel_width)
    depth = max(30.0, 0.3 * panel_height)
    height = slot_d + 4.0
    top = slot_w + 2 * 6.0
    # profile in the (y, z) plane, extruded along x
    profile = Polygon([(-depth / 2, 0), (depth / 2, 0), (top / 2, height), (-top / 2, height)])
    profile = profile.difference(box(-slot_w / 2, height - slot_d, slot_w / 2, height + 1))
    m = to_mesh(extrude_xy(profile, length))  # x = profile y, y = profile z, z = along
    m = remap(m, (2, 0, 1)).moved((-length / 2, 0, 0))
    info = {
        "slot_mm": [round(slot_w, 2), round(slot_d, 2)],
        "size_mm": [round(length, 1), round(depth, 1), round(height, 1)],
    }
    return m, info


def lamp_base(
    diameter: float, frame_t: float, rim: float, led_d: float = 40.0, led_h: float = 18.0
) -> tuple[Mesh, dict]:
    """A base for a cylinder lamp: a recess the cylinder sits in, a pocket for a tealight
    LED (or a puck light) under it, and a notch for a cable."""
    from build123d import Align, Box, Cylinder, Pos

    floor, wall, depth = 2.0, 3.0, max(2.0, min(rim, 5.0))
    r_recess = diameter / 2 + CLR_LAMP
    r_led = led_d / 2 + 0.5
    inner = diameter / 2 - frame_t
    if r_led > inner - 1.0:
        raise ValueError(
            f"the LED ({led_d:g} mm) does not fit inside the cylinder "
            f"(inner diameter {2 * inner:.0f} mm): use a wider cylinder or "
            "--led-diameter"
        )
    rb = r_recess + wall
    hb = floor + led_h
    low = (Align.CENTER, Align.CENTER, Align.MIN)
    part = Cylinder(rb, hb, align=low)
    part -= Pos(0, 0, hb - depth) * Cylinder(r_recess, depth + 1, align=low)
    part -= Pos(0, 0, floor) * Cylinder(r_led, hb, align=low)
    part -= Pos(0, rb / 2, floor) * Box(8.0, rb + 1, 5.0, align=low)
    info = {
        "outer_d_mm": round(2 * rb, 1),
        "height_mm": round(hb, 1),
        "recess_mm": [round(2 * r_recess, 2), round(depth, 1)],
        "led_pocket_mm": [round(2 * r_led, 1), round(led_h, 1)],
    }
    return to_mesh(part, 0.02), info
