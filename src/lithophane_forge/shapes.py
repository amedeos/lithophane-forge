"""Outlines for shaped panels: heart, oval, circle, or your own SVG or PNG.

An outline is a shapely polygon in millimetres, as the viewer sees it (y up), with its box
from -width/2 to width/2 and from 0 to height.
"""

from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

NAMES = ("heart", "oval", "circle")
POINTS = 720


def _heart() -> Polygon:
    t = np.linspace(0, 2 * np.pi, POINTS, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return Polygon(np.column_stack([x, y]))


def _ellipse() -> Polygon:
    t = np.linspace(0, 2 * np.pi, POINTS, endpoint=False)
    return Polygon(np.column_stack([np.cos(t), np.sin(t)]))


def _largest(geom) -> Polygon:
    geom = shapely.make_valid(geom)
    polys = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, Polygon)]
    polys += [p for g in getattr(geom, "geoms", []) if isinstance(g, MultiPolygon) for p in g.geoms]
    if not polys:
        raise ValueError("the outline has no area")
    return max(polys, key=lambda p: p.area)


def _svg(path: Path) -> Polygon:
    from svgelements import SVG, Shape
    from svgelements import Path as SvgPath

    rings = []
    for element in SVG.parse(str(path)).elements():
        if not isinstance(element, Shape):
            continue
        for sub in SvgPath(element).as_subpaths():
            sub = SvgPath(sub)
            if sub.length() == 0:
                continue
            xy = np.asarray(sub.npoint(np.linspace(0, 1, POINTS)), dtype=float)
            pts = np.column_stack([xy[:, 0], -xy[:, 1]])  # SVG y points down
            if len(pts) >= 3:
                ring = shapely.make_valid(Polygon(pts))
                if ring.area > 0:
                    rings.append(ring)
    if not rings:
        raise ValueError(f"{path.name}: no closed shape found")
    return _largest(unary_union(rings))


def _png(path: Path) -> Polygon:
    """Dark pixels are the shape (transparent counts as white)."""
    from skimage import measure

    from .image import load, luminance

    dark = luminance(load(path)) < 0.5
    contours = measure.find_contours(np.pad(dark, 1).astype(float), 0.5)
    if not contours:
        raise ValueError(f"{path.name}: no dark shape found")
    polys = [Polygon(np.column_stack([c[:, 1], -c[:, 0]])) for c in contours if len(c) >= 4]
    return _largest(unary_union([shapely.make_valid(p) for p in polys]))


def unit(mask: str) -> Polygon:
    """The outline in its own units: a name from NAMES or a path to an SVG or PNG file."""
    if mask == "heart":
        return _heart()
    if mask in ("oval", "circle"):
        return _ellipse()
    path = Path(mask)
    if not path.is_file():
        raise ValueError(f"unknown mask {mask!r}: use {', '.join(NAMES)} or an SVG/PNG file")
    if path.suffix.lower() == ".svg":
        return _svg(path)
    return _png(path)


def aspect(poly: Polygon) -> float:
    x0, y0, x1, y1 = poly.bounds
    return (x1 - x0) / (y1 - y0)


def fit(poly: Polygon, width: float, height: float) -> Polygon:
    """Scale the outline to fill a width x height box (x centred on 0, y from 0)."""
    x0, y0, x1, y1 = poly.bounds
    sx, sy = width / (x1 - x0), height / (y1 - y0)

    def coords(pts):
        return [((x - x0) * sx - width / 2, (y - y0) * sy) for x, y in pts]

    out = Polygon(coords(poly.exterior.coords), [coords(r.coords) for r in poly.interiors])
    return shapely.make_valid(out) if not out.is_valid else out


def size(
    mask: str,
    poly: Polygon,
    width: float | None,
    height: float | None,
    photo_aspect: float,
    default: float = 100.0,
) -> tuple[float, float]:
    """Outer width and height: a circle stays round, an oval takes the photo's aspect,
    the others keep their own."""
    if mask == "circle":
        d = min(v for v in (width, height) if v) if (width or height) else default
        return d, d
    natural = photo_aspect if mask == "oval" else aspect(poly)
    if width and height:
        return width, height
    if height:
        return height * natural, height
    w = width or default
    return w, w / natural


def cells_inside(poly: Polygon, width: float, height: float, rows: int, cols: int) -> np.ndarray:
    """Grid cells whose centre lies inside the polygon ((rows-1, cols-1), row 0 at y = 0)."""
    u = np.linspace(-width / 2, width / 2, cols)
    v = np.linspace(0, height, rows)
    cu, cv = np.meshgrid((u[:-1] + u[1:]) / 2, (v[:-1] + v[1:]) / 2)
    return shapely.contains_xy(poly, cu, cv)


def points_inside(poly: Polygon, width: float, height: float, rows: int, cols: int) -> np.ndarray:
    """Grid points inside the polygon ((rows, cols), row 0 at y = 0)."""
    u, v = np.meshgrid(np.linspace(-width / 2, width / 2, cols), np.linspace(0, height, rows))
    return shapely.contains_xy(poly, u, v)
