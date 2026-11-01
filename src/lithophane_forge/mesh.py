"""Thickness map -> closed triangle mesh, all in numpy.

A lithophane is a grid of thicknesses. We sample it on a grid of points (row i along the
panel height v, column j along the panel width u), then map every point twice to 3D:
once on the flat face (thickness 0) and once on the relief face (thickness t). The two
faces and the walls between their edges make one closed shell.

The flat face only needs the points on its edge: inside, it is one plane (or one
cylinder) and long strips are enough. That keeps the triangle count close to two per
grid point.

Frames of the panel (u, v, w): u runs left to right along the width, v bottom to top,
w through the thickness (0 on the flat face). See `standing`, `lying` and `arc` for the
mappings to printer coordinates (x, y on the bed, z up).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray  # (n, 3) float
    faces: np.ndarray  # (m, 3) int

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.vertices.min(0), self.vertices.max(0)])

    @property
    def size(self) -> np.ndarray:
        lo, hi = self.bounds
        return hi - lo

    def moved(self, offset) -> "Mesh":
        return Mesh(self.vertices + np.asarray(offset, float), self.faces)

    def on_bed(self, centre=(0.0, 0.0)) -> "Mesh":
        """Move it so that it sits on z = 0 with its box centred on `centre` (x, y)."""
        lo, hi = self.bounds
        c = (lo + hi) / 2
        return self.moved((centre[0] - c[0], centre[1] - c[1], -lo[2]))

    def volume(self) -> float:
        v = self.vertices[self.faces]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6)

    def flipped(self) -> "Mesh":
        return Mesh(self.vertices, self.faces[:, ::-1].copy())

    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]


def merge(meshes: list[Mesh]) -> Mesh:
    """All the meshes in one (separate shells, no boolean)."""
    verts, faces, n = [], [], 0
    for m in meshes:
        verts.append(m.vertices)
        faces.append(m.faces + n)
        n += len(m.vertices)
    return Mesh(np.concatenate(verts), np.concatenate(faces))


def _cells(idx: np.ndarray, wrap: bool) -> np.ndarray:
    """Two triangles per grid cell of an (nr, nc) index array, counterclockwise in (u, v)."""
    if wrap:
        idx = np.concatenate([idx, idx[:, :1]], axis=1)
    a, b = idx[:-1, :-1], idx[:-1, 1:]
    c, d = idx[1:, 1:], idx[1:, :-1]
    return np.concatenate(
        [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    )


def _quads(p0, p1, q1, q0) -> np.ndarray:
    """Triangles of the quads (p0[k], p1[k], q1[k], q0[k]), counterclockwise as given."""
    return np.concatenate([np.stack([p0, p1, q1], -1), np.stack([p0, q1, q0], -1)])


def shell(front: np.ndarray, back: np.ndarray, wrap: bool = False) -> Mesh:
    """Closed mesh between two (nr, nc, 3) point grids: front = relief, back = flat face.

    With wrap=True the grid closes on itself along u (a cylinder): there are no left and
    right walls. The result faces outwards (positive volume)."""
    nr, nc = front.shape[:2]
    if nr < 2 or nc < 4:
        raise ValueError(f"grid too small: {nr} x {nc} points")
    fidx = np.arange(nr * nc).reshape(nr, nc)
    # back points: only the edge of the grid (rows 0 and nr-1, and columns 0, nc-1 unless wrap)
    used = np.zeros((nr, nc), bool)
    used[[0, -1], :] = True
    if not wrap:
        used[:, [0, -1]] = True
    bidx = np.full((nr, nc), -1)
    bidx[used] = nr * nc + np.arange(used.sum())
    verts = np.concatenate([front.reshape(-1, 3), back[used]])

    tris = [_cells(fidx, wrap)]
    cols = np.arange(nc)
    nxt = (cols + 1) % nc if wrap else cols[1:]
    cur = cols if wrap else cols[:-1]
    # bottom wall (row 0) and top wall (row nr-1); the top one is reversed
    f0, b0 = fidx[0], bidx[0]
    tris.append(_quads(b0[cur], b0[nxt], f0[nxt], f0[cur]))
    f1, b1 = fidx[-1], bidx[-1]
    tris.append(_quads(b1[cur], b1[nxt], f1[nxt], f1[cur])[:, ::-1])
    if wrap:
        # back face: one quad strip per column between the bottom and top rows (reversed)
        tris.append(_quads(b0[cur], b0[nxt], b1[nxt], b1[cur])[:, ::-1])
    else:
        rows = np.arange(nr - 1)
        left_f, left_b = fidx[:, 0], bidx[:, 0]
        right_f, right_b = fidx[:, -1], bidx[:, -1]
        # left wall (column 0) and right wall (column nc-1, reversed)
        tris.append(_quads(left_b[rows + 1], left_b[rows], left_f[rows], left_f[rows + 1]))
        tris.append(
            _quads(right_b[rows + 1], right_b[rows], right_f[rows], right_f[rows + 1])[:, ::-1]
        )
        # back face: inner strips are quads between the bottom and top rows; the first and
        # last strips fan out to every point of the left and right columns
        inner = np.arange(1, nc - 2)
        tris.append(_quads(b0[inner], b0[inner + 1], b1[inner + 1], b1[inner])[:, ::-1])
        r0 = b0[1]
        fan = [
            np.stack([np.full(nr - 1, r0), left_b[rows + 1], left_b[rows]], -1),
            [[r0, b1[1], b1[0]]],
        ]
        tris.append(np.concatenate(fan)[:, ::-1])
        l0 = b0[-2]
        fan = [
            np.stack([np.full(nr - 1, l0), right_b[rows], right_b[rows + 1]], -1),
            [[l0, b1[-1], b1[-2]]],
        ]
        tris.append(np.concatenate(fan)[:, ::-1])
    mesh = Mesh(verts, np.concatenate(tris).astype(np.int64))
    return mesh if mesh.volume() > 0 else mesh.flipped()


def _grid(t: np.ndarray, width: float, height: float, wrap: bool = False):
    """u and v of every grid point: (nr, nc) arrays; row 0 is the bottom of the image."""
    nr, nc = t.shape
    u = (np.arange(nc) * width / nc if wrap else np.linspace(0, width, nc)) - width / 2
    v = np.linspace(0, height, nr)
    return np.meshgrid(u, v)


def standing(t: np.ndarray, width: float, height: float) -> Mesh:
    """Flat panel printed standing: u -> x, v -> z, thickness w -> +y (flat face at y = 0).

    `t` is the thickness map with row 0 at the bottom. Seen from the flat face (from -y),
    column 0 is on the left."""
    u, v = _grid(t, width, height)
    front = np.stack([u, t, v], -1)
    back = np.stack([u, np.zeros_like(t), v], -1)
    return shell(front, back)


def lying(t: np.ndarray, width: float, height: float) -> Mesh:
    """Flat panel printed lying on the bed: u -> x, v -> y, thickness w -> z.

    Seen from above (the relief side), column 0 is on the left; seen from below (the bed
    face), it is on the right."""
    u, v = _grid(t, width, height)
    front = np.stack([u, v, t], -1)
    back = np.stack([u, v, np.zeros_like(t)], -1)
    return shell(front, back)


def arc(t: np.ndarray, width: float, height: float, radius: float, wrap: bool = False) -> Mesh:
    """Panel bent around the z axis, printed standing: the flat face on the outside.

    The flat face lies on `radius`; the relief goes inwards to radius - t. The width is the
    arc length on the flat face, so the picture keeps its proportions there. The middle of
    the panel faces -y; seen from outside (from -y), column 0 is on the left. With
    wrap=True the panel closes into a cylinder (width = 2 pi radius)."""
    if np.max(t) >= radius:
        raise ValueError(f"the panel is thicker ({np.max(t):.1f} mm) than its radius")
    u, v = _grid(t, width, height, wrap)
    phi = u / radius
    s, c = np.sin(phi), -np.cos(phi)
    r = radius - t
    front = np.stack([r * s, r * c, v], -1)
    back = np.stack([radius * s, radius * c, v], -1)
    return shell(front, back, wrap)


def write_stl(mesh: Mesh, path: Path, name: str = "lithophane-forge") -> None:
    """Binary STL."""
    tri = mesh.triangles().astype(np.float32)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    record = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    record["n"], record["v"] = normal, tri
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b"\0"))
        f.write(np.uint32(len(tri)).tobytes())
        f.write(record.tobytes())
