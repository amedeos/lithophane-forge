"""From a photo to the files you print: the whole pipeline of `lithophane-forge make`."""

import glob
import json
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import focus, image, mesh, profiles, render, slicer, threemf
from .focus import DEVICES
from .focus import FLOOR as FOCUS_FLOOR
from .image import Tone
from .optics import Optics
from .palette import LIGHTS
from .printers import Setup

SHAPES = ("flat", "curved", "cylinder", "shaped")
MODES = ("mono", "color")
VIEWS = ("flat", "relief")
COLOR_RANGE = (0.6, 3.0)  # mm: colour panels, from the thinnest stack to the frame
MIN_PATCH = 1.0  # mm²: colour panels, the smallest island or hole a layer keeps
MAX_COLORS = 8  # colour panels: filaments
STANDING = {"flat", "curved", "cylinder"}  # the others print lying on the bed
DEFAULT_WIDTH = 100.0
DEFAULT_DIAMETER = 80.0
CYLINDER_HEIGHT = 1.25  # default cylinder height, as a share of the diameter
MAX_ARC = 300.0  # degrees: a curved panel bent further would almost close (and topple)
BRIM = 8.0  # mm of brim for a panel printed standing
BED_MARGIN = 2.0  # keep this far from the bed edge (plus the brim)
MIN_SHAPED_FRAME = 1.5  # a shaped panel's frame hides the grid's stepped edge


@dataclass(frozen=True)
class Job:
    photo: Path
    setup: Setup
    optics: Optics
    out: Path = Path("out")
    name: str | None = None
    shape: str = "flat"
    mode: str = "mono"  # "color": filament changes (M600), printed lying
    colors: int = 4  # colour mode: number of filaments
    filaments: Path | None = None  # colour mode: your own filaments instead of the catalog
    allow_abrasive: bool = False  # colour mode: also suggest abrasive filaments
    allow_unmeasured: bool = False  # room light: also translucent filaments without k and s
    color_range: tuple[float, float] = COLOR_RANGE
    light: str = "back"  # colour mode: "front" for a picture you look at in room light
    min_patch: float = MIN_PATCH  # colour mode: see remove_specks (0: off)
    focus: str | None = None  # colour mode: the subject (a mask, X,Y,W,H, centre or subject)
    focus_floor: float = FOCUS_FLOOR  # the weight of the rest of the picture (focus.py)
    device: str = "auto"  # --focus subject: where the models run (vision.py)
    width: float | None = None  # outer size in mm, frame included
    height: float | None = None
    arc: float = 120.0  # degrees, for shape="curved"
    diameter: float | None = None  # outer diameter, for shape="cylinder"
    repeat: int = 1  # cylinder: copies of the photo around it
    mask: str = "heart"  # shape="shaped": heart, oval, circle, or an SVG/PNG file
    frame: float = 3.0  # frame width in mm (0: no frame)
    frame_t: float | None = None  # frame thickness; default t_max + 0.8
    pitch: float = 0.2  # mm between grid points
    max_triangles: int = 1_000_000
    view: str | None = None  # the face that looks at the viewer; None: from the light (face)
    max_overhang: float = 60.0  # degrees from vertical on the relief side (90: no limit)
    crop: tuple[int, int, int, int] | None = None
    tone: Tone = field(default_factory=Tone)
    hole: bool = False  # hanging tab (flat and shaped)
    stand: bool = False  # stand with a slot (flat)
    lamp_base: bool = False  # base with an LED pocket (cylinder)
    led_d: float = 40.0
    led_h: float = 18.0
    slice: bool = False

    @property
    def stem(self) -> str:
        return self.name or self.photo.stem

    @property
    def face(self) -> str:
        """The face that looks at the viewer: `view`, or when it is None the one the light
        asks for (in room light the top face, for a lithophane the face on the bed). A
        property, not a value set once, so that replace(job, light=...) follows the light."""
        if self.view is not None:
            return self.view
        return "relief" if self.light == "front" else "flat"

    @property
    def front_lit(self) -> bool:
        """A colour picture for room light (check() allows front light only in colour)."""
        return self.light == "front"

    def part_name(self, name: str) -> str:
        """How the output names a part: the panel is a picture when it is front-lit."""
        return "picture" if name == "lithophane" and self.front_lit else name

    @property
    def folder(self) -> Path:
        return self.out / self.stem

    @property
    def frame_thickness(self) -> float:
        if self.frame_t is not None:
            return self.frame_t
        if self.mode == "color":  # the whole stack: the frame shows the top of the path
            return self.color_range[1]
        return round(self.optics.t_max + 0.8, 2)

    @property
    def standing(self) -> bool:
        return self.shape in STANDING and self.mode == "mono"

    def check(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"unknown shape {self.shape!r}; choose from {', '.join(SHAPES)}")
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; choose from {', '.join(MODES)}")
        if self.light not in LIGHTS:
            raise ValueError(f"unknown light {self.light!r}; choose from {', '.join(LIGHTS)}")
        if self.front_lit:
            if self.mode != "color":
                raise ValueError(
                    "--light front is for colour prints; a white print (--mode mono) is a "
                    "lithophane and needs light from behind"
                )
            if self.face != "relief":
                raise ValueError(
                    "with light from the front you look at the top face (--view relief): "
                    "the flat face lies on the bed and shows only the first filament"
                )
        if self.mode == "color":
            if self.shape not in ("flat", "shaped"):
                raise ValueError("colour panels print lying: use --shape flat or shaped")
            if not 2 <= self.colors <= MAX_COLORS:
                raise ValueError(f"--colors must be between 2 and {MAX_COLORS}")
            lo, hi = self.color_range
            if not 0.3 <= lo < hi:
                raise ValueError(
                    f"colour thickness range {lo:g} to {hi:g} mm is not valid: --t-min must "
                    "be at least 0.3 and smaller than --t-max"
                )
            from .palette import heights

            n = len(heights(lo, hi, self.setup.first_layer, self.setup.layer))
            if n < max(self.colors, 2):
                raise ValueError(
                    f"the colour range {lo:g} to {hi:g} mm holds {n} layer tops of "
                    f"{self.setup.layer:g} mm; {self.colors} colours need at least "
                    f"{max(self.colors, 2)}: widen --t-min/--t-max"
                )
            if self.tone.invert or self.tone.clahe > 0:
                raise ValueError("--invert and --clahe work in mono mode only (--mode mono)")
            if not math.isfinite(self.min_patch) or self.min_patch < 0:
                raise ValueError("--min-patch must be a number, 0 (off) or more")
        if self.focus is not None:
            if self.mode != "color":
                raise ValueError("--focus is for colour prints, not --mode mono")
            focus.parse(self.focus)  # a box that is not X,Y,W,H
            if not focus.MIN_FLOOR <= self.focus_floor <= 1:
                raise ValueError(f"--focus-floor must be between {focus.MIN_FLOOR:g} and 1")
        if self.device not in DEVICES:
            raise ValueError(f"--device must be one of {', '.join(DEVICES)}")
        found = self.focus is not None and focus.parse(self.focus).kind == "subject"
        if self.device != "auto" and not found:
            raise ValueError("--device goes with --focus subject")
        if found:
            from . import vision

            if not vision.available():
                raise ValueError(vision.NEEDS)
        if self.hole and self.shape not in ("flat", "shaped"):
            raise ValueError("--hole works with the flat and shaped panels")
        if self.stand and self.shape != "flat":
            raise ValueError("--stand is for the flat panel (the others stand on their own)")
        if self.lamp_base and self.shape != "cylinder":
            raise ValueError("--lamp-base is for the cylinder")
        if self.shape == "cylinder" and self.width:
            raise ValueError("a cylinder takes --diameter and --height, not --width")
        if self.shape == "shaped" and self.frame < MIN_SHAPED_FRAME:
            raise ValueError(f"a shaped panel needs a frame of at least {MIN_SHAPED_FRAME} mm")
        if self.repeat < 1:
            raise ValueError("--repeat must be at least 1")
        if self.shape == "shaped":
            top = self.color_range[1] if self.mode == "color" else self.optics.t_max
            if self.frame_thickness < top - 1e-9:
                raise ValueError(
                    f"a shaped panel's frame must be at least {top:g} mm thick (--frame-t) "
                    "to hide the stepped edge of the picture"
                )
        if self.shape == "curved" and not 10 <= self.arc <= MAX_ARC:
            raise ValueError(
                f"the arc must be 10 to {MAX_ARC:g} degrees (got {self.arc:g}); "
                "for a full circle use --shape cylinder"
            )


@dataclass
class Layout:
    """Where everything goes, in the panel frame (u from 0 to width, v from 0 to height)."""

    width: float  # along u (a cylinder: its circumference)
    height: float
    rows: int
    cols: int
    pitch: float
    photo_box: tuple[float, float, float, float]  # u0, u1, v0, v1 of the picture
    outline: object = None  # shaped: shapely polygon, u centred on 0
    radius: float | None = None


@dataclass
class Result:
    files: dict[str, Path] = field(default_factory=dict)
    layout: Layout | None = None
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)  # on the bed: x, y, z
    triangles: int = 0
    volume_cm3: float = 0.0
    notes: list[str] = field(default_factory=list)
    sliced: dict[str, dict] = field(default_factory=dict)  # part -> PrusaSlicer summary
    accessories: dict[str, dict] = field(default_factory=dict)
    colour: object = None  # palette.Plan in colour mode
    colour_changes: list[tuple[float, str]] = field(default_factory=list)
    speck_points: int = 0  # colour mode: grid points of the smallest patch a layer keeps
    subject_share: float | None = None  # with --focus: the share of the picture it marks
    found: object = None  # with --focus subject: what found it (vision.Run)


def grid(
    width: float, height: float, pitch: float, max_triangles: int, per_point: float = 2
) -> tuple[int, int, float]:
    """Rows, columns and the pitch actually used (coarser if the mesh would be too big)."""
    tris = per_point * (width / pitch + 1) * (height / pitch + 1)
    if tris > max_triangles:
        pitch *= math.sqrt(tris / max_triangles)
    return round(height / pitch) + 1, round(width / pitch) + 1, pitch


def _from_photo(job: Job, inner_aspect: float, f2: float) -> tuple[float, float]:
    if job.width and job.height:
        return job.width, job.height
    if job.height:
        return (job.height - f2) * inner_aspect + f2, job.height
    w = job.width or DEFAULT_WIDTH
    return w, (w - f2) / inner_aspect + f2


def layout(job: Job, photo_aspect: float) -> Layout:
    """Sizes, grid and the picture's box for the job's shape."""
    f = job.frame
    outline, radius, per_point = None, None, 2
    if job.shape == "cylinder":
        d = job.diameter or DEFAULT_DIAMETER
        radius = d / 2
        width = math.pi * d
        height = job.height or CYLINDER_HEIGHT * d  # a lamp, not a tower: see the crop note
        box = (0.0, width, f, height - f)
    elif job.shape == "shaped":
        from . import shapes

        unit = shapes.unit(job.mask)
        width, height = shapes.size(job.mask, unit, job.width, job.height, photo_aspect)
        outline = shapes.fit(unit, width, height)
        inner = outline.buffer(-f)
        if inner.is_empty:  # its bounds would be NaN
            raise ValueError(
                f"the panel ({width:.0f} x {height:.0f} mm) is too small for a {f:g} mm frame"
            )
        x0, y0, x1, y1 = inner.bounds
        box = (x0 + width / 2, x1 + width / 2, y0, y1)
        per_point = 4  # both faces are full grids
    else:
        width, height = _from_photo(job, photo_aspect, 2 * f)
        box = (f, width - f, f, height - f)
        if job.shape == "curved":
            radius = width / math.radians(job.arc)
    if box[1] - box[0] < 5 or box[3] - box[2] < 5:
        raise ValueError(
            f"the panel ({width:.0f} x {height:.0f} mm) is too small for a {f:g} mm frame"
        )
    rows, cols, pitch = grid(width, height, job.pitch, job.max_triangles, per_point)
    return Layout(width, height, rows, cols, pitch, box, outline, radius)


def picture_cells(job: Job, lay: Layout) -> tuple[np.ndarray, np.ndarray, int]:
    """The grid rows and columns that hold the picture (row 0 at the top), and how many
    copies of the photo go around (a cylinder with --repeat)."""
    wrap = job.shape == "cylinder"
    u = np.arange(lay.cols) * lay.width / lay.cols if wrap else np.linspace(0, lay.width, lay.cols)
    v = np.linspace(lay.height, 0, lay.rows)  # row 0 at the top
    u0, u1, v0, v1 = lay.photo_box
    in_u = (u >= u0 - 1e-9) & (u <= u1 + 1e-9)
    in_v = (v >= v0 - 1e-9) & (v <= v1 + 1e-9)
    return in_v, in_u, job.repeat if wrap else 1


def thickness_map(job: Job, rgb: np.ndarray, lay: Layout) -> tuple[np.ndarray, np.ndarray]:
    """The thickness at every grid point as the viewer sees it (row 0 at the top), and the
    framed photo it comes from."""
    in_v, in_u, copies = picture_cells(job, lay)
    u0, u1, v0, v1 = lay.photo_box
    ci, ri = int(in_u.sum()), int(in_v.sum())
    per_copy = math.ceil(ci / copies)
    photo = image.frame(rgb, (u1 - u0) / copies / (v1 - v0), job.crop)
    light = image.light_map(image.resize(photo, ri, per_copy), job.tone)
    light = np.tile(light, (1, copies))[:, :ci]
    t = np.full((lay.rows, lay.cols), job.frame_thickness if job.frame > 0 else job.optics.t_max)
    r0, c0 = int(np.argmax(in_v)), int(np.argmax(in_u))
    t[r0 : r0 + ri, c0 : c0 + ci] = job.optics.thickness(light)
    return t, photo


@dataclass
class ColourMap:
    plan: object  # palette.Plan
    t_view: np.ndarray  # heights, as the viewer sees them (row 0 at the top)
    ideal: np.ndarray  # simulated print with the ideal colours (linear RGB)
    real: np.ndarray  # simulated print with the chosen filaments
    photo: np.ndarray
    speck_points: int = 0  # remove_specks: islands and holes below this many points go
    notes: list[str] = field(default_factory=list)
    focus: np.ndarray | None = None  # with --focus: the map in photo space (focus.png)
    subject: np.ndarray | None = None  # the picture's grid points that are the subject
    subject_share: float | None = None  # their share of the points you see
    found: object = None  # with --focus subject: what found it (vision.Run)


def colour_map(job: Job, rgb: np.ndarray, lay: Layout) -> ColourMap:
    """Colour mode: the filaments, the change heights and the height of every grid point."""
    from dataclasses import replace

    from scipy.ndimage import median_filter

    from . import palette

    in_v, in_u, _ = picture_cells(job, lay)  # colour panels are never cylinders: one copy
    u0, u1, v0, v1 = lay.photo_box
    ci, ri = int(in_u.sum()), int(in_v.sum())
    size = (rgb.shape[1], rgb.shape[0])
    win = image.window(size, (u1 - u0) / (v1 - v0), job.crop)
    photo = rgb[win[1] : win[3], win[0] : win[2]]
    target = palette.target(image.resize(photo, ri, ci), job.tone.levels, job.tone.gamma)
    box = np.ix_(in_v, in_u)
    seen = visible_points(job, lay)
    values = weight = subject = share = found = None
    notes = []
    if job.focus is not None:
        spec = focus.parse(job.focus)
        values, found = focus.build(spec, rgb, win, job.device)
        # no subject where a shaped panel's frame hides the picture: those points keep the
        # weight of the rest, as they have without --focus
        weight = np.where(seen[box], focus.on_grid(values, size, win, ri, ci), 0.0)
        subject = weight >= focus.SUBJECT
        focus.check_subject(subject, job.focus)
        share = float(subject[seen[box]].mean())
        notes += [n for n in [focus.mask_note(spec, size)] if n]
    points = round(job.min_patch / lay.pitch**2)
    if points >= seen.sum():
        raise ValueError(
            f"--min-patch {job.min_patch:g} mm² is as large as the whole picture "
            f"({seen.sum() * lay.pitch**2:.0f} mm²): use a smaller value"
        )
    lo, hi = job.color_range
    h = palette.heights(lo, hi, job.setup.first_layer, job.setup.layer)
    choices, source = palette.choices(
        job.filaments, job.allow_abrasive, job.light, job.allow_unmeasured
    )
    plan = palette.plan(target, job.colors, h, choices, source, job.light, weight, job.focus_floor)
    frame = int(np.clip(np.searchsorted(h, job.frame_thickness + 1e-9) - 1, 0, len(h) - 1))

    def smoothed(fit):
        """The heights that print (a median filter, then no specks, with the frame around
        the picture), and the fit with the Delta E of those heights where the picture shows."""
        full = np.full((lay.rows, lay.cols), frame)
        full[box] = median_filter(palette.assign(target, palette.lab(fit.colours)), size=3)
        idx = remove_specks(full, points, ~seen, frame)[box]
        inside = seen[box]
        on_subject = None if subject is None else subject[inside]
        return idx, palette.score(fit, target[inside], idx[inside], on_subject)

    real_idx, real = smoothed(plan.real)
    ideal_idx, ideal = smoothed(plan.ideal)
    plan = replace(plan, ideal=ideal, real=real)
    t = np.full((lay.rows, lay.cols), job.frame_thickness)
    t[box] = h[real_idx]

    def sim(fit, idx):
        out = np.broadcast_to(fit.colours[-1], (lay.rows, lay.cols, 3)).copy()
        out[box] = fit.colours[idx]
        return out

    top = np.searchsorted(real.starts, h[real_idx[seen[box]]] - 1e-6, side="right") - 1
    if len(real.filaments) > 1 and not (top == len(real.filaments) - 1).any():
        where = "only on the frame" if job.frame > 0 else "nowhere"
        why = " (--min-patch 0 keeps the specks)" if points > 1 else ""
        notes.append(
            f"once the heights are smoothed, no point of the picture reaches the last "
            f"filament, {real.filaments[-1].name}: it shows {where}{why}"
        )
    sims = sim(ideal, ideal_idx), sim(real, real_idx)
    return ColourMap(plan, t, *sims, photo, points, notes, values, subject, share, found)


def visible_points(job: Job, lay: Layout) -> np.ndarray:
    """The grid points where the viewer sees the picture, not the frame (row 0 at the top)."""
    in_v, in_u, _ = picture_cells(job, lay)
    seen = np.zeros((lay.rows, lay.cols), dtype=bool)
    seen[np.ix_(in_v, in_u)] = True
    if lay.outline is not None:
        seen &= inner_points(job, lay)
    return seen


def inner_points(job: Job, lay: Layout) -> np.ndarray:
    """Shaped panels: the grid points inside the frame (row 0 at the top)."""
    from . import shapes

    dims = (lay.width, lay.height, lay.rows, lay.cols)
    return np.flipud(shapes.points_inside(lay.outline.buffer(-job.frame), *dims))


def remove_specks(
    idx: np.ndarray, cells: int, hidden: np.ndarray | None = None, frame: int = 0
) -> np.ndarray:
    """Colour mode: no layer of the print keeps an island or a hole smaller than `cells`
    grid points. `idx` holds a height (index) per point; a layer is the set of points at
    least that high. An island smaller than about two nozzle widths prints as a blob, and
    in the picture it is a speck of the wrong colour.

    The `hidden` points lie under the frame, at height `frame`: a patch that joins the
    frame is no island, and a hole that runs under the frame counts only its visible part.
    They come back at height `frame`.

    An area opening lowers every small island to the height around it, then an area closing
    raises every small hole. The closing only fills holes, so it joins the filled points to
    the large patch around them and makes no new island: one pass is enough. A thin line
    that joins a large patch stays, and so does a thin ring around one."""
    if hidden is not None:
        idx = np.where(hidden, frame, idx)
    if cells <= 1:
        return idx
    if cells > idx.size:  # scikit-image would return -1 everywhere
        raise ValueError(f"no patch of {cells} points fits in {idx.size} points")
    from skimage.morphology import area_closing, area_opening

    out = area_opening(idx, cells, connectivity=1)
    return area_closing(out, cells, connectivity=1)


def limit_overhangs(t_view: np.ndarray, angle: float, dz: float) -> np.ndarray:
    """Thicken the plastic under thick parts so that the relief side never leans out more
    than `angle` degrees from vertical (the panel prints standing, row 0 at the top).

    A thick part above a thin one is a ledge on the relief side; PrusaSlicer's MK4 presets
    ask for support beyond 50 degrees. Below such a ledge the thickness now falls as a
    ramp: the frame gets a chamfer and dark parts a short soft edge underneath."""
    if angle >= 90:
        return t_view
    step = np.tan(np.radians(angle)) * dz
    ramp = np.arange(t_view.shape[0])[:, None] * step
    limited = np.maximum.accumulate(t_view + ramp, axis=0) - ramp
    return np.where(limited > t_view + 1e-9, limited, t_view)  # no rounding noise elsewhere


def _mirror_x(m: mesh.Mesh) -> mesh.Mesh:
    v = m.vertices.copy()
    v[:, 0] *= -1
    return mesh.Mesh(v, m.faces[:, ::-1].copy())


def volumes(job: Job, lay: Layout, t_view: np.ndarray) -> list[tuple[str, mesh.Mesh]]:
    """The volumes of the panel object, before placement.

    Standing shapes: the flat face is at y = 0 (or on the outer radius) and the viewer of
    the flat face stands at -y. Lying shapes: the flat face is on the bed and is seen from
    below, so left and right swap."""
    t = np.flipud(t_view)  # the mesh grid has row 0 at the bottom
    w, h, ft = lay.width, lay.height, job.frame_thickness
    if not job.standing:  # shaped panels and colour panels lie on the bed
        from shapely.geometry import box

        from . import accessories, shapes

        if job.shape == "shaped":
            f = job.frame
            cells = shapes.cells_inside(lay.outline.buffer(-f / 2), w, h, lay.rows, lay.cols)
            parts = [
                ("lithophane", mesh.lying_masked(t, cells, w, h)),
                ("frame", accessories.frame_ring(lay.outline, f, ft)),
            ]
        else:
            parts = [("lithophane", mesh.lying(t, w, h))]
        if job.hole:
            outline = lay.outline if lay.outline is not None else box(-w / 2, 0, w / 2, h)
            parts.append(("hanging tab", accessories.tab(outline, ft)))
        if job.face == "flat":  # seen from the bed side
            parts = [(n, _mirror_x(m)) for n, m in parts]
        return parts
    if job.face == "relief":
        t = np.fliplr(t)  # seen from the relief side, left and right swap
    if job.shape == "flat":
        parts = [("lithophane", mesh.standing(t, w, h))]
        if job.hole:
            from shapely.geometry import box

            from . import accessories

            tab = accessories.tab(box(-w / 2, 0, w / 2, h), ft)
            parts.append(("hanging tab", accessories.remap(tab, (0, 2, 1))))
        return parts
    if job.shape == "curved":
        return [("lithophane", mesh.arc(t, w, h, radius=lay.radius))]
    return [("lithophane", mesh.arc(t, w, h, radius=lay.radius, wrap=True))]


def panel(job: Job, t_view: np.ndarray, width: float, height: float) -> mesh.Mesh:
    """The lithophane mesh alone, in its own frame (a rectangular panel)."""
    box = (job.frame, width - job.frame, job.frame, height - job.frame)
    rows, cols = t_view.shape
    radius = width / math.radians(job.arc) if job.shape == "curved" else None
    lay = Layout(width, height, rows, cols, width / (cols - 1), box, None, radius)
    return volumes(job, lay, t_view)[0][1]


def place(
    parts: list[tuple[str, mesh.Mesh]], bed: tuple[float, float, float], brim: float
) -> list[tuple[str, mesh.Mesh]]:
    """Centre the parts together on the bed, with room for the brim."""
    allv = np.concatenate([m.vertices for _, m in parts])
    lo, hi = allv.min(0), allv.max(0)
    sx, sy, sz = hi - lo
    room = (bed[0] - 2 * (brim + BED_MARGIN), bed[1] - 2 * (brim + BED_MARGIN))
    if sz > bed[2]:
        raise ValueError(
            f"the panel is {sz:.0f} mm tall, but the printer prints at most "
            f"{bed[2]:.0f} mm: make it smaller"
        )
    if sx > room[0] or sy > room[1]:
        raise ValueError(
            f"the panel ({sx:.0f} x {sy:.0f} mm on the bed{', plus the brim' if brim else ''}) "
            f"does not fit the {bed[0]:.0f} x {bed[1]:.0f} mm bed: make it smaller"
        )
    c = (lo + hi) / 2
    offset = (bed[0] / 2 - c[0], bed[1] / 2 - c[1], -lo[2])
    return [(n, m.moved(offset)) for n, m in parts]


def object_settings(job: Job, line_width: float = 0.45) -> dict[str, str]:
    """Per-object PrusaSlicer settings for the panel.

    Standing: enough perimeters to fill the thickest part from both sides, so the light
    crosses whole walls and no infill pattern shows; 100 % infill for the rest; the seam on
    the relief side; a brim against wobbling. Lying: solid plastic (100 % infill)."""
    solid = {"fill_density": "100%", "fill_pattern": "rectilinear"}
    if not job.standing:
        return solid
    perimeters = math.ceil(job.frame_thickness / (2 * line_width)) + 1
    return {
        "perimeters": str(perimeters),
        **solid,
        "seam_position": "rear",
        "brim_type": "outer_only",
        "brim_width": f"{BRIM:g}",
    }


ACCESSORY_SETTINGS = {"layer_height": "0.2"}  # plain parts: faster layers


def accessory_parts(job: Job, lay: Layout) -> dict[str, tuple[mesh.Mesh, dict]]:
    from . import accessories

    out = {}
    if job.stand:
        out["stand"] = accessories.stand(lay.width, lay.height, job.frame_thickness, job.frame)
    if job.lamp_base:
        out["lamp-base"] = accessories.lamp_base(
            2 * lay.radius, job.frame_thickness, job.frame, job.led_d, job.led_h
        )
    return out


def make(job: Job) -> Result:
    job.check()
    res = Result()
    rgb = image.load(job.photo)
    cropped = image.crop(rgb, job.crop)  # the panel takes the aspect of the user's crop
    lay = layout(job, cropped.shape[1] / cropped.shape[0])
    res.layout = lay
    if lay.pitch > job.pitch + 1e-9:
        res.notes.append(
            f"grid made coarser to {lay.pitch:.2f} mm (at most "
            f"{job.max_triangles:,} triangles); raise --max-triangles for more "
            "detail"
        )
    cmap = None
    if job.mode == "color":
        cmap = colour_map(job, rgb, lay)
        t_view, photo = cmap.t_view, cmap.photo
        res.colour = cmap.plan
        res.colour_changes = cmap.plan.real.colour_changes(job.setup.layer)
        res.speck_points = cmap.speck_points
        res.subject_share = cmap.subject_share
        res.found = cmap.found
        res.notes += colour_notes(job, cmap.plan) + cmap.notes
    else:
        t_view, photo = thickness_map(job, rgb, lay)
    if job.standing:
        t_view = limit_overhangs(t_view, job.max_overhang, lay.height / (lay.rows - 1))
    in_v, in_u, copies = picture_cells(job, lay)  # the grid one copy of the photo fills
    grid = (int(in_v.sum()), math.ceil(in_u.sum() / copies))
    res.notes += image.analyse(photo, grid, colour=job.mode == "color").notes
    if job.shape == "cylinder":
        res.notes += cylinder_notes(job, lay, cropped.shape[1] / cropped.shape[0])
    parts = place(volumes(job, lay, t_view), job.setup.bed, BRIM if job.standing else 0.0)
    whole = mesh.merge([m for _, m in parts])
    res.size = tuple(float(s) for s in whole.size)
    res.triangles = len(whole.faces)
    # every part counts; where they join they overlap a little, so this is slightly more
    # than the plastic (PrusaSlicer's estimate is exact)
    res.volume_cm3 = sum(m.volume() for _, m in parts) / 1000

    folder = job.folder
    folder.mkdir(parents=True, exist_ok=True)
    clear_outputs(folder, job.stem, keep=job.photo)  # a --focus mask here was read already
    if cmap is not None and cmap.focus is not None:
        res.files["focus"] = focus.save(cmap.focus, folder / "focus.png")
    mask, inner, t_seen = None, None, t_view
    if lay.outline is not None:  # show the frame volume as the viewer will see it
        from . import shapes

        dims = (lay.width, lay.height, lay.rows, lay.cols)
        mask = np.flipud(shapes.points_inside(lay.outline, *dims))
        inner = inner_points(job, lay)
        t_seen = np.where(inner, t_view, job.frame_thickness)
    if cmap is None:
        lit = render.backlit(t_seen, job.optics, mask=mask)
        unrolled = ", unrolled" if job.shape == "cylinder" else ""
        panels = [
            ("photo", image.to_pil(photo)),
            ("lit from behind (simulated)" + unrolled, lit),
            relief_panel(job, t_seen, lay.pitch, mask),
        ]
    else:
        lit = render.colour(with_frame(cmap.real, inner, cmap.plan.real), mask)
        lit_by = "room light" if job.front_lit else "lit from behind"
        ideal = with_frame(cmap.ideal, inner, cmap.plan.ideal)
        shown = ("photo", image.to_pil(photo))
        if cmap.subject is not None:
            shown = ("photo, subject outlined", render.outlined(shown[1], cmap.subject))
        panels = [
            shown,
            (f"ideal colours (simulated, {lit_by})", render.colour(ideal, mask)),
            (f"suggested filaments (simulated, {lit_by})", lit),
            ("filament changes", render.palette_strip(cmap.plan, job.setup.layer)),
        ]
    res.files["preview"] = render.sheet(panels, folder / "preview.png")
    res.files["stl"] = folder / f"{job.stem}.stl"
    mesh.write_stl(whole, res.files["stl"])
    projects = {}
    project = folder / f"{job.stem}.3mf"
    threemf.write(
        project,
        [threemf.Part(job.stem, [threemf.Volume(n, m) for n, m in parts], object_settings(job))],
        colour_changes=res.colour_changes,
        thumbnail=render.thumbnail(lit),
    )
    res.files["3mf"] = projects["lithophane"] = project

    for name, (m, info) in accessory_parts(job, lay).items():
        m = place([(name, m)], job.setup.bed, 0.0)[0][1]
        path = folder / f"{job.stem}-{name}.3mf"
        threemf.write(
            path,
            [
                threemf.Part(
                    f"{job.stem} {name}", [threemf.Volume(name, m)], dict(ACCESSORY_SETTINGS)
                )
            ],
        )
        mesh.write_stl(m, folder / f"{job.stem}-{name}.stl")
        res.files[f"{name} 3mf"] = projects[name] = path
        res.files[f"{name} stl"] = folder / f"{job.stem}-{name}.stl"
        res.accessories[name] = info

    cmd = slicer.command()
    if cmd is None:
        res.notes.append(
            "PrusaSlicer not found: the 3MF has the model and its settings but "
            "not the printer presets or the filament colour; pick them in PrusaSlicer"
        )
    else:
        for name, path in projects.items():
            # accessories print in any colour: they keep the preset's
            colour = first_colour(job, res) if name == "lithophane" else None
            res.sliced[name] = slice_project(job, cmd, path, res, name, colour)
    write_plan(job, res)
    return res


PATHS = {"back": "from light to dark", "front": "from the first filament to the last"}
SUBJECT_WORSE = 1.0  # Delta E: a note when the subject is this much further than the picture


def colour_notes(job: Job, plan) -> list[str]:
    notes = []
    if plan.real.mean_de > 15:
        notes.append(
            f"the print will differ from the photo (mean Delta E {plan.real.mean_de:.0f}):"
            " one extruder can only follow one path of colours, "
            + PATHS[job.light]
            + ", and this photo's colours do not lie on one path; see preview.png, or try "
            "--colors with one more filament"
        )
    used = len(plan.real.filaments)
    if plan.available < job.colors:
        notes.append(
            f"the filament list has only {plan.available} filaments for {job.colors} "
            f"colours; the plan uses {used}"
        )
    elif used < job.colors:
        notes.append(
            f"the plan uses {used} of the {job.colors} filaments: the search found no "
            "place for more that brings the print closer to the photo"
        )
    if plan.real.focus_de is not None and plan.real.focus_de > plan.real.mean_de + SUBJECT_WORSE:
        tries = ["a lower --focus-floor"]
        if used == job.colors and job.colors < MAX_COLORS:  # one more may find a place
            tries.insert(0, f"--colors {job.colors + 1}")
        notes.append(
            f"even with --focus the subject is further from the photo (mean Delta E "
            f"{plan.real.focus_de:.1f}) than the whole picture ({plan.real.mean_de:.1f}): its "
            f"colours do not lie on the path of the others; try {' or '.join(tries)}"
        )
    if job.setup.material != "petg" and job.filaments is None:
        notes.append(
            "the suggested filaments come from the Prusament PETG table; pass "
            "--filaments with your own list for another material"
        )
    return notes + base_notes(job, plan)


def base_notes(job: Job, plan) -> list[str]:
    """What limits the first filament. In room light it must hide what is behind the
    panel at the thinnest point; lit from behind it should be light, neutral and
    translucent. The search takes such a filament if the list has one.

    In room light a light filament needs about its TD to hide the back (Signal White
    about 1.8 mm), so a light first filament is rare: the light colours come from light
    filaments over a dark base. No note asks for one."""
    from . import palette

    h0 = float(plan.real.heights[0])
    light = palette.light_model(job.light, plan.real.heights)
    base = plan.real.filaments[0]
    if light.is_base(base):
        return []
    if job.front_lit:
        z = light.hides_from(base)
        fix = (
            f"add an opaque filament to the list, or raise --t-min to {z:g} mm, where it hides it"
            if z is not None
            else "it does not hide it even at the full thickness of the panel: add an opaque "
            "filament to the list"
        )
        return [
            (
                f"no filament in the list hides what is behind the panel at its thinnest point "
                f"({h0:g} mm), so the first filament, {base.name} (TD {base.td:g}), lets the "
                f"wall behind it show through; {fix}"
            )
        ]
    return [
        (
            "no filament in the list is light, neutral and translucent, so the first "
            f"filament, {base.name}, tints every point of the lithophane"
        )
    ]


def cylinder_notes(job: Job, lay: Layout, photo_aspect: float) -> list[str]:
    """Tell how much of the photo a cylinder crops away, and the --repeat that would fit."""
    u0, u1, v0, v1 = lay.photo_box
    copy_aspect = (u1 - u0) / (v1 - v0) / job.repeat
    kept = min(copy_aspect, photo_aspect) / max(copy_aspect, photo_aspect)
    if kept > 0.6:
        return []
    best = max(1, round((u1 - u0) / (v1 - v0) / photo_aspect))
    hint = f"; try --repeat {best}" if best != job.repeat else ""
    note = f"the cylinder keeps only {kept:.0%} of the photo{hint}, or set --height"
    return [note]


def with_frame(sim: np.ndarray, inner: np.ndarray | None, fit) -> np.ndarray:
    """A simulated colour print with the frame of a shaped panel: outside `inner` the frame
    volume covers the picture with the whole stack (the last colour of the path: the
    darkest lit from behind, the top filament in room light)."""
    if inner is None:
        return sim
    return np.where(inner[..., None], sim, fit.colours[-1])


OUTPUTS = [f"{{s}}{part}.{ext}" for part in ("", "-*") for ext in ("stl", "3mf", "gcode", "bgcode")]
OUTPUTS += ["preview.png", "plan.json", "report.md", "focus.png"]


def clear_outputs(folder: Path, stem: str, keep: Path | None = None) -> None:
    """Remove the files an earlier run wrote under this name (only the kinds we write, and
    never `keep`, the photo). Above all its G-code: a run without --slice would leave G-code
    with other filaments and change heights next to the new project and report. Every file
    here is an output: a focus map to edit goes next to the photo (report.md says so)."""
    keep = keep.resolve() if keep else None
    for pattern in OUTPUTS:
        for path in folder.glob(pattern.format(s=glob.escape(stem))):
            if path.is_file() and path.resolve() != keep:
                path.unlink()


def relief_panel(job: Job, t_view: np.ndarray, pitch: float, mask=None):
    """The relief in raking light, as you see it: from behind when the flat face is the front."""
    if job.face == "flat":
        flip = None if mask is None else np.fliplr(mask)
        return "relief side (from behind)", render.relief(np.fliplr(t_view), pitch, flip)
    return "relief side (front)", render.relief(t_view, pitch, mask)


PLAIN_WHITE = "#FFFFFF"  # white filament with no catalog colour (PLA)


def first_colour(job: Job, res: Result) -> str:
    """The colour of the filament the print starts with: the first filament of the colour
    plan; for white prints Signal White for PETG (from the catalog), plain white for PLA.
    The filament presets in the printer table all say orange (#FF8000)."""
    if res.colour is not None:
        return res.colour.real.filaments[0].hex
    if job.setup.material == "petg":
        from .palette import white

        return white().hex
    return PLAIN_WHITE


def slice_project(
    job: Job, cmd: list[str], project: Path, res: Result, name: str, colour: str | None = None
) -> dict:
    """Slice once to read the configuration into the project; with --slice, once more from
    the project to the G-code you print. `colour` replaces the filament preset's colour, so
    the project (and the G-code made from it) shows the filament you load first."""
    setup = job.setup
    with (
        tempfile.TemporaryDirectory() as tmp,
        slicer.datadir(setup, profiles.find_bundle(), cmd) as dd,
    ):
        gcode = Path(tmp) / "probe.gcode"
        own = ("--filament-colour", colour) if colour is not None else ()
        slicer.run(cmd, project, gcode, setup.presets(), extra=(*dd, *own))
        info = slicer.summary(gcode)
        slicer.check_presets(info, setup)
        threemf.add_config(project, slicer.config_from_gcode(gcode, setup.binary_gcode))
        if job.slice:
            ext = "bgcode" if setup.binary_gcode else "gcode"
            out = project.with_suffix(f".{ext}")
            slicer.run(cmd, project, out, binary=setup.binary_gcode, extra=tuple(dd))
            res.files[ext if name == "lithophane" else f"{name} {ext}"] = out
    return info


def colour_plan(res: Result) -> dict | None:
    """The colour mode's decisions, for plan.json."""
    if res.colour is None:
        return None
    p = res.colour

    def measured(x) -> dict:  # a filament's measured k and s, if it has them
        return {"k": list(x.optics[0]), "s": x.optics[1]} if x.optics else {}

    def fit(f):
        return {
            "filaments": [
                {"name": x.name, "hex": x.hex, "td": x.td, **measured(x), "from_mm": round(z, 3)}
                for x, z in zip(f.filaments, f.starts)
            ],
            "mean_delta_e": round(f.mean_de, 2),
            "p95_delta_e": round(f.p95_de, 2),
            "subject_delta_e": None if f.focus_de is None else round(f.focus_de, 2),
        }

    return {
        "ideal": fit(p.ideal),
        "print": fit(p.real),
        "source": p.source,
        "closest": [  # per ideal colour: the closest filament by colour alone
            {"ideal": i.hex, "filament": f.name, "hex": f.hex, "delta_e": round(d, 1)}
            for i, (f, d) in zip(p.ideal.filaments, p.matches)
        ],
    }


def write_plan(job: Job, res: Result) -> None:
    """plan.json (everything, for tools and tests) and report.md (for people)."""
    s, lay = job.setup, res.layout
    plan = {
        "photo": str(job.photo),
        "mode": job.mode,
        "shape": job.shape,
        "pose": "standing" if job.standing else "lying",
        "outer_mm": [round(lay.width, 2), round(lay.height, 2)],
        "arc_deg": job.arc if job.shape == "curved" else None,
        "diameter_mm": round(2 * lay.radius, 2) if job.shape == "cylinder" else None,
        "repeat": job.repeat if job.shape == "cylinder" else None,
        "mask": job.mask if job.shape == "shaped" else None,
        "frame_mm": job.frame,
        "frame_thickness_mm": job.frame_thickness,
        "view": job.face,
        "max_overhang_deg": job.max_overhang if job.standing else None,
        "grid": [lay.rows, lay.cols],
        "pitch_mm": round(lay.pitch, 4),
        "optics": asdict(job.optics) if job.mode == "mono" else None,
        "color_range_mm": list(job.color_range) if job.mode == "color" else None,
        "min_patch_mm2": job.min_patch if job.mode == "color" else None,
        "min_patch_points": res.speck_points if job.mode == "color" else None,
        "focus": focus_plan(job, res),
        "light": job.light,
        "tone": asdict(job.tone),
        "printer": {
            "key": s.printer.key,
            "printer": s.printer_preset,
            "print": s.print_preset,
            "filament": s.filament_preset,
            # the colour the project carries: only PrusaSlicer writes it into the 3MF
            "filament_colour": first_colour(job, res) if "lithophane" in res.sliced else None,
            "material": s.material,
            "layer_mm": s.layer,
        },
        "object_settings": object_settings(job),
        "bed_size_mm": [round(v, 2) for v in res.size],
        "triangles": res.triangles,
        "volume_cm3": round(res.volume_cm3, 2),
        "accessories": res.accessories,
        "colour": colour_plan(res),
        "colour_changes": [[z, c] for z, c in res.colour_changes],
        "sliced": res.sliced,
        "files": {k: p.name for k, p in res.files.items()},
        "notes": res.notes,
    }
    folder = job.folder
    (folder / "plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    res.files["plan"] = folder / "plan.json"
    res.files["report"] = folder / "report.md"
    res.files["report"].write_text(report(job, res), encoding="utf-8")


def focus_plan(job: Job, res: Result) -> dict | None:
    """The --focus part of plan.json."""
    if job.focus is None or res.subject_share is None:
        return None
    return {
        "spec": job.focus,
        "floor": job.focus_floor,
        "map": "focus.png",
        "subject_share": round(res.subject_share, 4),
        "found": None
        if res.found is None
        else {"models": list(res.found.models), "device": res.found.device},
    }


def colour_report(job: Job, res: Result) -> list[str]:
    p = res.colour
    first = p.real.filaments[0]

    def known(f):  # filaments with measured k and s
        return ", measured" if f.optics else ""

    lines = [f"- Load **{first.name}** ({first.hex}{known(first)}) before you start."]
    for (z, _), f in zip(res.colour_changes, p.real.filaments[1:]):
        lines.append(
            f"- At Z {z:.2f} mm the printer stops (M600): load **{f.name}** ({f.hex}{known(f)})."
        )
    lines += [
        (
            f"- With these filaments the print is a mean Delta E {p.real.mean_de:.1f} away from "
            f"the photo (95 % of it within {p.real.p95_de:.0f}); the ideal colours, fitted to "
            f"the photo alone, give {p.ideal.mean_de:.1f}."
        ),
        *focus_lines(job, res),
        speck_line(job, res),
        "",
        "## Ideal colours",
        "",
        (
            "The colours that suit this photo best, whatever filaments exist, and the closest "
            "filament to each. The print above uses the filaments that bring the whole "
            "picture closest to the photo"
            + (", the subject first" if job.focus is not None else "")
            + ", which can differ from the closest colours."
        ),
        "",
        "| # | Ideal colour | Ideal TD (mm) | Closest filament | Delta E |",
        "|---|---|---|---|---|",
    ]
    for k, (i, (f, d)) in enumerate(zip(p.ideal.filaments, p.matches), start=1):
        lines.append(
            f"| {k} | {i.hex} | {i.td:g} | {f.name} ({f.hex}, TD {f.td:g}{known(f)}) | {d:.0f} |"
        )
    hides = (
        " Lit from the front, a thin layer tints what lies under it, like coloured glass, and"
        " a filament needs about its TD to hide it. For a filament marked measured the plan"
        " uses its measured absorption and scattering (k and s); for the others hex and TD"
        " are an estimate."
        if job.front_lit
        else ""
    )
    lines += [
        "",
        (
            f"Filaments from: {p.source}. TD is the thickness that lets about 10 % of the "
            f"light through; a small TD is an opaque filament.{hides}"
        ),
        "",
    ]
    return lines


def focus_lines(job: Job, res: Result) -> list[str]:
    p = res.colour
    if p.real.focus_de is None:
        return []
    ideal = "" if p.ideal.focus_de is None else f" (the ideal colours: {p.ideal.focus_de:.1f})"
    found = []
    if res.found is not None:
        found = [
            (
                f"- The subject was found by {len(res.found.models)} models "
                f"({', '.join(res.found.models)}) on the {res.found.device} in "
                f"{res.found.seconds:.1f} s: the people when the photo has people, else its "
                "main object. Check its outline in `preview.png`."
            )
        ]
    return found + [
        (
            f"- On the subject (`--focus {job.focus}`, {res.subject_share:.0%} of the "
            f"picture) the print is a mean Delta E {p.real.focus_de:.1f} away from the "
            f"photo{ideal}. The colours and the change heights serve the subject first, so "
            "the rest of the picture can be further from the photo."
        ),
        (
            "- `focus.png` shows the focus map (white = the subject). To refine it, copy it "
            f"next to the photo, edit the copy and pass it with `--focus`: every run "
            f"rewrites `{job.folder}`."
        ),
    ]


def speck_line(job: Job, res: Result) -> str:
    if res.speck_points > 1:
        return (
            f"- Islands and holes smaller than {job.min_patch:g} mm² in a layer are merged "
            "into the area around them: they would print as blobs (--min-patch, 0 keeps them)."
        )
    if job.min_patch > 0:
        return (
            f"- --min-patch {job.min_patch:g} mm² is not larger than one grid point "
            f"({res.layout.pitch:.2f} mm apart), so small islands and holes are kept: some "
            "may print as blobs."
        )
    return "- Small islands and holes are kept (--min-patch 0): some may print as blobs."


def report(job: Job, res: Result) -> str:
    s, lay = job.setup, res.layout
    t_min, t_max = job.color_range if job.mode == "color" else (job.optics.t_min, job.optics.t_max)
    what = {
        "flat": "flat panel",
        "curved": f"curved panel ({job.arc:g}° arc)",
        "cylinder": f"cylinder, {2 * (lay.radius or 0):.0f} mm across",
        "shaped": f"{Path(job.mask).name if Path(job.mask).suffix else job.mask} panel",
    }
    kind = "Colour picture" if job.front_lit else "Lithophane"
    lines = [
        f"# {job.stem}",
        "",
        (
            f"{kind} from `{job.photo.name}`: {what[job.shape]}, {lay.width:.1f} x "
            f"{lay.height:.1f} mm (frame {job.frame:g} mm), thickness {t_min:g} to "
            f"{t_max:g} mm."
        ),
        "",
        "## Print",
        "",
        (
            f"- Printer: {s.printer.title}, presets `{s.printer_preset}`, `{s.print_preset}`, "
            f"`{s.filament_preset}`."
        ),
        (
            f"- Open `{job.stem}.3mf` in PrusaSlicer: the object has its own settings ("
            + ", ".join(f"{k} {v}" for k, v in object_settings(job).items())
            + ")."
        ),
    ]
    if job.mode == "color":
        lines += colour_report(job, res)
    else:
        lines.append(f"- Use **white** {s.material.upper()}.")
    if job.standing:
        lines.append("- It prints standing, with a brim. Remove the brim with care.")
    else:
        if job.front_lit:
            lines.append(
                "- It prints lying on the bed, flat face down. You look at the top face, the "
                "last one printed: PrusaSlicer shows it as you will see it, not mirrored."
            )
        elif job.face == "flat":
            lines.append(
                "- It prints lying on the bed, flat face down: that face is the one you "
                "look at, so a smooth or satin sheet gives the cleanest result."
            )
        else:
            lines.append(
                "- It prints lying on the bed, flat face down. The relief side looks at you "
                "(--view relief); the flat face goes toward the light."
            )
    if s.material == "petg":
        lines.append(
            "- PETG: use the textured sheet, or a release agent on the smooth sheet. "
            "Dry the filament if it strings."
        )
    for name, info in res.sliced.items():
        lines.append(
            f"- PrusaSlicer estimate for the {job.part_name(name)}: {info['time']}, "
            f"{info['filament_g']} g."
        )
    if "stand" in res.accessories:
        a = res.accessories["stand"]
        lines.append(
            f"- Stand: `{job.stem}-stand.3mf`, any colour, 0.2 mm layers. The slot is "
            f"{a['slot_mm'][0]} mm wide and {a['slot_mm'][1]} mm deep."
        )
    if "lamp-base" in res.accessories:
        a = res.accessories["lamp-base"]
        lines.append(
            f"- Lamp base: `{job.stem}-lamp-base.3mf`, an opaque colour. The pocket "
            f"takes an LED tealight or puck light up to {a['led_pocket_mm'][0] - 1:g} "
            f"mm across and {a['led_pocket_mm'][1]:g} mm tall; the notch takes a cable."
        )
    if job.front_lit:
        look = [
            (
                "- Look at the top face like a photo, in daylight or under a neutral white "
                "lamp in front of it. It needs no light behind it."
            ),
            "- `preview.png` simulates it in room light.",
        ]
    else:
        look = [
            (
                f"- The {'flat' if job.face == 'flat' else 'relief'} face looks at you; put "
                "the light behind the other face. A diffuse, neutral white LED (about 4000 K) "
                "works best."
            ),
            "- `preview.png` simulates the lit panel.",
        ]
    lines += ["", "## Look at it", "", *look, ""]
    if res.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in res.notes] + [""]
    return "\n".join(lines)
