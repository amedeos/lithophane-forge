"""From a photo to the files you print: the whole pipeline of `lithophane-forge make`."""

import json
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import image, mesh, profiles, render, slicer, threemf
from .image import Tone
from .optics import Optics
from .printers import Setup

SHAPES = ("flat", "curved", "cylinder", "shaped")
VIEWS = ("flat", "relief")
STANDING = {"flat", "curved", "cylinder"}  # the others print lying on the bed
DEFAULT_WIDTH = 100.0
DEFAULT_DIAMETER = 80.0
CYLINDER_HEIGHT = 1.25  # default cylinder height, as a share of the diameter
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
    view: str = "flat"  # the face that looks at the viewer
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
    def folder(self) -> Path:
        return self.out / self.stem

    @property
    def frame_thickness(self) -> float:
        return self.frame_t if self.frame_t is not None else round(self.optics.t_max + 0.8, 2)

    @property
    def standing(self) -> bool:
        return self.shape in STANDING

    def check(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"unknown shape {self.shape!r}; choose from {', '.join(SHAPES)}")
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
        x0, y0, x1, y1 = outline.buffer(-f).bounds
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


def thickness_map(job: Job, rgb: np.ndarray, lay: Layout) -> tuple[np.ndarray, np.ndarray]:
    """The thickness at every grid point as the viewer sees it (row 0 at the top), and the
    framed photo it comes from."""
    wrap = job.shape == "cylinder"
    u = np.arange(lay.cols) * lay.width / lay.cols if wrap else np.linspace(0, lay.width, lay.cols)
    v = np.linspace(lay.height, 0, lay.rows)  # row 0 at the top
    u0, u1, v0, v1 = lay.photo_box
    in_u = (u >= u0 - 1e-9) & (u <= u1 + 1e-9)
    in_v = (v >= v0 - 1e-9) & (v <= v1 + 1e-9)
    ci, ri = int(in_u.sum()), int(in_v.sum())
    copies = job.repeat if wrap else 1
    per_copy = math.ceil(ci / copies)
    photo = image.frame(rgb, (u1 - u0) / copies / (v1 - v0), job.crop)
    light = image.light_map(image.resize(photo, ri, per_copy), job.tone)
    light = np.tile(light, (1, copies))[:, :ci]
    t = np.full((lay.rows, lay.cols), job.frame_thickness if job.frame > 0 else job.optics.t_max)
    r0, c0 = int(np.argmax(in_v)), int(np.argmax(in_u))
    t[r0 : r0 + ri, c0 : c0 + ci] = job.optics.thickness(light)
    return t, photo


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
    if job.shape == "shaped":
        from . import accessories, shapes

        f = job.frame
        cells = shapes.cells_inside(lay.outline.buffer(-f / 2), w, h, lay.rows, lay.cols)
        parts = [
            ("lithophane", mesh.lying_masked(t, cells, w, h)),
            ("frame", accessories.frame_ring(lay.outline, f, ft)),
        ]
        if job.hole:
            parts.append(("hanging tab", accessories.tab(lay.outline, ft)))
        if job.view == "flat":  # seen from the bed side
            parts = [(n, _mirror_x(m)) for n, m in parts]
        return parts
    if job.view == "relief":
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
    cropped = image.frame(rgb, rgb.shape[1] / rgb.shape[0], job.crop)
    lay = layout(job, cropped.shape[1] / cropped.shape[0])
    res.layout = lay
    if lay.pitch > job.pitch + 1e-9:
        res.notes.append(
            f"grid made coarser to {lay.pitch:.2f} mm (at most "
            f"{job.max_triangles:,} triangles); raise --max-triangles for more "
            "detail"
        )
    t_view, photo = thickness_map(job, rgb, lay)
    if job.standing:
        t_view = limit_overhangs(t_view, job.max_overhang, lay.height / (lay.rows - 1))
    res.notes += image.analyse(photo, (lay.rows, lay.cols)).notes
    if job.shape == "cylinder":
        res.notes += cylinder_notes(job, lay, cropped.shape[1] / cropped.shape[0])
    parts = place(volumes(job, lay, t_view), job.setup.bed, BRIM if job.standing else 0.0)
    whole = mesh.merge([m for _, m in parts])
    res.size = tuple(float(s) for s in whole.size)
    res.triangles = len(whole.faces)
    res.volume_cm3 = parts[0][1].volume() / 1000

    folder = job.folder
    folder.mkdir(parents=True, exist_ok=True)
    mask, t_seen = None, t_view
    if lay.outline is not None:  # show the frame volume as the viewer will see it
        from . import shapes

        dims = (lay.width, lay.height, lay.rows, lay.cols)
        mask = np.flipud(shapes.points_inside(lay.outline, *dims))
        inner = np.flipud(shapes.points_inside(lay.outline.buffer(-job.frame), *dims))
        t_seen = np.where(inner, t_view, job.frame_thickness)
    lit = render.backlit(t_seen, job.optics, mask=mask)
    res.files["preview"] = render.sheet(
        [
            ("photo", render._u8(photo)),
            (
                "lit from behind (simulated)" + (", unrolled" if job.shape == "cylinder" else ""),
                lit,
            ),
            relief_panel(job, t_seen, lay.pitch, mask),
        ],
        folder / "preview.png",
    )
    res.files["stl"] = folder / f"{job.stem}.stl"
    mesh.write_stl(whole, res.files["stl"])
    projects = {}
    project = folder / f"{job.stem}.3mf"
    threemf.write(
        project,
        [threemf.Part(job.stem, [threemf.Volume(n, m) for n, m in parts], object_settings(job))],
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
            "not the printer presets; pick them in PrusaSlicer"
        )
    else:
        for name, path in projects.items():
            res.sliced[name] = slice_project(job, cmd, path, res, name)
    write_plan(job, res)
    return res


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


def relief_panel(job: Job, t_view: np.ndarray, pitch: float, mask=None):
    """The relief in raking light, as you see it: from behind when the flat face is the front."""
    if job.view == "flat":
        flip = None if mask is None else np.fliplr(mask)
        return "relief side (from behind)", render.relief(np.fliplr(t_view), pitch, flip)
    return "relief side (front)", render.relief(t_view, pitch, mask)


def slice_project(job: Job, cmd: list[str], project: Path, res: Result, name: str) -> dict:
    """Slice once to read the configuration into the project; with --slice, once more from
    the project to the G-code you print."""
    setup = job.setup
    with (
        tempfile.TemporaryDirectory() as tmp,
        slicer.datadir(setup, profiles.find_bundle(), cmd) as dd,
    ):
        gcode = Path(tmp) / "probe.gcode"
        slicer.run(cmd, project, gcode, setup.presets(), extra=tuple(dd))
        info = slicer.summary(gcode)
        threemf.add_config(project, slicer.config_from_gcode(gcode, setup.binary_gcode))
        if job.slice:
            ext = "bgcode" if setup.binary_gcode else "gcode"
            out = project.with_suffix(f".{ext}")
            slicer.run(cmd, project, out, binary=setup.binary_gcode, extra=tuple(dd))
            res.files[ext if name == "lithophane" else f"{name} {ext}"] = out
    return info


def write_plan(job: Job, res: Result) -> None:
    """plan.json (everything, for tools and tests) and report.md (for people)."""
    s, lay = job.setup, res.layout
    plan = {
        "photo": str(job.photo),
        "mode": "mono",
        "shape": job.shape,
        "pose": "standing" if job.standing else "lying",
        "outer_mm": [round(lay.width, 2), round(lay.height, 2)],
        "arc_deg": job.arc if job.shape == "curved" else None,
        "diameter_mm": round(2 * lay.radius, 2) if job.shape == "cylinder" else None,
        "repeat": job.repeat if job.shape == "cylinder" else None,
        "mask": job.mask if job.shape == "shaped" else None,
        "frame_mm": job.frame,
        "frame_thickness_mm": job.frame_thickness,
        "view": job.view,
        "max_overhang_deg": job.max_overhang if job.standing else None,
        "grid": [lay.rows, lay.cols],
        "pitch_mm": round(lay.pitch, 4),
        "optics": asdict(job.optics),
        "tone": asdict(job.tone),
        "printer": {
            "key": s.printer.key,
            "printer": s.printer_preset,
            "print": s.print_preset,
            "filament": s.filament_preset,
            "material": s.material,
            "layer_mm": s.layer,
        },
        "object_settings": object_settings(job),
        "bed_size_mm": [round(v, 2) for v in res.size],
        "triangles": res.triangles,
        "volume_cm3": round(res.volume_cm3, 2),
        "accessories": res.accessories,
        "colour_changes": [],
        "sliced": res.sliced,
        "files": {k: p.name for k, p in res.files.items()},
        "notes": res.notes,
    }
    folder = job.folder
    (folder / "plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    res.files["plan"] = folder / "plan.json"
    res.files["report"] = folder / "report.md"
    res.files["report"].write_text(report(job, res), encoding="utf-8")


def report(job: Job, res: Result) -> str:
    s, o, lay = job.setup, job.optics, res.layout
    what = {
        "flat": "flat panel",
        "curved": f"curved panel ({job.arc:g}° arc)",
        "cylinder": f"cylinder, {2 * (lay.radius or 0):.0f} mm across",
        "shaped": f"{Path(job.mask).name if Path(job.mask).suffix else job.mask} panel",
    }
    lines = [
        f"# {job.stem}",
        "",
        (
            f"Lithophane from `{job.photo.name}`: {what[job.shape]}, {lay.width:.1f} x "
            f"{lay.height:.1f} mm (frame {job.frame:g} mm), thickness {o.t_min:g} to "
            f"{o.t_max:g} mm."
        ),
        "",
        "## Print",
        "",
        (
            f"- Printer: {s.printer.title}, presets `{s.printer_preset}`, `{s.print_preset}`, "
            f"`{s.filament_preset}`."
        ),
        (
            f"- Use **white** {s.material.upper()}. Open `{job.stem}.3mf` in PrusaSlicer: the "
            "object has its own settings ("
            + ", ".join(f"{k} {v}" for k, v in object_settings(job).items())
            + ")."
        ),
    ]
    if job.standing:
        lines.append("- It prints standing, with a brim. Remove the brim with care.")
    else:
        lines.append(
            "- It prints lying on the bed, flat face down: that face is the one you "
            "look at, so a smooth or satin sheet gives the cleanest result."
        )
    if s.material == "petg":
        lines.append(
            "- PETG: use the textured sheet, or a release agent on the smooth sheet. "
            "Dry the filament if it strings."
        )
    for name, info in res.sliced.items():
        lines.append(
            f"- PrusaSlicer estimate for the {name}: {info['time']}, {info['filament_g']} g."
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
    lines += [
        "",
        "## Look at it",
        "",
        (
            f"- The {'flat' if job.view == 'flat' else 'relief'} face looks at you; put the "
            "light behind the other face. A diffuse, neutral white LED (about 4000 K) works best."
        ),
        "- `preview.png` simulates the lit panel.",
        "",
    ]
    if res.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in res.notes] + [""]
    return "\n".join(lines)
