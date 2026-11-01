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

SHAPES = ("flat", "curved")
VIEWS = ("flat", "relief")
DEFAULT_WIDTH = 100.0
MAX_ARC = 300.0  # degrees: a curved panel bent further would almost close (and topple)
BRIM = 8.0  # mm of brim for a panel printed standing
BED_MARGIN = 2.0  # keep this far from the bed edge (plus the brim)


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
    frame: float = 3.0  # frame width in mm (0: no frame)
    frame_t: float | None = None  # frame thickness; default t_max + 0.8
    pitch: float = 0.2  # mm between grid points
    max_triangles: int = 1_000_000
    view: str = "flat"  # the face that looks at the viewer
    max_overhang: float = 60.0  # degrees from vertical on the relief side (90: no limit)
    crop: tuple[int, int, int, int] | None = None
    tone: Tone = field(default_factory=Tone)
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


@dataclass
class Result:
    files: dict[str, Path] = field(default_factory=dict)
    grid: tuple[int, int] = (0, 0)
    pitch: float = 0.0
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)  # on the bed: x, y, z
    triangles: int = 0
    volume_cm3: float = 0.0
    notes: list[str] = field(default_factory=list)
    sliced: dict = field(default_factory=dict)


def outer_size(job: Job, photo_aspect: float) -> tuple[float, float]:
    """Width and height of the panel: what the user gave, the rest from the photo."""
    f2 = 2 * job.frame
    if job.width and job.height:
        w, h = job.width, job.height
    elif job.height:
        h = job.height
        w = (h - f2) * photo_aspect + f2
    else:
        w = job.width or DEFAULT_WIDTH
        h = (w - f2) / photo_aspect + f2
    if w <= f2 + 5 or h <= f2 + 5:
        raise ValueError(
            f"the panel ({w:.0f} x {h:.0f} mm) is too small for a {job.frame:g} mm frame"
        )
    return w, h


def grid(width: float, height: float, pitch: float, max_triangles: int) -> tuple[int, int, float]:
    """Rows, columns and the pitch actually used (coarser if the mesh would be too big)."""
    tris = 2 * (width / pitch + 1) * (height / pitch + 1)
    if tris > max_triangles:
        pitch *= math.sqrt(tris / max_triangles)
    return round(height / pitch) + 1, round(width / pitch) + 1, pitch


def thickness_map(
    job: Job, rgb: np.ndarray, width: float, height: float, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray]:
    """The thickness at every grid point as the viewer sees it (row 0 at the top), and the
    framed photo it comes from."""
    u = np.linspace(0, width, cols)
    v = np.linspace(0, height, rows)
    f = job.frame
    inside_u = (u >= f - 1e-9) & (u <= width - f + 1e-9)
    inside_v = (v >= f - 1e-9) & (v <= height - f + 1e-9)
    ci, ri = int(inside_u.sum()), int(inside_v.sum())
    photo = image.frame(rgb, (width - 2 * f) / (height - 2 * f), job.crop)
    light = image.light_map(image.resize(photo, ri, ci), job.tone)
    t = np.full((rows, cols), job.frame_thickness if f > 0 else job.optics.t_max)
    r0, c0 = int(np.argmax(inside_v)), int(np.argmax(inside_u))
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


def build(job: Job, t_view: np.ndarray, width: float, height: float) -> mesh.Mesh:
    """The printable mesh, placed in the middle of the bed."""
    return place(panel(job, t_view, width, height), job.setup.bed)


def panel(job: Job, t_view: np.ndarray, width: float, height: float) -> mesh.Mesh:
    """The mesh in its own frame: the curved panel bends around the z axis."""
    t = np.flipud(t_view)  # the mesh grid has row 0 at the bottom
    if job.view == "relief":
        t = np.fliplr(t)  # seen from the relief side, left and right swap
    if job.shape == "flat":
        return mesh.standing(t, width, height)
    if job.shape == "curved":
        if not 10 <= job.arc <= MAX_ARC:
            raise ValueError(f"the arc must be 10 to {MAX_ARC:g} degrees (got {job.arc:g})")
        return mesh.arc(t, width, height, radius=width / math.radians(job.arc))
    raise ValueError(f"unknown shape {job.shape!r}; choose from {', '.join(SHAPES)}")


def place(m: mesh.Mesh, bed: tuple[float, float, float]) -> mesh.Mesh:
    """Centre it on the bed, with room for the brim."""
    sx, sy, sz = m.size
    room = (bed[0] - 2 * (BRIM + BED_MARGIN), bed[1] - 2 * (BRIM + BED_MARGIN))
    if sz > bed[2]:
        raise ValueError(
            f"the panel is {sz:.0f} mm tall, but the printer prints at most "
            f"{bed[2]:.0f} mm: make it smaller"
        )
    if sx > room[0] or sy > room[1]:
        raise ValueError(
            f"the panel ({sx:.0f} x {sy:.0f} mm on the bed, plus the brim) does "
            f"not fit the {bed[0]:.0f} x {bed[1]:.0f} mm bed: make it smaller"
        )
    return m.on_bed((bed[0] / 2, bed[1] / 2))


def object_settings(job: Job, line_width: float = 0.45) -> dict[str, str]:
    """Per-object PrusaSlicer settings for a lithophane printed standing.

    Enough perimeters to fill the thickest part from both sides, so the light crosses
    whole walls and no infill pattern shows; 100 % infill for the rest; the seam on the
    relief side; a brim against wobbling."""
    perimeters = math.ceil(job.frame_thickness / (2 * line_width)) + 1
    return {
        "perimeters": str(perimeters),
        "fill_density": "100%",
        "fill_pattern": "rectilinear",
        "seam_position": "rear",
        "brim_type": "outer_only",
        "brim_width": f"{BRIM:g}",
    }


def make(job: Job) -> Result:
    res = Result()
    rgb = image.load(job.photo)
    cropped = image.crop(rgb, job.crop)  # the panel takes the aspect of the user's crop
    width, height = outer_size(job, cropped.shape[1] / cropped.shape[0])
    rows, cols, pitch = grid(width, height, job.pitch, job.max_triangles)
    res.grid, res.pitch = (rows, cols), pitch
    if pitch > job.pitch + 1e-9:
        res.notes.append(
            f"grid made coarser to {pitch:.2f} mm (at most {job.max_triangles:,} "
            "triangles); raise --max-triangles for more detail"
        )
    t_view, photo = thickness_map(job, rgb, width, height, rows, cols)
    t_view = limit_overhangs(t_view, job.max_overhang, height / (rows - 1))
    res.notes += image.analyse(photo, (rows, cols)).notes
    m = build(job, t_view, width, height)
    res.size = tuple(float(s) for s in m.size)
    res.triangles = len(m.faces)
    res.volume_cm3 = m.volume() / 1000

    folder = job.folder
    folder.mkdir(parents=True, exist_ok=True)
    lit = render.backlit(t_view, job.optics)
    res.files["preview"] = render.sheet(
        [
            ("photo", render._u8(photo)),
            ("lit from behind (simulated)", lit),
            relief_panel(job, t_view, pitch),
        ],
        folder / "preview.png",
    )
    res.files["stl"] = folder / f"{job.stem}.stl"
    mesh.write_stl(m, res.files["stl"])
    project = folder / f"{job.stem}.3mf"
    part = threemf.Part(job.stem, [threemf.Volume("lithophane", m)], object_settings(job))
    threemf.write(project, [part], thumbnail=render.thumbnail(lit))
    res.files["3mf"] = project

    cmd = slicer.command()
    if cmd is None:
        res.notes.append(
            "PrusaSlicer not found: the 3MF has the model and its settings but "
            "not the printer presets; pick them in PrusaSlicer"
        )
    else:
        res.sliced = slice_project(job, cmd, project, res)
    write_plan(job, res, width, height)
    return res


def relief_panel(job: Job, t_view: np.ndarray, pitch: float):
    """The relief in raking light, as you see it: from behind when the flat face is the front."""
    if job.view == "flat":
        return "relief side (from behind)", render.relief(np.fliplr(t_view), pitch)
    return "relief side (front)", render.relief(t_view, pitch)


def slice_project(job: Job, cmd: list[str], project: Path, res: Result) -> dict:
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
        slicer.check_presets(info, setup)
        threemf.add_config(project, slicer.config_from_gcode(gcode, setup.binary_gcode))
        if job.slice:
            ext = "bgcode" if setup.binary_gcode else "gcode"
            out = job.folder / f"{job.stem}.{ext}"
            slicer.run(cmd, project, out, binary=setup.binary_gcode, extra=tuple(dd))
            res.files[ext] = out
    return info


def write_plan(job: Job, res: Result, width: float, height: float) -> None:
    """plan.json (everything, for tools and tests) and report.md (for people)."""
    s = job.setup
    plan = {
        "photo": str(job.photo),
        "mode": "mono",
        "shape": job.shape,
        "outer_mm": [round(width, 2), round(height, 2)],
        "arc_deg": job.arc if job.shape == "curved" else None,
        "frame_mm": job.frame,
        "frame_thickness_mm": job.frame_thickness,
        "view": job.view,
        "max_overhang_deg": job.max_overhang,
        "grid": list(res.grid),
        "pitch_mm": round(res.pitch, 4),
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
        "colour_changes": [],
        "sliced": res.sliced,
        "files": {k: p.name for k, p in res.files.items()},
        "notes": res.notes,
    }
    folder = job.folder
    (folder / "plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    res.files["plan"] = folder / "plan.json"
    res.files["report"] = folder / "report.md"
    res.files["report"].write_text(report(job, res, width, height), encoding="utf-8")


def report(job: Job, res: Result, width: float, height: float) -> str:
    s, o = job.setup, job.optics
    lines = [
        f"# {job.stem}",
        "",
        (
            f"Lithophane from `{job.photo.name}`: {job.shape}, {width:.1f} x {height:.1f} mm "
            f"(frame {job.frame:g} mm), thickness {o.t_min:g} to {o.t_max:g} mm."
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
            "object has its own settings (100 % infill, perimeters, brim, seam on the relief "
            "side)."
        ),
        "- The panel prints standing, with a brim. Remove the brim with care.",
    ]
    if s.material == "petg":
        lines.append(
            "- PETG: use the textured sheet, or a release agent on the smooth sheet. "
            "Dry the filament if it strings."
        )
    if res.sliced:
        lines.append(f"- PrusaSlicer estimate: {res.sliced['time']}, {res.sliced['filament_g']} g.")
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
