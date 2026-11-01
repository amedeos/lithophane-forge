"""A calibration print: steps of known thickness, to measure your white filament.

The strip prints standing, with the same settings as a lithophane, so the light goes
through the same structure of walls and infill. Under every step its thickness is
written in thin plastic inside a darker, 2 mm label band: lit from behind, the numbers glow.

Hold the strip against the light you will use and read:
- t_max: the first step that looks black (thicker steps look the same);
- td: about 0.55 x that thickness (the step that lets about 10 % of the light through);
- t_min: the thinnest step that is still even and strong (0.8 mm is a safe choice).
Then use them: lithophane-forge make photo.jpg --t-min ... --t-max ... --td ...
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import forge, mesh, render, slicer, threemf
from .forge import Job, Layout, Result

STEPS = (0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0, 3.6, 4.2, 5.0)
STEP_W = 10.0  # mm per step
STEP_H = 20.0  # mm of plain plastic per step
LABEL_H = 8.0  # mm of label band under the steps
FRAME = 3.0
FRAME_T = round(STEPS[-1] + 0.8, 2)
# the label band: 0.8 mm glyphs in it let about 3x more light through, and its pockets are
# shallow enough to print without support (PrusaSlicer's support probe in test_slicer.py)
BAND_T = 2.0


def labels(texts: list[str], cols: int, rows: int) -> np.ndarray:
    """A (rows, cols) mask of the texts, centred in equal cells side by side."""
    im = Image.new("L", (cols, rows), 0)
    draw = ImageDraw.Draw(im)
    cell = cols / len(texts)
    size = int(rows * 0.7)
    font = ImageFont.load_default(size=size)
    while size > 6 and max(draw.textlength(t, font=font) for t in texts) > cell * 0.9:
        size -= 1
        font = ImageFont.load_default(size=size)
    for k, text in enumerate(texts):
        draw.text(((k + 0.5) * cell, rows / 2), text, fill=255, font=font, anchor="mm")
    return np.asarray(im) > 127


def wedge(job: Job) -> tuple[np.ndarray, Layout]:
    """The thickness map (as the viewer sees it) and its layout, ready to print standing.

    The overhang limit comes before the labels: every glyph is a narrow pocket in the label
    band, and the limit would fill it. The band is only BAND_T thick, so the pockets are
    shallow: their tops are short bridges and the counters (inside an 8 or a 4) print
    without support."""
    width = len(STEPS) * STEP_W + 2 * FRAME
    height = STEP_H + LABEL_H + 2 * FRAME
    rows, cols, pitch = forge.grid(width, height, job.pitch, job.max_triangles)
    u = np.linspace(0, width, cols)
    v = np.linspace(height, 0, rows)  # row 0 at the top
    frame_t = job.frame_thickness
    t = np.full((rows, cols), frame_t)
    step = np.floor((u - FRAME) / STEP_W).astype(int)
    inside_u = (u >= FRAME) & (u <= width - FRAME)
    steps = (v <= height - FRAME) & (v >= FRAME + LABEL_H)
    for k, thick in enumerate(STEPS):
        t[np.ix_(steps, inside_u & (step == k))] = thick
    label_area = (v < FRAME + LABEL_H) & (v >= FRAME)
    t[np.ix_(label_area, inside_u)] = BAND_T
    t = forge.limit_overhangs(t, job.max_overhang, height / (rows - 1))
    band = (v < FRAME + LABEL_H - 1) & (v > FRAME + 1)
    ri, ci = int(band.sum()), int(inside_u.sum())
    mask = labels([f"{s:g}" for s in STEPS], ci, ri)
    r0, c0 = int(np.argmax(band)), int(np.argmax(inside_u))
    sub = t[r0 : r0 + ri, c0 : c0 + ci]
    sub[mask] = max(job.optics.t_min, 0.8)  # thin text glows in the thick band
    box = (FRAME, width - FRAME, FRAME + LABEL_H, height - FRAME)
    return t, Layout(width, height, rows, cols, pitch, box)


def make(job: Job) -> Result:
    """Write the calibration strip: STL, 3MF project (and G-code with --slice), preview."""
    job = replace(
        job,
        frame_t=job.frame_t or FRAME_T,
        shape="flat",
        mode="mono",
        hole=False,
        stand=False,
    )
    t, lay = wedge(job)
    res = Result(layout=lay)
    parts = forge.place(forge.volumes(job, lay, t), job.setup.bed, forge.BRIM)
    whole = mesh.merge([m for _, m in parts])
    res.size = tuple(float(s) for s in whole.size)
    res.triangles = len(whole.faces)
    res.volume_cm3 = whole.volume() / 1000
    folder = job.folder
    folder.mkdir(parents=True, exist_ok=True)
    lit = render.backlit(t, job.optics)
    res.files["preview"] = render.sheet(
        [("lit from behind (simulated, default td)", lit), forge.relief_panel(job, t, lay.pitch)],
        folder / "preview.png",
    )
    res.files["stl"] = folder / f"{job.stem}.stl"
    mesh.write_stl(whole, res.files["stl"])
    project = folder / f"{job.stem}.3mf"
    threemf.write(
        project,
        [
            threemf.Part(
                job.stem, [threemf.Volume(n, m) for n, m in parts], forge.object_settings(job)
            )
        ],
        thumbnail=render.thumbnail(lit),
    )
    res.files["3mf"] = project
    cmd = slicer.command()
    if cmd is None:
        res.notes.append(
            "PrusaSlicer not found: the 3MF has the model and its settings but "
            "not the printer presets"
        )
    else:
        res.sliced["calibration"] = forge.slice_project(
            job, cmd, project, res, "lithophane", forge.first_colour(job, res)
        )
    res.files["report"] = folder / "report.md"
    res.files["report"].write_text(report(job, res), encoding="utf-8")
    return res


def report(job: Job, res: Result) -> str:
    s = job.setup
    steps = ", ".join(f"{v:g}" for v in STEPS)
    lines = [
        f"# {job.stem}",
        "",
        f"Calibration strip for white {s.material.upper()}: steps of {steps} mm.",
        "",
        "## Print",
        "",
        (
            f"- Printer: {s.printer.title}, presets `{s.printer_preset}`, `{s.print_preset}`, "
            f"`{s.filament_preset}`."
        ),
        (
            "- Use the spool you will print lithophanes with. The strip prints standing, with the "
            "same settings as a lithophane."
        ),
    ]
    for info in res.sliced.values():
        lines.append(f"- PrusaSlicer estimate: {info['time']}, {info['filament_g']} g.")
    lines += [
        "",
        "## Measure",
        "",
        (
            "Hold the strip against the light you will use (a diffuse, neutral white LED) and "
            "read the numbers that glow under the steps:"
        ),
        "",
        "- **t_max**: the first step that looks black; thicker steps look the same.",
        "- **td**: about 0.55 x t_max (the thickness that lets about 10 % of the light through).",
        "- **t_min**: the thinnest step that is still even and strong (0.8 mm is safe).",
        "",
        "Then: `lithophane-forge make photo.jpg --t-min T_MIN --t-max T_MAX --td TD`.",
        "",
    ]
    return "\n".join(lines)


def default_job(setup, optics, out: Path, pitch: float = 0.2, slice_: bool = False) -> Job:
    return Job(
        photo=Path("calibration"),
        setup=setup,
        optics=optics,
        out=out,
        name=f"calibration-{setup.material}",
        frame_t=FRAME_T,
        pitch=pitch,
        slice=slice_,
    )
