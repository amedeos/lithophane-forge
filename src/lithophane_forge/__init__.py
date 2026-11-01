"""lithophane-forge: turn a photo into a printable lithophane and a PrusaSlicer project."""

import argparse
import sys
from pathlib import Path

__version__ = "0.1.0"

EXAMPLES = """\
examples:
  lithophane-forge analyze photo.jpg --colors 4
  lithophane-forge make photo.jpg --width 120 --slice
  lithophane-forge make photo.jpg --colors 5 --width 150
  lithophane-forge make photo.jpg --shape shaped --mask heart --width 90 --hole
  lithophane-forge make photo.jpg --mode mono --width 120
  lithophane-forge make photo.jpg --shape curved --height 100 --arc 110
  lithophane-forge make photo.jpg --shape cylinder --diameter 80 --repeat 2 --lamp-base
  lithophane-forge make photo.jpg --printer coreone --slice
  lithophane-forge calibrate --slice
  lithophane-forge printers

make prints in colour, lying on the bed, with filament changes (M600); --mode mono
prints white filament standing. Curved panels and cylinders always print mono.
"""


class UsageError(Exception):
    pass


ABRASIVE_HELP = (
    "also suggest abrasive filaments (Tungsten 75 %%): only with a hardened steel nozzle"
)


def _number(kind=float, low: float | None = None, high: float | None = None):
    def parse(text: str):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if low is not None and value < low:
            raise argparse.ArgumentTypeError(f"must be at least {low:g} (got {text})")
        if high is not None and value > high:
            raise argparse.ArgumentTypeError(f"must be at most {high:g} (got {text})")
        return value

    return parse


def _levels(text: str) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"levels must be LOW,HIGH (got {text!r})") from None
    if not 0 <= lo < hi <= 100:
        raise argparse.ArgumentTypeError("levels must satisfy 0 <= LOW < HIGH <= 100")
    return lo, hi


def parser() -> argparse.ArgumentParser:
    from .forge import MODES, SHAPES, VIEWS
    from .printers import DEFAULT_MATERIAL, DEFAULT_PRINTER, MATERIALS, PRINTERS

    p = argparse.ArgumentParser(
        prog="lithophane-forge",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Turn a photo into a lithophane and a ready-to-print PrusaSlicer project.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    a = sub.add_parser("analyze", help="check a photo and get advice before printing")
    a.add_argument("photo", type=Path)
    a.add_argument(
        "--colors",
        type=_number(int, 2),
        help="also find the ideal colours for a colour print with this many filaments",
    )
    a.add_argument("--filaments", type=Path, help="your filaments (TOML) instead of the catalog")
    a.add_argument("--allow-abrasive", action="store_true", help=ABRASIVE_HELP)

    m = sub.add_parser(
        "make",
        help="make the lithophane files from a photo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    m.add_argument("photo", type=Path)
    size = m.add_argument_group("size and shape")
    size.add_argument("--shape", choices=SHAPES, default="flat")
    size.add_argument(
        "--mode",
        choices=MODES,
        help="color (the default for flat and shaped panels): filament changes (M600), "
        "printed lying; mono: white filament, printed standing (always for curved and "
        "cylinder)",
    )
    size.add_argument(
        "--width",
        type=_number(float, 10),
        help="outer width in mm, frame included (default 100 if no height is given)",
    )
    size.add_argument(
        "--height",
        type=_number(float, 10),
        help="outer height in mm, frame included (default: from the photo)",
    )
    size.add_argument(
        "--arc",
        type=_number(float, 10, 300),
        default=120.0,
        help="curved: degrees of arc (10 to 300)",
    )
    size.add_argument(
        "--diameter", type=_number(float, 20), help="cylinder: outer diameter in mm (default 80)"
    )
    size.add_argument(
        "--repeat", type=_number(int, 1), default=1, help="cylinder: copies of the photo around it"
    )
    size.add_argument(
        "--mask",
        default="heart",
        help="shaped: heart, oval, circle, or an SVG or PNG file (dark = inside)",
    )
    size.add_argument(
        "--frame", type=_number(float, 0), default=3.0, help="frame width in mm (0: none)"
    )
    size.add_argument(
        "--frame-t",
        type=_number(float, 0.5),
        help="frame thickness in mm (default: t_max + 0.8 in mono, the top of the stack in colour)",
    )
    size.add_argument(
        "--view",
        choices=VIEWS,
        default="flat",
        help="the face that looks at the viewer (the light goes behind the other)",
    )
    size.add_argument(
        "--max-overhang",
        type=_number(float, 30),
        default=60.0,
        help="steepest ledge on the relief side, degrees from vertical (90: off)",
    )
    look = m.add_argument_group("picture and thickness")
    look.add_argument("--crop", help="X,Y,W,H in photo pixels, before the automatic crop")
    look.add_argument(
        "--levels",
        type=_levels,
        default=(0.5, 99.5),
        help="percentiles of lightness that become black and white",
    )
    look.add_argument(
        "--gamma",
        type=_number(float, 0.1),
        default=1.0,
        help="> 1 darkens the mid-tones, < 1 brightens them",
    )
    look.add_argument(
        "--clahe", type=_number(float, 0), default=0.0, help="local contrast (0: off, 0.01: mild)"
    )
    look.add_argument("--invert", action="store_true", help="print a negative")
    look.add_argument("--t-min", type=_number(float, 0.3), help="thinnest (brightest) mm")
    look.add_argument("--t-max", type=_number(float, 0.5), help="thickest (darkest) mm")
    look.add_argument(
        "--td",
        type=_number(float, 0.1),
        help="mm of your filament that let 10%% of the light through",
    )
    look.add_argument(
        "--pitch", type=_number(float, 0.05), default=0.2, help="mm between grid points"
    )
    look.add_argument("--max-triangles", type=_number(int, 10_000), default=1_000_000)
    extra = m.add_argument_group("extras")
    extra.add_argument(
        "--hole", action="store_true", help="flat and shaped: a hanging tab with a hole on top"
    )
    extra.add_argument(
        "--stand", action="store_true", help="flat: also make a stand with a slot for the panel"
    )
    extra.add_argument(
        "--lamp-base",
        action="store_true",
        help="cylinder: also make a base with a pocket for an LED",
    )
    extra.add_argument(
        "--led-diameter", type=_number(float, 10), default=40.0, help="LED pocket diameter, mm"
    )
    extra.add_argument(
        "--led-height", type=_number(float, 5), default=18.0, help="LED pocket height, mm"
    )
    col = m.add_argument_group("colour mode")
    col.add_argument(
        "--colors", type=_number(int, 2), default=4, help="number of filaments (2 to 8)"
    )
    col.add_argument(
        "--filaments",
        type=Path,
        help="TOML file with your filaments ([[filament]] name, hex, td) instead of the "
        "Prusament PETG catalog",
    )
    col.add_argument("--allow-abrasive", action="store_true", help=ABRASIVE_HELP)
    prn = m.add_argument_group("printer")
    prn.add_argument("--printer", choices=list(PRINTERS), default=DEFAULT_PRINTER)
    prn.add_argument("--material", choices=MATERIALS, default=DEFAULT_MATERIAL)
    prn.add_argument("--printer-preset", help="PrusaSlicer printer preset (overrides the table)")
    prn.add_argument("--print-preset", help="PrusaSlicer print preset (overrides the table)")
    prn.add_argument("--filament-preset", help="PrusaSlicer filament preset (overrides the table)")
    prn.add_argument(
        "--slice",
        action="store_true",
        help="also write the G-code (.bgcode on printers that use it)",
    )
    out = m.add_argument_group("output")
    out.add_argument("-o", "--out", type=Path, default=Path("out"), help="output folder")
    out.add_argument("--name", help="name of the files (default: the photo's name)")

    c = sub.add_parser(
        "calibrate", help="make a strip of known thicknesses to measure your filament"
    )
    c.add_argument("--printer", choices=list(PRINTERS), default=DEFAULT_PRINTER)
    c.add_argument("--material", choices=MATERIALS, default=DEFAULT_MATERIAL)
    c.add_argument("--slice", action="store_true", help="also write the G-code")
    c.add_argument("-o", "--out", type=Path, default=Path("out"), help="output folder")

    sub.add_parser("printers", help="list the printers and check their presets")
    return p


def job_from(args: argparse.Namespace):
    from . import image, optics, printers
    from .forge import COLOR_RANGE, Job

    try:
        setup = printers.setup(
            args.printer,
            args.material,
            args.printer_preset,
            args.print_preset,
            args.filament_preset,
        )
        mode = args.mode or ("color" if args.shape in ("flat", "shaped") else "mono")
        mono = mode == "mono"
        if not mono and args.td is not None:
            raise ValueError(
                "--td is for white prints (--mode mono); in colour the filaments' own TD "
                "values count, and --t-min/--t-max set the height range of the stack"
            )
        opt = optics.optics(
            args.material, args.t_min if mono else None, args.t_max if mono else None, args.td
        )
        crop = image.parse_crop(args.crop) if args.crop else None
    except ValueError as e:
        raise UsageError(str(e)) from None
    if not args.photo.is_file():
        raise UsageError(f"no such photo: {args.photo}")
    if args.filaments and not args.filaments.is_file():
        raise UsageError(f"no such filament list: {args.filaments}")
    tone = image.Tone(levels=args.levels, gamma=args.gamma, clahe=args.clahe, invert=args.invert)
    lo, hi = COLOR_RANGE
    color_range = (args.t_min or lo, args.t_max or hi)
    return Job(
        photo=args.photo,
        setup=setup,
        optics=opt,
        out=args.out,
        name=args.name,
        shape=args.shape,
        mode=mode,
        colors=args.colors,
        filaments=args.filaments,
        allow_abrasive=args.allow_abrasive,
        color_range=color_range,
        width=args.width,
        height=args.height,
        arc=args.arc,
        diameter=args.diameter,
        repeat=args.repeat,
        mask=args.mask,
        hole=args.hole,
        stand=args.stand,
        lamp_base=args.lamp_base,
        led_d=args.led_diameter,
        led_h=args.led_height,
        frame=args.frame,
        frame_t=args.frame_t,
        pitch=args.pitch,
        max_triangles=args.max_triangles,
        view=args.view,
        max_overhang=min(args.max_overhang, 90.0),
        crop=crop,
        tone=tone,
        slice=args.slice,
    )


def cmd_make(args) -> int:
    from .forge import make

    job = job_from(args)
    try:
        res = make(job)
    except ValueError as e:
        raise UsageError(str(e)) from None
    sx, sy, sz = res.size
    lay = res.layout
    print(
        f"{job.stem}: {job.shape}, {job.mode}, printed {'standing' if job.standing else 'lying'}"
        f", {sx:.1f} x {sy:.1f} x {sz:.1f} mm on the bed, "
        f"grid {lay.cols} x {lay.rows} at {lay.pitch:.2f} mm, "
        f"{res.triangles:,} triangles, {res.volume_cm3:.1f} cm3"
    )
    s = job.setup
    print(f"printer: {s.printer_preset} | {s.print_preset} | {s.filament_preset}")
    if res.colour is not None:
        print_colour(res.colour, res.colour_changes)
    for name, info in res.sliced.items():
        print(f"PrusaSlicer, {name}: {info['time']}, {info['filament_g']} g")
    for note in res.notes:
        print(f"note: {note}")
    for kind, path in res.files.items():
        print(f"{kind:16s} {path}")
    return 0


def print_colour(plan, changes) -> None:
    print(
        "ideal colours, from the bed up:",
        ", ".join(f"{f.hex} (TD {f.td:g})" for f in plan.ideal.filaments),
        f"-> mean Delta E {plan.ideal.mean_de:.1f}",
    )
    print(f"suggested filaments ({plan.source}):")
    print(f"  start with {plan.real.filaments[0].name} {plan.real.filaments[0].hex}")
    for (z, _), f in zip(changes, plan.real.filaments[1:]):
        print(f"  M600 at Z {z:.2f} mm: {f.name} {f.hex}")
    print(f"  -> mean Delta E {plan.real.mean_de:.1f}, 95 % within {plan.real.p95_de:.0f}")


def cmd_analyze(args) -> int:
    from . import image

    if not args.photo.is_file():
        raise UsageError(f"no such photo: {args.photo}")
    if args.filaments and not args.filaments.is_file():
        raise UsageError(f"no such filament list: {args.filaments}")
    try:
        rgb = image.load(args.photo)
    except ValueError as e:
        raise UsageError(str(e)) from None
    a = image.analyse(rgb)
    print(f"{args.photo.name}: {a.size[0]} x {a.size[1]} px")
    print(
        f"lightness L*: mean {a.mean_lstar:.0f}, 1%..99% {a.p1_lstar:.0f}..{a.p99_lstar:.0f}"
        f" (range {a.dynamic_range:.0f})"
    )
    print(
        f"near black {a.shadows:.0%}, near white {a.highlights:.0%}, "
        f"colourfulness {a.colourfulness:.0f}"
    )
    # with --colors the colour plan says how well the colours fit; the mono warning about
    # colourful photos does not apply
    notes = [n for n in a.notes if not (args.colors and "colourful" in n)]
    for note in notes or ["looks fine for a lithophane"]:
        print(f"- {note}")
    if args.colors:
        from . import palette, printers
        from .forge import COLOR_RANGE

        if not 2 <= args.colors <= 8:
            raise UsageError("--colors must be between 2 and 8")
        s = printers.setup()
        small = image.resize(rgb, 240, max(1, round(240 * rgb.shape[1] / rgb.shape[0])))
        h = palette.heights(*COLOR_RANGE, s.first_layer, s.layer)
        try:
            choices, source = palette.choices(args.filaments, args.allow_abrasive)
        except ValueError as e:
            raise UsageError(str(e)) from None
        plan = palette.plan(palette.target(small), args.colors, h, choices, source)
        n = len(plan.real.filaments)
        print(f"colour print with {n} of {args.colors} filaments ({s.printer.title} layers):")
        print_colour(plan, plan.real.colour_changes(s.layer))
    return 0


def cmd_calibrate(args) -> int:
    from . import calibrate, optics, printers

    try:
        setup = printers.setup(args.printer, args.material)
    except ValueError as e:
        raise UsageError(str(e)) from None
    job = calibrate.default_job(setup, optics.DEFAULTS[args.material], args.out, slice_=args.slice)
    try:
        res = calibrate.make(job)
    except ValueError as e:  # for example PrusaSlicer took other presets
        raise UsageError(str(e)) from None
    sx, sy, sz = res.size
    print(
        f"{job.stem}: {sx:.1f} x {sy:.1f} x {sz:.1f} mm on the bed, steps "
        + ", ".join(f"{v:g}" for v in calibrate.STEPS)
        + " mm"
    )
    for info in res.sliced.values():
        print(f"PrusaSlicer: {info['time']}, {info['filament_g']} g")
    for note in res.notes:
        print(f"note: {note}")
    for kind, path in res.files.items():
        print(f"{kind:16s} {path}")
    print(f"then read {res.files['report']} to measure t_min, t_max and td")
    return 0


def cmd_printers(args) -> int:
    from . import profiles, slicer
    from .printers import MATERIALS, PRINTERS, setup

    bundle = profiles.find_bundle()
    print(
        f"PrusaSlicer bundle: {bundle or 'not found'}"
        + (f" (version {profiles.load(bundle).version})" if bundle else "")
    )
    try:
        cmd = slicer.command()
    except ValueError as e:
        raise UsageError(str(e)) from None
    found = f"{' '.join(cmd)} (version {slicer.version(cmd)})" if cmd else "not found"
    print(f"PrusaSlicer command: {found}")
    problems = 0
    for key, p in PRINTERS.items():
        status = "OK"
        for material in MATERIALS:
            try:
                s = setup(key, material)
            except ValueError as e:
                status, problems = str(e), problems + 1
        bed = f"{s.bed[0]:.0f} x {s.bed[1]:.0f} x {s.bed[2]:.0f}" if status == "OK" else "?"
        print(f"{key:9s} {p.title:52s} {bed:>15s} mm  {status}")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    handler = {
        "make": cmd_make,
        "analyze": cmd_analyze,
        "calibrate": cmd_calibrate,
        "printers": cmd_printers,
    }[args.command]
    try:
        return handler(args)
    except UsageError as e:
        print(f"lithophane-forge: error: {e}", file=sys.stderr)
        return 2
