"""lithophane-forge: turn a photo into a printable lithophane and a PrusaSlicer project."""

import argparse
import sys
from pathlib import Path

__version__ = "0.1.0"

EXAMPLES = """\
examples:
  lithophane-forge analyze photo.jpg
  lithophane-forge make photo.jpg --width 120
  lithophane-forge make photo.jpg --shape curved --height 100 --arc 110
  lithophane-forge make photo.jpg --shape cylinder --diameter 80 --repeat 2 --lamp-base
  lithophane-forge make photo.jpg --shape shaped --mask heart --width 90 --hole
  lithophane-forge make photo.jpg --printer coreone --material pla --slice
  lithophane-forge printers
"""


class UsageError(Exception):
    pass


def _number(kind=float, low: float | None = None):
    def parse(text: str):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if low is not None and value < low:
            raise argparse.ArgumentTypeError(f"must be at least {low:g} (got {text})")
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
    from .forge import SHAPES, VIEWS
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

    m = sub.add_parser(
        "make",
        help="make the lithophane files from a photo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    m.add_argument("photo", type=Path)
    size = m.add_argument_group("size and shape")
    size.add_argument("--shape", choices=SHAPES, default="flat")
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
        "--arc", type=_number(float, 10), default=120.0, help="curved: degrees of arc"
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
        "--frame-t", type=_number(float, 0.5), help="frame thickness in mm (default: t_max + 0.8)"
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

    sub.add_parser("printers", help="list the printers and check their presets")
    return p


def job_from(args: argparse.Namespace):
    from . import image, optics, printers
    from .forge import Job

    try:
        setup = printers.setup(
            args.printer,
            args.material,
            args.printer_preset,
            args.print_preset,
            args.filament_preset,
        )
        opt = optics.optics(args.material, args.t_min, args.t_max, args.td)
        crop = image.parse_crop(args.crop) if args.crop else None
    except ValueError as e:
        raise UsageError(str(e)) from None
    if not args.photo.is_file():
        raise UsageError(f"no such photo: {args.photo}")
    tone = image.Tone(levels=args.levels, gamma=args.gamma, clahe=args.clahe, invert=args.invert)
    return Job(
        photo=args.photo,
        setup=setup,
        optics=opt,
        out=args.out,
        name=args.name,
        shape=args.shape,
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
        f"{job.stem}: {job.shape}, {sx:.1f} x {sy:.1f} x {sz:.1f} mm on the bed, "
        f"grid {lay.cols} x {lay.rows} at {lay.pitch:.2f} mm, "
        f"{res.triangles:,} triangles, {res.volume_cm3:.1f} cm3"
    )
    s = job.setup
    print(f"printer: {s.printer_preset} | {s.print_preset} | {s.filament_preset}")
    for name, info in res.sliced.items():
        print(f"PrusaSlicer, {name}: {info['time']}, {info['filament_g']} g")
    for note in res.notes:
        print(f"note: {note}")
    for kind, path in res.files.items():
        print(f"{kind:16s} {path}")
    return 0


def cmd_analyze(args) -> int:
    from . import image

    if not args.photo.is_file():
        raise UsageError(f"no such photo: {args.photo}")
    a = image.analyse(image.load(args.photo))
    print(f"{args.photo.name}: {a.size[0]} x {a.size[1]} px")
    print(
        f"lightness L*: mean {a.mean_lstar:.0f}, 1%..99% {a.p1_lstar:.0f}..{a.p99_lstar:.0f}"
        f" (range {a.dynamic_range:.0f})"
    )
    print(
        f"near black {a.shadows:.0%}, near white {a.highlights:.0%}, "
        f"colourfulness {a.colourfulness:.0f}"
    )
    for note in a.notes or ["looks fine for a lithophane"]:
        print(f"- {note}")
    return 0


def cmd_printers(args) -> int:
    from . import profiles, slicer
    from .printers import MATERIALS, PRINTERS, setup

    bundle = profiles.find_bundle()
    print(
        f"PrusaSlicer bundle: {bundle or 'not found'}"
        + (f" (version {profiles.load(bundle).version})" if bundle else "")
    )
    cmd = slicer.command()
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
    handler = {"make": cmd_make, "analyze": cmd_analyze, "printers": cmd_printers}[args.command]
    try:
        return handler(args)
    except UsageError as e:
        print(f"lithophane-forge: error: {e}", file=sys.stderr)
        return 2
