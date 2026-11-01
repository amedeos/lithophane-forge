"""PrusaSlicer command line: turn our 3MF into a full project, and optionally into G-code.

PrusaSlicer writes its complete configuration at the end of every G-code file, in the same
"; key = value" form that it stores inside project 3MF files. The command line cannot
export a project on a machine without a display, so we slice once to text G-code, take
the configuration from there and add it to the 3MF (the idea comes from the owner's
3d-linux project).

The command line finds presets only in a data folder where the printer model is
installed, and a fresh or headless machine has none. So every run gets a temporary data
folder with the vendor bundle that `printers.setup` checked the preset names against, and
a PrusaSlicer.ini that installs the printer model. The result does not depend on your own
PrusaSlicer settings.

Set PRUSA_SLICER to the command if `prusa-slicer` is not on the PATH. For the Flatpak:
PRUSA_SLICER="flatpak run --filesystem=host --filesystem=/tmp --command=prusa-slicer
com.prusa3d.PrusaSlicer" (on one line). If that command has its own --datadir, we use it
as given.
"""

import contextlib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from .printers import Setup

ENV = "PRUSA_SLICER"


def command() -> list[str] | None:
    """The PrusaSlicer command, or None if it is not installed."""
    if os.environ.get(ENV):
        return shlex.split(os.environ[ENV])
    for name in ("prusa-slicer", "PrusaSlicer", "prusa-slicer-console"):
        if shutil.which(name):
            return [name]
    return None


def version(cmd: list[str]) -> str:
    result = subprocess.run([*cmd, "--help"], capture_output=True, text=True, check=False)
    m = re.search(r"PrusaSlicer-([\w.+-]+)", result.stdout + result.stderr)
    return m.group(1) if m else "?"


@contextlib.contextmanager
def datadir(setup: Setup, bundle: Path | None, cmd: list[str] | None = None) -> Iterator[list[str]]:
    """Extra arguments for PrusaSlicer: a temporary --datadir with our vendor bundle."""
    if bundle is None or any(a.startswith("--datadir") for a in cmd or []):
        yield []
        return
    with tempfile.TemporaryDirectory(prefix="lithophane-forge-") as tmp:
        folder = Path(tmp)
        (folder / "vendor").mkdir()
        shutil.copyfile(bundle, folder / "vendor" / "PrusaResearch.ini")
        (folder / "PrusaSlicer.ini").write_text(
            "[presets]\n"
            f"print = {setup.print_preset}\n"
            f"filament = {setup.filament_preset}\n"
            f"printer = {setup.printer_preset}\n\n"
            "[vendor:PrusaResearch]\n"
            f"model:{setup.model} = {setup.variant}\n",
            encoding="utf-8",
        )
        yield ["--datadir", str(folder)]


def run(
    cmd: list[str],
    source: Path,
    out: Path,
    presets: dict[str, str] | None = None,
    binary: bool = False,
    extra: tuple[str, ...] = (),
) -> None:
    """Slice `source` (a 3MF) to `out`. presets=None: use the configuration inside the 3MF."""
    args = [a for kv in (presets or {}).items() for a in kv]
    gcode = ["--binary-gcode" if binary else "--no-binary-gcode", "--export-gcode"]
    result = subprocess.run(
        [*cmd, *extra, *args, *gcode, "-o", str(out), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not out.is_file():
        raise RuntimeError(
            f"PrusaSlicer could not slice {source.name}:\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )


def summary(gcode: Path) -> dict:
    """Print time, filament, colour changes and printer, read from text G-code."""
    text = gcode.read_text(encoding="utf-8", errors="replace")

    def value(key: str) -> str:
        m = re.search(rf"^; {re.escape(key)} = (.*)$", text, re.MULTILINE)
        return m.group(1).strip() if m else "?"

    m600_z = [
        float(z) for z in re.findall(r"^;Z:([\d.]+)\n(?:(?!;Z:).*\n)*?M600", text, re.MULTILINE)
    ]
    return {
        "time": value("estimated printing time (normal mode)"),
        "filament_g": value("filament used [g]"),
        "filament_m": value("filament used [mm]"),
        "m600_z": m600_z,
        "support_lines": text.count(";TYPE:Support material"),
        "printer": value("printer_settings_id"),
        "print": value("print_settings_id"),
        "filament": _setting(value("filament_settings_id")),
    }


def _setting(value: str) -> str:
    """One preset name from a G-code setting: `"A";"A"` (one per extruder) gives A."""
    return value.split(";")[0].strip().strip('"')


def check_presets(info: dict, setup: Setup) -> None:
    """PrusaSlicer quietly takes other presets when the asked ones do not fit together or
    the printer model is not installed; make that an error."""
    wrong = [
        f"{kind} {_setting(info[kind])!r} instead of {want!r}"
        for kind, want in (
            ("printer", setup.printer_preset),
            ("print", setup.print_preset),
            ("filament", setup.filament_preset),
        )
        if _setting(info[kind]) != want
    ]
    if wrong:
        raise ValueError(
            "PrusaSlicer used other presets: "
            + "; ".join(wrong)
            + ". Check that the three presets go together (same printer and nozzle), or set "
            "LITHOPHANE_FORGE_PROFILES to PrusaSlicer's PrusaResearch.ini"
        )


def config_from_gcode(gcode: Path, binary_gcode: bool) -> str:
    """The configuration block that PrusaSlicer appends to every G-code file."""
    lines = gcode.read_text(encoding="utf-8", errors="replace").splitlines()
    start = lines.index("; prusaslicer_config = begin")
    end = lines.index("; prusaslicer_config = end")
    header = lines[0] if lines[0].startswith("; generated by PrusaSlicer") else ""
    body = [line for line in lines[start + 1 : end] if line.startswith("; ")]
    # we sliced to text only to read it; give the project the printer's own G-code format
    flag = f"; binary_gcode = {int(binary_gcode)}"
    body = [flag if line.startswith("; binary_gcode =") else line for line in body]
    return "\n".join([header, *body]) + "\n"
