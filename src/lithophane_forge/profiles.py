"""Read PrusaSlicer's vendor bundle (PrusaResearch.ini) and resolve presets.

A bundle is an INI file with one section per preset, for example
`[printer:Original Prusa MK4 Input Shaper 0.4 nozzle]`. A preset may name parents with
`inherits = a; b`: their values come first, in order, and the preset's own values win.
Abstract parents have names like `*common*`. We only read values; we never evaluate
`compatible_printers_condition`.
"""

import os
from functools import cache
from pathlib import Path

ENV = "LITHOPHANE_FORGE_PROFILES"
SEARCH = (
    Path("/usr/share/PrusaSlicer/profiles"),
    Path.home() / ".config/PrusaSlicer/vendor",
    Path("/usr/local/share/PrusaSlicer/profiles"),
)
BUNDLE = "PrusaResearch.ini"


def find_bundle() -> Path | None:
    """The PrusaResearch.ini to use, or None if PrusaSlicer's profiles are not installed.

    $LITHOPHANE_FORGE_PROFILES may name the file itself or the folder that holds it."""
    candidates = []
    if os.environ.get(ENV):
        env = Path(os.environ[ENV]).expanduser()
        candidates.append(env if env.suffix == ".ini" else env / BUNDLE)
    candidates += [folder / BUNDLE for folder in SEARCH]
    return next((c for c in candidates if c.is_file()), None)


Sections = dict[tuple[str, str], dict[str, str]]


def parse(text: str) -> Sections:
    """Section (kind, name) -> raw key/value pairs, without inheritance."""
    sections: Sections = {}
    current: dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            kind, _, name = line[1:-1].partition(":")
            current = sections.setdefault((kind.strip(), name.strip()), {})
            continue
        if current is not None and "=" in line:
            key, _, value = line.partition("=")
            current[key.strip()] = value.strip()
    return sections


class Bundle:
    """A parsed vendor bundle."""

    def __init__(self, text: str, path: Path | None = None):
        self.path = path
        self.sections = parse(text)
        self.version = self.sections.get(("vendor", ""), {}).get("config_version", "?")

    def names(self, kind: str) -> set[str]:
        return {name for k, name in self.sections if k == kind}

    def has(self, kind: str, name: str) -> bool:
        return (kind, name) in self.sections

    def resolve(self, kind: str, name: str) -> dict[str, str]:
        """Every value of a preset, with its parents applied first."""
        return dict(self._resolve(kind, name, ()))

    def _resolve(self, kind: str, name: str, seen: tuple[str, ...]) -> dict[str, str]:
        if name in seen:
            raise ValueError(f"{kind} preset {name!r} inherits from itself")
        if (kind, name) not in self.sections:
            raise KeyError(f"no {kind} preset {name!r} in {self.path or 'the bundle'}")
        own = self.sections[(kind, name)]
        values: dict[str, str] = {}
        for parent in filter(None, (p.strip() for p in own.get("inherits", "").split(";"))):
            values.update(self._resolve(kind, parent, (*seen, name)))
        values.update({k: v for k, v in own.items() if k != "inherits"})
        return values


@cache
def load(path: Path) -> Bundle:
    return Bundle(path.read_text(encoding="utf-8", errors="replace"), path)


def bundle() -> Bundle | None:
    """The installed bundle, or None."""
    path = find_bundle()
    return load(path) if path else None


def bed_size(printer: dict[str, str]) -> tuple[float, float]:
    """Width and depth of the bed from a resolved printer preset (`bed_shape`)."""
    pts = [tuple(float(v) for v in p.split("x")) for p in printer["bed_shape"].split(",")]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return max(xs) - min(xs), max(ys) - min(ys)
