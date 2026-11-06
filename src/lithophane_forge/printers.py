"""The Prusa printers we know, with the PrusaSlicer presets that suit a lithophane.

Every name below exists in PrusaSlicer's PrusaResearch bundle (config_version 2.4.14,
PrusaSlicer 2.9.6). `tests/test_printers.py` checks them against the installed bundle.
We use the finest "0.10mm" print preset: lying, the layer height is the step of the
thickness (the tones of a white panel, the change heights of a colour one); standing, it
is the vertical resolution. Only the 0.4 mm nozzle is in the table; for
other nozzles pass the three preset names yourself (--printer-preset and friends).
"""

from dataclasses import dataclass

MATERIALS = ("petg", "pla")
DEFAULT_PRINTER = "mk4"
DEFAULT_MATERIAL = "petg"


@dataclass(frozen=True)
class Printer:
    key: str
    title: str
    model: str  # printer_model id in the bundle (for PrusaSlicer.ini: model:<id> = 0.4)
    printer: str  # PrusaSlicer printer preset
    print: str  # PrusaSlicer print preset
    filaments: dict[str, str]  # material -> filament preset
    bed: tuple[float, float, float]  # x, y, max print height (fallback without the bundle)
    nozzle: float = 0.4
    binary: bool = True  # the printer reads binary G-code (fallback without the bundle)

    def filament(self, material: str) -> str:
        return self.filaments[material]


PRINTERS: dict[str, Printer] = {
    p.key: p
    for p in (
        Printer(
            "mk4",
            "Original Prusa MK4 (Input Shaper firmware)",
            "MK4IS",
            "Original Prusa MK4 Input Shaper 0.4 nozzle",
            "0.10mm FAST DETAIL @MK4IS 0.4",
            {"petg": "Prusament PETG @PGIS", "pla": "Prusament PLA @PGIS"},
            (250, 210, 220),
        ),
        Printer(
            "mk4s",
            "Original Prusa MK4S",
            "MK4S",
            "Original Prusa MK4S 0.4 nozzle",
            "0.10mm FAST DETAIL @MK4S 0.4",
            {"petg": "Prusament PETG @MK4S", "pla": "Prusament PLA @MK4S"},
            (250, 210, 220),
        ),
        Printer(
            "mk3.5",
            "Original Prusa MK3.5 / MK3.5S",
            "MK3.5",
            "Original Prusa MK3.5 & MK3.5S 0.4 nozzle",
            "0.10mm FAST DETAIL @MK3.5 0.4",
            {"petg": "Prusament PETG", "pla": "Prusament PLA @MK3.5"},
            (250, 210, 210),
        ),
        Printer(
            "mk3s",
            "Original Prusa i3 MK3S / MK3S+",
            "MK3S",
            "Original Prusa i3 MK3S & MK3S+",
            "0.10mm DETAIL @MK3",
            {"petg": "Prusament PETG", "pla": "Prusament PLA"},
            (250, 210, 210),
            binary=False,
        ),
        Printer(
            "coreone",
            "Prusa CORE One",
            "COREONE",
            "Prusa CORE One 0.4 nozzle",
            "0.10mm FAST DETAIL @COREONE 0.4",
            {"petg": "Prusament PETG @COREONE", "pla": "Prusament PLA @COREONE"},
            (250, 220, 270),
        ),
        Printer(
            "coreonel",
            "Prusa CORE One L",
            "COREONEL",
            "Prusa CORE One L 0.4 nozzle",
            "0.10mm FAST DETAIL @COREONEL 0.4",
            {"petg": "Prusament PETG @COREONE", "pla": "Prusament PLA @COREONE"},
            (300, 300, 330),
        ),
        Printer(
            "xl",
            "Original Prusa XL (Input Shaper, one tool)",
            "XLIS",
            "Original Prusa XL Input Shaper 0.4 nozzle",
            "0.10mm FAST DETAIL @XLIS 0.4",
            {"petg": "Prusament PETG @XL", "pla": "Prusament PLA @XLIS"},
            (360, 360, 360),
        ),
        Printer(
            "mini",
            "Original Prusa MINI / MINI+ (Input Shaper firmware)",
            "MINIIS",
            "Original Prusa MINI & MINI+ Input Shaper",
            "0.10mm FAST DETAIL @MINIIS 0.4",
            {"petg": "Prusament PETG @MINIIS", "pla": "Prusament PLA @MINIIS"},
            (180, 180, 180),
        ),
    )
}


@dataclass(frozen=True)
class Setup:
    """What we slice with: one printer, its three presets and the print volume."""

    printer: Printer
    material: str
    printer_preset: str
    print_preset: str
    filament_preset: str
    bed: tuple[float, float, float]
    layer: float
    first_layer: float
    binary_gcode: bool
    model: str  # printer_model and printer_variant of the printer preset: what PrusaSlicer
    variant: str  # must have installed to accept the presets (see slicer.datadir)

    def presets(self) -> dict[str, str]:
        return {
            "--printer-profile": self.printer_preset,
            "--print-profile": self.print_preset,
            "--material-profile": self.filament_preset,
        }


def setup(
    key: str = DEFAULT_PRINTER,
    material: str = DEFAULT_MATERIAL,
    printer_preset: str | None = None,
    print_preset: str | None = None,
    filament_preset: str | None = None,
) -> Setup:
    """The presets for a printer, checked against the installed bundle when there is one.

    Without the bundle we trust the table and its fallback numbers (0.10 mm layers)."""
    from . import profiles

    if key not in PRINTERS:
        raise ValueError(f"unknown printer {key!r}; choose from {', '.join(PRINTERS)}")
    if material not in MATERIALS:
        raise ValueError(f"unknown material {material!r}; choose from {', '.join(MATERIALS)}")
    p = PRINTERS[key]
    names = {
        "printer": printer_preset or p.printer,
        "print": print_preset or p.print,
        "filament": filament_preset or p.filament(material),
    }
    found = profiles.bundle()
    bed, layer, first, binary = p.bed, 0.10, 0.20, p.binary
    model, variant = p.model, f"{p.nozzle:g}"
    if found is not None:
        missing = [
            f"{kind} preset {name!r}" for kind, name in names.items() if not found.has(kind, name)
        ]
        if missing:
            raise ValueError(
                f"not in {found.path} (version {found.version}): " + ", ".join(missing)
            )
        printer = found.resolve("printer", names["printer"])
        print_ = found.resolve("print", names["print"])
        bed = (*profiles.bed_size(printer), float(printer.get("max_print_height", p.bed[2])))
        layer = float(print_.get("layer_height", layer))
        first = float(print_.get("first_layer_height", first))
        binary = printer.get("binary_gcode", "0") == "1"
        model = printer.get("printer_model", model)
        variant = printer.get("printer_variant", variant)
    return Setup(
        p,
        material,
        names["printer"],
        names["print"],
        names["filament"],
        bed,
        layer,
        first,
        binary,
        model,
        variant,
    )
