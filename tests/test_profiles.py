import pytest

from lithophane_forge import printers, profiles

SAMPLE = """
# comment
[vendor]
config_version = 9.9.9

[printer:*common*]
bed_shape = 0x0,250x0,250x210,0x210
max_print_height = 220
gcode = a

[printer:*fast*]
gcode = b

[printer:Child]
inherits = *common*; *fast*
max_print_height = 200
"""


def test_inherits_in_order():
    b = profiles.Bundle(SAMPLE)
    child = b.resolve("printer", "Child")
    assert child["gcode"] == "b"
    assert child["max_print_height"] == "200"
    assert "inherits" not in child
    assert b.version == "9.9.9"
    assert profiles.bed_size(child) == (250, 210)


def test_unknown_preset():
    with pytest.raises(KeyError):
        profiles.Bundle(SAMPLE).resolve("printer", "Nope")


def test_loop():
    with pytest.raises(ValueError):
        profiles.Bundle("[print:a]\ninherits = b\n[print:b]\ninherits = a\n").resolve("print", "a")


def test_env_names_the_bundle(tmp_path, monkeypatch):
    (tmp_path / "PrusaResearch.ini").write_text(SAMPLE, encoding="utf-8")
    monkeypatch.setenv(profiles.ENV, str(tmp_path))
    assert profiles.find_bundle() == tmp_path / "PrusaResearch.ini"


installed = pytest.mark.skipif(
    profiles.find_bundle() is None, reason="PrusaSlicer's PrusaResearch.ini is not installed"
)


@installed
@pytest.mark.parametrize("key", sorted(printers.PRINTERS))
@pytest.mark.parametrize("material", printers.MATERIALS)
def test_every_preset_exists(key, material):
    s = printers.setup(key, material)
    assert s.layer == pytest.approx(0.10)
    p = printers.PRINTERS[key]
    # the fallback values in the table match the bundle
    assert s.bed == pytest.approx(p.bed)
    assert s.binary_gcode == p.binary
    assert (s.model, s.variant) == (p.model, f"{p.nozzle:g}")


@installed
def test_other_nozzle_installs_its_variant():
    """With presets for a 0.6 nozzle, PrusaSlicer must get the 0.6 variant installed."""
    s = printers.setup(
        "mk4",
        "petg",
        printer_preset="Original Prusa MK4 Input Shaper 0.6 nozzle",
        print_preset="0.15mm STRUCTURAL @MK4IS 0.6",
        filament_preset="Prusament PETG @PGIS 0.6",
    )
    assert (s.model, s.variant) == ("MK4IS", "0.6")


@installed
def test_unknown_preset_name_is_an_error():
    with pytest.raises(ValueError, match="not in"):
        printers.setup("mk4", "petg", print_preset="0.10mm NOPE")


def test_setup_without_bundle(monkeypatch):
    """Without PrusaSlicer's profiles we trust the table and its fallback numbers."""
    monkeypatch.setattr(profiles, "bundle", lambda: None)
    s = printers.setup("coreone", "pla")
    assert s.bed == printers.PRINTERS["coreone"].bed
    assert s.filament_preset == "Prusament PLA @COREONE"
    assert s.layer == 0.10 and s.binary_gcode
    assert (s.model, s.variant) == ("COREONE", "0.4")
    assert not printers.setup("mk3s", "petg").binary_gcode  # the MK3S reads text G-code


def test_other_presets_are_an_error():
    """PrusaSlicer may take other presets than the ones we ask for; that must not pass."""
    from lithophane_forge import slicer

    s = printers.setup("mk4", "petg")
    info = {"printer": s.printer_preset, "print": s.print_preset, "filament": s.filament_preset}
    slicer.check_presets(info, s)
    slicer.check_presets({**info, "filament": f'"{s.filament_preset}";"{s.filament_preset}"'}, s)
    with pytest.raises(ValueError, match="instead of"):
        slicer.check_presets({**info, "filament": "3D-Fuel Pro PCTG @PG"}, s)
