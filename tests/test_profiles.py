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
    # the fallback numbers in the table match the bundle
    assert s.bed == pytest.approx(p.bed)


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
