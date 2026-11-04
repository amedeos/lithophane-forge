import re
import shlex
from pathlib import Path

import pytest

import lithophane_forge
from lithophane_forge import main

README = Path(__file__).parents[1] / "README.md"


def commands():
    """Every `uv run lithophane-forge ...` line in the README's sh blocks, and the --help
    examples."""
    lines = []
    for block in re.findall(r"```sh\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL):
        lines += [
            ln.split("#")[0].strip()
            for ln in block.splitlines()
            if ln.strip().startswith("uv run lithophane-forge ")
        ]
    lines += [
        ln.strip()
        for ln in lithophane_forge.EXAMPLES.splitlines()
        if ln.strip().startswith("lithophane-forge ")
    ]
    return [shlex.split(ln.removeprefix("uv run ")) for ln in lines]


def test_there_are_examples():
    assert len(commands()) >= 5


@pytest.mark.parametrize("argv", commands(), ids=lambda a: " ".join(a[1:]))
def test_examples_parse(argv, monkeypatch, tmp_path, photo):
    """The documented commands parse and reach their handler (the work itself is skipped).
    They assume the vision extra (--focus subject)."""
    from lithophane_forge import vision

    monkeypatch.setattr(vision, "available", lambda: True)
    calls = []
    for name in ("cmd_make", "cmd_analyze", "cmd_calibrate", "cmd_printers", "cmd_models"):
        monkeypatch.setattr(lithophane_forge, name, lambda a, n=name: calls.append(n) or 0)
    args = [str(photo) if a.endswith((".jpg", ".png")) else a for a in argv[1:]]
    assert main(args) == 0
    assert calls
    if args[0] == "make":  # the options also fit together (mode, shape, tone ...)
        lithophane_forge.job_from(lithophane_forge.parser().parse_args(args)).check()


def test_make_writes_files(photo, tmp_path, no_slicer, capsys):
    assert (
        main(["make", str(photo), "--width", "40", "--pitch", "0.5", "-o", str(tmp_path / "o")])
        == 0
    )
    out = capsys.readouterr().out
    assert "triangles" in out and (tmp_path / "o" / "photo" / "photo.3mf").is_file()


@pytest.mark.parametrize(
    "argv, mode",
    [
        ([], "color"),
        (["--shape", "shaped"], "color"),
        (["--mode", "mono"], "mono"),
        (["--shape", "curved"], "mono"),
        (["--shape", "cylinder"], "mono"),
    ],
)
def test_colour_is_the_default(photo, argv, mode):
    """Flat and shaped panels print in colour, lying; curved panels and cylinders stand."""
    from lithophane_forge import job_from, parser

    assert job_from(parser().parse_args(["make", str(photo), *argv])).mode == mode


@pytest.mark.parametrize(
    "argv, light, view",
    [
        ([], "front", "relief"),
        (["--light", "back"], "back", "flat"),
        (["--light", "back", "--view", "relief"], "back", "relief"),
        (["--mode", "mono"], "back", "flat"),
        (["--shape", "curved"], "back", "flat"),
    ],
)
def test_front_light_is_the_colour_default(photo, argv, light, view):
    """A colour print is a picture you look at in room light, from the top, unless
    --light back asks for a lithophane."""
    from lithophane_forge import job_from, parser

    job = job_from(parser().parse_args(["make", str(photo), *argv]))
    assert (job.light, job.face) == (light, view)
    job.check()


def test_analyze(photo, capsys):
    assert main(["analyze", str(photo)]) == 0
    assert "lightness" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--light", "back"], "add --colors"),
        (["--mode", "mono", "--colors", "3"], "not --mode mono"),
        (["--mode", "mono", "--light", "front"], "not --mode mono"),
        (["--allow-unmeasured"], "add --colors"),
        (["--colors", "3", "--light", "back", "--allow-unmeasured"], "not --light back"),
        (["--focus", "centre"], "add --colors"),
        (["--colors", "2", "--focus-floor", "0.2"], "--focus-floor goes with --focus"),
        (["--colors", "2", "--focus", "nothing.png"], "no such focus mask"),
        (["--colors", "2", "--focus", "1,2,3"], "X,Y,W,H"),
        (["--colors", "2", "--focus", "999,999,5,5"], "outside the photo"),
        (["--colors", "2", "--device", "cpu"], "--device goes with --focus subject"),
        (["--colors", "2", "--focus", "centre", "--device", "gpu"], "goes with --focus subject"),
    ],
)
def test_analyze_options_that_do_nothing_are_errors(photo, capsys, argv, message):
    assert main(["analyze", str(photo), *argv]) == 2
    assert message in capsys.readouterr().err


def test_help_shows_no_none_defaults(capsys):
    for command in ("make", "analyze"):
        with pytest.raises(SystemExit):
            main([command, "--help"])
        out = capsys.readouterr().out
        assert "(default: None)" not in out and "(default: flat)" not in out
        assert "front (the default)" in out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["make", "missing.jpg"], "no such photo"),
        (["make", "{photo}", "--mode", "mono", "--t-min", "3", "--t-max", "2"], "t_min < t_max"),
        (["make", "{photo}", "--t-min", "3", "--t-max", "2"], "range 3 to 2 mm"),
        (["make", "{photo}", "--crop", "1,2,3"], "X,Y,W,H"),
        (
            ["make", "{photo}", "--mode", "mono", "--height", "500", "--pitch", "1"],
            "prints at most",
        ),
        (["make", "{photo}", "--height", "500", "--pitch", "1"], "does not fit"),
        (["make", "{photo}", "--clahe", "0.01"], "--mode mono"),
        (["make", "{photo}", "--td", "2"], "--td is for white prints"),
        (["make", "{photo}", "--mode", "mono", "--light", "front"], "for colour prints"),
        (["make", "{photo}", "--view", "flat"], "--view relief"),
        (["make", "{photo}", "--mode", "mono", "--min-patch", "2"], "not --mode mono"),
        (["make", "{photo}", "--shape", "cylinder", "--min-patch", "2"], "cylinder (always mono)"),
        (["make", "{photo}", "--light", "back", "--allow-unmeasured"], "not --light back"),
        (["make", "{photo}", "--mode", "mono", "--allow-unmeasured"], "not --mode mono"),
        (
            ["make", "{photo}", "--shape", "curved", "--allow-unmeasured"],
            "curved (always mono)",
        ),
        (["make", "{photo}", "--mode", "mono", "--focus", "centre"], "not --mode mono"),
        (["make", "{photo}", "--shape", "cylinder", "--focus", "centre"], "(always mono)"),
        (["make", "{photo}", "--focus-floor", "0.2"], "--focus-floor goes with --focus"),
        (["make", "{photo}", "--focus", "missing.png"], "no such focus mask"),
        (["make", "{photo}", "--focus", "1,2,3"], "X,Y,W,H"),
        (["make", "{photo}", "--focus", "500,500,10,10"], "outside the photo"),
        (["make", "{photo}", "--mode", "mono", "--focus", "subject"], "not --mode mono"),
        (["make", "{photo}", "--device", "gpu"], "--device goes with --focus subject"),
        (["make", "{photo}", "--focus", "centre", "--device", "cpu"], "goes with --focus subject"),
        (["make", "{photo}", "--min-patch-subject", "1"], "--min-patch-subject goes with --focus"),
        (["make", "{photo}", "--focus", "centre", "--min-patch-subject", "2"], "raise --min-patch"),
        (["make", "{photo}", "--smooth-background", "1"], "--smooth-background goes with --focus"),
        (["make", "{photo}", "--mode", "mono", "--min-patch-subject", "1"], "not --mode mono"),
        (
            ["make", "{photo}", "--shape", "curved", "--smooth-background", "0"],
            "curved (always mono)",
        ),
    ],
)
def test_errors_exit_with_2(argv, message, photo, capsys, no_slicer, tmp_path):
    argv = [a.format(photo=photo) for a in argv] + ["-o", str(tmp_path)]
    assert main(argv) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--printer", "nope"],
        ["--shape", "curved", "--arc", "400"],
        ["--min-patch", "inf"],
        ["--min-patch", "nan"],
        ["--width", "inf"],
        ["--focus", "centre", "--focus-floor", "0"],
        ["--focus", "centre", "--focus-floor", "1.5"],
        ["--focus", "subject", "--device", "tpu"],
        ["--focus", "centre", "--min-patch-subject", "-1"],
        ["--focus", "centre", "--smooth-background", "nan"],
        ["--focus", "centre", "--smooth-background", "3"],
    ],
)
def test_bad_option_exits_with_2(argv):
    with pytest.raises(SystemExit) as e:
        main(["make", "x.jpg", *argv])
    assert e.value.code == 2


def test_min_patch_reaches_the_job(photo, capsys):
    from lithophane_forge import forge, job_from, parser

    p = parser()
    assert job_from(p.parse_args(["make", str(photo)])).min_patch == forge.MIN_PATCH
    assert job_from(p.parse_args(["make", str(photo), "--min-patch", "0"])).min_patch == 0
    with pytest.raises(SystemExit):
        main(["make", "--help"])
    assert f"(default {forge.MIN_PATCH:g}; 0 keeps them)" in " ".join(
        capsys.readouterr().out.split()
    )


def test_allow_abrasive_reaches_the_job(photo, tmp_path):
    from lithophane_forge import job_from, parser

    args = parser().parse_args(["make", str(photo), "--mode", "color", "--allow-abrasive"])
    assert job_from(args).allow_abrasive
    assert not job_from(parser().parse_args(["make", str(photo)])).allow_abrasive


def test_allow_unmeasured_reaches_the_job(photo):
    from lithophane_forge import job_from, parser

    args = parser().parse_args(["make", str(photo), "--allow-unmeasured"])
    assert job_from(args).allow_unmeasured
    assert not job_from(parser().parse_args(["make", str(photo)])).allow_unmeasured


@pytest.mark.parametrize("command", ["make", "analyze"])
def test_a_file_that_is_not_a_photo(command, tmp_path, capsys, no_slicer):
    fake = tmp_path / "notes.jpg"
    fake.write_text("not a photo", encoding="utf-8")
    assert main([command, str(fake)] + (["-o", str(tmp_path)] if command == "make" else [])) == 2
    assert "cannot read" in capsys.readouterr().err


def test_a_wrong_prusa_slicer_is_an_error(monkeypatch, capsys):
    from lithophane_forge import slicer

    monkeypatch.setenv("PRUSA_SLICER", "/no/such/prusa-slicer --datadir x")
    with pytest.raises(ValueError, match="does not name a command"):
        slicer.command()
    assert main(["printers"]) == 2
    assert "PRUSA_SLICER" in capsys.readouterr().err


def test_calibrate_errors_exit_with_2(monkeypatch, capsys, tmp_path):
    from lithophane_forge import calibrate

    def fail(job):
        raise ValueError("PrusaSlicer used other presets")

    monkeypatch.setattr(calibrate, "make", fail)
    assert main(["calibrate", "-o", str(tmp_path)]) == 2
    assert "other presets" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--min-patch-subject", "--smooth-background"])
def test_analyze_has_no_focus_region_options(photo, option):
    """analyze shows the plan before the heights are smoothed: the options are make's."""
    with pytest.raises(SystemExit) as e:
        main(["analyze", str(photo), "--colors", "3", "--focus", "centre", option, "1"])
    assert e.value.code == 2


def test_focus_regions_reach_the_job(photo):
    from lithophane_forge import forge, job_from, parser

    job = job_from(parser().parse_args(["make", str(photo), "--focus", "centre"]))
    assert job.min_patch_subject is None and job.subject_patch == forge.MIN_PATCH_SUBJECT
    assert job.smooth_background == forge.SMOOTH_BACKGROUND
    argv = ["--focus", "centre", "--min-patch-subject", "0.3", "--smooth-background", "0"]
    job = job_from(parser().parse_args(["make", str(photo), *argv]))
    assert (job.min_patch_subject, job.smooth_background) == (0.3, 0)


def test_focus_reaches_the_job(photo):
    from lithophane_forge import focus, job_from, parser

    job = job_from(parser().parse_args(["make", str(photo), "--focus", "centre"]))
    assert (job.focus, job.focus_floor) == ("centre", focus.FLOOR)
    args = ["make", str(photo), "--focus", str(photo), "--focus-floor", "0.05"]
    job = job_from(parser().parse_args(args))
    assert job.focus == str(photo) and job.focus_floor == 0.05
    assert job_from(parser().parse_args(["make", str(photo)])).focus is None


def test_subject_needs_the_extra(photo, capsys, monkeypatch, tmp_path):
    from lithophane_forge import vision

    monkeypatch.setattr(vision, "available", lambda: False)
    for argv in (
        ["make", str(photo), "-o", str(tmp_path)],
        ["analyze", str(photo), "--colors", "2"],
    ):
        assert main([*argv, "--focus", "subject"]) == 2
        assert "needs ONNX Runtime: uv sync --extra vision" in capsys.readouterr().err


def test_device_reaches_the_job(photo, fake_subject):
    from lithophane_forge import job_from, parser

    argv = ["make", str(photo), "--focus", "subject"]
    assert job_from(parser().parse_args([*argv, "--device", "cpu"])).device == "cpu"
    assert job_from(parser().parse_args(argv)).device == "auto"


def test_analyze_finds_the_subject(photo, capsys, fake_subject):
    argv = ["analyze", str(photo), "--colors", "2", "--focus", "subject", "--device", "cpu"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "was found by u2net_human_seg, modnet, isnet on the CPU in 0.2 s" in out
    assert out.count("on the subject") == 2 and fake_subject == ["cpu"]


def test_models_command(capsys, monkeypatch, tmp_path):
    from lithophane_forge import vision

    monkeypatch.setenv(vision.FOLDER_ENV, str(tmp_path))
    monkeypatch.setattr(vision, "available", lambda: False)
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert f"model folder: {tmp_path} ($LITHOPHANE_FORGE_MODELS)" in out
    assert "ONNX Runtime: not installed (uv sync --extra vision" in out
    assert out.count(" missing ") == 3 and "--download fetches them" in out

    def offline(model, path):
        raise ValueError(f"cannot download {model.url}")

    monkeypatch.setattr(vision, "fetch", offline)
    assert main(["models", "--download"]) == 1
    assert capsys.readouterr().err.count("cannot download") == 3


def test_analyze_with_focus(photo, capsys):
    assert main(["analyze", str(photo), "--colors", "2", "--focus", "10,10,60,40"]) == 0
    out = capsys.readouterr().out
    assert out.count("on the subject") == 2  # the ideal colours and the filaments
