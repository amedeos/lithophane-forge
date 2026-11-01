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
    """The documented commands parse and reach their handler (the work itself is skipped)."""
    calls = []
    for name in ("cmd_make", "cmd_analyze", "cmd_calibrate", "cmd_printers"):
        monkeypatch.setattr(lithophane_forge, name, lambda a, n=name: calls.append(n) or 0)
    args = [str(photo) if a.endswith((".jpg", ".png")) else a for a in argv[1:]]
    assert main(args) == 0
    assert calls


def test_make_writes_files(photo, tmp_path, no_slicer, capsys):
    assert (
        main(["make", str(photo), "--width", "40", "--pitch", "0.5", "-o", str(tmp_path / "o")])
        == 0
    )
    out = capsys.readouterr().out
    assert "triangles" in out and (tmp_path / "o" / "photo" / "photo.3mf").is_file()


def test_analyze(photo, capsys):
    assert main(["analyze", str(photo)]) == 0
    assert "lightness" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["make", "missing.jpg"], "no such photo"),
        (["make", "{photo}", "--t-min", "3", "--t-max", "2"], "t_min < t_max"),
        (["make", "{photo}", "--crop", "1,2,3"], "X,Y,W,H"),
        (["make", "{photo}", "--height", "500", "--pitch", "1"], "prints at most"),
    ],
)
def test_errors_exit_with_2(argv, message, photo, capsys, no_slicer, tmp_path):
    argv = [a.format(photo=photo) for a in argv] + ["-o", str(tmp_path)]
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_bad_option_exits_with_2():
    with pytest.raises(SystemExit) as e:
        main(["make", "x.jpg", "--printer", "nope"])
    assert e.value.code == 2
