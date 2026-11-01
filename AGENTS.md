# AGENTS.md

Guidance for coding agents working in this repository.

## Project

`lithophane-forge` turns a photo into a printable lithophane and a ready-to-print PrusaSlicer project. It follows the conventions of the owner's earlier project `/workspace/3d-linux` (uv, build123d, one CLI, a numpy renderer, PrusaSlicer CLI for projects).

- **Printer:** Prusa MK4 (Input Shaper firmware) with **one extruder**: no MMU. Colours come only from M600 colour changes by layer height. Other Prusa printers are in `printers.PRINTERS`.
- **Material:** **PETG** by default (`--material pla` for PLA). The thickness defaults in `optics.DEFAULTS` are estimates until the owner calibrates a spool.
- **Modes:** mono (white filament, printed **standing**, thickness along y) is done. Colour (M600 changes, printed **lying**, thickness along z, backlit, ideal colours found from the photo and then matched to real PETG) is planned, together with cylinder, shaped panels, build123d accessories and the calibration print. The colour analysis must not depend on the filaments the owner has; owned filaments come in only through an explicit option.
- **License:** GPL-3.0-only.

## Language

- Write all repository text in **English at B2/C1 level**: README, docstrings, comments. Use clear, short sentences and common words.
- In Markdown, write each paragraph and each list item on **one line** (no hard wraps).
- The owner chats in Italian.

## Environment and commands

- Python 3.14 (3.13 also works), [uv](https://docs.astral.sh/uv/), numpy, Pillow, scikit-image, build123d 0.13 (accessories, later), PrusaSlicer 2.9 (optional).

```sh
uv sync
uv run pytest                     # ~20 s; -m "not slicer" skips the PrusaSlicer runs
uv run ruff check . && uv run ruff format --check .
uv run lithophane-forge analyze PHOTO
uv run lithophane-forge make PHOTO [--shape flat|curved] [--width MM] [--height MM] [--slice]
uv run lithophane-forge printers
```

- Output goes to `out/<name>/` (git-ignored): `<name>.stl`, `<name>.3mf` (project), `<name>.bgcode` with `--slice`, `preview.png`, `plan.json`, `report.md`.
- `$LITHOPHANE_FORGE_PROFILES` names the PrusaResearch.ini (or its folder) when it is not in `profiles.SEARCH` (system install, PrusaSlicer's data folder, Flatpak). `$PRUSA_SLICER` names the PrusaSlicer command.

## Layout

| File | Role |
|---|---|
| `src/lithophane_forge/__init__.py` | The `lithophane-forge` command (argparse). Usage errors exit with 2. `EXAMPLES` and the README's `sh` blocks are tested |
| `printers.py` | `Printer` table (preset names per printer and material, 0.4 mm nozzle) and `setup()`, which checks the names against the installed bundle and reads bed, height limit, layer heights and `binary_gcode` |
| `profiles.py` | Finds and parses PrusaResearch.ini, resolves `inherits` |
| `image.py` | Load (EXIF), crop to the aspect, resize, tone (levels in L*, CLAHE, gamma) → linear light; `analyse()` gives advice |
| `optics.py` | Beer-Lambert: thickness ↔ light. `td` = mm that let 10 % through |
| `mesh.py` | Thickness grid → closed mesh (`shell`), mappings `standing`, `lying`, `arc` (also closed cylinder), binary STL |
| `forge.py` | The `make` pipeline: sizes, grid, frame, overhang limit, mesh, placement, files, slicing, `plan.json`, `report.md` |
| `threemf.py` | Our own 3MF writer: multi-volume objects, per-object settings, M600 list, thumbnail, full config |
| `slicer.py` | PrusaSlicer CLI: temporary `--datadir`, slice, summary, config from G-code |
| `render.py` | Simulated backlit view, relief hillshade, preview sheet, thumbnail |
| `tests/` | pytest suite. Keep it green. `slicer`-marked tests need PrusaSlicer |

## How a lithophane is built

- **Light.** The photo's lightness (L*) gets levels, optional CLAHE and gamma, then becomes linear light. `Optics.thickness` maps light to thickness with Beer-Lambert between `t_min` (brightest) and `t_max` (darkest).
- **Grid.** Points are `pitch` apart (0.2 mm); the outer size is exact (`linspace`). Above `max_triangles` the pitch grows and a note says so. The frame (`frame` mm wide, `t_max + 0.8` thick) is part of the grid; the photo fills the inside.
- **Mesh.** `mesh.shell` joins the relief face (every grid point) and the flat face (only its edge points: long strips, and fans at the two end strips). About two triangles per grid point. Always closed, positive volume.
- **Overhangs.** A standing panel has ledges on the relief side where a thick part lies above a thin one. `forge.limit_overhangs` thickens the plastic below so that nothing leans out more than `max_overhang` (60° from vertical). PrusaSlicer's MK4 presets want support beyond 50°; the test slices with the threshold at our limit + 5° and expects no support.
- **Viewing side.** By default the flat face looks at the viewer and the light goes behind the relief. The tests check that the picture reads the right way round from the viewer's side, for both views and both shapes.

## Frames

- **Panel frame:** u along the width (left to right as the viewer sees it), v up, w through the thickness (0 on the flat face).
- **Standing (flat):** u → x, v → z, w → +y. The viewer of the flat face stands at −y.
- **Curved:** bent around the z axis, the flat face on the outer radius, the middle facing −y. The width is the arc length on the flat face.
- **Lying (colour mode, later):** u → x, v → y, w → z. Seen from below (the bed face), left and right swap.
- `thickness_map` returns the grid as the viewer sees it (row 0 at the top); `panel()` turns it into mesh order (row 0 at the bottom, columns swapped for `view="relief"`).

## 3MF and PrusaSlicer notes

- We write the 3MF ourselves (numpy → XML). `Metadata/Slic3r_PE_model.config` gives each object its volumes (triangle ranges) and per-object settings. PrusaSlicer joins overlapping volumes of one object when it slices (checked by hand with two crossing boxes: the filament matches the union).
- Per-object settings keep the system presets clean: the project shows "Original Prusa MK4 Input Shaper 0.4 nozzle", "0.10mm FAST DETAIL @MK4IS 0.4", "Prusament PETG @PGIS", plus the object's own perimeters, 100 % infill, rear seam and 8 mm brim.
- The CLI crashes on `--export-3mf` without a display. So `make` slices once to text G-code, copies the `; prusaslicer_config` block into `Metadata/Slic3r_PE.config` and sets `binary_gcode` back to the printer's value. `--slice` slices the project a second time to `.bgcode`. The auto-support probe and the "project carries the printer" check live in `tests/test_slicer.py`, not in every run.
- Every slice gets a temporary `--datadir` with the vendor bundle and a `PrusaSlicer.ini` that installs the printer model and the variant (nozzle) of the printer preset in use, so results do not depend on the user's PrusaSlicer settings. PrusaSlicer silently takes other presets when the model or variant is missing, so `slicer.check_presets` compares the G-code's preset names with the asked ones and fails loudly.
- 100 % infill needs `fill_pattern = rectilinear`: the presets' default pattern refuses 100 %.
- M600 entries need `bed_idx="0"` (PrusaSlicer 2.9), or the GUI silently drops them.
- Headless slicing makes no G-code thumbnail for the printer's screen. The 3MF has our own `Metadata/thumbnail.png`.

## Checks after any change

1. `uv run ruff check . && uv run ruff format --check .`
2. `uv run pytest` (all green; with PrusaSlicer installed the `slicer` tests slice every printer in the table).
3. `uv run lithophane-forge make <photo> --slice` and look at `out/<name>/preview.png` and `report.md`.

## Git

Work on `feature/*` branches and merge them into `main` by pull request; CI (`.github/workflows/ci.yml`) must be green. Do not commit or push unless the owner asks. In commit messages, the co-author line is `Co-Authored-By: Claude Opus <noreply@anthropic.com>`, without a version number.
