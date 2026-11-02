# AGENTS.md

Guidance for coding agents working in this repository.

## Project

`lithophane-forge` turns a photo into a printable colour picture or lithophane and a ready-to-print PrusaSlicer project. It follows the conventions of the owner's earlier project `/workspace/3d-linux` (uv, build123d, one CLI, a numpy renderer, PrusaSlicer CLI for projects).

- **Printer:** Prusa MK4 (Input Shaper firmware) with **one extruder**: no MMU. Colours come only from M600 colour changes by layer height. Other Prusa printers are in `printers.PRINTERS`.
- **Material:** **PETG** by default (`--material pla` for PLA). The thickness defaults in `optics.DEFAULTS` are estimates until the owner calibrates a spool.
- **Advice:** `image.analyse(..., colour=True)` gives advice for colour prints (no thickness claims, no `--clahe`, no warning about colourful photos); `make` passes the mode, `analyze` takes `--mode` (colour by default). `analyze --light` without `--colors`, or `--colors`/`--light` with `--mode mono`, are usage errors.
- **Modes:** colour is the **default** on the command line for flat and shaped panels (`job_from` picks it when `--mode` is not given; `Job` itself still defaults to mono, which the tests rely on). Mono (`--mode mono`, white filament): flat, curved and cylinder panels print **standing** (thickness along y or radial), shaped panels print **lying** (thickness along z); curved and cylinder are always mono. Colour: M600 changes, flat or shaped, printed **lying**. **Light:** a colour print is by default a picture seen in **room light, from the top** (`--light front`, `view="relief"`, not mirrored; `job_from` picks it, `Job` itself defaults to `light="back"` like the mode); `--light back` makes a colour lithophane, lit from behind (default `view="flat"`). Mono is always backlit. The owner asked for front light as the default on 2026-11-02, because they want to look at the print like a photo, with no lamp. The ideal colours come from the photo alone and only then are matched to real PETG (Prusa's published hex and TD values) or to `--filaments`. The colour analysis must not depend on the filaments the owner has; owned filaments come in only through that explicit option.
- **License:** GPL-3.0-only.

## Language

- **Always** write in **English at B2/C1 level** everything that goes into the repository or to GitHub: README, AGENTS.md, docstrings, comments, help and error messages, test names, commit messages, pull request titles and descriptions. Use clear, short sentences and common words.
- In Markdown, write each paragraph and each list item on **one line** (no hard wraps).
- The owner chats in Italian: answer the owner in Italian, but keep everything above in English.

## Environment and commands

- Python 3.14 (3.13 also works), [uv](https://docs.astral.sh/uv/), numpy, Pillow, scikit-image, build123d 0.13 (accessories, later), PrusaSlicer 2.9 (optional).

```sh
uv sync
uv run pytest                     # ~90 s; -m "not slicer" skips the PrusaSlicer runs
uv run ruff check . && uv run ruff format --check .
uv run lithophane-forge analyze PHOTO [--mode mono]   # advice for a colour print (or mono)
uv run lithophane-forge make PHOTO [--colors N] [--filaments FILE] [--width MM] [--slice]   # colour picture, room light, lying
uv run lithophane-forge make PHOTO --light back [--colors N] [--slice]   # colour lithophane, lit from behind
uv run lithophane-forge make PHOTO --mode mono [--shape flat|curved|cylinder] [--slice]   # white, standing
uv run lithophane-forge analyze PHOTO --colors N [--light back]
uv run lithophane-forge calibrate [--slice]
uv run lithophane-forge printers
```

- Test photos go in `photos/` (git-ignored, only `.gitkeep` is tracked): the owner's personal photos never go into the repository.
- Output goes to `out/<name>/` (git-ignored). `forge.clear_outputs` first removes what an earlier run wrote there under the same name (only the kinds of file we write, never the photo), so no stale G-code stays next to a new project: `<name>.stl`, `<name>.3mf` (project), `<name>.bgcode` with `--slice`, `preview.png`, `plan.json`, `report.md`.
- `$LITHOPHANE_FORGE_PROFILES` names the PrusaResearch.ini (or its folder) when it is not in `profiles.SEARCH` (system install, PrusaSlicer's data folder, Flatpak). `$PRUSA_SLICER` names the PrusaSlicer command; a value that names no command is a usage error, not "PrusaSlicer not found".

## Layout

| File | Role |
|---|---|
| `src/lithophane_forge/__init__.py` | The `lithophane-forge` command (argparse). Usage errors exit with 2. `EXAMPLES` and the README's `sh` blocks are tested |
| `printers.py` | `Printer` table (preset names per printer and material, 0.4 mm nozzle) and `setup()`, which checks the names against the installed bundle and reads bed, height limit, layer heights and `binary_gcode` |
| `profiles.py` | Finds and parses PrusaResearch.ini, resolves `inherits` |
| `image.py` | Load (EXIF), crop to the aspect, resize, tone (levels in L*, CLAHE, gamma) → linear light; `analyse()` gives advice |
| `optics.py` | Beer-Lambert: thickness ↔ light. `td` = mm that let 10 % through |
| `mesh.py` | Thickness grid → closed mesh (`shell`), mappings `standing`, `lying`, `arc` (also closed cylinder); `masked` / `lying_masked` for shaped panels (`connect_diagonals` first); binary STL |
| `forge.py` | The `make` pipeline: `Job` (and `check()`), `layout()` (sizes, grid, picture box per shape), `thickness_map`, overhang limit, `volumes()` (the panel object: lithophane, frame, tab), `place()`, accessories, files, slicing, `plan.json`, `report.md` |
| `shapes.py` | Outlines for shaped panels: heart, oval, circle, SVG (svgelements, y flipped), PNG (dark = inside); `fit`, `size`, `cells_inside` |
| `accessories.py` | build123d parts: `frame_ring`, `tab` (rectangle with a round top and a hole clear of the outline), `stand`, `lamp_base`; `to_mesh` merges the tessellation into a closed mesh; `remap` swaps axes |
| `threemf.py` | Our own 3MF writer: multi-volume objects, per-object settings, M600 list, thumbnail, full config |
| `slicer.py` | PrusaSlicer CLI: temporary `--datadir`, slice, summary, config from G-code |
| `render.py` | Simulated backlit view, relief hillshade, simulated colour print, palette strip, preview sheet, thumbnail |
| `palette.py` | Colour mode: light models of a stack (`Backlight`: per-channel Beer-Lambert from hex and td; `Frontlight`: `coverage` of each filament over the one below), `Fitter` (ideal colours, td and change heights with Powell; `choose` picks real filaments by their effect on the whole print), `plan()`, `assign()`, CIELAB helpers |
| `data/petg_catalog.toml` | Prusament PETG hex and TD values from Prusa's HueForge table (source URL inside). `abrasive = true` marks filaments that need a hardened steel nozzle (Tungsten 75 %); `palette.choices` leaves them out unless `--allow-abrasive` |
| `calibrate.py` | The calibration strip: steps of known thickness with glowing labels (0.8 mm glyphs in a 2 mm band, stamped after the overhang limit), same settings as a lithophane |
| `examples/filaments.toml` | Example `--filaments` list: the owner's shelf from 3d-linux, td estimated |
| `tests/` | pytest suite. Keep it green. `slicer`-marked tests need PrusaSlicer |

## How a lithophane is built

- **Light.** The photo's lightness (L*) gets levels, optional CLAHE and gamma, then becomes linear light. `Optics.thickness` maps light to thickness with Beer-Lambert between `t_min` (brightest) and `t_max` (darkest).
- **Grid.** Points are `pitch` apart (0.2 mm); the outer size is exact (`linspace`). Above `max_triangles` the pitch grows and a note says so. The frame (`frame` mm wide, `t_max + 0.8` thick) is part of the grid; the photo fills the inside.
- **Mesh.** `mesh.shell` joins the relief face (every grid point) and the flat face (only its edge points: long strips, and fans at the two end strips). About two triangles per grid point. Always closed, positive volume.
- **Overhangs.** A standing panel has ledges on the relief side where a thick part lies above a thin one. `forge.limit_overhangs` thickens the plastic below so that nothing leans out more than `max_overhang` (60° from vertical). PrusaSlicer's MK4 presets want support beyond 50°; the test slices with the threshold at our limit + 5° and expects no support.
- **Viewing side.** For lithophanes the flat face looks at the viewer by default and the light goes behind the relief. A colour picture in room light (`light="front"`) shows its top face (`view="relief"`; `Job.check` refuses `view="flat"` with it). The tests check that the picture reads the right way round from the viewer's side, for both views and every shape.
- **Shaped panels.** The grid cells whose centre lies inside the outline inset by half the frame width form the lithophane (`mesh.lying_masked`): its edge is stepped. The frame (build123d, the full frame width inside the outline) is a second volume of the same object and covers the steps; PrusaSlicer joins the two. Standing shaped panels would need support under the lower half of the outline, so they print lying.
- **Cylinder.** The grid wraps around (`mesh.arc(wrap=True)`); the frame is only at the top and bottom (rims). The middle of the picture faces −y, the seam is at the back. The default height is 1.25 × the diameter; `cylinder_notes` says how much of the photo is cropped and which `--repeat` fits it.
- **Accessories.** Stand (slot `frame_t + 2 × 0.10` mm, as deep as the frame) and lamp base (recess `diameter + 2 × 0.30` mm, LED pocket, cable notch) are separate 3MF projects with `layer_height = 0.2` as an object setting.

## How a colour print is chosen

- **Model, light from behind (`Backlight`).** A filament with linear colour c and transmission distance td has per-channel attenuation a = ln(10) / td × ln(c) / ln(Y(c)): a grey lets 10 % through at t = td on every channel, a coloured filament passes its own colour. A point of height h holds filament k between `starts[k]` and `starts[k+1]`; the light through it is exp(−Σ d_k a_k). The viewer's white is light through the thinnest stack of white PETG (td 3.4).
- **Model, light from the front (`Frontlight`).** From the bed up, each filament moves the colour seen (in sRGB) from the colour under it toward its own hex by `coverage(d, td)` = min(1, sqrt(d / (0.1 td))): white PETG hides black from 0.34 mm. This is AutoForge's calibrated curve (https://github.com/hvoss-techfak/AutoForge, CC BY-NC-SA: we use the idea, not its code), without its slower curve for dark on light; it is an estimate until a test print checks it. Prusa's TD table was measured for HueForge, that is for this kind of print. The base must hide what is behind the panel (td ≤ thinnest stack / 0.1, `Frontlight.is_base`); behind it the model puts black. The viewer's white is Signal White, thick (its luminance). The model computes all layers at once: filament k shows with weight cover_k × Π_{j>k}(1 − cover_j) (a test checks it against laying the filaments on one by one). With 0.1 mm layers a filament has only about 3 or 4 tones before it hides what is under it.
- **First filament notes (`forge.base_notes`).** `choose` takes the base from the filaments that can be one (`is_base` of the light model) and falls back to any filament when the list has none: then a note says so (in room light: the wall shows through; lit from behind: the base tints every point). In room light a thin `--t-min` can also rule out every light filament (at 0.3 mm the limit is TD 3, Signal White has 3.4): a note names the `--t-min` that allows the lightest one.
- **Fit.** `Fitter.cost` is the mean weighted CIELAB distance (lightness × 1.5) from 3000 sampled pixels to the path sampled at every layer top. `ideal()` moves the base (light from behind: grey level, td ≥ 2; light from the front: any colour, td up to `Frontlight.base_td`) and colour and td of the others, plus the change heights; it never returns something worse than its start. It never changes the order of the filaments, so `plan()` runs it from two k-means starts, lightest first, and with light from the front also darkest first. Its heights are free and can lose quality when they are snapped to layer tops; in room light it also stops early (a run thicker than 0.1 td does not depend on its td, so the cost is flat there; on the owner's Machu Picchu photo the ideal was once worse than the print, 9.9 against 8.7). So `plan()` also runs `choose` on the photo's own colours (`photo_colours`: 8 k-means colours and 4 neutral greys, each at 0.5 and 1.5 × `ideal_td`), on layer tops, and refines that result with `ideal()`. The ideal is the best of all of them, compared with the changes on layer tops (it costs about 0.6 s per plan). The ideal colours depend on the photo alone (a test checks that two filament lists give the same ideal). Then `choose` picks the real filaments. It starts from the closest colours, with every change on a layer top. Then it takes one position at a time and tries every filament at every layer top between the changes below and above; it keeps the best pair and repeats until nothing changes. The changes never leave the layer tops, so every cost it compares is the cost of a plan that can print. (An earlier version chose filaments with the heights fixed and got stuck: a dark filament low in the stack hides every change above it. Another fitted the heights freely and then lost quality when they were snapped to layers.) `Fitter.costs` scores all candidates of a position in one pass, and the heights below the change under that position are computed once. The search for real filaments starts from the ideal colours and from the first two k-means starts (seeds 0 and 1 lit from behind, both orders of seed 0 in room light), and keeps the best. The real print is usually worse than the ideal, but nothing guarantees it. `kmeans_lab` asks scipy for no more clusters than the photo has colours and silences its warnings: they reached the terminal. `report.md` lists, for every ideal colour, the closest filament by colour alone.
- **Heights.** Changes sit on layer tops below the top layer; the M600 is at the top of the first layer of the new filament (`Fit.colour_changes`). `_on_layers` drops filaments that no point holds (a change snapped to the top layer, or two changes at one height), so a plan may use fewer filaments than `--colors`. `assign()` gives every pixel its height (same weighted distance), then a 3 × 3 median filter smooths the surface. The frame is the whole stack (`color_range[1]`): it shows the last colour of the path (the darkest lit from behind, the top filament in room light).
- **Checks.** `tests/test_slicer.py` slices a colour project and compares the G-code's M600 heights with `plan.json`.

## Frames

- **Panel frame:** u along the width (left to right as the viewer sees it), v up, w through the thickness (0 on the flat face).
- **Standing (flat):** u → x, v → z, w → +y. The viewer of the flat face stands at −y.
- **Curved:** bent around the z axis, the flat face on the outer radius, the middle facing −y. The width is the arc length on the flat face.
- **Lying (colour and shaped panels):** u → x, v → y, w → z. Seen from below (the bed face), left and right swap: with `view="flat"` all volumes are mirrored in x. With `view="relief"` (always for light from the front) nothing is mirrored: PrusaSlicer's top view shows the picture as the viewer sees it.
- **Cylinder:** like curved, closed: u from −π r to π r around the z axis.
- `thickness_map` returns the grid as the viewer sees it (row 0 at the top); `panel()` turns it into mesh order (row 0 at the bottom; for standing panels the columns are swapped for `view="relief"`, while lying panels are mirrored in x for `view="flat"` and left as they are for `view="relief"`).
- `Job.view` is None by default and `__post_init__` sets it from the light: `"relief"` for `light="front"`, else `"flat"`. `Job.front_lit` and `Job.part_name` hold the front-light wording in one place.

## 3MF and PrusaSlicer notes

- We write the 3MF ourselves (numpy → XML). `Metadata/Slic3r_PE_model.config` gives each object its volumes (triangle ranges) and per-object settings. PrusaSlicer joins overlapping volumes of one object when it slices (checked by hand with two crossing boxes: the filament matches the union).
- Per-object settings keep the system presets clean (the one exception is the filament colour, below): the project shows "Original Prusa MK4 Input Shaper 0.4 nozzle", "0.10mm FAST DETAIL @MK4IS 0.4", "Prusament PETG @PGIS", plus the object's own perimeters, 100 % infill, rear seam and 8 mm brim.
- The CLI crashes on `--export-3mf` without a display. So `make` slices once to text G-code, copies the `; prusaslicer_config` block into `Metadata/Slic3r_PE.config` and sets `binary_gcode` back to the printer's value. `--slice` slices the project a second time to `.bgcode`. The auto-support probe and the "project carries the printer" check live in `tests/test_slicer.py`, not in every run.
- Every slice gets a temporary `--datadir` with the vendor bundle and a `PrusaSlicer.ini` that installs the printer model and the variant (nozzle) of the printer preset in use, so results do not depend on the user's PrusaSlicer settings. PrusaSlicer silently takes other presets when the model or variant is missing, so `slicer.check_presets` compares the G-code's preset names with the asked ones and fails loudly.
- 100 % infill needs `fill_pattern = rectilinear`: the presets' default pattern refuses 100 %.
- M600 entries need `bed_idx="0"` (PrusaSlicer 2.9), or the GUI silently drops them.
- The filament presets in the printer table all have `filament_colour = #FF8000` (orange; other Prusament presets in the bundle differ). `slice_project` passes `--filament-colour` to the first slice with `forge.first_colour`: the colour plan's first filament, or for white prints Signal White (`palette.white()`, from the catalog) in PETG and plain white in PLA. So the project and its G-code start with the filament you really load; the preset name does not change, but PrusaSlicer's GUI shows the filament preset as modified. Accessories keep the preset's colour. Without PrusaSlicer the 3MF has no configuration, so plan.json's `printer.filament_colour` is null.
- `load_filaments` accepts only `#RRGGBB` (or `RRGGBB`) and writes it as `#RRGGBB`: the same string goes to the preview, the M600 list and PrusaSlicer.
- Headless slicing makes no G-code thumbnail for the printer's screen. The 3MF has our own `Metadata/thumbnail.png`.

## Checks after any change

1. `uv run ruff check . && uv run ruff format --check .`
2. `uv run pytest` (all green; with PrusaSlicer installed the `slicer` tests slice every printer in the table).
3. `uv run lithophane-forge make photos/<photo> --slice` and look at `out/<name>/preview.png` and `report.md`.

## Git

- **Always** use a feature branch (`feature/<topic>`, from an up-to-date `main`) for every change, small fixes and documentation included. Never commit to `main` directly.
- Merge into `main` by pull request, with a merge commit (not squash), when CI (`.github/workflows/ci.yml`) is green.
- Stacked pull requests: after merging one, retarget the next to `main` (`gh pr edit N --base main`) before you delete the merged branch. Deleting the base branch of an open pull request closes it.
- Do not commit or push unless the owner asks.
- In commit messages, the co-author line is `Co-Authored-By: Claude Opus <noreply@anthropic.com>`, without a version number.
