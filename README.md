# 🖼️ lithophane-forge

Turn a photo into a 3D-printable lithophane, with a ready-to-print PrusaSlicer project for your Prusa printer (Original Prusa MK4 and white PETG by default). It also makes colour lithophanes on a printer with one extruder: it finds the ideal colours for the photo, suggests the closest real PETG filaments and plans the filament changes. Everything runs on the command line, on the CPU, in a few seconds; no GPU and no neural network are needed.

## How it works

- The photo's lightness is cleaned up (levels, optional local contrast, gamma) and turned into the light you want to see.
- Light through plastic falls off exponentially (Beer-Lambert), so each point gets the thickness that lets the right amount of light through: thin is bright, thick is dark.
- numpy turns the thickness map into a closed mesh with a frame: a flat panel, a curved one that stands on its own, a cylinder for a lamp, or a shaped panel (heart, oval, circle, or your own outline).
- Flat, curved and cylinder panels print standing, for the finest detail. Under every thick part, the plastic slopes down at no more than 60° from vertical, so they print without supports. Shaped panels print lying on the bed, so their outline needs no support either.
- build123d makes the plain parts: the frame of a shaped panel, a hanging tab, a stand for flat panels and a lamp base with a pocket for an LED.
- The 3MF is a PrusaSlicer project: the printer, print and filament presets are the official Prusa ones, and the object carries its own settings (100 % infill, enough perimeters, rear seam, brim).
- With `--slice` you also get the G-code (binary `.bgcode` on the MK4), ready for the printer.
- `calibrate` prints a strip of known thicknesses, so you can measure your own spool.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.13 or 3.14 (uv installs it if needed).
- PrusaSlicer 2.9 (optional, but needed for the full project and for `--slice`). The tool finds `prusa-slicer` on the PATH; otherwise set `PRUSA_SLICER`, for example `PRUSA_SLICER="flatpak run --filesystem=host --filesystem=/tmp --command=prusa-slicer com.prusa3d.PrusaSlicer"`.

```sh
uv sync
uv run lithophane-forge analyze photo.jpg                 # advice before printing
uv run lithophane-forge make photo.jpg --width 120        # flat panel, MK4, white PETG
uv run lithophane-forge make photo.jpg --shape curved --height 100 --arc 110
uv run lithophane-forge make photo.jpg --width 120 --stand --hole
uv run lithophane-forge make photo.jpg --shape cylinder --diameter 80 --repeat 2 --lamp-base
uv run lithophane-forge make photo.jpg --shape shaped --mask heart --width 90 --hole
uv run lithophane-forge make photo.jpg --shape shaped --mask outline.svg --width 100
uv run lithophane-forge analyze photo.jpg --colors 4             # ideal colours, no files
uv run lithophane-forge make photo.jpg --mode color --colors 4 --width 120 --slice
uv run lithophane-forge make photo.jpg --mode color --filaments examples/filaments.toml
uv run lithophane-forge calibrate --slice                 # measure your white filament
uv run lithophane-forge make photo.jpg --printer mk4s --slice
uv run lithophane-forge make photo.jpg --height 80 --gamma 0.8 --clahe 0.01
uv run lithophane-forge printers                          # supported printers and presets
```

## Output

Everything goes to `out/<name>/`, where `<name>` is the photo's name (or `--name`).

| File | What it is |
|---|---|
| `<name>.3mf` | PrusaSlicer project: model, presets, per-object settings, thumbnail |
| `<name>.bgcode` | G-code for the printer (only with `--slice`; `.gcode` on printers without binary G-code) |
| `<name>.stl` | The bare mesh, for other slicers (frame and tab are separate shells that overlap the panel) |
| `<name>-stand.3mf`, `<name>-lamp-base.3mf` | The accessories, as their own projects (with `--stand` or `--lamp-base`), plus STL files |
| `preview.png` | The photo, the lit panel (simulated) and the relief side |
| `report.md` | What to print with, and how to light it |
| `plan.json` | Every parameter and result, for tools |

## Options

- **Size and shape:** `--width` and `--height` are the outer size in mm, frame included; give one and the photo sets the other, or both to crop the photo. `--frame` sets the frame width (0 for none).
- **Curved:** `--shape curved` bends the panel on an arc of `--arc` degrees, 10 to 300 (the width is the arc length on the outside).
- **Cylinder:** `--shape cylinder --diameter D --height H` (the height is 1.25 × the diameter if you leave it out). The picture goes once around; a normal photo then loses its top and bottom, so `--repeat 2` puts it around twice (the tool suggests the best number). `--lamp-base` adds a base with a pocket for an LED tealight or puck light (`--led-diameter`, `--led-height`) and a notch for a cable.
- **Shaped:** `--shape shaped --mask heart|oval|circle`, or `--mask` with an SVG file (its largest closed shape) or a PNG file (dark pixels are the shape). An oval takes the photo's proportions, a circle stays round. The frame follows the outline.
- **Extras:** `--hole` adds a hanging tab with a 5 mm hole (flat and shaped panels); `--stand` makes a stand with a slot for a flat panel.
- **Colour:** `--mode color --colors N` (2 to 8 filaments), `--filaments FILE` for your own list, `--t-min`/`--t-max` for the height range of the stack (0.6 to 3.0 mm by default). See below.
- **Picture:** `--crop X,Y,W,H` (photo pixels; the panel takes the crop's shape unless you give both sizes), `--levels LOW,HIGH` (percentiles that become black and white), `--gamma`, `--clahe` (local contrast), `--invert` (a negative in lightness), `--view relief` (the relief faces the viewer instead of the flat face).
- **Thickness:** `--t-min`, `--t-max` and `--td` (mm of your filament that let 10 % of the light through). The defaults for white PETG (0.8, 3.6 and 2.3 mm) are estimates: calibrate your spool for the best result.
- **Printer:** `--printer` (`mk4`, `mk4s`, `mk3.5`, `mk3s`, `coreone`, `coreonel`, `xl`, `mini`), `--material petg|pla`, or your own presets with `--printer-preset`, `--print-preset` and `--filament-preset`.
- **Detail:** `--pitch` (mm between grid points, 0.2 by default; a 0.4 mm nozzle gains nothing below that) and `--max-triangles`.

## Printing

- Use white filament. PETG lets more light through than PLA, so its thickness range is wider.
- Standing panels print with a brim. Shaped panels print lying, flat face down: that face is the one you look at, so a smooth or satin sheet gives the cleanest result.
- On the MK4, print PETG on the textured sheet, or use a release agent on the smooth sheet, and keep the filament dry.
- Print the stand and the lamp base in any opaque colour; they use 0.2 mm layers.
- Light it from behind with a diffuse, neutral white LED (about 4000 K). By default the flat face looks at you.

## Colour lithophanes

A printer with one extruder changes colour only between layers: the printer stops (M600), you load the next filament, and it goes on. So a colour lithophane prints lying on the bed, and every point is a stack: the first filament from the bed up, then the second from the first change, and so on, cut at the point's own height. Lit from behind, the light crosses the whole stack.

- **One path of colours.** The colours a point can show lie on one path, from the first filament alone to the whole stack, from light to dark. A sunset, a portrait or a sepia photo fits such a path well; a photo with red and blue of the same lightness side by side does not. The tool tells you how close it gets (mean ΔE, CIEDE2000) and shows the simulated result in `preview.png`.
- **Ideal colours first.** The tool looks for the colours and the change heights that bring the path closest to the photo, with no filament list in mind: k-means in CIELAB gives a start, then an optimiser moves the colours, how translucent each one is, and the heights. The base stays white or natural and translucent: it lies behind every point. `analyze --colors N` shows the ideal colours without making any file.
- **Then real filaments.** For every ideal colour it suggests the Prusament PETG filament that makes the whole print closest to the photo, using the hex codes and TD (transparency) values that Prusa publishes, and fits the change heights again for them. With `--filaments FILE` it picks from your own list instead (see `examples/filaments.toml`).
- **The project.** The 3MF carries the colour changes, so PrusaSlicer shows them and the MK4 stops at the right layers. `report.md` lists which filament to load at which height.

## Calibrate your filament

```sh
uv run lithophane-forge calibrate --slice
```

The strip has twelve steps from 0.6 to 5 mm, printed standing like a lithophane, with each thickness written in thin plastic under its step (it glows when lit). Hold it against your light: the first step that looks black is `--t-max`, about 0.55 × that is `--td`, and the thinnest even step is `--t-min`. `report.md` explains it again next to the files.

## Development

```sh
uv run pytest                     # add -m "not slicer" to skip the PrusaSlicer runs
uv run ruff check . && uv run ruff format --check .
```

GitHub Actions runs ruff and the tests on every push and pull request.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
