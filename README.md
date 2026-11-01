# 🖼️ lithophane-forge

Turn a photo into a 3D-printable lithophane, with a ready-to-print PrusaSlicer project for your Prusa printer (Original Prusa MK4 and white PETG by default). It also makes colour lithophanes on a printer with one extruder: it finds the ideal colours for the photo, picks real PETG filaments for them and plans the filament changes. Everything runs on the command line, on the CPU, in a few seconds. You need no GPU and no neural network.

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

## Step by step

From a photo on your computer to a lit lithophane on your shelf. The examples use an Original Prusa MK4 and white PETG, the defaults.

### 1. Install

Install [uv](https://docs.astral.sh/uv/) and [PrusaSlicer](https://github.com/prusa3d/PrusaSlicer/releases) 2.9, then get the tool:

```sh
git clone https://github.com/amedeos/lithophane-forge.git
cd lithophane-forge
uv sync
uv run lithophane-forge printers                          # checks PrusaSlicer's presets
```

If `printers` cannot find PrusaSlicer's profiles, set `LITHOPHANE_FORGE_PROFILES` to the folder that holds `PrusaResearch.ini`. If PrusaSlicer is not on the PATH (for example the Flatpak), set `PRUSA_SLICER` as shown under Requirements.

### 2. Calibrate your white filament (once per spool)

```sh
uv run lithophane-forge calibrate --slice
```

Print `out/calibration-petg/calibration-petg.bgcode` (about 1 hour 20 minutes on the MK4), hold the strip against the light you will use, and follow `out/calibration-petg/report.md`. Write down three numbers: `--t-min`, `--t-max` and `--td`. Without them the tool uses estimates for white PETG, which work, but your own numbers give truer tones.

### 3. Choose a photo and check it

Copy the photo into `photos/` (git ignores it) and ask for advice:

```sh
uv run lithophane-forge analyze photos/lake.jpg
```

Good photos have a clear subject, good contrast and few large black areas: black prints at full thickness and shows no detail. A lithophane shows only lightness, so a blue sky and green grass of the same lightness look the same. `analyze` warns you about these cases.

### 4. Make the lithophane

```sh
uv run lithophane-forge make photos/lake.jpg --width 120 --t-min 0.8 --t-max 3.4 --td 2.1 --slice
```

Use your own numbers from step 2. `--width` or `--height` sets the size in mm; `--shape curved`, `cylinder` or `shaped` change the form (see Options). Everything lands in `out/lake/`.

### 5. Look before you print

Open `out/lake/preview.png`: it shows the photo, the panel lit from behind (simulated) and the relief. If the picture is too dark, try `--gamma 0.8`; if it is flat, try `--clahe 0.01`; to keep only part of the photo, use `--crop X,Y,W,H`. Run `make` again until you like the preview: with `--slice` it takes about a minute.

### 6. Print

Copy `out/lake/lake.bgcode` to the printer (USB stick, PrusaLink or Prusa Connect), or open `out/lake/lake.3mf` in PrusaSlicer to check it first. Load white PETG. A flat, curved or cylinder panel prints standing with a brim; a 120 mm panel takes about 3 to 5 hours. `out/lake/report.md` sums up the settings and the estimate.

### 7. Light it

Remove the brim with care and put a diffuse, neutral white LED (about 4000 K) behind the panel. By default the flat face looks at you.

### 8. Try colour (optional)

```sh
uv run lithophane-forge analyze photos/sunset.jpg --colors 4
uv run lithophane-forge make photos/sunset.jpg --mode color --colors 4 --width 120 --slice
```

Sunsets, portraits and sepia photos work best (see Colour lithophanes). `out/sunset/report.md` lists the filament to load first and the height of every change. The printer stops at each change (M600) and asks you to load the next filament. The panel prints lying, in about 2 hours for 120 mm.

## Output

Put the photos you want to try in `photos/`: git ignores everything in it, so your own photos never end up in the repository. Everything the tool makes goes to `out/<name>/` (also ignored), where `<name>` is the photo's name (or `--name`).

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
- **Colour:** `--mode color --colors N` (2 to 8 filaments), `--filaments FILE` for your own list, `--t-min`/`--t-max` for the height range of the stack (0.6 to 3.0 mm by default), `--allow-abrasive` to also use filaments that need a hardened steel nozzle. See below.
- **Picture:** `--crop X,Y,W,H` (photo pixels; the panel takes the crop's shape unless you give both sizes), `--levels LOW,HIGH` (percentiles that become black and white), `--gamma`, `--clahe` (local contrast), `--invert` (a negative in lightness), `--view relief` (the relief faces the viewer instead of the flat face).
- **Thickness:** `--t-min`, `--t-max` and `--td` (mm of your filament that let 10 % of the light through). The defaults for white PETG (0.8, 3.6 and 2.3 mm) are estimates: calibrate your spool for the best result.
- **Printer:** `--printer` (`mk4`, `mk4s`, `mk3.5`, `mk3s`, `coreone`, `coreonel`, `xl`, `mini`), `--material petg|pla`, or your own presets with `--printer-preset`, `--print-preset` and `--filament-preset`.
- **Detail:** `--pitch` (mm between grid points, 0.2 by default; a 0.4 mm nozzle gains nothing below that) and `--max-triangles`.

## Printing

- Use white filament. PETG lets more light through than PLA, so its thickness range is wider.
- Standing panels print with a brim. Shaped and colour panels print lying, flat face down. By default you look at that face, so a smooth or satin sheet gives the cleanest result.
- On the MK4, print PETG on the textured sheet, or use a release agent on the smooth sheet, and keep the filament dry.
- Print the stand and the lamp base in any opaque colour; they use 0.2 mm layers.
- Light it from behind with a diffuse, neutral white LED (about 4000 K). By default the flat face looks at you.

## Colour lithophanes

A printer with one extruder changes colour only between layers: the printer stops (M600), you load the next filament, and it goes on. So a colour lithophane prints lying on the bed, and every point is a stack of filaments. The first filament starts on the bed, the second starts at the first change, and so on; each point stops at its own height. Lit from behind, the light crosses the whole stack.

- **One path of colours.** The colours a point can show lie on one path, from the first filament alone to the whole stack, from light to dark. A sunset, a portrait or a sepia photo fits such a path well; a photo with red and blue of the same lightness side by side does not. The tool tells you how close it gets (mean ΔE, CIEDE2000) and shows the simulated result in `preview.png`.
- **Ideal colours first.** The tool looks for the colours and the change heights that bring the path closest to the photo, with no filament list in mind. k-means in CIELAB gives a start. Then an optimiser moves the colours, how translucent each one is, and the heights. The base stays white or natural and translucent, because it lies behind every point. `analyze --colors N` shows the ideal colours without making any file.
- **Then real filaments.** It picks the Prusament PETG filaments that make the whole print closest to the photo, using the hex codes and TD (transparency) values that Prusa publishes, and fits the change heights again for them. They can differ from the filaments closest in colour to the ideal ones, which `report.md` also lists. With `--filaments FILE` it picks from your own list instead (see `examples/filaments.toml`). If fewer filaments do just as well, the plan uses fewer and says so. Abrasive filaments, such as Prusament PETG Tungsten 75 %, wear out a brass nozzle, so the tool leaves them out unless you pass `--allow-abrasive` (for a hardened steel nozzle).
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

GitHub Actions runs ruff, the tests on Python 3.13 and 3.14, and a slicing check with PrusaSlicer on every pull request and on every push to `main`.

## Contributing

- Work on a feature branch (`feature/<topic>`) and open a pull request to `main`, for every change, small fixes included. A pull request is merged when CI is green; nobody commits to `main` directly.
- Write everything in the repository in English at B2/C1 level: code comments, docs, commit messages and pull requests. Use short sentences and common words.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
