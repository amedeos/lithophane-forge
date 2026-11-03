# 🖼️ lithophane-forge

Turn a photo into a colour picture or a lithophane for a Prusa printer with one extruder, with a ready-to-print PrusaSlicer project (Original Prusa MK4 and Prusament PETG by default). The tool finds the ideal colours for the photo, picks real PETG filaments for them and plans the filament changes. The panel prints lying on the bed, and the printer stops at each change so that you can load the next filament. By default the panel is a picture that you look at in room light, like a photo; `--light back` makes a colour lithophane instead, lit from behind. It also makes classic white lithophanes (`--mode mono`): flat and curved panels and lamp cylinders, printed standing. Everything runs on the command line, on the CPU, in a few seconds. You need no GPU and no neural network.

## How it works

- **Colour (the default).** A printer with one extruder changes colour only between layers, so the panel prints lying and every point is a stack of filaments cut at its own height. Each height shows one colour: in room light you see the top of the stack (the default), and lit from behind (`--light back`) the light crosses the whole stack. The tool finds the colours and the change heights that bring the print closest to the photo, then gives every point its height (see Colour prints).
- **Mono (`--mode mono`).** The photo's lightness is cleaned up (levels, optional local contrast, gamma) and turned into thickness in white filament: thin is bright, thick is dark. Light through plastic falls off exponentially (Beer-Lambert).
- numpy turns the height map into a closed mesh with a frame: a flat panel or a shaped one (heart, oval, circle, or your own outline). In mono there are also curved panels that stand on their own and cylinders for a lamp.
- Colour and shaped panels print lying on the bed and need no support. Mono flat, curved and cylinder panels print standing, for the finest tones. Under every thick part the plastic slopes down at no more than 60° from vertical, so they print without supports.
- build123d makes the plain parts: the frame of a shaped panel, a hanging tab, a stand for flat panels and a lamp base with a pocket for an LED.
- The 3MF is a PrusaSlicer project: the printer, print and filament presets are the official Prusa ones, the object carries its own settings, and in colour the filament changes are already in place.
- With `--slice` you also get the G-code (binary `.bgcode` on the MK4), ready for the printer.
- `calibrate` prints a strip of known thicknesses, so you can measure your white spool for mono prints.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.13 or 3.14 (uv installs it if needed).
- PrusaSlicer 2.9 (optional, but needed for the full project and for `--slice`). The tool finds `prusa-slicer` on the PATH; otherwise set `PRUSA_SLICER`, for example `PRUSA_SLICER="flatpak run --filesystem=host --filesystem=/tmp --command=prusa-slicer com.prusa3d.PrusaSlicer"`.
- For `--focus subject` (optional): ONNX Runtime, with `uv sync --extra vision` (the CPU, about 25 MB) or `uv sync --extra vision-gpu` (also an NVIDIA card with a driver for CUDA 13, about 1.5 GB more). Install one of the two, not both, and keep the option on every later `uv sync`: a plain `uv sync` removes ONNX Runtime again. The three models (about 380 MB) download on first use.

```sh
uv sync
uv run lithophane-forge analyze photo.jpg --colors 4      # advice and ideal colours, no files
uv run lithophane-forge analyze photo.jpg --mode mono     # advice for a white lithophane
uv run lithophane-forge make photo.jpg --width 120 --slice    # colour picture, lying, MK4, PETG
uv run lithophane-forge make photo.jpg --colors 5 --width 150
uv run lithophane-forge make photo.jpg --light back --width 120    # colour lithophane, lit from behind
uv run lithophane-forge make photo.jpg --filaments examples/filaments.toml
uv run lithophane-forge make photo.jpg --shape shaped --mask heart --width 90 --hole
uv run lithophane-forge make photo.jpg --shape shaped --mask outline.svg --width 100
uv run lithophane-forge make photo.jpg --width 120 --stand --hole
uv run lithophane-forge make photo.jpg --mode mono --width 120 --slice    # white, standing
uv run lithophane-forge make photo.jpg --mode mono --height 80 --gamma 0.8 --clahe 0.01
uv run lithophane-forge make photo.jpg --shape curved --height 100 --arc 110    # always mono
uv run lithophane-forge make photo.jpg --shape cylinder --diameter 80 --repeat 2 --lamp-base
uv run lithophane-forge calibrate --slice                 # measure your white filament
uv run lithophane-forge make photo.jpg --printer mk4s --slice
uv run lithophane-forge printers                          # supported printers and presets
uv run lithophane-forge models                            # the models of --focus subject
```

## Step by step

From a photo on your computer to a colour picture on your shelf. The examples use an Original Prusa MK4 and Prusament PETG, the defaults.

### 1. Install

Install [uv](https://docs.astral.sh/uv/) and [PrusaSlicer](https://github.com/prusa3d/PrusaSlicer/releases) 2.9, then get the tool:

```sh
git clone https://github.com/amedeos/lithophane-forge.git
cd lithophane-forge
uv sync
uv run lithophane-forge printers                          # checks PrusaSlicer's presets
```

If `printers` cannot find PrusaSlicer's profiles, set `LITHOPHANE_FORGE_PROFILES` to the folder that holds `PrusaResearch.ini`. If PrusaSlicer is not on the PATH (for example the Flatpak), set `PRUSA_SLICER` as shown under Requirements.

### 2. Choose a photo and check it

Copy the photo into `photos/` (git ignores it) and ask for advice and colours:

```sh
uv run lithophane-forge analyze photos/sunset.jpg --colors 4
```

You get advice on the photo's lightness, the ideal colours, the Prusament PETG filaments that come closest and how close the print gets (mean ΔE: below about 5 it is very close to the photo, above about 10 you see the difference). One extruder can show only the colours on one path, from the first filament to the last, so photos with a few related colours work best; a blue sky next to green grass and an orange roof does not fit one path. `analyze` plans a picture for room light, like `make`; add `--light back` for a lithophane, or use `analyze photo.jpg --mode mono` for advice on a white lithophane.

### 3. Make the picture

```sh
uv run lithophane-forge make photos/sunset.jpg --width 120 --slice
```

This makes a colour panel 120 mm wide with 4 filaments, printed lying, to look at in room light. `--light back` makes a colour lithophane instead. `--colors N` changes the number of filaments (2 to 8), `--shape shaped --mask heart` gives it a shape, `--hole` adds a hanging tab (see Options). Everything lands in `out/sunset/`.

### 4. Look before you print

Open `out/sunset/preview.png`: it shows the photo, the print with the ideal colours, the print with the suggested filaments (both simulated) and the stack of filaments with their change heights. Try `--colors 3` or `--colors 5`, `--gamma 0.8` for a darker photo, or `--crop X,Y,W,H` to keep only part of it. Run `make` again until you like the preview: with `--slice` it takes about a minute.

If the photo has a subject, such as people in front of a landscape, the colours can go to it first. With the `vision` extra the tool finds it by itself; without it, paint a mask:

```sh
uv sync --extra vision                                     # ONNX Runtime for the CPU
uv run lithophane-forge make photos/sunset.jpg --width 120 --focus subject
uv run lithophane-forge make photos/sunset.jpg --width 120 --focus photos/sunset-mask.png
```

`--focus subject` finds the people in the photo, or its main object when there are none (an open landscape has none: then it says so). Paint a mask over the photo in an image editor: white where the subject is, black (or transparent) for the rest. `--focus X,Y,W,H` (a box in photo pixels) and `--focus centre` need neither. The map goes to `out/sunset/focus.png`, and the preview outlines the subject on the photo: if the models marked too much or too little, copy `focus.png` next to the photo, fix it and pass it with `--focus` (see Colour prints).

### 5. Get the filaments

`out/sunset/report.md` lists the filaments in printing order, and the ideal colour next to each. If you want to use the spools you already have, write them into a TOML file like `examples/filaments.toml` (name, hex colour, TD) and pass `--filaments yours.toml`: the plan then picks only from your list.

### 6. Print

Copy `out/sunset/sunset.bgcode` to the printer (USB stick, PrusaLink or Prusa Connect), or open `out/sunset/sunset.3mf` in PrusaSlicer to check it first. Load the first filament in `report.md`. The panel prints lying, flat face down, in a few hours (PrusaSlicer's estimate is in `report.md`). At each change the printer stops (M600) and asks for the next filament: load the one `report.md` names for that height.

### 7. Look at it

By default you look at the top face, the last one printed, like a photo: hang it (`--hole`) or put it on a stand (`--stand`) where daylight or a neutral white lamp falls on it. PrusaSlicer shows that face as you will see it.

A colour lithophane (`--light back`) needs a diffuse, neutral white LED (about 4000 K) behind it. By default the face that lay on the bed looks at you, so a smooth or satin sheet gives the cleanest result; in PrusaSlicer, seen from above, the picture is mirrored, because you see it from behind.

### 8. White lithophanes (optional)

```sh
uv run lithophane-forge calibrate --slice
uv run lithophane-forge make photos/lake.jpg --mode mono --width 120 --t-min 0.8 --t-max 3.4 --td 2.1 --slice
```

`--mode mono` makes a classic lithophane in white PETG. It prints standing with a brim, for the finest tones, and a 120 mm panel takes about 3 to 5 hours. Curved panels and cylinders are always mono. Print the calibration strip first (about 1 hour 20 minutes), read it with `out/calibration-petg/report.md`, and pass your own `--t-min`, `--t-max` and `--td`.

## Output

Put the photos you want to try in `photos/`: git ignores everything in it, so your own photos never end up in the repository. Everything the tool makes goes to `out/<name>/` (also ignored), where `<name>` is the photo's name (or `--name`). Each run first removes the files an earlier run wrote there under the same name, so an old `.bgcode` never sits next to a new project.

| File | What it is |
|---|---|
| `<name>.3mf` | PrusaSlicer project: model, presets, per-object settings, thumbnail |
| `<name>.bgcode` | G-code for the printer (only with `--slice`; `.gcode` on printers without binary G-code) |
| `<name>.stl` | The bare mesh, for other slicers (frame and tab are separate shells that overlap the panel) |
| `<name>-stand.3mf`, `<name>-lamp-base.3mf` | The accessories, as their own projects (with `--stand` or `--lamp-base`), plus STL files |
| `preview.png` | The photo, the lit panel (simulated) and the relief side |
| `focus.png` | Colour prints with `--focus`: the focus map (white = the subject; with `--focus subject`, what the models found), at the photo's proportions |
| `report.md` | What to print with, and how to light it |
| `plan.json` | Every parameter and result, for tools |

## Options

- **Mode:** colour by default for flat and shaped panels: filament changes, printed lying. `--mode mono` makes a white lithophane printed standing; curved panels and cylinders are always mono.
- **Light (colour):** `--light front` (the default) plans a picture that you look at in room light, from the top. `--light back` plans a colour lithophane, lit from behind. Mono prints are always lithophanes.
- **Size and shape:** `--width` and `--height` are the outer size in mm, frame included; give one and the photo sets the other, or both to crop the photo. `--frame` sets the frame width (0 for none) and `--frame-t` its thickness; a shaped panel's frame must reach the top of the picture, or its stepped edge shows.
- **Curved (mono):** `--shape curved` bends the panel on an arc of `--arc` degrees, 10 to 300 (the width is the arc length on the outside).
- **Cylinder (mono):** `--shape cylinder --diameter D --height H` (the height is 1.25 × the diameter if you leave it out). The picture goes once around; a normal photo then loses its top and bottom, so `--repeat 2` puts it around twice (the tool suggests the best number). `--lamp-base` adds a base with a pocket for an LED tealight or puck light (`--led-diameter`, `--led-height`) and a notch for a cable.
- **Shaped:** `--shape shaped --mask heart|oval|circle`, or `--mask` with an SVG file (its largest closed shape) or a PNG file (dark pixels are the shape). An oval takes the photo's proportions, a circle stays round. The frame follows the outline.
- **Extras:** `--hole` adds a hanging tab with a 5 mm hole (flat and shaped panels); `--stand` makes a stand with a slot for a flat panel.
- **Colour:** `--colors N` (2 to 8 filaments, 4 by default), `--filaments FILE` for your own list, `--t-min`/`--t-max` for the height range of the stack (0.6 to 3.0 mm by default), `--allow-abrasive` to also use filaments that need a hardened steel nozzle, `--allow-unmeasured` to also use translucent filaments without measured `k` and `s` in room light, `--min-patch` for the smallest island or hole a layer keeps (1 mm² by default, 0 keeps them all), `--focus MASK|X,Y,W,H|centre|subject` for the subject that gets the colours first, `--focus-floor` for the weight of the rest of the picture (0.1 by default), and `--device auto|cpu|gpu` for where the models of `--focus subject` run. See below.
- **Picture:** `--crop X,Y,W,H` (photo pixels; the panel takes the crop's shape unless you give both sizes), `--levels LOW,HIGH` (percentiles that become black and white), `--gamma`, `--clahe` (local contrast, mono only), `--invert` (a negative in lightness, mono only), `--view` (the face that looks at you: the relief, which is the default with light from the front and the only choice, or the flat face, the default for lithophanes).
- **Thickness (mono):** `--t-min`, `--t-max` and `--td` (mm of your filament that let 10 % of the light through). The defaults for white PETG (0.8, 3.6 and 2.3 mm) are estimates: calibrate your spool for the best result. In colour, `--t-min` and `--t-max` set the height range of the stack, and `--td` is an error: the filaments' own TD values count.
- **Printer:** `--printer` (`mk4`, `mk4s`, `mk3.5`, `mk3s`, `coreone`, `coreonel`, `xl`, `mini`), `--material petg|pla`, or your own presets with `--printer-preset`, `--print-preset` and `--filament-preset`.
- **Detail:** `--pitch` (mm between grid points, 0.2 by default; a 0.4 mm nozzle gains nothing below that) and `--max-triangles`.

## Printing

- Colour panels (the default) and shaped panels print lying, flat face down. A picture for room light (the default) shows its top face; a lithophane shows the face on the bed by default, so a smooth or satin sheet gives the cleanest result. `report.md` says which filament to load first and at which height to change it.
- Mono panels use white filament and print standing with a brim. PETG lets more light through than PLA, so its thickness range is wider.
- On the MK4, print PETG on the textured sheet, or use a release agent on the smooth sheet, and keep the filament dry.
- Print the stand and the lamp base in any opaque colour; they use 0.2 mm layers.
- Light a lithophane from behind with a diffuse, neutral white LED (about 4000 K). By default its flat face looks at you.

## Colour prints

A printer with one extruder changes colour only between layers: the printer stops (M600), you load the next filament, and it goes on. So a colour panel prints lying on the bed, and every point is a stack of filaments. The first filament starts on the bed, the second starts at the first change, and so on; each point stops at its own height.

- **Two kinds of light.** In room light (`--light front`, the default) you look at the top of the stack. Most filaments are translucent: a thin layer works like coloured glass over what lies under it, not like paint, and a filament needs about its TD to hide what lies under it. Thin yellow over white is a bright lemon, thin blue over yellow a dark green, and three layers (0.3 mm) of white over black are still a light grey. The tool computes this with a two-flux (Kubelka-Munk) model, which matched the owner's first room-light print. Lit from behind (`--light back`), the light crosses the whole stack and every filament in it counts.
- **One path of colours.** The colours a point can show lie on one path, from the first filament alone to the whole stack: from dark to light or from light to dark in room light, from light to dark when lit from behind. A sunset, a portrait or a sepia photo fits such a path well; a photo with red and blue of the same lightness side by side does not. The tool tells you how close it gets (mean ΔE, CIEDE2000) and shows the simulated result in `preview.png`.
- **Ideal colours first.** The tool looks for the colours and the change heights that bring the path closest to the photo, with no filament list in mind. k-means in CIELAB gives a start. Then an optimiser moves the colours, how translucent each one is, and the heights. The same search that picks real filaments also runs on the photo's own colours, with every change on a layer, and the best of both is the ideal. In room light the base can have any colour, but it must hide what is behind the panel at the thinnest point of the panel, so at the default 0.6 mm it is an opaque filament (white PETG hides the back only from about 1.8 mm), and the search tries the colours both lightest first and darkest first. Lit from behind, the base stays white or natural and translucent, because it lies behind every point. If your list has no filament that fits as the base, a note says so. `analyze --colors N` shows the ideal colours without making any file.
- **Then real filaments.** It picks the Prusament PETG filaments that make the whole print closest to the photo, using the hex codes and TD (transparency) values that Prusa publishes, and fits the change heights again for them. They can differ from the filaments closest in colour to the ideal ones, which `report.md` also lists. With `--filaments FILE` it picks from your own list instead (see `examples/filaments.toml`). The plan can use fewer filaments than `--colors`: when the search finds no place where one more brings the print closer, or when your list is shorter. A note in the output says which. Abrasive filaments, such as Prusament PETG Tungsten 75 %, wear out a brass nozzle, so the tool leaves them out unless you pass `--allow-abrasive` (for a hardened steel nozzle). In room light, hex and TD predict opaque filaments and white or grey ones well enough, but not translucent coloured ones (one TD cannot say how much a coloured filter absorbs on each colour channel) or nearly clear ones such as natural and clear PETG (their hex says little about how much they scatter). Such a filament needs its measured absorption and scattering, `k = [red, green, blue]` and `s` (1/mm), in the list; without them the tool leaves it out unless you pass `--allow-unmeasured`, and `report.md` says which filaments it left out. Yellow Gold and Ultramarine Blue Transparent have preliminary values in the catalog, fitted from a photo of a print; a calibration print will replace them.
- **No specks.** Every point gets the height whose colour is closest to the photo, so noise and fine texture in the photo give tiny islands and holes in a layer. An island smaller than about two nozzle widths prints as a blob, and in the picture it is a speck of the wrong colour. So the tool lowers every island smaller than `--min-patch` (1 mm² by default) to the height around it, and raises every hole of that size, in every layer. A thin line or ring that joins a larger patch stays, and so does a patch that joins the frame. At 60 mm wide this also blurs small details such as eyes; print larger for them. The ΔE in `report.md` is measured on the heights that print, after this smoothing, and the simulated panels in `preview.png` use the same heights (`analyze` shows the plan before smoothing).
- **Focus on the subject.** Without help, the tool fits every pixel the same, so a large sky or lawn wins over two small people in front of it. `--focus` tells it where the subject is: a mask image (white = the subject, black or transparent = the rest, stretched to the photo, so paint it over the photo), a box `X,Y,W,H` in photo pixels (like `--crop`), or `centre` (a wide soft spot in the middle of the picture). The fit then counts the subject's pixels 10 times more than the others (`--focus-floor 0.1` is the weight of the rest; lower favours the subject more), so the colours and the change heights serve the subject first. It cannot add colours that one path of filaments does not have, and the rest of the picture can get further from the photo. `report.md` gives the mean ΔE over the subject next to the one over the whole picture, `preview.png` outlines the subject on the photo, and `focus.png` holds the map. To refine it, copy it next to the photo, edit the copy and pass it with `--focus`: every run rewrites `out/<name>/`. A mask whose proportions differ from the photo's gets a note, because it would be stretched (for example a mask painted over a crop, or over a copy that your editor did not turn upright). Mark a wide region (the whole person, not only the face): a small subject weighs little even at full weight. At 60 mm a face is only about 3 mm wide, room for three to five patches of colour; print 120 mm or more for faces.
- **Finding the subject (`--focus subject`).** Three small segmentation models find it: [U²-Net](https://github.com/xuebinqin/U-2-Net) for people, [MODNet](https://github.com/ZHKKKe/MODNet) for portraits and [IS-Net](https://github.com/xuebinqin/DIS) for the main object of a photo (all Apache-2.0). Each one alone sometimes loses a hand or adds a rock or a fence; the subject is where at least two of the three agree (their mean is 0.5 or more), and parts smaller than 0.2 % of the photo are dropped. With people in the photo it finds the people; without people, its main object (a plate of food, a car); in an open landscape nothing, and then `--focus centre` or a mask is the way. They run once per photo through [ONNX Runtime](https://onnxruntime.ai), in a second or two on a recent desktop CPU (a few seconds on a small laptop), so a graphics card is not needed. `--device auto` (the default) uses an NVIDIA card when ONNX Runtime can (the `vision-gpu` extra), and the CPU otherwise; `--device cpu` always uses the CPU, and `--device gpu` is an error without CUDA. The models download on first use (about 380 MB, checked by SHA-256) into `~/.cache/lithophane-forge/models`, or into the folder that `LITHOPHANE_FORGE_MODELS` names. `lithophane-forge models` lists them, checks them and says where they would run; `models --download` fetches them now, for example before going offline. `report.md` and `plan.json` say which models found the subject and on which device. The map is written to `focus.png`, so passing it back with `--focus` gives the same print again without the models, on any machine.
- **The project.** The 3MF carries the colour changes, so PrusaSlicer shows them and the MK4 stops at the right layers, and it starts with the colour of the first filament (the filament presets would show orange). `report.md` lists which filament to load at which height.

## Calibrate your filament

```sh
uv run lithophane-forge calibrate --slice
```

For mono prints. The strip has twelve steps from 0.6 to 5 mm, printed standing like a lithophane, with each thickness written in thin plastic under its step (it glows when lit). Hold it against your light: the first step that looks black is `--t-max`, about 0.55 × that is `--td`, and the thinnest even step is `--t-min`. `report.md` explains it again next to the files.

## Development

```sh
uv run pytest                     # add -m "not slicer" to skip the PrusaSlicer runs
uv run ruff check . && uv run ruff format --check .
```

The tests that need the real models (`-m vision`) are skipped without the `vision` extra or without the models in their folder: run `uv sync --extra vision` and `uv run lithophane-forge models --download` first. GitHub Actions runs ruff, the tests on Python 3.13 and 3.14, a slicing check with PrusaSlicer and a check with the real models on every pull request and on every push to `main`.

## Contributing

- Work on a feature branch (`feature/<topic>`) and open a pull request to `main`, for every change, small fixes included. A pull request is merged when CI is green; nobody commits to `main` directly.
- Write everything in the repository in English at B2/C1 level: code comments, docs, commit messages and pull requests. Use short sentences and common words.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
