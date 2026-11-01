"""Write 3MF files that PrusaSlicer opens as projects.

We write the 3MF ourselves: the meshes come straight from numpy (hundreds of thousands of
triangles), and we need PrusaSlicer's own metadata next to them:

- `Metadata/Slic3r_PE_model.config`: object names, volumes (several parts of one object,
  which PrusaSlicer joins when it slices) and per-object settings (for example 100 %
  infill). The print presets themselves stay untouched.
- `Metadata/Prusa_Slicer_custom_gcode_per_print_z.xml`: the M600 colour changes.
- `Metadata/Slic3r_PE.config`: the full print configuration (added by `slicer.py`).
- `Metadata/thumbnail.png`: a preview picture, if we have one.

Vertices are written in bed coordinates and every build item has the identity transform.
"""

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .mesh import Mesh

MODEL = "3D/3dmodel.model"
MODEL_CONFIG = "Metadata/Slic3r_PE_model.config"
CONFIG = "Metadata/Slic3r_PE.config"
M600 = "Metadata/Prusa_Slicer_custom_gcode_per_print_z.xml"
THUMBNAIL = "Metadata/thumbnail.png"
IDENTITY = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
MESH_STATS = (
    '   <mesh edges_fixed="0" degenerate_facets="0" facets_removed="0" '
    'facets_reversed="0" backwards_edges="0"/>\n'
)

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
</Types>
"""


def _rels(thumbnail: bool) -> str:
    thumb = (
        ' <Relationship Target="/Metadata/thumbnail.png" Id="rel-2" Type="http://schemas'
        '.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>\n'
        if thumbnail
        else ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft'
        '.com/3dmanufacturing/2013/01/3dmodel"/>\n' + thumb + "</Relationships>\n"
    )


@dataclass
class Volume:
    name: str
    mesh: Mesh
    kind: str = "ModelPart"


@dataclass
class Part:
    """One PrusaSlicer object: one or more volumes and its own settings."""

    name: str
    volumes: list[Volume]
    settings: dict[str, str] = field(default_factory=dict)


def _mesh_xml(volumes: list[Volume]) -> tuple[str, list[tuple[int, int]]]:
    out, ranges, nv, nt = [], [], 0, 0
    verts, tris = [], []
    for vol in volumes:
        verts.append(vol.mesh.vertices)
        tris.append(vol.mesh.faces + nv)
        nv += len(vol.mesh.vertices)
        ranges.append((nt, nt + len(vol.mesh.faces) - 1))
        nt += len(vol.mesh.faces)
    buf = io.StringIO()
    np.savetxt(buf, np.concatenate(verts), fmt='<vertex x="%.4f" y="%.4f" z="%.4f"/>')
    out.append("<vertices>\n" + buf.getvalue() + "</vertices>\n")
    buf = io.StringIO()
    np.savetxt(buf, np.concatenate(tris), fmt='<triangle v1="%d" v2="%d" v3="%d"/>')
    out.append("<triangles>\n" + buf.getvalue() + "</triangles>\n")
    return "".join(out), ranges


def m600_xml(changes: list[tuple[float, str]]) -> str:
    """PrusaSlicer's colour-change list: [(print_z, '#rrggbb'), ...].

    bed_idx: PrusaSlicer 2.9 keeps colour changes per bed; without it the GUI silently
    drops them (the CLI still applies them). Older versions ignore the attribute."""
    codes = "".join(
        f'<code bed_idx="0" print_z="{z:.3f}" type="0" extruder="1" '
        f'color="{colour}" extra="" gcode="M600"/>\n'
        for z, colour in changes
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n<custom_gcodes_per_print_z>\n'
        f'{codes}<mode bed_idx="0" value="SingleExtruder"/>\n'
        "</custom_gcodes_per_print_z>\n"
    )


def write(
    path: Path,
    parts: list[Part],
    colour_changes: list[tuple[float, str]] | None = None,
    thumbnail: bytes | None = None,
    config: str | None = None,
) -> None:
    objects, items, model_cfg = [], [], []
    for k, part in enumerate(parts, start=1):
        mesh, ranges = _mesh_xml(part.volumes)
        objects.append(
            f'<object id="{k}" type="model" name={quoteattr(part.name)}>\n'
            f"<mesh>\n{mesh}</mesh>\n</object>\n"
        )
        items.append(f'<item objectid="{k}" transform="1 0 0 0 1 0 0 0 1 0 0 0" printable="1"/>\n')
        cfg = [
            f' <object id="{k}" instances_count="1">\n',
            f'  <metadata type="object" key="name" value={quoteattr(part.name)}/>\n',
        ]
        cfg += [
            f'  <metadata type="object" key={quoteattr(key)} value={quoteattr(value)}/>\n'
            for key, value in part.settings.items()
        ]
        for vol, (first, last) in zip(part.volumes, ranges):
            cfg += [
                f'  <volume firstid="{first}" lastid="{last}">\n',
                f'   <metadata type="volume" key="name" value={quoteattr(vol.name)}/>\n',
                f'   <metadata type="volume" key="volume_type" value="{vol.kind}"/>\n',
                f'   <metadata type="volume" key="matrix" value="{IDENTITY}"/>\n',
                MESH_STATS,
                "  </volume>\n",
            ]
        cfg.append(" </object>\n")
        model_cfg += cfg
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">\n'
        '<metadata name="slic3rpe:Version3mf">1</metadata>\n'
        '<metadata name="Application">lithophane-forge</metadata>\n'
        "<resources>\n" + "".join(objects) + "</resources>\n"
        "<build>\n" + "".join(items) + "</build>\n</model>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", _rels(thumbnail is not None))
        z.writestr(MODEL, model)
        z.writestr(
            MODEL_CONFIG,
            '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'
            + "".join(model_cfg)
            + "</config>\n",
        )
        if colour_changes:
            z.writestr(M600, m600_xml(colour_changes))
        if thumbnail is not None:
            z.writestr(THUMBNAIL, thumbnail)
        if config is not None:
            z.writestr(CONFIG, config)


def add_config(path: Path, config: str) -> None:
    """Add (or replace) the full print configuration of a 3MF written by `write`."""
    with zipfile.ZipFile(path) as z:
        entries = {n: z.read(n) for n in z.namelist() if n != CONFIG}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in entries.items():
            z.writestr(name, data)
        z.writestr(CONFIG, config)
