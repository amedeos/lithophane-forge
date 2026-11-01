import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import trimesh

from lithophane_forge import mesh, threemf

NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


def box(size, centre):
    b = trimesh.creation.box(size)
    b.apply_translation(centre)
    return mesh.Mesh(np.asarray(b.vertices), np.asarray(b.faces))


def test_parts_volumes_and_settings(tmp_path):
    a, b = box((20, 5, 30), (125, 105, 15)), box((5, 20, 30), (125, 105, 15))
    path = tmp_path / "x.3mf"
    part = threemf.Part(
        "cross", [threemf.Volume("a", a), threemf.Volume("b", b)], {"fill_density": "100%"}
    )
    threemf.write(path, [part], colour_changes=[(1.2, "#ff0000")], thumbnail=b"png")
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        model = ET.fromstring(z.read(threemf.MODEL))
        cfg = z.read(threemf.MODEL_CONFIG).decode()
        m600 = z.read(threemf.M600).decode()
        rels = z.read("_rels/.rels").decode()
    assert {threemf.MODEL, threemf.MODEL_CONFIG, threemf.M600, threemf.THUMBNAIL} <= names
    verts = model.findall(".//m:vertex", NS)
    tris = model.findall(".//m:triangle", NS)
    assert len(verts) == 16 and len(tris) == 24
    xyz = np.array([[float(v.get(k)) for k in "xyz"] for v in verts])
    assert np.allclose(xyz, np.concatenate([a.vertices, b.vertices]), atol=1e-4)
    assert 'firstid="0" lastid="11"' in cfg and 'firstid="12" lastid="23"' in cfg
    assert 'key="fill_density" value="100%"' in cfg
    assert re.search(r'<code bed_idx="0" print_z="1.200" type="0" .*gcode="M600"/>', m600)
    assert '<mode bed_idx="0" value="SingleExtruder"/>' in m600
    assert "thumbnail" in rels


def test_add_config_replaces(tmp_path):
    path = tmp_path / "x.3mf"
    threemf.write(path, [threemf.Part("a", [threemf.Volume("a", box((1, 1, 1), (0, 0, 0.5)))])])
    threemf.add_config(path, "; one = 1\n")
    threemf.add_config(path, "; two = 2\n")
    with zipfile.ZipFile(path) as z:
        assert z.namelist().count(threemf.CONFIG) == 1
        assert z.read(threemf.CONFIG).decode() == "; two = 2\n"
        assert threemf.MODEL in z.namelist()
