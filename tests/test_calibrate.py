import numpy as np
import pytest
import trimesh

from lithophane_forge import calibrate


def test_wedge_has_every_step(setup, petg, tmp_path):
    job = calibrate.default_job(setup, petg, tmp_path, pitch=0.4)
    t, lay = calibrate.wedge(job)
    assert lay.width == pytest.approx(len(calibrate.STEPS) * calibrate.STEP_W + 6)
    found = set(np.round(np.unique(t), 3))
    assert set(calibrate.STEPS) <= found
    # the labels: thin plastic inside the thicker band under the steps
    band = t[-int(calibrate.LABEL_H / lay.pitch) : -int(calibrate.FRAME / lay.pitch)]
    assert (band <= 0.8 + 1e-9).any() and np.isclose(band, calibrate.BAND_T).any()


def test_labels_survive_the_overhang_limit(setup, petg, tmp_path):
    """Every glyph stays thin plastic in the printed strip (the limit used to fill them)."""
    job = calibrate.default_job(setup, petg, tmp_path)
    t, lay = calibrate.wedge(job)
    v = np.linspace(lay.height, 0, lay.rows)
    band = (v < calibrate.FRAME + calibrate.LABEL_H - 1) & (v > calibrate.FRAME + 1)
    u = np.linspace(0, lay.width, lay.cols)
    inside = (u >= calibrate.FRAME) & (u <= lay.width - calibrate.FRAME)
    mask = calibrate.labels([f"{s:g}" for s in calibrate.STEPS], int(inside.sum()), band.sum())
    glyphs = t[np.ix_(band, inside)] < 1
    assert glyphs.sum() == mask.sum() > 500


def test_labels_fit_their_cells():
    mask = calibrate.labels(["0.6", "5"], 100, 30)
    assert mask.shape == (30, 100)
    assert mask[:, :50].any() and mask[:, 50:].any()


def test_make_calibration(setup, petg, tmp_path, no_slicer):
    job = calibrate.default_job(setup, petg, tmp_path, pitch=0.5)
    res = calibrate.make(job)
    assert trimesh.load(res.files["stl"]).is_watertight
    assert res.size[2] == pytest.approx(calibrate.STEP_H + calibrate.LABEL_H + 6)
    assert "t_max" in res.files["report"].read_text(encoding="utf-8")
