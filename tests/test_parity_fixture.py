"""The committed Python/TypeScript parity fixture must match the current Python lab.

``frontend/src/lab/__fixtures__/parity.json`` stores inputs and Python outputs; the
TypeScript tests compare against it. Re-deriving the outputs here closes the loop: if
``satwiser.lab`` changes without regenerating the fixture
(``python scripts/make_parity_fixtures.py``), this test fails.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from satwiser import lab

FIXTURE = Path(__file__).resolve().parents[1] / "frontend" / "src" / "lab" / "__fixtures__" / \
    "parity.json"
DATA = json.loads(FIXTURE.read_text())
STATES = np.column_stack([np.asarray(DATA["states"][c]) for c in
                          ("rx", "ry", "rz", "vx", "vy", "vz")])
ORBIT = np.asarray(DATA["orbit"])
T = np.asarray(DATA["t_offset_s"])
TEMPLATE = np.asarray(DATA["template_a"])


def as_array(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


@pytest.mark.parametrize("case", DATA["cases"], ids=[c["name"] for c in DATA["cases"]])
def test_fixture_matches_python_lab(case):
    det = lab.LabDetector(**(case.get("detector") or DATA["detector"]))
    deg = lab.Degradation(case["sigma_m"], case["points_per_day"], case["rho"])
    idx = lab.subsample(len(T), deg.points_per_day, case["phase"], step_s=DATA["step_s"])
    assert idx.tolist() == case["idx"]
    normals = np.asarray(case["normals"]).reshape(-1, 6)
    a = lab.mean_a_nosp(lab.degrade(STATES[idx], deg, normals))
    a = a + 2 * DATA["a_ref"] * case["dv_add"] / DATA["v_ref"] * (T[idx] > DATA["event_s"])
    np.testing.assert_allclose(a, case["a_nosp"], atol=1e-6)
    b = lab.bin_revolutions(deg.points_per_day)
    expected = lab.expected_per_bin(deg.points_per_day, b)
    means = lab.bin_means(a, ORBIT[idx], TEMPLATE, b, expected)[1]
    np.testing.assert_allclose(means, as_array(case["means"]), atol=1e-6)
    w = lab.detector_window(det, b)
    assert (b, w) == (case["revs_per_bin"], case["window"])
    d, z = lab.window_scores(means, w, det.normalisation, det.floor_m)
    np.testing.assert_allclose(d, as_array(case["d"]), atol=1e-6)
    np.testing.assert_allclose(z, as_array(case["z"]), atol=1e-6)
    assert lab.window_detect(z, det.threshold, w) == case["detections"]


def test_fixture_covers_both_normalisations():
    kinds = {(c.get("detector") or DATA["detector"])["normalisation"] for c in DATA["cases"]}
    assert kinds == {"diff", "window"}
