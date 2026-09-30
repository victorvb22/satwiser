"""Write the Python/TypeScript parity fixtures of the lab pipeline.

The fixture holds a four-day slice of a real lab window (60 s states), fixed-seed
standard normals and, for several degradation settings, every intermediate result of
the Python reference (``satwiser.lab``). The front-end test suite
(``frontend/src/lab/pipeline.test.ts``) replays the same inputs through the TypeScript
worker code and compares.

    python scripts/make_parity_fixtures.py
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from satwiser import lab
from satwiser.config import REPO_ROOT, data_dir
from satwiser.labels import A_REF, V_REF

OUT = REPO_ROOT / "frontend" / "src" / "lab" / "__fixtures__" / "parity.json"
SLICE_DAYS = 4
SEED = 12345
CASES = [
    {"name": "pod_dense", "sigma_m": 0.0, "points_per_day": 1440, "rho": 0.0, "dv_add": 0.0},
    {"name": "noisy_hourly", "sigma_m": 10.0, "points_per_day": 144, "rho": 0.0,
     "dv_add": 0.1},
    {"name": "correlated_sparse", "sigma_m": 30.0, "points_per_day": 24, "rho": 0.9,
     "dv_add": 0.5},
    {"name": "tle_like", "sigma_m": 1000.0, "points_per_day": 2, "rho": 0.9, "dv_add": 0.0},
]


def main() -> None:
    sat = json.loads((data_dir() / "app" / "satellites.json").read_text())[0]
    with gzip.open(data_dir() / "app" / "lab" / f"{sat['default_lab_event']}.json.gz", "rt",
                   encoding="utf-8") as fh:
        window = json.load(fh)
    t = np.asarray(window["t_offset_s"])
    event_s = (pd.Timestamp(window["event_time"]) - pd.Timestamp(window["start"])
               ).total_seconds()
    keep = np.abs(t - event_s) <= SLICE_DAYS / 2 * 86400
    t = t[keep] - t[keep][0]
    event_s -= float(np.asarray(window["t_offset_s"])[keep][0])
    states = np.column_stack([np.asarray(window["states"][c])[keep]
                              for c in ("rx", "ry", "rz", "vx", "vy", "vz")])
    orbit = np.asarray(window["orbit"])[keep]
    template = np.asarray(window["template_a"])
    det = lab.LabDetector(**{k: window["detector"][k] for k in
                             ("window_revs", "threshold", "normalisation", "floor_m")})
    rng = np.random.default_rng(SEED)
    cases = []
    for case in CASES:
        deg = lab.Degradation(case["sigma_m"], case["points_per_day"], case["rho"])
        idx = lab.subsample(len(t), deg.points_per_day, phase=3, step_s=window["step_s"])
        z = rng.standard_normal((len(idx), 6))
        noisy = lab.degrade(states[idx], deg, z)
        a = lab.mean_a_nosp(noisy)
        a = a + 2 * A_REF * case["dv_add"] / V_REF * (t[idx] > event_s)
        b = lab.bin_revolutions(deg.points_per_day)
        means = lab.bin_means(a, orbit[idx], template, b,
                              lab.expected_per_bin(deg.points_per_day, b))[1]
        w = lab.detector_window(det, b)
        d, zs = lab.window_scores(means, w, det.normalisation, det.floor_m)
        cases.append({**case, "phase": 3, "idx": idx.tolist(), "normals": z.ravel().tolist(),
                      "a_nosp": a.tolist(), "revs_per_bin": b, "window": w,
                      "means": [None if np.isnan(v) else v for v in means],
                      "d": [None if np.isnan(v) else v for v in d],
                      "z": [None if np.isnan(v) else v for v in zs],
                      "detections": lab.window_detect(zs, det.threshold, w)})
    fixture = {
        "source_event": window["event_id"], "step_s": window["step_s"],
        "event_s": event_s, "t_offset_s": t.tolist(), "orbit": orbit.tolist(),
        "states": {c: states[:, k].tolist() for k, c in
                   enumerate(("rx", "ry", "rz", "vx", "vy", "vz"))},
        "template_a": template.tolist(), "detector": det.__dict__,
        "a_ref": A_REF, "v_ref": V_REF, "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture))
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.2f} MB), cases: "
          + ", ".join(f"{c['name']}={c['detections']}" for c in cases))


if __name__ == "__main__":
    main()
