"""How much of the sparse-sampling floor would a sample-level ground-track template remove?

The residual of the J2-corrected osculating semi-major axis around its revolution mean
is what sparse sampling fails to average out. This check fits a two-dimensional
template (position in the 175-revolution cycle x argument of latitude) on the quiet
ten-day windows cached by ``run_step4.py`` before 2020, and measures the residual RMS on
the windows from 2020 onwards, with and without the template. It is a diagnostic for
the report, not part of the pipeline: such a template is prior knowledge from precise
orbits and would make the degraded presets optimistic.

    python scripts/check_short_period_floor.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from satwiser import lab
from satwiser.config import REPO_ROOT, processed_dir
from satwiser.model.groundtrack import REPEAT
from satwiser.orbit.elements import keplerian

U_BINS = 72
SPLIT = pd.Timestamp("2020-01-01")


def residuals(states: pd.DataFrame) -> pd.DataFrame:
    x = states[["rx", "ry", "rz", "vx", "vy", "vz"]].to_numpy()
    a = lab.mean_a_nosp(x)
    u = keplerian(x[:, :3], x[:, 3:])["u"]
    orbit = states["orbit"].to_numpy()
    frame = pd.DataFrame({"a": a, "orbit": orbit,
                          "ubin": (u / (2 * np.pi) * U_BINS).astype(int) % U_BINS})
    counts = frame.groupby("orbit")["a"].transform("size")
    frame = frame[counts >= 0.9 * counts.max()]  # complete revolutions only
    frame["resid"] = frame["a"] - frame.groupby("orbit")["a"].transform("mean")
    frame["phase"] = frame["orbit"] % REPEAT
    return frame


def main() -> None:
    paths = sorted(processed_dir("lab_windows").glob("S1A_quiet_*.parquet"))
    train, test = [], []
    for path in paths:
        start = pd.Timestamp(path.stem.split("_")[-1])
        (train if start < SPLIT else test).append(residuals(pd.read_parquet(path)))
    n_train = len(train)
    train, test = pd.concat(train), pd.concat(test)
    template = train.groupby(["phase", "ubin"])["resid"].mean()
    key = pd.MultiIndex.from_arrays([test["phase"], test["ubin"]])
    fitted = template.reindex(key).to_numpy()
    covered = ~np.isnan(fitted)
    before = float(np.sqrt(np.mean(test["resid"][covered] ** 2)))
    after = float(np.sqrt(np.mean((test["resid"][covered] - fitted[covered]) ** 2)))
    result = {"train_windows": n_train, "test_windows": len(paths) - n_train,
              "coverage": float(covered.mean()), "rms_before_m": before, "rms_after_m": after}
    out = REPO_ROOT / "reports" / "step4" / "short_period_floor.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
