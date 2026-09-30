"""Drag decay of the mean semi-major axis as a function of solar and geomagnetic activity.

Between manoeuvres the template-corrected mean semi-major axis decays almost linearly.
The decay rate per revolution is modelled as a power law of the solar flux:

    -da/drev = exp(b0) * F81^b1 * (F / F81)^b2 * (1 + Ap)^b3

where F is the daily observed F10.7, F81 its 81-day centred mean and Ap the daily
geomagnetic index. It is fitted by weighted least squares on the log of the slopes of
quiet segments of the calibration period. An exponential law in F81 fitted equally well
in calibration but overshot the 2024 solar maximum by a factor of two; the power law
extrapolates within a few percent (``reports/step3/model.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from satwiser.model.grid import segments


def _design(f81: np.ndarray, f: np.ndarray, ap: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(f81)), np.log(f81), np.log(f / f81), np.log1p(ap)])


@dataclass
class DragModel:
    beta: np.ndarray

    def rate(self, f81, f, ap) -> np.ndarray:
        """Semi-major axis change per revolution (m, negative)."""
        x = _design(np.asarray(f81, float), np.asarray(f, float), np.asarray(ap, float))
        return -np.exp(x @ self.beta)

    def grid_rate(self, grid: pd.DataFrame) -> np.ndarray:
        return self.rate(grid["f107_81d"], grid["f107_obs"], grid["ap"])

    def cumulative(self, grid: pd.DataFrame) -> np.ndarray:
        """Cumulative modelled decay along the orbit grid (m), zero at the first orbit."""
        r = self.grid_rate(grid)
        return np.concatenate(([0.0], np.cumsum(r[1:])))

    def to_dict(self) -> dict:
        return {"form": "exp(b0) F81^b1 (F/F81)^b2 (1+Ap)^b3", "beta": self.beta.tolist()}


def segment_slopes(grid: pd.DataFrame, column: str, mask: np.ndarray, min_len: int = 30
                   ) -> pd.DataFrame:
    """Linear slope (per revolution) and mean covariates of each quiet segment."""
    y = grid[column].to_numpy()
    rows = []
    for run in segments(mask, min_len):
        x = run - run.mean()
        p = np.polyfit(x, y[run], 1)
        sub = grid.iloc[run]
        rows.append({
            "t": sub["t"].iloc[len(run) // 2], "n": len(run), "slope": p[0],
            "rms": float(np.std(y[run] - np.polyval(p, x))),
            "f81": sub["f107_81d"].mean(), "f": sub["f107_obs"].mean(), "ap": sub["ap"].mean(),
        })
    return pd.DataFrame(rows)


def fit_drag(slopes: pd.DataFrame) -> DragModel:
    s = slopes[slopes["slope"] < 0]
    w = np.sqrt(s["n"].to_numpy())[:, None]
    x = _design(s["f81"].to_numpy(), s["f"].to_numpy(), s["ap"].to_numpy())
    beta, *_ = np.linalg.lstsq(x * w, np.log(-s["slope"].to_numpy()) * w[:, 0], rcond=None)
    return DragModel(beta)
