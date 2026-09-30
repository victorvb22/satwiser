"""Ground-track repeat signature of the per-revolution mean elements.

Sentinel-1 flies a 12-day repeat orbit of 175 revolutions. Tesseral terms of the gravity
field make the per-revolution mean elements depend on where the revolution lies on the
ground track, and the pattern repeats every 175 revolutions: in the step 3 analysis the
autocorrelation of the mean semi-major axis residuals is 0.99 at a lag of 175. This
deterministic signature (about 23 m peak to peak in a, 4.4 mdeg in i) was the dominant
"noise" of the step 2 baseline.

The template is estimated on quiet revolutions of the calibration period only,
alternating two steps: remove a low-order polynomial from each quiet segment of the
template-corrected series, then average the residuals by position in the cycle. Every
segment covers a different part of the cycle, so the two components separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from satwiser.model.grid import segments

REPEAT = 175


@dataclass
class GroundTrackTemplate:
    values: dict[str, np.ndarray]  # column -> array of length REPEAT
    repeat: int = REPEAT

    def signature(self, orbits: np.ndarray, column: str) -> np.ndarray:
        return self.values[column][np.asarray(orbits) % self.repeat]

    def correct(self, grid: pd.DataFrame, columns=("a", "e", "i")) -> pd.DataFrame:
        """Copy of ``grid`` with ``<col>_c = col - signature`` for each column."""
        out = grid.copy()
        for col in columns:
            out[f"{col}_c"] = grid[col] - self.signature(grid.index.to_numpy(), col)
        return out

    def to_dict(self) -> dict:
        return {"repeat": self.repeat, "values": {k: v.tolist() for k, v in self.values.items()}}


def operational_start(grid: pd.DataFrame, column: str = "a_c", factor: float = 2.0,
                      run: int = 6
                      ) -> pd.Timestamp:
    """First month from which the corrected series stays at its nominal noise level.

    During orbit acquisition (Sentinel-1A: April to July 2014) the satellite is not yet on
    its reference ground track, so the repeat-cycle template does not apply. The noise of
    each month is the robust sigma of consecutive-revolution differences (no labels
    needed); the operational phase starts at the first month that opens a run of ``run``
    consecutive months below ``factor`` times the mission median.
    """
    y = grid[column].to_numpy()
    d = pd.Series(np.diff(y, prepend=np.nan), index=grid["t"])
    d = d[grid["valid"].to_numpy() & np.roll(grid["valid"].to_numpy(), 1)].dropna()
    monthly = d.groupby(d.index.to_period("M")).agg(
        lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))
    low = (monthly <= factor * monthly.median()).to_numpy()
    for k in range(len(low) - run + 1):
        if low[k:k + run].all():
            return monthly.index[k].to_timestamp()
    return monthly.index[0].to_timestamp()


def fit_template(grid: pd.DataFrame, quiet: np.ndarray, train: np.ndarray,
                 columns=("a", "e", "i"), degree: int = 2, iterations: int = 6,
                 min_len: int = 40, repeat: int = REPEAT) -> GroundTrackTemplate:
    """Estimate the repeat-cycle signature of each column (zero-mean over the cycle)."""
    phase = grid.index.to_numpy() % repeat
    runs = segments(quiet & train, min_len)
    values = {}
    for col in columns:
        y = grid[col].to_numpy()
        g = np.zeros(repeat)
        for _ in range(iterations):
            z = y - g[phase]
            resid = np.full(len(y), np.nan)
            for run in runs:
                x = run - run.mean()
                resid[run] = z[run] - np.polyval(np.polyfit(x, z[run], degree), x)
            ok = ~np.isnan(resid)
            update = pd.Series(resid[ok]).groupby(phase[ok]).mean()
            update = update.reindex(range(repeat)).fillna(0.0).to_numpy()
            g = g + update - update.mean()
        values[col] = g
    return GroundTrackTemplate(values, repeat)
