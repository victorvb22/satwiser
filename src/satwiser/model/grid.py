"""Per-revolution series on a regular grid of absolute orbit numbers.

Missing or incomplete revolutions become NaN rows, so that orbit-indexed operations
(repeat-cycle template, cumulative drag, trailing windows) stay aligned in time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from satwiser.evaluation import OrbitClock

ELEMENTS = ("a", "e", "i")
WEATHER = ("f107_obs", "f107_81d", "ap")


def to_grid(revs: pd.DataFrame) -> pd.DataFrame:
    """Reindex a revolution table on consecutive orbit numbers.

    Space-weather columns are forward filled over missing revolutions (they are daily
    values; a back fill only covers the first rows if they lack indices); orbital elements
    are left NaN. ``valid`` marks observed revolutions.
    """
    unique = revs.drop_duplicates("orbit")
    r = unique.set_index("orbit")
    orbits = np.arange(r.index.min(), r.index.max() + 1)
    g = r.reindex(orbits)
    g.index.name = "orbit"
    g["valid"] = g["a"].notna()
    known = g["valid"].to_numpy()
    mid_ns = unique.index.to_numpy().astype("datetime64[ns]").astype(np.int64)
    g["t"] = pd.to_datetime(np.interp(orbits, r.index.to_numpy(), mid_ns).astype(np.int64))
    for col in ("t_start", "t_stop"):
        ns = r[col].to_numpy().astype("datetime64[ns]").astype(np.int64)
        g[col] = pd.to_datetime(np.interp(orbits, r.index.to_numpy(), ns).astype(np.int64))
    g.loc[~known, "n"] = 0
    for col in WEATHER:
        if col in g:
            g[col] = g[col].ffill().bfill()
    return g


def quiet_mask(grid: pd.DataFrame, mans: pd.DataFrame, before: int = 1, after: int = 2
               ) -> np.ndarray:
    """Valid revolutions farther than ``before`` / ``after`` revolutions from any manoeuvre.

    Uses the ESA record: only for fitting nuisance models on the calibration period.
    """
    clock = OrbitClock(grid[grid["valid"]].reset_index())
    o0 = clock.orbit_at(mans["start"])
    o1 = clock.orbit_at(mans["stop"])
    orbits = grid.index.to_numpy()
    near = np.zeros(len(grid), bool)
    first = orbits[0]
    for lo, hi in zip(o0 - before, o1 + after, strict=True):
        near[max(lo - first, 0):max(hi - first + 1, 0)] = True
    return grid["valid"].to_numpy() & ~near


def segments(mask: np.ndarray, min_len: int) -> list[np.ndarray]:
    """Index arrays of the runs of consecutive ``True`` values at least ``min_len`` long."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    return [run for run in np.split(idx, breaks) if len(run) >= min_len]
