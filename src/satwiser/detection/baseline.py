"""Baseline manoeuvre detector on the per-revolution mean semi-major axis.

For each revolution k, the step statistic is the difference between the mean of the
``window`` revolutions after k and the mean of the ``window`` revolutions before k
(revolution k itself, which contains the burn and is only partly shifted, is skipped):

    d_k = mean(a[k+1 .. k+W]) - mean(a[k-W .. k-1])

Averaging W revolutions on each side divides the revolution noise by sqrt(W/2), which a
single revolution-to-revolution jump cannot do. Drag decay adds a slowly varying offset
to d_k; it is removed by normalising with a rolling median and MAD:

    z_k = (d_k - median(d)) / (1.4826 * MAD(d))     over +-``norm_half_width`` revolutions

A manoeuvre produces a triangular bump in |z| centred on its revolution; detections are
the maxima of runs where |z| exceeds the threshold.

The windows are indexed by absolute orbit number, so missing revolutions (data gaps,
incomplete revolutions) shrink the windows instead of silently stretching them in time.
The algorithm deliberately uses only means, medians and fixed windows so that it can be
reimplemented identically in the browser (the lab's Web Worker).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineConfig:
    window: int = 8
    threshold: float = 5.0
    norm_half_width: int = 220  # about 15 days of revolutions
    min_count_ratio: float = 0.5


DEFAULT = BaselineConfig()


def _regular(orbit: np.ndarray, values: np.ndarray) -> pd.Series:
    """Values on a gap-free grid of orbit numbers (NaN where missing)."""
    series = pd.Series(values, index=orbit)
    series = series[~series.index.duplicated(keep="first")]
    return series.reindex(np.arange(orbit.min(), orbit.max() + 1))


def step_statistic(orbit: np.ndarray, a: np.ndarray, window: int,
                   min_count_ratio: float = 0.5) -> pd.Series:
    """``d_k`` on the regular orbit grid (NaN when a side has too few revolutions)."""
    s = _regular(orbit, a)
    min_periods = max(1, int(np.ceil(window * min_count_ratio)))
    before = s.shift(1).rolling(window, min_periods=min_periods).mean()
    after = s[::-1].shift(1).rolling(window, min_periods=min_periods).mean()[::-1]
    return after - before


def robust_z(d: pd.Series, half_width: int) -> pd.Series:
    span = 2 * half_width + 1
    min_periods = max(10, half_width // 4)
    med = d.rolling(span, center=True, min_periods=min_periods).median()
    mad = (d - med).abs().rolling(span, center=True, min_periods=min_periods).median()
    return (d - med) / (1.4826 * mad)


def scores(revs: pd.DataFrame, cfg: BaselineConfig = DEFAULT) -> pd.DataFrame:
    """Per-orbit step statistic, drift-corrected jump and z-score."""
    orbit = revs["orbit"].to_numpy()
    d = step_statistic(orbit, revs["a"].to_numpy(), cfg.window, cfg.min_count_ratio)
    span = 2 * cfg.norm_half_width + 1
    med = d.rolling(span, center=True, min_periods=max(10, cfg.norm_half_width // 4)).median()
    z = robust_z(d, cfg.norm_half_width)
    out = pd.DataFrame({"d": d, "jump_m": d - med, "z": z})
    out.index.name = "orbit"
    return out


def detect(score: pd.DataFrame, cfg: BaselineConfig = DEFAULT) -> pd.DataFrame:
    """One detection per run of |z| > threshold, located at the run's |z| maximum.

    Runs closer than ``window`` revolutions are merged: a single step produces a bump of
    half-width ``window``.
    """
    above = score["z"].abs() > cfg.threshold
    idx = score.index[above.to_numpy()]
    if idx.empty:
        return pd.DataFrame(columns=["orbit", "z", "jump_m"])
    run = np.concatenate(([0], np.cumsum(np.diff(idx) > cfg.window)))
    rows = []
    for r in np.unique(run):
        members = idx[run == r]
        sub = score.loc[members]
        k = sub["z"].abs().idxmax()
        rows.append({"orbit": int(k), "z": float(sub.loc[k, "z"]),
                     "jump_m": float(sub.loc[k, "jump_m"]),
                     "first_orbit": int(members.min())})
    return pd.DataFrame(rows)
