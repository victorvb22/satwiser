"""Average osculating elements over each revolution.

A revolution runs from one ascending node crossing to the next (argument of latitude
wrapping from 2*pi to 0). Averaging over a full revolution removes the J2 short-period
terms (twice-per-revolution in the semi-major axis, ~18 km peak to peak for Sentinel-1),
which otherwise hide manoeuvre signatures of a few tens of metres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from satwiser.orbit.elements import MU_EARTH


def revolution_ids(u: np.ndarray) -> np.ndarray:
    """Integer revolution counter that increments at each ascending node crossing."""
    wraps = np.diff(u) < -np.pi
    return np.concatenate(([0], np.cumsum(wraps)))


def revolution_means(elements: pd.DataFrame, nominal_step_s: float = 10.0,
                     min_coverage: float = 0.95, trim_edges: bool = True) -> pd.DataFrame:
    """Per-revolution mean of ``a``, ``e`` and ``i``.

    ``a`` is averaged from ``a_nosp`` (J2 short-period term removed) when available.
    Only complete revolutions are returned: at least ``min_coverage`` of a period worth
    of samples, and no internal gap (at least 99 % of the samples expected between the
    first and last epoch). With ``trim_edges`` the first and last groups, truncated by
    the data span, are dropped outright; otherwise the coverage test decides.

    Optional columns ``orbit`` (absolute orbit number) and ``pod_flag`` (POD quality
    flag, kept for label cross-checks only) are aggregated when present. The index is
    the mean epoch of the samples; ``t_start`` / ``t_stop`` bound the revolution.
    """
    rev = revolution_ids(elements["u"].to_numpy())
    frame = elements[["e", "i"]].copy()
    frame["a"] = elements["a_nosp"] if "a_nosp" in elements else elements["a"]
    frame["rev"] = rev
    frame["t"] = elements.index.as_unit("ns").asi8
    aggs = {
        "t_ns": ("t", "mean"), "t_start": ("t", "min"), "t_stop": ("t", "max"),
        "n": ("a", "size"), "a": ("a", "mean"), "e": ("e", "mean"), "i": ("i", "mean"),
    }
    if "orbit" in elements:
        frame["orbit"] = elements["orbit"].to_numpy()
        aggs["orbit"] = ("orbit", "median")
    if "pod_flag" in elements:
        frame["pod_flag"] = elements["pod_flag"].to_numpy()
        aggs["pod_flag"] = ("pod_flag", "any")
    out = frame.groupby("rev").agg(**aggs)
    if trim_edges:
        out = out.iloc[1:-1]
    span = (out["t_stop"] - out["t_start"]) * 1e-9 + nominal_step_s
    expected = span / nominal_step_s
    period = 2 * np.pi * np.sqrt(out["a"] ** 3 / MU_EARTH)
    complete = (out["n"] >= min_coverage * period / nominal_step_s) & (out["n"] >= 0.99 * expected)
    out = out[complete].copy()
    out.index = pd.to_datetime(out.pop("t_ns").astype(np.int64))
    out.index.name = "utc"
    out["t_start"] = pd.to_datetime(out["t_start"])
    out["t_stop"] = pd.to_datetime(out["t_stop"])
    if "orbit" in out:
        out["orbit"] = np.round(out["orbit"]).astype(np.int64)
    return out
