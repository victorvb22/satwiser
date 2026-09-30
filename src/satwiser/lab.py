"""Degradation lab: reference implementation of what the browser recomputes live.

A window of precise state vectors (about ten days) is degraded, turned into a mean
semi-major axis series and passed to the windowed step detector. The same algorithm runs
in the app's Web Worker (TypeScript); the two are kept in step by parity tests on shared
fixtures, so every operation here is deliberately elementary.

Degradation model (documented approximation, not a sensor simulation):

- position error ``sigma`` (m) per axis and velocity error ``n * sigma`` (m/s) per axis,
  ``n`` being the mean motion: the velocity error of an orbit-like position error of
  amplitude ``sigma``. For Sentinel-1 this maps to a semi-major axis error of about
  2.8 ``sigma`` per sample;
- both errors follow independent AR(1) processes with lag-one correlation ``rho``
  between consecutive *output* samples (temporal correlation of the errors);
- the product keeps ``points_per_day`` samples, evenly spaced, from the 10 s POD grid.

Processing chain (identical to the per-revolution pipeline, minus what cannot be
done in a browser):

1. osculating elements of each degraded state, minus the first-order J2 short-period
   term of the semi-major axis;
2. minus the ground-track repeat signature of the revolution (175-value template);
3. mean per bin of ``b`` revolutions, ``b = 1`` when a revolution holds at least one
   sample, otherwise the smallest ``b`` that does; truncated bins at the window edges are
   dropped when sampling is dense;
4. step statistic ``d_k = mean(next W bins) - mean(previous W bins)`` and robust z-score
   over the whole window (median / MAD), detections at the maxima of runs above the
   threshold.

With sparse sampling the per-bin mean no longer averages out the higher-order
short-period terms left after the first-order J2 correction (tens of metres within a
revolution): this sets the floor of a simple pipeline on sparse data, independently of
the injected noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from satwiser.io.eof import merge, read_eof
from satwiser.orbit.elements import (
    MU_EARTH,
    ecef_to_inertial,
    gmst,
    j2_short_period_a,
    keplerian,
)

STEP_S = 10.0
SAMPLES_PER_DAY = int(86400 / STEP_S)
MIN_BINS = 6
MIN_DENSE = 10


@dataclass(frozen=True)
class Degradation:
    sigma_m: float = 0.0
    points_per_day: float = SAMPLES_PER_DAY
    rho: float = 0.0


@dataclass(frozen=True)
class LabDetector:
    window_revs: int = 12
    threshold: float = 4.0
    normalisation: str = "window"  # "window" (median/MAD of d) or "diff" (bin differences)
    floor_m: float = 0.0  # lower bound on the scale of d, "diff" normalisation only


PRESETS = {
    "pod": ("POD Copernicus", Degradation(sigma_m=0.03, points_per_day=8640, rho=0.0)),
    "radar": ("Radar", Degradation(sigma_m=30.0, points_per_day=24, rho=0.5)),
    "tle": ("Type TLE", Degradation(sigma_m=1000.0, points_per_day=2, rho=0.9)),
}


# ------------------------------------------------------------------------- window data

def load_states(files: list[Path], start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    """Inertial states (m, m/s) and absolute orbit numbers on the 10 s grid."""
    raw = merge(read_eof(f) for f in files)
    raw = raw[(raw.index >= start) & (raw.index < stop)]
    utc = raw.index.values.astype("datetime64[us]")
    ut1 = utc + (raw["ut1_utc"].to_numpy() * 1e6).astype("timedelta64[us]")
    r, v = ecef_to_inertial(raw[["x", "y", "z"]].to_numpy(), raw[["vx", "vy", "vz"]].to_numpy(),
                            gmst(ut1))
    out = pd.DataFrame(np.hstack([r, v]), columns=["rx", "ry", "rz", "vx", "vy", "vz"],
                       index=raw.index)
    out["orbit"] = raw["orbit"].to_numpy()
    return out


# ------------------------------------------------------------------------- degradation

def ar1(normals: np.ndarray, rho: float) -> np.ndarray:
    """AR(1) process with unit variance from standard normals (rows are time steps).

    ``out[0] = normals[0]`` and ``out[k] = rho * out[k-1] + sqrt(1 - rho^2) * normals[k]``,
    evaluated with a recursive filter.
    """
    if rho == 0:
        return normals.copy()
    scale = np.sqrt(1.0 - rho**2)
    x = normals.copy()
    x[0] = x[0] / scale
    return lfilter([scale], [1.0, -rho], x, axis=0)


def subsample(n_samples: int, points_per_day: float, phase: int = 0) -> np.ndarray:
    """Indices of evenly spaced samples at ``points_per_day`` on the 10 s grid."""
    stride = max(1, int(round(SAMPLES_PER_DAY / points_per_day)))
    return np.arange(phase % stride, n_samples, stride)


def degrade(states: np.ndarray, deg: Degradation, normals: np.ndarray) -> np.ndarray:
    """Add correlated position/velocity errors to ``states`` (N x 6).

    ``normals`` holds N x 6 standard normal draws (the generator is left to the caller
    so that Python and TypeScript can share fixtures).
    """
    if deg.sigma_m == 0:
        return states.copy()
    a = np.linalg.norm(states[:, :3], axis=1).mean()
    n = np.sqrt(MU_EARTH / a**3)
    err = ar1(normals, deg.rho) * deg.sigma_m
    err[:, 3:] *= n
    return states + err


def mean_a_nosp(states: np.ndarray) -> np.ndarray:
    """Osculating semi-major axis minus its first-order J2 short-period term."""
    el = keplerian(states[:, :3], states[:, 3:])
    return el["a"] - j2_short_period_a(el["a"], el["i"], el["u"])


def bin_revolutions(points_per_day: float, period_s: float = 5924.0) -> int:
    per_rev = points_per_day * period_s / 86400.0
    return 1 if per_rev >= 1 else int(np.ceil(1.0 / per_rev))


def expected_per_bin(points_per_day: float, revs_per_bin: int, period_s: float = 5924.0
                     ) -> float:
    return points_per_day * revs_per_bin * period_s / 86400.0


def bin_means(a: np.ndarray, orbit: np.ndarray, template: np.ndarray, revs_per_bin: int,
              expected: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Template-corrected mean of ``a`` per bin of revolutions.

    Returns (bin index relative to the first orbit of the window, mean); empty bins are
    NaN so that bin indices stay aligned in time. When a bin should hold at least
    ``MIN_DENSE`` samples (``expected``), bins with less than 90 % of them (the truncated
    revolutions at the window edges) are also NaN: their mean would be biased by the
    short-period terms left after the J2 correction.
    """
    corrected = a - template[orbit % len(template)]
    rel = (orbit - orbit.min()) // revs_per_bin
    n_bins = int(rel.max()) + 1
    sums = np.bincount(rel, weights=corrected, minlength=n_bins)
    counts = np.bincount(rel, minlength=n_bins)
    keep = counts > 0
    if expected >= MIN_DENSE:
        keep &= counts >= 0.9 * expected
    means = np.full(n_bins, np.nan)
    means[keep] = sums[keep] / counts[keep]
    return np.arange(n_bins), means


# --------------------------------------------------------------------------- detection

def window_scores(means: np.ndarray, window: int, normalisation: str = "window",
                  floor_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Step statistic and robust z-score (NaN-aware).

    ``window``: centre and scale are the median and MAD of ``d`` over the window. Simple,
    but each manoeuvre inflates the MAD over ``2 W`` values of ``d``, which hurts in busy
    windows (several manoeuvres in ten days at solar maximum).

    ``diff``: centre and scale come from the differences between consecutive bins, which
    a step affects only once. With a per-bin drift ``r`` (median of the consecutive
    differences) and per-bin noise ``s`` (their MAD / sqrt 2), the expected value of
    ``d_k`` without manoeuvre is ``r`` times the gap between the mean positions of the
    bins used after and before k, and its standard deviation is
    ``sqrt(s^2 (1/n_before + 1/n_after) + gap^2 var(r))`` (the second term is the
    uncertainty of the drift estimate, a median of N differences), floored at
    ``floor_m`` to absorb slow drag variations. Both are evaluated per position, so edges
    and gaps (fewer bins) are scored correctly.
    """
    n = len(means)
    d = np.full(n, np.nan)
    gap = np.full(n, np.nan)
    inv = np.full(n, np.nan)
    min_count = max(1, (window + 1) // 2)
    positions = np.arange(n)
    for k in range(n):
        lo, hi = max(0, k - window), min(n, k + 1 + window)
        b_idx = positions[lo:k][~np.isnan(means[lo:k])]
        a_idx = positions[k + 1:hi][~np.isnan(means[k + 1:hi])]
        if len(b_idx) >= min_count and len(a_idx) >= min_count:
            d[k] = means[a_idx].mean() - means[b_idx].mean()
            gap[k] = a_idx.mean() - b_idx.mean()
            inv[k] = 1.0 / len(b_idx) + 1.0 / len(a_idx)
    ok = ~np.isnan(d)
    z = np.full(n, np.nan)
    if ok.sum() < MIN_BINS:
        return d, z
    if normalisation == "window":
        centre = np.median(d[ok])
        scale = 1.4826 * np.median(np.abs(d[ok] - centre))
        if scale > 0:
            z[ok] = (d[ok] - centre) / scale
        return d, z
    valid = np.flatnonzero(~np.isnan(means))
    consecutive = np.diff(valid) == 1
    steps = np.diff(means[valid])[consecutive]
    if len(steps) < 2:
        return d, z
    drift = np.median(steps)
    noise = 1.4826 * np.median(np.abs(steps - drift)) / np.sqrt(2.0)
    # Variance of the median drift estimate (Gaussian): (pi / 2) * var(step) / N.
    var_drift = np.pi / 2 * 2 * noise**2 / len(steps)
    scale = np.maximum(np.sqrt(noise**2 * inv[ok] + gap[ok] ** 2 * var_drift), floor_m)
    good = scale > 0
    z_ok = np.full(ok.sum(), np.nan)
    z_ok[good] = (d[ok][good] - drift * gap[ok][good]) / scale[good]
    z[ok] = z_ok
    return d, z


def window_detect(z: np.ndarray, threshold: float, window: int) -> list[int]:
    """Bin index of the |z| maximum of each run above the threshold."""
    above = np.flatnonzero(np.abs(np.nan_to_num(z)) > threshold)
    if above.size == 0:
        return []
    runs = np.split(above, np.flatnonzero(np.diff(above) > window) + 1)
    return [int(run[np.argmax(np.abs(z[run]))]) for run in runs]


def detector_window(det: LabDetector, revs_per_bin: int) -> int:
    return max(2, int(round(det.window_revs / revs_per_bin)))


def analyse(states: pd.DataFrame, template: np.ndarray, deg: Degradation, det: LabDetector,
            rng: np.random.Generator, phase: int = 0) -> dict:
    """Degrade a window and run the lab detector (offline convenience wrapper)."""
    idx = subsample(len(states), deg.points_per_day, phase)
    raw = states[["rx", "ry", "rz", "vx", "vy", "vz"]].to_numpy()[idx]
    noisy = degrade(raw, deg, rng.standard_normal(raw.shape))
    b = bin_revolutions(deg.points_per_day)
    bins, means = bin_means(mean_a_nosp(noisy), states["orbit"].to_numpy()[idx], template, b,
                            expected_per_bin(deg.points_per_day, b))
    w = detector_window(det, b)
    d, z = window_scores(means, w, det.normalisation, det.floor_m)
    return {"bins": bins, "means": means, "d": d, "z": z, "revs_per_bin": b, "window": w,
            "detections": window_detect(z, det.threshold, w)}
