"""Multichannel CUSUM change detector on the corrected per-revolution series.

Each channel is a residual series in which a manoeuvre appears as a step:

- ``a``: mean semi-major axis minus its ground-track signature and minus the cumulative
  drag decay predicted from F10.7 / Ap (:mod:`satwiser.model.drag`);
- ``i``: mean inclination minus its ground-track signature.

At revolution k the expected value is predicted from the trailing ``memory`` valid
revolutions of the current segment (since the last detected change): their mean for a
drag-corrected channel, a straight-line extrapolation for a channel with an unmodelled
slow drift. The standardised innovation ``z = (y - y_hat) / sigma`` feeds a two-sided
CUSUM (Page, 1954):

    S+ = max(0, S+ + z - kappa),   S- = max(0, S- - z - kappa)

The scale ``sigma`` is either fixed (estimated on quiet calibration revolutions) or, with
``scale_window``, a causal rolling MAD of the channel's recent innovations, floored at
``min_scale_ratio`` times the fixed value; the rolling scale absorbs periods where the
drag model is less accurate (solar maximum). Innovations are clipped at ``clip``
(Huber-style) so that a single aberrant revolution cannot raise an alarm on its own when
``h > clip - kappa``; the clipped value is also winsorised in the reference used by the
trailing predictor, so the outlier does not bias the following predictions. An alarm is
raised when either sum exceeds ``h`` on any channel. The change point c is the revolution
following the last time that sum was zero; the reference of the new segment starts at
``c + 1 + guard`` (the burn revolution itself is only partly shifted).

The recursion itself uses no revolution after k to decide at k. Its inputs are not all
strictly causal, though: the drag correction uses same-day F10.7 and Ap (look-ahead of
at most one day on these covariates) and a trailing 81-day F10.7 mean, and delays are
counted in orbit time, without the publication latency of the precise orbit products
(about three weeks for AUX_POEORB). Step sizes are estimated afterwards
from straight-line fits on both sides of each change.

The loop is compiled with numba when available (same arithmetic as the pure-Python
fallback).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from satwiser.labels import dv_from_da

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is a declared dependency
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

PREDICTORS = {"mean": 0, "linear": 1}


@dataclass(frozen=True)
class Channel:
    name: str
    values: np.ndarray
    sigma: float
    predictor: str = "mean"  # "mean" or "linear"


@dataclass(frozen=True)
class CusumConfig:
    kappa: float = 1.0
    h: float = 10.0
    memory: int = 20
    min_ref: int = 4
    guard: int = 1
    est_window: int = 8
    clip: float = np.inf
    scale_window: int = 0  # 0: fixed sigma; otherwise causal rolling MAD of innovations
    min_scale_ratio: float = 0.5


@njit(cache=True)
def _predict_at(y, good, lo, hi, k, predictor):
    """Prediction at position k from ``y[good[lo:hi]]`` (mean or line through them)."""
    n = hi - lo
    ym = 0.0
    xm = 0.0
    for j in range(lo, hi):
        ym += y[good[j]]
        xm += good[j] - k
    ym /= n
    xm /= n
    if predictor == 0:
        return ym
    sxx = 0.0
    sxy = 0.0
    for j in range(lo, hi):
        dx = good[j] - k - xm
        sxx += dx * dx
        sxy += dx * (y[good[j]] - ym)
    if sxx == 0.0:
        return ym
    return ym - sxy / sxx * xm


def _predict(y: np.ndarray, idx: np.ndarray, k: int, predictor: str) -> float:
    """Prediction at k from ``y[idx]`` (Python entry point of :func:`_predict_at`)."""
    idx = np.asarray(idx, dtype=np.int64)
    return float(_predict_at(np.asarray(y, dtype=float), idx, 0, len(idx), k,
                             PREDICTORS[predictor]))


@njit(cache=True)
def _innovations_core(y, good, memory, min_ref, predictor):
    out = np.full(len(y), np.nan)
    for j in range(len(good)):
        lo = max(0, j - memory)
        if j - lo >= min_ref:
            k = good[j]
            out[k] = y[k] - _predict_at(y, good, lo, j, k, predictor)
    return out


def innovations(ch: Channel, valid: np.ndarray, memory: int, min_ref: int) -> np.ndarray:
    """One-step prediction errors without resets (used to estimate ``sigma``)."""
    good = np.flatnonzero(valid).astype(np.int64)
    return _innovations_core(np.asarray(ch.values, dtype=float), good, memory, min_ref,
                             PREDICTORS[ch.predictor])


def robust_sigma(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


@njit(cache=True)
def _cusum_core(values, good, predictors, sigmas, kappa, h, memory, min_ref, guard, clip,
                scale_window, min_scale_ratio):
    """CUSUM recursion; returns (change, alarm, channel, sign, stat) arrays of alarms."""
    n_ch = values.shape[0]
    n_good = len(good)
    ref = values.copy()
    s_pos = np.zeros(n_ch)
    s_neg = np.zeros(n_ch)
    zero_pos = np.full(n_ch, -1, dtype=np.int64)
    zero_neg = np.full(n_ch, -1, dtype=np.int64)
    hist = np.empty((n_ch, n_good))
    n_hist = np.zeros(n_ch, dtype=np.int64)
    out_change = np.empty(n_good, dtype=np.int64)
    out_alarm = np.empty(n_good, dtype=np.int64)
    out_channel = np.empty(n_good, dtype=np.int64)
    out_sign = np.empty(n_good, dtype=np.int64)
    out_stat = np.empty(n_good)
    n_out = 0
    seg_lo = 0
    ref_start = 0
    for j in range(n_good):
        k = good[j]
        while ref_start < j and good[ref_start] < seg_lo:
            ref_start += 1
        lo = max(ref_start, j - memory)
        if j - lo < min_ref:
            for c in range(n_ch):
                zero_pos[c] = k
                zero_neg[c] = k
            continue
        best_ratio = -1.0
        best_c = -1
        best_zero = 0
        best_sign = 0
        best_value = 0.0
        for c in range(n_ch):
            pred = _predict_at(ref[c], good, lo, j, k, predictors[c])
            resid = values[c, k] - pred
            sigma = sigmas[c]
            if scale_window > 0:
                if n_hist[c] >= scale_window // 4:
                    first = max(0, n_hist[c] - scale_window)
                    recent = hist[c, first:n_hist[c]]
                    med = np.median(recent)
                    local = 1.4826 * np.median(np.abs(recent - med))
                    sigma = max(local, min_scale_ratio * sigmas[c])
                hist[c, n_hist[c]] = resid
                n_hist[c] += 1
            z = resid / sigma
            if abs(z) > clip:
                ref[c, k] = pred + np.sign(z) * clip * sigma
                z = np.sign(z) * clip
            sp = max(0.0, s_pos[c] + z - kappa)
            sn = max(0.0, s_neg[c] - z - kappa)
            if sp == 0.0:
                zero_pos[c] = k
            if sn == 0.0:
                zero_neg[c] = k
            s_pos[c] = sp
            s_neg[c] = sn
            if sp > h and sp / h > best_ratio:
                best_ratio, best_value = sp / h, sp
                best_c, best_zero, best_sign = c, zero_pos[c], 1
            if sn > h and sn / h > best_ratio:
                best_ratio, best_value = sn / h, sn
                best_c, best_zero, best_sign = c, zero_neg[c], -1
        if best_c >= 0:
            change = k
            if best_zero + 1 <= k:
                change = good[np.searchsorted(good, best_zero + 1)]
            out_change[n_out] = change
            out_alarm[n_out] = k
            out_channel[n_out] = best_c
            out_sign[n_out] = best_sign
            out_stat[n_out] = best_value
            n_out += 1
            seg_lo = change + 1 + guard
            for c in range(n_ch):
                for m in range(change, k + 1):
                    ref[c, m] = values[c, m]
                s_pos[c] = 0.0
                s_neg[c] = 0.0
                zero_pos[c] = k
                zero_neg[c] = k
    return (out_change[:n_out], out_alarm[:n_out], out_channel[:n_out], out_sign[:n_out],
            out_stat[:n_out])


def run_cusum(channels: list[Channel], valid: np.ndarray, cfg: CusumConfig) -> pd.DataFrame:
    """Return one row per alarm: change index, alarm index, channel and CUSUM value.

    Indices are positions in the input arrays (orbit grid positions).
    """
    good = np.flatnonzero(valid).astype(np.int64)
    values = np.vstack([np.asarray(c.values, dtype=float) for c in channels])
    predictors = np.array([PREDICTORS[c.predictor] for c in channels], dtype=np.int64)
    sigmas = np.array([c.sigma for c in channels], dtype=float)
    change, alarm, channel, sign, stat = _cusum_core(
        values, good, predictors, sigmas, float(cfg.kappa), float(cfg.h), int(cfg.memory),
        int(cfg.min_ref), int(cfg.guard), float(cfg.clip), int(cfg.scale_window),
        float(cfg.min_scale_ratio))
    names = np.array([c.name for c in channels], dtype=object)
    return pd.DataFrame({"change_pos": change.astype(int), "alarm_pos": alarm.astype(int),
                         "channel": names[channel] if len(channel) else [],
                         "sign": sign.astype(int), "stat": stat},
                        columns=["change_pos", "alarm_pos", "channel", "sign", "stat"])


def _side_value(y: np.ndarray, idx: np.ndarray, at: int) -> float:
    if len(idx) == 0:
        return np.nan
    if len(idx) < 3:
        return float(y[idx].mean())
    return _predict(y, idx, at, "linear")


def step_sizes(series: dict[str, np.ndarray], valid: np.ndarray, changes: np.ndarray,
               window: int, guard: int) -> pd.DataFrame:
    """Jump of each series at each change, from line fits on both sides.

    The before side uses the valid revolutions among the ``window`` grid positions
    preceding the change (never crossing the previous change); the after side uses the
    first ``window`` valid revolutions from ``change + guard`` (never crossing the next
    change).
    """
    good = np.flatnonzero(valid)
    changes = np.sort(np.asarray(changes, int))
    rows = []
    for m, c in enumerate(changes):
        prev_c = changes[m - 1] + 1 + guard if m > 0 else 0
        next_c = changes[m + 1] if m + 1 < len(changes) else len(valid)
        before = good[(good >= max(prev_c, c - window)) & (good < c)]
        after = good[(good >= c + guard) & (good < next_c)][:window]
        row = {"change_pos": int(c), "n_before": len(before), "n_after": len(after)}
        for name, y in series.items():
            row[f"d{name}"] = _side_value(y, after, c) - _side_value(y, before, c)
        rows.append(row)
    return pd.DataFrame(rows)


def detect(grid: pd.DataFrame, channels: list[Channel], cfg: CusumConfig,
           estimate: dict[str, np.ndarray]) -> pd.DataFrame:
    """Alarms with orbit numbers, timing and estimated jumps (``da``, ``de``, ``di``...).

    Output columns follow the baseline detector's convention for evaluation: ``orbit``
    (change revolution), ``alarm_orbit``, ``z`` (CUSUM value) and ``jump_m`` (Δa).
    """
    valid = grid["valid"].to_numpy()
    alarms = run_cusum(channels, valid, cfg)
    orbits = grid.index.to_numpy()
    if alarms.empty:
        return pd.DataFrame(columns=["orbit", "alarm_orbit", "channel", "z", "jump_m"])
    sizes = step_sizes(estimate, valid, alarms["change_pos"].to_numpy(), cfg.est_window,
                       cfg.guard)
    out = alarms.merge(sizes, on="change_pos", how="left")
    out["orbit"] = orbits[out["change_pos"]]
    out["alarm_orbit"] = orbits[out["alarm_pos"]]
    out["first_orbit"] = out["orbit"]
    out["z"] = out["stat"] * out["sign"]
    out["jump_m"] = out["da"]
    out["dv_est"] = dv_from_da(out["da"])
    return out.drop(columns=["change_pos", "alarm_pos"])
