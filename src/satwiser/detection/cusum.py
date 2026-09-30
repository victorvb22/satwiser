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
raised when either sum exceeds ``h`` on any channel. The change point is the revolution
following the last time that sum was zero; the reference then restarts ``guard``
revolutions after it (the burn revolution itself is only partly shifted).

The detector is causal: an alarm at revolution k uses no data after k. Step sizes are
estimated afterwards from straight-line fits on both sides of each change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from satwiser.labels import dv_from_da


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


def _predict(y: np.ndarray, idx: np.ndarray, k: int, predictor: str) -> float:
    if predictor == "mean":
        return float(y[idx].mean())
    x = idx - k
    xm, ym = x.mean(), y[idx].mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0:
        return float(ym)
    slope = ((x - xm) * (y[idx] - ym)).sum() / sxx
    return float(ym - slope * xm)


def innovations(ch: Channel, valid: np.ndarray, memory: int, min_ref: int) -> np.ndarray:
    """One-step prediction errors without resets (used to estimate ``sigma``)."""
    y = ch.values
    out = np.full(len(y), np.nan)
    good = np.flatnonzero(valid)
    for j, k in enumerate(good):
        idx = good[max(0, j - memory):j]
        if len(idx) >= min_ref:
            out[k] = y[k] - _predict(y, idx, k, ch.predictor)
    return out


def robust_sigma(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def run_cusum(channels: list[Channel], valid: np.ndarray, cfg: CusumConfig) -> pd.DataFrame:
    """Return one row per alarm: change index, alarm index, channel and CUSUM value.

    Indices are positions in the input arrays (orbit grid positions).
    """
    good = np.flatnonzero(valid)
    s_pos = {c.name: 0.0 for c in channels}
    s_neg = {c.name: 0.0 for c in channels}
    zero_pos = {c.name: -1 for c in channels}
    zero_neg = {c.name: -1 for c in channels}
    history: dict[str, list[float]] = {c.name: [] for c in channels}
    # Reference copies: aberrant values are winsorised so that they do not bias the
    # trailing predictor; restored to the raw values once a change is confirmed.
    ref = {c.name: np.array(c.values, dtype=float) for c in channels}
    seg_lo = 0
    ref_start = 0  # position in ``good`` of the first revolution of the current segment
    rows = []
    for j, k in enumerate(good):
        while ref_start < j and good[ref_start] < seg_lo:
            ref_start += 1
        idx = good[max(ref_start, j - cfg.memory):j]
        if len(idx) < cfg.min_ref:
            for c in channels:
                zero_pos[c.name] = zero_neg[c.name] = k
            continue
        best = None
        for c in channels:
            pred = _predict(ref[c.name], idx, k, c.predictor)
            resid = c.values[k] - pred
            sigma = c.sigma
            if cfg.scale_window:
                hist = history[c.name]
                if len(hist) >= cfg.scale_window // 4:
                    recent = np.asarray(hist[-cfg.scale_window:])
                    local = 1.4826 * np.median(np.abs(recent - np.median(recent)))
                    sigma = max(local, cfg.min_scale_ratio * c.sigma)
                hist.append(resid)
            z = resid / sigma
            if abs(z) > cfg.clip:
                ref[c.name][k] = pred + np.sign(z) * cfg.clip * sigma
                z = np.sign(z) * cfg.clip
            sp = max(0.0, s_pos[c.name] + z - cfg.kappa)
            sn = max(0.0, s_neg[c.name] - z - cfg.kappa)
            if sp == 0.0:
                zero_pos[c.name] = k
            if sn == 0.0:
                zero_neg[c.name] = k
            s_pos[c.name], s_neg[c.name] = sp, sn
            for value, zero, sign in ((sp, zero_pos[c.name], 1), (sn, zero_neg[c.name], -1)):
                if value > cfg.h and (best is None or value / cfg.h > best[0]):
                    best = (value / cfg.h, c.name, zero, sign, value)
        if best is not None:
            _, name, zero, sign, value = best
            change = int(good[np.searchsorted(good, zero + 1)]) if zero + 1 <= k else k
            rows.append({"change_pos": change, "alarm_pos": int(k), "channel": name,
                         "sign": sign, "stat": value})
            seg_lo = change + 1 + cfg.guard
            for c in channels:
                ref[c.name][change:k + 1] = c.values[change:k + 1]
                s_pos[c.name] = s_neg[c.name] = 0.0
                zero_pos[c.name] = zero_neg[c.name] = k
    return pd.DataFrame(rows, columns=["change_pos", "alarm_pos", "channel", "sign", "stat"])


def _side_value(y: np.ndarray, idx: np.ndarray, at: int) -> float:
    if len(idx) == 0:
        return np.nan
    if len(idx) < 3:
        return float(y[idx].mean())
    return _predict(y, idx, at, "linear")


def step_sizes(series: dict[str, np.ndarray], valid: np.ndarray, changes: np.ndarray,
               window: int, guard: int) -> pd.DataFrame:
    """Jump of each series at each change, from line fits on both sides.

    Each side uses up to ``window`` valid revolutions and never crosses a neighbouring
    change.
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
