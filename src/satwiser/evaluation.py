"""Match detections to ESA manoeuvres and compute detection metrics.

A manoeuvre spans the orbits ``[o0, o1]`` containing its first burn start and last burn
end. A detection at orbit ``o`` matches it when ``o0 - 1 <= o <= o1 + 1`` (tolerance of
one revolution). Each manoeuvre is matched at most once; further detections inside the
same tolerance window are counted as duplicates, not as false alarms.

A manoeuvre is *observable* when the ``observe_window`` revolutions on each side hold at
least half of their revolutions; unobservable ones (data gaps) are reported but excluded
from recall. Pass the same ``observe_window`` to every detector of a comparison so that
all of them are scored on the same manoeuvres.

Detection delay is counted in orbit time, from the manoeuvre start to the end of the
last revolution the alarm needs; it excludes the publication latency of the precise
orbit products (about three weeks for AUX_POEORB). Detections carrying an
``alarm_orbit`` (CUSUM) are available at the end of that orbit. For the windowed
baseline the statistic at the first orbit ``f`` of the detection run needs revolutions
up to ``f + window``, but its robust normalisation uses a centred window of several days,
so the baseline delay is not causal and only indicative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from satwiser.labels import dv_from_da

SOLAR_PHASES = (
    ("cycle 24 decline", "2014-01-01", "2018-01-01"),
    ("solar minimum", "2018-01-01", "2021-01-01"),
    ("cycle 25 rise", "2021-01-01", "2023-07-01"),
    ("cycle 25 maximum", "2023-07-01", "2027-01-01"),
)
DA_BINS = (0, 5, 10, 20, 40, np.inf)


def solar_phase(t: pd.Series) -> pd.Series:
    out = pd.Series("", index=t.index, dtype=object)
    for name, start, stop in SOLAR_PHASES:
        out[(t >= start) & (t < stop)] = name
    return out


class OrbitClock:
    """Maps UTC times to absolute orbit numbers and back, from the revolution table."""

    def __init__(self, revs: pd.DataFrame):
        self.t = revs["t_start"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        self.t_stop = revs["t_stop"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        self.orbit = revs["orbit"].to_numpy().astype(float)

    def orbit_at(self, times: pd.Series) -> np.ndarray:
        x = times.to_numpy().astype("datetime64[ns]").astype(np.int64)
        return np.floor(np.interp(x, self.t, self.orbit)).astype(np.int64)

    def end_of(self, orbits: np.ndarray) -> pd.DatetimeIndex:
        x = np.interp(np.asarray(orbits, float), self.orbit, self.t_stop)
        return pd.to_datetime(x.astype(np.int64))


def evaluate(detections: pd.DataFrame, mans: pd.DataFrame, revs: pd.DataFrame,
             window: int, start: str | None = None, stop: str | None = None,
             observe_window: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (manoeuvre table, detection table) annotated with match results.

    ``window`` is the detector window (baseline delay); ``observe_window`` (default:
    ``window``) sets which manoeuvres are observable. ``start`` / ``stop`` restrict both
    tables to a time range (evaluation split): manoeuvres by start time, matched
    detections with their manoeuvre, unmatched ones by their own time.
    """
    clock = OrbitClock(revs)
    present = set(revs["orbit"].to_numpy().tolist())
    m = mans.copy()
    m["o0"] = clock.orbit_at(m["start"])
    m["o1"] = clock.orbit_at(m["stop"])
    obs_w = observe_window or window
    need = int(np.ceil(obs_w / 2))

    def count(lo: int, hi: int) -> int:
        return sum(o in present for o in range(lo, hi + 1))

    m["observable"] = [
        count(o0 - obs_w, o0 - 1) >= need and count(o1 + 1, o1 + obs_w) >= need
        for o0, o1 in zip(m["o0"], m["o1"], strict=True)
    ]
    d = detections.copy()
    d["time"] = clock.end_of(d["orbit"].to_numpy() - 1)
    t_first = revs["t_start"].iloc[0]
    t_last = revs["t_stop"].iloc[-1]
    m = m[(m["start"] >= t_first) & (m["stop"] <= t_last)].reset_index(drop=True)
    d = d.reset_index(drop=True)
    m, d = _match(m, d, clock, window)
    # Period restriction after matching: a detection belongs to the period of the
    # manoeuvre it matched, so a boundary detection of a manoeuvre outside the period
    # is neither a false alarm nor a miss inside it.
    in_m = pd.Series(True, index=m.index)
    in_d = pd.Series(True, index=d.index)
    if start is not None:
        in_m &= m["start"] >= start
        in_d &= d["time"] >= start
    if stop is not None:
        in_m &= m["start"] < stop
        in_d &= d["time"] < stop
    matched = d["manoeuvre"] >= 0
    keep_d = np.where(matched, in_m.reindex(d["manoeuvre"], fill_value=False).to_numpy(),
                      in_d.to_numpy()).astype(bool)
    new_index = pd.Series(-1, index=m.index)
    new_index[in_m] = np.arange(int(in_m.sum()))
    m = m[in_m].reset_index(drop=True)
    d = d[keep_d].reset_index(drop=True)
    d["manoeuvre"] = np.where(d["manoeuvre"] >= 0,
                              new_index.reindex(d["manoeuvre"]).to_numpy(), -1)
    m["dv_est"] = dv_from_da(m["jump_m"])
    m["phase"] = solar_phase(m["start"])
    d["phase"] = solar_phase(d["time"])
    return m, d


def _match(m: pd.DataFrame, d: pd.DataFrame, clock: OrbitClock, window: int
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy one-to-one matching of detections to manoeuvres (see module docstring)."""
    d["status"] = "false_alarm"
    d["manoeuvre"] = -1
    m["detected"] = False
    m["det_orbit"] = np.nan
    m["jump_m"] = np.nan
    m["z"] = np.nan
    m["delay_h"] = np.nan
    det_orbits = d["orbit"].to_numpy()
    for j, man in m.iterrows():
        hits = np.flatnonzero((det_orbits >= man["o0"] - 1) & (det_orbits <= man["o1"] + 1))
        if hits.size == 0:
            continue
        free = [h for h in hits if d.at[h, "status"] == "false_alarm"]
        if not free:
            continue
        best = max(free, key=lambda h: abs(d.at[h, "z"]))
        for h in free:
            d.at[h, "status"] = "duplicate"
            d.at[h, "manoeuvre"] = j
        d.at[best, "status"] = "detected"
        m.at[j, "detected"] = True
        m.at[j, "det_orbit"] = d.at[best, "orbit"]
        m.at[j, "jump_m"] = d.at[best, "jump_m"]
        m.at[j, "z"] = d.at[best, "z"]
        if "alarm_orbit" in d:
            alarm_orbit = d.at[best, "alarm_orbit"]
        else:
            alarm_orbit = d.at[best, "first_orbit"] + window
        alarm = clock.end_of(np.array([alarm_orbit]))[0]
        m.at[j, "delay_h"] = (alarm - man["start"]).total_seconds() / 3600
    return m, d


def summary(m: pd.DataFrame, d: pd.DataFrame) -> dict[str, float]:
    obs = m[m["observable"]]
    tp = int(obs["detected"].sum())
    fn = int((~obs["detected"]).sum())
    fp = int((d["status"] == "false_alarm").sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    sk = obs[obs["detected"] & (obs["type"] == "station_keeping")]
    dv_err = (sk["dv_est"] - sk["dv_t"]).abs()
    return {
        "manoeuvres": len(m), "observable": len(obs), "tp": tp, "fn": fn, "fp": fp,
        "duplicates": int((d["status"] == "duplicate").sum()),
        "recall": recall, "precision": precision, "f1": f1,
        "dv_abs_err_median_mm_s": float(dv_err.median() * 1e3) if len(sk) else np.nan,
        "dv_rel_err_median": float((dv_err / sk["dv_t"].abs()).median()) if len(sk) else np.nan,
        "delay_median_h": float(obs.loc[obs["detected"], "delay_h"].median()) if tp else np.nan,
    }


def recall_by(m: pd.DataFrame, column: str) -> pd.DataFrame:
    obs = m[m["observable"]]
    g = obs.groupby(column, observed=True)["detected"]
    return pd.DataFrame({"n": g.size(), "detected": g.sum(), "recall": g.mean()})


def da_bin(m: pd.DataFrame) -> pd.Series:
    labels = [f"{lo:g}-{hi:g} m" if np.isfinite(hi) else f">{lo:g} m"
              for lo, hi in zip(DA_BINS[:-1], DA_BINS[1:], strict=True)]
    return pd.cut(m["da_pred_m"].abs(), DA_BINS, labels=labels, right=False)
