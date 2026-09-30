"""Match detections to ESA manoeuvres and compute detection metrics.

A manoeuvre spans the orbits ``[o0, o1]`` containing its first burn start and last burn
end. A detection at orbit ``o`` matches it when ``o0 - 1 <= o <= o1 + 1`` (tolerance of
one revolution). Each manoeuvre is matched at most once; further detections inside the
same tolerance window are counted as duplicates, not as false alarms.

A manoeuvre is *observable* when both fitting windows around it hold at least half of
their revolutions; unobservable ones (data gaps) are reported but excluded from recall.

Detection delay is measured for a causal reading of the detector. Detections carrying
an ``alarm_orbit`` (causal detectors such as CUSUM) are available at the end of that
orbit. For the windowed baseline, the statistic at the first orbit ``f`` of the
detection run needs revolutions up to ``f + window``.
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
             window: int, start: str | None = None, stop: str | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (manoeuvre table, detection table) annotated with match results.

    ``start`` / ``stop`` restrict both tables to a time range (evaluation split).
    """
    clock = OrbitClock(revs)
    present = set(revs["orbit"].to_numpy().tolist())
    m = mans.copy()
    m["o0"] = clock.orbit_at(m["start"])
    m["o1"] = clock.orbit_at(m["stop"])
    need = int(np.ceil(window / 2))

    def count(lo: int, hi: int) -> int:
        return sum(o in present for o in range(lo, hi + 1))

    m["observable"] = [
        count(o0 - window, o0 - 1) >= need and count(o1 + 1, o1 + window) >= need
        for o0, o1 in zip(m["o0"], m["o1"], strict=True)
    ]
    d = detections.copy()
    d["time"] = clock.end_of(d["orbit"].to_numpy() - 1)
    t_first = revs["t_start"].iloc[0]
    t_last = revs["t_stop"].iloc[-1]
    m = m[(m["start"] >= t_first) & (m["stop"] <= t_last)]
    if start is not None:
        m = m[m["start"] >= start]
        d = d[d["time"] >= start]
    if stop is not None:
        m = m[m["start"] < stop]
        d = d[d["time"] < stop]
    m = m.reset_index(drop=True)
    d = d.reset_index(drop=True)

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
    m["dv_est"] = dv_from_da(m["jump_m"])
    m["phase"] = solar_phase(m["start"])
    d["phase"] = solar_phase(d["time"])
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
