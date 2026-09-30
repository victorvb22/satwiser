"""Export the aggregates served by the API into ``$SATWISER_DATA_DIR/app``.

Everything the app shows is produced here from the pipeline outputs; nothing is typed
in by hand. Files:

- ``satellites.json``: list of satellites with their time span and summary counters;
- ``series_<SAT>.parquet``: one row per revolution (orbit, time, mean semi-major axis
  with the ground-track signature removed, F10.7, Ap);
- ``daily_<SAT>.parquet``: daily means for the mission overview;
- ``events_<SAT>.parquet``: detected manoeuvres, missed manoeuvres and false alarms of
  the main detector, with estimated and ESA Δv, jumps and class;
- ``robustness.json``: the precomputed degradation grid (step 4);
- ``lab/<event id>.json.gz``: ten-day windows of inertial states (60 s) for a curated set
  of events, used by the browser lab.

    python scripts/export_app_data.py --satellite S1A
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from satwiser import lab
from satwiser.classify import rule
from satwiser.config import REPO_ROOT, data_dir, processed_dir, raw_dir
from satwiser.labels import dv_from_da
from satwiser.pipeline.series import orbit_files, validity_start

NAMES = {"S1A": "Sentinel-1A", "S1B": "Sentinel-1B", "S1C": "Sentinel-1C"}
LAB_STEP = 6  # keep one state every 6 x 10 s = 60 s
LAB_PER_YEAR = 4
WINDOW_DAYS = 10
TYPE_TO_CLASS = {"station_keeping": "station_keeping", "inclination": "orbit_change",
                 "sequence": "orbit_change", "lowering": "orbit_change"}


def app_dir():
    path = data_dir() / "app"
    (path / "lab").mkdir(parents=True, exist_ok=True)
    return path


def build_events(sat: str, model: dict) -> pd.DataFrame:
    det = pd.read_parquet(processed_dir() / f"events_{sat}.parquet")
    mans = pd.read_parquet(processed_dir() / f"manoeuvres_eval_{sat}.parquet")
    ops = pd.Timestamp(model["operational_start"])
    coverage_end = mans["stop"].max()
    sigma = model["sigma"]
    di_thr, da_thr = 5 * sigma["i"], 3 * sigma["a"]

    det = det[det["status"] != "duplicate"].copy()
    det["class"] = rule(pd.DataFrame({"da": det["da"], "abs_di": det["di"].abs()}),
                        di_thr, da_thr)
    kind = {"detected": "detected", "false_alarm": "false_alarm", "unlabelled": "unscored"}
    events = pd.DataFrame({
        "satellite": sat,
        "kind": det["status"].map(kind),
        "time": pd.to_datetime(det["time"]),
        "orbit": det["orbit"].astype(int),
        "class": det["class"],
        "esa_type": det["esa_type"],
        "dv_est_mm_s": det["dv_est"] * 1e3,
        "dv_esa_mm_s": det["esa_dv_t"] * 1e3,
        "da_m": det["da"],
        "di_mdeg": det["di"],
        "de_1e6": det["de"],
        "statistic": det["z"].abs(),
        "channel": det["channel"],
        "alarm_delay_revs": det["alarm_orbit"] - det["orbit"],
        "split": det["split"],
    })
    scored = (mans["start"] >= ops) & (mans["stop"] <= coverage_end) & mans["observable"]
    missed = mans[scored & ~mans["detected"]]
    events = pd.concat([events, pd.DataFrame({
        "satellite": sat,
        "kind": "missed",
        "time": missed["start"],
        "orbit": missed["o0"].astype(int),
        "class": missed["type"].map(TYPE_TO_CLASS),
        "esa_type": missed["type"],
        "dv_est_mm_s": np.nan,
        "dv_esa_mm_s": missed["dv_t"] * 1e3,
        "da_m": np.nan, "di_mdeg": np.nan, "de_1e6": np.nan, "statistic": np.nan,
        "channel": None, "alarm_delay_revs": np.nan,
        "split": np.where(missed["start"] < pd.Timestamp(model["split"]), "calibration", "test"),
    })], ignore_index=True)
    events = events.sort_values("time").reset_index(drop=True)
    prefix = {"detected": "D", "false_alarm": "F", "missed": "M", "unscored": "U"}
    events.insert(0, "id", [f"{sat}-{prefix[k]}{o}" for k, o in
                            zip(events["kind"], events["orbit"], strict=True)])
    events["esa_da_m"] = events["dv_esa_mm_s"] * 1e-3 / dv_from_da(1.0)
    return events


def pick_lab_events(events: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """A few events per year for the lab: detected ones of each class, some misses and
    false alarms. Deterministic."""
    rng = np.random.default_rng(seed)
    chosen = []
    ev = events[events["kind"].isin(["detected", "missed", "false_alarm"])]
    for _, year in ev.groupby(ev["time"].dt.year):
        det = year[year["kind"] == "detected"]
        for cls in ("station_keeping", "orbit_change"):
            sub = det[det["class"] == cls]
            if len(sub):
                chosen.append(sub.iloc[rng.integers(len(sub))])
        for kind in ("missed", "false_alarm"):
            sub = year[year["kind"] == kind]
            if len(sub):
                chosen.append(sub.iloc[rng.integers(len(sub))])
    return pd.DataFrame(chosen).drop_duplicates("id")


def default_lab_event(events: pd.DataFrame) -> str:
    """Detected station-keeping manoeuvre of the test period whose ESA Δv is closest to
    the median of that population."""
    sk = events[(events["kind"] == "detected") & (events["class"] == "station_keeping")
                & (events["split"] == "test") & events["dv_esa_mm_s"].notna()]
    target = sk["dv_esa_mm_s"].median()
    return str(sk.iloc[(sk["dv_esa_mm_s"] - target).abs().argmin()]["id"])


def _lab_window(args):
    event, files, template, mans, detector, out = args
    center = pd.Timestamp(event["time"])
    start = (center - pd.Timedelta(days=WINDOW_DAYS / 2)).floor("h")
    stop = start + pd.Timedelta(days=WINDOW_DAYS)
    states = lab.load_states(files, start, stop).iloc[::LAB_STEP]
    inside = mans[(mans["start"] >= start) & (mans["start"] < stop)]
    payload = {
        "event_id": event["id"], "satellite": event["satellite"],
        "start": start.isoformat(), "step_s": lab.STEP_S * LAB_STEP,
        "event_time": center.isoformat(),
        "t_offset_s": ((states.index - start).total_seconds()).round(3).tolist(),
        "orbit": states["orbit"].astype(int).tolist(),
        "states": {c: states[c].round(4).tolist() for c in
                   ("rx", "ry", "rz", "vx", "vy", "vz")},
        "template_a": template,
        "detector": detector,
        "esa_manoeuvres": [{"start": s.isoformat(), "dv_t_mm_s": float(d * 1e3), "type": t}
                           for s, d, t in zip(inside["start"], inside["dv_t"], inside["type"],
                                              strict=True)],
    }
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return out


def export_lab(sat: str, chosen: pd.DataFrame, model: dict, grid: dict, workers: int) -> None:
    mans = pd.read_parquet(processed_dir() / f"manoeuvres_{sat}.parquet")
    all_files = orbit_files(raw_dir("poeorb", sat), sat)
    vstarts = np.array([validity_start(p) for p in all_files])
    jobs = []
    for _, event in chosen.iterrows():
        center = pd.Timestamp(event["time"])
        start = (center - pd.Timedelta(days=WINDOW_DAYS / 2)).floor("h")
        stop = start + pd.Timedelta(days=WINDOW_DAYS)
        files = [f for f, v in zip(all_files, vstarts, strict=True)
                 if start - pd.Timedelta(days=1) <= v < stop]
        jobs.append((event, files, model["template"]["values"]["a"], mans, grid["detector"],
                     app_dir() / "lab" / f"{event['id']}.json.gz"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_lab_window, jobs))


def _noise(values: np.ndarray) -> float:
    d = np.diff(values)
    d = d[np.isfinite(d)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


def build_metrics(sat: str) -> dict:
    """Headline numbers for the Method page, all read from the pipeline outputs."""
    step3 = json.loads((REPO_ROOT / "reports" / "step3" / "metrics.json").read_text())
    series = pd.read_parquet(processed_dir() / f"series_{sat}.parquet")
    model = json.loads((processed_dir() / f"model_{sat}.json").read_text())
    ops = model["operational_start"]
    template = model["template"]["values"]
    s = series[series["valid"] & (series["t"] >= ops)]
    s = s[np.diff(s.index.to_numpy(), prepend=s.index[0] - 1) == 1]  # consecutive orbits
    return {
        "satellite": sat,
        "detector": step3["cusum_config"],
        "test": step3["test"],
        "comparison": step3["comparison"],
        "noise_m": {"raw": _noise(s["a"].to_numpy()), "template": _noise(s["a_c"].to_numpy())},
        "model": {
            "operational_start": ops, "split": model["split"],
            "repeat_revolutions": model["template"]["repeat"],
            "template_ptp_a_m": float(np.ptp(template["a"])),
            "template_ptp_i_mdeg": float(np.ptp(template["i"]) * np.rad2deg(1.0) * 1e3),
            "drag_beta": model["drag"]["beta"],
            "sigma": model["sigma"],
        },
        "revolutions": int(len(series[series["valid"]])),
        "source": "reports/step3/metrics.json, processed series",
    }


def summary(events: pd.DataFrame) -> dict:
    scored = events[events["kind"].isin(["detected", "missed", "false_alarm"])]

    def counts(frame):
        return {"esa_manoeuvres": int(frame["kind"].isin(["detected", "missed"]).sum()),
                "detected": int((frame["kind"] == "detected").sum()),
                "missed": int((frame["kind"] == "missed").sum()),
                "false_alarms": int((frame["kind"] == "false_alarm").sum())}

    return {"all": counts(scored),
            "by_split": {s: counts(g) for s, g in scored.groupby("split")},
            "by_year": {str(y): counts(g) for y, g in scored.groupby(scored["time"].dt.year)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--satellite", default="S1A")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    sat = args.satellite
    out = app_dir()
    model = json.loads((processed_dir() / f"model_{sat}.json").read_text())
    grid = json.loads((processed_dir() / "robustness_grid.json").read_text())

    series = pd.read_parquet(processed_dir() / f"series_{sat}.parquet")
    series = series[series["valid"]].reset_index()
    rev = pd.DataFrame({"satellite": sat, "orbit": series["orbit"].astype(int),
                        "time": series["t"], "a_m": series["a_c"],
                        "f107": series["f107_obs"], "ap": series["ap"]})
    rev.to_parquet(out / f"series_{sat}.parquet", index=False)
    daily = rev.groupby(rev["time"].dt.normalize()).agg(
        a_m=("a_m", "mean"), f107=("f107", "first"), ap=("ap", "first")).reset_index()
    daily.insert(0, "satellite", sat)
    daily.rename(columns={"time": "date"}).to_parquet(out / f"daily_{sat}.parquet", index=False)

    events = build_events(sat, model)
    chosen = pick_lab_events(events)
    default_id = default_lab_event(events)
    if default_id not in set(chosen["id"]):
        chosen = pd.concat([chosen, events[events["id"] == default_id]])
    export_lab(sat, chosen, model, grid, args.workers)
    events["lab_available"] = events["id"].isin(set(chosen["id"]))
    events.to_parquet(out / f"events_{sat}.parquet", index=False)

    (out / "robustness.json").write_text(json.dumps(grid))
    (out / "metrics.json").write_text(json.dumps(build_metrics(sat), default=float))
    satellites = [{
        "id": sat, "name": NAMES[sat],
        "first": rev["time"].min().isoformat(), "last": rev["time"].max().isoformat(),
        "operational_start": model["operational_start"], "split": model["split"],
        "labels_end": events.loc[events["kind"] != "unscored", "time"].max().isoformat(),
        "years": sorted(rev["time"].dt.year.unique().tolist()),
        "default_lab_event": default_id,
        "summary": summary(events),
    }]
    (out / "satellites.json").write_text(json.dumps(satellites, indent=2))
    sizes = {p.name: p.stat().st_size for p in out.glob("*") if p.is_file()}
    lab_size = sum(p.stat().st_size for p in (out / "lab").glob("*.json.gz"))
    print(json.dumps({"files": sizes, "lab_windows": len(chosen), "lab_bytes": lab_size},
                     indent=2))


if __name__ == "__main__":
    main()
