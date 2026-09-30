"""Re-test the two known anomalies with the template-corrected series.

Step 1 looked for an orbital signature of the Sentinel-1A particle impact
(2016-08-23 17:07 UTC) and of the Sentinel-1B power anomaly (2021-12-23) on the raw
per-revolution series, whose noise (about 4.5 m) hides steps of a few metres. This
script repeats the test after removing the ground-track signature (step 3), where the
revolution noise is about 0.2 m.

For every revolution k, the jump of the corrected mean semi-major axis (and
inclination) is the difference between the means of the ``WINDOW`` revolutions on each
side (k skipped), corrected for drift. The jump at the event is compared with the
largest jumps of the same statistic over the quiet revolutions of the window (farther
than ``GUARD`` revolutions from any manoeuvre in the ESA record or flagged by the POD):
slow residual oscillations dominate this statistic, so the empirical range is a fairer
reference than a Gaussian z-score.

Sentinel-1B flies the same reference ground track as Sentinel-1A with a different orbit
numbering, so the Sentinel-1A template is applied after a phase shift chosen to
minimise the revolution-to-revolution noise of the corrected series.

    python scripts/check_anomalies.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from satwiser.config import REPO_ROOT, processed_dir, raw_dir
from satwiser.io.esa_history import read_manoeuvres
from satwiser.labels import manoeuvres
from satwiser.model.grid import to_grid
from satwiser.model.groundtrack import REPEAT
from satwiser.pipeline import series
from satwiser.plotting import DATA, EVENT, FAINT, figure, save

OUT = REPO_ROOT / "reports" / "anomalies"
WINDOW = 8
GUARD = 10
MDEG = np.rad2deg(1.0) * 1e3
EVENTS = [
    {"key": "s1a_impact", "satellite": "S1A", "label": "Sentinel-1A particle impact",
     "time": "2016-08-23 17:07", "search_h": 6, "span": ("2016-08-10", "2016-09-06")},
    {"key": "s1b_power", "satellite": "S1B", "label": "Sentinel-1B power anomaly",
     "time": "2021-12-23 12:00", "search_h": 12, "span": ("2021-12-10", "2022-01-06")},
]


def robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def jumps(y: np.ndarray, valid: np.ndarray, quiet: np.ndarray, window: int = WINDOW
          ) -> np.ndarray:
    """Step statistic at each grid position: mean of the ``window`` revolutions after k
    minus the mean of the ``window`` before (k skipped), minus the drift over the gap.

    The drift per revolution is the median consecutive difference on quiet revolutions.
    Means are less sensitive than line extrapolation to the slow residual oscillations of
    the corrected series.
    """
    ok = quiet[1:] & quiet[:-1]
    drift = float(np.median(np.diff(y)[ok]))
    out = np.full(len(y), np.nan)
    pos = np.arange(len(y))
    for k in range(window, len(y) - window):
        before = pos[k - window:k][valid[k - window:k]]
        after = pos[k + 1:k + 1 + window][valid[k + 1:k + 1 + window]]
        if len(before) < window // 2 or len(after) < window // 2:
            continue
        out[k] = y[after].mean() - y[before].mean() - drift * (after.mean() - before.mean())
    return out


def s1b_revolutions(span: tuple[str, str]) -> pd.DataFrame:
    folder = raw_dir("poeorb", "S1B")
    start, stop = pd.Timestamp(span[0]), pd.Timestamp(span[1])
    files = [p for p in series.orbit_files(folder, "S1B")
             if start - pd.Timedelta(days=1) <= series.validity_start(p) < stop]
    revs = series.stream_revolutions(series.iter_elements(files, workers=6), verbose=False)
    return revs[(revs.index >= start) & (revs.index < stop)]


def best_shift(grid: pd.DataFrame, template: np.ndarray, column: str, quiet: np.ndarray
               ) -> int:
    orbit = grid.index.to_numpy()
    y = grid[column].to_numpy()
    scores = []
    for shift in range(REPEAT):
        corrected = y - template[(orbit + shift) % REPEAT]
        d = np.diff(corrected)
        ok = quiet[1:] & quiet[:-1]
        scores.append(robust_sigma(d[ok]))
    return int(np.argmin(scores))


def analyse(ev: dict, model: dict, mans_a: pd.DataFrame) -> dict:
    tpl_a = np.asarray(model["template"]["values"]["a"])
    tpl_i = np.asarray(model["template"]["values"]["i"])
    start, stop = pd.Timestamp(ev["span"][0]), pd.Timestamp(ev["span"][1])
    if ev["satellite"] == "S1A":
        revs = pd.read_parquet(processed_dir() / "revolutions_S1A.parquet")
        revs = revs[(revs.index >= start) & (revs.index < stop)]
        man_times = list(mans_a.loc[(mans_a["start"] >= start) & (mans_a["start"] < stop),
                                    "start"])
    else:
        revs = s1b_revolutions(ev["span"])
        man_times = []  # no ESA history is published for Sentinel-1B
    g = to_grid(revs)
    valid = g["valid"].to_numpy()
    orbit = g.index.to_numpy()
    flagged = g["pod_flag"].fillna(False).to_numpy().astype(bool)
    near = np.zeros(len(g), bool)
    t = g["t"]
    for mt in man_times:
        k = int(np.argmin(np.abs((t - mt).dt.total_seconds().to_numpy())))
        near[max(0, k - GUARD):k + GUARD + 1] = True
    for k in np.flatnonzero(flagged):
        near[max(0, k - GUARD):k + GUARD + 1] = True
    quiet = valid & ~near
    shift = 0 if ev["satellite"] == "S1A" else best_shift(g, tpl_a, "a", quiet)
    a_c = g["a"].to_numpy() - tpl_a[(orbit + shift) % REPEAT]
    i_c = (g["i"].to_numpy() - tpl_i[(orbit + shift) % REPEAT]) * MDEG
    ja, ji = jumps(a_c, valid, quiet), jumps(i_c, valid, quiet)
    sa, si = robust_sigma(ja[quiet]), robust_sigma(ji[quiet])
    event = pd.Timestamp(ev["time"])
    around = np.abs((t - event).dt.total_seconds().to_numpy()) <= ev["search_h"] * 3600
    # Revolutions whose statistic windows touch a manoeuvre cannot test the anomaly.
    testable = around & ~near & np.isfinite(ja)
    za, zi = ja / sa, ji / si
    quiet_max_a = float(np.nanmax(np.abs(za[quiet])))
    quiet_max_i = float(np.nanmax(np.abs(zi[quiet])))
    manoeuvre_nearby = bool(near[around].any())
    tested = t[testable]
    if testable.any():
        k_a = np.flatnonzero(testable)[np.nanargmax(np.abs(za[testable]))]
        k_i = np.flatnonzero(testable)[np.nanargmax(np.abs(zi[testable]))]

    fig, axes = figure(2, 1, width=11, height=2.6, sharex=True)
    panels = ((axes[0, 0], a_c, "corrected a [m]"), (axes[1, 0], i_c, "corrected i [mdeg]"))
    for ax, y, unit in panels:
        ax.plot(t[valid], (y - np.nanmedian(y))[valid], color=DATA, lw=0.8)
        ax.axvline(event, color=EVENT, lw=1.2, ls="--")
        for mt in man_times:
            ax.axvline(mt, color=FAINT, lw=0.6)
        for k in np.flatnonzero(flagged):
            ax.axvline(t.iloc[k], color=FAINT, lw=0.6, ls=":")
        ax.set_ylabel(unit)
    axes[0, 0].set_title(f"{ev['label']}: template-corrected series (orange: event, grey: ESA "
                         "manoeuvres, dotted: POD manoeuvre flags)", fontsize=10, loc="left")
    save(fig, OUT / f"{ev['key']}.png")

    if not testable.any():
        return {"label": ev["label"], "satellite": ev["satellite"], "event_time": ev["time"],
                "search_window_h": ev["search_h"], "testable_revolutions": 0,
                "manoeuvre_within_search_window": manoeuvre_nearby,
                "figure": f"{ev['key']}.png"}
    return {
        "label": ev["label"], "satellite": ev["satellite"], "event_time": ev["time"],
        "testable_revolutions": int(testable.sum()),
        "tested_from": str(tested.min()), "tested_to": str(tested.max()),
        "search_window_h": ev["search_h"], "template_shift": shift,
        "revolutions": int(valid.sum()), "quiet_revolutions": int(quiet.sum()),
        "noise_jump_a_m": sa, "noise_jump_i_mdeg": si,
        "max_z_a_near_event": float(za[k_a]), "jump_a_m_near_event": float(ja[k_a]),
        "time_max_z_a": str(t.iloc[k_a]),
        "max_z_i_near_event": float(zi[k_i]), "jump_i_mdeg_near_event": float(ji[k_i]),
        "max_abs_z_a_quiet": quiet_max_a, "max_abs_z_i_quiet": quiet_max_i,
        "manoeuvre_within_search_window": manoeuvre_nearby,
        "figure": f"{ev['key']}.png",
    }


def verdict(r: dict) -> str:
    if not r["testable_revolutions"]:
        return "inconclusive: every revolution of the search window is next to a manoeuvre"
    a_out = abs(r["max_z_a_near_event"]) > r["max_abs_z_a_quiet"]
    i_out = abs(r["max_z_i_near_event"]) > r["max_abs_z_i_quiet"]
    scope = (" on the part of the search window not masked by a manoeuvre"
             if r["manoeuvre_within_search_window"] else "")
    if a_out or i_out:
        return "signature" + scope + ": a jump exceeds every jump seen on quiet revolutions"
    return ("no signature" + scope + ": jumps stay within the range of quiet revolutions "
            f"(|Δa| ≤ {abs(r['jump_a_m_near_event']):.2f} m, |Δi| ≤ "
            f"{abs(r['jump_i_mdeg_near_event']):.3f} mdeg)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = json.loads((processed_dir() / "model_S1A.json").read_text())
    mans_a = manoeuvres(read_manoeuvres(raw_dir("esa") / "s1a.man"))
    results = [analyse(ev, model, mans_a) for ev in EVENTS]
    for r in results:
        r["verdict"] = verdict(r)
    (OUT / "anomalies.json").write_text(json.dumps(results, indent=2))
    lines = ["# Known anomalies, re-tested on the template-corrected series", "",
             "Generated by `scripts/check_anomalies.py`.", ""]
    for r in results:
        if not r["testable_revolutions"]:
            lines += [f"## {r['label']}", "", f"- **Verdict: {r['verdict']}.**", ""]
            continue
        lines += [
            f"## {r['label']} ({r['event_time']} UTC, ±{r['search_window_h']} h)", "",
            f"- Tested revolutions: {r['testable_revolutions']}, from {r['tested_from']} to "
            f"{r['tested_to']} (revolutions within {GUARD} of a manoeuvre are excluded).",
            f"- {r['revolutions']} revolutions in the window, {r['quiet_revolutions']} quiet; "
            f"template phase shift {r['template_shift']}.",
            f"- Jump noise on quiet revolutions: {r['noise_jump_a_m']:.3f} m (a), "
            f"{r['noise_jump_i_mdeg']:.4f} mdeg (i).",
            f"- Largest jump near the event: a {r['jump_a_m_near_event']:+.3f} m "
            f"(z = {r['max_z_a_near_event']:+.1f}, at {r['time_max_z_a']}), i "
            f"{r['jump_i_mdeg_near_event']:+.4f} mdeg (z = {r['max_z_i_near_event']:+.1f}).",
            f"- Largest |z| on quiet revolutions of the window: a {r['max_abs_z_a_quiet']:.1f}, "
            f"i {r['max_abs_z_i_quiet']:.1f}.",
            f"- Manoeuvre inside the search window: {r['manoeuvre_within_search_window']}.",
            f"- **Verdict: {r['verdict']}.**", f"- Figure: `{r['figure']}`.", "",
        ]
    (OUT / "anomalies.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
