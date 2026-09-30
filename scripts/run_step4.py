"""Step 4: degradation lab, robustness grid and presets.

1. Ten-day windows of precise state vectors are extracted: quiet windows (no ESA
   manoeuvre) for injection experiments, and windows centred on real test-period
   manoeuvres for replay.
2. Injection grid: for every combination of position noise, sampling and error
   correlation, each quiet window is degraded with a fresh noise draw, a manoeuvre of
   known along-track Δv (Δa = 2 a Δv / v) is added at a random epoch, and the lab
   detector is run. P(detection) is the fraction of trials with a detection within one
   bin of the injected epoch; the false-alarm rate comes from the same draws without
   injection.
3. Replay: the lab detector on real manoeuvre windows under each preset and along a
   noise sweep, scored against the ESA record.

Outputs: ``$SATWISER_DATA_DIR/processed/robustness_grid.json`` (for the app heatmap) and
``reports/step4``. Every number in the report comes from this script.

    python scripts/run_step4.py --satellite S1A
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from satwiser import lab
from satwiser.config import REPO_ROOT, processed_dir, raw_dir
from satwiser.labels import A_REF, V_REF
from satwiser.pipeline.series import orbit_files, validity_start
from satwiser.plotting import DATA, EVENT, FAINT, MUTED, TEXT, figure, legend, save

OUT = REPO_ROOT / "reports" / "step4"
WINDOW_DAYS = 10
SIGMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
DVS_CM = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
POINTS = (8640, 1440, 144, 24, 4, 2, 1)
RHOS = (0.0, 0.5, 0.9, 0.95)
N_QUIET = 40
TRIALS = 60
N_REAL = 60
SEED = 20260930

DETECTOR_GRID = [lab.LabDetector(w, thr, "window", 0.0)
                 for w, thr in product((6, 12, 24), (3.0, 4.0, 5.0, 6.0, 8.0))]
DETECTOR_GRID += [lab.LabDetector(w, thr, "diff", floor)
                  for w, thr, floor in product((6, 12, 24), (3.0, 4.0, 5.0, 6.0, 8.0),
                                               (0.0, 0.1, 0.3, 1.0))]

_WINDOWS: list[pd.DataFrame] = []
_TEMPLATE: np.ndarray | None = None
_DET: lab.LabDetector = lab.LabDetector()


# ----------------------------------------------------------------------------- windows

def quiet_windows(mans: pd.DataFrame, start: pd.Timestamp, stop: pd.Timestamp,
                  margin_days: float = 1.0) -> list[pd.Timestamp]:
    """Start epochs of ten-day windows lying in gaps between manoeuvres."""
    m = mans[(mans["start"] >= start) & (mans["stop"] <= stop)].sort_values("start")
    length = pd.Timedelta(days=WINDOW_DAYS)
    margin = pd.Timedelta(days=margin_days)
    starts = []
    for prev_stop, next_start in zip(m["stop"].iloc[:-1], m["start"].iloc[1:], strict=True):
        if next_start - prev_stop >= length + 2 * margin:
            starts.append((prev_stop + margin).ceil("h"))
    return starts


def _load(args):
    files, start, stop, path = args
    if not path.exists():
        lab.load_states(files, start, stop).to_parquet(path)
    return path


def cache_windows(starts: list[pd.Timestamp], kind: str, sat: str, workers: int = 6):
    folder = processed_dir("lab_windows")
    all_files = orbit_files(raw_dir("poeorb", sat), sat)
    vstarts = np.array([validity_start(p) for p in all_files])
    jobs = []
    for start in starts:
        stop = start + pd.Timedelta(days=WINDOW_DAYS)
        sel = [f for f, v in zip(all_files, vstarts, strict=True)
               if start - pd.Timedelta(days=1) <= v < stop]
        jobs.append((sel, start, stop, folder / f"{sat}_{kind}_{start:%Y%m%dT%H}.parquet"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_load, jobs))


# ------------------------------------------------------------------------ injection grid

def _init(paths, template, det):
    global _WINDOWS, _TEMPLATE, _DET
    _WINDOWS = [pd.read_parquet(p) for p in paths]
    _TEMPLATE = np.asarray(template)
    _DET = det


def _cell(args):
    """All Δv for one (sigma, points, rho): hits per Δv and false alarms."""
    sigma, points, rho, seed = args
    rng = np.random.default_rng(seed)
    deg = lab.Degradation(sigma, points, rho)
    det = _DET
    b = lab.bin_revolutions(points)
    w = lab.detector_window(det, b)
    expected = lab.expected_per_bin(points, b)
    hits = np.zeros(len(DVS_CM), int)
    false_alarms = 0
    usable = 0
    for trial in range(TRIALS):
        states = _WINDOWS[trial % len(_WINDOWS)]
        idx = lab.subsample(len(states), points, int(rng.integers(0, 8640)))
        raw = states[["rx", "ry", "rz", "vx", "vy", "vz"]].to_numpy()[idx]
        a = lab.mean_a_nosp(lab.degrade(raw, deg, rng.standard_normal(raw.shape)))
        orbit = states["orbit"].to_numpy()[idx]
        t = idx.astype(float)
        t0 = rng.uniform(0.2, 0.8) * len(states)
        o0 = orbit[np.searchsorted(t, t0).clip(0, len(orbit) - 1)]
        k0 = (o0 - orbit.min()) // b
        bins, means = lab.bin_means(a, orbit, _TEMPLATE, b, expected)
        _, z = lab.window_scores(means, w, det.normalisation, det.floor_m)
        if np.isnan(z).all():
            continue
        usable += 1
        false_alarms += len(lab.window_detect(z, det.threshold, w))
        for j, dv in enumerate(DVS_CM):
            da = 2 * A_REF * dv * 1e-2 / V_REF
            _, means_j = lab.bin_means(a + da * (t > t0), orbit, _TEMPLATE, b, expected)
            _, zj = lab.window_scores(means_j, w, det.normalisation, det.floor_m)
            found = lab.window_detect(zj, det.threshold, w)
            hits[j] += any(abs(k - k0) <= 1 for k in found)
    return sigma, points, rho, hits.tolist(), false_alarms, usable


def min_detectable(dvs: np.ndarray, p: np.ndarray, level: float = 0.9) -> float:
    """Smallest Δv reaching ``level`` detection probability (log interpolation)."""
    above = np.flatnonzero(p >= level)
    if above.size == 0:
        return np.nan
    j = above[0]
    if j == 0:
        return float(dvs[0])
    x0, x1 = np.log(dvs[j - 1]), np.log(dvs[j])
    frac = (level - p[j - 1]) / (p[j] - p[j - 1])
    return float(np.exp(x0 + frac * (x1 - x0)))


# ----------------------------------------------------------------------------- figures

def fig_heatmaps(grid: pd.DataFrame, path):
    fig, axes = figure(1, 3, width=13, height=3.6)
    for ax, (label, deg) in zip(axes[0], lab.PRESETS.values(), strict=True):
        cell = grid[(grid["points"] == deg.points_per_day) & (grid["rho"] == deg.rho)]
        mat = cell.pivot(index="dv_cm", columns="sigma", values="p_detect").sort_index(
            ascending=False)
        ax.imshow(mat.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(mat.columns)), [f"{s:g}" for s in mat.columns], rotation=60)
        ax.set_yticks(range(len(mat.index)), [f"{d:g}" for d in mat.index])
        ci = list(mat.columns).index(deg.sigma_m)
        ax.add_patch(Rectangle((ci - 0.5, -0.5), 1, len(mat.index), fill=False,
                               edgecolor=EVENT, lw=1.5))
        ax.set_xlabel("position noise σ [m]")
        ax.set_ylabel("Δv [cm/s]")
        ax.set_title(f"P(detect): {deg.points_per_day:g} pts/day, ρ = {deg.rho:g} ({label})",
                     fontsize=9, loc="left")
    save(fig, path)


def fig_min_dv(grid_min: pd.DataFrame, path):
    fig, axes = figure(1, 1, width=9, height=3.6)
    ax = axes[0, 0]
    colors = [DATA, "#7FA8E0", "#6590C8", MUTED, FAINT, EVENT, TEXT]
    for c, points in zip(colors, POINTS, strict=True):
        sub = grid_min[(grid_min["points"] == points) & (grid_min["rho"] == 0.0)]
        ax.plot(sub["sigma"], sub["min_dv_cm"], marker="o", ms=3, lw=1, color=c,
                label=f"{points:g} pts/day")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("position noise σ [m]")
    ax.set_ylabel("smallest Δv detected at 90 % [cm/s]")
    ax.set_title("Robustness curve (uncorrelated errors; missing points: never reaches 90 %"
                 " up to 1 m/s)", fontsize=9, loc="left")
    legend(ax, loc="upper left", ncol=2)
    save(fig, path)




# ------------------------------------------------------------------------------ replay

def replay_many(windows, mans, template, deg: lab.Degradation, seed: int,
                detectors: list[lab.LabDetector]) -> list[dict]:
    """Score several detector settings on real manoeuvre windows.

    Each window is degraded and binned once (same noise draw for every detector), then
    every detector runs on the same bin means. Truth bins use the first *sampled* orbit
    as origin, like :func:`satwiser.lab.bin_means`.
    """
    rng = np.random.default_rng(seed)
    counts = [{"tp": 0, "fn": 0, "fp": 0} for _ in detectors]
    for states in windows:
        binned = lab.degraded_means(states, template, deg, rng, int(rng.integers(0, 8640)))
        b, o_first = binned["revs_per_bin"], binned["first_orbit"]
        n_bins = len(binned["bins"])
        orbit = states["orbit"].to_numpy()
        t = states.index
        inside = mans[(mans["start"] >= t[0]) & (mans["stop"] < t[-1])]
        spans = [((orbit[np.searchsorted(t, m["start"])] - o_first) // b,
                  (orbit[min(np.searchsorted(t, m["stop"]), len(t) - 1)] - o_first) // b)
                 for _, m in inside.iterrows()]
        for det, count in zip(detectors, counts, strict=True):
            res = lab.detect_on_means(binned, det)
            if np.isnan(res["z"]).all():
                continue
            w = res["window"]
            truth = [(k, k1) for k, k1 in spans if w <= k and k1 < n_bins - w]
            found = list(res["detections"])
            matched = set()
            for k, k1 in truth:
                hit = [f for f in found if k - 1 <= f <= k1 + 1 and f not in matched]
                if hit:
                    matched.add(hit[0])
                    count["tp"] += 1
                else:
                    count["fn"] += 1
            # Detections next to any manoeuvre (scorable or at the edges) are not false
            # alarms.
            count["fp"] += sum(1 for f in found if f not in matched
                               and all(abs(f - k) > 1 for k, _ in spans))
    out = []
    for count in counts:
        tp, fn, fp = count["tp"], count["fn"], count["fp"]
        out.append({**count, "recall": tp / (tp + fn) if tp + fn else np.nan,
                    "precision": tp / (tp + fp) if tp + fp else np.nan,
                    "f1": 2 * tp / (2 * tp + fn + fp) if tp else 0.0})
    return out


def detector_from(row) -> lab.LabDetector:
    return lab.LabDetector(int(row["window_revs"]), float(row["threshold"]),
                           str(row["normalisation"]), float(row["floor_m"]))


# ------------------------------------------------------------------------------ stages

def select_windows(mans, ops, split, coverage_end, sat, workers) -> dict:
    """Quiet windows (injection), test-period and calibration-period manoeuvre windows."""
    rng = np.random.default_rng(SEED)
    half = pd.Timedelta(days=WINDOW_DAYS / 2)
    starts = quiet_windows(mans, ops, coverage_end)
    pick = np.sort(rng.choice(len(starts), size=min(N_QUIET, len(starts)), replace=False))
    quiet_starts = [starts[k] for k in pick]

    def centred(sub):
        chosen = np.sort(rng.choice(len(sub), size=min(N_REAL, len(sub)), replace=False))
        return [(sub["start"].iloc[k] - half).floor("h") for k in chosen]

    real_starts = centred(mans[(mans["start"] >= split + half)
                               & (mans["stop"] <= coverage_end - half)])
    cal_starts = centred(mans[(mans["start"] >= ops + half) & (mans["stop"] <= split - half)])
    return {
        "quiet_starts": quiet_starts,
        "quiet_paths": cache_windows(quiet_starts, "quiet", sat, workers),
        "real": [pd.read_parquet(p) for p in cache_windows(real_starts, "real", sat, workers)],
        "calib": [pd.read_parquet(p) for p in cache_windows(cal_starts, "calib", sat, workers)],
    }


def calibrate_lab_detector(win: dict, mans, template) -> dict:
    """Select the lab detector on calibration-period windows (POD preset)."""
    pod = lab.PRESETS["pod"][1]
    scores = replay_many(win["calib"], mans, template, pod, SEED + 3, DETECTOR_GRID)
    det_grid = pd.DataFrame([{**c.__dict__, **s} for c, s in zip(DETECTOR_GRID, scores,
                                                                  strict=True)])
    det_grid.to_csv(OUT / "lab_detector_calibration.csv", index=False)
    # Ties on calibration F1 and precision are broken towards the difference-based scale,
    # whose value does not depend on how many manoeuvres the window holds.
    det_grid["prefer"] = (det_grid["normalisation"] == "diff").astype(int)
    ranked = det_grid.sort_values(["f1", "precision", "prefer"], ascending=False)
    best = ranked.iloc[0]
    det = detector_from(best)
    tied = ranked[(ranked["f1"] == best["f1"]) & (ranked["precision"] == best["precision"])]
    ref = det_grid[(det_grid["window_revs"] == 12) & (det_grid["threshold"] == 4.0)
                   & (det_grid["normalisation"] == "window")].iloc[0]
    candidates = [detector_from(r) for _, r in tied.iterrows()] + [det, lab.LabDetector()]
    test = replay_many(win["real"], mans, template, pod, SEED + 1, candidates)
    tied_tab = pd.DataFrame([{"W": c.window_revs, "threshold": c.threshold,
                              "normalisation": c.normalisation, "floor_m": c.floor_m,
                              "calibration f1": row["f1"], "test recall": s["recall"],
                              "test precision": s["precision"]}
                             for c, (_, row), s in zip(candidates[:len(tied)], tied.iterrows(),
                                                       test[:len(tied)], strict=True)])
    return {"det": det, "best": best, "ref": ref, "tied": tied_tab,
            "test_best": test[len(tied)], "test_ref": test[len(tied) + 1]}


def run_injection_grid(win: dict, template, det, workers
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = [(s, p, r, SEED + k) for k, (s, p, r) in enumerate(product(SIGMAS, POINTS, RHOS))]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(win["quiet_paths"], template, det)) as pool:
        for sigma, points, rho, hits, fa, usable in pool.map(_cell, cells, chunksize=2):
            for dv, h in zip(DVS_CM, hits, strict=True):
                rows.append({"sigma": sigma, "points": points, "rho": rho, "dv_cm": dv,
                             "p_detect": h / usable if usable else np.nan,
                             "fa_per_window": fa / usable if usable else np.nan,
                             "trials": usable})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "injection_grid.csv", index=False)
    grid_min = (grid.groupby(["sigma", "points", "rho"])
                .apply(lambda g: pd.Series({
                    "min_dv_cm": min_detectable(g["dv_cm"].to_numpy(), g["p_detect"].to_numpy()),
                    "fa_per_window": g["fa_per_window"].iloc[0]}), include_groups=False)
                .reset_index())
    return grid, grid_min


def presets_and_sweep(win: dict, mans, template, det, grid_min
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for key, (label, deg) in lab.PRESETS.items():
        g = grid_min[(grid_min["sigma"] == deg.sigma_m)
                     & (grid_min["points"] == deg.points_per_day)
                     & (grid_min["rho"] == deg.rho)].iloc[0]
        rep = replay_many(win["real"], mans, template, deg, SEED + 1, [det])[0]
        rows.append({"preset": label, "key": key, "sigma_m": deg.sigma_m,
                     "points_per_day": deg.points_per_day, "rho": deg.rho,
                     "min_dv_90_cm_s": g["min_dv_cm"],
                     "false_alarms_per_10_days": g["fa_per_window"],
                     "replay_recall": rep["recall"], "replay_precision": rep["precision"],
                     "replay_tp": rep["tp"], "replay_fn": rep["fn"], "replay_fp": rep["fp"]})
    presets = pd.DataFrame(rows).set_index("preset")
    sweep = pd.DataFrame([{"sigma_m": sigma, **replay_many(
        win["real"], mans, template, lab.Degradation(sigma, 8640, 0.0), SEED + 2, [det])[0]}
        for sigma in SIGMAS]).set_index("sigma_m")
    return presets, sweep


def export_grid(grid, grid_min, presets, det, template) -> None:
    """Grid for the app heatmap, with the lab configuration (detector, template)."""
    def cell(s, dv, p, r):
        return float(grid[(grid["sigma"] == s) & (grid["dv_cm"] == dv) & (grid["points"] == p)
                          & (grid["rho"] == r)]["p_detect"].iloc[0])

    def min_dv(s, p, r):
        v = grid_min[(grid_min["sigma"] == s) & (grid_min["points"] == p)
                     & (grid_min["rho"] == r)]["min_dv_cm"].iloc[0]
        return None if np.isnan(v) else float(v)

    export = {
        "axes": {"sigma_m": SIGMAS, "dv_cm_s": DVS_CM, "points_per_day": POINTS, "rho": RHOS},
        "p_detect": [[[[cell(s, dv, p, r) for r in RHOS] for p in POINTS] for dv in DVS_CM]
                     for s in SIGMAS],
        "p_detect_index_order": ["sigma_m", "dv_cm_s", "points_per_day", "rho"],
        "min_dv_90_cm_s": [[[min_dv(s, p, r) for r in RHOS] for p in POINTS] for s in SIGMAS],
        "presets": {row["key"]: {"label": name, "sigma_m": row["sigma_m"],
                                 "points_per_day": row["points_per_day"], "rho": row["rho"]}
                    for name, row in presets.iterrows()},
        "detector": det.__dict__, "trials_per_cell": TRIALS, "window_days": WINDOW_DAYS,
        "template_a": [float(v) for v in template],
    }
    (processed_dir() / "robustness_grid.json").write_text(json.dumps(export))


def render_report(win, cal, presets, grid_min, sweep, ops, split) -> str:
    det, best, ref = cal["det"], cal["best"], cal["ref"]
    years = pd.Series([s.year for s in win["quiet_starts"]]).value_counts().sort_index()
    base = grid_min[(grid_min["points"] == 8640) & (grid_min["rho"] == 0.0)].set_index("sigma")
    lines = [
        "# Step 4 — degradation and robustness curve",
        "",
        "Generated by `scripts/run_step4.py`. All numbers are computed by the script.",
        "",
        "## Setup",
        "",
        "- Lab detector (same as the browser): windowed step statistic on the template-"
        f"corrected mean semi-major axis over {WINDOW_DAYS}-day windows, W scaled to bins "
        "when sampling is sparse. Settings calibrated below.",
        "- Degradation: position noise σ per axis and velocity noise n·σ (orbit-like error), "
        "AR(1) correlation ρ between output samples, even subsampling. See `satwiser.lab`.",
        f"- Injection grid: {len(win['quiet_starts'])} quiet ten-day windows (no ESA manoeuvre; "
        f"years: {', '.join(f'{y}: {n}' for y, n in years.items())}), {TRIALS} trials per "
        f"cell, {len(SIGMAS)} noise × {len(DVS_CM)} Δv × {len(POINTS)} sampling × "
        f"{len(RHOS)} correlation levels. A trial counts as detected when an alarm falls "
        "within one bin of the injected epoch.",
        f"- Replay: {len(win['real'])} ten-day windows centred on randomly drawn test-period "
        "ESA manoeuvres; all ESA manoeuvres whose detector windows fit in the data are scored "
        "(one-bin tolerance, bins counted from the first sampled orbit).",
        "- Quiet windows need ten days without manoeuvre, which is rare at solar maximum: the "
        "injection windows are biased towards low solar activity (see years above).",
        "",
        "## Lab detector calibration",
        "",
        f"- {len(DETECTOR_GRID)} settings (W, threshold, normalisation, floor) scored by F1 "
        f"on {len(win['calib'])} ten-day windows centred on calibration-period manoeuvres "
        f"({ops:%Y-%m} to {split:%Y-%m}), POD preset.",
        f"- Selected: **W = {det.window_revs}, |z| > {det.threshold:g}, "
        f"{det.normalisation} normalisation"
        + (f", floor {det.floor_m:g} m" if det.normalisation == "diff" else "") + "**; "
        f"calibration F1 {best['f1']:.3f} (recall {best['recall']:.3f}, precision "
        f"{best['precision']:.3f}).",
        f"- {len(cal['tied'])} setting(s) tie on calibration F1 and precision; ties are broken "
        "towards the difference-based normalisation, whose scale does not depend on how many "
        "manoeuvres a window holds (at solar maximum a ten-day window often holds three). "
        "This rule was fixed after an earlier run of this script had shown the test "
        "behaviour of both normalisations, so the test scores of all tied settings are "
        "reported:",
        "",
        cal["tied"].to_markdown(index=False, floatfmt=".3f"),
        "",
        f"- Starting point (W = 12, |z| > 4, window-wide median/MAD): calibration F1 "
        f"{ref['f1']:.3f}.",
        f"- On the test windows, POD preset: selected recall {cal['test_best']['recall']:.3f}, "
        f"precision {cal['test_best']['precision']:.3f}; starting point recall "
        f"{cal['test_ref']['recall']:.3f}, precision {cal['test_ref']['precision']:.3f}.",
        "- Window-wide normalisation suffers in busy windows: each manoeuvre inflates the "
        "MAD over 2W values of the statistic. The difference-based scale is affected by a "
        "step only once.",
        "",
        "## Presets",
        "",
        "Illustrative settings, not sensor specifications. `Type TLE` is an approximation: "
        "real TLEs are SGP4 mean elements with non-Gaussian, strongly along-track errors; "
        "here it is Gaussian orbit-like noise at TLE-like amplitude and cadence.",
        "",
        presets.drop(columns="key").to_markdown(floatfmt=".3g"),
        "",
        "## Robustness curve (full 10 s sampling, uncorrelated errors)",
        "",
        base[["min_dv_cm", "fa_per_window"]].rename(columns={
            "min_dv_cm": "smallest Δv at 90 % [cm/s]",
            "fa_per_window": "false alarms per 10 days"}).to_markdown(floatfmt=".3g"),
        "",
        "Empty values: 90 % is not reached within the tested range (up to 1 m/s).",
        "",
        "## Replay on real manoeuvres (full sampling, uncorrelated errors)",
        "",
        sweep[["tp", "fn", "fp", "recall", "precision"]].to_markdown(floatfmt=".3f"),
        "",
        "## Effect of sampling and correlation (smallest Δv at 90 %, cm/s)",
        "",
    ]
    for rho in RHOS:
        tab = grid_min[grid_min["rho"] == rho].pivot(index="sigma", columns="points",
                                                      values="min_dv_cm")[list(POINTS)]
        lines += [f"ρ = {rho:g}", "", tab.to_markdown(floatfmt=".3g"), ""]
    lines += [
        "## Reading the curves",
        "",
        "- At 144 samples per day and below (one every ten minutes or fewer) the curves "
        "flatten: the per-bin mean no longer averages out the short-period terms that the "
        "first-order J2 correction leaves in the semi-major axis, so the floor does not "
        "depend on the injected noise. Classical higher-order zonal models would not remove "
        "it: `scripts/check_short_period_floor.py` shows that this residual is itself a "
        "ground-track signature (see `short_period_floor.json` for the out-of-sample "
        "reduction obtained with a sample-level template). It is not used for the presets: "
        "it is prior knowledge from precise orbits that a TLE-only user would not have.",
        "- Correlated errors (large rho) hurt most at dense sampling, where averaging "
        "many samples no longer reduces the noise.",
        "",
        "## Figures",
        "",
        "- `heatmaps_presets.png`: P(detection) over noise × Δv at each preset's sampling and "
        "correlation (orange frame: preset noise).",
        "- `robustness_curve.png`: smallest Δv detected at 90 % against noise, per sampling.",
        "",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--satellite", default="S1A")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = json.loads((processed_dir() / f"model_{args.satellite}.json").read_text())
    template = np.asarray(model["template"]["values"]["a"])
    mans = pd.read_parquet(processed_dir() / f"manoeuvres_{args.satellite}.parquet")
    ops, split = pd.Timestamp(model["operational_start"]), pd.Timestamp(model["split"])

    win = select_windows(mans, ops, split, mans["stop"].max(), args.satellite, args.workers)
    cal = calibrate_lab_detector(win, mans, template)
    grid, grid_min = run_injection_grid(win, template, cal["det"], args.workers)
    presets, sweep = presets_and_sweep(win, mans, template, cal["det"], grid_min)
    export_grid(grid, grid_min, presets, cal["det"], template)
    fig_heatmaps(grid, OUT / "heatmaps_presets.png")
    fig_min_dv(grid_min, OUT / "robustness_curve.png")
    report = render_report(win, cal, presets, grid_min, sweep, ops, split)
    (OUT / "robustness.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
