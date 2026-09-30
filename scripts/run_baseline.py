"""Calibrate and evaluate the baseline detector against the ESA manoeuvre history.

Strict temporal split: the window and threshold are chosen on the calibration period
only (best F1), then frozen and applied to the test period. Every number written to
``reports/step2`` comes from this script.

    python scripts/run_baseline.py --satellite S1A
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from satwiser.config import REPO_ROOT, processed_dir, raw_dir  # noqa: E402
from satwiser.detection.baseline import BaselineConfig, detect, scores  # noqa: E402
from satwiser.evaluation import da_bin, evaluate, recall_by, summary  # noqa: E402
from satwiser.io.esa_history import read_manoeuvres  # noqa: E402
from satwiser.labels import MAX_GAP_H, manoeuvres  # noqa: E402

OUT = REPO_ROOT / "reports" / "step2"
CALIBRATION = ("2014-04-01", "2020-01-01")
TEST = ("2020-01-01", "2027-01-01")
WINDOWS = (2, 4, 6, 8, 12, 16)
THRESHOLDS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0)

BG, DATA, EVENT, FAINT, MUTED, TEXT = "#06080E", "#9CC2FF", "#F4A259", "#5E6782", "#8C95AB", \
    "#E8ECF4"


def run(revs: pd.DataFrame, mans: pd.DataFrame, cfg: BaselineConfig, period: tuple[str, str]
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    sc = scores(revs, cfg)
    det = detect(sc, cfg)
    m, d = evaluate(det, mans, revs, cfg.window, *period)
    return sc, m, d, summary(m, d)


def calibrate(revs: pd.DataFrame, mans: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in WINDOWS:
        sc = scores(revs, BaselineConfig(window=w))
        for thr in THRESHOLDS:
            cfg = BaselineConfig(window=w, threshold=thr)
            m, d = evaluate(detect(sc, cfg), mans, revs, w, *CALIBRATION)
            rows.append({"window": w, "threshold": thr, **summary(m, d)})
    return pd.DataFrame(rows)


def style(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#1E2436")
    ax.tick_params(colors=FAINT, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color="#161B2A", lw=0.6)


def year_figure(revs, m, d, year: int, path):
    r = revs[revs.index.year == year]
    fig, (ax, axf) = plt.subplots(2, 1, figsize=(12, 4.6), height_ratios=(4, 1), sharex=True)
    fig.patch.set_facecolor(BG)
    for a_ in (ax, axf):
        style(a_)
    ax.plot(r.index, (r["a"] - 7.07e6), color=DATA, lw=0.6)
    y = r["a"].reindex(r.index)
    t_of = pd.Series(revs.index, index=revs["orbit"])
    tp = m[m["detected"] & (m["start"].dt.year == year)]
    fn = m[~m["detected"] & m["observable"] & (m["start"].dt.year == year)]
    fa = d[(d["status"] == "false_alarm") & (d["time"].dt.year == year)]

    def level(times):
        idx = np.searchsorted(r.index, times).clip(0, len(r) - 1)
        return y.iloc[idx].to_numpy() - 7.07e6

    ax.scatter(tp["start"], level(tp["start"]), s=22, color=EVENT, zorder=3, label="detected")
    ax.scatter(fn["start"], level(fn["start"]), s=26, facecolors="none", edgecolors=EVENT,
               zorder=3, label="missed")
    fa_t = t_of.reindex(fa["orbit"]).dropna()
    ax.scatter(fa_t, level(fa_t), s=24, marker="D", color=FAINT, zorder=3, label="false alarm")
    ax.set_ylabel("mean a − 7070 km [m]")
    ax.set_title(f"Sentinel-1A {year}: per-revolution mean semi-major axis and baseline "
                 "detections", fontsize=10, loc="left")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for text in leg.get_texts():
        text.set_color(MUTED)
    axf.fill_between(r.index, r["f107_obs"], color=DATA, alpha=0.25, lw=0)
    axf.set_ylabel("F10.7")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


def recall_figure(m_test, path):
    tab = recall_by(m_test.assign(bin=da_bin(m_test)), "bin")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor(BG)
    style(ax)
    ax.bar(range(len(tab)), tab["recall"], color=EVENT, width=0.6)
    ax.set_xticks(range(len(tab)), [f"{b}\n(n={n})" for b, n in zip(tab.index, tab["n"],
                                                                     strict=True)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall")
    ax.set_title("Test-period recall by expected |Δa| (ESA record)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


def fmt(x, digits=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--satellite", default="S1A")
    args = parser.parse_args()
    sat = args.satellite
    OUT.mkdir(parents=True, exist_ok=True)

    revs = pd.read_parquet(processed_dir() / f"revolutions_{sat}.parquet")
    burns = read_manoeuvres(raw_dir("esa") / f"{sat.lower()}.man")
    mans = manoeuvres(burns)
    gaps_h = burns["start"].diff().dt.total_seconds().dropna() / 3600
    edges = [0, 1, 2, 3, 5, 8, 24, 96, np.inf]
    counts = np.histogram(gaps_h, bins=edges)[0]
    gap_hist = {f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}": int(n)
                for lo, hi, n in zip(edges[:-1], edges[1:], counts, strict=True)}
    mans.to_parquet(processed_dir() / f"manoeuvres_{sat}.parquet")

    grid = calibrate(revs, mans)
    grid.to_csv(OUT / "calibration_grid.csv", index=False)
    best = grid.sort_values(["f1", "precision"], ascending=False).iloc[0]
    cfg = BaselineConfig(window=int(best["window"]), threshold=float(best["threshold"]))
    # The brief's starting point, single-revolution jumps at z = 5, for reference.
    ref = grid[(grid["window"] == 2) & (grid["threshold"] == 5.0)].iloc[0]

    sc, m_cal, d_cal, s_cal = run(revs, mans, cfg, CALIBRATION)
    _, m_test, d_test, s_test = run(revs, mans, cfg, TEST)
    s_test_inplane = summary(m_test[m_test["type"] != "inclination"], d_test)
    det_all = detect(sc, cfg)
    det_all.to_parquet(processed_dir() / f"detections_baseline_{sat}.parquet")
    m_all, d_all = evaluate(det_all, mans, revs, cfg.window)

    by_type = recall_by(m_test, "type")
    by_bin = recall_by(m_test.assign(bin=da_bin(m_test)), "bin")
    by_phase_rows = []
    for phase, g in m_all.groupby("phase"):
        s = summary(g, d_all[d_all["phase"] == phase])
        by_phase_rows.append({"phase": phase, **s})
    by_phase = pd.DataFrame(by_phase_rows).set_index("phase")

    # Revolution-to-revolution noise per year, away from manoeuvres.
    near = np.zeros(len(revs), bool)
    orb = revs["orbit"].to_numpy()
    for o0, o1 in zip(m_all["o0"], m_all["o1"], strict=True):
        near |= (orb >= o0 - 1) & (orb <= o1 + 2)
    jumps = pd.Series(np.diff(revs["a"].to_numpy(), prepend=np.nan), index=revs.index)
    consecutive = np.diff(orb, prepend=orb[0] - 2) == 1
    quiet_j = jumps[~near & consecutive]
    noise_year = quiet_j.groupby(quiet_j.index.year).agg(
        lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))
    f107_year = revs["f107_obs"].groupby(revs.index.year).mean()
    noise_tab = pd.DataFrame({"noise_m": noise_year, "mean_f107": f107_year,
                              "revolutions": revs.groupby(revs.index.year).size()})
    noise_tab.index.name = "year"

    # POD flag cross-check of the labels (never used by the detector).
    flagged = revs.loc[revs["pod_flag"], "orbit"].to_numpy()
    covered = np.zeros(len(flagged), bool)
    for o0, o1 in zip(m_all["o0"], m_all["o1"], strict=True):
        covered |= (flagged >= o0 - 1) & (flagged <= o1 + 1)
    obs_all = m_all[m_all["observable"]]
    man_flagged = np.array([((flagged >= o0 - 1) & (flagged <= o1 + 1)).any()
                            for o0, o1 in zip(obs_all["o0"], obs_all["o1"], strict=True)])

    fa = d_all[d_all["status"] == "false_alarm"]
    unexplained = flagged[~covered]
    fa_on_flag = sum(bool((np.abs(unexplained - o) <= 1).any()) for o in fa["orbit"])

    for year in (2016, 2023):
        year_figure(revs, m_all, d_all, year, OUT / f"mission_{year}.png")
    recall_figure(m_test, OUT / "recall_by_da.png")

    metrics = {"config": cfg.__dict__, "calibration": s_cal, "test": s_test,
               "test_in_plane": s_test_inplane,
               "reference_w2_z5_calibration": ref.to_dict()}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    per = (revs["t_stop"] - revs["t_start"]).dt.total_seconds().median() / 3600
    lines = [
        "# Step 2 — per-revolution series and baseline detector",
        "",
        "Generated by `scripts/run_baseline.py`. All values are computed from the data.",
        "",
        "## Series",
        "",
        f"- {len(revs):,} complete revolutions of {sat}, {revs.index[0]:%Y-%m-%d} to "
        f"{revs.index[-1]:%Y-%m-%d} (orbits {revs['orbit'].min()} to {revs['orbit'].max()}; "
        f"{revs['orbit'].max() - revs['orbit'].min() + 1 - len(revs):,} missing or incomplete).",
        "- Gaps between consecutive ESA burns (hours: count): "
        + ", ".join(f"{k}: {v}" for k, v in gap_hist.items())
        + f". Burns less than {MAX_GAP_H:g} h apart form one manoeuvre.",
        f"- ESA record: {len(burns):,} burns grouped into {len(mans):,} manoeuvres "
        f"({', '.join(f'{k}: {v}' for k, v in mans['type'].value_counts().items())}).",
        "",
        "Revolution-to-revolution noise of the mean semi-major axis (robust σ of consecutive "
        "differences, manoeuvre revolutions excluded) and mean F10.7 per year:",
        "",
        noise_tab.to_markdown(floatfmt=".2f"),
        "",
        "## Calibration (" + " to ".join(CALIBRATION) + ", exclusive end)",
        "",
        f"- Grid: window W in {list(WINDOWS)} revolutions, threshold in {list(THRESHOLDS)}. "
        f"Selected by F1: **W = {cfg.window}, threshold = {cfg.threshold:g}**.",
        f"- Reference (W = 2, z = 5, closest to single-revolution jumps): recall "
        f"{fmt(ref['recall'])}, precision {fmt(ref['precision'])}, F1 {fmt(ref['f1'])}.",
        f"- Selected configuration on calibration: recall {fmt(s_cal['recall'])}, precision "
        f"{fmt(s_cal['precision'])}, F1 {fmt(s_cal['f1'])}.",
        "",
        "## Test (" + " to ".join((TEST[0], f"{revs.index[-1]:%Y-%m-%d}")) + ")",
        "",
        f"- Manoeuvres: {s_test['manoeuvres']} ({s_test['observable']} observable). "
        f"TP {s_test['tp']}, FN {s_test['fn']}, false alarms {s_test['fp']} "
        f"(duplicates inside a manoeuvre window: {s_test['duplicates']}).",
        f"- **Recall {fmt(s_test['recall'])}, precision {fmt(s_test['precision'])}, "
        f"F1 {fmt(s_test['f1'])}.**",
        f"- In-plane manoeuvres only (inclination manoeuvres excluded: they leave the "
        f"semi-major axis unchanged by design): recall {fmt(s_test_inplane['recall'])}, "
        f"precision {fmt(s_test_inplane['precision'])}, F1 {fmt(s_test_inplane['f1'])}.",
        f"- Δv estimate (Δv = Δa v / 2a) on detected station-keeping manoeuvres: median "
        f"absolute error {fmt(s_test['dv_abs_err_median_mm_s'], 2)} mm/s, median relative "
        f"error {fmt(s_test['dv_rel_err_median'] * 100, 0)} %.",
        f"- Detection delay (causal reading, alarm available once W revolutions follow the "
        f"manoeuvre): median {fmt(s_test['delay_median_h'], 1)} h "
        f"(one revolution = {per * 60:.1f} min).",
        "",
        "### Recall by manoeuvre type (test)",
        "",
        by_type.to_markdown(floatfmt=".3f"),
        "",
        "Manoeuvres typed `inclination` are those whose cross-track Δv dominates; the ones "
        "detected are those that also carry a sizeable along-track component.",
        "",
        "### Recall by expected |Δa| (test)",
        "",
        by_bin.to_markdown(floatfmt=".3f"),
        "",
        "## By solar-cycle phase (whole mission, frozen configuration)",
        "",
        by_phase[["observable", "tp", "fn", "fp", "recall", "precision", "f1"]]
        .to_markdown(floatfmt=".3f"),
        "",
        "Phases are fixed date ranges (see `satwiser.evaluation.SOLAR_PHASES`). The first "
        "phase overlaps the calibration period; only the test section above is out of sample.",
        "",
        "## Label cross-check with the POD manoeuvre flag",
        "",
        f"- Observable ESA manoeuvres with a flagged revolution within ±1 revolution: "
        f"{man_flagged.sum()} / {len(obs_all)}.",
        f"- Flagged revolutions not explained by an ESA manoeuvre: "
        f"{int((~covered).sum())} / {len(flagged)}.",
        f"- False alarms (whole mission) within ±1 revolution of such an unexplained POD "
        f"flag: {fa_on_flag} / {len(fa)}. These may be manoeuvres missing from the ESA "
        "record rather than detector errors; they are still counted as false alarms.",
        "",
        "## Figures",
        "",
        "- `mission_2016.png`, `mission_2023.png`: mean semi-major axis with detections "
        "(filled: detected, hollow: missed, diamond: false alarm), F10.7 below.",
        "- `recall_by_da.png`: test recall against the expected |Δa| of each manoeuvre.",
        "",
    ]
    (OUT / "baseline.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
