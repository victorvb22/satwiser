"""Step 1 data reconnaissance.

Answers the open questions of the data survey from the downloaded files and writes
figures plus a Markdown summary to ``reports/step1``. Every number in the summary is
computed here.

Prerequisites::

    python scripts/collect.py esa
    python scripts/collect.py spaceweather
    python scripts/collect.py poeorb --satellite S1A --start 2023-06-01 --end 2023-07-01
    python scripts/collect.py poeorb --satellite S1A --start 2016-08-09 --end 2016-09-07
    python scripts/collect.py poeorb --satellite S1B --start 2021-12-09 --end 2022-01-07
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from satwiser.collect import spaceweather  # noqa: E402
from satwiser.config import REPO_ROOT, raw_dir  # noqa: E402
from satwiser.io import eof, esa_history  # noqa: E402
from satwiser.orbit.elements import MU_EARTH, elements_from_eof  # noqa: E402
from satwiser.orbit.revolutions import revolution_means  # noqa: E402

OUT = REPO_ROOT / "reports" / "step1"
DATA = "#9CC2FF"
EVENT = "#F4A259"
FAINT = "#5E6782"


@dataclass
class Window:
    satellite: str
    start: str
    end: str


def load_window(w: Window) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    folder = raw_dir("poeorb", w.satellite)
    start, end = pd.Timestamp(w.start), pd.Timestamp(w.end)
    files = sorted(folder.glob(f"{w.satellite}_OPER_AUX_POEORB_OPOD_*.EOF"),
                   key=lambda p: p.name.split("_V")[1])
    frames = []
    for path in files:
        vstart = pd.Timestamp(path.name.split("_V")[1][:15])
        if start - pd.Timedelta(days=1) <= vstart < end:
            frames.append(eof.read_eof(path))
    merged = eof.merge(frames)
    merged = merged[(merged.index >= start) & (merged.index < end)]
    return merged, elements_from_eof(merged), frames


def sampling_report(frames: list[pd.DataFrame]) -> dict[str, object]:
    steps = np.concatenate([np.diff(f.index.asi8) * 1e-9 for f in frames])
    overlaps, pos_diff = [], []
    for prev, nxt in zip(frames[:-1], frames[1:], strict=True):
        common = prev.index.intersection(nxt.index)
        overlaps.append(len(common))
        if len(common):
            d = prev.loc[common, ["x", "y", "z"]].to_numpy() - nxt.loc[common, ["x", "y", "z"]]
            pos_diff.append(np.linalg.norm(d.to_numpy(), axis=1).max())
    values, counts = np.unique(np.round(steps, 3), return_counts=True)
    return {
        "files": len(frames),
        "osv_per_file_median": int(np.median([len(f) for f in frames])),
        "step_values_s": dict(zip(values.tolist(), counts.tolist(), strict=True)),
        "overlap_epochs_median": int(np.median(overlaps)),
        "overlap_hours": float(np.median(overlaps)) * 10 / 3600,
        "overlap_pos_diff_max_m": float(np.max(pos_diff)) if pos_diff else float("nan"),
        "overlap_pos_diff_median_m": float(np.median(pos_diff)) if pos_diff else float("nan"),
        "non_nominal_quality": int(sum((f["quality"] != "NOMINAL").sum() for f in frames)),
    }


def cluster_burns(burns: pd.DataFrame, max_gap_h: float = 2.0) -> np.ndarray:
    """Label burns separated by less than ``max_gap_h`` hours as one manoeuvre."""
    gaps = burns["start"].diff() > pd.Timedelta(hours=max_gap_h)
    return np.cumsum(gaps.to_numpy())


def burn_check(elements: pd.DataFrame, means: pd.DataFrame, burns: pd.DataFrame,
               window: int = 8) -> pd.DataFrame:
    """Compare observed mean-element jumps with those implied by the ESA burn records.

    Assumes accelerations in km/s^2 with components (radial, along-track, cross-track).
    Burns less than two hours apart form one manoeuvre. Straight lines are fitted to the
    ``window`` revolutions before the first burn and after the last one, and the jump is
    their difference at the manoeuvre mid-time. A manoeuvre is ``isolated`` when no other
    manoeuvre falls inside those fitting windows.
    """
    t_rev = means.index.as_unit("ns").asi8 * 1e-9
    a0 = means["a"].median()
    v0 = np.sqrt(MU_EARTH / a0)
    t_el = elements.index.as_unit("ns").asi8
    u_el = np.unwrap(elements["u"].to_numpy())
    burns = burns.copy()
    burns["dv_t"] = burns["a2"] * 1e3 * burns["duration_s"]
    burns["dv_n"] = burns["a3"] * 1e3 * burns["duration_s"]
    u = np.interp(burns["start"].to_numpy().astype("datetime64[ns]").astype(np.int64), t_el, u_el)
    # Gauss equations, near-circular orbit: da = 2 a dv_T / v, di = dv_N cos(u) / v.
    burns["di_pred"] = burns["dv_n"] * np.cos(u) / v0
    burns["cluster"] = cluster_burns(burns)

    spans = []
    for _, g in burns.groupby("cluster"):
        first, last = g["start"].min(), g["stop"].max()
        k0 = int(np.searchsorted(means["t_stop"].to_numpy(), first.to_datetime64()))
        k1 = int(np.searchsorted(means["t_start"].to_numpy(), last.to_datetime64())) - 1
        spans.append((g, first, last, k0, k1))
    rows = []
    for j, (g, first, last, k0, k1) in enumerate(spans):
        before = np.arange(k0 - window, k0)
        after = np.arange(k1 + 1, k1 + window + 1)
        if before[0] < 0 or after[-1] >= len(means):
            continue
        others = [(s[3], s[4]) for m, s in enumerate(spans) if m != j]
        isolated = all(o1 < before[0] or o0 > after[-1] for o0, o1 in others)
        t_mid = (first.value + (last.value - first.value) / 2) * 1e-9
        rec = {
            "start": first, "stop": last, "n_burns": len(g),
            "dv_t": g["dv_t"].sum(), "dv_t_abs": g["dv_t"].abs().sum(),
            "da_pred_m": 2 * a0 * g["dv_t"].sum() / v0,
            "di_pred_deg": float(np.rad2deg(g["di_pred"].sum())),
            "isolated": isolated,
        }
        for col, scale, name in (("a", 1.0, "da_obs_m"), ("i", np.rad2deg(1.0), "di_obs_deg")):
            y = means[col].to_numpy() * scale
            pb = np.polyfit(t_rev[before] - t_mid, y[before], 1)
            pa = np.polyfit(t_rev[after] - t_mid, y[after], 1)
            rec[name] = pa[1] - pb[1]
            if col == "a":
                resid = np.concatenate((y[before] - np.polyval(pb, t_rev[before] - t_mid),
                                        y[after] - np.polyval(pa, t_rev[after] - t_mid)))
                rec["fit_rms_m"] = float(np.sqrt(np.mean(resid**2)))
        rows.append(rec)
    result = pd.DataFrame(rows)
    da = means["a"].diff()
    touched = np.zeros(len(means), bool)
    for _, _, _, k0, k1 in spans:
        touched[max(k0, 0):k1 + 2] = True
    quiet = ~touched & da.notna().to_numpy()
    result["noise_da_m"] = 1.4826 * np.median(np.abs(da[quiet] - da[quiet].median()))
    return result


def detrended(means: pd.DataFrame, col: str, exclude: pd.DatetimeIndex, deg: int = 2
              ) -> pd.Series:
    t = (means.index - means.index[0]).total_seconds().to_numpy() / 86400
    keep = ~means.index.isin(exclude)
    coef = np.polyfit(t[keep], means[col].to_numpy()[keep], deg)
    return means[col] - np.polyval(coef, t)


def style(ax: plt.Axes) -> None:
    ax.set_facecolor("#06080E")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#1E2436")
    ax.tick_params(colors=FAINT, labelsize=8)
    ax.yaxis.label.set_color("#8C95AB")
    ax.xaxis.label.set_color("#8C95AB")
    ax.title.set_color("#E8ECF4")
    ax.grid(color="#161B2A", lw=0.6)


def figure(rows: int, height: float = 3.0) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(rows, 1, figsize=(11, height * rows), squeeze=False)
    fig.patch.set_facecolor("#06080E")
    for ax in axes[:, 0]:
        style(ax)
    return fig, axes[:, 0]


def pod_manoeuvre_flags(raw: pd.DataFrame) -> list[pd.Timestamp]:
    """Start of each run of state vectors flagged ``DEGRADED-MANOEUVRE`` by the POD.

    Used only to annotate the reconnaissance figures (and later as a cross-check of the
    labels); the detector must never see this flag.
    """
    flagged = raw.index[raw["quality"] == "DEGRADED-MANOEUVRE"]
    if flagged.empty:
        return []
    starts = flagged[np.concatenate(([True], np.diff(flagged.as_unit("ns").asi8) > 11e9))]
    return list(starts)


def event_window(label: str, w: Window, event: pd.Timestamp, burns: pd.DataFrame | None
                 ) -> dict[str, object]:
    raw, el, _ = load_window(w)
    means = revolution_means(el)
    flagged = pod_manoeuvre_flags(raw)
    if burns is not None:
        b = burns[(burns["start"] >= means.index[0]) & (burns["start"] <= means.index[-1])]
        burn_times = list(b["start"])
    else:
        burn_times = []
    # Exclude revolutions touched by known manoeuvres from trend fits and noise estimates.
    # Without an ESA history (Sentinel-1B), the POD flags are the only record available.
    known = burn_times if burns is not None else flagged
    touched = means.index[np.zeros(len(means), bool)]
    for t in known:
        near = np.abs((means.index - t).total_seconds()) < 2 * 6000
        touched = touched.union(means.index[near])
    res_a = detrended(means, "a", touched)
    res_e = detrended(means, "e", touched)
    res_i = detrended(means, "i", touched)
    da = means["a"].diff()
    quiet = ~means.index.isin(touched) & da.notna().to_numpy()
    noise = 1.4826 * np.median(np.abs(da[quiet] - da[quiet].median()))
    k = int(np.searchsorted(means.index, event))
    jump = float((means["a"].iloc[k + 1] - means["a"].iloc[k - 1]) - 2 * da[quiet].median())
    z = (da[quiet] - da[quiet].median()) / noise
    zev = (da - da[quiet].median()) / noise
    win = np.abs((means.index - event).total_seconds()) < 6 * 3600

    fig, axes = figure(3, 2.4)
    for ax, series, unit, scale in ((axes[0], res_a, "m", 1.0), (axes[1], res_e, "1e-6", 1e6),
                                    (axes[2], np.rad2deg(res_i), "mdeg", 1e3)):
        ax.plot(means.index, series * scale, color=DATA, lw=0.9, marker=".", ms=2)
        ax.axvline(event, color=EVENT, lw=1.2, ls="--")
        for t in burn_times:
            ax.axvline(t, color=FAINT, lw=0.6)
        for t in flagged:
            ax.axvline(t, color=FAINT, lw=0.8, ls=":")
        ax.set_ylabel(unit)
    axes[0].set_title(f"{label} — detrended per-revolution mean a, e, i (orange: event, "
                      "grey: ESA burns, dotted: POD manoeuvre flag)", fontsize=10, loc="left")
    fig.tight_layout()
    name = f"event_{w.satellite.lower()}_{event:%Y%m%d}.png"
    fig.savefig(OUT / name, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "label": label, "figure": name, "revs": len(means), "burns_in_window": len(burn_times),
        "pod_flags": [t.strftime("%Y-%m-%d %H:%M") for t in flagged],
        "noise_da_m": float(noise), "jump_at_event_m": jump,
        "max_abs_z_near_event": float(np.abs(zev[win]).max()),
        "max_abs_z_quiet_window": float(np.abs(z).max()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    esa_dir = raw_dir("esa")
    burns_a = esa_history.read_manoeuvres(esa_dir / "s1a.man")
    burns_c = esa_history.read_manoeuvres(esa_dir / "s1c.man")
    outages_a = esa_history.read_outages(esa_dir / "s1a.out")
    mass_a = esa_history.read_mass_history(esa_dir / "s1a.mhf")
    sw = spaceweather.parse(raw_dir("spaceweather") / spaceweather.FILENAME)

    # 1. Sampling and overlaps -----------------------------------------------------
    month = Window("S1A", "2023-06-01", "2023-07-01")
    raw, el, frames = load_window(month)
    samp = sampling_report(frames)
    gaps = eof.find_gaps(raw.index)
    means = revolution_means(el)
    revs_per_day = len(means) / 30

    # 2. Burn records vs orbit ---------------------------------------------------------
    b_month = burns_a[(burns_a["start"] >= month.start) & (burns_a["start"] < month.end)]
    check = burn_check(el, means, b_month)
    check.to_csv(OUT / "burn_check_2023-06.csv", index=False)

    fig, axes = figure(3, 2.6)
    one_day = el[el.index < pd.Timestamp("2023-06-02")]
    axes[0].plot(one_day.index, (one_day["a"] - 7.07e6) / 1e3, color=DATA, lw=0.8)
    axes[0].set_ylabel("a - 7070 km [km]")
    axes[0].set_title("Osculating semi-major axis over one day (J2 short-period terms)",
                      fontsize=10, loc="left")
    axes[1].plot(means.index, means["a"] - means["a"].iloc[0], color=DATA, lw=0.9,
                 marker=".", ms=2)
    for t in b_month["start"]:
        axes[1].axvline(t, color=EVENT, lw=0.6, alpha=0.8)
    axes[1].set_ylabel("mean a [m]")
    axes[1].set_title("Per-revolution mean semi-major axis, June 2023 (orange: ESA burns)",
                      fontsize=10, loc="left")
    ok = check[check["isolated"]]
    axes[2].scatter(ok["da_pred_m"], ok["da_obs_m"], color=EVENT, s=14)
    lim = np.nanmax(np.abs(ok[["da_pred_m", "da_obs_m"]].to_numpy())) * 1.1
    axes[2].plot([-lim, lim], [-lim, lim], color=FAINT, lw=0.8)
    axes[2].set_xlabel("Δa predicted from ESA record, km/s² along-track [m]")
    axes[2].set_ylabel("Δa observed [m]")
    fig.tight_layout()
    fig.savefig(OUT / "s1a_2023-06.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 3. Volume estimate -------------------------------------------------------------
    t0 = burns_a["start"].min().normalize()
    t1 = pd.Timestamp(mass_a.index.max())
    idx_a = pd.read_csv(raw_dir("poeorb") / "index_S1A.csv", parse_dates=["validity_start"])
    first_orbit_day = idx_a["validity_start"].min()
    last_orbit_day = idx_a["validity_start"].max()
    mission_days = (last_orbit_day - first_orbit_day).days
    n_rev_a = mission_days * revs_per_day
    # One row per revolution: timestamp, bounds, rev number, a, e, i, residual, score,
    # flags, F10.7, Ap -> ~14 numeric columns; Postgres adds ~24 B tuple header + index.
    bytes_per_row = 14 * 8 + 24 + 40
    n_rev_c = (pd.Timestamp("2026-09-30") - pd.Timestamp("2024-12-07")).days * revs_per_day
    volume_mb = (n_rev_a + n_rev_c) * bytes_per_row / 1e6

    # 4. Events ---------------------------------------------------------------------
    events = [
        event_window("Sentinel-1A particle impact, 2016-08-23",
                     Window("S1A", "2016-08-10", "2016-09-06"),
                     pd.Timestamp("2016-08-23 17:07"), burns_a),
        event_window("Sentinel-1B power anomaly, 2021-12-23",
                     Window("S1B", "2021-12-10", "2022-01-06"),
                     pd.Timestamp("2021-12-23 18:00"), None),
    ]

    # 5. Space weather coverage -----------------------------------------------------
    sw_mission = sw.loc["2014-04-01":]
    sw_missing = int(sw_mission[["f107_obs", "ap"]].isna().any(axis=1).sum())

    write_summary(samp, gaps, means, revs_per_day, check, burns_a, burns_c, outages_a, mass_a,
                  first_orbit_day, last_orbit_day, len(idx_a), n_rev_a, n_rev_c,
                  bytes_per_row, volume_mb, events, sw_mission, sw_missing, t0, t1)
    print((OUT / "recon.md").read_text(encoding="utf-8"))


def write_summary(samp, gaps, means, revs_per_day, check, burns_a, burns_c, outages_a, mass_a,
                  first_day, last_day, n_files, n_rev_a, n_rev_c, bytes_per_row, volume_mb,
                  events, sw_mission, sw_missing, t0, t1) -> None:
    along = check[check["isolated"]]
    big = along[along["da_pred_m"].abs() > 3 * along["noise_da_m"]]
    pairs = ", ".join(f"{o:.1f}/{p:.1f}" for o, p in zip(big["da_obs_m"], big["da_pred_m"],
                                                          strict=True))
    corr = np.corrcoef(along["da_pred_m"], along["da_obs_m"])[0, 1] if len(along) > 2 else np.nan
    normal = check[check["di_pred_deg"].abs() > 1e-4]
    dv_all = np.abs(burns_a["a2"]) * 1e3 * burns_a["duration_s"]
    lines = [
        "# Step 1 — data reconnaissance",
        "",
        "Generated by `scripts/recon_step1.py`. All values below are computed from the files.",
        "",
        "## Precise orbits (AUX_POEORB, Sentinel-1A, June 2023)",
        "",
        f"- Files: {samp['files']}, median {samp['osv_per_file_median']} state vectors per file.",
        f"- Sampling steps found (s: count): {samp['step_values_s']}.",
        f"- Overlap between consecutive daily files: {samp['overlap_epochs_median']} epochs "
        f"({samp['overlap_hours']:.2f} h). Max position difference on shared epochs: "
        f"{samp['overlap_pos_diff_max_m']:.3f} m (median of per-file max "
        f"{samp['overlap_pos_diff_median_m']:.3f} m).",
        f"- Non-NOMINAL quality flags: {samp['non_nominal_quality']} epochs, all "
        "`DEGRADED-MANOEUVRE`, in ~10 min runs around each ESA burn. This flag is a direct "
        "label leak: the detector must not use it.",
        f"- Gaps > 15 s after merging: {len(gaps)}.",
        "- Reference frame: EARTH_FIXED (rotated to a quasi-inertial frame before computing "
        "elements).",
        f"- Complete revolutions: {len(means)} ({revs_per_day:.2f} per day).",
        f"- Bucket index: {n_files} S1A files, validity starts {first_day:%Y-%m-%d} to "
        f"{last_day:%Y-%m-%d}.",
        "",
        "## ESA manoeuvre history",
        "",
        f"- `s1a.man`: {len(burns_a)} burns, {burns_a['start'].min():%Y-%m-%d} to "
        f"{burns_a['start'].max():%Y-%m-%d}. `s1c.man`: {len(burns_c)} burns, "
        f"{burns_c['start'].min():%Y-%m-%d} to {burns_c['start'].max():%Y-%m-%d}.",
        "- Content: burn start and end time (UTC) and three constant acceleration components. "
        "No unit, frame, manoeuvre type or Δv is given explicitly.",
        f"- `s1a.out`: {int((outages_a['kind'] == 'MAN').sum())} MAN windows and "
        f"{int((outages_a['kind'] == 'GAP').sum())} GNSS gaps. `s1a.mhf`: "
        f"{len(mass_a)} mass samples, {mass_a['mass_kg'].iloc[0]:.1f} kg to "
        f"{mass_a['mass_kg'].iloc[-1]:.1f} kg.",
        "- Sentinel-1B: no manoeuvre history file is published.",
        f"- June 2023: {int(check['n_burns'].sum())} burns grouped into {len(check)} "
        f"manoeuvres (burns < 2 h apart), {len(along)} isolated (no other manoeuvre within "
        "the 8-revolution fitting windows).",
        "- Unit check, assuming km/s² and component 2 = along-track: on isolated manoeuvres, "
        f"correlation between observed and predicted Δa {corr:.3f}; observed/predicted Δa (m) "
        f"for those above 3σ: {pairs}. Multi-burn +/- sequences have a near-zero net Δa, as "
        "predicted.",
        f"- Revolution-to-revolution noise of mean a (robust σ, quiet revolutions): "
        f"{check['noise_da_m'].iloc[0]:.2f} m, after removing the J2 short-period term.",
        f"- Out-of-plane manoeuvres in June 2023: {len(normal)}; Δi predicted "
        + ", ".join(f"{p * 1e3:.2f}" for p in normal["di_pred_deg"])
        + " mdeg, observed "
        + ", ".join(f"{o * 1e3:.2f}" for o in normal["di_obs_deg"])
        + " mdeg. Mean inclination carries a ~±2 mdeg daily oscillation (ground-track "
        "dependent), larger than a single inclination manoeuvre: Δi is not measurable with "
        "simple before/after fits.",
        f"- Along-track Δv per burn over the mission (km/s² assumption): median "
        f"{dv_all.median() * 100:.2f} cm/s, 90th percentile {dv_all.quantile(0.9) * 100:.2f} "
        "cm/s.",
        "",
        "## Volume estimate for the database",
        "",
        f"- S1A: {n_rev_a:,.0f} revolutions; S1C (since 2024-12-07): {n_rev_c:,.0f}.",
        f"- At ~{bytes_per_row} B per row (14 numeric columns + tuple header + index): "
        f"{volume_mb:.1f} MB, against the 500 MB free-tier limit.",
        "",
        "## Event windows",
        "",
    ]
    for ev in events:
        lines += [
            f"### {ev['label']}",
            "",
            f"- Figure: `{ev['figure']}`; {ev['revs']} revolutions, "
            f"{ev['burns_in_window']} ESA burns in the window. POD manoeuvre flags: "
            f"{', '.join(ev['pod_flags']) or 'none'}.",
            f"- Noise of rev-to-rev mean a changes: {ev['noise_da_m']:.2f} m. Jump across the "
            f"event (drift removed): {ev['jump_at_event_m']:.2f} m. Max |z| within ±6 h: "
            f"{ev['max_abs_z_near_event']:.1f} (max |z| over quiet revolutions: "
            f"{ev['max_abs_z_quiet_window']:.1f}).",
            "",
        ]
    lines += [
        "## Space weather (GFZ Potsdam)",
        "",
        f"- Daily F10.7 (observed and adjusted) and Ap from {sw_mission.index[0]:%Y-%m-%d} to "
        f"{sw_mission.index[-1]:%Y-%m-%d}; days with a missing value: {sw_missing}.",
        "- Licence: CC BY 4.0 for Kp/ap/Ap and F10.7 (sunspot number column, CC BY-NC 4.0, "
        "is dropped).",
        "",
    ]
    (OUT / "recon.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
