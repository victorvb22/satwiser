"""Step 3: main model (ground-track signature + drag + CUSUM) and event classification.

Everything that is fitted or tuned (ground-track template, drag model, detector
settings, classifier) uses the calibration period only; the test period is untouched
until the final evaluation, except where a post-hoc choice is disclosed in the report.
Every number written to ``reports/step3`` comes from here.

    python scripts/run_step3.py --satellite S1A
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from satwiser import classify
from satwiser.config import CALIBRATION_END, REPO_ROOT, processed_dir
from satwiser.detection import baseline
from satwiser.detection.cusum import Channel, CusumConfig, detect, innovations, robust_sigma
from satwiser.evaluation import da_bin, evaluate, recall_by, summary
from satwiser.model.drag import fit_drag, segment_slopes
from satwiser.model.grid import quiet_mask, segments, to_grid
from satwiser.model.groundtrack import REPEAT, fit_template, operational_start
from satwiser.plotting import DATA, EVENT, FAINT, MUTED, figure, legend, save
from satwiser.reporting import fmt

OUT = REPO_ROOT / "reports" / "step3"
SPLIT = CALIBRATION_END
MDEG = np.rad2deg(1.0) * 1e3
#: Observability window shared by every detector of the comparison, so that all of them
#: are scored on the same set of manoeuvres (the largest detector window, 12).
OBSERVE_W = 12

GRID = {
    "variant": ("full", "no_drag"),
    "memory": (20, 30),
    "kappa": (1.0, 2.0, 3.0),
    "h": (6.0, 8.0, 12.0, 20.0, 30.0),
    "scale_window": (0, 200),
}
STORM_AP = 30
FEATURE_SETS = {
    "all": classify.FEATURES,
    "geometry": ("da", "de", "di", "abs_da", "abs_di", "alarm_lag", "on_i_channel",
                 "days_since_prev"),
}
CLIP = 6.0
MIN_REF = 4


def noise(y: np.ndarray, mask: np.ndarray) -> float:
    """Robust sigma of differences between consecutive revolutions that are both in mask."""
    d = np.diff(y, prepend=np.nan)
    ok = mask & np.roll(mask, 1)
    return robust_sigma(d[ok])


# ------------------------------------------------------------------------------ model

class MainModel:
    """Fitted nuisance models and channels for the CUSUM detector."""

    def __init__(self, revs: pd.DataFrame, mans: pd.DataFrame):
        g = to_grid(revs)
        quiet = quiet_mask(g, mans)
        t = g["t"]
        # Pass 1 on all pre-split data only to locate the end of orbit acquisition.
        first = fit_template(g, quiet, (t < SPLIT).to_numpy())
        self.ops_start = operational_start(first.correct(g))
        self.train = ((t >= self.ops_start) & (t < SPLIT)).to_numpy()
        self.template = fit_template(g, quiet, self.train)
        g = self.template.correct(g)
        self.slopes = segment_slopes(g, "a_c", quiet & self.train)
        self.slopes_all = segment_slopes(g, "a_c", quiet)
        self.drag = fit_drag(self.slopes)
        g["drag_cum"] = self.drag.cumulative(g)
        g["y_a"] = g["a_c"] - g["drag_cum"]
        g["y_i"] = g["i_c"] * MDEG
        g["y_e"] = g["e_c"] * 1e6
        self.grid, self.quiet = g, quiet
        self.raw_grid = to_grid(revs)
        self._sigma: dict[tuple, float] = {}

    def channel(self, name: str, column: str, predictor: str, memory: int) -> Channel:
        y = self.grid[column].to_numpy()
        key = (column, predictor, memory)
        if key not in self._sigma:
            valid = self.grid["valid"].to_numpy()
            inn = innovations(Channel(name, y, 1.0, predictor), valid, memory, MIN_REF)
            self._sigma[key] = robust_sigma(inn[self.quiet & self.train])
        return Channel(name, y, self._sigma[key], predictor)

    def channels(self, memory: int, variant: str = "full") -> list[Channel]:
        a_col, a_pred = ("a_c", "linear") if variant == "no_drag" else ("y_a", "mean")
        chans = [self.channel("a", a_col, a_pred, memory)]
        if variant != "a_only":
            chans.append(self.channel("i", "y_i", "linear", memory))
        return chans

    def estimate(self) -> dict[str, np.ndarray]:
        return {"a": self.grid["y_a"].to_numpy(), "i": self.grid["y_i"].to_numpy(),
                "e": self.grid["y_e"].to_numpy()}

    def detect(self, cfg: CusumConfig, variant: str = "full") -> pd.DataFrame:
        return detect(self.grid, self.channels(cfg.memory, variant), cfg, self.estimate())


@dataclass
class Context:
    """Inputs shared by the analysis stages."""

    sat: str
    revs: pd.DataFrame
    mans: pd.DataFrame
    model: MainModel
    cal: tuple[str, str]
    test: tuple[str, str]
    coverage_end: pd.Timestamp
    results: dict = field(default_factory=dict)

    def evaluate(self, det, window, period=(None, None)):
        return evaluate(det, self.mans, self.revs, window, *period, observe_window=OBSERVE_W)


def cusum_config(row) -> CusumConfig:
    return CusumConfig(kappa=float(row["kappa"]), h=float(row["h"]), memory=int(row["memory"]),
                       min_ref=MIN_REF, clip=CLIP, scale_window=int(row["scale_window"]))


# ------------------------------------------------------------------------ analyses

def nuisance_diagnostics(ctx: Context) -> dict:
    """Noise before/after the template, repeat-cycle autocorrelation, drag-model fit."""
    m = ctx.model
    g, raw, quiet, train = m.grid, m.raw_grid, m.quiet, m.train
    test_mask = (g["t"] >= SPLIT).to_numpy()
    series = {
        "a raw [m]": raw["a"].to_numpy(), "a corrected [m]": g["a_c"].to_numpy(),
        "i raw [mdeg]": raw["i"].to_numpy() * MDEG, "i corrected [mdeg]": g["y_i"].to_numpy(),
        "e raw [1e-6]": raw["e"].to_numpy() * 1e6, "e corrected [1e-6]": g["y_e"].to_numpy(),
    }
    noise_tab = pd.DataFrame({
        "calibration": [noise(y, quiet & train) for y in series.values()],
        "test": [noise(y, quiet & test_mask) for y in series.values()],
    }, index=list(series))

    resid = np.full(len(g), np.nan)
    y_raw = raw["a"].to_numpy()
    for run in segments(quiet & (g["t"] >= m.ops_start).to_numpy(), 60):
        x = run - run.mean()
        resid[run] = y_raw[run] - np.polyval(np.polyfit(x, y_raw[run], 2), x)
    resid = pd.Series(resid)
    acf = {lag: resid.corr(resid.shift(lag)) for lag in (1, 29, REPEAT)}

    s = m.slopes_all.copy()
    s["pred"] = m.drag.rate(s["f81"], s["f"], s["ap"])
    s["cal"] = s["t"] < SPLIT
    rel_err = ((s["slope"] - s["pred"]) / s["slope"]).abs()
    # Alternative considered: exponential law in F81, fitted on the same segments.
    cal_seg = m.slopes[m.slopes["slope"] < 0]
    x_cal = np.column_stack([np.ones(len(cal_seg)), cal_seg["f81"],
                             np.log(cal_seg["f"] / cal_seg["f81"]), np.log1p(cal_seg["ap"])])
    w = np.sqrt(cal_seg["n"].to_numpy())[:, None]
    beta_exp, *_ = np.linalg.lstsq(x_cal * w, np.log(-cal_seg["slope"].to_numpy()) * w[:, 0],
                                   rcond=None)
    x_all = np.column_stack([np.ones(len(s)), s["f81"], np.log(s["f"] / s["f81"]),
                             np.log1p(s["ap"])])
    s["pred_exp"] = -np.exp(x_all @ beta_exp)
    peak = s[s["t"].dt.year == 2024]
    exp_ratio_2024 = float(peak["pred_exp"].mean() / peak["slope"].mean())
    pow_ratio_2024 = float(peak["pred"].mean() / peak["slope"].mean())
    drag_year = s.groupby(s["t"].dt.year)[["slope", "pred"]].mean().rename(
        columns={"slope": "observed", "pred": "model"})
    drag_year.index.name = "year"
    return {"noise_tab": noise_tab, "acf": acf, "rel_err_cal": rel_err[s["cal"]].median(),
            "rel_err_test": rel_err[~s["cal"]].median(), "drag_year": drag_year,
            "dropped_positive": int((m.slopes["slope"] >= 0).sum()),
            "exp_ratio_2024": exp_ratio_2024, "pow_ratio_2024": pow_ratio_2024}


def select_cusum(ctx: Context) -> dict:
    """Score every CUSUM setting; select on calibration F1 only."""
    rows, cache = [], {}
    for variant, memory, kappa, h, sw in product(*GRID.values()):
        key = (variant, memory)
        if key not in cache:
            cache[key] = ctx.model.channels(memory, variant)
        row = {"variant": variant, "memory": memory, "kappa": kappa, "h": h, "scale_window": sw}
        det = detect(ctx.model.grid, cache[key], cusum_config(row), ctx.model.estimate())
        for name, per in (("calibration", ctx.cal), ("test", ctx.test)):
            sm = summary(*ctx.evaluate(det, 6, per))
            row.update({f"{name} {k}": sm[k] for k in ("recall", "precision", "f1")})
        rows.append(row)
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["calibration f1", "calibration precision"], ascending=False).iloc[0]
    cfg, variant = cusum_config(best), str(best["variant"])
    same = ((grid["variant"] == variant) & (grid["memory"] == cfg.memory)
            & (grid["kappa"] == cfg.kappa) & (grid["scale_window"] == cfg.scale_window))
    return {"grid": grid, "cfg": cfg, "variant": variant,
            "sensitivity": grid[same].set_index("h")[["calibration f1", "test recall",
                                                       "test precision", "test f1"]],
            "best_test": grid.sort_values("test f1", ascending=False).iloc[0],
            "det": ctx.model.detect(cfg, variant)}


def calibrate_baseline(ctx: Context, revs_a: pd.DataFrame) -> baseline.BaselineConfig:
    best, best_f1 = None, -1.0
    for w, thr in product((2, 4, 6, 8, 12), (3.0, 4.0, 5.0, 6.0, 8.0)):
        cfg = baseline.BaselineConfig(window=w, threshold=thr)
        f1 = summary(*ctx.evaluate(baseline.detect(baseline.scores(revs_a, cfg), cfg), w,
                                   ctx.cal))["f1"]
        if f1 > best_f1:
            best, best_f1 = cfg, f1
    return best


def compare_detectors(ctx: Context, main: dict) -> dict:
    step2 = json.loads((REPO_ROOT / "reports" / "step2" / "metrics.json").read_text())
    cfg_b2 = baseline.BaselineConfig(**{k: step2["config"][k] for k in ("window", "threshold")})
    revs_c = ctx.revs.copy()
    revs_c["a"] = revs_c["a"] - ctx.model.template.signature(revs_c["orbit"].to_numpy(), "a")
    cfg_bc = calibrate_baseline(ctx, revs_c)
    main_name = ("Main: template + drag + CUSUM" if main["variant"] == "full"
                 else "Main: template + local line + CUSUM")
    detectors = {
        f"Baseline, raw series (step 2, W={cfg_b2.window}, z={cfg_b2.threshold:g})":
            (baseline.detect(baseline.scores(ctx.revs, cfg_b2), cfg_b2), cfg_b2.window),
        f"Baseline, template-corrected (W={cfg_bc.window}, z={cfg_bc.threshold:g})":
            (baseline.detect(baseline.scores(revs_c, cfg_bc), cfg_bc), cfg_bc.window),
        main_name: (main["det"], 6),
    }
    rows, tables = [], {}
    for name, (det, w) in detectors.items():
        row = {"detector": name}
        for pname, per in (("calibration", ctx.cal), ("test", ctx.test)):
            m, d = ctx.evaluate(det, w, per)
            sm = summary(m, d)
            row.update({f"{pname} {k}": sm[k] for k in ("recall", "precision", "f1")})
            if pname == "test":
                row.update({"test Δv err [mm/s]": sm["dv_abs_err_median_mm_s"],
                            "test delay [h]": sm["delay_median_h"],
                            "causal delay": name.startswith("Main")})
                tables[name] = m
        rows.append(row)
    return {"table": pd.DataFrame(rows).set_index("detector"), "tables": tables}


def ablations(ctx: Context, cfg: CusumConfig) -> pd.DataFrame:
    rows = []
    for variant, label in (("full", "drag model + trailing mean"),
                           ("no_drag", "no drag model (trailing line)"),
                           ("a_only", "semi-major axis channel only")):
        m, d = ctx.evaluate(ctx.model.detect(cfg, variant), 6, ctx.test)
        sm = summary(m, d)
        rows.append({"variant": label, "recall": sm["recall"], "precision": sm["precision"],
                     "f1": sm["f1"], "inclination recall": recall_by(m, "type")
                     .get("recall", {}).get("inclination", np.nan)})
    return pd.DataFrame(rows).set_index("variant")


def main_results(ctx: Context, det: pd.DataFrame) -> dict:
    m_test, d_test = ctx.evaluate(det, 6, ctx.test)
    m_cal, d_cal = ctx.evaluate(det, 6, ctx.cal)
    m_all, d_all = ctx.evaluate(det, 6, (ctx.cal[0], str(ctx.coverage_end)))
    sk = m_test[m_test["detected"] & (m_test["type"] == "station_keeping")]
    return {
        "m_test": m_test, "d_test": d_test, "m_cal": m_cal, "d_cal": d_cal,
        "summary": summary(m_test, d_test), "by_type": recall_by(m_test, "type"),
        "by_bin": recall_by(m_test.assign(bin=da_bin(m_test)), "bin"),
        "by_phase": pd.DataFrame([{"phase": p, **summary(gm, d_all[d_all["phase"] == p])}
                                  for p, gm in m_all.groupby("phase")]).set_index("phase"),
        "dv_rel": ((sk["dv_est"] - sk["dv_t"]) / sk["dv_t"]).abs().median(),
    }


def false_alarm_analysis(ctx: Context, d_test: pd.DataFrame) -> dict:
    """Storm association and POD-flag coincidence of the test false alarms."""
    revs = ctx.revs
    ap_day = revs["ap"].groupby(revs.index.normalize()).first()
    storm_day = ap_day.rolling(3, min_periods=1).max() >= STORM_AP

    def storm_rate(frame):
        days = pd.to_datetime(frame["time"]).dt.normalize()
        return storm_day.reindex(days).fillna(False).astype(bool)

    fa = d_test[d_test["status"] == "false_alarm"]
    fa_storm = storm_rate(fa)
    test_days = storm_day[(storm_day.index >= SPLIT) & (storm_day.index <= ctx.coverage_end)]
    flagged = revs.loc[revs["pod_flag"], "orbit"].to_numpy()
    fa_flag = np.array([bool((np.abs(flagged - o) <= 1).any()) for o in fa["orbit"]], bool)
    big = fa["da"].abs().to_numpy() > 3.0
    return {"n": len(fa), "storm": int(fa_storm.sum()),
            "storm_rate_fa": float(fa_storm.mean()) if len(fa) else np.nan,
            "storm_rate_tp": float(storm_rate(d_test[d_test["status"] == "detected"]).mean()),
            "storm_rate_days": float(test_days.mean()), "small": int((~big).sum()),
            "big": int(big.sum()), "big_with_flag": int((big & fa_flag).sum())}


def classification(ctx: Context, res: dict, cfg: CusumConfig) -> dict:
    g = ctx.model.grid
    d_cal = res["d_cal"].assign(label=classify.label_detections(res["d_cal"], res["m_cal"]))
    d_test = res["d_test"].assign(label=classify.label_detections(res["d_test"], res["m_test"]))
    x_cal, x_test = classify.features(d_cal, g), classify.features(d_test, g)
    di_thr = 5 * ctx.model.channel("i", "y_i", "linear", cfg.memory).sigma
    da_thr = 3 * ctx.model.channel("a", "y_a", "mean", cfg.memory).sigma
    rule_pred = classify.rule(x_test, di_thr, da_thr)
    esa_type = res["m_test"]["type"].reindex(d_test["manoeuvre"].to_numpy()).fillna("none")
    # Feature set chosen by time-ordered cross-validation inside the calibration period.
    order = np.argsort(pd.to_datetime(d_cal["time"]).to_numpy())
    cv_scores = {}
    for name, cols in FEATURE_SETS.items():
        scores = []
        for tr, va in TimeSeriesSplit(n_splits=3).split(order):
            tr_idx, va_idx = order[tr], order[va]
            if d_cal["label"].iloc[tr_idx].nunique() < 2:
                continue
            fold = classify.train_model(x_cal.iloc[tr_idx][list(cols)], d_cal["label"].iloc[tr_idx])
            scores.append(classify.report(d_cal["label"].iloc[va_idx],
                                          fold.predict(x_cal.iloc[va_idx][list(cols)]))["macro_f1"])
        cv_scores[name] = float(np.mean(scores))
    feature_set = max(cv_scores, key=cv_scores.get)
    cols = list(FEATURE_SETS[feature_set])
    clf = classify.train_model(x_cal[cols], d_cal["label"])
    return {
        "rule": classify.report(d_test["label"], rule_pred), "di_thr": di_thr, "da_thr": da_thr,
        "rule_by_type": pd.crosstab(pd.Series(esa_type.to_numpy(), name="ESA type"),
                                    pd.Series(rule_pred, name="rule prediction")),
        "ml": classify.report(d_test["label"], clf.predict(x_test[cols])), "clf": clf,
        "cols": cols, "cv_scores": cv_scores, "feature_set": feature_set,
        "counts": pd.DataFrame({"calibration": d_cal["label"].value_counts(),
                                "test": d_test["label"].value_counts()}).fillna(0).astype(int),
        "n_unexp_cal": int((d_cal["label"] == "unexplained").sum()),
        "n_unexp_test": int((d_test["label"] == "unexplained").sum()),
    }


# -------------------------------------------------------------------------- outputs

def export_outputs(ctx: Context, main: dict, cls: dict, diag: dict, comparison: dict,
                   res: dict) -> pd.DataFrame:
    """Events, series, model parameters and metrics consumed by later stages and the app."""
    sat, g, model = ctx.sat, ctx.model.grid, ctx.model
    m_ev, d_ev = ctx.evaluate(main["det"], 6)
    d_ev = d_ev.assign(label=classify.label_detections(d_ev, m_ev))
    proba = cls["clf"].predict_proba(classify.features(d_ev, g)[cls["cols"]])
    d_ev["predicted_class"] = cls["clf"].classes_[proba.argmax(1)]
    d_ev["confidence"] = proba.max(1)
    d_ev["esa_dv_t"] = m_ev["dv_t"].reindex(d_ev["manoeuvre"].to_numpy()).to_numpy()
    d_ev["esa_type"] = m_ev["type"].reindex(d_ev["manoeuvre"].to_numpy()).to_numpy()
    d_ev["split"] = np.where(d_ev["time"] < SPLIT, "calibration", "test")
    beyond = (d_ev["time"] > ctx.coverage_end) | (d_ev["time"] < model.ops_start)
    d_ev.loc[beyond, ["status", "label"]] = "unlabelled"
    d_ev.to_parquet(processed_dir() / f"events_{sat}.parquet")
    m_ev.to_parquet(processed_dir() / f"manoeuvres_eval_{sat}.parquet")
    g[["t", "t_start", "t_stop", "valid", "a", "a_c", "y_a", "y_i", "y_e", "drag_cum",
       "f107_obs", "f107_81d", "ap"]].to_parquet(processed_dir() / f"series_{sat}.parquet")
    cfg, variant = main["cfg"], main["variant"]
    params = {
        "satellite": sat, "operational_start": str(model.ops_start.date()), "split": SPLIT,
        "template": model.template.to_dict(), "drag": model.drag.to_dict(),
        "cusum": cfg.__dict__ | {"clip": CLIP, "variant": variant},
        "sigma": {c.name: c.sigma for c in model.channels(cfg.memory, variant)},
    }
    (processed_dir() / f"model_{sat}.json").write_text(json.dumps(params, default=float))
    noise_tab = diag["noise_tab"]
    metrics = {
        "cusum_config": cfg.__dict__ | {"variant": variant}, "test": res["summary"],
        "comparison": comparison["table"].reset_index().to_dict("records"),
        "observe_window": OBSERVE_W,
        "drag_2024_ratio": {"power_law": diag["pow_ratio_2024"],
                            "exponential": diag["exp_ratio_2024"]},
        "noise": {"a_raw_m": noise_tab.loc["a raw [m]", "calibration"],
                  "a_corrected_m": noise_tab.loc["a corrected [m]", "calibration"],
                  "i_raw_mdeg": noise_tab.loc["i raw [mdeg]", "calibration"],
                  "i_corrected_mdeg": noise_tab.loc["i corrected [mdeg]", "calibration"],
                  "scope": "quiet revolutions of the calibration period"},
        "classification_rule_macro_f1": cls["rule"]["macro_f1"],
        "classification_model_macro_f1": cls["ml"]["macro_f1"],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return d_ev


def fig_template(model: MainModel, path) -> None:
    fig, axes = figure(1, 2, width=11, height=3.0)
    for ax, col, scale, unit in ((axes[0, 0], "a", 1.0, "m"), (axes[0, 1], "i", MDEG, "mdeg")):
        ax.plot(np.arange(REPEAT), model.template.values[col] * scale, color=DATA, lw=1)
        ax.set_xlabel("position in the 175-revolution repeat cycle")
        ax.set_ylabel(f"{col} signature [{unit}]")
    axes[0, 0].set_title("Ground-track repeat signature (fitted on calibration)", fontsize=10,
                         loc="left")
    save(fig, path)


def fig_example(model: MainModel, det: pd.DataFrame, mans: pd.DataFrame, start, stop, path):
    g = model.grid[(model.grid["t"] >= start) & (model.grid["t"] < stop) & model.grid["valid"]]
    raw = model.raw_grid.loc[g.index]
    fig, axes = figure(3, 1, width=11, height=2.4, sharex=True)
    panels = ((raw["a"], FAINT, "mean a [m]", "Raw per-revolution mean a (step 2 input)"),
              (g["y_a"], DATA, "corrected a [m]", "Minus ground-track signature and modelled drag"),
              (g["y_i"], DATA, "corrected i [mdeg]",
               "Corrected mean i (orange: CUSUM detections, dotted: ESA manoeuvres)"))
    m = mans[(mans["start"] >= start) & (mans["start"] < stop)]
    dd = det[(det["time"] >= start) & (det["time"] < stop)]
    for ax, (y, colour, label, title) in zip(axes[:, 0], panels, strict=True):
        ax.plot(g["t"], y - y.mean(), color=colour, lw=0.8)
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=10, loc="left")
        for t in m["start"]:
            ax.axvline(t, color=FAINT, lw=0.6, ls=":")
        for t in dd["time"]:
            ax.axvline(t, color=EVENT, lw=0.9, alpha=0.8)
    save(fig, path)


def fig_drag(model: MainModel, path) -> None:
    s = model.slopes_all.copy()
    s["pred"] = model.drag.rate(s["f81"], s["f"], s["ap"])
    fig, axes = figure(1, 1, width=11, height=3.0)
    ax = axes[0, 0]
    ax.scatter(s["t"], -s["slope"], s=6, color=DATA, label="observed (quiet segments)")
    ax.plot(s["t"], -s["pred"], color=EVENT, lw=1, label="power-law model")
    ax.axvline(pd.Timestamp(SPLIT), color=FAINT, ls="--", lw=0.8)
    ax.set_ylabel("decay [m / revolution]")
    ax.set_title("Drag decay of the mean semi-major axis (model fitted left of the dashed "
                 "line)", fontsize=10, loc="left")
    legend(ax, loc="upper left")
    save(fig, path)


def fig_recall(tabs: dict[str, pd.DataFrame], path) -> None:
    fig, axes = figure(1, 1, width=8, height=3.2)
    ax = axes[0, 0]
    colours = (FAINT, MUTED, EVENT)
    width = 0.8 / len(tabs)
    for j, (name, tab) in enumerate(tabs.items()):
        x = np.arange(len(tab)) + (j - (len(tabs) - 1) / 2) * width
        ax.bar(x, tab["recall"], width=width, color=colours[j % 3], label=name)
    first = next(iter(tabs.values()))
    ax.set_xticks(range(len(first)),
                  [f"{b}\n(n={n})" for b, n in zip(first.index, first["n"], strict=True)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall")
    ax.set_title("Test recall by expected |Δa|", fontsize=10, loc="left")
    legend(ax, loc="lower right")
    save(fig, path)


def render_report(ctx: Context, diag: dict, main: dict, comparison: dict, abl: pd.DataFrame,
                  res: dict, fa: dict, cls: dict) -> str:
    model, cfg, variant = ctx.model, main["cfg"], main["variant"]
    s_test = res["summary"]
    n_comm = int(((ctx.mans["start"] < model.ops_start)
                  & (ctx.mans["start"] >= ctx.revs.index[0])).sum())
    best_test = main["best_test"]
    lines = [
        "# Step 3 — main model and event classification",
        "",
        "Generated by `scripts/run_step3.py`. Everything fitted or tuned uses the "
        f"calibration period only ({ctx.cal[0]} to {SPLIT}); the test period is {SPLIT} to "
        f"{ctx.coverage_end:%Y-%m-%d}, the end of the ESA manoeuvre record (orbits run to "
        f"{ctx.revs.index[-1]:%Y-%m-%d}; later detections cannot be scored).",
        "",
        "## 1. Ground-track repeat signature",
        "",
        f"- Autocorrelation of the raw mean-a residuals at a lag of {REPEAT} revolutions "
        f"(the 12-day repeat cycle): **{diag['acf'][REPEAT]:.3f}** (lag 1: {diag['acf'][1]:.3f}, "
        f"lag 29: {diag['acf'][29]:.3f}), computed on quadratically detrended quiet segments. "
        "Most of the step 2 \"noise\" is a deterministic function of the position on the "
        "ground track.",
        f"- Template amplitude (peak to peak): a {np.ptp(model.template.values['a']):.1f} m, "
        f"i {np.ptp(model.template.values['i']) * MDEG:.2f} mdeg, "
        f"e {np.ptp(model.template.values['e']) * 1e6:.1f} ×1e-6.",
        "- Revolution-to-revolution noise (robust σ of differences between consecutive quiet "
        "revolutions), before and after removing the template fitted on calibration:",
        "",
        diag["noise_tab"].to_markdown(floatfmt=".3f"),
        "",
        f"- Orbit acquisition: the corrected noise reaches its nominal level from "
        f"**{model.ops_start:%Y-%m}**, detected from the data. The noise level used as "
        "reference is the median monthly noise over the whole mission (an unsupervised "
        f"statistic, no labels). The {n_comm} ESA manoeuvres before that date (orbit "
        "acquisition, satellite not yet on its reference ground track) are excluded from the "
        "evaluation of every detector below.",
        "",
        "## 2. Drag model",
        "",
        f"- Power law fitted on {len(model.slopes)} quiet calibration segments "
        f"({diag['dropped_positive']} with a non-negative slope are left out of the log fit): "
        f"`-da/drev = exp({model.drag.beta[0]:.2f}) · F81^{model.drag.beta[1]:.2f} · "
        f"(F/F81)^{model.drag.beta[2]:.2f} · (1+Ap)^{model.drag.beta[3]:.2f}` (m/rev), where "
        "F81 is the trailing 81-day mean of F10.7.",
        f"- Median relative error of the segment decay rate: calibration "
        f"{diag['rel_err_cal'] * 100:.0f} %, test {diag['rel_err_test'] * 100:.0f} % (test "
        "includes the 2024 solar maximum, outside the calibration range of F10.7).",
        f"- Alternative: an exponential law in F81 fitted on the same segments predicts "
        f"{diag['exp_ratio_2024']:.2f} times the observed mean 2024 decay, against "
        f"{diag['pow_ratio_2024']:.2f} for the power law, which is why the power law is "
        "kept (2024 lies beyond the calibration range of F10.7). Mean decay per year:",
        "",
        diag["drag_year"].to_markdown(floatfmt=".3f"),
        "",
        "## 3. Detector comparison",
        "",
        "- Main detector: two-sided CUSUM on the template-corrected semi-major axis and "
        f"inclination, innovations clipped at {CLIP:g}σ. Two variants for the semi-major "
        "axis: drag model + trailing mean, or trailing line without drag model. Selected by "
        f"calibration F1 over {len(main['grid'])} settings: **{variant}**, memory "
        f"{cfg.memory}, κ = {cfg.kappa:g}, h = {cfg.h:g}, scale window "
        f"{cfg.scale_window or 'fixed'}.",
        "- The windowed baseline is recalibrated on the template-corrected series for a fair "
        "comparison. All detectors share the evaluation protocol of step 2 (one-revolution "
        f"tolerance, ESA manoeuvres as truth) and the same observability window "
        f"({OBSERVE_W} revolutions on each side), so they are scored on the same manoeuvres.",
        "- Delays are orbit time from the manoeuvre start to the alarm, without the "
        "publication latency of the precise orbits (about three weeks). The CUSUM decides "
        "from past revolutions only (its drag correction uses same-day F10.7 and Ap and "
        "the trailing 81-day F10.7 mean, a look-ahead of at most one day on these "
        "covariates); the baselines normalise with a centred window of about 15 days, so their "
        "delays are not causal and only indicative.",
        "",
        comparison["table"].to_markdown(floatfmt=".3f"),
        "",
        "### Main detector on the test period",
        "",
        f"- Manoeuvres: {s_test['observable']}. TP {s_test['tp']}, FN {s_test['fn']}, false "
        f"alarms {s_test['fp']}, duplicates inside a manoeuvre window {s_test['duplicates']}.",
        f"- **Recall {fmt(s_test['recall'])}, precision {fmt(s_test['precision'])}, F1 "
        f"{fmt(s_test['f1'])}.** Median detection delay {fmt(s_test['delay_median_h'], 1)} h "
        "(orbit time, see above).",
        f"- Δv of detected station-keeping manoeuvres: median absolute error "
        f"{fmt(s_test['dv_abs_err_median_mm_s'], 2)} mm/s, median relative error "
        f"{res['dv_rel'] * 100:.0f} %.",
        "",
        res["by_type"].to_markdown(floatfmt=".3f"),
        "",
        res["by_bin"].to_markdown(floatfmt=".3f"),
        "",
        "### Sensitivity to the alarm threshold (post hoc)",
        "",
        "Test scores are shown for information only; they played no part in the selection. "
        "The calibration period (declining cycle 24 and solar minimum) has weak drag, so the "
        "calibration favours sensitive settings that raise more false alarms at the 2024 "
        "maximum.",
        "",
        main["sensitivity"].to_markdown(floatfmt=".3f"),
        "",
        f"Best test F1 anywhere in the grid (hindsight, not a result): "
        f"{best_test['test f1']:.3f} ({best_test['variant']}, memory {best_test['memory']}, "
        f"κ = {best_test['kappa']:g}, h = {best_test['h']:g}).",
        "",
        "### False alarms on the test period",
        "",
        f"- {fa['n']} false alarms. {fa['storm']} fall within two days of a geomagnetic storm "
        f"(daily Ap ≥ {STORM_AP}): {fa['storm_rate_fa'] * 100:.0f} % of false alarms, against "
        f"{fa['storm_rate_tp'] * 100:.0f} % of true detections and "
        f"{fa['storm_rate_days'] * 100:.0f} % of test days. Storms raise the thermospheric "
        "density for a day or two, which the daily-index drag model only partly captures; "
        "storm-associated alarms are consistent with density-driven decay changes rather "
        "than manoeuvres.",
        f"- {fa['small']} false alarms have |Δa| ≤ 3 m (drag-model residuals); {fa['big']} have "
        f"|Δa| > 3 m, of which {fa['big_with_flag']} coincide(s) with a POD manoeuvre flag "
        "(probable manoeuvres missing from the ESA record). They are still counted as false "
        "alarms.",
        "",
        "### Ablations (test period, same settings)",
        "",
        abl.to_markdown(floatfmt=".3f"),
        "",
        "### By solar-cycle phase (operational mission, frozen settings)",
        "",
        res["by_phase"][["observable", "tp", "fn", "fp", "recall", "precision", "f1"]]
        .to_markdown(floatfmt=".3f"),
        "",
        "## 4. Classification of detected events",
        "",
        "Classes: `station_keeping` (matched to an ESA station-keeping manoeuvre), "
        "`orbit_change` (matched to an inclination, multi-burn sequence or lowering "
        "manoeuvre), `unexplained` (no ESA manoeuvre within tolerance). The third class is "
        "the operational definition agreed at step 1: the ESA record has no anomaly class.",
        "",
        cls["counts"].to_markdown(),
        "",
        f"- Rule baseline (|Δi| > {cls['di_thr']:.2f} mdeg or Δa < −{cls['da_thr']:.2f} m → "
        f"orbit change; Δa > {cls['da_thr']:.2f} m → station keeping; otherwise unexplained): "
        f"macro F1 **{cls['rule']['macro_f1']:.3f}**, accuracy {cls['rule']['accuracy']:.3f}.",
        "- Gradient-boosted trees trained on calibration detections. Feature set chosen by "
        "3-fold time-ordered cross-validation inside the calibration period (macro F1: "
        + ", ".join(f"{k} {v:.3f}" for k, v in cls["cv_scores"].items())
        + f"): **{cls['feature_set']}** ({', '.join(cls['cols'])}). Test macro F1 "
        f"**{cls['ml']['macro_f1']:.3f}**, accuracy {cls['ml']['accuracy']:.3f}.",
        f"- `unexplained` examples: {cls['n_unexp_cal']} in calibration, "
        f"{cls['n_unexp_test']} in test. "
        + ("The learned model beats the rule on the test period."
           if cls["ml"]["macro_f1"] > cls["rule"]["macro_f1"] else
           "The learned model does not beat the rule on the test period: the calibration "
           "period (quiet Sun) holds few `unexplained` examples, while the test period is "
           "dominated by solar-maximum drag events."),
        "- Disclosure: the choice of which classifier the app displays was made after "
        "looking at these test scores, so it is not an out-of-sample model selection.",
        "",
        "Rule, test confusion matrix:",
        "",
        cls["rule"]["confusion"].to_markdown(),
        "",
        "Rule predictions by ESA manoeuvre type (test; `none` = no ESA match):",
        "",
        cls["rule_by_type"].to_markdown(),
        "",
        "Model, test confusion matrix:",
        "",
        cls["ml"]["confusion"].to_markdown(),
        "",
        cls["ml"]["per_class"][["precision", "recall", "f1-score", "support"]]
        .to_markdown(floatfmt=".3f"),
        "",
        "## Figures",
        "",
        "- `groundtrack_template.png`: repeat-cycle signature of a and i.",
        "- `drag_model.png`: observed segment decay vs the power-law model.",
        "- `example_2023-06.png`: raw vs corrected series with detections, June 2023.",
        "- `recall_by_da.png`: test recall by expected |Δa| for the three detectors.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--satellite", default="S1A")
    sat = parser.parse_args().satellite
    OUT.mkdir(parents=True, exist_ok=True)
    revs = pd.read_parquet(processed_dir() / f"revolutions_{sat}.parquet")
    mans = pd.read_parquet(processed_dir() / f"manoeuvres_{sat}.parquet")
    model = MainModel(revs, mans)
    coverage_end = mans["stop"].max() + pd.Timedelta(hours=12)
    ctx = Context(sat, revs, mans, model, (str(model.ops_start.date()), SPLIT),
                  (SPLIT, str(coverage_end)), coverage_end)

    diag = nuisance_diagnostics(ctx)
    main_sel = select_cusum(ctx)
    main_sel["grid"].to_csv(OUT / "cusum_grid.csv", index=False)
    comparison = compare_detectors(ctx, main_sel)
    abl = ablations(ctx, main_sel["cfg"])
    res = main_results(ctx, main_sel["det"])
    fa = false_alarm_analysis(ctx, res["d_test"])
    cls = classification(ctx, res, main_sel["cfg"])
    d_ev = export_outputs(ctx, main_sel, cls, diag, comparison, res)

    fig_template(model, OUT / "groundtrack_template.png")
    fig_drag(model, OUT / "drag_model.png")
    fig_example(model, d_ev, mans, pd.Timestamp("2023-06-01"), pd.Timestamp("2023-07-01"),
                OUT / "example_2023-06.png")
    short_names = ("Baseline, raw", "Baseline, corrected", "Main model")
    fig_recall({short: recall_by(m.assign(bin=da_bin(m)), "bin")
                for short, m in zip(short_names, comparison["tables"].values(), strict=True)},
               OUT / "recall_by_da.png")

    report = render_report(ctx, diag, main_sel, comparison, abl, res, fa, cls)
    (OUT / "model.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
