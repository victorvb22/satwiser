import numpy as np
import pandas as pd
import pytest

from satwiser.classify import rule
from satwiser.detection.cusum import Channel, CusumConfig, detect, run_cusum, step_sizes
from satwiser.model.drag import DragModel, fit_drag, segment_slopes
from satwiser.model.grid import segments, to_grid
from satwiser.model.groundtrack import REPEAT, fit_template

PERIOD = pd.Timedelta(seconds=5924)


def synthetic_grid(n=3000, steps=None, rate=-0.05, noise=0.2, seed=0, amp=10.0):
    """Revolution table with a repeat-cycle signature, linear decay and step manoeuvres."""
    rng = np.random.default_rng(seed)
    orbit = np.arange(1000, 1000 + n)
    signature = amp * np.sin(2 * np.pi * (orbit % REPEAT) / REPEAT * 3)
    a = 7_071_000.0 + rate * np.arange(n) + signature + noise * rng.standard_normal(n)
    i = np.deg2rad(98.18) + 1e-6 * rng.standard_normal(n)
    for k, da in (steps or {}).items():
        a[k:] += da
    t0 = pd.Timestamp("2021-01-01") + PERIOD * np.arange(n)
    revs = pd.DataFrame({"orbit": orbit, "a": a, "e": 1e-3, "i": i, "n": 590,
                         "t_start": t0, "t_stop": t0 + PERIOD,
                         "f107_obs": 100.0, "f107_81d": 100.0, "ap": 8.0}, index=t0 + PERIOD / 2)
    return revs, signature


def test_segments():
    mask = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], bool)
    assert [list(s) for s in segments(mask, 3)] == [[0, 1, 2], [7, 8, 9, 10]]


def test_template_recovers_repeat_signature():
    revs, signature = synthetic_grid()
    g = to_grid(revs)
    mask = np.ones(len(g), bool)
    tpl = fit_template(g, mask, mask, columns=("a",))
    est = tpl.signature(g.index.to_numpy(), "a")
    assert np.corrcoef(est, signature)[0, 1] > 0.99
    corrected = tpl.correct(g, columns=("a",))["a_c"].to_numpy()
    assert np.std(np.diff(corrected)) < 0.5


def test_drag_power_law_fit():
    rows = []
    for f81 in (70, 90, 120, 160, 200):
        rows.append({"n": 100, "slope": -np.exp(-15) * f81**2.6, "f81": f81, "f": f81, "ap": 8})
    model = fit_drag(pd.DataFrame(rows))
    assert model.beta[1] == pytest.approx(2.6, abs=1e-6)
    assert model.rate([150], [150], [8])[0] == pytest.approx(-np.exp(-15) * 150**2.6, rel=1e-6)


def test_segment_slopes_and_cumulative():
    revs, _ = synthetic_grid(amp=0.0, rate=-0.1)
    g = to_grid(revs)
    s = segment_slopes(g, "a", np.ones(len(g), bool))
    assert s["slope"].iloc[0] == pytest.approx(-0.1, abs=0.01)
    drag = DragModel(np.array([np.log(0.1), 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(np.diff(drag.cumulative(g)), -0.1)


def test_cusum_detects_steps_at_right_revolution_and_estimates_size():
    y = 0.2 * np.random.default_rng(1).standard_normal(1000)
    y[300:] += 2.0
    y[700:] -= 3.0
    valid = np.ones(1000, bool)
    ch = Channel("a", y, 0.2, "mean")
    cfg = CusumConfig(kappa=2.0, h=12.0, memory=20, clip=6.0)
    alarms = run_cusum([ch], valid, cfg)
    assert list(alarms["change_pos"]) == [300, 700]
    assert (alarms["alarm_pos"] - alarms["change_pos"]).max() <= 3
    sizes = step_sizes({"a": y}, valid, alarms["change_pos"].to_numpy(), 8, 0)
    np.testing.assert_allclose(sizes["da"], [2.0, -3.0], atol=0.4)


def test_cusum_clip_ignores_single_outlier():
    y = 0.2 * np.random.default_rng(2).standard_normal(500)
    y[250] += 50.0
    ch = Channel("a", y, 0.2, "mean")
    alarms = run_cusum([ch], np.ones(500, bool), CusumConfig(kappa=2.0, h=12.0, clip=6.0))
    assert alarms.empty


def test_cusum_detect_on_grid_reports_orbits():
    revs, _ = synthetic_grid(amp=0.0, rate=0.0, steps={1500: 2.0})
    g = to_grid(revs)
    y = g["a"].to_numpy() - g["a"].iloc[0]
    det = detect(g, [Channel("a", y, 0.2, "mean")], CusumConfig(kappa=2.0, h=12.0),
                 {"a": y})
    assert len(det) == 1
    assert det["orbit"].iloc[0] == 2500
    assert det["jump_m"].iloc[0] == pytest.approx(2.0, abs=0.4)


def test_rule_classes():
    x = pd.DataFrame({"da": [15.0, -8.0, 2.0, 0.5], "abs_di": [0.0, 0.0, 0.5, 0.01]})
    assert list(rule(x, di_threshold_mdeg=0.2, da_threshold_m=1.5)) == [
        "station_keeping", "orbit_change", "orbit_change", "unexplained"]


def test_cusum_adaptive_scale_absorbs_a_variance_change():
    rng = np.random.default_rng(5)
    y = np.concatenate([0.2 * rng.standard_normal(600), 1.0 * rng.standard_normal(600)])
    valid = np.ones(len(y), bool)
    fixed = run_cusum([Channel("a", y, 0.2, "mean")], valid,
                      CusumConfig(kappa=2.0, h=12.0, memory=20, clip=6.0))
    adaptive = run_cusum([Channel("a", y, 0.2, "mean")], valid,
                         CusumConfig(kappa=2.0, h=12.0, memory=20, clip=6.0, scale_window=200))
    assert len(fixed) > 5
    assert len(adaptive) <= 1


def test_cusum_linear_predictor_follows_a_drift():
    rng = np.random.default_rng(6)
    y = 0.05 * np.arange(800) + 0.2 * rng.standard_normal(800)
    y[400:] += 3.0
    alarms = run_cusum([Channel("a", y, 0.2, "linear")], np.ones(800, bool),
                       CusumConfig(kappa=2.0, h=12.0, memory=30, clip=6.0))
    assert list(alarms["change_pos"]) == [400]


def test_cusum_multichannel_alarm_on_the_right_channel():
    rng = np.random.default_rng(7)
    a = 0.2 * rng.standard_normal(1000)
    i = 0.02 * rng.standard_normal(1000)
    i[600:] += 0.3
    alarms = run_cusum([Channel("a", a, 0.2, "mean"), Channel("i", i, 0.02, "linear")],
                       np.ones(1000, bool), CusumConfig(kappa=2.0, h=12.0, memory=20, clip=6.0))
    assert list(alarms["channel"]) == ["i"]
    assert alarms["change_pos"].iloc[0] == 600
