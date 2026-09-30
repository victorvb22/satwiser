import numpy as np
import pandas as pd
import pytest

from satwiser.detection.baseline import BaselineConfig, detect, scores, step_statistic
from satwiser.evaluation import evaluate, summary
from satwiser.labels import cluster_burns, dv_from_da, manoeuvre_type, manoeuvres

PERIOD = pd.Timedelta(seconds=5924)


def synthetic_revs(n: int = 1500, steps: dict[int, float] | None = None, noise: float = 4.4,
                   drag: float = -0.2, seed: int = 0, missing: range | None = None
                   ) -> pd.DataFrame:
    """Per-revolution mean a with white noise, linear drag decay and step manoeuvres."""
    rng = np.random.default_rng(seed)
    orbit = np.arange(10_000, 10_000 + n)
    a = 7_071_000.0 + drag * np.arange(n) + noise * rng.standard_normal(n)
    for k, da in (steps or {}).items():
        a[k + 1:] += da
        a[k] += da / 2  # burn revolution is half shifted
    t0 = pd.Timestamp("2023-01-01") + PERIOD * np.arange(n)
    revs = pd.DataFrame({"orbit": orbit, "a": a, "t_start": t0, "t_stop": t0 + PERIOD},
                        index=t0 + PERIOD / 2)
    if missing is not None:
        revs = revs.drop(revs.index[list(missing)])
    return revs


def test_step_statistic_measures_step_and_skips_burn_revolution():
    orbit = np.arange(100)
    a = np.where(orbit > 50, 20.0, 0.0)
    a[50] = 10.0
    d = step_statistic(orbit, a, window=5)
    assert d.loc[50] == pytest.approx(20.0)
    assert d.loc[40] == pytest.approx(0.0)


def test_baseline_finds_injected_manoeuvres_at_right_revolution():
    steps = {400: 20.0, 800: -25.0, 1200: 15.0}
    revs = synthetic_revs(steps=steps)
    cfg = BaselineConfig(window=8, threshold=5.0)
    det = detect(scores(revs, cfg), cfg)
    found = sorted(det["orbit"] - 10_000)
    assert len(found) == 3
    for k, f in zip(sorted(steps), found, strict=True):
        assert abs(f - k) <= 1
    jumps = det.sort_values("orbit")["jump_m"].to_numpy()
    np.testing.assert_allclose(jumps, [20.0, -25.0, 15.0], atol=5.0)


def test_baseline_is_quiet_on_noise_only():
    cfg = BaselineConfig(window=8, threshold=5.0)
    det = detect(scores(synthetic_revs(n=3000, seed=3), cfg), cfg)
    assert len(det) <= 1


def test_baseline_tolerates_gaps():
    revs = synthetic_revs(steps={600: 20.0}, missing=range(300, 340))
    cfg = BaselineConfig(window=8, threshold=5.0)
    det = detect(scores(revs, cfg), cfg)
    assert list(det["orbit"] - 10_000) == [600]


def burn_table(rows):
    return pd.DataFrame(rows, columns=["start", "stop", "duration_s", "a1", "a2", "a3"]).assign(
        start=lambda f: pd.to_datetime(f["start"]), stop=lambda f: pd.to_datetime(f["stop"]))


def test_manoeuvre_clustering_and_types():
    burns = burn_table([
        ("2023-06-07 21:40:00", "2023-06-07 21:40:22", 22.0, 0.0, 2.7e-7, 1e-8),
        ("2023-06-07 22:30:00", "2023-06-07 22:30:22", 22.0, 0.0, 2.7e-7, -1e-8),
        ("2023-06-15 00:34:46", "2023-06-15 00:36:35", 109.0, 0.0, -2.5e-8, -3.9e-7),
        ("2023-06-23 13:02:34", "2023-06-23 13:03:19", 45.0, 0.0, 2.7e-7, 0.0),
        ("2023-06-23 14:41:18", "2023-06-23 14:42:03", 45.0, 0.0, -3.1e-7, 0.0),
    ])
    assert list(cluster_burns(burns)) == [0, 0, 1, 2, 2]
    m = manoeuvres(burns)
    assert list(m["type"]) == ["station_keeping", "inclination", "sequence"]
    assert m.loc[0, "dv_t"] == pytest.approx(2 * 2.7e-7 * 1e3 * 22.0)
    assert m.loc[0, "da_pred_m"] == pytest.approx(2 * 7_071_000 * m.loc[0, "dv_t"] / 7508, rel=1e-3)
    assert manoeuvre_type(np.array([-0.01]), np.array([0.0])) == "lowering"


def test_dv_from_da_inverts_gauss_equation():
    assert dv_from_da(19.0) == pytest.approx(0.0101, abs=2e-4)


def test_evaluation_counts():
    steps = {400: 20.0, 800: 25.0}
    revs = synthetic_revs(steps=steps)
    cfg = BaselineConfig(window=8, threshold=5.0)
    det = detect(scores(revs, cfg), cfg)
    # Add a false alarm far from any manoeuvre.
    det = pd.concat([det, pd.DataFrame([{"orbit": 11_100, "z": 6.0, "jump_m": 10.0,
                                         "first_orbit": 11_095}])], ignore_index=True)
    t = revs.set_index("orbit")["t_start"]
    mans = pd.DataFrame({
        "start": [t[10_400] + pd.Timedelta(minutes=30), t[10_800] + pd.Timedelta(minutes=30),
                  t[11_300] + pd.Timedelta(minutes=30)],
        "n_burns": 1, "dv_t": [0.0106, 0.0133, 0.0053], "da_pred_m": [20.0, 25.0, 1.0],
        "type": "station_keeping",
    })
    mans["stop"] = mans["start"] + pd.Timedelta(seconds=30)
    m, d = evaluate(det, mans, revs, window=cfg.window)
    s = summary(m, d)
    assert (s["tp"], s["fn"], s["fp"]) == (2, 1, 1)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["precision"] == pytest.approx(2 / 3)
    assert m.loc[0, "delay_h"] > 0

    # A period starting inside manoeuvre 0's tolerance window: its detection belongs to
    # that manoeuvre (outside the period), so it is neither a false alarm nor a match.
    m_p, d_p = evaluate(det, mans, revs, window=cfg.window,
                        start=str(mans.loc[0, "start"] + pd.Timedelta(minutes=1)))
    s_p = summary(m_p, d_p)
    assert (s_p["tp"], s_p["fn"], s_p["fp"]) == (1, 1, 1)
    matched = d_p["manoeuvre"] >= 0
    assert d_p.loc[matched, "manoeuvre"].isin(m_p.index).all()
