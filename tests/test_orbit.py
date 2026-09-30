import numpy as np
import pandas as pd
import pytest

from satwiser.orbit.elements import (
    J2,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH,
    ecef_to_inertial,
    j2_short_period_a,
    keplerian,
)
from satwiser.orbit.revolutions import revolution_ids, revolution_means

A_S1 = 7_071_000.0
I_S1 = np.deg2rad(98.18)


def circular_states(a: float, inc: float, raan: float, u: np.ndarray):
    """Inertial states on a circular orbit at arguments of latitude ``u``."""
    v = np.sqrt(MU_EARTH / a)
    cu, su = np.cos(u), np.sin(u)
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(raan), np.sin(raan)
    r = a * np.column_stack((co * cu - so * su * ci, so * cu + co * su * ci, su * si))
    vel = v * np.column_stack((-co * su - so * cu * ci, -so * su + co * cu * ci, cu * si))
    return r, vel


def test_keplerian_circular_orbit():
    u = np.linspace(0.1, 6.0, 7)
    r, v = circular_states(A_S1, I_S1, 0.7, u)
    el = keplerian(r, v)
    np.testing.assert_allclose(el["a"], A_S1, rtol=1e-12)
    np.testing.assert_allclose(el["e"], 0.0, atol=1e-12)
    np.testing.assert_allclose(el["i"], I_S1, atol=1e-12)
    np.testing.assert_allclose(el["raan"], 0.7, atol=1e-12)
    np.testing.assert_allclose(el["u"], u, atol=1e-10)


def test_keplerian_eccentric_energy():
    # Perigee state of an orbit with known a and e.
    a, e = 7.2e6, 0.01
    rp = a * (1 - e)
    vp = np.sqrt(MU_EARTH * (1 + e) / rp)
    el = keplerian(np.array([[rp, 0.0, 0.0]]),
                   np.array([[0.0, vp * np.cos(I_S1), vp * np.sin(I_S1)]]))
    assert el["a"][0] == pytest.approx(a, rel=1e-12)
    assert el["e"][0] == pytest.approx(e, rel=1e-9)


def test_ecef_rotation_preserves_semi_major_axis():
    r, v = circular_states(A_S1, I_S1, 1.2, np.array([0.3, 2.0]))
    # Build Earth-fixed states for an arbitrary sidereal angle, then convert back.
    theta = np.array([0.4, 0.4])
    c, s = np.cos(-theta), np.sin(-theta)
    r_e = np.column_stack((c * r[:, 0] - s * r[:, 1], s * r[:, 0] + c * r[:, 1], r[:, 2]))
    v_rot = np.column_stack((c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]))
    v_e = v_rot - np.cross([0.0, 0.0, OMEGA_EARTH], r_e)
    r_i, v_i = ecef_to_inertial(r_e, v_e, theta)
    np.testing.assert_allclose(r_i, r, atol=1e-6)
    np.testing.assert_allclose(v_i, v, atol=1e-9)


def test_j2_short_period_amplitude_matches_sentinel1():
    amp = j2_short_period_a(np.array([A_S1]), np.array([I_S1]), np.array([0.0]))[0]
    assert amp == pytest.approx(1.5 * J2 * R_EARTH**2 / A_S1 * np.sin(I_S1) ** 2)
    assert 9_000 < amp < 9_300


def test_revolution_ids_increment_at_node():
    u = np.mod(np.linspace(0, 6 * np.pi, 1000) + 0.5, 2 * np.pi)
    ids = revolution_ids(u)
    assert ids[0] == 0 and ids[-1] == 3


def synthetic_elements(n_rev: int = 40, burn_rev: int = 20, da_burn: float = 20.0,
                       step_s: float = 10.0, drag_m_per_rev: float = -0.15) -> pd.DataFrame:
    """Osculating a with J2 short-period term, drag decay and one along-track burn."""
    period = 2 * np.pi * np.sqrt(A_S1**3 / MU_EARTH)
    t = np.arange(0.0, n_rev * period, step_s)
    u = np.mod(2 * np.pi * t / period + 0.37, 2 * np.pi)
    mean_a = A_S1 + drag_m_per_rev * t / period + da_burn * (t > burn_rev * period + 1000)
    a = mean_a + j2_short_period_a(mean_a, np.full_like(t, I_S1), u)
    index = pd.Timestamp("2023-06-01") + pd.to_timedelta(t, unit="s")
    frame = pd.DataFrame({"a": a, "e": 1e-3, "i": I_S1, "u": u}, index=index)
    frame["a_nosp"] = frame["a"] - j2_short_period_a(frame["a"].to_numpy(), frame["i"], u)
    return frame


def test_revolution_means_remove_j2_and_expose_burn():
    el = synthetic_elements()
    means = revolution_means(el)
    assert len(means) >= 36
    jumps = means["a"].diff().dropna()
    # The burn revolution is partly shifted, so the step spreads over two differences.
    top2 = jumps.abs().nlargest(2)
    assert jumps[top2.index].sum() == pytest.approx(20.0 - 2 * 0.15, abs=0.5)
    burn_epoch = pd.Timestamp("2023-06-01") + pd.Timedelta(
        seconds=20 * 2 * np.pi * np.sqrt(A_S1**3 / MU_EARTH) + 1000)
    burn_rev = means[(means["t_start"] <= burn_epoch) & (means["t_stop"] >= burn_epoch)].index
    assert burn_rev[0] in top2.index
    # Without the correction the boundary jitter alone would be ~15 m per revolution.
    assert jumps.drop(top2.index).abs().max() < 1.0


def test_plain_average_suffers_boundary_jitter():
    el = synthetic_elements(da_burn=0.0).drop(columns="a_nosp")
    jumps = revolution_means(el)["a"].diff().dropna()
    assert jumps.abs().max() > 3.0
