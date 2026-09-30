import numpy as np
import pytest

from satwiser import lab
from satwiser.orbit.elements import MU_EARTH

A = 7_071_000.0


def circular_states(n: int, inc: float = np.deg2rad(98.18)) -> np.ndarray:
    period = 2 * np.pi * np.sqrt(A**3 / MU_EARTH)
    u = 2 * np.pi * np.arange(n) * lab.STEP_S / period
    v = np.sqrt(MU_EARTH / A)
    r = A * np.column_stack((np.cos(u), np.sin(u) * np.cos(inc), np.sin(u) * np.sin(inc)))
    vel = v * np.column_stack((-np.sin(u), np.cos(u) * np.cos(inc), np.cos(u) * np.sin(inc)))
    return np.hstack([r, vel])


def test_ar1_matches_recursion_and_has_unit_variance():
    x = np.random.default_rng(0).standard_normal((20000, 2))
    rho = 0.9
    y = lab.ar1(x, rho)
    ref = np.empty_like(x)
    ref[0] = x[0]
    for k in range(1, len(x)):
        ref[k] = rho * ref[k - 1] + np.sqrt(1 - rho**2) * x[k]
    np.testing.assert_allclose(y, ref, atol=1e-10)
    assert y.std() == pytest.approx(1.0, abs=0.1)
    assert np.corrcoef(y[1:, 0], y[:-1, 0])[0, 1] == pytest.approx(rho, abs=0.02)


def test_subsample_and_bin_size():
    assert len(lab.subsample(8640, 8640)) == 8640
    assert list(lab.subsample(8640, 4, phase=5)) == [5, 2165, 4325, 6485]
    assert lab.bin_revolutions(8640) == 1
    assert lab.bin_revolutions(24) == 1
    assert lab.bin_revolutions(2) == 8


def test_degradation_maps_to_semi_major_axis_error():
    states = circular_states(5000)
    clean = lab.mean_a_nosp(states)
    sigma = 10.0
    noisy = lab.degrade(states, lab.Degradation(sigma_m=sigma),
                        np.random.default_rng(1).standard_normal(states.shape))
    ratio = np.std(lab.mean_a_nosp(noisy) - clean) / sigma
    assert 2.0 < ratio < 3.5
    np.testing.assert_array_equal(lab.degrade(states, lab.Degradation(), np.ones(states.shape)),
                                  states)


def test_bin_means_with_template_and_gaps():
    orbit = np.array([100, 100, 101, 103, 103])
    a = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    template = np.zeros(175)
    template[101 % 175] = 1.0
    bins, means = lab.bin_means(a, orbit, template, 1)
    np.testing.assert_array_equal(bins, [0, 1, 2, 3])
    np.testing.assert_allclose(means[[0, 1, 3]], [2.0, 4.0, 8.0])
    assert np.isnan(means[2])


def test_window_detector_finds_step():
    rng = np.random.default_rng(3)
    means = 0.3 * rng.standard_normal(146)
    means[80:] += 3.0
    _, z = lab.window_scores(means, 12)
    found = lab.window_detect(z, 4.0, 12)
    assert len(found) == 1 and abs(found[0] - 80) <= 1


def test_window_detector_insufficient_data():
    _, z = lab.window_scores(np.array([1.0, 2.0, 3.0]), 2)
    assert np.isnan(z).all()
    assert lab.window_detect(z, 4.0, 2) == []


def test_diff_normalisation_is_calibrated_at_edges_and_gaps():
    rng = np.random.default_rng(4)
    extremes = []
    for _ in range(200):
        means = 1.0 * rng.standard_normal(40) - 0.5 * np.arange(40)
        means[rng.choice(40, 6, replace=False)] = np.nan
        _, z = lab.window_scores(means, 6, "diff", 0.0)
        extremes.append(np.nanmax(np.abs(z)))
    # Pure noise plus drift: |z| > 5 should be rare everywhere, edges included.
    assert np.mean(np.array(extremes) > 5) < 0.05
