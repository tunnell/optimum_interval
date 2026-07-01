"""Tests for the analytic max-gap statistic C_0 (Yellin Eq. 2)."""

import numpy as np
import pytest

from optimum_interval.analytic import c0, x0


def test_bounds():
    assert c0(0.0, 5.0) == 0.0
    assert c0(10.0, 5.0) == 1.0                       # x > mu  =>  gap certainly smaller
    # x == mu: a zero-event experiment has max gap = mu (not < mu), so
    # C0(mu, mu) = P(>=1 event) = 1 - e^{-mu}, NOT 1.
    assert c0(5.0, 5.0) == pytest.approx(1.0 - np.exp(-5.0))


def test_x0_raises_below_threshold():
    # For mu < 2.3026 the max attainable C0 is < 0.9, so no x0(0.9, mu) exists.
    with pytest.raises(ValueError):
        x0(0.9, 1.5)
    # Just above the threshold it is defined.
    assert x0(0.9, 2.5) > 0


def test_in_unit_range_and_monotonic():
    mu = 6.0
    xs = np.linspace(0.01, mu, 40)
    vals = c0(xs, mu)
    assert np.all((vals >= 0.0) & (vals <= 1.0))
    assert np.all(np.diff(vals) >= -1e-12)   # non-decreasing in x


def test_singular_point_is_finite():
    # mu = k*x occurs e.g. at x = mu/3; the telescoped form must stay finite.
    mu = 6.0
    for divisor in (1, 2, 3, 4):
        assert np.isfinite(c0(mu / divisor, mu))


def test_x0_inverts_c0():
    mu = 4.0
    x = x0(0.9, mu)
    assert c0(x, mu) == pytest.approx(0.9, abs=1e-6)


def test_vectorized_matches_scalar():
    mu = 7.0
    xs = np.linspace(0.1, mu, 15)
    np.testing.assert_allclose(c0(xs, mu), [c0(float(x), mu) for x in xs])
