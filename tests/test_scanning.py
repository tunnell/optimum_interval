"""Tests for the parameter-scan helpers (optimum_interval.scanning)."""

import numpy as np
import pytest

from optimum_interval import (
    excluded_interval,
    new_table,
    poisson_upper_limit,
    round_log,
    scan_extremeness,
    spectrum_from_rate,
)


def test_round_log_shares_grid_points():
    assert round_log(1.0) == pytest.approx(1.0)
    # values closer than half a grid step round to the same point
    a, b = round_log(3.000), round_log(3.005)
    assert a == b
    # and the grid is 2%-spaced in the log
    assert np.isclose(np.log10(round_log(3.2) / round_log(3.0)), 0.02, atol=1e-9)


def test_spectrum_from_rate_trims_and_normalizes():
    x = np.linspace(1.0, 10.0, 100)
    rate = np.where(x < 5.0, 1.0, 0.0)  # flat rate with dead tail
    mu, cdf, x_lo, x_hi = spectrum_from_rate(x, rate, exposure=2.0)
    assert x_lo == 1.0
    assert x_hi == pytest.approx(5.0, abs=0.1)   # trailing zeros trimmed
    assert mu == pytest.approx(2.0 * (x_hi - 1.0), rel=1e-2)
    assert cdf(x_lo) == pytest.approx(0.0, abs=1e-12)
    assert cdf(x_hi) == pytest.approx(1.0, abs=1e-12)
    assert spectrum_from_rate(x, np.zeros_like(x), 1.0) is None


def test_scan_extremeness_zero_events_matches_poisson():
    """With no events, p crosses 0.95 exactly at the zero-event Poisson limit."""
    table = new_table(seed=3)
    x = np.linspace(0.0, 1.0, 200)
    events = np.array([])
    mu95 = poisson_upper_limit(0, 0.95)   # ~3.0

    def p_at(scale):
        rate = np.full_like(x, scale)
        p, mu = scan_extremeness(table, events, x, rate, exposure=1.0, n=6000)
        return p, mu

    p_below, _ = p_at(mu95 * 0.8)
    p_above, _ = p_at(mu95 * 1.2)
    assert p_below < 0.95 < p_above


def test_scan_extremeness_shortcuts_and_window():
    table = new_table(seed=1)
    x = np.linspace(0.0, 1.0, 50)
    rate = np.ones_like(x)
    # mu below the floor: nothing expected, p = 0
    p, mu = scan_extremeness(table, [0.5], x, rate, exposure=0.01)
    assert (p, round(mu, 2)) == (0.0, 0.01)
    # mu above the cap: overwhelming, p = 1, no MC
    p, mu = scan_extremeness(table, [0.5], x, rate, exposure=1000.0)
    assert p == 1.0 and mu == pytest.approx(1000.0)
    # events beyond the support window are dropped: same as no events
    rate_short = np.where(x < 0.5, 1.0, 0.0)
    p_out, _ = scan_extremeness(table, [0.9], x, rate_short, exposure=8.0,
                                n=4000)
    p_none, _ = scan_extremeness(table, [], x, rate_short, exposure=8.0,
                                 n=4000)
    assert p_out == pytest.approx(p_none)


def test_excluded_interval_edges():
    params = np.geomspace(1e-8, 1e-2, 61)
    # band excluded between 1e-6 and 1e-4
    ps = np.where((params > 1e-6) & (params < 1e-4), 1.0, 0.0)
    lo, hi = excluded_interval(params, ps, level=0.95)
    # crossings interpolate within one grid step of the true edges
    assert 1e-6 <= lo < 1.3e-6
    assert 0.7e-4 < hi <= 1e-4
    # nothing excluded
    lo, hi = excluded_interval(params, np.zeros_like(params))
    assert np.isnan(lo) and np.isnan(hi)
    # excluded through the end of the scan: edge saturates
    ps = np.where(params > 1e-6, 1.0, 0.0)
    lo, hi = excluded_interval(params, ps)
    assert hi == params[-1]
