"""Tests for the Monte-Carlo tables and the upper-limit solver.

The key correctness check is that the k=0 Monte-Carlo max-gap distribution
reproduces the analytic C_0 (Yellin Eq. 2), which needs no simulation.
"""

import numpy as np
import pytest

from optimum_interval.analytic import c0
from optimum_interval.montecarlo import OptimumIntervalTable


def test_mc_maxgap_matches_analytic_c0():
    """Empirical CDF of the k=0 gap must match C_0 within Monte-Carlo error.

    Units: MC gap sizes are fractions of [0, 1]; Yellin's ``x`` is in expected
    events, so a fraction ``f`` maps to ``x = mu * f``.
    """
    mu = 5.0
    n = 20000
    table = OptimumIntervalTable(rng=np.random.default_rng(0))
    table.generate(mu, n)

    gap_fractions = table.itv_sizes[mu][0]
    gap_events = mu * gap_fractions  # convert to Yellin's x units

    tol = 4.0 / np.sqrt(n)  # a few-sigma binomial band
    for x in (1.0, 2.0, 3.0, 4.0):
        empirical = np.count_nonzero(gap_events < x) / n
        assert empirical == pytest.approx(c0(x, mu), abs=tol)


def test_reproducible_with_seed():
    a = OptimumIntervalTable(rng=np.random.default_rng(42))
    b = OptimumIntervalTable(rng=np.random.default_rng(42))
    a.generate(6.0, 500)
    b.generate(6.0, 500)
    np.testing.assert_array_equal(a.opt_itvs[6.0], b.opt_itvs[6.0])


def test_bar_c_max_in_range():
    table = OptimumIntervalTable(rng=np.random.default_rng(1))
    table.generate(5.0, 5000)
    value = table.bar_c_max(0.9, 5.0)
    assert 0.85 <= value <= 1.0


def test_upper_limit_smoke():
    """A run with many events yields a finite limit above the event count scale."""
    rng = np.random.default_rng(7)
    table = OptimumIntervalTable(rng=rng)
    events = np.sort(rng.random(40))  # 40 events in cumulant space
    limit = table.upper_limit(events, confidence=0.9, n=400)
    assert np.isfinite(limit)
    assert limit > events.size  # limit above the observed count


def test_upper_limit_small_n_experiments():
    """Small-N experiments (the method's target regime) must give sane limits.

    Regression test: the solver used to silently return mu_scan_start (10.0) for
    0- or 1-event runs.  The 0-event 90% limit should be near the classic ~2.3.
    """
    table = OptimumIntervalTable(rng=np.random.default_rng(3))
    zero = table.upper_limit(np.array([]), confidence=0.9, n=4000)
    assert 1.5 < zero < 3.5  # classic 0-event Poisson limit ~2.3
    one = table.upper_limit(np.array([0.4]), confidence=0.9, n=4000)
    assert zero < one < 6.0  # one event -> larger limit


def test_upper_limit_raises_when_unbracketed():
    """No silent wrong answer: raise if the scan range can't bracket the limit."""
    table = OptimumIntervalTable(rng=np.random.default_rng(4))
    events = np.sort(np.random.default_rng(5).random(30))
    with pytest.raises(RuntimeError):
        table.upper_limit(events, confidence=0.9, n=300, mu_scan_stop=3)
