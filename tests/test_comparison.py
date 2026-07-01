"""Tests for the per-experiment comparison limits (Fig. 3 / 4 machinery)."""

import numpy as np
import pytest

from optimum_interval import (
    ComparisonEngine,
    max_gap_upper_limit,
    poisson_upper_limit,
)


def test_poisson_upper_limit_known_values():
    # Classic 90% CL Poisson upper limits.
    assert poisson_upper_limit(0, 0.9) == pytest.approx(2.302585, abs=1e-4)
    assert poisson_upper_limit(1, 0.9) == pytest.approx(3.8897, abs=1e-3)
    assert poisson_upper_limit(3, 0.9) == pytest.approx(6.6808, abs=1e-3)


def test_max_gap_limit_zero_event_case():
    # A single full-range gap (0 observed events) must reproduce the 0-event
    # max-gap limit, which equals the Poisson 0-event limit 2.3026.
    assert max_gap_upper_limit(1.0, 0.9) == pytest.approx(2.302585, abs=1e-3)


def test_max_gap_limit_monotonic_in_gap():
    # A larger observed gap fraction => stronger (lower) upper limit.
    assert max_gap_upper_limit(0.8) < max_gap_upper_limit(0.4)


def test_comparison_engine_limits_are_sane():
    grid = np.linspace(2.4, 40.0, 20)
    engine = ComparisonEngine(grid, n_cal=3000, rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    events = np.sort(rng.random(10))            # 10 events in cumulant space
    cmax = engine.cmax_upper_limit(events)
    pmax = engine.pmax_upper_limit(events)
    for limit in (cmax, pmax):
        assert np.isfinite(limit)
        assert grid[0] <= limit <= grid[-1]
        assert limit > events.size / 2          # limit is in a plausible range
