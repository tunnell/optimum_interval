"""Slow calibration/coverage test.

Confirms the 90% construction actually covers: build the calibration on one
Monte-Carlo sample, then draw *fresh* (out-of-sample) background-free experiments
and check that ~90% of them are NOT excluded (limit above the true mu).  Skip
with ``pytest -m 'not slow'``.
"""

import numpy as np
import pytest

from optimum_interval import OptimumIntervalTable


@pytest.mark.slow
@pytest.mark.parametrize("mu0", [5.0, 15.0, 30.0])
def test_out_of_sample_coverage(mu0):
    table = OptimumIntervalTable(rng=np.random.default_rng(0))
    table.generate(mu0, 20000)
    threshold = table.bar_c_max(0.9, mu0)

    rng = np.random.default_rng(1)
    n_exp = 3000
    excluded = 0
    for _ in range(n_exp):
        fresh = rng.random(rng.poisson(mu0))  # endpoints added inside the statistic
        if table.optimum_interval_statistic(fresh, mu0) > threshold:
            excluded += 1
    coverage = 1.0 - excluded / n_exp
    # Yellin's construction guarantees at least ~90% coverage (possibly
    # conservative from discreteness); allow a band for MC noise.
    assert 0.88 <= coverage <= 0.97, f"mu0={mu0}: coverage={coverage:.3f}"
