"""Fast golden-value regression tests distilled from reproduce_figures.py.

These lock the load-bearing reproduction facts into the test suite at reduced
statistics, without committing the (copyrighted) paper figures.
"""

import numpy as np
import pytest

from optimum_interval import OptimumIntervalTable


def test_bar_cmax_plateau_at_0p9():
    """Below the n=1 threshold (mu < 3.89) only n=0 contributes, so the
    bar_C_max(0.9) plateau sits at ~0.90 (the 90th percentile of a uniform)."""
    table = OptimumIntervalTable(rng=np.random.default_rng(0))
    for mu in (2.6, 3.2, 3.7):
        table.generate(mu, 20000)
        assert table.bar_c_max(0.9, mu) == pytest.approx(0.90, abs=0.015)


def test_bar_cmax_rises_above_plateau():
    """bar_C_max(0.9) increases past the plateau as mu grows (Fig. 2 shape)."""
    table = OptimumIntervalTable(rng=np.random.default_rng(1))
    table.generate(10.0, 20000)
    table.generate(30.0, 20000)
    assert table.bar_c_max(0.9, 30.0) > table.bar_c_max(0.9, 10.0) > 0.9
