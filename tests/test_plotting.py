"""Smoke test for the plotting helpers."""

import matplotlib
import numpy as np

matplotlib.use("Agg")


def test_plot_bar_c_max_smoke():
    from optimum_interval import OptimumIntervalTable
    from optimum_interval.plotting import bar_c_max_curve, plot_bar_c_max

    table = OptimumIntervalTable(rng=np.random.default_rng(0))
    mu_values = np.array([3.0, 5.0])
    curve = bar_c_max_curve(table, mu_values, n=400)
    assert curve.shape == mu_values.shape
    ax = plot_bar_c_max(mu_values, curve)  # exercises threshold overlay too
    assert ax.get_xlabel()
