"""Plotting helpers, kept separate so headless Monte-Carlo has no GUI dependency."""

from __future__ import annotations

import numpy as np

from .montecarlo import OptimumIntervalTable

__all__ = ["bar_c_max_curve", "plot_bar_c_max"]

# Thresholds in mu at which intervals with n events first contribute to C_max
# (Yellin Table I).  bar_c_max(0.9, mu) steps upward as each is crossed.
TABLE_I_THRESHOLDS = [
    2.303, 3.890, 5.800, 7.491, 9.059, 10.548, 12.009, 13.433, 14.824, 16.196,
    17.540, 18.891, 20.208, 21.520, 22.821,
]


def bar_c_max_curve(
    table: OptimumIntervalTable,
    mu_values: np.ndarray,
    confidence: float = 0.9,
    n: int = 10000,
) -> np.ndarray:
    """Evaluate :math:`\\bar C_\\mathrm{max}(\\text{confidence}, \\mu)` on a grid.

    Builds (or reuses cached) Monte-Carlo tables for each ``mu`` and returns the
    ``confidence`` quantile of the C_max distribution -- the curve of Fig. 2.
    """
    curve = np.empty(len(mu_values))
    for i, mu in enumerate(mu_values):
        table.generate(mu, n)
        curve[i] = table.bar_c_max(confidence, mu)
    return curve


def plot_bar_c_max(
    mu_values: np.ndarray,
    curve: np.ndarray,
    ax=None,
    confidence: float = 0.9,
    mark_thresholds: bool = True,
):
    """Plot a reproduction of Yellin Fig. 2.  Returns the matplotlib ``Axes``."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mu_values, curve, lw=1.5, color="C0")
    if mark_thresholds:
        for mu_thr in TABLE_I_THRESHOLDS:
            if mu_values.min() <= mu_thr <= mu_values.max():
                ax.axvline(mu_thr, color="0.8", lw=0.8, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(mu_values.min(), mu_values.max())
    ax.set_xlabel("Total expected number of events  $\\mu$")
    ax.set_ylabel(rf"${int(confidence * 100)}\%\ \bar C_{{\mathrm{{Max}}}}$")
    ax.set_title(r"Reproduction of Yellin Fig. 2: $\bar C_{\mathrm{Max}}(0.9,\mu)$")
    return ax
