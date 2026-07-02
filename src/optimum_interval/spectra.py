"""Helpers for building a normalized ``spectrum_cdf``.

The optimum-interval methods need a signal-shape CDF that maps the analysis
window onto ``[0, 1]`` (see ``TUTORIAL.md``).  These build one from a signal
density ``dN/dE`` (up to normalization) or from tabulated CDF samples, so you do
not have to integrate and normalize by hand.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .intervals import SpectrumCdf

__all__ = ["spectrum_cdf_from_pdf", "spectrum_cdf_from_samples"]


def spectrum_cdf_from_pdf(
    pdf: Callable[[np.ndarray], np.ndarray],
    e_min: float,
    e_max: float,
    n_grid: int = 2048,
) -> SpectrumCdf:
    """Normalized CDF on ``[e_min, e_max]`` from a signal density ``pdf``.

    ``pdf`` need not be normalized -- only its *shape* matters (the overall rate,
    i.e. ``mu``, is what the method limits and must not be folded in).  The
    density is cumulatively integrated (trapezoidal) on a fine grid and the
    result is normalized so ``cdf(e_min) == 0`` and ``cdf(e_max) == 1``.

    Parameters
    ----------
    pdf : callable
        Non-negative density ``dN/dE`` evaluated on an array of energies.
    e_min, e_max : float
        Analysis-window bounds.
    n_grid : int, optional
        Integration/interpolation grid size.
    """
    grid = np.linspace(e_min, e_max, n_grid)
    density = np.asarray(pdf(grid), dtype=float)
    if np.any(density < 0.0):
        raise ValueError("pdf must be non-negative on [e_min, e_max]")
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))]
    )
    total = cumulative[-1]
    if total <= 0.0:
        raise ValueError("pdf integrates to zero over [e_min, e_max]")
    cumulative /= total

    def cdf(energies):
        e = np.clip(np.asarray(energies, dtype=float), e_min, e_max)
        return np.interp(e, grid, cumulative)

    return cdf


def spectrum_cdf_from_samples(energies, cdf_values) -> SpectrumCdf:
    """Normalized ``spectrum_cdf`` interpolating tabulated ``(energy, CDF)`` points.

    Useful when the CDF is only known at sampled energies (e.g. from a
    simulation).  ``cdf_values`` must be non-decreasing; they are rescaled so the
    endpoints map to exactly ``[0, 1]``.
    """
    e = np.asarray(energies, dtype=float)
    c = np.asarray(cdf_values, dtype=float)
    if e.ndim != 1 or e.shape != c.shape:
        raise ValueError("energies and cdf_values must be 1-D and the same length")
    if np.any(np.diff(e) <= 0.0):
        raise ValueError("energies must be strictly increasing")
    if np.any(np.diff(c) < 0.0):
        raise ValueError("cdf_values must be non-decreasing")
    c = (c - c[0]) / (c[-1] - c[0])

    def cdf(x):
        x = np.clip(np.asarray(x, dtype=float), e[0], e[-1])
        return np.interp(x, e, c)

    return cdf
