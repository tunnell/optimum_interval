"""Pure interval geometry for Yellin's optimum-interval method.

The functions here are deterministic and side-effect free.  They operate on
points that already live in *cumulant space*: the probability-integral
transform :math:`\\epsilon(E)` has already mapped observed energies onto the
unit interval, where the expected signal is uniform with unit density (see
:mod:`optimum_interval` package docstring and ``EXPLANATION.md``).

Nothing here does Monte Carlo, I/O, or plotting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["cumulant_points", "k_largest_intervals"]


def k_largest_intervals(
    points: np.ndarray,
    spectrum_cdf: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[int, float]:
    """Size of the largest interval containing exactly ``k`` events, for each ``k``.

    An "interval" is a stretch of the parameter (energy) axis; its **size** is
    the expected fraction of signal events it contains.  The *k-largest*
    interval of a run is the largest interval that happens to contain exactly
    ``k`` observed events -- i.e. an unusually empty region.  Intervals are
    delimited by observed events (and, for a real measurement, by the range
    endpoints, which the caller must include; see :func:`cumulant_points`).

    Parameters
    ----------
    points : array_like
        1-D event positions.  If ``spectrum_cdf`` is ``None`` the points are
        assumed to already be in cumulant space (uniform ``[0, 1]``).  Boundary
        points (0 and 1) are **not** added here -- the caller decides, so that
        Monte-Carlo trials and real data are treated identically and boundaries
        are never double-counted.
    spectrum_cdf : callable, optional
        Monotonic CDF mapping raw positions to cumulants.  Applied *after*
        sorting.  Defaults to the identity.

    Returns
    -------
    dict[int, float]
        Mapping ``k -> size`` where ``size`` is the largest interval containing
        exactly ``k`` events.  ``k`` runs from 0 (the maximum gap) up to
        ``len(points) - 2``.

    Notes
    -----
    For a sorted point list ``p`` the largest interval containing ``k`` events
    is ``max_i (p[i + k + 1] - p[i])`` -- the widest span between two points
    that are ``k + 1`` apart in the sorted order (hence ``k`` points strictly
    between them).  ``k = 0`` reduces to the maximum gap of the max-gap method.

    Examples
    --------
    >>> k_largest_intervals(np.array([0.0, 0.1, 0.2, 0.84, 0.85]))[0]
    0.64
    """
    p = np.sort(np.asarray(points, dtype=float))  # copy => no in-place side effect
    if spectrum_cdf is not None:
        p = spectrum_cdf(p)

    n_points = p.size
    sizes: dict[int, float] = {}
    for k in range(n_points - 1):
        gap = k + 1
        sizes[k] = float(np.max(p[gap:] - p[:-gap]))
    return sizes


def cumulant_points(
    events: np.ndarray,
    spectrum_cdf: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Map observed events to cumulant space and append the range endpoints.

    This is the transform every *real* measurement must undergo before calling
    :func:`k_largest_intervals`: apply the (normalized) spectrum CDF so the
    signal becomes uniform on ``[0, 1]``, then add the experimental-range
    boundaries 0 and 1 (which act as interval delimiters exactly like events).

    The Monte-Carlo path builds equivalent point lists directly (interior
    uniforms plus 0 and 1), so both paths are perfectly consistent.

    Parameters
    ----------
    events : array_like
        Raw event positions (e.g. recoil energies) inside the analysis range.
    spectrum_cdf : callable, optional
        Monotonic CDF mapping ``events`` onto ``[0, 1]``.  Defaults to the
        identity (use this when ``events`` are already cumulants).

    Returns
    -------
    numpy.ndarray
        Sorted cumulants with 0 prepended and 1 appended.

    Raises
    ------
    ValueError
        If the resulting cumulants fall outside ``[0, 1]`` or are not
        non-decreasing -- i.e. ``spectrum_cdf`` is not a CDF normalized so the
        analysis range maps onto ``[0, 1]``.
    """
    e = np.sort(np.asarray(events, dtype=float))
    if spectrum_cdf is not None:
        e = np.asarray(spectrum_cdf(e), dtype=float)
    if e.size:
        tol = 1e-9
        if e[0] < -tol or e[-1] > 1.0 + tol:
            raise ValueError(
                f"cumulants must lie in [0, 1]; got [{e.min():.4g}, {e.max():.4g}]. "
                "Is spectrum_cdf normalized so the analysis range maps onto [0, 1]?"
            )
        if np.any(np.diff(e) < -tol):
            raise ValueError("cumulants must be non-decreasing (spectrum_cdf monotonic).")
        e = np.clip(e, 0.0, 1.0)  # tidy tiny float excursions before adding 0/1
    return np.concatenate(([0.0], e, [1.0]))
