"""Parameter scans when the spectrum shape depends on the parameter.

``OptimumIntervalTable.upper_limit`` solves for the signal normalization
directly, which requires the spectrum *shape* to be fixed while ``mu`` varies.
When the shape itself moves with the parameter being limited (a finite-range
mediator, overburden attenuation, a velocity-dependent coupling, ...) that
factorization breaks, and the limit is instead a level set of the extremeness
surface: scan the parameter(s), evaluate :func:`scan_extremeness` at each grid
point, and take the confidence-level crossing (:func:`excluded_interval` for a
1-D scan, a contour for a 2-D grid).

Two ingredients keep large scans cheap. Expected counts are rounded onto a
fine log grid (:func:`round_log`) so a single Monte-Carlo calibration table
serves every grid point with the same rounded ``mu``; and grid points with
``mu`` below/above configurable bounds skip the Monte Carlo entirely (nothing
expected, or overwhelmingly excluded).
"""

from __future__ import annotations

import numpy as np

from .montecarlo import OptimumIntervalTable
from .spectra import spectrum_cdf_from_samples

__all__ = [
    "excluded_interval",
    "new_table",
    "round_log",
    "scan_extremeness",
    "spectrum_from_rate",
]


def new_table(seed=0):
    """A seeded Monte-Carlo calibration table (reusable across a whole scan)."""
    return OptimumIntervalTable(rng=np.random.default_rng(seed))


def round_log(x, dex=0.02):
    """Round onto a ``dex``-spaced log grid so tables are shared between points."""
    return 10 ** (np.round(np.log10(x) / dex) * dex)


def spectrum_from_rate(x, rate, exposure):
    """Expected counts and normalized spectrum CDF from a differential rate.

    Parameters
    ----------
    x, rate : arrays
        Differential rate ``dR/dx`` (per unit time per unit ``x``) sampled on
        the grid ``x``. Trailing zero-rate points are trimmed, so the support
        window ends at the last point with positive rate.
    exposure : float
        Livetime in the inverse time unit of ``rate``.

    Returns
    -------
    (mu, cdf, x_lo, x_hi), or ``None`` if the rate has no support.
    """
    x = np.asarray(x, dtype=float)
    rate = np.maximum(np.asarray(rate, dtype=float), 0.0)
    if not np.any(rate > 0):
        return None
    hi = np.max(np.where(rate > 0)[0])
    x, rate = x[: hi + 1], rate[: hi + 1]
    mu = float(np.trapezoid(rate, x)) * exposure
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(x))])
    if cum[-1] <= 0:
        return None
    cdf = spectrum_cdf_from_samples(x, cum / cum[-1])
    return mu, cdf, x[0], x[-1]


def scan_extremeness(table, events, x, rate, exposure,
                     n=2500, mu_floor=0.2, mu_cap=40.0, mu_dex=0.02):
    """Optimum-interval extremeness of ``events`` for one scan point's spectrum.

    Returns ``(p, mu)`` where ``p`` is the probability that a background-free
    pseudo-experiment under this hypothesis looks *less* extreme than the
    data (so the limit is where ``p`` crosses the confidence level), and
    ``mu`` the expected counts. Events outside the spectrum's support window
    are dropped automatically: they cannot be signal at this scan point.

    Shortcuts: ``mu < mu_floor`` returns ``p = 0`` (nothing expected);
    ``mu > mu_cap`` returns ``p = 1`` (with few observed events the exclusion
    is overwhelming; no Monte Carlo needed — raise the cap if you scan with
    many events). ``mu`` is rounded onto a ``mu_dex`` log grid so calibration
    tables are reused across the scan.
    """
    spec = spectrum_from_rate(x, rate, exposure)
    if spec is None:
        return 0.0, 0.0
    mu, cdf, x_lo, x_hi = spec
    if mu < mu_floor:
        return 0.0, mu
    if mu > mu_cap:
        return 1.0, mu
    inside = np.asarray(events, dtype=float)
    inside = inside[(inside > x_lo) & (inside < x_hi)]
    mu_r = round_log(mu, mu_dex)
    table.generate(mu_r, n)
    stat = table.optimum_interval_statistic(inside, mu_r, spectrum_cdf=cdf)
    return table.extremeness_of_opt_itv_stat(stat, mu_r), mu


def excluded_interval(params, extremeness, level=0.95):
    """(low, high) edges of the excluded interval along a 1-D parameter scan.

    ``extremeness`` is the :func:`scan_extremeness` value at each of the
    (positive, increasing) ``params``; edges are log-interpolated crossings
    of ``level``. Returns ``(nan, nan)`` when nothing is excluded; an edge
    saturates at the end of the scan if the crossing lies beyond it.
    """
    params = np.asarray(params, dtype=float)
    ps = np.asarray(extremeness, dtype=float)
    above = ps >= level
    if not above.any():
        return np.nan, np.nan
    idx = np.where(above)[0]
    if idx[0] > 0:
        lo = np.interp(level, ps[idx[0] - 1: idx[0] + 1],
                       np.log10(params[idx[0] - 1: idx[0] + 1]))
    else:
        lo = np.log10(params[0])
    if idx[-1] < len(params) - 1:
        hi = np.interp(-level, -ps[idx[-1]: idx[-1] + 2],
                       np.log10(params[idx[-1]: idx[-1] + 2]))
    else:
        hi = np.log10(params[-1])
    return 10 ** lo, 10 ** hi
