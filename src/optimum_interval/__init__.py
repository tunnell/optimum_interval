"""Optimum-interval upper limits in the presence of unknown background.

A small, dependency-light implementation of the method of S. Yellin,
*"Finding an Upper Limit in the Presence of Unknown Background"*,
Phys. Rev. **D66** (2002) 032005 (arXiv:physics/0203002).

The method sets a frequentist upper limit on a signal normalization (the
Poisson mean ``mu`` of expected events) using only the known signal *shape*,
without a model of the background and without binning.  See ``EXPLANATION.md``
for a physicist-oriented walk-through and derivation.

Quick start
-----------
>>> import numpy as np
>>> from optimum_interval import OptimumIntervalTable
>>> rng = np.random.default_rng(0)
>>> table = OptimumIntervalTable(rng=rng)
>>> events = np.sort(rng.random(8))          # 8 events, already in cumulant space
>>> mu_limit = table.upper_limit(events, confidence=0.9, n=2000)  # doctest: +SKIP

Public API
----------
``k_largest_intervals`` / ``cumulant_points``
    Pure interval geometry (:mod:`optimum_interval.intervals`).
``c0`` / ``x0`` / ``max_gap_upper_limit`` / ``poisson_upper_limit``
    Analytic statistics: Yellin Eq. 2 and the counting limits
    (:mod:`optimum_interval.analytic`).
``OptimumIntervalTable``
    Monte-Carlo calibration tables and the upper-limit solver
    (:mod:`optimum_interval.montecarlo`).
``spectrum_cdf_from_pdf`` / ``spectrum_cdf_from_samples``
    Build a normalized ``spectrum_cdf`` (:mod:`optimum_interval.spectra`).
``ComparisonEngine``
    Fast per-experiment limits for method comparisons
    (:mod:`optimum_interval.comparison`).
"""

from __future__ import annotations

from .analytic import c0, max_gap_upper_limit, poisson_upper_limit, x0
from .comparison import ComparisonEngine
from .intervals import cumulant_points, k_largest_intervals
from .montecarlo import OptimumIntervalTable
from .spectra import spectrum_cdf_from_pdf, spectrum_cdf_from_samples

__all__ = [
    "ComparisonEngine",
    "OptimumIntervalTable",
    "c0",
    "cumulant_points",
    "k_largest_intervals",
    "max_gap_upper_limit",
    "poisson_upper_limit",
    "spectrum_cdf_from_pdf",
    "spectrum_cdf_from_samples",
    "x0",
]

__version__ = "0.2.0"
