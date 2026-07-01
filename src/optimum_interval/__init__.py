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
``c0`` / ``x0``
    Analytic max-gap statistic, Yellin Eq. 2 (:mod:`optimum_interval.analytic`).
``OptimumIntervalTable``
    Monte-Carlo tables and the upper-limit solver
    (:mod:`optimum_interval.montecarlo`).
"""

from __future__ import annotations

from .analytic import c0, max_gap_upper_limit, poisson_upper_limit, x0
from .comparison import ComparisonEngine
from .intervals import cumulant_points, k_largest_intervals
from .montecarlo import DEFAULT_CACHE, OptimumIntervalTable

__all__ = [
    "k_largest_intervals",
    "cumulant_points",
    "c0",
    "x0",
    "poisson_upper_limit",
    "max_gap_upper_limit",
    "OptimumIntervalTable",
    "ComparisonEngine",
    "DEFAULT_CACHE",
]

__version__ = "0.1.0"
