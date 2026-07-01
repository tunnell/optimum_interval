"""Monte-Carlo tabulation and the optimum-interval upper-limit solver.

All state that used to live in module-level globals (``itvSizes``, ``optItvs``,
``mcTrials``) is now held on a single :class:`OptimumIntervalTable` instance.
Construct one, let it build tables on demand, optionally persist it to disk.

The physics: for a proposed signal with Poisson mean ``mu`` (and no
background), simulate many experiments, record for each the distribution of
k-largest interval sizes, and from those build the calibration distribution of
the C_max statistic.  An upper limit on ``mu`` is the value at which the
observed C_max sits at the requested confidence quantile of that distribution.
See ``EXPLANATION.md`` for the full derivation.
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import brenth

from .intervals import cumulant_points, k_largest_intervals

log = logging.getLogger(__name__)

DEFAULT_CACHE = "saved_intervals.p"

CdfType = Callable[[np.ndarray], np.ndarray] | None


class OptimumIntervalTable:
    """Monte-Carlo interval-size tables plus the upper-limit solver.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random generator.  Pass a seeded ``np.random.default_rng(seed)`` for
        reproducible tables.  Defaults to a fresh unseeded generator.

    Attributes
    ----------
    itv_sizes : dict[float, dict[int, numpy.ndarray]]
        ``itv_sizes[mu][k]`` is the array of k-largest interval sizes over all
        trials that contained at least ``k`` events.
    opt_itvs : dict[float, numpy.ndarray]
        ``opt_itvs[mu]`` is the per-trial C_max statistic (the calibration
        distribution whose quantile gives ``bar_c_max``).
    n_trials : dict[float, int]
        Number of Monte-Carlo trials generated for each ``mu``.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.itv_sizes: dict[float, dict[int, np.ndarray]] = {}
        self.opt_itvs: dict[float, np.ndarray] = {}
        self.n_trials: dict[float, int] = {}
        self.rng: np.random.Generator = (
            np.random.default_rng() if rng is None else rng
        )

    # ------------------------------------------------------------------ #
    # Monte-Carlo table construction
    # ------------------------------------------------------------------ #
    def generate_trials(self, mu: float, n: int) -> list[np.ndarray]:
        """Draw ``n`` background-free trial experiments at Poisson mean ``mu``.

        Each trial is a sorted point list on ``[0, 1]``: ``Poisson(mu)`` interior
        uniforms plus the range endpoints 0 and 1.  This mirrors exactly the
        cumulant-space representation of real data (:func:`cumulant_points`).
        """
        counts = self.rng.poisson(mu, size=n)
        trials = []
        for count in counts:
            interior = np.sort(self.rng.random(count))
            trials.append(np.concatenate(([0.0], interior, [1.0])))
        return trials

    def generate(self, mu: float, n: int) -> None:
        """Ensure tables for ``mu`` are built from at least ``n`` trials.

        Populates :attr:`itv_sizes`, :attr:`opt_itvs` and :attr:`n_trials`.
        Cached: returns immediately if ``mu`` already has ``>= n`` trials.
        """
        if self.n_trials.get(mu, 0) >= n:
            return

        log.info("generating optimum-interval table for mu=%s (n=%d)", mu, n)
        trials = self.generate_trials(mu, n)

        # k-largest sizes per trial, grouped by k with the originating trial
        # index kept so we can scatter the extremeness back per trial.
        sizes_by_k: dict[int, list[float]] = defaultdict(list)
        trials_by_k: dict[int, list[int]] = defaultdict(list)
        for t, trial in enumerate(trials):
            for k, size in k_largest_intervals(trial).items():
                sizes_by_k[k].append(size)
                trials_by_k[k].append(t)

        itv_sizes = {k: np.asarray(v) for k, v in sizes_by_k.items()}

        # C_max per trial = max over k of the empirical CDF (extremeness) of
        # that trial's k-largest size.  extremeness uses strict "<" and divides
        # by the total trial count n (Yellin's denominator convention: a trial
        # with too few events to have a k-largest interval does not count as
        # "smaller").  Vectorised per k via searchsorted, then scattered.
        opt = np.zeros(n)
        for k, obs in itv_sizes.items():
            reference = np.sort(obs)
            extremeness = np.searchsorted(reference, obs, side="left") / n
            np.maximum.at(opt, np.asarray(trials_by_k[k]), extremeness)

        self.itv_sizes[mu] = itv_sizes
        self.opt_itvs[mu] = opt
        self.n_trials[mu] = n

    # ------------------------------------------------------------------ #
    # Statistics that query the tables
    # ------------------------------------------------------------------ #
    def extremeness_of_interval(self, x: float, k: int, mu: float) -> float:
        """Fraction of MC k-largest intervals at ``mu`` smaller than ``x``.

        This is the code's empirical stand-in for :math:`C_k(x, \\mu)`.  Returns
        0 if no trial produced a k-largest interval (missing ``k``).
        """
        reference = self.itv_sizes[mu].get(k)
        if reference is None:
            return 0.0
        return float(np.count_nonzero(reference < x) / self.n_trials[mu])

    def optimum_interval_statistic(
        self, events: np.ndarray, mu: float, spectrum_cdf: CdfType = None
    ) -> float:
        """C_max of a run: the most extreme k-largest interval.

        ``events`` are raw positions; they are transformed to cumulant space and
        the range endpoints 0 and 1 are added (:func:`cumulant_points`) so the
        calculation matches the Monte-Carlo calibration.
        """
        points = cumulant_points(events, spectrum_cdf)
        sizes = k_largest_intervals(points)  # already in cumulant space
        return max(
            self.extremeness_of_interval(size, k, mu) for k, size in sizes.items()
        )

    def extremeness_of_opt_itv_stat(self, stat: float, mu: float) -> float:
        """Fraction of MC C_max values at ``mu`` smaller than ``stat``.

        The confidence quantile of this distribution is ``bar_c_max``.
        """
        opt = self.opt_itvs[mu]
        return float(np.count_nonzero(opt < stat) / self.n_trials[mu])

    def bar_c_max(self, confidence: float, mu: float) -> float:
        """Threshold :math:`\\bar C_\\mathrm{max}(C, \\mu)` (Yellin Fig. 2).

        The value of C_max reached with probability ``confidence`` under the
        background-free hypothesis: the ``confidence`` quantile of
        :attr:`opt_itvs`.  ``generate`` must have been called for ``mu`` first.
        """
        return float(np.quantile(self.opt_itvs[mu], confidence))

    # ------------------------------------------------------------------ #
    # Upper limit
    # ------------------------------------------------------------------ #
    def upper_limit(
        self,
        events: np.ndarray,
        confidence: float = 0.9,
        spectrum_cdf: CdfType = None,
        n: int = 1000,
        *,
        mu_scan_start: int = 10,
        mu_scan_stop: int | None = None,
        bracket: float = 5.0,
        xtol: float = 1e-2,
    ) -> float:
        """``confidence``-level optimum-interval upper limit on ``mu``.

        Finds ``mu`` such that ``extremeness_of_opt_itv_stat(C_max(events, mu),
        mu) == confidence`` -- i.e. where the observed C_max equals
        :math:`\\bar C_\\mathrm{max}(\\text{confidence}, \\mu)`.

        Parameters
        ----------
        events : array_like
            Observed event positions (raw; transformed via ``spectrum_cdf``).
        confidence : float, optional
            Confidence level, e.g. ``0.9``.
        spectrum_cdf : callable, optional
            Signal CDF mapping ``events`` to cumulants.  Identity by default.
        n : int, optional
            Monte-Carlo trials per ``mu``.
        mu_scan_start, mu_scan_stop : int, optional
            Integer bracket-search range.  ``mu_scan_stop`` defaults to
            ``2 * n_events``.
        bracket : float, optional
            Half-width of the ``brenth`` bracket around the seed ``mu``.
        xtol : float, optional
            Absolute tolerance on ``mu`` for the root find.

        Returns
        -------
        float
            The upper limit on ``mu``.  Falls back to the seed ``mu`` if the
            root find fails to converge.
        """
        events = np.asarray(events, dtype=float)

        def excess(mu: float) -> float:
            self.generate(mu, n)
            stat = self.optimum_interval_statistic(events, mu, spectrum_cdf)
            return self.extremeness_of_opt_itv_stat(stat, mu) - confidence

        stop = 2 * events.size if mu_scan_stop is None else mu_scan_stop
        seed = mu_scan_start
        for seed in np.arange(mu_scan_start, stop):
            if excess(seed) > 0:
                log.info("bracket seed mu=%s", seed)
                break

        try:
            limit = brenth(
                lambda mu: excess(mu), seed - bracket, seed + bracket, xtol=xtol
            )
        except ValueError:
            log.warning("root find failed near mu=%s; returning seed", seed)
            return float(seed)
        return float(limit)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path = DEFAULT_CACHE) -> None:
        """Pickle the tables to ``path`` (single dict, one payload)."""
        payload = {
            "itv_sizes": self.itv_sizes,
            "opt_itvs": self.opt_itvs,
            "n_trials": self.n_trials,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_CACHE, rng: np.random.Generator | None = None
    ) -> "OptimumIntervalTable":
        """Load tables previously written by :meth:`save`."""
        table = cls(rng=rng)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        table.itv_sizes = payload["itv_sizes"]
        table.opt_itvs = payload["opt_itvs"]
        table.n_trials = payload["n_trials"]
        return table
