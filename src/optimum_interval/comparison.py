"""Fast per-experiment upper limits for the $C_\\max$ and $p_\\max$ methods.

Reproducing Yellin's method-comparison figures (Fig. 3 & 4) needs an upper limit
for *every* one of many thousands of toy experiments.  Re-running a full Monte
Carlo per experiment (as :meth:`OptimumIntervalTable.upper_limit` does) would be
far too slow.  Instead, :class:`ComparisonEngine` precomputes the background-free
calibration **once** on a grid of ``mu`` -- the reference $k$-largest size
distributions and the $C_\\max$ / $p_\\max$ trial distributions -- and then each
experiment's limit is a cheap root-find of the extremeness against that grid.

This mirrors what Yellin's Fortran did with tabulated, interpolated functions.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import poisson

from .intervals import cumulant_points, k_largest_intervals

__all__ = ["ComparisonEngine"]


class ComparisonEngine:
    """Precomputed-grid ``C_max`` and ``p_max`` upper-limit solver.

    Parameters
    ----------
    mu_grid : array_like
        Grid of ``mu`` values over which the calibration is tabulated.  Must
        span the range in which the limits are expected to fall.
    n_cal : int, optional
        Monte-Carlo trials per grid point.
    rng : numpy.random.Generator, optional
        Seed for reproducibility.
    confidence : float, optional
        Confidence level (default 0.9).
    """

    def __init__(
        self,
        mu_grid,
        n_cal: int = 40000,
        rng: np.random.Generator | None = None,
        confidence: float = 0.9,
    ) -> None:
        self.mu_grid = np.sort(np.asarray(mu_grid, dtype=float))
        self.n_cal = int(n_cal)
        self.confidence = float(confidence)
        self.rng = np.random.default_rng() if rng is None else rng

        # Per grid mu: sorted k-largest reference sizes, and the sorted
        # C_max / p_max trial distributions used as calibration.
        self._ref: list[dict[int, np.ndarray]] = []
        self._opt: list[np.ndarray] = []
        self._pmax: list[np.ndarray] = []
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        n = self.n_cal
        for mu in self.mu_grid:
            counts = self.rng.poisson(mu, size=n)
            per_trial: list[dict[int, float]] = []
            sizes_by_k: dict[int, list[float]] = defaultdict(list)
            trials_by_k: dict[int, list[int]] = defaultdict(list)
            for t, count in enumerate(counts):
                pts = np.concatenate(([0.0], np.sort(self.rng.random(count)), [1.0]))
                sizes = k_largest_intervals(pts)
                per_trial.append(sizes)
                for k, size in sizes.items():
                    sizes_by_k[k].append(size)
                    trials_by_k[k].append(t)

            sorted_ref = {k: np.sort(v) for k, v in sizes_by_k.items()}

            # C_max per trial (empirical CDF of each k-largest, max over k).
            opt = np.zeros(n)
            for k, values in sizes_by_k.items():
                extremeness = np.searchsorted(sorted_ref[k], values, side="left") / n
                np.maximum.at(opt, np.asarray(trials_by_k[k]), extremeness)

            # p_max per trial: max over k of Poisson P(>k | mu * size).
            pmax = np.empty(n)
            for t, sizes in enumerate(per_trial):
                ks = np.fromiter(sizes.keys(), dtype=int)
                xs = mu * np.fromiter(sizes.values(), dtype=float)
                pmax[t] = poisson.sf(ks, xs).max()

            self._ref.append(sorted_ref)
            self._opt.append(np.sort(opt))
            self._pmax.append(np.sort(pmax))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sizes_array(events, spectrum_cdf) -> np.ndarray:
        """Observed k-largest sizes as a dense array indexed by k."""
        sizes = k_largest_intervals(cumulant_points(events, spectrum_cdf))
        return np.array([sizes[k] for k in range(len(sizes))])

    def _crossing(self, g: np.ndarray) -> float:
        """First ``mu`` on the grid where increasing curve ``g`` reaches confidence.

        Linear interpolation between grid points.  Clamps (with no silent
        surprise -- the grid should be chosen to contain the limit) to the grid
        ends if the crossing lies outside.
        """
        c = self.confidence
        grid = self.mu_grid
        above = g >= c
        if not above.any():
            return float(grid[-1])  # limit above grid: clamp (widen mu_grid)
        i = int(np.argmax(above))
        if i == 0:
            return float(grid[0])  # limit at/below grid start
        y0, y1 = g[i - 1], g[i]
        if y1 == y0:
            return float(grid[i])
        return float(grid[i - 1] + (c - y0) * (grid[i] - grid[i - 1]) / (y1 - y0))

    # ------------------------------------------------------------------ #
    def cmax_upper_limit(self, events, spectrum_cdf=None) -> float:
        """Optimum-interval (``C_max``) upper limit on ``mu`` for one experiment."""
        sizes = self._sizes_array(events, spectrum_cdf)
        g = np.empty(self.mu_grid.size)
        for j in range(self.mu_grid.size):
            ref = self._ref[j]
            cmax = 0.0
            for k in range(min(sizes.size, max(ref) + 1 if ref else 0)):
                arr = ref.get(k)
                if arr is not None:
                    e = np.searchsorted(arr, sizes[k], side="left") / self.n_cal
                    if e > cmax:
                        cmax = e
            g[j] = np.searchsorted(self._opt[j], cmax, side="left") / self.n_cal
        return self._crossing(g)

    def pmax_upper_limit(self, events, spectrum_cdf=None) -> float:
        """``p_max`` (Poisson-probability) upper limit on ``mu`` for one experiment."""
        sizes = self._sizes_array(events, spectrum_cdf)
        ks = np.arange(sizes.size)
        g = np.empty(self.mu_grid.size)
        for j, mu in enumerate(self.mu_grid):
            pmax = poisson.sf(ks, mu * sizes).max()
            g[j] = np.searchsorted(self._pmax[j], pmax, side="left") / self.n_cal
        return self._crossing(g)
