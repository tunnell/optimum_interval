r"""Closed-form maximum-gap statistic :math:`C_0` (Yellin Eq. 2).

For the special case of the maximum gap (intervals with **zero** events) the
confidence level has an exact analytic form and needs no Monte Carlo.  This is
Eq. (2) of Yellin (2002):

.. math::

    C_0(x, \mu) = \sum_{k=0}^{m}
        \frac{(k x - \mu)^k \, e^{-k x}}{k!}
        \left(1 + \frac{k}{\mu - k x}\right),
    \qquad m = \lfloor \mu / x \rfloor,

where :math:`x` is the maximum-gap size measured in *expected events* and
:math:`\mu` is the total expected number of events over the whole range.
:math:`C_0(x, \mu)` is the probability that a background-free experiment with
Poisson mean :math:`\mu` has a maximum gap **smaller** than :math:`x`.

We use this as the ground-truth validator for the ``k = 0`` Monte Carlo (see
``reproduce_figures.py`` and ``tests/test_montecarlo.py``).

Units note
----------
The Monte-Carlo code works in normalised cumulant space where interval sizes
are fractions of ``[0, 1]``.  Yellin's :math:`x` is in *expected events*.  A
normalised gap fraction ``f`` corresponds to ``x = mu * f`` expected events, so
``c0(mu * f, mu)`` is what should equal the empirical CDF of the MC gap
fractions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

__all__ = ["c0", "x0"]


# Above this many terms the maximum gap is far smaller than the typical gap, so
# C_0 underflows to 0; the alternating series would also lose precision / overflow.
# All physically relevant evaluations (C_0 anywhere near 0.9) have m of order a
# few -- the paper notes the series truncates at small k.
_M_CAP = 60


def _c0_scalar(x: float, mu: float) -> float:
    """``c0`` for scalar ``x`` (see :func:`c0`)."""
    if x <= 0.0:
        return 0.0
    if x >= mu:
        # m = 0: only the k = 0 term survives and equals 1.
        return 1.0

    m = int(np.floor(mu / x))
    if m > _M_CAP:
        # x is far below the typical gap size => essentially impossible for the
        # maximum gap to be this small => C_0 ~ 0 (see _M_CAP note above).
        return 0.0

    # Telescoped, division-free form of Eq. (2).  The paper's factor
    # (1 + k/(mu - k x)) is singular at mu = k x; multiplying it in gives the
    # algebraically identical but finite expression
    #     term_k = e^{-k x}/k! * [ (k x - mu)^k - k (k x - mu)^{k-1} ].
    total = 1.0  # k = 0 term
    for k in range(1, m + 1):
        base = k * x - mu
        log_prefactor = -k * x - gammaln(k + 1)  # e^{-k x} / k!
        bracket = base**k - k * base ** (k - 1)
        total += np.exp(log_prefactor) * bracket
    return float(np.clip(total, 0.0, 1.0))  # absorb tiny round-off past [0, 1]


# Vectorised over x (and mu) while keeping the clear scalar core above.
_c0_vec = np.vectorize(_c0_scalar, otypes=[float])


def c0(x, mu):
    r"""Probability the maximum gap is smaller than ``x`` (Yellin Eq. 2).

    Parameters
    ----------
    x : float or array_like
        Maximum-gap size in **expected events** (``0 <= x <= mu``).
    mu : float
        Total expected number of events over the analysis range.

    Returns
    -------
    float or numpy.ndarray
        :math:`C_0(x, \mu) \in [0, 1]`, monotonically increasing in ``x``.
    """
    result = _c0_vec(x, mu)
    return float(result) if np.ndim(x) == 0 and np.ndim(mu) == 0 else result


def x0(confidence: float, mu: float) -> float:
    r"""Invert :func:`c0`: the gap size at which ``c0(x, mu) == confidence``.

    This is Yellin's :math:`x_0(C, \mu)`.  For a max-gap 90% upper limit one
    raises ``mu`` until the observed gap equals ``x0(0.9, mu)``.

    Parameters
    ----------
    confidence : float
        Target confidence level, e.g. ``0.9``.
    mu : float
        Total expected number of events.

    Returns
    -------
    float
        The gap size ``x`` (in expected events) solving ``c0(x, mu) = confidence``.
    """
    # c0 is 0 at x->0 and 1 at x=mu, so a root is bracketed by (tiny, mu).
    return float(brentq(lambda x: _c0_scalar(x, mu) - confidence, 1e-12, mu))
