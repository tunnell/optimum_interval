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
The Monte-Carlo code works in normalized cumulant space where interval sizes
are fractions of ``[0, 1]``.  Yellin's :math:`x` is in *expected events*.  A
normalized gap fraction ``f`` corresponds to ``x = mu * f`` expected events, so
``c0(mu * f, mu)`` is what should equal the empirical CDF of the MC gap
fractions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaincinv, gammaln

__all__ = ["c0", "max_gap_upper_limit", "poisson_upper_limit", "x0"]


# A term whose magnitude exceeds exp(_LOG_HUGE) ~ 1e13 means the alternating
# series is losing > 13 significant digits to cancellation.  That happens only
# deep in the C_0 ~ 0 tail (the requested gap x is far below the typical maximum
# gap), where the true value rounds to 0 -- so we return 0.0 there rather than
# form a garbage (or overflowing) sum.  This replaces a former fixed m-cap that
# wrongly zeroed valid large-mu evaluations.
_LOG_HUGE = 30.0


def _c0_scalar(x: float, mu: float) -> float:
    r"""``c0`` for scalar ``x`` (see :func:`c0`).

    Evaluates Yellin Eq. 2 in its telescoped, division-free form

        term_k = e^{-k x} / k! * base^{k-1} * (base - k),   base = k x - mu,

    working in log-magnitude so individual terms never overflow.  For ``k > mu``
    the terms decay monotonically and the sum is truncated once negligible; if a
    term ever exceeds ~1e13 in magnitude we are in the ``C_0 ~ 0`` tail (see
    ``_LOG_HUGE``) and return 0.
    """
    if x <= 0.0:
        return 0.0
    if x > mu:
        # m = 0: the max gap (at most mu) is certainly smaller than x > mu.
        return 1.0
    # x == mu falls through: m = 1 gives C_0(mu, mu) = 1 - e^{-mu}, the
    # probability of >= 1 event (a zero-event run has its max gap equal to the
    # whole range mu, which is NOT smaller than x = mu).

    m = int(np.floor(mu / x))
    total = 1.0  # k = 0 term
    for k in range(1, m + 1):
        base = k * x - mu  # <= 0 for k <= m
        if base == 0.0:
            # k x == mu: base^k - k base^{k-1} is -1 for k == 1, else 0.
            term = -np.exp(-k * x - gammaln(k + 1)) if k == 1 else 0.0
        else:
            log_mag = (
                -k * x
                - gammaln(k + 1)
                + (k - 1) * np.log(abs(base))
                + np.log(abs(base - k))
            )
            if log_mag > _LOG_HUGE:
                return 0.0  # C_0 ~ 0 tail; see _LOG_HUGE note
            # base <= 0 and base - k < 0, so the sign is (-1)^k.
            sign = -1.0 if k % 2 else 1.0
            term = sign * np.exp(log_mag)
        total += term
        if k > mu and abs(term) < 1e-15:
            break  # monotone tail; remaining terms are negligible
    return float(np.clip(total, 0.0, 1.0))  # clip tiny round-off past [0, 1]


# Vectorized over x (and mu) while keeping the clear scalar core above.
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

    Raises
    ------
    ValueError
        If ``confidence`` exceeds the largest attainable ``C_0``, which is
        ``C_0(mu, mu) = 1 - e^{-mu}``.  For a 90% level this happens when
        ``mu < 2.3026`` -- exactly the regime where no 90% CL exists (Appendix B).
    """
    c_max = _c0_scalar(mu, mu)  # = 1 - e^{-mu}, the largest attainable C_0
    if confidence >= c_max:
        raise ValueError(
            f"C_0 cannot reach {confidence} for mu={mu}: its maximum is "
            f"{c_max:.4f} at x=mu. (A 90% max-gap limit requires mu > 2.3026.)"
        )
    # c0 is 0 at x->0 and c_max at x=mu, so the root is bracketed by (tiny, mu).
    return float(brentq(lambda x: _c0_scalar(x, mu) - confidence, 1e-12, mu))


def poisson_upper_limit(n_observed: int, confidence: float = 0.9) -> float:
    r"""Classical Poisson upper limit on the mean given ``n_observed`` events.

    The standard total-count limit: ``mu_up`` is the mean for which a fraction
    ``confidence`` of experiments would see *more* than ``n_observed`` events,
    i.e. :math:`\sum_{k=0}^{n} e^{-\mu_\text{up}}\mu_\text{up}^k/k! = 1-C`.  This
    is the inverse incomplete Gamma function ``gammaincinv(n+1, confidence)``.
    (Used only for the method-comparison figures; e.g. ``n=0`` gives ``2.3026``.)
    """
    return float(gammaincinv(n_observed + 1, confidence))


def max_gap_upper_limit(max_gap_fraction: float, confidence: float = 0.9) -> float:
    r"""Maximum-gap (:math:`C_0`) upper limit on ``mu`` for one experiment.

    Solve :math:`C_0(\mu\,f, \mu) = C` for ``mu``, where ``f`` is the observed
    maximum-gap size as a *fraction* of the range (its expected-event size is
    ``mu * f``).  ``C_0`` increases with ``mu`` here, so the root is unique.

    Parameters
    ----------
    max_gap_fraction : float
        Largest gap between adjacent events (including the range endpoints) in
        cumulant space, in ``[0, 1]``.
    confidence : float, optional
        Confidence level, e.g. ``0.9``.
    """
    f = float(max_gap_fraction)
    if not 0.0 < f <= 1.0:
        raise ValueError(f"max_gap_fraction must be in (0, 1]; got {f}")

    def g(mu: float) -> float:
        return _c0_scalar(mu * f, mu) - confidence

    hi = max(1.0, 1.0 / f)
    while g(hi) < 0.0 and hi < 1e6:
        hi *= 2.0
    return float(brentq(g, 1e-9, hi))
