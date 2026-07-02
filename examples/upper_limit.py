#!/usr/bin/env python3
"""How many signal events can you exclude? A generic upper-limit walkthrough.

Runnable companion to ``TUTORIAL.md`` -- no dark-matter specifics. You have
events measured along some observable ``x``, you know the signal's *shape*, and
you want a 90% CL upper limit on the expected number of signal events ``mu``,
using only that shape (no background model, no binning).

For turning ``mu`` into a rate / cross section, see
``examples/dark_matter_exclusion.py``.

Run:  python examples/upper_limit.py
"""

from __future__ import annotations

import numpy as np

from optimum_interval import (
    OptimumIntervalTable,
    cumulant_points,
    max_gap_upper_limit,
    spectrum_cdf_from_pdf,
)


def limits(events, table, *, spectrum_cdf=None, confidence=0.9, n=5000):
    """Max-gap (analytic) and optimum-interval (MC) upper limits on ``mu``.

    Omit ``spectrum_cdf`` when ``events`` already live on ``[0, 1]`` (uniform
    signal, or you applied the CDF yourself).
    """
    max_gap = float(np.diff(cumulant_points(events, spectrum_cdf)).max())
    return {
        "max_gap": max_gap_upper_limit(max_gap, confidence),
        "optimum_interval": table.upper_limit(
            events, spectrum_cdf=spectrum_cdf, confidence=confidence, n=n
        ),
    }


def run(*, n=5000, rng=None):
    """Compute limits for two illustrative cases; return a results dict."""
    rng = np.random.default_rng(0) if rng is None else rng
    table = OptimumIntervalTable(rng=rng)

    # (a) Signal uniform in the observable: events already on [0, 1].
    uniform_events = np.array([0.08, 0.11, 0.55, 0.58, 0.62, 0.95])
    uniform = limits(uniform_events, table, n=n)

    # (b) Signal with a known falling shape on [0, 10] (any observable/units).
    cdf = spectrum_cdf_from_pdf(lambda x: np.exp(-x / 3.0), 0.0, 10.0)
    shaped_events = np.array([0.3, 0.5, 0.9, 1.4, 6.0, 8.5])
    shaped = limits(shaped_events, table, spectrum_cdf=cdf, n=n)

    return {"uniform": uniform, "shaped": shaped}


def main():
    print("Upper limit on the expected number of signal events (mu)\n")
    for name, lim in run().items():
        print(
            f"{name:>8} signal:  mu_UL(max gap) = {lim['max_gap']:.2f}   "
            f"mu_UL(optimum interval) = {lim['optimum_interval']:.2f}"
        )
    print(
        "\nSee TUTORIAL.md; for rate/cross-section conversion, dark_matter_exclusion.py."
    )


if __name__ == "__main__":
    main()
