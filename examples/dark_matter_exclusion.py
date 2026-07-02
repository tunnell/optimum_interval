#!/usr/bin/env python3
"""Applied example: a dark-matter cross-section exclusion curve.

A dedicated, domain-specific companion to the generic ``TUTORIAL.md`` /
``examples/upper_limit.py`` (which cover the core "upper limit on the number of
events" workflow).  Here we take that limit on ``mu`` all the way to a
cross-section exclusion curve:

1. build a *normalized* recoil-spectrum CDF (``spectrum_cdf``),
2. transform observed energies into cumulant space,
3. get an upper limit on the expected signal count ``mu`` two ways
   (analytic max gap, and Monte-Carlo optimum interval),
4. convert ``mu`` limits into cross-section limits and sweep over "mass" to
   trace a toy exclusion curve.

The astrophysics here is a deliberately simple toy (an exponential recoil
spectrum, a made-up counts-per-cross-section); a real analysis would plug in a
package like ``wimprates``. The *statistics* is the real thing.

Run:  python examples/dark_matter_exclusion.py
"""

from __future__ import annotations

import numpy as np

from optimum_interval import (
    OptimumIntervalTable,
    cumulant_points,
    max_gap_upper_limit,
)


def exponential_spectrum_cdf(e0: float, e_min: float, e_max: float):
    """Return a normalized CDF for dN/dE ∝ exp(-E/e0) on [e_min, e_max].

    "Normalized" is the key requirement (see TUTORIAL.md): the CDF must map the
    analysis window onto exactly [0, 1] -- ``cdf(e_min) == 0``, ``cdf(e_max) == 1``.
    """
    lo = np.exp(-e_min / e0)
    norm = lo - np.exp(-e_max / e0)

    def cdf(energies):
        energies = np.clip(np.asarray(energies, dtype=float), e_min, e_max)
        return (lo - np.exp(-energies / e0)) / norm

    return cdf


def limits_for_dataset(energies, spectrum_cdf, table, *, confidence=0.9, n=2000):
    """Max-gap (analytic) and optimum-interval (Monte-Carlo) upper limits on mu."""
    max_gap_fraction = np.diff(cumulant_points(energies, spectrum_cdf)).max()
    mu_maxgap = max_gap_upper_limit(max_gap_fraction, confidence)
    mu_optint = table.upper_limit(
        energies, confidence=confidence, spectrum_cdf=spectrum_cdf, n=n
    )
    return mu_maxgap, mu_optint


def run_exclusion(*, n=2000, n_masses=8, make_plot=False, rng=None):
    """Run the whole workflow and return a results dict."""
    rng = np.random.default_rng(0) if rng is None else rng
    e_min, e_max = 1.0, 40.0  # keV analysis window

    # A toy "observed" dataset: a handful of energies, clustered low (as an
    # unknown background would be) plus a couple higher up.
    energies = np.array([2.1, 2.4, 3.0, 3.8, 5.2, 6.0, 22.0, 31.0])

    # One Monte-Carlo calibration table is reused across every mass, because the
    # calibration lives in cumulant space and is spectrum-independent.
    table = OptimumIntervalTable(rng=rng)

    # Sweep a toy "WIMP mass": heavier -> harder recoil spectrum (larger e0).
    masses = np.linspace(10.0, 100.0, n_masses)

    def e0_of_mass(m):
        return 3.0 + 0.08 * m  # keV; toy mass -> spectrum-hardness map

    def counts_per_cross_section(m):
        # Toy expected counts per unit cross section (exposure x rate). Real
        # analyses compute this from astrophysics x form factor x exposure.
        return 40.0 * np.exp(-((m - 25.0) ** 2) / 1500.0) + 2.0

    rows = []
    for m in masses:
        cdf = exponential_spectrum_cdf(e0_of_mass(m), e_min, e_max)
        mu_mg, mu_oi = limits_for_dataset(energies, cdf, table, n=n)
        mu1 = counts_per_cross_section(m)
        rows.append(
            dict(
                mass=m,
                mu_maxgap=mu_mg,
                mu_optint=mu_oi,
                sigma_maxgap=mu_mg / mu1,
                sigma_optint=mu_oi / mu1,
            )
        )

    if make_plot:
        _plot_exclusion(rows)

    return dict(energies=energies, rows=rows)


def _plot_exclusion(rows):
    import matplotlib.pyplot as plt

    m = [r["mass"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(m, [r["sigma_optint"] for r in rows], "C0-o", label="optimum interval")
    ax.plot(m, [r["sigma_maxgap"] for r in rows], "C1--s", label="max gap")
    ax.set_yscale("log")
    ax.set_xlabel("toy WIMP mass  [GeV]")
    ax.set_ylabel(r"90% CL upper limit on $\sigma$  [arb. units]")
    ax.set_title("Toy exclusion curve (optimum interval vs max gap)")
    ax.legend()
    fig.tight_layout()
    from pathlib import Path

    out = Path(__file__).resolve().parent.parent / "figures" / "dm_exclusion.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("Dark-matter exclusion: recoil spectrum -> mu limit -> sigma(mass)\n")
    result = run_exclusion(n=2000, n_masses=8, make_plot=True)
    print(f"observed energies (keV): {result['energies']}\n")
    print(
        f"{'mass[GeV]':>9} {'mu_UL(maxgap)':>13} {'mu_UL(optint)':>13} "
        f"{'sigma(optint)':>14}"
    )
    for r in result["rows"]:
        print(
            f"{r['mass']:>9.1f} {r['mu_maxgap']:>13.2f} {r['mu_optint']:>13.2f} "
            f"{r['sigma_optint']:>14.3f}"
        )
    print(
        "\nBoth limits used only the signal shape -- no background model -- and the "
        "same MC table served every mass. See TUTORIAL.md for the walkthrough."
    )


if __name__ == "__main__":
    main()
