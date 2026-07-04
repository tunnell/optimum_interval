#!/usr/bin/env python3
"""Applied example: ultraheavy dark matter via momentum kicks on a levitated sensor.

A levitated-sensor experiment (e.g. a Meissner-levitated magnet) searches for
ultraheavy dark matter (UHDM) coupled to neutron number through a long-range
Yukawa "fifth force".  A DM particle flying past delivers a single impulse, and
each candidate event has one measured observable: the momentum transfer ``q``.
That makes it an optimum-interval problem:

- observable per event: kick momentum ``q`` on ``[q_th, q_max(m_DM)]``;
- known signal *shape*: for a massless mediator the differential rate is
  ``dR/dq ∝ q^-3 [exp(-(q/2 m v0)^2) - exp(-(v_esc/v0)^2)]``
  (halo velocity integral over the classical Coulomb-like cross section);
- unknown normalization: the rate scales as ``(alpha_n N_n)^2`` — *quadratic*
  in the coupling per neutron ``alpha_n``;
- unknown background: vibrational/seismic transients cluster just above
  threshold and cannot be modeled to subtraction precision.

The common "zero observed events => <N> = 3 at 95% CL" limit is the zero-event
special case of the optimum interval (mu_UL = -ln 0.05 ~ 3.0).  With candidate
kicks present, the optimum interval keeps using the empty high-``q`` region and
degrades gracefully, where plain Poisson counting takes the full hit.

All experimental parameters here are illustrative fiducials.  See
``TUTORIAL_UHDM.md`` for the walkthrough.

Run:  python examples/uhdm_momentum_kicks.py
"""

from __future__ import annotations

import numpy as np

from optimum_interval import (
    OptimumIntervalTable,
    poisson_upper_limit,
    spectrum_cdf_from_samples,
)

# --- constants and fiducial experiment parameters (natural units: GeV, c = 1)
C_KM_S = 299792.458
V0 = 220.0 / C_KM_S  # halo velocity dispersion parameter
VESC = 544.0 / C_KM_S  # galactic escape velocity
RHO_DM = 2.3e-42  # total local DM density, 0.3 GeV/cm^3 in GeV^4
F_X = 0.1  # fraction of local DM in this species (evades self-interaction constraints)
GEV_PER_S = 1.519e24  # 1 GeV of rate = 1.519e24 s^-1  (hbar = 1)

Q_TH = 8400.0  # momentum threshold [GeV]  (~4.5e-15 kg m/s)
N_NEUTRONS = 1e18  # neutrons in the sensor (fiducial)
T_OBS_S = 30 * 86400.0  # 30 days of livetime

# Toy candidate kicks [GeV]: transients piled just above threshold + one outlier.
KICKS = np.array([8600.0, 8900.0, 9300.0, 9800.0, 30000.0])

CONFIDENCE = 0.95  # the community convention for these searches


def q_max(m_dm: float) -> float:
    """Largest kinematically allowed kick: q_max = 2 mu v_esc, with mu ~ m_DM."""
    return 2.0 * m_dm * VESC


MASS_FLOOR = Q_TH / (2.0 * VESC)  # below this, no kick can top the threshold


def drdq_alpha1(q, m_dm):
    """Differential rate dR/dq at unit total coupling (alpha = alpha_n N_n = 1).

    Massless-mediator (Coulomb-like) scattering integrated over a simplified
    Maxwell-Boltzmann halo; natural units throughout.
    """
    q = np.asarray(q, dtype=float)
    n_dm = F_X * RHO_DM / m_dm  # this species is a fraction f_X of local DM
    velocity_factor = np.exp(-((q / (2.0 * m_dm * V0)) ** 2)) - np.exp(
        -((VESC / V0) ** 2)
    )
    return 16.0 * np.sqrt(np.pi) * n_dm / (V0 * q**3) * np.clip(velocity_factor, 0, None)


def _q_grid(m_dm: float, n_grid: int = 4000) -> np.ndarray:
    return np.geomspace(Q_TH, q_max(m_dm), n_grid)


def rate_alpha1(m_dm: float) -> float:
    """Total rate above threshold at unit coupling, in s^-1."""
    q = _q_grid(m_dm)
    return float(np.trapezoid(drdq_alpha1(q, m_dm), q)) * GEV_PER_S


def kick_spectrum_cdf(m_dm: float):
    """Normalized signal CDF in q on [q_th, q_max] — the ``spectrum_cdf``."""
    q = _q_grid(m_dm)
    pdf = drdq_alpha1(q, m_dm)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(q))])
    return spectrum_cdf_from_samples(q, cdf / cdf[-1])


def coupling_limit(mu_ul: float, m_dm: float) -> float:
    """Convert a limit on expected counts to a limit on alpha_n.

    N(alpha_n) = (alpha_n N_n)^2 R(alpha=1) T_obs is *quadratic* in the
    coupling, so alpha_n^lim = sqrt(mu_UL / (R1 T)) / N_n.
    """
    return float(np.sqrt(mu_ul / (rate_alpha1(m_dm) * T_OBS_S)) / N_NEUTRONS)


def run(*, n=4000, n_masses=32, make_plot=False, rng=None):
    """Sweep DM masses; return per-mass mu and alpha_n limits for three methods."""
    rng = np.random.default_rng(0) if rng is None else rng
    table = OptimumIntervalTable(rng=rng)  # one calibration reused for every mass

    mu_zero_event = poisson_upper_limit(0, CONFIDENCE)  # the "<N> ~ 3" benchmark
    masses = np.geomspace(1.1 * MASS_FLOOR, 1.22e19, n_masses)  # up to m_Planck

    rows = []
    for m in masses:
        # Only kicks inside the kinematically allowed window can be signal.
        inside = KICKS[(KICKS > Q_TH) & (KICKS < q_max(m))]
        mu_oi = table.upper_limit(
            inside, confidence=CONFIDENCE, spectrum_cdf=kick_spectrum_cdf(m), n=n
        )
        mu_pois = poisson_upper_limit(inside.size, CONFIDENCE)
        rows.append(
            dict(
                mass=m,
                n_candidates=int(inside.size),
                mu_optint=mu_oi,
                mu_poisson=mu_pois,
                alpha_optint=coupling_limit(mu_oi, m),
                alpha_poisson=coupling_limit(mu_pois, m),
                alpha_zero_event=coupling_limit(mu_zero_event, m),
            )
        )

    if make_plot:
        _plot_exclusion(rows)
    return dict(mu_zero_event=mu_zero_event, rows=rows)


def _plot_exclusion(rows):
    import matplotlib.pyplot as plt

    m = np.array([r["mass"] for r in rows])
    a_oi = np.array([r["alpha_optint"] for r in rows])
    a_po = np.array([r["alpha_poisson"] for r in rows])
    a_id = np.array([r["alpha_zero_event"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(m, a_oi, 1.0, color="#1f77b4", alpha=0.12, lw=0)
    ax.plot(
        m, a_oi, "-", color="#1f77b4", lw=2, label="optimum interval (with candidates)"
    )
    ax.plot(
        m, a_po, "--", color="#d62728", lw=1.6, label="Poisson counting (with candidates)"
    )
    ax.plot(
        m,
        a_id,
        ":",
        color="#7f7f7f",
        lw=1.4,
        label=r"zero-event ideal ($\langle N\rangle=3$)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(m.min(), m.max())
    ax.set_ylim(a_id.min() / 3, 1.0)
    ax.set_xlabel(r"Dark matter mass  $m_{\rm DM}$  [GeV]")
    ax.set_ylabel(r"95% CL upper limit on coupling per neutron  $\alpha_n$")
    ax.set_title("Momentum-kick exclusion: optimum interval vs counting (toy)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    from pathlib import Path

    out = Path(__file__).resolve().parent.parent / "figures" / "uhdm_exclusion.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("UHDM momentum-kick search: kicks -> mu limit -> coupling exclusion\n")
    print(f"candidate kicks [TeV]: {KICKS / 1000}")
    print(f"threshold {Q_TH / 1000:.1f} TeV | mass floor {MASS_FLOOR:.2e} GeV\n")
    result = run(make_plot=True)
    print(
        f"zero-event benchmark: mu_UL = {result['mu_zero_event']:.2f} "
        "(the '<N> ~ 3' rule)\n"
    )
    print(
        f"{'mass[GeV]':>10} {'in-window':>9} {'mu_OI':>7} {'mu_Pois':>8} "
        f"{'alpha_OI':>10} {'alpha_Pois':>11}"
    )
    for r in result["rows"][::4]:
        print(
            f"{r['mass']:>10.2e} {r['n_candidates']:>9d} {r['mu_optint']:>7.2f} "
            f"{r['mu_poisson']:>8.2f} {r['alpha_optint']:>10.2e} "
            f"{r['alpha_poisson']:>11.2e}"
        )
    print("\nSee TUTORIAL_UHDM.md for the walkthrough.")


if __name__ == "__main__":
    main()
