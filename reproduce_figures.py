#!/usr/bin/env python3
"""Regenerate and *verify* the figures for the optimum-interval method.

Run this to reproduce, from the cleaned library, the one plot the original repo
produced (Yellin Fig. 2) plus independent cross-checks and the explanatory
figures used by ``EXPLANATION.md``.  Every reproduction of a *paper* figure is
saved next to the original (extracted from the arXiv source) so the agreement
can be seen directly.

Usage
-----
    python reproduce_figures.py --quick     # fast, lower statistics (~1-2 min)
    python reproduce_figures.py --full      # publication statistics (~10-15 min)

Outputs go to ``figures/``.  A deterministic seed makes every run reproducible.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
FIGS = REPO / "figures"
PAPER_SRC = REPO / "paper_figs"

# arXiv EPS filenames are scrambled relative to the compiled figure numbers:
#   figure-1 -> Fig 1,  figure-3 -> Fig 2,  figure-2 -> Fig 5
PAPER_EPS = {"fig1": "figure-1.eps", "fig2": "figure-3.eps", "fig5": "figure-2.eps"}


# --------------------------------------------------------------------------- #
# paper-figure extraction (read-only, for side-by-side comparison)
# --------------------------------------------------------------------------- #
def paper_png(which: str) -> Path | None:
    """Convert a paper EPS to PNG (once) and return its path, or None."""
    eps = PAPER_SRC / PAPER_EPS[which]
    if not eps.exists():
        return None
    out = FIGS / f"paper_{which}.png"
    if not out.exists():
        tool = shutil.which("magick") or shutil.which("convert")
        if tool is None:
            return None
        cmd = [tool, "-density", "150", str(eps), str(out)]
        if tool.endswith("magick"):
            cmd = [tool] + cmd[1:]
        subprocess.run(cmd, check=True, capture_output=True)
    return out if out.exists() else None


def side_by_side(ours: Path, paper: Path | None, out: Path, title: str) -> None:
    """Save a two-panel image: our reproduction next to the paper figure."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].imshow(mpimg.imread(ours))
    axes[0].set_title("This library")
    axes[0].axis("off")
    if paper is not None:
        axes[1].imshow(mpimg.imread(paper))
        axes[1].set_title("Yellin (2002), original")
    else:
        axes[1].text(0.5, 0.5, "paper figure unavailable", ha="center")
    axes[1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


# --------------------------------------------------------------------------- #
# Fig. 2  --  bar_c_max(0.9, mu)  (the plot the original repo produced)
# --------------------------------------------------------------------------- #
def reproduce_fig2(seed: int, n: int, n_mu: int) -> None:
    import matplotlib.pyplot as plt

    from optimum_interval import OptimumIntervalTable
    from optimum_interval.plotting import bar_c_max_curve, plot_bar_c_max

    print(f"[Fig 2] bar_c_max(0.9, mu): {n_mu} mu points, n={n} trials each")
    table = OptimumIntervalTable(rng=np.random.default_rng(seed))
    mu_values = np.geomspace(3.0, 100.0, n_mu)
    curve = bar_c_max_curve(table, mu_values, confidence=0.9, n=n)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_bar_c_max(mu_values, curve, ax=ax)
    ax.set_ylim(0.88, 0.98)
    fig.tight_layout()
    ours = FIGS / "fig02_barCmax_reproduction.png"
    fig.savefig(ours, dpi=140)
    plt.close(fig)
    print(f"  wrote {ours.relative_to(REPO)}")

    side_by_side(
        ours, paper_png("fig2"), FIGS / "fig02_side_by_side.png",
        r"Fig. 2 reproduction: $\bar C_{\mathrm{Max}}(0.9,\mu)$ vs $\mu$",
    )
    # Quantitative agreement summary.
    lo = curve[mu_values < 3.9]
    print(f"  plateau (2.3<mu<3.9) mean = {lo.mean():.3f} (Yellin: ~0.90);"
          f" rises to {curve[-1]:.3f} at mu={mu_values[-1]:.0f} (Yellin: ~0.97)")


# --------------------------------------------------------------------------- #
# Cross-check: k=0 Monte Carlo max-gap CDF vs analytic C_0 (Eq. 2)
# --------------------------------------------------------------------------- #
def reproduce_c0_validation(seed: int, n: int) -> None:
    import matplotlib.pyplot as plt

    from optimum_interval import OptimumIntervalTable, c0

    print(f"[C0 check] MC max-gap CDF vs analytic Eq. 2, n={n} trials")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mu in zip(axes, (3.0, 5.0)):
        table = OptimumIntervalTable(rng=np.random.default_rng(seed))
        table.generate(mu, n)
        gap_events = mu * np.sort(table.itv_sizes[mu][0])   # fraction -> events
        empirical = np.arange(1, gap_events.size + 1) / gap_events.size
        xs = np.linspace(0, mu, 400)
        ax.plot(xs, c0(xs, mu), "C3-", lw=2, label="analytic $C_0$ (Eq. 2)")
        ax.plot(gap_events, empirical, "C0.", ms=1.5, alpha=0.4,
                label="Monte-Carlo max gap")
        ax.set_title(rf"$\mu = {mu:.0f}$")
        ax.set_xlabel("max-gap size $x$  (expected events)")
        ax.set_ylabel(r"$P(\mathrm{max\ gap} < x)$")
        ax.legend(loc="lower right", fontsize=8)
        # Report body agreement, excluding the probability atom at x = mu (the
        # e^{-mu} fraction of zero-event trials whose max gap equals the full range).
        body = gap_events < 0.98 * mu
        max_dev = np.max(np.abs(empirical[body] - c0(gap_events[body], mu)))
        print(f"  mu={mu:.0f}: max |MC - C0| (body) = {max_dev:.4f}"
              f"  [MC noise ~ {1.0 / np.sqrt(n):.4f}]")
    fig.suptitle("Validation: Monte-Carlo max gap reproduces analytic $C_0$")
    fig.tight_layout()
    out = FIGS / "c0_validation.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


# --------------------------------------------------------------------------- #
# Fig. 5  --  bar_p_max(0.9, mu)  (bonus: the p_max method, paper Appendix C)
# --------------------------------------------------------------------------- #
def bar_p_max(table, mu: float, n: int, confidence: float = 0.9) -> float:
    """90th percentile of the p_max statistic under background-free MC.

    p_max = max over intervals of P(more than n events | expected mu*size).
    Since P(>n | .) grows with size, the max over intervals with n events uses
    the n-largest interval -- the same k-largest sizes as C_max.
    """
    from scipy.stats import poisson

    from optimum_interval.intervals import k_largest_intervals

    p_maxes = np.empty(n)
    for i, trial in enumerate(table.generate_trials(mu, n)):
        sizes = k_largest_intervals(trial)
        ks = np.fromiter(sizes.keys(), dtype=int)
        xs = mu * np.fromiter(sizes.values(), dtype=float)
        p_maxes[i] = poisson.sf(ks, xs).max()   # vectorised over k
    return float(np.quantile(p_maxes, confidence))


def reproduce_fig5(seed: int, n: int, n_mu: int) -> None:
    import matplotlib.pyplot as plt

    from optimum_interval import OptimumIntervalTable, x0

    print(f"[Fig 5] bar_p_max(0.9, mu): {n_mu} mu points, n={n} trials each")
    table = OptimumIntervalTable(rng=np.random.default_rng(seed))
    mu_values = np.geomspace(3.0, 70.0, n_mu)
    curve = np.array([bar_p_max(table, mu, n) for mu in mu_values])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mu_values, curve, "C0-", lw=1.5, label="Monte Carlo")
    # Low-mu analytic anchor (only n=0 contributes): p_max = 1 - exp(-x0).
    lo = mu_values[mu_values < 5.156]
    ax.plot(lo, [1 - np.exp(-x0(0.9, m)) for m in lo], "C3--", lw=1.5,
            label=r"$1-e^{-x_0(0.9,\mu)}$ (low-$\mu$)")
    ax.axvline(5.156, color="0.8", lw=0.8, ls="--", zorder=0)  # Table II threshold
    ax.set_xscale("log")
    ax.set_xlim(3, 70)
    ax.set_xlabel(r"Total expected number of events $\mu$")
    ax.set_ylabel(r"$90\%\ \bar p_{\mathrm{Max}}$")
    ax.set_title(r"Reproduction of Yellin Fig. 5: $\bar p_{\mathrm{Max}}(0.9,\mu)$")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    ours = FIGS / "fig05_barpmax_reproduction.png"
    fig.savefig(ours, dpi=140)
    plt.close(fig)
    print(f"  wrote {ours.relative_to(REPO)}")
    side_by_side(
        ours, paper_png("fig5"), FIGS / "fig05_side_by_side.png",
        r"Fig. 5 reproduction: $\bar p_{\mathrm{Max}}(0.9,\mu)$ vs $\mu$",
    )


# --------------------------------------------------------------------------- #
# Explanatory figures for EXPLANATION.md
# --------------------------------------------------------------------------- #
def explanatory_figures(seed: int) -> None:
    import matplotlib.pyplot as plt

    print("[explain] cumulant transform + k-largest-interval schematics")
    rng = np.random.default_rng(seed)

    # (A) probability-integral / cumulant transform of an exponential spectrum.
    e0, e_max, mu = 10.0, 50.0, 5.0
    norm = 1.0 - np.exp(-e_max / e0)

    def cdf(e):  # normalised exponential CDF -> uniform [0, 1]
        return (1.0 - np.exp(-e / e0)) / norm

    events_E = np.array([4.0, 9.0, 22.0, 41.0])
    events_u = cdf(events_E)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    E = np.linspace(0, e_max, 400)
    axL.plot(E, np.exp(-E / e0), "k-", lw=1.5)
    axL.vlines(events_E, 0, np.exp(-events_E / e0), color="C3")
    axL.plot(events_E, np.exp(-events_E / e0), "C3o")
    axL.set_xlabel("recoil energy $E$ [keV]")
    axL.set_ylabel(r"$dN/dE \propto e^{-E/E_0}$")
    axL.set_title("Signal spectrum (proposed cross section)")

    axR.plot(E, cdf(E), "k-", lw=1.5)
    for e, u in zip(events_E, events_u):
        axR.plot([e, e, 0], [0, u, u], "C3", lw=0.8)
    axR.plot(events_E, events_u, "C3o")
    axR.set_xlabel("recoil energy $E$ [keV]")
    axR.set_ylabel(r"cumulant $\epsilon(E)\in[0,1]$")
    axR.set_title("Probability-integral transform -> uniform")
    fig.suptitle(r"The cumulant transform maps any spectrum to uniform $[0,1]$")
    fig.tight_layout()
    out = FIGS / "explain_cumulant_transform.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")

    # (B) k-largest interval schematic on the unit interval.
    from optimum_interval.intervals import cumulant_points, k_largest_intervals

    pts = cumulant_points(events_u)
    sizes = k_largest_intervals(pts)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.hlines(0, 0, 1, color="0.6", lw=1)
    ax.plot(pts, np.zeros_like(pts), "ko", ms=7)
    ax.plot([0, 1], [0, 0], "s", color="C7", ms=9, label="range endpoints")
    # Highlight the widest interval for k = 0, 1, 2.
    colors = {0: "C0", 1: "C1", 2: "C2"}
    for k, color in colors.items():
        gap = k + 1
        diffs = pts[gap:] - pts[:-gap]
        i = int(np.argmax(diffs))
        y = 0.15 * (k + 1)
        ax.annotate("", xy=(pts[i + gap], y), xytext=(pts[i], y),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=2))
        ax.text((pts[i] + pts[i + gap]) / 2, y + 0.03,
                f"$k={k}$ (size {sizes[k]:.2f})", color=color, ha="center", fontsize=9)
    ax.set_ylim(-0.1, 0.7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks([])
    ax.set_xlabel(r"cumulant coordinate $\epsilon\in[0,1]$")
    ax.set_title("k-largest intervals: widest span containing exactly $k$ events")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIGS / "explain_klargest_schematic.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="publication statistics")
    parser.add_argument("--quick", action="store_true", help="fast, low statistics")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--only", choices=["fig2", "c0", "fig5", "explain"],
                        help="run just one figure")
    args = parser.parse_args(argv)

    full = args.full and not args.quick
    FIGS.mkdir(exist_ok=True)

    if full:
        cfg = dict(fig2_n=50000, fig2_mu=45, c0_n=200000, fig5_n=40000, fig5_mu=30)
    else:
        cfg = dict(fig2_n=8000, fig2_mu=25, c0_n=60000, fig5_n=8000, fig5_mu=18)
    print(f"seed={args.seed}  mode={'full' if full else 'quick'}  config={cfg}\n")

    run = args.only
    if run in (None, "fig2"):
        reproduce_fig2(args.seed, cfg["fig2_n"], cfg["fig2_mu"])
    if run in (None, "c0"):
        reproduce_c0_validation(args.seed, cfg["c0_n"])
    if run in (None, "fig5"):
        reproduce_fig5(args.seed, cfg["fig5_n"], cfg["fig5_mu"])
    if run in (None, "explain"):
        explanatory_figures(args.seed)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
