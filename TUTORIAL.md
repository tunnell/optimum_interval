# Tutorial: an upper limit on the number of events

The core question this package answers: **given your data, what is the largest
number of signal events you cannot rule out?** — a 90% CL frequentist upper limit
on the expected signal count `mu` (a Poisson mean), using only the signal's
*shape*, with no background model and no binning.

This is a hands-on how-to; for *why* it works see [`EXPLANATION.md`](EXPLANATION.md).
Everything here is runnable in [`examples/upper_limit.py`](examples/upper_limit.py):

```bash
python examples/upper_limit.py
```

## Install

```bash
pip install -e ".[dev]"    # not yet on PyPI
```

## The one-minute mental model

You measure some quantity `x` for each event (an energy, a time, a position, a
BDT score — anything 1-D). You know the *shape* of how signal would be
distributed in `x`, but not how many signal events there are. The method finds
regions of `x` that are anomalously **empty** relative to the proposed signal and
turns "too empty to be that strong a signal" into an upper limit on `mu`. It all
happens in *cumulant space*: the signal's CDF maps `x` onto `[0, 1]`, where the
signal is uniform.

## Simplest case: signal uniform in your observable

If the signal is uniform in `x` (or you've already applied its CDF yourself), your
events live on `[0, 1]` and you go straight to a limit:

```python
import numpy as np
from optimum_interval import OptimumIntervalTable

events = np.array([0.08, 0.11, 0.55, 0.58, 0.62, 0.95])   # 6 events on [0, 1]
table = OptimumIntervalTable(rng=np.random.default_rng(0))  # seed = reproducible
mu_ul = table.upper_limit(events, confidence=0.9, n=5000)
print(mu_ul)
```

**Interpretation:** `mu_ul` is the largest *expected* number of signal events
compatible with the observed emptiness at 90% CL. If your signal model predicts
more than `mu_ul` events, it is excluded at 90% CL. (`n` is the number of
Monte-Carlo trials per candidate `mu` — bigger = less noisy limit.)

## No Monte Carlo needed: the maximum-gap limit

For the maximum-gap statistic there is a closed form (Yellin Eq. 2), so this
limit needs no simulation at all — just the size of the largest empty gap:

```python
from optimum_interval import cumulant_points, max_gap_upper_limit

max_gap = np.diff(cumulant_points(events)).max()   # largest gap, as a fraction
mu_ul_maxgap = max_gap_upper_limit(max_gap, confidence=0.9)
```

The optimum-interval limit above (which also uses intervals containing a few
events) is generally a bit stronger; the max-gap limit is the fast, exact
special case.

## Non-uniform signal shape

If the signal is *not* uniform in `x`, pass its normalized CDF as `spectrum_cdf`.
A helper builds one from any density (PDF), up to normalization:

```python
from optimum_interval import spectrum_cdf_from_pdf

# any known shape on the analysis window [0, 10] -- here a falling exponential
cdf = spectrum_cdf_from_pdf(lambda x: np.exp(-x / 3.0), 0.0, 10.0)

events_x = np.array([0.3, 0.5, 0.9, 1.4, 6.0, 8.5])          # events in x
mu_ul = table.upper_limit(events_x, spectrum_cdf=cdf, confidence=0.9, n=5000)
```

The one requirement: `spectrum_cdf` must be **normalized** so the analysis window
maps onto exactly `[0, 1]` (`cdf(x_min) == 0`, `cdf(x_max) == 1`). **Do not fold
`mu` into it** — `mu` (the total count) is what you are limiting; only the *shape*
goes in the CDF. (`cumulant_points` raises if the CDF is unnormalized or
non-monotonic.)

## From a count to a rate or cross section

`mu_ul` is a limit on the expected *count*. If your signal count scales linearly
with some physical parameter — `mu = theta * mu_1` (e.g. `theta` a rate or cross
section and `mu_1` the expected count per unit `theta` = exposure × efficiency ×
…) — then `theta_ul = mu_ul / mu_1`. Sweeping a model parameter and re-limiting at
each point traces an exclusion curve. A full worked example (a dark-matter
recoil-spectrum → cross-section exclusion curve) is in
[`examples/dark_matter_exclusion.py`](examples/dark_matter_exclusion.py).

## Tips

- **Reproducibility:** pass a seeded `np.random.default_rng(seed)` to
  `OptimumIntervalTable`; the limit is then deterministic.
- **Persistence:** `table.save("tables.p")` / `OptimumIntervalTable.load(...)`
  reuse a calibration across sessions.
- **Max gap only:** for the simplest, MC-free limit use `max_gap_upper_limit`
  (or `c0` / `x0` directly).
- **Under the hood & validation:** `EXPLANATION.md` (derivation + reimplement
  recipe) and `python reproduce_figures.py` (reproduces the paper's figures).
