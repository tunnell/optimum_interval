# Changelog

## 0.3.0 (2026-07-06)

- New `optimum_interval.scanning` module: the parameter-scan pattern for
  spectra whose shape depends on the parameter being limited
  (`scan_extremeness`, `spectrum_from_rate`, `excluded_interval`,
  `new_table`, `round_log`), with a docs page. Previously this logic lived
  only in tutorials and downstream analyses.
- The finite-range UHDM tutorial now uses the scanning API instead of
  hand-rolling the loop.

## 0.2.0 (2026-07-02)

- Deterministic `upper_limit`: fixed cached mu grid with interpolated crossing;
  reproducible given the generator seed.
- `c0` evaluated in log-magnitude; correct at large mu. `x0` raises below the
  attainable confidence. `cumulant_points` validates that `spectrum_cdf` is
  normalized and monotonic.
- New: `spectrum_cdf_from_pdf` / `spectrum_cdf_from_samples`,
  `poisson_upper_limit`, `max_gap_upper_limit`, `ComparisonEngine`.
- Tutorials moved to `examples/`; added the momentum-kick (ultraheavy dark
  matter) tutorial and example.
- Docs site (mkdocs, ReadTheDocs), CITATION.cff + Zenodo DOI, figures
  regenerated at high statistics (all five Yellin figures reproduced).

## 0.1.0 (2026-07-01)

First tagged release: packaged, tested implementation of Yellin's maximum-gap
and optimum-interval methods, with `EXPLANATION.md` and figure reproduction.
