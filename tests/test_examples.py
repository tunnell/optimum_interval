"""Smoke tests for the runnable examples (fast, low statistics).

Also exercises the raw-observable -> limit path (the spectrum_cdf argument end to
end), which the rest of the suite does not otherwise cover.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name):
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upper_limit_example_runs():
    result = _load("upper_limit.py").run(n=300, rng=np.random.default_rng(0))
    for case in ("uniform", "shaped"):
        for key in ("max_gap", "optimum_interval"):
            assert np.isfinite(result[case][key]) and result[case][key] > 0


def test_dark_matter_exclusion_example_runs():
    result = _load("dark_matter_exclusion.py").run_exclusion(
        n=300, n_masses=3, make_plot=False, rng=np.random.default_rng(0)
    )
    rows = result["rows"]
    assert len(rows) == 3
    for r in rows:
        for key in ("mu_maxgap", "mu_optint", "sigma_optint"):
            assert np.isfinite(r[key]) and r[key] > 0


def test_uhdm_momentum_kick_example_runs():
    result = _load("uhdm_momentum_kicks.py").run(
        n=500, n_masses=4, make_plot=False, rng=np.random.default_rng(0)
    )
    # The "<N> ~ 3" rule is the zero-event 95% CL Poisson limit.
    assert result["mu_zero_event"] == pytest.approx(-np.log(0.05), abs=1e-3)
    rows = result["rows"]
    assert len(rows) == 4
    for r in rows:
        assert np.isfinite(r["alpha_optint"]) and r["alpha_optint"] > 0
        # Optimum interval never exceeds the mu range bracketed by 0-event
        # Poisson below and full-multiplicity Poisson well above.
        assert 1.0 < r["mu_optint"] < 2 * r["mu_poisson"]
