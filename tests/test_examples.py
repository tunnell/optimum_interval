"""Smoke tests for the runnable examples (fast, low statistics).

Also exercises the raw-observable -> limit path (the spectrum_cdf argument end to
end), which the rest of the suite does not otherwise cover.
"""

import importlib.util
from pathlib import Path

import numpy as np

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
