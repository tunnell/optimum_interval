"""Smoke test for examples/tutorial.py.

Exercises the raw-energy -> limit path (the spectrum_cdf argument end to end),
which the rest of the suite does not otherwise cover.
"""

import importlib.util
from pathlib import Path

import numpy as np


def _load_tutorial():
    path = Path(__file__).resolve().parent.parent / "examples" / "tutorial.py"
    spec = importlib.util.spec_from_file_location("oi_tutorial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tutorial_runs_end_to_end():
    tutorial = _load_tutorial()
    result = tutorial.run_tutorial(
        n=300, n_masses=3, make_plot=False, rng=np.random.default_rng(0)
    )
    rows = result["rows"]
    assert len(rows) == 3
    for r in rows:
        for key in ("mu_maxgap", "mu_optint", "sigma_optint"):
            assert np.isfinite(r[key]) and r[key] > 0
