"""Test the calibration builder's worker and payload compatibility.

Exercises ``_build_one`` directly (no multiprocessing, so it is cross-platform)
and checks the assembled payload loads via ``OptimumIntervalTable.load``.
"""

import importlib.util
import pickle
from pathlib import Path

from optimum_interval import OptimumIntervalTable


def _load_build_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_tables.py"
    spec = importlib.util.spec_from_file_location("oi_build_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_one_output_is_loadable(tmp_path):
    build = _load_build_script()
    payload = {"itv_sizes": {}, "opt_itvs": {}, "n_trials": {}}
    for index, mu in enumerate((4.0, 6.0)):
        mu_out, itv_sizes, opt_itvs, n = build._build_one(mu, 300, seed=0, index=index)
        assert mu_out == mu and n == 300 and opt_itvs.shape == (300,)
        payload["itv_sizes"][mu] = itv_sizes
        payload["opt_itvs"][mu] = opt_itvs
        payload["n_trials"][mu] = n

    out = tmp_path / "tables.p"
    with open(out, "wb") as f:
        pickle.dump(payload, f)

    table = OptimumIntervalTable.load(out)
    assert set(table.n_trials) == {4.0, 6.0}
    assert 0.0 < table.bar_c_max(0.9, 4.0) <= 1.0
