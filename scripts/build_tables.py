#!/usr/bin/env python3
"""Build optimum-interval calibration tables in parallel.

The calibration depends only on ``mu`` (not on any data), so it can be generated
once at high statistics on a big machine and reused everywhere.  This computes it
on a ``mu`` grid across worker *processes* (the Monte Carlo is CPU-bound, so
processes -- not threads -- give real parallelism under the GIL) and saves a
pickle loadable with :meth:`OptimumIntervalTable.load`.

Reproducible: each grid point gets an independent, deterministic RNG stream
derived from ``--seed`` and its index.

Example
-------
    python scripts/build_tables.py --mu-max 60 --n-mu 60 --n-trials 200000 \\
        --workers 8 --out tables.p

    # then, anywhere:
    from optimum_interval import OptimumIntervalTable
    table = OptimumIntervalTable.load("tables.p")
"""

from __future__ import annotations

import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from optimum_interval import OptimumIntervalTable


def _build_one(mu: float, n: int, seed: int, index: int):
    """Worker: build the calibration for a single ``mu`` (module-level = picklable)."""
    rng = np.random.default_rng([seed, index])  # independent, deterministic stream
    table = OptimumIntervalTable(rng=rng)
    table.generate(mu, n)
    return mu, table.itv_sizes[mu], table.opt_itvs[mu], n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mu-min", type=float, default=2.303)
    parser.add_argument("--mu-max", type=float, default=60.0)
    parser.add_argument("--n-mu", type=int, default=60, help="number of grid points")
    parser.add_argument("--geomspace", action="store_true", help="log-spaced grid")
    parser.add_argument("--n-trials", type=int, default=200000)
    parser.add_argument("--workers", type=int, default=None, help="default: all cores")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="tables.p")
    args = parser.parse_args(argv)

    space = np.geomspace if args.geomspace else np.linspace
    mus = space(args.mu_min, args.mu_max, args.n_mu)
    print(
        f"building {len(mus)} mu points x {args.n_trials} trials "
        f"on {args.workers or 'all'} workers -> {args.out}",
        flush=True,
    )

    payload = {"itv_sizes": {}, "opt_itvs": {}, "n_trials": {}}
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_build_one, float(mu), args.n_trials, args.seed, i)
            for i, mu in enumerate(mus)
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            mu, itv_sizes, opt_itvs, n = future.result()
            payload["itv_sizes"][mu] = itv_sizes
            payload["opt_itvs"][mu] = opt_itvs
            payload["n_trials"][mu] = n
            print(
                f"  [{done}/{len(mus)}] mu={mu:.3f}  "
                f"({time.time() - start:.0f}s elapsed)",
                flush=True,
            )

    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(
        f"wrote {args.out} in {time.time() - start:.0f}s; "
        f"load with OptimumIntervalTable.load('{args.out}')"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
