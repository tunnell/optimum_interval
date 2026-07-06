# Parameter scans

`OptimumIntervalTable.upper_limit` solves for the signal normalization
directly, which requires the spectrum *shape* to stay fixed while $\mu$
varies. Many real signals break that factorization: the shape moves with the
parameter you are limiting (a finite-range mediator changes the momentum
endpoint with the coupling; overburden attenuation reshapes the arrival
spectrum; velocity-dependent couplings tilt recoil spectra). The limit is
then a **level set of the extremeness surface**, not a single solve.

The `optimum_interval.scanning` module packages that pattern:

```python
import numpy as np
from optimum_interval import (new_table, scan_extremeness,
                              excluded_interval)

table = new_table(seed=0)           # one calibration table for the whole scan
couplings = np.geomspace(1e-9, 1.0, 41)

ps = []
for g in couplings:
    x, rate = my_physics(g)         # your dR/dx on a grid, for this coupling
    p, mu = scan_extremeness(table, events, x, rate, exposure)
    ps.append(p)

low, high = excluded_interval(couplings, ps, level=0.95)
```

`scan_extremeness` returns the *extremeness* $p$: the fraction of
background-free pseudo-experiments under that hypothesis whose most
anomalously empty stretch looks less empty than the data. The rule "exclude
where $p \ge C$" has frequentist coverage $C$, any confidence level is a
level set of the same surface (no rescan for 90% vs 95%), and the test is
one-sided: background can weaken an exclusion but never fake one.

Three details make large scans cheap and honest:

- **Table sharing.** Expected counts are rounded onto a 2%-spaced log grid
  (`round_log`), so one Monte-Carlo table serves every scan point with the
  same rounded $\mu$.
- **Shortcuts.** Points with $\mu$ below `mu_floor` return $p = 0$ without
  Monte Carlo (nothing expected); points above `mu_cap` return $p = 1$
  (overwhelmingly excluded — raise the cap if your event list is long).
- **Kinematic windows.** Events beyond a scan point's spectrum endpoint are
  dropped automatically: they cannot be signal *at that point*.

For a 2-D scan (e.g. mass × coupling), evaluate `scan_extremeness` on the
grid and draw the confidence-level contour of the surface. The
[momentum-kick tutorial](uhdm.md) and the finite-range notebook in
`examples/` show the pattern end to end.
