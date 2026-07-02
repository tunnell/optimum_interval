# optimum_interval

Yellin's **maximum-gap** and **optimum-interval** methods for frequentist upper
limits in the presence of an unknown, non-subtractable background — using only
the signal *shape*, with no background model and no binning.

- **[Tutorial](tutorial.md)** — from a signal spectrum to an exclusion curve.
- **[Explanation](explanation.md)** — the derivation and a reimplement-it-yourself recipe.
- **[API reference](api.md)** — the public functions and classes.

```python
import numpy as np
from optimum_interval import OptimumIntervalTable

table = OptimumIntervalTable(rng=np.random.default_rng(0))
events = np.sort(np.random.default_rng(1).random(8))   # events in cumulant space
print(table.upper_limit(events, confidence=0.9, n=2000))
```

Method reference: S. Yellin, *"Finding an Upper Limit in the Presence of Unknown
Background"*, Phys. Rev. **D66** (2002) 032005,
[arXiv:physics/0203002](https://arxiv.org/abs/physics/0203002).
