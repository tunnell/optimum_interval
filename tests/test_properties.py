"""Property-based (Hypothesis) tests for the pure functions."""

from itertools import pairwise

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from optimum_interval import c0, k_largest_intervals


@st.composite
def point_lists(draw):
    """A sorted list of >= 2 distinct points on [0, 1]."""
    xs = draw(
        st.lists(
            st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=12,
            unique=True,
        )
    )
    return np.array(sorted(xs))


@settings(max_examples=60, deadline=None)
@given(point_lists())
def test_k_largest_nests_and_does_not_mutate(points):
    before = points.copy()
    sizes = k_largest_intervals(points)
    np.testing.assert_array_equal(points, before)  # input untouched
    values = [sizes[k] for k in sorted(sizes)]  # s_0 <= s_1 <= ... (nesting)
    assert all(a <= b + 1e-12 for a, b in pairwise(values))


@settings(max_examples=60, deadline=None)
@given(point_lists())
def test_k_largest_is_permutation_invariant(points):
    shuffled = points.copy()
    np.random.default_rng(0).shuffle(shuffled)
    a = k_largest_intervals(points)
    b = k_largest_intervals(shuffled)
    assert a.keys() == b.keys()
    assert all(a[k] == b[k] for k in a)


@settings(max_examples=80, deadline=None)
@given(
    st.floats(2.5, 60.0),
    st.floats(0.02, 0.98),
    st.floats(0.02, 0.98),
)
def test_c0_bounded_and_monotone(mu, f_lo, f_hi):
    x_lo, x_hi = mu * min(f_lo, f_hi), mu * max(f_lo, f_hi)
    v_lo, v_hi = c0(x_lo, mu), c0(x_hi, mu)
    assert 0.0 <= v_lo <= 1.0 and 0.0 <= v_hi <= 1.0
    assert v_lo <= v_hi + 1e-9  # non-decreasing in x
    assert c0(mu * 1.5, mu) == 1.0  # x > mu
