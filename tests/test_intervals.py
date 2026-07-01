"""Tests for the pure interval geometry."""

from itertools import pairwise

import numpy as np
import pytest

from optimum_interval.intervals import cumulant_points, k_largest_intervals


def test_max_gap_zero_events():
    # Ported from the original inline asserts, now with tolerant float compare.
    sizes = k_largest_intervals(np.array([0.0, 0.1, 0.2, 0.84, 0.85]))
    assert sizes[0] == pytest.approx(0.84 - 0.2)


def test_two_event_interval():
    sizes = k_largest_intervals(np.array([0.0, 0.1, 0.2, 0.84, 0.85]))
    assert sizes[2] == pytest.approx(0.84 - 0.0)


def test_unsorted_input_gives_same_answer():
    sizes = k_largest_intervals(np.array([0.85, 0.0, 0.1, 0.84, 0.2]))
    assert sizes[2] == pytest.approx(0.84 - 0.0)


def test_input_is_not_mutated():
    # Regression test for the original in-place ``list_of_energies.sort()`` bug.
    events = np.array([0.85, 0.0, 0.1, 0.84, 0.2])
    before = events.copy()
    k_largest_intervals(events)
    np.testing.assert_array_equal(events, before)


def test_k_range_and_monotonic_in_k():
    # For N points there are k = 0 .. N-2 intervals, and the k-largest interval
    # can only grow with k (it contains one more event, over a wider span).
    points = np.array([0.0, 0.2, 0.5, 0.55, 1.0])
    sizes = k_largest_intervals(points)
    assert set(sizes) == set(range(len(points) - 1))
    values = [sizes[k] for k in sorted(sizes)]
    assert all(a <= b for a, b in pairwise(values))


def test_spectrum_cdf_applied():
    # A monotonic CDF transforms sizes; identity is the default.
    events = np.array([0.0, 1.0, 2.0, 3.0])
    ident = k_largest_intervals(events)
    scaled = k_largest_intervals(events, spectrum_cdf=lambda e: e / 3.0)
    assert scaled[0] == pytest.approx(ident[0] / 3.0)


def test_cumulant_points_adds_endpoints_and_sorts():
    pts = cumulant_points(np.array([0.6, 0.1, 0.3]))
    np.testing.assert_allclose(pts, [0.0, 0.1, 0.3, 0.6, 1.0])


def test_cumulant_points_edge_cases():
    # 0-event run -> just the endpoints; single event; points on the boundaries.
    np.testing.assert_allclose(cumulant_points(np.array([])), [0.0, 1.0])
    np.testing.assert_allclose(cumulant_points(np.array([0.3])), [0.0, 0.3, 1.0])
    np.testing.assert_allclose(
        cumulant_points(np.array([0.0, 1.0])), [0.0, 0.0, 1.0, 1.0]
    )


def test_cumulant_points_rejects_unnormalized_cdf():
    with pytest.raises(ValueError):
        cumulant_points(np.array([0.2, 1.3]))  # cumulant > 1
    with pytest.raises(ValueError):
        cumulant_points(np.array([-0.1, 0.5]))  # cumulant < 0
    with pytest.raises(ValueError):  # non-monotone CDF
        cumulant_points(np.array([0.1, 0.9]), spectrum_cdf=lambda e: 1.0 - e)
