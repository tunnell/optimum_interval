"""Tests for the spectrum_cdf helpers."""

import numpy as np
import pytest

from optimum_interval import (
    cumulant_points,
    spectrum_cdf_from_pdf,
    spectrum_cdf_from_samples,
)


def test_from_pdf_matches_analytic_exponential():
    e0, e_min, e_max = 8.0, 1.0, 40.0
    cdf = spectrum_cdf_from_pdf(lambda e: np.exp(-e / e0), e_min, e_max)

    def analytic(e):
        lo = np.exp(-e_min / e0)
        return (lo - np.exp(-e / e0)) / (lo - np.exp(-e_max / e0))

    e = np.linspace(e_min, e_max, 50)
    np.testing.assert_allclose(cdf(e), analytic(e), atol=1e-4)
    # Endpoints map onto [0, 1].
    assert cdf(np.array([e_min]))[0] == pytest.approx(0.0, abs=1e-6)
    assert cdf(np.array([e_max]))[0] == pytest.approx(1.0, abs=1e-6)


def test_from_pdf_usable_end_to_end():
    # The produced CDF must satisfy cumulant_points' normalization check.
    cdf = spectrum_cdf_from_pdf(lambda e: np.exp(-e / 5.0), 1.0, 40.0)
    pts = cumulant_points(np.array([2.0, 5.0, 30.0]), spectrum_cdf=cdf)
    assert pts[0] == 0.0 and pts[-1] == 1.0
    assert np.all(np.diff(pts) >= 0)


def test_from_pdf_rejects_bad_density():
    with pytest.raises(ValueError):
        spectrum_cdf_from_pdf(lambda e: -np.ones_like(e), 1.0, 40.0)  # negative
    with pytest.raises(ValueError):
        spectrum_cdf_from_pdf(lambda e: np.zeros_like(e), 1.0, 40.0)  # integrates to 0


def test_from_samples_rescales_endpoints():
    e = np.array([1.0, 2.0, 5.0, 10.0])
    cdf = spectrum_cdf_from_samples(e, [0.2, 0.4, 0.7, 0.9])  # not spanning [0,1]
    assert cdf(np.array([1.0]))[0] == pytest.approx(0.0)
    assert cdf(np.array([10.0]))[0] == pytest.approx(1.0)
    assert np.all(np.diff(cdf(np.linspace(1, 10, 20))) >= 0)


def test_from_samples_rejects_bad_input():
    with pytest.raises(ValueError):
        spectrum_cdf_from_samples([1.0, 1.0, 2.0], [0.0, 0.5, 1.0])  # not increasing
    with pytest.raises(ValueError):
        spectrum_cdf_from_samples([1.0, 2.0, 3.0], [0.0, 0.6, 0.5])  # CDF decreases
