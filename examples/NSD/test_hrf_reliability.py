"""HRF reliability compares full temporal curves at each grayordinate."""

import importlib

import numpy as np
import pytest

from boldtailor.hrf_library import HrfLibrary


def reliability():
    try:
        return importlib.import_module("examples.NSD.hrf_reliability")
    except ModuleNotFoundError:
        pytest.fail("Full-HRF correlation diagnostics have not been implemented")


@pytest.fixture
def library():
    # Different durations also exercise the library's common padded time grid.
    return HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 40]]
    )


def test_curve_correlations_match_pearson_over_full_common_time_grid(library):
    odd = np.array([0, 1, 2, 1, np.nan, -1, 2])
    even = np.array([1, 2, 2, np.nan, 0, -1, -1])
    result = reliability().hrf_curve_correlations(library, odd, even)
    expected = np.full((3, len(odd)), np.nan)
    for i, (a, b) in enumerate(zip(odd, even, strict=True)):
        for row, (x, y) in enumerate(((a, b), (a, 0), (b, 0))):
            if np.isfinite(x) and np.isfinite(y) and x >= 0 and y >= 0:
                expected[row, i] = np.corrcoef(
                    library.curves[int(x)], library.curves[int(y)]
                )[0, 1]
    np.testing.assert_allclose(result, expected, atol=1e-14)
    assert result[0, 2] == pytest.approx(1.0)
    assert result[1, 0] == pytest.approx(1.0)
    assert np.isfinite(result[1, 3])  # Missing even HRF doesn't erase odd baseline.
    assert np.isfinite(result[2, 4])
    assert np.isnan(result[:, 5]).all()
    assert np.all(np.abs(result[np.isfinite(result)]) <= 1)


@pytest.mark.parametrize(
    "odd,even",
    [
        ([0], [0, 1]),
        ([[0]], [[0]]),
        ([0.5], [0]),
        ([3], [0]),
        ([-2], [0]),
        ([np.inf], [0]),
    ],
)
def test_invalid_selections_are_not_silently_reinterpreted(library, odd, even):
    with pytest.raises(ValueError, match="HRF|indices|shape"):
        reliability().hrf_curve_correlations(library, odd, even)


def test_empty_selections_preserve_three_map_shape(library):
    assert reliability().hrf_curve_correlations(library, [], []).shape == (3, 0)
