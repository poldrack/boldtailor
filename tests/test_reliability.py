"""HRF agreement compares full temporal curves at each grayordinate."""

from itertools import combinations

import numpy as np
import pytest

from boldtailor import reliability
from boldtailor.hrf_library import HrfLibrary


@pytest.fixture
def padded_library():
    # Different durations also exercise the library's common padded time grid.
    return HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 40]]
    )


def test_curve_correlations_match_pearson_over_full_common_time_grid(padded_library):
    odd = np.array([0, 1, 2, 1, np.nan, -1, 2])
    even = np.array([1, 2, 2, np.nan, 0, -1, -1])
    result = reliability.curve_correlations(padded_library, odd, even)
    expected = np.full((3, len(odd)), np.nan)
    for i, (a, b) in enumerate(zip(odd, even, strict=True)):
        for row, (x, y) in enumerate(((a, b), (a, 0), (b, 0))):
            if np.isfinite(x) and np.isfinite(y) and x >= 0 and y >= 0:
                expected[row, i] = np.corrcoef(
                    padded_library.curves[int(x)], padded_library.curves[int(y)]
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
def test_invalid_selections_are_not_silently_reinterpreted(padded_library, odd, even):
    with pytest.raises(ValueError, match="HRF|indices|shape"):
        reliability.curve_correlations(padded_library, odd, even)


def test_empty_selections_preserve_three_map_shape(padded_library):
    assert reliability.curve_correlations(padded_library, [], []).shape == (3, 0)


@pytest.fixture
def library():
    return HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )


def test_all_pairs_and_canonical_baselines_match_direct_curve_correlations(library):
    ids = np.array([[1, 0, np.nan, np.nan], [1, 2, 2, np.nan], [2, 2, -1, np.nan]])
    result = reliability.compare_hrfs(library, ids, ["ses-10", "ses-11", "ses-12"])
    pairs = list(combinations(range(3), 2))
    expected = np.full((3, 4), np.nan)
    baseline = np.full((3, 4), np.nan)
    for row, (a, b) in enumerate(pairs):
        for g in range(4):
            if all(np.isfinite(ids[s, g]) and ids[s, g] >= 0 for s in (a, b)):
                curves = library.curves[ids[[a, b], g].astype(int)]
                expected[row, g] = np.corrcoef(curves)[0, 1]
                baseline[row, g] = np.mean(
                    [np.corrcoef(curve, library.curves[0])[0, 1] for curve in curves]
                )
    np.testing.assert_allclose(result["pairwise"], expected, atol=1e-14)
    np.testing.assert_allclose(result["pair_baseline"], baseline, atol=1e-14)
    np.testing.assert_allclose(result["pair_delta"], expected - baseline, atol=1e-14)
    assert result["pair_names"] == [
        "ses-10_vs_ses-11",
        "ses-10_vs_ses-12",
        "ses-11_vs_ses-12",
    ]
    np.testing.assert_allclose(result["summary"][0, :2], expected[:, :2].mean(axis=0))
    np.testing.assert_allclose(result["summary"][1, :2], baseline[:, :2].mean(axis=0))
    np.testing.assert_allclose(
        result["summary"][2, :2],
        (expected - baseline)[:, :2].mean(axis=0),
        atol=1e-14,  # Independent Pearson implementations differ at roundoff near zero.
    )
    np.testing.assert_array_equal(result["summary"][3:], [[3, 3, 1, 0], [3, 3, 0, 0]])
    assert np.isnan(result["summary"][:3, 2:]).all()
    assert np.isfinite(result["canonical"][1, 2])
    assert np.isnan(result["canonical"][:, 3]).all()


def test_parameter_variability_uses_values_not_categorical_ids(library):
    result = reliability.compare_hrfs(library, [[1, 1], [2, 1]], ["a", "b"])
    names = [
        "response_delay",
        "undershoot_delay",
        "response_dispersion",
        "undershoot_dispersion",
        "response_undershoot_ratio",
        "onset_delay",
        "peak_time",
    ]
    expected = library.parameter_table.loc[[1, 2], names].to_numpy().std(axis=0, ddof=1)
    assert result["parameter_names"] == names
    np.testing.assert_allclose(result["parameter_sd"][:, 0], expected)
    np.testing.assert_array_equal(result["parameter_sd"][:, 1], np.zeros(7))
    assert result["pairwise"][0, 1] == pytest.approx(1)
    assert result["pair_delta"][0, 1] > 0


@pytest.mark.parametrize(
    "ids,sessions",
    [
        ([[0]], ["a"]),
        ([[0], [1]], ["a", "a"]),
        ([[0], [1]], ["a"]),
        ([[0], [3]], ["a", "b"]),
        ([[0], [0.5]], ["a", "b"]),
        ([[0], [-2]], ["a", "b"]),
        ([[0], [np.inf]], ["a", "b"]),
    ],
)
def test_invalid_session_selections_are_rejected(library, ids, sessions):
    with pytest.raises(ValueError):
        reliability.compare_hrfs(library, ids, sessions)


def test_library_indices_map_missing_selections_to_minus_one():
    ids = reliability.library_indices([0, np.nan, -1, 2.0], 3)
    np.testing.assert_array_equal(ids, [0, -1, -1, 2])
    assert ids.dtype.kind == "i"


def test_finite_mean_ignores_nonfinite_values_and_keeps_empty_columns_nan():
    values = np.array([[1.0, np.nan, np.inf], [3.0, np.nan, np.nan]])
    np.testing.assert_allclose(
        reliability.finite_mean(values), [2.0, np.nan, np.nan], equal_nan=True
    )
