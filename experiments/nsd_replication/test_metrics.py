import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication.metrics import (
    columnwise_corr,
    decoding_accuracy,
    lagged_correlation,
    rdm,
    rdm_agreement,
    threshold_curves,
    voxel_reliability,
)


def test_columnwise_corr_nan_for_constant_or_nonfinite():
    a = np.array([[1.0, 1, 1, np.nan], [2, 1, 3, 1], [3, 1, 2, 2]])
    b = np.array([[1.0, 2, 3, 1], [2, 3, 1, 2], [3, 4, 2, 3]])
    out = columnwise_corr(a, b)
    assert out[0] == pytest.approx(1.0)
    assert np.isnan(out[1:2]).all() and np.isnan(out[3])
    assert out[2] == pytest.approx(np.corrcoef(a[:, 2], b[:, 2])[0, 1])


def test_reliability_hand_computed():
    reps = np.zeros((3, 4, 1))
    reps[:, :, 0] = [[1, 2, 3, 4], [1, 2, 3, 5], [2, 2, 3, 4]]
    expected = np.mean(
        [
            np.corrcoef(reps[k, :, 0], np.delete(reps, k, 0).mean(0)[:, 0])[0, 1]
            for k in range(3)
        ]
    )
    np.testing.assert_allclose(voxel_reliability(reps), [expected])


def test_reliability_constant_feature_is_nan():
    reps = np.ones((3, 4, 1))
    with np.errstate(all="raise"):
        assert np.isnan(voxel_reliability(reps)).all()


def test_threshold_curves_difference_from_composite():
    rel = {"b1": np.array([0.1, 0.3, 0.5]), "b4": np.array([0.3, 0.5, 0.7])}
    # Composite is [0.2, 0.4, 0.6]; thresholds sit between composite values.
    curves = threshold_curves(
        rel, np.array([True, True, False]), thresholds=np.array([0.15, 0.35])
    )
    row = curves.query("version == 'b4' and threshold == 0.15").iloc[0]
    assert row.n_features == 2 and row.mean_difference == pytest.approx(0.1)
    row = curves.query("version == 'b1' and threshold == 0.35").iloc[0]
    assert row.n_features == 1 and row.mean_difference == pytest.approx(-0.1)


def _lag_trials():
    return pd.DataFrame(
        {
            "session": "s",
            "run": ["r1", "r1", "r1", "r2"],
            "trial": [0, 1, 2, 0],
            "onset": [0, 4, 8, 0],
            "image": [1, 2, 3, 4],
        }
    )


def test_lagged_correlation_same_run_only():
    betas = np.array([[1.0, 0, 0], [1, 0.1, 0], [0, 0, 1], [1, 0, 0]])
    table = lagged_correlation(betas, _lag_trials(), np.ones(3, bool), max_lag=2)
    lag1 = table.query("lag == 1").iloc[0]
    assert lag1.n_pairs == 2
    expected = np.mean(
        [np.corrcoef(betas[0], betas[1])[0, 1], np.corrcoef(betas[1], betas[2])[0, 1]]
    )
    assert lag1.mean_r == pytest.approx(expected)


def test_lagged_correlation_uses_positional_rows_for_nondefault_index():
    betas = np.array([[1.0, 0, 0], [1, 0.1, 0], [0, 0, 1], [1, 0, 0]])
    shifted = _lag_trials().set_index(pd.Index([10, 11, 12, 13]))
    expected = lagged_correlation(betas, _lag_trials(), np.ones(3, bool), max_lag=2)
    got = lagged_correlation(betas, shifted, np.ones(3, bool), max_lag=2)
    pd.testing.assert_frame_equal(got, expected)


def test_lagged_correlation_length_mismatch_raises():
    betas = np.ones((3, 3))
    with pytest.raises(ValueError, match="trials"):
        lagged_correlation(betas, _lag_trials(), np.ones(3, bool))


def test_rdm_and_agreement():
    patterns = np.array([[1.0, 2, 3], [3, 2, 1], [1, 2, 4]])
    d = rdm(patterns)
    assert d[0, 1] == pytest.approx(2.0) and np.allclose(np.diag(d), 0)
    table = rdm_agreement({"sub-01": d, "sub-02": d})
    assert table.iloc[0].r == pytest.approx(1.0)


def test_decoding_perfect_on_separable_images():
    rng = np.random.default_rng(0)
    prototypes = rng.normal(size=(5, 20)) * 5
    reps = prototypes[None] + rng.normal(scale=0.1, size=(3, 5, 20))
    result = decoding_accuracy(reps, np.ones(20, bool))
    assert result["accuracy"] == 1.0 and result["chance"] == pytest.approx(0.2)
    assert result["n_classes"] == 5
    assert 0 <= result["n_unconverged"] <= 3  # counts folds hitting max_iter


def test_decoding_too_few_features_returns_nan():
    reps = np.zeros((3, 5, 20))
    mask = np.zeros(20, bool)
    mask[:5] = True
    assert np.isnan(decoding_accuracy(reps, mask)["accuracy"])
