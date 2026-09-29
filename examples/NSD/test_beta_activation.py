"""Descriptive beta-versus-zero maps pool independent trial observations."""

import importlib

import numpy as np
import pytest
from scipy.stats import ttest_1samp


def activation():
    try:
        return importlib.import_module("examples.NSD.beta_activation")
    except ModuleNotFoundError as error:
        pytest.fail(f"Missing example beta activation analysis: {error}")


def test_trial_weighted_t_test_matches_scipy_and_preserves_inputs():
    runs = [
        np.array([[1, -2, 0], [2, -5, 2]], dtype=np.float32),
        np.array([[4, -6, -1], [7, -1, 3], [8, -3, 1]], dtype=np.float32),
    ]
    before = [run.copy() for run in runs]
    result = activation().beta_activation(runs)
    pooled = np.concatenate(runs).astype(float)
    expected = ttest_1samp(pooled, 0, axis=0)
    np.testing.assert_allclose(result["mean_beta"], [4.4, -3.4, 1])
    np.testing.assert_allclose(result["t"], expected.statistic)
    np.testing.assert_allclose(result["p_uncorrected"], expected.pvalue)
    np.testing.assert_array_equal(result["n_trials"], [5, 5, 5])
    np.testing.assert_array_equal(result["df"], [4, 4, 4])
    for original, run in zip(before, runs, strict=True):
        np.testing.assert_array_equal(run, original)


def test_missing_values_and_degenerate_columns_are_explicit():
    runs = [
        np.array(
            [
                [1, np.nan, 4, 0, 3],
                [3, np.inf, np.nan, 0, 3],
                [np.nan, np.nan, np.nan, 0, 3],
            ]
        )
    ]
    result = activation().beta_activation(runs)
    np.testing.assert_allclose(result["mean_beta"], [2, np.nan, 4, 0, 3])
    np.testing.assert_allclose(result["n_trials"], [2, np.nan, 1, 3, 3])
    np.testing.assert_allclose(result["df"], [1, np.nan, 0, 2, 2])
    assert result["t"][0] == pytest.approx(2)
    assert result["p_uncorrected"][0] == pytest.approx(ttest_1samp([1, 3], 0).pvalue)
    assert np.isnan(result["t"][1:]).all()
    assert np.isnan(result["p_uncorrected"][1:]).all()


def test_large_offsets_use_centered_sample_variance():
    values = 1e10 + np.array([[1, -2], [2, 0], [3, 2]], dtype=float)
    result = activation().beta_activation([values[:1], values[1:]])
    expected = ttest_1samp(values, 0, axis=0)
    np.testing.assert_allclose(result["t"], expected.statistic)


@pytest.mark.parametrize("runs", [[], [np.ones(3)], [np.ones((2, 3)), np.ones((2, 4))]])
def test_invalid_beta_dimensions_are_rejected(runs):
    with pytest.raises(ValueError, match="trial.*grayordinate|at least one"):
        activation().beta_activation(runs)


def test_activation_exports_preserve_map_names_axis_and_values(tmp_path):
    import nibabel as nib
    from boldtailor.publication import publish_artifact_set
    from examples.NSD import workflow_outputs

    brain = nib.cifti2.BrainModelAxis.from_surface([2, 0, 1], 4, name="CortexLeft")
    maps = {
        "mean_beta": np.array([2, -2, np.nan]),
        "t": np.array([3, -3, np.nan]),
        "p_uncorrected": np.array([0.04, 0.04, np.nan]),
        "n_trials": np.array([5, 5, np.nan]),
        "df": np.array([4, 4, np.nan]),
    }
    assert hasattr(
        workflow_outputs, "_activation_artifacts"
    ), "Activation exports are missing"
    artifacts = workflow_outputs._activation_artifacts(
        "sub-07", brain, {"CanonicalTrialOLS": maps}
    )
    paths = publish_artifact_set(tmp_path, artifacts)
    image = nib.load(paths[0])
    assert image.header.get_axis(0).name.tolist() == list(maps)
    assert image.header.get_axis(1) == brain
    np.testing.assert_allclose(image.get_fdata(), np.stack(list(maps.values())))
    assert "CanonicalTrialOLS_stat-activation.dscalar.nii" in paths[0].name
