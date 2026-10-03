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
