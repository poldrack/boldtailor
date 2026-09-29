"""Cross-session comparisons preserve spatial identity and matched observations."""

import importlib
import json

import nibabel as nib
import numpy as np
import pytest

from boldtailor.hrf_library import HrfLibrary
from boldtailor.publication import publish_artifact_set
from examples.NSD.hrf_artifacts import npz_artifact
from examples.NSD.single_trial_artifacts import (
    json_artifact, scalar_artifact, table_artifact,
)


def api(module):
    try:
        return importlib.import_module("examples.NSD.multisession_" + module)
    except ModuleNotFoundError:
        pytest.fail(f"Multi-session {module} is not implemented")


@pytest.fixture
def saved_sessions(tmp_path):
    brain = nib.cifti2.BrainModelAxis.from_surface([0, 2, 3], 5, "CortexLeft")
    library = HrfLibrary.from_parameters([[3, 10, .5, .5, 2, 0, 36]])
    sessions = ["ses-nsd10", "ses-nsd11", "ses-nsd12"]
    for i, session in enumerate(sessions):
        base = f"sub-07/{session}/func/sub-07_{session}_task-nsdcore"
        meta = dict(settings=dict(subject="sub-07", session=session, ridge_mode="off"),
                    library_fingerprint=library.fingerprint)
        artifacts = [
            json_artifact(base + "_desc-notebook_metadata.json", meta),
            table_artifact(base + "_desc-notebookHRF_library.tsv", library.parameter_table),
            npz_artifact(base + "_desc-notebookHRF_library.npz",
                         times=library.times, curves=library.curves),
        ]

        def add(descriptor, stat, values, labels):
            artifacts.append(scalar_artifact(
                base + f"_space-fsLR_den-91k_desc-notebook{descriptor}_stat-{stat}.dscalar.nii",
                brain, values, labels,
            ))

        ids = [i % 2, 1, np.nan]
        add("HRFAll", "selection", [ids, [.1, .2, np.nan], [.1, .1, np.nan], [0, .1, np.nan]],
            ["hrf_id", "selected_cv_r2", "canonical_cv_r2", "delta_cv_r2"])
        for optimized, prefix in enumerate(("Canonical", "Optimized")):
            shift = optimized * (i + 1)
            add(prefix + "TrialOLS", "activation",
                [[2 + shift, -2 - shift, np.nan], [4 + shift, -4 - shift, np.nan],
                 [.1, .1, np.nan], [20, 20, np.nan], [19, 19, np.nan]],
                ["mean_beta", "t", "p_uncorrected", "n_trials", "df"])
            add(prefix + "TrialOLS", "rsquared",
                [[.8, .8, np.nan], [.7 - .01 * shift, .7, np.nan],
                 [.1 + .01 * shift, .1, np.nan]],
                ["full_r2", "confounds_r2", "task_delta_r2"])
            add(prefix + "TrialOLS", "rtcorrelation",
                [[-.2 - .1 * shift, .2, np.nan], [.1, .1, np.nan], [.1, .1, np.nan]],
                ["all_runs", "odd_runs", "even_runs"])
        publish_artifact_set(tmp_path, artifacts)
    return tmp_path, sessions, library, brain


def test_loading_and_hrf_correlations_use_full_curves(saved_sessions):
    root, sessions, library, brain = saved_sessions
    loaded = api("inputs").load_sessions(root, "sub-07", sessions, estimators=["OLS"])
    assert loaded["brain"] == brain
    result = api("analysis").analyze_sessions(loaded)
    expected = np.corrcoef(library.curves[0], library.curves[1])[0, 1]
    np.testing.assert_allclose(result["hrf"]["pairwise"][:, 0], [expected, 1, expected])
    np.testing.assert_allclose(result["hrf"]["pairwise"][:, 1], 1)
    assert np.isnan(result["hrf"]["pairwise"][:, 2]).all()
    summary = result["beta"]["OLS"]
    names = result["metrics"]
    np.testing.assert_allclose(summary["difference_mean"][names.index("mean_beta"), :2], [2, -2])
    np.testing.assert_allclose(summary["difference_mean"][names.index("rt_r"), 0], -.2, atol=1e-7)
    np.testing.assert_allclose(summary["difference_mean"][names.index("rt_abs_r"), 0], .2, atol=1e-7)
    np.testing.assert_array_equal(summary["valid_sessions"][:, 2], 0)


def test_paired_summary_uses_same_sessions_for_both_models():
    canonical = np.array([[1., 2, np.nan], [3, np.nan, np.nan], [5, 8, np.nan]])
    optimized = np.array([[2., np.nan, np.nan], [np.nan, 4, np.nan], [8, 10, np.nan]])
    result = api("analysis").paired_summary(canonical, optimized)
    np.testing.assert_array_equal(result["valid_sessions"], [2, 1, 0])
    np.testing.assert_allclose(result["canonical_mean"], [3, np.nan, np.nan])
    np.testing.assert_allclose(result["optimized_mean"], [5, np.nan, np.nan])
    np.testing.assert_allclose(result["difference_mean"], [2, np.nan, np.nan])
    np.testing.assert_allclose(result["difference_sd"], [np.sqrt(2), np.nan, np.nan])
    np.testing.assert_allclose(result["positive_fraction"], [1, np.nan, np.nan])


def test_loader_rejects_misaligned_grayordinates(saved_sessions):
    root, sessions, _, _ = saved_sessions
    path = next((root / "sub-07" / sessions[1]).rglob("*CanonicalTrialOLS_stat-activation*"))
    image = nib.load(path)
    axis = nib.cifti2.BrainModelAxis.from_surface([2, 0, 3], 5, "CortexLeft")
    nib.save(nib.Cifti2Image(image.get_fdata(), header=nib.Cifti2Header.from_axes(
        (image.header.get_axis(0), axis))), path)
    with pytest.raises(ValueError, match="axis|grayordinate"):
        api("inputs").load_sessions(root, "sub-07", sessions, estimators=["OLS"])


def test_loader_rejects_wrong_scalar_names(saved_sessions):
    root, sessions, _, brain = saved_sessions
    path = next((root / "sub-07" / sessions[0]).rglob("*CanonicalTrialOLS_stat-rsquared*"))
    image = nib.load(path)
    nib.save(nib.Cifti2Image(image.get_fdata(), header=nib.Cifti2Header.from_axes(
        (nib.cifti2.ScalarAxis(["a", "b", "c"]), brain))), path)
    with pytest.raises(ValueError, match="map names"):
        api("inputs").load_sessions(root, "sub-07", sessions, estimators=["OLS"])


def test_loader_rejects_incompatible_libraries_and_missing_results(saved_sessions):
    root, sessions, _, _ = saved_sessions
    path = next((root / "sub-07" / sessions[0]).rglob("*_desc-notebook_metadata.json"))
    metadata = json.loads(path.read_text())
    metadata["library_fingerprint"] = "different"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="library"):
        api("inputs").load_sessions(root, "sub-07", sessions, estimators=["OLS"])
    with pytest.raises(FileNotFoundError, match="ses-nsd99"):
        api("inputs").load_sessions(root, "sub-07", ["ses-nsd99", "ses-nsd98"], estimators=["OLS"])


def test_at_least_two_distinct_sessions_are_required(saved_sessions):
    root, sessions, _, _ = saved_sessions
    for selected in ([sessions[0]], [sessions[0], sessions[0]]):
        with pytest.raises(ValueError, match="two.*unique"):
            api("inputs").load_sessions(root, "sub-07", selected, estimators=["OLS"])
