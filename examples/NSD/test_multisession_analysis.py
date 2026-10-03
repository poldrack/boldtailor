"""Cross-session comparisons preserve spatial identity and matched observations."""

import importlib
import json

import nibabel as nib
import numpy as np
import pytest


def api(module):
    try:
        return importlib.import_module("examples.NSD.multisession_" + module)
    except ModuleNotFoundError:
        pytest.fail(f"Multi-session {module} is not implemented")


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
    np.testing.assert_allclose(
        summary["difference_mean"][names.index("mean_beta"), :2], [2, -2]
    )
    np.testing.assert_allclose(
        summary["difference_mean"][names.index("rt_r"), 0], -0.2, atol=1e-7
    )
    np.testing.assert_allclose(
        summary["difference_mean"][names.index("rt_abs_r"), 0], 0.2, atol=1e-7
    )
    np.testing.assert_array_equal(summary["valid_sessions"][:, 2], 0)


def test_paired_summary_uses_same_sessions_for_both_models():
    canonical = np.array([[1.0, 2, np.nan], [3, np.nan, np.nan], [5, 8, np.nan]])
    optimized = np.array([[2.0, np.nan, np.nan], [np.nan, 4, np.nan], [8, 10, np.nan]])
    result = api("analysis").paired_summary(canonical, optimized)
    np.testing.assert_array_equal(result["valid_sessions"], [2, 1, 0])
    np.testing.assert_allclose(result["canonical_mean"], [3, np.nan, np.nan])
    np.testing.assert_allclose(result["optimized_mean"], [5, np.nan, np.nan])
    np.testing.assert_allclose(result["difference_mean"], [2, np.nan, np.nan])
    np.testing.assert_allclose(result["difference_sd"], [np.sqrt(2), np.nan, np.nan])
    np.testing.assert_allclose(result["positive_fraction"], [1, np.nan, np.nan])


def test_mean_peak_time_averages_session_peaks_and_exports_counts(saved_sessions):
    root, sessions, _, brain = saved_sessions
    loaded = api("inputs").load_sessions(root, "sub-07", sessions, estimators=["OLS"])
    loaded["records"][0]["hrf_indices"][1] = -1
    loaded["records"][2]["hrf_indices"][1] = np.nan
    result = api("analysis").analyze_sessions(loaded)
    # These library curves peak at 5.1 s (SPM) and 2.5 s (custom), not 6/3 s.
    expected = [12.7 / 3, 2.5, np.nan]
    np.testing.assert_allclose(result["hrf"]["peak_time_mean"], expected)
    paths = api("outputs").save_multisession(root, loaded, result)
    image = nib.load(next(p for p in paths if "HRF_stat-peaktime" in p.name))
    assert image.header.get_axis(1) == brain
    assert image.header.get_axis(0).name.tolist() == [
        "mean_peak_time_seconds",
        "valid_sessions",
    ]
    np.testing.assert_allclose(image.get_fdata()[0], expected, atol=1e-6)
    np.testing.assert_array_equal(image.get_fdata()[1], [3, 1, 0])
    metadata = json.loads(
        next(p for p in paths if p.name.endswith("_metadata.json")).read_text()
    )
    assert metadata["hrf_peak_time"]["units"] == "seconds"
    assert metadata["hrf_peak_time"]["minimum_sessions"] == 1


def test_glm_effect_means_match_sessions_and_export_units(saved_sessions):
    root, sessions, _, brain = saved_sessions
    path = next((root / "sub-07" / sessions[2]).rglob("*OptimizedGLM_stat-effects*"))
    image = nib.load(path)
    values = image.get_fdata()
    values[1, 1] = np.nan
    nib.save(nib.Cifti2Image(values, header=image.header), path)
    loaded = api("inputs").load_sessions(root, "sub-07", sessions, estimators=["OLS"])
    result = api("analysis").analyze_sessions(loaded)
    assert result["glm_metrics"] == ("task", "response_time")
    expected = {
        "canonical_mean": [[11, -5, np.nan], [0.2, -0.3, np.nan]],
        "optimized_mean": [[13, -7, np.nan], [0.4, -0.6, np.nan]],
        "difference_mean": [[2, -2, np.nan], [0.2, -0.3, np.nan]],
        "valid_sessions": [[3, 3, 0], [3, 2, 0]],
    }
    for stat, values in expected.items():
        np.testing.assert_allclose(result["glm"][stat], values, atol=1e-7)
    paths = api("outputs").save_multisession(root, loaded, result)
    for stat, values in expected.items():
        saved = nib.load(next(p for p in paths if f"GLM_stat-{stat}." in p.name))
        assert saved.header.get_axis(1) == brain
        assert saved.header.get_axis(0).name.tolist() == ["task", "response_time"]
        np.testing.assert_allclose(saved.get_fdata(), values, atol=1e-7)
    meta = json.loads(
        next(p for p in paths if p.name.endswith("_metadata.json")).read_text()
    )
    assert meta["glm_units"] == {
        "task": "native signal units",
        "response_time": "native signal units per second",
    }
    sources = {row["path"] for row in meta["sources"]}
    assert str(path) in sources
    assert len([p for p in sources if "GLM_stat-effects" in p]) == 6


def test_loader_rejects_misaligned_grayordinates(saved_sessions):
    root, sessions, _, _ = saved_sessions
    path = next(
        (root / "sub-07" / sessions[1]).rglob("*CanonicalTrialOLS_stat-activation*")
    )
    image = nib.load(path)
    axis = nib.cifti2.BrainModelAxis.from_surface([2, 0, 3], 5, "CortexLeft")
    nib.save(
        nib.Cifti2Image(
            image.get_fdata(),
            header=nib.Cifti2Header.from_axes((image.header.get_axis(0), axis)),
        ),
        path,
    )
    with pytest.raises(ValueError, match="axis|grayordinate"):
        api("inputs").load_sessions(root, "sub-07", sessions, estimators=["OLS"])


@pytest.mark.parametrize(
    "pattern", ["*CanonicalTrialOLS_stat-rsquared*", "*CanonicalGLM_stat-effects*"]
)
def test_loader_rejects_wrong_scalar_names(saved_sessions, pattern):
    root, sessions, _, brain = saved_sessions
    path = next((root / "sub-07" / sessions[0]).rglob(pattern))
    image = nib.load(path)
    nib.save(
        nib.Cifti2Image(
            image.get_fdata(),
            header=nib.Cifti2Header.from_axes(
                (nib.cifti2.ScalarAxis(["a", "b", "c"]), brain)
            ),
        ),
        path,
    )
    with pytest.raises(ValueError, match="map names"):
        api("inputs").load_sessions(root, "sub-07", sessions, estimators=["OLS"])


def test_loader_rejects_incompatible_libraries_and_missing_results(saved_sessions):
    root, sessions, _, _ = saved_sessions
    path = next((root / "sub-07" / sessions[0]).rglob("*_desc-boldtailor_metadata.json"))
    metadata = json.loads(path.read_text())
    metadata["library_fingerprint"] = "different"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="library"):
        api("inputs").load_sessions(root, "sub-07", sessions, estimators=["OLS"])
    with pytest.raises(FileNotFoundError, match="ses-nsd99"):
        api("inputs").load_sessions(
            root, "sub-07", ["ses-nsd99", "ses-nsd98"], estimators=["OLS"]
        )


def test_at_least_two_distinct_sessions_are_required(saved_sessions):
    root, sessions, _, _ = saved_sessions
    for selected in ([sessions[0]], [sessions[0], sessions[0]]):
        with pytest.raises(ValueError, match="two.*unique"):
            api("inputs").load_sessions(root, "sub-07", selected, estimators=["OLS"])


def test_loader_rejects_different_global_alpha_selection_percentiles(saved_sessions):
    root, sessions, _, _ = saved_sessions
    for i, session in enumerate(sessions):
        folder = root / "sub-07" / session / "func"
        for source in folder.glob("*TrialOLS_*.dscalar.nii"):
            source.with_name(
                source.name.replace("TrialOLS", "TrialRidgeCV")
            ).write_bytes(source.read_bytes())
        path = next(folder.glob("*_desc-boldtailor_metadata.json"))
        meta = json.loads(path.read_text())
        meta["settings"].update(ridge_mode="cv", ridge_percentile=50 if i == 0 else 90)
        path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="ridge_percentile"):
        api("inputs").load_sessions(root, "sub-07", sessions, estimators=["RidgeCV"])


def test_sessions_with_different_task_models_cannot_be_pooled(saved_sessions):
    root, sessions, _, _ = saved_sessions
    inputs = api("inputs")
    first, other = (
        inputs.load_one_session(root, "sub-07", s, ["OLS"]) for s in sessions[:2]
    )
    first["metadata"]["task_model_fingerprint"] = "1" * 64
    other["metadata"]["task_model_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="task_model_fingerprint"):
        inputs._compatible(first, other, ["OLS"])


def test_sessions_with_different_hrf_normalization_cannot_be_pooled(saved_sessions):
    root, sessions, _, _ = saved_sessions
    inputs = api("inputs")
    first, other = (
        inputs.load_one_session(root, "sub-07", s, ["OLS"]) for s in sessions[:2]
    )
    first["metadata"]["hrf_normalization"] = "peak_one_event_response"
    other["metadata"]["hrf_normalization"] = "discrete_sum_one"
    with pytest.raises(ValueError, match="hrf_normalization"):
        inputs._compatible(first, other, ["OLS"])
