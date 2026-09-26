"""Spatial grouping must preserve trial/feature order and normalized estimates."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_selection import select_hrf
from boldtailor.single_trial import fit_single_trials


@pytest.fixture
def selected_fixture():
    library = HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )
    events = []
    signals = []
    times = []
    confounds = []
    for r in range(3):
        t = 0.775 + 1.6 * np.arange(75 + r * 3)
        e = pd.DataFrame(
            dict(
                onset=[30.1 + r, 8.2 + r, 53.3 + r],
                duration=[1.2, 3.0, 2.0],
                image=[4, 4, 5],
                response_time=[1.1, np.nan, 0.7],
                details=[{"tags": [r]}, None, None],
            )
        )
        n = pd.DataFrame(dict(motion=np.linspace(-1, 1, len(t))))
        columns = []
        for cid in [1, 0, 1, 2]:
            c = library.candidates[cid]
            x = np.column_stack(
                [
                    compute_regressor(
                        np.array([[o], [d], [1.0]]), "spm" if cid == 0 else c.kernel, t
                    )[0][:, 0]
                    for o, d in zip(e.onset, e.duration, strict=True)
                ]
            )
            columns.append(x @ np.array([2.9, 3.0, 3.1]) + n.motion * (r + 1) + 50)
        y = np.column_stack([*columns, np.ones(len(t)) * 100])
        events.append(e)
        times.append(t)
        confounds.append(n)
        signals.append(y)
    data = from_arrays(signals, events, frame_times=times, confounds=confounds)
    selection = select_hrf(data, library=library, feature_signature="ordered-axis")
    np.testing.assert_array_equal(selection.hrf_indices, [1, 0, 1, 2, -1])
    return data, selection


@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_grouped_betas_and_r_squared_match_augmented_ols(selected_fixture, alpha):
    from boldtailor.single_trial import fit_selected_hrfs

    data, selection = selected_fixture
    result = fit_selected_hrfs(
        data, selection=selection, ridge_alpha=alpha, feature_signature="ordered-axis"
    )
    np.testing.assert_array_equal(result.hrf_indices, selection.hrf_indices)
    sses = []
    nulls = []
    totals = []
    for r, (y, e, t, n) in enumerate(
        zip(data.signals, data.events, data.frame_times, data.confounds, strict=True)
    ):
        nuisance = np.column_stack([n, np.ones(len(t))])
        sse = []
        null = []
        total = []
        for v, cid in enumerate(selection.hrf_indices[:4]):
            c = selection.library.candidates[cid]
            x = np.column_stack(
                [
                    compute_regressor(
                        np.array([[o], [d], [1.0]]), "spm" if cid == 0 else c.kernel, t
                    )[0][:, 0]
                    for o, d in zip(e.onset, e.duration, strict=True)
                ]
            )
            xr = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0]
            penalty = np.column_stack(
                [
                    np.diag(np.sqrt(alpha) * np.linalg.norm(xr, axis=0)),
                    np.zeros((3, nuisance.shape[1])),
                ]
            )
            matrix = np.column_stack([x, nuisance])
            expected = np.linalg.lstsq(
                np.vstack([matrix, penalty]), np.r_[y[:, v], np.zeros(3)], rcond=None
            )[0]
            np.testing.assert_allclose(
                result.run_betas[r][:, v], expected[:3], atol=1e-11
            )
            np.testing.assert_array_equal(result.group_designs[r, int(cid)], matrix)
            sse.append(np.sum((y[:, v] - matrix @ expected) ** 2))
            null.append(
                np.sum(
                    (
                        y[:, v]
                        - nuisance @ np.linalg.lstsq(nuisance, y[:, v], rcond=None)[0]
                    )
                    ** 2
                )
            )
            total.append(np.sum((y[:, v] - y[:, v].mean()) ** 2))
        sses.append(sse)
        nulls.append(null)
        totals.append(total)
        assert np.isnan(result.run_betas[r][:, 4]).all()
    full = 1 - np.sum(sses, axis=0) / np.sum(totals, axis=0)
    nuisance = 1 - np.sum(nulls, axis=0) / np.sum(totals, axis=0)
    np.testing.assert_allclose(result.full_r2[:4], full, atol=1e-12)
    np.testing.assert_allclose(result.nuisance_r2[:4], nuisance, atol=1e-12)
    np.testing.assert_allclose(result.delta_r2[:4], full - nuisance, atol=1e-12)
    assert np.isnan(result.full_r2[4])
    assert result.trial_table.image.tolist() == [4, 4, 5] * 3
    assert result.trial_table.response_time.isna().sum() == 3


def test_identity_checks_and_apply_to_new_runs(selected_fixture):
    from boldtailor.single_trial import fit_selected_hrfs

    data, selection = selected_fixture
    for signature in (None, "different-axis"):
        with pytest.raises(ValueError, match="signature"):
            fit_selected_hrfs(data, selection=selection, feature_signature=signature)
    incompatible = replace(
        selection, library=HrfLibrary.from_parameters([[4, 12, 1, 1, 6, 0, 36]])
    )
    with pytest.raises(ValueError, match="library|fingerprint|identity"):
        fit_selected_hrfs(
            data, selection=incompatible, feature_signature="ordered-axis"
        )
    invalid = replace(selection, hrf_indices=[1, 0, 1, 99, -1])
    with pytest.raises(ValueError, match="HRF|hrf|identity"):
        fit_selected_hrfs(data, selection=invalid, feature_signature="ordered-axis")
    subset = from_arrays(
        data.signals[1],
        data.events[1],
        frame_times=data.frame_times[1],
        confounds=data.confounds[1],
    )
    result = fit_selected_hrfs(
        subset,
        selection=selection,
        feature_signature="ordered-axis",
        run_labels=["new-run"],
    )
    assert len(result.run_betas) == 1
    assert result.trial_table.run_label.unique().tolist() == ["new-run"]
    fewer = from_arrays(
        data.signals[1][:, :3],
        data.events[1],
        frame_times=data.frame_times[1],
        confounds=data.confounds[1],
    )
    with pytest.raises(ValueError, match="feature"):
        fit_selected_hrfs(fewer, selection=selection, feature_signature="ordered-axis")


def test_canonical_only_matches_legacy_including_per_run_constant(selected_fixture):
    from boldtailor.single_trial import fit_selected_hrfs

    data, _ = selected_fixture
    ys = [np.array(y) for y in data.signals]
    ys[1][:, 0] = 100
    data = from_arrays(
        ys, data.events, frame_times=data.frame_times, confounds=data.confounds
    )
    selection = select_hrf(data, library=HrfLibrary.from_parameters([]))
    for alpha in [0.0, 0.1]:
        actual = fit_selected_hrfs(data, selection=selection, ridge_alpha=alpha)
        legacy = fit_single_trials(data, ridge_alpha=alpha)
        for a, b in zip(actual.run_betas, legacy.run_betas, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(actual.full_r2, legacy.full_r2, atol=1e-12)
        np.testing.assert_allclose(actual.nuisance_r2, legacy.nuisance_r2, atol=1e-12)


def test_grouped_results_own_nested_metadata_and_record_designs(selected_fixture):
    from boldtailor.single_trial import fit_selected_hrfs

    data, selection = selected_fixture
    result = fit_selected_hrfs(
        data, selection=selection, feature_signature="ordered-axis"
    )
    for values in (
        *result.run_betas,
        result.hrf_indices,
        *result.group_designs.values(),
    ):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    groups = result.group_designs
    groups.pop((0, 1))
    assert (0, 1) in result.group_designs
    table = result.trial_table
    table.loc[0, "details"]["tags"].append(999)
    assert result.trial_table.loc[0, "details"] == {"tags": [0]}
    assert result.selection_provenance == selection.provenance
    info = result.provenance.to_dict()["activities"][-1]
    assert info["library_fingerprint"] == selection.library.fingerprint
    assert info["design_fingerprint"]
    assert info["hrf_assignment_fingerprint"]
