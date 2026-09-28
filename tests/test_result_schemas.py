from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from boldtailor.fractional_ridge import score_fraction_candidates
from boldtailor.ridge_selection import score_ridge_candidates


@pytest.mark.parametrize("fractional", [False, True])
def test_candidate_scores_name_the_grid_and_scientific_basis(ridge_problem, fractional):
    from boldtailor import ridge_results

    data, predictors, _ = ridge_problem
    function = score_fraction_candidates if fractional else score_ridge_candidates
    options = {"fractions": [0.3, 1, 0.7]} if fractional else {"alphas": [1, 0, 0.1]}
    result = function(data, predictors, **options)
    assert type(result) is ridge_results.CandidateScores
    assert result.regularization == (
        "fractional_ridge" if fractional else "normalized_ridge"
    )
    assert result.grid == ((1.0, 0.7, 0.3) if fractional else (0.0, 0.1, 1.0))
    assert result.cv_r2.shape == (3, data.n_features)
    np.testing.assert_allclose(
        result.cv_r2,
        1 - result.fold_sse.sum(axis=0) / result.fold_sst.sum(axis=0),
    )
    assert result.fold_hrf_indices.dtype == np.int64
    assert all(mask.dtype == bool for mask in result.trial_masks)
    with pytest.raises(ValueError, match="regularization"):
        replace(result, regularization="unknown")


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("fractional", [False, True])
def test_trial_results_share_science_and_name_their_design(
    selected_fixture, selected, fractional
):
    from boldtailor.single_trial import fit_single_trials, fit_selected_hrfs
    from boldtailor import single_trial_results as results

    data, selection = selected_fixture
    function = fit_selected_hrfs if selected else fit_single_trials
    options = (
        dict(selection=selection, feature_signature="ordered-axis") if selected else {}
    )
    options.update({"ridge_fraction": 0.5} if fractional else {"ridge_alpha": 0.1})
    result = function(data, **options)
    assert type(result) is results.SingleTrialResult
    expected_type = (
        results.SelectedTrialDesign if selected else results.SharedTrialDesign
    )
    assert type(result.design) is expected_type
    assert len(result.run_betas) == data.n_runs
    assert np.isnan(result.run_betas[0][:, -1]).all()
    if selected:
        np.testing.assert_array_equal(result.design.hrf_indices, selection.hrf_indices)
        assert result.design.selection_provenance == selection.provenance
        matrices = result.design.matrices
        assert (0, 1) in matrices
        matrices.clear()
        assert (0, 1) in result.design.matrices
    else:
        exposed = result.design.matrices[0]
        expected = exposed.copy(deep=True)
        exposed.iloc[:, :] = 0
        pd.testing.assert_frame_equal(result.design.matrices[0], expected)


def test_shared_trial_design_owns_its_input_table():
    from boldtailor.single_trial_results import SharedTrialDesign

    matrix = pd.DataFrame({"trial": [1.0, 2.0], "constant": [1.0, 1.0]})
    expected = matrix.copy(deep=True)
    design = SharedTrialDesign((matrix,))
    matrix.iloc[:, :] = 99
    pd.testing.assert_frame_equal(design.matrices[0], expected)


def test_selected_trial_design_owns_arrays_and_mapping(selected_fixture):
    from boldtailor.single_trial_results import SelectedTrialDesign

    _, selection = selected_fixture
    matrix = np.arange(6.0).reshape(3, 2)
    expected = matrix.copy()
    ids = np.array([1, -1])
    mapping = {(0, 1): matrix}
    design = SelectedTrialDesign(ids, mapping, selection.provenance)
    matrix[:] = 99
    ids[:] = 0
    mapping.clear()
    np.testing.assert_array_equal(design.matrices[0, 1], expected)
    np.testing.assert_array_equal(design.hrf_indices, [1, -1])
    assert not design.matrices[0, 1].flags.writeable
    assert not design.hrf_indices.flags.writeable


def test_custom_hrf_has_a_shared_trial_design(selected_fixture):
    from boldtailor.single_trial import fit_single_trials
    from boldtailor.single_trial_results import SharedTrialDesign

    data, selection = selected_fixture
    result = fit_single_trials(data, hrf=selection.library.candidates[1])
    assert type(result.design) is SharedTrialDesign
    assert len(result.design.matrices) == data.n_runs
