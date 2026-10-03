from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from boldtailor.fractional_ridge import score_fraction_candidates
from boldtailor.ridge_selection import score_ridge_candidates
from tests.test_hrf_glm import hrf_glm_problem  # noqa: F401  (fixture)


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
        exposed = result.design.matrix(0, 1).copy()
        expected = exposed.copy()
        exposed[:] = 0
        np.testing.assert_array_equal(result.design.matrix(0, 1), expected)
    else:
        exposed = result.design.matrices[0]
        expected = exposed.copy(deep=True)
        exposed.iloc[:, :] = 0
        pd.testing.assert_frame_equal(result.design.matrices[0], expected)


def test_shared_trial_design_owns_its_input_table():
    from boldtailor.single_trial_results import SharedTrialDesign

    matrix = pd.DataFrame({"trial": [1.0, 2.0], "constant": [1.0, 1.0]})
    expected = matrix.copy(deep=True)
    design = SharedTrialDesign(_matrices=(matrix,))
    matrix.iloc[:, :] = 99
    pd.testing.assert_frame_equal(design.matrices[0], expected)


def test_selected_trial_design_owns_arrays_and_rebuilt_matrices(selected_fixture):
    from boldtailor.single_trial_results import SelectedTrialDesign

    _, selection = selected_fixture
    matrix = np.arange(6.0).reshape(3, 2)
    expected = matrix.copy()
    ids = np.array([1, -1])
    design = SelectedTrialDesign(
        hrf_indices=ids,
        design_fingerprint="0" * 64,
        selection_provenance=selection.provenance,
        _rebuild=lambda run, hrf_id: matrix,
    )
    returned = design.matrix(0, 1)
    matrix[:] = 99
    ids[:] = 0
    np.testing.assert_array_equal(returned, expected)
    np.testing.assert_array_equal(design.hrf_indices, [1, -1])
    assert not design.matrix(0, 1).flags.writeable
    assert not design.hrf_indices.flags.writeable
    assert design.design_fingerprint == "0" * 64


def test_custom_hrf_has_a_shared_trial_design(selected_fixture):
    from boldtailor.single_trial import fit_single_trials
    from boldtailor.single_trial_results import SharedTrialDesign

    data, selection = selected_fixture
    result = fit_single_trials(data, hrf=selection.library.candidates[1])
    assert type(result.design) is SharedTrialDesign
    assert len(result.design.matrices) == data.n_runs


def _glm_instances(problem):
    from boldtailor.fit import fit, task_delta_r2

    data, model, selection, _ = problem
    canonical = replace(model, hrf_model="spm")
    full = fit(data, canonical)
    return {
        "AnalysisResult": lambda: full,
        "TaskDeltaR2Result": lambda: task_delta_r2(data, canonical, full),
        "HrfAnalysisResult": lambda: fit(
            data, model, hrf_selection=selection, feature_signature="axis-v1"
        ),
    }


def _selected_instances(selected_fixture):
    from boldtailor.hrf_selection import evaluate_hrf_split
    from boldtailor.single_trial import fit_selected_hrfs, fit_single_trials

    data, selection = selected_fixture
    signature = dict(feature_signature="ordered-axis")
    selected = lambda: fit_selected_hrfs(data, selection=selection, **signature)
    return {
        "HrfSelectionResult": lambda: selection,
        "HrfEvaluationResult": lambda: evaluate_hrf_split(
            data, library=selection.library, train_runs=[0, 1], test_runs=[2]
        ),
        "SingleTrialResult": lambda: fit_single_trials(data),
        "SharedTrialDesign": lambda: fit_single_trials(data).design,
        "SelectedTrialDesign": lambda: selected().design,
    }


def _ridge_instances(ridge_problem):
    from boldtailor.fractional_ridge import select_ridge_fractions
    from boldtailor.ridge_selection import select_ridge_penalty
    from boldtailor.single_trial import fit_single_trials
    from boldtailor.trial_encoding import evaluate_trial_encoding

    data, predictors, _ = ridge_problem
    scores = [[0.1, 0.4, 0.2], [0.3, 0.1, 0.2]]
    return {
        "TrialEncodingResult": lambda: evaluate_trial_encoding(
            fit_single_trials(data).run_betas,
            predictors,
            train_runs=[0, 1, 2],
            test_runs=[3, 4],
        ),
        "RidgeSelection": lambda: select_ridge_penalty(scores, [0.0, 1.0]),
        "CandidateScores": lambda: score_ridge_candidates(
            data, predictors, alphas=[0.0, 1.0]
        ),
        "FractionSelection": lambda: select_ridge_fractions(scores, [1.0, 0.5]),
    }


_GROUPS = {
    "hrf_glm_problem": (
        _glm_instances,
        ["AnalysisResult", "TaskDeltaR2Result", "HrfAnalysisResult"],
    ),
    "selected_fixture": (
        _selected_instances,
        [
            "HrfSelectionResult",
            "HrfEvaluationResult",
            "SingleTrialResult",
            "SharedTrialDesign",
            "SelectedTrialDesign",
        ],
    ),
    "ridge_problem": (
        _ridge_instances,
        [
            "TrialEncodingResult",
            "RidgeSelection",
            "CandidateScores",
            "FractionSelection",
        ],
    ),
}
RESULT_CLASSES = [
    (fixture, name) for fixture, (_, names) in _GROUPS.items() for name in names
]


@pytest.fixture(params=RESULT_CLASSES, ids=[name for _, name in RESULT_CLASSES])
def result_instance(request):
    fixture, name = request.param
    builders = _GROUPS[fixture][0](request.getfixturevalue(fixture))
    instance = builders[name]()
    assert type(instance).__name__ == name
    return instance


def test_result_classes_reject_positional_construction(result_instance):
    cls = type(result_instance)
    assert all(item.kw_only for item in fields(cls))
    values = [getattr(result_instance, f.name) for f in fields(cls) if f.init]
    with pytest.raises(TypeError):
        cls(*values)


def _writable_copies(value):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, tuple) and value and isinstance(value[0], np.ndarray):
        return tuple(np.array(v, copy=True) for v in value)
    if isinstance(value, tuple) and value and isinstance(value[0], pd.DataFrame):
        return tuple(frame.copy(deep=True) for frame in value)
    return None


def _items(value):
    return value if isinstance(value, tuple) else (value,)


def _scribble(value):
    for item in _items(value):
        if isinstance(item, pd.DataFrame):
            item.iloc[:, :] = 7.0
        else:
            item[...] = 1 if item.dtype == bool else 7


def _assert_kept(kept, original):
    if isinstance(original, pd.DataFrame):
        pd.testing.assert_frame_equal(kept, original)
        return
    np.testing.assert_array_equal(kept, original)
    assert not kept.flags.writeable


def test_result_classes_own_their_arrays(result_instance):
    owned = 0
    for item in fields(result_instance):
        source = _writable_copies(getattr(result_instance, item.name))
        if source is None:
            continue
        expected = _writable_copies(source)
        rebuilt = replace(result_instance, **{item.name: source})
        _scribble(source)
        stored = getattr(rebuilt, item.name)
        for kept, original in zip(_items(stored), _items(expected), strict=True):
            _assert_kept(kept, original)
        owned += 1
    assert owned
