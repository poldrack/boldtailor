import warnings

import numpy as np
import pandas as pd
import pytest
from nilearn.glm import compute_contrast
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm

from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.model import ModelSpec


@pytest.fixture
def single_run_problem():
    rng = np.random.default_rng(7)
    events = pd.DataFrame(
        {
            "onset": [0.0, 8.0, 16.0, 24.0, 32.0, 40.0],
            "duration": np.ones(6),
            "trial_type": ["face", "house", "face", "house", "face", "house"],
        }
    )
    frame_times = np.arange(30) * 2.0
    design = make_first_level_design_matrix(
        frame_times,
        events=events,
        hrf_model="glover",
        drift_model=None,
        min_onset=-24.0,
    )
    beta = np.array([[2.0, 1.0], [0.5, 1.5], [10.0, 12.0]])
    signals = design.to_numpy() @ beta + rng.normal(0.0, 0.05, (30, 2))
    model = ModelSpec(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        drift_model=None,
        noise_model="ols",
    )
    return signals, events, design, model


def _nilearn_contrast(signals, design, noise_model):
    labels, regression_results = run_glm(
        signals,
        design.to_numpy(),
        noise_model=noise_model,
    )
    vector = expression_to_contrast_vector("face - house", design.columns)
    return compute_contrast(
        labels,
        regression_results,
        vector,
        stat_type="t",
    )


def _nilearn_original_space_r2(signals, design, noise_model):
    matrix = design.to_numpy()
    labels, regression_results = run_glm(
        signals,
        matrix,
        noise_model=noise_model,
    )
    prediction = np.empty_like(signals)
    for label, result in regression_results.items():
        prediction[:, labels == label] = matrix @ result.theta
    residual_sum = np.sum((signals - prediction) ** 2, axis=0)
    total_sum = np.sum((signals - signals.mean(axis=0)) ** 2, axis=0)
    return 1.0 - residual_sum / total_sum


def test_fit_matches_nilearn_ols_contrast(single_run_problem):
    signals, events, design, model = single_run_problem

    result = fit(from_arrays(signals, events, tr=2.0), model)
    expected = _nilearn_contrast(signals, design, "ols")

    np.testing.assert_allclose(result.effect("face_gt_house"), expected.effect_size())
    np.testing.assert_allclose(
        result.variance("face_gt_house"), expected.effect_variance()
    )
    np.testing.assert_allclose(result.stat("face_gt_house"), expected.stat())
    np.testing.assert_allclose(result.z_score("face_gt_house"), expected.z_score())
    np.testing.assert_allclose(
        result.one_sided_p_value("face_gt_house"),
        expected.p_value(),
    )
    assert result.contrast_names == ("face_gt_house",)


def test_fit_matches_nilearn_ar1_contrast(single_run_problem):
    signals, events, design, _ = single_run_problem
    model = ModelSpec(
        contrasts={"face_gt_house": "face - house"},
        drift_model=None,
        noise_model="ar1",
    )

    result = fit(from_arrays(signals, events, tr=2.0), model)
    expected = _nilearn_contrast(signals, design, "ar1")

    np.testing.assert_allclose(result.effect("face_gt_house"), expected.effect_size())
    np.testing.assert_allclose(
        result.variance("face_gt_house"), expected.effect_variance()
    )
    np.testing.assert_allclose(result.stat("face_gt_house"), expected.stat())


@pytest.mark.parametrize("noise_model", ["ols", "ar1"])
def test_fit_accepts_zero_sst_features_without_inference_warnings(
    single_run_problem,
    noise_model,
):
    signals, events, design, _ = single_run_problem
    varying = signals[:, :1]
    mixed = np.column_stack(
        (
            varying[:, 0],
            np.zeros(len(signals)),
            np.full(len(signals), 5.0),
        )
    )
    model = ModelSpec(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        drift_model=None,
        noise_model=noise_model,
    )
    expected = _nilearn_contrast(varying, design, noise_model)
    expected_r2 = _nilearn_original_space_r2(varying, design, noise_model)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fit(from_arrays(mixed, events, tr=2.0), model)

    name = "face_gt_house"
    np.testing.assert_allclose(result.effect(name)[0], expected.effect_size()[0])
    np.testing.assert_allclose(
        result.variance(name)[0],
        expected.effect_variance()[0],
    )
    np.testing.assert_allclose(result.stat(name)[0], expected.stat()[0])
    np.testing.assert_allclose(result.z_score(name)[0], expected.z_score()[0])
    np.testing.assert_allclose(
        result.one_sided_p_value(name)[0],
        expected.p_value()[0],
    )
    for values in (result.run_r2[0], result.r2):
        np.testing.assert_allclose(values[0], expected_r2)
        assert np.isnan(values[1:]).all()


def test_fit_rejects_run_without_positive_residual_degrees_of_freedom(
    single_run_problem,
):
    signals, events, _, _ = single_run_problem
    saturated_times = np.array([0.0, 2.0])
    saturated_events = pd.DataFrame(
        {
            "onset": [0.0],
            "duration": [1.0],
            "trial_type": ["face"],
        }
    )
    saturated_design = make_first_level_design_matrix(
        saturated_times,
        events=saturated_events,
        hrf_model="glover",
        drift_model=None,
        min_onset=-24.0,
    )
    saturated_signals = saturated_design.to_numpy() @ np.array([[1.0], [2.0]])
    model = ModelSpec(
        contrasts={"face": {"face": 1.0}},
        drift_model=None,
        noise_model="ols",
    )

    assert saturated_design.shape == (2, 2)
    assert np.linalg.matrix_rank(saturated_design) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            ValueError,
            match="run 1.*residual degrees of freedom.*0",
        ):
            fit(
                from_arrays(
                    (signals[:, :1], saturated_signals),
                    (events, saturated_events),
                    frame_times=(np.arange(30) * 2.0, saturated_times),
                ),
                model,
            )


def test_fit_returns_strictly_immutable_arrays(single_run_problem):
    signals, events, design, model = single_run_problem

    result = fit(from_arrays(signals, events, tr=2.0), model)
    coefficients = np.linalg.lstsq(design.to_numpy(), signals, rcond=None)[0]
    prediction = design.to_numpy() @ coefficients
    residual_sum = np.sum((signals - prediction) ** 2, axis=0)
    total_sum = np.sum((signals - signals.mean(axis=0)) ** 2, axis=0)
    expected_r2 = 1.0 - residual_sum / total_sum

    assert result.r2.shape == (2,)
    np.testing.assert_allclose(result.r2, expected_r2)
    assert len(result.run_r2) == 1
    np.testing.assert_allclose(result.run_r2[0], expected_r2)
    arrays = (
        result.r2,
        result.run_r2[0],
        result.effect("face_gt_house"),
        result.variance("face_gt_house"),
        result.stat("face_gt_house"),
        result.z_score("face_gt_house"),
        result.one_sided_p_value("face_gt_house"),
    )
    for values in arrays:
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)


def test_fit_owns_designs_and_exposes_immutable_provenance(single_run_problem):
    signals, events, _, model = single_run_problem
    result = fit(from_arrays(signals, events, tr=2.0), model)

    returned = result.design_matrices[0]
    returned.iloc[0, 0] = -99.0

    assert result.design_matrices[0].iloc[0, 0] != -99.0
    assert result.design_provenance[0]["excluded_event_count"] == 0
    assert result.design_provenance[0]["min_onset_cutoff"] == -24.0
    with pytest.raises(TypeError):
        result.design_provenance[0]["excluded_event_count"] = 3


def test_fit_does_not_mutate_inputs(single_run_problem):
    signals, events, _, model = single_run_problem
    original_signals = signals.copy()
    original_events = events.copy(deep=True)

    fit(from_arrays(signals, events, tr=2.0), model)

    np.testing.assert_array_equal(signals, original_signals)
    pd.testing.assert_frame_equal(events, original_events)


@pytest.mark.parametrize(
    "contrast",
    ["not_a_column", {"not_a_column": 1.0}],
)
def test_fit_rejects_unknown_semantic_contrast(single_run_problem, contrast):
    signals, events, _, _ = single_run_problem
    model = ModelSpec(
        contrasts={"missing": contrast},
        drift_model=None,
        noise_model="ols",
    )

    with pytest.raises(ValueError, match="run 0.*contrast 'missing'"):
        fit(from_arrays(signals, events, tr=2.0), model)
