import warnings

import numpy as np
import pandas as pd
import pytest
from nilearn.glm import compute_contrast
from nilearn.glm.contrasts import expression_to_contrast_vector
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.first_level.hemodynamic_models import glover_hrf

from boldtailor.data import from_arrays
from boldtailor.fit import fit, task_delta_r2
from boldtailor.model import ModelSpec
from boldtailor.provenance import RunSources, SourceRef
from boldtailor.results import make_task_delta_r2_result


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


def _complete_sources() -> tuple[RunSources, ...]:
    return (
        RunSources(
            signal=SourceRef(
                role="signal",
                uri="sub-01/func/sub-01_task-localizer_run-01_bold.tsv",
                media_type="text/tab-separated-values",
                byte_size=2048,
                modified_at="2026-08-08T12:00:00Z",
            ),
            events=SourceRef(
                role="events",
                uri="sub-01/func/sub-01_task-localizer_run-01_events.tsv",
                media_type="text/tab-separated-values",
                byte_size=512,
                modified_at="2026-08-08T12:01:00Z",
            ),
        ),
    )


def _delta_r2_sources(*, signal_byte_size=4096) -> tuple[RunSources, ...]:
    return (
        RunSources(
            signal=SourceRef(
                role="signal",
                uri="sub-01/func/sub-01_task-delta_run-01_bold.tsv",
                media_type="text/tab-separated-values",
                byte_size=signal_byte_size,
                modified_at="2026-08-09T12:00:00Z",
            ),
            events=SourceRef(
                role="events",
                uri="sub-01/func/sub-01_task-delta_run-01_events.tsv",
                media_type="text/tab-separated-values",
                byte_size=1024,
                modified_at="2026-08-09T12:01:00Z",
            ),
            confounds=SourceRef(
                role="confounds",
                uri="sub-01/func/sub-01_task-delta_run-01_confounds.tsv",
                media_type="text/tab-separated-values",
                byte_size=2048,
                modified_at="2026-08-09T12:02:00Z",
            ),
        ),
    )


@pytest.fixture(scope="module")
def delta_r2_problem():
    rng = np.random.default_rng(20260809)
    n_scans = 100
    frame_times = np.arange(n_scans) * 1.5
    onsets = np.arange(3.0, 138.0, 6.0)
    events = pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.full(onsets.shape, 1.5),
            "trial_type": np.resize(["face", "house"], onsets.shape),
        }
    )
    confounds = pd.DataFrame(
        {"motion_x": np.sin(np.linspace(0.0, 4.0 * np.pi, n_scans))}
    )
    model = ModelSpec(
        contrasts={"face_gt_house": "face - house"},
        confounds=("motion_x",),
        drift_model="cosine",
        high_pass=0.01,
        noise_model="ar1",
    )
    design = make_first_level_design_matrix(
        frame_times,
        events=events,
        hrf_model=model.hrf_model,
        drift_model=model.drift_model,
        high_pass=model.high_pass,
        drift_order=model.drift_order,
        add_regs=confounds,
        min_onset=model.min_onset,
        oversampling=model.oversampling,
    )
    coefficients = np.zeros((design.shape[1], 4))
    coefficients[design.columns.get_loc("face")] = [2.5, 2.0, 3.0, 2.2]
    coefficients[design.columns.get_loc("house")] = [-1.5, -2.0, -1.0, -2.5]
    coefficients[design.columns.get_loc("motion_x")] = [0.5, -0.4, 0.3, -0.2]
    coefficients[design.columns.get_loc("constant")] = [10.0, 12.0, 8.0, 9.0]
    signals = design.to_numpy() @ coefficients
    signals += rng.normal(0.0, 0.15, signals.shape)
    data = from_arrays(
        signals,
        events,
        frame_times=frame_times,
        confounds=confounds,
        sources=_delta_r2_sources(),
    )
    return data, model, fit(data, model)


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


def _nilearn_pooled_ols_r2(signals, designs):
    residual_sums = []
    total_sums = []
    for observations, design in zip(signals, designs, strict=True):
        matrix = design.to_numpy()
        labels, regression_results = run_glm(
            observations,
            matrix,
            noise_model="ols",
        )
        prediction = np.empty_like(observations)
        for label, result in regression_results.items():
            prediction[:, labels == label] = matrix @ result.theta
        residual_sums.append(np.sum((observations - prediction) ** 2, axis=0))
        total_sums.append(
            np.sum((observations - observations.mean(axis=0)) ** 2, axis=0)
        )
    return 1.0 - np.sum(residual_sums, axis=0) / np.sum(total_sums, axis=0)


def _full_ols_designs(data, model):
    return tuple(
        make_first_level_design_matrix(
            frame_times,
            events=events,
            hrf_model=model.hrf_model,
            drift_model=model.drift_model,
            high_pass=model.high_pass,
            drift_order=model.drift_order,
            add_regs=confounds.loc[:, list(model.confounds)],
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
        for frame_times, events, confounds in zip(
            data.frame_times,
            data.events,
            data.confounds,
            strict=True,
        )
    )


def _nuisance_ols_designs(data, model):
    return tuple(
        make_first_level_design_matrix(
            frame_times,
            events=None,
            hrf_model=None,
            drift_model=model.drift_model,
            high_pass=model.high_pass,
            drift_order=model.drift_order,
            add_regs=confounds.loc[:, list(model.confounds)],
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
        for frame_times, confounds in zip(
            data.frame_times,
            data.confounds,
            strict=True,
        )
    )


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


def test_fit_returns_readonly_arrays(single_run_problem):
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
        assert type(values) is np.ndarray
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 0


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


def test_fit_extends_input_provenance_and_exposes_result_provenance(single_run_problem):
    signals, events, _, model = single_run_problem
    data = from_arrays(signals, events, tr=2.0, sources=_complete_sources())
    before = data.provenance.to_dict()

    first = fit(data, model)
    second = fit(data, model)

    assert data.provenance.to_dict() == before
    assert first.provenance.to_dict()["sources"] == before["sources"]
    assert first.provenance.execution_id != data.provenance.execution_id
    assert second.provenance.execution_id != first.provenance.execution_id
    assert len(first.provenance.activities) == len(data.provenance.activities) + 1
    assert [activity["name"] for activity in first.provenance.activities] == [
        "normalize",
        "fit",
    ]


def test_fit_analysis_fingerprint_depends_on_data_and_model(single_run_problem):
    signals, events, _, model = single_run_problem
    data = from_arrays(signals, events, tr=2.0, sources=_complete_sources())

    same = fit(
        data,
        ModelSpec(
            contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
            drift_model=None,
            noise_model="ols",
        ),
    )
    changed = fit(
        data,
        ModelSpec(
            contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
            drift_model=None,
            noise_model="ar1",
        ),
    )
    anonymous = fit(from_arrays(signals, events, tr=2.0), model)

    assert same.provenance.analysis_fingerprint is not None
    assert (
        same.provenance.analysis_fingerprint
        == fit(data, model).provenance.analysis_fingerprint
    )
    assert (
        changed.provenance.analysis_fingerprint != same.provenance.analysis_fingerprint
    )
    assert anonymous.provenance.analysis_fingerprint is None
    assert any(
        warning["code"] == "provenance_quality"
        for warning in anonymous.provenance.warnings
    )


def test_fit_serializes_model_spec_and_run_diagnostics(single_run_problem):
    signals, events, _, _ = single_run_problem
    signals = signals.copy()
    signals[0, 0] = 987654.5
    confounds = pd.DataFrame({"motion_x": np.linspace(-1.0, 1.0, len(events) * 5)})
    model = ModelSpec(
        contrasts={
            "face_gt_house": "face_glover_hrf - house_glover_hrf",
            "face_only": {"face_glover_hrf": 1.0},
        },
        confounds=("motion_x",),
        hrf_model=glover_hrf,
        drift_model="cosine",
        high_pass=0.02,
        oversampling=20,
        min_onset=-10.0,
        noise_model="ols",
    )

    result = fit(
        from_arrays(
            signals, events, tr=2.0, confounds=confounds, sources=_complete_sources()
        ),
        model,
    )
    activity = result.provenance.activities[-1]
    run = activity["runs"][0]
    serialized = result.provenance.canonical_json()

    assert activity["model"] == {
        "contrasts": {
            "face_gt_house": {
                "kind": "expression",
                "value": "face_glover_hrf - house_glover_hrf",
            },
            "face_only": {
                "kind": "weights",
                "weights": {"face_glover_hrf": 1.0},
            },
        },
        "confounds": ["motion_x"],
        "hrf_model": {
            "kind": "callable",
            "module": "nilearn.glm.first_level.hemodynamic_models",
            "qualname": "glover_hrf",
        },
        "drift_model": "cosine",
        "high_pass": 0.02,
        "drift_order": 1,
        "oversampling": 20,
        "min_onset": -10.0,
        "noise_model": "ols",
    }
    assert run["timing_source"] == "tr"
    assert run["n_scans"] == signals.shape[0]
    assert run["n_features"] == signals.shape[1]
    assert run["design_columns"] == list(result.design_matrices[0].columns)
    assert run["design_rank"] == int(
        np.linalg.matrix_rank(result.design_matrices[0].to_numpy())
    )
    assert (
        run["residual_dof"] == result.design_matrices[0].shape[0] - run["design_rank"]
    )
    assert run["excluded_event_count"] == 0
    assert run["min_onset_cutoff"] == -10.0
    assert "987654.5" not in serialized
    assert "signals" not in serialized
    assert "design_values" not in serialized
    assert "feature_diagnostics" not in serialized
    assert "effect_size" not in serialized


def test_fit_marks_local_callable_hrf_as_partially_reproducible(single_run_problem):
    signals, events, _, _ = single_run_problem

    def local_hrf(frame_times, oversampling=50):
        return np.asarray(glover_hrf(2.0, oversampling=oversampling))

    model = ModelSpec(
        contrasts={"face_gt_house": {"face_local_hrf": 1.0, "house_local_hrf": -1.0}},
        hrf_model=local_hrf,
        drift_model=None,
        noise_model="ols",
    )

    result = fit(
        from_arrays(signals, events, tr=2.0, sources=_complete_sources()),
        model,
    )
    activity = result.provenance.activities[-1]
    serialized = result.provenance.canonical_json()

    assert activity["model"]["hrf_model"] == {
        "kind": "callable",
        "reproducibility": "partial",
    }
    assert any(
        "callable" in warning["message"] for warning in result.provenance.warnings
    )
    assert "local_hrf" not in str(activity["model"]["hrf_model"])
    assert "<function" not in serialized
    assert "0x" not in serialized


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


def test_task_delta_r2_compares_complete_and_nuisance_models(delta_r2_problem):
    data, model, full_result = delta_r2_problem

    comparison = task_delta_r2(data, model, full_result)
    expected_full_ols_r2 = _nilearn_pooled_ols_r2(
        data.signals,
        _full_ols_designs(data, model),
    )
    expected_nuisance_ols_r2 = _nilearn_pooled_ols_r2(
        data.signals,
        _nuisance_ols_designs(data, model),
    )
    expected_raw = expected_full_ols_r2 - expected_nuisance_ols_r2

    np.testing.assert_allclose(comparison.full_r2, expected_full_ols_r2)
    np.testing.assert_allclose(comparison.nuisance_r2, expected_nuisance_ols_r2)
    np.testing.assert_allclose(comparison.raw_delta_r2, expected_raw)
    assert np.all(comparison.raw_delta_r2 >= -1e-12)
    np.testing.assert_allclose(comparison.delta_r2, np.maximum(expected_raw, 0.0))
    assert comparison.negative_voxel_count == int(
        np.count_nonzero(comparison.raw_delta_r2 < 0.0)
    )
    assert comparison.raw_min == pytest.approx(comparison.raw_delta_r2.min())

    arrays = (
        comparison.full_r2,
        comparison.nuisance_r2,
        comparison.raw_delta_r2,
        comparison.delta_r2,
    )
    assert {values.shape for values in arrays} == {(data.n_features,)}
    for values in arrays:
        assert values.ndim == 1
        assert values.dtype == np.dtype("float64")
        assert values.flags.owndata is True
        assert not np.shares_memory(values, full_result.r2)
        assert type(values) is np.ndarray
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 0

    returned = comparison.nuisance_design_matrices[0]
    returned.iloc[0, 0] = -99.0
    assert comparison.nuisance_design_matrices[0].iloc[0, 0] != -99.0


def test_make_task_delta_r2_result_clips_and_owns_values(delta_r2_problem):
    _, _, full_result = delta_r2_problem
    full = np.array([0.25, 0.4, 0.8])
    nuisance = np.array([0.5, 0.4, 0.5])
    nuisance_design = pd.DataFrame({"constant": [1.0, 1.0]})

    result = make_task_delta_r2_result(
        full_r2=full,
        nuisance_r2=nuisance,
        nuisance_designs=(nuisance_design,),
        provenance=full_result.provenance,
    )
    full[0] = 99.0
    nuisance[0] = 99.0
    nuisance_design.iloc[0, 0] = 99.0

    np.testing.assert_allclose(result.raw_delta_r2, [-0.25, 0.0, 0.3])
    np.testing.assert_allclose(result.delta_r2, [0.0, 0.0, 0.3])
    np.testing.assert_allclose(result.full_r2, [0.25, 0.4, 0.8])
    np.testing.assert_allclose(result.nuisance_r2, [0.5, 0.4, 0.5])
    assert result.negative_voxel_count == 1
    assert result.raw_min == -0.25
    assert result.nuisance_design_matrices[0].iloc[0, 0] == 1.0


@pytest.mark.parametrize(
    ("full", "nuisance"),
    [
        (np.array([]), np.array([])),
        (np.ones((1, 1)), np.ones(1)),
        (np.ones(2), np.ones(1)),
        (np.array([np.nan]), np.zeros(1)),
        (np.zeros(1), np.array([np.inf])),
    ],
)
def test_make_task_delta_r2_result_rejects_invalid_inputs(
    delta_r2_problem,
    full,
    nuisance,
):
    _, _, full_result = delta_r2_problem

    with pytest.raises(ValueError):
        make_task_delta_r2_result(
            full_r2=full,
            nuisance_r2=nuisance,
            nuisance_designs=(pd.DataFrame({"constant": [1.0]}),),
            provenance=full_result.provenance,
        )


def test_make_task_delta_r2_result_rejects_overflow_without_warning(
    delta_r2_problem,
):
    _, _, full_result = delta_r2_problem
    maximum = np.finfo(np.float64).max

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            ValueError,
            match="derived delta r-squared values must be finite",
        ):
            make_task_delta_r2_result(
                full_r2=np.array([maximum]),
                nuisance_r2=np.array([-maximum]),
                nuisance_designs=(pd.DataFrame({"constant": [1.0]}),),
                provenance=full_result.provenance,
            )


@pytest.mark.parametrize(
    "nuisance_design",
    [
        pd.DataFrame({"payload": [{"nested": [1.0]}]}),
        pd.DataFrame({"constant": [np.nan]}),
    ],
    ids=("nested-object", "nonfinite"),
)
def test_make_task_delta_r2_result_rejects_nonfinite_or_nonnumeric_nuisance_designs(
    delta_r2_problem,
    nuisance_design,
):
    _, _, full_result = delta_r2_problem

    with pytest.raises(
        ValueError,
        match="nuisance designs must be finite numeric matrices",
    ):
        make_task_delta_r2_result(
            full_r2=np.array([0.4]),
            nuisance_r2=np.array([0.2]),
            nuisance_designs=(nuisance_design,),
            provenance=full_result.provenance,
        )


def test_task_delta_r2_rejects_full_result_for_changed_model(delta_r2_problem):
    data, model, full_result = delta_r2_problem
    changed_model = ModelSpec(
        contrasts=model.contrasts,
        confounds=model.confounds,
        drift_model=model.drift_model,
        high_pass=model.high_pass,
        noise_model="ols",
    )

    with pytest.raises(ValueError, match="full result does not match data and model"):
        task_delta_r2(data, changed_model, full_result)


def test_task_delta_r2_rejects_full_result_for_changed_data(delta_r2_problem):
    data, model, full_result = delta_r2_problem
    changed_data = from_arrays(
        data.signals,
        data.events,
        frame_times=data.frame_times,
        confounds=data.confounds,
        sources=_delta_r2_sources(signal_byte_size=4097),
    )

    with pytest.raises(ValueError, match="full result does not match data and model"):
        task_delta_r2(changed_data, model, full_result)


def test_task_delta_r2_requires_fingerprintable_provenance(delta_r2_problem):
    data, model, _ = delta_r2_problem
    anonymous = from_arrays(
        data.signals,
        data.events,
        frame_times=data.frame_times,
        confounds=data.confounds,
    )
    anonymous_result = fit(anonymous, model)

    with pytest.raises(ValueError, match="requires fingerprintable"):
        task_delta_r2(anonymous, model, anonymous_result)


def test_task_delta_r2_records_parent_model_and_diagnostics(delta_r2_problem):
    data, model, full_result = delta_r2_problem

    comparison = task_delta_r2(data, model, full_result)
    activity = comparison.provenance.activities[-1]

    assert activity["name"] == "task_delta_r2"
    assert activity["parent_analysis_id"] == full_result.provenance.analysis_fingerprint
    assert activity["definition"] == "full_r2 - nuisance_r2"
    assert activity["clip_below_zero"] is True
    assert activity["diagnostic_noise_model"] == "ols"
    assert activity["inferential_noise_model"] == "ar1"
    assert activity["nuisance_model"]["events"] is False
    assert activity["nuisance_model"]["noise_model"] == "ols"
    assert activity["clip_policy"] == "numerical_roundoff_guard"
    assert activity["diagnostics"]["negative_voxel_count"] == (
        comparison.negative_voxel_count
    )
    assert activity["runs"][0]["design_columns"] == list(
        comparison.nuisance_design_matrices[0].columns
    )
    serialized = comparison.provenance.canonical_json()
    assert "design_values" not in serialized
    assert "signals" not in serialized


@pytest.mark.parametrize("outcome", ["success", "late_failure", "early_failure"])
def test_ordinary_complete_lifecycle(delta_r2_problem, caplog, monkeypatch, outcome):
    import json
    import logging

    data, model, full = delta_r2_problem
    caplog.set_level(logging.INFO, logger="boldtailor")
    caplog.clear()
    failure = ValueError("private late result detail")

    def reject(*args, **kwargs):
        raise failure

    if outcome == "late_failure":
        monkeypatch.setattr("boldtailor.fit.make_result", reject)
    if outcome == "success":
        result = fit(data, model)
    else:
        with pytest.raises(ValueError) as caught:
            if outcome == "early_failure":
                fit(data, model, feature_signature="unexpected")
            else:
                fit(data, model)
        if outcome == "late_failure":
            assert caught.value is failure
    records = [
        json.loads(r.getMessage()) for r in caplog.records if r.name == "boldtailor"
    ]
    ending = "completed" if outcome == "success" else "failed"
    assert [r["event"] for r in records] == ["fit_started", f"fit_{ending}"]
    assert "analysis_id" not in records[0]
    assert records[0]["execution_id"] == records[1]["execution_id"]
    assert "private" not in caplog.text
    if outcome == "success":
        assert dict(result.provenance.events[-1]) == records[-1]
        assert records[-1]["execution_id"] == result.provenance.execution_id
        assert records[-1].get("analysis_id") == result.provenance.analysis_fingerprint
        assert records[-1].get("data_id") == result.provenance.metadata_fingerprint
        assert len(result.provenance.events) <= 8
    else:
        assert records[-1]["error_code"] == "invalid_input"


@pytest.mark.parametrize("outcome", ["success", "late_failure", "early_failure"])
def test_comparison_complete_lifecycle(delta_r2_problem, caplog, monkeypatch, outcome):
    import json
    import logging
    from dataclasses import replace

    data, model, full = delta_r2_problem
    caplog.set_level(logging.INFO, logger="boldtailor")
    caplog.clear()
    failure = ValueError("private late result detail")

    def reject(*args, **kwargs):
        raise failure

    if outcome == "late_failure":
        monkeypatch.setattr("boldtailor._fit_lifecycle.extend_provenance", reject)
    if outcome == "success":
        result = task_delta_r2(data, model, full)
    else:
        with pytest.raises(ValueError) as caught:
            if outcome == "early_failure":
                task_delta_r2(
                    data,
                    replace(
                        model,
                        noise_model="ar1" if model.noise_model == "ols" else "ols",
                    ),
                    full,
                )
            else:
                task_delta_r2(data, model, full)
        if outcome == "late_failure":
            assert caught.value is failure
    records = [
        json.loads(r.getMessage()) for r in caplog.records if r.name == "boldtailor"
    ]
    ending = "completed" if outcome == "success" else "failed"
    assert [r["event"] for r in records] == [
        "task_delta_r2_started",
        f"task_delta_r2_{ending}",
    ]
    assert "analysis_id" not in records[0]
    assert records[0]["execution_id"] == records[1]["execution_id"]
    assert "private" not in caplog.text
    if outcome == "success":
        assert dict(result.provenance.events[-1]) == records[-1]
        assert records[-1]["execution_id"] == result.provenance.execution_id
        assert records[-1].get("analysis_id") == result.provenance.analysis_fingerprint
        assert records[-1].get("data_id") == result.provenance.metadata_fingerprint
        assert len(result.provenance.events) <= 8
    else:
        assert records[-1]["error_code"] == "invalid_input"
