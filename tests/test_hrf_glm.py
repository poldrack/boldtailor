"""Conventional voxelwise HRFs must preserve semantic contrasts and inference."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from nilearn.glm import compute_contrast
from nilearn.glm.first_level import (
    compute_regressor,
    make_first_level_design_matrix,
    run_glm,
)

from boldtailor.data import from_arrays
from boldtailor.fit import fit, task_delta_r2
from boldtailor.hrf_library import HrfLibrary
from boldtailor.hrf_selection import select_hrf
from boldtailor.model import ModelSpec, Modulator, TaskModel
from boldtailor.provenance import RunSources, SourceRef
from tests.oracles import SHARED_DELTA_ACTIVITY_KEYS
from tests.oracles import scaled_condition


def _sources(run):
    return RunSources(
        **{
            role: SourceRef(
                role,
                uri=f"run-{run}_{role}.tsv",
                byte_size=1024,
                modified_at="2026-09-27T12:00:00Z",
            )
            for role in ("signal", "events", "confounds")
        }
    )


def _oracle_design(events, times, confounds, candidate, model):
    columns = {}
    for condition in sorted(events.trial_type.unique()):
        selected = events.loc[events.trial_type == condition]
        values, _ = compute_regressor(
            scaled_condition(
                selected.onset,
                selected.duration,
                selected.modulation,
                candidate.kernel,
                times,
                model.oversampling,
                model.min_onset,
            ),
            candidate.kernel,
            times,
            oversampling=model.oversampling,
            min_onset=model.min_onset,
        )
        columns[condition] = values[:, 0]
    nuisance = make_first_level_design_matrix(
        times,
        events=None,
        drift_model=model.drift_model,
        high_pass=model.high_pass,
        drift_order=model.drift_order,
        add_regs=confounds.loc[:, list(model.confounds)],
    )
    return pd.DataFrame(columns, index=times).join(nuisance)


@pytest.fixture(scope="module")
def hrf_glm_problem(two_candidate_library):
    library = two_candidate_library
    ids = [1, 0, 1, 2]
    model = ModelSpec(
        contrasts={"stimulus": {"stimulus": 1}, "rt_effect": "rt"},
        confounds=("motion",),
        high_pass=0.01,
        oversampling=20,
        min_onset=-10,
        noise_model="ols",
    )
    events, times, confounds, training, signals, designs = [], [], [], [], [], {}
    rng = np.random.default_rng(734)
    for run in range(3):
        t = 0.775 + 1.6 * np.arange(85 + 5 * run)
        stimulus = pd.DataFrame(
            dict(
                onset=np.array([5.3, 21.1, 42.2, 64.4, 88.5, 110.2]) + run,
                duration=[1.2, 2.0, 0.7, 1.5, 1.1, 2.3],
                trial_type="stimulus",
                modulation=1.0,
            )
        )
        rt = stimulus.assign(
            trial_type="rt", modulation=[-0.3, 0.1, 0.5, -0.4, 0.3, -0.2]
        )
        e = pd.concat([stimulus, rt], ignore_index=True)
        n = pd.DataFrame(
            dict(motion=np.linspace(-1, 1, len(t)), unused=rng.normal(size=len(t)))
        )
        mean_columns, target_columns = [], []
        for feature, cid in enumerate(ids):
            candidate = library.candidates[cid]
            mean = compute_regressor(
                stimulus[["onset", "duration", "modulation"]].to_numpy().T,
                "spm" if cid == 0 else candidate.kernel,
                t,
            )[0][:, 0]
            mean_columns.append(3 * mean + 100 + n.motion)
            design = _oracle_design(e, t, n, candidate, model)
            designs[run, cid] = design
            coefficients = np.zeros(design.shape[1])
            coefficients[design.columns.get_loc("stimulus")] = 2 + feature + run
            coefficients[design.columns.get_loc("rt")] = 0.4 - feature / 4
            coefficients[design.columns.get_loc("motion")] = 1.3
            coefficients[design.columns.get_loc("constant")] = 100 + run
            noise = rng.normal(0, 0.03, len(t))
            for scan in range(1, len(t)):
                noise[scan] += 0.5 * noise[scan - 1]
            target_columns.append(design.to_numpy() @ coefficients + noise)
        events.append(e)
        times.append(t)
        confounds.append(n)
        training.append(np.column_stack([*mean_columns, np.full(len(t), 100.0)]))
        signals.append(np.column_stack([*target_columns, rng.normal(100, 1, len(t))]))
    training_data = from_arrays(
        training,
        [e.iloc[:6] for e in events],
        frame_times=times,
        confounds=[n[["motion"]] for n in confounds],
    )
    selection = select_hrf(training_data, library=library, feature_signature="axis-v1")
    np.testing.assert_array_equal(selection.hrf_indices, [*ids, -1])
    data = from_arrays(
        signals,
        events,
        frame_times=times,
        confounds=confounds,
        sources=[_sources(r) for r in range(3)],
    )
    return data, model, selection, designs


def _selected_fit(data, model, selection):
    return fit(data, model, hrf_selection=selection, feature_signature="axis-v1")


@pytest.mark.parametrize("noise_model", ["ols", "ar1"])
def test_voxelwise_contrasts_match_independent_run_glms(hrf_glm_problem, noise_model):
    data, model, selection, designs = hrf_glm_problem
    model = replace(model, noise_model=noise_model)
    result = _selected_fit(data, model, selection)
    for feature, cid in enumerate(selection.hrf_indices[:4]):
        contrasts = {name: [] for name in model.contrasts}
        sse, sst = [], []
        for run, y in enumerate(data.signals):
            design = designs[run, cid]
            matrix = design.to_numpy()
            observed = y[:, feature : feature + 1]
            labels, fitted = run_glm(observed, matrix, noise_model=noise_model)
            prediction = matrix @ fitted[labels[0]].theta
            sse.append(np.sum((observed - prediction) ** 2))
            sst.append(np.sum((observed - observed.mean()) ** 2))
            np.testing.assert_allclose(
                result.run_r2[run][feature], 1 - sse[-1] / sst[-1]
            )
            pd.testing.assert_frame_equal(result.group_design(run, cid), design)
            for name, column in [("stimulus", "stimulus"), ("rt_effect", "rt")]:
                vector = np.zeros(matrix.shape[1])
                vector[design.columns.get_loc(column)] = 1
                contrasts[name].append(
                    compute_contrast(labels, fitted, vector, stat_type="t")
                )
        np.testing.assert_allclose(
            result.r2[feature], 1 - sum(sse) / sum(sst), atol=1e-12
        )
        for name, runs in contrasts.items():
            expected = (1 / 3) * (runs[0] + runs[1] + runs[2])
            for accessor, oracle in [
                ("effect", "effect_size"),
                ("variance", "effect_variance"),
                ("stat", "stat"),
                ("z_score", "z_score"),
                ("one_sided_p_value", "p_value"),
            ]:
                np.testing.assert_allclose(
                    getattr(result, accessor)(name)[feature],
                    getattr(expected, oracle)()[0],
                    rtol=1e-8,
                    atol=1e-10,
                )
    assert result.contrast_names == model.contrast_names
    assert np.isnan(result.effect("stimulus")[-1])
    assert np.isnan(result.r2[-1])
    with pytest.raises(KeyError, match="unknown contrast"):
        result.effect("missing")


@pytest.mark.parametrize("noise_model", ["ols", "ar1"])
def test_canonical_assignment_matches_existing_api(hrf_glm_problem, noise_model):
    data, model, _, _ = hrf_glm_problem
    selection_data = from_arrays(
        data.signals,
        [e.iloc[:6] for e in data.events],
        frame_times=data.frame_times,
        confounds=data.confounds,
    )
    selection = select_hrf(selection_data, library=HrfLibrary.from_parameters([]))
    model = replace(model, hrf_model="spm", noise_model=noise_model)
    actual = fit(data, model, hrf_selection=selection)
    expected = fit(data, model)
    for name in model.contrasts:
        for accessor in ("effect", "variance", "stat", "z_score", "one_sided_p_value"):
            np.testing.assert_allclose(
                getattr(actual, accessor)(name),
                getattr(expected, accessor)(name),
                atol=1e-12,
            )
    np.testing.assert_allclose(actual.r2, expected.r2, atol=1e-12)
    for run, design in enumerate(expected.design_matrices):
        pd.testing.assert_frame_equal(actual.group_design(run, 0), design)


def test_delta_r2_uses_nested_ols_with_selected_hrfs(hrf_glm_problem):
    data, model, selection, designs = hrf_glm_problem
    model = replace(model, noise_model="ar1")
    result = _selected_fit(data, model, selection)
    delta = task_delta_r2(data, model, result)
    for feature, cid in enumerate(selection.hrf_indices[:4]):
        full_sse, null_sse, total = [], [], []
        for run, y in enumerate(data.signals):
            design = designs[run, cid]
            x = design.to_numpy()
            n = design.drop(columns=["stimulus", "rt"]).to_numpy()
            observed = y[:, feature]
            full_sse.append(
                np.sum(
                    (observed - x @ np.linalg.lstsq(x, observed, rcond=None)[0]) ** 2
                )
            )
            null_sse.append(
                np.sum(
                    (observed - n @ np.linalg.lstsq(n, observed, rcond=None)[0]) ** 2
                )
            )
            total.append(np.sum((observed - observed.mean()) ** 2))
        full = 1 - sum(full_sse) / sum(total)
        nuisance = 1 - sum(null_sse) / sum(total)
        np.testing.assert_allclose(delta.full_r2[feature], full, atol=1e-12)
        np.testing.assert_allclose(delta.nuisance_r2[feature], nuisance, atol=1e-12)
        np.testing.assert_allclose(delta.delta_r2[feature], full - nuisance, atol=1e-12)
    assert np.isnan(delta.delta_r2[-1])
    activity = delta.provenance.to_dict()["activities"][-1]
    assert set(activity) >= SHARED_DELTA_ACTIVITY_KEYS
    assert activity["diagnostic_noise_model"] == "ols"
    assert activity["undefined_features"] == 1


@pytest.mark.parametrize(
    ("with_selection", "signature", "message"),
    [
        (True, None, "signature"),
        (True, "reordered-axis", "signature"),
        (False, "axis-v1", "selection"),
    ],
)
def test_rejects_spatial_signature_mismatch(
    hrf_glm_problem, with_selection, signature, message
):
    data, model, selection, _ = hrf_glm_problem
    selection = selection if with_selection else None
    with pytest.raises(ValueError, match=message):
        fit(data, model, hrf_selection=selection, feature_signature=signature)


def test_rejects_bad_assignment_and_library_identity(hrf_glm_problem):
    data, model, selection, _ = hrf_glm_problem
    for bad in [
        replace(selection, hrf_indices=[1, 0, 1, 99, -1]),
        replace(selection, hrf_indices=[0, 1, 1, 2, -1]),
        replace(selection, library=HrfLibrary.from_parameters([])),
    ]:
        with pytest.raises(ValueError, match="identity|fingerprint|HRF|hrf"):
            _selected_fit(data, model, bad)
    fewer = from_arrays(
        data.signals[0][:, :3], data.events[0], frame_times=data.frame_times[0]
    )
    with pytest.raises(ValueError, match="feature"):
        _selected_fit(fewer, model, selection)


def test_selection_can_transfer_to_one_new_run(hrf_glm_problem):
    data, model, selection, _ = hrf_glm_problem
    single = from_arrays(
        data.signals[1],
        data.events[1],
        frame_times=data.frame_times[1],
        confounds=data.confounds[1],
    )
    result = _selected_fit(single, model, selection)
    assert len(result.run_r2) == 1
    assert result.effect("stimulus").shape == (5,)
    assert np.isfinite(result.r2[:4]).all()


def test_result_owns_designs_and_records_effective_model(hrf_glm_problem):
    data, model, selection, _ = hrf_glm_problem
    result = _selected_fit(data, model, selection)
    for values in (
        result.r2,
        *result.run_r2,
        result.hrf_indices,
        result.effect("stimulus"),
    ):
        assert type(values) is np.ndarray
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 0
    design = result.group_design(0, 1)
    design.iloc[:, :] = 0
    assert result.group_design(0, 1).to_numpy().any()
    np.testing.assert_array_equal(result.hrf_indices, selection.hrf_indices)
    assert result.selection_provenance == selection.provenance
    info = result.provenance.to_dict()["activities"][-1]
    assert info["library_fingerprint"] == selection.library.fingerprint
    assert info["hrf_assignment_fingerprint"]
    assert info["design_fingerprint"]
    assert info["model"]["hrf_model"]["kind"] == "selected"
    assert info["hrf_normalization"] == "peak_one_event_response"
    assert result.provenance.analysis_fingerprint
    # An ignored fixed-HRF choice must not alter the effective model identity.
    other = _selected_fit(data, replace(model, hrf_model="spm"), selection)
    assert (
        result.provenance.analysis_fingerprint == other.provenance.analysis_fingerprint
    )


@pytest.mark.parametrize("change", ["model", "events", "confounds", "sources"])
def test_delta_rejects_a_different_analysis(hrf_glm_problem, change):
    data, model, selection, _ = hrf_glm_problem
    result = _selected_fit(data, model, selection)
    if change == "model":
        model = replace(model, noise_model="ar1")
    else:
        events, confounds = data.events, data.confounds
        sources = [_sources(r) for r in range(3)]
        if change == "events":
            events[0].loc[6, "modulation"] += 0.2
        elif change == "confounds":
            confounds[0].loc[3, "motion"] += 0.2
        else:
            sources = [_sources(r + 10) for r in range(3)]
        data = from_arrays(
            data.signals,
            events,
            frame_times=data.frame_times,
            confounds=confounds,
            sources=sources,
        )
    with pytest.raises(ValueError, match="match|identity|fingerprint"):
        task_delta_r2(data, model, result)


def test_all_undefined_hrfs_return_nan_maps(hrf_glm_problem):
    data, model, _, _ = hrf_glm_problem
    constant = from_arrays(
        [np.full_like(y, 100) for y in data.signals],
        [e.iloc[:6] for e in data.events],
        frame_times=data.frame_times,
        confounds=data.confounds,
    )
    selection = select_hrf(constant, library=HrfLibrary.from_parameters([]))
    result = fit(data, model, hrf_selection=selection)
    assert np.isnan(result.r2).all()
    assert np.isnan(result.z_score("stimulus")).all()
    delta = task_delta_r2(data, model, result)
    assert np.isnan(delta.delta_r2).all()
    assert delta.negative_voxel_count == 0
    # Provenance remains JSON-safe even when no R² is defined.
    assert "NaN" not in delta.provenance.canonical_json()


def test_custom_hrf_keeps_condition_and_confound_names_distinct(hrf_glm_problem):
    data, model, selection, _ = hrf_glm_problem
    confounds = data.confounds
    for frame in confounds:
        frame["stimulus_kernel"] = np.sin(np.arange(len(frame)) / 3)
    data = from_arrays(
        data.signals, data.events, frame_times=data.frame_times, confounds=confounds
    )
    model = replace(model, confounds=("motion", "stimulus_kernel"))
    result = _selected_fit(data, model, selection)
    for run, cid in result.group_design_provenance:
        actual = result.group_design(run, cid)
        expected = _oracle_design(
            data.events[run],
            data.frame_times[run],
            confounds[run],
            selection.library.candidates[cid],
            model,
        )
        pd.testing.assert_frame_equal(actual, expected)


def test_selected_delta_requires_complete_sources(hrf_glm_problem):
    data, model, selection, _ = hrf_glm_problem
    anonymous = from_arrays(
        data.signals,
        data.events,
        frame_times=data.frame_times,
        confounds=data.confounds,
    )
    result = _selected_fit(anonymous, model, selection)
    with pytest.raises(ValueError, match="fingerprint|source|provenance"):
        task_delta_r2(anonymous, model, result)


def test_selected_delta_handles_constant_target_after_hrf_transfer(hrf_glm_problem):
    data, model, selection, _ = hrf_glm_problem
    signals = [y.copy() for y in data.signals]
    for y in signals:
        y[:, 0] = 100
    target = from_arrays(
        signals,
        data.events,
        frame_times=data.frame_times,
        confounds=data.confounds,
        sources=[_sources(r) for r in range(3)],
    )
    result = _selected_fit(target, model, selection)
    assert np.isnan(result.r2[0])
    delta = task_delta_r2(target, model, result)
    assert np.isnan(delta.full_r2[0])
    assert np.isnan(delta.nuisance_r2[0])
    assert np.isnan(delta.delta_r2[0])
    assert np.isfinite(delta.delta_r2[1:4]).all()


def test_selected_glm_masks_constant_feature_contrasts(hrf_glm_problem):
    data, model, selection, _ = hrf_glm_problem
    signals = [y.copy() for y in data.signals]
    for y in signals:
        y[:, 0] = 100
    target = from_arrays(
        signals,
        data.events,
        frame_times=data.frame_times,
        confounds=data.confounds,
        sources=[_sources(r) for r in range(3)],
    )
    result = _selected_fit(target, model, selection)
    for name in model.contrasts:
        assert np.isnan(result.stat(name)[0])
        assert np.isfinite(result.stat(name)[1])


def _undefined_selection(data):
    constant = from_arrays(
        [np.full_like(y, 100) for y in data.signals],
        [e.iloc[:6] for e in data.events],
        frame_times=data.frame_times,
        confounds=data.confounds,
    )
    return select_hrf(constant, library=HrfLibrary.from_parameters([]))


@pytest.mark.parametrize(
    ("undefined_assignment", "contrast", "message"),
    [
        (False, "absent", "missing|invalid"),
        (True, "absent", "invalid|zero|missing"),
        (True, "stimulus - stimulus", "invalid|zero|missing"),
    ],
)
def test_fit_rejects_invalid_contrasts_for_any_assignment(
    hrf_glm_problem, undefined_assignment, contrast, message
):
    data, model, selection, _ = hrf_glm_problem
    signature = "axis-v1"
    if undefined_assignment:
        selection, signature = _undefined_selection(data), None
    bad_model = replace(model, contrasts={"bad": contrast})
    with pytest.raises(ValueError, match=message):
        fit(data, bad_model, hrf_selection=selection, feature_signature=signature)


def _nsd_model():
    from boldtailor.model import Modulator, TaskModel

    return TaskModel(
        (
            Modulator("response_time", center=True, missing="indicator"),
            Modulator("trial_type", center=False),
        )
    )


@pytest.fixture(scope="module")
def task_model_problem(two_candidate_library):
    """Raw trials with RT and trial type; one missing RT in run 1."""
    library = two_candidate_library
    rng = np.random.default_rng(1207)
    events, times, confounds, signals = [], [], [], []
    for run in range(3):
        t = 0.775 + 1.6 * np.arange(85 + 5 * run)
        rt = np.array([0.8, 1.3, 0.9, 1.7, 1.1, 1.4])
        if run == 1:
            rt[3] = np.nan
        events.append(
            pd.DataFrame(
                dict(
                    onset=np.array([5.3, 21.1, 42.2, 64.4, 88.5, 110.2]) + run,
                    duration=[1.2, 2.0, 0.7, 1.5, 1.1, 2.3],
                    trial_type=[0, 1, 1, 0, 1, 0],
                    response_time=rt,
                )
            )
        )
        times.append(t)
        confounds.append(pd.DataFrame(dict(motion=np.linspace(-1, 1, len(t)))))
        signals.append(rng.normal(100, 1, size=(len(t), 4)))
    data = from_arrays(
        signals,
        events,
        frame_times=times,
        confounds=confounds,
        sources=[_sources(r) for r in range(3)],
    )
    model = ModelSpec(
        contrasts={name: {name: 1} for name in ("task", "response_time", "trial_type")},
        confounds=("motion",),
        hrf_model="spm",
        drift_model=None,
        noise_model="ols",
        task_model=_nsd_model(),
    )
    selection = select_hrf(
        data, library=library, feature_signature="axis-tm", task_model=_nsd_model()
    )
    return data, model, selection, library


def test_selected_glm_group_designs_equal_scored_task_columns(task_model_problem):
    from boldtailor._hrf_cv import prepare_runs
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import expand_events, task_columns

    data, model, selection, library = task_model_problem
    result = fit(data, model, hrf_selection=selection, feature_signature="axis-tm")
    runs = prepare_runs(data, library, model.task_model)
    assert result.group_design_provenance
    for run, cid in result.group_design_provenance:
        design = result.group_design(run, cid)
        scored = runs[run].task_design(cid)
        expected = task_columns(
            expand_events(data.events[run], model.task_model, run),
            data.frame_times[run],
            hrf_model(library.candidates[cid]),
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
        pd.testing.assert_frame_equal(scored, expected)
        assert list(design.columns[: scored.shape[1]]) == list(scored.columns)
        np.testing.assert_array_equal(
            design.iloc[:, : scored.shape[1]].to_numpy(), scored.to_numpy()
        )
        assert list(design.columns[scored.shape[1] :]) == ["motion", "constant"]
    assert (
        "missing_response_time"
        in result.group_design(1, int(selection.hrf_indices[0])).columns
    )
    assert (
        "missing_response_time"
        not in result.group_design(0, int(selection.hrf_indices[0])).columns
    )
    activity = result.provenance.to_dict()["activities"][-1]
    assert activity["task_model"] == model.task_model.to_dict()
    assert activity["task_model_fingerprint"] == model.task_model.fingerprint
    assert np.isfinite(result.effect("response_time")).all()
    assert np.isfinite(result.effect("trial_type")).all()


def test_selected_glm_spm_group_also_uses_shared_task_columns(task_model_problem):
    from boldtailor._task_design import expand_events, task_columns

    data, model, selection, library = task_model_problem
    canonical = HrfLibrary.from_parameters([])
    spm_only = select_hrf(
        data, library=canonical, feature_signature="axis-tm", task_model=_nsd_model()
    )
    result = fit(data, model, hrf_selection=spm_only, feature_signature="axis-tm")
    for run in range(data.n_runs):
        expected = task_columns(
            expand_events(data.events[run], model.task_model, run),
            data.frame_times[run],
            "spm",
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
        design = result.group_design(run, 0)
        np.testing.assert_array_equal(
            design.iloc[:, : expected.shape[1]].to_numpy(), expected.to_numpy()
        )


_RT_INDICATOR = Modulator("response_time", missing="indicator")


@pytest.mark.parametrize(
    ("model_changes", "selection_task_model", "message"),
    [
        ({"task_model": None}, None, "task_model"),
        ({"task_model": TaskModel((_RT_INDICATOR,))}, None, "task_model"),
        (
            {},
            TaskModel((_RT_INDICATOR, Modulator("trial_type", center=True))),
            "task_model",
        ),
        ({"oversampling": 20}, None, "oversampling|min_onset"),
        ({"min_onset": -10.0}, None, "oversampling|min_onset"),
    ],
)
def test_selected_glm_rejects_mismatched_selection_settings(
    task_model_problem, model_changes, selection_task_model, message
):
    data, model, selection, library = task_model_problem
    if selection_task_model is not None:
        selection = select_hrf(
            data,
            library=library,
            feature_signature="axis-tm",
            task_model=selection_task_model,
        )
    with pytest.raises(ValueError, match=message):
        fit(
            data,
            replace(model, **model_changes),
            hrf_selection=selection,
            feature_signature="axis-tm",
        )


def test_selected_glm_accepts_selection_on_a_subset_task_model(task_model_problem):
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import expand_events, task_columns
    from boldtailor.model import Modulator, TaskModel

    data, model, selection, library = task_model_problem
    subset = TaskModel((Modulator("trial_type", center=False),))
    narrow = select_hrf(
        data, library=library, feature_signature="axis-tm", task_model=subset
    )
    result = fit(data, model, hrf_selection=narrow, feature_signature="axis-tm")
    for run, cid in result.group_design_provenance:
        design = result.group_design(run, cid)
        expected = task_columns(
            expand_events(data.events[run], model.task_model, run),
            data.frame_times[run],
            hrf_model(library.candidates[cid]),
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
        np.testing.assert_array_equal(
            design.iloc[:, : expected.shape[1]].to_numpy(), expected.to_numpy()
        )
    activity = result.provenance.to_dict()["activities"][-1]
    assert activity["task_model_fingerprint"] == model.task_model.fingerprint
    assert activity["selection_task_model_fingerprint"] == subset.fingerprint
    assert np.isfinite(result.effect("response_time")).all()
    plain = select_hrf(data, library=library, feature_signature="axis-tm")
    assert fit(
        data, model, hrf_selection=plain, feature_signature="axis-tm"
    ).group_design_provenance


def test_selected_glm_reports_design_errors_with_the_run_once(task_model_problem):
    from boldtailor.model import Modulator, TaskModel

    data, model, selection, library = task_model_problem
    strict = TaskModel((Modulator("response_time", missing="error"),))
    complete = from_arrays(
        data.signals,
        [e.assign(response_time=1.0 + e.index) for e in data.events],
        frame_times=data.frame_times,
        confounds=data.confounds,
    )
    chosen = select_hrf(
        complete, library=library, feature_signature="axis-tm", task_model=strict
    )
    with pytest.raises(
        ValueError, match=r"^run 1 design compilation failed: (?!run 1)"
    ):
        fit(
            data,
            replace(model, task_model=strict, contrasts={"task": {"task": 1}}),
            hrf_selection=chosen,
            feature_signature="axis-tm",
        )


_TRUE_CIDS = (1, 0, 2, 1)
_AMPLITUDES = np.array([3.0, 0.8, -0.6])


def _oracle_run(library, task_model, run):
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import run_task_columns

    rng = np.random.default_rng(4100 + run)
    t = 0.775 + 1.6 * np.arange(90 + 4 * run)
    events = pd.DataFrame(
        dict(
            onset=np.array([5.3, 21.1, 42.2, 64.4, 88.5, 110.2]) + run,
            duration=[1.2, 2.0, 0.7, 1.5, 1.1, 2.3],
            trial_type=[0, 1, 1, 0, 1, 0],
            response_time=np.array([0.8, 1.3, 0.9, 1.7, 1.1, 1.4]) + 0.1 * run,
        )
    )
    motion = np.linspace(-1, 1, len(t)) * (run + 1)
    features = []
    for cid in _TRUE_CIDS:
        kernel = hrf_model(library.candidates[cid])
        columns = run_task_columns(
            events,
            task_model,
            t,
            kernel,
            run=run,
            min_onset=-24.0,
            oversampling=50,
        )
        noise = rng.normal(0.0, 0.05, len(t))
        features.append(columns.to_numpy() @ _AMPLITUDES + 0.3 * motion + 50 + noise)
    return np.column_stack(features), events, t, pd.DataFrame(dict(motion=motion))


@pytest.fixture(scope="module")
def known_amplitude_problem(two_candidate_library):
    library, task_model = two_candidate_library, _nsd_model()
    runs = [_oracle_run(library, task_model, run) for run in range(3)]
    signals, events, times, confounds = (list(part) for part in zip(*runs))
    data = from_arrays(signals, events, frame_times=times, confounds=confounds)
    selection = select_hrf(
        data, library=library, feature_signature="axis-oracle", task_model=task_model
    )
    return data, selection, task_model


def _nilearn_equal_weight(signals, designs, feature, column):
    runs = []
    for y, design in zip(signals, designs, strict=True):
        vector = np.zeros(design.shape[1])
        vector[design.columns.get_loc(column)] = 1.0
        labels, fitted = run_glm(y[:, [feature]], design.to_numpy(), noise_model="ols")
        runs.append(compute_contrast(labels, fitted, vector, stat_type="t"))
    return (1 / 3) * (runs[0] + runs[1] + runs[2])


def test_task_model_selected_glm_matches_nilearn_on_known_amplitudes(
    known_amplitude_problem,
):
    data, selection, task_model = known_amplitude_problem
    np.testing.assert_array_equal(selection.hrf_indices, _TRUE_CIDS)
    model = ModelSpec(
        contrasts={"rt": {"response_time": 1.0}},
        confounds=("motion",),
        hrf_model="spm",
        noise_model="ols",
        drift_model=None,
        task_model=task_model,
    )
    result = fit(data, model, hrf_selection=selection, feature_signature="axis-oracle")
    for feature, cid in enumerate(_TRUE_CIDS):
        designs = [result.group_design(run, cid) for run in range(3)]
        expected = _nilearn_equal_weight(
            data.signals, designs, feature, "response_time"
        )
        for accessor, oracle in [
            ("effect", "effect_size"),
            ("variance", "effect_variance"),
            ("stat", "stat"),
        ]:
            np.testing.assert_allclose(
                getattr(result, accessor)("rt")[feature],
                getattr(expected, oracle)()[0],
                rtol=1e-8,
            )


def test_selected_delta_analysis_id_hashes_shared_identity(hrf_glm_problem):
    from boldtailor._fit_diagnostics import delta_r2_identity
    from boldtailor.model import nuisance_model_settings
    from boldtailor.provenance import analysis_fingerprint

    data, model, selection, _ = hrf_glm_problem
    result = _selected_fit(data, model, selection)
    delta = task_delta_r2(data, model, result)
    identity = delta_r2_identity(
        name="task_delta_r2",
        inferential_noise_model=model.noise_model,
        nuisance_model=nuisance_model_settings(model),
    )
    expected = analysis_fingerprint(result.provenance.analysis_fingerprint, identity)
    assert delta.provenance.analysis_fingerprint == expected


def test_selected_glm_rebuilds_group_designs_on_demand(hrf_glm_problem):
    from boldtailor._hrf_glm_design import compile_group_designs

    data, model, selection, _ = hrf_glm_problem
    result = _selected_fit(data, model, selection)
    compiled, _ = compile_group_designs(data, model, selection)
    assert set(result.group_design_provenance) == set(compiled)
    for (run, cid), design in compiled.items():
        pd.testing.assert_frame_equal(result.group_design(run, cid), design.matrix)
    assert not hasattr(result, "group_designs")
    assert not any(
        isinstance(value, pd.DataFrame)
        or (
            isinstance(value, dict)
            and any(isinstance(v, pd.DataFrame) for v in value.values())
        )
        for value in vars(result).values()
    )


@pytest.mark.parametrize("key", [(0, 0), (0, 3), (3, 1), (-1, 1), (0, -1)])
def test_selected_glm_group_design_rejects_pairs_that_were_not_fitted(
    hrf_glm_problem, key
):
    from tests.test_fit import replace_indices

    data, model, selection, _ = hrf_glm_problem
    narrow = replace_indices(selection, np.where(selection.hrf_indices >= 0, 1, -1))
    result = _selected_fit(data, model, narrow)
    assert set(result.group_design_provenance) == {(r, 1) for r in range(3)}
    with pytest.raises(KeyError):
        result.group_design(*key)
