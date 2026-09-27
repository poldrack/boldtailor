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
from boldtailor.model import ModelSpec
from boldtailor.provenance import RunSources, SourceRef


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
            selected[["onset", "duration", "modulation"]].to_numpy().T,
            "spm" if candidate.id == 0 else candidate.kernel,
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
def hrf_glm_problem():
    library = HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )
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
            pd.testing.assert_frame_equal(result.group_designs[run, cid], design)
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
        pd.testing.assert_frame_equal(actual.group_designs[run, 0], design)


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
    assert (
        delta.provenance.to_dict()["activities"][-1]["diagnostic_noise_model"] == "ols"
    )


@pytest.mark.parametrize("signature", [None, "reordered-axis"])
def test_rejects_spatial_signature_mismatch(hrf_glm_problem, signature):
    data, model, selection, _ = hrf_glm_problem
    with pytest.raises(ValueError, match="signature"):
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
        with pytest.raises(ValueError):
            values.setflags(write=True)
    design = result.group_designs[0, 1]
    design.iloc[:, :] = 0
    assert result.group_designs[0, 1].to_numpy().any()
    np.testing.assert_array_equal(result.hrf_indices, selection.hrf_indices)
    assert result.selection_provenance == selection.provenance
    info = result.provenance.to_dict()["activities"][-1]
    assert info["library_fingerprint"] == selection.library.fingerprint
    assert info["hrf_assignment_fingerprint"]
    assert info["design_fingerprint"]
    assert info["model"]["hrf_model"]["kind"] == "selected"
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


def test_signature_without_selection_is_not_silently_ignored(hrf_glm_problem):
    data, model, _, _ = hrf_glm_problem
    with pytest.raises(ValueError, match="selection"):
        fit(data, model, feature_signature="axis-v1")


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
    for (run, cid), actual in result.group_designs.items():
        expected = _oracle_design(
            data.events[run],
            data.frame_times[run],
            confounds[run],
            selection.library.candidates[cid],
            model,
        )
        pd.testing.assert_frame_equal(actual, expected)


def test_grouped_fit_rejects_missing_contrast_column(hrf_glm_problem):
    data, model, selection, _ = hrf_glm_problem
    with pytest.raises(ValueError, match="missing|invalid"):
        _selected_fit(data, replace(model, contrasts={"bad": "absent"}), selection)


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
