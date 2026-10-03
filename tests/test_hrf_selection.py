"""Independent stacked-OLS oracles and strict train/test isolation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from boldtailor._hrf_design import stimulus_regressor
from boldtailor._single_trial_design import compile_trial_run
from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from tests.oracles import (
    replace_indices,
    loro_oracle,
    oracle_cv,
    stacked_training_ols_oracle,
    task_model_oracle,
)


@pytest.fixture
def cv_fixture(two_candidate_library):
    library = two_candidate_library
    rng = np.random.default_rng(510)
    signals, events, times, confounds = [], [], [], []
    for r, length in enumerate([65, 80, 73, 95]):
        t = 0.774 + 1.6 * np.arange(length)
        e = pd.DataFrame(
            dict(
                onset=np.array([8.13, 25.37, 45.22]) + r,
                duration=[3.0, 1.2, 2.0],
                image=np.arange(3) + r * 3,
                response_time=[0.7, np.nan, 1.5],
            )
        )
        n = pd.DataFrame(dict(motion=np.sin(np.arange(length) / 11 + r)))
        columns = []
        for c in library.candidates:
            x, _, _ = compile_trial_run(e, t, n, f"run-{r}", hrf=c)
            columns.append(x.to_numpy() @ (np.array([2.7, 3, 3.3]) + 0.06 * r))
        y = np.column_stack(columns)
        noise = rng.normal(size=y.shape) * 0.015
        for k in range(1, length):
            noise[k] += 0.3 * noise[k - 1]
        y += noise + (r + 1) * n.to_numpy() + 70 + r * 13
        signals.append(np.column_stack([y, np.full(length, 100.0), 3 * n.motion + 9]))
        events.append(e)
        times.append(t)
        confounds.append(n)
    data = from_arrays(signals, events, frame_times=times, confounds=confounds)
    return data, library


def replace_data(data, signals=None, events=None, confounds=None):
    return from_arrays(
        data.signals if signals is None else signals,
        data.events if events is None else events,
        frame_times=data.frame_times,
        confounds=data.confounds if confounds is None else confounds,
    )


@pytest.mark.parametrize("batch", [1, 2, 32])
def test_pooled_loro_matches_stacked_training_ols(cv_fixture, batch):
    from boldtailor.hrf_selection import select_hrfs

    data, library = cv_fixture
    result = select_hrfs(data, library=library, candidate_batch_size=batch)
    expected = oracle_cv(data, library)[:, :3]
    np.testing.assert_array_equal(result.hrf_indices[:3], [0, 1, 2])
    np.testing.assert_allclose(result.cv_r2[:3], expected.max(axis=0), atol=1e-12)
    np.testing.assert_allclose(result.canonical_cv_r2[:3], expected[0], atol=1e-12)
    np.testing.assert_allclose(
        result.delta_cv_r2[:3], expected.max(axis=0) - expected[0], atol=1e-12
    )
    assert np.all(result.hrf_indices[3:] == -1)
    assert np.isnan(result.cv_r2[3:]).all()


def test_noise_free_mean_response_and_rank_redundant_nuisance(cv_fixture):
    from boldtailor.hrf_selection import select_hrfs

    data, library = cv_fixture
    ys = [
        stimulus_regressor(e, t, library.candidates[2])[:, None] * 3
        + n.to_numpy() * 7
        + 31
        for e, t, n in zip(data.events, data.frame_times, data.confounds, strict=True)
    ]
    ns = [n.assign(duplicate=n.motion * 1e4) for n in data.confounds]
    clean = replace_data(data, signals=ys, confounds=ns)
    result = select_hrfs(clean, library=library)
    np.testing.assert_array_equal(result.hrf_indices, [2])
    np.testing.assert_allclose(result.cv_r2, 1, atol=1e-13)


def test_outer_test_changes_cannot_select_the_hrf(cv_fixture):
    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = cv_fixture
    ys = [np.array(y) for y in data.signals]
    es = list(data.events)
    for r in [1, 3]:
        ys[r] = np.random.default_rng(r).normal(size=ys[r].shape) * 100
        es[r] = es[r].assign(response_time=[99, 88, 77])
    changed = replace_data(data, signals=ys, events=es)
    args = dict(library=library, train_runs=[0, 2], test_runs=[1, 3])
    original = evaluate_hrf_split(data, **args)
    altered = evaluate_hrf_split(changed, **args)
    np.testing.assert_array_equal(
        original.training_selection.hrf_indices, altered.training_selection.hrf_indices
    )
    np.testing.assert_array_equal(
        original.training_amplitudes, altered.training_amplitudes
    )
    assert not np.allclose(original.test_r2[:3], altered.test_r2[:3])
    for v, cid in enumerate(original.training_selection.hrf_indices[:3]):
        beta, loss, null = task_model_oracle(
            data, library.candidates[cid], [0, 2], [1, 3]
        )
        np.testing.assert_allclose(
            original.training_amplitudes[0, v], beta[v], atol=1e-12
        )
        np.testing.assert_allclose(
            original.test_r2[v], 1 - loss[v] / null[v], atol=1e-12
        )


def test_amplitude_shift_is_not_refitted_on_test_runs(cv_fixture):
    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = cv_fixture
    ys = [
        stimulus_regressor(e, t, library.candidates[1])[:, None]
        * (3 if r % 2 == 0 else -9)
        + n.to_numpy() * 3
        + 40
        for r, (e, t, n) in enumerate(
            zip(data.events, data.frame_times, data.confounds, strict=True)
        )
    ]
    result = evaluate_hrf_split(
        replace_data(data, signals=ys),
        library=library,
        train_runs=[0, 2],
        test_runs=[1, 3],
    )
    np.testing.assert_array_equal(result.training_selection.hrf_indices, [1])
    np.testing.assert_allclose(result.training_amplitudes, 3, atol=1e-12)
    np.testing.assert_allclose(result.test_r2, 1 - 16 / 9, atol=1e-12)


@pytest.mark.parametrize(
    "train,test",
    [
        ([0], [1, 3]),
        ([0, 2], [2, 3]),
        ([0, 0], [1]),
        ([0, 4], [1]),
        ([0, 2], []),
        ([0, 2], [True]),
        ([0, 2], [1.0]),
    ],
)
def test_invalid_folds_rejected(cv_fixture, train, test):
    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = cv_fixture
    with pytest.raises(ValueError):
        evaluate_hrf_split(data, library=library, train_runs=train, test_runs=test)


def test_selection_requires_two_runs_and_valid_batch(cv_fixture):
    from boldtailor.hrf_selection import select_hrfs

    data, library = cv_fixture
    single = from_arrays(
        data.signals[0],
        data.events[0],
        frame_times=data.frame_times[0],
        confounds=data.confounds[0],
    )
    with pytest.raises(ValueError, match="two|2"):
        select_hrfs(single, library=library)
    for batch in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            select_hrfs(data, library=library, candidate_batch_size=batch)


def test_canonical_only_negative_scores_and_stable_ties(cv_fixture):
    from boldtailor.hrf_selection import select_hrfs

    data, _ = cv_fixture
    library = HrfLibrary.from_parameters([])
    ys = [
        stimulus_regressor(e, t, library.candidates[0])[:, None] * (-1) ** r
        for r, (e, t) in enumerate(zip(data.events, data.frame_times, strict=True))
    ]
    negative = replace_data(data, signals=ys)
    result = select_hrfs(negative, library=library)
    assert result.cv_r2[0] < 0
    assert result.hrf_indices[0] == 0
    # Custom candidates differing only after the acquired response are tied.
    tied = HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [3, 10, 0.5, 0.5, 2, 0, 37]]
    )
    e = pd.DataFrame(dict(onset=[75.0], duration=[2.0]))
    t = np.arange(60) * 1.6
    x = stimulus_regressor(e, t, tied.candidates[1])
    short = from_arrays([x[:, None], x[:, None]], [e, e], frame_times=[t, t])
    answer = select_hrfs(short, library=tied)
    assert answer.hrf_indices[0] == 1


def test_structurally_invalid_candidate_excluded_and_canonical_nan(cv_fixture):
    from boldtailor.hrf_selection import select_hrfs

    data, library = cv_fixture
    # Canonical trial columns are all absorbed by supplied nuisance columns.
    ns = []
    for e, t, n in zip(data.events, data.frame_times, data.confounds, strict=True):
        x, _, _ = compile_trial_run(e, t, n, "r")
        ns.append(pd.concat([n, x.rename(columns=lambda c: "null_" + c)], axis=1))
    altered = replace_data(data, confounds=ns)
    result = select_hrfs(altered, library=library)
    assert not np.any(result.hrf_indices == 0)
    assert np.isnan(result.canonical_cv_r2).all()
    assert np.isnan(result.delta_cv_r2).all()
    assert result.eligibility.loc[0, "eligible"] == False
    with pytest.raises(ValueError, match="eligib|estimab"):
        select_hrfs(altered, library=HrfLibrary.from_parameters([]))


def test_results_owned_metadata_independent_and_fingerprinted(cv_fixture):
    from boldtailor.hrf_selection import evaluate_hrf_split, select_hrfs

    data, library = cv_fixture
    result = select_hrfs(data, library=library, feature_signature="axis-a")
    es = [e.assign(response_time=[-2, -3, -4], image=[7, 7, 7]) for e in data.events]
    same = select_hrfs(
        replace_data(data, events=es), library=library, feature_signature="axis-a"
    )
    key = lambda r: r.provenance.to_dict()["activities"][-1]["design_fingerprint"]
    assert key(result) == key(same)
    es[0].loc[0, "onset"] += 0.3
    assert key(result) != key(
        select_hrfs(replace_data(data, events=es), library=library)
    )
    for array in (
        result.hrf_indices,
        result.cv_r2,
        result.canonical_cv_r2,
        result.delta_cv_r2,
    ):
        assert type(array) is np.ndarray
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
    table = result.eligibility
    table.loc[0, "reason"] = "changed"
    assert result.eligibility.loc[0, "reason"] != "changed"
    assert result.feature_signature == "axis-a"
    a = evaluate_hrf_split(data, library=library, train_runs=[0, 2], test_runs=[1, 3])
    b = evaluate_hrf_split(data, library=library, train_runs=[1, 3], test_runs=[0, 2])
    assert (
        a.provenance.to_dict()["activities"][-1]
        != b.provenance.to_dict()["activities"][-1]
    )


def nsd_model():
    from boldtailor.model import Modulator, TaskModel

    return TaskModel(
        (
            Modulator("response_time", missing="indicator"),
            Modulator("trial_type"),
        )
    )


def with_trial_types(data):
    events = [
        pd.DataFrame(
            dict(
                onset=np.array([8.13, 25.37, 45.22, 60.5, 80.1]) + r,
                duration=[3.0, 1.2, 2.0, 1.5, 2.2],
                trial_type=[0, 1, 1, 0, 1],
                response_time=[0.7, np.nan, 1.5, 0.9, 1.2],
            )
        )
        for r in range(data.n_runs)
    ]
    return replace_data(data, events=events)


def test_select_hrf_with_task_model_matches_task_model_oracle(cv_fixture):
    from boldtailor.hrf_selection import select_hrfs

    data, library = cv_fixture
    data = with_trial_types(data)
    result = select_hrfs(data, library=library, task_model=nsd_model())
    expected = loro_oracle(data, library, nsd_model())[:, :3]
    np.testing.assert_allclose(result.cv_r2[:3], expected.max(axis=0), atol=1e-10)
    np.testing.assert_allclose(result.canonical_cv_r2[:3], expected[0], atol=1e-10)
    assert result.task_model == nsd_model()
    activity = result.provenance.to_dict()["activities"][-1]
    assert activity["task_model"] == nsd_model().to_dict()
    assert activity["task_model_fingerprint"] == nsd_model().fingerprint
    assert activity["task_regressors"] == ["task", "response_time", "trial_type"]
    assert activity["profiled_regressors"] == ["missing_response_time"]
    assert activity["profiled_columns"] == (
        "fit in-sample per run with the candidate kernel; see user guide"
    )
    assert activity["min_onset"] == -24.0
    assert activity["oversampling"] == 50
    assert activity["score"] == "nuisance_adjusted_task_model_prediction_r2"
    assert activity["hrf_normalization"] == "peak_one_event_response"


def test_task_model_changes_selection_identity_and_rt_now_matters(cv_fixture):
    from boldtailor.hrf_selection import select_hrfs

    data, library = cv_fixture
    data = with_trial_types(data)

    def key(result):
        return result.provenance.to_dict()["activities"][-1]["design_fingerprint"]

    plain = select_hrfs(data, library=library)
    modeled = select_hrfs(data, library=library, task_model=nsd_model())
    assert key(plain) != key(modeled)
    changed = replace_data(
        data,
        events=[
            e.assign(response_time=[0.9, np.nan, 1.1, 0.8, 1.3]) for e in data.events
        ],
    )
    assert key(modeled) != key(
        select_hrfs(changed, library=library, task_model=nsd_model())
    )
    assert key(plain) == key(select_hrfs(changed, library=library))


def test_evaluate_split_with_task_model_returns_named_amplitude_rows(cv_fixture):
    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = cv_fixture
    data = with_trial_types(data)
    result = evaluate_hrf_split(
        data,
        library=library,
        train_runs=[0, 2],
        test_runs=[1, 3],
        task_model=nsd_model(),
    )
    assert result.amplitude_names == ("task", "response_time", "trial_type")
    assert result.training_amplitudes.shape == (3, data.n_features)
    assert not result.training_amplitudes.flags.writeable
    for v, cid in enumerate(result.training_selection.hrf_indices[:3]):
        beta, loss, null = stacked_training_ols_oracle(
            data, library, int(cid), nsd_model(), [0, 2], [1, 3]
        )
        np.testing.assert_allclose(
            result.training_amplitudes[:, v], beta[:, v], atol=1e-10
        )
        np.testing.assert_allclose(result.test_r2[v], 1 - loss[v] / null[v], atol=1e-10)
    assert np.isnan(result.training_amplitudes[:, 3:]).all()


def test_default_evaluation_amplitudes_are_one_row_named_task(cv_fixture):
    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = cv_fixture
    result = evaluate_hrf_split(
        data, library=library, train_runs=[0, 2], test_runs=[1, 3]
    )
    assert result.amplitude_names == ("task",)
    assert result.training_amplitudes.shape == (1, data.n_features)


def test_task_model_argument_is_validated(cv_fixture):
    from boldtailor.hrf_selection import evaluate_hrf_split, select_hrfs

    data, library = cv_fixture
    with pytest.raises(ValueError, match="TaskModel"):
        select_hrfs(data, library=library, task_model={"modulators": []})
    with pytest.raises(ValueError, match="TaskModel"):
        evaluate_hrf_split(
            data, library=library, train_runs=[0, 2], test_runs=[1], task_model="nsd"
        )


def test_task_model_selection_feeds_single_trial_fits(cv_fixture):
    from boldtailor.hrf_selection import select_hrfs
    from boldtailor.single_trial import fit_selected_hrfs

    data, library = cv_fixture
    data = with_trial_types(data)
    selection = select_hrfs(data, library=library, task_model=nsd_model())
    result = fit_selected_hrfs(data, hrf_selection=selection)
    assert result.run_betas[0].shape == (5, data.n_features)
    assert np.isfinite(result.run_betas[0][:, :3]).all()


def test_evaluation_result_rejects_mismatched_amplitude_rows(cv_fixture):
    from dataclasses import replace

    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = cv_fixture
    result = evaluate_hrf_split(
        data, library=library, train_runs=[0, 2], test_runs=[1, 3]
    )
    with pytest.raises(ValueError, match="amplitude"):
        replace(result, amplitude_names=("task", "extra"))
    with pytest.raises(ValueError, match="amplitude"):
        replace(result, training_amplitudes=result.training_amplitudes[0])


def test_selection_provenance_records_library_origin(cv_fixture):
    from boldtailor.hrf_selection import select_hrfs

    data, library = cv_fixture
    activity = select_hrfs(data, library=library).provenance.to_dict()["activities"][-1]
    assert activity["library"] == dict(library.origin)
    assert activity["library_fingerprint"] == library.fingerprint


def _edge_oracle(library, cid):
    bounds = library.parameter_bounds
    values = np.asarray(library.candidates[cid].parameters[:6])
    width = (bounds["high"] - bounds["low"]).to_numpy()
    low = values - bounds["low"].to_numpy() <= 0.02 * width
    high = bounds["high"].to_numpy() - values <= 0.02 * width
    return np.column_stack([low, high])


def _flag_oracle(library, indices):
    flags = np.zeros((len(indices), 6, 2), bool)
    for feature, cid in enumerate(indices):
        if cid > 0:
            flags[feature] = _edge_oracle(library, cid)
    return flags


def test_selection_flags_each_parameter_edge_of_the_selected_kernel(cv_fixture):
    from boldtailor.hrf_selection import select_hrfs

    data, library = cv_fixture
    selection = select_hrfs(data, library=library)
    flags = selection.parameter_bound_flags
    expected = _flag_oracle(library, selection.hrf_indices)
    assert expected.any()
    np.testing.assert_array_equal(flags, expected)
    assert flags.dtype == bool and not flags.flags.writeable
    assert not flags[selection.hrf_indices <= 0].any()
    # two-level parameters are uninformative, so no pick is flagged overall
    assert not selection.at_parameter_bound.any()
    assert selection.at_parameter_bound.dtype == bool
    assert not selection.at_parameter_bound.flags.writeable


def _grid_id(library, parameters):
    return next(c.id for c in library.candidates if c.parameters == parameters)


def test_expanded_grid_does_not_flag_every_custom_pick(selected_fixture):
    from boldtailor.hrf_library import expanded_hrf_library

    _, selection = selected_fixture
    library = expanded_hrf_library()
    interior = _grid_id(library, (4.5, 16.0, 1.0, 1.5, 4.0, 1.0, 36.0))
    edge = _grid_id(library, (3.0, 10.0, 1.0, 1.5, 4.0, 1.0, 36.0))
    picked = replace_indices(
        replace(selection, library=library), [interior, edge, interior, 0, -1]
    )
    custom = picked.hrf_indices > 0
    # the two-level undershoot_delay grid puts every custom pick at an edge
    assert picked.parameter_bound_flags[custom][:, 1].any(axis=-1).all()
    np.testing.assert_array_equal(
        picked.at_parameter_bound, [False, True, False, False, False]
    )
    assert picked.at_parameter_bound[custom].mean() < 1.0


def test_sobol_pick_near_response_delay_low_edge_flags_only_that_edge(
    selected_fixture,
):
    from boldtailor.hrf_library import sobol_hrf_library

    _, selection = selected_fixture
    library = sobol_hrf_library(64)
    cid = 2  # response_delay 3.055 against a sampled low edge of 3.036
    picked = replace_indices(replace(selection, library=library), [cid, 0, 0, 0, -1])
    expected = np.zeros((6, 2), bool)
    expected[0, 0] = True
    np.testing.assert_array_equal(_edge_oracle(library, cid), expected)
    np.testing.assert_array_equal(picked.parameter_bound_flags[0], expected)
    np.testing.assert_array_equal(
        picked.at_parameter_bound, [True, False, False, False, False]
    )


def test_parameter_bound_table_summarizes_the_flags(cv_fixture):
    from boldtailor.hrf_library import PARAMETER_NAMES
    from boldtailor.hrf_selection import select_hrfs

    data, library = cv_fixture
    selection = select_hrfs(data, library=library)
    table = selection.parameter_bound_table()
    assert list(table.columns) == ["parameter", "edge", "fraction_flagged"]
    assert len(table) == 12
    custom = selection.hrf_indices > 0
    flags = selection.parameter_bound_flags[custom]
    for row in table.itertuples():
        p = PARAMETER_NAMES.index(row.parameter)
        e = ("low", "high").index(row.edge)
        assert row.fraction_flagged * custom.sum() == pytest.approx(
            flags[:, p, e].sum()
        )
    assert list(table.parameter.unique()) == list(PARAMETER_NAMES[:6])


def test_select_hrfs_is_the_name_and_select_hrf_a_deprecated_alias(selected_fixture):
    from boldtailor import hrf_selection

    data, selection = selected_fixture
    options = dict(library=selection.library, feature_signature="ordered-axis")
    current = hrf_selection.select_hrfs(data, **options)
    with pytest.warns(DeprecationWarning, match="select_hrfs"):
        old = hrf_selection.select_hrf(data, **options)
    np.testing.assert_array_equal(current.hrf_indices, old.hrf_indices)
    np.testing.assert_array_equal(current.cv_r2, old.cv_r2)
    assert (
        current.provenance.analysis_fingerprint == old.provenance.analysis_fingerprint
    )


def test_evaluate_hrf_split_threads_candidate_batch_size(selected_fixture, monkeypatch):
    from boldtailor import hrf_selection

    data, selection = selected_fixture
    split = dict(library=selection.library, train_runs=[0, 1], test_runs=[2])
    default = hrf_selection.evaluate_hrf_split(data, **split)
    batches = []
    original = hrf_selection.signal_statistics

    def spy(runs, signals, batch):
        batches.append(batch)
        return original(runs, signals, batch)

    monkeypatch.setattr(hrf_selection, "signal_statistics", spy)
    small = hrf_selection.evaluate_hrf_split(data, **split, candidate_batch_size=7)
    assert batches and set(batches) == {7}
    np.testing.assert_array_equal(
        small.training_selection.hrf_indices, default.training_selection.hrf_indices
    )
    np.testing.assert_array_equal(small.test_r2, default.test_r2)
    with pytest.raises(ValueError, match="candidate_batch_size"):
        hrf_selection.evaluate_hrf_split(data, **split, candidate_batch_size=0)


def test_selection_defaults_to_the_default_library(cv_fixture):
    from boldtailor.hrf_library import default_hrf_library
    from boldtailor.hrf_selection import select_hrfs

    data, _ = cv_fixture
    result = select_hrfs(data, library=default_hrf_library(n_samples=8))
    default = select_hrfs.__kwdefaults__["library"]
    assert default is None
    assert result.library.fingerprint == default_hrf_library(n_samples=8).fingerprint
    with_default = select_hrfs(data)
    assert with_default.library.fingerprint == default_hrf_library().fingerprint
    activity = with_default.provenance.to_dict()["activities"][-1]
    assert activity["library"]["kind"] == "default"
