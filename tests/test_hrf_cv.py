"""Task-model statistics reduce to the mean-stimulus method and match stacked OLS."""

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import run_glm
from scipy.linalg import block_diag

from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.model import Modulator, TaskModel

NSD = TaskModel(
    (
        Modulator("response_time", center=True, missing="indicator"),
        Modulator("trial_type", center=False),
    )
)


@pytest.fixture
def task_fixture():
    """Four runs; runs 1 and 3 have one missing RT; three modulated features."""
    library = HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )
    rng = np.random.default_rng(93)
    signals, events, times, confounds = [], [], [], []
    for r, length in enumerate([70, 82, 76, 90]):
        t = 0.774 + 1.6 * np.arange(length)
        rt = np.array([0.7, 1.4, 1.1, 0.9, 1.6])
        if r % 2:
            rt[2] = np.nan
        e = pd.DataFrame(
            dict(
                onset=np.array([8.1, 25.4, 45.2, 66.3, 88.0]) + r,
                duration=[3.0, 1.2, 2.0, 1.5, 2.2],
                trial_type=[0, 1, 1, 0, 1],
                response_time=rt,
            )
        )
        n = pd.DataFrame(dict(motion=np.sin(np.arange(length) / 11 + r)))
        events.append(e)
        times.append(t)
        confounds.append(n)
        signals.append(rng.normal(size=(length, 3)) * 0.02 + 50)
    data = from_arrays(signals, events, frame_times=times, confounds=confounds)
    return data, library


def columns_for(data, library, cid, task_model):
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import expand_events, task_columns

    out = []
    for r, (e, t) in enumerate(zip(data.events, data.frame_times, strict=True)):
        frame = task_columns(
            expand_events(e, task_model, r), t, hrf_model(library.candidates[cid])
        )
        out.append(frame)
    return out


def generated(data, library, cid, task_model, amplitudes, profiled_gain=2.0):
    """Noise-free signals from the task model plus run-specific nuisance."""
    signals = []
    for r, frame in enumerate(columns_for(data, library, cid, task_model)):
        x = frame[list(task_model.regressor_names)].to_numpy() @ np.asarray(amplitudes)
        for name in task_model.profiled_names:
            if name in frame:
                x = x + profiled_gain * (r + 1) * frame[name].to_numpy()
        n = data.confounds[r].to_numpy()[:, 0]
        signals.append(np.column_stack([x + 3 * n + 40 + r, 2 * x - n + 9]))
    return from_arrays(
        signals, data.events, frame_times=data.frame_times, confounds=data.confounds
    )


def stacked_oracle(data, library, cid, task_model, train, test):
    """Nilearn OLS on a stacked training design, then frozen held-out prediction."""
    frames = columns_for(data, library, cid, task_model)
    xs, zs, ns = [], [], []
    for r, frame in enumerate(frames):
        xs.append(frame[list(task_model.regressor_names)].to_numpy())
        n = np.column_stack([data.confounds[r].to_numpy(), np.ones(len(frame))])
        profiled = [c for c in task_model.profiled_names if c in frame]
        zs.append(np.column_stack([n, frame[profiled].to_numpy()]) if profiled else n)
        ns.append(n)
    design = np.column_stack(
        [np.concatenate([xs[r] for r in train]), block_diag(*[zs[r] for r in train])]
    )
    y = np.concatenate([data.signals[r] for r in train])
    labels, results = run_glm(y, design, noise_model="ols")
    beta = results[labels[0]].theta[: xs[0].shape[1]]
    loss, null = np.zeros(data.n_features), np.zeros(data.n_features)
    for r in test:
        residual = data.signals[r] - xs[r] @ beta
        residual = residual - zs[r] @ np.linalg.lstsq(zs[r], residual, rcond=None)[0]
        yr = (
            data.signals[r]
            - ns[r] @ np.linalg.lstsq(ns[r], data.signals[r], rcond=None)[0]
        )
        loss += np.sum(residual**2, axis=0)
        null += np.sum(yr**2, axis=0)
    return beta, loss, null


def loro_oracle(data, library, task_model):
    scores = []
    for cid in range(len(library.candidates)):
        folds = [
            stacked_oracle(
                data,
                library,
                cid,
                task_model,
                [i for i in range(data.n_runs) if i != r],
                [r],
            )
            for r in range(data.n_runs)
        ]
        scores.append(1 - sum(f[1] for f in folds) / sum(f[2] for f in folds))
    return np.array(scores)


def test_statistics_shapes_and_profiled_columns(task_fixture):
    from boldtailor._hrf_cv import prepare_runs, signal_statistics

    data, library = task_fixture
    runs = prepare_runs(data, library, NSD)
    assert all(run.k == 3 for run in runs)
    assert [run.block(1).qp.shape[1] for run in runs] == [0, 1, 0, 1]
    assert list(runs[1].task_design(1).columns) == [
        "task",
        "response_time",
        "trial_type",
        "missing_response_time",
    ]
    a, b, c, energy = signal_statistics(runs, data.signals, 2)
    assert a.shape == (4, 3, 3, 3)
    assert b.shape == (4, 3, 3, 3)
    assert c.shape == (4, 3, 3)
    assert energy.shape == (4, 3)
    # Profiled columns only lower the candidate's C in runs that have them.
    np.testing.assert_allclose(c[0], np.broadcast_to(energy[0][None], c[0].shape))
    assert np.all(c[1] < energy[1][None])
    for run in runs:
        np.testing.assert_allclose(run.block(1).a, run.block(1).x.T @ run.block(1).x)
        assert np.allclose(run.q.T @ run.block(1).x, 0, atol=1e-12)
        assert np.allclose(run.block(1).qp.T @ run.block(1).x, 0, atol=1e-12)


@pytest.mark.parametrize("batch", [1, 2, 32])
def test_task_model_loro_matches_stacked_nilearn_ols(task_fixture, batch):
    from boldtailor._hrf_cv import loro_scores, prepare_runs, signal_statistics

    data, library = task_fixture
    runs = prepare_runs(data, library, NSD)
    a, b, c, energy = signal_statistics(runs, data.signals, batch)
    scores = loro_scores(a, b, c, energy)
    np.testing.assert_allclose(scores, loro_oracle(data, library, NSD), atol=1e-10)


def test_task_only_model_reproduces_mean_stimulus_statistics(task_fixture):
    from boldtailor._hrf_cv import loro_scores, prepare_runs, signal_statistics
    from tests.test_hrf_selection import oracle_cv

    data, library = task_fixture
    runs = prepare_runs(data, library)
    a, b, c, energy = signal_statistics(runs, data.signals, 32)
    assert a.shape[-2:] == (1, 1)
    np.testing.assert_allclose(c, np.broadcast_to(energy[:, None, :], c.shape))
    np.testing.assert_allclose(
        loro_scores(a, b, c, energy), oracle_cv(data, library), atol=1e-12
    )


def test_noise_free_task_model_scores_one_only_with_the_full_model(task_fixture):
    from boldtailor._hrf_cv import loro_scores, prepare_runs, signal_statistics

    data, library = task_fixture
    clean = generated(data, library, 2, NSD, amplitudes=[1.0, 4.0, -2.5])
    runs = prepare_runs(clean, library, NSD)
    full = loro_scores(*signal_statistics(runs, clean.signals, 32))
    np.testing.assert_allclose(full[2], 1, atol=1e-10)
    assert np.all(full[[0, 1]] < 1 - 1e-6)
    task_only = loro_scores(
        *signal_statistics(prepare_runs(clean, library), clean.signals, 32)
    )
    assert np.all(task_only[2] < 1 - 1e-6)


def test_singular_pooled_training_matrix_marks_candidate_ineligible():
    from boldtailor._hrf_cv import loro_scores, pooled_amplitude

    a = np.zeros((3, 2, 2, 2))
    b = np.zeros((3, 2, 2, 4))
    c = np.ones((3, 2, 4))
    energy = np.ones((3, 4))
    a[:, 0] = np.eye(2)
    a[:, 1] = [[1, 1], [1, 1]]
    b[:, 0] = 0.5
    b[:, 1] = 0.5
    amplitude, ok = pooled_amplitude(a[:2].sum(0), b[:2].sum(0))
    np.testing.assert_array_equal(ok, [True, False])
    np.testing.assert_allclose(amplitude[0], 0.5)
    np.testing.assert_array_equal(amplitude[1], 0)
    scores = loro_scores(a, b, c, energy)
    assert np.all(np.isfinite(scores[0]))
    assert np.all(scores[1] == -np.inf)


def test_run_level_rank_failures_are_candidate_specific(task_fixture):
    from boldtailor._hrf_cv import prepare_runs

    data, library = task_fixture
    frames = columns_for(data, library, 0, NSD)
    confounds = [
        n.assign(null_task=f["task"].to_numpy()) if r == 2 else n
        for r, (n, f) in enumerate(zip(data.confounds, frames, strict=True))
    ]
    altered = from_arrays(
        data.signals, data.events, frame_times=data.frame_times, confounds=confounds
    )
    runs = prepare_runs(altered, library, NSD)
    ok, reason = runs[2].eligible(0)
    assert not ok and "support" in reason
    assert runs[2].eligible(1) == (True, "")
    assert runs[0].eligible(0) == (True, "")
    assert np.all(runs[2].block(0).x == 0)
    assert np.all(runs[2].block(0).a == 0)


def test_profiled_column_inside_nuisance_span_is_ineligible(task_fixture):
    from boldtailor._hrf_cv import prepare_runs

    data, library = task_fixture
    frames = columns_for(data, library, 1, NSD)
    confounds = list(data.confounds)
    confounds[1] = confounds[1].assign(
        indicator=frames[1]["missing_response_time"].to_numpy()
    )
    altered = from_arrays(
        data.signals, data.events, frame_times=data.frame_times, confounds=confounds
    )
    ok, reason = prepare_runs(altered, library, NSD)[1].eligible(1)
    assert not ok and "profiled" in reason


def test_trial_eligibility_is_separate_from_task_model_eligibility(task_fixture):
    from boldtailor._hrf_cv import prepare_runs

    data, library = task_fixture
    run = prepare_runs(data, library, NSD)[0]
    assert run.trial_eligible(1) == (True, "")
    assert run.trial_matrix(1).shape == (len(data.frame_times[0]), 5)
    assert len(run.events) == 5


def test_cache_key_includes_task_model_and_modulator_values(task_fixture):
    from boldtailor._hrf_cv import prepare_runs

    data, library = task_fixture
    base = prepare_runs(data, library, NSD)[0]
    assert prepare_runs(data, library, NSD)[0] is base
    assert prepare_runs(data, library)[0] is not base
    assert prepare_runs(data, library)[0].fingerprint != base.fingerprint
    events = list(data.events)
    events[0] = events[0].assign(response_time=[0.9, 1.4, 1.1, 0.9, 1.6])
    changed = from_arrays(
        data.signals, events, frame_times=data.frame_times, confounds=data.confounds
    )
    assert prepare_runs(changed, library, NSD)[0].fingerprint != base.fingerprint
    assert (
        prepare_runs(changed, library)[0].fingerprint
        == prepare_runs(data, library)[0].fingerprint
    )


def test_onsets_outside_supported_window_are_rejected(task_fixture):
    from boldtailor._hrf_cv import prepare_runs

    data, library = task_fixture
    events = list(data.events)
    events[0] = events[0].assign(onset=events[0].onset - 40)
    early = from_arrays(
        data.signals, events, frame_times=data.frame_times, confounds=data.confounds
    )
    with pytest.raises(ValueError, match="supported sampled response"):
        prepare_runs(early, library, NSD)


def test_run_ineligible_candidate_scores_minus_inf(task_fixture):
    from boldtailor._hrf_cv import loro_scores, prepare_runs, signal_statistics

    data, library = task_fixture
    frames = columns_for(data, library, 0, NSD)
    confounds = [
        n.assign(null_task=f["task"].to_numpy()) if r == 2 else n
        for r, (n, f) in enumerate(zip(data.confounds, frames, strict=True))
    ]
    altered = from_arrays(
        data.signals, data.events, frame_times=data.frame_times, confounds=confounds
    )
    runs = prepare_runs(altered, library, NSD)
    scores = loro_scores(*signal_statistics(runs, altered.signals, 32))
    assert np.all(scores[0] == -np.inf)
    assert np.all(np.isfinite(scores[1]))


def test_prediction_loss_tolerance_sums_absolute_regressor_cross_terms():
    from boldtailor._hrf_cv import prediction_loss

    a = np.eye(2)[None]
    b = np.array([[[1e8], [-1e8]]])
    amplitude = np.ones((1, 2, 1))
    c = np.array([[-2.0 - 1e-9]])
    # The cross terms cancel, so only per-regressor magnitudes bound roundoff.
    np.testing.assert_array_equal(prediction_loss(a, b, c, amplitude), [[0.0]])
