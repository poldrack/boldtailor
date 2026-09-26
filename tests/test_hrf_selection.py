"""Independent stacked-OLS oracles and strict train/test isolation."""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag

from boldtailor._hrf_design import stimulus_regressor
from boldtailor._single_trial_design import compile_trial_run
from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary


@pytest.fixture
def cv_fixture():
    library = HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )
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


def oracle(data, candidate, train, test):
    xs, ns = [], []
    for e, t, n in zip(data.events, data.frame_times, data.confounds, strict=True):
        xs.append(stimulus_regressor(e, t, candidate))
        ns.append(np.column_stack([n, np.ones(len(t))]))
    design = np.column_stack(
        [np.concatenate([xs[r] for r in train]), block_diag(*[ns[r] for r in train])]
    )
    beta = np.linalg.lstsq(
        design, np.concatenate([data.signals[r] for r in train]), rcond=None
    )[0][0]
    loss, null = np.zeros(data.n_features), np.zeros(data.n_features)
    for r in test:
        y, n = data.signals[r], ns[r]
        residual = y - xs[r][:, None] * beta
        residual -= n @ np.linalg.lstsq(n, residual, rcond=None)[0]
        yr = y - n @ np.linalg.lstsq(n, y, rcond=None)[0]
        loss += np.sum(residual**2, axis=0)
        null += np.sum(yr**2, axis=0)
    return beta, loss, null


def oracle_cv(data, library):
    scores = []
    for c in library.candidates:
        folds = [
            oracle(data, c, [i for i in range(data.n_runs) if i != r], [r])
            for r in range(data.n_runs)
        ]
        scores.append(1 - sum(f[1] for f in folds) / sum(f[2] for f in folds))
    return np.array(scores)


@pytest.mark.parametrize("batch", [1, 2, 32])
def test_pooled_loro_matches_stacked_training_ols(cv_fixture, batch):
    from boldtailor.hrf_selection import select_hrf

    data, library = cv_fixture
    result = select_hrf(data, library=library, candidate_batch_size=batch)
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
    from boldtailor.hrf_selection import select_hrf

    data, library = cv_fixture
    ys = [
        stimulus_regressor(e, t, library.candidates[2])[:, None] * 3
        + n.to_numpy() * 7
        + 31
        for e, t, n in zip(data.events, data.frame_times, data.confounds, strict=True)
    ]
    ns = [n.assign(duplicate=n.motion * 1e4) for n in data.confounds]
    clean = replace_data(data, signals=ys, confounds=ns)
    result = select_hrf(clean, library=library)
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
        beta, loss, null = oracle(data, library.candidates[cid], [0, 2], [1, 3])
        np.testing.assert_allclose(original.training_amplitudes[v], beta[v], atol=1e-12)
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
    from boldtailor.hrf_selection import select_hrf

    data, library = cv_fixture
    single = from_arrays(
        data.signals[0],
        data.events[0],
        frame_times=data.frame_times[0],
        confounds=data.confounds[0],
    )
    with pytest.raises(ValueError, match="two|2"):
        select_hrf(single, library=library)
    for batch in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            select_hrf(data, library=library, candidate_batch_size=batch)


def test_canonical_only_negative_scores_and_stable_ties(cv_fixture):
    from boldtailor.hrf_selection import select_hrf

    data, _ = cv_fixture
    library = HrfLibrary.from_parameters([])
    ys = [
        stimulus_regressor(e, t, library.candidates[0])[:, None] * (-1) ** r
        for r, (e, t) in enumerate(zip(data.events, data.frame_times, strict=True))
    ]
    negative = replace_data(data, signals=ys)
    result = select_hrf(negative, library=library)
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
    answer = select_hrf(short, library=tied)
    assert answer.hrf_indices[0] == 1


def test_structurally_invalid_candidate_excluded_and_canonical_nan(cv_fixture):
    from boldtailor.hrf_selection import select_hrf

    data, library = cv_fixture
    # Canonical trial columns are all absorbed by supplied nuisance columns.
    ns = []
    for e, t, n in zip(data.events, data.frame_times, data.confounds, strict=True):
        x, _, _ = compile_trial_run(e, t, n, "r")
        ns.append(pd.concat([n, x.rename(columns=lambda c: "null_" + c)], axis=1))
    altered = replace_data(data, confounds=ns)
    result = select_hrf(altered, library=library)
    assert not np.any(result.hrf_indices == 0)
    assert np.isnan(result.canonical_cv_r2).all()
    assert np.isnan(result.delta_cv_r2).all()
    assert result.eligibility.loc[0, "eligible"] == False
    with pytest.raises(ValueError, match="eligib|estimab"):
        select_hrf(altered, library=HrfLibrary.from_parameters([]))


def test_results_owned_metadata_independent_and_fingerprinted(cv_fixture):
    from boldtailor.hrf_selection import select_hrf, evaluate_hrf_split

    data, library = cv_fixture
    result = select_hrf(data, library=library, feature_signature="axis-a")
    es = [e.assign(response_time=[-2, -3, -4], image=[7, 7, 7]) for e in data.events]
    same = select_hrf(
        replace_data(data, events=es), library=library, feature_signature="axis-a"
    )
    key = lambda r: r.provenance.to_dict()["activities"][-1]["design_fingerprint"]
    assert key(result) == key(same)
    es[0].loc[0, "onset"] += 0.3
    assert key(result) != key(
        select_hrf(replace_data(data, events=es), library=library)
    )
    for array in (
        result.hrf_indices,
        result.cv_r2,
        result.canonical_cv_r2,
        result.delta_cv_r2,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
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
