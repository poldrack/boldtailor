"""Catch timing shifts, basis orthogonalization and metadata-dependent designs."""

import numpy as np
import pandas as pd
import pytest

from boldtailor._single_trial_design import compile_trial_run
from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.single_trial import fit_single_trials


@pytest.fixture
def timing_fixture():
    events = pd.DataFrame(
        dict(
            onset=[8.13, 17.37],
            duration=[3.0, 1.2],
            image=[9, 9],
            response_time=[0.5, np.nan],
        )
    )
    times = 0.774 + 1.6 * np.arange(60)
    return events, times, pd.DataFrame(dict(motion=np.linspace(-1, 1, 60)))


@pytest.fixture
def candidates():
    return HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    ).candidates


def test_default_single_trial_design_is_unchanged(timing_fixture, candidates):
    implicit = compile_trial_run(*timing_fixture, "run-01")
    for hrf in ("spm", candidates[0]):
        explicit = compile_trial_run(*timing_fixture, "run-01", hrf=hrf)
        for a, b in zip(implicit, explicit, strict=True):
            pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("offset", [0.774, 0.775])
@pytest.mark.parametrize("duration", [0.0, 1.2])
def test_custom_convolution_matches_independent_sampled_boxcars(
    timing_fixture, candidates, offset, duration
):
    from boldtailor._hrf_design import stimulus_regressor

    events, _, nuisance = timing_fixture
    events.loc[1, "duration"] = duration
    times = offset + 1.6 * np.arange(60)
    # Independent event sampling and direct convolution, matching Nilearn's
    # documented oversampled grid and sampled impulse convention.
    end = times[-1] * 60 / 59
    count = round((end - times[0] + 24) / (times[-1] - times[0]) * 59 * 50 + 1)
    grid = np.linspace(times[0] - 24, end, count)
    for candidate in candidates[1:]:
        kernel = candidate.kernel(1.6, 50)
        expected_trials = []
        for onset, dur in zip(events.onset, events.duration, strict=True):
            train = np.zeros(count)
            start = np.searchsorted(grid, onset)
            stop = max(start + 1, np.searchsorted(grid, onset + dur))
            train[start:stop] = 1
            expected_trials.append(
                np.interp(times, grid, np.convolve(train, kernel)[:count])
            )
        x, _, table = compile_trial_run(
            events, times, nuisance, "run-01", hrf=candidate
        )
        np.testing.assert_allclose(x, np.array(expected_trials).T, atol=1e-13)
        np.testing.assert_allclose(
            stimulus_regressor(events, times, candidate), x.sum(axis=1), atol=1e-13
        )
        assert np.all(x.to_numpy()[times < events.onset.min() - 1.6 / 50] == 0)
        pd.testing.assert_frame_equal(table[events.columns], events)
        changed = events.assign(image=[1, 2], response_time=[99, -99])
        np.testing.assert_array_equal(
            stimulus_regressor(changed, times, candidate),
            stimulus_regressor(events, times, candidate),
        )


def test_custom_boundary_and_run_isolation(timing_fixture, candidates):
    from boldtailor._hrf_design import stimulus_regressor

    events, times, nuisance = timing_fixture
    for onset in (times[0] - 24 - 0.001, times[-1], times[-1] - 0.001):
        bad = events.assign(onset=[onset, onset])
        with pytest.raises(ValueError, match="support|onset|sample"):
            compile_trial_run(bad, times, nuisance, "run-01", hrf=candidates[2])
        with pytest.raises(ValueError, match="support|onset|sample"):
            stimulus_regressor(bad, times, candidates[2])
    early = events.iloc[:1].assign(onset=times[0] - 24)
    x = stimulus_regressor(early, times, candidates[2])
    assert np.any(x)
    normal = stimulus_regressor(events, times, candidates[2])
    assert np.all(normal[times < 8] == 0)


def test_identified_hrf_fit_and_provenance(timing_fixture, candidates):
    events, times, nuisance = timing_fixture
    data = from_arrays(
        np.random.default_rng(20).normal(size=(60, 3)),
        events,
        frame_times=times,
        confounds=nuisance,
    )
    default = fit_single_trials(data)
    explicit = fit_single_trials(data, hrf=candidates[0])
    custom = fit_single_trials(data, hrf=candidates[1])
    np.testing.assert_array_equal(default.run_betas[0], explicit.run_betas[0])
    assert not np.allclose(custom.run_betas[0], default.run_betas[0])
    metadata = custom.provenance.to_dict()["activities"][-1]
    assert metadata["hrf"]["parameters"] == list(candidates[1].parameters)
    assert (
        metadata["design_fingerprint"]
        != default.provenance.to_dict()["activities"][-1]["design_fingerprint"]
    )


def test_unidentified_hrfs_are_rejected(timing_fixture):
    for hrf in ("glover", lambda tr, oversampling: np.ones(10)):
        with pytest.raises(ValueError, match="HRF|hrf"):
            compile_trial_run(*timing_fixture, "run-01", hrf=hrf)
