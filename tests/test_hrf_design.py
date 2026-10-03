"""Catch timing shifts, basis orthogonalization and metadata-dependent designs."""

import numpy as np
import pandas as pd
import pytest

from boldtailor._single_trial_design import compile_trial_run
from boldtailor.data import from_arrays
from boldtailor.single_trial import fit_single_trials
from tests.oracles import scaled_condition


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
def candidates(two_candidate_library):
    return two_candidate_library.candidates


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
            samples = max(1, round(dur / (1.6 / 50)))
            scale = 1 / np.convolve(np.ones(samples), kernel).max()
            expected_trials.append(
                scale * np.interp(times, grid, np.convolve(train, kernel)[:count])
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


@pytest.mark.parametrize("offset", [0.774, 0.775])
@pytest.mark.parametrize("irregular", [False, True])
def test_batched_trial_convolution_matches_nilearn_across_library(offset, irregular):
    from nilearn.glm.first_level import compute_regressor

    from boldtailor._hrf_design import trial_regressors
    from boldtailor.hrf_library import expanded_hrf_library

    times = offset + 1.6 * np.arange(188)
    if irregular:
        times[1:-1] += np.random.default_rng(412).uniform(-0.01, 0.01, len(times) - 2)
    events = pd.DataFrame(
        dict(
            onset=[times[0] - 24, -0.07, 8.13, 17.37, 99.15, 274.3],
            duration=[0, 3, 1.2, 0, 6.7, 3],
        )
    )
    for cid in [0, 1, 137, 291, 456, 648]:
        candidate = expanded_hrf_library().candidates[cid]
        actual = trial_regressors(events, times, candidate)
        expected = np.column_stack(
            [
                compute_regressor(
                    scaled_condition([o], [d], 1.0, candidate.kernel, times),
                    candidate.kernel,
                    times,
                )[0][:, 0]
                for o, d in zip(events.onset, events.duration, strict=True)
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=2e-12)


@pytest.mark.parametrize("name", ["spm", "glover"])
def test_string_models_resolve_to_peak_normalized_kernels(name):
    from boldtailor._hrf_design import hrf_kernel, hrf_model

    kernel = hrf_kernel(name, 2.0, 50)
    assert kernel.max() == pytest.approx(1.0)
    assert not kernel.flags.writeable
    model = hrf_model(name)
    assert callable(model)
    assert model.__name__ == "kernel"


def test_candidate_kernels_are_peak_normalized_through_hrf_kernel(candidates):
    from boldtailor._hrf_design import hrf_kernel, hrf_model

    for candidate in candidates:
        assert hrf_kernel(candidate, 1.6).max() == pytest.approx(1.0)
        assert hrf_model(candidate).__name__ == "kernel"


def test_trial_regressors_match_nilearn_with_the_peak_kernel(two_candidate_library):
    from nilearn.glm.first_level import compute_regressor

    from boldtailor._hrf_design import trial_regressors

    times = 0.5 + 1.6 * np.arange(120)
    events = pd.DataFrame(dict(onset=[10.0, 40.3, 77.1], duration=[1.0, 2.5, 0.0]))
    for candidate in two_candidate_library.candidates:
        fast = trial_regressors(events, times, candidate)
        for j, (o, d) in enumerate(zip(events.onset, events.duration, strict=True)):
            condition = scaled_condition([o], [d], 1.0, candidate.kernel, times)
            expected = compute_regressor(condition, candidate.kernel, times)[0][:, 0]
            np.testing.assert_allclose(fast[:, j], expected, atol=1e-10)


def test_canonical_metadata_records_peak_normalization():
    from boldtailor._hrf_design import hrf_metadata
    from boldtailor.hrf_library import CANONICAL_PARAMETERS

    metadata = hrf_metadata("spm")
    assert metadata["id"] == 0
    assert metadata["kind"] == "spm"
    assert metadata["parameters"] == list(CANONICAL_PARAMETERS)
    assert metadata["normalization"] == "peak_one_event_response"
    assert len(metadata["kernel_fingerprint"]) == 64


def test_custom_metadata_records_peak_normalization(candidates):
    from boldtailor._hrf_design import hrf_metadata

    assert hrf_metadata(candidates[1])["normalization"] == "peak_one_event_response"
    assert hrf_metadata(candidates[0]) == hrf_metadata("spm")


def test_single_trial_provenance_records_peak_normalization(timing_fixture):
    events, times, nuisance = timing_fixture
    data = from_arrays(
        np.random.default_rng(21).normal(size=(60, 2)),
        events,
        frame_times=times,
        confounds=nuisance,
    )
    activity = fit_single_trials(data).provenance.to_dict()["activities"][-1]
    assert activity["hrf_normalization"] == "peak_one_event_response"


@pytest.mark.parametrize("duration", [0.0, 0.05, 3.0, 40.0])
def test_event_response_peak_matches_brute_force_convolution(candidates, duration):
    from boldtailor._hrf_design import event_response_peak

    dt = 1.6 / 50
    for candidate in candidates:
        kernel = candidate.kernel(1.6, 50)
        n = max(1, int(round(duration / dt)))
        expected = np.convolve(np.ones(n), kernel).max()
        assert event_response_peak(kernel, duration, dt) == pytest.approx(
            expected, rel=1e-12
        )


def test_event_response_scales_invert_the_peak_per_duration(candidates):
    from boldtailor._hrf_design import event_response_peak, event_response_scales

    kernel_fn = candidates[1].kernel
    scales = event_response_scales(kernel_fn, [0.0, 3.0, 0.0], 1.6, 50)
    kernel = kernel_fn(1.6, 50)
    expected = [1 / event_response_peak(kernel, d, 1.6 / 50) for d in (0, 3, 0)]
    np.testing.assert_allclose(scales, expected, rtol=1e-12)


def _peak_models(library):
    return [*library.candidates, "spm", "glover"]


def test_single_event_responses_peak_at_one(two_candidate_library):
    from boldtailor._hrf_design import trial_regressors

    times = np.arange(0, 40, 0.1)
    for model in _peak_models(two_candidate_library):
        # A 3 s boxcar's peak may fall between 0.1 s frames, and Nilearn may
        # sample one more or fewer boxcar points than round(duration / dt):
        # allow 5 %. Impulses are bounded only by frame sampling of the
        # smooth peak (observed < 7e-5).
        long = pd.DataFrame(dict(onset=[0.0], duration=[3.0]))
        assert trial_regressors(long, times, model).max() == pytest.approx(
            1.0, rel=0.05
        )
        impulse = pd.DataFrame(dict(onset=[0.0], duration=[0.0]))
        assert trial_regressors(impulse, times, model).max() == pytest.approx(
            1.0, abs=1e-4
        )


def test_stimulus_regressor_scales_each_event_to_unit_peak(candidates):
    from boldtailor._hrf_design import stimulus_regressor

    times = np.arange(0, 80, 0.1)
    events = pd.DataFrame(dict(onset=[0.0, 40.0], duration=[0.0, 6.0]))
    for candidate in candidates:
        x = stimulus_regressor(events, times, candidate)
        assert x[times < 38].max() == pytest.approx(1.0, abs=1e-4)
        assert x[times >= 40].max() == pytest.approx(1.0, rel=0.05)
