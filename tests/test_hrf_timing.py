"""Timing parameterization of the SPM double gamma and a timing-space sampler."""

import numpy as np
import pytest

from boldtailor.hrf_library import (
    CANONICAL_PARAMETERS,
    PARAMETER_NAMES,
    REALIZED_NAMES,
    TIMING_NAMES,
    HrfLibrary,
    realized_timing,
    sobol_hrf_library,
    spm_parameters,
    spm_parameters_from_realized,
    timing_hrf_library,
    timing_parameters,
)


@pytest.mark.parametrize(
    "spm",
    [
        CANONICAL_PARAMETERS,
        (3.0, 10.0, 0.5, 0.5, 2.0, 0.0, 36.0),
        (6.0, 16.0, 1.5, 2.5, 8.0, 2.0, 36.0),
        (4.2, 12.7, 0.9, 1.8, 3.3, 0.7, 36.0),
    ],
)
def test_spm_and_timing_parameterizations_round_trip_exactly(spm):
    timing = timing_parameters(spm)
    assert len(timing) == len(TIMING_NAMES) == 7
    np.testing.assert_allclose(spm_parameters(timing), spm, rtol=0, atol=1e-12)


def test_canonical_spm_has_the_expected_timing():
    timing = dict(zip(TIMING_NAMES, timing_parameters(CANONICAL_PARAMETERS)))
    assert timing["response_peak"] == pytest.approx(5.0)
    assert timing["response_sd"] == pytest.approx(np.sqrt(6.0))
    assert timing["undershoot_peak"] == pytest.approx(15.0)
    assert timing["undershoot_sd"] == pytest.approx(4.0)
    # closed form: gamma.pdf(15, 16) / 6 / gamma.pdf(5, 6) for shape/scale (16,1)/(6,1)
    assert timing["undershoot_depth"] == pytest.approx(0.09730, abs=1e-4)
    assert timing["onset"] == 0.0 and timing["duration"] == 32.0


def test_response_peak_predicts_the_sampled_kernel_peak():
    library = sobol_hrf_library(n_samples=64, seed=0)
    errors = []
    for candidate in library.candidates[1:]:
        kernel = candidate.kernel(0.1, 1)
        realized = 0.1 * np.argmax(kernel)
        errors.append(realized - timing_parameters(candidate.parameters)[0])
    assert np.quantile(np.abs(errors), 0.95) < 0.15


def test_realized_timing_measures_the_combined_curve():
    peak_time, fwhm, trough_time, undershoot_fwhm, depth = realized_timing(
        CANONICAL_PARAMETERS
    )
    assert peak_time == pytest.approx(5.0, abs=0.02)
    assert fwhm == pytest.approx(5.26, abs=0.05)
    assert trough_time == pytest.approx(15.75, abs=0.05)
    assert undershoot_fwhm == pytest.approx(7.36, abs=0.05)
    assert depth == pytest.approx(0.089, abs=0.002)
    kernel = HrfLibrary.from_parameters([]).candidates[0].kernel(0.1, 1)
    assert -kernel.min() == pytest.approx(depth, abs=2e-3)
    # half-maximum width measured independently on the 0.1 s kernel
    above = np.flatnonzero(kernel >= 0.5)
    assert 0.1 * (above[-1] - above[0]) == pytest.approx(fwhm, abs=0.15)


@pytest.mark.parametrize(
    "target",
    [
        (5.0, 5.0, 15.5, 7.5, 0.10, 0.0, 36.0),
        (4.5, 4.0, 13.0, 8.0, 0.20, 0.0, 36.0),
        (6.5, 5.5, 17.0, 10.0, 0.08, 0.0, 36.0),
        (5.5, 4.5, 14.0, 7.0, 0.30, 0.5, 36.0),
    ],
)
def test_realized_targets_are_hit_after_refinement(target):
    spm = spm_parameters_from_realized(target)
    peak_time, fwhm, trough_time, undershoot_fwhm, depth = realized_timing(spm)
    assert peak_time == pytest.approx(target[0], abs=0.01)
    assert fwhm == pytest.approx(target[1], abs=0.02)
    assert trough_time == pytest.approx(target[2], abs=0.01)
    assert undershoot_fwhm == pytest.approx(target[3], abs=0.02)
    assert depth == pytest.approx(target[4], rel=1e-3)
    assert spm[5] == target[5] and spm[6] == target[6]


def test_infeasible_realized_targets_are_rejected():
    # a trough 5 s after a 6 s peak cannot coexist with a 3.8 s response FWHM
    with pytest.raises(ValueError, match="feasible"):
        spm_parameters_from_realized((6.05, 3.81, 11.29, 6.10, 0.335, 0.0, 36.0))
    with pytest.raises(ValueError, match="feasible"):
        spm_parameters_from_realized((7.5, 3.0, 10.0, 6.0, 0.05, 0.0, 36.0))


def test_timing_sampler_hits_realized_bounds_and_records_rejections():
    bounds = {
        "peak_time": (3.5, 7.5),
        "response_fwhm": (3.0, 6.0),
        "trough_time": (11.0, 18.0),
        "undershoot_fwhm": (6.0, 12.0),
        "trough_depth": (0.05, 0.4),
    }
    library = timing_hrf_library(n_samples=64, seed=3, bounds=bounds)
    assert isinstance(library, HrfLibrary)
    assert library.candidates[0].kind == "spm" and len(library.candidates) == 65
    for candidate in library.candidates[1:]:
        realized = realized_timing(candidate.parameters)
        for value, name, slack in zip(
            realized, REALIZED_NAMES[:5], (0.02, 0.03, 0.02, 0.03, None), strict=True
        ):
            low, high = bounds[name]
            if slack is None:
                assert low * 0.99 <= value <= high * 1.01
            else:
                assert low - slack <= value <= high + slack
        assert candidate.parameters[5] == 0.0 and candidate.parameters[6] == 36.0
    origin = dict(library.origin)
    assert origin["rejected"] > 0 and origin["kind"] == "timing_sobol"
    assert origin["bounds"] == {k: list(v) for k, v in bounds.items()}
    assert origin["n_samples"] == 64 and origin["seed"] == 3


def test_timing_sampler_is_deterministic_and_rejects_bad_requests():
    first = timing_hrf_library(n_samples=16, seed=1)
    second = timing_hrf_library(n_samples=16, seed=1)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != timing_hrf_library(n_samples=16, seed=2).fingerprint
    with pytest.raises(ValueError, match="power of two"):
        timing_hrf_library(n_samples=12)
    with pytest.raises(ValueError, match="bounds"):
        timing_hrf_library(n_samples=16, bounds={"peak_time": (7.0, 3.0)})
    with pytest.raises(ValueError, match="bounds"):
        timing_hrf_library(n_samples=16, bounds={"not_a_timing_name": (0.0, 1.0)})


def test_timing_table_reports_both_parameterizations():
    library = timing_hrf_library(n_samples=8, seed=0)
    table = library.timing_table
    assert list(table.columns[:2]) == ["hrf_id", "kind"]
    assert set(TIMING_NAMES) <= set(table.columns)
    assert set(REALIZED_NAMES[:5]) <= set(table.columns)
    assert table.loc[0, "response_fwhm"] == pytest.approx(5.26, abs=0.05)
    assert set(PARAMETER_NAMES) <= set(table.columns)
    assert len(table) == 9 and table.loc[0, "response_peak"] == pytest.approx(5.0)
    assert table.loc[0, "peak_time"] == pytest.approx(5.0, abs=0.02)
