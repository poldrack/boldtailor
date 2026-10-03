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


def test_realized_timing_measures_the_sampled_curve():
    peak_time, trough_time, depth = realized_timing(CANONICAL_PARAMETERS)
    assert peak_time == pytest.approx(5.0, abs=0.02)
    assert 14.5 < trough_time < 16.5
    assert 0.08 < depth < 0.12
    kernel = HrfLibrary.from_parameters([]).candidates[0].kernel(0.1, 1)
    assert -kernel.min() == pytest.approx(depth, abs=2e-3)


@pytest.mark.parametrize(
    "target",
    [
        (5.0, 2.0, 15.0, 4.0, 0.2, 0.0, 36.0),
        (4.0, 1.2, 12.0, 3.0, 0.4, 0.0, 36.0),
        (7.0, 2.5, 17.0, 5.0, 0.08, 0.0, 36.0),
        (6.0, 1.5, 14.0, 3.5, 0.3, 0.5, 36.0),
    ],
)
def test_realized_targets_are_hit_after_refinement(target):
    spm = spm_parameters_from_realized(target)
    peak_time, trough_time, depth = realized_timing(spm)
    assert peak_time == pytest.approx(target[0], abs=0.01)
    assert trough_time == pytest.approx(target[2], abs=0.01)
    assert depth == pytest.approx(target[4], rel=1e-3)
    # the lobe SDs and onset are carried through unchanged
    lobe = timing_parameters(spm)
    assert lobe[1] == pytest.approx(target[1]) and lobe[3] == pytest.approx(target[3])
    assert lobe[5] == target[5] and lobe[6] == target[6]


def test_infeasible_realized_targets_are_rejected():
    with pytest.raises(ValueError, match="feasible"):
        spm_parameters_from_realized((7.5, 3.0, 10.0, 2.5, 0.05, 0.0, 36.0))


def test_timing_sampler_hits_realized_bounds_and_records_rejections():
    bounds = {
        "peak_time": (3.5, 7.5),
        "response_sd": (1.0, 3.0),
        "trough_time": (10.0, 18.0),
        "undershoot_sd": (2.5, 6.0),
        "trough_depth": (0.05, 0.5),
    }
    library = timing_hrf_library(n_samples=64, seed=3, bounds=bounds)
    assert isinstance(library, HrfLibrary)
    assert library.candidates[0].kind == "spm" and len(library.candidates) == 65
    for candidate in library.candidates[1:]:
        peak_time, trough_time, depth = realized_timing(candidate.parameters)
        lobe = timing_parameters(candidate.parameters)
        assert (
            bounds["peak_time"][0] - 0.02 <= peak_time <= bounds["peak_time"][1] + 0.02
        )
        assert (
            bounds["trough_time"][0] - 0.02
            <= trough_time
            <= bounds["trough_time"][1] + 0.02
        )
        assert (
            bounds["trough_depth"][0] * 0.99
            <= depth
            <= bounds["trough_depth"][1] * 1.01
        )
        assert bounds["response_sd"][0] <= lobe[1] <= bounds["response_sd"][1]
        assert bounds["undershoot_sd"][0] <= lobe[3] <= bounds["undershoot_sd"][1]
        assert lobe[5] == 0.0 and lobe[6] == 36.0
    origin = dict(library.origin)
    assert origin["rejected"] >= 0 and origin["kind"] == "timing_sobol"
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
    assert set(PARAMETER_NAMES) <= set(table.columns)
    assert len(table) == 9 and table.loc[0, "response_peak"] == pytest.approx(5.0)
    assert table.loc[0, "peak_time"] == pytest.approx(5.0, abs=0.02)
