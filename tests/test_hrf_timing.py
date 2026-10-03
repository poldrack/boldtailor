"""Timing parameterization of the SPM double gamma and a timing-space sampler."""

import numpy as np
import pytest

from boldtailor.hrf_library import (
    CANONICAL_PARAMETERS,
    PARAMETER_NAMES,
    TIMING_NAMES,
    HrfLibrary,
    sobol_hrf_library,
    spm_parameters,
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


def test_timing_sampler_builds_a_valid_library_within_its_bounds():
    bounds = {
        "response_peak": (3.5, 7.5),
        "response_sd": (1.0, 3.0),
        "undershoot_peak": (10.0, 18.0),
        "undershoot_sd": (2.5, 6.0),
        "undershoot_depth": (0.05, 0.5),
    }
    library = timing_hrf_library(n_samples=64, seed=3, bounds=bounds)
    assert isinstance(library, HrfLibrary)
    assert library.candidates[0].kind == "spm" and len(library.candidates) == 65
    timing = np.array([timing_parameters(c.parameters) for c in library.candidates[1:]])
    for i, name in enumerate(TIMING_NAMES[:5]):
        low, high = bounds[name]
        assert np.all(timing[:, i] >= low - 1e-9) and np.all(
            timing[:, i] <= high + 1e-9
        )
    assert np.all(timing[:, 5] == 0.0) and np.all(timing[:, 6] == 36.0)
    assert dict(library.origin) == dict(
        kind="timing_sobol",
        n_samples=64,
        seed=3,
        onset=0.0,
        duration=36.0,
        bounds={k: list(v) for k, v in bounds.items()},
    )


def test_timing_sampler_is_deterministic_and_rejects_bad_requests():
    first = timing_hrf_library(n_samples=16, seed=1)
    second = timing_hrf_library(n_samples=16, seed=1)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != timing_hrf_library(n_samples=16, seed=2).fingerprint
    with pytest.raises(ValueError, match="power of two"):
        timing_hrf_library(n_samples=12)
    with pytest.raises(ValueError, match="bounds"):
        timing_hrf_library(n_samples=16, bounds={"response_peak": (7.0, 3.0)})
    with pytest.raises(ValueError, match="bounds"):
        timing_hrf_library(n_samples=16, bounds={"not_a_timing_name": (0.0, 1.0)})


def test_timing_table_reports_both_parameterizations():
    library = timing_hrf_library(n_samples=8, seed=0)
    table = library.timing_table
    assert list(table.columns[:2]) == ["hrf_id", "kind"]
    assert set(TIMING_NAMES) <= set(table.columns)
    assert set(PARAMETER_NAMES) <= set(table.columns)
    assert len(table) == 9 and table.loc[0, "response_peak"] == pytest.approx(5.0)
