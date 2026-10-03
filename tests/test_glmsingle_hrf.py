"""Empirical GLMsingle HRFs as library candidates and the default library."""

import numpy as np
import pytest

from boldtailor.hrf_library import (
    GLMSINGLE_HRF_COUNT,
    PARAMETER_NAMES,
    HrfCandidate,
    HrfLibrary,
    default_hrf_library,
    glmsingle_hrf_library,
    glmsingle_hrf_curves,
    timing_hrf_library,
)


def test_glmsingle_curves_are_the_published_library_at_unit_peak():
    curves = glmsingle_hrf_curves()
    assert curves.shape == (GLMSINGLE_HRF_COUNT, 501) and GLMSINGLE_HRF_COUNT == 20
    np.testing.assert_allclose(curves.max(axis=1), 1.0)
    assert not curves.flags.writeable
    # the published file is sampled at 0.1 s from stimulus onset; the first
    # HRF peaks earliest (2.7 s) and the last peaks latest (5.7 s)
    peaks = 0.1 * np.argmax(curves, axis=1)
    assert peaks.min() == pytest.approx(2.7) and peaks.max() == pytest.approx(5.7)


def test_glmsingle_candidate_kernel_is_the_curve_resampled_to_the_design_grid():
    candidate = HrfCandidate(3, "glmsingle", (7,))
    np.testing.assert_array_equal(candidate.kernel(0.1, 1), glmsingle_hrf_curves()[6])
    fine = candidate.kernel(1.6, 50)
    assert len(fine) == 50.0 / (1.6 / 50) and fine.max() == pytest.approx(1.0)
    assert not fine.flags.writeable
    grid = np.arange(len(fine)) * (1.6 / 50)
    np.testing.assert_allclose(
        fine, np.interp(grid, 0.1 * np.arange(501), glmsingle_hrf_curves()[6])
    )


@pytest.mark.parametrize("parameters", [(0,), (21,), (1.5,), (1, 2), ()])
def test_glmsingle_candidate_requires_a_valid_library_index(parameters):
    with pytest.raises(ValueError, match="GLMsingle"):
        HrfCandidate(1, "glmsingle", parameters)


def test_glmsingle_library_is_canonical_plus_twenty_empirical_kernels():
    library = glmsingle_hrf_library()
    assert len(library.candidates) == 21
    assert [c.kind for c in library.candidates[1:]] == ["glmsingle"] * 20
    assert [c.parameters for c in library.candidates[1:]] == [
        (k,) for k in range(1, 21)
    ]
    assert dict(library.origin) == {"kind": "glmsingle", "n_candidates": 20}
    table = library.parameter_table
    assert table.loc[1:, list(PARAMETER_NAMES)].isna().all().all()
    assert table.loc[1:, "source_index"].tolist() == list(range(1, 21))
    assert table.loc[0, "source_index"] != table.loc[0, "source_index"]  # NaN
    assert table.peak_time[1:].between(2.7, 5.7).all()


def test_timing_table_measures_empirical_kernels_and_leaves_gammas_blank():
    table = glmsingle_hrf_library().timing_table
    assert (
        table.loc[1:, ["response_peak", "response_sd", "response_delay"]]
        .isna()
        .all()
        .all()
    )
    assert table.loc[1:, "peak_time"].between(2.6, 5.8).all()
    assert table.loc[1:, "response_fwhm"].between(2.8, 5.0).all()
    # three GLMsingle HRFs have no undershoot; their trough columns are NaN
    assert table.loc[1:, "trough_depth"].isna().sum() == 3


def test_bound_diagnostics_ignore_empirical_kernels():
    library = HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]],
        include_glmsingle=True,
    )
    assert len(library.candidates) == 23
    assert library.parameter_bounds.loc["response_delay"].tolist() == [3.0, 6.0]
    flags = library.candidate_bound_flags()
    assert flags.shape == (23, 6, 2)
    assert flags[1:3].any() and not flags[3:].any() and not flags[0].any()


def test_default_library_is_timing_samples_plus_glmsingle_plus_canonical():
    library = default_hrf_library()
    timing = timing_hrf_library()
    assert len(library.candidates) == 1 + 512 + 20
    assert [c.parameters for c in library.candidates[:513]] == [
        c.parameters for c in timing.candidates
    ]
    assert [c.kind for c in library.candidates[513:]] == ["glmsingle"] * 20
    assert tuple(c.id for c in library.candidates) == tuple(range(533))
    origin = dict(library.origin)
    assert origin["kind"] == "default" and origin["glmsingle"] == 20
    assert origin["timing"] == dict(timing.origin)
    assert default_hrf_library() is library  # cached: the default is built once
    assert default_hrf_library(n_samples=16, seed=1).fingerprint != library.fingerprint


def test_library_rebuilds_exactly_from_its_parameter_table():
    for library in (default_hrf_library(n_samples=8), glmsingle_hrf_library()):
        rebuilt = HrfLibrary.from_table(library.parameter_table)
        assert rebuilt.fingerprint == library.fingerprint
        np.testing.assert_array_equal(rebuilt.curves, library.curves)
    plain = HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])
    assert HrfLibrary.from_table(plain.parameter_table).fingerprint == plain.fingerprint
