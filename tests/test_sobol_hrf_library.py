"""Continuous coverage and reproducibility of the approved Sobol HRF library."""

import numpy as np
import pytest
from nilearn.glm.first_level.hemodynamic_models import spm_hrf

from boldtailor import hrf_library


def sobol(*args, **kwargs):
    factory = getattr(hrf_library, "sobol_hrf_library", None)
    assert factory is not None, "The Sobol HRF factory is not implemented"
    return factory(*args, **kwargs)


@pytest.fixture(scope="module")
def library():
    return sobol()


def test_default_matches_approved_preview_and_canonical_anchor(library):
    """Verify default matches approved preview and canonical anchor."""
    assert library.curves.shape == (513, 360)
    assert np.isfinite(library.curves).all()
    np.testing.assert_allclose(library.curves.max(axis=1), 1, atol=1e-14)
    reference = spm_hrf(1.6, 50)
    np.testing.assert_allclose(
        library.candidates[0].kernel(1.6), reference / reference.max()
    )
    assert len(np.unique(library.curves, axis=0)) == 513


def test_sampling_spans_continuous_parameter_box(library):
    """Verify sampling spans continuous parameter box."""
    rows = np.array([c.parameters for c in library.candidates[1:]])
    lower = np.array([3, 10, 0.5, 0.5, 2, 0])
    width = np.array([3, 6, 1, 2, 6, 2])
    unit = (rows[:, :6] - lower) / width
    assert ((unit >= 0) & (unit < 1)).all()
    np.testing.assert_array_equal(rows[:, 6], np.full(512, 36))
    # One point in each marginal stratum rules out random or discrete-grid draws.
    for column in unit.T:
        np.testing.assert_array_equal(
            np.sort((column * 512).astype(int)), np.arange(512)
        )


def test_seed_controls_library_identity_without_affecting_global_rng(library):
    """Verify seed controls library identity without affecting global rng."""
    before = np.random.get_state()
    assert sobol().fingerprint == library.fingerprint
    assert sobol(seed=1).fingerprint != library.fingerprint
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("count", [1, 2, np.int64(8)])
def test_power_of_two_counts_and_numpy_integer_seed(count):
    """Verify power of two counts and NumPy integer seed."""
    result = sobol(count, seed=np.int64(7))
    assert len(result.candidates) == count + 1
    assert result.fingerprint == sobol(int(count), seed=7).fingerprint


@pytest.mark.parametrize("count", [0, -2, 3, 513, 2.0, True, np.bool_(False), None])
def test_invalid_sample_count_is_rejected(count):
    """Verify invalid sample count is rejected."""
    with pytest.raises(ValueError, match="n_samples.*power of two"):
        sobol(count)


@pytest.mark.parametrize("seed", [-1, 1.5, 0.0, True, np.bool_(False), None, "0"])
def test_invalid_seed_is_rejected(seed):
    """Verify invalid seed is rejected."""
    with pytest.raises(ValueError, match="seed.*nonnegative integer"):
        sobol(seed=seed)


def test_sobol_library_records_its_constructor():
    """Verify sobol library records its constructor."""
    library = sobol(n_samples=8, seed=3)
    assert dict(library.origin) == {
        "kind": "sobol",
        "n_samples": 8,
        "seed": 3,
        "duration": 36.0,
    }


def test_origin_is_frozen_and_excluded_from_identity():
    """Verify origin is frozen and excluded from identity."""
    explicit = hrf_library.HrfLibrary.from_parameters([(5, 15, 1, 1, 5, 0, 32)])
    noted = hrf_library.HrfLibrary.from_parameters(
        [(5, 15, 1, 1, 5, 0, 32)], origin={"kind": "custom"}
    )
    assert dict(explicit.origin) == {"kind": "explicit", "n_candidates": 1}
    assert dict(noted.origin) == {"kind": "custom"}
    assert explicit.fingerprint == noted.fingerprint
    with pytest.raises(TypeError):
        noted.origin["kind"] = "other"


def test_expanded_library_records_its_grid():
    """Verify expanded library records its grid."""
    assert dict(hrf_library.expanded_hrf_library().origin) == {"kind": "expanded_grid"}


def test_every_sampled_sobol_parameter_is_informative():
    """Verify every sampled sobol parameter is informative."""
    assert hrf_library.sobol_hrf_library(64).informative_parameters == tuple(
        hrf_library.PARAMETER_NAMES[:6]
    )


def test_origin_cannot_be_changed_through_nested_values_or_the_source():
    """Nested origin values are owned: neither the caller's mapping nor a value
    read from the library can alter what the library reports.
    """
    source = {"kind": "custom", "bounds": {"peak_time": [3.0, 8.0]}}
    library = hrf_library.HrfLibrary.from_parameters(
        [(5, 15, 1, 1, 5, 0, 32)], origin=source
    )
    source["bounds"]["peak_time"][0] = 99.0
    library.origin["bounds"]["peak_time"][1] = 99.0
    assert dict(library.origin) == {
        "kind": "custom",
        "bounds": {"peak_time": [3.0, 8.0]},
    }
