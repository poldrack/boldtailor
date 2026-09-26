"""Guard library coverage, notebook reproduction and owned numerical values."""

from dataclasses import FrozenInstanceError
from math import gamma

import numpy as np
import pytest
from nilearn.glm.first_level.hemodynamic_models import spm_hrf


def library_api():
    from boldtailor.hrf_library import HrfLibrary, expanded_hrf_library

    return HrfLibrary, expanded_hrf_library


def test_expanded_library_includes_legacy_anchor():
    _, expanded = library_api()
    library = expanded()
    assert len(library.candidates) == 649
    assert tuple(c.id for c in library.candidates) == tuple(range(649))
    assert library.candidates[0].kind == "spm"
    assert library.candidates[1].parameters == (3.0, 10.0, 0.5, 0.5, 2.0, 0.0, 36.0)
    assert library.candidates[-1].parameters == (6.0, 16.0, 1.5, 2.5, 8.0, 2.0, 36.0)
    np.testing.assert_array_equal(
        library.candidates[0].kernel(1.6, 50), spm_hrf(1.6, 50)
    )
    curves = library.curves
    assert curves.shape == (649, 360)
    assert np.isfinite(curves).all()
    np.testing.assert_allclose(curves.sum(axis=1), 1, atol=1e-14)
    assert len(np.unique(curves, axis=0)) == 649


@pytest.mark.parametrize(
    "parameters, reference",
    [
        (
            [3, 10, 0.5, 0.5, 2, 0, 36],
            [
                0.01443576360494855,
                0.06251737272184145,
                0.014386877454852731,
                -0.017745077913319016,
                -1.9199764534157618e-05,
                -8.772840212315701e-11,
            ],
        ),
        (
            [6, 16, 1.5, 2.5, 8, 2, 36],
            [
                0,
                0,
                0.01370810664336449,
                0.00877025021299874,
                -0.0006191775737963713,
                -0.00014941184380342322,
            ],
        ),
    ],
)
def test_custom_kernel_matches_notebook_and_independent_density(parameters, reference):
    cls, _ = library_api()
    kernel = cls.from_parameters([parameters]).candidates[1].kernel(0.1, 1)
    # Literal reference values from the supplied notebook's spm_hrf function.
    np.testing.assert_allclose(
        kernel[[10, 20, 50, 100, 200, 300]], reference, atol=1e-15
    )
    t = np.arange(360) * 0.1 - parameters[5]
    density = lambda a, scale: np.where(
        t > 0,
        np.maximum(t, 0) ** (a - 1)
        * np.exp(-np.maximum(t, 0) / scale)
        / (gamma(a) * scale**a),
        0,
    )
    expected = (
        density(parameters[0] / parameters[2], parameters[2])
        - density(parameters[1] / parameters[3], parameters[3]) / parameters[4]
    )
    np.testing.assert_allclose(kernel, expected / expected.sum(), atol=1e-15)


def test_library_owns_parameters_tables_and_curves():
    cls, _ = library_api()
    parameters = [[6, 16, 1, 1, 6, 0, 36]]
    library = cls.from_parameters(parameters)
    fingerprint = library.fingerprint
    parameters[0][0] = 9
    table = library.parameter_table
    table.loc[1, "response_delay"] = 9
    assert library.parameter_table.loc[1, "response_delay"] == 6
    assert library.fingerprint == fingerprint
    assert fingerprint == cls.from_parameters([[6, 16, 1, 1, 6, 0, 36]]).fingerprint
    assert fingerprint != cls.from_parameters(parameters).fingerprint
    with pytest.raises(ValueError):
        library.curves[1, 20] = 0
    with pytest.raises(ValueError):
        library.curves.setflags(write=True)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        library.candidates = ()


@pytest.mark.parametrize(
    "rows",
    [
        [[6, 16, 1, 1, 6, 0]],
        [[6, 16, 0, 1, 6, 0, 36]],
        [[6, 16, 1, 1, 0, 0, 36]],
        [[6, 16, 1, 1, 6, -1, 36]],
        [[6, 16, 1, 1, 6, 36, 36]],
        [[float("nan"), 16, 1, 1, 6, 0, 36]],
        [[6, 16, 1, 1, 6, 0, 0]],
        [[6, 16, 1, 1, 6, 0, 36]] * 2,
        [[6, 6, 1, 1, 1, 0, 36]],
    ],
)
def test_invalid_or_duplicate_parameters_are_rejected(rows):
    cls, _ = library_api()
    with pytest.raises(ValueError):
        cls.from_parameters(rows)


def test_parameter_order_is_stable_and_empty_grid_is_canonical():
    cls, _ = library_api()
    a, b = [6, 16, 1, 1, 6, 0, 36], [3, 10, 0.5, 0.5, 2, 0, 36]
    assert (
        cls.from_parameters([a, b]).fingerprint
        == cls.from_parameters([b, a]).fingerprint
    )
    assert len(cls.from_parameters([]).candidates) == 1


@pytest.mark.parametrize(
    "tr,oversampling", [(0, 50), (np.nan, 50), (1.6, 0), (1.6, 1.5), (1.6, True)]
)
def test_invalid_sampling_rejected(tr, oversampling):
    cls, _ = library_api()
    with pytest.raises(ValueError):
        cls.from_parameters([]).candidates[0].kernel(tr, oversampling)


@pytest.mark.parametrize(
    "parameters", [(3, 16, 1, 1, 6, 0, 32), (6, 16, 1, 1, 6, 0, 36)]
)
def test_direct_spm_candidate_cannot_mislabel_fixed_kernel(parameters):
    from boldtailor.hrf_library import HrfCandidate

    with pytest.raises(ValueError, match="canonical|SPM|spm"):
        HrfCandidate(0, "spm", parameters)
