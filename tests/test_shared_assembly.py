"""Helpers that several fit paths share have one definition each."""

from hashlib import sha256
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from boldtailor._fractional_ridge import fit_fraction_run
from boldtailor._single_trial_fit import fit_trial_run
from boldtailor.hrf_results import hrf_assignment_fingerprint
from boldtailor.provenance import ProvenanceRecord
from boldtailor.single_trial_results import SharedTrialDesign, assemble_trial_result


@pytest.mark.parametrize("dtype", [np.int64, np.int32, "<i8", ">i8"])
def test_assignment_fingerprint_is_the_little_endian_int64_digest(dtype):
    """Every writer and checker of an HRF assignment hashes the same bytes."""
    ids = np.array([[2, -1, 0], [1, 1, 0]], dtype=dtype)
    expected = sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    assert hrf_assignment_fingerprint(ids) == expected
    assert hrf_assignment_fingerprint(ids.tolist()) == expected


@pytest.fixture
def run_problem():
    rng = np.random.default_rng(3)
    return rng.normal(size=(30, 4)), np.ones((30, 1)), rng.normal(size=(30, 5))


def test_run_fits_carry_implied_alphas_as_a_field(run_problem):
    """Fractional fits name their implied alphas; shared-alpha fits have none."""
    x, nuisance, signals = run_problem
    fractional = fit_fraction_run(x, nuisance, signals, fractions=np.full(5, 0.5))
    assert fractional.ridge_alphas.shape == (5,)
    np.testing.assert_array_equal(
        fractional.ridge_alphas, fractional.diagnostics["ridge_alphas"]
    )
    assert fit_trial_run(x, nuisance, signals, alpha=1.0).ridge_alphas is None


def _fit(full, null, total, alphas=None):
    return SimpleNamespace(
        betas=np.zeros((2, len(full))),
        full_sse=np.array(full),
        nuisance_sse=np.array(null),
        total_ss=np.array(total),
        ridge_alphas=alphas,
    )


@pytest.mark.parametrize("fractions", [None, np.array([0.5, 1.0])])
def test_one_assembler_pools_run_sums_of_squares(fractions):
    """Run R² come from each run; pooled R² come from summed sums of squares."""
    alphas = None if fractions is None else np.array([3.0, 0.0])
    fits = [_fit([1.0, 2.0], [2.0, 4.0], [4.0, 8.0], alphas)] * 2 + [
        _fit([3.0, 0.0], [6.0, 8.0], [12.0, 8.0], alphas)
    ]
    result = assemble_trial_result(
        fits,
        trial_table=pd.DataFrame({"trial_id": ["a"]}),
        design=SharedTrialDesign(_matrices=()),
        ridge_alpha=None if fractions is not None else 0.5,
        ridge_fraction=fractions,
        diagnostics=({"run": 0},),
        provenance=ProvenanceRecord(execution_id=str(uuid4())),
    )
    np.testing.assert_allclose(result.full_r2, [1 - 5 / 20, 1 - 4 / 24])
    np.testing.assert_allclose(result.nuisance_r2, [1 - 10 / 20, 1 - 16 / 24])
    np.testing.assert_allclose(result.delta_r2, result.full_r2 - result.nuisance_r2)
    np.testing.assert_allclose(result.run_full_r2[2], [0.75, 1.0])
    assert len(result.run_betas) == 3 and result.diagnostics == ({"run": 0},)
    if fractions is None:
        assert result.run_ridge_alphas is None and result.ridge_alpha == 0.5
    else:
        assert len(result.run_ridge_alphas) == 3
        np.testing.assert_array_equal(result.run_ridge_alphas[0], alphas)


def test_neither_single_trial_path_keeps_its_own_assembler():
    """Both paths call the shared assembler instead of a private copy."""
    from boldtailor import _selected_hrf_fit, _shared_hrf_fit

    assert not hasattr(_shared_hrf_fit, "_assemble_result")
    assert not hasattr(_selected_hrf_fit, "_result")
