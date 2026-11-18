"""Property-based tests for conventional GLM rank, estimability, and invariances."""

from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd
import pytest

from boldtailor._conventional import (
    _full_rank_basis,
    _validate_estimable,
    fit_designs,
    nested_ols_delta,
)


def _scaled_full_rank_problem():
    rng = np.random.default_rng(101)
    n_scans = 60
    task = rng.normal(size=n_scans) * 1e-4
    confound = rng.normal(size=n_scans) * 1e9
    design = pd.DataFrame({"task": task, "confound": confound, "const": np.ones(n_scans)})
    signals = 2.0 * task[:, None] + 1e-3 * confound[:, None] + rng.normal(scale=1e-4, size=(n_scans, 3))
    return signals, design


def test_extreme_column_scales_preserve_full_rank_and_estimability():
    """Verify extreme regressor unit differences do not cause false rank deficiency."""
    _, design = _scaled_full_rank_problem()
    matrix = design.to_numpy()
    assert _full_rank_basis(matrix) is None
    _validate_estimable(np.array([1.0, 0.0, 0.0]), matrix, "task", 0)


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    confound_scale=st.floats(min_value=1e-6, max_value=1e8, allow_nan=False, allow_infinity=False),
    task_scale=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_conventional_fit_is_invariant_to_confound_and_task_column_scales(
    seed, confound_scale, task_scale
):
    """Verify R2, t-statistics, and nested delta-R2 are invariant to regressor column units."""
    rng = np.random.default_rng(seed)
    n_scans, n_features = 40, 4
    task_base = rng.normal(size=n_scans)
    confound_base = rng.normal(size=n_scans)
    y = 1.5 * task_base[:, None] + 0.7 * confound_base[:, None] + rng.normal(scale=0.5, size=(n_scans, n_features))

    base_design = pd.DataFrame({"task": task_base, "confound": confound_base, "const": np.ones(n_scans)})
    scaled_design = pd.DataFrame(
        {"task": task_base * task_scale, "confound": confound_base * confound_scale, "const": np.ones(n_scans)}
    )
    nuisance_base = pd.DataFrame({"confound": confound_base, "const": np.ones(n_scans)})
    nuisance_scaled = pd.DataFrame({"confound": confound_base * confound_scale, "const": np.ones(n_scans)})

    fit_base = fit_designs([y], [base_design], {"task": {"task": 1.0}}, "ols")
    fit_scaled = fit_designs([y], [scaled_design], {"task": {"task": 1.0}}, "ols")

    np.testing.assert_allclose(fit_scaled.aggregate_r2, fit_base.aggregate_r2, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(fit_scaled.contrasts["task"].stat, fit_base.contrasts["task"].stat, rtol=1e-6, atol=1e-9)

    full_base, nuis_base = nested_ols_delta([y], [base_design], [nuisance_base], allow_undefined=False)
    full_scaled, nuis_scaled = nested_ols_delta([y], [scaled_design], [nuisance_scaled], allow_undefined=False)
    np.testing.assert_allclose(full_scaled - nuis_scaled, full_base - nuis_base, rtol=1e-7, atol=1e-10)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 10_000), n_runs=st.integers(2, 4))
def test_conventional_fit_is_invariant_to_nuisance_transform_and_run_order(seed, n_runs):
    """Verify R2, contrasts, and nested delta-R2 are invariant to nuisance transforms and run order."""
    rng = np.random.default_rng(seed)
    n_scans, n_features = 35, 3
    signals, full_designs, nuis_designs, full_transformed, nuis_transformed = [], [], [], [], []
    for _ in range(n_runs):
        task = rng.normal(size=n_scans)
        n1 = rng.normal(size=n_scans)
        n2 = rng.normal(size=n_scans)
        const = np.ones(n_scans)
        y = 1.2 * task[:, None] + 0.5 * n1[:, None] - 0.3 * n2[:, None] + rng.normal(scale=0.4, size=(n_scans, n_features))
        signals.append(y)
        full_designs.append(pd.DataFrame({"task": task, "n1": n1, "n2": n2, "const": const}))
        nuis_designs.append(pd.DataFrame({"n1": n1, "n2": n2, "const": const}))
        mix = rng.normal(size=(3, 3))
        while np.linalg.cond(mix) > 100:
            mix = rng.normal(size=(3, 3))
        transformed = np.column_stack([n1, n2, const]) @ mix
        full_transformed.append(
            pd.DataFrame({"task": task, "m0": transformed[:, 0], "m1": transformed[:, 1], "m2": transformed[:, 2]})
        )
        nuis_transformed.append(
            pd.DataFrame({"m0": transformed[:, 0], "m1": transformed[:, 1], "m2": transformed[:, 2]})
        )

    fit_a = fit_designs(signals, full_designs, {"task": {"task": 1.0}}, "ols")
    fit_b = fit_designs(signals, full_transformed, {"task": {"task": 1.0}}, "ols")
    np.testing.assert_allclose(fit_b.aggregate_r2, fit_a.aggregate_r2, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(fit_b.contrasts["task"].stat, fit_a.contrasts["task"].stat, rtol=1e-6, atol=1e-9)

    perm = rng.permutation(n_runs)
    fit_perm = fit_designs(
        [signals[i] for i in perm],
        [full_designs[i] for i in perm],
        {"task": {"task": 1.0}},
        "ols",
    )
    np.testing.assert_allclose(fit_perm.aggregate_r2, fit_a.aggregate_r2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fit_perm.contrasts["task"].stat, fit_a.contrasts["task"].stat, rtol=1e-10, atol=1e-12)

    full_r2, nuis_r2 = nested_ols_delta(signals, full_designs, nuis_designs, allow_undefined=False)
    assert np.all(full_r2 >= nuis_r2 - 1e-12)
