"""Independent numerical and held-out-isolation checks for the experiment."""

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag
from scipy.optimize import brentq

from boldtailor._fractional_ridge import fraction_beta_path
from boldtailor.trial_encoding import evaluate_trial_encoding


def experiment():
    try:
        return importlib.import_module("examples.validation.fractional_ridge_ablation")
    except ImportError as error:
        pytest.fail(f"Experimental solver missing: {error}")


@pytest.fixture
def runs():
    rng = np.random.default_rng(128)
    designs, signals, predictors = [], [], []
    for r, n in enumerate((37, 43, 39, 41)):
        x = rng.normal(size=(n, 5)) * [1, 0.3, 4, 2, 0.1]
        x[:, 1] += x[:, 0] * 0.2
        nuisance = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
        p = np.arange(5.0) + r
        beta = np.column_stack([2 + 3 * p + r * 10, -1 - p + r])
        y = x @ beta + 20 + rng.normal(size=(n, 2))
        designs.append((x, nuisance))
        signals.append(y)
        predictors.append(pd.DataFrame({"value": p}))
    return designs, signals, predictors


def oracle(designs, signals, basis, fraction=None, alpha=None):
    """One augmented least-squares problem with block-diagonal run designs."""
    scales = []
    for x, n in designs:
        residual = x - n @ np.linalg.lstsq(n, x, rcond=None)[0]
        scales.append(
            np.linalg.norm(residual, axis=0)
            if basis == "normalized"
            else np.ones(x.shape[1])
        )
    x = block_diag(*[x / s for (x, n), s in zip(designs, scales)])
    n = block_diag(*[n for x, n in designs])
    matrix = np.column_stack([x, n])
    target = np.vstack(signals)

    def solve(a):
        penalty = np.column_stack(
            [np.sqrt(a) * np.eye(x.shape[1]), np.zeros((x.shape[1], n.shape[1]))]
        )
        return np.linalg.lstsq(
            np.vstack([matrix, penalty]),
            np.vstack([target, np.zeros((x.shape[1], target.shape[1]))]),
            rcond=None,
        )[0][: x.shape[1]]

    ols = solve(0)
    alphas = []
    beta = np.empty_like(ols)
    for v in range(target.shape[1]):
        a = (
            (
                0
                if fraction == 1
                else brentq(
                    lambda a: np.linalg.norm(solve(a)[:, v]) / np.linalg.norm(ols[:, v])
                    - fraction,
                    0,
                    1e7,
                    xtol=1e-12,
                )
            )
            if alpha is None
            else alpha[v]
        )
        alphas.append(a)
        beta[:, v] = solve(a)[:, v] / np.concatenate(scales)
    return np.split(beta, np.cumsum([len(s) for s in scales])[:-1]), np.array(alphas)


@pytest.mark.parametrize("basis", ["normalized", "raw"])
@pytest.mark.parametrize("scope", ["per_run", "pooled_train"])
@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_split_matches_augmented_oracle(runs, basis, scope, fraction):
    designs, signals, predictors = runs
    exp = experiment()
    prepared = exp.prepare_runs(designs, signals, basis=basis)
    actual = exp.fit_split(
        prepared,
        predictors,
        train=[0, 1],
        test=[2, 3],
        fraction=fraction,
        target="fixed_ols",
        scope=scope,
    )
    if scope == "pooled_train":
        expected, alpha = oracle(designs[:2], signals[:2], basis, fraction=fraction)
        np.testing.assert_allclose(
            actual["training_alphas"], np.tile(alpha, (2, 1)), rtol=1e-7, atol=1e-9
        )
        for a, b in zip(actual["train_betas"], expected):
            np.testing.assert_allclose(a, b, atol=1e-7)
    for j, r in enumerate([0, 1, 2, 3]):
        expected, a = oracle(
            [designs[r]],
            [signals[r]],
            basis,
            fraction=fraction,
            alpha=alpha if scope == "pooled_train" else None,
        )
        beta = actual["train_betas"][r] if r < 2 else actual["test_betas"][r - 2]
        np.testing.assert_allclose(beta, expected[0], atol=1e-7)
        if r >= 2:
            np.testing.assert_allclose(actual["test_alphas"][r - 2], a, atol=1e-7)


def test_raw_per_run_ablation_agrees_with_production(runs):
    designs, signals, predictors = runs
    exp = experiment()
    fits = [
        next(fraction_beta_path(x, n, y, fractions=[0.5]))[1]
        for (x, n), y in zip(designs, signals)
    ]
    expected = evaluate_trial_encoding(
        fits, predictors, train_runs=[0, 1], test_runs=[2, 3]
    )
    actual = exp.fit_split(
        exp.prepare_runs(designs, signals, basis="raw"),
        predictors,
        train=[0, 1],
        test=[2, 3],
        fraction=0.5,
        target="candidate",
        scope="per_run",
    )
    np.testing.assert_allclose(
        actual["coefficients"], expected.coefficients, atol=1e-10
    )
    np.testing.assert_allclose(actual["r2"], expected.r2, atol=1e-10)
    np.testing.assert_allclose(actual["test_fractions"], 0.5, atol=1e-10)


@pytest.mark.parametrize("scope", ["per_run", "pooled_train"])
def test_held_out_data_never_changes_training_state(runs, scope):
    designs, signals, predictors = runs
    exp = experiment()
    kwargs = dict(
        train=[0, 1], test=[2, 3], fraction=0.5, target="fixed_ols", scope=scope
    )
    before = exp.fit_split(
        exp.prepare_runs(designs, signals, basis="raw"), predictors, **kwargs
    )
    changed = [*signals[:2], signals[2] * -3 + 55, signals[3] * 7]
    after = exp.fit_split(
        exp.prepare_runs(designs, changed, basis="raw"), predictors, **kwargs
    )
    for key in ["training_alphas", "coefficients", "train_intercepts", "calibration"]:
        np.testing.assert_array_equal(before[key], after[key])
    for key in ["train_betas", "predictions", "calibrated_predictions"]:
        for a, b in zip(before[key], after[key]):
            np.testing.assert_array_equal(a, b)
    assert not np.allclose(before["sse"], after["sse"])
    altered_design = [*designs[:2], (designs[2][0] * 2, designs[2][1]), designs[3]]
    altered = exp.fit_split(
        exp.prepare_runs(altered_design, changed, basis="raw"), predictors, **kwargs
    )
    for key in ["training_alphas", "coefficients", "calibration"]:
        np.testing.assert_array_equal(before[key], altered[key])
    if scope == "pooled_train":
        np.testing.assert_array_equal(before["test_alphas"], altered["test_alphas"])
        assert not np.allclose(before["test_fractions"], altered["test_fractions"])


def test_fixed_targets_do_not_shrink_with_candidate(runs):
    designs, signals, predictors = runs
    exp = experiment()
    prepared = exp.prepare_runs(designs, signals, basis="normalized")
    kw = dict(train=[0, 1], test=[2, 3], target="fixed_ols", scope="per_run")
    a = exp.fit_split(prepared, predictors, fraction=1, **kw)
    b = exp.fit_split(prepared, predictors, fraction=0.2, **kw)
    for x, y in zip(a["targets"], b["targets"]):
        np.testing.assert_array_equal(x, y)
    assert not np.allclose(a["sse"], b["sse"])


def test_training_only_calibration_and_identity_fallback():
    exp = experiment()
    raw = np.array([[1, 1, 2], [2, 1, 3], [4, 1, 6]], float)
    ols = np.column_stack([3 * raw[:, 0] + 2, np.arange(3), -raw[:, 2]])
    mapping = exp.fit_calibration([raw], [ols])
    np.testing.assert_allclose(mapping, [[3, 1, 1], [2, 0, 0]], atol=1e-12)
    test = np.array([[10, 4, 5]], float)
    np.testing.assert_allclose(exp.apply_calibration([test], mapping)[0], [[32, 4, 5]])


def test_basis_and_pooling_are_distinct_models(runs):
    designs, signals, predictors = runs
    exp = experiment()
    fits = []
    for basis, scope in [
        ("normalized", "per_run"),
        ("raw", "per_run"),
        ("raw", "pooled_train"),
    ]:
        fits.append(
            exp.fit_split(
                exp.prepare_runs(designs, signals, basis=basis),
                predictors,
                train=[0, 1],
                test=[2, 3],
                fraction=0.5,
                target="candidate",
                scope=scope,
            )
        )
    assert not np.allclose(fits[0]["test_betas"], fits[1]["test_betas"])
    assert not np.allclose(fits[1]["test_betas"], fits[2]["test_betas"])
    assert not np.allclose(fits[2]["test_fractions"], 0.5)
