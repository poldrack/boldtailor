"""Fractional fits match an independent augmented-lstsq/root-finding oracle."""

import gc
import importlib
import weakref

import numpy as np
import pytest

from boldtailor._single_trial_design import compile_trial_run
from boldtailor.single_trial import fit_single_trials, fit_selected_hrfs
from tests.oracles import fraction_beta_path, fractional_beta_oracle as oracle


def fractional():
    try:
        return importlib.import_module("boldtailor._fractional_ridge")
    except ImportError as error:
        pytest.fail(f"Fractional ridge solver is missing: {error}")


@pytest.fixture
def regression():
    rng = np.random.default_rng(829)
    x = rng.normal(size=(55, 4))
    x[:, 1] = x[:, 0] + 0.2 * x[:, 1]
    x *= [1, 5, 0.2, 3]
    trend = np.linspace(-1, 1, len(x))
    n = np.column_stack([np.ones(len(x)), trend, trend])
    y = x @ np.array([[2, -4], [0.3, 2], [-1, 3], [4, 0.1]])
    y += rng.normal(size=y.shape) * 0.03 + n @ np.array([[100, 50], [9, 1], [0, 0]])
    return x, n, y


def test_fraction_path_matches_requested_norm_and_oracle(regression):
    x, n, y = regression
    outputs = list(fraction_beta_path(x, n, y, fractions=[1, 0.8, 0.2]))
    assert [row[0] for row in outputs] == [1, 0.8, 0.2]
    for fraction, betas, alphas in outputs:
        for v in range(2):
            expected, _, alpha = oracle(x, n, y[:, v], fraction)
            np.testing.assert_allclose(betas[:, v], expected, atol=1e-9)
            np.testing.assert_allclose(alphas[v], alpha, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        np.linalg.norm(outputs[1][1], axis=0) / np.linalg.norm(outputs[0][1], axis=0),
        0.8,
        atol=1e-10,
    )
    assert abs(outputs[1][2][0] - outputs[1][2][1]) > 0.01


def test_fraction_mapping_preserves_target_scaling_and_unpenalized_confounds(
    regression,
):
    x, n, y = regression
    solver = fractional()
    base = solver.fit_fraction_run(x, n, y, fractions=[0.3, 1])
    altered = solver.fit_fraction_run(x, n, y * [5, -2] + 321, fractions=[0.3, 1])
    np.testing.assert_allclose(altered.betas, base.betas * [5, -2], atol=1e-9)
    np.testing.assert_allclose(
        altered.diagnostics["ridge_alphas"],
        base.diagnostics["ridge_alphas"],
        atol=1e-10,
    )
    for v, fraction in enumerate([0.3, 1]):
        beta, nuisance, _ = oracle(x, n, y[:, v], fraction)
        np.testing.assert_allclose(base.betas[:, v], beta, atol=1e-9)
        np.testing.assert_allclose(
            n @ base.nuisance_betas[:, v], n @ nuisance, atol=1e-9
        )
        np.testing.assert_allclose(
            base.full_sse[v],
            np.sum((y[:, v] - x @ beta - n @ nuisance) ** 2),
            atol=1e-8,
        )


@pytest.mark.parametrize("prepared", [False, True])
def test_fraction_solver_handles_ill_conditioning_and_undefined_features(
    regression, prepared
):
    x, n, y = regression
    y = np.column_stack([y, np.ones(len(y)) * 100, n[:, 1]])
    fit = fractional().fit_fraction_run(x, n, y, fractions=[0.5, np.nan, 0.5, 0.5])
    if prepared:
        betas, alphas = (
            fractional().prepare_fraction_betas(x, n, y).solve([0.5, np.nan, 0.5, 0.5])
        )
        np.testing.assert_allclose(betas, fit.betas)
        np.testing.assert_allclose(alphas, fit.diagnostics["ridge_alphas"])
    assert np.isfinite(fit.betas[:, 0]).all()
    assert np.isnan(fit.betas[:, 1:]).all()
    assert np.isnan(fit.diagnostics["ridge_alphas"][1:]).all()
    t = np.arange(60)
    first = np.sin(t)
    second = first + 1e-8 * np.cos(t)
    design = np.column_stack([first, second])
    signal = (first - second)[:, None]
    nuisance = np.ones((60, 1))
    _, ols, _ = next(fraction_beta_path(design, nuisance, signal, fractions=[1]))
    _, shrunk, _ = next(fraction_beta_path(design, nuisance, signal, fractions=[0.4]))
    if prepared:
        solver = fractional().prepare_fraction_betas(design, nuisance, signal)
        np.testing.assert_allclose(solver.betas_at(1), ols)
        np.testing.assert_allclose(solver.betas_at(0.4), shrunk)
    assert np.linalg.norm(shrunk[:, 0]) / np.linalg.norm(ols[:, 0]) == pytest.approx(
        0.4, rel=1e-7
    )


@pytest.mark.parametrize(
    "fractions", [[0], [-0.1], [1.1], [np.inf], [True], [], [0.5, 0.5]]
)
def test_invalid_fraction_grids_fail(regression, fractions):
    x, n, y = regression
    with pytest.raises(ValueError):
        list(fraction_beta_path(x, n, y, fractions=fractions))


@pytest.mark.parametrize("selected", [False, True])
def test_public_fraction_fits_keep_trial_units_and_hrf_groups(
    selected_fixture, selected
):
    data, selection = selected_fixture
    fit = fit_selected_hrfs if selected else fit_single_trials
    kwargs = (
        dict(hrf_selection=selection, feature_signature="ordered-axis")
        if selected
        else {}
    )
    fractions = np.array([1, 0.7, 0.4, 0.2, np.nan])
    result = fit(data, ridge_fraction=fractions, **kwargs)
    assert result.ridge_alpha is None
    np.testing.assert_allclose(result.ridge_fraction, fractions, equal_nan=True)
    assert not result.ridge_fraction.flags.writeable
    assert all(not a.flags.writeable for a in result.run_ridge_alphas)
    for r, (y, events, times, confounds) in enumerate(
        zip(data.signals, data.events, data.frame_times, data.confounds)
    ):
        for v in range(4):
            hrf = (
                selection.library.candidates[selection.hrf_indices[v]]
                if selected
                else "spm"
            )
            x, n, _ = compile_trial_run(
                events, times, confounds, f"run-{r+1:02}", hrf=hrf
            )
            beta, _, alpha = oracle(np.asarray(x), np.asarray(n), y[:, v], fractions[v])
            np.testing.assert_allclose(result.run_betas[r][:, v], beta, atol=1e-8)
            np.testing.assert_allclose(result.run_ridge_alphas[r][v], alpha, atol=1e-9)
        assert np.isnan(result.run_betas[r][:, -1]).all()
    assert (
        result.provenance.to_dict()["activities"][-1]["regularization"]
        == "fractional_ridge"
    )
    activity = result.provenance.to_dict()["activities"][-1]
    assert (
        activity["fraction_norm_basis"]
        == "raw_trial_coefficients_after_nuisance_projection"
    )
    assert activity["normalization"] == "none_after_nuisance_projection"
    with pytest.raises(ValueError):
        fit(data, ridge_alpha=0.1, ridge_fraction=0.5, **kwargs)


def test_fraction_one_matches_existing_ols(selected_fixture):
    data, _ = selected_fixture
    ordinary = fit_single_trials(data)
    fractional_fit = fit_single_trials(data, ridge_fraction=1)
    for a, b in zip(ordinary.run_betas, fractional_fit.run_betas):
        np.testing.assert_allclose(a, b, atol=1e-11)
    np.testing.assert_allclose(ordinary.full_r2, fractional_fit.full_r2, atol=1e-11)


def test_prepared_fraction_betas_reuse_state_in_any_order(regression, monkeypatch):
    x, n, y = regression
    prepared = fractional().prepare_fraction_betas(x, n, y)
    expected = {
        fraction: np.column_stack(
            [oracle(x, n, y[:, feature], fraction)[0] for feature in range(y.shape[1])]
        )
        for fraction in (0.2, 1.0, 0.8)
    }

    def refactorization(*args, **kwargs):
        raise AssertionError("candidate evaluation must reuse prepared SVD")

    monkeypatch.setattr(np.linalg, "svd", refactorization)
    for fraction in (0.2, 1.0, 0.8, 0.2):
        betas = prepared.betas_at(fraction)
        np.testing.assert_allclose(betas, expected[fraction], atol=1e-9)
        reference = weakref.ref(betas)
        del betas
        gc.collect()
        assert reference() is None


@pytest.mark.parametrize(
    "fraction", [0, -0.1, 1.1, np.inf, np.nan, True, [], [0.5, 0.5, 0.5]]
)
def test_prepared_fraction_rejects_invalid_values(regression, fraction):
    prepared = fractional().prepare_fraction_betas(*regression)
    with pytest.raises(ValueError):
        prepared.betas_at(fraction)


@pytest.mark.parametrize("selected", [False, True])
def test_fraction_results_own_arrays_without_solver_mutation(
    selected_fixture, selected
):
    from dataclasses import replace

    data, selection = selected_fixture
    fit = fit_selected_hrfs if selected else fit_single_trials
    options = (
        dict(hrf_selection=selection, feature_signature="ordered-axis")
        if selected
        else {}
    )

    result = fit(data, ridge_fraction=[1, 0.7, 0.4, 0.2, np.nan], **options)
    fractions = result.ridge_fraction.copy()
    alphas = tuple(a.copy() for a in result.run_ridge_alphas)
    copied = replace(result, ridge_fraction=fractions, run_ridge_alphas=alphas)
    fractions[:] = 0.1
    for array in alphas:
        array[:] = 999
    np.testing.assert_array_equal(copied.ridge_fraction, result.ridge_fraction)
    for actual, expected in zip(
        copied.run_ridge_alphas, result.run_ridge_alphas, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable
    assert not copied.ridge_fraction.flags.writeable


def test_prepare_fraction_betas_factorizes_once(regression, monkeypatch):
    x, n, y = regression
    calls = []
    original = np.linalg.svd

    def counting(*args, **kwargs):
        calls.append(np.shape(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counting)
    fractional().prepare_fraction_betas(x, n, y)
    assert len(calls) == 2  # the nuisance span and the raw residualized design
    assert sorted(calls) == sorted([n.shape, x.shape])
