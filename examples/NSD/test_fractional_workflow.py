"""The NSD adapter preserves featurewise decisions through fitting."""

import numpy as np
import pytest

from boldtailor.fractional_ridge import (
    score_fraction_candidates,
    select_ridge_fractions,
)
from boldtailor.hrf_selection import select_hrfs
from boldtailor.trial_encoding import evaluate_trial_encoding
from boldtailor.single_trial import fit_single_trials, fit_selected_hrfs
from boldtailor.workflow.beta_series import fit_cv_beta_series, trial_predictors
from boldtailor.workflow.inputs import NSD_TASK_MODEL, load_session, load_block


def fit(runs, root, library=None, **kwargs):
    return fit_cv_beta_series(
        runs, root, library=library, fractions=[0.3, 0.7, 1], **kwargs
    )


@pytest.mark.parametrize("optimized", [False, True])
def test_fraction_workflow_matches_whole_array(
    six_run_dataset, cv_library, tmp_path, optimized
):
    root, prep = six_run_dataset
    runs = load_session(root, prep)
    library = cv_library if optimized else None
    result = fit(runs, root, library, block_size=1)
    for scope, indices in [
        ("odd", [0, 2, 4]),
        ("even", [1, 3, 5]),
        ("all", list(range(6))),
    ]:
        subset = [runs[r] for r in indices]
        scores = score_fraction_candidates(
            load_block(subset, root, np.arange(4)),
            trial_predictors(subset),
            library=library,
            fractions=[0.3, 0.7, 1],
        )
        expected = select_ridge_fractions(scores.cv_r2, scores.grid)
        actual = result["tuning"][scope]["selection"]
        np.testing.assert_allclose(actual.ridge_fraction, expected.ridge_fraction)
        np.testing.assert_allclose(actual.selected_r2, expected.selected_r2, atol=1e-10)
    data = load_block(runs, root, np.arange(4))
    # Outer scores use OLS targets with HRFs learned only on outer training runs.
    train, test = [0, 2, 4], [1, 3, 5]
    chosen = result["tuning"]["odd"]["selection"].ridge_fraction
    outer_selection = (
        None
        if library is None
        else select_hrfs(
            load_block([runs[r] for r in train], root, np.arange(4)),
            library=library,
            task_model=NSD_TASK_MODEL,
        )
    )
    fitter = fit_single_trials if outer_selection is None else fit_selected_hrfs
    kwargs = {} if outer_selection is None else {"hrf_selection": outer_selection}
    shrunk = fitter(data, ridge_fraction=chosen, **kwargs)
    ols = fitter(
        data, ridge_fraction=np.where(np.isfinite(chosen), 1.0, np.nan), **kwargs
    )
    mixed = [ols.run_betas[r] if r in test else shrunk.run_betas[r] for r in range(6)]
    outer = evaluate_trial_encoding(
        mixed, trial_predictors(runs), train_runs=train, test_runs=test
    )
    actual_outer = result["evaluation"]["odd_to_even"]
    for target_beta, r in zip(actual_outer["targets"], test):
        np.testing.assert_allclose(target_beta, ols.run_betas[r], atol=2e-6)
    np.testing.assert_allclose(actual_outer["run_sse"], outer.run_sse, atol=1e-8)
    np.testing.assert_allclose(actual_outer["run_sst"], outer.run_sst, atol=1e-8)
    for actual_beta, r in zip(actual_outer["betas"], test):
        np.testing.assert_allclose(actual_beta, shrunk.run_betas[r], atol=1e-8)
    fractions = result["tuning"]["all"]["selection"].ridge_fraction
    selection = (
        None
        if library is None
        else select_hrfs(data, library=library, task_model=NSD_TASK_MODEL)
    )
    expected = (
        fit_single_trials(data, ridge_fraction=fractions)
        if selection is None
        else fit_selected_hrfs(data, hrf_selection=selection, ridge_fraction=fractions)
    )
    for beta, reference in zip(result["final"]["betas"], expected.run_betas):
        np.testing.assert_allclose(beta, reference, atol=2e-6)
    np.testing.assert_allclose(
        result["final"]["run_ridge_alphas"], expected.run_ridge_alphas, atol=1e-9
    )
    assert result["final"]["ridge_alpha"] is None
    assert np.isnan(fractions[-1])
    assert np.isnan(result["final"]["run_ridge_alphas"][:, -1]).all()
    decision = result["tuning"]["all"]["provenance"].analysis_fingerprint
    assert decision
    assert (
        result["final"]["provenance"][0]["record"]["activities"][-1][
            "tuning_analysis_fingerprint"
        ]
        == decision
    )
