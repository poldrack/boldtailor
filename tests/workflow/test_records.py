"""Workflow results are frozen, named records with one statement of regularization."""

from dataclasses import FrozenInstanceError, is_dataclass

import numpy as np
import pytest

from boldtailor.task_detection import detect_task_model
from boldtailor.workflow import records, analysis, beta_series, inputs

RECORDS = ("GlmFit", "BetaFit", "OuterEvaluation", "TuningResult", "CvBetaResult")


@pytest.fixture
def session(four_runs, settings_for):
    root, _ = four_runs
    runs = inputs.load_session(settings_for(root))
    model = detect_task_model([r.events for r in runs])
    return root, runs, model, inputs.make_blocks(runs, block_size=2)


@pytest.fixture
def cv_session(six_run_dataset, settings_for):
    root, _ = six_run_dataset
    runs = list(inputs.load_session(settings_for(root)))
    return root, runs, detect_task_model([r.events for r in runs])


def cv_fit(cv_session, **grid):
    root, runs, model = cv_session
    return beta_series.fit_cv_beta_series(
        runs, root, task_model=model, library=None, block_size=2, **grid
    )


def assert_frozen(record, field):
    with pytest.raises(FrozenInstanceError):
        setattr(record, field, None)
    with pytest.raises(TypeError):
        record[field]


@pytest.mark.parametrize("name", RECORDS)
def test_assembled_results_are_frozen_dataclasses(name):
    """Every assembled workflow result is a frozen dataclass, not a mutable mapping."""
    record = getattr(records, name)
    assert is_dataclass(record) and record.__dataclass_params__.frozen
    assert not record.__dataclass_params__.eq  # arrays have no scalar equality


def test_glm_fit_is_assembled_once_and_cannot_be_rebound(session):
    """A GLM fit exposes its maps, designs and block provenance as fixed fields."""
    root, runs, model, blocks = session
    fitted = analysis.fit_glms(runs, root, blocks, inputs.glm_model(runs, model))
    assert isinstance(fitted, records.GlmFit)
    assert fitted.effects.shape == fitted.t.shape == (3, 4)
    assert fitted.r2.shape == (3, 4) and len(fitted.provenance) == len(blocks)
    assert set(fitted.designs) == {(run, 0) for run in range(len(runs))}
    assert_frozen(fitted, "effects")


def test_beta_fit_states_its_regularization_explicitly(session):
    """Shared-alpha and fractional fits differ by a field, not by which keys exist."""
    root, runs, model, blocks = session
    ridge = analysis.fit_beta_series(
        runs, root, blocks, task_model=model, ridge_alpha=0.5
    )
    fractions = np.full(runs[0].image.shape[1], 0.5)
    fractional = analysis.fit_beta_series(
        runs, root, blocks, task_model=model, ridge_fraction=fractions
    )
    assert isinstance(ridge, records.BetaFit)
    assert ridge.regularization == "normalized_ridge" and not ridge.fractional
    assert ridge.ridge_alpha == 0.5
    assert ridge.ridge_fraction is None and ridge.run_ridge_alphas is None
    assert fractional.regularization == "fractional_ridge" and fractional.fractional
    assert fractional.ridge_alpha is None
    np.testing.assert_array_equal(fractional.ridge_fraction, fractions)
    assert fractional.run_ridge_alphas.shape == (len(runs), len(fractions))
    assert ridge.hrf_selections is None and len(ridge.trial_table) == sum(
        len(r.events) for r in runs
    )
    assert_frozen(ridge, "betas")


@pytest.mark.parametrize(
    "grid,kind",
    [
        (dict(alphas=[0.0, 1.0]), "normalized_ridge"),
        (dict(fractions=[0.5, 1.0]), "fractional_ridge"),
    ],
)
def test_cv_result_carries_one_regularization_through_every_record(
    cv_session, grid, kind
):
    """Tuning, outer evaluations and the final fit agree on the regularization."""
    result = cv_fit(cv_session, **grid)
    fractional = kind == "fractional_ridge"
    assert isinstance(result, records.CvBetaResult)
    assert set(result.tuning) == {"odd", "even", "all"}
    assert set(result.evaluation) == {"odd_to_even", "even_to_odd"}
    for tuned in result.tuning.values():
        assert isinstance(tuned, records.TuningResult)
        assert tuned.regularization == kind and tuned.fractional is fractional
    for outer in result.evaluation.values():
        assert isinstance(outer, records.OuterEvaluation)
        assert outer.regularization == kind and outer.fractional is fractional
        assert (outer.ridge_alpha is None) is fractional
        assert (outer.ridge_fraction is None) is not fractional
        assert (outer.run_ridge_alphas is None) is not fractional
    assert result.final.regularization == kind
    assert_frozen(result, "final")
    assert_frozen(result.tuning["all"], "selection")
    assert_frozen(result.evaluation["odd_to_even"], "encoding_r2")


def test_outer_evaluation_is_complete_when_it_is_returned(cv_session):
    """Nothing is attached to an outer evaluation after it is built."""
    result = cv_fit(cv_session, alphas=[0.0, 1.0])
    for scope, split in (("odd", "odd_to_even"), ("even", "even_to_odd")):
        outer, tuned = result.evaluation[split], result.tuning[scope]
        assert (
            outer.tuning_analysis_fingerprint == tuned.provenance.analysis_fingerprint
        )
        assert outer.predictor_means.shape == (len(result.predictors[0].columns),)
        assert len(outer.trial_masks) == len(cv_session[1])
        assert len(outer.provenance) == 2  # one record per block of two features


def test_beta_model_reads_fractional_from_the_fit(session):
    """A model is fractional because its fit says so."""
    root, runs, model, blocks = session
    fit = analysis.fit_beta_series(runs, root, blocks, task_model=model)
    fractional = analysis.fit_beta_series(
        runs, root, blocks, task_model=model, ridge_fraction=np.full(4, 0.5)
    )
    build = beta_series.BetaModel
    assert not build(name="a", hrf="canonical", estimator="OLS", fit=fit).fractional
    assert build(
        name="b", hrf="canonical", estimator="FractionalCV", fit=fractional
    ).fractional
