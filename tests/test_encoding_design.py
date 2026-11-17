"""An encoding design is prepared once from predictors and scores any betas."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from boldtailor import _ridge_cv
from boldtailor.ridge_selection import score_ridge_candidates
from boldtailor.trial_encoding import EncodingDesign, evaluate_trial_encoding

RESULT_ARRAYS = (
    "coefficients",
    "predictor_means",
    "run_sse",
    "run_sst",
    "r2",
    "train_run_predictor_means",
    "train_run_intercepts",
    "scoring_offsets",
)


@pytest.fixture
def problem(ridge_problem):
    data, predictors, _ = ridge_problem
    rng = np.random.default_rng(11)
    betas = [rng.normal(size=(len(p), 5)) for p in predictors]
    return predictors, betas


@pytest.mark.parametrize("mode", ["within_run", "absolute"])
def test_scoring_a_prepared_design_matches_the_one_step_evaluation(problem, mode):
    """Preparing first and scoring later is the same computation."""
    predictors, betas = problem
    design = EncodingDesign.prepare(
        predictors, train_runs=[0, 2, 4], test_runs=[1, 5], encoding_mode=mode
    )
    actual = design.score(betas)
    expected = evaluate_trial_encoding(
        betas, predictors, train_runs=[0, 2, 4], test_runs=[1, 5], encoding_mode=mode
    )
    for name in RESULT_ARRAYS:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for one, two in zip(actual.predictions, expected.predictions, strict=True):
        np.testing.assert_array_equal(one, two)
    assert actual.predictor_names == expected.predictor_names
    assert actual.encoding_mode == mode


def test_a_design_holds_only_predictor_statistics_and_is_frozen(problem):
    """The design depends on predictors alone, so it can score many beta sets."""
    predictors, betas = problem
    design = EncodingDesign.prepare(predictors, train_runs=[0, 1, 2], test_runs=[3])
    first = design.score(betas)
    assert design.predictor_names == ("task", *predictors[0].columns)
    np.testing.assert_array_equal(design.predictor_means, first.predictor_means)
    np.testing.assert_array_equal(
        design.train_run_predictor_means, first.train_run_predictor_means
    )
    assert len(design.trial_masks) == len(predictors)
    with pytest.raises(FrozenInstanceError):
        design.encoding_mode = "absolute"
    shifted = design.score([b + 1.0 for b in betas])
    np.testing.assert_array_equal(shifted.predictor_means, first.predictor_means)
    again = design.score(betas)
    np.testing.assert_array_equal(again.r2, first.r2)


def test_preparation_rejects_what_the_one_step_evaluation_rejects(problem):
    """Invalid splits and rank-deficient predictors fail before any betas are seen."""
    predictors, betas = problem
    with pytest.raises(ValueError, match="disjoint"):
        EncodingDesign.prepare(predictors, train_runs=[0, 1], test_runs=[1])
    constant = [p.assign(response_time=1.0) for p in predictors]
    with pytest.raises(ValueError, match="rank deficient"):
        EncodingDesign.prepare(constant, train_runs=[0, 1], test_runs=[2])
    design = EncodingDesign.prepare(predictors, train_runs=[0, 1], test_runs=[2])
    with pytest.raises(ValueError, match="predictor rows"):
        design.score([b[:-1] for b in betas])


def test_ridge_cv_prepares_one_design_and_one_table_copy_per_fold(
    ridge_problem, monkeypatch
):
    """Neither the encoding design nor the run tables depend on the candidate."""
    data, predictors, _ = ridge_problem
    prepared, copies = [], []
    prepare, tables = EncodingDesign.prepare, _ridge_cv.run_tables

    def observe_prepare(*args, **kwargs):
        prepared.append(kwargs["test_runs"])
        return prepare(*args, **kwargs)

    def observe_tables(*args):
        copies.append(1)
        return tables(*args)

    monkeypatch.setattr(EncodingDesign, "prepare", staticmethod(observe_prepare))
    monkeypatch.setattr(_ridge_cv, "run_tables", observe_tables)
    score_ridge_candidates(data, predictors, alphas=[0.0, 0.1, 1.0, 10.0])
    assert sorted(tuple(t) for t in prepared) == [(r,) for r in range(data.n_runs)]
    assert len(copies) == 1
