"""One regularizer strategy describes each penalty family to every caller."""

import ast
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import boldtailor
from boldtailor import _regularizers, _ridge_cv
from boldtailor._fractional_ridge import PreparedFractionBetas, prepare_fraction_betas
from boldtailor._regularizers import FRACTIONAL_RIDGE, NORMALIZED_RIDGE
from boldtailor.fractional_ridge import (
    score_fraction_candidates,
    select_ridge_fractions,
)
from boldtailor.ridge_selection import score_ridge_candidates, select_ridge_penalty

FAMILIES = [
    (NORMALIZED_RIDGE, score_ridge_candidates, "alphas", [1.0, 0.0]),
    (FRACTIONAL_RIDGE, score_fraction_candidates, "fractions", [0.5, 1.0]),
]


def test_the_two_families_differ_only_in_their_declared_fields():
    """Each family names its grid, its fit option and its validation target."""
    assert NORMALIZED_RIDGE.name == "normalized_ridge"
    assert NORMALIZED_RIDGE.grid_name == "alphas"
    assert NORMALIZED_RIDGE.fit_option == "ridge_alpha"
    assert NORMALIZED_RIDGE.validation_target == "candidate_regularized_betas"
    assert not NORMALIZED_RIDGE.per_feature
    assert FRACTIONAL_RIDGE.name == "fractional_ridge"
    assert FRACTIONAL_RIDGE.grid_name == "fractions"
    assert FRACTIONAL_RIDGE.fit_option == "ridge_fraction"
    assert FRACTIONAL_RIDGE.validation_target == "fixed_ols_betas"
    assert FRACTIONAL_RIDGE.per_feature


@pytest.mark.parametrize("family", [NORMALIZED_RIDGE, FRACTIONAL_RIDGE])
def test_families_are_found_by_name(family):
    """Results that record a regularization name lead back to its family."""
    assert _regularizers.regularizer_named(family.name) is family
    with pytest.raises(ValueError, match="unknown regularization"):
        _regularizers.regularizer_named("lasso")


def test_grids_are_validated_and_ordered_from_least_to_most_shrinkage():
    """Alphas ascend and fractions descend, so candidate zero shrinks least."""
    assert NORMALIZED_RIDGE.grid([1.0, 0.0, 0.1]) == (0.0, 0.1, 1.0)
    assert FRACTIONAL_RIDGE.grid([0.3, 1, 0.7]) == (1.0, 0.7, 0.3)
    with pytest.raises(ValueError, match="alphas"):
        NORMALIZED_RIDGE.grid([0.1, 0.1])
    with pytest.raises(ValueError):
        NORMALIZED_RIDGE.grid([-1.0])
    with pytest.raises(ValueError, match="fraction"):
        FRACTIONAL_RIDGE.grid([0.0, 1.5])


@pytest.mark.parametrize("family,scorer,keyword,values", FAMILIES)
def test_the_shared_scorer_takes_the_family_instead_of_a_flag(
    ridge_problem, family, scorer, keyword, values
):
    """The public scorers are the shared scorer applied to one family."""
    data, predictors, _ = ridge_problem
    expected = scorer(data, predictors, **{keyword: values})
    actual = _ridge_cv.score_candidates(
        data, predictors, family.grid(values), None, None, None, regularizer=family
    )
    assert actual.regularization == expected.regularization == family.name
    assert actual.grid == expected.grid
    np.testing.assert_array_equal(actual.cv_r2, expected.cv_r2)
    np.testing.assert_array_equal(actual.fold_sse, expected.fold_sse)
    activity = actual.provenance.to_dict()["activities"][-1]
    assert activity[family.grid_name] == list(family.grid(values))
    assert activity["validation_target"] == family.validation_target


def test_both_selections_name_their_chosen_penalty(ridge_problem):
    """A selection exposes its choice as ``penalty`` whatever the family."""
    data, predictors, _ = ridge_problem
    shared = select_ridge_penalty(
        score_ridge_candidates(data, predictors, alphas=[0.0, 1.0])
    )
    mapped = select_ridge_fractions(
        score_fraction_candidates(data, predictors, fractions=[0.5, 1.0])
    )
    assert shared.penalty == shared.ridge_alpha
    np.testing.assert_array_equal(mapped.penalty, mapped.ridge_fraction)


def _parameters(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = node.args
            for argument in (*arguments.args, *arguments.kwonlyargs):
                yield node.name, argument.arg


def test_no_function_threads_a_fractional_flag():
    """The family travels as a strategy or a record field, never as a boolean."""
    offenders = sorted(
        f"{path.name}:{function}"
        for path in Path(boldtailor.__file__).parent.rglob("*.py")
        for function, argument in _parameters(ast.parse(path.read_text()))
        if argument == "fractional"
    )
    assert not offenders, "\n".join(offenders)


def test_prepared_fraction_betas_retain_no_signal_sized_arrays():
    """A prepared path keeps the factorization, not copies of the run's signals."""
    rng = np.random.default_rng(5)
    x, nuisance = rng.normal(size=(40, 6)), np.ones((40, 1))
    signals = rng.normal(size=(40, 9))
    prepared = prepare_fraction_betas(x, nuisance, signals)
    assert isinstance(prepared, PreparedFractionBetas)
    held = [getattr(prepared, f.name) for f in fields(prepared)]
    assert not any(getattr(value, "shape", None) == signals.shape for value in held)
    assert prepared.betas_at(0.5).shape == (6, 9)


def test_each_family_states_its_tie_rule_once():
    """Tuning metadata and provenance quote the family's tie rule, not their own copy."""
    from boldtailor.workflow import manifest, ridge_provenance

    assert NORMALIZED_RIDGE.tie_rule.startswith("smallest alpha within")
    assert FRACTIONAL_RIDGE.tie_rule.startswith("largest fraction within")
    for module in (manifest, ridge_provenance):
        source = Path(module.__file__).read_text()
        assert "within 1e-12 of the maximum" not in source
