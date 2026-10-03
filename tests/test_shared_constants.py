"""Design constants and scalar predicates have one home each."""

import importlib
import inspect

import numpy as np
import pytest


def module(name):
    return importlib.import_module(f"boldtailor.{name}")


def test_hrf_design_owns_the_design_constants():
    design = module("_hrf_design")
    assert design.OVERSAMPLING == 50
    assert design.MIN_ONSET == -24.0
    assert design.TIE_TOLERANCE == 1e-12


@pytest.mark.parametrize(
    "consumer", ["_hrf_cv", "hrf_selection", "_task_design", "single_trial"]
)
def test_consumers_import_the_shared_constants(consumer):
    design = module("_hrf_design")
    used = module(consumer)
    for name in ("OVERSAMPLING", "MIN_ONSET"):
        if hasattr(used, name):
            assert getattr(used, name) is getattr(design, name)
    assert used.OVERSAMPLING is design.OVERSAMPLING


def test_defaults_come_from_the_shared_constants():
    design = module("_hrf_design")
    model = module("model").ModelSpec(contrasts={"task": "task"})
    assert model.oversampling == design.OVERSAMPLING
    assert model.min_onset == design.MIN_ONSET
    columns = inspect.signature(module("_task_design").task_columns).parameters
    assert columns["oversampling"].default == design.OVERSAMPLING
    assert columns["min_onset"].default == design.MIN_ONSET
    kernel = inspect.signature(module("hrf_library").HrfCandidate.kernel).parameters
    assert kernel["oversampling"].default == design.OVERSAMPLING


def test_ridge_penalty_ties_use_the_shared_tolerance(monkeypatch):
    selection = module("ridge_selection")
    scores = [[0.25], [0.2]]
    assert selection.select_ridge_penalty(scores, [1.0, 0.0]).ridge_alpha == 1.0
    monkeypatch.setattr(selection, "TIE_TOLERANCE", 0.1)
    assert selection.select_ridge_penalty(scores, [1.0, 0.0]).ridge_alpha == 0.0


def test_fraction_ties_use_the_shared_tolerance(monkeypatch):
    fractional = module("fractional_ridge")
    scores = [[0.2], [0.25]]
    assert (
        fractional.select_ridge_fractions(scores, [1.0, 0.5]).ridge_fraction[0] == 0.5
    )
    monkeypatch.setattr(fractional, "TIE_TOLERANCE", 0.1)
    assert (
        fractional.select_ridge_fractions(scores, [1.0, 0.5]).ridge_fraction[0] == 1.0
    )


@pytest.mark.parametrize(
    "value,boolean,integer,real",
    [
        (True, True, False, False),
        (np.bool_(False), True, False, False),
        (3, False, True, True),
        (np.int64(3), False, True, True),
        (3.5, False, False, True),
        (np.float32(1.0), False, False, True),
        ("3", False, False, False),
        (None, False, False, False),
    ],
)
def test_public_scalar_predicates(value, boolean, integer, real):
    model = module("model")
    assert model.is_boolean(value) is boolean
    assert model.is_integer(value) is integer
    assert model.is_real(value) is real


def test_model_options_accept_numpy_integers_stored_as_builtin_ints():
    # Requirement change over the Task 4.7 pin, which demanded builtin ints.
    spec = module("model").ModelSpec
    model = spec(
        contrasts={"task": "task"}, drift_order=np.int64(2), oversampling=np.int64(20)
    )
    assert (model.drift_order, model.oversampling) == (2, 20)
    assert type(model.drift_order) is int and type(model.oversampling) is int


def test_model_options_still_reject_booleans_and_floats():
    spec = module("model").ModelSpec
    with pytest.raises(ValueError, match="drift_order"):
        spec(contrasts={"task": "task"}, drift_order=np.float64(1.0))
    with pytest.raises(ValueError, match="oversampling"):
        spec(contrasts={"task": "task"}, oversampling=True)
