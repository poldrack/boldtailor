"""Helpers used by examples are importable from public modules."""

import importlib

import pytest

PUBLIC = [
    ("design", "expand_events", "_task_design"),
    ("design", "task_columns", "_task_design"),
    ("design", "hrf_model", "_hrf_design"),
    ("design", "event_response_scales", "_hrf_design"),
    ("hrf_selection", "prepare_runs", "_hrf_cv"),
    ("hrf_selection", "subset_runs", "_hrf_cv"),
    ("fractional_ridge", "fraction_grid", "_fractional_ridge"),
    ("fractional_ridge", "regularization", "_fractional_ridge"),
    ("fractional_ridge", "NORM_BASIS", "_fractional_ridge"),
    ("single_trial", "r_squared", "_single_trial_fit"),
    ("single_trial", "validate_alpha", "_single_trial_fit"),
    ("single_trial", "compile_trial_run", "_single_trial_design"),
    ("model", "HRF_NORMALIZATION", "_hrf_design"),
    ("model", "is_boolean", "_scalars"),
    ("model", "is_integer", "_scalars"),
    ("model", "is_real", "_scalars"),
]


@pytest.mark.parametrize("public,name,origin", PUBLIC)
def test_public_module_reexports_the_implementation(public, name, origin):
    module = importlib.import_module(f"boldtailor.{public}")
    implementation = importlib.import_module(f"boldtailor.{origin}")
    assert getattr(module, name) is getattr(implementation, name)


ANNOTATED = ["hrf_selection", "_hrf_glm", "_selected_hrf_fit", "ridge_selection"]
ANNOTATED += ["fractional_ridge"]


@pytest.mark.parametrize("module_name", ANNOTATED)
def test_public_functions_are_annotated(module_name):
    import inspect

    module = importlib.import_module(f"boldtailor.{module_name}")
    functions = [
        value
        for name, value in vars(module).items()
        if inspect.isfunction(value)
        and not name.startswith("_")
        and value.__module__ == module.__name__
    ]
    assert functions
    for function in functions:
        signature = inspect.signature(function)
        assert signature.return_annotation is not inspect.Signature.empty, function
        for parameter in signature.parameters.values():
            if parameter.kind is not inspect.Parameter.VAR_KEYWORD:
                assert parameter.annotation is not inspect.Parameter.empty, (
                    function,
                    parameter.name,
                )
