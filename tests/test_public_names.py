"""Helpers used by examples are importable from public modules."""

import importlib

import pytest

PUBLIC = [
    ("design", "expand_events", "_task_design"),
    ("design", "task_columns", "_task_design"),
    ("design", "hrf_model", "_hrf_design"),
    ("design", "event_response_scales", "_hrf_design"),
    ("hrf_selection", "prepare_runs", "_hrf_cv"),
    ("hrf_selection", "subset_runs", "_ridge_cv"),
    ("fractional_ridge", "fraction_grid", "_fractional_ridge"),
    ("fractional_ridge", "regularization", "_fractional_ridge"),
    ("fractional_ridge", "NORM_BASIS", "_fractional_ridge"),
    ("single_trial", "r_squared", "_single_trial_fit"),
    ("single_trial", "validate_alpha", "_single_trial_fit"),
    ("single_trial", "compile_trial_run", "_single_trial_design"),
    ("model", "HRF_NORMALIZATION", "_hrf_design"),
]


@pytest.mark.parametrize("public,name,origin", PUBLIC)
def test_public_module_reexports_the_implementation(public, name, origin):
    module = importlib.import_module(f"boldtailor.{public}")
    implementation = importlib.import_module(f"boldtailor.{origin}")
    assert getattr(module, name) is getattr(implementation, name)
