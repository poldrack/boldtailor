"""Retired aliases and test-only helpers are gone from the package."""

import importlib

import pytest

from boldtailor import _hrf_design, _ridge_cv, hrf_selection, reml
from boldtailor._run_design import RunDesign
from boldtailor.hrf_glm_results import HrfAnalysisResult
from boldtailor.single_trial_results import SelectedTrialDesign

REMOVED = [
    (hrf_selection, "select_hrf"),
    (SelectedTrialDesign, "matrix"),
    (HrfAnalysisResult, "group_design"),
    (_hrf_design, "stimulus_regressor"),
    (_hrf_design, "convolve_events"),
    (_ridge_cv, "prepare_run_beta_path"),
    (RunDesign, "trial_eligible"),
    (reml, "criterion_slope"),
    (reml, "_log_grid"),
]


@pytest.mark.parametrize(
    "owner,name", REMOVED, ids=[f"{o.__name__}.{n}" for o, n in REMOVED]
)
def test_retired_name_is_not_defined(owner, name):
    """One-release aliases and helpers that only tests called are not shipped."""
    assert not hasattr(owner, name)


def test_the_deprecation_module_is_gone():
    """With no shims left, there is nothing to deprecate through."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("boldtailor._deprecation")
