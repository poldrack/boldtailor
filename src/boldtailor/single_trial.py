"""Public entry points for shared- and selected-HRF single-trial fits."""

from collections.abc import Sequence

import numpy as np

from boldtailor._deprecation import UNSET, Unset, renamed_keyword
from boldtailor._single_trial_design import compile_trial_run  # public re-export
from boldtailor._single_trial_fit import (  # public re-exports
    project_trial_design,
    r_squared,
    validate_alpha,
)
from boldtailor._shared_hrf_fit import fit_shared_hrf_trials
from boldtailor._selected_hrf_fit import fit_selected_hrf_trials
from boldtailor.data import AnalysisData
from boldtailor.hrf_library import HrfCandidate
from boldtailor.hrf_results import HrfSelectionResult
from boldtailor.single_trial_results import SingleTrialResult


def fit_single_trials(
    data: AnalysisData,
    *,
    ridge_alpha: float = 0.0,
    run_labels: Sequence[str] | None = None,
    hrf_model: str | HrfCandidate | Unset = UNSET,
    ridge_fraction: float | np.ndarray | None = None,
    hrf: str | HrfCandidate | Unset = UNSET,
) -> SingleTrialResult:
    """Estimate one native-unit beta per event row, independently in each run.

    Nuisance regressors and an intercept are unpenalized. Positive alpha penalizes
    trial coefficients after nuisance projection and column normalization. RT and
    condition labels are retained as metadata and never enter the design.

    Alternatively, ridge_fraction specifies the coefficient-norm ratio to OLS
    in the raw trial-coefficient basis after nuisance projection, as a scalar or
    one value per feature. Values must
    be in (0, 1]; NaNs in a feature map exclude features. A positive ridge_alpha
    cannot be combined with ridge_fraction. Fractional results include the
    selected fractions and implied alpha for each run and feature.

    ``hrf_model`` defaults to 'spm', or can be one identified ``HrfCandidate`` used everywhere;
    the old keyword ``hrf`` is a deprecated alias.
    """
    resolved_hrf = renamed_keyword(
        "fit_single_trials", "hrf", "hrf_model", hrf, hrf_model, default="spm"
    )
    return fit_shared_hrf_trials(
        data,
        ridge_alpha=ridge_alpha,
        run_labels=run_labels,
        hrf_model=resolved_hrf,
        ridge_fraction=ridge_fraction,
    )


def fit_selected_hrfs(
    data: AnalysisData,
    *,
    hrf_selection: HrfSelectionResult | Unset = UNSET,
    ridge_alpha: float = 0.0,
    run_labels: Sequence[str] | None = None,
    feature_signature: str | None = None,
    ridge_fraction: float | np.ndarray | None = None,
    selection: HrfSelectionResult | Unset = UNSET,
) -> SingleTrialResult:
    """Fit unrestricted trial betas using each feature's previously selected HRF.

    The selection may come from different runs. A supplied spatial signature
    must match; for anonymous arrays, the caller must preserve feature order.
    OLS, fixed-alpha ridge, and fractional ridge use the identical selection
    and peak-normalized HRFs. ridge_fraction follows fit_single_trials semantics;
    its implied alpha is computed separately for each run and feature.
    The old keyword ``selection`` is a deprecated alias of ``hrf_selection``.
    """
    resolved_selection = renamed_keyword(
        "fit_selected_hrfs", "selection", "hrf_selection", selection, hrf_selection
    )
    return fit_selected_hrf_trials(
        data,
        selection=resolved_selection,
        ridge_alpha=ridge_alpha,
        run_labels=run_labels,
        feature_signature=feature_signature,
        ridge_fraction=ridge_fraction,
    )
