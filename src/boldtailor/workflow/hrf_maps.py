"""Shared map schemas for CV and restricted-likelihood HRF selection."""

import numpy as np

from boldtailor.hrf_results import HrfSelectionResult

CV_SELECTION_NAMES = ["hrf_id", "selected_cv_r2", "canonical_cv_r2", "delta_cv_r2"]
REML_SELECTION_NAMES = [
    "hrf_id",
    "criterion",
    "canonical_criterion",
    "delta_criterion",
    "criterion_gap",
    "variance_ratio",
    "noise_variance",
    "effect_variance",
    "rho",
    "at_upper",
    "unrefined",
    "any_at_upper",
    "any_unrefined",
    "status",
    "canonical_status",
    "canonical_at_upper",
    "canonical_unrefined",
]
STATUS_CODES = {
    name: i
    for i, name in enumerate(
        (
            "ok",
            "variance_zero",
            "no_residual_signal",
            "numerically_unresolved_residual",
            "unidentifiable",
        )
    )
}


def selection_names(method):
    """Map labels must agree with the estimator that generated the numbers."""
    return REML_SELECTION_NAMES if method == "reml" else CV_SELECTION_NAMES


def selection_method(selections):
    methods = {bundle["all"].method for bundle in selections.values()}
    if len(methods) != 1:
        raise ValueError("selection blocks must have one common method")
    return methods.pop()


def _status_values(fit):
    return np.array([STATUS_CODES[s] for s in fit.status], dtype=float)


def selection_values(result: HrfSelectionResult):
    ids = np.where(result.hrf_indices >= 0, result.hrf_indices, np.nan)
    if result.method == "cv":
        # HrfSelectionResult validates the method-specific fields on construction.
        assert result.cv_r2 is not None
        assert result.canonical_cv_r2 is not None
        assert result.delta_cv_r2 is not None
        return np.stack([ids, result.cv_r2, result.canonical_cv_r2, result.delta_cv_r2])
    diagnostics = result.reml
    assert diagnostics is not None
    selected, canonical = diagnostics.selection, diagnostics.canonical_fit
    fit = selected.fit
    missing = np.full(ids.shape, np.nan)
    return np.stack(
        [
            ids,
            fit.criterion,
            missing if canonical is None else canonical.criterion,
            diagnostics.delta_criterion,
            selected.gap,
            fit.ratio,
            fit.noise_variance,
            fit.effect_variance,
            fit.rho,
            fit.at_upper,
            fit.unrefined,
            selected.any_at_upper,
            selected.any_unrefined,
            _status_values(fit),
            missing if canonical is None else _status_values(canonical),
            missing if canonical is None else canonical.at_upper,
            missing if canonical is None else canonical.unrefined,
        ]
    )


def all_selection_maps(selections, n_features):
    method = selection_method(selections)
    maps = np.full((len(selection_names(method)), n_features), np.nan)
    for indices, bundle in selections.items():
        maps[:, list(indices)] = selection_values(bundle["all"])
    return {"all": maps}


def reml_record(result):
    """Numeric likelihoods live in maps; these records explain their codes and IDs."""
    diagnostics = result.reml
    canonical = diagnostics.canonical_fit
    return dict(
        candidate_ids=diagnostics.candidate_ids.tolist(),
        reference_hrf_id=int(diagnostics.reference_hrf_id),
        status_codes=STATUS_CODES,
        status=list(diagnostics.selection.fit.status),
        canonical_status=None if canonical is None else list(canonical.status),
        qualification="any_unrefined marks provisional comparisons, including losing candidates",
        score="negative_twice_restricted_log_likelihood; lower is better",
    )
