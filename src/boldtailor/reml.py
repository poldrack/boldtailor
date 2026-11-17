"""Scalar restricted maximum likelihood for one random-effects variance.

For each feature (for example, a voxel), group r follows the model

    y_r = C_r beta_r + Z_r b_r + epsilon_r
    b_r ~ N(0, tau_squared I)
    epsilon_r ~ N(0, sigma_squared R(rho))

The random effects are the coefficients b_r of the supplied columns of Z_r.
Their interpretation comes from those columns. With one HRF-convolved column
per trial, they are trial amplitudes, modeled as independent zero-mean draws
with a common variance. Each group has its own coefficients; tau_squared and
sigma_squared are shared across groups for that feature. Features are fitted
separately. A RemlGroup is an observation block, not an automatic random
intercept for a run or subject.

C_r contains unpenalised fixed effects with unrestricted coefficients beta_r.
To model trial amplitudes as deviations around a nonzero mean response, put
that mean response in C_r; this module does not add it or center Z_r for you.
The supplied rho fixes the AR(1) noise correlation; these functions do not
estimate rho.

REML integrates over b_r and accounts for the fixed effects. The marginal
signal covariance is tau_squared Z_r Z_r.T + sigma_squared R(rho).
The fitted variance ratio is lambda = tau_squared / sigma_squared.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from boldtailor._arrays import (
    copy_array_fields_in_place,
    freeze_sequence_fields_in_place,
)
from boldtailor._reml_inputs import checked_random, feature_rho, validate_groups
from boldtailor._reml_criterion import (
    criterion,
    pool_terms,
    pooled_quadratic,
    profiled_variance,
)
from boldtailor._reml_search import RatioOptimum, minimize_ratio
from boldtailor._reml_terms import (
    FixedProjection,
    RemlTerms,
    project_fixed,
    reml_terms,
    reproject_signals,
)

__all__ = [
    "FixedProjection",
    "RatioOptimum",
    "RemlFit",
    "RemlGroup",
    "RemlSelection",
    "RemlTerms",
    "criterion",
    "fit_reml",
    "minimize_ratio",
    "profiled_variance",
    "project_fixed",
    "reml_terms",
    "select_random_effects_design",
]


@dataclass(frozen=True)
class RemlGroup:
    """Signals ``(scans, features)`` and their unpenalised fixed design ``C``."""

    signals: np.ndarray
    fixed_design: np.ndarray

    def __post_init__(self):
        copy_array_fields_in_place(self, ("signals", "fixed_design"))


@dataclass(frozen=True)
class RemlFit:
    """Per-feature REML variance estimates; ``noise_variance`` is marginal.

    ``unrefined`` marks a positive ratio taken from the search grid without a
    verified stationary point; ``at_upper`` marks those at the grid's upper edge.

    ``no_residual_signal`` means residual energy is within projection rounding.
    ``numerically_unresolved_residual`` means it exceeds rounding but remains
    within fixed-span uncertainty. Both skip fitting and have NaN variance
    estimates, ratio and criterion. ``variance_zero`` instead denotes a fitted
    zero random-effects variance with resolved residual signal.
    """

    ratio: np.ndarray
    noise_variance: np.ndarray
    effect_variance: np.ndarray
    criterion: np.ndarray
    status: tuple[str, ...]
    at_upper: np.ndarray
    unrefined: np.ndarray
    rho: np.ndarray
    restricted_dof: int

    def __post_init__(self):
        copy_array_fields_in_place(self, ("ratio", "noise_variance", "effect_variance"))
        copy_array_fields_in_place(self, ("criterion", "rho"))
        copy_array_fields_in_place(self, ("at_upper", "unrefined"), dtype=bool)
        freeze_sequence_fields_in_place(self, ("status",))


# ---------------------------------------------------------------- fitting


@dataclass(frozen=True)
class _RhoFeatureSubset:
    """Features with resolved residual signal sharing one noise correlation.

    ``fit_feature_indices`` identifies features retained for fitting.
    ``unresolved_feature_indices`` identifies residuals above projection
    rounding but within the uncertainty of the computed fixed span.
    """

    fit_feature_indices: np.ndarray
    projections: list
    dof: int
    unresolved_feature_indices: np.ndarray


def _project_out_fixed_effects_by_group(groups, feature_indices, rho):
    return [
        project_fixed(g.signals[:, feature_indices], g.fixed_design, rho)
        for g in groups
    ]


def _pooled_energy_exceeds_floor(
    projections: Sequence[FixedProjection], floor: str
) -> NDArray[np.bool_]:
    """Features whose pooled projected energy exceeds the pooled named floor."""
    energy = sum(p.energy for p in projections)
    return np.greater(energy, sum(getattr(p, floor) for p in projections))


def _prepare_estimable_features_for_rho(groups, feature_indices, rho):
    """Prepare resolved residuals for one rho; this does not establish identifiability."""
    projections = _project_out_fixed_effects_by_group(groups, feature_indices, rho)
    has_resolved_residual_signal = _pooled_energy_exceeds_floor(
        projections, "uncertainty_floor"
    )
    unresolved_feature_indices = feature_indices[
        np.logical_not(has_resolved_residual_signal)
        & _pooled_energy_exceeds_floor(projections, "rounding_floor")
    ]
    if has_resolved_residual_signal.any() and not has_resolved_residual_signal.all():
        projections = [
            reproject_signals(
                p, g.signals[:, feature_indices[has_resolved_residual_signal]]
            )
            for p, g in zip(projections, groups, strict=True)
        ]
    dof = sum(p.dof for p in projections)
    return _RhoFeatureSubset(
        feature_indices[has_resolved_residual_signal],
        projections,
        dof,
        unresolved_feature_indices,
    )


def _prepare_feature_subsets_by_rho(groups, rho):
    """Batch features by identical rho to reuse whitening and fixed projections.

    This is a computational grouping: every feature has its own variance
    estimates. The same prepared subsets are reused across candidate designs.
    """
    return [
        _prepare_estimable_features_for_rho(
            groups, np.flatnonzero(rho == value), float(value)
        )
        for value in np.unique(rho)
    ]


def _classify_variance_ratio_optimum(optimum):
    status = np.where(optimum.at_zero, "variance_zero", "ok")
    return np.where(optimum.unidentifiable, "unidentifiable", status)


def _fit_subset_variance_components(subset, random_design):
    """Fit variance components in the order of subset.fit_feature_indices."""
    terms = [
        reml_terms(p, z) for p, z in zip(subset.projections, random_design, strict=True)
    ]
    pooled = pool_terms(terms)
    optimum = minimize_ratio(pooled)
    return {
        "ratio": optimum.ratio,
        "noise_variance": pooled_quadratic(optimum.ratio, pooled) / pooled.dof,
        "criterion": optimum.criterion,
        "status": _classify_variance_ratio_optimum(optimum),
        "at_upper": optimum.at_upper,
        "unrefined": optimum.unrefined,
    }


def _allocate_fit_buffers(n_features):
    """Allocate NaN estimates; fitting overwrites features with resolved residuals."""
    return {
        "ratio": np.full(n_features, np.nan),
        "noise_variance": np.full(n_features, np.nan),
        "criterion": np.full(n_features, np.nan),
        "status": np.full(n_features, "no_residual_signal", dtype=object),
        "at_upper": np.zeros(n_features, dtype=bool),
        "unrefined": np.zeros(n_features, dtype=bool),
    }


def _copy_fit_values_into(destination, feature_indices, values, keep=slice(None)):
    """Copy values into destination feature positions; leave sources intact."""
    destination_feature_indices = feature_indices[keep]
    for name, array in values.items():
        destination[name][destination_feature_indices] = array[keep]


def _build_reml_fit(fit_buffers, rho, restricted_dof) -> RemlFit:
    ratio, noise = fit_buffers["ratio"], fit_buffers["noise_variance"]
    return RemlFit(
        ratio=ratio,
        noise_variance=noise,
        effect_variance=ratio * noise,
        criterion=fit_buffers["criterion"],
        status=tuple(str(s) for s in fit_buffers["status"]),
        at_upper=fit_buffers["at_upper"],
        unrefined=fit_buffers["unrefined"],
        rho=rho,
        restricted_dof=restricted_dof,
    )


def _require_common_restricted_dof(subsets) -> int:
    """The restricted dof shared by every rho subset; rank must not flip with rho."""
    dofs = {subset.dof for subset in subsets}
    if len(dofs) != 1:
        raise ValueError(f"restricted degrees of freedom differ across rho: {dofs}")
    return dofs.pop()


def _mark_unresolved_in_place(fit_buffers, subsets):
    """Mark features whose residual signal is numerically unresolved."""
    for subset in subsets:
        fit_buffers["status"][
            subset.unresolved_feature_indices
        ] = "numerically_unresolved_residual"
        fit_buffers["ratio"][subset.unresolved_feature_indices] = np.nan


def _prepare_reml_fit(groups, rho):
    """Return feature correlations, prepared subsets, shared dof and fit buffers."""
    rho = feature_rho(rho, groups[0].signals.shape[1])
    subsets = _prepare_feature_subsets_by_rho(groups, rho)
    fit_buffers = _allocate_fit_buffers(rho.size)
    _mark_unresolved_in_place(fit_buffers, subsets)
    return rho, subsets, _require_common_restricted_dof(subsets), fit_buffers


def _fit_subsets_into_buffers(fit_buffers, subsets, random_design):
    """Fit each subset and write estimates into the supplied buffers in place."""
    for subset in subsets:
        if subset.fit_feature_indices.size:
            _copy_fit_values_into(
                fit_buffers,
                subset.fit_feature_indices,
                _fit_subset_variance_components(subset, random_design),
            )


def fit_reml(
    groups: Sequence[RemlGroup],
    random_design: Iterable[np.ndarray | None],
    *,
    rho: float | np.ndarray = 0.0,
) -> RemlFit:
    """Estimate the variance ratio of the random effects, feature by feature."""
    validate_groups(groups)
    random_design = checked_random(groups, random_design, "random_design")
    rho, subsets, dof, fit_buffers = _prepare_reml_fit(groups, rho)
    _fit_subsets_into_buffers(fit_buffers, subsets, random_design)
    return _build_reml_fit(fit_buffers, rho, dof)


# ---------------------------------------------------------------- selection


@dataclass(frozen=True)
class RemlSelection:
    """Per-feature best random design among candidates, by restricted likelihood.

    ``any_at_upper`` marks features for which some candidate's criterion was
    truncated at the upper edge of its search range. That criterion is an
    upper bound on the candidate's optimum, so the comparison is provisional.
    ``any_unrefined`` also includes interior grid fallbacks of any candidate.
    """

    winner: np.ndarray
    fit: RemlFit
    gap: np.ndarray
    n_candidates: int
    any_at_upper: np.ndarray
    any_unrefined: np.ndarray

    def __post_init__(self):
        copy_array_fields_in_place(self, ("winner",), dtype=np.int64)
        copy_array_fields_in_place(self, ("gap",))
        copy_array_fields_in_place(self, ("any_at_upper", "any_unrefined"), dtype=bool)


def _initialize_candidate_selection(n_features):
    return {
        "winner": np.full(n_features, -1, dtype=np.int64),
        "best_criterion": np.full(n_features, np.inf),
        "runner_up_criterion": np.full(n_features, np.inf),
        "any_at_upper": np.zeros(n_features, dtype=bool),
        "any_unrefined": np.zeros(n_features, dtype=bool),
    }


def _update_selection_from_fit(
    selection_state, fit_buffers, feature_indices, values, candidate_index
):
    """Update candidate rankings and winning-fit buffers using eligible fits."""
    ok = values["status"] == "ok"
    value = values["criterion"]
    best_criterion, runner_up_criterion = (
        selection_state["best_criterion"][feature_indices],
        selection_state["runner_up_criterion"][feature_indices],
    )
    better = ok & (value < best_criterion)
    runner_up = np.where(better, best_criterion, np.minimum(runner_up_criterion, value))
    selection_state["runner_up_criterion"][feature_indices] = np.where(
        ok, runner_up, runner_up_criterion
    )
    selection_state["best_criterion"][feature_indices] = np.where(
        better, value, best_criterion
    )
    selection_state["winner"][feature_indices[better]] = candidate_index
    selection_state["any_at_upper"][feature_indices] |= values["at_upper"]
    selection_state["any_unrefined"][feature_indices] |= values["unrefined"]
    _copy_fit_values_into(fit_buffers, feature_indices, values, better)


def _fit_candidate_and_update_selection(
    selection_state, fit_buffers, subsets, random_design, candidate_index
):
    """Fit one candidate and update selection state and fit buffers in place."""
    for subset in subsets:
        if subset.fit_feature_indices.size:
            values = _fit_subset_variance_components(subset, random_design)
            _update_selection_from_fit(
                selection_state,
                fit_buffers,
                subset.fit_feature_indices,
                values,
                candidate_index,
            )


def _compute_runner_up_criterion_gap(selection_state):
    supported = (selection_state["winner"] >= 0) & np.isfinite(
        selection_state["runner_up_criterion"]
    )
    gap = np.full(supported.shape, np.nan)
    gap[supported] = (
        selection_state["runner_up_criterion"][supported]
        - selection_state["best_criterion"][supported]
    )
    return gap


def _evaluate_candidates_and_update_fit_buffers(
    groups, candidates, subsets, fit_buffers
):
    """Update winning-fit buffers; return selection state and candidate count."""
    selection_state, n_candidates = (
        _initialize_candidate_selection(fit_buffers["ratio"].size),
        0,
    )
    for index, candidate in enumerate(candidates):
        label = f"candidate {index} random_design"
        random_design = checked_random(groups, candidate, label)
        _fit_candidate_and_update_selection(
            selection_state, fit_buffers, subsets, random_design, index
        )
        n_candidates = index + 1
    if n_candidates == 0:
        raise ValueError("at least one candidate is required")
    return selection_state, n_candidates


def select_random_effects_design(
    groups: Sequence[RemlGroup],
    candidates: Iterable[Iterable[np.ndarray | None]],
    *,
    rho: float | np.ndarray = 0.0,
) -> RemlSelection:
    """Select random-effects designs by REML, independently for each feature.

    Each candidate supplies one Z matrix per group, in group order; None means
    no random-effect columns for that group. The signals, fixed designs C,
    and supplied rho are held constant across candidates. For example, a
    candidate can contain trial-response designs made with one HRF, while
    another candidate uses a different HRF for the same trials.

    For every candidate, estimate the shared-across-groups variance ratio and
    noise variance, then compare its optimized restricted-likelihood criterion
    (lower is better). This selects the candidate covariance structure, not
    random columns or random observations. It uses the supplied data directly;
    it is not cross-validation and does not return fitted coefficients b_r.

    winner contains zero-based candidate indices. Only fits with status "ok"
    compete; zero-variance or unidentifiable fits do not become winners. A
    winner of -1 means no supported candidate; inspect fit.status to distinguish
    zero random-effects variance, absent residual signal, and numerically
    unresolved residual signal. Exact ties favor the earlier candidate. fit
    holds the winning estimates, or the no-random-effects baseline when no
    candidate wins. Features skipped for absent or unresolved residuals retain
    NaN estimates. gap is the runner-up criterion minus the best criterion,
    or NaN when fewer than two candidates qualify.
    any_at_upper and any_unrefined flag limitations in any candidate's search.

    Candidates are consumed once, with only the best and runner-up retained.
    """
    validate_groups(groups)
    rho, subsets, dof, fit_buffers = _prepare_reml_fit(groups, rho)
    _fit_subsets_into_buffers(fit_buffers, subsets, [None] * len(groups))
    selection_state, count = _evaluate_candidates_and_update_fit_buffers(
        groups, candidates, subsets, fit_buffers
    )
    return RemlSelection(
        winner=selection_state["winner"],
        fit=_build_reml_fit(fit_buffers, rho, dof),
        gap=_compute_runner_up_criterion_gap(selection_state),
        n_candidates=count,
        any_at_upper=selection_state["any_at_upper"],
        any_unrefined=selection_state["any_unrefined"],
    )
