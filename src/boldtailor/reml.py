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

from boldtailor._arrays import own_fields, own_tuples
from boldtailor._reml_inputs import checked_random, feature_rho, validate_groups
from boldtailor._reml_criterion import pool_terms, pooled_quadratic
from boldtailor._reml_search import minimize_ratio
from boldtailor._reml_terms import project_fixed, reml_terms, reproject_signals


@dataclass(frozen=True)
class RemlGroup:
    """Signals ``(scans, features)`` and their unpenalised fixed design ``C``."""

    signals: np.ndarray
    fixed_design: np.ndarray

    def __post_init__(self):
        own_fields(self, ("signals", "fixed_design"))


@dataclass(frozen=True)
class RemlFit:
    """Per-feature REML variance estimates; ``noise_variance`` is marginal.

    ``unrefined`` marks a positive ratio taken from the search grid without a
    verified stationary point; ``at_upper`` marks those at the grid's upper edge.
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
        own_fields(self, ("ratio", "noise_variance", "effect_variance"))
        own_fields(self, ("criterion", "rho"))
        own_fields(self, ("at_upper", "unrefined"), dtype=bool)
        own_tuples(self, ("status",))


# ---------------------------------------------------------------- fitting


@dataclass(frozen=True)
class _Subset:
    """Live columns of one rho value and their fixed projections per group.

    ``unresolved`` holds the columns whose energy is above projection rounding
    but within the uncertainty of the computed fixed span.
    """

    columns: np.ndarray
    projections: list
    dof: int
    unresolved: np.ndarray


def _project(groups, columns, rho):
    return [project_fixed(g.signals[:, columns], g.fixed_design, rho) for g in groups]


def _above(projections, floor):
    """Features whose pooled projected energy exceeds the pooled named floor."""
    energy = sum(p.energy for p in projections)
    return energy > sum(getattr(p, floor) for p in projections)


def _live_subset(groups, columns, rho):
    """Project one rho subset; when it mixes in dead features, re-project the live."""
    projections = _project(groups, columns, rho)
    live = _above(projections, "uncertainty_floor")
    unresolved = columns[~live & _above(projections, "rounding_floor")]
    if live.any() and not live.all():
        projections = [
            reproject_signals(p, g.signals[:, columns[live]])
            for p, g in zip(projections, groups, strict=True)
        ]
    dof = sum(p.dof for p in projections)
    return _Subset(columns[live], projections, dof, unresolved)


def _subsets(groups, rho):
    return [
        _live_subset(groups, np.flatnonzero(rho == value), float(value))
        for value in np.unique(rho)
    ]


def _status(optimum):
    status = np.where(optimum.at_zero, "variance_zero", "ok")
    return np.where(optimum.unidentifiable, "unidentifiable", status)


def _fit_values(subset, random_design):
    """Fit one candidate on one subset; arrays are indexed like ``subset.columns``."""
    terms = [
        reml_terms(p, z) for p, z in zip(subset.projections, random_design, strict=True)
    ]
    pooled = pool_terms(terms)
    optimum = minimize_ratio(pooled)
    return {
        "ratio": optimum.ratio,
        "noise_variance": pooled_quadratic(optimum.ratio, pooled) / pooled.dof,
        "criterion": optimum.criterion,
        "status": _status(optimum),
        "at_upper": optimum.at_upper,
        "unrefined": optimum.unrefined,
    }


def _empty_result(n_features):
    """Every feature starts as no_signal; fits overwrite the live ones."""
    return {
        "ratio": np.zeros(n_features),
        "noise_variance": np.full(n_features, np.nan),
        "criterion": np.full(n_features, np.nan),
        "status": np.full(n_features, "no_signal", dtype=object),
        "at_upper": np.zeros(n_features, dtype=bool),
        "unrefined": np.zeros(n_features, dtype=bool),
    }


def _write(result, columns, values, keep=slice(None)):
    for name, array in values.items():
        result[name][columns[keep]] = array[keep]


def _assemble(result, rho, restricted_dof) -> RemlFit:
    ratio, noise = result["ratio"], result["noise_variance"]
    return RemlFit(
        ratio=ratio,
        noise_variance=noise,
        effect_variance=ratio * noise,
        criterion=result["criterion"],
        status=tuple(str(s) for s in result["status"]),
        at_upper=result["at_upper"],
        unrefined=result["unrefined"],
        rho=rho,
        restricted_dof=restricted_dof,
    )


def _single_dof(subsets) -> int:
    """The restricted dof shared by every rho subset; rank must not flip with rho."""
    dofs = {subset.dof for subset in subsets}
    if len(dofs) != 1:
        raise ValueError(f"restricted degrees of freedom differ across rho: {dofs}")
    return dofs.pop()


def _mark_unresolved(result, subsets):
    """Unresolved features claim neither a zero ratio nor an absent signal."""
    for subset in subsets:
        result["status"][subset.unresolved] = "unresolved"
        result["ratio"][subset.unresolved] = np.nan


def _start(groups, rho):
    """Per-feature rho, its subsets, their dof and the result to fill in."""
    rho = feature_rho(rho, groups[0].signals.shape[1])
    subsets = _subsets(groups, rho)
    result = _empty_result(rho.size)
    _mark_unresolved(result, subsets)
    return rho, subsets, _single_dof(subsets), result


def _fill(result, subsets, random_design):
    for subset in subsets:
        if subset.columns.size:
            _write(result, subset.columns, _fit_values(subset, random_design))


def fit_reml(
    groups: Sequence[RemlGroup],
    random_design: Iterable[np.ndarray | None],
    *,
    rho: float | np.ndarray = 0.0,
) -> RemlFit:
    """Estimate the variance ratio of the random effects, feature by feature."""
    validate_groups(groups)
    random_design = checked_random(groups, random_design, "random_design")
    rho, subsets, dof, result = _start(groups, rho)
    _fill(result, subsets, random_design)
    return _assemble(result, rho, dof)


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
        own_fields(self, ("winner",), dtype=np.int64)
        own_fields(self, ("gap",))
        own_fields(self, ("any_at_upper", "any_unrefined"), dtype=bool)


def _empty_race(n_features):
    return {
        "winner": np.full(n_features, -1, dtype=np.int64),
        "best": np.full(n_features, np.inf),
        "second": np.full(n_features, np.inf),
        "any_at_upper": np.zeros(n_features, dtype=bool),
        "any_unrefined": np.zeros(n_features, dtype=bool),
    }


def _compete(race, result, columns, values, candidate):
    """Update the running best and runner-up with one candidate's ok features."""
    ok = values["status"] == "ok"
    value = values["criterion"]
    best, second = race["best"][columns], race["second"][columns]
    better = ok & (value < best)
    runner_up = np.where(better, best, np.minimum(second, value))
    race["second"][columns] = np.where(ok, runner_up, second)
    race["best"][columns] = np.where(better, value, best)
    race["winner"][columns[better]] = candidate
    race["any_at_upper"][columns] |= values["at_upper"]
    race["any_unrefined"][columns] |= values["unrefined"]
    _write(result, columns, values, better)


def _run_candidate(race, result, subsets, random_design, candidate):
    for subset in subsets:
        if subset.columns.size:
            values = _fit_values(subset, random_design)
            _compete(race, result, subset.columns, values, candidate)


def _gap(race):
    supported = (race["winner"] >= 0) & np.isfinite(race["second"])
    gap = np.full(supported.shape, np.nan)
    gap[supported] = race["second"][supported] - race["best"][supported]
    return gap


def _run_race(groups, candidates, subsets, result):
    """Fit candidates one at a time as they are drawn; return the race and count."""
    race, n_candidates = _empty_race(result["ratio"].size), 0
    for index, candidate in enumerate(candidates):
        label = f"candidate {index} random_design"
        random_design = checked_random(groups, candidate, label)
        _run_candidate(race, result, subsets, random_design, index)
        n_candidates = index + 1
    if n_candidates == 0:
        raise ValueError("at least one candidate is required")
    return race, n_candidates


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
    zero variance, absent signal, and numerically unresolved signal. Exact ties
    favor the earlier candidate. fit holds the winning variance estimates, or
    the no-random-effects baseline when no candidate wins. gap is the runner-up
    criterion minus the best, or NaN when fewer than two candidates qualify.
    any_at_upper and any_unrefined flag limitations in any candidate's search.

    Candidates are consumed once, with only the best and runner-up retained.
    """
    validate_groups(groups)
    rho, subsets, dof, result = _start(groups, rho)
    _fill(result, subsets, [None] * len(groups))
    race, count = _run_race(groups, candidates, subsets, result)
    return RemlSelection(
        winner=race["winner"],
        fit=_assemble(result, rho, dof),
        gap=_gap(race),
        n_candidates=count,
        any_at_upper=race["any_at_upper"],
        any_unrefined=race["any_unrefined"],
    )
