"""Scalar restricted maximum likelihood for one random-effect variance."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import own_fields, own_tuples
from boldtailor._reml_criterion import profiled_variance
from boldtailor._reml_search import minimize_ratio
from boldtailor._reml_terms import project_fixed, reml_terms

_NO_SIGNAL_FACTOR = 64.0


@dataclass(frozen=True)
class RemlGroup:
    """Signals ``(scans, features)`` and their unpenalised fixed design ``C``."""

    signals: np.ndarray
    fixed_design: np.ndarray

    def __post_init__(self):
        own_fields(self, ("signals", "fixed_design"))


@dataclass(frozen=True)
class RemlFit:
    """Per-feature REML variance estimates; ``noise_variance`` is marginal."""

    ratio: np.ndarray
    noise_variance: np.ndarray
    effect_variance: np.ndarray
    criterion: np.ndarray
    status: tuple[str, ...]
    at_upper: np.ndarray
    rho: np.ndarray
    restricted_dof: int

    def __post_init__(self):
        own_fields(self, ("ratio", "noise_variance", "effect_variance"))
        own_fields(self, ("criterion", "rho"))
        own_fields(self, ("at_upper",), dtype=bool)
        own_tuples(self, ("status",))


# ---------------------------------------------------------------- validation


def _check_finite(values, name, index):
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} of group {index} contains non-finite values")


def _check_matrix(values, name, index):
    if np.ndim(values) != 2:
        raise ValueError(f"{name} of group {index} must be 2-D (scans, columns)")
    _check_finite(values, name, index)


def _check_group(group, index):
    _check_matrix(group.signals, "signals", index)
    _check_matrix(group.fixed_design, "fixed_design", index)
    if group.signals.shape[0] != group.fixed_design.shape[0]:
        raise ValueError(f"scan counts differ within group {index}")


def _check_random(group, random, index):
    if random is None:
        return
    _check_matrix(random, "random_design", index)
    if np.shape(random)[0] != group.signals.shape[0]:
        raise ValueError(f"scan counts differ within group {index}")


def _validate_groups(groups):
    if len(groups) == 0:
        raise ValueError("at least one group is required")
    for index, group in enumerate(groups):
        _check_group(group, index)
    if len({g.signals.shape[1] for g in groups}) != 1:
        raise ValueError("feature counts differ across groups")
    if groups[0].signals.shape[1] == 0:
        raise ValueError("signals must have at least one feature")


def _checked_random(groups, random_design, label):
    """Validate one random design per group and return it as float arrays."""
    if len(random_design) != len(groups):
        raise ValueError(f"{label} must have one entry per group")
    for index, (group, random) in enumerate(zip(groups, random_design, strict=True)):
        _check_random(group, random, index)
    return [_as_random(z) for z in random_design]


def _feature_rho(rho, n_features):
    values = np.asarray(rho, dtype=float)
    if values.ndim == 0:
        values = np.full(n_features, float(values))
    if values.shape != (n_features,):
        raise ValueError("rho must be a scalar or have shape (features,)")
    if not np.all(np.abs(values) < 1):
        raise ValueError("rho must be finite with |rho| < 1")
    return values


def _as_random(random):
    return None if random is None else np.asarray(random, dtype=float)


# ---------------------------------------------------------------- fitting


@dataclass(frozen=True)
class _Subset:
    """Live columns of one rho value and their fixed projections per group."""

    columns: np.ndarray
    projections: list
    dof: int


def _project(groups, columns, rho):
    return [project_fixed(g.signals[:, columns], g.fixed_design, rho) for g in groups]


def _live(projections):
    energy = sum(p.energy for p in projections)
    raw_energy = sum(p.raw_energy for p in projections)
    return energy > _NO_SIGNAL_FACTOR * np.finfo(float).eps * raw_energy


def _live_subset(groups, columns, rho):
    """Project one rho subset; when it mixes in dead features, re-project the live."""
    projections = _project(groups, columns, rho)
    live = _live(projections)
    if live.any() and not live.all():
        projections = _project(groups, columns[live], rho)
    return _Subset(columns[live], projections, sum(p.dof for p in projections))


def _subsets(groups, rho):
    return [
        _live_subset(groups, np.flatnonzero(rho == value), float(value))
        for value in np.unique(rho)
    ]


def _fit_values(subset, random_design):
    """Fit one candidate on one subset; arrays are indexed like ``subset.columns``."""
    terms = [
        reml_terms(p, z) for p, z in zip(subset.projections, random_design, strict=True)
    ]
    optimum = minimize_ratio(terms)
    return {
        "ratio": optimum.ratio,
        "noise_variance": profiled_variance(optimum.ratio, terms),
        "criterion": optimum.criterion,
        "status": np.where(optimum.at_zero, "variance_zero", "ok"),
        "at_upper": optimum.at_upper,
    }


def _empty_result(n_features):
    return {
        "ratio": np.zeros(n_features),
        "noise_variance": np.full(n_features, np.nan),
        "criterion": np.full(n_features, np.nan),
        "status": np.full(n_features, "no_signal", dtype=object),
        "at_upper": np.zeros(n_features, dtype=bool),
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
        rho=rho,
        restricted_dof=restricted_dof,
    )


def _prepare(groups, rho):
    rho = _feature_rho(rho, groups[0].signals.shape[1])
    return rho, _subsets(groups, rho)


def _fill(result, subsets, random_design):
    for subset in subsets:
        if subset.columns.size:
            _write(result, subset.columns, _fit_values(subset, random_design))


def fit_reml(
    groups: Sequence[RemlGroup],
    random_design: Sequence[np.ndarray | None],
    *,
    rho: float | np.ndarray = 0.0,
) -> RemlFit:
    """Estimate the variance ratio of the random effects, feature by feature."""
    _validate_groups(groups)
    random_design = _checked_random(groups, random_design, "random_design")
    rho, subsets = _prepare(groups, rho)
    result = _empty_result(rho.size)
    _fill(result, subsets, random_design)
    return _assemble(result, rho, subsets[-1].dof)


# ---------------------------------------------------------------- selection


@dataclass(frozen=True)
class RemlSelection:
    """Per-feature best random design among candidates, by restricted likelihood."""

    winner: np.ndarray
    fit: RemlFit
    gap: np.ndarray
    n_candidates: int

    def __post_init__(self):
        own_fields(self, ("winner",), dtype=np.int64)
        own_fields(self, ("gap",))


def _empty_race(n_features):
    return {
        "winner": np.full(n_features, -1, dtype=np.int64),
        "best": np.full(n_features, np.inf),
        "second": np.full(n_features, np.inf),
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
    race, count = _empty_race(result["ratio"].size), 0
    for count, candidate in enumerate(candidates, start=1):
        label = f"candidate {count - 1}"
        random_design = _checked_random(groups, candidate, label)
        _run_candidate(race, result, subsets, random_design, count - 1)
    if count == 0:
        raise ValueError("at least one candidate is required")
    return race, count


def select_random_design(
    groups: Sequence[RemlGroup],
    candidates: Iterable[Sequence[np.ndarray | None]],
    *,
    rho: float | np.ndarray = 0.0,
) -> RemlSelection:
    """Pick, per feature, the candidate random design with the best criterion."""
    _validate_groups(groups)
    rho, subsets = _prepare(groups, rho)
    result = _empty_result(rho.size)
    _fill(result, subsets, [None] * len(groups))
    race, count = _run_race(groups, candidates, subsets, result)
    return RemlSelection(
        winner=race["winner"],
        fit=_assemble(result, rho, subsets[-1].dof),
        gap=_gap(race),
        n_candidates=count,
    )
