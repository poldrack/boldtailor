"""Scalar restricted maximum likelihood for one random-effect variance."""

from collections.abc import Sequence
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


def _check_group(group, random, index):
    _check_matrix(group.signals, "signals", index)
    _check_matrix(group.fixed_design, "fixed_design", index)
    scans = {group.signals.shape[0], group.fixed_design.shape[0]}
    if random is not None:
        _check_matrix(random, "random_design", index)
        scans.add(np.shape(random)[0])
    if len(scans) != 1:
        raise ValueError(f"scan counts differ within group {index}")


def _validate(groups, random_design):
    if len(groups) == 0:
        raise ValueError("at least one group is required")
    if len(random_design) != len(groups):
        raise ValueError("random_design must have one entry per group")
    for index, (group, random) in enumerate(zip(groups, random_design, strict=True)):
        _check_group(group, random, index)
    if len({g.signals.shape[1] for g in groups}) != 1:
        raise ValueError("feature counts differ across groups")
    if groups[0].signals.shape[1] == 0:
        raise ValueError("signals must have at least one feature")


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


def _group_terms(groups, random_design, columns, rho):
    return [
        reml_terms(project_fixed(g.signals[:, columns], g.fixed_design, rho), z)
        for g, z in zip(groups, random_design, strict=True)
    ]


def _live(terms):
    energy = sum(t.energy for t in terms)
    raw_energy = sum(t.raw_energy for t in terms)
    return energy > _NO_SIGNAL_FACTOR * np.finfo(float).eps * raw_energy


def _empty_result(n_features):
    return {
        "ratio": np.zeros(n_features),
        "noise_variance": np.full(n_features, np.nan),
        "criterion": np.full(n_features, np.nan),
        "status": np.full(n_features, "no_signal", dtype=object),
        "at_upper": np.zeros(n_features, dtype=bool),
    }


def _store(result, index, terms):
    optimum = minimize_ratio(terms)
    result["ratio"][index] = optimum.ratio
    result["noise_variance"][index] = profiled_variance(optimum.ratio, terms)
    result["criterion"][index] = optimum.criterion
    result["status"][index] = np.where(optimum.at_zero, "variance_zero", "ok")
    result["at_upper"][index] = optimum.at_upper


def _fit_subset(result, groups, random_design, columns, rho):
    """Fit one rho subset; dead features are dropped before the live refit."""
    terms = _group_terms(groups, random_design, columns, rho)
    live = _live(terms)
    if live.any() and not live.all():
        terms = _group_terms(groups, random_design, columns[live], rho)
    if live.any():
        _store(result, columns[live], terms)
    return sum(t.dof for t in terms)


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


def fit_reml(
    groups: Sequence[RemlGroup],
    random_design: Sequence[np.ndarray | None],
    *,
    rho: float | np.ndarray = 0.0,
) -> RemlFit:
    """Estimate the variance ratio of the random effects, feature by feature."""
    _validate(groups, random_design)
    random_design = [_as_random(z) for z in random_design]
    n_features = groups[0].signals.shape[1]
    rho = _feature_rho(rho, n_features)
    result = _empty_result(n_features)
    for value in np.unique(rho):
        columns = np.flatnonzero(rho == value)
        dof = _fit_subset(result, groups, random_design, columns, float(value))
    return _assemble(result, rho, dof)
