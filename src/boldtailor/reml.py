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
from boldtailor._numerics import (
    conditioned_basis,
    project_out,
    rank_tolerance,
    residualise,
)
from boldtailor.ar1 import whiten

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


# ---------------------------------------------------------------- inputs


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


def _check_random(group, random, label, index):
    if random is None:
        return
    _check_matrix(random, label, index)
    if np.shape(random)[0] != group.signals.shape[0]:
        raise ValueError(
            f"scan counts differ between signals and {label} of group {index}"
        )


def _as_random(random):
    return None if random is None else np.asarray(random, dtype=float)


def validate_groups(groups):
    """Raise ValueError unless the groups are finite, 2-D and share features."""
    if len(groups) == 0:
        raise ValueError("at least one group is required")
    for index, group in enumerate(groups):
        _check_group(group, index)
    if len({g.signals.shape[1] for g in groups}) != 1:
        raise ValueError("feature counts differ across groups")
    if groups[0].signals.shape[1] == 0:
        raise ValueError("signals must have at least one feature")


def checked_random(groups, random_design, label):
    """One validated float random design (or None) per group from any iterable."""
    designs = [_as_random(z) for z in random_design]
    if len(designs) != len(groups):
        raise ValueError(f"{label} must have one entry per group")
    for index, (group, random) in enumerate(zip(groups, designs, strict=True)):
        _check_random(group, random, label, index)
    return designs


def feature_rho(rho, n_features):
    """Broadcast rho to one stationary value per feature."""
    values = np.asarray(rho, dtype=float)
    if values.ndim == 0:
        values = np.full(n_features, float(values))
    if values.shape != (n_features,):
        raise ValueError("rho must be a scalar or have shape (features,)")
    if not np.all(np.abs(values) < 1):
        raise ValueError("rho must be finite with |rho| < 1")
    return values


# ---------------------------------------------------------------- terms


@dataclass(frozen=True, eq=False)
class FixedProjection:
    """Whitened signals with the whitened fixed design projected out.

    ``energy`` at or below ``rounding_floor`` is projection rounding; at or
    below ``uncertainty_floor`` it cannot be told apart from the error of the
    computed fixed span.
    """

    basis: np.ndarray
    signals: np.ndarray
    energy: np.ndarray
    rounding_floor: np.ndarray
    uncertainty_floor: np.ndarray
    dof: int
    rho: float

    def __post_init__(self):
        copy_array_fields_in_place(self, ("basis", "signals", "energy"))
        copy_array_fields_in_place(self, ("rounding_floor", "uncertainty_floor"))


def project_fixed(signals, fixed_design, rho: float = 0.0) -> FixedProjection:
    """Whiten by AR(1) and remove the span of the fixed design."""
    y = whiten(signals, rho)
    basis, amplification = conditioned_basis(whiten(fixed_design, rho))
    dof = y.shape[0] - basis.shape[1]
    if dof <= 0:
        raise ValueError("positive restricted degrees of freedom required")
    residual, rounding, uncertainty = project_out(y, basis, amplification)
    return FixedProjection(
        basis=basis,
        signals=residual,
        energy=np.sum(residual**2, axis=0),
        rounding_floor=rounding,
        uncertainty_floor=uncertainty,
        dof=dof,
        rho=float(rho),
    )


def reproject_signals(projection: FixedProjection, signals) -> FixedProjection:
    """Whiten and project ``signals`` onto an existing fixed-design basis."""
    residual = residualise(whiten(signals, projection.rho), projection.basis)
    zeros = np.zeros(residual.shape[1])
    return FixedProjection(
        basis=projection.basis,
        signals=residual,
        energy=np.sum(residual**2, axis=0),
        rounding_floor=zeros,
        uncertainty_floor=zeros,
        dof=projection.dof,
        rho=projection.rho,
    )


@dataclass(frozen=True, eq=False)
class RemlTerms:
    """Sufficient statistics of one group for the REML criterion."""

    singular2: np.ndarray
    coordinates2: np.ndarray
    residual_ss: np.ndarray
    dof: int
    energy: np.ndarray

    def __post_init__(self):
        copy_array_fields_in_place(
            self, ("singular2", "coordinates2", "residual_ss", "energy")
        )


def _smaller_gram(matrix):
    """Z'Z or ZZ', whichever is smaller; both share the nonzero eigenvalues."""
    rows, columns = matrix.shape
    return matrix.T @ matrix if columns <= rows else matrix @ matrix.T


def _spectral_norm(matrix):
    """Largest singular value from the smaller Gram matrix, without a second SVD."""
    peak = np.abs(matrix).max()
    if peak == 0.0:
        return 0.0
    return peak * np.sqrt(
        max(np.linalg.eigvalsh(_smaller_gram(matrix / peak))[-1], 0.0)
    )


def _projected_random(projection, random_design):
    z = whiten(random_design, projection.rho)
    return residualise(z, projection.basis), _spectral_norm(z)


def _random_directions(projection, random_design):
    n = projection.signals.shape[0]
    if random_design is None or np.shape(random_design)[1] == 0:
        return np.empty((n, 0)), np.empty(0)
    projected, norm = _projected_random(projection, np.asarray(random_design, float))
    u, s, _ = np.linalg.svd(projected, full_matrices=False)
    keep = s > rank_tolerance(projected.shape, norm)
    return u[:, keep], s[keep]


def _orthogonal_energy(projection, directions, coordinates):
    """Energy outside the retained random span, without subtracting energies.

    At full restricted rank the complement has dimension zero. Otherwise
    forming the residual preserves small orthogonal signals that disappear
    in ``||y||² - ||U'y||²``. Its rounding error is at the amplitude scale.
    """
    if directions.shape[1] == projection.dof:
        return np.zeros(projection.signals.shape[1])
    residual = projection.signals - directions @ coordinates
    return np.sum(residual**2, axis=0)


def reml_terms(
    projection: FixedProjection, random_design: np.ndarray | None
) -> RemlTerms:
    """Reduce a projected group to singular values and squared coordinates."""
    u, s = _random_directions(projection, random_design)
    coordinates = u.T @ projection.signals
    energy = projection.energy
    return RemlTerms(
        singular2=s**2,
        coordinates2=coordinates**2,
        residual_ss=_orthogonal_energy(projection, u, coordinates),
        dof=projection.dof,
        energy=energy,
    )


# ---------------------------------------------------------------- criterion


@dataclass(frozen=True, eq=False)
class PooledTerms:
    """Sufficient statistics of all groups together."""

    singular2: np.ndarray
    coordinates2: np.ndarray
    residual_ss: np.ndarray
    dof: int


def pool_terms(terms: Sequence[RemlTerms]) -> PooledTerms:
    """Concatenate singular values and coordinates; sum residuals and dof."""
    return PooledTerms(
        singular2=np.concatenate([t.singular2 for t in terms]),
        coordinates2=np.concatenate([t.coordinates2 for t in terms], axis=0),
        residual_ss=np.sum([t.residual_ss for t in terms], axis=0),
        dof=sum(t.dof for t in terms),
    )


def _scaled(ratios, singular2):
    """ratio * s2 with shape (..., k, features)."""
    return ratios[..., None, :] * singular2[:, None]


def pooled_quadratic(ratios, pooled: PooledTerms) -> np.ndarray:
    """Q(lambda), the whitened residual quadratic form, shaped like ``ratios``."""
    shrunk = 1.0 + _scaled(ratios, pooled.singular2)
    return pooled.residual_ss + (pooled.coordinates2 / shrunk).sum(axis=-2)


def pooled_criterion(ratios, pooled: PooledTerms) -> np.ndarray:
    """N log(Q/N) + sum log(1 + lambda s2) from pooled statistics."""
    scaled = _scaled(ratios, pooled.singular2)
    logdet = np.log1p(scaled).sum(axis=-2)
    scaled += 1.0
    q = pooled.residual_ss + (pooled.coordinates2 / scaled).sum(axis=-2)
    return pooled.dof * np.log(q / pooled.dof) + logdet


def pooled_criterion_at_zero(pooled: PooledTerms) -> np.ndarray:
    """The criterion at lambda = 0, N log(energy / N), without (k, features) work."""
    q = pooled.residual_ss + pooled.coordinates2.sum(axis=0)
    return pooled.dof * np.log(q / pooled.dof)


def pooled_slope(ratios, pooled: PooledTerms) -> np.ndarray:
    """Derivative of the criterion in log(lambda), from pooled statistics."""
    scaled = _scaled(ratios, pooled.singular2)
    shrunk = 1.0 + scaled
    fraction = scaled / shrunk
    weighted = pooled.coordinates2 / shrunk
    q = pooled.residual_ss + weighted.sum(axis=-2)
    q_slope = -(weighted * fraction).sum(axis=-2)
    return pooled.dof * q_slope / q + fraction.sum(axis=-2)


def pooled_value_and_slope(ratios, pooled: PooledTerms):
    """Criterion and its slope in log(lambda), sharing one pass of temporaries."""
    scaled = _scaled(ratios, pooled.singular2)
    logdet = np.log1p(scaled).sum(axis=-2)
    shrunk = 1.0 + scaled
    scaled /= shrunk
    weighted = pooled.coordinates2 / shrunk
    q = pooled.residual_ss + weighted.sum(axis=-2)
    weighted *= scaled
    slope = scaled.sum(axis=-2) - pooled.dof * weighted.sum(axis=-2) / q
    return pooled.dof * np.log(q / pooled.dof) + logdet, slope


def profiled_variance(ratios, terms: Sequence[RemlTerms]) -> np.ndarray:
    """Profiled noise variance Q(lambda) / N, shaped like ``ratios``."""
    pooled = pool_terms(terms)
    return pooled_quadratic(np.asarray(ratios, dtype=float), pooled) / pooled.dof


def criterion(ratios, terms: Sequence[RemlTerms]) -> np.ndarray:
    """Profiled REML criterion N log(Q/N) + sum log(1 + lambda s2)."""
    return pooled_criterion(np.asarray(ratios, dtype=float), pool_terms(terms))


# ---------------------------------------------------------------- search


_RELATIVE_SPAN = 1e8
_ROUNDING_ULPS = 64
# A memory knob, not a speed knob: timing select_random_effects_design at budgets from
# 0.25 MiB to unbatched (3k-100k features, 12-120 pooled directions) varied by
# only a few percent from 1 MiB upward, and was 5-25% slower below that. 8 MiB
# sits clear of the slow end while keeping each temporary small.
_WORKING_BYTES = 8 * 2**20
_FLAGS = ("at_zero", "at_upper", "unrefined", "unidentifiable")


@dataclass(frozen=True, eq=False)
class RatioOptimum:
    """Per-feature variance-ratio optimum and where it sits.

    ``unrefined`` marks a positive ratio that is a grid point, not a slope
    root; ``at_upper`` marks those at the upper edge of the grid.
    """

    ratio: np.ndarray
    criterion: np.ndarray
    at_zero: np.ndarray
    at_upper: np.ndarray
    unrefined: np.ndarray
    unidentifiable: np.ndarray

    def __post_init__(self):
        copy_array_fields_in_place(self, ("ratio", "criterion"))
        copy_array_fields_in_place(self, _FLAGS, dtype=bool)


def _search_grid(singular2, size):
    """Log-ratio grid spanning 1e-8 / max(s2) to 1e8 / min(s2).

    ``size`` points cover a single singular value; a wider spectrum adds
    points so that the spacing never exceeds that base step.
    """
    low = np.log(1.0 / _RELATIVE_SPAN / singular2.max())
    high = np.log(_RELATIVE_SPAN / singular2.min())
    step = 2.0 * np.log(_RELATIVE_SPAN) / (size - 1)
    return np.linspace(low, high, max(size, int(np.ceil((high - low) / step)) + 1))


def _chunk_rows(pooled: PooledTerms) -> int:
    """Grid rows per chunk so one (rows, k, features) temporary stays bounded."""
    row_bytes = 8 * pooled.singular2.size * pooled.residual_ss.size
    return max(1, _WORKING_BYTES // max(row_bytes, 1))


def _batch_features(pooled: PooledTerms, grid_points: int) -> int:
    """Features per batch: one (k, features) temporary and one (grid, features)
    table within the budget, and never fewer than one feature."""
    return max(1, _WORKING_BYTES // (8 * max(pooled.singular2.size, grid_points)))


def _grid_chunks(pooled: PooledTerms, log_grid):
    """Criterion and slope of each chunk of grid rows, for every feature."""
    n_features, rows = pooled.residual_ss.size, _chunk_rows(pooled)
    ratios = np.exp(log_grid)
    for start in range(0, ratios.size, rows):
        part = ratios[start : start + rows]
        yield pooled_value_and_slope(
            np.broadcast_to(part[:, None], (part.size, n_features)), pooled
        )


def _on_grid(pooled: PooledTerms, log_grid):
    """Criterion and slope on the grid, each of shape (grid, features)."""
    values, slopes = zip(*_grid_chunks(pooled, log_grid), strict=True)
    return np.concatenate(values, axis=0), np.concatenate(slopes, axis=0)


def _take(pooled: PooledTerms, features) -> PooledTerms:
    """The pooled statistics of the given feature columns."""
    return PooledTerms(
        singular2=pooled.singular2,
        coordinates2=pooled.coordinates2[:, features],
        residual_ss=pooled.residual_ss[features],
        dof=pooled.dof,
    )


def _bisect_slope(pooled, low, high, iterations):
    """Vectorised bisection for the zero of the slope in log ratio."""
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        rising = pooled_slope(np.exp(middle), pooled) > 0
        low, high = np.where(rising, low, middle), np.where(rising, middle, high)
    return 0.5 * (low + high)


def _refined(pooled, log_grid, cell, feature, iterations):
    """Slope root and its criterion for each (cell, feature) crossing."""
    crossed = _take(pooled, feature)
    root = _bisect_slope(crossed, log_grid[cell], log_grid[cell + 1], iterations)
    return root, pooled_criterion(np.exp(root), crossed)


def _stacked(parts, position):
    """Concatenate one output of the refined chunks; empty when there are none."""
    return np.concatenate([np.empty(0), *(part[position] for part in parts)])


def _crossing_roots(pooled, log_grid, slopes, iterations):
    """Feature, slope root and criterion of every cell where the slope turns up."""
    cell, feature = np.nonzero((slopes[:-1] < 0) & (slopes[1:] > 0))
    step = _batch_features(pooled, 1)
    parts = [
        _refined(
            pooled, log_grid, cell[i : i + step], feature[i : i + step], iterations
        )
        for i in range(0, cell.size, step)
    ]
    return feature, _stacked(parts, 0), _stacked(parts, 1)


def _lowest_root(n_features, feature, root, value):
    """Per feature, its lowest-criterion root; value inf where it has none."""
    lowest = np.full(n_features, np.inf)
    np.minimum.at(lowest, feature, value)
    position = np.full(n_features, np.nan)
    is_lowest = value == lowest[feature]
    position[feature[is_lowest]] = root[is_lowest]
    return position, lowest


def _rounding_slack(value, dof):
    """Criterion rounding noise. On a flat criterion a grid point can sit a few
    ulps below the better-determined slope root, which must still win; at an
    exact rounding tie with lambda = 0 the result may therefore flip to zero."""
    return _ROUNDING_ULPS * np.finfo(float).eps * (np.abs(value) + dof)


def _best_grid_point(values):
    index = np.argmin(values, axis=0)
    return index, np.take_along_axis(values, index[None], axis=0)[0]


def _best_positive(pooled, log_grid, iterations):
    values, slopes = _on_grid(pooled, log_grid)
    index, grid_value = _best_grid_point(values)
    roots = _crossing_roots(pooled, log_grid, slopes, iterations)
    root, root_value = _lowest_root(index.size, *roots)
    accept = root_value <= grid_value + _rounding_slack(grid_value, pooled.dof)
    grid_stationary = np.take_along_axis(slopes, index[None], axis=0)[0] == 0.0
    refined = accept | grid_stationary
    best = np.where(accept, root, log_grid[index])
    # The grid's upper edge is a truncation, not an optimum: flag it.
    on_edge = (index == log_grid.size - 1) & ~refined
    return np.exp(best), np.where(accept, root_value, grid_value), on_edge, refined


def _boundary_optimum(pooled, ratio, *, at_zero) -> RatioOptimum:
    """An optimum that needs no search; its criterion is the value at zero."""
    n_features = pooled.residual_ss.size
    flags = np.full(n_features, at_zero)
    return RatioOptimum(
        ratio=np.full(n_features, ratio),
        criterion=pooled_criterion_at_zero(pooled),
        at_zero=flags,
        at_upper=np.zeros(n_features, dtype=bool),
        unrefined=np.zeros(n_features, dtype=bool),
        unidentifiable=~flags,
    )


def _confounded(pooled: PooledTerms) -> bool:
    """True when the random covariance is c I on the whole restricted space.

    That needs a retained direction for every restricted degree of freedom
    and one common singular value, up to the rounding of the decompositions
    that produced them (relative error about dof * eps per singular value).
    """
    singular2 = pooled.singular2
    spread = singular2.max() - singular2.min()
    tolerance = 4 * pooled.dof * np.finfo(float).eps * singular2.max()
    return singular2.size == pooled.dof and spread <= tolerance


def _search_batch(pooled, log_grid, iterations) -> RatioOptimum:
    """Search one batch of features and compare the result with zero."""
    ratio, value, on_edge, refined = _best_positive(pooled, log_grid, iterations)
    at_zero_value = pooled_criterion_at_zero(pooled)
    at_zero = at_zero_value <= value + _rounding_slack(at_zero_value, pooled.dof)
    return RatioOptimum(
        ratio=np.where(at_zero, 0.0, ratio),
        criterion=np.where(at_zero, at_zero_value, value),
        at_zero=at_zero,
        at_upper=on_edge & ~at_zero,
        unrefined=~refined & ~at_zero,
        unidentifiable=np.zeros(value.size, dtype=bool),
    )


def _joined(parts: Sequence[RatioOptimum]) -> RatioOptimum:
    names = ("ratio", "criterion", *_FLAGS)
    return RatioOptimum(
        **{name: np.concatenate([getattr(p, name) for p in parts]) for name in names}
    )


def _searched_optimum(pooled, grid_size, iterations) -> RatioOptimum:
    """Search the features batch by batch within the working budget."""
    log_grid = _search_grid(pooled.singular2, grid_size)
    n_features = pooled.residual_ss.size
    step = _batch_features(pooled, log_grid.size)
    batches = (slice(i, i + step) for i in range(0, n_features, step))
    return _joined(
        [_search_batch(_take(pooled, b), log_grid, iterations) for b in batches]
    )


def minimize_ratio(
    terms: Sequence[RemlTerms] | PooledTerms,
    *,
    grid_size: int = 121,
    iterations: int = 48,
) -> RatioOptimum:
    """Minimise the profiled criterion over lambda >= 0 for every feature.

    ``iterations`` is the number of slope bisections inside each bracket.
    Unidentifiable fits have a NaN ratio and the criterion's constant value.
    """
    pooled = terms if isinstance(terms, PooledTerms) else pool_terms(terms)
    if pooled.singular2.size == 0:
        return _boundary_optimum(pooled, 0.0, at_zero=True)
    if _confounded(pooled):
        return _boundary_optimum(pooled, np.nan, at_zero=False)
    return _searched_optimum(pooled, grid_size, iterations)


# ---------------------------------------------------------------- fitting


@dataclass(frozen=True, eq=False)
class RemlGroup:
    """Signals ``(scans, features)`` and their unpenalised fixed design ``C``."""

    signals: np.ndarray
    fixed_design: np.ndarray

    def __post_init__(self):
        copy_array_fields_in_place(self, ("signals", "fixed_design"))


@dataclass(frozen=True, eq=False)
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


@dataclass(eq=False)
class _FitBuffers:
    """Per-feature estimates written subset by subset, then frozen to a ``RemlFit``."""

    ratio: np.ndarray
    noise_variance: np.ndarray
    criterion: np.ndarray
    status: np.ndarray
    at_upper: np.ndarray
    unrefined: np.ndarray

    @classmethod
    def empty(cls, n_features):
        """NaN estimates; fitting overwrites features with resolved residuals."""
        return cls(
            *(np.full(n_features, np.nan) for _ in range(3)),
            status=np.full(n_features, "no_residual_signal", dtype=object),
            at_upper=np.zeros(n_features, dtype=bool),
            unrefined=np.zeros(n_features, dtype=bool),
        )

    def write(self, feature_indices, values, keep=slice(None)):
        """Copy a subset's kept values into their feature positions."""
        destination = feature_indices[keep]
        for name, array in vars(values).items():
            getattr(self, name)[destination] = array[keep]

    def mark_unresolved(self, feature_indices):
        """Mark features whose residual signal is numerically unresolved."""
        self.status[feature_indices] = "numerically_unresolved_residual"
        self.ratio[feature_indices] = np.nan

    def freeze(self, rho, restricted_dof) -> RemlFit:
        return RemlFit(
            ratio=self.ratio,
            noise_variance=self.noise_variance,
            effect_variance=self.ratio * self.noise_variance,
            criterion=self.criterion,
            status=tuple(str(s) for s in self.status),
            at_upper=self.at_upper,
            unrefined=self.unrefined,
            rho=rho,
            restricted_dof=restricted_dof,
        )


@dataclass(frozen=True, eq=False)
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
    return _FitBuffers(
        ratio=optimum.ratio,
        noise_variance=pooled_quadratic(optimum.ratio, pooled) / pooled.dof,
        criterion=optimum.criterion,
        status=_classify_variance_ratio_optimum(optimum),
        at_upper=optimum.at_upper,
        unrefined=optimum.unrefined,
    )


def _require_common_restricted_dof(subsets) -> int:
    """The restricted dof shared by every rho subset; rank must not flip with rho."""
    dofs = {subset.dof for subset in subsets}
    if len(dofs) != 1:
        raise ValueError(f"restricted degrees of freedom differ across rho: {dofs}")
    return dofs.pop()


def _prepare_reml_fit(groups, rho):
    """Return feature correlations, prepared subsets, shared dof and fit buffers."""
    rho = feature_rho(rho, groups[0].signals.shape[1])
    subsets = _prepare_feature_subsets_by_rho(groups, rho)
    fit_buffers = _FitBuffers.empty(rho.size)
    for subset in subsets:
        fit_buffers.mark_unresolved(subset.unresolved_feature_indices)
    return rho, subsets, _require_common_restricted_dof(subsets), fit_buffers


def _fit_subsets_into_buffers(fit_buffers, subsets, random_design):
    """Fit each subset and write estimates into the supplied buffers in place."""
    for subset in subsets:
        if subset.fit_feature_indices.size:
            fit_buffers.write(
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
    return fit_buffers.freeze(rho, dof)


# ---------------------------------------------------------------- selection


@dataclass(frozen=True, eq=False)
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


@dataclass(eq=False)
class _SelectionBuffers:
    """Best and runner-up criteria per feature, updated candidate by candidate."""

    winner: np.ndarray
    best_criterion: np.ndarray
    runner_up_criterion: np.ndarray
    any_at_upper: np.ndarray
    any_unrefined: np.ndarray

    @classmethod
    def empty(cls, n_features):
        return cls(
            winner=np.full(n_features, -1, dtype=np.int64),
            best_criterion=np.full(n_features, np.inf),
            runner_up_criterion=np.full(n_features, np.inf),
            any_at_upper=np.zeros(n_features, dtype=bool),
            any_unrefined=np.zeros(n_features, dtype=bool),
        )

    def update(self, feature_indices, values, candidate_index):
        """Rank one candidate's fits; return which of them became the winner."""
        ok, value = values.status == "ok", values.criterion
        best = self.best_criterion[feature_indices]
        runner_up = self.runner_up_criterion[feature_indices]
        better = ok & (value < best)
        contender = np.where(better, best, np.minimum(runner_up, value))
        self.runner_up_criterion[feature_indices] = np.where(ok, contender, runner_up)
        self.best_criterion[feature_indices] = np.where(better, value, best)
        self.winner[feature_indices[better]] = candidate_index
        self.any_at_upper[feature_indices] |= values.at_upper
        self.any_unrefined[feature_indices] |= values.unrefined
        return better

    def gap(self):
        """Runner-up minus best criterion; NaN with fewer than two supported fits."""
        supported = (self.winner >= 0) & np.isfinite(self.runner_up_criterion)
        gap = np.full(supported.shape, np.nan)
        gap[supported] = (
            self.runner_up_criterion[supported] - self.best_criterion[supported]
        )
        return gap

    def freeze(self, fit, n_candidates) -> "RemlSelection":
        return RemlSelection(
            winner=self.winner,
            fit=fit,
            gap=self.gap(),
            n_candidates=n_candidates,
            any_at_upper=self.any_at_upper,
            any_unrefined=self.any_unrefined,
        )


def _fit_candidate_and_update_selection(
    selection_state, fit_buffers, subsets, random_design, candidate_index
):
    """Fit one candidate and update selection state and fit buffers in place."""
    for subset in subsets:
        if subset.fit_feature_indices.size:
            values = _fit_subset_variance_components(subset, random_design)
            indices = subset.fit_feature_indices
            better = selection_state.update(indices, values, candidate_index)
            fit_buffers.write(indices, values, better)


def _evaluate_candidates_and_update_fit_buffers(
    groups, candidates, subsets, fit_buffers
):
    """Update winning-fit buffers; return selection state and candidate count."""
    selection_state, n_candidates = _SelectionBuffers.empty(fit_buffers.ratio.size), 0
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
    return selection_state.freeze(fit_buffers.freeze(rho, dof), count)
