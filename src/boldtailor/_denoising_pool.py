"""GLMsingle ON-OFF R² noise pool, scoring features, and run-wise pool PCA.

The pool statistic is GLMsingle's ON-OFF R²: one task regressor (every
trial, amplitude 1) convolved with the library's canonical HRF (candidate
0), a coefficient shared by all runs, and each run's baseline confounds plus
intercept as run-specific nuisance, fit in-sample on all runs (GLMsingle
uses polynomial drift terms; Boldtailor analyses carry confounds instead).
It ignores the caller's modulators and HRF choices. Every input feature is a
candidate: the core is anatomy-agnostic and takes no mask. The pool holds
finite values below the threshold (GLMsingle ``badR2``), the scoring
features finite values above it with a defined HRF (``pcR2cutoff``, same
threshold), or the 100 best when none passes. The threshold is fixed or,
with ``"auto"``, GLMsingle's two-component Gaussian-mixture tail threshold
of the finite values (see :mod:`boldtailor._mixture_threshold`). Pool time
series are projected off the run's baseline confounds plus intercept,
numerically zero columns are discarded, and the rest are scaled to unit L2
norm before a reduced SVD.

This module reimplements procedures from GLMsingle
(https://github.com/cvnlab/GLMsingle), Copyright (c) 2021, Kendrick Kay,
distributed under the BSD 3-Clause License; see
src/boldtailor/_resources/GLMsingle-LICENSE.txt for the copyright
notice, conditions, and disclaimer.
It is an independent reimplementation, not a copy of
GLMsingle code. Follows the GLMdenoise noise-pool/PC procedure used by
GLMsingle (ON-OFF R² noise pool, run-wise temporal PCs of the pool).
Reference: Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr,
M.J., Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. eLife, 11, e77599.
https://doi.org/10.7554/eLife.77599
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from boldtailor._arrays import own_fields, readonly_array
from boldtailor._hrf_cv import pooled_amplitude, prepare_runs
from boldtailor._mixture_threshold import MixtureThreshold, mixture_threshold
from boldtailor._scalars import is_integer, is_real
from boldtailor._single_trial_design import _nuisance_matrix
from boldtailor._single_trial_fit import nuisance_span
from boldtailor.data import run_labels_for
from boldtailor.model import TaskModel

_EPS = np.finfo(float).eps
AUTO = "auto"
FALLBACK_SIZE = 100  # GLMsingle scores its 100 best features when none passes
CANONICAL = 0  # every HrfLibrary holds canonical SPM at candidate ID 0


# ---- ON-OFF R² ------------------------------------------------------------------


def _onoff_terms(run, y, label):
    """Projected ON column, projected BOLD, and its energy for one run."""
    ok, reason = run.eligible(CANONICAL)
    if not ok:
        raise ValueError(f"run '{label}': ON-OFF task design is invalid: {reason}")
    yr = y - run.q @ (run.q.T @ y)
    # Numerical zero only, as in HRF selection: no weak-signal threshold.
    yr[:, np.linalg.norm(yr, axis=0) <= np.linalg.norm(y, axis=0) * len(y) * _EPS] = 0
    return run.block(CANONICAL).x, yr


def onoff_r2(data, library, *, run_labels=None) -> np.ndarray:
    """GLMsingle ON-OFF R² per feature over all runs; NaN for zero energy.

    ``1 - sum_r ||M_r y_r - M_r x_r b||^2 / sum_r ||M_r y_r||^2`` with ``M_r``
    projecting off run r's confounds plus intercept and ``b`` shared.
    """
    labels = run_labels_for(data, run_labels)
    runs = prepare_runs(data, library, TaskModel(), labels=labels)
    terms = [_onoff_terms(*args) for args in zip(runs, data.signals, labels)]
    a = sum(x.T @ x for x, _ in terms)
    b = sum(x.T @ yr for x, yr in terms)
    amplitude, ok = pooled_amplitude(a[None], b[None])
    if not ok[0]:
        raise ValueError("ON-OFF task design is singular across runs")
    sse = sum(np.sum((yr - x @ amplitude[0]) ** 2, axis=0) for x, yr in terms)
    sst = sum(np.sum(yr**2, axis=0) for _, yr in terms)
    ratio = np.full(data.n_features, np.nan)
    np.divide(sse, sst, out=ratio, where=sst > 0)
    return readonly_array(1 - ratio)


# ---- pool and scoring masks -----------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class PoolMasks:
    """Pool and scoring feature masks fixed for every candidate PC count.

    ``mixture`` is the fitted mixture when the threshold was automatic.
    ``fallback`` marks GLMsingle's best-100 scoring set, which may overlap
    the pool; otherwise the masks are disjoint.
    """

    pool: np.ndarray
    scoring: np.ndarray
    threshold: float
    mixture: MixtureThreshold | None = None
    fallback: bool = False

    def __post_init__(self):
        own_fields(self, ("pool", "scoring"), dtype=bool)
        if self.pool.shape != self.scoring.shape:
            raise ValueError("pool and scoring masks must be aligned")
        if not self.fallback and (self.pool & self.scoring).any():
            raise ValueError("pool and scoring masks must be disjoint")

    @property
    def pool_size(self) -> int:
        return int(self.pool.sum())

    @property
    def scoring_size(self) -> int:
        return int(self.scoring.sum())


def validate_threshold(value, name="pool_r2_threshold") -> float | str:
    """``"auto"`` (mixture rule) or a finite real number, as a float."""
    if isinstance(value, str) and value == AUTO:
        return AUTO
    if not is_real(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be 'auto' or a finite real number")
    return float(value)


def validate_feature_mask(mask, n_features, name="pool") -> np.ndarray:
    """Read-only copy of a Boolean vector with one entry per feature."""
    array = np.asarray(mask)
    if array.dtype != bool or array.shape != (n_features,):
        raise ValueError(f"{name} must be a Boolean vector of {n_features} features")
    return readonly_array(array, dtype=bool)


def _validate_statistic(statistic, n_features):
    try:
        scores = np.asarray(statistic, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("statistic must be a real vector, one per feature") from None
    if scores.shape != (n_features,):
        raise ValueError("statistic must be a real vector, one per feature")
    return scores


def _mixture(values, context):
    try:
        return mixture_threshold(values)
    except ValueError as error:
        raise ValueError(
            f"{context}: automatic pool_r2_threshold failed: {error}; "
            "pass a fixed pool_r2_threshold instead"
        ) from error


def _best_features(scores, candidates):
    """The ``FALLBACK_SIZE`` candidates with the highest scores."""
    ranked = np.flatnonzero(candidates)[np.argsort(-scores[candidates], kind="stable")]
    best = np.zeros(len(scores), dtype=bool)
    best[ranked[:FALLBACK_SIZE]] = True
    return best


def pool_masks(
    statistic: np.ndarray,
    threshold: float | str,
    *,
    hrf_indices: np.ndarray,
    context: str = "the noise pool",
) -> PoolMasks:
    """Pool: finite statistic < threshold; scoring: > threshold, HRF defined.

    With ``"auto"`` the threshold is the mixture tail threshold of all finite
    values; a failed fit raises naming ``context``. When no feature passes,
    the 100 highest-scoring candidates are scored (``fallback``). An empty
    pool or scoring set is returned as such; callers decide if it is fatal.
    """
    threshold = validate_threshold(threshold)
    indices = np.asarray(hrf_indices)
    scores = _validate_statistic(statistic, len(indices))
    finite = np.isfinite(scores)
    mixture = _mixture(scores[finite], context) if threshold == AUTO else None
    threshold = threshold if mixture is None else mixture.threshold
    candidates = finite & (indices >= 0)
    scoring = candidates & (scores > threshold)
    fallback = not scoring.any()
    return PoolMasks(
        pool=finite & (scores < threshold),
        scoring=_best_features(scores, candidates) if fallback else scoring,
        threshold=threshold,
        mixture=mixture,
        fallback=fallback,
    )


# ---- normalized pool PCA ---------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RunComponents:
    """Temporal PCs (time x rank) of one run's normalized, projected pool."""

    components: np.ndarray
    singular_values: np.ndarray
    rank_tolerance: float
    pool_size: int
    retained_columns: int

    def __post_init__(self):
        own_fields(self, ("components", "singular_values"))

    @property
    def rank(self) -> int:
        return self.components.shape[1]

    def splits_tied_block(self, count: int) -> bool:
        """True when s[count-1] and s[count] agree within the rank tolerance."""
        if not 0 < count < self.rank:
            return False
        gap = self.singular_values[count - 1] - self.singular_values[count]
        return bool(gap <= self.rank_tolerance)

    def unavailable_reason(self, count: int) -> str:
        """Empty when ``count`` leading PCs are uniquely defined on this run."""
        count = _validate_count(count)
        if count == 0:
            return ""
        if self.pool_size == 0:
            return "empty noise pool"
        if self.rank == 0:
            return "noise pool has no signal outside the baseline nuisance span"
        if count > self.rank:
            return f"count {count} exceeds pool PCA rank {self.rank}"
        if self.splits_tied_block(count):
            return f"count {count} splits a tied singular-value block"
        return ""

    def prefix(self, count: int) -> np.ndarray:
        """Read-only leading ``count`` PCs; raises when they are not unique."""
        reason = self.unavailable_reason(count)
        if reason:
            raise ValueError(reason)
        return readonly_array(self.components[:, :count])


def _validate_count(count) -> int:
    if not is_integer(count) or count < 0:
        raise ValueError("component count must be a nonnegative integer")
    return int(count)


def _projected_pool(signal, confounds, pool):
    """Unit-norm projected pool columns and each column's norm amplification."""
    q, _ = nuisance_span(_nuisance_matrix(confounds, len(signal)).to_numpy())
    y = np.asarray(signal, dtype=float)[:, pool]
    residual = y - q @ (q.T @ y)
    # Numerical zero only, as in HRF selection: no weak-signal threshold.
    raw, norms = np.linalg.norm(y, axis=0), np.linalg.norm(residual, axis=0)
    keep = norms > raw * len(y) * _EPS
    return residual[:, keep] / norms[keep], raw[keep] / norms[keep]


def _rank_tolerance(normalized, amplification):
    """``max(shape) * eps * scale`` as elsewhere, with scale >= s[0].

    Projection roundoff is ``eps * |y_j|`` per column; normalization by the
    residual norm amplifies it by ``|y_j| / |r_j|``. The 2-norm of those
    factors bounds the normalized perturbation and is at least sqrt(columns),
    hence at least the largest singular value.
    """
    return max(normalized.shape) * _EPS * float(np.linalg.norm(amplification))


def _signed(u):
    """Flip each column so its largest absolute entry is positive."""
    peaks = u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])]
    return u * np.where(peaks < 0, -1.0, 1.0)


def _svd(matrix):
    """Reduced SVD; falls back to LAPACK gesvd when gesdd does not converge.

    gesdd (NumPy's driver) can fail on well-conditioned input with some
    LAPACK builds; gesvd is slower but more robust and gives the same
    decomposition.
    """
    try:
        return np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(matrix, full_matrices=False, lapack_driver="gesvd")


def run_components(signal, confounds, pool) -> RunComponents:
    """Normalized pool PCA for one run; ``confounds`` excludes the intercept."""
    pool = validate_feature_mask(pool, np.shape(signal)[1], name="pool")
    normalized, amplification = _projected_pool(signal, confounds, pool)
    u, s, _ = _svd(normalized)
    tolerance = _rank_tolerance(normalized, amplification)
    rank = int(np.sum(s > tolerance))
    return RunComponents(
        components=_signed(u[:, :rank]),
        singular_values=s,
        rank_tolerance=tolerance,
        pool_size=int(pool.sum()),
        retained_columns=normalized.shape[1],
    )


def analysis_components(data, pool) -> tuple[RunComponents, ...]:
    """Run-specific normalized pool PCs for every run of an AnalysisData."""
    return tuple(
        run_components(signal, confounds, pool)
        for signal, confounds in zip(data.signals, data.confounds, strict=True)
    )
