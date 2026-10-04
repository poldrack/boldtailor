"""Task-guided noise-pool masks and run-wise normalized pool PCA.

The pool is defined from a training HRF selection's time-series CV R² of each
feature's winning HRF, with the held-out denominator projected off the same
baseline confounds and missing-value indicators as the prediction (see
:func:`pool_statistic`). A low score means weak prediction by the specified
task model, not absence of neural activity. Pool time series are projected off the run's baseline
confounds plus intercept, numerically zero columns are discarded, and the rest
are scaled to unit L2 norm before a reduced SVD.
"""

from dataclasses import dataclass

import numpy as np

from boldtailor._arrays import own_fields, readonly_array
from boldtailor._hrf_cv import pooled_amplitude, prediction_loss, prepare_runs
from boldtailor._scalars import is_integer, is_real
from boldtailor._single_trial_design import _nuisance_matrix
from boldtailor._single_trial_fit import nuisance_span
from boldtailor.hrf_results import HrfSelectionResult

_EPS = np.finfo(float).eps


@dataclass(frozen=True, kw_only=True)
class PoolMasks:
    """Disjoint Boolean feature masks fixed for every candidate PC count."""

    pool: np.ndarray
    scoring: np.ndarray
    threshold: float

    def __post_init__(self):
        own_fields(self, ("pool", "scoring"), dtype=bool)
        if self.pool.shape != self.scoring.shape or (self.pool & self.scoring).any():
            raise ValueError("pool and scoring masks must be disjoint and aligned")

    @property
    def pool_size(self) -> int:
        return int(self.pool.sum())

    @property
    def scoring_size(self) -> int:
        return int(self.scoring.sum())


def validate_threshold(value, name="pool_r2_threshold") -> float:
    if not is_real(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def validate_feature_mask(mask, n_features, name="brain_mask") -> np.ndarray:
    """Read-only copy of a Boolean vector with one entry per feature."""
    array = np.asarray(mask)
    if array.dtype != bool or array.shape != (n_features,):
        raise ValueError(f"{name} must be a Boolean vector of {n_features} features")
    return readonly_array(array, dtype=bool)


def _check_selection(selection):
    if not isinstance(selection, HrfSelectionResult):
        raise ValueError("selection must be an HrfSelectionResult")


def _validate_statistic(statistic, n_features):
    try:
        scores = np.asarray(statistic, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("statistic must be a real vector, one per feature") from None
    if scores.shape != (n_features,):
        raise ValueError("statistic must be a real vector, one per feature")
    return scores


def pool_masks(
    selection: HrfSelectionResult,
    brain_mask: np.ndarray,
    threshold: float,
    *,
    statistic: np.ndarray,
) -> PoolMasks:
    """Pool: in-brain finite statistic <= threshold; scoring: > threshold.

    ``statistic`` is normally :func:`pool_statistic`. Features without a
    defined HRF assignment belong to neither mask. Empty masks are returned
    as such; callers decide whether they are fatal.
    """
    _check_selection(selection)
    threshold = validate_threshold(threshold)
    scores = _validate_statistic(statistic, len(selection.hrf_indices))
    brain = validate_feature_mask(brain_mask, len(scores))
    defined = brain & np.isfinite(scores) & (selection.hrf_indices >= 0)
    return PoolMasks(
        pool=defined & (scores <= threshold),
        scoring=defined & (scores > threshold),
        threshold=threshold,
    )


def _run_statistics(run, hrf_id, y):
    """A, B, and indicator-projected energy C for one run and HRF."""
    block = run.block(hrf_id)
    yr = y - run.q @ (run.q.T @ y)
    # Numerical zero only, exactly as in HRF selection.
    yr[:, np.linalg.norm(yr, axis=0) <= np.linalg.norm(y, axis=0) * len(y) * _EPS] = 0
    residual = yr - block.qp @ (block.qp.T @ yr)
    return block.a, block.x.T @ yr, np.sum(residual**2, axis=0)


def _winner_cv_r2(runs, signals, hrf_id, features):
    stats = [_run_statistics(r, hrf_id, y[:, features]) for r, y in zip(runs, signals)]
    a, b, c = (np.array(values) for values in zip(*stats))
    loss = np.zeros(len(features))
    for held in range(len(runs)):
        others = [r for r in range(len(runs)) if r != held]
        amplitude, ok = pooled_amplitude(
            a[others].sum(axis=0)[None], b[others].sum(axis=0)[None]
        )
        if not ok[0]:
            return np.full(len(features), np.nan)
        loss += prediction_loss(a[held][None], b[held][None], c[held][None], amplitude)[
            0
        ]
    total = c.sum(axis=0)
    statistic = np.full(len(features), np.nan)
    np.divide(loss, total, out=statistic, where=total > 0)
    return 1 - statistic


def pool_statistic(data, selection: HrfSelectionResult) -> np.ndarray:
    """LORO CV R² of each feature's selected HRF over ``data``'s runs.

    Unlike ``selection.cv_r2`` (confound-only denominator), each held-out
    run's energy is projected off its confounds, intercept, and the
    missing-value indicators convolved with the same HRF, so in-sample
    indicator profiling does not inflate the score. Without indicators the
    two agree. ``data`` must be the runs ``selection`` was fit on; features
    without an HRF get NaN.
    """
    _check_selection(selection)
    if data.n_runs != len(selection.run_labels):
        raise ValueError("data must have the runs the selection was fit on")
    ids = np.asarray(selection.hrf_indices)
    if len(ids) != data.n_features:
        raise ValueError("data and selection must have the same features")
    runs = prepare_runs(data, selection.library, selection.task_model)
    statistic = np.full(data.n_features, np.nan)
    for hrf_id in np.unique(ids[ids >= 0]):
        features = np.flatnonzero(ids == hrf_id)
        statistic[features] = _winner_cv_r2(runs, data.signals, int(hrf_id), features)
    return readonly_array(statistic)


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


def run_components(signal, confounds, pool) -> RunComponents:
    """Normalized pool PCA for one run; ``confounds`` excludes the intercept."""
    pool = validate_feature_mask(pool, np.shape(signal)[1], name="pool")
    normalized, amplification = _projected_pool(signal, confounds, pool)
    u, s, _ = np.linalg.svd(normalized, full_matrices=False)
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
