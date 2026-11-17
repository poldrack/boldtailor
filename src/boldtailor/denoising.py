"""Opt-in task-guided denoising following GLMsingle's GLMdenoise stage.

``select_denoising`` follows GLMsingle with stated deviations. It selects
per-feature HRFs once on all runs and freezes them; computes GLMsingle's
ON-OFF R² (canonical HRF, one task regressor, shared coefficient, each run's
confounds plus intercept as nuisance; GLMsingle uses polynomials); pools
features below GLMsingle's Gaussian-mixture tail threshold (by default) and
scores features above it (the 100 best when none passes); extracts run-wise
PCs of that single pool; and chooses one PC count with GLMsingle's pcstop
rule over the median of fold-pooled held-out R², then (a Boldtailor
addition, not part of GLMsingle) keeps that count only if the PCs pass a
per-feature F-test gate (AR(1) prewhitened by default) with a binomial test
across features. Counts are
scored by leave-one-run-out time-series prediction against a fixed target,
because repeated conditions are not assumed. Every input feature is a
candidate: the core is anatomy-agnostic. ``with_denoising`` appends the chosen PCs to the
baseline confounds of the same analysis for ordinary HRF selection and
fitting.

This module reimplements procedures from GLMsingle
(https://github.com/cvnlab/GLMsingle), Copyright (c) 2021, Kendrick Kay,
distributed under the BSD 3-Clause License; see
src/boldtailor/_resources/GLMsingle-LICENSE.txt for the copyright
notice, conditions, and disclaimer.
It is an independent reimplementation, not a copy of
GLMsingle code. Follows the GLMdenoise stage of GLMsingle (noise pool,
run-wise PCs, cross-validated PC count, select_noise_regressors).
Reference: Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr,
M.J., Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. eLife, 11, e77599.
https://doi.org/10.7554/eLife.77599
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import warnings

import numpy as np
import pandas as pd
import scipy.linalg
from nilearn.glm import ARModel
from nilearn.glm.first_level.first_level import _yule_walker
from scipy import stats

from boldtailor._arrays import (
    copy_array_fields_in_place,
    freeze_sequence_fields_in_place,
    readonly_array,
    set_fields_in_place,
    sum_arrays,
)
from boldtailor._fit_lifecycle import build_fit_provenance, fit_operation
from boldtailor._hrf_cv import (
    _basis,
    _check_task_rank,
    pooled_amplitude,
    prepare_runs,
)
from boldtailor._mixture_threshold import MixtureThreshold, mixture_threshold
from boldtailor._scalars import is_integer, is_real
from boldtailor._single_trial_design import _nuisance_matrix
from boldtailor._single_trial_fit import nuisance_span
from boldtailor.data import AnalysisData, _owned_table, run_labels_for
from boldtailor.denoising_results import (
    DenoisingFold,
    DenoisingResult,
    PcaDiagnostics,
    RunIdentity,
    SignificanceGate,
)
from boldtailor.hrf_library import HrfLibrary, default_hrf_library
from boldtailor.hrf_selection import select_hrfs
from boldtailor.model import TaskModel
from boldtailor.provenance import (
    RunSources,
    SourceRef,
    analysis_fingerprint,
    identity_activity,
)

_EPS = np.finfo(float).eps
_AUGMENTATION = "denoising_augmentation"

__all__ = [
    "DenoisingFold",
    "DenoisingResult",
    "PcaDiagnostics",
    "select_denoising",
    "with_denoising",
]


# ---- identity ---------------------------------------------------------------------


def array_digest(values) -> str:
    """sha256 of an array's shape and little-endian float64 values."""
    array = np.ascontiguousarray(values, dtype="<f8")
    header = json.dumps(list(array.shape)).encode()
    return sha256(header + array.tobytes()).hexdigest()


def table_digest(frame: pd.DataFrame) -> str:
    """sha256 of a table's row count, column names, dtypes, and values."""
    digest = sha256(
        json.dumps(
            [len(frame), [[str(c), str(t)] for c, t in frame.dtypes.items()]]
        ).encode()
    )
    for column in frame.columns:
        values = pd.util.hash_pandas_object(frame[column], index=False)
        digest.update(values.to_numpy().tobytes())
    return digest.hexdigest()


def run_identities(data) -> tuple[RunIdentity, ...]:
    return tuple(
        RunIdentity(
            n_scans=len(signal),
            signal=array_digest(signal),
            frame_times=array_digest(times),
            confounds=table_digest(confounds),
            confound_columns=tuple(str(c) for c in confounds.columns),
            events=table_digest(events),
        )
        for signal, times, confounds, events in zip(
            data.signals, data.frame_times, data.confounds, data.events, strict=True
        )
    )


def _check_identity_runs(expected, actual, n_features, data):
    if len(actual) != len(expected):
        raise ValueError(
            f"data has {len(actual)} runs; the denoising result was selected "
            f"on {len(expected)} runs"
        )
    if data.n_features != n_features:
        raise ValueError(
            f"data has {data.n_features} features; the denoising result was "
            f"selected on {n_features} features"
        )
    signals = [run.signal for run in expected]
    observed = [run.signal for run in actual]
    if observed != signals and sorted(observed) == sorted(signals):
        raise ValueError(
            "runs are not in the order the denoising result was selected on"
        )


_IDENTITY_FIELDS = (
    ("frame_times", "time grid of run {label} differs"),
    ("signal", "signals of run {label} differ (features reordered or data changed)"),
    ("confounds", "baseline confounds of run {label} differ"),
    ("events", "events of run {label} differ"),
)


def _check_run(expected, actual, label):
    if actual.n_scans != expected.n_scans:
        raise ValueError(
            f"run {label} has {actual.n_scans} rows; the denoising components "
            f"have {expected.n_scans}"
        )
    for name, message in _IDENTITY_FIELDS:
        if getattr(actual, name) != getattr(expected, name):
            raise ValueError(
                message.format(label=label) + " from the analysis the denoising "
                "result was selected on"
            )


def check_identity(expected, n_features, data, labels) -> None:
    """Raise unless ``data`` is the analysis the result was selected on."""
    actual = run_identities(data)
    _check_identity_runs(expected, actual, n_features, data)
    for want, have, label in zip(expected, actual, labels, strict=True):
        _check_run(want, have, f"'{label}'")


# ---- noise pool and PCA -----------------------------------------------------------


AUTO = "auto"
FALLBACK_SIZE = 100  # GLMsingle scores its 100 best features when none passes
CANONICAL = 0  # every HrfLibrary holds canonical SPM at candidate ID 0


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
    a = sum_arrays(x.T @ x for x, _ in terms)
    b = sum_arrays(x.T @ yr for x, yr in terms)
    amplitude, ok = pooled_amplitude(a[None], b[None])
    if not ok[0]:
        raise ValueError("ON-OFF task design is singular across runs")
    sse = sum_arrays(np.sum((yr - x @ amplitude[0]) ** 2, axis=0) for x, yr in terms)
    sst = sum_arrays(np.sum(yr**2, axis=0) for _, yr in terms)
    ratio = np.full(data.n_features, np.nan)
    np.divide(sse, sst, out=ratio, where=sst > 0)
    return readonly_array(1 - ratio)


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
        copy_array_fields_in_place(self, ("pool", "scoring"), dtype=bool)
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
        mixture = mixture_threshold(values)
    except ValueError as error:
        raise ValueError(
            f"{context}: automatic pool_r2_threshold failed: {error}; "
            "pass a fixed pool_r2_threshold instead"
        ) from error
    if not mixture.converged:
        _warn_unconverged(mixture, context)
    return mixture


def _warn_unconverged(mixture, context):
    warnings.warn(
        f"{context}: the Gaussian mixture for the automatic pool_r2_threshold "
        f"did not converge after {mixture.n_iter} iterations; its threshold "
        f"{mixture.threshold:g} is used. Pass a fixed pool_r2_threshold to "
        "avoid relying on it",
        RuntimeWarning,
        stacklevel=4,
    )


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


@dataclass(frozen=True, kw_only=True)
class RunComponents:
    """Temporal PCs (time x rank) of one run's normalized, projected pool."""

    components: np.ndarray
    singular_values: np.ndarray
    rank_tolerance: float
    pool_size: int
    retained_columns: int

    def __post_init__(self):
        copy_array_fields_in_place(self, ("components", "singular_values"))

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


# ---- cross-validated component-count selection -----------------------------------


NO_TASK_SIGNAL = "no supported task signal for component-count selection"


def validate_counts(counts) -> tuple[int, ...]:
    """Sorted, deduplicated nonnegative integer counts that include zero."""
    message = "counts must be nonnegative integers (not booleans) including 0"
    try:
        values = tuple(counts)
    except TypeError:
        raise ValueError(message) from None
    if any(not is_integer(c) or c < 0 for c in values) or 0 not in values:
        raise ValueError(message)
    return tuple(sorted({int(c) for c in values}))


def validate_pcstop(value) -> float:
    """GLMsingle's ``pcstop``: a finite real number of at least 1."""
    if not is_real(value) or not np.isfinite(value) or value < 1:
        raise ValueError("pcstop must be a finite real number >= 1")
    return float(value)


def _check_inputs(data, indices, components, scoring):
    if indices.shape != (data.n_features,) or not np.issubdtype(
        indices.dtype, np.integer
    ):
        raise ValueError("hrf_indices must be an integer vector, one per feature")
    if len(components) != data.n_runs or any(
        not isinstance(c, RunComponents) or c.components.shape[0] != len(t)
        for c, t in zip(components, data.frame_times)
    ):
        raise ValueError("components must hold one RunComponents per run")
    if (scoring & (indices < 0)).any():
        raise ValueError("scoring features need a defined HRF assignment")
    if not scoring.any():
        raise ValueError(f"{NO_TASK_SIGNAL}: the scoring set is empty")


@dataclass(frozen=True)
class _Terms:
    """One run's task design and BOLD for one HRF, off baseline + indicators.

    ``baseline`` (confounds plus intercept) and ``profiled`` (indicators) are
    orthonormal bases of the nuisance span that ``x`` and ``y`` are off.
    """

    x: np.ndarray
    raw: np.ndarray
    y: np.ndarray
    profiled: np.ndarray
    baseline: np.ndarray
    nuisance_rank: int
    zero: np.ndarray


@dataclass(frozen=True)
class _HrfGroup:
    """Scoring features sharing one frozen HRF, with every run's terms."""

    hrf_id: int
    features: np.ndarray
    runs: tuple[_Terms, ...]


@dataclass(frozen=True, kw_only=True, eq=False)
class ScoringSetup:
    """Frozen full-data inputs and cached projected terms for every run.

    ``zero_target`` (runs x features) marks numerically zero held-out
    targets; ``scored`` is ``scoring`` minus features with a zero target in
    any run, fixed for every fold and count. ``target_energy`` (runs x
    features) is each held-out target's energy, NaN outside ``scored``.
    """

    run_labels: tuple[str, ...]
    hrf_indices: np.ndarray
    components: tuple[RunComponents, ...]
    scoring: np.ndarray
    scored: np.ndarray
    zero_target: np.ndarray
    target_energy: np.ndarray
    n_regressors: int
    groups: tuple[_HrfGroup, ...] = field(repr=False)

    def __post_init__(self):
        copy_array_fields_in_place(
            self, ("scoring", "scored", "zero_target"), dtype=bool
        )
        copy_array_fields_in_place(self, ("hrf_indices",), dtype=np.int64)
        copy_array_fields_in_place(self, ("target_energy",))
        freeze_sequence_fields_in_place(self, ("run_labels", "components", "groups"))

    @property
    def n_runs(self) -> int:
        return len(self.run_labels)


def _run_terms(run, hrf_id, y, label):
    ok, reason = run.eligible(hrf_id)
    if not ok:
        raise ValueError(
            f"run '{label}': HRF {hrf_id} task design is invalid: {reason}"
        )
    block = run.block(hrf_id)
    projected = y - run.q @ (run.q.T @ y)
    projected -= block.qp @ (block.qp.T @ projected)
    # Numerical zero only, as in HRF selection: no weak-signal threshold.
    tolerance = np.linalg.norm(y, axis=0) * len(y) * _EPS
    zero = np.linalg.norm(projected, axis=0) <= tolerance
    rank = run.q.shape[1] + block.qp.shape[1]
    return _Terms(block.x, block.raw, projected, block.qp, run.q, rank, zero)


def _hrf_groups(data, runs, labels, indices, scoring):
    groups = []
    for hrf_id in np.unique(indices[scoring]):
        features = np.flatnonzero(scoring & (indices == hrf_id))
        terms = tuple(
            _run_terms(run, int(hrf_id), y[:, features], label)
            for run, y, label in zip(runs, data.signals, labels)
        )
        groups.append(_HrfGroup(int(hrf_id), features, terms))
    return tuple(groups)


def _target_summary(groups, n_runs, n_features):
    zero = np.zeros((n_runs, n_features), dtype=bool)
    energy = np.full((n_runs, n_features), np.nan)
    for group in groups:
        for r, terms in enumerate(group.runs):
            zero[r, group.features] = terms.zero
            energy[r, group.features] = np.sum(terms.y**2, axis=0)
    return zero, energy


def _checked_scored(scoring, zero, labels):
    scored = scoring & ~zero.any(axis=0)
    if not scored.any():
        runs = ", ".join(f"'{labels[r]}'" for r in np.flatnonzero(zero.any(axis=1)))
        raise ValueError(
            f"{NO_TASK_SIGNAL}: every scoring feature has a numerically zero "
            f"held-out target in run(s) {runs}"
        )
    return scored


def prepare_scoring(
    data,
    *,
    hrf_indices,
    components,
    scoring,
    task_model,
    library,
    run_labels=None,
) -> ScoringSetup:
    """Validate frozen inputs and cache every run's projected terms.

    Messages name runs by ``run_labels`` (default ``run-01``, ...).
    """
    labels = run_labels_for(data, run_labels)
    indices = np.asarray(hrf_indices)
    scoring = validate_feature_mask(scoring, data.n_features, name="scoring")
    components = tuple(components)
    _check_inputs(data, indices, components, scoring)
    runs = prepare_runs(data, library, task_model, labels=labels)
    groups = _hrf_groups(data, runs, labels, indices, scoring)
    zero, energy = _target_summary(groups, data.n_runs, data.n_features)
    scored = _checked_scored(scoring, zero, labels)
    energy[:, ~scored] = np.nan
    return ScoringSetup(
        run_labels=labels,
        hrf_indices=indices,
        components=components,
        scoring=scoring,
        scored=scored,
        zero_target=zero,
        target_energy=energy,
        n_regressors=len(task_model.regressor_names),
        groups=groups,
    )


def _extra_basis(pcs, profiled):
    """Orthonormal PC directions outside the (orthogonal) indicator span."""
    if pcs.shape[1] == 0:
        return pcs
    w = pcs - profiled @ (profiled.T @ pcs)
    u, s, _ = np.linalg.svd(w, full_matrices=False)
    # PCs have unit norm, so the rank tolerance is absolute.
    return u[:, s > max(w.shape) * _EPS]


def _training_statistics(terms, pcs):
    """A and B after also projecting off the PCs; raises if not estimable."""
    w = _extra_basis(pcs, terms.profiled)
    x = terms.x - w @ (w.T @ terms.x)
    y = terms.y - w @ (w.T @ terms.y)
    dof = len(x) - terms.nuisance_rank - w.shape[1]
    _check_task_rank(x, terms.raw, dof)
    return x.T @ x, x.T @ y


def _run_statistics(setup, run, count):
    """Per HRF group (A, B) for one run as a training run, or a reason."""
    comps = setup.components[run]
    reason = comps.unavailable_reason(count)
    if reason:
        return None, reason
    pcs = comps.prefix(count)
    stats = []
    for group in setup.groups:
        try:
            stats.append(_training_statistics(group.runs[run], pcs))
        except ValueError as error:
            return None, f"HRF {group.hrf_id}: {error}"
    return stats, ""


def _training_reason(setup, per_run, held_out):
    for run, (_, reason) in enumerate(per_run):
        if run != held_out and reason:
            return f"training run '{setup.run_labels[run]}': {reason}"
    return ""


def _fold_coefficients(setup, per_run, held_out):
    """Shared task coefficients from the training runs, or a reason."""
    reason = _training_reason(setup, per_run, held_out)
    if reason:
        return None, reason
    train = [stats for run, (stats, _) in enumerate(per_run) if run != held_out]
    beta = np.full((setup.n_regressors, len(setup.scored)), np.nan)
    for g, group in enumerate(setup.groups):
        # Explicit sums avoid cancellation when one run has much larger energy.
        a_sum = sum_arrays(stats[g][0] for stats in train)
        b_sum = sum_arrays(stats[g][1] for stats in train)
        amplitude, ok = pooled_amplitude(a_sum[None], b_sum[None])
        if not ok[0]:
            return None, f"HRF {group.hrf_id}: pooled task design is singular"
        beta[:, group.features] = amplitude[0]
    return beta, ""


def _heldout_sse(setup, held_out, beta):
    sse = np.full(len(setup.scored), np.nan)
    for group in setup.groups:
        held = group.runs[held_out]
        residual = held.y - held.x @ beta[:, group.features]
        sse[group.features] = np.sum(residual**2, axis=0)
    sse[~setup.scored] = np.nan
    return sse


@dataclass(frozen=True, kw_only=True, eq=False)
class CountScore:
    """Per-fold coefficients and errors, pooled feature R², and median perf.

    Arrays are indexed by held-out run. ``perf`` and ``feature_r2`` are NaN
    unless the count is available in every fold.
    """

    count: int
    reason: str
    fold_reasons: tuple[str, ...]
    coefficients: np.ndarray
    fold_sse: np.ndarray
    fold_sst: np.ndarray
    feature_r2: np.ndarray
    perf: float

    def __post_init__(self):
        freeze_sequence_fields_in_place(self, ("fold_reasons",))
        copy_array_fields_in_place(
            self, ("coefficients", "fold_sse", "fold_sst", "feature_r2")
        )

    def fold_median(self, scored, held_out) -> float:
        """Median held-out R² of one fold over ``scored`` (diagnostic)."""
        sse, sst = self.fold_sse[held_out, scored], self.fold_sst[held_out, scored]
        return float(np.median(1 - sse / sst))


def _pooled(setup, sse, available):
    r2 = np.full(len(setup.scored), np.nan)
    if not available:
        return r2, np.nan
    scored = setup.scored
    sst = setup.target_energy[:, scored].sum(axis=0)
    r2[scored] = 1 - sse[:, scored].sum(axis=0) / sst
    return r2, float(np.median(r2[scored]))


def _aggregate_reason(setup, reasons):
    return "; ".join(
        f"fold holding out run '{label}': {reason}"
        for label, reason in zip(setup.run_labels, reasons)
        if reason
    )


def score_count(setup: ScoringSetup, count: int) -> CountScore:
    """Score one count in every fold; an invalid zero count raises.

    A positive count is unavailable in a fold when a training run cannot
    supply that many unique PCs or a projected task design loses rank or
    residual degrees of freedom; it must be available in every fold.
    """
    per_run = [_run_statistics(setup, r, count) for r in range(setup.n_runs)]
    shape = (setup.n_runs, len(setup.scored))
    beta = np.full((setup.n_runs, setup.n_regressors, shape[1]), np.nan)
    sse, reasons = np.full(shape, np.nan), []
    for held_out in range(setup.n_runs):
        values, reason = _fold_coefficients(setup, per_run, held_out)
        reasons.append(reason)
        if not reason:
            beta[held_out] = values
            sse[held_out] = _heldout_sse(setup, held_out, values)
    reason = _aggregate_reason(setup, reasons)
    if reason and count == 0:
        raise ValueError(f"zero-component task design is invalid: {reason}")
    r2, perf = _pooled(setup, sse, not reason)
    return CountScore(
        count=int(count),
        reason=reason,
        fold_reasons=tuple(reasons),
        coefficients=beta,
        fold_sse=sse,
        fold_sst=setup.target_energy,
        feature_r2=r2,
        perf=perf,
    )


def fold_score_table(setup, scores) -> pd.DataFrame:
    """Per fold and count: eligibility, reason, fold median R², and sizes."""
    rows = []
    for held_out, label in enumerate(setup.run_labels):
        for score in scores:
            reason = score.fold_reasons[held_out]
            median = np.nan if reason else score.fold_median(setup.scored, held_out)
            rows.append(
                dict(
                    validation_run=held_out,
                    validation_label=label,
                    count=score.count,
                    eligible=not reason,
                    reason=reason,
                    median_r2=median,
                    n_scored=int(setup.scored.sum()),
                    n_zero_target=int(setup.zero_target[held_out].sum()),
                )
            )
    return pd.DataFrame(rows)


def score_table(scores) -> pd.DataFrame:
    """Per count: eligibility, median performance, curve, and reasons."""
    perf = np.array([s.perf for s in scores])
    zero = perf[[s.count for s in scores].index(0)]
    return pd.DataFrame(
        dict(
            count=[s.count for s in scores],
            eligible=[not s.reason for s in scores],
            perf=perf,
            curve=perf - zero,
            reason=[s.reason for s in scores],
        )
    )


def choose_count(counts, perf, pcstop) -> int:
    """GLMsingle ``select_noise_regressors`` over the available counts.

    With ``curve = perf - perf[0]``, walk counts in increasing order keeping
    the best curve value so far and its count; stop at the first count where
    ``best * pcstop >= max(curve)`` and return the count holding that best.
    NaN marks an unavailable count (skipped). Zero is always a candidate,
    and ``max(curve) <= 0`` chooses 0. This matches GLMsingle's walk over
    ``p = 0 .. numpcstotry``, except for a roundoff slack of 64 eps (as in
    HRF choice) that keeps numerically equal values equal.
    """
    counts, perf = tuple(counts), np.asarray(perf, dtype=float)
    if len(perf) != len(counts):
        raise ValueError("choose_count needs one score per count")
    if 0 not in counts or not np.isfinite(perf[counts.index(0)]):
        raise ValueError("count zero must be a scored candidate")
    curve = perf - perf[counts.index(0)]
    top = float(np.max(curve[np.isfinite(curve)]))
    if top <= 0:
        return 0
    slack = 64 * _EPS * max(1.0, abs(top))
    best, chosen = -np.inf, 0
    for i in np.argsort(counts, kind="stable"):
        if not np.isfinite(curve[i]):
            continue
        if curve[i] > best:
            best, chosen = curve[i], counts[i]
        if best * pcstop >= top - slack:
            break
    return chosen


@dataclass(frozen=True, kw_only=True, eq=False)
class CountSelection:
    """Chosen count, the scoring setup, per-count scores, and score tables."""

    n_components: int
    counts: tuple[int, ...]
    setup: ScoringSetup
    count_scores: tuple[CountScore, ...]
    fold_scores: pd.DataFrame
    scores: pd.DataFrame

    def __post_init__(self):
        freeze_sequence_fields_in_place(self, ("counts", "count_scores"))
        set_fields_in_place(
            self,
            fold_scores=_owned_table(self.fold_scores),
            scores=_owned_table(self.scores),
        )


def select_component_count(
    data,
    *,
    hrf_indices,
    components,
    scoring,
    task_model,
    library,
    counts,
    pcstop,
    run_labels=None,
) -> CountSelection:
    """Leave-one-run-out PC-count choice by pcstop; requires three runs."""
    counts, pcstop = validate_counts(counts), validate_pcstop(pcstop)
    if data.n_runs < 3:
        raise ValueError("component-count selection requires at least three runs")
    setup = prepare_scoring(
        data,
        hrf_indices=hrf_indices,
        components=components,
        scoring=scoring,
        task_model=task_model,
        library=library,
        run_labels=run_labels,
    )
    scores = tuple(score_count(setup, count) for count in counts)
    table = score_table(scores)
    return CountSelection(
        n_components=choose_count(counts, table["perf"], pcstop),
        counts=counts,
        setup=setup,
        count_scores=scores,
        fold_scores=fold_score_table(setup, scores),
        scores=table,
    )


# ---- significance gate ------------------------------------------------------------


KEPT, REJECTED = "kept", "rejected"
SKIPPED, DISABLED = "skipped_zero_count", "disabled"
ZERO_TARGET = "numerically zero target in at least one run (not scored)"
NOISE_MODELS = ("ar1", "ols")
# Nilearn's run_glm default number of AR(1) bins.
AR1_BINS = 100


def _alpha(value, name) -> float:
    if not is_real(value) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite real number in (0, 1)")
    return float(value)


def validate_gate(enabled, alpha, binomial_alpha) -> tuple[bool, float, float]:
    """``significance_gate`` (a bool) and its two levels in (0, 1)."""
    if type(enabled) is not bool:
        raise ValueError("significance_gate must be True or False")
    return (
        enabled,
        _alpha(alpha, "gate_alpha"),
        _alpha(binomial_alpha, "gate_binomial_alpha"),
    )


def validate_noise_model(value) -> str:
    """``gate_noise_model``: ``"ar1"`` (prewhitened) or ``"ols"``."""
    if not isinstance(value, str) or value not in NOISE_MODELS:
        raise ValueError(f"gate_noise_model must be one of {NOISE_MODELS}")
    return value


@dataclass(frozen=True)
class _Fit:
    sse: np.ndarray
    rank: int
    residuals: tuple[np.ndarray, ...]


def _rank_reason(x, raw, dof) -> str:
    """Why the projected task design is not estimable, or empty when it is."""
    try:
        _check_task_rank(x, raw, dof)
    except ValueError as error:
        return str(error)
    return ""


def _stacked_fit(group, bases) -> tuple[_Fit | None, str]:
    """Pooled task fit after projecting each run off its nuisance and ``bases``.

    Returns ``(None, reason)`` when the stacked task design is rank
    deficient after projection or leaves no residual degrees of freedom.
    """
    parts = [
        (terms.x - w @ (w.T @ terms.x), terms.y - w @ (w.T @ terms.y))
        for terms, w in zip(group.runs, bases)
    ]
    x = np.vstack([p[0] for p in parts])
    nuisance = sum(t.nuisance_rank + w.shape[1] for t, w in zip(group.runs, bases))
    raw = np.vstack([t.raw for t in group.runs])
    reason = _rank_reason(x, raw, len(x) - nuisance)
    if reason:
        return None, reason
    a, b = x.T @ x, sum_arrays(p[0].T @ p[1] for p in parts)
    amplitude, ok = pooled_amplitude(a[None], b[None])
    if not ok[0]:
        return None, "pooled task design is singular"
    residuals = tuple(y - xr @ amplitude[0] for xr, y in parts)
    sse = sum(np.sum(e**2, axis=0) for e in residuals)
    return _Fit(sse, nuisance + x.shape[1], residuals), ""


def ar1_coefficients(residuals) -> np.ndarray:
    """Binned lag-1 coefficients per column of (scans x features) residuals.

    Nilearn's Yule-Walker estimate, truncated to ``1 / AR1_BINS`` as
    ``run_glm`` does. Columns without residual energy get 0 (no whitening).
    """
    rho = np.zeros(residuals.shape[1])
    ok = np.sum(residuals**2, axis=0) > 0
    if ok.any():
        rho[ok] = _yule_walker(residuals[:, ok].T, 1)[:, 0]
    return (rho * AR1_BINS).astype(int) / AR1_BINS


def _off(nuisance, x, y):
    """``x`` and ``y`` projected off the span of ``nuisance``."""
    q = _basis(nuisance)
    return x - q @ (q.T @ x), y - q @ (q.T @ y)


def _whitened_run(terms, basis, rho, cols):
    """One run's reduced and full pieces for features ``cols``, AR(1) whitened.

    Nilearn's ``ARModel`` whitens the run's design ``[task, baseline,
    indicators, PCs]`` and its BOLD; each model's whitened task design and
    BOLD are then projected off that model's whitened nuisance columns.
    """
    k, m = terms.x.shape[1], terms.baseline.shape[1] + terms.profiled.shape[1]
    design = np.column_stack([terms.x, terms.baseline, terms.profiled, basis])
    model = ARModel(design, float(rho))
    white, y = model.whitened_design, model.whiten(terms.y[:, cols])
    x = white[:, :k]
    return (cols, *_off(white[:, k : k + m], x, y)), (cols, *_off(white[:, k:], x, y))


def _whitened_pieces(group, bases, rho):
    """Reduced and full ``(features, x, y)`` pieces per run and coefficient bin."""
    reduced, full = [], []
    for terms, basis, run_rho in zip(group.runs, bases, rho):
        for value in np.unique(run_rho):
            cols = np.flatnonzero(run_rho == value)
            r, f = _whitened_run(terms, basis, value, cols)
            reduced.append(r)
            full.append(f)
    return reduced, full


def _pooled_sse(pieces, n, k):
    """Per-feature SSE of the pooled task fit over whitened run pieces."""
    a, b = np.zeros((n, k, k)), np.zeros((n, k, 1))
    for cols, x, y in pieces:
        a[cols] += x.T @ x
        b[cols, :, 0] += (x.T @ y).T
    amplitude, ok = pooled_amplitude(a, b)
    if not ok.all():
        # Whitening is invertible, so this is a numerical failure, not a reason.
        raise ArithmeticError("whitened pooled task design is singular")
    sse = np.zeros(n)
    for cols, x, y in pieces:
        sse[cols] += np.sum((y - x @ amplitude[cols, :, 0].T) ** 2, axis=0)
    return sse


@dataclass(frozen=True)
class _GroupTest:
    features: np.ndarray
    f: np.ndarray | None = None
    df1: int = 0
    df2: int = 0
    reason: str = ""
    rho: np.ndarray | None = None


def _nested_sse(group, bases, fits, noise_model):
    """(SSE reduced, SSE full, rho per run x feature or None)."""
    reduced, full = fits
    if noise_model == "ols":
        return reduced.sse, full.sse, None
    rho = np.array([ar1_coefficients(e) for e in full.residuals])
    n, k = len(group.features), group.runs[0].x.shape[1]
    pieces = _whitened_pieces(group, bases, rho)
    sse_r, sse_f = (_pooled_sse(p, n, k) for p in pieces)
    return sse_r, sse_f, rho


def _group_test(setup, group, count, noise_model) -> _GroupTest:
    pcs = [c.prefix(count) for c in setup.components]
    bases = [_extra_basis(p, t.profiled) for p, t in zip(pcs, group.runs)]
    empty = [np.zeros((len(t.y), 0)) for t in group.runs]
    (reduced, reason), (full, full_reason) = (
        _stacked_fit(group, empty),
        _stacked_fit(group, bases),
    )
    if reduced is None or full is None:
        return _GroupTest(group.features, reason=reason or full_reason)
    df1 = full.rank - reduced.rank
    if df1 <= 0:
        return _GroupTest(group.features, reason="PCs add no columns to the model")
    df2 = sum(len(t.y) for t in group.runs) - full.rank
    sse_r, sse_f, rho = _nested_sse(group, bases, (reduced, full), noise_model)
    gain = np.maximum(sse_r - sse_f, 0.0)
    # Zero-target features (0 / 0) are excluded by the caller.
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (gain / df1) / (sse_f / df2)
    return _GroupTest(group.features, f, df1, df2, rho=rho)


def _arrays(n_runs, n_features):
    values = {key: np.full(n_features, np.nan) for key in ("f", "p", "df1", "df2")}
    values.update(
        tested=np.zeros(n_features, bool),
        excluded=np.zeros(n_features, bool),
        rho=np.full((n_runs, n_features), np.nan),
    )
    return values


def _record_test_results_in_place(values, exclusions, hrf_id, test, zero):
    """Update values arrays and append exclusion records; do not modify test."""
    if test.reason:
        values["excluded"][test.features] = True
        exclusions.append((hrf_id, len(test.features), test.reason))
        return
    keep, drop = test.features[~zero], test.features[zero]
    if len(drop):
        values["excluded"][drop] = True
        exclusions.append((hrf_id, len(drop), ZERO_TARGET))
    values["tested"][keep] = True
    values["f"][keep] = test.f[~zero]
    values["p"][keep] = stats.f.sf(test.f[~zero], test.df1, test.df2)
    values["df1"][keep], values["df2"][keep] = test.df1, test.df2
    if test.rho is not None:
        values["rho"][:, keep] = test.rho[:, ~zero]


def _feature_tests(setup, count, noise_model):
    values, exclusions = _arrays(setup.n_runs, len(setup.scoring)), []
    for group in setup.groups:
        # The gate tests exactly the features pcstop scored.
        zero = ~setup.scored[group.features]
        test = _group_test(setup, group, count, noise_model)
        _record_test_results_in_place(values, exclusions, group.hrf_id, test, zero)
    return values, exclusions


def _gate(settings, count, decision, values, exclusions=(), m=0, binomial_p=np.nan):
    enabled, alpha, binomial_alpha, noise_model = settings
    return SignificanceGate(
        enabled=enabled,
        alpha=alpha,
        binomial_alpha=binomial_alpha,
        noise_model=noise_model,
        pcstop_count=int(count),
        n_components=int(count) if decision in (KEPT, DISABLED) else 0,
        decision=decision,
        f_statistic=values["f"],
        p_value=values["p"],
        df1=values["df1"],
        df2=values["df2"],
        tested=values["tested"],
        excluded=values["excluded"],
        ar_coefficients=values["rho"],
        exclusions=tuple(exclusions),
        m=int(m),
        n=int(values["tested"].sum()),
        binomial_p=float(binomial_p),
    )


def apply_gate(
    setup: ScoringSetup,
    count: int,
    *,
    enabled,
    alpha,
    binomial_alpha,
    noise_model,
) -> SignificanceGate:
    """Keep the pcstop ``count`` only if the PCs pass the F-test gate.

    ``count`` must be available in every run (as a pcstop choice is).
    ``noise_model`` is ``"ar1"`` (prewhitened F-tests) or ``"ols"``.
    With the gate disabled the count is returned untested; a zero count
    skips the gate. With no testable feature the count is rejected.
    """
    settings = (
        *validate_gate(enabled, alpha, binomial_alpha),
        validate_noise_model(noise_model),
    )
    if not enabled or count == 0:
        decision = DISABLED if not enabled else SKIPPED
        empty = _arrays(setup.n_runs, len(setup.scoring))
        return _gate(settings, count, decision, empty)
    values, exclusions = _feature_tests(setup, count, noise_model)
    tested = values["tested"]
    m, n = int(np.sum(values["p"][tested] < alpha)), int(tested.sum())
    if n == 0:
        return _gate(settings, count, REJECTED, values, exclusions)
    p = stats.binomtest(m, n, alpha, alternative="greater").pvalue
    decision = KEPT if p < binomial_alpha else REJECTED
    return _gate(settings, count, decision, values, exclusions, m, p)


# ---- validation and labels --------------------------------------------------------


def _check_data(data):
    if not isinstance(data, AnalysisData):
        raise ValueError("data must be an AnalysisData")


@dataclass(frozen=True)
class _Settings:
    """Validated, owned selection settings."""

    task_model: TaskModel
    library: HrfLibrary
    counts: tuple[int, ...]
    threshold: float | str
    pcstop: float
    gate: tuple[bool, float, float, str]
    feature_signature: str | None


def _check_models(task_model, library, signature):
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel")
    if not isinstance(library, HrfLibrary):
        raise ValueError("library must be an HrfLibrary")
    if signature is not None and (not isinstance(signature, str) or not signature):
        raise ValueError("feature_signature must be a nonempty string or None")


def _settings(data, task_model, library, grid, signature):
    """Validate everything before any fitting.

    ``grid`` is (counts, r2 threshold, pcstop, (gate, alpha, binomial alpha,
    noise model)).
    """
    _check_data(data)
    _check_models(task_model, library, signature)
    counts, threshold, pcstop, gate = grid
    return _Settings(
        task_model=task_model,
        library=library,
        counts=validate_counts(counts),
        threshold=validate_threshold(threshold),
        pcstop=validate_pcstop(pcstop),
        gate=(*validate_gate(*gate[:3]), validate_noise_model(gate[3])),
        feature_signature=signature,
    )


def _check_run_designs(data, labels, settings):
    """At least three runs; every run design valid, errors named by label."""
    if data.n_runs < 3:
        raise ValueError("denoising selection requires at least three runs")
    prepare_runs(data, settings.library, settings.task_model, labels=labels)
    prepare_runs(data, settings.library, TaskModel(), labels=labels)


def _digest(values, dtype) -> str:
    return sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


# ---- full-data HRFs, pool, and PCs ---------------------------------------------------


def _pool(data, labels, settings):
    """Frozen full-data HRFs, ON-OFF R², masks, and run-wise pool PCs."""
    selection = select_hrfs(
        data,
        library=settings.library,
        task_model=settings.task_model,
        run_labels=labels,
        feature_signature=settings.feature_signature,
    )
    statistic = onoff_r2(data, settings.library, run_labels=labels)
    runs = ", ".join(f"'{label}'" for label in labels)
    masks = pool_masks(
        statistic,
        settings.threshold,
        hrf_indices=selection.hrf_indices,
        context=f"the noise pool (runs {runs})",
    )
    if masks.scoring_size == 0:
        raise ValueError(
            f"{NO_TASK_SIGNAL}: no feature has a finite ON-OFF R² and a defined HRF"
        )
    return selection, statistic, masks, analysis_components(data, masks.pool)


def _count_selection(data, labels, settings, pool):
    """pcstop's count, then Boldtailor's significance gate on it."""
    selection, _, masks, components = pool
    count = select_component_count(
        data,
        hrf_indices=selection.hrf_indices,
        components=components,
        scoring=masks.scoring,
        task_model=settings.task_model,
        library=settings.library,
        counts=settings.counts,
        pcstop=settings.pcstop,
        run_labels=labels,
    )
    enabled, alpha, binomial_alpha, noise_model = settings.gate
    gate = apply_gate(
        count.setup,
        count.n_components,
        enabled=enabled,
        alpha=alpha,
        binomial_alpha=binomial_alpha,
        noise_model=noise_model,
    )
    return count, gate


# ---- result parts ----------------------------------------------------------------------


def _public_tables(count):
    """Aggregate scores and per-fold scores keyed by the held-out run label."""
    folds = count.fold_scores.copy()
    folds["validation_run"] = folds.pop("validation_label")
    return count.scores, folds


def _diagnostics(components, labels, pool_size):
    return PcaDiagnostics(
        run_labels=labels,
        pool_size=pool_size,
        ranks=[c.rank for c in components],
        singular_values=[c.singular_values for c in components],
        rank_tolerances=[c.rank_tolerance for c in components],
        retained_columns=[c.retained_columns for c in components],
    )


def _folds(setup):
    labels = setup.run_labels
    return tuple(
        DenoisingFold(
            validation_run=label,
            training_runs=tuple(x for x in labels if x != label),
            zero_target=setup.zero_target[v],
            target_energy=setup.target_energy[v],
        )
        for v, label in enumerate(labels)
    )


# ---- provenance ---------------------------------------------------------------------------


def _baseline_confounds(data, labels):
    return [
        dict(
            run=label,
            columns=list(identity.confound_columns),
            fingerprint=identity.confounds,
        )
        for label, identity in zip(labels, run_identities(data))
    ]


def _excluded_counts(scores):
    failed = scores[~scores["eligible"]]
    return [
        dict(count=int(c), reason=r) for c, r in zip(failed["count"], failed["reason"])
    ]


def _floats(values):
    return [None if not np.isfinite(v) else float(v) for v in values]


def _rho_summary(gate, labels):
    """Per run: min, median, and max lag-1 coefficient over tested features."""
    if not gate.tested.any():
        return []
    return [
        dict(
            run=label,
            min=float(rho.min()),
            median=float(np.median(rho)),
            max=float(rho.max()),
        )
        for label, rho in zip(labels, gate.ar_coefficients[:, gate.tested])
    ]


def _noise_model_record(gate, labels):
    if gate.noise_model == "ols":
        return dict(
            test="ols_nested_f_in_sample_all_runs",
            noise_model="ols",
            ar1_estimate=None,
            ar1_bins=None,
            ar_coefficient_summary=None,
        )
    return dict(
        test="ar1_prewhitened_nested_f_in_sample_all_runs",
        noise_model="ar1",
        ar1_estimate="yule_walker_full_model_ols_residuals_per_run_and_feature",
        ar1_bins=AR1_BINS,
        ar_coefficient_summary=_rho_summary(gate, labels),
    )


def _gate_record(gate, labels):
    return dict(
        enabled=gate.enabled,
        **_noise_model_record(gate, labels),
        ar_coefficient_fingerprint=_digest(gate.ar_coefficients, "<f8"),
        aggregate="one_sided_binomial_over_tested_features",
        gate_alpha=gate.alpha,
        gate_binomial_alpha=gate.binomial_alpha,
        pcstop_count=gate.pcstop_count,
        n_components=gate.n_components,
        decision=gate.decision,
        m=gate.m,
        n=gate.n,
        n_excluded=gate.n_excluded,
        binomial_p=_floats([gate.binomial_p])[0],
        f_statistic_fingerprint=_digest(gate.f_statistic, "<f8"),
        p_value_fingerprint=_digest(gate.p_value, "<f8"),
        exclusions=[
            dict(hrf_index=h, n_features=n, reason=r) for h, n, r in gate.exclusions
        ],
    )


def _pool_records(settings, statistic, masks, scored):
    rule = "fixed" if masks.mixture is None else masks.mixture.to_dict()["method"]
    return dict(
        pool_statistic="glmsingle_onoff_r2",
        pool_r2_threshold=settings.threshold,
        pool_threshold_rule=rule,
        noise_pool_threshold=masks.threshold,
        noise_pool_mixture=None if masks.mixture is None else masks.mixture.to_dict(),
        scoring_fallback=masks.fallback,
        onoff_r2_fingerprint=_digest(statistic, "<f8"),
        noise_pool_fingerprint=_digest(masks.pool, "|b1"),
        noise_pool_size=masks.pool_size,
        scoring_mask_fingerprint=_digest(masks.scoring, "|b1"),
        scoring_mask_size=masks.scoring_size,
        scored_fingerprint=_digest(scored, "|b1"),
    )


def _count_records(settings, count):
    scores, labels = count.scores, count.setup.run_labels
    return dict(
        counts=list(settings.counts),
        count_rule="glmsingle_pcstop",
        pcstop=settings.pcstop,
        performance="median_over_scored_features_of_fold_pooled_r2",
        perf=_floats(scores["perf"]),
        curve=_floats(scores["curve"]),
        excluded_counts=_excluded_counts(scores),
        score="heldout_task_prediction_r2_fixed_baseline_target",
        conditional_prediction="missing-value indicators profiled on held-out BOLD",
        selection_statistic=True,
        folds=[
            dict(train=[x for x in labels if x != label], test=[label])
            for label in labels
        ],
    )


def _selection_activity(context, pool, selected, prefixes):
    data, labels, settings = context
    selection, statistic, masks, _ = pool
    count, gate = selected
    task_model, library = settings.task_model, settings.library
    return dict(
        name="denoising_selection",
        task_model=task_model.to_dict(),
        task_model_fingerprint=task_model.fingerprint,
        task_regressors=list(task_model.regressor_names),
        profiled_regressors=list(task_model.profiled_names),
        library=dict(library.origin),
        library_fingerprint=library.fingerprint,
        run_labels=list(labels),
        feature_signature=settings.feature_signature,
        n_components=prefixes[0].shape[1],
        **_pool_records(settings, statistic, masks, count.setup.scored),
        **_count_records(settings, count),
        significance_gate=_gate_record(gate, labels),
        initial_hrf_assignment_fingerprint=_digest(selection.hrf_indices, "<i8"),
        initial_selection=identity_activity(selection.provenance),
        component_fingerprints=[_digest(p, "<f8") for p in prefixes],
        baseline_confounds=_baseline_confounds(data, labels),
        data_identity=[identity.to_dict() for identity in run_identities(data)],
    )


# ---- public selection ------------------------------------------------------------------------


def _assemble(context, pool, selected, *, prefixes, provenance):
    """Construct a denoising result from completed calculations and provenance."""
    data, labels, settings = context
    selection, statistic, masks, components = pool
    count, gate = selected
    scores, fold_scores = _public_tables(count)
    return DenoisingResult(
        n_components=gate.n_components,
        pcstop_count=count.n_components,
        significance_gate=gate,
        counts=count.counts,
        pool_r2_threshold=settings.threshold,
        pcstop=settings.pcstop,
        noise_pool_threshold=masks.threshold,
        noise_pool_mixture=masks.mixture,
        noise_pool=masks.pool,
        scoring_mask=masks.scoring,
        scoring_fallback=masks.fallback,
        scored=count.setup.scored,
        onoff_r2=statistic,
        initial_selection=selection,
        run_components=prefixes,
        components=_diagnostics(components, labels, masks.pool_size),
        folds=_folds(count.setup),
        _candidate_scores=scores,
        _fold_scores=fold_scores,
        run_labels=labels,
        feature_signature=settings.feature_signature,
        source_identity=run_identities(data),
        provenance=provenance,
    )


def select_denoising(
    data: AnalysisData,
    *,
    task_model: TaskModel = TaskModel(),
    library: HrfLibrary | None = None,
    counts: tuple[int, ...] = tuple(range(11)),
    pool_r2_threshold: float | str = "auto",
    pcstop: float = 1.05,
    significance_gate: bool = True,
    gate_alpha: float = 0.05,
    gate_binomial_alpha: float = 0.05,
    gate_noise_model: str = "ar1",
    feature_signature: str | None = None,
    run_labels: Sequence[str] | None = None,
) -> DenoisingResult:
    """Choose a temporal-PC count as GLMsingle does; return full-data PCs.

    Requires at least three runs. Every feature is a candidate (no mask).
    HRFs are selected once on all runs with baseline confounds and frozen.
    The pool is the features whose ON-OFF R² falls below
    ``pool_r2_threshold`` (``"auto"``: GLMsingle's Gaussian-mixture tail
    threshold; a degenerate fit raises suggesting a fixed value); features
    above it are scored (the 100 best when none passes). Counts are scored by
    leave-one-run-out held-out task prediction, pooled per feature across
    folds, with the median over features, and chosen by GLMsingle's
    ``pcstop`` rule (``>= 1``). Positive counts that any fold cannot support
    are unavailable; zero always remains. Scores are selection statistics,
    not independent performance estimates.

    ``significance_gate`` (a Boldtailor addition, not part of GLMsingle)
    then keeps the pcstop count only if adding those PCs passes per-feature
    in-sample F-tests (``p < gate_alpha``) in more scoring features than
    chance, by a one-sided binomial test at ``gate_binomial_alpha``;
    otherwise zero PCs are chosen. ``gate_noise_model="ar1"`` (default)
    prewhitens each run's data and design with its own lag-1 coefficient per
    feature, following Nilearn's AR(1) convention; ``"ols"`` does not.
    ``significance_gate=False`` reproduces GLMsingle's pcstop-only choice.
    """
    library = default_hrf_library() if library is None else library
    gate = (significance_gate, gate_alpha, gate_binomial_alpha, gate_noise_model)
    grid = (counts, pool_r2_threshold, pcstop, gate)
    settings = _settings(data, task_model, library, grid, feature_signature)
    labels = run_labels_for(data, run_labels)
    _check_run_designs(data, labels, settings)
    with fit_operation("denoising_selection", data.provenance) as operation:
        pool = _pool(data, labels, settings)
        selected = _count_selection(data, labels, settings, pool)
        _, gate = selected
        # Every run supports the chosen count because each trains some fold.
        _, _, _, components = pool
        prefixes = tuple(c.prefix(gate.n_components) for c in components)
        context = (data, labels, settings)
        activity = _selection_activity(context, pool, selected, prefixes)
        operation.analysis_id = analysis_fingerprint(
            data.provenance.metadata_fingerprint, activity
        )
        provenance = build_fit_provenance(operation, activity)
        return _assemble(
            context, pool, selected, prefixes=prefixes, provenance=provenance
        )


# ---- augmentation -------------------------------------------------------------------------------


def _already_applied(data, result):
    token = result.provenance.execution_id
    return any(
        activity.get("name") == _AUGMENTATION
        and activity.get("denoising_execution_id") == token
        for activity in data.provenance.activities
    )


def _check_application(data, result, signature):
    _check_data(data)
    if not isinstance(result, DenoisingResult):
        raise ValueError("result must be a DenoisingResult")
    if _already_applied(data, result):
        raise ValueError("this denoising result was already applied to the data")
    if signature != result.feature_signature:
        raise ValueError("feature_signature must match the denoising result")
    check_identity(
        result.source_identity, len(result.noise_pool), data, result.run_labels
    )
    names = set(result.component_names)
    for label, frame in zip(result.run_labels, data.confounds):
        clash = sorted(names.intersection(map(str, frame.columns)))
        if clash:
            raise ValueError(
                f"run '{label}' confounds already contain {', '.join(clash)}"
            )


def _augmented_confounds(data, result):
    names = list(result.component_names)
    return tuple(
        pd.concat(
            [frame, pd.DataFrame(components, columns=names, index=frame.index)], axis=1
        )
        for frame, components in zip(data.confounds, result.run_components, strict=True)
    )


def _augmentation_activity(result):
    return dict(
        name=_AUGMENTATION,
        denoising_execution_id=result.provenance.execution_id,
        denoising_analysis_id=result.provenance.analysis_fingerprint,
        n_components=result.n_components,
        added_columns=list(result.component_names),
        component_fingerprints=[_digest(c, "<f8") for c in result.run_components],
        run_labels=list(result.run_labels),
        feature_signature=result.feature_signature,
    )


def _annotated(ref, mark):
    values = ref.to_dict()
    values["annotations"] = {**values.get("annotations", {}), _AUGMENTATION: mark}
    return SourceRef.from_dict(values)


def _annotated_run(sources, mark):
    """Mark the confounds source (else the signal); file identity is kept."""
    if sources.confounds is not None:
        confounds = _annotated(sources.confounds, mark)
        return RunSources(sources.signal, sources.events, confounds)
    return RunSources(_annotated(sources.signal, mark), sources.events, None)


def _annotated_record(data, result):
    """Parent provenance whose sources identify the augmented confounds.

    Downstream analysis ids derive from source metadata, so distinct
    augmentations of the same files must not share it.
    """
    mark = dict(
        selection_execution_id=result.provenance.execution_id,
        component_fingerprints=[_digest(c, "<f8") for c in result.run_components],
        n_components=result.n_components,
    )
    sources = tuple(_annotated_run(s, mark) for s in data.provenance.sources)
    return data.provenance.replace(sources=sources)


def with_denoising(
    data: AnalysisData,
    result: DenoisingResult,
    *,
    feature_signature: str | None = None,
) -> AnalysisData:
    """Append ``denoise_pc_000``, ... to the baseline confounds of each run.

    ``data`` must be the analysis ``result`` was selected on: same runs in the
    same order, time grids, signals (feature order), baseline confounds, and
    events, and a matching ``feature_signature``. Applying a result twice or
    colliding with existing confound names raises. Signals, events, timing,
    and source file identities are preserved. Each run's confounds source
    (the signal source when there are no confounds) gains a
    ``denoising_augmentation`` annotation, so downstream analysis ids differ
    between augmentations, and provenance gains one activity. A zero-count
    result adds no columns but is still recorded.
    """
    _check_application(data, result, feature_signature)
    parent = _annotated_record(data, result)
    with fit_operation(_AUGMENTATION, parent) as operation:
        confounds = _augmented_confounds(data, result)
        activity = _augmentation_activity(result)
        operation.analysis_id = analysis_fingerprint(
            parent.metadata_fingerprint, activity
        )
        provenance = build_fit_provenance(operation, activity)
        return AnalysisData(
            _signals=data.signals,
            _events=data.events,
            _confounds=confounds,
            _frame_times=data.frame_times,
            _timing_source=data.timing_source,
            _provenance=provenance,
        )
