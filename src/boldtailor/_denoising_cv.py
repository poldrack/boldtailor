"""Leave-one-run-out choice of a temporal-PC count by held-out task prediction.

Each fold selects HRFs, the noise pool, and pool PCs from its training runs
only. For every candidate count, shared task coefficients are fit across the
training runs after projecting both the task design and BOLD off each run's
baseline confounds, intercept, missing-value indicators (convolved with the
frozen HRF), and that count's leading PCs. The held-out target is projected
off its baseline confounds, intercept, and indicators only. It never contains
PCs, so the target, denominator, and scored features are identical for every
count. Indicator coefficients are profiled on held-out BOLD, so this is
conditional task prediction, and the aggregate is a selection statistic, not
an independent performance estimate.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from boldtailor._arrays import own_fields, own_tuples, rebind
from boldtailor._denoising_pool import (
    PoolMasks,
    RunComponents,
    analysis_components,
    pool_masks,
    pool_statistic,
    validate_feature_mask,
    validate_threshold,
)
from boldtailor._hrf_cv import (
    _check_task_rank,
    _validate_onsets,
    pooled_amplitude,
    prepare_runs,
    subset_runs,
)
from boldtailor._scalars import is_integer, is_real
from boldtailor._single_trial_design import _nuisance_matrix, _validate_events
from boldtailor._task_design import expand_events
from boldtailor.data import _owned_table, run_labels_for
from boldtailor.hrf_library import default_hrf_library
from boldtailor.hrf_results import HrfSelectionResult
from boldtailor.hrf_selection import select_hrfs

_EPS = np.finfo(float).eps
_NO_TASK_SIGNAL = "no supported task signal for component-count selection"


# ---- validation ----------------------------------------------------------------


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


def validate_tolerance(value, name="score_tolerance") -> float:
    if not is_real(value) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return float(value)


def _check_run_design(events, times, confounds, task_model, label):
    name = f"run '{label}'"
    try:
        _validate_events(events, np.asarray(times, dtype=float), label)
        _validate_onsets(events, times)
        _nuisance_matrix(confounds, len(times))
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error
    expand_events(events, task_model, f"'{label}'")  # messages lead with name


def check_run_designs(data, task_model, labels) -> None:
    """Per-run design inputs, as prepare_runs checks them, named by label.

    Run subsets used in folds renumber runs, so design errors are caught
    here on the full analysis, where each run has its own label.
    """
    for events, times, confounds, label in zip(
        data.events, data.frame_times, data.confounds, labels, strict=True
    ):
        _check_run_design(events, times, confounds, task_model, label)


# ---- fold preparation ------------------------------------------------------------


@dataclass(frozen=True)
class _Terms:
    """One run's task design and BOLD for one HRF, off baseline + indicators."""

    x: np.ndarray
    raw: np.ndarray
    y: np.ndarray
    profiled: np.ndarray
    nuisance_rank: int
    zero: np.ndarray


@dataclass(frozen=True)
class _HrfGroup:
    """Scoring features sharing one frozen HRF: training and held-out terms."""

    hrf_id: int
    features: np.ndarray
    train: tuple[_Terms, ...]
    validation: _Terms


@dataclass(frozen=True, kw_only=True, eq=False)
class FoldSetup:
    """Training-only HRFs, masks, and PCs plus held-out terms for one fold.

    ``scored`` is the scoring mask minus held-out targets with numerically
    zero energy (``zero_target``); both are fixed for every count.
    """

    validation_run: int
    training_runs: tuple[int, ...]
    run_labels: tuple[str, ...]
    selection: HrfSelectionResult
    pool_statistic: np.ndarray
    masks: PoolMasks
    components: tuple[RunComponents, ...]
    scored: np.ndarray
    zero_target: np.ndarray
    target_energy: np.ndarray
    n_regressors: int
    groups: tuple[_HrfGroup, ...] = field(repr=False)

    def __post_init__(self):
        own_fields(self, ("scored", "zero_target"), dtype=bool)
        own_fields(self, ("target_energy", "pool_statistic"))
        own_tuples(self, ("training_runs", "run_labels", "components", "groups"))


def _training_runs(data, validation_run):
    if not is_integer(validation_run) or not 0 <= validation_run < data.n_runs:
        raise ValueError("validation_run must be a valid run index")
    train = tuple(r for r in range(data.n_runs) if r != validation_run)
    if len(train) < 2:
        raise ValueError("each fold needs at least two training runs")
    return train


def _run_terms(run, hrf_id, y, label):
    ok, reason = run.eligible(hrf_id)
    if not ok:
        raise ValueError(f"{label}: HRF {hrf_id} task design is invalid: {reason}")
    block = run.block(hrf_id)
    raw = run.task_design(hrf_id)[list(run.task_model.regressor_names)].to_numpy()
    projected = y - run.q @ (run.q.T @ y)
    projected -= block.qp @ (block.qp.T @ projected)
    # Numerical zero only, as in HRF selection: no weak-signal threshold.
    tolerance = np.linalg.norm(y, axis=0) * len(y) * _EPS
    zero = np.linalg.norm(projected, axis=0) <= tolerance
    rank = run.q.shape[1] + block.qp.shape[1]
    return _Terms(block.x, raw, projected, block.qp, rank, zero)


def _hrf_group(data, runs, split, hrf_id, features):
    train, held_out, labels = split
    terms = tuple(
        _run_terms(
            runs[r],
            hrf_id,
            data.signals[r][:, features],
            f"training run '{labels[r]}'",
        )
        for r in train
    )
    y = data.signals[held_out][:, features]
    label = f"held-out run '{labels[held_out]}'"
    validation = _run_terms(runs[held_out], hrf_id, y, label)
    return _HrfGroup(hrf_id, features, terms, validation)


def _hrf_groups(data, runs, split, selection, scoring):
    ids = selection.hrf_indices
    return tuple(
        _hrf_group(data, runs, split, int(h), np.flatnonzero(scoring & (ids == h)))
        for h in np.unique(ids[scoring])
    )


def _target_summary(groups, n_features):
    zero = np.zeros(n_features, dtype=bool)
    energy = np.full(n_features, np.nan)
    for group in groups:
        zero[group.features] = group.validation.zero
        energy[group.features] = np.sum(group.validation.y**2, axis=0)
    return zero, energy


def _training_selection(training, train, labels, library, task_model):
    return select_hrfs(
        training,
        library=library,
        task_model=task_model,
        run_labels=[labels[r] for r in train],
    )


def _checked_masks(selection, statistic, brain_mask, threshold, held_out):
    masks = pool_masks(selection, brain_mask, threshold, statistic=statistic)
    if masks.scoring_size == 0:
        raise ValueError(
            f"{_NO_TASK_SIGNAL}: the fold holding out run '{held_out}' "
            f"has an empty scoring mask at threshold {masks.threshold}"
        )
    return masks


def prepare_fold(
    data,
    validation_run,
    *,
    brain_mask,
    task_model,
    library,
    threshold,
    run_labels=None,
) -> FoldSetup:
    """Select HRFs, masks, and PCs from training runs; cache projected terms.

    Messages name runs by ``run_labels`` (default ``run-01``, ...).
    """
    labels = run_labels_for(data, run_labels)
    train = _training_runs(data, validation_run)
    held_out = labels[validation_run]
    training = subset_runs(data, train)
    selection = _training_selection(training, train, labels, library, task_model)
    statistic = pool_statistic(training, selection)
    masks = _checked_masks(selection, statistic, brain_mask, threshold, held_out)
    components = analysis_components(training, masks.pool)
    runs = prepare_runs(data, library, task_model)
    split = (train, validation_run, labels)
    groups = _hrf_groups(data, runs, split, selection, masks.scoring)
    zero, energy = _target_summary(groups, data.n_features)
    scored = masks.scoring & ~zero
    if not scored.any():
        raise ValueError(
            f"{_NO_TASK_SIGNAL}: held-out run '{held_out}' has no nonzero "
            "target in the scoring mask"
        )
    return FoldSetup(
        validation_run=validation_run,
        training_runs=train,
        run_labels=labels,
        selection=selection,
        pool_statistic=statistic,
        masks=masks,
        components=components,
        scored=scored,
        zero_target=zero,
        target_energy=energy,
        n_regressors=len(task_model.regressor_names),
        groups=groups,
    )


# ---- scoring one count -------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, eq=False)
class CountScore:
    """Held-out R² of one count on one fold; NaN everywhere when unavailable."""

    count: int
    reason: str
    mean_r2: float
    feature_r2: np.ndarray
    coefficients: np.ndarray
    target_energy: np.ndarray

    def __post_init__(self):
        own_fields(self, ("feature_r2", "coefficients", "target_energy"))


def _component_reason(fold, count):
    for run, comps in zip(fold.training_runs, fold.components, strict=True):
        reason = comps.unavailable_reason(count)
        if reason:
            return f"training run '{fold.run_labels[run]}': {reason}"
    return ""


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


def _group_coefficients(group, prefixes, run_labels):
    """Shared task coefficients for one HRF group, or a reason they do not exist."""
    a_sum, b_sum = 0.0, 0.0
    for terms, pcs, run in zip(group.train, prefixes, run_labels, strict=True):
        try:
            a, b = _training_statistics(terms, pcs)
        except ValueError as error:
            return None, f"training run '{run}', HRF {group.hrf_id}: {error}"
        a_sum, b_sum = a_sum + a, b_sum + b
    amplitude, ok = pooled_amplitude(a_sum[None], b_sum[None])
    if not ok[0]:
        return None, f"HRF {group.hrf_id}: pooled task design is singular"
    return amplitude[0], ""


def _coefficients(fold, count):
    prefixes = [c.prefix(count) for c in fold.components]
    labels = [fold.run_labels[r] for r in fold.training_runs]
    beta = np.full((fold.n_regressors, len(fold.scored)), np.nan)
    for group in fold.groups:
        values, reason = _group_coefficients(group, prefixes, labels)
        if reason:
            return None, reason
        beta[:, group.features] = values
    return beta, ""


def _heldout_r2(fold, beta):
    r2 = np.full(len(fold.scored), np.nan)
    for group in fold.groups:
        held = group.validation
        sse = np.sum((held.y - held.x @ beta[:, group.features]) ** 2, axis=0)
        r2[group.features] = 1 - sse / np.sum(held.y**2, axis=0)
    r2[~fold.scored] = np.nan
    return r2


def _count_score(fold, count, reason, beta=None):
    n = len(fold.scored)
    if beta is None:
        beta = np.full((fold.n_regressors, n), np.nan)
        r2, mean = np.full(n, np.nan), np.nan
    else:
        r2 = _heldout_r2(fold, beta)
        mean = float(np.mean(r2[fold.scored]))
    return CountScore(
        count=count,
        reason=reason,
        mean_r2=mean,
        feature_r2=r2,
        coefficients=beta,
        target_energy=fold.target_energy,
    )


def score_count(fold: FoldSetup, count: int) -> CountScore:
    """Fit shared coefficients with ``count`` PCs and score the fixed target.

    A positive count is unavailable when any training run cannot supply that
    many unique PCs or any projected task design loses rank or residual
    degrees of freedom. An invalid zero count raises instead.
    """
    reason = _component_reason(fold, count)
    beta = None
    if not reason:
        beta, reason = _coefficients(fold, count)
    if reason and count == 0:
        raise ValueError(f"zero-component task design is invalid: {reason}")
    return _count_score(fold, int(count), reason, None if reason else beta)


# ---- aggregation and choice -------------------------------------------------------


def fold_score_table(folds, counts) -> pd.DataFrame:
    """One row per fold and count: eligibility, reason, and fold-mean R²."""
    rows = []
    for fold in folds:
        for count in counts:
            score = score_count(fold, count)
            rows.append(
                dict(
                    validation_run=fold.validation_run,
                    validation_label=fold.run_labels[fold.validation_run],
                    count=count,
                    eligible=not score.reason,
                    reason=score.reason,
                    mean_r2=score.mean_r2,
                    n_scored=int(fold.scored.sum()),
                    n_zero_target=int(fold.zero_target.sum()),
                )
            )
    return pd.DataFrame(rows)


def _aggregate_row(rows, count):
    eligible = bool(rows["eligible"].all())
    failed = rows[~rows["eligible"]]
    reason = "; ".join(
        f"fold holding out run '{v}': {r}"
        for v, r in zip(failed["validation_label"], failed["reason"])
    )
    mean = float(rows["mean_r2"].mean()) if eligible else np.nan
    return dict(count=count, eligible=eligible, mean_r2=mean, reason=reason)


def aggregate_scores(fold_scores, counts) -> pd.DataFrame:
    """Equal-weight mean of fold means; a count must be eligible in every fold."""
    return pd.DataFrame(
        [
            _aggregate_row(fold_scores[fold_scores["count"] == count], count)
            for count in counts
        ]
    )


def choose_count(counts, scores, tolerance) -> int:
    """Smallest count within ``tolerance`` of the best finite aggregate score.

    NaN marks an unavailable count. A roundoff slack of 64 eps (as in HRF
    choice) keeps numerically tied scores tied, so ties select fewer PCs.
    """
    counts, scores = tuple(counts), np.asarray(scores, dtype=float)
    if len(scores) != len(counts):
        raise ValueError("choose_count needs one score per count")
    finite = np.isfinite(scores)
    if 0 not in counts or not finite[counts.index(0)]:
        raise ValueError("count zero must be a scored candidate")
    best = float(scores[finite].max())
    floor = best - tolerance - 64 * _EPS * max(1.0, abs(best))
    return min(c for c, s, ok in zip(counts, scores, finite) if ok and s >= floor)


@dataclass(frozen=True, kw_only=True, eq=False)
class CountSelection:
    """Chosen count, the fold setups, and per-fold and aggregate score tables."""

    n_components: int
    counts: tuple[int, ...]
    folds: tuple[FoldSetup, ...]
    fold_scores: pd.DataFrame
    scores: pd.DataFrame

    def __post_init__(self):
        own_tuples(self, ("counts", "folds"))
        rebind(
            self,
            fold_scores=_owned_table(self.fold_scores),
            scores=_owned_table(self.scores),
        )


def select_component_count(
    data,
    *,
    brain_mask,
    task_model,
    library,
    counts,
    threshold,
    tolerance,
    run_labels=None,
) -> CountSelection:
    """Leave-one-run-out PC-count choice; requires at least three runs."""
    counts, tolerance = validate_counts(counts), validate_tolerance(tolerance)
    threshold = validate_threshold(threshold)
    if data.n_runs < 3:
        raise ValueError("component-count selection requires at least three runs")
    brain_mask = validate_feature_mask(brain_mask, data.n_features)
    library = default_hrf_library() if library is None else library
    labels = run_labels_for(data, run_labels)
    check_run_designs(data, task_model, labels)
    folds = tuple(
        prepare_fold(
            data,
            v,
            brain_mask=brain_mask,
            task_model=task_model,
            library=library,
            threshold=threshold,
            run_labels=labels,
        )
        for v in range(data.n_runs)
    )
    fold_scores = fold_score_table(folds, counts)
    scores = aggregate_scores(fold_scores, counts)
    return CountSelection(
        n_components=choose_count(counts, scores["mean_r2"], tolerance),
        counts=counts,
        folds=folds,
        fold_scores=fold_scores,
        scores=scores,
    )
