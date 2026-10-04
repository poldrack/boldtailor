"""Leave-one-run-out scoring of temporal-PC counts and GLMsingle's pcstop rule.

HRFs, the noise pool, its run-wise PCs, and the scoring features are
full-data inputs, frozen for every fold and count (as GLMsingle selects HRFs
and the pool once before GLMdenoise). For each held-out run and candidate
count, shared task coefficients are fit across the training runs after
projecting both the task design and BOLD off each training run's baseline
confounds, intercept, missing-value indicators (convolved with the frozen
HRF), and that run's leading PCs. The held-out run's BOLD never enters its
fold's fit, and its PCs never enter the fit or the target: the target is
projected off its baseline confounds, intercept, and indicators only, so the
target, denominator, and scored features are identical for every count.

Each scored feature's SSE and SST are pooled across folds,
``r2_f(k) = 1 - sum SSE / sum SST``, and the performance of a count is the
median of ``r2_f(k)`` over scored features (GLMsingle uses the median). The
count follows GLMsingle's ``select_noise_regressors`` with ``pcstop``.
Deviation, stated: GLMsingle scores counts by cross-validated single-trial
beta consistency across repeated conditions (and disables denoising without
repeats); repeats are not assumed here, so counts are scored by held-out
time-series prediction, close to the original GLMdenoise. Indicator
coefficients are profiled on held-out BOLD (conditional prediction), and the
scores are selection statistics, not independent performance estimates.

This module reimplements procedures from GLMsingle
(https://github.com/cvnlab/GLMsingle), Copyright (c) 2021, Kendrick Kay,
distributed under the BSD 3-Clause License; see
LICENSES/GLMsingle-BSD-3-Clause.txt for the copyright notice, conditions,
and disclaimer. It is an independent reimplementation, not a copy of
GLMsingle code. Follows GLMsingle's select_noise_regressors stopping rule
(pcstop) and the GLMdenoise cross-validated choice of the PC count.
Reference: Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr,
M.J., Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. eLife, 11, e77599.
https://doi.org/10.7554/eLife.77599
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from boldtailor._arrays import own_fields, own_tuples, rebind
from boldtailor._denoising_pool import RunComponents, validate_feature_mask
from boldtailor._hrf_cv import _check_task_rank, pooled_amplitude, prepare_runs
from boldtailor._scalars import is_integer, is_real
from boldtailor.data import _owned_table, run_labels_for

_EPS = np.finfo(float).eps
NO_TASK_SIGNAL = "no supported task signal for component-count selection"


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


# ---- scoring setup -------------------------------------------------------------


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
        own_fields(self, ("scoring", "scored", "zero_target"), dtype=bool)
        own_fields(self, ("hrf_indices",), dtype=np.int64)
        own_fields(self, ("target_energy",))
        own_tuples(self, ("run_labels", "components", "groups"))

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
    return _Terms(block.x, block.raw, projected, block.qp, rank, zero)


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


# ---- per-run statistics for one count ---------------------------------------------


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


# ---- one fold and one count -----------------------------------------------------------


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
        a_sum = sum(stats[g][0] for stats in train)
        b_sum = sum(stats[g][1] for stats in train)
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
        own_tuples(self, ("fold_reasons",))
        own_fields(self, ("coefficients", "fold_sse", "fold_sst", "feature_r2"))

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


# ---- tables and choice -----------------------------------------------------------------


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
        own_tuples(self, ("counts", "count_scores"))
        rebind(
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
