"""Per-feature OLS F-test significance gate on the pcstop PC count.

This gate is a Boldtailor addition, not part of GLMsingle. GLMsingle's
pcstop rule is relative and has no absolute floor, so chance gains on
independent noise can choose a positive count. After pcstop chooses ``k*``,
every feature pcstop scored is fit in-sample on all runs by OLS (no
prewhitening) with the frozen HRF under two nested models: reduced (task
regressors shared across runs plus each run's baseline confounds,
intercept, and missing-value indicators) and full (reduced plus each run's
first ``k*`` PCs as run-specific columns). Per feature,
``F = ((SSE_r - SSE_f) / df1) / (SSE_f / df2)`` with ``df1 = rank(full) -
rank(reduced)`` and ``df2 = N - rank(full)``, using the existing rank
tolerances. With ``m`` of ``n`` tested features at ``p < alpha``, ``k*`` is
kept only if the one-sided binomial test of ``m`` against ``alpha`` gives
``p < binomial_alpha``; otherwise the count is 0. Features whose stacked
design is rank deficient, has ``df2 <= 0``, or gains no PC columns
(``df1 <= 0``), and scoring features with a numerically zero target in any
run (which pcstop does not score), are excluded from ``n``.

The F-test asks whether the PCs explain variance in the scoring features,
not whether removing them improves task prediction. PCs of independent,
autocorrelated noise span its low-frequency directions, so they can pass:
in an illustrative probe (20k features, 12 runs, no shared noise; not a
calibration) white noise was rejected (about 5% of features at p < 0.05)
but independent AR(1) noise with coefficient 0.5 was kept (97.5-100% of
features at p < 0.05, binomial p near 0). OLS F-tests are anti-conservative
under autocorrelated noise, and the binomial test treats features as
independent, which is optimistic for spatially correlated features; both
are deliberate, lenient choices. With few tested features the binomial
decision is coarse (with n = 6, one feature can flip it). The gate reuses
the frozen-HRF scoring terms of :mod:`boldtailor._denoising_cv` (whose
GLMsingle attribution applies to that module).
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from boldtailor._denoising_cv import ScoringSetup, _extra_basis
from boldtailor._hrf_cv import _check_task_rank, pooled_amplitude
from boldtailor._scalars import is_real
from boldtailor.denoising_results import SignificanceGate

KEPT, REJECTED = "kept", "rejected"
SKIPPED, DISABLED = "skipped_zero_count", "disabled"
ZERO_TARGET = "numerically zero target in at least one run (not scored)"


# ---- settings ----------------------------------------------------------------------


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


# ---- nested OLS fits per HRF group -------------------------------------------------


@dataclass(frozen=True)
class _Fit:
    sse: np.ndarray
    rank: int


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
    a, b = x.T @ x, sum(p[0].T @ p[1] for p in parts)
    amplitude, ok = pooled_amplitude(a[None], b[None])
    if not ok[0]:
        return None, "pooled task design is singular"
    sse = sum(np.sum((y - xr @ amplitude[0]) ** 2, axis=0) for xr, y in parts)
    return _Fit(sse, nuisance + x.shape[1]), ""


@dataclass(frozen=True)
class _GroupTest:
    features: np.ndarray
    f: np.ndarray | None = None
    df1: int = 0
    df2: int = 0
    reason: str = ""


def _group_test(setup, group, count) -> _GroupTest:
    pcs = [c.prefix(count) for c in setup.components]
    bases = [_extra_basis(p, t.profiled) for p, t in zip(pcs, group.runs)]
    empty = [np.zeros((len(t.y), 0)) for t in group.runs]
    (reduced, reason), (full, full_reason) = (
        _stacked_fit(group, empty),
        _stacked_fit(group, bases),
    )
    if reason or full_reason:
        return _GroupTest(group.features, reason=reason or full_reason)
    df1 = full.rank - reduced.rank
    if df1 <= 0:
        return _GroupTest(group.features, reason="PCs add no columns to the model")
    df2 = sum(len(t.y) for t in group.runs) - full.rank
    gain = np.maximum(reduced.sse - full.sse, 0.0)
    # Zero-target features (0 / 0) are excluded by the caller.
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (gain / df1) / (full.sse / df2)
    return _GroupTest(group.features, f, df1, df2)


# ---- assembling the decision -------------------------------------------------------


def _arrays(n_features):
    values = {key: np.full(n_features, np.nan) for key in ("f", "p", "df1", "df2")}
    values.update(
        tested=np.zeros(n_features, bool), excluded=np.zeros(n_features, bool)
    )
    return values


def _record(values, exclusions, hrf_id, test, zero):
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


def _feature_tests(setup, count):
    values, exclusions = _arrays(len(setup.scoring)), []
    for group in setup.groups:
        # The gate tests exactly the features pcstop scored.
        zero = ~setup.scored[group.features]
        test = _group_test(setup, group, count)
        _record(values, exclusions, group.hrf_id, test, zero)
    return values, exclusions


def _gate(settings, count, decision, values, exclusions=(), m=0, binomial_p=np.nan):
    enabled, alpha, binomial_alpha = settings
    return SignificanceGate(
        enabled=enabled,
        alpha=alpha,
        binomial_alpha=binomial_alpha,
        pcstop_count=int(count),
        n_components=int(count) if decision in (KEPT, DISABLED) else 0,
        decision=decision,
        f_statistic=values["f"],
        p_value=values["p"],
        df1=values["df1"],
        df2=values["df2"],
        tested=values["tested"],
        excluded=values["excluded"],
        exclusions=tuple(exclusions),
        m=int(m),
        n=int(values["tested"].sum()),
        binomial_p=float(binomial_p),
    )


def apply_gate(
    setup: ScoringSetup, count: int, *, enabled, alpha, binomial_alpha
) -> SignificanceGate:
    """Keep the pcstop ``count`` only if the PCs pass the F-test gate.

    ``count`` must be available in every run (as a pcstop choice is).
    With the gate disabled the count is returned untested; a zero count
    skips the gate. With no testable feature the count is rejected.
    """
    settings = validate_gate(enabled, alpha, binomial_alpha)
    if not enabled or count == 0:
        decision = DISABLED if not enabled else SKIPPED
        return _gate(settings, count, decision, _arrays(len(setup.scoring)))
    values, exclusions = _feature_tests(setup, count)
    tested = values["tested"]
    m, n = int(np.sum(values["p"][tested] < alpha)), int(tested.sum())
    if n == 0:
        return _gate(settings, count, REJECTED, values, exclusions)
    p = stats.binomtest(m, n, alpha, alternative="greater").pvalue
    decision = KEPT if p < binomial_alpha else REJECTED
    return _gate(settings, count, decision, values, exclusions, m, p)
