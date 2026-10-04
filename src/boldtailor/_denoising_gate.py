"""Per-feature nested F-test significance gate on the pcstop PC count.

This gate is a Boldtailor addition, not part of GLMsingle. GLMsingle's
pcstop rule is relative and has no absolute floor, so chance gains on
independent noise can choose a positive count. After pcstop chooses ``k*``,
every feature pcstop scored is fit in-sample on all runs with the frozen HRF
under two nested models: reduced (task regressors shared across runs plus
each run's baseline confounds, intercept, and missing-value indicators) and
full (reduced plus each run's first ``k*`` PCs as run-specific columns). Per
feature, ``F = ((SSE_r - SSE_f) / df1) / (SSE_f / df2)`` with ``df1 =
rank(full) - rank(reduced)`` and ``df2 = N - rank(full)``, using the
existing rank tolerances. With ``m`` of ``n`` tested features at
``p < alpha``, ``k*`` is kept only if the one-sided binomial test of ``m``
against ``alpha`` gives ``p < binomial_alpha``; otherwise the count is 0.
Features whose stacked design is rank deficient, has ``df2 <= 0``, or gains
no PC columns (``df1 <= 0``), and scoring features with a numerically zero
target in any run (which pcstop does not score), are excluded from ``n``.

By default (``noise_model="ar1"``) the F-tests are AR(1) prewhitened,
following Nilearn's first-level ``noise_model="ar1"`` convention as far as
the pooled multi-run design allows. Per run and feature, the lag-1
coefficient is estimated from the full model's OLS residuals in that run by
Nilearn's Yule-Walker estimator and truncated to Nilearn's 1/100 bins (as
``nilearn.glm.first_level.run_glm`` does). That run's data and all of its
design columns (task, baseline, indicators, PCs) are whitened by
``nilearn.glm.ARModel`` (``x[t] - rho * x[t - 1]``, first scan unscaled),
and both nested models are refit on the stacked whitened runs with the same
degrees of freedom (whitening is invertible, so ranks are unchanged).
``nilearn.glm.first_level.first_level._yule_walker`` is private: there is no
public Nilearn path to the coefficients of residuals from a pooled
multi-run fit (``run_glm`` fits one design), so it is imported directly;
Nilearn is pinned below 0.15. ``noise_model="ols"`` skips the whitening.

The F-test asks whether the PCs explain variance in the scoring features
(variance explained), not whether removing them improves task prediction.
Without prewhitening, PCs of independent autocorrelated noise pass: in an
illustrative probe (20k features, 12 runs, no shared noise; not a
calibration) the OLS gate kept independent AR(1) noise with coefficient
0.5. Prewhitening addresses lag-1 autocorrelation only: AR(1) can
under-whiten higher-order autocorrelation. The gate is in-sample, the
binomial test treats features as independent, which is optimistic for
spatially correlated features, and with few tested features its decision is
coarse (with n = 6, one feature can flip it). The gate reuses the
frozen-HRF scoring terms of :mod:`boldtailor._denoising_cv` (whose GLMsingle
attribution applies to that module).
"""

from dataclasses import dataclass

import numpy as np
from nilearn.glm import ARModel
from nilearn.glm.first_level.first_level import _yule_walker
from scipy import stats

from boldtailor._denoising_cv import ScoringSetup, _extra_basis
from boldtailor._hrf_cv import _basis, _check_task_rank, pooled_amplitude
from boldtailor._scalars import is_real
from boldtailor.denoising_results import SignificanceGate

KEPT, REJECTED = "kept", "rejected"
SKIPPED, DISABLED = "skipped_zero_count", "disabled"
ZERO_TARGET = "numerically zero target in at least one run (not scored)"
NOISE_MODELS = ("ar1", "ols")
# Nilearn's run_glm default number of AR(1) bins.
AR1_BINS = 100


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


def validate_noise_model(value) -> str:
    """``gate_noise_model``: ``"ar1"`` (prewhitened) or ``"ols"``."""
    if not isinstance(value, str) or value not in NOISE_MODELS:
        raise ValueError(f"gate_noise_model must be one of {NOISE_MODELS}")
    return value


# ---- nested OLS fits per HRF group -------------------------------------------------


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
    a, b = x.T @ x, sum(p[0].T @ p[1] for p in parts)
    amplitude, ok = pooled_amplitude(a[None], b[None])
    if not ok[0]:
        return None, "pooled task design is singular"
    residuals = tuple(y - xr @ amplitude[0] for xr, y in parts)
    sse = sum(np.sum(e**2, axis=0) for e in residuals)
    return _Fit(sse, nuisance + x.shape[1], residuals), ""


# ---- AR(1) prewhitening -------------------------------------------------------------


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
    if reason or full_reason:
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


# ---- assembling the decision -------------------------------------------------------


def _arrays(n_runs, n_features):
    values = {key: np.full(n_features, np.nan) for key in ("f", "p", "df1", "df2")}
    values.update(
        tested=np.zeros(n_features, bool),
        excluded=np.zeros(n_features, bool),
        rho=np.full((n_runs, n_features), np.nan),
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
    if test.rho is not None:
        values["rho"][:, keep] = test.rho[:, ~zero]


def _feature_tests(setup, count, noise_model):
    values, exclusions = _arrays(setup.n_runs, len(setup.scoring)), []
    for group in setup.groups:
        # The gate tests exactly the features pcstop scored.
        zero = ~setup.scored[group.features]
        test = _group_test(setup, group, count, noise_model)
        _record(values, exclusions, group.hrf_id, test, zero)
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
