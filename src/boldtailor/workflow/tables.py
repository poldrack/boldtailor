"""Summary tables and boundary diagnostics for workflow stages."""

import numpy as np
import pandas as pd

from boldtailor.ridge_results import FractionSelection


def _predictor_table(runs, predictors):
    tables = []
    for r, (run, predictor) in enumerate(zip(runs, predictors, strict=True)):
        table = predictor.reset_index(drop=True).copy()
        table.insert(0, "event_index", np.arange(len(table)))
        table.insert(
            0, "trial_id", [f"{run.label}_trial-{j + 1:04d}" for j in range(len(table))]
        )
        table.insert(0, "run_label", run.label)
        table.insert(0, "run_index", r)
        table["usable"] = np.isfinite(predictor.to_numpy()).all(axis=1)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _tuned(beta_models):
    return [m for m in beta_models.values() if m.tuning]


def tuning_rows(mode, scope, tuned):
    selected, scores = tuned["selection"], tuned["scores"]
    percentile = tuned["summary_percentile"]
    mask = selected.scoring_mask
    summaries = (
        np.percentile(scores.cv_r2[:, mask], percentile, axis=1)
        if mask.any()
        else np.full(len(scores.grid), np.nan)
    )
    return [
        dict(
            mode=mode,
            scope=scope,
            fraction=f,
            summary_r2=float(r),
            percentile=percentile,
            grayordinates=int(mask.sum()),
            selected_grayordinates=int(np.count_nonzero(selected.ridge_fraction == f)),
        )
        for f, r in zip(scores.grid, summaries, strict=True)
    ]


def _alpha_rows(mode, scope, tuned):
    selected = tuned["selection"]
    return [
        dict(
            mode=mode,
            scope=scope,
            alpha=alpha,
            objective=float(value),
            selected=alpha == selected.ridge_alpha,
            percentile=selected.percentile,
            grayordinates=int(selected.scoring_mask.sum()),
        )
        for alpha, value in zip(selected.alphas, selected.objective_scores, strict=True)
    ]


def tuning_table(beta_models):
    """One row per candidate, scope, and tuned model (mode is the HRF model)."""
    rows = []
    for model in _tuned(beta_models):
        mode = model.hrf.capitalize()
        for scope, tuned in model.tuning.items():
            fractional = isinstance(tuned["selection"], FractionSelection)
            rows.extend(
                (tuning_rows if fractional else _alpha_rows)(mode, scope, tuned)
            )
    return pd.DataFrame(rows)


def _boundary_fraction(selection):
    flags = np.asarray(selection.at_boundary, dtype=bool)
    if flags.ndim == 0:
        return float(flags)
    scored = flags[selection.scoring_mask]
    return float(scored.mean()) if scored.size else float("nan")


def ridge_boundary_summary(beta_models):
    """Fraction of scored grayordinates whose ridge winner is a grid endpoint."""
    return [
        dict(
            mode=model.hrf.capitalize(),
            scope=scope,
            fraction=_boundary_fraction(tuned["selection"]),
        )
        for model in _tuned(beta_models)
        for scope, tuned in model.tuning.items()
    ]


def _scope_selection(bundle, scope):
    return bundle[scope] if scope == "all" else bundle[scope].training_selection


def _pooled_bound_rows(picked, scope):
    counts = [int((p.hrf_indices > 0).sum()) for p in picked]
    tables = [p.parameter_bound_table() for p in picked]
    total = sum(counts)
    rows = []
    for i, row in tables[0].iterrows():
        flagged = sum(t.fraction_flagged[i] * n for t, n in zip(tables, counts) if n)
        fraction = flagged / total if total else float("nan")
        rows.append(
            dict(
                scope=scope,
                parameter=row.parameter,
                edge=row.edge,
                fraction_flagged=float(fraction),
                n_custom=total,
            )
        )
    return rows


def hrf_boundary_summary(selections):
    """``parameter_bound_table`` rows pooled over grayordinate blocks per scope."""
    if not selections:
        return None
    rows = []
    for scope in ("all", "odd", "even"):
        if not all(scope in b for b in selections.values()):
            continue
        picked = [_scope_selection(b, scope) for b in selections.values()]
        rows.extend(_pooled_bound_rows(picked, scope))
    return rows


def hrf_boundary_table(selections):
    """``hrf_boundary_summary`` as a DataFrame, or None without selections."""
    rows = hrf_boundary_summary(selections)
    return None if rows is None else pd.DataFrame(rows)
