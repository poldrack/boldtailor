"""Small tables for the HTML report, built from results already in memory."""

import numpy as np
import pandas as pd

from boldtailor.workflow.inputs import run_summary
from boldtailor.workflow.outputs import (
    _all_selection_maps,
    _tuned,
    ridge_boundary_summary,
)


def median(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def activation_summary(activation):
    if not activation:
        return None
    rows = [dict(model=k, median_t=median(v["t"])) for k, v in activation.items()]
    return pd.DataFrame(rows)


def rt_summary(activation, beta_models):
    if activation is None:
        return None
    rows = [
        dict(model=k, scope=s, median_r=median(m.fit["rt"][s]))
        for k, m in beta_models.items()
        if m.fit["rt"] is not None
        for s in ("all", "odd", "even")
    ]
    return pd.DataFrame(rows) if rows else None


def library_head(library, rows=10):
    """The first rows of the HRF library parameter table."""
    return library.parameter_table.head(rows)


def selected_hrfs(selections, library, n_features, top=10):
    """The most often selected HRFs (all runs) with their peak times."""
    if not selections:
        return None
    ids = _all_selection_maps(selections, n_features)["all"][0]
    ids = ids[np.isfinite(ids)].astype(int)
    if not ids.size:
        return None
    counts = pd.Series(ids).value_counts().head(top)
    table = library.parameter_table.iloc[counts.index]
    return pd.DataFrame(
        dict(
            hrf_id=table.hrf_id.to_numpy(),
            selected_peak_time=table.peak_time.to_numpy(),
            grayordinates=counts.to_numpy(),
            share=counts.to_numpy() / ids.size,
        )
    )


def encoding_scores(beta_models):
    """Median outer (held-out half) encoding R² per tuned model and split."""
    rows = [
        dict(
            model=model.name,
            split=split,
            median_encoding_r2=median(outer["encoding_r2"]),
            grayordinates=int(np.isfinite(outer["encoding_r2"]).sum()),
        )
        for model in _tuned(beta_models)
        for split, outer in model.evaluation.items()
    ]
    return pd.DataFrame(rows) if rows else None


def ridge_boundary(beta_models):
    """Share of scored grayordinates whose ridge winner is a grid endpoint."""
    rows = ridge_boundary_summary(beta_models)
    if not rows:
        return None
    return pd.DataFrame(rows).rename(columns={"fraction": "boundary_fraction"})


def input_tables(runs, task_model, library):
    """Run summary, confound names and library head for the inputs section."""
    return dict(
        run_summary=run_summary(runs, task_model),
        confounds=list(runs[0].confounds.columns),
        library_table=library_head(library),
    )
