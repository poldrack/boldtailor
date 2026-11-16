"""Fixed-reference mean models and streamed HRF random-effect designs."""

import numpy as np

from boldtailor._hrf_cv import prepare_runs
from boldtailor._single_trial_design import compile_trial_run
from boldtailor.reml import RemlGroup


def prepare_reml_groups(data, library, task_model, options, labels):
    """Build one fixed mean/nuisance span per run, shared by all candidates."""
    options.validate(data.n_features, library)
    runs = prepare_runs(data, library, task_model, labels=labels)
    return tuple(
        RemlGroup(
            y,
            np.column_stack(
                [run.nuisance, _reference_task_columns(run, options.reference_hrf_id)]
            ),
        )
        for y, run in zip(data.signals, runs, strict=True)
    )


def iter_trial_designs(data, library, candidate_ids, labels):
    """Yield one candidate at a time without imposing trial OLS rank rules."""
    inputs = tuple(
        zip(data.events, data.frame_times, data.confounds, labels, strict=True)
    )
    for cid in candidate_ids:
        yield tuple(
            compile_trial_run(
                events, times, confounds, label, hrf=library.candidates[cid]
            )[0].to_numpy()
            for events, times, confounds, label in inputs
        )


def _reference_task_columns(run, reference_id):
    """Aggregate unit-peak trial columns with the shared task-model amplitudes."""
    expanded = run.expanded
    weights = np.column_stack(
        [
            expanded.loc[expanded.trial_type == name, "modulation"].to_numpy()
            for name in dict.fromkeys(expanded.trial_type)
        ]
    )
    return run.trial_matrix(reference_id) @ weights
