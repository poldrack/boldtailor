"""Identify library HRFs with the package's ``select_hrfs``.

Only the mapping from the experiment's runs to package inputs lives here.
Valid trials form the task model; missing and incorrect trials stay in the
nuisance in a fixed canonical basis, as in ``experiments.hrf_simulations``.
"""

import numpy as np

from boldtailor.data import from_arrays
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.hrf_selection import select_hrfs
from boldtailor.model import Modulator, TaskModel
from boldtailor.provenance import identity_activity
from experiments.hrf_estimation.design import build_design, partition_events
from experiments.hrf_estimation.estimators import HrfEstimate, validate_features

RT_MODES = ("amplitude", "duration", "none")
JUNK_COLUMNS = ("selection_missing", "selection_incorrect")


def task_model(rt_mode):
    """Two conditions, plus RT as an amplitude modulator in amplitude mode.

    Duration mode instead gives every valid trial its RT as the event duration.
    """
    if rt_mode not in RT_MODES:
        raise ValueError("rt_mode must be amplitude, duration, or none")
    modulators = [Modulator("trial_type", kind="categorical", levels=(0, 1))]
    if rt_mode == "amplitude":
        modulators.append(Modulator("response_time"))
    return TaskModel(tuple(modulators))


def _run_inputs(run, library, rt_mode):
    events = partition_events(run.events)
    valid = events.loc[events.category == "valid"].copy()
    if rt_mode == "duration":
        valid["duration"] = valid.response_time
    junk = build_design(run, library.candidates[0], "none").junk
    nuisance = run.confounds.drop(columns=["intercept"], errors="ignore").copy()
    for name, column in zip(JUNK_COLUMNS, junk.T, strict=True):
        nuisance[name] = column
    return valid, nuisance


def selection_data(runs, library, rt_mode="amplitude"):
    """The package data and task model for one set of training runs."""
    model = task_model(rt_mode)
    validate_features(runs, runs[0].feature_ids, runs[0].brain_axis)
    inputs = [_run_inputs(run, library, rt_mode) for run in runs]
    data = from_arrays(
        [run.signals for run in runs],
        [events for events, _ in inputs],
        frame_times=[run.frame_times for run in runs],
        confounds=[nuisance for _, nuisance in inputs],
    )
    return data, model


def select(runs, library, rt_mode, method, rho=None):
    """The package's selection result for these runs."""
    data, model = selection_data(runs, library, rt_mode)
    return select_hrfs(
        data,
        library=library,
        task_model=model,
        method=method,
        reml_options=None if method == "cv" else RemlHrfOptions(rho=rho),
        run_labels=[run.run_id for run in runs],
    )


def estimate(runs, library, result, status, diagnostics, hrf_indices=None):
    """Wrap a package selection in the experiment's common result contract.

    ``hrf_indices`` overrides the package's winners, for a documented fallback.
    """
    ids = np.asarray(result.hrf_indices if hrf_indices is None else hrf_indices)
    kernels = tuple(library.candidates[i] if i >= 0 else None for i in ids)
    return HrfEstimate(
        runs[0].feature_ids.copy(),
        kernels,
        tuple(status),
        tuple(run.run_id for run in runs),
        dict(
            diagnostics,
            library_fingerprint=library.fingerprint,
            task_model=result.task_model.to_dict(),
            package_selection=identity_activity(result.provenance),
        ),
        runs[0].brain_axis,
    )
