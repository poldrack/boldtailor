"""Map simulation trials onto the package's supported identification models."""

from boldtailor.data import from_arrays
from boldtailor.model import Modulator, TaskModel
from experiments.hrf_estimation.design import partition_events
from experiments.hrf_simulations.design import CONVENTIONS, durations, task_design

IDENTIFICATION_MODELS = ("ConsDurNoRT", "RTDur", "ConsDurRTMod")


def selection_model(model):
    """Reference-coded conditions span the two original condition mean responses."""
    if model not in IDENTIFICATION_MODELS:
        raise ValueError(f"{model} is not supported by package HRF identification")
    mods = [Modulator("trial_type", kind="categorical", levels=(0, 1))]
    if model == "ConsDurRTMod":
        mods.append(Modulator("response_time"))
    return TaskModel(tuple(mods))


def _run_inputs(run, library, model, convention):
    events = partition_events(run.events)
    valid = events.loc[events.category == "valid"].copy()
    valid["duration"] = (
        valid.response_time if model == "RTDur" else durations(valid, convention)
    )
    # Junk remains in a fixed canonical basis for both selection methods.
    junk = task_design(run, library.candidates[0], "ConsDurNoRT", "stimulus").junk
    nuisance = run.confounds.drop(columns=["intercept"], errors="ignore").copy()
    nuisance["selection_missing"] = junk[:, 0]
    nuisance["selection_incorrect"] = junk[:, 1]
    return valid, nuisance


def selection_data(runs, library, model, convention, cache):
    """Cache timing/nuisance only, then attach the current training signals."""
    task_model = selection_model(model)
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown duration convention: {convention}")
    key = ("package_selection", library.fingerprint, model, convention)
    prepared = [
        cache.get(r, key, lambda r=r: _run_inputs(r, library, model, convention))
        for r in runs
    ]
    data = from_arrays(
        [r.signals for r in runs],
        [p[0] for p in prepared],
        frame_times=[r.frame_times for r in runs],
        confounds=[p[1] for p in prepared],
    )
    return data, task_model
