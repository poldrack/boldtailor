"""Feature experiments: denoising gate, task modulators, HRF choice, RT."""

import dataclasses

import numpy as np
import pandas as pd

from boldtailor.hrf_library import default_hrf_library, glmsingle_hrf_library
from boldtailor.model import TaskModel
from boldtailor.single_trial import fit_single_trials

from experiments.nsd_replication.inputs import analysis_data
from experiments.nsd_replication.ladder import fit_ladder, select_session_denoising

HRF_SKIPPED = "released HRF index maps not available"
GATE_COLUMNS = ("condition", "gate", "n_components", "pcstop_count", "decision")


# ------------------------------------------------------------ null control


def circular_shift(events, run_duration, rng, minimum=30.0):
    """Shift all onsets by one offset in [minimum, duration - minimum], wrapped."""
    offset = rng.uniform(minimum, run_duration - minimum)
    shifted = events.assign(onset=np.mod(events["onset"] + offset, run_duration))
    return shifted.sort_values("onset", kind="stable").reset_index(drop=True)


def _run_duration(frame_times):
    times = np.asarray(frame_times, dtype=float)
    return len(times) * float(np.median(np.diff(times)))


def _shifted_run(events, frame_times, rng):
    """Circularly shift within the run's time base; onsets wrapped past the
    last frame (no sampled response) move one period earlier, which is the
    same circular position."""
    start, duration = float(frame_times[0]), _run_duration(frame_times)
    local = events.assign(onset=events["onset"] - start)
    shifted = circular_shift(local, duration, rng)
    shifted["onset"] += start
    late = shifted["onset"] >= frame_times[-1]
    shifted.loc[late, "onset"] -= duration
    return shifted.sort_values("onset", kind="stable").reset_index(drop=True)


def null_session(session, seed):
    """Each run's events circularly shifted with seed ``[seed, run index]``."""
    events = tuple(
        _shifted_run(e, t, np.random.default_rng([seed, i]))
        for i, (e, t) in enumerate(zip(session.events, session.frame_times))
    )
    return dataclasses.replace(session, events=events)


# ------------------------------------------------------------ gate


def _gate_row(condition, gate, result):
    return dict(
        condition=condition,
        gate=gate,
        n_components=result.n_components,
        pcstop_count=result.pcstop_count,
        decision=result.significance_gate.decision,
    )


def gate_experiment(session, seed, *, library=None):
    """Denoising with the gate on and off, on real and null-shifted events."""
    library = default_hrf_library() if library is None else library
    sessions = {"real": session, "null": null_session(session, seed)}
    rows = [
        _gate_row(name, gate, select_session_denoising(s, s.task_model, library, gate))
        for name, s in sessions.items()
        for gate in (True, False)
    ]
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS)).assign(seed=seed)


# ------------------------------------------------------------ modulators


def _names(task_model):
    return list(task_model.regressor_names)


def _b4(session, task_model, level, **kwargs):
    denoising = select_session_denoising(session, task_model, default_hrf_library())
    fit = fit_ladder(
        session,
        levels=("b4",),
        task_model=task_model,
        ridge_task_model=session.task_model,
        denoising=denoising,
        **kwargs,
    )["b4"]
    record = dict(
        fit.record,
        level=level,
        task_model=_names(task_model),
        ridge_task_model=_names(session.task_model),
    )
    return dataclasses.replace(fit, record=record)


def modulator_experiment(session, **kwargs):
    """b4 with the task regressor alone versus the session's task model.

    The task model drives HRF selection and denoising scoring. Encoding ridge
    CV has no task-only objective (it needs a modulator), so both fits tune
    ridge fractions on the session's task model. ``kwargs`` (block_size,
    n_jobs) pass through to ``fit_ladder``.
    """
    return {
        "b4": _b4(session, session.task_model, "b4", **kwargs),
        "b4-taskonly": _b4(session, TaskModel(), "b4-taskonly", **kwargs),
    }


# ------------------------------------------------------------ HRF choice


def _valid_indices(indices):
    values = np.nan_to_num(np.asarray(indices, dtype=float), nan=0.0)
    return values.astype(int)


def _group_betas(session, columns, candidate):
    result = fit_single_trials(
        analysis_data(session, columns),
        hrf_model=candidate,
        run_labels=list(session.labels),
    )
    return np.vstack(result.run_betas)


def _glmsingle_choice(session, indices, n_trials):
    library, chosen = glmsingle_hrf_library(), _valid_indices(indices)
    betas = np.full((n_trials, len(chosen)), np.nan)
    for index in np.unique(chosen[chosen > 0]):
        columns = np.flatnonzero(chosen == index)
        betas[:, columns] = _group_betas(session, columns, library.candidates[index])
    return betas


def hrf_choice_experiment(session, released_hrf_indices, **kwargs):
    """OLS single-trial fits with boldtailor's b2-lib20 choice versus the
    released GLMsingle index (one-based; 0 or NaN means none)."""
    if released_hrf_indices is None:
        return {"skipped": HRF_SKIPPED}
    ours = fit_ladder(session, levels=("b2-lib20",), **kwargs)["b2-lib20"]
    return {
        "boldtailor": ours.betas,
        "glmsingle": _glmsingle_choice(session, released_hrf_indices, len(ours.betas)),
        "hrf_indices_boldtailor": ours.extras["hrf_indices"],
        "hrf_indices_glmsingle": np.asarray(released_hrf_indices),
    }


# ------------------------------------------------------------ RT


def rt_correlation(betas, events, roi):
    """Pearson r between trial RT and mean ROI beta over trials with finite RT."""
    rt = pd.concat(list(events), ignore_index=True)["response_time"].to_numpy(float)
    if len(rt) != len(betas):
        raise ValueError(f"betas rows ({len(betas)}) must match events ({len(rt)})")
    mean = np.asarray(betas, dtype=float)[:, np.asarray(roi, dtype=bool)].mean(axis=1)
    keep = np.isfinite(rt) & np.isfinite(mean)
    return float(np.corrcoef(mean[keep], rt[keep])[0, 1])


# ------------------------------------------------------------ one session


def _rt_table(fits, events, roi):
    rows = [(level, rt_correlation(f.betas, events, roi)) for level, f in fits.items()]
    return pd.DataFrame(rows, columns=["version", "r"])


def session_features(session, roi, seed, label, **kwargs):
    """All feature experiments for one session; tables carry ``session=label``.

    Released HRF index maps are not available, so the HRF-choice part records
    its skip reason.
    """
    fits = modulator_experiment(session, **kwargs)
    hrf = hrf_choice_experiment(session, None)
    tag = dict(session=label)
    return dict(
        fits=fits,
        gate=gate_experiment(session, seed).assign(**tag),
        rt=_rt_table(fits, session.events, roi).assign(**tag),
        hrf=pd.DataFrame([dict(tag, **hrf)]),
    )
