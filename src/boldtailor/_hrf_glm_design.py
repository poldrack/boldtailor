"""Compile conventional designs without changing semantic regressor names."""

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from boldtailor._hrf_design import hrf_model
from boldtailor._task_design import run_task_columns
from boldtailor._conventional import _preflight_contrasts, _validate_designs
from boldtailor.design import (
    CompiledDesign,
    compile_designs,
    compile_nuisance_designs,
    kernel_task_columns,
    _select_modeled_events,
    _validate_design_matrix,
)


def compile_group_designs(data, model, selection):
    nuisance = compile_nuisance_designs(data, model)
    events = data.events
    groups = {}
    ids = np.unique(selection.hrf_indices[selection.hrf_indices >= 0])
    if not len(ids):
        # No signal will be fitted, but malformed models must still fail.
        reference = compile_designs(data, replace(model, hrf_model="spm"))
        matrices = tuple(compiled.matrix for compiled in reference)
        _preflight_contrasts(matrices, model.contrasts)
        _validate_designs(matrices)
    for cid in ids:
        candidate = selection.library.candidates[cid]
        if model.task_model is not None:
            designs = tuple(
                _task_model_design(e, t, n, candidate, model, run)
                for run, (e, t, n) in enumerate(
                    zip(events, data.frame_times, nuisance, strict=True)
                )
            )
        else:
            designs = tuple(
                _custom_design(e, t, n, candidate, model, run)
                for run, (e, t, n) in enumerate(
                    zip(events, data.frame_times, nuisance, strict=True)
                )
            )
        groups.update({(run, int(cid)): design for run, design in enumerate(designs)})
    return groups, nuisance


def _task_model_design(events, times, nuisance, candidate, model, run):
    modeled, excluded, cutoff = _select_modeled_events(
        events, times, model.min_onset, run
    )
    task = run_task_columns(
        modeled,
        model.task_model,
        times,
        hrf_model(candidate),
        run=run,
        min_onset=model.min_onset,
        oversampling=model.oversampling,
    )
    matrix = pd.concat([task, nuisance.matrix], axis=1)
    _validate_design_matrix(matrix, run)
    return CompiledDesign(matrix, excluded, cutoff)


def _custom_design(events, times, nuisance, candidate, model, run):
    modeled, excluded, cutoff = _select_modeled_events(
        events, times, model.min_onset, run
    )
    task = kernel_task_columns(times, modeled, hrf_model(candidate), model, run)
    matrix = pd.concat([task, nuisance.matrix], axis=1)
    _validate_design_matrix(matrix, run)
    return CompiledDesign(matrix, excluded, cutoff)


def design_identity(groups, nuisance, frame_times):
    digest = sha256()
    for key, compiled in sorted(groups.items()):
        digest.update(json.dumps(key).encode())
        _hash_design(digest, compiled.matrix)
    for compiled, times in zip(nuisance, frame_times, strict=True):
        _hash_design(digest, compiled.matrix)
        digest.update(np.asarray(times, dtype="<f8").tobytes())
    return digest.hexdigest()


def _hash_design(digest, matrix):
    digest.update(json.dumps([matrix.shape, list(matrix.columns)]).encode())
    digest.update(np.asarray(matrix, dtype="<f8").tobytes())


def group_diagnostics(groups):
    output = {}
    for key, compiled in groups.items():
        matrix = compiled.matrix.to_numpy()
        rank = int(np.linalg.matrix_rank(matrix))
        output[key] = dict(
            excluded_event_count=compiled.excluded_event_count,
            min_onset_cutoff=compiled.min_onset_cutoff,
            design_rank=rank,
            residual_dof=matrix.shape[0] - rank,
        )
    return output
