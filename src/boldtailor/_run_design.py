"""Data-independent run designs, cached process-wide and shared by every estimator."""

from collections import namedtuple
from functools import lru_cache
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from boldtailor._hrf_design import hrf_model
from boldtailor._constants import MIN_ONSET, OVERSAMPLING
from boldtailor._single_trial_design import compile_trial_run
from boldtailor._design_inputs import (
    nuisance_matrix,
    validate_events,
    validate_onsets,
)
from boldtailor._single_trial_fit import project_trial_design
from boldtailor._task_design import expand_events, task_columns
from boldtailor.model import TaskModel

_Block = namedtuple("_Block", ["x", "qp", "a", "raw"])


def _sealed(array):
    """The array itself, marked read-only: cached designs are shared process-wide."""
    array.setflags(write=False)
    return array


def _sealed_block(x, qp, a, raw):
    return _Block(_sealed(x), _sealed(qp), _sealed(a), _sealed(raw))


def column_basis(columns):
    u, s, _ = np.linalg.svd(columns, full_matrices=False)
    return u[:, s > s[0] * max(columns.shape) * np.finfo(float).eps]


def _profiled_basis(p, q):
    """Orthonormal basis of profiled columns after confound projection."""
    if p.shape[1] == 0:
        return np.zeros((len(q), 0))
    pr = p - q @ (q.T @ p)
    tolerance = np.linalg.norm(p, axis=0) * len(p) * np.finfo(float).eps
    if np.any(np.linalg.norm(pr, axis=0) <= tolerance):
        raise ValueError("profiled column has no support outside nuisance span")
    basis = column_basis(pr)
    if basis.shape[1] != p.shape[1]:
        raise ValueError("profiled columns are rank deficient")
    return basis


def check_task_rank(xr, x, dof):
    scale = np.linalg.norm(xr, axis=0)
    tolerance = np.linalg.norm(x, ord=2) * max(x.shape) * np.finfo(float).eps
    if np.any(scale <= tolerance):
        raise ValueError("task column has no support outside nuisance span")
    if column_basis(xr / scale).shape[1] != x.shape[1]:
        raise ValueError("task columns are rank deficient after nuisance projection")
    if dof - x.shape[1] <= 0:
        raise ValueError("task model requires positive residual degrees of freedom")


class RunDesign:
    """Cached timing, modulator amplitudes, and confounds: never retains BOLD."""

    def __init__(self, expanded, times, nuisance, library, task_model, fingerprint):
        self.expanded = expanded
        task_rows = expanded.trial_type == "task"
        self.events = expanded.loc[task_rows, ["onset", "duration"]].reset_index(
            drop=True
        )
        self.times, self.nuisance, self.library = times, nuisance, library
        self.task_model, self.fingerprint = task_model, fingerprint
        self.q = _sealed(column_basis(nuisance))
        self.k = len(task_model.regressor_names)
        self._eligibility, self._trial_eligibility = {}, {}
        self._blocks, self._trial_matrices = {}, {}

    def task_design(self, candidate_id):
        """Nilearn task and profiled columns for one candidate on this run."""
        return task_columns(
            self.expanded,
            self.times,
            hrf_model(self.library.candidates[candidate_id]),
            min_onset=MIN_ONSET,
            oversampling=OVERSAMPLING,
        )

    def _build(self, candidate_id):
        columns = self.task_design(candidate_id)
        x = columns[list(self.task_model.regressor_names)].to_numpy()
        profiled = [c for c in self.task_model.profiled_names if c in columns]
        qp = _profiled_basis(columns[profiled].to_numpy(), self.q)
        q = np.column_stack([self.q, qp])
        xr = x - q @ (q.T @ x)
        check_task_rank(xr, x, len(self.times) - q.shape[1])
        return _sealed_block(xr, qp, xr.T @ xr, x)

    def block(self, candidate_id):
        """Projected and raw task columns, profiled basis, A; zeros if ineligible."""
        if candidate_id not in self._blocks:
            try:
                self._blocks[candidate_id] = self._build(candidate_id)
                self._eligibility[candidate_id] = (True, "")
            except ValueError as error:
                self._eligibility[candidate_id] = (False, str(error))
                self._blocks[candidate_id] = self._empty_block()
        return self._blocks[candidate_id]

    def _empty_block(self):
        empty = np.zeros((len(self.times), self.k))
        return _sealed_block(
            empty, np.zeros((len(self.times), 0)), np.zeros((self.k, self.k)), empty
        )

    def eligible(self, candidate_id):
        self.block(candidate_id)
        return self._eligibility[candidate_id]

    def trial_matrix(self, candidate_id):
        """Read-only single-trial columns for one candidate, built once per design."""
        if candidate_id not in self._trial_matrices:
            self._trial_matrices[candidate_id] = self._build_trial_matrix(candidate_id)
        return self._trial_matrices[candidate_id]

    def _build_trial_matrix(self, candidate_id):
        confounds = pd.DataFrame(self.nuisance[:, :-1])
        confounds.columns = [f"n{i}" for i in range(confounds.shape[1])]
        x, _, _ = compile_trial_run(
            self.events,
            self.times,
            confounds,
            "design",
            hrf=self.library.candidates[candidate_id],
        )
        values = x.to_numpy()
        values.setflags(write=False)
        return values

    def trial_eligible(self, candidate_id):
        """Estimability of the single-trial design; not used by selection."""
        if candidate_id not in self._trial_eligibility:
            try:
                project_trial_design(self.trial_matrix(candidate_id), self.nuisance)
                self._trial_eligibility[candidate_id] = (True, "")
            except ValueError as error:
                self._trial_eligibility[candidate_id] = (False, str(error))
        return self._trial_eligibility[candidate_id]


@lru_cache(maxsize=32)
def _cached_design(
    expanded_bytes,
    names_bytes,
    times_bytes,
    nuisance_bytes,
    n_columns,
    library,
    task_model,
):
    values = np.frombuffer(expanded_bytes, dtype="<f8").reshape(-1, 3)
    expanded = pd.DataFrame(values, columns=["onset", "duration", "modulation"])
    expanded = expanded.assign(trial_type=json.loads(names_bytes))
    times = np.frombuffer(times_bytes, dtype="<f8")
    nuisance = np.frombuffer(nuisance_bytes, dtype="<f8").reshape(len(times), n_columns)
    digest = sha256(
        expanded_bytes
        + names_bytes
        + times_bytes
        + nuisance_bytes
        + str(n_columns).encode()
        + library.fingerprint.encode()
        + task_model.fingerprint.encode()
    ).hexdigest()
    return RunDesign(expanded, times, nuisance, library, task_model, digest)


def _run_inputs(events, times, confounds, task_model, r):
    validate_events(events, times, f"run-{r}")
    validate_onsets(events["onset"], times)
    expanded = expand_events(events, task_model, r)
    return expanded, nuisance_matrix(confounds, len(times))


def _labeled_inputs(events, times, confounds, task_model, label):
    """As ``_run_inputs``, but every message leads with ``run '<label>'``."""
    try:
        validate_events(events, times, label)
        validate_onsets(events["onset"], times)
        nuisance = nuisance_matrix(confounds, len(times))
    except ValueError as error:
        raise ValueError(f"run '{label}': {error}") from error
    return expand_events(events, task_model, f"'{label}'"), nuisance


def prepare_runs(data, library, task_model=TaskModel(), labels=None):
    """Cached run designs; with ``labels``, design errors name runs by label."""
    runs = []
    for r, (events, times, confounds) in enumerate(
        zip(data.events, data.frame_times, data.confounds, strict=True)
    ):
        times = np.asarray(times, dtype=float)
        if labels is None:
            expanded, nuisance = _run_inputs(events, times, confounds, task_model, r)
        else:
            expanded, nuisance = _labeled_inputs(
                events, times, confounds, task_model, labels[r]
            )
        payloads = (
            np.asarray(
                expanded[["onset", "duration", "modulation"]], dtype="<f8"
            ).tobytes(),
            json.dumps(list(expanded.trial_type)).encode(),
            np.asarray(times, dtype="<f8").tobytes(),
            np.asarray(nuisance, dtype="<f8").tobytes(),
        )
        runs.append(_cached_design(*payloads, nuisance.shape[1], library, task_model))
    return tuple(runs)
