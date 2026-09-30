"""Nuisance-projected sufficient statistics and data-independent design caches."""

from collections import namedtuple
from functools import lru_cache
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from boldtailor._hrf_design import hrf_model
from boldtailor._single_trial_design import (
    _nuisance_matrix,
    _validate_events,
    compile_trial_run,
)
from boldtailor._single_trial_fit import _project_design
from boldtailor._task_design import expand_events, task_columns
from boldtailor.model import TaskModel

MIN_ONSET = -24.0
OVERSAMPLING = 50

_Block = namedtuple("_Block", ["x", "qp", "a"])


def _basis(columns):
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
    basis = _basis(pr)
    if basis.shape[1] != p.shape[1]:
        raise ValueError("profiled columns are rank deficient")
    return basis


def _check_task_rank(xr, x, dof):
    scale = np.linalg.norm(xr, axis=0)
    tolerance = np.linalg.norm(x, ord=2) * max(x.shape) * np.finfo(float).eps
    if np.any(scale <= tolerance):
        raise ValueError("task column has no support outside nuisance span")
    if _basis(xr / scale).shape[1] != x.shape[1]:
        raise ValueError("task columns are rank deficient after nuisance projection")
    if dof - x.shape[1] <= 0:
        raise ValueError("task model requires positive residual degrees of freedom")


class RunDesign:
    """Cached timing, modulator amplitudes, and confounds: never retains BOLD."""

    def __init__(self, expanded, times, nuisance, library, task_model, fingerprint):
        self.expanded = expanded
        task_rows = expanded.trial_type == "task"
        self.events = expanded.loc[task_rows, ["onset", "duration"]].reset_index(drop=True)
        self.times, self.nuisance, self.library = times, nuisance, library
        self.task_model, self.fingerprint = task_model, fingerprint
        self.q = _basis(nuisance)
        self.k = len(task_model.regressor_names)
        self.eligibility, self.trial_eligibility, self._blocks = {}, {}, {}

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
        _check_task_rank(xr, x, len(self.times) - q.shape[1])
        return _Block(xr, qp, xr.T @ xr)

    def block(self, candidate_id):
        """Projected task columns, profiled basis, and A; zeros when ineligible."""
        if candidate_id not in self._blocks:
            try:
                self._blocks[candidate_id] = self._build(candidate_id)
                self.eligibility[candidate_id] = (True, "")
            except ValueError as error:
                self.eligibility[candidate_id] = (False, str(error))
                empty = np.zeros((len(self.times), self.k))
                self._blocks[candidate_id] = _Block(
                    empty, np.zeros((len(self.times), 0)), np.zeros((self.k, self.k))
                )
        return self._blocks[candidate_id]

    def eligible(self, candidate_id):
        self.block(candidate_id)
        return self.eligibility[candidate_id]

    @lru_cache(maxsize=256)
    def trial_matrix(self, candidate_id):
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
        if candidate_id not in self.trial_eligibility:
            try:
                _project_design(self.trial_matrix(candidate_id), self.nuisance)
                self.trial_eligibility[candidate_id] = (True, "")
            except ValueError as error:
                self.trial_eligibility[candidate_id] = (False, str(error))
        return self.trial_eligibility[candidate_id]


@lru_cache(maxsize=32)
def _cached_design(
    expanded_bytes, names_bytes, times_bytes, nuisance_bytes, n_columns, library, task_model
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


def _validate_onsets(events, times):
    onsets = events["onset"].to_numpy(dtype=float)
    if np.any(onsets < times[0] + MIN_ONSET) or np.any(onsets >= times[-1]):
        raise ValueError("onset has no supported sampled response")


def prepare_runs(data, library, task_model=TaskModel()):
    runs = []
    for r, (events, times, confounds) in enumerate(
        zip(data.events, data.frame_times, data.confounds, strict=True)
    ):
        times = np.asarray(times, dtype=float)
        _validate_events(events, times, f"run-{r}")
        _validate_onsets(events, times)
        expanded = expand_events(events, task_model, r)
        nuisance = _nuisance_matrix(confounds, len(times))
        payloads = (
            np.asarray(expanded[["onset", "duration", "modulation"]], dtype="<f8").tobytes(),
            json.dumps(list(expanded.trial_type)).encode(),
            np.asarray(times, dtype="<f8").tobytes(),
            np.asarray(nuisance, dtype="<f8").tobytes(),
        )
        runs.append(_cached_design(*payloads, nuisance.shape[1], library, task_model))
    return tuple(runs)


def _batched_products(run, yr, batch_size):
    """B for every candidate as (H, K, F), multiplying batch_size candidates at once."""
    n = len(run.library.candidates)
    b = np.empty((n, run.k, yr.shape[1]))
    for start in range(0, n, batch_size):
        ids = range(start, min(start + batch_size, n))
        stacked = np.concatenate([run.block(cid).x for cid in ids], axis=1)
        b[start : start + len(ids)] = (stacked.T @ yr).reshape(len(ids), run.k, -1)
    return b


def signal_statistics(runs, signals, batch_size):
    """Project Y once per run; form A, B, C per candidate and confound-only energy."""
    a_all, b_all, c_all, energy = [], [], [], []
    for run, y in zip(runs, signals, strict=True):
        yr = y - run.q @ (run.q.T @ y)
        # Numerical zero only: no statistical threshold or weak-signal fallback.
        tolerance = np.linalg.norm(y, axis=0) * len(y) * np.finfo(float).eps
        yr[:, np.linalg.norm(yr, axis=0) <= tolerance] = 0
        c0 = np.sum(yr**2, axis=0)
        n = len(run.library.candidates)
        a = np.stack([run.block(cid).a for cid in range(n)])
        c = np.tile(c0, (n, 1))
        for cid in range(n):
            qp = run.block(cid).qp
            if qp.shape[1]:
                c[cid] -= np.sum((qp.T @ yr) ** 2, axis=0)
        a_all.append(a)
        b_all.append(_batched_products(run, yr, batch_size))
        c_all.append(c)
        energy.append(c0)
    return np.array(a_all), np.array(b_all), np.array(c_all), np.array(energy)


def pooled_amplitude(a_sum, b_sum):
    """Solve pooled normal equations per candidate; singular ones get zeros and False."""
    s = np.linalg.svd(a_sum, compute_uv=False)
    ok = s[:, -1] > s[:, 0] * a_sum.shape[-1] * np.finfo(float).eps
    amplitude = np.zeros_like(b_sum)
    if ok.any():
        amplitude[ok] = np.linalg.solve(a_sum[ok], b_sum[ok])
    return amplitude, ok


def prediction_loss(a, b, c, amplitude):
    cross = 2 * np.einsum("hkf,hkf->hf", amplitude, b)
    fitted = np.einsum("hkf,hkj,hjf->hf", amplitude, a, amplitude)
    loss = c - cross + fitted
    tolerance = 64 * np.finfo(float).eps * (c + np.abs(cross) + np.abs(fitted))
    if np.any(loss < -tolerance):
        raise ArithmeticError("negative HRF prediction SSE beyond roundoff tolerance")
    return np.maximum(loss, 0)


def loro_scores(a, b, c, energy):
    loss = np.zeros(c.shape[1:])
    eligible = np.ones(c.shape[1], dtype=bool)
    for r in range(len(a)):
        others = [i for i in range(len(a)) if i != r]
        # Explicit sums avoid cancellation when one run has much larger energy.
        amplitude, ok = pooled_amplitude(a[others].sum(axis=0), b[others].sum(axis=0))
        eligible &= ok
        loss += prediction_loss(a[r], b[r], c[r], amplitude)
    total = energy.sum(axis=0)
    scores = np.full_like(loss, np.nan)
    np.divide(loss, total[None, :], out=scores, where=total[None, :] > 0)
    scores = 1 - scores
    scores[~eligible] = -np.inf
    return scores


def choose_eligible(scores, runs):
    """Validate winners lazily; structural exclusions apply to every feature."""
    scores = scores.copy()
    checked = {}

    def check(cid):
        if cid not in checked:
            outcomes = [r.eligible(cid) for r in runs]
            checked[cid] = (
                all(ok for ok, _ in outcomes),
                "; ".join(
                    f"run {r}: {reason}"
                    for r, (ok, reason) in enumerate(outcomes)
                    if not ok
                ),
            )
            if not checked[cid][0]:
                scores[cid] = -np.inf
        return checked[cid][0]

    check(0)
    valid = np.any(np.isfinite(scores), axis=0)
    while True:
        best = np.max(np.where(np.isfinite(scores), scores, -np.inf), axis=0)
        indices = np.full(scores.shape[1], -1, dtype=int)
        for v in np.flatnonzero(valid & np.isfinite(best)):
            tolerance = 64 * np.finfo(float).eps * max(1, abs(best[v]))
            indices[v] = np.flatnonzero(scores[:, v] >= best[v] - tolerance)[0]
        if all([check(int(cid)) for cid in np.unique(indices[indices >= 0])]):
            break
    if not any(ok for ok, _ in checked.values()):
        for cid in range(len(scores)):
            if check(cid):
                break
        else:
            raise ValueError("no eligible, estimable HRF candidate across all runs")
    if np.any(valid & (indices < 0)):
        raise ValueError("no eligible HRF with finite prediction score")
    table = pd.DataFrame(
        [
            dict(
                hrf_id=cid,
                eligible=checked.get(cid, (None, "unchecked"))[0],
                reason=checked.get(cid, (None, "unchecked"))[1],
            )
            for cid in range(len(scores))
        ]
    )
    return indices, scores, table
