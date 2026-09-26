"""Nuisance-projected sufficient statistics and data-independent design caches."""

from functools import lru_cache
from hashlib import sha256

import numpy as np
import pandas as pd

from boldtailor._hrf_design import stimulus_regressor
from boldtailor._single_trial_design import _nuisance_matrix, compile_trial_run
from boldtailor._single_trial_fit import _project_design


class RunDesign:
    """Cached timing/confounds only: never retains observed BOLD or RT."""

    def __init__(self, timing, times, nuisance, library, fingerprint):
        self.events = pd.DataFrame(timing, columns=["onset", "duration"])
        self.times, self.nuisance, self.library = times, nuisance, library
        self.fingerprint = fingerprint
        u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
        self.q = u[:, s > s[0] * max(nuisance.shape) * np.finfo(float).eps]
        self.x = np.zeros((len(times), len(library.candidates)))
        self.eligibility = {}
        for candidate in library.candidates:
            try:
                x = stimulus_regressor(self.events, times, candidate)
                xr = x - self.q @ (self.q.T @ x)
                if (
                    np.linalg.norm(xr)
                    <= np.linalg.norm(x) * len(times) * np.finfo(float).eps
                ):
                    raise ValueError("stimulus has no support outside nuisance span")
                self.x[:, candidate.id] = xr
            except ValueError as error:
                self.eligibility[candidate.id] = (False, str(error))
        self.a = np.sum(self.x**2, axis=0)

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

    def eligible(self, candidate_id):
        if candidate_id not in self.eligibility:
            try:
                _project_design(self.trial_matrix(candidate_id), self.nuisance)
                self.eligibility[candidate_id] = (True, "")
            except ValueError as error:
                self.eligibility[candidate_id] = (False, str(error))
        return self.eligibility[candidate_id]


@lru_cache(maxsize=32)
def _cached_design(timing_bytes, times_bytes, nuisance_bytes, n_columns, library):
    timing = np.frombuffer(timing_bytes, dtype="<f8").reshape(-1, 2)
    times = np.frombuffer(times_bytes, dtype="<f8")
    nuisance = np.frombuffer(nuisance_bytes, dtype="<f8").reshape(len(times), n_columns)
    digest = sha256(
        timing_bytes
        + times_bytes
        + nuisance_bytes
        + str(n_columns).encode()
        + library.fingerprint.encode()
    ).hexdigest()
    return RunDesign(timing, times, nuisance, library, digest)


def prepare_runs(data, library):
    runs = []
    for events, times, confounds in zip(
        data.events, data.frame_times, data.confounds, strict=True
    ):
        nuisance = _nuisance_matrix(confounds, len(times))
        values = [events[["onset", "duration"]], times, nuisance]
        payloads = [np.asarray(v, dtype="<f8").tobytes() for v in values]
        runs.append(_cached_design(*payloads, nuisance.shape[1], library))
    return tuple(runs)


def signal_statistics(runs, signals, batch_size):
    """Project Y once per run and form A, B, C in candidate batches."""
    bs, cs = [], []
    for run, y in zip(runs, signals, strict=True):
        yr = y - run.q @ (run.q.T @ y)
        # Numerical zero only: no statistical threshold or weak-signal fallback.
        tolerance = np.linalg.norm(y, axis=0) * len(y) * np.finfo(float).eps
        yr[:, np.linalg.norm(yr, axis=0) <= tolerance] = 0
        b = np.empty((run.x.shape[1], y.shape[1]))
        for start in range(0, len(b), batch_size):
            b[start : start + batch_size] = run.x[:, start : start + batch_size].T @ yr
        bs.append(b)
        cs.append(np.sum(yr**2, axis=0))
    return np.array([r.a for r in runs]), np.array(bs), np.array(cs)


def prediction_loss(a, b, c, amplitude):
    cross = 2 * amplitude * b
    fitted = amplitude**2 * a[:, None]
    loss = c[None, :] - cross + fitted
    tolerance = 64 * np.finfo(float).eps * (c[None, :] + np.abs(cross) + np.abs(fitted))
    if np.any(loss < -tolerance):
        raise ArithmeticError("negative HRF prediction SSE beyond roundoff tolerance")
    return np.maximum(loss, 0)


def loro_scores(a, b, c):
    loss = np.zeros_like(b[0])
    eligible = np.all(a > 0, axis=0)
    for r in range(len(a)):
        others = [i for i in range(len(a)) if i != r]
        # Explicit sums avoid cancellation when one run has much larger energy.
        numerator = b[others].sum(axis=0)
        denominator = a[others].sum(axis=0)
        amplitude = np.divide(
            numerator,
            denominator[:, None],
            out=np.zeros_like(numerator),
            where=denominator[:, None] > 0,
        )
        loss += prediction_loss(a[r], b[r], c[r], amplitude)
    total = c.sum(axis=0)
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
