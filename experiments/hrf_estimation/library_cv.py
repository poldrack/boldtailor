"""Library selection by inner leave-one-run-out task prediction."""

import numpy as np

from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.estimators import HrfEstimate
from experiments.hrf_estimation.scoring import residuals


def fit_library_cv(runs, library, settings):
    if len(runs) < 2:
        raise ValueError('library CV needs at least two training runs')
    ids = runs[0].feature_ids
    if any(not np.array_equal(r.feature_ids, ids) for r in runs):
        raise ValueError('training feature identities/order differ')
    mode = settings.get('rt_mode', 'amplitude')
    scores = np.stack([_candidate_scores(runs, candidate, mode)
                       for candidate in library.candidates])
    finite = np.isfinite(scores)
    best = np.argmax(np.where(finite, scores, -np.inf), axis=0)
    valid = finite.any(axis=0)
    kernels = tuple(library.candidates[i] if ok else None for i, ok in zip(best, valid))
    selected = scores[best, np.arange(len(ids))]
    return HrfEstimate(ids.copy(), kernels,
        tuple('ok' if ok else 'no_signal' for ok in valid),
        tuple(r.run_id for r in runs),
        {'selected_cv_r2': selected, 'library_fingerprint': library.fingerprint,
         'selection': 'leave_one_run_out_prediction', 'rt_mode': mode})


def _project_run(run, candidate, mode):
    parts = build_design(run, candidate, mode)
    nuisance = np.column_stack([parts.nuisance, parts.junk])
    task, _ = residuals(nuisance, parts.task)
    signal, _ = residuals(nuisance, run.signals)
    return task, signal


def _candidate_scores(runs, candidate, mode):
    projected = [_project_run(run, candidate, mode) for run in runs]
    loss = np.zeros(runs[0].signals.shape[1])
    energy = sum(np.sum(y**2, axis=0) for _, y in projected)
    raw_energy = sum(np.sum(r.signals**2, axis=0) for r in runs)
    for held_out, (x, y) in enumerate(projected):
        others = [pair for i, pair in enumerate(projected) if i != held_out]
        train_x = np.vstack([pair[0] for pair in others])
        train_y = np.vstack([pair[1] for pair in others])
        beta, _, rank, _ = np.linalg.lstsq(train_x, train_y, rcond=None)
        if rank != train_x.shape[1]:
            return np.full_like(loss, np.nan)
        loss += np.sum((y-x@beta)**2, axis=0)
    valid = energy > 64*np.finfo(float).eps*raw_energy
    score = np.full_like(loss, np.nan)
    np.divide(loss, energy, out=score, where=valid)
    return 1-score
