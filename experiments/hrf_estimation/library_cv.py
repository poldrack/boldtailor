"""Library selection by inner leave-one-run-out task prediction."""

import numpy as np
from boldtailor._hrf_cv import loro_scores

from experiments.hrf_estimation.design import build_design
from experiments.hrf_estimation.estimators import HrfEstimate, validate_features
from experiments.hrf_estimation.scoring import residuals


def fit_library_cv(runs, library, settings):
    if len(runs) < 2:
        raise ValueError("library CV needs at least two training runs")
    ids = runs[0].feature_ids
    validate_features(runs, ids, runs[0].brain_axis)
    mode = settings.get("rt_mode", "amplitude")
    prepared = tuple(_fixed_projection(run) for run in runs)
    results = [
        _candidate_scores(runs, candidate, mode, prepared)
        for candidate in library.candidates
    ]
    scores = np.stack([result[0] for result in results])
    any_full_rank = any(result[1] for result in results)
    finite = np.isfinite(scores)
    best = np.argmax(np.where(finite, scores, -np.inf), axis=0)
    valid = finite.any(axis=0)
    kernels = tuple(library.candidates[i] if ok else None for i, ok in zip(best, valid))
    selected = scores[best, np.arange(len(ids))]
    return HrfEstimate(
        ids.copy(),
        kernels,
        tuple(
            "ok" if ok else ("no_signal" if any_full_rank else "rank_deficient")
            for ok in valid
        ),
        tuple(r.run_id for r in runs),
        {
            "selected_cv_r2": selected,
            "library_fingerprint": library.fingerprint,
            "selection": "leave_one_run_out_prediction",
            "rt_mode": mode,
        },
        runs[0].brain_axis,
    )


def _fixed_projection(run):
    nuisance = run.confounds.to_numpy(dtype=float)
    u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
    keep = s > s[0] * max(nuisance.shape) * np.finfo(float).eps
    basis = u[:, keep]
    return basis, run.signals - basis @ (basis.T @ run.signals)


def _project_run(run, candidate, mode, prepared):
    parts = build_design(run, candidate, mode)
    basis, signal = prepared
    task = parts.task - basis @ (basis.T @ parts.task)
    junk = parts.junk - basis @ (basis.T @ parts.junk)
    task, _ = residuals(junk, task)
    signal, _ = residuals(junk, signal)
    return task, signal


def _candidate_scores(runs, candidate, mode, prepared):
    projected = [
        _project_run(run, candidate, mode, fixed)
        for run, fixed in zip(runs, prepared, strict=True)
    ]
    a = np.stack([x.T @ x for x, _ in projected])[:, None]
    b = np.stack([x.T @ y for x, y in projected])[:, None]
    energy = np.stack([np.sum(y**2, axis=0) for _, y in projected])
    score = loro_scores(a, b, energy[:, None], energy)[0]
    full_rank = not np.isneginf(score).all()
    raw_energy = sum(np.sum(r.signals**2, axis=0) for r in runs)
    valid = energy.sum(axis=0) > 64 * np.finfo(float).eps * raw_energy
    score[~valid] = np.nan
    return score, full_rank
