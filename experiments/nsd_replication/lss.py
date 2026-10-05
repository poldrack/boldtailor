"""Least-squares-separate single-trial betas from boldtailor trial designs."""

import numpy as np


def lss_run(x, nuisance, y):
    total = x.sum(axis=1)
    betas = np.empty((x.shape[1], y.shape[1]))
    for i in range(x.shape[1]):
        design = np.column_stack([x[:, i], total - x[:, i], nuisance])
        betas[i] = np.linalg.lstsq(design, y, rcond=None)[0][0]
    return betas


def n_trials(result, run):
    return int(np.sum(result.trial_table["run_index"] == run))


def lss_canonical(result, signals):
    out = []
    for run, (matrix, y) in enumerate(
        zip(result.design.matrices, signals, strict=True)
    ):
        k = n_trials(result, run)
        values = matrix.to_numpy()
        out.append(lss_run(values[:, :k], values[:, k:], np.asarray(y, float)))
    return np.vstack(out)


def _selected_run(result, run, y):
    ids = result.design.hrf_indices
    k = n_trials(result, run)
    betas = np.full((k, len(ids)), np.nan)
    for hrf in np.unique(ids[ids >= 0]):
        cols = np.flatnonzero(ids == hrf)
        matrix = result.design.matrix(run, int(hrf))
        betas[:, cols] = lss_run(matrix[:, :k], matrix[:, k:], y[:, cols])
    return betas


def lss_selected(result, signals):
    return np.vstack(
        [
            _selected_run(result, run, np.asarray(y, float))
            for run, y in enumerate(signals)
        ]
    )
