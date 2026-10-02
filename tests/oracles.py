"""Independent numerical references and run subsets for scientific tests."""

import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.first_level.hemodynamic_models import glover_hrf, spm_hrf
from scipy.linalg import block_diag
from scipy.optimize import brentq

from boldtailor._hrf_design import stimulus_regressor
from boldtailor.data import from_arrays

_NILEARN_SHAPES = {"spm": spm_hrf, "glover": glover_hrf}


def peak_kernel(name):
    """Nilearn's named HRF shape rescaled to unit peak, built independently."""
    shape = _NILEARN_SHAPES[name]

    def kernel(tr, oversampling=50):
        values = shape(tr, oversampling)
        return values / values.max()

    return kernel


def peak_design_matrix(frame_times, *, hrf_model="glover", **kwargs):
    """Nilearn's design matrix with a named basis replaced by its peak kernel."""
    if hrf_model not in _NILEARN_SHAPES:
        return make_first_level_design_matrix(
            frame_times, hrf_model=hrf_model, **kwargs
        )
    matrix = make_first_level_design_matrix(
        frame_times, hrf_model=peak_kernel(hrf_model), **kwargs
    )
    matrix.columns = [name.removesuffix("_kernel") for name in matrix.columns]
    return matrix


def fractional_beta_oracle(x, n, y, fraction):
    """Solve augmented regression directly, not through the production SVD."""
    scale = np.ones(x.shape[1])
    matrix = np.column_stack([x / scale, n])

    def coefficients(alpha):
        penalty = np.column_stack(
            [np.sqrt(alpha) * np.eye(x.shape[1]), np.zeros((x.shape[1], n.shape[1]))]
        )
        return np.linalg.lstsq(
            np.vstack([matrix, penalty]), np.r_[y, np.zeros(x.shape[1])], rcond=None
        )[0]

    baseline = np.linalg.norm(coefficients(0)[: x.shape[1]])
    alpha = (
        0
        if fraction == 1
        else brentq(
            lambda a: np.linalg.norm(coefficients(a)[: x.shape[1]]) / baseline
            - fraction,
            0,
            1e7,
            xtol=1e-13,
        )
    )
    coef = coefficients(alpha)
    return coef[: x.shape[1]] / scale, coef[x.shape[1] :], alpha


def subset_runs(data, indices, *, signals=None, events=None):
    return from_arrays(
        [(data.signals if signals is None else signals)[i] for i in indices],
        [(data.events if events is None else events)[i] for i in indices],
        frame_times=[data.frame_times[i] for i in indices],
        confounds=[data.confounds[i] for i in indices],
        sources=[data.provenance.sources[i] for i in indices],
    )


def columns_for(data, library, cid, task_model):
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import expand_events, task_columns

    out = []
    for r, (e, t) in enumerate(zip(data.events, data.frame_times, strict=True)):
        frame = task_columns(
            expand_events(e, task_model, r), t, hrf_model(library.candidates[cid])
        )
        out.append(frame)
    return out


def stacked_training_ols_oracle(data, library, cid, task_model, train, test):
    """Nilearn OLS on a stacked training design, then frozen held-out prediction."""
    frames = columns_for(data, library, cid, task_model)
    xs, zs, ns = [], [], []
    for r, frame in enumerate(frames):
        xs.append(frame[list(task_model.regressor_names)].to_numpy())
        n = np.column_stack([data.confounds[r].to_numpy(), np.ones(len(frame))])
        profiled = [c for c in task_model.profiled_names if c in frame]
        zs.append(np.column_stack([n, frame[profiled].to_numpy()]) if profiled else n)
        ns.append(n)
    design = np.column_stack(
        [np.concatenate([xs[r] for r in train]), block_diag(*[zs[r] for r in train])]
    )
    y = np.concatenate([data.signals[r] for r in train])
    labels, results = run_glm(y, design, noise_model="ols")
    beta = results[labels[0]].theta[: xs[0].shape[1]]
    loss, null = np.zeros(data.n_features), np.zeros(data.n_features)
    for r in test:
        residual = data.signals[r] - xs[r] @ beta
        residual = residual - zs[r] @ np.linalg.lstsq(zs[r], residual, rcond=None)[0]
        yr = (
            data.signals[r]
            - ns[r] @ np.linalg.lstsq(ns[r], data.signals[r], rcond=None)[0]
        )
        loss += np.sum(residual**2, axis=0)
        null += np.sum(yr**2, axis=0)
    return beta, loss, null


def loro_oracle(data, library, task_model):
    scores = []
    for cid in range(len(library.candidates)):
        folds = [
            stacked_training_ols_oracle(
                data,
                library,
                cid,
                task_model,
                [i for i in range(data.n_runs) if i != r],
                [r],
            )
            for r in range(data.n_runs)
        ]
        scores.append(1 - sum(f[1] for f in folds) / sum(f[2] for f in folds))
    return np.array(scores)


def task_model_oracle(data, candidate, train, test):
    xs, ns = [], []
    for e, t, n in zip(data.events, data.frame_times, data.confounds, strict=True):
        xs.append(stimulus_regressor(e, t, candidate))
        ns.append(np.column_stack([n, np.ones(len(t))]))
    design = np.column_stack(
        [np.concatenate([xs[r] for r in train]), block_diag(*[ns[r] for r in train])]
    )
    beta = np.linalg.lstsq(
        design, np.concatenate([data.signals[r] for r in train]), rcond=None
    )[0][0]
    loss, null = np.zeros(data.n_features), np.zeros(data.n_features)
    for r in test:
        y, n = data.signals[r], ns[r]
        residual = y - xs[r][:, None] * beta
        residual -= n @ np.linalg.lstsq(n, residual, rcond=None)[0]
        yr = y - n @ np.linalg.lstsq(n, y, rcond=None)[0]
        loss += np.sum(residual**2, axis=0)
        null += np.sum(yr**2, axis=0)
    return beta, loss, null


def oracle_cv(data, library):
    scores = []
    for c in library.candidates:
        folds = [
            task_model_oracle(data, c, [i for i in range(data.n_runs) if i != r], [r])
            for r in range(data.n_runs)
        ]
        scores.append(1 - sum(f[1] for f in folds) / sum(f[2] for f in folds))
    return np.array(scores)


def nilearn_original_space_diagnostics(signals, design, noise_model):
    """(R², SSE, SST) from nilearn's fit, scored on unwhitened predictions."""
    matrix = np.asarray(design, dtype=float)
    labels, results = run_glm(signals, matrix, noise_model=noise_model)
    prediction = np.empty_like(signals, dtype=float)
    for label, result in results.items():
        prediction[:, labels == label] = matrix @ result.theta
    sse = ((signals - prediction) ** 2).sum(axis=0)
    sst = ((signals - signals.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - sse / sst, sse, sst


def nilearn_original_space_r2(signals, design, noise_model):
    """Pooled R² from nilearn's fit, scored on unwhitened predictions."""
    return nilearn_original_space_diagnostics(signals, design, noise_model)[0]


def nilearn_pooled_ols_r2(signals, designs):
    """R² pooled over runs: summed SSE and SST of per-run nilearn OLS fits."""
    sums = [
        nilearn_original_space_diagnostics(y, d, "ols")[1:]
        for y, d in zip(signals, designs, strict=True)
    ]
    return 1.0 - sum(s[0] for s in sums) / sum(s[1] for s in sums)
