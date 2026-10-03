"""Independent numerical references and run subsets for scientific tests."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pandas as pd
from nilearn.glm.first_level import (
    compute_regressor,
    make_first_level_design_matrix,
    run_glm,
)
from nilearn.glm.first_level.hemodynamic_models import (
    _sample_condition,
    glover_hrf,
    spm_hrf,
)
from scipy.linalg import block_diag
from scipy.optimize import brentq

from boldtailor._fractional_ridge import fraction_grid, prepare_fraction_betas
from boldtailor._hrf_design import stimulus_regressor
from boldtailor._single_trial_fit import prepare_trial_betas, validate_alpha
from boldtailor.data import from_arrays
from boldtailor.provenance import ProvenanceRecord, RunSources, SourceRef

_NILEARN_SHAPES = {"spm": spm_hrf, "glover": glover_hrf}


def peak_kernel(name):
    """Nilearn's named HRF shape rescaled to unit peak, built independently."""
    shape = _NILEARN_SHAPES[name]

    def kernel(tr, oversampling=50):
        values = shape(tr, oversampling)
        return values / values.max()

    return kernel


def realized_event_response(
    kernel, onset, duration, times, oversampling=50, min_onset=-24.0
):
    """One event's response on Nilearn's own oversampled boxcar and grid.

    The response is not truncated at the run end: the event's peak is that of
    its full response, even when acquisition stops before it.
    """
    boxcar, grid = _sample_condition(
        np.array([[onset], [duration], [1.0]]), times, oversampling, min_onset
    )
    return np.convolve(boxcar, np.asarray(kernel)), grid


def oracle_event_scales(
    kernel_fn, onsets, durations, times, oversampling=50, min_onset=-24.0
):
    """1 / peak of each event's realized oversampled response (brute force)."""
    kernel = kernel_fn(float(np.min(np.diff(times))), oversampling)
    return np.array(
        [
            1
            / realized_event_response(kernel, o, d, times, oversampling, min_onset)[
                0
            ].max()
            for o, d in zip(onsets, durations, strict=True)
        ]
    )


def scaled_condition(
    onsets, durations, amplitudes, kernel_fn, times, oversampling=50, min_onset=-24.0
):
    """Nilearn condition rows with amplitudes scaled to unit event peaks."""
    onsets = np.asarray(onsets, dtype=float)
    scales = oracle_event_scales(
        kernel_fn, onsets, durations, times, oversampling, min_onset
    )
    amplitudes = np.broadcast_to(np.asarray(amplitudes, dtype=float), scales.shape)
    return np.array([onsets, durations, amplitudes * scales], dtype=float)


def peak_design_matrix(frame_times, *, hrf_model="glover", events=None, **kwargs):
    """Nilearn's design matrix with a named basis replaced by its peak kernel,
    each event's amplitude scaled so its response peaks at one."""
    if hrf_model not in _NILEARN_SHAPES:
        return make_first_level_design_matrix(
            frame_times, hrf_model=hrf_model, events=events, **kwargs
        )
    kernel = peak_kernel(hrf_model)
    if events is not None:
        scales = oracle_event_scales(
            kernel,
            events.onset,
            events.duration,
            frame_times,
            kwargs.get("oversampling", 50),
            kwargs.get("min_onset", -24.0),
        )
        events = events.assign(modulation=events.get("modulation", 1.0) * scales)
    matrix = make_first_level_design_matrix(
        frame_times, hrf_model=kernel, events=events, **kwargs
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


SHARED_DELTA_ACTIVITY_KEYS = {
    "name",
    "stage",
    "parent_analysis_id",
    "definition",
    "clip_below_zero",
    "clip_policy",
    "diagnostic_noise_model",
    "inferential_noise_model",
    "nuisance_model",
    "undefined_features",
}


def trial_beta_path(x, nuisance, signals, *, alphas):
    """Yield (alpha, betas) from one shared normalized factorization."""
    alphas = tuple(validate_alpha(a) for a in alphas)
    prepared = prepare_trial_betas(x, nuisance, signals)
    for alpha in alphas:
        yield alpha, prepared.betas_at(alpha)


def fraction_beta_path(x, nuisance, signals, *, fractions):
    """Yield (fraction, betas, alphas) from one prepared raw-basis solver."""
    grid = fraction_grid(fractions)
    prepared = prepare_fraction_betas(x, nuisance, signals)
    for fraction in grid:
        betas, alphas = prepared.solve(fraction)
        yield fraction, betas, alphas


def glm_run_sources(run):
    """Complete source references for one run of the selected-HRF problem."""
    return RunSources(
        **{
            role: SourceRef(
                role,
                uri=f"run-{run}_{role}.tsv",
                byte_size=1024,
                modified_at="2026-09-27T12:00:00Z",
            )
            for role in ("signal", "events", "confounds")
        }
    )


def hrf_glm_oracle_design(events, times, confounds, candidate, model):
    """One run's design built from Nilearn regressors with a fixed kernel."""
    columns = {}
    for condition in sorted(events.trial_type.unique()):
        selected = events.loc[events.trial_type == condition]
        values, _ = compute_regressor(
            scaled_condition(
                selected.onset,
                selected.duration,
                selected.modulation,
                candidate.kernel,
                times,
                model.oversampling,
                model.min_onset,
            ),
            candidate.kernel,
            times,
            oversampling=model.oversampling,
            min_onset=model.min_onset,
        )
        columns[condition] = values[:, 0]
    nuisance = make_first_level_design_matrix(
        times,
        events=None,
        drift_model=model.drift_model,
        high_pass=model.high_pass,
        drift_order=model.drift_order,
        add_regs=confounds.loc[:, list(model.confounds)],
    )
    return pd.DataFrame(columns, index=times).join(nuisance)


def replace_indices(selection, indices):
    """Rebuild a selection with new HRF IDs and a matching assignment identity."""
    indices = np.asarray(indices, dtype=np.int64)
    record = selection.provenance.to_dict()
    record["activities"][-1]["hrf_assignment_fingerprint"] = sha256(
        indices.astype("<i8").tobytes()
    ).hexdigest()
    provenance = ProvenanceRecord.from_dict(record)
    return replace(selection, hrf_indices=indices, provenance=provenance)
