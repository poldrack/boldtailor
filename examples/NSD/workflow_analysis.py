"""Blockwise public-API calls used by the NSD tutorial."""

from contextlib import redirect_stdout
import os

import numpy as np
import pandas as pd

from boldtailor.fit import fit, task_delta_r2
from boldtailor import hrf_selection as selection_api
from boldtailor.hrf_selection import evaluate_hrf_split
from boldtailor.single_trial import fit_single_trials, fit_selected_hrfs
from boldtailor.cifti import spatial_signature
from boldtailor.diagnostics import correlate_rt
from boldtailor.parallel import map_blocks
from boldtailor.workflow.files import odd_even_parity, reaction_times
from boldtailor.workflow.inputs import (
    NSD_TASK_MODEL,
    REGRESSORS,
    load_block,
)


def _signature(runs, indices):
    return spatial_signature(runs[0].image.header.get_axis(1), indices)


def _select_block(indices, runs, root, library, task_model):
    data = load_block(runs, root, indices)
    options = dict(
        library=library,
        run_labels=[r.label for r in runs],
        feature_signature=_signature(runs, indices),
        task_model=task_model,
    )
    odd, even = odd_even_parity(runs).values()
    return dict(
        all=selection_api.select_hrfs(data, **options),
        odd=evaluate_hrf_split(data, train_runs=odd, test_runs=even, **options),
        even=evaluate_hrf_split(data, train_runs=even, test_runs=odd, **options),
    )


def select_hrfs(runs, root, blocks, library, *, n_jobs=1, task_model=NSD_TASK_MODEL):
    """Select on all runs and on each half; retain objects for later fits.

    The GLM keeps the full NSD task model; pass a subset (for example
    selection_task_model(False)) to score HRFs without the RT regressor.
    """
    selections = {}
    for indices, result in map_blocks(
        _select_block, blocks, args=(runs, root, library, task_model), n_jobs=n_jobs
    ):
        selections[tuple(indices)] = result
        print(f"HRF selection: {indices[0]}–{indices[-1]}", flush=True)
    return selections


def selection_maps(selections, n_features):
    maps = {
        name: np.full((4 if name in ("all", "odd", "even") else 3, n_features), np.nan)
        for name in ("all", "odd", "even", "odd_to_even", "even_to_odd")
    }
    for indices, bundle in selections.items():
        for name in ("all", "odd", "even"):
            result = bundle[name] if name == "all" else bundle[name].training_selection
            ids = np.where(result.hrf_indices >= 0, result.hrf_indices, np.nan)
            maps[name][:, list(indices)] = np.stack(
                [ids, result.cv_r2, result.canonical_cv_r2, result.delta_cv_r2]
            )
        for name in ("odd", "even"):
            result = bundle[name]
            target = "odd_to_even" if name == "odd" else "even_to_odd"
            maps[target][:, list(indices)] = np.stack(
                [result.test_r2, result.canonical_test_r2, result.delta_test_r2]
            )
    return maps


def _glm_block(job, runs, root, model):
    indices, selection = job
    data = load_block(runs, root, indices)
    options = (
        {}
        if selection is None
        else dict(hrf_selection=selection, feature_signature=_signature(runs, indices))
    )
    # Nilearn prints modulation notices without a verbosity option. Suppress
    # stdout within each worker; warnings/stderr and parent progress stay visible.
    with open(os.devnull, "w") as sink, redirect_stdout(sink):
        result = fit(data, model, **options)
        comparison = task_delta_r2(data, model, result)
    designs = (
        {(i, 0): x for i, x in enumerate(result.design_matrices)}
        if selection is None
        else {
            (i, hrf): result.group_design(i, hrf)
            for i, hrf in _fitted_pairs(result.hrf_indices, len(runs))
        }
    )
    return dict(
        effects=np.stack([result.effect(c) for c in REGRESSORS]),
        variances=np.stack([result.variance(c) for c in REGRESSORS]),
        t=np.stack([result.stat(c) for c in REGRESSORS]),
        z=np.stack([result.z_score(c) for c in REGRESSORS]),
        r2=np.stack([comparison.full_r2, comparison.nuisance_r2, comparison.delta_r2]),
        designs=designs,
        provenance=comparison.provenance.to_dict(),
    )


def _jobs(blocks, selections):
    return [
        (indices, None if selections is None else selections[tuple(indices)]["all"])
        for indices in blocks
    ]


def fit_glms(runs, root, blocks, model, *, selections=None, n_jobs=1):
    """Fit the same three contrasts using SPM or the selected HRF per feature."""
    n = runs[0].image.shape[1]
    result = {
        key: np.full((3, n), np.nan) for key in ("effects", "variances", "t", "z", "r2")
    }
    result.update(designs={}, provenance=[])
    for (indices, _), block in map_blocks(
        _glm_block, _jobs(blocks, selections), args=(runs, root, model), n_jobs=n_jobs
    ):
        for key in ("effects", "variances", "t", "z", "r2"):
            result[key][:, indices] = block[key]
        _collect_metadata(result, block, indices)
        print(f"GLM: {indices[0]}–{indices[-1]}", flush=True)
    return result


def _collect_metadata(result, block, indices):
    result["designs"].update(block["designs"])
    result["provenance"].append(
        dict(
            grayordinate_indices=np.asarray(indices).tolist(),
            record=block["provenance"],
        )
    )


def _fitted_pairs(hrf_indices, n_runs):
    """Every (run, HRF) pair a selected fit used, run-major, HRF ascending."""
    ids = np.unique(hrf_indices[hrf_indices >= 0])
    return [(i, int(hrf)) for i in range(n_runs) for hrf in ids]


def _trial_designs(result, runs, selection):
    if selection is None:
        return {(i, 0): x for i, x in enumerate(result.design.matrices)}
    designs = {}
    for i, hrf in _fitted_pairs(result.design.hrf_indices, len(runs)):
        values = result.design.matrix(i, hrf)
        trials = result.trial_table.query("run_index == @i").trial_id.tolist()
        columns = trials + list(runs[i].confounds.columns) + ["constant"]
        designs[i, hrf] = pd.DataFrame(
            values, columns=columns, index=runs[i].frame_times
        )
    return designs


def _beta_block(job, runs, root, alpha, fractions=None):
    indices, selection = job
    data = load_block(runs, root, indices)
    options = dict(ridge_alpha=alpha, run_labels=[r.label for r in runs])
    if fractions is not None:
        options["ridge_fraction"] = fractions[indices]
    if selection is None:
        result = fit_single_trials(data, hrf_model="spm", **options)
    else:
        result = fit_selected_hrfs(
            data,
            hrf_selection=selection,
            feature_signature=_signature(runs, indices),
            **options,
        )
    return dict(
        betas=result.run_betas,
        trial_table=result.trial_table,
        r2=np.stack([result.full_r2, result.nuisance_r2, result.delta_r2]),
        designs=_trial_designs(result, runs, selection),
        provenance=result.provenance.to_dict(),
        run_ridge_alphas=result.run_ridge_alphas,
    )


def fit_beta_series(
    runs,
    root,
    blocks,
    *,
    selections=None,
    ridge_alpha=0.0,
    n_jobs=1,
    ridge_fraction=None,
):
    """Fit raw trials at the supplied penalty; tuning is a separate operation."""
    hrf = "canonical" if selections is None else "optimized"
    from boldtailor.fractional_ridge import regularization

    alpha, fractions = regularization(
        ridge_alpha, ridge_fraction, runs[0].image.shape[1]
    )
    label = f"Beta series ({hrf}, " + (
        f"alpha={alpha:g})" if fractions is None else "fractional ridge)"
    )
    print(f"{label}: fitting {len(runs)} runs in {len(blocks)} blocks", flush=True)
    n = runs[0].image.shape[1]
    result = dict(
        betas=[np.full((len(r.events), n), np.nan, dtype=np.float32) for r in runs],
        r2=np.full((3, n), np.nan),
        designs={},
        provenance=[],
        ridge_alpha=alpha,
    )
    if fractions is not None:
        result.update(
            ridge_fraction=fractions, run_ridge_alphas=np.full((len(runs), n), np.nan)
        )
    for (indices, _), block in map_blocks(
        _beta_block,
        _jobs(blocks, selections),
        args=(runs, root, ridge_alpha, fractions),
        n_jobs=n_jobs,
    ):
        for target, values in zip(result["betas"], block["betas"], strict=True):
            target[:, indices] = values
        result["r2"][:, indices] = block["r2"]
        if fractions is not None:
            result["run_ridge_alphas"][:, indices] = np.asarray(
                block["run_ridge_alphas"]
            )
        result["trial_table"] = block["trial_table"]
        _collect_metadata(result, block, indices)
    result["rt"] = correlate_rt(
        result["betas"],
        reaction_times(runs),
        run_numbers=[r.number for r in runs],
    )
    print(f"{label}: complete", flush=True)
    return result
