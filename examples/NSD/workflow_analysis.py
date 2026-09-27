"""Blockwise public-API calls used by the NSD tutorial."""

from contextlib import redirect_stdout
import os

import numpy as np
import pandas as pd

from boldtailor.fit import fit, task_delta_r2
from boldtailor.hrf_selection import select_hrf, evaluate_hrf_split
from boldtailor.single_trial import fit_single_trials, fit_selected_hrfs
from .nsd_hrf import spatial_signature
from .parallel_blocks import map_blocks
from .rt_diagnostics import correlate_rt
from .workflow_inputs import load_block, REGRESSORS


def _signature(runs, indices):
    return spatial_signature(runs[0].image.header.get_axis(1), indices)


def _select_block(indices, runs, root, library):
    data = load_block(runs, root, indices)
    options = dict(
        library=library,
        run_labels=[r.label for r in runs],
        feature_signature=_signature(runs, indices),
    )
    odd = [i for i, r in enumerate(runs) if r.number % 2]
    even = [i for i, r in enumerate(runs) if not r.number % 2]
    return dict(
        all=select_hrf(data, **options),
        odd=evaluate_hrf_split(data, train_runs=odd, test_runs=even, **options),
        even=evaluate_hrf_split(data, train_runs=even, test_runs=odd, **options),
    )


def select_hrfs(runs, root, blocks, library, *, n_jobs=1):
    """Select on all runs and on each half; retain objects for later fits."""
    selections = {}
    for indices, result in map_blocks(
        _select_block, blocks, args=(runs, root, library), n_jobs=n_jobs
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
    data = load_block(runs, root, indices, glm=True)
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
        else result.group_designs
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


def _trial_designs(result, runs, selection):
    if selection is None:
        return {(i, 0): x for i, x in enumerate(result.design_matrices)}
    designs = {}
    for (i, hrf), values in result.group_designs.items():
        trials = result.trial_table.query("run_index == @i").trial_id.tolist()
        columns = trials + list(runs[i].confounds.columns) + ["constant"]
        designs[i, hrf] = pd.DataFrame(
            values, columns=columns, index=runs[i].frame_times
        )
    return designs


def _beta_block(job, runs, root, alpha):
    indices, selection = job
    data = load_block(runs, root, indices)
    options = dict(ridge_alpha=alpha, run_labels=[r.label for r in runs])
    if selection is None:
        result = fit_single_trials(data, hrf="spm", **options)
    else:
        result = fit_selected_hrfs(
            data,
            selection=selection,
            feature_signature=_signature(runs, indices),
            **options,
        )
    return dict(
        betas=result.run_betas,
        trial_table=result.trial_table,
        r2=np.stack([result.full_r2, result.nuisance_r2, result.delta_r2]),
        designs=_trial_designs(result, runs, selection),
        provenance=result.provenance.to_dict(),
    )


def fit_beta_series(runs, root, blocks, *, selections=None, ridge_alpha=0.0, n_jobs=1):
    """Fit one coefficient per raw trial; RT never tunes HRFs or ridge strength."""
    n = runs[0].image.shape[1]
    result = dict(
        betas=[np.full((len(r.events), n), np.nan, dtype=np.float32) for r in runs],
        r2=np.full((3, n), np.nan),
        designs={},
        provenance=[],
        ridge_alpha=ridge_alpha,
    )
    for (indices, _), block in map_blocks(
        _beta_block,
        _jobs(blocks, selections),
        args=(runs, root, ridge_alpha),
        n_jobs=n_jobs,
    ):
        for target, values in zip(result["betas"], block["betas"], strict=True):
            target[:, indices] = values
        result["r2"][:, indices] = block["r2"]
        result["trial_table"] = block["trial_table"]
        _collect_metadata(result, block, indices)
        print(
            f"Beta series (alpha={ridge_alpha:g}): {indices[0]}–{indices[-1]}",
            flush=True,
        )
    result["rt"] = correlate_rt(
        result["betas"],
        [r.events.response_time.to_numpy() for r in runs],
        run_numbers=[r.number for r in runs],
    )
    return result
