"""Fraction-specific maps and descriptive summaries of voxelwise selection."""

import numpy as np

from boldtailor.fractional_ridge import NORM_BASIS
from .hrf_artifacts import npz_artifact
from .single_trial_artifacts import json_artifact, scalar_artifact, table_artifact


def tuning_rows(mode, scope, tuned):
    selected, scores = tuned["selection"], tuned["scores"]
    percentile = tuned["summary_percentile"]
    mask = selected.scoring_mask
    summaries = (
        np.percentile(scores.cv_r2[:, mask], percentile, axis=1)
        if mask.any()
        else np.full(len(scores.grid), np.nan)
    )
    return [
        dict(
            mode=mode,
            scope=scope,
            fraction=f,
            summary_r2=float(r),
            percentile=percentile,
            grayordinates=int(mask.sum()),
            selected_grayordinates=int(np.count_nonzero(selected.ridge_fraction == f)),
        )
        for f, r in zip(scores.grid, summaries, strict=True)
    ]


def plot_tuning(axis, mode, result):
    for scope, tuned in result["tuning"].items():
        rows = tuning_rows(mode, scope, tuned)
        axis.plot(
            [r["fraction"] for r in rows],
            [r["summary_r2"] for r in rows],
            marker="o",
            label=scope,
        )
    axis.set(
        xlabel="Ridge fraction",
        ylabel=f"{rows[0]['percentile']:g}th percentile R² (summary)",
        title=f"{mode}: per-grayordinate CV selection",
    )
    axis.legend()


def fraction_fit_artifacts(stem, brain, descriptor, result, runs):
    prefix = f"{stem}_space-fsLR_den-91k_desc-notebook{descriptor}"
    return [
        scalar_artifact(
            prefix + "_stat-ridgefraction.dscalar.nii",
            brain,
            result["ridge_fraction"][None],
            ["selected_ridge_fraction"],
        ),
        scalar_artifact(
            prefix + "_stat-ridgealpha.dscalar.nii",
            brain,
            result["run_ridge_alphas"],
            [r.label for r in runs],
        ),
    ]


def tuning_artifacts(stem, brain, mode, scope, tuned, table):
    from .ridge_outputs import _map

    descriptor = f"{mode}FractionalCV{scope.title()}"
    base = f"{stem}_desc-notebook{descriptor}"
    scores, selected = tuned["scores"], tuned["selection"]
    ids = np.where(scores.fold_hrf_indices >= 0, scores.fold_hrf_indices, np.nan)
    metadata = dict(
        run_labels=list(scores.run_labels),
        fractions=list(scores.grid),
        selection_rule="maximum_encoding_r2_per_grayordinate",
        validation_target="fixed_ols_betas",
        fraction_norm_basis=NORM_BASIS,
        percentile_role="descriptive_only",
        summary_percentile=tuned["summary_percentile"],
        scoring_grayordinates=int(selected.scoring_mask.sum()),
        excluded_grayordinates=int((~selected.scoring_mask).sum()),
        trial_masks=[m.tolist() for m in scores.trial_masks],
        score="1 - sum(validation SSE) / sum(within-run validation SST)",
        tie_rule="largest fraction within 1e-12 of the maximum score",
        selection_statistic=True,
    )
    return [
        _map(
            stem,
            descriptor,
            "encodingcvr2",
            brain,
            scores.cv_r2,
            [f"encoding_inner_cv_r2_fraction-{f:g}" for f in scores.grid],
        ),
        _map(
            stem,
            descriptor,
            "ridgefraction",
            brain,
            selected.ridge_fraction[None],
            ["selected_ridge_fraction"],
        ),
        _map(
            stem,
            descriptor,
            "selectedencodingr2",
            brain,
            selected.selected_r2[None],
            ["selected_inner_cv_r2"],
        ),
        _map(
            stem,
            descriptor,
            "scoringmask",
            brain,
            selected.scoring_mask[None],
            ["common_scoring_mask"],
        ),
        _map(
            stem,
            descriptor,
            "foldhrfindex",
            brain,
            ids,
            [f"validation_{label}_training_hrf" for label in scores.run_labels],
        ),
        npz_artifact(base + "_folds.npz", sse=scores.fold_sse, sst=scores.fold_sst),
        table_artifact(
            base + "_scores.tsv",
            table.loc[(table["mode"] == mode) & (table.scope == scope)],
        ),
        json_artifact(base + "_metadata.json", metadata),
        json_artifact(base + "_provenance.json", tuned["provenance"].to_dict()),
    ]
