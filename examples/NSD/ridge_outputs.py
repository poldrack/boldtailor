"""Auditable ridge selection scores, held-out predictions, and tuning plots."""

from pathlib import Path
from boldtailor.trial_encoding import encoding_metadata

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from boldtailor.ridge_results import FractionSelection
from boldtailor.fractional_ridge import NORM_BASIS
from .fractional_outputs import (
    tuning_rows,
    plot_tuning,
    tuning_artifacts,
    fraction_fit_artifacts,
)

from boldtailor.cifti import scalar_artifact
from .workflow_artifacts import (
    figure_artifact,
    json_artifact,
    notebook_map,
    npz_artifact,
    parameter_artifact,
    table_artifact,
)


def tuning_table(results):
    rows = []
    for mode, result in results.items():
        for scope, tuned in result["tuning"].items():
            selected = tuned["selection"]
            if isinstance(selected, FractionSelection):
                rows.extend(tuning_rows(mode, scope, tuned))
                continue
            for alpha, value in zip(
                selected.alphas, selected.objective_scores, strict=True
            ):
                rows.append(
                    dict(
                        mode=mode,
                        scope=scope,
                        alpha=alpha,
                        objective=float(value),
                        selected=alpha == selected.ridge_alpha,
                        percentile=selected.percentile,
                        grayordinates=int(selected.scoring_mask.sum()),
                    )
                )
    return pd.DataFrame(rows)


def tuning_figure(results):
    figure = Figure(figsize=(6 * len(results), 4), layout="constrained")
    axes = figure.subplots(1, len(results), squeeze=False)[0]
    for axis, (mode, result) in zip(axes, results.items(), strict=True):
        if isinstance(result["tuning"]["all"]["selection"], FractionSelection):
            plot_tuning(axis, mode, result)
            continue
        positive = []
        for scope, tuned in result["tuning"].items():
            selected = tuned["selection"]
            axis.plot(
                selected.alphas,
                selected.objective_scores,
                marker="o",
                label=f"{scope}: alpha={selected.ridge_alpha:g}",
            )
            positive.extend(a for a in selected.alphas if a > 0)
        if positive:
            axis.set_xscale("symlog", linthresh=min(positive))
        axis.set(
            title=f"{mode}: inner CV selection",
            xlabel="Ridge alpha",
            ylabel=f"{selected.percentile:g}th percentile of encoding R²",
        )
        axis.legend()
    return figure


def _tuning_artifacts(stem, brain, mode, scope, result, table):
    if isinstance(result["selection"], FractionSelection):
        return tuning_artifacts(stem, brain, mode, scope, result, table)
    descriptor = f"{mode}RidgeCV{scope.title()}"
    base = f"{stem}_desc-notebook{descriptor}"
    scores, selected = result["scores"], result["selection"]
    ids = np.where(scores.fold_hrf_indices >= 0, scores.fold_hrf_indices, np.nan)
    return [
        notebook_map(
            stem,
            brain,
            descriptor,
            "encodingcvr2",
            scores.cv_r2,
            [f"encoding_inner_cv_r2_alpha-{a:g}" for a in scores.grid],
        ),
        notebook_map(
            stem,
            brain,
            descriptor,
            "scoringmask",
            selected.scoring_mask[None],
            ["common_scoring_mask"],
        ),
        notebook_map(
            stem,
            brain,
            descriptor,
            "foldhrfindex",
            ids,
            [f"validation_{label}_training_hrf" for label in scores.run_labels],
        ),
        npz_artifact(base + "_folds.npz", sse=scores.fold_sse, sst=scores.fold_sst),
        table_artifact(
            base + "_scores.tsv",
            table.loc[(table["mode"] == mode) & (table.scope == scope)],
        ),
        json_artifact(base + "_provenance.json", result["provenance"].to_dict()),
        json_artifact(
            base + "_metadata.json",
            dict(
                run_labels=list(scores.run_labels),
                alphas=list(scores.grid),
                selected_alpha=selected.ridge_alpha,
                percentile=selected.percentile,
                objective_scores=selected.objective_scores.tolist(),
                scoring_grayordinates=int(selected.scoring_mask.sum()),
                excluded_grayordinates=int((~selected.scoring_mask).sum()),
                trial_masks=[m.tolist() for m in scores.trial_masks],
                score="1 - sum(validation SSE) / sum(within-run validation SST)",
                spatial_reduction="linear percentile over the common finite scoring mask",
                tie_rule="smallest alpha within 1e-12 of the maximum objective",
                validation_target="candidate_regularized_betas",
                selection_statistic=True,
            ),
        ),
    ]


def _prediction_artifacts(stem, brain, descriptor, runs, outer):
    directory = str(Path(stem).parent)
    by_label = {r.label: r for r in runs}
    artifacts = []
    for i, label in enumerate(outer["test_run_labels"]):
        run = by_label[label]
        names = [f"{label}_trial-{j + 1:04d}" for j in range(len(run.events))]
        for suffix, key in (("predictions", "predictions"), ("targets", "targets")):
            path = f"{directory}/{run.inputs.stem}_space-fsLR_den-91k_desc-notebook{descriptor}_{suffix}.dscalar.nii"
            artifacts.append(scalar_artifact(path, brain, outer[key][i], names))
    return artifacts


def _outer_artifacts(stem, brain, runs, mode, scope, outer, library):
    fractional = "ridge_fraction" in outer
    kind = "FractionalCV" if fractional else "RidgeCV"
    descriptor = mode + kind + "".join(word.title() for word in scope.split("_"))
    base = f"{stem}_desc-notebook{descriptor}"
    ids = np.where(outer["hrf_indices"] >= 0, outer["hrf_indices"], np.nan)
    artifacts = [
        notebook_map(
            stem,
            brain,
            descriptor,
            "encodingpredictionr2",
            outer["encoding_r2"][None],
            [f"encoding_outer_r2_{scope}"],
        ),
        notebook_map(
            stem,
            brain,
            descriptor,
            "coefficients",
            outer["coefficients"],
            ["task", "trial_type", "response_time"],
        ),
        notebook_map(
            stem, brain, descriptor, "hrfindex", ids[None], ["training_hrf_id"]
        ),
        npz_artifact(
            base + "_loss.npz",
            sse=outer["run_sse"],
            sst=outer["run_sst"],
            train_run_intercepts=outer["train_run_intercepts"],
            train_run_predictor_means=outer["train_run_predictor_means"],
            scoring_offsets=outer["scoring_offsets"],
            train_run_labels=np.asarray(outer["train_run_labels"]),
            test_run_labels=np.asarray(outer["test_run_labels"]),
        ),
        json_artifact(base + "_provenance.json", outer["provenance"]),
        json_artifact(
            base + "_metadata.json",
            dict(
                **encoding_metadata(outer["encoding_mode"]),
                task_coefficient="pooled_training_beta_mean",
                train_run_intercept_coordinates="raw_predictors",
                predictions_include_scoring_offsets=False,
                train_run_labels=outer["train_run_labels"],
                test_run_labels=outer["test_run_labels"],
                ridge_alpha=outer["ridge_alpha"],
                predictor_names=["task", "trial_type", "response_time"],
                predictor_means=outer["predictor_means"].tolist(),
                trial_masks=[m.tolist() for m in outer["trial_masks"]],
                trial_mask_run_labels=[r.label for r in runs],
                validation_target=(
                    "fixed_ols_betas"
                    if fractional
                    else "selected_penalty_regularized_betas"
                ),
                tuning_analysis_fingerprint=outer.get("tuning_analysis_fingerprint"),
                **(
                    dict(
                        fraction_norm_basis=NORM_BASIS,
                        alpha_run_labels=[r.label for r in runs],
                        alpha_interpretation="per-run conversion of the fixed training-selected fraction",
                    )
                    if fractional
                    else {}
                ),
                interpretation=(
                    "Within-run prediction of target beta variation; held-out means removed for scoring only"
                    if outer["encoding_mode"] == "within_run"
                    else "Absolute held-out encoding prediction of held-out beta targets"
                ),
            ),
        ),
    ]
    parameter_path = f"{stem}_space-fsLR_den-91k_desc-notebook{descriptor}_stat-hrfparameters.dscalar.nii"
    artifacts.append(parameter_artifact(brain, library, ids, parameter_path))
    if fractional:
        artifacts.extend(fraction_fit_artifacts(stem, brain, descriptor, outer, runs))
    return [*artifacts, *_prediction_artifacts(stem, brain, descriptor, runs, outer)]


def _predictor_table(runs, predictors):
    tables = []
    for r, (run, predictor) in enumerate(zip(runs, predictors, strict=True)):
        table = predictor.reset_index(drop=True).copy()
        table.insert(0, "event_index", np.arange(len(table)))
        table.insert(
            0, "trial_id", [f"{run.label}_trial-{j + 1:04d}" for j in range(len(table))]
        )
        table.insert(0, "run_label", run.label)
        table.insert(0, "run_index", r)
        table["usable"] = np.isfinite(predictor.to_numpy()).all(axis=1)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def ridge_artifacts(stem, brain, runs, results, library):
    """Return one publishable group; final beta images use existing artifacts."""
    table = tuning_table(results)
    first = next(iter(results.values()))
    kind = "FractionalCV" if "ridge_fraction" in first["final"] else "RidgeCV"
    artifacts = [
        table_artifact(
            f"{stem}_desc-notebook{kind}_predictors.tsv",
            _predictor_table(runs, first["predictors"]),
        ),
        figure_artifact(
            f"{stem}_desc-notebook{kind}_tuning.png", tuning_figure(results)
        ),
        json_artifact(
            f"{stem}_desc-notebook{kind}_metadata.json",
            {mode: r["provenance"] for mode, r in results.items()},
        ),
    ]
    for mode, result in results.items():
        for scope, tuned in result["tuning"].items():
            artifacts.extend(_tuning_artifacts(stem, brain, mode, scope, tuned, table))
        for scope, outer in result["evaluation"].items():
            artifacts.extend(
                _outer_artifacts(stem, brain, runs, mode, scope, outer, library)
            )
    return artifacts
