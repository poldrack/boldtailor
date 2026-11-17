"""CIFTI scalar map, design, table, and figure artifact builders."""

from pathlib import Path

import numpy as np

from boldtailor.cifti import scalar_artifact
from boldtailor.design import expand_events
from boldtailor.diagnostics import ONE_SAMPLE_T_NAMES as ACTIVATION_MAP_NAMES
from boldtailor.reliability import CORRELATION_NAMES, curve_correlations
from boldtailor.workflow.artifacts import (
    figure_artifact,
    json_artifact,
    npz_artifact,
    parameter_artifact,
    scalar_map,
    table_artifact,
)
from boldtailor.workflow.hrf_maps import (
    all_selection_maps,
    selection_maps,
    selection_names,
)
from boldtailor.workflow.inputs import run_summary
from boldtailor.workflow.manifest import (
    alpha_tuning_metadata,
    fraction_tuning_metadata,
    outer_metadata,
    selection_records,
)
from boldtailor.workflow.tables import predictor_table, tuning_table

R2_NAMES = ["full_r2", "confounds_r2", "task_delta_r2"]
PREDICTION_NAMES = ["selected_test_r2", "canonical_test_r2", "delta_test_r2"]


def _map(settings, brain, descriptor, statistic, values, names):
    return scalar_map(
        settings.stem,
        settings.space_entity,
        brain,
        descriptor,
        statistic,
        values,
        names,
    )


def _map_path(settings, descriptor, statistic):
    return f"{settings.stem}_{settings.space_entity}_desc-{descriptor}_stat-{statistic}.dscalar.nii"


def _run_path(settings, run, descriptor, suffix):
    directory = str(Path(settings.stem).parent)
    return f"{directory}/{run.inputs.stem}_{settings.space_entity}_desc-{descriptor}_{suffix}.dscalar.nii"


def _title(name):
    return "".join(part.title() for part in name.split("_"))


def _design_artifact(settings, descriptor, result, runs):
    arrays = {}
    for (i, hrf), frame in sorted(result.designs.items()):
        key = f"{runs[i].label}_hrf-{hrf:04d}"
        arrays[key] = frame.to_numpy(dtype=float)
        arrays[key + "_columns"] = np.asarray(frame.columns, dtype=str)
        arrays[key + "_frame_times"] = runs[i].frame_times
    return npz_artifact(f"{settings.stem}_desc-{descriptor}_designs.npz", **arrays)


def _fit_artifacts(settings, brain, descriptor, result, runs):
    return [
        _map(settings, brain, descriptor, "rsquared", result.r2, R2_NAMES),
        _design_artifact(settings, descriptor, result, runs),
        json_artifact(
            f"{settings.stem}_desc-{descriptor}_provenance.json", result.provenance
        ),
    ]


def glm_artifacts(settings, brain, glms, runs, task_model):
    """R², designs, provenance and contrast maps per GLM, plus their comparison."""
    names = list(task_model.regressor_names)
    artifacts = []
    for descriptor, result in glms.items():
        artifacts.extend(_fit_artifacts(settings, brain, descriptor, result, runs))
        for stat in ("effects", "variances", "t", "z"):
            artifacts.append(
                _map(settings, brain, descriptor, stat, getattr(result, stat), names)
            )
    if {"CanonicalGLM", "OptimizedGLM"} <= set(glms):
        difference = glms["OptimizedGLM"].r2[0] - glms["CanonicalGLM"].r2[0]
        artifacts.append(
            _map(
                settings,
                brain,
                "GLMComparison",
                "deltarsquared",
                difference[None],
                ["optimized_minus_canonical_full_r2"],
            )
        )
    return artifacts


def _library_artifacts(settings, library):
    stem = settings.stem
    return [
        table_artifact(f"{stem}_desc-HRF_library.tsv", library.parameter_table),
        npz_artifact(
            f"{stem}_desc-HRF_library.npz", times=library.times, curves=library.curves
        ),
    ]


def _selection_artifacts(settings, brain, library, maps, name):
    descriptor = "HRF" + name.title()
    return [
        _map(
            settings,
            brain,
            descriptor,
            "selection",
            maps[name],
            selection_names(settings.hrf_method),
        ),
        parameter_artifact(
            brain,
            library,
            maps[name][0],
            _map_path(settings, descriptor, "hrfparameters"),
        ),
    ]


def _split_artifacts(settings, brain, library, maps):
    correlations = curve_correlations(library, maps["odd"][0], maps["even"][0])
    artifacts = [
        _map(
            settings,
            brain,
            "HRFReliability",
            "curvecorrelation",
            correlations,
            list(CORRELATION_NAMES),
        )
    ]
    for name in ("odd", "even"):
        artifacts.extend(_selection_artifacts(settings, brain, library, maps, name))
    for name in ("odd_to_even", "even_to_odd"):
        artifacts.append(
            _map(
                settings,
                brain,
                "HRF" + _title(name),
                "prediction",
                maps[name],
                PREDICTION_NAMES,
            )
        )
    return artifacts


def hrf_artifacts(settings, brain, selections, library, *, include_splits=True):
    """All-run selection, library and provenance; odd/even splits on request."""
    maps = (
        selection_maps(selections, len(brain))
        if include_splits
        else all_selection_maps(selections, len(brain))
    )
    scopes = ("all", "odd", "even") if include_splits else ("all",)
    artifacts = [
        *_library_artifacts(settings, library),
        *_selection_artifacts(settings, brain, library, maps, "all"),
        json_artifact(
            f"{settings.stem}_desc-HRF_provenance.json",
            selection_records(selections, scopes),
        ),
    ]
    if include_splits:
        artifacts.extend(_split_artifacts(settings, brain, library, maps))
    return artifacts


def _beta_maps(settings, brain, descriptor, result, runs):
    artifacts = []
    for i, run in enumerate(runs):
        ids = result.trial_table.query("run_index == @i").trial_id.tolist()
        path = _run_path(settings, run, descriptor, "betas")
        artifacts.append(scalar_artifact(path, brain, result.betas[i], ids))
    return artifacts


def _rt_map(settings, brain, descriptor, rt):
    values = np.stack([rt[scope] for scope in ("all", "odd", "even")])
    names = ["all_runs", "odd_runs", "even_runs"]
    return _map(settings, brain, descriptor, "rtcorrelation", values, names)


def fraction_fit_artifacts(settings, brain, descriptor, result, runs):
    return [
        _map(
            settings,
            brain,
            descriptor,
            "ridgefraction",
            result.ridge_fraction[None],
            ["selected_ridge_fraction"],
        ),
        _map(
            settings,
            brain,
            descriptor,
            "ridgealpha",
            result.run_ridge_alphas,
            [r.label for r in runs],
        ),
    ]


def beta_model_artifacts(settings, brain, runs, model, library):
    """Fit, beta and RT maps for one model; tuning and outer splits when tuned."""
    name, fit = model.name, model.fit
    artifacts = _fit_artifacts(settings, brain, name, fit, runs)
    if model.fractional:
        artifacts.extend(fraction_fit_artifacts(settings, brain, name, fit, runs))
    artifacts.append(
        table_artifact(f"{settings.stem}_desc-{name}_trials.tsv", fit.trial_table)
    )
    artifacts.extend(_beta_maps(settings, brain, name, fit, runs))
    if fit.rt is not None:
        artifacts.append(_rt_map(settings, brain, name, fit.rt))
    if model.tuning:
        artifacts.extend(_tuned_artifacts(settings, brain, runs, model, library))
    return artifacts


def _tuned_artifacts(settings, brain, runs, model, library):
    table = tuning_table({model.name: model})
    mode = model.hrf.capitalize()
    base = f"{settings.stem}_desc-{model.name}"
    artifacts = [
        table_artifact(
            base + "_predictors.tsv", predictor_table(runs, model.predictors)
        ),
        json_artifact(base + "_metadata.json", model.cv_provenance),
    ]
    for scope, tuned in model.tuning.items():
        artifacts.extend(_tuning_artifacts(settings, brain, mode, scope, tuned, table))
    for scope, outer in model.evaluation.items():
        artifacts.extend(
            _outer_artifacts(settings, brain, runs, mode, scope, outer, library)
        )
    return artifacts


def _tuning_maps(settings, brain, descriptor, tuned, grid_names):
    scores, selected = tuned.scores, tuned.selection
    ids = np.where(scores.fold_hrf_indices >= 0, scores.fold_hrf_indices, np.nan)
    return [
        _map(settings, brain, descriptor, "encodingcvr2", scores.cv_r2, grid_names),
        _map(
            settings,
            brain,
            descriptor,
            "scoringmask",
            selected.scoring_mask[None],
            ["common_scoring_mask"],
        ),
        _map(
            settings,
            brain,
            descriptor,
            "foldhrfindex",
            ids,
            [f"validation_{label}_training_hrf" for label in scores.run_labels],
        ),
    ]


def _tuning_files(settings, descriptor, mode, scope, tuned, table, meta):
    base = f"{settings.stem}_desc-{descriptor}"
    scores = tuned.scores
    return [
        npz_artifact(base + "_folds.npz", sse=scores.fold_sse, sst=scores.fold_sst),
        table_artifact(
            base + "_scores.tsv",
            table.loc[(table["mode"] == mode) & (table.scope == scope)],
        ),
        json_artifact(base + "_provenance.json", tuned.provenance.to_dict()),
        json_artifact(base + "_metadata.json", meta),
    ]


def _fraction_selection_maps(settings, brain, descriptor, selected):
    return [
        _map(
            settings,
            brain,
            descriptor,
            "ridgefraction",
            selected.ridge_fraction[None],
            ["selected_ridge_fraction"],
        ),
        _map(
            settings,
            brain,
            descriptor,
            "selectedencodingr2",
            selected.selected_r2[None],
            ["selected_inner_cv_r2"],
        ),
    ]


def _tuning_artifacts(settings, brain, mode, scope, tuned, table):
    fractional = tuned.fractional
    kind, unit = ("FractionalCV", "fraction") if fractional else ("RidgeCV", "alpha")
    descriptor = f"{mode}{kind}{scope.title()}"
    grid = [f"encoding_inner_cv_r2_{unit}-{v:g}" for v in tuned.scores.grid]
    artifacts = _tuning_maps(settings, brain, descriptor, tuned, grid)
    if fractional:
        artifacts.extend(
            _fraction_selection_maps(settings, brain, descriptor, tuned.selection)
        )
    meta = (
        fraction_tuning_metadata(tuned) if fractional else alpha_tuning_metadata(tuned)
    )
    return artifacts + _tuning_files(
        settings, descriptor, mode, scope, tuned, table, meta
    )


def _prediction_artifacts(settings, brain, descriptor, runs, outer):
    by_label = {r.label: r for r in runs}
    artifacts = []
    for i, label in enumerate(outer.test_run_labels):
        run = by_label[label]
        names = [f"{label}_trial-{j + 1:04d}" for j in range(len(run.events))]
        for suffix in ("predictions", "targets"):
            path = _run_path(settings, run, descriptor, suffix)
            artifacts.append(
                scalar_artifact(path, brain, getattr(outer, suffix)[i], names)
            )
    return artifacts


def _outer_maps(settings, brain, descriptor, scope, outer, ids):
    return [
        _map(
            settings,
            brain,
            descriptor,
            "encodingpredictionr2",
            outer.encoding_r2[None],
            [f"encoding_outer_r2_{scope}"],
        ),
        _map(
            settings,
            brain,
            descriptor,
            "coefficients",
            outer.coefficients,
            list(outer.coefficient_names),
        ),
        _map(settings, brain, descriptor, "hrfindex", ids[None], ["training_hrf_id"]),
    ]


def _outer_loss(base, outer):
    return npz_artifact(
        base + "_loss.npz",
        sse=outer.run_sse,
        sst=outer.run_sst,
        train_run_intercepts=outer.train_run_intercepts,
        train_run_predictor_means=outer.train_run_predictor_means,
        scoring_offsets=outer.scoring_offsets,
        train_run_labels=np.asarray(outer.train_run_labels),
        test_run_labels=np.asarray(outer.test_run_labels),
    )


def _outer_artifacts(settings, brain, runs, mode, scope, outer, library):
    fractional = outer.fractional
    kind = "FractionalCV" if fractional else "RidgeCV"
    descriptor = mode + kind + _title(scope)
    base = f"{settings.stem}_desc-{descriptor}"
    ids = np.where(outer.hrf_indices >= 0, outer.hrf_indices, np.nan)
    artifacts = [
        *_outer_maps(settings, brain, descriptor, scope, outer, ids),
        _outer_loss(base, outer),
        json_artifact(base + "_provenance.json", outer.provenance),
        json_artifact(base + "_metadata.json", outer_metadata(runs, outer)),
        parameter_artifact(
            brain, library, ids, _map_path(settings, descriptor, "hrfparameters")
        ),
    ]
    if fractional:
        artifacts.extend(
            fraction_fit_artifacts(settings, brain, descriptor, outer, runs)
        )
    return [
        *artifacts,
        *_prediction_artifacts(settings, brain, descriptor, runs, outer),
    ]


def activation_artifacts(settings, brain, activation):
    return [
        _map(
            settings,
            brain,
            descriptor,
            "activation",
            np.stack([result[name] for name in ACTIVATION_MAP_NAMES]),
            list(ACTIVATION_MAP_NAMES),
        )
        for descriptor, result in activation.items()
    ]


def input_artifacts(settings, runs, task_model):
    """Run summary plus the expanded events and retained confounds per run."""
    stem = settings.stem
    artifacts = [
        table_artifact(
            f"{stem}_desc-boldtailor_runs.tsv", run_summary(runs, task_model)
        )
    ]
    directory = str(Path(stem).parent)
    for run in runs:
        base = f"{directory}/{run.inputs.stem}_desc-boldtailor"
        confounds = run.confounds.copy()
        confounds.insert(0, "frame_time", run.frame_times)
        confounds.insert(0, "original_frame", run.retained_frames)
        events = expand_events(run.events, task_model, run.number)
        artifacts.append(table_artifact(base + "_events.tsv", events))
        artifacts.append(table_artifact(base + "_confounds.tsv", confounds))
    return artifacts


def figure_name(settings, name):
    """A figure's PNG path relative to the derivative root."""
    return f"{settings.stem}_desc-{name}_plot.png"


def figure_artifacts(settings, figures):
    return [
        figure_artifact(figure_name(settings, name), figure)
        for name, figure in (figures or {}).items()
    ]
