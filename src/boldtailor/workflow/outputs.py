"""Publish every workflow stage's maps, designs, tables, and provenance together."""

from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from boldtailor.cifti import scalar_artifact
from boldtailor.design import expand_events
from boldtailor.diagnostics import ONE_SAMPLE_T_NAMES as ACTIVATION_MAP_NAMES
from boldtailor.fractional_ridge import NORM_BASIS
from boldtailor.model import HRF_NORMALIZATION
from boldtailor.publication import Artifact, publish_artifact_set
from boldtailor.reliability import CORRELATION_NAMES, curve_correlations
from boldtailor.ridge_results import FractionSelection
from boldtailor.trial_encoding import encoding_metadata
from boldtailor.workflow.analysis import selection_maps
from boldtailor.workflow.artifacts import (
    dataset_description,
    figure_artifact,
    json_artifact,
    npz_artifact,
    parameter_artifact,
    scalar_map,
    table_artifact,
)
from boldtailor.workflow.files import input_paths
from boldtailor.workflow.inputs import run_summary, selection_task_model

R2_NAMES = ["full_r2", "confounds_r2", "task_delta_r2"]
SELECTION_NAMES = ["hrf_id", "selected_cv_r2", "canonical_cv_r2", "delta_cv_r2"]
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


def check_output(settings):
    """Refuse to replace this subject/session/task's outputs unless told to."""
    if settings.existing_results == "overwrite":
        return None
    directory = settings.output_dir / settings.subject / settings.session / "func"
    pattern = f"{settings.subject}_{settings.session}_task-{settings.task}_*"
    if any(directory.glob(pattern)):
        raise FileExistsError(
            f"boldtailor outputs already exist in {directory}; set "
            "existing_results to overwrite, or choose a new output_dir"
        )
    return None


def _design_artifact(settings, descriptor, result, runs):
    arrays = {}
    for (i, hrf), frame in sorted(result["designs"].items()):
        key = f"{runs[i].label}_hrf-{hrf:04d}"
        arrays[key] = frame.to_numpy(dtype=float)
        arrays[key + "_columns"] = np.asarray(frame.columns, dtype=str)
        arrays[key + "_frame_times"] = runs[i].frame_times
    return npz_artifact(f"{settings.stem}_desc-{descriptor}_designs.npz", **arrays)


def _fit_artifacts(settings, brain, descriptor, result, runs):
    return [
        _map(settings, brain, descriptor, "rsquared", result["r2"], R2_NAMES),
        _design_artifact(settings, descriptor, result, runs),
        json_artifact(
            f"{settings.stem}_desc-{descriptor}_provenance.json", result["provenance"]
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
                _map(settings, brain, descriptor, stat, result[stat], names)
            )
    if {"CanonicalGLM", "OptimizedGLM"} <= set(glms):
        difference = glms["OptimizedGLM"]["r2"][0] - glms["CanonicalGLM"]["r2"][0]
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
        _map(settings, brain, descriptor, "selection", maps[name], SELECTION_NAMES),
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
    maps = selection_maps(selections, len(brain))
    scopes = ("all", "odd", "even") if include_splits else ("all",)
    artifacts = [
        *_library_artifacts(settings, library),
        *_selection_artifacts(settings, brain, library, maps, "all"),
        json_artifact(
            f"{settings.stem}_desc-HRF_provenance.json",
            _selection_records(selections, scopes),
        ),
    ]
    if include_splits:
        artifacts.extend(_split_artifacts(settings, brain, library, maps))
    return artifacts


def _selection_records(selections, scopes):
    records = []
    for indices, bundle in selections.items():
        records.append(
            dict(
                grayordinate_indices=[int(i) for i in indices],
                scopes={name: _scope_record(bundle, name) for name in scopes},
            )
        )
    return records


def _scope_record(bundle, name):
    result = bundle[name]
    selected = _scope_selection(bundle, name)
    return dict(
        provenance=result.provenance.to_dict(),
        selection_provenance=selected.provenance.to_dict(),
        hrf_indices=selected.hrf_indices.tolist(),
    )


def _beta_maps(settings, brain, descriptor, result, runs):
    artifacts = []
    for i, run in enumerate(runs):
        ids = result["trial_table"].query("run_index == @i").trial_id.tolist()
        path = _run_path(settings, run, descriptor, "betas")
        artifacts.append(scalar_artifact(path, brain, result["betas"][i], ids))
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
            result["ridge_fraction"][None],
            ["selected_ridge_fraction"],
        ),
        _map(
            settings,
            brain,
            descriptor,
            "ridgealpha",
            result["run_ridge_alphas"],
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
        table_artifact(f"{settings.stem}_desc-{name}_trials.tsv", fit["trial_table"])
    )
    artifacts.extend(_beta_maps(settings, brain, name, fit, runs))
    if fit["rt"] is not None:
        artifacts.append(_rt_map(settings, brain, name, fit["rt"]))
    if model.tuning:
        artifacts.extend(_tuned_artifacts(settings, brain, runs, model, library))
    return artifacts


def _tuned_artifacts(settings, brain, runs, model, library):
    table = tuning_table({model.name: model})
    mode = model.hrf.capitalize()
    base = f"{settings.stem}_desc-{model.name}"
    artifacts = [
        table_artifact(
            base + "_predictors.tsv", _predictor_table(runs, model.predictors)
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


def _tuned(beta_models):
    return [m for m in beta_models.values() if m.tuning]


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


def _alpha_rows(mode, scope, tuned):
    selected = tuned["selection"]
    return [
        dict(
            mode=mode,
            scope=scope,
            alpha=alpha,
            objective=float(value),
            selected=alpha == selected.ridge_alpha,
            percentile=selected.percentile,
            grayordinates=int(selected.scoring_mask.sum()),
        )
        for alpha, value in zip(selected.alphas, selected.objective_scores, strict=True)
    ]


def tuning_table(beta_models):
    """One row per candidate, scope, and tuned model (mode is the HRF model)."""
    rows = []
    for model in _tuned(beta_models):
        mode = model.hrf.capitalize()
        for scope, tuned in model.tuning.items():
            fractional = isinstance(tuned["selection"], FractionSelection)
            rows.extend(
                (tuning_rows if fractional else _alpha_rows)(mode, scope, tuned)
            )
    return pd.DataFrame(rows)


def plot_tuning(axis, mode, tuning):
    for scope, tuned in tuning.items():
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


def _plot_alpha_tuning(axis, mode, tuning):
    positive = []
    for scope, tuned in tuning.items():
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


def tuning_figure(beta_models):
    """One tuning-curve panel per tuned model."""
    tuned = _tuned(beta_models)
    figure = Figure(figsize=(6 * len(tuned), 4), layout="constrained")
    axes = figure.subplots(1, len(tuned), squeeze=False)[0]
    for axis, model in zip(axes, tuned, strict=True):
        plot = plot_tuning if model.fractional else _plot_alpha_tuning
        plot(axis, model.hrf.capitalize(), model.tuning)
    return figure


def _tuning_maps(settings, brain, descriptor, tuned, label, grid_names):
    scores, selected = tuned["scores"], tuned["selection"]
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


def _tuning_files(settings, descriptor, mode, scope, tuned, table, metadata):
    base = f"{settings.stem}_desc-{descriptor}"
    scores = tuned["scores"]
    return [
        npz_artifact(base + "_folds.npz", sse=scores.fold_sse, sst=scores.fold_sst),
        table_artifact(
            base + "_scores.tsv",
            table.loc[(table["mode"] == mode) & (table.scope == scope)],
        ),
        json_artifact(base + "_provenance.json", tuned["provenance"].to_dict()),
        json_artifact(base + "_metadata.json", metadata),
    ]


def _shared_tuning_metadata(tuned):
    scores, selected = tuned["scores"], tuned["selection"]
    return dict(
        run_labels=list(scores.run_labels),
        scoring_grayordinates=int(selected.scoring_mask.sum()),
        excluded_grayordinates=int((~selected.scoring_mask).sum()),
        trial_masks=[m.tolist() for m in scores.trial_masks],
        score="1 - sum(validation SSE) / sum(within-run validation SST)",
        selection_statistic=True,
    )


def _alpha_tuning_metadata(tuned):
    scores, selected = tuned["scores"], tuned["selection"]
    return dict(
        **_shared_tuning_metadata(tuned),
        alphas=list(scores.grid),
        selected_alpha=selected.ridge_alpha,
        percentile=selected.percentile,
        objective_scores=selected.objective_scores.tolist(),
        spatial_reduction="linear percentile over the common finite scoring mask",
        tie_rule="smallest alpha within 1e-12 of the maximum objective",
        validation_target="candidate_regularized_betas",
    )


def _fraction_tuning_metadata(tuned):
    return dict(
        **_shared_tuning_metadata(tuned),
        fractions=list(tuned["scores"].grid),
        selection_rule="maximum_encoding_r2_per_grayordinate",
        validation_target="fixed_ols_betas",
        fraction_norm_basis=NORM_BASIS,
        percentile_role="descriptive_only",
        summary_percentile=tuned["summary_percentile"],
        tie_rule="largest fraction within 1e-12 of the maximum score",
    )


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
    fractional = isinstance(tuned["selection"], FractionSelection)
    kind, unit = ("FractionalCV", "fraction") if fractional else ("RidgeCV", "alpha")
    descriptor = f"{mode}{kind}{scope.title()}"
    grid = [f"encoding_inner_cv_r2_{unit}-{v:g}" for v in tuned["scores"].grid]
    artifacts = _tuning_maps(settings, brain, descriptor, tuned, scope, grid)
    if fractional:
        artifacts.extend(
            _fraction_selection_maps(settings, brain, descriptor, tuned["selection"])
        )
    metadata = (
        _fraction_tuning_metadata(tuned)
        if fractional
        else _alpha_tuning_metadata(tuned)
    )
    return artifacts + _tuning_files(
        settings, descriptor, mode, scope, tuned, table, metadata
    )


def _prediction_artifacts(settings, brain, descriptor, runs, outer):
    by_label = {r.label: r for r in runs}
    artifacts = []
    for i, label in enumerate(outer["test_run_labels"]):
        run = by_label[label]
        names = [f"{label}_trial-{j + 1:04d}" for j in range(len(run.events))]
        for suffix in ("predictions", "targets"):
            path = _run_path(settings, run, descriptor, suffix)
            artifacts.append(scalar_artifact(path, brain, outer[suffix][i], names))
    return artifacts


def _outer_maps(settings, brain, descriptor, scope, outer, ids):
    return [
        _map(
            settings,
            brain,
            descriptor,
            "encodingpredictionr2",
            outer["encoding_r2"][None],
            [f"encoding_outer_r2_{scope}"],
        ),
        _map(
            settings,
            brain,
            descriptor,
            "coefficients",
            outer["coefficients"],
            list(outer["coefficient_names"]),
        ),
        _map(settings, brain, descriptor, "hrfindex", ids[None], ["training_hrf_id"]),
    ]


def _outer_loss(base, outer):
    return npz_artifact(
        base + "_loss.npz",
        sse=outer["run_sse"],
        sst=outer["run_sst"],
        train_run_intercepts=outer["train_run_intercepts"],
        train_run_predictor_means=outer["train_run_predictor_means"],
        scoring_offsets=outer["scoring_offsets"],
        train_run_labels=np.asarray(outer["train_run_labels"]),
        test_run_labels=np.asarray(outer["test_run_labels"]),
    )


def _fraction_outer_metadata(runs):
    return dict(
        fraction_norm_basis=NORM_BASIS,
        alpha_run_labels=[r.label for r in runs],
        alpha_interpretation="per-run conversion of the fixed training-selected fraction",
    )


def _outer_metadata(runs, outer, fractional):
    within = outer["encoding_mode"] == "within_run"
    return dict(
        **encoding_metadata(outer["encoding_mode"]),
        task_coefficient="pooled_training_beta_mean",
        train_run_intercept_coordinates="raw_predictors",
        predictions_include_scoring_offsets=False,
        train_run_labels=outer["train_run_labels"],
        test_run_labels=outer["test_run_labels"],
        ridge_alpha=outer["ridge_alpha"],
        predictor_names=list(outer["coefficient_names"]),
        predictor_means=outer["predictor_means"].tolist(),
        trial_masks=[m.tolist() for m in outer["trial_masks"]],
        trial_mask_run_labels=[r.label for r in runs],
        validation_target=(
            "fixed_ols_betas" if fractional else "selected_penalty_regularized_betas"
        ),
        tuning_analysis_fingerprint=outer.get("tuning_analysis_fingerprint"),
        **(_fraction_outer_metadata(runs) if fractional else {}),
        interpretation=(
            "Within-run prediction of target beta variation; held-out means removed for scoring only"
            if within
            else "Absolute held-out encoding prediction of held-out beta targets"
        ),
    )


def _outer_artifacts(settings, brain, runs, mode, scope, outer, library):
    fractional = "ridge_fraction" in outer
    kind = "FractionalCV" if fractional else "RidgeCV"
    descriptor = mode + kind + _title(scope)
    base = f"{settings.stem}_desc-{descriptor}"
    ids = np.where(outer["hrf_indices"] >= 0, outer["hrf_indices"], np.nan)
    artifacts = [
        *_outer_maps(settings, brain, descriptor, scope, outer, ids),
        _outer_loss(base, outer),
        json_artifact(base + "_provenance.json", outer["provenance"]),
        json_artifact(
            base + "_metadata.json", _outer_metadata(runs, outer, fractional)
        ),
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


def _boundary_fraction(selection):
    flags = np.asarray(selection.at_boundary, dtype=bool)
    if flags.ndim == 0:
        return float(flags)
    scored = flags[selection.scoring_mask]
    return float(scored.mean()) if scored.size else float("nan")


def ridge_boundary_summary(beta_models):
    """Fraction of scored grayordinates whose ridge winner is a grid endpoint."""
    return [
        dict(
            mode=model.hrf.capitalize(),
            scope=scope,
            fraction=_boundary_fraction(tuned["selection"]),
        )
        for model in _tuned(beta_models)
        for scope, tuned in model.tuning.items()
    ]


def _scope_selection(bundle, scope):
    return bundle[scope] if scope == "all" else bundle[scope].training_selection


def _pooled_bound_rows(picked, scope):
    counts = [int((p.hrf_indices > 0).sum()) for p in picked]
    tables = [p.parameter_bound_table() for p in picked]
    total = sum(counts)
    rows = []
    for i, row in tables[0].iterrows():
        flagged = sum(t.fraction_flagged[i] * n for t, n in zip(tables, counts) if n)
        fraction = flagged / total if total else float("nan")
        rows.append(
            dict(
                scope=scope,
                parameter=row.parameter,
                edge=row.edge,
                fraction_flagged=float(fraction),
                n_custom=total,
            )
        )
    return rows


def hrf_boundary_summary(selections):
    """``parameter_bound_table`` rows pooled over grayordinate blocks per scope."""
    if not selections:
        return None
    rows = []
    for scope in ("all", "odd", "even"):
        picked = [_scope_selection(b, scope) for b in selections.values()]
        rows.extend(_pooled_bound_rows(picked, scope))
    return rows


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


def _ridge_metadata(settings, task_model, beta_models):
    tuned = _tuned(beta_models)
    if not tuned:
        return None
    definition = tuned[0].cv_provenance
    return dict(
        **encoding_metadata(definition["encoding_mode"]),
        validation_target=definition["validation_target"],
        objective=definition["objective"],
        percentile=settings.ridge_percentile,
        percentile_role=definition.get("percentile_role", "selection_objective"),
        fraction_norm_basis=definition.get("fraction_norm_basis"),
        encoding_predictors=["task", *[m.column for m in task_model.modulators]],
        task="Pooled training beta mean, used as the prediction reference level",
        outer_splits="odd_to_even_and_even_to_odd",
        final_fit="Separate all-run tuning and refit; final RT correlations are descriptive",
        at_boundary_fraction=ridge_boundary_summary(beta_models),
        at_boundary="Winner at the shrinkage end (smallest fraction / largest alpha): extend the grid; winner at fraction 1.0 or alpha 0: no regularization preferred",
    )


def _activation_metadata():
    return dict(
        method="One-sample t test across finite trial betas pooled over runs",
        null_mean=0,
        assume_independent_trials=True,
        alternative="two-sided",
        weighting="Equal weight per finite trial; no within-run centering",
        multiple_comparison_correction=None,
        map_order=list(ACTIVATION_MAP_NAMES),
        undefined="NaN t/p for fewer than two trials or zero sample variance; all maps NaN if no finite trials",
        interpretation="Descriptive activation-style map relative to the fitted model baseline, not an explicit task-versus-rest contrast; trial covariance and HRF/ridge selection uncertainty are ignored",
    )


def _selection_description(task_model):
    names = ", ".join(task_model.regressor_names)
    return (
        f"leave-one-run-out task-model prediction over {names}; missing-RT "
        "indicator profiled per run when present; pooled held-out error over "
        "confound-adjusted energy"
    )


def _rt_check_description(include_rt, ridge_cv):
    if ridge_cv:
        base = (
            "RT/type tune CV ridge strength; final all-run RT correlations are "
            "descriptive. Outer test runs are excluded from HRF and penalty selection."
        )
        tail = (
            " RT also enters HRF selection."
            if include_rt
            else " RT never enters HRF selection."
        )
        return base + tail
    if include_rt:
        return (
            "Descriptive within-run-centered correlation; RT enters HRF selection as a "
            "task-model regressor, so all-run optimized correlations are not independent checks"
        )
    return (
        "Descriptive within-run-centered correlation, never used to select HRFs or "
        "fixed ridge strength; all-run optimized HRFs use both halves"
    )


def _task_model_description(task_model):
    columns = [m.column for m in task_model.modulators]
    modulated = f" modulated by {', '.join(columns)}" if columns else ""
    return (
        f"One task regressor per presentation{modulated}. Modulators are "
        "uncentered; the task contrast is the response at modulator value zero."
    )


def _response_time_prose():
    return dict(
        response_time="Positive finite seconds, uncentered; zero modulation for unavailable RT; coefficient per second",
        missing_response_time=dict(
            regressor="missing_response_time",
            unavailable="Nonfinite or nonpositive response_time",
            coding="One for unavailable RT, zero otherwise; omitted in complete runs",
            convolution="Same onset, duration, and HRF as the other task regressors",
            interpretation="Additive mean response difference for trials with unavailable RT, conditional on the other modulators",
            contrast_exported=False,
            task_delta_r2="Includes the missing-RT indicator among task predictors",
        ),
    )


def _model_prose(task_model):
    columns = {m.column for m in task_model.modulators}
    prose = dict(
        task_model_description=_task_model_description(task_model),
        task="One unit per presentation; the response with every modulator and missing-value indicator at zero",
    )
    if "response_time" in columns:
        prose.update(_response_time_prose())
    if "trial_type" in columns:
        prose["trial_type"] = (
            "Binary codes 0/1, uncentered; the task coefficient is the response "
            "on trial_type 0 trials"
        )
    return prose


def _curve_correlation_metadata(library):
    return dict(
        method="Pearson over HRF time samples, without temporal shifting",
        map_order=list(CORRELATION_NAMES),
        canonical_hrf_id=0,
        time_range_seconds=[float(library.times[0]), float(library.times[-1])],
        sample_interval_seconds=0.1,
        time_grid="Full stored library grid, including zero-padded tails of shorter curves",
        interpretation="Shape similarity, invariant to amplitude scale and offset; not BOLD prediction accuracy",
    )


def _design_metadata(runs):
    return dict(
        noise_model="ols",
        hrf_model="spm_or_selected_per_grayordinate",
        hrf_normalization=HRF_NORMALIZATION,
        orthogonalization=False,
        retained_scans=[len(r.frame_times) for r in runs],
        trimming="Leading nonsteady volumes removed; original acquisition times and event onsets retained",
        nuisance_columns=list(runs[0].confounds.columns) + ["constant"],
        high_pass="fMRIPrep cosines only; no additional drift basis",
        r2="1 - sum(run SSE) / sum(within-run SST), on retained scans; native signal units",
        glm_comparison="Descriptive in-sample optimized minus canonical full R²; not an independent validation of HRF selection",
        inference="Contrasts use equal-run fixed effects; conditional on selected HRFs, without selection uncertainty correction",
        split_prediction="Train HRF and mean amplitude on one half; freeze both for the other half; nuisance projection is conditional on each run",
        undefined="NaN for constant, undefined, or unprocessed grayordinates; CIFTI axis preserved",
    )


def _library_metadata(library, selections):
    return dict(
        hrf_boundary_summary=hrf_boundary_summary(selections),
        hrf_boundary="Fraction of custom picks (ID > 0) within 2% of the library box width of each parameter edge; see HrfSelectionResult.parameter_bound_table",
        library_candidates=len(library.candidates),
        library_fingerprint=library.fingerprint,
        peak_time="Argmax of each full HRF curve on a 0.1-second grid",
        hrf_curve_correlations=_curve_correlation_metadata(library),
    )


def metadata(
    runs,
    library,
    settings,
    task_model,
    *,
    beta_models,
    activation,
    selections,
    skipped,
    report,
):
    """The settings file: analysis description plus what ran and what was skipped."""
    include_rt = settings.hrf_selection_rt
    return dict(
        regressors=list(task_model.regressor_names),
        task_model=task_model.to_dict(),
        task_model_fingerprint=task_model.fingerprint,
        **_model_prose(task_model),
        **_design_metadata(runs),
        hrf_selection=_selection_description(
            selection_task_model(task_model, include_rt)
        ),
        rt_check=_rt_check_description(include_rt, bool(_tuned(beta_models))),
        ridge_cv=_ridge_metadata(settings, task_model, beta_models),
        beta_activation=_activation_metadata() if activation else None,
        **_library_metadata(library, selections),
        runs=[r.label for r in runs],
        skipped=list(skipped),
        report=report,
        settings=settings.to_dict(),
    )


def _report_path(settings):
    return f"{settings.subject}_{settings.session}_task-{settings.task}_report.html"


def _stage_artifacts(
    settings, runs, task_model, library, selections, glms, beta_models, **options
):
    brain = runs[0].image.header.get_axis(1)
    artifacts = glm_artifacts(settings, brain, glms, runs, task_model)
    if selections:
        artifacts.extend(
            hrf_artifacts(
                settings,
                brain,
                selections,
                library,
                include_splits=options["include_hrf_splits"],
            )
        )
    for model in beta_models.values():
        artifacts.extend(beta_model_artifacts(settings, brain, runs, model, library))
    if options["activation"]:
        artifacts.extend(activation_artifacts(settings, brain, options["activation"]))
    return artifacts + input_artifacts(settings, runs, task_model)


def _figure_artifacts(settings, figures):
    return [
        figure_artifact(f"{settings.stem}_desc-{name}_plot.png", figure)
        for name, figure in (figures or {}).items()
    ]


def _root_artifacts(settings, report, report_html):
    """The report, plus the dataset description unless another session wrote it."""
    artifacts = [] if report is None else [Artifact(report, report_html)]
    description = settings.output_dir / "dataset_description.json"
    if settings.existing_results == "overwrite" or not description.exists():
        artifacts.append(dataset_description("boldtailor"))
    return artifacts


def save_workflow(
    settings,
    runs,
    task_model,
    library,
    selections,
    glms,
    beta_models,
    *,
    figures,
    activation,
    skipped,
    report_html,
    include_hrf_splits=True,
):
    """Publish every stage, the settings file, figures, and the report at once.

    ``include_hrf_splits`` writes the odd/even HRF artifacts; pass whether the
    reliability stage ran.
    """
    report = None if report_html is None else _report_path(settings)
    artifacts = _stage_artifacts(
        settings,
        runs,
        task_model,
        library,
        selections,
        glms,
        beta_models,
        activation=activation,
        include_hrf_splits=include_hrf_splits,
    )
    artifacts.append(
        json_artifact(
            f"{settings.stem}_desc-boldtailor_metadata.json",
            metadata(
                runs,
                library,
                settings,
                task_model,
                beta_models=beta_models,
                activation=activation,
                selections=selections,
                skipped=skipped,
                report=report,
            ),
        )
    )
    artifacts.extend(_figure_artifacts(settings, figures))
    artifacts.extend(_root_artifacts(settings, report, report_html))
    return publish_artifact_set(
        settings.output_dir,
        artifacts,
        source_paths=[p for r in runs for p in input_paths(r.inputs)],
        overwrite=settings.existing_results == "overwrite",
    )
