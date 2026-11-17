"""JSON metadata and provenance manifests for workflow outputs."""

import numpy as np

from boldtailor.diagnostics import ONE_SAMPLE_T_NAMES as ACTIVATION_MAP_NAMES
from boldtailor.fractional_ridge import NORM_BASIS
from boldtailor.model import HRF_NORMALIZATION
from boldtailor.reliability import CORRELATION_NAMES
from boldtailor.trial_encoding import encoding_metadata
from boldtailor.workflow.hrf_maps import reml_record
from boldtailor.workflow.inputs import selection_task_model
from boldtailor.workflow.tables import (
    _scope_selection,
    _tuned,
    hrf_boundary_summary,
    ridge_boundary_summary,
)


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
    record = dict(
        provenance=result.provenance.to_dict(),
        selection_provenance=selected.provenance.to_dict(),
        hrf_indices=selected.hrf_indices.tolist(),
    )
    if selected.method == "reml":
        record["reml"] = reml_record(selected)
    return record


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
        encoding_predictors=list(task_model.regressor_names),
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
        f"leave-one-run-out task-model prediction over {names}"
        f"{_indicator_clause(task_model.profiled_names)}; pooled held-out error "
        "over confound-adjusted energy"
    )


def _indicator_clause(profiled):
    if not profiled:
        return ""
    if tuple(profiled) == ("missing_response_time",):
        return "; missing-RT indicator profiled per run when present"
    return f"; {', '.join(profiled)} indicators profiled per run when present"


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


def _rt_check(include_rt, has_rt, beta_models):
    """RT correlations are computed only when RT is a modulator."""
    if not has_rt:
        return {}
    tuned = bool(_tuned(beta_models))
    return dict(rt_check=_rt_check_description(include_rt, tuned))


_CATEGORICAL_CODING = (
    "One 0/1 indicator per non-reference level, uncentered; each coefficient "
    "is the level response minus the reference response"
)


def _task_model_description(task_model):
    columns = [m.column for m in task_model.modulators]
    modulated = f" modulated by {', '.join(columns)}" if columns else ""
    return (
        f"One task regressor per presentation{modulated}. Modulators are "
        "uncentered; the task contrast is the response at numeric modulator "
        "value zero and at each categorical modulator's reference level."
    )


def _categorical_prose(task_model):
    return {
        m.column: dict(
            levels=list(m.levels), reference=m.reference, coding=_CATEGORICAL_CODING
        )
        for m in task_model.modulators
        if m.kind == "categorical"
    }


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
        task=(
            "One unit per presentation; the response with every numeric modulator "
            "at zero, every categorical modulator at its reference level, and "
            "every missing-value indicator at zero"
        ),
    )
    if "response_time" in columns:
        prose.update(_response_time_prose())
    categorical = _categorical_prose(task_model)
    if categorical:
        prose["categorical_modulators"] = categorical
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


def _json_bound_rows(selections):
    """Bound rows with an undefined fraction (no custom picks) as null."""
    rows = hrf_boundary_summary(selections)
    if rows is None:
        return None
    return [
        {**row, "fraction_flagged": None} if np.isnan(row["fraction_flagged"]) else row
        for row in rows
    ]


def _library_metadata(library, selections):
    return dict(
        hrf_boundary_summary=_json_bound_rows(selections),
        hrf_boundary="Fraction of custom picks (ID > 0) within 2% of the library box width of each parameter edge (null without custom picks); see HrfSelectionResult.parameter_bound_table",
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
    artifacts=None,
    notes=(),
):
    """The settings file: analysis description plus what ran and what was skipped.

    ``artifacts`` lists every path this run publishes for the session (not the
    shared dataset description); a later overwrite removes only those files.
    """
    has_rt = "response_time" in task_model.regressor_names
    include_rt = settings.hrf_selection_rt and has_rt
    return dict(
        regressors=list(task_model.regressor_names),
        task_model=task_model.to_dict(),
        task_model_fingerprint=task_model.fingerprint,
        **_model_prose(task_model),
        **_design_metadata(runs),
        hrf_selection=_selection_description(
            selection_task_model(task_model, include_rt)
        ),
        **_rt_check(include_rt, has_rt, beta_models),
        ridge_cv=_ridge_metadata(settings, task_model, beta_models),
        beta_activation=_activation_metadata() if activation else None,
        **_library_metadata(library, selections),
        runs=[r.label for r in runs],
        skipped=list(skipped),
        notes=list(notes),
        report=report,
        artifacts=sorted(artifacts or ()),
        settings=settings.to_dict(),
    )
