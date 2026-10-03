"""Publish notebook maps, designs, trial metadata, and provenance together."""

from pathlib import Path

import numpy as np

from boldtailor.cifti import scalar_artifact
from boldtailor.design import expand_events
from boldtailor.diagnostics import ONE_SAMPLE_T_NAMES as ACTIVATION_MAP_NAMES
from boldtailor.model import HRF_NORMALIZATION
from boldtailor.publication import publish_artifact_set
from boldtailor.reliability import CORRELATION_NAMES, curve_correlations
from .workflow_artifacts import (
    figure_artifact,
    json_artifact,
    notebook_map,
    npz_artifact,
    parameter_artifact,
    table_artifact,
)
from boldtailor.workflow.files import input_paths
from .workflow_analysis import selection_maps
from .workflow_inputs import (
    NSD_TASK_MODEL,
    REGRESSORS,
    run_summary,
    selection_task_model,
)

R2_NAMES = ["full_r2", "confounds_r2", "task_delta_r2"]


def _stem(subject, session):
    return f"{subject}/{session}/func/{subject}_{session}_task-nsdcore"


def check_output(
    output, subject="sub-07", session="ses-nsd10", *, existing_results="error"
):
    """Return whether to reuse a saved analysis; otherwise authorize a fresh fit."""
    if existing_results not in ("error", "reuse", "overwrite"):
        raise ValueError("existing_results must be error, reuse, or overwrite")
    directory = Path(output) / subject / session / "func"
    exists = any(directory.glob(f"{subject}_{session}_task-nsdcore*desc-notebook*"))
    if exists and existing_results == "error":
        raise FileExistsError(
            f"Notebook outputs already exist in {directory}; set existing_results "
            "to reuse or overwrite, or choose a new output_root"
        )
    if exists and existing_results == "reuse":
        if not (
            Path(output) / f"{_stem(subject, session)}_desc-notebook_metadata.json"
        ).is_file():
            raise ValueError(
                "Saved notebook results are incomplete; choose overwrite to refit"
            )
        return True
    return False


def _design_artifact(stem, descriptor, result, runs):
    arrays = {}
    for (i, hrf), frame in sorted(result["designs"].items()):
        key = f"{runs[i].label}_hrf-{hrf:04d}"
        arrays[key] = frame.to_numpy(dtype=float)
        arrays[key + "_columns"] = np.asarray(frame.columns, dtype=str)
        arrays[key + "_frame_times"] = runs[i].frame_times
    return npz_artifact(f"{stem}_desc-notebook{descriptor}_designs.npz", **arrays)


def _fit_artifacts(stem, brain, descriptor, result, runs):
    return [
        notebook_map(stem, brain, descriptor, "rsquared", result["r2"], R2_NAMES),
        _design_artifact(stem, descriptor, result, runs),
        json_artifact(
            f"{stem}_desc-notebook{descriptor}_provenance.json", result["provenance"]
        ),
    ]


def _glm_artifacts(stem, brain, glms, runs):
    artifacts = []
    for descriptor, result in glms.items():
        artifacts.extend(_fit_artifacts(stem, brain, descriptor, result, runs))
        for stat in ("effects", "variances", "t", "z"):
            artifacts.append(
                notebook_map(
                    stem, brain, descriptor, stat, result[stat], list(REGRESSORS)
                )
            )
    difference = glms["OptimizedGLM"]["r2"][0] - glms["CanonicalGLM"]["r2"][0]
    artifacts.append(
        notebook_map(
            stem,
            brain,
            "GLMComparison",
            "deltarsquared",
            difference[None],
            ["optimized_minus_canonical_full_r2"],
        )
    )
    return artifacts


def _hrf_artifacts(stem, brain, selections, library):
    maps = selection_maps(selections, len(brain))
    artifacts = [
        table_artifact(f"{stem}_desc-notebookHRF_library.tsv", library.parameter_table),
        npz_artifact(
            f"{stem}_desc-notebookHRF_library.npz",
            times=library.times,
            curves=library.curves,
        ),
        notebook_map(
            stem,
            brain,
            "HRFReliability",
            "curvecorrelation",
            curve_correlations(library, maps["odd"][0], maps["even"][0]),
            list(CORRELATION_NAMES),
        ),
    ]
    for name in ("all", "odd", "even"):
        descriptor = "HRF" + name.title()
        artifacts.append(
            notebook_map(
                stem,
                brain,
                descriptor,
                "selection",
                maps[name],
                ["hrf_id", "selected_cv_r2", "canonical_cv_r2", "delta_cv_r2"],
            )
        )
        path = f"{stem}_space-fsLR_den-91k_desc-notebook{descriptor}_stat-hrfparameters.dscalar.nii"
        artifacts.append(parameter_artifact(brain, library, maps[name][0], path))
    for name in ("odd_to_even", "even_to_odd"):
        descriptor = "HRF" + "".join(part.title() for part in name.split("_"))
        artifacts.append(
            notebook_map(
                stem,
                brain,
                descriptor,
                "prediction",
                maps[name],
                ["selected_test_r2", "canonical_test_r2", "delta_test_r2"],
            )
        )
    records = _selection_records(selections)
    artifacts.append(json_artifact(f"{stem}_desc-notebookHRF_provenance.json", records))
    return artifacts


def _selection_records(selections):
    records = []
    for indices, bundle in selections.items():
        scopes = {}
        for name, result in bundle.items():
            selected = result if name == "all" else result.training_selection
            scopes[name] = dict(
                provenance=result.provenance.to_dict(),
                selection_provenance=selected.provenance.to_dict(),
                hrf_indices=selected.hrf_indices.tolist(),
            )
        records.append(
            dict(grayordinate_indices=[int(i) for i in indices], scopes=scopes)
        )
    return records


def _beta_artifacts(stem, brain, betas, runs):
    artifacts = []
    directory = str(Path(stem).parent)
    for descriptor, result in betas.items():
        artifacts.extend(_fit_artifacts(stem, brain, descriptor, result, runs))
        if "ridge_fraction" in result:
            from .fractional_outputs import fraction_fit_artifacts

            artifacts.extend(
                fraction_fit_artifacts(stem, brain, descriptor, result, runs)
            )
        artifacts.append(
            table_artifact(
                f"{stem}_desc-notebook{descriptor}_trials.tsv", result["trial_table"]
            )
        )
        for i, run in enumerate(runs):
            ids = result["trial_table"].query("run_index == @i").trial_id.tolist()
            path = f"{directory}/{run.inputs.stem}_space-fsLR_den-91k_desc-notebook{descriptor}_betas.dscalar.nii"
            artifacts.append(scalar_artifact(path, brain, result["betas"][i], ids))
        rt = np.stack([result["rt"][scope] for scope in ("all", "odd", "even")])
        artifacts.append(
            notebook_map(
                stem,
                brain,
                descriptor,
                "rtcorrelation",
                rt,
                ["all_runs", "odd_runs", "even_runs"],
            )
        )
    return artifacts


def _ridge_metadata(settings, results):
    if not results:
        return None
    from boldtailor.trial_encoding import encoding_metadata

    definition = next(iter(results.values()))["provenance"]
    return dict(
        **encoding_metadata(definition["encoding_mode"]),
        validation_target=definition["validation_target"],
        objective=definition["objective"],
        percentile=settings.get("ridge_percentile", 90.0),
        percentile_role=definition.get("percentile_role", "selection_objective"),
        fraction_norm_basis=definition.get("fraction_norm_basis"),
        encoding_predictors=["task", "trial_type", "response_time"],
        task="Pooled training beta mean, used as the prediction reference level",
        outer_splits="odd_to_even_and_even_to_odd",
        final_fit="Separate all-run tuning and refit; final RT correlations are descriptive",
        at_boundary_fraction=ridge_boundary_summary(results),
        at_boundary="Winner at the shrinkage end (smallest fraction / largest alpha): extend the grid; winner at fraction 1.0 or alpha 0: no regularization preferred",
    )


def _boundary_fraction(selection):
    flags = np.asarray(selection.at_boundary, dtype=bool)
    if flags.ndim == 0:
        return float(flags)
    scored = flags[selection.scoring_mask]
    return float(scored.mean()) if scored.size else float("nan")


def ridge_boundary_summary(results):
    """Fraction of scored grayordinates whose ridge winner is a grid endpoint."""
    return [
        dict(mode=mode, scope=scope, fraction=_boundary_fraction(tuned["selection"]))
        for mode, result in results.items()
        for scope, tuned in result["tuning"].items()
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


def _activation_artifacts(stem, brain, activation):
    return [
        notebook_map(
            stem,
            brain,
            descriptor,
            "activation",
            np.stack([result[name] for name in ACTIVATION_MAP_NAMES]),
            list(ACTIVATION_MAP_NAMES),
        )
        for descriptor, result in activation.items()
    ]


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


def _metadata(
    runs, library, settings, ridge_cv=None, activation=None, *, selections=None
):
    include_rt = bool(settings.get("hrf_selection_rt", True))
    return dict(
        regressors=list(REGRESSORS),
        noise_model="ols",
        hrf_model="spm_or_selected_per_grayordinate",
        response_time="Positive finite seconds centered within run over observed RTs; zero modulation for unavailable RT; coefficient per second",
        missing_response_time=dict(
            regressor="missing_response_time",
            unavailable="Nonfinite or nonpositive response_time",
            coding="One for unavailable RT, zero otherwise; omitted in complete runs",
            convolution="Same onset, duration, and HRF as the other task regressors",
            interpretation="Additive mean response difference for trials with unavailable RT, conditional on trial type",
            contrast_exported=False,
            task_delta_r2="Includes the missing-RT indicator among task predictors",
        ),
        task_model_fingerprint=NSD_TASK_MODEL.fingerprint,
        hrf_normalization=HRF_NORMALIZATION,
        trial_type="Binary codes 0/1, uncentered; the task coefficient is the response on trial_type 0 trials at the run-mean RT",
        task="One unit per presentation; observed RT at its run mean, trial type 0, missing-RT indicator zero",
        orthogonalization=False,
        retained_scans=[len(r.frame_times) for r in runs],
        trimming="Leading nonsteady volumes removed; original acquisition times and event onsets retained",
        nuisance_columns=list(runs[0].confounds.columns) + ["constant"],
        high_pass="fMRIPrep cosines only; no additional drift basis",
        r2="1 - sum(run SSE) / sum(within-run SST), on retained scans; native signal units",
        glm_comparison="Descriptive in-sample optimized minus canonical full R²; not an independent validation of HRF selection",
        inference="Contrasts use equal-run fixed effects; conditional on selected HRFs, without selection uncertainty correction",
        hrf_selection=_selection_description(selection_task_model(include_rt)),
        split_prediction="Train HRF and mean amplitude on one half; freeze both for the other half; nuisance projection is conditional on each run",
        rt_check=_rt_check_description(include_rt, ridge_cv),
        ridge_cv=_ridge_metadata(settings, ridge_cv),
        beta_activation=_activation_metadata() if activation else None,
        hrf_boundary_summary=hrf_boundary_summary(selections),
        hrf_boundary="Fraction of custom picks (ID > 0) within 2% of the library box width of each parameter edge; see HrfSelectionResult.parameter_bound_table",
        library_candidates=len(library.candidates),
        library_fingerprint=library.fingerprint,
        peak_time="Argmax of each full HRF curve on a 0.1-second grid",
        hrf_curve_correlations=dict(
            method="Pearson over HRF time samples, without temporal shifting",
            map_order=list(CORRELATION_NAMES),
            canonical_hrf_id=0,
            time_range_seconds=[float(library.times[0]), float(library.times[-1])],
            sample_interval_seconds=0.1,
            time_grid="Full stored library grid, including zero-padded tails of shorter curves",
            interpretation="Shape similarity, invariant to amplitude scale and offset; not BOLD prediction accuracy",
        ),
        undefined="NaN for constant, undefined, or unprocessed grayordinates; CIFTI axis preserved",
        settings=settings,
    )


def save_workflow(
    runs,
    root,
    output,
    library,
    selections,
    glms,
    betas,
    *,
    settings,
    figures=None,
    ridge_cv=None,
    activation=None,
    subject="sub-07",
    session="ses-nsd10",
):
    """Use separate notebook descriptors so earlier script outputs are preserved."""
    policy = settings.get("existing_results", "error")
    if check_output(output, subject, session, existing_results=policy):
        directory = Path(output) / subject / session / "func"
        return tuple(
            sorted(directory.glob(f"{subject}_{session}_task-nsdcore*desc-notebook*"))
        )
    stem, brain = _stem(subject, session), runs[0].image.header.get_axis(1)
    artifacts = _glm_artifacts(stem, brain, glms, runs)
    artifacts.extend(_hrf_artifacts(stem, brain, selections, library))
    artifacts.extend(_beta_artifacts(stem, brain, betas, runs))
    if activation:
        artifacts.extend(_activation_artifacts(stem, brain, activation))
    if ridge_cv:
        from .ridge_outputs import ridge_artifacts

        artifacts.extend(ridge_artifacts(stem, brain, runs, ridge_cv, library))
    artifacts.extend(_input_artifacts(stem, runs))
    artifacts.append(
        json_artifact(
            f"{stem}_desc-notebook_metadata.json",
            _metadata(
                runs, library, settings, ridge_cv, activation, selections=selections
            ),
        )
    )
    for name, figure in (figures or {}).items():
        artifacts.append(
            figure_artifact(f"{stem}_desc-notebook{name}_plot.png", figure)
        )
    return publish_artifact_set(
        output,
        artifacts,
        source_paths=[p for r in runs for p in input_paths(r.inputs)],
        overwrite=policy == "overwrite",
    )


def _input_artifacts(stem, runs):
    artifacts = [table_artifact(f"{stem}_desc-notebook_runs.tsv", run_summary(runs))]
    directory = str(Path(stem).parent)
    for run in runs:
        base = f"{directory}/{run.inputs.stem}_desc-notebook"
        confounds = run.confounds.copy()
        confounds.insert(0, "frame_time", run.frame_times)
        confounds.insert(0, "original_frame", run.retained_frames)
        artifacts.extend(
            [
                table_artifact(
                    base + "_events.tsv",
                    expand_events(run.events, NSD_TASK_MODEL, run.number),
                ),
                table_artifact(base + "_confounds.tsv", confounds),
            ]
        )
    return artifacts
