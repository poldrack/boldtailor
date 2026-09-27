"""Publish notebook maps, designs, trial metadata, and provenance together."""

from pathlib import Path

import numpy as np

from boldtailor.publication import publish_artifact_set
from .hrf_artifacts import npz_artifact, parameter_artifact, figure_artifact
from .hrf_reliability import CORRELATION_NAMES, hrf_curve_correlations
from .nsd_cifti import _input_paths
from .single_trial_artifacts import json_artifact, table_artifact, scalar_artifact
from .workflow_analysis import selection_maps
from .workflow_inputs import REGRESSORS, glm_events, run_summary

R2_NAMES = ["full_r2", "confounds_r2", "task_delta_r2"]


def _stem(subject, session):
    return f"{subject}/{session}/func/{subject}_{session}_task-nsdcore"


def check_output(output, subject="sub-07", session="ses-nsd10"):
    """Catch prior notebook outputs before launching a long analysis."""
    directory = Path(output) / subject / session / "func"
    if any(directory.glob(f"{subject}_{session}_task-nsdcore*desc-notebook*")):
        raise FileExistsError(
            f"Notebook outputs already exist in {directory}; choose a new output_root"
        )


def _map(stem, brain, descriptor, statistic, values, names):
    path = f"{stem}_space-fsLR_den-91k_desc-notebook{descriptor}_stat-{statistic}.dscalar.nii"
    return scalar_artifact(path, brain, values, names)


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
        _map(stem, brain, descriptor, "rsquared", result["r2"], R2_NAMES),
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
                _map(stem, brain, descriptor, stat, result[stat], list(REGRESSORS))
            )
    difference = glms["OptimizedGLM"]["r2"][0] - glms["CanonicalGLM"]["r2"][0]
    artifacts.append(
        _map(
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
        _map(
            stem,
            brain,
            "HRFReliability",
            "curvecorrelation",
            hrf_curve_correlations(library, maps["odd"][0], maps["even"][0]),
            list(CORRELATION_NAMES),
        ),
    ]
    for name in ("all", "odd", "even"):
        descriptor = "HRF" + name.title()
        artifacts.append(
            _map(
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
            _map(
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
            _map(
                stem,
                brain,
                descriptor,
                "rtcorrelation",
                rt,
                ["all_runs", "odd_runs", "even_runs"],
            )
        )
    return artifacts


def _metadata(runs, library, settings, ridge_cv=None):
    return dict(
        regressors=list(REGRESSORS),
        noise_model="ols",
        hrf_model="spm_or_selected_per_grayordinate",
        response_time="Seconds, centered within run; coefficient per second",
        trial_type="Binary codes 0/1, centered within run; coefficient type 1 minus type 0",
        task="One unit per presentation; other covariates held at their run means",
        orthogonalization=False,
        retained_scans=[len(r.frame_times) for r in runs],
        trimming="Leading nonsteady volumes removed; original acquisition times and event onsets retained",
        nuisance_columns=list(runs[0].confounds.columns) + ["constant"],
        high_pass="fMRIPrep cosines only; no additional drift basis",
        r2="1 - sum(run SSE) / sum(within-run SST), on retained scans; native signal units",
        glm_comparison="Descriptive in-sample optimized minus canonical full R²; not an independent validation of HRF selection",
        inference="Contrasts use equal-run fixed effects; conditional on selected HRFs, without selection uncertainty correction",
        hrf_selection="Sum leave-one-run-out mean-stimulus prediction errors, then choose one HRF per grayordinate",
        split_prediction="Train HRF and mean amplitude on one half; freeze both for the other half; nuisance projection is conditional on each run",
        rt_check=(
            "RT/type tune CV ridge strength; final all-run RT correlations are descriptive. Outer test runs are excluded from HRF and penalty selection."
            if ridge_cv
            else "Descriptive within-run-centered correlation, never used to select HRFs or fixed ridge strength; all-run optimized HRFs use both halves"
        ),
        ridge_cv=(
            dict(
                validation_target="candidate_regularized_betas",
                percentile=settings.get("ridge_percentile", 90.0),
                encoding_predictors=["task", "trial_type", "response_time"],
                task="Shared trial-encoding intercept, not an additional all-ones column",
                objective="Percentile across a common grayordinate mask of pooled within-run encoding R2",
                outer_splits="odd_to_even_and_even_to_odd",
                final_fit="Separate all-run tuning and refit; final RT correlations are descriptive",
            )
            if ridge_cv
            else None
        ),
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
    subject="sub-07",
    session="ses-nsd10",
):
    """Use separate notebook descriptors so earlier script outputs are preserved."""
    check_output(output, subject, session)
    stem, brain = _stem(subject, session), runs[0].image.header.get_axis(1)
    artifacts = _glm_artifacts(stem, brain, glms, runs)
    artifacts.extend(_hrf_artifacts(stem, brain, selections, library))
    artifacts.extend(_beta_artifacts(stem, brain, betas, runs))
    if ridge_cv:
        from .ridge_outputs import ridge_artifacts

        artifacts.extend(ridge_artifacts(stem, brain, runs, ridge_cv, library))
    artifacts.extend(_input_artifacts(stem, runs))
    artifacts.append(
        json_artifact(
            f"{stem}_desc-notebook_metadata.json",
            _metadata(runs, library, settings, ridge_cv),
        )
    )
    for name, figure in (figures or {}).items():
        artifacts.append(
            figure_artifact(f"{stem}_desc-notebook{name}_plot.png", figure)
        )
    return publish_artifact_set(
        output,
        artifacts,
        source_paths=[p for r in runs for p in _input_paths(r.inputs)],
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
                table_artifact(base + "_events.tsv", glm_events(run.events)),
                table_artifact(base + "_confounds.tsv", confounds),
            ]
        )
    return artifacts
