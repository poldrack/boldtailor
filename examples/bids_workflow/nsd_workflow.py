# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (boldtailor)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NSD: conventional GLMs, optimized HRFs, and beta series
#
# This notebook analyzes every run of **sub-07 / ses-nsd10** using fMRIPrep CIFTI time series.
# Each section calls the underlying analysis functions explicitly and leaves its results
# available for inspection. These are the stages behind `boldtailor run`, without calling
# its workflow orchestrator. We fit the same GLM twice (canonical
# SPM HRF, then an HRF selected separately at each grayordinate), estimates single-trial beta
# series with OLS and the selected ridge method, maps odd/even HRF reliability, and checks
# beta–reaction-time associations.
#
# The task model is detected from the events: one **task** regressor per presentation,
# modulated by **response_time** and **trial_type** when those columns are present. Modulators
# are uncentered, so the task map is the response at modulator value zero. Runs with
# unavailable RTs also include a convolved missing-RT indicator, retaining every stimulus.
#
# Run from a checkout with `uv sync --group dev` and select its `.venv` Python kernel. The
# local data are not distributed with boldtailor; see [the NSD guide](README.md).

# %% [markdown]
# ## 1. Configuration
#
# Every key in the configuration is a `WorkflowSettings` field (see
# `src/boldtailor/workflow/settings.py`); `bids_root`, `fmriprep_root`, and `output_root`
# are accepted for `bids_dir`, `fmriprep_dir`, and `output_dir`. The local default is
# `/Volumes/extdata1/NSD/BIDS`; `NSD_BIDS_ROOT` overrides it. Outputs default to this
# checkout's ignored `results/nsd_workflow_explicit/` directory, keeping worktrees independent.
# `NSD_FMRIPREP_ROOT` and `NSD_OUTPUT_ROOT` optionally override the derivative locations.
# Otherwise fMRIPrep is discovered under `derivatives/fmriprep*`. An `NSD_CONFIG`
# dictionary defined before this cell overrides all these choices.
#
# The default `max_grayordinates=128` is a smoke run: it uses **every run** and
# exports the original spatial axis, with NaN outside the requested subset. For a full-brain
# analysis set `max_grayordinates=None`, `block_size=4096`, and a new `output_root`. Lower `n_jobs` or
# `block_size` to reduce memory use. Existing outputs for this session stop the run unless
# `existing_results="overwrite"`.

# %%
from pathlib import Path
import sys

repo = next(
    p
    for p in (Path.cwd(), *Path.cwd().parents)
    if (p / "src/boldtailor").is_dir() and (p / "pyproject.toml").is_file()
)
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

import json
import logging
import os

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import FileLink, display
from boldtailor.workflow.report import command_line
from boldtailor.cifti import spatial_signature
from boldtailor.diagnostics import one_sample_t
from boldtailor.fit import fit, task_delta_r2
from boldtailor.hrf_selection import select_hrfs, evaluate_hrf_split
from boldtailor.model import ModelSpec
from boldtailor.reliability import curve_correlations
from boldtailor.task_detection import detect_task_model, task_model_notes
from boldtailor.workflow import (
    analysis,
    inputs,
    outputs,
    plots,
    report,
    summaries,
    surfaces,
)
from boldtailor.workflow.beta_series import BetaModel, fit_cv_beta_series
from boldtailor.workflow.files import odd_even_parity
from boldtailor.workflow.run import WorkflowResult, describe_inputs
from boldtailor.workflow.settings import STAGES
from examples.bids_workflow.nsd_settings import nsd_paths, nsd_settings

config = dict(
    subject="sub-07",
    session="ses-nsd10",
    bids_root=os.environ.get("NSD_BIDS_ROOT", "/Volumes/extdata1/NSD/BIDS"),
    output_root=os.environ.get(
        "NSD_OUTPUT_ROOT", str(repo / "results/nsd_workflow_explicit")
    ),
    max_grayordinates=128,  # None for the complete 91k spatial axis
    n_jobs=4,
    block_size=128,
    existing_results="error",  # set "overwrite" deliberately when rerunning
    # ridge_mode="fractional_cv",  # "fractional_cv", "cv", "fixed" (with ridge_alpha), or "off"
    # hrf_library="default", hrf_n_samples=512, hrf_seed=0, hrf_selection_rt=True,
)
config.update(globals().get("NSD_CONFIG", {}))
config.update(nsd_paths(config))

logging.basicConfig(stream=sys.stderr, format="%(asctime)s %(message)s")
logging.getLogger("boldtailor.workflow").setLevel(logging.INFO)


# %%
settings = nsd_settings(config)
print("Equivalent command:")
print(command_line(settings))

# %% [markdown]
# ## 2. Validate inputs, then run the workflow
#
# Preflight reads event tables, CIFTI headers, timing and confounds, detects the task model,
# checks run counts for nested ridge CV, and checks for existing outputs. It does not fit
# models or publish files. Resolve any reported problem before starting the expensive cell.

# %%
preflight = describe_inputs(settings)
display(pd.DataFrame({"run": preflight.get("runs", [])}))
print(f"HRF candidates: {preflight['library_candidates']}")
print(f"Stages: {', '.join(preflight['stages'])}")
print(f"fMRIPrep: {preflight['fmriprep_dir']}")
print(f"Output: {preflight['output_dir']}")
display(preflight.get("task_model", {}))
for note in preflight["notes"]:
    print(note)
if preflight["problems"]:
    display(pd.DataFrame(preflight["problems"]))
    raise ValueError("Preflight failed; resolve the problems above before fitting")

# %% [markdown]
# ## 3. Load the session and define the task model
#
# `load_session` discovers matching events, CIFTI images, BOLD sidecars and confounds.
# It removes **leading nonsteady scans**, retaining the original acquisition times and
# event onsets. Each returned run contains `.events`, `.confounds`, `.frame_times`,
# `.retained_frames`, and a lazily loaded `.image`. No extra smoothing or filtering is
# applied. Confounds are 24 motion terms, six retained combined-mask aCompCor components,
# fMRIPrep cosine columns, and (when fitting) a run intercept.
#
# Task detection resolves the actual event columns across all runs. The NSD design has
# task, uncentered response time, and reference-coded trial type. Missing/nonpositive RT
# becomes missing modulation with a separate indicator: the stimulus is still modeled.
# Inspect both the raw event rows and resolved model before fitting.
#
# This tutorial requires every stage. For intentionally partial pipelines, use the CLI
# or adapt the dependencies explicitly rather than running cells out of order.

# %%
if settings.stages != frozenset(STAGES):
    raise ValueError("This walkthrough requires all four workflow stages")
outputs.check_output(settings)
runs = inputs.load_session(settings)
task_model = detect_task_model(
    [run.events for run in runs],
    settings.modulators,
    labels=[run.label for run in runs],
)
notes = task_model_notes([run.events for run in runs], settings.modulators)
run_table = inputs.run_summary(runs, task_model)
display(run_table)
display(runs[0].events.head())
display(runs[0].confounds.head())
display(task_model.to_dict())

# %% [markdown]
# ## 4. Choose the spatial blocks and HRF library
#
# A block is an array of grayordinate indices, not a group of runs. Each fit sees every
# run for those spatial locations. `make_blocks` excludes constant signals; output
# arrays retain the full brain-model axis and use NaN for undefined/unprocessed locations.
# `load_block` assembles an AnalysisData object from BOLD arrays, raw trial tables,
# confounds, frame times, and source provenance.
#
# The default library contains canonical SPM, 512 timing-space Sobol samples, and 20
# GLMsingle HRFs. The parameter table and curves define the meaning of each HRF ID.
# `selection_model` controls the task predictors used to choose HRFs; the GLM and final
# beta models continue to use the full `task_model`.

# %%
brain = runs[0].image.header.get_axis(1)
blocks = inputs.make_blocks(
    runs, block_size=settings.block_size, max_grayordinates=settings.max_grayordinates
)
library = settings.build_library()
selection_model = inputs.selection_task_model(task_model, settings.hrf_selection_rt)
run_labels = [run.label for run in runs]
halves = odd_even_parity(runs)
print(f"{sum(map(len, blocks))} nonconstant grayordinates in {len(blocks)} blocks")
print({half: [run_labels[i] for i in indices] for half, indices in halves.items()})
display(library.parameter_table.head(10))
figures = {"Library": plots.library_figure(library)}
display(figures["Library"])

# %% [markdown]
# ## 5. Select HRFs and test them on independent runs
#
# For each grayordinate, `select_hrfs` evaluates each candidate by leave-one-run-out
# prediction of the task-model response, after nuisance adjustment. It pools prediction
# errors across folds and chooses a single winner. The **all-run** winner is used for
# final fits; its selection score is not an independent test score.
#
# `evaluate_hrf_split` selects an HRF using only the training half, learns the task
# amplitude there, then holds both fixed to predict the other half. Calling it in both
# directions gives independent odd→even and even→odd predictions and separate half-session
# HRFs for the reliability analysis. The two halves are determined by run numbers.
#
# The explicit loop below processes spatial blocks sequentially so the data flow is
# visible. Later blockwise fit/CV functions honor `n_jobs`. A feature signature ties
# selections to the exact spatial locations, preventing reuse on the wrong vertices.

# %%
selections = {}
for indices in blocks:
    data = inputs.load_block(runs, settings.bids_dir, indices, selection_model)
    selection_options = dict(
        library=library,
        task_model=selection_model,
        run_labels=run_labels,
        feature_signature=spatial_signature(brain, indices),
    )
    all_selection = select_hrfs(data, **selection_options)
    odd_to_even = evaluate_hrf_split(
        data, train_runs=halves["odd"], test_runs=halves["even"], **selection_options
    )
    even_to_odd = evaluate_hrf_split(
        data, train_runs=halves["even"], test_runs=halves["odd"], **selection_options
    )
    selections[tuple(indices)] = dict(
        all=all_selection, odd=odd_to_even, even=even_to_odd
    )
    print(f"Selected HRFs for grayordinates {indices[0]}–{indices[-1]}")

# %%
selection_maps = analysis.selection_maps(selections, len(brain))
display(summaries.selected_hrfs(selections, library, len(brain)))
display(
    pd.DataFrame(
        {
            "split": ["odd_to_even", "even_to_odd"],
            "median_selected_test_r2": [
                summaries.median(selection_maps[name][0])
                for name in ("odd_to_even", "even_to_odd")
            ],
            "median_canonical_test_r2": [
                summaries.median(selection_maps[name][1])
                for name in ("odd_to_even", "even_to_odd")
            ],
        }
    )
)

# %% [markdown]
# ## 6. Fit canonical and optimized conventional GLMs
#
# Here the actual `ModelSpec`, `fit`, contrasts, and `task_delta_r2` calls are visible.
# Both models use the same trials, nuisance columns, scans, OLS noise model, and one
# contrast per task regressor. Drift is disabled because cosine terms already appear
# among the confounds. Supplying `hrf_selection` changes the HRF per grayordinate.
#
# `fit` expands/convolves the task model, fits each run, and combines contrast estimates
# with equal-run fixed effects. `task_delta_r2` compares full and nuisance-only fits.
# Neither is an out-of-sample accuracy measure. The optimized-minus-canonical full R²
# difference is descriptive and can be negative.
#
# Results are assembled onto the original CIFTI axis for export. Selected-HRF designs
# vary by HRF group, so their keys are `(run_index, hrf_id)`; a canonical fit has one
# design per run. The resulting `glms` dictionaries keep contrasts, R², designs, and
# provenance rather than just the displayed figure.

# %%
model = ModelSpec(
    contrasts={name: {name: 1} for name in task_model.regressor_names},
    confounds=tuple(runs[0].confounds.columns),
    hrf_model="spm",
    drift_model=None,
    noise_model="ols",
    task_model=task_model,
)
glms = {}
for model_name in ("CanonicalGLM", "OptimizedGLM"):
    maps = {
        key: np.full((len(task_model.regressor_names), len(brain)), np.nan)
        for key in ("effects", "variances", "t", "z")
    }
    maps.update(r2=np.full((3, len(brain)), np.nan), designs={}, provenance=[])
    for indices in blocks:
        data = inputs.load_block(runs, settings.bids_dir, indices, task_model)
        fit_options = {}
        if model_name == "OptimizedGLM":
            fit_options = dict(
                hrf_selection=selections[tuple(indices)]["all"],
                feature_signature=spatial_signature(brain, indices),
            )
        fitted = fit(data, model, **fit_options)
        comparison = task_delta_r2(data, model, fitted)
        for key, extract in (
            ("effects", fitted.effect),
            ("variances", fitted.variance),
            ("t", fitted.stat),
            ("z", fitted.z_score),
        ):
            maps[key][:, indices] = np.stack(
                [extract(name) for name in task_model.regressor_names]
            )
        maps["r2"][:, indices] = np.stack(
            [comparison.full_r2, comparison.nuisance_r2, comparison.delta_r2]
        )
        if model_name == "CanonicalGLM":
            maps["designs"].update(
                {(i, 0): design for i, design in enumerate(fitted.design_matrices)}
            )
        else:
            for run_index in range(len(runs)):
                for hrf_id in np.unique(fitted.hrf_indices[fitted.hrf_indices >= 0]):
                    maps["designs"][run_index, int(hrf_id)] = fitted.group_design(
                        run_index, int(hrf_id)
                    )
        maps["provenance"].append(
            dict(
                grayordinate_indices=indices.tolist(),
                record=comparison.provenance.to_dict(),
            )
        )
    glms[model_name] = maps

# %%
first_design = glms["CanonicalGLM"]["designs"][0, 0]
display(first_design.head())
figures["Design"] = plots.design_figure(
    runs[0].frame_times, first_design, task_model.regressor_names
)
glm_summary, figures["GLMComparison"] = plots.glm_comparison(glms)
display(glm_summary)
display(figures["Design"])
display(figures["GLMComparison"])

# %% [markdown]
# ## 7. Compare the independently selected HRF shapes
#
# These calls reuse the odd/even selections; they do not refit the BOLD signal.
# Parameter agreement describes the chosen library parameters. `curve_correlations`
# compares complete HRF curves on the saved time grid, including undershoot, without
# shifting their peaks. Odd/even and half/canonical correlations measure shape agreement,
# not held-out BOLD prediction. The independent prediction scores were shown in section 5.

# %%
odd_ids, even_ids = selection_maps["odd"][0], selection_maps["even"][0]
reliability, figures["HRFReliability"] = plots.parameter_agreement(
    library, odd_ids, even_ids
)
curve_r = curve_correlations(library, odd_ids, even_ids)
curve_reliability, figures["HRFCurveReliability"] = plots.curve_agreement(curve_r)
display(reliability)
display(curve_reliability)
display(figures["HRFReliability"])
display(figures["HRFCurveReliability"])

# %% [markdown]
# ## 8. Estimate OLS single-trial beta series
#
# A conventional GLM estimates one coefficient per task predictor; a single-trial GLM
# instead has **one column per event row**, plus unpenalized nuisance columns and an
# intercept. RT and trial type remain trial metadata, not extra single-trial regressors.
# Missing RT never removes the event from this model.
#
# `analysis.fit_beta_series` is the blockwise adapter around `fit_single_trials`
# (canonical HRF) and `fit_selected_hrfs` (optimized HRFs). It loads blocks, fits each
# run, restores the full spatial axis, and retains trial tables, designs, R² and provenance.
# With `ridge_alpha=0.0` this is OLS. The explicit `selections` argument shows which
# model uses all-run selected HRFs. Betas retain the input signal scale.

# %%
ols_fits = {}
ols_fits["canonical"] = analysis.fit_beta_series(
    runs,
    settings.bids_dir,
    blocks,
    task_model=task_model,
    selections=None,
    ridge_alpha=0.0,
    n_jobs=settings.n_jobs,
)
ols_fits["optimized"] = analysis.fit_beta_series(
    runs,
    settings.bids_dir,
    blocks,
    task_model=task_model,
    selections=selections,
    ridge_alpha=0.0,
    n_jobs=settings.n_jobs,
)
beta_models = {}
for hrf, trial_fit in ols_fits.items():
    name = f"{hrf.capitalize()}TrialOLS"
    beta_models[name] = BetaModel(name=name, hrf=hrf, estimator="OLS", fit=trial_fit)
    print(name, "first-run trials × grayordinates:", trial_fit["betas"][0].shape)
display(ols_fits["optimized"]["trial_table"].head())
display(ols_fits["canonical"]["designs"][0, 0].iloc[:5, :8])

# %% [markdown]
# ## 9. Tune regularization, evaluate independently, then refit
#
# Ridge stabilizes the single-trial coefficients; it does not penalize nuisance terms.
# The default **fractional CV** searches a fraction of the nuisance-residualized OLS
# coefficient norm per grayordinate (1 means OLS, smaller values mean more shrinkage).
# Candidate fractions are selected by trial-modulator encoding accuracy, using task-model
# predictors derived from RT and trial type. In `within_run` mode, evaluation centers
# within each run; this evaluates trial variation rather than absolute run offsets.
#
# `fit_cv_beta_series` manages the nested folds, not the whole workflow:
#
# 1. Within the odd training half, score each fraction by leave-one-run-out encoding.
#    Optimized HRFs are selected without each inner validation run. Select one fraction
#    per grayordinate, then fit the training-half HRF and encoding coefficients.
# 2. Hold those choices fixed and evaluate on even runs. Reverse odd/even and repeat.
#    Fractional CV scores against fixed OLS targets under training-selected HRFs, avoiding
#    an apparent gain from merely shrinking the validation target toward zero.
# 3. Tune again on **all runs**, select all-run HRFs, and fit the final beta series.
#    This final fit is for downstream use; its tuning score is not an independent test.
#
# The returned `tuning` contains candidate scores and selected penalties for odd, even,
# and all scopes; `evaluation` contains independent outer predictions/targets/scores;
# `final` contains the all-run betas. We inspect these separately below. Canonical and
# optimized HRFs are tuned separately. `library=None` explicitly fixes canonical SPM.
#
# Alternative settings remain available: `cv` selects one shared alpha using the configured
# percentile of encoding scores; its targets are candidate-regularized betas. `fixed`
# applies the supplied alpha without tuning; `off` keeps only the OLS fits above.

# %%
cv_results = {}
for hrf, candidates in (("canonical", None), ("optimized", library)):
    if settings.ridge_mode in ("fractional_cv", "cv"):
        penalty_grid = (
            dict(fractions=settings.ridge_fractions)
            if settings.ridge_mode == "fractional_cv"
            else dict(alphas=settings.ridge_alphas)
        )
        tuned = fit_cv_beta_series(
            runs,
            settings.bids_dir,
            task_model=task_model,
            selection_model=selection_model,
            library=candidates,
            **penalty_grid,
            percentile=settings.ridge_percentile,
            encoding_mode=settings.encoding_mode,
            block_size=settings.block_size,
            max_grayordinates=settings.max_grayordinates,
            n_jobs=settings.n_jobs,
        )
        cv_results[hrf] = tuned
        estimator = (
            "FractionalCV" if settings.ridge_mode == "fractional_cv" else "RidgeCV"
        )
        name = f"{hrf.capitalize()}Trial{estimator}"
        beta_models[name] = BetaModel(
            name=name,
            hrf=hrf,
            estimator=estimator,
            fit=tuned["final"],
            tuning=tuned["tuning"],
            evaluation=tuned["evaluation"],
            cv_provenance=tuned["provenance"],
            predictors=tuple(tuned["predictors"]),
        )
    elif settings.ridge_mode == "fixed":
        ridge_fit = analysis.fit_beta_series(
            runs,
            settings.bids_dir,
            blocks,
            task_model=task_model,
            selections=None if hrf == "canonical" else selections,
            ridge_alpha=settings.ridge_alpha,
            n_jobs=settings.n_jobs,
        )
        name = f"{hrf.capitalize()}TrialRidge"
        beta_models[name] = BetaModel(
            name=name, hrf=hrf, estimator="Ridge", fit=ridge_fit
        )

# %%
tuning = None
if cv_results:
    # Rows of cv_r2 are candidate penalties; columns are grayordinates.
    all_tuning = cv_results["optimized"]["tuning"]["all"]
    scores = all_tuning["scores"]
    display(
        pd.DataFrame(
            {
                "candidate": scores.grid,
                "median_encoding_cv_r2": [
                    summaries.median(row) for row in scores.cv_r2
                ],
            }
        )
    )
    display(all_tuning["selection"])
    tuning = outputs.tuning_table(beta_models)
    figures["RidgeTuning"] = outputs.tuning_figure(beta_models)
    display(tuning)
    display(figures["RidgeTuning"])
    if "selected_grayordinates" in tuning:
        figures["FractionSelection"] = plots.fraction_selection_figure(tuning)
        display(figures["FractionSelection"])
    display(summaries.encoding_scores(beta_models))

# %% [markdown]
# ## 10. Activation, RT associations, and cortical diagnostics
#
# `one_sample_t` summarizes trial betas against zero. It treats trials as independent;
# these unthresholded, uncorrected maps are descriptive, not confirmatory inference.
# The beta fits already contain `correlate_rt` results: beta and positive finite RT are
# centered within run; unavailable RTs are excluded from scoring, not beta estimation.
# The RT figure selects a location using canonical-OLS odd-run correlations and displays
# even-run trials. All-run associations are descriptive, especially when RT helped select
# HRFs or ridge. The separate outer encoding scores above provide independent checks.

# %%
activation = {
    name: one_sample_t(model.fit["betas"]) for name, model in beta_models.items()
}
figures["BetaActivation"] = plots.activation_histogram(activation)
display(figures["BetaActivation"])
rt_figure = plots.rt_check_figure(beta_models, runs, brain)
if rt_figure is not None:
    figures["RTCheck"] = rt_figure
    display(rt_figure)
display(summaries.rt_summary(activation, beta_models))

# %% [markdown]
# Surface rendering is optional and changes no estimates. Meshes must use the same fsLR
# ordering as the CIFTI brain-model axis. Missing/ambiguous automatically discovered meshes
# are recorded as a note; an invalid explicitly supplied path remains an error.

# %%
meshes = None
if settings.surface_maps:
    try:
        meshes = surfaces.find_surface_meshes(
            settings.fmriprep_dir, settings.subject, paths=settings.surface_meshes
        )
    except ValueError as error:
        notes.append(f"Skipped surface figures: {error}")
    else:
        if meshes is None:
            notes.append(
                "Skipped surface figures: no fsLR 32k midthickness meshes found"
            )
if meshes is not None:
    surface_panels = (
        ("GLMR2Surface", {name: fit["r2"][0] for name, fit in glms.items()}, "r2"),
        (
            "BetaR2Surface",
            {name: model.fit["r2"][2] for name, model in beta_models.items()},
            "delta_r2",
        ),
        (
            "BetaActivationSurface",
            {name: values["t"] for name, values in activation.items()},
            "t",
        ),
        (
            "RTSurface",
            {
                name: model.fit["rt"]["all"]
                for name, model in beta_models.items()
                if model.fit["rt"] is not None
            },
            "rt",
        ),
    )
    for name, maps, statistic in surface_panels:
        if maps:
            figures[name] = surfaces.surface_figure(
                maps, brain, meshes, statistic=statistic
            )
            display(figures[name])

# %% [markdown]
# ## 11. Serialize results and publish the report
#
# Fitting has finished: the remaining calls only serialize the objects above.
# `workflow_artifacts` builds CIFTI maps, trial tables, grouped designs, provenance JSON,
# metadata, and PNG payloads. `render_report` embeds those same PNGs and summary tables.
# `publish_workflow` writes the artifact set and report together; it does **not** run any
# analysis. Nothing has been published if a fitting cell fails. All intermediate results
# remain inspectable in the kernel, and existing-output protection applies at publication.

# %%
skipped = []
artifacts = outputs.workflow_artifacts(
    settings,
    runs,
    task_model,
    library,
    selections,
    glms,
    beta_models,
    figures=figures,
    activation=activation,
    skipped=skipped,
    notes=notes,
    report=outputs.report_name(settings),
    include_hrf_splits=True,
)
payloads = {artifact.path: artifact.payload for artifact in artifacts}
# Associate each output with its descriptor's provenance, including HRF split outputs.
manifest = []
for path in payloads:
    descriptor = path.partition("_desc-")[2].partition("_")[0]
    candidates = [f"{settings.stem}_desc-{descriptor}_provenance.json"]
    if descriptor.startswith("HRF"):
        candidates.append(f"{settings.stem}_desc-HRF_provenance.json")
    provenance = next((p for p in candidates if p != path and p in payloads), None)
    manifest.append((path, provenance))

# %%
html = report.render_report(
    settings,
    runs=runs,
    task_model=task_model,
    library=library,
    glm_summary=glm_summary,
    hrf_summary=outputs.hrf_boundary_table(selections),
    reliability=reliability,
    tuning=tuning,
    skipped=skipped,
    notes=notes,
    manifest=manifest,
    figures={name: payloads[outputs.figure_name(settings, name)] for name in figures},
    selected_hrfs=summaries.selected_hrfs(selections, library, len(brain)),
    curve_reliability=curve_reliability,
    encoding=summaries.encoding_scores(beta_models),
    ridge_boundary=summaries.ridge_boundary(beta_models),
    activation_summary=summaries.activation_summary(activation),
    rt_summary=summaries.rt_summary(activation, beta_models),
    **summaries.input_tables(runs, task_model, library),
)
paths = outputs.publish_workflow(settings, runs, artifacts, html.encode())
result = WorkflowResult(
    settings=settings,
    paths=tuple(paths),
    report_path=settings.output_dir / outputs.report_name(settings),
    skipped=tuple(skipped),
    task_model=task_model,
    library_fingerprint=library.fingerprint,
)
for figure in figures.values():
    plt.close(figure)
print(f"Saved {len(paths)} files under {settings.output_dir}")
display(pd.DataFrame({"path": [str(path) for path in paths]}))

# %% [markdown]
# ## 12. Read the saved results for downstream analysis
#
# The CIFTI scalar axis contains map names; the brain-model axis identifies cortical
# vertices and subcortical voxels. Read these axes rather than assuming map ordering or
# reshaping a 91k array into a volume. Here we summarize optimized-GLM R² maps using only
# finite grayordinates. In the smoke run, all unprocessed locations are NaN.

# %%
glm_path = settings.output_dir / (
    f"{settings.stem}_{settings.space_entity}_desc-OptimizedGLM_stat-rsquared.dscalar.nii"
)
glm_image = nib.load(glm_path)
glm_values = glm_image.get_fdata()
map_summary = pd.DataFrame(
    {
        "map": glm_image.header.get_axis(0).name,
        "finite_grayordinates": np.isfinite(glm_values).sum(axis=1),
        "median": [
            np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan
            for row in glm_values
        ],
    }
)
display(map_summary)
print(f"Spatial axis: {len(glm_image.header.get_axis(1)):,} grayordinates")

# %% [markdown]
# Each beta CIFTI contains **trials × grayordinates** for one run. The session-wide trial
# table contains the same trial IDs, plus onsets, durations, RTs, and run identifiers.
# Align using the saved IDs before relating betas to behavioral variables. This example
# reads the first run's optimized OLS betas and one finite grayordinate; it makes no
# statistical selection or inference. Ridge models have separate files and trial tables.

# %%
beta_path = sorted(
    p
    for p in result.paths
    if p.name.endswith("_desc-OptimizedTrialOLS_betas.dscalar.nii")
)[0]
beta_image = nib.load(beta_path)
trial_table = pd.read_csv(
    settings.output_dir / f"{settings.stem}_desc-OptimizedTrialOLS_trials.tsv", sep="\t"
)
beta_values = beta_image.get_fdata()
finite_columns = np.flatnonzero(np.isfinite(beta_values).any(axis=0))
if not len(finite_columns):
    raise ValueError(
        "No finite optimized OLS betas; inspect the report and input signals"
    )
grayordinate = int(finite_columns[0])
beta_preview = (
    trial_table.set_index("trial_id")
    .loc[list(beta_image.header.get_axis(0).name)]
    .reset_index()
)
beta_preview["beta"] = beta_values[:, grayordinate]
print(f"{beta_path.name}: {beta_image.shape}; grayordinate {grayordinate}")
display(beta_preview.head(10))

# %% [markdown]
# The metadata records the resolved settings and model definitions. The HTML report is
# self-contained and can be opened directly in a browser; keep the derivative directory
# for the maps, tables, designs, and linked provenance records.
#
# For interpretation, distinguish in-sample GLM R² and selected-HRF CV scores from the
# independent odd→even/even→odd prediction and outer encoding scores. A larger in-sample
# optimized-HRF R² alone does not establish generalization. The small spatial smoke run
# verifies execution and file formats; use the full spatial analysis for scientific work.

# %%
metadata_path = settings.output_dir / f"{settings.stem}_desc-boldtailor_metadata.json"
metadata = json.loads(metadata_path.read_text())
display(metadata["settings"])
print(f"Open report in a browser: {result.report_path}")
display(FileLink(str(result.report_path)))
