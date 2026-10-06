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
# It runs the package workflow, `boldtailor.workflow.run.run_workflow`, which is the same
# analysis as the `boldtailor run` command. The workflow fits the same GLM twice (canonical
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
# are accepted for `bids_dir`, `fmriprep_dir`, and `output_dir`. Set `NSD_BIDS_ROOT` in the
# environment, or supply `bids_root` in an `NSD_CONFIG` dictionary before this cell.
# `NSD_FMRIPREP_ROOT` and `NSD_OUTPUT_ROOT` optionally override the derivative locations;
# otherwise fMRIPrep is found under `derivatives/fmriprep*` and outputs go to
# `derivatives/boldtailor_hrf-..._ridge-...`.
#
# For a quick check, set `max_grayordinates=128`: the workflow still uses **every run** and
# exports the original spatial axis, with NaN outside the requested subset. Lower `n_jobs` or
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

import logging, sys
import pandas as pd
from IPython.display import Image, display
from boldtailor.workflow.report import command_line
from boldtailor.workflow.run import run_workflow
from examples.NSD.nsd_settings import nsd_paths, nsd_settings

config = dict(
    subject="sub-07",
    session="ses-nsd10",
    bids_root="/Volumes/extdata1/NSD/BIDS",
    n_jobs=12,
    # max_grayordinates=128, n_jobs=4, block_size=4096, existing_results="overwrite",
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
# ## 2. Run the workflow
#
# This is the expensive cell. The workflow loads the session, removes leading nonsteady
# volumes (keeping original acquisition times and event onsets), selects HRFs, fits the GLMs
# and beta series, and publishes every map, table, figure, and provenance file together
# with a self-contained HTML report. A failure leaves no partial outputs.

# %%
result = run_workflow(settings)
print(f"Saved {len(result.paths)} files under {settings.output_dir}")
print(f"Report: {result.report_path}")
for stage, reason in result.skipped:
    print(f"Skipped {stage}: {reason}")

shown = set()


def show(*names):
    """Display saved figures, read back from their desc-<name>_plot.png files."""
    for name in names:
        path = settings.output_dir / f"{settings.stem}_desc-{name}_plot.png"
        if name not in shown and path.is_file():
            shown.add(name)
            display(Image(filename=str(path)))


# %%

# %% [markdown]
# ## 3. Inputs and nuisance model
#
# Each model includes 24 motion parameters, the six retained combined-mask aCompCor
# components with the greatest explained variance, fMRIPrep's cosine high-pass columns, and
# a run intercept. Coefficients retain native signal units. R² is evaluated on the retained
# scans. The run table lists trials, retained and dropped scans, and modulator summaries.

# %%
runs = pd.read_csv(
    settings.output_dir / f"{settings.stem}_desc-boldtailor_runs.tsv", sep="\t"
)
display(runs)

# %% [markdown]
# ## 4. Conventional GLMs
#
# The canonical and optimized GLMs share events, confounds, scans, and contrasts; only the
# HRF differs. Contrasts combine runs with equal-run fixed effects. The comparison map is
# **optimized full R² minus canonical full R²**: a descriptive in-sample comparison that can
# be negative. Cortical maps appear when the subject's fsLR 32k surfaces are found, or when
# `surface_meshes` is set.

# %%
show("Design", "GLMComparison", "GLMR2Surface")

# %% [markdown]
# ## 5. HRF library and odd/even reliability
#
# The default library contains canonical SPM plus Sobol samples of the six double-gamma
# parameters. Selection predicts each held-out run's task-model response from the other runs
# and picks **one winning HRF per grayordinate**. Independent odd- and even-run selections
# compare parameters and complete HRF curves (Pearson over time samples, without peak
# alignment) against the canonical baseline. These are descriptive shape agreements, not
# BOLD prediction accuracy.

# %%
show("Library", "HRFReliability", "HRFCurveReliability")

# %% [markdown]
# ## 6. Single-trial beta series
#
# Every stimulus receives its own coefficient. In CV modes, trial predictors (task and the
# modulators) tune ridge strength; odd/even outer evaluations keep test runs out of HRF and
# penalty selection, and a separate all-run tune produces the final betas. The activation
# maps are descriptive one-sample t tests of trial betas against zero, treating trials as
# independent; they are unthresholded and uncorrected.

# %%
show(
    "RidgeTuning",
    "FractionSelection",
    "BetaR2Surface",
    "BetaActivation",
    "BetaActivationSurface",
)

# %% [markdown]
# ## 7. Reaction-time associations
#
# Correlations use trial betas and RT after centering both within each run, excluding
# missing or nonpositive RTs. When RT participates in HRF or penalty selection, all-run
# correlations are descriptive; the outer encoding scores remain independent checks.

# %%
show("RTCheck", "RTSurface")

# %% [markdown]
# ## 8. Saved outputs
#
# The report links every output to its provenance. Files use `desc-<Name>` descriptors under
# `sub-07/ses-nsd10/func/`, and the settings file is `..._desc-boldtailor_metadata.json`. Any
# remaining figures are shown below.

# %%
remaining = sorted(
    p.name.split("_desc-")[1].removesuffix("_plot.png")
    for p in (settings.output_dir / Path(settings.stem).parent).glob("*_plot.png")
)
show(*remaining)
display(pd.DataFrame({"path": [str(p) for p in result.paths]}))
