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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NSD: HRF reliability across sessions
#
# Estimate HRFs independently for **sub-07, ses-nsd10 through ses-nsd19**, then compare the full selected HRF curves at each grayordinate. Every session uses the same **512 Sobol samples plus canonical SPM**, with seed 0.
#
# Within each session, the model selects an HRF using leave-one-run-out prediction of the shared task-model response (task with uncentered RT and trial-type modulators). It includes motion24, the top six retained combined-mask aCompCor components, fMRIPrep cosine high-pass terms, and run intercepts. Leading nonsteady scans are removed while retaining original acquisition times. No information from another session selects that session's HRF.
#
# This targeted notebook fits HRF selection only. It does not fit beta series or tune ridge penalties. Run it from the project's `.venv` kernel after `uv sync --group dev`.
#
# Trials with missing reaction times are retained; runs with unavailable RTs add a missing-RT indicator that selection profiles out per run.
#

# %% [markdown]
# ## 1. Settings and library
#
# Sessions run sequentially, with parallel grayordinate blocks within each session. `max_grayordinates=128` provides a quick check using all runs; `None` processes the full brain.
#
# Completed session estimates are saved immediately. Rerunning reuses compatible estimates from this notebook or the boldtailor workflow (`boldtailor run` or `nsd_workflow.ipynb`). Matching requires the same library, source identities, nuisance model, trimming, coverage, and CIFTI axis. Old grid/untrimmed script results remain separate. `reuse_roots` can point to additional derivative roots containing full-workflow outputs.
#
#
# Set `NSD_BIDS_ROOT` in the environment before starting the kernel, or supply
# `bids_root` in a `HRF_RELIABILITY_CONFIG` dictionary before the setup cell.
# Optional `NSD_FMRIPREP_ROOT` and `NSD_OUTPUT_ROOT` override derivative locations.
# Explicit dictionary paths take precedence. The helpers are checkout-local examples;
# Boldtailor itself models arrays without requiring NSD or a particular imaging format.
#

# %%
from pathlib import Path
import sys

repo = next(
    p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/boldtailor").is_dir()
)
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from examples.NSD.nsd_settings import nsd_paths

overrides = globals().get("HRF_RELIABILITY_CONFIG", {})
settings = dict(
    reuse_roots=[],
    subject="sub-07",
    sessions=[f"ses-nsd{i}" for i in range(10, 20)],
    hrf_n_samples=512,
    hrf_seed=0,
    hrf_selection_rt=True,  # False scores HRFs without the RT regressor.
    n_jobs=4,
    block_size=4096,
    max_grayordinates=None,
)
settings.update(overrides)
settings.update(nsd_paths(overrides))
settings.setdefault("fmriprep_root", None)  # None: derivatives/fmriprep*
settings.setdefault(
    "output_root", str(Path(settings["bids_root"]) / "derivatives/boldtailor")
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from boldtailor.hrf_library import sobol_hrf_library
from examples.NSD.session_hrf import estimate_sessions
from boldtailor.reliability import compare_hrfs, SUMMARY_NAMES
from examples.NSD.session_hrf_outputs import session_table, save_reliability
from examples.NSD.session_hrf_plots import (
    agreement_figure,
    parameter_variability_figure,
)

library = sobol_hrf_library(settings["hrf_n_samples"], seed=settings["hrf_seed"])
print(
    f"{len(settings['sessions'])} sessions; {len(library.candidates)} HRFs; seed {settings['hrf_seed']}"
)
print(f"Library fingerprint: {library.fingerprint}")
display(library.parameter_table.head())


# %% [markdown]
# ## 2. Estimate or reuse each session
#
# This is the expensive cell on the first run. All sessions are checked before fitting starts. Each session uses all of its runs; the resulting map records one winning HRF per grayordinate. Completed caches from this notebook are reused without fitting. Imported full-workflow estimates also undergo a spatial-coverage check against the current data.
#
# A changed library, input file, or spatial subset gets a separate cache. A damaged cache is recomputed. Comparisons require exactly matching grayordinate axes across sessions; undefined or unprocessed locations remain NaN.
#

# %%
estimates = estimate_sessions(
    settings["bids_root"],
    settings["fmriprep_root"],
    settings["output_root"],
    library=library,
    sessions=settings["sessions"],
    subject=settings["subject"],
    n_jobs=settings["n_jobs"],
    block_size=settings["block_size"],
    max_grayordinates=settings["max_grayordinates"],
    reuse_roots=settings["reuse_roots"],
    include_rt=settings["hrf_selection_rt"],
)
display(session_table(estimates))


# %% [markdown]
# ## 3. Compare full HRF curves, with canonical SPM as a baseline
#
# For each grayordinate, calculate Pearson correlation over the complete HRF time grid, including the undershoot, without shifting curves to align their peaks. Ten sessions give **45 unordered session pairs** and **10 session-to-canonical comparisons**.
#
# For a pair of sessions A and B, the baseline is the mean of `r(A, canonical)` and `r(B, canonical)` at that same grayordinate. The delta is `r(A, B) − baseline`. A positive value means those selected shapes are more similar to each other than to canonical SPM on average.
#
# Summary maps take arithmetic means over valid pairs. The baseline and delta use exactly the same pairs. Counts show how many sessions and pairs contributed. The heatmap averages each comparison over its available grayordinates; the histograms compare matched per-grayordinate summaries.
#
# These are descriptive measures of curve shape, not ICCs or BOLD prediction accuracy. Correlation ignores amplitude scaling; session pairs share data and are not independent observations.
#

# %%
hrf_ids = np.stack([estimate.maps[0] for estimate in estimates])
comparison = compare_hrfs(
    library, hrf_ids, [estimate.session for estimate in estimates]
)
figures = {"agreement": agreement_figure(comparison)}
plt.show()
valid = np.isfinite(comparison["summary"][:3]).all(axis=0)
display(
    pd.DataFrame(comparison["summary"][:, valid].T, columns=SUMMARY_NAMES).describe()
)


# %% [markdown]
# ## 4. Variability of individual HRF parameters
#
# Full-curve agreement is the primary comparison because different parameter combinations can produce similar curves. Also save the sample standard deviation of each of the six varied parameters and full-HRF time to peak across sessions. At least two valid estimates are required. Duration is excluded from this variability summary.
#
# These standard deviations describe variability of the selected library entries; they are not confidence intervals for the underlying physiological parameters.
#

# %%
figures["parameter_variability"] = parameter_variability_figure(comparison)
plt.show()


# %% [markdown]
# ## 5. Save the comparisons
#
# Session caches live under `sub-07/ses-nsdXX/func/`, with `desc-sessionHRF...` names. Each includes selection scores, HRF indices and parameter maps, exact library parameters and curves, provenance, and a cache manifest.
#
# Across-session outputs live under `sub-07/func/`, with `desc-sessionHRFReliability...` names. CIFTIs contain all pairwise correlations, matched canonical baselines and deltas, session-to-canonical correlations, five summary maps, parameter SD maps, and one HRF-index map per session. TSVs identify sessions and pairs; the metadata JSON defines every comparison. Saved plots reproduce the figures above.
#
# Rerunning refreshes this report while reusing completed session estimates. Different inputs, libraries, or session sets receive different output names.
#

# %%
paths = save_reliability(
    settings["output_root"],
    estimates,
    library,
    comparison,
    subject=settings["subject"],
    settings=settings,
    figures=figures,
)
print(f"Saved {len(paths)} comparison files under {settings['output_root']}")
display(pd.DataFrame({"path": [str(path) for path in paths]}))
