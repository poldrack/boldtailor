# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: boldtailor (3.12.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # HRF estimation: staged VTC pilot results
#
# **Subject 07 · sessions 11 and 12 · bilateral anatomical fusiform cortex (1,750 vertices).**
#
# These are completed local pilots within the planned sessions 10–19 study. Each session uses all 12 runs and one complementary pair of 6/6 run splits. Library CV and direct REML are compared with the canonical HRF. The continuous optimizers are shown separately as a four-vertex smoke test.
#
# **Endpoint:** only the HRF is frozen on evaluation runs; all coefficients are refitted. These are **HRF-transfer R²** results, not fully held-out BOLD predictions. REML and continuous estimation remain experimental.
#
# This notebook reads saved results only. Outputs are embedded for viewing without access to the original NSD data. Rerunning requires this experiment checkout and its `outputs/` directories.
#
# This historical report reads a compact, hash-verified snapshot under `reports/legacy_notebook`, so it can be rebuilt without local fit-output directories. Rendered notebooks and regenerated figure files remain local; the Python percent script is the version-controlled source.

# %%
# %matplotlib inline
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

candidates = [Path.cwd(), *Path.cwd().parents]
EXPERIMENT = next((p for root in candidates
                   for p in (root, root / "experiments/hrf_estimation")
                   if (p / "report.py").is_file() and (p / "reports/pilot_summary.tsv").is_file()), None)
if EXPERIMENT is None:
    raise FileNotFoundError("Run from the experiment directory or repository root.")
sys.path.insert(0, str(EXPERIMENT.parents[1]))
from experiments.hrf_estimation.report import summarize_transfer

OUTPUTS = EXPERIMENT / "reports/legacy_notebook"
FIGURES = EXPERIMENT / "outputs/notebook-figures"
FIGURES.mkdir(parents=True, exist_ok=True)
METHODS = ["canonical", "librarycv", "reml"]
LABELS = {"canonical": "Canonical", "librarycv": "Library CV", "reml": "Direct REML"}
COLORS = {"canonical": "#777777", "librarycv": "#0072B2", "reml": "#D55E00"}
plt.rcParams.update({"figure.figsize": (10, 4.8), "figure.dpi": 120,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 11})

def finish(ax, filename, ylabel):
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES / filename, dpi=150, bbox_inches="tight")
    plt.show()

def method_bars(table, index, metric, title, filename, ylabel):
    wide = table.pivot(index=index, columns="method", values=metric)[METHODS]
    ax = wide.rename(columns=LABELS).plot.bar(
        color=[COLORS[m] for m in METHODS], rot=0, title=title)
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1))
    finish(ax, filename, ylabel)



# %% [markdown]
# ## Inputs and integrity
#
# The model includes binary trial type, a common RT amplitude modulator, separate missing-response and incorrect-response junk regressors, 24 motion parameters derived from NSD confounds, and model-generated cosine drifts with an **80-second cutoff**. There is no scan censoring. One HRF is shared by task and junk effects at each location.
#
# The checks below verify saved artifact hashes; they do not rerun the fits or assert that the historical run used the current source revision.

# %%
tables, manifests, integrity = {}, {}, []
for session in (11, 12):
    directory = OUTPUTS / f"vtc-session{session}"
    tables[session] = pd.read_csv(directory / "scores.tsv.gz", sep="\t")
    manifests[session] = json.loads((directory / "manifest.json").read_text())
    hashes = json.loads((directory / "COMPLETE.json").read_text())["artifact_hashes"]
    for name, expected in hashes.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    integrity.append({"session": session, "vertices": tables[session].feature_id.nunique(),
                      "runs": len(set(sum([s["train"] + s["test"] for s in manifests[session]["splits"]], []))),
                      "verified_artifacts": len(hashes), "seed": manifests[session]["seed"]})
display(pd.DataFrame(integrity))
failures = pd.concat([t.assign(session=s) for s, t in tables.items()])
display(failures[failures.status != "ok"].groupby(["session", "method", "evaluation", "status"]).size().rename("count").to_frame())

# %% [markdown]
# ## HRF-transfer explained variance
#
# For each vertex, sum the residual sums of squares across the complementary held-out halves **before** forming ratios:
#
# - Total R² = 1 − SSE(full) / SST (run-mean reference).
# - Partial task R² = 1 − SSE(full) / SSE(nuisance).
# - Incremental task R² = [SSE(nuisance) − SSE(full)] / SST.
#
# The nuisance-only fit includes junk regressors, whose shapes depend on the HRF. Total R² also includes nuisance variance. Tables and bars report medians over the **same valid vertices across all three methods and both halves**, separately per session (1,749 in session 11; 1,750 in session 12). One session-11 REML fit has zero estimated task variance.

# %%
summary = summarize_transfer(tables)
expected = pd.read_csv(EXPERIMENT / "reports/pilot_summary.tsv", sep="\t")
pd.testing.assert_frame_equal(summary, expected, check_exact=False, rtol=1e-12, atol=1e-12)
display({"text/html": summary.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
         "text/plain": summary.to_string(index=False),
         "application/json": {"kind": "hrf_pooled_summary", "records": summary.to_dict("records")}}, raw=True)

# %%
method_bars(summary, "session", "r2_task_partial", "Partial task variance explained", "r2_task_partial.png", "Median partial task R²")

# %%
method_bars(summary, "session", "r2_full", "Total variance explained", "r2_full.png", "Median total R²")

# %%
method_bars(summary, "session", "delta_r2_task", "Incremental task variance explained", "delta_r2_task.png", "Median incremental task R²")

# %% [markdown]
# Both fitted-HRF methods improve the median transfer scores over the canonical HRF in these two sessions. Library CV and direct REML are close. These descriptive pilot medians do not establish a general winner; vertices are spatially correlated and only one split pair per session has been evaluated.

# %% [markdown]
# ## Estimated HRF shapes
#
# Curves below use HRFs estimated from all 12 runs in each session (the `insample` estimates), **not** the training-half curves used for transfer scoring. Each valid vertex curve is normalized to unit positive peak before taking the pointwise median. The resulting median curve need not itself have a peak of exactly one.

# %%
curves = {}
for session in (11, 12):
    for method in METHODS:
        directory = OUTPUTS / f"vtc-session{session}"
        path = next(directory.glob(f"*desc-{method}insample_hrf.npz"))
        with np.load(path) as data:
            curves[f"{LABELS[method]} · {session}"] = data["median_curve"]
            times = data["times"]
frame = pd.DataFrame(curves, index=times)
ax = frame.plot.line(color=[COLORS[m] for s in (11, 12) for m in METHODS],
                     style=["-", "-", "-", "--", "--", "--"], title="Median fitted HRFs")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xlabel("Time after onset (seconds)")
ax.legend(ncol=2, fontsize=9)
ax.set_ylabel("Unit-peak normalized response")
ax.figure.tight_layout()
ax.figure.savefig(FIGURES / "hrf_shapes.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## HRF stability
#
# Within-session comparisons pair HRFs estimated from complementary six-run halves. Between-session comparisons use the all-run estimates. The table reports method-specific valid vertices, unlike the common-vertex transfer summary above. Waveform correlations are descriptive; smooth constrained curves can correlate highly even when timing differs. Canonical stability is perfect by construction.

# %%
stability = pd.read_csv(EXPERIMENT / "reports/pilot_stability.tsv", sep="\t")
display(stability.round(4))
method_bars(stability, "comparison", "curve_correlation", "Median waveform correlation",
            "stability_correlation.png", "Pearson correlation")

# %%
method_bars(stability, "comparison", "peak_difference", "Median absolute peak-time difference",
            "stability_peak.png", "Absolute difference (seconds)")

# %% [markdown]
# ## Continuous-optimizer smoke test — separate scope
#
# This check uses **four vertices, four session-11 runs, a 2/2 split, and 32 timing-library samples** for initialization. It is not a full VTC method comparison and must not be ranked against the 533-kernel library results above. Regularization is selected by inner cross-validation. The displayed elapsed time includes inner tuning and is a single workflow measurement, not a benchmark.

# %%
smoke_dir = OUTPUTS / "continuous-smoke"
smoke = pd.read_csv(smoke_dir / "scores.tsv.gz", sep="\t")
config = json.loads((smoke_dir / "config.json").read_text())
display(smoke[["optimizer", "seconds", "regularization"]].drop_duplicates().reset_index(drop=True))
display(smoke[["feature_id", "optimizer", "r2_task_partial", "status"]])
ax = smoke.pivot(index="feature_id", columns="optimizer", values="r2_task_partial").plot.bar(
    rot=0, title="Continuous smoke test: partial task R²", color=["#009E73", "#CC79A7"])
finish(ax, "continuous_smoke.png", "Partial task R²")

# %% [markdown]
# ## Completed four-vertex REML validation
#
# This section compares original REML with **AR(1) noise**, **separate condition/RT/junk variance components**, **shared task effects across runs**, and **all three combined**. It uses four fixed anatomical vertices, all 12 runs in each session, the full 533-candidate library, and the same complementary 6/6 run splits. It is an execution and method check, **not a VTC-wide ranking**; do not compare its medians directly with the 1,750-vertex summaries above.
#
# AR(1) is estimated only from training data, corrected for the effect of reference-design projection, and held fixed across candidate HRFs. Shared effects connect only training runs. Every method still freezes only the HRF and refits amplitudes on evaluation runs; scores below are unwhitened HRF-transfer R².
#
# The earlier Powell smoke test is historical. **Powell has been removed from the active queues.**

# %%
REML_METHODS = ["reml", "remlar1", "remlcomponents", "remlshared", "remlcombined"]
REML_LABELS = {"reml": "Original REML", "remlar1": "+ AR(1)",
               "remlcomponents": "+ Separate variances", "remlshared": "+ Shared task",
               "remlcombined": "Combined"}
REML_COLORS = ["#777777", "#0072B2", "#E69F00", "#009E73", "#CC79A7"]
reml_tables, reml_directories = {}, {}
for session in (11, 12):
    directory = OUTPUTS / f"reml-final-session{session}"
    completion = json.loads((directory / "COMPLETE.json").read_text())
    for name, digest in completion["artifact_hashes"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
    reml_directories[session] = directory
    reml_tables[session] = pd.read_csv(directory / "scores.tsv.gz", sep="\t")
reml_summary = summarize_transfer(reml_tables)
reml_expected = pd.read_csv(EXPERIMENT / "reports/reml_validation_summary.tsv", sep="\t")
pd.testing.assert_frame_equal(reml_summary, reml_expected, check_exact=False, rtol=1e-12, atol=1e-12)
display({"text/html": reml_summary.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
         "text/plain": reml_summary.to_string(index=False),
         "application/json": {"kind": "reml_validation_summary", "records": reml_summary.to_dict("records")}}, raw=True)

def reml_bars(table, index, metric, title, filename, ylabel):
    wide = table.pivot(index=index, columns="method", values=metric)[REML_METHODS]
    ax = wide.rename(columns=REML_LABELS).plot.bar(
        color=REML_COLORS, rot=0, figsize=(11, 4.8), title=title)
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1))
    finish(ax, filename, ylabel)


# %%
reml_bars(reml_summary, "session", "r2_task_partial",
          "REML validation: partial task transfer R² (four vertices)",
          "reml_validation_partial.png", "Median partial task R²")

# %%
reml_bars(reml_summary, "session", "delta_r2_task",
          "REML validation: incremental task transfer R² (four vertices)",
          "reml_validation_incremental.png", "Median incremental task R²")

# %% [markdown]
# The score changes vary by session. For example, the combined model increases median partial task R² from 0.1354 to 0.1389 in session 11, but decreases it from 0.1021 to 0.1004 in session 12. Four vertices cannot establish a reliable improvement. Total R² is included in the table; it also reflects nuisance fit.

# %% [markdown]
# ### Stability of the estimated shapes
#
# Within-session stability compares the two independently trained six-run halves. Between-session stability compares the all-run estimates at the same four vertices. Values are descriptive medians, with no independent-vertex significance tests. Greater similarity alone does not establish better HRF recovery: shrinkage can increase similarity while adding bias.

# %%
reml_stability = pd.read_csv(EXPERIMENT / "reports/reml_validation_stability.tsv", sep="\t")
reml_stability["comparison"] = reml_stability.comparison.map({
    "within11": "Within session 11", "within12": "Within session 12",
    "between11and12": "Sessions 11 vs 12"})
reml_stability["comparison"] = pd.Categorical(
    reml_stability.comparison,
    categories=["Within session 11", "Within session 12", "Sessions 11 vs 12"], ordered=True)
display(reml_stability[["comparison", "method", "n_valid", "curve_correlation",
                        "curve_rmse", "peak_difference", "fwhm_difference"]].round(4))
reml_bars(reml_stability, "comparison", "curve_correlation",
          "REML validation: waveform stability (four vertices)",
          "reml_validation_stability.png", "Median Pearson correlation")

# %%
reml_bars(reml_stability, "comparison", "peak_difference",
          "REML validation: peak-time stability (four vertices)",
          "reml_validation_peak.png", "Median absolute difference (seconds)")

# %% [markdown]
# AR(1) and the combined model improve waveform agreement between halves in session 11, while their agreement is lower than original REML in session 12. Between-session correlations also show no consistent improvement. This mixed pattern motivates the larger comparison rather than selecting a winner from this validation.

# %% [markdown]
# ### Fitted HRF curves
#
# These curves summarize the **all-run** estimates used for the between-session comparison. Each vertex is normalized to unit positive peak before taking the pointwise median. The median curve is descriptive and is not the HRF used to score all vertices; each location has its own HRF.

# %%
reml_curves = {}
for method in REML_METHODS:
    path = next(reml_directories[11].glob(f"*desc-{method}insample_hrf.npz"))
    with np.load(path) as data:
        reml_curves[REML_LABELS[method]] = data["median_curve"]
        reml_times = data["times"]
ax = pd.DataFrame(reml_curves, index=reml_times).plot.line(
    color=REML_COLORS, figsize=(11, 4.8), title="Session 11: median fitted HRF (four vertices)")
ax.axhline(0, color="black", linewidth=.6)
ax.set_xlabel("Time after onset (seconds)")
ax.set_ylabel("Unit-peak normalized response")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
ax.figure.tight_layout()
ax.figure.savefig(FIGURES / "reml_validation_hrfs11.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
reml_curves = {}
for method in REML_METHODS:
    path = next(reml_directories[12].glob(f"*desc-{method}insample_hrf.npz"))
    with np.load(path) as data:
        reml_curves[REML_LABELS[method]] = data["median_curve"]
        reml_times = data["times"]
ax = pd.DataFrame(reml_curves, index=reml_times).plot.line(
    color=REML_COLORS, figsize=(11, 4.8), title="Session 12: median fitted HRF (four vertices)")
ax.axhline(0, color="black", linewidth=.6)
ax.set_xlabel("Time after onset (seconds)")
ax.set_ylabel("Unit-peak normalized response")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
ax.figure.tight_layout()
ax.figure.savefig(FIGURES / "reml_validation_hrfs12.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Fit diagnostics and comparison status
#
# All 120 vertex/evaluation records in this completed validation have `ok` status, and all selected extended-REML fits converged. The AR coefficients below are training-only estimates; they do not establish that AR(1) fully captures the resampling-induced noise.
#
# The progress table is a snapshot taken when this notebook is executed. A completion marker denotes a finished restart unit. Partial output directories are not completed comparisons. The synthetic cases use four synthetic runs and a 2/2 split; their numerical scores must not be pooled with the empirical twelve-run 6/6 results.

# %%
reml_status = pd.concat([table.assign(session=s) for s, table in reml_tables.items()])
display(reml_status.groupby(["session", "method", "status"]).size().rename("records").to_frame())
reml_diagnostics = pd.read_csv(EXPERIMENT / "reports/reml_validation_diagnostics.tsv", sep="\t")
display(reml_diagnostics[reml_diagnostics.method.isin(["remlar1", "remlcombined"])]
        .pivot(index=["session", "evaluation"], columns="method", values="rho").round(4))
print("Selected extended fits all converged:", bool(reml_diagnostics.all_converged.all()))
from datetime import datetime, timezone
print("Completion snapshot (UTC):", datetime.now(timezone.utc).isoformat(timespec="seconds"))
progress = []
for name, planned, scope in [
    ("full-vtc-lbfgsb", 16, "Full VTC: library comparisons + L-BFGS-B, two RT models"),
    ("noise-rt-lbfgsb", 288, "Synthetic noise/RT: original methods + L-BFGS-B"),
    ("full-vtc-reml", 2, "Full VTC: original + extended REML, primary amplitude RT"),
    ("noise-rt-reml", 288, "Synthetic noise/RT: original + extended REML")]:
    completed = len(list((OUTPUTS / name).glob("*/COMPLETE.json")))
    progress.append({"scope": scope, "completion_markers": completed, "planned_units": planned,
                     "all_units_complete": completed == planned, "output_directory": name})
display(pd.DataFrame(progress))

# %% [markdown]
# For implementation details and validation limits, see [the REML validation report](reml_validation.md) and [the covariance/noise specification](reml_model.md). The full-VTC and simulation comparisons are not yet summarized here as final results. Their progress counts refresh when this notebook is rerun; the results above always refer to the completed four-vertex validation.

# %% [markdown]
# ## Provenance and next steps
#
# The saved run fingerprints below identify the inputs, library, implementation, and environment used for these results. The all-VTC runs precede later CV efficiency/cache changes; the optimized code reproduced the earlier 256-vertex pilot scores exactly. Historical manifests are retained.
#
# The planned ten-session study and repeated split evaluation remain outstanding, as do full-VTC continuous optimization and extended noise/RT sensitivity checks. These results support continuing staged pilots; they are not a completed full-study comparison.
#
# See [the pilot report](pilot_report.md) and [the original experimental plan](plan.md). Standalone figures are saved in `reports/figures/`.

# %%
fingerprints = pd.DataFrame({s: {k: v for k, v in m.items() if k.endswith("fingerprint")}
                             for s, m in manifests.items()}).T
with pd.option_context("display.max_colwidth", None):
    display(fingerprints.T)
print("Figures saved to:", FIGURES)
