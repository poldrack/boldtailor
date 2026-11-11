# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
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
# # RT mechanisms and HRF recovery
#
# Four task models follow Mumford et al. (2024): fixed-duration conditions
# without RT, RT-duration conditions, fixed-duration conditions plus common
# RT-amplitude modulation, and fixed-duration conditions plus common RT duration.
# Each is fitted with iid REML and pooled AR(1) REML. Known-rho AR(1) is a
# diagnostic available only when AR(1) noise generated the data.
#
# The primary endpoint is recovery of a known HRF, not goodness of fit.
# Shape-error summaries condition on successful fits; failure rates include
# every replicate. Observed-data scores refit amplitudes on evaluation runs.
#
# Set `HRF_SIM_REPORT` to a directory produced by the `report` command. This
# notebook does not need access to NSD. A smoke report contains synthetic test
# data, not scientific results. The overview figure pools conditions; use the
# stratified table to compare matching signal, noise, and SNR settings.

# %%
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

no_long_text_outputs = True
report = Path(
    os.environ.get("HRF_SIM_REPORT", "./reports/pilot-v1")
)
print(report)
if not (report / "summary.tsv").is_file():
    raise FileNotFoundError(
        f"Generate a simulation report first, or set HRF_SIM_REPORT: {report}"
    )
summary = pd.read_csv(report / "summary.tsv", sep="\t")
paired = pd.read_csv(report / "paired.tsv", sep="\t")
noise = pd.read_csv(report / "noise.tsv", sep="\t")
stability = pd.read_csv(report / "stability.tsv", sep="\t")
display(summary.head(12) if no_long_text_outputs else summary)

# %% [markdown]
# ## Recovery and failures
# Select a session, truth, generating mechanism, duration convention and SNR
# before interpreting estimator differences. Failed fits are not assigned an
# invented HRF error; compare their frequency alongside successful-fit error.

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.imshow(plt.imread(report / "recovery.png"))
ax.axis("off")
plt.show()
display(summary.groupby(["model", "estimator"])[["n", "n_failed"]].sum())

# %% [markdown]
# ## Paired estimator differences
# Negative AR(1)-minus-iid RMSE favors AR(1). Each pair uses identical signal,
# noise, and trial timing. Missing differences indicate that at least one fit
# failed, not a zero difference. Replicates are Monte Carlo draws for one
# participant's timing; they are not a group of independent participants.

# %%
if "ar1_minus_iid_rmse" in paired:
    pairs = paired[paired.split == "full"]
    fig, ax = plt.subplots(figsize=(9, 4))
    pairs.boxplot(column="ar1_minus_iid_rmse", by="model", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Paired HRF RMSE difference")
    fig.suptitle("")
    plt.show()

# %% [markdown]
# ## Noise and split-half diagnostics
# Noise variance and ACF vary across realizations. Random-phase noise preserves
# donor spectra; its donor rho is a summary, not an exact AR(1) parameter.

# %%
display(
    noise.groupby("noise")[["variance", "acf1", "donor_variance", "donor_rho"]].mean()
)
if len(stability):
    display(
        stability.groupby(["model", "estimator"])[
            ["both_recovered", "half_shape_correlation"]
        ].mean().sort_values('half_shape_correlation', ascending=False)
    )

# %% [markdown]
# ## Example HRF shapes
# These are the first at most 200 full-session fits in sorted case order, a
# deterministic illustration rather than a representative sampling claim.

# %%
shapes = json.loads((report / "shapes.json").read_text())
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
for ax, row in zip(axes.ravel(), shapes[:6]):
    times = np.arange(len(row["truth"])) * 0.1
    ax.plot(times, row["truth"], color="black", label="True HRF")
    ax.plot(times, row["estimated"], label="Estimated HRF")
    ax.set_title(f"{row['model']} / {row['estimator']}")
    ax.set_xlabel("Seconds")
axes[0, 0].legend()
fig.tight_layout()
plt.show()
