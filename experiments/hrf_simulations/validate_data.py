# %% [markdown]
# # Load and validate the NSD simulation bundle
# Read-only checks of all exported runs, followed by an example session load.
# Missing/incorrect responses are valid junk categories, not validation failures.
# The default checks three BOLD columns per run; enable the full scan to check
# every stored signal value. Neither mode revalidates anatomical registration
# or compares the bundle with the original NSD files.

# %%
# ruff: noqa: E402 -- Allow opening this notebook from the experiment directory.
import os
from pathlib import Path
import sys

root = next(
    p
    for p in (Path.cwd(), *Path.cwd().parents)
    if (p / "experiments/hrf_simulations").is_dir()
)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from experiments.hrf_simulations.bundle import inventory, roi_inventory, load_session
from experiments.hrf_simulations.bundle_validation import validate_bundle

bundle_path = Path(
    os.environ.get("HRF_SIM_BUNDLE", "/Volumes/extdata1/NSD/nsd_three_rois.h5")
)
full_signal_scan = False  # True reads all BOLD values, one run at a time.
no_long_text_outputs = True
assert bundle_path.is_file(), f"Bundle not found: {bundle_path}"
print(f"{bundle_path.name}: {bundle_path.stat().st_size / 2**30:.2f} GiB")

# %% [markdown]
# ## Contents and validation
# Global checks cover feature IDs, brain-axis length, ROI membership, and run count.
# Every run is checked for aligned signal/timing dimensions, regular sampling,
# valid event categories, and an aligned motion-24 plus 80-second HPF design.
# Skipped exports, constant vertices, and out-of-acquisition events are reported
# separately for inspection; the latter two are not automatically rejected.

# %%
subjects = inventory(bundle_path)
display(
    pd.DataFrame(
        [
            dict(subject=s, sessions=len(v), first_session=min(v), last_session=max(v))
            for s, v in subjects.items()
        ]
    )
)
display(
    pd.Series(roi_inventory(bundle_path), name="vertices").rename_axis("ROI").to_frame()
)
validation = validate_bundle(bundle_path, full_signal_scan=full_signal_scan)
run_checks = validation["runs"]
display(validation["global_checks"])
display(
    run_checks.groupby("subject").agg(runs=("run_id", "size"), passed=("ok", "sum"))
)
print(
    f"Signal checks: {'all vertices' if full_signal_scan else 'three sampled vertices per run'}; {len(validation['skipped'])} runs omitted during export."
)
failures = run_checks[~run_checks.ok]
if len(failures):
    display(failures.head(10) if no_long_text_outputs else failures)

# %% [markdown]
# ## Coverage, omissions, and basic data quality
# The tables below retain every run in `run_checks` and every omission in
# `validation['skipped']`; only displayed rows are limited.

# %%
coverage = (
    run_checks.groupby(["subject", "session"]).size().unstack("session", fill_value=0)
)
fig, ax = plt.subplots(figsize=(13, 3))
im = ax.imshow(coverage, aspect="auto", vmin=0)
ax.set_yticks(range(len(coverage)), coverage.index)
ax.set_xticks(range(0, len(coverage.columns), 4), coverage.columns[::4])
ax.set(xlabel="Session", title="Exported runs per subject/session")
fig.colorbar(im, ax=ax, label="Runs")
plt.show()
skipped = validation["skipped"]
if len(skipped):
    display(skipped.head(8) if no_long_text_outputs else skipped)
valid_checks = run_checks[run_checks.ok]
display(
    valid_checks[["n_frames", "tr", "n_valid", "n_missing", "n_incorrect", "rt_median"]]
    .describe()
    .round(3)
)
print("Constant checked vertices:", int(valid_checks.constant_vertices.sum()))
print("Events outside acquisition:", int(valid_checks.events_outside_acquisition.sum()))
assert validation["global_checks"].ok.all() and failures.empty, (
    "Validation failures: inspect the tables above."
)

# %% [markdown]
# ## Load one session for analysis
# Change these selectors to inspect another subject/session/ROI. This loads all
# runs for one ROI using the same reader as the simulation workflow.

# %%
subject, session, roi = "sub-07", 11, "fusiform"
runs = load_session(bundle_path, subject, session, roi=roi)
example = runs[0]
print(
    f"{len(runs)} runs loaded; first run {example.run_id}: {example.signals.shape} (frames × vertices)"
)
display(example.events.head())
print(f"Nuisance design: {example.confounds.shape}; ROI: {roi}")
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
signal = example.signals[:, 0]
axes[0].plot(example.frame_times, signal - signal.mean())
axes[0].set(
    xlabel="Seconds", ylabel="Mean-centered stored BOLD", title="One vertex, first run"
)
rt = pd.concat([r.events for r in runs]).response_time
axes[1].hist(rt[np.isfinite(rt) & (rt > 0)], bins=30)
axes[1].set(
    xlabel="Response time (seconds)",
    ylabel="Trials",
    title="Positive observed RTs, session",
)
plt.tight_layout()
plt.show()
