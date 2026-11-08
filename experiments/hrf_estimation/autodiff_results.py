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
# # Autodiff for continuous HRF estimation
#
# A matched comparison of L-BFGS-B with finite-difference versus JAX gradients. **Autodiff reduced fitting time while preserving the fitted objective and held-out scores** in this eight-vertex session-11 benchmark.
#
# This notebook reads the committed benchmark table; it does not rerun optimization and needs neither the NSD drive nor JAX. Run it from the repository root or this notebook's directory.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
# %matplotlib inline

candidates = [Path.cwd(), *Path.cwd().parents]
root = next((p for p in candidates if (p / 'experiments/hrf_estimation/reports/autodiff_benchmark_session11.tsv').exists()), None)
if root is None:
    raise FileNotFoundError('Run this notebook inside the boldtailor checkout containing reports/autodiff_benchmark_session11.tsv')
data_path = root / 'experiments/hrf_estimation/reports/autodiff_benchmark_session11.tsv'
results = pd.read_csv(data_path, sep='\t')
assert not results.duplicated(['mode', 'gradient', 'feature_id']).any()
assert results['status'].eq('ok').all()
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
backends = ['finite_difference', 'autodiff_cold', 'autodiff_warm']
labels = ['Finite differences', 'Autodiff: cold', 'Autodiff: warm']
colors = ['#727b8a', '#2588a2', '#db8b30']
display({'application/json': {'kind': 'autodiff_benchmark', 'records': results.to_dict('records')}}, raw=True)

# %% [markdown]
# ## Benchmark scope
#
# - Subject 07, session 11; eight anatomical VTC vertices, shared across all comparisons.
# - Six training and six held-out runs, split seed 20261107; only the HRF is frozen on transfer, with coefficients refitted.
# - Full 533-member initialization library, four starts, fixed regularization λ = 0.01, maxiter = 500, maxfun = 5000, one worker.
# - Binary task condition, separate missing/incorrect response regressors, NSD motion model (24 parameters), cosine HPF at 80 seconds, no censoring.
# - Both RT amplitude and RT duration models. All backends use the same scientific objective and environment.
#
# CPU environment: JAX/jaxlib 0.12.0.1, NumPy 2.2.6, SciPy 1.16.3, 64-bit JAX arithmetic. Implementation commit: `cda1183`. Raw fits and configuration are stored locally under `outputs/autodiff-benchmark-session11/`; the tracked table is sufficient for every figure below.

# %% [markdown]
# ## Fitting time and function evaluations
#
# Each timing is for the complete eight-vertex fit, including initialization and any compilation, but excluding held-out scoring. The same group timing is repeated on each vertex record: **take it once, do not sum it**. Function evaluations are summed across vertices. Cold runs include compilation; warm runs reuse compiled functions. These are single timing observations, not replicated runtime estimates.

# %%
summary = results.groupby(['mode', 'gradient'], sort=False).agg(
    fit_seconds=('fit_seconds', 'first'),
    function_evaluations=('function_evaluations', 'sum'),
    vertices=('feature_id', 'nunique'))
assert results.groupby(['mode', 'gradient']).fit_seconds.nunique().eq(1).all()
for mode in ['amplitude', 'duration']:
    reference = summary.loc[(mode, 'finite_difference'), 'fit_seconds']
    summary.loc[mode, 'speedup'] = reference / summary.loc[mode, 'fit_seconds'].to_numpy()
display(summary.round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, metric, title in zip(axes, ['fit_seconds', 'function_evaluations'], ['Eight-vertex fitting time (s)', 'Total function evaluations']):
    values = summary[metric].unstack('gradient').reindex(index=['amplitude', 'duration'], columns=backends)
    values.plot.bar(ax=ax, color=colors, rot=0)
    ax.set(title=title, xlabel='RT model', ylabel=title)
    ax.legend(labels, fontsize=8)
plt.show()

# %% [markdown]
# ## Agreement of fitted objectives and held-out scores
#
# The optimization objective includes the fixed regularization penalty. Held-out total R² measures the full model; partial task R² and incremental task R² measure task contributions. Differences below compare each autodiff fit with its matched finite-difference vertex. No pooling of per-vertex R² is used.

# %%
metrics = ['objective', 'r2_full', 'r2_task_partial', 'delta_r2_task']
reference = results.query("gradient == 'finite_difference'").set_index(['mode', 'feature_id'])
comparisons = []
for backend in backends[1:]:
    current = results.query('gradient == @backend').set_index(['mode', 'feature_id'])
    differences = current[metrics] - reference[metrics]
    differences['gradient'] = backend
    comparisons.append(differences.reset_index())
differences = pd.concat(comparisons, ignore_index=True)
display(differences.groupby(['mode', 'gradient'])[metrics].agg(lambda x: x.abs().max()).style.format('{:.3e}'))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, mode in zip(axes, ['amplitude', 'duration']):
    x = reference.loc[mode, 'r2_task_partial']
    y = results.query("mode == @mode and gradient == 'autodiff_cold'").set_index('feature_id').loc[x.index, 'r2_task_partial']
    limits = [min(x.min(), y.min()) - .02, max(x.max(), y.max()) + .02]
    ax.plot(limits, limits, '--', color='0.5', linewidth=1)
    ax.scatter(x, y, color=colors[1], s=45)
    ax.set(title=f'RT {mode}', xlabel='Finite differences: partial task R²', ylabel='Autodiff: partial task R²', xlim=limits, ylim=limits)
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, mode in zip(axes, ['amplitude', 'duration']):
    selected = differences.query("mode == @mode and gradient == 'autodiff_cold'")
    ax.axhline(0, color='0.5', linewidth=1)
    ax.scatter(np.arange(len(selected)), selected.r2_task_partial, color=colors[1])
    ax.set_xticks(np.arange(len(selected)), selected.feature_id.astype(str), rotation=55, ha='right')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set(title=f'RT {mode}: paired score differences', xlabel='VTC vertex ID', ylabel='Autodiff − finite differences: partial task R²')
plt.show()

# %% [markdown]
# ## Interpretation and limits
#
# Autodiff was approximately **3.8–3.9× faster with compilation included**, and **5.4–6.0× faster with compilation reused**. Objective differences were below 1.2 × 10⁻¹¹; held-out partial-R² differences were below 2.2 × 10⁻⁷. Warm and cold autodiff results agree. All 48 score records were valid.
#
# The double-gamma HRF is differentiated through event sampling and normalization. Linear coefficients are solved exactly; the variable-projection gradient holds those optimal coefficients constant during differentiation. Peak normalization is piecewise differentiable and rank transitions remain a potential difficulty.
#
# This is a **fixed-regularization, eight-vertex benchmark**, not a full-VTC or nested-tuning runtime estimate. Compilation is cached separately per worker, and timings were collected while other local analysis jobs were running. A matched nested-tuning pilot is needed before projecting these gains to the full workflow. The existing full-VTC jobs were not changed for this benchmark.
#
# See [benchmark notes](autodiff_benchmark.md) for implementation and environment details.
