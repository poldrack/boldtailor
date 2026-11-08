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
# # HRF models across sessions: current and historical results
#
# Snapshot of sub-07 sessions **10–19**, including canonical HRF, library CV, original REML, parametric-library variants, continuous L-BFGS-B, AR(1), and the historical complex REML variants. Earlier pilots and Powell are retained in a separate historical section.
#
# **Coverage is incomplete across methods.** Only AR(1) covers all ten sessions. Missing results are not zero scores. Full-VTC RT-amplitude and RT-duration analyses are shown separately. This notebook reads portable report snapshots and does not require the NSD drive or rerun any analysis. Existing notebooks are unchanged.

# %%
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from IPython.display import display
# %matplotlib inline

root = next((p / 'experiments/hrf_estimation' for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'experiments/hrf_estimation/reports/all_methods_summary.tsv').exists()), None)
if root is None:
    raise FileNotFoundError('Run from inside the boldtailor checkout containing reports/all_methods_summary.tsv')
reports = root / 'reports'
summary = pd.read_csv(reports / 'all_methods_summary.tsv', sep='\t')
coverage = pd.read_csv(reports / 'all_methods_coverage.tsv', sep='\t')
historical = pd.read_csv(reports / 'all_methods_historical.tsv', sep='\t')
strict = pd.read_csv(reports / 'all_methods_converged.tsv', sep='\t')
provenance = pd.DataFrame(json.loads((reports / 'all_methods_provenance.json').read_text()))
methods = coverage.method.drop_duplicates().tolist()
colors = dict(zip(methods, plt.get_cmap('tab10').colors))
plt.rcParams.update({'figure.dpi':120, 'axes.spines.top':False, 'axes.spines.right':False})
for kind, frame in [('all_methods_summary',summary),('model_coverage',coverage),('historical_models',historical)]:
    serial = json.loads(frame.to_json(orient='records',double_precision=15))
    display({'application/json': {'kind':kind, 'records':serial}}, raw=True)
print(f'{len(summary)} complete model/session/RT combinations; sessions 10–19.')

# %% [markdown]
# ## What was modeled
#
# | Approach | HRF selection / fitting |
# |---|---|
# | Canonical | Fixed canonical HRF |
# | Library CV | Inner cross-validation over the full 533-HRF library |
# | Parametric library CV | CV restricted to the parametric library |
# | Original REML | Direct HRF likelihood comparison with one random task/junk variance |
# | Parametric REML | Original REML restricted to the parametric library |
# | AR(1) REML | Training-only pooled temporal-noise correction, then scalar REML |
# | Continuous L-BFGS-B | Continuous double-gamma optimization with nested regularization tuning |
# | Separate components | Separate condition, RT and junk variance components |
# | Shared effects | Adds task effects shared across training runs |
# | Combined | AR(1), separate components and shared task effects |
#
# Separate components, shared effects and combined were discontinued for future runs; their completed results remain useful sensitivity analyses. Powell appears only in the historical optimizer smoke check.
#
# The current comparisons use anatomical VTC (1,750 vertices), binary trial type, separate missing/incorrect junk regressors, NSD motion confounds (24 parameters), and an 80-second cosine HPF. Only the HRF is frozen on transfer; amplitudes are refitted per held-out run. Each session has one complementary 6/6 partition, evaluated in both directions. This is HRF transfer scoring, not fully frozen-model prediction.

# %% [markdown]
# ## Availability
#
# **Complete** means all-run and both transfer fits are saved. **Partial** means some fits exist, but the model is excluded from comparative scores. A finished individual method can be recovered from a stopped multi-method queue; those sources are identified in the provenance table below. For complete directories, artifact hashes were checked. For recovered methods, all required fit files and expected evaluation run IDs were checked and their hashes recorded.

# %%
fig, axes = plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
for ax, mode in zip(axes,['amplitude','duration']):
    table=coverage.query('mode == @mode').pivot(index='method',columns='session',values='state').reindex(index=methods,columns=range(10,20))
    values=table.replace({'missing':0,'partial':1,'complete':2}).to_numpy(dtype=float)
    ax.imshow(values,aspect='auto',cmap=ListedColormap(['#eeeeee','#e9ba61','#329a91']),vmin=0,vmax=2)
    ax.set_xticks(range(10),range(10,20));ax.set_yticks(range(len(methods)),methods)
    ax.set(title=f'RT {mode}',xlabel='Session')
    for i in range(len(methods)):
        for j in range(10):
            ax.text(j,i,['—','P','✓'][int(values[i,j])],ha='center',va='center',fontsize=9)
fig.legend(handles=[Patch(color=c,label=l) for c,l in [('#eeeeee','Missing'),('#e9ba61','Partial'),('#329a91','Complete')]],loc='outside lower center',ncol=3)
plt.show()


# %% [markdown]
# ## Held-out performance across sessions
#
# For each session and RT mode, all methods are compared on the same vertices with finite scores in both transfer directions. SSE and denominators are summed across vertices and held-out runs **before** forming R². Total R² includes all regressors; partial task R² measures task improvement relative to nuisance plus junk. Incremental task R² uses total signal variance as its denominator.
#
# The common vertex set depends on the methods available in that session, so these plots are descriptive session comparisons, not a longitudinal test on a globally fixed mask. Sessions 14–19 contain AR(1) only and cannot establish its advantage over another method. Gaps are left blank, not interpolated.

# %%
def plot_sessions(frame, metrics, titles, title):
    fig, axes=plt.subplots(1,len(metrics),figsize=(13,4.5),constrained_layout=True)
    axes=np.atleast_1d(axes)
    for method in methods:
        g=frame[frame.method.eq(method)].set_index('session').reindex(range(10,20))
        if not g[metrics[0]].notna().any(): continue
        for ax,metric,label in zip(axes,metrics,titles):
            ax.plot(g.index,g[metric],marker='o',markersize=4,color=colors[method],label=method)
            ax.set(xlabel='Session',ylabel=label,xticks=list(range(10,20)))
    fig.suptitle(title)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8)
    plt.show()

amplitude=summary.query("mode == 'amplitude'")
plot_sessions(amplitude,['r2_full','r2_task_partial'],['Total held-out R²','Partial task held-out R²'],'RT amplitude: pooled transfer performance')
display(amplitude[['session','method','n_common','n_methods','r2_full','r2_task_partial','delta_r2_task']].style.format({'r2_full':'{:.5f}','r2_task_partial':'{:.5f}','delta_r2_task':'{:.5f}'}))

# %% [markdown]
# ## Matched comparisons with earlier modeling approaches
#
# Sessions 11 and 12 contain canonical, library-CV, original REML, parametric-library and continuous results. The differences below use the full-library CV score as the reference within each session, with the same common vertices used for every method. Values are **percentage-point changes in partial task R²**, not percent relative improvements. Complex-model coverage differs between the two sessions.
#
# For RT amplitude, continuous optimization exceeds full-library CV by only **0.043 percentage points in session 11** and **0.031 points in session 12**. It is highest in session 11; parametric-library CV is slightly higher in session 12. The much larger difference is between canonical and adapted HRFs. The lower panels expand the scale to make the small differences among adapted methods visible.

# %%
fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
for column,session in enumerate([11,12]):
    g=amplitude[amplitude.session.eq(session)].set_index('method').reindex(methods)
    delta=100*(g.r2_task_partial-g.loc['Library CV','r2_task_partial'])
    for row in [0,1]:
        ax=axes[row,column]
        valid=delta.notna()
        if row==1: valid &= delta.index!='Canonical'
        names=np.array(methods)[valid]
        ax.barh(names,delta[valid],color=[colors[m] for m in names])
        ax.axvline(0,color='0.4',linewidth=1)
        ax.set(title=f'Session {session}: '+('all methods' if row==0 else 'adapted HRFs, expanded scale'),xlabel='Partial task R² difference vs library CV (percentage points)')
        ax.invert_yaxis()
plt.show()

# %% [markdown]
# ## RT-duration sensitivity
#
# These results use the RT-duration formulation, which changes the model. Session 12 continuous optimization was stopped before producing a complete result and is therefore missing. The amplitude and duration panels use independently matched vertex sets; they should not be treated as paired RT-model effect estimates.

# %%
duration=summary.query("mode == 'duration'")
plot_sessions(duration,['r2_full','r2_task_partial'],['Total held-out R²','Partial task held-out R²'],'RT duration: available comparisons')
display(duration[['session','method','n_common','r2_full','r2_task_partial','delta_r2_task']].round(5))

# %% [markdown]
# ## HRF stability and fit diagnostics
#
# Shape stability is the median Pearson correlation between each vertex’s two HRFs estimated from complementary training halves, on the same common vertex set. Greater agreement can arise from shrinkage and does not by itself prove greater accuracy. Nonconverged fits with finite scores are retained in the primary results; the converged-only sensitivity below excludes them jointly across all available methods and both directions.

# %%
plot_sessions(amplitude,['median_curve_correlation'],['Median split-half HRF curve correlation'],'RT amplitude: HRF shape agreement')
checks=summary.merge(strict,on=['session','mode','method'],suffixes=('','_converged'))
checks['converged_minus_primary']=checks.r2_task_partial_converged-checks.r2_task_partial
display(checks[['session','mode','method','n_common','n_common_converged','nonconverged_records','zero_variance_records','converged_minus_primary']].round(6))

# %% [markdown]
# ## Earlier pilots and optimizer implementations
#
# The table retains the earlier canonical/library/original-REML pilots, their full-VTC versions, the four-vertex REML validation, and the L-BFGS-B/Powell smoke comparison. **Compare methods within a source only**: vertex sets, run counts, library sizes, optimizer budgets and code revisions differ. Some early outputs have legacy completion markers without artifact hashes; their validation level is shown.
#
# The Powell comparison used four vertices and a 2/2 run split. Its selected regularization differs from L-BFGS-B, so it is not a controlled comparison of optimizer speed at a fixed objective. It is not evidence about full-VTC Powell performance. The newer autodiff benchmark is in [autodiff_results.ipynb](autodiff_results.ipynb); autodiff changes the numerical implementation, not the scientific model.

# %%
display(historical.style.format({'r2_full':'{:.5f}','r2_task_partial':'{:.5f}','seconds':'{:.2f}','regularization':'{:.3g}'}))
smoke=historical.query("source == 'continuous-smoke'")
fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
for ax,metric,label in zip(axes,['r2_task_partial','seconds'],['Pooled partial task R²','Recorded fit time (seconds)']):
    ax.bar(smoke.method,smoke[metric],color=['#2588a2','#db8b30'])
    ax.set(ylabel=label,title='Historical four-vertex smoke check')
plt.show()

# %% [markdown]
# ## Scope, provenance and interpretation
#
# - This is an empirical session comparison. Synthetic noise/RT sweeps have no NSD session identity and are not pooled into these plots.
# - Complex REML variants remain historical sensitivity results; no further runs are implied by their inclusion.
# - AR(1) sessions 10–13 used the earlier generic solver; sessions 14–19 used the new scalar solver. Full session-11 validation matched all 1,750 HRF winners, but earlier transfer convergence flags are retained as recorded.
# - The original REML versus AR(1)/complex comparison uses common held-out OLS scoring, not whitened likelihood or inferential calibration.
# - No across-session grand ranking is computed: methods have unequal session coverage. Conclusions should rely on within-session matched comparisons and consistency across the available sessions.
#
# The supporting tables include pooled total/partial/incremental R², converged-only sensitivity, coverage, and historical results. The compressed vertex-level snapshot allows recomputation without the NSD drive. Source paths and artifact hashes are retained in `reports/all_methods_provenance.json`.

# %%
display(provenance[['session','mode','method','source','whole_directory_complete']].sort_values(['session','mode','method']).reset_index(drop=True))
print('Portable vertex-level snapshot:',reports/'all_methods_scores.tsv.gz')
print('Insample results (descriptive; separate common-vertex counts):')
display(summary[['session','mode','method','insample_n_common','insample_r2_full','insample_r2_task_partial']].round(5))
