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
# # Regularized AR(1) HRF-selection pilot
#
# This pilot asks whether allowing **regularized spatial differences in noise autocorrelation** improves on a single pooled AR(1) coefficient. Sessions **11 and 16** were chosen to contrast relatively stable and unstable pooled-AR(1) results; this is an exploratory comparison, not an independent confirmatory sample.
#
# HRFs are fitted at **256 fixed vertices**, sampled before these fits from the existing common shape mask. Noise maps use all **1,750 anatomical VTC vertices** in the training runs. Each session uses the original complementary **6/6 run splits**, full **533-candidate HRF library**, binary conditions, RT amplitude, separate missing/incorrect junk regressors, NSD motion24, and 80-second cosine high-pass regressors. Only the HRF is transferred for the predictive score; test-run amplitudes are refitted using the same unwhitened OLS evaluation for every method.
#
# - **Pooled AR(1):** one median, bias-corrected coefficient over training runs and vertices, matching the existing implementation.
# - **Shrinkage AR(1):** per-vertex mean on the `atanh(rho)` scale, shrunk toward the pooled training estimate. Run-to-run variability estimates uncertainty; its variance is moderated toward the median with four prior degrees of freedom. A nonnegative moment estimate determines between-vertex variance. This is approximate empirical Bayes and treats real run variation as uncertainty as well as sampling noise.
# - **Surface AR(1):** average bias-corrected coefficients across training runs, then smooth using an area-weighted 5-mm FWHM Gaussian of mesh-edge geodesic distance. Use subject-specific fsLR midthickness surfaces, separate hemispheres, and VTC-only contributions with normalized boundary weights. This smooths noise coefficients, not BOLD signals or selected HRFs.
# - **Original REML and library CV:** existing results on the identical vertices and splits, included as references.
#
# All noise estimates use the same fixed fast/canonical-like/slow task-and-junk reference model, with the existing residual-projection bias correction. Coefficients are limited to ±0.8. Spatial coefficients are rounded to 0.02-wide groups to share design calculations; the pooled value is kept exact. No test-run data choose the noise map or regularization strength.
#
# The [surface-regularization study by Parlak et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9847678/) motivates the local method; our mesh-edge implementation and VTC-restricted neighborhood are an approximation, not a reproduction of that study.
#
# Set `no_long_text_outputs = False` below to see detailed tables. All figures use saved report data; the NSD drive is not needed to rerun this notebook.

# %%
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
# %matplotlib inline
no_long_text_outputs = True
root=next((p/'experiments/hrf_estimation' for p in [Path.cwd(),*Path.cwd().parents]
           if (p/'experiments/hrf_estimation/reports/regularized_ar1/summary.tsv').exists()),None)
if root is None:
    raise FileNotFoundError('Run inside the boldtailor checkout containing the pilot reports.')
reports=root/'reports/regularized_ar1'
summary=pd.read_csv(reports/'summary.tsv',sep='\t')
white=pd.read_csv(reports/'whitening.tsv.gz',sep='\t')
rho=pd.read_csv(reports/'rho.tsv.gz',sep='\t')
methods=['Library CV','Original REML','pooled','shrinkage','surface']
labels={'Library CV':'Library CV','Original REML':'Original REML','pooled':'Pooled AR(1)',
        'shrinkage':'Shrinkage AR(1)','surface':'Surface AR(1)'}
colors=dict(zip(methods,plt.get_cmap('tab10').colors))
plt.rcParams.update({'figure.dpi':120,'axes.spines.top':False,'axes.spines.right':False})
print(f"Scores and HRF curves use {summary.n_common.iloc[0]} vertices valid in both halves, both sessions, and all five methods.")
print('Noise distribution plots include all 256 pilot vertices; whitening summaries use the common mask.')
display(summary.round(5))

# %% [markdown]
# ## Transfer performance and HRF stability
#
# R² is computed by pooling sums of squares over both transfer directions and matched vertices. The shape statistic is the median same-vertex Pearson correlation between HRFs selected using the two independent training halves. The comparison mask is fixed across both sessions and all methods. These descriptive pilot differences do not provide population-level significance tests.

# %%
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for ax,session in zip(axes,[11,16]):
    data=summary[summary.session.eq(session)].set_index('method').reindex(methods)
    delta=100*(data.r2_task_partial-data.loc['pooled','r2_task_partial'])
    ax.barh([labels[m] for m in methods],delta,color=[colors[m] for m in methods])
    ax.axvline(0,color='0.4',lw=1);ax.invert_yaxis()
    ax.set(title=f'Session {session}',xlabel='Partial task R² difference vs pooled AR(1) (percentage points)')
plt.show()
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for ax,session in zip(axes,[11,16]):
    data=summary[summary.session.eq(session)].set_index('method').reindex(methods)
    ax.barh([labels[m] for m in methods],data.median_shape_r,color=[colors[m] for m in methods])
    ax.set_xlim(min(.8,summary.median_shape_r.min()-.01),1);ax.invert_yaxis()
    ax.set(title=f'Session {session}',xlabel='Median split-half HRF curve correlation')
plt.show()

# %% [markdown]
# ## Noise coefficients and their split-half consistency
#
# These plots use the coefficients before rounding into computational groups. Histograms show both training halves; scatterplots compare the same vertex across halves. A pooled map is constant across vertices, so spatial correlation is undefined and is not reported as evidence of reliability. Greater map agreement alone does not establish more accurate whitening.

# %%
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for ax,session in zip(axes,[11,16]):
    for method in ['shrinkage','surface']:
        values=rho[(rho.session==session)&(rho.method==method)].rho
        ax.hist(values,bins=np.linspace(-.2,.8,31),histtype='step',lw=2,label=labels[method],color=colors[method])
    for i,value in enumerate(rho[(rho.session==session)&rho.method.eq('pooled')].rho.unique()):
        ax.axvline(value,color=colors['pooled'],ls='--',label='Pooled halves' if i==0 else None)
    ax.set(title=f'Session {session}',xlabel='Training rho before grouping',ylabel='Vertex–half count');ax.legend(fontsize=8)
plt.show()
fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
map_rows=[]
for i,session in enumerate([11,16]):
    for j,method in enumerate(['shrinkage','surface']):
        data=rho[(rho.session==session)&(rho.method==method)].pivot(index='feature_id',columns='evaluation',values='rho')
        x,y=data.transfer01,data.transfer02
        corr=np.corrcoef(x,y)[0,1]
        map_rows.append(dict(session=session,method=labels[method],rho_map_r=corr,median_absolute_difference=np.median(abs(x-y))))
        ax=axes[i,j];ax.scatter(x,y,s=8,alpha=.5,color=colors[method]);ax.plot([-.2,.8],[-.2,.8],color='0.5',ls='--')
        ax.set(title=f'{session}: {labels[method]} (r={corr:.3f})',xlabel='Training half 1 rho',ylabel='Training half 2 rho',xlim=(-.2,.8),ylim=(-.2,.8))
plt.show()
display(pd.DataFrame(map_rows).round(4))

# %% [markdown]
# The fitted empirical-Bayes shrinkage is **mild**: median weights on the vertex-specific departure from the pooled mean are about **0.94–0.96**. Under this approximate model, between-vertex variation is large relative to uncertainty in the six-run means. Thus this pilot does not test strong shrinkage. The table below summarizes all 1,750 noise-map vertices, including the fraction reaching the inherited upper bound of 0.8.

# %%
display(pd.read_csv(reports/'shrinkage_strength.tsv',sep='\t').round(4))

# %% [markdown]
# ## Held-out residual-whitening diagnostics
#
# Fit the same flexible reference task/junk/nuisance design to each held-out run, first by OLS and then by GLS using the **training-derived** coefficient. Summaries show median absolute lag-1 residual autocorrelation and median sum of squared residual autocorrelations over lags 1–10. Lower is better descriptively, but regression itself changes the residual ACF: these are not calibrated whiteness tests or false-positive-rate estimates. Each plotted median pools vertex–run observations on the common mask; observations are not independent replications.
#
# A coefficient can improve these noise diagnostics without improving HRF selection. Conversely, greater HRF stability can result from stronger constraints without greater accuracy.

# %%
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
whitening_rows=[]
for i,session in enumerate([11,16]):
    for j,metric in enumerate(['abs_lag1','acf_energy']):
        data=white[white.session.eq(session)]
        before=data[data.method.eq('pooled')]['before_'+metric].median()
        vals=[before]+[data[data.method.eq(m)]['after_'+metric].median() for m in ['pooled','shrinkage','surface']]
        ax=axes[i,j];ax.bar(['Before','Pooled','Shrinkage','Surface'],vals,color=['gray']+[colors[m] for m in ['pooled','shrinkage','surface']])
        ax.set(title=f'Session {session}',ylabel='Median '+('absolute lag-1 ACF' if j==0 else 'squared ACF sum, lags 1–10'))
        whitening_rows.append(dict(session=session,metric=metric,before=vals[0],pooled=vals[1],shrinkage=vals[2],surface=vals[3]))
plt.show()
display(pd.DataFrame(whitening_rows).round(4))
if not no_long_text_outputs:
    display(white.groupby(['session','method','evaluation'])[['before_lag1','after_lag1']].median())

# %% [markdown]
# ## HRF curves from the two training halves
#
# Each curve averages individually peak-normalized HRFs over the common vertices. Solid and dashed lines show the complementary training halves. These are six-run estimates, not all-run HRFs. The reference methods remain in the score and stability plots above.

# %%
with np.load(reports/'mean_shapes.npz') as data:
    curves=data['curves'];times=data['times'];shape_methods=data['methods'].tolist()
fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey=True,constrained_layout=True)
for i,session in enumerate([11,16]):
    for j,method in enumerate(['pooled','shrinkage','surface']):
        ax=axes[i,j];k=shape_methods.index(method)
        for half,style in enumerate(['-','--']):
            ax.plot(times,curves[i,k,half],style,color=colors[method],label=f'Half {half+1}')
        ax.axhline(0,color='0.6',lw=.5)
        ax.set(title=f'{session}: {labels[method]}',xlabel='Time after event (s)',ylabel='Mean peak-normalized HRF',xlim=(0,36));ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## Computational grouping sensitivity
#
# The primary fits round spatial coefficients to a grid with spacing 0.02. The sensitivity report below compares spacing 0.01 on session 16, transfer direction 1, using the same training coefficients and full candidate library. This checks numerical grouping, not an independently tuned noise model.

# %%
sensitivity_path=reports/'binning_sensitivity.tsv'
if sensitivity_path.exists():
    display(pd.read_csv(sensitivity_path,sep='\t').round(6))
else:
    print('Finer-grid sensitivity has not yet completed.')

# %% [markdown]
# The finer grid preserves **98.8% of shrinkage selections** and **98.0% of surface selections**. Its partial-R² changes are **+0.0019** and **−0.0036 percentage points**, respectively; median curve correlation with the primary fit is 1.0. This supports the grouping approximation for this checked split, without establishing it for every possible dataset.

# %% [markdown]
# ## Interpretation and provenance
#
# **Regularized spatial noise maps improve whitening, but do not materially improve HRF selection in this pilot.** All 256 selected vertices have valid curves and scores in every method, session, and direction.
#
# - **Session 11:** median absolute held-out lag-1 residual correlation falls from **0.104** with pooled AR(1) to **0.071** with shrinkage and **0.073** with surface smoothing. Relative to pooled AR(1), partial task R² changes by **−0.025** and **−0.045 percentage points**, respectively. Split-half HRF correlations increase slightly, from **0.9771** to **0.9788** for both methods.
# - **Session 16:** median absolute lag-1 correlation falls from **0.121** to **0.092** and **0.094**. Partial task R² changes by **+0.021** and **+0.035 percentage points**, respectively. Split-half HRF correlations remain around **0.913–0.915**, below original REML (**0.937**) and library CV (**0.951**).
#
# This weakens the hypothesis that spatial pooling of rho is the main cause of poor AR(1) HRF stability in session 16. It does not establish that AR(1) is an adequate noise model, or that the likelihood objective aligns with the predictive evaluation. The empirical-Bayes weights indicate only mild shrinkage; the 0.8 cap is active for a minority of vertices. Residual ACF summaries are descriptive and include the effects of regression projection.
#
# The grouped pooled implementation reproduced **all selected HRFs exactly** for both halves of both sessions against the previous saved pooled-AR(1) fits. Source fit, surface, and implementation hashes are saved alongside the report. The regularization settings were fixed in advance; neither these two chosen sessions nor vertex–run observations constitute independent population replications. The existing ten-session notebook remains separate.
#
# Rebuild the main data summaries with `uv run python -m experiments.hrf_estimation.regularized_report`. Full fit artifacts are under `outputs/regularized-ar1-pilot`; portable report tables and mean curves are under `reports/regularized_ar1`.
