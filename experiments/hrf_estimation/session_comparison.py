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
# # HRF model comparisons
#
# Three complete comparisons for sub-07:
#
# 1. **Sessions 10–19:** canonical, full-library CV, parametric-library CV, original REML, and AR(1) REML.
# 2. **Sessions 10–19:** original REML with and without RT, with crossed evaluation and HRF shape comparisons.
# 3. **Sessions 10 and 11:** original REML, AR(1), separate components, shared effects, and combined.
#
# No partial-coverage approaches, optimizer pilots, RT-duration results, or synthetic experiments are included. Results are saved snapshots; rerunning this notebook needs neither the NSD drive nor any model fitting.
#
# ### What the models have in common
#
# The **hemodynamic response function (HRF)** describes how a brief neural event produces a BOLD response over time. These approaches differ in how they choose its shape. Each adaptive method selects **one HRF per cortical vertex**, shared by the condition, reaction-time (RT), and junk regressors at that vertex. The selected shape is shared across the runs used for fitting; it is not a separate HRF for each condition or trial.
#
# Except for the explicit no-RT ablation, the event model includes the two binary trial-type conditions, an RT-amplitude regressor, and separate regressors for missing and incorrect responses. Motion and low-frequency drift are nuisance effects. In the RT-amplitude formulation, reaction time modulates a regressor's amplitude rather than the event's duration.
#
# For the transfer comparisons, HRF selection uses only the training runs. **Only the selected HRF is frozen:** all regression amplitudes are estimated again in each held-out run. Thus the comparison asks whether an HRF learned from one set of runs describes another set well, rather than whether a completely frozen model predicts new runs.
#
# ### Models compared across all ten sessions
#
# | Model | What it estimates and how it chooses the HRF |
# |---|---|
# | **Canonical** | Uses the same predefined canonical double-gamma HRF at every vertex. It estimates regression amplitudes but does not adapt the HRF shape to the data. This is the fixed-HRF reference. |
# | **Library CV** | Evaluates the full library of **533 candidate HRFs**: the canonical HRF, 512 additional parametric shapes, and 20 empirical shapes. Within the outer training set, it leaves out one run at a time, fits task coefficients on the remaining runs, and evaluates task prediction on the omitted run after accounting for nuisance and junk effects. Each vertex receives the candidate with the best inner cross-validation score. |
# | **Parametric library CV** | Uses the same cross-validation procedure, but only the **513 parametric HRFs**—the canonical shape and the 512 additional double-gamma shapes. Comparing it with full-library CV tests whether adding empirical HRF shapes improves transfer performance. This is discrete library selection, not continuous parameter optimization. |
# | **Original REML** | Searches the full 533-HRF library using **restricted maximum likelihood (REML)** instead of inner cross-validation. Nuisance coefficients are fixed effects. The condition, RT, and junk coefficients are modeled as zero-mean random effects with **one common variance component**, independent across runs. For each candidate and vertex, the model estimates that variance relative to residual noise and chooses the HRF with the best restricted likelihood. Residual temporal noise is assumed independent. |
# | **Original REML without RT** | Uses the original REML likelihood and full HRF library but removes the RT-amplitude regressor. Two condition and two junk coefficients remain, with one common random-effect variance. Trial inclusion and the fixed nuisance model are unchanged. Section 2 compares this model with original REML under both scoring designs. |
# | **AR(1) REML** | Uses the same one-component task model and HRF-selection criterion as original REML, but accounts for correlation between adjacent time points with a first-order autoregressive noise model. One autocorrelation coefficient is estimated from the training data, pooled across runs and vertices, and held fixed across HRF candidates. The data and design are whitened before REML selection. This changes the noise assumption used to choose the HRF; it does not add HRF-shape parameters. |
#
# **CV versus REML:** CV chooses the HRF by predicting omitted training runs. REML chooses it by a likelihood that accounts for fixed nuisance effects and estimates the variance of random task effects. The random-effect coefficients here are condition/RT/junk regression coefficients, not a separate random amplitude for every trial.
#
# ### Additional REML variants compared in sessions 10 and 11
#
# These variants keep the full HRF library and REML selection rule, but change the covariance assumptions for the task coefficients. Their variance components are estimated separately for each vertex and HRF candidate.
#
# | Variant | Change relative to original REML |
# |---|---|
# | **Separate components** | Replaces the single task/junk variance with **three variances**: one shared by the two condition coefficients, one for RT, and one shared by the missing/incorrect junk coefficients. Coefficients remain independent across runs, and residual temporal noise remains independent. This lets condition, RT, and junk effects have different amounts of shrinkage. |
# | **Shared effects** | Adds a **shared task component across training runs**, alongside independent run-specific task/junk effects. There are two variance components: independent task/junk effects and shared condition/RT effects. Junk effects are not shared across runs. Residual temporal noise remains independent. This represents task responses that are consistent across runs while allowing run-specific deviations. |
# | **Combined** | Combines AR(1) whitening with separate variances and shared task effects. It has **five task variance components**: independent conditions, RT, and junk, plus shared conditions and shared RT. It allows different shrinkage by effect type, cross-run task consistency, and temporal noise correlation. |
#
# All methods are evaluated with the **same unwhitened held-out OLS scoring procedure**. Consequently, the reported R² values compare the transfer quality of their selected HRFs, not the quality of their noise estimates or the calibration of statistical tests. The more complex REML variants are included here only as completed sensitivity analyses.
#
#
# **Output control:** `no_long_text_outputs = True` in the first code cell suppresses detailed tables and raw record dumps. Set it to `False` and rerun the notebook to show them. Plots and compact summaries remain visible.
#

# %%
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
# %matplotlib inline
# Set False to display detailed tables and raw JSON records.
no_long_text_outputs = True


def display_long_output(value, **kwargs):
    """Display detailed output only when requested."""
    if not no_long_text_outputs:
        display(value, **kwargs)


root=next((p/'experiments/hrf_estimation' for p in [Path.cwd(),*Path.cwd().parents]
           if (p/'experiments/hrf_estimation/reports/ten_session_core_summary.tsv').exists()),None)
if root is None:
    raise FileNotFoundError('Run inside the boldtailor checkout containing the comparison reports.')
reports=root/'reports'
core=pd.read_csv(reports/'ten_session_core_summary.tsv',sep='\t')
overall=pd.read_csv(reports/'ten_session_core_overall.tsv',sep='\t')
strict=pd.read_csv(reports/'ten_session_core_converged.tsv',sep='\t')
reml=pd.read_csv(reports/'two_session_reml_comparison.tsv',sep='\t')
core_methods=['Canonical','Library CV','Parametric library CV','Original REML','AR(1) REML']
reml_methods=['Original REML','AR(1) REML','Separate components','Shared effects','Combined']
colors=dict(zip(dict.fromkeys(core_methods+reml_methods),plt.get_cmap('tab10').colors))
plt.rcParams.update({'figure.dpi':120,'axes.spines.top':False,'axes.spines.right':False})
for kind,frame in [('ten_session_comparison',core),('reml_session_comparison',reml)]:
    display_long_output({'application/json':{'kind':kind,'records':json.loads(frame.to_json(orient='records',double_precision=15))}},raw=True)

# %% [markdown]
# ## 1. Five methods across all ten sessions
#
# Every method has an all-run fit and two complementary 6/6 transfer fits in each session. Only the HRF is transferred; amplitudes are refitted independently on each held-out run. All models use binary trial type, separate missing/incorrect junk regressors, NSD motion confounds (24 parameters), and an 80-second cosine HPF without censoring. Full-library methods use 533 HRFs; parametric CV excludes the empirical library members.
#
# The primary comparison uses the **same 1,518 of 1,750 VTC vertices across all methods, both transfer directions, and all ten sessions**. The other 232 lack finite evaluable scores in at least one combination. Thus the comparison is within a common evaluable subset of VTC.
#
# SSE is pooled before calculating R². Total R² includes all regressors; partial task R² measures improvement over nuisance plus junk; incremental task R² divides that improvement by total signal variance. These are descriptive HRF-transfer scores, not fully frozen-model predictions.

# %%
table=overall.set_index('method').reindex(core_methods)
display(table[['n_common','n_sessions','r2_full','r2_task_partial','delta_r2_task']].style.format({'r2_full':'{:.5f}','r2_task_partial':'{:.5f}','delta_r2_task':'{:.5f}'}))
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for ax,metric,title in zip(axes,['r2_full','r2_task_partial'],['Pooled total held-out R²','Pooled partial task held-out R²']):
    ax.barh(core_methods,table[metric],color=[colors[m] for m in core_methods])
    ax.invert_yaxis();ax.set(xlabel=title,title='All ten sessions pooled')
    ax.set_xlim(0,float(table[metric].max())*1.12)
    for i,value in enumerate(table[metric]):ax.text(value+.002,i,f'{value:.4f}',va='center',fontsize=9)
plt.show()

# %% [markdown]
# The large improvement is from adapting the HRF rather than keeping it canonical. Pooled partial task R² is **0.23844 for full-library CV**, **0.23835 for parametric CV**, **0.23726 for original REML**, and **0.23648 for AR(1)**, compared with **0.17417 for canonical**. The full-library and parametric-CV results are nearly identical.
#
# These pooled values weight sessions and vertices according to their residual energy. The session-specific comparisons below show whether that overall ordering is consistent; the small differences among adapted-HRF methods are not presented as statistically established differences.

# %%
fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
for method in core_methods:
    g=core[core.method.eq(method)].sort_values('session')
    for ax,metric,title in zip(axes,['r2_full','r2_task_partial'],['Total held-out R²','Partial task held-out R²']):
        ax.plot(g.session,g[metric],marker='o',label=method,color=colors[method])
        ax.set(xlabel='Session',ylabel=title,xticks=list(range(10,20)))
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=3)
plt.show()
display(core.pivot(index='session',columns='method',values='r2_task_partial').reindex(columns=core_methods).round(5))

# %%
reference=core.query("method == 'Library CV'").set_index('session').r2_task_partial
paired=core.copy()
paired['difference_pp']=100*(paired.r2_task_partial-paired.session.map(reference))
fig,ax=plt.subplots(figsize=(11,4.3),constrained_layout=True)
for method in core_methods[1:]:
    g=paired[paired.method.eq(method)].sort_values('session')
    ax.plot(g.session,g.difference_pp,marker='o',label=method,color=colors[method])
ax.axhline(0,color='0.5',linewidth=1)
ax.set(xlabel='Session',ylabel='Partial task R² difference vs full-library CV (percentage points)',xticks=list(range(10,20)),title='Expanded scale: adapted-HRF methods')
ax.legend(fontsize=9)
plt.show()
display(paired.groupby('method').difference_pp.agg(['median','min','max']).reindex(core_methods).round(4))

# %% [markdown]
# ### Convergence sensitivity
#
# The primary table includes finite scores from fits marked nonconverged. Requiring `ok` status across every method, session and transfer direction leaves **1,467 vertices**. The comparison below applies that stricter shared mask to all five methods. Zero-variance fits have no selected task HRF and are excluded from the evaluable-mask comparison.
#
# AR(1) sessions 10–13 retain the earlier generic solver's recorded convergence flags; sessions 14–19 use the new scalar solver. Full session-11 validation matched all 1,750 all-run HRF selections between solvers, but earlier transfer fits have not been rerun.

# %%
sensitivity=core.merge(strict,on=['session','method'],suffixes=('','_strict'))
sensitivity['strict_minus_primary']=sensitivity.r2_task_partial_strict-sensitivity.r2_task_partial
display_long_output(sensitivity[['session','method','n_common','n_common_strict','nonconverged','variance_zero','strict_minus_primary']].round(6))

# %% [markdown]
# ### HRF shapes: all five ten-session methods
#
# This section includes **canonical, full-library CV, parametric-library CV, original REML, and AR(1)**. It uses the same **1,487 vertices** with valid, nonconstant HRFs in all five methods, all ten sessions, and all three fit types. This is slightly stricter than the performance mask because all-run HRF estimates must also exist.
#
# Each vertex's HRF is divided by its positive peak to compare shape rather than amplitude. Curves are sampled every 0.1 seconds from 0 to 35.9 seconds. Shape correlations are calculated between the **same vertex's curves**, then summarized across vertices—not calculated between ROI-average curves.
#
# The canonical HRF has perfect repeatability by construction. Its correlation of 1 is not evidence of superior estimation. Conversely, more repeatable adaptive shapes may reflect shrinkage rather than greater biological accuracy.

# %%
within_shape=pd.read_csv(reports/'core_hrf_within_session.tsv',sep='\t')
cross_shape=pd.read_csv(reports/'core_hrf_cross_session.tsv',sep='\t')
between_shape=pd.read_csv(reports/'core_hrf_between_models.tsv',sep='\t')
with np.load(reports/'core_hrf_shape_summary.npz') as archive:
    shape_methods=archive['methods'].tolist()
    shape_sessions=archive['sessions'].copy()
    shape_times=archive['times'].copy()
    mean_shapes=archive['session_mean_curves'].copy()
    shape_vertices=archive['feature_ids'].copy()
for kind,frame in [('core_shape_stability',within_shape),('cross_session_shapes',cross_shape)]:
    display_long_output({'application/json':{'kind':kind,'records':json.loads(frame.to_json(orient='records',double_precision=15))}},raw=True)
print(f'Shape comparisons use {len(shape_vertices):,} matched vertices.')
fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
for i,method in enumerate(shape_methods):
    axes[0].plot(shape_times,mean_shapes[i].mean(axis=0),label=method,color=colors[method])
axes[0].axhline(0,color='0.6',linewidth=.7)
axes[0].set(xlabel='Time after event (seconds)',ylabel='Mean peak-normalized HRF',title='All-run HRFs: mean over vertices and ten sessions',xlim=(0,36))
axes[0].legend(fontsize=8)
comparison=between_shape.pivot(index='method_a',columns='method_b',values='median_curve_r').reindex(index=shape_methods,columns=shape_methods)
im=axes[1].imshow(comparison,vmin=.6,vmax=1,cmap='viridis')
axes[1].set_xticks(range(5),shape_methods,rotation=45,ha='right',fontsize=8)
axes[1].set_yticks(range(5),shape_methods,fontsize=8)
axes[1].set_title('Between-model shape agreement')
for i in range(5):
    for j in range(5):axes[1].text(j,i,f'{comparison.iloc[i,j]:.3f}',ha='center',va='center',fontsize=9)
fig.colorbar(im,ax=axes[1],label='Median same-vertex, same-session curve correlation')
plt.show()

# %% [markdown]
# #### Within-session split-half agreement
#
# Each point is the median correlation across matched vertices between HRFs learned from complementary six-run training halves. This extends the earlier REML-only stability comparison to both CV models and canonical HRFs, for every session. These estimates use six training runs per HRF; the across-session analyses below use all twelve runs per session, so their absolute correlation levels are not directly comparable.

# %%
fig,ax=plt.subplots(figsize=(11,4.5),constrained_layout=True)
for method in core_methods:
    g=within_shape[within_shape.method.eq(method)].sort_values('session')
    ax.plot(g.session,g.median_curve_r,marker='o',color=colors[method],label=method)
ax.set(xlabel='Session',ylabel='Median split-half HRF curve correlation',xticks=list(range(10,20)),ylim=(min(.9,within_shape.median_curve_r.min()-.01),1.005))
ax.legend(fontsize=9)
plt.show()
display(within_shape.pivot(index='session',columns='method',values='median_curve_r').reindex(columns=core_methods).round(4))

# %% [markdown]
# #### HRF curves across all sessions
#
# Each panel shows one method, with a separate curve for each session. Curves are averages of the same vertices' peak-normalized **all-run** HRFs. A smooth mean can hide substantial vertex-level variability, and a small mean difference does not imply that individual vertices select the same HRF. All panels share the same axis limits.

# %%
fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,sharey=True,constrained_layout=True)
cmap=plt.get_cmap('viridis')
for i,method in enumerate(shape_methods):
    ax=axes.flat[i]
    for j,session in enumerate(shape_sessions):
        ax.plot(shape_times,mean_shapes[i,j],color=cmap(j/9),alpha=.9,linewidth=1.3)
    ax.axhline(0,color='0.65',linewidth=.6)
    ax.set(title=method,xlabel='Time after event (seconds)',ylabel='Mean peak-normalized HRF',xlim=(0,36))
axes.flat[5].set_visible(False)
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(10,19),cmap=cmap),ax=list(axes.flat[:5]),label='Session',ticks=list(range(10,20)))
plt.show()

# %% [markdown]
# #### Same-vertex shape agreement between sessions
#
# Each cell is the median correlation across the 1,487 matched vertices between HRFs fitted using all twelve runs of the two indicated sessions. This answers whether **the same cortical locations retain similar HRF shapes across sessions**. The diagonal is self-correlation and is excluded from the summary below. All heatmaps use the same 0.90–1.00 color scale.
#
# The summary reports the median of the 45 distinct session-pair medians, plus their minimum and maximum. These pairs share sessions and are not independent replications; no significance test is implied.

# %%
fig,axes=plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
for i,method in enumerate(core_methods):
    ax=axes.flat[i]
    matrix=cross_shape[cross_shape.method.eq(method)].pivot(index='session_a',columns='session_b',values='median_curve_r').reindex(index=range(10,20),columns=range(10,20))
    im=ax.imshow(matrix,vmin=.9,vmax=1,cmap='viridis')
    ax.set_xticks(range(10),range(10,20),fontsize=8);ax.set_yticks(range(10),range(10,20),fontsize=8)
    ax.set(title=method,xlabel='Session',ylabel='Session')
    for row in range(10):
        for col in range(10):
            value=matrix.iloc[row,col]
            ax.text(col,row,f'{value:.3f}',ha='center',va='center',fontsize=6,color='white' if value<.95 else 'black')
axes.flat[5].set_visible(False)
fig.colorbar(im,ax=list(axes.flat[:5]),label='Median same-vertex HRF curve correlation',shrink=.75)
plt.show()
off_diagonal=cross_shape.query('session_a < session_b')
display(off_diagonal.groupby('method').median_curve_r.agg(['median','min','max']).reindex(core_methods).round(4))

# %% [markdown]
# Among the adaptive methods, parametric-library CV and full-library CV have the highest median across-session agreement in this matched subset (about **0.976** and **0.975**), followed by original REML (**0.967**) and AR(1) (**0.957**). This descriptive ranking concerns HRF shape consistency, not predictive accuracy or the validity of the noise model. The earlier two-session REML-only comparison remains below and uses its original, larger per-session masks.
#
# These shape results are portable snapshots in `reports/core_hrf_*`. The provenance file records the hashes of all 150 source HRF fit files; no new model fitting was performed.

# %% [markdown]
# ### Empirical versus parametric HRF selection
#
# The **full library** contains the canonical HRF, 512 parametric double-gamma HRFs, and 20 empirical GLMsingle HRFs. The **parametric-only library** contains the same canonical and 512 parametric candidates; its empirical selection rate is necessarily zero. Here “parametric” includes the canonical candidate. Both families are library HRFs; the distinction is empirical versus parametric origin.
#
# This report uses **all-run CV selections (12 runs per session)** on the same **1,487 matched vertices** as the shape comparisons. Candidate identities were verified against each saved library fingerprint and fit hash. These are selection frequencies, not evidence that a family significantly outperforms another. Rates are not adjusted for the unequal number of candidates. The six-run transfer fits are not pooled with these selections.
#

# %%
selection_rates=pd.read_csv(reports/'empirical_selection_rates.tsv',sep='\t')
selection_pairs=pd.read_csv(reports/'empirical_selection_agreement.tsv',sep='\t')
selection_frequency=pd.read_csv(reports/'empirical_selection_frequency.tsv',sep='\t')
display_long_output(selection_rates.round(2))
display_long_output({'application/json':{'kind':'empirical_selection_rates','records':selection_rates.to_dict('records')}},raw=True)
display_long_output({'application/json':{'kind':'empirical_selection_agreement','records':selection_pairs.to_dict('records')}},raw=True)
fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
for method,g in selection_rates.groupby('method',sort=False):
    axes[0].plot(g.session,g.empirical_percent,marker='o',label=method)
axes[0].set(xlabel='Session',ylabel='Vertices selecting empirical HRFs (%)',xticks=range(10,20),ylim=(-1,45))
axes[0].legend()
counts=selection_frequency.n_empirical_sessions.value_counts().reindex(range(11),fill_value=0)
axes[1].bar(counts.index,counts.values)
axes[1].set(xlabel='Number of sessions selecting an empirical HRF',ylabel='Number of matched vertices',xticks=range(11))
plt.show()

# %% [markdown]
# #### Consistency of empirical selections across sessions
#
# All 45 session pairs compare the **same vertices**. Family agreement counts both empirical→empirical and parametric→parametric selections. **Cohen’s κ** adjusts this agreement for the two sessions’ marginal selection rates. **Empirical Jaccard overlap** is the number selecting empirical in both sessions divided by the number selecting empirical in either; it excludes parametric→parametric agreements.
#
# The conditional exact-ID measure asks: among vertices choosing empirical in both sessions, what fraction chose the **same one of the 20 empirical HRFs**? The table also compares exact candidate-ID agreement for the full and parametric-only methods. Exact-ID agreement is stricter than HRF shape correlation: nearby candidates can have very similar curves.
#

# %%
fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
for ax,metric,title in zip(axes,['family_agreement','empirical_jaccard','same_empirical_id_given_both'],
                          ['Same family','Empirical Jaccard overlap','Same empirical ID | both empirical']):
    matrix=np.eye(10)
    for row in selection_pairs.itertuples():
        matrix[row.session_a-10,row.session_b-10]=getattr(row,metric)
        matrix[row.session_b-10,row.session_a-10]=getattr(row,metric)
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis')
    ax.set(title=title,xlabel='Session',ylabel='Session',xticks=range(10),yticks=range(10))
    ax.set_xticklabels(range(10,20)); ax.set_yticklabels(range(10,20))
fig.colorbar(im,ax=axes,label='Fraction',shrink=.8)
plt.show()
metrics=['family_agreement','family_kappa','empirical_jaccard','same_empirical_id_given_both',
         'full_exact_id_agreement','parametric_exact_id_agreement']
display(selection_pairs[metrics].agg(['mean','median','min','max']).T.round(4))
display_long_output(selection_pairs.round(4))

# %% [markdown]
# Empirical selection varies substantially by session (**4.4%–36.4%**). Across the 45 distinct session pairs, mean family agreement is **73.4%**, but mean κ is only **0.143** and mean empirical Jaccard overlap is **16.8%**. Among vertices selecting empirical in both sessions, the mean pairwise rate of selecting the exact same empirical HRF is **24.3%**.
#
# Exact candidate agreement averages **5.5% for full-library CV** and **6.3% for parametric-only CV**. This low identity stability coexists with the high curve correlations above: candidate identity is much less stable than broad HRF shape. Of the 1,487 vertices, **430 never select empirical**, **1,057 select empirical at least once**, and **none select empirical in all ten sessions**. All averages here weight the 45 session pairs equally; they are descriptive summaries, not independent statistical replications.
#

# %% [markdown]
# ## 2. Original REML with versus without response-time modeling
#
# This additional comparison covers **sessions 10–19 and the full 1,750-vertex VTC input ROI**, using the same 533 HRF candidates and complementary 6/6 run splits as original REML. A fixed complete-case mask across both models, all sessions, and both transfer directions is used below; valid all-run curves are also required. Its size is printed below and can differ from the earlier five-method comparison.
#
# **With RT** is the existing original REML model: two binary-condition effects, a common RT-amplitude effect, and separate missing/incorrect junk effects, all treated as independent run-specific random effects with one shared variance. **Without RT** omits only the RT column, leaving four random effects per run. Both use iid residual noise, the original scalar REML solver, and the same fixed nuisance model (NSD motion24, 80-second cosine drift, intercept). Stimulus durations, trial inclusion, and junk categories remain unchanged. Response-time validity still helps identify missing responses; its numeric values are not predictors in the no-RT model.
#
# The existing with-RT fits are reused after verifying artifact hashes, input fingerprints, libraries, and splits. The no-RT fits include all-run estimates and both training halves. These are experimental HRF-selection comparisons within one participant.
#
# ### Predictive comparison and crossed evaluation
#
# The native comparison selects and evaluates each HRF using its own model. To separate HRF-selection effects from the additional RT predictor, both sets of HRFs are also evaluated under **both** scoring designs. Every scoring fit refits amplitudes using unwhitened OLS. Thus the direct RT-regressor gain is not fully held-out prediction: for a fixed HRF, adding RT cannot increase OLS residual SSE. Total and partial-task R² are both reported; HRF-dependent junk regressors mean the partial-R² denominator can differ across selected HRFs.

# %%
rt_dir=reports/'reml_rt_ablation'
rt_summary=pd.read_csv(rt_dir/'summary.tsv',sep='	')
rt_pooled=pd.read_csv(rt_dir/'pooled.tsv',sep='	')
rt_shape=pd.read_csv(rt_dir/'split_half_shapes.tsv',sep='	')
rt_between=pd.read_csv(rt_dir/'between_models.tsv',sep='	')
rt_cross=pd.read_csv(rt_dir/'cross_session_shapes.tsv',sep='	')
rt_labels={'amplitude':'With RT','none':'Without RT'}
rt_colors={'amplitude':'tab:blue','none':'tab:orange'}
print(f'RT ablation uses {rt_summary.n_common.iloc[0]:,} matched vertices in every session and evaluation.')
display(rt_pooled.round(6))
display_long_output({'application/json':{'kind':'rt_ablation_comparison','records':rt_summary.to_dict('records')}},raw=True)
display_long_output({'application/json':{'kind':'rt_ablation_shapes','records':rt_shape.to_dict('records')}},raw=True)
fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
for mode in ['amplitude','none']:
    g=rt_summary[(rt_summary.selection_rt==mode)&(rt_summary.evaluation_rt==mode)]
    for ax,metric,title in zip(axes,['r2_full','r2_task_partial'],['Total R²','Partial task R²']):
        ax.plot(g.session,g[metric],marker='o',label=rt_labels[mode],color=rt_colors[mode])
        ax.set(xlabel='Session',ylabel=title,xticks=range(10,20),title='Native selection and evaluation')
axes[0].legend();plt.show()
display_long_output(rt_summary.round(6))

# %%
fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
for scoring in ['amplitude','none']:
    data=rt_summary[rt_summary.evaluation_rt.eq(scoring)].pivot(index='session',columns='selection_rt',values='r2_task_partial')
    axes[0].plot(data.index,100*(data.amplitude-data['none']),marker='o',label='Evaluate '+rt_labels[scoring].lower())
for selected in ['amplitude','none']:
    data=rt_summary[rt_summary.selection_rt.eq(selected)].pivot(index='session',columns='evaluation_rt',values='r2_task_partial')
    axes[1].plot(data.index,100*(data.amplitude-data['none']),marker='o',label='Select '+rt_labels[selected].lower())
axes[0].set(title='Effect of RT during HRF selection',ylabel='With-RT minus no-RT selection (percentage points)')
axes[1].set(title='Effect of RT during amplitude refitting',ylabel='With-RT minus no-RT evaluation (percentage points)')
for ax in axes:
    ax.axhline(0,color='0.5',lw=.8);ax.set(xlabel='Session',xticks=range(10,20));ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# **Result:** On 1,746 matched vertices, pooled native total R² is **0.68532 with RT versus 0.68036 without RT**; partial task R² is **0.22846 versus 0.21647**. However, when both sets of HRFs are evaluated with RT, no-RT selection gives partial R² **0.22870**, compared with **0.22846** for RT selection. Under no-RT evaluation the corresponding values are **0.21647 versus 0.21555**. Thus the native advantage of including RT is primarily associated with the extra scoring regressor, rather than improved HRF selection. The selection differences are small and vary by session; these descriptive comparisons do not establish equivalence or statistical significance.

# %% [markdown]
# ### HRF shape and stability
#
# Split-half correlations compare the same vertex's HRFs selected from complementary six-run training sets. Between-model correlations compare the same vertex's all-run HRF with versus without RT. Curves are normalized to unit positive peak before averaging over vertices; correlations are calculated per vertex, not from the averaged curves.

# %%
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for mode in ['amplitude','none']:
    data=rt_shape[rt_shape.selection_rt.eq(mode)]
    axes[0].plot(data.session,data.median_curve_r,marker='o',label=rt_labels[mode],color=rt_colors[mode])
axes[1].plot(rt_between.session,rt_between.median_curve_r,marker='o',color='tab:green')
axes[1].fill_between(rt_between.session,rt_between.q25,rt_between.q75,color='tab:green',alpha=.15,label='Vertex interquartile range')
axes[0].set(title='Within-model split-half agreement',ylabel='Median HRF correlation')
axes[1].set(title='With-RT versus no-RT all-run HRFs',ylabel='Same-vertex HRF correlation')
for ax in axes:ax.set(xlabel='Session',xticks=range(10,20));ax.legend(fontsize=8)
plt.show()
with np.load(rt_dir/'mean_shapes.npz') as data:
    rt_curves=data['curves'];rt_times=data['times']
fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True,constrained_layout=True)
for i,mode in enumerate(['amplitude','none']):
    for j,session in enumerate(range(10,20)):
        axes[i].plot(rt_times,rt_curves[i,j],color=plt.get_cmap('viridis')(j/9),lw=1.2)
    axes[i].axhline(0,color='0.5',lw=.7)
    axes[i].set(title=rt_labels[mode],xlabel='Time after event (s)',ylabel='Mean peak-normalized HRF',xlim=(0,36))
fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(10,19),cmap='viridis'),ax=list(axes),label='Session',ticks=range(10,20))
plt.show()

# %%
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
rt_min=float(rt_cross.median_curve_r.min())
for ax,mode in zip(axes,['amplitude','none']):
    data=rt_cross[rt_cross.selection_rt.eq(mode)].pivot(index='session_a',columns='session_b',values='median_curve_r')
    im=ax.imshow(data,vmin=min(.9,rt_min),vmax=1,cmap='viridis')
    ax.set(title=rt_labels[mode],xlabel='Session',ylabel='Session',xticks=range(10),yticks=range(10))
    ax.set_xticklabels(range(10,20));ax.set_yticklabels(range(10,20))
fig.colorbar(im,ax=list(axes),label='Median same-vertex all-run HRF correlation')
plt.show()
display(rt_cross.query('session_a < session_b').groupby('selection_rt').median_curve_r.agg(['median','min','max']).round(4))

# %% [markdown]
# The two heatmaps use the same color scale and compare all 45 distinct session pairs. Each session contributes one all-run HRF per vertex. These pairwise comparisons overlap in sessions and are descriptive, not independent replications. Detailed artifacts and source hashes are under `reports/reml_rt_ablation`.
#
# Omitting RT slightly increases the mean of the ten session-specific median split-half correlations (**0.9499 versus 0.9458**). Across the 45 session pairs, the median of the pair-specific median correlations is **0.9615 without RT versus 0.9579 with RT**. The two models nevertheless select similar shapes: within-session median all-run between-model correlations range from **0.9851 to 1.0000**.

# %% [markdown]
# ## 3. REML variants in sessions 10 and 11
#
# All five variants are complete in both sessions:
#
# - **Original REML:** one variance for independent task/junk effects.
# - **AR(1):** adds training-only pooled autocorrelation correction and whitening.
# - **Separate components:** distinct condition, RT and junk variances.
# - **Shared effects:** adds task effects shared across training runs.
# - **Combined:** AR(1), separate components and shared condition/RT effects.
#
# Here the matched set is defined within each session across these five variants: **1,716 vertices in session 10 and 1,728 in session 11**. Scores therefore differ slightly from Section 1's smaller fixed ten-session mask; compare methods within each section. Evaluation uses the same unwhitened OLS scoring model for all variants.
#
# The complex variants remain discontinued for future analyses; this section summarizes their completed sensitivity results only.

# %%
display(reml[['session','method','n_common','r2_full','r2_task_partial','delta_r2_task']].sort_values(['session','method']).round(6))
fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
for ax,session in zip(axes,[10,11]):
    g=reml[reml.session.eq(session)].set_index('method').reindex(reml_methods)
    change=100*(g.r2_task_partial-g.loc['Original REML','r2_task_partial'])
    ax.barh(reml_methods,change,color=[colors[m] for m in reml_methods])
    ax.axvline(0,color='0.5',linewidth=1);ax.invert_yaxis()
    ax.set(title=f'Session {session}',xlabel='Partial task R² difference vs original REML (percentage points)')
plt.show()

# %% [markdown]
# ### HRF shape agreement
#
# The next plot shows median Pearson correlation between each vertex’s HRFs estimated from the two independent training halves. AR(1) and combined improve descriptive shape agreement in both sessions. Increased stability can result from shrinkage and does not prove greater recovery accuracy.

# %%
fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
for i,session in enumerate([10,11]):
    g=reml[reml.session.eq(session)].set_index('method').reindex(reml_methods)
    ax.plot(range(len(reml_methods)),g.median_curve_r,marker='o',label=f'Session {session}')
ax.set(xticks=range(len(reml_methods)),xticklabels=reml_methods,ylabel='Median split-half HRF curve correlation',ylim=(.94,1.0))
ax.legend();plt.show()
display(reml[['session','method','median_curve_r','same_hrf']].round(4))

# %% [markdown]
# The combined model has the largest predictive gain in session 11 (**+0.120 percentage points** in partial task R² over original REML), but almost no gain in session 10. Separate components and shared effects alone add little. Combined has the highest HRF shape agreement, with AR(1) providing much of that improvement at lower model complexity.
#
# These two sessions provide a sensitivity check, not evidence for a broadly superior complex model. Together, the two sections show that the larger canonical-to-adapted-HRF gain is much clearer than differences among the adapted approaches.
#
# ### Provenance
#
# All new baseline completion hashes, row counts, input fingerprints and splits were verified against the corresponding AR(1) analyses. The notebook uses portable reports under `reports/ten_session_core_*` and `reports/two_session_reml_comparison.tsv`. Requested-method vertex-level scores are in `reports/focused_comparison_scores.tsv.gz`; source paths and fit artifact hashes are in `reports/focused_comparison_provenance.json`. No significance test or independent-participant generalization is implied by these within-participant comparisons.
