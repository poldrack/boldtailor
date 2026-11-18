# User guide

[README](../README.md) · [Practical guide](practical-guide.md) · [API reference](api.md) · [BIDS workflow examples (NSD data)](../examples/bids_workflow/README.md)

For a quick introduction to each component, start with the
[practical guide](practical-guide.md). This guide covers the statistical
methods, modeling assumptions, and command-line workflow options.

## Preparing your data

The array API takes one signal matrix per run, shaped **time points × features**.
Features must have the same order in every run. Keep the mask or CIFTI
brain-model axis used to extract the signals so you can map results back to
image space.

Each run requires an event DataFrame with `onset` and `duration` in seconds.
Include `trial_type` for condition models, a `modulation` column if using
conventional Nilearn event modulation, or trial covariates (such as
`response_time` and `trial_type`) for `TaskModel` and single-trial analyses.

Pass a single array and DataFrame for one run, or matching lists for multiple
runs:

```python
from boldtailor.data import from_arrays

data = from_arrays(
    signals_by_run,
    events_by_run,
    frame_times=frame_times_by_run,
    confounds=confounds_by_run,
)
```

Supply either `tr=2.0` (for uniform sampling starting at 0 s) or strictly
increasing `frame_times` per run. Signals and confounds must be finite, with one
confound row per scan. Preprocessing, smoothing, scaling, and spatial alignment
happen before calling Boldtailor.

Confound handling depends on the fitting function:

| Model | Nuisance regressors included |
| --- | --- |
| `fit()` | Columns named in `ModelSpec.confounds`, requested drift terms, and a run intercept |
| `fit_prepared()` | Exactly the columns in your design matrix |
| Single-trial and HRF-selection APIs | Every supplied confound column plus a run intercept (supply high-pass columns in `confounds`) |

When passing fMRIPrep cosine columns in `ModelSpec.confounds`, set
`drift_model=None` to avoid adding a second set of cosine drifts.

## Condition effects and contrasts

Use `fit()` when each condition has a shared response amplitude within a run.
Continuing the [README example](../README.md#a-first-model):

```python
from boldtailor.design import compile_designs, compile_nuisance_designs

designs = compile_designs(data, model)
first_run_design = designs[0].matrix
confound_designs = compile_nuisance_designs(data, model)

effect = result.effect("face_gt_house")
variance = result.variance("face_gt_house")
t_statistic = result.stat("face_gt_house")
z_score = result.z_score("face_gt_house")
p_value = result.one_sided_p_value("face_gt_house")
```

Contrasts can be string expressions (`"face - house"`) or weight mappings
(`{"face": 1, "house": -1}`). Because they reference column names, differing
column orders across runs do not affect the contrast. Only t contrasts are
supported, and every run must estimably support each contrast.

`ModelSpec` defaults to a Glover HRF (`hrf_model="glover"`), cosine drifts with
a 0.01 Hz cutoff, and AR(1) noise (`noise_model="ar1"`). Set `hrf_model="spm"`
for the canonical SPM HRF or `noise_model="ols"` for ordinary least squares.
Nilearn derivative bases and custom callable kernels are also supported.

Across multiple runs, each run is fitted independently and contrasts are
combined as equal-weight fixed effects (unweighted across run lengths and noise
levels, with summed degrees of freedom). P-values are one-sided and uncorrected.
`result.run_r2` stores per-run $R^2$, while `result.r2` pools residual and total
sums of squares across runs. If a feature's signal is constant in any run, its
contrast statistics and that run's `run_r2` are NaN; pooled `r2` is NaN only if
the feature is constant in every run.

Rank-deficient designs are fitted in an orthonormal basis of their column space
with `n - rank` residual degrees of freedom, and emit a warning naming the run
and rank. Rank and estimability checks normalize nonzero columns to unit norm so
arbitrary column scaling does not alter the detected rank.

## Using your own design matrix

Use `PreparedDesignAnalysis` when your design matrices are already built:

```python
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared

prepared = PreparedDesignAnalysis.from_arrays(
    signals,
    design,
    tr=2.0,
    column_roles={
        "stimulus": "task",
        "motion": "nuisance",
        "constant": "intercept",
    },
)
prepared_result = fit_prepared(
    prepared,
    contrasts={"stimulus": {"stimulus": 1.0}},
    noise_model="ols",
)
```

Every column requires a role: `task`, `nuisance`, `intercept`, or `other`.
Boldtailor fits the supplied columns without adding or transforming regressors.
Computing task-versus-nuisance $\Delta R^2$ requires a complete partition with
no `other` columns.

## Measuring task-related variance

Pooled $R^2$ is computed from sums of squares across runs:

```text
R² = 1 - sum_run(residual sum of squares)
         / sum_run(sum_time((signal - run mean)²))
```

Boldtailor distinguishes three $R^2$ comparisons:

| Question | Quantity |
| --- | --- |
| How much variance do task regressors add beyond confounds? | Full-model $R^2$ minus nuisance-only $R^2$ (`delta_r2`) |
| Does a selected HRF improve the fit over canonical SPM? | Selected-HRF full $R^2$ minus canonical full $R^2$ |
| Does a task model predict a held-out run? | Out-of-fold or split-test prediction $R^2$ after nuisance adjustment |

For conventional and prepared GLMs, `task_delta_r2(data, model, result)` and
`task_delta_r2_prepared(prepared, prepared_result)` refit full and nuisance-only
models with OLS (even when contrast inference used AR(1)). They return
`full_r2`, `nuisance_r2`, `raw_delta_r2`, and `delta_r2` (which clips tiny
negative floating-point roundoff to zero). Both functions require complete
`RunSources` metadata to verify that `result` matches the supplied inputs and
model; see [source records](api.md#source-records-and-saving). Common-HRF and
prepared comparisons require finite pooled $R^2$ at all supplied features, while
selected-HRF `task_delta_r2` preserves NaNs at undefined or constant features.

Single-trial results directly expose `full_r2`, `nuisance_r2`, and `delta_r2`
from the fitted OLS or ridge estimator (which can be negative under shrinkage).

## Estimating a beta for every trial

`fit_single_trials(data, ridge_alpha=0.0)` fits one regressor per event row,
preserving event order, durations, and sub-TR onsets. By default it uses the
canonical SPM HRF (`hrf_model="spm"`); pass an `HrfCandidate` to use a specific
library kernel across all features.

```python
from boldtailor.single_trial import fit_single_trials

ols = fit_single_trials(data)
ridge = fit_single_trials(data, ridge_alpha=0.1)
first_run_betas = ridge.run_betas[0]
trials = ridge.trial_table
```

Both `fit_single_trials` and `fit_selected_hrfs` return a `SingleTrialResult`:
- `run_betas`: tuple of `(trials, features)` arrays, one per run.
- `trial_table`: DataFrame with `trial_index`, `trial_id`, `run_label`,
  `event_index`, original event columns, and `run_index`.
- `design`: `SharedTrialDesign` (`result.design.matrices`) for a fixed HRF, or
  `SelectedTrialDesign` (`result.design.rebuild_matrix(run_index, hrf_id)`) for
  location-specific HRFs.

With `ridge_alpha > 0`, trial columns are projected off the run's nuisance space
(all supplied confounds plus an added intercept), scaled to unit L2 norm,
penalized by `ridge_alpha`, and rescaled back to native signal units. Nuisance
coefficients are unpenalized, and single-trial fits do not use AR(1) whitening.
Designs with unidentifiable trial columns or nonpositive residual degrees of
freedom are rejected.

## Fractional ridge at each grayordinate

Fractional ridge specifies the L2 norm ratio of nuisance-projected trial
coefficients relative to OLS (`1.0` gives OLS; smaller fractions apply stronger
shrinkage). Unlike `ridge_alpha`, fractional ridge operates on raw
nuisance-projected trial columns without column-length normalization or post-fit
affine rescaling.

```python
from boldtailor.fractional_ridge import (
    score_fraction_candidates, select_ridge_fractions,
)
from boldtailor.single_trial import fit_single_trials

# data requires at least two runs; predictors must be numeric.
predictors = [e[["trial_type", "response_time"]] for e in data.events]
scores = score_fraction_candidates(
    data, predictors, fractions=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
)
choice = select_ridge_fractions(scores)
betas = fit_single_trials(data, ridge_fraction=choice.ridge_fraction)
```

Each feature independently selects the candidate fraction that maximizes pooled
held-out trial-encoding $R^2$ against fixed OLS validation betas. Ties within
`1e-12` favor the largest fraction (least shrinkage). `choice.at_boundary` flags
winners at either end of the grid; a winner at the smallest fraction indicates
that stronger shrinkage may be needed, whereas `1.0` indicates no shrinkage was
preferred. Features with nonfinite candidate scores or zero OLS trial-beta norm
receive NaN.

The chosen fraction is shared across runs while the implied ridge penalty is
solved per run and feature (`result.run_ridge_alphas`; `result.ridge_alpha` is
`None`). Pass `library=library` to `score_fraction_candidates` to re-select HRFs
inside each inner training fold (requiring at least three runs with CV HRF
selection, or two with REML). Validation targets remain OLS betas under that
fold's training-selected HRFs.

## Choosing a shared ridge penalty by trial-level prediction

`score_ridge_candidates` and `select_ridge_penalty` select a single normalized
`ridge_alpha` across all features using a spatial percentile (90th percentile
by default):

```python
from boldtailor.ridge_selection import score_ridge_candidates, select_ridge_penalty
from boldtailor.single_trial import fit_single_trials

predictors = [e[["trial_type", "response_time"]] for e in data.events]
scores = score_ridge_candidates(
    data, predictors, alphas=[0., .001, .01, .1, 1., 10., 100.]
)
choice = select_ridge_penalty(scores, percentile=90)
betas = fit_single_trials(data, ridge_alpha=choice.ridge_alpha)
```

In this shared-alpha workflow, both training and validation betas use the
candidate `ridge_alpha` (rather than fixed OLS validation targets). Ties within
`1e-12` favor the smallest alpha. When scoring across parallel feature blocks,
concatenate the block score maps before calling `select_ridge_penalty`.

### Trial-encoding model details

Both ridge scorers use `boldtailor.trial_encoding` (`encoding_mode="within_run"`
by default):
- Predictors and betas are centered within each training run over complete
  predictor rows to fit shared OLS slopes.
- Validation residuals are centered within each test run and feature for scoring
  only (`1 - sum(SSE) / sum(within-run SST)`). Validation betas never influence
  fitted slopes or predictions.
- Nonfinite predictor rows (and nonpositive RTs when built with
  `trial_predictors`) are omitted from encoding while their events remain in the
  BOLD beta-series design.
- Training predictors must have full rank after within-run centering, at least
  one complete row per training run, and positive residual degrees of freedom.
  Each test run needs at least two complete trials.

Because tuning optimizes predictability from the supplied covariates, evaluate
generalization on held-out outer runs not used during HRF or ridge selection.

## Selecting an HRF for each location

`select_hrfs` chooses a candidate HRF per feature by leave-one-run-out
cross-validation (`method="cv"`, default, requiring at least two runs) or by
restricted maximum likelihood (`method="reml"`, requiring at least one estimable
run; see [Selecting HRFs with REML](practical-guide.md#11-select-hrfs-with-reml)):

```python
from boldtailor.hrf_selection import select_hrfs
from boldtailor.single_trial import fit_selected_hrfs

selection = select_hrfs(multi_run_data)  # uses default_hrf_library()
optimized = fit_selected_hrfs(multi_run_data, hrf_selection=selection)
optimized_ridge = fit_selected_hrfs(
    multi_run_data, hrf_selection=selection, ridge_alpha=0.1
)
```

### HRF libraries and parameter bounds

`default_hrf_library()` (533 candidates, cached) combines:
1. Canonical SPM at ID 0 (`CANONICAL_PARAMETERS = (6, 16, 1, 1, 6, 0, 32)`).
2. 512 double-gamma candidates from `timing_hrf_library(n_samples=512, seed=0)`.
3. The 20 empirical HRFs from GLMsingle (`glmsingle_hrf_curves()`, kind
   `"glmsingle"`).

Other library constructors in `boldtailor.hrf_library`:
- `timing_hrf_library()`: Samples realized waveform timing via scrambled Sobol
  sequences over `TIMING_BOUNDS` (peak time 2.5–8.5 s, response FWHM 2.0–6.5 s,
  trough time 8.0–19.0 s, undershoot FWHM 4.0–10.0 s, and log-uniform trough
  depth 0.01–0.4 of peak) and converts realizable draws to SPM double-gamma
  parameters with `spm_parameters_from_realized()`. This avoids the
  onset-versus-delay waveform redundancy of raw parameter sampling. Rejected
  infeasible draws are counted in `library.origin["rejected"]`.
- `sobol_hrf_library()`: Samples raw double-gamma parameters directly over
  response delay 3–6 s, undershoot delay 10–16 s, response dispersion 0.5–1.5,
  undershoot dispersion 0.5–2.5, response/undershoot ratio 2–8, and onset delay
  0–2 s (36 s duration; implied peak times ~1.5–7.5 s).
- `expanded_hrf_library()`: The 649-candidate discrete parameter grid over the
  same ranges.
- `glmsingle_hrf_library()`: Canonical SPM plus the 20 empirical GLMsingle
  curves alone.
- `HrfLibrary.from_parameters()` and `HrfLibrary.from_table()`: Build custom
  libraries or reconstruct a library from its saved `parameter_table`.

`selection.parameter_bound_flags` (shape `(n_features, 6, 2)`) marks double-gamma
picks whose parameters fall within 2% of the library's low or high box edge.
`selection.at_parameter_bound` is `True` when any informative parameter (at
least three distinct library values and nonzero width,
`library.informative_parameters`) is flagged, and `selection.parameter_bound_table()`
summarizes edge rates across custom picks.

### Task models, normalization, and split evaluation

Under `method="cv"`, each candidate HRF convolves the `TaskModel` regressors,
projects them and the signals off each run's confounds, fits shared task
amplitudes on the training runs, and predicts the held-out run. Errors are
pooled across folds and divided by the pooled confound-adjusted signal energy.

By default, `TaskModel()` has a single `task` regressor (mean stimulus
response). Pass `task_model=` to include trial modulators during selection:

```python
from boldtailor.model import Modulator, TaskModel

task_model = TaskModel((
    Modulator("response_time", missing="indicator"),
    Modulator("trial_type", kind="categorical", levels=("face", "house")),
))
selection = select_hrfs(multi_run_data, library=library, task_model=task_model)
```

With `missing="indicator"`, missing trials receive zero modulation and runs with
missing values include a `missing_<column>` regressor whose amplitude is
profiled per run.

For all built-in SPM, Glover, and library HRFs, each event's predicted response
on the oversampled grid is scaled to a peak of one (and library kernels are
stored at unit peak). Betas therefore measure peak BOLD response in signal units
independently of TR, oversampling, event duration, or selected HRF. Because
longer event durations do not increase regressor peak height under unit-peak
scaling, include duration as a modulator if you want duration-scaled amplitudes.
Keep at least one HRF duration of scans after the final event so responses are
not truncated at the end of a run.

To evaluate selected HRFs on held-out runs, use `evaluate_hrf_split`:

```python
from boldtailor.hrf_selection import evaluate_hrf_split

evaluation = evaluate_hrf_split(
    multi_run_data,
    library=library,
    train_runs=[0, 2],
    test_runs=[1, 3],
)
held_out_r2 = evaluation.test_r2
improvement_over_spm = evaluation.delta_test_r2
```

`evaluate_hrf_split` selects HRFs and estimates task amplitudes on `train_runs`
only, then scores predictions on nuisance-adjusted `test_runs` signals.

## Voxelwise HRFs in conventional GLMs

Pass `hrf_selection=selection` to `fit()` to use location-specific HRFs in a
conventional GLM:

```python
from boldtailor.fit import fit, task_delta_r2

result = fit(data, model, hrf_selection=selection)
effects = result.effect("face_gt_house")
z_scores = result.z_score("face_gt_house")
r_squared = result.r2

comparison = task_delta_r2(data, model, result)
task_added_r2 = comparison.delta_r2
```

`hrf_selection` replaces `ModelSpec.hrf_model`. When `ModelSpec.task_model` is
set, the selection's `task_model` must equal it or be a subset with identical
modulator settings (`TaskModel.is_subset_of`), and `oversampling` and
`min_onset` must match. Without `ModelSpec.task_model`, the selection must have
used the default `TaskModel()`.

`fit(..., hrf_selection=selection)` returns an `HrfAnalysisResult`, which
exposes the standard contrast methods, `run_r2`, `r2`, `hrf_indices`,
`hrf_selection`, `selection_provenance`, `group_design_provenance`,
`rebuild_group_design(run_index, hrf_id)`, and `materialize_group_designs()`.
Pass `feature_signature` to both selection and fitting when applying a selection
across datasets to verify spatial alignment. Locations with undefined HRFs
(`-1`) produce NaN contrast statistics and $R^2$. Contrast p-values treat the
selected HRFs as fixed.

## Comparing HRFs between sets of runs

Call `evaluate_hrf_split()` twice with swapped `train_runs` and `test_runs` to
compare `training_selection` across disjoint run splits, or use
`curve_correlations` and `compare_hrfs` in `boldtailor.reliability`. Because
candidate IDs are arbitrary labels, compare HRF curves or timing parameters
rather than correlating integer IDs.

## Task-guided denoising

`select_denoising()` is an opt-in stage modeled on GLMsingle's GLMdenoise
procedure. It defines a noise pool of features with low canonical ON-OFF $R^2$,
extracts per-run temporal principal components (PCs) from that pool, selects a
PC count by leave-one-run-out task prediction (`pcstop`) followed by an F-test
significance gate, and appends the chosen PCs to each run's confounds via
`with_denoising()`.

**Attribution.** The denoising procedure follows GLMsingle's GLMdenoise
stage (Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M.J.,
Kay, K.N. (2022). Improving the accuracy of single-trial fMRI response
estimates using GLMsingle. *eLife*, 11, e77599.
[doi:10.7554/eLife.77599](https://doi.org/10.7554/eLife.77599)), with the
deviations listed in this section. It is an independent reimplementation;
GLMsingle (Copyright (c) 2021, Kendrick Kay) is distributed under the BSD
3-Clause License, reproduced in `src/boldtailor/_resources/GLMsingle-LICENSE.txt`.
GLMsingle's authors have not reviewed or endorsed Boldtailor.

```python
from boldtailor.denoising import select_denoising, with_denoising
from boldtailor.fit import fit
from boldtailor.hrf_selection import select_hrfs

denoising = select_denoising(data, task_model=task_model, library=library)
augmented = with_denoising(data, denoising)
selection = select_hrfs(augmented, library=library, task_model=task_model)
fitted_model = model.replace(confounds=tuple(augmented.confounds[0].columns))
result = fit(augmented, fitted_model, hrf_selection=selection)
```

**Requirements.** `data` needs at least three runs. Every input feature is a
candidate (restrict to brain voxels or cortical grayordinates before building
`AnalysisData`). `library=None` uses `default_hrf_library()`. If you pass
`feature_signature`, pass the same signature to `with_denoising()`.

**Procedure.**
1. **Freeze initial HRFs**: Select HRFs once on all runs with baseline confounds
   (`initial_selection`).
2. **Compute ON-OFF $R^2$**: Fit one shared canonical-HRF task regressor (all
   events with unit amplitude) plus per-run baseline confounds and intercepts
   in-sample across all runs (`onoff_r2`).
3. **Define noise pool and scoring set**: With `pool_r2_threshold="auto"` (or a
   fixed float in $[0, 1]$), fit a two-component Gaussian mixture
   (`findtailthreshold`) to finite ON-OFF $R^2$ values. Features below the
   threshold form the noise pool; features above it with a defined HRF form the
   scoring set (falling back to the top 100 features with a defined HRF if none
   exceed the threshold, flagged by `scoring_fallback`).
4. **Extract run PCs**: In each run, project noise-pool time series off baseline
   confounds and the intercept, drop numerically zero columns, normalize
   remaining columns to unit L2 norm, and compute temporal PCs by SVD (for
   candidate counts `0..10` by default).
5. **Cross-validate candidate counts**: For each held-out run and candidate PC
   count, fit shared task amplitudes on the training runs (with per-run baseline
   confounds, missing-value indicators, and candidate PCs) using the frozen
   HRFs, and predict the held-out run.
6. **Score against a fixed target**: Project the held-out run's signal off its
   baseline confounds (and any missing-value indicators) only—never off
   candidate PCs—pool SSE and SST across folds for each scored feature, and take
   the median $R^2$ across scored features (`perf`).
7. **Apply `pcstop`**: With `curve = perf - perf[0]`, walk available counts in
   increasing order, tracking the running maximum `best`, stop at the first
   count where `best * pcstop >= max(curve)` (`pcstop=1.05` by default), and
   record the count achieving `best` as `pcstop_count`.
8. **Significance gate** (`significance_gate=True`, `gate_noise_model="ar1"` by
   default): If `pcstop_count > 0`, compare reduced (no PCs) and full (adding
   each run's first `pcstop_count` PCs) in-sample nested models across all runs
   at each scored feature. Under `"ar1"`, each run and feature is prewhitened
   using Nilearn's Yule-Walker lag-1 estimate from the full model's OLS
   residuals (binned to 0.01 steps via `ARModel`). If $m$ of $n$ testable
   features have $F$-test $p < \text{gate\_alpha}$ (0.05), keep `pcstop_count`
   only when the one-sided binomial test of $m/n$ against `gate_alpha` has
   $p < \text{gate\_binomial\_alpha}$ (0.05); otherwise set `n_components = 0`.
   Set `significance_gate=False` for GLMsingle's `pcstop`-only rule.

**Deviations from GLMsingle and caveats.**
- Run nuisance uses the analysis's supplied confounds plus an intercept instead
  of polynomial drifts, and count scoring uses held-out task time-series
  prediction against a PC-free target instead of repeated-condition beta
  consistency.
- No mean-intensity `bright` filter is applied, ON-OFF $R^2$ is in $[0, 1]$
  rather than percent, `robustrange` uses all finite values, and a mixture fit
  whose tail posterior never crosses 0.5 raises `ValueError`. A mixture fit that
  does not converge emits a `RuntimeWarning` and records `converged=False` in
  `noise_pool_mixture`.
- The significance gate prevents `pcstop` from selecting positive PC counts on
  pure noise (where relative gains of $\sim 10^{-4}$ can trigger `pcstop`), and
  AR(1) prewhitening prevents unwhitened OLS F-tests from firing on independent
  autocorrelated noise. However, the gate tests in-sample variance explained by
  the PCs rather than out-of-sample task-prediction gain, single-lag AR(1) can
  under-whiten higher-order autocorrelation, and the binomial test assumes
  independence across scored features.

**Reading `DenoisingResult`.**
- `n_components` (count after the gate), `pcstop_count` (count before the gate),
  and `run_components` (tuple of `(scans, n_components)` arrays).
- `significance_gate`: `SignificanceGate` holding settings, `decision`
  (`"kept"`, `"rejected"`, `"skipped_zero_count"`, `"disabled"`), per-feature
  `f_statistic`, `p_value`, `df1`, `df2`, `ar_coefficients`, `tested` and
  `excluded` masks, `exclusions`, `m`, `n`, and `binomial_p`.
- `pool_masks` (`PoolMasks`) and convenience properties `noise_pool`,
  `scoring_mask`, `noise_pool_threshold`, `noise_pool_mixture`, and
  `scoring_fallback`, plus `onoff_r2` and `scored`.
- `initial_selection`, `selection_cv_r2`, and `initial_hrf_indices`.
- `pca_diagnostics` (`PcaDiagnostics`): call `pca_diagnostics.table()` for
  per-run pool size, retained columns, rank, and tolerance, and inspect
  `pca_diagnostics.singular_values`.
- `candidate_scores`, `perf`, `curve`, `fold_scores`, `folds`, `component_names`,
  `source_identity`, and `provenance`.

`with_denoising()` verifies that `data` matches the exact runs, timing, signals,
baseline confounds, and events used in `select_denoising()`, rejects duplicate
application or column-name collisions, appends `denoise_pc_000`, ..., and
records a `denoising_augmentation` annotation on each run's confound
`SourceRef`. For independent evaluation of denoising, run `select_denoising()`
on outer-training runs only and freeze the pool mask and PC count before
evaluating outer test runs.

## Reaction-time checks

`correlate_rt` in `boldtailor.diagnostics` computes within-run centered Pearson
correlations between trial betas and positive RTs for each run and pooled across
`all`, `odd`, and `even` runs (requiring at least 3 valid trials and at least 2
pooled residual degrees of freedom after within-run centering). Missing or
nonpositive RTs are excluded from the correlation check without removing trials
from the beta-series fit. When RT was used to tune ridge regularization on the
same runs, RT correlations are descriptive rather than an independent test.

## Saving results and analysis records

Every result object carries a `ProvenanceRecord`:

```python
record = result.provenance.to_dict()
record_json = result.provenance.canonical_json()
```

Pass `RunSources` (containing `SourceRef` descriptors for `signal`, `events`,
and optional `confounds`) to `from_arrays` or `PreparedDesignAnalysis.from_arrays`
to establish a `data_fingerprint` and `analysis_fingerprint`. Supplying `sha256`
on each `SourceRef` makes the fingerprint content-based. Use
`project_bids_provenance()` and `publish_artifact_set()` to write results and
BIDS derivative metadata with lock-based overwrite protection and rollback; see
[source records and saving](api.md#source-records-and-saving).

## Editing returned arrays

Returned NumPy arrays are marked read-only (`writeable=False`). Copy an array
before modifying it in place:

```python
editable_betas = result.run_betas[0].copy()
editable_betas[:, 0] = 0
```

## Running the full workflow

`boldtailor run` runs the BIDS + fMRIPrep fsLR-91k CIFTI workflow for one or
more sessions of a subject and task, publishing BIDS derivatives and a
self-contained HTML report:

```bash
uv run boldtailor run --bids-dir /data/nsd --subject sub-01 --session ses-nsd01 --task nsdcore
```

Pass multiple session labels after `--session` to run each session in sorted
order after preflighting all listed sessions:

```bash
uv run boldtailor run --bids-dir /data/nsd --subject sub-01 --session ses-nsd01 ses-nsd02 --task nsdcore
```

Add `--dry-run` to print the resolved plan and any input/output problems as JSON
without fitting. The same workflow is callable from Python via `WorkflowSettings`
(`boldtailor.workflow.settings`) and `run_workflow` (`boldtailor.workflow.run`).

### Flags

| Group | Flag | Meaning (default) |
| --- | --- | --- |
| inputs | `--bids-dir`, `--subject`, `--session`, `--task` | Required. Subject (`sub-01`), one or more sessions (`ses-nsd01`), and task label |
| inputs | `--fmriprep-dir` | fMRIPrep directory (auto-detected under `<bids-dir>/derivatives/fmriprep*`) |
| inputs | `--output-dir` | Derivative root (defaults to `<bids-dir>/derivatives/boldtailor_hrf-<library>_ridge-<mode>[_method-reml_...]`) |
| inputs | `--space` | `fsLR-91k` |
| inputs | `--modulator COLUMN[:OPTION[,OPTION...]]` | Explicit modulator with options `indicator`, `categorical`, or `reference=LEVEL` (repeatable; overrides auto-detection) |
| inputs | `--no-modulators` | Fit the `task` regressor alone |
| HRF selection | `--hrf-method` | `cv` or `reml` (`cv`) |
| HRF selection | `--reml-rho`, `--reml-reference-hrf-id` | Scalar AR(1) correlation (`0.0`) and fixed-mean reference HRF ID (`0`) when `--hrf-method reml` |
| HRF selection | `--hrf-library` | `default`, `sobol`, `expanded`, or `canonical` (`default`) |
| HRF selection | `--hrf-n-samples`, `--hrf-seed` | Sample count and seed for `default` or `sobol` (`512`, `0`) |
| HRF selection | `--no-rt-in-hrf-selection` | Exclude `response_time` from HRF selection while keeping other modulators |
| beta series | `--ridge-mode` | `fractional_cv`, `cv`, `fixed`, or `off` (`fractional_cv`) |
| beta series | `--ridge-alpha` | Penalty for `--ridge-mode fixed` (`0.1`) |
| beta series | `--ridge-fractions`, `--ridge-alphas` | Candidate grids for `fractional_cv` and `cv` |
| beta series | `--ridge-percentile` | Spatial percentile for shared-alpha `cv` (`90`) |
| beta series | `--encoding-mode` | `within_run` or `absolute` (`within_run`) |
| stages and figures | `--skip-stage` | `reliability`, `betas`, or `summaries` (repeatable) |
| stages and figures | `--no-surface-maps`, `--surface-mesh left=PATH` / `right=PATH` | Skip cortical surface figures or provide explicit fsLR 32k meshes |
| execution | `--n-jobs`, `--block-size`, `--max-grayordinates` | Worker count (`4`), block size (`4096`), and optional grayordinate cap |
| execution | `--existing-results` | `error` or `overwrite` (`error`) |
| execution | `--dry-run`, `--quiet` | Print JSON plan and exit; log warnings only |

### Modulators

By default, `boldtailor run` inspects the event tables across runs:
- If `response_time` is present in every run, it adds
  `Modulator("response_time", missing="indicator")`.
- If `trial_type` has at least two distinct non-missing levels across runs, it
  adds a reference-coded categorical modulator. Levels are canonicalized as text
  (`1`, `"1"`, and `1.0` map to `"1"`), sorted numerically or lexically, and use
  the first level as reference (`task`), adding `<column>[<level>]` indicator
  regressors for the remaining levels. Every level must occur in every run.

Passing `--modulator` replaces automatic detection, and `--no-modulators` fits a
task-only model.

### Stages and run requirements

1. `glms` (always run): Canonical-HRF and selected-HRF conventional GLMs.
2. `reliability`: Odd-run and even-run HRF selections and cross-prediction
   (skipped automatically when a split has too few runs).
3. `betas`: Single-trial OLS and ridge beta series for canonical and selected
   HRFs.
4. `summaries`: Trial beta one-sample t maps and RT correlation summaries.

Run-count requirements depend on `--hrf-method`:
- With `--hrf-method cv`, `reliability` needs at least 2 odd and 2 even runs, and
  tuned ridge (`fractional_cv` or `cv`) needs at least 3 odd and 3 even runs plus
  at least one modulator.
- With `--hrf-method reml`, `reliability` needs at least 1 odd and 1 even run,
  tuned ridge needs at least 2 odd and 2 even runs, and `glms` alone can run on
  1 run.

### Outputs and exit codes

Outputs are written under `<output_dir>/<sub>/<ses>/func/<sub>_<ses>_task-<task>_*`
alongside `<output_dir>/dataset_description.json` and
`<output_dir>/<sub>_<ses>_task-<task>_report.html`:

| Descriptor (`desc-`) | Contents |
| --- | --- |
| `CanonicalGLM`, `OptimizedGLM` | Contrast effects, variances, t, z, $R^2$ maps, designs, and provenance |
| `GLMComparison` | Optimized-minus-canonical full-model $R^2$ |
| `HRF`, `HRFOdd`, `HRFEven` | Selected HRF IDs, CV $R^2$ or REML criteria/variances/flags, parameter maps, library tables, and provenance |
| `HRFReliability`, `HRFOddToEven`, `HRFEvenToOdd` | Odd/even curve correlations and split-test prediction $R^2$ |
| `CanonicalTrial...`, `OptimizedTrial...` (`OLS`, `Ridge`, `RidgeCV`, `FractionalCV`) | Per-run beta series, trial tables, $R^2$, RT maps, selected fractions/alphas, and tuning/outer-evaluation metadata |
| `boldtailor` | `<stem>_desc-boldtailor_metadata.json` recording settings and written artifacts |

| Exit code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | CLI usage/settings error or existing outputs under `--existing-results error` |
| `2` | Input discovery, loading, or validation error (missing files, missing modulator levels, or insufficient runs for CV) |

## Common problems

| Symptom | What to check |
| --- | --- |
| Missing confound or contrast column | Column names in each run's events/confounds and in `ModelSpec` |
| Rank-deficient design | Duplicate or overlapping trial regressors, collinear modulators, or too many columns for the run length |
| Rejected event timing | Onset/duration units, acquisition offsets, or events lacking sampled support inside the run |
| NaN $R^2$ or HRF selection | Constant signal, zero variance after nuisance projection, or no `"ok"` REML candidate |
| Categorical level missing in a run (exit 2) | Ensure every level occurs in every run, pass `--modulator <col>:categorical,indicator`, or omit the column |
| High memory usage | Reduce `--n-jobs` or `--block-size` |
