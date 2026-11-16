# A practical guide to Boldtailor

Boldtailor estimates task responses from preprocessed fMRI time series. It can
fit condition effects, estimate a response for each trial, and choose a
hemodynamic response function (HRF) for each voxel or grayordinate. It also
provides tools for regularization, denoising, and evaluation across runs.

This guide introduces the main components and shows how to use them. The
[detailed guide](user-guide.md) explains the methods and options; the
[API reference](api.md) lists function arguments and result fields.
The package is under active development, and its interface may change.

## 1. Choose an analysis

Start with the quantity you want to estimate.

| Question | Component | Main function |
| --- | --- | --- |
| What is the average response to a condition? | Conventional general linear model (GLM) | `fit` |
| Can I fit a design matrix I already have? | Prepared designs | `fit_prepared` |
| How much variance does the task explain beyond nuisance regressors? | Task variance comparison | `task_delta_r2` |
| What is the response to each trial? | Single-trial GLM | `fit_single_trials` |
| Which HRF best describes a location's response? | HRF selection by cross-validation or REML | `select_hrfs` |
| Does the selected HRF predict new runs? | Independent HRF evaluation | `evaluate_hrf_split` |
| How much should trial estimates be regularized? | Fractional ridge selection | `score_fraction_candidates`, `select_ridge_fractions` |
| Can trial variables predict trial responses? | Trial encoding | `evaluate_trial_encoding` |
| Can additional nuisance regressors improve the analysis? | Task-guided denoising | `select_denoising`, `with_denoising` |
| Can I run the CIFTI analysis from the shell? | BIDS session workflow | `boldtailor run` |

These components can be used separately. HRF selection and ridge selection
are optional. A conventional GLM is a useful starting point for checking
events, timing, confounds, and contrasts.

## 2. Install and run a first model

From a checkout of the repository, using Python 3.12 or later:

```bash
uv sync --group dev
```

Save the following as `first_model.py`, then run
`uv run python first_model.py`. It uses random signals to demonstrate the
interface; the resulting statistics have no scientific interpretation.

```python
import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor.fit import fit
from boldtailor.model import ModelSpec

signals = 100 + np.random.default_rng(7).normal(size=(100, 3))
events = pd.DataFrame({
    "onset": [10.0, 40.0, 70.0, 100.0],
    "duration": [2.0, 2.0, 2.0, 2.0],
    "trial_type": ["face", "house", "face", "house"],
})

data = from_arrays(signals, events, tr=2.0)
model = ModelSpec(
    contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
    hrf_model="spm",
    noise_model="ar1",
)
result = fit(data, model)
print(result.effect("face_gt_house"))
```

Import functions from their named modules, as above. Later examples use this
`data`, or explicitly identify the additional inputs they need.

## 3. Prepare the data

The array interface uses one matrix per run, with **time points in rows and
brain locations in columns**. A location may be a voxel, vertex, or CIFTI
grayordinate. Locations must have the same order in every run. Keep the mask
or CIFTI axis used to extract them so that you can reconstruct images later.

Each run also needs an event table. `onset` and `duration` are in seconds.
`trial_type` identifies conditions. Other columns can hold response times,
stimulus identities, or other trial variables. Single-trial models use one
event row per presentation, including repeated presentations of the same image.

For several runs, supply matching lists:

```python
# Replace these lists with your own arrays and pandas tables.
multi_run_data = from_arrays(
    signals_by_run,
    events_by_run,
    tr=2.0,
    confounds=confounds_by_run,
)
```

Use `frame_times=frame_times_by_run` instead of `tr` when the acquisition
times need to be specified explicitly. Events and frame times must use the
same time origin. Runs may differ in length and number of events.
Signals and confounds must be finite, and confounds need one row per time point.

The array interface does not preprocess, align, smooth, or scale images.
Choose the signal units before fitting. Betas are returned in those units;
they are not automatically percent signal change.

Nuisance handling differs between components:

| Component | Nuisance regressors |
| --- | --- |
| Conventional GLM | Confounds named in `ModelSpec.confounds`, the requested drift terms, and an intercept |
| Prepared design | Only the columns you supply |
| Single-trial fitting and HRF selection | All supplied confound columns and an added intercept |

For the last row, supply high-pass regressors yourself and omit a column named
`constant`. For a conventional model using existing cosine confounds, set
`drift_model=None` to avoid adding a second set of cosine regressors.

## 4. Fit condition effects and contrasts

`ModelSpec` describes a conventional GLM. It specifies the HRF, nuisance
regressors, drift model, noise model, and contrasts. The defaults are a Glover
HRF, cosine drifts with a 0.01 Hz cutoff, and AR(1) noise. The first example
explicitly chooses the SPM HRF.

Inspect the design before interpreting the fit:

```python
from boldtailor.design import compile_designs

designs = compile_designs(data, model)
print(designs[0].matrix)

effect = result.effect("face_gt_house")
variance = result.variance("face_gt_house")
t_value = result.stat("face_gt_house")
z_value = result.z_score("face_gt_house")
p_value = result.one_sided_p_value("face_gt_house")
```

Each output contains one value per location. A contrast refers to column
names, so it remains valid if column order changes between runs. The package
currently supports t contrasts. P-values are one-sided and uncorrected for
multiple comparisons.

With several runs, each run is fitted separately and contrasts are combined
using equal-weight fixed effects. This estimates a within-subject effect;
it is not a group analysis. Runs contribute equally even if their lengths
or noise levels differ.

`result.run_r2` contains the fit R² for each run. `result.r2` pools residual
and total sums of squares across runs. Neither is a measure of performance
on new data.

### Trial variables in a shared task model

`TaskModel` describes an overall task response and changes associated with
trial variables. For example, this model includes response time and a
face-versus-house difference:

```python
from boldtailor.model import Modulator, TaskModel

task_model = TaskModel((
    Modulator("response_time", missing="indicator"),
    Modulator(
        "trial_type", kind="categorical",
        levels=("face", "house"), reference="face",
    ),
))
```

Use it as `ModelSpec(task_model=task_model, ...)` or pass it to HRF selection.
The input events must contain both columns. Each categorical level must occur
in each run. Numeric modulators are not centered: `task` is the face response
at response time zero, and `trial_type[house]` is the house-minus-face effect.
The missing-value option keeps trials with missing RT and adds an indicator
regressor. It does not estimate their missing response times.

`detect_task_model` in `boldtailor.task_detection` can construct a model from
event tables. Inspect its result before fitting; automatic detection is a
convenience, not a scientific model specification.

## 5. Use an existing design matrix

Use a prepared design when another tool has already constructed your
regressors. Supply a pandas DataFrame with one row per scan and a name for
each column. Include every intercept, drift, and confound you need.

```python
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared

# design has columns stimulus, motion, and constant.
prepared = PreparedDesignAnalysis.from_arrays(
    signals, design, tr=2.0,
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

Column roles describe the supplied matrix; they do not change it. This
interface is useful for custom designs but does not automatically import a
BIDS Stats Model or a FitLins analysis.

## 6. Measure the variance added by the task

`task_delta_r2`, from `boldtailor.fit`, compares a full model with a model
containing only nuisance regressors. Its `delta_r2` is full-model R² minus
nuisance-only R². This is additional explained variance, not partial R².
`task_delta_r2_prepared`, from `boldtailor.prepared_fit`, provides the same
comparison for prepared designs.

Both functions refit the models using ordinary least squares (OLS), even if
contrast inference used AR(1). They require complete source records to check
that the result belongs to the supplied analysis; the small synthetic example
above does not provide those records. See [source records](api.md#source-records-and-saving).

Single-trial results already contain `full_r2`, `nuisance_r2`, and `delta_r2`.
For ridge fits, the difference uses the regularized fit and can be negative.

## 7. Estimate a response for each trial

A single-trial model assigns one coefficient to each event rather than one
to each condition:

```python
from boldtailor.single_trial import fit_single_trials

trials = fit_single_trials(data)
betas = trials.run_betas[0]
trial_table = trials.trial_table
```

`betas` has **trials in rows and brain locations in columns**. The trial table
preserves event metadata and adds run and trial identifiers. Use these
identifiers when joining betas to behavioral data.

The default uses the SPM HRF and OLS. Set `ridge_alpha=0.1`, for example, to
apply a fixed ridge penalty. This shrinks coefficients and may reduce noise
when trial responses overlap. The value is a penalty, not a fraction.
Trial columns are adjusted for nuisance regressors and normalized before
this penalty is applied; returned betas are restored to the original scale.
Nuisance coefficients are not penalized. Single-trial fitting does not use AR(1).

For the built-in SPM, Glover, and library HRFs, each event response is scaled
to peak at one. Its beta represents peak response amplitude in signal units.
Derivative bases, FIR designs, and user-supplied kernels have different
scaling conventions; see [HRF normalization](api.md#hrf-libraries-selection-and-evaluation).

## 8. Select an HRF at each location

HRF selection chooses a response shape at each brain location. Two methods
are cross-validation and restricted maximum likelihood (REML). Cross-validation
compares prediction across runs; REML compares how well candidate HRFs account
for the data under a model of trial-to-trial variation.

The `select_hrfs` interface defaults to cross-validation. The
[REML section](#11-select-hrfs-with-reml) describes the alternative and its
current interface.

Cross-validation needs at least two runs. For each candidate, task amplitudes estimated
from training runs predict the held-out run. The candidate with the best
pooled score is selected at each location. Repeated stimuli are not required,
but task amplitudes must be sufficiently stable across runs.

```python
from boldtailor.hrf_library import default_hrf_library
from boldtailor.hrf_selection import select_hrfs
from boldtailor.single_trial import fit_selected_hrfs

library = default_hrf_library()
selection = select_hrfs(multi_run_data, library=library)
optimized_trials = fit_selected_hrfs(
    multi_run_data, hrf_selection=selection,
)
```

The default library contains canonical SPM, 512 sampled HRFs, and 20 empirical
HRFs from GLMsingle. Other factories provide a parameter grid, parameter-space
Sobol samples, timing-space samples, or the empirical library alone. Inspect
`library.curves` and `library.parameter_table`; save them with the results.

By default, selection models one mean task response. Pass
`task_model=task_model` to include the trial variables described above.
`selection.hrf_indices` identifies the selected curves;
`selection.delta_cv_r2` gives their score improvement over canonical SPM.
Undefined selections have ID `-1`. IDs are labels, so correlations between
ID maps do not measure HRF similarity.

The same selection can be used in a conventional GLM:

```python
selected_model = ModelSpec(
    contrasts={"task": {"task": 1.0}},
    task_model=TaskModel(),
    hrf_model="spm",
    noise_model="ols",
    drift_model=None,
    confounds=tuple(multi_run_data.confounds[0].columns),
)
selected_result = fit(
    multi_run_data, selected_model, hrf_selection=selection,
)
```

This example assumes the confound names are the same in every run. Use a
compatible task model for selection and fitting. If you add modulators to
selection, revise the fitted model accordingly.

Contrast statistics treat selected HRFs as fixed. Selecting and testing on
the same BOLD data does not account for selection uncertainty. For independent
evaluation, reserve test runs before selecting an HRF:

```python
from boldtailor.hrf_selection import evaluate_hrf_split

# Requires at least three runs; indices are zero-based.
evaluation = evaluate_hrf_split(
    multi_run_data, library=library,
    train_runs=[0, 1], test_runs=[2],
)
test_r2 = evaluation.test_r2
```

`test_r2` evaluates the training-selected HRF and amplitudes. A negative score
is possible and should be retained. A selection score used to choose an HRF
is not an independent estimate of its predictive performance.

## 9. Choose regularization using trial prediction

Fractional ridge expresses shrinkage as the coefficient length retained
relative to OLS. A fraction of 1 gives OLS; smaller fractions give stronger
shrinkage. Boldtailor can choose a separate fraction at each location by
asking how well trial variables predict betas in held-out runs.

The following example requires multiple runs with the response-time and
condition columns used in `task_model`:

```python
from boldtailor.fractional_ridge import (
    score_fraction_candidates, select_ridge_fractions,
)
from boldtailor.trial_encoding import trial_predictors

predictors = trial_predictors(multi_run_data.events, task_model)
scores = score_fraction_candidates(
    multi_run_data, predictors, fractions=[0.1, 0.25, 0.5, 0.75, 1.0],
)
choice = select_ridge_fractions(scores)
regularized_trials = fit_single_trials(
    multi_run_data, ridge_fraction=choice.ridge_fraction,
)
```

`trial_predictors` converts the task variables to numeric encoding predictors,
including categorical indicators. It retains event-row alignment. Missing
predictors can exclude a trial from encoding without deleting its event from
the BOLD model.

The example uses canonical HRFs and needs at least two runs. Supplying
`library=library` to the scorer selects HRFs inside each training fold and
needs at least three runs. For the final fit in that case, select HRFs on the
training data and use `fit_selected_hrfs` with the selected fraction map.

Fractional selection uses fixed OLS validation betas for all candidate
fractions. The objective measures prediction of noisy trial estimates; it
does not directly measure recovery of the true neural response. The fraction
is shared across runs, but the corresponding penalty can differ by run and
location. Fractional ridge uses raw projected coefficient norms, whereas
fixed `ridge_alpha` uses normalized trial columns.

The older `score_ridge_candidates` and `select_ridge_penalty` functions in
`boldtailor.ridge_selection` choose one alpha for all locations using a
spatial percentile. Their validation betas change with the candidate penalty,
so their selection objective differs from fractional ridge.

Use `evaluate_trial_encoding` in `boldtailor.trial_encoding` to evaluate
predictors against beta series with explicit training and test run indices.
The default score measures variation within runs; `encoding_mode="absolute"`
also evaluates absolute levels. Keep outer test runs out of HRF and ridge
tuning. An RT association used for tuning is not an independent validation
on those same data.

## 10. Add task-guided denoising

Denoising is optional and available through the array interface. It requires
at least three runs. It identifies poorly task-predicted locations, extracts
temporal principal components from their signals, and selects a component
count using task prediction and a significance gate.

```python
from boldtailor.denoising import select_denoising, with_denoising

denoising = select_denoising(
    multi_run_data, task_model=task_model, library=library,
)
augmented_data = with_denoising(multi_run_data, denoising)
```

`augmented_data` contains additional confound columns. Rerun HRF selection and
fitting on it. Single-trial fits use those columns automatically; conventional
models must include them in `ModelSpec.confounds`.

This procedure follows GLMdenoise ideas but differs from GLMsingle's
implementation. Pool definition and initial HRF selection use all supplied
runs, so its internal prediction scores are not a fully independent test of
the entire denoising procedure. Inspect the selected count and diagnostics;
zero components is a valid result. See the
[denoising method description](user-guide.md#task-guided-denoising).

## 11. Select HRFs with REML

REML is an alternative to cross-validation for HRF selection. Each candidate
HRF defines a set of trial-response regressors. Their coefficients are modeled
as random effects, allowing response amplitudes to vary across trials. REML
estimates the trial-effect variance and noise variance for each candidate,
then selects the HRF with the best restricted likelihood at each location.
Variance estimation is part of this selection procedure.

Use the same selection and fitting interfaces as for cross-validation:

```python
from boldtailor.hrf_options import RemlHrfOptions
from boldtailor.hrf_selection import select_hrfs
from boldtailor.single_trial import fit_selected_hrfs

selection = select_hrfs(
    data, library=library, method="reml",
    reml_options=RemlHrfOptions(rho=0.0, reference_hrf_id=0),
)
trial_fit = fit_selected_hrfs(data, hrf_selection=selection)
diagnostics = selection.reml
```

One run is sufficient if the restricted model is estimable. The fixed mean
contains the run's confounds, an intercept, and the requested `TaskModel`
convolved with one reference HRF. Missing-value indicators use that same
reference. The default reference is canonical SPM (ID 0). This fixed design
stays unchanged across candidates. A wrong reference mean can affect selection
by leaving mean structure that is then modeled as trial variance.

Each trial contributes one response column, using the candidate's unit-peak
HRF and the event duration. Trial effects are independent, zero-mean deviations.
Their variance and the noise variance are estimated per location and shared
across runs. Repeated stimulus IDs do not imply shared random effects.

`rho` supplies AR(1) noise correlation. Zero means independent noise. The
array API accepts a scalar or one value per feature; the command line accepts
a scalar. Correlation is not estimated automatically. For independent
evaluation, estimate any supplied correlation using training data only.

`selection.hrf_indices` holds library IDs, with -1 where no supported
candidate won. The CV score fields are `None`. Inspect
`diagnostics.selection.fit` for variance estimates and status,
`diagnostics.delta_criterion` for canonical minus selected criterion, and
`diagnostics.selection.gap` for the runner-up difference. Smaller criteria
are better. These are likelihood comparisons, not R² or significance tests.

Check `any_unrefined` and `any_at_upper` on `diagnostics.selection`, as well
as `unrefined` and `at_upper` on the winning fit. An unresolved losing
candidate makes the comparison provisional. Zero variance and unidentifiable
fits do not win. No canonical fallback is inserted for unsupported features.
The [REML reference](api.md#scalar-reml) describes the statuses.

`evaluate_hrf_split(..., method="reml", reml_options=...)` selects using
training runs only. It evaluates the existing mean-response prediction model,
not predictions of unobserved trial deviations. It needs one training and one
disjoint test run. REML can select an HRF even when a wide trial design cannot
support an OLS trial fit; downstream fitting retains its estimability checks.

Both ridge scorers accept `hrf_method="reml"` and `reml_options` when a
library is supplied. They repeat HRF selection within each training fold and
need at least two runs, compared with three for CV HRF selection.
Task-guided denoising retains its internal CV procedure. REML can be used for
HRF selection after denoising.

The lower-level `boldtailor.reml` interface remains available for models
specified directly by fixed and random design matrices.

## 12. Examine reliability and save results

Compare HRFs selected from disjoint run sets using
`curve_correlations` in `boldtailor.reliability`. `compare_hrfs` summarizes
agreement across sessions. Both require a common library and matching spatial
order. Similar curves can have different parameter values, so examine curve
agreement as well as parameter agreement.

`boldtailor.diagnostics` provides trial-beta summaries and within-run beta/RT
correlations. Its `one_sample_t` assumes independent trials. Overlapping
responses and regularization can induce dependence between trial estimates,
so that summary is not a substitute for a justified inferential model.

Core result objects contain arrays, tables, and analysis records. Most numeric
results are read-only; use `.copy()` before editing them. Save trial metadata
alongside beta arrays and preserve feature order. For HRF analyses, save the
library and selection information as well.

`SourceRef` and `RunSources` in `boldtailor.provenance` describe input files.
Supply these when constructing an analysis if you need traceable source
identities. `project_bids_provenance` prepares metadata, and
`publish_artifact_set` writes a collection of results with overwrite protection.
The [saving example](api.md#source-records-and-saving) shows these together.

For CIFTI data, `boldtailor.cifti.scalar_artifact` creates scalar-map artifacts
using the original brain-model axis. `read_scalar` checks that axis when
reading saved maps. For NIfTI data, reconstruct maps with the original mask
using your imaging tools.

`boldtailor.parallel.map_blocks` distributes feature blocks across workers.
Reduce block size or worker count if memory is limited. Configure the standard
Python logger named `boldtailor` to record progress in a custom script.

## 13. Run the BIDS CIFTI workflow

The command-line workflow reads BIDS events and fMRIPrep fsLR-91k CIFTI
time series, confounds, and sidecars. It is specific to those inputs; use the
array interface for other image formats or preprocessing pipelines.

First inspect the resolved inputs and settings:

```bash
uv run boldtailor run --bids-dir /data/study --subject sub-01 \
    --session ses-01 --task example --dry-run
```

Replace the paths and labels with those of your dataset. Remove `--dry-run`
to fit the models. Use `uv run boldtailor run --help` for all options.

The workflow fits canonical and selected-HRF GLMs, evaluates HRF reliability,
estimates trial betas, and writes summaries and an HTML report. It detects
response time and categorical trial type from the event tables; explicit
`--modulator` options replace that detection. Check the dry-run task model
before fitting.

The default fractional ridge workflow needs at least three odd and three
even runs and at least one modulator. With fewer runs, use `--ridge-mode off`
for OLS, `--ridge-mode fixed` for a fixed penalty, or `--skip-stage betas`.
Choose REML with `--hrf-method reml --reml-rho 0.0`; optionally set
`--reml-reference-hrf-id`. REML needs two odd and two even runs for nested
ridge tuning, or one per half for HRF reliability. An HRF-only REML analysis
can use a single run. Denoising remains an array-level component.

REML maps contain criteria, variance estimates, rho, status codes, and search
flags. The JSON metadata explains the status codes; the report counts
unsupported and provisional selections. Independent prediction maps still
contain R². Automatic output paths distinguish CV and REML settings.

Results are written under the BIDS derivatives directory unless
`--output-dir` is supplied. Outputs include CIFTI maps, trial beta series,
design matrices, trial tables, settings, provenance, and the report. Existing
outputs are protected by default. See the
[workflow reference](user-guide.md#running-the-full-workflow) for filenames,
stage controls, overwrite behavior, and exit codes.

The same workflow is available from Python through `WorkflowSettings` in
`boldtailor.workflow.settings` and `run_workflow` in `boldtailor.workflow.run`.
The [example notebooks](../examples/bids_workflow/README.md) show analyses of
NSD data. Code under `experiments/` records research analyses and is not part
of the installed package interface.

## 14. Checks before interpreting an analysis

- Check event timing against the retained scans, especially after removing
  initial volumes or applying acquisition-time offsets.
- Inspect design matrices for redundant columns and poorly separated trials.
- Confirm signal units, confound choices, and spatial order across runs.
- Keep fit R², selection scores, and independent test scores distinct.
- Inspect undefined results and selection boundaries. NaN is not a zero
  response; an endpoint winner may indicate that the candidate range is too small.
- Record the package version, source files, model settings, HRF library,
  regularization grid, and training/test split with the outputs.
