# User guide

[README](../README.md) · [API reference](api.md) · [NSD example](../examples/NSD/README.md)

## Preparing your data

The Python API takes one signal array per run, shaped **time points × features**.
Features must have the same meaning and order in every run. For images, keep
the mask or CIFTI brain-model axis you used to extract the signals so you can
put the results back into the correct locations.

Each run also needs an event table with `onset` and `duration` in seconds.
Add `trial_type` for condition-based models. A `modulation` column supplies
event amplitudes for conventional designs. Trial-level metadata such as image
identity and response time can stay in the event table for single-trial models.

For one run, pass an array and a DataFrame. For multiple runs, pass matching
lists. Runs can have different lengths and event counts:

```python
from boldtailor.data import from_arrays

# signals_by_run, events_by_run, and confounds_by_run are matching lists.
data = from_arrays(
    signals_by_run,
    events_by_run,
    frame_times=frame_times_by_run,
    confounds=confounds_by_run,
)
```

Use `tr=2.0` for uniformly sampled data beginning at zero, or provide explicit
frame times for acquisition offsets or nonuniform sampling. Supply one of
these timing options, not both. Frame times must increase strictly.
Signals and confounds must be finite; confounds need one row per time point.
Preprocessing, smoothing, scaling, and spatial alignment happen before this API.

Confound handling depends on the model:

| Model | Which nuisance regressors are used? |
| --- | --- |
| `fit()` | Columns named in `ModelSpec.confounds`, plus the requested drift terms and an intercept |
| `fit_prepared()` | Exactly the columns in your design matrix |
| Single-trial and HRF-selection APIs | Every supplied confound column, plus a run intercept; supply high-pass regressors yourself |

If you already use fMRIPrep cosine regressors, avoid also adding a second set
of cosine drifts. For conventional models, set `drift_model=None` when passing
the existing cosine columns as selected confounds.

## Condition effects and contrasts

Use `fit()` when each condition has a shared response amplitude within a run.
The [README example](../README.md#a-first-model) defines `data`, `model`, and
`result`; the following operations continue that example:

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

Contrasts can be expressions such as `"face - house"` or mappings such as
`{"face": 1, "house": -1}`. They refer to column names, so a different column
order between runs does not change the contrast. Every run must support the
requested contrast. The current interface supports t contrasts.

`ModelSpec` defaults to a Glover HRF, cosine drifts with a 0.01 Hz cutoff, and
AR(1) noise. Set `hrf_model="spm"` for the SPM HRF or `noise_model="ols"` for
ordinary least squares. Designs are constructed with Nilearn; supported HRF
bases can include temporal/dispersion derivatives or a callable kernel. A
fixed custom design gives you control over any additional regressors.

For several runs, Boldtailor fits each run independently and combines
contrasts as equal-weight fixed effects. P-values are directional, one-sided,
and uncorrected for multiple comparisons. `result.run_r2` contains one R²
array per run; `result.r2` pools residual and total sums of squares across runs.

## Using your own design matrix

Use `PreparedDesignAnalysis` when you already have a labeled design matrix.
This is useful for custom parametric regressors or designs built by another
package. Boldtailor fits the supplied columns as they are; include the
intercept, drift terms, and confounds you want in the model.

For a single run, suppose `signals` has time points in rows and `design` is a
DataFrame with columns `stimulus`, `motion`, and `constant`:

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

For multiple runs, pass lists of arrays, DataFrames, and role mappings. Every
column needs a role: `task`, `nuisance`, `intercept`, or `other`. Roles describe
the existing columns; they do not create or transform them. A task-versus-
confounds R² comparison requires a complete partition without `other` columns.

This API can consume a design prepared for FitLins or another workflow, but
there is no automatic FitLins integration or BIDS Stats Models parser.

## Measuring task-related variance

Pooled R² is calculated as:

```text
R² = 1 - sum_run(residual sum of squares)
         / sum_run(sum_time((signal - run mean)²))
```

Runs with different means or variances are handled through these sums; the
result is not an average of run-wise R² values. Constant signals have undefined
R², represented by NaN. A higher in-sample R² alone does not demonstrate better
prediction or more reliable trial estimates.

There are three distinct comparisons:

| Question | Result |
| --- | --- |
| How much variance do task regressors add beyond confounds? | Full R² minus nuisance-only R² |
| Does a selected HRF improve the single-trial fit over the canonical HRF? | Optimized full R² minus canonical full R², using the same estimator |
| Does a mean stimulus response predict a separate run? | HRF prediction R² after nuisance adjustment |

For conventional models, `task_delta_r2(data, model, result)` and
`task_delta_r2_prepared(prepared, prepared_result, contrasts=..., ...)` compute
the first comparison. Both refit full and nuisance models with OLS, even if
the original contrast inference used AR(1). Their result includes `full_r2`,
`nuisance_r2`, `raw_delta_r2`, and `delta_r2`; the latter clips only tiny negative
roundoff values. This is additional explained variance, not partial R².

These comparison functions require complete source metadata so they can check
that the supplied result belongs to the same input and model. Pass the same
contrasts, noise setting, and model metadata used for the original fit. See
[source records](api.md#source-records-and-saving) for the required fields.
Both comparison functions currently require finite pooled R² for every supplied
feature. Exclude constant features before comparing models, then restore NaNs
at those locations when reconstructing an image.

Single-trial results already contain `full_r2`, `nuisance_r2`, and `delta_r2`.
Their difference uses the actual OLS or ridge estimator and retains negative
values. The [NSD output reference](../examples/NSD/README.md#finding-your-results)
distinguishes these images from HRF-selection and independent prediction scores.

## Estimating a beta for every trial

`fit_single_trials(data, ridge_alpha=0.0)` creates a separate regressor for each
event row. It preserves event order, recorded durations, and sub-TR timing.
Repeated image IDs receive separate coefficients. The default HRF is SPM;
pass an `HrfCandidate` through `hrf=` to use one specified library HRF everywhere.

```python
from boldtailor.single_trial import fit_single_trials

ols = fit_single_trials(data)
ridge = fit_single_trials(data, ridge_alpha=0.1)
first_run_betas = ridge.run_betas[0]
trials = ridge.trial_table
```

Each beta array is **trials × features**. The trial table retains your event
metadata and adds run labels, within-run event indices, session-wide trial
indices, and unique trial IDs. Missing RT does not remove a trial from the fit.

Ridge is optional and uses a fixed, nonnegative penalty. Trial regressors are
adjusted for nuisances and scaled to unit length before regularization; returned
betas are restored to the original signal scale. Nuisance coefficients are
unpenalized. The value is an ordinary ridge penalty, not a fractional-ridge
fraction. There is no automatic choice of penalty or AR(1) single-trial fitting.

Supply your intended high-pass and other nuisance columns in `data.confounds`.
Do not include a `constant` column: the single-trial API adds it. Designs with
unidentifiable trial coefficients or no residual degrees of freedom are rejected.

## Selecting an HRF for each location

HRF selection requires at least two runs with matching feature order. In the
following example, `multi_run_data` is an `AnalysisData` object created from
lists of runs as described above:

```python
from boldtailor.hrf_library import expanded_hrf_library
from boldtailor.hrf_selection import select_hrf
from boldtailor.single_trial import fit_selected_hrfs

library = expanded_hrf_library()
selection = select_hrf(multi_run_data, library=library)
optimized = fit_selected_hrfs(multi_run_data, selection=selection)
optimized_ridge = fit_selected_hrfs(
    multi_run_data, selection=selection, ridge_alpha=0.1
)
```

The library contains the exact canonical SPM HRF plus 648 double-gamma curves
with varied delays, dispersions, undershoot ratios, and onset delays. You can
inspect `library.parameter_table`, `library.times`, and `library.curves`, or
create a smaller/custom library with `HrfLibrary.from_parameters()`.

At each feature, selection evaluates every candidate HRF in turn. For each
held-out run, it learns a mean stimulus amplitude from the remaining runs and
predicts that run. Prediction errors are pooled across folds before choosing
one winner. **The session map contains that winner's parameters; it does not
average parameters selected for individual runs.** Final beta fits use the
selected HRF and estimate unrestricted amplitudes for individual trials.

This method assumes that a mean stimulus response transfers across runs. It
does not require repeated images or equal responses to repeated presentations.
RT never enters HRF selection. Kernels are normalized to sum to one, so beta
values depend on that convention and are not estimates of the HRF's peak height.

The winning selection-CV score was used to choose the HRF. For independent
evaluation, select within a training set and predict a separate test set:

```python
from boldtailor.hrf_selection import evaluate_hrf_split

# Four runs in order; indices are zero-based.
evaluation = evaluate_hrf_split(
    multi_run_data,
    library=library,
    train_runs=[0, 2],
    test_runs=[1, 3],
)
held_out_r2 = evaluation.test_r2
improvement_over_spm = evaluation.delta_test_r2
```

Training needs at least two runs; testing needs at least one. HRFs and mean
amplitudes are fixed before scoring test signals. Candidate eligibility may
use test-run timing and confounds, but not test BOLD. Prediction R² uses
nuisance-adjusted signals, so it has a different denominator from full-model R².
Negative scores are retained.

## Comparing HRFs between sets of runs

Use `evaluate_hrf_split()` twice, reversing the train and test indices, and
compare the two `training_selection` results. Each set needs at least two runs
to perform its own internal cross-validation. Both calls enforce candidate
eligibility against the same set of timing/confound designs.

The NSD example does this automatically for odd and even runs. It saves matched
parameter images, including the peak time of each full HRF. HRF IDs are labels;
compare parameter values or reconstructed curves rather than correlating IDs.
Undefined selections use `-1` in Python and NaN in CIFTI maps.

You can apply an in-memory selection to other runs with the same feature
ordering using `fit_selected_hrfs()`. Supplying `feature_signature` to selection
and fitting lets Boldtailor check an identifier for that ordering; the NSD
example derives it from the CIFTI spatial axis. There is currently no loader
that turns the saved HRF maps back into a reusable selection object.

## Reaction-time checks

The NSD single-trial example correlates betas with RT after centering both
within runs. Missing or nonpositive RT values are excluded from this diagnostic,
not from beta estimation. It exports all-run, odd-run, even-run, and per-run
correlations and valid-trial counts.

Scatterplots use cortical vertices selected by canonical-OLS odd-run RT
correlations, then show even-run trials. With optimized HRFs, those plots use
HRFs selected only from odd runs. The production RT maps use all-run HRFs and
are descriptive. These checks provide neither significance tests nor a way to
tune the model. The helper functions are listed in the [API reference](api.md#example-workflows).

## Saving results and analysis records

Fits return arrays and tables. Save them using your preferred tools, or use
the image-writing examples to preserve NIfTI geometry or CIFTI axes. Each
result also carries a provenance record:

```python
record = result.provenance.to_dict()
record_json = result.provenance.canonical_json()
```

For file-backed analyses, provide `RunSources` descriptors when creating the
input object. These record relative filenames and file metadata; they do not
read or hash input contents. Arrays without source descriptors can still be
fitted, but cannot establish a stable source identity.

`project_bids_provenance()` prepares metadata files, and
`publish_artifact_set()` saves a collection of files together with overwrite
protection and rollback if writing fails. The examples use this to keep images
and metadata together. See the [API reference](api.md#source-records-and-saving)
for an example and the [developer guide](development.md) for storage details.

## Common problems

| Symptom | What to check |
| --- | --- |
| Confound or contrast column is missing | Column names in each run, and the names selected in `ModelSpec` |
| A design is rank deficient | Duplicate/overlapping trial regressors, redundant task columns, or too many regressors for the run length |
| Timing or trial support is rejected | Onset units, acquisition offset, run boundaries, and whether each event has a sampled response |
| R² or HRF parameters are NaN | A constant signal, no variance after nuisance adjustment, or unavailable split evaluation; inspect metadata |
| An output file already exists | Choose a new output directory for the example rather than overwriting a previous analysis |
| Parallel fitting uses too much RAM | Reduce `--n-jobs` or `--block-size`; final beta arrays also occupy memory |
