# A practical guide to Boldtailor

Boldtailor estimates task responses from preprocessed fMRI time series. It fits
condition contrasts, estimates single-trial betas, selects location-specific
hemodynamic response functions (HRFs), and supports encoding-guided
regularization and task-guided denoising.

This guide introduces the main components with runnable examples. See the
[user guide](user-guide.md) for methodological details and the
[API reference](api.md) for complete signatures and result schemas.

## 1. Choose an analysis

| Question | Component | Main function |
| --- | --- | --- |
| What is the average response to a condition? | Conventional GLM | `fit` |
| Can I fit a design matrix I already have? | Prepared designs | `fit_prepared` |
| How much variance does the task explain beyond nuisance regressors? | Task variance comparison | `task_delta_r2` |
| What is the response to each trial? | Single-trial GLM | `fit_single_trials` |
| Which HRF best describes a location's response? | HRF selection (CV or REML) | `select_hrfs` |
| Does the selected HRF predict new runs? | Independent HRF evaluation | `evaluate_hrf_split` |
| How much should trial estimates be regularized? | Fractional ridge selection | `score_fraction_candidates`, `select_ridge_fractions` |
| Can trial variables predict trial responses? | Trial encoding | `evaluate_trial_encoding` |
| Can data-derived noise PCs improve the fit? | Task-guided denoising | `select_denoising`, `with_denoising` |
| Can I run the BIDS CIFTI workflow from the shell? | Session workflow CLI | `boldtailor run` |

Each component can be used on its own; HRF selection, ridge tuning, and
denoising are optional.

## 2. Install and run a first model

From a checkout of the repository (Python 3.12+):

```bash
uv sync --group dev
```

Save the following as `first_model.py` and run `uv run python first_model.py`:

```python
import numpy as np
import pandas as pd

from boldtailor import ModelSpec, fit, from_arrays

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

Primary entry points (`from_arrays`, `ModelSpec`, `TaskModel`, `fit`,
`task_delta_r2`, `fit_single_trials`, `HrfLibrary`, `HrfCandidate`, `fit_reml`,
`select_random_effects_design`, `select_denoising`, `with_denoising`) are
exported at the top level of `boldtailor` and from their defining submodules.

## 3. Prepare the data

The array API takes one matrix per run, shaped **time points × features**
(voxels, vertices, or grayordinates), with identical feature ordering across
runs. Keep the mask or CIFTI brain-model axis used during extraction so you can
map results back to image space.

Each run requires an event DataFrame with `onset` and `duration` in seconds.
Include `trial_type` for condition models, and any trial covariates (such as
`response_time`) needed for modulators or encoding. Single-trial models fit one
regressor per event row.

For multiple runs, pass matching lists:

```python
multi_run_data = from_arrays(
    signals_by_run,
    events_by_run,
    tr=2.0,
    confounds=confounds_by_run,
)
```

Pass either `tr` (for uniform sampling starting at 0 s) or explicit
`frame_times` per run. Signals and confounds must be finite, with one confound
row per time point. Preprocessing, alignment, smoothing, and signal scaling must
be done beforehand; betas are returned in the input signal units.

Nuisance handling depends on the API:

| Component | Nuisance regressors included |
| --- | --- |
| Conventional GLM (`fit`) | Columns named in `ModelSpec.confounds`, requested drift terms, and a run intercept |
| Prepared design (`fit_prepared`) | Exactly the columns in your design matrix |
| Single-trial fitting and HRF selection | All supplied confound columns plus an added run intercept |

For single-trial fitting and HRF selection, include high-pass regressors in
`confounds` and omit any constant column. For conventional GLMs that already
include fMRIPrep cosine regressors in `ModelSpec.confounds`, set
`drift_model=None`.

## 4. Fit condition effects and contrasts

`ModelSpec` configures the HRF (`"glover"` by default), nuisance columns, drift
model (cosine high-pass at 0.01 Hz by default), noise model (`"ar1"` or
`"ols"`), and t contrasts.

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

Contrasts reference column names (via expressions or weight dicts), so column
order differences across runs do not affect them. P-values are one-sided and
uncorrected. Across multiple runs, runs are fitted independently and combined
via equal-weight fixed effects. `result.run_r2` gives per-run $R^2$, and
`result.r2` pools residual and total sums of squares across runs.

### Trial variables in a shared task model

`TaskModel` models a baseline task response alongside trial-level modulators:

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

Pass `task_model` to `ModelSpec(task_model=task_model, ...)` or `select_hrfs`.
Every categorical level must appear in every run. Modulators are uncentered:
`task` is the reference-level response at modulator value zero, and
`trial_type[house]` is the house-minus-face difference. `missing="indicator"`
assigns zero modulation to missing trials and adds a per-run `missing_<column>`
regressor when needed.

`detect_task_model` in `boldtailor.task_detection` can infer modulators from
event tables; inspect its output before fitting.

## 5. Use an existing design matrix

Use `PreparedDesignAnalysis` when your design matrix is already built:

```python
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared

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

Every column requires a role (`task`, `nuisance`, `intercept`, or `other`).
Boldtailor fits the matrix as supplied without adding or rescaling columns.

## 6. Measure variance added by the task

`task_delta_r2` (in `boldtailor.fit`) and `task_delta_r2_prepared` (in
`boldtailor.prepared_fit`) refit full and nuisance-only models with OLS and
report `full_r2`, `nuisance_r2`, `raw_delta_r2`, and zero-clipped `delta_r2`
(additional explained variance, not partial $R^2$). Both require complete
`RunSources` metadata on the input object to verify that the result matches the
parent analysis; see [source records](api.md#source-records-and-saving).

Single-trial results directly include `full_r2`, `nuisance_r2`, and `delta_r2`
computed from the fitted OLS or ridge model.

## 7. Estimate a response for each trial

`fit_single_trials` fits one regressor per event row:

```python
from boldtailor.single_trial import fit_single_trials

trials = fit_single_trials(data)
betas = trials.run_betas[0]
trial_table = trials.trial_table
```

Each `run_betas` entry has shape **trials × features**, and `trial_table`
records run and trial identifiers alongside the original event columns.

By default, `fit_single_trials` uses the canonical SPM HRF and OLS
(`ridge_alpha=0.0`). Passing `ridge_alpha > 0` applies L2 shrinkage to
nuisance-projected, unit-norm trial columns and restores betas to native signal
units; nuisance regressors are unpenalized. For built-in SPM, Glover, and
library HRFs, each event's predicted response is scaled to unit peak, so betas
represent peak response amplitude in signal units; see
[HRF normalization](api.md#hrf-libraries-selection-and-evaluation).

## 8. Select an HRF at each location

`select_hrfs` chooses a candidate HRF per location using leave-one-run-out
cross-validation (`method="cv"`, the default, requiring at least two runs) or
restricted maximum likelihood (`method="reml"`, see [Section 11](#11-select-hrfs-with-reml)):

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

`default_hrf_library()` includes canonical SPM (ID 0), 512 timing-space Sobol
candidates, and 20 empirical HRFs from GLMsingle. `selection.hrf_indices` stores
the winning candidate ID per feature (`-1` where undefined), and
`selection.delta_cv_r2` gives the CV $R^2$ gain over canonical SPM.

Pass `hrf_selection=selection` to `fit()` to use selected HRFs in a conventional
GLM:

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

Because contrast inference treats selected HRFs as fixed, selecting and testing
on the same runs does not account for selection uncertainty. To evaluate HRF
generalization on held-out runs, use `evaluate_hrf_split`:

```python
from boldtailor.hrf_selection import evaluate_hrf_split

evaluation = evaluate_hrf_split(
    multi_run_data, library=library,
    train_runs=[0, 1], test_runs=[2],
)
test_r2 = evaluation.test_r2
```

## 9. Choose regularization using trial prediction

Fractional ridge specifies the retained norm ratio of nuisance-projected trial
coefficients relative to OLS (`1.0` is OLS; smaller fractions shrink more).
`score_fraction_candidates` and `select_ridge_fractions` pick a fraction at each
location by held-out trial-encoding $R^2$ against fixed OLS beta targets:

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

Pass `library=library` to re-select HRFs inside each inner training fold
(requiring at least three runs for CV HRF selection, or two for REML). Alternatively,
`score_ridge_candidates` and `select_ridge_penalty` in
`boldtailor.ridge_selection` choose a single shared `ridge_alpha` across
features using a spatial percentile. Use `evaluate_trial_encoding` (or a
reusable `EncodingDesign`) in `boldtailor.trial_encoding` to evaluate trial
prediction on held-out outer test runs.

## 10. Add task-guided denoising

`select_denoising` (requiring at least three runs) identifies a noise pool of
features with low canonical ON-OFF $R^2$, extracts per-run temporal principal
components, and chooses a PC count using held-out task prediction (`pcstop`)
followed by an AR(1)-prewhitened F-test significance gate:

```python
from boldtailor.denoising import select_denoising, with_denoising

denoising = select_denoising(
    multi_run_data, task_model=task_model, library=library,
)
augmented_data = with_denoising(multi_run_data, denoising)
```

`with_denoising` appends the selected PC columns (`denoise_pc_000`, ...) to each
run's confounds. Rerun HRF selection and fitting on `augmented_data`; for
conventional GLMs, include the added column names in `ModelSpec.confounds`. See
[task-guided denoising](user-guide.md#task-guided-denoising) for details.

## 11. Select HRFs with REML

With `method="reml"`, `select_hrfs` models trial-to-trial amplitude variations
as zero-mean random effects and selects the candidate HRF with the best
restricted likelihood at each location:

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

REML can run on a single estimable run. The fixed design (confounds, intercept,
and `TaskModel` convolved with `reference_hrf_id`) and the supplied AR(1)
parameter `rho` are held fixed across candidates. Inspect
`diagnostics.selection.fit` (`status`, `ratio`, `noise_variance`,
`effect_variance`, `at_upper`, `unrefined`), `diagnostics.delta_criterion`
(canonical minus selected criterion; lower criteria are better), and
`diagnostics.selection.any_at_upper` / `any_unrefined`. Features without an
`"ok"` candidate receive HRF ID `-1`. See the [REML API reference](api.md#scalar-reml)
for direct fixed/random design fitting with `boldtailor.reml`.

## 12. Examine reliability and save results

- **HRF agreement**: `curve_correlations` and `compare_hrfs` in
  `boldtailor.reliability` measure waveform and parameter agreement across splits
  or sessions.
- **Trial summaries**: `one_sample_t` and `correlate_rt` in
  `boldtailor.diagnostics` summarize trial betas and within-run RT correlations.
- **Provenance and atomic saving**: Supply `SourceRef` and `RunSources` from
  `boldtailor.provenance` when building inputs, project BIDS metadata with
  `project_bids_provenance`, and write artifacts transactionally with
  `publish_artifact_set` in `boldtailor.publication`.
- **CIFTI and parallel helpers**: Use `scalar_artifact` and `read_scalar` in
  `boldtailor.cifti` for dense scalar maps, and `map_blocks` in
  `boldtailor.parallel` to process feature blocks across worker processes.

## 13. Run the BIDS CIFTI workflow

`boldtailor run` executes the end-to-end BIDS + fMRIPrep fsLR-91k CIFTI
workflow and generates a BIDS derivative dataset with an HTML report. Inspect
the resolved plan first with `--dry-run`:

```bash
uv run boldtailor run --bids-dir /data/study --subject sub-01 \
    --session ses-01 --task example --dry-run
```

Remove `--dry-run` to run the analysis, or pass `--help` to see all options.
The workflow fits canonical and selected-HRF GLMs, evaluates odd/even HRF
reliability, estimates single-trial betas, and writes summary maps and tables.
Use `--hrf-method reml --reml-rho 0.0` for REML HRF selection, or
`--ridge-mode off` / `--ridge-mode fixed` when fewer runs are available. The
same workflow is callable from Python via `WorkflowSettings`
(`boldtailor.workflow.settings`) and `run_workflow` (`boldtailor.workflow.run`).

## 14. Pre-analysis checklist

- Verify event onsets and durations against retained scans and acquisition-time
  offsets.
- Check design matrices for collinearity or unidentifiable trial columns.
- Confirm signal units, confound selection, and identical spatial ordering
  across runs.
- Distinguish in-sample fit $R^2$, cross-validated selection scores, and
  held-out outer-test $R^2$.
- Check NaN masks and grid-boundary flags (`at_boundary`, `at_parameter_bound`,
  `any_at_upper`).
