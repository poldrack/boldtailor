# Shared Task Model for HRF Selection and GLM Fitting

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

## Goal

HRF selection and conventional GLM fitting must score and fit the same task
model. For the NSD workflow that model is: an all-trials task regressor,
within-run centered response-time modulation, uncentered trial type, the
selected confounds, and a missing-RT indicator in runs that have missing RTs.

## Current behavior

- `select_hrf` scores each candidate with a single unit-amplitude all-trials
  regressor. RT and trial type never enter. Eligibility checks the
  single-trial design rather than the GLM design.
- The selected-HRF GLM (`fit(..., hrf_selection=...)`) builds its design from
  Nilearn-format events supplied by the caller. The NSD example expands raw
  trials into those events in `glm_events`, centering both RT and trial type.
- Selection and fitting therefore use different task models built by
  different code paths.

## Decisions

These were settled during design review.

1. **Declarative task model.** A small frozen `TaskModel` describes
   modulators. The default is task-only, so existing callers and fixtures
   keep working. The NSD model is one instance.
2. **Missing-RT indicator is a per-run profiled regressor.** Its coefficient
   is free within each run, like a confound, but convolved with the candidate
   HRF. It appears only in runs with missing RTs.
3. **Eligibility is the GLM design.** A candidate is eligible when its task
   model design is full rank with residual degrees of freedom in every run and
   the pooled training design is invertible in every fold. The single-trial
   check leaves the selection path.
4. **Held-out prediction objective is kept.** Task amplitudes are learned from
   the other runs and frozen for the held-out run. In-sample R² was
   considered and rejected.
5. **Nilearn builds every design.** Both selection and fitting call
   `make_first_level_design_matrix`. No new convolution code.
6. **Trial type is uncentered.** This changes the NSD example, which currently
   centers it.

## Design

### Task model specification

Add to `boldtailor/model.py`:

```python
@dataclass(frozen=True)
class Modulator:
    column: str
    center: bool = True
    missing: str = "error"      # "error" | "indicator"

@dataclass(frozen=True)
class TaskModel:
    modulators: tuple[Modulator, ...] = ()

    @property
    def regressor_names(self) -> tuple[str, ...]: ...   # ("task", *columns)
    @property
    def fingerprint(self) -> str: ...
    def to_dict(self) -> dict: ...
```

Rules:

- Regressor `task` is always present with unit amplitude for every trial.
- Each modulator produces one regressor named after its event column.
  Amplitude is the column value, centered within run over observed trials
  when `center` is true. Observed means finite.
- `missing="indicator"` sets the modulator amplitude to zero for non-finite
  trials and adds a regressor `missing_<column>` with unit amplitude on those
  trials, only in runs where at least one value is missing.
- `missing="error"` rejects non-finite values.
- Column names must be unique, must not be `task`, `constant`, `onset`, or
  `duration`, and must not begin with `missing_`. A raw events column named
  `trial_type` is an ordinary modulator column; the expanded Nilearn frame
  uses `trial_type` for regressor names, but that frame is derived, not input.
- Every task regressor must have a nonzero amplitude somewhere in every run
  after centering. A violation is a data error raised before any candidate is
  scored, because it does not depend on the HRF.
- `fingerprint` is the SHA-256 of the canonical JSON of `to_dict()`.

The NSD model:

```python
NSD_TASK_MODEL = TaskModel((
    Modulator("response_time", center=True, missing="indicator"),
    Modulator("trial_type", center=False),
))
```

The NSD loader converts nonpositive RTs to NaN so that the package rule
"missing means non-finite" applies.

### Event expansion

Add `expand_events(events, task_model) -> pd.DataFrame` in a new module
`boldtailor/_task_design.py`. It maps raw per-trial events to Nilearn's events
format: one row per trial per regressor, with columns `onset`, `duration`,
`trial_type` (the regressor name), and `modulation` (the amplitude). This is
the only new construction code, and it is a table transform, not a signal
computation. `TaskModel.regressor_names` lists the **task** regressors whose
amplitudes are shared across runs; `TaskModel.profiled_names` lists the
`missing_<column>` indicators whose coefficients are free within each run.

### Design construction

One helper in `_task_design.py` builds a run's task columns for any HRF:

```python
def task_columns(expanded_events, frame_times, hrf, model_settings) -> pd.DataFrame
```

It calls `make_first_level_design_matrix` with `drift_model=None`, no added
regressors, and `hrf_model` equal to `"spm"`, `"glover"`, or the candidate's
kernel callable, then drops the constant and strips Nilearn's callable name
suffix, as `_custom_design` already does. Columns are ordered as task
regressors first, then whichever profiled regressors the run contains.
Selection and GLM fitting both call this helper, so the task columns are
identical by construction. Selection always uses `min_onset=-24.0` and
`oversampling=50`, records both in its provenance, and rejects any onset
earlier than the first frame plus `min_onset` or at or after the last frame,
as it does today. A selected-HRF GLM with a task model must use the same two
values, so no event is excluded on one side and kept on the other.

Nuisance columns are not shared. Selection projects onto every supplied
confound column plus a constant, as documented. The GLM takes drifts and the
`ModelSpec.confounds` subset from Nilearn's nuisance builder, as it does today.
Callers who want identical nuisance sets pass every confound column to
`ModelSpec` and set `drift_model=None`, which the NSD example already does.

If Nilearn emits stdout notices for modulated events, the helper suppresses
them locally the way the NSD example already does; this is verified during
implementation.

### Selection objective

For run r and candidate h, let N_r be the nuisance matrix (selected confounds
plus constant), P_{r,h} the convolved profiled columns for that run (empty when
no RT is missing), and X_{r,h} the K task columns. Let Q_{r,h} be an orthonormal
basis for the span of [N_r, P_{r,h}]. Define

- X̂_{r,h} = X_{r,h} − Q_{r,h} Q_{r,h}ᵀ X_{r,h}
- ŷ_{r,h} = y_r − Q_{r,h} Q_{r,h}ᵀ y_r
- A_{r,h} = X̂ᵀX̂ (K×K), B_{r,h} = X̂ᵀŷ (K×F), C_{r,h} = ‖ŷ‖² (F)

Because X̂ is orthogonal to Q, B_{r,h} = X̂_{r,h}ᵀ y_r and does not require
re-projecting y for every candidate. C_{r,h} equals the confound-only
projected energy minus the energy explained by the projected profiled
columns, a rank-|P| correction.

Leave-one-run-out for held-out run r:

- β_{−r,h} = (Σ_{s≠r} A_{s,h})⁻¹ Σ_{s≠r} B_{s,h}, a K×F matrix solved per
  feature batch.
- loss_{r,h} = C_{r,h} − 2 Σ_k β_k B_{r,h,k} + βᵀ A_{r,h} β, clipped at zero
  within the existing roundoff tolerance.

Score:

```
cv_r2_h = 1 − Σ_r loss_{r,h} / Σ_r C_r
```

where C_r is the confound-only projected energy and therefore does not depend
on the candidate. Rankings depend only on pooled loss. When the task model is
task-only, K = 1 and P is empty, and every quantity reduces exactly to the
current implementation.

`evaluate_hrf_split` follows the same algebra with training runs supplying
A and B and test runs scored with the frozen β. `training_amplitudes` becomes
K×F.

Memory: B grows by a factor of K over today. The existing candidate batch loop
is retained; no new knob is added.

### Eligibility

A candidate is eligible when, in every run, the matrix [X_{r,h}, P_{r,h}, N_r]
has full column rank at the existing SVD tolerance and positive residual
degrees of freedom, and when, in every fold, Σ_{train} A_{s,h} is invertible at
the same tolerance. Both failures mark the candidate `-inf` for all features,
using the existing lazy `choose_eligible` mechanism. Reasons are recorded in
the eligibility table as they are now. The tolerance is numerical only; no
statistical threshold is introduced.

### Cached run design

`RunDesign` in `_hrf_cv.py` changes as follows:

- Construction takes the expanded events and the task model. The cache key
  and fingerprint include the modulator amplitudes and the task model
  fingerprint. The docstring no longer claims RT is never retained; modulator
  values are part of the design.
- Per candidate it lazily builds and caches the Nilearn task columns via the
  shared helper, splits them into task and profiled blocks, forms Q_{r,h},
  X̂_{r,h}, A_{r,h}, and the C correction.
- `task_design(candidate_id)` returns the run's Nilearn task and profiled
  columns for that candidate, in the order described above. The GLM compiler
  produces the same frame by calling the same helper on the same expanded
  events; tests assert equality.
- `eligible(candidate_id)` implements the run-level rank check above.
- `trial_matrix(candidate_id)` and a renamed `trial_eligible(candidate_id)`
  remain for the single-trial, ridge CV, and the older NSD single-trial
  example, which check estimability of their own designs.

### GLM integration

- `ModelSpec` gains `task_model: TaskModel | None = None`. When set,
  `hrf_model` must be `"spm"` or `"glover"`; derivative or dispersion bases
  raise, because the task model defines exactly one column per regressor.
  Events supplied to the data object must then be raw per-trial tables with
  the modulator columns.
- `compile_designs` with a task model builds task columns through the shared
  helper and concatenates the Nilearn nuisance matrix (drifts, confounds,
  constant) from `_make_nuisance_matrix`. Without a task model it is
  unchanged.
- `compile_group_designs` with a task model builds every candidate's task
  columns, including SPM, through the shared helper and appends the
  `ModelSpec` nuisance matrix, so the selected-HRF task design is the scored
  task design. Without a task model it is unchanged.
- `fit(..., hrf_selection=selection)` with `model.task_model` set requires
  the selection's task model fingerprint to match and `model.oversampling`
  and `model.min_onset` to equal the values recorded in the selection
  provenance. With `model.task_model=None`, the selection must carry the
  default task-only model; this preserves today's Nilearn event path. The
  user guide states that such a selection scored only the mean stimulus
  response. A selection always carries a task model, defaulting to task-only.
- `select_hrf` and `evaluate_hrf_split` gain `task_model=TaskModel()`.
  `_ridge_cv` passes its default through unchanged; threading a task model
  into ridge CV is out of scope.

### Results

- `HrfSelectionResult` gains `task_model`.
- `HrfEvaluationResult.training_amplitudes` becomes shape (K, n_features)
  with a new `amplitude_names` tuple equal to the task regressor names.

### Provenance

Selection, evaluation, and selected-GLM activities record
`task_model=task_model.to_dict()` and `task_model_fingerprint`. The selection
`score` string becomes `nuisance_adjusted_task_model_prediction_r2`, and
`nuisance` becomes `conditional_projection_of_confounds_and_profiled_task_columns_per_run`.
The run design fingerprint incorporates modulator amplitudes, so any change
in RT values changes the design identity.

### NSD example

- Remove `glm_events` and the `glm=` flag from `load_block`. Every block loads
  raw events. The loader sets nonpositive RTs to NaN and validates that
  `trial_type` contains both codes 0 and 1 in every run.
- `glm_model` sets `task_model=NSD_TASK_MODEL`. `select_hrfs` and
  `session_hrf` pass the same object.
- `REGRESSORS` stays `("task", "response_time", "trial_type")`, so effect,
  variance, t, z, and R² surfaces and the multisession code keep their shapes
  and names.
- Provenance metadata `event_encoding` becomes `raw_trials_with_task_model`.
- Session HRF caches are invalidated because the request identity includes
  selection provenance; the notebook recomputes.
- `nsd_hrf.py` uses `trial_eligible(0)` where it previously used
  `eligible(0)`, since it guards single-trial fits.

### Documentation

- User guide: rewrite "Selecting an HRF for each location" to describe the
  task model, remove "RT never enters HRF selection", and describe the score.
  Update "Voxelwise HRFs in conventional GLMs" to state that the fitted design
  is the scored design when a task model is set.
- API reference: `TaskModel`, `Modulator`, the new `ModelSpec` field, the new
  `select_hrf` argument, and the changed result fields.

## Out of scope

- Threading a task model into ridge CV inner-fold selection.
- Derivative or dispersion HRF bases with a task model.
- Any change to single-trial or fractional ridge fitting.
- Making the task model mandatory for selected-HRF fits.

## Testing

Tests are written and committed before implementation, following the repo's
RED-GREEN-Refactor rule.

1. **Regression.** `TaskModel()` reproduces the current `select_hrf` indices,
   `cv_r2`, eligibility table, and design-fingerprint invariances (changing
   RT values does not change the fingerprint) on the existing fixtures, and
   `fit(..., hrf_selection=...)` without a task model is unchanged. The
   fingerprint payload itself now includes the expanded events and the
   task-model fingerprint, so its value differs from earlier releases even
   for the default model.
2. **Event expansion.** For a fixture with RT and trial type, the expanded
   frame has `task` with unit amplitudes, `response_time` centered over
   observed trials and zero where missing, `trial_type` uncentered, and
   `missing_response_time` present only in runs with a missing RT. Validation
   errors fire for reserved names, duplicate columns, non-finite values under
   `missing="error"`, and a regressor with no nonzero amplitude in a run.
3. **Design equality.** For every candidate and run, `RunDesign.task_design`
   equals the design matrix that `compile_group_designs` fits, column for
   column, and the canonical `fit` with a task model equals
   `make_first_level_design_matrix` on the expanded events with `"spm"`.
4. **Objective oracle.** With K > 1 and a run containing missing RTs, pooled
   leave-one-run-out loss equals Nilearn `run_glm` on a stacked training
   design with shared task columns and block-diagonal nuisance plus profiled
   columns, followed by explicit held-out prediction with per-run nuisance
   and profiled coefficients refit. Extends
   `test_pooled_loro_matches_stacked_training_ols`.
5. **Recovery.** Noise-free simulated data whose response is modulated by
   RT and trial type under a known library HRF, with a missing RT in some
   runs: the full task model recovers the generating HRF at every feature
   with a score of one, while the task-only model scores that HRF strictly
   below one because the modulated variance is unexplained.
6. **Eligibility.** A candidate whose task design is rank deficient in one run
   is `-inf` everywhere with a run-specific reason; a fold with a singular
   pooled training A excludes the candidate; the canonical row is NaN when
   SPM is ineligible.
7. **Guards.** Mismatched task models between `ModelSpec` and the selection
   raise; a task model with a derivative basis raises; results and provenance
   carry the task model and its fingerprint; `training_amplitudes` has shape
   (K, n_features) with matching `amplitude_names`.
8. **NSD example.** Existing example tests updated for raw-event loading; a
   test asserts that the selection design fingerprint for a block equals the
   GLM group design fingerprint for the same block and candidate.
