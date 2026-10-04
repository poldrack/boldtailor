# Task-guided denoising implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this
> plan task by task. Steps use checkboxes. This document authorizes no
> implementation by itself; the present request is to write the plan.

**Goal:** Add an opt-in GLMsingle-style denoising stage that identifies a
noise pool using task-model time-series CV R², extracts run-specific temporal
PCs, and selects one PC count by held-out task prediction.

**Architecture:** Reuse the existing task design and HRF-selection machinery.
Keep noise-pool construction, PCA, component-count scoring, and result
assembly in small modules. Return nuisance regressors that can be appended
to `AnalysisData` and consumed by the existing fitting APIs.

**Tech stack:** Existing Python >=3.12, NumPy, pandas, SciPy, Nilearn, uv, pytest;
no new dependencies.

**Design basis:** The denoising discussion in this conversation and the
bounded design below. This is a proposed implementation, not a description
of an already validated method.

## Scope and scientific contract

Implement a single-pass, task-guided noise pool and a single component count
per input analysis. The pool uses the time-series task model, including any
explicit RT/condition modulators, not the trial-beta encoding score. Low
score means weak prediction by this specified model, not absence of neural
activity.

Initial HRFs are selected with baseline confounds only. Freeze those HRFs
and the pool while comparing PC counts within a fold. After selecting the
count, callers may rerun ordinary HRF selection and fitting on augmented
data. This is sequential tuning, not joint optimization.

Excluded: iterative pool refinement, voxelwise PC counts, automatic R²
threshold fitting, ridge tuning, new anatomical segmentation, aCompCor
selection, automatic notebook/default changes, spatial smoothing, new
parallel backends, and claims of equivalence or superiority to GLMsingle.

## Global constraints

- Use uv for package management and `uv run` for Python and pytest commands.
- Every `__init__.py` remains completely empty.
- Prefer short functions, pytest functions, and fixtures for reusable data.
- Write failing tests, verify RED, and commit those tests before implementation
  for each task. Then implement, verify GREEN, refactor, and commit production code.
- Preserve existing fit/selection behavior unless the new API is invoked.
- Preserve the user's existing files and uncommitted changes.
- Reuse existing convolution, task expansion, rank tolerances, result ownership,
  and provenance conventions. Do not introduce an alternate HRF generator.

## Proposed public interface

In `src/boldtailor/denoising.py`:

```python
def select_denoising(
    data: AnalysisData,
    *,
    brain_mask: np.ndarray,
    task_model: TaskModel = TaskModel(),
    library: HrfLibrary | None = None,
    counts: tuple[int, ...] = (0, 1, 2, 4, 6, 8, 10),
    pool_r2_threshold: float = 0.0,
    score_tolerance: float = 0.001,
    feature_signature: str | None = None,
) -> DenoisingResult:
    ...

def with_denoising(data: AnalysisData, result: DenoisingResult) -> AnalysisData:
    ...
```

`library=None` resolves to the existing default library. Require at least
three runs: each count-selection fold leaves at least two training runs for
initial HRF selection. `brain_mask` is a required, caller-supplied Boolean
mask in feature order; do not infer anatomy from arbitrary array intensities.

`DenoisingResult` in `denoising_results.py` owns immutable array fields and
defensive copies of tables. It contains `n_components`, final `noise_pool`,
final initial-selection R², `run_components` (time × selected count), a
candidate/fold score table, fold pool/scoring masks and HRF assignments,
eligibility reasons, run labels, feature signature, and provenance. Include
per-run effective PCA ranks and singular values in diagnostics. These are
selection diagnostics, not independent performance estimates.

`with_denoising` validates source-analysis identity, ordered runs, time grids,
feature signature, and row counts before appending `denoise_pc_000`, etc.
Reject name collisions and applying a result twice. Preserve original data,
events, baseline confounds, source records, and timing; extend provenance.
For a zero-count result, preserve all numerical inputs exactly and record
that no PCs were added.

## Exact selection procedure

For each omitted run:

1. Subset training runs and call `select_hrfs` with baseline confounds and
   the specified task model/library. Never pass validation signals into this
   initial selection. Use the training selection's winning CV R² as a
   screening statistic; it is not an unbiased performance estimate.
2. Define the pool as `brain_mask & finite(score) & (score <= threshold)`.
   Define the scoring mask as `brain_mask & finite(score) & (score > threshold)`.
   Exclude undefined HRF assignments. Both masks are fixed across counts and
   disjoint. Record their sizes; do not use validation task fit to choose them.
3. For each training run, project pool time series off baseline confounds
   plus the intercept, discard numerically zero columns, normalize remaining
   columns to unit L2 norm, and compute temporal PCs by SVD. Use a consistent
   sign convention (largest absolute temporal entry positive). Count selection
   does not need validation-run PCs.
4. For each count, fit shared task coefficients across training runs with
   run-specific baseline nuisance coefficients, missing-value indicators, and
   PC coefficients. Project BOTH the task design and BOLD off each run's
   nuisance span. Freeze the initial voxelwise HRF assignment across counts.
5. Predict the held-out task time series using those shared coefficients.
   The scoring projection contains only the fixed baseline nuisance set and,
   where applicable, missing-value indicators convolved with the frozen HRF.
   It must never contain the candidate PCs. The projected target, design,
   denominator, and scored features must be identical across counts.
6. Compute voxelwise R² against that fixed target and average over the fold's
   scoring mask. Remove numerically zero-energy validation targets using the
   same mask for every count; record exclusions. Average the fold means with
   equal fold weights. Report per-fold values as well as the aggregate.
7. Select the smallest eligible count within `score_tolerance` of the maximum
   aggregate R². The default 0.001 is an explicit practical tolerance in R²
   units, not a significance threshold or standard-error estimate. Zero is
   always a candidate; exact ties select fewer PCs.

For a feature's fixed held-out projection `M`, the scoring oracle is:

```python
observed = M @ y_test
predicted = M @ X_test @ beta_train
score = 1.0 - ((observed - predicted) ** 2).sum() / (observed ** 2).sum()
```

Use a basis projection in production rather than constructing dense `M`.
Missing-value indicator coefficients are profiled using held-out BOLD, as in
the existing task model. This is conditional task prediction; document that
qualification. It does not vary with PC count because HRFs are frozen.

After count selection, construct a new initial selection and pool using all
provided runs, derive PCs for each run, and return the selected prefix.
Do not iterate the pool after adding PCs. This full-data result is for final
fitting, not for evaluating generalization.

## Eligibility and failure behavior

- Validate finite thresholds/tolerances, nonnegative tolerance, a correctly
  shaped Boolean mask, and integer counts (reject Boolean counts). Require
  zero, sort/deduplicate counts, and reject negative values.
- Empty pools or zero-rank pool signals make positive counts unavailable;
  retain zero and report the reason. Never substitute gray matter or aCompCor.
- Empty scoring masks or no nonzero validation targets raise a clear error:
  there is no supported task signal for this selection objective.
- A positive count is eligible only if every training run supports that many
  PCs and projected task designs remain full rank with positive residual
  degrees of freedom in every fold. Do not silently cap a count per run.
- Compare all counts on the same features within each fold. A candidate that
  invalidates one required task design is unavailable, rather than scored on
  a favorable subset. If zero is invalid, raise rather than fabricate scores.
- If the final full-data pool cannot support the chosen count, raise with its
  size/rank and the selected count. Do not silently change the selected result.
- PCA within an exactly degenerate singular-value block has no unique basis.
  Test subspace equivalence; report a cutoff that splits a numerically tied
  block as unavailable rather than depending on an arbitrary rotation.

## File map

| File | Responsibility |
| --- | --- |
| `src/boldtailor/denoising.py` | Public selection and augmentation APIs |
| `src/boldtailor/_denoising_pool.py` | Masks, baseline projection, normalized PCA |
| `src/boldtailor/_denoising_cv.py` | Training-only fold setup, count scores, choice |
| `src/boldtailor/denoising_results.py` | Owned results and diagnostics |
| `tests/test_denoising_pool.py` | Pool and subspace/rank tests |
| `tests/test_denoising_cv.py` | Independent numerical oracle and leakage tests |
| `tests/test_denoising.py` | Public contract, provenance, fit compatibility |
| `tests/denoising_fixtures.py` | Seeded multi-run synthetic fixtures |
| `docs/user-guide.md`, `docs/api.md`, `docs/glmsingle-comparison.md` | Usage, limits, comparison |

Read `hrf_selection.py`, `_hrf_cv.py`, `_task_design.py`, `_single_trial_fit.py`,
`data.py`, `hrf_results.py`, and `provenance.py` before implementation. Keep
existing modules unchanged unless a small shared helper is genuinely needed.

## Review focus

1. Strong task responses with unpredictable trial-beta variation must not be
   treated as noise merely because beta encoding fails (Task 1).
2. Constant signals, duplicate nuisances, short runs, and rank loss must not
   generate spurious PCs or silently change the comparison population (Tasks 1–2).
3. Changing validation signals must not change training HRFs, masks, PCs, or
   coefficients; validation scores are expected to change (Task 2).
4. More nuisance regressors must not win by changing the validation target
   or denominator (Task 2).
5. Reordered features/runs and mutable returned tables must not corrupt an
   existing analysis or allow an incompatible augmentation (Task 3).

## Task 1: Noise pool and PCA

- [ ] Add `tests/denoising_fixtures.py` with seeded, unequal-length runs,
  known task responses, a shared non-task latent time series, pool voxels,
  baseline motion/drift, and constant features. Include a strong mean task
  response whose trial amplitudes are unrelated to RT.
- [ ] Write tests in `test_denoising_pool.py` for threshold boundaries,
  nonfinite scores, masks, empty pools, and exclusion of task-responsive
  features based on time-series scores. Add a hand-built rank-two matrix
  oracle: recovered PCs span the normalized projected pool, are orthonormal,
  and are orthogonal to baseline nuisance regressors. Cover tied cutoffs.
- [ ] Run `uv run pytest tests/test_denoising_pool.py -q -W error`; verify
  failure due to missing behavior and commit the failing tests alone.
- [ ] Implement small pool/PCA helpers in `_denoising_pool.py`. Use reduced
  SVD, numerical rank tolerance consistent with existing code, and owned
  outputs. Pool construction consumes training `HrfSelectionResult` scores.
- [ ] Rerun the focused tests, refactor, and commit implementation.

## Task 2: Component-count cross-validation

- [ ] Write `test_denoising_cv.py` using independent stacked least squares:
  construct a shared task block plus block-diagonal run nuisance matrices,
  solve with `np.linalg.lstsq`, and score with the equation above. Do not
  derive expected answers by calling production CV helpers.
- [ ] Cover counts zero and positive, RT modulators, missing-RT indicators,
  unequal run lengths, baseline-confound collinearity, infeasible counts,
  fixed validation denominators, and fixed within-fold scored masks.
- [ ] Perturb only held-out signals and verify unchanged training choices
  and coefficients. Include adversarial PCs aligned with task signal: they
  must not gain by being projected out of the validation target.
- [ ] Pin conservative choice with a concrete selector test:

  ```python
  # Sorted eligible counts and their aggregate R² values.
  counts = (0, 1, 2, 4)
  scores = (0.10, 0.20, 0.2005, 0.19)
  # tolerance=0.001 -> choose 1; tolerance=0 -> choose 2.
  # All equal -> choose 0. Unavailable counts cannot win.
  ```

- [ ] Run `uv run pytest tests/test_denoising_cv.py -q -W error`, verify RED,
  and commit failing tests before adding `_denoising_cv.py`.
- [ ] Implement fold preparation, projected pooled task fitting, fixed-target
  prediction, eligibility, and count choice as separate functions. Cache
  designs/PCA within a fold, not globally across training partitions.
- [ ] Rerun both denoising test files, refactor, and commit implementation.

## Task 3: Public result, augmentation, and final fitting

- [ ] Write `test_denoising.py` for public input validation, final pool
  reconstruction, zero-count behavior, insufficient final rank, immutability,
  provenance, duplicate application, column collisions, and mismatched inputs.
- [ ] Add a downstream integration test comparing a conventional fit on
  augmented data against the same model with the expected PCs explicitly
  appended. Verify the original `AnalysisData` remains numerically unchanged.
- [ ] Run `uv run pytest tests/test_denoising.py -q -W error`, verify RED,
  and commit tests before implementing the public API/result.
- [ ] Implement `DenoisingResult`, `select_denoising`, and `with_denoising`.
  Record task-model/library fingerprints, baseline-confound identity, masks,
  count grid, threshold, tolerance, fold run labels, reasons for exclusions,
  initial HRFs, PC basis fingerprints, and `selection_statistic=True`.
- [ ] Run all three new test files; refactor and commit implementation.

## Task 4: Scientific validation and documentation

- [ ] Add integration tests for recovery in a fixed-seed synthetic dataset with
  shared noise, plus a no-benefit dataset where zero wins or ties.
  Use known task coefficients and a genuinely untouched outer run to check
  recovery; do not use the winning inner-CV score as evidence of improvement.
  Predeclare seeds and effect sizes, and do not tune them to force success.
  If these validation tests expose a defect, verify and commit the failing
  regression before correcting production code. If they already pass, record
  that result without manufacturing a failure or changing production code.
- [ ] Implement only corrections exposed by these tests; rerun focused tests.
- [ ] Document the minimal usage below, the sequential tuning order, the
  required mask, missing-value semantics, empty-pool behavior, and diagnostic
  score interpretation. Update the comparison to state that this is an
  optional task-guided pool, with a different scoring criterion from GLMsingle.

  ```python
  from dataclasses import replace
  from boldtailor.denoising import select_denoising, with_denoising
  from boldtailor.fit import fit
  from boldtailor.hrf_selection import select_hrfs

  denoising = select_denoising(
      data, brain_mask=brain_mask, task_model=task_model, library=library
  )
  augmented = with_denoising(data, denoising)
  selection = select_hrfs(augmented, library=library, task_model=task_model)
  fitted_model = replace(
      model, confounds=tuple(augmented.confounds[0].columns)
  )
  result = fit(augmented, fitted_model, hrf_selection=selection)
  ```

  This example assumes matching baseline confound names across runs and a
  `model.task_model` compatible with the selection. Do not impose matching
  run-specific nuisance columns on the lower-level selection implementation.

- [ ] Explain independent evaluation: call selection on outer-training runs
  only; freeze its count and pool before applying any denoising to an outer
  test run. Task-only validation as specified above needs no test-run PCs.
  A later evaluation of denoised test-run betas can compute run-local PCs from
  the frozen pool, but must not reselect the pool/count using those outcomes.
- [ ] Run `uv run pytest -q -W error` and `git diff --check`. Record actual
  results and any limitations, then commit implementation/documentation.

## Acceptance criteria

- A caller can request denoising, inspect the fixed pool and candidate scores,
  append selected PCs, and use existing conventional or single-trial fits.
- Count selection matches the independent oracle and never changes its
  validation target across counts. Fold training choices are isolated from
  validation BOLD.
- Empty/rank-deficient inputs have the explicit behavior above. No silent
  anatomical fallback or per-run component truncation occurs.
- Original inputs and existing default APIs retain their behavior.
- Synthetic recovery and regression checks pass; their results establish
  implementation behavior, not empirical superiority to GLMsingle.

## Primary methodological references

- [GLMsingle source: ON–OFF R² pool, runwise PCA, component-count selection](https://github.com/cvnlab/GLMsingle/blob/main/glmsingle/glmsingle.py)
- [GLMdenoise: task-prediction-based denoising](https://pmc.ncbi.nlm.nih.gov/articles/PMC3865440/)

The zero pool threshold, practical score tolerance, and sequential frozen-HRF
search are explicit proposed choices for this bounded implementation. Their
empirical sensitivity should be assessed before recommending universal defaults.
