# Expanded HRF Selection Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` for direct
> implementation. Use subagent-driven execution only if the user selects it.
> Every implementation task follows RED–commit tests–GREEN–refactor.

**Goal:** Select a grayordinate-specific HRF from an expanded library using
mean-stimulus prediction across runs, then estimate unrestricted trial betas.

**Architecture:** Separate deterministic HRF generation, candidate convolution,
run-wise CV scoring, and grouped single-trial fitting. Reuse the existing
numerical kernel, provenance conventions, NSD inputs, and atomic publisher.

**Tech Stack:** Python >=3.12, uv, NumPy, pandas, SciPy, Nilearn, NiBabel,
matplotlib, pytest. Declare SciPy directly because the new generator imports it.

**Spec:** `docs/superpowers/specs/2026-09-26-hrf-selection-design.md`.

**Status:** Plan only. User chose mean-stimulus prediction for HRF selection;
implementation and full-data execution have not started.

## Global Constraints

- All local commands use `uv run`; dependency changes use uv.
- Every `__init__.py` stays empty. Use short modular functions and pytest fixtures.
- Commit failing tests before implementation; never weaken scientific oracles.
- Library = notebook's 648 code-grid candidates plus the exact current SPM baseline.
- HRFs use sum-normalized kernels, oversampling 50, and exact frame times.
- Whole runs are folds. No held-out task-amplitude refit during prediction scoring.
- Nuisance regressors remain unpenalized and include a run intercept.
- RT and image IDs do not enter selection. Ridge alpha stays fixed at 0 or 0.1.
- One all-run HRF map is shared by final OLS/ridge fits; trial amplitudes remain free.
- Numerical fitting performs no I/O. New artifacts preserve previous results.
- The best selection-CV score is not an unbiased test of selected-model performance.

## Review Focus

1. Held-out amplitudes must remain frozen: Task 3 tests arbitrary test-signal edits.
2. Library scale, candidate ordering, and frame offsets must not silently change
   models: Tasks 1–2 test normalization, identity, and .774/.775-second sampling.
3. Slow HRFs and late events can make trial models unidentifiable: Tasks 3–4
   test common structural eligibility, deterministic exclusions, and no dropped trials.
4. Spatially varying HRFs need distinct designs and correct feature mapping:
   Tasks 4–5 test interleaved HRF groups, immutable metadata, and exact CIFTI axes.
5. All-run HRF selection can contaminate an odd/even RT check: Task 5 tests that
   the independent even-run plots use odd-selected HRFs and canonical odd vertices.

## Task 1: Define and reproduce the expanded library

**Files:** Create `src/boldtailor/hrf_library.py`, `tests/test_hrf_library.py`;
update `pyproject.toml` and `uv.lock` only when implementation starts.

**Interfaces:** `expanded_hrf_library() -> HrfLibrary`;
`HrfLibrary.from_parameters(parameters) -> HrfLibrary` (always adds the SPM anchor);
`library.candidates` is an immutable ordered tuple of `HrfCandidate`;
each candidate has `id`, `kind`, `parameters`, and `kernel(tr, oversampling)`.
The library owns a parameter table, sampled curves, and a deterministic fingerprint.
Candidate 0 routes to the existing `"spm"` path; custom IDs follow the sorted
parameter product. Duplicate parameter rows are rejected.

- [ ] Write tests for 649 IDs, exact parameter values, finite nonzero curves,
  normalization, immutable ownership, bad parameters, and stable fingerprints.
  Test the custom generator against an independently expressed gamma formula
  and selected numerical values from the notebook, not a second call to itself.

```python
def test_expanded_library_includes_legacy_anchor():
    library = expanded_hrf_library()
    assert len(library.candidates) == 649
    assert library.candidates[0].kind == "spm"
    assert library.candidates[1].parameters == (3., 10., .5, .5, 2., 0., 36.)
    assert library.candidates[-1].parameters == (6., 16., 1.5, 2.5, 8., 2., 36.)
```

- [ ] Run `uv run pytest tests/test_hrf_library.py -q -W error`, observe missing
  functionality, and commit failing tests.
- [ ] Implement the double-gamma generator at `dt = tr / oversampling`:
  `gamma(t-onset, response_delay/response_dispersion, scale=response_dispersion)`
  minus the undershoot gamma divided by response/undershoot ratio; truncate at
  duration, divide by its discrete sum, reject invalid/nonfinite/zero-sum kernels.
  Add SciPy with `uv add scipy`. Keep notebook loading out of runtime code.
- [ ] Run the task tests, compare 0.1-second custom curves with the notebook,
  format, and commit implementation.

## Task 2: Convolve specified HRFs without changing the canonical path

**Files:** Create `src/boldtailor/_hrf_design.py`, `tests/test_hrf_design.py`;
modify `_single_trial_design.py`, `single_trial.py`, and their existing tests.

**Interfaces:** `stimulus_regressor(events, frame_times, candidate) -> ndarray`;
extend `compile_trial_run(..., *, hrf="spm")` and
`fit_single_trials(..., hrf="spm")` to accept one identified `HrfCandidate`.
The default returns the existing `SingleTrialResult` without API field changes.
Public fits record candidate parameters/fingerprint and sampled-design identity.

- [ ] Write tests comparing a summed-presentation regressor with the sum of
  separately convolved trials; compare custom kernels with direct discrete
  convolution plus interpolation. Cover variable durations, zero-duration
  impulses, sub-TR onsets, both NSD offsets, causal support, and run boundaries.
  Use an identity fixture with repeated image IDs and missing RT.

```python
def test_default_single_trial_design_is_unchanged(timing_fixture):
    events, times, confounds = timing_fixture
    implicit = compile_trial_run(events, times, confounds, "run-01")
    explicit = compile_trial_run(events, times, confounds, "run-01", hrf="spm")
    np.testing.assert_array_equal(implicit[0], explicit[0])
```

  Define `timing_fixture` in the test file with onsets `[8.13, 17.37]`,
  durations `[3., 1.2]`, `times=.774+1.6*np.arange(60)`, and one finite motion
  column. Also verify explicit candidate 0 equals this default.
- [ ] Run the new tests plus `tests/test_single_trial_design.py` and
  `tests/test_single_trial.py`; commit the new failing tests before code.
- [ ] Dispatch candidate 0 to `"spm"`, custom candidates to individual Nilearn
  callable HRFs. Supply one candidate per call to avoid cross-candidate
  orthogonalization. Preserve the existing event-window checks and IDs.
- [ ] Run all affected tests with `-W error`; commit the tested extension.

## Task 3: Select HRFs with run-wise prediction and independent evaluation

**Files:** Create `src/boldtailor/_hrf_cv.py`, `hrf_selection.py`,
`hrf_results.py`, and `tests/test_hrf_selection.py`.

**Interfaces:**

```python
select_hrf(data, *, library, run_labels=None, feature_signature=None,
           candidate_batch_size=32)
# -> HrfSelectionResult
evaluate_hrf_split(data, *, library, train_runs, test_runs, run_labels=None,
                   feature_signature=None)
# -> HrfEvaluationResult
```

Selection fields: `hrf_indices`, `cv_r2`, `canonical_cv_r2`, `delta_cv_r2`,
`library`, `eligibility`, `run_labels`, `feature_signature`, `provenance`.
Indices are stable library IDs; undefined features use -1 internally and
NaN in imaging exports. Evaluation fields: `training_selection`,
`training_amplitudes`, `test_r2`, `canonical_test_r2`, `delta_test_r2`,
`train_runs`, `test_runs`, `provenance`. Scores use the spec's nuisance-adjusted
denominator, not the single-trial full-model denominator.

- [ ] Define a fixture with four runs having different onsets/durations,
  nonidentical simulated trial amplitudes, known candidate HRFs per feature,
  run-specific nuisances and means, and seeded independent/AR noise. No image
  occurs in more than one run. The principal exact numerical oracle uses
  noise-free stimulus means plus nuisances, fitted independently by stacking
  training stimulus columns and block-diagonal nuisance columns with NumPy
  least squares. Hold the resulting task coefficient fixed when scoring test
  data after profiling only test nuisance coefficients.
- [ ] Test candidate scores/winners against that oracle; unequal run lengths
  and variance must expose accidental mean-of-R² pooling. Test constant and
  nuisance-only features, duplicate nuisance columns, insufficient runs,
  invalid folds, negative scores, candidate ties/permutations, and batch size.

```python
def test_outer_test_changes_cannot_select_the_hrf(cv_fixture):
    data, changed_test_data, library = cv_fixture
    original = evaluate_hrf_split(data, library=library,
                                 train_runs=[0, 2], test_runs=[1, 3])
    changed = evaluate_hrf_split(changed_test_data, library=library,
                                train_runs=[0, 2], test_runs=[1, 3])
    np.testing.assert_array_equal(original.training_selection.hrf_indices,
                                  changed.training_selection.hrf_indices)
    np.testing.assert_array_equal(original.training_amplitudes,
                                  changed.training_amplitudes)
```

  `changed_test_data` changes only the signals/RT in runs 1 and 3, retaining
  events' timing, confounds, and dimensions. A separate amplitude-shift test
  confirms worse held-out prediction, rather than a concealed task refit.
- [ ] Run `uv run pytest tests/test_hrf_selection.py -q -W error`, observe RED,
  and commit tests.
- [ ] Implement Q-span nuisance removal and `A/B/C` accumulation in candidate
  batches. Fit each training amplitude by `sum(B)/sum(A)` and calculate the
  spec's held-out loss. Implement independent-split evaluation by calling
  selection only on training signals and then freezing the chosen HRF/amplitude.
  Record labels and fold memberships; validate every index and disjoint split.
- [ ] Enforce structural eligibility across the relevant run designs. Start
  with lazy winner validation: retain the block's candidate score matrix,
  test each proposed winner's full trial design in all relevant runs, exclude
  any structurally invalid candidate globally, and reselect until all winners
  are valid. Cache outcomes by timing/confound/library fingerprint. For an
  outer split, eligibility may inspect test timing/confounds, never test BOLD.
  Reject if no candidate is estimable; never give an invalid candidate a
  zero-loss score or silently drop an event.
- [ ] Validate the canonical comparison candidate under the same structural
  rules. If it is ineligible, report its comparison score/difference as NaN
  with an eligibility reason. Always retain stable IDs after exclusions.
- [ ] Run tests, verify input/result immutability and provenance changes for
  library/timing/folds, and commit implementation. Keep RT/image metadata out
  of the selection fingerprint except existing source-file identity.

## Task 4: Fit and represent spatially varying single-trial HRFs

**Files:** Extend `single_trial.py` and `hrf_results.py`; create
`tests/test_selected_hrf_fit.py`. Reuse `_single_trial_fit.py` unchanged unless
a failing numerical regression demonstrates a required fix.

**Interface:** `fit_selected_hrfs(data, *, selection, ridge_alpha=0.0,
run_labels=None, feature_signature=None) -> HrfSingleTrialResult`. This exposes `run_betas`,
`trial_table`, `hrf_indices`, `group_designs`, run/pooled full and nuisance R²,
`delta_r2`, `diagnostics`, `selection_provenance`, and fit `provenance`.
`group_designs` owns matrices keyed by `(run_index, stable_hrf_id)`; selection
may come from other runs, but feature ordering and library identity must match.
If selection has an explicit spatial signature, require the same signature
here. With anonymous arrays, check feature count and document caller-owned
ordering; do not claim that raw signal values establish spatial identity.

- [ ] Test interleaved feature IDs `[h1, h0, h1, h2, undefined]`; independently
  reconstruct each feature's trial/nuisance matrix and compare OLS and fixed
  normalized ridge with NumPy/augmented least squares. Test trial row order,
  repeated image IDs, missing RT, per-run constants, identity mismatches,
  grouped-design immutability, and a canonical-only library matching legacy fits.

```python
# Numerical oracle for each selected feature; x/n come from independent convolution.
xr = x - n @ np.linalg.lstsq(n, x, rcond=None)[0]
penalty = np.column_stack([np.diag(np.sqrt(alpha)*np.linalg.norm(xr, axis=0)),
                          np.zeros((x.shape[1], n.shape[1]))])
matrix = np.column_stack([x, n])
expected = np.linalg.lstsq(np.vstack([matrix, penalty]),
                          np.r_[y, np.zeros(x.shape[1])], rcond=None)[0]
```

- [ ] Run `uv run pytest tests/test_selected_hrf_fit.py -q -W error`; commit RED.
- [ ] Group feature indices by chosen candidate, compile each needed run/design
  once, call the existing run solver, and scatter outputs back to original
  indices. Use identical chosen HRFs for OLS/ridge and preserve beta units.
  Fill constant/undefined locations with NaNs according to the existing rule.
- [ ] Aggregate sums of squares exactly as in the current API. Hash each
  grouped design and feature-to-HRF mapping; record fitting and selection as
  distinct provenance activities. Run affected package tests and commit GREEN.

## Task 5: NSD integration, held-out checks, and reproducible artifacts

**Files:** Extend `examples/NSD/nsd_single_trial.py`,
`single_trial_artifacts.py`, `README.md`; create
`examples/NSD/test_nsd_hrf_selection.py`.

**Interface:** Add `hrf_library="canonical"|"expanded"` to the example and
CLI `--hrf-library expanded`. Default remains canonical. Expanded mode uses
the same CIFTIs, source discovery, motion24, six retained combined aCompCor
components, cosines, NSS indicators, and exact run timing. For this session,
predeclare odd-run training and even-run independent evaluation.

- [ ] Extend real miniature-CIFTI fixtures with at least four runs, differing
  event timings, two known HRFs at interleaved vertices, and complete nuisance
  files. Test HRF-map axes/indices, beta trial axes, exact spatial ordering,
  grouped-design association, all statistics and metadata, block invariance,
  existing-output preservation, and publication rollback. Change even BOLD/RT
  and assert odd-trained HRFs and selected cortical vertices are unchanged.
- [ ] Run the new fixture tests and commit RED before runner/writer edits.
- [ ] Run all-run selection per feature block, fit selected OLS/ridge, and
  aggregate the existing beta/R² products. Do not fit every candidate's trial
  coefficients at every grayordinate. Store library tables/curves once and
  group designs once per `(run, candidate)` actually used: one compressed NPZ
  archive per run, storing each trial matrix as float64 and one shared nuisance
  matrix, frame times, and string-valued column/HRF IDs without object arrays.
  Verify loading with `allow_pickle=False` reconstructs every full design.
  Supply a feature signature hashing the ordered BrainModel axis and block
  indices to every selection/fit/evaluation call. Evaluate mean
  prediction with odd-trained HRFs/amplitudes on even runs.
- [ ] Require at least two training runs and one test run for independent
  evaluation. If an example has too few odd/even runs, complete ordinary
  all-run selection when possible and record why independent evaluation/plots
  are unavailable. Never silently substitute an in-sample score.
- [ ] For the independent RT scatter, retain canonical-OLS odd-run vertex
  selection and refit only the selected vertices' even-run trial betas using
  odd-selected HRFs. These free beta fits are for RT diagnostics, not the
  frozen-amplitude prediction score. All-run optimized RT maps are explicitly
  descriptive. Record which HRF map each diagnostic used.
- [ ] Publish new descriptors `hrfOptOLS`, `hrfOptRidge`, and `hrfSelection`:
  selected IDs/parameters/peak times; selected/canonical/delta selection-CV
  scores; odd-trained IDs and independent test scores; trial betas and existing
  full/nuisance/ΔR²; grouped designs, trial mapping, library and fold tables,
  RT diagnostics, and complete provenance. Preflight collisions before fitting.
  Distinguish selection scores from full-model in-sample R² in filenames/metadata.
- [ ] Run package and NSD tests and formatting checks. Benchmark 10 candidates
  on two runs, then the complete library on a small fixed grayordinate sample.
  Measure convolution, eligibility, scoring, final fits, and peak memory
  separately; report estimates before the full expanded-library run. Start
  with candidate batches of 32 and feature blocks of 4096; reduce either to
  bound memory without changing the model. Optimize repeated design work only
  if profiling justifies it and equivalence tests protect it.
- [ ] Run all 12 runs with the predeclared 649 candidates and ridge 0.1. Audit
  at least 25 grayordinates against independent CV and grouped-fit calculations;
  verify fold isolation, exact CIFTI axes, all 750 trials, and prior-file hashes.
  Report held-out improvement or deterioration and RT checks without choosing
  a library or penalty from the even-run results. Visually inspect the saved
  HRF library and selected-vertex curves/scatters. Obtain one final code review.

## Verification commands

```bash
uv run pytest tests examples/NSD -q -W error
uv run black --check src tests examples/NSD
uv run git diff --check
uv run python examples/NSD/nsd_single_trial.py --hrf-library expanded --ridge-alpha 0.1
```

Use a new output directory for reruns; the first expanded run uses new names
under `/Volumes/extdata1/NSD/BIDS/derivatives/boldtailor`.

## Self-review and handoff

The five review concerns each have assigned tests. Library IDs, result fields,
and grouped-design mappings have explicit owners. The mean-prediction target
matches the user's answer. Independent evaluation is one predeclared split,
not an additional optimization criterion. RT and ridge are kept out of HRF
selection, and no repeated-stimulus assumption is introduced.

Recommended execution: direct implementation in this session with one final
independent review. Review this plan before starting implementation.
