# Within-run encoding and run-specific intercepts

**Status:** Implemented and verified on `feat/within-run-encoding`, 2026-09-28.
Full suite: 837 passed; four additional independent-review coverage tests passed.
See `docs/validation/within-run-encoding-2026-09-28.md` for scientific results.

**Goal:** Make ridge and fractional-ridge tuning measure prediction of within-run trial variation, with shared encoding slopes estimated after accounting for separate training-run intercepts.

**Design basis:** User request of 2026-09-28 and `docs/validation/ridge-objectives-2026-09-27.md`. The complete proposed statistical and interface contract is below; review this document before execution.

**Architecture:** Keep trial-beta estimation unchanged. Implement the encoding model and scoring in `trial_encoding.py`, pass one explicit encoding mode through both candidate scorers and the NSD workflow, and record its semantics in results, exports, and provenance.

**Tech stack:** Existing NumPy, pandas, pytest, and uv; no new dependencies.

## Constraints

- Use clean, modular code and short functions. Every `__init__.py` stays empty.
- Use `uv run` for Python, tests, and formatting.
- For each implementation task: write tests, demonstrate a meaningful RED failure, commit those tests, implement, demonstrate GREEN, refactor, and commit implementation.
- Preserve legacy tests under an explicit legacy mode. Changes to expectations must cite the changed statistical requirement, not accommodate implementation output.
- The current checkout has extensive tracked and untracked work, including result types and validation code used here. Record its baseline before execution; do not reset, discard, or broadly stage it. A clean-HEAD worktree alone would omit required changes. Reconcile the working snapshot before choosing isolation.
- No claims that centering resolves candidate-dependent target distortion or guarantees latent-beta recovery.

## Proposed statistical and API contract

### Modes and alternatives

Add keyword-only `encoding_mode="within_run"` to:

```python
evaluate_trial_encoding(beta_runs, predictors, *, train_runs, test_runs,
                        encoding_mode="within_run")
score_ridge_candidates(data, predictors, *, alphas, library=None,
                       run_labels=None, feature_signature=None,
                       encoding_mode="within_run")
score_fraction_candidates(data, predictors, *, fractions, library=None,
                          run_labels=None, feature_signature=None,
                          encoding_mode="within_run")
```

Also add it to `examples/NSD/ridge_workflow.py:fit_cv_beta_series` and thread it through inner tuning and outer evaluation. Validate it before expensive HRF/beta fitting. Accept exactly `"within_run"` and `"absolute"`; unknown values raise `ValueError` naming the argument and choices.

- **Recommended/default `within_run`:** run-specific training intercepts, shared slopes, and centered residual scoring per held-out run and feature.
- **Explicit `absolute`:** preserve the current pooled training centering, shared intercept, predictions, and uncentered residual scoring. This provides reproduction and a clearly different scientific estimand.

One mode couples the two scientifically consistent behaviors and avoids expanding the public API to four combinations. The simulation can still evaluate shared-intercept/centered-scoring as an ablation using `absolute` predictions and its independent scoring helper. Merely changing validation scoring would leave training slopes vulnerable to between-run confounding; removing the old behavior entirely would make historical reproduction harder.

### Training model

For each training run r, compute means over rows with complete finite predictors:

```
Xc_r = X_r - mean(X_r)
Yc_r = Y_r - mean(Y_r)
B = lstsq(vstack(Xc_r), vstack(Yc_r))
a_r = mean(Y_r) - mean(X_r) @ B
```

This is OLS `Y_r = a_r + X_r @ B + error`, with one unpenalized intercept per training run and shared slopes. Do not ridge-penalize the encoding model. Every complete trial has equal weight; do not give equal weight to runs of different lengths.

Require at least one complete predictor row in every training run, full column rank of the stacked within-run centered predictors, and positive residual degrees of freedom `sum(n_r) > n_training_runs + n_predictors`. A one-row training run can contribute an intercept but no slope information. A predictor constant within each run is unidentifiable even if its values differ between runs; reject with a clear rank-deficiency error. Keep absolute-mode validation behavior unchanged.

Maintain existing feature validity: a nonfinite training beta on an included row invalidates that feature; a nonfinite test beta invalidates that feature's score for that run. Nonfinite predictors exclude rows from encoding only. Excluded output rows remain NaN, and stimulus rows remain in beta estimation.

### Predictions, scoring, and result fields

Preserve existing `coefficients` shape `(1 + p, n_features)`, `predictor_names=("task", *columns)`, and pooled training `predictor_means` shape `(p,)`. In within-run mode define:

```
mu = pooled training predictor mean
c = pooled training beta mean
coefficients = vstack([c, B])
prediction_r = c + (X_r - mu) @ B
```

Here `task` is a **training reference level**, not the fitted intercept of every run. The raw-coordinate training intercepts `a_r` are recorded separately. The reference prediction is an arbitrary training-derived baseline for within-run scoring, not an estimate of a new run's unknown intercept. This choice preserves array shapes and ensures changing test betas cannot change returned predictions.

For each test run and valid feature:

```
residual = observed - prediction
offset = residual.mean(axis=0)
SSE_r = sum((residual - offset)**2, axis=0)
SST_r = sum((observed - observed.mean(axis=0))**2, axis=0)
R2 = 1 - sum(SSE_r, axis=0) / sum(SST_r, axis=0)
```

Do not standardize targets, normalize slope amplitudes, average run R² values, or substitute correlation squared. Wrong slopes must still reduce scores. Retain negative R² and existing undefined/zero-SST behavior. Scores are descriptive pooled within-run prediction scores; do not add a degrees-of-freedom correction to SSE or SST.

Append fields to `TrialEncodingResult`:

- `encoding_mode: str`.
- `train_run_predictor_means`: `(len(train_runs), p)` in `train_runs` order.
- `train_run_intercepts`: `(len(train_runs), n_features)`, raw-coordinate `a_r`. In absolute mode repeat the shared raw-coordinate intercept `c - mu @ B`.
- `scoring_offsets`: `(len(test_runs), n_features)` in `test_runs` order. Zero for valid absolute-mode features; NaN where scores are invalid.

Use owned immutable arrays, matching existing results. Preserve old positional field ordering, but require the new fields in internal keyword construction rather than silently fabricating metadata; document that direct dataclass construction changes. Returned `predictions` never include `scoring_offsets`. Users can reproduce centered SSE from targets, predictions, masks, and offsets.

### CV, provenance, and exports

Keep candidate-specific validation beta targets, nested training-only HRF selection, alpha/fraction grids, ties, feature eligibility, percentile aggregation, and fractional alpha conversion unchanged. The new encoding mode applies at both inner and outer levels.

Add `encoding_mode` and `encoding_objective_version=2` to scoring provenance and analysis fingerprints. Within-run provenance must state:

```
score = pooled_within_run_centered_trial_encoding_r2
encoding_model = ols_with_run_specific_intercepts
predictor_transform = center_within_each_complete_training_run
validation_intercept = remove_per_run_feature_mean_residual_for_scoring_only
prediction_reference = pooled_training_beta_mean_at_pooled_predictor_mean
```

Absolute mode retains its existing descriptive strings and explicitly records no validation-intercept adjustment. Fold records include ordered training indices/labels and training-run predictor means. Do not store large candidate-by-feature intercept arrays in JSON provenance; outer numerical exports carry fitted intercepts and scoring offsets.

NSD block aggregation must assemble intercepts and offsets across feature blocks. Export them with run labels in the loss/encoding NPZ and explain the prediction reference and scoring rule in metadata. Existing prediction images remain raw training-reference predictions. Update coefficient-map labeling/metadata so `task` is not described as a common fitted intercept in within-run mode. Both alpha and fractional workflows use the same mode throughout.

## Review focus

1. Between-run-only predictors and folds that lose within-run rank: explicit rejection, including candidate-scorer preflight (Tasks 1–2).
2. Unequal run sizes, missing predictors, and single-row/empty training runs: correct weighting, masks, and residual degrees of freedom (Task 1).
3. Test-outcome contamination: changing test betas changes offsets/scores but never training fits or raw predictions; offsets are never fed back into fitting (Tasks 1–2).
4. Multi-feature NaNs, zero SST, and block assembly: no silent NaN dropping, preserved undefined features and immutable outputs (Tasks 1–3).
5. Historical simulations silently inheriting new defaults: explicit legacy calls and preserved archived artifacts (Task 4).

## Execution tasks

### Task 1: Implement and test the encoding contract

**Files:** `src/boldtailor/trial_encoding.py`, `src/boldtailor/ridge_results.py`, `tests/test_trial_encoding.py`.

**Interfaces:** Produces the evaluator signature and result fields above. Keep `_training_design`'s current absolute path available; introduce small helpers for within-run design preparation, fitting, and residual scoring. The CV preflight in Task 2 must call the same mode-aware design validator used by the evaluator.

- [x] Record baseline: `uv run pytest tests/test_trial_encoding.py tests/test_ridge_cv.py tests/test_fractional_cv.py tests/test_ridge_objective_simulation.py -q -W error`. Record pre-existing failures separately.
- [x] Preserve existing shared-intercept oracle tests by explicitly passing `encoding_mode="absolute"`; retain their exact mathematical expectations.
- [x] Write a deterministic fixture with unequal lengths, shifted predictor distributions, distinct run offsets correlated with predictor means, and at least two features. Compare within-run slopes/intercepts to an **independent run-dummy OLS oracle**, not the production demeaning helper:

```python
xs = [predictors[r].to_numpy()[masks[r]] for r in train]
ys = [betas[r][masks[r]] for r in train]
run_ids = np.repeat(np.arange(len(train)), [len(x) for x in xs])
dummies = np.eye(len(train))[run_ids]
oracle = np.linalg.lstsq(
    np.column_stack([dummies, np.vstack(xs)]), np.vstack(ys), rcond=None
)[0]
np.testing.assert_allclose(result.train_run_intercepts, oracle[:len(train)])
np.testing.assert_allclose(result.coefficients[1:], oracle[len(train):])
```

- [x] Add an exact offset/slopes test. For `x=[-1,0,1,2]`, training `y=2+3*x` and `y=22+3*x`, held-out `y=102+3*x` gives slope 3 and centered R² 1. Replace held-out slope with 4: R² must equal `15/16`, despite the large offset. Verify raw predictions remain unchanged and SSE reconstructs from `scoring_offsets`.
- [x] Independently add constants to each training beta run: slopes and within-run scores stay invariant, run intercepts shift accordingly. Add a constant to a held-out beta run: coefficients and raw predictions stay invariant, its scoring offset shifts, and centered SSE stays invariant. Shift both training X and Y consistently to verify intercept coordinate definitions.
- [x] Test pooled SSE/SST against manual arithmetic on unequal held-out runs and explicitly distinguish from averaging R². Test slope error large enough to give negative R² and constant-target undefined scores.
- [x] Parameterize masks, run ordering, a feature with NaNs, immutable new arrays, test covariate changes, invalid mode, one training run, an empty training run, a single-row training run, insufficient residual degrees of freedom, and a between-run-only predictor. Include rank loss after masking.
- [x] Run `uv run pytest tests/test_trial_encoding.py -q -W error`; confirm failures are the new missing contract/behavior. Commit only Task 1 tests before implementation.
- [x] Implement the contract with keyword result construction, small model/scoring helpers, and mode-specific validation. Keep training and test statistics visibly separate.
- [x] Run the focused tests to GREEN, refactor without changing expectations, and commit implementation.

### Task 2: Integrate alpha and fraction CV with independent oracles

**Files:** `src/boldtailor/_ridge_cv.py`, `src/boldtailor/ridge_selection.py`, `src/boldtailor/fractional_ridge.py`, `tests/test_ridge_cv.py`, `tests/test_fractional_cv.py`.

**Interfaces:** Both public scorers gain `encoding_mode`; the shared scorer, `_validate`, `_score_fold`, and `_provenance` receive it explicitly. Existing positional private arguments remain in place; add keyword-only mode propagation.

- [x] Parameterize the existing independent alpha/fraction reference tests over both modes and canonical/selected HRFs. For within-run mode replace the reference shared-intercept fit with run-dummy OLS, then compute test predictions as `Xtest @ slopes` and center their residuals directly. A reference baseline is unnecessary because residual centering removes it. Continue deriving beta targets from the existing independent augmented-OLS/fractional oracles.
- [x] Preserve the legacy reference branch exactly. Explain oracle changes as new requirements. Add assertions that default calls match explicit within-run calls, all per-fold SSE/SST and pooled scores match the reference, and targets retain the candidate penalty/fraction.
- [x] Extend held-out contamination tests to training-run means/intercepts, unchanged HRF assignments, and raw predictions. Verify invalid modes and within-fold rank failures are raised before `_fold_selection` or beta fitting via a monkeypatched sentinel that fails if reached.
- [x] Assert both modes' provenance strings, ordered fold means, and different analysis fingerprints for different modes. Preserve selection masks, tie behavior, and boundary diagnostic tests.
- [x] Run `uv run pytest tests/test_ridge_cv.py tests/test_fractional_cv.py -q -W error`, inspect RED, and commit tests.
- [x] Thread the mode through shared CV, reuse the model-specific validator, and update provenance. Do not refactor HRF or ridge-beta estimation.
- [x] Run those tests plus `tests/test_trial_encoding.py` and `tests/test_selection_boundaries.py` to GREEN; commit implementation.

### Task 3: Propagate semantics through NSD outer evaluation and exports

**Files:** `examples/NSD/ridge_workflow.py`, `examples/NSD/ridge_outputs.py`, `examples/NSD/ridge_provenance.py`, `examples/NSD/fractional_outputs.py` as needed; `examples/NSD/test_ridge_workflow.py`, `examples/NSD/test_fractional_workflow.py`, `examples/NSD/test_ridge_outputs.py`, `examples/NSD/test_ridge_provenance.py` (create if absent).

**Interfaces:** `fit_cv_beta_series(..., encoding_mode="within_run")` propagates into every tuning/evaluation block. Outer dictionaries add `encoding_mode`, `train_run_predictor_means`, `train_run_intercepts`, and `scoring_offsets` with the shapes above.

- [x] Add end-to-end small-fixture tests for each regularizer and mode. Assert inner scoring provenance and outer evaluation use the same mode. Compare outer fits to an independent run-dummy oracle for within-run mode, retaining explicit absolute-mode reproduction.
- [x] Test feature-block invariance with a block boundary through valid/invalid features. Verify intercept/offset columns, training/test run ordering, masks, and raw predictions agree for different block sizes.
- [x] Test exported NPZ intercepts and offsets plus run-label metadata. Reconstruct centered SSE from exported targets, predictions, masks, and offsets; compare to stored SSE. Absolute mode must reconstruct the old SSE with zero valid offsets.
- [x] Assert exported interpretation and coefficient metadata distinguish training-reference predictions from within-run scores. Require mode-aware objective descriptions in tuning/final provenance.
- [x] Run `uv run pytest examples/NSD/test_ridge_workflow.py examples/NSD/test_fractional_workflow.py examples/NSD/test_ridge_outputs.py examples/NSD/test_ridge_provenance.py -q -W error`, inspect RED, and commit tests.
- [x] Implement explicit mode plumbing, feature-block field assembly, NPZ exports, and metadata. Use existing immutable result fields; do not silently add held-out offsets to prediction images.
- [x] Run the focused workflow/export tests to GREEN; commit implementation.

### Task 4: Extend the scientific comparison without relabeling history

**Files:** `examples/validation/ridge_objective_simulation.py`, `tests/test_ridge_objective_simulation.py`; new `docs/validation/within-run-encoding-2026-09-28.md` and adjacent result archive.

**Interfaces:** Retain `current`, `centered`, and `fixed_ols` as historical experimental identifiers, pinned to `encoding_mode="absolute"` for their fits. Add `within_run` using the new default model and centered scoring. Explain that `current` denotes the September 27 baseline, not the new production default.

- [x] Test that historical objectives explicitly use absolute-mode coefficients and preserve their original scores. Add a fixture where predictor means and run offsets covary; shared-intercept slopes must differ from the known truth, while the new noiseless within-run fit recovers it.
- [x] Update exact output cardinality tests for four objectives: eight selected rows and 48 candidate rows per scenario. Preserve two untouched outer runs and training-only candidate selection. Add independent checks for within-run beta RMSE/amplitude summaries, rather than evaluating only whole-beta amplitude dominated by run offsets.
- [x] Test the large-alpha matched-filter limit in both modes; keep the original absolute-mode check. Do not impose a stochastic requirement that every scenario must improve or select an interior penalty.
- [x] Run `uv run pytest tests/test_ridge_objective_simulation.py -q -W error`, inspect RED, and commit tests before changing the simulation.
- [x] Add the fourth objective, explicit legacy fits, and within-run-centered recovery metrics. For centered amplitude report projection of run-centered estimates onto run-centered latent betas; undefined zero-norm truth stays undefined.
- [x] Run tests to GREEN, then `uv run python examples/validation/ridge_objective_simulation.py --seeds 10 --output-dir /private/tmp/boldtailor-within-run-encoding-2026-09-28`.
- [x] Archive settings and summaries in a new location; preserve the September 27 CSVs and conclusions as historical evidence. Report boundary rates, whole-beta and within-run RMSE/amplitude, correlations, and across-seed uncertainty. Include a separate predictor-mean/offset-confounding stress experiment with reproducible settings. Numerical acceptance is oracle agreement and offset invariance, not a preselected empirical improvement threshold.

### Task 5: Documentation, migration, and final verification

**Files:** `docs/api.md`, `docs/user-guide.md`, `README.md`, `examples/NSD/README.md`, relevant explanatory cells in `examples/NSD/nsd_workflow.ipynb` and `examples/NSD/nsd_session_hrf_reliability.ipynb`; the new validation report.

- [x] Document the two modes, changed default, equations, output shapes, rank requirements, single-row run behavior, and meaning of `task`, predictions, run intercepts, and scoring offsets. Show `encoding_mode="absolute"` for old-result reproduction.
- [x] Replace statements that production still uses a shared intercept or that its default is unchanged. Label old validation results by date and link the new report; do not rewrite archived numerical conclusions.
- [x] Clarify that test outcomes supply scoring offsets only, that no unknown test intercept is predicted, and that candidate-specific target distortion remains. Explain that scientific effects constant within a run cannot be estimated by this within-run encoding objective.
- [x] Update notebook explanations and explicit configuration without rerunning large real-data analyses or presenting stale executed outputs as new results. Existing saved outputs must be labeled historical or cleared where they describe changed computations.
- [x] Run `uv run pytest tests examples/NSD -q -W error`. Investigate failures against the recorded baseline; never weaken mathematical tests to obtain GREEN.
- [x] Run `uv run black --check` on changed Python files and `git diff --check`. Inspect the scoped diff for unrelated changes and nonempty `__init__.py` files.
- [x] Commit only changes belonging to this work. Report test results, simulation findings, limitations, and any baseline failures that remain.

## Completion criteria and handoff

The default pipeline agrees with independent run-dummy OLS and centered-loss references for alpha and fractional ridge; it is insensitive to pure run-level beta shifts at the encoding stage; slope errors remain penalized; held-out outcomes never alter training fits or raw predictions; and exported numerical data reproduce the reported scores. Explicit absolute mode reproduces the previous behavior. Scientific results and metadata clearly identify which objective was used.

Recommended execution: implement sequentially in this session because the CV, result, and export interfaces depend on the encoding contract, followed by a focused independent review. Execution was authorized by the user and is complete.

## Execution notes

- Worked on a feature branch in the existing checkout because required prior
  changes were uncommitted. Saved the starting files under
  `/private/tmp/within-run-baseline` and execution evidence under
  `/private/tmp/within-run-progress.md`. Earlier user edits remain in place.
- Consolidated cross-regularizer NSD oracle/export tests in
  `examples/NSD/test_within_run_encoding.py` instead of creating a separate
  provenance test file or duplicating integration tests.
- Added `tests/test_within_run_cv_isolation.py` after independent review to
  cover holdout-specific rank loss and outcome-only candidate prediction isolation.
  These exercise already-implemented behavior and required no implementation fixes.
- The HRF reliability notebook contains no affected encoding workflow and was
  left untouched. The main notebook exposes `encoding_mode`; its saved outputs
  were cleared to avoid presenting historical results under the new default.
- Notebook mode propagation also required `examples/NSD/workflow_outputs.py`
  so top-level exported metadata agrees with the actual scoring objective.
- The initial full-suite sandbox run passed 825 tests; 11 Jupyter/runtime or
  multiprocessing access failures resolved when rerun with required permissions.
  The final full-suite run, including the added notebook-mode case, passed 837.
- Independent review found no blocking defects. Four additional review coverage
  tests passed. Black checked all changed Python files; whitespace checks passed.
- Existing simulation source/tests were untracked prerequisites and are now
  tracked with this feature. Other prior untracked files were not staged.
  Documentation edits inseparable from earlier uncommitted prose remain in the
  working tree; separable feature changes and the new report are committed.
