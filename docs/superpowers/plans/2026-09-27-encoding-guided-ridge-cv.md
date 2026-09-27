# Encoding-guided ridge selection implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Use pytest RED–GREEN–Refactor and commit failing tests before implementation.

**Goal:** Select the single-trial ridge penalty by how well a trial-level encoding model predicts regularized betas in held-out runs, using the 90th percentile of prediction R² across grayordinates.

**Architecture:** Keep the existing single-trial estimator and penalty convention. Add trial-level OLS prediction, a leave-one-run-out candidate scorer, and a separate spatial reduction that chooses one penalty across the complete scoring mask. The NSD adapter handles parallel grayordinate blocks, outer evaluation, final refitting, and CIFTI publication.

**Tech stack:** Existing Python, NumPy, pandas, SciPy/Nilearn, nibabel, joblib, pytest, and Jupyter dependencies; uv for local Python commands.

**Spec:** The [agreed design](#agreed-design) below records the user's approved objective and the implementation choices proposed here. This document is a plan; no ridge behavior has been changed.

## Global constraints

- Write clean, modular code. Prefer short functions over long functions.
- Use `uv run` for all local Python and test commands.
- Every `__init__.py` must remain completely empty.
- Write and commit failing tests before implementation. Never weaken a test or simplify a requirement to make an incomplete implementation pass.
- Use the same candidate penalty for the training and validation beta estimates. Unregularized validation targets are **not** the requested objective.
- Keep ordinary normalized ridge: nuisance regressors/intercept are unpenalized; trial columns have unit L2 norm after nuisance projection. Do not introduce fractional ridge or regularize the encoding model.
- Preserve fixed-alpha APIs, the conventional OLS GLM, existing HRF library generation, unrelated working-tree edits, and existing derivative files.
- The NSD example uses the 512-sample Sobol HRF library plus canonical SPM, seed 0.
- No full-brain production rerun is part of implementing this feature; validate on synthetic data and a bounded real-data smoke run first.

## Agreed design

### Scientific objective

The user approved estimating beta series at each candidate penalty, fitting a model such as `beta ~ trial_type + response_time` on training runs, and predicting regularized beta estimates in validation runs. This optimizes relationships with the specified predictors. It does not establish fidelity of every trial-specific response or independent accuracy against a common response target.

The encoding model is OLS with one intercept (named `task` in its coefficient table), binary `trial_type`, and continuous `response_time` for NSD. A task regressor of all ones would duplicate the intercept and must not be added separately. There are no interactions, automatic feature selection, run intercepts fitted to validation betas, or repeated-image matching. The shared intercept and coefficients transfer across runs; real trial deviations remain possible.

Both training and validation betas use the candidate alpha. Include alpha zero as an ordinary candidate, not as the validation reference. HRF choice is constant across alphas within a fold. Uniform scaling alone does not improve an OLS encoding R²; changes in the relative trial pattern can change the score.

### Fold structure and HRFs

For each outer split, all tuning inputs come from its training runs. Inner validation leaves out one training run at a time. In each inner fold:

1. Select HRFs using only the inner-training runs and the existing mean-stimulus selection objective. Canonical mode skips selection.
2. Freeze these HRFs, then estimate beta series separately in the inner-training and inner-validation runs for every alpha.
3. Fit the encoding coefficients on inner-training betas and predictors; predict inner-validation betas using validation predictors.
4. Accumulate prediction losses and target variation. Validation BOLD is used to estimate the scoring targets, never to select that fold's HRFs or fit encoding coefficients.

Do not pass the existing all-run HRF assignment into tuning: it has seen the inner-validation responses. Reuse fold assignments and prepared trial-design factorizations across alphas. For selected-HRF mode, inner training needs at least two runs; the candidate scorer therefore needs at least three runs. Canonical scoring needs at least two.

**Proposed NSD outer folds:** reuse the established odd/even split in both directions. Tuning on the six odd runs predicts the six even runs, then reverse. This needs at least three runs in each half for optimized HRFs. After alpha selection, reselect HRFs on the whole outer-training half, refit training betas and encoding coefficients, and freeze them for outer evaluation. Estimate outer-test target betas at the selected alpha using these training-selected HRFs.

For final production beta images, perform a separate leave-one-run-out tune using all session runs, select one alpha, select HRFs on all runs, and fit final betas. Final all-run results are descriptive fits; the two outer evaluations provide held-out evidence. Do not average the two outer penalties to obtain the final penalty.

Canonical and optimized modes tune independently and report their own choices. There is one shared penalty per mode and training scope, not one per grayordinate or processing block. Outer R² compares prediction of each pipeline's own regularized targets; do not present differences between pipelines as prediction of an identical target.

### Prediction and R² definitions

Predictor tables contain numeric columns in a declared order and one row per original event. Core code adds the intercept. Center predictor columns using the concatenated valid **training** rows and apply that same centering to validation rows. Do not center betas, estimate a validation intercept, rescale predictions to validation targets, or fit any predictor transform on validation runs.

For candidate `a`, grayordinate `g`, and inner-validation run `r`, calculate on the same predictor-complete trials for every alpha:

```text
SSE[a,r,g] = sum_trials((beta[a,r,g] - prediction[a,r,g]) ** 2)
SST[a,r,g] = sum_trials((beta[a,r,g] - mean_trials(beta[a,r,g])) ** 2)
CV_R2[a,g] = 1 - sum_runs(SSE[a,r,g]) / sum_runs(SST[a,r,g])
objective[a] = percentile(CV_R2[a, scoring_mask], 90)
```

The target mean in SST defines the usual within-run R² denominator; it does not enter prediction. Pool SSE and SST before division, rather than averaging run R² values. Preserve negative R². A zero denominator yields NaN, not zero or one. The denominator changes with alpha because the agreed objective evaluates the candidate's own beta series.

Use a fixed anatomical/requested mask, intersected with features having finite predictions/targets in every scored fold and nonzero pooled SST for **every** alpha. Record this common mask and exclusion counts. Do not select a different finite subset for each alpha, select vertices by RT first, or average block percentiles. Compute NumPy's linear-interpolated 90th percentile only after assembling all grayordinates. Reject an empty common mask. Negative objectives are valid; report them without changing the selection rule.

Proposed defaults: `alphas=(0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)`, `percentile=90.0`. Validate distinct finite nonnegative real alphas, reject booleans, sort ascending, and retain the sorted grid in results. Break objective ties within absolute tolerance `1e-12` by the smallest alpha. Keep the candidate grid and percentile configurable and show the complete objective curve; an endpoint winner is recorded, without automatically searching a new grid on outer-test results.

### Missing data and validity

- Keep every stimulus in beta estimation, including trials with missing RT. Exclude rows with missing/nonfinite predictors only from encoding training/scoring, using the same row mask at every alpha. Preserve row order and report counts per run.
- The NSD adapter accepts nonmissing trial-type codes 0/1 and treats nonpositive RT as unavailable for encoding. Invalid nonmissing type codes fail explicitly.
- Require at least two predictor-complete validation trials per run, and more complete training rows than encoding columns including the intercept. Require full rank of the pooled training encoding design; a single category in one validation run is allowed.
- Constant BOLD, missing HRF assignments, or nonfinite betas produce undefined feature scores. Do not silently fit different trial subsets per feature or alpha.
- Reject mismatched predictor/run counts, event-row counts, predictor column order, overlapping/duplicate split indices, nonfinite BOLD, and invalid penalty settings before expensive fitting where possible.
- Existing full-notebook conventional GLMs still require complete covariates. A standalone beta-CV workflow may use the existing `load_session(..., hrf_only=True)` path to retain missing-RT trials; do not weaken GLM validation as part of this work.

### Outputs and interpretation

Save, separately for canonical/optimized mode and odd-training/even-training/all-run scope:

- Selected alpha, full candidate grid, percentile, common scoring mask, counts, and the candidate objective table/plot.
- Per-candidate inner-CV encoding R² maps, and fold/run labels, trial masks/counts, SSE/SST summaries, predictor transforms, and HRF assignment identifiers needed to audit the calculation.
- Outer-test encoding R² maps and trial-level predictions, clearly separated from inner selection scores and BOLD time-series fit R².
- Final all-run beta-series CIFTIs, selected alpha, HRF parameters/library, trial ordering, and provenance.

RT now participates in penalty tuning. Reports must distinguish that use from an independent check: outer test runs remain excluded from both HRF and alpha selection, while all-run beta/RT correlations are descriptive. Do not calculate naive confirmatory p-values from the tuning maps. The existing session-to-session HRF reliability notebook remains unchanged.

## Review focus

1. Changing outer-test BOLD or behavioral values cannot change alpha, training-selected HRFs, training predictor transforms, or encoding coefficients. Tested in Tasks 3 and 4.
2. A different block size or worker count cannot select a different penalty by changing the spatial reduction or fold population. Tested in Tasks 1 and 4.
3. Missing RT must not delete a stimulus from the time-series design or shift beta/trial alignment. Tested in Tasks 1, 3, and 5.
4. Constant targets, negative R², rank-deficient encoding designs, and sparse categories must have explicit, reproducible behavior. Tested in Tasks 1 and 3.
5. Old fixed-alpha outputs and all-run HRF selections must not be mislabeled or reused as nested-CV results; cache/provenance identity must include predictors, folds, mask, grid, and objective. Tested in Tasks 4 and 5.

## File responsibilities and interfaces

| File | Responsibility |
| --- | --- |
| New `src/boldtailor/trial_encoding.py` | Numeric trial-predictor validation, training-only centering, OLS coefficients, held-out predictions, SSE/SST |
| New `src/boldtailor/ridge_results.py` | Owned/read-only `TrialEncodingResult`, `RidgeCandidateScores`, and `RidgeSelection` results |
| Modify `src/boldtailor/_single_trial_fit.py` | Reuse a nuisance-projected trial-design factorization across an alpha path while preserving the existing solver |
| New `src/boldtailor/_ridge_cv.py` | Run subsetting with source provenance, fold HRF selection, grouped beta paths, loss accumulation |
| New `src/boldtailor/ridge_selection.py` | Public candidate scoring and global spatial penalty selection |
| New `examples/NSD/ridge_workflow.py` | Trial-type/RT predictor adapter, geometry-based blocks, outer folds, global choice, final beta refit |
| New `examples/NSD/ridge_outputs.py` | CIFTI/TSV/JSON/NPZ artifacts and score plots for tuning/evaluation |
| Modify `examples/NSD/workflow_analysis.py` | Reuse existing beta fitting after a scalar alpha is selected; update applicable descriptions |
| Modify `examples/NSD/workflow_outputs.py` | Optional CV artifact group and estimator-specific metadata |
| Modify `examples/NSD/nsd_workflow.ipynb` | CV settings, tuning/outer diagnostics, final tuned beta series |
| New `tests/test_trial_encoding.py`, `tests/test_ridge_selection.py`, `tests/test_ridge_cv.py` | Numerical oracles, edge cases, fold isolation, ownership/provenance |
| Modify `tests/test_single_trial.py` | Alpha-path equivalence to independently augmented least squares |
| New `examples/NSD/test_ridge_workflow.py` | Global block reduction, parallel equivalence, synthetic CIFTI execution/publication |
| Modify `examples/NSD/test_nsd_workflow.py` | Executed notebook behavior in CV, fixed, and disabled modes |
| Modify `README.md`, `docs/api.md`, `docs/user-guide.md`, `examples/NSD/README.md` | Human-facing use, score interpretation, fixed-alpha compatibility |

Public interfaces to implement:

```python
# boldtailor.trial_encoding
evaluate_trial_encoding(
    beta_runs, predictors, *, train_runs, test_runs
) -> TrialEncodingResult

# boldtailor.ridge_selection
score_ridge_candidates(
    data, predictors, *, alphas, library=None,
    run_labels=None, feature_signature=None,
) -> RidgeCandidateScores

select_ridge_penalty(
    candidate_r2, alphas, *, percentile=90.0, feature_mask=None,
) -> RidgeSelection

# examples.NSD.ridge_workflow
fit_cv_beta_series(
    runs, root, *, library, alphas, percentile=90.0,
    block_size=4096, max_grayordinates=None, n_jobs=1,
) -> dict
```

`library=None` means canonical SPM. A library requests fresh fold-specific optimized HRFs; the scoring API deliberately does not accept an all-run `HrfSelectionResult`.

`TrialEncodingResult` contains `coefficients` (intercept first), `predictor_means`, `predictor_names`, `train_runs`, `test_runs`, `trial_masks` (one per supplied run), `predictions` (one array per test run, original rows with NaN at excluded trials), `run_sse`, `run_sst`, and pooled `r2`. Arrays are owned/read-only; input tables are not modified.

`RidgeCandidateScores` contains sorted `alphas`, `cv_r2` (alpha × feature), `fold_sse`/`fold_sst` (validation run × alpha × feature), `fold_hrf_indices` (validation run × feature; zero for canonical), `trial_masks`, `run_labels`, and `provenance`. Undefined/incomplete feature scores remain NaN. The result records selection statistics, not outer-test accuracy.

`RidgeSelection` contains `ridge_alpha`, sorted `alphas`, `objective_scores`, `percentile`, and `scoring_mask`. It is a numeric decision result; the workflow combines it with source/fold provenance. It does not change beta units or contain a spatially varying alpha map.

`fit_cv_beta_series` returns `final` (the existing beta-result dictionary accepted by `_beta_artifacts`), `tuning` (results keyed by `odd`, `even`, `all`), `evaluation` (results keyed by `odd_to_even`, `even_to_odd`), and `provenance`. Each tuning entry contains `scores` (`RidgeCandidateScores` restored to the full spatial axis), `selection` (`RidgeSelection`), and training `run_labels`. It handles one HRF mode per call. Evaluation dictionaries contain `encoding_r2`, original-order test `predictions`, test `betas`, training encoding coefficients/transforms, training HRF assignments, and explicit train/test run labels. Top-level publication combines canonical and optimized calls.

## Task 1: Trial-level prediction and the spatial selection rule

**Files:** new `trial_encoding.py`, `ridge_results.py`, `ridge_selection.py`, and their `test_trial_encoding.py`/`test_ridge_selection.py` tests. Initially `ridge_selection.py` exposes only the spatial reducer; Task 3 adds candidate scoring.

**Consumes:** raw beta arrays and numeric predictor tables in original event order; candidate R² arrays from a caller.

**Produces:** `evaluate_trial_encoding`, `select_ridge_penalty`, and the result types above.

- [ ] Write numerical tests using hand-built beta arrays and an independent `np.linalg.lstsq` oracle. Include this train/test offset case so scoring cannot quietly refit a validation intercept:

```python
def test_encoding_does_not_remove_a_validation_offset():
    x = np.array([-1., 0., 1., 2.])
    predictors = [pd.DataFrame({"response_time": x}) for _ in range(3)]
    betas = [(2 + 3*x)[:, None], (2 + 3*x)[:, None], (12 + 3*x)[:, None]]
    result = evaluate_trial_encoding(betas, predictors, train_runs=[0, 1], test_runs=[2])
    np.testing.assert_allclose(result.predictions[0][:, 0], 2 + 3*x)
    expected = 1 - np.sum((betas[2][:, 0] - (2 + 3*x))**2) / np.sum((3*x - (3*x).mean())**2)
    np.testing.assert_allclose(result.r2, expected)
    assert result.r2[0] < 0
```

- [ ] Add unequal-run-length pooling tests; missing-predictor rows that remain in output arrays; multiple predictor columns with training-only centering; single-category validation runs; rank-deficient training designs; insufficient rows; invalid split indices; nonfinite betas at one feature; immutable results and unchanged inputs.
- [ ] Add reducer tests for a common finite mask, an explicit anatomical mask, all-negative objectives, zero eligible features, alpha/percentile validation, exact/interpolated percentile results, and stable lowest-alpha ties. Pin the critical block behavior with:

```python
def test_percentile_is_over_grayordinates_not_block_percentiles():
    # Averaging each block's 90th percentile selects alpha 0.1 incorrectly.
    scores = np.array([[0., 0., 0., 0., 1., 1.], [.6, .6, .6, .6, .6, .6]])
    selection = select_ridge_penalty(scores, [0., .1])
    assert selection.ridge_alpha == 0.
    np.testing.assert_allclose(selection.objective_scores, [1., .6])
```

- [ ] Run `uv run --no-cache --no-sync pytest tests/test_trial_encoding.py tests/test_ridge_selection.py -q -W error -p no:cacheprovider`. Verify failures name missing behavior, then commit the failing tests only.
- [ ] Implement short validation, training-transform, fit, prediction, and reduction helpers. Use explicit SSE/SST and `np.percentile(..., method="linear")`; do not use a scorer that silently forces constant-target R² to finite values. Sort the alpha axis together with its scores. Include an unsorted-grid regression test.
- [ ] Run the same command to GREEN; refactor and commit the implementation separately.

## Task 2: Reuse the trial solver across candidate penalties

**Files:** modify `_single_trial_fit.py` and `tests/test_single_trial.py`.

**Consumes:** the existing `_project_design` factorization and fixed-alpha semantics.

**Produces:** private `trial_beta_path(x, nuisance, signals, *, alphas)`, an iterator of `(alpha, betas)` in supplied alpha order. It computes trial betas only; final full-fit diagnostics still come from the existing public fitting functions.

- [ ] Extend the existing independent augmented-OLS test to the path. For each alpha, compare to a directly solved system with rows `[X, N]` and penalty rows `[sqrt(alpha) * diag(projected_column_norms), 0]`. Include alpha zero, nonorthogonal/unequal-norm trial columns, duplicate nuisance columns, constant signal columns, and an ill-conditioned but full-rank design.
- [ ] Add one factorization-count test to enforce that the path reuses the projected design across alphas, and preserve the existing rank/no-residual-DOF rejection tests.
- [ ] Run `uv run --no-cache --no-sync pytest tests/test_single_trial.py -q -W error -p no:cacheprovider`; verify the new tests fail and commit them before implementation.
- [ ] Share the existing projected solver without altering its alpha-zero behavior. For positive alpha the beta path uses:

```python
weights = (vt.T * (s / (s*s + alpha))) @ projected_signal_coordinates
beta = weights / scale[:, None]
```

  Keep any observed-signal projections local to the current run/fold call; do not add a global data cache. Yield one candidate array at a time. Fit every event row regardless of predictor completeness.
- [ ] Run the single-trial, selected-HRF, and trial-design tests to GREEN, then commit the implementation. Existing fixed-alpha results must remain numerically equivalent.

## Task 3: Inner run CV with fold-specific HRFs

**Files:** new `_ridge_cv.py`, extend `ridge_selection.py`/`ridge_results.py`, new `tests/test_ridge_cv.py`.

**Consumes:** Tasks 1–2, `AnalysisData`, `select_hrf`, canonical/selected trial-design compilation, and run-source provenance.

**Produces:** `score_ridge_candidates(...) -> RidgeCandidateScores`.

- [ ] Build a pytest fixture with six unequal-length runs, eight trials per run, two nuisance columns, unique stimulus IDs, independent trial type/RT predictors, mixed known library HRFs, noisy trial effects, and one constant grayordinate. Generate BOLD by convolving known trial amplitudes; use a small three-candidate HRF library for test speed. Set one observed RT to NaN after generating the signal.
- [ ] Add an independent reference loop: for each held-out run, select HRFs on the other runs, solve candidate trial betas with augmented least squares, fit encoding coefficients with `np.linalg.lstsq`, and calculate raw SSE/SST. Compare every returned candidate/fold map, selected HRF ID, and row mask. Do not call the production encoding or beta-path helper in the oracle.
- [ ] Add an explicit changing-target regression case where a nonzero candidate's validation betas differ from OLS. Assert the score matches that candidate's targets and differs from the OLS-target alternative.
- [ ] Prove isolation: perturb only validation BOLD and confirm that fold's HRFs and training coefficients are unchanged; perturb validation RT and confirm HRF/encoding fitting does not see it. Other inner folds may legitimately change because the perturbed run trains those folds. Instrument the HRF selector's input run labels and capture the real `evaluate_trial_encoding` results to inspect coefficients without replacing either computation with a stub.
- [ ] Test no repeated-image requirement, missing RT retaining all trial regressors, validation-only rank failure reporting with fold labels, canonical mode without HRF selection, too few runs, empty common masks, and immutability.
- [ ] Run `uv run --no-cache --no-sync pytest tests/test_ridge_cv.py -q -W error -p no:cacheprovider`; inspect the RED failure and commit tests.
- [ ] Implement run subsetting through `from_arrays` using corresponding signals, events, frame times, confounds, and `data.provenance.sources`. Preserve caller run labels and spatial signatures. Choose each fold's HRFs once, assemble grouped trial-beta paths, evaluate each alpha, then reduce:

```python
pooled_sse = fold_sse.sum(axis=0)
pooled_sst = fold_sst.sum(axis=0)
cv_r2 = np.full_like(pooled_sse, np.nan)
np.divide(pooled_sse, pooled_sst, out=cv_r2, where=pooled_sst > 0)
cv_r2 = 1 - cv_r2
```

  Propagate any invalid required fold to NaN instead of using `nansum` to give that candidate fewer trials. Stream beta paths and retain scores/assignments rather than every candidate's beta series. Shared design factorization reuse must not share HRF selections across different training sets.
- [ ] Record predictor column/value fingerprints (including missingness), training predictor transforms per fold, alpha grid, normalization, folds, library identity, HRF assignments, source identities, trial masks/counts, and score definition. Changing behavior values must alter the candidate-score provenance, even though fixed-alpha beta fitting alone ignores those values.
- [ ] Run Tasks 1–3 and existing HRF selection/selected-HRF fitting tests to GREEN; commit implementation.

## Task 4: NSD global selection, outer evaluation, and final fits

**Files:** new `ridge_workflow.py` and `test_ridge_workflow.py`; targeted changes to `workflow_analysis.py` only where a fitting helper is shared.

**Consumes:** candidate scores per spatial block; the existing `map_blocks`, `load_block`, selected-HRF fitting, and scalar-alpha final beta-series functions.

**Produces:** `fit_cv_beta_series(...)` with the dictionary contract above.

- [ ] Add six-run synthetic CIFTI fixtures and a serial reference workflow. Assert both odd/even outer splits, one alpha across all blocks in each scope, and a separate final all-run tune. Test `n_jobs=1` versus `2` and several uneven block sizes; compare selected alphas, masks, score arrays, and final betas.
- [ ] Add an outer-test perturbation regression test through the full workflow: change only even-run BOLD/RT, then assert the odd-training tuning result, selected alpha, HRF assignment, predictor transforms, and training encoding coefficients are identical. The even-to-odd and final fits may change. Check the reverse direction separately.
- [ ] Test geometric masking independent of outer-test responses, missing/nonpositive RT handling, invalid type codes, insufficient runs per half, input CIFTI axis mismatch, and output row order.
- [ ] Run `uv run --no-cache --no-sync pytest examples/NSD/test_ridge_workflow.py -q -W error -p no:cacheprovider`; inspect RED and commit tests.
- [ ] Partition the requested CIFTI axis geometrically. Do not use an all-run response/RT threshold to construct outer-training tuning masks. Worker calls to `score_ridge_candidates` receive only the relevant outer-training runs. In the parent, restore all candidate maps to original grayordinate positions and call `select_ridge_penalty` **once** for the entire scope.
- [ ] After the scalar decision, perform a separate fixed-alpha pass: select HRFs on the complete outer-training half, estimate its beta series and test beta targets, and call `evaluate_trial_encoding`. Outer-test predictors/BOLD never enter the preceding alpha decision. Keep unused/unprocessed grayordinates NaN on the full original CIFTI axis.
- [ ] Repeat tuning on all runs for final output. Call existing final beta-series fitting with the chosen scalar and all-run HRF selection. Keep final beta dictionaries compatible with `_beta_artifacts`; attach the corresponding tuning provenance.
- [ ] Parallelize spatial blocks only with existing bounded `map_blocks`; keep scopes sequential, avoid nested process pools, and report one start/completion message per scope. Retain scalar maps and selected fits, not all candidate betas. Run tests to GREEN and commit.

## Task 5: Notebook, exports, and documentation

**Files:** new `ridge_outputs.py`; modify `workflow_outputs.py`, `nsd_workflow.ipynb`, notebook/adapter tests, and the four documentation files listed above.

**Consumes:** Task 4's final/tuning/evaluation dictionaries and existing artifact publication utilities.

**Produces:** executed NSD notebook support and reviewable saved diagnostics/final beta series.

- [ ] Add an executed six-run synthetic notebook case for CV mode and retain the existing four-run fixed/disabled cases. Inject a short alpha grid and small HRF library in tests. Inspect exported scalar axes, beta/trial alignment, alpha tables, score definitions, masks, and nested fold/source provenance.
- [ ] Assert old `NSD_CONFIG` overrides containing only `ridge_alpha` retain fixed/disabled behavior. New CV defaults use `ridge_mode="cv"`, `ridge_alphas=[0., .001, .01, .1, 1., 10., 100.]`, and `ridge_percentile=90.`. Explicit `ridge_mode="fixed"` uses positive `ridge_alpha`; `"off"` retains OLS alone. Alpha zero may win CV and must not be rejected by the old fixed-positive check.
- [ ] Test publication collision handling and distinct descriptors: `CanonicalTrialRidgeCV`/`OptimizedTrialRidgeCV` for final betas, with separate `RidgeCV` descriptors for odd/even/all tuning and outer prediction. Verify fixed-alpha outputs are untouched and are not loaded as CV results. Existing HRF caches may be reused only if their exact training-run scope/library/preprocessing matches; all-run assignments cannot serve inner folds.
- [ ] Run the targeted notebook and adapter tests to RED and commit tests before editing the notebook/source.
- [ ] Add CV cells showing alpha versus 90th-percentile inner R², the two outer prediction maps/summaries, and the final chosen alpha. Keep canonical/optimized OLS comparisons. Save final tuned beta series through existing publication, adding CV artifacts as one complete set. Use clear scalar names such as `encoding_inner_cv_r2_alpha-0.1` and `encoding_outer_r2_odd_to_even`; do not reuse BOLD `full_r2` labels for these scores.
- [ ] Save the exact trial predictor tables/missingness, original trial IDs, and training transforms with the model inputs. Save common scoring masks and full candidate maps; JSON/TSV must identify target betas as regularized at the candidate or selected alpha. NaNs in numeric artifacts remain NaN; JSON uses null where needed. Candidate-grid/fold/objective/predictor changes invalidate any CV artifact reuse.
- [ ] Update human-facing descriptions: task is the encoding intercept; HRFs still use mean stimulus prediction; RT/type now tune the beta penalty; inner scores are selection statistics; outer scores predict regularized targets; conventional GLMs remain OLS. Explain separate all-run final fitting and why final RT correlations are descriptive. Remove blanket claims that RT never tunes ridge, while preserving that statement for explicitly fixed-alpha paths.
- [ ] Preserve unrelated notebook edits. Update only affected source cells, clear stale outputs for changed analysis cells, and do not rewrite unrelated outputs/metadata. Leave the across-session HRF reliability notebook unchanged.
- [ ] Execute CV/fixed/off notebook tests and existing workflow/session reliability regressions to GREEN; commit implementation and documentation.

## Task 6: Independent verification and bounded NSD smoke run

**Files:** new `docs/validation/nsd-ridge-cv.md` records executed checks and limits.

- [ ] Run all tests with warnings as errors:

```sh
uv run --no-cache --no-sync pytest tests examples/NSD -q -W error -p no:cacheprovider --tb=short
```

- [ ] Run the repository's formatter check on touched Python files and `git diff --check`. Confirm every `__init__.py` remains empty and unrelated tracked edits are preserved. Inspect the diff for any accidental changes to existing fixed-alpha behavior or HRF cache semantics.
- [ ] Execute the CV workflow on all 12 runs of `sub-07/ses-nsd10`, the full 513-HRF library, and a small fixed grayordinate subset, writing under a new temporary output root. Exercise both outer splits and the all-run final fit. Start with a three-alpha grid for the smoke check; the configured production default remains the seven-value grid above.
- [ ] Independently reconstruct the small smoke subset's candidate betas from original BOLD and saved fold HRF assignments, then recompute trial encoding predictions, pooled SSE/SST, spatial percentiles, and winning alphas. This verification refits the bounded subset; production need not save every candidate beta array. Cross-check saved outer predictions against their beta/predictor artifacts and final betas against a direct fixed-alpha fit using the saved selection. Inspect the score plot and verify full 91,282-grayordinate output axes with NaN outside the tested subset.
- [ ] Record elapsed time, peak memory if available, selected grid/HRFs, number of grayordinates/runs/trials, exclusions, numerical tolerances, test results, and that this does not establish whole-brain scientific performance. Do not tune the grid or percentile based on outer-test results.

## Completion criteria

The feature is ready when the numerical oracle tests pass, outer-test perturbations leave training decisions unchanged, serial/parallel and block-size results agree, the synthetic notebook publishes the expected artifacts, fixed-alpha regressions pass, and the bounded real-data check is documented. The deliverable is executable API/notebook support for the agreed objective, not a claim that the selected beta series recover every component of neural trial variability.
