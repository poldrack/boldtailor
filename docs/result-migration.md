# Result API migration

Candidate scoring now returns one `CandidateScores` container. Its
`regularization` identifies the scientific penalty and `grid` labels the score
rows. Normalized-ridge grids ascend; fractional-ridge grids descend.

| Before | After |
| --- | --- |
| `RidgeCandidateScores`, `FractionCandidateScores` | `CandidateScores` |
| alpha candidate scores `.alphas` | `.grid`, `.regularization == "normalized_ridge"` |
| fraction candidate scores `.fractions` | `.grid`, `.regularization == "fractional_ridge"` |

For example:

```python
from boldtailor.fractional_ridge import score_fraction_candidates, select_ridge_fractions

scores = score_fraction_candidates(data, predictors, fractions=[1.0, 0.7, 0.3])
selection = select_ridge_fractions(scores)
```

Selection results retain `RidgeSelection.alphas` and
`FractionSelection.fractions`: their selection operations differ. Saved
scientific artifacts and provenance retain their `alphas` and `fractions`
keys. Numerical values, ordering, ownership, and scoring definitions are unchanged.


Single-trial fits now return `SingleTrialResult` for both shared and
feature-specific HRFs. Common numerical fields stay directly on the result.

| Before | After |
| --- | --- |
| shared trial result `.design_matrices` | `.design.matrices` (tuple of DataFrames) |
| selected trial result `.group_designs` | `.design.matrix(run, hrf)` (rebuilt on request) |
| selected trial `.design.matrices` | `.design.matrix(run, hrf)` (rebuilt on request) |
| `HrfAnalysisResult.group_designs` | `.group_design(run, hrf)` (rebuilt on request) |
| selected trial result `.hrf_indices` | `.design.hrf_indices` |
| selected trial result `.selection_provenance` | `.design.selection_provenance` |
| `HrfSingleTrialResult` | `SingleTrialResult` with `SelectedTrialDesign` |

`SharedTrialDesign` describes a design shared across features, including a
shared custom HRF. `SelectedTrialDesign` holds the feature assignments, the
design fingerprint, and selection provenance, and rebuilds grouped matrices on
request. Both live in
`boldtailor.single_trial_results`.

`run_betas`, `trial_table`, R² arrays, `diagnostics`, penalty fields, and
`provenance` keep their existing access paths. Arrays remain owned and
read-only; public tables and design mappings are defensive copies.
Conventional GLM results retain their existing design access paths.

## Entry-point names and keyword-only results

| Before | After |
| --- | --- |
| `select_hrf(data, ...)` | `select_hrfs(data, ...)` |
| `fit_selected_hrfs(data, selection=...)` | `fit_selected_hrfs(data, hrf_selection=...)` |
| `fit_single_trials(data, hrf=...)` | `fit_single_trials(data, hrf_model=...)` |
| `task_delta_r2_prepared(prepared, result, contrasts=..., noise_model=..., model_metadata=...)` | `task_delta_r2_prepared(prepared, result)` |
| `select_ridge_penalty(scores.cv_r2, scores.grid)` | `select_ridge_penalty(scores)` (the array form still works) |
| `select_ridge_fractions(scores.cv_r2, scores.grid)` | `select_ridge_fractions(scores)` (the array form still works) |

The old names and keywords still work for one release and emit a
`DeprecationWarning`. Old prepared-comparison keywords must match the model
stored in the result's provenance. `evaluate_hrf_split` now accepts
`candidate_batch_size` like `select_hrfs`.

Every result dataclass is keyword-only: positional construction raises
`TypeError`. Public attribute names are unchanged. The selected-HRF
`trial_table` now has the same column order as the shared-HRF table:
`trial_index`, `trial_id`, `run_label`, `event_index`, the event columns, then
`run_index`.
