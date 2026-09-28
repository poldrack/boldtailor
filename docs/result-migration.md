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
selection = select_ridge_fractions(scores.cv_r2, scores.grid)
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
| selected trial result `.group_designs` | `.design.matrices` (same `(run, hrf)` keys) |
| selected trial result `.hrf_indices` | `.design.hrf_indices` |
| selected trial result `.selection_provenance` | `.design.selection_provenance` |
| `HrfSingleTrialResult` | `SingleTrialResult` with `SelectedTrialDesign` |

`SharedTrialDesign` describes a design shared across features, including a
shared custom HRF. `SelectedTrialDesign` holds the feature assignments,
grouped matrices, and selection provenance. Both live in
`boldtailor.single_trial_results`.

`run_betas`, `trial_table`, R² arrays, `diagnostics`, penalty fields, and
`provenance` keep their existing access paths. Arrays remain owned and
read-only; public tables and design mappings are defensive copies.
Conventional GLM results retain their existing design access paths.
