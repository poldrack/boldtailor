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
