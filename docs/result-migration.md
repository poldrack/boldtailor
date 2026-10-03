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
keys. Ordering, ownership, and scoring definitions are unchanged; beta
magnitudes are not (see "2026-10-02 beta scale and identities" below).


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

## 2026-10-02 beta scale and identities

Every event response is now scaled to a peak of one on Nilearn's oversampled
grid. This applies to plain `ModelSpec(hrf_model="spm")` and
`ModelSpec(hrf_model="glover")` as well as to selected HRFs, so effect sizes
and their variances differ from Nilearn `FirstLevelModel` on the same events
and from earlier boldtailor releases. t, z, p, and R² are unchanged. The
factor depends on event duration and kernel: for canonical SPM, 3 s events,
and TR 1.6 s, an event's response under the earlier sum-to-one kernel peaked at
87.3 / 148.2 (event peak against kernel sum) ≈ 0.59, so current betas are about
0.59 times the earlier ones. Shorter events give smaller factors and longer
events larger ones. Nilearn derivative and FIR bases and user-supplied kernels
are not rescaled.

Related changes to saved records:

- Analysis IDs changed, because the recorded model and HRF normalization
  changed; earlier saved IDs do not match new runs of the same analysis.
- The `hrf` entry of single-trial provenance (`hrf_metadata("spm")`) is now a
  dictionary (`id`, `kind`, `parameters`, `kernel_fingerprint`,
  `normalization`) instead of the string `"spm"`.
- BIDS provenance `Command` values name the public entry point, for example
  `boldtailor.select_hrfs`, `boldtailor.fit`, `boldtailor.fit_single_trials`,
  `boldtailor.fit_selected_hrfs`, `boldtailor.evaluate_hrf_split`,
  `boldtailor.score_ridge_candidates`, and `boldtailor.score_fraction_candidates`.
- Publication control data (lock, staging, backups, and `failures.jsonl`) moved
  out of the dataset into the sibling `<destination>.boldtailor/` directory;
  see [publication-migration.md](publication-migration.md).
- The selected-HRF `trial_table` (and saved `trials.tsv`) column order is now
  `trial_index`, `trial_id`, `run_label`, `event_index`, the event columns,
  then `run_index`, matching the shared-HRF table.
