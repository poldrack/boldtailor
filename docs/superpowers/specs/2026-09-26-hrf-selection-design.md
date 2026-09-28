# Expanded HRF library and run-wise selection design

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

Status: implemented and validated on 2026-09-26. See the
[validation record](../validation/2026-09-26-hrf-selection.md) for results,
review decisions, and the deferred diagnostic-only design archive enhancement.

## Intent and agreed validation target

Add one selected HRF per grayordinate to boldtailor's existing single-trial
analysis. Use an expanded, deterministic library and cross-validation across
whole runs. Keep unrestricted trial amplitudes, existing confounds, OLS and
optional fixed ridge, and RT as an independent diagnostic.

The user explicitly chose **mean-stimulus prediction for HRF selection**:
learn a common mean stimulus amplitude on training runs, predict a held-out
run, then fit free single-trial amplitudes afterward. This requires a mean
response that transfers between runs, not repeated images or equal amplitudes
for individual presentations. No empirical Bayes or ridge/RT tuning is added.

## Library

Use the executable `make_parameter_sets()` grid in
`/Users/poldrack/Dropbox/code/GLMsingle/notebooks/component_generation.ipynb`.
The notebook's prose gives different ranges; the code is the reference here.
Inspected notebook SHA256:
`6e116a740645988332d997ce53dc682977c832a1d8dd3ce82e226d364fa96208`.

| Parameter | Values |
| --- | --- |
| Response delay | 3, 4.5, 6 s |
| Undershoot delay | 10, 16 s |
| Response dispersion | 0.5, 1, 1.5 |
| Undershoot dispersion | 0.5, 1.5, 2.5 |
| Response/undershoot ratio | 2, 4, 6, 8 |
| Onset delay | 0, 1, 2 s |
| Kernel duration | 36 s |

There are 648 combinations. Add candidate **0**, the exact existing Nilearn
`"spm"` model with its current 32-second support, for **649 candidates**.
Do not silently replace that baseline with a 36-second approximation.
The notebook grid does not include undershoot dispersion 1, so its candidates
do not already contain the standard SPM parameter vector.

Represent candidates with stable IDs, named parameters, generator identity,
and a content fingerprint. Generate tabulated curves at 0.1 s for export;
evaluate the double-gamma formula at the actual convolution resolution
`TR / 50`. Normalize each discrete convolution kernel to sum to one, matching
the current SPM convention. The notebook's later peak normalization is for
its simulation comparison and will not be copied into beta estimation.
Publish normalization explicitly; cross-vertex betas remain conditional on
the HRF normalization convention, not estimates of peak BOLD response.

## Design construction

Use the existing Nilearn `compute_regressor` path with oversampling 50 and
exact supplied frame times. Each candidate is evaluated separately. Passing
all candidate functions as one HRF basis would orthogonalize their columns
and change the candidate models.

For selection, convolve all presentations with unit amplitudes to make **one
stimulus regressor per candidate per run**. For final estimation, convolve
each original event row separately. Use recorded durations, preserve trial
IDs and sub-TR timing, and keep the current supported zero-duration impulse
convention and pre-acquisition event-window validation. Never round .774/.775
second NSD start offsets to the notebook's 0.1-second grid. No convolution
crosses a run boundary.

RT, image identity, and trial ordering metadata do not determine HRF choice.
The current canonical-only API must retain its exact behavior when the new
option is unused.

## Cross-validation criterion

Let `N_r` contain the supplied nuisance columns plus a run intercept, and
`M_r = I - Q_r Q_r.T`, where `Q_r` spans `N_r`. For candidate `h`, define:

```text
x[r,h] = M_r @ stimulus_regressor[r,h]
y[r]   = M_r @ signals[r]

A[r,h]   = x[r,h].T @ x[r,h]
B[r,h,v] = x[r,h].T @ y[r,v]
C[r,v]   = y[r,v].T @ y[r,v]

b[-r,h,v] = sum(s != r, B[s,h,v]) / sum(s != r, A[s,h])
L[r,h,v]  = C[r,v] - 2*b[-r,h,v]*B[r,h,v] + b[-r,h,v]^2*A[r,h]
score[h,v] = 1 - sum(r, L[r,h,v]) / sum(r, C[r,v])
```

Choose the candidate with the lowest pooled held-out SSE at each grayordinate.
Pool sums, rather than averaging run-wise R². Retain negative scores.
This score measures **nuisance-adjusted mean-stimulus prediction**, not the
existing full-model in-sample R². Task amplitude is never fitted on the held-out
run. Nuisance projection is explicitly conditional on that run's fixed
confound design; it is not prediction of scanner drift or motion effects.
Use native signal units with no implicit per-run standardization.

At least two runs are needed. Nuisance redundancy is handled by its span.
Reject unsupported or rank-deficient stimulus/trial designs using timing and
confounds only, never held-out signals. Final trial models require positive
residual degrees of freedom. Candidate eligibility is common across compared
runs, so scores always refer to the same observations. Structural checks can
be performed lazily on winners, caching each candidate's eligibility across
all relevant runs and moving to the next scored candidate if it is ineligible;
never silently drop trials or change the observation set for a candidate.

For all-constant features return no selected HRF, NaN scores/betas, and a
documented invalid-feature mask. Roundoff-sized SSE negatives may be set to
zero under a documented tolerance; substantial negatives are errors. Resolve
score ties deterministically by stable ID, preferring candidate 0 on a tie.
Low or negative CV support remains visible; do not silently impose a new
data-dependent fallback or a significance threshold.

## Selection, evaluation, and RT independence

1. **Production selection:** leave one run out at a time across all 12 runs;
   choose one HRF per grayordinate and estimate all trial betas using it.
2. **Independent NSD evaluation:** predeclare odd runs as training and even
   runs as test. Select HRFs by leave-one-run-out CV entirely within the odd
   runs. Refit the mean stimulus amplitude using all odd runs, then predict
   even runs without changing that amplitude or HRF. Compare with a canonical
   predictor trained on exactly the same odd runs.
3. **RT reality check:** choose vertices from canonical-OLS odd-run RT maps,
   as before. Fit free even-run trial betas using the **odd-selected** HRFs
   for those vertices and produce even-run RT plots for OLS and fixed ridge.
   Keep the same selected vertices for each model. Even-run signals and RT
   must not influence the odd-run HRF choice or vertex selection.

The best all-run CV score was used to choose among 649 candidates; label it
as a selection score. Report the separate odd-to-even result as the independent
evaluation. Production RT maps may still be reported descriptively, but must
not be labeled as independent HRF validation. No full nested leave-one-run-out
framework or extra parameter search is needed in the initial implementation.

## API and data flow

Keep numerical code separate from CIFTI I/O:

- `hrf_library.py`: immutable candidate/library descriptions and generation.
- `_hrf_design.py`: single-candidate stimulus and trial convolution helpers.
- `_hrf_cv.py`: run sufficient statistics, leave-one-run-out scoring, eligibility.
- `hrf_selection.py`: public selection/evaluation entry points and lifecycle.
- `hrf_results.py`: owned immutable selection and grouped-fit results.
- Extend `single_trial.py` with a single specified HRF, and provide a grouped
  fitting entry point that applies a selected HRF map to features.
- Extend the NSD example and artifact writer for the new mode.

Proposed interfaces:

```python
library = expanded_hrf_library()  # deterministic 649-candidate default
selection = select_hrf(data, library=library, run_labels=labels,
                       feature_signature=spatial_signature)
optimized = fit_selected_hrfs(data, selection=selection, ridge_alpha=0.0,
                              feature_signature=spatial_signature)
ridge = fit_selected_hrfs(data, selection=selection, ridge_alpha=0.1,
                          feature_signature=spatial_signature)
evaluation = evaluate_hrf_split(data, library=library,
                                train_runs=odd_indices, test_runs=even_indices,
                                feature_signature=spatial_signature)
```

Selection exposes selected indices, selected/canonical CV scores, their
difference, candidate eligibility, run labels, and provenance. Evaluation
exposes the training selection, frozen training amplitudes, and pooled
test/canonical scores. All returned arrays are immutable; table access owns
nested metadata as in the current API. Selection records can be applied to
new runs with the same feature ordering. An optional caller-supplied
`feature_signature` must match between selection and fitting; the NSD wrapper
always supplies a hash of the ordered BrainModel axis and block indices.
For anonymous arrays without a signature, validate feature count and document
that the caller must preserve feature order; numerical values alone cannot
detect a permutation. Also verify the library fingerprint and ID mapping.

Group features by selected HRF and reuse the existing OLS/fixed-ridge kernel.
Both estimators use the same HRF map. Return betas in original feature/trial
order and compute the existing pooled full/nuisance/ΔR². A single run now has
multiple designs: represent and export designs by `(run, HRF ID)`, with a
feature-to-HRF map; do not populate the old single-design-per-run field with
a misleading representative matrix.

## Computation and outputs

Use feature blocks and candidate batches. Residualize each signal block once
per run, reuse nuisance bases, and build stimulus candidate matrices once per
run. Scores use `A/B/C`, not one full GLM per candidate and grayordinate.
Only final selected candidates need feature-group single-trial fits. Keep
structural eligibility checks separate from data-dependent scoring, cache their
outcomes, and profile design compilation before considering extra optimization.
Never materialize a candidates × trials × all-grayordinates beta array.

Save new names under the existing derivative root, preserving all previous
outputs. Export selected HRF IDs, parameter/peak-time maps, selection scores
versus canonical, odd-trained HRF IDs and held-out scores, trial beta maps,
existing in-sample R² maps, and labeled RT diagnostics. Save the library table,
sampled HRFs, normalization/sampling metadata, fold membership, grouped designs,
eligibility diagnostics, and provenance fingerprints. Publication remains an
atomic artifact set. Expand the writer rather than fabricating one shared
design TSV for spatially varying HRFs. Use one compressed NumPy archive per
run containing each used candidate's trial matrix, one shared nuisance matrix,
frame times, and column/HRF IDs. Store float64 design values. This avoids
thousands of TSVs repeating the same confounds while preserving exact designs
for audit; the existing canonical TSV format stays unchanged.

## Validation and boundaries

Use pytest RED–commit–GREEN for each implementation task. Verify against
independent double-gamma/convolution and explicit stacked-design least-squares
oracles. Simulations must use different onsets and nonidentical amplitudes
across runs, include AR noise, and never require repeated images. Include
known noncanonical HRFs, a canonical-only library, noise-only data, nuisance
scaling/rank edge cases, constants, invalid candidates, and block invariance.
Tests must prove held-out amplitudes stay frozen and even-run edits cannot
change the training selection. Benchmark the expanded library on a subset
before the full 12-run/91,282-grayordinate example; report measured runtime,
memory, held-out improvement or failure, and RT associations honestly.

Reference behavior: [Nilearn custom HRF sampling](https://nilearn.github.io/stable/modules/generated/nilearn.glm.first_level.compute_regressor.html),
[SPM double-gamma parameters](https://raw.githubusercontent.com/spm/spm/main/spm_hrf.m),
and [separating model selection from test evaluation](https://scikit-learn.org/stable/modules/cross_validation.html).
