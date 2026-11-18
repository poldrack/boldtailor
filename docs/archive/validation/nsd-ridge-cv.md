# Encoding-guided ridge CV validation

> **Dated validation record:** Results and test counts apply to the recorded
> revision/settings, not necessarily this checkout. See the
> [documentation index](../README.md) for current API and methods guidance.

Validated on 2026-09-27 using `sub-07/ses-nsd10`. This checks numerical
correctness and output alignment on a bounded subset; it does not establish
whole-brain scientific performance or the best production penalty grid.

## Automated checks

The new tests cover same-alpha validation targets, training-only predictor
centering and HRF selection, pooled SSE/SST, global percentile reduction,
negative and undefined scores, missing RTs, rank-deficient encoding designs,
spatial block/worker invariance, and outer-test perturbation isolation.
Executed synthetic notebooks cover CV, fixed, and disabled ridge modes,
including older `NSD_CONFIG` overrides.

The initial isolated checkout passed 647 tests, with three failures in the
unchanged stop-signal notebook. Including the user's existing stop-signal
notebook/test edits passed all 650 tests. Those edits were preserved and were
not committed as part of this feature.

A fresh code review identified one provenance gap. Two regression tests first
failed, then passed after final beta provenance was linked to an identified
all-run tuning decision. Changing predictors, grid, percentile, or scoring
population now changes that identity even when the chosen alpha is unchanged.
The final combined-tree suite passed **651 tests**, with warnings treated as
errors (97 seconds).

Verification command:

```sh
MPLCONFIGDIR=/private/tmp/ridge-mpl-cache uv run --no-cache --no-sync \
  pytest tests examples/NSD -q -W error -p no:cacheprovider --tb=short
```

Touched Python files pass Black; `git diff --check` passes. Package
`__init__.py` files remain empty.

## Real NSD check

Inputs were all 12 CIFTI runs and 750 presentations from
`/Volumes/extdata1/NSD/BIDS/sub-07/ses-nsd10`, with fMRIPrep 25.2.5 confounds.
The check used the first eight grayordinates, blocks of four, two workers,
and the canonical-plus-512 Sobol library (seed 0). Both HRF modes evaluated
the preselected alpha grid `[0, 0.1, 1]`. No trials were excluded for missing
encoding predictors.

Both odd/even outer directions and separate all-run tuning/refitting completed.
Every scope selected alpha 1 on this small grid. All-run 90th-percentile inner
R² was 0.0319892 for canonical HRFs and 0.0202974 for optimized HRFs. These are
selection statistics for eight locations and are not evidence of relative
whole-brain performance. The notebook's production grid remains
`[0, 0.001, 0.01, 0.1, 1, 10, 100]`.

The run published 159 files under `/private/tmp/nsd-ridge-cv-smoke`, including
candidate scores, masks, fold HRF assignments, outer-test predictions and
targets, and final beta series. Every CIFTI retained the full 91,282-element
BrainModel axis, with NaNs outside the tested subset (zeros for mask images).
Original trial IDs were preserved. The tuning plot was visually inspected.

The timed run took 144 seconds including fitting, publication, and numerical
audit. Canonical fitting took 38.8 seconds and optimized fitting 25.6 seconds;
these sequential timings have different cache/process-start conditions and
are not a speed comparison. Parent-process peak RSS was approximately 4.71 GiB;
worker memory is not included. Full-axis beta/prediction export buffers are
allocated even for this small fitting subset. Whole-brain runtime and memory
have not been benchmarked.

## Independent numerical reconstruction

The audit reconstructed trial betas from original BOLD, nuisance matrices,
and the selected HRFs using augmented least squares, independently of the
production ridge path. It refitted the OLS encoding model, pooled fold losses,
recomputed spatial percentiles, and verified each winning alpha. A second
audit loaded the saved fold HRF maps and repeated this reconstruction against
the saved candidate maps, fold losses, and metadata. It also checked saved
outer coefficients, predictions, targets, trial IDs, and final beta images.

Both modes passed. Float64 losses used relative tolerance `2e-7` and absolute
tolerance `1e-5`; R²/objectives used absolute tolerances of `1e-8` in memory
and `1e-7` after CIFTI serialization. Float32 beta/prediction exports used
relative tolerance `2e-6` and absolute tolerance `1e-4` in native signal units.

Temporary audit scripts and records:

- `/private/tmp/verify_ridge_nsd.py`
- `/private/tmp/audit_ridge_saved.py`
- `/private/tmp/nsd-ridge-cv-smoke/audit.json`
- `/private/tmp/nsd-ridge-cv-smoke/saved-audit.json`

The numerical smoke preceded the final provenance-only fix; automated
publication and notebook tests verify the added decision linkage. No fitted
values or selection calculations changed in that fix.

The score measures prediction of candidate-regularized betas. Because each
candidate changes its targets, it does not measure recovery of a common
unobserved ground-truth beta series. Outer-test scores were not used to alter
the grid, percentile, or algorithm.
