# Per-grayordinate fractional ridge validation

> **Dated validation record:** Results and test counts apply to the recorded
> revision/settings, not necessarily this checkout. See the
> [documentation index](../README.md) for current API and methods guidance.

Validated on 2026-09-27 using `sub-07/ses-nsd10`. The checks establish numerical
and workflow correctness on synthetic data and a bounded NSD subset; whole-brain
runtime and scientific performance have not been evaluated.

This historical real-data audit used normalized trial columns and
candidate-regularized targets. Current fractional fitting uses raw coefficient
norms and fixed OLS targets; see the [adoption record](fractional-defaults-2026-09-28.md)
and [current methods](../user-guide.md#fractional-ridge-at-each-grayordinate).
The measurements below have not been rerun under those newer defaults.

## Automated checks

The combined checkout passed **680 tests**, with warnings treated as errors:
553 core tests in 36.22 seconds and 127 NSD example tests in 91.70 seconds.
The user's existing stop-signal notebook/test fixes were included for validation
and preserved separately from these feature commits.

```sh
MPLCONFIGDIR=/private/tmp/ridge-mpl-cache uv run --no-cache --no-sync \
  pytest tests examples/NSD -q -W error -p no:cacheprovider
```

The two test directories were run separately. Tests cover independent numerical
solutions for coefficient-norm fractions, fraction 1 versus OLS, unequal design
column norms, response scaling, distinct alphas across targets and runs,
unpenalized nuisance terms, and native beta units. Selection tests cover
different optimal fractions at different features, negative scores, ties,
missing candidates, undefined task responses, and entirely invalid blocks.

Workflow tests reconstruct the nested scorer independently, perturb outer-test
runs, compare spatial blocks and worker counts, and inspect exported maps and
provenance. Executed notebooks cover fractional CV, shared-alpha CV, fixed ridge,
and OLS, including older configuration overrides. Package `__init__.py` files
remain empty; touched Python passes Black and `git diff --check` passes.

## Real NSD check

The run used all 12 CIFTI runs and 750 presentations from
`/Volumes/extdata1/NSD/BIDS/sub-07/ses-nsd10`, with fMRIPrep 25.2.5 confounds.
It fitted the first eight grayordinates in blocks of four with two workers.
Optimized fits used the complete canonical-plus-512 Sobol HRF library (seed 0).
Both HRF modes evaluated fractions `[1, 0.7, 0.3]`, with separate odd/even outer
evaluations and all-run final fitting.

All eight locations chose fraction 0.3 in both final fits. Their implied alphas
differed across grayordinates. These few locations and three candidate fractions
do not establish an appropriate production grid; the notebook retains its
preselected fractions 0.1 through 1.0. Outer-test scores did not inform changes
to the grid or algorithm.

The run published 181 artifacts under `/private/tmp/nsd-fractional-cv-smoke`.
Every CIFTI retained the full 91,282-grayordinate BrainModel axis; unprocessed
locations were NaN, or zero in eligibility masks. Outputs include fraction maps,
per-run alpha maps, candidate scores and fold losses, outer predictions and
targets, final beta series, the HRF library, and linked tuning provenance.

Fitting took 37.4 seconds for canonical HRFs and 24.3 seconds for optimized HRFs.
Different cache and process-start conditions make these timings unsuitable for
a speed comparison. Fitting, publication, and the independent audit together
took 450 seconds. Parent-process peak RSS was 4.72 GiB; worker memory is excluded.
Full-axis export buffers are allocated even when only eight locations are fitted.

## Independent reconstruction of saved outputs

The audit used augmented least squares and scalar root finding, independently
of the production SVD/bisection solver. Starting from original BOLD, confounds,
and saved HRF assignments, it reconstructed trial betas and verified their
coefficient-norm fractions in the nuisance-projected, normalized trial basis.

It independently refitted encoding models and verified saved candidate R²,
fold SSE/SST, selected fractions, outer coefficients/predictions/targets,
per-run implied alphas, and final beta maps. Final provenance linked each block
to its saved all-run selection decision. Both HRF modes passed.

Norm ratios used relative tolerance `2e-7` and absolute tolerance `1e-8`.
Saved R² used absolute tolerance `1e-7`; fold losses used relative tolerance
`2e-7` and absolute tolerance `1e-4`. Float32 beta/prediction exports used
relative tolerance `2e-6` and absolute tolerance `1e-4` in native signal units.

The temporary verification scripts and logs were not kept; see
`examples/validation/` for the retained fractional-ridge validation scripts.

The objective predicts candidate-regularized responses, whose values change
with the fraction. It is not a measure of recovery of a common, unobserved
ground-truth beta series. Final all-run RT correlations remain descriptive.

## Independent code review

The final review found no correctness blockers and independently passed the
20 solver/CV tests. It left two minor follow-ups: an additional fractional
provenance regression for different settings yielding identical selected maps,
and updating the NSD adapter's legacy global-alpha docstrings. The user guide,
API reference, README, and notebook describe both regularization modes.
The exported tuning and final-fit identities were checked in the existing tests
and real-data audit.
