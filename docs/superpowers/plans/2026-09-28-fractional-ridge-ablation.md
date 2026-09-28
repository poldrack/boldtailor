# Leakage-safe fractional-ridge ablation

**Goal:** Test whether fixed OLS targets, raw trial-column scaling, training-pooled
fractions, and training-only affine calibration improve known beta recovery.
User authorized execution on 2026-09-28. Production defaults stay unchanged.

**Architecture:** Experimental code under `examples/validation/`, with a numerical
fitting module and a separate generator/runner. Reuse production within-run
encoding and fraction root solving; verify experimental fits against independent
augmented least squares and block-diagonal oracles. Continue on the existing
feature branch; preserve all earlier working-tree changes.

## Contract

- Four runs tune using leave-one-run-out CV; two untouched runs evaluate.
- Full 2 × 2 × 2 ablation: candidate versus fixed-OLS targets; unit-column norm
  versus raw-column coefficient basis; per-run fraction versus pooled-training
  fraction with training-derived alpha frozen on validation/test runs.
- The pooled fraction is the coefficient norm ratio over all training runs.
  Apply its alpha to held-out runs without recalculating it from held-out BOLD.
  Record achieved held-out fractions, which need not equal the requested fraction.
- Per-run fractions reproduce the production baseline. A held-out run can define
  its own alpha for beta estimation, but cannot affect training alphas or encoding.
- Encoding always has run-specific training intercepts and centered scoring.
  Fixed targets use OLS betas; the same target is used for every candidate.
- Select one fraction per feature from [1, .8, .5, .2, .1], with ties within
  1e-12 favoring less shrinkage. No latent truth enters selection.
- After selection, compare raw estimates to affine calibration fitted only on
  training ridge/OLS beta pairs. Freeze this mapping before outer evaluation.
  Nonpositive scales or degenerate fits fall back to identity. This is a
  leakage-safe adaptation, not a claim of exact GLMsingle compatibility.
- Apply the same calibration to encoding predictions. Primary comparisons use
  latent beta RMSE, centered RMSE, amplitude projection, within-run correlation,
  slope error, and common fixed-OLS outer prediction scores. Report selected
  fractions, boundary rates, scale/offset and achieved held-out norm ratios.
- Generators: 6 runs × 24 trials × 3 features; ISI 4/12 s; uniform/variable
  durations; white-low/AR(1)-heteroscedastic-high noise; matched/mismatched HRF;
  run-offset increment 0/1. Ten seeds, 32 scenarios/seed. Features include
  strong and weak linear effects and an unmodeled nonlinear effect.
- Average scenarios/features within seeds before uncertainty across seeds.
  Compare paired changes against the production-equivalent baseline and report
  stratified results, tradeoffs, and limits. Do not choose a default from a pooled
  average alone. Archive configuration, candidates, selected rows and summaries.

## Tasks and checks

1. Write `tests/test_fractional_ridge_ablation.py` solver tests first: independent
   per-run and block-diagonal pooled augmented-regression oracles; fraction-one
   equivalence; raw versus normalized differences; training-only affine fit;
   invariance of training state to arbitrary held-out outcome/design changes;
   fixed targets independent of candidate; baseline agreement with production.
   Run RED and commit tests, then implement
   `examples/validation/fractional_ridge_ablation.py` and verify GREEN.
2. Write runner tests: deterministic generation, noise/HRF/duration stress,
   nested selections unaffected by outer outcomes or latent truth, complete
   per-feature exports, common scoring and paired summaries, noiseless recovery.
   Run RED and commit before implementing
   `examples/validation/fractional_ridge_ablation_simulation.py`.
3. Run ten-seed experiment; independently audit archived numeric summaries and
   write `docs/validation/fractional-ridge-ablation-2026-09-28.md`. Run the full
   pytest suite, Black and whitespace checks; obtain an independent code review.

All Python commands use `uv run`; tests are pytest functions/fixtures. No changes
to package `__init__.py` or production fitting. Log execution evidence in
`/private/tmp/fractional-ablation-progress.md`. Only this task's files are staged.

## Review focus

1. Outer responses or designs influencing training alpha, calibration or choices.
2. Incorrect coefficient basis or separate-alpha approximation to pooled norm.
3. Calibration evaluated on the same held-out outcomes that estimated it.
4. Candidate-dependent scores misrepresented as comparable scientific recovery.
5. Seed/scenario/feature pseudoreplication or hidden poor-performing subgroups.

## Execution complete

- Solver tests: RED committed `a64ce7d`; GREEN implementation `82e8754`.
- Runner tests: RED committed `3d2d92a`; GREEN implementation `4a5820f`.
- Ten seeds × 32 scenarios complete: 16,320 selected rows and 38,400 candidates.
- Independent review: no blockers. Report clarifies the realized-training-truth
  slope reference; independent numeric audit verifies aggregation across all
  scenarios and features within seeds before computing uncertainty.
- Full suite: 860 passed in 169.91 s with warnings as errors. Black and
  `git diff --check` passed.
- Findings and full archive: `docs/validation/fractional-ridge-ablation-2026-09-28.md`
  and its companion directory. Production defaults unchanged.
- Reproduce using module invocation (`uv run python -m
  examples.validation.fractional_ridge_ablation_simulation`); direct script-path
  invocation does not put the repository's `examples` package on the import path.
