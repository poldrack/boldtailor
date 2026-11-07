# REML extension validation

This is an execution/recovery validation on **four deterministic anatomical vertices**, not a full-VTC stability ranking. Sessions 11 and 12 each use all 12 runs, the full 533-candidate library, and the same complementary 6/6 run splits as the main experiment. The original REML, AR(1), separate-component, shared-task, and combined variants each produce an all-run fit and two transfer fits.

All 120 vertex/evaluation records have `ok` status. Every selected extended-model variance fit converged. Output artifact hashes were verified for both completed session directories. The experiment suite passes 100 tests, including dense restricted-covariance agreement, analytic-gradient checks, reduction to original scalar REML, zero/inactive boundaries, noise calibration, and full harness output tests.

The validation does not demonstrate a consistent gain: median partial task transfer R² changes vary by session. The four sampled vertices cannot establish VTC-wide stability or performance. The full-ROI comparison and 288-case synthetic sensitivity queues are now running from fixed code revision c79a2a5.

## Median partial task transfer R² (four common vertices)

| Method | Session 11 | Session 12 |
|---|---:|---:|
| reml | 0.1354 | 0.1021 |
| remlar1 | 0.1383 | 0.0888 |
| remlcomponents | 0.1350 | 0.0927 |
| remlshared | 0.1370 | 0.1021 |
| remlcombined | 0.1389 | 0.1004 |

## Artifacts and running jobs

- `outputs/reml-final-session11` and `outputs/reml-final-session12`: completed four-vertex validation, fixed c79a2a5 code checkout.
- `reports/reml_validation_summary.tsv`, `reml_validation_stability.tsv`, `reml_validation_diagnostics.tsv`: pooled transfer metrics, descriptive curve/timing stability, and selected-model diagnostics.
- `outputs/full-vtc-reml.log`: full-ROI primary RT-amplitude comparison, sessions 11 and 12, original REML plus all four variants.
- `outputs/noise-rt-reml.log`: 288-case known-truth grid with both generating/fitted RT formulations, original REML plus all four variants. Synthetic four-run 2/2 scores must not be interpreted as empirical twelve-run 6/6 benchmarks.
- `outputs/reml-launch.json`: exact commands, PIDs, output locations, and fixed execution checkout.

The earlier `reml-validation-session*` directories are incomplete pre-fix attempts and must not be used. The original baseline/L-BFGS-B queues continue under `full-vtc-lbfgsb` and `noise-rt-lbfgsb`. Powell is excluded from all active queues.

AR estimation uses one pooled training-only coefficient, limiting its ability to represent run/vertex-specific noise. Shrinkage can increase shape similarity while adding bias; stability and transfer fit must be interpreted together. An extreme custom-variance-bound numerical-floor limitation is documented in `full_vtc_plan.md`; current runs use the tested default bound.
