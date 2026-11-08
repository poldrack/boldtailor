# Scalar AR(1) REML validation

The AR(1)-only extension now uses a vectorized one-component restricted-likelihood solver. It diagonalizes the task Gram matrix once per HRF, evaluates all vertices together, and performs a bounded grid search followed by golden-section refinement. Exact zero and upper boundaries are included. The normalized variance bound, noise estimation, whitening, common nuisance projection, covariance model, and inactive/no-signal handling are retained. The multiple-component variants keep their existing generic optimizer.

On a paired session-11 benchmark with all 12 runs, 128 fixed anatomical VTC vertices and 32 evenly spaced library candidates, the old generic solver took 10.282 s and the scalar solver took 0.707 s (14.55x faster). All selected HRFs matched; maximum absolute restricted-likelihood difference was 5.46e-12.

A full session-11 validation used all 1,750 VTC vertices and all 533 HRFs. Fitting took 37.995 s; all 1,750 winners matched the previously saved generic AR(1) fit, the estimated noise metadata matched exactly, and the maximum likelihood difference was 5.46e-12. The earlier generic fit took 2,372.3 s (39.5 minutes), although that historical timing was collected under a different concurrent workload. Use the paired subset for a controlled timing comparison.

Summary: `reports/reml_scalar_validation.json`. Raw benchmark script, diagnostics and full-VTC HRFs: `outputs/reml-scalar-benchmark*`. Source revision: `f3b95bf`. Tests cover spectral versus generic likelihoods, optimized criteria and ratios, exact zero/upper boundaries, no signal, inactive covariance, and AR(1) dispatch bypassing the generic optimizer. All 117 experiment tests passed.

All previously ongoing full-VTC and REML analysis queues were terminated at the user's request on 2026-11-08. Saved results remain intact. Only the bounded validation benchmarks above were run; the full queues have not been restarted. User edits to both notebooks were preserved.
