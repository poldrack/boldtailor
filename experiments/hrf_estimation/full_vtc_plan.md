# Full-VTC continuous comparison and sensitivity execution

User authorization: proceed with full-VTC continuous optimization and extended noise/RT checks; investigate REML stability improvements.

Scope: sessions 11 and 12, all 1750 anatomical fusiform vertices, all 12 runs, the existing seed 20261107 and one complementary 6/6 partition. This is a matched extension of the completed pilots, not yet the ten-session repeated-split study.

Primary: canonical, full-library CV/REML, parametric-only library CV/REML, continuous L-BFGS-B and Powell; retain identical training/test IDs and ordinary unwhitened transfer scoring. Continuous tuning retains all training vertices and leave-one-run-out inner CV, lambda grid 0/.001/.01/.1/1, four starts, 500 iterations and 5000 evaluations per start. RT amplitude is primary; RT duration is a separately labelled full-ROI sensitivity. No outer-test-dependent tuning.

Execution steps:
1. RED/commit/GREEN: cached design and nuisance-projected objective; compare with explicit original least squares including duration RT, collinear nuisance, and noisy data. Cache library initialization across vertices; parallelize independent feature optimizations with bounded workers.
2. RED/commit/GREEN: reproducible simulation generation with known HRFs, condition-correlated RT, motion/drift and junk, independent or AR(1) noise, amplitude/duration ground truth; retain trial and signal seeds.
3. Launch restartable method/session/RT jobs with progress logs and immutable settings. Include full and parametric libraries in the same manifest-based comparison system. Run simulation checks separately and report bias, transfer, convergence, and timing without asserting a winning method.
4. Update results notebook when completed; retain output data in the current checkout. Report pending jobs honestly.

REML follow-up proposals (not silently added to baseline): candidate-independent training-only temporal covariance; separate task/RT/junk variance components with fixed scaling; weak HRF-shape shrinkage tuned only inside training folds. Pooling across an evaluated split/session is excluded from stability assessments. Diagnose near-tied candidate objectives and zero-variance cases before interpreting instability.

Ruling: retain sessions 11–12 for this stage because these are the completed full-ROI pilots and provide matched baselines; the cost is limited between-session generalizability. Ruling: retain the original OLS transfer endpoint for all comparisons; noise simulations probe misspecification rather than silently changing the estimator.

## Progress and current data availability

- Cached event sampling, batched initial-library scoring, and bounded process parallelism implemented. Tests compare amplitude/duration designs and noisy profile losses against explicit original least squares.
- Independent review identified projection-roundoff rank loss and bypassed timing validation. Added failing tests first and fixed both; near-degenerate nuisance overlap falls back to the original full-design least-squares solve.
- Full-VTC runner defines 20 restart units: 2 sessions × 2 RT formulations × (full-library baselines, parametric CV, parametric REML, L-BFGS-B, Powell). All use identical seeded complementary splits. Restarts are per completed method/session/formulation directory, not partial optimizer checkpoints.
- Synthetic sensitivity grid: 4 true shapes (canonical, fast, off-grid, empirical) × 2 generating RT models × 2 fitted RT models × AR coefficients 0/.4/.7 × SNR .5/2 × 3 noise seeds = 288 cases. Each case compares 7 methods using training runs 1–2 and held-out runs 3–4, with the full 533-candidate library and unchanged continuous budgets/tuning. Fixed synthetic NSD-like timing includes 24 motion terms, 80 s drift terms, junk trials, and condition-correlated RT. These are simulations, not reconstructed NSD events. Three seeds provide an initial sensitivity screen, not precise Monte Carlo uncertainty.
- Ruling: synthetic timing permits independent checks while the external data drive is unavailable; cost: simulation conclusions cannot capture all properties of the actual NSD events/noise.
- Data blocker: `/Volumes/extdata1/NSD` is absent (only Macintosh HD and Recovery mounted). Asked user to mount drive or supply new path. No full-VTC optimization has been launched.

## Commands

Run from the repository root with the existing uv environment. Preserve output directories for reproducibility.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 UV_CACHE_DIR=/tmp/boldtailor-uv-cache uv run --no-sync python -m experiments.hrf_estimation.full_vtc --bids-root /Volumes/extdata1/NSD/BIDS --output experiments/hrf_estimation/outputs/full-vtc-continuous --workers 4
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 UV_CACHE_DIR=/tmp/boldtailor-uv-cache uv run --no-sync python -m experiments.hrf_estimation.sensitivity_runner --output experiments/hrf_estimation/outputs/noise-rt-sensitivity
```

`--job 0..19` or `--case 0..287` runs a single restart unit. Keep the code/settings fixed during a run; manifests reject reuse after scientific code changes. The output hierarchy is under the main checkout, so notebooks will find it locally.

Update: user reports the NSD drive mounted again. Both sessions discovered with 12 matched runs; proceed with empirical jobs after final code commit. Independent review found no remaining important issues after the two cached-objective fixes. Simulations use four synthetic runs and a single 2/2 split; their scores are not directly comparable with twelve-run empirical 6/6-transfer scores.
