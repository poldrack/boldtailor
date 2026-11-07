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
