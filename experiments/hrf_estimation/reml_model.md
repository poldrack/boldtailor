# Direct REML HRF-library selection

The user selected direct REML with random task effects on 2026-11-07. The following covariance is the initial experimental model; all amplitude refitting for final R² remains ordinary least squares.

For vertex v and run r, let y_r = C_r a_r + Z_r(h) b_r + e_r. C contains the intercept, 24 motion terms, and 80-second cosine drifts. Z contains the two valid condition columns, common RT column, missing-response column, and incorrect-response column, all convolved with h. b_r ~ N(0, tau² I), e_r ~ N(0, sigma² I), independently across runs. The variance ratio lambda=tau²/sigma² and sigma² are shared across the training runs at a vertex; a_r and b_r are run-specific. Empty junk columns contribute no covariance. This initial task-level model does not add single-trial random deviations; that is a separate future sensitivity analysis, not a hidden extra parameter family.

This is a proper zero-mean Gaussian amplitude prior for HRF selection, not a claim that each fitted task effect is zero. Condition/junk event responses have the package's unit-peak convention; the RT predictor is in numeric seconds (a fixed one-second reference unit). A shared variance therefore regularizes coefficients on that explicitly declared scale. Alternative component variances and temporally correlated noise are sensitivities, not part of the initial selector. The same normalizations apply to every candidate.

Project out the common C space using an orthonormal basis. For each run write the projected random design as U S Vᵀ; c=Uᵀ y_projected, R=||y_projected||²-||c||², d=n-rank(C). Across runs let N=sum(d), Q(lambda)=sum_r[R_r+sum_i c_ri²/(1+lambda*s_ri²)]. Up to constants common to candidates, the profiled negative twice restricted log likelihood is:

    N * log(Q(lambda)/N) + sum_r,i log(1 + lambda*s_ri²)

The profiled sigma² is Q/N. C and its rank are candidate-independent. Its determinant terms are therefore constant across candidate HRFs; no candidate-dependent fixed-effect determinant is discarded. Random-design singular values and the full restricted residual dimension remain in the objective, including residual dimensions outside Z.

Search lambda=0 explicitly, plus a log grid spanning lambda*s² from approximately 1e-8 to 1e8 and vectorized bounded golden-section refinement near the best grid point. Record upper-grid hits and lambda=0 boundaries. A zero-random-variance optimum cannot identify an HRF and is flagged `variance_zero`; zero nuisance-residual energy is flagged `no_signal`. Equal finite objectives use earlier library order. Training runs alone determine candidates, variances, and validity.

Before real-data fitting, tests compare this objective with direct dense covariance calculations in the restricted space, check invariance to invertible nuisance reparameterization, variance-zero behavior, and planted HRF recovery with added noise. This is a task-level adaptation of the neighboring prototype's scalar variance-ratio machinery, not an unchanged use of its single-trial model.

## Candidate stability extensions for the next stage

These are proposed sensitivities; the matched baseline remains the scalar independent-run random-effect model above.

1. **Training-only temporal covariance.** Estimate a modest AR noise model from outer-training data using a common, HRF-independent policy, freeze it across candidate HRFs, and transform both nuisance and event designs consistently. Estimate again within each inner training fold if tuning. Keep unwhitened transfer R² as the common endpoint. This addresses the independent-noise assumption; temporally resampled NSD errors need not be exactly AR(1). SPM documents serial-correlation treatment during ReML estimation: https://www.fil.ion.ucl.ac.uk/spm/docs/manual/fmri_spec/fmri_spec/.
2. **Separate component variances.** Replace tau² I by prespecified task/RT/junk blocks, with scales declared independently of candidate HRF. This prevents junk/RT coefficient scale from setting all shrinkage. Added variance components can themselves be unstable, so use simulation recovery and boundary frequencies to decide whether the extension helps.
3. **A shared task component across training runs.** A hierarchical random-effects extension can model b_r = b_shared + u_r using one shared covariance component plus block-diagonal run-deviation covariance. It keeps C HRF-independent and represents consistent nonzero task responses without treating HRF-dependent task columns as unpenalized fixed effects in a direct REML comparison. This is a model change, not a numerical fix.
4. **Weak shape shrinkage or candidate averaging.** Stabilize near-tied HRF candidates with a declared reference/prior and tune its strength within training folds. Pooling a reference across the very sessions/halves used to assess stability would contaminate that assessment; any cross-session training requires a separate held-out-session design. Averaging shapes would also need an explicit kernel representation and an independently labelled estimator.

Priority: diagnose likelihood flatness and variance boundaries, then test temporal covariance; add only a small number of component variances before considering shape pooling. Stability must be evaluated alongside transfer fit and recovery bias: shrinkage can improve correlation simply by making every HRF similar.
