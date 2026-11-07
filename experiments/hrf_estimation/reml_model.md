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

## Authorized implementation: three stability extensions

The user authorized all three proposed extensions. Expose separately named variants `remlar1`, `remlcomponents`, `remlshared`, and `remlcombined`, retaining `reml` unchanged.

AR(1): estimate a single robust training-fold coefficient as the median lag-one residual correlation over valid run/vertex pairs, clipped to [-.8,.8]. Residualize against C plus a fixed three-HRF task/junk basis (fast, canonical-like, slow); this is a noise-estimation device fixed before HRF selection, not candidate-dependent REML fixed effects. Record rho, estimation run IDs, and valid count. Prewhiten y, C and all random designs using W y = [sqrt(1-rho²)y0, y1-rho*y0, ...]. The resulting fixed-effect space remains common to candidate HRFs. Candidate-independent transformation constants can be omitted within a variant; restricted likelihood values are not compared between noise variants.

Independent-run component model: separate condition (two columns), RT (one column), and junk (two columns) variances. Scaling conventions remain unit-peak event responses and RT seconds. Empty components are zero covariance.

Shared model: b_r = b_shared + u_r for the three valid-task columns; junk effects have only run-specific components. The shared component yields off-diagonal cross-run covariance. With separate components enabled, shared conditions and shared RT have separate variances too (five variance ratios total: independent conditions/RT/junk, shared conditions/RT). Without separate components, use the original scalar independent variance plus one shared-task variance.

Compute restricted likelihood with fixed nuisance removed, V = I + B diag(ratio_by_column) Bᵀ. Use exact low-rank determinant/Woodbury identities, analytic variance gradients, and explicit zero boundaries; validate against dense restricted covariance and numerical gradients. Optimize variance ratios with bounded L-BFGS-B in log1p scaled coordinates, two deterministic starts (zero and unit normalized variance), retaining the best finite point. Scaling changes optimization coordinates only; record ratios in original design units, convergence, boundaries, and top-two candidate likelihood gap. No shape pooling across held-out runs or sessions is introduced.

Ruling: one robust pooled AR coefficient reduces the number of estimated noise parameters and is common across candidates. Cost: it cannot represent vertex/run-specific temporal covariance or arbitrary resampling-induced noise. Treat this as an AR(1) sensitivity, not established whitening adequacy. Dense-reference tests and small VTC pilots precede any expanded queue launch.
