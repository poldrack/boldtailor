# Regularized AR(1) pilot

Results notebook: `../../regularized_ar1_results.ipynb`.

Two sessions (11, 16), 256 fixed matched vertices, two complementary six-run training halves. Estimate noise from all 1,750 VTC vertices within each training split. Compare pooled AR(1), approximate empirical-Bayes shrinkage, and 5-mm surface regularization. Existing original REML and library CV provide references on the same vertices and splits.

The local noise models reduce held-out residual autocorrelation but change partial task R² by less than 0.05 percentage points in these sessions. They do not resolve session 16's poor HRF shape stability. The estimated empirical-Bayes shrinkage is mild (median weights approximately 0.94–0.96). These are exploratory descriptive results, not a population-level test.

- `summary.tsv`: pooled predictive scores and median split-half HRF correlations, all five methods.
- `scores.tsv.gz`: matched-vertex transfer scores underlying the summary.
- `shape_stability.tsv.gz` and `mean_shapes.npz`: per-vertex shape agreement and mean normalized curves.
- `rho.tsv.gz`: unrounded and computationally grouped coefficients for all 256 pilot vertices.
- `shrinkage_strength.tsv`: weights and upper-bound frequencies over all 1,750 noise vertices.
- `whitening.tsv.gz`: held-out fixed-reference-design residual ACF before OLS / after GLS whitening, on the common mask.
- `binning_sensitivity.tsv`: comparison of 0.02 and 0.01 coefficient spacing in session 16, transfer direction 1.
- `pooled_reproduction.json`: exact reproduction of all prior pooled-AR(1) HRF choices for both sessions and directions.
- `provenance.json`: source fit and metadata hashes.

Primary run commands (set BLAS/OMP threads to 1):

```sh
uv run python -m experiments.hrf_estimation.regularized_pilot --session 11 --output experiments/hrf_estimation/outputs/regularized-ar1-pilot/session11
uv run python -m experiments.hrf_estimation.regularized_pilot --session 16 --output experiments/hrf_estimation/outputs/regularized-ar1-pilot/session16
uv run python -m experiments.hrf_estimation.regularized_report
```

All regularization is training-only and candidate-independent. The predictive score uses unwhitened OLS amplitudes for all methods. Whitening diagnostics refit a fixed flexible reference model by GLS with the training-derived coefficient; residual ACF reflects both whitening and regression projection and is not a calibrated whiteness test. Surface smoothing uses subject-specific anatomical fsLR meshes from fMRIPrep, but all motion inputs remain NSD-preprocessed confounds. Mesh-edge geodesics, VTC-only contributions, and area weighting approximate surface Gaussian smoothing; no BOLD or HRF smoothing is performed.
