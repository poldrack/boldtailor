# Descriptive NSD beta-series activation maps

Implemented in `5411cdc`, following failing tests committed in `c0982e1`.
The user explicitly accepted treating trial betas as independent observations.

Section 8d of the NSD workflow notebook pools finite trial betas across runs,
without centering, for a one-sample t-test against zero per grayordinate and
model. Mean, t, uncorrected two-sided p, trial count and df are saved on the
original CIFTI axis. Histograms and optional cortical surfaces are unthresholded.
The notebook and metadata describe the independent-trial assumption and the
model-relative baseline; this is not an explicit task-versus-rest contrast.
No core package APIs or dependencies changed.

Verification:

- RED: 11 failed, 16 passed before implementation (missing calculation,
  exports, surface statistic and notebook integration).
- Targeted GREEN: 34 passed in 23.60 s. Checks include SciPy agreement,
  unequal run lengths, finite observations, undefined/constant columns,
  large offsets, input preservation, CIFTI axes/map order, signed surface
  scales, and executed notebook exports for OLS and ridge/fractional models.
- Full suite: `uv run pytest -q -W error` — **896 passed in 159.67 s**.
- Black: 121 Python files pass; notebook validates and compiles with no
  stored outputs. All initializers remain empty; `git diff --check` passes.
- One independent read-only review approved `48f3c9a..5411cdc` with no
  findings and independently ran 15 numerical/surface tests successfully.

The reviewer did not rerun the full suite or real NSD data. Calibration of
p-values under dependent trial estimates is outside this explicitly descriptive
analysis; trial covariance, shrinkage and selection uncertainty are ignored.
Zero sample variance or fewer than two finite trials yields NaN t/p;
grayordinates with no finite trials have NaN in every exported map.

The calculation processes one run at a time and centers before summing squared
deviations, avoiding a full-session beta copy and unstable subtraction of raw
moments. Both notebook computation and export reuse existing fits.
