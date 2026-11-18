# Developer guide

[User guide](user-guide.md) · [API reference](api.md) · [Code-review guide](code-review.md) · [Testing guide](testing.md)

This guide describes the architecture, module boundaries, and numerical
conventions used when extending Boldtailor.

## Environment and checks

Use Python 3.12 or later and `uv`:

```bash
uv sync --group dev
# During edits: affected module tests, then the integration suite
uv run pytest tests/test_reml*.py -q -W error
uv run pytest -m integration -q -W error
# Before merging
uv run pytest -q -W error --durations=20
uv run black --check src tests examples/bids_workflow
uv run git diff --check
```

See [Choosing tests during development](testing.md) for module-to-test mappings
and opt-in suites. `uv run pytest` runs the package suite in `tests/`. Example
and notebook tests are opt-in:
- `uv run pytest examples/bids_workflow`
- `uv run pytest examples/validation`
- `uv run pytest --run-notebooks examples/bids_workflow examples/validation`

Notebooks are tracked as `jupytext` `py:percent` scripts paired with gitignored
`.ipynb` files (`jupytext.toml`). If you edit an `.ipynb` outside a
jupytext-enabled Jupyter server, sync it back with `uv run jupytext --sync <notebook>`.

To verify the built wheel in an isolated environment without editable-install or
dev dependencies:

```bash
uv build --wheel
uv run --isolated --no-project --with ./dist/boldtailor-0.1.0-py3-none-any.whl python tests/check_installed_package.py
```

### Structural rules

- Keep functions short and separate numerical work from file I/O.
- The top-level `boldtailor/__init__.py` exports the primary public entry points
  and nothing else; `boldtailor/workflow/__init__.py` is empty.
- A leading underscore on a function or class means module-private: symbols
  imported across modules have no leading underscore, and shared internal
  modules are named `_<module>.py`. `tests/test_import_structure.py` enforces an
  acyclic import graph, no cross-module imports of `_private` symbols,
  module-level package imports outside `cli.py`, and no re-export facades in
  `workflow`.
- Follow RED-GREEN-Refactor: write and commit failing pytest tests before
  implementing behavioral changes, and give every test function a short purpose
  docstring.

## Code organization

| Area | Modules |
| --- | --- |
| Owned arrays, event tables, timing, and source records | `data`, `_run_inputs`, `_arrays`, `_scalars`, `provenance` |
| Event-based conventional GLMs | `model`, `design`, `_task_design`, `fit`, `_standard_glm`, `_conventional`, `results` |
| Fixed designs supplied by callers | `prepared`, `prepared_fit` |
| Shared conventional/prepared diagnostics | `_fit_diagnostics`, `model` |
| Task-model detection from event tables | `task_detection` |
| Encoding and penalty selection | `trial_encoding`, `ridge_selection`, `fractional_ridge`, `_regularizers`, `_ridge_cv`, `_fractional_ridge`, `ridge_results` |
| Trial design construction and OLS/ridge estimation | `single_trial`, `_shared_hrf_fit`, `_design_inputs`, `_single_trial_design`, `_single_trial_fit`, `single_trial_results` |
| Candidate HRFs, CV/REML selection, evaluation, and grouped fits | `hrf_library`, `hrf_options`, `hrf_selection`, `hrf_results`, `_hrf_design`, `_run_design`, `_hrf_cv`, `_hrf_reml`, `_selected_hrf_fit` |
| Conventional GLMs using voxelwise HRFs | `_hrf_glm_design`, `_hrf_glm`, `hrf_glm_results` |
| Scalar REML and AR(1) whitening/autocorrelation | `reml`, `ar1`, `_ar1_moments`, `_numerics` |
| Task-guided (GLMdenoise-style) denoising | `denoising`, `denoising_results`, `_mixture_threshold` |
| Provenance, lifecycle logging, and transactional publication | `provenance`, `logging`, `_fit_lifecycle`, `bids_provenance`, `publication` |
| Shared constants | `_constants` |
| CIFTI scalar I/O, descriptive diagnostics, HRF agreement, parallel blocks | `cifti`, `diagnostics`, `reliability`, `parallel` |
| Session workflow and `boldtailor run` | `workflow` (`settings`, `inputs`, `files`, `analysis`, `beta_series`, `records`, `ridge_provenance`, `hrf_maps`, `run`, `outputs`, `export_cifti`, `figures`, `tables`, `manifest`, `artifacts`, `summaries`, `plots`, `surfaces`, `report`), `cli` |
| Notebook-only BIDS workflow helpers | `examples/bids_workflow` |
| Research and simulation scripts (not shipped in wheel) | `experiments/` (`nsd_replication`, `hrf_estimation`, `hrf_simulations`) |

Public fitting functions delegate to complete orchestration modules:

| Public entry point | Orchestration implementation |
| --- | --- |
| `fit.fit` | `_standard_glm.fit_standard_glm`, `_hrf_glm.fit_selected_glm` |
| `fit.task_delta_r2` | `_standard_glm.standard_task_delta_r2`, `_hrf_glm.selected_task_delta_r2` |
| `single_trial.fit_single_trials` | `_shared_hrf_fit.fit_shared_hrf_trials` |
| `single_trial.fit_selected_hrfs` | `_selected_hrf_fit.fit_selected_hrf_trials` |

Each orchestration function owns input validation, numerical fitting, the
`fit_operation` logging scope, provenance construction (`build_fit_provenance`
or `finish_fit`), and result assembly. Low-level numerical helpers do not manage
logging scopes.

### Array ownership and dataclass equality

Public numeric arrays are owned NumPy arrays marked read-only via
`_arrays.readonly_array` (`float64` for floats, `int64` for indices, `bool` for
masks). Construction copies inputs; attribute access does not copy again. Public
DataFrame and dictionary accessors return defensive copies. Every dataclass that
holds a NumPy array or pandas DataFrame sets `eq=False` (comparing by object
identity), enforced by `tests/test_dataclass_equality.py`.

`SingleTrialResult.design` is either a `SharedTrialDesign` (holding per-run
DataFrames) or a `SelectedTrialDesign` (rebuilding `(run, hrf_id)` design arrays
on demand via `rebuild_matrix(run, hrf_id)`). `HrfAnalysisResult` similarly
rebuilds grouped conventional designs on demand via
`rebuild_group_design(run, hrf_id)`. Use `materialize_designs()` or
`materialize_group_designs()` to allocate all fitted designs explicitly.

## Numerical conventions

### Conventional GLMs and $\Delta R^2$

- Conventional fits use Nilearn's OLS/AR(1) estimator and t contrasts, fitting
  each run independently and combining contrasts via equal-weight fixed effects.
- Rank-deficient designs are projected onto an orthonormal basis of their column
  space (`_conventional._full_rank_basis`) so residual degrees of freedom equal
  `n - rank`. Rank and contrast estimability checks normalize nonzero columns to
  unit L2 norm before SVD thresholding so column scaling never changes rank or
  degrees of freedom.
- `task_delta_r2` and `task_delta_r2_prepared` refit full and nuisance-only
  models with OLS (`_conventional.nested_ols_delta`), clipping tiny negative
  roundoff within `-NESTED_OLS_TOLERANCE` to zero in `delta_r2` while preserving
  `raw_delta_r2`.

### Single-trial estimation and ridge CV

- Single-trial ridge projects trial columns and signals off each run's nuisance
  span (supplied confounds plus intercept).
- Normalized ridge (`ridge_alpha`, `PreparedTrialBetas` in `_single_trial_fit`)
  scales projected trial columns to unit L2 norm before applying `ridge_alpha`
  and restores coefficients to native signal units.
- Fractional ridge (`ridge_fraction`, `PreparedFractionBetas` in
  `_fractional_ridge`) checks rank on unit-norm columns but decomposes the
  unscaled projected trial columns so the L2 norm ratio
  $\|\hat{\beta}(f)\|_2 / \|\hat{\beta}_{\text{OLS}}\|_2 = f$ holds in the raw
  trial coefficient basis.
- `_regularizers.py` defines `NORMALIZED_RIDGE` and `FRACTIONAL_RIDGE` strategies
  shared by `_ridge_cv.py` and `workflow/beta_series.py`. Within each fold,
  `trial_encoding.EncodingDesign` prepares the predictor design once and scores
  all candidate penalty grids. Fractional CV scores against fixed OLS validation
  betas; normalized-alpha CV scores against candidate-regularized validation
  betas.

### HRF libraries and CV selection

- Candidate 0 in every library is Nilearn's canonical SPM kernel (32 s support)
  scaled to unit peak. Custom candidates have 36 s support (or 50 s for the 20
  empirical GLMsingle curves).
- Every event's predicted response on the oversampled grid (`TR / 50`) is scaled
  to peak at 1 (`HRF_NORMALIZATION = "peak_one_event_response"`).
- Under `method="cv"`, `select_hrfs` pools leave-one-run-out prediction errors
  for the `TaskModel` regressors against confound-adjusted signals, checking
  candidate rank and residual degrees of freedom lazily on proposed winners
  (`_hrf_cv.choose_eligible`).

### REML conventions

`boldtailor.reml` fits, per feature, the linear mixed model
$$y_g = C_g a_g + Z_g b_g + e_g, \quad b_g \sim \mathcal{N}(0, \tau^2 I), \quad e_g \sim \mathcal{N}(0, \sigma^2 R_g(\rho))$$
across groups (runs) $g$, where $C_g$ is `fixed_design`, $Z_g$ is
`random_design`, and $R_g(\rho)$ is the stationary AR(1) correlation matrix. The
variance ratio $\lambda = \tau^2 / \sigma^2$ and marginal noise variance
$\sigma^2$ are shared across groups.

- **AR(1) whitening and estimation**: `ar1.whiten` normalizes rows so
  $W R(\rho) W^\top = I$, preserving the interpretation of $\sigma^2$ and
  $\lambda$ across $\rho$. `ar1.estimate_rho` corrects residual lag-1
  autocorrelation for fixed-design projection bias by inverting the monotone
  branch of the expected lag-1 curve across `rho_range` (default `(-0.2, 0.95)`).
- **Scale-invariant fixed projection**: `_numerics` normalizes nonzero columns
  of $C_g$ to unit norm before computing its orthonormal basis with tolerance
  $\max(\text{shape}) \cdot \varepsilon \cdot \sigma_{\max}$. Residual signal
  energy is checked against two float64 floors ($8 n \varepsilon$ times the
  unamplified and condition-amplified signal norms):
  - At or below the rounding floor: `status = "no_residual_signal"` (all
    estimates NaN).
  - Between the rounding and conditioning-uncertainty floors:
    `status = "numerically_unresolved_residual"` (all estimates NaN).
- **Identifiability**: When the retained random directions span the entire
  restricted space with equal pooled singular values (within
  $4 \cdot \text{dof} \cdot \varepsilon$), $\tau^2$ and $\sigma^2$ are
  confounded (`status = "unidentifiable"`, `ratio` and variances NaN,
  ineligible to win selection).
- **Criterion and optimization**: The profiled $-2$ restricted log-likelihood is
  evaluated from pooled singular values and squared coordinates across groups
  (using $y - U(U^\top y)$ directly to avoid catastrophic cancellation at high
  SNR). Optimization scans a log-spaced grid over
  $[10^{-8} / \max s^2, \, 10^8 / \min s^2]$, refines sign changes of the
  analytic derivative by bisection, and requires a positive optimum to beat
  $\lambda = 0$ by more than $64 \varepsilon (|F| + \text{dof})$ (otherwise
  `status = "variance_zero"`, $\lambda = 0$). Grid optima without a verified
  derivative root set `unrefined = True`, and optima at the upper grid edge set
  `at_upper = True`.
- **Selection**: `select_random_effects_design` chooses the `"ok"` candidate
  with the lowest criterion at fixed $C_g$ and per-feature $\rho$ (`winner = -1`
  if no candidate is `"ok"`). `any_at_upper` and `any_unrefined` flag features
  where any candidate (winner or loser) hit the upper bound or remained
  unrefined.

## Provenance, logging, and file publication

- **Fingerprints**: `SourceRef` and `RunSources` describe input files without
  reading them. `ProvenanceRecord.data_fingerprint` hashes complete source
  metadata (including optional `sha256`), and
  `ProvenanceRecord.analysis_fingerprint` hashes `data_fingerprint` together with
  the model or activity configuration. `ProvenanceRecord.latest_activity` returns
  an owned copy of the most recent activity dictionary.
- **Lifecycle logging**: Entry points use `_fit_lifecycle.fit_operation(name, parent)`
  (and `logging.normalization(stage)` for input construction) to emit JSON
  `*_started`, `*_completed`, or `*_failed` events to the `boldtailor` logger.
  Failure events log a categorical `error_code` (`invalid_input`,
  `numerical_failure`, `io_failure`, `operation_failed`) without exporting
  exception messages, and re-raise the original exception.
- **Transactional publication**: `publication.publish_artifact_set` acquires a
  lock in `<destination>.boldtailor/`, stages new files in `stage-<uuid>/`,
  moves overwritten and stale manifest-listed files to `backup-<uuid>/`,
  `os.replace`s and `fsync`s each file, and restores backups on failure.

## Dependency notes

- `pyproject.toml` pins `nilearn>=0.14.0,<0.15` because `_hrf_design` and
  `tests/oracles.py` import Nilearn's `_sample_condition` for peak-normalized
  oversampled boxcar scaling. Re-verify against `tests/oracles.py` before
  widening the upper bound.
