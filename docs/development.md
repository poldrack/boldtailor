# Developer guide

For running an analysis, start with the [user guide](user-guide.md).
This page describes the implementation conventions used when extending Boldtailor.
For a guided reading of the implementation, use the [human code-review guide](code-review.md)
and its [complete module/function inventory](api-review/README.md).

## Environment and checks

Use Python 3.12 or later and uv:

```bash
uv sync --group dev
# During edits: affected module tests, then the integration suite
uv run pytest tests/test_reml*.py -q -W error  # example area
uv run pytest -m integration -q -W error
# Before merging
uv run pytest -q -W error --durations=20
uv run black --check src tests examples/bids_workflow
uv run git diff --check
```

See [Choosing tests during development](testing.md) for module-to-test mappings,
integration coverage, and when to run broader checks. Run affected module tests
plus the integration suite before committing; the full default suite is required
before merging and for test-selection changes.

`uv run pytest` runs the package suite in `tests/`. Example and notebook tests are
opt-in: `uv run pytest examples/bids_workflow`, `uv run pytest examples/validation`, and
`uv run pytest --run-notebooks examples/bids_workflow`.
Tests use synthetic arrays and small generated imaging fixtures; the example
data on the external NSD volume are not required for the test suite.
Notebooks are tracked as jupytext py:percent scripts and paired with gitignored
`.ipynb` files by `jupytext.toml`; notebook tests read the `.py` sources. After
editing an `.ipynb` outside a jupytext-enabled Jupyter server, run
`uv run jupytext --sync <notebook>` so the change reaches the `.py`.

CI splits the suites. On every push and pull request it runs the package
suite in `tests/` with warnings treated as errors, uploads its line coverage
to Codecov (the README badge), and checks Python formatting on a clean
Python 3.12 runner. A separate `examples` job, run only on manual dispatch and
on schedule, runs `examples/bids_workflow` and `examples/validation` with
`--run-notebooks` and warnings as errors. The push job also builds and tests
the installed wheel in an isolated environment, without the checkout's
editable installation or notebook development dependencies:

```bash
uv build --wheel
uv run --isolated --no-project --with ./dist/boldtailor-0.1.0-py3-none-any.whl python tests/check_installed_package.py
```

The smoke check verifies an installed-package import, absence of `ipykernel`,
and a synthetic OLS fit against NumPy least squares. Update the wheel filename
when changing the package version. Notebook execution uses `ipykernel` from the
development dependency group, not the library's runtime requirements.

Software versions are resolved when provenance is created. An uninstalled
Boldtailor checkout reports its own version as `unknown`; missing dependency
metadata still raises an error rather than inventing a version.

Keep functions short and separate numerical work from file I/O. The top-level
`__init__.py` exports the primary public names and nothing else; the workflow
initializer stays empty. A leading underscore means module-private: a name
used by another module has no underscore, and shared internals live in
`_private` modules. `tests/test_import_structure.py` enforces this, an
acyclic import graph, and module-level imports outside the CLI. For behavioral changes, write and run failing
pytest tests, commit the tests, then implement the change. Use numerical
oracles that exercise the scientific behavior rather than tests that duplicate
implementation details. Repository-specific rules are in [AGENTS.md](../AGENTS.md).

## Code organization

| Area | Modules |
| --- | --- |
| Owned arrays, event tables, timing, and source records | `data`, `_run_inputs`, `_arrays`, `_scalars`, `provenance` |
| Event-based conventional GLMs | `model`, `design`, `_task_design`, `fit`, `_standard_glm`, `_conventional`, `results` |
| Fixed designs supplied by callers | `prepared`, `prepared_fit` |
| Shared conventional/prepared diagnostics | `_fit_diagnostics`, `model.contrast_metadata` |
| Task-model detection from event tables | `task_detection` |
| Encoding and penalty selection | `trial_encoding`, `ridge_selection`, `fractional_ridge`, `_regularizers`, `_ridge_cv`, `_fractional_ridge`, `ridge_results` |
| Trial design construction and OLS/ridge estimation | `single_trial`, `_shared_hrf_fit`, `_design_inputs`, `_single_trial_design`, `_single_trial_fit`, `single_trial_results` |
| Candidate HRFs, selection, evaluation, and grouped fits | `hrf_library`, `hrf_selection`, `hrf_results`, `_hrf_design`, `_run_design`, `_hrf_cv`, `_selected_hrf_fit` |
| Conventional GLMs using voxelwise HRFs | `_hrf_assignment`, `_hrf_glm_design`, `_hrf_glm`, `hrf_glm_results` |
| Scalar REML of one random-effect variance, AR(1) whitening and noise autocorrelation | `reml`, `ar1`, `_ar1_moments`, `_numerics` |
| Task-guided (GLMdenoise-style) denoising | `denoising`, `denoising_results`, `_mixture_threshold` |
| Records, log events, and file publication | `provenance`, `logging`, `bids_provenance`, `publication`, `_software` |
| Shared constants and deprecation shims | `_constants`, `_deprecation` |
| CIFTI scalar I/O, descriptive diagnostics, HRF agreement, process batches | `cifti`, `diagnostics`, `reliability`, `parallel` |
| Session workflow and `boldtailor run` | `workflow` (`settings`, `inputs`, `analysis`, `beta_series`, `run`, `outputs`, `export_cifti`, `figures`, `tables`, `manifest`, `artifacts`, `files`, `summaries`, `plots`, `surfaces`, `report`), `cli` |
| Notebook-only BIDS workflow helpers | `examples/bids_workflow` |

Public fitting entry points delegate to complete operations at the same level:

| Public entry point | Complete implementations |
| --- | --- |
| `fit.fit` | `_standard_glm.fit_standard_glm`, `_hrf_glm.fit_selected_glm` |
| `fit.task_delta_r2` | `_standard_glm.standard_task_delta_r2`, `_hrf_glm.selected_task_delta_r2` |
| `single_trial.fit_single_trials` | `_shared_hrf_fit.fit_shared_hrf_trials` |
| `single_trial.fit_selected_hrfs` | `_selected_hrf_fit.fit_selected_hrf_trials` |

These implementations own validation, numerical fitting, the logging scope,
provenance construction, and result assembly. Public wrappers choose the
implementation and resolve deprecated keyword aliases. Numerical helpers do
not own logging scopes.

HRF selection and evaluation likewise keep operation identity and provenance
construction in their orchestration functions. Their calculation helpers return
result fields and activity metadata without accepting a logging operation.
Denoising selection explicitly computes the pool, selects a component count,
builds provenance, and passes that completed provenance to result assembly.
The nested training selection during HRF evaluation remains a separate logged
operation, with its own execution identity.

The session workflow separates loaded `_Session` inputs from completed
`_Analysis` outputs. `_analyze()` passes selections into GLM and beta fits,
and beta models into summaries; stage functions return their results without
populating a shared state object. Missing prerequisites are rejected explicitly.
The figure registry is passed separately and owned by `run_workflow()` for
cleanup on success or failure. Report sections consume a named `ReportInputs`
record assembled explicitly, rather than the local Python namespace.

The core numerical functions accept arrays and tables and return results in
memory. Example workflows own dataset discovery, image loading, spatial axes,
and image reconstruction. The prepared-design API is the entry point for
externally compiled designs; it does not perform BIDS parsing or transformations.

The installed `boldtailor.workflow` package and the `boldtailor run` command
(`boldtailor.cli`) own BIDS + fMRIPrep CIFTI discovery, loading, fitting stages,
output naming, plotting, and the HTML report. Generic CIFTI dense-scalar I/O
lives in `boldtailor.cifti` (nibabel is a runtime dependency). Package modules
never import from `examples`. The remaining NSD example modules
(`nsd_settings.py`, `session_hrf*.py`, `multisession_*.py`) and the notebooks
import only public modules; plotting uses matplotlib, a runtime dependency.
Shared example fixtures live in `examples/bids_workflow/conftest.py`; each notebook has
one `@pytest.mark.notebook` kernel smoke test. See the
[NSD setup instructions](../examples/bids_workflow/README.md#full-workflow-notebook).

`tests/workflow` runs in the default suite (`uv run pytest -q`) using a small
synthetic BIDS dataset. `examples/bids_workflow` tests are opt-in
(`uv run pytest -q examples/bids_workflow`), and the notebook smoke tests additionally
need `--run-notebooks`.

Public numeric arrays are owned, ordinary NumPy arrays marked read-only through
`_arrays.readonly_array`. Float values use float64, indices use int64, and masks
use bool. Construction copies inputs; accessing an array does not copy it again.
These flags prevent accidental writes. Deliberately re-enabling writes can
invalidate the owning analysis and its provenance assumptions. Copy an array
before editing it. Mutable table/dictionary accessors return defensive copies,
including nested event metadata. Private containers are not a public mutation
interface.

Prepared fitting reads `PreparedDesignAnalysis._design_matrices` and
`_column_roles` internally. These are already owned at construction; fitting
and diagnostics must not mutate them. Public accessors keep returning copies,
and result construction still makes its own design copies. This avoids copying
all runs' tables repeatedly inside each run's diagnostics.

## Two kinds of code

- **The `boldtailor run` CLI** (`src/boldtailor/cli.py`, `src/boldtailor/workflow/`)
  analyses a BIDS dataset: raw BIDS events, fMRIPrep fsLR CIFTI time series, and
  fMRIPrep confounds. The examples in `examples/bids_workflow/` exercise this path.
- **The NSD replication** (`experiments/nsd_replication/`) is research code,
  not shipped in the package. It reads the NSD team's preprocessed CIFTI files
  from `derivatives/nsdpreproc`, raw BIDS events, and GLMsingle polynomial drift
  terms, and it uses only boldtailor's public core API. A static test
  (`experiments/nsd_replication/test_boundary.py`) blocks imports of
  `boldtailor.workflow`/`boldtailor.cli` and fMRIPrep CIFTI file names.

## Scientific result schemas

`ridge_results.CandidateScores` owns the common cross-validation arrays and
labels their rows with `grid` and an explicit `regularization` kind. Grid
validation and sorting stay in the scoring functions; result construction
does not reorder scores.

`single_trial_results.SingleTrialResult` owns trial estimates, diagnostics,
and penalty metadata. Its `design` is a `SharedTrialDesign` for one HRF across
features or a `SelectedTrialDesign` for grouped HRF assignments. Each design
container owns its matrices (a selected design rebuilds them on request
instead of retaining one per `(run, hrf_id)`), so the result does not duplicate
that copying logic. Selected designs retain named reconstruction recipes, without BOLD signals, instead of closures. Use `rebuild_group_design()` or `design.rebuild_matrix()` to make reconstruction explicit. `materialize_group_designs()` and `materialize_designs()` allocate all fitted designs on request. Immutable mappings elsewhere in results still limit standard `pickle` support; loky's cloudpickle can move results between processes, while arrays or dicts keep transfers smaller. The NSD examples return arrays and dicts. Common numerical
fields stay directly on the result. See
[result migration](result-migration.md) for API changes.

## Numerical conventions

Conventional fits use Nilearn's OLS/AR(1) estimator and t contrasts. Each run
has independent coefficients. Contrasts are combined through Nilearn contrast
addition and equal-run averaging. Full-model R² uses predictions on the original
signal scale and pools within-run sums of squares.

`fit(..., hrf_selection=...)` dispatches to grouped conventional fitting.
Each HRF group uses the same numerical contrast engine as a common-HRF fit.
Custom-HRF task columns are compiled separately from nuisances so Nilearn's
callable-name suffix cannot rename a semantic contrast or collide with a
confound. Canonical groups use the existing SPM design path. Group results
are restored to input feature order; undefined HRFs remain NaN.

The grouped result exposes a design for each `(run_index, hrf_id)`, with owned
arrays and copied DataFrames. Its identity includes effective model settings,
the HRF library/assignment, and labeled design values plus frame times. The
ignored fixed `hrf_model` setting does not affect this identity. The grouped
delta-R² path checks the parent identity, refits nested OLS models, and retains
NaNs for undefined selections or constant signals. Common-HRF/prepared delta
paths still require finite pooled R².

Selected-HRF completion logs include the analysis identity. Start records use
execution/source IDs because the identity is established during design
compilation.

Conventional and prepared fits share rank warnings, dimension validation,
and nested-OLS thresholds in `_fit_diagnostics.py`. Their contrast provenance
uses `model.contrast_metadata`. Fit-specific scientific operations remain
visible in the entry points; grouped-HRF validation retains its NaN contract.

Task-versus-nuisance diagnostics use nested OLS fits. Small negative differences
from floating-point roundoff are clipped in `TaskDeltaR2Result.delta_r2`;
`raw_delta_r2` retains the original values. Single-trial ridge comparisons use
the actual penalized fit and preserve signed differences.
All ΔR² entry points record one shared activity schema built by
`delta_r2_activity`, and compute R² pairs through
`_conventional.nested_ols_delta`; the
result constructor rejects differences below `-NESTED_OLS_TOLERANCE`.

Single-trial ridge projects task columns and signals off the nuisance span,
normalizes projected trial columns to unit L2 norm, applies a fixed penalty,
then restores coefficient units. Nuisance coefficients are unpenalized.
Trial identifiability and positive residual degrees of freedom are required.

The trial solvers separate preparation from candidate evaluation. For a run,
`x` has shape `(scans, trials)`, nuisance columns have shape `(scans, confounds)`,
and signals have shape `(scans, features)`. `_single_trial_fit` removes the
nuisance span and stores the normalized design's SVD in `ProjectedTrialDesign`.
`PreparedTrialBetas.betas_at(alpha)` uses stored signal coordinates to return
fresh `(trials, features)` coefficients. Alpha zero uses the same decomposition
for OLS. Full fits recover nuisance coefficients and sums of squares separately.

Fractional ridge uses `PreparedFractionBetas` in `_fractional_ridge`. Its rank
check uses normalized columns, but its decomposition for shrinkage uses raw
projected columns. This distinction preserves the requested ratio of raw trial
coefficient norms to their OLS norms. `solve` returns betas and feature-specific
implied alphas; `betas_at` returns only betas. Neither solver retains candidate
outputs. The older `trial_beta_path` and `fraction_beta_path` iterators now
live in `tests/oracles.py` as adapters for tests and validation scripts.

In `_ridge_cv`, `RunBetaPath` groups features by their training-selected HRF and
restores their original column positions. Each fold prepares each run once,
then explicitly evaluates a requested candidate. Fractional CV evaluates the
validation run at fraction one once and retains that fixed OLS target;
normalized ridge CV evaluates its validation target at each candidate alpha.
Only one candidate's beta arrays are assembled at a time. HRF selection and
encoding training still use only the training runs.

Shared numerical references live in `tests/oracles.py`: independent augmented
least squares and root finding provide checks against the production SVD.
Reusable synthetic datasets are pytest fixtures in `tests/conftest.py`.

HRF ID 0 is the Nilearn SPM kernel (32-second support) divided by its maximum.
`expanded_hrf_library()` adds 648 double-gamma candidates with 36-second support.
`default_hrf_library()`, used by `boldtailor run` and the workflow and
multisession notebooks, holds canonical SPM, 512 timing-space Sobol candidates
(`timing_hrf_library`), and GLMsingle's 20 empirical HRFs (533 in total). The
session-reliability notebook uses `sobol_hrf_library(n_samples=512, seed=0)`,
512 Sobol samples over the original grid's gamma-parameter ranges.
Candidate order is deterministic. Exported curves use a 0.1-second grid;
convolution uses TR/50 and the supplied acquisition times. Kernels are stored
at unit peak and each event's response is scaled to peak one
(`HRF_NORMALIZATION = "peak_one_event_response"`), so a trial beta is the peak
BOLD response to that event.

HRF selection pools leave-one-run-out prediction errors for a shared mean
stimulus response. Training amplitudes are held fixed on the omitted run.
RT and stimulus identity are not selection targets. Structural eligibility
uses timing and confounds, with lazy validation of proposed winners. A candidate
cannot drop trials or change the observation set to obtain a better score.

The independent split evaluation selects inside training runs, then freezes
the HRF and mean amplitude for test prediction. Both halves of the NSD reliability
comparison use the same timing/confound eligibility context. Production HRFs
selected over all runs are shared by the final OLS and ridge fits.

Nilearn 0.14 divides the OLS/AR(1) dispersion by `n - columns` (`regression.py:196`), inconsistent with its `df_residuals = n - rank`; Boldtailor avoids it by fitting rank-deficient designs in a full-rank basis (`_conventional._full_rank_basis`).

### REML conventions

`boldtailor.reml` fits, per feature, the model `y_g = C_g a_g + Z_g b_g + e_g`
for each group (run) `g`, with `b_g ~ N(0, τ² I)` and `e_g ~ N(0, σ² R_g(ρ))`.
`C_g` is `fixed_design`, the unpenalised nuisance regressors with their own
coefficients in every group. `Z_g` is `random_design`, the matrix whose
coefficients are random effects. Neither name refers to random numbers.
`R_g` is the identity or the stationary AR(1) correlation matrix with known
`ρ`. One variance ratio `λ = τ²/σ²` and one `σ²` are shared across groups.

`σ²` is the marginal noise variance. The whitener in `boldtailor.ar1.whiten`
is normalised so that `W R(ρ) W'` is the identity (first row unchanged, later
rows `(x_t − ρ x_{t−1}) / sqrt(1 − ρ²)`), so `λ` and `σ²` mean the same thing
at every `ρ`. In one synthetic scenario (AR(1) noise at `ρ = 0.5`, 16 groups of
100 scans) a fit that ignored the autocorrelation underestimated the marginal
noise variance by about 2%, while its median variance ratio was not worse than
the AR(1) fit's; this is a finding from that scenario, not a general law.

The criterion is the exact restricted likelihood, profiled over `σ²`. Its
constant depends only on `C` and `ρ`, so criteria are comparable across random
designs only at fixed `C` and `ρ`, never across `ρ` or `C`.
`select_random_effects_design` therefore compares candidates per feature, at the
groups' `fixed_design` and that feature's `rho`; `rho` may differ between
features, and features are never compared with each other.

Per-feature `status` is `ok`, `variance_zero` (the optimum is `λ = 0`, so the
data do not support the random design), `no_residual_signal` (the energy left after
projecting out `C` is within the rounding of that projection; all its estimates,
including `ratio`, are NaN), `numerically_unresolved_residual` (the energy left is above
that rounding but within the uncertainty of the computed fixed span, so it may
be real but cannot be certified; all its estimates, `ratio` included, are NaN),
or `unidentifiable`. `numerically_unresolved_residual` arises only when the fixed design has nearly
collinear columns and the signal loads on the direction they barely determine;
see the projection floors below. A fit is unidentifiable when the random covariance is a
multiple of the identity on the whole restricted space: every restricted
degree of freedom has a retained random direction and all pooled singular
values are equal (within `4 · dof · eps` relative). The covariance is then
`(σ² + c τ²) I`, the profiled criterion is constant in `λ`, and only the sum
is determined; `ratio` and both variances are NaN, `criterion` is that constant
(the `λ = 0` value), and the fit is not eligible in a selection. The check is
on the pooled spectrum, so groups that are each isotropic but at different
scales remain identifiable. A nearly confounded design is not flagged: its
profile is nearly flat, the data carry little information about the split, and
the estimate is reported as `ok` or `variance_zero` with that caveat.
`unrefined` flags a positive ratio that is a point of the search grid, not a
verified root of the criterion's slope; `at_upper` flags those at the upper
edge of the search range. That edge depends on the candidate's singular
values, so the criterion of an `at_upper` fit is a truncation, an upper bound
on that candidate's optimum, not a comparable optimum. In a selection, the
winner is the candidate with the lowest criterion among those with status
`ok`; ties go to the earlier candidate. A feature with no `ok` candidate has
winner `-1`, a NaN `gap` (the criterion difference to the runner-up, also NaN
when only one candidate is `ok`), and its `fit` is the `λ = 0` fit, or NaN
estimates for absent or numerically unresolved residuals.
`RemlSelection.any_at_upper` marks the
features for which any candidate, winner or not, was truncated: a truncated
loser might have won at its true optimum, so those selections are provisional.
`any_unrefined` additionally records interior fallbacks, including losing
candidates. Both HRF experiment adapters preserve these qualifications in their
diagnostics; `provisional` is `any_unrefined`, not a significance decision.

Orthogonal residual energy is computed from `y - U(U'y)`, not the subtraction
`||y||² - ||U'y||²`. The latter loses small noise energy at high SNR and can
distort the likelihood at large ratios. When retained random rank fills the
restricted space, its orthogonal complement has dimension zero and its energy
is exactly zero. Otherwise the explicit residual calculation has rounding
error in amplitude units; it does not guarantee accuracy for information below
that arithmetic scale.

The criterion depends on the groups only through pooled statistics (all
singular values, the stacked squared coordinates, the summed residual sum of
squares and dof), so the search pools them once and every evaluation is one
vectorised pass whatever the number of groups. The search is a log grid over
the variance ratio from `1e-8 / max s²` to `1e8 / min s²`, with 121 points when
all singular values are equal and more for a wider spectrum, so the spacing in
`log λ` never exceeds `log(1e16) / 120 ≈ 0.31`. The criterion and its analytic
derivative with respect to `log λ` are evaluated on the grid; bisection refines
the root inside every grid cell where the derivative turns from negative to
positive, and the lowest of those roots replaces the best grid point unless it
is worse beyond floating-point rounding. Refining only around the best grid
point is not enough: a minimum narrower than the spacing can be bracketed by a
sign change while both neighbouring grid values are worse than `λ = 0`. The
result is then compared explicitly with `λ = 0`, and a positive ratio must
improve on zero by more than the criterion's rounding noise
(`64 · eps · (|F| + dof)`). The guarantees, rather than the iteration counts,
are the contract: a positive ratio that is not flagged `unrefined` is a
stationary point of the criterion to near machine precision; the reported
optimum is never worse than the best grid point beyond rounding; `λ = 0` is an
explicit boundary; and `at_upper` reports a boundary hit. When no slope root is
accepted the best grid point stands, keeps status `ok`, and is flagged
`unrefined`: it need not be stationary. At the default grid this was not
observed away from the upper edge in the oracle, property-style and adversarial
test problems, but it is not excluded; a coarse grid produces it readily
(`tests/test_reml_search.py`). The search is not a certified global
minimisation: a minimum and a maximum that both fall strictly inside one grid
cell leave no sign change and are not refined.

Memory: features are searched in batches sized so that each temporary of the
search (`(grid rows, k, features)` or `(k, crossings)`, with `k` the pooled
number of random directions) and each retained `(grid, features)` table stays
within 8 MiB; about ten such arrays are alive at once, and one feature is the
smallest batch, so a temporary is never below `8 k` bytes. The bound covers
the search only. The pooled statistics, `(k, features)` per fit, and the
projected signals, `(scans, features)` per group and distinct `ρ`, are held in
full, so this is not a bound on total process memory. Measured with
`tracemalloc` on 100 directions by 30,000 features (22.9 MiB of pooled
statistics), the search's peak extra allocation was 78 MiB at the default
budget, 31 MiB at 1 MiB and 221 MiB at 64 MiB, against 168 MiB before
batching; run time differed by a few percent across those budgets.

Rank and degeneracy policy shared by `reml` and `ar1` lives in `_numerics`.
The columns of a projected-out design (`C`, or the AR(1) reference) are
normalised before its basis is computed. The fit depends on such a design only
through its column span, so results, rank and restricted dof do not depend on
the units of its columns; only exactly zero columns are dropped for their
size, and rank among the normalised columns uses one tolerance,
`max(shape) · eps · σ_max`. A column of rounding residue is therefore a
direction like any other and costs a degree of freedom; remove such columns
before fitting.

A projection residual is judged against two floors, both `8 · n · eps` times
a norm. The rounding floor uses the norm of the signal: with normalised
columns, signals lying exactly in the span leave at most about 1.2 times
`n · eps · ‖y‖` when the design is ill-conditioned only through its units. The
uncertainty floor uses the norm of the signal with its coordinate on each
basis vector amplified by `σ_max / σ_i` of the normalised design: a basis
computed from `C` spans the range of `C + E` with `‖E‖ ~ n · eps`, so signal
along a direction that nearly collinear columns barely determine leaves
proportionally more, up to `1e5` times the rounding scale in the cases pinned
in `tests/test_numerics.py`, and at most 1.0 times the amplified one. Energy
at or below the rounding floor is `no_residual_signal`; between the floors it is
`numerically_unresolved_residual`; above the uncertainty floor the feature is fitted. `ar1` returns
NaN at or below the uncertainty floor and has no status to tell the two cases
apart. The uncertainty floor is a bound, not a measurement: a residual below
it is not shown to be noise, which is why it is not reported as `no_residual_signal`.
Both are floating-point floors in float64, not significance tests: adding
`C a` to the signals leaves the fit unchanged for as long as the residual is
still representable and resolved, and data whose residual is only quantisation
noise from a lower-precision file format are not masked; callers mask such
features. For random designs, a column is rejected against the unprojected norm of `Z`,
computed from the smaller of `Z'Z` and `ZZ'`. The internal `project_fixed` (not public API) runs once per
group per distinct `ρ`, plus once more for a `ρ` subset that mixes features
with and without signal, so results for live features are bit-identical
whether or not dead features are present. Every candidate's `Z` is whitened,
projected and decomposed once per group per distinct `ρ`, so cost grows with
the number of distinct `ρ` values: unbinned per-vertex `ρ` (for example 1,750
values × 533 candidates × 12 runs) is impractical, and callers bin it
(`regularized_pilot.py` uses steps of 0.02). All `ρ` subsets must share one
restricted dof (the rank of the whitened `C` must not change with `ρ`);
otherwise `ValueError` is raised. Input validation lives in `reml`.
Zero-feature signals, non-finite inputs, shape mismatches, `|ρ| >= 1`, an empty
candidate iterable, and a candidate with the wrong number of groups raise
`ValueError`; candidate errors name the candidate index, and a candidate may be
any iterable of per-group designs. `ar1.estimate_rho` corrects the residual
lag-one ratio for projection bias on a fixed grid. Its signal-independent work
(an orthonormal basis of the reference design and the expected lag ratio on the
grid) can be computed once with `ar1.prepare_reference` and reused; the
simulations cache it per run. A feature is degenerate (NaN) when its residual
denominator does not exceed the uncertainty floor above. `RhoEstimate.pooled` is the median
over all groups and features of the finite corrected values, clipped to
`rho_range` (default `(-0.2, 0.95)`), and `0.0` when none is finite.
`RhoEstimate.pooled_at_bound` marks a pooled value on a limit of that range.
The lower limit is a prior, not a property of the estimator: strong negative
autocorrelation is not plausible for fMRI noise, mildly negative estimates
are ordinary sampling error around zero, and a strongly negative pooled value
signals an upstream problem that the flag keeps visible. Clipping commutes
with the median, so the limits change the pooled value only when the median
itself lies outside them. In the regularized AR(1) pilot (256 vertices,
sessions 11 and 16) every coefficient was between 0.22 and 0.80 and some sat
at the former 0.8 cap, which is why the upper limit is now the end of the
correction grid.
Inversion requires the curve to be finite and strictly monotone across the
pooling range, `rho_range` cut to the grid, with adjacent differences larger
than `64 eps max(1, max(abs(expected)))`, a roundoff guard on the
dimensionless lag ratio. The monotone branch that spans that range, extended
as far as it stays monotone, is inverted by interpolation for every value,
and values beyond it are clipped to its ends, as a fully monotone curve clips
at the ends of the grid. Decreasing branches are reversed for interpolation.
A curve that is nonfinite, flat, or turns inside the pooling range yields NaN
for every value. Pooling excludes those values. The zero fallback when none
remain is explicitly not an inferred absence of autocorrelation.

Two alternatives were tried and rejected. Rejecting the whole curve for a bend
anywhere on the grid discards usable estimates: confound-heavy designs often
bend only beyond `|ρ| > 0.9` (about 8% of designs drawn with 200 scans and 100
censored volumes), and a true `ρ` of 0.4 was then pooled as 0.0. Inverting
value by value, with NaN at the levels a bend repeats, drops observations
according to their value and biases the pooled median: on designs with ten
residual degrees of freedom it sat at the then 0.8 limit for a true `ρ` of 0.4. The
branch rule drops no value and keeps their order, and on those designs
reproduces the estimates of plain monotone interpolation.

Scientific regression coverage includes independent diagonal covariance and
derivative oracles, known orthogonal energies near saturation, generated
confounding classifications, dense AR(1) moment traces and time reversal.
Bounded Monte Carlo checks cover estimated-rho recovery/null-tail behavior
for matched references at rho ±0.5, and null selection as library size grows.
A winner is not a significance test. Shared-variance estimates under unequal
run variances summarize a pooled quantity, not each run's variance. These
checks do not establish robustness to arbitrary temporal covariance,
heterogeneous rho, correlated random coefficients or an incomplete reference
design; those remain modeling sensitivity questions.

`tests/reml_oracles.py` imports nothing from `boldtailor`. It holds a
dense-covariance criterion and optimiser (a scan over `log λ` followed by a
bounded search around every local minimum of the scan), a `statsmodels` `MixedLM(reml=True)`
fit (BFGS), and a finite-sample null reference for the restricted
likelihood-ratio statistic. Two oracle cases are strict xfails because MixedLM
does not reach the optimum there; the dense optimiser confirms the package
value. Heavier Monte Carlo runs are opt-in:
`uv run pytest tests/test_reml_statistics.py -q --run-slow`.

## Source identity and provenance

Threat model: cooperating writers on one host, no adversary, no power-loss
guarantee. Records are scientific metadata, not a privacy or security boundary.

`SourceRef` holds a role, a dataset-relative or `bids:<dataset>:<path>` URI,
media type, byte size, UTC `modified_at` (`Z` or `+00:00`), optional `sha256`,
and JSON annotations; `RunSources` groups signal, events, and confounds. One
rule, `provenance.validate_relative_path`, governs URIs and BIDS projection
paths: POSIX, no traversal or empty parts, components `[A-Za-z0-9+_.-]+`.
Annotation, metadata, and model-key text is stored as given, even if it looks
like a path: callers must not put secrets or environment details in
annotations or metadata; they are written verbatim. `data_fingerprint`
hashes source metadata (not file contents) and is absent for incomplete sources. Each activity records `software`, which
analysis identities exclude. Execution IDs are fresh UUIDs per attempt.

## Logging

The library emits JSON messages through the standard `boldtailor` logger and
does not install handlers. Applications choose the destination and log level.
`bind_context()` carries execution, source, analysis, and run IDs;
`emit_event()` creates a lifecycle record; `append_event_history()` retains
the most recent eight events in provenance.

Fit, comparison, HRF-selection and ridge-scoring entry points use the private
`fit_operation()` context manager. Their statistical steps remain explicit in each caller. The context
owns execution IDs and logging context. Normal exit logs completion; an exception
logs failure and is re-raised. No call on the yielded metadata is required to
complete the scope. Callers set `operation.analysis_fingerprint` explicitly when the model
identity becomes available.

`build_fit_provenance(operation, activity, warnings=...)` independently constructs
a record without changing the operation or emitting log messages. It may be
called repeatedly, and its use is optional from the logging scope's perspective.
Fit entry points build provenance and construct the result inside the scope, so
errors in either still produce a failure log. The provenance completion snapshot
and the completion log share event names and identities but have separate
timestamps and sequence numbers: the log is generated only on successful exit.
Start records omit `analysis_fingerprint`; completion includes it when metadata identity
is available.
Each nested fit or normalization starts a fresh ID scope and restores its
caller’s context on exit. Omitted `bind_context()` fields inherit the parent value; explicitly passing
`None` removes that field for the scope. `inherit=False` starts with no parent
fields. `emit_event()` uses the same omission/None distinction for that event
only; it does not modify the enclosing context.

Failure records export an `error_code`: `invalid_input`, `numerical_failure`,
`io_failure`, or `operation_failed`. Exceptions are re-raised unchanged for
callers. The logging boundary never exports exception text or inspects the
environment; the prepared-fit sanitizer has been removed. Event names, IDs,
and annotations are caller-supplied metadata, so this is not a blanket privacy
guarantee. Existing saved records retain their old fields. See the
[lifecycle migration](lifecycle-migration.md) for event prefixes and category rules.

Input normalization constructs one final provenance record. Extension creates
a validated record directly, avoiding a parent-record serialization round trip.
Source identity formulas, scientific activities, and quality warnings are
preserved; source fingerprints describe metadata, not BOLD contents.

Ridge CV reads the final penalty explicitly from `tuning["all"]["selection"]`.
The final refit does not depend on which tuning scope ran last.
`tuning_provenance()` and `link_final_provenance()` (in
`workflow.ridge_provenance`) take `encoding_mode`
explicitly, so additional provenance activities do not change where metadata
is read. `link_final_provenance()` returns a new result with independent block
provenance; callers must use its return value. Numerical arrays are shared and
left untouched. It rejects already linked fits to prevent duplicate refit
activities; repeated calls on the original fit leave that fit unchanged.

## BIDS metadata projection

`project_bids_provenance()` returns relative paths mapped to bytes without I/O:
BIDS 1.11.1 metadata plus an optional (`export_bids_prov`) draft subset pinned
to `BEP028@02172700aac8d1bdd67b45191f43533f426848dc`. The logs equal the record
(`logs/boldtailor_provenance.json` is its canonical JSON); no keys are dropped.
Each Activity's `Command` names the entry point (`boldtailor.fit`,
`boldtailor.select_hrfs`, ...); the last carries `StartedAtTime`/`EndedAtTime`
from its lifecycle events; `Environments[0]` has `Python`, `Platform`, and a
`Software` list taken from the last activity's `software`.

## Writing result files

Workflows build `Artifact` objects in memory and pass them to
`publish_artifact_set()`. Under a `FileLock`, it checks targets, stages files in
`<destination>.boldtailor/stage-<uuid>/`, moves overwritten files to
`backup-<uuid>/`, `os.replace`s each file into place, and fsyncs files (not
directories). With `manifest_path`, the old JSON manifest is read under the same
lock; stale listed files are moved into those backups before promotion. Failure
restores removed files as well as overwritten ones. No stale-file deletion occurs
after releasing the lock. Missing/legacy manifests authorize no deletion;
corrupt or unreadable manifests stop publication. Writers serialize; readers can
see a partial set. New artifact payloads are not parsed. The destination is resolved first (so a
linked destination or macOS `/tmp` works); symlinks inside it are refused, as
are unrequested overwrites and outputs overlapping `source_paths`.
`<destination>.boldtailor/` is a sibling, not part of the dataset (so no
`.bidsignore`); it holds `lock` and `failures.jsonl`, whose fixed-field records
omit exception text. If rollback fails, `PublicationError.recovery_directory`
names the retained `backup-<uuid>/` (or `stage-<uuid>/` when nothing was backed
up); `rollback_errors` lists the failures.

## Workflow exports and parallel execution

The session workflow (`boldtailor.workflow`) loads feature blocks, preserves CIFTI axes, and
reconstruct full output arrays in the parent process. `n_jobs>1` uses
`boldtailor.parallel.map_blocks`: bounded batches of joblib loky processes,
each with all runs for its assigned feature block and one numerical-library
thread. Workers return arrays and dictionaries (see the serialization note
above), never write derivatives, and
a worker failure prevents publication. Library/timing/design caches are
process-local. Full output beta arrays and the assembled artifact set still
consume memory.

Each `<stem>_desc-<GLM descriptor>_designs.npz` stores, per run label and HRF ID, the
float64 design (`<run>_hrf-<id>`), its column names (`_columns`), and frame
times (`_frame_times`). Load with `allow_pickle=False`. The installed core has
no general loader that reconstructs an `HrfSelectionResult` from image files.
The session-reliability example reuses compatible saved selection maps and
provenance through its dedicated cache/import helpers.

## Provenance digests

`SourceRef` accepts an optional caller-supplied `sha256` (64 lowercase hex
characters). It enters the data fingerprint and is emitted as the BEP028
`Digest` of the file entity. The earlier ban on a `digest` key was removed on
2026-10-02: a content digest of imaging data is not identifying, and BEP028
defines `Digest`.

## Design history and validation

Current behavior is described by the [user guide](user-guide.md) and
[API reference](api.md). The `docs/superpowers` tree contains implementation
plans, design decisions, and historical validation records; some older plans
describe features that have since shipped.

- [NSD session results and benchmarks](validation/nsd-session.md)
- [Expanded HRF validation](superpowers/validation/2026-09-26-hrf-selection.md)
- [Parallel execution validation](superpowers/validation/2026-09-26-nsd-parallel.md)
- [Odd/even HRF validation](superpowers/validation/2026-09-27-split-hrf.md)

## Dependency notes

- `uv.lock` pins nilearn 0.14.0, a yanked release. `pyproject.toml` allows
  `nilearn>=0.14.0,<0.15`.
- `_hrf_design` imports Nilearn's private `_sample_condition` to build the
  oversampled boxcar that the peak-normalized event scales are computed on.
  `tests/oracles.py` imports it as well to build independent references.
- Decision: stay on `<0.15` until the fast convolution path (the private
  `_sample_condition` contract and `compute_regressor` numerics) is re-verified
  against the next Nilearn release by the Nilearn oracle tests.

## Branch inventory

Branches merged into `main` and not checked out in a worktree were deleted with
`git branch -d`. `fix/dynamic-contrast-artifacts` is merged but kept because
its worktree (`.worktrees/dynamic-contrast-artifacts`) still exists. The
remaining branches, with purposes taken from their last commit, are:

| Branch | Purpose (last commit) |
| --- | --- |
| `fix/dynamic-contrast-artifacts` | Merged; BIDS-safe contrast labels (kept for its worktree) |
| `feature/pre-fitlins-remediation` | Remediation work before FitLins integration (privacy coverage tests) |
| `fix/red-team-remediation` | Red-team remediation (shared-checkout integration verification) |
| `safety/prepared-notebook-source-access` | Safety copy: prepared notebook source-access documentation |
| `safety/task-delta-r2-task4-d86b117` | Safety copy: task delta R-squared notebook diagnostics |
| `safety/task-delta-r2-task4-fd2db4c` | Safety copy: task delta R-squared display |
| `safety/task3-pre-clipped-regression` | Safety copy: serialized task delta R-squared map |
| `safety/task4-before-missing-term-review` | Safety copy: prepared design-matrix fits |
| `task3-review-safety-dbc5b14` | Safety copy: whole-brain notebook task report |
| `task5-review-safety-daa401f` | Safety copy: prepared fit provenance |

The `safety/` and `task*-safety` branches are review snapshots; delete them once
the maintainer confirms they are no longer needed.

## Renamed keyword defaults

While an old keyword remains supported, both old and new parameter defaults must
use `UNSET`. `renamed_keyword(..., default=...)` resolves the effective default
only after checking which names were supplied. Supplying both names raises,
including when the new value equals its effective default. Do not infer argument
omission from object identity or value equality. `fit_single_trials` still uses
canonical SPM when neither HRF keyword is supplied.
