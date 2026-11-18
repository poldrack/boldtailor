# Choosing tests during development

Run module-focused tests while editing, then run the integration suite before
committing. Run the full default suite before merging and whenever changing test
selection rules.

## Everyday commands

```bash
# Example: REML and its numerical helpers.
uv run pytest tests/test_reml*.py -q -W error

# Representative cross-module checks, including a synthetic BIDS workflow run.
uv run pytest -m integration -q -W error

# Preview which integration cases will run.
uv run pytest -m integration --collect-only -q

# Full default suite before merging; reports the 20 slowest phases.
uv run pytest -q -W error --durations=20
```

Run module tests and `-m integration` as separate commands; passing both filters
to only the integration tests inside the named files. The `integration` marker
selects a representative cross-module subset, while unmarked tests cover unit
behavior, scientific regressions, and property-based invariants. When editing
shared code, include its consumers from the table below or run the full suite.

## Module-to-test mapping

Paths are relative to the repository root. Pass them to `uv run pytest -q -W error`,
then run `uv run pytest -m integration -q -W error`.

| Changed area | Focused test paths | Additional consumers |
| --- | --- | --- |
| `reml` | `tests/test_reml*.py` | `tests/test_hrf_reml*.py tests/workflow/test_reml_workflow.py` |
| `_numerics`, `ar1`, `_ar1_moments` | `tests/test_numerics*.py tests/test_ar1.py tests/test_reml*.py` | `tests/test_fit.py tests/test_prepared_fit.py tests/test_hrf_glm.py tests/test_single_trial.py` |
| `model`, `design`, `_task_design`, `task_detection` | `tests/test_model.py tests/test_design.py tests/test_task_model.py tests/test_task_design.py tests/test_task_detection.py` | Conventional, selected-HRF, and single-trial fitting tests |
| `fit`, `_standard_glm`, `_conventional`, `_fit_diagnostics`, `results` | `tests/test_fit.py tests/test_multirun.py tests/test_fit_diagnostics.py tests/test_conventional_properties.py tests/test_result_schemas.py` | `tests/test_prepared_fit.py tests/test_hrf_glm.py` |
| `prepared`, `prepared_fit` | `tests/test_prepared.py tests/test_prepared_fit.py` | `tests/test_fit_diagnostics.py tests/test_hrf_glm.py` |
| `hrf_library`, `hrf_options`, `hrf_selection`, `hrf_results`, `_hrf_design`, `_run_design`, `_design_inputs`, `_hrf_cv`, `_hrf_reml` | `tests/test_hrf*.py tests/test_choose_eligible.py tests/test_design_inputs.py tests/test_glmsingle_hrf.py tests/test_sobol_hrf_library.py` | `tests/test_selected_hrf_fit.py tests/test_ridge_cv.py tests/test_fractional_cv.py tests/test_denoising.py` |
| `_hrf_glm`, `_hrf_glm_design`, `hrf_glm_results` | `tests/test_hrf_glm.py tests/test_fit.py` | `tests/workflow/test_analysis.py` |
| `single_trial`, `_shared_hrf_fit`, `_selected_hrf_fit`, `_single_trial_design`, `_single_trial_fit`, `single_trial_results` | `tests/test_single_trial*.py tests/test_selected_hrf_fit.py tests/test_shared_assembly.py` | `tests/test_ridge_cv.py tests/test_fractional_cv.py tests/test_denoising.py` |
| `ridge_selection`, `fractional_ridge`, `_regularizers`, `_ridge_cv`, `_fractional_ridge`, `ridge_results`, `trial_encoding` | `tests/test_ridge*.py tests/test_fractional*.py tests/test_regularizers.py tests/test_trial_encoding*.py tests/test_encoding_design.py` | `tests/workflow/test_beta_series.py` |
| `denoising`, `denoising_results`, `_mixture_threshold` | `tests/test_denoising*.py tests/test_mixture_threshold.py tests/test_within_run_cv_isolation.py` | `tests/test_hrf_selection.py tests/test_fit.py` |
| `data`, `_run_inputs`, `_arrays`, `_scalars`, result dataclasses | `tests/test_data.py tests/test_arrays.py tests/test_mutation_helpers.py tests/test_dataclass_equality.py tests/test_result_schemas.py` | Broad fan-out: run the full suite for shared helper changes |
| `provenance`, `bids_provenance`, `logging`, `_fit_lifecycle` | `tests/test_provenance.py tests/test_bids_provenance.py tests/test_logging.py tests/test_lifecycle.py tests/test_fit_lifecycle.py tests/test_identity_names.py tests/test_software.py` | `tests/test_fit.py tests/test_prepared_fit.py tests/test_denoising.py` |
| `publication`, `cifti` | `tests/test_publication*.py tests/test_cifti.py` | `tests/workflow/test_outputs.py tests/workflow/test_files.py` |
| `reliability`, `diagnostics` | `tests/test_reliability.py tests/test_diagnostics.py tests/test_fit_diagnostics.py` | `tests/workflow/test_analysis.py tests/workflow/test_plots.py` |
| `workflow`, `cli` | `tests/workflow/` | Select a matching file during edits; run `tests/workflow/` for cross-stage changes |
| `parallel` | `tests/test_parallel.py` | `tests/workflow/test_analysis.py tests/workflow/test_beta_series.py` |
| Packaging, import structure, retired shims | `tests/test_import_structure.py tests/test_package_build.py tests/test_distribution.py tests/test_removed_shims.py tests/workflow/test_package_layout.py` | Full suite and installed-wheel check in the [developer guide](development.md#environment-and-checks) |

The default suite uses `testpaths = ["tests"]` and excludes `examples/` and
`experiments/`. When changing HRF selection or REML adapters used by the
experiments, also run:

```bash
uv run python -m pytest -q -W error experiments/hrf_estimation/test_reml_selection.py experiments/hrf_simulations/test_fitting.py
```

## Integration coverage

The `@pytest.mark.integration` suite covers conventional OLS and AR(1) fits
against an external oracle, prepared and selected-HRF fitting, REML design
selection, ridge CV and encoding, fitting with denoising components, CIFTI
serialization, transactional publication and rollback, BIDS provenance, parallel
execution, and an end-to-end synthetic BIDS workflow run.

Mark one representative test with `@pytest.mark.integration` when adding a new
public workflow or module boundary, and keep detailed edge cases in the module
suite.

## Slow scientific checks and notebooks

Opt-in suites run separately from the default test run:

```bash
# Slow Monte Carlo REML calibration checks.
uv run pytest tests/test_reml_statistics.py --run-slow -q -W error

# Example and validation suites, including notebook execution.
uv run pytest examples/bids_workflow examples/validation --run-notebooks -q -W error
```

Run `--run-slow` when changing statistical estimators or numerical tolerances,
and `--run-notebooks` when changing notebook workflows or their dependencies.

## Test purpose and maintenance

Give every test function a one- or two-sentence docstring stating the behavior,
invariant, or failure mode it checks. Keep setup rationale in comments.
Parameterized or shared-helper tests covering distinct regimes are not
redundant. Prefer executing documented code examples over matching prose
substrings. See the archived
[test-suite review](archive/test-suite-review-2026-11-16.md) for background.
