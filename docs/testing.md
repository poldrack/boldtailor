# Choosing tests during development

Use module-focused tests while editing, then run the integration suite before
committing. Run the full default suite before merging; CI also runs it on every
push and pull request. Changes to test selection itself require a full run.
A successful targeted run is not evidence that the full suite passed.

## Everyday commands

```bash
# Example: REML and its numerical helpers (shell expands the filename glob).
uv run pytest tests/test_reml*.py -q -W error

# Representative cross-module checks, including a complete synthetic BIDS run.
uv run pytest -m integration -q -W error

# Preview exactly which integration cases will run.
uv run pytest -m integration --collect-only -q

# Full default suite before merging; slowest setup/call/teardown phases reported.
uv run pytest -q -W error --durations=20
```

Run the module command and the integration command separately. Adding
`-m integration` to the module command would select only the marked tests in
those files, not the repository-wide integration suite. Module runs intentionally
include their integration cases; rerunning a few cases is simpler than excluding
relevant checks during editing.

The `integration` marker identifies a maintained representative suite, not an
exhaustive classification of every test. Unmarked tests include unit tests,
scientific regressions, property tests, and additional integration scenarios.
There is no automatic changed-file or dependency inference. When editing shared
code, include its consumers using the table below; use the full suite when the
impact is unclear.

## Module-to-test mapping

Paths below are relative to the repository root. Pass the listed paths to
`uv run pytest`, followed by `-q -W error`. Globs are expanded by the shell.
The integration suite is additional to every row.

| Changed area | Focused test paths | Additional consumers to consider |
| --- | --- | --- |
| `reml`, `_reml_*` | `tests/test_reml*.py` | `tests/test_hrf_reml*.py tests/workflow/test_reml_workflow.py`; optional experiment checks below |
| `_numerics`, AR(1), `_ar1_moments` | `tests/test_numerics*.py tests/test_ar1.py tests/test_reml*.py` | `tests/test_fit.py tests/test_prepared_fit.py tests/test_hrf_glm.py tests/test_single_trial.py` |
| `model`, `design`, `_task_design`, task detection | `tests/test_model.py tests/test_design.py tests/test_task_model.py tests/test_task_design.py tests/test_task_detection.py` | Conventional, selected-HRF and single-trial fitting tests |
| `fit`, `_standard_glm`, `_conventional`, results | `tests/test_fit.py tests/test_multirun.py tests/test_fit_diagnostics.py tests/test_result_schemas.py` | `tests/test_prepared_fit.py tests/test_hrf_glm.py` |
| `prepared`, `prepared_fit` | `tests/test_prepared_fit.py` | `tests/test_fit_diagnostics.py tests/test_hrf_glm.py` |
| HRF libraries, designs, selection, CV | `tests/test_hrf*.py tests/test_glmsingle_hrf.py tests/test_sobol_hrf_library.py` | `tests/test_selected_hrf_fit.py tests/test_ridge_cv.py tests/test_fractional_cv.py tests/test_denoising.py` |
| `_hrf_glm*`, `_hrf_assignment` | `tests/test_hrf_glm.py tests/test_fit.py` | `tests/workflow/test_analysis.py` |
| Single-trial fitting, `_selected_hrf_fit`, `_shared_hrf_fit` | `tests/test_single_trial*.py tests/test_selected_hrf_fit.py` | `tests/test_ridge_cv.py tests/test_fractional_cv.py tests/test_denoising.py` |
| Ridge selection, fractional ridge, encoding | `tests/test_ridge*.py tests/test_fractional*.py tests/test_trial_encoding.py` | `tests/workflow/test_beta_series.py` |
| Denoising and mixture thresholds | `tests/test_denoising*.py tests/test_mixture_threshold.py tests/test_within_run_cv_isolation.py` | `tests/test_hrf_selection.py tests/test_fit.py` |
| Data ownership, `_arrays`, `_scalars`, result dataclasses | `tests/test_data.py tests/test_arrays.py tests/test_mutation_helpers.py tests/test_result_schemas.py` | Broad fan-out: run full suite for shared helper changes |
| Provenance, logging, fit lifecycle | `tests/test_provenance.py tests/test_bids_provenance.py tests/test_logging.py tests/test_lifecycle.py tests/test_fit_lifecycle.py tests/test_software.py` | `tests/test_fit.py tests/test_prepared_fit.py tests/test_denoising.py` |
| Publication and CIFTI | `tests/test_publication*.py tests/test_cifti.py` | `tests/workflow/test_outputs.py tests/workflow/test_files.py` |
| Reliability and diagnostics | `tests/test_reliability.py tests/test_diagnostics.py tests/test_fit_diagnostics.py` | `tests/workflow/test_analysis.py tests/workflow/test_plots.py` |
| Workflow, CLI, reports | `tests/workflow/` | Select a matching file during edits; run the directory for cross-stage changes |
| Parallel execution | `tests/test_parallel.py` | `tests/workflow/test_analysis.py tests/workflow/test_beta_series.py` |
| Packaging, public names, dependencies | `tests/test_package_build.py tests/test_distribution.py tests/test_public_names.py tests/workflow/test_package_layout.py` | Full suite and installed-wheel check in the developer guide |
| Test markers or selection rules | `tests/test_test_selection.py` | Integration selection and full suite |

The full default suite uses `testpaths = ["tests"]`. It does not automatically
include experiment or example directories. For REML adapter changes, also run:

```bash
uv run python -m pytest -q -W error experiments/hrf_estimation/test_reml_selection.py experiments/hrf_simulations/test_fitting.py
```

Using `python -m pytest` here makes the repository root available for experiment
imports. Other experiment changes need their corresponding tests explicitly.

## Integration coverage

The marked suite checks conventional OLS and AR(1) against an external oracle,
prepared and selected-HRF fitting, REML design selection, ridge CV and encoding,
fitting with denoising components, CIFTI serialization, transactional publication
and rollback, BIDS provenance, process execution, and a synthetic workflow from
BIDS loading through maps and an HTML report. It uses generated data, not the
external NSD volume.

Mark a representative test with `@pytest.mark.integration` when adding a new
public workflow or module boundary. Retain focused validation and edge-case
checks in the module suite. Do not mark entire files simply because some tests
cross module boundaries. Review `--durations=20` when expanding this suite.

## Slow scientific checks and notebooks

These existing opt-in checks remain separate:

```bash
# Includes the slow statistical cases normally skipped by default.
uv run pytest tests/test_reml_statistics.py --run-slow -q -W error

# Executes example notebooks; requires development dependencies.
uv run pytest examples/bids_workflow examples/validation --run-notebooks -q -W error
```

Run the relevant scientific checks when changing the statistical method or
numerical tolerances. Run notebook checks when their workflow or dependencies
change. GitHub Actions runs only the default package test suite and reports the 20
slowest phases. Run example and experiment notebook tests locally when needed.
The package suite is not reduced by the integration marker.

## Test purpose and maintenance

Every test function has a short purpose docstring. The documentation check scans
package, experiment and example tests without executing the optional suites:

```bash
uv run pytest tests/test_test_documentation.py -q
```

Describe the behavior or failure being checked; include the oracle or boundary
when it matters. Keep one or two sentences, and retain detailed setup rationale
in comments. Shared bodies with different parameter grids are not redundant.
Prefer executing documented examples over asserting that prose contains specific
words. Keep license, distribution and explicit architecture requirements tested.
See the [test-suite review](test-suite-review-2026-11-16.md) for this review's
removals, retained cases and timing findings.

Public REML HRF adapter checks are in `tests/test_hrf_reml*.py`, including
generated invariants, a dense likelihood oracle, and seeded model recovery.
Workflow coverage includes `tests/workflow/test_reml_workflow.py`.
