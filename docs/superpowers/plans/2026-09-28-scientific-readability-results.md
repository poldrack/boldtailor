# Shared Scientific Results and Fit Diagnostics Implementation Plan

> **Development plan:** Includes proposed work as well as implemented changes.
> See the [documentation index](../../README.md) for current API and methods guidance.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the preserved native execution method. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Remove duplicate scientific result containers and repeated conventional/prepared fit diagnostics while keeping numerical outputs easy to access.

**Architecture:** One candidate-score schema names its regularization kind and grid. One single-trial result owns common numerical fields and contains either a design shared across features or selected-HRF design information. Ordinary shared functions handle common diagnostics and contrast metadata; scientific fitting remains visible in its current entry points.

**Tech Stack:** Python >=3.12, NumPy, pandas, pytest, uv; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-28-scientific-readability-design.md`

**Status:** All three tasks implemented from `d4b6783` (789-test baseline); 801 tests, formatting, installed-wheel verification, and independent review passed. See [validation](../../validation/scientific-readability-results-2026-09-28.md).

## Scope and sequencing

This completes the result/diagnostic portion of roadmap Stage 3. Shared fit
lifecycle work moves alongside Stage 4's unified logging policy and provenance
simplification. Both changes touch the same exception boundaries, event histories,
and completion records; implementing them together avoids migrating each fit
entry point twice. The lifecycle requirement remains open, not dropped.

No compatibility mixins, `__getattr__` forwarding, or aliases to removed result
classes are needed: the user permits documented API changes, and all identified
in-repository consumers will migrate in the same change.

## Global Constraints

- Keep main and stash `c4cfbec5fccd5397ea650c550f5b485ea588c2cd` unchanged; work on `refactor/scientific-readability`.
- Use `uv run` for Python, pytest, and formatting. Every `__init__.py` stays empty.
- Observe and commit failing tests before implementation. Test access-path changes must reflect the documented new schema; numerical assertions and tolerances remain intact.
- Preserve coefficient units, raw versus normalized penalties, fixed fractional validation targets, candidate-specific alpha targets, HRF isolation, ordering, ties, NaNs, and pooled SSE/SST.
- Preserve existing serialized scientific artifacts and provenance fields. The result-class API changes; saved `alphas`/`fractions` keys do not change in this increment.
- Preserve the approved ownership model: owned ordinary read-only arrays and defensive table/dictionary accessors.
- Keep conventional GLM result APIs unchanged. In particular, do not mechanically migrate `HrfAnalysisResult.group_designs` when migrating single-trial results.
- Do not change failure-log privacy policy, lifecycle events, or publication code in this increment.

## Review Focus

1. Candidate grid direction and regularization kind must remain aligned with score rows, including after NSD block assembly (Task 1).
2. Canonical and selected-HRF results must preserve coefficient units, NaN masks, original feature order, and nested trial metadata under both penalties (Task 2).
3. Design-container construction and public access must isolate caller-owned mutable tables/mappings (Task 2).
4. Prepared and conventional diagnostics must preserve their different input labels while sharing dimension and rank checks (Task 3).
5. Contrast metadata and nested-OLS thresholds must preserve analysis fingerprints and numerical rejection boundaries (Task 3).

## Proposed API migration

| Before | After |
| --- | --- |
| `RidgeCandidateScores`, `FractionCandidateScores` | `CandidateScores` |
| alpha candidate result `.alphas` | `.grid`, with `.regularization == "normalized_ridge"` |
| fraction candidate result `.fractions` | `.grid`, with `.regularization == "fractional_ridge"` |
| canonical trial result `.design_matrices` | `.design.matrices` |
| selected trial result `.group_designs` | `.design.matrices` (same `(run, hrf)` keys) |
| selected trial result `.hrf_indices` | `.design.hrf_indices` |
| selected trial result `.selection_provenance` | `.design.selection_provenance` |
| `HrfSingleTrialResult` | `SingleTrialResult` with `SelectedTrialDesign` |

`run_betas`, `trial_table`, all R² arrays, `diagnostics`, penalty fields, and
`provenance` remain directly on the trial result. The penalty-selection results
`RidgeSelection` and `FractionSelection` keep their existing names and fields:
choosing one shared alpha and choosing a fraction per feature are distinct operations.

## Files and interfaces

| File | Responsibility |
| --- | --- |
| `ridge_results.py` | `CandidateScores(regularization, grid, cv_r2, fold_sse, fold_sst, fold_hrf_indices, trial_masks, run_labels, provenance)` |
| `_ridge_cv.py`, `examples/NSD/ridge_workflow.py` | Construct that same schema locally and after assembling feature blocks |
| `single_trial_results.py` | `SingleTrialResult` plus `SharedTrialDesign` and `SelectedTrialDesign` composition |
| `hrf_results.py` | HRF selection/evaluation results only; remove duplicate trial result |
| `single_trial.py`, `_selected_hrf_fit.py` | Construct unified trial results with explicit keyword arguments |
| `examples/NSD/workflow_analysis.py` and identified callers | Migrate single-trial design access without changing conventional GLM access |
| new `_fit_diagnostics.py` | Shared rank warnings, nested-OLS constants/validation, and conventional/prepared result-dimension checks |
| `model.py` | Shared `contrast_metadata(contrasts)` representation |
| `fit.py`, `prepared_fit.py` | Use shared functions while retaining fit-specific scientific operations and metadata |
| `tests/test_result_schemas.py`, `tests/test_fit_diagnostics.py`, existing suites | New schema, ownership, boundary behavior, and existing scientific equivalence |
| `docs/api.md`, `docs/development.md`, `docs/result-migration.md`, validation note | API examples and actual verification evidence |

## Task 1: Consolidate candidate scores

- [x] Add a schema test to `tests/test_result_schemas.py` using the shared `ridge_problem` fixture:

```python
from dataclasses import replace

import numpy as np
import pytest

from boldtailor.fractional_ridge import score_fraction_candidates
from boldtailor.ridge_selection import score_ridge_candidates


@pytest.mark.parametrize("fractional", [False, True])
def test_candidate_scores_name_the_grid_and_scientific_basis(ridge_problem, fractional):
    from boldtailor import ridge_results

    data, predictors, _ = ridge_problem
    function = score_fraction_candidates if fractional else score_ridge_candidates
    options = {"fractions": [0.3, 1, 0.7]} if fractional else {"alphas": [1, 0, 0.1]}
    result = function(data, predictors, **options)
    assert type(result) is ridge_results.CandidateScores
    assert result.regularization == ("fractional_ridge" if fractional else "normalized_ridge")
    assert result.grid == ((1.0, 0.7, 0.3) if fractional else (0.0, 0.1, 1.0))
    assert result.cv_r2.shape == (3, data.n_features)
    np.testing.assert_allclose(
        result.cv_r2,
        1 - result.fold_sse.sum(axis=0) / result.fold_sst.sum(axis=0),
    )
    assert result.fold_hrf_indices.dtype == np.int64
    assert all(mask.dtype == bool for mask in result.trial_masks)
    with pytest.raises(ValueError, match="regularization"):
        replace(result, regularization="unknown")
```

  Update existing CV oracle tests from candidate-score `.alphas`/`.fractions`
  to `.grid` while preserving all expected scientific values, sorted ordering,
  provenance checks, and ownership assertions. Do not change selection-result
  `.alphas`/`.fractions` fields. Run the new schema tests and both CV suites;
  expect missing-class/field failures. Commit tests.

- [x] Replace the two candidate-score classes with this one field layout:

```python
@dataclass(frozen=True)
class CandidateScores:
    regularization: Literal["normalized_ridge", "fractional_ridge"]
    grid: tuple[float, ...]
    cv_r2: np.ndarray
    fold_sse: np.ndarray
    fold_sst: np.ndarray
    fold_hrf_indices: np.ndarray
    trial_masks: tuple[np.ndarray, ...]
    run_labels: tuple[str, ...]
    provenance: ProvenanceRecord
```

  Import `Literal` from `typing`. Reject any regularization kind other than the
  two declared values with `ValueError("unknown regularization kind")`.
  Reuse the existing ownership assignments once, changing the grid field name.
  Constructors use keyword arguments and supply the explicit kind. Preserve
  current grid validation/sorting in the public scoring functions; the container
  does not re-sort arrays or reinterpret scores. Update NSD `_global_scores` to
  construct the same type after block assembly.

- [x] Migrate identified consumers by object type and surrounding use, not a global
  string replacement. Search all source, examples, tests, Markdown, and notebooks
  for old class names and candidate-result grid access. Keep serialized provenance
  `alphas`/`fractions` fields unchanged. Historical plans and review reports stay
  as historical records; migrate executable code and current user documentation.
  Add the above migration table and a code
  example to `docs/result-migration.md`.

- [x] Run `uv run pytest tests/test_result_schemas.py tests/test_ridge_cv.py tests/test_fractional_cv.py tests/test_within_run_cv_isolation.py examples/NSD/test_ridge_workflow.py examples/NSD/test_fractional_workflow.py -q -W error`.
  Existing independent oracle, leakage, block-size, and real-process tests must
  pass unchanged numerically. Format changed Python, check whitespace, and commit.

## Task 2: Compose one trial result with explicit design information

- [x] Add to `tests/test_result_schemas.py`:

```python
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("fractional", [False, True])
def test_trial_results_share_science_and_name_their_design(
    selected_fixture, selected, fractional
):
    from boldtailor.single_trial import fit_single_trials, fit_selected_hrfs
    from boldtailor import single_trial_results as results

    data, selection = selected_fixture
    function = fit_selected_hrfs if selected else fit_single_trials
    options = dict(selection=selection, feature_signature="ordered-axis") if selected else {}
    options.update({"ridge_fraction": 0.5} if fractional else {"ridge_alpha": 0.1})
    result = function(data, **options)
    assert type(result) is results.SingleTrialResult
    expected_type = results.SelectedTrialDesign if selected else results.SharedTrialDesign
    assert type(result.design) is expected_type
    assert len(result.run_betas) == data.n_runs
    assert np.isnan(result.run_betas[0][:, -1]).all()
    if selected:
        np.testing.assert_array_equal(result.design.hrf_indices, selection.hrf_indices)
        assert result.design.selection_provenance == selection.provenance
        matrices = result.design.matrices
        assert (0, 1) in matrices
        matrices.clear()
        assert (0, 1) in result.design.matrices
    else:
        exposed = result.design.matrices[0]
        expected = exposed.copy(deep=True)
        exposed.iloc[:, :] = 0
        pd.testing.assert_frame_equal(result.design.matrices[0], expected)
```

  Add `import pandas as pd`. Adapt existing selected-HRF and canonical trial
  tests to the documented design access paths, keeping their independent
  augmented-regression/root-finding references and nested-event mutation checks.
  Add these direct constructor checks:

```python
def test_shared_trial_design_owns_its_input_table():
    from boldtailor.single_trial_results import SharedTrialDesign

    matrix = pd.DataFrame({"trial": [1.0, 2.0], "constant": [1.0, 1.0]})
    expected = matrix.copy(deep=True)
    design = SharedTrialDesign((matrix,))
    matrix.iloc[:, :] = 99
    pd.testing.assert_frame_equal(design.matrices[0], expected)


def test_selected_trial_design_owns_arrays_and_mapping(selected_fixture):
    from boldtailor.single_trial_results import SelectedTrialDesign

    _, selection = selected_fixture
    matrix = np.arange(6.0).reshape(3, 2)
    expected = matrix.copy()
    ids = np.array([1, -1])
    mapping = {(0, 1): matrix}
    design = SelectedTrialDesign(ids, mapping, selection.provenance)
    matrix[:] = 99
    ids[:] = 0
    mapping.clear()
    np.testing.assert_array_equal(design.matrices[0, 1], expected)
    np.testing.assert_array_equal(design.hrf_indices, [1, -1])
    assert not design.matrices[0, 1].flags.writeable
    assert not design.hrf_indices.flags.writeable
```

  Add a shared custom-HRF case to the schema test by calling
  `fit_single_trials(data, hrf=selection.library.candidates[1])` and requiring
  `SharedTrialDesign`. The name describes shared feature design, not a promise
  that every shared design uses canonical SPM. Run the new tests and affected
  trial suites; commit observed RED failures.

- [x] Define the two small containers in `single_trial_results.py`:

```python
@dataclass(frozen=True)
class SharedTrialDesign:
    _matrices: tuple[pd.DataFrame, ...]

    def __post_init__(self):
        object.__setattr__(self, "_matrices", tuple(d.copy(deep=True) for d in self._matrices))

    @property
    def matrices(self):
        return tuple(d.copy(deep=True) for d in self._matrices)


@dataclass(frozen=True)
class SelectedTrialDesign:
    hrf_indices: np.ndarray
    _matrices: dict[tuple[int, int], np.ndarray]
    selection_provenance: ProvenanceRecord

    def __post_init__(self):
        object.__setattr__(self, "hrf_indices", readonly_array(self.hrf_indices, dtype=np.int64))
        object.__setattr__(self, "_matrices", {k: readonly_array(v) for k, v in self._matrices.items()})

    @property
    def matrices(self):
        return dict(self._matrices)
```

  Change `SingleTrialResult._design_matrices` to
  `design: SharedTrialDesign | SelectedTrialDesign`, removing its design
  copying/property because the container owns that responsibility. Retain all
  common numerical, trial-table, diagnostic, penalty, and provenance fields and
  their current ownership logic. Remove `HrfSingleTrialResult` from `hrf_results.py`
  and now-unused imports. Both fitting entry points return `SingleTrialResult`.
  Use keyword constructors so scientific arrays cannot be confused by position.

- [x] Migrate single-trial consumers, including `_trial_designs` in the NSD workflow,
  using the documented access paths. Do not alter conventional `_glm_block` or
  `HrfAnalysisResult` access. Update result annotations and API documentation.
  Keep `.run_betas`, R², `.provenance`, `.trial_table`, and penalty access unchanged.

- [x] Run `uv run pytest tests/test_result_schemas.py tests/test_single_trial.py tests/test_selected_hrf_fit.py tests/test_fractional_ridge.py tests/test_hrf_glm.py examples/NSD -q -W error`.
  The conventional GLM suite guards against accidental migration of similarly
  named fields. Run full default pytest if notebook or validation-script consumers
  changed. Format, check the diff, and commit the unified result plus migration docs.

## Task 3: Share conventional/prepared diagnostics and contrast metadata

- [x] Add `tests/test_fit_diagnostics.py` boundary tests before extraction:

```python
import numpy as np
import pytest


def test_nested_ols_tolerance_keeps_roundoff_but_rejects_material_loss():
    from boldtailor._fit_diagnostics import validate_nested_ols_delta

    validate_nested_ols_delta(np.array([-1e-12, 0.0, 0.5]))
    with pytest.raises(ValueError, match="nested OLS monotonicity violated"):
        validate_nested_ols_delta(np.array([-1.1e-12]))


def test_rank_warning_names_the_affected_run():
    from boldtailor._fit_diagnostics import rank_warnings

    assert rank_warnings(3, 3, 2) == []
    assert rank_warnings(2, 3, 2) == ["run 2 design rank is 2 for 3 columns"]


def test_contrast_metadata_preserves_expression_weights_and_ownership():
    from boldtailor.model import contrast_metadata

    weights = {"face": 1.0, "house": -1.0}
    actual = contrast_metadata({"difference": weights, "expression": "face - house"})
    weights["face"] = 99
    assert actual == {
        "difference": {"kind": "weights", "weights": {"face": 1.0, "house": -1.0}},
        "expression": {"kind": "expression", "value": "face - house"},
    }
```

  Keep existing public invalid-parent-dimension and fingerprint tests. Move the
  existing test that calls `fit._validate_nested_ols_delta` to the shared function
  without altering its boundary values. Run the new tests and observe missing
  interfaces; commit before extraction.

- [x] Extract, without changing formulas or messages, into `_fit_diagnostics.py`:

```python
TASK_DELTA_R2_DEFINITION = "full_r2 - nuisance_r2"
DIAGNOSTIC_NOISE_MODEL = "ols"
NESTED_OLS_TOLERANCE = 1e-12


def validate_nested_ols_delta(raw_delta_r2):
    if np.any(raw_delta_r2 < -NESTED_OLS_TOLERANCE):
        raise ValueError("nested OLS monotonicity violated")


def rank_warnings(rank, columns, run):
    return [] if rank == columns else [f"run {run} design rank is {rank} for {columns} columns"]
```

  Extract the result-dimension traversal into the same module:

```python
def validate_result_dimensions(data, result, *, input_label):
    designs, run_r2 = result.design_matrices, result.run_r2
    if len(designs) != data.n_runs or len(run_r2) != data.n_runs:
        raise ValueError(f"full result run dimensions do not match {input_label}")
    for run, (signals, design, values) in enumerate(zip(data.signals, designs, run_r2, strict=True)):
        if design.shape[0] != signals.shape[0]:
            raise ValueError(f"full result run {run} dimensions do not match {input_label}")
        if values.shape != (data.n_features,):
            raise ValueError(f"full result run {run} feature dimensions do not match {input_label}")
    if result.r2.shape != (data.n_features,):
        raise ValueError(f"full result feature dimensions do not match {input_label}")
    if not np.isfinite(result.r2).all():
        raise ValueError("full result r-squared values must be finite")
```

  Pass `"data"` or `"prepared input"` to preserve existing error messages. Keep HRF
  grouped validation separate because its undefined-feature contract differs.

  Add the existing expression/weights representation to `model.py`:

```python
def contrast_metadata(contrasts):
    return {
        name: ({"kind": "expression", "value": value} if isinstance(value, str)
               else {"kind": "weights", "weights": dict(value)})
        for name, value in contrasts.items()
    }
```
 Replace duplicate serializers in `fit.py` and `prepared_fit.py`.
  Remove the no-op `drift_order` dictionary overwrite from `_model_provenance`;
  a plain `activity.copy()` preserves the current fingerprint payload.

- [x] Run `uv run pytest tests/test_fit_diagnostics.py tests/test_fit.py tests/test_prepared_fit.py tests/test_logging.py tests/test_hrf_glm.py -q -W error`.
  Existing parent-identity, event, provenance, and contrast tests must pass
  unchanged. Do not migrate logging policies or finite/NaN scientific rules.
  Format and commit after GREEN.

## Completion

Run `uv run pytest -q -W error`, the standard Black check, `git diff --check`,
`uv build --wheel --quiet`, and the isolated installed-wheel smoke command in
`docs/development.md`. Record RED/GREEN commits, schema migrations, preserved
scientific checks, and limitations in
`docs/validation/scientific-readability-results-2026-09-28.md`.
Request one independent review of the complete increment. Keep the refactoring
branch unmerged. The next plan combines fit lifecycle, logging, and provenance;
publication and imaging work remain visible in the roadmap.
