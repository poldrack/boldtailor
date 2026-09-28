# Ordinary Array Ownership and Prepared-Design Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the preserved native execution method. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace special array immutability machinery with ordinary owned NumPy arrays and eliminate repeated copies of every prepared design during fitting.

**Architecture:** One `readonly_array` function owns and marks numeric arrays read-only. Public table accessors continue returning copies; package fitting code uses owned private tables directly without mutating them. Each result class handles its own optional fractional arrays instead of allowing a numerical module to mutate the result.

**Tech Stack:** Python >=3.12, NumPy, pandas, pytest, uv; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-28-scientific-readability-design.md`

**Status:** Approved by the user; implementation in progress.

This is the ownership portion of roadmap Stage 3. Result-schema consolidation
and shared fit lifecycle/diagnostics remain the next portion of Stage 3; they
are not claimed complete by this plan. Keeping this boundary separates an
explicit mutation-contract change from later result-schema and logging changes.

## Global Constraints

- Work on `refactor/scientific-readability`; leave main and stash `c4cfbec5fccd5397ea650c550f5b485ea588c2cd` unchanged.
- Use `uv run` for Python, formatting, and pytest. Keep every `__init__.py` empty.
- Observe and commit failing tests before implementation. Update existing tests only for the explicitly changed ownership contract.
- Preserve scientific arrays, dtypes, feature/run order, NaNs, contrasts, HRFs, CV objectives, and existing numerical tolerances.
- Own inputs at construction. Public numeric arrays are ordinary read-only NumPy arrays: this prevents accidental writes, not deliberate `setflags(write=True)` or private-state mutation.
- Public DataFrame/dictionary accessors still return defensive copies, including nested event metadata.
- Do not add per-access array copies, proxy objects, array subclasses, or a general fitting framework.
- Do not change public result field names or provenance schemas in this increment.

## Review Focus

1. Strided input arrays and ndarray subclasses must become independent, C-contiguous base NumPy arrays (Task 1).
2. Float, integer-index, and boolean-mask results must retain exact values and dtypes after ownership unification (Task 1).
3. Removing bulk table copies must not allow fitting or diagnostics to mutate prepared input designs or roles (Task 2).
4. Multi-run fits with different column orders and lengths must retain numerical and provenance diagnostics (Task 2).
5. Fraction maps and implied-alpha arrays must be owned by both canonical and selected-HRF results, including NaN exclusions (Task 3).

## Files and interfaces

| File | Responsibility |
| --- | --- |
| `src/boldtailor/_arrays.py` | `readonly_array(values, *, dtype=np.float64)` copies input into an ordinary C-contiguous ndarray and marks it read-only |
| `data.py`, `hrf_library.py`, `results.py`, `hrf_glm_results.py`, `single_trial_results.py`, `hrf_results.py`, `ridge_results.py`, `_fractional_ridge.py` | Call the shared ownership function, supplying integer/bool dtypes where needed |
| `prepared_fit.py` | Read owned prepared tables internally; public accessors remain unchanged |
| `single_trial_results.py`, `hrf_results.py` | Own their optional fractional fields locally |
| `tests/test_arrays.py`, existing ownership tests | New ownership contract, dtype preservation, no aliases to caller inputs |
| `tests/test_prepared_fit.py` | No bulk defensive accessor in internal fitting; independent numerical equivalence and unchanged inputs |
| `docs/development.md`, `docs/api.md`, `docs/user-guide.md`, migration and validation notes | Explain ownership guarantees and their limits |

## Task 1: One ordinary NumPy ownership function

- [ ] Add `tests/test_arrays.py` with actual dtype, layout, aliasing, and accidental-write checks:

```python
import numpy as np
import pytest

from boldtailor import _arrays


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.bool_])
def test_readonly_array_owns_values_and_keeps_dtype(dtype):
    source = np.arange(24).reshape(4, 6).astype(dtype)[:, ::2]
    expected = source.copy()
    result = _arrays.readonly_array(source, dtype=dtype)
    assert type(result) is np.ndarray
    assert result.flags.owndata and result.flags.c_contiguous
    assert result.dtype == np.dtype(dtype)
    assert not np.shares_memory(result, source)
    assert not result.flags.writeable
    np.testing.assert_array_equal(result, expected)
    source[...] = 0
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError):
        result.flat[0] = 0
    editable = result.copy()
    editable.flat[0] = 0
    assert editable.flags.writeable


def test_readonly_array_removes_subclass_behavior():
    class CallerArray(np.ndarray):
        pass

    source = np.arange(6.0).view(CallerArray)
    result = _arrays.readonly_array(source)
    assert type(result) is np.ndarray
    assert not np.shares_memory(result, source)
```

- [ ] Update existing array ownership tests before implementation. Replace assertions that `setflags(write=True)` must fail with actual element-assignment rejection on the initially read-only array. Retain all input-mutation, value, shape, dtype, copied-metadata, and scientific assertions. Rename tests claiming strict immutability. Add `type(values) is np.ndarray` checks where old float-subclass outputs are inspected. Explain that this is the approved ordinary-NumPy contract, not a workaround for a failing implementation.

  Relevant files are `test_data`, `test_prepared`, `test_prepared_fit`, `test_fit`, `test_single_trial`, `test_selected_hrf_fit`, `test_hrf_library`, `test_hrf_selection`, `test_hrf_glm`, `test_ridge_selection`, `test_ridge_cv`, and `test_trial_encoding`. Search all tests/examples for `setflags` and strict-immutability wording; do not modify unrelated scientific checks.

- [ ] Run `uv run pytest tests/test_arrays.py tests/test_data.py tests/test_prepared.py -q -W error`. Expect missing-function and base-array-type failures. Commit tests before implementation.

- [ ] Replace `_arrays.py`'s subclass/helper with:

```python
def readonly_array(values, *, dtype=np.float64):
    """Own numeric values and prevent accidental writes through this array."""
    array = np.array(values, dtype=dtype, order="C", copy=True, subok=False)
    array.setflags(write=False)
    return array
```

  Update float-array call sites. Replace `immutable_indices` with `readonly_array(..., dtype=np.int64)` and `immutable_bool_array` with `readonly_array(..., dtype=bool)` at their current callers; remove the two byte-buffer helpers. Do not introduce a compatibility alias for these private implementation helpers. Search the repository to migrate any validation-script imports too. Preserve default float64 conversion and NaNs.

- [ ] Document the changed contract with this user example:

```python
editable_betas = result.run_betas[0].copy()
editable_betas[:, 0] = 0
```

  State that deliberately making an exposed array writable can invalidate its owning analysis/result and provenance assumptions; callers should copy before editing. This is a normal ownership convention, not enforced immutability.

- [ ] Run the full default `uv run pytest -q -W error`, Black in the CI scope, and `git diff --check`. Resolve unintended dtype/value changes without changing expected numerical tolerances. Commit implementation and the ownership migration note.

## Task 2: Stop copying all prepared tables inside fitting

- [ ] Add this regression in `tests/test_prepared_fit.py`, using its existing multi-run `prepared_delta_problem` fixture and `_ols_r2_oracle`:

```python
def test_prepared_fitting_uses_owned_designs_without_bulk_copy(
    prepared_delta_problem, monkeypatch
):
    prepared, contrasts, metadata, _ = prepared_delta_problem
    original_designs = prepared.design_matrices
    original_roles = prepared.column_roles
    expected_full = _ols_r2_oracle(prepared.signals, original_designs)
    nuisance_designs = tuple(
        design.loc[:, [name for name in design if roles[name] in {"nuisance", "intercept"}]]
        for design, roles in zip(original_designs, original_roles, strict=True)
    )
    expected_null = _ols_r2_oracle(prepared.signals, nuisance_designs)

    def bulk_copy_forbidden(self):
        raise AssertionError("internal fitting copied every prepared table")

    with monkeypatch.context() as patch:
        patch.setattr(PreparedDesignAnalysis, "design_matrices", property(bulk_copy_forbidden))
        patch.setattr(PreparedDesignAnalysis, "column_roles", property(bulk_copy_forbidden))
        full = fit_prepared(prepared, contrasts=contrasts, noise_model="ols", model_metadata=metadata)
        delta = task_delta_r2_prepared(
            prepared, full, contrasts=contrasts, noise_model="ols", model_metadata=metadata
        )
    np.testing.assert_allclose(full.r2, expected_full)
    np.testing.assert_allclose(delta.full_r2, expected_full)
    np.testing.assert_allclose(delta.nuisance_r2, expected_null)
    for actual, expected in zip(prepared.design_matrices, original_designs, strict=True):
        pd.testing.assert_frame_equal(actual, expected)
    assert prepared.column_roles == original_roles
```

  Keep existing public-accessor mutation-isolation tests. This test exercises the full multi-run fit and comparison, not only an accessor count.

- [ ] Run the new test; expect the bulk-copy assertion to fail in current `fit_prepared`. Commit it.

- [ ] In `prepared_fit.py`, read `prepared._design_matrices` and `prepared._column_roles` for package-internal fitting and diagnostics. These are already owned at construction. Bind them once where a function makes several passes; keep them local and do not mutate or expose them as a new public API. `_nuisance_designs` selects nuisance columns as before. Result factories continue making their own copies. Public `PreparedDesignAnalysis.design_matrices` and `column_roles` remain defensive copies.

  Document this narrow internal convention next to `PreparedDesignAnalysis` and in the developer guide. Do not add a wrapper class merely to access an existing tuple.

- [ ] Run `uv run pytest tests/test_prepared.py tests/test_prepared_fit.py tests/test_fit.py tests/test_logging.py -q -W error`. Existing contrast, pooled-R², run diagnostics, fingerprints, and lifecycle checks must pass unchanged. Format and commit.

## Task 3: Make fractional result ownership local

- [ ] Add this boundary/ownership regression in `tests/test_fractional_ridge.py`:

```python
@pytest.mark.parametrize("selected", [False, True])
def test_fraction_results_own_arrays_without_solver_mutation(
    selected_fixture, selected, monkeypatch
):
    from dataclasses import replace
    from boldtailor import _fractional_ridge

    data, selection = selected_fixture
    fit = fit_selected_hrfs if selected else fit_single_trials
    options = dict(selection=selection, feature_signature="ordered-axis") if selected else {}

    def forbidden(*args):
        raise AssertionError("numerical module must not mutate a result instance")

    monkeypatch.setattr(_fractional_ridge, "freeze_fraction_result", forbidden, raising=False)
    result = fit(data, ridge_fraction=[1, 0.7, 0.4, 0.2, np.nan], **options)
    fractions = result.ridge_fraction.copy()
    alphas = tuple(a.copy() for a in result.run_ridge_alphas)
    copied = replace(result, ridge_fraction=fractions, run_ridge_alphas=alphas)
    fractions[:] = 0.1
    for array in alphas:
        array[:] = 999
    np.testing.assert_array_equal(copied.ridge_fraction, result.ridge_fraction)
    for actual, expected in zip(copied.run_ridge_alphas, result.run_ridge_alphas, strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable
    assert not copied.ridge_fraction.flags.writeable
```

  Keep the existing independent oracle test for both result classes. Run the new
  regression and observe the current result class calling the forbidden
  numerical-module mutation hook. Commit tests before implementation.

- [ ] In each result class's `__post_init__`, own the optional fields explicitly:

```python
if self.ridge_fraction is not None:
    object.__setattr__(self, "ridge_fraction", readonly_array(self.ridge_fraction))
    object.__setattr__(
        self, "run_ridge_alphas", tuple(readonly_array(a) for a in self.run_ridge_alphas)
    )
```

  Remove the import/call of `freeze_fraction_result` from both result classes and remove the helper and now-unused array import from `_fractional_ridge.py`. A few explicit local assignments are preferable to a cross-module function that mutates another module's frozen object. Result-schema composition is a later Stage 3 change, so do not introduce extra result types here.

- [ ] Run `uv run pytest tests/test_fractional_ridge.py tests/test_single_trial.py tests/test_selected_hrf_fit.py -q -W error`, then the full default suite, formatting, wheel build, and isolated installed-wheel smoke. Record RED/GREEN commits and observations in `docs/validation/scientific-readability-ownership-2026-09-28.md`. Commit and request one independent review of the complete ownership increment.

## Completion and remaining scope

All numeric values and metadata must remain equivalent under existing tests.
The deliberate public change is ordinary read-only NumPy ownership rather than
an attempted strict-immutability promise. No result schema migration is included.
The next Stage 3 plan still needs candidate-score consolidation, shared trial
result composition, common diagnostics, and a small fit lifecycle helper.
Stages 4 and 5 remain provenance/publication and packaged imaging workflows.
