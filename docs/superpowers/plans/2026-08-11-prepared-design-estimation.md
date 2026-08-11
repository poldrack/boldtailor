# Prepared-Design Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an immutable prepared-design input and a pure conventional fitting path that FitLins can call without exposing image, PyBIDS, Nipype, or FitLins objects to Boldtailor.

**Architecture:** `PreparedDesignAnalysis` normalizes signals, labeled design matrices, timing, column roles, run metadata, and source provenance. A focused internal conventional-GLM module serves both the existing event-level `fit()` and the new `fit_prepared()` entry point; prepared task delta R-squared is a separate operation over explicitly classified task and nuisance columns.

**Tech Stack:** Python 3.12+, NumPy, pandas, Nilearn 0.14.x, pytest, uv, Git.

## Global Constraints

- Run every local command through `uv run`; use `uv run --no-project` for Git commands that do not need the project environment.
- Use pytest with strict RED-GREEN-Refactor: commit failing tests before production implementation for every behavior change.
- Never weaken a test or simplify a requirement to make an incomplete implementation pass.
- Keep every `__init__.py` completely empty; do not export symbols from package initializers.
- Preserve the existing `fit()`, `from_arrays()`, `ModelSpec`, `AnalysisData`, and result behavior. Preserve the `task_delta_r2()` public interface while changing its approved diagnostic semantics to nested OLS.
- Add no runtime dependency and do not modify dependency declarations or `uv.lock`.
- Support only semantic t contrasts and noise models `ols` and `ar1` in this phase; reject unsupported contrast types before fitting.
- `fit_prepared()` and `task_delta_r2_prepared()` perform no filesystem writes.
- Public NumPy inputs and outputs must own immutable float64 C-contiguous storage; callers cannot re-enable writeability. Public DataFrames return defensive copies.
- Prepared designs may differ in columns and column order across runs, but all nonzero contrast terms must exist and be estimable in every contributing run.
- Do not implement FitLins changes, BIDS discovery, image masking/reconstruction, F contrasts, HRF optimization, GLMdenoise, or fractional ridge in this plan.
- Execute in an isolated Git worktree because the main checkout contains unrelated user-owned notebook and dependency edits.

## File Structure

- Create `src/boldtailor/prepared.py`: immutable prepared input, normalization, role validation, run metadata, and design identity.
- Create `src/boldtailor/_conventional.py`: shared private conventional GLM, contrast, fixed-effects, and R-squared kernels.
- Create `src/boldtailor/prepared_fit.py`: public `fit_prepared()` and `task_delta_r2_prepared()` orchestration, logging, provenance, and result assembly.
- Modify `src/boldtailor/fit.py`: delegate existing numerical work to `_conventional.py` without changing public behavior.
- Modify `src/boldtailor/results.py`: allow scalar prepared-design provenance fields while retaining defensive copies.
- Modify `src/boldtailor/provenance.py`: expose narrowly scoped internal JSON-freezing helpers only if needed by prepared metadata; do not change serialized schema behavior.
- Create `tests/test_prepared.py`: prepared input validation, ownership, timing, roles, metadata, fingerprints, and normalization provenance.
- Create `tests/test_prepared_fit.py`: numerical parity, multi-run behavior, errors, provenance, privacy, no-I/O, and prepared delta R-squared.
- Modify `README.md`: document the advanced prepared-design path and its fixed-design limitation.
- Do not modify `src/boldtailor/__init__.py`.

---

### Task 0: Correct task delta R-squared to a nested OLS diagnostic

**Files:**
- Modify: `tests/test_fit.py`
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `src/boldtailor/fit.py`
- Modify: `examples/stop_signal_demo.ipynb`

**Interfaces:**
- Consumes: the existing `task_delta_r2(data, model, full_result)` interface and compiled full/nuisance designs.
- Produces: unchanged `TaskDeltaR2Result` fields whose `full_r2` and `nuisance_r2` are both computed with OLS; AR(1) remains the optional inferential model used by `fit()`.

- [ ] **Step 1: Write the changed statistical-contract tests**

Update the existing delta fixture to keep `ModelSpec.noise_model="ar1"`. Build
the full and nuisance designs independently, fit both with Nilearn
`run_glm(..., noise_model="ols")`, pool each model's SSE and centered SST across
runs, and assert `task_delta_r2()` matches those OLS oracles rather than
`full_result.r2`.

Require:

```python
comparison = task_delta_r2(data, model, full_result)
expected_raw = expected_full_ols_r2 - expected_nuisance_ols_r2

np.testing.assert_allclose(comparison.full_r2, expected_full_ols_r2)
np.testing.assert_allclose(comparison.nuisance_r2, expected_nuisance_ols_r2)
np.testing.assert_allclose(comparison.raw_delta_r2, expected_raw)
assert np.all(comparison.raw_delta_r2 >= -1e-12)
np.testing.assert_allclose(comparison.delta_r2, np.maximum(expected_raw, 0.0))
```

Add a focused test seam that supplies a raw difference below `-1e-12` and
requires a `ValueError` mentioning nested OLS monotonicity. Differences in the
closed interval `[-1e-12, 0)` remain valid numerical roundoff and are floored
to zero.

Update provenance assertions to require:

```python
assert activity["diagnostic_noise_model"] == "ols"
assert activity["inferential_noise_model"] == "ar1"
assert activity["nuisance_model"]["noise_model"] == "ols"
assert activity["clip_policy"] == "numerical_roundoff_guard"
```

Update notebook execution/metadata assertions to require the displayed and
published configuration to distinguish OLS diagnostic fits from AR(1)
inference and to use the title `Task-attributable delta R-squared (OLS diagnostic)`.

- [ ] **Step 2: Commit the RED requirement change**

Run:

```bash
uv run pytest tests/test_fit.py tests/test_stop_signal_demo.py -k "delta_r2 or variance_partition" -q -W error -p no:cacheprovider
```

Expected: failures show the implementation still reuses AR(1) full R-squared,
fits the nuisance model with AR(1), and publishes the old clipping rationale.

Commit tests only:

```bash
uv run --no-project git add tests/test_fit.py tests/test_stop_signal_demo.py
uv run --no-project git commit -m "test: define delta r-squared as nested ols"
```

- [ ] **Step 3: Implement OLS diagnostic refits**

Compile the full and nuisance designs once inside `task_delta_r2()`. Fit both
sets with the shared R-squared path using the literal diagnostic noise model
`"ols"`. Continue using the supplied `full_result` only for parent-fingerprint
and dimension validation. Keep `fit()` and all contrast outputs governed by
`model.noise_model`.

Refactor `_fit_nuisance_analysis()` into a short design-agnostic helper if
needed, but do not change ordinary `AnalysisResult.r2` behavior in this task.
Record the diagnostic and inferential noise models separately and describe the
zero floor as a numerical-roundoff guard. Reject any raw difference below
`-1e-12` before constructing the public comparison result.

- [ ] **Step 4: Update the notebook source cells**

Change only source cells needed to label the diagnostic and publish accurate
metadata. Keep the existing map count, filename, common slices, compact
display, and no-raw-signal guarantees. Stored outputs need not be retained.

- [ ] **Step 5: Verify and commit GREEN**

Run:

```bash
uv run pytest tests/test_fit.py tests/test_stop_signal_demo.py -k "delta_r2 or variance_partition" -q -W error -p no:cacheprovider
uv run pytest -q -W error -p no:cacheprovider
uv run black --check src tests examples/stop_signal_demo.py
uv run --no-project git diff --check
```

Commit production and notebook changes:

```bash
uv run --no-project git add src/boldtailor/fit.py examples/stop_signal_demo.ipynb
uv run --no-project git commit -m "fix: use ols for task variance partitioning"
```

---

### Task 1: Immutable prepared-design inputs

**Files:**
- Create: `tests/test_prepared.py`
- Create: `src/boldtailor/prepared.py`

**Interfaces:**
- Consumes: `RunSources`, `ProvenanceRecord`, and existing timing/source normalization semantics.
- Produces: `PreparedDesignAnalysis.from_arrays(...) -> PreparedDesignAnalysis` with `n_runs`, `n_features`, `signals`, `design_matrices`, `frame_times`, `timing_source`, `column_roles`, `run_metadata`, and `provenance` properties. Task 2 adds the two fingerprint properties under its own RED cycle.

- [ ] **Step 1: Write the public construction and ownership tests**

Create fixtures and tests that import the module directly, never through `boldtailor.__init__`:

```python
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from boldtailor.prepared import PreparedDesignAnalysis


@pytest.fixture
def prepared_inputs():
    signals = [
        np.arange(24, dtype=float).reshape(8, 3),
        np.arange(30, dtype=float).reshape(10, 3),
    ]
    designs = [
        pd.DataFrame({"face": [0, 1] * 4, "motion": np.linspace(0, 1, 8), "constant": 1.0}),
        pd.DataFrame({"constant": 1.0, "motion": np.linspace(0, 1, 10), "face": [0, 1] * 5}),
    ]
    roles = [
        {"face": "task", "motion": "nuisance", "constant": "intercept"},
        {"constant": "intercept", "motion": "nuisance", "face": "task"},
    ]
    metadata = [
        {"subject": "01", "session": "01", "task": "faces", "run": "1"},
        {"subject": "01", "session": "02", "task": "faces", "run": "1"},
    ]
    return signals, designs, roles, metadata


def test_prepared_analysis_owns_inputs_and_returns_defensive_state(prepared_inputs):
    signals, designs, roles, metadata = prepared_inputs
    original = deepcopy((signals, designs, roles, metadata))
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=designs,
        tr=2.0,
        column_roles=roles,
        run_metadata=metadata,
    )

    signals[0][:] = -1
    designs[0].iloc[:, :] = -1
    roles[0]["face"] = "other"
    metadata[0]["subject"] = "changed"

    assert prepared.n_runs == 2
    assert prepared.n_features == 3
    np.testing.assert_array_equal(prepared.signals[0], original[0][0])
    pd.testing.assert_frame_equal(prepared.design_matrices[0], original[1][0])
    assert prepared.column_roles[0]["face"] == "task"
    assert prepared.run_metadata[0]["subject"] == "01"
    assert isinstance(prepared.provenance, ProvenanceRecord)


def test_prepared_analysis_arrays_are_strictly_immutable(prepared_inputs):
    prepared = _make_prepared(prepared_inputs)
    for values in (*prepared.signals, *prepared.frame_times):
        assert values.flags.owndata
        assert values.dtype == np.float64
        assert values.flags.c_contiguous
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)
```

Add a helper `_make_prepared()` in the test module that constructs the object without mutating the fixture.

- [ ] **Step 2: Write table, timing, run-count, role, and metadata validation tests**

Parameterize focused failures for:

```python
@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda case: case[1].__setitem__(0, np.ones((8, 3))), "pandas DataFrame"),
        (lambda case: case[1][0].rename(columns={"face": ""}, inplace=True), "nonempty strings"),
        (lambda case: setattr(case[1][0], "columns", ["face", "face", "constant"]), "duplicate"),
        (lambda case: case[1][0].__setitem__("face", [True] * 8), "finite numeric"),
        (lambda case: case[1][0].__setitem__("face", [np.nan] * 8), "finite numeric"),
        (lambda case: case[2][0].pop("motion"), "one role per design column"),
        (lambda case: case[2][0].__setitem__("motion", "learned"), "invalid column role"),
        (lambda case: case[3][0].__setitem__("path", "/private/data"), "path-like"),
    ],
)
def test_prepared_analysis_rejects_invalid_design_inputs(
    prepared_inputs, mutator, message
):
    case = deepcopy(prepared_inputs)
    mutator(case)
    with pytest.raises(ValueError, match=message):
        _make_prepared(case)
```

Also test zero rows/columns, design/signal row mismatch, inconsistent feature counts, mismatched numbers of designs/roles/metadata, nonmapping metadata, and the existing exactly-one-of-`tr`-or-`frame_times` contract. Verify explicit `other` roles are accepted.

- [ ] **Step 3: Commit the RED tests**

Run:

```bash
uv run pytest tests/test_prepared.py -q -W error -p no:cacheprovider
```

Expected: collection fails with `ModuleNotFoundError: No module named 'boldtailor.prepared'`.

Commit only the test file:

```bash
uv run --no-project git add tests/test_prepared.py
uv run --no-project git commit -m "test: specify prepared design inputs"
```

- [ ] **Step 4: Implement `PreparedDesignAnalysis` normalization**

Create a frozen dataclass with private fields and defensive properties:

```python
ColumnRole = Literal["task", "nuisance", "intercept", "other"]


@dataclass(frozen=True)
class PreparedDesignAnalysis:
    _signals: tuple[np.ndarray, ...]
    _design_matrices: tuple[pd.DataFrame, ...]
    _frame_times: tuple[np.ndarray, ...]
    _timing_source: str
    _column_roles: tuple[Mapping[str, ColumnRole], ...]
    _run_metadata: tuple[Mapping[str, object], ...]
    _provenance: ProvenanceRecord

    @classmethod
    def from_arrays(
        cls,
        signals: np.ndarray | Sequence[np.ndarray],
        design_matrices: pd.DataFrame | Sequence[pd.DataFrame],
        *,
        tr: float | None = None,
        frame_times: np.ndarray | Sequence[np.ndarray] | None = None,
        column_roles: Mapping[str, str] | Sequence[Mapping[str, str]],
        sources: Sequence[RunSources] | None = None,
        run_metadata: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        provenance_metadata: Mapping[str, object] | None = None,
    ) -> PreparedDesignAnalysis:
        return _prepare_analysis(
            cls=cls,
            signals=signals,
            design_matrices=design_matrices,
            tr=tr,
            frame_times=frame_times,
            column_roles=column_roles,
            sources=sources,
            run_metadata=run_metadata,
            provenance_metadata=provenance_metadata,
        )
```

Use short helpers for run coercion, design validation, role validation, metadata freezing, and run-count checks. Normalize design values to owned float64 DataFrames with a reset `RangeIndex`. Store roles and metadata as immutable mappings and return fresh mappings/DataFrames from public accessors. Reuse existing signal and timing validation semantics; do not expose or accept events/confounds in this type.

Implement `_prepare_analysis()` as the short orchestration helper that calls the
validation/copy helpers and constructs the dataclass. Do not add fingerprint
properties in this task; Task 2 introduces them only after its value-sensitive
identity tests are committed RED.

- [ ] **Step 5: Run focused and existing data tests**

Run:

```bash
uv run pytest tests/test_prepared.py tests/test_data.py -q -W error -p no:cacheprovider
uv run black --check src/boldtailor/prepared.py tests/test_prepared.py
```

Expected: all selected tests pass and Black reports both files unchanged.

- [ ] **Step 6: Commit the GREEN implementation**

```bash
uv run --no-project git add src/boldtailor/prepared.py
uv run --no-project git commit -m "feat: add immutable prepared designs"
```

---

### Task 2: Value-sensitive design identity and normalization provenance

**Files:**
- Modify: `tests/test_prepared.py`
- Modify: `src/boldtailor/prepared.py`
- Modify: `src/boldtailor/provenance.py` only if a shared metadata-freezing helper is required

**Interfaces:**
- Consumes: normalized prepared runs from Task 1.
- Produces: stable `run_design_fingerprints`, aggregate `design_fingerprint`, normalization lifecycle events, and a canonical `ProvenanceRecord` that never serializes design values.

- [ ] **Step 1: Write deterministic fingerprint tests**

Add tests proving:

```python
def test_prepared_design_fingerprint_is_stable_and_value_sensitive(prepared_inputs):
    first = _make_prepared(prepared_inputs)
    reordered_metadata = deepcopy(prepared_inputs)
    reordered_metadata[3][0] = dict(reversed(tuple(reordered_metadata[3][0].items())))
    same = _make_prepared(reordered_metadata)
    changed = deepcopy(prepared_inputs)
    changed[1][0].loc[0, "motion"] += 0.25
    different = _make_prepared(changed)

    assert first.design_fingerprint == same.design_fingerprint
    assert first.run_design_fingerprints == same.run_design_fingerprints
    assert different.run_design_fingerprints[0] != first.run_design_fingerprints[0]
    assert different.design_fingerprint != first.design_fingerprint
```

Add separate assertions that column order, role changes, frame-time changes, and run order change the fingerprint. Changing only run-metadata mapping insertion order must not change it. Assert every fingerprint matches `[0-9a-f]{64}`.

- [ ] **Step 2: Write provenance and privacy tests**

Construct complete `RunSources` fixtures and assert:

- normalization emits `normalization_started` and `normalization_completed` with stage `prepared_design`;
- the activity name is `normalize_prepared_design`;
- activity records run/feature counts, timing source, ordered columns, roles, per-run fingerprints, and the aggregate design fingerprint;
- two constructions have different execution IDs but identical source and design fingerprints;
- anonymous sources retain the normal provenance-quality warning;
- canonical JSON contains the fingerprints and column names but none of several sentinel design values, signal values, absolute paths, or mutable object representations.

- [ ] **Step 3: Commit the RED tests**

Run the exact new nodes and verify failure because Task 1's placeholder identity is not value-sensitive and lacks the required provenance activity:

```bash
uv run pytest tests/test_prepared.py -k "fingerprint or provenance or privacy" -q -W error -p no:cacheprovider
```

Commit:

```bash
uv run --no-project git add tests/test_prepared.py
uv run --no-project git commit -m "test: specify prepared design identity"
```

- [ ] **Step 4: Implement canonical design hashing**

Use SHA-256 over an unambiguous versioned byte stream. Normalize every matrix and frame-time vector to little-endian float64 C order. Hash length-prefixed canonical JSON metadata followed by length-prefixed numerical bytes:

```python
_DESIGN_ID_SCHEMA = "boldtailor.prepared-design/1"


def _hash_chunk(hasher: Any, payload: bytes) -> None:
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _run_design_fingerprint(
    design: pd.DataFrame,
    frame_times: np.ndarray,
    roles: Mapping[str, ColumnRole],
) -> str:
    hasher = hashlib.sha256()
    _hash_chunk(hasher, _DESIGN_ID_SCHEMA.encode("utf-8"))
    header = {
        "shape": list(design.shape),
        "columns": list(design.columns),
        "roles": [roles[name] for name in design.columns],
        "dtype": "<f8",
    }
    _hash_chunk(hasher, _canonical_json(header).encode("utf-8"))
    _hash_chunk(hasher, np.asarray(frame_times, dtype="<f8", order="C").tobytes())
    _hash_chunk(hasher, np.asarray(design, dtype="<f8", order="C").tobytes())
    return hasher.hexdigest()
```

Compute the aggregate fingerprint from the ordered sequence of per-run fingerprints and the schema identifier. Do not hash signals. Source metadata continues to identify input signals under the existing provenance contract.

- [ ] **Step 5: Implement normalization logging and provenance**

Mirror the existing `from_arrays()` lifecycle with a fresh execution UUID, context binding, start/completion/failure events, source metadata fingerprint, and immutable activity. Use stage `prepared_design`; sanitize failure messages through the existing logging path. Store only aggregate diagnostics and fingerprints, never matrix values.

- [ ] **Step 6: Verify and commit GREEN**

Run:

```bash
uv run pytest tests/test_prepared.py tests/test_provenance.py tests/test_logging.py -q -W error -p no:cacheprovider
uv run black --check src/boldtailor/prepared.py src/boldtailor/provenance.py tests/test_prepared.py
uv run --no-project git diff --check
```

Commit only production changes:

```bash
uv run --no-project git add src/boldtailor/prepared.py src/boldtailor/provenance.py
uv run --no-project git commit -m "feat: fingerprint prepared designs"
```

If `provenance.py` did not require a change, omit it from `git add`.

---

### Task 3: Extract the shared conventional numerical kernel

**Files:**
- Create: `src/boldtailor/_conventional.py`
- Modify: `src/boldtailor/fit.py`

**Interfaces:**
- Consumes: signal arrays, labeled pandas designs, normalized semantic contrasts, and `ols`/`ar1`.
- Produces: private `ConventionalFit`, `RunFit`, `fit_designs(...)`, and `fit_r2_designs(...)` used by both public fitting paths.

- [ ] **Step 1: Establish the refactor baseline**

Run before editing:

```bash
uv run pytest tests/test_fit.py tests/test_multirun.py -q -W error -p no:cacheprovider
```

Expected: all tests pass. Save the test count in the task report. This task is the Refactor portion of the preceding GREEN cycle; it adds no behavior and therefore must not change tests.

- [ ] **Step 2: Move the numerical operations into `_conventional.py`**

Define focused internal types and entry points:

```python
@dataclass(frozen=True)
class RunFit:
    contrasts: Mapping[str, object]
    r2: np.ndarray
    residual_sum: np.ndarray
    total_sum: np.ndarray


@dataclass(frozen=True)
class ConventionalFit:
    run_fits: tuple[RunFit, ...]
    contrasts: Mapping[str, object]
    aggregate_r2: np.ndarray


def fit_designs(
    signals: Sequence[np.ndarray],
    designs: Sequence[pd.DataFrame],
    contrasts: Mapping[str, ContrastValue],
    noise_model: str,
) -> ConventionalFit:
    run_fits = tuple(
        _fit_run(signal, design, contrasts, noise_model, run)
        for run, (signal, design) in enumerate(zip(signals, designs, strict=True))
    )
    combined = _combine_contrasts(run_fits, tuple(contrasts))
    residual_sum = np.sum([run.residual_sum for run in run_fits], axis=0)
    total_sum = np.sum([run.total_sum for run in run_fits], axis=0)
    return ConventionalFit(run_fits, combined, _r2_from_sums(residual_sum, total_sum))


def fit_r2_designs(
    signals: Sequence[np.ndarray],
    designs: Sequence[pd.DataFrame],
    noise_model: str,
) -> np.ndarray:
    fits = tuple(
        _fit_glm(signal, design.to_numpy(), noise_model, run)
        for run, (signal, design) in enumerate(zip(signals, designs, strict=True))
    )
    residual_sum = np.sum([fit.residual_sum for fit in fits], axis=0)
    total_sum = np.sum([fit.total_sum for fit in fits], axis=0)
    return _r2_from_sums(residual_sum, total_sum)
```

Move `_fit_glm`, contrast resolution, estimability validation, rank warning,
residual-DOF validation, predictions, sums of squares, R-squared aggregation,
and fixed-effects combination. Keep functions short and preserve every warning
message and numerical tolerance exactly.

- [ ] **Step 3: Delegate existing `fit.py` behavior to the shared kernel**

Keep event/design compilation and event-specific provenance in `fit.py`.
Replace its local numerical loop with:

```python
numerical = fit_designs(
    data.signals,
    tuple(item.matrix for item in compiled),
    model.contrasts,
    model.noise_model,
)
```

Use `numerical.run_fits`, `numerical.contrasts`, and
`numerical.aggregate_r2` to assemble the unchanged `AnalysisResult`.
Delegate nuisance R-squared to `fit_r2_designs()`.

- [ ] **Step 4: Verify behavior preservation**

Run:

```bash
uv run pytest tests/test_fit.py tests/test_multirun.py -q -W error -p no:cacheprovider
uv run pytest -q -W error -p no:cacheprovider
uv run black --check src tests examples/stop_signal_demo.py
uv run --no-project git diff --check
```

Expected: the focused and full test counts match the pre-refactor baseline, and Black reports no changes required.

- [ ] **Step 5: Commit the refactor**

```bash
uv run --no-project git add src/boldtailor/_conventional.py src/boldtailor/fit.py
uv run --no-project git commit -m "refactor: share conventional glm kernel"
```

---

### Task 4: Prepared conventional fitting and Nilearn parity

**Files:**
- Create: `tests/test_prepared_fit.py`
- Create: `src/boldtailor/prepared_fit.py`
- Modify: `src/boldtailor/results.py`

**Interfaces:**
- Consumes: `PreparedDesignAnalysis`, semantic `ContrastValue` mappings, `ols`/`ar1`, and optional curated `model_metadata`.
- Produces: `fit_prepared(prepared, *, contrasts, noise_model="ar1", model_metadata=None) -> AnalysisResult`.

- [ ] **Step 1: Write single-run OLS and AR(1) parity tests**

Generate full-rank labeled designs and signals with fixed RNG seeds. Use
`nilearn.glm.first_level.run_glm()` and `compute_contrast()` directly as the
oracle. For both noise models assert all output arrays:

```python
@pytest.mark.parametrize("noise_model", ["ols", "ar1"])
def test_fit_prepared_matches_nilearn(noise_model, prepared_problem):
    prepared, contrasts, expected = prepared_problem(noise_model)
    result = fit_prepared(
        prepared,
        contrasts=contrasts,
        noise_model=noise_model,
        model_metadata={"origin": "fitlins", "node": "run"},
    )

    np.testing.assert_allclose(result.effect("face_gt_house"), expected.effect_size())
    np.testing.assert_allclose(result.variance("face_gt_house"), expected.effect_variance())
    np.testing.assert_allclose(result.stat("face_gt_house"), expected.stat())
    np.testing.assert_allclose(result.z_score("face_gt_house"), expected.z_score())
    np.testing.assert_allclose(
        result.one_sided_p_value("face_gt_house"), expected.p_value()
    )
```

Reuse the existing warning filters and tolerances already justified in
`tests/test_fit.py`; do not broaden them.

- [ ] **Step 2: Write multi-run and error tests**

Test two run-specific designs whose common contrast terms occur in different
orders and whose unrelated columns differ. Compare the combined effect and
variance with the same equal-weight Nilearn contrast arithmetic used by the
existing multi-run tests. Assert pooled R-squared is computed from summed SSE
and SST, not averaged run R-squared.

Add focused failures for:

- empty contrast mappings;
- empty, nonnumeric, nonfinite, boolean, and all-zero weights;
- invalid expressions;
- referenced terms absent from one run;
- non-estimable contrasts;
- nonpositive residual degrees of freedom; and
- noise models other than `ols` and `ar1`.

- [ ] **Step 3: Write result ownership tests**

Assert prepared designs appear in `AnalysisResult.design_matrices` as defensive
copies. Assert `design_provenance` contains only scalar fields:

```python
assert result.design_provenance == (
    {
        "source": "prepared",
        "design_fingerprint": prepared.run_design_fingerprints[0],
    },
)
```

Attempt to mutate every result array and re-enable writeability. Verify the
stored result remains unchanged.

- [ ] **Step 4: Commit RED tests**

Run:

```bash
uv run pytest tests/test_prepared_fit.py -q -W error -p no:cacheprovider
```

Expected: collection fails because `boldtailor.prepared_fit` does not exist.

Commit:

```bash
uv run --no-project git add tests/test_prepared_fit.py
uv run --no-project git commit -m "test: specify prepared design fitting"
```

- [ ] **Step 5: Implement `fit_prepared()`**

Normalize contrasts using the existing `ModelSpec` semantic validation helpers
without adding HRF or drift defaults to prepared provenance. Validate
`noise_model` before calling the kernel. Freeze `model_metadata` using the same
curated JSON and path restrictions as provenance annotations.

Implement orchestration with short private helpers:

```python
def fit_prepared(
    prepared: PreparedDesignAnalysis,
    *,
    contrasts: Mapping[str, ContrastValue],
    noise_model: str = "ar1",
    model_metadata: Mapping[str, object] | None = None,
) -> AnalysisResult:
    fit_spec = _prepare_fit_spec(contrasts, noise_model, model_metadata)
    numerical = fit_designs(
        prepared.signals,
        prepared.design_matrices,
        fit_spec.contrasts,
        fit_spec.noise_model,
    )
    return make_result(
        numerical.contrasts,
        prepared.design_matrices,
        _prepared_design_provenance(prepared),
        tuple(run.r2 for run in numerical.run_fits),
        numerical.aggregate_r2,
        provenance,
    )
```

Task 5 supplies the complete fit provenance. During this task, construct a
minimal valid extension of the prepared provenance so all numerical behavior
is testable; do not omit provenance from the returned result.

Update `DesignProvenance` in `results.py` from `Mapping[str, int | float]` to
`Mapping[str, int | float | str]`. Keep values scalar so existing shallow
immutable wrapping remains sufficient.

- [ ] **Step 6: Verify and commit GREEN**

Run:

```bash
uv run pytest tests/test_prepared_fit.py tests/test_fit.py tests/test_multirun.py -q -W error -p no:cacheprovider
uv run black --check src/boldtailor/prepared_fit.py src/boldtailor/results.py tests/test_prepared_fit.py
uv run --no-project git diff --check
```

Commit:

```bash
uv run --no-project git add src/boldtailor/prepared_fit.py src/boldtailor/results.py
uv run --no-project git commit -m "feat: fit prepared design matrices"
```

---

### Task 5: Prepared-fit identity, logging, privacy, and no-I/O guarantee

**Files:**
- Modify: `tests/test_prepared_fit.py`
- Modify: `src/boldtailor/prepared_fit.py`

**Interfaces:**
- Consumes: prepared design/source fingerprints and normalized prepared fit settings.
- Produces: deterministic analysis fingerprints, complete fit activity, lifecycle logs, sanitized failures, and a tested zero-write contract.

- [ ] **Step 1: Write analysis identity and fit-activity tests**

With complete source metadata, assert repeated equivalent fits have distinct
execution IDs and identical analysis fingerprints. Assert the fingerprint
changes when any design value, contrast, noise model, or model metadata value
changes. With anonymous sources, assert the analysis fingerprint remains
unavailable and the provenance-quality warning remains present.

Require the final activity to have this bounded structure:

```python
activity = result.provenance.activities[-1]
assert activity["name"] == "fit_prepared"
assert activity["stage"] == "fit"
assert activity["model"] == {
    "kind": "prepared_design",
    "contrasts": {
        "face_gt_house": {
            "kind": "weights",
            "weights": {"face": 1.0, "house": -1.0},
        }
    },
    "noise_model": "ar1",
    "metadata": {"origin": "fitlins", "node": "run"},
    "design_fingerprint": prepared.design_fingerprint,
}
```

Require per-run diagnostics to include scan/feature counts, ordered columns,
role counts, rank, residual DOF, run design fingerprint, and rank warnings—no
design or signal values.

- [ ] **Step 2: Write lifecycle, failure, privacy, and filesystem tests**

Capture the `boldtailor` logger and require `fit_started`/`fit_completed` or
`fit_failed` with context reset after either outcome. Search canonical JSON and
captured log records for sentinel signal/design values, absolute paths,
environment-variable values, tracebacks, object reprs, and memory addresses.

Patch all ordinary write surfaces:

```python
monkeypatch.setattr(builtins, "open", guarded_open)
monkeypatch.setattr(io, "open", guarded_io_open)
monkeypatch.setattr(os, "open", fail_os_open)
monkeypatch.setattr(Path, "open", fail_path_open)
monkeypatch.setattr(Path, "write_text", fail_write)
monkeypatch.setattr(Path, "write_bytes", fail_write)
```

Permit read mode if an imported dependency needs it, but fail any mode or flags
that can create, truncate, or modify a file. Call `fit_prepared()` and assert it
returns successfully without invoking a write surface.

- [ ] **Step 3: Commit RED tests**

Run:

```bash
uv run pytest tests/test_prepared_fit.py -k "fingerprint or activity or lifecycle or privacy or filesystem" -q -W error -p no:cacheprovider
```

Expected: intended failures against Task 4's minimal provenance.

Commit:

```bash
uv run --no-project git add tests/test_prepared_fit.py
uv run --no-project git commit -m "test: specify prepared fit provenance"
```

- [ ] **Step 4: Implement complete fit provenance**

Build a deterministic model identity containing the prepared design
fingerprint, serialized contrasts, noise model, and curated model metadata.
Compute the analysis fingerprint with the existing
`analysis_fingerprint(source_fingerprint, model_identity)` helper so anonymous
or incomplete sources still prevent a reproducible analysis identity.

Mirror the existing fit lifecycle using a fresh UUID and context-local
`execution_id`, `data_id`, and `analysis_id`. Extend—not mutate—the prepared
normalization provenance. Reuse the existing rank-warning text and aggregate
diagnostics. Do not serialize numerical arrays.

- [ ] **Step 5: Verify and commit GREEN**

Run:

```bash
uv run pytest tests/test_prepared_fit.py tests/test_logging.py tests/test_provenance.py -q -W error -p no:cacheprovider
uv run black --check src/boldtailor/prepared_fit.py tests/test_prepared_fit.py
uv run --no-project git diff --check
```

Commit:

```bash
uv run --no-project git add src/boldtailor/prepared_fit.py
uv run --no-project git commit -m "feat: record prepared fit provenance"
```

---

### Task 6: Prepared task delta R-squared

**Files:**
- Modify: `tests/test_prepared_fit.py`
- Modify: `src/boldtailor/prepared_fit.py`

**Interfaces:**
- Consumes: `PreparedDesignAnalysis`, its matching full `AnalysisResult`, the original contrasts/noise/model metadata, and complete column roles.
- Produces: `task_delta_r2_prepared(...) -> TaskDeltaR2Result` with nuisance designs selected only from `nuisance` and `intercept` columns.

- [ ] **Step 1: Write the role-partition and numerical tests**

Define the public signature:

```python
def task_delta_r2_prepared(
    prepared: PreparedDesignAnalysis,
    full_result: AnalysisResult,
    *,
    contrasts: Mapping[str, ContrastValue],
    noise_model: str = "ar1",
    model_metadata: Mapping[str, object] | None = None,
) -> TaskDeltaR2Result:
    return _compare_prepared_models(
        prepared,
        full_result,
        contrasts=contrasts,
        noise_model=noise_model,
        model_metadata=model_metadata,
    )
```

Create nested full/nuisance synthetic models. The full `AnalysisResult` may use
AR(1) for inference, but both variance-partition fits must use OLS. Assert:

- nuisance matrices contain exactly `nuisance` and `intercept` columns in their
  original order;
- nuisance R-squared is pooled from summed SSE/SST across runs;
- `raw_delta_r2 == full_r2 - nuisance_r2` and every raw value is at least
  `-1e-12`;
- public `delta_r2 == maximum(raw_delta_r2, 0)`;
- the negative count and raw minimum are exact;
- all arrays own strict immutable storage; and
- nuisance design accessors are defensive copies.

- [ ] **Step 2: Write identity and invalid-role tests**

Require explicit failures when:

- any run contains an `other` role;
- no task column exists;
- no nuisance or intercept column exists;
- the supplied full result comes from changed source metadata, design values,
  contrasts, noise model, or model metadata;
- result dimensions do not match the prepared input; or
- source metadata are insufficient to establish the parent analysis identity.

Use a focused test seam to require rejection of a raw nested-OLS difference
below `-1e-12`; retain values in `[-1e-12, 0)` only for the numerical floor.

Require a separate `task_delta_r2_prepared` provenance activity with parent
analysis ID, definition `full_r2 - nuisance_r2`, diagnostic noise model `ols`,
separate inferential noise model, numerical-roundoff floor disclosure, exact
role-based nuisance rule, run diagnostics, and bounded aggregate diagnostics.

- [ ] **Step 3: Commit RED tests**

Run:

```bash
uv run pytest tests/test_prepared_fit.py -k "task_delta_r2_prepared" -q -W error -p no:cacheprovider
```

Expected: import or attribute failure because the operation is absent.

Commit:

```bash
uv run --no-project git add tests/test_prepared_fit.py
uv run --no-project git commit -m "test: specify prepared task delta r-squared"
```

- [ ] **Step 4: Implement role-based nuisance fitting**

Use a helper that requires a complete binary partition for this diagnostic:

```python
def _nuisance_designs(
    prepared: PreparedDesignAnalysis,
) -> tuple[pd.DataFrame, ...]:
    designs = []
    for run, (matrix, roles) in enumerate(
        zip(prepared.design_matrices, prepared.column_roles, strict=True)
    ):
        if "other" in roles.values():
            raise ValueError(f"run {run} column roles are incomplete for task delta r-squared")
        task = [name for name in matrix if roles[name] == "task"]
        nuisance = [name for name in matrix if roles[name] in {"nuisance", "intercept"}]
        if not task:
            raise ValueError(f"run {run} requires at least one task column")
        if not nuisance:
            raise ValueError(f"run {run} requires a nuisance or intercept column")
        designs.append(matrix.loc[:, nuisance])
    return tuple(designs)
```

Recreate the full fit identity from the exact public arguments and require it
to equal `full_result.provenance.analysis_fingerprint`. Delegate both the full
and nuisance diagnostic fits to `fit_r2_designs()` with the literal noise model
`"ols"`; retain the public `noise_model` argument only to identify and validate
the parent inferential fit. Use the existing `make_task_delta_r2_result()` so
the numerical floor and owned-storage behavior remain centralized.
Reject a raw difference below `-1e-12` before calling the result factory.

- [ ] **Step 5: Verify and commit GREEN**

Run:

```bash
uv run pytest tests/test_prepared_fit.py tests/test_fit.py -q -W error -p no:cacheprovider
uv run black --check src/boldtailor/prepared_fit.py tests/test_prepared_fit.py
uv run --no-project git diff --check
```

Commit:

```bash
uv run --no-project git add src/boldtailor/prepared_fit.py
uv run --no-project git commit -m "feat: compare prepared task variance"
```

---

### Task 7: Documentation and final verification

**Files:**
- Modify: `README.md`
- Verify only: `src/boldtailor/__init__.py`

**Interfaces:**
- Consumes: the final public APIs from Tasks 1, 4, and 6.
- Produces: concise usage documentation and release-quality verification evidence.

- [ ] **Step 1: Document the prepared-design path**

Add a short advanced-usage section that imports from modules directly:

```python
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared

prepared = PreparedDesignAnalysis.from_arrays(
    signals=signals,
    design_matrices=design_matrices,
    frame_times=frame_times,
    column_roles=column_roles,
    sources=sources,
    run_metadata=run_metadata,
)
result = fit_prepared(
    prepared,
    contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
    noise_model="ar1",
    model_metadata={"origin": "fitlins", "node": "run"},
)
```

State explicitly that this API accepts already-compiled fixed designs, performs
no image or file I/O, supports semantic t contrasts only in this phase, and is
not the future optimized-HRF or GLMdenoise interface.

- [ ] **Step 2: Commit documentation**

```bash
uv run --no-project git add README.md
uv run --no-project git commit -m "docs: explain prepared design estimation"
```

- [ ] **Step 3: Run the complete verification suite**

Run exactly:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -W error -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 uv run black --check src tests examples/stop_signal_demo.py
uv lock --check
uv run --no-project git diff --check HEAD~12..HEAD
uv run --no-project python -c 'from pathlib import Path; data = Path("src/boldtailor/__init__.py").read_bytes(); assert data == b""'
uv run --no-project git status --short
```

Expected:

- all tests pass with warnings treated as errors;
- Black reports every checked file unchanged;
- the lock is current and unchanged from the task base;
- the feature range has no whitespace errors;
- `src/boldtailor/__init__.py` is exactly zero bytes; and
- status contains no task-related uncommitted files or generated artifacts.

Use the actual merge-base or recorded task base instead of `HEAD~12` if the
number of RED/GREEN/refactor commits differs. Verify explicitly that
`pyproject.toml` and `uv.lock` have no diff from the task base.

- [ ] **Step 4: Review against the approved specification**

Confirm each requirement in
`docs/superpowers/specs/2026-08-11-fitlins-boldtailor-interoperability-design.md`
that belongs to prepared-design estimation is implemented. Record FitLins
integration, standalone BIDS adapters, HRF optimization, GLMdenoise,
fractional ridge, and F contrasts as intentionally deferred rather than gaps.

Do not modify or commit user-owned notebook outputs or dependency changes from
the main checkout.
