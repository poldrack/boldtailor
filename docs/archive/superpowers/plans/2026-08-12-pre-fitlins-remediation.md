# Pre-FitLins Remediation Implementation Plan

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the verified performance, privacy, anonymous-diagnostic, notebook-configuration, and versioning defects before conventional FitLins integration.

**Architecture:** Preserve the public array/DataFrame estimator API while adding private trusted design access and private in-process lineage. Centralize categorical failure events and safe version lookup in small internal modules. Keep the stop-signal notebook explicit and temporary, but make its configuration portable and its tests semantic rather than cosmetic.

**Tech Stack:** Python 3.12, NumPy, pandas, Nilearn, NiBabel, nbformat/nbclient, pytest, uv

## Global Constraints

- Use `uv` for package management and `uv run` for every local Python, pytest, Black, and lock command.
- Follow strict RED-GREEN-Refactor: commit failing tests before production changes; never weaken a test to accept an incomplete solution.
- Prefer short, focused functions and modules.
- Every `__init__.py` remains completely empty.
- Public fitting signatures, result accessors, deterministic fingerprint rules, and defensive-copy guarantees remain compatible.
- Internal lineage tokens are never public, logged, serialized, or included in deterministic fingerprints.
- Structured failure logs contain no exception message, traceback, path, environment value, object repr, memory address, signal, table, design, estimate, or statistic.
- `BOLDTAILOR_BIDS_ROOT` is the only required machine-specific notebook environment setting; session selection is an ordinary five-session notebook variable.
- No new dependency or lockfile change.
- Do not add FitLins code, imaging extraction, lifecycle consolidation, publication redesign, optimized HRFs, GLMdenoise, fractional ridge, or GLMsingle.
- The untracked architecture review and generated cache/coverage/egg-info artifacts belong to the user; do not commit or delete them.

## File Responsibilities

| File | Responsibility in this phase |
|---|---|
| `src/boldtailor/logging.py` | Emit ordinary lifecycle events and shared categorical failure events. |
| `src/boldtailor/data.py` | Normalize event-level inputs, assign private lineage, and use categorical normalization failures. |
| `src/boldtailor/prepared.py` | Normalize prepared designs, assign lineage, expose private trusted design access, and use categorical failures. |
| `src/boldtailor/results.py` | Retain private lineage in results and expose only private trusted design access. |
| `src/boldtailor/fit.py` | Propagate/validate lineage, use categorical failures, and remove redundant fingerprint code. |
| `src/boldtailor/prepared_fit.py` | Use trusted design access, propagate/validate lineage, and remove exception-message sanitization machinery. |
| `src/boldtailor/_versions.py` | Resolve installed versions with a safe `0+unknown` fallback. |
| `src/boldtailor/bids_provenance.py` | Project the dynamically resolved Boldtailor version. |
| `examples/stop_signal_demo.ipynb` | Require the dataset root and visibly select five sessions. |
| `tests/conftest.py` | Generate the complete five-session notebook fixture. |
| `tests/test_stop_signal_demo.py` | Verify semantic notebook behavior without presentation-only pinning. |
| `.gitignore` | Ignore generated Python, egg-info, and coverage artifacts. |

---

### Task 1: Shared Categorical Failure Logging

**Files:**
- Modify: `src/boldtailor/logging.py`
- Modify: `src/boldtailor/data.py`
- Modify: `src/boldtailor/prepared.py`
- Modify: `src/boldtailor/fit.py`
- Modify: `src/boldtailor/prepared_fit.py`
- Test: `tests/test_logging.py`
- Test: `tests/test_prepared_fit.py`

**Interfaces:**
- Produces: `emit_failure(event: str, *, stage: str, error: Exception, level: int = logging.ERROR, **context) -> Mapping[str, object]`.
- Produces failure fields `error_type` and `error_category`; removes the caller-derived `error` message field from new failure records.
- Categories are exactly `validation_error`, `numerical_error`, and `internal_error`.

- [ ] **Step 1: Write failing categorical failure tests**

Add tests that capture the `boldtailor` logger and force event-level and
prepared normalization, fit, and task-delta failures. Use exceptions whose
messages contain explicit sentinels:

```python
secret = "/private/input/sub-secret TOKEN_SENTINEL en_US.UTF-8 0xdeadbeef"

record = failure_records[-1]
assert record["error_type"] == "ValueError"
assert record["error_category"] == "validation_error"
assert "error" not in record
assert secret not in json.dumps(record)
```

Patch a numerical boundary to raise `np.linalg.LinAlgError(secret)` and assert
`numerical_error`. Patch a non-validation helper to raise `RuntimeError(secret)`
and assert `internal_error`. Confirm the same fields occur in retained event
history when the failure is appended there. Confirm `pytest.raises(...,
match="TOKEN_SENTINEL")` still sees the original exception message.

- [ ] **Step 2: Run the failure tests and observe RED**

Run:

```bash
uv run pytest tests/test_logging.py tests/test_prepared_fit.py -k "failure or private or sanitize" -q -W error
```

Expected: failures because current records contain `error`, lack
`error_type`/`error_category`, and prepared sanitization still scans the
environment.

- [ ] **Step 3: Commit the RED tests**

```bash
git add tests/test_logging.py tests/test_prepared_fit.py
git commit -m "test: specify categorical failure logging"
```

- [ ] **Step 4: Implement the shared failure emitter**

In `logging.py`, add short helpers:

```python
def emit_failure(
    event: str,
    *,
    stage: str,
    error: Exception,
    level: int = logging.ERROR,
    execution_id: str | None = None,
    data_id: str | None = None,
    analysis_id: str | None = None,
    run_index: int | None = None,
) -> Mapping[str, object]:
    return emit_event(
        event,
        stage=stage,
        level=level,
        error_type=type(error).__name__,
        error_category=_error_category(error),
        execution_id=execution_id,
        data_id=data_id,
        analysis_id=analysis_id,
        run_index=run_index,
    )


def _error_category(error: Exception) -> str:
    if isinstance(error, LinAlgError):
        return "numerical_error"
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return "validation_error"
    if isinstance(error, ArithmeticError):
        return "numerical_error"
    return "internal_error"
```

Import `LinAlgError` directly from `numpy.linalg`; it subclasses `ValueError`,
so its numerical check must precede the general validation check. Extend
`emit_event`/`_event_payload` with optional `error_type` and `error_category`
keyword fields. Add those two fields to `_history_entry`; retain legacy `error`
support only if unchanged callers outside the failure paths still require it.

Replace every normalization, fit, and delta failure emission in `data.py`,
`prepared.py`, `fit.py`, and `prepared_fit.py` with `emit_failure`. Remove
`_sanitize_error`, `_redact_environment_values`, path/traceback/repr/address
regexes, and the `os` import from `prepared_fit.py`.

- [ ] **Step 5: Run focused GREEN tests**

```bash
uv run pytest tests/test_logging.py tests/test_prepared_fit.py tests/test_data.py tests/test_prepared.py tests/test_fit.py -q -W error
uv run black --check src tests
uv run --no-project git diff --check
```

Expected: all selected tests pass; no raw exception messages are serialized.

- [ ] **Step 6: Commit production changes**

```bash
git add src/boldtailor/logging.py src/boldtailor/data.py src/boldtailor/prepared.py src/boldtailor/fit.py src/boldtailor/prepared_fit.py
git commit -m "fix: make failure logging categorical"
```

---

### Task 2: Linear Internal Design Access

**Files:**
- Modify: `src/boldtailor/prepared.py`
- Modify: `src/boldtailor/results.py`
- Modify: `src/boldtailor/prepared_fit.py`
- Modify: `src/boldtailor/fit.py`
- Test: `tests/test_prepared_fit.py`
- Test: `tests/test_fit.py`

**Interfaces:**
- Produces: private `PreparedDesignAnalysis._trusted_designs() -> tuple[pd.DataFrame, ...]`.
- Produces: private `AnalysisResult._trusted_designs() -> tuple[pd.DataFrame, ...]`.
- Produces: private `TaskDeltaR2Result._trusted_nuisance_designs() -> tuple[pd.DataFrame, ...]`.
- Public properties retain defensive deep copies.

- [ ] **Step 1: Write RED performance and ownership tests**

Patch `pd.DataFrame.copy` with a counting wrapper. For `n_runs` in
`(1, 2, 4, 8)`, construct a prepared analysis and call `fit_prepared()`.
Record the counts after construction and assert that the incremental fit count
is bounded by `4 * n_runs + 4`; also assert each count is no smaller than the
previous count. This is a complexity contract, not an exact implementation
count.

Add a delta-R² case with eight runs and the same linear bound. Retain direct
public mutation checks:

```python
returned = prepared.design_matrices
returned[0].iloc[0, 0] = -999.0
assert prepared.design_matrices[0].iloc[0, 0] != -999.0
```

Apply the same public check to `AnalysisResult.design_matrices` and
`TaskDeltaR2Result.nuisance_design_matrices`.

- [ ] **Step 2: Run the copy tests and observe RED**

```bash
uv run pytest tests/test_prepared_fit.py tests/test_fit.py -k "copy_complexity or defensive_design" -q -W error
```

Expected: complexity cases fail with approximately `n² + 3n` copies.

- [ ] **Step 3: Commit RED tests**

```bash
git add tests/test_prepared_fit.py tests/test_fit.py
git commit -m "test: bound internal design copying"
```

- [ ] **Step 4: Add private trusted accessors and migrate internals**

Implement one-line private methods returning stored tuples without copying.
Use them only inside Boldtailor fitting, comparison, validation, diagnostic,
and provenance construction. Never use them from examples or tests except the
copy-count instrumentation needed to establish internal behavior.

Replace patterns such as:

```python
design = prepared.design_matrices[run]
```

with one tuple retrieval per operation:

```python
designs = prepared._trusted_designs()
design = designs[run]
```

Do not alter public property implementations.

- [ ] **Step 5: Run GREEN and regression tests**

```bash
uv run pytest tests/test_prepared.py tests/test_prepared_fit.py tests/test_fit.py tests/test_multirun.py -q -W error
uv run black --check src tests
uv run --no-project git diff --check
```

- [ ] **Step 6: Commit production changes**

```bash
git add src/boldtailor/prepared.py src/boldtailor/results.py src/boldtailor/prepared_fit.py src/boldtailor/fit.py
git commit -m "perf: avoid repeated design copies"
```

---

### Task 3: Ephemeral Anonymous-Analysis Lineage

**Files:**
- Modify: `src/boldtailor/data.py`
- Modify: `src/boldtailor/prepared.py`
- Modify: `src/boldtailor/results.py`
- Modify: `src/boldtailor/fit.py`
- Modify: `src/boldtailor/prepared_fit.py`
- Test: `tests/test_fit.py`
- Test: `tests/test_prepared_fit.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Produces private `_lineage_id: object` fields on normalized analyses and results.
- Produces private `_model_identity: object` fields on fit results.
- `make_result(..., *, lineage_id: object) -> AnalysisResult` requires and retains the token.
- `make_result(..., *, model_identity: object) -> AnalysisResult` retains the normalized model identity used by the fit.
- `make_task_delta_r2_result(..., *, lineage_id: object, model_identity: object) -> TaskDeltaR2Result` retains both private identities.
- Delta validation requires lineage object identity and model-identity equality.

- [ ] **Step 1: Write RED anonymous-lineage tests**

For event-level and prepared APIs, omit `sources`, fit a full model, and assert
task delta R-squared succeeds and remains numerically equal to the explicit
source case. Assert both result provenance fingerprints remain `None` and keep
the provenance-quality warning.

Construct a second dimension-compatible normalized analysis, fit it, and pass
its result to the first analysis's delta operation. Assert a `ValueError` with
`"full result does not match"` before numerical fitting.

Serialize `provenance.to_dict()`, emitted records, and public object `repr()`;
assert no `lineage`, UUID-like lineage value, or private token appears. Do not
add a public lineage accessor.

- [ ] **Step 2: Run RED tests**

```bash
uv run pytest tests/test_fit.py tests/test_prepared_fit.py tests/test_provenance.py -k "anonymous and delta or lineage" -q -W error
```

Expected: anonymous delta cases raise the current fingerprintability error.

- [ ] **Step 3: Commit RED tests**

```bash
git add tests/test_fit.py tests/test_prepared_fit.py tests/test_provenance.py
git commit -m "test: specify anonymous analysis lineage"
```

- [ ] **Step 4: Implement private lineage propagation**

Create each token with `object()` during successful normalization and retain it
in the frozen owning dataclass with `field(repr=False, compare=False)`. Retain
private result model identity with the same dataclass flags. Pass both values to
result factories and comparison factories. When `data_id` or deterministic
model identity is missing, keep the public `analysis_fingerprint` as `None`; do
not substitute lineage.

Change parent validation to:

```python
if full_result._lineage_id is not analysis._lineage_id:
    raise ValueError("full result does not match data and model")
if expected_parent_id is not None and (
    full_result.provenance.analysis_fingerprint != expected_parent_id
):
    raise ValueError("full result does not match data and model")
```

For event-level fitting, retain the normalized `ModelSpec` as the private model
identity and compare it with the supplied delta model. For prepared fitting,
retain the path-safe frozen mapping returned by `_model_identity(prepared,
fit_spec)` and compare it with the mapping derived for the delta call. These
private values are never serialized or exposed. Reject a full result from the
same normalized data but a different contrast, HRF, drift, confound,
noise-model, or model-metadata specification.

- [ ] **Step 5: Run GREEN and full result tests**

```bash
uv run pytest tests/test_fit.py tests/test_prepared_fit.py tests/test_provenance.py tests/test_results.py -q -W error
uv run black --check src tests
uv run --no-project git diff --check
```

- [ ] **Step 6: Commit production changes**

```bash
git add src/boldtailor/data.py src/boldtailor/prepared.py src/boldtailor/results.py src/boldtailor/fit.py src/boldtailor/prepared_fit.py
git commit -m "feat: track ephemeral analysis lineage"
```

---

### Task 4: Safe Central Version Resolution

**Files:**
- Create: `src/boldtailor/_versions.py`
- Modify: `src/boldtailor/bids_provenance.py`
- Modify: `src/boldtailor/prepared.py`
- Modify: `src/boldtailor/prepared_fit.py`
- Modify: `src/boldtailor/fit.py`
- Test: `tests/test_versions.py`
- Test: `tests/test_bids_provenance.py`
- Test: `tests/test_prepared.py`
- Test: `tests/test_prepared_fit.py`

**Interfaces:**
- Produces: `distribution_version(name: str) -> str` in private module `_versions.py`.
- Returns `0+unknown` only for `importlib.metadata.PackageNotFoundError`.
- BIDS projection resolves Boldtailor version through this helper.

- [ ] **Step 1: Write RED version tests**

Create `tests/test_versions.py` with monkeypatched metadata lookup:

```python
def test_distribution_version_returns_installed_value(monkeypatch):
    monkeypatch.setattr(versions.metadata, "version", lambda name: "9.8.7")
    assert versions.distribution_version("boldtailor") == "9.8.7"


def test_distribution_version_falls_back_only_when_missing(monkeypatch):
    def missing(name):
        raise versions.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(versions.metadata, "version", missing)
    assert versions.distribution_version("boldtailor") == "0+unknown"
```

Add a test that an unrelated `RuntimeError` propagates. Patch the shared helper
before reloading or invoking BIDS projection and assert both `GeneratedBy` and
BEP028 software records use `9.8.7`. Add prepared provenance/backend tests that
assert the shared helper supplies Boldtailor, NumPy, pandas, and Nilearn values.

- [ ] **Step 2: Run RED tests**

```bash
uv run pytest tests/test_versions.py tests/test_bids_provenance.py tests/test_prepared.py tests/test_prepared_fit.py -k "version" -q -W error
```

Expected: `_versions` does not exist and BIDS output remains hardcoded `0.1.0`.

- [ ] **Step 3: Commit RED tests**

```bash
git add tests/test_versions.py tests/test_bids_provenance.py tests/test_prepared.py tests/test_prepared_fit.py
git commit -m "test: specify shared version resolution"
```

- [ ] **Step 4: Implement the helper and migrate consumers**

```python
from importlib import metadata


def distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "0+unknown"
```

Use functions or safely initialized mappings so source-checkout imports work.
Remove `BOLDTAILOR_VERSION = "0.1.0"` and ad hoc imports of
`importlib.metadata.version`. Remove the redundant fingerprint expression:

```python
fingerprint = activity if not callable_warnings else None
```

Copy the activity when mutation safety requires it; do not reinsert
`drift_order`.

- [ ] **Step 5: Run GREEN tests**

```bash
uv run pytest tests/test_versions.py tests/test_bids_provenance.py tests/test_prepared.py tests/test_prepared_fit.py tests/test_fit.py -q -W error
uv run black --check src tests
uv run --no-project git diff --check
```

- [ ] **Step 6: Commit production changes**

```bash
git add src/boldtailor/_versions.py src/boldtailor/bids_provenance.py src/boldtailor/prepared.py src/boldtailor/prepared_fit.py src/boldtailor/fit.py
git commit -m "fix: centralize package version metadata"
```

---

### Task 5: Portable Five-Session API Notebook

**Files:**
- Modify: `examples/stop_signal_demo.ipynb`
- Modify: `tests/conftest.py`
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `README.md`
- Modify: `.gitignore`
- Test: `tests/test_repository_contracts.py`

**Interfaces:**
- Notebook requires environment variable `BOLDTAILOR_BIDS_ROOT`.
- Notebook defines ordinary `SESSIONS = ("ses-02", "ses-04", "ses-06", "ses-08", "ses-10")`.
- No `BOLDTAILOR_SESSIONS` or `_configured_sessions` interface remains.

- [ ] **Step 1: Write RED configuration and semantic notebook tests**

Expand `tests/conftest.py::SESSIONS` to all five sessions. Make `_events`
exercise run-specific designs across the five sessions while ensuring each of
the notebook's three semantic contrasts remains estimable.

Replace environment-session tests with these contracts:

```python
def test_notebook_requires_bids_root():
    # Execute only the configuration cell with BOLDTAILOR_BIDS_ROOT absent.
    # Assert an actionable ValueError naming BOLDTAILOR_BIDS_ROOT.


def test_notebook_configuration_uses_visible_five_sessions(...):
    assert configuration["SESSIONS"] == (
        "ses-02", "ses-04", "ses-06", "ses-08", "ses-10"
    )
    assert "BOLDTAILOR_SESSIONS" not in configuration_source
    assert "_configured_sessions" not in configuration_source
```

Run both notebook working-directory cases against the unmodified five-session
fixture. Replace cosmetic assertions as follows:

- remove exact `display_count == 11`; keep compactness, no raw-signal memory,
  and required diagnostic fields;
- remove literal `magma`, `viridis`, and plot-title assertions;
- replace exact slice values with proof that there are six finite cut
  coordinates and all five maps receive the same coordinates;
- retain exact five map identities, nonnegative delta R², threshold disclosure,
  colorbar behavior, and scientific map/result matching.

Update README expectations to describe a five-session temporary API walkthrough
that requires `BOLDTAILOR_BIDS_ROOT`.

- [ ] **Step 2: Run RED notebook tests**

```bash
uv run pytest tests/test_stop_signal_demo.py -q -W error -p no:cacheprovider
```

Expected: failures because the fixture contains two sessions, the notebook
accepts `BOLDTAILOR_SESSIONS`, and it embeds a personal root fallback.

- [ ] **Step 3: Commit RED tests and fixture requirement**

```bash
git add tests/conftest.py tests/test_stop_signal_demo.py tests/test_repository_contracts.py
git commit -m "test: specify portable five-session notebook"
```

- [ ] **Step 4: Update the notebook, README, and ignore rules**

The configuration cell becomes:

```python
import os
from pathlib import Path

if "BOLDTAILOR_BIDS_ROOT" not in os.environ:
    raise ValueError(
        "Set BOLDTAILOR_BIDS_ROOT to the local BIDS dataset before running this notebook"
    )
BIDS_ROOT = Path(os.environ["BOLDTAILOR_BIDS_ROOT"])
SESSIONS = ("ses-02", "ses-04", "ses-06", "ses-08", "ses-10")
```

Do not change notebook cell IDs. Preserve existing stored outputs if any; tests
must remain output-tolerant. In the opening Markdown, label the notebook a
temporary executable API walkthrough for development.

Add repository-wide ignore entries:

```gitignore
__pycache__/
*.egg-info/
.coverage
```

Do not delete the user's existing generated files.

Add a repository-contract test that parses `.gitignore` and requires these
three exact entries. This is a deliberate repository hygiene policy, not a test
of the current worktree's generated contents.

- [ ] **Step 5: Run notebook GREEN tests and formatting**

```bash
uv run pytest tests/test_stop_signal_demo.py -q -W error -p no:cacheprovider
uv run black --check tests examples/stop_signal_demo.py
uv run --no-project git diff --check
```

- [ ] **Step 6: Commit documentation production changes**

```bash
git add examples/stop_signal_demo.ipynb README.md .gitignore
git commit -m "docs: make stop signal walkthrough portable"
```

---

### Task 6: Final Verification and Real-Data Handoff

**Files:**
- Verify: `src/boldtailor/`
- Verify: `tests/`
- Verify: `examples/stop_signal_demo.ipynb`
- Verify: `README.md`

**Interfaces:**
- Consumes the completed five remediation tasks.
- Produces a verified branch ready for the conventional FitLins adapter phase.

- [ ] **Step 1: Run focused warning-strict suites**

```bash
uv run pytest tests/test_logging.py tests/test_fit.py tests/test_prepared_fit.py tests/test_versions.py tests/test_bids_provenance.py tests/test_stop_signal_demo.py -q -W error -p no:cacheprovider
```

- [ ] **Step 2: Run the complete repository gates**

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -W error -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 uv run black --check src tests examples/stop_signal_demo.py
uv lock --check
uv run --no-project git diff --check main..HEAD
test ! -s src/boldtailor/__init__.py
uv run --no-project git diff --exit-code main..HEAD -- pyproject.toml uv.lock src/boldtailor/__init__.py
```

Expected: all tests pass, Black is clean, the lock is current, and no dependency
or initializer changes exist.

- [ ] **Step 3: Execute the five-session real notebook in memory**

Create a fresh `/private/tmp/boldtailor-pre-fitlins.*` root with separate
Jupyter, IPython, Matplotlib, and publication directories. Before and after the
run, assert that
`/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor` does not exist.
Set only:

```bash
BOLDTAILOR_BIDS_ROOT=/Users/poldrack/data_unsynced/rdoc_fmri
BOLDTAILOR_TEMP_ROOT=<fresh-publication-dir>
MPLBACKEND=Agg
JUPYTER_CONFIG_DIR=<fresh-jupyter-dir>
IPYTHONDIR=<fresh-ipython-dir>
MPLCONFIGDIR=<fresh-matplotlib-dir>
```

Do not set `BOLDTAILOR_SESSIONS`. Execute with `nbformat` and
`NotebookClient(timeout=7200, kernel_name="python3")` from the `examples/`
directory. Keep outputs in memory and assert:

- no error output;
- the rendered configuration lists all five session labels;
- `fit_prepared`, `task_delta_r2_prepared`, AR(1), and nested OLS are disclosed;
- all expected contrast, aggregate R², and delta R² maps are produced;
- exactly one temporary publication exists with 11 NIfTI images; and
- no executed notebook is written to the repository or dataset.

Poll the command at intervals shorter than 60 seconds.

- [ ] **Step 4: Record the handoff without a production commit**

Write the final verification evidence in this plan's ignored SDD workspace:
exact commit range, focused/full test counts, Black/lock/diff results, copy-count
measurements, anonymous-lineage checks, failure-record examples, five-session
real-data result, and final worktree status. Do not commit generated artifacts
or verification reports.
