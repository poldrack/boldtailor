# Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every finding in the 2026-10-01 full-project review and the 2026-10-02 test-suite review, in priority order, so that the package is numerically correct for degenerate inputs, its examples and documentation describe the task model the code actually fits, its provenance can identify inputs and software, and its default test run covers only the package.

**Architecture:** Five phases, each independently shippable. Phase 0 restructures the test suite (package-only default, pruning, consolidation). Phase 1 fixes correctness bugs. Phase 2 completes the provenance record. Phase 3 adds the missing scientific tests and diagnostics. Phase 4 pays down the architectural backlog. Phase 5 is hygiene. Within each phase, tasks are ordered so that each leaves the suite green.

**Tech Stack:** Python 3.12, uv, pytest, numpy, pandas, scipy, nilearn 0.14.x, nibabel (examples), Black.

**Spec:** `docs/review-2026-10-01-full-project.md` (findings S1–S6, P1–P6, N1–N6, L1–L9, V1–V7, E1–E6, T1–T6) and `docs/test-suite-review-2026-10-02.md` (per-test verdicts in its Appendix). Finding codes below refer to those documents.

## Global Constraints

- Use `uv run` for every Python and pytest command; never call `python` or `pytest` directly.
- Every `__init__.py` must stay empty (0 bytes).
- RED-GREEN-Refactor: write the failing test, run it and see it fail, commit the test, then implement. Never weaken a test to pass; a test may change only when the requirement changed (named in the task) or the test was wrong (named in the review).
- Default `uv run pytest` runs `tests/` only. Notebook-executing tests carry `@pytest.mark.notebook` and run only with `--run-notebooks`. `examples/NSD` tests run only when that path is given explicitly. (User decision, 2026-10-02.)
- The `trial_type` modulator is **uncentered** (`Modulator("trial_type", center=False)`); this was intentional. Fix prose and metadata to match the code; never change the code to match the prose.
- Every HRF kernel the package builds or selects is scaled to unit peak amplitude (user requirement, 2026-10-02; Task 1.5). Nilearn derivative/FIR basis strings are the only exception and are documented as such.
- Prefer Nilearn components over custom numerics; any custom numeric path needs a Nilearn or closed-form oracle test.
- Prefer short functions (under 40 lines) and small modules.
- Black-formatted (`uv run black --check src tests examples/NSD examples/stop_signal_demo.py` must pass).
- Commit after every GREEN step, tests before implementation. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Do not commit the currently modified notebooks or the untracked `examples/NSD/.env` / `sobol_hrfs_preview.png` (Task 1.3 handles them).

## Review Focus

Inputs the spec implies but no existing test exercises, most likely to bite first. Each is pinned to a task below.

1. A feature that is exactly constant in one run but varies in another must have NaN contrasts, not a spuriously large t (Task 1.1).
2. A saved NSD result fitted under a different task model (e.g. centered `trial_type`) must be rejected by reuse validation, not silently "Reused" (Task 1.3).
3. Two source files with identical path, size, and mtime but different content must produce different provenance identities once a sha256 is supplied (Task 2.2).
4. A BOLD run whose sidecar declares `StartTime` must have its frame times offset so onsets align (Task 1.2).
5. A ridge-fraction or HRF selection that lands on the edge of its grid or parameter box must be visible to the user (Task 3.2).

---

# Phase 0: Test suite restructure

### Task 0.1: Package-only default run, opt-in notebooks, split CI

**Files:**
- Modify: `pyproject.toml:37-39`
- Modify: `tests/conftest.py` (top of file)
- Create: `examples/NSD/conftest.py`
- Modify: `.github/workflows/tests.yml`
- Modify: `tests/test_stop_signal_demo.py` (add marker to every function that executes the notebook; grep `NotebookClient` and the helper wrapping it, mark each caller)
- Modify: `examples/NSD/test_*.py` (mark the 8 kernel-launching functions: `test_fractional_notebook.py:24`, `test_multisession_workflow.py:33,79,129`, `test_nsd_workflow.py:286`, `test_ridge_outputs.py:126`, `test_session_hrf.py:319`, `test_workflow_surfaces.py:438`)
- Modify: `docs/development.md` (testing section)

**Interfaces:**
- Produces: pytest option `--run-notebooks`; marker `notebook`.

- [ ] **Step 1: Record the current collection count**

Run: `uv run pytest --collect-only -q | tail -1`
Expected: `1021 tests collected` (record the number).

- [ ] **Step 2: Add marker and skip logic to `tests/conftest.py`**

Insert after the imports:

```python
def pytest_addoption(parser):
    parser.addoption(
        "--run-notebooks",
        action="store_true",
        default=False,
        help="execute notebook kernels (slow); off by default",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "notebook: executes a notebook kernel; needs --run-notebooks"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-notebooks"):
        return
    skip = pytest.mark.skip(reason="notebook execution needs --run-notebooks")
    for item in items:
        if "notebook" in item.keywords:
            item.add_marker(skip)
```

Create `examples/NSD/conftest.py` containing `import pytest` and the same three functions verbatim.

- [ ] **Step 3: Restrict `testpaths`**

Replace `pyproject.toml` lines 37-39 with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

(`pythonpath` is removed in Task 0.3.)

- [ ] **Step 4: Mark the notebook-executing tests** with `@pytest.mark.notebook` (see Files).

- [ ] **Step 5: Verify collection**

Run: `uv run pytest --collect-only -q | tail -1` → Expected: `758 tests collected`.
Run: `uv run pytest -q tests/test_stop_signal_demo.py | tail -2` → notebook tests skipped, rest pass.
Run: `uv run pytest -q --run-notebooks tests/test_stop_signal_demo.py | tail -1` → pass, no skips.
Run: `uv run pytest -q examples/NSD | tail -1` → pass, notebook tests skipped.

- [ ] **Step 6: Split CI**

Replace `.github/workflows/tests.yml` with:

```yaml
name: tests
on:
  push:
  pull_request:
  workflow_dispatch:
  schedule:
    - cron: '0 6 * * 1'
permissions:
  contents: read
jobs:
  package:
    runs-on: ubuntu-latest
    env:
      MPLBACKEND: Agg
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: '3.12'
      - run: uv sync --locked --group dev
      - run: uv run pytest -q -W error
      - run: uv run black --check src tests examples/NSD examples/stop_signal_demo.py
      - run: git diff --check
      - run: uv build --wheel
      - run: uv run --isolated --no-project --with "$(ls dist/boldtailor-*.whl)" python tests/check_installed_package.py
  examples:
    if: github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'
    runs-on: ubuntu-latest
    env:
      MPLBACKEND: Agg
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: '3.12'
      - run: uv sync --locked --group dev
      - run: uv run pytest -q -W error --run-notebooks examples/NSD examples/validation tests/test_stop_signal_demo.py
```

- [ ] **Step 7: Document**

In `docs/development.md`, replace the sentence describing `uv run pytest` collecting `tests` and `examples/NSD` with: "`uv run pytest` runs the package suite in `tests/`. Example and notebook tests are opt-in: `uv run pytest examples/NSD` and `uv run pytest --run-notebooks tests/test_stop_signal_demo.py examples/NSD`."

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml tests/conftest.py examples/NSD/conftest.py .github/workflows/tests.yml docs/development.md tests/test_stop_signal_demo.py examples/NSD/test_*.py
git commit -m "test: package-only default run; notebook tests opt-in; split CI jobs"
```

### Task 0.2: Delete the CUT tests and the wrong/vacuous tests

**Files:**
- Modify: every file named by the listing in Step 1.

**Interfaces:** none.

- [ ] **Step 1: List the CUT verdicts from the review appendix**

Save this as `/tmp/list_cuts.py` (scratch, not committed) and run `uv run python /tmp/list_cuts.py`:

```python
import re
cur = None
for line in open("docs/test-suite-review-2026-10-02.md"):
    m = re.match(r"^##+ .*?((?:tests|examples/NSD)/test_\w+\.py)", line)
    if m:
        cur = m.group(1)
        continue
    m = re.match(r"^\|\s*(\d+)\s*\|\s*(test_\w+)\s*\|\s*\w\s*\|\s*CUT\s*\|", line)
    if m and cur:
        print(f"{cur}:{m.group(1)} {m.group(2)}")
```

Expected: about 56 lines of `file:line test_name`.

- [ ] **Step 2: Delete each listed test function** (whole `def`, decorators, and helpers used only by it). Where the appendix reason says a sibling absorbs one assertion (e.g. `test_fit.py:395` folds a design-copy check into `:365`), move that assertion first.

Also delete these review §4 items regardless of verdict: `tests/test_hrf_glm.py:709`; `tests/test_fit.py:690` (clipping-contract pin; Task 4.1 defines the contract); the nonexistent-symbol monkeypatch in `tests/test_fractional_ridge.py:211` (keep the ownership asserts after it); `tests/test_prepared_fit.py:764`; `tests/test_publication.py:482,545`; `tests/test_provenance.py:311` and `tests/test_logging.py:137-140` (replaced in Step 3); the literal-row pin in `tests/test_sobol_hrf_library.py:21`; `tests/test_repository_contracts.py` (whole file); the nilearn warning-text pins in `tests/test_multirun.py:218,278` (keep the rank-deficiency behaviour asserts).

- [ ] **Step 3: Replace machine-specific leak asserts with a sentinel**

Where `tests/test_provenance.py:311` asserted the absence of `/Users/poldrack/...`, `Path.home()`, and the hostname, write:

```python
def test_serialized_record_excludes_injected_local_path(complete_sources):
    sentinel = "/sentinel-home/boldtailor-secret"
    record = make_record(complete_sources(1), metadata={"note": "ok"})
    assert sentinel not in record.canonical_json()
```

(`make_record` is whatever helper that module already uses to build a `ProvenanceRecord`; the sentinel assertion is the point.) Apply the same to `tests/test_logging.py:137-140`.

- [ ] **Step 4: Run** `uv run pytest -q` → pass; `--collect-only` count drops by roughly 60.

- [ ] **Step 5: Commit**

```bash
git add tests
git commit -m "test: remove vacuous, machine-specific, and bug-pinning tests per 2026-10-02 review"
```

### Task 0.3: Consolidate fixtures and oracles; move simulation tests out of the default run

**Files:**
- Modify: `tests/conftest.py`, `tests/oracles.py`
- Move: `tests/test_fractional_ridge_ablation.py`, `tests/test_fractional_ablation_simulation.py`, `tests/test_ridge_objective_simulation.py` → `examples/validation/`
- Create: `examples/validation/conftest.py`
- Modify: `tests/test_trial_encoding.py`, `tests/test_single_trial.py` (receive the two package oracles from `test_ridge_objective_simulation.py:68,94`)
- Modify: `tests/test_fit.py`, `tests/test_multirun.py`, `tests/test_prepared.py`, `tests/test_prepared_fit.py`, `tests/test_logging.py`, `tests/test_data.py`, `tests/test_hrf_glm.py`, `tests/test_hrf_selection.py`, `tests/test_hrf_cv.py`
- Modify: `pyproject.toml` (drop `pythonpath`)

**Interfaces:**
- Produces in `tests/conftest.py`: fixture `complete_sources` → callable `(n_runs: int) -> list[RunSources]`; fixture `two_candidate_library` → `HrfLibrary` from rows `[[3,10,0.5,0.5,2,0,36],[6,16,1.5,2.5,8,2,36]]`.
- Produces in `tests/oracles.py`: `nilearn_original_space_r2(signals, design, noise_model) -> np.ndarray`; `stacked_training_ols_oracle(...)`, `loro_oracle(...)` (moved verbatim from `tests/test_hrf_cv.py:79,105`); `task_model_oracle(...)` (moved from `tests/test_hrf_selection.py:58`).

- [ ] **Step 1: Add shared fixtures to `tests/conftest.py`**

```python
from boldtailor.provenance import RunSources, SourceRef


@pytest.fixture
def complete_sources():
    def build(n_runs):
        def ref(role, run):
            return SourceRef(
                role=role,
                uri=f"sub-01/func/run-{run:02d}_{role}.tsv",
                media_type="text/tab-separated-values",
                byte_size=1000 + run,
                modified_at="2026-01-01T00:00:00Z",
            )

        return [
            RunSources(
                signal=ref("signal", r),
                events=ref("events", r),
                confounds=ref("confounds", r),
            )
            for r in range(n_runs)
        ]

    return build


@pytest.fixture
def two_candidate_library():
    return HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )
```

Copy the exact field values from the existing `_complete_sources` in `tests/test_fit.py:48` if they differ; one definition is the goal.

- [ ] **Step 2: Replace the five `_complete_sources` copies and the three library literals** with the fixtures; run the affected files after each edit. Expected: pass.

- [ ] **Step 3: Move oracles into `tests/oracles.py`**

Replace `_nilearn_original_space_r2`, `_nilearn_pooled_ols_r2`, `_ols_r2_oracle`, `_original_space_ar1_diagnostics` (in test_fit, test_multirun, test_prepared_fit, test_hrf_glm) with one function:

```python
def nilearn_original_space_r2(signals, design, noise_model):
    """Pooled R² from nilearn's fit, scored on unwhitened predictions."""
    from nilearn.glm.first_level import run_glm

    labels, results = run_glm(signals, design, noise_model=noise_model)
    prediction = np.empty_like(signals, dtype=float)
    for label, result in results.items():
        prediction[:, labels == label] = design @ result.theta
    sse = ((signals - prediction) ** 2).sum(axis=0)
    sst = ((signals - signals.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - sse / sst
```

Move `stacked_oracle`/`loro_oracle` and `oracle` verbatim (renamed per Interfaces); replace `from tests.test_hrf_selection import oracle_cv` and `from tests.test_hrf_cv import ...` with `from tests.oracles import ...`.

- [ ] **Step 4: Relocate the simulation tests**

First cut the package oracles at `test_ridge_objective_simulation.py:68` (closed-form encoding check) and `:94` into `tests/test_trial_encoding.py` and `tests/test_single_trial.py`, unchanged except imports. Then:

```bash
git mv tests/test_fractional_ridge_ablation.py examples/validation/
git mv tests/test_fractional_ablation_simulation.py examples/validation/
git mv tests/test_ridge_objective_simulation.py examples/validation/
```

Create `examples/validation/conftest.py` identical to `examples/NSD/conftest.py`. Fix the moved files' imports to relative imports of the generator modules beside them.

- [ ] **Step 5: Drop `pythonpath`** from `pyproject.toml`. Run `uv run pytest -q` and `uv run pytest -q examples/validation`. Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add -A tests examples/validation pyproject.toml
git commit -m "test: shared fixtures and oracles; simulation tests opt-in under examples/validation"
```

### Task 0.4: One parametrized table per validator

**Files:**
- Modify: `tests/test_model.py`, `tests/test_task_model.py`, `tests/test_provenance.py`, `tests/test_publication.py`, `tests/test_bids_provenance.py`, `tests/test_data.py`, `tests/test_prepared.py`, `tests/test_fit.py`, `tests/test_multirun.py`, `tests/test_prepared_fit.py`, `tests/test_hrf_glm.py`

**Interfaces:** none.

- [ ] **Step 1: Convert `tests/test_model.py` to one table** (pattern for the rest)

```python
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"contrasts": {}}, "at least one contrast"),
        ({"contrasts": {"": {"a": 1.0}}}, "nonempty strings"),
        ({"contrasts": {"c": {"a": 0.0}}}, "nonzero weight"),
        ({"contrasts": {"c": {"a": True}}}, "must be numeric"),
        ({"contrasts": {"c": {"a": float("inf")}}}, "must be finite"),
        ({"contrasts": {"c": "a"}, "noise_model": "ols2"}, "noise_model must be"),
        ({"contrasts": {"c": "a"}, "high_pass": 0.0}, "high_pass must be positive"),
        ({"contrasts": {"c": "a"}, "drift_order": -1}, "drift_order must be"),
        ({"contrasts": {"c": "a"}, "oversampling": 0}, "oversampling must be"),
        ({"contrasts": {"c": "a"}, "min_onset": float("nan")}, "min_onset must be finite"),
        ({"contrasts": {"c": "a"}, "confounds": ["m", "m"]}, "confound names must be unique"),
    ],
)
def test_model_spec_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ModelSpec(**kwargs)
```

Keep the KEEP tests (`contrast_names` order, mapping immutability, defaults). Use only the distinguishing fragment of each message.

- [ ] **Step 2: Apply the same shape to** `Modulator`/`TaskModel` (`test_task_model.py`), `SourceRef`/`RunSources` (`test_provenance.py`), publication preflight (`test_publication.py`), `project_bids_provenance` options (`test_bids_provenance.py`), `from_arrays` shape/timing errors (`test_data.py`), `PreparedDesignAnalysis.from_arrays` (`test_prepared.py`), and the "rejects before GLM" cluster (`test_fit.py:625`, `test_multirun.py:278,341,363`, `test_prepared_fit.py:255,289,316`) as one test parametrized over `(entry_point, bad_contrast, message)` with one `fail_glm` monkeypatch asserting `run_glm` is never called.

- [ ] **Step 3: Run** `uv run pytest -q` → pass; collected count drops by about 120.

- [ ] **Step 4: Commit** `git add tests && git commit -m "test: one parametrized table per validator"`

### Task 0.5: One lifecycle test module

**Files:**
- Create: `tests/test_lifecycle.py`
- Modify (delete pasted copies): `tests/test_fit.py:856,900`, `tests/test_prepared_fit.py:1372,1425`, `tests/test_hrf_glm.py:465,509`, `tests/test_data.py`, `tests/test_prepared.py` (byte-identical `test_normalization_builds_one_record` / `..._catches_final_failures...`), `tests/test_single_trial.py:405`, `tests/test_selected_hrf_fit.py:176`, `tests/test_logging.py:345`

**Interfaces:**
- Consumes: entry points `fit`, `task_delta_r2`, `fit_single_trials`, `fit_selected_hrfs`; `caplog` records whose message is JSON with `event` and `execution_id`.

- [ ] **Step 1: Write the parametrized module**

```python
import json
import logging

import pytest

ENTRY_POINTS = ["fit", "task_delta_r2", "single_trial", "selected_hrf_single_trial"]


def _events(caplog, name):
    events = [json.loads(r.getMessage()) for r in caplog.records]
    return [e for e in events if e["event"].startswith(name)]


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_success_logs_started_then_completed_with_one_execution_id(
    name, run_entry_point, caplog
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    run_entry_point(name)
    events = _events(caplog, name)
    assert [e["event"] for e in events] == [f"{name}_started", f"{name}_completed"]
    assert len({e["execution_id"] for e in events}) == 1


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_late_failure_logs_failed_and_never_completed(
    name, run_entry_point, break_late, caplog
):
    caplog.set_level(logging.INFO, logger="boldtailor")
    with break_late(), pytest.raises(RuntimeError):
        run_entry_point(name)
    kinds = [e["event"] for e in _events(caplog, name)]
    assert kinds == [f"{name}_started", f"{name}_failed"]
```

In the same module add fixtures `run_entry_point` (maps each name to the concrete call copied from the existing pasted tests, using `ridge_problem`/`selected_fixture`) and `break_late` (a context manager that monkeypatches `boldtailor._fit_lifecycle.extend_provenance` to raise `RuntimeError("late")`).

- [ ] **Step 2: Run** `uv run pytest -q tests/test_lifecycle.py` → pass.
- [ ] **Step 3: Delete the pasted copies** listed in Files; `uv run pytest -q` → pass.
- [ ] **Step 4: Commit** `git add -A tests && git commit -m "test: single parametrized lifecycle module"`


---

# Phase 1: Correctness

### Task 1.1: Constant features get NaN contrast statistics (S1)

**Files:**
- Modify: `tests/test_fit.py:279-320` (the existing zero-SST test is incomplete; extend it)
- Modify: `tests/test_fit.py` (new multi-run test), `tests/test_prepared_fit.py` (new test)
- Modify: `src/boldtailor/results.py` (add `mask_contrast`)
- Modify: `src/boldtailor/_conventional.py:40-60,174-191`
- Modify: `docs/user-guide.md` (after line 84), `docs/api.md` (AnalysisResult section)

**Interfaces:**
- Produces: `results.mask_contrast(result: _ContrastResult, undefined: np.ndarray) -> _ContrastResult`.
- `ConventionalFit.contrasts[name]` fields are NaN wherever any run's total sum of squares is zero.

- [ ] **Step 1: Extend the existing test** at `tests/test_fit.py:279` by appending after the R² loop:

```python
    for accessor in ("effect", "variance", "stat", "z_score", "one_sided_p_value"):
        values = getattr(result, accessor)(name)
        assert np.isnan(values[1:]).all(), accessor
```

Add the multi-run case below it:

```python
def test_feature_constant_in_one_run_has_undefined_contrasts(single_run_problem):
    signals, events, _, _ = single_run_problem
    first = np.column_stack([signals[:, 0], signals[:, 0] * 0.5 + 1.0])
    second = first.copy()
    second[:, 1] = 7.0
    model = ModelSpec(
        contrasts={"face_gt_house": {"face": 1.0, "house": -1.0}},
        drift_model=None,
        noise_model="ols",
    )
    result = fit(from_arrays([first, second], [events, events], tr=2.0), model)
    assert np.isfinite(result.stat("face_gt_house")[0])
    assert np.isnan(result.stat("face_gt_house")[1])
    assert np.isnan(result.effect("face_gt_house")[1])
```

Add to `tests/test_prepared_fit.py`, next to the nilearn-oracle test at line 120 and using its prepared fixture:

```python
def test_fit_prepared_masks_constant_features(prepared_problem):
    prepared, contrasts, noise_model, metadata = prepared_problem
    signals = [s.copy() for s in prepared.signals]
    for s in signals:
        s[:, -1] = 3.0
    constant = replace_signals(prepared, signals)  # the module's existing rebuild helper
    result = fit_prepared(
        constant, contrasts=contrasts, noise_model=noise_model, model_metadata=metadata
    )
    name = next(iter(contrasts))
    assert np.isnan(result.z_score(name)[-1])
    assert np.isfinite(result.z_score(name)[0])
```

(Use the fixture and rebuild helper that module already has for the line-120 test; if none exists, build the `PreparedDesignAnalysis` inline the same way that test does.)

- [ ] **Step 2: Run** `uv run pytest -q tests/test_fit.py -k "zero_sst or constant_in_one_run" tests/test_prepared_fit.py -k constant`
Expected: FAIL (finite stat/z for the constant feature; the probe in the review showed z ≈ 6.9).

- [ ] **Step 3: Commit the tests** `git add tests && git commit -m "test: constant features must have undefined contrast statistics"`

- [ ] **Step 4: Implement**

In `src/boldtailor/results.py` add after `contrast_result`:

```python
def mask_contrast(result: _ContrastResult, undefined: np.ndarray) -> _ContrastResult:
    """Return a copy with every statistic set to NaN where ``undefined`` is True."""

    def masked(values: np.ndarray) -> np.ndarray:
        out = np.array(values, dtype=float, copy=True)
        out[undefined] = np.nan
        return readonly_array(out)

    return _ContrastResult(
        **{f.name: masked(getattr(result, f.name)) for f in fields(_ContrastResult)}
    )
```

(add `from dataclasses import dataclass, fields`). In `src/boldtailor/_conventional.py` change `fit_designs`:

```python
    combined = _combine_contrasts(run_fits, tuple(contrasts))
    undefined = np.any([run.total_sum <= 0 for run in run_fits], axis=0)
    combined = {name: mask_contrast(value, undefined) for name, value in combined.items()}
    residual_sum = np.sum([run.residual_sum for run in run_fits], axis=0)
```

and import `mask_contrast` from `boldtailor.results`. Replace the comment-free warning filter in `_nilearn_t_contrast` with a comment: `# Constant features divide 0/0 here; fit_designs masks them to NaN afterwards.`

- [ ] **Step 5: Run** the three tests → PASS; then `uv run pytest -q` → pass.

- [ ] **Step 6: Document.** In `docs/user-guide.md` after the pooled-R² sentence (line 84) add: "A feature whose signal is exactly constant in any run has NaN for every contrast statistic and for R²; constant features cannot support inference." Add the same sentence to the `AnalysisResult` entry in `docs/api.md`.

- [ ] **Step 7: Commit** `git add src docs tests && git commit -m "fix: undefined contrast statistics for constant features"`

### Task 1.2: Stop-signal demo: contrast weights, session count, StartTime (S5)

**Files:**
- Modify: `examples/stop_signal_demo.ipynb` cells `configuration` (index 1), `ingestion` (7), `design-fit` (9), and the markdown title cell (0)
- Modify: `examples/stop_signal_demo.py:115` (`load_run`), `:587` (`_frame_times`)
- Modify: `tests/conftest.py:59-94` (`stop_signal_bids_dataset` writes a BOLD sidecar)
- Modify: `tests/test_stop_signal_demo.py:52,57,887,656,731,1492` and the notebook-executing tests (collapse to one smoke run per Task 0.1 marker)

**Interfaces:**
- Produces: `stop_signal_demo._frame_times(n_scans: int, tr: float, start_time: float = 0.0) -> np.ndarray`; `load_run(...)` reads `<bold stem>.json` beside the NIfTI and passes its `StartTime` (default 0.0).
- Notebook contrast `stop_vs_go = "0.5 * stop_success + 0.5 * stop_failure - go_success"`.
- Notebook accepts any number of sessions ≥ 2; `_configured_sessions` message becomes "BOLDTAILOR_SESSIONS must select at least two non-empty sessions".

- [ ] **Step 1: Write the failing tests**

In `tests/conftest.py` `stop_signal_bids_dataset`, after saving the BOLD image add:

```python
        (derivative_func / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json").write_text(
            '{"RepetitionTime": 1.5, "StartTime": 0.75}\n'
        )
```

In `tests/test_stop_signal_demo.py` add:

```python
def test_frame_times_start_at_sidecar_start_time(stop_signal_bids_dataset):
    run = demo.load_run(stop_signal_bids_dataset, "sub-s4", "ses-02", **LOAD_KWARGS)
    assert run.frame_times[0] == pytest.approx(0.75)
    assert np.allclose(np.diff(run.frame_times), 1.5)


def test_stop_vs_go_contrast_averages_the_two_stop_conditions(notebook_source):
    assert '"stop_vs_go": "0.5 * stop_success + 0.5 * stop_failure - go_success"' in notebook_source


def test_notebook_default_sessions_are_accepted_by_the_validator(notebook_namespace):
    assert len(notebook_namespace["DEFAULT_SESSIONS"]) == 5
    assert notebook_namespace["_configured_sessions"]() == notebook_namespace["DEFAULT_SESSIONS"]
```

(`LOAD_KWARGS`, `notebook_source`, `notebook_namespace` are the module's existing helpers for the same purposes; reuse their names.) Change the pinned strings at lines 52, 57, 887, 656, 731 to the new contrast text and the `exactly two` expectation at 1492 to `at least two` — these are requirement changes named by review S5.

- [ ] **Step 2: Run** `uv run pytest -q tests/test_stop_signal_demo.py -k "start_time or stop_vs_go or default_sessions"` → FAIL.

- [ ] **Step 3: Commit tests** `git add tests && git commit -m "test: stop-signal contrast weights, session count, StartTime"`

- [ ] **Step 4: Implement**

`examples/stop_signal_demo.py`:

```python
def _frame_times(n_scans: int, tr: float, start_time: float = 0.0) -> np.ndarray:
    return _immutable_array(start_time + np.arange(n_scans, dtype=float) * tr, dtype=float)


def _sidecar_start_time(bold_path: Path) -> float:
    sidecar = bold_path.with_name(bold_path.name.split(".nii")[0] + ".json")
    if not sidecar.exists():
        return 0.0
    value = json.loads(sidecar.read_text()).get("StartTime", 0.0)
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"StartTime in {sidecar.name} must be a finite number")
    return float(value)
```

and in `load_run` pass `_sidecar_start_time(bold_path)` into `_frame_times`.

Notebook `configuration` cell: validator `if len(sessions) < 2 or not all(sessions): raise ValueError("BOLDTAILOR_SESSIONS must select at least two non-empty sessions")`. `ingestion` and `design-fit` cells: the contrast string above; `provenance_metadata["example"] = "multi-session stop-signal whole-brain"`. `design-fit`: `figure, design_axes = plt.subplots(1, len(SESSIONS), figsize=(7.5 * len(SESSIONS), 5), squeeze=False); design_axes = design_axes[0]`. Title cell: keep "Five-session" and make the first sentence say "any two or more sessions; the default is five".

- [ ] **Step 5: Run** `uv run pytest -q --run-notebooks tests/test_stop_signal_demo.py` → pass. Then collapse the notebook executions: keep one `@pytest.mark.notebook` smoke test that executes the notebook once on the fixture and carries the metadata/round-trip assertions; delete the remaining executions and cosmetic pins (`cmap`, `cut_coords`, titles, `display_count == 16`, the 17-digit memory estimate) per the appendix MERGE/CUT rows.

- [ ] **Step 6: Commit** `git add examples/stop_signal_demo.py examples/stop_signal_demo.ipynb tests && git commit -m "fix: stop-signal demo contrast weights, multi-session plotting, sidecar StartTime"`

### Task 1.3: NSD examples describe and verify the uncentered task model (S4, E1, E4)

**Files:**
- Modify: `examples/NSD/workflow_outputs.py:271-300` (`_metadata`), `examples/NSD/workflow_reuse.py:12-40`, `examples/NSD/multisession_inputs.py:132-152`
- Modify: `examples/NSD/nsd_workflow.ipynb` (markdown cell with the regressor table, HEAD version), `examples/NSD/nsd_multisession.ipynb` (markdown cell 9), `README.md:124-125,369-370`, `docs/glmsingle-comparison.md:68-69`, `examples/NSD/nsd_hrf.py:387`, `examples/NSD/hrf_artifacts.py:228`
- Modify: `.gitignore`
- Test: `examples/NSD/test_workflow_reuse.py`, `examples/NSD/test_multisession_analysis.py`

**Interfaces:**
- Produces: `workflow_outputs._metadata(...)["task_model_fingerprint"] == NSD_TASK_MODEL.fingerprint`; `workflow_reuse.validate_saved_settings(metadata, settings)` also raises when `metadata.get("task_model_fingerprint") != NSD_TASK_MODEL.fingerprint`; `multisession_inputs._compatible` compares `"task_model_fingerprint"`.

- [ ] **Step 1: Preserve, then restore, the uncommitted notebooks** (user decision recorded in the review: do not commit the executed outputs or hardcoded `NSD_CONFIG`).

```bash
git stash push examples/NSD/nsd_workflow.ipynb examples/NSD/nsd_multisession.ipynb -m "executed notebooks 2026-10-01 (outputs, personal paths)"
echo ".env" >> .gitignore
rm examples/NSD/.env examples/NSD/sobol_hrfs_preview.png
```

- [ ] **Step 2: Write the failing tests**

`examples/NSD/test_workflow_reuse.py`:

```python
def test_saved_results_with_other_task_model_are_rejected(saved_metadata, settings):
    stale = dict(saved_metadata, task_model_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="task_model_fingerprint"):
        validate_saved_settings(stale, settings)


def test_metadata_records_the_task_model_fingerprint(published_metadata):
    assert published_metadata["task_model_fingerprint"] == NSD_TASK_MODEL.fingerprint
```

`examples/NSD/test_multisession_analysis.py`:

```python
def test_sessions_with_different_task_models_cannot_be_pooled(two_session_records):
    first, other = two_session_records
    other["metadata"]["task_model_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="task_model_fingerprint"):
        _compatible(first, other, ["OLS"])
```

(Reuse the modules' existing fixtures for saved metadata and session records; names above are indicative of what they provide.)

- [ ] **Step 3: Run** `uv run pytest -q examples/NSD/test_workflow_reuse.py examples/NSD/test_multisession_analysis.py -k "task_model"` → FAIL.
- [ ] **Step 4: Commit tests.**
- [ ] **Step 5: Implement**

`workflow_outputs._metadata`: add `task_model_fingerprint=NSD_TASK_MODEL.fingerprint,` (import from `.workflow_inputs`). `workflow_reuse.validate_saved_settings`: after the settings comparison add

```python
    expected = NSD_TASK_MODEL.fingerprint
    if metadata.get("task_model_fingerprint") != expected:
        raise ValueError(
            "Saved analysis settings differ: task_model_fingerprint; "
            "use existing_results='overwrite' to refit"
        )
```

`multisession_inputs._compatible`: add `"task_model_fingerprint"` to `keys`. (Task 1.5 adds `"hrf_normalization"` beside it.)

- [ ] **Step 6: Fix the prose** (code is the reference; `trial_type` is uncentered):
  - `nsd_workflow.ipynb` regressor table row: `| trial_type | Binary code 0/1, uncentered | Type 1 minus type 0, controlling for RT and missing RT |` and the `task` row: `| task | One unit per presentation | Response on type-0 trials at the run-mean observed RT |`.
  - `nsd_multisession.ipynb` cell 9: "the task coefficient is the response on trial_type 0 trials at the run-mean observed RT".
  - `README.md:124-125`: "assumes that the task-model response (the mean response under the default `TaskModel()`) transfers between runs". `README.md:369-370`: "By default this notebook selects HRFs with the full task model (task, centered RT, uncentered trial type); pass `include_rt=False` for stimulus-timing-only selection."
  - `docs/glmsingle-comparison.md:68-69`: same wording as README:124.
  - `nsd_hrf.py:387` and `hrf_artifacts.py:228`: "task-model leave-one-run-out CV R²".

- [ ] **Step 7: Run** `uv run pytest -q examples/NSD` → pass. `uv run pytest -q` → pass.
- [ ] **Step 8: Commit** `git add -A examples/NSD README.md docs/glmsingle-comparison.md .gitignore && git commit -m "fix: NSD reuse checks the task-model fingerprint; prose matches uncentered trial_type"`

### Task 1.4: Formatting gate green

**Files:** the 17 files `uv run black --check ...` reports.

- [ ] **Step 1:** `uv run black src tests examples/NSD examples/stop_signal_demo.py`
- [ ] **Step 2:** `uv run pytest -q` → pass; `git diff --check` → clean.
- [ ] **Step 3:** `git commit -am "style: apply black"`

### Task 1.5: HRF kernels normalized to unit peak amplitude (user requirement, 2026-10-02)

**Why:** Sum-to-one normalization makes a beta the response per unit *integrated* kernel, which differs between kernels of different width, so betas fitted with different selected HRFs are not on a common scale. Peak-to-one normalization makes every beta the peak BOLD response to a unit event in signal units, comparable across grayordinates, across canonical and selected kernels, and with GLMsingle's convention.

**Files:**
- Modify: `src/boldtailor/hrf_library.py:67-88` (`HrfCandidate.kernel`), `src/boldtailor/_hrf_design.py:12-47` (`hrf_model`, `hrf_metadata`, `convolve_events`), `src/boldtailor/_task_design.py:61-76` (`task_columns`), `src/boldtailor/design.py:99-119` (`_make_design_matrix`), `src/boldtailor/_hrf_glm_design.py:43-44,74-91`, `src/boldtailor/_single_trial_design.py:16-19`, `src/boldtailor/_hrf_cv.py:70-78`
- Modify: provenance strings `hrf_normalization="discrete_sum_one"` → `"peak_one"` in `src/boldtailor/_hrf_glm.py:67`, `src/boldtailor/_selected_hrf_fit.py:102`, `src/boldtailor/single_trial.py:94-106` (add `hrf_normalization="peak_one"`), `src/boldtailor/hrf_selection.py` activity (add the same key)
- Modify: `examples/NSD/workflow_outputs.py:271-300` (`hrf_normalization="peak_one"` in `_metadata`; add `"hrf_normalization"` to `workflow_reuse.ANALYSIS_SETTINGS`-style check and to `multisession_inputs._compatible` keys, alongside Task 1.3's `task_model_fingerprint`), `examples/NSD/session_hrf_cache.py` request metadata (add `hrf_normalization`)
- Modify docs: `docs/user-guide.md:463-465` ("Kernels are normalized to sum to one…"), `docs/glmsingle-comparison.md` (normalization paragraph), `README.md` (any "sum to one" mention), `docs/api.md` (`HrfCandidate.kernel`, `HrfLibrary`)
- Test: `tests/test_hrf_library.py:26,62-77`, `tests/test_sobol_hrf_library.py:25`, `tests/test_hrf_design.py:130`, `tests/test_design.py:30,86,199`, `tests/test_task_design.py`, plus oracle tests that build nilearn designs with the string `"spm"`/`"glover"` (switch their oracle to `compute_regressor(..., hrf_kernel(...))`)

**Interfaces:**
- Produces: `_hrf_design.hrf_kernel(model: str | HrfCandidate, tr: float, oversampling: int = 50) -> np.ndarray` returning a read-only kernel with `kernel.max() == 1.0` for `"spm"`, `"glover"`, and every `HrfCandidate`; `HrfCandidate.kernel(tr, oversampling)` peak-normalized for both kinds (canonical = `spm_hrf(tr, oversampling) / spm_hrf(...).max()`); `_hrf_design.hrf_model(candidate)` returns a callable for **every** candidate (no `"spm"` string escape), with `__name__ == "kernel"` so `task_columns` keeps stripping `_kernel`; `design._make_design_matrix` converts plain `"spm"`/`"glover"` to the callable and strips the suffix, leaving derivative/FIR basis strings to nilearn unchanged (documented: those keep nilearn's sum normalization and are not scale-comparable with selected kernels).
- Requirement changes to existing tests: candidate 0 equals `spm_hrf / max(spm_hrf)` instead of `spm_hrf`; design oracles pass the callable to nilearn instead of the string.

- [ ] **Step 1: Failing tests**

`tests/test_hrf_library.py` (replace the line-26 exact-equality test):

```python
def test_every_candidate_kernel_peaks_at_one(two_candidate_library):
    for candidate in two_candidate_library.candidates:
        kernel = candidate.kernel(1.6, 50)
        assert kernel.max() == pytest.approx(1.0)
        assert not kernel.flags.writeable


def test_canonical_kernel_is_peak_scaled_nilearn_spm():
    canonical = HrfCandidate(0, "spm", CANONICAL_PARAMETERS).kernel(1.6, 50)
    reference = spm_hrf(1.6, 50)
    np.testing.assert_allclose(canonical, reference / reference.max())
```

`tests/test_hrf_design.py`:

```python
@pytest.mark.parametrize("name", ["spm", "glover"])
def test_string_models_resolve_to_peak_normalized_kernels(name):
    kernel = hrf_kernel(name, 2.0, 50)
    assert kernel.max() == pytest.approx(1.0)
    assert callable(hrf_model(name))


def test_trial_regressors_match_nilearn_with_the_peak_kernel(two_candidate_library):
    times = 0.5 + 1.6 * np.arange(120)
    events = pd.DataFrame(dict(onset=[10.0, 40.3, 77.1], duration=[1.0, 2.5, 0.0]))
    for candidate in two_candidate_library.candidates:
        fast = trial_regressors(events, times, candidate)
        for j, (o, d) in enumerate(zip(events.onset, events.duration, strict=True)):
            expected = compute_regressor(np.array([[o], [d], [1.0]]), candidate.kernel, times)[0][:, 0]
            np.testing.assert_allclose(fast[:, j], expected, atol=1e-10)
```

`tests/test_fit.py` (scale comparability is the point of the change):

```python
def test_canonical_selection_and_plain_spm_fit_agree_in_scale(selected_fixture):
    data, selection = selected_fixture
    canonical_only = replace_indices(selection, np.zeros(data.n_features, dtype=int))
    model = ModelSpec(contrasts={"task": "task"}, hrf_model="spm", drift_model=None, noise_model="ols", task_model=TaskModel())
    plain = fit(data, model)
    selected = fit(data, model, hrf_selection=canonical_only)
    np.testing.assert_allclose(selected.effect("task"), plain.effect("task"))
```

(`replace_indices` is a small test helper that rebuilds an `HrfSelectionResult` with new indices and a matching `hrf_assignment_fingerprint` in its provenance; `tests/test_hrf_glm.py:193` already does this for its canonical-equals-plain test, reuse that helper.) Also update the design oracles at `tests/test_design.py:30,86,199` and `tests/test_task_design.py` to call `make_first_level_design_matrix(..., hrf_model=hrf_kernel_callable("spm"))` where `hrf_kernel_callable` is `_hrf_design.hrf_model("spm")`, and the NSD metadata test to expect `hrf_normalization == "peak_one"`.

- [ ] **Step 2: Run** `uv run pytest -q tests/test_hrf_library.py tests/test_hrf_design.py tests/test_fit.py -k "peak or peak_kernel or agree_in_scale"` → FAIL. Commit tests: `test: HRF kernels peak-normalized; canonical and selected fits share a scale`.

- [ ] **Step 3: Implement**

`hrf_library.py`:

```python
def _peak_normalized(values):
    peak = np.max(values)
    if not np.isfinite(values).all() or peak <= 0:
        raise ValueError("HRF kernel must be finite with a positive peak")
    return readonly_array(values / peak)


    def kernel(self, tr, oversampling=50):
        dt = _sampling(tr, oversampling)
        if self.kind == "spm":
            return _peak_normalized(spm_hrf(tr, oversampling))
        a, b, c, d, ratio, onset, duration = self.parameters
        ratio_samples = duration / dt
        nearest = round(ratio_samples)
        if abs(ratio_samples - nearest) <= 1e-12 * max(1, ratio_samples):
            ratio_samples = nearest
        times = np.arange(int(np.ceil(ratio_samples))) * dt - onset
        values = gamma.pdf(times, a / c, scale=c) - gamma.pdf(times, b / d, scale=d) / ratio
        return _peak_normalized(values)
```

Update the module docstring ("Deterministic, peak-normalized HRFs; candidate 0 is nilearn's SPM shape").

`_hrf_design.py`:

```python
_CANONICAL = {"spm": HrfCandidate(0, "spm", CANONICAL_PARAMETERS)}


def _glover_kernel(tr, oversampling=50):
    from nilearn.glm.first_level.hemodynamic_models import glover_hrf

    values = glover_hrf(tr, oversampling)
    return readonly_array(values / values.max())


def hrf_kernel(model, tr, oversampling=50):
    """Peak-one kernel for a basis name or candidate."""
    return hrf_model(model)(tr, oversampling)


def hrf_model(candidate):
    if isinstance(candidate, str):
        if candidate == "spm":
            return _CANONICAL["spm"].kernel
        if candidate == "glover":
            return _glover_kernel
        raise ValueError("hrf must be 'spm', 'glover', or an identified HrfCandidate")
    if not isinstance(candidate, HrfCandidate):
        raise ValueError("hrf must be 'spm', 'glover', or an identified HrfCandidate")
    return candidate.kernel
```

Rename `_glover_kernel` so its `__name__` is `"kernel"` (`_glover_kernel.__name__ = "kernel"`) because `task_columns` strips `_kernel`. In `hrf_metadata`, return `dict(id=0, kind="spm", parameters=list(CANONICAL_PARAMETERS), kernel_fingerprint=..., normalization="peak_one")` for the canonical case instead of the bare string `"spm"`. In `convolve_events` and `trial_regressors`, drop the `if hrf_model(candidate) == "spm"` branch (all candidates use the fast path; the canonical case is covered by the new nilearn oracle test).

`_task_design.task_columns`: unchanged logic (callable everywhere; `_kernel` suffix stripped). `design._make_design_matrix`: `hrf = hrf_model(model.hrf_model) if model.hrf_model in ("spm", "glover") else model.hrf_model`; after nilearn returns, `matrix.columns = [c.removesuffix("_kernel") for c in matrix.columns]` when `callable(hrf)`. `_hrf_glm_design.compile_group_designs`: the `candidate.kind == "spm"` branch collapses into `_custom_design` (one path). `_hrf_cv.RunDesign.task_design`: unchanged (passes `hrf_model(candidate)`).

Provenance: replace every `"discrete_sum_one"` with `"peak_one"`; add `hrf_normalization="peak_one"` to the `single_trial` and `hrf_selection` activities and to the NSD `_metadata`; add `"hrf_normalization"` to the saved-settings and multisession compatibility checks introduced in Task 1.3 and to `session_hrf_cache` request metadata (old caches are rejected, which is correct: their betas are on a different scale).

- [ ] **Step 4: Run** `uv run pytest -q` → pass; `uv run pytest -q examples/NSD` → pass (expect to update tests that pinned `"discrete_sum_one"` or compared against string-model nilearn designs; each is a requirement change named here).

- [ ] **Step 5: Docs**

`docs/user-guide.md:463-465`: "Every kernel Boldtailor builds or selects, including canonical SPM and Glover, is scaled to a peak amplitude of one. A beta is therefore the peak BOLD response to a unit-amplitude event in signal units, and betas are comparable across grayordinates with different selected HRFs. Nilearn basis strings with derivatives or FIR are passed to Nilearn unchanged and keep its sum-to-one scaling." Update `docs/glmsingle-comparison.md` normalization paragraph ("Both packages scale kernels to unit peak; remaining scale differences come from signal units and nuisance handling"), `README.md`, and `docs/api.md`. Add a dated note to `docs/validation/nsd-session.md` that recorded beta magnitudes predate peak normalization.

- [ ] **Step 6: Commit** `feat: peak-normalized HRF kernels for scale-comparable betas`


---

# Phase 2: Provenance that identifies inputs and software (V1, S6, L2)

### Task 2.1: One `software_environment()` recorded by every operation

**Files:**
- Modify: `src/boldtailor/_software.py`, `src/boldtailor/_fit_lifecycle.py:28-51`, `src/boldtailor/single_trial.py:104-105`, `src/boldtailor/prepared.py:251-253`, `src/boldtailor/prepared_fit.py:308`
- Test: `tests/test_software.py`, `tests/test_lifecycle.py`

**Interfaces:**
- Produces: `_software.software_environment() -> dict[str, str]` with keys `python`, `platform`, `boldtailor`, `numpy`, `scipy`, `pandas`, `nilearn`.
- Every activity returned through `FitOperation.provenance` carries `activity["software"] == software_environment()`. `analysis_id` is computed by callers before this key is added, so identities stay version-independent (documented).

- [ ] **Step 1: Failing tests**

`tests/test_software.py`:

```python
def test_software_environment_lists_interpreter_platform_and_packages():
    import nilearn, numpy, scipy, pandas, platform
    env = software_environment()
    assert env["python"] == platform.python_version()
    assert env["platform"] == platform.platform()
    assert env["numpy"] == numpy.__version__
    assert env["scipy"] == scipy.__version__
    assert env["pandas"] == pandas.__version__
    assert env["nilearn"] == nilearn.__version__
    assert set(env) == {"python", "platform", "boldtailor", "numpy", "scipy", "pandas", "nilearn"}
```

`tests/test_lifecycle.py`:

```python
@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_every_operation_records_the_software_environment(name, run_entry_point):
    result = run_entry_point(name)
    activity = result.provenance.to_dict()["activities"][-1]
    assert activity["software"] == software_environment()
```

- [ ] **Step 2: Run** → FAIL (`software_environment` undefined). Commit tests.
- [ ] **Step 3: Implement**

```python
def software_environment() -> dict[str, str]:
    """Interpreter, platform, and package versions, resolved at call time."""
    import platform

    packages = ("boldtailor", "numpy", "scipy", "pandas", "nilearn")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{name: package_version(name) for name in packages},
    }
```

In `FitOperation.provenance` replace `activity=activity,` with `activity={**activity, "software": software_environment()},`. Delete the ad-hoc version keys in `single_trial._model_metadata` (`numpy_version`, `nilearn_version`), `prepared.py:251-253`, `prepared_fit.py:308`, and update any test asserting those old keys (requirement change: one software record).

- [ ] **Step 4: Run** `uv run pytest -q` → pass. Commit `feat: record one software environment in every operation's provenance`.

### Task 2.2: Optional content digest on `SourceRef`; emit BEP028 `Digest`

**Files:**
- Modify: `src/boldtailor/provenance.py:14,34-80,503-509`, `src/boldtailor/bids_provenance.py:29-39,221-248,489-498`
- Test: `tests/test_provenance.py`, `tests/test_bids_provenance.py:250`

**Interfaces:**
- Produces: `SourceRef(..., sha256: str | None = None)`; `to_dict()` includes `"sha256"` when set; `from_dict` reads it; `_metadata_fingerprint` changes when `sha256` changes; BEP028 file entity gets `"Digest": {"sha256": value}` when set.
- Removes: `_FORBIDDEN_TOP_LEVEL_FIELDS`; `"digest"` from `bids_provenance._SENSITIVE_KEYS`.

- [ ] **Step 1: Failing tests**

```python
def test_sha256_changes_the_metadata_fingerprint(complete_sources):
    a = complete_sources(1)
    b = complete_sources(1)
    with_digest = replace(b[0].signal, sha256="a" * 64)
    b = [replace(b[0], signal=with_digest)]
    assert _metadata_fingerprint(a) != _metadata_fingerprint(b)
    assert SourceRef.from_dict(with_digest.to_dict()) == with_digest


@pytest.mark.parametrize("bad", ["ABC", "g" * 64, "a" * 63, 7])
def test_sha256_must_be_64_lowercase_hex_characters(bad):
    with pytest.raises(ValueError, match="sha256"):
        SourceRef(role="signal", sha256=bad)
```

`tests/test_bids_provenance.py`: change the line-250 absence assertion (requirement change) to:

```python
def test_file_entities_carry_digest_when_supplied(record_with_digest):
    files = project_bids_provenance(record_with_digest)["ent"]["Files"]
    assert files[0]["Digest"] == {"sha256": "a" * 64}
```

- [ ] **Step 2: Run** → FAIL. Commit tests.
- [ ] **Step 3: Implement**

`provenance.py`: add field `sha256: str | None = None` after `modified_at`; in `__post_init__` add `object.__setattr__(self, "sha256", _validate_sha256(self.sha256))` with

```python
def _validate_sha256(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    return value
```

extend `to_dict`/`from_dict`; delete `_FORBIDDEN_TOP_LEVEL_FIELDS` and `_reject_forbidden_top_level_fields` (make `_extra_fields` return `extra`). `bids_provenance.py`: remove `"digest"` from `_SENSITIVE_KEYS`; in `_file_entity` add `if source.sha256: entity["Digest"] = {"sha256": source.sha256}`.

- [ ] **Step 4: Update messages.** In `fit.py:159`, `prepared_fit.py:156`, `_hrf_glm.py:151` change the parent mismatch text to `"full result provenance identity does not match this data and model (identity uses source metadata and sha256 when supplied)"`; update the matching `match=` fragments to `"provenance identity"`.
- [ ] **Step 5: Document** in `docs/user-guide.md` (provenance section): "Supply `sha256` on each `SourceRef` to make identity content-based; without it, identity rests on path, size, and modification time." Note in `docs/development.md` that the digest ban was removed on 2026-10-02 and why.
- [ ] **Step 6: Run** `uv run pytest -q` → pass. Commit `feat: optional sha256 on SourceRef; BEP028 Digest; content-aware identity`.

### Task 2.3: Record how the HRF library was built

**Files:**
- Modify: `src/boldtailor/hrf_library.py:91-188`, `src/boldtailor/hrf_selection.py:58-91`
- Test: `tests/test_hrf_library.py`, `tests/test_hrf_selection.py`

**Interfaces:**
- Produces: `HrfLibrary.origin: Mapping[str, object]` (frozen, `compare=False`); `sobol_hrf_library` sets `{"kind": "sobol", "n_samples": n, "seed": seed, "duration": 36.0}`; `expanded_hrf_library` sets `{"kind": "expanded_grid"}`; `from_parameters(rows, origin=None)` defaults to `{"kind": "explicit", "n_candidates": len(rows)}`. Selection activity gains `library=dict(library.origin)` beside `library_fingerprint`.

- [ ] **Step 1: Failing tests**

```python
def test_sobol_library_records_its_constructor():
    library = sobol_hrf_library(n_samples=8, seed=3)
    assert dict(library.origin) == {"kind": "sobol", "n_samples": 8, "seed": 3, "duration": 36.0}


def test_selection_provenance_records_library_origin(cv_fixture):
    data, library = cv_fixture
    activity = select_hrf(data, library=library).provenance.to_dict()["activities"][-1]
    assert activity["library"] == dict(library.origin)
    assert activity["library_fingerprint"] == library.fingerprint
```

- [ ] **Step 2: Run** → FAIL; commit tests.
- [ ] **Step 3: Implement**: add `origin: Mapping[str, object] = field(default_factory=dict, compare=False)` to `HrfLibrary`; in `__post_init__` freeze with `MappingProxyType(dict(self.origin))`; set in the three factories; add `library=dict(library.origin),` to `hrf_selection._provenance`.
- [ ] **Step 4: Run** `uv run pytest -q` → pass. Commit `feat: HRF library origin recorded in selection provenance`.

### Task 2.4: HRF selection runs inside the lifecycle (L2)

**Files:**
- Modify: `src/boldtailor/hrf_selection.py:58-91,94-135,192-244`
- Test: `tests/test_lifecycle.py` (add `"hrf_selection"` and `"hrf_independent_evaluation"` to `ENTRY_POINTS`)

**Interfaces:**
- Produces: `select_hrf`/`evaluate_hrf_split` emit `hrf_selection_started/completed/failed` (resp. `hrf_independent_evaluation_*`), carry `software`, and extend the parent's event history.

- [ ] **Step 1:** Add the two names to `ENTRY_POINTS` and the corresponding calls to `run_entry_point`; run `uv run pytest -q tests/test_lifecycle.py` → FAIL (no events).
- [ ] **Step 2:** Commit tests.
- [ ] **Step 3: Implement**: wrap the body of `select_hrf` in `with fit_operation("hrf_selection", data.provenance) as operation:` and build provenance through `operation.provenance(activity, analysis_id=analysis_fingerprint(data.provenance.metadata_fingerprint, activity))`; same for `evaluate_hrf_split` with name `"hrf_independent_evaluation"`. Reduce `_provenance` to an `_activity(...)` dict builder (no `extend_provenance` call).
- [ ] **Step 4:** `uv run pytest -q` → pass. Commit `refactor: HRF selection uses the shared fit lifecycle`.


---

# Phase 3: Scientific follow-through

### Task 3.1: Missing oracles for the conventional GLM (T2, review §6 items 2, 3, 7)

**Files:**
- Modify: `tests/test_multirun.py`, `tests/test_hrf_glm.py`
- Test only; implementation changes only if a test exposes a defect (report it, do not patch the test).

**Interfaces:**
- Consumes: `tests.oracles.nilearn_original_space_r2`; nilearn `run_glm`, `compute_contrast`.

- [ ] **Step 1: Multi-run AR(1) contrast oracle from the documented contract** (`tests/test_multirun.py`)

```python
def _per_run_nilearn(signals, designs, vector, noise_model):
    from nilearn.glm import compute_contrast
    from nilearn.glm.first_level import run_glm

    out = []
    for y, x in zip(signals, designs, strict=True):
        labels, results = run_glm(y, x.to_numpy(), noise_model=noise_model)
        out.append(compute_contrast(labels, results, vector, stat_type="t"))
    return out


@pytest.mark.parametrize("noise_model", ["ols", "ar1"])
def test_multirun_contrasts_are_equal_weight_fixed_effects(three_run_problem, noise_model):
    from scipy import stats

    signals, events, confounds = three_run_problem
    model = ModelSpec(contrasts={"c": {"face": 1.0, "house": -1.0}}, drift_model=None, noise_model=noise_model)
    data = from_arrays(signals, events, confounds=confounds, tr=2.0)
    result = fit(data, model)
    designs = [d.matrix for d in compile_designs(data, model)]
    vector = _weight_vector({"face": 1.0, "house": -1.0}, designs[0].columns, "c", 0)
    runs = _per_run_nilearn(signals, designs, vector, noise_model)
    n = len(runs)
    effect = sum(r.effect_size() for r in runs) / n
    variance = sum(r.effect_variance() for r in runs) / n**2
    dof = sum(r.dof for r in runs)
    t = effect / np.sqrt(variance)
    np.testing.assert_allclose(result.effect("c"), effect)
    np.testing.assert_allclose(result.variance("c"), variance)
    np.testing.assert_allclose(result.stat("c"), t)
    np.testing.assert_allclose(result.one_sided_p_value("c"), stats.t.sf(t, dof))
```

(`three_run_problem`: reuse the module's existing multi-run fixture; `_weight_vector` from `boldtailor._conventional`, or build the vector by column position.)

- [ ] **Step 2: Rank-deficient but estimable design** (`tests/test_multirun.py`)

```python
def test_duplicated_regressor_contrast_matches_reduced_design(single_run_problem):
    signals, events, design, _ = single_run_problem
    duplicated = design.assign(face_copy=design["face"])
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=[signals], design_matrices=[duplicated], frame_times=[np.arange(len(signals)) * 2.0],
        column_roles=[{c: ("task" if c in ("face", "face_copy", "house") else "nuisance") for c in duplicated.columns}],
    )
    with pytest.warns(UserWarning, match="design rank"):
        result = fit_prepared(prepared, contrasts={"c": {"face": 0.5, "face_copy": 0.5, "house": -1.0}}, noise_model="ols", model_metadata={})
    labels, results = run_glm(signals, design.to_numpy(), noise_model="ols")
    vector = np.array([1.0 if c == "face" else -1.0 if c == "house" else 0.0 for c in design.columns])
    expected = compute_contrast(labels, results, vector, stat_type="t")
    np.testing.assert_allclose(result.effect("c"), expected.effect_size())
    np.testing.assert_allclose(result.variance("c"), expected.effect_variance())
    np.testing.assert_allclose(result.stat("c"), expected.stat())
```

(Adapt `column_roles`/`from_arrays` keywords to the actual `PreparedDesignAnalysis.from_arrays` signature in `src/boldtailor/prepared.py`.)

- [ ] **Step 3: Task-model selected-GLM numerical oracle** (`tests/test_hrf_glm.py`, replaces the deleted `:709`)

Build signals with known amplitudes: for each feature choose a library candidate `cid`, compute `columns = run_task_columns(events, task_model, times, hrf_model(library.candidates[cid]), run=r, min_onset=-24.0, oversampling=50)` and set `y = columns @ [3.0, 0.8, -0.6] + confounds @ [0.3] + 50 + noise(sd 0.05)`. Then `selection = select_hrf(data, library=library, task_model=task_model)`, `result = fit(data, ModelSpec(contrasts={"rt": {"response_time": 1.0}}, hrf_model="spm", noise_model="ols", drift_model=None, task_model=task_model), hrf_selection=selection)`, and for each feature compare `result.effect("rt")[f]` with `compute_contrast` on `run_glm(y[:, [f]], result.group_designs[(0, cid)].to_numpy())` (single run case) using the `response_time` column vector. Assert `assert_allclose(..., rtol=1e-8)` and that `selection.hrf_indices` recovers `cid` for every feature.

- [ ] **Step 4: Run** the three tests → PASS expected. If any fails, stop and report the discrepancy as a finding (do not adjust tolerances). Commit `test: nilearn oracles for multi-run AR(1), rank-deficient, and task-model selected GLMs`.

### Task 3.2: Boundary diagnostics for ridge grids and the HRF parameter box (S3)

**Files:**
- Modify: `src/boldtailor/ridge_results.py:55-70,104-123`, `src/boldtailor/ridge_selection.py:18-49`, `src/boldtailor/fractional_ridge.py:9-36`, `src/boldtailor/hrf_library.py` (add `parameter_bounds`), `src/boldtailor/hrf_results.py:14-40`, `src/boldtailor/hrf_selection.py:94-135`
- Test: `tests/test_ridge_selection.py`, `tests/test_hrf_library.py`, `tests/test_hrf_selection.py`
- Docs: `docs/user-guide.md` (ridge and HRF sections), `docs/api.md`

**Interfaces:**
- Produces: `FractionSelection.at_boundary: np.ndarray[bool]` (True where the winner is the largest or smallest grid fraction and the feature is scored); `RidgeSelection.at_boundary: bool`; `HrfLibrary.parameter_bounds -> pd.DataFrame` (index `PARAMETER_NAMES[:6]`, columns `low`, `high`, over custom candidates); `HrfSelectionResult.at_parameter_bound: np.ndarray[bool]` (selected custom candidate has any of its first six parameters within 2 % of the box width of `low` or `high`; canonical and `-1` are False).

- [ ] **Step 1: Failing tests**

```python
def test_fraction_selection_flags_grid_endpoints():
    scores = np.array([[0.1, 0.5, 0.3], [0.2, 0.4, 0.9], [0.9, 0.1, 0.1]])  # rows: fractions
    choice = select_ridge_fractions(scores, (1.0, 0.5, 0.1))
    np.testing.assert_array_equal(choice.ridge_fraction, [0.1, 1.0, 0.5])
    np.testing.assert_array_equal(choice.at_boundary, [True, True, False])


def test_ridge_penalty_selection_flags_grid_endpoint():
    scores = np.array([[0.9, 0.9], [0.1, 0.1]])
    assert select_ridge_penalty(scores, (0.0, 1.0)).at_boundary is True


def test_parameter_bounds_cover_custom_candidates(two_candidate_library):
    bounds = two_candidate_library.parameter_bounds
    assert list(bounds.index) == list(PARAMETER_NAMES[:6])
    assert bounds.loc["response_delay", "low"] == 3 and bounds.loc["response_delay", "high"] == 6


def test_selection_flags_features_at_the_parameter_box_edge(cv_fixture):
    data, library = cv_fixture
    selection = select_hrf(data, library=library)
    expected = np.array([
        cid > 0 and _near_edge(library, cid) for cid in selection.hrf_indices
    ])
    np.testing.assert_array_equal(selection.at_parameter_bound, expected)
```

with the test helper

```python
def _near_edge(library, cid):
    bounds = library.parameter_bounds
    values = np.asarray(library.candidates[cid].parameters[:6])
    width = (bounds["high"] - bounds["low"]).to_numpy()
    return bool(np.any((values - bounds["low"].to_numpy() <= 0.02 * width) | (bounds["high"].to_numpy() - values <= 0.02 * width)))
```

- [ ] **Step 2: Run** → FAIL; commit tests.
- [ ] **Step 3: Implement**

`fractional_ridge.select_ridge_fractions`: after computing `winner`, `at_boundary = np.zeros(scores.shape[1], dtype=bool); at_boundary[mask] = np.isin(winner, [0, len(grid) - 1])`; pass to `FractionSelection(..., at_boundary=at_boundary)`. `ridge_selection.select_ridge_penalty`: `at_boundary = winner in (0, len(alphas) - 1)`. Add the fields (with `readonly_array(..., dtype=bool)` in `__post_init__`) to both result classes.

`hrf_library.HrfLibrary.parameter_bounds`:

```python
    @property
    def parameter_bounds(self):
        rows = np.array([c.parameters[:6] for c in self.candidates if c.kind != "spm"])
        if not len(rows):
            rows = np.array([CANONICAL_PARAMETERS[:6]])
        return pd.DataFrame(
            {"low": rows.min(axis=0), "high": rows.max(axis=0)}, index=PARAMETER_NAMES[:6]
        )
```

`hrf_selection._select`: compute `at_bound` with the same 2 % rule over `indices` and pass to `HrfSelectionResult(..., at_parameter_bound=at_bound)`; add the field (default `None` → zeros) to the dataclass.

- [ ] **Step 4: Docs.** User guide, ridge section: "`at_boundary` marks features whose winner sits at an end of the candidate grid; a high proportion means the grid should be extended." HRF section: "`at_parameter_bound` marks features whose selected kernel lies within 2 % of the sampled parameter box; the default box implies peak times of roughly 1.5–7.5 s, so late-peaking responses will saturate at the box edge." Add both to `docs/api.md`.
- [ ] **Step 5:** `uv run pytest -q` → pass. Commit `feat: boundary flags for ridge grids and the HRF parameter box`.

### Task 3.3: Recovery tests under realistic noise (T1; review §6 items 4–6)

**Files:**
- Create: `tests/test_recovery.py`

**Interfaces:**
- Consumes: `from_arrays`, `compile_trial_run`, `score_fraction_candidates`, `select_ridge_fractions`, `score_ridge_candidates`, `select_ridge_penalty`, `fit_single_trials`, `select_hrf`, `HrfLibrary.from_parameters`, `convolve_events`.

These are validation tests of the method, not of the code path. **If one fails, record the result in `docs/validation/recovery-2026-10.md` and report it; do not loosen thresholds or shrink the problem.**

- [ ] **Step 1: Fraction recovery**

```python
def _ar1(rng, n, rho, sd):
    e = rng.normal(0, sd, n)
    for t in range(1, n):
        e[t] += rho * e[t - 1]
    return e


def _dense_trial_problem(rng, *, n_runs=6, n_trials=24, isi=3.0, tr=1.0, noise_sd=1.5):
    signals, events, times, confounds, predictors, truth = [], [], [], [], [], []
    for r in range(n_runs):
        t = np.arange(int(n_trials * isi / tr) + 30) * tr
        rt = rng.uniform(0.4, 1.6, n_trials)
        kind = rng.integers(0, 2, n_trials)
        e = pd.DataFrame(dict(onset=6 + np.arange(n_trials) * isi, duration=1.0, response_time=rt, trial_type=kind))
        n = pd.DataFrame(dict(drift=np.linspace(-1, 1, len(t))))
        x, _, _ = compile_trial_run(e, t, n, f"run-{r}")
        beta = 2.0 + 1.0 * (rt - rt.mean()) - 0.8 * kind + rng.normal(0, 0.3, n_trials)
        y = x.to_numpy() @ beta + 100 + _ar1(rng, len(t), 0.4, noise_sd)
        signals.append(np.column_stack([y, y + rng.normal(0, 0.01, len(t))]))
        events.append(e); times.append(t); confounds.append(n)
        predictors.append(e[["response_time", "trial_type"]]); truth.append(beta)
    data = from_arrays(signals, events, frame_times=times, confounds=confounds)
    return data, predictors, truth


def test_fractional_cv_prefers_shrinkage_when_trials_overlap_and_noise_is_high():
    rng = np.random.default_rng(11)
    data, predictors, truth = _dense_trial_problem(rng)
    grid = (1.0, 0.8, 0.6, 0.4, 0.2)
    scores = score_fraction_candidates(data, predictors, fractions=grid)
    choice = select_ridge_fractions(scores.cv_r2, scores.grid)
    assert np.all(choice.ridge_fraction < 1.0)
    ols = fit_single_trials(data)
    ridge = fit_single_trials(data, ridge_fraction=choice.ridge_fraction)
    def centered_rmse(fit):
        err = [b[:, 0] - b[:, 0].mean() - (tb - tb.mean()) for b, tb in zip(fit.run_betas, truth)]
        return np.sqrt(np.mean(np.concatenate(err) ** 2))
    assert centered_rmse(ridge) < centered_rmse(ols)


def test_fractional_cv_selects_ols_when_noise_is_negligible():
    rng = np.random.default_rng(12)
    data, predictors, _ = _dense_trial_problem(rng, isi=8.0, noise_sd=0.01)
    scores = score_fraction_candidates(data, predictors, fractions=(1.0, 0.8, 0.6, 0.4, 0.2))
    choice = select_ridge_fractions(scores.cv_r2, scores.grid)
    assert np.all(choice.ridge_fraction == 1.0)
```

- [ ] **Step 2: Alpha recovery end-to-end**

```python
def test_shared_alpha_cv_prefers_a_positive_penalty_under_high_noise():
    rng = np.random.default_rng(13)
    data, predictors, _ = _dense_trial_problem(rng)
    scores = score_ridge_candidates(data, predictors, alphas=(0.0, 0.3, 1.0, 3.0, 10.0))
    assert select_ridge_penalty(scores.cv_r2, scores.grid, percentile=50.0).ridge_alpha > 0.0
```

- [ ] **Step 3: HRF recovery under AR(1) noise**

```python
def test_hrf_selection_recovers_the_generating_kernel_under_ar1_noise():
    rng = np.random.default_rng(21)
    library = HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [4, 12, 0.8, 1.0, 4, 0.5, 36], [5, 14, 1.0, 1.5, 6, 1.0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )
    true_ids = np.array([0, 1, 2, 3, 2, 1, 0, 3])
    signals, events, times, confounds = [], [], [], []
    for r in range(4):
        t = 0.8 + 1.6 * np.arange(150)
        e = pd.DataFrame(dict(onset=10 + np.arange(16) * 14.0 + rng.uniform(-1, 1, 16), duration=2.0))
        n = pd.DataFrame(dict(motion=np.sin(np.arange(len(t)) / 10 + r)))
        cols = []
        for cid in true_ids:
            x = convolve_events(e.onset, e.duration, t, library.candidates[int(cid)])
            cols.append(3.0 * x + 0.5 * n.motion.to_numpy() + 100 + _ar1(rng, len(t), 0.4, 1.0))
        signals.append(np.column_stack(cols)); events.append(e); times.append(t); confounds.append(n)
    data = from_arrays(signals, events, frame_times=times, confounds=confounds)
    selection = select_hrf(data, library=library)
    assert np.mean(selection.hrf_indices == true_ids) >= 0.75
    assert np.nanmedian(selection.delta_cv_r2[true_ids != 0]) > 0.0


def test_hrf_selection_keeps_canonical_when_truth_is_canonical():
    rng = np.random.default_rng(22)
    library = HrfLibrary.from_parameters([[4, 12, 0.8, 1.0, 4, 0.5, 36], [6, 16, 1.5, 2.5, 8, 2, 36]])
    signals, events, times, confounds = [], [], [], []
    for r in range(4):
        t = 0.8 + 1.6 * np.arange(150)
        e = pd.DataFrame(dict(onset=10 + np.arange(16) * 14.0, duration=2.0))
        n = pd.DataFrame(dict(motion=np.cos(np.arange(len(t)) / 9 + r)))
        x = convolve_events(e.onset, e.duration, t, "spm")
        y = np.column_stack([3.0 * x + 100 + _ar1(rng, len(t), 0.4, 1.0) for _ in range(6)])
        signals.append(y); events.append(e); times.append(t); confounds.append(n)
    selection = select_hrf(from_arrays(signals, events, frame_times=times, confounds=confounds), library=library)
    assert np.mean(selection.hrf_indices == 0) >= 0.75
```

- [ ] **Step 4: Run** `uv run pytest -q tests/test_recovery.py` → report outcome. If all pass: commit `test: recovery of fraction, alpha, and HRF under realistic noise`. If any fail: commit the tests marked `@pytest.mark.xfail(strict=True, reason="recovery finding 2026-10: <one line>")`, write the validation note, and surface it; the plan's Tier-2 estimand work (review P1) starts from that note.

### Task 3.4: Document the indicator-column in-sample term and the fixed-effects weighting (S2, P2, T6)

**Files:**
- Modify: `docs/user-guide.md` (HRF selection section after the `missing="indicator"` paragraph; conventional GLM section line 82), `docs/api.md` (fixed effects), `docs/glmsingle-comparison.md:98`, `docs/validation/documentation-audit-2026-09-28.md` (banner + sanitizer sentence), `docs/validation/scientific-readability-notebooks-2026-09-28.md` (banner), `docs/validation/nsd-fractional-ridge.md:86-88`, `docs/review-2026-09-28-full-project.md` (strike the "constant features remediated" claim with a dated note)
- Modify: `src/boldtailor/hrf_selection.py` `_activity` (add `profiled_columns_note`)

- [ ] **Step 1:** User guide, after the indicator paragraph: "Because the indicator column is convolved with each candidate kernel and fitted within the held-out run, a small part of each candidate's score is in-sample. With few missing trials this is negligible; with many, compare `at_parameter_bound` and `delta_cv_r2` against a `missing="error"` run on complete trials." Add to the selection activity: `profiled_columns="fit in-sample per run with the candidate kernel; see user guide"`.
- [ ] **Step 2:** User guide line 82 and `api.md`: append "This is not precision-weighted: each run contributes equally regardless of its length or noise level, and degrees of freedom are summed. For runs with very different noise, compare against nilearn's `compute_fixed_effects` with precision weighting."
- [ ] **Step 3:** Replace the personal path at `glmsingle-comparison.md:98` with "a local GLMsingle checkout"; add the dated-record banner to the two validation docs lacking it; fix the sanitizer sentence; replace the dangling temp-script reference; add to the 2026-09-28 review a one-line note: "2026-10-02: the constant-feature masking described in §2.1 was not present in the tree; fixed in Task 1.1 of the remediation plan."
- [ ] **Step 4:** `uv run pytest -q` → pass (a test pins the activity keys; update its expected set). Commit `docs: indicator in-sample note, fixed-effects weighting, stale references`.

---

# Phase 4: Architecture

### Task 4.1: One nested-OLS ΔR² helper and one activity schema (L1)

**Files:**
- Modify: `src/boldtailor/_fit_diagnostics.py`, `src/boldtailor/results.py:200-240` (`make_task_delta_r2_result`), `src/boldtailor/fit.py:79-118,135-147,160-188`, `src/boldtailor/prepared_fit.py:77-113,181-188`, `src/boldtailor/_hrf_glm.py:145-199`, `src/boldtailor/hrf_glm_results.py:60-90` (`_masked_delta_result` removed)
- Test: `tests/test_fit_diagnostics.py`, existing ΔR² oracles in `test_fit.py:637`, `test_prepared_fit.py:934`, `test_hrf_glm.py:217`

**Interfaces:**
- Produces: `_fit_diagnostics.nested_ols_delta(signals, full_designs, nuisance_designs, *, allow_undefined: bool) -> tuple[np.ndarray, np.ndarray]` (full R², nuisance R²; raises `ValueError("nested OLS monotonicity violated")` below `-NESTED_OLS_TOLERANCE`; raises on non-finite unless `allow_undefined`); `_fit_diagnostics.delta_r2_activity(*, name, parent_id, inferential_noise_model, nuisance_model, undefined_features) -> dict` (single schema: `name, stage, parent_analysis_id, definition, clip_below_zero, clip_policy, diagnostic_noise_model, inferential_noise_model, nuisance_model, undefined_features`); `results.make_task_delta_r2_result(..., allow_undefined=False)` enforcing the same tolerance (resolves the `test_fit.py:690` inconsistency: the constructor raises below tolerance instead of silently clipping).

- [ ] **Step 1: Failing tests** in `tests/test_fit_diagnostics.py`:

```python
def test_nested_ols_delta_returns_r2_pair_and_rejects_nonmonotone(single_run_problem):
    signals, _, design, _ = single_run_problem
    nuisance = design[["constant"]] if "constant" in design else design.iloc[:, -1:]
    full, null = nested_ols_delta([signals], [design], [nuisance], allow_undefined=False)
    assert np.all(full >= null - 1e-12)
    with pytest.raises(ValueError, match="monotonicity"):
        nested_ols_delta([signals], [nuisance], [design], allow_undefined=False)


def test_delta_activity_schema_is_shared():
    activity = delta_r2_activity(name="task_delta_r2", parent_id="p", inferential_noise_model="ar1", nuisance_model={"events": False}, undefined_features=0)
    assert set(activity) == {"name", "stage", "parent_analysis_id", "definition", "clip_below_zero", "clip_policy", "diagnostic_noise_model", "inferential_noise_model", "nuisance_model", "undefined_features"}


def test_delta_result_constructor_rejects_impossible_negative_delta():
    with pytest.raises(ValueError, match="monotonicity"):
        make_task_delta_r2_result(full_r2=np.array([0.1]), nuisance_r2=np.array([0.35]), nuisance_designs=(), provenance=None)
```

- [ ] **Step 2:** Run → FAIL; commit tests.
- [ ] **Step 3: Implement** the two helpers in `_fit_diagnostics.py` (move `_fit_r2_analysis`/`_fit_prepared_r2` bodies into `nested_ols_delta`, calling `fit_r2_designs(..., DIAGNOSTIC_NOISE_MODEL)` twice and `validate_nested_ols_delta`), make the three entry points call them, delete `_masked_delta_result` (the `allow_undefined=True` path replaces it), and build all three activities through `delta_r2_activity`. Update the three provenance-key tests to the shared schema (requirement change: one schema).
- [ ] **Step 4:** `uv run pytest -q` → pass; confirm `wc -l src/boldtailor/fit.py src/boldtailor/prepared_fit.py src/boldtailor/_hrf_glm.py` dropped by ≥ 100 lines total. Commit `refactor: single nested-OLS delta-R2 helper and activity schema`.

### Task 4.2: One run-label validator (L6)

**Files:**
- Modify: `src/boldtailor/data.py` (add `run_labels_for`), `src/boldtailor/hrf_selection.py:26-40` (delete; import from data), `src/boldtailor/single_trial.py:48-58`, `src/boldtailor/_single_trial_design.py:34-35`, `src/boldtailor/_selected_hrf_fit.py:24-29`, `src/boldtailor/_ridge_cv.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `data.run_labels_for(data: AnalysisData, run_labels: Sequence[str] | None) -> tuple[str, ...]` (defaults `run-01…`; requires unique, `[A-Za-z0-9_-]+`, count == `n_runs`; message `"run labels must be unique [A-Za-z0-9_-]+ strings, one per run"`).

- [ ] **Step 1:** Failing test:

```python
@pytest.mark.parametrize("labels", [["a", "a"], ["a b", "c"], ["only-one"], [1, 2]])
def test_run_labels_for_rejects_invalid_labels(two_run_data, labels):
    with pytest.raises(ValueError, match="run labels must be unique"):
        run_labels_for(two_run_data, labels)


@pytest.mark.parametrize("entry", ["fit_single_trials", "fit_selected_hrfs", "select_hrf", "score_ridge_candidates"])
def test_every_entry_point_uses_the_same_run_label_error(entry, run_entry_point_with_labels):
    with pytest.raises(ValueError, match="run labels must be unique"):
        run_entry_point_with_labels(entry, ["a b", "c"])
```

- [ ] **Step 2:** Run → FAIL (messages differ); commit.
- [ ] **Step 3: Implement** in `data.py`; replace the three validators; `_validate_events` keeps only an `assert`-free check that `run_label` is a string (labels are validated upstream).
- [ ] **Step 4:** `uv run pytest -q` → pass. Commit `refactor: one run-label validator`.

### Task 4.3: Break the `fit` ↔ `_hrf_glm` cycle; move model identity into `model.py` (L5)

**Files:**
- Modify: `src/boldtailor/model.py` (add `model_identity`), `src/boldtailor/fit.py:189-208` (delete `_model_provenance`; use `model_identity`), `src/boldtailor/_hrf_glm.py:130-135,168-174`, `src/boldtailor/prepared.py:430-434` (delete duplicate `_validate_run_count`; import from `data`)
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `model.model_identity(model: ModelSpec, *, hrf_model: object = "keep") -> ModelIdentity` where `ModelIdentity(activity: dict, fingerprint: dict, warnings: tuple)` is the renamed `_ModelProvenance`; `hrf_model={"kind": "selected"}` replaces the `replace(model, hrf_model=None, task_model=None)` dance in `_hrf_glm`.

- [ ] **Step 1:** Failing test:

```python
def test_model_identity_reports_selected_hrf_without_a_lazy_import():
    import boldtailor.model as model_module
    spec = ModelSpec(contrasts={"c": "a"}, task_model=TaskModel())
    identity = model_module.model_identity(spec, hrf_model={"kind": "selected"})
    assert identity.activity["hrf_model"] == {"kind": "selected"}
    assert identity.activity["task_model"] == spec.task_model.to_dict()
    assert "boldtailor.fit" not in sys.modules or "_model_provenance" not in dir(sys.modules["boldtailor.fit"])
```

- [ ] **Step 2:** Run → FAIL; commit.
- [ ] **Step 3: Implement**; remove both lazy imports in `_hrf_glm.py`; remove the lazy `_hrf_glm` import in `fit.py` by importing `fit_selected_glm` and `selected_task_delta_r2` at module top (no cycle remains once `_model_provenance` lives in `model.py`).
- [ ] **Step 4:** `uv run pytest -q` → pass. Commit `refactor: model identity lives in model.py; no fit/_hrf_glm import cycle`.

### Task 4.4: Stop retaining every (run, HRF) design matrix in results (N1, L3)

**Files:**
- Modify: `src/boldtailor/single_trial_results.py:28-45` (`SelectedTrialDesign`), `src/boldtailor/_selected_hrf_fit.py:45-78,112-154`, `src/boldtailor/hrf_glm_results.py` (`HrfAnalysisResult.group_designs`), `src/boldtailor/_hrf_glm.py:101-126`, `src/boldtailor/design.py:31-37`, `src/boldtailor/_ridge_cv.py:28-36,76-90`
- Modify: `examples/NSD/workflow_analysis.py:89-142`, `examples/NSD/nsd_single_trial.py:146`
- Test: `tests/test_selected_hrf_fit.py`, `tests/test_hrf_glm.py`, `tests/test_design.py`

**Interfaces:**
- Produces: `SelectedTrialDesign.matrix(run: int, hrf_id: int) -> np.ndarray` (rebuilt on demand from the retained `(events, frame_times, confounds, library)` references; identical values to today's stored matrices); `SelectedTrialDesign.design_fingerprint: str` (the sha256 already computed in `_fit_activity`); `SelectedTrialDesign.matrices` **removed**. `HrfAnalysisResult.group_design(run, hrf_id) -> pd.DataFrame` likewise; `group_designs` removed. `compile_nuisance_designs` and `_ridge_cv` hoist `events = data.events; confounds = data.confounds` once.

- [ ] **Step 1:** Failing tests:

```python
def test_selected_design_rebuilds_the_fitted_matrix_on_demand(selected_fixture):
    data, selection = selected_fixture
    result = fit_selected_hrfs(data, selection=selection, feature_signature="ordered-axis")
    run = prepare_runs(data, selection.library)[0]
    expected = np.column_stack([run.trial_matrix(1), run.nuisance])
    np.testing.assert_array_equal(result.design.matrix(0, 1), expected)
    assert not hasattr(result.design, "matrices")
    assert len(result.design.design_fingerprint) == 64


def test_result_memory_does_not_scale_with_selected_hrf_count(selected_fixture):
    import sys
    data, selection = selected_fixture
    result = fit_selected_hrfs(data, selection=selection, feature_signature="ordered-axis")
    retained = sum(a.nbytes for a in result.run_betas) + sum(sys.getsizeof(v) for v in vars(result.design).values())
    assert retained < 10 * sum(a.nbytes for a in result.run_betas)
```

- [ ] **Step 2:** Run → FAIL; commit.
- [ ] **Step 3: Implement**: `SelectedTrialDesign` keeps `hrf_indices`, `design_fingerprint`, `selection_provenance`, and a private `_rebuild: Callable[[int, int], np.ndarray]` (closure over `prepare_runs(data, library)` output, which already caches trial matrices); `_fit_run` stops collecting `designs` and instead feeds the running sha256; `HrfAnalysisResult` stores `_group_builder` and exposes `group_design(run, hrf_id)`. Update the two example call sites to call the methods for the `(run, hrf)` pairs they actually export. Hoist the per-run accessors in `design.compile_nuisance_designs`, `_ridge_cv.subset_runs`, and `prepare_run_beta_path`.
- [ ] **Step 4:** `uv run pytest -q && uv run pytest -q examples/NSD` → pass. Commit `perf: rebuild selected designs on demand instead of retaining every (run, HRF) matrix`.

### Task 4.5: Right-size provenance and publication (V2–V7)

**Files:**
- Modify: `src/boldtailor/provenance.py:21-30,349-358,380-383,386-446`, `src/boldtailor/prepared_fit.py:236-277`, `src/boldtailor/bids_provenance.py:105,158-163,195-202,413-424,489-498`, `src/boldtailor/publication.py:80-85,113,175-209,241-245,277-291,305,343-391`
- Test: `tests/test_provenance.py`, `tests/test_publication.py`, `tests/test_bids_provenance.py`
- Docs: `docs/development.md:193-327`, `docs/superpowers/specs/2026-08-07-general-first-level-fmri-package-design.md` (mark superseded publication sections)

**Interfaces:**
- Removes: `_looks_path_like` value checks, `_freeze_key` path branch, `prepared_fit._validate_path_safe_mapping_keys`/`_is_path_like_key`, `_FrozenSequence`, `publication._validate_metadata` payload parsing, pre-lock `_preflight`, directory fsyncs, case-fold scanning.
- Changes: one `provenance.validate_relative_path` (POSIX, no traversal, components `[A-Za-z0-9+_.-]+`) used by both modules; `_validate_modified_at` accepts `Z` or `+00:00`; `publication` resolves the destination before symlink checks and rejects symlinks only inside it; control data moves to a sibling `<destination>.boldtailor/`; BEP028 `Command` = `"boldtailor.<entry point>"`, `StartedAtTime`/`EndedAtTime` from lifecycle events, `Environments[0]` populated from `activity["software"]`; `_safe_json` filtering removed (projection equals record).

- [ ] **Step 1: Failing tests** (one per behaviour change):

```python
def test_annotations_may_contain_path_like_text():
    ref = SourceRef(role="signal", annotations={"note": "~5 mm smoothing", "see": "./README.md"})
    assert ref.annotations["note"] == "~5 mm smoothing"


@pytest.mark.parametrize("stamp", ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00+00:00"])
def test_modified_at_accepts_utc_forms(stamp):
    assert SourceRef(role="signal", modified_at=stamp).modified_at == stamp


def test_relative_path_rule_is_shared(complete_sources):
    with pytest.raises(ValueError, match="relative path"):
        SourceRef(role="signal", uri="sub-01/func/my file.tsv")


def test_publication_accepts_destination_under_symlinked_tmp(tmp_path):
    destination = Path("/tmp") / f"boldtailor-{uuid4()}"
    try:
        publish_artifact_set(destination, [Artifact("a.json", b"{}")])
        assert (destination / "a.json").exists()
    finally:
        shutil.rmtree(destination, ignore_errors=True)
        shutil.rmtree(str(destination) + ".boldtailor", ignore_errors=True)


def test_publication_does_not_parse_payloads(tmp_path):
    publish_artifact_set(tmp_path / "out", [Artifact("broken.json", b"{not json"), Artifact("t.tsv", b"a\tb\n1\n")])
    assert (tmp_path / "out" / "broken.json").read_bytes() == b"{not json"


def test_bids_activity_has_command_and_timestamps(record_from_fit):
    activity = project_bids_provenance(record_from_fit)["act"]["Activities"][-1]
    assert activity["Command"] == "boldtailor.fit"
    assert activity["StartedAtTime"] <= activity["EndedAtTime"]
    env = project_bids_provenance(record_from_fit)["env"]["Environments"][0]
    assert env["Python"] == software_environment()["python"]
```

- [ ] **Step 2:** Run → FAIL; commit tests. Delete the tests that pinned the removed behaviours (payload validation, case-fold collisions, `/tmp` rejection, path-like annotation rejection, digest stripping) as requirement changes.
- [ ] **Step 3: Implement** module by module (provenance → prepared_fit → bids_provenance → publication), running `uv run pytest -q tests/test_<module>.py` after each. Publication target: lock → stage in `<destination>.boldtailor/stage-<uuid>/` on the same filesystem → `os.replace` per file → fsync files → rollback by restoring backups on failure → failure ledger in `<destination>.boldtailor/failures.jsonl`.
- [ ] **Step 4:** Docs: rewrite `development.md:193-327` to the new contract in ≤ 40 lines with the positive threat model: "cooperating writers on one host, no adversary, no power-loss guarantee". Mark the superseded spec sections.
- [ ] **Step 5:** `uv run pytest -q` → pass; `wc -l src/boldtailor/{provenance,publication,bids_provenance}.py` ≤ 1,000 total. Commit `refactor: right-size provenance and publication; BEP028 command, timestamps, environment`.

### Task 4.6: Promote shared example helpers into the package; one notebook stack (E3, E6)

**Files:**
- Create: `src/boldtailor/cifti.py` (from `examples/NSD/single_trial_artifacts.scalar_artifact`, `workflow_surfaces.cortical_values`, the CIFTI axis checks in `workflow_reuse.read_image`), `src/boldtailor/reliability.py` (from `hrf_reliability.hrf_curve_correlations` + `session_hrf_reliability.compare_hrfs`, one implementation), `src/boldtailor/parallel.py` (from `parallel_blocks.map_blocks`), `src/boldtailor/diagnostics.py` (`beta_activation.one_sample_t`, `rt_diagnostics.correlate_rt`)
- Modify: `src/boldtailor/_task_design.py`, `_hrf_cv.py`, `_ridge_cv.py`, `_fractional_ridge.py`, `_single_trial_fit.py` (public re-exports: `expand_events`, `prepare_runs`, `subset_runs`, `fraction_grid`, `regularization`, `NORM_BASIS`, `r_squared` via `boldtailor.design`, `boldtailor.hrf_selection`, `boldtailor.fractional_ridge`, `boldtailor.single_trial` respectively)
- Create: `examples/NSD/settings.py` with `resolve_settings(config: Mapping) -> dict` extracted from notebook cells `de5dc917`/`f3d49ae3`
- Delete: `examples/NSD/nsd_cifti.py`, `nsd_single_trial.py`, `nsd_hrf.py`, `hrf_artifacts.py`, `single_trial_artifacts.py` and their tests once the notebook stack covers their outputs (confirm with `grep -rn "nsd_cifti\|nsd_hrf\b" examples docs README.md` and update the README command table)
- Test: move the kept example tests to use `examples/NSD/conftest.py` fixtures (from `test_nsd_cifti` and `test_ridge_workflow`); replace cell-by-id `exec()` tests with `resolve_settings` unit tests; keep one `@pytest.mark.notebook` smoke test per notebook.

**Interfaces:**
- Produces: `boldtailor.cifti.scalar_artifact(values, brain_axis, names) -> Artifact`, `boldtailor.cifti.cortical_values(image, hemisphere) -> np.ndarray`; `boldtailor.reliability.curve_correlations(curves_a, curves_b, ids_a, ids_b) -> np.ndarray`; `boldtailor.parallel.map_blocks(fn, n_features, block_size, n_jobs) -> list`; `boldtailor.diagnostics.one_sample_t(betas) -> (t, p, dof)`, `boldtailor.diagnostics.correlate_rt(betas, rt, runs) -> np.ndarray`; `examples.NSD.settings.resolve_settings`.

- [ ] **Step 1:** For each new module, copy the example function and its existing oracle test (e.g. `examples/NSD/test_beta_activation.py:17,35,55`, `test_rt_diagnostics.py:14,51`, `test_hrf_reliability.py:26`, `test_nsd_parallel.py:38`) into `tests/test_<module>.py`; run → FAIL on import; commit; add the module; run → PASS; commit per module.
- [ ] **Step 2:** Replace the 14 private imports in `examples/NSD` with the public names; `grep -rn "boldtailor\._" examples` → empty.
- [ ] **Step 3:** Extract `resolve_settings`; write `examples/NSD/test_settings.py` with the cases the cell-exec tests covered (ridge mode/alpha/fraction resolution, sobol/expanded/custom library, path expansion); delete the cell-exec tests; make the notebooks call `resolve_settings`.
- [ ] **Step 4:** Collapse to one smoke execution per notebook; delete the CLI stack per Files once `uv run pytest -q examples/NSD --run-notebooks` passes.
- [ ] **Step 5:** README command table and `examples/NSD/README.md` updated; `nsd_cifti.py:26` personal path gone with the module. Commit `refactor: shared imaging, reliability, and parallel helpers live in the package; one NSD stack`.

### Task 4.7: Ownership idiom, result construction, API naming (L4, L7, L8, L9)

**Files:**
- Modify: all result dataclasses (`results.py`, `hrf_results.py`, `ridge_results.py`, `single_trial_results.py`, `hrf_glm_results.py`): add `kw_only=True`; validate in `__post_init__`, own arrays with `readonly_array`, no positional construction anywhere (`grep -n "Result(" src | grep -v "=" ` → empty)
- Modify: `src/boldtailor/single_trial.py` (`fit_selected_hrfs(selection=…)` → `hrf_selection=…`; `fit_single_trials(hrf=…)` → `hrf_model=…`; keep old names one release as deprecated aliases emitting `DeprecationWarning`), `src/boldtailor/hrf_selection.py` (`select_hrf` → `select_hrfs`, alias kept; `evaluate_hrf_split(..., candidate_batch_size=32)` threaded), `src/boldtailor/prepared_fit.py:77-94` (`task_delta_r2_prepared(prepared, full_result)` derives contrasts/noise model/metadata from `full_result.provenance.activities[-1]["model"]`), `src/boldtailor/ridge_selection.py`/`fractional_ridge.py` (accept a `CandidateScores` positionally: `select_ridge_penalty(scores, *, percentile=90.0, feature_mask=None)` with the array+grid form still accepted)
- Modify: `src/boldtailor/_selected_hrf_fit.py:45-78` (return a `GroupRunFit` dataclass), `_single_trial_design.py` (one `trial_table(events_per_run, labels)`), `_hrf_design.py` (own `OVERSAMPLING = 50`, `MIN_ONSET = -24.0`, `TIE_TOLERANCE = 1e-12`; import everywhere), `model.py:221-230` (`is_boolean/is_integer/is_real` public and used by the 12 call sites)
- Test: `tests/test_result_schemas.py` (one ownership test per public result class, `kw_only` enforcement), `tests/test_single_trial.py`, `tests/test_prepared_fit.py`, `tests/test_hrf_selection.py`

- [ ] **Step 1:** Failing tests: `TypeError` on positional construction of each result class; `DeprecationWarning` from `fit_selected_hrfs(selection=...)`; `task_delta_r2_prepared(prepared, result)` equals the current three-keyword call; `select_ridge_penalty(scores)` equals `select_ridge_penalty(scores.cv_r2, scores.grid)`; `evaluate_hrf_split(..., candidate_batch_size=7)` gives identical indices to the default.
- [ ] **Step 2:** Commit tests; implement in the order Files lists; `uv run pytest -q` after each file.
- [ ] **Step 3:** Update `docs/api.md` names and `docs/user-guide.md` examples. Commit `refactor: keyword-only owned results; coherent entry-point names; shared constants`.

---

# Phase 5: Hygiene

### Task 5.1: Small code cleanups (N3, N6, review §2.3 small items)

**Files:**
- Modify: `src/boldtailor/_fractional_ridge.py:100-115` (single SVD: derive the normalized rank check from `s` and column norms of the raw residualized design; delete the `_project_design` call), `src/boldtailor/_single_trial_design.py:68-69` (delete `_trial_column`), move `trial_beta_path`/`fraction_beta_path` to `tests/oracles.py`, `src/boldtailor/model.py:193` ("noise_model must be 'ols' or 'ar1'"), `model.py:230` (`isinstance(value, Integral)`), `src/boldtailor/design.py:183-185` (report non-numeric confound columns by name before the finiteness check), `src/boldtailor/_hrf_glm.py:42-46` (message names the required values −24.0 and 50), `src/boldtailor/_ridge_cv.py:161-167` (compute `predictor_means` from the first fit, not the loop variable), `src/boldtailor/_hrf_design.py:7` (comment naming the relied-on `_sample_condition` behaviour and the `<0.15` pin)
- Test: `tests/test_fractional_ridge.py` (the brentq oracle already covers the solver; add `test_prepare_fraction_betas_factorizes_once` counting `np.linalg.svd` calls == 2 (nuisance + raw)), `tests/test_model.py` (`drift_order=np.int64(2)` accepted), `tests/test_design.py` (non-numeric confound error names the column)

- [ ] **Step 1:** Failing tests; commit. **Step 2:** Implement; `uv run pytest -q` → pass. **Step 3:** Commit `chore: single SVD in fractional prepare; dead code; validator consistency`.

### Task 5.2: Repository hygiene

- [ ] **Step 1:** Delete merged/stale branches after confirming each is contained in `main`: `for b in $(git branch --merged main | grep -v main); do git branch -d $b; done`; list the unmerged remainder (`git branch --no-merged main`) in `docs/development.md` with a one-line purpose each or delete with `-D` after the user confirms.
- [ ] **Step 2:** `docs/development.md`: note that `uv.lock` pins the yanked nilearn 0.14.0 and the private `_sample_condition` import; record the decision to stay on `<0.15` until the fast convolution path is re-verified against the next release.
- [ ] **Step 3:** `git stash list` still holds the executed notebooks from Task 1.3; drop it (`git stash drop`) once the user confirms the outputs are not needed.
- [ ] **Step 4:** Commit `docs: dependency notes and branch inventory`.

---

## Self-review

**Spec coverage.** User requirement (peak-normalized HRFs)→1.5; S1→1.1; S2→3.4; S3→3.2; S4/E1/E4→1.3; S5→1.2; S6/V1→2.1–2.3; P1→3.3 (evidence) with the estimand decision deferred to the recorded outcome; P2→3.4; P3→2.2 (messages + digest); P4 (per-fold amplitude spread) → not scheduled; add as follow-up if 3.3 passes; P5→5.1; P6→5.2; N1/L3→4.4; N2 (global caches) → folded into 4.4's `_rebuild` closure (explicit cache object; `_cached_design` lru removed there); N3/N6→5.1; N4 (`_score_fold` length) → 4.4 hoisting and 5.1 fix; N5→5.1; L1→4.1; L2→2.4; L4/L7/L8/L9→4.7; L5→4.3; L6→4.2; V2–V7→4.5; E2/E3/E6→4.6; E5 (curve tail inflation) → 4.6 moves the function; add a `support_seconds` argument there; T1→3.3; T2→3.1; T3/T4→0.2–0.5; T5→0.1/1.4; T6→3.4.

**Placeholder scan.** Every code step contains code; fixture names that must match existing modules are called out as such (`single_run_problem`, `cv_fixture`, `selected_fixture`, `ridge_problem` exist in `tests/conftest.py` and `tests/test_hrf_selection.py`; `prepared_problem`, `three_run_problem`, `two_session_records`, `saved_metadata`, `record_from_fit` are to be built from the nearest existing fixture in the named file).

**Type consistency.** `software_environment()` (2.1) is what 4.5 reads into `Environments`; `nested_ols_delta`/`delta_r2_activity` (4.1) are the only ΔR² builders; `run_labels_for` moves to `data.py` (4.2) and is imported by `hrf_selection`, `single_trial`, `_selected_hrf_fit`, `_ridge_cv`; `SelectedTrialDesign.matrix(run, hrf_id)` (4.4) replaces `.matrices` everywhere including `examples/NSD/workflow_analysis.py`.

**Review Focus coverage.** 1→1.1 (`test_feature_constant_in_one_run_has_undefined_contrasts`); 2→1.3 (`test_saved_results_with_other_task_model_are_rejected`); 3→2.2 (`test_sha256_changes_the_metadata_fingerprint`); 4→1.2 (`test_frame_times_start_at_sidecar_start_time`); 5→3.2 (`test_fraction_selection_flags_grid_endpoints`, `test_selection_flags_features_at_the_parameter_box_edge`).
