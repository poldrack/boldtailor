# Prepared-Design Stop-Signal Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the real-data stop-signal notebook perform its sole whole-brain fit through Boldtailor's prepared-design boundary while preserving its plots, derivatives, privacy, and event-level numerical results.

**Architecture:** The notebook continues to load BIDS/fMRIPrep data and normalize events/confounds, then uses the public design compiler to create full and nuisance-only matrices. A tested example helper resolves concrete column roles from those matrices; the notebook constructs `PreparedDesignAnalysis`, calls `fit_prepared()`, and computes nested-OLS task variance with `task_delta_r2_prepared()` without running the event-level estimator. Automated fixture instrumentation performs the expensive parity comparison outside the documentation notebook.

**Tech Stack:** Python 3.12, NumPy, pandas, Nilearn, nibabel, nbformat/nbclient, pytest, uv, Git.

## Global Constraints

- Run every local Python, test, formatting, notebook, and Git command through `uv run`; use `uv run --no-project` for Git commands.
- Follow strict RED-GREEN-Refactor and commit failing tests before production or notebook changes.
- Keep every `src/boldtailor/__init__.py` completely empty and do not change package exports.
- Do not change Boldtailor numerical APIs, dependency declarations, or `uv.lock`.
- Use only public `boldtailor.design`, `boldtailor.prepared`, and `boldtailor.prepared_fit` interfaces from the notebook.
- Perform only one real-data whole-brain fit in the notebook. Event-level parity belongs in the bounded automated fixture.
- Derive roles structurally from full versus nuisance-only compiler output; do not guess drift/confound roles from name prefixes.
- Map `constant` to `intercept`, other nuisance-design columns to `nuisance`, and full-only columns to `task`. Do not emit `other`.
- Preserve the common mask, five plot calls, shared slice coordinates `[-10, 5, 20, 35, 50, 65]`, 11 NIfTI images, derivative filenames, compact displays, and no-raw-signal-display contract.
- Preserve AR(1) contrast inference and the separate two-fit nested-OLS task delta R-squared diagnostic with a `-1e-12` monotonicity tolerance.
- Preserve relative/BIDS source identifiers and prevent raw signals, design values, or absolute dataset paths from entering displays or published provenance.
- Clear stale stored notebook outputs, but do not add any test or repository rule requiring notebooks to remain output-free.
- Leave generated cache and egg-info directories untouched.

## File Structure

- Modify `examples/stop_signal_demo.py`: add a small public example helper that resolves and validates per-run prepared column roles.
- Modify `tests/test_stop_signal_demo.py`: specify the helper, notebook source boundary, runtime prepared inputs, numerical parity, provenance, plotting, publication, privacy, and output-tolerant behavior.
- Modify `examples/stop_signal_demo.ipynb`: compile fixed designs, construct the prepared input, fit once through the prepared API, compute prepared task delta R-squared, update narrative/metadata, and clear stale outputs.
- Do not modify library source under `src/boldtailor/`.

---

### Task 1: Resolve prepared column roles from compiled designs

**Files:**
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.py`

**Interfaces:**
- Consumes: `Sequence[pd.DataFrame]` full designs, nuisance-only designs, and run labels.
- Produces: `prepared_column_roles(full_designs, nuisance_designs, run_labels) -> tuple[dict[str, str], ...]`.

- [ ] **Step 1: Write the role-resolution tests**

Add focused function tests without importing a nonexistent symbol at collection time:

```python
def test_prepared_column_roles_follow_nuisance_compiler():
    prepared_column_roles = _demo_module().prepared_column_roles
    full = (
        pd.DataFrame(
            columns=("go_success", "motion", "drift_1", "constant")
        ),
        pd.DataFrame(
            columns=("constant", "drift_1", "motion", "stop_success")
        ),
    )
    nuisance = (
        pd.DataFrame(columns=("motion", "drift_1", "constant")),
        pd.DataFrame(columns=("constant", "drift_1", "motion")),
    )

    roles = prepared_column_roles(full, nuisance, ("ses-02", "ses-04"))

    assert roles == (
        {
            "go_success": "task",
            "motion": "nuisance",
            "drift_1": "nuisance",
            "constant": "intercept",
        },
        {
            "constant": "intercept",
            "drift_1": "nuisance",
            "motion": "nuisance",
            "stop_success": "task",
        },
    )
```

Add separate tests requiring run-specific errors for:

```python
bad_nuisance = (
    nuisance[0],
    pd.DataFrame(columns=("constant", "missing_motion")),
)
with pytest.raises(ValueError, match="ses-04.*nuisance.*absent from full"):
    prepared_column_roles(full, bad_nuisance, ("ses-02", "ses-04"))

no_task_full = (pd.DataFrame(columns=("motion", "constant")),)
no_task_nuisance = (pd.DataFrame(columns=("motion", "constant")),)
with pytest.raises(ValueError, match="ses-02.*at least one task column"):
    prepared_column_roles(no_task_full, no_task_nuisance, ("ses-02",))

with pytest.raises(ValueError, match="same number of runs"):
    prepared_column_roles(full, nuisance[:1], ("ses-02", "ses-04"))
```

Also assert every returned mapping follows full-design column order, contains
exactly one entry per full column, and contains no `other` value.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest tests/test_stop_signal_demo.py -k prepared_column_roles -q -W error -p no:cacheprovider
```

Expected: failure because `examples.stop_signal_demo` has no
`prepared_column_roles` attribute. Fix any fixture or assertion error until the
failure is specifically the missing helper.

- [ ] **Step 3: Commit the RED tests**

```bash
uv run --no-project git add tests/test_stop_signal_demo.py
uv run --no-project git commit -m "test: specify prepared notebook role mapping"
```

- [ ] **Step 4: Implement the minimal role resolver**

Add short functions to `examples/stop_signal_demo.py`:

```python
def prepared_column_roles(
    full_designs: Sequence[pd.DataFrame],
    nuisance_designs: Sequence[pd.DataFrame],
    run_labels: Sequence[str],
) -> tuple[dict[str, str], ...]:
    full = tuple(full_designs)
    nuisance = tuple(nuisance_designs)
    labels = tuple(run_labels)
    if len(full) != len(nuisance) or len(full) != len(labels):
        raise ValueError("full, nuisance, and run labels must have the same number of runs")
    return tuple(
        _prepared_run_column_roles(full_run, nuisance_run, label)
        for full_run, nuisance_run, label in zip(
            full, nuisance, labels, strict=True
        )
    )


def _prepared_run_column_roles(
    full: pd.DataFrame,
    nuisance: pd.DataFrame,
    run_label: str,
) -> dict[str, str]:
    missing = [name for name in nuisance if name not in full]
    if missing:
        raise ValueError(
            f"{run_label} nuisance design columns absent from full design: "
            + ", ".join(missing)
        )
    nuisance_names = set(nuisance)
    roles = {
        name: (
            "intercept"
            if name == "constant"
            else "nuisance"
            if name in nuisance_names
            else "task"
        )
        for name in full
    }
    if not any(role == "task" for role in roles.values()):
        raise ValueError(f"{run_label} requires at least one task column")
    if not any(role in {"nuisance", "intercept"} for role in roles.values()):
        raise ValueError(f"{run_label} requires a nuisance or intercept column")
    return roles
```

Keep both functions short. Do not add library code or package exports.

- [ ] **Step 5: Verify GREEN and commit production**

Run:

```bash
uv run pytest tests/test_stop_signal_demo.py -k prepared_column_roles -q -W error -p no:cacheprovider
uv run black --check examples/stop_signal_demo.py tests/test_stop_signal_demo.py
uv run --no-project git diff --check
```

Commit only the helper:

```bash
uv run --no-project git add examples/stop_signal_demo.py
uv run --no-project git commit -m "feat: classify prepared notebook columns"
```

---

### Task 2: Make the prepared path the notebook's sole estimator

**Files:**
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.ipynb`

**Interfaces:**
- Consumes: `AnalysisData`, `ModelSpec`, `compile_designs()`,
  `compile_nuisance_designs()`, `prepared_column_roles()`, and the existing
  loaded runs/sources.
- Produces: notebook globals `compiled_designs`, `compiled_nuisance_designs`,
  `column_roles`, `run_metadata`, `prepared_analysis`, `result`, and
  `task_delta`, with `result` from `fit_prepared()` and `task_delta` from
  `task_delta_r2_prepared()`.

- [ ] **Step 1: Write the static prepared-boundary test**

Parse all notebook code-cell sources and require the public architecture:

```python
def test_notebook_uses_prepared_design_estimation_boundary():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    tree = ast.parse("\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    ))
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert "design.compile_designs" in calls
    assert "design.compile_nuisance_designs" in calls
    assert "PreparedDesignAnalysis.from_arrays" in calls
    assert "prepared_fit.fit_prepared" in calls
    assert "prepared_fit.task_delta_r2_prepared" in calls
    assert "fit.fit" not in calls
    assert "fit.task_delta_r2" not in calls
```

Also inspect imports and require direct module imports rather than package-root
exports:

```python
assert "import boldtailor.design as design" in source
assert "from boldtailor.prepared import PreparedDesignAnalysis" in source
assert "import boldtailor.prepared_fit as prepared_fit" in source
```

- [ ] **Step 2: Add a prepared runtime/parity audit to the fixture**

Define `PREPARED_AUDIT_PREFIX` and append an instrumentation cell after the
notebook's analysis cell. The injected cell may compute an event-level fit
because fixture arrays are small; the real notebook must not do so.

The audit must independently run:

```python
_event_result = fit(analysis_data, model_spec)
_event_delta = task_delta_r2(analysis_data, model_spec, _event_result)
_contrast_parity = {
    name: {
        field: bool(np.allclose(
            getattr(result, field)(name),
            getattr(_event_result, field)(name),
            equal_nan=True,
        ))
        for field in ("effect", "variance", "statistic", "z_score", "p_value")
    }
    for name in result.contrast_names
}
```

Print one JSON audit containing:

```python
{
    "n_runs": prepared_analysis.n_runs,
    "run_metadata": list(prepared_analysis.run_metadata),
    "roles": [dict(item) for item in prepared_analysis.column_roles],
    "design_fingerprint": prepared_analysis.design_fingerprint,
    "run_design_fingerprints": list(prepared_analysis.run_design_fingerprints),
    "activity_names": [item["name"] for item in task_delta.provenance.activities],
    "contrast_parity": _contrast_parity,
    "aggregate_r2_parity": bool(np.allclose(result.r2, _event_result.r2)),
    "delta_r2_parity": bool(np.allclose(task_delta.delta_r2, _event_delta.delta_r2)),
}
```

Assert both CWD executions report two fixture runs, complete subject/session/task/run
metadata, nonempty 64-character hexadecimal fingerprints, no `other` role,
all parity booleans true, and the ordered activities
`normalize_prepared_design`, `fit_prepared`, and `task_delta_r2_prepared`.

- [ ] **Step 3: Update publication and disclosure expectations before implementation**

Change expected provenance names and metadata to the prepared contract. Require:

```python
normalized["name"] == "normalize_prepared_design"
normalized["run_metadata"] == [
    {"subject": "s4", "session": "02", "task": "stopSignal", "run": "01"},
    {"subject": "s4", "session": "04", "task": "stopSignal", "run": "01"},
]
normalized["metadata"] == {
    "boundary": "prepared_design",
    "design_compiler": "boldtailor.design.compile_designs",
    "design_source": "events_and_confounds",
}
fitted["name"] == "fit_prepared"
fitted["model"]["noise_model"] == "ar1"
provenance["activities"][-1]["name"] == "task_delta_r2_prepared"
```

Require the rendered narrative to state that the notebook compiler stands in
for PyBIDS/FitLins, prepared fitting performs no image/file I/O, and task delta
R-squared uses separate nested OLS fits. Preserve all existing image, slice,
compact-display, privacy, and artifact assertions.

Do not add an assertion that notebook outputs are empty. Add a structural test
that strips every `outputs` and `execution_count` field in an in-memory copy and
confirms the source-boundary assertions still operate on source alone; this
proves tests tolerate both executed and unexecuted notebooks.

- [ ] **Step 4: Run the notebook tests and confirm RED**

Run:

```bash
uv run pytest tests/test_stop_signal_demo.py -k "prepared_design_estimation_boundary or prepared_runtime or variance_partition_notebook_executes or publishes_complete_private_metadata" -q -W error -p no:cacheprovider
```

Expected failures must show that the notebook still calls event-level
`fit.fit()`/`fit.task_delta_r2()`, lacks prepared globals/audit fields, and
publishes the old activity names. Do not change expectations to accommodate
the old workflow.

- [ ] **Step 5: Commit the complete RED notebook contract**

```bash
uv run --no-project git add tests/test_stop_signal_demo.py
uv run --no-project git commit -m "test: specify prepared stop signal workflow"
```

- [ ] **Step 6: Change notebook imports and narrative**

In the setup cell, replace `import boldtailor.fit as fit` with:

```python
import boldtailor.design as design
import boldtailor.prepared_fit as prepared_fit
from boldtailor.prepared import PreparedDesignAnalysis
```

Import `prepared_column_roles` from `examples.stop_signal_demo`. Update the
markdown immediately before design fitting to explain the local compiler and
the FitLins/PyBIDS handoff boundary.

- [ ] **Step 7: Compile both designs and materialize prepared inputs**

Replace the event-level fit portion of the `design-fit` cell with:

```python
compiled_designs = design.compile_designs(analysis_data, model_spec)
compiled_nuisance_designs = design.compile_nuisance_designs(
    analysis_data, model_spec
)
full_designs = tuple(item.matrix for item in compiled_designs)
nuisance_designs = tuple(item.matrix for item in compiled_nuisance_designs)
column_roles = prepared_column_roles(full_designs, nuisance_designs, SESSIONS)
run_metadata = tuple(
    {
        "subject": SUBJECT.removeprefix("sub-"),
        "session": session.removeprefix("ses-"),
        "task": TASK,
        "run": RUN.removeprefix("run-"),
    }
    for session in SESSIONS
)
prepared_analysis = PreparedDesignAnalysis.from_arrays(
    signals=analysis_data.signals,
    design_matrices=full_designs,
    frame_times=analysis_data.frame_times,
    column_roles=column_roles,
    sources=analysis_data.sources,
    run_metadata=run_metadata,
    provenance_metadata={
        "boundary": "prepared_design",
        "design_compiler": "boldtailor.design.compile_designs",
        "design_source": "events_and_confounds",
    },
)
```

Keep `ModelSpec` as the single description of contrasts, selected confounds,
HRF, drift, and inferential noise model.

- [ ] **Step 8: Fit once and compute prepared task delta R-squared**

Use:

```python
prepared_model_metadata = {
    "hrf_model": model_spec.hrf_model,
    "drift_model": model_spec.drift_model,
    "high_pass": model_spec.high_pass,
    "drift_order": model_spec.drift_order,
    "oversampling": model_spec.oversampling,
    "min_onset": model_spec.min_onset,
    "confounds": list(model_spec.confounds),
}
result = prepared_fit.fit_prepared(
    prepared_analysis,
    contrasts=model_spec.contrasts,
    noise_model=model_spec.noise_model,
    model_metadata=prepared_model_metadata,
)
task_delta = prepared_fit.task_delta_r2_prepared(
    prepared_analysis,
    result,
    contrasts=model_spec.contrasts,
    noise_model=model_spec.noise_model,
    model_metadata=prepared_model_metadata,
)
```

Build design displays from `result.design_matrices` as before. Build the
nuisance-design summary from `task_delta.nuisance_design_matrices`. Add a
compact role-count/fingerprint display without displaying design values or raw
signals.

- [ ] **Step 9: Update shareable metadata and narrative**

Preserve the existing variance-partition fields and add a compact
`prepared_design` section to the shareable configuration:

```python
"prepared_design": {
    "boundary": "fixed labeled design matrices",
    "design_compiler": "boldtailor.design.compile_designs",
    "fit_entry_point": "boldtailor.prepared_fit.fit_prepared",
    "task_delta_entry_point": "boldtailor.prepared_fit.task_delta_r2_prepared",
    "design_fingerprint": prepared_analysis.design_fingerprint,
    "run_design_fingerprints": list(prepared_analysis.run_design_fingerprints),
    "column_role_counts": [
        {
            role: sum(value == role for value in roles.values())
            for role in ("task", "nuisance", "intercept", "other")
        }
        for roles in prepared_analysis.column_roles
    ],
}
```

State explicitly that AR(1) governs contrast inference, task delta R-squared
refits both nested models with OLS, and the prepared estimator itself performs
no image/file I/O.

- [ ] **Step 10: Clear stale outputs mechanically**

Run:

```bash
uv run jupyter nbconvert --clear-output --inplace examples/stop_signal_demo.ipynb
```

Confirm source cells and stable cell IDs remain intact and every stored
`execution_count` is `null`. Do not add an output-free repository test.

- [ ] **Step 11: Run focused GREEN verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_stop_signal_demo.py -q -W error -p no:cacheprovider
uv run black --check examples/stop_signal_demo.py tests/test_stop_signal_demo.py
uv run --no-project git diff --check
```

Expected: all notebook tests pass, including both working directories and the
independent event/prepared parity audit.

- [ ] **Step 12: Commit the notebook implementation**

```bash
uv run --no-project git add examples/stop_signal_demo.ipynb
uv run --no-project git commit -m "docs: demonstrate prepared stop signal fitting"
```

---

### Task 3: Final documentation verification

**Files:**
- Verify: `examples/stop_signal_demo.ipynb`
- Verify: `examples/stop_signal_demo.py`
- Verify: `tests/test_stop_signal_demo.py`
- Verify: `README.md`

**Interfaces:**
- Consumes: the completed notebook and existing real-data fixture configuration.
- Produces: a verified documentation branch with no further production changes.

- [ ] **Step 1: Verify the entire repository**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -W error -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 uv run black --check src tests examples/stop_signal_demo.py
uv lock --check
uv run --no-project git diff --check main..HEAD
test ! -s src/boldtailor/__init__.py
uv run --no-project git diff --exit-code main..HEAD -- pyproject.toml uv.lock src/boldtailor/__init__.py
```

Expected: full suite green, Black clean, lock current, no whitespace errors,
empty initializer, and no dependency/lock change.

- [ ] **Step 2: Verify the notebook remains output-tolerant**

Use `uv run python -c` with `nbformat` to assert that all cells have stable
nonempty IDs and that source parsing succeeds. Do not assert that outputs are
empty. Confirm the committed notebook currently has cleared stale outputs only
as a one-time documentation cleanup.

- [ ] **Step 3: Perform a bounded real-data execution check**

Execute the notebook in memory with:

```bash
VERIFY_ROOT=$(mktemp -d /private/tmp/boldtailor-prepared-notebook.XXXXXX)
mkdir -p "$VERIFY_ROOT/jupyter" "$VERIFY_ROOT/ipython" "$VERIFY_ROOT/publication"
BOLDTAILOR_BIDS_ROOT=/Users/poldrack/data_unsynced/rdoc_fmri \
BOLDTAILOR_SESSIONS=ses-06,ses-08 \
BOLDTAILOR_TEMP_ROOT="$VERIFY_ROOT/publication" \
MPLBACKEND=Agg \
JUPYTER_CONFIG_DIR="$VERIFY_ROOT/jupyter" \
IPYTHONDIR="$VERIFY_ROOT/ipython" \
uv run python -c '
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path("examples/stop_signal_demo.ipynb").resolve()
notebook = nbformat.read(path, as_version=4)
executed = NotebookClient(
    notebook,
    timeout=7200,
    kernel_name="python3",
    resources={"metadata": {"path": str(path.parent)}},
).execute()
assert not any(
    output.get("output_type") == "error"
    for cell in executed.cells
    for output in cell.get("outputs", ())
)
rendered = "\n".join(
    str(output.get("text", output.get("data", {}).get("text/plain", "")))
    for cell in executed.cells
    for output in cell.get("outputs", ())
)
for phrase in (
    "fit_prepared",
    "task_delta_r2_prepared",
    "AR(1) inference",
    "OLS diagnostic",
):
    assert phrase in rendered
publications = tuple(
    Path(os.environ["BOLDTAILOR_TEMP_ROOT"]).glob("boldtailor-*")
)
assert len(publications) == 1
assert len(tuple(publications[0].glob("images/*.nii.gz"))) == 11
'
```

The script must read the notebook with `nbformat`, execute it from the
`examples/` directory with a timeout of 7200 seconds, keep executed outputs in
memory, assert no error outputs, assert the rendered text names
`fit_prepared`, `task_delta_r2_prepared`, AR(1) inference, and the nested-OLS
diagnostic, and assert exactly one temporary publication directory with 11
NIfTI images. Poll any long-running command at intervals under 60 seconds.

Before and after execution, assert that
`/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor` does not exist.
Do not write the executed notebook back to the repository or dataset.

- [ ] **Step 4: Record the verification handoff**

Write an uncommitted report under
`.superpowers/sdd/2026-08-11-prepared-notebook-workflow/final-report.md` with
exact commit IDs, focused/full test counts, real-data execution result,
format/lock/diff results, and final worktree status. Do not commit generated
cache, egg-info, temporary publications, or the report.
