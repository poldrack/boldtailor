# BIDS CLI / NSD Experiments Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The `boldtailor run` CLI stays purely BIDS + fMRIPrep. The NSD experiments read only nsdpreproc CIFTI files and use only boldtailor's public core API.

**Architecture:**
- Two generic helpers move from `boldtailor.workflow` into public core modules: task-model detection and trial predictors. The workflow and the experiments then both call them from core.
- The experiment loader switches from the hand-built ppdata layout to BIDS-style `derivatives/nsdpreproc` paths, and the `ppdata` source label becomes `nsdpreproc`.
- The `examples/NSD` notebooks become `examples/bids_workflow`.
- A static boundary test keeps `experiments/` away from workflow and CLI imports and from fMRIPrep CIFTI file names.

**Tech Stack:** Python 3.12, uv, pytest, pandas, nibabel, `ast`.

**Spec:** `docs/superpowers/specs/2026-10-08-bids-cli-nsd-separation-design.md`

## Global Constraints

- Use `uv run` for every command. Do not change dependencies.
- Every `__init__.py` stays empty.
- RED-GREEN-Refactor: commit failing tests before the implementation, in a separate commit.
- Never weaken a test to make it pass. Tests change only where this spec changes the requirement:
  - module locations;
  - `ValueError` in place of `InputError` from core detection;
  - the event-table signature of `trial_predictors`;
  - the `ppdata` → `nsdpreproc` rename.
- No compatibility re-exports from the old locations. Update every caller in `src/`, `tests/`, `examples/` and `experiments/`.
- The CLI's behaviour, flags and outputs do not change. Bad events still exit with code 2.
- NSD experiments use polynomial confounds only, raw BIDS events, and `onset_offset = 0.667`. They never read fMRIPrep CIFTI files (`*_space-fsLR_den-91k_bold.dtseries.nii`) or `derivatives/ppdata`. Other fMRIPrep derivatives are not banned.
- Dated records stay unchanged: `docs/validation/*`, `docs/review-*`, `docs/test-suite-review-*`, and earlier `docs/superpowers/` specs and plans.
- Format with `uv run black` on the touched paths before each commit.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Baseline (2026-10-09)

- `uv run pytest -q tests`: 1667 passed in about 155 s.
- `uv run pytest -q experiments/nsd_replication`: record its result before Task 1. It is opt-in; the default `testpaths` is `tests` only.

## Review Focus

1. A TOML with `ppdata_dir` set alongside `nsdpreproc_dir` should still be rejected with the rename message, not silently accepted. Pinned in Task 3.
2. Building `ExperimentConfig` directly, without `load_config`, should still get the default `nsdpreproc_dir` under `bids_dir`. Pinned in Task 3.
3. If a session has raw events but no nsdpreproc directory at all, the error should name the run and its expected path, not end in an opaque nibabel error. Pinned in Task 4.
4. If `trial_predictors` gets a `labels` list whose length doesn't match the tables, it should raise `ValueError`, not silently truncate. Pinned in Task 2.
5. The boundary scanner should catch every import form, not only `from boldtailor.workflow.x import y`: `import boldtailor.workflow.inputs as w`, `from boldtailor import workflow`, and `from boldtailor import cli`. Pinned in Task 6.

---

### Task 1: Core `boldtailor.task_detection`

**Files:**
- Create: `src/boldtailor/task_detection.py`
- Create: `tests/test_task_detection.py`
- Modify: `src/boldtailor/workflow/inputs.py` (delete lines 25–96: `TRIAL_TYPE_NOTE`, `_table_levels`, `observed_levels`, `_everywhere`, `_automatic`, `_resolved`, `detect_task_model`, `task_model_notes`; `load_session` at about line 222)
- Modify: `src/boldtailor/workflow/run.py:92-100` (`_session`)
- Modify (imports only): `tests/workflow/test_inputs.py`, `tests/workflow/test_beta_series.py`, `tests/workflow/test_analysis.py`, `tests/workflow/test_outputs.py`, `examples/NSD/session_hrf.py`, `examples/NSD/test_session_hrf.py`, `examples/NSD/test_within_run_encoding.py`, `experiments/nsd_replication/inputs.py`, `experiments/nsd_replication/conftest.py`
- Modify: `docs/api.md:770` (module column)

**Interfaces:**
- Produces:
  - `boldtailor.task_detection.detect_task_model(events_tables, modulators=None, labels=None) -> TaskModel`
  - `observed_levels(tables, column, labels=None) -> set[str]`
  - `task_model_notes(events_tables, modulators=None) -> list[str]`
  - `TRIAL_TYPE_NOTE: str`

  All of them raise plain `ValueError`.

- [ ] **Step 1: Write the failing core tests**

Create `tests/test_task_detection.py`. These tests move from `tests/workflow/test_inputs.py`. Their assertions are unchanged, except that errors are now `ValueError`.

```python
"""Task-model detection from event tables, independent of the BIDS workflow."""

import subprocess
import sys

import pandas as pd
import pytest

from boldtailor.model import Modulator, TaskModel
from boldtailor.task_detection import (
    TRIAL_TYPE_NOTE,
    detect_task_model,
    task_model_notes,
)

TT = Modulator("trial_type", kind="categorical", levels=("0", "1"))


@pytest.fixture
def events():
    return pd.DataFrame(
        {
            "onset": [8.0, 22.0, 38.0, 60.0, 90.0, 112.0],
            "duration": [3.0] * 6,
            "trial_type": [0, 1, 0, 1, 1, 0],
            "response_time": [0.5, 1.0, 2.5, 1.5, 3.0, 0.5],
        }
    )


def test_detection_uses_rt_and_categorical_trial_type(events):
    model = detect_task_model([events, events])
    assert model == TaskModel((Modulator("response_time", missing="indicator"), TT))
    assert model.regressor_names == ("task", "response_time", "trial_type[1]")
    without_rt = detect_task_model([events.drop(columns="response_time"), events])
    assert without_rt == TaskModel((TT,))
    assert detect_task_model([events[["onset", "duration"]]]) == TaskModel()


def test_string_and_mixed_dtype_trial_types_become_one_level_set(events):
    words = events.assign(trial_type=["face", "house", "scrambled"] * 2)
    model = detect_task_model([words, words])
    assert model.regressor_names[-2:] == ("trial_type[house]", "trial_type[scrambled]")
    as_text = events.assign(trial_type=events.trial_type.astype(str))
    assert detect_task_model([events, as_text]).regressor_names[-1] == "trial_type[1]"


def test_single_level_trial_type_is_left_out_with_a_note(events):
    flat = events.assign(trial_type="face")
    assert detect_task_model([flat]).regressor_names == ("task", "response_time")
    assert task_model_notes([flat]) == [TRIAL_TYPE_NOTE]


def test_explicit_categorical_resolves_levels_and_checks_reference(events):
    wanted = (Modulator("trial_type", kind="categorical", reference="1"),)
    assert detect_task_model([events, events], wanted).regressor_names == (
        "task",
        "trial_type[0]",
    )
    bad = (Modulator("trial_type", kind="categorical", reference="7"),)
    with pytest.raises(ValueError, match="trial_type.*7"):
        detect_task_model([events], bad)


def test_explicit_modulators_must_exist_in_every_run(events):
    wanted = (Modulator("stimulus_id"),)
    with pytest.raises(ValueError, match="run 2.*stimulus_id"):
        detect_task_model([events.assign(stimulus_id=1), events], wanted)
    assert detect_task_model([events.assign(stimulus_id=1)] * 2, wanted) == TaskModel(
        wanted
    )


def test_explicit_modulator_errors_name_the_labelled_run(events):
    wanted = (Modulator("stimulus_id"),)
    with pytest.raises(ValueError, match="run-02.*stimulus_id"):
        detect_task_model(
            [events.assign(stimulus_id=1), events], wanted, labels=["run-01", "run-02"]
        )


def test_detection_takes_any_multilevel_trial_type_without_a_note(events):
    for values in (["face", "house"] * 3, [0, 1, 2, 0, 1, 2]):
        tables = [events, events.assign(trial_type=values)]
        assert detect_task_model(tables).modulators[-1].kind == "categorical"
        assert task_model_notes(tables) == []


def test_detection_keeps_binary_trial_type_written_as_text(events):
    text = events.assign(trial_type=events.trial_type.astype(str))
    assert TT in detect_task_model([text]).modulators
    assert task_model_notes([text]) == []


def test_notes_are_only_for_automatic_detection(events):
    strings = events.assign(trial_type=["face", "house"] * 3)
    assert task_model_notes([strings], modulators=()) == []
    assert task_model_notes([strings], modulators=(Modulator("x"),)) == []


def test_no_modulators_means_a_task_only_model(events):
    assert detect_task_model([events], ()) == TaskModel()


def test_core_detection_does_not_import_the_workflow():
    code = (
        "import sys, boldtailor.task_detection; "
        "sys.exit(any(m.startswith('boldtailor.workflow') for m in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_task_detection.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'boldtailor.task_detection'`.

- [ ] **Step 3: Delete the moved tests from `tests/workflow/test_inputs.py` and commit RED**

These now live in `tests/test_task_detection.py`, so delete them from `tests/workflow/test_inputs.py`:
- `test_detection_uses_rt_and_categorical_trial_type`
- `test_string_and_mixed_dtype_trial_types_become_one_level_set`
- `test_single_level_trial_type_is_left_out_with_a_note`
- `test_explicit_categorical_resolves_levels_and_checks_reference`
- `test_explicit_modulators_must_exist_in_every_run`
- `test_detection_takes_any_multilevel_trial_type_without_a_note`
- `test_detection_keeps_binary_trial_type_written_as_text`
- `test_notes_are_only_for_automatic_detection`
- `test_explicit_modulator_errors_name_the_bids_run`

Keep every test that goes through `inputs.load_session`. They pin `InputError` and the BIDS run names at the workflow layer.

```bash
git add tests/test_task_detection.py tests/workflow/test_inputs.py
git commit -m "test: task-model detection is a public core module (RED)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Create `src/boldtailor/task_detection.py`**

```python
"""Detect a task model from event tables: response time and categorical trial type."""

from boldtailor.model import Modulator, TaskModel, level_name

TRIAL_TYPE_NOTE = "trial_type has fewer than two levels; not used as a modulator"


def _run_labels(tables, labels):
    return list(labels) if labels else [f"run {i}" for i in range(1, len(tables) + 1)]


def _table_levels(table, column, label):
    try:
        return {level_name(v) for v in table[column]}
    except ValueError as error:
        raise ValueError(f"{label}: {column!r}: {error}") from error


def observed_levels(tables, column, labels=None):
    """Canonical non-missing level names of ``column`` across every table."""
    names = set()
    for table, label in zip(tables, _run_labels(tables, labels), strict=True):
        names |= _table_levels(table, column, label)
    names.discard(None)
    return names


def _everywhere(column, tables):
    return bool(tables) and all(column in t.columns for t in tables)


def _automatic(tables, labels):
    modulators = []
    if _everywhere("response_time", tables):
        modulators.append(Modulator("response_time", missing="indicator"))
    if _everywhere("trial_type", tables):
        levels = observed_levels(tables, "trial_type", labels)
        if len(levels) >= 2:
            modulators.append(
                Modulator("trial_type", kind="categorical", levels=tuple(levels))
            )
    return tuple(modulators)


def _resolved(modulator, tables, labels):
    if modulator.resolved:
        return modulator
    try:
        return modulator.with_levels(observed_levels(tables, modulator.column, labels))
    except ValueError as error:
        raise ValueError(f"modulator {modulator.column!r}: {error}") from error


def _check_columns(tables, labels, modulators):
    for label, table in zip(labels, tables, strict=True):
        missing = [m.column for m in modulators if m.column not in table.columns]
        if missing:
            raise ValueError(f"{label} events lack modulator column(s) {missing}")


def detect_task_model(events_tables, modulators=None, labels=None):
    """The task model from event columns, or explicit modulators checked per run.

    Detection takes ``response_time`` and a categorical ``trial_type`` with at
    least two levels; unresolved categorical modulators take their levels from
    every run. ``labels`` name the runs in errors.
    """
    tables = list(events_tables)
    labels = _run_labels(tables, labels)
    if modulators is None:
        modulators = _automatic(tables, labels)
    _check_columns(tables, labels, modulators)
    return TaskModel(tuple(_resolved(m, tables, labels) for m in modulators))


def task_model_notes(events_tables, modulators=None):
    """Why automatic detection left out a column that every run has."""
    tables = list(events_tables)
    if modulators is None and _everywhere("trial_type", tables):
        if len(observed_levels(tables, "trial_type")) < 2:
            return [TRIAL_TYPE_NOTE]
    return []
```

- [ ] **Step 5: Point the workflow at core and keep `InputError` at the boundary**

In `src/boldtailor/workflow/inputs.py`:
- Delete the moved definitions (lines 25–96).
- Add `from boldtailor.task_detection import detect_task_model`.
- Drop `level_name` (and `Modulator`, if `grep -n "Modulator\b" src/boldtailor/workflow/inputs.py` finds no other use) from the `boldtailor.model` import.
- Add this helper above `load_session`:

```python
def _session_task_model(raw_runs, modulators):
    """Detect the model; event problems become InputError (CLI exit code 2)."""
    try:
        return detect_task_model(
            [r.events for r in raw_runs],
            modulators,
            labels=[r.label for r in raw_runs],
        )
    except ValueError as error:
        raise InputError(str(error)) from error
```

In `load_session`, replace the `task_model = detect_task_model(...)` call with:

```python
    task_model = _session_task_model(raw_runs, settings.modulators)
```

In `src/boldtailor/workflow/run.py`:
- Add `from boldtailor.task_detection import detect_task_model, task_model_notes`.
- Rewrite `_session`, so that the notes call also sits inside `_input_errors()`:

```python
def _session(settings):
    """Trimmed runs, the task model, and notes on what detection left out."""
    with _input_errors():
        runs = inputs.load_session(settings, hrf_only=True)
        events = [r.events for r in runs]
        task_model = detect_task_model(
            events, settings.modulators, labels=[r.label for r in runs]
        )
        notes = task_model_notes(events, settings.modulators)
    return runs, task_model, notes
```

- [ ] **Step 6: Update the remaining callers' imports**

Every remaining `inputs.detect_task_model(` in `tests/workflow/test_inputs.py`, `test_beta_series.py`, `test_analysis.py` and `test_outputs.py` becomes `detect_task_model(`, with `from boldtailor.task_detection import detect_task_model` at the top of each file.

```bash
uv run python - <<'EOF'
from pathlib import Path
for name in ["test_inputs", "test_beta_series", "test_analysis", "test_outputs"]:
    p = Path(f"tests/workflow/{name}.py")
    t = p.read_text().replace("inputs.detect_task_model(", "detect_task_model(")
    t = t.replace(
        "from boldtailor.model import",
        "from boldtailor.task_detection import detect_task_model\nfrom boldtailor.model import",
        1,
    )
    p.write_text(t)
EOF
grep -n "from boldtailor.task_detection" tests/workflow/test_{inputs,beta_series,analysis,outputs}.py
```

If one of the four files has no `from boldtailor.model import` line, `grep` shows it missing. Add the import by hand there.

Then update these:
- `examples/NSD/test_within_run_encoding.py:12`: `from boldtailor.workflow.inputs import load_session, make_blocks`, plus `from boldtailor.task_detection import detect_task_model`.
- `examples/NSD/test_session_hrf.py:12`: `from boldtailor.workflow.inputs import load_session`, plus `from boldtailor.task_detection import detect_task_model`.
- `examples/NSD/session_hrf.py:10`: remove `detect_task_model,` from the `boldtailor.workflow.inputs` import, and add `from boldtailor.task_detection import detect_task_model`.
- `experiments/nsd_replication/inputs.py:12` and `experiments/nsd_replication/conftest.py:18`: `from boldtailor.task_detection import detect_task_model`.
- `docs/api.md:770`: change the module cell from `boldtailor.workflow.inputs` to `boldtailor.task_detection`, and add "raises `ValueError`" to the description.

Verify: `grep -rn "inputs.detect_task_model\|inputs.task_model_notes\|inputs.TRIAL_TYPE_NOTE\|inputs.observed_levels\|workflow.inputs import.*detect_task_model" src tests examples experiments` prints nothing.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest -q tests/test_task_detection.py tests/workflow`
Expected: all pass. That includes `tests/workflow/test_cli.py::test_input_errors_exit_two_and_name_what_is_missing`, which pins `--modulator stimulus_id` → exit 2, and the `load_session` `InputError` tests.

Run: `uv run pytest -q tests`
Expected: all pass (the 1667 baseline minus the 9 moved tests, plus 11 new ones).

- [ ] **Step 8: Commit GREEN**

```bash
uv run black src/boldtailor/task_detection.py src/boldtailor/workflow tests examples/NSD experiments
git add -A src tests examples/NSD experiments docs/api.md
git commit -m "feat: task-model detection moves to public boldtailor.task_detection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `trial_predictors` in `boldtailor.trial_encoding`

**Files:**
- Modify: `src/boldtailor/trial_encoding.py` (append the functions)
- Modify: `src/boldtailor/workflow/beta_series.py:49-95` (delete), `:541` (call site), and its imports
- Modify: `tests/test_trial_encoding.py` (append tests)
- Modify: `tests/workflow/test_beta_series.py:86-150` (delete the moved tests) and its call sites at about lines 406, 435 and 539
- Modify: `experiments/nsd_replication/ladder.py:19,84-89`
- Modify: `docs/api.md` (add a row next to `evaluate_trial_encoding`)

**Interfaces:**
- Consumes: `boldtailor.model.TaskModel`, `Modulator`; `boldtailor._task_design.categorical_names`.
- Produces: `boldtailor.trial_encoding.trial_predictors(events_tables, task_model, labels=None) -> list[pd.DataFrame]`.
  - Each frame has one column per task-model regressor after `task`, in model order, indexed like its events.
  - Without labels, runs are named `run 1`, `run 2`, … in errors.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trial_encoding.py`:

```python
# --- trial_predictors -------------------------------------------------------

from boldtailor.model import Modulator, TaskModel

RT = Modulator("response_time", missing="indicator")
TT = Modulator("trial_type", kind="categorical", levels=("0", "1"))


def predictors(*args, **kwargs):
    from boldtailor.trial_encoding import trial_predictors

    return trial_predictors(*args, **kwargs)


@pytest.fixture
def trial_events():
    return pd.DataFrame(
        {
            "onset": [8.0, 22.0, 38.0, 60.0, 90.0, 112.0],
            "duration": [3.0] * 6,
            "trial_type": [0, 1, 0, 1, 1, 0],
            "response_time": [0.5, 1.0, 2.5, 1.5, 3.0, 0.5],
        }
    )


def test_trial_predictors_select_modulator_columns_in_model_order(trial_events):
    tables = predictors([trial_events, trial_events], TaskModel((RT, TT)))
    assert [list(t.columns) for t in tables] == [["response_time", "trial_type[1]"]] * 2
    assert all(len(t) == len(trial_events) for t in tables)
    assert tables[0]["trial_type[1]"].tolist() == [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]


def test_trial_predictors_follow_the_task_model(trial_events):
    (table,) = predictors([trial_events], TaskModel((TT,)))
    assert list(table.columns) == ["trial_type[1]"]


def test_trial_predictors_make_nonpositive_or_missing_rt_nan(trial_events):
    events = trial_events.copy()
    events.loc[:2, "response_time"] = [-1.0, 0.0, np.nan]
    (table,) = predictors([events], TaskModel((RT, TT)))
    assert table.response_time.iloc[:3].isna().all()
    assert table.response_time.iloc[3:].notna().all()


def test_trial_predictors_leave_other_columns_untouched():
    events = pd.DataFrame(dict(effort=[-1.0, 0.0, 2.0]))
    (table,) = predictors([events], TaskModel((Modulator("effort"),)))
    assert table.effort.tolist() == [-1.0, 0.0, 2.0]


def test_trial_predictors_missing_column_names_the_run(trial_events):
    broken = trial_events.drop(columns="response_time")
    with pytest.raises(ValueError, match="run-02.*response_time"):
        predictors(
            [trial_events, broken], TaskModel((RT,)), labels=["run-01", "run-02"]
        )
    with pytest.raises(ValueError, match="run 1.*response_time"):
        predictors([broken], TaskModel((RT,)))


def test_trial_predictors_reject_nonbinary_trial_type(trial_events):
    with pytest.raises(ValueError, match="trial_type"):
        predictors([trial_events.assign(trial_type=3)], TaskModel((TT,)))


def test_trial_predictors_emit_indicator_columns_with_missing_as_nan(trial_events):
    mod = Modulator(
        "trial_type", kind="categorical", levels=("0", "1"), missing="indicator"
    )
    events = trial_events.astype({"trial_type": float})
    events.loc[0, "trial_type"] = np.nan
    (table,) = predictors([events], TaskModel((mod,)))
    assert list(table.columns) == ["trial_type[1]"]
    assert np.isnan(table.iloc[0, 0])
    assert set(table.iloc[1:, 0]) <= {0.0, 1.0}


def test_trial_predictors_labels_must_match_the_tables(trial_events):
    with pytest.raises(ValueError, match="labels"):
        predictors([trial_events, trial_events], TaskModel((RT,)), labels=["run-01"])
```

Delete the seven `test_trial_predictors_*` tests and the `# --- trial_predictors` header from `tests/workflow/test_beta_series.py`, lines 86–150. Keep the `# --- shared-alpha CV` header.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_trial_encoding.py -q -k trial_predictors`
Expected: 8 failures with `ImportError: cannot import name 'trial_predictors' from 'boldtailor.trial_encoding'`.

- [ ] **Step 3: Commit RED**

```bash
git add tests/test_trial_encoding.py tests/workflow/test_beta_series.py
git commit -m "test: trial_predictors takes event tables in boldtailor.trial_encoding (RED)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Implement in `src/boldtailor/trial_encoding.py`**

Add these imports at the top:

```python
from dataclasses import replace

from boldtailor._task_design import categorical_names
```

Append:

```python
def _indicator_frame(values, modulator, label):
    """One 0/1 column per non-reference level; missing values become NaN."""
    try:
        names = categorical_names(values, replace(modulator, missing="indicator"))
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error
    levels = [lv for lv in modulator.levels if lv != modulator.reference]
    columns = {
        name: [np.nan if n is None else float(n == lv) for n in names]
        for name, lv in zip(modulator.regressor_names, levels, strict=True)
    }
    return pd.DataFrame(columns, index=values.index)


def _predictor_frame(events, modulator, label):
    if modulator.column not in events:
        raise ValueError(f"{label}: encoding needs event column {modulator.column!r}")
    values = events[modulator.column]
    if modulator.kind == "categorical":
        return _indicator_frame(values, modulator, label)
    return pd.to_numeric(values, errors="raise").rename(modulator.column).to_frame()


def _censor_response_time(table):
    if "response_time" in table:
        rt = table.response_time
        table.loc[~np.isfinite(rt) | (rt <= 0), "response_time"] = np.nan


def _run_predictors(events, task_model, label):
    frames = [_predictor_frame(events, m, label) for m in task_model.modulators]
    table = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=events.index)
    _censor_response_time(table)
    return table


def trial_predictors(events_tables, task_model, labels=None):
    """Predictor columns per run, named as the task-model regressors after task.

    Unavailable behavior (NaN) excludes only the affected encoding rows.
    ``labels`` name the runs in errors (default ``run 1``, ``run 2``, ...).
    """
    tables = list(events_tables)
    if labels is None:
        labels = [f"run {i}" for i in range(1, len(tables) + 1)]
    if len(labels) != len(tables):
        raise ValueError("labels must name every events table")
    return [
        _run_predictors(events, task_model, label)
        for events, label in zip(tables, labels)
    ]
```

- [ ] **Step 5: Switch the callers**

`src/boldtailor/workflow/beta_series.py`:
- Delete `_indicator_frame`, `_predictor_frame`, `_censor_response_time` and `trial_predictors` (lines 49–95).
- Remove `from boldtailor._task_design import categorical_names`.
- Remove `replace` from the `dataclasses` import, if `grep -n "replace(" src/boldtailor/workflow/beta_series.py` shows no other use.
- Add `trial_predictors` to the existing `from boldtailor.trial_encoding import (...)` block.
- At about line 541, the call becomes:

```python
    predictors = trial_predictors(
        [r.events for r in runs], task_model, [r.label for r in runs]
    )
```

`tests/workflow/test_beta_series.py`:
- Add `trial_predictors` to `from boldtailor.trial_encoding import evaluate_trial_encoding`.
- Replace each remaining `beta_series.trial_predictors(X, model)` with `trial_predictors([r.events for r in X], model)`. They are at about lines 406 (`X = subset`), 435 and 539 (`X = runs`).

`experiments/nsd_replication/ladder.py`:
- Replace line 19 with `from boldtailor.trial_encoding import trial_predictors`.
- Replace `_predictors` with the version below, and remove the `SimpleNamespace` import if nothing else uses it (`grep -n SimpleNamespace experiments/nsd_replication/ladder.py`).

```python
def _predictors(session, task_model):
    return trial_predictors(session.events, task_model, session.labels)
```

`docs/api.md`: add a row next to `evaluate_trial_encoding`:

| `trial_predictors(events_tables, task_model, labels=None)` | `boldtailor.trial_encoding` | One predictor table per run: task-model regressors after `task`, with RT ≤ 0 or non-finite set to NaN and categorical indicators NaN where missing. |

Verify: `grep -rn "beta_series.trial_predictors\|workflow.beta_series import trial_predictors" src tests examples experiments` prints nothing.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q tests/test_trial_encoding.py tests/workflow/test_beta_series.py`
Expected: all pass.

Run: `uv run pytest -q tests experiments/nsd_replication/test_ladder.py`
Expected: all pass.

- [ ] **Step 7: Commit GREEN**

```bash
uv run black src tests experiments
git add -A src tests experiments docs/api.md
git commit -m "feat: trial_predictors moves to boldtailor.trial_encoding and takes event tables

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Experiment config uses `nsdpreproc_dir`

**Files:**
- Modify: `experiments/nsd_replication/config.py`
- Modify: `experiments/nsd_replication/configs/{pilot,primary,extension}.toml` (line 4)
- Test: `experiments/nsd_replication/test_config.py`

**Interfaces:**
- Produces:
  - `ExperimentConfig.nsdpreproc_dir: Path`. It is always set after `__post_init__`, defaulting to `bids_dir / "derivatives" / "nsdpreproc"`.
  - The field `ppdata_dir` is gone.
  - `load_config` raises `ValueError` mentioning `nsdpreproc_dir` whenever the TOML has `ppdata_dir`.

- [ ] **Step 1: Write the failing tests**

In `experiments/nsd_replication/test_config.py`, replace `assert config.ppdata_dir is None` in `test_load_config_converts_paths_and_tuples` with:

```python
    assert config.nsdpreproc_dir == Path("/data/BIDS/derivatives/nsdpreproc")
```

Then append:

```python
def _direct(**extra):
    values = dict(
        bids_dir=Path("/b"),
        output_dir=Path("/o"),
        freesurfer_dir=Path("/l"),
        subjects=("sub-07",),
        sessions=("ses-nsd10",),
    )
    return ExperimentConfig(**{**values, **extra})


def test_direct_construction_defaults_nsdpreproc_under_bids():
    assert _direct().nsdpreproc_dir == Path("/b/derivatives/nsdpreproc")


def test_explicit_nsdpreproc_dir_is_honoured(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(TOML + 'nsdpreproc_dir = "/elsewhere/nsdpreproc"\n')
    assert load_config(path).nsdpreproc_dir == Path("/elsewhere/nsdpreproc")


@pytest.mark.parametrize(
    "extra",
    [
        'ppdata_dir = "/data/ppdata"\n',
        'ppdata_dir = "/data/ppdata"\nnsdpreproc_dir = "/data/nsdpreproc"\n',
    ],
)
def test_ppdata_dir_is_rejected_with_the_replacement_key(tmp_path, extra):
    path = tmp_path / "c.toml"
    path.write_text(TOML + extra)
    with pytest.raises(ValueError, match="ppdata_dir.*nsdpreproc_dir"):
        load_config(path)


def test_label_validation_does_not_need_the_workflow():
    import ast

    source = (Path(__file__).parent / "config.py").read_text()
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(m and m.startswith("boldtailor.workflow") for m in imported)
    with pytest.raises(ValueError, match="subjects"):
        _direct(subjects=("sub-07_x",))
```

Add this assertion to the end of `test_shipped_configs`:

```python
    assert config.nsdpreproc_dir == Path(
        "/Volumes/extdata1/NSD/BIDS/derivatives/nsdpreproc"
    )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest experiments/nsd_replication/test_config.py -q`
Expected:
- `AttributeError: 'ExperimentConfig' object has no attribute 'nsdpreproc_dir'`;
- the `ppdata_dir` cases fail, because no error is raised;
- the workflow-import test fails.

- [ ] **Step 3: Commit RED**

```bash
git add experiments/nsd_replication/test_config.py
git commit -m "test: experiment config reads nsdpreproc_dir and rejects ppdata_dir (RED)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Implement `config.py`**

Replace the `from boldtailor.workflow.files import bids_label` import with `import re`, and replace `_labels` with:

```python
def _is_label(value, entity):
    return isinstance(value, str) and re.fullmatch(rf"{entity}-[A-Za-z0-9]+", value)


def _labels(name, values, entity):
    values = tuple(values)
    if (
        not values
        or len(set(values)) != len(values)
        or not all(_is_label(v, entity) for v in values)
    ):
        raise ValueError(f"{name} must be distinct {entity}-<label> values: {values}")
    return values
```

In the dataclass, replace `ppdata_dir: Path | None = None` with `nsdpreproc_dir: Path | None = None`. At the end of `__post_init__`, add:

```python
        if self.nsdpreproc_dir is None:
            set_(self, "nsdpreproc_dir", Path(self.bids_dir) / "derivatives/nsdpreproc")
```

In `load_config`, right after `values = tomllib.loads(...)`, add:

```python
    if "ppdata_dir" in values:
        raise ValueError(
            "ppdata_dir is no longer read; set nsdpreproc_dir to the BIDS "
            "derivatives/nsdpreproc directory (defaults to <bids_dir>/derivatives/nsdpreproc)"
        )
```

In each of `configs/pilot.toml`, `primary.toml` and `extension.toml`, replace line 4 with:

```toml
nsdpreproc_dir = "/Volumes/extdata1/NSD/BIDS/derivatives/nsdpreproc"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest experiments/nsd_replication/test_config.py -q`
Expected: all pass.

Other experiment tests that still pass `ppdata_dir=` (`test_inputs.py`, `test_roi.py`, `test_released.py`) are data-gated and get fixed in Task 4. Check that `uv run pytest experiments/nsd_replication -q` fails only in those three files, and only where they pass `ppdata_dir`.

- [ ] **Step 6: Commit GREEN**

```bash
uv run black experiments
git add experiments/nsd_replication/config.py experiments/nsd_replication/configs
git commit -m "feat: experiment config takes nsdpreproc_dir; ppdata_dir is rejected

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: nsdpreproc loader

**Files:**
- Modify: `experiments/nsd_replication/inputs.py`
- Modify (callers): `experiments/nsd_replication/run.py:37,76,326`, `released.py:13,102`, `calibrate_offset.py:1,11,35`, `confounds.py:1`
- Modify (tests): `test_inputs.py`, `test_roi.py:86-104`, `test_released.py:83-96`, `test_calibrate_offset.py:33-37`, and the `ppdata_brain` monkeypatch strings in `test_features.py:232`, `test_pilot_report.py:27` and `test_run.py:32`
- Create: `experiments/nsd_replication/test_nsdpreproc_data.py`

**Interfaces:**
- Consumes: `ExperimentConfig.nsdpreproc_dir` (Task 3); `boldtailor.task_detection.detect_task_model` (Task 1).
- Produces, in `experiments.nsd_replication.inputs`:
  - `nsdpreproc_path(config, subject, session, run) -> Path`
  - `load_nsdpreproc(config, subject, session) -> Session`, with `source == "nsdpreproc"`
  - `nsdpreproc_brain(config, subject, session) -> BrainModelAxis`

  `load_ppdata`, `ppdata_brain` and `ppdata_path` are removed.

- [ ] **Step 1: Write the failing tests**

In `experiments/nsd_replication/test_inputs.py`:
- Change `source="ppdata"` in `test_analysis_data_selects_columns_and_keeps_runs` to `source="nsdpreproc"`.
- Delete `_pilot_config` and `test_ppdata_pilot_session_is_cortical_only`; they move to the new data file below.
- Append:

```python
TR = 4 / 3
N_FRAMES = 40


def _cifti(path, seed):
    brain = nib.cifti2.BrainModelAxis.from_surface(
        np.arange(3), 10, "CIFTI_STRUCTURE_CORTEX_LEFT"
    ) + nib.cifti2.BrainModelAxis.from_surface(
        np.arange(3), 10, "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    series = nib.cifti2.SeriesAxis(start=0.0, step=TR, size=N_FRAMES)
    rng = np.random.default_rng(seed)
    data = rng.normal(100.0, 1.0, (N_FRAMES, len(brain))).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.Cifti2Image(data, header=(series, brain)).to_filename(path)


def _tree_events():
    return pd.DataFrame(
        {
            "onset": [4.0, 12.0, 20.0, 28.0],
            "duration": [3.0] * 4,
            "trial_type": [0, 1, 0, 1],
            "response_time": [0.6, 0.8, 1.0, 0.7],
            "73k_id": [11, 12, 13, 14],
        }
    )


@pytest.fixture
def nsd_tree(tmp_path):
    """Raw events, nsdpreproc CIFTIs, and a decoy fMRIPrep CIFTI per run."""
    from experiments.nsd_replication.config import ExperimentConfig

    raw = tmp_path / "sub-07/ses-a/func"
    raw.mkdir(parents=True)
    pre = tmp_path / "derivatives/nsdpreproc/sub-07/ses-a/func"
    prep = tmp_path / "derivatives/fmriprep/sub-07/ses-a/func"
    for i, run in enumerate(("run-01", "run-02")):
        name = f"sub-07_ses-a_task-nsdcore_{run}"
        _tree_events().to_csv(raw / f"{name}_events.tsv", sep="\t", index=False)
        _cifti(pre / f"{name}_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii", i)
        _cifti(prep / f"{name}_space-fsLR_den-91k_bold.dtseries.nii", 99)
    config = ExperimentConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path / "out",
        freesurfer_dir=tmp_path,
        subjects=("sub-07",),
        sessions=("ses-a",),
    )
    return config, pre


def test_nsdpreproc_path_is_bids_layout(nsd_tree):
    from experiments.nsd_replication.inputs import nsdpreproc_path

    config, pre = nsd_tree
    assert nsdpreproc_path(config, "sub-07", "ses-a", "run-02") == (
        pre / "sub-07_ses-a_task-nsdcore_run-02_space-fsLR_den-32k_desc-layerB2_bold"
        ".dtseries.nii"
    )


def test_load_nsdpreproc_opens_only_nsdpreproc_files(nsd_tree, monkeypatch):
    from experiments.nsd_replication.inputs import load_nsdpreproc

    config, pre = nsd_tree
    opened, real_load = [], nib.load

    def recording_load(path, *args, **kwargs):
        opened.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", recording_load)
    session = load_nsdpreproc(config, "sub-07", "ses-a")
    monkeypatch.setattr(nib, "load", real_load)
    assert opened and all(p.is_relative_to(pre) for p in opened)
    assert session.source == "nsdpreproc"
    assert session.labels == ("run-01", "run-02")
    assert session.signals[0].shape == (N_FRAMES, 6)
    assert list(session.confounds[0].columns) == ["poly_0"]
    first = next(pre.glob("*run-01_*.dtseries.nii"))
    np.testing.assert_allclose(session.signals[0], real_load(first).get_fdata(), rtol=1e-6)


def test_missing_nsdpreproc_run_names_the_run_and_path(nsd_tree):
    from experiments.nsd_replication.inputs import load_nsdpreproc

    config, pre = nsd_tree
    next(pre.glob("*run-02*")).unlink()
    with pytest.raises(FileNotFoundError, match="run-02.*nsdpreproc"):
        load_nsdpreproc(config, "sub-07", "ses-a")


def test_missing_nsdpreproc_session_names_the_run_and_path(nsd_tree):
    import shutil

    from experiments.nsd_replication.inputs import load_nsdpreproc

    config, pre = nsd_tree
    shutil.rmtree(pre.parents[1])
    with pytest.raises(FileNotFoundError, match="run-01.*nsdpreproc/sub-07/ses-a"):
        load_nsdpreproc(config, "sub-07", "ses-a")
```

Create `experiments/nsd_replication/test_nsdpreproc_data.py`:

```python
"""Real NSD data: the pilot session loads from nsdpreproc only."""

from pathlib import Path

import numpy as np
import pytest

from experiments.nsd_replication.config import ExperimentConfig
from experiments.nsd_replication.inputs import (
    CORTEX,
    load_nsdpreproc,
    nsdpreproc_brain,
    same_grayordinates,
)

DATA = Path("/Volumes/extdata1/NSD/BIDS")
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="NSD data volume not mounted")


def _pilot_config():
    return ExperimentConfig(
        bids_dir=DATA,
        output_dir=DATA / "derivatives/nsd-replication",
        freesurfer_dir=DATA / "derivatives/freesurfer-NSD",
        subjects=("sub-07",),
        sessions=("ses-nsd10",),
    )


def test_pilot_session_loads_from_nsdpreproc():
    config = _pilot_config()
    assert config.nsdpreproc_dir == DATA / "derivatives/nsdpreproc"
    session = load_nsdpreproc(config, "sub-07", "ses-nsd10")
    assert session.source == "nsdpreproc"
    assert len(session.labels) == 12
    assert all(s.shape == (226, 59412) for s in session.signals)
    assert set(session.brain.name) == set(CORTEX)
    assert "73k_id" in session.events[0]
    assert list(session.confounds[0].columns) == [f"poly_{d}" for d in range(4)]
    np.testing.assert_allclose(np.diff(session.frame_times[0]), 4 / 3, rtol=1e-5)
    assert same_grayordinates(
        nsdpreproc_brain(config, "sub-07", "ses-nsd10"), session.brain
    )
    assert session.task_model is not None
```

Update the other data-gated tests:
- `test_roi.py:86-104`: rename to `test_pilot_roi_on_nsdpreproc_cortex`, delete the `ppdata_dir=...` argument, and import or call `nsdpreproc_brain`.
- `test_released.py:83-96`: rename to `test_pilot_released_matches_nsdpreproc_axis`, delete `ppdata_dir=...`, and use `nsdpreproc_brain`.

Update the monkeypatch targets: `"ppdata_brain"` → `"nsdpreproc_brain"` in `test_features.py:232`, `test_pilot_report.py:27` and `test_run.py:32`; and `"load_ppdata"` → `"load_nsdpreproc"` in `test_calibrate_offset.py:37`. In `test_calibrate_offset.py:33`, the docstring becomes `"""nsdpreproc at onset_offset o has frame times shifted by o; released is offset 0."""`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest experiments/nsd_replication -q`
Expected:
- `ImportError` for `load_nsdpreproc`, `nsdpreproc_path` and `nsdpreproc_brain`;
- `AttributeError` from monkeypatching the missing `nsdpreproc_brain` and `load_nsdpreproc`.

- [ ] **Step 3: Commit RED**

```bash
uv run black experiments
git add experiments/nsd_replication/test_*.py
git commit -m "test: NSD sessions load from BIDS derivatives/nsdpreproc only (RED)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Implement `inputs.py`**

- Replace the module docstring with `"""Load one NSD session from BIDS derivatives/nsdpreproc as cortical-only arrays."""`.
- Replace `_PPDATA` and `ppdata_path` with:

```python
_NSDPREPROC = (
    "{subject}/{session}/func/{subject}_{session}_task-{task}_{run}"
    "_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii"
)


def nsdpreproc_path(config, subject, session, run):
    name = _NSDPREPROC.format(
        subject=subject, session=session, task=config.task, run=run
    )
    return Path(config.nsdpreproc_dir) / name
```

- In `_existing_path`, call `nsdpreproc_path`. Its message stays `f"{subject} {session} {run}: missing {path}"`, which names the run and the full nsdpreproc path.
- In `_check_onsets`, the message becomes `"onsets fall outside the nsdpreproc time base"`.
- In `_cortex`, it becomes `"nsdpreproc runs differ in grayordinates"`.
- Rename `load_ppdata` → `load_nsdpreproc` and `ppdata_brain` → `nsdpreproc_brain`.
- Set `source="nsdpreproc"`.
- Leave everything else unchanged: cortex selection, polynomials, `onset_offset` and task-model detection.

- [ ] **Step 5: Update the callers**

- `run.py:37`: `from experiments.nsd_replication.inputs import load_nsdpreproc, nsdpreproc_brain`.
- `run.py:76`: `return load_nsdpreproc(config, subject, session)`.
- `run.py:326`: `brain = nsdpreproc_brain(config, subject, config.sessions[0])`.
- `released.py:13`: `nsdpreproc_brain,` in the import list. Line 102: `brain = nsdpreproc_brain(config, subject, session)`. Line 48: `"grayordinates differ from the nsdpreproc cortex"`.
- `calibrate_offset.py`: in the docstring, `"""Calibrate the constant offset between BIDS onsets and the nsdpreproc time base."""`. Line 11: `from experiments.nsd_replication.inputs import load_nsdpreproc`. Line 35: `data = load_nsdpreproc(config, subject, session)`.
- `confounds.py:1`: `"""GLMsingle's polynomial drift basis, used as boldtailor's confounds for nsdpreproc."""`.

Verify: `grep -rn "load_ppdata\|ppdata_brain\|ppdata_path\|ppdata_dir\|_PPDATA" experiments` prints nothing.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest experiments/nsd_replication -q`
Expected: all pass, including `test_nsdpreproc_data.py`, `test_roi.py` and `test_released.py` when `/Volumes/extdata1` is mounted. Tests that still use the `"ppdata"` source label pass, because the label hasn't been renamed yet.

- [ ] **Step 7: Commit GREEN**

```bash
uv run black experiments
git add -A experiments
git commit -m "feat: NSD experiments load CIFTI from BIDS derivatives/nsdpreproc

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Source label `ppdata` → `nsdpreproc`

**Files:**
- Modify: `experiments/nsd_replication/run.py` (lines 63, 74–75, 363, 393, 395, 456, 565, 611, 619, 631, 644, 691–692, 704, 738), `figure_stage.py:72,94,107`, `comparison.py:8,24`, `pilot_report.py:62`, `README.md`, `protocol.md`
- Modify (tests): `test_run.py`, `test_features.py`, `test_pilot_report.py`, `test_betas.py`, `conftest.py:96`

**Interfaces:**
- Produces:
  - `run.SOURCES == ("nsdpreproc", "released", "comparison")`; the `--source` default is `"nsdpreproc"`.
  - Output dirs: `fits/nsdpreproc/…` (through `fit_dir`) and `metrics/nsdpreproc/…`.
  - Version keys: `nsdpreproc:<level>`.

- [ ] **Step 1: Rewrite the tests for the new label**

```bash
uv run python - <<'EOF'
from pathlib import Path
import re
d = Path("experiments/nsd_replication")
for name in ["test_run", "test_features", "test_pilot_report", "test_betas", "conftest"]:
    p = d / f"{name}.py"
    t = p.read_text()
    t = t.replace('"ppdata"', '"nsdpreproc"').replace("metrics/ppdata", "metrics/nsdpreproc")
    t = t.replace('"ppdata:', '"nsdpreproc:').replace("ppdata:{", "nsdpreproc:{")
    t = t.replace('out / "ppdata"', 'out / "nsdpreproc"')
    t = t.replace("test_features_requires_ppdata", "test_features_requires_nsdpreproc")
    t = t.replace("test_ppdata_default_levels_include_lss", "test_nsdpreproc_default_levels_include_lss")
    t = re.sub(r'match="ppdata"', 'match="nsdpreproc"', t)
    p.write_text(t)
EOF
grep -n "ppdata" experiments/nsd_replication/test_*.py experiments/nsd_replication/conftest.py
```

Expected: the `grep` prints nothing. If some match remains, it's a path or key that wasn't anticipated. Rewrite it to `nsdpreproc` by hand.

Then append to `test_run.py` (the parser is `run._parser()`, defined at `run.py:700`):

```python
def test_sources_name_nsdpreproc_not_ppdata():
    assert run.SOURCES == ("nsdpreproc", "released", "comparison")
    args = run._parser().parse_args(["fit", "--config", "c.toml"])
    assert args.source == "nsdpreproc"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest experiments/nsd_replication -q`
Expected: failures in `test_run.py`, `test_features.py` and `test_pilot_report.py`, for example `ValueError: source 'nsdpreproc' has no fit stage` and missing `metrics/nsdpreproc/...` files.

- [ ] **Step 3: Commit RED**

```bash
git add experiments/nsd_replication/test_*.py experiments/nsd_replication/conftest.py
git commit -m "test: the fitted NSD source is labelled nsdpreproc (RED)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Rename in the implementation**

```bash
uv run python - <<'EOF'
from pathlib import Path
d = Path("experiments/nsd_replication")
for name in ["run", "figure_stage", "comparison", "pilot_report"]:
    p = d / f"{name}.py"
    t = p.read_text()
    t = t.replace('"ppdata"', '"nsdpreproc"').replace('"ppdata:', '"nsdpreproc:')
    t = t.replace("f\"ppdata:{", "f\"nsdpreproc:{").replace('("ppdata", "released")', '("nsdpreproc", "released")')
    t = t.replace("only ppdata is fitted", "only nsdpreproc is fitted")
    t = t.replace("run on ppdata only", "run on nsdpreproc only")
    p.write_text(t)
EOF
grep -n "ppdata" experiments/nsd_replication/{run,figure_stage,comparison,pilot_report}.py
```

Expected: the `grep` prints nothing. Fix any remaining match by hand.

Docs: in `experiments/nsd_replication/README.md` and `protocol.md`, rewrite the `ppdata` mentions:
- The inputs bullet becomes: "NSD layer-B2 time series resampled to fsLR 32k, as BIDS derivatives under `nsdpreproc_dir` (default `<bids_dir>/derivatives/nsdpreproc`: `sub-NN/ses-nsdYY/func/*_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii`). nsdpreproc is the BIDS form of NSD's ppdata. Only these CIFTI files are read; fMRIPrep CIFTI files are never used. `onset_offset` was calibrated by the pilot (0.667 s, set in every config)."
- `--source <ppdata|released|comparison>` becomes `--source <nsdpreproc|released|comparison>`.
- "ppdata fit/metrics/betas" becomes "nsdpreproc fit/metrics/betas".
- `metrics/ppdata/` becomes `metrics/nsdpreproc/`.
- `protocol.md:81`: "The cortical part is exactly the nsdpreproc grayordinates (same names and vertices)."

Verify: `grep -n "ppdata" experiments/nsd_replication/README.md experiments/nsd_replication/protocol.md` returns only lines that call nsdpreproc "the BIDS form of NSD's ppdata".

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest experiments/nsd_replication -q`
Expected: all pass.

- [ ] **Step 6: Commit GREEN**

```bash
uv run black experiments
git add -A experiments
git commit -m "feat: the fitted NSD source and its outputs are labelled nsdpreproc

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Import and data boundary test

**Files:**
- Create: `experiments/nsd_replication/boundary.py`
- Create: `experiments/nsd_replication/test_boundary.py`

**Interfaces:**
- Produces: `experiments.nsd_replication.boundary.violations(source: str) -> list[str]`. It returns human-readable findings for one module's source text, and an empty list when the module is clean.

- [ ] **Step 1: Write the failing tests**

```python
"""NSD experiments use only core boldtailor and never fMRIPrep CIFTI files."""

from pathlib import Path

import pytest

from experiments.nsd_replication.boundary import violations

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "source",
    [
        "from boldtailor.workflow.inputs import load_session\n",
        "import boldtailor.workflow.inputs as w\n",
        "import boldtailor.cli\n",
        "from boldtailor import workflow\n",
        "from boldtailor import cli, model\n",
        "from boldtailor.workflow import inputs\n",
        'P = "{s}_space-fsLR_den-91k_bold.dtseries.nii"\n',
        'P = "derivatives/ppdata/subj07"\n',
        'P = "subj{nn}/func1pt8mm/timeseries"\n',
    ],
)
def test_each_violation_form_is_caught(source):
    assert violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "from boldtailor.model import TaskModel\n",
        "from boldtailor import model\n",
        "import boldtailor.task_detection\n",
        '"""nsdpreproc is the BIDS form of NSD\'s ppdata."""\n',
        'P = "derivatives/fmriprep/sub-07/anat/x_desc-confounds_timeseries.tsv"\n',
        'P = "fmriprep_dir"\n',
        'P = "{s}_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii"\n',
        'P = "{s}_space-fsLR_den-91k_desc-assumehrf_stat-effect_statmap.dscalar.nii"\n',
    ],
)
def test_allowed_sources_pass(source):
    assert violations(source) == []


def test_experiments_respect_the_boundary():
    found = {
        str(path.relative_to(ROOT)): problems
        for path in sorted(ROOT.rglob("*.py"))
        if path.name not in {"test_boundary.py", "boundary.py"}
        and (problems := violations(path.read_text()))
    }
    assert found == {}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest experiments/nsd_replication/test_boundary.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'experiments.nsd_replication.boundary'`.

- [ ] **Step 3: Commit RED**

```bash
git add experiments/nsd_replication/test_boundary.py
git commit -m "test: NSD experiments stay off workflow/CLI imports and fMRIPrep CIFTI (RED)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Implement `boundary.py`**

```python
"""Static checks that keep the NSD experiments on core boldtailor and nsdpreproc."""

import ast

FORBIDDEN_MODULES = ("boldtailor.workflow", "boldtailor.cli")
FORBIDDEN_FRAGMENTS = (
    "den-91k_bold",  # fMRIPrep CIFTI BOLD: *_space-fsLR_den-91k_bold.dtseries.nii
    "derivatives/ppdata",
    "func1pt8mm",
)


def _forbidden(module):
    return any(module == m or module.startswith(m + ".") for m in FORBIDDEN_MODULES)


def _imported(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module] + [f"{node.module}.{a.name}" for a in node.names]
    return []


def _import_problems(tree):
    return [
        f"line {node.lineno}: imports {name}"
        for node in ast.walk(tree)
        for name in _imported(node)
        if _forbidden(name)
    ]


def _string_problems(tree):
    return [
        f"line {node.lineno}: names {fragment!r}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        for fragment in FORBIDDEN_FRAGMENTS
        if fragment in node.value
    ]


def violations(source):
    """Boundary violations in one module's source text."""
    tree = ast.parse(source)
    return _import_problems(tree) + _string_problems(tree)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest experiments/nsd_replication/test_boundary.py -q`
Expected: all pass. If `test_experiments_respect_the_boundary` lists a finding, fix the experiment module. The test is not wrong; the finding is a real leftover from Tasks 1–5.

Run: `uv run pytest experiments/nsd_replication -q`
Expected: all pass.

- [ ] **Step 6: Commit GREEN**

```bash
uv run black experiments
git add experiments/nsd_replication/boundary.py
git commit -m "feat: static boundary check for the NSD experiments

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Relabel `examples/NSD` as `examples/bids_workflow`; docs and CI

**Files:**
- Move: `examples/NSD/` → `examples/bids_workflow/`
- Modify: every `examples.NSD` import under `examples/bids_workflow/`; `examples/bids_workflow/README.md`; `README.md`; `docs/README.md`; `docs/user-guide.md`; `docs/api.md:817`; `docs/development.md`; `.github/workflows/tests.yml:27,42`
- Test: `tests/test_docs_mention_cli.py`, which already checks the docs; append one test

**Interfaces:**
- Produces: the import package `examples.bids_workflow.*`. Notebook settings helpers keep their names (`nsd_settings`, `nsd_paths`) and the `NSD_*` environment variables.

- [ ] **Step 1: Write the failing test**

The file already imports `Path` and reads paths relative to the repo root, since pytest runs from there. Append:

```python
REPO = Path(__file__).resolve().parents[1]
LIVING_DOCS = [
    "README.md",
    "docs/README.md",
    "docs/user-guide.md",
    "docs/api.md",
    "docs/development.md",
    ".github/workflows/tests.yml",
]


def test_living_docs_point_at_bids_workflow_examples_and_two_kinds_of_code():
    for name in LIVING_DOCS:
        text = (REPO / name).read_text()
        assert "examples/NSD" not in text, name
    assert (REPO / "examples/bids_workflow/README.md").is_file()
    assert not (REPO / "examples/NSD").exists()
    for name in ("README.md", "docs/development.md"):
        text = (REPO / name).read_text()
        assert "Two kinds of code" in text, name
        assert "experiments/nsd_replication" in text and "nsdpreproc" in text, name
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_docs_mention_cli.py -q`
Expected: FAIL with `README.md` (contains `examples/NSD`).

- [ ] **Step 3: Commit RED**

```bash
git add tests/test_docs_mention_cli.py
git commit -m "test: docs point at examples/bids_workflow and describe two kinds of code (RED)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Move the directory and rewrite the imports**

Before moving, look at what is untracked: `git status --ignored examples/NSD`. Expect `*.ipynb` and `__pycache__/`. The `.ipynb` files are the user's paired notebooks; move them along with the rest, and don't delete them.

```bash
git mv examples/NSD examples/bids_workflow
[ -d examples/NSD ] && mv examples/NSD/*.ipynb examples/bids_workflow/ 2>/dev/null
rm -rf examples/NSD
grep -rl "examples\.NSD\|examples/NSD" examples/bids_workflow | xargs sed -i '' 's/examples\.NSD/examples.bids_workflow/g; s#examples/NSD#examples/bids_workflow#g'
grep -rn "examples\.NSD\|examples/NSD" examples
```

Expected: the final `grep` prints nothing.

The `sed -i ''` form is for macOS. `jupytext.toml` and `.gitignore` have no `NSD` paths, so they need no change.

- [ ] **Step 5: Relabel the README and update the living docs and CI**

`examples/bids_workflow/README.md`: replace the title and first paragraph with:

```markdown
# BIDS workflow examples (NSD fMRIPrep data)

These notebooks demonstrate the **BIDS + fMRIPrep** path, the same one the
`boldtailor run` command uses: raw BIDS events plus fMRIPrep fsLR 91k CIFTI
time series and confounds. They use the Natural Scenes Dataset's fMRIPrep
outputs as the example dataset. Start with the [full workflow notebook](nsd_workflow.py);
the other two notebooks reuse or extend its results.

The NSD replication experiments are separate. They live in
[`experiments/nsd_replication/`](../../experiments/nsd_replication/README.md),
read only the NSD team's preprocessed CIFTI files (`derivatives/nsdpreproc`),
and never use fMRIPrep CIFTI files.
```

Keep the rest of that README. Its `examples/NSD` paths were already rewritten in Step 4.

Living docs:
- In `README.md`, `docs/README.md`, `docs/user-guide.md`, `docs/api.md` and `docs/development.md`, replace `examples/NSD` with `examples/bids_workflow`.
- In the `README.md` rows at lines 31–34 and 121–127, and in `docs/README.md:17`, change "NSD example"/"NSD guide" link text to "BIDS workflow examples (NSD data)".
- Leave everything under `docs/validation/`, `docs/review-*`, `docs/test-suite-review-*` and `docs/superpowers/` (except this plan's spec) alone.

```bash
sed -i '' 's#examples/NSD#examples/bids_workflow#g' README.md docs/README.md docs/user-guide.md docs/api.md docs/development.md .github/workflows/tests.yml
```

Add this section to `README.md`, right before `## Documentation` (line 155), and the same text to `docs/development.md` after its layout table, near line 73:

```markdown
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
```

In `docs/development.md:73`, change the layout table row to `| Notebook-only BIDS workflow helpers | examples/bids_workflow |`. In `docs/development.md:87`, the sentence becomes "Shared example fixtures live in `examples/bids_workflow/conftest.py`".

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_docs_mention_cli.py -q`
Expected: PASS.

Run: `uv run pytest -q examples/bids_workflow`
Expected: passes, with data-dependent tests skipped if NSD is not mounted. This checks that the import rewrite works. It does not run the notebooks.

Run: `uv run black --check src tests examples/bids_workflow experiments`
Expected: no changes needed.

Run: `uv run pytest -q tests`
Expected: all pass.

- [ ] **Step 7: Commit GREEN**

```bash
git add -A examples README.md docs/README.md docs/user-guide.md docs/api.md docs/development.md .github/workflows/tests.yml
git commit -m "docs: examples/NSD becomes examples/bids_workflow; describe the two kinds of code

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run the full suites**

Run: `uv run pytest -q -W error tests`. This matches CI. Expected: all pass.

Run: `uv run pytest -q experiments/nsd_replication`. Expected: all pass. With the NSD volume mounted, the data-gated tests run too.

Run: `uv run pytest -q examples/bids_workflow examples/validation`. Expected: all pass. Notebook execution stays opt-in behind `--run-notebooks`.

Run: `git diff --check main...HEAD`. Expected: no whitespace errors.

- [ ] **Step 2: Sweep for leftovers**

```bash
grep -rn "boldtailor.workflow\|boldtailor import workflow\|boldtailor.cli" experiments
grep -rn "ppdata_dir\|load_ppdata\|ppdata_brain\|\"ppdata\"" experiments src tests examples
grep -rn "examples/NSD\|examples\.NSD" README.md docs/README.md docs/user-guide.md docs/api.md docs/development.md .github examples
```

Expected: all three print nothing.

- [ ] **Step 3: Optional real-data check**

Only if `/Volumes/extdata1` is mounted, run `uv run python -m experiments.nsd_replication.run fit --config experiments/nsd_replication/configs/pilot.toml --source nsdpreproc --levels b1`.

Expected: it writes `/Volumes/extdata1/NSD/BIDS/derivatives/nsd-replication/fits/nsdpreproc/sub-07/ses-nsd10/b1/` (`betas.fit_dir` is `output_dir/fits/<source>/<subject>/<session>/<level>`). Report the output path. This writes outside the repo, so ask the user before running it.
