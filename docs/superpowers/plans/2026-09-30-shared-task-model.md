# Shared Task Model Implementation Plan

> **Development history:** This dated plan/design may describe proposed or
> superseded behavior. Use the [documentation index](../../README.md) for current guidance.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** HRF selection and conventional GLM fitting score and fit the same
task model: an all-trials task regressor, within-run centered RT modulation,
uncentered trial type, confounds, and a per-run missing-RT indicator.

**Architecture:** A small frozen `TaskModel` expands raw per-trial events into
Nilearn's events format. One helper builds task columns with
`make_first_level_design_matrix` for any HRF; selection and fitting both call
it. Selection generalizes its sufficient statistics from one regressor to K
task regressors with per-run profiled indicator columns, keeping the held-out
prediction objective.

**Tech Stack:** Python >=3.12, uv, NumPy, pandas, SciPy, Nilearn 0.14, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-shared-task-model-design.md`

## Global Constraints

- All local commands use `uv run`. Dependency changes use uv.
- Every `__init__.py` stays empty. Prefer short functions and pytest fixtures.
- Write and commit failing tests before implementation. Never weaken a test to pass.
- `TaskModel()` (task-only) must reproduce current selection indices, scores, eligibility, and fingerprint invariances exactly.
- Missing means non-finite. Centering is within run over observed trials.
- Selection uses `min_onset=-24.0` and `oversampling=50`, recorded in provenance.
- Denominator of the selection score is confound-only projected energy and is candidate-independent.
- Eligibility tolerances are numerical only: `largest * max(shape) * eps`.
- No new convolution code. Nilearn builds every design.
- Regressor names `task`, `constant`, `onset`, `duration`, and any name beginning with `missing_` are reserved.
- Existing analysis fingerprints for fits without a task model must not change: add provenance keys only when a task model is set.

## Review Focus

Inputs the spec implies but which needed explicit tests. Each is pinned in the task named.

1. A modulator column holding text such as `"invalid"` must raise a `ValueError` naming the column, not silently coerce to NaN. (Task 2)
2. A run in which every value of an `indicator` modulator is missing must raise, since the modulator then has no observed values to center on. (Task 2)
3. `ModelSpec(task_model=..., hrf_model="spm + derivative")` must raise, because a task model defines exactly one column per regressor. (Task 3)
4. A selection made with a non-default task model must still work with `fit_selected_hrfs`, whose run designs use the task-only default. (Task 5)
5. Nilearn must not write to stdout while building task columns, since the NSD workflow builds thousands of them inside worker processes. (Task 2)

---

### Task 1: TaskModel and Modulator specification

**Files:**
- Modify: `src/boldtailor/model.py`
- Test: `tests/test_task_model.py`

**Interfaces:**
- Produces:
  - `Modulator(column: str, center: bool = True, missing: str = "error")` with property `indicator_name -> str` equal to `f"missing_{column}"`.
  - `TaskModel(modulators: tuple[Modulator, ...] = ())` with properties `regressor_names -> tuple[str, ...]` (`("task", *columns)`), `profiled_names -> tuple[str, ...]` (indicator names of modulators with `missing="indicator"`), `fingerprint -> str`, and method `to_dict() -> dict`.
  - Both are frozen dataclasses, hashable, and comparable with `==`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_task_model.py`:

```python
"""A task model is a frozen, fingerprinted list of task regressors."""

import json

import pytest

from boldtailor.model import Modulator, TaskModel


def test_default_task_model_is_task_only():
    model = TaskModel()
    assert model.regressor_names == ("task",)
    assert model.profiled_names == ()
    assert model.to_dict() == {"regressors": ["task"], "modulators": []}
    assert model == TaskModel(())
    assert hash(model) == hash(TaskModel())


def test_nsd_task_model_names_and_dict():
    model = TaskModel(
        (
            Modulator("response_time", center=True, missing="indicator"),
            Modulator("trial_type", center=False),
        )
    )
    assert model.regressor_names == ("task", "response_time", "trial_type")
    assert model.profiled_names == ("missing_response_time",)
    assert model.to_dict() == {
        "regressors": ["task", "response_time", "trial_type"],
        "modulators": [
            {"column": "response_time", "center": True, "missing": "indicator"},
            {"column": "trial_type", "center": False, "missing": "error"},
        ],
    }


def test_fingerprint_is_sha256_of_canonical_dict_and_changes_with_settings():
    model = TaskModel((Modulator("response_time"),))
    assert len(model.fingerprint) == 64
    assert int(model.fingerprint, 16) >= 0
    json.dumps(model.to_dict(), sort_keys=True)  # must be JSON serializable
    assert model.fingerprint == TaskModel((Modulator("response_time"),)).fingerprint
    assert model.fingerprint != TaskModel().fingerprint
    assert (
        model.fingerprint
        != TaskModel((Modulator("response_time", center=False),)).fingerprint
    )
    assert (
        model.fingerprint
        != TaskModel((Modulator("response_time", missing="indicator"),)).fingerprint
    )


@pytest.mark.parametrize(
    "column", ["task", "constant", "onset", "duration", "missing_rt", "", None, 3]
)
def test_reserved_or_invalid_modulator_columns_rejected(column):
    with pytest.raises(ValueError, match="reserved|nonempty"):
        Modulator(column)


def test_trial_type_is_an_ordinary_modulator_column():
    assert Modulator("trial_type", center=False).column == "trial_type"


@pytest.mark.parametrize("missing", ["drop", "", None, True])
def test_invalid_missing_policy_rejected(missing):
    with pytest.raises(ValueError, match="missing"):
        Modulator("response_time", missing=missing)


@pytest.mark.parametrize("center", [1, "yes", None])
def test_center_must_be_boolean(center):
    with pytest.raises(ValueError, match="center"):
        Modulator("response_time", center=center)


def test_duplicate_or_non_modulator_entries_rejected():
    with pytest.raises(ValueError, match="unique"):
        TaskModel((Modulator("rt"), Modulator("rt", center=False)))
    with pytest.raises(ValueError, match="Modulator"):
        TaskModel(("rt",))


def test_task_model_is_immutable_and_owns_its_tuple():
    modulators = [Modulator("rt")]
    model = TaskModel(modulators)
    modulators.append(Modulator("other"))
    assert model.regressor_names == ("task", "rt")
    with pytest.raises(AttributeError):
        model.modulators = ()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_task_model.py -q`
Expected: FAIL with `ImportError: cannot import name 'Modulator'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_task_model.py
git commit -m "test: specify TaskModel and Modulator task regressors

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 4: Implement the dataclasses**

Add to `src/boldtailor/model.py` after the existing imports (add `from hashlib import sha256` and `import json` at the top):

```python
_RESERVED_TASK_COLUMNS = frozenset({"task", "constant", "onset", "duration"})
_MISSING_POLICIES = ("error", "indicator")


@dataclass(frozen=True)
class Modulator:
    """One parametric task regressor derived from a raw events column."""

    column: str
    center: bool = True
    missing: str = "error"

    def __post_init__(self) -> None:
        _validate_modulator_column(self.column)
        if not isinstance(self.center, bool):
            raise ValueError("modulator center must be a boolean")
        if self.missing not in _MISSING_POLICIES:
            raise ValueError("modulator missing policy must be 'error' or 'indicator'")

    @property
    def indicator_name(self) -> str:
        return f"missing_{self.column}"

    def to_dict(self) -> dict[str, object]:
        return {"column": self.column, "center": self.center, "missing": self.missing}


@dataclass(frozen=True)
class TaskModel:
    """Task regressor plus modulators; the default is the task regressor alone."""

    modulators: tuple[Modulator, ...] = ()

    def __post_init__(self) -> None:
        modulators = tuple(self.modulators)
        if any(not isinstance(m, Modulator) for m in modulators):
            raise ValueError("task model modulators must be Modulator instances")
        columns = [m.column for m in modulators]
        if len(set(columns)) != len(columns):
            raise ValueError("modulator columns must be unique")
        object.__setattr__(self, "modulators", modulators)

    @property
    def regressor_names(self) -> tuple[str, ...]:
        return ("task", *(m.column for m in self.modulators))

    @property
    def profiled_names(self) -> tuple[str, ...]:
        return tuple(m.indicator_name for m in self.modulators if m.missing == "indicator")

    def to_dict(self) -> dict[str, object]:
        return {
            "regressors": list(self.regressor_names),
            "modulators": [m.to_dict() for m in self.modulators],
        }

    @property
    def fingerprint(self) -> str:
        return sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def _validate_modulator_column(column: object) -> None:
    if not isinstance(column, str) or not column:
        raise ValueError("modulator column must be a nonempty string")
    if column in _RESERVED_TASK_COLUMNS or column.startswith("missing_"):
        raise ValueError(f"modulator column {column!r} is reserved")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_task_model.py tests/test_model.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/boldtailor/model.py
git commit -m "feat: add TaskModel and Modulator task regressor specification

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Event expansion and Nilearn task columns

**Files:**
- Create: `src/boldtailor/_task_design.py`
- Test: `tests/test_task_design.py`

**Interfaces:**
- Consumes: `TaskModel`, `Modulator` from Task 1; `hrf_model(candidate)` from `boldtailor._hrf_design` (returns `"spm"` or the candidate's kernel callable).
- Produces:
  - `expand_events(events: pd.DataFrame, task_model: TaskModel, run: int = 0) -> pd.DataFrame` with columns `onset`, `duration`, `trial_type`, `modulation`; one row per trial per regressor; regressor order is `task_model.regressor_names` then present profiled names.
  - `task_columns(expanded: pd.DataFrame, frame_times, hrf, *, min_onset=-24.0, oversampling=50) -> pd.DataFrame` indexed by frame times, columns in the expanded frame's first-appearance order, no constant, no drift.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_task_design.py`:

```python
"""Raw trials expand to Nilearn conditions; Nilearn builds the task columns."""

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import compute_regressor

from boldtailor.hrf_library import HrfLibrary
from boldtailor.model import Modulator, TaskModel

NSD = TaskModel(
    (
        Modulator("response_time", center=True, missing="indicator"),
        Modulator("trial_type", center=False),
    )
)


@pytest.fixture
def events():
    return pd.DataFrame(
        dict(
            onset=[8.0, 22.0, 38.0, 60.0, 90.0, 112.0],
            duration=[3.0, 1.0, 2.0, 3.0, 1.5, 2.5],
            trial_type=[0, 1, 0, 1, 1, 0],
            response_time=[1.0, np.nan, 3.0, 2.0, 4.0, 5.0],
            image=[10, 11, 12, 13, 14, 15],
        )
    )


def test_expansion_orders_regressors_and_computes_amplitudes(events):
    from boldtailor._task_design import expand_events

    original = events.copy(deep=True)
    result = expand_events(events, NSD)
    expected = {
        "task": [1, 1, 1, 1, 1, 1],
        "response_time": [-2, 0, 0, -1, 1, 2],
        "trial_type": [0, 1, 0, 1, 1, 0],
        "missing_response_time": [0, 1, 0, 0, 0, 0],
    }
    assert list(dict.fromkeys(result.trial_type)) == list(expected)
    assert list(result.columns) == ["onset", "duration", "trial_type", "modulation"]
    assert len(result) == 4 * len(events)
    for name, amplitudes in expected.items():
        rows = result.loc[result.trial_type == name]
        np.testing.assert_allclose(rows.modulation, amplitudes)
        np.testing.assert_array_equal(
            rows[["onset", "duration"]], events[["onset", "duration"]]
        )
    pd.testing.assert_frame_equal(events, original)


def test_indicator_absent_when_nothing_is_missing(events):
    from boldtailor._task_design import expand_events

    complete = events.assign(response_time=[1.0, 2.0, 3.0, 2.0, 4.0, 5.0])
    result = expand_events(complete, NSD)
    assert set(result.trial_type) == {"task", "response_time", "trial_type"}
    rt = result.loc[result.trial_type == "response_time", "modulation"]
    np.testing.assert_allclose(rt, complete.response_time - complete.response_time.mean())


def test_task_only_model_expands_to_unit_task_rows(events):
    from boldtailor._task_design import expand_events

    result = expand_events(events[["onset", "duration"]], TaskModel())
    assert list(result.trial_type.unique()) == ["task"]
    np.testing.assert_array_equal(result.modulation, 1.0)


def test_uncentered_modulator_keeps_raw_values(events):
    from boldtailor._task_design import expand_events

    result = expand_events(events, TaskModel((Modulator("trial_type", center=False),)))
    np.testing.assert_array_equal(
        result.loc[result.trial_type == "trial_type", "modulation"], events.trial_type
    )


def test_error_policy_rejects_missing_values(events):
    from boldtailor._task_design import expand_events

    with pytest.raises(ValueError, match="run 2.*response_time.*finite"):
        expand_events(events, TaskModel((Modulator("response_time"),)), run=2)


def test_text_in_modulator_column_is_an_error_not_missing(events):
    from boldtailor._task_design import expand_events

    events["response_time"] = events.response_time.astype(object)
    events.loc[1, "response_time"] = "invalid"
    with pytest.raises(ValueError, match="response_time.*numeric"):
        expand_events(events, NSD)


def test_all_missing_indicator_modulator_is_rejected(events):
    from boldtailor._task_design import expand_events

    with pytest.raises(ValueError, match="response_time.*observed"):
        expand_events(events.assign(response_time=np.nan), NSD)


def test_regressor_without_nonzero_amplitude_is_rejected(events):
    from boldtailor._task_design import expand_events

    with pytest.raises(ValueError, match="run 1.*trial_type.*nonzero"):
        expand_events(events.assign(trial_type=0), NSD, run=1)
    with pytest.raises(ValueError, match="response_time.*nonzero"):
        expand_events(events.assign(response_time=2.0), NSD)


def test_missing_modulator_column_or_timing_is_rejected(events):
    from boldtailor._task_design import expand_events

    with pytest.raises(ValueError, match="trial_type"):
        expand_events(events.drop(columns="trial_type"), NSD)
    with pytest.raises(ValueError, match="onset"):
        expand_events(events.drop(columns="onset"), TaskModel())


@pytest.mark.parametrize("cid", [0, 1])
def test_task_columns_match_compute_regressor_per_condition(events, cid, capsys):
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import expand_events, task_columns

    library = HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])
    times = 0.775 + 1.6 * np.arange(90)
    expanded = expand_events(events, NSD)
    hrf = hrf_model(library.candidates[cid])
    result = task_columns(expanded, times, hrf, min_onset=-24.0, oversampling=50)
    assert list(result.columns) == [
        "task",
        "response_time",
        "trial_type",
        "missing_response_time",
    ]
    np.testing.assert_array_equal(result.index, times)
    for name in result.columns:
        rows = expanded.loc[expanded.trial_type == name]
        expected, _ = compute_regressor(
            rows[["onset", "duration", "modulation"]].to_numpy().T,
            hrf,
            times,
            oversampling=50,
            min_onset=-24.0,
        )
        np.testing.assert_allclose(result[name], expected[:, 0], atol=1e-13)
    assert capsys.readouterr().out == ""


def test_task_columns_accept_string_hrfs_and_honor_settings(events):
    from boldtailor._task_design import expand_events, task_columns

    times = 0.775 + 1.6 * np.arange(90)
    expanded = expand_events(events, TaskModel())
    spm = task_columns(expanded, times, "spm", min_onset=-10.0, oversampling=20)
    glover = task_columns(expanded, times, "glover", min_onset=-10.0, oversampling=20)
    expected, _ = compute_regressor(
        expanded[["onset", "duration", "modulation"]].to_numpy().T,
        "spm",
        times,
        oversampling=20,
        min_onset=-10.0,
    )
    np.testing.assert_allclose(spm["task"], expected[:, 0], atol=1e-13)
    assert not np.allclose(spm["task"], glover["task"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_task_design.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'boldtailor._task_design'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_task_design.py
git commit -m "test: specify task model event expansion and Nilearn task columns

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 4: Implement the module**

Create `src/boldtailor/_task_design.py`:

```python
"""Expand raw trials into Nilearn conditions and let Nilearn build the columns."""

from contextlib import redirect_stdout
import io

import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

from boldtailor.model import TaskModel


def _numeric(events, column, run):
    if column not in events:
        raise ValueError(f"run {run}: events lack modulator column {column!r}")
    try:
        return pd.to_numeric(events[column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"run {run}: modulator {column!r} must be numeric") from error


def _modulator_amplitudes(events, modulator, run):
    values = _numeric(events, modulator.column, run)
    observed = np.isfinite(values)
    if not observed.any():
        raise ValueError(f"run {run}: modulator {modulator.column!r} has no observed values")
    if modulator.missing == "error" and not observed.all():
        raise ValueError(f"run {run}: modulator {modulator.column!r} must be finite")
    amplitude = np.zeros(len(values))
    offset = values[observed].mean() if modulator.center else 0.0
    amplitude[observed] = values[observed] - offset
    columns = {modulator.column: amplitude}
    if modulator.missing == "indicator" and not observed.all():
        columns[modulator.indicator_name] = (~observed).astype(float)
    return columns


def expand_events(events, task_model, run=0):
    """One Nilearn condition per task-model regressor; amplitudes become modulation."""
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel")
    if not {"onset", "duration"}.issubset(events.columns):
        raise ValueError(f"run {run}: events require onset and duration columns")
    amplitudes = {"task": np.ones(len(events))}
    for modulator in task_model.modulators:
        amplitudes.update(_modulator_amplitudes(events, modulator, run))
    for name, values in amplitudes.items():
        if not np.any(values != 0):
            raise ValueError(f"run {run}: regressor {name!r} has no nonzero amplitude")
    timing = events[["onset", "duration"]].reset_index(drop=True)
    return pd.concat(
        [timing.assign(trial_type=name, modulation=values) for name, values in amplitudes.items()],
        ignore_index=True,
    )


def task_columns(expanded, frame_times, hrf, *, min_onset=-24.0, oversampling=50):
    """Nilearn task columns for one HRF: no drift, no constant, semantic names."""
    times = np.asarray(frame_times, dtype=float)
    with redirect_stdout(io.StringIO()):
        matrix = make_first_level_design_matrix(
            times,
            events=expanded,
            hrf_model=hrf,
            drift_model=None,
            min_onset=min_onset,
            oversampling=oversampling,
        )
    matrix = matrix.drop(columns="constant")
    matrix.columns = [name.removesuffix("_kernel") for name in matrix.columns]
    return matrix.loc[:, list(dict.fromkeys(expanded.trial_type))]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_task_design.py -q`
Expected: all PASS. If the suffix stripped by Nilearn for a bound `HrfCandidate.kernel` is not `_kernel`, inspect `matrix.columns` and fix the strip to whatever suffix Nilearn appends; do not change the test.

- [ ] **Step 6: Commit**

```bash
git add src/boldtailor/_task_design.py
git commit -m "feat: expand task model events and build Nilearn task columns

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: ModelSpec.task_model and canonical design compilation

**Files:**
- Modify: `src/boldtailor/model.py` (`ModelSpec`)
- Modify: `src/boldtailor/design.py` (`_compile_run`)
- Modify: `src/boldtailor/fit.py` (`_model_provenance`)
- Test: `tests/test_model.py`, `tests/test_design.py`, `tests/test_fit.py`

**Interfaces:**
- Consumes: `TaskModel`, `expand_events`, `task_columns`.
- Produces: `ModelSpec(..., task_model: TaskModel | None = None)`. `compile_designs(data, model)` with a task model returns task columns then the Nilearn nuisance matrix. `_model_provenance(model).activity["task_model"]` is present only when a task model is set.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_model.py`:

```python
def test_model_spec_accepts_task_model_with_spm_or_glover():
    from boldtailor.model import Modulator, TaskModel

    task_model = TaskModel((Modulator("response_time", missing="indicator"),))
    for hrf in ("spm", "glover"):
        model = ModelSpec(contrasts={"task": {"task": 1}}, hrf_model=hrf, task_model=task_model)
        assert model.task_model == task_model
    assert ModelSpec(contrasts={"task": {"task": 1}}).task_model is None


@pytest.mark.parametrize(
    "hrf_model", ["spm + derivative", "glover + derivative + dispersion", None, "fir"]
)
def test_task_model_requires_single_column_string_hrf(hrf_model):
    from boldtailor.model import TaskModel

    with pytest.raises(ValueError, match="task_model requires hrf_model 'spm' or 'glover'"):
        ModelSpec(contrasts={"task": {"task": 1}}, hrf_model=hrf_model, task_model=TaskModel())


def test_task_model_must_be_a_task_model():
    with pytest.raises(ValueError, match="TaskModel"):
        ModelSpec(contrasts={"task": {"task": 1}}, hrf_model="spm", task_model="nsd")
```

Append to `tests/test_design.py` (add the imports it needs at the top of the file: `import numpy as np`, `import pandas as pd`, `from nilearn.glm.first_level import make_first_level_design_matrix`, `from boldtailor.data import from_arrays`, `from boldtailor.design import compile_designs`, `from boldtailor.model import ModelSpec, Modulator, TaskModel`):

```python
def _task_model_data():
    times = [0.775 + 1.6 * np.arange(80), 0.775 + 1.6 * np.arange(85)]
    events = [
        pd.DataFrame(
            dict(
                onset=[8.0, 22.0, 38.0, 60.0, 90.0],
                duration=[3.0, 1.0, 2.0, 3.0, 1.5],
                trial_type=[0, 1, 0, 1, 1],
                response_time=[1.0, np.nan if r else 2.0, 3.0, 2.0, 4.0],
            )
        )
        for r in range(2)
    ]
    confounds = [pd.DataFrame(dict(motion=np.linspace(-1, 1, len(t)))) for t in times]
    signals = [np.random.default_rng(r).normal(size=(len(t), 3)) for r, t in enumerate(times)]
    return from_arrays(signals, events, frame_times=times, confounds=confounds)


def test_compile_designs_with_task_model_uses_nilearn_task_columns_and_nuisance():
    from boldtailor._task_design import expand_events

    data = _task_model_data()
    task_model = TaskModel(
        (Modulator("response_time", missing="indicator"), Modulator("trial_type", center=False))
    )
    model = ModelSpec(
        contrasts={"task": {"task": 1}},
        confounds=("motion",),
        hrf_model="spm",
        drift_model="cosine",
        high_pass=0.01,
        task_model=task_model,
    )
    compiled = compile_designs(data, model)
    expected_names = [
        ["task", "response_time", "trial_type"],
        ["task", "response_time", "trial_type", "missing_response_time"],
    ]
    for run, design in enumerate(compiled):
        expected = make_first_level_design_matrix(
            data.frame_times[run],
            events=expand_events(data.events[run], task_model, run),
            hrf_model="spm",
            drift_model="cosine",
            high_pass=0.01,
            add_regs=data.confounds[run][["motion"]],
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
        assert list(design.matrix.columns[: len(expected_names[run])]) == expected_names[run]
        assert "constant" in design.matrix.columns
        assert any(c.startswith("drift") for c in design.matrix.columns)
        for name in expected_names[run]:
            np.testing.assert_allclose(design.matrix[name], expected[name], atol=1e-12)
        np.testing.assert_allclose(design.matrix["motion"], expected["motion"], atol=1e-12)


def test_compile_designs_with_task_model_reports_data_errors_with_run():
    data = _task_model_data()
    model = ModelSpec(
        contrasts={"task": {"task": 1}},
        hrf_model="spm",
        task_model=TaskModel((Modulator("response_time"),)),
    )
    with pytest.raises(ValueError, match="run 1.*response_time"):
        compile_designs(data, model)
```

Append to `tests/test_fit.py` (uses the existing imports of that file plus `from boldtailor.model import Modulator, TaskModel`):

```python
def test_model_provenance_adds_task_model_only_when_set():
    from boldtailor.fit import _model_provenance

    plain = ModelSpec(contrasts={"task": {"task": 1}}, hrf_model="spm")
    assert "task_model" not in _model_provenance(plain).activity
    task_model = TaskModel((Modulator("response_time", missing="indicator"),))
    with_model = ModelSpec(contrasts={"task": {"task": 1}}, hrf_model="spm", task_model=task_model)
    activity = _model_provenance(with_model).activity
    assert activity["task_model"] == task_model.to_dict()
    assert activity["task_model_fingerprint"] == task_model.fingerprint
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_model.py tests/test_design.py tests/test_fit.py -q -k "task_model"`
Expected: FAIL with `TypeError: ModelSpec.__init__() got an unexpected keyword argument 'task_model'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_model.py tests/test_design.py tests/test_fit.py
git commit -m "test: specify ModelSpec.task_model and canonical task-model designs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 4: Add the field and validation to ModelSpec**

In `src/boldtailor/model.py`, add the field after `noise_model` and before `_contrast_names`:

```python
    noise_model: str = "ar1"
    task_model: TaskModel | None = None
    _contrast_names: tuple[str, ...] = field(init=False, repr=False)
```

In `__post_init__`, after `_validate_noise_model(self.noise_model)` add:

```python
        _validate_task_model(self.task_model, self.hrf_model)
```

Add the validator near the other validators:

```python
def _validate_task_model(task_model: object, hrf_model: HRFModel) -> None:
    if task_model is None:
        return
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel or None")
    if hrf_model not in ("spm", "glover"):
        raise ValueError("task_model requires hrf_model 'spm' or 'glover'")
```

`TaskModel` and `Modulator` must be defined above `ModelSpec` in the file; move them if Task 1 placed them below.

- [ ] **Step 5: Branch design compilation**

In `src/boldtailor/design.py`, add `from boldtailor._task_design import expand_events, task_columns` and change `_compile_run` so the design line reads:

```python
    if model.task_model is None:
        design = _make_design_matrix(frame_times, modeled_events, selected, model, run)
    else:
        design = _make_task_model_design(frame_times, modeled_events, selected, model, run)
```

Add:

```python
def _make_task_model_design(
    frame_times: np.ndarray,
    events: pd.DataFrame,
    confounds: pd.DataFrame | None,
    model: ModelSpec,
    run: int,
) -> pd.DataFrame:
    try:
        task = task_columns(
            expand_events(events, model.task_model, run),
            frame_times,
            model.hrf_model,
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
    except ValueError as error:
        raise ValueError(f"run {run} design compilation failed: {error}") from error
    nuisance = _make_nuisance_matrix(frame_times, confounds, model, run)
    return pd.concat([task, nuisance], axis=1)
```

- [ ] **Step 6: Record the task model in model provenance**

In `src/boldtailor/fit.py` `_model_provenance`, after building `activity` and before `fingerprint = None`, add:

```python
    if model.task_model is not None:
        activity["task_model"] = model.task_model.to_dict()
        activity["task_model_fingerprint"] = model.task_model.fingerprint
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_model.py tests/test_design.py tests/test_fit.py -q`
Expected: all PASS, including the pre-existing tests in those files.

- [ ] **Step 8: Commit**

```bash
git add src/boldtailor/model.py src/boldtailor/design.py src/boldtailor/fit.py
git commit -m "feat: compile canonical designs from ModelSpec.task_model

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Task-model sufficient statistics in the run design cache

**Files:**
- Modify: `src/boldtailor/_hrf_cv.py`
- Modify: `examples/NSD/nsd_hrf.py:189,205` (rename `eligible(0)` to `trial_eligible(0)`)
- Test: `tests/test_hrf_cv.py` (new), `tests/test_hrf_selection.py` (existing tests must still pass unchanged)

**Interfaces:**
- Consumes: `TaskModel`, `expand_events`, `task_columns`, `hrf_model`.
- Produces, all in `boldtailor._hrf_cv`:
  - `MIN_ONSET = -24.0`, `OVERSAMPLING = 50`.
  - `prepare_runs(data, library, task_model=TaskModel()) -> tuple[RunDesign, ...]`.
  - `RunDesign.k`, `RunDesign.task_model`, `RunDesign.events` (task rows, onset/duration), `RunDesign.q` (confound basis), `RunDesign.task_design(cid) -> pd.DataFrame`, `RunDesign.block(cid) -> _Block(x, qp, a)`, `RunDesign.eligible(cid) -> (bool, str)` for the task-model design, `RunDesign.trial_eligible(cid) -> (bool, str)` for the single-trial design, `RunDesign.trial_matrix(cid)` unchanged.
  - `signal_statistics(runs, signals, batch_size) -> (a, b, c, energy)` with shapes `(R,H,K,K)`, `(R,H,K,F)`, `(R,H,F)`, `(R,F)`.
  - `pooled_amplitude(a_sum, b_sum) -> (amplitude (H,K,F), ok (H,) bool)`.
  - `prediction_loss(a, b, c, amplitude) -> (H,F)` for one run.
  - `loro_scores(a, b, c, energy) -> (H,F)`.
  - `choose_eligible(scores, runs)` unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hrf_cv.py`:

```python
"""Task-model statistics reduce to the mean-stimulus method and match stacked OLS."""

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import run_glm
from scipy.linalg import block_diag

from boldtailor.data import from_arrays
from boldtailor.hrf_library import HrfLibrary
from boldtailor.model import Modulator, TaskModel

NSD = TaskModel(
    (
        Modulator("response_time", center=True, missing="indicator"),
        Modulator("trial_type", center=False),
    )
)


@pytest.fixture
def task_fixture():
    """Four runs; runs 1 and 3 have one missing RT; three modulated features."""
    library = HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )
    rng = np.random.default_rng(93)
    signals, events, times, confounds = [], [], [], []
    for r, length in enumerate([70, 82, 76, 90]):
        t = 0.774 + 1.6 * np.arange(length)
        rt = np.array([0.7, 1.4, 1.1, 0.9, 1.6])
        if r % 2:
            rt[2] = np.nan
        e = pd.DataFrame(
            dict(
                onset=np.array([8.1, 25.4, 45.2, 66.3, 88.0]) + r,
                duration=[3.0, 1.2, 2.0, 1.5, 2.2],
                trial_type=[0, 1, 1, 0, 1],
                response_time=rt,
            )
        )
        n = pd.DataFrame(dict(motion=np.sin(np.arange(length) / 11 + r)))
        events.append(e)
        times.append(t)
        confounds.append(n)
        signals.append(rng.normal(size=(length, 3)) * 0.02 + 50)
    data = from_arrays(signals, events, frame_times=times, confounds=confounds)
    return data, library


def columns_for(data, library, cid, task_model):
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import expand_events, task_columns

    out = []
    for r, (e, t) in enumerate(zip(data.events, data.frame_times, strict=True)):
        frame = task_columns(
            expand_events(e, task_model, r), t, hrf_model(library.candidates[cid])
        )
        out.append(frame)
    return out


def generated(data, library, cid, task_model, amplitudes, profiled_gain=2.0):
    """Noise-free signals from the task model plus run-specific nuisance."""
    signals = []
    for r, frame in enumerate(columns_for(data, library, cid, task_model)):
        x = frame[list(task_model.regressor_names)].to_numpy() @ np.asarray(amplitudes)
        for name in task_model.profiled_names:
            if name in frame:
                x = x + profiled_gain * (r + 1) * frame[name].to_numpy()
        n = data.confounds[r].to_numpy()[:, 0]
        signals.append(np.column_stack([x + 3 * n + 40 + r, 2 * x - n + 9]))
    return from_arrays(
        signals, data.events, frame_times=data.frame_times, confounds=data.confounds
    )


def stacked_oracle(data, library, cid, task_model, train, test):
    """Nilearn OLS on a stacked training design, then frozen held-out prediction."""
    frames = columns_for(data, library, cid, task_model)
    xs, zs, ns = [], [], []
    for r, frame in enumerate(frames):
        xs.append(frame[list(task_model.regressor_names)].to_numpy())
        n = np.column_stack([data.confounds[r].to_numpy(), np.ones(len(frame))])
        profiled = [c for c in task_model.profiled_names if c in frame]
        zs.append(np.column_stack([n, frame[profiled].to_numpy()]) if profiled else n)
        ns.append(n)
    design = np.column_stack(
        [np.concatenate([xs[r] for r in train]), block_diag(*[zs[r] for r in train])]
    )
    y = np.concatenate([data.signals[r] for r in train])
    labels, results = run_glm(y, design, noise_model="ols")
    beta = results[labels[0]].theta[: xs[0].shape[1]]
    loss, null = np.zeros(data.n_features), np.zeros(data.n_features)
    for r in test:
        residual = data.signals[r] - xs[r] @ beta
        residual = residual - zs[r] @ np.linalg.lstsq(zs[r], residual, rcond=None)[0]
        yr = data.signals[r] - ns[r] @ np.linalg.lstsq(ns[r], data.signals[r], rcond=None)[0]
        loss += np.sum(residual**2, axis=0)
        null += np.sum(yr**2, axis=0)
    return beta, loss, null


def loro_oracle(data, library, task_model):
    scores = []
    for cid in range(len(library.candidates)):
        folds = [
            stacked_oracle(
                data, library, cid, task_model, [i for i in range(data.n_runs) if i != r], [r]
            )
            for r in range(data.n_runs)
        ]
        scores.append(1 - sum(f[1] for f in folds) / sum(f[2] for f in folds))
    return np.array(scores)


def test_statistics_shapes_and_profiled_columns(task_fixture):
    from boldtailor._hrf_cv import prepare_runs, signal_statistics

    data, library = task_fixture
    runs = prepare_runs(data, library, NSD)
    assert all(run.k == 3 for run in runs)
    assert [run.block(1).qp.shape[1] for run in runs] == [0, 1, 0, 1]
    assert list(runs[1].task_design(1).columns) == [
        "task",
        "response_time",
        "trial_type",
        "missing_response_time",
    ]
    a, b, c, energy = signal_statistics(runs, data.signals, 2)
    assert a.shape == (4, 3, 3, 3)
    assert b.shape == (4, 3, 3, 3)
    assert c.shape == (4, 3, 3)
    assert energy.shape == (4, 3)
    # Profiled columns only lower the candidate's C in runs that have them.
    np.testing.assert_allclose(c[0], energy[0][None])
    assert np.all(c[1] < energy[1][None])
    for run in runs:
        np.testing.assert_allclose(run.block(1).a, run.block(1).x.T @ run.block(1).x)
        assert np.allclose(run.q.T @ run.block(1).x, 0, atol=1e-12)
        assert np.allclose(run.block(1).qp.T @ run.block(1).x, 0, atol=1e-12)


@pytest.mark.parametrize("batch", [1, 2, 32])
def test_task_model_loro_matches_stacked_nilearn_ols(task_fixture, batch):
    from boldtailor._hrf_cv import loro_scores, prepare_runs, signal_statistics

    data, library = task_fixture
    runs = prepare_runs(data, library, NSD)
    a, b, c, energy = signal_statistics(runs, data.signals, batch)
    scores = loro_scores(a, b, c, energy)
    np.testing.assert_allclose(scores, loro_oracle(data, library, NSD), atol=1e-10)


def test_task_only_model_reproduces_mean_stimulus_statistics(task_fixture):
    from boldtailor._hrf_cv import loro_scores, prepare_runs, signal_statistics
    from tests.test_hrf_selection import oracle_cv

    data, library = task_fixture
    runs = prepare_runs(data, library)
    a, b, c, energy = signal_statistics(runs, data.signals, 32)
    assert a.shape[-2:] == (1, 1)
    np.testing.assert_allclose(c, energy[:, None, :])
    np.testing.assert_allclose(
        loro_scores(a, b, c, energy), oracle_cv(data, library), atol=1e-12
    )


def test_noise_free_task_model_scores_one_only_with_the_full_model(task_fixture):
    from boldtailor._hrf_cv import loro_scores, prepare_runs, signal_statistics

    data, library = task_fixture
    clean = generated(data, library, 2, NSD, amplitudes=[1.0, 4.0, -2.5])
    runs = prepare_runs(clean, library, NSD)
    full = loro_scores(*signal_statistics(runs, clean.signals, 32))
    np.testing.assert_allclose(full[2], 1, atol=1e-10)
    assert np.all(full[[0, 1]] < 1 - 1e-6)
    task_only = loro_scores(
        *signal_statistics(prepare_runs(clean, library), clean.signals, 32)
    )
    assert np.all(task_only[2] < 1 - 1e-6)


def test_singular_pooled_training_matrix_marks_candidate_ineligible():
    from boldtailor._hrf_cv import loro_scores, pooled_amplitude

    a = np.zeros((3, 2, 2, 2))
    b = np.zeros((3, 2, 2, 4))
    c = np.ones((3, 2, 4))
    energy = np.ones((3, 4))
    a[:, 0] = np.eye(2)
    a[:, 1] = [[1, 1], [1, 1]]
    b[:, 0] = 0.5
    b[:, 1] = 0.5
    amplitude, ok = pooled_amplitude(a[:2].sum(0), b[:2].sum(0))
    np.testing.assert_array_equal(ok, [True, False])
    np.testing.assert_allclose(amplitude[0], 0.5)
    np.testing.assert_array_equal(amplitude[1], 0)
    scores = loro_scores(a, b, c, energy)
    assert np.all(np.isfinite(scores[0]))
    assert np.all(scores[1] == -np.inf)


def test_run_level_rank_failures_are_candidate_specific(task_fixture):
    from boldtailor._hrf_cv import prepare_runs

    data, library = task_fixture
    frames = columns_for(data, library, 0, NSD)
    confounds = [
        n.assign(null_task=f["task"].to_numpy()) if r == 2 else n
        for r, (n, f) in enumerate(zip(data.confounds, frames, strict=True))
    ]
    altered = from_arrays(
        data.signals, data.events, frame_times=data.frame_times, confounds=confounds
    )
    runs = prepare_runs(altered, library, NSD)
    ok, reason = runs[2].eligible(0)
    assert not ok and "support" in reason
    assert runs[2].eligible(1) == (True, "")
    assert runs[0].eligible(0) == (True, "")
    assert np.all(runs[2].block(0).x == 0)
    assert np.all(runs[2].block(0).a == 0)


def test_profiled_column_inside_nuisance_span_is_ineligible(task_fixture):
    from boldtailor._hrf_cv import prepare_runs

    data, library = task_fixture
    frames = columns_for(data, library, 1, NSD)
    confounds = list(data.confounds)
    confounds[1] = confounds[1].assign(
        indicator=frames[1]["missing_response_time"].to_numpy()
    )
    altered = from_arrays(
        data.signals, data.events, frame_times=data.frame_times, confounds=confounds
    )
    ok, reason = prepare_runs(altered, library, NSD)[1].eligible(1)
    assert not ok and "profiled" in reason


def test_trial_eligibility_is_separate_from_task_model_eligibility(task_fixture):
    from boldtailor._hrf_cv import prepare_runs

    data, library = task_fixture
    run = prepare_runs(data, library, NSD)[0]
    assert run.trial_eligible(1) == (True, "")
    assert run.trial_matrix(1).shape == (len(data.frame_times[0]), 5)
    assert len(run.events) == 5


def test_cache_key_includes_task_model_and_modulator_values(task_fixture):
    from boldtailor._hrf_cv import prepare_runs

    data, library = task_fixture
    base = prepare_runs(data, library, NSD)[0]
    assert prepare_runs(data, library, NSD)[0] is base
    assert prepare_runs(data, library)[0] is not base
    assert prepare_runs(data, library)[0].fingerprint != base.fingerprint
    events = list(data.events)
    events[0] = events[0].assign(response_time=[0.9, 1.4, 1.1, 0.9, 1.6])
    changed = from_arrays(
        data.signals, events, frame_times=data.frame_times, confounds=data.confounds
    )
    assert prepare_runs(changed, library, NSD)[0].fingerprint != base.fingerprint
    assert (
        prepare_runs(changed, library)[0].fingerprint
        == prepare_runs(data, library)[0].fingerprint
    )


def test_onsets_outside_supported_window_are_rejected(task_fixture):
    from boldtailor._hrf_cv import prepare_runs

    data, library = task_fixture
    events = list(data.events)
    events[0] = events[0].assign(onset=events[0].onset - 40)
    early = from_arrays(
        data.signals, events, frame_times=data.frame_times, confounds=data.confounds
    )
    with pytest.raises(ValueError, match="supported sampled response"):
        prepare_runs(early, library, NSD)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_hrf_cv.py -q`
Expected: FAIL. The first failures are `TypeError: prepare_runs() takes 2 positional arguments but 3 were given` and `ImportError: cannot import name 'pooled_amplitude'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_hrf_cv.py
git commit -m "test: specify task-model sufficient statistics for HRF selection

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 4: Rewrite `_hrf_cv.py`**

Replace the contents of `src/boldtailor/_hrf_cv.py` above `choose_eligible` with the following; keep `choose_eligible` exactly as it is.

```python
"""Nuisance-projected sufficient statistics and data-independent design caches."""

from collections import namedtuple
from functools import lru_cache
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from boldtailor._hrf_design import hrf_model
from boldtailor._single_trial_design import (
    _nuisance_matrix,
    _validate_events,
    compile_trial_run,
)
from boldtailor._single_trial_fit import _project_design
from boldtailor._task_design import expand_events, task_columns
from boldtailor.model import TaskModel

MIN_ONSET = -24.0
OVERSAMPLING = 50

_Block = namedtuple("_Block", ["x", "qp", "a"])


def _basis(columns):
    u, s, _ = np.linalg.svd(columns, full_matrices=False)
    return u[:, s > s[0] * max(columns.shape) * np.finfo(float).eps]


def _profiled_basis(p, q):
    """Orthonormal basis of profiled columns after confound projection."""
    if p.shape[1] == 0:
        return np.zeros((len(q), 0))
    pr = p - q @ (q.T @ p)
    tolerance = np.linalg.norm(p, axis=0) * len(p) * np.finfo(float).eps
    if np.any(np.linalg.norm(pr, axis=0) <= tolerance):
        raise ValueError("profiled column has no support outside nuisance span")
    basis = _basis(pr)
    if basis.shape[1] != p.shape[1]:
        raise ValueError("profiled columns are rank deficient")
    return basis


def _check_task_rank(xr, x, dof):
    scale = np.linalg.norm(xr, axis=0)
    tolerance = np.linalg.norm(x, ord=2) * max(x.shape) * np.finfo(float).eps
    if np.any(scale <= tolerance):
        raise ValueError("task column has no support outside nuisance span")
    if _basis(xr / scale).shape[1] != x.shape[1]:
        raise ValueError("task columns are rank deficient after nuisance projection")
    if dof - x.shape[1] <= 0:
        raise ValueError("task model requires positive residual degrees of freedom")


class RunDesign:
    """Cached timing, modulator amplitudes, and confounds: never retains BOLD."""

    def __init__(self, expanded, times, nuisance, library, task_model, fingerprint):
        self.expanded = expanded
        task_rows = expanded.trial_type == "task"
        self.events = expanded.loc[task_rows, ["onset", "duration"]].reset_index(drop=True)
        self.times, self.nuisance, self.library = times, nuisance, library
        self.task_model, self.fingerprint = task_model, fingerprint
        self.q = _basis(nuisance)
        self.k = len(task_model.regressor_names)
        self.eligibility, self.trial_eligibility, self._blocks = {}, {}, {}

    def task_design(self, candidate_id):
        """Nilearn task and profiled columns for one candidate on this run."""
        return task_columns(
            self.expanded,
            self.times,
            hrf_model(self.library.candidates[candidate_id]),
            min_onset=MIN_ONSET,
            oversampling=OVERSAMPLING,
        )

    def _build(self, candidate_id):
        columns = self.task_design(candidate_id)
        x = columns[list(self.task_model.regressor_names)].to_numpy()
        profiled = [c for c in self.task_model.profiled_names if c in columns]
        qp = _profiled_basis(columns[profiled].to_numpy(), self.q)
        q = np.column_stack([self.q, qp])
        xr = x - q @ (q.T @ x)
        _check_task_rank(xr, x, len(self.times) - q.shape[1])
        return _Block(xr, qp, xr.T @ xr)

    def block(self, candidate_id):
        """Projected task columns, profiled basis, and A; zeros when ineligible."""
        if candidate_id not in self._blocks:
            try:
                self._blocks[candidate_id] = self._build(candidate_id)
                self.eligibility[candidate_id] = (True, "")
            except ValueError as error:
                self.eligibility[candidate_id] = (False, str(error))
                empty = np.zeros((len(self.times), self.k))
                self._blocks[candidate_id] = _Block(
                    empty, np.zeros((len(self.times), 0)), np.zeros((self.k, self.k))
                )
        return self._blocks[candidate_id]

    def eligible(self, candidate_id):
        self.block(candidate_id)
        return self.eligibility[candidate_id]

    @lru_cache(maxsize=256)
    def trial_matrix(self, candidate_id):
        confounds = pd.DataFrame(self.nuisance[:, :-1])
        confounds.columns = [f"n{i}" for i in range(confounds.shape[1])]
        x, _, _ = compile_trial_run(
            self.events,
            self.times,
            confounds,
            "design",
            hrf=self.library.candidates[candidate_id],
        )
        values = x.to_numpy()
        values.setflags(write=False)
        return values

    def trial_eligible(self, candidate_id):
        """Estimability of the single-trial design; not used by selection."""
        if candidate_id not in self.trial_eligibility:
            try:
                _project_design(self.trial_matrix(candidate_id), self.nuisance)
                self.trial_eligibility[candidate_id] = (True, "")
            except ValueError as error:
                self.trial_eligibility[candidate_id] = (False, str(error))
        return self.trial_eligibility[candidate_id]


@lru_cache(maxsize=32)
def _cached_design(
    expanded_bytes, names_bytes, times_bytes, nuisance_bytes, n_columns, library, task_model
):
    values = np.frombuffer(expanded_bytes, dtype="<f8").reshape(-1, 3)
    expanded = pd.DataFrame(values, columns=["onset", "duration", "modulation"])
    expanded = expanded.assign(trial_type=json.loads(names_bytes))
    times = np.frombuffer(times_bytes, dtype="<f8")
    nuisance = np.frombuffer(nuisance_bytes, dtype="<f8").reshape(len(times), n_columns)
    digest = sha256(
        expanded_bytes
        + names_bytes
        + times_bytes
        + nuisance_bytes
        + str(n_columns).encode()
        + library.fingerprint.encode()
        + task_model.fingerprint.encode()
    ).hexdigest()
    return RunDesign(expanded, times, nuisance, library, task_model, digest)


def _validate_onsets(events, times):
    onsets = events["onset"].to_numpy(dtype=float)
    if np.any(onsets < times[0] + MIN_ONSET) or np.any(onsets >= times[-1]):
        raise ValueError("onset has no supported sampled response")


def prepare_runs(data, library, task_model=TaskModel()):
    runs = []
    for r, (events, times, confounds) in enumerate(
        zip(data.events, data.frame_times, data.confounds, strict=True)
    ):
        times = np.asarray(times, dtype=float)
        _validate_events(events, times, f"run-{r}")
        _validate_onsets(events, times)
        expanded = expand_events(events, task_model, r)
        nuisance = _nuisance_matrix(confounds, len(times))
        payloads = (
            np.asarray(expanded[["onset", "duration", "modulation"]], dtype="<f8").tobytes(),
            json.dumps(list(expanded.trial_type)).encode(),
            np.asarray(times, dtype="<f8").tobytes(),
            np.asarray(nuisance, dtype="<f8").tobytes(),
        )
        runs.append(_cached_design(*payloads, nuisance.shape[1], library, task_model))
    return tuple(runs)


def _batched_products(run, yr, batch_size):
    """B for every candidate as (H, K, F), multiplying batch_size candidates at once."""
    n = len(run.library.candidates)
    b = np.empty((n, run.k, yr.shape[1]))
    for start in range(0, n, batch_size):
        ids = range(start, min(start + batch_size, n))
        stacked = np.concatenate([run.block(cid).x for cid in ids], axis=1)
        b[start : start + len(ids)] = (stacked.T @ yr).reshape(len(ids), run.k, -1)
    return b


def signal_statistics(runs, signals, batch_size):
    """Project Y once per run; form A, B, C per candidate and confound-only energy."""
    a_all, b_all, c_all, energy = [], [], [], []
    for run, y in zip(runs, signals, strict=True):
        yr = y - run.q @ (run.q.T @ y)
        # Numerical zero only: no statistical threshold or weak-signal fallback.
        tolerance = np.linalg.norm(y, axis=0) * len(y) * np.finfo(float).eps
        yr[:, np.linalg.norm(yr, axis=0) <= tolerance] = 0
        c0 = np.sum(yr**2, axis=0)
        n = len(run.library.candidates)
        a = np.stack([run.block(cid).a for cid in range(n)])
        c = np.tile(c0, (n, 1))
        for cid in range(n):
            qp = run.block(cid).qp
            if qp.shape[1]:
                c[cid] -= np.sum((qp.T @ yr) ** 2, axis=0)
        a_all.append(a)
        b_all.append(_batched_products(run, yr, batch_size))
        c_all.append(c)
        energy.append(c0)
    return np.array(a_all), np.array(b_all), np.array(c_all), np.array(energy)


def pooled_amplitude(a_sum, b_sum):
    """Solve pooled normal equations per candidate; singular ones get zeros and False."""
    s = np.linalg.svd(a_sum, compute_uv=False)
    ok = s[:, -1] > s[:, 0] * a_sum.shape[-1] * np.finfo(float).eps
    amplitude = np.zeros_like(b_sum)
    if ok.any():
        amplitude[ok] = np.linalg.solve(a_sum[ok], b_sum[ok])
    return amplitude, ok


def prediction_loss(a, b, c, amplitude):
    cross = 2 * np.einsum("hkf,hkf->hf", amplitude, b)
    fitted = np.einsum("hkf,hkj,hjf->hf", amplitude, a, amplitude)
    loss = c - cross + fitted
    tolerance = 64 * np.finfo(float).eps * (c + np.abs(cross) + np.abs(fitted))
    if np.any(loss < -tolerance):
        raise ArithmeticError("negative HRF prediction SSE beyond roundoff tolerance")
    return np.maximum(loss, 0)


def loro_scores(a, b, c, energy):
    loss = np.zeros(c.shape[1:])
    eligible = np.ones(c.shape[1], dtype=bool)
    for r in range(len(a)):
        others = [i for i in range(len(a)) if i != r]
        # Explicit sums avoid cancellation when one run has much larger energy.
        amplitude, ok = pooled_amplitude(a[others].sum(axis=0), b[others].sum(axis=0))
        eligible &= ok
        loss += prediction_loss(a[r], b[r], c[r], amplitude)
    total = energy.sum(axis=0)
    scores = np.full_like(loss, np.nan)
    np.divide(loss, total[None, :], out=scores, where=total[None, :] > 0)
    scores = 1 - scores
    scores[~eligible] = -np.inf
    return scores
```

- [ ] **Step 5: Update the two callers that still expect the old API**

In `src/boldtailor/hrf_selection.py`, temporarily make the existing callers compile against the new return shapes so the pre-existing selection tests keep passing before Task 5 adds the public `task_model` argument:

- In `_select`: `a, b, c, energy = signal_statistics(runs, signals, batch)` and `loro_scores(a, b, c, energy)`.
- In `evaluate_hrf_split`: `a, b, c, energy = signal_statistics(runs, data.signals, 32)` and `_predict(a, b, c, energy, train, test)`.
- Replace `_predict` with:

```python
def _predict(a, b, c, energy, train, test):
    amplitude, ok = pooled_amplitude(a[list(train)].sum(axis=0), b[list(train)].sum(axis=0))
    loss = sum(prediction_loss(a[r], b[r], c[r], amplitude) for r in test)
    total = energy[list(test)].sum(axis=0)
    score = np.full_like(loss, np.nan)
    np.divide(loss, total[None, :], out=score, where=total[None, :] > 0)
    score = 1 - score
    score[~ok] = np.nan
    return amplitude[:, 0, :], score
```

and import `pooled_amplitude` from `boldtailor._hrf_cv`. The `[:, 0, :]` keeps the K=1 shape until Task 5 changes the result schema.

In `examples/NSD/nsd_hrf.py`, change both `design.eligible(0)` calls (inside `_canonical_rt` and `_canonical_comparisons`) to `design.trial_eligible(0)`.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_hrf_cv.py tests/test_hrf_selection.py tests/test_selected_hrf_fit.py tests/test_ridge_cv.py tests/test_fractional_cv.py tests/test_hrf_glm.py -q`
Expected: all PASS. The pre-existing selection oracles are the task-only regression guarantee.

- [ ] **Step 7: Commit**

```bash
git add src/boldtailor/_hrf_cv.py src/boldtailor/hrf_selection.py examples/NSD/nsd_hrf.py
git commit -m "feat: score task-model designs with pooled multi-regressor statistics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Public selection API with a task model

**Files:**
- Modify: `src/boldtailor/hrf_selection.py`
- Modify: `src/boldtailor/hrf_results.py`
- Modify: `tests/test_hrf_selection.py:139-146` (amplitude indexing reflects the new 2-D shape)
- Test: `tests/test_hrf_selection.py`

**Interfaces:**
- Consumes: everything in Task 4.
- Produces:
  - `select_hrf(data, *, library, run_labels=None, feature_signature=None, candidate_batch_size=32, task_model=TaskModel())`.
  - `evaluate_hrf_split(data, *, library, train_runs, test_runs, run_labels=None, feature_signature=None, task_model=TaskModel())`.
  - `HrfSelectionResult.task_model: TaskModel`.
  - `HrfEvaluationResult.training_amplitudes` shape `(K, n_features)`; `HrfEvaluationResult.amplitude_names: tuple[str, ...]`.
  - Selection provenance activity keys: `task_model`, `task_model_fingerprint`, `task_regressors`, `profiled_regressors`, `min_onset`, `oversampling`, `score="nuisance_adjusted_task_model_prediction_r2"`, `nuisance="conditional_projection_of_confounds_and_profiled_task_columns_per_run"`.

- [ ] **Step 1: Update the two existing assertions that index amplitudes**

In `tests/test_hrf_selection.py`, in `test_outer_test_changes_cannot_select_the_hrf`, change

```python
        np.testing.assert_allclose(original.training_amplitudes[v], beta[v], atol=1e-12)
```

to

```python
        np.testing.assert_allclose(original.training_amplitudes[0, v], beta[v], atol=1e-12)
```

This reflects the schema change in the spec, not a weakening.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_hrf_selection.py`:

```python
def nsd_model():
    from boldtailor.model import Modulator, TaskModel

    return TaskModel(
        (
            Modulator("response_time", center=True, missing="indicator"),
            Modulator("trial_type", center=False),
        )
    )


def with_trial_types(data):
    events = [e.assign(trial_type=[0, 1, 1]) for e in data.events]
    return replace_data(data, events=events)


def test_select_hrf_with_task_model_matches_task_model_oracle(cv_fixture):
    from boldtailor.hrf_selection import select_hrf
    from tests.test_hrf_cv import loro_oracle

    data, library = cv_fixture
    data = with_trial_types(data)
    result = select_hrf(data, library=library, task_model=nsd_model())
    expected = loro_oracle(data, library, nsd_model())[:, :3]
    np.testing.assert_allclose(result.cv_r2[:3], expected.max(axis=0), atol=1e-10)
    np.testing.assert_allclose(result.canonical_cv_r2[:3], expected[0], atol=1e-10)
    assert result.task_model == nsd_model()
    activity = result.provenance.to_dict()["activities"][-1]
    assert activity["task_model"] == nsd_model().to_dict()
    assert activity["task_model_fingerprint"] == nsd_model().fingerprint
    assert activity["task_regressors"] == ["task", "response_time", "trial_type"]
    assert activity["profiled_regressors"] == ["missing_response_time"]
    assert activity["min_onset"] == -24.0
    assert activity["oversampling"] == 50
    assert activity["score"] == "nuisance_adjusted_task_model_prediction_r2"


def test_default_selection_carries_task_only_model_and_legacy_fingerprint_rules(cv_fixture):
    from boldtailor.hrf_selection import select_hrf
    from boldtailor.model import TaskModel

    data, library = cv_fixture
    result = select_hrf(data, library=library)
    assert result.task_model == TaskModel()
    activity = result.provenance.to_dict()["activities"][-1]
    assert activity["task_regressors"] == ["task"]
    assert activity["profiled_regressors"] == []


def test_task_model_changes_selection_identity_and_rt_now_matters(cv_fixture):
    from boldtailor.hrf_selection import select_hrf

    data, library = cv_fixture
    data = with_trial_types(data)
    key = lambda r: r.provenance.to_dict()["activities"][-1]["design_fingerprint"]
    plain = select_hrf(data, library=library)
    modeled = select_hrf(data, library=library, task_model=nsd_model())
    assert key(plain) != key(modeled)
    changed = replace_data(
        data, events=[e.assign(response_time=[0.9, np.nan, 1.1]) for e in data.events]
    )
    assert key(modeled) != key(select_hrf(changed, library=library, task_model=nsd_model()))
    assert key(plain) == key(select_hrf(changed, library=library))


def test_evaluate_split_with_task_model_returns_named_amplitude_rows(cv_fixture):
    from boldtailor.hrf_selection import evaluate_hrf_split
    from tests.test_hrf_cv import stacked_oracle

    data, library = cv_fixture
    data = with_trial_types(data)
    result = evaluate_hrf_split(
        data, library=library, train_runs=[0, 2], test_runs=[1, 3], task_model=nsd_model()
    )
    assert result.amplitude_names == ("task", "response_time", "trial_type")
    assert result.training_amplitudes.shape == (3, data.n_features)
    assert not result.training_amplitudes.flags.writeable
    for v, cid in enumerate(result.training_selection.hrf_indices[:3]):
        beta, loss, null = stacked_oracle(
            data, library, int(cid), nsd_model(), [0, 2], [1, 3]
        )
        np.testing.assert_allclose(result.training_amplitudes[:, v], beta[:, v], atol=1e-10)
        np.testing.assert_allclose(result.test_r2[v], 1 - loss[v] / null[v], atol=1e-10)
    assert np.isnan(result.training_amplitudes[:, 3:]).all()


def test_default_evaluation_amplitudes_are_one_row_named_task(cv_fixture):
    from boldtailor.hrf_selection import evaluate_hrf_split

    data, library = cv_fixture
    result = evaluate_hrf_split(data, library=library, train_runs=[0, 2], test_runs=[1, 3])
    assert result.amplitude_names == ("task",)
    assert result.training_amplitudes.shape == (1, data.n_features)


def test_task_model_argument_is_validated(cv_fixture):
    from boldtailor.hrf_selection import select_hrf, evaluate_hrf_split

    data, library = cv_fixture
    with pytest.raises(ValueError, match="TaskModel"):
        select_hrf(data, library=library, task_model={"modulators": []})
    with pytest.raises(ValueError, match="TaskModel"):
        evaluate_hrf_split(
            data, library=library, train_runs=[0, 2], test_runs=[1], task_model="nsd"
        )


def test_task_model_selection_feeds_single_trial_fits(cv_fixture):
    from boldtailor.hrf_selection import select_hrf
    from boldtailor.single_trial import fit_selected_hrfs

    data, library = cv_fixture
    data = with_trial_types(data)
    selection = select_hrf(data, library=library, task_model=nsd_model())
    result = fit_selected_hrfs(data, selection=selection)
    assert result.run_betas[0].shape == (3, data.n_features)
    assert np.isfinite(result.run_betas[0][:, :3]).all()


def test_evaluation_result_rejects_mismatched_amplitude_rows(cv_fixture):
    from boldtailor.hrf_results import HrfEvaluationResult
    from boldtailor.hrf_selection import evaluate_hrf_split
    from dataclasses import replace

    data, library = cv_fixture
    result = evaluate_hrf_split(data, library=library, train_runs=[0, 2], test_runs=[1, 3])
    with pytest.raises(ValueError, match="amplitude"):
        replace(result, amplitude_names=("task", "extra"))
    with pytest.raises(ValueError, match="amplitude"):
        replace(result, training_amplitudes=result.training_amplitudes[0])
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_hrf_selection.py -q`
Expected: the new tests FAIL with `TypeError: select_hrf() got an unexpected keyword argument 'task_model'`; the pre-existing tests still PASS (except the amplitude-indexing test, which fails until the schema changes).

- [ ] **Step 4: Commit the failing tests**

```bash
git add tests/test_hrf_selection.py
git commit -m "test: specify task_model argument and named amplitudes for HRF selection

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 5: Extend the result schemas**

In `src/boldtailor/hrf_results.py`, add `from boldtailor.model import TaskModel` and:

- In `HrfSelectionResult`, add the last field `task_model: TaskModel = TaskModel()` and in `__post_init__` add:

```python
        if not isinstance(self.task_model, TaskModel):
            raise ValueError("task_model must be a TaskModel")
```

- In `HrfEvaluationResult`, add the last field `amplitude_names: tuple[str, ...] = ("task",)` and extend `__post_init__`:

```python
        object.__setattr__(self, "amplitude_names", tuple(self.amplitude_names))
        amplitudes = self.training_amplitudes
        if amplitudes.ndim != 2 or amplitudes.shape[0] != len(self.amplitude_names):
            raise ValueError("training_amplitudes needs one row per amplitude name")
```

Place this after the loop that wraps arrays with `readonly_array`.

- [ ] **Step 6: Thread the task model through `hrf_selection.py`**

Make these edits in `src/boldtailor/hrf_selection.py`:

Imports: add `from boldtailor._hrf_cv import MIN_ONSET, OVERSAMPLING, pooled_amplitude` (merge with the existing `_hrf_cv` import) and `from boldtailor.model import TaskModel`.

`_validate` gains a fourth argument:

```python
def _validate(library, signature, batch, task_model):
    if not isinstance(library, HrfLibrary):
        raise ValueError("library must be an HrfLibrary")
    if not isinstance(task_model, TaskModel):
        raise ValueError("task_model must be a TaskModel")
    ...  # existing signature and batch checks unchanged
```

`_provenance` gains `task_model` after `signature`:

```python
def _provenance(data, runs, library, labels, signature, task_model, name, **extra):
    activity = dict(
        name=name,
        library_fingerprint=library.fingerprint,
        design_fingerprint=sha256(
            "|".join(r.fingerprint for r in runs).encode()
        ).hexdigest(),
        run_labels=list(labels),
        feature_signature=signature,
        task_model=task_model.to_dict(),
        task_model_fingerprint=task_model.fingerprint,
        task_regressors=list(task_model.regressor_names),
        profiled_regressors=list(task_model.profiled_names),
        score="nuisance_adjusted_task_model_prediction_r2",
        beta_units="native_signal",
        oversampling=OVERSAMPLING,
        min_onset=MIN_ONSET,
        nuisance="conditional_projection_of_confounds_and_profiled_task_columns_per_run",
        invalid_features="zero signal outside nuisance span at numerical precision",
        sse_roundoff_tolerance="64 * eps * (C + abs(2*b*B) + abs(b*b*A))",
        **extra,
    )
    ...  # unchanged
```

`_select` gains `task_model` as its last parameter, passes it to `_provenance(..., signature, task_model, "hrf_selection", ...)`, and constructs `HrfSelectionResult(..., provenance, task_model=task_model)`.

`select_hrf`:

```python
def select_hrf(
    data,
    *,
    library,
    run_labels=None,
    feature_signature=None,
    candidate_batch_size=32,
    task_model=TaskModel(),
):
    """Choose each feature's HRF by leave-one-run-out task-model prediction.

    This is a selection statistic. Confounds and missing-value indicators are
    profiled in each run; task-model amplitudes are learned only from other
    runs. Anonymous arrays without feature_signature require the caller to
    preserve feature order.
    """
    _validate(library, feature_signature, candidate_batch_size, task_model)
    if data.n_runs < 2:
        raise ValueError("HRF selection requires at least two runs")
    labels = run_labels_for(data, run_labels)
    runs = prepare_runs(data, library, task_model)
    return _select(
        data, runs, data.signals, library, labels, feature_signature,
        candidate_batch_size, runs, task_model,
    )
```

`_predict` returns the full `(H, K, F)` amplitude array (remove the `[:, 0, :]` from Task 4):

```python
    return amplitude, score
```

`evaluate_hrf_split`:

```python
def evaluate_hrf_split(
    data,
    *,
    library,
    train_runs,
    test_runs,
    run_labels=None,
    feature_signature=None,
    task_model=TaskModel(),
):
    """Select within training runs, then freeze HRF and amplitudes for test runs."""
    _validate(library, feature_signature, 32, task_model)
    train = _fold_indices(train_runs, data.n_runs, 2, "train_runs")
    test = _fold_indices(test_runs, data.n_runs, 1, "test_runs")
    if set(train) & set(test):
        raise ValueError("training and test runs must be disjoint")
    labels = run_labels_for(data, run_labels)
    runs = prepare_runs(data, library, task_model)
    selection = _select(
        data,
        tuple(runs[i] for i in train),
        tuple(data.signals[i] for i in train),
        library,
        tuple(labels[i] for i in train),
        feature_signature,
        32,
        tuple(runs[i] for i in (*train, *test)),
        task_model,
    )
    a, b, c, energy = signal_statistics(runs, data.signals, 32)
    amplitudes, scores = _predict(a, b, c, energy, train, test)
    ids = selection.hrf_indices
    valid = np.flatnonzero(ids >= 0)
    chosen = np.full(data.n_features, np.nan)
    coefficient = np.full((len(task_model.regressor_names), data.n_features), np.nan)
    chosen[valid] = scores[ids[valid], valid]
    coefficient[:, valid] = amplitudes[ids[valid], :, valid].T
    canonical = scores[0].copy()
    if not selection.eligibility.loc[0, "eligible"]:
        canonical[:] = np.nan
    canonical[ids < 0] = np.nan
    provenance = _provenance(
        data, runs, library, labels, feature_signature, task_model,
        "hrf_independent_evaluation",
        train_runs=list(train),
        test_runs=list(test),
        training_selection=selection.provenance.to_dict()["activities"][-1],
        frozen_task_amplitudes=True,
    )
    return HrfEvaluationResult(
        selection, coefficient, chosen, canonical, chosen - canonical, train, test,
        provenance, amplitude_names=task_model.regressor_names,
    )
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_hrf_selection.py tests/test_hrf_cv.py tests/test_result_schemas.py tests/test_selected_hrf_fit.py tests/test_ridge_cv.py tests/test_fractional_cv.py tests/test_hrf_glm.py -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/boldtailor/hrf_selection.py src/boldtailor/hrf_results.py
git commit -m "feat: select HRFs with a declared task model and named amplitudes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Selected-HRF GLM uses the scored task design

**Files:**
- Modify: `src/boldtailor/_hrf_glm_design.py`
- Modify: `src/boldtailor/_hrf_glm.py`
- Test: `tests/test_hrf_glm.py`

**Interfaces:**
- Consumes: `ModelSpec.task_model` (Task 3), `expand_events`, `task_columns`, `hrf_model`, `HrfSelectionResult.task_model` (Task 5), selection provenance keys `oversampling` and `min_onset` (Task 5).
- Produces: `fit(data, model, hrf_selection=selection, feature_signature=...)` with `model.task_model` set builds group designs whose leading columns equal `task_columns(expand_events(events, model.task_model, run), times, hrf_model(candidate), min_onset=model.min_onset, oversampling=model.oversampling)` for every `(run, cid)`. Selected-GLM activity records `task_model` and `task_model_fingerprint`. Mismatches raise `ValueError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hrf_glm.py`:

```python
def _nsd_model():
    from boldtailor.model import Modulator, TaskModel

    return TaskModel(
        (
            Modulator("response_time", center=True, missing="indicator"),
            Modulator("trial_type", center=False),
        )
    )


@pytest.fixture(scope="module")
def task_model_problem():
    """Raw trials with RT and trial type; one missing RT in run 1."""
    library = HrfLibrary.from_parameters(
        [[3, 10, 0.5, 0.5, 2, 0, 36], [6, 16, 1.5, 2.5, 8, 2, 36]]
    )
    rng = np.random.default_rng(1207)
    events, times, confounds, signals = [], [], [], []
    for run in range(3):
        t = 0.775 + 1.6 * np.arange(85 + 5 * run)
        rt = np.array([0.8, 1.3, 0.9, 1.7, 1.1, 1.4])
        if run == 1:
            rt[3] = np.nan
        events.append(
            pd.DataFrame(
                dict(
                    onset=np.array([5.3, 21.1, 42.2, 64.4, 88.5, 110.2]) + run,
                    duration=[1.2, 2.0, 0.7, 1.5, 1.1, 2.3],
                    trial_type=[0, 1, 1, 0, 1, 0],
                    response_time=rt,
                )
            )
        )
        times.append(t)
        confounds.append(pd.DataFrame(dict(motion=np.linspace(-1, 1, len(t)))))
        signals.append(rng.normal(100, 1, size=(len(t), 4)))
    data = from_arrays(
        signals, events, frame_times=times, confounds=confounds,
        sources=[_sources(r) for r in range(3)],
    )
    model = ModelSpec(
        contrasts={name: {name: 1} for name in ("task", "response_time", "trial_type")},
        confounds=("motion",),
        hrf_model="spm",
        drift_model=None,
        noise_model="ols",
        task_model=_nsd_model(),
    )
    selection = select_hrf(data, library=library, feature_signature="axis-tm", task_model=_nsd_model())
    return data, model, selection, library


def test_selected_glm_group_designs_equal_scored_task_columns(task_model_problem):
    from boldtailor._hrf_cv import prepare_runs
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import expand_events, task_columns

    data, model, selection, library = task_model_problem
    result = fit(data, model, hrf_selection=selection, feature_signature="axis-tm")
    runs = prepare_runs(data, library, model.task_model)
    assert result.group_designs
    for (run, cid), design in result.group_designs.items():
        scored = runs[run].task_design(cid)
        expected = task_columns(
            expand_events(data.events[run], model.task_model, run),
            data.frame_times[run],
            hrf_model(library.candidates[cid]),
            min_onset=model.min_onset,
            oversampling=model.oversampling,
        )
        pd.testing.assert_frame_equal(scored, expected)
        assert list(design.columns[: scored.shape[1]]) == list(scored.columns)
        np.testing.assert_array_equal(design.iloc[:, : scored.shape[1]].to_numpy(), scored.to_numpy())
        assert list(design.columns[scored.shape[1] :]) == ["motion", "constant"]
    assert "missing_response_time" in result.group_designs[1, int(selection.hrf_indices[0])].columns
    assert "missing_response_time" not in result.group_designs[0, int(selection.hrf_indices[0])].columns
    activity = result.provenance.to_dict()["activities"][-1]
    assert activity["task_model"] == model.task_model.to_dict()
    assert activity["task_model_fingerprint"] == model.task_model.fingerprint
    assert np.isfinite(result.effect("response_time")).all()
    assert np.isfinite(result.effect("trial_type")).all()


def test_selected_glm_spm_group_also_uses_shared_task_columns(task_model_problem):
    from boldtailor._task_design import expand_events, task_columns

    data, model, selection, library = task_model_problem
    canonical = HrfLibrary.from_parameters([])
    spm_only = select_hrf(data, library=canonical, feature_signature="axis-tm", task_model=_nsd_model())
    result = fit(data, model, hrf_selection=spm_only, feature_signature="axis-tm")
    for run in range(data.n_runs):
        expected = task_columns(
            expand_events(data.events[run], model.task_model, run), data.frame_times[run], "spm",
            min_onset=model.min_onset, oversampling=model.oversampling,
        )
        design = result.group_designs[run, 0]
        np.testing.assert_array_equal(design.iloc[:, : expected.shape[1]].to_numpy(), expected.to_numpy())


def test_selected_glm_requires_matching_task_model(task_model_problem):
    from boldtailor.model import Modulator, TaskModel

    data, model, selection, library = task_model_problem
    with pytest.raises(ValueError, match="task_model"):
        fit(data, replace(model, task_model=None), hrf_selection=selection, feature_signature="axis-tm")
    other = TaskModel((Modulator("response_time", missing="indicator"),))
    with pytest.raises(ValueError, match="task_model"):
        fit(data, replace(model, task_model=other), hrf_selection=selection, feature_signature="axis-tm")
    plain = select_hrf(data, library=library, feature_signature="axis-tm")
    with pytest.raises(ValueError, match="task_model"):
        fit(data, model, hrf_selection=plain, feature_signature="axis-tm")


def test_selected_glm_requires_matching_convolution_settings(task_model_problem):
    data, model, selection, library = task_model_problem
    with pytest.raises(ValueError, match="oversampling|min_onset"):
        fit(data, replace(model, oversampling=20), hrf_selection=selection, feature_signature="axis-tm")
    with pytest.raises(ValueError, match="oversampling|min_onset"):
        fit(data, replace(model, min_onset=-10.0), hrf_selection=selection, feature_signature="axis-tm")


def test_task_delta_r2_uses_the_same_task_model_designs(task_model_problem):
    data, model, selection, library = task_model_problem
    result = fit(data, model, hrf_selection=selection, feature_signature="axis-tm")
    comparison = task_delta_r2(data, model, result)
    assert np.isfinite(comparison.delta_r2).all()
    assert np.all(comparison.delta_r2 >= 0)


def test_legacy_selected_glm_without_task_model_is_unchanged(hrf_glm_problem):
    data, model, selection, designs = hrf_glm_problem
    result = _selected_fit(data, model, selection)
    for (run, cid), design in result.group_designs.items():
        np.testing.assert_allclose(design.to_numpy(), designs[run, cid][design.columns].to_numpy(), atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_hrf_glm.py -q -k "task_model or task_columns or convolution or legacy"`
Expected: the new tests FAIL. The design-equality test fails because `_custom_design` expands nothing and Nilearn sees numeric `trial_type` codes; the guard tests fail with no exception raised.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_hrf_glm.py
git commit -m "test: specify selected-HRF GLM designs built from the scored task model

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 4: Build task-model group designs**

In `src/boldtailor/_hrf_glm_design.py`, add imports `from boldtailor._hrf_design import hrf_model` and `from boldtailor._task_design import expand_events, task_columns`, then change the candidate loop in `compile_group_designs`:

```python
    for cid in ids:
        candidate = selection.library.candidates[cid]
        if model.task_model is not None:
            designs = tuple(
                _task_model_design(e, t, n, candidate, model, run)
                for run, (e, t, n) in enumerate(
                    zip(events, data.frame_times, nuisance, strict=True)
                )
            )
        elif candidate.kind == "spm":
            designs = compile_designs(data, replace(model, hrf_model="spm"))
        else:
            designs = tuple(
                _custom_design(e, t, n, candidate, model, run)
                for run, (e, t, n) in enumerate(
                    zip(events, data.frame_times, nuisance, strict=True)
                )
            )
        groups.update({(run, int(cid)): design for run, design in enumerate(designs)})
```

Add:

```python
def _task_model_design(events, times, nuisance, candidate, model, run):
    modeled, excluded, cutoff = _select_modeled_events(
        events, times, model.min_onset, run
    )
    task = task_columns(
        expand_events(modeled, model.task_model, run),
        times,
        hrf_model(candidate),
        min_onset=model.min_onset,
        oversampling=model.oversampling,
    )
    matrix = pd.concat([task, nuisance.matrix], axis=1)
    _validate_design_matrix(matrix, run)
    return CompiledDesign(matrix, excluded, cutoff)
```

- [ ] **Step 5: Guard and record the task model in `_hrf_glm.py`**

Add `from boldtailor.model import TaskModel` and a guard called at the top of `_prepare`, right after `validate_selection`:

```python
def _check_task_model(model, selection):
    if model.task_model is None:
        if selection.task_model != TaskModel():
            raise ValueError(
                "selection used a task_model; set ModelSpec.task_model to the same model"
            )
        return
    if model.task_model.fingerprint != selection.task_model.fingerprint:
        raise ValueError("ModelSpec.task_model must match the selection's task_model")
    activity = selection.provenance.to_dict()["activities"][-1]
    recorded = (activity.get("oversampling"), activity.get("min_onset"))
    if (model.oversampling, model.min_onset) != recorded:
        raise ValueError(
            "ModelSpec.oversampling and min_onset must match the selection's settings"
        )
```

In `_prepare`, after `settings = {...}`:

```python
    if model.task_model is not None:
        settings["task_model"] = model.task_model.to_dict()
        settings["task_model_fingerprint"] = model.task_model.fingerprint
    activity = dict(
        ...  # existing keys
    )
    if model.task_model is not None:
        activity["task_model"] = model.task_model.to_dict()
        activity["task_model_fingerprint"] = model.task_model.fingerprint
```

`fit_selected_glm` and `selected_task_delta_r2` compute `settings` through `_model_provenance(replace(model, hrf_model=None))`. `replace(model, hrf_model=None)` would now fail validation when `task_model` is set. Change both call sites to `_model_provenance(replace(model, hrf_model=None, task_model=None)).activity`; the task model is added back explicitly by the lines above, so provenance still records it.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_hrf_glm.py tests/test_fit.py tests/test_design.py -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/boldtailor/_hrf_glm_design.py src/boldtailor/_hrf_glm.py
git commit -m "feat: fit selected-HRF GLMs on the scored task-model design

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: NSD example uses one task model everywhere

**Files:**
- Modify: `examples/NSD/workflow_inputs.py`
- Modify: `examples/NSD/workflow_analysis.py:22-36,70-73`
- Modify: `examples/NSD/workflow_outputs.py:14,355`
- Modify: `examples/NSD/session_hrf.py:34-41`
- Modify: `examples/NSD/session_hrf_cache.py:22-47`
- Modify: `examples/NSD/nsd_workflow.ipynb` (import cell near line 111 and the model cell near line 1439 only)
- Test: `examples/NSD/test_nsd_workflow.py`, `examples/NSD/test_session_hrf.py`, `examples/NSD/test_workflow_reuse.py`

The two notebooks have uncommitted user edits in the working tree. Edit only the two named cells with `NotebookEdit` or a targeted JSON edit; do not reformat or revert anything else.

**Interfaces:**
- Consumes: `TaskModel`, `Modulator`, `expand_events`, `select_hrf(task_model=...)`, `ModelSpec(task_model=...)`.
- Produces in `examples/NSD/workflow_inputs.py`: `NSD_TASK_MODEL`, `validate_glm_events(events) -> None`, `load_block(runs, root, indices)` with no `glm` flag, `glm_model(runs)` with `task_model=NSD_TASK_MODEL`. `WorkflowRun.events` has nonpositive RTs replaced by NaN.

- [ ] **Step 1: Rewrite the event-encoding tests**

In `examples/NSD/test_nsd_workflow.py`, replace the five tests from `test_glm_events_preserve_timing_and_center_two_joint_modulators` through `test_glm_does_not_treat_malformed_rt_text_as_missing` with:

```python
def test_nsd_task_model_centers_rt_and_keeps_trial_type_uncentered(events):
    from boldtailor._task_design import expand_events
    from boldtailor.model import Modulator, TaskModel

    inputs = workflow()
    assert inputs.NSD_TASK_MODEL == TaskModel(
        (
            Modulator("response_time", center=True, missing="indicator"),
            Modulator("trial_type", center=False),
        )
    )
    original = events.copy(deep=True)
    result = expand_events(events, inputs.NSD_TASK_MODEL)
    amplitudes = {
        "task": np.ones(len(events)),
        "response_time": events.response_time - events.response_time.mean(),
        "trial_type": events.trial_type,
    }
    assert list(dict.fromkeys(result.trial_type)) == list(amplitudes)
    for name, expected in amplitudes.items():
        rows = result.loc[result.trial_type == name]
        np.testing.assert_allclose(rows.modulation, expected)
        np.testing.assert_allclose(rows[["onset", "duration"]], events[["onset", "duration"]])
    pd.testing.assert_frame_equal(events, original)


@pytest.mark.parametrize("column,value", [("trial_type", 2), ("trial_type", np.nan)])
def test_invalid_glm_covariates_fail_explicitly(events, column, value):
    events.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        workflow().validate_glm_events(events)


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_nonpositive_rt_becomes_missing_with_indicator(four_runs, missing):
    inputs = workflow()
    root, prep = four_runs
    path = next(root.rglob("*run-01_events.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.loc[1, "response_time"] = missing
    table.to_csv(path, sep="\t", index=False)
    runs = inputs.load_session(root, prep)
    run = next(r for r in runs if r.number == 1)
    assert np.isnan(run.events.response_time.iloc[1])
    assert np.isfinite(run.events.response_time.drop(index=1)).all()
    from boldtailor._task_design import expand_events

    expanded = expand_events(run.events, inputs.NSD_TASK_MODEL)
    indicator = expanded.loc[expanded.trial_type == "missing_response_time", "modulation"]
    np.testing.assert_array_equal(indicator, (np.arange(len(run.events)) == 1).astype(float))


def test_glm_requires_observed_rt_to_estimate_rt_effect(events):
    events["response_time"] = np.nan
    with pytest.raises(ValueError, match="response_time.*positive.*finite"):
        workflow().validate_glm_events(events)


def test_glm_does_not_treat_malformed_rt_text_as_missing(events):
    events["response_time"] = events.response_time.astype(object)
    events.loc[1, "response_time"] = "invalid"
    with pytest.raises(ValueError, match="response_time"):
        workflow().validate_glm_events(events)
```

Then change every `load_block(..., glm=True)` in the example tests to `load_block(...)`, and add this test after the GLM tests that use `inputs.glm_model(runs)` (near line 205):

```python
def test_glm_model_and_selection_share_the_nsd_task_model(four_runs, small_library):
    from boldtailor._hrf_design import hrf_model
    from boldtailor._task_design import expand_events, task_columns

    inputs, analysis = workflow(), workflow("workflow_analysis")
    root, prep = four_runs
    runs = inputs.load_session(root, prep)
    blocks = inputs.make_blocks(runs, block_size=2, max_grayordinates=4)
    model = inputs.glm_model(runs)
    assert model.task_model == inputs.NSD_TASK_MODEL
    selections = analysis.select_hrfs(runs, root, blocks, small_library)
    for bundle in selections.values():
        assert bundle["all"].task_model == inputs.NSD_TASK_MODEL
    fitted = analysis.fit_glms(runs, root, blocks, model, selections=selections)
    assert fitted["designs"]
    for (run, cid), design in fitted["designs"].items():
        expected = task_columns(
            expand_events(runs[run].events, inputs.NSD_TASK_MODEL, run),
            runs[run].frame_times,
            hrf_model(small_library.candidates[cid]),
        )
        np.testing.assert_array_equal(
            design.iloc[:, : expected.shape[1]].to_numpy(), expected.to_numpy()
        )
```

- [ ] **Step 2: Run the example tests to verify they fail**

Run: `uv run pytest examples/NSD/test_nsd_workflow.py -q -x`
Expected: FAIL with `AttributeError: module 'examples.NSD.workflow_inputs' has no attribute 'NSD_TASK_MODEL'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add examples/NSD/test_nsd_workflow.py
git commit -m "test: specify the NSD workflow's shared task model

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 4: Rewrite `workflow_inputs.py` event handling**

Replace `_rt_amplitudes` and `glm_events` with:

```python
from boldtailor.model import ModelSpec, Modulator, TaskModel

NSD_TASK_MODEL = TaskModel(
    (
        Modulator("response_time", center=True, missing="indicator"),
        Modulator("trial_type", center=False),
    )
)


def _numeric_column(events, name):
    if name not in events:
        raise ValueError(f"Missing {name}")
    try:
        return pd.to_numeric(events[name], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error


def validate_glm_events(events):
    """Require an observed positive RT and both binary trial_type codes."""
    rt = _numeric_column(events, "response_time")
    if not (np.isfinite(rt) & (rt > 0)).any():
        raise ValueError(
            "response_time needs positive finite observations for the RT effect"
        )
    trial_type = _numeric_column(events, "trial_type")
    if not np.isfinite(trial_type).all() or set(trial_type) != {0, 1}:
        raise ValueError("trial_type must contain both binary codes 0 and 1")


def _missing_nonpositive_rt(events):
    """Package semantics: missing means non-finite, so nonpositive RTs become NaN."""
    rt = _numeric_column(events, "response_time")
    return events.assign(response_time=np.where(np.isfinite(rt) & (rt > 0), rt, np.nan))
```

In `_trim`, replace the `if not hrf_only: glm_events(run.events)` lines with `validate_glm_events(run.events)` (always), and pass `_missing_nonpositive_rt(run.events)` as the `events` argument of the returned `WorkflowRun`.

Replace `load_block`:

```python
def load_block(runs, root, indices):
    """Raw trial rows for every analysis; the task model expands them."""
    return from_arrays(
        block_signals(runs, indices),
        [r.events for r in runs],
        frame_times=[r.frame_times for r in runs],
        confounds=[r.confounds for r in runs],
        sources=[_trimmed_sources(r, root, indices) for r in runs],
        provenance_metadata={
            "event_encoding": "raw_trials_with_task_model",
            "task_model": NSD_TASK_MODEL.to_dict(),
        },
    )
```

Replace `glm_model`:

```python
def glm_model(runs):
    return ModelSpec(
        contrasts={name: {name: 1} for name in REGRESSORS},
        confounds=tuple(runs[0].confounds.columns),
        hrf_model="spm",
        drift_model=None,
        noise_model="ols",
        task_model=NSD_TASK_MODEL,
    )
```

- [ ] **Step 5: Update the analysis, session, cache, and output helpers**

`examples/NSD/workflow_analysis.py`: import `NSD_TASK_MODEL` from `.workflow_inputs`; in `_select_block` add `task_model=NSD_TASK_MODEL` to `options`; in `_glm_block` change `load_block(runs, root, indices, glm=True)` to `load_block(runs, root, indices)`.

`examples/NSD/session_hrf.py`: import `NSD_TASK_MODEL` and pass `task_model=NSD_TASK_MODEL` in `_fit_block`.

`examples/NSD/session_hrf_cache.py` `request_metadata`: change `model="mean_stimulus_leave_one_run_out"` to `model="task_model_leave_one_run_out"` and add `task_model=NSD_TASK_MODEL.to_dict()` (import it from `.workflow_inputs`). This invalidates stored caches by design.

`examples/NSD/workflow_outputs.py`: replace the `glm_events` import with `from boldtailor._task_design import expand_events` plus `NSD_TASK_MODEL` from `.workflow_inputs`, and change the events artifact line to
`table_artifact(base + "_events.tsv", expand_events(run.events, NSD_TASK_MODEL, run.number))`.

`examples/NSD/nsd_workflow.ipynb`: in the import cell replace `glm_events` with `NSD_TASK_MODEL` and add `from boldtailor._task_design import expand_events`; in the model cell replace
`display(glm_events(runs[0].events).groupby("trial_type", sort=False).head(3))` with
`display(expand_events(runs[0].events, NSD_TASK_MODEL).groupby("trial_type", sort=False).head(3))`.

Search for any remaining `glm_events`, `glm=True`, or `hrf_only=` uses:

Run: `grep -rn "glm_events\|glm=True\|glm=glm\b" examples/NSD --include='*.py' --include='*.ipynb' | grep -v '"outputs"'`
Expected: only the `glm=glm` dictionary key in `multisession_analysis.py`, which is unrelated and stays.

- [ ] **Step 6: Run the example suites**

Run: `uv run pytest examples/NSD -q -x`
Expected: all PASS. If `test_session_hrf.py` asserts on the cache `model` string or `hrf_only` retention behavior, update the assertion to the new string `task_model_leave_one_run_out` and keep the retention assertions; trials with missing RTs are still retained because the indicator models them.

- [ ] **Step 7: Commit**

```bash
git add examples/NSD/workflow_inputs.py examples/NSD/workflow_analysis.py examples/NSD/workflow_outputs.py examples/NSD/session_hrf.py examples/NSD/session_hrf_cache.py examples/NSD/nsd_workflow.ipynb examples/NSD/test_session_hrf.py
git commit -m "feat: run the NSD workflow on one shared task model

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

Stage only the notebook cells you changed if the working tree holds other uncommitted notebook edits: use `git add -p examples/NSD/nsd_workflow.ipynb` and accept only the two hunks.

---

### Task 8: Documentation and full verification

**Files:**
- Modify: `docs/user-guide.md` (sections "Voxelwise HRFs in conventional GLMs" and "Selecting an HRF for each location")
- Modify: `docs/api.md` (ModelSpec table, `fit()` keyword notes, and the "HRF libraries, selection, and evaluation" section)

- [ ] **Step 1: Update the user guide**

In "Selecting an HRF for each location", replace the paragraph beginning "At each feature, selection evaluates every candidate HRF in turn." and the following paragraph ending "not estimates of the HRF's peak height." with:

```markdown
At each feature, selection evaluates every candidate HRF in turn. The
candidate's task-model regressors are convolved with Nilearn and projected off
the run's confounds. For each held-out run, selection learns one amplitude per
task regressor from the remaining runs and predicts that run. Prediction
errors are pooled across folds and divided by the pooled confound-adjusted
signal energy, so the denominator is the same for every candidate. **The
session map contains that winner's parameters; it does not average parameters
selected for individual runs.**

By default the task model has one regressor, `task`, so selection scores the
mean stimulus response and RT never enters. Pass `task_model=` to score the
same task model the GLM will fit:

```python
from boldtailor.model import Modulator, TaskModel

task_model = TaskModel((
    Modulator("response_time", center=True, missing="indicator"),
    Modulator("trial_type", center=False),
))
selection = select_hrf(multi_run_data, library=library, task_model=task_model)
```

Each modulator names a numeric column of the raw per-trial events. `center`
subtracts the within-run mean of observed values. `missing="indicator"` gives
missing (non-finite) trials zero modulation and adds a `missing_<column>`
regressor in runs that need it; its coefficient is fit freely within each run,
like a confound, but with the candidate HRF. `missing="error"` rejects
non-finite values. A candidate is eligible when its task, indicator, and
confound columns are full rank with residual degrees of freedom in every run
and the pooled training design is invertible in every fold.

This method assumes that task-model amplitudes transfer across runs. It does
not require repeated images. Kernels are normalized to sum to one, so beta
values depend on that convention and are not estimates of the HRF's peak height.
```

In "Voxelwise HRFs in conventional GLMs", after the first code block, replace the paragraph beginning "The selection replaces `ModelSpec.hrf_model` entirely" with:

```markdown
The selection replaces `ModelSpec.hrf_model` entirely, including any derivative
basis specified there. Each location uses one selected HRF for all its task
regressors. The final GLM estimates amplitudes independently within each run;
it does not reuse the amplitudes from HRF selection.

Set `ModelSpec(task_model=...)` to the task model used for selection. The
GLM then builds its task columns from the same raw events with the same
Nilearn call, so the fitted task design is the scored task design. `fit()`
rejects a task model, `oversampling`, or `min_onset` that differ from the
selection's. Drifts and the `confounds` subset still come from `ModelSpec`;
to match selection's nuisance exactly, pass every confound column and set
`drift_model=None`. Without a task model, events are Nilearn-format
conditions as before, and the selection must have used the default task-only
model, which scored only the mean stimulus response.
```

Remove the sentence "RT never enters HRF selection." wherever it still appears, and in the last paragraph of that section replace "for selection input, represent each presentation once, without extra event rows used to encode amplitude modulators in the target GLM." with "for selection input, supply raw per-trial events; the task model expands them for both selection and fitting."

- [ ] **Step 2: Update the API reference**

In the `ModelSpec` table in `docs/api.md`, add a row:

```markdown
| `task_model` | `None` | `TaskModel` shared with HRF selection; requires `hrf_model` `"spm"` or `"glover"` and raw per-trial events |
```

After the `fit()` keyword paragraph, add:

```markdown
With `model.task_model` set, the selection must carry the same task model and
the model's `oversampling` and `min_onset` must equal the selection's recorded
values. With `task_model=None`, the selection must use the default task-only
model.
```

In "HRF libraries, selection, and evaluation", document the new arguments and fields:

```markdown
`select_hrf(data, *, library, run_labels=None, feature_signature=None,
candidate_batch_size=32, task_model=TaskModel())` and
`evaluate_hrf_split(..., task_model=TaskModel())` score the task model's
regressors. `HrfSelectionResult.task_model` records it.
`HrfEvaluationResult.training_amplitudes` has one row per task regressor, in
`amplitude_names` order. Selection provenance records `task_model`,
`task_model_fingerprint`, `task_regressors`, `profiled_regressors`,
`min_onset`, and `oversampling`.

From `boldtailor.model`: `Modulator(column, center=True, missing="error")` and
`TaskModel(modulators=())`, with `regressor_names`, `profiled_names`,
`fingerprint`, and `to_dict()`.
```

- [ ] **Step 3: Run the complete suite**

Run: `uv run pytest -q`
Expected: all PASS. Record the total count.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide.md docs/api.md
git commit -m "docs: describe the shared task model for selection and GLM fitting

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Memory.** `B` is now `(runs, candidates, K, features)`. For the NSD Sobol library (513 candidates), K=3, 4096 features, and 12 runs this is about 600 MB per block. `C` adds about 200 MB. If a worker runs out of memory, lower the notebook `block_size`; do not add a new knob.
- **Speed.** `RunDesign.block` calls Nilearn once per candidate per run. The `lru_cache` on `_cached_design` means each worker process builds each run's blocks once and reuses them for every feature block.
- **Do not edit `choose_eligible`.** Its lazy structural check now reads `RunDesign.eligible`, which is the task-model check.
- **Ridge CV.** `_ridge_cv.py` keeps calling `prepare_runs(data, library)` and `select_hrf(...)` with the task-only default. Out of scope by the spec.
