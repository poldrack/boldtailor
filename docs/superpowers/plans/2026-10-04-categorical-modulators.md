# Categorical Modulators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand discrete event columns (above all BIDS `trial_type`) into reference-coded indicator regressors throughout the package and the `boldtailor run` workflow.

**Architecture:** `Modulator` gains `kind="categorical"` with `levels` and `reference`; `TaskModel.regressor_names` expands a categorical modulator to `column[level]` for each non-reference level, and `_task_design` builds the 0/1 amplitudes. The workflow resolves levels from the union of all runs at load time, validates every run before fitting, and feeds the indicator columns to beta-series encoding, outputs, and the report. Binary 0/1 `trial_type` becomes the categorical regressor `trial_type[1]`.

**Tech Stack:** Python 3.12, numpy, pandas, Nilearn 0.14, pytest, uv, Black.

**Spec:** `docs/superpowers/specs/2026-10-04-categorical-modulators-design.md`

## Global Constraints

- `uv run` for every command; Black formatting; `uv run pytest -q -W error` (default suite, `tests/`) passes after every task; `uv run pytest -q examples/NSD` passes after Task 6.
- Every `__init__.py` stays empty. No package module imports from `examples`.
- RED-GREEN: tests written and committed before implementation in every task. Never weaken a test to pass; an existing test changes only when this spec changes its requirement (say so in the commit message).
- Prefer Nilearn components; custom numerics only with a Nilearn oracle test.
- Indicator names are `<column>[<level>]`, uniformly, including binary 0/1 (`trial_type[1]`). Missing-value indicator name stays `missing_<column>`.
- Reference coding: `task` kept; k levels add k − 1 indicators; reference defaults to the first level in sorted order (numeric order when every level parses as a number, else lexical).
- Every level, including the reference, must occur in every run; otherwise an error naming the run and the level, raised before fitting (CLI exit 2).
- Missing values (`NaN`, `None`, `""`, `"n/a"`) in a categorical column: error by default; with `missing="indicator"` one `missing_<column>` regressor.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Spec clarification (binding):** `Modulator(kind="categorical")` may be constructed with `levels=None` ("unresolved"), because settings and the CLI store modulators before events are read; the at-least-two-levels and reference-membership checks run when levels are supplied. `TaskModel` rejects unresolved categorical modulators, so every fitted model has concrete levels.

## Review Focus

1. The same level written as `1` in one run's TSV and `"1"`/`1.0` in another (pandas dtypes differ per file) must be one level, `trial_type[1]` — Task 1 (`level_name`) and Task 4 (mixed-dtype detection) tests.
2. Level strings with spaces or punctuation (`"scrambled face"`) must survive into regressor names, CIFTI map names, and the HTML report (escaped) — Task 2 and Task 5 tests.
3. An explicit `reference=LEVEL` that never occurs in the events must be an input error naming the column and level, not a silent fallback — Task 4 test.
4. A value present in a run but outside the resolved levels (possible only with explicit `levels` from the Python API) must be an error, never silently coded as the reference — Task 2 and Task 4 (`trial_predictors`) tests.
5. A numeric column named like an indicator (`trial_type[1]`) alongside a categorical `trial_type` must be rejected as a duplicate regressor name — Task 1 test.

---

### Task 1: Categorical `Modulator` and expanded `TaskModel` names

**Files:**
- Modify: `src/boldtailor/model.py` (`Modulator`, `TaskModel`, new `level_name`)
- Test: `tests/test_task_model.py`

**Interfaces:**
- Produces:
  - `level_name(value) -> str | None` — canonical level string; `None` for missing (`None`, NaN, `""`, `"n/a"`); integral numbers become `"1"`, other finite numbers `repr(float)`; strings are stripped; other types raise `ValueError`.
  - `Modulator(column, missing="error", kind="numeric", levels=None, reference=None)`; properties `resolved: bool`, `regressor_names: tuple[str, ...]`, `indicator_name`; method `with_levels(values) -> Modulator`; `to_dict()` includes `kind` and, for categorical, `levels` (list or `None`) and `reference`.
  - `TaskModel.regressor_names` = `("task", *each modulator's regressor_names)`; `TaskModel` rejects unresolved categorical modulators and duplicate regressor names.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_task_model.py`)

```python
import math

import numpy as np
import pytest

from boldtailor.model import Modulator, TaskModel, level_name


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, "1"),
        (1.0, "1"),
        (np.int64(2), "2"),
        ("1", "1"),
        (" face ", "face"),
        (2.5, "2.5"),
        (None, None),
        (float("nan"), None),
        ("", None),
        ("n/a", None),
    ],
)
def test_level_name_canonicalises_values(value, expected):
    assert level_name(value) == expected


def test_level_name_rejects_other_types():
    with pytest.raises(ValueError, match="categorical value"):
        level_name([1])


def test_categorical_levels_are_sorted_and_reference_defaults_to_first():
    mod = Modulator("trial_type", kind="categorical", levels=("scrambled", "face", "house"))
    assert mod.levels == ("face", "house", "scrambled")
    assert mod.reference == "face"
    assert mod.regressor_names == ("trial_type[house]", "trial_type[scrambled]")


def test_numeric_levels_sort_numerically_and_normalise():
    mod = Modulator("cond", kind="categorical", levels=(10, 2.0, "1"))
    assert mod.levels == ("1", "2", "10")
    assert mod.regressor_names == ("cond[2]", "cond[10]")


def test_explicit_reference_and_binary_naming():
    mod = Modulator("trial_type", kind="categorical", levels=(0, 1), reference=1)
    assert mod.reference == "1"
    assert mod.regressor_names == ("trial_type[0]",)
    assert Modulator("trial_type", kind="categorical", levels=(1, 0)).regressor_names == (
        "trial_type[1]",
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(kind="ordinal"), "kind"),
        (dict(levels=("a", "b")), "categorical"),
        (dict(reference="a"), "categorical"),
        (dict(kind="categorical", levels=("a",)), "at least two"),
        (dict(kind="categorical", levels=("a", "a ")), "distinct"),
        (dict(kind="categorical", levels=("a", None)), "nonmissing"),
        (dict(kind="categorical", levels=("a", "b"), reference="c"), "reference"),
    ],
)
def test_invalid_categorical_modulators(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Modulator("cond", **kwargs)


def test_unresolved_categorical_gets_levels_later():
    mod = Modulator("trial_type", kind="categorical", reference="house")
    assert not mod.resolved
    with pytest.raises(ValueError, match="no levels"):
        mod.regressor_names
    resolved = mod.with_levels({"face", "house"})
    assert resolved.resolved and resolved.reference == "house"
    assert resolved.regressor_names == ("trial_type[face]",)


def test_task_model_expands_and_rejects_unresolved_or_duplicate_names():
    cat = Modulator("trial_type", kind="categorical", levels=("a", "b", "c"))
    model = TaskModel((Modulator("response_time", missing="indicator"), cat))
    assert model.regressor_names == ("task", "response_time", "trial_type[b]", "trial_type[c]")
    assert model.to_dict()["regressors"] == list(model.regressor_names)
    with pytest.raises(ValueError, match="levels"):
        TaskModel((Modulator("trial_type", kind="categorical"),))
    with pytest.raises(ValueError, match="unique"):
        TaskModel((cat, Modulator("trial_type[b]")))


def test_to_dict_and_fingerprint_record_kind_levels_and_reference():
    cat = Modulator("trial_type", kind="categorical", levels=("a", "b"), missing="indicator")
    assert cat.to_dict() == {
        "column": "trial_type",
        "missing": "indicator",
        "kind": "categorical",
        "levels": ["a", "b"],
        "reference": "a",
    }
    assert Modulator("rt").to_dict() == {"column": "rt", "missing": "error", "kind": "numeric"}
    other = Modulator("trial_type", kind="categorical", levels=("a", "b"), reference="b", missing="indicator")
    assert TaskModel((cat,)).fingerprint != TaskModel((other,)).fingerprint
    assert not TaskModel((cat,)).is_subset_of(TaskModel((other,)))
    assert Modulator(**cat.to_dict()) == cat
```

Also update the existing `to_dict` expectations in `tests/test_task_model.py` (and `tests/test_task_design.py::test_modulator_has_no_center_option`) to include `"kind": "numeric"` — a requirement change from this spec.

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest -q tests/test_task_model.py tests/test_task_design.py`
Expected: FAIL (`ImportError: level_name`, unexpected keyword `kind`).

- [ ] **Step 3: Commit the tests**

```bash
git add tests/test_task_model.py tests/test_task_design.py
git commit -m "test: categorical modulators, level names, expanded task-model regressors"
```

- [ ] **Step 4: Implement in `src/boldtailor/model.py`**

Add `replace` to the `dataclasses` import and these definitions (keep `_validate_modulator_column` as is):

```python
_KINDS = ("numeric", "categorical")
_MISSING_TEXT = frozenset({"", "n/a"})


def level_name(value: object) -> str | None:
    """Canonical string for one categorical value, or None when it is missing."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return None if text.lower() in _MISSING_TEXT else text
    if is_real(value):
        number = float(value)
        if not np.isfinite(number):
            return None
        return str(int(number)) if number.is_integer() else repr(number)
    raise ValueError(f"categorical value {value!r} must be a string or a number")


def _sorted_levels(levels):
    try:
        return tuple(sorted(levels, key=float))
    except ValueError:
        return tuple(sorted(levels))


@dataclass(frozen=True)
class Modulator:
    """One task regressor family derived from a raw events column.

    Numeric modulators enter uncentered, so the task regressor is the response
    at modulator value zero. Categorical modulators add one 0/1 indicator per
    non-reference level, so the task regressor is the reference-level response.
    """

    column: str
    missing: str = "error"
    kind: str = "numeric"
    levels: tuple[str, ...] | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        _validate_modulator_column(self.column)
        if self.missing not in _MISSING_POLICIES:
            raise ValueError("modulator missing policy must be 'error' or 'indicator'")
        if self.kind not in _KINDS:
            raise ValueError("modulator kind must be 'numeric' or 'categorical'")
        if self.kind == "numeric":
            if self.levels is not None or self.reference is not None:
                raise ValueError("levels and reference apply only to categorical modulators")
            return
        self._set_reference(self._reference_name())
        if self.levels is not None:
            self._resolve_levels()

    def _reference_name(self):
        if self.reference is None:
            return None
        name = level_name(self.reference)
        if name is None:
            raise ValueError(f"modulator {self.column!r} reference must be a nonmissing level")
        return name

    def _set_reference(self, name):
        object.__setattr__(self, "reference", name)

    def _resolve_levels(self):
        names = [level_name(v) for v in self.levels]
        if any(n is None for n in names):
            raise ValueError(f"modulator {self.column!r} levels must be nonmissing")
        if len(set(names)) != len(names):
            raise ValueError(f"modulator {self.column!r} levels must be distinct")
        if len(names) < 2:
            raise ValueError(f"categorical modulator {self.column!r} needs at least two levels")
        levels = _sorted_levels(names)
        if self.reference is not None and self.reference not in levels:
            raise ValueError(
                f"reference {self.reference!r} is not a level of {self.column!r}: {list(levels)}"
            )
        object.__setattr__(self, "levels", levels)
        self._set_reference(self.reference or levels[0])

    @property
    def resolved(self) -> bool:
        return self.kind == "numeric" or self.levels is not None

    def with_levels(self, values) -> "Modulator":
        return replace(self, levels=tuple(values))

    @property
    def regressor_names(self) -> tuple[str, ...]:
        if self.kind == "numeric":
            return (self.column,)
        if self.levels is None:
            raise ValueError(f"categorical modulator {self.column!r} has no levels yet")
        return tuple(f"{self.column}[{lv}]" for lv in self.levels if lv != self.reference)

    @property
    def indicator_name(self) -> str:
        return f"missing_{self.column}"

    def to_dict(self) -> dict[str, object]:
        values = {"column": self.column, "missing": self.missing, "kind": self.kind}
        if self.kind == "categorical":
            values["levels"] = None if self.levels is None else list(self.levels)
            values["reference"] = self.reference
        return values
```

In `TaskModel.__post_init__`, after the column-uniqueness check add:

```python
        if any(not m.resolved for m in modulators):
            raise ValueError("categorical modulators need levels before fitting")
        names = [n for m in modulators for n in m.regressor_names]
        if len(set(names)) != len(names) or "task" in names:
            raise ValueError("task-model regressor names must be unique")
```

and change `regressor_names` to:

```python
    @property
    def regressor_names(self) -> tuple[str, ...]:
        return ("task", *(n for m in self.modulators for n in m.regressor_names))
```

- [ ] **Step 5: GREEN**

Run: `uv run black -q src tests && uv run pytest -q -W error tests/test_task_model.py tests/test_task_design.py && uv run pytest -q -W error`
Expected: PASS. If other tests compare `Modulator.to_dict()` or `TaskModel.to_dict()` literally, add `"kind": "numeric"` there (requirement change) and list them in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/boldtailor/model.py tests
git commit -m "feat: categorical Modulator kind with canonical levels and reference-coded regressor names"
```

---

### Task 2: Indicator amplitudes in the design expansion

**Files:**
- Modify: `src/boldtailor/_task_design.py`
- Test: `tests/test_task_design.py`

**Interfaces:**
- Consumes: `Modulator.kind/levels/reference/regressor_names/indicator_name`, `level_name` (Task 1).
- Produces: `categorical_names(values, modulator) -> list[str | None]` (per-trial canonical level; raises `ValueError` without a run prefix for unknown values, missing values under `missing="error"`, or a declared level with no trials); `expand_events` handles categorical modulators.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_task_design.py`)

```python
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import make_first_level_design_matrix

from boldtailor._task_design import categorical_names, expand_events, task_columns
from boldtailor.model import Modulator, TaskModel


def _events(levels):
    n = len(levels)
    return pd.DataFrame(
        dict(onset=np.arange(n) * 12.0 + 2, duration=np.ones(n), cond=levels)
    )


CAT = Modulator("cond", kind="categorical", levels=("face", "house", "scrambled face"))


def test_categorical_expansion_builds_reference_coded_indicators():
    events = _events(["house", "face", "scrambled face", "house", "face", "scrambled face"])
    expanded = expand_events(events, TaskModel((CAT,)), run="run-01")
    by_name = {k: g.modulation.to_numpy() for k, g in expanded.groupby("trial_type", sort=False)}
    assert list(by_name) == ["task", "cond[house]", "cond[scrambled face]"]
    np.testing.assert_array_equal(by_name["task"], np.ones(6))
    np.testing.assert_array_equal(by_name["cond[house]"], [1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(by_name["cond[scrambled face]"], [0, 0, 1, 0, 0, 1])


def test_categorical_columns_match_nilearn_on_hand_built_conditions():
    events = _events(["house", "face", "scrambled face", "house", "face", "scrambled face"])
    frame_times = np.arange(90) * 1.0
    ours = task_columns(expand_events(events, TaskModel((CAT,))), frame_times, "glover")
    manual = pd.concat(
        [
            events.assign(trial_type="task", modulation=1.0),
            events[events.cond == "house"].assign(trial_type="cond[house]", modulation=1.0),
            events[events.cond == "scrambled face"].assign(
                trial_type="cond[scrambled face]", modulation=1.0
            ),
        ]
    )[["onset", "duration", "trial_type", "modulation"]]
    oracle = task_columns(manual, frame_times, "glover")
    pd.testing.assert_frame_equal(ours, oracle[ours.columns])
    raw = make_first_level_design_matrix(
        frame_times, manual, hrf_model="glover", drift_model=None
    )
    assert set(ours.columns) <= set(raw.columns)


@pytest.mark.parametrize(
    "levels, message",
    [
        (["face", "house", "face", "house"], "scrambled face"),  # level absent
        (["house", "scrambled face", "house", "scrambled face"], "face"),  # reference absent
        (["face", "house", "scrambled face", "cat"], "cat"),  # unknown value
        (["face", "house", "scrambled face", None], "missing"),
    ],
)
def test_invalid_categorical_runs_name_the_run_and_level(levels, message):
    with pytest.raises(ValueError, match=f"run-03.*{message}"):
        expand_events(_events(levels), TaskModel((CAT,)), run="run-03")


def test_missing_indicator_for_categorical_values():
    mod = Modulator("cond", kind="categorical", levels=("a", "b"), missing="indicator")
    expanded = expand_events(_events(["a", "b", "n/a", "a", "b"]), TaskModel((mod,)))
    by_name = {k: g.modulation.to_numpy() for k, g in expanded.groupby("trial_type", sort=False)}
    assert list(by_name) == ["task", "cond[b]", "missing_cond"]
    np.testing.assert_array_equal(by_name["cond[b]"], [0, 1, 0, 0, 1])
    np.testing.assert_array_equal(by_name["missing_cond"], [0, 0, 1, 0, 0])


def test_categorical_names_normalise_mixed_dtypes():
    mod = Modulator("cond", kind="categorical", levels=(0, 1))
    assert categorical_names(pd.Series([0, 1.0, "1", 0]), mod) == ["0", "1", "1", "0"]
```

- [ ] **Step 2: RED** — `uv run pytest -q tests/test_task_design.py` → FAIL (`ImportError: categorical_names`).

- [ ] **Step 3: Commit tests** — `git add tests/test_task_design.py && git commit -m "test: categorical indicator expansion, Nilearn oracle, per-run level errors"`

- [ ] **Step 4: Implement in `src/boldtailor/_task_design.py`**

```python
from boldtailor.model import TaskModel, level_name


def categorical_names(values, modulator):
    """Canonical level per trial (None when missing), checked against the declared levels."""
    names = [level_name(v) for v in values]
    unknown = sorted({n for n in names if n is not None and n not in modulator.levels})
    if unknown:
        raise ValueError(
            f"{modulator.column!r} values {unknown} are not levels {list(modulator.levels)}"
        )
    if modulator.missing == "error" and None in names:
        raise ValueError(f"{modulator.column!r} has missing values")
    absent = [lv for lv in modulator.levels if lv not in names]
    if absent:
        raise ValueError(f"no {modulator.column!r} trials at level(s) {absent}")
    return names


def _categorical_amplitudes(events, modulator, run):
    if modulator.column not in events:
        raise ValueError(f"run {run}: events lack modulator column {modulator.column!r}")
    try:
        names = categorical_names(events[modulator.column], modulator)
    except ValueError as error:
        raise ValueError(f"run {run}: {error}") from error
    columns = {
        f"{modulator.column}[{lv}]": np.array([n == lv for n in names], dtype=float)
        for lv in modulator.levels
        if lv != modulator.reference
    }
    observed = np.array([n is not None for n in names])
    if modulator.missing == "indicator" and not observed.all():
        columns[modulator.indicator_name] = (~observed).astype(float)
    return columns
```

Rename the existing `_modulator_amplitudes` to `_numeric_amplitudes` and add the dispatcher used by `expand_events`:

```python
def _modulator_amplitudes(events, modulator, run):
    if modulator.kind == "categorical":
        return _categorical_amplitudes(events, modulator, run)
    return _numeric_amplitudes(events, modulator, run)
```

`expand_events` already orders by `task_model.regressor_names + profiled_names`, which now contain the expanded names.

- [ ] **Step 5: GREEN** — `uv run black -q src tests && uv run pytest -q -W error tests/test_task_design.py && uv run pytest -q -W error` → PASS.

- [ ] **Step 6: Commit** — `git add src/boldtailor/_task_design.py && git commit -m "feat: design expansion builds reference-coded indicators for categorical modulators"`

---

### Task 3: `--modulator` option syntax and the report's command line

**Files:**
- Modify: `src/boldtailor/workflow/settings.py` (`parse_modulator`), `src/boldtailor/workflow/report.py` (`_modulator_flags`), `src/boldtailor/cli.py` (metavar/help)
- Test: `tests/workflow/test_settings.py`, `tests/workflow/test_cli.py`

**Interfaces:**
- Consumes: `Modulator(kind, reference)` (Task 1).
- Produces: `parse_modulator("COLUMN[:OPT[,OPT...]]") -> Modulator` with options `indicator`, `categorical`, `reference=LEVEL`; `report.modulator_text(mod) -> str` (inverse of `parse_modulator`).

- [ ] **Step 1: Failing tests**

In `tests/workflow/test_settings.py` replace `test_parse_modulator_syntax` (requirement change: new options) with:

```python
@pytest.mark.parametrize(
    "text, expected",
    [
        ("trial_type", Modulator("trial_type")),
        ("response_time:indicator", Modulator("response_time", missing="indicator")),
        ("trial_type:categorical", Modulator("trial_type", kind="categorical")),
        (
            "trial_type:categorical,reference=face",
            Modulator("trial_type", kind="categorical", reference="face"),
        ),
        (
            "cond:indicator,categorical",
            Modulator("cond", kind="categorical", missing="indicator"),
        ),
    ],
)
def test_parse_modulator_options(text, expected):
    assert parse_modulator(text) == expected


@pytest.mark.parametrize(
    "bad",
    ["", ":indicator", "rt:centered", "rt:indicator,indicator", "rt:reference=a",
     "rt:categorical,reference=", "rt:a:b"],
)
def test_parse_modulator_rejects_bad_options(bad):
    with pytest.raises(ValueError, match="modulator"):
        parse_modulator(bad)
```

In `tests/workflow/test_cli.py`, extend the command-line round-trip test's argv with `"--modulator", "trial_type:categorical,reference=house"` alongside its existing modulators, and add:

```python
def test_categorical_modulator_flag_round_trips(four_runs):
    root, _ = four_runs
    args = cli.build_parser().parse_args(
        _argv(root, "--modulator", "trial_type:categorical,reference=1,indicator")
    )
    settings = cli.settings_from_args(args)
    assert settings.modulators == (
        Modulator("trial_type", kind="categorical", reference="1", missing="indicator"),
    )
    again = cli.settings_from_args(
        cli.build_parser().parse_args(shlex.split(report.command_line(settings))[1:])
    )
    assert again == settings
```

(import `shlex` and `from boldtailor.workflow import report` if not already imported; match how the existing round-trip test strips the leading `boldtailor` token.)

- [ ] **Step 2: RED** — `uv run pytest -q tests/workflow/test_settings.py tests/workflow/test_cli.py` → FAIL.

- [ ] **Step 3: Commit tests** — `git commit -am "test: --modulator categorical and reference options round-trip"` (after `git add` of both files).

- [ ] **Step 4: Implement**

`settings.py`:

```python
_MODULATOR_SYNTAX = "COLUMN[:indicator|categorical|reference=LEVEL,...]"


def _modulator_options(text, options):
    seen, values = set(), {}
    for option in options:
        key, _, value = option.partition("=")
        if key in seen or key not in ("indicator", "categorical", "reference"):
            raise ValueError(f"modulator must be {_MODULATOR_SYNTAX}, not {text!r}")
        if (key == "reference") != bool(value):
            raise ValueError(f"modulator must be {_MODULATOR_SYNTAX}, not {text!r}")
        seen.add(key)
        values[key] = value
    if "reference" in values and "categorical" not in values:
        raise ValueError(f"modulator reference needs categorical, not {text!r}")
    return values


def parse_modulator(text):
    column, sep, rest = str(text).partition(":")
    if not column or ":" in rest or (sep and not rest):
        raise ValueError(f"modulator must be {_MODULATOR_SYNTAX}, not {text!r}")
    values = _modulator_options(text, rest.split(",") if rest else [])
    return Modulator(
        column,
        missing="indicator" if "indicator" in values else "error",
        kind="categorical" if "categorical" in values else "numeric",
        reference=values.get("reference"),
    )
```

`report.py`:

```python
def modulator_text(mod):
    options = ["indicator"] if mod.missing == "indicator" else []
    if mod.kind == "categorical":
        options.append("categorical")
        if mod.reference is not None:
            options.append(f"reference={mod.reference}")
    return mod.column + (":" + ",".join(options) if options else "")
```

and in `_modulator_flags` use `flags += _quoted("--modulator", modulator_text(mod))`.

`cli.py`: `metavar="COLUMN[:OPTIONS]"`, help: `"task modulator; OPTIONS is a comma list of indicator, categorical, reference=LEVEL; repeat to list all; default detects response_time and a categorical trial_type"`.

- [ ] **Step 5: GREEN** — `uv run black -q src tests && uv run pytest -q -W error tests/workflow && uv run pytest -q -W error` → PASS.

- [ ] **Step 6: Commit** — `git add src/boldtailor/workflow/settings.py src/boldtailor/workflow/report.py src/boldtailor/cli.py && git commit -m "feat: --modulator accepts categorical and reference options"`

---

### Task 4: Workflow detection, per-run validation, and encoding predictors

**Files:**
- Modify: `src/boldtailor/workflow/inputs.py` (detection, notes, `validate_glm_events`, `run_summary`), `src/boldtailor/workflow/beta_series.py` (`trial_predictors`), `src/boldtailor/workflow/outputs.py` (`encoding_predictors`)
- Test: `tests/workflow/test_inputs.py`, `tests/workflow/test_beta_series.py`; update expectations (requirement change `trial_type` → `trial_type[1]`) in `tests/workflow/test_analysis.py`, `test_outputs.py`, `test_run.py`, `test_cli.py`

**Interfaces:**
- Consumes: Tasks 1-2 (`level_name`, `categorical_names`, `Modulator.with_levels`).
- Produces:
  - `inputs.observed_levels(tables, column) -> set[str]`
  - `detect_task_model(events_tables, modulators=None, labels=None) -> TaskModel` — automatic: `response_time` (indicator) and categorical `trial_type` with ≥ 2 observed levels; explicit unresolved categorical modulators resolved from all runs; resolution failures raise `InputError` naming the column.
  - `task_model_notes(...)` returns `["trial_type has fewer than two levels; not used as a modulator"]` when applicable (constant `TRIAL_TYPE_NOTE`).
  - `validate_glm_events(events, task_model)` checks each categorical modulator with `categorical_names` (errors are prefixed with the run label by `_trim` and become `InputError`).
  - `run_summary` columns `n_<column>_<level>` per categorical level (replacing `type_0`/`type_1`).
  - `trial_predictors(runs, task_model)` returns, per run, numeric modulator columns plus indicator columns named as the regressors (NaN where the value is missing).
  - Metadata `encoding_predictors == list(task_model.regressor_names)` minus nothing (`["task", *regressors after task]`).

- [ ] **Step 1: Failing tests**

In `tests/workflow/test_inputs.py`, replace `test_detection_uses_rt_and_trial_type_when_present` (requirement change) with:

```python
TT = Modulator("trial_type", kind="categorical", levels=("0", "1"))


def test_detection_uses_rt_and_categorical_trial_type(events):
    model = inputs.detect_task_model([events, events])
    assert model == TaskModel((Modulator("response_time", missing="indicator"), TT))
    assert model.regressor_names == ("task", "response_time", "trial_type[1]")
    without_rt = inputs.detect_task_model([events.drop(columns="response_time"), events])
    assert without_rt == TaskModel((TT,))
    assert inputs.detect_task_model([events[["onset", "duration"]]]) == TaskModel()


def test_string_and_mixed_dtype_trial_types_become_one_level_set(events):
    words = events.assign(trial_type=["face", "house", "scrambled"] * (len(events) // 3))
    model = inputs.detect_task_model([words, words])
    assert model.regressor_names[-2:] == ("trial_type[house]", "trial_type[scrambled]")
    as_text = events.assign(trial_type=events.trial_type.astype(str))
    assert inputs.detect_task_model([events, as_text]).regressor_names[-1] == "trial_type[1]"


def test_single_level_trial_type_is_left_out_with_a_note(events):
    flat = events.assign(trial_type="face")
    assert inputs.detect_task_model([flat]).regressor_names == ("task", "response_time")
    assert inputs.task_model_notes([flat]) == [inputs.TRIAL_TYPE_NOTE]


def test_explicit_categorical_resolves_levels_and_checks_reference(events):
    wanted = (Modulator("trial_type", kind="categorical", reference="1"),)
    model = inputs.detect_task_model([events, events], wanted)
    assert model.regressor_names == ("task", "trial_type[0]")
    bad = (Modulator("trial_type", kind="categorical", reference="7"),)
    with pytest.raises(inputs.InputError, match="trial_type.*7"):
        inputs.detect_task_model([events], bad)


def test_run_missing_a_level_is_an_input_error_naming_the_run(four_runs, settings_for):
    root, _ = four_runs
    path = sorted((root / "sub-07" / "ses-nsd10" / "func").glob("*run-02_events.tsv"))[0]
    table = pd.read_csv(path, sep="\t")
    table.assign(trial_type=0).to_csv(path, sep="\t", index=False)
    with pytest.raises(inputs.InputError, match="run-02.*trial_type.*'1'"):
        inputs.load_session(settings_for(root))


def test_run_summary_counts_each_level(four_runs, settings_for):
    root, _ = four_runs
    runs = inputs.load_session(settings_for(root))
    model = inputs.detect_task_model([r.events for r in runs])
    summary = inputs.run_summary(runs, model)
    assert {"n_trial_type_0", "n_trial_type_1"} <= set(summary.columns)
    assert (summary.n_trial_type_0 + summary.n_trial_type_1 == summary.trials).all()
```

Update `test_load_session_detects_the_model_and_builds_the_glm` to expect contrasts `{"task", "response_time", "trial_type[1]"}`; delete the `("trial_type", 2)` case from the parametrized invalid-covariate test only if it asserted the binary rule, replacing it with an expectation that a run where `trial_type` is entirely `2` (so levels 0 and 1 are absent in that run) raises `InputError` matching `"trial_type"`; keep the `np.nan` case (now "missing values").

In `tests/workflow/test_beta_series.py`: `ORDER = ["response_time", "trial_type[1]"]`; the column assertion at the `trial_predictors` test becomes `["trial_type[1]"]`; keep `test_trial_predictors_reject_nonbinary_trial_type` and the case at its second use (value `3` is now "not levels", still a `ValueError` matching `trial_type`). Add:

```python
def test_trial_predictors_emit_indicator_columns_with_missing_as_nan(session):
    runs, _ = session
    mod = Modulator("trial_type", kind="categorical", levels=("0", "1"), missing="indicator")
    first = runs[0].events.copy()
    first.loc[first.index[0], "trial_type"] = np.nan
    runs = [replace(runs[0], events=first), *runs[1:]]
    tables = beta_series.trial_predictors(runs, TaskModel((mod,)))
    assert list(tables[0].columns) == ["trial_type[1]"]
    assert np.isnan(tables[0].iloc[0, 0])
    assert set(tables[0].iloc[1:, 0]) <= {0.0, 1.0}
```

(`session` is the existing fixture in that file returning loaded runs; adapt the unpacking to its actual return value.)

In `test_analysis.py`, `test_outputs.py`, `test_run.py`, `test_cli.py`: replace the regressor/contrast/map name `"trial_type"` with `"trial_type[1]"` wherever it denotes the fitted regressor (not the events column); `test_cli.py::test_dry_run_with_string_trial_type_is_task_only` becomes `test_dry_run_with_string_trial_type_expands_indicators`, asserting the dry-run `task_model["regressors"]` ends with the two non-reference string levels and that `plan["notes"]` does not contain the old binary note.

- [ ] **Step 2: RED** — `uv run pytest -q tests/workflow` → FAIL.

- [ ] **Step 3: Commit tests** — `git add tests/workflow && git commit -m "test: workflow detects categorical trial_type, validates levels per run, encodes indicators"`

- [ ] **Step 4: Implement**

`inputs.py` — replace `TRIAL_TYPE_NOTE`, `_AUTOMATIC`, `_binary`, `_detectable`, `detect_task_model`, `task_model_notes`:

```python
from boldtailor._task_design import categorical_names
from boldtailor.model import ModelSpec, Modulator, TaskModel, level_name

TRIAL_TYPE_NOTE = "trial_type has fewer than two levels; not used as a modulator"


def observed_levels(tables, column):
    names = {level_name(v) for table in tables for v in table[column]}
    names.discard(None)
    return names


def _everywhere(column, tables):
    return bool(tables) and all(column in t.columns for t in tables)


def _automatic(tables):
    modulators = []
    if _everywhere("response_time", tables):
        modulators.append(Modulator("response_time", missing="indicator"))
    if _everywhere("trial_type", tables):
        levels = observed_levels(tables, "trial_type")
        if len(levels) >= 2:
            modulators.append(Modulator("trial_type", kind="categorical", levels=tuple(levels)))
    return tuple(modulators)


def _resolved(modulator, tables):
    if modulator.resolved:
        return modulator
    try:
        return modulator.with_levels(observed_levels(tables, modulator.column))
    except ValueError as error:
        raise InputError(f"modulator {modulator.column!r}: {error}") from error


def detect_task_model(events_tables, modulators=None, labels=None):
    """The task model from event columns, or explicit modulators checked per run.

    Detection takes ``response_time`` and a categorical ``trial_type`` with at
    least two levels; ``labels`` (BIDS run labels) name the run in errors.
    """
    tables = list(events_tables)
    if modulators is None:
        modulators = _automatic(tables)
    labels = labels or [f"run {i}" for i in range(1, len(tables) + 1)]
    for label, table in zip(labels, tables, strict=True):
        missing = [m.column for m in modulators if m.column not in table.columns]
        if missing:
            raise InputError(f"{label} events lack modulator column(s) {missing}")
    return TaskModel(tuple(_resolved(m, tables) for m in modulators))


def task_model_notes(events_tables, modulators=None):
    """Why automatic detection left out a column that every run has."""
    tables = list(events_tables)
    if modulators is None and _everywhere("trial_type", tables):
        if len(observed_levels(tables, "trial_type")) < 2:
            return [TRIAL_TYPE_NOTE]
    return []
```

`validate_glm_events`: replace the `trial_type` block with

```python
    for modulator in task_model.modulators:
        if modulator.kind == "categorical":
            categorical_names(events[modulator.column], modulator)
```

and update its docstring to "Require an observed positive RT and every categorical level, if modeled." `_run_row(run, columns)` → `_run_row(run, task_model)`:

```python
def _level_counts(run, modulator):
    names = [level_name(v) for v in run.events[modulator.column]]
    return {f"n_{modulator.column}_{lv}": names.count(lv) for lv in modulator.levels}


def _run_row(run, task_model):
    row = dict(
        run=run.label,
        trials=len(run.events),
        scans=run.image.shape[0],
        retained_scans=len(run.frame_times),
        dropped_scans=len(run.retained_frames) and int(run.retained_frames[0]),
        first_frame_seconds=run.frame_times[0],
    )
    if "response_time" in _columns(task_model):
        row["mean_rt_seconds"] = run.events.response_time.mean()
    for modulator in task_model.modulators:
        if modulator.kind == "categorical":
            row.update(_level_counts(run, modulator))
    return row


def run_summary(runs, task_model):
    return pd.DataFrame([_run_row(r, task_model) for r in runs])
```

`beta_series.py` — delete `_check_trial_type` and `_numeric_predictors`; add:

```python
from boldtailor._task_design import categorical_names


def _indicator_frame(values, modulator, label):
    try:
        names = categorical_names(values, replace(modulator, missing="indicator"))
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error
    return pd.DataFrame(
        {
            name: [np.nan if n is None else float(n == lv) for n in names]
            for name, lv in zip(
                modulator.regressor_names,
                [lv for lv in modulator.levels if lv != modulator.reference],
            )
        },
        index=values.index,
    )


def _predictor_frame(run, modulator):
    if modulator.column not in run.events:
        raise ValueError(f"{run.label}: encoding needs event column {modulator.column!r}")
    values = run.events[modulator.column]
    if modulator.kind == "categorical":
        return _indicator_frame(values, modulator, run.label)
    return pd.to_numeric(values, errors="raise").rename(modulator.column).to_frame()


def trial_predictors(runs, task_model):
    """Predictor columns per run; unavailable behavior excludes only encoding rows."""
    tables = []
    for run in runs:
        frames = [_predictor_frame(run, m) for m in task_model.modulators]
        table = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=run.events.index)
        _censor_response_time(table)
        tables.append(table)
    return tables
```

(`replace` is `dataclasses.replace`; missing values become NaN predictors, which encoding already excludes. `categorical_names` with `missing="indicator"` still rejects unknown values and absent levels.)

`outputs.py`: `encoding_predictors=list(task_model.regressor_names),`.

- [ ] **Step 5: GREEN** — `uv run black -q src tests && uv run pytest -q -W error tests/workflow && uv run pytest -q -W error` → PASS.

- [ ] **Step 6: Commit** — `git add src/boldtailor/workflow && git commit -m "feat: workflow resolves categorical levels across runs, validates each run, and encodes indicator predictors"`

---

### Task 5: Metadata prose, report, and an end-to-end string `trial_type` run

**Files:**
- Modify: `src/boldtailor/workflow/outputs.py` (`_task_model_description`, `_model_prose`), `src/boldtailor/workflow/report.py` (`_inputs_body`)
- Test: `tests/workflow/test_run.py`, `tests/workflow/test_outputs.py`, `tests/workflow/test_report.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: metadata key `categorical_modulators` (dict column → `{"levels": [...], "reference": ..., "coding": "..."}`) replacing the binary `trial_type` prose; report inputs list item per categorical modulator.

- [ ] **Step 1: Failing tests**

`tests/workflow/test_run.py`:

```python
def test_string_trial_type_run_writes_indicator_maps_and_describes_them(
    four_runs, settings_for, tmp_path
):
    import json

    import pandas as pd

    root, _ = four_runs
    for path in (root / "sub-07" / "ses-nsd10" / "func").glob("*_events.tsv"):
        table = pd.read_csv(path, sep="\t")
        labels = ["face", "house", "scrambled face"] * (len(table) // 3 + 1)
        table.assign(trial_type=labels[: len(table)]).to_csv(path, sep="\t", index=False)
    settings = settings_for(
        root, output_dir=tmp_path / "out", hrf_library="canonical", ridge_mode="off"
    )
    result = workflow_run.run_workflow(settings)
    func = settings.output_dir / "sub-07" / "ses-nsd10" / "func"
    meta = json.loads((func / "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json").read_text())
    assert meta["task_model"]["regressors"] == [
        "task", "response_time", "trial_type[house]", "trial_type[scrambled face]",
    ]
    assert meta["categorical_modulators"]["trial_type"]["reference"] == "face"
    effects = next(p for p in result.paths if "desc-OptimizedGLM_stat-effects" in p.name)
    import nibabel as nib

    names = list(nib.load(effects).header.get_axis(0).name)
    assert names == meta["task_model"]["regressors"]
    html = result.report_path.read_text()
    assert "trial_type[scrambled face]" in html and "reference face" in html
```

(Check the fixture's trial count per run first; each run must contain all three labels — the slicing above guarantees it for runs with ≥ 3 trials.)

`tests/workflow/test_outputs.py`: in the metadata prose test, assert `meta["categorical_modulators"]["trial_type"] == {"levels": ["0", "1"], "reference": "0", "coding": "One 0/1 indicator per non-reference level, uncentered; each coefficient is the level response minus the reference response"}` and that the old `"trial_type"` prose key is absent (requirement change).

`tests/workflow/test_report.py`: render with `TaskModel((Modulator("cond", kind="categorical", levels=("a<b", "c")),))` and assert `"cond: levels a&lt;b, c (reference a&lt;b)"` appears (escaping).

- [ ] **Step 2: RED**, **Step 3: commit tests** (`git add tests/workflow && git commit -m "test: categorical metadata, report text, end-to-end string trial_type run"`).

- [ ] **Step 4: Implement**

`outputs.py`:

```python
_CATEGORICAL_CODING = (
    "One 0/1 indicator per non-reference level, uncentered; each coefficient "
    "is the level response minus the reference response"
)


def _task_model_description(task_model):
    columns = [m.column for m in task_model.modulators]
    modulated = f" modulated by {', '.join(columns)}" if columns else ""
    return (
        f"One task regressor per presentation{modulated}. Modulators are "
        "uncentered; the task contrast is the response at numeric modulator "
        "value zero and at each categorical modulator's reference level."
    )


def _categorical_prose(task_model):
    return {
        m.column: dict(levels=list(m.levels), reference=m.reference, coding=_CATEGORICAL_CODING)
        for m in task_model.modulators
        if m.kind == "categorical"
    }
```

In `_model_prose`, delete the `trial_type` block and add `categorical = _categorical_prose(task_model); if categorical: prose["categorical_modulators"] = categorical`. Update the `task` prose string to "One unit per presentation; the response with every numeric modulator at zero, every categorical modulator at its reference level, and every missing-value indicator at zero".

`report.py` `_inputs_body`, after the task-regressor item:

```python
    for mod in task_model.modulators:
        if mod.kind == "categorical":
            levels = ", ".join(mod.levels)
            items.append(
                f"<li>{html.escape(mod.column)}: levels {html.escape(levels)} "
                f"(reference {html.escape(mod.reference)})</li>"
            )
```

- [ ] **Step 5: GREEN** — `uv run black -q src tests && uv run pytest -q -W error` → PASS.

- [ ] **Step 6: Commit** — `git add src/boldtailor/workflow && git commit -m "feat: metadata and report describe categorical modulators"`

---

### Task 6: NSD examples use `trial_type[1]`

**Files:**
- Modify: `examples/NSD/conftest.py` (lines writing `regressors=["task", "response_time", "trial_type"]`), `examples/NSD/test_nsd_notebooks.py`, `examples/NSD/test_session_hrf.py`, `examples/NSD/nsd_workflow.ipynb`, `examples/NSD/nsd_multisession.ipynb`, `examples/NSD/README.md`, and any `examples/NSD/*.py` reader that names the `trial_type` regressor or map.

- [ ] **Step 1:** `grep -rn "trial_type" examples/NSD` and classify each hit: events-column uses stay; fitted-regressor, contrast, or map-name uses change to `trial_type[1]`. Update the opt-in tests first, run `uv run pytest -q examples/NSD` → expect FAIL where readers still use the old name; commit tests (`test(examples): NSD readers expect trial_type[1]`).
- [ ] **Step 2:** Update readers, saved-session fixture writers in `conftest.py`, notebooks (code cells only), and README prose.
- [ ] **Step 3:** `uv run pytest -q examples/NSD` → PASS; `uv run pytest -q examples/NSD --run-notebooks -k workflow` → PASS; `uv run pytest -q -W error` → PASS.
- [ ] **Step 4: Commit** — `git add examples/NSD && git commit -m "refactor(examples): NSD readers and notebooks use the trial_type[1] regressor"`

---

### Task 7: Documentation

**Files:**
- Modify: `docs/user-guide.md` (modulator section, `--modulator` row, Common problems), `docs/api.md` (`Modulator`, `level_name`), `docs/superpowers/specs/2026-10-03-workflow-cli-design.md` (modulator detection paragraph and CLI synopsis), `README.md` if it describes modulator detection
- Test: `tests/test_docs_mention_cli.py`

- [ ] **Step 1: Failing test** (append):

```python
def test_user_guide_documents_categorical_modulators():
    guide = Path("docs/user-guide.md").read_text()
    api = Path("docs/api.md").read_text()
    assert "categorical" in guide and "reference=" in guide and "trial_type[" in guide
    assert 'kind="categorical"' in api and "level_name" in api
```

- [ ] **Step 2:** RED, commit test.
- [ ] **Step 3:** Write the docs: the reference-coding rule (task = reference-level response; each indicator = level − reference), default reference (first sorted level; numeric order when all numeric), the `--modulator COLUMN[:indicator,categorical,reference=LEVEL]` syntax with the three examples from the spec, the every-level-in-every-run requirement and its exit code 2, missing-value policy, the `n_<column>_<level>` run-summary columns, and the `trial_type[1]` naming for binary columns (a behaviour change from earlier releases). In the workflow spec, replace the R12 paragraph with the categorical rule.
- [ ] **Step 4:** `uv run pytest -q -W error` → PASS; commit `docs: categorical modulators and the --modulator option syntax`.

---

### Task 8: Whole-branch verification

- [ ] `uv run pytest -q -W error`, `uv run pytest -q examples/NSD`, `uv run black --check src tests examples`, `git diff --check`.
- [ ] `grep -rn "type_0\|type_1\|binary 0/1" src docs/user-guide.md` returns nothing relevant.
- [ ] CLI smoke on the four-run synthetic dataset with string `trial_type` (`face`/`house`/`scrambled`): `uv run boldtailor run ... --dry-run --ridge-mode off` prints regressors ending `trial_type[house]`, `trial_type[scrambled]`; a real run with `--hrf-library canonical --ridge-mode off --no-surface-maps` exits 0 and writes the report; removing all `house` trials from one run makes the dry run exit 2 naming that run.
