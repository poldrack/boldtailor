# Workflow Package and CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the NSD notebook pipeline into `boldtailor.workflow`, generalise its inputs to any BIDS task on fMRIPrep CIFTI output, and expose it as the `boldtailor run` console script that writes a BIDS derivative with an HTML report.

**Architecture:** A frozen `WorkflowSettings` dataclass is the single source of truth for the CLI, the stage functions, the writer, and the saved metadata. Stage functions (`glms`, `reliability`, `betas`, `summaries`) take the loaded runs, the settings, and earlier results; every beta-series estimator yields one `BetaModel`, and one writer serialises it. The existing example modules move with their tests, module by module, and NSD constants become settings fields. Legacy scripts and result reuse are retired; output descriptors lose the `notebook` prefix.

**Tech Stack:** Python 3.12, numpy/pandas/scipy, nilearn 0.14 (pinned <0.15), nibabel, matplotlib (becomes a runtime dependency), argparse, pytest, uv, Black.

**Spec:** `docs/superpowers/specs/2026-10-03-workflow-cli-design.md`

## Global Constraints

- `uv run` for every command; Black formatting; `uv run pytest -q -W error` must pass after every task (default suite is `tests/` only).
- Every `__init__.py` is empty. No package module imports from `examples`.
- RED-GREEN: tests are written and committed before implementation in every task; never weaken a test to pass.
- Only CIFTI space `fsLR-91k` is supported; the space entity string is `space-fsLR_den-91k`.
- Output descriptors carry no `notebook` prefix. The stem is `<sub>/<ses>/func/<sub>_<ses>_task-<task>`.
- Modulators are never centered (package-wide).
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- The spec's "settings file" is realised as `<stem>_desc-boldtailor_metadata.json`: the existing analysis metadata dictionary plus `settings`, `runs`, `task_model`, `library_fingerprint`, `skipped`, and `report`.
- Descriptor names keep the existing estimator vocabulary (`CanonicalGLM`, `OptimizedGLM`, `GLMComparison`, `HRF`, `HRFAll`, `HRFOdd`, `HRFEven`, `HRFOddToEven`, `HRFEvenToOdd`, `HRFReliability`, `CanonicalTrialOLS`, `OptimizedTrialOLS`, `CanonicalTrialRidge`, `OptimizedTrialRidge`, `CanonicalTrialRidgeCV`, `OptimizedTrialRidgeCV`, `CanonicalTrialFractionalCV`, `OptimizedTrialFractionalCV`, tuning descriptors `{Canonical|Optimized}{RidgeCV|FractionalCV}{All|Odd|Even|OddToEven|EvenToOdd}`), because the multisession readers key on them. Task 1 records this and the two other deviations in the spec.

## Review Focus

1. A task whose events lack both `response_time` and `trial_type`: the workflow must run a task-only model, skip RT correlations, and say so in the report (Task 5 and Task 12 tests).
2. `--modulator` naming a column absent from one run: an input error (exit 2) naming the run, before any fitting (Task 5 test).
3. Two `derivatives/fmriprep*` directories under the BIDS root with no `--fmriprep-dir`: a settings error listing both (Task 3 test).
4. Fewer than two odd or two even runs with every stage enabled: the reliability stage is skipped with a recorded reason and the run completes (Task 11 test).
5. Running twice into the same output directory with the default `existing_results="error"`: the second run stops before loading data and exits 1 (Task 11 and Task 13 tests).

---

### Task 1: Spec amendments and repository prerequisites

**Files:**
- Modify: `docs/superpowers/specs/2026-10-03-workflow-cli-design.md`
- Modify: `pyproject.toml`
- Create: `src/boldtailor/workflow/__init__.py` (empty)
- Create: `tests/workflow/__init__.py` (empty)
- Test: `tests/workflow/test_package_layout.py`

**Interfaces:**
- Produces: the `boldtailor.workflow` package exists; matplotlib is a runtime dependency; `[project.scripts] boldtailor = "boldtailor.cli:main"` is declared (the module arrives in Task 13; declaring the script early keeps one pyproject edit).

- [ ] **Step 1: Write the failing test**

```python
# tests/workflow/test_package_layout.py
"""The workflow subpackage is importable, empty at the package level, and ships its runtime deps."""

import importlib
import tomllib
from pathlib import Path


def test_workflow_package_is_empty_and_importable():
    package = importlib.import_module("boldtailor.workflow")
    assert Path(package.__file__).read_text() == ""


def test_matplotlib_is_a_runtime_dependency_and_the_script_is_declared():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert any(dep.startswith("matplotlib") for dep in project["dependencies"])
    assert project["scripts"] == {"boldtailor": "boldtailor.cli:main"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest -q tests/workflow/test_package_layout.py`
Expected: FAIL (`ModuleNotFoundError: boldtailor.workflow`).

- [ ] **Step 3: Create the packages and edit pyproject**

```bash
mkdir -p src/boldtailor/workflow tests/workflow
: > src/boldtailor/workflow/__init__.py
: > tests/workflow/__init__.py
```

In `pyproject.toml`, add `"matplotlib>=3.11.1",` to `[project] dependencies` (keep the dev-group entry; uv tolerates both) and add after the `[project]` table:

```toml
[project.scripts]
boldtailor = "boldtailor.cli:main"
```

Then `uv sync --group dev`.

- [ ] **Step 4: Amend the spec** (three deviations, each one paragraph):

1. Under "Outputs", replace the descriptor list with the vocabulary in Global Constraints and add: "The spec's settings file is `<stem>_desc-boldtailor_metadata.json`, which carries the analysis metadata the multisession readers already consume plus `settings`, `runs`, `task_model`, `library_fingerprint`, `skipped`, and `report`."
2. Under "Package layout", change `rt_diagnostics` from "→ plots" to "retired (used only by the legacy scripts; the notebook's RT check uses `boldtailor.diagnostics.even_run_points` directly)".
3. Under "Testing", add: "The synthetic BIDS builders live in `tests/workflow/synthetic_bids.py`; `tests/workflow/conftest.py` and `examples/NSD/conftest.py` both register fixtures from it."

- [ ] **Step 5: Run the test and the suite**

Run: `uv run pytest -q -W error tests/workflow/test_package_layout.py && uv run pytest -q -W error`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/boldtailor/workflow/__init__.py tests/workflow docs/superpowers/specs/2026-10-03-workflow-cli-design.md
git commit -m "chore: boldtailor.workflow package, matplotlib runtime dependency, console script declaration, spec amendments"
```

---

### Task 2: Remove modulator centering from the package

**Files:**
- Modify: `src/boldtailor/model.py:29-50` (`Modulator`)
- Modify: `src/boldtailor/_task_design.py:38` (`_modulator_amplitudes`)
- Modify: `tests/test_task_model.py`, `tests/test_task_design.py`, `tests/test_hrf_glm.py:506,539`, `tests/test_hrf_cv.py:13-14`, `tests/test_hrf_selection.py:277-278`, `tests/test_design.py:220`
- Modify: `docs/user-guide.md`, `docs/api.md` (wherever `center=` appears)

**Interfaces:**
- Produces: `Modulator(column: str, missing: str = "error")` with `to_dict() -> {"column", "missing"}`; `expand_events` amplitudes are raw observed values (unobserved stay 0).

- [ ] **Step 1: Rewrite the centering tests to assert uncentered behaviour (RED)**

In `tests/test_task_design.py`:
- `test_expansion_orders_regressors_and_computes_amplitudes`: expect the RT amplitude column to equal the raw RT values with NaN → 0 (not `rt - mean`). Replace the expected vector accordingly (the fixture RTs are listed at the top of the file; the expected amplitudes are those values with missing set to 0.0).
- `test_indicator_absent_when_nothing_is_missing`: expect `RT` column equal to raw `rt`.
- Delete `test_uncentered_modulator_keeps_raw_values` (it becomes the only behaviour) and add:

```python
def test_modulator_has_no_center_option():
    import pytest
    from boldtailor.model import Modulator

    with pytest.raises(TypeError):
        Modulator("response_time", center=False)
    assert Modulator("response_time").to_dict() == {
        "column": "response_time",
        "missing": "error",
    }
```

In `tests/test_task_model.py`: remove `center` from the `to_dict` expectations (lines 22-32), the fingerprint-differs-with-center test (line 46) and the center-type rejections (lines 69-71); change `test_subset_requires_identical_shared_modulators` to use `missing` as the differing attribute (`Modulator("response_time", missing="indicator")` is not a subset of a model with `Modulator("response_time")`).

In `tests/test_hrf_glm.py:506`, `tests/test_hrf_cv.py:13-14`, `tests/test_hrf_selection.py:277-278`, `tests/test_design.py:220`: delete the `center=...` keyword from every `Modulator(...)` call. At `tests/test_hrf_glm.py:506` the "mismatched task model" case must still mismatch: use `Modulator("trial_type", missing="indicator")` versus `Modulator("trial_type")`.

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest -q tests/test_task_design.py tests/test_task_model.py tests/test_hrf_glm.py tests/test_hrf_cv.py tests/test_hrf_selection.py tests/test_design.py`
Expected: failures in the rewritten tests (`center` still accepted / amplitudes still centered).

- [ ] **Step 3: Commit the tests**

```bash
git add tests
git commit -m "test: modulators are uncentered; Modulator has no center option"
```

- [ ] **Step 4: Implement**

`src/boldtailor/model.py`: delete the `center` field, its validation line, and the `"center"` key in `to_dict`. Docstring: "One parametric task regressor derived from a raw events column; values enter uncentered, so the task regressor is the response at modulator value zero."

`src/boldtailor/_task_design.py:38`: replace `offset = values[observed].mean() if modulator.center else 0.0` and the subtraction with `amplitude[observed] = values[observed]`.

Docs: remove `center=` from every example in `docs/user-guide.md` and `docs/api.md`; add one sentence where `Modulator` is introduced: "Modulators are not centered; the `task` contrast is the response at modulator value zero, and R², ΔR², and HRF selection are unchanged by this choice."

- [ ] **Step 5: GREEN and full suite**

Run: `uv run black -q src tests && uv run pytest -q -W error`
Expected: PASS. (The `examples/NSD` tests still reference `center=`; they are updated when their modules are ported in Tasks 4-6 and retired in Task 14.)

- [ ] **Step 6: Commit**

```bash
git add src docs
git commit -m "feat: modulators are never centered; task contrast is the response at modulator value zero"
```

---

### Task 3: `WorkflowSettings`

**Files:**
- Create: `src/boldtailor/workflow/settings.py`
- Test: `tests/workflow/test_settings.py`

**Interfaces:**
- Produces:

```python
SUPPORTED_SPACES = ("fsLR-91k",)
SPACE_ENTITIES = {"fsLR-91k": "space-fsLR_den-91k"}
RIDGE_MODES = ("fractional_cv", "cv", "fixed", "off")
HRF_LIBRARIES = ("default", "sobol", "expanded", "canonical")
STAGES = ("glms", "reliability", "betas", "summaries")
EXISTING_RESULTS = ("error", "overwrite")

def resolve_fmriprep_dir(bids_dir: Path) -> Path
def parse_modulator(text: str) -> Modulator          # "col" or "col:indicator"

@dataclass(frozen=True, kw_only=True)
class WorkflowSettings:
    bids_dir: Path; subject: str; session: str; task: str
    fmriprep_dir: Path | None = None; output_dir: Path | None = None
    space: str = "fsLR-91k"; modulators: tuple[Modulator, ...] | None = None
    hrf_library: str = "default"; hrf_n_samples: int = 512; hrf_seed: int = 0
    hrf_selection_rt: bool = True
    ridge_mode: str = "fractional_cv"
    ridge_fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    ridge_alphas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    ridge_percentile: float = 90.0; ridge_alpha: float = 0.1
    encoding_mode: str = "within_run"
    stages: frozenset[str] = frozenset(STAGES)
    surface_maps: bool = True; surface_meshes: Mapping[str, Path] | None = None
    n_jobs: int = 4; block_size: int = 4096; max_grayordinates: int | None = None
    existing_results: str = "error"
    # derived
    def output_name(self) -> str
    @property stem(self) -> str              # "<sub>/<ses>/func/<sub>_<ses>_task-<task>"
    @property space_entity(self) -> str
    def build_library(self) -> HrfLibrary
    def to_dict(self) -> dict ; @classmethod from_dict(cls, values) -> WorkflowSettings
```

After `__post_init__`, `fmriprep_dir` and `output_dir` are always `Path`s (defaults resolved).

- [ ] **Step 1: Write the failing tests**

```python
# tests/workflow/test_settings.py
"""WorkflowSettings: defaults, derived paths, validation, round trip."""

import pytest

from boldtailor.hrf_library import default_hrf_library
from boldtailor.model import Modulator
from boldtailor.workflow.settings import (
    WorkflowSettings,
    parse_modulator,
    resolve_fmriprep_dir,
)


@pytest.fixture
def bids(tmp_path):
    root = tmp_path / "bids"
    (root / "derivatives" / "fmriprep-25.2.5").mkdir(parents=True)
    return root


def required(bids, **kwargs):
    return WorkflowSettings(
        bids_dir=bids, subject="sub-07", session="ses-nsd10", task="nsdcore", **kwargs
    )


def test_defaults_derive_fmriprep_and_output_directories(bids):
    settings = required(bids)
    assert settings.fmriprep_dir == bids / "derivatives" / "fmriprep-25.2.5"
    assert settings.output_name() == "boldtailor_hrf-default512s0_ridge-fractionalcv"
    assert settings.output_dir == bids / "derivatives" / settings.output_name()
    assert settings.stem == "sub-07/ses-nsd10/func/sub-07_ses-nsd10_task-nsdcore"
    assert settings.space_entity == "space-fsLR_den-91k"
    assert settings.stages == frozenset({"glms", "reliability", "betas", "summaries"})


@pytest.mark.parametrize(
    "kwargs, name",
    [
        (dict(hrf_library="sobol", hrf_n_samples=8, hrf_seed=3, ridge_mode="cv"), "boldtailor_hrf-sobol8s3_ridge-cv"),
        (dict(hrf_library="expanded", ridge_mode="fixed", ridge_alpha=0.1), "boldtailor_hrf-expanded_ridge-fixed0.1"),
        (dict(hrf_library="canonical", ridge_mode="off"), "boldtailor_hrf-canonical_ridge-off"),
    ],
)
def test_output_name_encodes_library_and_ridge_mode(bids, kwargs, name):
    assert required(bids, **kwargs).output_name() == name


def test_explicit_paths_are_expanded_and_kept(bids, tmp_path):
    out = tmp_path / "elsewhere"
    settings = required(bids, fmriprep_dir=bids / "derivatives" / "fmriprep-25.2.5", output_dir=out)
    assert settings.output_dir == out


def test_fmriprep_resolution_requires_a_unique_directory(tmp_path):
    root = tmp_path / "bids"
    (root / "derivatives").mkdir(parents=True)
    with pytest.raises(ValueError, match="no derivatives/fmriprep"):
        resolve_fmriprep_dir(root)
    (root / "derivatives" / "fmriprep-24.0").mkdir()
    assert resolve_fmriprep_dir(root) == root / "derivatives" / "fmriprep-24.0"
    (root / "derivatives" / "fmriprep-25.2.5").mkdir()
    with pytest.raises(ValueError, match="fmriprep-24.0.*fmriprep-25.2.5"):
        resolve_fmriprep_dir(root)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(space="MNI152NLin2009cAsym"), "space"),
        (dict(hrf_library="grid"), "hrf_library"),
        (dict(ridge_mode="lasso"), "ridge_mode"),
        (dict(ridge_mode="fixed", ridge_alpha=0.0), "ridge_alpha"),
        (dict(ridge_percentile=101), "ridge_percentile"),
        (dict(encoding_mode="pooled"), "encoding_mode"),
        (dict(stages=frozenset({"betas"})), "betas.*glms"),
        (dict(stages=frozenset({"glms", "summaries"})), "summaries.*betas"),
        (dict(stages=frozenset({"plots"})), "stages"),
        (dict(n_jobs=0), "n_jobs"),
        (dict(block_size=0), "block_size"),
        (dict(max_grayordinates=0), "max_grayordinates"),
        (dict(existing_results="reuse"), "existing_results"),
        (dict(hrf_n_samples=12), "power of two"),
        (dict(subject="07"), "subject"),
        (dict(session="nsd10"), "session"),
        (dict(task="nsd core"), "task"),
    ],
)
def test_invalid_settings_name_the_field(bids, kwargs, message):
    with pytest.raises(ValueError, match=message):
        required(bids, **kwargs)


def test_missing_bids_dir_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="bids_dir"):
        WorkflowSettings(bids_dir=tmp_path / "nope", subject="sub-01", session="ses-01", task="t")


def test_parse_modulator_syntax():
    assert parse_modulator("response_time:indicator") == Modulator("response_time", missing="indicator")
    assert parse_modulator("trial_type") == Modulator("trial_type")
    for bad in ("", "rt:centered", "rt:indicator:extra", ":indicator"):
        with pytest.raises(ValueError, match="modulator"):
            parse_modulator(bad)


def test_round_trip_through_dict_and_library_builder(bids):
    settings = required(bids, modulators=(Modulator("response_time", missing="indicator"),), hrf_n_samples=8, stages=frozenset({"glms"}))
    values = settings.to_dict()
    assert values["modulators"] == [{"column": "response_time", "missing": "indicator"}]
    assert values["stages"] == ["glms"]
    assert WorkflowSettings.from_dict(values) == settings
    assert settings.build_library().fingerprint == default_hrf_library(8, seed=0).fingerprint
    assert len(required(bids, hrf_library="canonical").build_library().candidates) == 1
```

- [ ] **Step 2: Run to verify RED**

Run: `uv run pytest -q tests/workflow/test_settings.py`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Commit tests**

```bash
git add tests/workflow/test_settings.py
git commit -m "test: WorkflowSettings defaults, output name, validation, modulator syntax, round trip"
```

- [ ] **Step 4: Implement `settings.py`**

```python
"""Resolved, validated settings for one workflow run; the CLI, stages, writer and metadata share it."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

from boldtailor._scalars import is_integer, is_real
from boldtailor.hrf_library import (
    HrfLibrary,
    default_hrf_library,
    expanded_hrf_library,
    sobol_hrf_library,
)
from boldtailor.model import Modulator

SUPPORTED_SPACES = ("fsLR-91k",)
SPACE_ENTITIES = {"fsLR-91k": "space-fsLR_den-91k"}
RIDGE_MODES = ("fractional_cv", "cv", "fixed", "off")
HRF_LIBRARIES = ("default", "sobol", "expanded", "canonical")
STAGES = ("glms", "reliability", "betas", "summaries")
STAGE_REQUIRES = {"betas": "glms", "summaries": "betas"}
EXISTING_RESULTS = ("error", "overwrite")
_LABEL = re.compile(r"[A-Za-z0-9]+")


def resolve_fmriprep_dir(bids_dir):
    candidates = sorted(p for p in (Path(bids_dir) / "derivatives").glob("fmriprep*") if p.is_dir())
    if not candidates:
        raise ValueError(f"no derivatives/fmriprep* directory under {bids_dir}; pass fmriprep_dir")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(f"several fMRIPrep directories under {bids_dir}/derivatives ({names}); pass fmriprep_dir")
    return candidates[0]


def parse_modulator(text):
    parts = str(text).split(":")
    if not parts[0] or len(parts) > 2 or (len(parts) == 2 and parts[1] != "indicator"):
        raise ValueError(f"modulator must be COLUMN or COLUMN:indicator, not {text!r}")
    return Modulator(parts[0], missing="indicator" if len(parts) == 2 else "error")


def _check_label(value, name, prefix):
    text = str(value)
    body = text.removeprefix(prefix + "-") if prefix else text
    if (prefix and not text.startswith(prefix + "-")) or not _LABEL.fullmatch(body):
        wanted = f"{prefix}-<alphanumeric>" if prefix else "alphanumeric"
        raise ValueError(f"{name} must be {wanted}, not {text!r}")
    return text


def _check_choice(value, name, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}, not {value!r}")
    return value


def _check_positive_int(value, name, *, optional=False):
    if optional and value is None:
        return None
    if not is_integer(value) or value < 1:
        raise ValueError(f"{name} must be a positive integer, not {value!r}")
    return int(value)


@dataclass(frozen=True, kw_only=True)
class WorkflowSettings:
    bids_dir: Path
    subject: str
    session: str
    task: str
    fmriprep_dir: Path | None = None
    output_dir: Path | None = None
    space: str = "fsLR-91k"
    modulators: tuple[Modulator, ...] | None = None
    hrf_library: str = "default"
    hrf_n_samples: int = 512
    hrf_seed: int = 0
    hrf_selection_rt: bool = True
    ridge_mode: str = "fractional_cv"
    ridge_fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    ridge_alphas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    ridge_percentile: float = 90.0
    ridge_alpha: float = 0.1
    encoding_mode: str = "within_run"
    stages: frozenset[str] = field(default_factory=lambda: frozenset(STAGES))
    surface_maps: bool = True
    surface_meshes: Mapping[str, Path] | None = None
    n_jobs: int = 4
    block_size: int = 4096
    max_grayordinates: int | None = None
    existing_results: str = "error"

    def __post_init__(self):
        self._validate_paths()
        self._validate_labels()
        self._validate_model()
        self._validate_execution()

    def _set(self, name, value):
        object.__setattr__(self, name, value)

    def _validate_paths(self):
        root = Path(self.bids_dir).expanduser()
        if not root.is_dir():
            raise ValueError(f"bids_dir {root} is not a directory")
        self._set("bids_dir", root)
        prep = self.fmriprep_dir
        self._set("fmriprep_dir", Path(prep).expanduser() if prep is not None else resolve_fmriprep_dir(root))
        self._set("space", _check_choice(self.space, "space", SUPPORTED_SPACES))

    def _validate_labels(self):
        self._set("subject", _check_label(self.subject, "subject", "sub"))
        self._set("session", _check_label(self.session, "session", "ses"))
        self._set("task", _check_label(self.task, "task", ""))

    def _validate_model(self):
        if self.modulators is not None:
            mods = tuple(self.modulators)
            if any(not isinstance(m, Modulator) for m in mods):
                raise ValueError("modulators must be Modulator instances or None")
            self._set("modulators", mods)
        _check_choice(self.hrf_library, "hrf_library", HRF_LIBRARIES)
        n = _check_positive_int(self.hrf_n_samples, "hrf_n_samples")
        if n & (n - 1):
            raise ValueError("hrf_n_samples must be a power of two")
        if not is_integer(self.hrf_seed) or self.hrf_seed < 0:
            raise ValueError("hrf_seed must be a nonnegative integer")
        if not isinstance(self.hrf_selection_rt, bool):
            raise ValueError("hrf_selection_rt must be a boolean")
        _check_choice(self.ridge_mode, "ridge_mode", RIDGE_MODES)
        self._set("ridge_fractions", tuple(float(f) for f in self.ridge_fractions))
        self._set("ridge_alphas", tuple(float(a) for a in self.ridge_alphas))
        if not is_real(self.ridge_percentile) or not 0 <= self.ridge_percentile <= 100:
            raise ValueError("ridge_percentile must lie in [0, 100]")
        if self.ridge_mode == "fixed" and (not is_real(self.ridge_alpha) or self.ridge_alpha <= 0):
            raise ValueError("ridge_alpha must be positive in fixed mode")
        _check_choice(self.encoding_mode, "encoding_mode", ("within_run", "absolute"))

    def _validate_execution(self):
        stages = frozenset(self.stages)
        unknown = stages - set(STAGES)
        if unknown:
            raise ValueError(f"stages must be drawn from {', '.join(STAGES)}; unknown: {sorted(unknown)}")
        for stage, needed in STAGE_REQUIRES.items():
            if stage in stages and needed not in stages:
                raise ValueError(f"stage {stage} requires stage {needed}")
        self._set("stages", stages)
        if not isinstance(self.surface_maps, bool):
            raise ValueError("surface_maps must be a boolean")
        if self.surface_meshes is not None:
            meshes = {k: Path(v).expanduser() for k, v in dict(self.surface_meshes).items()}
            if set(meshes) != {"left", "right"}:
                raise ValueError("surface_meshes must map exactly 'left' and 'right'")
            self._set("surface_meshes", meshes)
        self._set("n_jobs", _check_positive_int(self.n_jobs, "n_jobs"))
        self._set("block_size", _check_positive_int(self.block_size, "block_size"))
        self._set("max_grayordinates", _check_positive_int(self.max_grayordinates, "max_grayordinates", optional=True))
        _check_choice(self.existing_results, "existing_results", EXISTING_RESULTS)
        out = self.output_dir
        self._set("output_dir", Path(out).expanduser() if out is not None else self.bids_dir / "derivatives" / self.output_name())

    def output_name(self):
        library = self.hrf_library
        if library in ("default", "sobol"):
            library = f"{library}{self.hrf_n_samples}s{self.hrf_seed}"
        ridge = self.ridge_mode.replace("_", "")
        if self.ridge_mode == "fixed":
            ridge = f"fixed{self.ridge_alpha:g}"
        return f"boldtailor_hrf-{library}_ridge-{ridge}"

    @property
    def stem(self):
        return f"{self.subject}/{self.session}/func/{self.subject}_{self.session}_task-{self.task}"

    @property
    def space_entity(self):
        return SPACE_ENTITIES[self.space]

    def build_library(self):
        if self.hrf_library == "default":
            return default_hrf_library(self.hrf_n_samples, seed=self.hrf_seed)
        if self.hrf_library == "sobol":
            return sobol_hrf_library(self.hrf_n_samples, seed=self.hrf_seed)
        if self.hrf_library == "expanded":
            return expanded_hrf_library()
        return HrfLibrary.from_parameters([], origin={"kind": "canonical_only", "n_candidates": 0})

    def to_dict(self):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Path):
                value = str(value)
            elif item.name == "modulators" and value is not None:
                value = [m.to_dict() for m in value]
            elif item.name == "stages":
                value = [s for s in STAGES if s in value]
            elif item.name == "surface_meshes" and value is not None:
                value = {k: str(v) for k, v in value.items()}
            elif isinstance(value, tuple):
                value = list(value)
            values[item.name] = value
        return values

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if values.get("modulators") is not None:
            values["modulators"] = tuple(Modulator(**m) for m in values["modulators"])
        if "stages" in values:
            values["stages"] = frozenset(values["stages"])
        for name in ("ridge_fractions", "ridge_alphas"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)
```

- [ ] **Step 5: GREEN**

Run: `uv run black -q src tests && uv run pytest -q -W error tests/workflow/test_settings.py && uv run pytest -q -W error`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/boldtailor/workflow/settings.py
git commit -m "feat: WorkflowSettings with derived fMRIPrep and output paths, output_name, validation, round trip"
```

---

### Task 4: Synthetic BIDS fixtures and `files.py`

**Files:**
- Create: `tests/workflow/synthetic_bids.py` (builders moved from `examples/NSD/conftest.py:58-250`: `_events`-style event table, confounds table, dataset writer, four-run and six-run variants)
- Create: `tests/workflow/conftest.py`
- Modify: `examples/NSD/conftest.py` (import the builders from `tests.workflow.synthetic_bids`; keep the notebook marker hooks and the `saved_sessions`, `hrf_nsd`, `mini_nsd` fixtures until Task 14 retires them)
- Create: `src/boldtailor/workflow/files.py` (from `examples/NSD/workflow_files.py`)
- Move test: `examples/NSD/test_workflow_files.py` → `tests/workflow/test_files.py`; the `reaction_times` tests from `examples/NSD/test_workflow_artifacts.py` join it.

**Interfaces:**
- Produces (all in `files.py`): `MOTION`, `bids_label(value, entity)`, `RunInputs`, `RawRun`, `input_paths(run)`, `discover_runs(settings) -> tuple[RunInputs, ...]`, `select_confounds(table, metadata)`, `load_inputs(inputs)`, `load_runs(inputs) -> (tuple[RawRun], brain)`, `source_ref`, `run_sources(run, root, indices)`, `odd_even_parity(runs)`, `reaction_times(runs, *, missing_ok=False)`.
- Fixtures in `tests/workflow/conftest.py`: `confounds`, `events`, `dataset` (returns `(root, prep, signal_runs, full, nuisance, brain)` exactly as today), `four_runs`, `six_run_dataset`, `small_library`, `cv_library`, and `settings_for(root, **overrides)` helper fixture returning a function that builds `WorkflowSettings(bids_dir=root, subject="sub-07", session="ses-nsd10", task="nsdcore", n_jobs=1, block_size=2, surface_maps=False, **overrides)`.

- [ ] **Step 1: Move the builders**

Create `tests/workflow/synthetic_bids.py` with module-level functions `make_confounds()`, `make_events()`, `write_dataset(tmp_path, confounds, events)`, `add_runs_three_and_four(root, prep)`, `make_six_runs(root, prep)`, each copied verbatim from the body of the corresponding fixture in `examples/NSD/conftest.py` (`confounds`, `events`, `dataset`, `four_runs`, `six_run_dataset`), replacing `examples.NSD.workflow_artifacts` imports with `boldtailor.cifti.scalar_artifact`/`boldtailor.publication` equivalents only where those fixtures use them (the dataset writer uses nibabel directly; keep it so). `tests/workflow/conftest.py` wraps them as fixtures with the same names and return values. Add:

```python
@pytest.fixture
def settings_for():
    from boldtailor.workflow.settings import WorkflowSettings

    def build(root, **overrides):
        options = dict(subject="sub-07", session="ses-nsd10", task="nsdcore", n_jobs=1, block_size=2, surface_maps=False)
        options.update(overrides)
        return WorkflowSettings(bids_dir=root, **options)

    return build
```

Edit `examples/NSD/conftest.py` so `confounds`, `events`, `dataset`, `four_runs`, `six_run_dataset` call the builders from `tests.workflow.synthetic_bids` (the repo root is already on `sys.path` there).

- [ ] **Step 2: Move and adapt the tests (RED)**

`git mv examples/NSD/test_workflow_files.py tests/workflow/test_files.py`. Change imports to `from boldtailor.workflow import files as workflow_files`. `test_discovery_requires_every_run_to_have_cifti` and the other `dataset` tests call `discover_runs(settings_for(root))` instead of `discover_runs(root, prep, subject=..., session=...)`. Append the two `reaction_times` tests from `examples/NSD/test_workflow_artifacts.py` (delete them there). Add:

```python
def test_discovery_uses_the_task_and_space_from_settings(dataset, settings_for):
    root, prep, *_ = dataset
    runs = workflow_files.discover_runs(settings_for(root))
    assert [r.stem for r in runs] == ["sub-07_ses-nsd10_task-nsdcore_run-01", "sub-07_ses-nsd10_task-nsdcore_run-02"]
    assert runs[0].bold.name.endswith("_space-fsLR_den-91k_bold.dtseries.nii")
    with pytest.raises(FileNotFoundError, match="task-other"):
        workflow_files.discover_runs(settings_for(root, task="other"))
```

Run: `uv run pytest -q tests/workflow/test_files.py` → FAIL (`boldtailor.workflow.files` missing).

Commit: `git add -A tests examples/NSD/conftest.py && git commit -m "test: workflow file discovery tests move into the package suite with shared synthetic BIDS builders"`.

- [ ] **Step 3: Port `files.py`**

`git mv examples/NSD/workflow_files.py src/boldtailor/workflow/files.py`. Edits:
- `_run_inputs(event, derivative, space_entity)`: build the BOLD name as `f"{stem}_{space_entity}_bold.dtseries.nii"`.
- `discover_runs(settings)`: `raw = settings.bids_dir / settings.subject / settings.session / "func"`, `derivative = settings.fmriprep_dir / ...`, glob `f"{settings.subject}_{settings.session}_task-{settings.task}_run-*_events.tsv"`; the no-events error message includes the glob pattern; the CIFTI glob uses `settings.space_entity`.
- Delete the `subject="sub-07", session="ses-nsd10"` defaults everywhere. Nothing else changes.

Run: `uv run black -q src tests && uv run pytest -q -W error tests/workflow/test_files.py && uv run pytest -q -W error` → PASS.

Commit: `git add -A src/boldtailor/workflow/files.py examples/NSD && git commit -m "feat: boldtailor.workflow.files discovers runs for any task and supported space"`.

---

### Task 5: `inputs.py` with modulator detection

**Files:**
- Create: `src/boldtailor/workflow/inputs.py` (from `examples/NSD/workflow_inputs.py`)
- Test: `tests/workflow/test_inputs.py` (new, plus the input-related tests from `examples/NSD/test_nsd_workflow.py`: `test_invalid_glm_covariates_fail_explicitly`, `test_nonpositive_rt_becomes_missing_with_indicator`, `test_glm_requires_observed_rt_to_estimate_rt_effect`, `test_glm_does_not_treat_malformed_rt_text_as_missing`, `test_trimming_keeps_acquisition_times_and_matches_confounds`, `test_interior_nonsteady_flag_is_rejected`, `test_selection_task_model_switch_drops_only_rt`)

**Interfaces:**
- Produces:

```python
class InputError(ValueError)      # missing or malformed inputs; CLI exit code 2
def detect_task_model(events_tables, modulators=None) -> TaskModel   # raises InputError for absent columns
    # modulators None: response_time present in every run -> Modulator("response_time", missing="indicator");
    # trial_type present in every run -> Modulator("trial_type"); else task only.
    # explicit modulators: every column must exist in every run, else ValueError naming the run index and column.
def selection_task_model(task_model, include_rt=True) -> TaskModel   # drops response_time when include_rt is False
@dataclass(frozen=True) class WorkflowRun: inputs, image, events, confounds, frame_times, label, number, retained_frames
def load_session(settings, *, hrf_only=False) -> tuple[WorkflowRun, ...]
def glm_model(runs, task_model) -> ModelSpec
def load_block(runs, root, indices, task_model) -> AnalysisData
def make_blocks(runs, *, block_size=4096, max_grayordinates=None)
def run_summary(runs, task_model) -> pd.DataFrame
def block_signals(runs, indices); def _trimmed_sources(run, root, indices)
```

- [ ] **Step 1: Write the new tests (RED)** in `tests/workflow/test_inputs.py`

```python
"""Session loading, modulator detection, and the GLM model for any task."""

import numpy as np
import pandas as pd
import pytest

from boldtailor.model import Modulator, TaskModel
from boldtailor.workflow import inputs


def test_detection_uses_rt_and_trial_type_when_present(events):
    model = inputs.detect_task_model([events, events])
    assert model == TaskModel((Modulator("response_time", missing="indicator"), Modulator("trial_type")))
    without_rt = inputs.detect_task_model([events.drop(columns="response_time"), events])
    assert without_rt == TaskModel((Modulator("trial_type"),))
    assert inputs.detect_task_model([events[["onset", "duration"]]]) == TaskModel()


def test_explicit_modulators_must_exist_in_every_run(events):
    wanted = (Modulator("stimulus_id"),)
    with pytest.raises(ValueError, match="run 2.*stimulus_id"):
        inputs.detect_task_model([events.assign(stimulus_id=1), events], wanted)
    assert inputs.detect_task_model([events.assign(stimulus_id=1)] * 2, wanted) == TaskModel(wanted)


def test_load_session_detects_the_model_and_builds_the_glm(four_runs, settings_for):
    root, _ = four_runs
    runs = inputs.load_session(settings_for(root))
    model = inputs.detect_task_model([r.events for r in runs])
    glm = inputs.glm_model(runs, model)
    assert glm.task_model == model and glm.noise_model == "ols" and glm.drift_model is None
    assert set(glm.contrasts) == {"task", "response_time", "trial_type"}
    assert list(inputs.run_summary(runs, model).columns)[:5] == ["run", "trials", "scans", "retained_scans", "dropped_scans"]


def test_task_only_session_has_no_rt_columns(four_runs, settings_for):
    root, _ = four_runs
    for path in (root / "sub-07" / "ses-nsd10" / "func").glob("*_events.tsv"):
        pd.read_csv(path, sep="\t").drop(columns=["response_time", "trial_type"]).to_csv(path, sep="\t", index=False)
    runs = inputs.load_session(settings_for(root))
    model = inputs.detect_task_model([r.events for r in runs])
    assert model == TaskModel()
    assert set(inputs.glm_model(runs, model).contrasts) == {"task"}
    summary = inputs.run_summary(runs, model)
    assert "mean_rt_seconds" not in summary.columns


def test_hrf_only_loading_needs_two_runs_not_two_per_parity(dataset, settings_for):
    root, *_ = dataset
    assert len(inputs.load_session(settings_for(root), hrf_only=True)) == 2
    with pytest.raises(ValueError, match="two odd and two even"):
        inputs.load_session(settings_for(root))
```

Then `git mv`-free: copy the seven input tests listed above from `examples/NSD/test_nsd_workflow.py` into this file (delete them there), replacing `examples.NSD.workflow_inputs` with `boldtailor.workflow.inputs`, `NSD_TASK_MODEL` with `inputs.detect_task_model([events])`, `load_session(root, prep, ...)` with `inputs.load_session(settings_for(root))`, `selection_task_model(include_rt)` with `inputs.selection_task_model(model, include_rt)`, and any `center=` with nothing. `test_selection_task_model_switch_drops_only_rt` expects `TaskModel((Modulator("trial_type"),))`.

Run: `uv run pytest -q tests/workflow/test_inputs.py` → FAIL. Commit: `git add -A tests examples/NSD/test_nsd_workflow.py && git commit -m "test: workflow inputs detect modulators from events and load sessions from settings"`.

- [ ] **Step 2: Port `inputs.py`**

`git mv examples/NSD/workflow_inputs.py src/boldtailor/workflow/inputs.py`; imports become `from boldtailor.workflow.files import discover_runs, load_runs, odd_even_parity, run_sources`. Edits:
- Delete `REGRESSORS`, `NSD_TASK_MODEL`. Add `detect_task_model` and the new `selection_task_model(task_model, include_rt=True)`:

```python
class InputError(ValueError):
    """Inputs are missing or malformed; the CLI maps this to exit code 2."""


def detect_task_model(events_tables, modulators=None):
    tables = list(events_tables)
    if modulators is None:
        modulators = tuple(
            Modulator(column, missing=missing)
            for column, missing in (("response_time", "indicator"), ("trial_type", "error"))
            if all(column in t.columns for t in tables)
        )
    for index, table in enumerate(tables, 1):
        missing = [m.column for m in modulators if m.column not in table.columns]
        if missing:
            raise InputError(f"run {index} events lack modulator column(s) {missing}")
    return TaskModel(tuple(modulators))


def selection_task_model(task_model, include_rt=True):
    if not isinstance(include_rt, bool):
        raise ValueError("include_rt must be a boolean")
    if include_rt:
        return task_model
    return TaskModel(tuple(m for m in task_model.modulators if m.column != "response_time"))
```
- `validate_glm_events(events)` → `validate_glm_events(events, task_model)`: the RT check runs only when `response_time` is a modulator column, the trial_type check only when `trial_type` is; `_missing_nonpositive_rt` applies only when the column exists. `_trim(run, task_model)` passes it through; `load_session(settings, *, hrf_only=False)` discovers with `discover_runs(settings)`, computes `task_model = detect_task_model([r.events for r in raw_runs], settings.modulators)` before trimming, and keeps the two-runs and parity checks (message "The full workflow needs at least two odd and two even runs").
- `glm_model(runs, task_model)`: contrasts `{name: {name: 1} for name in task_model.regressor_names}`.
- `load_block(runs, root, indices, task_model)`: provenance metadata `task_model=task_model.to_dict()`.
- `run_summary(runs, task_model)`: include `mean_rt_seconds` only when `response_time` is a modulator and `type_0`/`type_1` only when `trial_type` is.

Run: `uv run black -q src tests && uv run pytest -q -W error tests/workflow && uv run pytest -q -W error` → PASS.

Commit: `git add -A src/boldtailor/workflow/inputs.py examples/NSD && git commit -m "feat: boldtailor.workflow.inputs loads a session from settings and detects the task model"`.

---

### Task 6: `analysis.py`

**Files:**
- Create: `src/boldtailor/workflow/analysis.py` (from `examples/NSD/workflow_analysis.py` plus `execution_settings` from `parallel_blocks.py` is **not** needed; delete `parallel_blocks.py` in Task 14)
- Test: `tests/workflow/test_analysis.py` (from `examples/NSD/test_nsd_workflow.py`: `test_both_glms_match_independent_ols_and_keep_spatial_order`, `test_glm_model_and_selection_share_the_nsd_task_model` renamed `..._share_the_detected_task_model`, `test_select_hrfs_without_rt_still_feeds_the_full_glm`, and the `independent_ols` helper)

**Interfaces:**
- Produces: `select_hrfs(runs, root, blocks, library, *, task_model, n_jobs=1)`, `selection_maps(selections, n_features)`, `fit_glms(runs, root, blocks, model, *, selections=None, n_jobs=1)` (regressor names from `model.task_model.regressor_names`), `fit_beta_series(runs, root, blocks, *, task_model, selections=None, ridge_alpha=0.0, ridge_fraction=None, n_jobs=1)` whose result has `rt=None` when `response_time` is not a modulator column.

- [ ] **Step 1: Move tests (RED)**: copy the three tests and `independent_ols` into `tests/workflow/test_analysis.py`; imports `from boldtailor.workflow import analysis, inputs`; replace `load_session(root, prep, ...)` by `inputs.load_session(settings_for(root))`, `NSD_TASK_MODEL` by `model = inputs.detect_task_model([r.events for r in runs])`, `glm_model(runs)` by `inputs.glm_model(runs, model)`, `select_hrfs(..., task_model=selection_task_model(...))` by `analysis.select_hrfs(..., task_model=inputs.selection_task_model(model, include_rt))`, `fit_beta_series(runs, root, blocks, ...)` adds `task_model=model`. The `independent_ols` oracle uses raw RT with NaN → 0 plus the indicator (no mean subtraction). Add:

```python
def test_beta_series_without_response_time_has_no_rt_correlations(four_runs, settings_for):
    import pandas as pd
    root, _ = four_runs
    for path in (root / "sub-07" / "ses-nsd10" / "func").glob("*_events.tsv"):
        pd.read_csv(path, sep="\t").drop(columns="response_time").to_csv(path, sep="\t", index=False)
    runs = inputs.load_session(settings_for(root))
    model = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=2)
    result = analysis.fit_beta_series(runs, root, blocks, task_model=model)
    assert result["rt"] is None and len(result["betas"]) == 4
```

Run → FAIL. Commit: `git commit -am "test: workflow analysis runs on the detected task model"` (after `git add -A tests examples/NSD/test_nsd_workflow.py`).

- [ ] **Step 2: Port** `git mv examples/NSD/workflow_analysis.py src/boldtailor/workflow/analysis.py`. Imports from `boldtailor.workflow.files`/`inputs`. Remove `NSD_TASK_MODEL`/`REGRESSORS`: `_select_block(indices, runs, root, library, task_model)` unchanged except no default; `_glm_block` reads `names = model.task_model.regressor_names` and allocates `(len(names), k)`; `fit_glms` allocates maps with `len(model.task_model.regressor_names)` rows; `_beta_block(job, runs, root, alpha, task_model, fractions=None)` passes `task_model` to `load_block`; `fit_beta_series` computes `rt = correlate_rt(...) if "response_time" in task_model.regressor_names else None`. Drop progress `print`s in favour of `logging.getLogger("boldtailor.workflow")` at INFO.

Run black and both suites → PASS. Commit: `git add -A src/boldtailor/workflow/analysis.py examples/NSD && git commit -m "feat: boldtailor.workflow.analysis fits GLMs, selections and beta series for any task model"`.

---

### Task 7: `beta_series.py` with `BetaModel`

**Files:**
- Create: `src/boldtailor/workflow/beta_series.py` (from `examples/NSD/ridge_workflow.py` and `ridge_provenance.py`)
- Test: `tests/workflow/test_beta_series.py` (from `examples/NSD/test_ridge_workflow.py`, `test_fractional_workflow.py` (the `fit_cv_beta_series` tests; export assertions move to Task 8), `test_within_run_encoding.py`, `test_ridge_provenance.py`)

**Interfaces:**
- Produces:

```python
def trial_predictors(runs, task_model) -> list[pd.DataFrame]   # modulator columns, numeric; response_time <= 0 -> NaN
def fit_cv_beta_series(runs, root, *, task_model, library, alphas=None, fractions=None, percentile=90.0,
                       block_size=4096, max_grayordinates=None, n_jobs=1, encoding_mode="within_run") -> dict
def tuning_provenance(scores, selection); def link_final_provenance(result, decision)

@dataclass(frozen=True, kw_only=True)
class BetaModel:
    name: str              # "CanonicalTrialOLS", "OptimizedTrialFractionalCV", ...
    hrf: str               # "canonical" | "optimized"
    estimator: str         # "OLS" | "Ridge" | "RidgeCV" | "FractionalCV"
    fit: Mapping[str, object]          # fit_beta_series result (betas, trial_table, r2, designs, provenance, rt, ...)
    tuning: Mapping[str, object] | None = None      # {"odd","even","all"} tuned dicts
    evaluation: Mapping[str, object] | None = None  # {"odd_to_even","even_to_odd"}
    cv_provenance: Mapping[str, object] | None = None
    predictors: tuple | None = None
    @property fractional(self) -> bool    # "ridge_fraction" in fit

def fit_beta_models(runs, root, blocks, settings, library, selections, task_model) -> dict[str, BetaModel]
    # off: Canonical/OptimizedTrialOLS; fixed adds ...TrialRidge (settings.ridge_alpha);
    # cv adds ...TrialRidgeCV; fractional_cv adds ...TrialFractionalCV, each from fit_cv_beta_series(...)["final"]
    # with tuning/evaluation/cv_provenance/predictors filled from the CV result.
```

- [ ] **Step 1: Move and extend tests (RED)**. Combine the listed example tests into `tests/workflow/test_beta_series.py` with imports `from boldtailor.workflow import beta_series, inputs`; sessions load via `inputs.load_session(settings_for(root))`; `task_model=model` is passed where `NSD_TASK_MODEL` was; `trial_predictors(runs)` becomes `trial_predictors(runs, model)`. Add:

```python
@pytest.mark.parametrize("mode", ["off", "fixed", "cv", "fractional_cv"])
def test_fit_beta_models_yields_one_model_per_hrf_and_estimator(six_run_dataset, cv_library, settings_for, mode):
    root, _ = six_run_dataset
    settings = settings_for(root, ridge_mode=mode, ridge_fractions=(0.4, 1.0), ridge_alphas=(0.0, 0.1), block_size=4)
    runs = inputs.load_session(settings)
    model = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=4)
    from boldtailor.workflow import analysis
    selections = analysis.select_hrfs(runs, root, blocks, cv_library, task_model=model)
    models = beta_series.fit_beta_models(runs, root, blocks, settings, cv_library, selections, model)
    estimator = {"off": None, "fixed": "Ridge", "cv": "RidgeCV", "fractional_cv": "FractionalCV"}[mode]
    expected = {"CanonicalTrialOLS", "OptimizedTrialOLS"} | ({f"CanonicalTrial{estimator}", f"OptimizedTrial{estimator}"} if estimator else set())
    assert set(models) == expected
    for name, item in models.items():
        assert item.name == name and item.hrf in ("canonical", "optimized")
        assert len(item.fit["betas"]) == 6
        tuned = item.estimator in ("RidgeCV", "FractionalCV")
        assert (item.tuning is not None) is tuned and (item.evaluation is not None) is tuned
        assert item.fractional == (item.estimator == "FractionalCV")
```

Run → FAIL. Commit tests.

- [ ] **Step 2: Port**. `git mv examples/NSD/ridge_workflow.py src/boldtailor/workflow/beta_series.py`; append the three functions of `ridge_provenance.py` (then `git rm examples/NSD/ridge_provenance.py`). Remove every `task_model=NSD_TASK_MODEL` default (make it required keyword); `trial_predictors(runs, task_model)` selects `[m.column for m in task_model.modulators]`, coerces numerically, applies the nonpositive→NaN rule to `response_time` only and the {0,1} check to `trial_type` only, and raises when a column is missing. `_outer_block` names coefficient rows `("task", *predictor columns)`. Add `BetaModel` and `fit_beta_models`:

```python
_ESTIMATOR = {"fixed": "Ridge", "cv": "RidgeCV", "fractional_cv": "FractionalCV"}


def fit_beta_models(runs, root, blocks, settings, library, selections, task_model):
    models = {}
    for hrf, selected, candidates in (("canonical", None, None), ("optimized", selections, library)):
        prefix = hrf.capitalize() + "Trial"
        models[prefix + "OLS"] = BetaModel(name=prefix + "OLS", hrf=hrf, estimator="OLS",
            fit=fit_beta_series(runs, root, blocks, task_model=task_model, selections=selected, n_jobs=settings.n_jobs))
        if settings.ridge_mode == "fixed":
            models[prefix + "Ridge"] = BetaModel(name=prefix + "Ridge", hrf=hrf, estimator="Ridge",
                fit=fit_beta_series(runs, root, blocks, task_model=task_model, selections=selected,
                                    ridge_alpha=settings.ridge_alpha, n_jobs=settings.n_jobs))
        if settings.ridge_mode in ("cv", "fractional_cv"):
            models[prefix + _ESTIMATOR[settings.ridge_mode]] = _cv_model(runs, root, settings, candidates, task_model, hrf)
    return models


def _cv_model(runs, root, settings, library, task_model, hrf):
    grid = dict(fractions=settings.ridge_fractions) if settings.ridge_mode == "fractional_cv" else dict(alphas=settings.ridge_alphas)
    result = fit_cv_beta_series(runs, root, task_model=inputs.selection_task_model(task_model, settings.hrf_selection_rt),
        library=library, **grid, encoding_mode=settings.encoding_mode, percentile=settings.ridge_percentile,
        block_size=settings.block_size, max_grayordinates=settings.max_grayordinates, n_jobs=settings.n_jobs)
    estimator = _ESTIMATOR[settings.ridge_mode]
    return BetaModel(name=f"{hrf.capitalize()}Trial{estimator}", hrf=hrf, estimator=estimator, fit=result["final"],
        tuning=result["tuning"], evaluation=result["evaluation"], cv_provenance=result["provenance"],
        predictors=tuple(result["predictors"]))
```

(`fit_beta_series` is imported from `boldtailor.workflow.analysis`; `inputs` from `boldtailor.workflow`.) Note the final fit in `fit_cv_beta_series` receives `task_model` for `fit_beta_series` and the selection model for `select_hrfs`; pass both where the existing code used one.

Run black and both suites → PASS. Commit: `git add -A src/boldtailor/workflow/beta_series.py examples/NSD && git commit -m "feat: boldtailor.workflow.beta_series with BetaModel for every ridge mode"`.

---

### Task 8: `artifacts.py` and `outputs.py` (one writer for every stage)

**Files:**
- Create: `src/boldtailor/workflow/artifacts.py` (from `examples/NSD/workflow_artifacts.py`)
- Create: `src/boldtailor/workflow/outputs.py` (from `workflow_outputs.py`, `ridge_outputs.py`, `fractional_outputs.py`)
- Test: `tests/workflow/test_outputs.py` (from `examples/NSD/test_beta_activation.py`, `test_ridge_outputs.py`, the export half of `test_fractional_workflow.py`, `test_workflow_artifacts.py::test_dataset_description_is_a_boldtailor_derivative`, and from `test_workflow_reuse.py`: `test_metadata_records_the_task_model_fingerprint`, `test_metadata_records_peak_hrf_normalization`, `test_metadata_pools_hrf_bound_flags_over_blocks`)

**Interfaces:**
- Produces in `artifacts.py`: `json_artifact`, `table_artifact`, `npz_artifact`, `figure_artifact`, `dataset_description(name)`, `parameter_artifact(brain, library, ids, path)`, `scalar_map(stem, space_entity, brain, descriptor, statistic, values, names)` → path `f"{stem}_{space_entity}_desc-{descriptor}_stat-{statistic}.dscalar.nii"`.
- Produces in `outputs.py`: `R2_NAMES`, `check_output(settings)` (raises `FileExistsError` for `error` when `settings.output_dir/<sub>/<ses>/func` has any `<sub>_<ses>_task-<task>*` file; returns None), `glm_artifacts(settings, brain, glms, runs)`, `hrf_artifacts(settings, brain, selections, library)`, `beta_model_artifacts(settings, brain, runs, model: BetaModel, library)` (fit artifacts + fraction maps when fractional + tuning and outer artifacts when tuned + per-run betas + rtcorrelation map when `fit["rt"]` is not None), `activation_artifacts(settings, brain, activation)`, `input_artifacts(settings, runs, task_model)`, `metadata(runs, library, settings, task_model, *, beta_models, activation, selections, skipped, report)`, `tuning_table(beta_models)`, `tuning_figure(beta_models)`, `hrf_boundary_summary(selections)`, `ridge_boundary_summary(beta_models)`, `save_workflow(settings, runs, task_model, library, selections, glms, beta_models, *, figures, activation, skipped, report_html) -> tuple[Path, ...]`.
- Metadata JSON path: `f"{stem}_desc-boldtailor_metadata.json"`; figures `f"{stem}_desc-{key}_plot.png"`; runs table `f"{stem}_desc-boldtailor_runs.tsv"`; per-run events/confounds `..._desc-boldtailor_events.tsv` / `_confounds.tsv`; report at `f"{subject}_{session}_task-{task}_report.html"` relative to the output root.

- [ ] **Step 1: Move and adapt tests (RED)**. Combine the listed tests into `tests/workflow/test_outputs.py`. Replace `_beta_artifacts(stem, brain, betas, runs)`/`ridge_artifacts(...)`/`fraction_fit_artifacts` calls with `outputs.beta_model_artifacts(settings, brain, runs, model, library)` over the `BetaModel`s from `beta_series.fit_beta_models`; replace `_activation_artifacts("sub-07", brain, {...})` with `outputs.activation_artifacts(settings, brain, {...})`; replace `_metadata(runs, library, settings_dict, ...)` with `outputs.metadata(runs, library, settings, model, beta_models={}, activation=None, selections=..., skipped=(), report=None)`. Every expected filename drops `notebook` from `desc-notebook`. Add:

```python
def test_beta_model_writer_covers_every_estimator(six_run_dataset, cv_library, settings_for):
    root, _ = six_run_dataset
    settings = settings_for(root, ridge_mode="fractional_cv", ridge_fractions=(0.4, 1.0), block_size=4)
    runs = inputs.load_session(settings); model = inputs.detect_task_model([r.events for r in runs])
    blocks = inputs.make_blocks(runs, block_size=4); brain = runs[0].image.header.get_axis(1)
    selections = analysis.select_hrfs(runs, root, blocks, cv_library, task_model=model)
    models = beta_series.fit_beta_models(runs, root, blocks, settings, cv_library, selections, model)
    names = {a.path for m in models.values() for a in outputs.beta_model_artifacts(settings, brain, runs, m, cv_library)}
    stem = settings.stem
    assert f"{stem}_space-fsLR_den-91k_desc-OptimizedTrialFractionalCV_stat-ridgefraction.dscalar.nii" in names
    assert f"{stem}_space-fsLR_den-91k_desc-OptimizedFractionalCVAll_stat-encodingcvr2.dscalar.nii" in names
    assert f"{stem}_space-fsLR_den-91k_desc-CanonicalTrialOLS_stat-rtcorrelation.dscalar.nii" in names
    assert not any("notebook" in n for n in names)


def test_check_output_errors_only_when_files_exist(four_runs, settings_for, tmp_path):
    root, _ = four_runs
    settings = settings_for(root, output_dir=tmp_path / "out")
    outputs.check_output(settings)  # nothing there
    target = tmp_path / "out" / "sub-07" / "ses-nsd10" / "func"; target.mkdir(parents=True)
    (target / "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json").write_text("{}")
    with pytest.raises(FileExistsError, match="existing_results"):
        outputs.check_output(settings)
    outputs.check_output(settings_for(root, output_dir=tmp_path / "out", existing_results="overwrite"))
```

Run → FAIL. Commit tests.

- [ ] **Step 2: Port**. `git mv examples/NSD/workflow_artifacts.py src/boldtailor/workflow/artifacts.py`; rename `notebook_map` → `scalar_map(stem, space_entity, brain, descriptor, statistic, values, names)`. `git mv examples/NSD/workflow_outputs.py src/boldtailor/workflow/outputs.py` and merge in `ridge_outputs.py` and `fractional_outputs.py` (then `git rm` them). In `outputs.py`:
- Every function takes `settings` and derives `stem = settings.stem`, `space = settings.space_entity`; every `f"...desc-notebook{X}..."` becomes `f"...desc-{X}..."`.
- `check_output(settings)` implements the `error`/`overwrite` policy only (no reuse); `save_workflow` publishes with `overwrite=settings.existing_results == "overwrite"`.
- `beta_model_artifacts` = former `_beta_artifacts` for one descriptor (`model.name`) + `fraction_fit_artifacts` when `model.fractional` + (when `model.tuning`) the tuning artifacts per scope and outer artifacts per evaluation (former `ridge_artifacts` body with `mode = model.hrf.capitalize()`, `kind = "FractionalCV" if model.fractional else "RidgeCV"`), plus `_predictors.tsv` from `model.predictors`, and the `rtcorrelation` map only when `model.fit["rt"] is not None`.
- `metadata(...)` replaces `NSD_TASK_MODEL`/`REGRESSORS` with `task_model`; `settings=settings.to_dict()`; adds `runs=[r.label for r in runs]`, `task_model=task_model.to_dict()`, `skipped=list(skipped)`, `report=report`; the task-model description sentence from Task 2 is added here; `encoding_predictors = ["task", *[m.column for m in task_model.modulators]]`; `response_time`/`trial_type` prose keyed on their presence.
- `tuning_table(beta_models)` / `tuning_figure(beta_models)` iterate `[m for m in beta_models.values() if m.tuning]`.
- `save_workflow(settings, runs, task_model, library, selections, glms, beta_models, *, figures, activation, skipped, report_html)` appends `Artifact(f"{settings.subject}_{settings.session}_task-{settings.task}_report.html", report_html)` when `report_html` is not None and `dataset_description("boldtailor")` at `dataset_description.json`.

Run black and both suites → PASS. Commit: `git add -A src/boldtailor/workflow examples/NSD && git commit -m "feat: boldtailor.workflow.outputs writes every stage with one BetaModel writer and boldtailor descriptors"`.

---

### Task 9: `plots.py` and `surfaces.py`

**Files:**
- Create: `src/boldtailor/workflow/plots.py` (from `workflow_plots.py`), `src/boldtailor/workflow/surfaces.py` (from `workflow_surfaces.py`)
- Test: `tests/workflow/test_plots.py` (plot tests from `examples/NSD/test_notebook_helpers.py`), `tests/workflow/test_surfaces.py` (from `examples/NSD/test_workflow_surfaces.py`)

**Interfaces:**
- Produces: `design_figure(frame_times, design, regressors)`, `library_figure(library)`, `glm_comparison(glms)`, `parameter_agreement(library, odd_ids, even_ids)`, `curve_agreement(curve_r)`, `activation_histogram(activation)` (the notebook's histogram cell, returns a Figure), `rt_check_figure(beta_models, runs, brain)` (the notebook's RT scatter cell: even-run points at the best odd-run vertex for the two OLS models; returns `None` when no RT), `fraction_selection_figure(table)`; `find_surface_meshes(fmriprep_dir, subject, *, paths=None)`, `surface_figure(maps, brain, meshes, *, statistic, title=None)`.

- [ ] **Step 1: Move tests (RED)**: `git mv examples/NSD/test_workflow_surfaces.py tests/workflow/test_surfaces.py`; create `tests/workflow/test_plots.py` with the three plot tests from `test_notebook_helpers.py` (imports `from boldtailor.workflow import plots`), plus:

```python
def test_rt_check_figure_is_none_without_reaction_times(two_models_without_rt):
    assert plots.rt_check_figure(two_models_without_rt, runs=[], brain=None) is None
```
where `two_models_without_rt` is a local fixture building two `BetaModel`s with `fit={"rt": None, "betas": []}`. Run → FAIL; commit.

- [ ] **Step 2: Port** both modules with `git mv`; add `activation_histogram`, `rt_check_figure`, `fraction_selection_figure` by lifting the corresponding notebook cell bodies (cells 24, 28, 30 of `nsd_workflow.ipynb`) into functions that return the Figure. `_label` in `surfaces.py` keeps the estimator vocabulary. Black, suites → PASS. Commit: `feat: boldtailor.workflow.plots and surfaces`.

---

### Task 10: `report.py`

**Files:**
- Create: `src/boldtailor/workflow/report.py`
- Test: `tests/workflow/test_report.py`

**Interfaces:**
- Produces: `render_report(settings, *, runs, task_model, library, glm_summary, hrf_summary, reliability, tuning, activation_summary, rt_summary, figures, skipped, manifest) -> str` (HTML text). All table arguments are `pd.DataFrame | None`; `figures` is `dict[str, matplotlib.figure.Figure]`; `manifest` is a sequence of `(relative_path, provenance_relative_path | None)`; `skipped` is a sequence of `(stage, reason)`.
- Helper: `embed_figure(figure) -> str` (base64 PNG `<img>` tag).

- [ ] **Step 1: Write the failing test**

```python
# tests/workflow/test_report.py
"""The HTML report is self-contained and covers every stage."""

from html.parser import HTMLParser

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from boldtailor.model import TaskModel
from boldtailor.workflow import report


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids, self.images = [], 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section" and "id" in attrs: self.ids.append(attrs["id"])
        if tag == "img": assert attrs["src"].startswith("data:image/png;base64,"); self.images += 1


def test_report_has_a_section_per_stage_embedded_figures_and_a_manifest(bids_settings):
    fig, ax = plt.subplots(); ax.plot([0, 1])
    html = report.render_report(
        bids_settings, runs=[], task_model=TaskModel(), library=None,
        glm_summary=pd.DataFrame({"model": ["CanonicalGLM"], "median": [0.1]}),
        hrf_summary=None, reliability=None, tuning=None,
        activation_summary=None, rt_summary=None,
        figures={"Design": fig}, skipped=[("reliability", "only one odd run")],
        manifest=[("sub-07/ses-nsd10/func/a.dscalar.nii", "sub-07/ses-nsd10/func/a_provenance.json")],
    )
    parser = _Collector(); parser.feed(html)
    assert parser.ids == ["settings", "inputs", "glms", "reliability", "betas", "summaries", "skipped", "files"]
    assert parser.images == 1
    assert "only one odd run" in html and "a.dscalar.nii" in html and "CanonicalGLM" in html
    assert "<link" not in html and "<script src" not in html
```

`bids_settings` is a fixture in `tests/workflow/conftest.py`: `settings_for(root)` over a `dataset`.

- [ ] **Step 2: RED, commit, implement**

```python
"""Self-contained HTML report: one section per stage, figures embedded as PNG."""

import base64
import html
from io import BytesIO

SECTIONS = ("settings", "inputs", "glms", "reliability", "betas", "summaries", "skipped", "files")
_STYLE = "body{font:15px/1.5 system-ui;margin:2rem auto;max-width:72rem;padding:0 1rem}table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #ccc;padding:.25rem .6rem;text-align:left}img{max-width:100%}section{margin:2rem 0}h2{border-bottom:1px solid #ddd}.note{color:#555}"


def embed_figure(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="figure" src="data:image/png;base64,{data}">'


def _table(frame):
    if frame is None or len(frame) == 0:
        return '<p class="note">No table for this stage.</p>'
    return frame.to_html(index=False, float_format=lambda v: f"{v:.4g}", border=0)


def _section(name, title, body):
    return f'<section id="{name}"><h2>{html.escape(title)}</h2>{body}</section>'


def _settings_body(settings):
    rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in settings.to_dict().items())
    return f"<table>{rows}</table>"


def _inputs_body(runs, task_model, library):
    items = [f"<li>{len(runs)} runs: {html.escape(', '.join(r.label for r in runs))}</li>",
             f"<li>Task regressors: {html.escape(', '.join(task_model.regressor_names))}</li>"]
    if library is not None:
        items.append(f"<li>HRF library: {len(library.candidates)} candidates, fingerprint {library.fingerprint[:12]}</li>")
    return "<ul>" + "".join(items) + "</ul>"


def _figures(figures, keys):
    return "".join(f"<h3>{html.escape(k)}</h3>{embed_figure(figures[k])}" for k in keys if k in figures)


def _skipped_body(skipped):
    if not skipped:
        return '<p class="note">Every enabled stage ran.</p>'
    return "<ul>" + "".join(f"<li><b>{html.escape(s)}</b>: {html.escape(r)}</li>" for s, r in skipped) + "</ul>"


def _files_body(manifest):
    rows = []
    for path, provenance in manifest:
        link = f'<a href="{html.escape(provenance)}">provenance</a>' if provenance else ""
        rows.append(f'<tr><td><a href="{html.escape(path)}">{html.escape(path)}</a></td><td>{link}</td></tr>')
    return f"<table><tr><th>file</th><th></th></tr>{''.join(rows)}</table>"


def render_report(settings, *, runs, task_model, library, glm_summary, hrf_summary, reliability,
                  tuning, activation_summary, rt_summary, figures, skipped, manifest):
    figures = dict(figures)
    bodies = {
        "settings": _settings_body(settings),
        "inputs": _inputs_body(runs, task_model, library) + _figures(figures, ("Design", "Library")),
        "glms": _table(glm_summary) + _table(hrf_summary) + _figures(figures, ("GLMComparison", "GLMR2Surface")),
        "reliability": _table(reliability) + _figures(figures, ("HRFReliability", "HRFCurveReliability")),
        "betas": _table(tuning) + _figures(figures, ("FractionSelection", "RidgeTuning", "BetaR2Surface")),
        "summaries": _table(activation_summary) + _table(rt_summary)
        + _figures(figures, ("BetaActivation", "BetaActivationSurface", "RTCheck", "RTSurface")),
        "skipped": _skipped_body(skipped),
        "files": _files_body(manifest),
    }
    titles = dict(settings="Settings", inputs="Inputs", glms="Canonical and optimized GLMs", reliability="HRF reliability",
                  betas="Single-trial beta series", summaries="Activation and reaction time", skipped="Skipped stages", files="Files")
    title = f"boldtailor report: {settings.subject} {settings.session} task-{settings.task}"
    sections = "".join(_section(n, titles[n], bodies[n]) for n in SECTIONS)
    return f"<!doctype html><html><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title><style>{_STYLE}</style></head><body><h1>{html.escape(title)}</h1>{sections}</body></html>"
```

Run black and suites → PASS. Commit: `feat: boldtailor.workflow.report renders a self-contained HTML report`.

---

### Task 11: `run.py` orchestration

**Files:**
- Create: `src/boldtailor/workflow/run.py`
- Test: `tests/workflow/test_run.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, kw_only=True)
class WorkflowResult:
    settings: WorkflowSettings; paths: tuple[Path, ...]; report_path: Path | None
    skipped: tuple[tuple[str, str], ...]; task_model: TaskModel; library_fingerprint: str
def run_workflow(settings) -> WorkflowResult
def describe_inputs(settings) -> dict   # for --dry-run: runs discovered, detected task model, library size, output dir
```

Order inside `run_workflow`: `check_output` → `load_session` → `detect_task_model` → `build_library` → blocks → stage `glms` → stage `reliability` (skippable) → stage `betas` → stage `summaries` → figures → report → `save_workflow`. Surface figures only when `settings.surface_maps` and meshes are found. Reliability is skipped with reason text "fewer than two odd or two even runs" when `odd_even_parity` has a half with < 2 runs (load_session is called with `hrf_only=True` so it does not raise; the parity check is done here).

- [ ] **Step 1: Write the failing tests**

```python
# tests/workflow/test_run.py
"""run_workflow executes the enabled stages, skips what it cannot do, and writes the derivative."""

import json
import pytest

from boldtailor.workflow import run as workflow_run


def _func(settings):
    return settings.output_dir / "sub-07" / "ses-nsd10" / "func"


def test_full_run_writes_every_stage_and_the_report(six_run_dataset, settings_for, tmp_path):
    root, _ = six_run_dataset
    settings = settings_for(root, output_dir=tmp_path / "out", hrf_library="sobol", hrf_n_samples=2,
                            ridge_mode="fractional_cv", ridge_fractions=(0.4, 1.0), block_size=4)
    result = workflow_run.run_workflow(settings)
    names = {p.name for p in result.paths}
    assert "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json" in names
    assert "sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k_desc-OptimizedGLM_stat-rsquared.dscalar.nii" in names
    assert "sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k_desc-HRFOdd_stat-selection.dscalar.nii" in names
    assert "sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k_desc-OptimizedTrialFractionalCV_stat-ridgefraction.dscalar.nii" in names
    assert "sub-07_ses-nsd10_task-nsdcore_space-fsLR_den-91k_desc-CanonicalTrialOLS_stat-activation.dscalar.nii" in names
    assert result.report_path == settings.output_dir / "sub-07_ses-nsd10_task-nsdcore_report.html"
    assert result.report_path.exists() and result.skipped == ()
    metadata = json.loads((_func(settings) / "sub-07_ses-nsd10_task-nsdcore_desc-boldtailor_metadata.json").read_text())
    assert metadata["settings"]["ridge_mode"] == "fractional_cv" and metadata["report"] == result.report_path.name
    assert (settings.output_dir / "dataset_description.json").exists()
    assert not any("notebook" in n for n in names)


def test_disabled_stages_write_nothing_of_their_own(four_runs, settings_for, tmp_path):
    root, _ = four_runs
    settings = settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical", stages=frozenset({"glms"}))
    result = workflow_run.run_workflow(settings)
    names = {p.name for p in result.paths}
    assert any("desc-OptimizedGLM" in n for n in names)
    assert not any("Trial" in n or "HRFOdd" in n or "activation" in n for n in names)


def test_reliability_is_skipped_with_a_reason_when_a_parity_is_short(dataset, settings_for, tmp_path):
    root, *_ = dataset  # two runs: one odd, one even
    settings = settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical", ridge_mode="off")
    result = workflow_run.run_workflow(settings)
    assert result.skipped == (("reliability", "fewer than two odd or two even runs"),)
    assert "fewer than two odd" in result.report_path.read_text()


def test_second_run_into_the_same_output_stops_before_loading(four_runs, settings_for, tmp_path):
    root, _ = four_runs
    settings = settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical", stages=frozenset({"glms"}))
    workflow_run.run_workflow(settings)
    with pytest.raises(FileExistsError):
        workflow_run.run_workflow(settings)
    again = workflow_run.run_workflow(settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical",
                                                   stages=frozenset({"glms"}), existing_results="overwrite"))
    assert again.paths


def test_describe_inputs_reports_runs_model_and_output(four_runs, settings_for, tmp_path):
    root, _ = four_runs
    info = workflow_run.describe_inputs(settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical"))
    assert info["runs"] == ["run-01", "run-02", "run-03", "run-04"]
    assert info["task_model"]["regressors"] == ["task", "response_time", "trial_type"]
    assert info["output_dir"] == str(tmp_path / "out") and info["library_candidates"] == 1
```

- [ ] **Step 2: RED, commit, implement**

```python
"""Run the enabled stages in order and publish one BIDS derivative with a report."""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from boldtailor.diagnostics import one_sample_t
from boldtailor.reliability import curve_correlations
from boldtailor.workflow import analysis, beta_series, inputs, outputs, plots, report, surfaces
from boldtailor.workflow.files import odd_even_parity
from boldtailor.workflow.settings import WorkflowSettings

matplotlib.use("Agg")
log = logging.getLogger("boldtailor.workflow")
_SHORT_PARITY = "fewer than two odd or two even runs"


@dataclass(frozen=True, kw_only=True)
class WorkflowResult:
    settings: WorkflowSettings
    paths: tuple[Path, ...]
    report_path: Path | None
    skipped: tuple[tuple[str, str], ...]
    task_model: object
    library_fingerprint: str


def describe_inputs(settings):
    runs = inputs.load_session(settings, hrf_only=True)
    task_model = inputs.detect_task_model([r.events for r in runs], settings.modulators)
    return dict(runs=[r.label for r in runs], task_model=task_model.to_dict(), output_dir=str(settings.output_dir),
                fmriprep_dir=str(settings.fmriprep_dir), library_candidates=len(settings.build_library().candidates),
                stages=[s for s in ("glms", "reliability", "betas", "summaries") if s in settings.stages])


def _glm_stage(runs, settings, blocks, library, task_model):
    root = settings.bids_dir
    model = inputs.glm_model(runs, task_model)
    canonical = analysis.fit_glms(runs, root, blocks, model, n_jobs=settings.n_jobs)
    selections = analysis.select_hrfs(runs, root, blocks, library, n_jobs=settings.n_jobs,
                                      task_model=inputs.selection_task_model(task_model, settings.hrf_selection_rt))
    optimized = analysis.fit_glms(runs, root, blocks, model, selections=selections, n_jobs=settings.n_jobs)
    glms = {"CanonicalGLM": canonical, "OptimizedGLM": optimized}
    return model, selections, glms, analysis.selection_maps(selections, len(runs[0].image.header.get_axis(1)))


def _reliability_stage(library, hrf_maps, figures):
    agreement, figures["HRFReliability"] = plots.parameter_agreement(library, hrf_maps["odd"][0], hrf_maps["even"][0])
    curve_r = curve_correlations(library, hrf_maps["odd"][0], hrf_maps["even"][0])
    curve_summary, figures["HRFCurveReliability"] = plots.curve_agreement(curve_r)
    return agreement, curve_summary


def _summaries_stage(beta_models, runs, brain, figures):
    activation = {name: one_sample_t(m.fit["betas"]) for name, m in beta_models.items()}
    figures["BetaActivation"] = plots.activation_histogram(activation)
    rt_figure = plots.rt_check_figure(beta_models, runs, brain)
    if rt_figure is not None:
        figures["RTCheck"] = rt_figure
    return activation


def _surface_figures(settings, brain, glms, beta_models, activation, figures):
    meshes = surfaces.find_surface_meshes(settings.fmriprep_dir, settings.subject, paths=settings.surface_meshes)
    if not settings.surface_maps or meshes is None:
        return
    figures["GLMR2Surface"] = surfaces.surface_figure({k: v["r2"][0] for k, v in glms.items()}, brain, meshes, statistic="r2")
    if beta_models:
        figures["BetaR2Surface"] = surfaces.surface_figure({k: m.fit["r2"][2] for k, m in beta_models.items()}, brain, meshes, statistic="delta_r2")
    if activation:
        figures["BetaActivationSurface"] = surfaces.surface_figure({k: a["t"] for k, a in activation.items()}, brain, meshes, statistic="t")
        rt = {k: m.fit["rt"]["all"] for k, m in beta_models.items() if m.fit["rt"] is not None}
        if rt:
            figures["RTSurface"] = surfaces.surface_figure(rt, brain, meshes, statistic="rt")


def run_workflow(settings):
    outputs.check_output(settings)
    runs = inputs.load_session(settings, hrf_only=True)
    task_model = inputs.detect_task_model([r.events for r in runs], settings.modulators)
    library = settings.build_library()
    brain = runs[0].image.header.get_axis(1)
    blocks = inputs.make_blocks(runs, block_size=settings.block_size, max_grayordinates=settings.max_grayordinates)
    figures, skipped, beta_models, activation = {}, [], {}, None
    reliability = tuning = None
    model, selections, glms, hrf_maps = _glm_stage(runs, settings, blocks, library, task_model)
    figures["Design"] = plots.design_figure(runs[0].frame_times, glms["CanonicalGLM"]["designs"][0, 0], task_model.regressor_names)
    figures["Library"] = plots.library_figure(library)
    glm_summary, figures["GLMComparison"] = plots.glm_comparison(glms)
    if "reliability" in settings.stages:
        if all(len(half) >= 2 for half in odd_even_parity(runs).values()):
            reliability = _reliability_stage(library, hrf_maps, figures)[0]
        else:
            skipped.append(("reliability", _SHORT_PARITY))
    if "betas" in settings.stages:
        beta_models = beta_series.fit_beta_models(runs, settings.bids_dir, blocks, settings, library, selections, task_model)
        tuning = outputs.tuning_table(beta_models) if any(m.tuning for m in beta_models.values()) else None
        if tuning is not None:
            figures["RidgeTuning"] = outputs.tuning_figure(beta_models)
    if "summaries" in settings.stages:
        activation = _summaries_stage(beta_models, runs, brain, figures)
    _surface_figures(settings, brain, glms, beta_models, activation, figures)
    html = _render(settings, runs, task_model, library, glm_summary, reliability, tuning, activation, beta_models, figures, skipped)
    paths = outputs.save_workflow(settings, runs, task_model, library, selections, glms, beta_models,
                                  figures=figures, activation=activation, skipped=skipped, report_html=html.encode())
    report_path = settings.output_dir / f"{settings.subject}_{settings.session}_task-{settings.task}_report.html"
    return WorkflowResult(settings=settings, paths=tuple(paths), report_path=report_path, skipped=tuple(skipped),
                          task_model=task_model, library_fingerprint=library.fingerprint)
```

`outputs.py` gains `workflow_artifacts(settings, runs, task_model, library, selections, glms, beta_models, *, figures, activation, skipped, report) -> list[Artifact]` (everything `save_workflow` writes except the report and `dataset_description.json`); `save_workflow` calls it and appends the two extras, so the manifest and the published set always agree. `_render` is:

```python
def _render(settings, runs, task_model, library, glm_summary, reliability, tuning, activation, beta_models, figures, skipped):
    import pandas as pd

    artifacts = outputs.workflow_artifacts(settings, runs, task_model, library, None, {}, beta_models,
                                           figures=figures, activation=activation, skipped=skipped, report=None)
    paths = [a.path for a in artifacts]
    manifest = [(p, p.rsplit("_", 1)[0] + "_provenance.json" if p.rsplit("_", 1)[0] + "_provenance.json" in paths else None) for p in paths]
    activation_summary = None if activation is None else pd.DataFrame(
        [dict(model=k, median_t=float(np.nanmedian(v["t"]))) for k, v in activation.items()])
    rt_rows = [dict(model=k, scope=s, median_r=float(np.nanmedian(m.fit["rt"][s])))
               for k, m in beta_models.items() if m.fit["rt"] is not None for s in ("all", "odd", "even")]
    report_name = f"{settings.subject}_{settings.session}_task-{settings.task}_report.html"
    return report.render_report(settings, runs=runs, task_model=task_model, library=library, glm_summary=glm_summary,
                                hrf_summary=outputs.hrf_boundary_table(None), reliability=reliability, tuning=tuning,
                                activation_summary=activation_summary, rt_summary=pd.DataFrame(rt_rows) if rt_rows else None,
                                figures=figures, skipped=skipped, manifest=manifest)
```

Pass `selections` and `glms` through `_render` rather than `None`/`{}` once the signature is in place (the snippet shows the shape; `run_workflow` has both in scope). `outputs.hrf_boundary_table(selections)` is the DataFrame form of `hrf_boundary_summary` (rows → `pd.DataFrame`, or `None`). The `glms["CanonicalGLM"]["designs"][0, 0]` key follows `analysis.fit_glms`.

Run black and suites → PASS. Commit: `feat: run_workflow orchestrates stages, skips short parities with a reason, and publishes the derivative with its report`.

---

### Task 12: Metadata and report content for task-only sessions

**Files:**
- Modify: `src/boldtailor/workflow/outputs.py`, `src/boldtailor/workflow/report.py`
- Test: append to `tests/workflow/test_run.py`

- [ ] **Step 1: Failing test**

```python
def test_task_only_session_runs_without_rt_or_trial_type(four_runs, settings_for, tmp_path):
    import pandas as pd
    root, _ = four_runs
    for path in (root / "sub-07" / "ses-nsd10" / "func").glob("*_events.tsv"):
        pd.read_csv(path, sep="\t").drop(columns=["response_time", "trial_type"]).to_csv(path, sep="\t", index=False)
    settings = settings_for(root, output_dir=tmp_path / "out", hrf_library="canonical", ridge_mode="off")
    result = workflow_run.run_workflow(settings)
    names = {p.name for p in result.paths}
    assert not any("rtcorrelation" in n or "RTCheck" in n for n in names)
    text = result.report_path.read_text()
    assert "Task regressors: task" in text and "No reaction-time column" in text
```

- [ ] **Step 2: RED, commit, implement**: `outputs.metadata` omits `response_time`/`missing_response_time`/`rt_check` keys when RT is absent; `report._inputs_body` appends `<li class="note">No reaction-time column; RT correlations were not computed.</li>` when `"response_time" not in task_model.regressor_names`; `_summaries_stage` already yields no RT figure. Suites → PASS. Commit: `feat: task-only sessions produce complete outputs and say so in the report`.

---

### Task 13: Command-line interface

**Files:**
- Create: `src/boldtailor/cli.py`
- Test: `tests/workflow/test_cli.py`

**Interfaces:**
- Produces: `build_parser() -> argparse.ArgumentParser`, `settings_from_args(args) -> WorkflowSettings`, `main(argv=None) -> int`.

- [ ] **Step 1: Failing tests**

```python
# tests/workflow/test_cli.py
"""boldtailor run: argument parsing, dry run, exit codes."""

import json
import pytest

from boldtailor import cli
from boldtailor.model import Modulator


def _argv(root, *extra):
    return ["run", "--bids-dir", str(root), "--subject", "sub-07", "--session", "ses-nsd10", "--task", "nsdcore", *extra]


def test_parser_builds_settings_with_defaults_and_overrides(four_runs):
    root, _ = four_runs
    args = cli.build_parser().parse_args(_argv(root, "--modulator", "response_time:indicator", "--modulator", "trial_type",
                                               "--ridge-mode", "cv", "--ridge-alphas", "0", "1", "--skip-stage", "reliability",
                                               "--n-jobs", "2", "--no-surface-maps", "--hrf-library", "sobol", "--hrf-n-samples", "8"))
    settings = cli.settings_from_args(args)
    assert settings.modulators == (Modulator("response_time", missing="indicator"), Modulator("trial_type"))
    assert settings.ridge_mode == "cv" and settings.ridge_alphas == (0.0, 1.0)
    assert settings.stages == frozenset({"glms", "betas", "summaries"})
    assert settings.n_jobs == 2 and settings.surface_maps is False
    assert settings.output_dir == root / "derivatives" / "boldtailor_hrf-sobol8s0_ridge-cv"


def test_dry_run_prints_the_plan_and_exits_zero(four_runs, capsys):
    root, _ = four_runs
    assert cli.main(_argv(root, "--dry-run", "--hrf-library", "canonical")) == 0
    out = capsys.readouterr().out
    assert "run-04" in out and "response_time" in out and "boldtailor_hrf-canonical_ridge-fractionalcv" in out


def test_settings_errors_exit_one_with_one_line(four_runs, capsys):
    root, _ = four_runs
    assert cli.main(_argv(root, "--ridge-mode", "lasso")) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "ridge_mode" in err[0]
    assert cli.main(_argv(root, "--surface-mesh", "left=a.gii")) == 1


def test_input_errors_exit_two_and_name_what_is_missing(four_runs, capsys):
    root, _ = four_runs
    assert cli.main(_argv(root, "--task", "other", "--dry-run")) == 2
    assert "task-other" in capsys.readouterr().err
    assert cli.main(_argv(root, "--modulator", "stimulus_id", "--dry-run")) == 2
    assert "stimulus_id" in capsys.readouterr().err


def test_run_writes_the_derivative_and_a_second_run_exits_one(four_runs, tmp_path, capsys):
    root, _ = four_runs
    argv = _argv(root, "--output-dir", str(tmp_path / "out"), "--hrf-library", "canonical", "--ridge-mode", "off",
                 "--skip-stage", "reliability", "--skip-stage", "betas", "--skip-stage", "summaries", "--n-jobs", "1", "--block-size", "2", "--no-surface-maps")
    assert cli.main(argv) == 0
    assert (tmp_path / "out" / "sub-07_ses-nsd10_task-nsdcore_report.html").exists()
    assert cli.main(argv) == 1
    assert "existing_results" in capsys.readouterr().err
    assert cli.main([*argv, "--existing-results", "overwrite"]) == 0


def test_skip_stage_cannot_remove_glms(four_runs):
    root, _ = four_runs
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(_argv(root, "--skip-stage", "glms"))
```

(`four_runs` comes from `tests/workflow/conftest.py`.)

- [ ] **Step 2: RED, commit, implement**

```python
"""Command line: `boldtailor run` executes the workflow on one BIDS subject and session."""

import argparse
import json
import sys
from pathlib import Path

from boldtailor.workflow.settings import HRF_LIBRARIES, RIDGE_MODES, SUPPORTED_SPACES, WorkflowSettings, parse_modulator

_SKIPPABLE = ("reliability", "betas", "summaries")


def _mesh(text):
    side, _, path = text.partition("=")
    if side not in ("left", "right") or not path:
        raise argparse.ArgumentTypeError("surface mesh must be left=PATH or right=PATH")
    return side, Path(path)


def build_parser():
    parser = argparse.ArgumentParser(prog="boldtailor", description="First-level fMRI modelling with tailored HRFs.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the full workflow on one subject and session")
    paths = run.add_argument_group("inputs")
    paths.add_argument("--bids-dir", type=Path, required=True)
    paths.add_argument("--subject", required=True); paths.add_argument("--session", required=True); paths.add_argument("--task", required=True)
    paths.add_argument("--fmriprep-dir", type=Path); paths.add_argument("--output-dir", type=Path)
    paths.add_argument("--space", choices=SUPPORTED_SPACES, default="fsLR-91k")
    paths.add_argument("--modulator", action="append", type=parse_modulator, metavar="COLUMN[:indicator]",
                       help="task modulator; repeat to list all; default detects response_time and trial_type")
    hrf = run.add_argument_group("HRF selection")
    hrf.add_argument("--hrf-library", choices=HRF_LIBRARIES, default="default")
    hrf.add_argument("--hrf-n-samples", type=int, default=512); hrf.add_argument("--hrf-seed", type=int, default=0)
    hrf.add_argument("--no-rt-in-hrf-selection", action="store_true")
    ridge = run.add_argument_group("beta series")
    ridge.add_argument("--ridge-mode", choices=RIDGE_MODES, default="fractional_cv")
    ridge.add_argument("--ridge-alpha", type=float, default=0.1)
    ridge.add_argument("--ridge-fractions", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ridge.add_argument("--ridge-alphas", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    ridge.add_argument("--ridge-percentile", type=float, default=90.0)
    ridge.add_argument("--encoding-mode", choices=("within_run", "absolute"), default="within_run")
    stages = run.add_argument_group("stages and figures")
    stages.add_argument("--skip-stage", action="append", choices=_SKIPPABLE, default=[])
    stages.add_argument("--no-surface-maps", action="store_true")
    stages.add_argument("--surface-mesh", action="append", type=_mesh, metavar="left=PATH|right=PATH")
    execution = run.add_argument_group("execution")
    execution.add_argument("--n-jobs", type=int, default=4); execution.add_argument("--block-size", type=int, default=4096)
    execution.add_argument("--max-grayordinates", type=int)
    execution.add_argument("--existing-results", choices=("error", "overwrite"), default="error")
    execution.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit")
    return parser


def settings_from_args(args):
    meshes = dict(args.surface_mesh) if args.surface_mesh else None
    return WorkflowSettings(
        bids_dir=args.bids_dir, subject=args.subject, session=args.session, task=args.task,
        fmriprep_dir=args.fmriprep_dir, output_dir=args.output_dir, space=args.space,
        modulators=tuple(args.modulator) if args.modulator else None,
        hrf_library=args.hrf_library, hrf_n_samples=args.hrf_n_samples, hrf_seed=args.hrf_seed,
        hrf_selection_rt=not args.no_rt_in_hrf_selection,
        ridge_mode=args.ridge_mode, ridge_alpha=args.ridge_alpha, ridge_fractions=tuple(args.ridge_fractions),
        ridge_alphas=tuple(args.ridge_alphas), ridge_percentile=args.ridge_percentile, encoding_mode=args.encoding_mode,
        stages=frozenset({"glms", *(s for s in _SKIPPABLE if s not in args.skip_stage)}),
        surface_maps=not args.no_surface_maps, surface_meshes=meshes,
        n_jobs=args.n_jobs, block_size=args.block_size, max_grayordinates=args.max_grayordinates,
        existing_results=args.existing_results,
    )


def _run(args):
    from boldtailor.workflow import run as workflow_run

    settings = settings_from_args(args)
    if args.dry_run:
        plan = dict(settings=settings.to_dict(), **workflow_run.describe_inputs(settings))
        print(json.dumps(plan, indent=2))
        return 0
    result = workflow_run.run_workflow(settings)
    print(f"wrote {len(result.paths)} files under {settings.output_dir}; report: {result.report_path}")
    return 0


def main(argv=None):
    from boldtailor.workflow.inputs import InputError

    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (InputError, FileNotFoundError) as error:
        print(f"input error: {error}", file=sys.stderr)
        return 2
    except (ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
```

`InputError` (Task 5) is a `ValueError`, so it must be caught first. `files.discover_runs` raises `FileNotFoundError` for missing events or CIFTI files, which also maps to exit 2. `check_output`'s message must contain "existing_results".

Run black and the suite → PASS; also `uv run boldtailor run --help` prints. Commit: `feat: boldtailor run command-line interface with dry run and exit codes`.

---

### Task 14: Re-point the remaining examples and retire the legacy scripts

**Files:**
- Delete: `examples/NSD/nsd_hrf.py`, `nsd_single_trial.py`, `nsd_cifti.py`, `hrf_artifacts.py`, `single_trial_artifacts.py`, `rt_diagnostics.py`, `parallel_blocks.py`, `workflow_reuse.py`, `ridge_reuse.py`, `notebook_paths.py`, `settings.py`, and tests `test_nsd_cifti.py`, `test_nsd_hrf_selection.py`, `test_nsd_parallel.py`, `test_nsd_single_trial.py`, `test_nsd_split_hrf.py`, `test_rt_diagnostics.py`, `test_workflow_reuse.py`, `test_settings.py`, `test_notebook_helpers.py` (path tests; plot tests moved in Task 9), `test_workflow_artifacts.py` (emptied by Tasks 4 and 8), `test_nsd_workflow.py` (emptied by Tasks 5-6 except the notebook test, which moves to `test_nsd_notebooks.py`)
- Modify: `examples/NSD/session_hrf.py`, `session_hrf_cache.py`, `session_hrf_import.py`, `session_hrf_outputs.py`, `multisession_inputs.py`, `multisession_workflow.py`, `multisession_outputs.py`, `multisession_plots.py`, `conftest.py`, the three notebooks, `examples/NSD/README.md`

**Interfaces:**
- Consumes: everything from Tasks 3-11. The examples build `WorkflowSettings(bids_dir=..., subject=..., session=..., task="nsdcore", ...)` from a `NSD_CONFIG` dict (`WorkflowSettings.from_dict` after mapping `bids_root → bids_dir`, `fmriprep_root → fmriprep_dir`, `output_root → output_dir`).

- [ ] **Step 1: Update the opt-in tests first (RED)**: in `test_session_hrf.py`, `test_multisession_*.py`, replace imports of `workflow_inputs`/`workflow_analysis`/`workflow_outputs`/`workflow_artifacts` with `boldtailor.workflow.*`; `_stem` → `settings.stem`; `_metadata(...)` → `outputs.metadata(...)`; `_input_artifacts(stem, runs)` → `outputs.input_artifacts(settings, runs, task_model)`; every `desc-notebook` → `desc-`; `saved_sessions` fixture writes `_desc-boldtailor_metadata.json`, `_desc-HRF_library.tsv`, etc. `multisession_workflow.ensure_session_outputs` is tested to call `boldtailor.workflow.run.run_workflow` (monkeypatch it) instead of executing the notebook. Create `examples/NSD/test_nsd_notebooks.py` holding the three `@notebook` tests. Run `uv run pytest -q examples/NSD` → FAIL. Commit.

- [ ] **Step 2: Re-point the modules**: in `session_hrf*.py` and `multisession_*.py` replace the relative imports with package imports (`from boldtailor.workflow.inputs import load_session, load_block, make_blocks, selection_task_model, detect_task_model`, `from boldtailor.workflow.artifacts import ...`, `from boldtailor.workflow.outputs import R2_NAMES, metadata, input_artifacts`, `from boldtailor.workflow.surfaces import surface_figure`), build settings through a small `examples/NSD/nsd_settings.py` helper:

```python
"""NSD notebooks: build WorkflowSettings from the notebook configuration dictionary."""

from boldtailor.workflow.settings import WorkflowSettings

_RENAMED = {"bids_root": "bids_dir", "fmriprep_root": "fmriprep_dir", "output_root": "output_dir"}


def nsd_settings(config, **overrides):
    values = {_RENAMED.get(k, k): v for k, v in dict(config).items()}
    values.setdefault("task", "nsdcore")
    values.update(overrides)
    return WorkflowSettings.from_dict(values)
```

`multisession_workflow.ensure_session_outputs` calls `run_workflow(nsd_settings(config, session=s, surface_maps=False, existing_results="overwrite"))` for each missing session. `multisession_inputs` reads `_desc-boldtailor_metadata.json` and the renamed descriptors. `session_hrf_import` compares against `outputs.metadata(...)` and reads `desc-HRFAll`, `desc-HRF_library.*`, `desc-HRF_provenance.json`, `desc-boldtailor_runs.tsv`, `desc-boldtailor_events.tsv`, `desc-boldtailor_confounds.tsv`.

- [ ] **Step 3: Rewrite `nsd_workflow.ipynb`** as a thin notebook: cell 1 config (`NSD_CONFIG`), cell 2 `settings = nsd_settings(NSD_CONFIG)`, cell 3 `result = run_workflow(settings)`, then cells that display the report path, `pd.read_csv` of the runs table, and the saved PNG figures by reading `desc-*_plot.png` files (keeping the section headings as markdown for narrative). The notebook test asserts the report exists and the beta/activation file counts as before (16 `_betas.`, 4 `_stat-activation.`, 4 `_trials.tsv`, retained scans). Re-execute the two other notebooks' code cells only as far as their imports (they need real data for the rest).

- [ ] **Step 4: Delete** the retired files with `git rm`, prune `examples/NSD/conftest.py` to the notebook hooks, the synthetic-BIDS fixture wrappers, `saved_sessions`, and anything the remaining tests use. Update `examples/NSD/README.md`: the "Command-line scripts" section becomes a pointer to `boldtailor run`.

- [ ] **Step 5: Run** `uv run pytest -q -W error && uv run pytest -q examples/NSD` → both PASS (the opt-in notebook tests need `--run-notebooks`; run `uv run pytest -q examples/NSD --run-notebooks -k workflow` once to confirm the thin notebook executes on the fixture).

Commit: `refactor(examples): NSD notebooks and session helpers use boldtailor.workflow; legacy scripts and result reuse retired`.

---

### Task 15: Documentation

**Files:**
- Modify: `README.md`, `docs/user-guide.md`, `docs/api.md`, `docs/README.md`, `docs/development.md`, `docs/glmsingle-comparison.md`
- Test: `tests/test_docs_mention_cli.py`

- [ ] **Step 1: Failing test**

```python
"""The CLI and workflow package are documented where users look first."""

from pathlib import Path


def test_readme_and_user_guide_document_boldtailor_run():
    readme = Path("README.md").read_text()
    guide = Path("docs/user-guide.md").read_text()
    api = Path("docs/api.md").read_text()
    assert "boldtailor run --bids-dir" in readme and "boldtailor run --bids-dir" in guide
    assert "--modulator" in guide and "derivatives/boldtailor_hrf-" in guide
    assert "WorkflowSettings" in api and "run_workflow" in api and "render_report" in api
```

- [ ] **Step 2: Write the docs**: README gets a "Command line" section with the one-line invocation and the output-directory convention. The user guide gets a "Running the full workflow" section: the command, every flag grouped as in the parser, modulator detection and syntax, the four stages and `--skip-stage`, the output layout and descriptor vocabulary, the report, exit codes, and `existing_results`. The API reference gets a `boldtailor.workflow` table (`WorkflowSettings`, `run_workflow`, `describe_inputs`, `load_session`, `detect_task_model`, `fit_beta_models`, `BetaModel`, `save_workflow`, `render_report`) and the `boldtailor.cli` entry points. `docs/development.md` notes that `tests/workflow` runs by default and `examples/NSD` is opt-in. `docs/glmsingle-comparison.md` row "HRF library" already updated; add a row "Command line" (`boldtailor run` vs GLMsingle's MATLAB/Python function call).

- [ ] **Step 3: Suite → PASS. Commit:** `docs: boldtailor run, workflow settings, outputs and report`.

---

### Task 16: Whole-branch verification

- [ ] `uv run pytest -q -W error` (default), `uv run pytest -q examples/NSD`, `uv run pytest -q examples/validation`, `uv run black --check src tests examples`, `git diff --check`.
- [ ] `grep -rn "examples" src/boldtailor` returns nothing; `grep -rn "desc-notebook" src examples docs` returns nothing; `grep -rn "center=" src tests examples docs` returns nothing.
- [ ] `uv run boldtailor run --help` exits 0; `uv run boldtailor run --bids-dir <fixture> ... --dry-run` prints JSON (use a `tmp` dataset created by `python -c "from tests.workflow.synthetic_bids import *"` or the pytest fixture through a tiny script).
- [ ] Record in `docs/superpowers/validation/2026-10-03-workflow-cli-ledger.md` the test counts before and after and any rulings made during execution.
