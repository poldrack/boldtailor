# Scientific Readability Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Establish a passing, portable baseline for simplifying Boldtailor without mixing in the preserved scientific remediation.

**Architecture:** Keep scientific algorithms and result interfaces unchanged in this stage. Correct a mismatched notebook fixture, expose software versions through a small lazy helper, and validate the installed distribution and complete suite through ordinary pytest and CI commands.

**Tech Stack:** Python >=3.12, uv, pytest, NumPy, pandas, Nilearn, nbclient, setuptools, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-28-scientific-readability-design.md`

## Global Constraints

- All Python commands use `uv run`; every `__init__.py` stays empty.
- Commit failing tests before implementation; retain independent numerical and leakage tests.
- Never apply stash `c4cfbec5fccd5397ea650c550f5b485ea588c2cd` wholesale.
- Preserve scientific calculations and interpretation.
- Public API and output-metadata changes are permitted when they materially simplify the project; this stage needs no public fit API changes.
- Keep `nilearn>=0.14.0,<0.15`, `scipy>=1.18.0`, and Python `>=3.12` as currently declared; do not upgrade dependencies incidentally.
- No generic workflow framework, numerical solver rewrite, result-schema migration, or publication rewrite in this stage.

## Review Focus

1. Notebook fixtures must retain the same trial labels and contrasts as the committed scientific example (Task 1).
2. Notebook execution must work from both repository root and the notebook directory, with private temporary paths absent from published metadata (Task 1).
3. Importing from a source checkout without Boldtailor distribution metadata must succeed, while actual missing dependency metadata must not be silently fabricated (Task 2).
4. Exported software versions must reflect the installed distribution instead of a stale hardcoded value (Task 2).
5. Default test collection must include NSD tests, and a wheel used outside the checkout must import without relying on the dev environment (Task 3).

## File responsibilities and dependencies

| File | Responsibility |
| --- | --- |
| `tests/test_stop_signal_demo.py` | Correct fixture and expected notebook labels; preserve scientific/runtime checks |
| `examples/stop_signal_demo.ipynb` | Add missing interactive views required by existing committed tests, without changing modeling |
| `src/boldtailor/_software.py` | Lazy package-version lookup with honest source-checkout fallback |
| `src/boldtailor/prepared.py` | Request versions when creating provenance, not at import |
| `src/boldtailor/prepared_fit.py` | Request numerical backend version when creating fit activity |
| `src/boldtailor/bids_provenance.py` | Obtain the installed Boldtailor version when projecting metadata |
| `tests/test_software.py` | Source-checkout and missing-metadata behavior |
| `tests/test_bids_provenance.py` | Installed-version propagation into output |
| `tests/test_distribution.py` | Runtime dependency and initializer requirements |
| `tests/check_installed_package.py` | Real installed-wheel import/OLS smoke check, invoked explicitly |
| `pyproject.toml`, `uv.lock` | Full collection and correct runtime/dev dependencies |
| `.github/workflows/tests.yml` | Clean-runner full-suite, formatting, and wheel checks |
| `.gitignore`, `docs/development.md` | Generated-file hygiene and reproducible developer commands |
| `docs/validation/scientific-readability-foundation-2026-09-28.md` | Actual commands, results, and justification for fixture corrections |

Tasks run in order. Task 1 removes the known failing baseline; Task 2 fixes
runtime metadata; Task 3 makes those checks portable and automatic.

## Task 1: Reconcile the committed notebook and its fixtures

**Files:** `tests/test_stop_signal_demo.py`, `examples/stop_signal_demo.ipynb`,
`docs/validation/scientific-readability-foundation-2026-09-28.md`.

**Interfaces:** Preserve the notebook's `TRIAL_TYPES`, `ModelSpec` contrasts,
prepared fit boundary, output artifact names, and statistical definitions.
No numerical package API changes.

- [x] Run the already committed failing regressions:

```sh
uv run pytest tests/test_stop_signal_demo.py -q -W error -k 'variance_partition_prepared_runtime or variance_partition_notebook_publishes'
```

Expected RED: three failures for undefined `go_success`. These tests are already
committed, satisfying the tests-before-implementation requirement. Record their
commit (`57acff5`), command, and failure in the validation note before editing.

- [x] Record the demonstrated test error: `_configure_notebook_fixture` changes
  `go_success`/`go_failure` into `go`; the committed notebook has neither that
  event contract nor the new `stop_success_vs_go` contrast. Correct only the
  erroneous fixture/expectations, retaining all metadata and parity checks:

```python
NOTEBOOK_CONTRAST_EXPRESSIONS = {
    "successful_inhibition": "stop_success - stop_failure",
    "stop_vs_go": "(stop_success + stop_failure) - go_success",
    "go_success_vs_baseline": "go_success",
}
```

Remove the event-file rewriting loop from `_configure_notebook_fixture`; keep
its temporary dataset/session configuration. Change the runtime rendered-name
assertion from `stop_success_vs_go` to `stop_vs_go`. Neither the task contrast
weights nor trial grouping may change. Run the same regressions to expose any
remaining failures; document them. Commit this justified test correction before
changing the notebook.

```sh
git add tests/test_stop_signal_demo.py docs/validation/scientific-readability-foundation-2026-09-28.md
git commit -m "test: align notebook fixtures with committed scientific model"
```

- [x] Preserve the committed interactive-view requirement. The current notebook
  does not call `view_img`, while its tests assert five views. Add those views
  in the results cell. Use the same images as the static plots:

```python
INTERACTIVE_CUT_COORDS = (0, 0, int(CUT_COORDS[len(CUT_COORDS) // 2]))
# Inside the existing contrast loop, after the static plot:
display(plotting.view_img(
    whole_brain_image(result.z_score(contrast_name), masker),
    cut_coords=INTERACTIVE_CUT_COORDS,
    threshold=None,
    colorbar=True,
    title=f"{contrast_name} interactive z-score",
))
# After the aggregate and delta static plots:
display(plotting.view_img(
    whole_brain_image(result.r2, masker),
    cut_coords=INTERACTIVE_CUT_COORDS,
    threshold=None,
    colorbar=True,
    cmap="viridis",
    symmetric_cmap=False,
    vmin=0,
    title="Aggregate fit quality (R-squared), interactive",
))
display(plotting.view_img(
    whole_brain_image(task_delta.delta_r2, masker),
    cut_coords=INTERACTIVE_CUT_COORDS,
    threshold=None,
    colorbar=True,
    cmap="magma",
    symmetric_cmap=False,
    vmin=0,
    title="Task-attributable delta R-squared, interactive",
))
```

Use nbformat through `uv run` to edit the identified cell and leave outputs
empty. This selectively restores missing presentation behavior already required
by committed tests; it does not restore the stash's numerical or timing changes.
Before writing, read `_assert_interactive_view_contract` for its full image and
argument contract. Do not weaken it to make absent views pass.

- [x] Run all notebook/example tests and then the complete baseline suite:

```sh
uv run pytest tests/test_stop_signal_demo.py -q -W error
uv run pytest tests examples/NSD -q -W error
uv run black --check tests/test_stop_signal_demo.py
git diff --check
```

If further mismatches appear, trace them to the committed notebook or test
contract; never delete scientific parity/privacy checks to obtain green. Record
the actual totals and corrected test assumptions, then commit the notebook and
validation note with `fix: reconcile committed stop-signal notebook baseline`.

## Task 2: Resolve software versions when used

**Files:** create `src/boldtailor/_software.py`, `tests/test_software.py`;
modify `prepared.py`, `prepared_fit.py`, `bids_provenance.py`, and
`tests/test_bids_provenance.py`.

**Interfaces:** `package_version(name: str) -> str` returns installed metadata;
only missing metadata for `boldtailor` returns `"unknown"`. No cache, import-time
metadata lookup, or version constant in `__init__.py`.

- [x] Add these tests to `tests/test_software.py`:

```python
import importlib
from importlib import metadata
import subprocess
import sys
import textwrap

import pytest


def test_source_checkout_version_is_explicitly_unknown(monkeypatch):
    software = importlib.import_module("boldtailor._software")

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(software.metadata, "version", missing)
    assert software.package_version("boldtailor") == "unknown"
    with pytest.raises(metadata.PackageNotFoundError):
        software.package_version("numpy")


def test_source_import_does_not_require_own_distribution_metadata():
    code = textwrap.dedent('''
        from importlib import metadata
        installed_version = metadata.version
        def version(name):
            if name == "boldtailor":
                raise metadata.PackageNotFoundError(name)
            return installed_version(name)
        metadata.version = version
        import boldtailor.prepared
        import boldtailor.prepared_fit
        import boldtailor.bids_provenance
    ''')
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_imports_do_not_query_versions():
    code = textwrap.dedent('''
        import importlib
        from importlib import metadata
        names = (
            "boldtailor.prepared", "boldtailor.prepared_fit",
            "boldtailor.bids_provenance",
        )
        modules = [importlib.import_module(name) for name in names]
        def unexpected(name):
            raise AssertionError(f"import-time version lookup: {name}")
        metadata.version = unexpected
        for module in modules:
            importlib.reload(module)
    ''')
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
```

Add this output-level test in `tests/test_bids_provenance.py`, reusing that
module's existing `provenance_record` and `projection_options` fixtures:

```python
def test_projection_uses_installed_version(
    monkeypatch, provenance_record, projection_options,
):
    from importlib import metadata

    original = metadata.version
    monkeypatch.setattr(
        metadata, "version",
        lambda name: "9.8.7" if name == "boldtailor" else original(name),
    )
    files = project_bids_provenance(provenance_record, **projection_options)
    description = json.loads(files["dataset_description.json"])
    generated = description["GeneratedBy"]
    assert next(x for x in generated if x["Name"] == "Boldtailor")["Version"] == "9.8.7"
    software = json.loads(files["prov/prov-boldtailor_soft.json"])["Software"]
    assert next(x for x in software if x["Label"] == "Boldtailor")["Version"] == "9.8.7"
```

- [x] Run RED and commit tests before adding `_software.py`:

```sh
uv run pytest tests/test_software.py tests/test_bids_provenance.py -q -W error
git add tests/test_software.py tests/test_bids_provenance.py
git commit -m "test: require lazy and accurate software versions"
```

Expected: missing helper/source import failure, import-time metadata lookup,
and hardcoded projection version.

- [x] Implement the small helper:

```python
"""Software versions for provenance, resolved only when requested."""

from importlib import metadata


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        if name != "boldtailor":
            raise
        return "unknown"
```

Replace `_SOFTWARE_VERSIONS` in `prepared.py` with the following expression
inside `_normalization_activity`:

```python
"software_versions": {
    name: package_version(name) for name in ("boldtailor", "numpy", "pandas")
},
```

Replace `_NUMERICAL_BACKEND` in `prepared_fit.py` at fit-activity construction:

```python
"numerical_backend": {"name": "nilearn", "version": package_version("nilearn")},
```

Both modules import the helper instead of importing metadata's version function
directly. Replace both uses of `BOLDTAILOR_VERSION` in `bids_provenance.py` with
`package_version("boldtailor")`; delete the constant. Keep MappingProxyType
imports if other code still uses them. Do not change provenance field names.

- [x] Run the affected suites and commit after GREEN:

```sh
uv run pytest tests/test_software.py tests/test_bids_provenance.py tests/test_prepared.py tests/test_prepared_fit.py -q -W error
uv run black src/boldtailor/_software.py tests/test_software.py tests/test_bids_provenance.py
git diff --check
git add src/boldtailor/_software.py src/boldtailor/prepared.py src/boldtailor/prepared_fit.py src/boldtailor/bids_provenance.py tests/test_software.py tests/test_bids_provenance.py
git commit -m "refactor: resolve software versions at provenance creation"
```

## Task 3: Make the full suite and installed package the standard checks

**Files:** create `tests/test_distribution.py`, `tests/check_installed_package.py`,
`.github/workflows/tests.yml`; modify `pyproject.toml`, `uv.lock`, `.gitignore`,
`docs/development.md`, and the validation note.

**Interfaces:** `uv run pytest -q -W error` collects `tests` and `examples/NSD`.
The core installed distribution imports and performs a synthetic OLS fit without
ipykernel. Notebook execution dependencies remain in the dev group.

- [x] Add `tests/test_distribution.py`:

```python
from importlib import metadata
from pathlib import Path


def test_notebook_kernel_is_not_a_runtime_requirement():
    requirements = metadata.requires("boldtailor") or []
    assert not any(req.lower().startswith("ipykernel") for req in requirements)


def test_all_package_initializers_are_empty():
    root = Path(__file__).parents[1] / "src" / "boldtailor"
    assert all(path.read_bytes() == b"" for path in root.rglob("__init__.py"))
```

Capture default collection before the config change with
`uv run pytest --collect-only -q`: NSD tests are absent.

- [x] Add a standalone wheel smoke script `tests/check_installed_package.py`:

```python
"""Run explicitly in an isolated environment containing the built wheel."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import boldtailor
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared


def main():
    assert "site-packages" in Path(boldtailor.__file__).parts
    assert importlib.util.find_spec("ipykernel") is None
    rng = np.random.default_rng(721)
    x = rng.normal(size=60)
    design = pd.DataFrame({"task": x, "constant": np.ones(60)})
    y = (2 * x + 10 + rng.normal(scale=0.1, size=60))[:, None]
    prepared = PreparedDesignAnalysis.from_arrays(
        y, design, tr=1.0,
        column_roles={"task": "task", "constant": "intercept"},
    )
    result = fit_prepared(prepared, contrasts={"task": "task"}, noise_model="ols")
    expected = np.linalg.lstsq(design.to_numpy(), y, rcond=None)[0][0]
    np.testing.assert_allclose(result.effect("task"), expected, atol=1e-10)


if __name__ == "__main__":
    main()
```

Run `uv run pytest tests/test_distribution.py -q -W error`; the dependency
assertion must fail. Run the smoke script against the baseline wheel to record
the notebook dependency failure: build with `uv build --wheel`, then run:

```sh
uv run --isolated --no-project --with ./dist/boldtailor-0.1.0-py3-none-any.whl python tests/check_installed_package.py
```

The isolated environment must install the wheel, not resolve the editable project.
The script checks the imported module path and an independent OLS reference.

- [x] Commit both failing checks before implementation:

```sh
git add tests/test_distribution.py tests/check_installed_package.py
git commit -m "test: require a standalone library without notebook dependencies"
```

- [x] Remove `ipykernel>=7.3.0` from runtime dependencies, retain it in dev, set
  pytest `testpaths = ["tests", "examples/NSD"]`, then run `uv lock` and
  `uv sync --locked --group dev`. Keep the existing pythonpath setting until
  the shared-oracle migration in Stage 2; avoid silently breaking fixture imports.
  Add these ignore entries without removing any existing user files:

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.coverage
.pytest_cache/
build/
dist/
```

- [x] Add `.github/workflows/tests.yml`:

```yaml
name: tests
on: [push, pull_request]
permissions:
  contents: read
jobs:
  test:
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
      - run: uv build --wheel
      - run: uv run --isolated --no-project --with ./dist/boldtailor-0.1.0-py3-none-any.whl python tests/check_installed_package.py
```

Document `uv run pytest` as the complete suite in `docs/development.md`, retain
the explicit-path command as an equivalent diagnostic, and describe wheel
validation. Consolidate the existing repository-contract initializer test with
the recursive one rather than keeping duplicates. Keep meaningful supported
Python/layout constraints; do not add tests pinning YAML text or action versions.

- [x] Run GREEN and final stage checks:

```sh
uv run pytest --collect-only -q
uv run pytest -q -W error
uv run black --check src tests examples/NSD examples/stop_signal_demo.py
uv build --wheel
uv run --isolated --no-project --with ./dist/boldtailor-0.1.0-py3-none-any.whl python tests/check_installed_package.py
git diff --check
```

Confirm collection includes `examples/NSD/test_fractional_workflow.py` and
`examples/NSD/test_workflow_surfaces.py`. If the broad formatting check identifies
pre-existing differences, inspect and format only those files with Black; keep
format-only changes clearly labeled. Do not relax the CI check to hide them.

Record totals, wheel smoke output, and known limitations in the validation note.
CI is configured here; without a remote/run, report it as unexecuted remotely.
Stage exact task files and commit as `build: validate full suite and installed package`.

## Stage completion and next handoff

Require three independent outcomes: notebook baseline passes, version output is
accurate and lazy, and the complete suite plus installed-wheel smoke pass.
Review the diff for numerical changes (none intended) and confirm stash `c4cfbec`
still exists. Update this plan's checkboxes with the actual evidence.

Continue to Stage 2 of the roadmap by writing its numerical/CV implementation
plan against the resulting tree. Do not claim the full architectural refactor
complete when this foundation stage finishes.
