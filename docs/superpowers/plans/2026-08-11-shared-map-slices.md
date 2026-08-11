# Shared Map Slices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all five stop-signal statistical maps use the same axial display mode and cut coordinates.

**Architecture:** Keep slice selection local to the notebook results cell as two shared constants. Extend the existing runtime plot audit so it verifies the exact settings passed to every Nilearn `plot_stat_map` call.

**Tech Stack:** Python 3.12+, uv, pytest, nbformat/nbclient, NumPy, Nilearn 0.14.x, Jupyter notebook JSON.

## Global Constraints

- Use `uv` for package management and `uv run` for every local command.
- Every `__init__.py` remains byte-empty.
- Commit a diagnostic failing test before changing notebook source.
- Preserve all stored notebook outputs, execution counts, cell identifiers, and unrelated metadata.
- Do not modify or commit the main checkout's existing `pyproject.toml`, `uv.lock`, notebook-output, or cache changes.
- Execute in an isolated worktree created from committed `main`; integrate only after warning-strict verification.
- Use `DISPLAY_MODE = "z"` and `CUT_COORDS = np.arange(-20, 60, 15)`.
- Preserve every map's existing values, threshold, color map, colorbar settings, symmetry settings, and title.

---

### Task 1: Share Slice Settings Across All Statistical Maps

**Files:**
- Modify: `tests/test_stop_signal_demo.py`
- Modify: `examples/stop_signal_demo.ipynb` (`results` cell source only)

**Interfaces:**
- Consumes: the existing `_record_plot_stat_map` instrumentation and five-call `_assert_plot_contract` runtime audit.
- Produces: five plot calls whose bound `display_mode` is `"z"` and whose bound `cut_coords` equal `[-20, -5, 10, 25, 40, 55]`.

- [ ] **Step 1: Create an isolated worktree**

Use `superpowers:using-git-worktrees` from committed `main`. Confirm the isolated checkout is clean and record its base commit. Do not copy the main checkout's stored notebook outputs or other uncommitted files into it.

- [ ] **Step 2: Extend the plot audit with a failing contract**

In `PLOT_AUDIT_REPORT`, serialize the bound plotting arguments:

```python
"display_mode": _arguments["display_mode"],
"cut_coords": (
    None
    if _arguments["cut_coords"] is None
    else np.asarray(_arguments["cut_coords"]).tolist()
),
```

At the start of `_assert_plot_contract`, require the exact shared settings for every recorded call:

```python
expected_cut_coords = [-20, -5, 10, 25, 40, 55]
assert len(audit) == 5
for call in audit:
    assert call["display_mode"] == "z"
    assert call["cut_coords"] == expected_cut_coords
```

Keep all existing value, threshold, color, title, and nonnegativity assertions unchanged.

- [ ] **Step 3: Verify RED**

Run the two real notebook execution cases with task-specific writable Jupyter directories and local socket permission:

```bash
JUPYTER_CONFIG_DIR=/private/tmp/boldtailor-shared-slices-red-jupyter-config \
JUPYTER_DATA_DIR=/private/tmp/boldtailor-shared-slices-red-jupyter-data \
JUPYTER_RUNTIME_DIR=/private/tmp/boldtailor-shared-slices-red-jupyter-runtime \
IPYTHONDIR=/private/tmp/boldtailor-shared-slices-red-ipython \
PYTHONDONTWRITEBYTECODE=1 \
uv run --no-cache --no-sync pytest \
  tests/test_stop_signal_demo.py::test_notebook_executes_against_fixture \
  -q -W error -p no:cacheprovider
```

Expected: both parameterized cases fail because the current R-squared calls bind Nilearn's default display mode/cut coordinates rather than the required shared settings.

- [ ] **Step 4: Commit the RED test only**

```bash
uv run --no-cache --no-project git add tests/test_stop_signal_demo.py
uv run --no-cache --no-project git commit -m "test: require shared statistical map slices"
```

Confirm the commit changes only `tests/test_stop_signal_demo.py`.

- [ ] **Step 5: Implement shared notebook constants**

At the start of the `results` cell, before constructing `contrast_rows`, add:

```python
DISPLAY_MODE = "z"
CUT_COORDS = np.arange(-20, 60, 15)
```

Pass the shared values explicitly in all five plot calls:

```python
display_mode=DISPLAY_MODE,
cut_coords=CUT_COORDS,
```

This replaces any per-contrast literal slice arguments and adds the same two arguments to aggregate R-squared and task-delta R-squared. Make only source-array edits in the notebook JSON; do not alter `outputs`, `execution_count`, cell IDs, or notebook metadata.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
JUPYTER_CONFIG_DIR=/private/tmp/boldtailor-shared-slices-green-jupyter-config \
JUPYTER_DATA_DIR=/private/tmp/boldtailor-shared-slices-green-jupyter-data \
JUPYTER_RUNTIME_DIR=/private/tmp/boldtailor-shared-slices-green-jupyter-runtime \
IPYTHONDIR=/private/tmp/boldtailor-shared-slices-green-ipython \
PYTHONDONTWRITEBYTECODE=1 \
uv run --no-cache --no-sync pytest \
  tests/test_stop_signal_demo.py::test_notebook_executes_against_fixture \
  -q -W error -p no:cacheprovider

JUPYTER_CONFIG_DIR=/private/tmp/boldtailor-shared-slices-full-jupyter-config \
JUPYTER_DATA_DIR=/private/tmp/boldtailor-shared-slices-full-jupyter-data \
JUPYTER_RUNTIME_DIR=/private/tmp/boldtailor-shared-slices-full-jupyter-runtime \
IPYTHONDIR=/private/tmp/boldtailor-shared-slices-full-ipython \
PYTHONDONTWRITEBYTECODE=1 \
uv run --no-cache --no-sync pytest -q -W error -p no:cacheprovider

uv run --no-cache --no-sync black --check src tests examples/stop_signal_demo.py
uv lock --check
uv run --no-cache --no-project git diff --check
```

Expected: focused and full suites pass warning-strict, Black and lock checks pass, and the diff is clean.

- [ ] **Step 7: Verify notebook-output preservation**

Compare the implementation commit candidate with the RED-test parent. Require that the notebook diff changes only `source` entries in the `results` cell and that the projection of every cell's `id`, `execution_count`, and `outputs`, plus notebook metadata, is identical.

- [ ] **Step 8: Commit the notebook implementation**

```bash
uv run --no-cache --no-project git add examples/stop_signal_demo.ipynb
uv run --no-cache --no-project git commit -m "docs: align r-squared map slices"
```

Confirm the commit changes only `examples/stop_signal_demo.ipynb`.

- [ ] **Step 9: Request review and integrate**

Use `superpowers:requesting-code-review` across the task base through the notebook commit. Require clean TDD ordering, exact settings on all five calls, preservation of every prior plot option/value, robust runtime assertions, and byte-identical notebook outputs/metadata. Route any Critical or Important finding through another committed RED-GREEN cycle. After approval, use `superpowers:finishing-a-development-branch` and preserve the main checkout's uncommitted files during integration.
