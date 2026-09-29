# Example notebook simplification implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Steps use checkbox syntax for tracking.

**Goal:** Make the NSD notebooks readable scientific narratives without adding dataset or imaging APIs to Boldtailor.

**Architecture:** The user approved keeping array modeling in the core and NSD/CIFTI/NIfTI adapters in examples. Extract presentation and path configuration into small example-local functions. Keep model specification, HRF construction, selection, fit calls, and interpretation visible in notebooks. No package moves, new extras, or scientific changes.

**Tech Stack:** Python, NumPy, pandas, Matplotlib, Jupyter, pytest, uv.

**Spec:** [Revised scientific readability design](../specs/2026-09-28-scientific-readability-design.md), as corrected by the user's request to retain generic modeling and example-local imaging.

## Global constraints

- Continue on `refactor/scientific-readability`; leave main and the preserved stash intact.
- Use `uv run` for Python, formatting, and tests; commit failing tests before implementation.
- Every `__init__.py` remains empty.
- Keep scientific models, data selection, units, output names, and NaN masks unchanged.
- No core dependency or package API changes. Example helpers are checkout-local.

## Review focus

- Missing data paths must fail with configuration guidance, without writing output.
- Explicit notebook overrides take precedence over environment defaults.
- Plots must preserve finite-value masks, paired comparisons, and canonical references.
- Undefined or constant HRF selections must remain readable without warnings.
- Notebooks must still run on synthetic inputs and export scientific artifacts.

### Task 1: Separate notebook presentation and local configuration

**Files:** Create `examples/NSD/notebook_paths.py`, `workflow_plots.py`, `session_hrf_plots.py`, and corresponding `test_notebook_helpers.py`; modify the two NSD notebooks and their documentation. Retain fitting modules in place.

**Interfaces:** `notebook_paths(overrides=None)` returns string paths for `bids_root`, `fmriprep_root`, and `output_root`, from explicit overrides or `NSD_BIDS_ROOT`, `NSD_FMRIPREP_ROOT`, `NSD_OUTPUT_ROOT`; derivative defaults follow the selected BIDS root. Plot functions consume existing library/results and return Figures or `(DataFrame, Figure)`, with no display or fit side effects.

- [ ] Write direct behavior tests for path precedence, missing paths, plotted HRF curves/peak-time colors, paired parameter/curve summaries, GLM comparisons, session matrices, and all-undefined inputs. For example:

  ```python
  monkeypatch.setenv("NSD_BIDS_ROOT", str(tmp_path))
  paths = helpers.notebook_paths({"output_root": str(tmp_path / "results")})
  assert paths["fmriprep_root"] == str(tmp_path / "derivatives/fmriprep-25.2.5")
  assert paths["output_root"] == str(tmp_path / "results")
  ```

- [ ] Run `uv run pytest examples/NSD/test_notebook_helpers.py -q -W error`; expect failures naming the absent helpers. Commit tests.
- [ ] Implement small example-local path and plot functions. Move the corresponding notebook plotting blocks into them, returning figures explicitly; keep all fitting calls visible. Replace personal paths with the helper, preserving injected configuration. Update the preview test setup to supply its synthetic BIDS root.
- [ ] Run direct helper tests and existing NSD notebook execution tests; expect all passing. Check actual figure data, not source-text spelling.
- [ ] Update NSD README, developer guide, roadmap and design to describe the example/core boundary and configuration. Clear notebook outputs and execution counts.
- [ ] Run full `uv run pytest -q -W error`, Black, `git diff --check`, wheel build and isolated installed-package smoke test. Commit verified implementation and documentation.
- [ ] One independent review of this increment; address substantive findings with RED–GREEN tests. Record evidence and remaining full-branch review separately. Keep branch unmerged.

## Execution rulings

The user explicitly approved this revised scope and asked to proceed; execute inline without another permission cycle. Continue in the existing refactor checkout. Scientific notebook instrumentation outside the NSD examples remains for the final branch review; this increment does not weaken those tests. Existing complete-notebook tests remain because they exercise different supported workflows.
