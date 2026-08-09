# Task 3 Report: Executable Whole-Brain Notebook

## Scope and base

- Worktree: `/Users/poldrack/Dropbox/code/boldtailor/.worktrees/whole-brain-notebook`
- Branch: `feature/whole-brain-notebook`
- Accepted Task 3 base: `1065890294a7d1520dffc6928177fa5613b7f729`
- Baseline verification: `uv run pytest -q -W error` -> `218 passed in 8.85s`

## RED evidence

Tests were changed before any helper, notebook, or README implementation.

- Command: `uv run pytest tests/test_stop_signal_demo.py -k "notebook or readme" -q`
- Result: `4 failed, 5 passed, 22 deselected in 6.60s`
- Expected failures:
  - `MASK_STRATEGY` was absent from notebook configuration.
  - `go_success_vs_baseline` was absent from both repository-root and notebook-directory executions.
  - The README notebook link did not describe a whole-brain example.
- Rewritten test-only commit: `073fa538dd23b095ebd325bf4ecafbfbfa605bea` (`test: specify whole-brain notebook story`)
- Commit contents: `tests/test_stop_signal_demo.py` only.

The RED commit also added explicit behavior coverage that loaded signals are owned, write-protected `float64` arrays and that the shared fitted masker gives identical feature ordering for the same spatial pattern in both runs. After senior review identified two Important coverage gaps, the local unmerged Task 3 history was rewritten so stronger behavior tests still precede production. A detached replay at the final test-only commit ran the exact plotting and metadata nodes and produced `3 failed in 19.19s`: both working-directory cases observed one statistical-map call rather than the required four, while the published configuration contained ROI metadata and lacked the required mask, masker, resource, signal, and contrast metadata.

## Implementation

- Replaced the notebook's bounded extraction workflow with one common intersection mask and one explicitly non-preprocessing fitted `NiftiMasker` shared by both runs.
- Added the 32 GiB minimum-memory and no-feature-chunking assumption, voxel/scan/shape/memory summaries, transformed signal shape and dtype provenance, and JSON-safe mask/masker/resource metadata.
- Fit exactly `successful_inhibition`, `stop_vs_go`, and `go_success_vs_baseline` with the approved expressions and AR(1) noise.
- Plotted three descriptive unthresholded whole-brain z maps and one aggregate R-squared map, with the multiple-comparison disclosure displayed by executable code.
- Published the Task 2 whole-brain artifact set with ten NIfTI images and the deterministic image manifest while retaining projected BIDS provenance, protected source paths, transactional publication, temporary-default output, exact persistent destination restriction, and privacy-safe displays.
- Removed all extraction-specific narrative, configuration, names, helpers, legacy `LoadedRun` spatial fields, legacy loading dispatch, and the all-omitted artifact signature branch.
- Updated the README link to describe the two-session common-mask whole-brain example.
- Ran `uv run black src tests examples/stop_signal_demo.py` during refactoring because Task 3 touched the files.

Rewritten implementation commit: `3e5d418d589332b44a4cd72b7a84e07aaa6b3d92` (`docs: demonstrate whole-brain stop-signal analysis`). It includes every actual implementation file: `README.md`, `examples/stop_signal_demo.ipynb`, `examples/stop_signal_demo.py`, plus Black-only formatting in `tests/test_stop_signal_demo.py`.

## GREEN and verification evidence

- `uv run pytest tests/test_stop_signal_demo.py -k "notebook or readme" -q` -> `9 passed, 22 deselected in 7.54s`
- Strengthened plotting and published-metadata nodes -> `3 passed in 11.06s`
- `uv run pytest tests/test_stop_signal_demo.py -q -W error` -> `32 passed in 11.55s`
- `uv run pytest -q -W error` -> `219 passed in 13.41s`
- `uv run black --check src tests examples/stop_signal_demo.py` -> `24 files would be left unchanged`
- `uv lock --check` -> `Resolved 76 packages in 3ms`
- `uv run pytest tests/test_stop_signal_demo.py -k "notebook_executes" -vv`:
  - repository-root: passed
  - notebook-directory: passed
  - total: `2 passed, 29 deselected in 7.30s`
- `uv run pytest tests/test_stop_signal_demo.py -k "masker_preserves_whole_brain_values or shared_masker_preserves_feature_order" -q` -> `2 passed, 29 deselected in 1.08s`
- Empty-initializer check -> `checked 1 initializers; nonempty=[]`
- `uv run git diff --check 1065890294a7d1520dffc6928177fa5613b7f729..HEAD` -> exit 0
- `uv run git diff --check` -> exit 0
- Production audit found no case-insensitive extraction-specific term or removed migration symbol in `examples/stop_signal_demo.py`, `examples/stop_signal_demo.ipynb`, or `README.md`.

## Coverage summary

The focused suite covers configuration defaults and validation, execution from both supported working directories, exact contrast expressions and AR(1), three unthresholded z-map calls and one aggregate R-squared map call, common-mask loading, immutable owned `float64` signals, cross-run feature ordering, image count and manifest publication, whole-brain artifact naming and round-trip geometry, complete published mask/masker/resource/signal metadata, absence of the absolute BIDS root from shareable outputs, protected inputs, and persistent-publication path/symlink restrictions.

## Deviations and concerns

- Real-data execution was not run because it is explicitly Task 4 scope; Task 3 used only the synthetic fixture and did not touch the real dataset or persistent derivative destination.
- The notebook had zero stored outputs at the Task 3 base. The mechanical source migration preserved cell output state and did not add an output-clearing requirement.
- The plan's sample production staging command omitted `examples/stop_signal_demo.py`; the implementation commit intentionally includes it, as required by the Task 3 brief.
- The pre-review Task 3 history remains recoverable at safety ref `task3-review-safety-dbc5b14`; only the local unmerged Task 3 commits were rewritten to restore strict RED-before-production ordering for the strengthened tests.
- The worktree retains only the untracked `__pycache__` directories that were already present at baseline; they were not modified or committed.
- `uv` prints an environment-selection notice because the parent checkout's `VIRTUAL_ENV` differs from the worktree `.venv`; all project commands nevertheless used the worktree environment and the warning-strict pytest suite passed.
