# Task 1 report: shared statistical-map slices

## Outcome

Implemented the shared slice contract for all five statistical map calls in
`examples/stop_signal_demo.ipynb`:

- `DISPLAY_MODE = "z"`
- `CUT_COORDS = np.arange(-20, 60, 15)`
- Explicit `display_mode=DISPLAY_MODE` and `cut_coords=CUT_COORDS` on the
  three contrast z-score maps, aggregate R-squared map, and task-delta
  R-squared map.

Extended the runtime plot audit in `tests/test_stop_signal_demo.py` to record
bound plotting arguments and require `display_mode == "z"` and cut coordinates
`[-20, -5, 10, 25, 40, 55]` for all five calls. Existing value, threshold,
color, title, and nonnegativity assertions were retained.

## TDD commits

1. `416843e test: require shared statistical map slices`
   - Test-only diagnostic RED commit.
   - Prescribed notebook execution failed in both parameterized cases because
     Nilearn bound the default `display_mode` (`"ortho"`) instead of `"z"`.
2. `d383236 docs: align r-squared map slices`
   - Notebook-only GREEN implementation commit.

## Verification

- RED focused execution: 2 failures, both at the new `display_mode` assertion.
- GREEN focused execution before implementation commit: `2 passed`.
- Full warning-strict suite before implementation commit: `246 passed`.
- Final focused execution after the amended implementation commit: `2 passed`.
- Black check: passed (`24 files would be left unchanged`).
- Lockfile check: passed.
- `git diff --check`: passed.
- Notebook preservation projection: passed. Only the `results` cell source
  changed; all cell IDs, execution counts, outputs, and notebook metadata were
  unchanged.

## Self-review and concerns

The implementation diff contains only the requested test additions and eight
notebook source-array additions. A temporary JSON indentation-only formatting
noise issue was caught during self-review and removed before the final commit.
The worktree still contains the pre-existing untracked generated directory
`src/boldtailor.egg-info/`; it was not touched or staged.
