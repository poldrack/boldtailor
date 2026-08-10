# Task 3 report: deterministic delta R-squared derivative

## Commits

- `0b6e59e test: specify delta r-squared derivative` (initial RED)
- `53f47de feat: serialize task delta r-squared map` (intentional raw-value
  implementation retained to demonstrate the regression)
- `a1a9a58 test: distinguish clipped delta r-squared derivative` (genuine RED)
- `0a40135 fix: serialize clipped delta r-squared values` (clipped fix)

## Delivered

- `result_artifacts` now requires keyword-only `task_delta: TaskDeltaR2Result`.
- It verifies that the supplied common mask has the fitted masker's shape,
  affine, and boolean support, and that the delta map has the fitted masker's
  voxel count.
- It serializes the clipped task-attributable map after the aggregate R-squared
  map as `desc-taskDelta_stat-r2_statmap.nii.gz`; the existing manifest picks
  it up automatically.
- The derivative test now uses a real `task_delta_r2` comparison and checks all
  11 image paths, deterministic bytes, NIfTI geometry/values, manifest fields,
  nonnegative output, and both mask-mismatch variants.
- A dedicated regression constructs a real comparison with raw delta R-squared
  equal to `-0.25` at one voxel, then verifies that its derivative payload is
  zero at that location, nonnegative everywhere, and equal to the clipped
  `delta_r2` array. Direct negative cases also cover mismatched common-mask
  shape and mismatched delta-map length.

## TDD evidence

- RED: `uv run pytest tests/test_stop_signal_demo.py -k result_artifacts -q`
  failed in all four selected cases with `TypeError: result_artifacts() got an
  unexpected keyword argument 'task_delta'`.
- GREEN: `uv run pytest tests/test_stop_signal_demo.py -k result_artifacts -q -W error`
  passed: `4 passed, 30 deselected`.
- `uv run git diff --check` passed before the production commit.

## Review regression evidence

- With `53f47de` deliberately serializing `task_delta.raw_delta_r2`, the new
  focused RED run produced `1 failed, 6 passed, 30 deselected`. The intended
  failure was exact: the decoded derivative's first value was `-0.25`, while
  the test required `0.0`.
- After `0a40135` restored `task_delta.delta_r2`, the parent reran
  `uv run pytest tests/test_stop_signal_demo.py -k result_artifacts -q -W error`:
  `7 passed, 30 deselected in 1.13s`.

## Broader-suite note

`uv run pytest -q -W error` completed with `239 passed, 3 failed`. The three
failures are the pre-Task-4 notebook tests: its publication cell still calls
`result_artifacts` without the now-required `task_delta=` keyword. No notebook
was changed because it is outside Task 3's assigned file scope; Task 4 is
expected to fit/pass that comparison into the call.

The requested broad Black check was interrupted before it produced a result;
the focused test file was Black-formatted before its RED commit, and the
production file was checked/formatted during the focused GREEN run.

## Preserved workspace state

Pre-existing untracked cache/egg-info directories were left untouched:
`examples/__pycache__/`, `src/boldtailor.egg-info/`,
`src/boldtailor/__pycache__/`, and `tests/__pycache__/`.
