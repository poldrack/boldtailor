# Task 3 senior re-review

**Range:** `5050e934eceabfc99ee5ef8166a5ef4da56dee1f..0a40135`

**Decision:** **Approved to proceed to Task 4.**

## Findings

No Critical, Important, or Minor findings remain.

## Resolution of prior findings

- **Raw-versus-clipped derivative gap: resolved.** Commit `53f47de`
  deliberately serializes `task_delta.raw_delta_r2`. Its child `a1a9a58` is a
  test-only commit that constructs a valid comparison with raw delta `-0.25`
  at one voxel, decodes the derivative, and requires that voxel to be `0.0`,
  all serialized values to be nonnegative, and the complete masked payload to
  equal `negative_comparison.delta_r2`
  (`tests/test_stop_signal_demo.py:719-756`). The supplied RED evidence is
  exact and diagnostic: `1 failed, 6 passed, 30 deselected`, with actual
  `-0.25` versus expected `0.0`. Commit `0a40135` changes only the serializer
  operand from `raw_delta_r2` to `delta_r2`; the controller's fresh strict
  focused run passes `7 passed, 30 deselected in 1.13s`. A raw serializer can
  no longer pass the focused suite.
- **Mask shape coverage: resolved.** The mismatch parameterization now covers
  `shape`, `voxel_support`, and `affine`
  (`tests/test_stop_signal_demo.py:759-784`). The production validation checks
  exact shape, affine closeness, and equality of boolean support
  (`examples/stop_signal_demo.py:235-246`).
- **Delta-length coverage: resolved.** The new test supplies a valid
  `TaskDeltaR2Result` one voxel shorter than the fitted mask and requires the
  dedicated dimension error (`tests/test_stop_signal_demo.py:787-810`). The
  implementation compares the one-dimensional delta size with the fitted
  mask's boolean voxel count (`examples/stop_signal_demo.py:249-254`), with
  `whole_brain_image` retaining its independent validation.

## TDD and implementation audit

- `0b6e59e` remains the initial test-only RED commit on parent `5050e93`.
- `53f47de` changes only `examples/stop_signal_demo.py` and intentionally
  contains the incomplete raw serializer used for the review regression.
- `a1a9a58` has parent exactly `53f47de` and changes only
  `tests/test_stop_signal_demo.py`; it is therefore a genuine committed RED
  before the clipping fix.
- `0a40135` has parent exactly `a1a9a58` and changes only
  `examples/stop_signal_demo.py`, replacing the raw array with the clipped
  array. No test is weakened by the fix.
- The final implementation requires keyword-only `task_delta`, validates the
  common mask against the fitted masker, validates delta dimensions, and emits
  exactly one delta image after aggregate complete-model R-squared.
- The final derivative path is exactly
  `images/sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_desc-taskDelta_stat-r2_statmap.nii.gz`.
- Deterministic gzip bytes still use `mtime=0`. The existing generic manifest
  includes the eleventh image and the focused test verifies exact ordered
  paths, media type, byte sizes, and SHA-256 hashes.
- `git diff --check` is clean for the reviewed range. Only the two scoped files
  change across the range.
- The three previously reported broad-suite failures remain the expected Task
  4 notebook call-site boundary; Task 3 correctly does not edit the notebook.
- No tests were run during this re-review. Existing untracked generated
  directories were not touched or deleted.

## Severity summary

- Critical: 0
- Important: 0
- Minor: 0

