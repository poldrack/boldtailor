# Final Important Fix Report

## Scope

Closed the final-review Important requiring every `TaskDeltaR2Result` array to
own its immutable float64 storage. The two deferred Minor review items were not
addressed.

## RED

- Strengthened `test_task_delta_r2_compares_complete_and_nuisance_models` so
  all four public result arrays require literal `flags.owndata is True`.
- Retained the existing one-dimensional shape, float64 dtype, no-alias, and
  `setflags(write=True)` rejection assertions.
- Confirmed the focused test failed diagnostically with `OWNDATA : False`.
- Committed the test-only RED state as `8ec4cda` (`test: require owned
  immutable result arrays`).

## GREEN

- Replaced bytes-buffer-backed arrays with a narrowly scoped ndarray subclass
  allocated directly with its own C-order float64 storage.
- Copied source values before freezing the owned storage.
- Rejected public `setflags(write=True)` re-enablement after construction.
- Preserved scalar, one-dimensional, multidimensional, and empty-shape array
  construction behavior.
- Committed the production-only fix as `b46147c` (`fix: own immutable result
  array storage`).

## Verification

- RED command: `uv run pytest tests/test_fit.py -k task_delta_r2_compares_complete_and_nuisance_models -q`.
  It failed at the new literal ownership assertion with `OWNDATA : False`:
  1 failed, 29 deselected.
- Local GREEN evidence before the production commit:
  - focused ownership regression: 1 passed, 29 deselected;
  - `tests/test_fit.py`: 30 passed;
  - full warning-strict suite: 246 passed;
  - Black: 24 files unchanged;
  - `git diff --check`: clean.
- Controller post-commit focused evidence:
  - `tests/test_fit.py`: 30 passed in 0.93 seconds;
  - Black: 2 files clean.
  - The chained git-diff check encountered the known transient uv macOS panic;
    the controller will perform final repository verification.

Existing untracked notebook, package, and test caches were left untouched.
