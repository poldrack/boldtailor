# Task 4 report

Implemented the notebook display and shareable provenance story on
`feature/task-delta-r2`.

## Commits

- Initial RED tests: `bd3f643 test: specify task delta r-squared notebook story`
- Deliberately incomplete review baseline:
  `5bac286 docs: display task delta r-squared`
- Review RED tests:
  `c7c7682 test: verify task delta notebook diagnostics`
- Raw-signal false-positive review baseline:
  `eb14c86 fix: publish complete task delta notebook diagnostics`
- Raw-display RED tests:
  `77712ec test: reject raw signal notebook displays`
- Final compact-display correction:
  `a3c4928 fix: keep task delta notebook display compact`

Complete notebook versions remain recoverable at
`safety/task-delta-r2-task4-fd2db4c` (`fd2db4c`) and
`safety/task-delta-r2-task4-d86b117` (`d86b117`).

## Implementation

- Fits `task_delta_r2` immediately after the complete model.
- Publishes a compact `variance_partition` configuration with nuisance-model
  settings and all four delta R-squared diagnostics.
- Displays nuisance-design columns and the compact comparison summary.
- Adds the fifth, nonnegative magma delta map with the required clipping title
  and descriptive/non-inferential disclosure.
- Projects comparison provenance and passes `task_delta` into artifact
  publication, producing the eleventh image.
- Preserved the notebook's existing output arrays; no output clearing was
  performed.
- Runtime tests bind all four published metrics to the live comparison, require
  the exact nuisance-model mapping, and audit the fifth map's minimum and
  nonnegative values.
- A structural display audit bounds displayed arrays/tables and rendered text,
  uses `np.shares_memory` to detect ndarray, DataFrame, and Series views into
  every loaded run's raw signals, and requires exactly 11 analysis displays.
- The display audit is reported only by a final injected cell after publication,
  so raw-signal displays added to later notebook cells remain observable.
- A static AST contract requires direct `display(variance_partition)` in the
  real design-fit cell and rejects displayed arguments that reference
  `.signals` or a `signals` name.

## Verification

Focused GREEN command:

```text
JUPYTER_CONFIG_DIR=/private/tmp/boldtailor-task4-red-jupyter-config JUPYTER_DATA_DIR=/private/tmp/boldtailor-task4-red-jupyter-data JUPYTER_RUNTIME_DIR=/private/tmp/boldtailor-task4-red-jupyter-runtime IPYTHONDIR=/private/tmp/boldtailor-task4-red-ipython PYTHONDONTWRITEBYTECODE=1 uv run --no-cache --no-sync pytest tests/test_stop_signal_demo.py -k notebook -q -p no:cacheprovider
```

Review RED against `5bac286`: `3 failed, 7 passed, 27 deselected in 10.97s`.
Both cwd executions failed because `nuisance_design_columns` was absent; the
metadata execution failed on the incomplete nuisance mapping after capturing
the live delta metrics.

Corrected GREEN result: `10 passed, 27 deselected in 10.98s`.

The concrete raw-display false positive at `eb14c86` replaced
`display(variance_partition)` with a shared-memory 80-by-1 raw-signal slice;
the prior focused suite incorrectly passed all 10 notebook tests in 10.67s.

Raw-display RED result: `3 failed, 8 passed, 27 deselected in 10.48s`. The
static test rejected the missing direct summary, while both cwd executions
detected the raw slice through shared memory.

Final strengthened GREEN result: `11 passed, 27 deselected in 10.42s`.

Each restored notebook passed an exact `git diff --exit-code` comparison against
its corresponding safety ref, establishing that the full notebook content,
including stored output fields, was preserved exactly.

`uv run --no-cache --no-sync --no-project git diff --check` also completed
successfully before the GREEN commit. Per controller instruction, no full-suite
command was run. Existing generated untracked directories were not modified or
removed.
