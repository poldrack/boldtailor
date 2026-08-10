# Task 5 verification report

Status: all real-data and repository verification gates passed. Final fix
verification completed at branch HEAD
`b46147caa5b00530c0694c97c972cc41e4ed28e1`.

## Real-data notebook

- Read-only dataset: `/Users/poldrack/data_unsynced/rdoc_fmri`
- Default sessions: `ses-06`, `ses-08`
- Persistent guard before: `publication_exists_before=False`
- Execution: `NotebookClient(timeout=7200)` from `examples/`, with task-specific
  Jupyter, IPython, Matplotlib, and publication directories under
  `/private/tmp/boldtailor-task5-real.P5R1Uc`
- Execution result: passed in approximately 36 seconds
- Rendered checks: both sessions, all three contrasts, all four delta-R-squared
  summary fields, `clipped at zero`, and `published_count`
- Temporary derivative checks: exactly 11 NIfTI images, including
  `sub-s4_task-stopSignal_space-MNI152NLin2009cAsym_res-2_desc-taskDelta_stat-r2_statmap.nii.gz`
- Persistent guard after: `publication_exists_after=False`

The dataset was not modified. Publication remained temporary.

## Repository gates

- Focused warning-strict notebook suite: `38 passed in 11.48s`
- Full warning-strict suite: `246 passed in 13.43s`
- Black: `24 files would be left unchanged`
- Lock check: `Resolved 76 packages in 10ms`
- Build: source distribution and wheel built successfully beneath
  `/private/tmp/boldtailor-task-delta-r2-dist`
- Repository contracts: `2 passed in 0.00s`
- Initializers: `checked 1 initializers; nonempty=[]`
- `git diff --check`: passed

After the final owned-array fix, the focused fit suite passed `30 tests in
0.93s`, the full warning-strict suite passed `246 tests in 12.55s`, Black left
all 24 files unchanged, the initializer check remained clean, and
`git diff --check` passed.

`src/boldtailor.egg-info` existed before Task 5, so it was not removed. A
transient uv macOS system-configuration panic occurred only while chaining a
read-only `git status` after the repository-contract test; the contract test
had already passed, and an immediate standalone status/diff retry exited zero.

## Final worktree status

Only pre-existing generated, untracked directories remain:

```text
?? examples/__pycache__/
?? src/boldtailor.egg-info/
?? src/boldtailor/__pycache__/
?? tests/__pycache__/
```

No tracked file was modified by verification.

## Deferred review note

Minor deferred from Task 2: direct regression coverage for multi-run pooled
R-squared and task-level zero-SST behavior is still missing.
