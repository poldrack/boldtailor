# Task 1 Implementation Report

## Commit range

- Base: `d8050aa7105a35c41f492419353ad2d3774aecf2`
- Implementation head before this report commit: `e55663d811e461abc91c49d5d148d690ea09ba3c`
- Initial test-only commit: `9e2bf11937670a4fdc5ca432420c6145fe4edfff`
  (`test: specify whole-brain example loading`)
- Test-only dimensionality correction: `37eb1f8748defc962445f1c676656cad1967a10f`
  (`test: correct masker round-trip dimensionality`)
- Production commit: `e55663d811e461abc91c49d5d148d690ea09ba3c`
  (`feat: load common-mask whole-brain signals`)
- The report commit is recorded in the final handoff because a commit cannot contain
  its own identifier.

## RED evidence

The required focused command was run before any production edit:

```text
uv run pytest tests/test_stop_signal_demo.py -k "common_brain or masker or whole_brain or signal_memory" -q
exit 2
ERROR tests/test_stop_signal_demo.py
ImportError: cannot import name 'common_brain_mask' from
'examples.stop_signal_demo'
1 error in 1.06s
```

That failure was the expected missing-interface failure. The tests were then
committed alone as `9e2bf11` before `examples/stop_signal_demo.py` was touched.

During GREEN, the round-trip test exposed a specification error. Nilearn 0.14
documents and implements a one-dimensional return from `NiftiMasker.transform`
for a three-dimensional image. Therefore the original assertion
`masker.transform(image)[0]` compared a scalar with the feature vector while the
same test also required the reconstructed image to remain three-dimensional.
The test was corrected to compare `masker.transform(image)` with the vector and
committed separately as the test-only commit `37eb1f8`.

Warning-strict execution then supplied a second RED case:

```text
uv run pytest tests/test_stop_signal_demo.py -k "common_brain or masker or whole_brain or signal_memory" -q -W error
exit 1
2 failed, 3 passed, 25 deselected in 1.17s
FutureWarning: boolean values for 'standardize' will be deprecated in nilearn 0.15.0.
Use 'zscore_sample' instead of 'True' or use 'None' instead of 'False'.
```

The warning originated at the required `standardize=False` transform boundary.
Production suppresses only that exact `FutureWarning` inside the masker transform
call. The direct round-trip test uses the same narrow local suppression.

## GREEN evidence

The focused warning-strict behavior selection passed after implementation:

```text
uv run pytest tests/test_stop_signal_demo.py -k "common_brain or masker or whole_brain or signal_memory" -q -W error
5 passed, 25 deselected in 1.05s
```

Fresh verification after production commit `e55663d`:

```text
uv run pytest tests/test_stop_signal_demo.py -q
30 passed in 6.69s

uv run pytest -q -W error
217 passed in 8.49s
```

`uv run git diff --check` exited 0, and the empty-`__init__.py` invariant check
exited 0. No `__init__.py` file was modified.

## Files changed

- `tests/test_stop_signal_demo.py`
- `examples/stop_signal_demo.py`
- `.superpowers/sdd/2026-08-09-whole-brain-notebook/task-1-report.md`

The SDD progress ledger was not modified.

## Requirement coverage

- Computes a nonempty logical intersection of run masks with preserved first-mask
  geometry and a uint8 header/data representation.
- Rejects mismatched run-mask shape or affine with the required message.
- Builds and fits one `NiftiMasker` with standardization, detrending, smoothing,
  low-pass filtering, high-pass filtering, and reports explicitly disabled.
- Rejects BOLD images that differ from the fitted common-mask geometry.
- Transforms every complete BOLD run to an owned, immutable float64
  `n_scans x n_common_voxels` array and validates dimensions, feature count, and
  finiteness.
- Reconstructs a finite one-dimensional feature vector through the fitted masker
  and preserves mask-space ordering on round trip.
- Estimates signal storage from positive scan counts, a positive voxel count, and
  eight bytes per float64 value.
- Updates the example fit fixture to use all 343 synthetic common-mask voxels and
  exactly the three approved contrast expressions.
- Retains confound, event, source metadata, publication-path, and protected-source
  behavior tests.
- Adds no compatibility aliases and does not touch the real dataset or main
  checkout.

## Concerns and scoped deviations

The unchanged notebook still imports and calls the legacy ROI helpers and passes
ROI indices to `load_run`. Deleting those helpers and the legacy `LoadedRun`
spatial fields in Task 1 would make the existing notebook execution tests fail
before Task 3 is allowed to update the notebook. The parent authorized a scoped
sequencing correction: existing ROI helpers, the legacy positional `load_run`
branch, and the legacy spatial fields are retained temporarily, with no new aliases
or deprecation behavior. Task 3 must migrate the notebook first and then delete
that legacy code in its production commit. This deviation preserves a GREEN suite
between tasks while leaving the approved final no-ROI behavior assigned to Task 3.

The round-trip assertion correction and exact Nilearn warning suppression are
compatibility adjustments required by the installed Nilearn 0.14 behavior. They
do not alter the requested whole-brain data flow or preprocessing settings.
