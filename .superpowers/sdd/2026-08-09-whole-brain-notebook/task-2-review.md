# Task 2 Senior Review

## Scope and verdict

- Range reviewed: `7016713ab1c1e95329afc18e278132b2b5b18377..a41808046788c3b431e2146cce2dbd5f28fca560`
- TDD verdict: **PASS**
- Spec verdict: **PASS**
- Quality verdict: **PASS**
- Readiness: **READY for Task 3**

No Critical, Important, or Minor findings were identified in the reviewed
range.

## TDD audit

The history has the required test-before-production order:

1. `fbcb961` changes only `tests/test_stop_signal_demo.py` and specifies the
   whole-brain artifact contract. The recorded RED is credible: the parent
   implementation has the old keyword-only ROI signature, so the new `space`
   argument fails before any image can be produced.
2. `720b851` changes only `tests/test_stop_signal_demo.py` and adds the partial
   spatial-context boundary before its guard implementation. The recorded RED
   (`AttributeError` while trying to serialize the absent common mask) proves
   the then-current image implementation did not yet satisfy the required
   explicit rejection.
3. `7166422` is the first production commit and changes only
   `examples/stop_signal_demo.py`; it implements both deterministic image
   derivatives and the all-or-none spatial-context guard.
4. `a418080` adds only the implementation report.

The boundary RED was observed against the uncommitted pre-guard image work, so
that exact `AttributeError` is documented in the report rather than
reconstructible from commit `720b851` alone. This does not invert the required
ordering: the boundary test was written, observed failing, and committed before
the guard production code, while the original image behavior was already
covered by the earlier committed RED test.

## Specification and implementation audit

- `result_artifacts` retains design TSVs, emits the renamed whole-brain contrast
  TSV and unchanged sorted shareable JSON configuration, then appends image
  artifacts and the image manifest in deterministic category order
  (`examples/stop_signal_demo.py:228-297`).
- The whole-brain branch produces exactly ten images for the specified two-run
  result: common mask; effect and z maps for the three fixed contrast mappings;
  two session-aligned run-R-squared maps; and aggregate R-squared. Effect, z,
  run-R-squared, and aggregate values are reconstructed through the fitted
  masker (`examples/stop_signal_demo.py:363-440`).
- The filenames use exactly `successfulInhibition`, `stopVsGo`, and
  `goSuccessVsBaseline`, with the required subject, session, task, space,
  resolution, statistic, and description entities
  (`examples/stop_signal_demo.py:23-27`, `393-440`).
- Each NIfTI is serialized with gzip level 9 and `mtime=0`; a fresh independent
  probe confirmed identical repeated bytes, a literal zero gzip timestamp, and
  successful nibabel shape/affine round-trip
  (`examples/stop_signal_demo.py:443-448`).
- The focused test independently decompresses all ten artifacts, loads every
  NIfTI, checks `(7, 7, 7)` geometry and affine, and verifies common-mask,
  contrast, per-run R-squared, and aggregate R-squared voxel values against the
  source result (`tests/test_stop_signal_demo.py:409-430`).
- The manifest has exactly `relative_path`, `media_type`, `byte_size`, and
  `sha256`; it includes only the ten images in the fixed image order, uses
  `application/gzip`, and derives size and SHA-256 from the actual immutable
  payload bytes (`examples/stop_signal_demo.py:451-466`,
  `tests/test_stop_signal_demo.py:432-452`).
- Exact artifact-path equality, absence of `roi` from the whole-brain branch,
  deterministic repeated output, TSV contents, privacy-safe JSON configuration,
  and transactional publication remain covered
  (`tests/test_stop_signal_demo.py:295-456`).
- The temporary legacy branch is narrower than the ordinary optional-default
  signature might suggest: it is selected only when masker, common mask, space,
  and resolution are all `None`; any partial context raises the specific
  `ValueError`; and a complete context always selects the whole-brain path
  (`examples/stop_signal_demo.py:247-277`, `300-313`). An independent exhaustive
  probe exercised all 14 partial combinations with zero failures. The unchanged
  notebook calls the helper with all four omitted, so the bridge preserves only
  the authorized old behavior. Task 3 must remove the branch and optional
  defaults after migrating that notebook call.
- Configuration serialization is unchanged apart from being factored into the
  shared report helper. Provenance projection remains notebook-owned and is not
  changed by this range. `protected_source_paths` remains intact and continues
  to cover events, BOLD, mask, and confounds for every run
  (`examples/stop_signal_demo.py:280-297`, `337-342`).
- Production helpers are short and single-purpose. Only the expected example,
  focused test, and report files changed. No `__init__.py` changed, all source
  initializers remain byte-empty, and `git diff --check` passes.

## Fresh verification

```text
uv run pytest tests/test_stop_signal_demo.py -k 'result_artifacts' -q -W error
2 passed, 29 deselected in 1.10s
exit 0

uv run pytest -q -W error
218 passed in 8.96s
exit 0

uv run git diff --check 7016713..a41808046788c3b431e2146cce2dbd5f28fca560
exit 0
```

`git status --short` contains only the pre-existing untracked `__pycache__/`
directories. This requested review file is present and uncommitted but is hidden
from status by the repository's `.superpowers/` ignore rule.

## Non-blocking baseline note

`uv run black --check examples/stop_signal_demo.py tests/test_stop_signal_demo.py`
reports that `examples/stop_signal_demo.py` would be reformatted at the compact
signatures currently around lines 207 and 495. The base revision `7016713` fails
Black on the same two unchanged lines, and neither line is part of the Task 2
diff, so this is not a finding against the reviewed range. It remains a known
whole-branch formatting gate to resolve before the final Task 4 verification.
