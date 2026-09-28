# Scientific readability: array and result ownership

Implementation base: `82ac3ff`. Work is on `refactor/scientific-readability`.
Main and stash `c4cfbec5fccd5397ea650c550f5b485ea588c2cd` remain unchanged.
The user approved the ownership implementation plan before execution.

## Ordinary arrays

The new `readonly_array` function uses a single owned C-contiguous NumPy copy
and disables writes. It replaces the float subclass and the separate byte-buffer
helpers for bool masks and int64 indices. Values, shapes, and dtypes are retained.
Existing ownership tests now check base ndarray type and actual assignment
rejection, following the approved contract. They retain their scientific
assertions, original tolerances, and input/table mutation-isolation checks.

RED: `uv run --no-cache --no-sync pytest tests/test_arrays.py tests/test_data.py
tests/test_prepared.py -q -W error` produced 6 failures and 60 passes: the new
helper was missing and outputs still used the old subclass. Tests were committed
as `2924b2c` before implementation. GREEN: the same command passed all 66 tests.

The new helper tests cover strided inputs, float/int/bool conversion, owned
storage, output layout, ndarray subclasses, rejection of writes, and editable
copies. Migration documentation explains that deliberate write-flag changes can
invalidate an analysis/result and its provenance assumptions.

Full suite after this change: `uv run pytest -q -W error`: **786 passed** in
167.88 seconds. Black passed all 110 files in the CI scope; whitespace checks passed.

## Prepared-design access

RED: the multi-run fit/comparison regression failed at the public bulk-copy
accessor (1 failed, 51 deselected); tests committed as `00bf062`. The regression
blocks internal use of the public bulk design/role accessors, compares both
full and nuisance R² with the existing oracle, and verifies original tables
and roles are unchanged.

GREEN: `uv run --no-cache --no-sync pytest tests/test_prepared.py
tests/test_prepared_fit.py tests/test_fit.py tests/test_logging.py -q -W error`:
**121 passed** in 0.52 seconds. Formatting and whitespace checks passed.
Internal fitting reads the owned private tables; public accessors and result
construction still copy tables. No wrapper or new public accessor was added.

## Fractional result ownership

Pending implementation and verification.

## Final verification and independent review

Pending. Scientific result schema and metadata fields remain unchanged by this
increment. Remote CI and the locked Nilearn integer-image limitation remain
unverified/outstanding as recorded in foundation validation.

## Execution decisions

- Continued in the user-requested branch/current checkout with exact-path staging;
  this has less filesystem isolation than another worktree.
- Recorded successful test runs directly rather than repeating them solely through
  the skill's ledger script, following the instruction to avoid redundant testing.
- The changed write-flag assertion is an approved requirement change. Ordinary
  arrays are protected against accidental assignment, not deliberate flag changes.
  No numerical test was weakened or tolerance changed to accommodate this work.
