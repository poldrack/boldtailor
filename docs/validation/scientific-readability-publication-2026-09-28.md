# Scientific readability: local publication

> **Dated validation record:** Results apply to the recorded revision/settings.
> See the [documentation index](../README.md) for current API guidance.

Base `a9f659b`, implementation through `59e5917` on
`refactor/scientific-readability`. Main remains `57acff5`; preserved stash
`c4cfbec5fccd5397ea650c550f5b485ea588c2cd` and unrelated files are untouched.

## Changes and guarantees

Publication now uses ordinary paths, one FileLock, staging on the destination
filesystem, per-file replacement, and explicit rollback. Descriptor anchoring,
platform probing, recovery copying, restore fallback, and failed-artifact
regeneration were removed. Existing path/metadata validation, source protection,
case-folded collisions, fsync, input-order return values, and scientific artifact
bytes are retained. Destination aliases cannot bypass source checks, and lock
timeouts must be finite.

`retain_incomplete` was removed. Failed rollback preserves the original
transaction in place; `PublicationError.recovery_directory` reports its location
and `rollback_errors` exposes the recovery exceptions. The original operation
exception remains `__cause__`. New ledger records use fixed error categories,
not exception text/class names. See [the migration guide](../publication-migration.md).

The scope is local cooperating writers. Individual replacements do not provide
an atomic set, crash recovery, consistent concurrent-reader snapshots, or
protection against active directory swaps. Static symlinks remain rejected.

## Test-first evidence

| Task | Observed RED | Test commits | GREEN |
| --- | --- | --- | --- |
| Validation | 5 failed, 42 passed | `eab6463` | 47 publication tests passed in 2.33 s |
| Ordinary paths/recovery | 8 failed, 43 passed; one additional collision failure | `0be94d1`, `dbd0387` | Publication + NSD trial/HRF exports: 68 passed in 10.13 s |
| Failure categories | 2 failed, 51 passed | `4d9e8c3` | 53 publication tests passed in 1.83 s |

Existing retention/recovery-copy/fallback and active-directory-swap tests were
replaced or removed only for explicitly approved contract changes. Static
symlink, source-read-only, every replacement-boundary failure, concurrency, and
NSD integration tests remain. No scientific oracle or tolerance changed.

## Verification and review

- Full suite before review fixes: **869 passed** in **161.30 seconds**.
- Final `uv run pytest -q -W error`: **871 passed** in **162.33 seconds**,
  including both protected-control-source regressions.
- Black: 115 Python files passed (`src tests examples/NSD examples/stop_signal_demo.py`).
- `git diff --check`: passed.
- `uv build --wheel --quiet`: passed.
- `uv run --isolated --no-project --with ./dist/boldtailor-0.1.0-py3-none-any.whl
  python tests/check_installed_package.py`: passed; installed-package location,
  dependency isolation, and independent OLS smoke check verified.
- Documentation: 178 local links/anchors passed across 66 README/docs/example
  Markdown and notebook files. Both README examples and the API publication
  example executed successfully in temporary directories.
- Source/test package initializers remain empty.

## Independent review

The reviewer identified a source-protection gap: the lock or diagnostic ledger
could modify a supplied input within `.boldtailor`, because preflight checked
only artifact targets. Two regression tests failed and were committed as
`9a4259c`. Fix `59e5917` includes the control directory in the existing source
overlap check, before any lock or diagnostic write. All **55 publication tests**
passed afterward, followed by the complete 871-test suite. The independent
review found no other Important/Critical issues and no Minor issues. It judged
the increment ready based on the reported fix and verification. It did not
reinspect the fix or rerun tests, consistent with the agreed single-review
workflow; the executor verified the fix through RED–GREEN and the full suite.

## Execution decisions

- Continued in the existing authorized refactor checkout; it provides less
  filesystem isolation than a separate worktree.
- Recorded observed test commands directly rather than rerunning them solely
  through the skill script; evidence is preserved here.
- Added a `created` flag to transaction state: cleanup owns only directories
  this attempt created. Without it, a UUID collision could delete an existing
  transaction's recovery data. A committed failing regression covers this case.

- The review set aside active directory swaps and non-cooperating writers.
  The approved local-writer contract stands; ordinary path checks do not
  defend against those races.
- Crash recovery, power-loss durability, and whole-set atomic visibility remain
  explicitly excluded. Interrupted publication or concurrent readers can see
  partial output; no stronger guarantee is claimed.
- Scientific estimators were unchanged and remain covered by the full suite;
  this publication review did not repeat the previous scientific-code reviews.
- Final fix verification was performed by the executor, not a second reviewer.
  Committed failing regressions, targeted GREEN, and the complete suite provide
  the evidence; there is no independent second inspection of the final fix.

No minor findings were deferred. Remote CI was not run. The previously recorded
locked-Nilearn integer-image limitation remains unchanged.

The branch stays unmerged. Packaged imaging, notebook simplification, and final
whole-branch review remain in the architectural roadmap.
