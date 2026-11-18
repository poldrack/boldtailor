# Scientific readability: fit lifecycle, logging, and provenance

> **Dated validation record:** Results apply to the recorded revision/settings.
> See the [documentation index](../README.md) for current API guidance.

Implementation base: `3188f7f`; implementation through `c62dff9` on
`refactor/scientific-readability`. The user approved the concrete plan and
inline execution. Main remains at `57acff5`; stash
`c4cfbec5fccd5397ea650c550f5b485ea588c2cd` and unrelated untracked files are
preserved.

## Changes

Eight fit/comparison paths now share a small context manager for execution
IDs, bounded history, and success/failure events. Statistical operations remain
explicit in the entry points. Early validation, provenance, and final result
construction are covered by the failure boundary. Completion is included in
returned provenance but emitted only after construction succeeds. Selected-HRF
trial fits now emit lifecycle events; start events omit analysis identity.

Failure logs use fixed `error_code` categories without formatting exceptions,
examining environment values, or exporting exception class names. Callers
receive the original exceptions. Nested fits and normalization have fresh ID
scopes and restore their caller's context. The prepared-fit sanitizer and
parallel lifecycle wrappers were removed.

Provenance extension directly constructs a validated record rather than
serializing and reparsing its parent. Normalization builds one final record,
replacing three constructions. Each record computes its source fingerprint
once for identity and quality-warning decisions. Scientific activity payloads,
identity formulas, ownership, warning order, and numerical algorithms remain
unchanged. Publication and arbitrary-annotation validation are outside this
increment. See the [migration guide](../lifecycle-migration.md).

## RED–GREEN evidence

| Task | RED | Test commit | GREEN |
| --- | --- | --- | --- |
| Error categories | 14 failed, 54 passed | `1860a3b` | Logging/prepared-fit: 68 passed |
| Shared lifecycle | 8 missing-module failures | `83d3ed8` | Lifecycle/logging: 24 passed |
| Fit migration | 26 failed, 164 passed | `4c5df05` | Fit, trial, ridge/fractional CV suites: 254 passed in 36.84 s |
| Provenance/normalization | 11 failed, 86 passed | `138ad60` | Provenance/data/prepared/logging/prepared-fit/BIDS: 192 passed |

Tests inject late constructor/provenance failures, cover early validation,
check exception identity, nested context restoration, completion/provenance
parity, source fingerprint equivalence, and owned extension metadata. Existing
scientific oracles and tolerances were not weakened.

## Final verification

- Before the review fix: `uv run pytest -q -W error` — **852 passed** in
  163.51 seconds, including NSD and notebook tests.
- Final suite including the four review regressions: **856 passed** in
  **162.39 seconds**, with warnings treated as errors.
- `uv run --no-cache --no-sync black --check src tests examples/NSD
  examples/stop_signal_demo.py` — **115 Python files** passed.
- `git diff --check` — passed.
- `uv build --wheel --quiet` — passed after the review fix.
- `uv run --isolated --no-project --with ./dist/boldtailor-0.1.0-py3-none-any.whl
  python tests/check_installed_package.py` — exit 0 on the rebuilt wheel;
  installed-package location, no ipykernel dependency, and independent OLS
  comparison passed.
- Documentation validation — **171 local links/anchors across 63 Markdown and
  notebook files** under README/docs/examples; both README examples and the API
  publication example executed successfully in a temporary directory.
- All source/test package initializers are empty.

## Independent review

The fresh reviewer inspected `3188f7f..c47bb1b`, surrounding callers, tests,
and current documentation. It found no numerical changes and no Critical or
Minor findings. One Important regression was reproduced: converting a log
level's display name back to an integer broke unnamed numeric levels such as
15. Tests committed at `1fc8ae7` failed in all four cases; `c62dff9` passes the
original numeric severity to the emitter. The lifecycle/logging suites then
passed **28 tests**. The final full suite includes these tests.

No second review was requested: the agreed process verifies Important fixes
with RED–GREEN tests and a final full-suite run.

## Execution decisions

- Continued in the existing refactor checkout under the user's authorization;
  this provides less filesystem isolation than another worktree.
- Recorded observed passing targeted commands directly rather than rerunning
  them solely through the skill ledger script. Verification evidence is kept
  here; unnecessary duplicate runs were avoided.
- Checked numerical exception types before `ValueError`, correcting the plan's
  sample implementation: NumPy `LinAlgError` inherits `ValueError`. Reversing
  that order would mislabel numerical failures as invalid inputs.

- Publication and imaging remain deferred under the approved roadmap; the
  whole architectural refactor is therefore not yet complete.
- Arbitrary annotations and manually populated private `_extra` fields retain
  existing validation. A broader schema redesign was excluded; caller-supplied
  metadata is not covered by a blanket privacy guarantee.
- Application logging-handler exceptions and `BaseException` interruptions
  retain normal Python behavior. They can interrupt execution without a terminal
  lifecycle event; they are not converted into ordinary fit failures.

The reviewer explicitly set aside the three scope/exception-policy topics
above; these are the executor's decisions on those items. No minor findings
were deferred. Remote CI was not run. The previously documented locked-Nilearn
integer-image limitation remains unchanged.

The branch remains unmerged. Publication simplification, packaged imaging and
notebook cleanup, and a final whole-branch review remain in the larger roadmap.
