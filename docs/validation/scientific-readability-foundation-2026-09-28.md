# Scientific readability foundation validation

Implementation base: `531dac3`; scientific baseline: `57acff5`.
Preserved work: stash `c4cfbec5fccd5397ea650c550f5b485ea588c2cd`, not applied.

## Notebook baseline

RED: `uv run pytest tests/test_stop_signal_demo.py -q -W error -k
'variance_partition_prepared_runtime or variance_partition_notebook_publishes'`
reported three failures and 44 deselected tests. The existing committed tests
failed because the fixture rewrote `go_success`/`go_failure` to `go`, while the
committed notebook models the original labels. The fixture and expected contrast
names were corrected to the notebook's existing scientific model; contrast
weights, trial grouping, numerical parity, and privacy checks were preserved.

The remaining interactive-view contract is retained: the committed tests require
three contrast views, one aggregate R² view, and one task ΔR² view.

After fixture correction: two runtime tests failed (`0` interactive views instead
of `5`), while the metadata test passed. Tests were committed as `bc3ee16` before
adding views. GREEN: all 47 tests in `tests/test_stop_signal_demo.py` passed
with warnings treated as errors. Black and whitespace checks also passed.
The complete baseline suite (`uv run pytest tests examples/NSD -q -W error`)
passed: **755 tests** in 164.97 seconds.

## Lazy version lookup

RED: four new tests failed (20 existing projection tests passed). The tests were
committed as `d30bc49` before implementation. They cover an uninstalled source
checkout, avoiding import-time lookups, missing dependency metadata, and both
stable/draft BIDS software-version output.

The affected suite exposed an existing test error: the no-filesystem-writes test
blocked all `Path.open` calls, including reads of installed metadata. Its existing
`io.open` guard already intercepts `Path.open` and rejects all write modes. The
blanket ban was removed, retaining write guards for builtins/io/os and Path writes.
GREEN: all 105 affected software, BIDS, prepared-data, and prepared-fit tests
passed with warnings treated as errors. The changed files passed Black.

## Packaging and collection

RED: the runtime-dependency test failed because installed metadata required
`ipykernel`; the isolated baseline-wheel check also failed because `ipykernel`
was installed. Both checks were committed as `7611c91` before packaging changes.
The new recursive initializer check replaces the old single-file check.

Default collection before the config change contained 621 core tests (including
new checks), with no NSD tests. Afterward it contains 760 tests, including all
140 NSD tests; the one-test difference is the consolidated initializer check.
The runtime-dependency tests now pass. The lockfile only removes the two runtime
references to ipykernel; all dependency versions are unchanged.

During synchronization, uv warned that the locked Nilearn 0.14.0 release is yanked
for casting cleaned signals to integers when extracting integer-valued images.
Changing scientific dependency versions is outside this foundation stage. Treat
integer-image use as an outstanding dependency issue, not as validated by these
synthetic floating-point regression checks.

## Final foundation checks

- `uv run pytest -q -W error`: **760 passed** in 170.70 seconds, including NSD.
- `uv run black --check src tests examples/NSD examples/stop_signal_demo.py`:
  all 107 Python files passed. Notebook JSON is checked by its execution tests;
  Black does not format notebooks in this environment.
- `uv build --wheel --quiet`: built the wheel successfully.
- `uv run --isolated --no-project --with ./dist/boldtailor-0.1.0-py3-none-any.whl
  python tests/check_installed_package.py`: exit 0, installed-package import,
  no ipykernel, and OLS coefficients matched independent NumPy least squares.
- `git diff --check`: passed.

GitHub Actions configuration is present but has not run remotely; this checkout
has no configured remote. The preserved stash is unchanged. No numerical
estimator or statistical objective was modified in this foundation stage.

## Independent review and execution decisions

An independent reviewer inspected `531dac3..4ffc5bd`, including surrounding
code and test history. Verdict: ready to merge, with no critical or important
findings. The branch remains available for the subsequent refactoring stages;
it has not been merged into main.

One minor test improvement is deferred: `test_imports_do_not_query_versions`
reloads the three consumer modules but not `_software` itself. Current helper
code has no eager lookup; including it in the guarded reload set would catch
a future eager dependency-version lookup added to that helper.

Execution decisions:

- Used the user-requested refactoring branch in the current checkout. This has
  less filesystem isolation than an additional worktree; exact-path staging
  kept existing user files out of commits.
- Recorded completed successful test runs directly instead of rerunning them
  solely through the skill's ledger script. This sacrifices automated ledger
  recording, not test evidence.
- Corrected the no-writes test to allow metadata reads through Path.open while
  retaining its io.open write guard. This enforces a no-writes contract rather
  than an all-I/O ban, consistent with the test's stated purpose.

The reviewer set aside three matters, resolved as follows:

- Broader architectural work, numerical/CV changes, and the preserved stash are
  separate stages. The foundation does not resolve those outstanding issues;
  the next numerical-stage plan is proposed for review.
- The yanked Nilearn release remains locked under the approved dependency freeze.
  Integer-image behavior requires a separate dependency/scientific validation
  change and remains a real limitation of this baseline.
- Remote CI could not be exercised without a configured remote. Local commands
  passed, but Linux-runner success is not claimed.
