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
