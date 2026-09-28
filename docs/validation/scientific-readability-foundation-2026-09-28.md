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
