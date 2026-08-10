# Task 1 Report: Nuisance-Only Design Compilation

- Base: `6fc4cc789719f41616e17c23ecf778dcd84382ae`
- Test commit: `992038a` (`test: specify nuisance-only design compilation`)
- Production commit: `6c2299b` (`feat: compile nuisance-only designs`)
- Formatting-only commit: `fac0498` (`style: format nuisance design tests`)
- Head: `fac0498`

## RED

Command:

```text
uv run pytest tests/test_design.py -k nuisance -q
```

Result: collection failed because `compile_nuisance_designs` was absent from
`boldtailor.design`, as required.

## GREEN

Commands and results:

- `uv run --no-cache --no-sync pytest tests/test_design.py -q -W error` — 10 passed in 0.87s.
- Full warning-strict suite, rerun by the controller with permitted local
  Jupyter ports — 223 passed in 12.60s.
- `uv run --no-cache --no-sync black --check src tests` — 23 files unchanged
  after the formatting-only test commit.

## Files

- `tests/test_design.py`: nuisance-only behavior, Nilearn parity, and selected
  confound validation parameterized across both compilers.
- `src/boldtailor/design.py`: `compile_nuisance_designs`, nuisance-run
  compilation, nuisance matrix construction, and shared design validation.
- This report.

## Coverage

The focused design suite covers nuisance event exclusion, selected confound
retention/filtering, cosine and no-drift designs, Nilearn parity, cutoff and
metadata values, missing/nonfinite confound errors for both compilers, and the
existing event-design behavior. The controller full suite passed all 223 tests.
No separate percentage coverage report was requested or generated.

## Concerns

Initial Git writes from this linked worktree were blocked by the sandbox’s
read-only main `.git` metadata and `uv` runtime/cache failures; the required
commits were subsequently recorded by the controller as listed above. The
controller’s full-suite run used permitted localhost Jupyter ports because the
sandbox-only run otherwise failed three notebook tests before execution on
socket binding.
