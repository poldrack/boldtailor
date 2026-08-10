# Task 2 Report: Delta-R-Squared Result, Fit, Logging, and Provenance

- Base: `1a8b6f47b23b83ca91b5ecac9efb1a439302177f`
- Test commit: `db38bcd` (`test: specify task delta r-squared comparison`)
- Production commit: `447be31` (`feat: compare task and nuisance model r-squared`)
- Production head: `447be31`

## RED

Command:

```text
uv run --no-cache --no-sync pytest tests/test_fit.py tests/test_logging.py -k delta_r2 -q
```

Result: collection failed in both modules because `task_delta_r2` was absent
from `boldtailor.fit`, exactly matching the planned missing-interface RED. The
test-only state was committed before either production file was changed.

## GREEN

Commands and results:

- `uv run --no-cache --no-sync pytest tests/test_fit.py tests/test_logging.py -k delta_r2 -q -W error` — 13 passed, 21 deselected in 0.87s.
- `uv run --no-cache --no-sync pytest tests/test_fit.py tests/test_logging.py -q -W error` — 34 passed in 1.04s.
- `uv run --no-cache --no-sync pytest -q -W error` — the pre-commit run passed
  236 tests in 12.10s, including notebook tests. A final post-commit rerun passed
  233 tests and hit the sandbox-only notebook startup issue described below.
- `uv run --no-cache --no-sync pytest -q -W error -k "not notebook"` — final
  post-commit verification passed 226 tests with 10 notebook-related tests
  deselected in 3.15s.
- `uv run --no-cache --no-sync black --check src tests` — 23 files unchanged.
- `git diff --check` — clean.
- `find src -name '__init__.py' -size +0 -print` — no output; all package initializers remain byte-empty.

## Files

- `tests/test_fit.py`: sourced 100-scan AR(1) comparison fixture; raw/clipped
  calculations; immutable owned values; defensive designs; factory validation;
  mismatch, anonymous-provenance, and provenance checks.
- `tests/test_logging.py`: correlated success lifecycle and failed-operation
  context-reset checks.
- `src/boldtailor/results.py`: frozen `TaskDeltaR2Result`, validated result
  factory, immutable arrays, exact diagnostics, and copied nuisance designs.
- `src/boldtailor/fit.py`: shared GLM internals, pooled nuisance R-squared fit,
  parent matching and dimension validation, deterministic comparison identity,
  structured logs, and comparison provenance.
- This report.

## Coverage

The 13 focused cases cover complete-minus-nuisance calculation, zero clipping,
negative-count and raw-min diagnostics, float64 shape and strict immutability,
input ownership, defensive design access, all required factory input failures,
changed data/model rejection, anonymous provenance rejection, nuisance model
and run provenance, success/failure structured logs, comparison-ID correlation,
and context reset. The complete warning-strict suite passed all 236 tests in
the pre-commit run, and the final non-notebook suite passed all 226 selected
tests. No separate percentage coverage report was requested or generated.

## Deviations

No behavioral or file-scope deviations. Verification commands used the
requested `--no-cache --no-sync` form. On the final post-commit full-suite run,
three notebook cases failed before kernel startup because the sandboxed
`/Users/poldrack/.ipython` directory was not writable and `-W error` promoted
IPython's fallback warning. Tests and notebook helpers were not changed or
suppressed; the requested non-notebook verification was run instead.

## Concerns

Pre-existing untracked generated directories (`examples/__pycache__`,
`src/boldtailor.egg-info`, `src/boldtailor/__pycache__`, and
`tests/__pycache__`) remain untouched as instructed. No implementation concern
or provenance-semantics conflict was found. The final notebook-only failure is
environmental and occurred before notebook execution.
