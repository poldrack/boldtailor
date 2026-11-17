# NSD process parallelism validation — 2026-09-26

> **Dated validation record:** Results and test counts apply to the recorded
> revision/settings, not necessarily this checkout. See the
> [documentation index](../../README.md) for current API and methods guidance.

Both canonical and expanded-HRF single-trial workflows now accept `n_jobs`
and CLI `--n-jobs`. The default remains 1. Multiple workers fit independent
grayordinate blocks, each using every required run and the same cross-validation
split. The parent merges blocks in spatial order and publishes only after
successful completion.

The implementation uses joblib's process backend, caps numerical-library
threads in workers to one, and dispatches at most one batch of `n_jobs` blocks
at a time. Worker caches contain timing/design information. Signals and fits
are computed per block. Source indices and spatial signatures retain their
global grayordinate positions; only temporary output arrays use local positions.
Jobs do not write derivative files. The `Execution` metadata records requested
and effective workers, block size, backend, and thread limit. The
[joblib configuration documentation](https://joblib.readthedocs.io/en/latest/generated/joblib.parallel_config.html)
describes the worker-thread control used here.

## Measurements

Fresh processes ran each worker count sequentially on the same first 16,384
grayordinates of sub-07/ses-nsd10. Each benchmark used all 12 runs, 750 trials,
649 HRFs, four blocks of 4096, OLS and ridge alpha 0.1. Process startup and cold
design caches were included. Final selected-vertex plot/design export and
transactional publication were outside the timing interval. No derivative
outputs were changed by these benchmarks.

| Workers | Seconds | Speedup | Sampled peak aggregate RSS, bytes |
| --- | ---: | ---: | ---: |
| 1 | 186.185 | 1.00× | 5,433,212,928 |
| 2 | 104.664 | 1.78× | 9,192,751,104 |
| 4 | 74.225 | 2.51× | 12,772,032,512 |

Hardware: 16 CPU cores, 137,438,953,472 bytes RAM (128 GiB). Memory monitoring
sampled the parent plus its process descendants every approximately 0.1 seconds.
The RSS sum may count shared pages multiple times and may miss short peaks.
These are fitting benchmarks for a subset, not measurements of a complete
parallel session including publication. Four workers are a measured starting
point, not an automatic setting or an optimum established for every dataset.

All saved benchmark arrays were compared across worker counts: selected HRF
IDs, selection and independent prediction scores, all 750 OLS/ridge trial beta
maps in the subset, full/confound/delta R², canonical baseline R² and odd-run
RT correlations. Two and four workers matched serial values exactly. Serial
benchmark arrays were also checked against the pre-existing derivative maps;
the largest discrepancy was `1.489e-8` from float32 map storage. Full details
are in [the numerical audit JSON](2026-09-26-nsd-parallel-audit.json).

## Regression tests and review

Eight new tests were committed failing before implementation (`5c7c74b`).
They exercise actual separate process IDs and concurrent execution, bounded
dispatch, deterministic result order, worker thread settings, canonical CLI
parity, expanded CIFTI/TSV/NPZ parity, a real worker fitting error without partial
publication, and invalid worker counts rejected before reading inputs.

One test-only correction (`97d76ed`) excluded the publisher's persistent private
lock when comparing CLI output files to the API's list of published artifacts.
This bookkeeping file exists in both modes but is absent from the API list.
All eight tests then passed. Other affected NSD tests passed, and Black checked
all 56 source/test/example files. Implementation is at `da59bfd`.

One fresh independent static review covered `634c2c3..da59bfd` and found no
Critical, Important, or supported Minor issue. It confirmed global/local index
handling, fixed CV folds, bounded dispatch, serialization and failure before
publication. The reviewer did not independently run the numerical benchmarks
or regression suite; those are executor-owned checks. Verdict: ready to merge
subject to final regression verification.

After local fast-forward integration, `uv run pytest tests examples/NSD -q
-W error --tb=short` passed **476 tests in 33.55 seconds**. Main-checkout Black
verification passed for all 56 files, the uv lockfile was current, and source
initializers remained empty. SHA256 checks confirmed the user's existing
stop-signal notebook and test edits were preserved. No remote push or
replacement of existing NSD derivative files occurred.
