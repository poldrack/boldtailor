# Continuous HRF autodiff benchmark

The optional `gradient="autodiff"` setting supplies JAX gradients to L-BFGS-B. The default remains `finite_difference`. This differentiates the existing SPM-style double-gamma family, preserving fixed support, event sampling, normalization, nuisance projection, and profiled least-squares objective. Linear coefficients are solved exactly and held constant in differentiation using the variable-projection envelope identity. Peak normalization remains piecewise differentiable; rank transitions also require care.

## Matched real-data comparison

Session 11, eight evenly spaced anatomical VTC vertices, six training and six held-out runs (seed 20261107), full 533-member initialization library, four starts, fixed lambda 0.01, maxiter 500, maxfun 5000, one worker. Both optimizers ran in the same environment. Timing includes fitting and compilation but excludes held-out scoring. This is a fixed-regularization benchmark, not a benchmark of the full nested tuning workflow.

| RT model | Finite differences | Autodiff, cold | Autodiff, warm | Cold speedup | Warm speedup |
|---|---:|---:|---:|---:|---:|
| Amplitude | 11.656 s | 3.052 s | 1.952 s | 3.82x | 5.97x |
| Duration | 13.601 s | 3.487 s | 2.504 s | 3.90x | 5.43x |

Function evaluations fell from 4,230 to 706 for amplitude and from 3,804 to 636 for duration. All 48 vertex/backend/mode score records were valid. Maximum absolute objective differences were 1.18e-11 (amplitude) and 1.35e-12 (duration); maximum held-out partial-R2 differences were 7.38e-8 and 2.12e-7 respectively. Warm repeats reuse compilation for the same static design; each worker has its own cache, so multi-worker compilation costs need separate measurement.

Per-vertex scores and timings are saved in `reports/autodiff_benchmark_session11.tsv`. Raw fits, diagnostics, exact feature IDs, run split, library fingerprint, and environment are in `outputs/autodiff-benchmark-session11/`. Implementation commit: cda1183.

## Environment and usage

The benchmark used CPU JAX/jaxlib 0.12.0.1, NumPy 2.2.6, SciPy 1.16.3, and 64-bit JAX arithmetic, in an isolated uv overlay. No base environment or running analysis was changed. Use `uv run --no-sync --with jax==0.12.0.1 python ...` and pass `gradient="autodiff"` in continuous optimizer settings. This requires the prepared design and L-BFGS-B backend. Limit BLAS threads when using worker processes.

Tests compare designs and objectives with the original implementation, gradients with central finite differences in both RT modes, nuisance overlap/rank loss, cache reuse across signals, and planted-HRF recovery with parallel fitting. Independent review found no important correctness problems. These results support a larger matched nested-tuning pilot before replacing existing full-VTC jobs.
