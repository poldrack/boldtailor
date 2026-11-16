# Fit lifecycle and logging migration

New failure events export `error_code` instead of exception text in `error`:

| Exception | Code |
| --- | --- |
| ValueError, TypeError | invalid_input |
| ArithmeticError, NumPy LinAlgError | numerical_failure |
| OSError | io_failure |
| Other exceptions or legacy string errors | operation_failed |

Callers still receive the original detailed exception. Logging does not inspect
exception text, environment values, object representations, or tracebacks.
Previously saved provenance records are unchanged. Caller-supplied event names,
identifiers, and annotations remain the caller's responsibility; categorical
errors do not make arbitrary metadata safe to share.

Each fit or task comparison emits one start followed by completion or failure.
Validation, provenance, and result assembly are inside that boundary. Completion
is emitted only after the returned object has been constructed successfully.
The logging context no longer requires a provenance-building method to be called:
normal exit alone is success. `FitOperation.provenance()` has been replaced by
`build_fit_provenance(operation, activity, warnings=...)`; callers set the optional
`operation.analysis_id` explicitly. The builder does not modify the operation,
emit logs, or control whether the scope succeeds.

Start records omit `analysis_id`. Completion records contain it when the inputs
support a metadata fingerprint; failure records contain it if established before
the error. These fingerprints identify recorded metadata, not signal contents.
Execution IDs distinguish attempts. Nested operations use fresh contexts and
restore the enclosing context afterward. Provenance retains the last eight events.

| Operation | Event prefix |
| --- | --- |
| Conventional, prepared, and selected-HRF conventional fit | `fit` |
| Conventional and selected-HRF task comparison | `task_delta_r2` |
| Prepared task comparison | `task_delta_r2_prepared` |
| Shared-HRF trial fit | `single_trial` |
| Selected-HRF trial fit | `selected_hrf_single_trial` |
| HRF selection | `hrf_selection` |
| Held-out HRF evaluation | `hrf_independent_evaluation` |
| Denoising selection and augmentation | `denoising_selection`, `denoising_augmentation` |

The selected-HRF trial events are new. Each prefix has `_started`, `_completed`,
and `_failed` variants. HRF selection and cross-validation emit only these
operation-level events, not per-candidate progress events.

The provenance completion snapshot is created when provenance is assembled; the
completion log is created independently at successful scope exit. They share
operation identities and event names, but their timestamps and sequence numbers
are no longer identical. Compare execution/analysis IDs rather than requiring
whole-event equality. A later result-construction error still emits failure,
never completion, and no result is returned. Snapshot construction may leave
sequence gaps; sequences need not be contiguous. Publication behavior and
previously saved provenance records are unchanged.
