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
