# Array ownership migration

Boldtailor now returns ordinary NumPy arrays instead of a float-array subclass
or byte-buffer-backed masks and indices. All are owned copies marked read-only.
Float64, int64, and bool dtypes, numerical values, shapes, and result field names
are unchanged. Public table and dictionary accessors still return defensive
copies; nested event metadata remains isolated.

Ordinary assignments still raise `ValueError`. Read-only flags protect against
accidental mutation, not deliberate calls to `setflags(write=True)`. Code that
expected that call itself to fail should use the documented ownership convention
instead. Changing an exposed array deliberately can invalidate its owning
analysis/result and provenance assumptions.

```python
# Read without another array copy.
betas = result.run_betas[0]

# Own an editable array for subsequent calculations.
editable_betas = betas.copy()
editable_betas[:, 0] = 0
```

The internal `_arrays.immutable_float_array`, `hrf_results.immutable_indices`,
and `ridge_results.immutable_bool_array` helpers have been replaced by
`_arrays.readonly_array(values, dtype=...)`. These were implementation helpers;
normal users only need to copy returned arrays before editing them.
