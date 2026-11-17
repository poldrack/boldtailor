# `boldtailor.parallel`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/parallel.py)

Bounded process batches; workers compute and the parent assembles outputs.

## Module contract

Bounded process batches; workers compute and the parent assembles outputs.

Result objects that hold rebuild closures (``SingleTrialResult`` with a
``SelectedTrialDesign``, ``HrfAnalysisResult``) cannot be serialized with the
standard ``pickle`` module; loky's cloudpickle can move them between
processes, but workers should prefer returning arrays or dicts to keep
transfers small.

## Definitions

- [`validate_n_jobs`](#definition-1)
- [`_serial`](#definition-2)
- [`map_blocks`](#definition-3)

<a id="definition-1"></a>

## `validate_n_jobs`

[Source, line 16](../../src/boldtailor/parallel.py#L16)

```python
validate_n_jobs(n_jobs)
```

Return a positive integer worker count; booleans are rejected.

Usage in package/tests:

- [src/boldtailor/parallel.py:34](../../src/boldtailor/parallel.py#L34)
- [src/boldtailor/workflow/beta_series.py:50](../../src/boldtailor/workflow/beta_series.py#L50)
- [tests/test_parallel.py:50](../../tests/test_parallel.py#L50)

<a id="definition-2"></a>

## `_serial`

[Source, line 23](../../src/boldtailor/parallel.py#L23)

```python
_serial(function, blocks, args)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/parallel.py:38](../../src/boldtailor/parallel.py#L38)

<a id="definition-3"></a>

## `map_blocks`

[Source, line 28](../../src/boldtailor/parallel.py#L28)

```python
map_blocks(function, blocks, *, args=(), n_jobs=1)
```

Yield ordered ``(block, result)`` pairs, dispatching at most n_jobs blocks.

Several workers run in loky processes with one inner BLAS/OpenMP thread;
each batch is released before the next one is dispatched.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:57](../../src/boldtailor/workflow/analysis.py#L57)
- [src/boldtailor/workflow/analysis.py:138](../../src/boldtailor/workflow/analysis.py#L138)
- [src/boldtailor/workflow/analysis.py:237](../../src/boldtailor/workflow/analysis.py#L237)
- [src/boldtailor/workflow/beta_series.py:134](../../src/boldtailor/workflow/beta_series.py#L134)
