# `boldtailor._hrf_glm_design`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/_hrf_glm_design.py)

Compile conventional designs without changing semantic regressor names.

## Definitions

- [`compile_group_designs`](#definition-1)
- [`compile_group_design`](#definition-2)
- [`group_design_builder`](#definition-3)
- [`group_design_builder.build`](#definition-4)
- [`_task_model_design`](#definition-5)
- [`_custom_design`](#definition-6)
- [`design_identity`](#definition-7)
- [`_update_design_hash`](#definition-8)
- [`group_diagnostics`](#definition-9)

<a id="definition-1"></a>

## `compile_group_designs`

[Source, line 23](../../src/boldtailor/_hrf_glm_design.py#L23)

```python
compile_group_designs(data, model, selection)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:74](../../src/boldtailor/_hrf_glm.py#L74)

<a id="definition-2"></a>

## `compile_group_design`

[Source, line 43](../../src/boldtailor/_hrf_glm_design.py#L43)

```python
compile_group_design(events, times, nuisance, candidate, model, run)
```

One run's design for one candidate HRF, with semantic column names.

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:58](../../src/boldtailor/_hrf_glm_design.py#L58)
- [src/boldtailor/_hrf_glm_design.py:39](../../src/boldtailor/_hrf_glm_design.py#L39)

<a id="definition-3"></a>

## `group_design_builder`

[Source, line 50](../../src/boldtailor/_hrf_glm_design.py#L50)

```python
group_design_builder(data, model, selection, nuisance, keys)
```

Recompile any fitted (run, HRF) design on request; KeyError otherwise.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:143](../../src/boldtailor/_hrf_glm.py#L143)

<a id="definition-4"></a>

## `group_design_builder.build`

[Source, line 55](../../src/boldtailor/_hrf_glm_design.py#L55)

```python
group_design_builder.build(run, hrf_id)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `_task_model_design`

[Source, line 66](../../src/boldtailor/_hrf_glm_design.py#L66)

```python
_task_model_design(events, times, nuisance, candidate, model, run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:46](../../src/boldtailor/_hrf_glm_design.py#L46)

<a id="definition-6"></a>

## `_custom_design`

[Source, line 84](../../src/boldtailor/_hrf_glm_design.py#L84)

```python
_custom_design(events, times, nuisance, candidate, model, run)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:47](../../src/boldtailor/_hrf_glm_design.py#L47)

<a id="definition-7"></a>

## `design_identity`

[Source, line 94](../../src/boldtailor/_hrf_glm_design.py#L94)

```python
design_identity(groups, nuisance, frame_times)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:81](../../src/boldtailor/_hrf_glm.py#L81)

<a id="definition-8"></a>

## `_update_design_hash`

[Source, line 105](../../src/boldtailor/_hrf_glm_design.py#L105)

```python
_update_design_hash(digest, matrix)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm_design.py:98](../../src/boldtailor/_hrf_glm_design.py#L98)
- [src/boldtailor/_hrf_glm_design.py:100](../../src/boldtailor/_hrf_glm_design.py#L100)

<a id="definition-9"></a>

## `group_diagnostics`

[Source, line 110](../../src/boldtailor/_hrf_glm_design.py#L110)

```python
group_diagnostics(groups)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:142](../../src/boldtailor/_hrf_glm.py#L142)
