# `boldtailor.cifti`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/cifti.py)

CIFTI-2 dense scalar export, checked reloads, and grayordinate identity.

## Module contract

CIFTI-2 dense scalar export, checked reloads, and grayordinate identity.

Grayordinate order always follows the input BrainModelAxis; undefined values
stay NaN at their original positions.

## Definitions

- [`scalar_artifact`](#definition-1)
- [`read_scalar`](#definition-2)
- [`cortical_values`](#definition-3)
- [`spatial_signature`](#definition-4)

<a id="definition-1"></a>

## `scalar_artifact`

[Source, line 19](../../src/boldtailor/cifti.py#L19)

```python
scalar_artifact(path, brain, values, names)
```

A float32 ``.dscalar.nii`` artifact with one named map per row.

Usage in package/tests:

- [src/boldtailor/workflow/artifacts.py:61](../../src/boldtailor/workflow/artifacts.py#L61)
- [src/boldtailor/workflow/artifacts.py:67](../../src/boldtailor/workflow/artifacts.py#L67)
- [src/boldtailor/workflow/outputs.py:278](../../src/boldtailor/workflow/outputs.py#L278)
- [src/boldtailor/workflow/outputs.py:591](../../src/boldtailor/workflow/outputs.py#L591)

<a id="definition-2"></a>

## `read_scalar`

[Source, line 29](../../src/boldtailor/cifti.py#L29)

```python
read_scalar(path, brain, names=None)
```

Load saved maps after checking the grayordinate axis and, optionally, names.

<a id="definition-3"></a>

## `cortical_values`

[Source, line 41](../../src/boldtailor/cifti.py#L41)

```python
cortical_values(values, brain)
```

Scatter a grayordinate vector onto each cortex, leaving absent vertices NaN.

Usage in package/tests:

- [src/boldtailor/workflow/surfaces.py:148](../../src/boldtailor/workflow/surfaces.py#L148)

<a id="definition-4"></a>

## `spatial_signature`

[Source, line 58](../../src/boldtailor/cifti.py#L58)

```python
spatial_signature(brain, indices)
```

SHA-256 of the ordered BrainModel axis XML and the selected indices.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:24](../../src/boldtailor/workflow/analysis.py#L24)
- [src/boldtailor/workflow/beta_series.py:90](../../src/boldtailor/workflow/beta_series.py#L90)
