# `boldtailor.workflow.surfaces`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/surfaces.py)

Offline cortical views of existing grayordinate statistics.

## Definitions

- [`find_surface_meshes`](#definition-1)
- [`_load_meshes`](#definition-2)
- [`_color_scale`](#definition-3)
- [`_label`](#definition-4)
- [`_draw_surface_panel`](#definition-5)
- [`surface_figure`](#definition-6)

<a id="definition-1"></a>

## `find_surface_meshes`

[Source, line 23](../../src/boldtailor/workflow/surfaces.py#L23)

```python
find_surface_meshes(fmriprep_dir, subject, *, paths=None)
```

Find subject fsLR 32k midthickness meshes, or use explicit left/right paths.

Return None if either hemisphere is unavailable. Multiple matches require
explicit paths, preventing an arbitrary choice across anatomical sessions.
Explicit meshes must use the same fsLR vertex ordering as the CIFTI data.

<a id="definition-2"></a>

## `_load_meshes`

[Source, line 53](../../src/boldtailor/workflow/surfaces.py#L53)

```python
_load_meshes(paths, brain)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/surfaces.py:149](../../src/boldtailor/workflow/surfaces.py#L149)

<a id="definition-3"></a>

## `_color_scale`

[Source, line 66](../../src/boldtailor/workflow/surfaces.py#L66)

```python
_color_scale(projected, statistic)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/surfaces.py:150](../../src/boldtailor/workflow/surfaces.py#L150)

<a id="definition-4"></a>

## `_label`

[Source, line 101](../../src/boldtailor/workflow/surfaces.py#L101)

```python
_label(name)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/surfaces.py:173](../../src/boldtailor/workflow/surfaces.py#L173)

<a id="definition-5"></a>

## `_draw_surface_panel`

[Source, line 116](../../src/boldtailor/workflow/surfaces.py#L116)

```python
_draw_surface_panel(axis, mesh, values, hemi, view, limits, cmap)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/surfaces.py:167](../../src/boldtailor/workflow/surfaces.py#L167)

<a id="definition-6"></a>

## `surface_figure`

[Source, line 136](../../src/boldtailor/workflow/surfaces.py#L136)

```python
surface_figure(maps, brain, meshes, *, statistic, title=None)
```

Compare maps in four cortical views with a shared scale and no threshold.

R² and ΔR² span 0–1 (extended below zero when needed). ΔR² uses a square-root
heat-color progression to emphasize small values, with ticks in original units.
RT uses a symmetric range covering all finite cortical values; t uses ±10.
Peak time uses a sequential scale spanning finite cortical values in seconds.
Volumetric structures
never affect the plots or color limits. Input maps and meshes are unchanged.
