# `boldtailor.workflow.plots`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/plots.py)

Presentation for the workflow; inputs are already fitted arrays and tables.

## Module contract

Presentation for the workflow; inputs are already fitted arrays and tables.

Functions return figures without displaying or saving them. They never refit a
model or change the inputs.

## Definitions

- [`design_figure`](#definition-1)
- [`library_figure`](#definition-2)
- [`glm_comparison`](#definition-3)
- [`_selected_parameters`](#definition-4)
- [`parameter_agreement`](#definition-5)
- [`_draw_curve_scatter`](#definition-6)
- [`curve_agreement`](#definition-7)
- [`activation_histogram`](#definition-8)
- [`_best_odd_vertex`](#definition-9)
- [`_plot_even_points`](#definition-10)
- [`rt_check_figure`](#definition-11)
- [`fraction_selection_figure`](#definition-12)

<a id="definition-1"></a>

## `design_figure`

[Source, line 19](../../src/boldtailor/workflow/plots.py#L19)

```python
design_figure(frame_times, design, regressors)
```

Plot named task regressors against original acquisition times.

<a id="definition-2"></a>

## `library_figure`

[Source, line 33](../../src/boldtailor/workflow/plots.py#L33)

```python
library_figure(library)
```

Plot every stored HRF, coloring noncanonical curves by time to peak.

<a id="definition-3"></a>

## `glm_comparison`

[Source, line 63](../../src/boldtailor/workflow/plots.py#L63)

```python
glm_comparison(glms)
```

Summarize finite R² values and plot optimized minus canonical full R².

<a id="definition-4"></a>

## `_selected_parameters`

[Source, line 92](../../src/boldtailor/workflow/plots.py#L92)

```python
_selected_parameters(library, ids, names)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/plots.py:102](../../src/boldtailor/workflow/plots.py#L102)
- [src/boldtailor/workflow/plots.py:103](../../src/boldtailor/workflow/plots.py#L103)

<a id="definition-5"></a>

## `parameter_agreement`

[Source, line 99](../../src/boldtailor/workflow/plots.py#L99)

```python
parameter_agreement(library, odd_ids, even_ids)
```

Compare selected parameters at grayordinates defined in both halves.

<a id="definition-6"></a>

## `_draw_curve_scatter`

[Source, line 120](../../src/boldtailor/workflow/plots.py#L120)

```python
_draw_curve_scatter(fig, axis, x, y, label, reference)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/plots.py:165](../../src/boldtailor/workflow/plots.py#L165)

<a id="definition-7"></a>

## `curve_agreement`

[Source, line 132](../../src/boldtailor/workflow/plots.py#L132)

```python
curve_agreement(curve_r)
```

Show three curve correlations on the same jointly finite grayordinates.

<a id="definition-8"></a>

## `activation_histogram`

[Source, line 171](../../src/boldtailor/workflow/plots.py#L171)

```python
activation_histogram(activation)
```

Overlay finite one-sample t histograms, one step histogram per model.

<a id="definition-9"></a>

## `_best_odd_vertex`

[Source, line 186](../../src/boldtailor/workflow/plots.py#L186)

```python
_best_odd_vertex(model, brain)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/plots.py:214](../../src/boldtailor/workflow/plots.py#L214)

<a id="definition-10"></a>

## `_plot_even_points`

[Source, line 194](../../src/boldtailor/workflow/plots.py#L194)

```python
_plot_even_points(ax, model, runs, vertex)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/plots.py:220](../../src/boldtailor/workflow/plots.py#L220)

<a id="definition-11"></a>

## `rt_check_figure`

[Source, line 206](../../src/boldtailor/workflow/plots.py#L206)

```python
rt_check_figure(beta_models, runs, brain)
```

Even-run (RT, beta) points at the odd-run vertex with the largest |r|.

Return None when the models have no response times.

<a id="definition-12"></a>

## `fraction_selection_figure`

[Source, line 227](../../src/boldtailor/workflow/plots.py#L227)

```python
fraction_selection_figure(table)
```

Bar panels of grayordinates choosing each ridge fraction, by scope.
