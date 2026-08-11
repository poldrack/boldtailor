# Shared Map Slices Design

## Goal

Display the aggregate R-squared and task-attributable delta R-squared maps on
the same axial slices as the three contrast maps in the stop-signal notebook.

## Design

The notebook results cell will define one shared display mode and one shared
set of cut coordinates:

```python
DISPLAY_MODE = "z"
CUT_COORDS = np.arange(-20, 60, 15)
```

All five `plot_stat_map` calls will pass these values explicitly. Contrast,
aggregate R-squared, and delta R-squared plots will therefore remain aligned if
the slice selection changes later. Existing thresholds, color maps, colorbar
settings, titles, and statistical values remain unchanged.

## Testing

The existing notebook plot instrumentation will record `display_mode` and
`cut_coords`. A runtime test will require all five calls to use `"z"` and the
exact coordinate sequence `[-20, -5, 10, 25, 40, 55]`. The test must fail
against the current notebook because both R-squared calls omit these settings.

The focused notebook tests and full warning-strict suite will run after the
change. Notebook outputs must be preserved; the change does not require an
output-free notebook.

## Scope

Only `tests/test_stop_signal_demo.py` and the results-cell source in
`examples/stop_signal_demo.ipynb` change during implementation. Existing
uncommitted notebook outputs, `pyproject.toml`, and `uv.lock` edits remain
outside the feature commits.
