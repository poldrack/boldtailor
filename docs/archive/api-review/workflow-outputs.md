# `boldtailor.workflow.outputs`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/outputs.py)

Publish every workflow stage's maps, designs, tables, and provenance together.

## Definitions

- [`_map`](#definition-1)
- [`_map_path`](#definition-2)
- [`_run_path`](#definition-3)
- [`_title`](#definition-4)
- [`_session_files`](#definition-5)
- [`check_output`](#definition-6)
- [`_names`](#definition-7)
- [`metadata_name`](#definition-8)
- [`_listed_artifacts`](#definition-9)
- [`_inside`](#definition-10)
- [`_remove_stale`](#definition-11)
- [`_design_artifact`](#definition-12)
- [`_fit_artifacts`](#definition-13)
- [`glm_artifacts`](#definition-14)
- [`_library_artifacts`](#definition-15)
- [`_selection_artifacts`](#definition-16)
- [`_split_artifacts`](#definition-17)
- [`_all_selection_maps`](#definition-18)
- [`hrf_artifacts`](#definition-19)
- [`_selection_records`](#definition-20)
- [`_scope_record`](#definition-21)
- [`_beta_maps`](#definition-22)
- [`_rt_map`](#definition-23)
- [`fraction_fit_artifacts`](#definition-24)
- [`beta_model_artifacts`](#definition-25)
- [`_tuned_artifacts`](#definition-26)
- [`_predictor_table`](#definition-27)
- [`_tuned`](#definition-28)
- [`tuning_rows`](#definition-29)
- [`_alpha_rows`](#definition-30)
- [`tuning_table`](#definition-31)
- [`plot_tuning`](#definition-32)
- [`_plot_alpha_tuning`](#definition-33)
- [`tuning_figure`](#definition-34)
- [`_tuning_maps`](#definition-35)
- [`_tuning_files`](#definition-36)
- [`_shared_tuning_metadata`](#definition-37)
- [`_alpha_tuning_metadata`](#definition-38)
- [`_fraction_tuning_metadata`](#definition-39)
- [`_fraction_selection_maps`](#definition-40)
- [`_tuning_artifacts`](#definition-41)
- [`_prediction_artifacts`](#definition-42)
- [`_outer_maps`](#definition-43)
- [`_outer_loss`](#definition-44)
- [`_fraction_outer_metadata`](#definition-45)
- [`_outer_metadata`](#definition-46)
- [`_outer_artifacts`](#definition-47)
- [`_boundary_fraction`](#definition-48)
- [`ridge_boundary_summary`](#definition-49)
- [`_scope_selection`](#definition-50)
- [`_pooled_bound_rows`](#definition-51)
- [`hrf_boundary_summary`](#definition-52)
- [`hrf_boundary_table`](#definition-53)
- [`activation_artifacts`](#definition-54)
- [`input_artifacts`](#definition-55)
- [`_ridge_metadata`](#definition-56)
- [`_activation_metadata`](#definition-57)
- [`_selection_description`](#definition-58)
- [`_indicator_clause`](#definition-59)
- [`_rt_check_description`](#definition-60)
- [`_rt_check`](#definition-61)
- [`_task_model_description`](#definition-62)
- [`_categorical_prose`](#definition-63)
- [`_response_time_prose`](#definition-64)
- [`_model_prose`](#definition-65)
- [`_curve_correlation_metadata`](#definition-66)
- [`_design_metadata`](#definition-67)
- [`_json_bound_rows`](#definition-68)
- [`_library_metadata`](#definition-69)
- [`metadata`](#definition-70)
- [`report_name`](#definition-71)
- [`_stage_artifacts`](#definition-72)
- [`figure_name`](#definition-73)
- [`_figure_artifacts`](#definition-74)
- [`workflow_artifacts`](#definition-75)
- [`_root_artifacts`](#definition-76)
- [`publish_workflow`](#definition-77)
- [`save_workflow`](#definition-78)

<a id="definition-1"></a>

## `_map`

[Source, line 38](../../src/boldtailor/workflow/outputs.py#L38)

```python
_map(settings, brain, descriptor, statistic, values, names)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:285](../../src/boldtailor/workflow/outputs.py#L285)
- [src/boldtailor/workflow/outputs.py:134](../../src/boldtailor/workflow/outputs.py#L134)
- [src/boldtailor/workflow/outputs.py:180](../../src/boldtailor/workflow/outputs.py#L180)
- [src/boldtailor/workflow/outputs.py:193](../../src/boldtailor/workflow/outputs.py#L193)

<a id="definition-2"></a>

## `_map_path`

[Source, line 50](../../src/boldtailor/workflow/outputs.py#L50)

```python
_map_path(settings, descriptor, statistic)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:185](../../src/boldtailor/workflow/outputs.py#L185)
- [src/boldtailor/workflow/outputs.py:679](../../src/boldtailor/workflow/outputs.py#L679)

<a id="definition-3"></a>

## `_run_path`

[Source, line 54](../../src/boldtailor/workflow/outputs.py#L54)

```python
_run_path(settings, run, descriptor, suffix)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:277](../../src/boldtailor/workflow/outputs.py#L277)
- [src/boldtailor/workflow/outputs.py:590](../../src/boldtailor/workflow/outputs.py#L590)

<a id="definition-4"></a>

## `_title`

[Source, line 59](../../src/boldtailor/workflow/outputs.py#L59)

```python
_title(name)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:668](../../src/boldtailor/workflow/outputs.py#L668)
- [src/boldtailor/workflow/outputs.py:209](../../src/boldtailor/workflow/outputs.py#L209)

<a id="definition-5"></a>

## `_session_files`

[Source, line 63](../../src/boldtailor/workflow/outputs.py#L63)

```python
_session_files(settings)
```

This subject/session/task's files in its func directory, and nothing else.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:75](../../src/boldtailor/workflow/outputs.py#L75)

<a id="definition-6"></a>

## `check_output`

[Source, line 70](../../src/boldtailor/workflow/outputs.py#L70)

```python
check_output(settings)
```

Refuse to replace this subject/session/task's outputs unless told to.

<a id="definition-7"></a>

## `_names`

[Source, line 84](../../src/boldtailor/workflow/outputs.py#L84)

```python
_names(paths, shown=5)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:78](../../src/boldtailor/workflow/outputs.py#L78)

<a id="definition-8"></a>

## `metadata_name`

[Source, line 90](../../src/boldtailor/workflow/outputs.py#L90)

```python
metadata_name(settings)
```

The settings file's path relative to the derivative root.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1098](../../src/boldtailor/workflow/outputs.py#L1098)
- [src/boldtailor/workflow/outputs.py:1112](../../src/boldtailor/workflow/outputs.py#L1112)
- [src/boldtailor/workflow/outputs.py:98](../../src/boldtailor/workflow/outputs.py#L98)

<a id="definition-9"></a>

## `_listed_artifacts`

[Source, line 95](../../src/boldtailor/workflow/outputs.py#L95)

```python
_listed_artifacts(settings)
```

The artifact paths an earlier run recorded in its settings file, if any.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1126](../../src/boldtailor/workflow/outputs.py#L1126)

<a id="definition-10"></a>

## `_inside`

[Source, line 107](../../src/boldtailor/workflow/outputs.py#L107)

```python
_inside(relative)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:104](../../src/boldtailor/workflow/outputs.py#L104)

<a id="definition-11"></a>

## `_remove_stale`

[Source, line 112](../../src/boldtailor/workflow/outputs.py#L112)

```python
_remove_stale(settings, listed, published, sources)
```

Delete files the earlier run listed that this run did not write again.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1137](../../src/boldtailor/workflow/outputs.py#L1137)

<a id="definition-12"></a>

## `_design_artifact`

[Source, line 122](../../src/boldtailor/workflow/outputs.py#L122)

```python
_design_artifact(settings, descriptor, result, runs)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:135](../../src/boldtailor/workflow/outputs.py#L135)

<a id="definition-13"></a>

## `_fit_artifacts`

[Source, line 132](../../src/boldtailor/workflow/outputs.py#L132)

```python
_fit_artifacts(settings, brain, descriptor, result, runs)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:312](../../src/boldtailor/workflow/outputs.py#L312)
- [src/boldtailor/workflow/outputs.py:147](../../src/boldtailor/workflow/outputs.py#L147)

<a id="definition-14"></a>

## `glm_artifacts`

[Source, line 142](../../src/boldtailor/workflow/outputs.py#L142)

```python
glm_artifacts(settings, brain, glms, runs, task_model)
```

R², designs, provenance and contrast maps per GLM, plus their comparison.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1036](../../src/boldtailor/workflow/outputs.py#L1036)

<a id="definition-15"></a>

## `_library_artifacts`

[Source, line 167](../../src/boldtailor/workflow/outputs.py#L167)

```python
_library_artifacts(settings, library)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:239](../../src/boldtailor/workflow/outputs.py#L239)

<a id="definition-16"></a>

## `_selection_artifacts`

[Source, line 177](../../src/boldtailor/workflow/outputs.py#L177)

```python
_selection_artifacts(settings, brain, library, maps, name)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:203](../../src/boldtailor/workflow/outputs.py#L203)
- [src/boldtailor/workflow/outputs.py:240](../../src/boldtailor/workflow/outputs.py#L240)

<a id="definition-17"></a>

## `_split_artifacts`

[Source, line 190](../../src/boldtailor/workflow/outputs.py#L190)

```python
_split_artifacts(settings, brain, library, maps)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:247](../../src/boldtailor/workflow/outputs.py#L247)

<a id="definition-18"></a>

## `_all_selection_maps`

[Source, line 218](../../src/boldtailor/workflow/outputs.py#L218)

```python
_all_selection_maps(selections, n_features)
```

The ``all`` selection maps alone; bundles need not hold odd/even splits.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:235](../../src/boldtailor/workflow/outputs.py#L235)
- [src/boldtailor/workflow/summaries.py:47](../../src/boldtailor/workflow/summaries.py#L47)

<a id="definition-19"></a>

## `hrf_artifacts`

[Source, line 230](../../src/boldtailor/workflow/outputs.py#L230)

```python
hrf_artifacts(settings, brain, selections, library, *, include_splits=True)
```

All-run selection, library and provenance; odd/even splits on request.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1039](../../src/boldtailor/workflow/outputs.py#L1039)

<a id="definition-20"></a>

## `_selection_records`

[Source, line 251](../../src/boldtailor/workflow/outputs.py#L251)

```python
_selection_records(selections, scopes)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:243](../../src/boldtailor/workflow/outputs.py#L243)

<a id="definition-21"></a>

## `_scope_record`

[Source, line 263](../../src/boldtailor/workflow/outputs.py#L263)

```python
_scope_record(bundle, name)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:257](../../src/boldtailor/workflow/outputs.py#L257)

<a id="definition-22"></a>

## `_beta_maps`

[Source, line 273](../../src/boldtailor/workflow/outputs.py#L273)

```python
_beta_maps(settings, brain, descriptor, result, runs)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:318](../../src/boldtailor/workflow/outputs.py#L318)

<a id="definition-23"></a>

## `_rt_map`

[Source, line 282](../../src/boldtailor/workflow/outputs.py#L282)

```python
_rt_map(settings, brain, descriptor, rt)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:320](../../src/boldtailor/workflow/outputs.py#L320)

<a id="definition-24"></a>

## `fraction_fit_artifacts`

[Source, line 288](../../src/boldtailor/workflow/outputs.py#L288)

```python
fraction_fit_artifacts(settings, brain, descriptor, result, runs)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:314](../../src/boldtailor/workflow/outputs.py#L314)
- [src/boldtailor/workflow/outputs.py:684](../../src/boldtailor/workflow/outputs.py#L684)

<a id="definition-25"></a>

## `beta_model_artifacts`

[Source, line 309](../../src/boldtailor/workflow/outputs.py#L309)

```python
beta_model_artifacts(settings, brain, runs, model, library)
```

Fit, beta and RT maps for one model; tuning and outer splits when tuned.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1048](../../src/boldtailor/workflow/outputs.py#L1048)

<a id="definition-26"></a>

## `_tuned_artifacts`

[Source, line 326](../../src/boldtailor/workflow/outputs.py#L326)

```python
_tuned_artifacts(settings, brain, runs, model, library)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:322](../../src/boldtailor/workflow/outputs.py#L322)

<a id="definition-27"></a>

## `_predictor_table`

[Source, line 345](../../src/boldtailor/workflow/outputs.py#L345)

```python
_predictor_table(runs, predictors)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:332](../../src/boldtailor/workflow/outputs.py#L332)

<a id="definition-28"></a>

## `_tuned`

[Source, line 360](../../src/boldtailor/workflow/outputs.py#L360)

```python
_tuned(beta_models)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:406](../../src/boldtailor/workflow/outputs.py#L406)
- [src/boldtailor/workflow/outputs.py:456](../../src/boldtailor/workflow/outputs.py#L456)
- [src/boldtailor/workflow/outputs.py:791](../../src/boldtailor/workflow/outputs.py#L791)
- [src/boldtailor/workflow/outputs.py:869](../../src/boldtailor/workflow/outputs.py#L869)

<a id="definition-29"></a>

## `tuning_rows`

[Source, line 364](../../src/boldtailor/workflow/outputs.py#L364)

```python
tuning_rows(mode, scope, tuned)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:418](../../src/boldtailor/workflow/outputs.py#L418)

<a id="definition-30"></a>

## `_alpha_rows`

[Source, line 387](../../src/boldtailor/workflow/outputs.py#L387)

```python
_alpha_rows(mode, scope, tuned)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-31"></a>

## `tuning_table`

[Source, line 403](../../src/boldtailor/workflow/outputs.py#L403)

```python
tuning_table(beta_models)
```

One row per candidate, scope, and tuned model (mode is the HRF model).

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:327](../../src/boldtailor/workflow/outputs.py#L327)

<a id="definition-32"></a>

## `plot_tuning`

[Source, line 416](../../src/boldtailor/workflow/outputs.py#L416)

```python
plot_tuning(axis, mode, tuning)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-33"></a>

## `_plot_alpha_tuning`

[Source, line 433](../../src/boldtailor/workflow/outputs.py#L433)

```python
_plot_alpha_tuning(axis, mode, tuning)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-34"></a>

## `tuning_figure`

[Source, line 454](../../src/boldtailor/workflow/outputs.py#L454)

```python
tuning_figure(beta_models)
```

One tuning-curve panel per tuned model.

<a id="definition-35"></a>

## `_tuning_maps`

[Source, line 465](../../src/boldtailor/workflow/outputs.py#L465)

```python
_tuning_maps(settings, brain, descriptor, tuned, grid_names)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:568](../../src/boldtailor/workflow/outputs.py#L568)

<a id="definition-36"></a>

## `_tuning_files`

[Source, line 489](../../src/boldtailor/workflow/outputs.py#L489)

```python
_tuning_files(settings, descriptor, mode, scope, tuned, table, metadata)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:578](../../src/boldtailor/workflow/outputs.py#L578)

<a id="definition-37"></a>

## `_shared_tuning_metadata`

[Source, line 503](../../src/boldtailor/workflow/outputs.py#L503)

```python
_shared_tuning_metadata(tuned)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:518](../../src/boldtailor/workflow/outputs.py#L518)
- [src/boldtailor/workflow/outputs.py:531](../../src/boldtailor/workflow/outputs.py#L531)

<a id="definition-38"></a>

## `_alpha_tuning_metadata`

[Source, line 515](../../src/boldtailor/workflow/outputs.py#L515)

```python
_alpha_tuning_metadata(tuned)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:576](../../src/boldtailor/workflow/outputs.py#L576)

<a id="definition-39"></a>

## `_fraction_tuning_metadata`

[Source, line 529](../../src/boldtailor/workflow/outputs.py#L529)

```python
_fraction_tuning_metadata(tuned)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:574](../../src/boldtailor/workflow/outputs.py#L574)

<a id="definition-40"></a>

## `_fraction_selection_maps`

[Source, line 542](../../src/boldtailor/workflow/outputs.py#L542)

```python
_fraction_selection_maps(settings, brain, descriptor, selected)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:571](../../src/boldtailor/workflow/outputs.py#L571)

<a id="definition-41"></a>

## `_tuning_artifacts`

[Source, line 563](../../src/boldtailor/workflow/outputs.py#L563)

```python
_tuning_artifacts(settings, brain, mode, scope, tuned, table)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:337](../../src/boldtailor/workflow/outputs.py#L337)

<a id="definition-42"></a>

## `_prediction_artifacts`

[Source, line 583](../../src/boldtailor/workflow/outputs.py#L583)

```python
_prediction_artifacts(settings, brain, descriptor, runs, outer)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:688](../../src/boldtailor/workflow/outputs.py#L688)

<a id="definition-43"></a>

## `_outer_maps`

[Source, line 595](../../src/boldtailor/workflow/outputs.py#L595)

```python
_outer_maps(settings, brain, descriptor, scope, outer, ids)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:672](../../src/boldtailor/workflow/outputs.py#L672)

<a id="definition-44"></a>

## `_outer_loss`

[Source, line 617](../../src/boldtailor/workflow/outputs.py#L617)

```python
_outer_loss(base, outer)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:673](../../src/boldtailor/workflow/outputs.py#L673)

<a id="definition-45"></a>

## `_fraction_outer_metadata`

[Source, line 630](../../src/boldtailor/workflow/outputs.py#L630)

```python
_fraction_outer_metadata(runs)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:656](../../src/boldtailor/workflow/outputs.py#L656)

<a id="definition-46"></a>

## `_outer_metadata`

[Source, line 638](../../src/boldtailor/workflow/outputs.py#L638)

```python
_outer_metadata(runs, outer, fractional)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:676](../../src/boldtailor/workflow/outputs.py#L676)

<a id="definition-47"></a>

## `_outer_artifacts`

[Source, line 665](../../src/boldtailor/workflow/outputs.py#L665)

```python
_outer_artifacts(settings, brain, runs, mode, scope, outer, library)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:340](../../src/boldtailor/workflow/outputs.py#L340)

<a id="definition-48"></a>

## `_boundary_fraction`

[Source, line 692](../../src/boldtailor/workflow/outputs.py#L692)

```python
_boundary_fraction(selection)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:706](../../src/boldtailor/workflow/outputs.py#L706)

<a id="definition-49"></a>

## `ridge_boundary_summary`

[Source, line 700](../../src/boldtailor/workflow/outputs.py#L700)

```python
ridge_boundary_summary(beta_models)
```

Fraction of scored grayordinates whose ridge winner is a grid endpoint.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:806](../../src/boldtailor/workflow/outputs.py#L806)
- [src/boldtailor/workflow/summaries.py:80](../../src/boldtailor/workflow/summaries.py#L80)

<a id="definition-50"></a>

## `_scope_selection`

[Source, line 713](../../src/boldtailor/workflow/outputs.py#L713)

```python
_scope_selection(bundle, scope)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:265](../../src/boldtailor/workflow/outputs.py#L265)
- [src/boldtailor/workflow/outputs.py:745](../../src/boldtailor/workflow/outputs.py#L745)

<a id="definition-51"></a>

## `_pooled_bound_rows`

[Source, line 717](../../src/boldtailor/workflow/outputs.py#L717)

```python
_pooled_bound_rows(picked, scope)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:746](../../src/boldtailor/workflow/outputs.py#L746)

<a id="definition-52"></a>

## `hrf_boundary_summary`

[Source, line 737](../../src/boldtailor/workflow/outputs.py#L737)

```python
hrf_boundary_summary(selections)
```

``parameter_bound_table`` rows pooled over grayordinate blocks per scope.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:752](../../src/boldtailor/workflow/outputs.py#L752)
- [src/boldtailor/workflow/outputs.py:964](../../src/boldtailor/workflow/outputs.py#L964)

<a id="definition-53"></a>

## `hrf_boundary_table`

[Source, line 750](../../src/boldtailor/workflow/outputs.py#L750)

```python
hrf_boundary_table(selections)
```

``hrf_boundary_summary`` as a DataFrame, or None without selections.

<a id="definition-54"></a>

## `activation_artifacts`

[Source, line 756](../../src/boldtailor/workflow/outputs.py#L756)

```python
activation_artifacts(settings, brain, activation)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1050](../../src/boldtailor/workflow/outputs.py#L1050)

<a id="definition-55"></a>

## `input_artifacts`

[Source, line 770](../../src/boldtailor/workflow/outputs.py#L770)

```python
input_artifacts(settings, runs, task_model)
```

Run summary plus the expanded events and retained confounds per run.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1051](../../src/boldtailor/workflow/outputs.py#L1051)

<a id="definition-56"></a>

## `_ridge_metadata`

[Source, line 790](../../src/boldtailor/workflow/outputs.py#L790)

```python
_ridge_metadata(settings, task_model, beta_models)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1015](../../src/boldtailor/workflow/outputs.py#L1015)

<a id="definition-57"></a>

## `_activation_metadata`

[Source, line 811](../../src/boldtailor/workflow/outputs.py#L811)

```python
_activation_metadata()
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1016](../../src/boldtailor/workflow/outputs.py#L1016)

<a id="definition-58"></a>

## `_selection_description`

[Source, line 825](../../src/boldtailor/workflow/outputs.py#L825)

```python
_selection_description(task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1011](../../src/boldtailor/workflow/outputs.py#L1011)

<a id="definition-59"></a>

## `_indicator_clause`

[Source, line 834](../../src/boldtailor/workflow/outputs.py#L834)

```python
_indicator_clause(profiled)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:829](../../src/boldtailor/workflow/outputs.py#L829)

<a id="definition-60"></a>

## `_rt_check_description`

[Source, line 842](../../src/boldtailor/workflow/outputs.py#L842)

```python
_rt_check_description(include_rt, ridge_cv)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:870](../../src/boldtailor/workflow/outputs.py#L870)

<a id="definition-61"></a>

## `_rt_check`

[Source, line 865](../../src/boldtailor/workflow/outputs.py#L865)

```python
_rt_check(include_rt, has_rt, beta_models)
```

RT correlations are computed only when RT is a modulator.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1014](../../src/boldtailor/workflow/outputs.py#L1014)

<a id="definition-62"></a>

## `_task_model_description`

[Source, line 879](../../src/boldtailor/workflow/outputs.py#L879)

```python
_task_model_description(task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:917](../../src/boldtailor/workflow/outputs.py#L917)

<a id="definition-63"></a>

## `_categorical_prose`

[Source, line 889](../../src/boldtailor/workflow/outputs.py#L889)

```python
_categorical_prose(task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:926](../../src/boldtailor/workflow/outputs.py#L926)

<a id="definition-64"></a>

## `_response_time_prose`

[Source, line 899](../../src/boldtailor/workflow/outputs.py#L899)

```python
_response_time_prose()
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:925](../../src/boldtailor/workflow/outputs.py#L925)

<a id="definition-65"></a>

## `_model_prose`

[Source, line 914](../../src/boldtailor/workflow/outputs.py#L914)

```python
_model_prose(task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1009](../../src/boldtailor/workflow/outputs.py#L1009)

<a id="definition-66"></a>

## `_curve_correlation_metadata`

[Source, line 932](../../src/boldtailor/workflow/outputs.py#L932)

```python
_curve_correlation_metadata(library)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:980](../../src/boldtailor/workflow/outputs.py#L980)

<a id="definition-67"></a>

## `_design_metadata`

[Source, line 944](../../src/boldtailor/workflow/outputs.py#L944)

```python
_design_metadata(runs)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1010](../../src/boldtailor/workflow/outputs.py#L1010)

<a id="definition-68"></a>

## `_json_bound_rows`

[Source, line 962](../../src/boldtailor/workflow/outputs.py#L962)

```python
_json_bound_rows(selections)
```

Bound rows with an undefined fraction (no custom picks) as null.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:975](../../src/boldtailor/workflow/outputs.py#L975)

<a id="definition-69"></a>

## `_library_metadata`

[Source, line 973](../../src/boldtailor/workflow/outputs.py#L973)

```python
_library_metadata(library, selections)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1017](../../src/boldtailor/workflow/outputs.py#L1017)

<a id="definition-70"></a>

## `metadata`

[Source, line 984](../../src/boldtailor/workflow/outputs.py#L984)

```python
metadata(runs, library, settings, task_model, *, beta_models, activation, selections, skipped, report, artifacts=None, notes=())
```

The settings file: analysis description plus what ran and what was skipped.

``artifacts`` lists every path this run publishes for the session (not the
shared dataset description); a later overwrite removes only those files.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1099](../../src/boldtailor/workflow/outputs.py#L1099)

<a id="definition-71"></a>

## `report_name`

[Source, line 1027](../../src/boldtailor/workflow/outputs.py#L1027)

```python
report_name(settings)
```

The report's path relative to the derivative root.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:74](../../src/boldtailor/workflow/outputs.py#L74)
- [src/boldtailor/workflow/outputs.py:1118](../../src/boldtailor/workflow/outputs.py#L1118)
- [src/boldtailor/workflow/outputs.py:1172](../../src/boldtailor/workflow/outputs.py#L1172)

<a id="definition-72"></a>

## `_stage_artifacts`

[Source, line 1032](../../src/boldtailor/workflow/outputs.py#L1032)

```python
_stage_artifacts(settings, runs, task_model, library, selections, glms, beta_models, **options)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1087](../../src/boldtailor/workflow/outputs.py#L1087)

<a id="definition-73"></a>

## `figure_name`

[Source, line 1054](../../src/boldtailor/workflow/outputs.py#L1054)

```python
figure_name(settings, name)
```

A figure's PNG path relative to the derivative root.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1061](../../src/boldtailor/workflow/outputs.py#L1061)

<a id="definition-74"></a>

## `_figure_artifacts`

[Source, line 1059](../../src/boldtailor/workflow/outputs.py#L1059)

```python
_figure_artifacts(settings, figures)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1097](../../src/boldtailor/workflow/outputs.py#L1097)

<a id="definition-75"></a>

## `workflow_artifacts`

[Source, line 1066](../../src/boldtailor/workflow/outputs.py#L1066)

```python
workflow_artifacts(settings, runs, task_model, library, selections, glms, beta_models, *, figures, activation, skipped, report, include_hrf_splits=True, notes=())
```

Everything ``save_workflow`` writes except the report and dataset description.

``report`` is the report name recorded in the settings file (or None);
``include_hrf_splits`` adds the odd/even HRF artifacts. Figures stay intact.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1161](../../src/boldtailor/workflow/outputs.py#L1161)

<a id="definition-76"></a>

## `_root_artifacts`

[Source, line 1115](../../src/boldtailor/workflow/outputs.py#L1115)

```python
_root_artifacts(report_html, settings)
```

The report and the dataset description (kept if another session wrote it).

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1130](../../src/boldtailor/workflow/outputs.py#L1130)

<a id="definition-77"></a>

## `publish_workflow`

[Source, line 1123](../../src/boldtailor/workflow/outputs.py#L1123)

```python
publish_workflow(settings, runs, artifacts, report_html)
```

Publish ``workflow_artifacts`` with the report and dataset description at once.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1175](../../src/boldtailor/workflow/outputs.py#L1175)

<a id="definition-78"></a>

## `save_workflow`

[Source, line 1141](../../src/boldtailor/workflow/outputs.py#L1141)

```python
save_workflow(settings, runs, task_model, library, selections, glms, beta_models, *, figures, activation, skipped, report_html, include_hrf_splits=True)
```

Publish every stage, the settings file, figures, and the report at once.

``include_hrf_splits`` writes the odd/even HRF artifacts; pass whether the
reliability stage ran.
