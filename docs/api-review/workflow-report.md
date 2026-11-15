# `boldtailor.workflow.report`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/report.py)

Self-contained HTML report: one section per stage, figures embedded as PNG.

## Definitions

- [`embed_png`](#definition-1)
- [`_default`](#definition-2)
- [`_quoted`](#definition-3)
- [`_path_flags`](#definition-4)
- [`modulator_text`](#definition-5)
- [`_modulator_flags`](#definition-6)
- [`_value_flags`](#definition-7)
- [`_switch_flags`](#definition-8)
- [`command_line`](#definition-9)
- [`_table`](#definition-10)
- [`_optional`](#definition-11)
- [`_section`](#definition-12)
- [`_settings_body`](#definition-13)
- [`_categorical_items`](#definition-14)
- [`_inputs_body`](#definition-15)
- [`_figures`](#definition-16)
- [`_skipped_body`](#definition-17)
- [`_files_body`](#definition-18)
- [`_note`](#definition-19)
- [`_stage_note`](#definition-20)
- [`_confounds`](#definition-21)
- [`_input_tables`](#definition-22)
- [`_bodies`](#definition-23)
- [`render_report`](#definition-24)

<a id="definition-1"></a>

## `embed_png`

[Source, line 62](../../src/boldtailor/workflow/report.py#L62)

```python
embed_png(data)
```

An inline image tag for already encoded PNG bytes.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:215](../../src/boldtailor/workflow/report.py#L215)

<a id="definition-2"></a>

## `_default`

[Source, line 68](../../src/boldtailor/workflow/report.py#L68)

```python
_default(name)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:113](../../src/boldtailor/workflow/report.py#L113)
- [src/boldtailor/workflow/report.py:116](../../src/boldtailor/workflow/report.py#L116)

<a id="definition-3"></a>

## `_quoted`

[Source, line 72](../../src/boldtailor/workflow/report.py#L72)

```python
_quoted(*words)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:137](../../src/boldtailor/workflow/report.py#L137)
- [src/boldtailor/workflow/report.py:83](../../src/boldtailor/workflow/report.py#L83)
- [src/boldtailor/workflow/report.py:86](../../src/boldtailor/workflow/report.py#L86)
- [src/boldtailor/workflow/report.py:106](../../src/boldtailor/workflow/report.py#L106)

<a id="definition-4"></a>

## `_path_flags`

[Source, line 76](../../src/boldtailor/workflow/report.py#L76)

```python
_path_flags(settings)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:148](../../src/boldtailor/workflow/report.py#L148)

<a id="definition-5"></a>

## `modulator_text`

[Source, line 90](../../src/boldtailor/workflow/report.py#L90)

```python
modulator_text(mod)
```

The ``--modulator`` argument that ``parse_modulator`` maps back to ``mod``.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:106](../../src/boldtailor/workflow/report.py#L106)
- [tests/workflow/test_settings.py:158](../../tests/workflow/test_settings.py#L158)
- [tests/workflow/test_settings.py:183](../../tests/workflow/test_settings.py#L183)

<a id="definition-6"></a>

## `_modulator_flags`

[Source, line 101](../../src/boldtailor/workflow/report.py#L101)

```python
_modulator_flags(settings)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:149](../../src/boldtailor/workflow/report.py#L149)

<a id="definition-7"></a>

## `_value_flags`

[Source, line 110](../../src/boldtailor/workflow/report.py#L110)

```python
_value_flags(settings)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:150](../../src/boldtailor/workflow/report.py#L150)

<a id="definition-8"></a>

## `_switch_flags`

[Source, line 121](../../src/boldtailor/workflow/report.py#L121)

```python
_switch_flags(settings)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:151](../../src/boldtailor/workflow/report.py#L151)

<a id="definition-9"></a>

## `command_line`

[Source, line 135](../../src/boldtailor/workflow/report.py#L135)

```python
command_line(settings)
```

The `boldtailor run` command that reproduces these settings.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:178](../../src/boldtailor/workflow/report.py#L178)

<a id="definition-10"></a>

## `_table`

[Source, line 156](../../src/boldtailor/workflow/report.py#L156)

```python
_table(frame)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:166](../../src/boldtailor/workflow/report.py#L166)
- [src/boldtailor/workflow/report.py:279](../../src/boldtailor/workflow/report.py#L279)
- [src/boldtailor/workflow/report.py:286](../../src/boldtailor/workflow/report.py#L286)
- [src/boldtailor/workflow/report.py:287](../../src/boldtailor/workflow/report.py#L287)

<a id="definition-11"></a>

## `_optional`

[Source, line 162](../../src/boldtailor/workflow/report.py#L162)

```python
_optional(title, frame)
```

A titled table, or nothing when the run produced no such table.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:263](../../src/boldtailor/workflow/report.py#L263)
- [src/boldtailor/workflow/report.py:261](../../src/boldtailor/workflow/report.py#L261)
- [src/boldtailor/workflow/report.py:277](../../src/boldtailor/workflow/report.py#L277)
- [src/boldtailor/workflow/report.py:280](../../src/boldtailor/workflow/report.py#L280)

<a id="definition-12"></a>

## `_section`

[Source, line 169](../../src/boldtailor/workflow/report.py#L169)

```python
_section(name, title, body)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:324](../../src/boldtailor/workflow/report.py#L324)

<a id="definition-13"></a>

## `_settings_body`

[Source, line 173](../../src/boldtailor/workflow/report.py#L173)

```python
_settings_body(settings)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:269](../../src/boldtailor/workflow/report.py#L269)

<a id="definition-14"></a>

## `_categorical_items`

[Source, line 182](../../src/boldtailor/workflow/report.py#L182)

```python
_categorical_items(task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:198](../../src/boldtailor/workflow/report.py#L198)

<a id="definition-15"></a>

## `_inputs_body`

[Source, line 191](../../src/boldtailor/workflow/report.py#L191)

```python
_inputs_body(runs, task_model, library, notes=())
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:270](../../src/boldtailor/workflow/report.py#L270)

<a id="definition-16"></a>

## `_figures`

[Source, line 213](../../src/boldtailor/workflow/report.py#L213)

```python
_figures(figures, keys)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:274](../../src/boldtailor/workflow/report.py#L274)
- [src/boldtailor/workflow/report.py:278](../../src/boldtailor/workflow/report.py#L278)
- [src/boldtailor/workflow/report.py:281](../../src/boldtailor/workflow/report.py#L281)
- [src/boldtailor/workflow/report.py:285](../../src/boldtailor/workflow/report.py#L285)

<a id="definition-17"></a>

## `_skipped_body`

[Source, line 221](../../src/boldtailor/workflow/report.py#L221)

```python
_skipped_body(skipped)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:291](../../src/boldtailor/workflow/report.py#L291)

<a id="definition-18"></a>

## `_files_body`

[Source, line 228](../../src/boldtailor/workflow/report.py#L228)

```python
_files_body(manifest)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:292](../../src/boldtailor/workflow/report.py#L292)

<a id="definition-19"></a>

## `_note`

[Source, line 239](../../src/boldtailor/workflow/report.py#L239)

```python
_note(text)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:256](../../src/boldtailor/workflow/report.py#L256)
- [src/boldtailor/workflow/report.py:246](../../src/boldtailor/workflow/report.py#L246)
- [src/boldtailor/workflow/report.py:249](../../src/boldtailor/workflow/report.py#L249)

<a id="definition-20"></a>

## `_stage_note`

[Source, line 243](../../src/boldtailor/workflow/report.py#L243)

```python
_stage_note(name, settings, skipped)
```

One-line note replacing a stage section that did not run, else None.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:324](../../src/boldtailor/workflow/report.py#L324)

<a id="definition-21"></a>

## `_confounds`

[Source, line 253](../../src/boldtailor/workflow/report.py#L253)

```python
_confounds(names)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:262](../../src/boldtailor/workflow/report.py#L262)

<a id="definition-22"></a>

## `_input_tables`

[Source, line 259](../../src/boldtailor/workflow/report.py#L259)

```python
_input_tables(data)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:273](../../src/boldtailor/workflow/report.py#L273)

<a id="definition-23"></a>

## `_bodies`

[Source, line 267](../../src/boldtailor/workflow/report.py#L267)

```python
_bodies(settings, data, figures)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:322](../../src/boldtailor/workflow/report.py#L322)

<a id="definition-24"></a>

## `render_report`

[Source, line 296](../../src/boldtailor/workflow/report.py#L296)

```python
render_report(settings, *, runs, task_model, library, glm_summary, hrf_summary, reliability, tuning, activation_summary, rt_summary, figures, skipped, manifest, notes=(), run_summary=None, confounds=(), library_table=None, selected_hrfs=None, curve_reliability=None, encoding=None, ridge_boundary=None)
```

The report; ``figures`` maps names to PNG bytes, optional tables may be None.
