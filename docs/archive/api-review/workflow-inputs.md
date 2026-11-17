# `boldtailor.workflow.inputs`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/inputs.py)

Session loading, modulator detection, and the GLM model for any task.

## Definitions

- [`InputError`](#definition-1)
- [`selection_task_model`](#definition-2)
- [`WorkflowRun`](#definition-3)
- [`_numeric_column`](#definition-4)
- [`_columns`](#definition-5)
- [`_check_response_time`](#definition-6)
- [`validate_glm_events`](#definition-7)
- [`_missing_nonpositive_rt`](#definition-8)
- [`_checked`](#definition-9)
- [`_categorical_failure`](#definition-10)
- [`_trim`](#definition-11)
- [`_session_task_model`](#definition-12)
- [`load_session`](#definition-13)
- [`block_signals`](#definition-14)
- [`_trimmed_sources`](#definition-15)
- [`load_block`](#definition-16)
- [`glm_model`](#definition-17)
- [`make_blocks`](#definition-18)
- [`_level_counts`](#definition-19)
- [`_run_row`](#definition-20)
- [`run_summary`](#definition-21)

<a id="definition-1"></a>

## `InputError`

[Source, line 22](../../src/boldtailor/workflow/inputs.py#L22)

```python
class InputError(ValueError)
```

Inputs are missing or malformed; the CLI maps this to exit code 2.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:168](../../src/boldtailor/workflow/inputs.py#L168)
- [src/boldtailor/workflow/inputs.py:170](../../src/boldtailor/workflow/inputs.py#L170)
- [src/boldtailor/workflow/inputs.py:172](../../src/boldtailor/workflow/inputs.py#L172)
- [src/boldtailor/workflow/inputs.py:136](../../src/boldtailor/workflow/inputs.py#L136)

<a id="definition-2"></a>

## `selection_task_model`

[Source, line 26](../../src/boldtailor/workflow/inputs.py#L26)

```python
selection_task_model(task_model, include_rt=True)
```

The GLM task model, or without RT when RT must not drive HRF selection.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:1012](../../src/boldtailor/workflow/outputs.py#L1012)

<a id="definition-3"></a>

## `WorkflowRun`

[Source, line 38](../../src/boldtailor/workflow/inputs.py#L38)

```python
@dataclass(frozen=True)
class WorkflowRun
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
inputs: object
image: object
events: pd.DataFrame
confounds: pd.DataFrame
frame_times: np.ndarray
label: str
number: int
retained_frames: np.ndarray
```

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:137](../../src/boldtailor/workflow/inputs.py#L137)

<a id="definition-4"></a>

## `_numeric_column`

[Source, line 49](../../src/boldtailor/workflow/inputs.py#L49)

```python
_numeric_column(events, name)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:66](../../src/boldtailor/workflow/inputs.py#L66)
- [src/boldtailor/workflow/inputs.py:93](../../src/boldtailor/workflow/inputs.py#L93)
- [src/boldtailor/workflow/inputs.py:86](../../src/boldtailor/workflow/inputs.py#L86)

<a id="definition-5"></a>

## `_columns`

[Source, line 61](../../src/boldtailor/workflow/inputs.py#L61)

```python
_columns(task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:267](../../src/boldtailor/workflow/inputs.py#L267)

<a id="definition-6"></a>

## `_check_response_time`

[Source, line 65](../../src/boldtailor/workflow/inputs.py#L65)

```python
_check_response_time(events)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:84](../../src/boldtailor/workflow/inputs.py#L84)

<a id="definition-7"></a>

## `validate_glm_events`

[Source, line 73](../../src/boldtailor/workflow/inputs.py#L73)

```python
validate_glm_events(events, task_model)
```

Require an observed positive RT and every categorical level, if modeled.

Other numeric modulators must parse as numbers.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:108](../../src/boldtailor/workflow/inputs.py#L108)

<a id="definition-8"></a>

## `_missing_nonpositive_rt`

[Source, line 89](../../src/boldtailor/workflow/inputs.py#L89)

```python
_missing_nonpositive_rt(events)
```

Package semantics: missing means non-finite, so nonpositive RTs become NaN.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:109](../../src/boldtailor/workflow/inputs.py#L109)

<a id="definition-9"></a>

## `_checked`

[Source, line 97](../../src/boldtailor/workflow/inputs.py#L97)

```python
_checked(run, task_model)
```

Nonsteady columns, retained frames, and events with RT missingness.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:129](../../src/boldtailor/workflow/inputs.py#L129)

<a id="definition-10"></a>

## `_categorical_failure`

[Source, line 120](../../src/boldtailor/workflow/inputs.py#L120)

```python
_categorical_failure(error, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:133](../../src/boldtailor/workflow/inputs.py#L133)

<a id="definition-11"></a>

## `_trim`

[Source, line 127](../../src/boldtailor/workflow/inputs.py#L127)

```python
_trim(run, task_model, automatic=False)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:166](../../src/boldtailor/workflow/inputs.py#L166)

<a id="definition-12"></a>

## `_session_task_model`

[Source, line 149](../../src/boldtailor/workflow/inputs.py#L149)

```python
_session_task_model(raw_runs, modulators)
```

Detect the model; event problems become InputError (CLI exit code 2).

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:164](../../src/boldtailor/workflow/inputs.py#L164)

<a id="definition-13"></a>

## `load_session`

[Source, line 161](../../src/boldtailor/workflow/inputs.py#L161)

```python
load_session(settings, *, hrf_only=False)
```

Keep original event onsets and acquisition times when dropping NSS scans.

<a id="definition-14"></a>

## `block_signals`

[Source, line 176](../../src/boldtailor/workflow/inputs.py#L176)

```python
block_signals(runs, indices)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:204](../../src/boldtailor/workflow/inputs.py#L204)
- [src/boldtailor/workflow/inputs.py:242](../../src/boldtailor/workflow/inputs.py#L242)

<a id="definition-15"></a>

## `_trimmed_sources`

[Source, line 187](../../src/boldtailor/workflow/inputs.py#L187)

```python
_trimmed_sources(run, root, indices)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:163](../../src/boldtailor/workflow/beta_series.py#L163)
- [src/boldtailor/workflow/inputs.py:208](../../src/boldtailor/workflow/inputs.py#L208)

<a id="definition-16"></a>

## `load_block`

[Source, line 201](../../src/boldtailor/workflow/inputs.py#L201)

```python
load_block(runs, root, indices, task_model)
```

Raw trial rows for every analysis; the task model expands them.

Usage in package/tests:

- [src/boldtailor/workflow/analysis.py:36](../../src/boldtailor/workflow/analysis.py#L36)
- [src/boldtailor/workflow/analysis.py:91](../../src/boldtailor/workflow/analysis.py#L91)
- [src/boldtailor/workflow/analysis.py:180](../../src/boldtailor/workflow/analysis.py#L180)
- [src/boldtailor/workflow/beta_series.py:256](../../src/boldtailor/workflow/beta_series.py#L256)

<a id="definition-17"></a>

## `glm_model`

[Source, line 216](../../src/boldtailor/workflow/inputs.py#L216)

```python
glm_model(runs, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-18"></a>

## `make_blocks`

[Source, line 227](../../src/boldtailor/workflow/inputs.py#L227)

```python
make_blocks(runs, *, block_size=4096, max_grayordinates=None)
```

Skip constant signals; exports restore their positions as NaN.

<a id="definition-19"></a>

## `_level_counts`

[Source, line 253](../../src/boldtailor/workflow/inputs.py#L253)

```python
_level_counts(run, modulator)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:271](../../src/boldtailor/workflow/inputs.py#L271)

<a id="definition-20"></a>

## `_run_row`

[Source, line 258](../../src/boldtailor/workflow/inputs.py#L258)

```python
_run_row(run, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:276](../../src/boldtailor/workflow/inputs.py#L276)

<a id="definition-21"></a>

## `run_summary`

[Source, line 275](../../src/boldtailor/workflow/inputs.py#L275)

```python
run_summary(runs, task_model)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/outputs.py:775](../../src/boldtailor/workflow/outputs.py#L775)
- [src/boldtailor/workflow/summaries.py:89](../../src/boldtailor/workflow/summaries.py#L89)
