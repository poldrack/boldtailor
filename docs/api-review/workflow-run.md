# `boldtailor.workflow.run`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/run.py)

Run the enabled stages in order and publish one BIDS derivative with a report.

## Definitions

- [`WorkflowResult`](#definition-1)
- [`_State`](#definition-2)
- [`_State.brain`](#definition-3)
- [`_State.enabled`](#definition-4)
- [`_input_errors`](#definition-5)
- [`_session`](#definition-6)
- [`describe_inputs`](#definition-7)
- [`_describe_session`](#definition-8)
- [`_load`](#definition-9)
- [`_find_meshes`](#definition-10)
- [`_splits_possible`](#definition-11)
- [`_check_ridge_cv`](#definition-12)
- [`_reliability_runs`](#definition-13)
- [`_hrf_selection`](#definition-14)
- [`_library_figure`](#definition-15)
- [`_glm_stage`](#definition-16)
- [`_glm_figures`](#definition-17)
- [`_reliability_stage`](#definition-18)
- [`_beta_stage`](#definition-19)
- [`_tuning_figures`](#definition-20)
- [`_summaries_stage`](#definition-21)
- [`_surface_figures`](#definition-22)
- [`_surface_maps`](#definition-23)
- [`_provenance`](#definition-24)
- [`_manifest`](#definition-25)
- [`_stage_tables`](#definition-26)
- [`_render`](#definition-27)
- [`_publish`](#definition-28)
- [`_close`](#definition-29)
- [`run_workflow`](#definition-30)

<a id="definition-1"></a>

## `WorkflowResult`

[Source, line 43](../../src/boldtailor/workflow/run.py#L43)

```python
@dataclass(frozen=True, kw_only=True)
class WorkflowResult
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
settings: WorkflowSettings
paths: tuple[Path, ...]
report_path: Path | None
skipped: tuple[tuple[str, str], ...]
task_model: TaskModel
library_fingerprint: str
```

Usage in package/tests:

- [src/boldtailor/workflow/run.py:436](../../src/boldtailor/workflow/run.py#L436)

<a id="definition-2"></a>

## `_State`

[Source, line 53](../../src/boldtailor/workflow/run.py#L53)

```python
@dataclass(kw_only=True)
class _State
```

The session inputs and every stage's results, filled in stage order.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
settings: WorkflowSettings
runs: tuple
task_model: TaskModel
library: object
blocks: list
selections: dict | None = None
glms: dict = field(default_factory=dict)
glm_summary: pd.DataFrame | None = None
reliability: pd.DataFrame | None = None
tuning: pd.DataFrame | None = None
beta_models: dict = field(default_factory=dict)
activation: dict | None = None
curve_reliability: pd.DataFrame | None = None
figures: dict = field(default_factory=dict)
skipped: list = field(default_factory=list)
notes: list = field(default_factory=list)
meshes: dict | None = None
```

Usage in package/tests:

- [src/boldtailor/workflow/run.py:151](../../src/boldtailor/workflow/run.py#L151)

<a id="definition-3"></a>

## `_State.brain`

[Source, line 75](../../src/boldtailor/workflow/run.py#L75)

```python
@property
_State.brain(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `_State.enabled`

[Source, line 78](../../src/boldtailor/workflow/run.py#L78)

```python
_State.enabled(self, stage)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `_input_errors`

[Source, line 83](../../src/boldtailor/workflow/run.py#L83)

```python
@contextmanager
_input_errors()
```

Loading and validation failures are input errors (CLI exit code 2).

Usage in package/tests:

- [src/boldtailor/workflow/run.py:95](../../src/boldtailor/workflow/run.py#L95)
- [src/boldtailor/workflow/run.py:141](../../src/boldtailor/workflow/run.py#L141)

<a id="definition-6"></a>

## `_session`

[Source, line 93](../../src/boldtailor/workflow/run.py#L93)

```python
_session(settings)
```

Trimmed runs, the task model, and notes on what detection left out.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:131](../../src/boldtailor/workflow/run.py#L131)
- [src/boldtailor/workflow/run.py:139](../../src/boldtailor/workflow/run.py#L139)

<a id="definition-7"></a>

## `describe_inputs`

[Source, line 105](../../src/boldtailor/workflow/run.py#L105)

```python
describe_inputs(settings)
```

What a run would use (for ``--dry-run``), without fitting anything.

``problems`` lists what would stop the run, each with a ``kind``
(``existing_results`` or ``input``) and a ``message``, in run order.

<a id="definition-8"></a>

## `_describe_session`

[Source, line 130](../../src/boldtailor/workflow/run.py#L130)

```python
_describe_session(settings, plan)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:124](../../src/boldtailor/workflow/run.py#L124)

<a id="definition-9"></a>

## `_load`

[Source, line 138](../../src/boldtailor/workflow/run.py#L138)

```python
_load(settings)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:429](../../src/boldtailor/workflow/run.py#L429)

<a id="definition-10"></a>

## `_find_meshes`

[Source, line 162](../../src/boldtailor/workflow/run.py#L162)

```python
_find_meshes(settings, notes)
```

Meshes for surface figures; a missing explicit mesh is an input error.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:135](../../src/boldtailor/workflow/run.py#L135)
- [src/boldtailor/workflow/run.py:148](../../src/boldtailor/workflow/run.py#L148)

<a id="definition-11"></a>

## `_splits_possible`

[Source, line 182](../../src/boldtailor/workflow/run.py#L182)

```python
_splits_possible(runs, minimum=2)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:191](../../src/boldtailor/workflow/run.py#L191)
- [src/boldtailor/workflow/run.py:198](../../src/boldtailor/workflow/run.py#L198)
- [src/boldtailor/workflow/run.py:257](../../src/boldtailor/workflow/run.py#L257)

<a id="definition-12"></a>

## `_check_ridge_cv`

[Source, line 186](../../src/boldtailor/workflow/run.py#L186)

```python
_check_ridge_cv(settings, runs, task_model)
```

Ridge CV needs three runs per half and a modulator; refuse before fitting.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:134](../../src/boldtailor/workflow/run.py#L134)
- [src/boldtailor/workflow/run.py:140](../../src/boldtailor/workflow/run.py#L140)

<a id="definition-13"></a>

## `_reliability_runs`

[Source, line 197](../../src/boldtailor/workflow/run.py#L197)

```python
_reliability_runs(state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:203](../../src/boldtailor/workflow/run.py#L203)
- [src/boldtailor/workflow/run.py:215](../../src/boldtailor/workflow/run.py#L215)
- [src/boldtailor/workflow/run.py:404](../../src/boldtailor/workflow/run.py#L404)

<a id="definition-14"></a>

## `_hrf_selection`

[Source, line 201](../../src/boldtailor/workflow/run.py#L201)

```python
_hrf_selection(state)
```

All-run selections, with odd/even splits only for the reliability stage.

<a id="definition-15"></a>

## `_library_figure`

[Source, line 219](../../src/boldtailor/workflow/run.py#L219)

```python
_library_figure(state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-16"></a>

## `_glm_stage`

[Source, line 223](../../src/boldtailor/workflow/run.py#L223)

```python
_glm_stage(state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-17"></a>

## `_glm_figures`

[Source, line 245](../../src/boldtailor/workflow/run.py#L245)

```python
_glm_figures(state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:242](../../src/boldtailor/workflow/run.py#L242)

<a id="definition-18"></a>

## `_reliability_stage`

[Source, line 254](../../src/boldtailor/workflow/run.py#L254)

```python
_reliability_stage(state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-19"></a>

## `_beta_stage`

[Source, line 272](../../src/boldtailor/workflow/run.py#L272)

```python
_beta_stage(state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-20"></a>

## `_tuning_figures`

[Source, line 289](../../src/boldtailor/workflow/run.py#L289)

```python
_tuning_figures(state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:286](../../src/boldtailor/workflow/run.py#L286)

<a id="definition-21"></a>

## `_summaries_stage`

[Source, line 298](../../src/boldtailor/workflow/run.py#L298)

```python
_summaries_stage(state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-22"></a>

## `_surface_figures`

[Source, line 310](../../src/boldtailor/workflow/run.py#L310)

```python
_surface_figures(state)
```

Cortical maps, only when requested and the subject's meshes are found.

<a id="definition-23"></a>

## `_surface_maps`

[Source, line 321](../../src/boldtailor/workflow/run.py#L321)

```python
_surface_maps(state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:314](../../src/boldtailor/workflow/run.py#L314)

<a id="definition-24"></a>

## `_provenance`

[Source, line 332](../../src/boldtailor/workflow/run.py#L332)

```python
_provenance(path, stem, listed)
```

The provenance JSON sharing this file's descriptor, when one was written.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:347](../../src/boldtailor/workflow/run.py#L347)

<a id="definition-25"></a>

## `_manifest`

[Source, line 345](../../src/boldtailor/workflow/run.py#L345)

```python
_manifest(paths, stem)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:382](../../src/boldtailor/workflow/run.py#L382)

<a id="definition-26"></a>

## `_stage_tables`

[Source, line 350](../../src/boldtailor/workflow/run.py#L350)

```python
_stage_tables(state)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:384](../../src/boldtailor/workflow/run.py#L384)

<a id="definition-27"></a>

## `_render`

[Source, line 363](../../src/boldtailor/workflow/run.py#L363)

```python
_render(state, artifacts)
```

The report, embedding the figure PNGs exactly as published.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:406](../../src/boldtailor/workflow/run.py#L406)

<a id="definition-28"></a>

## `_publish`

[Source, line 388](../../src/boldtailor/workflow/run.py#L388)

```python
_publish(state)
```

Write every artifact and, last in the same publication, the report.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:433](../../src/boldtailor/workflow/run.py#L433)

<a id="definition-29"></a>

## `_close`

[Source, line 421](../../src/boldtailor/workflow/run.py#L421)

```python
_close(figures)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/run.py:435](../../src/boldtailor/workflow/run.py#L435)

<a id="definition-30"></a>

## `run_workflow`

[Source, line 426](../../src/boldtailor/workflow/run.py#L426)

```python
run_workflow(settings)
```

Run the enabled stages and publish the derivative; refuse existing outputs.
