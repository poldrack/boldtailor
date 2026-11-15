# `boldtailor.workflow.settings`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/workflow/settings.py)

Resolved, validated settings for one workflow run; the CLI, stages, writer and metadata share it.

## Definitions

- [`resolve_fmriprep_dir`](#definition-1)
- [`check_modulator_expressible`](#definition-2)
- [`_modulator_options`](#definition-3)
- [`parse_modulator`](#definition-4)
- [`_check_unresolved`](#definition-5)
- [`_check_expressible`](#definition-6)
- [`_nested`](#definition-7)
- [`_check_label`](#definition-8)
- [`_check_choice`](#definition-9)
- [`_check_positive_int`](#definition-10)
- [`_check_bool`](#definition-11)
- [`_fraction`](#definition-12)
- [`_penalty`](#definition-13)
- [`_check_grid`](#definition-14)
- [`WorkflowSettings`](#definition-15)
- [`WorkflowSettings.__post_init__`](#definition-16)
- [`WorkflowSettings._set`](#definition-17)
- [`WorkflowSettings._validate_paths`](#definition-18)
- [`WorkflowSettings._validate_labels`](#definition-19)
- [`WorkflowSettings._validate_modulators`](#definition-20)
- [`WorkflowSettings._validate_hrf`](#definition-21)
- [`WorkflowSettings._validate_ridge`](#definition-22)
- [`WorkflowSettings._validate_stages`](#definition-23)
- [`WorkflowSettings._validate_surfaces`](#definition-24)
- [`WorkflowSettings._validate_execution`](#definition-25)
- [`WorkflowSettings._resolve_output_dir`](#definition-26)
- [`WorkflowSettings._check_output_overlap`](#definition-27)
- [`WorkflowSettings.output_name`](#definition-28)
- [`WorkflowSettings.stem`](#definition-29)
- [`WorkflowSettings.space_entity`](#definition-30)
- [`WorkflowSettings.build_library`](#definition-31)
- [`WorkflowSettings.to_dict`](#definition-32)
- [`WorkflowSettings.from_dict`](#definition-33)
- [`_to_plain`](#definition-34)

<a id="definition-1"></a>

## `resolve_fmriprep_dir`

[Source, line 29](../../src/boldtailor/workflow/settings.py#L29)

```python
resolve_fmriprep_dir(bids_dir)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:79](../../src/boldtailor/workflow/report.py#L79)
- [src/boldtailor/workflow/settings.py:221](../../src/boldtailor/workflow/settings.py#L221)
- [tests/workflow/test_settings.py:70](../../tests/workflow/test_settings.py#L70)
- [tests/workflow/test_settings.py:72](../../tests/workflow/test_settings.py#L72)

<a id="definition-2"></a>

## `check_modulator_expressible`

[Source, line 50](../../src/boldtailor/workflow/settings.py#L50)

```python
check_modulator_expressible(mod)
```

Raise ValueError if ``--modulator`` syntax cannot spell ``mod``.

Usage in package/tests:

- [src/boldtailor/workflow/report.py:92](../../src/boldtailor/workflow/report.py#L92)
- [src/boldtailor/workflow/settings.py:100](../../src/boldtailor/workflow/settings.py#L100)

<a id="definition-3"></a>

## `_modulator_options`

[Source, line 62](../../src/boldtailor/workflow/settings.py#L62)

```python
_modulator_options(text, options)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:81](../../src/boldtailor/workflow/settings.py#L81)

<a id="definition-4"></a>

## `parse_modulator`

[Source, line 77](../../src/boldtailor/workflow/settings.py#L77)

```python
parse_modulator(text)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [tests/workflow/test_settings.py:133](../../tests/workflow/test_settings.py#L133)
- [tests/workflow/test_settings.py:151](../../tests/workflow/test_settings.py#L151)
- [tests/workflow/test_settings.py:183](../../tests/workflow/test_settings.py#L183)

<a id="definition-5"></a>

## `_check_unresolved`

[Source, line 90](../../src/boldtailor/workflow/settings.py#L90)

```python
_check_unresolved(mod)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:237](../../src/boldtailor/workflow/settings.py#L237)

<a id="definition-6"></a>

## `_check_expressible`

[Source, line 98](../../src/boldtailor/workflow/settings.py#L98)

```python
_check_expressible(mod)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:238](../../src/boldtailor/workflow/settings.py#L238)

<a id="definition-7"></a>

## `_nested`

[Source, line 105](../../src/boldtailor/workflow/settings.py#L105)

```python
_nested(path, other)
```

Whether two resolved paths are equal or one contains the other.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:312](../../src/boldtailor/workflow/settings.py#L312)

<a id="definition-8"></a>

## `_check_label`

[Source, line 110](../../src/boldtailor/workflow/settings.py#L110)

```python
_check_label(value, name, prefix)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:226](../../src/boldtailor/workflow/settings.py#L226)
- [src/boldtailor/workflow/settings.py:227](../../src/boldtailor/workflow/settings.py#L227)
- [src/boldtailor/workflow/settings.py:228](../../src/boldtailor/workflow/settings.py#L228)

<a id="definition-9"></a>

## `_check_choice`

[Source, line 119](../../src/boldtailor/workflow/settings.py#L119)

```python
_check_choice(value, name, choices)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:242](../../src/boldtailor/workflow/settings.py#L242)
- [src/boldtailor/workflow/settings.py:251](../../src/boldtailor/workflow/settings.py#L251)
- [src/boldtailor/workflow/settings.py:267](../../src/boldtailor/workflow/settings.py#L267)
- [src/boldtailor/workflow/settings.py:300](../../src/boldtailor/workflow/settings.py#L300)

<a id="definition-10"></a>

## `_check_positive_int`

[Source, line 125](../../src/boldtailor/workflow/settings.py#L125)

```python
_check_positive_int(value, name, *, optional=False)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:243](../../src/boldtailor/workflow/settings.py#L243)
- [src/boldtailor/workflow/settings.py:292](../../src/boldtailor/workflow/settings.py#L292)
- [src/boldtailor/workflow/settings.py:293](../../src/boldtailor/workflow/settings.py#L293)
- [src/boldtailor/workflow/settings.py:296](../../src/boldtailor/workflow/settings.py#L296)

<a id="definition-11"></a>

## `_check_bool`

[Source, line 133](../../src/boldtailor/workflow/settings.py#L133)

```python
_check_bool(value, name)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:248](../../src/boldtailor/workflow/settings.py#L248)
- [src/boldtailor/workflow/settings.py:283](../../src/boldtailor/workflow/settings.py#L283)

<a id="definition-12"></a>

## `_fraction`

[Source, line 139](../../src/boldtailor/workflow/settings.py#L139)

```python
_fraction(value)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-13"></a>

## `_penalty`

[Source, line 143](../../src/boldtailor/workflow/settings.py#L143)

```python
_penalty(value)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-14"></a>

## `_check_grid`

[Source, line 147](../../src/boldtailor/workflow/settings.py#L147)

```python
_check_grid(values, name, accept, wanted)
```

A non-empty grid of distinct real values, each accepted by ``accept``.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:254](../../src/boldtailor/workflow/settings.py#L254)
- [src/boldtailor/workflow/settings.py:260](../../src/boldtailor/workflow/settings.py#L260)

<a id="definition-15"></a>

## `WorkflowSettings`

[Source, line 160](../../src/boldtailor/workflow/settings.py#L160)

```python
@dataclass(frozen=True, kw_only=True)
class WorkflowSettings
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
bids_dir: Path
subject: str
session: str
task: str
fmriprep_dir: Path | None = None
output_dir: Path | None = None
space: str = 'fsLR-91k'
modulators: tuple[Modulator, ...] | None = None
hrf_library: str = 'default'
hrf_n_samples: int = 512
hrf_seed: int = 0
hrf_selection_rt: bool = True
ridge_mode: str = 'fractional_cv'
ridge_fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ridge_alphas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
ridge_percentile: float = 90.0
ridge_alpha: float = 0.1
encoding_mode: str = 'within_run'
stages: frozenset[str] = field(default_factory=lambda: frozenset(STAGES))
surface_maps: bool = True
surface_meshes: Mapping[str, Path] | None = None
n_jobs: int = 4
block_size: int = 4096
max_grayordinates: int | None = None
existing_results: str = 'error'
```

Usage in package/tests:

- [src/boldtailor/cli.py:170](../../src/boldtailor/cli.py#L170)
- [tests/workflow/test_settings.py:24](../../tests/workflow/test_settings.py#L24)
- [tests/workflow/test_settings.py:107](../../tests/workflow/test_settings.py#L107)

<a id="definition-16"></a>

## `WorkflowSettings.__post_init__`

[Source, line 198](../../src/boldtailor/workflow/settings.py#L198)

```python
WorkflowSettings.__post_init__(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-17"></a>

## `WorkflowSettings._set`

[Source, line 210](../../src/boldtailor/workflow/settings.py#L210)

```python
WorkflowSettings._set(self, name, value)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-18"></a>

## `WorkflowSettings._validate_paths`

[Source, line 213](../../src/boldtailor/workflow/settings.py#L213)

```python
WorkflowSettings._validate_paths(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-19"></a>

## `WorkflowSettings._validate_labels`

[Source, line 225](../../src/boldtailor/workflow/settings.py#L225)

```python
WorkflowSettings._validate_labels(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-20"></a>

## `WorkflowSettings._validate_modulators`

[Source, line 230](../../src/boldtailor/workflow/settings.py#L230)

```python
WorkflowSettings._validate_modulators(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-21"></a>

## `WorkflowSettings._validate_hrf`

[Source, line 241](../../src/boldtailor/workflow/settings.py#L241)

```python
WorkflowSettings._validate_hrf(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-22"></a>

## `WorkflowSettings._validate_ridge`

[Source, line 250](../../src/boldtailor/workflow/settings.py#L250)

```python
WorkflowSettings._validate_ridge(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-23"></a>

## `WorkflowSettings._validate_stages`

[Source, line 269](../../src/boldtailor/workflow/settings.py#L269)

```python
WorkflowSettings._validate_stages(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-24"></a>

## `WorkflowSettings._validate_surfaces`

[Source, line 282](../../src/boldtailor/workflow/settings.py#L282)

```python
WorkflowSettings._validate_surfaces(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-25"></a>

## `WorkflowSettings._validate_execution`

[Source, line 291](../../src/boldtailor/workflow/settings.py#L291)

```python
WorkflowSettings._validate_execution(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-26"></a>

## `WorkflowSettings._resolve_output_dir`

[Source, line 302](../../src/boldtailor/workflow/settings.py#L302)

```python
WorkflowSettings._resolve_output_dir(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-27"></a>

## `WorkflowSettings._check_output_overlap`

[Source, line 308](../../src/boldtailor/workflow/settings.py#L308)

```python
WorkflowSettings._check_output_overlap(self)
```

Outputs may never share a directory tree with the inputs.

<a id="definition-28"></a>

## `WorkflowSettings.output_name`

[Source, line 318](../../src/boldtailor/workflow/settings.py#L318)

```python
WorkflowSettings.output_name(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-29"></a>

## `WorkflowSettings.stem`

[Source, line 328](../../src/boldtailor/workflow/settings.py#L328)

```python
@property
WorkflowSettings.stem(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-30"></a>

## `WorkflowSettings.space_entity`

[Source, line 332](../../src/boldtailor/workflow/settings.py#L332)

```python
@property
WorkflowSettings.space_entity(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-31"></a>

## `WorkflowSettings.build_library`

[Source, line 335](../../src/boldtailor/workflow/settings.py#L335)

```python
WorkflowSettings.build_library(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-32"></a>

## `WorkflowSettings.to_dict`

[Source, line 346](../../src/boldtailor/workflow/settings.py#L346)

```python
WorkflowSettings.to_dict(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-33"></a>

## `WorkflowSettings.from_dict`

[Source, line 353](../../src/boldtailor/workflow/settings.py#L353)

```python
@classmethod
WorkflowSettings.from_dict(cls, values)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-34"></a>

## `_to_plain`

[Source, line 365](../../src/boldtailor/workflow/settings.py#L365)

```python
_to_plain(name, value)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/workflow/settings.py:348](../../src/boldtailor/workflow/settings.py#L348)
