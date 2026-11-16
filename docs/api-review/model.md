# `boldtailor.model`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/model.py)

Specify contrasts, nuisance settings and shared task regressors.

## Definitions

- [`level_name`](#definition-1)
- [`_parse_number`](#definition-2)
- [`_number_name`](#definition-3)
- [`_sorted_levels`](#definition-4)
- [`Modulator`](#definition-5)
- [`Modulator.__post_init__`](#definition-6)
- [`Modulator._reference_name`](#definition-7)
- [`Modulator._set_reference`](#definition-8)
- [`Modulator._set_resolved_levels`](#definition-9)
- [`Modulator.resolved`](#definition-10)
- [`Modulator.with_levels`](#definition-11)
- [`Modulator.regressor_names`](#definition-12)
- [`Modulator.indicator_name`](#definition-13)
- [`Modulator.to_dict`](#definition-14)
- [`TaskModel`](#definition-15)
- [`TaskModel.__post_init__`](#definition-16)
- [`TaskModel.regressor_names`](#definition-17)
- [`TaskModel.profiled_names`](#definition-18)
- [`TaskModel.to_dict`](#definition-19)
- [`TaskModel.fingerprint`](#definition-20)
- [`TaskModel.is_subset_of`](#definition-21)
- [`_validate_modulator_column`](#definition-22)
- [`ModelSpec`](#definition-23)
- [`ModelSpec.__post_init__`](#definition-24)
- [`ModelSpec.contrast_names`](#definition-25)
- [`contrast_metadata`](#definition-26)
- [`contrasts_from_metadata`](#definition-27)
- [`_prepare_contrasts`](#definition-28)
- [`_prepare_contrast`](#definition-29)
- [`_prepare_weights`](#definition-30)
- [`_prepare_confound_names`](#definition-31)
- [`_validate_noise_model`](#definition-32)
- [`_validate_task_model`](#definition-33)
- [`_validate_design_options`](#definition-34)
- [`ModelIdentity`](#definition-35)
- [`nuisance_model_settings`](#definition-36)
- [`model_identity`](#definition-37)
- [`_identity_activity`](#definition-38)
- [`_hrf_normalization`](#definition-39)
- [`_serialize_hrf`](#definition-40)
- [`_callable_identity`](#definition-41)

<a id="definition-1"></a>

## `level_name`

[Source, line 32](../../src/boldtailor/model.py#L32)

```python
level_name(value: object) -> str | None
```

Canonical string for one categorical value, or None when it is missing.

Numbers and strings that parse as numbers share one form (1, 1.0 and "1.0"
all become "1"), so a level written differently across runs stays one
level. NaN, None, "" and "n/a" are missing; infinities and booleans are
rejected with a ValueError rather than silently becoming levels or gaps.

Usage in package/tests:

- [src/boldtailor/_task_design.py:47](../../src/boldtailor/_task_design.py#L47)
- [src/boldtailor/model.py:109](../../src/boldtailor/model.py#L109)
- [src/boldtailor/model.py:120](../../src/boldtailor/model.py#L120)
- [src/boldtailor/task_detection.py:14](../../src/boldtailor/task_detection.py#L14)

<a id="definition-2"></a>

## `_parse_number`

[Source, line 53](../../src/boldtailor/model.py#L53)

```python
_parse_number(text: str) -> float | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:44](../../src/boldtailor/model.py#L44)
- [src/boldtailor/model.py:70](../../src/boldtailor/model.py#L70)

<a id="definition-3"></a>

## `_number_name`

[Source, line 60](../../src/boldtailor/model.py#L60)

```python
_number_name(number: float) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:49](../../src/boldtailor/model.py#L49)
- [src/boldtailor/model.py:45](../../src/boldtailor/model.py#L45)

<a id="definition-4"></a>

## `_sorted_levels`

[Source, line 68](../../src/boldtailor/model.py#L68)

```python
_sorted_levels(levels)
```

Numeric order when every level is a number, else lexical order.

Usage in package/tests:

- [src/boldtailor/model.py:129](../../src/boldtailor/model.py#L129)

<a id="definition-5"></a>

## `Modulator`

[Source, line 76](../../src/boldtailor/model.py#L76)

```python
@dataclass(frozen=True)
class Modulator
```

One task regressor family derived from a raw events column.

Numeric modulators enter uncentered, so the task regressor is the response
at modulator value zero. Categorical modulators add one 0/1 indicator per
non-reference level, so the task regressor is the reference-level response.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
column: str
missing: str = 'error'
kind: str = 'numeric'
levels: tuple[str, ...] | None = None
reference: str | None = None
```

Usage in package/tests:

- [src/boldtailor/task_detection.py:35](../../src/boldtailor/task_detection.py#L35)
- [src/boldtailor/task_detection.py:40](../../src/boldtailor/task_detection.py#L40)
- [src/boldtailor/workflow/settings.py:82](../../src/boldtailor/workflow/settings.py#L82)
- [src/boldtailor/workflow/settings.py:356](../../src/boldtailor/workflow/settings.py#L356)

<a id="definition-6"></a>

## `Modulator.__post_init__`

[Source, line 90](../../src/boldtailor/model.py#L90)

```python
Modulator.__post_init__(self) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `Modulator._reference_name`

[Source, line 106](../../src/boldtailor/model.py#L106)

```python
Modulator._reference_name(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-8"></a>

## `Modulator._set_reference`

[Source, line 116](../../src/boldtailor/model.py#L116)

```python
Modulator._set_reference(self, name)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `Modulator._set_resolved_levels`

[Source, line 119](../../src/boldtailor/model.py#L119)

```python
Modulator._set_resolved_levels(self)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-10"></a>

## `Modulator.resolved`

[Source, line 139](../../src/boldtailor/model.py#L139)

```python
@property
Modulator.resolved(self) -> bool
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-11"></a>

## `Modulator.with_levels`

[Source, line 142](../../src/boldtailor/model.py#L142)

```python
Modulator.with_levels(self, values) -> 'Modulator'
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-12"></a>

## `Modulator.regressor_names`

[Source, line 146](../../src/boldtailor/model.py#L146)

```python
@property
Modulator.regressor_names(self) -> tuple[str, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-13"></a>

## `Modulator.indicator_name`

[Source, line 156](../../src/boldtailor/model.py#L156)

```python
@property
Modulator.indicator_name(self) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-14"></a>

## `Modulator.to_dict`

[Source, line 159](../../src/boldtailor/model.py#L159)

```python
Modulator.to_dict(self) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-15"></a>

## `TaskModel`

[Source, line 168](../../src/boldtailor/model.py#L168)

```python
@dataclass(frozen=True)
class TaskModel
```

Task regressor plus modulators; the default is the task regressor alone.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
modulators: tuple[Modulator, ...] = ()
```

Usage in package/tests:

- [src/boldtailor/_denoising_pool.py:74](../../src/boldtailor/_denoising_pool.py#L74)
- [src/boldtailor/_hrf_cv.py:188](../../src/boldtailor/_hrf_cv.py#L188)
- [src/boldtailor/_hrf_glm.py:51](../../src/boldtailor/_hrf_glm.py#L51)
- [src/boldtailor/denoising.py:428](../../src/boldtailor/denoising.py#L428)

<a id="definition-16"></a>

## `TaskModel.__post_init__`

[Source, line 173](../../src/boldtailor/model.py#L173)

```python
TaskModel.__post_init__(self) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-17"></a>

## `TaskModel.regressor_names`

[Source, line 188](../../src/boldtailor/model.py#L188)

```python
@property
TaskModel.regressor_names(self) -> tuple[str, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-18"></a>

## `TaskModel.profiled_names`

[Source, line 192](../../src/boldtailor/model.py#L192)

```python
@property
TaskModel.profiled_names(self) -> tuple[str, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-19"></a>

## `TaskModel.to_dict`

[Source, line 197](../../src/boldtailor/model.py#L197)

```python
TaskModel.to_dict(self) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-20"></a>

## `TaskModel.fingerprint`

[Source, line 204](../../src/boldtailor/model.py#L204)

```python
@property
TaskModel.fingerprint(self) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-21"></a>

## `TaskModel.is_subset_of`

[Source, line 207](../../src/boldtailor/model.py#L207)

```python
TaskModel.is_subset_of(self, other: 'TaskModel') -> bool
```

True when every modulator here appears unchanged in the other model.

<a id="definition-22"></a>

## `_validate_modulator_column`

[Source, line 214](../../src/boldtailor/model.py#L214)

```python
_validate_modulator_column(column: object) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:91](../../src/boldtailor/model.py#L91)

<a id="definition-23"></a>

## `ModelSpec`

[Source, line 222](../../src/boldtailor/model.py#L222)

```python
@dataclass(frozen=True)
class ModelSpec
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
contrasts: Mapping[str, ContrastValue]
confounds: Sequence[str] = ()
hrf_model: HRFModel = 'glover'
drift_model: str | None = 'cosine'
high_pass: float = 0.01
drift_order: int = 1
oversampling: int = OVERSAMPLING
min_onset: float = MIN_ONSET
noise_model: str = 'ar1'
task_model: TaskModel | None = None
_contrast_names: tuple[str, ...] = field(init=False, repr=False)
```

Usage in package/tests:

- [src/boldtailor/workflow/inputs.py:217](../../src/boldtailor/workflow/inputs.py#L217)
- [tests/conftest.py:218](../../tests/conftest.py#L218)
- [tests/conftest.py:230](../../tests/conftest.py#L230)
- [tests/test_denoising.py:727](../../tests/test_denoising.py#L727)

<a id="definition-24"></a>

## `ModelSpec.__post_init__`

[Source, line 235](../../src/boldtailor/model.py#L235)

```python
ModelSpec.__post_init__(self) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-25"></a>

## `ModelSpec.contrast_names`

[Source, line 254](../../src/boldtailor/model.py#L254)

```python
@property
ModelSpec.contrast_names(self) -> tuple[str, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-26"></a>

## `contrast_metadata`

[Source, line 258](../../src/boldtailor/model.py#L258)

```python
contrast_metadata(contrasts)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:404](../../src/boldtailor/model.py#L404)
- [src/boldtailor/prepared_fit.py:302](../../src/boldtailor/prepared_fit.py#L302)

<a id="definition-27"></a>

## `contrasts_from_metadata`

[Source, line 269](../../src/boldtailor/model.py#L269)

```python
contrasts_from_metadata(metadata)
```

Invert :func:`contrast_metadata` back to expressions and weight maps.

Usage in package/tests:

- [src/boldtailor/prepared_fit.py:157](../../src/boldtailor/prepared_fit.py#L157)

<a id="definition-28"></a>

## `_prepare_contrasts`

[Source, line 279](../../src/boldtailor/model.py#L279)

```python
_prepare_contrasts(contrasts: Mapping[str, ContrastValue]) -> dict[str, ContrastValue]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:236](../../src/boldtailor/model.py#L236)
- [src/boldtailor/prepared_fit.py:277](../../src/boldtailor/prepared_fit.py#L277)

<a id="definition-29"></a>

## `_prepare_contrast`

[Source, line 292](../../src/boldtailor/model.py#L292)

```python
_prepare_contrast(value: ContrastValue, name: str) -> ContrastValue
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:288](../../src/boldtailor/model.py#L288)

<a id="definition-30"></a>

## `_prepare_weights`

[Source, line 305](../../src/boldtailor/model.py#L305)

```python
_prepare_weights(values: Mapping[str, float], name: str) -> ContrastWeights
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:302](../../src/boldtailor/model.py#L302)

<a id="definition-31"></a>

## `_prepare_confound_names`

[Source, line 326](../../src/boldtailor/model.py#L326)

```python
_prepare_confound_names(values: Sequence[str]) -> tuple[str, ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:237](../../src/boldtailor/model.py#L237)

<a id="definition-32"></a>

## `_validate_noise_model`

[Source, line 335](../../src/boldtailor/model.py#L335)

```python
_validate_noise_model(value: str) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:238](../../src/boldtailor/model.py#L238)
- [src/boldtailor/prepared_fit.py:278](../../src/boldtailor/prepared_fit.py#L278)

<a id="definition-33"></a>

## `_validate_task_model`

[Source, line 340](../../src/boldtailor/model.py#L340)

```python
_validate_task_model(task_model: object, hrf_model: HRFModel) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:239](../../src/boldtailor/model.py#L239)

<a id="definition-34"></a>

## `_validate_design_options`

[Source, line 349](../../src/boldtailor/model.py#L349)

```python
_validate_design_options(high_pass: float, drift_order: int, oversampling: int, min_onset: float) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:240](../../src/boldtailor/model.py#L240)

<a id="definition-35"></a>

## `ModelIdentity`

[Source, line 366](../../src/boldtailor/model.py#L366)

```python
@dataclass(frozen=True)
class ModelIdentity
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
activity: dict[str, object]
fingerprint: dict[str, object] | None
warnings: tuple[Mapping[str, object], ...]
```

Usage in package/tests:

- [src/boldtailor/model.py:397](../../src/boldtailor/model.py#L397)

<a id="definition-36"></a>

## `nuisance_model_settings`

[Source, line 372](../../src/boldtailor/model.py#L372)

```python
nuisance_model_settings(model: ModelSpec) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:224](../../src/boldtailor/_hrf_glm.py#L224)
- [src/boldtailor/_hrf_glm.py:233](../../src/boldtailor/_hrf_glm.py#L233)
- [src/boldtailor/fit.py:142](../../src/boldtailor/fit.py#L142)
- [src/boldtailor/fit.py:182](../../src/boldtailor/fit.py#L182)

<a id="definition-37"></a>

## `model_identity`

[Source, line 383](../../src/boldtailor/model.py#L383)

```python
model_identity(model: ModelSpec, *, hrf_model: object='keep') -> ModelIdentity
```

Return the activity and fingerprint settings that identify a model.

``hrf_model`` overrides how the HRF is recorded, for example
``{"kind": "selected"}`` when per-feature HRFs replace ``model.hrf_model``.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:156](../../src/boldtailor/_hrf_glm.py#L156)
- [src/boldtailor/_hrf_glm.py:196](../../src/boldtailor/_hrf_glm.py#L196)
- [src/boldtailor/fit.py:51](../../src/boldtailor/fit.py#L51)
- [src/boldtailor/fit.py:79](../../src/boldtailor/fit.py#L79)

<a id="definition-38"></a>

## `_identity_activity`

[Source, line 400](../../src/boldtailor/model.py#L400)

```python
_identity_activity(model: ModelSpec, hrf_model: object, normalization: str | None) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:395](../../src/boldtailor/model.py#L395)

<a id="definition-39"></a>

## `_hrf_normalization`

[Source, line 422](../../src/boldtailor/model.py#L422)

```python
_hrf_normalization(value: object) -> str | None
```

Peak-one for named canonical shapes; Nilearn's sum-one for its
canonical derivative bases; nothing for FIR and user-supplied kernels,
which are used exactly as given.

Usage in package/tests:

- [src/boldtailor/model.py:395](../../src/boldtailor/model.py#L395)

<a id="definition-40"></a>

## `_serialize_hrf`

[Source, line 433](../../src/boldtailor/model.py#L433)

```python
_serialize_hrf(value: object) -> tuple[object, tuple[Mapping[str, object], ...]]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:394](../../src/boldtailor/model.py#L394)

<a id="definition-41"></a>

## `_callable_identity`

[Source, line 448](../../src/boldtailor/model.py#L448)

```python
_callable_identity(value: object) -> dict[str, str] | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/model.py:438](../../src/boldtailor/model.py#L438)
