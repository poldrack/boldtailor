# `boldtailor.provenance`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/provenance.py)

Identify source data and extend analysis records and fingerprints.

## Definitions

- [`SourceRef`](#definition-1)
- [`SourceRef.__post_init__`](#definition-2)
- [`SourceRef.to_dict`](#definition-3)
- [`SourceRef.from_dict`](#definition-4)
- [`RunSources`](#definition-5)
- [`RunSources.__post_init__`](#definition-6)
- [`RunSources.to_dict`](#definition-7)
- [`RunSources.from_dict`](#definition-8)
- [`ProvenanceRecord`](#definition-9)
- [`ProvenanceRecord.__post_init__`](#definition-10)
- [`ProvenanceRecord.metadata_fingerprint`](#definition-11)
- [`ProvenanceRecord.analysis_fingerprint`](#definition-12)
- [`ProvenanceRecord.to_dict`](#definition-13)
- [`ProvenanceRecord.canonical_json`](#definition-14)
- [`ProvenanceRecord.from_dict`](#definition-15)
- [`analysis_fingerprint`](#definition-16)
- [`identity_activity`](#definition-17)
- [`extend_provenance`](#definition-18)
- [`_copy_source_ref`](#definition-19)
- [`_copy_run_sources`](#definition-20)
- [`_optional_source_ref`](#definition-21)
- [`_mapping_items`](#definition-22)
- [`_require_mapping`](#definition-23)
- [`_require_sequence`](#definition-24)
- [`_validate_role`](#definition-25)
- [`_validate_optional_text`](#definition-26)
- [`_validate_byte_size`](#definition-27)
- [`_validate_sha256`](#definition-28)
- [`_validate_modified_at`](#definition-29)
- [`_validate_uri`](#definition-30)
- [`_validate_bids_uri`](#definition-31)
- [`validate_relative_path`](#definition-32)
- [`_has_non_bids_scheme`](#definition-33)
- [`_validate_uuid`](#definition-34)
- [`_validate_schema`](#definition-35)
- [`_freeze_mapping_sequence`](#definition-36)
- [`_freeze_mapping`](#definition-37)
- [`_freeze_key`](#definition-38)
- [`_freeze_json`](#definition-39)
- [`_thaw`](#definition-40)
- [`_augment_warnings`](#definition-41)
- [`_metadata_fingerprint`](#definition-42)
- [`_sources_complete`](#definition-43)
- [`_source_complete`](#definition-44)
- [`_extra_fields`](#definition-45)

<a id="definition-1"></a>

## `SourceRef`

[Source, line 23](../../src/boldtailor/provenance.py#L23)

```python
@dataclass(frozen=True)
class SourceRef
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
role: str
uri: str | None = None
media_type: str | None = None
byte_size: int | None = None
modified_at: str | None = None
sha256: str | None = None
annotations: Mapping[str, object] = field(default_factory=dict)
```

Usage in package/tests:

- [src/boldtailor/data.py:272](../../src/boldtailor/data.py#L272)
- [src/boldtailor/data.py:273](../../src/boldtailor/data.py#L273)
- [src/boldtailor/data.py:274](../../src/boldtailor/data.py#L274)
- [src/boldtailor/workflow/files.py:182](../../src/boldtailor/workflow/files.py#L182)

<a id="definition-2"></a>

## `SourceRef.__post_init__`

[Source, line 32](../../src/boldtailor/provenance.py#L32)

```python
SourceRef.__post_init__(self) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-3"></a>

## `SourceRef.to_dict`

[Source, line 41](../../src/boldtailor/provenance.py#L41)

```python
SourceRef.to_dict(self) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-4"></a>

## `SourceRef.from_dict`

[Source, line 58](../../src/boldtailor/provenance.py#L58)

```python
@classmethod
SourceRef.from_dict(cls, data: Mapping[str, object]) -> SourceRef
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-5"></a>

## `RunSources`

[Source, line 71](../../src/boldtailor/provenance.py#L71)

```python
@dataclass(frozen=True)
class RunSources
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
signal: SourceRef
events: SourceRef
confounds: SourceRef | None = None
```

Usage in package/tests:

- [src/boldtailor/data.py:271](../../src/boldtailor/data.py#L271)
- [src/boldtailor/denoising.py:539](../../src/boldtailor/denoising.py#L539)
- [src/boldtailor/denoising.py:538](../../src/boldtailor/denoising.py#L538)
- [src/boldtailor/workflow/files.py:196](../../src/boldtailor/workflow/files.py#L196)

<a id="definition-6"></a>

## `RunSources.__post_init__`

[Source, line 76](../../src/boldtailor/provenance.py#L76)

```python
RunSources.__post_init__(self) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-7"></a>

## `RunSources.to_dict`

[Source, line 86](../../src/boldtailor/provenance.py#L86)

```python
RunSources.to_dict(self) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-8"></a>

## `RunSources.from_dict`

[Source, line 96](../../src/boldtailor/provenance.py#L96)

```python
@classmethod
RunSources.from_dict(cls, data: Mapping[str, object]) -> RunSources
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-9"></a>

## `ProvenanceRecord`

[Source, line 105](../../src/boldtailor/provenance.py#L105)

```python
@dataclass(frozen=True)
class ProvenanceRecord
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Declared fields (private names are implementation storage; `field(...)` describes dataclass construction):

```python
execution_id: str
sources: Sequence[RunSources] = ()
activities: Sequence[Mapping[str, object]] = ()
events: Sequence[Mapping[str, object]] = ()
warnings: Sequence[Mapping[str, object]] = ()
schema: str = SCHEMA_ID
_extra: Mapping[str, object] = field(default_factory=dict, repr=False)
_metadata_fingerprint: str | None = field(init=False, repr=False)
```

Usage in package/tests:

- [src/boldtailor/data.py:124](../../src/boldtailor/data.py#L124)
- [src/boldtailor/prepared.py:194](../../src/boldtailor/prepared.py#L194)
- [src/boldtailor/workflow/beta_series.py:161](../../src/boldtailor/workflow/beta_series.py#L161)
- [tests/test_fit_lifecycle.py:12](../../tests/test_fit_lifecycle.py#L12)

<a id="definition-10"></a>

## `ProvenanceRecord.__post_init__`

[Source, line 115](../../src/boldtailor/provenance.py#L115)

```python
ProvenanceRecord.__post_init__(self) -> None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-11"></a>

## `ProvenanceRecord.metadata_fingerprint`

[Source, line 147](../../src/boldtailor/provenance.py#L147)

```python
@property
ProvenanceRecord.metadata_fingerprint(self) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-12"></a>

## `ProvenanceRecord.analysis_fingerprint`

[Source, line 151](../../src/boldtailor/provenance.py#L151)

```python
@property
ProvenanceRecord.analysis_fingerprint(self) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-13"></a>

## `ProvenanceRecord.to_dict`

[Source, line 155](../../src/boldtailor/provenance.py#L155)

```python
ProvenanceRecord.to_dict(self) -> dict[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-14"></a>

## `ProvenanceRecord.canonical_json`

[Source, line 168](../../src/boldtailor/provenance.py#L168)

```python
ProvenanceRecord.canonical_json(self) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-15"></a>

## `ProvenanceRecord.from_dict`

[Source, line 177](../../src/boldtailor/provenance.py#L177)

```python
@classmethod
ProvenanceRecord.from_dict(cls, data: Mapping[str, object]) -> ProvenanceRecord
```

No source docstring. Read the linked implementation and call sites for the precise contract.

<a id="definition-16"></a>

## `analysis_fingerprint`

[Source, line 201](../../src/boldtailor/provenance.py#L201)

```python
analysis_fingerprint(metadata_fingerprint: str | None, model: Mapping[str, object]) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:98](../../src/boldtailor/_hrf_glm.py#L98)
- [src/boldtailor/_hrf_glm.py:212](../../src/boldtailor/_hrf_glm.py#L212)
- [src/boldtailor/_ridge_cv.py:233](../../src/boldtailor/_ridge_cv.py#L233)
- [src/boldtailor/_selected_hrf_fit.py:168](../../src/boldtailor/_selected_hrf_fit.py#L168)

<a id="definition-17"></a>

## `identity_activity`

[Source, line 215](../../src/boldtailor/provenance.py#L215)

```python
identity_activity(record: ProvenanceRecord) -> dict[str, object]
```

The record's last activity without environment-only keys, for embedding in identities.

Usage in package/tests:

- [src/boldtailor/_hrf_glm.py:86](../../src/boldtailor/_hrf_glm.py#L86)
- [src/boldtailor/_ridge_cv.py:105](../../src/boldtailor/_ridge_cv.py#L105)
- [src/boldtailor/_selected_hrf_fit.py:122](../../src/boldtailor/_selected_hrf_fit.py#L122)
- [src/boldtailor/denoising.py:369](../../src/boldtailor/denoising.py#L369)

<a id="definition-18"></a>

## `extend_provenance`

[Source, line 222](../../src/boldtailor/provenance.py#L222)

```python
extend_provenance(record: ProvenanceRecord, *, execution_id: str, activity: Mapping[str, object], events: Sequence[Mapping[str, object]], warnings: Sequence[Mapping[str, object]], analysis_id: str | None) -> ProvenanceRecord
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/_fit_lifecycle.py:43](../../src/boldtailor/_fit_lifecycle.py#L43)
- [src/boldtailor/_ridge_cv.py:227](../../src/boldtailor/_ridge_cv.py#L227)
- [src/boldtailor/workflow/beta_series.py:611](../../src/boldtailor/workflow/beta_series.py#L611)
- [src/boldtailor/workflow/beta_series.py:662](../../src/boldtailor/workflow/beta_series.py#L662)

<a id="definition-19"></a>

## `_copy_source_ref`

[Source, line 241](../../src/boldtailor/provenance.py#L241)

```python
_copy_source_ref(value: SourceRef, name: str) -> SourceRef
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:77](../../src/boldtailor/provenance.py#L77)
- [src/boldtailor/provenance.py:78](../../src/boldtailor/provenance.py#L78)
- [src/boldtailor/provenance.py:83](../../src/boldtailor/provenance.py#L83)

<a id="definition-20"></a>

## `_copy_run_sources`

[Source, line 250](../../src/boldtailor/provenance.py#L250)

```python
_copy_run_sources(value: RunSources) -> RunSources
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:121](../../src/boldtailor/provenance.py#L121)

<a id="definition-21"></a>

## `_optional_source_ref`

[Source, line 256](../../src/boldtailor/provenance.py#L256)

```python
_optional_source_ref(value: object) -> SourceRef | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:100](../../src/boldtailor/provenance.py#L100)

<a id="definition-22"></a>

## `_mapping_items`

[Source, line 262](../../src/boldtailor/provenance.py#L262)

```python
_mapping_items(value: object, name: str) -> tuple[Mapping[str, object], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:194](../../src/boldtailor/provenance.py#L194)
- [src/boldtailor/provenance.py:195](../../src/boldtailor/provenance.py#L195)
- [src/boldtailor/provenance.py:196](../../src/boldtailor/provenance.py#L196)

<a id="definition-23"></a>

## `_require_mapping`

[Source, line 269](../../src/boldtailor/provenance.py#L269)

```python
_require_mapping(value: object, name: str) -> Mapping[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:259](../../src/boldtailor/provenance.py#L259)
- [src/boldtailor/provenance.py:264](../../src/boldtailor/provenance.py#L264)
- [src/boldtailor/provenance.py:98](../../src/boldtailor/provenance.py#L98)
- [src/boldtailor/provenance.py:99](../../src/boldtailor/provenance.py#L99)

<a id="definition-24"></a>

## `_require_sequence`

[Source, line 275](../../src/boldtailor/provenance.py#L275)

```python
_require_sequence(value: object, name: str) -> Sequence[object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:265](../../src/boldtailor/provenance.py#L265)
- [src/boldtailor/provenance.py:192](../../src/boldtailor/provenance.py#L192)

<a id="definition-25"></a>

## `_validate_role`

[Source, line 281](../../src/boldtailor/provenance.py#L281)

```python
_validate_role(role: object) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:33](../../src/boldtailor/provenance.py#L33)

<a id="definition-26"></a>

## `_validate_optional_text`

[Source, line 287](../../src/boldtailor/provenance.py#L287)

```python
_validate_optional_text(value: object) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:35](../../src/boldtailor/provenance.py#L35)

<a id="definition-27"></a>

## `_validate_byte_size`

[Source, line 295](../../src/boldtailor/provenance.py#L295)

```python
_validate_byte_size(value: object) -> int | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:36](../../src/boldtailor/provenance.py#L36)

<a id="definition-28"></a>

## `_validate_sha256`

[Source, line 305](../../src/boldtailor/provenance.py#L305)

```python
_validate_sha256(value: object) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:38](../../src/boldtailor/provenance.py#L38)

<a id="definition-29"></a>

## `_validate_modified_at`

[Source, line 313](../../src/boldtailor/provenance.py#L313)

```python
_validate_modified_at(value: object) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:37](../../src/boldtailor/provenance.py#L37)

<a id="definition-30"></a>

## `_validate_uri`

[Source, line 328](../../src/boldtailor/provenance.py#L328)

```python
_validate_uri(value: object) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:34](../../src/boldtailor/provenance.py#L34)

<a id="definition-31"></a>

## `_validate_bids_uri`

[Source, line 340](../../src/boldtailor/provenance.py#L340)

```python
_validate_bids_uri(value: str) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:334](../../src/boldtailor/provenance.py#L334)

<a id="definition-32"></a>

## `validate_relative_path`

[Source, line 348](../../src/boldtailor/provenance.py#L348)

```python
validate_relative_path(text: object, *, name: str='path') -> str
```

Return ``text`` if it is a dataset-relative POSIX path of safe components.

Usage in package/tests:

- [src/boldtailor/bids_provenance.py:384](../../src/boldtailor/bids_provenance.py#L384)
- [src/boldtailor/bids_provenance.py:428](../../src/boldtailor/bids_provenance.py#L428)
- [src/boldtailor/bids_provenance.py:258](../../src/boldtailor/bids_provenance.py#L258)
- [src/boldtailor/bids_provenance.py:260](../../src/boldtailor/bids_provenance.py#L260)

<a id="definition-33"></a>

## `_has_non_bids_scheme`

[Source, line 364](../../src/boldtailor/provenance.py#L364)

```python
_has_non_bids_scheme(value: str) -> bool
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:335](../../src/boldtailor/provenance.py#L335)

<a id="definition-34"></a>

## `_validate_uuid`

[Source, line 374](../../src/boldtailor/provenance.py#L374)

```python
_validate_uuid(value: object) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:117](../../src/boldtailor/provenance.py#L117)

<a id="definition-35"></a>

## `_validate_schema`

[Source, line 383](../../src/boldtailor/provenance.py#L383)

```python
_validate_schema(value: object) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:116](../../src/boldtailor/provenance.py#L116)

<a id="definition-36"></a>

## `_freeze_mapping_sequence`

[Source, line 389](../../src/boldtailor/provenance.py#L389)

```python
_freeze_mapping_sequence(values: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:126](../../src/boldtailor/provenance.py#L126)
- [src/boldtailor/provenance.py:131](../../src/boldtailor/provenance.py#L131)
- [src/boldtailor/provenance.py:137](../../src/boldtailor/provenance.py#L137)

<a id="definition-37"></a>

## `_freeze_mapping`

[Source, line 395](../../src/boldtailor/provenance.py#L395)

```python
_freeze_mapping(value: object) -> Mapping[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:418](../../src/boldtailor/prepared.py#L418)
- [src/boldtailor/prepared.py:426](../../src/boldtailor/prepared.py#L426)
- [src/boldtailor/prepared_fit.py:279](../../src/boldtailor/prepared_fit.py#L279)
- [src/boldtailor/provenance.py:420](../../src/boldtailor/provenance.py#L420)

<a id="definition-38"></a>

## `_freeze_key`

[Source, line 404](../../src/boldtailor/provenance.py#L404)

```python
_freeze_key(value: object) -> str
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:400](../../src/boldtailor/provenance.py#L400)

<a id="definition-39"></a>

## `_freeze_json`

[Source, line 410](../../src/boldtailor/provenance.py#L410)

```python
_freeze_json(value: object) -> object
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:400](../../src/boldtailor/provenance.py#L400)
- [src/boldtailor/provenance.py:422](../../src/boldtailor/provenance.py#L422)

<a id="definition-40"></a>

## `_thaw`

[Source, line 426](../../src/boldtailor/provenance.py#L426)

```python
_thaw(value: object) -> object
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/prepared.py:255](../../src/boldtailor/prepared.py#L255)
- [src/boldtailor/prepared.py:251](../../src/boldtailor/prepared.py#L251)
- [src/boldtailor/prepared.py:119](../../src/boldtailor/prepared.py#L119)
- [src/boldtailor/provenance.py:209](../../src/boldtailor/provenance.py#L209)

<a id="definition-41"></a>

## `_augment_warnings`

[Source, line 434](../../src/boldtailor/provenance.py#L434)

```python
_augment_warnings(warnings: tuple[Mapping[str, object], ...], metadata_fingerprint: str | None) -> tuple[Mapping[str, object], ...]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:137](../../src/boldtailor/provenance.py#L137)

<a id="definition-42"></a>

## `_metadata_fingerprint`

[Source, line 445](../../src/boldtailor/provenance.py#L445)

```python
_metadata_fingerprint(sources: Sequence[RunSources]) -> str | None
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/data.py:101](../../src/boldtailor/data.py#L101)
- [src/boldtailor/prepared.py:160](../../src/boldtailor/prepared.py#L160)
- [src/boldtailor/provenance.py:133](../../src/boldtailor/provenance.py#L133)
- [tests/test_provenance.py:375](../../tests/test_provenance.py#L375)

<a id="definition-43"></a>

## `_sources_complete`

[Source, line 455](../../src/boldtailor/provenance.py#L455)

```python
_sources_complete(sources: Sequence[RunSources]) -> bool
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:446](../../src/boldtailor/provenance.py#L446)

<a id="definition-44"></a>

## `_source_complete`

[Source, line 467](../../src/boldtailor/provenance.py#L467)

```python
_source_complete(source: SourceRef) -> bool
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:462](../../src/boldtailor/provenance.py#L462)

<a id="definition-45"></a>

## `_extra_fields`

[Source, line 475](../../src/boldtailor/provenance.py#L475)

```python
_extra_fields(data: Mapping[str, object], known: set[str]) -> Mapping[str, object]
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/provenance.py:197](../../src/boldtailor/provenance.py#L197)
